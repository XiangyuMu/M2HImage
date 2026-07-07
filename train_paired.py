from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from conditions import (
    FluxIdentityAdapter, assert_real_controlnet, atomic_torch_save, choose_dtype, load_yaml, make_image_ids,
    make_text_ids, save_yaml, seed_everything, short_hash_path,
)
from dataset import PairedWarmupDataset, ResumeDistributedSampler


class WarmupFlowModel(torch.nn.Module):
    def __init__(self, transformer, controlnet, adapter: FluxIdentityAdapter, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.transformer = transformer
        self.controlnet = controlnet
        self.adapter = adapter
        self.cfg = cfg
        self.resolution = int(cfg['data']['resolution'])
        self.control_mode = int(cfg['model']['control_mode'])
        self.controlnet_scale = float(cfg['model']['controlnet_scale'])

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        dtype = next(self.transformer.parameters()).dtype
        device = next(self.transformer.parameters()).device
        z0 = batch['target_latents'].to(device=device, dtype=dtype)
        z1 = torch.randn_like(z0)
        tau = torch.rand(z0.shape[0], device=device, dtype=dtype)
        z_tau = (1.0 - tau.view(-1, 1, 1)) * z0 + tau.view(-1, 1, 1) * z1
        target_v = z1 - z0
        model_timestep = tau
        prompt = batch['prompt_embeds'].to(device=device, dtype=dtype)
        pooled = batch['pooled_prompt_embeds'].to(device=device, dtype=dtype)
        if prompt.ndim == 2:
            prompt = prompt.unsqueeze(0).expand(z0.shape[0], -1, -1)
        if pooled.ndim == 1:
            pooled = pooled.unsqueeze(0).expand(z0.shape[0], -1)
        adapter_tokens = self.adapter(
            batch['identity'].to(device=device, dtype=dtype),
            batch['appearance'].to(device=device, dtype=dtype),
            batch['garment'].to(device=device, dtype=dtype),
            batch['head_pose'].to(device=device, dtype=dtype),
        )
        encoder_hidden_states = torch.cat([prompt, adapter_tokens], dim=1)
        txt_ids = make_text_ids(encoder_hidden_states.shape[1], device, dtype)
        img_ids = make_image_ids(self.resolution, device, dtype)
        with torch.no_grad():
            cn = self.controlnet(
                hidden_states=z_tau,
                controlnet_cond=batch['pose_latents'].to(device=device, dtype=dtype),
                controlnet_mode=torch.full((z0.shape[0], 1), self.control_mode, device=device, dtype=torch.long),
                conditioning_scale=self.controlnet_scale,
                encoder_hidden_states=prompt,
                pooled_projections=pooled,
                timestep=model_timestep,
                img_ids=img_ids,
                txt_ids=make_text_ids(prompt.shape[1], device, dtype),
                guidance=torch.full((z0.shape[0],), 3.5, device=device, dtype=dtype),
                return_dict=True,
            )
        pred = self.transformer(
            hidden_states=z_tau,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled,
            timestep=model_timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=torch.full((z0.shape[0],), 3.5, device=device, dtype=dtype),
            controlnet_block_samples=cn.controlnet_block_samples,
            controlnet_single_block_samples=cn.controlnet_single_block_samples,
            return_dict=True,
        ).sample
        loss = F.mse_loss(pred.float(), target_v.float())
        head_null = batch.get('head_pose_is_null')
        head_null_ratio = head_null.float().mean().detach() if head_null is not None else torch.zeros((), device=device, dtype=torch.float32)
        return loss, {'loss_pair': loss.detach(), 'head_pose_null_ratio': head_null_ratio}


def setup_dist() -> tuple[int, int, int]:
    if 'RANK' not in os.environ:
        return 0, 1, int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl')
    return int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])


def cleanup_dist() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def attach_lora(transformer, rank: int) -> str:
    from peft import LoraConfig
    for param in transformer.parameters():
        param.requires_grad_(False)
    target_modules = ['to_q', 'to_k', 'to_v', 'to_out.0', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out']
    transformer.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights='gaussian', target_modules=target_modules))
    for name, param in transformer.named_parameters():
        param.requires_grad_('lora' in name.lower())
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    return f'LoRA rank={rank}, trainable={trainable:,}'


def load_components(cfg: dict[str, Any], device: torch.device, dtype: torch.dtype):
    from diffusers import AutoencoderKL, FluxControlNetModel, FluxTransformer2DModel
    transformer = FluxTransformer2DModel.from_pretrained(cfg['model']['base'], subfolder='transformer', torch_dtype=dtype, local_files_only=True)
    if cfg['model'].get('gradient_checkpointing', {}).get('enabled', True):
        transformer.enable_gradient_checkpointing()
    transformer.to(device=device, dtype=dtype).train()
    lora_note = attach_lora(transformer, int(cfg['model']['lora_rank']))
    control_info = assert_real_controlnet(cfg['model']['controlnet'])
    controlnet = FluxControlNetModel.from_pretrained(cfg['model']['controlnet'], torch_dtype=dtype, local_files_only=True)
    controlnet.requires_grad_(False).to(device=device, dtype=dtype).eval()
    vae = None
    if cfg['model'].get('load_vae_in_train', False):
        vae = AutoencoderKL.from_pretrained(cfg['model']['base'], subfolder='vae', torch_dtype=dtype, local_files_only=True)
        vae.requires_grad_(False).to(device=device, dtype=dtype).eval()
    adapter = FluxIdentityAdapter(cfg['model']['identity_adapter']).to(device=device, dtype=dtype)
    return transformer, controlnet, vae, adapter, {'controlnet': control_info, 'lora': lora_note, 'adapter': adapter.launch_note()}


def build_optimizer(module: torch.nn.Module, cfg: dict[str, Any]):
    lr = float(cfg['_runtime']['effective_lr'])
    adapter_mult = float(cfg['model'].get('identity_adapter', {}).get('fallback_projection_lr_mult', 5.0))
    lora_params = []
    adapter_params = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if '.adapter.' in name or name.startswith('adapter.') or name.startswith('module.adapter.'):
            adapter_params.append(param)
        else:
            lora_params.append(param)
    groups = []
    if lora_params:
        groups.append({'params': lora_params, 'lr': lr})
    if adapter_params:
        groups.append({'params': adapter_params, 'lr': lr * adapter_mult})
    if cfg['training'].get('optimizer') == 'paged_adamw8bit':
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(groups)
    return torch.optim.AdamW(groups)


def recompute_batch_runtime(cfg: dict[str, Any], world_size: int, micro: int) -> None:
    accum = math.ceil(int(cfg['training']['baseline_global_batch']) / (world_size * micro))
    global_batch = world_size * micro * accum
    cfg['_runtime']['micro_batch'] = micro
    cfg['_runtime']['grad_accum'] = accum
    cfg['_runtime']['global_batch'] = global_batch
    cfg['_runtime']['effective_lr'] = float(cfg['training']['baseline_lr']) * global_batch / int(cfg['training']['baseline_global_batch'])


def make_loader(dataset, sampler, cfg: dict[str, Any]) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg['_runtime']['micro_batch']),
        sampler=sampler,
        num_workers=int(cfg['training']['num_workers_per_rank']),
        pin_memory=bool(cfg['training']['pin_memory']),
        persistent_workers=bool(cfg['training']['persistent_workers']),
        prefetch_factor=int(cfg['training']['prefetch_factor']),
        drop_last=True,
    )


def sync_probe_ok(ok: bool, device: torch.device) -> bool:
    if not dist.is_initialized():
        return ok
    flag = torch.tensor(1 if ok else 0, device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def maybe_probe_micro_batch(
    model: WarmupFlowModel,
    dataset: PairedWarmupDataset,
    cfg: dict[str, Any],
    world_size: int,
    rank: int,
    device: torch.device,
) -> None:
    if not cfg['training'].get('auto_micro_batch_probe', False):
        return
    preferred = int(cfg['training'].get('preferred_micro_batch', cfg['_runtime']['micro_batch']))
    current = int(cfg['_runtime']['micro_batch'])
    if preferred <= current:
        return
    probe_loader = DataLoader(dataset, batch_size=preferred, shuffle=False, num_workers=0, drop_last=True)
    try:
        batch = next(iter(probe_loader))
    except StopIteration:
        if rank == 0:
            print('[rank0] micro-batch probe skipped: not enough cached samples on this rank', flush=True)
        return
    old = dict(cfg['_runtime'])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    ok = True
    err = ''
    peak = 0.0
    try:
        loss, _ = model(batch)
        loss.backward()
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        ok = peak <= 44.0
        model.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError as exc:
        ok = False
        err = str(exc).split('\n')[0]
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        if 'out of memory' not in str(exc).lower():
            raise
        ok = False
        err = str(exc).split('\n')[0]
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    ok = sync_probe_ok(ok, device)
    if ok:
        recompute_batch_runtime(cfg, world_size, preferred)
        if rank == 0:
            print(f'[rank0] micro-batch probe accepted: micro={preferred}, peak_gib={peak:.2f}, runtime={cfg["_runtime"]}', flush=True)
    else:
        cfg['_runtime'] = old
        if rank == 0:
            reason = f'peak_gib={peak:.2f} > 44.0' if peak else err
            print(f'[rank0] micro-batch probe fallback: keep micro={current}; reason={reason}', flush=True)


def save_checkpoint(path: Path, model: WarmupFlowModel, optimizer, step: int, sampler: ResumeDistributedSampler, cfg: dict[str, Any]) -> None:
    from peft import get_peft_model_state_dict
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        'step': step,
        'adapter': model.adapter.state_dict(),
        'transformer_lora': get_peft_model_state_dict(model.transformer),
        'optimizer': optimizer.state_dict(),
        'sampler': sampler.state_dict(),
        'config': cfg,
    }
    atomic_torch_save(payload, path / 'trainable.pt')
    (path / 'READY').write_text(str(step), encoding='utf-8')


def load_checkpoint(path: Path, model: WarmupFlowModel, optimizer=None, sampler=None) -> int:
    from peft import set_peft_model_state_dict
    payload = torch.load(path / 'trainable.pt', map_location='cpu')
    adapter_state = payload['adapter']
    current = model.adapter.state_dict()
    compatible = {k: v for k, v in adapter_state.items() if k in current and tuple(current[k].shape) == tuple(v.shape)}
    dropped = sorted(set(adapter_state) - set(compatible))
    missing, unexpected = model.adapter.load_state_dict(compatible, strict=False)
    if dropped or missing or unexpected:
        print(f'[warn] adapter checkpoint loaded partially from {path}: dropped={len(dropped)}, missing={len(missing)}, unexpected={len(unexpected)}', flush=True)
    set_peft_model_state_dict(model.transformer, payload['transformer_lora'])
    if optimizer is not None and 'optimizer' in payload:
        optimizer.load_state_dict(payload['optimizer'])
    if sampler is not None and 'sampler' in payload:
        sampler.load_state_dict(payload['sampler'])
    return int(payload.get('step', 0))


def configure_runtime(cfg: dict[str, Any], world_size: int, all_gpus_train: bool) -> None:
    micro = int(cfg['training']['micro_batch'])
    if cfg['training'].get('grad_accum') == 'auto':
        accum = math.ceil(int(cfg['training']['baseline_global_batch']) / (world_size * micro))
    else:
        accum = int(cfg['training']['grad_accum'])
    global_batch = world_size * micro * accum
    effective_lr = float(cfg['training']['baseline_lr']) * global_batch / int(cfg['training']['baseline_global_batch'])
    cfg['_runtime'] = {
        'world_size': world_size,
        'micro_batch': micro,
        'grad_accum': accum,
        'global_batch': global_batch,
        'effective_lr': effective_lr,
        'all_gpus_train': bool(all_gpus_train),
        'arch_note': 'Default is 3-card DDP + GPU3 watcher. Same A6000 cards and Phase0 30.4GiB single-card peak make DDP simpler/faster than FSDP/DeepSpeed; frozen modules have requires_grad=False so DDP does not sync them.',
    }


def train(args: argparse.Namespace) -> None:
    rank, world_size, local_rank = setup_dist()
    cfg = load_yaml(args.config)
    all_gpus_train = bool(args.all_gpus_train)
    if not all_gpus_train and world_size != 3 and not args.dev_single_gpu:
        raise RuntimeError(f'default Phase1 launch expects 3 training ranks, leaving GPU3 for watcher; got world_size={world_size}. Use --all-gpus-train for 4-rank training, or --dev-single-gpu for smoke only.')
    if all_gpus_train and world_size != 4:
        raise RuntimeError(f'--all-gpus-train expects world_size=4, got {world_size}')
    if args.dev_single_gpu and world_size != 1:
        raise RuntimeError('--dev-single-gpu is only for one-process smoke runs')
    configure_runtime(cfg, world_size, all_gpus_train)
    if args.override_total_steps is not None:
        cfg['training']['total_steps'] = int(args.override_total_steps)
    if args.smoke_steps > 0:
        cfg['training']['total_steps'] = int(args.smoke_steps)
        cfg['_runtime']['grad_accum'] = 1
        cfg['_runtime']['global_batch'] = world_size * int(cfg['_runtime']['micro_batch'])
        cfg['_runtime']['effective_lr'] = float(cfg['training']['baseline_lr']) * cfg['_runtime']['global_batch'] / int(cfg['training']['baseline_global_batch'])
    seed_everything(int(cfg['experiment']['seed']) + rank)
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dtype = choose_dtype(cfg['model']['precision'])

    dataset = PairedWarmupDataset(cfg, 'train', require_coverage=bool(cfg['cache'].get('require_coverage', True)) and not args.allow_partial_cache)
    if args.allow_partial_cache:
        dataset.ids = [sid for sid in dataset.ids if dataset.sample_path(sid).exists()]
        if not dataset.ids:
            raise RuntimeError('allow_partial_cache requested but no cached train samples were found')
        if rank == 0:
            print(f'[rank0] allow_partial_cache: using {len(dataset.ids)} cached train samples for smoke only', flush=True)
    sampler = ResumeDistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=int(cfg['experiment']['seed']))
    transformer, controlnet, vae, adapter, load_notes = load_components(cfg, device, dtype)
    model = WarmupFlowModel(transformer, controlnet, adapter, cfg)
    if not args.smoke_steps:
        maybe_probe_micro_batch(model, dataset, cfg, world_size, rank, device)
    loader = make_loader(dataset, sampler, cfg)
    if cfg['model'].get('compile', False):
        try:
            model.transformer = torch.compile(model.transformer)
        except Exception as exc:  # noqa: BLE001
            if rank == 0:
                print(f'[warn] torch.compile disabled after failure: {exc}', flush=True)
    # DDP is intentionally used instead of FSDP/DeepSpeed: each A6000 fits the full Phase1 model,
    # cards are homogeneous, and frozen ControlNet/VAE/encoders have requires_grad=False so they are not placed in gradient buckets.
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False) if world_size > 1 else model
    optimizer = build_optimizer(ddp, cfg)

    output = Path(cfg['experiment']['output_root']) / cfg['experiment']['id']
    ckpt_dir = output / 'checkpoints'
    log_dir = output / 'logs'
    start_step = 0
    if cfg['training'].get('resume'):
        start_step = load_checkpoint(Path(cfg['training']['resume']), ddp.module if hasattr(ddp, 'module') else ddp, optimizer, sampler)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        save_yaml(output / 'resolved_config.yaml', cfg)
        (output / 'launch.json').write_text(json.dumps({'load_notes': load_notes, 'base_hash': short_hash_path(cfg['model']['base']), 'rank0_device': torch.cuda.get_device_name(local_rank)}, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'[rank0] runtime={cfg["_runtime"]}', flush=True)
        print(f'[rank0] controlnet={load_notes["controlnet"]}', flush=True)
    if dist.is_initialized():
        dist.barrier()

    global_step = start_step
    accum = int(cfg['_runtime']['grad_accum'])
    benchmark_start = None
    benchmark_done = False
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    while global_step < int(cfg['training']['total_steps']):
        sampler.set_epoch(global_step // max(1, len(loader)))
        for batch_idx, batch in enumerate(loader):
            if benchmark_start is None:
                torch.cuda.reset_peak_memory_stats(device)
                benchmark_start = time.perf_counter()
            loss, metrics = ddp(batch)
            (loss / accum).backward()
            micro_step += 1
            if micro_step % accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_([p for p in ddp.parameters() if p.requires_grad], float(cfg['training']['max_grad_norm']))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if rank == 0 and (global_step % int(cfg['training']['log_every']) == 0 or global_step <= 3):
                log_dir.mkdir(parents=True, exist_ok=True)
                row = {'step': global_step, 'loss_pair': float(metrics['loss_pair'].float().cpu()), 'head_pose_null_ratio': float(metrics.get('head_pose_null_ratio', torch.tensor(0.0)).float().cpu()), 'lr': cfg['_runtime']['effective_lr'], 'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3}
                with (log_dir / 'train.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + '\n')
                print(f'[rank0] {row}', flush=True)
            if not benchmark_done and global_step >= int(cfg['training']['benchmark_steps']):
                elapsed = time.perf_counter() - benchmark_start
                imgs = int(cfg['_runtime']['global_batch']) * int(cfg['training']['benchmark_steps'])
                bench = {'steps': int(cfg['training']['benchmark_steps']), 'seconds': elapsed, 'img_per_sec': imgs / elapsed, 'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3, 'estimated_6000_step_hours': elapsed / int(cfg['training']['benchmark_steps']) * 6000 / 3600}
                if rank == 0:
                    (log_dir / 'benchmark.json').write_text(json.dumps(bench, indent=2), encoding='utf-8')
                    print(f'[rank0] benchmark={bench}', flush=True)
                benchmark_done = True
            if rank == 0 and global_step % int(cfg['training']['checkpoint_every']) == 0:
                save_checkpoint(ckpt_dir / f'step-{global_step:06d}', ddp.module if hasattr(ddp, 'module') else ddp, optimizer, global_step, sampler, cfg)
            if global_step >= int(cfg['training']['total_steps']):
                break
    if rank == 0 and cfg['training'].get('save_final', True):
        save_checkpoint(ckpt_dir / 'final', ddp.module if hasattr(ddp, 'module') else ddp, optimizer, global_step, sampler, cfg)
    cleanup_dist()


def main() -> None:
    parser = argparse.ArgumentParser(description='Phase 1 paired-flow warmup training for MA-RA-CDT B2 baseline.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--all-gpus-train', action='store_true')
    parser.add_argument('--dev-single-gpu', action='store_true', help='Smoke-test only: bypass 3-rank default and run one process without DDP.')
    parser.add_argument('--allow-partial-cache', action='store_true', help='Smoke-test only: restrict train IDs to cached samples instead of requiring 100% coverage.')
    parser.add_argument('--smoke-steps', type=int, default=0, help='Smoke-test only: override total_steps and grad_accum for a short run.')
    parser.add_argument('--override-total-steps', type=int, default=None, help='Run a shorter real training job without changing grad_accum, used by speed_bench.py.')
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
