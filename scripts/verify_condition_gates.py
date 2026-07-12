from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from conditions import choose_dtype, load_yaml, seed_everything
from dataset import PairedWarmupDataset
from train_paired import WarmupFlowModel, build_optimizer, load_components


def main() -> None:
    parser = argparse.ArgumentParser(description='Real one-step gate gradient/update check.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed_everything(int(cfg['experiment']['seed']))
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = choose_dtype(cfg['model']['precision'])
    cfg['_runtime'] = {
        'world_size': 3,
        'micro_batch': 1,
        'grad_accum': 6,
        'global_batch': 18,
        'effective_lr': float(cfg['training']['baseline_lr']) * 18 / int(cfg['training']['baseline_global_batch']),
        'all_gpus_train': False,
    }

    dataset = PairedWarmupDataset(cfg, 'train', require_coverage=True)
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    transformer, controlnet, _vae, adapter, pulid, _notes = load_components(cfg, device, dtype)
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg).train()
    optimizer = build_optimizer(model, cfg)

    gate_names = ('appearance_gate', 'garment_gate', 'pose_gate')
    before = {name: float(getattr(adapter, name).detach().cpu()) for name in gate_names}
    dtypes = {name: str(getattr(adapter, name).dtype) for name in gate_names}

    optimizer.zero_grad(set_to_none=True)
    loss, _metrics = model(batch)
    loss.backward()
    grads = {
        name: float(getattr(adapter, name).grad.detach().float().cpu())
        for name in gate_names
    }
    torch.nn.utils.clip_grad_norm_(
        [param for param in model.parameters() if param.requires_grad],
        float(cfg['training']['max_grad_norm']),
    )
    optimizer.step()
    after = {name: float(getattr(adapter, name).detach().cpu()) for name in gate_names}
    delta = {name: after[name] - before[name] for name in gate_names}

    result = {
        'loss': float(loss.detach().cpu()),
        'before': before,
        'after': after,
        'delta': delta,
        'grad': grads,
        'dtype': dtypes,
        'gate_group_lr': next(
            group['lr'] for group in optimizer.param_groups
            if group.get('group_name') == 'condition_gates_fp32'
        ),
        'peak_gib': torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    print(json.dumps(result, indent=2))
    for name in gate_names:
        if dtypes[name] != 'torch.float32':
            raise RuntimeError(f'{name} is not fp32: {dtypes[name]}')
        if not torch.isfinite(torch.tensor(grads[name])) or abs(grads[name]) <= 1e-8:
            raise RuntimeError(f'{name} gradient is invalid: {grads[name]}')
        if abs(delta[name]) <= 1e-4:
            raise RuntimeError(f'{name} did not update enough: delta={delta[name]}')


if __name__ == '__main__':
    main()
