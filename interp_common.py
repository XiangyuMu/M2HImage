from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from conditions import choose_dtype, get_resolution, seed_everything


TRAINABLE_GROUPS = ('transformer_lora', 'adapter')


def resolve_root_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def alpha_label(alpha: float) -> str:
    value = int(round(float(alpha) * 100.0))
    if not 0 <= value <= 100:
        raise ValueError(f'alpha must be in [0,1], got {alpha}')
    return f'alpha{value:03d}'


def t_label(value: float) -> str:
    percent = int(round(float(value) * 100.0))
    if not 0 <= percent <= 100:
        raise ValueError(f't must be in [0,1], got {value}')
    return f't{percent:03d}'


def identity_image_name(mid: str, jid: str, kid: str, t: float, seed: int) -> str:
    return f'{mid}__j{jid}__k{kid}__{t_label(t)}__seed{int(seed)}.png'


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_csv(path: str | Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


def sha256_file(path: str | Path, length: int = 16) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def checkpoint_run_dir(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    if path.name == 'trainable.pt':
        path = path.parent
    if path.name == 'final' and path.parent.name == 'checkpoints':
        return path.parents[1]
    raise RuntimeError(f'expected <run>/checkpoints/final checkpoint, got {checkpoint}')


def verify_endpoint_fairness(
    b2_checkpoint: str | Path,
    a4_checkpoint: str | Path,
    expected_resume_hash: str,
) -> dict[str, Any]:
    launches = {
        'b2cont': read_json(checkpoint_run_dir(b2_checkpoint) / 'launch.json'),
        'a4': read_json(checkpoint_run_dir(a4_checkpoint) / 'launch.json'),
    }
    fields = (
        'resume_trainable_hash', 'resume_sampler_state', 'train_ids_hash',
        'train_sample_count', 'excluded_train_ids', 'run_origin_step', 'target_step',
    )
    mismatches = {
        field: [launches['b2cont'].get(field), launches['a4'].get(field)]
        for field in fields
        if launches['b2cont'].get(field) != launches['a4'].get(field)
    }
    actual = launches['b2cont'].get('resume_trainable_hash')
    if mismatches or actual != expected_resume_hash:
        raise RuntimeError(
            'weight interpolation endpoints fail fairness assertion: '
            f'mismatches={mismatches}, expected_resume_hash={expected_resume_hash}, actual={actual}'
        )
    return {'status': 'pass', 'fields': {field: launches['b2cont'].get(field) for field in fields}}


def _trainable_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    return path if path.name == 'trainable.pt' else path / 'trainable.pt'


def load_trainable_state(checkpoint: str | Path) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    path = _trainable_path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f'trainable checkpoint missing: {path}')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    state: dict[str, dict[str, torch.Tensor]] = {}
    for group in TRAINABLE_GROUPS:
        if group not in payload or not isinstance(payload[group], dict):
            raise RuntimeError(f'{path} has no tensor mapping {group!r}')
        non_tensors = [key for key, value in payload[group].items() if not isinstance(value, torch.Tensor)]
        if non_tensors:
            raise RuntimeError(f'{path} {group} contains non-tensor entries: {non_tensors[:20]}')
        state[group] = {key: value.detach().cpu() for key, value in payload[group].items()}
    metadata = {
        'path': str(path),
        'step': int(payload.get('step', -1)),
        'continuation_origin_step': payload.get('continuation_origin_step'),
    }
    return state, metadata


def state_schema(state: dict[str, dict[str, torch.Tensor]]) -> dict[str, dict[str, tuple[tuple[int, ...], str]]]:
    return {
        group: {key: (tuple(value.shape), str(value.dtype)) for key, value in values.items()}
        for group, values in state.items()
    }


def schema_hash(state: dict[str, dict[str, torch.Tensor]]) -> str:
    serial = json.dumps(state_schema(state), sort_keys=True, default=list).encode('utf-8')
    return hashlib.sha256(serial).hexdigest()[:16]


def assert_trainable_compatible(
    b2_state: dict[str, dict[str, torch.Tensor]],
    a4_state: dict[str, dict[str, torch.Tensor]],
) -> dict[str, Any]:
    errors: list[str] = []
    summary: dict[str, Any] = {}
    for group in TRAINABLE_GROUPS:
        left = b2_state.get(group, {})
        right = a4_state.get(group, {})
        left_keys, right_keys = set(left), set(right)
        only_left = sorted(left_keys - right_keys)
        only_right = sorted(right_keys - left_keys)
        mismatched = [
            {
                'key': key,
                'b2_shape': list(left[key].shape),
                'a4_shape': list(right[key].shape),
                'b2_dtype': str(left[key].dtype),
                'a4_dtype': str(right[key].dtype),
            }
            for key in sorted(left_keys & right_keys)
            if left[key].shape != right[key].shape or left[key].dtype != right[key].dtype
        ]
        non_floating = [key for key, value in left.items() if not torch.is_floating_point(value)]
        if only_left or only_right or mismatched or non_floating:
            errors.append(
                f'{group}: only_b2={only_left[:20]}, only_a4={only_right[:20]}, '
                f'shape_dtype={mismatched[:20]}, non_floating={non_floating[:20]}'
            )
        summary[group] = {
            'keys': len(left_keys),
            'elements': int(sum(value.numel() for value in left.values())),
            'only_b2': only_left,
            'only_a4': only_right,
            'shape_dtype_mismatches': mismatched,
        }
    if errors:
        raise RuntimeError('trainable key-set assertion failed:\n' + '\n'.join(errors))
    summary['schema_hash'] = schema_hash(b2_state)
    return summary


def interpolate_trainable_state(
    b2_state: dict[str, dict[str, torch.Tensor]],
    a4_state: dict[str, dict[str, torch.Tensor]],
    alpha: float,
) -> dict[str, dict[str, torch.Tensor]]:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f'alpha must be in [0,1], got {alpha}')
    assert_trainable_compatible(b2_state, a4_state)
    return {
        group: {
            key: torch.lerp(left.float(), a4_state[group][key].float(), alpha).to(dtype=left.dtype)
            for key, left in b2_state[group].items()
        }
        for group in TRAINABLE_GROUPS
    }


def assert_model_schema(model, endpoint_state: dict[str, dict[str, torch.Tensor]]) -> None:
    from peft import get_peft_model_state_dict

    current = {
        'transformer_lora': get_peft_model_state_dict(model.transformer),
        'adapter': model.adapter.state_dict(),
    }
    assert_trainable_compatible(current, endpoint_state)


def apply_trainable_state(model, state: dict[str, dict[str, torch.Tensor]]) -> None:
    from peft import set_peft_model_state_dict

    missing, unexpected = model.adapter.load_state_dict(state['adapter'], strict=True)
    if missing or unexpected:
        raise RuntimeError(f'adapter hot-load failed: missing={missing}, unexpected={unexpected}')
    set_peft_model_state_dict(model.transformer, state['transformer_lora'])


def load_endpoint_states(cfg: dict[str, Any]) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    icfg = cfg['inference_interpolation']
    endpoints = icfg['endpoints']
    fairness = verify_endpoint_fairness(
        endpoints['b2cont_checkpoint'], endpoints['a4_checkpoint'], icfg['expected_resume_trainable_hash']
    )
    b2_state, b2_meta = load_trainable_state(endpoints['b2cont_checkpoint'])
    a4_state, a4_meta = load_trainable_state(endpoints['a4_checkpoint'])
    schema = assert_trainable_compatible(b2_state, a4_state)
    if b2_meta['step'] != a4_meta['step']:
        raise RuntimeError(f'endpoint steps differ: b2={b2_meta["step"]}, a4={a4_meta["step"]}')
    metadata = {'fairness': fairness, 'schema': schema, 'b2cont': b2_meta, 'a4': a4_meta}
    return b2_state, a4_state, metadata


def load_inference_model(cfg: dict[str, Any], device: torch.device):
    from diffusers import AutoencoderKL

    from train_paired import WarmupFlowModel, load_components

    seed_everything(int(cfg['experiment']['seed']))
    dtype = choose_dtype(cfg['model']['precision'])
    transformer, controlnet, vae, adapter, pulid, _ = load_components(cfg, device, dtype)
    if vae is None:
        vae = AutoencoderKL.from_pretrained(
            cfg['model']['base'], subfolder='vae', torch_dtype=dtype, local_files_only=True
        ).to(device).eval()
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
    model.eval().requires_grad_(False)
    return model, vae, dtype


def runtime_shard(args) -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get('WORLD_SIZE', getattr(args, 'num_shards', 1)))
    rank = int(os.environ.get('RANK', getattr(args, 'shard_index', 0)))
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device(f'cuda:{local_rank}')
    return rank, world_size, torch.device(args.device)


def assigned_mids(mids: Iterable[str], rank: int, world_size: int) -> set[str]:
    return {str(mid) for index, mid in enumerate(mids) if index % int(world_size) == int(rank)}


def subset_for_mids(subset: dict[str, Any], mids: Iterable[str]) -> dict[str, Any]:
    selected = [str(mid) for mid in mids]
    keep = set(selected)
    payload = dict(subset)
    payload['mannequins'] = selected
    payload['pairs'] = [dict(row) for row in subset['pairs'] if str(row['mannequin_id']) in keep]
    payload['identity_pool'] = sorted({str(row['identity_id']) for row in payload['pairs']})
    payload['garment_types'] = {
        mid: subset.get('garment_types', {}).get(mid, 'unknown') for mid in selected
    }
    return payload


def stratified_mids(subset: dict[str, Any], count: int) -> list[str]:
    rows_by_mid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset['pairs']:
        rows_by_mid[str(row['mannequin_id'])].append(row)
    groups: dict[str, list[str]] = defaultdict(list)
    for mid_value in subset['mannequins']:
        mid = str(mid_value)
        rows = rows_by_mid[mid]
        garment = str(subset.get('garment_types', {}).get(mid) or rows[0].get('garment_type', 'unknown'))
        groups[garment].append(mid)
    names = sorted(groups)
    quota, remainder = divmod(int(count), max(1, len(names)))
    selected: list[str] = []
    for index, name in enumerate(names):
        selected.extend(groups[name][: quota + int(index < remainder)])
    if len(selected) < int(count):
        used = set(selected)
        selected.extend(str(mid) for mid in subset['mannequins'] if str(mid) not in used)
    selected = selected[: int(count)]
    if len(selected) != int(count):
        raise RuntimeError(f'could select only {len(selected)} of {count} stratified mids')
    return selected


def farthest_pair(ids: list[str], embeddings: dict[str, np.ndarray]) -> tuple[str, str, float]:
    if len(set(ids)) < 2:
        raise RuntimeError('identity interpolation requires at least two identities per mid')
    candidates = []
    for jid, kid in combinations(sorted(set(ids)), 2):
        if jid not in embeddings or kid not in embeddings:
            raise RuntimeError(f'missing selection embedding for pair {jid}/{kid}')
        distance = float(np.clip(1.0 - np.dot(embeddings[jid], embeddings[kid]), 0.0, 2.0))
        candidates.append((distance, jid, kid))
    distance, jid, kid = max(candidates, key=lambda row: (row[0], row[1], row[2]))
    return jid, kid, distance


def slerp_tokens(start: torch.Tensor, end: torch.Tensor, t: float, eps: float = 1e-6) -> tuple[torch.Tensor, dict[str, Any]]:
    if start.shape != end.shape:
        raise RuntimeError(f'PuLID interpolation shape mismatch: {tuple(start.shape)} vs {tuple(end.shape)}')
    t = float(t)
    if t <= 0.0:
        return start.clone(), {'fallback_tokens': 0, 'token_count': int(start.reshape(-1, start.shape[-1]).shape[0])}
    if t >= 1.0:
        return end.clone(), {'fallback_tokens': 0, 'token_count': int(end.reshape(-1, end.shape[-1]).shape[0])}
    x = start.float()
    y = end.float()
    x_norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    y_norm = torch.linalg.vector_norm(y, dim=-1, keepdim=True)
    x_unit = x / x_norm.clamp_min(eps)
    y_unit = y / y_norm.clamp_min(eps)
    dot = (x_unit * y_unit).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    fallback = (
        (x_norm <= eps) | (y_norm <= eps) | (sin_omega.abs() <= 1e-4)
        | ~torch.isfinite(omega) | ~torch.isfinite(sin_omega)
    )
    direction = (
        torch.sin((1.0 - t) * omega) / sin_omega.clamp_min(eps) * x_unit
        + torch.sin(t * omega) / sin_omega.clamp_min(eps) * y_unit
    )
    lerp_direction = torch.nn.functional.normalize(torch.lerp(x_unit, y_unit, t), dim=-1)
    direction = torch.where(fallback, lerp_direction, direction)
    norm = torch.lerp(x_norm, y_norm, t)
    output = (direction * norm).to(dtype=start.dtype)
    if not torch.isfinite(output).all():
        output = torch.lerp(start.float(), end.float(), t).to(dtype=start.dtype)
        fallback = torch.ones_like(fallback, dtype=torch.bool)
    return output, {
        'fallback_tokens': int(fallback.sum().item()),
        'token_count': int(fallback.numel()),
    }


def interpolate_identity_batch(
    batch_j: dict[str, torch.Tensor],
    batch_k: dict[str, torch.Tensor],
    t: float,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    pulid, diagnostics = slerp_tokens(batch_j['pulid_id_embed'], batch_k['pulid_id_embed'], t)
    if batch_j['appearance'].shape != batch_k['appearance'].shape:
        raise RuntimeError('appearance interpolation shape mismatch')
    batch = dict(batch_j)
    batch['pulid_id_embed'] = pulid
    batch['appearance'] = torch.lerp(batch_j['appearance'].float(), batch_k['appearance'].float(), float(t))
    return batch, diagnostics


def run_command(command: list[str], cwd: str | Path | None = None) -> None:
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_official_metrics(
    config: str | Path,
    subset: str | Path,
    gen_dir: str | Path,
    out_dir: str | Path,
    report: str | Path,
    device: str,
) -> None:
    run_command([
        sys.executable,
        'eval_b2_metrics.py',
        '--config', str(config),
        '--subset', str(subset),
        '--gen-dir', str(gen_dir),
        '--out-dir', str(out_dir),
        '--report', str(report),
        '--metrics', 'all',
        '--device', str(device),
    ])

