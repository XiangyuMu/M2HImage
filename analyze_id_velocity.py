from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist

from conditions import choose_dtype, get_resolution, load_yaml, make_image_ids, seed_everything
from sampling_protected import (
    ensure_batch,
    load_token_masks,
    normalized_region_partition,
    prepare_conditions,
    resolve_root_path,
    transformer_velocity,
)


REGIONS = ('face', 'cloth_safe', 'body_bg', 'other')
DELTA_KINDS = ('delta_v_jk', 'delta_v_io')


def tau_bin_label(low: float, high: float) -> str:
    return f'{low:.2f}-{min(high, 1.0):.2f}'


def find_tau_bin(tau: float, bins: list[list[float]]) -> int:
    for index, (low, high) in enumerate(bins):
        if float(low) <= tau < float(high):
            return index
    raise ValueError(f'tau={tau} is outside configured bins={bins}')


def select_stratified_pairs(subset: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Select equal garment-type quotas and the first deterministic j/k pair per mid."""
    rows_by_mid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset['pairs']:
        rows_by_mid[str(row['mannequin_id'])].append(row)
    garment_types = subset.get('garment_types', {})
    groups: dict[str, list[str]] = defaultdict(list)
    for mid in subset['mannequins']:
        mid = str(mid)
        garment_type = str(garment_types.get(mid) or rows_by_mid[mid][0].get('garment_type', 'unknown'))
        groups[garment_type].append(mid)
    names = sorted(groups)
    if not names:
        raise RuntimeError('evaluation subset has no mannequin groups')
    quota, remainder = divmod(int(count), len(names))
    selected: list[str] = []
    for index, name in enumerate(names):
        selected.extend(groups[name][: quota + int(index < remainder)])
    if len(selected) < count:
        already = set(selected)
        selected.extend(mid for mid in subset['mannequins'] if mid not in already)
    selected = selected[:count]
    probes = []
    for mid in selected:
        identities = list(dict.fromkeys(str(row['identity_id']) for row in rows_by_mid[mid]))
        if len(identities) < 2:
            raise RuntimeError(f'mid={mid} has fewer than two distinct counterfactual identities')
        probes.append({
            'mid': mid,
            'jid': identities[0],
            'kid': identities[1],
            'seed': 0,
            'garment_type': str(garment_types.get(mid) or rows_by_mid[mid][0].get('garment_type', 'unknown')),
        })
    if len(probes) != count:
        raise RuntimeError(f'expected {count} stratified probes, selected {len(probes)}')
    return probes


def distributed_context(args: argparse.Namespace) -> tuple[int, int, torch.device, bool]:
    world_size = int(os.environ.get('WORLD_SIZE', args.num_shards))
    rank = int(os.environ.get('RANK', args.shard_index))
    distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    if distributed:
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        return rank, world_size, torch.device(f'cuda:{local_rank}'), True
    return rank, world_size, torch.device(args.device), False


def write_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ['mid', 'jid', 'kid', 'seed', 'garment_type', 'status', 'error', 'raw_path', 'mask_source']
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fields})


@torch.no_grad()
def run_probe(
    model,
    batch_j: dict[str, torch.Tensor],
    batch_k: dict[str, torch.Tensor],
    masks: dict[str, np.ndarray],
    mask_source: str,
    probe: dict[str, Any],
    *,
    steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(int(probe['seed']))
    z = torch.randn(
        1,
        (model.height // 16) * (model.width // 16),
        64,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    prompt, pooled, cond_j = prepare_conditions(model, batch_j, device, dtype, appearance_scale=1.0)
    _, _, cond_k = prepare_conditions(model, batch_k, device, dtype, appearance_scale=1.0)
    _, _, cond_off = prepare_conditions(model, batch_j, device, dtype, appearance_scale=0.0)
    pulid_j = ensure_batch(batch_j['pulid_id_embed'].to(device=device, dtype=dtype), 2)
    pulid_k = ensure_batch(batch_k['pulid_id_embed'].to(device=device, dtype=dtype), 2)
    pose = ensure_batch(batch_j['pose_latents'].to(device=device, dtype=dtype), 2)
    img_ids = make_image_ids(model.width, model.height, device, dtype)
    velocities: dict[str, list[torch.Tensor]] = {'v_j': [], 'v_k': [], 'v_off': []}
    taus = []
    for step_index in range(steps):
        tau_value = 1.0 - step_index / steps
        tau = torch.full((1,), tau_value, device=device, dtype=dtype)
        # ControlNet is identity-independent in this architecture, so one result is reused.
        cn_samples = model._controlnet_forward(z, tau, prompt, pooled, pose, img_ids)
        v_j = model._transformer_forward(
            z, tau, cond_j, pulid_j, cn_samples, pooled=pooled, img_ids=img_ids
        )
        v_k = model._transformer_forward(
            z, tau, cond_k, pulid_k, cn_samples, pooled=pooled, img_ids=img_ids
        )
        v_off = transformer_velocity(
            model,
            z,
            tau,
            cond_off,
            pulid_j,
            cn_samples,
            pooled=pooled,
            img_ids=img_ids,
            id_weight=0.0,
        )
        velocities['v_j'].append(v_j[0].to(device='cpu', dtype=torch.bfloat16))
        velocities['v_k'].append(v_k[0].to(device='cpu', dtype=torch.bfloat16))
        velocities['v_off'].append(v_off[0].to(device='cpu', dtype=torch.bfloat16))
        taus.append(tau_value)
        # The recorded trajectory is the real unmodified identity-j A4 trajectory.
        z = z - (1.0 / steps) * v_j
    v_j_tensor = torch.stack(velocities['v_j'])
    return {
        'metadata': {
            **probe,
            'steps': int(steps),
            'trajectory': 'unmodified A4 identity-j Euler trajectory',
            'controlnet_forwards_per_step': 1,
            'unique_transformer_forwards_per_step': 3,
            'v_on_storage': 'logical alias of v_j (identical full identity-j branch)',
            'mask_source': mask_source,
        },
        'tau': torch.tensor(taus, dtype=torch.float32),
        'v_j': v_j_tensor,
        'v_k': torch.stack(velocities['v_k']),
        'v_on': v_j_tensor,
        'v_off': torch.stack(velocities['v_off']),
        'masks': {key: torch.from_numpy(value.astype(np.float16)) for key, value in masks.items()},
    }


def deltas(payload: dict[str, Any], kind: str) -> torch.Tensor:
    if kind == 'delta_v_jk':
        return payload['v_j'].float() - payload['v_k'].float()
    if kind == 'delta_v_io':
        return payload['v_on'].float() - payload['v_off'].float()
    raise ValueError(kind)


def energy_rows(payloads: list[dict[str, Any]], bins: list[list[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        meta = payload['metadata']
        masks = {key: value.float().numpy() for key, value in payload['masks'].items()}
        partition = normalized_region_partition(masks)
        taus = payload['tau'].numpy()
        for kind in DELTA_KINDS:
            delta = deltas(payload, kind)
            token_energy = delta.square().sum(dim=-1).numpy()
            for step_index, tau in enumerate(taus):
                raw = {
                    region: float(np.sum(token_energy[step_index] * partition[region]))
                    for region in REGIONS
                }
                total = max(sum(raw.values()), 1e-20)
                row = {
                    'mid': meta['mid'],
                    'jid': meta['jid'],
                    'kid': meta['kid'],
                    'garment_type': meta['garment_type'],
                    'step': step_index,
                    'tau': float(tau),
                    'tau_bin': tau_bin_label(*bins[find_tau_bin(float(tau), bins)]),
                    'delta_kind': kind,
                    'total_energy': total,
                }
                for region in REGIONS:
                    row[f'{region}_energy'] = raw[region]
                    row[f'{region}_share'] = raw[region] / total
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_energy(rows: list[dict[str, Any]], kind: str, selector=None) -> dict[str, float]:
    selected = [row for row in rows if row['delta_kind'] == kind and (selector is None or selector(row))]
    totals = {region: sum(float(row[f'{region}_energy']) for row in selected) for region in REGIONS}
    denominator = max(sum(totals.values()), 1e-20)
    return {region: totals[region] / denominator for region in REGIONS}


def plot_energy(rows: list[dict[str, Any]], kind: str, path: Path) -> None:
    by_tau = sorted({float(row['tau']) for row in rows if row['delta_kind'] == kind}, reverse=True)
    plt.figure(figsize=(8.5, 5.2))
    for region in REGIONS:
        values = []
        for tau in by_tau:
            subset = [row for row in rows if row['delta_kind'] == kind and float(row['tau']) == tau]
            numerator = sum(float(row[f'{region}_energy']) for row in subset)
            denominator = sum(float(row['total_energy']) for row in subset)
            values.append(numerator / max(denominator, 1e-20))
        plt.plot(by_tau, values, marker='o', markersize=3, label=region)
    plt.xlabel('tau (sampling runs 1 to 0)')
    plt.ylabel('identity-induced velocity energy share')
    plt.title(kind.replace('_', ' '))
    plt.xlim(1.02, -0.02)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    plt.close()


def bin_step_indices(taus: torch.Tensor, bins: list[list[float]], bin_index: int) -> list[int]:
    return [
        index for index, tau in enumerate(taus.tolist())
        if find_tau_bin(float(tau), bins) == bin_index
    ]


def low_rank_analysis(
    payloads: list[dict[str, Any]],
    bins: list[list[float]],
    q_limit: int,
    niter: int,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    spectra: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    torch.manual_seed(90421)
    for kind in DELTA_KINDS:
        for bin_index, bounds in enumerate(bins):
            vectors = []
            for payload in payloads:
                indices = bin_step_indices(payload['tau'], bins, bin_index)
                if indices:
                    vectors.extend(deltas(payload, kind)[indices].reshape(len(indices), -1).unbind(0))
            if not vectors:
                continue
            matrix = torch.stack(vectors).to(device=device, dtype=torch.float32)
            q = min(int(q_limit), min(matrix.shape))
            u, singular, v = torch.svd_lowrank(matrix, q=q, niter=int(niter))
            singular = torch.sort(singular.detach().cpu(), descending=True).values
            total_energy = float(matrix.square().sum().detach().cpu())
            cumulative = torch.cumsum(singular.square(), dim=0) / max(total_energy, 1e-20)
            hits = torch.nonzero(cumulative >= 0.90)
            r90 = int(hits[0].item() + 1) if len(hits) else None
            label = tau_bin_label(*bounds)
            spectra[kind].append((label, singular.numpy()))
            for rank, value in enumerate(singular.tolist(), start=1):
                rows.append({
                    'delta_kind': kind,
                    'tau_bin': label,
                    'rank': rank,
                    'singular_value': value,
                    'cumulative_total_energy': float(cumulative[rank - 1]),
                    'r90': r90 if r90 is not None else f'>{q}',
                    'matrix_rows': int(matrix.shape[0]),
                    'matrix_features': int(matrix.shape[1]),
                    'computed_rank': q,
                })
            del matrix, u, v, singular, cumulative
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    for kind, entries in spectra.items():
        plt.figure(figsize=(8.5, 5.2))
        for label, singular in entries:
            normalized = np.square(singular) / max(float(np.sum(np.square(singular))), 1e-20)
            plt.plot(np.arange(1, len(normalized) + 1), normalized, label=f'tau {label}')
        plt.yscale('log')
        plt.xlabel('rank')
        plt.ylabel('fraction of computed singular energy')
        plt.title(f'{kind} singular spectrum (uncentered)')
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'{kind}_singular_spectrum.png', dpi=170)
        plt.close()
    return rows


def cosine_similarity_rows(
    payloads: list[dict[str, Any]], bins: list[list[float]], output_dir: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in DELTA_KINDS:
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), squeeze=False)
        for bin_index, bounds in enumerate(bins):
            sample_vectors: list[tuple[str, torch.Tensor, torch.Tensor]] = []
            for payload in payloads:
                indices = bin_step_indices(payload['tau'], bins, bin_index)
                if not indices:
                    continue
                direction = deltas(payload, kind)[indices].mean(dim=0)
                masks = {key: value.float().numpy() for key, value in payload['masks'].items()}
                face = torch.from_numpy(normalized_region_partition(masks)['face']).sqrt().unsqueeze(-1)
                sample_vectors.append((
                    payload['metadata']['mid'],
                    direction.reshape(-1),
                    (direction * face).reshape(-1),
                ))
            values: dict[str, list[float]] = {'full': [], 'face': []}
            for left in range(len(sample_vectors)):
                for right in range(left + 1, len(sample_vectors)):
                    for region, vector_index in (('full', 1), ('face', 2)):
                        a = sample_vectors[left][vector_index].float()
                        b = sample_vectors[right][vector_index].float()
                        denominator = float(a.norm() * b.norm())
                        cosine = float(torch.dot(a, b) / denominator) if denominator > 1e-20 else float('nan')
                        rows.append({
                            'delta_kind': kind,
                            'tau_bin': tau_bin_label(*bounds),
                            'region': region,
                            'mid_a': sample_vectors[left][0],
                            'mid_b': sample_vectors[right][0],
                            'cosine': cosine,
                        })
                        if np.isfinite(cosine):
                            values[region].append(cosine)
            axis = axes[bin_index // 2][bin_index % 2]
            bins_hist = np.linspace(-1.0, 1.0, 31)
            for region, color in (('full', '#2c6eaa'), ('face', '#d1495b')):
                axis.hist(
                    values[region],
                    bins=bins_hist,
                    alpha=0.5,
                    color=color,
                    label=f'{region} mean={np.mean(values[region]):.3f}',
                )
            axis.set_title(f'tau {tau_bin_label(*bounds)}')
            axis.set_xlim(-1.0, 1.0)
            axis.legend(fontsize=8)
        figure.suptitle(f'{kind} cross-sample cosine of bin-mean directions')
        figure.tight_layout()
        figure.savefig(output_dir / f'{kind}_cross_sample_cosine.png', dpi=170)
        plt.close(figure)
    return rows


def finalize_analysis(
    output_dir: Path,
    cfg: dict[str, Any],
    probes: list[dict[str, Any]],
    stats_device: torch.device,
) -> dict[str, Any]:
    raw_dir = output_dir / 'velocities'
    payloads = []
    failures = []
    for probe in probes:
        path = raw_dir / f"{probe['mid']}__id{probe['jid']}__id{probe['kid']}__seed{probe['seed']}.pt"
        if not path.exists():
            failures.append({**probe, 'error': 'raw velocity file missing'})
            continue
        payloads.append(torch.load(path, map_location='cpu'))
    if not payloads:
        raise RuntimeError('all identity velocity probes failed; no analysis can be produced')
    bins = [[float(value) for value in row] for row in cfg['inference_protection']['analysis']['tau_bins']]
    rows = energy_rows(payloads, bins)
    write_csv(output_dir / 'region_energy.csv', rows)
    for kind in DELTA_KINDS:
        plot_energy(rows, kind, output_dir / f'{kind}_region_energy.png')
    svd_rows = low_rank_analysis(
        payloads,
        bins,
        int(cfg['inference_protection']['analysis'].get('svd_q', 100)),
        int(cfg['inference_protection']['analysis'].get('svd_niter', 4)),
        stats_device,
        output_dir,
    )
    write_csv(output_dir / 'singular_spectrum.csv', svd_rows)
    cosine_rows = cosine_similarity_rows(payloads, bins, output_dir)
    write_csv(output_dir / 'cross_sample_cosine.csv', cosine_rows)

    global_energy = {kind: aggregate_energy(rows, kind) for kind in DELTA_KINDS}
    io_bins = []
    for bounds in bins:
        label = tau_bin_label(*bounds)
        shares = aggregate_energy(rows, 'delta_v_io', lambda row, label=label: row['tau_bin'] == label)
        io_bins.append({'range': bounds, 'label': label, **shares})
    recommendation = max(io_bins, key=lambda row: row['cloth_safe'])
    threshold = float(cfg['inference_protection']['analysis']['cloth_energy_null_threshold'])
    cloth_share = float(global_energy['delta_v_io']['cloth_safe'])
    null_signal = cloth_share < threshold

    r90_summary: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in svd_rows:
        r90_summary[row['delta_kind']][row['tau_bin']] = row['r90']
    grouped_cosines: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in cosine_rows:
        if np.isfinite(float(row['cosine'])):
            grouped_cosines[(row['delta_kind'], row['tau_bin'], row['region'])].append(float(row['cosine']))
    cosine_summary: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (kind, label, region), values in grouped_cosines.items():
        cosine_summary[kind][label][region] = float(np.mean(values))
    provenance = Counter(payload['metadata']['mask_source'] for payload in payloads)
    summary = {
        'status': 'ok',
        'requested_samples': len(probes),
        'successful_samples': len(payloads),
        'failed_samples': failures,
        'garment_type_counts': dict(Counter(payload['metadata']['garment_type'] for payload in payloads)),
        'steps': int(cfg['inference_protection']['steps']),
        'trajectory': 'unmodified A4 identity-j Euler trajectory, tau 1 to 0',
        'controlnet_reuse': 'one ControlNet forward per step shared by v_j, v_k, and v_off',
        'velocity_storage_dtype': 'bfloat16',
        'mask_partition': 'overlapping soft face/cloth/body masks plus residual are normalized per token',
        'mask_provenance': dict(provenance),
        'global_energy_share': global_energy,
        'delta_v_io_cloth_energy_share': cloth_share,
        'cloth_energy_null_threshold': threshold,
        'cloth_energy_below_threshold': null_signal,
        'part_c_recommended_without_confirmation': not null_signal,
        'delta_v_io_tau_bins': io_bins,
        'recommended_protect_tau_range': recommendation['range'],
        'recommended_tau_bin_reason': 'fixed tau bin with highest aggregate delta_v_io cloth_safe energy share',
        'r90': {kind: dict(values) for kind, values in r90_summary.items()},
        'cross_sample_cosine_mean': {
            kind: {label: dict(values) for label, values in labels.items()}
            for kind, labels in cosine_summary.items()
        },
        'artifacts': {
            'region_energy_csv': str(output_dir / 'region_energy.csv'),
            'singular_spectrum_csv': str(output_dir / 'singular_spectrum.csv'),
            'cross_sample_cosine_csv': str(output_dir / 'cross_sample_cosine.csv'),
            'velocity_dir': str(raw_dir),
        },
    }
    (output_dir / 'analysis_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    verdict = (
        'STOP BEFORE FULL PART C: cloth-safe identity-induced energy is below 10%; user confirmation is required.'
        if null_signal
        else 'PROCEED TO PART B/C: cloth-safe identity-induced energy is at least 10%.'
    )
    lines = [
        '# Identity Velocity Subspace Analysis',
        '',
        '## Feasibility First',
        '',
        f'**{verdict}**',
        '',
        f'- Delta v_io cloth_safe global energy share: **{cloth_share:.2%}** (threshold {threshold:.0%})',
        f"- Delta v_jk cloth_safe global energy share: {global_energy['delta_v_jk']['cloth_safe']:.2%}",
        f"- successful probes: {len(payloads)}/{len(probes)}; garment strata: {summary['garment_type_counts']}",
        f"- recommended fixed protection window: tau {recommendation['range']} "
        f"(cloth share {recommendation['cloth_safe']:.2%})",
        '',
        'Region energies use a normalized soft partition so face, cloth_safe, body_bg, and other sum to one at every token. '
        'Missing eval packed masks were projected in memory with the exact existing 16x16 average-pooling rule; no asset was written.',
        '',
        '## Region Energy',
        '',
        '| direction | face | cloth_safe | body_bg | other |',
        '|---|---:|---:|---:|---:|',
    ]
    for kind in DELTA_KINDS:
        values = global_energy[kind]
        lines.append(
            f"| {kind} | {values['face']:.2%} | {values['cloth_safe']:.2%} | "
            f"{values['body_bg']:.2%} | {values['other']:.2%} |"
        )
    lines.extend([
        '',
        f"- Delta v_jk curve: {output_dir / 'delta_v_jk_region_energy.png'}",
        f"- Delta v_io curve: {output_dir / 'delta_v_io_region_energy.png'}",
        '',
        '## Low Rank',
        '',
        'The spectra are uncentered velocity-direction matrices. r90 is the number of singular directions needed to explain '
        '90% of the full matrix Frobenius energy; a >q value means the configured low-rank probe did not capture 90%.',
        '',
        '| direction | tau bin | r90 |',
        '|---|---|---:|',
    ])
    for kind in DELTA_KINDS:
        for label, value in r90_summary[kind].items():
            lines.append(f'| {kind} | {label} | {value} |')
    lines.extend([
        '',
        '## Cross-sample Consistency',
        '',
        'Each sample contributes its mean direction within a tau bin. Face-restricted cosine uses the square root of the '
        'normalized face soft mask.',
        '',
        '| direction | tau bin | full cosine mean | face cosine mean |',
        '|---|---|---:|---:|',
    ])
    for kind in DELTA_KINDS:
        for bounds in bins:
            label = tau_bin_label(*bounds)
            values = cosine_summary[kind].get(label, {})
            lines.append(
                f"| {kind} | {label} | {values.get('full', float('nan')):.4f} | "
                f"{values.get('face', float('nan')):.4f} |"
            )
    lines.extend([
        '',
        f'Raw bf16 velocities: {raw_dir}',
        f'Mask provenance: {dict(provenance)}',
    ])
    (output_dir / 'analysis_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Probe identity-induced FLUX velocities along real A4 sampling trajectories.'
    )
    parser.add_argument('--config', default='configs/a4_protected.yaml')
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--subset', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stats-device', default=None)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--finalize-only', action='store_true')
    args = parser.parse_args()

    rank, world_size, device, distributed = distributed_context(args)
    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    pcfg = cfg['inference_protection']
    output_dir = resolve_root_path(root, pcfg['analysis']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_path = resolve_root_path(root, args.subset or pcfg['subset'])
    subset = json.loads(subset_path.read_text(encoding='utf-8'))
    probes = select_stratified_pairs(subset, int(pcfg['analysis']['sample_count']))
    if rank == 0:
        (output_dir / 'probe_protocol.json').write_text(
            json.dumps({'subset': str(subset_path), 'probes': probes}, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
    if distributed:
        dist.barrier()
    if args.finalize_only:
        if distributed:
            raise RuntimeError('--finalize-only must run as a single process')
        stats_device = torch.device(args.stats_device or str(device))
        summary = finalize_analysis(output_dir, cfg, probes, stats_device)
        print(json.dumps({
            'report': str(output_dir / 'analysis_report.md'),
            'cloth_energy_share': summary['delta_v_io_cloth_energy_share'],
            'part_c_recommended_without_confirmation': summary['part_c_recommended_without_confirmation'],
            'recommended_tau_range': summary['recommended_protect_tau_range'],
        }, indent=2))
        return
    assigned = [probe for index, probe in enumerate(probes) if index % world_size == rank]
    seed_everything(int(cfg['experiment']['seed']))
    dtype = choose_dtype(cfg['model']['precision'])

    from eval_b2 import make_cf_batch
    from train_paired import WarmupFlowModel, load_checkpoint, load_components

    analysis_cfg = dict(cfg)
    analysis_cfg['model'] = dict(cfg['model'])
    analysis_cfg['model']['load_vae_in_train'] = False
    transformer, controlnet, _vae, adapter, pulid, _ = load_components(analysis_cfg, device, dtype)
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, analysis_cfg)
    load_checkpoint(Path(args.ckpt or pcfg['checkpoint']), model)
    model.eval().requires_grad_(False)
    width, height = get_resolution(cfg['data']['resolution'])
    cache = root / cfg['data']['cache_dir'] / 'samples'
    text_cache = root / cfg['data']['cache_dir'] / 'text' / 'prompt.npz'
    raw_dir = output_dir / 'velocities'
    raw_dir.mkdir(parents=True, exist_ok=True)
    status_rows = []
    for probe in assigned:
        raw_path = raw_dir / f"{probe['mid']}__id{probe['jid']}__id{probe['kid']}__seed{probe['seed']}.pt"
        base = {**probe, 'raw_path': str(raw_path), 'error': ''}
        if raw_path.exists() and not args.overwrite:
            status_rows.append({**base, 'status': 'existing'})
            continue
        try:
            masks, source = load_token_masks(
                root,
                probe['mid'],
                width,
                height,
                cfg['data'].get('region_masks_z_dir', 'derived/region_masks_z'),
            )
            payload = run_probe(
                model,
                make_cf_batch(cache, text_cache, probe['mid'], probe['jid']),
                make_cf_batch(cache, text_cache, probe['mid'], probe['kid']),
                masks,
                source,
                probe,
                steps=int(pcfg['steps']),
                device=device,
                dtype=dtype,
            )
            torch.save(payload, raw_path)
            status_rows.append({**base, 'status': 'ok', 'mask_source': source})
        except Exception as exc:  # noqa: BLE001
            status_rows.append({**base, 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_status(output_dir / f'probe_status.shard{rank}.csv', status_rows)
    del model, transformer, controlnet, adapter, pulid
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    if distributed:
        dist.barrier()
    if rank == 0:
        stats_device = torch.device(args.stats_device or str(device))
        summary = finalize_analysis(output_dir, cfg, probes, stats_device)
        print(json.dumps({
            'report': str(output_dir / 'analysis_report.md'),
            'cloth_energy_share': summary['delta_v_io_cloth_energy_share'],
            'part_c_recommended_without_confirmation': summary['part_c_recommended_without_confirmation'],
            'recommended_tau_range': summary['recommended_protect_tau_range'],
        }, indent=2))
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
