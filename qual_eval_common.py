from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


MAIN_RUNS = ('b2cont', 'alpha075', 'a4')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
ALL_RUNS = ('b2cont', 'alpha075', 'a4', 'a2')


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle)
    parent = payload.pop('extends', None) if isinstance(payload, dict) else None
    if not parent:
        return payload
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    base = load_config(parent_path)

    def merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        output = dict(left)
        for key, value in right.items():
            if isinstance(value, dict) and isinstance(output.get(key), dict):
                output[key] = merge(output[key], value)
            else:
                output[key] = value
        return output

    return merge(base, payload)


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'required CSV missing: {path}')
    with path.open('r', encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
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


def find_image(directory: str | Path, sample_id: str) -> Path:
    directory = Path(directory)
    for suffix in IMAGE_EXTS:
        path = directory / f'{sample_id}{suffix}'
        if path.exists():
            return path
    matches = sorted(path for path in directory.glob(f'{sample_id}.*') if path.is_file())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f'no image for id={sample_id} under {directory}')
    raise RuntimeError(f'ambiguous images for id={sample_id}: {matches}')


def generated_name(mid: str, jid: str, seed: int) -> str:
    return f'{mid}__id{jid}__seed{int(seed)}.png'


def identities_by_mid(subset: dict[str, Any]) -> dict[str, list[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in subset['pairs']:
        grouped[str(row['mannequin_id'])].add(str(row['identity_id']))
    return {mid: sorted(values) for mid, values in grouped.items()}


def load_delta_metrics(metrics_dir: Path) -> dict[tuple[str, str, int], dict[str, float | str]]:
    output: dict[tuple[str, str, int], dict[str, float | str]] = {}
    for row in read_csv(metrics_dir / 'deltaid_per_image.csv'):
        if row.get('status') != 'ok':
            continue
        try:
            key = (str(row['mid']), str(row['jid']), int(row['seed']))
            output[key] = {
                'sim_target': float(row['sim_target']),
                'sim_source': float(row['sim_source']),
                'delta_id': float(row['delta_id']),
                'det_conf': float(row['det_conf']),
                'garment_type': str(row.get('garment_type', 'unknown')),
            }
        except (KeyError, TypeError, ValueError):
            continue
    return output


def load_garment_per_mid(metrics_dir: Path) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in read_csv(metrics_dir / 'garment_pairwise_dino.csv'):
        try:
            grouped[str(row['mid'])].append(float(row['dino_cosine']))
        except (KeyError, TypeError, ValueError):
            continue
    return {mid: float(np.mean(values)) for mid, values in grouped.items() if values}


def run_assets(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = Path(cfg['data']['root'])
    configured = cfg['qualitative_eval']['runs']
    missing = sorted(set(ALL_RUNS) - set(configured))
    if missing:
        raise RuntimeError(f'qualitative run config is incomplete: {missing}')
    output = {}
    for name in ALL_RUNS:
        values = configured[name]
        gen_dir = resolve_path(root, values['gen_dir'])
        metrics_dir = resolve_path(root, values['metrics_dir'])
        output[name] = {
            'name': name,
            'label': str(values['label']),
            'gen_dir': gen_dir,
            'metrics_dir': metrics_dir,
            'delta': load_delta_metrics(metrics_dir),
            'garment': load_garment_per_mid(metrics_dir),
        }
    return output


def _mean_delta_gain(
    left: dict[tuple[str, str, int], dict[str, float | str]],
    right: dict[tuple[str, str, int], dict[str, float | str]],
    mids: Iterable[str],
) -> dict[str, float]:
    output = {}
    shared = set(left) & set(right)
    for mid in mids:
        values = [
            float(right[key]['delta_id']) - float(left[key]['delta_id'])
            for key in shared
            if key[0] == str(mid)
        ]
        if not values:
            raise RuntimeError(f'no paired DeltaID rows for mid={mid}')
        output[str(mid)] = float(np.mean(values))
    return output


def select_panel_mids(
    subset: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    target_count: int,
) -> tuple[list[str], dict[str, list[str]], dict[str, Any]]:
    all_mids = [str(mid) for mid in subset['mannequins']]
    garment_types = {str(mid): str(value) for mid, value in subset['garment_types'].items()}
    grouped: dict[str, list[str]] = defaultdict(list)
    for mid in all_mids:
        grouped[garment_types[mid]].append(mid)
    annotations: dict[str, list[str]] = defaultdict(list)
    selected: list[str] = []
    for garment_type in sorted(grouped):
        values = sorted(grouped[garment_type])
        positions = [('first', 0), ('middle', len(values) // 2), ('last', len(values) - 1)]
        for position, index in positions:
            mid = values[index]
            annotations[mid].append(f'stratified-{position}')
            if mid not in selected:
                selected.append(mid)

    garment_delta = {
        mid: float(runs['a4']['garment'][mid] - runs['b2cont']['garment'][mid])
        for mid in all_mids
    }
    deltaid_gain = _mean_delta_gain(runs['b2cont']['delta'], runs['a4']['delta'], all_mids)
    worst_ranking = sorted(all_mids, key=lambda mid: (garment_delta[mid], mid))
    best_ranking = sorted(all_mids, key=lambda mid: (-deltaid_gain[mid], mid))
    global_worst = worst_ranking[:2]
    global_best = best_ranking[:2]
    for mid in global_worst:
        annotations[mid].append('worst-garment')
        if mid not in selected:
            selected.append(mid)
    for mid in global_best:
        annotations[mid].append('best-identity')
        if mid not in selected:
            selected.append(mid)

    # The requested strata can overlap. Retain every global extreme, then
    # continue each registered ranking in alternation until exactly 16 unique
    # mids are available. This is deterministic and never inspects images.
    ranking_sources = (
        ('rank-fill-worst-garment', worst_ranking),
        ('rank-fill-best-identity', best_ranking),
    )
    while len(selected) < int(target_count):
        progressed = False
        for tag, ranking in ranking_sources:
            candidate = next((mid for mid in ranking if mid not in selected), None)
            if candidate is None:
                continue
            selected.append(candidate)
            annotations[candidate].append(tag)
            progressed = True
            if len(selected) == int(target_count):
                break
        if not progressed:
            raise RuntimeError(f'could not fill panel selection to {target_count} mids')
    if len(selected) != int(target_count):
        raise RuntimeError(f'panel selection has {len(selected)} mids, expected {target_count}')

    diagnostics = {
        'global_worst_garment': [
            {'mid': mid, 'a4_minus_b2cont': garment_delta[mid]} for mid in global_worst
        ],
        'global_best_identity': [
            {'mid': mid, 'deltaid_gain': deltaid_gain[mid]} for mid in global_best
        ],
        'garment_delta': garment_delta,
        'deltaid_gain': deltaid_gain,
        'overlap_note': (
            'Global extremes already present in the 12 stratified mids remain tagged; '
            'rank-fill entries continue the same two rankings to preserve 16 unique mids.'
        ),
    }
    return selected, {mid: sorted(set(tags)) for mid, tags in annotations.items()}, diagnostics


def verify_selected_assets(
    root: Path,
    selected: list[str],
    identities: dict[str, list[str]],
    runs: dict[str, dict[str, Any]],
    seed: int,
    expected_identity_count: int,
) -> dict[str, dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    sources: dict[str, dict[str, Any]] = {}
    for mid in selected:
        mids_identities = identities.get(mid, [])
        if len(mids_identities) != int(expected_identity_count):
            missing.append({
                'kind': 'identity-count', 'mid': mid,
                'expected': int(expected_identity_count), 'actual': len(mids_identities),
            })
            continue
        try:
            mannequin = find_image(root / 'images/mannequin', mid)
        except Exception as exc:  # noqa: BLE001
            missing.append({'kind': 'mannequin', 'mid': mid, 'error': str(exc)})
            mannequin = None
        mask = root / 'derived/region_masks' / f'{mid}.npz'
        if not mask.exists():
            missing.append({'kind': 'region-mask', 'mid': mid, 'path': str(mask)})
        face_paths = {}
        for jid in mids_identities:
            try:
                face_paths[jid] = find_image(root / 'derived/face_crops/human', jid)
            except Exception as exc:  # noqa: BLE001
                missing.append({'kind': 'identity-face', 'mid': mid, 'jid': jid, 'error': str(exc)})
        generated = {}
        for run_name, run in runs.items():
            if mid not in run['garment']:
                missing.append({'kind': 'garment-metric', 'run': run_name, 'mid': mid})
            for jid in mids_identities:
                key = (mid, jid, int(seed))
                image = run['gen_dir'] / generated_name(mid, jid, seed)
                generated[(run_name, jid)] = image
                if not image.exists():
                    missing.append({
                        'kind': 'generated-image', 'run': run_name,
                        'mid': mid, 'jid': jid, 'seed': int(seed), 'path': str(image),
                    })
                if key not in run['delta']:
                    missing.append({
                        'kind': 'deltaid-metric', 'run': run_name,
                        'mid': mid, 'jid': jid, 'seed': int(seed),
                    })
        sources[mid] = {
            'mannequin': mannequin,
            'mask': mask,
            'faces': face_paths,
            'generated': generated,
        }
    if missing:
        details = json.dumps(missing, indent=2, ensure_ascii=False)
        raise RuntimeError(f'qualitative evaluation asset check failed ({len(missing)} issues):\n{details}')
    return sources


def build_panel_context(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    root = Path(cfg['data']['root'])
    qcfg = cfg['qualitative_eval']
    subset_path = resolve_path(root, qcfg['subset'])
    subset = read_json(subset_path)
    identities = identities_by_mid(subset)
    runs = run_assets(cfg)
    selected, annotations, diagnostics = select_panel_mids(
        subset,
        runs,
        int(qcfg['panel_mid_count']),
    )
    sources = verify_selected_assets(
        root,
        selected,
        identities,
        runs,
        int(qcfg['seed_for_panels']),
        int(qcfg['identities_per_mid']),
    )
    return {
        'cfg': cfg,
        'qcfg': qcfg,
        'root': root,
        'subset_path': subset_path,
        'subset': subset,
        'identities': identities,
        'runs': runs,
        'selected': selected,
        'annotations': annotations,
        'diagnostics': diagnostics,
        'sources': sources,
    }


def panel_manifest(context: dict[str, Any]) -> dict[str, Any]:
    qcfg = context['qcfg']
    subset = context['subset']
    runs = context['runs']
    seed = int(qcfg['seed_for_panels'])
    rows = []
    for order, mid in enumerate(context['selected']):
        jids = context['identities'][mid]
        run_metrics = {}
        for run_name in ALL_RUNS:
            run = runs[run_name]
            sim_values = [float(run['delta'][(mid, jid, seed)]['sim_target']) for jid in jids]
            run_metrics[run_name] = {
                'label': run['label'],
                'garment_sim': float(run['garment'][mid]),
                'sim_target_seed0_mean': float(np.mean(sim_values)),
                'sim_target_seed0_by_jid': {
                    jid: float(run['delta'][(mid, jid, seed)]['sim_target']) for jid in jids
                },
            }
        rows.append({
            'order': order,
            'mid': mid,
            'garment_type': str(subset['garment_types'][mid]),
            'tags': context['annotations'][mid],
            'identity_ids': jids,
            'a4_minus_b2cont_garment': context['diagnostics']['garment_delta'][mid],
            'a4_minus_b2cont_deltaid': context['diagnostics']['deltaid_gain'][mid],
            'runs': run_metrics,
        })
    return {
        'schema_version': 1,
        'protocol': {
            'seed': int(qcfg['seed']),
            'image_seed': seed,
            'selection': (
                'Within each garment type, sort mid IDs and take first, upper-middle (n//2), and last; '
                'force global two worst A4-B2cont GarmentSim and global two best DeltaID gains; '
                'continue the same rankings only when overlap must be filled to 16 unique mids.'
            ),
            'metric_source': 'existing deltaid_per_image.csv and garment_pairwise_dino.csv; no recomputation',
            'subset': str(context['subset_path']),
            'subset_hash': sha256_file(context['subset_path']),
            'run_dirs': {name: str(context['runs'][name]['gen_dir']) for name in ALL_RUNS},
            'metrics_dirs': {name: str(context['runs'][name]['metrics_dir']) for name in ALL_RUNS},
            'overlap_note': context['diagnostics']['overlap_note'],
        },
        'diagnostics': {
            'global_worst_garment': context['diagnostics']['global_worst_garment'],
            'global_best_identity': context['diagnostics']['global_best_identity'],
        },
        'rows': rows,
    }
