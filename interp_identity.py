from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from conditions import find_one, load_yaml
from dataset import IdentityBank
from eval_b2 import make_cf_batch
from eval_watcher import decode_tokens, generate
from interp_common import (
    apply_trainable_state,
    assert_model_schema,
    assigned_mids,
    farthest_pair,
    identity_image_name,
    interpolate_identity_batch,
    load_endpoint_states,
    load_inference_model,
    read_json,
    resolve_root_path,
    runtime_shard,
    sha256_file,
    stratified_mids,
    write_csv,
    write_json,
)


STATUS_FIELDS = (
    'checkpoint', 'mid', 'jid', 'kid', 't', 'seed', 'distance_jk', 'path',
    'status', 'error', 'seconds', 'slerp_fallback_tokens', 'slerp_token_count', 'rank', 'world_size',
)


def _load_or_extend_embeddings(
    cfg: dict[str, Any],
    requested_ids: list[str],
    output: Path,
    device: torch.device,
    overwrite: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = Path(cfg['data']['root'])
    identity_cfg = cfg['inference_interpolation']['identity']
    bank_path = resolve_root_path(root, identity_cfg['identity_bank'])
    bank = IdentityBank(bank_path)
    bank_map = {sample_id: bank.embeds[index] for sample_id, index in bank.id_to_index.items()}
    embeddings = {sample_id: np.asarray(bank_map[sample_id], dtype=np.float32) for sample_id in requested_ids if sample_id in bank_map}
    sources = {sample_id: 'identity_bank_v2' for sample_id in embeddings}
    missing = sorted(set(requested_ids) - set(embeddings))

    if output.exists() and not overwrite:
        cached = np.load(output, allow_pickle=False)
        cached_ids = np.asarray(cached['ids']).astype(str).tolist()
        cached_map = {
            sample_id: np.asarray(cached['embeds'][index], dtype=np.float32)
            for index, sample_id in enumerate(cached_ids)
        }
        cached_sources = np.asarray(cached['sources']).astype(str).tolist()
        for index, sample_id in enumerate(cached_ids):
            if sample_id in requested_ids and sample_id not in embeddings:
                embeddings[sample_id] = cached_map[sample_id]
                sources[sample_id] = cached_sources[index]
        missing = sorted(set(requested_ids) - set(embeddings))

    if missing:
        if not bool(identity_cfg.get('allow_eval_embedding_extension', False)):
            raise RuntimeError(
                f'identity_bank_v2 is train-only and misses eval IDs {missing}; '
                'set allow_eval_embedding_extension=true to use the same F_train recognizer in-memory'
            )
        # This is the training-side Glint360K ArcFace space, not held-out evaluation recognition.
        from train_recognizer import RetinaFaceGeometryDetector, TrainArcFaceRecognizer

        recognizer_cfg = cfg['model']['train_recognizer']
        recognizer = TrainArcFaceRecognizer(recognizer_cfg, device)
        detector = RetinaFaceGeometryDetector(recognizer_cfg, device_id=-1)
        aligned_images = []
        aligned_ids = []
        for sample_id in missing:
            face_path = find_one(root / 'derived/face_crops/human', sample_id)
            aligned, geometry = detector.align_path(
                face_path, image_size=int(recognizer_cfg.get('input_size', 112))
            )
            threshold = float(recognizer_cfg.get('bank_min_det_conf', 0.3))
            if geometry.confidence < threshold:
                raise RuntimeError(
                    f'eval extension face confidence too low for {sample_id}: '
                    f'{geometry.confidence:.4f} < {threshold:.4f}'
                )
            aligned_images.append(aligned)
            aligned_ids.append(sample_id)
        values = recognizer.embed_aligned_rgb_arrays(aligned_images)
        for sample_id, value in zip(aligned_ids, values, strict=True):
            embeddings[sample_id] = value
            sources[sample_id] = 'eval extension using bank-v2 F_train Glint360K recognizer'

    ordered = sorted(requested_ids)
    matrix = np.stack([embeddings[sample_id] for sample_id in ordered]).astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as handle:
        np.savez(
            handle,
            ids=np.asarray(ordered),
            embeds=matrix,
            sources=np.asarray([sources[sample_id] for sample_id in ordered]),
        )
    metadata = {
        'bank': str(bank_path),
        'bank_hash': sha256_file(bank_path),
        'bank_count': len(bank.ids),
        'requested_count': len(ordered),
        'direct_bank_hits': sum(sources[sample_id] == 'identity_bank_v2' for sample_id in ordered),
        'eval_extensions': [sample_id for sample_id in ordered if sources[sample_id] != 'identity_bank_v2'],
        'embedding_space': 'F_train Glint360K iresnet100 used by identity_bank_v2',
        'recognizer_checkpoint': str(cfg['model']['train_recognizer']['checkpoint']),
        'recognizer_hash': sha256_file(cfg['model']['train_recognizer']['checkpoint']),
        'heldout_recognizer_used': False,
        'output': str(output),
        'output_hash': sha256_file(output),
    }
    return {sample_id: matrix[index] for index, sample_id in enumerate(ordered)}, metadata


def prepare_selection(args: argparse.Namespace) -> Path:
    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    icfg = cfg['inference_interpolation']
    identity_cfg = icfg['identity']
    subset_path = resolve_root_path(root, args.subset or icfg['subset'])
    subset = read_json(subset_path)
    selected_mids = stratified_mids(subset, int(identity_cfg['sample_count']))
    rows_by_mid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in subset['pairs']:
        rows_by_mid[str(row['mannequin_id'])].append(row)
    requested_ids = sorted({str(row['identity_id']) for mid in selected_mids for row in rows_by_mid[mid]})
    embeddings_path = resolve_root_path(root, identity_cfg['selection_embeddings'])
    embeddings, embedding_meta = _load_or_extend_embeddings(
        cfg,
        requested_ids,
        embeddings_path,
        torch.device(args.device),
        overwrite=args.overwrite,
    )
    selection_rows = []
    for mid in selected_mids:
        identity_ids = [str(row['identity_id']) for row in rows_by_mid[mid]]
        jid, kid, distance = farthest_pair(identity_ids, embeddings)
        selection_rows.append({
            'mid': mid,
            'jid': jid,
            'kid': kid,
            'distance_jk': distance,
            'seed': int(identity_cfg['seed']),
            'garment_type': str(
                subset.get('garment_types', {}).get(mid)
                or rows_by_mid[mid][0].get('garment_type', 'unknown')
            ),
        })
    payload = {
        'protocol': '20 stratified mids; farthest pair among each mid evaluation identities in bank-v2 F_train space',
        'subset': str(subset_path),
        'sample_count': len(selection_rows),
        'ts': [float(value) for value in identity_cfg['ts']],
        'embedding_provenance': embedding_meta,
        'rows': selection_rows,
    }
    selection_path = resolve_root_path(root, identity_cfg['selection'])
    write_json(selection_path, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    return selection_path


def run_generation(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    icfg = cfg['inference_interpolation']
    identity_cfg = icfg['identity']
    selection_path = resolve_root_path(root, args.selection or identity_cfg['selection'])
    if not selection_path.exists():
        raise FileNotFoundError(
            f'identity interpolation selection missing: {selection_path}; run interp_identity.py --prepare-selection first'
        )
    selection = read_json(selection_path)
    rows = list(selection['rows'])
    rank, world_size, device = runtime_shard(args)
    if args.smoke:
        rows = rows[: int(identity_cfg['smoke_mid_count'])]
        output_root = resolve_root_path(root, args.output_root or identity_cfg['smoke_output_root'])
        rank, world_size = 0, 1
    else:
        output_root = resolve_root_path(root, args.output_root or identity_cfg['output_root'])
    keep = assigned_mids([row['mid'] for row in rows], rank, world_size)
    rows = [row for row in rows if row['mid'] in keep]
    output_root.mkdir(parents=True, exist_ok=True)

    b2_state, a4_state, endpoint_metadata = load_endpoint_states(cfg)
    model, vae, dtype = load_inference_model(cfg, device)
    assert_model_schema(model, b2_state)
    if rank == 0:
        write_json(output_root / 'interpolation_manifest.json', {
            'protocol': 'PuLID tokenwise normalized slerp with linear norm; appearance lerp',
            'selection': str(selection_path),
            'ts': identity_cfg['ts'],
            'steps': int(icfg['steps']),
            'endpoint_validation': endpoint_metadata,
        })

    cache = root / cfg['data']['cache_dir'] / 'samples'
    text_cache = root / cfg['data']['cache_dir'] / 'text' / 'prompt.npz'
    batch_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    status_rows: list[dict[str, Any]] = []
    endpoints = (('b2cont', b2_state), ('a4', a4_state))
    for checkpoint_name, endpoint_state in endpoints:
        apply_trainable_state(model, endpoint_state)
        branch_dir = output_root / checkpoint_name
        branch_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            mid, jid, kid = str(row['mid']), str(row['jid']), str(row['kid'])
            for identity_id in (jid, kid):
                key = (mid, identity_id)
                if key not in batch_cache:
                    batch_cache[key] = make_cf_batch(cache, text_cache, mid, identity_id)
            for t_value in identity_cfg['ts']:
                t = float(t_value)
                seed = int(row.get('seed', identity_cfg['seed']))
                output = branch_dir / identity_image_name(mid, jid, kid, t, seed)
                base = {
                    'checkpoint': checkpoint_name,
                    'mid': mid,
                    'jid': jid,
                    'kid': kid,
                    't': t,
                    'seed': seed,
                    'distance_jk': row['distance_jk'],
                    'path': str(output),
                    'error': '',
                    'rank': rank,
                    'world_size': world_size,
                }
                if output.exists() and not args.overwrite:
                    status_rows.append({**base, 'status': 'existing', 'seconds': 0.0})
                    continue
                started = time.perf_counter()
                try:
                    batch, diagnostics = interpolate_identity_batch(
                        batch_cache[(mid, jid)], batch_cache[(mid, kid)], t
                    )
                    tokens = generate(
                        model,
                        batch,
                        int(icfg['steps']),
                        seed=seed,
                        device=device,
                        dtype=dtype,
                    )
                    decode_tokens(vae, tokens, cfg['data']['resolution']).save(output)
                    status_rows.append({
                        **base,
                        'status': 'ok',
                        'seconds': time.perf_counter() - started,
                        'slerp_fallback_tokens': diagnostics['fallback_tokens'],
                        'slerp_token_count': diagnostics['token_count'],
                    })
                except Exception as exc:  # noqa: BLE001
                    status_rows.append({
                        **base,
                        'status': 'failed',
                        'error': f'{type(exc).__name__}: {exc}',
                        'seconds': time.perf_counter() - started,
                    })
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    status_root = resolve_root_path(root, identity_cfg['status_root'])
    suffix = 'smoke' if args.smoke else f'shard{rank}'
    status_path = status_root / f'generation.{suffix}.csv'
    write_csv(status_path, status_rows, list(STATUS_FIELDS))
    failed = sum(row['status'] == 'failed' for row in status_rows)
    print(json.dumps({
        'rank': rank,
        'world_size': world_size,
        'paths': len(status_rows),
        'failed': failed,
        'output_root': str(output_root),
        'status_csv': str(status_path),
    }, indent=2), flush=True)
    if failed:
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description='Pure-inference identity-condition interpolation comparison.')
    parser.add_argument('--config', default='configs/interpolation.yaml')
    parser.add_argument('--subset', default=None)
    parser.add_argument('--selection', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--output-root', default=None)
    parser.add_argument('--prepare-selection', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.prepare_selection:
        prepare_selection(args)
    else:
        run_generation(args)


if __name__ == '__main__':
    main()

