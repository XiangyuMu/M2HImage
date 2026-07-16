from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from build_identity_bank import load_attributes
from conditions import find_one, load_yaml, read_ids, sha256_file
from train_recognizer import RetinaFaceGeometryDetector, TrainArcFaceRecognizer


def setup_dist() -> tuple[int, int, int]:
    if 'RANK' not in os.environ:
        return 0, 1, int(os.environ.get('LOCAL_RANK', 0))
    dist.init_process_group(backend='nccl')
    return int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])


def flush_batch(
    recognizer: TrainArcFaceRecognizer,
    pending_images: list[np.ndarray],
    pending_indices: list[int],
    embeds: np.memmap,
    done: np.memmap,
) -> None:
    if not pending_images:
        return
    output = recognizer.embed_aligned_rgb_arrays(pending_images)
    for local_index, embedding in zip(pending_indices, output, strict=True):
        embeds[local_index] = embedding
        done[local_index] = True
    pending_images.clear()
    pending_indices.clear()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build identity_bank_v2 with frozen PyTorch Glint360K ArcFace embeddings.'
    )
    parser.add_argument('--config', default='configs/a4_directed.yaml')
    parser.add_argument('--output', default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--keep-work', action='store_true')
    args = parser.parse_args()

    rank, world_size, local_rank = setup_dist()
    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    output = (
        Path(args.output)
        if args.output
        else root / cfg['data'].get('identity_bank_v2', 'derived/identity_bank_v2.npz')
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    ids = read_ids(root / cfg['data']['train_split'])
    if args.limit is not None:
        ids = ids[: int(args.limit)]
    if len(ids) != len(set(ids)):
        raise RuntimeError('train split contains duplicate IDs')

    shard_positions = [position for position in range(len(ids)) if position % world_size == rank]
    shard_ids = [ids[position] for position in shard_positions]
    work_dir = output.with_suffix(output.suffix + '.work')
    work_dir.mkdir(parents=True, exist_ok=True)
    embeds_path = work_dir / f'embeds.rank{rank}.npy'
    done_path = work_dir / f'done.rank{rank}.npy'
    if args.overwrite:
        embeds_path.unlink(missing_ok=True)
        done_path.unlink(missing_ok=True)
    if embeds_path.exists() and done_path.exists():
        embeds = np.lib.format.open_memmap(
            embeds_path, mode='r+', dtype=np.float32, shape=(len(shard_ids), 512)
        )
        done = np.lib.format.open_memmap(
            done_path, mode='r+', dtype=np.bool_, shape=(len(shard_ids),)
        )
    else:
        embeds = np.lib.format.open_memmap(
            embeds_path, mode='w+', dtype=np.float32, shape=(len(shard_ids), 512)
        )
        done = np.lib.format.open_memmap(
            done_path, mode='w+', dtype=np.bool_, shape=(len(shard_ids),)
        )
        done[:] = False
        embeds.flush()
        done.flush()

    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    recognizer = TrainArcFaceRecognizer(cfg['model']['train_recognizer'], device)
    detector = RetinaFaceGeometryDetector(cfg['model']['train_recognizer'], device_id=-1)
    pending_images: list[np.ndarray] = []
    pending_indices: list[int] = []
    failures: list[dict[str, str]] = []
    progress = tqdm(
        range(len(shard_ids)),
        desc=f'identity bank v2 rank {rank}/{world_size}',
        disable=rank != 0,
    )
    for local_index in progress:
        if bool(done[local_index]):
            continue
        sample_id = shard_ids[local_index]
        try:
            face_path = find_one(root / 'derived/face_crops/human', sample_id)
            aligned, geometry = detector.align_path(
                face_path,
                image_size=int(cfg['model']['train_recognizer'].get('input_size', 112)),
            )
            if geometry.confidence < float(cfg['model']['train_recognizer'].get('bank_min_det_conf', 0.3)):
                raise RuntimeError(f'face detection confidence too low: {geometry.confidence:.4f}')
            pending_images.append(aligned)
            pending_indices.append(local_index)
            if len(pending_images) >= int(args.batch_size):
                flush_batch(recognizer, pending_images, pending_indices, embeds, done)
        except Exception as exc:  # noqa: BLE001
            failures.append({'id': sample_id, 'error': str(exc)})
    flush_batch(recognizer, pending_images, pending_indices, embeds, done)
    embeds.flush()
    done.flush()
    if failures:
        (work_dir / f'failures.rank{rank}.json').write_text(
            json.dumps(failures, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
    local_ok = not failures and bool(np.all(done))
    if dist.is_initialized():
        flag = torch.tensor(1 if local_ok else 0, device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        all_ok = bool(flag.item())
    else:
        all_ok = local_ok
    if not all_ok:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise RuntimeError(
            f'identity bank v2 shard failed on rank={rank}; '
            f'done={int(done.sum())}/{len(done)}, failures={len(failures)}, work={work_dir}'
        )
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        merged = np.empty((len(ids), 512), dtype=np.float32)
        for shard_rank in range(world_size):
            positions = [
                position for position in range(len(ids))
                if position % world_size == shard_rank
            ]
            shard = np.load(work_dir / f'embeds.rank{shard_rank}.npy', mmap_mode='r')
            if shard.shape != (len(positions), 512):
                raise RuntimeError(
                    f'identity bank v2 shard {shard_rank} shape={shard.shape}, '
                    f'expected {(len(positions), 512)}'
                )
            merged[np.asarray(positions, dtype=np.int64)] = np.asarray(shard)
        norms = np.linalg.norm(merged, axis=1)
        if not np.isfinite(merged).all() or np.max(np.abs(norms - 1.0)) > 1e-3:
            raise RuntimeError(
                f'identity bank v2 embeddings invalid: '
                f'finite={np.isfinite(merged).all()}, max_norm_error={np.max(np.abs(norms - 1.0))}'
            )
        attrs = [load_attributes(root, sample_id) for sample_id in tqdm(ids, desc='identity attributes')]
        payload = {
            'ids': np.asarray(ids, dtype=f'<U{max(len(value) for value in ids)}'),
            'embeds': merged,
            'gender': np.asarray([row['gender'] for row in attrs], dtype='<U16'),
            'age': np.asarray([float(row['age_raw']) for row in attrs], dtype=np.float32),
            'age_group': np.asarray([row['age_group'] for row in attrs], dtype='<U32'),
            'skin_cluster': np.asarray([int(row['skin_cluster']) for row in attrs], dtype=np.int16),
        }
        tmp = output.with_suffix(output.suffix + '.tmp')
        with tmp.open('wb') as handle:
            np.savez(handle, **payload)
        tmp.replace(output)
        manifest = {
            'config': args.config,
            'count': len(ids),
            'embedding': recognizer.launch_note(),
            'detector': {
                'type': 'InsightFace antelopev2 RetinaFace, alignment only',
                'min_confidence': cfg['model']['train_recognizer'].get('bank_min_det_conf', 0.3),
            },
            'compatibility': 'same gender, abs(skin_cluster)<=1, abs(age)<=15',
            'output': str(output),
            'sha256': sha256_file(output),
            'keys': {key: list(value.shape) for key, value in payload.items()},
        }
        output.with_suffix('.json').write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + '\n',
            encoding='utf-8',
        )
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        if not args.keep_work:
            del embeds
            del done
            shutil.rmtree(work_dir)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
