from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from conditions import arcface_embedding_from_path, find_one, load_yaml, read_ids, sha256_file


def load_attributes(root: Path, sample_id: str) -> dict:
    path = root / 'derived/id_attributes' / f'{sample_id}.json'
    if not path.exists():
        raise FileNotFoundError(f'missing identity attributes: {path}')
    row = json.loads(path.read_text(encoding='utf-8'))
    if row.get('status') != 'ok':
        raise RuntimeError(f'identity attributes not usable for {sample_id}: status={row.get("status")}')
    required = ('gender', 'age_raw', 'age_group', 'skin_cluster')
    missing = [key for key in required if row.get(key) is None]
    if missing:
        raise RuntimeError(f'identity attributes missing for {sample_id}: {missing}')
    return row


def normalize_embedding(embedding: np.ndarray, sample_id: str) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    if embedding.shape != (512,):
        raise RuntimeError(f'{sample_id} ArcFace shape={embedding.shape}, expected (512,)')
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm < 1e-8:
        raise RuntimeError(f'{sample_id} ArcFace embedding norm is invalid: {norm}')
    return embedding / norm


def embed_one(task: tuple) -> tuple[int, str, np.ndarray | None, str | None]:
    (
        index,
        sample_id,
        face_path,
        helper_python,
        helper_script,
        model_root,
        device_id,
    ) = task
    try:
        embedding = arcface_embedding_from_path(
            face_path,
            helper_python=helper_python,
            helper_script=helper_script,
            model_root=model_root,
            device_id=device_id,
            force_helper=True,
        )
        return index, sample_id, normalize_embedding(embedding, sample_id), None
    except Exception as exc:  # noqa: BLE001
        return index, sample_id, None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the train-split ArcFace/attribute identity bank for A2 sampling.')
    parser.add_argument('--config', default='configs/a2_diff.yaml')
    parser.add_argument('--output', default=None)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--flush-every', type=int, default=250)
    parser.add_argument('--keep-work', action='store_true')
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    output = Path(args.output) if args.output else root / cfg['data'].get('identity_bank', 'derived/identity_bank.npz')
    output.parent.mkdir(parents=True, exist_ok=True)
    ids = read_ids(root / cfg['data']['train_split'])
    if len(ids) != len(set(ids)):
        raise RuntimeError('train split contains duplicate IDs; identity bank order would be ambiguous')

    attrs = [load_attributes(root, sample_id) for sample_id in tqdm(ids, desc='identity attributes')]
    work_dir = output.with_suffix(output.suffix + '.work')
    work_dir.mkdir(parents=True, exist_ok=True)
    embeds_path = work_dir / 'embeds.npy'
    done_path = work_dir / 'done.npy'
    if embeds_path.exists() and done_path.exists():
        embeds = np.lib.format.open_memmap(embeds_path, mode='r+', dtype=np.float32, shape=(len(ids), 512))
        done = np.lib.format.open_memmap(done_path, mode='r+', dtype=np.bool_, shape=(len(ids),))
    else:
        embeds = np.lib.format.open_memmap(embeds_path, mode='w+', dtype=np.float32, shape=(len(ids), 512))
        done = np.lib.format.open_memmap(done_path, mode='w+', dtype=np.bool_, shape=(len(ids),))
        done[:] = False
        embeds.flush()
        done.flush()

    cache_cfg = cfg.get('cache', {})
    failures: list[dict[str, str]] = []
    completed_since_flush = 0
    helper_python = cache_cfg.get('arcface_helper_python')
    helper_script = cache_cfg.get('arcface_bank_helper_script', 'tools/arcface_bank_server.py')
    model_root = cache_cfg.get('arcface_model_root', '/data/muxiangyu/modelLibrary/insightface')
    tasks = [
        (
            index,
            sample_id,
            str(find_one(root / 'derived/face_crops/human', sample_id)),
            helper_python,
            helper_script,
            model_root,
            args.device_id,
        )
        for index, sample_id in enumerate(ids)
        if not bool(done[index])
    ]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = executor.map(embed_one, tasks, chunksize=1)
        for index, sample_id, embedding, error in tqdm(
            results,
            total=len(tasks),
            desc=f'ArcFace identity bank ({args.workers} workers)',
        ):
            if error is not None or embedding is None:
                failures.append({'id': sample_id, 'error': error or 'missing embedding'})
                continue
            embeds[index] = embedding
            done[index] = True
            completed_since_flush += 1
            if completed_since_flush >= args.flush_every:
                embeds.flush()
                done.flush()
                completed_since_flush = 0
    embeds.flush()
    done.flush()
    if failures or not bool(np.all(done)):
        failure_path = work_dir / 'failures.json'
        failure_path.write_text(json.dumps(failures, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        raise RuntimeError(
            f'identity bank incomplete: done={int(done.sum())}/{len(ids)}, failures={len(failures)}; '
            f'see {failure_path}'
        )

    payload = {
        'ids': np.asarray(ids, dtype=f'<U{max(len(value) for value in ids)}'),
        'embeds': np.asarray(embeds, dtype=np.float32),
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
        'train_split': str(root / cfg['data']['train_split']),
        'count': len(ids),
        'embedding': 'InsightFace antelopev2 ArcFace 512-d normalized embedding; sampling/calibration only',
        'compatibility': 'same gender, abs(skin_cluster)<=1, abs(age)<=15',
        'keys': {key: list(value.shape) for key, value in payload.items()},
        'output': str(output),
        'sha256': sha256_file(output),
    }
    output.with_suffix('.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    if not args.keep_work:
        del embeds
        del done
        shutil.rmtree(work_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
