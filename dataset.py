from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

from conditions import get_resolution, read_ids

SAMPLE_CACHE_KEYS = (
    'target_latents', 'pose_latents', 'pulid_id_embed', 'appearance', 'garment_grid', 'head_pose'
)
TEXT_CACHE_KEYS = ('prompt_embeds', 'pooled_prompt_embeds')
DIFFERENTIAL_MASK_KEYS = ('cloth_safe_z', 'body_bg_z', 'face_z')


class IdentityBank:
    REQUIRED_KEYS = ('ids', 'embeds', 'gender', 'age', 'age_group', 'skin_cluster')

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f'A2 identity bank missing: {self.path}')
        self.payload = np.load(self.path, mmap_mode='r', allow_pickle=False)
        missing = [key for key in self.REQUIRED_KEYS if key not in self.payload.files]
        if missing:
            raise RuntimeError(f'A2 identity bank missing keys {missing}: {self.path}')
        self.ids = np.asarray(self.payload['ids']).astype(str)
        self.embeds = np.asarray(self.payload['embeds'], dtype=np.float32)
        self.gender = np.asarray(self.payload['gender']).astype(str)
        self.age = np.asarray(self.payload['age'], dtype=np.float32)
        self.age_group = np.asarray(self.payload['age_group']).astype(str)
        self.skin_cluster = np.asarray(self.payload['skin_cluster'], dtype=np.int16)
        count = len(self.ids)
        if self.embeds.shape != (count, 512):
            raise RuntimeError(f'A2 identity bank embeds shape={self.embeds.shape}, expected {(count, 512)}')
        if any(len(value) != count for value in (self.gender, self.age, self.age_group, self.skin_cluster)):
            raise RuntimeError(f'A2 identity bank attribute lengths do not match ids: {self.path}')
        if not np.isfinite(self.embeds).all() or not np.isfinite(self.age).all():
            raise RuntimeError(f'A2 identity bank contains non-finite values: {self.path}')
        norms = np.linalg.norm(self.embeds, axis=1)
        if np.max(np.abs(norms - 1.0)) > 1e-3:
            raise RuntimeError(f'A2 identity bank embeddings are not normalized: max_error={np.max(np.abs(norms - 1.0))}')
        self.id_to_index = {sample_id: index for index, sample_id in enumerate(self.ids.tolist())}
        if len(self.id_to_index) != count:
            raise RuntimeError(f'A2 identity bank contains duplicate IDs: {self.path}')
        self._buckets: dict[tuple[str, int], np.ndarray] = {}
        for gender in sorted(set(self.gender.tolist())):
            for skin in sorted(set(int(value) for value in self.skin_cluster.tolist())):
                indices = np.flatnonzero((self.gender == gender) & (self.skin_cluster == skin))
                if len(indices):
                    self._buckets[(gender, skin)] = indices
        self._compatible_cache: dict[tuple[str, int, float], np.ndarray] = {}

    def compatible_indices(self, source_index: int) -> np.ndarray:
        gender = str(self.gender[source_index])
        skin = int(self.skin_cluster[source_index])
        age = float(self.age[source_index])
        key = (gender, skin, age)
        cached = self._compatible_cache.get(key)
        if cached is None:
            groups = [
                self._buckets[(gender, candidate_skin)]
                for candidate_skin in range(skin - 1, skin + 2)
                if (gender, candidate_skin) in self._buckets
            ]
            candidates = np.concatenate(groups) if groups else np.empty((0,), dtype=np.int64)
            candidates = candidates[np.abs(self.age[candidates] - age) <= 15.0]
            self._compatible_cache[key] = candidates
            cached = candidates
        return cached[cached != source_index]

    def sample_pair(self, sample_id: str, seed: int) -> tuple[int, int, float]:
        if sample_id not in self.id_to_index:
            raise RuntimeError(f'train sample {sample_id} is absent from A2 identity bank')
        source_index = self.id_to_index[sample_id]
        candidates = self.compatible_indices(source_index)
        if len(candidates) < 2:
            raise RuntimeError(
                f'A2 compatible identity pool has fewer than two entries for {sample_id}: {len(candidates)}'
            )
        rng = np.random.default_rng(np.uint64(seed))
        selected = rng.choice(candidates, size=2, replace=False)
        j_index, k_index = int(selected[0]), int(selected[1])
        cosine = float(np.dot(self.embeds[j_index], self.embeds[k_index]))
        delta_arc = float(np.clip(1.0 - cosine, 0.0, 2.0))
        return j_index, k_index, delta_arc

    def validate_sources(self, sample_ids: list[str]) -> dict[str, int]:
        missing = [sample_id for sample_id in sample_ids if sample_id not in self.id_to_index]
        insufficient = []
        minimum = len(self.ids)
        for sample_id in sample_ids:
            if sample_id not in self.id_to_index:
                continue
            count = len(self.compatible_indices(self.id_to_index[sample_id]))
            minimum = min(minimum, count)
            if count < 2:
                insufficient.append((sample_id, count))
        if missing or insufficient:
            raise RuntimeError(
                'A2 identity bank cannot satisfy strict j!=k!=i compatible sampling; '
                f'missing={missing[:10]}, insufficient={insufficient[:10]}'
            )
        return {
            'source_count': len(sample_ids),
            'minimum_compatible_identities': minimum if sample_ids else 0,
        }


class PairedWarmupDataset(Dataset):
    def __init__(self, config: dict[str, Any], split: str = 'train', require_coverage: bool = True) -> None:
        self.config = config
        self.root = Path(config['data']['root'])
        split_key = f'{split}_split'
        self.ids = read_ids(self.root / config['data'][split_key])
        self.excluded_ids: list[str] = []
        if split == 'train':
            requested = [str(value) for value in config['data'].get('exclude_train_ids', [])]
            unknown = sorted(set(requested) - set(self.ids))
            if unknown:
                raise RuntimeError(f'exclude_train_ids are absent from the train split: {unknown}')
            excluded = set(requested)
            self.excluded_ids = [sample_id for sample_id in self.ids if sample_id in excluded]
            self.ids = [sample_id for sample_id in self.ids if sample_id not in excluded]
        self.cache_dir = self.root / config['data']['cache_dir']
        width, height = get_resolution(config['data']['resolution'])
        self.token_count = (height // 16) * (width // 16)
        self.prompt_cache = self.cache_dir / 'text' / 'prompt.npz'
        self.split = split
        self.head_pose_dropout = float(config.get('training', {}).get('head_pose_dropout', 0.0)) if split == 'train' else 0.0
        self.base_seed = int(config.get('experiment', {}).get('seed', 0))
        differential = config.get('training', {}).get('differential', {})
        self.differential_enabled = bool(differential.get('enabled', False)) and split == 'train'
        self.region_masks_z_dir = self.root / config['data'].get('region_masks_z_dir', 'derived/region_masks_z')
        self.identity_bank = None
        if self.differential_enabled:
            bank_path = self.root / config['data'].get('identity_bank', 'derived/identity_bank.npz')
            self.identity_bank = IdentityBank(bank_path)
            self.identity_bank.validate_sources(self.ids)
        if require_coverage:
            self.assert_coverage()

    def __len__(self) -> int:
        return len(self.ids)

    def sample_path(self, sample_id: str) -> Path:
        return self.cache_dir / 'samples' / f'{sample_id}.npz'

    def assert_coverage(self) -> None:
        missing = []
        bad = []
        for sid in self.ids:
            path = self.sample_path(sid)
            if not path.exists():
                missing.append(str(path))
                continue
            try:
                with np.load(path, mmap_mode='r') as row:
                    absent = [key for key in SAMPLE_CACHE_KEYS if key not in row.files]
                    if absent:
                        bad.append(f'{path}: missing keys {absent}')
            except Exception as exc:  # noqa: BLE001
                bad.append(f'{path}: unreadable cache ({exc})')
            if self.differential_enabled:
                mask_path = self.region_masks_z_dir / f'{sid}.npz'
                if not mask_path.exists():
                    missing.append(str(mask_path))
                else:
                    try:
                        with np.load(mask_path, mmap_mode='r') as mask_row:
                            absent = [key for key in DIFFERENTIAL_MASK_KEYS if key not in mask_row.files]
                            if absent:
                                bad.append(f'{mask_path}: missing keys {absent}')
                            for key in DIFFERENTIAL_MASK_KEYS:
                                if key in mask_row.files and np.asarray(mask_row[key]).shape != (self.token_count,):
                                    bad.append(
                                        f'{mask_path}: {key} shape={np.asarray(mask_row[key]).shape}, '
                                        f'expected {(self.token_count,)}'
                                    )
                    except Exception as exc:  # noqa: BLE001
                        bad.append(f'{mask_path}: unreadable differential masks ({exc})')
        if not self.prompt_cache.exists():
            missing.append(str(self.prompt_cache))
        else:
            try:
                with np.load(self.prompt_cache, mmap_mode='r') as text:
                    absent = [key for key in TEXT_CACHE_KEYS if key not in text.files]
                    if absent:
                        bad.append(f'{self.prompt_cache}: missing keys {absent}')
            except Exception as exc:  # noqa: BLE001
                bad.append(f'{self.prompt_cache}: unreadable prompt cache ({exc})')
        if missing or bad:
            preview = '\n'.join((missing + bad)[:20])
            raise RuntimeError(
                f'Phase 1 cache coverage is not 100%; missing_files={len(missing)} bad_files={len(bad)}, '
                f'first entries:\n{preview}'
            )

    def __getitem__(self, index: int) -> dict[str, Any]:
        epoch, index = divmod(int(index), len(self.ids))
        sid = self.ids[index]
        with np.load(self.sample_path(sid), mmap_mode='r') as row:
            cached = {key: np.asarray(row[key]) for key in SAMPLE_CACHE_KEYS}
        with np.load(self.prompt_cache, mmap_mode='r') as text:
            prompt_embeds = np.asarray(text['prompt_embeds'])
            pooled_prompt_embeds = np.asarray(text['pooled_prompt_embeds'])
        head_pose = torch.from_numpy(cached['head_pose']).float()
        dropped = False
        if self.head_pose_dropout > 0.0 and torch.rand(()) < self.head_pose_dropout:
            head_pose = torch.zeros_like(head_pose)
            dropped = True
        item = {
            'index': torch.tensor(index, dtype=torch.long),
            'sample_epoch': torch.tensor(epoch, dtype=torch.long),
            'sample_id': sid,
            'target_latents': torch.from_numpy(cached['target_latents']).float(),
            'pose_latents': torch.from_numpy(cached['pose_latents']).float(),
            'prompt_embeds': torch.from_numpy(prompt_embeds).float(),
            'pooled_prompt_embeds': torch.from_numpy(pooled_prompt_embeds).float(),
            'pulid_id_embed': torch.from_numpy(cached['pulid_id_embed']).float(),
            'appearance': torch.from_numpy(cached['appearance']).float(),
            'garment': torch.from_numpy(cached['garment_grid']).float(),
            'head_pose': head_pose,
            'head_pose_is_null': torch.tensor(dropped or bool(torch.all(head_pose == 0)), dtype=torch.float32),
        }
        if self.differential_enabled:
            assert self.identity_bank is not None
            sample_seed = (
                self.base_seed * 1_000_003
                + epoch * 97_409
                + index * 65_537
            ) & ((1 << 63) - 1)
            j_index, k_index, delta_arc = self.identity_bank.sample_pair(sid, sample_seed)
            j_id = str(self.identity_bank.ids[j_index])
            k_id = str(self.identity_bank.ids[k_index])
            with np.load(self.sample_path(j_id), mmap_mode='r') as j_row:
                j_pulid = np.asarray(j_row['pulid_id_embed'])
                j_appearance = np.asarray(j_row['appearance'])
            with np.load(self.sample_path(k_id), mmap_mode='r') as k_row:
                k_pulid = np.asarray(k_row['pulid_id_embed'])
                k_appearance = np.asarray(k_row['appearance'])
            with np.load(self.region_masks_z_dir / f'{sid}.npz', mmap_mode='r') as masks:
                mask_values = {key: np.asarray(masks[key]) for key in DIFFERENTIAL_MASK_KEYS}
            item.update({
                'cf_j_id': j_id,
                'cf_k_id': k_id,
                'cf_j_pulid_id_embed': torch.from_numpy(j_pulid).float(),
                'cf_k_pulid_id_embed': torch.from_numpy(k_pulid).float(),
                'cf_j_appearance': torch.from_numpy(j_appearance).float(),
                'cf_k_appearance': torch.from_numpy(k_appearance).float(),
                'delta_arc_jk': torch.tensor(delta_arc, dtype=torch.float32),
                'cloth_safe_z': torch.from_numpy(mask_values['cloth_safe_z']).float(),
                'body_bg_z': torch.from_numpy(mask_values['body_bg_z']).float(),
                'face_z': torch.from_numpy(mask_values['face_z']).float(),
            })
        return item


class ResumeDistributedSampler(DistributedSampler):
    def __iter__(self):
        size = len(self.dataset)
        return iter([self.epoch * size + int(index) for index in super().__iter__()])

    def state_dict(self) -> dict[str, int]:
        return {'epoch': int(self.epoch)}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state.get('epoch', 0))


def write_cache_manifest(cache_dir: str | Path, payload: dict[str, Any]) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / 'manifest.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
