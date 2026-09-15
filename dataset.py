from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

from conditions import get_resolution, read_ids

BASE_SAMPLE_CACHE_KEYS = (
    'target_latents', 'pose_latents', 'pulid_id_embed', 'appearance', 'garment_grid', 'head_pose'
)
SPATIAL_GARMENT_KEYS = ('garment_ref_latents',)
SPATIAL_HEAD_KEYS = ('pose_synth_latents',)
SPATIAL_HAIR_TOKEN_KEYS = ('hair_ref_tokens', 'hair_ref_positions', 'hair_ref_mask')
SPATIAL_HAIR_SEMANTIC_KEYS = ('hair_semantic_tokens', 'hair_semantic_mask')
SPATIAL_HAIR_REFERENCE_KEYS = ('hair_ref_latents', 'hair_ref_empty')
# Backward-compatible import used by older differential tooling.
SPATIAL_HAIR_KEYS = SPATIAL_HAIR_TOKEN_KEYS
# Retained for imports in older tooling. Runtime coverage uses sample_cache_keys(config).
SAMPLE_CACHE_KEYS = BASE_SAMPLE_CACHE_KEYS
TEXT_CACHE_KEYS = ('prompt_embeds', 'pooled_prompt_embeds')
DIFFERENTIAL_MASK_KEYS = ('cloth_safe_z', 'body_bg_z', 'face_z')
ASYNC_FLOW_MASK_KEYS = ('cloth_safe_z', 'hair_z', 'face_z')
PAIRED_REGION_MASK_KEYS = ('cloth_safe_z', 'hair_z', 'face_z')


def sample_cache_keys(config: dict[str, Any]) -> tuple[str, ...]:
    spatial = config.get('model', {}).get('spatial_conditions', {})
    adapter = config.get('model', {}).get('identity_adapter', {})
    keys = list(BASE_SAMPLE_CACHE_KEYS)
    if spatial.get('garment_reference', {}).get('enabled', False):
        keys.extend(SPATIAL_GARMENT_KEYS)
    if spatial.get('head_control', {}).get('enabled', False):
        keys.extend(SPATIAL_HEAD_KEYS)
    hair_cfg = spatial.get('hair', {})
    hair_loss_enabled = bool(
        config.get('training', {}).get('hair_loss', {}).get('enabled', False)
    )
    hair_tokens_enabled = bool(adapter.get('use_hair_tokens', False))
    hair_token_mode = str(adapter.get('hair_token_mode', 'dense_with_position')).lower()
    retain_legacy = bool(adapter.get('retain_legacy_hair_parameters_for_resume', False))
    if hair_loss_enabled or retain_legacy or (
        hair_tokens_enabled and hair_token_mode != 'semantic_dense'
    ):
        keys.extend(SPATIAL_HAIR_TOKEN_KEYS)
    if hair_tokens_enabled and hair_token_mode == 'semantic_dense':
        keys.extend(SPATIAL_HAIR_SEMANTIC_KEYS)
    if hair_cfg.get('reference', {}).get('enabled', False):
        keys.extend(SPATIAL_HAIR_REFERENCE_KEYS)
    return tuple(dict.fromkeys(keys))


class IdentityBank:
    REQUIRED_KEYS = ('ids', 'embeds', 'gender', 'age', 'age_group', 'skin_cluster')
    RELAXATION_NAMES = ('strict', 'relax_age', 'relax_skin')

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
        self._compatible_cache: dict[tuple[str, int, float, int], np.ndarray] = {}

    def _candidate_base(self, source_index: int, relaxation: int = 0) -> np.ndarray:
        if relaxation not in (0, 1, 2):
            raise ValueError(f'unknown identity compatibility relaxation={relaxation}')
        gender = str(self.gender[source_index])
        skin = int(self.skin_cluster[source_index])
        age = float(self.age[source_index])
        key = (gender, skin if relaxation < 2 else -1, age if relaxation == 0 else -1.0, relaxation)
        cached = self._compatible_cache.get(key)
        if cached is None:
            if relaxation < 2:
                groups = [
                    self._buckets[(gender, candidate_skin)]
                    for candidate_skin in range(skin - 1, skin + 2)
                    if (gender, candidate_skin) in self._buckets
                ]
                candidates = np.concatenate(groups) if groups else np.empty((0,), dtype=np.int64)
            else:
                candidates = np.flatnonzero(self.gender == gender)
            if relaxation == 0:
                candidates = candidates[np.abs(self.age[candidates] - age) <= 15.0]
            cached = candidates.astype(np.int64, copy=False)
            self._compatible_cache[key] = cached
        return cached

    def candidate_indices(self, source_index: int, relaxation: int = 0) -> np.ndarray:
        cached = self._candidate_base(source_index, relaxation)
        return cached[cached != source_index]

    def compatible_indices(self, source_index: int) -> np.ndarray:
        """A2-compatible strict pool retained for backward compatibility."""
        return self.candidate_indices(source_index, relaxation=0)

    @staticmethod
    def _farthest_pair(indices: np.ndarray, embeds: np.ndarray) -> tuple[int, int, float]:
        if len(indices) < 2:
            raise RuntimeError('semi-hard pool must contain at least two identities')
        selected = np.sort(np.asarray(indices, dtype=np.int64))
        features = embeds[selected]
        distances = 1.0 - features @ features.T
        distances[np.tril_indices(len(selected))] = -np.inf
        flat = int(np.argmax(distances))
        row, col = np.unravel_index(flat, distances.shape)
        distance = float(np.clip(distances[row, col], 0.0, 2.0))
        return int(selected[row]), int(selected[col]), distance

    @staticmethod
    def _sample_without_replacement(
        rng: np.random.Generator,
        candidates: np.ndarray,
        count: int,
        exclude_index: int | None = None,
    ) -> np.ndarray:
        available = len(candidates) - int(exclude_index is not None)
        count = min(int(count), available)
        if count == available:
            if exclude_index is None:
                return np.asarray(candidates, dtype=np.int64)
            return candidates[candidates != exclude_index].astype(np.int64, copy=False)
        positions: set[int] = set()
        while len(positions) < count:
            position = int(rng.integers(0, len(candidates)))
            if exclude_index is not None and int(candidates[position]) == exclude_index:
                continue
            positions.add(position)
        return np.asarray([candidates[position] for position in sorted(positions)], dtype=np.int64)

    def sample_pair_details(
        self,
        sample_id: str,
        seed: int,
        sampling: str = 'random',
        semihard_pool: int = 8,
    ) -> dict[str, int | float | str]:
        if sample_id not in self.id_to_index:
            raise RuntimeError(f'train sample {sample_id} is absent from identity bank')
        source_index = self.id_to_index[sample_id]
        rng = np.random.default_rng(np.uint64(seed))
        sampling = str(sampling).lower()
        relaxation = 0
        if sampling == 'random':
            candidates = self.candidate_indices(source_index, relaxation=0)
            candidate_count = len(candidates)
            if candidate_count < 2:
                raise RuntimeError(
                    f'strict compatible identity pool has fewer than two entries for {sample_id}: {candidate_count}'
                )
            # Preserve the exact A2 random-policy RNG behavior for reproducibility.
            selected = rng.choice(candidates, size=2, replace=False)
            j_index, k_index = int(selected[0]), int(selected[1])
            delta_arc = float(np.clip(1.0 - np.dot(self.embeds[j_index], self.embeds[k_index]), 0.0, 2.0))
            pool_count = 2
        elif sampling == 'semihard':
            requested = max(2, int(semihard_pool))
            candidates = self._candidate_base(source_index, relaxation=0)
            candidate_count = len(candidates) - 1
            while candidate_count < requested and relaxation < 2:
                relaxation += 1
                candidates = self._candidate_base(source_index, relaxation=relaxation)
                candidate_count = len(candidates) - 1
            if candidate_count < 2:
                raise RuntimeError(
                    f'same-gender identity pool has fewer than two entries for {sample_id}: {candidate_count}'
                )
            pool_count = min(requested, candidate_count)
            pool = self._sample_without_replacement(
                rng,
                candidates,
                pool_count,
                exclude_index=source_index,
            )
            j_index, k_index, delta_arc = self._farthest_pair(pool, self.embeds)
        else:
            raise ValueError(f'unsupported differential sampling={sampling!r}; expected random or semihard')
        return {
            'j_index': j_index,
            'k_index': k_index,
            'delta_arc': delta_arc,
            'relaxation': relaxation,
            'relaxation_name': self.RELAXATION_NAMES[relaxation],
            'candidate_count': int(candidate_count),
            'pool_count': int(pool_count),
        }

    def sample_pair(self, sample_id: str, seed: int) -> tuple[int, int, float]:
        details = self.sample_pair_details(sample_id, seed, sampling='random')
        return int(details['j_index']), int(details['k_index']), float(details['delta_arc'])

    def validate_sources(self, sample_ids: list[str], sampling: str = 'random') -> dict[str, int]:
        missing = [sample_id for sample_id in sample_ids if sample_id not in self.id_to_index]
        insufficient = []
        minimum = len(self.ids)
        for sample_id in sample_ids:
            if sample_id not in self.id_to_index:
                continue
            source_index = self.id_to_index[sample_id]
            relaxation = 0 if sampling == 'random' else 2
            count = (
                len(self.candidate_indices(source_index, relaxation=0))
                if sampling == 'random'
                else len(self._candidate_base(source_index, relaxation=relaxation)) - 1
            )
            minimum = min(minimum, count)
            if count < 2:
                insufficient.append((sample_id, count))
        if missing or insufficient:
            raise RuntimeError(
                f'identity bank cannot satisfy j!=k!=i sampling={sampling}; '
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
        self.sample_cache_keys = sample_cache_keys(config)
        self.prompt_cache = self.cache_dir / 'text' / 'prompt.npz'
        self.split = split
        self.head_pose_dropout = float(config.get('training', {}).get('head_pose_dropout', 0.0)) if split == 'train' else 0.0
        self.base_seed = int(config.get('experiment', {}).get('seed', 0))
        spatial = config.get('model', {}).get('spatial_conditions', {})
        self.garment_reference_enabled = bool(spatial.get('garment_reference', {}).get('enabled', False))
        head_cfg = spatial.get('head_control', {})
        self.head_control_enabled = bool(head_cfg.get('enabled', False))
        self.synthetic_head_probability = float(head_cfg.get('synthetic_probability', 0.5)) if split == 'train' else 0.0
        hair_cfg = spatial.get('hair', {})
        self.spatial_hair_enabled = bool(hair_cfg.get('enabled', False))
        adapter_cfg = config.get('model', {}).get('identity_adapter', {})
        self.hair_enabled = bool(adapter_cfg.get('use_hair_tokens', False))
        self.hair_token_mode = str(
            adapter_cfg.get('hair_token_mode', 'dense_with_position')
        ).lower()
        if self.hair_token_mode not in {'dense_with_position', 'semantic_dense'}:
            raise ValueError(f'unknown hair_token_mode={self.hair_token_mode!r}')
        self.semantic_hair_enabled = bool(
            self.hair_enabled and self.hair_token_mode == 'semantic_dense'
        )
        self.hair_loss_target_assets_enabled = bool(
            config.get('training', {}).get('hair_loss', {}).get('enabled', False)
        )
        self.hair_token_assets_enabled = bool(
            self.hair_loss_target_assets_enabled
            or adapter_cfg.get('retain_legacy_hair_parameters_for_resume', False)
            or (self.hair_enabled and not self.semantic_hair_enabled)
        )
        self.hair_reference_enabled = bool(
            hair_cfg.get('reference', {}).get('enabled', False)
        )
        if not 0.0 <= self.synthetic_head_probability <= 1.0:
            raise ValueError(
                f'head synthetic_probability must be in [0,1], got {self.synthetic_head_probability}'
            )
        differential = config.get('training', {}).get('differential', {})
        self.differential_enabled = bool(differential.get('enabled', False)) and split == 'train'
        experiment_name = str(
            config.get('experiment_method', {}).get('name', 'baseline')
        ).lower()
        self.async_flow_enabled = (
            experiment_name in {'c', 'async_flow', 'async'} and split == 'train'
        )
        pair_region = config.get('training', {}).get('paired_region_weighting', {})
        hair_loss = config.get('training', {}).get('hair_loss', {})
        self.paired_region_weighting_enabled = (
            bool(pair_region.get('enabled', False)) and split == 'train'
        )
        self.paired_hair_loss_enabled = (
            bool(hair_loss.get('enabled', False)) and split == 'train'
        )
        self.region_mask_keys = set()
        if self.differential_enabled:
            self.region_mask_keys.update(DIFFERENTIAL_MASK_KEYS)
        if self.async_flow_enabled:
            self.region_mask_keys.update(ASYNC_FLOW_MASK_KEYS)
        if self.paired_region_weighting_enabled or self.paired_hair_loss_enabled:
            self.region_mask_keys.update(PAIRED_REGION_MASK_KEYS)
        self.differential_sampling = str(differential.get('sampling', 'random')).lower()
        self.semihard_pool = int(differential.get('semihard_pool', 8))
        self.region_masks_z_dir = self.root / config['data'].get('region_masks_z_dir', 'derived/region_masks_z')
        self.identity_bank = None
        if self.differential_enabled:
            bank_path = self.root / config['data'].get('identity_bank', 'derived/identity_bank.npz')
            self.identity_bank = IdentityBank(bank_path)
            self.identity_bank.validate_sources(self.ids, sampling=self.differential_sampling)
        if require_coverage:
            self.assert_coverage()

    def __len__(self) -> int:
        return len(self.ids)

    def sample_path(self, sample_id: str) -> Path:
        return self.cache_dir / 'samples' / f'{sample_id}.npz'

    def _sample_shape_errors(self, path: Path, row) -> list[str]:
        errors = []
        latent_keys = ['target_latents', 'pose_latents']
        if self.garment_reference_enabled:
            latent_keys.append('garment_ref_latents')
        if self.hair_reference_enabled:
            latent_keys.append('hair_ref_latents')
        if self.head_control_enabled:
            latent_keys.append('pose_synth_latents')
        for key in latent_keys:
            if key in row.files and np.asarray(row[key]).shape != (self.token_count, 64):
                errors.append(
                    f'{path}: {key} shape={np.asarray(row[key]).shape}, '
                    f'expected {(self.token_count, 64)}'
                )
        if self.hair_token_assets_enabled:
            adapter_cfg = self.config['model']['identity_adapter']
            count = int(adapter_cfg.get('hair_ref_max_tokens', 64))
            dim = int(adapter_cfg.get('hair_ref_dim', 768))
            expected = {
                'hair_ref_tokens': (count, dim),
                'hair_ref_positions': (count, 2),
                'hair_ref_mask': (count,),
            }
            for key, shape in expected.items():
                if key in row.files and np.asarray(row[key]).shape != shape:
                    errors.append(f'{path}: {key} shape={np.asarray(row[key]).shape}, expected {shape}')
            if 'hair_ref_mask' in row.files:
                valid = np.asarray(row['hair_ref_mask'], dtype=np.float32)
                if not np.isfinite(valid).all():
                    errors.append(f'{path}: hair_ref_mask contains non-finite values')
        if self.semantic_hair_enabled:
            adapter_cfg = self.config['model']['identity_adapter']
            count = int(adapter_cfg.get('hair_semantic_max_tokens', 32))
            dim = int(adapter_cfg.get('hair_ref_dim', 768))
            expected = {
                'hair_semantic_tokens': (count, dim),
                'hair_semantic_mask': (count,),
            }
            for key, shape in expected.items():
                if key in row.files and np.asarray(row[key]).shape != shape:
                    errors.append(f'{path}: {key} shape={np.asarray(row[key]).shape}, expected {shape}')
        if self.hair_reference_enabled and 'hair_ref_empty' in row.files:
            empty = np.asarray(row['hair_ref_empty'])
            if empty.shape not in ((), (1,)):
                errors.append(
                    f'{path}: hair_ref_empty shape={empty.shape}, expected scalar'
                )
            elif int(empty.reshape(-1)[0]) not in (0, 1):
                errors.append(f'{path}: hair_ref_empty must be 0 or 1')
        return errors

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
                    absent = [key for key in self.sample_cache_keys if key not in row.files]
                    if absent:
                        bad.append(f'{path}: missing keys {absent}')
                    bad.extend(self._sample_shape_errors(path, row))
            except Exception as exc:  # noqa: BLE001
                bad.append(f'{path}: unreadable cache ({exc})')
            if self.region_mask_keys:
                mask_path = self.region_masks_z_dir / f'{sid}.npz'
                if not mask_path.exists():
                    missing.append(str(mask_path))
                else:
                    try:
                        with np.load(mask_path, mmap_mode='r') as mask_row:
                            absent = [
                                key for key in sorted(self.region_mask_keys)
                                if key not in mask_row.files
                            ]
                            if absent:
                                bad.append(f'{mask_path}: missing keys {absent}')
                            for key in sorted(self.region_mask_keys):
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
            cached = {key: np.asarray(row[key]) for key in self.sample_cache_keys}
        with np.load(self.prompt_cache, mmap_mode='r') as text:
            prompt_embeds = np.asarray(text['prompt_embeds'])
            pooled_prompt_embeds = np.asarray(text['pooled_prompt_embeds'])
        head_pose = torch.from_numpy(cached['head_pose']).float()
        dropped = False
        use_synthetic_head = False
        if self.head_control_enabled:
            control_seed = (
                self.base_seed * 1_000_003
                + epoch * 97_409
                + index * 65_537
                + 31_337
            ) & ((1 << 63) - 1)
            control_rng = np.random.default_rng(np.uint64(control_seed))
            dropped = self.head_pose_dropout > 0.0 and control_rng.random() < self.head_pose_dropout
            use_synthetic_head = control_rng.random() < self.synthetic_head_probability
        elif self.head_pose_dropout > 0.0:
            dropped = bool(torch.rand(()) < self.head_pose_dropout)
        if dropped:
            head_pose = torch.zeros_like(head_pose)
        pose_key = 'pose_synth_latents' if use_synthetic_head else 'pose_latents'
        item = {
            'index': torch.tensor(index, dtype=torch.long),
            'sample_epoch': torch.tensor(epoch, dtype=torch.long),
            'sample_id': sid,
            'target_latents': torch.from_numpy(cached['target_latents']).float(),
            'pose_latents': torch.from_numpy(cached[pose_key]).float(),
            'prompt_embeds': torch.from_numpy(prompt_embeds).float(),
            'pooled_prompt_embeds': torch.from_numpy(pooled_prompt_embeds).float(),
            'pulid_id_embed': torch.from_numpy(cached['pulid_id_embed']).float(),
            'appearance': torch.from_numpy(cached['appearance']).float(),
            'garment': torch.from_numpy(cached['garment_grid']).float(),
            'head_pose': head_pose,
            'head_pose_is_null': torch.tensor(dropped or bool(torch.all(head_pose == 0)), dtype=torch.float32),
            'head_control_is_synthetic': torch.tensor(use_synthetic_head, dtype=torch.float32),
        }
        if self.garment_reference_enabled:
            item['garment_ref_latents'] = torch.from_numpy(cached['garment_ref_latents']).float()
        if self.hair_reference_enabled:
            item['hair_ref_latents'] = torch.from_numpy(
                cached['hair_ref_latents']
            ).float()
            item['hair_ref_empty'] = torch.tensor(
                int(np.asarray(cached['hair_ref_empty']).reshape(-1)[0]),
                dtype=torch.float32,
            )
        if self.hair_token_assets_enabled:
            item.update({
                'hair_ref_tokens': torch.from_numpy(cached['hair_ref_tokens']).float(),
                'hair_ref_positions': torch.from_numpy(cached['hair_ref_positions']).float(),
                'hair_ref_mask': torch.from_numpy(cached['hair_ref_mask']).float(),
            })
        if self.semantic_hair_enabled:
            item.update({
                'hair_semantic_tokens': torch.from_numpy(cached['hair_semantic_tokens']).float(),
                'hair_semantic_mask': torch.from_numpy(cached['hair_semantic_mask']).float(),
            })
        mask_values: dict[str, np.ndarray] = {}
        if self.region_mask_keys:
            with np.load(self.region_masks_z_dir / f'{sid}.npz', mmap_mode='r') as masks:
                mask_values = {
                    key: np.asarray(masks[key])
                    for key in self.region_mask_keys
                }
            item.update({
                key: torch.from_numpy(value).float()
                for key, value in mask_values.items()
            })
        if self.differential_enabled:
            assert self.identity_bank is not None
            sample_seed = (
                self.base_seed * 1_000_003
                + epoch * 97_409
                + index * 65_537
            ) & ((1 << 63) - 1)
            pair = self.identity_bank.sample_pair_details(
                sid,
                sample_seed,
                sampling=self.differential_sampling,
                semihard_pool=self.semihard_pool,
            )
            j_index = int(pair['j_index'])
            k_index = int(pair['k_index'])
            delta_arc = float(pair['delta_arc'])
            j_id = str(self.identity_bank.ids[j_index])
            k_id = str(self.identity_bank.ids[k_index])
            with np.load(self.sample_path(j_id), mmap_mode='r') as j_row:
                j_pulid = np.asarray(j_row['pulid_id_embed'])
                j_appearance = np.asarray(j_row['appearance'])
                j_hair = {
                    key: np.asarray(j_row[key])
                    for key in SPATIAL_HAIR_KEYS
                } if self.hair_token_assets_enabled else {}
                j_semantic_hair = {
                    key: np.asarray(j_row[key])
                    for key in SPATIAL_HAIR_SEMANTIC_KEYS
                } if self.semantic_hair_enabled else {}
                j_hair_reference = {
                    key: np.asarray(j_row[key])
                    for key in SPATIAL_HAIR_REFERENCE_KEYS
                } if self.hair_reference_enabled else {}
            with np.load(self.sample_path(k_id), mmap_mode='r') as k_row:
                k_pulid = np.asarray(k_row['pulid_id_embed'])
                k_appearance = np.asarray(k_row['appearance'])
                k_hair = {
                    key: np.asarray(k_row[key])
                    for key in SPATIAL_HAIR_KEYS
                } if self.hair_token_assets_enabled else {}
                k_semantic_hair = {
                    key: np.asarray(k_row[key])
                    for key in SPATIAL_HAIR_SEMANTIC_KEYS
                } if self.semantic_hair_enabled else {}
                k_hair_reference = {
                    key: np.asarray(k_row[key])
                    for key in SPATIAL_HAIR_REFERENCE_KEYS
                } if self.hair_reference_enabled else {}
            item.update({
                'cf_j_id': j_id,
                'cf_k_id': k_id,
                'cf_j_pulid_id_embed': torch.from_numpy(j_pulid).float(),
                'cf_k_pulid_id_embed': torch.from_numpy(k_pulid).float(),
                'cf_j_appearance': torch.from_numpy(j_appearance).float(),
                'cf_k_appearance': torch.from_numpy(k_appearance).float(),
                'cf_j_train_embed': torch.from_numpy(np.asarray(self.identity_bank.embeds[j_index])).float(),
                'cf_k_train_embed': torch.from_numpy(np.asarray(self.identity_bank.embeds[k_index])).float(),
                'delta_arc_jk': torch.tensor(delta_arc, dtype=torch.float32),
                'cf_sampling_relaxation': torch.tensor(int(pair['relaxation']), dtype=torch.int64),
                'cf_sampling_candidate_count': torch.tensor(int(pair['candidate_count']), dtype=torch.int64),
                'cf_sampling_pool_count': torch.tensor(int(pair['pool_count']), dtype=torch.int64),
            })
            if self.hair_token_assets_enabled:
                item.update({
                    'cf_j_hair_ref_tokens': torch.from_numpy(j_hair['hair_ref_tokens']).float(),
                    'cf_j_hair_ref_positions': torch.from_numpy(j_hair['hair_ref_positions']).float(),
                    'cf_j_hair_ref_mask': torch.from_numpy(j_hair['hair_ref_mask']).float(),
                    'cf_k_hair_ref_tokens': torch.from_numpy(k_hair['hair_ref_tokens']).float(),
                    'cf_k_hair_ref_positions': torch.from_numpy(k_hair['hair_ref_positions']).float(),
                    'cf_k_hair_ref_mask': torch.from_numpy(k_hair['hair_ref_mask']).float(),
                })
            if self.semantic_hair_enabled:
                item.update({
                    'cf_j_hair_semantic_tokens': torch.from_numpy(
                        j_semantic_hair['hair_semantic_tokens']
                    ).float(),
                    'cf_j_hair_semantic_mask': torch.from_numpy(j_semantic_hair['hair_semantic_mask']).float(),
                    'cf_k_hair_semantic_tokens': torch.from_numpy(
                        k_semantic_hair['hair_semantic_tokens']
                    ).float(),
                    'cf_k_hair_semantic_mask': torch.from_numpy(k_semantic_hair['hair_semantic_mask']).float(),
                })
            if self.hair_reference_enabled:
                item.update({
                    'cf_j_hair_ref_latents': torch.from_numpy(
                        j_hair_reference['hair_ref_latents']
                    ).float(),
                    'cf_k_hair_ref_latents': torch.from_numpy(
                        k_hair_reference['hair_ref_latents']
                    ).float(),
                    'cf_j_hair_ref_empty': torch.tensor(
                        int(np.asarray(j_hair_reference['hair_ref_empty']).reshape(-1)[0]),
                        dtype=torch.float32,
                    ),
                    'cf_k_hair_ref_empty': torch.tensor(
                        int(np.asarray(k_hair_reference['hair_ref_empty']).reshape(-1)[0]),
                        dtype=torch.float32,
                    ),
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
