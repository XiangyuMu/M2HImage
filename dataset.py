from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

from conditions import read_ids

SAMPLE_CACHE_KEYS = (
    'target_latents', 'pose_latents', 'pulid_id_embed', 'appearance', 'garment_grid', 'head_pose'
)
TEXT_CACHE_KEYS = ('prompt_embeds', 'pooled_prompt_embeds')


class PairedWarmupDataset(Dataset):
    def __init__(self, config: dict[str, Any], split: str = 'train', require_coverage: bool = True) -> None:
        self.config = config
        self.root = Path(config['data']['root'])
        split_key = f'{split}_split'
        self.ids = read_ids(self.root / config['data'][split_key])
        self.cache_dir = self.root / config['data']['cache_dir']
        self.prompt_cache = self.cache_dir / 'text' / 'prompt.npz'
        self.split = split
        self.head_pose_dropout = float(config.get('training', {}).get('head_pose_dropout', 0.0)) if split == 'train' else 0.0
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
        sid = self.ids[index]
        row = np.load(self.sample_path(sid), mmap_mode='r')
        text = np.load(self.prompt_cache, mmap_mode='r')
        head_pose = torch.from_numpy(np.asarray(row['head_pose'])).float()
        dropped = False
        if self.head_pose_dropout > 0.0 and torch.rand(()) < self.head_pose_dropout:
            head_pose = torch.zeros_like(head_pose)
            dropped = True
        item = {
            'index': torch.tensor(index, dtype=torch.long),
            'sample_id': sid,
            'target_latents': torch.from_numpy(np.asarray(row['target_latents'])).float(),
            'pose_latents': torch.from_numpy(np.asarray(row['pose_latents'])).float(),
            'prompt_embeds': torch.from_numpy(np.asarray(text['prompt_embeds'])).float(),
            'pooled_prompt_embeds': torch.from_numpy(np.asarray(text['pooled_prompt_embeds'])).float(),
            'pulid_id_embed': torch.from_numpy(np.asarray(row['pulid_id_embed'])).float(),
            'appearance': torch.from_numpy(np.asarray(row['appearance'])).float(),
            'garment': torch.from_numpy(np.asarray(row['garment_grid'])).float(),
            'head_pose': head_pose,
            'head_pose_is_null': torch.tensor(dropped or bool(torch.all(head_pose == 0)), dtype=torch.float32),
        }
        return item


class ResumeDistributedSampler(DistributedSampler):
    def state_dict(self) -> dict[str, int]:
        return {'epoch': int(self.epoch)}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.epoch = int(state.get('epoch', 0))


def write_cache_manifest(cache_dir: str | Path, payload: dict[str, Any]) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / 'manifest.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
