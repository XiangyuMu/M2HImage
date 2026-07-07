from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

from conditions import read_ids

CACHE_KEYS = (
    'target_latents', 'pose_latents', 'prompt_embeds', 'pooled_prompt_embeds',
    'identity', 'appearance', 'garment_grid', 'head_pose'
)


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
        for sid in self.ids:
            path = self.sample_path(sid)
            if not path.exists():
                missing.append(str(path))
        if not self.prompt_cache.exists():
            missing.append(str(self.prompt_cache))
        if missing:
            preview = '\n'.join(missing[:20])
            raise RuntimeError(f'Phase 1 cache coverage is not 100%; missing {len(missing)} files, first entries:\n{preview}')

    def __getitem__(self, index: int) -> dict[str, Any]:
        sid = self.ids[index]
        row = np.load(self.sample_path(sid), mmap_mode='r')
        text = np.load(self.prompt_cache, mmap_mode='r')
        head_pose = torch.from_numpy(np.asarray(row['head_pose'])).float()
        dropped = False
        if self.head_pose_dropout > 0.0 and torch.rand(()) < self.head_pose_dropout:
            head_pose = torch.zeros_like(head_pose)
            dropped = True
        garment_key = 'garment_grid' if 'garment_grid' in row.files else 'garment'
        item = {
            'index': torch.tensor(index, dtype=torch.long),
            'sample_id': sid,
            'target_latents': torch.from_numpy(np.asarray(row['target_latents'])).float(),
            'pose_latents': torch.from_numpy(np.asarray(row['pose_latents'])).float(),
            'prompt_embeds': torch.from_numpy(np.asarray(text['prompt_embeds'])).float(),
            'pooled_prompt_embeds': torch.from_numpy(np.asarray(text['pooled_prompt_embeds'])).float(),
            'identity': torch.from_numpy(np.asarray(row['identity'])).float(),
            'appearance': torch.from_numpy(np.asarray(row['appearance'])).float(),
            'garment': torch.from_numpy(np.asarray(row[garment_key])).float(),
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
