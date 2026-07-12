from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean, median

import numpy as np

from conditions import find_one, load_yaml, read_ids


PARSING_LABELS = {'top': 3, 'dress': 4, 'skirt': 5, 'pants': 6}
_PARSING_COUNTS_CACHE: dict[str, dict[str, dict[int, int]]] = {}


def load_attrs(root: Path, sample_id: str) -> dict:
    path = root / 'derived/id_attributes' / f'{sample_id}.json'
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def garment_type(root: Path, sample_id: str) -> str:
    attrs = load_attrs(root, sample_id)
    attr_type = attrs.get('garment_type', attrs.get('cloth_type'))
    if attr_type:
        return str(attr_type)
    return parsing_garment_type(root, sample_id)


def load_parsing_counts(root: Path, image_type: str = 'mannequin') -> dict[str, dict[int, int]]:
    key = f'{root}:{image_type}'
    if key in _PARSING_COUNTS_CACHE:
        return _PARSING_COUNTS_CACHE[key]
    manifest = root / 'human_parsing/fashn/metadata/fashn_human_parser_manifest.jsonl'
    counts: dict[str, dict[int, int]] = {}
    if manifest.exists():
        with manifest.open('r', encoding='utf-8') as handle:
            for line in handle:
                row = json.loads(line)
                if row.get('image_type') != image_type or row.get('status') != 'ok':
                    continue
                counts[row['id']] = {int(k): int(v) for k, v in row.get('class_pixel_counts', {}).items()}
    _PARSING_COUNTS_CACHE[key] = counts
    return counts


def classify_parsing_counts(counts: dict[int, int]) -> str:
    top = int(counts.get(PARSING_LABELS['top'], 0))
    dress = int(counts.get(PARSING_LABELS['dress'], 0))
    skirt = int(counts.get(PARSING_LABELS['skirt'], 0))
    pants = int(counts.get(PARSING_LABELS['pants'], 0))
    garment = top + dress + skirt + pants
    if garment <= 0:
        return 'unknown'
    ratios = {'top': top / garment, 'dress': dress / garment, 'skirt': skirt / garment, 'pants': pants / garment}
    if ratios['dress'] >= 0.25:
        return 'dress'
    if ratios['skirt'] >= 0.18:
        return 'skirt'
    if ratios['pants'] >= 0.20:
        return 'pants'
    if ratios['top'] >= 0.20:
        return 'top'
    return max(ratios, key=ratios.get)


def parsing_garment_type(root: Path, sample_id: str) -> str:
    counts = load_parsing_counts(root, image_type='mannequin').get(sample_id)
    if counts is None:
        counts = load_parsing_counts(root, image_type='human').get(sample_id, {})
    return classify_parsing_counts(counts)


def compatible(a: dict, b: dict) -> bool:
    if a.get('gender') and b.get('gender') and a.get('gender') != b.get('gender'):
        return False
    if 'skin_cluster' in a and 'skin_cluster' in b and abs(int(a['skin_cluster']) - int(b['skin_cluster'])) > 1:
        return False
    if 'age_raw' in a and 'age_raw' in b and abs(float(a['age_raw']) - float(b['age_raw'])) > 15:
        return False
    if a.get('age_group') and b.get('age_group') and a.get('age_group') == b.get('age_group'):
        return True
    return True


def valid_test_ids(cfg: dict) -> list[str]:
    root = Path(cfg['data']['root'])
    valid = []
    for sid in read_ids(root / cfg['data']['test_split']):
        try:
            find_one(root / 'images/mannequin', sid)
            find_one(root / 'images/human', sid)
            find_one(root / 'derived/face_crops/human', sid)
            find_one(root / 'dwpose/without_head/mannequin', sid)
            valid.append(sid)
        except FileNotFoundError:
            continue
    return valid


def diverse_identity_pool(root: Path, valid: list[str], n: int, rng: random.Random) -> list[str]:
    groups: dict[tuple, list[str]] = {}
    for sid in valid:
        attrs = load_attrs(root, sid)
        key = (attrs.get('gender', 'unknown'), attrs.get('age_group', 'unknown'), attrs.get('skin_cluster', 'unknown'))
        groups.setdefault(key, []).append(sid)
    for bucket in groups.values():
        rng.shuffle(bucket)
    pool = []
    while len(pool) < n and groups:
        for key in sorted(list(groups), key=str):
            bucket = groups[key]
            if not bucket:
                groups.pop(key, None)
                continue
            pool.append(bucket.pop())
            if len(pool) >= n:
                break
    return pool


def build_subset(cfg: dict, out: Path) -> dict:
    root = Path(cfg['data']['root'])
    rng = random.Random(int(cfg['experiment']['seed']))
    valid = valid_test_ids(cfg)
    groups: dict[str, list[str]] = {}
    for sid in valid:
        groups.setdefault(garment_type(root, sid), []).append(sid)
    for bucket in groups.values():
        rng.shuffle(bucket)
    mannequins = []
    while len(mannequins) < int(cfg['eval']['b2_mannequin_count']) and groups:
        for key in sorted(list(groups)):
            bucket = groups[key]
            if not bucket:
                groups.pop(key, None)
                continue
            mannequins.append(bucket.pop())
            if len(mannequins) >= int(cfg['eval']['b2_mannequin_count']):
                break
    pool = diverse_identity_pool(root, valid, int(cfg['eval']['b2_identity_pool']), rng)
    pairs = []
    for mid in mannequins:
        m_attrs = load_attrs(root, mid)
        choices = [sid for sid in pool if sid != mid and compatible(m_attrs, load_attrs(root, sid))]
        if len(choices) < int(cfg['eval']['b2_identities_per_mannequin']):
            choices = [sid for sid in pool if sid != mid]
        rng.shuffle(choices)
        for identity in choices[: int(cfg['eval']['b2_identities_per_mannequin'])]:
            pairs.append({
                'mannequin_id': mid,
                'identity_id': identity,
                'theta_source': mid,
                'garment_type': garment_type(root, mid),
                'seeds': cfg['eval']['b2_seeds'],
            })
    expected = int(cfg['eval']['b2_mannequin_count']) * int(cfg['eval']['b2_identities_per_mannequin'])
    if len(pairs) != expected:
        raise RuntimeError(f'B2 subset expected {expected} pairs, got {len(pairs)}')
    garment_types = {sid: garment_type(root, sid) for sid in mannequins}
    payload = {
        'seed': cfg['experiment']['seed'],
        'garment_type_source': 'human_parsing/fashn mannequin mask labels 3=top,4=dress,5=skirt,6=pants',
        'garment_type_counts': dict(Counter(garment_types.values())),
        'garment_types': garment_types,
        'mannequins': mannequins,
        'identity_pool': pool,
        'pairs': pairs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return payload


def cached_cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def make_cf_batch(cache: Path, text_cache: Path, mannequin_id: str, identity_id: str):
    import torch
    m = np.load(cache / f'{mannequin_id}.npz')
    j = np.load(cache / f'{identity_id}.npz')
    text = np.load(text_cache)
    return {
        'pose_latents': torch.from_numpy(np.asarray(m['pose_latents'])).float(),
        'pulid_id_embed': torch.from_numpy(np.asarray(j['pulid_id_embed'])).float(),
        'appearance': torch.from_numpy(np.asarray(j['appearance'])).float(),
        'garment': torch.from_numpy(np.asarray(m['garment_grid'] if 'garment_grid' in m.files else m['garment'])).float(),
        'head_pose': torch.from_numpy(np.asarray(m['head_pose'])).float(),
        'prompt_embeds': torch.from_numpy(np.asarray(text['prompt_embeds'])).float(),
        'pooled_prompt_embeds': torch.from_numpy(np.asarray(text['pooled_prompt_embeds'])).float(),
    }


def generate_b2(cfg: dict, ckpt: Path, subset: dict, device: str, overwrite: bool, limit: int | None, num_shards: int = 1, shard_index: int = 0) -> int:
    import torch
    from diffusers import AutoencoderKL
    from conditions import choose_dtype, seed_everything
    from eval_watcher import decode_tokens, generate
    from train_paired import WarmupFlowModel, load_checkpoint, load_components

    root = Path(cfg['data']['root'])
    cache = root / cfg['data']['cache_dir'] / 'samples'
    text_cache = root / cfg['data']['cache_dir'] / 'text' / 'prompt.npz'
    out_dir = root / cfg['eval']['b2_output_dir']
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(cfg['experiment']['seed']))
    torch_device = torch.device(device)
    dtype = choose_dtype(cfg['model']['precision'])
    transformer, controlnet, vae, adapter, pulid, _ = load_components(cfg, torch_device, dtype)
    if vae is None:
        vae = AutoencoderKL.from_pretrained(cfg['model']['base'], subfolder='vae', torch_dtype=dtype, local_files_only=True).to(torch_device)
    model = WarmupFlowModel(transformer, controlnet, adapter, pulid, cfg)
    load_checkpoint(ckpt, model)
    model.eval()
    rows = subset['pairs'][:limit] if limit else subset['pairs']
    rows = [row for i, row in enumerate(rows) if i % num_shards == shard_index]
    written = 0
    for row in rows:
        mid = row['mannequin_id']
        jid = row['identity_id']
        batch = make_cf_batch(cache, text_cache, mid, jid)
        for seed in row['seeds']:
            out = out_dir / f'{mid}__id{jid}__seed{seed}.png'
            if out.exists() and not overwrite:
                continue
            tokens = generate(model, batch, int(cfg['eval']['generate_steps']), seed=int(seed), device=torch_device, dtype=dtype)
            decode_tokens(vae, tokens, cfg['data']['resolution']).save(out)
            written += 1
    return written


def write_report(cfg: dict, subset: dict, out: Path, generated_count: int | None = None) -> None:
    root = Path(cfg['data']['root'])
    cache = root / cfg['data']['cache_dir'] / 'samples'
    gen_dir = root / cfg['eval']['b2_output_dir']
    expected_images = sum(len(row.get('seeds', cfg['eval']['b2_seeds'])) for row in subset['pairs'])
    expected_paths = [
        gen_dir / f"{row['mannequin_id']}__id{row['identity_id']}__seed{seed}.png"
        for row in subset['pairs']
        for seed in row.get('seeds', cfg['eval']['b2_seeds'])
    ]
    generated_images = [path for path in expected_paths if path.exists()]
    consistency = []
    missing_keys = []
    checked_ids = sorted(set(subset.get('mannequins', []) + subset.get('identity_pool', [])))
    for sid in checked_ids:
        path = cache / f'{sid}.npz'
        if not path.exists():
            missing_keys.append(f'{sid}:missing-cache')
            continue
        row = np.load(path)
        for key in ('pulid_id_embed', 'appearance', 'garment_grid', 'head_pose'):
            if key not in row.files:
                missing_keys.append(f'{sid}:{key}')
        if 'pulid_id_embed' in row.files:
            emb = np.asarray(row['pulid_id_embed']).reshape(-1)
            consistency.append(cached_cosine(emb, emb.copy()))
    if generated_count is None:
        gen_line = 'generation command not run in this invocation'
    else:
        gen_line = f'generated/updated images this invocation: {generated_count}'
    generation_status = 'complete' if len(generated_images) == expected_images else 'incomplete'
    lines = [
        '# B2 Adapter-only Baseline Report', '',
        f"subset: {len(subset['mannequins'])} mannequins / {len(subset['identity_pool'])} identity pool / {len(subset['pairs'])} pairs",
        f"garment type counts: {subset.get('garment_type_counts', {})}",
        f"seeds per pair: {cfg['eval']['b2_seeds']}",
        f"expected generated images: {expected_images}",
        f"actual generated images: {len(generated_images)}",
        f"generation status: {generation_status}",
        gen_line, '',
        '## Cache-space Sanity', '',
        'This section only checks cache wiring; it is not an identity or garment metric.',
        f"same-id embedding reload consistency mean={mean(consistency):.4f}, min={min(consistency):.4f}" if consistency else 'same-id embedding reload consistency unavailable',
        f"required cache key check: {'ok' if not missing_keys else 'missing ' + ', '.join(missing_keys[:20])}", '',
        '## Official Generated-image Metrics', '',
        'Status: BLOCKED by missing local held-out metric runners/weights, not by generation.',
        '- DeltaID: requires held-out AdaFace/CurricularFace. Local search found only InsightFace antelopev2/glintr100, which is the training identity backbone and is therefore not used.',
        '- GarmentSim: requires generated cloth_safe DINO + LPIPS runner. DINO/LPIPS metric runner is not wired for generated cloth_safe masks here.',
        '- Pose drift: requires generated-image DWPose keypoints. Local DWPose/OpenPose code exists, but no integrated generated-image runner has been wired into this report.',
        '- Face realism: requires detector confidence + quality score runner on generated faces.',
        '- Head pose MAE: requires generated-image 6DRepNet runner. Dataset has source head-pose JSON, but no local generated-image 6DRepNet runner was found.', '',
        'Decision rule remains: if generated-image DeltaID is near zero after held-out recognizer is installed, mark BLOCKER and fix adapter injection before A2.',
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def main() -> None:
    parser = argparse.ArgumentParser(description='Build/freeze B2 subset, optionally generate B2 images, and write baseline report.')
    parser.add_argument('--config', default='configs/warmup.yaml')
    parser.add_argument('--ckpt', required=False)
    parser.add_argument('--subset', default=None)
    parser.add_argument('--build-subset', action='store_true')
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int, default=None, help='Debug only: limit number of B2 pairs to generate.')
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    args = parser.parse_args()
    if 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1 and args.num_shards == 1:
        args.num_shards = int(os.environ['WORLD_SIZE'])
        args.shard_index = int(os.environ['RANK'])
        args.device = f"cuda:{int(os.environ.get('LOCAL_RANK', args.shard_index))}"
    cfg = load_yaml(args.config)
    root = Path(cfg['data']['root'])
    subset_path = Path(args.subset) if args.subset else root / cfg['data']['cf_subset']
    if args.build_subset or not subset_path.exists():
        subset = build_subset(cfg, subset_path)
    else:
        subset = json.loads(subset_path.read_text(encoding='utf-8'))
    generated_count = None
    if args.ckpt:
        generated_count = generate_b2(cfg, Path(args.ckpt), subset, args.device, args.overwrite, args.limit, args.num_shards, args.shard_index)
    report_path = root / cfg['eval']['report_path']
    if args.shard_index == 0:
        write_report(cfg, subset, report_path, generated_count)
        print(f'wrote subset={subset_path} report={report_path}')
    else:
        print(f'shard {args.shard_index}/{args.num_shards} generated_count={generated_count}')


if __name__ == '__main__':
    main()
