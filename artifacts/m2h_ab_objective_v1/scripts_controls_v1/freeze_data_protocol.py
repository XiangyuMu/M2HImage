"""CPU-only, append-only exploratory protocol freeze; never a biological ID oracle.

Main agent runs this against existing audits. No image/model inference, network,
dependency downloads, original split edits, or destructive operations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

PROTOCOL = 'm2h-layered-data-v1'
GLINT_SHA256 = '4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf'
PAIR_THRESHOLD = 0.3
DHASH_RADIUS = 4
SELECTION_SEED = 20260907


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_group(row):
    """Only strip a view index in the explicitly observed multiimages grammar.

    In particular, singleimage_x2__10158 is an item, not view 10158.
    Unknown grammars remain opaque; this is a provenance proxy, not a SKU/ID.
    """
    key = row['source_key']
    match = re.fullmatch(r'(multiimages_x2__\d+)_\d+', key)
    return row['source_dataset'] + ':' + (match.group(1) if match else key)


def stable_order(value, seed=SELECTION_SEED):
    return hashlib.sha256(f'{seed}:{value}'.encode()).hexdigest()


def duplicate_risk(a, b, image_audit, radius=DHASH_RADIUS):
    # Compare all human/mannequin combinations as a conservative exclusion.
    for left in ('human', 'mannequin'):
        for right in ('human', 'mannequin'):
            x, y = image_audit[a][left], image_audit[b][right]
            if x['sha256'] == y['sha256']:
                return True
            if bin(int(x['dhash'], 16) ^ int(y['dhash'], 16)).count('1') <= radius:
                return True
    return False


def validate_pairs(payload, expected_m=64):
    pairs = payload['pairs']
    by_m = defaultdict(list)
    for row in pairs:
        if not all(isinstance(row[k], str) and row[k] for k in ('mid', 'jid')):
            raise ValueError('mid/jid must be nonempty strings; retain leading zeroes')
        if type(row['seed']) is not int or row['seed'] < 0:
            raise ValueError('seed must be a nonnegative integer')
        by_m[row['mid']].append(row)
    if len(by_m) != expected_m or len(pairs) != 2 * expected_m:
        raise ValueError('Expected exactly two pairs for each M')
    if set(by_m) & {p['jid'] for p in pairs}:
        raise ValueError('M/I file pools overlap')
    for rows in by_m.values():
        if len(rows) != 2 or rows[0]['jid'] == rows[1]['jid']:
            raise ValueError('Each M needs exactly two distinct references')
        if rows[0]['seed'] != rows[1]['seed']:
            raise ValueError('Teacher permutation requires a shared seed per M')
        for i, row in enumerate(rows):
            if row.get('donor_jid', rows[1-i]['jid']) != rows[1-i]['jid']:
                raise ValueError('donor_jid must be the other reference for this M')
    return by_m


def select_dev_pairs(rows, mids, embedding_ids, vectors, image_audit,
                     expected_m=64, max_refs=128):
    """Public API: normalized full-bank vectors -> {'pairs': [...], ...}.

    Raw Glint cosine is computed directly, never inferred from missing top-k
    neighbors. Each chosen pair and its two refs satisfy strict cosine < .3.
    No result/output/teacher metric enters selection. No threshold fallback.
    """
    import numpy as np
    meta = {r['id']: r for r in rows}
    index = {sid: i for i, sid in enumerate(embedding_ids)}
    if len(meta) != len(rows) or len(index) != len(embedding_ids):
        raise ValueError('Duplicate metadata/feature IDs')
    if len(mids) != expected_m or len(set(mids)) != expected_m:
        raise ValueError('Must preserve exactly the pre-existing M IDs')
    if any(s not in index or meta[s]['split'] != 'val' for s in mids):
        raise ValueError('Every M must be old-val with a feature')
    if not 2 <= max_refs <= 128:
        raise ValueError('Reference pool must be bounded to 2..128 IDs')
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) != len(embedding_ids):
        raise ValueError('Feature shape mismatch')
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or not np.allclose(norms, 1, atol=1e-4):
        raise ValueError('Expected finite normalized full-bank vectors')
    vectors = vectors / norms[:, None]
    mset = set(mids)
    mgroups = {source_group(meta[s]) for s in mids}
    candidates = sorted((s for s in meta if meta[s]['split'] == 'val'
                         and s not in mset and s in index
                         and source_group(meta[s]) not in mgroups), key=stable_order)
    # At most 1,932 candidates in the current dataset. Bounded all-M checks.
    candidates = [s for s in candidates
                  if not any(duplicate_risk(s, m, image_audit) for m in mids)]
    if len(candidates) < 2:
        raise ValueError('Too few source/hash-safe old-val references')
    cosine = vectors[[index[m] for m in mids]] @ vectors[[index[j] for j in candidates]].T
    cindex = {s: i for i, s in enumerate(candidates)}
    uses, selected, pairs = Counter(), set(), []
    for i, mid in enumerate(mids):
        eligible = [j for j in candidates if cosine[i, cindex[j]] < PAIR_THRESHOLD]
        # Reuse the frozen pool before extending it; count balances reuse.
        ordered = sorted(eligible, key=lambda j: (j not in selected, uses[j], stable_order(mid+':'+j)))
        chosen = None
        for a in ordered:
            if a not in selected and len(selected) >= max_refs:
                continue
            if a not in selected and any(source_group(meta[a]) == source_group(meta[s])
                                         or duplicate_risk(a, s, image_audit) for s in selected):
                continue
            for b in ordered:
                if a == b or len(selected | {a, b}) > max_refs:
                    continue
                if source_group(meta[a]) == source_group(meta[b]) or duplicate_risk(a, b, image_audit):
                    continue
                if b not in selected and any(source_group(meta[b]) == source_group(meta[s])
                                             or duplicate_risk(b, s, image_audit) for s in selected):
                    continue
                ref_cosine = float(vectors[index[a]] @ vectors[index[b]])
                if ref_cosine < PAIR_THRESHOLD:
                    chosen = (a, b, ref_cosine)
                    break
            if chosen:
                break
        if chosen is None:
            raise ValueError(f'No admissible two-ref allocation for M={mid}; no threshold relaxation')
        a, b, ref_cosine = chosen
        for jid, donor in ((a, b), (b, a)):
            pairs.append({'mid': mid, 'jid': jid, 'seed': 0, 'donor_jid': donor,
                          'raw_glint_cosine': float(cosine[i, cindex[jid]]),
                          'ref_ref_raw_glint_cosine': ref_cosine,
                          'm_source_group': source_group(meta[mid]),
                          'j_source_group': source_group(meta[jid]),
                          'm_sha256': image_audit[mid]['mannequin']['sha256'],
                          'j_sha256': image_audit[jid]['human']['sha256']})
            uses[jid] += 1
            selected.add(jid)
    payload = {'schema_version': PROTOCOL, 'pairs': pairs,
               'purpose': 'exploratory_dev128_not_formal_holdout',
               'formal_split_ready': False, 'selection_seed': SELECTION_SEED,
               'thresholds': {'raw_glint_cosine_strict_lt': PAIR_THRESHOLD,
                              'dhash_hamming_exclude_lte': DHASH_RADIUS,
                              'calibration': 'preregistered engineering proxies; uncalibrated biological identity'},
               'mids': list(mids), 'reference_ids': sorted(selected),
               'reference_use_counts': dict(sorted(uses.items())),
               'eligible_reference_candidates': len(candidates),
               'm_unique_source_groups': len(mgroups),
               'pair_max_raw_glint_cosine': max(p['raw_glint_cosine'] for p in pairs),
               'identity_scope': 'H_i raw Glint versus H_j; H_i is selection-only audit data, never inference input'}
    validate_pairs(payload, expected_m)
    return payload


def load_inputs(root, artifacts):
    import numpy as np
    manifest = root / 'metadata/manifest.csv'
    with manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    meta = {r['id']: r for r in rows}
    if len(meta) != len(rows) or not rows:
        raise ValueError('Empty/duplicate metadata IDs')
    splits = {s: set((root / f'splits/{s}.txt').read_text().split()) for s in ('train', 'val', 'test')}
    if any(splits[s] != {r['id'] for r in rows if r['split'] == s} for s in splits):
        raise ValueError('Original split/manifest mismatch')
    if set.union(*splits.values()) != set(meta):
        raise ValueError('Unknown original split')
    features = artifacts / 'identity_features_full'
    audit = artifacts / 'full_duplicates'
    feature_manifest = json.loads((features / 'manifest.json').read_text())
    if feature_manifest['manifest_sha256'] != sha256(manifest):
        raise ValueError('Feature/metadata digest mismatch')
    if feature_manifest['models'].get('glintr100.onnx') != GLINT_SHA256:
        raise ValueError('Unexpected grouping recognizer; final evaluator is forbidden')
    summary = json.loads((audit / 'summary.json').read_text())
    if summary['expected'] != len(rows)*2 or summary['completed'] != len(rows)*2 or summary['failed']:
        raise ValueError('Incomplete all-role image audit')
    image_audit = defaultdict(dict)
    with (audit / 'per_image.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            sid, role = r['id'], r['role']
            if sid not in meta or role not in ('human', 'mannequin') or role in image_audit[sid]:
                raise ValueError('Unexpected/duplicate image audit key')
            if r['status'] != 'ok' or r['path'] != meta[sid][role+'_path'] or r['split'] != meta[sid]['split']:
                raise ValueError('Invalid/mismatched audited image')
            if not re.fullmatch('[0-9a-f]{64}', r['sha256']) or not re.fullmatch('[0-9a-f]{16}', r['dhash']):
                raise ValueError('Malformed image digest')
            image_audit[sid][role] = r
    if set(image_audit) != set(meta) or any(len(v) != 2 for v in image_audit.values()):
        raise ValueError('Image audit lacks full human/mannequin coverage')
    if not (features / 'FEATURES_READY').is_file():
        raise ValueError('Missing FEATURES_READY')
    feature_summary = json.loads((features / 'summary.json').read_text())
    if feature_summary['ok'] != len(rows) or feature_summary['failed'] or feature_summary['completed'] != len(rows):
        raise ValueError('Incomplete feature coverage')
    checked = set()
    with (features / 'per_image.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            sid = r['id']
            if sid in checked or sid not in meta or r['status'] != 'ok':
                raise ValueError('Invalid feature audit coverage')
            if r['input_sha256'] != image_audit[sid]['human']['sha256']:
                raise ValueError('Feature/image audit digest mismatch')
            checked.add(sid)
    ids, arrays = [], []
    shards = sorted(features.glob('embeddings_*.npz'))
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            ids.extend(z['ids'].tolist())
            arrays.append(z['embeds'])
    if len(ids) != len(set(ids)) or set(ids) != set(meta) or checked != set(meta):
        raise ValueError('Full-bank ID coverage mismatch')
    vectors = np.concatenate(arrays).astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1)
    if vectors.shape != (len(rows), 512) or not np.isfinite(vectors).all() or not np.allclose(norms, 1, atol=1e-4):
        raise ValueError('Invalid normalized Glint bank')
    vectors /= norms[:, None]
    mids_path = artifacts / 'p0_metadata_20260907/old_val_probe_ids.json'
    mids = json.loads(mids_path.read_text())
    paths = [manifest, mids_path, features/'manifest.json', features/'summary.json',
             features/'per_image.jsonl', audit/'summary.json', audit/'per_image.jsonl',
             audit/'exact_groups.json', audit/'dhash_equal_candidates.json', *shards,
             *(root / f'splits/{s}.txt' for s in splits)]
    provenance = {str(p): sha256(p) for p in paths}
    provenance[str(Path(__file__).resolve())] = sha256(__file__)
    return rows, mids, ids, vectors, image_audit, provenance


def write_json(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('--out must not exist; choose a new version, never overwrite a freeze')
    if args.out.resolve().is_relative_to(args.root.resolve()):
        parser.error('--out must be outside the dataset')
    rows, mids, ids, vectors, images, provenance = load_inputs(args.root, args.artifacts)
    payload = select_dev_pairs(rows, mids, ids, vectors, images)
    payload['provenance_sha256'] = provenance
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out/'dev128_pairs.json', payload)
    status = {'exploratory_dev128_ready': True, 'formal_split_ready': False,
              'clean_retraining_ready': False, 'pairs': len(payload['pairs']),
              'm_count': len(mids), 'reference_count': len(payload['reference_ids']),
              'pair_max_raw_glint_cosine': payload['pair_max_raw_glint_cosine'],
              'dev128_sha256': sha256(args.out/'dev128_pairs.json'),
              'open': ['formal source/identity grouping and bridge isolation',
                       'uncalibrated grouping thresholds and nonzero-Hamming near duplicates outside dev selection',
                       'formal train/dev/final manifests and clean retraining',
                       'input-only cache provenance and final-real availability']}
    write_json(args.out/'summary.json', status)
    with (args.out/'DEV128_READY').open('x') as f:
        f.write(status['dev128_sha256']+'\n')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
