"""CPU-only fixed-threshold grouping evidence and non-destructive candidates.

Never a biological identity oracle, formal split freeze, or dev128 selector.
Only writes a fresh --out directory; inputs and original splits are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import time

THRESHOLD = 0.9
TOP_K = 16
SEED = 20260907
GLINT = '4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_group(row):
    key = row['source_key']
    match = re.fullmatch(r'(multiimages_x2__\d+)_\d+', key)
    return row['source_dataset'] + ':' + (match.group(1) if match else key)


def legacy_source_group(row):
    key = row['source_key']
    stem, sep, suffix = key.rpartition('_')
    return row['source_dataset'] + ':' + (stem if sep and suffix.isdigit() else key)


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        i, j = self.find(i), self.find(j)
        if i == j:
            return
        if self.size[i] < self.size[j]:
            i, j = j, i
        self.parent[j] = i
        self.size[i] += self.size[j]

    def groups(self):
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            groups[self.find(i)].append(i)
        return list(groups.values())


def group_id(members, ids):
    return hashlib.sha256('\n'.join(sorted(ids[i] for i in members)).encode()).hexdigest()


def stats(groups, rows):
    sizes = Counter(map(len, groups))
    cross = [g for g in groups if len({rows[i]['split'] for i in g}) > 1]
    patterns = Counter('|'.join(sorted({rows[i]['split'] for i in g})) for g in cross)
    return {'components': len(groups), 'singletons': sizes[1],
            'non_singleton_components': len(groups) - sizes[1],
            'largest_component': max(map(len, groups)),
            'size_histogram': dict(sorted(sizes.items())),
            'cross_original_split_components': len(cross),
            'records_in_cross_original_split_components': sum(map(len, cross)),
            'cross_original_split_patterns': dict(patterns)}


def write_json(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def write_csv(path, fields, records):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def self_test():
    def row(key, dataset='a'):
        return {'source_key': key, 'source_dataset': dataset}
    require(source_group(row('singleimage_x2__10158')) == 'a:singleimage_x2__10158', 'single key lost')
    require(source_group(row('singleimage_x2__10159')) != source_group(row('singleimage_x2__10158')), 'single collapse')
    require(legacy_source_group(row('singleimage_x2__10159')) == legacy_source_group(row('singleimage_x2__10158')), 'legacy reproduction')
    require(source_group(row('multiimages_x2__00001_1')) == source_group(row('multiimages_x2__00001_2')), 'view union')
    for key in ('train__14684_00', 'test__00006_00', 'unknown_123', 'multiimages_x2__a_1'):
        require(source_group(row(key)) == 'a:' + key, 'opaque key changed')
    require(source_group(row('singleimage_x2__10158', 'b')) != source_group(row('singleimage_x2__10158')), 'namespace lost')
    uf = UnionFind(4)
    uf.union(0, 1)
    uf.union(2, 3)
    uf.union(1, 2)
    require(sorted(map(len, uf.groups())) == [4], 'transitive union failed')
    print('SELF_TEST_PASS: singleimage regression, multi-view, opaque namespaces, transitive components', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path)
    p.add_argument('--artifacts', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    self_test()
    if args.self_test:
        return
    require(all((args.root, args.artifacts, args.out)), 'root/artifacts/out required')
    require(not args.out.exists(), 'Output must be fresh; never overwrite')
    require(not args.out.resolve().is_relative_to(args.root.resolve()), 'Cannot write inside dataset')
    import numpy as np
    start = time.monotonic()
    root, art = args.root, args.artifacts
    manifest = root / 'metadata/manifest.csv'
    with manifest.open(newline='') as f:
        metadata = list(csv.DictReader(f))
    meta = {r['id']: r for r in metadata}
    require(meta and len(meta) == len(metadata), 'Empty or duplicate metadata IDs')
    input_paths = [manifest, Path(__file__).resolve()]
    original_counts = {}
    for split in ('train', 'val', 'test'):
        path = root / f'splits/{split}.txt'
        split_ids = path.read_text().split()
        require(len(split_ids) == len(set(split_ids)), 'Duplicate split IDs')
        require(set(split_ids) == {sid for sid, r in meta.items() if r['split'] == split}, 'Split/metadata mismatch')
        original_counts[split] = len(split_ids)
        input_paths.append(path)
    require(sum(original_counts.values()) == len(meta), 'Unknown split')
    before_original = {str(x): sha256(x) for x in [manifest, *(root / f'splits/{s}.txt' for s in original_counts)]}
    feature_dir = art / 'identity_features_full'
    audit_dir = art / 'full_duplicates'
    neighbor_dir = art / 'identity_neighbors_full'
    fm = json.loads((feature_dir / 'manifest.json').read_text())
    fs = json.loads((feature_dir / 'summary.json').read_text())
    require(fm['manifest_sha256'] == sha256(manifest) and fm['models']['glintr100.onnx'] == GLINT, 'Feature provenance mismatch')
    require((feature_dir / 'FEATURES_READY').is_file(), 'Incomplete features')
    require(fs['ok'] == len(meta) and fs['completed'] == len(meta) and not fs['failed'], 'Feature coverage failed')
    neighbor_path = neighbor_dir / 'neighbors.npz'
    with np.load(neighbor_path, allow_pickle=False) as z:
        ids, v, j = z['ids'].tolist(), z['cosine'].copy(), z['neighbor_index'].copy()
    n = len(ids)
    require(len(set(ids)) == n and set(ids) == set(meta), 'Neighbor IDs mismatch')
    require(v.shape == j.shape == (n, TOP_K), 'Expected top16 bank')
    require(np.issubdtype(j.dtype, np.integer) and j.min() >= 0 and j.max() < n, 'Invalid neighbor index')
    require(np.isfinite(v).all() and np.abs(v).max() <= 1.00001, 'Invalid cosine')
    require((np.diff(v, axis=1) <= 1e-7).all(), 'Neighbor values not descending')
    require(not (j == np.arange(n)[:, None]).any(), 'Self neighbors')
    require(all(len(set(a.tolist())) == TOP_K for a in j), 'Duplicate neighbors')
    rows = [meta[sid] for sid in ids]
    index = {sid: i for i, sid in enumerate(ids)}
    image_rows, sha_members = {}, defaultdict(list)
    audit_summary = json.loads((audit_dir / 'summary.json').read_text())
    require(audit_summary['expected'] == audit_summary['completed'] == n * 2 and not audit_summary['failed'], 'Incomplete image audit')
    with (audit_dir / 'per_image.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            sid, role = r['id'], r['role']
            require(sid in meta and role in ('human', 'mannequin') and (sid, role) not in image_rows, 'Invalid image audit key')
            require(r['status'] == 'ok' and r['split'] == meta[sid]['split'] and r['path'] == meta[sid][role + '_path'], 'Image audit mismatch')
            require(re.fullmatch('[0-9a-f]{64}', r['sha256']), 'Invalid image hash')
            image_rows[sid, role] = r
            sha_members[r['sha256']].append((index[sid], role))
    require(len(image_rows) == n * 2, 'Missing roles')
    feature_seen = set()
    with (feature_dir / 'per_image.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            require(r['id'] in meta and r['id'] not in feature_seen and r['status'] == 'ok', 'Feature audit mismatch')
            require(r['input_sha256'] == image_rows[r['id'], 'human']['sha256'], 'Feature/input digest mismatch')
            feature_seen.add(r['id'])
    require(feature_seen == set(ids), 'Feature audit IDs incomplete')
    shards = sorted(feature_dir.glob('embeddings_*.npz'))
    bank_ids, bank = [], []
    for path in shards:
        with np.load(path, allow_pickle=False) as z:
            bank_ids.extend(z['ids'].tolist())
            bank.append(z['embeds'])
    require(bank_ids == ids, 'Neighbor/bank order mismatch')
    x = np.concatenate(bank).astype(np.float64)
    norms = np.linalg.norm(x, axis=1)
    require(x.shape == (n, 512) and np.isfinite(x).all() and np.allclose(norms, 1, atol=1e-4), 'Invalid embedding bank')
    x /= norms[:, None]
    max_error = 0.0
    for lo in range(0, n, 256):
        hi = min(n, lo + 256)
        direct = np.einsum('nd,nkd->nk', x[lo:hi], x[j[lo:hi]])
        max_error = max(max_error, float(np.max(np.abs(direct - v[lo:hi]))))
    require(max_error < 1e-5, 'Cached neighbor cosine/bank mismatch')
    input_paths += [feature_dir / name for name in ('manifest.json', 'summary.json', 'per_image.jsonl', 'FEATURES_READY')]
    input_paths += [audit_dir / name for name in ('summary.json', 'per_image.jsonl', 'exact_groups.json')]
    input_paths += [neighbor_path, neighbor_dir / 'summary.json', *shards]
    provenance = {str(path): sha256(path) for path in input_paths}
    print(json.dumps({'inputs_validated': n, 'neighbor_cosine_max_abs_error': max_error}), flush=True)
    args.out.mkdir(parents=True, exist_ok=False)
    out = args.out
    source, old = defaultdict(list), defaultdict(list)
    for i, r in enumerate(rows):
        source[source_group(r)].append(i)
        old[legacy_source_group(r)].append(i)
    exact = [sorted({i for i, _ in members}) for members in sha_members.values() if len({i for i, _ in members}) > 1]
    names = ('legacy_source', 'corrected_source', 'exact_only', 'identity_only', 'source_exact', 'source_exact_identity', 'legacy_source_identity')
    ufs = {name: UnionFind(n) for name in names}
    edge_fields = ['kind', 'left_id', 'right_id', 'evidence', 'cosine']
    with (out / 'edges.csv').open('x', newline='') as f:
        ew = csv.DictWriter(f, fieldnames=edge_fields)
        ew.writeheader()
        for kind, grouped, targets in (
            ('legacy', old, ('legacy_source', 'legacy_source_identity')),
            ('source', source, ('corrected_source', 'source_exact', 'source_exact_identity')),
            ('exact', {h: sorted({i for i, _ in m}) for h, m in sha_members.items() if len({i for i, _ in m}) > 1}, ('exact_only', 'source_exact', 'source_exact_identity')),
        ):
            for key, members in grouped.items():
                for i in members[1:]:
                    for name in targets:
                        ufs[name].union(members[0], i)
                    if kind != 'legacy':
                        ew.writerow(dict(kind=kind, left_id=ids[members[0]], right_id=ids[i], evidence=key, cosine=''))
        a, b = np.where(v >= THRESHOLD)
        identity_edges = set()
        cross_edges = 0
        for left, col in zip(a.tolist(), b.tolist()):
            right = int(j[left, col])
            identity_edges.add(tuple(sorted((left, right))))
            cross_edges += rows[left]['split'] != rows[right]['split']
            for name in ('identity_only', 'source_exact_identity', 'legacy_source_identity'):
                ufs[name].union(left, right)
            ew.writerow(dict(kind='identity_top16', left_id=ids[left], right_id=ids[right], evidence='cached_glint_human_cosine_gte_0.9', cosine=float(v[left, col])))
    stage_stats = {}
    with (out / 'components.jsonl').open('x') as f:
        for name in names:
            groups = ufs[name].groups()
            stage_stats[name] = stats(groups, rows)
            for g in sorted(groups, key=lambda g: (-len(g), min(ids[i] for i in g))):
                f.write(json.dumps({'stage': name, 'component_id': group_id(g, ids), 'size': len(g),
                                    'original_split_counts': dict(Counter(rows[i]['split'] for i in g)),
                                    'source_dataset_counts': dict(Counter(rows[i]['source_dataset'] for i in g)),
                                    'members': sorted(ids[i] for i in g)}) + '\n')
    saturated = set(np.where(v[:, -1] >= THRESHOLD)[0].tolist())
    groups = ufs['source_exact_identity'].groups()
    # Predeclared engineering rule: isolate entire components too large for the
    # original smaller holdout, plus any component touching saturated top16 rows.
    # No threshold search, manual labels, or outcome-based reassignment.
    holdout_budget = min(original_counts['val'], original_counts['test'])
    require(holdout_budget > 0, 'Original holdout empty')
    quarantine, retained, reasons = [], [], {}
    for g in groups:
        why = []
        if len(g) > holdout_budget:
            why.append('component_exceeds_original_smaller_holdout_budget')
        if saturated.intersection(g):
            why.append('component_contains_top16_saturated_row')
        if why:
            quarantine.append(g)
            reasons[group_id(g, ids)] = '|'.join(why)
        else:
            retained.append(g)
    retained_n = sum(map(len, retained))
    fractions = {'train': original_counts['train'] / n, 'dev': original_counts['val'] / n, 'final': original_counts['test'] / n}
    targets = {s: retained_n * fraction for s, fraction in fractions.items()}
    counts = Counter({s: 0 for s in fractions})
    assignment = {}
    for g in sorted(retained, key=lambda g: (-len(g), hashlib.sha256(f'{SEED}:{group_id(g, ids)}'.encode()).hexdigest())):
        split = max(fractions, key=lambda s: (targets[s] - counts[s], s))
        counts[split] += len(g)
        for i in g:
            assignment[i] = split
    for g in quarantine:
        for i in g:
            assignment[i] = 'quarantine'
    require(len(assignment) == n, 'Candidate coverage incomplete')
    require(all(len({assignment[i] for i in g}) == 1 for g in groups), 'Observed component crosses candidate partitions')
    records = []
    for g in groups:
        gid = group_id(g, ids)
        for i in g:
            records.append({'id': ids[i], 'original_split': rows[i]['split'], 'candidate_split': assignment[i],
                            'component_id': gid, 'component_size': len(g), 'source_group': source_group(rows[i]),
                            'source_dataset': rows[i]['source_dataset'], 'source_key': rows[i]['source_key'],
                            'human_path': rows[i]['human_path'], 'mannequin_path': rows[i]['mannequin_path'],
                            'quarantine_reason': reasons.get(gid, ''), 'formal_split_ready': 'false'})
    records.sort(key=lambda r: r['id'])
    fields = list(records[0])
    write_csv(out / 'candidate_all.csv', fields, records)
    for split in (*fractions, 'quarantine'):
        filename = 'quarantine.csv' if split == 'quarantine' else f'candidate_{split}.csv'
        write_csv(out / filename, fields, (r for r in records if r['candidate_split'] == split))
    write_csv(out / 'source_key_corrections.csv', ['id', 'source_key', 'legacy_source_group', 'corrected_source_group'],
              ({'id': r['id'], 'source_key': r['source_key'], 'legacy_source_group': legacy_source_group(r),
                'corrected_source_group': source_group(r)} for r in rows if source_group(r) != legacy_source_group(r)))
    unchanged = all(sha256(path) == digest for path, digest in before_original.items())
    require(unchanged, 'Original metadata/splits changed during audit')
    summary = {'schema': 'protocol_group_audit_v1', 'audit_complete': True, 'formal_split_ready': False,
               'clean_retraining_ready': False, 'candidate_status': 'DIAGNOSTIC_ONLY_FORMAL_BLOCKED',
               'cpu_only_this_run': True, 'gpu_hours_this_run': 0, 'records': n,
               'original_split_counts': original_counts, 'original_metadata_splits_unchanged': unchanged,
               'fixed_policy': {'identity_top_k': TOP_K, 'identity_cosine_gte': THRESHOLD, 'seed': SEED,
                                'quarantine_component_size_gt': holdout_budget,
                                'quarantine_top16_saturated_components': True,
                                'assignment': 'largest first; SHA256 seed tie order; maximum remaining target deficit',
                                'candidate_fractions_from_original': fractions},
               'source_key_changed_records': sum(source_group(r) != legacy_source_group(r) for r in rows),
               'stages': stage_stats,
               'exact': {'all_role_sha_groups_with_multiple_images': sum(len(m) > 1 for m in sha_members.values()),
                         'all_role_sha_groups_with_multiple_ids': len(exact),
                         'cross_original_split_sha_groups': sum(len({rows[i]['split'] for i in g}) > 1 for g in exact),
                         'historical_role_specific_summary': audit_summary},
               'identity': {'directed_threshold_edges': len(a), 'unique_undirected_threshold_edges': len(identity_edges),
                            'cross_original_split_directed_edges': cross_edges, 'top16_saturated_rows': len(saturated),
                            'cached_cosine_vs_full_bank_max_abs_error': max_error,
                            'cached_edge_values_within_1e_5_of_threshold': int((np.abs(v - THRESHOLD) <= 1e-5).sum()),
                            'full_threshold_graph_recomputed': False},
               'candidate': {'counts': dict(counts), 'retained_records': retained_n, 'quarantined_records': n - retained_n,
                             'quarantined_components': len(quarantine), 'retained_components': len(retained),
                             'observed_components_cross_candidate_partitions': 0,
                             'target_counts': targets,
                             'oversize_components': sum(len(g) > holdout_budget for g in groups),
                             'full_corpus_balanced_holdout_status': 'BLOCKED_OVERSIZE_COMPONENTS' if any(len(g) > holdout_budget for g in groups) else 'NOT_CERTIFIED',
                             'quarantine_semantics': 'entire proxy components excluded from all candidate train/dev/final use; no files moved'},
               'gaps': ['Unknown biological identities; raw Glint >=0.9 is uncalibrated.',
                        'Source grammar is provenance proxy; exporter/SKU/garment ontology unavailable.',
                        'Truncated top16 can omit threshold edges; connected components can only merge in the full graph.',
                        'Audit reuses prior GPU-produced Glint features/neighbors; this audit runs CPU only.',
                        'Audited image immutability assumed; image bytes not rehashed in this run.',
                        'Non-exact/semantic duplicates and nonzero-Hamming near duplicates not cleared.',
                        'Quarantine changes population; retained subset has no representativeness/power guarantee.',
                        'Historical checkpoint contamination, clean retraining, and final evaluation remain open.'],
               'seconds': time.monotonic() - start}
    write_json(out / 'provenance.json', {'sha256': provenance, 'original_before_sha256': before_original})
    write_json(out / 'summary.json', summary)
    checksums = {path.name: sha256(path) for path in sorted(out.iterdir()) if path.is_file()}
    write_json(out / 'artifact_sha256.json', checksums)
    with (out / 'AUDIT_COMPLETE_NOT_FORMAL_READY').open('x') as f:
        f.write(sha256(out / 'summary.json') + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
