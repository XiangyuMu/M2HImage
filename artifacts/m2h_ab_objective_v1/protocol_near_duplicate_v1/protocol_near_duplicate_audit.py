"""Exact radius-4 dHash enumeration; proxy risks, never semantic/identity labels.

Writes only a fresh output directory. Five disjoint blocks cover all 64 bits:
four or fewer flipped bits leave at least one block equal (pigeonhole proof).
Each unordered unique-hash candidate is deduplicated before exact bit_count.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import re
import time

RADIUS = 4
WIDTHS = (13, 13, 13, 13, 12)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def blocks(value):
    result, offset = [], 0
    for width in WIDTHS:
        result.append((value >> offset) & ((1 << width) - 1))
        offset += width
    return result


def candidates(values):
    postings = [defaultdict(list) for _ in WIDTHS]
    for i, value in enumerate(values):
        keys = blocks(value)
        prior = set()
        for table, key in zip(postings, keys):
            prior.update(table[key])
            table[key].append(i)
        yield i, prior


class UF:
    def __init__(self, n):
        self.p = list(range(n))
        self.s = [1] * n

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.s[a] < self.s[b]:
            a, b = b, a
        self.p[b] = a
        self.s[a] += self.s[b]

    def groups(self):
        g = defaultdict(list)
        for i in range(len(self.p)):
            g[self.find(i)].append(i)
        return list(g.values())


def dump(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def csv_out(path, fields, rows):
    with path.open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def self_test():
    rng = random.Random(20260907)
    vals = [rng.getrandbits(64) for _ in range(80)]
    for base in vals[:10]:
        for distance in range(1, 6):
            vals.append(base ^ sum(1 << b for b in rng.sample(range(64), distance)))
    vals += [0, (1 << 64) - 1, sum(1 << b for b in (0, 13, 26, 39)), 1 << 63]
    vals = sorted(set(vals))
    actual = {(j, i) for i, js in candidates(vals) for j in js if (vals[i] ^ vals[j]).bit_count() <= RADIUS}
    expected = {(j, i) for i in range(len(vals)) for j in range(i) if (vals[i] ^ vals[j]).bit_count() <= RADIUS}
    require(actual == expected and sum(WIDTHS) == 64, 'Pigeonhole enumeration regression')
    require(list(candidates([0, 1, 2]))[-1][1] == {0, 1}, 'Candidate dedup regression')
    print('SELF_TEST_PASS: indexed versus exhaustive all-pairs; block boundaries and radius 0..5', flush=True)


class BudgetStop(Exception):
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--max-seconds', type=int, default=360)
    p.add_argument('--max-image-pairs', type=int, default=10000000)
    p.add_argument('--max-id-edges', type=int, default=2000000)
    args = p.parse_args()
    self_test()
    if args.self_test:
        return
    require(args.artifacts and args.out and not args.out.exists(), 'Require artifacts and fresh out')
    require(args.out.resolve().parent.name == 'protocol_near_duplicate_v1', 'Output must be inside new protocol_near_duplicate_v1')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    start = time.monotonic()
    art, out = args.artifacts, args.out
    prior = art / 'protocol_group_audit_v1/results'
    audit = art / 'full_duplicates/per_image.jsonl'
    manifest = prior / 'candidate_all.csv'
    inputs = [audit, art / 'full_duplicates/summary.json', manifest, prior / 'summary.json',
              prior / 'provenance.json', art / 'identity_neighbors_full/neighbors.npz', Path(__file__).resolve()]
    input_sha = {str(path): sha(path) for path in inputs}
    with manifest.open(newline='') as f:
        rows = list(csv.DictReader(f))
    ids = [r['id'] for r in rows]
    index = {sid: i for i, sid in enumerate(ids)}
    require(len(index) == len(rows) and rows, 'Invalid candidate IDs')
    parts = {r['id']: r['candidate_split'] for r in rows}
    require(set(parts.values()) <= {'train', 'dev', 'final', 'quarantine'}, 'Unknown partition')
    images, seen, hash_images = [], set(), defaultdict(list)
    with audit.open() as f:
        for line in f:
            r = json.loads(line)
            sid, role = r['id'], r['role']
            require(sid in index and role in ('human', 'mannequin') and (sid, role) not in seen, 'Invalid image key')
            m = rows[index[sid]]
            require(r['status'] == 'ok' and r['path'] == m[role + '_path'] and r['split'] == m['original_split'], 'Audit/manifest mismatch')
            require(re.fullmatch('[0-9a-f]{16}', r['dhash']) and re.fullmatch('[0-9a-f]{64}', r['sha256']), 'Malformed hashes')
            seen.add((sid, role))
            hash_images[int(r['dhash'], 16)].append(len(images))
            images.append(r)
    ds = json.loads((art / 'full_duplicates/summary.json').read_text())
    require(len(images) == len(rows) * 2 == ds['expected'] == ds['completed'] and not ds['failed'], 'Incomplete image inputs')
    old_prov = json.loads((prior / 'provenance.json').read_text())['sha256']
    require(old_prov[str(audit)] == input_sha[str(audit)], 'Image audit changed since grouping')
    require(old_prov[str(art / 'identity_neighbors_full/neighbors.npz')] == input_sha[str(art / 'identity_neighbors_full/neighbors.npz')], 'Neighbors changed since grouping')
    vals = sorted(hash_images)
    members = [hash_images[v] for v in vals]
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'hash_members.jsonl').open('x') as f:
        for v, ms in zip(vals, members):
            f.write(json.dumps({'dhash': f'{v:016x}', 'images': [{'image_index': i, 'id': images[i]['id'], 'role': images[i]['role'], 'partition': parts[images[i]['id']]} for i in ms]}) + '\n')
    id_uf, image_uf, joint = UF(len(ids)), UF(len(images)), UF(len(ids))
    old_groups = defaultdict(list)
    for i, r in enumerate(rows):
        old_groups[r['component_id']].append(i)
    for g in old_groups.values():
        require(len({parts[ids[i]] for i in g}) == 1, 'Old component crosses partition')
        for i in g[1:]:
            joint.union(g[0], i)
    counters, distances, role_pairs, part_pairs = Counter(), Counter(), Counter(), Counter()
    edge_map = {}
    complete, stopped = True, None
    processed = 0
    def budget():
        if time.monotonic() - start > args.max_seconds:
            raise BudgetStop('elapsed enumeration budget')
        if counters['image_pairs'] >= args.max_image_pairs:
            raise BudgetStop('image-pair safety cap')
        if len(edge_map) >= args.max_id_edges:
            raise BudgetStop('ID-edge safety cap')
    with gzip.open(out / 'image_pairs.csv.gz', 'xt', compresslevel=1, newline='') as f, (out / 'hash_pairs.csv').open('x', newline='') as hf:
        iw = csv.writer(f)
        iw.writerow(['left_image_index', 'right_image_index', 'hamming', 'sha256_equal'])
        hw = csv.writer(hf)
        hw.writerow(['left_dhash', 'right_dhash', 'hamming', 'expected_image_pairs'])
        def record_hash_pair(a, b, distance):
            pairs = itertools.combinations(members[a], 2) if a == b else itertools.product(members[a], members[b])
            expected = len(members[a]) * (len(members[a]) - 1) // 2 if a == b else len(members[a]) * len(members[b])
            hw.writerow([f'{vals[a]:016x}', f'{vals[b]:016x}', distance, expected])
            counters['matched_hash_pairs_including_equal_buckets'] += 1
            for left, right in pairs:
                if counters['image_pairs'] % 1024 == 0:
                    budget()
                l, r = images[left], images[right]
                equal = l['sha256'] == r['sha256']
                iw.writerow([left, right, distance, int(equal)])
                counters['image_pairs'] += 1
                counters['sha256_equal_image_pairs'] += equal
                counters['same_hash_image_pairs'] += distance == 0
                distances[distance] += 1
                role_pairs['|'.join(sorted((l['role'], r['role'])))] += 1
                lp, rp = parts[l['id']], parts[r['id']]
                part_pairs['|'.join(sorted((lp, rp)))] += 1
                counters['cross_partition_image_pairs'] += lp != rp
                image_uf.union(left, right)
                a_id, b_id = sorted((index[l['id']], index[r['id']]))
                if a_id == b_id:
                    counters['same_id_image_pairs'] += 1
                    continue
                key = a_id * len(ids) + b_id
                if key not in edge_map:
                    edge_map[key] = [distance, 0, 0]
                rec = edge_map[key]
                rec[0] = min(rec[0], distance)
                rec[1] += 1
                rec[2] += equal
                id_uf.union(a_id, b_id)
                joint.union(a_id, b_id)
        try:
            for a, ms in enumerate(members):
                budget()
                if len(ms) > 1:
                    record_hash_pair(a, a, 0)
            counters['equal_hash_phase_complete'] = 1
            for i, js in candidates(vals):
                budget()
                counters['unique_hash_candidate_comparisons'] += len(js)
                for offset, j in enumerate(sorted(js)):
                    if offset % 4096 == 0:
                        budget()
                    distance = (vals[i] ^ vals[j]).bit_count()
                    counters['unique_hash_candidates_verified'] += 1
                    if distance <= RADIUS:
                        counters['distinct_hash_pairs_within_radius'] += 1
                        record_hash_pair(j, i, distance)
                processed = i + 1
                if processed % 5000 == 0:
                    print(json.dumps({'processed_unique_hashes': processed, 'total_unique_hashes': len(vals), 'image_pairs': counters['image_pairs'], 'seconds': time.monotonic() - start}), flush=True)
        except BudgetStop as exc:
            complete, stopped = False, str(exc)
    edges = []
    for key, (distance, count, exact) in sorted(edge_map.items()):
        a, b = divmod(key, len(ids))
        edges.append({'left_id': ids[a], 'right_id': ids[b], 'left_partition': parts[ids[a]],
                      'right_partition': parts[ids[b]], 'minimum_hamming': distance,
                      'image_pair_witnesses': count, 'exact_sha256_witnesses': exact})
    fields = ['left_id', 'right_id', 'left_partition', 'right_partition', 'minimum_hamming', 'image_pair_witnesses', 'exact_sha256_witnesses']
    csv_out(out / 'id_edges.csv', fields, edges)
    cross_edges = [e for e in edges if e['left_partition'] != e['right_partition']]
    csv_out(out / 'cross_partition_id_edges.csv', fields, cross_edges)
    graph_stats, risk_ids, proposed = {}, set(), set()
    with (out / 'components.jsonl').open('x') as f:
        for name, uf, image_level in (('image_dhash', image_uf, True), ('id_dhash', id_uf, False), ('old_components_plus_dhash', joint, False)):
            groups = uf.groups()
            crossing, crossing_members = 0, 0
            for g in groups:
                sids = [images[i]['id'] for i in g] if image_level else [ids[i] for i in g]
                pc = Counter(parts[sid] for sid in sids)
                is_cross = len(pc) > 1
                crossing += is_cross
                crossing_members += len(g) if is_cross else 0
                if name == 'old_components_plus_dhash' and is_cross:
                    risk_ids.update(sids)
                    proposed.update(sid for sid in sids if parts[sid] != 'quarantine')
                f.write(json.dumps({'graph': name, 'component_anchor': min(g), 'size': len(g), 'partition_counts': dict(pc),
                                    'cross_partition': is_cross, 'members': g if image_level else sids,
                                    'coverage_complete': complete}) + '\n')
            graph_stats[name] = {'components': len(groups), 'largest': max(map(len, groups)),
                                 'non_singletons': sum(len(g) > 1 for g in groups), 'cross_partition_components': crossing,
                                 'members_in_cross_partition_components': crossing_members,
                                 'size_histogram': dict(sorted(Counter(map(len, groups)).items()))}
    rfields = ['id', 'candidate_partition', 'original_component_id', 'action', 'reason', 'coverage_complete']
    risk_rows = [{'id': sid, 'candidate_partition': parts[sid], 'original_component_id': rows[index[sid]]['component_id'],
                  'action': 'suggest_additional_quarantine' if sid in proposed else 'already_quarantined',
                  'reason': 'cross_partition_component_after_old_group_and_dhash_union', 'coverage_complete': complete} for sid in sorted(risk_ids)]
    csv_out(out / 'risk_ids.csv', rfields, risk_rows)
    csv_out(out / 'additional_quarantine_recommendations.csv', rfields, (r for r in risk_rows if r['id'] in proposed))
    remaining = Counter(parts[sid] for sid in ids if parts[sid] != 'quarantine' and sid not in proposed)
    require(all(e['left_id'] in proposed or parts[e['left_id']] == 'quarantine' for e in cross_edges), 'Isolation closure left')
    require(all(e['right_id'] in proposed or parts[e['right_id']] == 'quarantine' for e in cross_edges), 'Isolation closure right')
    # Optional cheap logic audit, no model inference or all-bank pairwise repeat.
    import numpy as np
    with np.load(art / 'identity_neighbors_full/neighbors.npz', allow_pickle=False) as z:
        ni, v, j = z['ids'].tolist(), z['cosine'], z['neighbor_index']
        require(set(ni) == set(ids) and len(ni) == len(ids) and v.shape == j.shape == (len(ids), 16), 'Neighbor shape/ID mismatch')
        require(np.isfinite(v).all() and (np.diff(v, axis=1) <= 1e-7).all(), 'Invalid sorted cosine')
        require(np.issubdtype(j.dtype, np.integer) and j.min() >= 0 and j.max() < len(ids), 'Invalid neighbors')
        require(not (j == np.arange(len(ids))[:, None]).any() and all(len(set(js.tolist())) == 16 for js in j), 'Self/duplicate neighbors')
        npart = np.array([parts[sid] for sid in ni])
        retained = npart != 'quarantine'
        above = v >= 0.9
        top16 = {'retained_rows': int(retained.sum()), 'retained_saturated_rows': int((retained & (v[:, -1] >= 0.9)).sum()),
                 'retained_max_kth_cosine': float(v[retained, -1].max()),
                 'retained_threshold_directed_edges': int(above[retained].sum()),
                 'retained_cross_partition_threshold_edges': int((above & retained[:, None] & (npart[:, None] != npart[j])).sum()),
                 'global_saturated_rows': int((v[:, -1] >= 0.9).sum()),
                 'retained_entries_within_1e_5_of_threshold': int((np.abs(v[retained] - 0.9) <= 1e-5).sum()),
                 'logical_certificate': 'If these are exact descending global top16 for the same score, kth < 0.9 implies every >=0.9 neighbor is present. All retained rows are checked; rank completeness is an input assumption, not recomputed.',
                 'full_bank_ranking_recomputed': False, 'biological_identity_certified': False}
    unchanged = all(sha(path) == digest for path, digest in input_sha.items())
    require(unchanged, 'Read-only input changed')
    summary = {'schema': 'protocol_near_duplicate_v1', 'radius': RADIUS, 'block_widths': WIDTHS,
               'coverage_complete': complete, 'stop_reason': stopped,
               'coverage_meaning': 'all unordered image pairs at audited 64-bit dHash Hamming <=4, including cross-role and within-ID pairs' if complete else 'PARTIAL LOWER-BOUND COUNTS; NO GLOBAL CLEARANCE',
               'formal_split_ready': False, 'semantic_duplicate_or_identity_certified': False,
               'cpu_threads_limit': 2, 'gpu_hours': 0, 'image_records': len(images), 'ids': len(ids),
               'unique_hashes': len(vals), 'unique_hashes_processed': processed,
               'equal_hash_buckets': sum(len(ms) > 1 for ms in members),
               'expected_equal_hash_image_pairs': sum(len(ms) * (len(ms) - 1) // 2 for ms in members),
               'counters': dict(counters), 'hamming_image_pair_histogram': dict(distances),
               'role_pair_counts': dict(role_pairs), 'partition_image_pair_counts': dict(part_pairs),
               'distinct_id_edges': len(edges), 'cross_partition_id_edges': len(cross_edges),
               'cross_retained_partition_id_edges': sum('quarantine' not in (e['left_partition'], e['right_partition']) for e in cross_edges),
               'graphs': graph_stats,
               'isolation_proposal': {'policy': 'suggest whole old-component-plus-dHash closure for every cross-partition risk, including links to existing quarantine; do not mutate manifests',
                                      'risk_ids_including_existing_quarantine': len(risk_ids),
                                      'additional_quarantine_ids': len(proposed),
                                      'additional_by_candidate_partition': dict(Counter(parts[sid] for sid in proposed)),
                                      'remaining_candidate_counts_if_adopted': dict(remaining),
                                      'observed_cross_partition_edges_after_proposal': 0,
                                      'formal_usable': False, 'coverage_complete': complete},
               'top16_retained_logic': top16, 'inputs_unchanged': unchanged,
               'limits': {'max_seconds': args.max_seconds, 'max_image_pairs': args.max_image_pairs, 'max_id_edges': args.max_id_edges},
               'gaps': ['dHash closeness is a visual hash risk proxy, not verified semantic duplication or identity.',
                        'Exactness covers audited hashes, not image immutability or all forms of perceptual similarity.',
                        'Quarantine proposal changes population and may overexclude; no manual labels or threshold tuning.',
                        'Biological identity, source ontology, clean retraining and final protocol remain open.'],
               'seconds': time.monotonic() - start}
    dump(out / 'provenance.json', {'input_sha256': input_sha})
    dump(out / 'summary.json', summary)
    dump(out / 'artifact_sha256.json', {x.name: sha(x) for x in sorted(out.iterdir()) if x.is_file()})
    marker = 'AUDIT_COMPLETE_NOT_FORMAL_READY' if complete else 'COVERAGE_INCOMPLETE'
    with (out / marker).open('x') as f:
        f.write(sha(out / 'summary.json') + '\n')
    compact = dict(summary)
    compact['graphs'] = {k: {a: b for a, b in v.items() if a != 'size_histogram'} for k, v in graph_stats.items()}
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == '__main__':
    main()
