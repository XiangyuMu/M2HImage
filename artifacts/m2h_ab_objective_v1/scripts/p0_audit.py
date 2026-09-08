"""Read-only dataset/provenance audit; writes only a fresh output directory.

Source-key sibling relations are conservative grouping candidates, not proof
of biological identity or duplicate pixels. No formal clean split is asserted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source_group(row: dict) -> str:
    key = row['source_key']
    stem, sep, suffix = key.rpartition('_')
    parent = stem if sep and suffix.isdigit() else key
    return row['source_dataset'] + ':' + parent


def summarize(rows: list[dict], splits: dict[str, set[str]]) -> tuple[dict, dict]:
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate manifest IDs')
    membership = defaultdict(list)
    for split, values in splits.items():
        for sample_id in values:
            membership[sample_id].append(split)
    groups = defaultdict(list)
    mismatch = []
    for row in rows:
        groups[source_group(row)].append(row)
        if membership.get(row['id']) != [row['split']]:
            mismatch.append(row['id'])
    crossing = {
        key: [{'id': row['id'], 'split': row['split']} for row in values]
        for key, values in groups.items()
        if len({row['split'] for row in values}) > 1
    }
    report = {
        'rows': len(rows), 'unique_ids': len(set(ids)),
        'counts_by_split': dict(Counter(r['split'] for r in rows)),
        'counts_by_source': dict(Counter(r['source_dataset'] for r in rows)),
        'source_parent_candidate_groups': len(groups),
        'cross_split_parent_candidate_groups': len(crossing),
        'cross_split_parent_candidate_rows': sum(map(len, crossing.values())),
        'split_manifest_mismatch_ids': mismatch,
        'split_ids_missing_manifest': sorted(set(membership) - set(ids)),
        'interpretation': 'Parent-key overlap is a grouping risk, not proven identity or image duplication.',
        'formal_split_ready': False,
        'pending': ['all-role identity clustering', 'exact/near duplicate grouping',
                    'input-only inference audit', 'group-wise clean split and clean retraining'],
    }
    return report, crossing


def select_old_val(rows: list[dict], count: int) -> list[str]:
    used = set()
    selected = []
    for row in sorted(rows, key=lambda r: hashlib.sha256(('20260907:' + r['id']).encode()).hexdigest()):
        group = source_group(row)
        if row['split'] != 'val' or group in used:
            continue
        selected.append(row['id'])
        used.add(group)
        if len(selected) == count:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--probe-count', type=int, default=64)
    parser.add_argument('--hash-crossing', action='store_true')
    args = parser.parse_args()
    if args.probe_count < 1:
        parser.error('--probe-count must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    manifest = args.root / 'metadata/manifest.csv'
    with manifest.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    splits = {s: set((args.root / f'splits/{s}.txt').read_text().split()) for s in ('train', 'val', 'test')}
    report, crossing = summarize(rows, splits)
    refs = ['train_paired.py', 'spatial_conditions.py', 'dataset.py', 'build_cache.py',
            'paired_eval_common.py', 'eval_b2.py', 'metrics_v2/parsing.py',
            'configs/spatial_hair_ab_space.yaml', 'configs/warmup.yaml']
    git = lambda *cmd: subprocess.check_output(['git', '-C', str(args.repo), *cmd])
    report['provenance'] = {
        'manifest_sha256': digest(manifest), 'git_commit': git('rev-parse', 'HEAD').decode().strip(),
        'working_diff_sha256': hashlib.sha256(git('diff', 'HEAD', '--binary')).hexdigest(),
        'files_sha256': {str(args.repo / f): digest(args.repo / f) for f in refs},
    }
    (args.out / 'git_status.txt').write_bytes(git('status', '--short'))
    # Preserve exact task input code and dirty diff for subsequent provenance.
    (args.out / 'preexisting_changes.patch').write_bytes(git('diff', 'HEAD', '--binary'))
    (args.out / 'cross_split_source_candidates.json').write_text(json.dumps(crossing, indent=2))
    selected = select_old_val(rows, args.probe_count)
    (args.out / 'old_val_probe_ids.json').write_text(json.dumps(selected, indent=2))
    report['old_val_probe'] = {'ids': selected, 'role': 'exploratory diagnostics, NOT final holdout'}
    if args.hash_crossing:
        candidate_ids = {item['id'] for values in crossing.values() for item in values}
        hashes = defaultdict(list)
        failures = []
        with (args.out / 'candidate_hashes.jsonl').open('x') as output:
            for i, row in enumerate(r for r in rows if r['id'] in candidate_ids):
                for role in ('human', 'mannequin'):
                    path = Path(row[f'{role}_path'])
                    try:
                        entry = {'id': row['id'], 'split': row['split'], 'role': role, 'sha256': digest(path)}
                        hashes[(role, entry['sha256'])].append(entry)
                    except OSError as exc:
                        entry = {'id': row['id'], 'role': role, 'error': str(exc)}
                        failures.append(entry)
                    output.write(json.dumps(entry) + '\n')
                if i % 100 == 0:
                    output.flush()
                    print(json.dumps({'phase': 'candidate_sha256', 'rows_done': i + 1}), flush=True)
        dupes = [v for v in hashes.values() if len({x['split'] for x in v}) > 1]
        report['exact_hash_audit'] = {'scope': 'cross-source-parent candidates ONLY',
                                    'sample_rows': len(candidate_ids), 'cross_split_duplicate_groups': len(dupes),
                                    'errors': failures}
        (args.out / 'exact_duplicate_groups.json').write_text(json.dumps(dupes, indent=2))
    report['seconds'] = time.time() - start
    report['status'] = 'audit_complete_not_training_ready'
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2))
    (args.out / 'READY').write_text('audit artifacts complete; formal split not ready\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
