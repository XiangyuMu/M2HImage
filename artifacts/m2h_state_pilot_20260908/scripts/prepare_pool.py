"""Freeze train-only provenance groups and reference pairs; no score selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def group(r):
    m = re.fullmatch(r'(multiimages_x2__\d+)_\d+', r['source_key'])
    return r['source_dataset'] + ':' + (m[1] if m else r['source_key'])


def main():
    p = argparse.ArgumentParser()
    for key in ('root', 'repo', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args()
    art = a.repo / 'artifacts/m2h_ab_objective_v1'
    paths = [a.root / 'metadata/manifest.csv', a.root / 'splits/train.txt',
             a.root / 'derived/identity_bank_v2.npz',
             art / 'data_protocol_v2/dev128_pairs.json', art / 'full_duplicates/exact_groups.json']
    with paths[0].open() as f:
        meta = {r['id']: r for r in csv.DictReader(f)}
    train = set(paths[1].read_text().split())
    assert train == {k for k, r in meta.items() if r['split'] == 'train'}
    dev = json.loads(paths[3].read_text())['pairs']
    held = {k for k, r in meta.items() if r['split'] != 'train'} | {'47160'}
    held |= {r[key] for r in dev for key in ('mid', 'jid')}
    # Exclude all members of audited exact-duplicate groups, even train-only.
    held |= {m['id'] for g in json.loads(paths[4].read_text()) for m in g['members']}
    heldgroups = {group(meta[k]) for k in held if k in meta}
    with np.load(paths[2], allow_pickle=False) as z:
        ids = z['ids'].astype(str)
        gender, age, skin = z['gender'].astype(str), z['age'], z['skin_cluster']
    index = {k: i for i, k in enumerate(ids)}
    eligible = sorted(k for k in train & set(index) if k not in held and group(meta[k]) not in heldgroups
                      and (a.root / 'phase1/cache_768x1024/samples' / (k+'.npz')).is_file()
                      and (a.root / 'derived/region_masks_z' / (k+'.npz')).is_file())
    rng = np.random.default_rng(20260908)
    order = rng.permutation(eligible).tolist()
    refs, refgroups = [], set()
    for k in order:
        g = group(meta[k])
        if g not in refgroups:
            refs.append(k)
            refgroups.add(g)
        if len(refs) == 32:
            break
    rows, used = [], set(refgroups)
    for k in order:
        g = group(meta[k])
        if g in used:
            continue
        i = index[k]
        compatible = [j for j in refs if gender[index[j]] == gender[i]
                      and abs(float(age[index[j]])-float(age[i])) <= 15
                      and abs(int(skin[index[j]])-int(skin[i])) <= 1]
        if len(compatible) < 2:
            continue
        j, q = rng.choice(compatible, 2, replace=False).tolist()
        rows.append(dict(mid=k, jid=j, kid=q, source_group=g))
        used.add(g)
        if len(rows) == 256:
            break
    assert len(rows) == 256 and len({r['source_group'] for r in rows}) == 256
    pairs = [dict(mid=r['mid'], jid=r[key], seed=0) for r in rows for key in ('jid', 'kid')]
    assert len({(r['mid'], r['jid']) for r in pairs}) == 512
    a.out.mkdir(parents=True, exist_ok=False)
    common = dict(schema=1, seed=20260908, inputs={str(x): sha(x) for x in paths},
                  sampling='32 random train reference groups; two strict-compatible references per source, fixed random without replacement',
                  source_group_rule='namespace + multiimages_x2 parent; other keys opaque',
                  excluded_ids=len(held), excluded_groups=len(heldgroups), eligible_ids=len(eligible),
                  limitation='No semantic identity-disjoint certification; no new manual annotations; historical A2/A4 not clean retrained.')
    (a.out / 'pool.json').write_text(json.dumps(dict(common, rows=rows), indent=2))
    (a.out / 'pairs.json').write_text(json.dumps(dict(common, pairs=pairs), indent=2))
    print(json.dumps(dict(groups=len(rows), pairs=len(pairs), references=len({r['jid'] for r in pairs}),
                          pool_sha256=sha(a.out/'pool.json'))), flush=True)


if __name__ == '__main__':
    main()
