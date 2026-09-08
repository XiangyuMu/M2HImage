"""All-input exact SHA and perceptual-hash candidates; not final identity labels."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import time

from PIL import Image
import numpy as np


def inspect(item):
    entry = dict(item)
    try:
        digest = hashlib.sha256()
        path = Path(entry['path'])
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024*1024), b''):
                digest.update(block)
        with Image.open(path) as im:
            gray=np.asarray(im.convert('L').resize((9,8),Image.Resampling.LANCZOS))
            dhash=np.packbits(gray[:,1:]>gray[:,:-1]).tobytes().hex()
            size=im.size
        entry.update(status='ok',sha256=digest.hexdigest(),dhash=dhash,size=size)
    except Exception as exc:
        entry.update(status='failed',error=str(exc))
    return entry


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    with (args.root/'metadata/manifest.csv').open(newline='') as h:
        rows=list(csv.DictReader(h))
    items=[{'id':r['id'],'split':r['split'],'role':role,'path':r[f'{role}_path']}
           for r in rows for role in ('human','mannequin')]
    sha,perceptual=defaultdict(list),defaultdict(list)
    failed=[]
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor, (args.out/'per_image.jsonl').open('x') as handle:
        for n,result in enumerate(executor.map(inspect,items),1):
            handle.write(json.dumps(result)+'\n')
            if result['status']=='ok':
                ref={k:result[k] for k in ('id','split','role')}
                sha[(result['role'],result['sha256'])].append(ref)
                perceptual[(result['role'],result['dhash'])].append(ref)
            else:
                failed.append(result)
            if n%500==0 or n==len(items):
                handle.flush()
                progress={'completed':n,'expected':len(items),'failed':len(failed),'seconds':time.monotonic()-start}
                (args.out/'progress.json').write_text(json.dumps(progress))
                print(json.dumps(progress),flush=True)
    exact=[{'role':key[0],'sha256':key[1],'members':v}
           for key,v in sha.items() if len(v)>1]
    dhash=[{'role':key[0],'dhash':key[1],'members':v}
           for key,v in perceptual.items() if len(v)>1]
    (args.out/'exact_groups.json').write_text(json.dumps(exact))
    (args.out/'dhash_equal_candidates.json').write_text(json.dumps(dhash))
    summary={'expected':len(items),'completed':len(items),'failed':len(failed),
             'exact_duplicate_groups':len(exact),
             'cross_split_exact_duplicate_groups':sum(len({x['split'] for x in v['members']})>1 for v in exact),
             'equal_dhash_candidate_groups':len(dhash),'formal_split_ready':False,
             'warning':'Equal dHash is only a candidate, not exact duplication. Nonzero Hamming near-duplicates still unaudited.',
             'seconds':time.monotonic()-start,'gpu_hours':0}
    (args.out/'failures.json').write_text(json.dumps(failed))
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
