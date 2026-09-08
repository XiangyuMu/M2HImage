"""All-corpus cosine-neighbor diagnostics; no threshold-selected formal split.

Top-k edges are only candidates and do not enumerate every above-threshold
pair. Reports deliberately do not call their components ground-truth IDs.
"""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--batch',type=int,default=512)
    p.add_argument('--top-k',type=int,default=16)
    args=p.parse_args()
    if not (args.features/'FEATURES_READY').exists():
        raise RuntimeError('Incomplete identity features')
    ids,embeds=[],[]
    for f in sorted(args.features.glob('embeddings_*.npz')):
        with np.load(f,allow_pickle=False) as z:
            ids.extend(z['ids'].tolist()); embeds.append(z['embeds'])
    if len(set(ids))!=len(ids):
        raise ValueError('Duplicate feature IDs')
    args.out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    torch.backends.cuda.matmul.allow_tf32=False
    x=torch.tensor(np.concatenate(embeds),device=args.device,dtype=torch.float32)
    x=torch.nn.functional.normalize(x,dim=1)
    vals,indices=[],[]
    for lo in range(0,len(ids),args.batch):
        hi=min(len(ids),lo+args.batch)
        sim=x[lo:hi]@x.T
        sim[torch.arange(hi-lo,device=args.device),torch.arange(lo,hi,device=args.device)]=-2
        v,j=sim.topk(min(args.top_k,len(ids)-1),dim=1)
        vals.append(v.cpu().numpy()); indices.append(j.cpu().numpy())
        print(json.dumps({'completed':hi,'total':len(ids)}),flush=True)
    vals,indices=np.concatenate(vals),np.concatenate(indices)
    np.savez(args.out/'neighbors.npz',ids=np.array(ids),cosine=vals,neighbor_index=indices)
    with args.manifest.open(newline='') as h:
        metadata={r['id']:r for r in csv.DictReader(h)}
    splits=np.array([metadata[s]['split'] for s in ids])
    crossing=splits[:,None]!=splits[indices]
    report={'count':len(ids),'top_k':args.top_k,'formal_split_ready':False,
            'purpose':'Dataset grouping diagnostics only. No model selection or final evaluator.',
            'nearest_cosine_quantiles':dict(zip(['p0','p25','p50','p75','p95','p99','max'],
                np.quantile(vals[:,0],[0,.25,.5,.75,.95,.99,1]).tolist())),
            'thresholds':{}}
    for threshold in (.4,.5,.6,.7,.8,.9):
        above=vals>=threshold
        report['thresholds'][str(threshold)]={
            'rows_with_neighbor':int(above.any(1).sum()),
            'rows_with_cross_split_neighbor':int((above&crossing).any(1).sum()),
            'rows_with_kth_neighbor_above_threshold':int(above[:,-1].sum()),
            'note':'Exploratory thresholds not calibrated identity labels; top-k truncation explicitly reported.'}
    torch.cuda.synchronize(args.device)
    report['seconds']=time.monotonic()-start
    report['gpu_hours']=report['seconds']/3600
    (args.out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
