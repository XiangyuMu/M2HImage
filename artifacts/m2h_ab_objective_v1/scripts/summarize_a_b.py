"""Auditable paired A/B diagnostic summaries; no formal hypothesis-test verdict.

Two-factor pigeonhole bootstrap resamples input source groups and reference
files, not independent images. File proxies do not certify biological identity
independence. No training seed uncertainty is estimated by these intervals.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

FIELDS=('id_penalized','garment_dino','garment_hf_lpips','body_penalized',
        'head_penalized','body_coverage','head_coverage','carryin_id_penalized',
        'id_penalized_change_from_carryin','noisy_state_id_penalized',
        'id_penalized_change_from_noisy_state','carryin_dino','carryin_hf_lpips',
        'carryin_body_penalized','carryin_head_penalized')

def read_rows(p): return [json.loads(line) for line in p.read_text().splitlines()]
def key(r): return (r['mid'],r['jid'],r['seed'],r['branch'],r['tau'],r['k'])
def pairkey(r): return (r['mid'],r['jid'],r['seed'])
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def merge_run(run):
    rows={key(r):dict(r) for r in read_rows(run/'rows.jsonl')}
    statuses={}
    for stage in ('image','id','pose'):
        p=run/f'metrics_{stage}'/'per_image.jsonl'; rr=read_rows(p)
        if len(rr)!=len(rows) or {key(r) for r in rr}!=set(rows): raise ValueError('metric row coverage')
        statuses[stage]=dict(Counter(r['metric_status'] for r in rr))
        if any(r['metric_status'] in ('failed','reference_failed') for r in rr): raise ValueError('metric infrastructure failed')
        for r in rr:
            rows[key(r)].update({k:v for k,v in r.items() if k not in ('metric_status','error')})
            rows[key(r)]['status_'+stage]=r['metric_status']
    return list(rows.values()),statuses

def grouped_means(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r['branch'],r['tau'],r['k'])].append(r)
    out={}
    for (b,t,k),rr in sorted(groups.items()):
        metrics={}
        for field in FIELDS:
            v=[r[field] for r in rr if r.get(field) is not None]
            metrics[field]={'n':len(v),'missing':len(rr)-len(v),'mean':float(np.mean(v)) if v else None}
        out[f'{b}__t{t}__k{k}']={'pairs':len(rr),'metrics':metrics}
    return out

def weights_for(pairs, draws=10000):
    sources=sorted({r.get('m_source_group',r['mid']) for r in pairs})
    refs=sorted({r['jid'] for r in pairs})
    si={s:i for i,s in enumerate(sources)}; ji={s:i for i,s in enumerate(refs)}
    rng=np.random.default_rng(20260907)
    sw=rng.multinomial(len(sources),[1/len(sources)]*len(sources),size=draws)
    jw=rng.multinomial(len(refs),[1/len(refs)]*len(refs),size=draws)
    weights=sw[:,[si[r.get('m_source_group',r['mid'])] for r in pairs]]*jw[:,[ji[r['jid']] for r in pairs]]
    return weights,{'source_groups':len(sources),'reference_files':len(refs),'draws':draws}

def paired_effect(left,right,field):
    left={pairkey(r):r for r in left}; right={pairkey(r):r for r in right}
    if set(left)!=set(right): raise ValueError('unmatched experimental pairs')
    keys=sorted(left); eligible=[k for k in keys if left[k].get(field) is not None and right[k].get(field) is not None]
    if not eligible: return {'expected':len(keys),'valid':0,'delta_mean':None,'ci95':None}
    pairs=[left[k] for k in eligible]; w,info=weights_for(pairs)
    delta=np.array([left[k][field]-right[k][field] for k in eligible],np.float64)
    denom=w.sum(axis=1); valid=denom>0
    samples=(w[valid]@delta)/denom[valid]
    return {'expected':len(keys),'valid':len(eligible),'missing':len(keys)-len(eligible),
        'delta_mean':float(delta.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
        'bootstrap':{**info,'nonempty_replicates':int(valid.sum())}}

def comparisons(rows,controls):
    result={}
    for t,k in sorted({(r['tau'],r['k']) for r in rows}):
        for left,right in controls:
            a=[r for r in rows if (r['tau'],r['k'],r['branch'])==(t,k,left)]
            b=[r for r in rows if (r['tau'],r['k'],r['branch'])==(t,k,right)]
            result[f'{left}-minus-{right}__t{t}__k{k}']={f:paired_effect(a,b,f) for f in FIELDS}
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if not (a.run/'READY').is_file(): raise RuntimeError('Full pipeline not complete')
    a.out.mkdir(parents=True,exist_ok=False)
    rows=[]; statuses={}; inputs=[]; gpu_seconds=0
    for i in range(4):
        run=a.run/f'shard{i}'; rr,ss=merge_run(run);rows.extend(rr);statuses[run.name]=ss
        inputs.append(run/'rows.jsonl')
        for stage in ('image','id','pose'): inputs.append(run/f'metrics_{stage}'/'per_image.jsonl')
        for log in (a.run/'logs').glob(f'shard{i}_*.exit.json'):
            gpu_seconds+=json.loads(log.read_text())['seconds']
    if len(rows)!=1296 or len({key(r) for r in rows})!=1296: raise ValueError('A coverage')
    br,bs=merge_run(a.run/'b_metrics');statuses['B']=bs
    if len(br)!=512: raise ValueError('B coverage')
    for stage in ('image','id','pose'):
        inputs.append(a.run/'b_metrics'/f'metrics_{stage}'/'per_image.jsonl')
    for log in (a.run/'logs').glob('b_metrics_*.exit.json'):
        gpu_seconds+=json.loads(log.read_text())['seconds']
    result={'formal_result':False,'training_steps':0,'A_predictions':len(rows),'B_predictions':len(br),
        'status_counts':statuses,'GPU_process_wall_hours_including_startup':gpu_seconds/3600,
        'A_means':grouped_means(rows),'B_means':grouped_means(br),
        'A_paired_effects':comparisons(rows,[('matched','mismatched'),('matched','original'),('mismatched','original')]),
        'B_paired_effects':comparisons(br,[('inplace_feather','base'),('warp_feather','inplace_feather'),('warp_multiband','inplace_feather')]),
        'statistics':'Exploratory two-factor pigeonhole bootstrap on source-group/ref-file proxies, 10000 draws seed20260907; unadjusted descriptive intervals, not final gates or independent biological identities; no training-seed inference.',
        'A_warning':'Matched teacher already carries desired ID. Difference of carry-in-adjusted changes is diagnostic, not a learned-training causal benefit. Endpoint re-noising is not an on-policy trajectory.',
        'B_warning':'Fixed source-mask scores favor source copying. Raw boundary gradient/laplacian are not calibrated real-defect rates. All method failures/fallbacks remain separate.',
        'sha256':{str(p):digest(p) for p in inputs},'script_sha256':digest(Path(__file__))}
    for name,rr in [('A_rows',rows),('B_rows',br)]:
        with (a.out/(name+'.jsonl')).open('x') as f:
            for r in rr: f.write(json.dumps(r,allow_nan=False)+'\n')
    (a.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (a.out/'READY').write_text('all rows joined and paired objective summaries computed\n')
    print(json.dumps({'A':len(rows),'B':len(br),'gpu_process_hours':gpu_seconds/3600}))

if __name__=='__main__': main()
