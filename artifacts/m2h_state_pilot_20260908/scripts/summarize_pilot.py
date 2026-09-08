"""Three-arm paired pilot evidence; never selects a checkpoint or drops failed detections."""
import argparse
import json
from pathlib import Path
import sys
from prepare_pool import sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--batch',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(a.repo/'artifacts/m2h_minimal_revalidation_20260907/scripts'))
    import summarize_revalidation as s
    s.ARMS=('C-H','C-perm','E-match')
    pairpath=a.repo/'artifacts/m2h_ab_objective_v1/data_protocol_v2/dev128_pairs.json'
    pairset,by_pair,_=s.load_manifest_pairs(pairpath)
    rows=[];statuses={};inputs={}
    for arm in s.ARMS:
        run=a.batch/'dev'/arm
        generated=s.read_jsonl(run/'rows.jsonl')
        assert len(generated)==128
        for r in generated:
            assert sha(r['path'])==r['sha256']
        merged,stat,paths=s.merge_metrics(run)
        assert {s.full_key(r) for r in generated}=={s.full_key(r) for r in merged}
        rows.extend(merged);statuses[arm]=stat
        inputs.update({str(x):sha(x) for x in paths+[run/'rows.jsonl']})
    s.attach_manifest(rows,by_pair)
    s.validate_rows(rows,pairset)
    byarm={arm:[r for r in rows if r['branch']==arm] for arm in s.ARMS}
    effects={ctrl:{field:s.paired_delta(byarm['E-match'],byarm[ctrl],field,draws=10000,seed=20260908)
                   for field in s.METRIC_FIELDS} for ctrl in ('C-H','C-perm')}
    summary=dict(status='complete',n_pairs=128,images=384,training_seeds=1,means=s.means(rows),effects=effects,status_counts=statuses,
                 sources=inputs,dev_pairs_sha256=sha(pairpath),
                 interpretation='Exploratory source-group/reference-file paired bootstrap; not training-seed uncertainty, not formal CVPR evidence. No extra stages authorized.')
    out=a.batch/'report';out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    s.write_jsonl(out/'per_image.jsonl',rows)
    lines=['# M2H三臂短训练pilot','', '同一A2初始化、同一步数；仅CF状态来源/身份对应改变。一次训练seed，开发集结果，非独立最终检验。','',
           '| E-match相对控制 | 指标 | 差值 | 探索性95%区间 |','|---|---|---:|---|']
    for arm,fields in effects.items():
        for field,v in fields.items():
            lines.append(f"| {arm} | {field} | {v['delta_mean']:.6f} | {v['ci95']} |")
    lines+=['','身份越高越好，DINO越高越好，HF/身体/头部距离越低越好。检测失败保留在分母。',
            '不依据此报告自动启动第二seed、600步或B训练。历史A2/A4仅背景，不能代替新等步控制。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'READY').write_text('Three arms paired dev metrics and384 hashes verified\n')


if __name__=='__main__':
    main()
