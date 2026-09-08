"""Strict paired exploratory analysis; no metric-selected sample deletion."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ARMS = ('b2_inputOnly_freshI','a2_inputOnly_freshI','a4_inputOnly_freshI')
REGIONS = ('garment','interior','boundary','hightexture','lowtexture')
FIELDS = ('id_penalized','garment_dino','garment_hf_lpips','body_penalized','head_penalized')
P = Path('/data/muxiangyu/pythonPrograms/M2HImage')
R = P/'artifacts/m2h_minimal_revalidation_20260907'


def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def key(r): return (r['mid'],r['jid'],r['seed'])
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def validate(rows, expected):
    if set(r['branch'] for r in rows)!=set(ARMS): raise ValueError('arm mismatch')
    for arm in ARMS:
        rr=[r for r in rows if r['branch']==arm]
        kk=[key(r) for r in rr]
        if len(kk)!=len(set(kk)) or set(kk)!=set(expected): raise ValueError('missing/duplicate pairs')
        if any(r['tau']!=0 or r['k']!=1 for r in rr): raise ValueError('solver key mismatch')


def weights(rows, draws=10000, seed=20260908, cross=False):
    rng=np.random.default_rng(seed)
    sources=sorted({str(r.get('m_source_group',r['mid'])) for r in rows})
    refs=sorted({str(r[n]) for r in rows for n in (('jid_left','jid_right') if cross else ('jid',))})
    sm={x:i for i,x in enumerate(sources)}; rm={x:i for i,x in enumerate(refs)}
    sw=rng.multinomial(len(sources),np.full(len(sources),1/len(sources)),size=draws)
    rw=rng.multinomial(len(refs),np.full(len(refs),1/len(refs)),size=draws)
    w=sw[:,[sm[str(r.get('m_source_group',r['mid']))] for r in rows]].astype(float)
    for name in (('jid_left','jid_right') if cross else ('jid',)):
        w*=rw[:,[rm[str(r[name])] for r in rows]]
    return w


def estimate(values,w):
    values=np.asarray(values,float)
    if values.size==0 or not np.isfinite(values).all(): raise ValueError('empty/nonfinite metric')
    den=w.sum(axis=1); valid=den>0
    samples=(w[valid]@values)/den[valid]
    return {'n':len(values),'mean':float(values.mean()),
            'ci95':np.quantile(samples,[.025,.975]).tolist(),'bootstrap_nonempty':int(valid.sum())}


def effects(rows,fields,cross=False):
    k=(lambda r:(r['mid'],r['jid_left'],r['jid_right'],r['seed'])) if cross else key
    by={a:{k(r):r for r in rows if r['branch']==a} for a in ARMS}
    kk=sorted(by[ARMS[0]])
    if not kk or any(set(x)!=set(kk) for x in by.values()): raise ValueError('unpaired effect')
    w=weights([by[ARMS[0]][x] for x in kk],cross=cross)
    result={'means':{a:{} for a in ARMS},'contrasts':{}}
    for f in fields:
        for a in ARMS:
            vals=[by[a][x][f] for x in kk]
            if any(v is None or not np.isfinite(v) for v in vals): raise ValueError('metric missing '+f)
            result['means'][a][f]=float(np.mean(vals))
    for left,right in ((ARMS[1],ARMS[0]),(ARMS[2],ARMS[1]),(ARMS[2],ARMS[0])):
        name=left.split('_')[0]+'_minus_'+right.split('_')[0]
        result['contrasts'][name]={f:estimate([by[left][x][f]-by[right][x][f] for x in kk],w) for f in fields}
    return result


def region_effects(rows,cross=False):
    result={}
    for region in REGIONS:
        prefix=region+('_pair_' if cross else '_')
        ef=prefix+'eligible'
        kk=(lambda r:r['mid']) if cross else key
        by={a:{kk(r):r for r in rows if r['branch']==a} for a in ARMS}
        common=set(by[ARMS[0]])
        if any(set(v)!=common for v in by.values()): raise ValueError('region rows mismatch')
        for x in common:
            if len({(by[a][x][ef],by[a][x][prefix+'support']) for a in ARMS})!=1:
                raise ValueError('output-dependent source eligibility')
        eligible={x for x in common if by[ARMS[0]][x][ef]}
        suffixes=('hf_lpips','pixel_mae','gradient_mae') if cross else ('source_dino','source_hf_lpips','source_pixel_mae','source_gradient_mae')
        fields=[prefix+s for s in suffixes]
        selected=[r for r in rows if kk(r) in eligible]
        result[region]={'source_eligible_per_arm':len(eligible),'source_ineligible_per_arm':len(common)-len(eligible),
                        'statistics':effects(selected,fields,cross) if eligible else None}
    return result


def widen_region_rows(rows, cross=False):
    """Validate all five long-form region records before paired aggregation."""
    grouped={}; seen={}
    names=('mid','branch','tau','k','seed','jid_left','jid_right','path_left','path_right') if cross else ('mid','jid','branch','tau','k','seed','path')
    suffixes=('hf_lpips','pixel_mae','gradient_mae') if cross else ('dino','hf_lpips','pixel_mae','gradient_mae')
    for r in rows:
        k=tuple(r[n] for n in names)
        region=r['region']
        if region not in REGIONS or region in seen.setdefault(k,set()): raise ValueError('duplicate/unknown region')
        seen[k].add(region)
        if r['metric_status']!='ok': raise RuntimeError('region failure cannot be dropped')
        out=grouped.setdefault(k,{**{n:r[n] for n in names},'metric_status':'ok'})
        prefix=region+('_pair_' if cross else '_')
        out[prefix+'support']=r['support']; out[prefix+'eligible']=r['eligible']
        if bool(r['eligible']) != (r['support']>0): raise ValueError('eligibility/support contradiction')
        for f in suffixes:
            v=r[('cross_ref_' if cross else 'region_')+f]
            if r['eligible'] and (v is None or not np.isfinite(v)): raise ValueError('eligible metric missing')
            if not r['eligible'] and v is not None: raise ValueError('ineligible metric should be null')
            out[prefix+('' if cross else 'source_')+f]=v
    if any(v!=set(REGIONS) for v in seen.values()): raise ValueError('missing region record')
    return list(grouped.values())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--regions',type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    sys.path.insert(0,str(R/'scripts'))
    from summarize_revalidation import merge_metrics
    if not (a.run/'A2_READY').is_file(): raise RuntimeError('A2 not ready')
    a.out.mkdir(parents=True,exist_ok=False)
    paths=[R/'report_recovered_v2/rows.jsonl',R/'protocol_v1/manifest.json',a.run/'combined_rows.jsonl']
    old=read(paths[0]); new,status,metricpaths=merge_metrics(a.run/'a2_metrics'); paths+=metricpaths
    rows=new+[r for r in old if r['branch'] in (ARMS[0],ARMS[2])]
    pairs_path=P/'artifacts/m2h_ab_objective_v1/data_protocol_v2/dev128_pairs.json'
    paths.append(pairs_path)
    pairs=json.loads(pairs_path.read_text())['pairs']
    meta={key(r):r for r in pairs}
    validate(rows,meta)
    generated=read(a.run/'combined_rows.jsonl')
    validate(generated,meta)
    gen={(r['branch'],*key(r)):r for r in generated}
    for r in rows:
        g=gen[(r['branch'],*key(r))]
        if r['path']!=g['path'] or digest(g['path'])!=g['sha256']: raise ValueError('PNG/source-row mismatch')
        r['m_source_group']=meta[key(r)].get('m_source_group',r['mid'])
    protocol=json.loads(paths[1].read_text())
    sensitive={key(pairs[i]) for i in protocol['sensitivity_pair_indices']}
    subsets={'all128':rows,'sensitivity101':[r for r in rows if key(r) in sensitive]}
    summary={'status':'ordinary_complete_regions_pending','training_steps':0,
             'ordinary':{name:effects(rr,FIELDS) for name,rr in subsets.items()},
             'a2_status_counts':status,
             'statistics_note':'Exploratory10000 source-group x reference-file pigeonhole draws seed20260908; no training-seed claim. Cross-reference pairs share one reference resampling pool and use counts for both incident reference files.',
             'region_interpretation':'Compare arms WITHIN each fixed source region; absolute scores across different masks are not comparable. Sparse masks/resize can influence HF-LPIPS. Native gradient MAE corroborates but is also alignment-sensitive.',
             'limitations':['A2/A4 differ in identity loss and sampling/bank details; not an isolated identity-loss causal ablation.',
                            'No G2/manual annotations. Hightexture is a source-gradient proxy, not text/print truth.',
                            'Near-duplicate sensitivity101 is not semantic-clean certification; identity overlap alone is not answer leakage.',
                            'Lower cross-reference clothing drift can arise from ignoring identity: inspect identity fidelity alongside it.']}
    if a.regions:
        if not (a.regions/'READY').is_file(): raise RuntimeError('regions not ready')
        ip=a.regions/'per_image.jsonl'; cp=a.regions/'cross_ref.jsonl'
        paths += [ip,cp,a.regions/'summary.json',a.regions/'mask_config.json']
        raw_ir=read(ip); raw_cr=read(cp)
        ir=widen_region_rows(raw_ir); cr=widen_region_rows(raw_cr,True)
        validate(ir,meta)
        if len(cr)!=192 or len({(r['mid'],r['branch']) for r in cr})!=192:
            raise ValueError('cross-ref row coverage')
        if any(r['metric_status']!='ok' for r in ir+cr): raise RuntimeError('region failures remain; no successful-only report')
        mids_to_pairs={mid:{x for x in meta if x[0]==mid} for mid,_,_ in meta}
        for r in ir:
            if r['path']!=gen[(r['branch'],*key(r))]['path']: raise ValueError('regional image path mismatch')
            r['m_source_group']=meta[key(r)].get('m_source_group',r['mid'])
        for r in cr:
            refkeys={(r['mid'],r['jid_left'],r['seed']),(r['mid'],r['jid_right'],r['seed'])}
            if refkeys!=mids_to_pairs[r['mid']]: raise ValueError('cross reference mismatch')
            r['m_source_group']=meta[next(iter(refkeys))].get('m_source_group',r['mid'])
        maxdiff={}
        ordinary_by={(r['branch'],*key(r)):r for r in rows}
        for regional,ordinary in (('garment_source_dino','garment_dino'),('garment_source_hf_lpips','garment_hf_lpips')):
            maxdiff[regional]=max(abs(r[regional]-ordinary_by[(r['branch'],*key(r))][ordinary]) for r in ir)
            tolerance = 1e-5 if ordinary == 'garment_dino' else 1e-3
            if maxdiff[regional]>tolerance: raise ValueError('CPU/GPU numerical discrepancy exceeds disclosed bound')
        smids={m for m,kk in mids_to_pairs.items() if kk <= sensitive}
        summary['regions']={
            'all128':region_effects(ir),
            'sensitivity101':region_effects([r for r in ir if key(r) in sensitive]),
        }
        summary['cross_reference']={
            'all64':region_effects(cr,True),
            'sensitivity_complete_M_pairs':region_effects([r for r in cr if r['mid'] in smids],True),
        }
        summary['whole_garment_reproduction_max_abs_diff']=maxdiff
        summary['regional_backend']={'device':'cpu','comparison':'all three arms recomputed on CPU; ordinary table retains original GPU scores','dino_tolerance':1e-5,'hf_lpips_tolerance':1e-3,'tolerance_note':'LPIPS tolerance revised after fixed2M CPU smoke, before full run, to bound backend discrepancy; not a method-success threshold.'}
        summary.update(status='complete',regional_images=len(ir),cross_reference_pairs=len(cr),
                       regional_rows=len(raw_ir),cross_reference_rows=len(raw_cr))
    summary['input_sha256']={str(path):digest(path) for path in paths}
    save(a.out/'summary.json',summary)
    lines=['# A2补测与服装定位报告','',
        '本批无参数训练。新增A2 128张，复用B2/A4各128张；同一纯M/I输入协议。',
        '区间为探索性来源组×参考文件bootstrap，不代表训练seed不确定性。','',
        '## 整体客观指标（128对）','',
        '| 模型 | ID↑ | garment DINO↑ | HF-LPIPS↓ | body↓ | head↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        mm=summary['ordinary']['all128']['means'][arm]
        lines.append('| '+arm.split('_')[0].upper()+' | '+' | '.join(f'{mm[f]:.6f}' for f in FIELDS)+' |')
    for name,eff in summary['ordinary']['all128']['contrasts'].items():
        lines+=['',f'### {name}','']
        for f,v in eff.items(): lines.append(f'- {f}: {v["mean"]:+.6f}, 95%CI [{v["ci95"][0]:+.6f}, {v["ci95"][1]:+.6f}]')
    if a.regions:
        lines+=['','## 分区：A4−A2（同一区域内配对，不比较不同区域绝对分数）','',
                '| 区域 | n | DINO差 | HF-LPIPS差 [95%CI] | 原分辨率梯度MAE差 |',
                '|---|---:|---:|---:|---:|']
        for region,v in summary['regions']['all128'].items():
            if not v['statistics']: continue
            st=v['statistics']['contrasts']['a4_minus_a2']
            d=st[region+'_source_dino']; h=st[region+'_source_hf_lpips']; g=st[region+'_source_gradient_mae']
            lines.append(f'| {region} | {h["n"]} | {d["mean"]:+.6f} | {h["mean"]:+.6f} [{h["ci95"][0]:+.6f}, {h["ci95"][1]:+.6f}] | {g["mean"]:+.6f} |')
        lines+=['','## 换参考图时的服装变化（64个人台，每M两参考）','',
                '| 区域 | B2 HF-LPIPS↓ | A2↓ | A4↓ |','|---|---:|---:|---:|']
        for region,v in summary['cross_reference']['all64'].items():
            if v['statistics']:
                mm=v['statistics']['means']; f=region+'_pair_hf_lpips'
                lines.append('| '+region+' | '+' | '.join(f'{mm[arm][f]:.6f}' for arm in ARMS)+' |')
        lines+=['','全量和101对敏感性分区、三组完整差值及区间见summary.json。',
                '分区统一使用CPU；完整衣服DINO与旧GPU差≤1e-5、HF-LPIPS差≤1e-3，实际差值见summary.json，不宣称逐值相等。']
    lines+=['','## 解释边界','']+['- '+x for x in summary['limitations']]
    lines+=['- 边界梯度不等于自然度；分区分数也不能证明梯度冲突或定位具体训练原因。',
            '- 本批不据分数自动启动训练；原始数据划分和历史结果不变。']
    (a.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    if a.regions:
        save(a.out/'COMPLETE.json',{'status':'complete','training_steps':0,'new_predictions':128,
             'reused_predictions':256,'regional_images':384,'cross_reference_pairs':192,
             'regional_rows':1920,'cross_reference_rows':960,
             'summary_sha256':digest(a.out/'summary.json'),'report_sha256':digest(a.out/'REPORT.md')})
    (a.out/'READY').write_text(summary['status']+'\n')
    print(json.dumps({'status':summary['status'],'out':str(a.out)}))


if __name__=='__main__': main()

