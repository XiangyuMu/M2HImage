"""Finish only current authorized batch: bounded region smoke/full/report."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from run_a2 import P,D,R,N,OLD,PY,read_rows,write_json,sha256


def preflight(run):
    """A2_READY is provisional; independently validate rows before spending more."""
    from summarize_a2 import validate, key, ARMS, FIELDS
    pairs=json.loads((OLD/'data_protocol_v2/dev128_pairs.json').read_text())['pairs']
    expected={key(r) for r in pairs}
    generated=read_rows(run/'combined_rows.jsonl')
    validate(generated,expected)
    by={(r['branch'],*key(r)):r for r in generated}
    for r in generated:
        if sha256(r['path'])!=r['sha256']: raise ValueError('preflight image hash')
    sys.path.insert(0,str(R/'scripts'))
    from summarize_revalidation import merge_metrics, canonical_metric_row
    for stage in ('image','id','pose'):
        rows=read_rows(run/'a2_metrics'/f'metrics_{stage}/per_image.jsonl')
        kk=[key(r) for r in rows]
        if len(kk)!=128 or set(kk)!=expected: raise ValueError('metric pair keys')
        for r in rows:
            canonical_metric_row(r)
            if r['branch']!='a2_inputOnly_freshI' or r['tau']!=0 or r['k']!=1:
                raise ValueError('metric arm/solver')
            if r['path']!=by[(r['branch'],*key(r))]['path']: raise ValueError('metric path mismatch')
    merged,_,_=merge_metrics(run/'a2_metrics')
    import math
    if any(r.get(f) is None or not math.isfinite(r[f]) for r in merged for f in FIELDS):
        raise ValueError('missing ordinary metric')
    return {'generated128_plus_reused256_verified':True,'a2_three_stage_keys_paths_verified':True}


def check_regions(out,n):
    if not (out/'READY').is_file(): raise RuntimeError('regions missing READY')
    image=read_rows(out/'per_image.jsonl'); cross=read_rows(out/'cross_ref.jsonl')
    if len(image)!=n*6*5 or len(cross)!=n*3*5: raise ValueError('region coverage count')
    if any(r['metric_status']!='ok' for r in image+cross): raise RuntimeError('region failures recorded')
    return image,cross


def main():
    run=N/'run_v1'; out=N/'localization_v1'; out.mkdir(exist_ok=False)
    (out/'logs').mkdir()
    started=time.monotonic()
    state={'pid':os.getpid(),'status':'waiting_A2','training_steps':0,'processes':[],
           'scripts_sha256':{str(x):sha256(x) for x in (Path(__file__),N/'scripts/region_diagnostics.py',N/'scripts/summarize_a2.py')}}
    write_json(out/'manifest.json',state)
    env=dict(os.environ,HF_HOME='/data/muxiangyu/modelLibrary',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
             NO_ALBUMENTATIONS_UPDATE='1',CUDA_VISIBLE_DEVICES='0')
    def execute(name,cmd,gpu=True,timeout=7200):
        if shutil.disk_usage(N).free<50*1024**3: raise RuntimeError('storage reserve')
        if any(sha256(path)!=value for path,value in state['scripts_sha256'].items()):
            raise RuntimeError('script changed after launch')
        info={'name':name,'command':list(map(str,cmd)),'started':time.time(),'gpu':0 if gpu else None}
        tick=time.monotonic()
        with (out/'logs'/f'{name}.log').open('x') as log:
            child=subprocess.Popen(info['command'],env=env,cwd=P,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            info['pid']=child.pid; write_json(out/'logs'/f'{name}.command.json',info)
            try: rc=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL); child.wait()
                rc=-124
        info.update(returncode=rc,seconds=time.monotonic()-tick,gpu_hours=(time.monotonic()-tick)/3600 if gpu else 0)
        state['processes'].append(info)
        write_json(out/'logs'/f'{name}.result.json',info); write_json(out/'manifest.json',state)
        if rc: raise RuntimeError(name+' failed')
        return info
    try:
        while not (run/'A2_READY').exists():
            if (run/'FAILED.json').exists(): raise RuntimeError('A2 generation/metrics failed')
            if time.monotonic()-started>7500: raise TimeoutError('A2 wait exceeded')
            time.sleep(10)
        a2=json.loads((run/'manifest.json').read_text())
        state['preflight']=preflight(run)
        base=[PY,'-u',N/'scripts/region_diagnostics.py','--rows',run/'combined_rows.jsonl',
              '--repo',P,'--root',D,'--cache',R/'fresh_cache_v2','--device','cuda:0','--max-hours',2]
        state['status']='region_smoke'; write_json(out/'manifest.json',state)
        smoke=execute('smoke_first2M',base+['--out',out/'smoke_first2M','--limit-mids',2])
        smi,smc=check_regions(out/'smoke_first2M',2)
        # Conservative linear32x whole smoke cost, including smoke startup.
        projection=a2['gpu_hours']+smoke['gpu_hours']*(1+32)
        state['projected_total_gpu_hours']=projection
        if projection>12: raise RuntimeError('smoke projection exceeds12GPUh')
        state['status']='region_full'; write_json(out/'manifest.json',state)
        full=execute('full64M',base+['--out',out/'full64M'])
        fi,fc=check_regions(out/'full64M',64)
        # Recomputed smoke entries must agree with full pass on the same inputs.
        for small,large,keys in ((smi,fi,('mid','jid','seed','branch','region')),
                                 (smc,fc,('mid','jid_left','jid_right','seed','branch','region'))):
            by={tuple(r[k] for k in keys):r for r in large}
            for r in small:
                v=by[tuple(r[k] for k in keys)]
                for k,x in r.items():
                    if k in ('region_dino','region_hf_lpips','region_pixel_mae','region_gradient_mae',
                             'cross_ref_hf_lpips','cross_ref_pixel_mae','cross_ref_gradient_mae'):
                        if (x is None)!=(v[k] is None) or x is not None and abs(x-v[k])>1e-6:
                            raise ValueError('smoke/full metric inconsistency')
        state['status']='report'; write_json(out/'manifest.json',state)
        report=N/'report_v1'
        execute('report',[PY,'-u',N/'scripts/summarize_a2.py','--run',run,'--regions',out/'full64M','--out',report],False,1800)
        budget={'generation_and_ordinary_metrics':a2['processes'],'localization_and_report':state['processes'],
                'total_new_gpu_hours':a2['gpu_hours']+sum(x['gpu_hours'] for x in state['processes']),
                'scope':'new batch only, process wall including startup; old256 images/metrics reused, original costs not recharged',
                'cap_gpu_hours':12,'training_steps':0}
        if budget['total_new_gpu_hours']>12: raise RuntimeError('actual budget exceeded')
        write_json(report/'BUDGET.json',budget)
        with (report/'REPORT.md').open('a') as f:
            f.write(f'\n## 执行核验与计算消耗\n\n新增批次 GPU 子进程墙钟（含启动）合计{budget["total_new_gpu_hours"]:.4f} GPUh，上限12 GPUh。CPU汇总另计。\n')
            f.write('128张A2及复用256张PNG哈希核验通过；区域小样本与全量重算一致。无参数训练。\n')
        complete=json.loads((report/'COMPLETE.json').read_text())
        complete.update(budget_sha256=sha256(report/'BUDGET.json'),report_sha256=sha256(report/'REPORT.md'),
                        region_summary_sha256=sha256(out/'full64M/summary.json'),
                        input_rows_sha256=sha256(run/'combined_rows.jsonl'))
        write_json(report/'COMPLETE.json',complete)
        state.update(status='complete',report=str(report),new_gpu_hours=budget['total_new_gpu_hours'])
        write_json(out/'manifest.json',state)
        (out/'READY').write_text('A2 regions statistics accounting complete; no new training\n')
        print(json.dumps(state),flush=True)
    except Exception as exc:
        state.update(status='failed',error=repr(exc)); write_json(out/'FAILED.json',state); raise


if __name__=='__main__': main()
