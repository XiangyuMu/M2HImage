"""Recover region-only outputs after reboot; never regenerate model images."""
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from run_localization import preflight, check_regions
from run_a2 import P,D,R,N,PY,read_rows,write_json,sha256


def main():
    run=N/'run_v1'; out=N/'localization_recovery_v2'; report=N/'report_recovered_v2'
    out.mkdir(exist_ok=False); (out/'logs').mkdir()
    old=json.loads((N/'localization_v1/manifest.json').read_text())
    state={'pid':os.getpid(),'status':'preflight','training_steps':0,'processes':[],
           'scripts_sha256':{str(p):sha256(p) for p in (Path(__file__),N/'scripts/region_diagnostics.py',N/'scripts/summarize_a2.py')}}
    write_json(out/'manifest.json',state)
    env=dict(os.environ,HF_HOME='/data/muxiangyu/modelLibrary',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
             NO_ALBUMENTATIONS_UPDATE='1',CUDA_VISIBLE_DEVICES='0')
    def execute(name,cmd,gpu=True,timeout=7200):
        info={'name':name,'command':list(map(str,cmd)),'started':time.time(),'gpu':0 if gpu else None}
        tick=time.monotonic()
        with (out/'logs'/f'{name}.log').open('x') as log:
            proc=subprocess.Popen(info['command'],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            info['pid']=proc.pid; write_json(out/'logs'/f'{name}.command.json',info)
            try: rc=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                rc=-124
        info.update(returncode=rc,seconds=time.monotonic()-tick,gpu_hours=(time.monotonic()-tick)/3600 if gpu else 0)
        state['processes'].append(info);write_json(out/'logs'/f'{name}.result.json',info)
        write_json(out/'manifest.json',state)
        if rc: raise RuntimeError(name+' failed')
        return info
    try:
        state['preflight']=preflight(run)
        import shutil
        if shutil.disk_usage(N).free<50*1024**3: raise RuntimeError('disk reserve')
        for path,expected in old['scripts_sha256'].items():
            if sha256(path)!=expected: raise ValueError('old script hash changed: '+path)
        a2=json.loads((run/'manifest.json').read_text())
        # Lost full-pass launch record: the preserved old smoke launch is earlier
        # than full launch. Bound lost full task by that launch through boot.
        boot=time.mktime(time.strptime('2026-09-08 13:02:38','%Y-%m-%d %H:%M:%S'))
        smoke_old=old['processes'][0]
        lost_bound=max(0,boot-smoke_old['started'])/3600
        base_cost=a2['gpu_hours']+smoke_old['gpu_hours']+lost_bound
        if base_cost+4>12: raise RuntimeError('remaining budget insufficient')
        cmd=[PY,'-u',N/'scripts/region_diagnostics.py','--rows',run/'combined_rows.jsonl',
             '--repo',P,'--root',D,'--cache',R/'fresh_cache_v2','--device','cuda:0','--max-hours',2]
        state['status']='smoke';write_json(out/'manifest.json',state)
        smoke=execute('smoke_first2M',cmd+['--out',out/'smoke_first2M','--limit-mids',2])
        smi,smc=check_regions(out/'smoke_first2M',2)
        if base_cost+smoke['gpu_hours']*33>12: raise RuntimeError('smoke budget projection')
        state['status']='full_regions';write_json(out/'manifest.json',state)
        execute('full64M',cmd+['--out',out/'full64M'])
        fi,fc=check_regions(out/'full64M',64)
        for small,large,keys in ((smi,fi,('mid','jid','seed','branch','region')),
                                 (smc,fc,('mid','jid_left','jid_right','seed','branch','region'))):
            by={tuple(r[k] for k in keys):r for r in large}
            for r in small:
                v=by[tuple(r[k] for k in keys)]
                for k,x in r.items():
                    if k.startswith(('region_','cross_ref_')) and isinstance(x,(int,float)):
                        if v[k] is None or abs(x-v[k])>1e-6: raise ValueError('smoke/full inconsistency')
        state['status']='report';write_json(out/'manifest.json',state)
        execute('report',[PY,'-u',N/'scripts/summarize_a2.py','--run',run,'--regions',out/'full64M','--out',report],False,1800)
        budget={'a2_and_ordinary_gpu_hours':a2['gpu_hours'],'old_smoke_gpu_hours':smoke_old['gpu_hours'],
                'lost_full_gpu_hours_upper_bound':lost_bound,'lost_bound_basis':'old smoke launch to reboot13:02:38+08; full started later',
                'recovery_processes':state['processes'],'recorded_plus_lost_bound_gpu_hours':base_cost+sum(x['gpu_hours'] for x in state['processes']),
                'cap_gpu_hours':12,'training_steps':0,'scope':'new A2 batch; old256 images reused, no image regeneration'}
        if budget['recorded_plus_lost_bound_gpu_hours']>12: raise RuntimeError('budget exceeded')
        write_json(report/'BUDGET.json',budget)
        with (report/'REPORT.md').open('a') as f:
            f.write(f'\n## 重启恢复及计算消耗\n\n384张图及A2三阶段指标重新校验完整；旧分区结果为零字节，保留原目录，重算分区。未重新生成图片。\n')
            f.write(f'记录GPU墙钟加丢失任务保守上界共{budget["recorded_plus_lost_bound_gpu_hours"]:.4f} GPUh；不是全项精确计时，上限12 GPUh。\n')
        c=json.loads((report/'COMPLETE.json').read_text())
        c.update(report_sha256=sha256(report/'REPORT.md'),budget_sha256=sha256(report/'BUDGET.json'),
                 region_summary_sha256=sha256(out/'full64M/summary.json'),input_rows_sha256=sha256(run/'combined_rows.jsonl'))
        write_json(report/'COMPLETE.json',c)
        state.update(status='complete',report=str(report));write_json(out/'manifest.json',state)
        (out/'READY').write_text('regions report and accounting complete; no training\n')
        print(json.dumps(state),flush=True)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));write_json(out/'FAILED.json',state);raise


if __name__=='__main__': main()
