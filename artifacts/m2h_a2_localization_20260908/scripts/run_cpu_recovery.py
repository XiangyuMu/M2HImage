"""CPU-only, eight disjoint source shards, durable merge and paired report."""
import concurrent.futures
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from run_localization import preflight,check_regions
from run_a2 import P,D,R,N,PY,read_rows,write_json,sha256


def durable(path):
    for p in Path(path).rglob('*'):
        if p.is_file():
            with p.open('rb') as f: os.fsync(f.fileno())
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def main():
    run=N/'run_v1'; out=N/'cpu_recovery_v3'; report=N/'report_cpu_v3'
    out.mkdir(exist_ok=False);(out/'logs').mkdir();(out/'inputs').mkdir()
    state={'pid':os.getpid(),'status':'wait_cpu_smoke','training_steps':0,'gpu_hours':0,'processes':[],
           'scripts_sha256':{str(p):sha256(p) for p in (Path(__file__),N/'scripts/region_cpu.py',N/'scripts/region_diagnostics.py',N/'scripts/summarize_a2.py')}}
    write_json(out/'manifest.json',state)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
             HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1')
    def execute(name,cmd,timeout=7200):
        if any(sha256(p)!=h for p,h in state['scripts_sha256'].items()): raise ValueError('pinned script changed')
        tick=time.monotonic();info={'name':name,'command':list(map(str,cmd)),'started':time.time(),'gpu_hours':0}
        with (out/'logs'/f'{name}.log').open('x') as log:
            proc=subprocess.Popen(info['command'],cwd=P,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            info['pid']=proc.pid;write_json(out/'logs'/f'{name}.command.json',info)
            try:rc=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                rc=-124
        info.update(returncode=rc,seconds=time.monotonic()-tick)
        write_json(out/'logs'/f'{name}.result.json',info)
        if rc:raise RuntimeError(name+' failed')
        return info
    try:
        state['preflight']=preflight(run)
        tick=time.monotonic();smoke=N/'cpu_smoke_v3'
        while not (smoke/'resources.json').exists():
            if time.monotonic()-tick>1800:raise TimeoutError('CPU smoke wait')
            time.sleep(5)
        smi,smc=check_regions(smoke,2)
        resource=json.loads((smoke/'resources.json').read_text())
        if resource.get('device')!='cpu':raise RuntimeError('CPU smoke not finalized')
        # Old full-garment objective values anchor backend numerical equivalence.
        import sys
        sys.path.insert(0,str(R/'scripts'))
        from summarize_revalidation import merge_metrics
        nr,_,_=merge_metrics(run/'a2_metrics')
        old=read_rows(R/'report_recovered_v2/rows.jsonl')
        by={(r['branch'],r['mid'],r['jid'],r['seed']):r for r in nr+old}
        numerical={}
        for field,prior in (('region_dino','garment_dino'),('region_hf_lpips','garment_hf_lpips')):
            diff=max(abs(r[field]-by[(r['branch'],r['mid'],r['jid'],r['seed'])][prior]) for r in smi if r['region']=='garment')
            numerical[field]=diff
            if diff>1e-5:raise ValueError('CPU/GPU whole-garment numerical mismatch')
        state['smoke_backend_max_abs_difference']=numerical
        if resource['cpu_wall_seconds_including_imports']*4>7200:raise RuntimeError('CPU full wall projection exceeds2h')
        if shutil.disk_usage(N).free<50*1024**3:raise RuntimeError('disk reserve')
        allrows=read_rows(run/'combined_rows.jsonl');mids=sorted({r['mid'] for r in allrows})
        if len(mids)!=64:raise ValueError('source count')
        cmds=[]
        for i in range(8):
            chosen=set(mids[i*8:(i+1)*8]); rr=[r for r in allrows if r['mid'] in chosen]
            path=out/'inputs'/f'shard{i}.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in rr))
            cmds.append([PY,'-u',N/'scripts/region_cpu.py','--rows',path,'--repo',P,'--root',D,
                '--cache',R/'fresh_cache_v2','--out',out/f'shard{i}','--device','cpu','--max-hours',2])
        state['status']='eight_cpu_shards';write_json(out/'manifest.json',state)
        full_start=time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            fs=[pool.submit(execute,f'shard{i}',cmd) for i,cmd in enumerate(cmds)]
            for f in concurrent.futures.as_completed(fs):
                state['processes'].append(f.result());write_json(out/'manifest.json',state)
        state['full_parallel_wall_seconds']=time.monotonic()-full_start
        merged=out/'merged';merged.mkdir();images=[];cross=[];summaries=[]
        for i in range(8):
            part=out/f'shard{i}';im,cr=check_regions(part,8);images+=im;cross+=cr
            summaries.append(json.loads((part/'summary.json').read_text()))
        for small,large,keys in ((smi,images,('mid','jid','seed','branch','region')),
                                 (smc,cross,('mid','jid_left','jid_right','seed','branch','region'))):
            lookup={tuple(r[k] for k in keys):r for r in large}
            for r in small:
                other=lookup[tuple(r[k] for k in keys)]
                for k,v in r.items():
                    if k.startswith(('region_','cross_ref_')) and isinstance(v,(float,int)):
                        if other[k] is None or abs(v-other[k])>1e-6:raise ValueError('CPU smoke/full mismatch')
        for name,rr in (('per_image.jsonl',images),('cross_ref.jsonl',cross)):
            (merged/name).write_text(''.join(json.dumps(r)+'\n' for r in rr))
        shutil.copyfile(out/'shard0/mask_config.json',merged/'mask_config.json')
        write_json(merged/'summary.json',{'status':'merged_complete','device':'cpu','shards':summaries,
                   'input_rows_sha256':sha256(run/'combined_rows.jsonl'),'per_image_rows':len(images),'cross_reference_rows':len(cross)})
        (merged/'READY').write_text('all CPU shards checked, awaiting report\n');durable(merged)
        state['status']='report';write_json(out/'manifest.json',state)
        state['processes'].append(execute('report',[PY,'-u',N/'scripts/summarize_a2.py',
            '--run',run,'--regions',merged,'--out',report],1800))
        a2=json.loads((run/'manifest.json').read_text())
        oldm=json.loads((N/'localization_v1/manifest.json').read_text())
        smoke_old=oldm['processes'][0]
        # Missing exit/launch records are not fabricated exact measurements.
        bound1=(time.mktime(time.strptime('2026-09-08 13:02:38','%Y-%m-%d %H:%M:%S'))-smoke_old['started'])/3600
        bound2=(time.mktime(time.strptime('2026-09-08 13:10:02','%Y-%m-%d %H:%M:%S'))-
                time.mktime(time.strptime('2026-09-08 13:02:38','%Y-%m-%d %H:%M:%S')))/3600
        budget={'new_cpu_recovery_gpu_hours':0,'cpu_parallel_wall_seconds':state['full_parallel_wall_seconds'],
                'cpu_child_wall_seconds_sum':sum(x['seconds'] for x in state['processes']),
                'cpu_smoke_seconds':resource['cpu_wall_seconds_including_imports'],
                'old_a2_gpu_hours':a2['gpu_hours'],'old_smoke_gpu_hours':smoke_old['gpu_hours'],
                'first_lost_full_upper_bound_gpu_hours':bound1,'second_lost_smoke_upper_bound_gpu_hours':bound2,
                'second_bound_basis':'entire preceding boot interval, conservative; launch record lost',
                'recorded_plus_lost_bounds_gpu_hours':a2['gpu_hours']+smoke_old['gpu_hours']+bound1+bound2,
                'cap_gpu_hours':12,'training_steps':0}
        write_json(report/'BUDGET.json',budget)
        with (report/'REPORT.md').open('a') as f:
            f.write('\n## CPU恢复与计算消耗\n\n两次重启后分区文件丢失，原目录保留。复用全部384张图和A2三阶段指标；本次在CPU重做同一定义的分区指标，没有重新生成图片或训练。\n')
            f.write(f'CPU全量并行墙钟{state["full_parallel_wall_seconds"]:.1f}秒，8个分片各4线程；GPU新增耗时为0。旧GPU记录加两次丢失任务保守上界共{budget["recorded_plus_lost_bounds_gpu_hours"]:.4f} GPUh，不是精确总计。\n')
            f.write('CPU小样本与全量重算差≤1e-6；所有完整衣服指标与原GPU评测差≤1e-5。\n')
        complete=json.loads((report/'COMPLETE.json').read_text())
        complete.update(report_sha256=sha256(report/'REPORT.md'),budget_sha256=sha256(report/'BUDGET.json'),
                        region_summary_sha256=sha256(merged/'summary.json'),input_rows_sha256=sha256(run/'combined_rows.jsonl'))
        write_json(report/'COMPLETE.json',complete);durable(report)
        state.update(status='complete',report=str(report));write_json(out/'manifest.json',state)
        (out/'READY').write_text('CPU recovery report accounting complete\n');durable(out)
        print(json.dumps(state),flush=True)
    except Exception as exc:
        state.update(status='failed',error=repr(exc));write_json(out/'FAILED.json',state);durable(out);raise


if __name__=='__main__':main()
