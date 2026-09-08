"""Bounded A-D0 -> objective metrics -> shared-teacher B control supervisor.

No parameter training. Fresh directories only. Child failures terminate their
lane and forbid the dependent B comparison. Original datasets are read-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

PY='/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python'
POSEPY='/home/muxiangyu/miniconda3/envs/StableAnimator/bin/python'

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run_stage(cmd, log, gpu, timeout):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1',
         'TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    record={'command':list(map(str,cmd)),'physical_gpu':gpu,'started':time.time(),'timeout_seconds':timeout}
    log.with_suffix('.command.json').write_text(json.dumps(record,indent=2))
    with log.open('x') as handle:
        result=subprocess.run(record['command'],env=env,stdout=handle,stderr=subprocess.STDOUT,timeout=timeout)
    record.update(returncode=result.returncode,seconds=time.time()-record['started'])
    log.with_suffix('.exit.json').write_text(json.dumps(record,indent=2))
    if result.returncode: raise RuntimeError(f'Failed {log}: exit {result.returncode}')

def verify_metrics(run):
    for stage in ('image','id','pose'):
        folder=run/('metrics_'+stage)
        if not (folder/'READY').is_file(): raise RuntimeError('metric missing: '+str(folder))
        s=json.loads((folder/'summary.json').read_text())
        statuses=[r.get('metric_status') for r in map(json.loads,(folder/'per_image.jsonl').read_text().splitlines())]
        if len(statuses)!=s['expected_rows'] or any(v in ('failed','reference_failed') for v in statuses):
            raise RuntimeError('Metric infrastructure/reference failure: '+str(folder))
        # Output face/pose failures remain valid failure-penalized observations.

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    logs=a.out/'logs'; logs.mkdir()
    scripts=a.artifacts/'scripts'; controls=a.artifacts/'scripts_controls_v1'
    audit=json.loads((a.cache/'audit.json').read_text()); root=Path(audit['config']['data']['root'])
    if len(audit['pairs'])!=128: raise RuntimeError('Expected frozen dev128')
    inputs=[scripts/'a_d0_probe.py',scripts/'a_d0_metrics_v2.py',controls/'b_alignment_boundary_v2.py',scripts/'b_metric_view.py',a.cache/'audit.json']
    (a.out/'manifest.json').write_text(json.dumps({'training_steps':0,'formal_result':False,
        'scripts_sha256':{str(f):digest(f) for f in inputs},'pid':os.getpid(),'started':time.time(),
        'gpu_partition':[0,1,2,3],'cache':str(a.cache),'method_changes':'none after smoke'},indent=2))
    start=time.time()
    def metric_passes(run,gpu,cache):
        for stage in ('image','id','pose'):
            cmd=[PY if stage=='image' else POSEPY,'-B','-u',scripts/'a_d0_metrics_v2.py',
                '--repo',a.repo,'--cache',cache,'--run',run,'--stage',stage,'--device','cuda:0']
            run_stage(cmd,logs/(run.name+'_'+stage+'.log'),gpu,3600)
        verify_metrics(run)
    def lane(gpu):
        run=a.out/f'shard{gpu}'
        run_stage([PY,'-B','-u',scripts/'a_d0_probe.py','--repo',a.repo,'--cache',a.cache,
            '--out',run,'--device','cuda:0','--shards','4','--shard',str(gpu),'--max-hours','6'],
            logs/f'shard{gpu}_generation.log',gpu,22200)
        metric_passes(run,gpu,a.cache)
        return run
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            runs=list(pool.map(lane,range(4)))
        rows=[json.loads(line) for r in runs for line in (r/'rows.jsonl').read_text().splitlines()]
        if len(rows)!=1296 or len({(r['mid'],r['jid'],r['tau'],r['branch'],r['k']) for r in rows})!=1296:
            raise RuntimeError('A row count/deduplication mismatch')
        probe=a.out/'b_probe'; probe.mkdir(); (probe/'outputs').mkdir()
        (probe/'masks').symlink_to(a.cache/'masks',target_is_directory=True)
        (probe/'audit.json').write_text(json.dumps(audit,indent=2))
        for r in runs:
            for image in (r/'teacher').glob('*.png'):
                (probe/'outputs'/image.name).symlink_to(image.resolve())
        if len(list((probe/'outputs').glob('*.png')))!=128: raise RuntimeError('Teacher count mismatch')
        b=a.out/'b_alignment128'
        run_stage([PY,'-B','-u',controls/'b_alignment_boundary_v2.py','--root',root,'--probe',probe,
            '--out',b,'--threads','2','--max-seconds','3600'],logs/'b_alignment128.log','',3900)
        view=a.out/'b_metrics'
        run_stage([PY,'-B',scripts/'b_metric_view.py','--b-run',b,'--probe',probe,'--out',view],logs/'b_view.log','',120)
        metric_passes(view,0,view)
        summary={'status':'A_D0_and_B_objective_complete','A_predictions':len(rows),'B_pairs':128,
                 'seconds':time.time()-start,'formal_split_ready':False,'training_steps':0}
        (a.out/'summary.json').write_text(json.dumps(summary,indent=2))
        (a.out/'READY').write_text('generation and all objective passes complete, formal protocol remains separate\n')
    except Exception as exc:
        (a.out/'FAILED.json').write_text(json.dumps({'error':str(exc),'seconds':time.time()-start},indent=2))
        raise

if __name__=='__main__': main()
