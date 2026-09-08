"""Single owned queue: fixed teacher + C-H health, three arms, frozen dev evaluation.

No restart loop, no quality-based teacher filtering, no second seed/extension.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from prepare_pool import sha

P=Path('/data/muxiangyu/pythonPrograms/M2HImage')
D=Path('/data/muxiangyu/datasets/M2HImage/M2H_Final_v2')
R=P/'artifacts/m2h_state_pilot_20260908'
S=R/'scripts'
PY='/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python'
IDPY='/home/muxiangyu/miniconda3/envs/StableAnimator/bin/python'
CONFIG=D/'phase1/phase2_a4_directed_r16_4000_768x1024/resolved_config.yaml'
MET=P/'artifacts/m2h_ab_objective_v1/scripts/a_d0_metrics_v2.py'
CHILDREN=[]
STATE={}
CAPS={'prepare':8.,'train':60.,'dev':10.}


def write(path,obj):
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def cost(category):
    return sum(x['gpu_hours'] for x in STATE['jobs'] if x['category']==category and 'gpu_hours' in x)


def check():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=STATE['boot_id']:
        raise RuntimeError('boot changed; expansion stopped')
    if shutil.disk_usage(R).free<50*1024**3:
        raise RuntimeError('disk reserve <50GiB')
    for c in CAPS:
        running=sum((time.time()-x['started'])*x['gpu_count']/3600 for x in STATE['jobs']
                    if x['category']==c and 'gpu_hours' not in x)
        if cost(c)+running>CAPS[c]:
            raise TimeoutError(c+' GPUh cap reached')


def env(gpus):
    return dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,HF_HOME='/data/muxiangyu/modelLibrary',HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',NO_ALBUMENTATIONS_UPDATE='1',
                PYTHONPATH=str(P)+':'+str(P/'artifacts/m2h_ab_objective_v1/scripts'))


def launch(name,cmd,gpus,category):
    check()
    log=(R/(name+'.log')).open('x')
    proc=subprocess.Popen(list(map(str,cmd)),cwd=P,env=env(gpus),stdout=log,stderr=subprocess.STDOUT,
                          stdin=subprocess.DEVNULL,start_new_session=True)
    log.close()
    row=dict(name=name,pid=proc.pid,cmd=list(map(str,cmd)),gpus=gpus,gpu_count=len(gpus.split(',')) if gpus else 0,
             category=category,started=time.time(),status='running')
    STATE['jobs'].append(row);CHILDREN.append((proc,row))
    write(R/'queue.json',STATE)
    return proc,row


def finish(job):
    proc,row=job
    if 'gpu_hours' not in row:
        row.update(exit_code=proc.returncode,finished=time.time(),gpu_hours=(time.time()-row['started'])*row['gpu_count']/3600,
                   status='complete' if proc.returncode==0 else 'failed')
        write(R/'queue.json',STATE)
    if proc.returncode!=0:
        raise RuntimeError(row['name']+' failed exit '+str(proc.returncode))


def wait(job):
    while job[0].poll() is None:
        check()
        for p,r in CHILDREN:
            if p.poll() is not None:
                finish((p,r))
        time.sleep(5)
    finish(job)


def run(name,cmd,gpus,category):
    wait(launch(name,cmd,gpus,category))


def gen(cache,pairs,config,ckpt,out,arm,count):
    return [PY,S/'generate_pilot.py','--repo',P,'--root',D,'--cache',cache,'--pairs',pairs,
            '--config',config,'--checkpoint',ckpt,'--out',out,'--arm',arm,'--count',count,'--device','cuda:0']


def train(arm,extra):
    return [PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=3',S/'train_state_pilot.py',
            '--project-root',P,'--data-root',D,'--config',CONFIG,'--pool',R/'protocol/pool.json',
            '--cache-root',R/'cache','--arm',arm,'--output-root',R/'train','--output-id','seed0',
            '--verify-restore',*extra]


def main():
    lock=(R/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (R/'queue.json').exists():
        raise RuntimeError('queue state already exists; no automatic restart')
    assert (R/'cache/READY').is_file()
    audit=json.loads((R/'cache/audit.json').read_text())
    assert audit['status']=='complete' and len(audit['M_roles'])==256 and len(audit['I_roles'])==288
    assert audit['undeclared_read_denied']
    # Metric adapter expects config, without mutating the sealed input cache audit.
    metriccache=R/'teacher_metric_cache';metriccache.mkdir()
    (metriccache/'masks').symlink_to(R/'cache/masks',target_is_directory=True)
    write(metriccache/'audit.json',dict(config=dict(data=dict(root=str(D)))))
    STATE.update(status='running',pid=os.getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 pool_sha256=sha(R/'protocol/pool.json'),scope='first3arms only, no secondseed/600/B',caps_gpu_hours=CAPS,
                 jobs=[dict(name='fresh_cache',category='prepare',status='complete',gpu_hours=audit['resources']['gpu_hours'])],
                 script_hashes={p.name:sha(p) for p in S.glob('*.py')})
    write(R/'queue.json',STATE)
    try:
        teacher=launch('teacher',gen(R/'cache',R/'protocol/pairs.json',CONFIG,
                                    D/'phase1/phase2_a4_directed_r16_4000_768x1024/checkpoints/final',R/'teacher','teacher',512)
                       +['--endpoints',R/'cache/endpoints','--max-hours',7.5],'3','prepare')
        run('health_C-H',train('C-H',['--health','--max-gpu-hours',4.]),'0,1,2','train')
        healthrows=[json.loads(x) for x in (R/'train/seed0/C-H/logs/train.jsonl').read_text().splitlines()]
        assert healthrows[-1]['pilot_update_step']==20 and healthrows[-1]['step']==8420
        run('health_restore_cpu',[PY,S/'train_state_pilot.py','--arm','C-H','--verify-checkpoint',
                                  R/'train/seed0/C-H/checkpoints/step-008420'],'','train')
        seconds=float(healthrows[-1]['seconds_per_optimizer_step'])
        if seconds>120:
            raise RuntimeError('health mean step >120sec; re-budget before expansion')
        steps=300 if seconds<=80 else 200
        STATE['frozen_total_updates_per_arm']=steps
        STATE['health_seconds_per_update']=seconds
        write(R/'budget_decision.json',dict(updates_per_arm=steps,health_mean_seconds=seconds,
                                          decision='pre-efficacy; all arms same total',train_cap_gpu_hours=60))
        write(R/'queue.json',STATE)
        wait(teacher)
        erows=[json.loads(x) for x in (R/'teacher/rows.jsonl').read_text().splitlines()]
        assert len(erows)==512
        for r in erows:
            assert sha(r['endpoint'])==r['endpoint_sha256'] and sha(r['path'])==r['sha256']
        write(R/'teacher_pool_verified.json',dict(count=512,rows_sha256=sha(R/'teacher/rows.jsonl'),no_score_filtering=True))
        for stage in ('image','id'):
            run('teacher_metrics_'+stage,[PY if stage=='image' else IDPY,MET,'--repo',P,'--cache',metriccache,
                                         '--run',R/'teacher','--stage',stage,'--device','cuda:0'],'3','prepare')
        # Reserve conservative120sec/update and actual initialization cost against cap.
        for arm in ('C-H','C-perm','E-match'):
            extra=['--total-steps',steps,'--max-gpu-hours',max(.01,60-cost('train'))]
            if arm=='C-H':
                extra+=['--resume',R/'train/seed0/C-H/checkpoints/step-008420']
            run('train_'+arm,train(arm,extra),'0,1,2','train')
        devpairs=P/'artifacts/m2h_ab_objective_v1/data_protocol_v2/dev128_pairs.json'
        devcache=P/'artifacts/m2h_minimal_revalidation_20260907/fresh_cache_v2'
        assert sha(devpairs)=='9cc9bc517e65486c2794cde0ef1d002d8db0f3f186bc2adc0c8fcdc5843929be'
        for arm in ('C-H','C-perm','E-match'):
            run('dev_'+arm,gen(devcache,devpairs,R/'train/seed0'/arm/'resolved_config.yaml',
                               R/'train/seed0'/arm/'checkpoints/final',R/'dev'/arm,arm,128)+['--max-hours',2.5],'0','dev')
            for stage in ('image','id','pose'):
                run('metrics_'+arm+'_'+stage,[PY if stage=='image' else IDPY,MET,'--repo',P,'--cache',devcache,
                    '--run',R/'dev'/arm,'--stage',stage,'--device','cuda:0'],'0','dev')
        run('report',[PY,S/'summarize_pilot.py','--repo',P,'--batch',R],'','dev')
        assert (R/'report/READY').exists()
        STATE['status']='complete'
        STATE['budget_used_gpu_hours']={k:cost(k) for k in CAPS}
        write(R/'queue.json',STATE)
    except BaseException as e:
        STATE.update(status='failed',error=repr(e))
        # Only this supervisor's own live children; never unrelated GPU jobs.
        for p,row in CHILDREN:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                row.update(status='terminated_own_queue_on_failure',gpu_hours=(time.time()-row['started'])*row['gpu_count']/3600)
        write(R/'queue.json',STATE)
        raise


if __name__=='__main__':
    main()
