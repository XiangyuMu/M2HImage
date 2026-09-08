"""Bounded new128 A2 inference; hash-verified reuse of old256 predictions."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

P = Path('/data/muxiangyu/pythonPrograms/M2HImage')
D = Path('/data/muxiangyu/datasets/M2HImage/M2H_Final_v2')
R = P/'artifacts/m2h_minimal_revalidation_20260907'
N = P/'artifacts/m2h_a2_localization_20260908'
OLD = P/'artifacts/m2h_ab_objective_v1'
PY = '/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python'
PYMET = '/home/muxiangyu/miniconda3/envs/StableAnimator/bin/python'
sys.path.insert(0, str(R))
from revalidation_common import sha256, write_json, load_pairs


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def checked_rows(rows, expected):
    from PIL import Image
    keys = [(r['mid'], r['jid'], r['seed']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(expected):
        raise ValueError('pair mismatch')
    for r in rows:
        if sha256(r['path']) != r['sha256']:
            raise ValueError('image hash mismatch: '+r['path'])
        with Image.open(r['path']) as im:
            im.verify()
    return rows


def main():
    run = N/'run_v1'
    run.mkdir(exist_ok=False)
    (run/'logs').mkdir()
    pairs_path = OLD/'data_protocol_v2/dev128_pairs.json'
    pairs = load_pairs(pairs_path)
    if sha256(pairs_path) != '9cc9bc517e65486c2794cde0ef1d002d8db0f3f186bc2adc0c8fcdc5843929be':
        raise ValueError('pair manifest changed')
    expected = [(r['mid'], r['jid'], r['seed']) for r in pairs]
    env = dict(os.environ, HF_HOME='/data/muxiangyu/modelLibrary',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4',
               NO_ALBUMENTATIONS_UPDATE='1',
               PYTHONPATH=str(OLD/'scripts')+':'+str(R)+':'+str(P))
    state = {'pid': os.getpid(), 'status':'generation', 'training_steps':0,
             'pairs_sha256':sha256(pairs_path), 'budget_gpu_hours':12,
             'scripts_sha256':{str(x):sha256(x) for x in
                              (Path(__file__), N/'scripts/generate_a2.py')}, 'processes':[]}
    write_json(run/'manifest.json', state)

    def execute(name, cmd, gpu, timeout=7200):
        if shutil.disk_usage(N).free < 50*1024**3:
            raise RuntimeError('storage reserve')
        info = {'name':name, 'gpu':gpu, 'command':list(map(str,cmd)), 'started':time.time()}
        tick = time.monotonic()
        with (run/'logs'/f'{name}.log').open('x') as log:
            proc = subprocess.Popen(info['command'], cwd=P, env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            info['pid'] = proc.pid
            write_json(run/'logs'/f'{name}.command.json', info)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                rc = -124
        info.update(returncode=rc,seconds=time.monotonic()-tick,
                    gpu_hours=(time.monotonic()-tick)/3600)
        write_json(run/'logs'/f'{name}.result.json', info)
        if rc: raise RuntimeError(f'{name} failed: {rc}')
        return info

    try:
        conf = D/'phase1/phase2_a2_diff_r16_4000_768x1024'
        cmds = []
        for i in range(4):
            cmds.append([PY,'-u',N/'scripts/generate_a2.py','--repo',P,'--root',D,
                '--cache',R/'fresh_cache_v2','--pairs',pairs_path,'--config',conf/'resolved_config.yaml',
                '--checkpoint',conf/'checkpoints/final','--out',run/f'shard{i}',
                '--arm','a2_inputOnly_freshI','--start',i*32,'--count',32,'--max-hours',2])
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(execute,f'shard{i}',cmd,i) for i,cmd in enumerate(cmds)]
            for f in concurrent.futures.as_completed(futures):
                state['processes'].append(f.result())
                write_json(run/'manifest.json',state)
        a2 = []
        for i in range(4):
            part = run/f'shard{i}'
            if not (part/'READY').exists(): raise RuntimeError('shard not ready')
            a2.extend(checked_rows(read_rows(part/'rows.jsonl'),expected[i*32:(i+1)*32]))
        checked_rows(a2,expected)
        mr = run/'a2_metrics'; mr.mkdir()
        (mr/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in a2))
        (mr/'READY').write_text('128 A2 PNG hashes verified; metrics pending\n')
        oldrows = read_rows(R/'run_v1/main_metrics/rows.jsonl')
        combined = list(a2)
        for arm in ('b2_inputOnly_freshI','a4_inputOnly_freshI'):
            combined.extend(checked_rows([r for r in oldrows if r['branch']==arm],expected))
        (run/'combined_rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in combined))
        state['combined_rows_sha256'] = sha256(run/'combined_rows.jsonl')
        state['status'] = 'ordinary_metrics'
        write_json(run/'manifest.json',state)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = []
            for gpu,stage in enumerate(('image','id','pose')):
                cmd=[PY if stage=='image' else PYMET,'-u',OLD/'scripts/a_d0_metrics_v2.py',
                     '--repo',P,'--cache',R/'fresh_cache_v2','--run',mr,'--stage',stage,'--device','cuda:0']
                futures.append(pool.submit(execute,'metrics_'+stage,cmd,gpu,1800))
            for f in concurrent.futures.as_completed(futures):
                state['processes'].append(f.result())
                write_json(run/'manifest.json',state)
        for stage in ('image','id','pose'):
            rows=read_rows(mr/f'metrics_{stage}/per_image.jsonl')
            if len(rows)!=128 or any(r['metric_status'] in ('failed','reference_failed') for r in rows):
                raise RuntimeError('metrics incomplete or infrastructure failure')
        state.update(status='a2_complete_regions_pending',new_predictions=128,reused_predictions=256,
                     gpu_hours=sum(r['gpu_hours'] for r in state['processes']))
        if state['gpu_hours'] > 12: raise RuntimeError('budget exceeded')
        write_json(run/'manifest.json',state)
        (run/'A2_READY').write_text('A2 and input384 verified; regional diagnostics/report pending\n')
        print(json.dumps(state),flush=True)
    except Exception as exc:
        state.update(status='failed',error=repr(exc))
        write_json(run/'FAILED.json',state)
        raise


if __name__ == '__main__': main()
