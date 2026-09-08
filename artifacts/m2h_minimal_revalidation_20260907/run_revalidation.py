"""Bounded resumeless supervisor: smoke -> verified full -> objective metrics.

No parameter training. All generation directories are fresh. 32-pair smoke
outputs are intentionally reused as the first32 of the128-pair main matrix.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from revalidation_common import ARMS, sha256, write_json, load_pairs

PY = '/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python'
PYMET = '/home/muxiangyu/miniconda3/envs/StableAnimator/bin/python'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    logs = a.out / 'logs'; logs.mkdir()
    old = a.repo / 'artifacts/m2h_ab_objective_v1'
    pairsfile = old / 'data_protocol_v2/dev128_pairs.json'
    pairs = load_pairs(pairsfile)
    if len(pairs) != 128: raise ValueError('128 pairs required')
    cache = a.artifacts / 'fresh_cache_v2'
    if not (cache / 'READY').exists(): raise RuntimeError('cache not ready')
    baseenv = dict(os.environ, HF_HOME='/data/muxiangyu/modelLibrary',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', NO_ALBUMENTATIONS_UPDATE='1',
                   PYTHONPATH=str(old / 'scripts') + ':' + str(a.artifacts) + ':' + str(a.repo))
    ledger = []
    manifest = {'status': 'smoke', 'pid': os.getpid(), 'training_steps': 0,
                'arms': ARMS, 'pairs_sha256': sha256(pairsfile),
                'cache_audit_sha256': sha256(cache / 'audit.json'),
                'smoke_pairs': 32, 'full_additional_pairs': 96,
                'main_predictions': 512, 'bridge_oldI_predictions': 16,
                'bridge_rule': 'first8 fixed pairs per model legacyM_oldI vs common freshI; no output-based selection',
                'budget_gpu_hours': 40, 'gpu_resources': [0, 1, 2, 3],
                'metrics_adapter': 'self carryin paths unused by report; only output metrics matter',
                'scripts_sha256': {str(x): sha256(x) for x in [Path(__file__), a.artifacts/'generate_revalidation.py',
                                    a.artifacts/'revalidation_common.py', old/'scripts/a_d0_metrics_v2.py']}}
    write_json(a.out / 'manifest.json', manifest)

    def execute(name, cmd, gpu, timeout=10800):
        env = dict(baseenv, CUDA_VISIBLE_DEVICES=str(gpu))
        info = {'name': name, 'command': [str(c) for c in cmd], 'gpu': gpu, 'started': time.time()}
        write_json(logs / f'{name}.command.json', info)
        tick = time.monotonic()
        with (logs / f'{name}.log').open('x') as f:
            proc = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT,
                                    env=env, cwd=a.repo, start_new_session=True)
            info['pid'] = proc.pid
            write_json(logs / f'{name}.command.json', info)
            try:
                rc = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # Only this supervisor's exact child is terminated on its budget.
                proc.terminate()
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
                rc = -124
        info.update(returncode=rc, seconds=time.monotonic()-tick,
                    gpu_hours=(time.monotonic()-tick)/3600)
        ledger.append(info)
        write_json(logs / f'{name}.result.json', info)
        if rc != 0: raise RuntimeError(f'{name} failed {rc}; see log')
        return info

    def gen(arm, start, count, stage, gpu):
        model = 'a4' if arm.startswith('a4_') else 'b2'
        run = a.root / ('phase1/phase2_a4_directed_r16_4000_768x1024' if model=='a4' else
                        'phase1/phase2_b2_cont_r16_4000_768x1024')
        out = a.out / stage / arm
        out.parent.mkdir(exist_ok=True)
        cmd = [PY, '-u', a.artifacts/'generate_revalidation.py', '--repo', a.repo,
               '--root', a.root, '--cache', cache, '--pairs', pairsfile,
               '--config', run/'resolved_config.yaml', '--checkpoint', run/'checkpoints/final',
               '--out', out, '--arm', arm, '--start', start, '--count', count,
               '--device', 'cuda:0', '--max-hours', '3']
        return execute(stage+'_'+arm, cmd, gpu, 11000)

    def verify_part(stage, arm, count):
        path = a.out / stage / arm
        if not (path/'READY').exists(): raise RuntimeError('generation marker missing')
        m = json.loads((path/'manifest.json').read_text())
        rows = [json.loads(x) for x in (path/'rows.jsonl').read_text().splitlines()]
        if len(rows) != count: raise ValueError('wrong row count')
        ev = m['read_guard']
        if len(ev['denied_reads']) != 1 or ev['denied_subprocesses'] or not m['target_denial_test_passed']:
            raise ValueError('read guard failed')
        if 'inputOnly' in arm and ev['dataset_reads']:
            raise ValueError('input-only generation accessed dataset')
        for r in rows:
            if sha256(r['path']) != r['sha256']: raise ValueError('output hash changed')
        if m['resources']['peak_gib'] > 44: raise RuntimeError('memory gate exceeded')
        return rows

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda x: gen(x[1], 0, 32, 'smoke32', x[0]), enumerate(ARMS)))
        for arm in ARMS: verify_part('smoke32', arm, 32)
        # Extrapolate conservatively from complete smoke process wall, startup included.
        estimate = sum(x['gpu_hours'] for x in ledger) * 4 + 4
        manifest['estimated_generation_and_metric_gpu_hours'] = estimate
        if estimate > 40: raise RuntimeError('estimated budget exceeds40 GPUh')
        manifest['status'] = 'full_generation'
        write_json(a.out/'manifest.json', manifest)
        (a.out/'SMOKE_READY').write_text('32 pairs per arm; guards and hashes verified\n')
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda x: gen(x[1], 32, 96, 'full96', x[0]), enumerate(ARMS)))
        view = a.out/'main_metrics'; view.mkdir()
        rows = []
        for arm in ARMS:
            rows += verify_part('smoke32', arm, 32) + verify_part('full96', arm, 96)
        if len(rows) != 512: raise ValueError('full matrix incomplete')
        (view/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        (view/'READY').write_text('512 verified output rows; metrics pending\n')
        manifest['status'] = 'metrics_and_I_bridge'
        write_json(a.out/'manifest.json', manifest)
        metric_script = old/'scripts/a_d0_metrics_v2.py'
        def metric(stage, gpu):
            return execute('main_metrics_'+stage,
                [PY if stage=='image' else PYMET, '-u', metric_script,
                 '--repo', a.repo, '--cache', cache, '--run', view, '--stage', stage, '--device', 'cuda:0'], gpu)
        def bridges():
            # Sequential on fourth GPU, no conflict with metric GPUs0-2.
            gen('b2_legacyM_oldI', 0, 8, 'bridge8', 3)
            gen('a4_legacyM_oldI', 0, 8, 'bridge8', 3)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            tasks = [pool.submit(metric, stage, i) for i,stage in enumerate(('image','id','pose'))]
            tasks += [pool.submit(bridges)]
            for task in tasks: task.result()
        bridgeview = a.out/'bridge_metrics'; bridgeview.mkdir()
        bridgerows = []
        for arm in ('b2_legacyM_oldI','a4_legacyM_oldI'):
            bridgerows += verify_part('bridge8', arm, 8)
        for arm in ('b2_legacyM_freshI','a4_legacyM_freshI'):
            bridgerows += verify_part('smoke32', arm, 32)[:8]
        (bridgeview/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bridgerows))
        (bridgeview/'READY').write_text('16 oldI and16 shared freshI bridge rows\n')
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            tasks = [pool.submit(execute, 'bridge_metrics_'+s,
                [PY if s=='image' else PYMET, '-u', metric_script, '--repo', a.repo,
                 '--cache', cache, '--run', bridgeview, '--stage', s, '--device', 'cuda:0'], i)
                 for i,s in enumerate(('image','id','pose'))]
            for t in tasks: t.result()
        manifest['status'] = 'generation_and_metrics_complete_summary_pending'
        write_json(a.out/'manifest.json', manifest)
        write_json(a.out/'budget.json', {'processes': ledger, 'gpu_hours_including_startup': sum(x['gpu_hours'] for x in ledger),
                    'excluding': 'prior cache work/failed cache initialization/protocol CPU/tests; separately in ledger'})
        (a.out/'METRICS_READY').write_text('main512 and bridge32 metric passes completed; report validation still required\n')
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error'] = repr(exc)
        write_json(a.out/'manifest.json', manifest)
        write_json(a.out/'FAILED.json', {'error': repr(exc), 'processes': ledger})
        raise


if __name__ == '__main__': main()
