"""Recover the interrupted report without overwriting any historical output."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from revalidation_common import sha256, write_json


def main():
    root = Path('/data/muxiangyu/pythonPrograms/M2HImage')
    base = root/'artifacts/m2h_minimal_revalidation_20260907'
    old = base/'run_v1'
    new = base/'recovery_run_v1'
    repaired = base/'recovery_bridge1_v1'
    if not (repaired/'READY').exists(): raise RuntimeError('replacement not ready')
    # The pre-reboot supervisor assembled bridge_metrics/rows.jsonl only after
    # both bridge generators returned. It died before that assembly; the two
    # durable generator row files are the authoritative 16-row source.
    original = []
    for arm in ('a4_legacyM_oldI', 'b2_legacyM_oldI'):
        original.extend(json.loads(x) for x in
                        (old/'bridge8'/arm/'rows.jsonl').read_text().splitlines())
    replacement = json.loads((repaired/'rows.jsonl').read_text().strip())
    key = lambda r: (r['mid'],r['jid'],r['seed'],r['branch'])
    corrupt = [r for r in original if sha256(r['path'])!=r['sha256']]
    if len(corrupt)!=1 or key(corrupt[0])!=key(replacement):
        raise ValueError('unexpected corruption scope')
    if sha256(replacement['path'])!=replacement['sha256']: raise ValueError('bad replacement')
    new.mkdir(exist_ok=False)
    (new/'logs').mkdir()
    (new/'main_metrics').symlink_to(old/'main_metrics', target_is_directory=True)
    bridge=new/'bridge_metrics';bridge.mkdir()
    rows=[replacement if key(r)==key(replacement) else r for r in original]
    for r in rows:
        if sha256(r['path'])!=r['sha256']: raise ValueError('bridge image integrity')
    (bridge/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (bridge/'READY').write_text('32 verified image rows; metrics pending\n')
    for stage in ('image','id','pose'):
        path=old/'main_metrics'/f'metrics_{stage}'/'per_image.jsonl'
        rs=[json.loads(x) for x in path.read_text().splitlines()]
        if len(rs)!=512 or any(r['metric_status']!='ok' for r in rs):
            raise ValueError('main metric integrity')
    manifest={'reason':'server reboot; boot time2026-09-07T22:14:54+08:00; no surviving workers',
              'original_run':str(old), 'original_unchanged':True,
              'corrupt_image':corrupt[0], 'replacement':replacement,
              'replacement_byte_identical_to_pre_reboot_hash':replacement['sha256']==corrupt[0]['sha256'],
              'main_rows_reused':512,'bridge_metric_rows_recomputed':32,'training_steps':0}
    write_json(new/'manifest.json',manifest)
    py='/home/muxiangyu/miniconda3/envs/refton_m2h/bin/python'
    pym='/home/muxiangyu/miniconda3/envs/StableAnimator/bin/python'
    script=root/'artifacts/m2h_ab_objective_v1/scripts/a_d0_metrics_v2.py'
    env=dict(os.environ,HF_HOME='/data/muxiangyu/modelLibrary',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',NO_ALBUMENTATIONS_UPDATE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
             PYTHONPATH=str(script.parent)+':'+str(base)+':'+str(root))
    def metric(stage,gpu):
        command=[py if stage=='image' else pym,'-u',str(script),'--repo',str(root),
                 '--cache',str(base/'fresh_cache_v2'),'--run',str(bridge),'--stage',stage,'--device','cuda:0']
        start=time.monotonic()
        with (new/'logs'/f'{stage}.log').open('x') as log:
            subprocess.run(command,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),cwd=root,
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        item={'name':'recovery_bridge_metrics_'+stage,'command':command,'gpu':gpu,
              'seconds':time.monotonic()-start,'gpu_hours':(time.monotonic()-start)/3600,'returncode':0}
        write_json(new/'logs'/f'{stage}.result.json',item)
        return item
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        recovered=list(pool.map(lambda x:metric(x[1],x[0]),enumerate(('image','id','pose'))))
    # Recover recorded process durations; bound missing/empty result durations
    # by their recorded launch time and verified reboot time, never zero them.
    boot=1788790494.0  # 2026-09-07 22:14:54 +08:00
    ledger=[]
    for path in sorted((old/'logs').glob('*.command.json')):
        command=json.loads(path.read_text());result=path.with_name(path.name.replace('.command.json','.result.json'))
        try:
            item=json.loads(result.read_text())
        except (FileNotFoundError,json.JSONDecodeError):
            seconds=max(0.,boot-command['started'])
            item={**command,'seconds':seconds,'gpu_hours':seconds/3600,
                  'accounting':'upper bound launch-to-boot; result unavailable after reboot'}
        ledger.append(item)
    repair=json.loads((repaired/'summary.json').read_text())
    ledger += recovered + [{'name':'replacement_generation','seconds':repair['seconds'],
                            'gpu_hours':repair['gpu_hours'],'accounting':'generator wall excluding initial imports'}]
    write_json(new/'budget.json',{'processes':ledger,
                'gpu_hours_including_startup':sum(r['gpu_hours'] for r in ledger),
                'accounting_note':'sum contains conservative launch-to-reboot bounds for3 missing result files; replacement/import startup not fully timed'})
    (new/'METRICS_READY').write_text('main512 preserved; bridge32 recomputed; accounted recovery\n')
    subprocess.run([py,str(base/'finish_revalidation.py'),'--artifacts',str(base),
                    '--run',str(new),'--out',str(base/'report_recovered_v1')],check=True,timeout=300,env=env)
    report=base/'report_recovered_v1/REPORT.md'
    report.write_text(report.read_text()+'\n## 重启恢复记录\n\n'
        +'服务器于2026-09-07 22:14:54重启。512张主图及主指标完整复用；1张损坏桥接图按原配置/种子恢复，32条桥接记录三类指标全部重算。原损坏文件和空记录保留。\n'
        +f'\n恢复图与重启前记录SHA完全一致：{manifest["replacement_byte_identical_to_pre_reboot_hash"]}。\n'
        +'\n资源合计含3份缺失退出记录的“启动至重启”保守时间上界，不是全部精确GPU占用测量；补图的Python导入时间另有小额未计部分。\n')
    cp=base/'report_recovered_v1/COMPLETE.json'
    complete=json.loads(cp.read_text())
    complete.update(report_sha256=sha256(report),recovery_manifest_sha256=sha256(new/'manifest.json'))
    write_json(cp,complete)
    print(json.dumps({'complete':True,'report':str(base/'report_recovered_v1'),
                      'replacement_matches_hash':manifest['replacement_byte_identical_to_pre_reboot_hash']}),flush=True)


if __name__=='__main__':main()
