"""CPU-only validated merge of16 recovered old-I and16 existing fresh-I rows."""
import json
import subprocess
import sys
from pathlib import Path
from revalidation_common import sha256, write_json


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(r):
    return (r['mid'], r['jid'], r['seed'], r['branch'])


def combine(old, fresh, expected):
    old_keys = set(expected)
    if len(old)!=16 or {key(r) for r in old}!=old_keys:
        raise ValueError('recovered old-I rows not exactly the16 expected keys')
    fresh_keys = {(m,j,s,b.replace('_oldI','_freshI')) for m,j,s,b in old_keys}
    selected = [r for r in fresh if key(r) in fresh_keys]
    if len(selected)!=16 or {key(r) for r in selected}!=fresh_keys:
        raise ValueError('fresh-I counterpart missing or duplicate')
    merged = old+selected
    if len({key(r) for r in merged})!=32:
        raise ValueError('bridge merge duplicate')
    return merged


def main():
    base=Path('/data/muxiangyu/pythonPrograms/M2HImage/artifacts/m2h_minimal_revalidation_20260907')
    main_run=base/'run_v1/main_metrics'
    recovered=base/'recovery_run_v1'
    new=base/'recovery_run_v2'
    out=base/'report_recovered_v2'
    oldgen=read_rows(recovered/'bridge_metrics/rows.jsonl')
    maingen=read_rows(main_run/'rows.jsonl')
    expected={key(r) for r in oldgen}
    gen=combine(oldgen,maingen,expected)
    if len(maingen)!=512: raise ValueError('main generation count')
    bykey={key(r):r for r in gen}
    for r in maingen+oldgen:
        if sha256(r['path'])!=r['sha256']:raise ValueError('image hash changed: '+r['path'])
    stages={};provenance={}
    for stage in ('image','id','pose'):
        paths=[recovered/f'bridge_metrics/metrics_{stage}/per_image.jsonl', main_run/f'metrics_{stage}/per_image.jsonl']
        rs=combine(read_rows(paths[0]),read_rows(paths[1]),expected)
        for r in rs:
            if r['path']!=bykey[key(r)]['path']:raise ValueError('metric points at wrong generation')
            if r['metric_status']!='ok':raise ValueError('non-ok bridge metric')
        stages[stage]=rs
        provenance.update({str(p):sha256(p) for p in paths})
    new.mkdir(exist_ok=False)
    (new/'main_metrics').symlink_to(main_run,target_is_directory=True)
    bridge=new/'bridge_metrics';bridge.mkdir()
    (bridge/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in gen))
    for stage,rs in stages.items():
        dest=bridge/f'metrics_{stage}';dest.mkdir()
        (dest/'per_image.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rs))
        write_json(dest/'summary.json',{'rows':32,'ok':32,'recovered_oldI':16,'reused_main_freshI':16})
        (dest/'READY').write_text('32 path/key validated merged metric records\n')
    budget=json.loads((recovered/'budget.json').read_text())
    bounded=[r['name'] for r in budget['processes'] if 'upper bound' in r.get('accounting','')]
    budget['accounting_note']=f'{len(bounded)} missing/empty result records bounded by launch-to-reboot; names explicitly listed'
    budget['bounded_process_names']=bounded
    write_json(new/'budget.json',budget)
    rep=json.loads((recovered/'manifest.json').read_text())
    manifest={'recovery_v1_row_count_bug':'only16 oldI rows assembled; freshI counterparts omitted',
              'main_images_verified':512,'bridge_unique_oldI_images_verified':16,
              'bridge_metric_rows_per_stage':32,'bridge_metric_recomputed_rows_per_stage':16,
              'bridge_metric_reused_main_rows_per_stage':16,'new_gpu_hours':0,
              'replacement_matches_pre_reboot_sha':rep['replacement_byte_identical_to_pre_reboot_hash'],
              'input_sha256':provenance,'script_sha256':sha256(__file__),
              'historical_files_preserved':True}
    write_json(new/'manifest.json',manifest)
    (new/'METRICS_READY').write_text('verified512 main and32 bridge rows\n')
    subprocess.run([sys.executable,str(base/'finish_revalidation.py'),'--artifacts',str(base),
                    '--run',str(new),'--out',str(out)],check=True,timeout=300)
    report=out/'REPORT.md'
    note='\n## 重启与恢复\n\n服务器在2026-09-07 22:14:54重启；512张主图及指标完整复用。恢复1张桥接图，重算16条old-I指标；从主实验精确合并16条fresh-I对照。本次合并不消耗GPU。先前恢复v1漏合并对照，已保留其失败记录。\n'
    note+=f'\n恢复图与重启前SHA一致：{manifest["replacement_matches_pre_reboot_sha"]}。预算含{len(bounded)}份缺失退出记录的启动至重启保守上界；不宣称全项精确计时。\n'
    report.write_text(report.read_text()+note)
    cp=out/'COMPLETE.json';complete=json.loads(cp.read_text())
    complete.update(report_sha256=sha256(report),recovery_manifest_sha256=sha256(new/'manifest.json'))
    write_json(cp,complete)
    print(json.dumps({'complete':True,'report':str(out),'bounded_process_count':len(bounded)}))


if __name__=='__main__':main()
