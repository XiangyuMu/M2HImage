"""No model calls: validate durable CPU shards and finish report after reboot."""
import json
import os
import shutil
import subprocess
import time
from run_a2 import P,N,R,PY,read_rows,write_json,sha256
from run_localization import preflight,check_regions
from run_cpu_recovery_v4 import durable


def main():
    out=N/'assembly_cpu_v5';out.mkdir(exist_ok=False)
    run=N/'run_v1';prior=N/'cpu_recovery_v4';report=N/'report_cpu_v5'
    audit={'pid':os.getpid(),'preflight':preflight(run),'training_steps':0,'new_model_calls':0,'sources':{}}
    imgs=[];cross=[];summaries=[];resources=[]
    manifest=json.loads((prior/'manifest.json').read_text())
    for path,h in manifest['scripts_sha256'].items():
        if sha256(path)!=h:raise ValueError('script changed')
    for i in range(8):
        part=prior/f'shard{i}';ir,cr=check_regions(part,8)
        imgs+=ir;cross+=cr
        summaries.append(json.loads((part/'summary.json').read_text()))
        resources.append(json.loads((part/'resources.json').read_text()))
        for name in ('per_image.jsonl','cross_ref.jsonl','summary.json','resources.json','mask_config.json'):
            audit['sources'][str(part/name)]=sha256(part/name)
    smi,smc=check_regions(N/'cpu_smoke_v3',2)
    for small,large,keys in ((smi,imgs,('mid','jid','seed','branch','region')),
                             (smc,cross,('mid','jid_left','jid_right','seed','branch','region'))):
        by={tuple(r[k] for k in keys):r for r in large}
        if len(by)!=len(large):raise ValueError('duplicate region key')
        for r in small:
            v=by[tuple(r[k] for k in keys)]
            for k,x in r.items():
                if k.startswith(('region_','cross_ref_')) and isinstance(x,(int,float)):
                    if v[k] is None or abs(x-v[k])>1e-6:raise ValueError('smoke/full differs')
    merged=out/'merged';merged.mkdir()
    for name,rr in (('per_image.jsonl',imgs),('cross_ref.jsonl',cross)):
        (merged/name).write_text(''.join(json.dumps(r)+'\n' for r in rr))
    shutil.copyfile(prior/'shard0/mask_config.json',merged/'mask_config.json')
    write_json(merged/'summary.json',{'status':'verified','device':'cpu','shards':summaries,
              'per_image_region_rows':len(imgs),'cross_reference_rows':len(cross),'input_rows_sha256':sha256(run/'combined_rows.jsonl')})
    (merged/'READY').write_text('verified1920+960, no model calls\n');durable(merged)
    write_json(out/'manifest.json',audit)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    subprocess.run([PY,str(N/'scripts/summarize_cpu.py'),'--run',str(run),'--regions',str(merged),'--out',str(report)],env=env,check=True,timeout=600)
    a2=json.loads((run/'manifest.json').read_text())
    old=json.loads((N/'localization_v1/manifest.json').read_text())['processes'][0]
    epoch=lambda s:time.mktime(time.strptime(s,'%Y-%m-%d %H:%M:%S'))
    bound1=(epoch('2026-09-08 13:02:38')-old['started'])/3600
    bound2=(epoch('2026-09-08 13:10:02')-epoch('2026-09-08 13:02:38'))/3600
    budget={'a2_and_ordinary_gpu_hours':a2['gpu_hours'],'old_smoke_gpu_hours':old['gpu_hours'],
            'lost_full_gpu_hours_upper_bound':bound1,'lost_second_smoke_gpu_hours_upper_bound':bound2,
            'lost_bounds_basis':'first old smoke launch through13:02:38 reboot; second whole13:02:38-13:10:02 boot interval',
            'recorded_plus_lost_bounds_gpu_hours':a2['gpu_hours']+old['gpu_hours']+bound1+bound2,
            'cpu_shard_wall_seconds_sum':sum(x['cpu_wall_seconds_including_imports'] for x in resources),
            'cpu_shard_max_wall_seconds':max(x['cpu_wall_seconds_including_imports'] for x in resources),
            'cpu_smoke_wall_seconds':json.loads((N/'cpu_smoke_v3/resources.json').read_text())['cpu_wall_seconds_including_imports'],
            'cpu_recovery_gpu_hours':0,'training_steps':0,'cap_gpu_hours':12,
            'note':'CPU summary/verification not counted as GPU time. Lost GPU intervals conservative, not exact all-inclusive total.'}
    write_json(report/'BUDGET.json',budget)
    with (report/'REPORT.md').open('a') as f:
        f.write('\n## 重启恢复与计算消耗\n\n384张图及A2三类指标完整复用；CPU重算分区，8个分片各4线程。13:28:19再次重启，但各分片已完整落盘，随后只合并报告，没有重跑评测或生成图片。\n')
        f.write(f'CPU分片最长墙钟{budget["cpu_shard_max_wall_seconds"]:.1f}秒；CPU恢复新增GPUh为0。原GPU记录加丢失任务保守上界共{budget["recorded_plus_lost_bounds_gpu_hours"]:.4f} GPUh，并非精确总计。\n')
        f.write('CPU小样本与全量重算差≤1e-6；DINO与旧GPU差≤1e-5、HF-LPIPS差≤1e-3，具体最大差见summary.json。\n')
    complete=json.loads((report/'COMPLETE.json').read_text())
    complete.update(report_sha256=sha256(report/'REPORT.md'),budget_sha256=sha256(report/'BUDGET.json'),
                    assembly_manifest_sha256=sha256(out/'manifest.json'),region_summary_sha256=sha256(merged/'summary.json'))
    write_json(report/'COMPLETE.json',complete);durable(report)
    (out/'READY').write_text('full batch complete, no training\n');durable(out)
    print(json.dumps({'complete':True,'report':str(report)}),flush=True)


if __name__=='__main__':main()
