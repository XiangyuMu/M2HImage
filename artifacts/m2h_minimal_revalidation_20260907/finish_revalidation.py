"""Finish already-authorized diagnostics once supervisor metric outputs exist.

Waits only on this run. No training, regeneration, or recovery launches.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from revalidation_common import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    started = time.monotonic()
    if a.out.exists(): raise FileExistsError(a.out)
    while not (a.run/'METRICS_READY').exists():
        if (a.run/'FAILED.json').exists(): raise RuntimeError('supervisor failed; no reporting success')
        if time.monotonic() - started > 6*3600: raise TimeoutError('metric wait exceeded6h')
        time.sleep(30)
    script = a.artifacts/'scripts/summarize_revalidation.py'
    protocol = a.artifacts/'protocol_v1/manifest.json'
    cmd = [sys.executable, str(script), '--run', str(a.run/'main_metrics'),
           '--manifest-json', str(protocol), '--sensitivity-json', str(protocol), '--out', str(a.out)]
    subprocess.run(cmd, check=True)
    if not (a.out/'READY').exists(): raise RuntimeError('main report did not validate')
    sys.path.insert(0, str(a.artifacts/'scripts'))
    from summarize_revalidation import merge_metrics, METRIC_FIELDS
    rows, statuses, inputs = merge_metrics(a.run/'bridge_metrics')
    by = {(r['branch'], r['mid'], r['jid'], r['seed']): r for r in rows}
    if len(by) != 32: raise ValueError('bridge must have32 rows')
    if any(r.get(k) is None for r in rows for k in METRIC_FIELDS): raise ValueError('bridge metric missing')
    results = {}
    for model in ('b2', 'a4'):
        old = [r for r in rows if r['branch'] == model+'_legacyM_oldI']
        fresh = [r for r in rows if r['branch'] == model+'_legacyM_freshI']
        keys = {(r['mid'], r['jid'], r['seed']) for r in old}
        if len(keys)!=8 or keys != {(r['mid'],r['jid'],r['seed']) for r in fresh}:
            raise ValueError('bridge pair mismatch')
        results[model] = {'pairs': 8, 'metrics': {}}
        for k in METRIC_FIELDS:
            deltas = [by[(model+'_legacyM_freshI',*key)][k]-by[(model+'_legacyM_oldI',*key)][k] for key in sorted(keys)]
            results[model]['metrics'][k] = {
                'oldI_mean': statistics.fmean(r[k] for r in old),
                'freshI_mean': statistics.fmean(r[k] for r in fresh),
                'fresh_minus_old_mean': statistics.fmean(deltas),
                'paired_deltas': deltas}
    bridge = {'scope': 'first8 fixed pairs per model; descriptive feature-rebuild bridge, no population/seed claims',
              'status_counts': statuses, 'results': results,
              'input_sha256': {str(f): sha256(f) for f in inputs}}
    write_json(a.out/'I_BRIDGE.json', bridge)
    budget = json.loads((a.run/'budget.json').read_text())
    cache = json.loads((a.artifacts/'fresh_cache_v2/audit.json').read_text())
    plumbing = json.loads((a.artifacts/'input_only_cache32/audit.json').read_text())
    budget.update(fresh_cache_gpu_hours=cache['resources']['gpu_hours'],
                  initial_plumbing_gpu_hours=plumbing['resources']['gpu_hours'],
                  failed_cache_v1_gpu_hours_upper_bound=0.5,
                  accounting_limitations=['initial cache times exclude Python/import startup; failed cache initialization had no exit timing; conservative1800s timeout bound used',
                                          'supervisor child wall includes startup but marks process as GPU task during initialization',
                                          'protocol/calibration/tests/reporting CPU not GPU hours'])
    budget['recorded_gpu_hours_plus_failure_bound'] = budget['gpu_hours_including_startup'] + cache['resources']['gpu_hours'] + plumbing['resources']['gpu_hours'] + .5
    write_json(a.out/'BUDGET.json', budget)
    report = (a.out/'REPORT.md').read_text()
    report += '\n## 参考特征重建桥接（预固定8对／模型）\n\n'
    report += '主四臂均使用共同fresh I；历史M列不是历史全部输入的精确复现。以下是固定legacy M下fresh I减old I，样本很小，仅作描述性诊断。\n\n'
    report += '| 模型 | ID差值 | 服装DINO差值 | HF-LPIPS差值 | 身体误差差值 | 头部误差差值 |\n|---|---:|---:|---:|---:|---:|\n'
    for model in ('b2','a4'):
        v=results[model]['metrics']
        report += '| '+model+' | '+' | '.join(f'{v[k]["fresh_minus_old_mean"]:+.6f}' for k in
                     ('id_penalized','garment_dino','garment_hf_lpips','body_penalized','head_penalized'))+' |\n'
    report += '\n## 解释边界\n\n'
    report += '- 无参数训练；这是历史checkpoint换输入条件后的敏感性复验，不证明新A状态训练机制成立。\n'
    report += '- legacy M使用额外H派生条件，只作诊断；纯输入M改动不包括姿态定义变化，64份重建pose与旧cache逐元素相同。\n'
    report += '- 近重复阈值使用global SSIM代理与训练侧合成扰动，仅在校准样本上区分通过；101对敏感性子集不能认证为全局无近重复。参考图重叠不等于答案泄露。\n'
    report += '- 报告身份收益与服装/姿态代价，不用单个显著性结果自动启动训练。没有人工标注、G2或原始split修改。\n'
    report += f'\nGPU子进程墙钟（含启动）{budget["gpu_hours_including_startup"]:.4f}GPUh；缓存另计，失败初始化使用≤0.5GPUh保守上界，详见BUDGET.json。\n'
    (a.out/'REPORT.md').write_text(report)
    write_json(a.out/'COMPLETE.json', {'status':'minimal_revalidation_complete', 'training_steps':0,
                 'main_predictions':512,'additional_oldI_predictions':16,'bridge_evaluated_rows':32,
                 'protocol_sha256':sha256(protocol),'report_sha256':sha256(a.out/'REPORT.md'),
                 'main_summary_sha256':sha256(a.out/'summary.json')})
    (a.run/'READY').write_text('generation metrics main report and I-bridge complete; no training\n')


if __name__ == '__main__': main()
