"""Synthetic end-to-end reporting fixture; never consumes actual model outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


class ReportEndToEnd(unittest.TestCase):
    def test_complete_marker_requires_main_and_bridge(self):
        here = Path(__file__).resolve().parent
        finisher = here / 'finish_revalidation.py'
        summary = here / 'summarize_revalidation.py'
        if not finisher.exists(): finisher = here.parent / 'finish_revalidation.py'
        with tempfile.TemporaryDirectory(prefix='m2h-synthetic-report-test-') as d:
            root = Path(d); run = root/'run'; out = root/'report'
            (root/'scripts').mkdir(); (root/'scripts/summarize_revalidation.py').symlink_to(summary)
            pairs=[{'mid':f'{i//2:05d}', 'jid':f'{100+i%32:05d}', 'seed':0,
                    'm_source_group':f'source{i//2}', 'pair_index':i} for i in range(128)]
            put(root/'protocol_v1/manifest.json', {'pairs':pairs,'sensitivity_pair_indices':list(range(101))})
            put(root/'fresh_cache_v2/audit.json', {'resources':{'gpu_hours':.01}})
            put(root/'input_only_cache32/audit.json', {'resources':{'gpu_hours':.01}})
            put(run/'budget.json', {'gpu_hours_including_startup':.1})
            fields={'image':['garment_dino','garment_hf_lpips'], 'id':['id_penalized'],
                    'pose':['body_penalized','head_penalized','body_coverage','head_coverage']}
            arms=['b2_legacyM_freshI','a4_legacyM_freshI','b2_inputOnly_freshI','a4_inputOnly_freshI']
            for view in ('main_metrics','bridge_metrics'):
                for stage,keys in fields.items():
                    path=run/view/('metrics_'+stage)/'per_image.jsonl'
                    path.parent.mkdir(parents=True,exist_ok=True)
                    rows=[]
                    selected=arms if view=='main_metrics' else ['b2_legacyM_oldI','a4_legacyM_oldI']+arms[:2]
                    for arm in selected:
                        for pair in pairs if view=='main_metrics' else pairs[:8]:
                            row={**pair,'branch':arm,'tau':0,'k':1,'path':'synthetic-only', 'metric_status':'ok'}
                            row.update({k:.5 for k in keys})
                            row['carryin_corrupt_ignored']='unused'
                            rows.append(row)
                    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (run/'METRICS_READY').write_text('synthetic fixture\n')
            result=subprocess.run([sys.executable,str(finisher),'--artifacts',str(root),
                   '--run',str(run),'--out',str(out)],capture_output=True,text=True,timeout=180,
                   env=dict(os.environ,OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'))
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((out/'COMPLETE.json').exists())
            self.assertTrue((run/'READY').exists())
            report=json.loads((out/'summary.json').read_text())
            self.assertEqual(report['sensitivity']['n_pairs'],101)
            self.assertEqual(report['effects']['difference_in_differences']['id_penalized']['delta_mean'],0)
            bridge=json.loads((out/'I_BRIDGE.json').read_text())
            self.assertEqual(bridge['results']['a4']['metrics']['id_penalized']['fresh_minus_old_mean'],0)


if __name__=='__main__': unittest.main()
