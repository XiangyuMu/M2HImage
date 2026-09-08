"""Make an append-only B view for the existing three objective metric passes."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--b-run',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    summary=json.loads((a.b_run/'summary.json').read_text())
    if summary['status']!='complete': raise RuntimeError('B incomplete')
    audit=json.loads((a.probe/'audit.json').read_text())
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'audit.json').write_text(json.dumps(audit,indent=2))
    (a.out/'masks').symlink_to((a.probe/'masks').resolve(),target_is_directory=True)
    with (a.out/'rows.jsonl').open('x') as f:
        for pair in audit['pairs']:
            name=f'{pair["mid"]}__id{pair["jid"]}__seed{pair["seed"]}.png'
            for method in ('base','inplace_feather','warp_feather','warp_multiband'):
                path=(a.b_run/method/name).resolve()
                if not path.is_file(): raise FileNotFoundError(path)
                f.write(json.dumps({**pair,'branch':method,'tau':0.,'k':0,'path':str(path),
                    'clean_carryin_path':str((a.b_run/'base'/name).resolve()),'noisy_state_path':None})+'\n')
    (a.out/'READY').write_text('B metric adapter, no new generated predictions\n')

if __name__=='__main__': main()
