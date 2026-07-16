from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conditions import load_yaml, read_ids, sha256_file
from train_paired import configure_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description='Fail-fast fairness and asset preflight for the unique A4 run.')
    parser.add_argument('--a4-config', default='configs/a4_directed.yaml')
    parser.add_argument('--b2cont-config', default='configs/b2_cont.yaml')
    args = parser.parse_args()
    a4 = load_yaml(args.a4_config)
    b2 = load_yaml(args.b2cont_config)
    configure_runtime(a4, world_size=3, all_gpus_train=False)
    root = Path(a4['data']['root'])
    b2_run = Path(b2['experiment']['output_root']) / b2['experiment']['id']
    b2_launch = json.loads((b2_run / 'launch.json').read_text(encoding='utf-8'))
    b2_resolved = load_yaml(b2_run / 'resolved_config.yaml')

    train_ids = read_ids(root / a4['data']['train_split'])
    excluded = set(str(value) for value in a4['data'].get('exclude_train_ids', []))
    train_ids = [sample_id for sample_id in train_ids if sample_id not in excluded]
    train_ids_hash = hashlib.sha256(('\n'.join(train_ids) + '\n').encode('utf-8')).hexdigest()[:16]
    resume = Path(a4['training']['resume'])
    resume_hash = sha256_file(resume / 'trainable.pt')[:16]
    fields = {
        'resume_trainable_hash': (resume_hash, b2_launch['resume_trainable_hash']),
        'train_ids_hash': (train_ids_hash, b2_launch['train_ids_hash']),
        'train_sample_count': (len(train_ids), b2_launch['train_sample_count']),
        'excluded_train_ids': (sorted(excluded), sorted(b2_launch['excluded_train_ids'])),
        'seed': (a4['experiment']['seed'], b2_resolved['experiment']['seed']),
        'global_batch': (a4['_runtime']['global_batch'], b2_resolved['_runtime']['global_batch']),
        'effective_lr': (a4['_runtime']['effective_lr'], b2_resolved['_runtime']['effective_lr']),
        'continuation_steps': (a4['training']['additional_steps'], b2_resolved['_runtime']['continuation_steps']),
        'lora_rank': (a4['model']['lora_rank'], b2_resolved['model']['lora_rank']),
        'lr_schedule': (a4['training'].get('lr_schedule', 'constant'), b2_resolved['training'].get('lr_schedule', 'constant')),
    }
    mismatches = {name: values for name, values in fields.items() if values[0] != values[1]}

    bank_path = root / a4['data']['identity_bank_v2']
    bank_manifest = json.loads(bank_path.with_suffix('.json').read_text(encoding='utf-8'))
    recognizer = Path(a4['model']['train_recognizer']['checkpoint'])
    if 'adaface' in str(recognizer).lower():
        raise RuntimeError('A4 F_train points to forbidden held-out AdaFace')
    smoke_dir = Path(a4['experiment']['output_root']) / 'phase2_a4_smoke_20_jk'
    smoke_status = json.loads((smoke_dir / 'training_status.json').read_text(encoding='utf-8'))
    smoke_rows = [json.loads(line) for line in (smoke_dir / 'logs/train.jsonl').read_text(encoding='utf-8').splitlines()]
    branches = sorted({
        int(row['id_decode_branch'])
        for row in smoke_rows
        if float(row.get('id_loss_triggered', 0.0)) > 0.0
    })
    vram_report = root / 'phase1/vram_report_a4_768x1024.md'
    vram_text = vram_report.read_text(encoding='utf-8')
    checks = {
        'fairness_fields_match': not mismatches,
        'identity_bank_count': bank_manifest['count'],
        'identity_bank_hash_matches_manifest': sha256_file(bank_path) == bank_manifest['sha256'],
        'train_recognizer_checkpoint': str(recognizer),
        'train_recognizer_hash': sha256_file(recognizer)[:16],
        'smoke_complete': smoke_status.get('status') == 'complete' and smoke_status.get('run_step') == 20,
        'smoke_decode_branches': branches,
        'vram_fallback_adopted': 'use latent_scale=0.50, decode_freq=3' in vram_text,
        'formal_output_absent': not (Path(a4['experiment']['output_root']) / a4['experiment']['id']).exists(),
    }
    failed = [name for name, value in checks.items() if name in {
        'fairness_fields_match',
        'identity_bank_hash_matches_manifest',
        'smoke_complete',
        'vram_fallback_adopted',
        'formal_output_absent',
    } and not value]
    if branches != [0, 1]:
        failed.append('smoke_decode_branches')
    if bank_manifest['count'] != len(read_ids(root / a4['data']['train_split'])):
        failed.append('identity_bank_count')
    payload = {'fields': fields, 'mismatches': mismatches, 'checks': checks, 'failed': failed}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
