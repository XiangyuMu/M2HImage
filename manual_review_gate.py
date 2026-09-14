from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


APPROVAL_FILENAME = 'HUMAN_REVIEW_APPROVED.json'
DEFAULT_CHECKLIST = (
    'garment_same_item',
    'hair_matches_reference',
    'head_follows_mannequin',
)


def required_checklist(cfg: dict[str, Any]) -> tuple[str, ...]:
    values = cfg.get('eval', {}).get('watcher_manual_checklist', DEFAULT_CHECKLIST)
    checklist = tuple(str(value) for value in values)
    if not checklist:
        raise RuntimeError(
            'eval.watcher_manual_checklist cannot be empty when manual_review_step is enabled'
        )
    return checklist


def review_status(
    cfg: dict[str, Any],
    experiment_dir: str | Path,
    completed_run_steps: int,
) -> dict[str, Any]:
    experiment_dir = Path(experiment_dir)
    required_step = int(cfg.get('eval', {}).get('manual_review_step', 0) or 0)
    checklist = required_checklist(cfg)
    approval_path = experiment_dir / APPROVAL_FILENAME
    result: dict[str, Any] = {
        'enabled': required_step > 0,
        'due': required_step > 0 and int(completed_run_steps) >= required_step,
        'approved': False,
        'required_step': required_step,
        'completed_run_steps': int(completed_run_steps),
        'checklist': list(checklist),
        'approval_path': str(approval_path),
        'reason': 'not_due',
    }
    if not result['enabled']:
        result['reason'] = 'disabled'
        return result
    if not result['due']:
        return result
    if not approval_path.exists():
        result['reason'] = 'approval_missing'
        return result
    try:
        payload = json.loads(approval_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        result['reason'] = f'approval_unreadable:{exc}'
        return result
    result['approval'] = payload
    if str(payload.get('experiment_id')) != str(cfg.get('experiment', {}).get('id')):
        result['reason'] = 'experiment_id_mismatch'
        return result
    if int(payload.get('review_step', -1)) != required_step:
        result['reason'] = 'review_step_mismatch'
        return result
    checks = payload.get('checks', {})
    missing = [name for name in checklist if checks.get(name) is not True]
    if missing:
        result['reason'] = f'unapproved_checks:{missing}'
        return result
    if not str(payload.get('reviewer', '')).strip():
        result['reason'] = 'reviewer_missing'
        return result
    result['approved'] = True
    result['reason'] = 'approved'
    return result


def write_approval(
    cfg: dict[str, Any],
    experiment_dir: str | Path,
    reviewer: str,
) -> Path:
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError('reviewer must be non-empty')
    required_step = int(cfg.get('eval', {}).get('manual_review_step', 0) or 0)
    if required_step <= 0:
        raise RuntimeError('eval.manual_review_step must be positive before approving a review')
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    destination = experiment_dir / APPROVAL_FILENAME
    payload = {
        'experiment_id': str(cfg['experiment']['id']),
        'review_step': required_step,
        'reviewer': reviewer,
        'approved_unix_time': time.time(),
        'checks': {name: True for name in required_checklist(cfg)},
    }
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    temporary.replace(destination)
    return destination
