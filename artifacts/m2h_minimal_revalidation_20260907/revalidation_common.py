"""Small contracts shared by the frozen-model input-protocol revalidation."""
import hashlib
import json
import os
from pathlib import Path
import sys

SOURCE_KEYS = ('pose_latents', 'garment_grid', 'head_pose')
IDENTITY_KEYS = ('pulid_id_embed', 'appearance')
ARMS = ('b2_legacyM_freshI', 'a4_legacyM_freshI',
        'b2_inputOnly_freshI', 'a4_inputOnly_freshI')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    tmp.replace(path)


def load_pairs(path):
    p = json.loads(Path(path).read_text())['pairs']
    keys = [(r['mid'], r['jid'], int(r['seed'])) for r in p]
    if len(keys) != len(set(keys)) or not keys:
        raise ValueError('empty/duplicate pair keys')
    if any(not str(x).isdigit() for key in keys for x in key[:2]):
        raise ValueError('non-numeric IDs')
    if {r['mid'] for r in p} & {r['jid'] for r in p}:
        raise ValueError('M/I file pools must be disjoint')
    return p


def image_path(root, role, sid):
    root = Path(root).resolve()
    if role not in ('images/mannequin', 'images/human',
                    'dwpose/without_head/mannequin') or not str(sid).isdigit():
        raise ValueError('invalid image role/id')
    for suffix in ('.png', '.jpg', '.jpeg'):
        p = root / role / (sid + suffix)
        if p.is_file():
            if not p.resolve().is_relative_to(root / role):
                raise ValueError('image symlink escapes role')
            return p
    raise FileNotFoundError(f'{role}/{sid}')


def install_guard(root, allowed):
    root = Path(root).resolve()
    allowed = {Path(p).resolve() for p in allowed}
    events = {'dataset_reads': [], 'denied_reads': [], 'denied_subprocesses': []}

    def hook(event, args):
        if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec'):
            events['denied_subprocesses'].append(event)
            raise PermissionError('subprocess disabled after model initialization')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(args[0])).resolve()
            if p.is_relative_to(root):
                if p not in allowed:
                    events['denied_reads'].append(str(p))
                    raise PermissionError('dataset read denied: ' + str(p))
                events['dataset_reads'].append(str(p))
    sys.addaudithook(hook)
    return events


def bait_test(root, mid):
    try:
        (Path(root) / 'images/human' / (mid + '.jpg')).open('rb').close()
    except PermissionError:
        return True
    raise RuntimeError('target read was not denied')


def validate_payload(payload, keys):
    import numpy as np
    if set(payload) != set(keys):
        raise ValueError('unexpected cache keys: ' + str(set(payload)))
    if any(not np.isfinite(v).all() for v in payload.values()):
        raise ValueError('nonfinite cache')
    if 'head_pose' in payload and payload['head_pose'].shape != (7,):
        raise ValueError('head pose shape')
    if 'pulid_id_embed' in payload:
        if payload['pulid_id_embed'].shape != (32, 2048):
            raise ValueError('PuLID shape')
        if payload['appearance'].shape != (1024,):
            raise ValueError('appearance shape')


def batch_from_arrays(m, i, text):
    import numpy as np
    import torch
    validate_payload(m, SOURCE_KEYS)
    validate_payload(i, IDENTITY_KEYS)
    if set(text) != {'prompt_embeds', 'pooled_prompt_embeds'}:
        raise ValueError('text keys')
    b = {k: torch.from_numpy(np.array(v)).float() for k, v in {**m, **i, **text}.items()}
    b['garment'] = b.pop('garment_grid')
    return b
