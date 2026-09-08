"""Reuse hash-pinned generation unchanged; extend only the A2 arm label.

Strict checkpoint shape/key verification precedes the historical loader.
"""
import hashlib
import json
from pathlib import Path
import runpy
import sys

P = Path('/data/muxiangyu/pythonPrograms/M2HImage')
R = P / 'artifacts/m2h_minimal_revalidation_20260907'
PINNED = {
    'generate_revalidation.py': '5b8678903f0b21072bffa7aa4649a84c0398734bdeaa3503cb9d13511ad97fb7',
    'revalidation_common.py': 'd67a81c3004ec8895065cd5bdae2a56284b8271cad43c6179ce40408be886e49',
}


def check_state(saved, current, allowed_missing=()):
    missing = set(current) - set(saved)
    extra = set(saved) - set(current)
    shapes = [k for k in set(saved) & set(current) if tuple(saved[k].shape) != tuple(current[k].shape)]
    if missing - set(allowed_missing) or extra or shapes:
        raise ValueError(f'checkpoint mismatch: missing={missing}, extra={extra}, shapes={shapes}')
    return sorted(missing)


def main():
    for name, expected in PINNED.items():
        if hashlib.sha256((R/name).read_bytes()).hexdigest() != expected:
            raise ValueError('historical generator changed: '+name)
    sys.path[:0] = [str(R), str(P)]
    import revalidation_common as common
    common.ARMS = (*common.ARMS, 'a2_inputOnly_freshI')
    import train_paired
    original = train_paired.load_checkpoint

    def checked(path, model, *args, **kwargs):
        import torch
        from peft import get_peft_model_state_dict
        payload = torch.load(path/'trainable.pt', map_location='cpu', weights_only=False)
        allowed = ('hair_gate', 'hair_pos_proj.weight', 'hair_proj.bias', 'hair_proj.weight')
        if getattr(model, 'spatial_hair_enabled', False):
            allowed = ()
        missing = check_state(payload['adapter'], model.adapter.state_dict(), allowed)
        check_state(payload['transformer_lora'], get_peft_model_state_dict(model.transformer))
        result = original(path, model, *args, **kwargs)
        out = Path(sys.argv[sys.argv.index('--out')+1])
        common.write_json(out/'checkpoint_compatibility.json', {
            'step': result, 'inactive_missing_adapter_keys': missing,
            'active_missing_keys': [], 'lora_keys_shapes_match': True,
            'wrapper_sha256': common.sha256(__file__),
            'source_sha256': {str(P/n): common.sha256(P/n) for n in
                              ('train_paired.py', 'eval_watcher.py', 'conditions.py')},
        })
        return result

    train_paired.load_checkpoint = checked
    runpy.run_path(str(R/'generate_revalidation.py'), run_name='__main__')


if __name__ == '__main__':
    main()
