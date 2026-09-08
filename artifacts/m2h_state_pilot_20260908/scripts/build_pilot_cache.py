"""Fresh separated input roles and explicit legacy H supervision for small pilot."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from prepare_pool import sha


def main():
    p = argparse.ArgumentParser()
    for key in ('root', 'repo', 'pool', 'config', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--max-hours', type=float, default=2)
    a = p.parse_args()
    rows = json.loads(a.pool.read_text())['rows']
    mids = sorted({r['mid'] for r in rows})
    refs = sorted({r[k] for r in rows for k in ('jid','kid')})
    assert len(mids) == 256 and not set(mids) & set(refs)
    sys.path[:0] = [str(a.repo), str(a.repo/'artifacts/m2h_minimal_revalidation_20260907')]
    from revalidation_common import image_path, install_guard, validate_payload, SOURCE_KEYS, IDENTITY_KEYS, write_json
    import torch
    import cv2
    from PIL import Image
    from diffusers import AutoencoderKL
    from conditions import load_yaml, load_clip_vision, clip_patch_grid_feature, pooled_clip_feature, expanded_head_crop, mask_bbox
    from build_cache import encode_image_to_packed_latents
    from pulid_flux import PuLIDIdentityEmbedder
    from metrics_v2.parsing import FashnParser
    a.out.mkdir(parents=True, exist_ok=False)
    for key in ('M','I','H','masks'):
        (a.out/key).mkdir()
    cfg = load_yaml(a.config)
    device, dtype = torch.device(a.device), torch.bfloat16
    torch.cuda.set_device(device)
    start = time.monotonic()
    audit = dict(status='initializing', pool_sha256=sha(a.pool), M_roles=[], I_roles=[], H_roles=[],
                 roles='M conditions only original M+Mpose. I from full supplied human image; paired I_mid is legal paired identity input, CF I_j/k separate. H reused only target/masks; never H-derived garment/head M conditions.',
                 guard_limitation='Python file audit, not native syscall proof')
    write_json(a.out/'audit.json', audit)
    vae = AutoencoderKL.from_pretrained(cfg['model']['base'], subfolder='vae', torch_dtype=dtype,
                                      local_files_only=True).to(device).eval().requires_grad_(False)
    clip = load_clip_vision(cfg['cache']['clip_vision_model'], device, dtype)
    parser = FashnParser(a.repo/'models/hf/fashn-ai/fashn-human-parser', a.device)
    embedder = PuLIDIdentityEmbedder(cfg['model']['pulid'], device, dtype)
    bankpath = a.root/'derived/identity_bank_v2.npz'
    with np.load(bankpath, allow_pickle=False) as z:
        bank = {str(k): v.copy() for k, v in zip(z['ids'], z['embeds'])}
    audit['train_bank_sha256'] = sha(bankpath)
    audit['train_recognizer_sha256'] = sha(cfg['model']['train_recognizer']['checkpoint'])
    assert audit['train_recognizer_sha256'].startswith('4ab1d6435d639628')
    textpath = a.root/cfg['data']['cache_dir']/'text/prompt.npz'
    samples = a.root/cfg['data']['cache_dir']/'samples'
    allowed = [textpath]
    allowed += [image_path(a.root, role, m) for m in mids for role in ('images/mannequin','dwpose/without_head/mannequin')]
    allowed += [image_path(a.root, 'images/human', i) for i in mids+refs]
    allowed += [samples/(m+'.npz') for m in mids]
    allowed += [a.root/'derived/region_masks_z'/(m+'.npz') for m in mids]
    audit['read_guard'] = install_guard(a.root, allowed)
    audit['allowed_dataset_paths'] = list(map(str, allowed))
    try:
        (a.root/'images/human'/'NOT_AUTHORIZED.jpg').open('rb')
        raise RuntimeError('guard failed')
    except PermissionError:
        audit['undeclared_read_denied'] = True
    def progress(status):
        audit['status'] = status
        audit['seconds'] = time.monotonic()-start
        write_json(a.out/'audit.json',audit)
        print(json.dumps(dict(status=status, M=len(audit['M_roles']), I=len(audit['I_roles']), H=len(audit['H_roles']), seconds=audit['seconds'])),flush=True)
        if audit['seconds'] > 3600*a.max_hours:
            raise TimeoutError('cache budget reached')
    try:
        with np.load(textpath, allow_pickle=False) as z:
            np.savez(a.out/'prompt.npz', **{k:z[k] for k in ('prompt_embeds','pooled_prompt_embeds')})
        with torch.inference_mode():
            for m in mids:
                mp, pp = [image_path(a.root, role, m) for role in ('images/mannequin','dwpose/without_head/mannequin')]
                im=Image.open(mp).convert('RGB')
                assert im.size == (768,1024)
                arr=np.asarray(im)
                mask=np.isin(parser.predict([arr])[0],(3,4,5,6,7,10)).astype(np.uint8)
                if mask.mean()<.005:
                    raise ValueError('empty source mask '+m)
                maskim=Image.fromarray(mask*255)
                maskim.save(a.out/'masks'/(m+'.png'))
                crop=Image.fromarray(np.where(mask[:,:,None],arr,255).astype(np.uint8)).crop(mask_bbox(maskim))
                val=dict(pose_latents=encode_image_to_packed_latents(vae,Image.open(pp),cfg['data']['resolution'],device,dtype,output_dtype='float16'),
                         garment_grid=clip_patch_grid_feature(clip,crop,device,dtype,max_tokens=64),head_pose=np.zeros(7,np.float32))
                validate_payload(val,SOURCE_KEYS)
                dest=a.out/'M'/(m+'.npz')
                np.savez(dest,**val)
                audit['M_roles'].append(dict(mid=m,cache_sha256=sha(dest),inputs={str(x):sha(x) for x in (mp,pp)},keys=list(val)))
                hp=samples/(m+'.npz'); maskp=a.root/'derived/region_masks_z'/(m+'.npz')
                with np.load(hp,allow_pickle=False) as z:
                    h=dict(target_latents=z['target_latents'].copy())
                with np.load(maskp,allow_pickle=False) as z:
                    h.update({k:z[k].copy() for k in ('cloth_safe_z','body_bg_z','face_z')})
                assert h['target_latents'].shape==(3072,64)
                assert all(h[k].shape==(3072,) for k in ('cloth_safe_z','body_bg_z','face_z'))
                assert all(np.isfinite(v).all() for v in h.values())
                dest=a.out/'H'/(m+'.npz'); np.savez(dest,**h)
                audit['H_roles'].append(dict(mid=m,cache_sha256=sha(dest),supervision_sources={str(x):sha(x) for x in (hp,maskp)},keys=list(h)))
                progress('building_M_H')
            for i in mids+refs:
                ip=image_path(a.root,'images/human',i)
                im=Image.open(ip).convert('RGB')
                faces=embedder.pipeline.app.get(cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR))
                if not faces:
                    raise ValueError('reference face missing '+i)
                face=max(faces,key=lambda f:float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
                crop=expanded_head_crop(im,tuple(map(float,face.bbox)))
                val=dict(pulid_id_embed=embedder.embed_image(im).astype('float32'),appearance=pooled_clip_feature(clip,crop,device,dtype).astype('float16'))
                validate_payload(val,IDENTITY_KEYS)
                val['train_embed']=bank[i].astype('float32')
                dest=a.out/'I'/(i+'.npz');np.savez(dest,**val)
                audit['I_roles'].append(dict(jid=i,role='paired_identity' if i in mids else 'CF_reference',cache_sha256=sha(dest),input_sha256=sha(ip),det_score=float(face.det_score),keys=list(val)))
                progress('building_I')
        assert len(audit['read_guard']['denied_reads'])==1
        assert not audit['read_guard']['denied_subprocesses']
        audit['resources']=dict(gpu_hours=(time.monotonic()-start)/3600,peak_gib=torch.cuda.max_memory_allocated(device)/1024**3)
        progress('complete')
        (a.out/'READY').write_text('Fresh M/I and separated H supervision ready\n')
    except Exception as e:
        audit.update(status='failed',error=repr(e),seconds=time.monotonic()-start)
        write_json(a.out/'audit.json',audit)
        raise


if __name__=='__main__':
    main()
