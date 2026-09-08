"""Build role-separated M/I and supervised-H caches for a frozen dev manifest.

H is used only as the original-state diagnostic endpoint; never a garment,
head/pose or identity condition of cross-identity generation.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from input_only_probe import IDENTITY_KEYS, identity_payload, sha256, source_path


def validate_pairs(pairs, valid):
    if not pairs:
        raise ValueError('empty pairs')
    mids={r['mid'] for r in pairs}; jids={r['jid'] for r in pairs}
    if mids & jids:
        raise ValueError('M/I file pools must be disjoint')
    if not (mids|jids).issubset(valid):
        raise ValueError('diagnostic inputs must belong to old-val')
    if any(not str(x).isdigit() for x in mids|jids):
        raise ValueError('invalid path-like ID')
    keys=[(r['mid'],r['jid'],int(r['seed'])) for r in pairs]
    if len(keys)!=len(set(keys)):
        raise ValueError('duplicate pair')
    for mid in mids:
        local=[r for r in pairs if r['mid']==mid]
        if len(local)!=2 or len({r['jid'] for r in local})!=2:
            raise ValueError('exactly two reference identities required per M')
    return sorted(mids),sorted(jids)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--pairs-json',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:2')
    args=p.parse_args()
    payload=json.loads(args.pairs_json.read_text())
    pairs=payload['pairs']
    mids,jids=validate_pairs(pairs,set((args.root/'splits/val.txt').read_text().split()))
    args.out.mkdir(parents=True,exist_ok=False)
    for role in ('M','I','H_supervision','masks'):
        (args.out/role).mkdir()
    sys.path.insert(0,str(args.repo))
    import torch
    from PIL import Image
    import cv2
    from diffusers import AutoencoderKL
    from conditions import load_yaml,load_clip_vision,clip_patch_grid_feature
    from build_cache import encode_image_to_packed_latents
    from metrics_v2.parsing import FashnParser
    cfg=load_yaml(args.repo/'configs/spatial_hair_ab_space.yaml')
    device=torch.device(args.device); dtype=torch.bfloat16
    torch.cuda.set_device(device)
    start=time.monotonic()
    vae=AutoencoderKL.from_pretrained(cfg['model']['base'],subfolder='vae',torch_dtype=dtype,local_files_only=True).eval().to(device)
    vae.requires_grad_(False)
    clip=load_clip_vision(cfg['cache']['clip_vision_model'],device,dtype)
    parser=FashnParser(args.repo/'models/hf/fashn-ai/fashn-human-parser',args.device)
    old=args.root/cfg['data']['cache_dir']
    audit={'formal_result':False,'pairs':pairs,'config':cfg,'M_roles':[],'I_roles':[],'H_roles':[],
           'source_manifest_sha256':sha256(args.pairs_json),'script_sha256':sha256(__file__),
           'H_policy':'H_supervision endpoints ONLY for original-H-state diagnostic; never inference conditions',
           'I_policy':'only selected I-side fields from existing traced cache; not fully rebuilt here',
           'head_policy':'zero7-vector; M-derived with-head DWPose; no H-derived synthetic head control'}
    textpath=old/'text/prompt.npz'
    with np.load(textpath,allow_pickle=False) as z:
        np.savez(args.out/'prompt.npz',**{k:z[k] for k in ('prompt_embeds','pooled_prompt_embeds')})
    audit['text_source']={'path':str(textpath),'sha256':sha256(textpath)}
    for mid in mids:
        mp=source_path(args.root,'images/mannequin',mid)
        pp=source_path(args.root,'dwpose/with_head/mannequin',mid)
        im=Image.open(mp).convert('RGB'); arr=np.asarray(im)
        if im.size!=(768,1024): raise ValueError('Unexpected native size')
        mask=np.isin(parser.predict([arr])[0],(3,4,5,6,7,10)).astype(np.uint8)
        if mask.mean()<.005: raise ValueError('M parsing failure: '+mid)
        Image.fromarray(mask*255).save(args.out/'masks'/f'{mid}.png')
        yy,xx=np.nonzero(mask)
        crop=Image.fromarray(np.where(mask[:,:,None]>0,arr,255).astype(np.uint8)).crop((int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1))
        eroded=cv2.erode(mask,np.ones((5,5),np.uint8))
        if not eroded.any(): raise ValueError('empty eroded garment: '+mid)
        canvas=Image.fromarray(np.where(eroded[:,:,None]>0,arr,227).astype(np.uint8))
        source={'pose_latents':encode_image_to_packed_latents(vae,Image.open(pp).convert('RGB'),(768,1024),device,dtype),
                'garment_grid':clip_patch_grid_feature(clip,crop,device,dtype,max_tokens=64),
                'garment_ref_latents':encode_image_to_packed_latents(vae,canvas,(768,1024),device,dtype),
                'head_pose':np.zeros(7,np.float32)}
        np.savez(args.out/'M'/f'{mid}.npz',**source)
        audit['M_roles'].append({'mid':mid,'paths_sha256':{str(f):sha256(f) for f in (mp,pp)},'keys':list(source),'mask_area':float(mask.mean())})
        # Supervision is explicitly resolved only after M conditions are built.
        hp=next((args.root/'images/human'/f'{mid}{ext}' for ext in ('.jpg','.png','.jpeg') if (args.root/'images/human'/f'{mid}{ext}').is_file()),None)
        if hp is None: raise FileNotFoundError('H endpoint missing for '+mid)
        h=encode_image_to_packed_latents(vae,Image.open(hp).convert('RGB'),(768,1024),device,dtype)
        np.savez(args.out/'H_supervision'/f'{mid}.npz',target_latents=h)
        audit['H_roles'].append({'mid':mid,'path':str(hp),'sha256':sha256(hp),'role':'original state endpoint only'})
        print(json.dumps({'M_cached':len(audit['M_roles']),'total_M':len(mids)}),flush=True)
    for jid in jids:
        ip=old/'samples'/f'{jid}.npz'
        with np.load(ip,allow_pickle=False) as z: identity=identity_payload(z)
        if not all(np.isfinite(v).all() for v in identity.values()): raise ValueError('nonfinite I '+jid)
        np.savez(args.out/'I'/f'{jid}.npz',**identity)
        audit['I_roles'].append({'jid':jid,'path':str(ip),'sha256':sha256(ip),'keys':list(identity)})
    torch.cuda.synchronize(device)
    audit['resources']={'seconds':time.monotonic()-start,'gpu_hours':(time.monotonic()-start)/3600,
                         'peak_gib':torch.cuda.max_memory_allocated(device)/1024**3}
    audit['status']='diagnostic_cache_ready_not_formal_training'
    (args.out/'audit.json').write_text(json.dumps(audit,indent=2))
    (args.out/'READY').write_text(audit['status']+'\n')
    print(json.dumps({'status':audit['status'],'resources':audit['resources']}),flush=True)


if __name__=='__main__': main()
