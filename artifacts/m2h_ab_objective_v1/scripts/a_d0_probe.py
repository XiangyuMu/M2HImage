"""Frozen-model A-D0 state-condition matching diagnostic, no optimization.

Same teacher endpoint multiset is used by matched and mismatched states.
Reports endpoints and the identity carry-in explicitly. Sampling midpoints
of three noise windows does not claim full-window integration.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from input_only_probe import sha256, SOURCE_KEYS


def make_state(endpoint,noise,tau):
    if not 0<=tau<=1: raise ValueError('tau outside [0,1]')
    return (1-tau)*endpoint+tau*noise


def other_reference(pairs,mid,jid):
    others=sorted({r['jid'] for r in pairs if r['mid']==mid and r['jid']!=jid})
    if len(others)!=1: raise ValueError('need exactly two references per M')
    return others[0]


def load_batch(cache,mid,jid):
    import torch
    with np.load(cache/'M'/f'{mid}.npz',allow_pickle=False) as m, np.load(cache/'I'/f'{jid}.npz',allow_pickle=False) as j,np.load(cache/'prompt.npz',allow_pickle=False) as text:
        batch={k:torch.from_numpy(np.array(m[k])).float() for k in SOURCE_KEYS}
        batch['garment']=batch.pop('garment_grid')
        batch.update({k:torch.from_numpy(np.array(j[k])).float() for k in j.files})
        batch.update({k:torch.from_numpy(np.array(text[k])).float() for k in text.files})
    assert 'target_latents' not in batch
    return batch


def context(model,batch,device,dtype):
    from conditions import make_image_ids
    prompt=batch['prompt_embeds'].to(device=device,dtype=dtype).unsqueeze(0)
    pooled=batch['pooled_prompt_embeds'].to(device=device,dtype=dtype).unsqueeze(0)
    tokens=model._condition_tokens(prompt,batch['appearance'].to(device=device,dtype=dtype).unsqueeze(0),
        batch['garment'].to(device=device,dtype=dtype).unsqueeze(0),batch['head_pose'].to(device=device,dtype=dtype).unsqueeze(0),
        *model._hair_inputs(batch,device,dtype))
    return prompt,pooled,tokens,make_image_ids(model.width,model.height,device,dtype)


def velocity(model,batch,ctx,z,tau,device,dtype):
    import torch
    prompt,pooled,tokens,img_ids=ctx
    t=torch.full((1,),tau,device=device,dtype=dtype)
    cn=model._controlnet_forward(z,t,prompt,pooled,batch['pose_latents'].to(device=device,dtype=dtype).unsqueeze(0),img_ids)
    return model._transformer_forward(z,t,tokens,batch['pulid_id_embed'].to(device=device,dtype=dtype).unsqueeze(0),cn,
        pooled=pooled,img_ids=img_ids,garment_ref_latents=batch.get('garment_ref_latents'),hair_ref_latents=batch.get('hair_ref_latents'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--limit-m',type=int,default=0)
    p.add_argument('--teacher-steps',type=int,default=20)
    p.add_argument('--sensitivity-m',type=int,default=8)
    p.add_argument('--max-hours',type=float,default=6)
    args=p.parse_args()
    if not 0<=args.shard<args.shards: p.error('bad shard index')
    if not (args.cache/'READY').exists(): raise RuntimeError('cache not ready')
    audit=json.loads((args.cache/'audit.json').read_text()); cfg=audit['config']; pairs=audit['pairs']
    all_mids=sorted({r['mid'] for r in pairs})
    mids=[m for i,m in enumerate(all_mids) if i%args.shards==args.shard]
    if args.limit_m: mids=mids[:args.limit_m]
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('teacher','teacher_latents','original_H','predictions','carryin_state'):
        (args.out/name).mkdir()
    sys.path.insert(0,str(args.repo))
    import torch
    from train_paired import load_components,WarmupFlowModel,load_checkpoint
    from eval_watcher import generate,decode_tokens
    device=torch.device(args.device); dtype=torch.bfloat16
    torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    start=time.monotonic()
    transformer,controlnet,vae,adapter,pulid,_=load_components(cfg,device,dtype)
    model=WarmupFlowModel(transformer,controlnet,adapter,pulid,cfg)
    checkpoint=Path(cfg['data']['root'])/'phase1/phase1_spatial_hair_ab_space_r16_8400_768x1024/checkpoints/final'
    load_checkpoint(checkpoint,model); model.eval(); model.requires_grad_(False)
    if vae is None: raise RuntimeError('VAE expected from selected config')
    manifest={'experiment':'A-D0','formal_result':False,'training_steps':0,
        'pairs':[r for r in pairs if r['mid'] in mids],'all_mids':all_mids,'selected_mids':mids,
        'teacher_steps':args.teacher_steps,'tau_points':[.1,.4,.8],
        'noise_windows':[[0,.2],[.2,.6],[.6,1]],
        'endpoint_estimators':[1,4],'k4_scope_mids':all_mids[:args.sensitivity_m],
        'cache_audit_sha256':sha256(args.cache/'audit.json'),'script_sha256':sha256(__file__),
        'checkpoint':str(checkpoint),'checkpoint_trainable_sha256':sha256(checkpoint/'trainable.pt'),
        'warning':'Matched endpoint carries the requested identity before denoising; compare carry-in and prediction, not raw scores alone.',
        'mismatch':'Swap the two endpoints for each M; exact same endpoint multiset, t, noise and target condition pool.',
        'H_role':'H latent only initializes original-state branch; never a model condition',
        'resume_policy':'Fresh output dirs; do not merge or silently skip failed pairs'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    written=0; done_m=0
    with (args.out/'rows.jsonl').open('x') as handle, torch.inference_mode():
        for mid in mids:
            if time.monotonic()-start>args.max_hours*3600: raise TimeoutError('A-D0 budget exhausted')
            local=[r for r in pairs if r['mid']==mid]
            teacher={}; batches={}; contexts={}
            for r in local:
                jid=r['jid']; batch=load_batch(args.cache,mid,jid)
                batches[jid]=batch
                endpoint=generate(model,batch,args.teacher_steps,seed=int(r['seed']),device=device,dtype=dtype).detach()
                if not torch.isfinite(endpoint).all(): raise ValueError('nonfinite teacher')
                teacher[jid]=endpoint
                torch.save(endpoint.cpu(),args.out/'teacher_latents'/f'{mid}__id{jid}.pt')
                decode_tokens(vae,endpoint,cfg['data']['resolution']).save(args.out/'teacher'/f'{mid}__id{jid}__seed{r["seed"]}.png')
                contexts[jid]=context(model,batch,device,dtype)
            with np.load(args.cache/'H_supervision'/f'{mid}.npz',allow_pickle=False) as z:
                original=torch.from_numpy(np.array(z['target_latents'])).unsqueeze(0).to(device=device,dtype=dtype)
            original_image=args.out/'original_H'/f'{mid}.png'
            decode_tokens(vae,original,cfg['data']['resolution']).save(original_image)
            for r in local:
                jid=r['jid']; donor=other_reference(pairs,mid,jid); batch=batches[jid]; ctx=contexts[jid]
                noise=torch.randn(original.shape,device=device,dtype=dtype,generator=torch.Generator(device=device).manual_seed(20260907+int(r['seed'])))
                endpoints={'original':original,'matched':teacher[jid],'mismatched':teacher[donor]}
                for tau in (.1,.4,.8):
                    for branch,endpoint in endpoints.items():
                        state=make_state(endpoint.float(),noise.float(),tau).to(dtype)
                        v=velocity(model,batch,ctx,state,tau,device,dtype)
                        estimate=(state.float()-tau*v.float()).to(dtype)
                        for k in ([1,4] if mid in all_mids[:args.sensitivity_m] else [1]):
                            y=estimate
                            if k==4:
                                y=(state.float()-(tau/4)*v.float()).to(dtype)
                                for step in range(1,4):
                                    vt=velocity(model,batch,ctx,y,tau*(1-step/4),device,dtype)
                                    y=(y.float()-(tau/4)*vt.float()).to(dtype)
                            if not torch.isfinite(y).all(): raise ValueError('nonfinite prediction')
                            name=f'{mid}__id{jid}__t{tau:.1f}__{branch}__k{k}.png'
                            path=args.out/'predictions'/name
                            decode_tokens(vae,y,cfg['data']['resolution']).save(path)
                            carry=original_image if branch=='original' else args.out/'teacher'/f'{mid}__id{jid if branch=="matched" else donor}__seed{r["seed"]}.png'
                            statefile=None
                            if mid in all_mids[:args.sensitivity_m] and k==1:
                                statefile=args.out/'carryin_state'/name
                                decode_tokens(vae,state,cfg['data']['resolution']).save(statefile)
                            row={**r,'branch':branch,'tau':tau,'k':k,'donor_jid':donor,'path':str(path.resolve()),
                                 'clean_carryin_path':str(carry.resolve()),'noisy_state_path':str(statefile.resolve()) if statefile else None,
                                 'latent_mse_to_matched_teacher':float((y.float()-teacher[jid].float()).square().mean()),
                                 'latent_displacement_from_carryin':float((y.float()-endpoint.float()).square().mean()),
                                 'status':'ok'}
                            handle.write(json.dumps(row,allow_nan=False)+'\n'); handle.flush(); written+=1
            done_m+=1
            progress={'completed_m':done_m,'expected_m':len(mids),'predictions':written,'seconds':time.monotonic()-start}
            (args.out/'progress.json').write_text(json.dumps(progress)); print(json.dumps(progress),flush=True)
    torch.cuda.synchronize(device)
    summary={'status':'A_D0_generation_complete','training_steps':0,'mids':done_m,'pairs':done_m*2,'predictions':written,
             'seconds':time.monotonic()-start,'gpu_hours':(time.monotonic()-start)/3600,
             'peak_gib':torch.cuda.max_memory_allocated(device)/1024**3,'metrics_status':'external objective evaluation pending'}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    (args.out/'READY').write_text('generation complete; objective evaluation pending\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
