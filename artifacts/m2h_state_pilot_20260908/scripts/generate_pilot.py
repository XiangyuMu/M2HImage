"""Fixed noise-start teacher/dev generator; no target-image reads or score filtering."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from prepare_pool import sha


def main():
    p=argparse.ArgumentParser()
    for key in ('repo','root','cache','pairs','config','checkpoint','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--arm',required=True)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--count',type=int,required=True)
    p.add_argument('--endpoints',type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--max-hours',type=float,default=3)
    a=p.parse_args()
    sys.path[:0]=[str(a.repo),str(a.repo/'artifacts/m2h_minimal_revalidation_20260907')]
    from revalidation_common import load_pairs,write_json,install_guard,bait_test,batch_from_arrays,SOURCE_KEYS,IDENTITY_KEYS
    pairs=load_pairs(a.pairs)[a.start:a.start+a.count]
    assert len(pairs)==a.count and a.start>=0
    assert (a.cache/'READY').is_file() and (a.checkpoint/'READY').is_file()
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'outputs').mkdir()
    if a.endpoints:
        a.endpoints.mkdir(parents=True,exist_ok=True)
    import torch
    from peft import get_peft_model_state_dict,set_peft_model_state_dict
    from conditions import load_yaml,seed_everything
    from train_paired import load_components,WarmupFlowModel
    from eval_watcher import generate,decode_tokens
    cfg=load_yaml(a.config)
    cfg['model']['load_vae_in_train']=True
    dtype,device=torch.bfloat16,torch.device(a.device)
    torch.cuda.set_device(device)
    seed_everything(20260908)
    start=time.monotonic()
    manifest=dict(status='initializing',arm=a.arm,pair_manifest_sha256=sha(a.pairs),pairs=pairs,
                  checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint/'trainable.pt'),
                  cache_audit_sha256=sha(a.cache/'audit.json'),script_sha256=sha(__file__),
                  sampler='noise-start Euler tau1->0,20steps,seed0',training_steps=0,
                  endpoint_policy='fixed detached teacher states, never ground-truth labels; no score filtering',
                  condition_policy='M/I-only fresh cache; no H files',inputs=[])
    write_json(a.out/'manifest.json',manifest)
    try:
        tr,cn,vae,adapter,pulid,note=load_components(cfg,device,dtype)
        model=WarmupFlowModel(tr,cn,adapter,pulid,cfg)
        payload=torch.load(a.checkpoint/'trainable.pt',map_location='cpu',weights_only=False)
        allowed={'hair_gate','hair_pos_proj.weight','hair_proj.bias','hair_proj.weight'} if not model.spatial_hair_enabled else set()
        for key,current in [('adapter',adapter.state_dict()),('transformer_lora',get_peft_model_state_dict(tr))]:
            saved=payload[key]
            missing=set(current)-set(saved)
            if set(saved)-set(current) or (missing-(allowed if key=='adapter' else set())):
                raise ValueError('checkpoint keys mismatch '+key+str(missing))
            if any(saved[k].shape!=current[k].shape for k in saved):
                raise ValueError('checkpoint shape mismatch '+key)
        adapter.load_state_dict(payload['adapter'],strict=False)
        set_peft_model_state_dict(tr,payload['transformer_lora'])
        manifest['checkpoint_step']=int(payload['step'])
        if a.endpoints and int(payload['step'])!=8400:
            raise ValueError('teacher step changed')
        del payload
        model.eval().requires_grad_(False)
        events=install_guard(a.root,[])
        manifest['read_guard']=events
        manifest['target_denial_test_passed']=bait_test(a.root,pairs[0]['mid'])
        audit=json.loads((a.cache/'audit.json').read_text())
        hashes={str(a.cache/'M'/(r['mid']+'.npz')):r['cache_sha256'] for r in audit['M_roles']}
        hashes.update({str(a.cache/'I'/(r['jid']+'.npz')):r['cache_sha256'] for r in audit['I_roles']})
        data={}
        for r in pairs:
            for role,key,keys in [('M','mid',SOURCE_KEYS),('I','jid',IDENTITY_KEYS)]:
                path=a.cache/role/(r[key]+'.npz')
                if str(path) not in data:
                    digest=sha(path)
                    assert digest==hashes[str(path)],str(path)
                    with np.load(path,allow_pickle=False) as z:
                        data[str(path)]={k:z[k].copy() for k in keys}
                    manifest['inputs'].append(dict(path=str(path),sha256=digest,keys=list(keys)))
        with np.load(a.cache/'prompt.npz',allow_pickle=False) as z:
            text={k:z[k].copy() for k in ('prompt_embeds','pooled_prompt_embeds')}
        manifest['status']='generating'
        write_json(a.out/'manifest.json',manifest)
        with (a.out/'rows.jsonl').open('x') as f,torch.inference_mode():
            for i,r in enumerate(pairs):
                if time.monotonic()-start>a.max_hours*3600:
                    raise TimeoutError('generation budget exceeded')
                batch=batch_from_arrays(data[str(a.cache/'M'/(r['mid']+'.npz'))],data[str(a.cache/'I'/(r['jid']+'.npz'))],text)
                tick=time.monotonic()
                tokens=generate(model,batch,20,int(r['seed']),device,dtype)
                if not torch.isfinite(tokens).all():
                    raise ValueError('nonfinite endpoint')
                path=a.out/'outputs'/f"{r['mid']}__id{r['jid']}__seed{r['seed']}.png"
                decode_tokens(vae,tokens,cfg['data']['resolution']).save(path)
                row=dict(r,branch=a.arm,tau=0,k=1,path=str(path.resolve()),sha256=sha(path),clean_carryin_path=str(path.resolve()),
                         carryin_adapter_note='self path only; do not use carryin metrics',seconds=time.monotonic()-tick)
                if a.endpoints:
                    ep=a.endpoints/f"{r['mid']}__{r['jid']}.npz"
                    if ep.exists():
                        raise FileExistsError('refuse overwrite endpoint '+str(ep))
                    with ep.with_suffix('.tmp').open('xb') as h:
                        np.savez(h,target_latents=tokens[0].detach().float().cpu().numpy())
                        h.flush();os.fsync(h.fileno())
                    ep.with_suffix('.tmp').replace(ep)
                    row.update(endpoint=str(ep),endpoint_sha256=sha(ep))
                f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                prog=dict(completed=i+1,expected=len(pairs),seconds=time.monotonic()-start,last_image_seconds=row['seconds'])
                write_json(a.out/'progress.json',prog)
                print(json.dumps(prog),flush=True)
        assert len(events['denied_reads'])==1 and not events['denied_subprocesses']
        manifest.update(status='complete',resources=dict(gpu_hours=(time.monotonic()-start)/3600,seconds=time.monotonic()-start))
        write_json(a.out/'manifest.json',manifest)
        (a.out/'READY').write_text('complete; metrics pending\n')
    except Exception as e:
        manifest.update(status='failed',error=repr(e),seconds=time.monotonic()-start)
        write_json(a.out/'manifest.json',manifest)
        raise


if __name__=='__main__':
    main()
