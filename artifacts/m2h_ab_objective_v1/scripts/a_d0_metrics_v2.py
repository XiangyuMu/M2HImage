"""Objective A-D0 image/identity/pose evaluation, isolated model roles.

Use refton_m2h Python for image stage and StableAnimator for id/pose stages.
No AdaFace is imported. Missing output detections remain in penalized means.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


def row_key(r): return (r['mid'],r['jid'],r['branch'],r['tau'],r['k'])


def summarize(rows,fields):
    result={}
    keys=sorted({(r['branch'],r['tau'],r['k']) for r in rows})
    for branch,tau,k in keys:
        selected=[r for r in rows if (r['branch'],r['tau'],r['k'])==(branch,tau,k)]
        entry={'n':len(selected),'status_counts':{s:sum(r['metric_status']==s for r in selected) for s in sorted({r['metric_status'] for r in selected})}}
        for field in fields:
            vals=[r[field] for r in selected if r.get(field) is not None and np.isfinite(r[field])]
            entry[field]={'n':len(vals),'mean':float(np.mean(vals)) if vals else None}
        result[f'{branch}__t{tau}__k{k}']=entry
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=['image','id','pose'],required=True)
    p.add_argument('--device',default='cuda:3')
    args=p.parse_args()
    if not (args.run/'READY').exists(): raise RuntimeError('generation is not complete')
    out=args.run/f'metrics_{args.stage}'
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(args.repo))
    from PIL import Image
    import cv2
    from input_only_probe import sha256
    audit=json.loads((args.cache/'audit.json').read_text()); root=Path(audit['config']['data']['root'])
    rows=[json.loads(line) for line in (args.run/'rows.jsonl').read_text().splitlines()]
    if len({row_key(r) for r in rows})!=len(rows): raise ValueError('duplicate diagnostic rows')
    start=time.monotonic(); cache={}; metrics=[]
    read=lambda path:np.asarray(Image.open(path).convert('RGB'))
    info={'stage':args.stage,'formal_result':False,'script_sha256':sha256(__file__),
          'generation_rows_sha256':sha256(args.run/'rows.jsonl'),'missing_outputs':'penalty and failure counts, never silent exclusion'}
    if args.stage=='image':
        import torch
        from conditions import load_yaml
        from metrics_v2.features import RegionFeatureExtractor,cosine
        cfg=load_yaml(args.repo/'configs/metrics_v2.yaml')
        model=RegionFeatureExtractor(cfg,args.device)
        fields=['garment_dino','garment_hf_lpips','carryin_dino','carryin_hf_lpips','dino_change_from_carryin','hf_lpips_change_from_carryin']
        info.update(source_mask='input-only M mask fixed for output and carry-in',sigma=2,crop_size=256,
                    dino_checkpoint_sha256=sha256(cfg['metrics_v2']['dino']['checkpoint']))
        def image_metric(path,mid):
            key=(str(path),mid)
            if key not in cache:
                sourcekey=('source',mid)
                if sourcekey not in cache:
                    source=read(root/'images/mannequin'/f'{mid}.png')
                    mask=(np.asarray(Image.open(args.cache/'masks'/f'{mid}.png'))>127).astype(np.uint8)
                    cache[sourcekey]=(source,mask,model.dino_feature(source,mask))
                source,mask,ref=cache[sourcekey]
                image=read(path)
                cache[key]=(float(cosine(ref,model.dino_feature(image,mask))),
                            model.high_frequency_lpips_distance(source,mask,image,mask,sigma=2,output_size=256))
            return cache[key]
    elif args.stage=='id':
        os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
        import insightface
        device_id=int(args.device.split(':')[-1])
        providers=[('CUDAExecutionProvider',{'device_id':device_id}),'CPUExecutionProvider']
        app=insightface.app.FaceAnalysis(name='antelopev2',root='/data/muxiangyu/modelLibrary/insightface',allowed_modules=['detection'],providers=providers)
        app.prepare(ctx_id=device_id,det_size=(640,640))
        weight=Path('/home/muxiangyu/.insightface/models/buffalo_l/w600k_r50.onnx')
        rec=insightface.model_zoo.get_model(str(weight),providers=providers); rec.prepare(ctx_id=device_id)
        if 'CUDAExecutionProvider' not in rec.session.get_providers(): raise RuntimeError('unexpected CPU fallback')
        info.update(identity_role='frozen development w600k_r50, not final AdaFace',weight_sha256=sha256(weight))
        fields=['id_cosine','id_penalized','carryin_id_cosine','id_change_from_carryin','noisy_state_id_cosine','id_change_from_noisy_state',
                'carryin_id_penalized','id_penalized_change_from_carryin','noisy_state_id_penalized','id_penalized_change_from_noisy_state']
        def embed(path):
            key=str(path)
            if key not in cache:
                try:
                    bgr=cv2.imread(key)
                    if bgr is None: raise ValueError('unreadable image')
                    faces=app.get(bgr)
                    if not faces: raise ValueError('face missing')
                    face=max(faces,key=lambda f:float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
                    if face.det_score<.3: raise ValueError('face confidence below .3')
                    v=np.asarray(rec.get(bgr,face)).reshape(-1)
                    if not np.isfinite(v).all() or np.linalg.norm(v)<1e-6: raise ValueError('bad embedding')
                    cache[key]=(v/np.linalg.norm(v),None)
                except Exception as exc: cache[key]=(None,str(exc))
            return cache[key]
    else:
        # CUDA_VISIBLE_DEVICES must be set by the caller: helper sessions use index0.
        from metrics_v2.dwpose_server import DWPoseBatchModel
        model=DWPoseBatchModel(Path('/data/muxiangyu/pythonPrograms/StableAnimator/DWPose'),
            Path('/data/muxiangyu/pythonPrograms/StableAnimator/checkpoints/DWPose/yolox_l.onnx'),
            Path('/data/muxiangyu/pythonPrograms/StableAnimator/checkpoints/DWPose/dw-ll_ucoco_384.onnx'),'CUDAExecutionProvider')
        fields=['body_distance','body_penalized','head_distance','head_penalized','body_coverage','head_coverage']
        fields += ['carryin_'+name for name in fields[:]]
        info.update(input_qualification='M keypoints confidence >= .3',failure_penalty='missing generated point receives normalized distance1')
        def pose(path):
            if str(path) not in cache:
                image=read(path); xy,score=model(image)
                xy=xy/np.array([image.shape[1],image.shape[0]],np.float32)
                cache[str(path)]=(xy,score)
            return cache[str(path)]
    with (out/'per_image.jsonl').open('x') as handle:
        for i,r in enumerate(rows):
            value={k:r[k] for k in ('mid','jid','seed','branch','tau','k','path')}
            value['metric_status']='ok'
            try:
                if args.stage=='image':
                    d,h=image_metric(r['path'],r['mid']); cd,ch=image_metric(r['clean_carryin_path'],r['mid'])
                    value.update(garment_dino=d,garment_hf_lpips=h,carryin_dino=cd,carryin_hf_lpips=ch,dino_change_from_carryin=d-cd,hf_lpips_change_from_carryin=h-ch)
                elif args.stage=='id':
                    refpath=next((root/'images/human'/f'{r["jid"]}{ext}' for ext in ('.jpg','.png','.jpeg') if (root/'images/human'/f'{r["jid"]}{ext}').exists()),None)
                    ref,err=embed(refpath)
                    if ref is None:
                        value.update(metric_status='reference_failed',error=err,id_penalized=None)
                    else:
                        emb,err=embed(r['path']); carry,_=embed(r['clean_carryin_path'])
                        score=float(np.clip(ref@emb,-1,1)) if emb is not None else None
                        cscore=float(np.clip(ref@carry,-1,1)) if carry is not None else None
                        value.update(id_cosine=score,id_penalized=score if score is not None else -1,
                                     carryin_id_cosine=cscore,id_change_from_carryin=score-cscore if score is not None and cscore is not None else None)
                        value.update(carryin_id_penalized=cscore if cscore is not None else -1,
                            id_penalized_change_from_carryin=(score if score is not None else -1)-(cscore if cscore is not None else -1))
                        if r.get('noisy_state_path'):
                            noisy,_=embed(r['noisy_state_path'])
                            nscore=float(np.clip(ref@noisy,-1,1)) if noisy is not None else None
                            value.update(noisy_state_id_cosine=nscore,id_change_from_noisy_state=score-nscore if score is not None and nscore is not None else None)
                            value.update(noisy_state_id_penalized=nscore if nscore is not None else -1,
                                id_penalized_change_from_noisy_state=(score if score is not None else -1)-(nscore if nscore is not None else -1))
                        if emb is None: value.update(metric_status='output_face_failed',error=err)
                else:
                    targetpath=root/'dwpose/keypoints/mannequin'/f'{r["mid"]}.npz'
                    if str(targetpath) not in cache:
                        with np.load(targetpath) as z: cache[str(targetpath)]=(np.array(z['body']),np.array(z['body_scores']))
                    target,scores=cache[str(targetpath)]
                    try: generated,gs=pose(r['path'])
                    except Exception as exc:
                        generated=np.zeros_like(target); gs=np.zeros_like(scores); value.update(metric_status='pose_failed',error=str(exc))
                    for name,indices in [('body',list(range(1,14))),('head',[0,14,15,16,17])]:
                        eligible=[idx for idx in indices if scores[idx]>=.3]
                        valid=[idx for idx in eligible if gs[idx]>=.3]
                        dist=np.linalg.norm((generated-target)*np.array([768,1024]),axis=1)/1280
                        value[name+'_distance']=float(dist[valid].mean()) if valid else None
                        value[name+'_penalized']=float(sum(min(float(dist[idx]),1.) if gs[idx]>=.3 else 1. for idx in eligible)/len(eligible)) if eligible else None
                        value[name+'_coverage']=len(valid)/len(eligible) if eligible else None
                    try: generated,gs=pose(r['clean_carryin_path'])
                    except Exception as exc:
                        generated=np.zeros_like(target); gs=np.zeros_like(scores); value['carryin_pose_error']=str(exc)
                    for name,indices in [('body',list(range(1,14))),('head',[0,14,15,16,17])]:
                        eligible=[idx for idx in indices if scores[idx]>=.3]
                        valid=[idx for idx in eligible if gs[idx]>=.3]
                        dist=np.linalg.norm((generated-target)*np.array([768,1024]),axis=1)/1280
                        value['carryin_'+name+'_distance']=float(dist[valid].mean()) if valid else None
                        value['carryin_'+name+'_penalized']=float(sum(min(float(dist[idx]),1.) if gs[idx]>=.3 else 1. for idx in eligible)/len(eligible)) if eligible else None
                        value['carryin_'+name+'_coverage']=len(valid)/len(eligible) if eligible else None
            except Exception as exc:
                value.update(metric_status='failed',error=str(exc))
                if args.stage=='id': value['id_penalized']=-1
            metrics.append(value); handle.write(json.dumps(value,allow_nan=False)+'\n'); handle.flush()
            if (i+1)%20==0: print(json.dumps({'stage':args.stage,'completed':i+1,'total':len(rows)}),flush=True)
    seconds=time.monotonic()-start
    result={'metadata':info,'expected_rows':len(rows),'completed_rows':len(metrics),'seconds':seconds,'gpu_hours':seconds/3600,
            'groups':summarize(metrics,fields),'interpretation':'Descriptive development diagnostics; no causal or final-test success verdict.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'READY').write_text('metrics computed; failures are included in summary\n')
    print(json.dumps({'stage':args.stage,'rows':len(metrics),'seconds':seconds}),flush=True)


if __name__=='__main__': main()
