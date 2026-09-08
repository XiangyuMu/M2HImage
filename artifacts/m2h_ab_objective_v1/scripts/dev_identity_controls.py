"""Independent existing w600k_r50 development recognizer, never AdaFace."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['NO_ALBUMENTATIONS_UPDATE']='1'
import cv2
import insightface
import numpy as np
import onnxruntime as ort


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--controls',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',type=int,default=2)
    p.add_argument('--recognizer',type=Path,default=Path('/home/muxiangyu/.insightface/models/buffalo_l/w600k_r50.onnx'))
    args=p.parse_args()
    if not args.recognizer.is_file() or args.recognizer.name!='w600k_r50.onnx':
        raise ValueError('Expected frozen existing dev weights; do not substitute final evaluator')
    args.out.mkdir(parents=True,exist_ok=False)
    if 'CUDAExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('GPU ONNX runtime required')
    providers=[('CUDAExecutionProvider',{'device_id':args.device}),'CPUExecutionProvider']
    app=insightface.app.FaceAnalysis(name='antelopev2',root='/data/muxiangyu/modelLibrary/insightface',
                                    allowed_modules=['detection'],providers=providers)
    app.prepare(ctx_id=args.device,det_size=(640,640))
    rec=insightface.model_zoo.get_model(str(args.recognizer),providers=providers)
    rec.prepare(ctx_id=args.device)
    if 'CUDAExecutionProvider' not in rec.session.get_providers():
        raise RuntimeError('GPU recognizer fallback detected')
    audit=json.loads((args.probe/'audit.json').read_text())
    manifest={'role':'development only; not final AdaFace',
              'checkpoint':str(args.recognizer),'checkpoint_sha256':sha256(args.recognizer),
              'detector':'existing antelopev2 detector; recognition module excluded',
              'detector_threshold':.3,'alignment':'InsightFace norm_crop from detected keypoints',
              'failure_penalty_cosine':-1,'script_sha256':sha256(__file__)}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def embedding(path):
        bgr=cv2.imread(str(path))
        if bgr is None:
            raise ValueError('unreadable image')
        faces=app.get(bgr)
        if not faces:
            raise ValueError('no face')
        face=max(faces,key=lambda f:float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))
        if float(face.det_score)<.3:
            raise ValueError('low detection confidence')
        v=np.asarray(rec.get(bgr,face),dtype=np.float32).reshape(-1)
        if not np.isfinite(v).all() or np.linalg.norm(v)<1e-6:
            raise ValueError('invalid embedding')
        return v/np.linalg.norm(v)
    start=time.monotonic()
    rows=[]; ref_cache={}
    methods=('base','feather_e4_f8','feather_e8_f16','poisson_e4')
    with (args.out/'per_image.jsonl').open('x') as handle:
        for pair in audit['pairs']:
            mid,jid,seed=pair['mid'],pair['jid'],pair['seed']
            if jid not in ref_cache:
                source=next((args.root/'images/human'/f'{jid}{ext}' for ext in ('.jpg','.png','.jpeg') if (args.root/'images/human'/f'{jid}{ext}').is_file()),None)
                try: ref_cache[jid]=(embedding(source),None)
                except Exception as exc: ref_cache[jid]=(None,str(exc))
            ref,error=ref_cache[jid]
            name=f'{mid}__id{jid}__seed{seed}.png'
            for method in methods:
                path=(args.probe/'outputs'/name) if method=='base' else args.controls/method/name
                row={**pair,'method':method,'reference_eligible':ref is not None,'status':'ok'}
                if ref is None:
                    row.update(status='reference_failed',error=error,cosine=None,penalized_cosine=None)
                else:
                    try:
                        emb=embedding(path)
                        score=float(np.clip(ref@emb,-1,1))
                        row.update(cosine=score,penalized_cosine=score)
                    except Exception as exc:
                        row.update(status='output_failed',error=str(exc),cosine=None,penalized_cosine=-1)
                rows.append(row); handle.write(json.dumps(row)+'\n'); handle.flush()
    summary={'formal_result':False,'identity_model':manifest,'methods':{},'seconds':time.monotonic()-start}
    for method in methods:
        subset=[r for r in rows if r['method']==method]
        valid=[r['cosine'] for r in subset if r['cosine'] is not None]
        penalized=[r['penalized_cosine'] for r in subset if r['penalized_cosine'] is not None]
        summary['methods'][method]={'expected':len(subset),'reference_eligible':len(penalized),
            'output_faces_detected':len(valid),'valid_mean':float(np.mean(valid)) if valid else None,
            'failure_penalized_mean':float(np.mean(penalized)) if penalized else None}
    summary['gpu_hours']=summary['seconds']/3600
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
