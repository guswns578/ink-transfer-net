"""Export labelled actual checkpoint outputs, size checks and honest diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, gaussian_filter
from model import Reinker, compose
from prepare import read_alpha, save_alpha, features, metrics, sha


def compare(arrays, labels, title, dest, size=512):
    font = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 19)
    small = ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', 16)
    sheet = Image.new('RGB', (len(arrays)*size, size+90), '#e9edf0')
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 6), title, font=font, fill='#172235')
    for i, (a,label) in enumerate(zip(arrays,labels)):
        im = Image.fromarray(np.uint8(np.rint((1-np.clip(a,0,1))*255))).convert('RGB')
        im.thumbnail((size-8,size-8),Image.Resampling.LANCZOS)
        sheet.paste(im,(i*size+4,78))
        draw.text((i*size+8,44),label,font=small,fill='#172235')
    dest.parent.mkdir(parents=True,exist_ok=True)
    sheet.save(dest)


@torch.inference_mode()
def predict(model, x, device='cuda'):
    f,s = features(x)
    tensor = torch.from_numpy(f[None].copy()).to(device)
    support = torch.from_numpy(s[None,None].copy()).to(device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        _,y = compose(tensor,model(tensor),support)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    return y[0,0].float().cpu().numpy(), elapsed


def structure(x,p):
    ink, out = x>.5,p>.5
    do,dp = distance_transform_edt(~ink),distance_transform_edt(~out)
    return dict(mae_from_source=float(np.abs(p-x).mean()),
                ink_pct_before=float(ink.mean()*100),ink_pct_after=float(out.mean()*100),
                original_ink_farther_than_2px_pct=float((dp[ink]>2).mean()*100) if ink.any() else 0,
                new_ink_farther_than_3px_pct=float((do[out]>3).mean()*100) if out.any() else 0,
                source_to_prediction_p95_px=float(np.percentile(dp[ink],95)) if ink.any() else 0,
                prediction_to_source_p95_px=float(np.percentile(do[out],95)) if out.any() else 0,
                note='Ink-support diagnostics, NOT verified centerline displacement, character identity, topology or style quality.')


def target_score(x,y,p):
    roi=distance_transform_edt(~((x>.1)|(y>.1)))<=3
    base=float(np.abs(x-y)[roi].mean())
    error=float(np.abs(p-y)[roi].mean())
    return dict(identity_roi_mae=base,prediction_roi_mae=error,
                roi_error_reduction_percent=100*(1-error/base),
                identity_full_mae=float(np.abs(x-y).mean()),prediction_full_mae=float(np.abs(p-y).mean()))


def simple_baseline(x,sigma,k):
    a=gaussian_filter(x,sigma) if sigma>0 else x
    return a/(a+k*(1-a)+1e-8)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--experiment',required=True)
    ap.add_argument('--checkpoint',required=True)
    args=ap.parse_args()
    root=Path(args.experiment).resolve()
    cp=Path(args.checkpoint).resolve()
    out=root/'actual_model_outputs'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    manifest=json.loads((root/'manifest.json').read_text('utf-8'))
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ckpt=torch.load(cp,map_location='cpu',weights_only=True)
    assert ckpt['config']['manifest_sha256']==sha(root/'manifest.json')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(.60)
    model=Reinker(ckpt['config']['width']).cuda().eval()
    model.load_state_dict(ckpt['model'])
    train_data=[]
    for r in manifest['records']:
        if r['id'] in manifest['train']:
            with np.load(root/r['data']) as d:
                train_data.append((d['x'],d['y']))
    # Tune this deliberately weak non-neural baseline on TRAIN ONLY. It diagnoses
    # whether the model has learned more than a common blur/tone operation.
    choices=[]
    for sigma in (0,.35,.6,.9):
        for k in (.7,1.,1.4,2.):
            error=np.mean([target_score(x,y,simple_baseline(x,sigma,k))['prediction_roi_mae'] for x,y in train_data])
            choices.append((float(error),sigma,k))
    _,sigma,k=min(choices)
    results=[]
    paired={r['id']:r for r in manifest['records']}
    for name,r in paired.items():
        with np.load(root/r['data']) as data:
            x,y=data['x'],data['y']
        p,elapsed=predict(model,x)
        split='TRAIN_SEEN' if name in manifest['train'] else ('VALIDATION_UNSEEN_PAGE' if name in manifest['validation'] else 'QUARANTINED_NOT_INDEPENDENT')
        folder=out/split
        save_alpha(x,folder/f'{name}_0_original.png')
        save_alpha(y,folder/f'{name}_1_GPT_pseudo_target.png')
        save_alpha(p,folder/f'{name}_2_LOCAL_MODEL_raw.png')
        b=simple_baseline(x,sigma,k)
        save_alpha(b,folder/f'{name}_3_simple_blur_tone_baseline.png')
        title={'TRAIN_SEEN':'학습에 사용한 이미지 — 일반화 성능 아님',
               'VALIDATION_UNSEEN_PAGE':'학습하지 않은 페이지 — 목표는 GPT 후보, 정답 확정 아님',
               'QUARANTINED_NOT_INDEPENDENT':'학습 제외 후보 — 다른 학습 컷과 같은 페이지'}[split]
        compare([x,y,p],['원본','GPT 목표 후보','실제 로컬 모델 출력 (보정 없음)'],title,
                folder/f'00_{name}_comparison.png')
        # Specific native 384px windows show line quality without whole-face downscaling.
        crop=(slice(256,640),slice(256,640))
        compare([x[crop],y[crop],p[crop]],['원본','GPT 목표 후보','실제 로컬 모델 출력'],title+' / 동일 영역 확대',
                folder/f'01_{name}_detail.png',size=512)
        results.append(dict(id=name,split=split,seconds=elapsed,shape=list(p.shape),size_preserved=p.shape==x.shape,
                            prediction_sha256=sha(folder/f'{name}_2_LOCAL_MODEL_raw.png'),
                            target_score=target_score(x,y,p),simple_baseline_score=target_score(x,y,b),
                            structure=structure(x,p)))
    for r in manifest['qualitative']:
        x=read_alpha(r['source'])
        p,elapsed=predict(model,x)
        folder=out/'UNSEEN_VISUAL_ONLY'
        save_alpha(x,folder/f'{r["id"]}_0_original.png')
        save_alpha(p,folder/f'{r["id"]}_1_LOCAL_MODEL_raw.png')
        compare([x,p],['원본','실제 로컬 모델 출력 (보정 없음)'],r['id']+' / 학습하지 않은 페이지 / '+r['category'],
                folder/f'00_{r["id"]}_comparison.png')
        # Historic LoRA and OIT used different processing/resolution. Display only;
        # no pixel-accuracy rankings across these nominally aligned old baselines.
        old_lora,old_oit=read_alpha(r['old_lora']),read_alpha(r['old_oit'])
        compare([x,old_lora,old_oit,p],['원본','과거 LoRA (조건 다름)','과거 LoRA→OIT V0','새 로컬 모델 (원본 직접 입력)'],
                r['id']+' / 기존 방식은 조건이 다른 참고 비교',folder/f'02_{r["id"]}_legacy_context.png',size=384)
        results.append(dict(id=r['id'],split='UNSEEN_VISUAL_ONLY',page=r['page'],category=r['category'],
                            seconds=elapsed,shape=list(p.shape),size_preserved=p.shape==x.shape,
                            prediction_sha256=sha(folder/f'{r["id"]}_1_LOCAL_MODEL_raw.png'),
                            target_score=None,structure=structure(x,p)))
    # Determinism and blank-background checks are executable tests, not promises.
    probe=train_data[0][0]
    a,_=predict(model,probe)
    b,_=predict(model,probe)
    blank,_=predict(model,np.zeros((257,389),np.float32))
    assert all(r['size_preserved'] for r in results)
    assert blank.shape==(257,389) and float(blank.max())==0
    report=dict(checkpoint=str(cp),checkpoint_sha256=sha(cp),checkpoint_step=ckpt['step'],
                parameter_count=ckpt['config']['parameter_count'],training_data=manifest['train'],
                actual_inference='local PyTorch only; no GPT/Forge calls',postprocess='none; float alpha quantized to 8-bit PNG',
                simple_baseline=dict(sigma=sigma,k=k,fitted_on='training pairs only'),
                repeat_max_absolute_difference=float(np.max(np.abs(a-b))),blank_background_exactly_white=True,
                results=results,limitations=['2 provisional training pairs from 2 source pages',
                'One paired validation target has geometric mismatch; target-match score is not authentic style/identity evaluation',
                'Not an artist-approved dataset; not a production model',
                '17 qualitative crops are not 17 independent pages',
                'No semantic text/bubble/face protection; border and distance band only'])
    (out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
