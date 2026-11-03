"""Can the actual network fit four fixed TRAINING patches?

This is an intentional overfit/plumbing test, never a generalization result.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
import torch
from model import Reinker, compose
from prepare import sha, save_alpha
from evaluate import compare, target_score
from train import load_samples


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--experiment',required=True)
    ap.add_argument('--steps',type=int,default=4000)
    args=ap.parse_args()
    root=Path(args.experiment).resolve()
    run=root/'runs'/'fixed_patch_memorization'
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    manifest=json.loads((root/'manifest.json').read_text('utf-8'))
    loaded=load_samples(root,manifest['train'],manifest)
    dmap=dict(zip(manifest['train'],loaded))
    # Coordinates are measured in ORIGINAL crop pixels, no resizing/warping here.
    windows=[('R01_detail',512,768,'female_neck'),('R01_detail',768,704,'female_hair_tip'),
             ('R03_detail',256,128,'male_hair'),('R03_detail',704,768,'male_neck')]
    batch=[]
    for name,left,top,label in windows:
        d=dmap[name]
        sl=np.s_[top:top+256,left:left+256]
        batch.append((d['features'][:,sl[0],sl[1]],d['y'][sl][None],d['support'][sl][None],d['weights'][sl][None]))
    tensors=[torch.from_numpy(np.stack([r[i] for r in batch])).cuda() for i in range(4)]
    f,y,s,w=tensors
    torch.manual_seed(1907)
    torch.cuda.manual_seed_all(1907)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.cuda.set_per_process_memory_fraction(.55)
    model=Reinker(24).cuda()
    optim=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=0)
    config=dict(width=24,steps=args.steps,seed=1907,mode='INTENTIONAL_TRAINING_PATCH_MEMORIZATION',
                source_pages=['041_001','041_002'],windows=windows,patch=256,augmentation='none',
                initialization='scratch',manifest_sha256=sha(root/'manifest.json'),
                loss='masked weighted Charbonnier; fixed 4 patches every update',
                warning='Not for production. Successful memorization would NOT demonstrate unseen-image style transfer.')
    (run/'config.json').write_text(json.dumps(config,indent=2),'utf-8')
    source=f[:,:1].detach().cpu().numpy()
    target=y.detach().cpu().numpy()
    baseline=float((((f[:,:1]-y).abs())*w).sum()/w.sum())
    history=[]
    started=time.perf_counter()
    for step in range(1,args.steps+1):
        optim.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            raw,p=compose(f,model(f),s)
            loss=((torch.sqrt((raw-y).float().square()+1e-6)-.001)*w).sum()/w.sum()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optim.step()
        if step in (2000,3200):
            for group in optim.param_groups:
                group['lr'] *= .4
        if step==1 or step%200==0:
            error=float(((p.detach()-y).abs()*w).sum()/w.sum())
            row=dict(step=step,weighted_mae=error,identity_weighted_mae=baseline,
                     reduction_percent=100*(1-error/baseline),elapsed_sec=time.perf_counter()-started)
            history.append(row)
            print(json.dumps(row),flush=True)
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        _,p=compose(f,model(f),s)
    predictions=p.float().cpu().numpy()
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=config,step=args.steps),run/'memorized_only.pt')
    results=[]
    for i,(_,_,_,label) in enumerate(windows):
        x,yy,pp=source[i,0],target[i,0],predictions[i,0]
        save_alpha(x,run/f'{label}_0_original.png')
        save_alpha(yy,run/f'{label}_1_GPT_target.png')
        save_alpha(pp,run/f'{label}_2_LOCAL_MODEL_seen_patch.png')
        compare([x,yy,pp],['원본','GPT 목표 후보','실제 로컬 모델: 이 영역을 학습함'],
                '암기 능력 진단 / '+label+' / 새로운 이미지 성능이 아님',run/f'00_{label}_comparison.png',size=384)
        results.append(dict(label=label,**target_score(x,yy,pp)))
    (run/'history.json').write_text(json.dumps(history,indent=2),'utf-8')
    (run/'results.json').write_text(json.dumps(dict(results=results,total_sec=time.perf_counter()-started),indent=2),'utf-8')
    print('MEMORIZATION_COMPLETE '+json.dumps(results),flush=True)


if __name__=='__main__':
    main()
