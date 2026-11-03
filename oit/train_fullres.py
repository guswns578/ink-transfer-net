"""Matched data/sampling/loss/budget comparison against the Sept 7 NAF pilot."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode=True
import numpy as np
import torch
from fullres_model import FullResReinker
from model import compose
from train import load_samples,batch,evaluate
from prepare import sha


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--experiment',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--steps',type=int,default=1600)
    ap.add_argument('--integrity',action='store_true',help='Add native edge/ink-contrast preservation losses')
    args=ap.parse_args()
    root,out=Path(args.experiment).resolve(),Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'manifest.json').read_text('utf-8'))
    torch.manual_seed(907); torch.cuda.manual_seed_all(907)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=True
    torch.cuda.set_per_process_memory_fraction(.55)
    model=FullResReinker().cuda()
    train=load_samples(root,manifest['train'],manifest)
    val=load_samples(root,manifest['validation'],manifest)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.0001)
    cfg=dict(model='FullResReinker',width=48,seed=907,steps=args.steps,patch=256,batch=2,
             train=manifest['train'],validation=manifest['validation'],
             parameters=sum(p.numel() for p in model.parameters()),
             loss='support-masked weighted Charbonnier'+(' + 0.35 source edge loss + 0.35 ink anchor' if args.integrity else ''),
             integrity=args.integrity,
             matched_to='Sept7 paired_native_v1: same data, sampler RNG, augmentation, LR, steps'+('; source-preservation loss added' if args.integrity else ', loss'),
             caveat='Architecture also removes global pooling and normalization; does not isolate downsampling alone.',
             source_manifest_sha256=sha(root/'manifest.json'),
             code_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    (out/'config.json').write_text(json.dumps(cfg,indent=2),'utf-8')
    print(json.dumps(cfg),flush=True)
    rng=np.random.default_rng(907)
    history=[];best=float('inf');t=time.perf_counter()
    for step in range(1,args.steps+1):
        f,y,s,w=batch(train,rng,256,2,'cuda')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            raw,p=compose(f,model(f),s)
            loss=((torch.sqrt((raw-y).float().square()+1e-6)-.001)*w).sum()/w.sum().clamp_min(1)
            if args.integrity:
                # Penalize damage to existing edge contrast, not blur the output.
                x=f[:,:1]
                gx=p[:,:,:,1:]-p[:,:,:,:-1]; tx=x[:,:,:,1:]-x[:,:,:,:-1]
                gy=p[:,:,1:,:]-p[:,:,:-1,:]; ty=x[:,:,1:,:]-x[:,:,:-1,:]
                wx=w[:,:,:,1:];wy=w[:,:,1:,:]
                edge=((gx-tx).abs()*wx).sum()/wx.sum().clamp_min(1)
                edge+=((gy-ty).abs()*wy).sum()/wy.sum().clamp_min(1)
                black=(x>.95).float()*s
                white=(x<.02).float()*s
                anchor=(torch.relu(.85-p)*black).sum()/black.sum().clamp_min(1)
                anchor+=.25*(torch.relu(p-.05)*white).sum()/white.sum().clamp_min(1)
                loss=loss+.35*edge+.35*anchor
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        for group in optimizer.param_groups:
            group['lr']=.0004*(.15+.85*.5*(1+math.cos(math.pi*step/args.steps)))
        if step==1 or step%100==0:
            print(json.dumps(dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-t,
                                  peak_gb=torch.cuda.max_memory_allocated()/1e9)),flush=True)
        if step%200==0 or step==args.steps:
            row=dict(step=step,seconds=time.perf_counter()-t,train=evaluate(model,train,'cuda'),validation=evaluate(model,val,'cuda'))
            history.append(row)
            if row['train']['weighted_mae']<best:
                best=row['train']['weighted_mae']
                torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,step=step),out/'best_train.pt')
            (out/'history.json').write_text(json.dumps(history,indent=2),'utf-8')
            print(json.dumps(row),flush=True)
    print('FULLRES_TRAIN_COMPLETE',flush=True)


if __name__=='__main__':
    main()
