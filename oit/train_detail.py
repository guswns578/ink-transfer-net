"""Localized pseudo-target supervision; input-only inference, native pixels."""
import json, math, sys, time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from fullres_model import FullResReinker
from model import compose
from prepare import features, read_alpha, save_alpha, sha
from train import load_samples
from eval_integrity import load, predict, render, noise, OLD, OUT, ROOT
from evaluate import target_score

DEST=ROOT/'outputs'/'OIT_detail_transfer_20260908'
# Training annotations only, never supplied at inference. Broad regions rather
# than target-derived per-pixel masks. Rectangles are x0,y0,x1,y1 native pixels.
REGIONS={'R01_detail':[(704,640,1024,1024),(256,800,704,1024)],
         'R03_detail':[(576,704,1024,1024),(128,64,768,384)]}

def main():
    DEST.mkdir(exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(908); torch.cuda.manual_seed_all(908)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.60)
    cp=OUT/'fullres_integrity'/'best_train.pt'
    base=load(cp,'integrity'); model=load(cp,'integrity').train()
    manifest=json.loads((OLD/'manifest.json').read_text('utf-8'))
    data=load_samples(OLD,manifest['train'],manifest)
    for name,d in zip(manifest['train'],data):
        mask=np.zeros_like(d['x'])
        for l,t,r,b in REGIONS[name]:mask[t:b,l:r]=1
        d['detail']=gaussian_filter(mask,8).astype('float32')
        d['base']=predict(base,d['features'],d['support'])
        save_alpha(1-d['detail'],DEST/f'{name}_training_roi.png')
    cfg=dict(model='FullResReinker',width=48,steps=1600,initial_checkpoint=str(cp),
             initial_sha256=sha(cp),train=manifest['train'],regions=REGIONS,
             target_provenance='Two provisional GPT pseudo-pairs; not artist-certified',
             inference='Input only; no ROI, target image, GPT or postfilter',
             loss='ROI target Charbonnier + 0.3 target gradients; outside ROI 2x clean baseline Charbonnier + 0.5 baseline gradients',
             checkpoint_selection='fixed final step, no validation selection',
             code_sha256=sha(Path(__file__)))
    (DEST/'config.json').write_text(json.dumps(cfg,indent=2),'utf-8')
    rng=np.random.default_rng(908);opt=torch.optim.AdamW(model.parameters(),lr=.00015)
    history=[];started=time.perf_counter()
    for step in range(1,1601):
        arrays=[]
        for j in range(2):
            d=data[int(rng.integers(2))];h,w=d['x'].shape
            for attempt in range(20):
                t=int(rng.integers(h-256+1));l=int(rng.integers(w-256+1))
                sl=np.s_[t:t+256,l:l+256]
                if d['detail'][sl].mean()>.25 or j==1:break
            aa=[d['features'][:,sl[0],sl[1]]]+[d[k][sl][None] for k in ('y','support','weights','detail','base')]
            if rng.random()<.5:aa=[a[...,::-1] for a in aa]
            arrays.append(aa)
        f,y,s,w,m,b=[torch.from_numpy(np.stack([np.ascontiguousarray(a[k]) for a in arrays])).cuda() for k in range(6)]
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            raw,p=compose(f,model(f),s)
            target=m*y+(1-m)*b
            weight=w*(m+2*(1-m))
            loss=((torch.sqrt((raw-target).float().square()+1e-6)-.001)*weight).sum()/weight.sum().clamp_min(1)
            edge=0
            for dim in (2,3):
                dp=torch.diff(p,dim=dim);dy=torch.diff(y,dim=dim);db=torch.diff(b,dim=dim)
                mm=m.narrow(dim,0,m.shape[dim]-1);ww=w.narrow(dim,0,w.shape[dim]-1)
                edge+=((.3*mm*(dp-dy).abs()+.5*(1-mm)*(dp-db).abs())*ww).sum()/ww.sum().clamp_min(1)
            loss=loss+edge
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        for g in opt.param_groups:g['lr']=.00015*(.15+.85*.5*(1+math.cos(math.pi*step/1600)))
        if step%200==0:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-started)
            history.append(row);print(json.dumps(row),flush=True)
    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,step=1600),DEST/'detail_experimental.pt')
    (DEST/'history.json').write_text(json.dumps(history,indent=2),'utf-8')
    model.eval();paired=[];unseen=[]
    for rec in manifest['records']:
        if rec['id'] in manifest['quarantined']:continue
        with np.load(OLD/rec['data']) as d:x,y,f,s=d['x'],d['y'],d['features'],d['support']
        b=predict(base,f,s);p=predict(model,f,s)
        name=rec['id'];folder=DEST/'paired';folder.mkdir(exist_ok=True)
        for key,a in [('original',x),('clean',b),('detail',p),('target',y)]:save_alpha(a,folder/f'{name}_{key}.png')
        paired.append(dict(id=name,split='train' if name in manifest['train'] else 'validation',clean=target_score(x,y,b),detail=target_score(x,y,p)))
        l,t=(768,704) if name=='R01_detail' else (704,768) if name=='R03_detail' else (256,256)
        sl=np.s_[t:t+256,l:l+256]
        render([a[sl] for a in (x,b,p,y)],['원본','선 보존 모델','디테일 추가 실험','GPT 목표 후보'],folder/f'00_{name}.png',name+' / 학습 여부는 report.json 참조 / 1:1')
    for rec in manifest['qualitative']:
        x=read_alpha(rec['source']);f,s=features(x);b=predict(base,f,s);p=predict(model,f,s)
        folder=DEST/'unseen';folder.mkdir(exist_ok=True);name=rec['id']
        for key,a in [('original',x),('clean',b),('detail',p)]:save_alpha(a,folder/f'{name}_{key}.png')
        t=min(256,x.shape[0]-256);l=min(256,x.shape[1]-256);sl=np.s_[t:t+256,l:l+256]
        render([a[sl] for a in (x,b,p)],['원본','선 보존 모델','디테일 추가 실험'],folder/f'00_{name}.png',name+' / 미학습 페이지 / 1:1')
        unseen.append(dict(id=name,page=rec['page'],shape=list(x.shape),clean=noise(x,b),detail=noise(x,p),change_from_clean=float(np.abs(p-b).mean())))
    report=dict(paired=paired,unseen=unseen,note='Training ROI annotations are never used at inference. Two pairs cannot establish general style transfer.')
    (DEST/'report.json').write_text(json.dumps(report,indent=2),'utf-8')
    print(json.dumps(dict(paired=paired,mean_far2={k:float(np.mean([r[k]['original_ink_farther_than_2px_pct'] for r in unseen])) for k in ('clean','detail')})),flush=True)
    print('DONE '+str(DEST),flush=True)

if __name__=='__main__':main()
