"""Bounded capacity test followed by out-of-page inference, not production training."""
import json,math,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from detail_branch import DetailBranch,compose_detail
from eval_integrity import ROOT,OLD,OUT,load,predict,render,noise
from prepare import features,read_alpha,save_alpha,sha
from evaluate import target_score

DEST=ROOT/'outputs'/'OIT_detail_branch_20260911'
# Fixed diagnostic regions intentionally test memorization, not generalization.
WINDOWS=[('R01_detail',768,704,'female_hair_tip',True),
         ('R01_detail',256,768,'female_neck',True),
         ('R03_detail',704,768,'male_neck',True),
         ('R01_detail',192,320,'female_eye_control',False),
         ('R03_detail',384,512,'male_eye_control',False)]

@torch.inference_mode()
def infer(model,f,s):
    ff=torch.from_numpy(f[None].copy()).cuda();ss=torch.from_numpy(s[None,None].copy()).cuda()
    return compose_detail(ff,model(ff),ss)[1][0,0].cpu().numpy()

def main():
    DEST.mkdir(exist_ok=False)
    torch.manual_seed(911);torch.cuda.manual_seed_all(911);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.6)
    manifest=json.loads((OLD/'manifest.json').read_text('utf-8'))
    cp=OUT/'fullres_integrity'/'best_train.pt';base=load(cp,'integrity')
    for p in base.parameters():p.requires_grad_(False)
    model=DetailBranch().cuda();data={}
    for rec in manifest['records']:
        if rec['id'] not in manifest['train']:continue
        with np.load(OLD/rec['data']) as z:d={k:z[k].copy() for k in z.files}
        b=predict(base,d['features'],d['support']);d['clean']=b
        d['input']=np.concatenate([d['features'],b[None]])
        data[rec['id']]=d
    crops=[]
    for name,l,t,label,positive in WINDOWS:
        d=data[name];sl=np.s_[t:t+256,l:l+256]
        # Actual source context, edge padding only at original image boundaries.
        f=np.pad(d['input'],((0,0),(40,40),(40,40)),mode='edge')[:,t:t+336,l:l+336]
        y=d['y'][sl] if positive else d['clean'][sl]
        b=d['clean'][sl];s=d['support'][sl]
        changed=np.abs(y-b)>.12
        ink=distance_transform_edt(~((y>.1)|(b>.1)))<=3
        weight=(1+3*ink+12*changed)*s
        crops.append(dict(name=label,positive=positive,f=f,y=y,b=b,s=s,w=weight.astype('float32'),x=d['x'][sl]))
        render([d['x'][sl],b,y],['원본','고정 선 보존','학습 목표'],DEST/'training_regions'/f'{label}.png','학습 영역 / '+label)
    cfg=dict(model='DetailBranch',width=32,steps=2400,seed=911,base_checkpoint=str(cp),base_sha256=sha(cp),
             train_pages=['041_001','041_002'],windows=WINDOWS,context_radius=38,patch_with_context=336,
             loss='changed-pixel weighted raw Charbonnier + 0.2 target gradient L1; unedited eye controls',
             selection='fixed final 2400 steps; intermediate 1200 retained, no validation selection',
             caveat='Intentional tiny-region capacity test. GPT pseudo-targets not artist-certified. No inference masks or target access.',
             code_sha256={n:sha(Path(__file__).with_name(n)) for n in ['run_detail_branch.py','detail_branch.py']})
    (DEST/'config.json').write_text(json.dumps(cfg,indent=2),'utf-8')
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
    rng=np.random.default_rng(911);history=[];start=time.perf_counter()
    for step in range(1,2401):
        # One positive detail crop plus a second positive/control crop.
        ids=[int(rng.integers(3)),int(rng.integers(5))]
        f,y,b,s,w=[torch.from_numpy(np.stack([crops[i][k] for i in ids])).cuda() for k in ('f','y','b','s','w')]
        y=y[:,None];s=s[:,None];w=w[:,None]
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            residual=model(f)[...,40:-40,40:-40]
            raw,p=compose_detail(f[...,40:-40,40:-40],residual,s)
            loss=((torch.sqrt((raw-y).float().square()+1e-6)-.001)*w).sum()/w.sum().clamp_min(1)
            for dim in (2,3):
                ww=w.narrow(dim,0,w.shape[dim]-1)
                loss=loss+.2*((torch.diff(p,dim=dim)-torch.diff(y,dim=dim)).abs()*ww).sum()/ww.sum().clamp_min(1)
        assert torch.isfinite(loss)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        assert torch.isfinite(grad)
        if step==1:assert grad>0,'Dead initial gradient'
        opt.step()
        for g in opt.param_groups:g['lr']=.0005*(.1+.9*.5*(1+math.cos(math.pi*step/2400)))
        if step%200==0 or step==1:
            row=dict(step=step,loss=float(loss.detach()),grad=float(grad),seconds=time.perf_counter()-start)
            history.append(row);print(json.dumps(row),flush=True)
        if step in (1200,2400):
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,step=step),DEST/f'branch_{step}.pt')
    (DEST/'history.json').write_text(json.dumps(history,indent=2),'utf-8');model.eval()
    seen=[]
    for c in crops:
        fs=torch.from_numpy(c['f'][None].copy()).cuda()
        with torch.inference_mode():r=model(fs)[...,40:-40,40:-40]
        with torch.inference_mode():p=compose_detail(fs[...,40:-40,40:-40],r,torch.from_numpy(c['s'][None,None].copy()).cuda())[1][0,0].cpu().numpy()
        render([c['x'],c['b'],p,c['y']],['원본','선 보존 모델','별도 디테일 모델','학습 목표'],DEST/'seen'/f'00_{c["name"]}.png','학습 영역 재현 검사 (일반화 아님)')
        save_alpha(p,DEST/'seen'/f'{c["name"]}_output.png')
        seen.append(dict(name=c['name'],positive=c['positive'],clean_mae=float(np.abs(c['b']-c['y']).mean()),branch_mae=float(np.abs(p-c['y']).mean()),diagnostics=noise(c['x'],p)))
    results=[]
    for rec in manifest['qualitative']:
        x=read_alpha(rec['source']);f,s=features(x);b=predict(base,f,s)
        p=infer(model,np.concatenate([f,b[None]]),s);name=rec['id']
        for k,a in [('original',x),('clean',b),('branch',p)]:save_alpha(a,DEST/'unseen'/f'{name}_{k}.png')
        l=min(256,x.shape[1]-256);t=min(256,x.shape[0]-256);sl=np.s_[t:t+256,l:l+256]
        render([a[sl] for a in (x,b,p)],['원본','선 보존 모델','별도 디테일 모델'],DEST/'unseen'/f'00_{name}.png',name+' / 미학습 페이지 / 원본 픽셀 1:1')
        results.append(dict(id=name,page=rec['page'],shape=list(x.shape),clean=noise(x,b),branch=noise(x,p)))
    # Blank support and tiled finite-context checks with the trained branch.
    x=np.zeros((257,319),dtype='float32');f,s=features(x);ff=np.concatenate([f,x[None]])
    blank=infer(model,ff,s)
    c=data['R01_detail'];ff=c['input'][:,:512,:512];ss=c['support'][:512,:512]
    full=infer(model,ff,ss);tile=np.empty_like(full)
    for t in range(0,512,128):
        for l in range(0,512,128):
            a=max(0,t-40);b=max(0,l-40);z=min(512,t+168);q=min(512,l+168)
            p=infer(model,ff[:,a:z,b:q],ss[a:z,b:q]);tile[t:t+128,l:l+128]=p[t-a:t-a+128,l-b:l-b+128]
    checks=dict(blank_max=float(blank.max()),odd_shape=list(blank.shape),tile_max_error=float(np.abs(full-tile).max()))
    assert checks['blank_max']==0 and checks['tile_max_error']<1e-5
    report=dict(seen=seen,unseen=results,checks=checks,base_unchanged=sha(cp)==cfg['base_sha256'])
    (DEST/'report.json').write_text(json.dumps(report,indent=2),'utf-8')
    print(json.dumps(dict(seen=seen,checks=checks,unseen_far2={k:float(np.mean([r[k]['original_ink_farther_than_2px_pct'] for r in results])) for k in ('clean','branch')})),flush=True)
    print('DONE '+str(DEST),flush=True)

if __name__=='__main__':main()
