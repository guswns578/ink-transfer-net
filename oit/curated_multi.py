"""Expanded curated-target pilot: native context, negative controls, jittered crops."""
import argparse,json,math,sys,time,shutil
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from PIL import Image,ImageDraw
from scipy.ndimage import gaussian_filter,distance_transform_edt
from detail_branch import DetailBranch,compose_detail
from eval_integrity import ROOT,OLD,OUT,load,predict,render,noise
from prepare import features,read_alpha,save_alpha,sha
from run_detail_branch import WINDOWS,infer

def curated_mask(name,shape):
    im=Image.new('L',(shape[1],shape[0]));d=ImageDraw.Draw(im)
    if name=='R01_detail':
        d.rectangle((768,752,999,827),fill=255)
        # Interior of the neck shadow; keep the jaw and outer shadow contour.
        d.polygon([(330,953),(420,965),(600,966),(565,988),(422,1011),(330,1005)],fill=255)
    else:
        d.polygon([(677,1005),(727,970),(778,911),(807,846),(827,794),
                   (823,858),(803,920),(760,972),(699,1016)],fill=255)
    return gaussian_filter(np.asarray(im,dtype='float32')/255,3)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--steps',type=int,default=8000);ap.add_argument('--width',type=int,default=64)
    args=ap.parse_args();out=args.out.resolve();out.mkdir(exist_ok=False)
    torch.manual_seed(912);torch.cuda.manual_seed_all(912);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.6)
    manifest=json.loads((OLD/'manifest.json').read_text('utf-8'))
    cp=OUT/'fullres_integrity'/'best_train.pt';base=load(cp,'integrity')
    data={}
    for rec in manifest['records']:
        if rec['id'] not in manifest['train']:continue
        with np.load(OLD/rec['data']) as z:d={k:z[k].copy() for k in z.files}
        b=predict(base,d['features'],d['support']);m=curated_mask(rec['id'],b.shape)
        d['clean']=b;d['raw_target']=d['y'].copy();d['y']=m*d['y']+(1-m)*b
        d['input']=np.concatenate([d['features'],b[None]]);d['mask']=m
        data[rec['id']]=d
        for k,a in [('source',d['x']),('target',d['y']),('training_mask',1-m)]:save_alpha(a,out/'data'/f'{rec["id"]}_{k}.png')
    crops=[];pos=[];neg=[]
    for name,l,t,label,positive in WINDOWS:
        d=data[name];h,w=d['x'].shape;seen=set()
        for dy in (-16,0,16):
            for dx in (-16,0,16):
                tt=max(0,min(h-256,t+dy));ll=max(0,min(w-256,l+dx))
                if (ll,tt) in seen:continue
                seen.add((ll,tt));sl=np.s_[tt:tt+256,ll:ll+256]
                f=np.pad(d['input'],((0,0),(40,40),(40,40)),mode='edge')[:,tt:tt+336,ll:ll+336]
                y=d['y'][sl];b=d['clean'][sl];s=d['support'][sl];m=d['mask'][sl]
                changed=np.abs(y-b)>.12;ink=distance_transform_edt(~((y>.1)|(b>.1)))<=3
                weights=((1+3*ink+12*changed)*s).astype('float32')
                c=dict(f=f,y=y,s=s,w=weights,x=d['x'][sl],b=b,name=label,xy=[ll,tt],positive=positive)
                (pos if positive else neg).append(len(crops));crops.append(c)
    model=DetailBranch(args.width).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.0004,weight_decay=.0001)
    cfg=dict(width=args.width,steps=args.steps,seed=912,train=manifest['train'],windows=WINDOWS,crop_count=len(crops),
             base_checkpoint=str(cp),base_sha256=sha(cp),model='DetailBranch',
             loss='weighted Charbonnier + 0.2 gradient loss; paired positive and negative sampling',
             jitter='native shifts +/-16 pixels, horizontal flips; no rescaling',
             targets='Two provisional GPT targets curated with fixed training-only annotations',
             limitation='All shifted crops are from two pages; they are not independent new data.',
             code_sha256=sha(Path(__file__)),selection='fixed final step, no validation selection')
    (out/'config.json').write_text(json.dumps(cfg,indent=2),'utf-8')
    rng=np.random.default_rng(912);history=[];start=time.perf_counter()
    for step in range(1,args.steps+1):
        ids=[int(rng.choice(pos)),int(rng.choice(neg if step%2==0 else pos))]
        arrays=[]
        for i in ids:
            c=crops[i];aa=[c['f']]+[c[k][None] for k in ('y','s','w')]
            if rng.random()<.5:aa=[a[...,::-1] for a in aa]
            arrays.append(aa)
        f,y,s,w=[torch.from_numpy(np.stack([np.ascontiguousarray(a[k]) for a in arrays])).cuda() for k in range(4)]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            r=model(f)[...,40:-40,40:-40];raw,p=compose_detail(f[...,40:-40,40:-40],r,s)
            loss=((torch.sqrt((raw-y).float().square()+1e-6)-.001)*w).sum()/w.sum().clamp_min(1)
            for dim in (2,3):
                ww=w.narrow(dim,0,w.shape[dim]-1)
                loss=loss+.2*((torch.diff(p,dim=dim)-torch.diff(y,dim=dim)).abs()*ww).sum()/ww.sum().clamp_min(1)
        assert torch.isfinite(loss)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(grad)
        optimizer.step()
        for g in optimizer.param_groups:g['lr']=.0004*(.1+.9*.5*(1+math.cos(math.pi*step/args.steps)))
        if step==1 or step%250==0:
            row=dict(step=step,loss=float(loss.detach()),seconds=time.perf_counter()-start)
            print(json.dumps(row),flush=True);history.append(row)
        if step%2000==0 or step==args.steps:
            torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,step=step),out/f'branch_{step}.pt')
            model.eval()
            for name,l,t,label,positive in WINDOWS:
                d=data[name];sl=np.s_[t:t+256,l:l+256]
                ff=np.pad(d['input'],((0,0),(40,40),(40,40)),mode='edge')[:,t:t+336,l:l+336]
                ft=torch.from_numpy(ff[None].copy()).cuda();st=torch.from_numpy(d['support'][sl][None,None].copy()).cuda()
                with torch.inference_mode():pp=compose_detail(ft[...,40:-40,40:-40],model(ft)[...,40:-40,40:-40],st)[1][0,0].cpu().numpy()
                render([d['x'][sl],d['clean'][sl],pp,d['y'][sl]],['원본','선 보존','확장 학습','정리한 목표'],out/'seen'/f'{step}_{label}.png',str(step)+' step / 학습 영역 / 1:1')
            model.train()
    (out/'history.json').write_text(json.dumps(history,indent=2),'utf-8');model.eval()
    results=[]
    for rec in manifest['qualitative']:
        x=read_alpha(rec['source']);f,s=features(x);b=predict(base,f,s);p=infer(model,np.concatenate([f,b[None]]),s)
        name=rec['id']
        for k,a in [('original',x),('clean',b),('output',p)]:save_alpha(a,out/'unseen'/f'{name}_{k}.png')
        t=min(256,x.shape[0]-256);l=min(256,x.shape[1]-256);sl=np.s_[t:t+256,l:l+256]
        render([a[sl] for a in (x,b,p)],['원본','선 보존','확장 학습'],out/'unseen'/f'00_{name}.png',name+' / 미학습 개발 평가 / 1:1')
        results.append(dict(id=name,page=rec['page'],shape=list(x.shape),clean=noise(x,b),output=noise(x,p)))
    (out/'unseen.json').write_text(json.dumps(results,indent=2),'utf-8')
    assert sha(cp)==cfg['base_sha256']
    (out/'code_snapshot').mkdir()
    for filename in ('curated_multi.py','detail_branch.py'):shutil.copy2(Path(__file__).with_name(filename),out/'code_snapshot'/filename)
    print('DONE '+str(out),flush=True)

if __name__=='__main__':main()
