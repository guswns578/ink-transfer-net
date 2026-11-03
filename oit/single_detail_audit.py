"""One-crop capacity audit; matched BF16/FP32 trials, no production writes."""
import argparse,contextlib,json,math,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt,gaussian_filter
from detail_branch import DetailBranch,compose_detail
from eval_integrity import ROOT,OLD,OUT,load,predict,render,noise
from prepare import features,read_alpha,save_alpha,sha
from run_detail_branch import infer

DEST=ROOT/'outputs'/'OIT_single_detail_audit_20260911'

def main():
    global DEST
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path)
    ap.add_argument('--edge-weight',type=float,default=.2)
    ap.add_argument('--precision',choices=['both','bf16','fp32'],default='both')
    ap.add_argument('--target-mode',choices=['raw','localized_tip'],default='raw')
    args=ap.parse_args()
    if args.out:DEST=args.out.resolve()
    precisions=('bf16','fp32') if args.precision=='both' else (args.precision,)
    DEST.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.6)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cp=OUT/'fullres_integrity'/'best_train.pt';base=load(cp,'integrity')
    manifest=json.loads((OLD/'manifest.json').read_text('utf-8'))
    rec=next(r for r in manifest['records'] if r['id']=='R01_detail')
    with np.load(OLD/rec['data']) as z:d={k:z[k].copy() for k in z.files}
    b=predict(base,d['features'],d['support']);full_f=np.concatenate([d['features'],b[None]])
    l,t=768,704;sl=np.s_[t:t+256,l:l+256]
    f=np.pad(full_f,((0,0),(40,40),(40,40)),mode='edge')[:,t:t+336,l:l+336]
    x,y,bb,s=d['x'][sl],d['y'][sl],b[sl],d['support'][sl]
    if args.target_mode=='localized_tip':
        # Training target curation only: keep the ear and unrelated contours.
        # Fixed spatial annotation chosen from the displayed target, not inferred
        # segmentation. This mask is NEVER supplied at inference.
        mask=np.zeros_like(y);mask[48:124,0:232]=1
        mask=gaussian_filter(mask,3).astype('float32')
        save_alpha(y,DEST/'uncurated_gpt_target.png')
        save_alpha(1-mask,DEST/'training_annotation.png')
        y=mask*y+(1-mask)*bb
    save_alpha(y,DEST/'training_target.png')
    changed=np.abs(y-bb)>.12
    ink=distance_transform_edt(~((y>.1)|(bb>.1)))<=3
    w=((1+3*ink+12*changed)*s).astype('float32')
    ft=torch.from_numpy(f[None].copy()).cuda();yt=torch.from_numpy(y[None,None].copy()).cuda()
    st=torch.from_numpy(s[None,None].copy()).cuda();wt=torch.from_numpy(w[None,None].copy()).cuda()
    cfg=dict(steps=3000,seed=911,train_crop=[l,t,256,256],source='R01_detail',
             base_sha256=sha(cp),loss='weighted raw Charbonnier + edge_weight * gradient L1',edge_weight=args.edge_weight,
             target_mode=args.target_mode,target_annotation=([0,48,232,124] if args.target_mode=='localized_tip' else None),
             precision_trials=list(precisions),checkpoint_selection='fixed final; intermediate logged only',
             caveat='Identical initialization/input/loss/steps/LR; only training precision differs between these two runs. One memorized crop cannot establish generalization.',
             code_sha256=sha(Path(__file__)))
    (DEST/'config.json').write_text(json.dumps(cfg,indent=2),'utf-8')
    metrics=[];outputs={}
    for precision in precisions:
        torch.manual_seed(911);torch.cuda.manual_seed_all(911)
        model=DetailBranch().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.0001)
        folder=DEST/precision;folder.mkdir();hist=[];start=time.perf_counter()
        for step in range(1,3001):
            model.train();opt.zero_grad(set_to_none=True)
            ctx=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else contextlib.nullcontext()
            with ctx:
                r=model(ft)[...,40:-40,40:-40];raw,p=compose_detail(ft[...,40:-40,40:-40],r,st)
                loss=((torch.sqrt((raw-yt).float().square()+1e-6)-.001)*wt).sum()/wt.sum().clamp_min(1)
                for dim in (2,3):
                    ww=wt.narrow(dim,0,wt.shape[dim]-1)
                    loss=loss+args.edge_weight*((torch.diff(p,dim=dim)-torch.diff(yt,dim=dim)).abs()*ww).sum()/ww.sum().clamp_min(1)
            assert torch.isfinite(loss)
            loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(grad)
            opt.step()
            for g in opt.param_groups:g['lr']=.0005*(.1+.9*.5*(1+math.cos(math.pi*step/3000)))
            if step==1 or step%250==0:
                row=dict(precision=precision,step=step,loss=float(loss.detach()),seconds=time.perf_counter()-start)
                hist.append(row);print(json.dumps(row),flush=True)
            if step in (500,1500,3000):
                model.eval()
                with torch.inference_mode():
                    pp=compose_detail(ft[...,40:-40,40:-40],model(ft)[...,40:-40,40:-40],st)[1][0,0].cpu().numpy()
                save_alpha(pp,folder/f'step_{step}.png')
                render([x,bb,pp,y],['원본','선 보존','한 영역 학습 '+precision,'GPT 목표 후보'],folder/f'00_step_{step}.png','학습 영역 재현 검사 / '+str(step)+' step / 1:1')
                metrics.append(dict(precision=precision,step=step,mae=float(np.abs(pp-y).mean()),changed_mae=float(np.abs(pp-y)[changed].mean()),clean_mae=float(np.abs(bb-y).mean())))
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},config=cfg,precision=precision),folder/'memorized_only.pt')
        (folder/'history.json').write_text(json.dumps(hist,indent=2),'utf-8');outputs[precision]=pp
        probes=[]
        # Preselected diagnostic crops, never used for training/checkpoint selection.
        for name in ('R09','R11','R14'):
            rec=next(r for r in manifest['qualitative'] if r['id']==name)
            xx=read_alpha(rec['source']);ff,ss=features(xx);bbfull=predict(base,ff,ss)
            ppfull=infer(model,np.concatenate([ff,bbfull[None]]),ss)
            for k,a in [('original',xx),('clean',bbfull),('output',ppfull)]:save_alpha(a,folder/'unseen'/f'{name}_{k}.png')
            tt=min(256,xx.shape[0]-256);ll=min(256,xx.shape[1]-256);crop=np.s_[tt:tt+256,ll:ll+256]
            render([a[crop] for a in (xx,bbfull,ppfull)],['원본','선 보존','한 영역 학습 '+precision],folder/'unseen'/f'00_{name}.png',name+' / 미학습 페이지 / 1:1')
            probes.append(dict(id=name,clean=noise(xx,bbfull),output=noise(xx,ppfull)))
        (folder/'unseen.json').write_text(json.dumps(probes,indent=2),'utf-8')
    render([x]+[outputs[k] for k in precisions]+[y],['원본']+[k+' 학습' for k in precisions]+['GPT 목표 후보'],DEST/'00_학습결과.png','한 영역 학습 / 3000 step / 학습 재현일 뿐 일반화 아님')
    (DEST/'metrics.json').write_text(json.dumps(metrics,indent=2),'utf-8')
    assert sha(cp)==cfg['base_sha256']
    print(json.dumps(metrics),flush=True);print('DONE '+str(DEST),flush=True)

if __name__=='__main__':main()
