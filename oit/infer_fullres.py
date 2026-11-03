"""Offline native-size inference for FullResReinker. No image postprocessing."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from fullres_model import FullResReinker
from model import compose
from prepare import read_alpha,save_alpha,features,sha


@torch.inference_mode()
def tiled(model,x,device='cuda',core=512):
    f,s=features(x)
    h,w=x.shape
    result=np.empty_like(x)
    halo=32  # Larger than the architecture's 26px receptive-field radius.
    for top in range(0,h,core):
        for left in range(0,w,core):
            bottom,right=min(h,top+core),min(w,left+core)
            y0,y1=max(0,top-halo),min(h,bottom+halo)
            x0,x1=max(0,left-halo),min(w,right+halo)
            ft=torch.from_numpy(f[None,:,y0:y1,x0:x1].copy()).to(device)
            st=torch.from_numpy(s[None,None,y0:y1,x0:x1].copy()).to(device)
            _,p=compose(ft,model(ft),st)
            p=p[0,0].cpu().numpy()
            result[top:bottom,left:right]=p[top-y0:bottom-y0,left-x0:right-x0]
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--input',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=ap.parse_args()
    source,dest,cp=Path(args.input).resolve(),Path(args.output).resolve(),Path(args.checkpoint).resolve()
    if dest==source or dest.exists() or dest.with_suffix('.json').exists():raise FileExistsError(dest)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.device=='cuda':torch.cuda.set_per_process_memory_fraction(.60)
    ck=torch.load(cp,map_location='cpu',weights_only=True)
    assert ck['config']['model']=='FullResReinker'
    model=FullResReinker(ck['config']['width']).to(args.device).eval()
    model.load_state_dict(ck['model'])
    x=read_alpha(source);start=time.perf_counter()
    y=tiled(model,x,args.device)
    assert x.shape==y.shape
    save_alpha(y,dest)
    meta=dict(input=str(source),input_sha256=sha(source),checkpoint_sha256=sha(cp),
              model='FullResReinker',integrity_loss=ck['config'].get('integrity',False),
              shape=list(y.shape),seconds=time.perf_counter()-start,postprocess='none',
              source_resize='none',inference='local float32; 512 core + 32 halo; global input features',
              warning='Line-integrity research model, not a validated target style-transfer product')
    dest.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),'utf-8')
    print(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
