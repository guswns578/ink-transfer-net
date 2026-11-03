"""Native-size and halo tests for a saved two-stage candidate."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from PIL import Image,ImageDraw
from fullres_model import FullResReinker
from detail_branch import DetailBranch
from infer_fullres import tiled as base_tiled
from fresh_panels import detail_tiled
from prepare import features,sha

ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True,type=Path);ap.add_argument('--out',required=True,type=Path);args=ap.parse_args()
assert not args.out.exists()
torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.6)
torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
ck=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
model=DetailBranch(ck['config']['width']).cuda().eval();model.load_state_dict(ck['model'])
bp=Path(ck['config']['base_checkpoint']);bc=torch.load(bp,map_location='cpu',weights_only=True)
base=FullResReinker(bc['config']['width']).cuda().eval();base.load_state_dict(bc['model'])
im=Image.new('L',(389,321));d=ImageDraw.Draw(im)
d.line((35,30,350,30),fill=255,width=1);d.line((40,55,40,285),fill=255,width=1)
d.line((70,70,320,270),fill=255,width=3);d.arc((120,40,340,250),0,300,fill=255,width=1)
x=np.asarray(im,dtype='float32')/255
def run(xx,core):
    bb=base_tiled(base,xx,core=core);ff,ss=features(xx)
    return detail_tiled(model,np.concatenate([ff,bb[None]]),ss,core=core)
p=run(x,128);q=run(x,256);blank=run(np.zeros_like(x),128)
far=distance_transform_edt(~(x>.1))>16
report=dict(checkpoint_sha256=sha(args.checkpoint),shape=list(p.shape),tile_max_difference=float(np.abs(p-q).max()),
            blank_max=float(blank.max()),outside_edit_band_max=float(p[far].max()),
            source_black_pixels=int((x>.99).sum()),source_black_pixels_below_half=int((p[x>.99]<.5).sum()),
            minimum_on_source_black=float(p[x>.99].min()),note='Synthetic diagnostics only, not identity/style/semantic validation.')
assert p.shape==x.shape and report['blank_max']==0 and report['outside_edit_band_max']==0
assert report['tile_max_difference']<1e-5
args.out.write_text(json.dumps(report,indent=2),'utf-8');print(json.dumps(report,indent=2))
