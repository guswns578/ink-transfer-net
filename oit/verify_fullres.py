import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
from fullres_model import FullResReinker
from model import compose
from prepare import features,sha
from infer_fullres import tiled


def main():
    root=Path(__file__).resolve().parents[1]/'outputs'/'OIT_line_integrity_20260908'
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ck=torch.load(root/'fullres_integrity'/'best_train.pt',map_location='cpu',weights_only=True)
    m=FullResReinker().cuda().eval();m.load_state_dict(ck['model'])
    x=np.zeros((321,389),np.float32)
    x[64,30:340]=1
    x[128:131,30:340]=1
    for t in range(40,260):x[t,t]=1
    f,s=features(x)
    with torch.inference_mode():
        _,p=compose(torch.from_numpy(f[None]).cuda(),m(torch.from_numpy(f[None]).cuda()),torch.from_numpy(s[None,None]).cuda())
        p=p[0,0].cpu().numpy()
    q=tiled(m,x,core=128)
    assert p.shape==x.shape and q.shape==x.shape
    tile_error=float(np.max(np.abs(p-q)))
    assert tile_error<2e-5,tile_error
    minimum=float(p[x>.95].min())
    assert minimum>.5,minimum
    blank=tiled(m,np.zeros((257,389),np.float32),core=128)
    assert float(blank.max())==0
    # Nonzero parameter gradients at identity initialization.
    fresh=FullResReinker(width=8)
    ft=torch.rand(1,3,33,41)
    r=fresh(ft);assert r.shape==(1,1,33,41)
    assert float(r.detach().abs().max())==0
    (r-.1).square().mean().backward()
    assert float(fresh.head.weight.grad.abs().sum())>0
    report=dict(odd_size_preserved=True,blank_exactly_white=True,
                tested_black_line_pixels=int((x>.95).sum()),min_output_on_black_test_lines=minimum,
                tiled_vs_whole_max_error=tile_error,initial_gradient_alive=True,
                checkpoint_sha256=sha(root/'fullres_integrity'/'best_train.pt'),
                note='Finite synthetic test, not a guarantee for all real lineart.')
    (root/'verification.json').write_text(json.dumps(report,indent=2),'utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
