"""Read experiment outputs, check saved dimensions, quantify unreachable targets."""
import json,sys
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np
from PIL import Image
from single_detail_audit import DEST
from eval_integrity import OLD,OUT,render
from prepare import read_alpha

metrics=json.loads((DEST/'metrics.json').read_text('utf-8'))
with np.load(OLD/'data/R01_detail.npz') as d:
    x=d['x'][704:960,768:1024];y=d['y'][704:960,768:1024];s=d['support'][704:960,768:1024]
# Inference support is the only hard spatial restriction. This target-known
# bound is not an algorithm or learned result.
oracle=np.where(s>0,y,x)
bound=float(np.abs(oracle-y).mean())
sizes=[]
for precision in ('bf16','fp32'):
    for name in ('R09','R11','R14'):
        paths=[DEST/precision/'unseen'/f'{name}_{k}.png' for k in ('original','clean','output')]
        dims=[Image.open(p).size for p in paths];assert len(set(dims))==1
        sizes.append(dict(precision=precision,id=name,size=list(dims[0])))
    arrays=[read_alpha(DEST/precision/f'step_{step}.png') for step in (500,1500,3000)]
    render(arrays+[y],['500 step','1500 step','3000 step','GPT 목표 후보'],DEST/f'01_{precision}_학습경과.png','동일 학습 영역 / 실제 로컬 출력 / 원본 픽셀 1:1')
audit=dict(support_only_oracle_mae=bound,note='Target-known representability lower bound, not a prediction.',saved_dimensions=sizes)
(DEST/'audit.json').write_text(json.dumps(audit,indent=2),'utf-8')
print(json.dumps(dict(metrics=metrics,audit=audit),indent=2))
