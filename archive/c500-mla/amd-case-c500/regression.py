"""Repeat the original 18-shape boundary contract on the final default kernel."""
from pathlib import Path
import json
import torch
from benchmark_mla_decode_maca import check_kernel
from example_mla_decode_maca import mla_decode_maca,default_schedule
torch.set_float32_matmul_precision('highest')
p=Path('regression.json')
rows=json.loads(p.read_text()) if p.exists() else {}
for b,n in [(1,n) for n in (1,31,32,33,2047,2048,2049,6144,8193,12288,65535,65536)]+[(b,n) for b in (2,4) for n in (33,2049,8192)]:
    key=f'{b}x16x{n}'
    if key in rows:continue
    k=mla_decode_maca(b,n)
    def f(*xs):
        out=torch.empty_like(xs[0]);k(*xs,out);return out
    cases=[('random',i) for i in range(8)]+[(s,0) for s in ('early-peak','late-peak','large-v')]
    rows[key]={'default':default_schedule(b,16,n),'checks':len(cases),'max_abs':check_kernel('final',f,b,n,cases,'cuda',False,16)}
    p.write_text(json.dumps(rows,indent=2,allow_nan=False))
