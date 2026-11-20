"""Test one low-workload split policy in the transition to a 64-head tile."""
from pathlib import Path
from statistics import median
import functools,json
import torch
from example_mla_decode_maca import mla_decode_maca,default_schedule
from benchmark_mla_decode_maca import make_inputs,reference,time_call
torch.set_float32_matmul_precision('highest')
p=Path('transition.json')
rows=json.loads(p.read_text()) if p.exists() else {}
# First two shapes motivated the hypothesis; later shapes probe adjacent batches and lengths.
for b,h,n in ((16,128,2048),(4,128,8192),(16,64,2048),(8,128,4096),(8,64,4096),(12,128,2048),
              (6,128,4096),(24,64,2048),(15,128,2113),(5,128,6145),
              (17,128,2048),(20,128,2048),(9,128,4096),(10,128,4096),(5,128,8191),
              (16,128,2305),(8,128,4500),(12,128,4096)):
    key=f'{b}x{h}x{n}'
    if key in rows:continue
    t,s=default_schedule(b,h,n)
    assert t==64
    groups=b*(h//64)
    low=min(s,1<<(((64+groups-1)//groups-1).bit_length()))
    xs=make_inputs(b,n,1321,'random','cuda',h);ref=reference(*xs)
    calls={}
    for cfg in sorted({low,s}):
        k=mla_decode_maca(b,n,h,cfg,t)
        def f(k=k):
            out=torch.empty_like(xs[0]);k(*xs,out);return out
        torch.testing.assert_close(f().float(),ref,atol=.002,rtol=.002)
        calls[cfg]=f
    samples={cfg:[] for cfg in calls}
    for rep in range(6):
        for cfg in (list(calls) if rep%2==0 else list(reversed(calls))):
            samples[cfg].append(time_call(calls[cfg],20,100,'wall'))
    rows[key]={'old_splits':s,'proposed_splits':low,'samples':samples,'wall_ratio':median(samples[low])/median(samples[s])}
    p.write_text(json.dumps(rows,indent=2,allow_nan=False))
