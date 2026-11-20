"""Run serial correctness and paired comparisons, checkpointing each shape."""
import functools
import json
import statistics
import sys
from pathlib import Path

import torch
import tilelang
import stable_mla
import upstream_example
from example_mla_decode_maca import mla_decode_maca
from benchmark_mla_decode_maca import check_kernel, make_inputs, reference, time_call

torch.set_float32_matmul_precision("highest")
device = torch.device("cuda")
cases = [("random", i) for i in range(8)] + [(p, 0) for p in ("early-peak","late-peak","large-v")]
lengths = [2048,4096,8192,16384,32768,65536]
stage = sys.argv[1]
file = Path(stage + ".json")
results = json.loads(file.read_text()) if file.exists() else {}
def save():
    temp=file.with_suffix(".tmp")
    temp.write_text(json.dumps(results,indent=2,allow_nan=False))
    temp.replace(file)
def candidate(batch,ctx):
    compiled=mla_decode_maca(batch,ctx)
    def run(*xs):
        out=torch.empty_like(xs[0])
        compiled(*xs,out)
        return out
    return run

def trial(batch,ctx):
    key=f"{batch}x{ctx}"
    if key in results and results[key].get("complete"): return
    print(stage,key,flush=True)
    shape=(batch,16,1,ctx,512,64)
    fn=candidate(batch,ctx)
    row=results.setdefault(key,{})
    def check(name,f):
        row.setdefault("max_abs",{})[name]=check_kernel(name,f,batch,ctx,cases,device,False)
        save()
    check("candidate",fn)
    if stage!="boundary":
        xs=make_inputs(batch,ctx,23,"random",device)
        if stage=="archive":
            compiled=stable_mla._compile(*shape)
            def other(*inputs):
                out=torch.empty_like(inputs[0])
                compiled(*inputs,out)
                return out
        else:
            def build(s): return upstream_example.flashattn(*shape,32,16,s,1/24.)
            if ctx < 16384:
                winner={2048:64,4096:64,8192:128}[ctx]
            else:
                trial_xs=make_inputs(batch,ctx,17,"random",device)
                expected=reference(*trial_xs)
                kernels={}
                for power in range((ctx//32).bit_length()):
                    splits=2**power
                    k=build(splits)
                    torch.testing.assert_close(k(*trial_xs).float(),expected,rtol=.002,atol=.002)
                    kernels[splits]=functools.partial(k,*trial_xs)
                screen={s:[] for s in kernels}
                keys=list(kernels)
                for i in range(3):
                    shift=i*len(keys)//3
                    for s in keys[shift:]+keys[:shift]: screen[s].append(time_call(kernels[s],20,60,"event"))
                row["screen_us"]=screen
                winner=min(screen,key=lambda s: statistics.median(screen[s]))
            row["splits"]=winner
            other=build(winner)
        check(stage,other)
        calls={"candidate":functools.partial(fn,*xs),stage:functools.partial(other,*xs)}
        for clock in ("event","wall"):
            rows=[]
            for i in range(6):
                order=(stage,"candidate") if i%2==0 else ("candidate",stage)
                rows.append({name:time_call(calls[name],200,2000,clock) for name in order})
            row[clock]=rows
            save()
    row["complete"]=True
    save()

matrix=[(1,n) for n in (1,31,32,33,2047,2048,2049,6144,8193,12288,65535,65536)]+[(b,n) for b in (2,4) for n in (33,2049,8192)] if stage=="boundary" else [(1,n) for n in lengths]
for batch,ctx in matrix: trial(batch,ctx)
