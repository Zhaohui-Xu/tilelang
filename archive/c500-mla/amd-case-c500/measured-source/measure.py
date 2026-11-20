"""Independent split screening and held-out full-call timing on dense C500 MLA."""
import functools
import gc
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import torch
from flash_mla import flash_mla_with_kvcache, get_mla_metadata
from benchmark_mla_decode_maca import reference, time_call
from example_mla_decode_maca import mla_decode_maca
from upstream_example import flashattn

stage=sys.argv[1]
result=Path(stage+'.json')
rows=json.loads(result.read_text()) if result.exists() else {}
torch.set_float32_matmul_precision('highest')
def save():
    tmp=result.with_suffix('.tmp')
    tmp.write_text(json.dumps(rows,indent=2,allow_nan=False))
    tmp.replace(result)
def inputs(b,h,n,seed):
    rng=torch.Generator(device='cuda').manual_seed(seed)
    return tuple(torch.randn(size,device='cuda',dtype=torch.float16,generator=rng).contiguous() for size in ((b,h,512),(b,h,64),(b,n,1,512),(b,n,1,64)))
def expected(xs):
    # Keep the full reference output, compute its rows in bounded chunks.
    return torch.cat([reference(*(t[i:i+2] for t in xs)) for i in range(0,len(xs[0]),2)])
def run(b,h,n):
    key=f'{b}x{h}x{n}'
    if rows.get(key,{}).get('complete'): return
    print(key,flush=True)
    row=rows[key]={'screen_us':{},'max_abs':{},'timings':{}}
    ref_cfg = tuple(2**i for i in range((n//32).bit_length()) if b*h*(2**i)*512*2 <= 3*1024**3)
    configs={'c500':tuple(s for s in ref_cfg if b*h*s*514*4 <= 3*1024**3),'dense':tuple(s for s in ref_cfg if n%(32*s)==0)}
    def compile_one(name,splits):
        if name=='dense': return flashattn(b,h,1,n,512,64,32,16,splits,1/24.)
        k=mla_decode_maca(b,n,h,splits)
        def apply(*args):
            o=torch.empty_like(args[0])
            k(*args,o)
            return o
        return apply
    def validate(out,ref):
        assert out.shape==ref.shape and out.dtype==torch.float16
        torch.testing.assert_close(out.float(),ref,atol=.002,rtol=.002)
        return (out.float()-ref).abs().max().item()
    xs=inputs(b,h,n,17)
    target=expected(xs)
    winners={}
    for name,choices in configs.items():
        fns={}
        for s in choices:
            print(key,name,s,flush=True)
            fn=compile_one(name,s)
            validate(fn(*xs),target)
            fns[s]=fn
        sample={s:[] for s in choices}
        for i in range(3):
            offset=i*len(choices)//3
            for s in choices[offset:]+choices[:offset]:
                sample[s].append(time_call(functools.partial(fns[s],*xs),3,12,'event'))
        row['screen_us'][name]=sample
        best=min(sample,key=lambda s:statistics.median(sample[s]))
        winners[name]=fns[best]
        row[name+'_splits']=best
        save()
    del xs,target
    xs=inputs(b,h,n,23)
    target=expected(xs)
    calls={k:functools.partial(f,*xs) for k,f in winners.items()}
    if n%64==0:
        q=torch.cat(xs[:2],-1).contiguous().unsqueeze(1)
        kv=torch.cat(xs[2:],-1).contiguous().view(-1,64,1,576)
        table=torch.arange(b*(n//64),device='cuda',dtype=torch.int32).reshape(b,-1)
        lens=torch.full((b,),n,dtype=torch.int32,device='cuda')
        meta,splits=get_mla_metadata(lens,h,1)
        def paged():
            return flash_mla_with_kvcache(q,kv,table,lens,512,meta,splits,causal=False)[0].squeeze(1)
        try:
            row['flashmla_error_abs']=validate(paged(),target)
            calls['paged_native']=paged
            row['flashmla_version']=importlib.metadata.version('flash_mla')
        except (RuntimeError,AssertionError) as exc:
            row['flashmla_exclusion']=str(exc)[:500]
    for k,call in calls.items(): row['max_abs'][k]=validate(call(),target)
    # Compare independently of screening. Rotate order per batch to limit drift.
    for clock in ('event','wall'):
        samples=[]
        for rep in range(6):
            names=list(calls)
            names=names[rep%len(names):]+names[:rep%len(names)]
            if rep%2: names.reverse()
            samples.append({name:time_call(calls[name],4,20,clock) for name in names})
        row['timings'][clock]=samples
        save()
    row['complete']=True
    save()
if stage=='amd':
    grid=[(b,128,n) for b in (64,128) for n in (1024,2048,4096,8192,16384)]
elif stage=='small':
    grid=[(1,16,n) for n in (2048,4096,8192,16384,32768,65536)] + [(b,16,n) for b in (2,4) for n in (2048,8192,32768)]
else: raise ValueError(stage)
for shape in grid:
    run(*shape)
    gc.collect()
    torch.cuda.empty_cache()
