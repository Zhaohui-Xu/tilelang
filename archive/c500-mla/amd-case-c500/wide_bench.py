"""Screen head-tile width and split count on large-batch dense MLA; retain held-out comparisons."""
import functools
import gc
import json
from pathlib import Path
from statistics import median
import torch
from wide_candidate import wide, packed_wide
from benchmark_mla_decode_maca import reference, time_call
from example_mla_decode_maca import mla_decode_maca
from upstream_example import flashattn
import sys

torch.set_float32_matmul_precision('highest')
prior=json.loads(Path('amd.json').read_text())
path=Path('wide.json')
data=json.loads(path.read_text()) if path.exists() else {}
def save():
    p=path.with_suffix('.tmp')
    p.write_text(json.dumps(data,indent=2,allow_nan=False))
    p.replace(path)

def sample(b,n,seed):
    rng=torch.Generator(device='cuda').manual_seed(seed)
    return tuple(torch.randn(s,device='cuda',dtype=torch.float16,generator=rng).contiguous() for s in ((b,128,512),(b,128,64),(b,n,1,512),(b,n,1,64)))
def wrap(k):
    def run(*xs):
        out=torch.empty_like(xs[0]); k(*xs,out); return out
    return run

def build(b,n,cfg):
    bh,bn,s,t=cfg
    if bn==0:  # Packed K576, register Q; 128 threads per 16-head row group.
        return packed_wide(b,n,128,s,bh)
    return wide(b,128,n,bh,bn,s,t)

def run(b,n,choices):
    key=f'{b}x128x{n}'
    row=data.setdefault(key,{'screen':{},'excluded':{}})
    if row.get('complete'): return
    if 'pending' in row:
        cfg=row.pop('pending')
        row['excluded'][cfg]='previous process exited during this configuration; see run log'
        save()
    xs=sample(b,n,97); ref=reference(*xs)
    for cfg in choices:
        name=','.join(map(str,cfg))
        if name in row['screen'] or name in row['excluded']: continue
        row['pending']=name
        save()
        print('screen',key,name,flush=True)
        try:
            fn=wrap(build(b,n,cfg))
            torch.testing.assert_close(fn(*xs).float(),ref,atol=.002,rtol=.002)
            call=functools.partial(fn,*xs)
            row['screen'][name]=[time_call(call,3,12,'event') for _ in range(3)]
        except Exception as e:
            row['excluded'][name]=str(e)[-900:]
            print('excluded',type(e).__name__,flush=True)
        row.pop('pending')
        save()
    if not row['screen']:
        row['complete']=True; save();return
    best=min(row['screen'],key=lambda k: median(row['screen'][k]))
    row['best']=best
    cfg=tuple(map(int,best.split(',')))
    orig=prior[key]
    row['dense_screen']={}
    row['dense_excluded']={}
    shared_limit=getattr(torch.cuda.get_device_properties(0),'shared_memory_per_block',65536)
    row['shared_limit']=shared_limit
    for bh in (16,32,64):
        for bn in (16,32):
            for splits in sorted({1,4,8,orig['dense_splits']}):
                if n%(bn*splits): continue
                name=f'{bh},{bn},{splits}'
                # Q and KV share this memory budget during QK. P uses an
                # additional shared tile. Exclude oversize combinations.
                if (bh+bn)*576*2+bh*bn*2 > shared_limit:
                    row['dense_excluded'][name]='shared-memory estimate above device limit'
                    continue
                try:
                    k=flashattn(b,128,1,n,512,64,bn,bh,splits,1/24.)
                    torch.testing.assert_close(k(*xs).float(),ref,atol=.002,rtol=.002)
                    call=functools.partial(k,*xs)
                    row['dense_screen'][name]=[time_call(call,3,12,'event') for _ in range(3)]
                except Exception as exc:
                    row['dense_excluded'][name]=str(exc)[-900:]
    if not row['dense_screen']:
        raise RuntimeError('No dense baseline passed')
    best_dense=min(row['dense_screen'],key=lambda k:median(row['dense_screen'][k]))
    row['best_dense']=best_dense
    dbh,dbn,ds=map(int,best_dense.split(','))
    xs=sample(b,n,213); ref=reference(*xs)
    fns={'wide':wrap(build(b,n,cfg)), 'c500':wrap(mla_decode_maca(b,n,128,orig['c500_splits'])), 'dense':flashattn(b,128,1,n,512,64,dbn,dbh,ds,1/24.)}
    row['max_abs']={}
    for label,fn in fns.items():
        out=fn(*xs).float()
        torch.testing.assert_close(out,ref,atol=.002,rtol=.002)
        row['max_abs'][label]=(out-ref).abs().max().item()
    for clock in ('event','wall'):
        rows=[]
        for i in range(6):
            names=list(fns); names=names[i%3:]+names[:i%3]
            if i%2: names.reverse()
            rows.append({name:time_call(functools.partial(fns[name],*xs),4,20,clock) for name in names})
        row[clock]=rows
    # Exercise the retained wide schedule with extremal logits at both ends.
    row['stress_max_abs']={}
    for profile in ('early-peak','late-peak','large-v'):
        q,qp,kv,kp=xs
        q.zero_();qp.zero_()
        if profile=='large-v':
            kv[...,::2]=2048
            kv[...,1::2]=-2048
        else:
            kv[...,0]=0
            q[...,0]=32
            kv[:,0 if profile=='early-peak' else -1,:,0]=64
        ref=reference(*xs)
        out=fns['wide'](*xs).float()
        torch.testing.assert_close(out,ref,atol=.002,rtol=.002)
        row['stress_max_abs'][profile]=(out-ref).abs().max().item()
    row['complete']=True
    save()
pilots=[(64,1024),(128,8192)]
# Each config is (query heads per tile, key tokens per tile, KV splits, threads).
for b,n in pilots:
    choices=[(bh,bn,s,t) for bh,t in ((32,128),(32,256),(64,128),(64,256),(128,256)) for bn in (16,32) for s in (1,4,8)]
    choices += [(bh,0,s,128*(bh//16)) for bh in (32,64,128) if 128*(bh//16) <= getattr(torch.cuda.get_device_properties(0),'max_threads_per_block',512) for s in (1,4,8)]
    run(b,n,choices)
    torch.cuda.empty_cache()
valid=[k for k in data if data[k].get('best')]
wins=any(median(x['wide'] for x in data[k]['wall']) < .97*median(x['c500'] for x in data[k]['wall']) for k in valid)
if wins:
    configs={tuple(map(int,data[k]['best'].split(','))) for k in valid}
    for key in prior:
        b,_,n=map(int,key.split('x'))
        run(b,n,sorted(configs))
        gc.collect()
        torch.cuda.empty_cache()
