"""Check the integrated factory and screen head and split schedules on C500."""
import functools
import gc
import hashlib
import json
from pathlib import Path
from statistics import median
import torch
from flash_mla import flash_mla_with_kvcache,get_mla_metadata
from example_mla_decode_maca import mla_decode_maca,default_schedule
from benchmark_mla_decode_maca import reference,time_call,check_kernel
from upstream_example import flashattn
from measured_kernel import mla_decode_maca as measured16
from wide_candidate import packed_wide

torch.set_float32_matmul_precision('highest')
archive=Path('confirm.json')
confirmed=json.loads(archive.read_text())
amd=json.loads(Path('amd.json').read_text())
wide=json.loads(Path('wide.json').read_text())
path=Path('release.json')
rows=json.loads(path.read_text()) if path.exists() else {}
def save():
    p=path.with_suffix('.tmp');p.write_text(json.dumps(rows,indent=2,allow_nan=False));p.replace(path)
def wrap(kernel):
    def fn(*xs):
        out=torch.empty_like(xs[0]);kernel(*xs,out);return out
    return fn
def inputs(b,h,n,seed):
    gen=torch.Generator(device='cuda').manual_seed(seed)
    return tuple(torch.randn(shape,device='cuda',dtype=torch.float16,generator=gen).contiguous() for shape in
                 ((b,h,512),(b,h,64),(b,n,1,512),(b,n,1,64)))
def verify(fn,xs,ref):
    out=fn(*xs)
    assert out.shape==ref.shape and out.dtype==torch.float16
    torch.testing.assert_close(out.float(),ref,atol=.002,rtol=.002)
    return (out.float()-ref).abs().max().item()
def codehash(kernel):
    return hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()
def tests16():
    for key,v in confirmed.items():
        if rows.get(key,{}).get('complete'):continue
        b,h,n=map(int,key.split('x'));s=v['candidate_splits']
        current=mla_decode_maca(b,n,h,s,16);old=measured16(b,n,h,s)
        xs=inputs(b,h,n,719);ref=reference(*xs)
        a,bhash=codehash(current),codehash(old)
        bh,default_split=default_schedule(b,h,n)
        default=mla_decode_maca(b,n,h)
        row=rows[key]={'max_abs':verify(wrap(current),xs,ref),'default_max_abs':verify(wrap(default),xs,ref),
                       'default_splits':default_split,'device_identical':a==bhash,'candidate_splits':s,'code_sha256':a}
        if default_split!=s:
            funcs={'default':wrap(default),'selected':wrap(current)}
            row['default_vs_selected']=[{k:time_call(functools.partial(funcs[k],*xs),100,500,'wall')
                                         for k in (list(funcs) if i%2==0 else list(reversed(funcs)))} for i in range(6)]
        if a!=bhash:
            fns={'current':wrap(current),'measured':wrap(old)}
            for clock in ('event','wall'):
                row[clock]=[{k:time_call(functools.partial(fns[k],*xs),40,300,clock) for k in (list(fns) if i%2==0 else list(reversed(fns)))} for i in range(6)]
        row['complete']=True;save()
def run(key,prior):
    if rows.get(key,{}).get('complete'):return
    b,h,n=map(int,key.split('x'))
    previous=rows.get(key,{})
    excluded=previous.get('excluded',{})
    if previous.get('pending'): excluded[previous['pending']]='interrupted during compilation or validation; see job log'
    row=rows[key]={'screen_us':{},'max_abs':{},'excluded':excluded}
    xs=inputs(b,h,n,157);ref=reference(*xs)
    fns={}
    # Include the earlier 16-head winner and search all legal 32/64-head splits.
    choices=[(16,amd[key]['c500_splits'])]
    choices += [(bh,2**i) for bh in ((32,64,128) if Path("head128.ok").exists() else (32,64)) for i in range((n//32).bit_length()) if b*h*(2**i)*514*4 <= 3*1024**3]
    for bh,s in choices:
        label=f'{bh}/{s}'
        print(key,label,flush=True)
        if label in excluded:continue
        row['pending']=label;save()
        try:
            f=wrap(mla_decode_maca(b,n,h,s,bh))
            verify(f,xs,ref)
            fns[label]=f
        except Exception as exc:
            excluded[label]=str(exc)[-600:]
        row.pop('pending');save()
    for i in range(3):
        keys=list(fns);offset=i*len(keys)//3
        for label in keys[offset:]+keys[:offset]:
            row['screen_us'].setdefault(label,[]).append(time_call(functools.partial(fns[label],*xs),3,12,'event'))
    short=sorted(fns,key=lambda k:median(row['screen_us'][k]))[:3]
    fixed='/'.join(map(str,default_schedule(b,h,n)))
    if fixed not in short:short.append(fixed)
    xs=inputs(b,h,n,317)
    refine={k:[] for k in short}
    for order in (short,list(reversed(short)),short[1:]+short[:1]):
        for label in order:refine[label].append(time_call(functools.partial(fns[label],*xs),10,40,'event'))
    selected=min(short,key=lambda k:median(refine[k]))
    row.update(selected=selected,default=fixed,refine_us=refine)
    dbh,dbn,ds=map(int,prior['best_dense'].split(','))
    xs=inputs(b,h,n,613);ref=reference(*xs)
    paged_q=torch.cat(xs[:2],-1).contiguous().unsqueeze(1)
    cache=torch.cat(xs[2:],-1).contiguous().view(-1,64,1,576)
    table=torch.arange(b*n//64,device='cuda',dtype=torch.int32).view(b,-1)
    lens=torch.full((b,),n,device='cuda',dtype=torch.int32)
    meta,split_meta=get_mla_metadata(lens,h,1)
    def paged(*unused):
        return flash_mla_with_kvcache(paged_q,cache,table,lens,512,meta,split_meta,causal=False)[0].squeeze(1)
    both={'selected':fns[selected],'dense':flashattn(b,h,1,n,512,64,dbn,dbh,ds,1/24.),'paged_native':paged}
    if selected!=fixed:both['default']=fns[fixed]
    row['baseline_config']=prior['best_dense']
    for name,fn in both.items():row['max_abs'][name]=verify(fn,xs,ref)
    for clock in ('event','wall'):
        samples=[]
        for i in range(6):
            names=list(both);offset=i%len(names);names=names[offset:]+names[:offset]
            if i%2:names.reverse()
            samples.append({k:time_call(functools.partial(both[k],*xs),5,30,clock) for k in names})
        row[clock]=samples
        save()
    for profile in ('early','late','large-v'):
        q,qp,kv,kp=xs;q.zero_();qp.zero_()
        if profile=='large-v':
            kv[...,::2]=2048;kv[...,1::2]=-2048
        else:
            kv[...,0]=0;q[...,0]=32;kv[:,0 if profile=='early' else -1,:,0]=64
        row['max_abs'][profile]=verify(fns[selected],xs,reference(*xs))
    row['complete']=True;save()
tests16()
for key,prior in wide.items():
    if 'best_dense' in prior:
        run(key,prior)
        gc.collect();torch.cuda.empty_cache()

# The default policy must also handle tiles and batch sizes outside the tuning grid.
extra=Path('unseen.json')
observations={}
for b,h,n in [(1,16,1),(1,16,31),(1,16,32),(1,16,33),(1,16,2047),(1,16,2049),
              (1,16,24577),(1,16,49151),(3,16,6145),(3,16,24575),(8,32,10001),
              (2,128,8193),(64,128,1025)]:
    bh,s=default_schedule(b,h,n)
    fn=wrap(mla_decode_maca(b,n,h))
    cases=[('random',i) for i in range(3)]+[(p,0) for p in ('early-peak','late-peak','large-v')]
    item={'head_tile':bh,'default_splits':s,'checks':len(cases),'max_abs':check_kernel('default',fn,b,n,cases,'cuda',False,h)}
    if n in (24577,49151,6145,24575,10001,8193,1025):
        xs=inputs(b,h,n,937);ref=reference(*xs);samples={}
        opts=sorted({max(1,s//2),s,min(2048,2*s)})
        for splits in opts:
            f=wrap(mla_decode_maca(b,n,h,splits,bh))
            verify(f,xs,ref)
            samples[splits]=[time_call(functools.partial(f,*xs),20,100,'wall') for _ in range(3)]
        item['neighbor_wall_samples']=samples
    observations[f'{b}x{h}x{n}']=item
    extra.write_text(json.dumps(observations,indent=2,allow_nan=False))
