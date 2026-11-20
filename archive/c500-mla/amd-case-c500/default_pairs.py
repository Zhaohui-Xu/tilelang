"""Time the unconfigured public factory, including all output allocations."""
import functools
import json
from pathlib import Path
import torch
from benchmark_mla_decode_maca import check_kernel,make_inputs,prepare_native_flashmla,reference,time_call
from example_mla_decode_maca import mla_decode_maca,default_schedule
from upstream_example import flashattn

torch.set_float32_matmul_precision('highest')
path=Path('defaults.json')
rows=json.loads(path.read_text()) if path.exists() else {}
def save():
    p=path.with_suffix('.tmp');p.write_text(json.dumps(rows,indent=2,allow_nan=False));p.replace(path)
def run(key,v,large):
    if rows.get(key,{}).get('complete'):return
    b,h,n=map(int,key.split('x'))
    bh,s=default_schedule(b,h,n)
    dbh,dbn,ds=map(int,v['baseline_config'].split(',')) if large else (16,32,v['baseline_splits'])
    cfg=mla_decode_maca(b,n,h)
    def cand(*xs):
        out=torch.empty_like(xs[0]);cfg(*xs,out);return out
    dense=flashattn(b,h,1,n,512,64,dbn,dbh,ds,1/24.)
    row=rows[key]={'default':[bh,s],'dense_config':[dbh,dbn,ds],'max_abs':{}}
    if not large:
        cases=[('random',i) for i in range(8)]+[(x,0) for x in ('early-peak','late-peak','large-v')]
        for name,f in [('default',cand),('dense',dense)]:row['max_abs'][name]=check_kernel(name,f,b,n,cases,'cuda',False,h)
        xs=make_inputs(b,n,367,'random','cuda',h)
    else:
        gen=torch.Generator(device='cuda').manual_seed(367)
        xs=tuple(torch.randn(sh,device='cuda',dtype=torch.float16,generator=gen).contiguous() for sh in ((b,h,512),(b,h,64),(b,n,1,512),(b,n,1,64)))
    calls={'default':functools.partial(cand,*xs),'dense':functools.partial(dense,*xs)}
    if large:calls['paged_native']=prepare_native_flashmla(*xs)
    ref=reference(*xs)
    for name,f in calls.items():
        out=f().float();torch.testing.assert_close(out,ref,atol=.002,rtol=.002)
        row['max_abs'][name+'_timed']=(out-ref).abs().max().item()
    for clock in ('event','wall'):
        samples=[]
        for i in range(6):
            keys=list(calls);off=i%len(keys);keys=keys[off:]+keys[:off]
            if len(keys)==3 and i%2:keys.reverse()
            samples.append({k:time_call(calls[k],5 if large else 200,30 if large else 2000,clock) for k in keys})
        row[clock]=samples;save()
    if large:
        for profile in ('first_peak','last_peak','large_v'):
            q,qp,kv,kp=xs;q.zero_();qp.zero_()
            if profile=='large_v':kv[...,::2]=2048;kv[...,1::2]=-2048
            else:q[...,0]=32;kv[...,0]=0;kv[:,0 if profile=='first_peak' else -1,:,0]=64
            out=cand(*xs).float();ref=reference(*xs)
            torch.testing.assert_close(out,ref,rtol=.002,atol=.002)
            row['max_abs'][profile]=(out-ref).abs().max().item()
    row['complete']=True;save()
for key,v in json.loads(Path('confirm.json').read_text()).items():run(key,v,False)
for key,v in json.loads(Path('release.json').read_text()).items():
    if 'baseline_config' in v:run(key,v,True)

# Replay the finalized policy on held-out sizes, including ragged tile tails.
edge=Path('default_edges.json')
edge_rows=json.loads(edge.read_text()) if edge.exists() else {}
for b,h,n in [(1,16,1),(1,16,31),(1,16,32),(1,16,33),(1,16,2047),(1,16,2049),
              (1,16,24577),(1,16,49151),(3,16,6145),(3,16,24575),(8,32,10001),
              (2,128,8193),(64,128,1025)]:
    key=f'{b}x{h}x{n}'
    if key in edge_rows:continue
    cfg=mla_decode_maca(b,n,h)
    def call(*xs):
        out=torch.empty_like(xs[0]);cfg(*xs,out);return out
    cases=[('random',i) for i in range(3)]+[(x,0) for x in ('early-peak','late-peak','large-v')]
    err=check_kernel('default',call,b,n,cases,'cuda',False,h)
    edge_rows[key]={'default':default_schedule(b,h,n),'passed':len(cases),'max_abs':err}
    edge.write_text(json.dumps(edge_rows,indent=2,allow_nan=False))

from statistics import median
cross_path=Path('cross_defaults.json')
cross_rows=json.loads(cross_path.read_text()) if cross_path.exists() else {}
for key,previous in json.loads(Path('cross.json').read_text()).items():
    if key in cross_rows:continue
    b,h,n=map(int,key.split('x'))
    best=min(previous['samples'],key=lambda k:median(previous['samples'][k]))
    tile,splits=default_schedule(b,h,n)
    default=f'{tile}/{splits}'
    xs=make_inputs(b,n,929,'random','cuda',h);ref=reference(*xs)
    opts=sorted({default,best,f'{tile}/{max(1,splits//2)}',f'{tile}/{min(n//32,splits*2)}'})
    calls={};row={'default':default,'prior_winner':best,'samples':{},'max_abs':{}}
    for label in opts:
        bh,s=map(int,label.split('/'))
        kernel=mla_decode_maca(b,n,h) if label==default else mla_decode_maca(b,n,h,s,bh)
        def call(kernel=kernel):
            out=torch.empty_like(xs[0]);kernel(*xs,out);return out
        out=call().float()
        torch.testing.assert_close(out,ref,rtol=.002,atol=.002)
        row['max_abs'][label]=(out-ref).abs().max().item()
        row['samples'][label]=[]
        calls[label]=call
    for i in range(6):
        off=i%len(opts);order=opts[off:]+opts[:off]
        if i%2 and len(order)>2:order.reverse()
        for label in order:row['samples'][label].append(time_call(calls[label],10,50,'wall'))
    cross_rows[key]=row
    cross_path.write_text(json.dumps(cross_rows,indent=2,allow_nan=False))
