"""Verify identical measured grid code and independently time changed default schedules."""
import hashlib,json
from pathlib import Path
import torch
from example_mla_decode_maca import default_schedule,mla_decode_maca
from previous_public import default_schedule as old_schedule,mla_decode_maca as old_compile
from benchmark_mla_decode_maca import make_inputs,reference,time_call
torch.set_float32_matmul_precision('highest')
path=Path('adopt.json')
data=json.loads(path.read_text()) if path.exists() else {}
def save():
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False));tmp.replace(path)
def hashes(k):
    return [hashlib.sha256(s.encode()).hexdigest() for s in (k.get_kernel_source(), k.adapter.get_host_source())]
def check(k,xs,ref):
    def f():
        out=torch.empty_like(xs[0]);k(*xs,out);return out
    out=f().float();torch.testing.assert_close(out,ref,atol=.002,rtol=.002)
    return f,(out-ref).abs().max().item()
for key,v in json.loads(Path('defaults.json').read_text()).items():
    if key in data:continue
    b,h,n=map(int,key.split('x'))
    assert list(default_schedule(b,h,n))==v['default']==list(old_schedule(b,h,n))
    k=mla_decode_maca(b,n,h);old=old_compile(b,n,h)
    newhash=hashes(k);oldhash=hashes(old)
    assert newhash==oldhash,(key,newhash,oldhash)
    data[key]={'identical':True,'device_and_host_sha256':newhash};save()
# Include all the correctness-only shapes too, so prior results transfer exactly.
for file in ('default_edges.json','regression.json'):
    for key,v in json.loads(Path(file).read_text()).items():
        if key in data:continue
        b,h,n=map(int,key.split('x'));t,s=default_schedule(b,h,n)
        if list((t,s))==v['default']:
            k=mla_decode_maca(b,n,h);old=old_compile(b,n,h);a=hashes(k);assert a==hashes(old)
            data[key]={'identical':True,'device_and_host_sha256':a};save()
        else:
            raise RuntimeError('Unexpected change on boundary workload '+key)
for key,v in json.loads(Path('transition.json').read_text()).items():
    if v['old_splits']==v['proposed_splits']:continue
    b,h,n=map(int,key.split('x'));tile,s=default_schedule(b,h,n);assert tile==64 and s==v['proposed_splits']
    xs=make_inputs(b,n,2221,'random','cuda',h);ref=reference(*xs)
    old=old_compile(b,n,h);k=mla_decode_maca(b,n,h)
    f,err=check(k,xs,ref);g,old_err=check(old,xs,ref);calls={'new':f,'old':g}
    row={'max_abs':err,'previous_max_abs':old_err,'splits':s}
    for clock in ('event','wall'):
        row[clock]=[{name:time_call(calls[name],20,100,clock) for name in (list(calls) if i%2==0 else list(reversed(calls)))} for i in range(6)]
    data[key]=row;save()
