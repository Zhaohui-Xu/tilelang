"""Confirm the selected small-batch splits with longer held-out timings."""
import functools
import json
from statistics import median
from pathlib import Path
import torch
from benchmark_mla_decode_maca import make_inputs,check_kernel,time_call
from upstream_example import flashattn
from example_mla_decode_maca import mla_decode_maca
torch.set_float32_matmul_precision('highest')
prior=json.loads(Path('small.json').read_text())
path=Path('confirm.json')
rows=json.loads(path.read_text()) if path.exists() else {}
def save():
    p=path.with_suffix('.tmp');p.write_text(json.dumps(rows,indent=2,allow_nan=False));p.replace(path)
for key,trial in prior.items():
    if rows.get(key,{}).get('complete'):continue
    b,h,n=map(int,key.split('x'))
    splits=trial['c500_splits']
    print(key,splits,flush=True)
    def wrapped(s):
        k=mla_decode_maca(b,n,h,s)
        def fn(*xs):
            out=torch.empty_like(xs[0]);k(*xs,out);return out
        return fn
    def dense(s): return flashattn(b,h,1,n,512,64,32,16,s,1/24.)
    screen_x=make_inputs(b,n,73,'random','cuda',h)
    refine={}
    selected={}
    for label,build in [('c500',wrapped),('dense',dense)]:
        samples=trial['screen_us'][label]
        choices=[int(s) for s in sorted(samples,key=lambda s:median(samples[s]))[:3]]
        if label=='c500' and 64 not in choices: choices.append(64)
        calls={s:functools.partial(build(s),*screen_x) for s in choices}
        times={s:[] for s in choices}
        for order in (choices,list(reversed(choices)),choices[1:]+choices[:1]):
            for s in order: times[s].append(time_call(calls[s],50,300,'event'))
        refine[label]=times
        selected[label]=min(choices,key=lambda s:median(times[s]))
    splits=selected['c500']
    fns={'selected':wrapped(splits),'dense':dense(selected['dense'])}
    if splits!=64:fns['fixed64']=wrapped(64)
    cases=[('random',i) for i in range(8)]+[(x,0) for x in ('early-peak','late-peak','large-v')]
    row=rows[key]={'candidate_splits':splits,'baseline_splits':selected['dense'],'refine_us':refine,'max_abs':{}}
    for name,fn in fns.items():row['max_abs'][name]=check_kernel(name,fn,b,n,cases,'cuda',False,h)
    xs=make_inputs(b,n,113,'random','cuda',h)
    for clock in ('event','wall'):
        samples=[]
        for i in range(6):
            names=list(fns);p=i%len(names);names=names[p:]+names[:p]
            if len(names)==3 and i%2:names.reverse()
            samples.append({k:time_call(functools.partial(fns[k],*xs),200,2000,clock) for k in names})
        row[clock]=samples;save()
    row['complete']=True;save()
