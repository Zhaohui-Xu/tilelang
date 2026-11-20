"""Locate head-sharing crossover on batch and head counts outside the AMD grid."""
import json
import functools
import statistics
from pathlib import Path
import torch
from example_mla_decode_maca import mla_decode_maca
from benchmark_mla_decode_maca import reference,time_call

torch.set_float32_matmul_precision('highest')
path=Path('cross.json')
rows=json.loads(path.read_text()) if path.exists() else {}
def save():
    p=path.with_suffix('.tmp');p.write_text(json.dumps(rows,indent=2,allow_nan=False));p.replace(path)
def pow2(n):return 1<<(max(n,1)-1).bit_length()
def recommend(b,h,n,tile):
    tiles=(n+31)//32;groups=b*(h//tile)
    cap=1<<(tiles.bit_length()-1)
    if tile==16:
        occ=pow2((64+groups-1)//groups)
        return min(cap,occ if tiles<=8*occ else pow2((tiles+3)//4))
    occ=pow2((512+groups-1)//groups)
    return min(cap,occ,1<<(max(1,tiles//16).bit_length()-1))
def run(b,h,n):
    key=f'{b}x{h}x{n}'
    if rows.get(key,{}).get('complete'):return
    gen=torch.Generator(device='cuda').manual_seed(433)
    xs=tuple(torch.randn(size,device='cuda',dtype=torch.float16,generator=gen).contiguous() for size in ((b,h,512),(b,h,64),(b,n,1,512),(b,n,1,64)))
    ref=reference(*xs)
    row=rows[key]={'samples':{},'max_abs':{}}
    calls={}
    for tile in (16,32,64):
        if h%tile:continue
        s=recommend(b,h,n,tile)
        # Widen the wide schedules toward more splits when the grid is small.
        opts=sorted({s,min(n//32,4*s)}) if tile>16 and b*(h//tile)<64 else [s]
        for splits in opts:
            k=mla_decode_maca(b,n,h,splits,tile)
            def call(k=k):
                out=torch.empty_like(xs[0]);k(*xs,out);return out
            out=call().float();torch.testing.assert_close(out,ref,atol=.002,rtol=.002)
            label=f'{tile}/{splits}'
            calls[label]=call
            row['max_abs'][label]=(out-ref).abs().max().item()
            row['samples'][label]=[]
    keys=list(calls)
    for i in range(6):
        shift=i%len(keys);order=keys[shift:]+keys[:shift]
        if i%2 and len(keys)>2:order.reverse()
        for label in order:row['samples'][label].append(time_call(calls[label],10,50,'wall'))
    row['complete']=True
    save()
for b,h,n in [(1,128,2048),(4,128,2048),(16,128,2048),(32,128,2048),(1,128,8192),(4,128,8192),(16,128,8192),(32,128,8192),(16,64,2048),(32,64,8192),(64,64,2048),(32,32,8192)]:
    print(b,h,n,flush=True);run(b,h,n)
