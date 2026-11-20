import json
from pathlib import Path
from statistics import median
root=Path(__file__).resolve().parent
def read(name): return json.loads((root/(name+'.json')).read_text())
lines=['# C500 参数化 MLA 验证','', '开发版 d0c9f99，归档稳定版 8606870，公开比较源码 tile-ai/tilelang-metax 90d49f35。C500 50% 算力/32000 MiB，PyTorch 2.8.0+metax3.7.1.3、TileLang f549117c。', '', '## 全输出正确性','','FP32 参考，连续 FP16；seed 0–7，另测首峰、末峰和大 V，rtol=atol=.002。','','| B × KV | 通过 | 最大绝对误差 |','|---|---:|---:|']
for k,v in read('boundary').items():
    lines.append(f'| {k} | {11 if v["complete"] else 0} | {v["max_abs"]["candidate"]:.7g} |')
for stage,title in [('archive','归档稳定版'),('upstream','公开基线')]:
    lines+=['',f'## 对比{title}','','单位 µs，取六组 AB/BA 的中位数。每项计时含输出分配和完整调用，输入 seed 23，每批 200 次预热及 2000 次测量。加速比为对照除以开发版。','','| B × KV | 对照 split | event 对照 / 开发 | 加速比 | wall 对照 / 开发 | 加速比 | 开发胜批 event / wall |','|---|---:|---:|---:|---:|---:|---:|']
    for k,v in read(stage).items():
        cells=[];wins=[]
        for clock in ('event','wall'):
            rows=v[clock]; a=median(x[stage] for x in rows); b=median(x['candidate'] for x in rows)
            cells.extend([f'{a:.2f} / {b:.2f}',f'{a/b:.3f}×']);wins.append(str(sum(x[stage]>x['candidate'] for x in rows)))
        lines.append('| '+' | '.join([k,str(v.get('splits',64)),*cells,' / '.join(wins)])+' |')
lines+=['','公开基线在 2K/4K/8K 沿用上一轮筛选配置。在 16K/32K/64K 枚举 32-token tile 可整除的二次幂 split，以三轮各 20 次预热、60 次 event 测量的中位数选择，再独立配对计时。搜索范围固定 tile N32/H16。','', '命令与输入生成见 suite.py、continue.py；逐批指标与日志见同目录对应 stage 文件。']
(root/'ResultSummary.md').write_text('\n'.join(lines)+'\n')
