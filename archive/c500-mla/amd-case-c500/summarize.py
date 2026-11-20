"""Render measured medians; keep native paged results identified by their own ABI."""
from pathlib import Path
import json,statistics
root=Path(__file__).resolve().parent
med=statistics.median
lines=['# C500 调度筛选与 AMD 标称形状复测','', 'C500 50% 算力、32000 MiB，固定 MACA PyTorch 2.8.0 与 TileLang-MetaX f549117c。所有时间为 µs。','']
if (root/'batchperf.json').is_file():
    lines+=['## 原版 H16、64 splits 的小 batch 回归','', '| B | N | 基线 split | 基线 wall | 原版 wall | 比值 |','|---:|---:|---:|---:|---:|---:|']
    for k,v in json.loads((root/'batchperf.json').read_text()).items():
        if not v.get('complete'): continue
        a,b=[med(r[name] for r in v['wall']) for name in ('batchperf','candidate')]
        lines.append('| '+' | '.join([*k.split('x'),str(v['splits']),f'{a:.2f}',f'{b:.2f}',f'{a/b:.2f}×'])+' |')
    lines+=['','每个实现 8 个随机输入及首峰、尾峰、大 V；每个时钟六批 200 warmup、2000 次迭代，AB/BA。','']
for stage,title in [('amd','128 头大 batch'),('small','16 头小 batch')]:
    if not (root/(stage+'.json')).is_file(): continue
    data=json.loads((root/(stage+'.json')).read_text())
    lines.extend([f'## {title}','','| B | H | N | 候选 / dense split | 候选 event | dense event | 候选 wall | dense wall | dense / 候选 wall | FlashMLA 原生 wall |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for k,v in data.items():
        if not v.get('complete'): continue
        a,b,c,d=[med(r[name] for r in v['timings'][clock]) for clock in ('event','wall') for name in ('c500','dense')]
        paged=[r['paged_native'] for r in v['timings']['wall'] if 'paged_native' in r]
        lines.append('| '+' | '.join([*k.split('x'),f"{v['c500_splits']} / {v['dense_splits']}",*(f'{t:.2f}' for t in (a,b,c,d)),f'{d/c:.2f}×',f'{med(paged):.2f}' if paged else '未通过验证'])+' |')
    lines+=['']
lines+=['大小 batch 各自用 seed 17 检验并筛选配置，seed 23 再做独立全输出 FP32 对拍和计时。筛选三轮，每轮 3 次预热、12 次计时；复测每个时钟 6 轮，每轮 4 次预热、20 次计时。dense 固定 N32/H16 tile；分片数为合法二次幂，partial workspace 搜索上限为 3 GiB。选中配置对应筛选中位数。','', 'FlashMLA（如该行通过验证）使用打包好的 Q、页大小 64 的 KV 和预生成的页表及调度元数据，原生调用同时返回 LSE。原生 API 耗时单列。其数据准备不进入计时。dense 的输入为四个分开的连续 FP16 张量，计时含输出和内部 workspace。','', 'AMD AITER 脚本张量为 BF16 且序列长度 N+2i，本表使用统一长度、FP16。共同维度为标称 B/H/N、单 query token、KVH1、D512/PE64。没有跨设备加速结论。']
(root/'ResultSummary.md').write_text('\n'.join(lines)+'\n')
if (root/'wide.json').exists():
    lines+=['','## 宽 head tile 同卡比较','','配置按每块 query heads、KV tile、splits、线程数排列，KV tile 标记 0 表示 K576 打包实现（实际 N tile=32）。dense 配置顺序为头块、KV tile、splits。','','| B×H×N | 宽块配置 | dense 配置 | 宽块 wall µs | 原 16 头块 wall µs | dense wall µs | dense / 宽块 |','|---|---|---|---:|---:|---:|---:|']
    for key,v in json.loads((root/'wide.json').read_text()).items():
        if 'wall' not in v: continue
        a,b,c=[med(r[n] for r in v['wall']) for n in ('wide','c500','dense')]
        lines.append(f"| {key} | {v['best']} | {v['best_dense']} | {a:.2f} | {b:.2f} | {c:.2f} | {c/a:.2f}× |")
    lines+=['','代表形状 B64/H128/1K、B128/H128/8K 先筛候选宽块。其余形状复验 pilot 入围配置。dense 比较在适合共享内存边界的 head tile、KV tile 及 split 上进一步筛选。所有入围配置用 seed 213 重新配对。']
if (root/'confirm.json').exists():
    lines+=['','## 小 batch 独立长计时复核','','粗筛后用 seed 73，在各实现前三名（候选额外含固定 64）中二次筛选；三组 50/300。独立 seed 113，六组交错顺序 200/2000。每个入围实现另过 8 个随机 seed、首尾峰值和大 V。','','| B×H×N | 候选 / dense split | 候选 event | dense event | 候选 wall | dense wall | dense / 候选 wall | 原 64 / 候选 wall |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for k,v in json.loads((root/'confirm.json').read_text()).items():
        if not v.get('complete'):continue
        a,b,c,d=[med(x[name] for x in v[clock]) for clock in ('event','wall') for name in ('selected','dense')]
        old=med(x.get('fixed64',x['selected']) for x in v['wall'])
        lines.append(f"| {k} | {v['candidate_splits']} / {v['baseline_splits']} | {a:.2f} | {b:.2f} | {c:.2f} | {d:.2f} | {d/c:.2f}× | {old/c:.2f}× |")
(root/'ResultSummary.md').write_text('\n'.join(lines)+'\n')
if (root/'defaults.json').exists():
    final=json.loads((root/'defaults.json').read_text())
    tuned=json.loads((root/'release.json').read_text())
    head=['# C500 MLA 公开候选最终验证','',
          '同一份 TileLang 源码，单 token、dense contiguous FP16、KVH1/D512/PE64。下表用公开 factory 的默认参数；绝对时间为完整调用的 µs。dense 为固定版本 TileLang-MetaX 公开实现，经过同卡配置筛选。','',
          '| B×H×N | 默认头块 / split | dense 头块 / KV tile / split | C500 event | dense event | C500 wall | dense wall | wall 加速 | 厂商原生 wall |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,v in final.items():
        if not v.get('complete'):continue
        a,b,c,d=[med(x[name] for x in v[clock]) for clock in ('event','wall') for name in ('default','dense')]
        native=[x['paged_native'] for x in v['wall'] if 'paged_native' in x]
        head.append('| '+' | '.join([k,' / '.join(map(str,v['default'])),' / '.join(map(str,v['dense_config'])),*(f'{x:.2f}' for x in (a,b,c,d)),f'{d/c:.2f}×',f'{med(native):.2f}' if native else '—'])+' |')
    head+=['','小 batch 采用 seed 367 和 6×200/2000（warmup/iterations），两方 AB/BA。大 batch 使用相同 seed、6×5/30，三个实现交错顺序。另见独立粗筛、二次筛选、复测数据。厂商原生列为预先打包好的 Q、分页 KV、页表和调度元数据，计算还返回 LSE。','',
           '## 默认策略以外的调优空间','','下表同一组 seed 613 输入、相同 6×5/30 计时边界，记录大 batch 最快入围配置、当轮默认配置与已筛选 dense 对照。存在轮次波动。','','| B×H×N | 复筛选中 头块/split | 选中 wall | dense wall | 原生分页 wall |','|---|---:|---:|---:|---:|']
    for k,v in tuned.items():
        if 'baseline_config' not in v:continue
        c,d,f=[med(x[n] for x in v['wall']) for n in ('selected','dense','paged_native')]
        head.append(f"| {k} | {v['selected']} | {c:.2f} | {d:.2f} | {f:.2f} |")
    if (root/'unseen.json').exists():
        obs=json.loads((root/'unseen.json').read_text())
        head+=['','## 整合版网格外邻居筛查（最终选型前）','',f'{len(obs)} 种额外输入组合，共 {sum(v["checks"] for v in obs.values())} 次全输出 FP32 对拍。下列形状另与相邻 split 比较，列出默认耗时相对最佳已测邻居的比值。','','| B×H×N | 默认 split | 默认 / 最快邻居 wall |','|---|---:|---:|']
        for k,v in obs.items():
            if 'neighbor_wall_samples' in v:
                samples={int(s):med(t) for s,t in v['neighbor_wall_samples'].items()}
                head.append(f"| {k} | {v['default_splits']} | {samples[v['default_splits']]/min(samples.values()):.3f}× |")
    head+=['','## 实验过程与对照','']
    (root/'ResultSummary.md').write_text('\n'.join(head+lines[1:])+'\n')
if (root/'regression.json').exists():
    exact=json.loads((root/'regression.json').read_text())
    extra=json.loads((root/'default_edges.json').read_text())
    cross=json.loads((root/'cross_defaults.json').read_text())
    md=(root/'ResultSummary.md').read_text()
    md+='\n## 最终 correctness 与中间配置复核\n\n'
    md+=f'最终默认入口重放原 {len(exact)} 个形状，{sum(v["checks"] for v in exact.values())} 次完整输出对拍通过；另一个 {len(extra)} 组合的检查集合共 {sum(v["passed"] for v in extra.values())} 次通过，两组形状有重合。\n\n'
    md+='在最后的降 split 改动前，以独立 seed 929 复测中间形状的默认调度、前一轮胜者和默认 split 的相邻二次幂。以下均为完整调用 wall 微秒（六轮 10/50）。\n\n| B×H×N | 默认头块/split | 邻居复核最快 | 默认 wall | 最快 wall | 默认 / 最快 |\n|---|---:|---:|---:|---:|---:|\n'
    for k,v in cross.items():
        times={n:med(x) for n,x in v['samples'].items()};best=min(times,key=times.get);a=times[v['default']];b=times[best]
        md+=f"| {k} | {v['default']} | {best} | {a:.2f} | {b:.2f} | {a/b:.3f}× |\n"
    if (root/'transition.json').exists():
        d=json.loads((root/'transition.json').read_text())
        md+='\n## 进入宽头块时减少 split 的独立试验\n\n前两个形状为发现问题的样本；后续形状用于检验外推。比值为 proposed wall / 现行默认 wall，越小越快。\n\n| B×H×N | 原 split | proposed split | 比值 |\n|---|---:|---:|---:|\n'
        for k,v in d.items():md+=f"| {k} | {v['old_splits']} | {v['proposed_splits']} | {v['wall_ratio']:.3f}× |\n"
    (root/'ResultSummary.md').write_text(md)
if (root/'adopt.json').exists():
    adopted=json.loads((root/'adopt.json').read_text())
    d=json.loads((root/'transition.json').read_text())
    held=[v['wall_ratio'] for v in list(d.values())[2:] if v['old_splits']!=v['proposed_splits']]
    md=(root/'ResultSummary.md').read_text()
    intact=[k for k,v in adopted.items() if v.get('identical')]
    changed={k:v for k,v in adopted.items() if 'wall' in v}
    md+='\n## 最终采用的中间负载规则\n\n'
    md+=f'在九个额外且配置确实改变的输入上，降低 split 后 wall 耗时减少 {(1-max(held))*100:.1f}%–{(1-min(held))*100:.1f}%，中位减少 {(1-med(held))*100:.1f}%。按预先记录的采纳门保留。最终 factory 在 {len(intact)} 个唯一形状上生成的设备和 host 启动源码与产生表中数据的版本 SHA-256 均相同，这些形状覆盖完整 22 点默认性能表及原边界集合。对以下变化再用独立 seed 2221，通过最终 factory 进行全输出对拍和六批 20/100 AB/BA 计时。\n\n'
    md+='| B×H×N | 新 split | 原 wall µs | 新 wall µs | 原 / 新 |\n|---|---:|---:|---:|---:|\n'
    for k,v in changed.items():
        old,new=[med(x[n] for x in v['wall']) for n in ('old','new')]
        md+=f"| {k} | {v['splits']} | {old:.2f} | {new:.2f} | {old/new:.3f}× |\n"
    md+='\n本次证据适用于所列工作量与 C500 的 50% 算力切片。大 batch 的 native FlashMLA 使用分页、预打包与调度元数据。其他形状可用公开 CLI 的 --tune 搜索并复核。主仓快照把 MACA 列为生态后端，本例在固定的 TileLang-MetaX 栈执行。\n'
    (root/'ResultSummary.md').write_text(md)
if (root/'cli-results.json').exists():
    cli=json.loads((root/'cli-results.json').read_text())
    with (root/'ResultSummary.md').open('a') as f:
        f.write('\n## 公开调优入口最后核验\n\nB16/H128/2K 的全搜索选中默认 64 头块 / 2 splits。B9/H128/4K 全搜索在本次复筛选中 64 头块 / 16 splits，默认使用 4 splits；同轮复筛 event 中位数分别为 454.30 与 517.91 µs。该次最终 wall 只对选中配置与 dense 配对，未对 4/16 做独立配对 wall 加速验收。两个 CLI 各通过 6 个全输出输入。该结果保留为自动调优的作用边界。\n')
