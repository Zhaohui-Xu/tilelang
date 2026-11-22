# C500 MLA 竞赛实现恢复

## 当前版本入口（2026-11-23）

本目录中的候选和结论保留其历史环境与适用范围。当前两个完整公开示例分别固定保存为：

- [旧栈验证版：6e71b2a](https://github.com/Zhaohui-Xu/tilelang/tree/6e71b2aa09825ba1f9fbfe50823f9d7ae978a49e/examples/deepseek_mla/maca)，对应 TileLang-MetaX f549117c。
- [Dev 适配版：f1f283ab](https://github.com/Zhaohui-Xu/tilelang/tree/f1f283ababe3d26f17f5f9855827a1f5961cdfd7/examples/deepseek_mla/maca)，对应 TileLang-MetaX dev90d49f35。

两版各有 README、原始性能 JSON 和图，不混用跨环境结果。当前提交目标是子仓库 dev；此处的历史比赛得分不代表最新通用实现的正确性或性能。

本目录保留比赛实现恢复与通用化过程的历史记录。环境为 C500、PyTorch `2.8.0+metax3.7.1.3`、TileLang-MetaX `f549117c`。在加载 MACA SDK 和已构建的 TileLang 后，将本目录加入 Python 模块搜索路径。

**竞赛 Best 固定为 A275（OJ 159816）。** 复现比赛时导入 `contest_mla.run_kernel`。验收依据为 OJ 返回源码、原始 ABI 与分支、历史六项 SPJ 回执。后续通用化从该冻结源派生，各候选单独记录。

- `contest_mla.py` 保留 OJ 159816 的完整原文，SHA-256 为 `c18eb31cb42c01de9b11b61cfbb85123da3855046431329cb3eefa321a937191`。用这个入口复核历史实现。
- `full_token_mla.py` 保留相同的 2K、4K、8K 分支，另从比赛档案 A254 的完整组合中提取 16K、32K、64K 三个 factory。提取保留装饰器和函数体。长序列按长度选择对应 factory，缓存仅含编译好的 kernel。它处理六种长度的全部 token，固定指数偏移和 FP16 部分量仍有输入范围限制。

- `stable_mla.py` 是从 A275 主循环派生的稳定化候选，保留概率 fragment 对 PV 和分母的复用，使用逐行动态最大值和 FP32 分块结果。以 `from stable_mla import run_kernel` 选取此候选。各长度的同轮对照与压力用例见[稳定化实验记录](stable-best/ResultSummary.md)。

三个入口均按比赛 ABI 将结果写入预先提供的输出。调用示例中的 `q`、`q_pe`、`kv`、`k_pe` 是同一 C500 上的连续 FP16 张量；输出和 `q` 同形，`heads=16`、`kv_heads=1`、`dim=512`、`pe_dim=64`。本轮在 batch 1 下验证。

```python
from contest_mla import run_kernel

batch, heads, dim = q.shape
_, kv_length, kv_heads, _ = kv.shape
output = q.new_empty(q.shape)
run_kernel(q, q_pe, kv, k_pe, output, batch, heads, kv_heads, kv_length, dim, q_pe.shape[-1])
```

| KV 长度 | OJ 原提交路径 | 完整 token 组合 |
|---|---|---|
| 2K | A217 主核，原 JIT，PV 与分母共用概率 fragment | 直接调用原提交 |
| 4K | 原 4K 专用 kernel，Cython 与显式打包工作区 | 直接调用原提交 |
| 8K | A223 主核，原 JIT，PV 与分母共用概率 fragment | 直接调用原提交 |
| 16K | 每个 split 读取 6/8 个 tile | A231 全部 8 个 tile |
| 32K | 原动态最大值与 FP16 部分输出，读取全部 16 个 tile | A254 分两组处理全部 16 个 tile |
| 64K | 每个 split 读取 15/32 个 tile | A235R1 分四组处理全部 32 个 tile |

历史 OJ 成绩归属于原提交。常规输入性能和不同输入族的完整输出误差分别记录。六种长度的速度及失败用例见[同卡实测记录](contest-recovery/ResultSummary.md)。

## 竞赛 Best 成绩与本地复核

原始 OJ 总分 97.33，六项 Accepted，SPJ 记录的 kernel 时间相加为 355 µs。本地复核使用同一 A275、单张 50% C500 切片和独立生成的输入。下表末列给出标准正态输入的完整调用 event 中位数，含新输出分配。比赛 ABI 由评测器传入输出缓冲区。

| KV 长度 | 历史 OJ 时间 µs | 单项分数 | 本地 A275 时间 µs |
|---|---:|---:|---|
| 2K | 19 | 95 | 20.62 |
| 4K | 24 | 97 | 25.34 |
| 8K | 35 | 97 | 38.80 |
| 16K | 52 | 98 | 正态对拍 0/100，未计时 |
| 32K | 116 | 98 | 120.16 |
| 64K | 109 | 99 | 正态对拍 0/100，未计时 |

本地归档含各题点 SPJ 报告、输入文件名和公开形状配置。回执记录了原始评测结果；逐输入复测仍需要当时的隐藏输入及计时环境。OJ 原始回执保存在本地比赛工作区，未包含在本公开归档中；源码见本目录的 `contest_mla.py`。

## 后续通用化的已有对照

当前 C500 上标准正态输入的完整调用，六组 AB/BA 中位数，单位 µs。短序列使用原 A275 入口，长序列使用全 token 组合。对照为之前的 dense Cython 版本。

| KV | 恢复路径 event | 对照 event | 恢复路径 wall | 对照 wall |
|---|---:|---:|---:|---:|
| 2K | 20.62 | 73.46 | 22.16 | 74.01 |
| 4K | 25.34 | 74.71 | 27.21 | 77.47 |
| 8K | 38.80 | 79.08 | 40.33 | 82.28 |
| 16K | 61.40 | 79.32 | 63.16 | 79.98 |
| 32K | 108.83 | 123.59 | 110.09 | 125.09 |
| 64K | 197.28 | 251.48 | 199.16 | 253.64 |

六个长度各通过 100/100 组标准正态输入。末尾峰值压力输入使六条恢复路径均产生非有限输出，64K 的大幅 V 压力用例也产生非有限输出。以上速度限定于通过检查的常规输入。通用 dense 对照在同轮全部 102 项上通过。
