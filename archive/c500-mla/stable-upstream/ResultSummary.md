# 当前稳定版与开源 C500 MLA 的同卡比较

2026-11-20。候选是独立的 `stable_mla.py`，SHA-256 `3567b13a77c1194f2a6303de460dc95ce37e66eb51ed746587c154b2c4339688`。实际开源对照为 GitHub 默认分支 TileLang-MetaX `90d49f35` 的 [MLA decode 实现](https://github.com/tile-ai/tilelang-metax/blob/90d49f353ec88bd0d346474f143003d865dd3b79/examples/deepseek_mla/example_mla_decode.py)，Git blob `1404fac5`。与固定环境旧路径的 `flashattn` 装饰器和函数体经 AST 核对相同，两个内置示例调用的参考误差阈值有更新。本轮显式加载新的完整源文件。双方使用相同的已装运行栈 TileLang-MetaX `f549117c`、PyTorch `2.8.0+metax3.7.1.3`、MACA 3.7.1.5，没有以新编译器重建环境。[版本清单](revisions.json)

## 实测结果

C500（50% 算力、32,000 MiB）、batch 1、H=16、1 个 KV head、内容维 512、位置维 64、FP16 输入输出、连续 KV、单 token decode。两版均运行完整调用，包含输出和工作区分配、设备执行及归并。候选固定 64 splits。上游沿用源文件的 32-token、16-head tile；对已完成筛选的 2K、4K、8K 遍历从 1 到 KV 长度/32 的全部合法二次幂 split 数。在 seed 17 上检查各配置并用 event 的三批中位数筛选（每批 20 warmup/60 iter，轮换顺序），再对入选配置与候选做正式验证和独立配对计时。该筛选覆盖 split 数，未搜索其他 tile 大小或不同硬件。

| KV | 口径 | 上游 splits | 上游 event µs | 稳定版 event µs | event 加速 | wall µs（上游 / 稳定版） | wall 加速 | event 胜出批数 | 固定 64 splits 的 event 加速 |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| [2K](tuned-2k/metrics.json) | 筛选后 | 64 | 34.83 | 23.87 | 1.46× | 36.44 / 25.35 | 1.44× | 6/6 | [1.45×](upstream-2k/metrics.json) |
| [4K](tuned-4k/metrics.json) | 筛选后 | 64 | 46.86 | 29.88 | 1.57× | 48.44 / 31.41 | 1.54× | 6/6 | [1.57×](upstream-4k/metrics.json) |
| [8K](tuned-8k/metrics.json) | 筛选后 | 128 | 89.94 | 43.60 | 2.06× | 91.49 / 45.02 | 2.03× | 6/6 | [2.68×](upstream-8k/metrics.json) |
| [16K](upstream-16k/metrics.json) | 固定对照 | 64 | 208.25 | 77.19 | 2.70× | 210.98 / 79.02 | 2.67× | 6/6 | [2.70×](upstream-16k/metrics.json) |
| [32K](upstream-32k/metrics.json) | 固定对照 | 64 | 387.17 | 170.92 | 2.27× | 389.12 / 172.49 | 2.26× | 6/6 | [2.27×](upstream-32k/metrics.json) |
| [64K](upstream-64k/metrics.json) | 固定对照 | 64 | 746.12 | 286.03 | 2.61× | 747.88 / 287.34 | 2.60× | 6/6 | [2.61×](upstream-64k/metrics.json) |

正式正确性门为 100 个随机 seed、首尾峰值及大幅 V 共 103 个全输出 FP32 对拍（rtol=atol=0.002）；计时输入单独用 seed 23。每个时钟各六批 AB/BA，200 次热身、2,000 次测量，报告六批中位数之比。所有配对原始数据、命令和源码哈希对应表中 run 子目录；未插桩计时。[核算结果](paired-summary.json)

用户确认租机到时关机。16K 的 split 筛选任务未回收到可验证的完整结果，32K/64K 筛选尚未启动。这三行使用关机前已回收的 64-split 对照。所有六个长度均另有 32-split 原始对照。

## 其他开源实现的可比性

- [TileOPs-Metax dense MLA](https://github.com/MetaX-MACA/TileOPs-Metax/blob/916221870e092fff9c2219ef7d15f2e52299e141/benchmarks/ops/bench_mla.py) 有独立的 dense API，并支持在 MACA benchmark 中请求调优。实测环境未安装 `tileops`。该提交的标准包依赖见 [pyproject.toml](https://github.com/MetaX-MACA/TileOPs-Metax/blob/916221870e092fff9c2219ef7d15f2e52299e141/pyproject.toml)。本表对照为 TileLang-MetaX 示例。
- [MetaX FlashMLA](https://github.com/MetaX-MACA/FlashMLA/blob/b246af19465084b1a97d4de480c3ea0c1b356e4c/flash_mla/flash_mla_interface.py) 接受打包的 Q 与分页 KV，返回输出和 LSE，还使用预先计算的调度元数据。当前卡上的安装包为 `1.0.1+metax3.7.1.3torch2.8`，Python 接口 Git blob 为 `f515c1fd`，与所核实的默认分支接口 `777dc900` 不同。FlashMLA 需要单独定义预打包缓存及转换费用的对照。
