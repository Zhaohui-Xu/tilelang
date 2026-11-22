# 面向沐曦 C500 的高性能 TileLang MLA Decode

**2026-11-22 用户确认的现行大纲。** 报告、公开 README 和 PR 材料均以此为叙述依据。README 采用精简版，代码旁保留与实现直接相关的设计理由。

**报告要回答的问题**　怎样让更多查询头共享一次 KV 载入，同时让每个 C500 wave 承担得起自己的输出累加和片上数据？正文沿 AMD case 的顺序，从 Hopper 的列分工讲到 C500 的实现取舍，最后用调优 dense 基线的同卡完整调用说明收益。25% 切片的映射对照放在性能部分的折叠项。

**开篇**　从已有 Hopper MLA 示例切入。同一份 QK、softmax、PV 数学表达中，头块宽度同时影响累加器分工、一次 KV 载入服务的头数和可并行的 block 数。本文给出针对 C500 wave 与 MFMA 映射的解决方案。

## 1. Architectural Considerations and Optimization Strategies｜架构差异与优化设计

### 1.1 从 Hopper 到 C500：一块 64 × 512 输出由谁计算？
官方 Hopper 示例的 WGMMA 路径以 128 线程 warpgroup 协作，M 维最小为 64。两个 warpgroup 沿输出列分担累加器，通过 shared memory 取得所需概率。C500 的 PV 映射用一对 64-lane wave 覆盖 16 个头的 512 列，再沿行组合四对形成 64 头块。16 × 256 是一个 wave 多次 MFMA 负责的软件输出区域。用简短文字交代两者的累加器归属。

### 1.2 片上存储预算：为什么宽头块要改变 Q 的放置？
FP16 的 32 × 576 key tile 占 36 KiB，将头块由 16 扩到 64，Q 从 18 KiB 增到 72 KiB。若两块都放在 shared memory，仅 Q 与 key 就需 108 KiB。区分逻辑尺寸、编译后的共享内存使用和寄存器压力。对照 shared Q 与 fragment Q，解释宽块中多个行方向的 wave 如何消费同一个 KV tile。B64/H128/1K 实测约 1248 → 648 µs，支持整套宽块调度，包含 Q 放置、线程组织与 split 的共同变化。

### 1.3 从 block 到 wave：复用概率 fragment
沿一个 KV tile 追踪打包的 512+64 列、score GEMM、在线 softmax 和 shared P。wave 将适合 MFMA 的概率 fragment 用于 PV，再与全 1 操作数相乘累加分母。区分 block 内对 KV 的复用和 wave 内对概率操作数的复用。

### 1.4 并行度与归并预算：联合选择头块和 split
主核 block 数为 B × (H / head_tile) × splits，多 split 的 FP32 workspace 为 4 × B × H × splits × (512+2) 字节。小负载需要并行工作；宽块过渡区已有足够头组时，降低 split 可节省写回与归并。九个额外且配置改变的形状，完整调用 wall 减少 10.4%–14.8%。

**本章收束**　对于采用协作式矩阵指令和块内共享存储的后端，沿输出列分工决定累加器归属，沿查询头分工决定 KV 复用范围，沿序列分工决定并行度与归并成本。合法 fragment 映射由目标指令决定，头块和 split 的选择还取决于寄存器、共享内存和独立工作量。

## 2. Performance Evaluation｜性能评估

### 2.1 大 batch 吞吐
先展示 AMD 风格 B64/B128 双面板柱状图。H128、KV 1K 至 16K 的十个标称形状，默认 kernel 对调优 dense 基线的 wall 加速为 1.80×–2.88×。预打包分页 FlashMLA 在十个点上均更快，列明数据准备和输出边界。

### 2.2 小 batch 延迟
B1/2/4、H16，共十二个实测形状，wall 加速为 1.43×–2.79×。正文用 B1/H16/8K 的 92.24 → 45.10 µs 说明单次调用收益。

### 2.3 测量口径与证据边界
报告 C500 50% 切片、FP16、基线提交与搜索范围、预热和重复次数。热态完整调用包括输出分配和所有 kernel launch，编译在计时前完成。Hopper 提供列分工思路。另在 25% C500 切片上固定 B64/H128/1K，分别以 4、2 个 split 比较两种 PV 映射。

## 3. Running the Example｜运行与复现

给出 TileLang-MetaX 与 PyTorch-MACA 版本、连续 FP16、单 token、KVH1、D512/PE64、同批等长、H 为 16 的倍数、KV 长度 1–65536。原 18 形状共 198 次全输出对拍，覆盖尾块、首尾峰值和大 V。保留正确性检查、同卡基线对比与可选自动调优入口。

[同卡实验记录](amd-case-c500/ResultSummary.md)


## 执行记录（2026-11-22）

公开报告沿用 AMD README 的节奏：一段承接 Hopper，将架构差异和 C500 设计决定放在同一小节，先展示 B64/B128 主图，最后给复现入口。现有 22 点性能结果足以成稿。公开 README 已用文字和一行分配代码说明映射及存储取舍，仅保留两张实测性能图。逻辑 tile 尺寸与编译资源分别标注。

同卡映射对照已完成。4 个 split 的沿列版延迟低 4.70%，默认 2 个 split 下高 74.54%；继续使用现有的公开默认分工。[实测、资源与范围](wave-mapping-c500/ResultSummary.md)
