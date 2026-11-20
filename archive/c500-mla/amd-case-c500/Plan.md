# AMD 对照与 C500 调度搜索
核对版本：TileLang 主仓 4ed9b9e；公开 dense 基线 TileLang-MetaX 90d49f35；运行栈 TileLang-MetaX f549117c、PyTorch 2.8.0+metax3.7.1.3，C500 50%/32000 MiB。
- 已发布对照目录使用 batch 64/128、128 Q heads、1 KV head、512+64，名义长度 1024/2048/4096/8192/16384。本实验在 C500 运行相同名义形状、均匀 dense FP16、单 token；AMD AITER 脚本自身配置为 BF16 和每项长度 N+2i。数据不能跨设备解释为加速比。
- 先在原 H16 版本补 B2/4、2K/8K/32K 同卡 split 筛选、11 个正确性输入、六批配对 200/2000。
- 在新版本中按16头分块，保留 B1 默认路径，试单 split 直接归一输出和多 split 合并。先按独立目录验证 H16 回归、H128 尾部及大小 split。
- 大 batch 名义形状和小 batch 性能网格：同一输入验证每种候选/基线，候选参数为不超过 token tile 数的二次幂，含单 split 直接归一输出、跨 wave lane 的多 split 归并，FP32 workspace 上限 3 GiB。dense 基线 tile N32/H16，全部合法二次幂 split，上限为 partial output 3 GiB；归档范围与逐点可选配置。三个轮次交错做短筛，独立 seed 23 重新实测六轮两时钟，输出含分配。全张量 FP32 对拍分块做，减少临时内存。
- 已安装的 MetaX FlashMLA 独立测本机原生分页调用：用相同输入预打包 Q/KV，连续页表与元数据预先生成，返回 O 与 LSE。只比较 attention 输出，单列原生接口时间；不宣称 dense API 端到端等价。
- 固定环境无需安装 TileOPs。TileOPs-Metax 91622187 的 pyproject 要求 Python>=3.12、torch>=2.13,<2.14、tilelang==0.1.12，当前栈不同。主仓快照不含 tilelang.maca.intrinsics；文档明确 MACA build 前提。
- 不推送。保存工作代码、组合选择、未获益项、原始时钟与版本。

## 实测驱动的后续决定
1. 初版 H16/64 splits 在 AMD 网格的短序列落后，保留全部材料；引入 16 头组及可调 split。第一轮 H128 测量后，针对不同头组重复读取 KV 的问题，在 B64/H128/1K 与 B128/H128/8K 比较普通 register-Q 两次 QK GEMM 与 K576 打包、显式 MMA、多 row wave 路径。入围配置再跨形状、用新输入验证。
2. 同时扩展 dense 对照，检查 H16/32/64、N16/32、1/4/8 及原胜出 split，记录可用共享内存边界与不合法配置。厂商分页路径通过独立 reference 对拍，使用真实安装版本及原生页表。
3. 已查硬件支持每块最多 1024 线程。合并实现将 16/32/64/128 的头块与合法二次幂 split 放入有界搜索；保留多 seed、最长 64K、偏离比赛网格和非对齐长度对拍。初筛及 shortlist 后用独立 seed 再计时。
4. 通过中间 batch/head 数实测确定默认切换策略，公开入口不需要逐形状表。重新用默认 factory 直接跑 22 个配对，不传 split 或 head tile，再回归原 18 形状与附加尾块。对两个邻居更快的中间形状，另作低 split 尝试，保留命令及接受/拒绝记录。

### 最后一项 transition 采纳口径
已观察到的 B16/H128/2K、B4/H128/8K 为提出假设的两个点，不参与外推门。对刚启用 64 头块且宽头组数 × KV tile 数不超过 3072 的工作量，比较现行 split 与降低到最多一批约 64 个独立 block 的方案。只在至少五个确实改变配置的额外形状上，中位 wall 收益至少 3%、没有超过 5% 的回退，并保持 22 点已验证配置时采纳；否则保留现行默认和显式调优入口。六轮 AB/BA，20 warmup、100 iterations，全输出检查，seed 1321。
