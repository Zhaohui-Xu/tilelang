# C500 上 MLA 的 wave 分工与 KV split

在相同的 64 × 512 输出块和 8 个 wave 上，现有实现让每个 wave 负责 16 头 × 256 列；列分工让每个 wave 负责全部 64 头、连续 64 列。两者均使用 C500 的 MFMA。列分工移植的是 Hopper 示例沿输出列切分的思想。每个 wave 的输出同为 4096 个 FP32 累加值；概率 fragment、V fragment 和分母累加器的形状随分工改变。

## 完整调用

B64/H128/N1024，连续 FP16，单 KV 头，D512/PE64，64 头块、32-token KV tile、512 主核线程。新租 C500 为 25% 算力 / 16000 MiB，MACA 3.7.1.5、驱动 3.8.30、TileLang-MetaX f549117c、PyTorch 2.8.0+metax3.7.1.3。历史 22 点主表来自另一片 50% 切片。

| splits | 每个 split 的 KV tile 数 | 现有 wall / event (µs) | 列分工 wall / event (µs) | 列分工 wall 延迟变化 |
|---:|---:|---:|---:|---:|
| 4 | 8 | 650.56 / 646.81 | 619.96 / 616.56 | −4.70% |
| 2（公开默认） | 16 | 612.84 / 609.73 | 1069.65 / 1066.00 | +74.54% |

同一行使用相同输入、QK、softmax 公式、workspace 规格、归并及输出分配。每个时钟 6 轮，AB/BA 交错，每组 5 次预热、30 次计时，汇报批次中位数。wall 覆盖分配、主核和归并，末尾同步。两组的两个时钟均 6/6 同方向。每个 split 配置使用两组随机输入和首尾峰值各一组，两个实现均通过 FP32 全张量对照，atol=rtol=0.002。种子、误差与原始时间见 [4-split 样本](results/artifacts/measure.json) 和 [2-split 样本](results/default-two/artifacts/measure.json)。

## 编译与分工核查

逐配置核对实际计时的 JIT 缓存源码与保存的设备源码，再从对应 executable.so 的内嵌 kernel metadata 读取主核资源。动态 shared 取自 host launch，四个主核均为 45312 B，静态 shared 均为 0。生成源码中的输出索引反映不同的行、列归属。每个 split 配置内，两个实现的 host 源码相同，归并核设备源码也相同。

| splits | 分工 | MT registers | ST registers | private memory (B) |
|---:|---|---:|---:|---:|
| 4 | 16 × 256 | 248 | 20 | 0 |
| 4 | 64 × 64 | 256 | 38 | 12 |
| 2 | 16 × 256 | 246 | 60 | 0 |
| 2 | 64 × 64 | 256 | 58 | 700 |

在列分工中，每个 wave 覆盖四个 16 头子块，分母与概率 fragment 的逻辑长度均从 4 增至 16，V fragment 从 64 降至 16。相同的输出累加值数量因而可以对应不同的寄存器需求。两 split 时，列分工产生了更大的私有存储需求，且耗时显著增加。独立生成的汇编清单也出现寄存器溢出和重载；四 split 的列分工清单与计时二进制的 private memory 字段分别为 28 B 和 12 B。资源表以实际计时二进制为准。没有硬件计数器单独量化 spill 流量或耗时占比。[4-split 核查](results/audit.json) · [2-split 核查](results/default-two/audit.json)

**决定**　公开实现保留 16 × 256 分工及 B64/H128/1K 的默认两 split。列分工保存在实验目录。可迁移的 TileLang 经验是联合评估 PV 输出分工和每个 split 的 KV 循环长度：即便每个 wave 的输出元素数、每个 block 的共享内存都相同，两个调度参数也可能改变性能排序。结论限于本切片与形状。
