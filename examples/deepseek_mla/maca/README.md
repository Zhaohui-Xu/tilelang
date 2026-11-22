# High-Performance MLA Decode Using TileLang on MetaX C500 GPUs

Building on TileLang’s MLA decode example for NVIDIA Hopper, we present an optimized implementation for MetaX C500 GPUs. We examine how the C500’s matrix instructions and shared memory capacity shape two design choices: how waves divide the output and how many query heads can share a KV tile. [Hopper implementation](../README.md#implementation)

## Architectural considerations and optimization strategies

**Partition the accumulator across waves.** The Hopper example uses a WGMMA path with a minimum of 64 rows per instruction. In our C500 value product, two 64-lane waves each accumulate 256 output columns for the same 16 heads through a sequence of MFMA instructions. Four such pairs can process 64 heads in one block, all reading the same shared KV tile.

**Choose where Q lives.** A 32-token FP16 key tile occupies 36 KiB at width 576. Widening the head tile from 16 to 64 lets four times as many heads reuse that key tile, but also grows Q from 18 to 72 KiB. Staging both wide tiles in shared memory would require 108 KiB before the softmax buffers. The kernel selects Q storage by head tile size.

```python
alloc_query = T.alloc_shared if heads == 16 else T.alloc_fragment
```

The wider schedule keeps Q in a fragment. Each wave also reuses its softmax probability fragment for both the value product and the denominator. [Kernel source](example_mla_decode_maca.py)

**Balance KV reuse against parallelism.** A wider head tile leaves fewer independent head groups. Splitting the KV sequence restores parallel work, but each additional split writes FP32 partials for a merge kernel. At batch 64, 128 heads and 1K tokens, the 16-head schedule with eight splits took **1248 µs** per complete call. The 64-head schedule with four splits took **648 µs** and halved the partial workspace from 128.5 to 64.25 MiB. The default schedule uses the number of independent head groups and key tiles to choose the head width and split count together.

## Performance evaluation

We measured full-call wall latency on a **50% C500 slice (32000 MiB)**. Each call includes output allocation and all kernel launches after compilation. The tables give median latency, and the figures show the corresponding throughput or speedup over the tuned baseline. [TileLang-MetaX baseline](https://github.com/tile-ai/tilelang-metax/blob/90d49f353ec88bd0d346474f143003d865dd3b79/examples/deepseek_mla/example_mla_decode.py).

**Large batches.** At batch sizes 64 and 128 with 128 query heads, the default C500 kernel achieved **1.80×–2.88×** speedup over the tuned dense baseline. These shapes follow the AMD example. Our inputs use FP16 and all sequences in a batch have the same KV length. [AMD example](../amd)

![Effective throughput on C500 at batch 64 and 128](../figures/flashmla-maca.png)

Throughput counts `2 × B × H × N × (576 + 512)` operations for the score and value products. Each implementation ran six interleaved batches with five warmups and 30 measured calls per batch.

<details>
<summary>Per-shape wall latency and configuration</summary>

The dense baseline used a 16-head, 32-token tile and one split at all ten shapes after screening head tiles, key tiles and split counts.

| Batch |    KV | Default head tile / splits | C500 wall (µs) | Dense baseline wall (µs) | Dense speedup |
| ----: | ----: | -------------------------: | --------------: | ------------------------: | ------------: |
|    64 |  1024 |                     64 / 2 |          610.99 |                   1098.53 |        1.80× |
|    64 |  2048 |                     64 / 4 |         1033.73 |                   2150.67 |        2.08× |
|    64 |  4096 |                     64 / 4 |         1771.24 |                   4248.83 |        2.40× |
|    64 |  8192 |                     64 / 4 |         3240.63 |                   8436.91 |        2.60× |
|    64 | 16384 |                     64 / 4 |         6187.32 |                  16849.09 |        2.72× |
|   128 |  1024 |                     64 / 1 |         1020.29 |                   2187.06 |        2.14× |
|   128 |  2048 |                     64 / 2 |         1735.53 |                   4285.20 |        2.47× |
|   128 |  4096 |                     64 / 2 |         3144.87 |                   8474.82 |        2.69× |
|   128 |  8192 |                     64 / 2 |         5985.78 |                  16862.73 |        2.82× |
|   128 | 16384 |                     64 / 2 |        11671.82 |                  33638.07 |        2.88× |

</details>

**Small batches.** Across 12 shapes with 16 query heads, wall speedup was **1.43×–2.79×**. At batch 1 and 8K tokens, latency fell from **92.24 to 45.10 µs**. Each marker uses six alternating batches of 200 warmups and 2000 measured calls.

![Small-batch wall speedup over the tuned dense baseline](../figures/flashmla-maca-small.png)

<details>
<summary>Per-shape wall latency and configuration</summary>

| Batch |    KV | C500 splits | Baseline splits | C500 wall (µs) | Baseline wall (µs) | Speedup |
| ----: | ----: | ----------: | --------------: | --------------: | ------------------: | ------: |
|     1 |  2048 |          64 |              64 |           25.01 |               35.84 |  1.43× |
|     1 |  4096 |          64 |              64 |           31.42 |               48.12 |  1.53× |
|     1 |  8192 |          64 |             128 |           45.10 |               92.24 |  2.05× |
|     1 | 16384 |          64 |             256 |           78.75 |              157.47 |  2.00× |
|     1 | 32768 |         256 |             512 |          123.80 |              284.62 |  2.30× |
|     1 | 65536 |         512 |            1024 |          201.91 |              563.88 |  2.79× |
|     2 |  2048 |          32 |              32 |           30.00 |               44.42 |  1.48× |
|     2 |  8192 |          32 |             128 |           77.79 |              131.23 |  1.69× |
|     2 | 32768 |         256 |             512 |          188.08 |              441.83 |  2.35× |
|     4 |  2048 |          16 |              32 |           42.75 |               74.00 |  1.73× |
|     4 |  8192 |          64 |             128 |          112.94 |              192.74 |  1.71× |
|     4 | 32768 |         256 |             512 |          339.61 |              739.37 |  2.18× |

</details>

<details>
<summary>How KV splitting affects the wave partition</summary>

On a separate 25% C500 slice, we held batch 64, 128 heads, 1K tokens, a 64-head tile and 512 threads per block fixed. With four KV splits, changing each wave's output from 16 heads × 256 columns to 64 heads × 64 columns reduced full-call wall latency from 651 to 620 µs. At the default two splits, the same change increased latency from 613 to 1070 µs. The compiled main kernel's private memory rose from 0 to 700 bytes in the two-split comparison.

</details>

## Running the example

Use a TileLang-MetaX MACA build and matching PyTorch-MACA. We tested `f549117c` with PyTorch `2.8.0+metax3.7.1.3`. For a source build, put the backend checkout on `PYTHONPATH` and expose the MACA SDK. Set `TILELANG_METAX_ROOT` to the baseline checkout linked above. [Backend repository](https://github.com/tile-ai/tilelang-metax)

Compile with `mla_decode_maca(B, N, query_heads=H)`. Pass contiguous FP16 tensors for Q, positional Q, KV, positional K and output with shapes `[B,H,512]`, `[B,H,64]`, `[B,N,1,512]`, `[B,N,1,64]` and `[B,H,512]`. H must be divisible by 16 and N ranges from 1 to 65536. Call `.contiguous()` after device or layout conversion.

Check a partial KV tile against the FP32 reference, then compare complete calls at an aligned length. The 18-shape validation matrix passed 198 full-output checks at `atol=rtol=0.002`.

```bash
python benchmark_mla_decode_maca.py --kv-length 2049 --batch-size 2 --check-only
```

```bash
python benchmark_mla_decode_maca.py --kv-length 8192 --baseline-splits 128 --seeds 100 \
  --baseline-file "$TILELANG_METAX_ROOT/examples/deepseek_mla/example_mla_decode.py"
```

Use `--tune` to screen head tiles and split counts or `--tune-splits` to fix the head tile. The CLI reports the selected schedule, reference error and event and wall samples.
