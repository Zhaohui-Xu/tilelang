# High-Performance MLA Decode Using TileLang on MetaX C500 GPUs

Building on TileLang’s MLA decode example for NVIDIA Hopper, we present an optimized implementation for MetaX C500 GPUs. We examine how the C500’s matrix instructions and shared memory capacity shape two design choices: how waves divide the output and how many query heads can share a KV tile. [Hopper implementation](../README.md#implementation)

## Architectural Considerations and Optimization Strategies

1. **Matrix instruction granularity.** The Hopper reference implementation uses a WGMMA path with a minimum M dimension of 64 (query heads in single-token decode) across a 128-thread warpgroup, whereas C500’s built-in FP16 MFMA operates on `16×16×16` tiles within a 64-lane wave. For a fixed output width, assigning more query heads to a warpgroup or wave increases the output accumulator and normalization state it holds, while narrow head blocks limit KV reuse. On C500, pairing two 64-lane waves across the output columns and adding wave pairs along query heads expands block-level KV reuse while keeping each wave’s output-tile dimensions and assigned head count fixed as the head tile widens.

2. **Shared memory allocation.** C500 provides a maximum of 64 KiB of shared memory per block (including opt-in), compared with H100’s 227 KiB per-block opt-in limit. Because MLA’s query heads share a single KV head, widening the head tile causes Q storage to grow with the head count while the KV tile size stays fixed. Staging both operands in shared memory therefore limits head-tile width and KV reuse. Keeping KV in shared memory while placing the combined Q tile in register fragments on wider schedules moves the growing query storage out of the per-block shared-memory budget, at the cost of register storage. [H100 resource limits](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#occupancy) · [Dense baseline](https://github.com/tile-ai/tilelang-metax/blob/90d49f353ec88bd0d346474f143003d865dd3b79/examples/deepseek_mla/example_mla_decode.py)

   Baseline

   ```python
   Q_shared = T.alloc_shared([block_H, dim], dtype)
   Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
   ```

   Query allocation for wider head tiles

   ```python
   alloc_query = T.alloc_shared if heads == 16 else T.alloc_fragment
   q_tile = alloc_query([heads, dim + pe_dim], dtype)
   ```

The schedule selects the head tile and KV split count together to balance intra-block KV reuse, block parallelism, and the cost of merging partial results.

## Performance Evaluation

Benchmarks use FP16 inputs and contiguous KV storage on a MetaX C500 instance configured with 32,000 MiB of GPU memory. The C500 default is compared with a tuned [Triton reference adapted for contiguous KV](benchmark_mla_decode_maca_triton.py) and a tuned [TileLang-MetaX dense baseline](https://github.com/tile-ai/tilelang-metax/blob/90d49f353ec88bd0d346474f143003d865dd3b79/examples/deepseek_mla/example_mla_decode.py). All results use full-call wall time.

**Large batches.** Across 10 shapes with 128 query heads, batches 64/128 and KV lengths 1K–16K, speedup is **5.49–9.50×** over the tested Triton adaptation and **1.79–2.88×** over dense. The figure shows effective throughput.

![Three-way large-batch MLA decode performance on MetaX C500](../figures/flashmla-maca.png)

**Small batches.** Across 12 tested shapes with 16 query heads, batches 1/2/4 and KV lengths spanning 2K–64K, speedup is **2.18–3.37×** over the tested Triton adaptation and **1.46–2.79×** over dense. Bars are normalized to dense = 1; higher is better.

![Three-way small-batch MLA decode performance on MetaX C500](../figures/flashmla-maca-small.png)

<details>
<summary>Measurement details and reproducibility</summary>

The stack was TileLang-MetaX `f549117c`, PyTorch `2.8.0+metax3.7.1.3`, and Triton `3.0.0` with the MetaX backend. Workloads use single-token decode, one KV head, 512 latent dimensions and 64 positional dimensions. Five FP32-reference checks per implementation and shape passed at `atol=rtol=0.002` (330 checks total).

Reported latency is the median of six batch-average timings covering all six implementation-order permutations. Each call includes output/workspace allocation, main kernels and merge, with synchronization at batch end. JIT compilation and configuration search are excluded. Warmups/timed calls per batch are 5/30 for large batches and 200/2,000 for small batches. Effective throughput counts `2 × B × H × N × (576 + 512)` operations per call.

The dense baseline uses previously screened 16-head, 32-token configurations, re-measured in this session. Triton underwent finite head/key-tile, wave-count and split screening, followed by split refinement for the selected layout. [Raw samples, selected configurations, versions and validation errors](benchmark_results.json) cover all 22 shapes.

The sGPU instance reported a 50% compute quota, best-effort scheduling and a 10 ms time slice.

</details>

<details>
<summary>How KV splitting affects the wave partition</summary>

A separate experiment used a 16,000 MiB C500 sGPU instance with a reported 25% compute quota. Batch 64, 128 heads, 1K tokens, a 64-head tile and 512 threads were fixed. Changing each wave's output tile from the default `16 heads × 256 columns` to `64 heads × 64 columns` reduced full-call latency from 651 to 620 µs at four splits, but increased it from 613 to 1,070 µs at the default two splits. The two-split variant also increased compiled main-kernel private memory from 0 to 700 bytes.

</details>

## Running the example

For hardware rental, see the [Moark compute marketplace](https://moark.com/compute). This example requires a MetaX C500 instance with a compatible MACA software stack.

This example uses [TileLang’s MetaX MACA ecosystem backend](../../../README.md#platform-and-backend-support), maintained in TileLang-MetaX (`tilelang.maca`, `target="maca"`). Use its MACA build and matching PyTorch-MACA (tested with TileLang-MetaX `f549117c` and PyTorch `2.8.0+metax3.7.1.3`); the standard TileLang release wheel does not include this backend. [Backend build instructions](https://github.com/tile-ai/tilelang-metax)

<details>
<summary>One-time path setup for a source build</summary>

Activate the MACA Python environment and replace the paths below with your installed SDK, built backend and baseline checkout:

```bash
export MACA_PATH=/path/to/maca
export PATH="$MACA_PATH/mxgpu_llvm/bin:$PATH"
export PYTHONPATH="/path/to/built/tilelang-metax${PYTHONPATH:+:$PYTHONPATH}"
export TILELANG_METAX_ROOT=/path/to/baseline/tilelang-metax
```

`PYTHONPATH` selects the built compiler backend; `TILELANG_METAX_ROOT` selects the reference implementation used for comparison. They can point to different checkouts. For the baseline checkout, use revision `90d49f353ec88bd0d346474f143003d865dd3b79`; it supplies Python source and does not need a separate build. Report mode verifies the baseline source hash.

</details>

From this directory, run the correctness smoke test with one command:

```bash
python benchmark_mla_decode_maca.py
```

The default smoke preset checks B1/H16 with 2,049 KV tokens, including a partial final tile, against the FP32 reference. It prints a `PASS` summary and JSON results. The first invocation includes compilation.

For the recorded B1/H16/2K three-way comparison, also use a matching MetaX Triton installation and set `TILELANG_METAX_ROOT` to the baseline checkout above:

```bash
python benchmark_mla_decode_maca.py --preset compare
```

This preset loads the dense and Triton configurations from `benchmark_results.json`, checks all three outputs, and reports full-call latency. An explicit `--baseline-file /path/to/example_mla_decode.py` overrides the environment variable.

To rerun all 22 report shapes on a 32 GB C500 instance, use the same dependencies and the pinned dense baseline linked above:

```bash
python benchmark_mla_decode_maca.py --preset report --output results/c500-report.json
```

Report mode reuses the published configurations without tuning. It generates FP16 inputs on the GPU with seed 9123, repeats the five recorded validation cases per implementation, and uses six implementation orders with the recorded warmup/iteration counts. It writes fresh JSON and CSV files, including the current environment and source hashes. Existing outputs are not overwritten; partial results are retained if a run fails or is interrupted. To check report mode on one shape, add `--report-shapes 1x16x2048`; subset runs cannot generate the full report charts.

Regenerate both charts from the published measurements without a GPU (requires Matplotlib):

```bash
python plot_mla_decode_maca.py --output-dir results/published-plots
```

Or plot a completed new report:

```bash
python plot_mla_decode_maca.py --input results/c500-report.json --output-dir results/new-plots
```

The plotter exports PNG/PDF figures, a summary CSV and speedup ranges. It rejects missing shapes, incomplete checks and invalid timing samples. The smoke, compare and custom-shape commands use the original CPU-generated inputs; report mode follows the published GPU-input protocol.

For custom shapes, omit the preset and pass the desired parameters:

```bash
python benchmark_mla_decode_maca.py --kv-length 8192 --baseline-splits 128 --seeds 100 \
  --baseline-file "$TILELANG_METAX_ROOT/examples/deepseek_mla/example_mla_decode.py"
```

For application use, compile with `mla_decode_maca(B, N, query_heads=H)` and call the returned kernel with `(q, q_pe, kv, k_pe, output)`. Pass contiguous FP16 tensors with shapes `[B,H,512]`, `[B,H,64]`, `[B,N,1,512]`, `[B,N,1,64]` and `[B,H,512]`, respectively. H must be divisible by 16 and N ranges from 1 to 65536. Call `.contiguous()` after device or layout conversion.

Use `--tune` to screen head tiles and split counts or `--tune-splits` to fix the head tile. The CLI reports the selected schedule, reference error and event and wall samples.
