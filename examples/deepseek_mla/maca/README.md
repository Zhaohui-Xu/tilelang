# Dense MLA decode on MetaX C500

This example targets single-token decode with contiguous FP16 inputs, 16 query heads, one KV head, a 512-column latent vector and 64 positional columns. Compile with `mla_decode_maca(batch, kv_length)`, then call the result with Q, positional Q, KV, positional K and a preallocated FP16 output shaped like Q.

The main kernel distributes 32-token tiles over 64 splits. It packs latent and positional columns for the score GEMM and reuses the probability fragment for the value product and denominator. A partial tile uses guarded loads and masks padded logits before softmax. Empty splits contribute zero. The merge combines FP32 partial outputs using each split's maximum and sum.

The parameterized source targets lengths 1 through 65536. On a 50% C500 slice, full-output checks passed at batch 1 for lengths 1, 31, 32, 33, 2047, 2048, 2049, 6144, 8193, 12288, 65535 and 65536, and at batches 2 and 4 for lengths 33, 2049 and 8192. Each combination used eight random seeds, first- and last-token peaks and a large-value input, with FP32 reference tolerances of 0.002 (absolute and relative).

Use a TileLang-MetaX build with MACA support and matching PyTorch-MACA. From this directory, check an uneven length against the FP32 reference:

```bash
python benchmark_mla_decode_maca.py --kv-length 2049 --batch-size 2 --check-only
```

For an aligned comparison, point the harness at the reference checkout. The split baseline requires whole 32-token tiles per split. The harness checks random inputs, first- and last-token peaks and large values before measuring complete calls in alternating order with device events and host wall time.

```bash
python benchmark_mla_decode_maca.py --kv-length 8192 --baseline-splits 128 --seeds 100 \
  --baseline-file "$TILELANG_METAX_ROOT/examples/deepseek_mla/example_mla_decode.py"
```

### Batch-1 performance on C500

Full-call median latency on a 50% compute slice (32000 MiB), with PyTorch 2.8.0+metax3.7.1.3 and TileLang-MetaX f549117c. The baseline is the [TileLang-MetaX dense MLA example](https://github.com/tile-ai/tilelang-metax/blob/90d49f353ec88bd0d346474f143003d865dd3b79/examples/deepseek_mla/example_mla_decode.py). For its 32-key, 16-head configuration we used the fastest measured power-of-two split count per length. Each pair used six alternating batches with 200 warmups and 2000 measured calls, including output allocation and both kernel launches.

| KV length | Baseline splits | Baseline event (µs) | C500 event (µs) | Event speedup | Wall speedup |
|---:|---:|---:|---:|---:|---:|
| 2048 | 64 | 35.52 | 23.69 | 1.50× | 1.43× |
| 4096 | 64 | 46.84 | 30.18 | 1.55× | 1.53× |
| 8192 | 128 | 90.76 | 43.93 | 2.07× | 2.03× |
| 16384 | 256 | 154.91 | 77.47 | 2.00× | 1.98× |
| 32768 | 512 | 284.50 | 172.28 | 1.65× | 1.65× |
| 65536 | 1024 | 563.07 | 288.89 | 1.95× | 1.95× |
