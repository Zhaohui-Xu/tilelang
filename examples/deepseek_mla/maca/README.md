# Dense MLA decode on MetaX C500

This example targets single-token decode with contiguous FP16 inputs, 16 query heads, one KV head, a 512-column latent vector and 64 positional columns. Compile with `mla_decode_maca(batch, kv_length)`, then call the result with Q, positional Q, KV, positional K and a preallocated FP16 output shaped like Q.

The main kernel distributes 32-token tiles over 64 splits. It packs latent and positional columns for the score GEMM and reuses the probability fragment for the value product and denominator. A partial tile uses guarded loads and masks padded logits before softmax. Empty splits contribute zero. The merge combines FP32 partial outputs using each split's maximum and sum.

The parameterized source targets lengths 1 through 65536. Device validation of this revision is pending. The earlier aligned implementation was tested at batch 1 on six power-of-two lengths from 2048 through 65536.

Use a TileLang-MetaX build with MACA support and matching PyTorch-MACA. From this directory, check an uneven length against the FP32 reference:

```bash
python benchmark_mla_decode_maca.py --kv-length 2049 --batch-size 2 --check-only
```

For an aligned comparison, point the harness at the reference checkout. The split baseline requires whole 32-token tiles per split. The harness checks random inputs, first- and last-token peaks and large values before measuring complete calls in alternating order with device events and host wall time.

```bash
python benchmark_mla_decode_maca.py --kv-length 8192 --baseline-splits 128 --seeds 100 \
  --baseline-file "$TILELANG_METAX_ROOT/examples/deepseek_mla/example_mla_decode.py"
```
