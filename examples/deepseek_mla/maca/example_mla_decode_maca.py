"""Dense single-token MLA decode for the MetaX C500."""
from functools import lru_cache
import math

import tilelang
import tilelang.language as T
from tilelang.maca.intrinsics import TensorCoreIntrinEmitter


@lru_cache(maxsize=None)
@tilelang.jit(target="maca", pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def mla_decode_maca(batch, kv_ctx):
    heads, kv_heads, dim, pe_dim = 16, 1, 512, 64
    assert batch > 0
    assert 1 <= kv_ctx <= 65536
    dtype, accum_dtype = T.float16, T.float32
    block_n, num_split, wave_size = 32, 64, 64
    main_threads, merge_threads = 128, 256
    num_tiles = (kv_ctx + block_n - 1) // block_n
    base_tiles, extra = divmod(num_tiles, num_split)
    max_tiles = base_tiles + int(extra > 0)
    group_tiles = min(max_tiles, 8)
    stats_offset = batch * num_split * heads * dim
    max_offset = stats_offset + batch * num_split * heads
    scale = math.log2(math.e) / math.sqrt(dim + pe_dim)

    pv_mma = TensorCoreIntrinEmitter(
        a_dtype=dtype, b_dtype=dtype, accum_dtype=accum_dtype,
        a_transposed=False, b_transposed=False,
        block_row_warps=1, block_col_warps=2,
        warp_row_tiles=heads, warp_col_tiles=dim // 2, chunk=block_n,
    )
    _, pv_b_reverse = pv_mma.get_ldmatrix_index_map(is_b=True)

    @T.prim_func
    def main(
        q: T.Tensor([batch, heads, dim], dtype),
        q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        kv: T.Tensor([batch, kv_ctx, kv_heads, dim], dtype),
        k_pe: T.Tensor([batch, kv_ctx, kv_heads, pe_dim], dtype),
        output: T.Tensor([batch, heads, dim], dtype),
    ):
        workspace = T.alloc_global([batch * num_split * heads * (dim + 2)], accum_dtype)

        with T.Kernel(num_split, batch, threads=main_threads) as (sid, bid):
            qk_shared = T.alloc_shared([heads, dim + pe_dim], dtype)
            key_shared = T.alloc_shared([block_n, dim + pe_dim], dtype)
            prob_shared = T.alloc_shared([heads, block_n], dtype)
            T.annotate_layout(
                {
                    qk_shared: tilelang.layout.make_swizzled_layout(qk_shared),
                    key_shared: tilelang.layout.make_swizzled_layout(key_shared),
                    prob_shared: tilelang.layout.make_swizzled_layout(prob_shared),
                }
            )
            score = T.alloc_fragment([heads, block_n], accum_dtype)
            running_max = T.alloc_fragment([heads], accum_dtype)
            previous_max = T.alloc_fragment([heads], accum_dtype)
            rescale = T.alloc_shared([heads], accum_dtype)
            output_acc = T.alloc_local([64], accum_dtype)
            pv_a = T.alloc_local([4], dtype)
            pv_b = T.alloc_local([64], dtype)
            ones_b = T.alloc_local([4], dtype)
            denom_acc = T.alloc_local([4], accum_dtype)
            tx_main = T.get_thread_binding()
            lane_main = tx_main % 64
            wave_main = tx_main // 64

            T.copy(q[bid, :, :], qk_shared[:, :dim])
            T.copy(q_pe[bid, :, :], qk_shared[:, dim : dim + pe_dim])
            T.fill(output_acc, 0)
            T.fill(denom_acc, 0)
            T.fill(ones_b, 1.0)
            T.fill(running_max, -T.infinity(accum_dtype))

            # Distribute whole tiles; idle splits retain (-inf, 0, 0) for the merge.
            first_tile = sid * base_tiles + T.min(sid, extra)
            for group in T.serial((max_tiles + group_tiles - 1) // group_tiles):
                for tile in T.unroll(group_tiles):
                    local_tile = group * group_tiles + tile
                    if local_tile < base_tiles or (local_tile == base_tiles and sid < extra):
                        kv_start = (first_tile + local_tile) * block_n
                        if kv_ctx % block_n and kv_start + block_n > kv_ctx:
                            for n, k in T.Parallel(block_n, dim):
                                key_shared[n, k] = T.if_then_else(kv_start + n < kv_ctx, kv[bid, kv_start + n, 0, k], 0)
                            for n, k in T.Parallel(block_n, pe_dim):
                                key_shared[n, dim + k] = T.if_then_else(kv_start + n < kv_ctx, k_pe[bid, kv_start + n, 0, k], 0)
                        else:
                            T.copy(kv[bid, kv_start : kv_start + block_n, 0, :], key_shared[:, :dim])
                            T.copy(k_pe[bid, kv_start : kv_start + block_n, 0, :], key_shared[:, dim : dim + pe_dim])
                        T.gemm(
                            qk_shared, key_shared, score, transpose_B=True,
                            policy=T.GemmWarpPolicy.FullCol, clear_accum=True,
                        )
                        if kv_ctx % block_n:
                            for h, n in T.Parallel(heads, block_n):
                                if kv_start + n >= kv_ctx:
                                    score[h, n] = -T.infinity(accum_dtype)
                        if max_tiles > 1:
                            T.copy(running_max, previous_max)
                        T.reduce_max(score, running_max, dim=1, clear=True)
                        if max_tiles > 1:
                            for h in T.Parallel(heads):
                                running_max[h] = T.max(running_max[h], previous_max[h])
                                rescale[h] = T.exp2((previous_max[h] - running_max[h]) * scale)
                            T.sync_threads()
                            factor = rescale[lane_main % heads]
                            for i in T.vectorized(4):
                                denom_acc[i] *= factor
                            for j in T.serial(16):
                                for i in T.vectorized(4):
                                    output_acc[j * 4 + i] *= factor
                        for h, n in T.Parallel(heads, block_n):
                            score[h, n] = T.exp2((score[h, n] - running_max[h]) * scale)
                        T.copy(score, prob_shared)
                        T.sync_threads()
                        for ki in T.serial(2):
                            pv_mma.ldmatrix_a(pv_a, prob_shared, ki)
                            for j in T.serial(16):
                                for local_id in T.vectorized(4):
                                    row, col = T.meta_var(pv_b_reverse(lane_main, local_id))
                                    pv_b[j * 4 + local_id] = key_shared[
                                        ki * 16 + row, wave_main * 256 + j * 16 + col
                                    ]
                            for j in T.serial(16):
                                T.tvm_mfma(
                                    pv_mma.mma_suffix, "row", "row", "float16x4", "float16x4", "float32x4",
                                    pv_b.data, j, pv_a.data, 0, output_acc.data, j, dtype="float32x4",
                                )
                            # Reuse the probability fragment to accumulate the denominator.
                            T.tvm_mfma(
                                pv_mma.mma_suffix, "row", "row", "float16x4", "float16x4", "float32x4",
                                ones_b.data, 0, pv_a.data, 0, denom_acc.data, 0, dtype="float32x4",
                            )
            for j in T.serial(16):
                for local_id in T.vectorized(4):
                    row = lane_main % 16
                    col = wave_main * 256 + j * 16 + (lane_main // 16) * 4 + local_id
                    workspace[((bid * num_split + sid) * heads + row) * dim + col] = output_acc[j * 4 + local_id]
            if tx_main < heads:
                workspace[stats_offset + (bid * num_split + sid) * heads + tx_main] = denom_acc[0]
            for h in T.Parallel(heads):
                workspace[max_offset + (bid * num_split + sid) * heads + h] = running_max[h]

        with T.Kernel(heads, batch, threads=merge_threads) as (merge_head, merge_batch):
            tx = T.get_thread_binding()
            lane, wave = tx % wave_size, tx // wave_size
            weights = T.alloc_shared((merge_threads // wave_size, num_split), accum_dtype)
            acc = T.alloc_local((8, 2), accum_dtype)
            result = T.alloc_local((2,), accum_dtype)
            global_max = T.alloc_var(accum_dtype)
            total = T.alloc_var(accum_dtype)
            peer = T.alloc_var(accum_dtype)
            weight = T.alloc_var(accum_dtype)

            # Each wave computes its own weights and owns disjoint output columns.
            global_max = workspace[max_offset + (merge_batch * num_split + lane) * heads + merge_head]
            for step in T.unroll(6):
                distance = 32 >> step
                peer = T.shfl_down(global_max, distance, wave_size)
                if lane < distance:
                    global_max = T.max(global_max, peer)
            global_max = T.shfl_sync(global_max, 0, wave_size)
            weight = T.exp2((workspace[max_offset + (merge_batch * num_split + lane) * heads + merge_head] - global_max) * scale)
            weights[wave, lane] = weight

            total = weight * workspace[stats_offset + (merge_batch * num_split + lane) * heads + merge_head]
            for step in T.unroll(6):
                distance = 32 >> step
                peer = T.shfl_down(total, distance, wave_size)
                if lane < distance:
                    total += peer
            total = T.shfl_sync(total, 0, wave_size)
            T.sync_warp()

            T.fill(acc, 0)
            for group in T.unroll(num_split // 8):
                for chain in T.unroll(8):
                    split = group * 8 + chain
                    for vec in T.vectorized(2):
                        acc[chain, vec] += workspace[((merge_batch * num_split + split) * heads + merge_head) * dim + tx * 2 + vec] * weights[wave, split]
            T.fill(result, 0)
            for chain in T.unroll(8):
                for vec in T.vectorized(2):
                    result[vec] += acc[chain, vec]
            for vec in T.vectorized(2):
                output[merge_batch, merge_head, tx * 2 + vec] = result[vec] / total

    return main
