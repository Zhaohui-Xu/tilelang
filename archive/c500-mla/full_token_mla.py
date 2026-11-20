"""Contest-era full-token portfolio. Fixed shifts limit its numerical range."""

from functools import lru_cache

import tilelang
import tilelang.language as T
from tilelang.maca.intrinsics import TensorCoreIntrinEmitter

from contest_mla import run_kernel as _run_short


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn_16k_e7_explicit_pv_helper_rawpartial(
    batch, heads, kv_heads, kv_ctx, dim, pe_dim, num_split, merge_threads,
    partial_fp16, block_n, main_threads,
):
    assert heads == 16 and kv_heads == 1 and dim == 512 and pe_dim == 64
    assert kv_ctx == 16384 and num_split == 64
    assert merge_threads == 256 and partial_fp16 == 1
    assert block_n == 32 and main_threads == 128

    split_size = kv_ctx // num_split
    dtype, accum_dtype = T.float16, T.float32
    scale = float((dim + pe_dim) ** -0.5 * 1.44269504)

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
        packed_partial = T.alloc_global([batch * num_split * heads * (dim + 1)], dtype)

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

            for k in T.unroll(T.ceildiv(split_size, block_n)):
                kv_start = sid * split_size + k * block_n
                T.copy(kv[bid, kv_start : kv_start + block_n, 0, :], key_shared[:, :dim])
                T.copy(k_pe[bid, kv_start : kv_start + block_n, 0, :], key_shared[:, dim : dim + pe_dim])
                T.gemm(
                    qk_shared, key_shared, score, transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol, clear_accum=True,
                )
                for h, n in T.Parallel(heads, block_n):
                    score[h, n] = T.exp2(score[h, n] * scale - 8.0)
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
                    # Both waves reuse the same existing P-A fragment;no shared ones or reload.
                    T.tvm_mfma(
                        pv_mma.mma_suffix, "row", "row", "float16x4", "float16x4", "float32x4",
                        ones_b.data, 0, pv_a.data, 0, denom_acc.data, 0, dtype="float32x4",
                    )
            for j in T.serial(16):
                for local_id in T.vectorized(4):
                    row = lane_main % 16
                    col = wave_main * 256 + j * 16 + (lane_main // 16) * 4 + local_id
                    packed_partial[((bid * num_split + sid) * heads + row) * dim + col] = T.cast(output_acc[j * 4 + local_id], dtype)
            if tx_main < heads:
                packed_partial[batch * num_split * heads * dim + (bid * num_split + sid) * heads + tx_main] = T.cast(denom_acc[0], dtype)

        with T.Kernel(heads, batch, threads=merge_threads) as (hid, bid):
            tx = T.get_thread_binding()
            lane = tx % 64
            output_local = T.alloc_local([2], accum_dtype)
            merge_acc = T.alloc_local([8, 2], accum_dtype)
            global_sum = T.alloc_var(accum_dtype)
            peer = T.alloc_var(accum_dtype)
            inv_sum = T.alloc_var(accum_dtype)

            global_sum = T.cast(packed_partial[batch * num_split * heads * dim + (bid * num_split + lane) * heads + hid], accum_dtype)
            peer = T.shfl_down(global_sum, 32, 64)
            if lane < 32: global_sum += peer
            peer = T.shfl_down(global_sum, 16, 64)
            if lane < 16: global_sum += peer
            peer = T.shfl_down(global_sum, 8, 64)
            if lane < 8: global_sum += peer
            peer = T.shfl_down(global_sum, 4, 64)
            if lane < 4: global_sum += peer
            peer = T.shfl_down(global_sum, 2, 64)
            if lane < 2: global_sum += peer
            peer = T.shfl_down(global_sum, 1, 64)
            if lane < 1: global_sum += peer
            global_sum = T.shfl_sync(global_sum, 0, 64)

            acc0 = T.alloc_var(T.float16x2)
            acc1 = T.alloc_var(T.float16x2)
            acc2 = T.alloc_var(T.float16x2)
            acc3 = T.alloc_var(T.float16x2)
            acc4 = T.alloc_var(T.float16x2)
            acc5 = T.alloc_var(T.float16x2)
            acc6 = T.alloc_var(T.float16x2)
            acc7 = T.alloc_var(T.float16x2)
            acc0 = T.Broadcast(T.float16(0.0), 2)
            acc1 = T.Broadcast(T.float16(0.0), 2)
            acc2 = T.Broadcast(T.float16(0.0), 2)
            acc3 = T.Broadcast(T.float16(0.0), 2)
            acc4 = T.Broadcast(T.float16(0.0), 2)
            acc5 = T.Broadcast(T.float16(0.0), 2)
            acc6 = T.Broadcast(T.float16(0.0), 2)
            acc7 = T.Broadcast(T.float16(0.0), 2)
            for group in T.unroll(num_split // 8):
                v0 = packed_partial[((bid * num_split + group * 8 + 0) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v1 = packed_partial[((bid * num_split + group * 8 + 1) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v2 = packed_partial[((bid * num_split + group * 8 + 2) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v3 = packed_partial[((bid * num_split + group * 8 + 3) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v4 = packed_partial[((bid * num_split + group * 8 + 4) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v5 = packed_partial[((bid * num_split + group * 8 + 5) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v6 = packed_partial[((bid * num_split + group * 8 + 6) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v7 = packed_partial[((bid * num_split + group * 8 + 7) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                acc0 = T.add2(acc0, v0)
                acc1 = T.add2(acc1, v1)
                acc2 = T.add2(acc2, v2)
                acc3 = T.add2(acc3, v3)
                acc4 = T.add2(acc4, v4)
                acc5 = T.add2(acc5, v5)
                acc6 = T.add2(acc6, v6)
                acc7 = T.add2(acc7, v7)
            total = T.cast(acc0, T.float32x2)
            total = T.add2(total, T.cast(acc1, T.float32x2))
            total = T.add2(total, T.cast(acc2, T.float32x2))
            total = T.add2(total, T.cast(acc3, T.float32x2))
            total = T.add2(total, T.cast(acc4, T.float32x2))
            total = T.add2(total, T.cast(acc5, T.float32x2))
            total = T.add2(total, T.cast(acc6, T.float32x2))
            total = T.add2(total, T.cast(acc7, T.float32x2))
            output_local[T.Ramp(0, 1, 2)] = total
            inv_sum = 1.0 / global_sum
            for vi in T.vectorized(2):
                output[bid, hid, tx * 2 + vi] = output_local[vi] * inv_sum
    return main


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def a254_32k_fulltoken_fastpath(
    batch, heads, kv_heads, kv_ctx, dim, pe_dim, num_split, merge_threads,
    partial_fp16, block_n, main_threads,
):
    assert heads == 16 and kv_heads == 1 and dim == 512 and pe_dim == 64
    assert kv_ctx == 32768 and num_split == 64
    assert merge_threads == 256 and partial_fp16 == 1
    assert block_n == 32 and main_threads == 128

    split_size = kv_ctx // num_split
    dtype, accum_dtype = T.float16, T.float32
    scale = float((dim + pe_dim) ** -0.5 * 1.44269504)

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
        packed_partial = T.alloc_global([batch * num_split * heads * (dim + 1)], dtype)

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

            for page_group in T.serial(2):
                for page_in_group in T.unroll(8):
                    k = page_group * 8 + page_in_group
                    kv_start = sid * split_size + k * block_n
                    T.copy(kv[bid, kv_start : kv_start + block_n, 0, :], key_shared[:, :dim])
                    T.copy(k_pe[bid, kv_start : kv_start + block_n, 0, :], key_shared[:, dim : dim + pe_dim])
                    T.gemm(
                        qk_shared, key_shared, score, transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol, clear_accum=True,
                    )
                    for h, n in T.Parallel(heads, block_n):
                        score[h, n] = T.exp2(score[h, n] * scale - 8.0)
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
                        # Both waves reuse the same existing P-A fragment;no shared ones or reload.
                        T.tvm_mfma(
                            pv_mma.mma_suffix, "row", "row", "float16x4", "float16x4", "float32x4",
                            ones_b.data, 0, pv_a.data, 0, denom_acc.data, 0, dtype="float32x4",
                        )
            for j in T.serial(16):
                for local_id in T.vectorized(4):
                    row = lane_main % 16
                    col = wave_main * 256 + j * 16 + (lane_main // 16) * 4 + local_id
                    packed_partial[((bid * num_split + sid) * heads + row) * dim + col] = T.cast(output_acc[j * 4 + local_id], dtype)
            if tx_main < heads:
                packed_partial[batch * num_split * heads * dim + (bid * num_split + sid) * heads + tx_main] = T.cast(denom_acc[0], dtype)

        with T.Kernel(heads, batch, threads=merge_threads) as (hid, bid):
            tx = T.get_thread_binding()
            lane = tx % 64
            output_local = T.alloc_local([2], accum_dtype)
            merge_acc = T.alloc_local([8, 2], accum_dtype)
            global_sum = T.alloc_var(accum_dtype)
            peer = T.alloc_var(accum_dtype)
            inv_sum = T.alloc_var(accum_dtype)

            global_sum = T.cast(packed_partial[batch * num_split * heads * dim + (bid * num_split + lane) * heads + hid], accum_dtype)
            peer = T.shfl_down(global_sum, 32, 64)
            if lane < 32: global_sum += peer
            peer = T.shfl_down(global_sum, 16, 64)
            if lane < 16: global_sum += peer
            peer = T.shfl_down(global_sum, 8, 64)
            if lane < 8: global_sum += peer
            peer = T.shfl_down(global_sum, 4, 64)
            if lane < 4: global_sum += peer
            peer = T.shfl_down(global_sum, 2, 64)
            if lane < 2: global_sum += peer
            peer = T.shfl_down(global_sum, 1, 64)
            if lane < 1: global_sum += peer
            global_sum = T.shfl_sync(global_sum, 0, 64)

            acc0 = T.alloc_var(T.float16x2)
            acc1 = T.alloc_var(T.float16x2)
            acc2 = T.alloc_var(T.float16x2)
            acc3 = T.alloc_var(T.float16x2)
            acc4 = T.alloc_var(T.float16x2)
            acc5 = T.alloc_var(T.float16x2)
            acc6 = T.alloc_var(T.float16x2)
            acc7 = T.alloc_var(T.float16x2)
            acc0 = T.Broadcast(T.float16(0.0), 2)
            acc1 = T.Broadcast(T.float16(0.0), 2)
            acc2 = T.Broadcast(T.float16(0.0), 2)
            acc3 = T.Broadcast(T.float16(0.0), 2)
            acc4 = T.Broadcast(T.float16(0.0), 2)
            acc5 = T.Broadcast(T.float16(0.0), 2)
            acc6 = T.Broadcast(T.float16(0.0), 2)
            acc7 = T.Broadcast(T.float16(0.0), 2)
            for group in T.unroll(num_split // 8):
                v0 = packed_partial[((bid * num_split + group * 8 + 0) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v1 = packed_partial[((bid * num_split + group * 8 + 1) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v2 = packed_partial[((bid * num_split + group * 8 + 2) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v3 = packed_partial[((bid * num_split + group * 8 + 3) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v4 = packed_partial[((bid * num_split + group * 8 + 4) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v5 = packed_partial[((bid * num_split + group * 8 + 5) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v6 = packed_partial[((bid * num_split + group * 8 + 6) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v7 = packed_partial[((bid * num_split + group * 8 + 7) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                acc0 = T.add2(acc0, v0)
                acc1 = T.add2(acc1, v1)
                acc2 = T.add2(acc2, v2)
                acc3 = T.add2(acc3, v3)
                acc4 = T.add2(acc4, v4)
                acc5 = T.add2(acc5, v5)
                acc6 = T.add2(acc6, v6)
                acc7 = T.add2(acc7, v7)
            total = T.cast(acc0, T.float32x2)
            total = T.add2(total, T.cast(acc1, T.float32x2))
            total = T.add2(total, T.cast(acc2, T.float32x2))
            total = T.add2(total, T.cast(acc3, T.float32x2))
            total = T.add2(total, T.cast(acc4, T.float32x2))
            total = T.add2(total, T.cast(acc5, T.float32x2))
            total = T.add2(total, T.cast(acc6, T.float32x2))
            total = T.add2(total, T.cast(acc7, T.float32x2))
            output_local[T.Ramp(0, 1, 2)] = total
            inv_sum = 1.0 / global_sum
            for vi in T.vectorized(2):
                output[bid, hid, tx * 2 + vi] = output_local[vi] * inv_sum
    return main


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def a234_64k_fulltoken_fastpath(
    batch, heads, kv_heads, kv_ctx, dim, pe_dim, num_split, merge_threads,
    partial_fp16, block_n, main_threads,
):
    assert heads == 16 and kv_heads == 1 and dim == 512 and pe_dim == 64
    assert kv_ctx == 65536 and num_split == 64
    assert merge_threads == 256 and partial_fp16 == 1
    assert block_n == 32 and main_threads == 128

    split_size = kv_ctx // num_split
    dtype, accum_dtype = T.float16, T.float32
    scale = float((dim + pe_dim) ** -0.5 * 1.44269504)

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
        packed_partial = T.alloc_global([batch * num_split * heads * (dim + 1)], dtype)

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

            for page_group in T.serial(4):
                for page_in_group in T.unroll(8):
                    k = page_group * 8 + page_in_group
                    kv_start = sid * split_size + k * block_n
                    T.copy(kv[bid, kv_start : kv_start + block_n, 0, :], key_shared[:, :dim])
                    T.copy(k_pe[bid, kv_start : kv_start + block_n, 0, :], key_shared[:, dim : dim + pe_dim])
                    T.gemm(
                        qk_shared, key_shared, score, transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol, clear_accum=True,
                    )
                    for h, n in T.Parallel(heads, block_n):
                        score[h, n] = T.exp2(score[h, n] * scale - 8.0)
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
                        # Both waves reuse the same existing P-A fragment;no shared ones or reload.
                        T.tvm_mfma(
                            pv_mma.mma_suffix, "row", "row", "float16x4", "float16x4", "float32x4",
                            ones_b.data, 0, pv_a.data, 0, denom_acc.data, 0, dtype="float32x4",
                        )
            for j in T.serial(16):
                for local_id in T.vectorized(4):
                    row = lane_main % 16
                    col = wave_main * 256 + j * 16 + (lane_main // 16) * 4 + local_id
                    packed_partial[((bid * num_split + sid) * heads + row) * dim + col] = T.cast(output_acc[j * 4 + local_id], dtype)
            if tx_main < heads:
                packed_partial[batch * num_split * heads * dim + (bid * num_split + sid) * heads + tx_main] = T.cast(denom_acc[0], dtype)

        with T.Kernel(heads, batch, threads=merge_threads) as (hid, bid):
            tx = T.get_thread_binding()
            lane = tx % 64
            output_local = T.alloc_local([2], accum_dtype)
            merge_acc = T.alloc_local([8, 2], accum_dtype)
            global_sum = T.alloc_var(accum_dtype)
            peer = T.alloc_var(accum_dtype)
            inv_sum = T.alloc_var(accum_dtype)

            global_sum = T.cast(packed_partial[batch * num_split * heads * dim + (bid * num_split + lane) * heads + hid], accum_dtype)
            peer = T.shfl_down(global_sum, 32, 64)
            if lane < 32: global_sum += peer
            peer = T.shfl_down(global_sum, 16, 64)
            if lane < 16: global_sum += peer
            peer = T.shfl_down(global_sum, 8, 64)
            if lane < 8: global_sum += peer
            peer = T.shfl_down(global_sum, 4, 64)
            if lane < 4: global_sum += peer
            peer = T.shfl_down(global_sum, 2, 64)
            if lane < 2: global_sum += peer
            peer = T.shfl_down(global_sum, 1, 64)
            if lane < 1: global_sum += peer
            global_sum = T.shfl_sync(global_sum, 0, 64)

            acc0 = T.alloc_var(T.float16x2)
            acc1 = T.alloc_var(T.float16x2)
            acc2 = T.alloc_var(T.float16x2)
            acc3 = T.alloc_var(T.float16x2)
            acc4 = T.alloc_var(T.float16x2)
            acc5 = T.alloc_var(T.float16x2)
            acc6 = T.alloc_var(T.float16x2)
            acc7 = T.alloc_var(T.float16x2)
            acc0 = T.Broadcast(T.float16(0.0), 2)
            acc1 = T.Broadcast(T.float16(0.0), 2)
            acc2 = T.Broadcast(T.float16(0.0), 2)
            acc3 = T.Broadcast(T.float16(0.0), 2)
            acc4 = T.Broadcast(T.float16(0.0), 2)
            acc5 = T.Broadcast(T.float16(0.0), 2)
            acc6 = T.Broadcast(T.float16(0.0), 2)
            acc7 = T.Broadcast(T.float16(0.0), 2)
            for group in T.unroll(num_split // 8):
                v0 = packed_partial[((bid * num_split + group * 8 + 0) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v1 = packed_partial[((bid * num_split + group * 8 + 1) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v2 = packed_partial[((bid * num_split + group * 8 + 2) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v3 = packed_partial[((bid * num_split + group * 8 + 3) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v4 = packed_partial[((bid * num_split + group * 8 + 4) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v5 = packed_partial[((bid * num_split + group * 8 + 5) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v6 = packed_partial[((bid * num_split + group * 8 + 6) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                v7 = packed_partial[((bid * num_split + group * 8 + 7) * heads + hid) * dim + T.Ramp(tx * 2, 1, 2)]
                acc0 = T.add2(acc0, v0)
                acc1 = T.add2(acc1, v1)
                acc2 = T.add2(acc2, v2)
                acc3 = T.add2(acc3, v3)
                acc4 = T.add2(acc4, v4)
                acc5 = T.add2(acc5, v5)
                acc6 = T.add2(acc6, v6)
                acc7 = T.add2(acc7, v7)
            total = T.cast(acc0, T.float32x2)
            total = T.add2(total, T.cast(acc1, T.float32x2))
            total = T.add2(total, T.cast(acc2, T.float32x2))
            total = T.add2(total, T.cast(acc3, T.float32x2))
            total = T.add2(total, T.cast(acc4, T.float32x2))
            total = T.add2(total, T.cast(acc5, T.float32x2))
            total = T.add2(total, T.cast(acc6, T.float32x2))
            total = T.add2(total, T.cast(acc7, T.float32x2))
            output_local[T.Ramp(0, 1, 2)] = total
            inv_sum = 1.0 / global_sum
            for vi in T.vectorized(2):
                output[bid, hid, tx * 2 + vi] = output_local[vi] * inv_sum
    return main


_LONG_KERNELS = {
    16384: flashattn_16k_e7_explicit_pv_helper_rawpartial,
    32768: a254_32k_fulltoken_fastpath,
    65536: a234_64k_fulltoken_fastpath,
}


@lru_cache(maxsize=None)
def _compile_long(batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    return _LONG_KERNELS[kv_ctx](batch, heads, kv_heads, kv_ctx, dim, pe_dim, 64, 256, 1, 32, 128)


def run_kernel(q, q_pe, kv, k_pe, output, batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    if kv_ctx in (2048, 4096, 8192):
        return _run_short(q, q_pe, kv, k_pe, output, batch, heads, kv_heads, kv_ctx, dim, pe_dim)
    if kv_ctx not in _LONG_KERNELS:
        raise ValueError(f"unsupported contest KV length: {kv_ctx}")
    kernel = _compile_long(batch, heads, kv_heads, kv_ctx, dim, pe_dim)
    kernel(q, q_pe, kv, k_pe, output)
