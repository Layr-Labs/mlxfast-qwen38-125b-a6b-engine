// TrackVATransposes.swift -- 2-pass vector SDPA second pass, batched transposes.
//
// COPIED from Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h
// (`sdpa_vector_2pass_1` and `sdpa_vector_2pass_2`). Vendor/ is not edited.
// The copy is a Runner-owned MLXFast.metalKernel string, the same pattern as
// TrackFastKernels. First pass is the vendor walk for decode (S=1, no mask,
// no sinks). Host launch stays 1024 threads per output head.
//
// TRACK_VA_BATCHED_TRANSPOSE (default ON): at BF16 and D=256 the second pass
// gives each of the 8 float32 output components its own 1024-float plane
// (32 KiB), writes all eight, executes ONE threadgroup barrier, then the
// original transposed simd_sum + divide. `0` skips this intercept so MLX's
// original dispatch runs byte-for-byte. Every other dtype/width keeps the
// vendor single-plane loop (2 barriers per component).

import Foundation
import Metal
import MLX

enum TrackVATransposes {
    static let enabled =
        ProcessInfo.processInfo.environment["TRACK_VA_BATCHED_TRANSPOSE"] != "0"

    static func resolves(_ value: String?) -> Bool { value != "0" }

    static let bn = 32
    static let bd = 32
    static let minKeys = 1024
    /// 8 planes x 32 x 32 floats = 32,768 B. Board-promoted geometry (4470c626).
    static let batchedPlanes = 8
    static let batchedThreadgroupBytes = batchedPlanes * bn * bd * 4

    private static let archTail: Character = {
        MTLCreateSystemDefaultDevice()?.architecture.name.last ?? "s"
    }()

    /// Vendor `sdpa_vector_2pass` block heuristic (scaled_dot_product_attention.cpp).
    static func blockCount(N: Int, gqa: Int, qLen: Int) -> Int {
        if let raw = ProcessInfo.processInfo.environment["MLX_SDPA_BLOCKS"],
            let env = Int(raw), env > 0
        {
            return ((env + 31) / 32) * 32
        }
        let nSimds = gqa * qLen
        switch archTail {
        case "s":
            var blocks = 64
            if N > 1024 && nSimds > 4 {
                if N <= 8192 {
                    blocks = 128
                } else if N <= 32768 {
                    blocks = 256
                } else if N <= 65536 {
                    blocks = 512
                } else {
                    blocks = 1024
                }
            }
            return blocks
        case "d":
            var blocks = 128
            if nSimds <= 2 && N > 8192 {
                blocks = 256
            } else if nSimds >= 6 {
                if N >= 16384 && N < 65536 {
                    blocks = 512
                } else if N >= 65536 {
                    blocks = 1024
                }
            }
            return blocks
        default:
            return nSimds >= 4 ? 64 : 32
        }
    }

    static func eligibleShape(
        batch: Int, heads: Int, qLen: Int, dim: Int, kHeads: Int, kLen: Int,
        cacheRows: Int, isBF16: Bool, batched: Bool
    ) -> Bool {
        guard batched else { return false }
        guard isBF16, dim == 256 else { return false }
        guard cacheRows == 1, batch == 1, qLen == 1 else { return false }
        guard kHeads >= 1, heads % kHeads == 0 else { return false }
        guard dim % bn == 0, kLen >= minKeys else { return false }
        return true
    }

    static func eligible(q: MLXArray, kHeads: Int, kLen: Int, cacheRows: Int) -> Bool {
        guard q.ndim == 4 else { return false }
        return eligibleShape(
            batch: q.dim(0), heads: q.dim(1), qLen: q.dim(2), dim: q.dim(3),
            kHeads: kHeads, kLen: kLen, cacheRows: cacheRows,
            isBF16: q.dtype == .bfloat16, batched: enabled)
    }

    static func attend(
        q: MLXArray, k: MLXArray, v: MLXArray, scale: Float,
        batched: Bool? = nil, blocks: Int? = nil
    ) -> MLXArray {
        let B = q.dim(0), HQ = q.dim(1), S = q.dim(2), D = q.dim(3)
        let HK = k.dim(1), N = k.dim(2)
        let gqa = HQ / HK
        let nBlocks = blocks ?? blockCount(N: N, gqa: gqa, qLen: S)
        let useBatched = (batched ?? enabled) && D == 256 && q.dtype == .bfloat16
        let nKV = MLXArray(Int32(N))
        let nBlk = MLXArray(Int32(nBlocks))
        let scaleA = TrackFastKernels.scalar(scale, dtype: .float32)
        let partials = firstKernel(
            [q, k, v, nKV, nBlk, scaleA],
            template: [("InT", q.dtype), ("D", D)],
            grid: (bn * HK, gqa * B, nBlocks),
            threadGroup: (bn, gqa, 1),
            outputShapes: [
                [B * HQ * S, nBlocks, D], [B * HQ * S, nBlocks], [B * HQ * S, nBlocks],
            ],
            outputDTypes: [q.dtype, .float32, .float32])
        let reduced = reduce(
            partials: partials[0], sums: partials[1], maxs: partials[2],
            batched: useBatched)
        return reduced.reshaped([B, HQ, S, D])
    }

    static func reduce(
        partials: MLXArray, sums: MLXArray, maxs: MLXArray, batched: Bool
    ) -> MLXArray {
        let heads = partials.dim(0)
        let D = partials.dim(2)
        let nBlocks = partials.dim(1)
        let nBlk = MLXArray(Int32(nBlocks))
        let useBatched = batched && D == 256 && partials.dtype == .bfloat16
        return secondKernel(
            [partials, sums, maxs, nBlk],
            template: [("InT", partials.dtype), ("D", D), ("BATCHED", useBatched)],
            grid: (1024, heads, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [[heads, D]],
            outputDTypes: [partials.dtype])[0]
    }

    static let header = """
        #include <metal_stdlib>
        #include <metal_simdgroup>
        using namespace metal;
        METAL_FUNC float va_finite_min() {
            return -metal::numeric_limits<float>::max();
        }
        """

    /// Copied sdpa_vector_2pass_1 walk: strided keys `block, block+blocks, ...`.
    /// Decode only (S=1, no mask, no sinks). Arithmetic is the vendor order.
    static let firstSource = """
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = D / BD;
        using U = float;
        const int kv_head_idx = (int)threadgroup_position_in_grid.x;
        const int batch_idx = (int)threadgroup_position_in_grid.y;
        const int block_idx = (int)threadgroup_position_in_grid.z;
        const int gqa_factor = (int)threads_per_threadgroup.y;
        const int p = (int)thread_position_in_threadgroup.y;
        const uint simd_lid = thread_index_in_simdgroup;
        const int q_head = gqa_factor * kv_head_idx + p;
        const int num_kv_heads = (int)threadgroups_per_grid.x;
        const int num_q_heads = num_kv_heads * gqa_factor;
        const int N = (int)n_kv;
        const int blocks = (int)n_blocks;
        const int64_t k_head_stride = keys_shape[1] == 1 ? keys_strides[0] : keys_strides[1];
        const int64_t k_seq_stride = keys_strides[2];
        const int64_t v_head_stride = values_shape[1] == 1 ? values_strides[0] : values_strides[1];
        const int64_t v_seq_stride = values_strides[2];
        const int64_t q_head_stride = queries_shape[1] == 1 ? queries_strides[0] : queries_strides[1];
        const int kv_batch_head_idx = batch_idx * num_kv_heads + kv_head_idx;
        const device InT* kp = keys + (int64_t)kv_batch_head_idx * k_head_stride
            + (int64_t)block_idx * k_seq_stride + (int64_t)simd_lid * qk_per_thread;
        const device InT* vp = values + (int64_t)kv_batch_head_idx * v_head_stride
            + (int64_t)block_idx * v_seq_stride + (int64_t)simd_lid * v_per_thread;
        thread U q[qk_per_thread];
        thread U o[v_per_thread];
        const int q_batch_head = batch_idx * num_q_heads + q_head;
        const device InT* qp = queries + (int64_t)q_batch_head * q_head_stride
            + (int64_t)simd_lid * qk_per_thread;
        for (int i = 0; i < qk_per_thread; i++) {
            q[i] = scale * static_cast<U>(qp[i]);
        }
        U max_score = va_finite_min();
        U sum_exp_score = 0;
        for (int i = 0; i < v_per_thread; i++) { o[i] = 0; }
        for (int i = block_idx; i < N; i += blocks) {
            U score = 0;
            for (int j = 0; j < qk_per_thread; j++) {
                score += q[j] * static_cast<U>(kp[j]);
            }
            score = simd_sum(score);
            U new_max = max(max_score, score);
            U factor = fast::exp(max_score - new_max);
            U exp_score = fast::exp(score - new_max);
            max_score = new_max;
            sum_exp_score = sum_exp_score * factor + exp_score;
            for (int j = 0; j < v_per_thread; j++) {
                o[j] = o[j] * factor + exp_score * static_cast<U>(vp[j]);
            }
            kp += (int64_t)blocks * k_seq_stride;
            vp += (int64_t)blocks * v_seq_stride;
        }
        if (simd_lid == 0) {
            sums[(int64_t)q_batch_head * blocks + block_idx] = sum_exp_score;
            maxs[(int64_t)q_batch_head * blocks + block_idx] = max_score;
        }
        device InT* op = out + ((int64_t)q_batch_head * blocks + block_idx) * D
            + (int64_t)simd_lid * v_per_thread;
        for (int j = 0; j < v_per_thread; j++) {
            op[j] = static_cast<InT>(o[j]);
        }
        """

    /// Copied sdpa_vector_2pass_2. Grid is dispatch_threads (1024, heads, 1).
    /// PLANES=8 only when BATCHED && D==256; otherwise the vendor single-plane
    /// loop (write, barrier, simd_sum, divide, barrier) per component.
    static let secondSource = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = D / BD;
        constexpr int PLANES = (BATCHED && D == 256) ? 8 : 1;
        using U = float;
        thread U o[elem_per_thread] = {0};
        threadgroup U outputs[PLANES * BN * BD];
        const int head_idx = (int)threadgroup_position_in_grid.y;
        const int blocks = (int)n_blocks;
        const uint simd_gid = simdgroup_index_in_threadgroup;
        const uint simd_lid = thread_index_in_simdgroup;
        const device InT* partials_p = partials
            + (int64_t)head_idx * blocks * D
            + (int64_t)simd_gid * D + (int64_t)simd_lid * elem_per_thread;
        const device float* sums_p = sums + (int64_t)head_idx * blocks;
        const device float* maxs_p = maxs + (int64_t)head_idx * blocks;
        device InT* out_p = out + (int64_t)head_idx * D
            + (int64_t)simd_gid * elem_per_thread;
        U sum_exp_score = 0;
        U max_score = va_finite_min();
        for (int b = 0; b < blocks / BN; ++b) {
            max_score = max(max_score, maxs_p[simd_lid + BN * b]);
        }
        max_score = simd_max(max_score);
        for (int b = 0; b < blocks / BN; ++b) {
            U factor = fast::exp(maxs_p[simd_lid + BN * b] - max_score);
            sum_exp_score += factor * sums_p[simd_lid + BN * b];
        }
        sum_exp_score = simd_sum(sum_exp_score);
        for (int b = 0; b < blocks / BN; ++b) {
            U factor = fast::exp(maxs_p[simd_gid] - max_score);
            for (int i = 0; i < elem_per_thread; i++) {
                o[i] += factor * static_cast<U>(partials_p[i]);
            }
            maxs_p += BN;
            sums_p += BN;
            partials_p += BN * D;
        }
        if (PLANES > 1) {
            for (int i = 0; i < elem_per_thread; i++) {
                outputs[i * BN * BD + simd_lid * BD + simd_gid] = o[i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int i = 0; i < elem_per_thread; i++) {
                o[i] = simd_sum(outputs[i * BN * BD + simd_gid * BD + simd_lid]);
                o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
            }
        } else {
            for (int i = 0; i < elem_per_thread; i++) {
                outputs[simd_lid * BD + simd_gid] = o[i];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
                o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if (simd_lid == 0) {
            for (int i = 0; i < elem_per_thread; i++) {
                out_p[i] = static_cast<InT>(o[i]);
            }
        }
        """

    nonisolated(unsafe) static let firstKernel = MLXFast.metalKernel(
        name: "track_va_sdpa_2pass_1",
        inputNames: ["queries", "keys", "values", "n_kv", "n_blocks", "scale"],
        outputNames: ["out", "sums", "maxs"],
        source: firstSource, header: header, ensureRowContiguous: false)

    nonisolated(unsafe) static let secondKernel = MLXFast.metalKernel(
        name: "track_va_sdpa_2pass_2",
        inputNames: ["partials", "sums", "maxs", "n_blocks"],
        outputNames: ["out"],
        source: secondSource, header: header, ensureRowContiguous: true)

    // MARK: - CPU replica of the second-pass reduction (toggle exactness without Metal)

    static let finiteMin: Float = -Float.greatestFiniteMagnitude

    /// Butterfly `simd_sum` over 32 lanes (xor-shuffle association).
    static func simdSum(_ xs: [Float]) -> Float {
        var v = xs
        var offset = 1
        while offset < bn {
            var next = v
            for i in 0 ..< bn {
                next[i] = v[i] + v[i ^ offset]
            }
            v = next
            offset *= 2
        }
        return v[0]
    }

    /// Second-pass reduction for `heads` independent outputs.
    /// `batched` selects the 8-plane single-barrier transpose only at D=256.
    static func cpuSecondPass(
        partials: [Float], sums: [Float], maxs: [Float],
        heads: Int, dim: Int, blocks: Int, batched: Bool
    ) -> [Float] {
        let ept = dim / bd
        let planes = (batched && dim == 256) ? batchedPlanes : 1
        var out = [Float](repeating: 0, count: heads * dim)
        for head in 0 ..< heads {
            let pBase = head * blocks * dim
            let sBase = head * blocks
            var laneMax = [Float](repeating: finiteMin, count: bn)
            for lid in 0 ..< bn {
                var m = finiteMin
                for b in 0 ..< (blocks / bn) {
                    m = max(m, maxs[sBase + lid + bn * b])
                }
                laneMax[lid] = m
            }
            let maxScore = laneMax.max() ?? finiteMin
            var laneSum = [Float](repeating: 0, count: bn)
            for lid in 0 ..< bn {
                var s: Float = 0
                for b in 0 ..< (blocks / bn) {
                    s += exp(maxs[sBase + lid + bn * b] - maxScore)
                        * sums[sBase + lid + bn * b]
                }
                laneSum[lid] = s
            }
            let sumExp = simdSum(laneSum)
            var o = [[[Float]]](
                repeating: [[Float]](
                    repeating: [Float](repeating: 0, count: ept), count: bn),
                count: bn)
            for gid in 0 ..< bn {
                for lid in 0 ..< bn {
                    for b in 0 ..< (blocks / bn) {
                        let factor = exp(maxs[sBase + bn * b + gid] - maxScore)
                        let pOff = pBase + (bn * b + gid) * dim + lid * ept
                        for i in 0 ..< ept {
                            o[gid][lid][i] += factor * partials[pOff + i]
                        }
                    }
                }
            }
            if planes > 1 {
                var tg = [Float](repeating: 0, count: planes * bn * bd)
                for i in 0 ..< ept {
                    for gid in 0 ..< bn {
                        for lid in 0 ..< bn {
                            tg[i * bn * bd + lid * bd + gid] = o[gid][lid][i]
                        }
                    }
                }
                for i in 0 ..< ept {
                    for gid in 0 ..< bn {
                        var laneVals = [Float](repeating: 0, count: bn)
                        for lid in 0 ..< bn {
                            laneVals[lid] = tg[i * bn * bd + gid * bd + lid]
                        }
                        let reduced = simdSum(laneVals)
                        let v = sumExp == 0 ? reduced : reduced / sumExp
                        for lid in 0 ..< bn { o[gid][lid][i] = v }
                    }
                }
            } else {
                for i in 0 ..< ept {
                    var tg = [Float](repeating: 0, count: bn * bd)
                    for gid in 0 ..< bn {
                        for lid in 0 ..< bn {
                            tg[lid * bd + gid] = o[gid][lid][i]
                        }
                    }
                    for gid in 0 ..< bn {
                        var laneVals = [Float](repeating: 0, count: bn)
                        for lid in 0 ..< bn {
                            laneVals[lid] = tg[gid * bd + lid]
                        }
                        let reduced = simdSum(laneVals)
                        let v = sumExp == 0 ? reduced : reduced / sumExp
                        for lid in 0 ..< bn { o[gid][lid][i] = v }
                    }
                }
            }
            for gid in 0 ..< bn {
                for i in 0 ..< ept {
                    out[head * dim + gid * ept + i] = o[gid][0][i]
                }
            }
        }
        return out
    }
}
