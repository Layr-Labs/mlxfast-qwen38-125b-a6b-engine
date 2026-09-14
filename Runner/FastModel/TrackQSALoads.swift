// TrackQSALoads.swift -- 2-pass vector SDPA first pass, with two load opts.
//
// COPIED from Vendor/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/sdpa_vector.h
// (`sdpa_vector_2pass_1` and `sdpa_vector_2pass_2`). Vendor/ is not edited.
// The copy is a Runner-owned MLXFast.metalKernel string, the same pattern as
// TrackSplitQSA / TrackFastKernels.
//
// Two independently switchable mechanisms, both default ON, each falling
// back to the copied original walk when the env is "0" or the shape/layout
// does not admit the transform:
//
//   TRACK_QSA_VECLOAD  -- K/V/Q move as metal::vec<T,4> (16-byte / 8-byte
//                         transactions) when D is 128 or 256 and the last
//                         dimension is contiguous. Arithmetic is still the
//                         original per-component product/add order.
//   TRACK_QSA_KV_REUSE -- one simdgroup walks two consecutive query heads
//                         that share a KV head (GQA even). One K/V load
//                         feeds both heads' independent online-softmax.
//                         Host launch geometry is unchanged; Y >= GQA/2
//                         returns without writing.
//
// Eligible decode: one query position, bf16, no mask/sinks, N >= 1024.
// The second pass is the copied sdpa_vector_2pass_2, bit-identical.

import Foundation
import Metal
import MLX

enum TrackQSALoads {
    static let vecLoadEnabled =
        ProcessInfo.processInfo.environment["TRACK_QSA_VECLOAD"] != "0"
    static let kvReuseEnabled =
        ProcessInfo.processInfo.environment["TRACK_QSA_KV_REUSE"] != "0"

    static func resolves(_ value: String?) -> Bool { value != "0" }

    private static let bn = 32
    private static let minKeys = 1024

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

    static func vecLoadLayoutOK(q: MLXArray, k: MLXArray, v: MLXArray) -> Bool {
        let d = q.dim(-1)
        guard d == 128 || d == 256 else { return false }
        guard q.ndim == 4, k.ndim == 4, v.ndim == 4 else { return false }
        // Contiguous last dim is this track's KV layout (stride 1). The
        // kernel still addresses through the real strides; vec loads are
        // only issued when D packs into metal::vec<T,4> groups of 4.
        return q.dim(-1) == d && k.dim(-1) == d && v.dim(-1) == d
    }

    /// Decode window this kernel can serve. `kLen` is the post-append KV length.
    /// `kHeads` is the KV-head count (from the new K chunk or the cache).
    static func eligibleShape(
        batch: Int, heads: Int, qLen: Int, dim: Int, kHeads: Int, kLen: Int,
        cacheRows: Int, isBF16: Bool, vecLoad: Bool, kvReuse: Bool
    ) -> Bool {
        guard vecLoad || kvReuse else { return false }
        guard cacheRows == 1, isBF16, batch == 1, qLen == 1 else { return false }
        guard kHeads >= 1, heads % kHeads == 0 else { return false }
        guard dim % bn == 0, dim >= 64, kLen >= minKeys else { return false }
        let gqa = heads / kHeads
        let wantVec = vecLoad && (dim == 128 || dim == 256)
        let wantReuse = kvReuse && gqa % 2 == 0
        return wantVec || wantReuse
    }

    static func eligible(q: MLXArray, kHeads: Int, kLen: Int, cacheRows: Int) -> Bool {
        guard q.ndim == 4 else { return false }
        return eligibleShape(
            batch: q.dim(0), heads: q.dim(1), qLen: q.dim(2), dim: q.dim(3),
            kHeads: kHeads, kLen: kLen, cacheRows: cacheRows,
            isBF16: q.dtype == .bfloat16,
            vecLoad: vecLoadEnabled, kvReuse: kvReuseEnabled)
    }

    static func eligibleKV(
        q: MLXArray, k: MLXArray, v: MLXArray, cacheRows: Int
    ) -> Bool {
        guard k.ndim == 4, v.ndim == 4, k.dtype == q.dtype, v.dtype == q.dtype else {
            return false
        }
        guard k.dim(0) == q.dim(0), v.dim(0) == q.dim(0) else { return false }
        guard k.dim(2) == v.dim(2), k.dim(3) == q.dim(3), v.dim(3) == q.dim(3) else {
            return false
        }
        guard eligible(q: q, kHeads: k.dim(1), kLen: k.dim(2), cacheRows: cacheRows) else {
            return false
        }
        let gqa = q.dim(1) / k.dim(1)
        let wantVec = vecLoadEnabled && vecLoadLayoutOK(q: q, k: k, v: v)
        let wantReuse = kvReuseEnabled && gqa % 2 == 0
        return wantVec || wantReuse
    }

    static func attend(
        q: MLXArray, k: MLXArray, v: MLXArray, scale: Float,
        vecLoad: Bool? = nil, kvReuse: Bool? = nil, blocks: Int? = nil
    ) -> MLXArray {
        let B = q.dim(0), HQ = q.dim(1), S = q.dim(2), D = q.dim(3)
        let HK = k.dim(1), N = k.dim(2)
        let gqa = HQ / HK
        let useVec = (vecLoad ?? vecLoadEnabled) && vecLoadLayoutOK(q: q, k: k, v: v)
        let useReuse = (kvReuse ?? kvReuseEnabled) && gqa % 2 == 0 && S == 1
        let nBlocks = blocks ?? blockCount(N: N, gqa: gqa, qLen: S)
        let nKV = MLXArray(Int32(N))
        let nBlk = MLXArray(Int32(nBlocks))
        let scaleA = TrackFastKernels.scalar(scale, dtype: .float32)
        let partials = firstKernel(
            [q, k, v, nKV, nBlk, scaleA],
            template: [
                ("InT", q.dtype), ("D", D), ("VECLOAD", useVec), ("KV_REUSE", useReuse),
            ],
            grid: (bn * HK, gqa * B, nBlocks),
            threadGroup: (bn, gqa, 1),
            outputShapes: [[B * HQ * S, nBlocks, D], [B * HQ * S, nBlocks], [B * HQ * S, nBlocks]],
            outputDTypes: [q.dtype, .float32, .float32])
        let out = secondKernel(
            [partials[0], partials[1], partials[2], nBlk],
            template: [("InT", q.dtype), ("D", D)],
            grid: (1024, B * HQ * S, 1),
            threadGroup: (1024, 1, 1),
            outputShapes: [[B, HQ, S, D]],
            outputDTypes: [q.dtype])[0]
        return out
    }

    // MARK: - copied first pass (sdpa_vector_2pass_1) with the two opts

    static let header = """
        #include <metal_stdlib>
        #include <metal_simdgroup>
        using namespace metal;
        METAL_FUNC float qsa_finite_min() {
            return -metal::numeric_limits<float>::max();
        }
        template <typename T>
        METAL_FUNC float4 qsa_load4(const device T* p) {
            const device metal::vec<T, 4> r = *(const device metal::vec<T, 4>*)p;
            return float4(float(r[0]), float(r[1]), float(r[2]), float(r[3]));
        }
        template <typename T>
        METAL_FUNC void qsa_store4(device T* p, float4 x) {
            metal::vec<T, 4> r;
            r[0] = static_cast<T>(x.x);
            r[1] = static_cast<T>(x.y);
            r[2] = static_cast<T>(x.z);
            r[3] = static_cast<T>(x.w);
            *(device metal::vec<T, 4>*)p = r;
        }
        """

    /// Copied sdpa_vector_2pass_1 walk: strided keys `block, block+blocks, ...`.
    /// VECLOAD only changes the K/V/Q transaction. KV_REUSE only changes how
    /// many heads consume each loaded K/V vector. Score products stay in
    /// increasing coordinate order; each head keeps its own max/sum/o.
    static let firstSource = """
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = D / BD;
        constexpr int HPT = KV_REUSE ? 2 : 1;
        using U = float;
        const int kv_head_idx = (int)threadgroup_position_in_grid.x;
        const int batch_idx = (int)threadgroup_position_in_grid.y;
        const int block_idx = (int)threadgroup_position_in_grid.z;
        const int gqa_factor = (int)threads_per_threadgroup.y;
        const int p = (int)thread_position_in_threadgroup.y;
        const uint simd_lid = thread_index_in_simdgroup;
        if (KV_REUSE && p >= gqa_factor / 2) return;
        const int q_head0 = gqa_factor * kv_head_idx + (KV_REUSE ? 2 * p : p);
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
        thread U q[HPT][qk_per_thread];
        thread U o[HPT][v_per_thread];
        U max_score[HPT];
        U sum_exp_score[HPT];
        const bool use_vec = VECLOAD && (qk_per_thread % 4 == 0);
        for (int h = 0; h < HPT; h++) {
            const int q_batch_head = batch_idx * num_q_heads + q_head0 + h;
            const device InT* qp = queries + (int64_t)q_batch_head * q_head_stride
                + (int64_t)simd_lid * qk_per_thread;
            if (use_vec) {
                for (int u = 0; u < qk_per_thread / 4; ++u) {
                    float4 vv = qsa_load4(qp + 4 * u);
                    q[h][4 * u + 0] = scale * vv.x;
                    q[h][4 * u + 1] = scale * vv.y;
                    q[h][4 * u + 2] = scale * vv.z;
                    q[h][4 * u + 3] = scale * vv.w;
                }
            } else {
                for (int i = 0; i < qk_per_thread; i++) {
                    q[h][i] = scale * static_cast<U>(qp[i]);
                }
            }
            max_score[h] = qsa_finite_min();
            sum_exp_score[h] = 0;
            for (int i = 0; i < v_per_thread; i++) { o[h][i] = 0; }
        }
        for (int i = block_idx; i < N; i += blocks) {
            thread U kreg[qk_per_thread];
            thread U vreg[v_per_thread];
            if (use_vec) {
                for (int u = 0; u < qk_per_thread / 4; ++u) {
                    float4 kk = qsa_load4(kp + 4 * u);
                    float4 vv = qsa_load4(vp + 4 * u);
                    kreg[4 * u + 0] = kk.x; kreg[4 * u + 1] = kk.y;
                    kreg[4 * u + 2] = kk.z; kreg[4 * u + 3] = kk.w;
                    vreg[4 * u + 0] = vv.x; vreg[4 * u + 1] = vv.y;
                    vreg[4 * u + 2] = vv.z; vreg[4 * u + 3] = vv.w;
                }
            } else {
                for (int j = 0; j < qk_per_thread; j++) {
                    kreg[j] = static_cast<U>(kp[j]);
                    vreg[j] = static_cast<U>(vp[j]);
                }
            }
            for (int h = 0; h < HPT; h++) {
                U score = 0;
                for (int j = 0; j < qk_per_thread; j++) { score += q[h][j] * kreg[j]; }
                score = simd_sum(score);
                U new_max = max(max_score[h], score);
                U factor = fast::exp(max_score[h] - new_max);
                U exp_score = fast::exp(score - new_max);
                max_score[h] = new_max;
                sum_exp_score[h] = sum_exp_score[h] * factor + exp_score;
                for (int j = 0; j < v_per_thread; j++) {
                    o[h][j] = o[h][j] * factor + exp_score * vreg[j];
                }
            }
            kp += (int64_t)blocks * k_seq_stride;
            vp += (int64_t)blocks * v_seq_stride;
        }
        for (int h = 0; h < HPT; h++) {
            const int q_batch_head = batch_idx * num_q_heads + q_head0 + h;
            const int o_offset = q_batch_head;
            if (simd_lid == 0) {
                sums[(int64_t)o_offset * blocks + block_idx] = sum_exp_score[h];
                maxs[(int64_t)o_offset * blocks + block_idx] = max_score[h];
            }
            device InT* op = out + ((int64_t)o_offset * blocks + block_idx) * D
                + (int64_t)simd_lid * v_per_thread;
            if (use_vec) {
                for (int u = 0; u < v_per_thread / 4; ++u) {
                    qsa_store4(op + 4 * u, float4(
                        o[h][4 * u + 0], o[h][4 * u + 1],
                        o[h][4 * u + 2], o[h][4 * u + 3]));
                }
            } else {
                for (int j = 0; j < v_per_thread; j++) {
                    op[j] = static_cast<InT>(o[h][j]);
                }
            }
        }
        """

    /// Copied sdpa_vector_2pass_2. Grid is dispatch_threads (1024, heads, 1).
    static let secondSource = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = D / BD;
        using U = float;
        thread U o[elem_per_thread] = {0};
        threadgroup U outputs[BN * BD];
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
        U max_score = qsa_finite_min();
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
        for (int i = 0; i < elem_per_thread; i++) {
            outputs[simd_lid * BD + simd_gid] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
            o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (simd_lid == 0) {
            for (int i = 0; i < elem_per_thread; i++) {
                out_p[i] = static_cast<InT>(o[i]);
            }
        }
        """

    nonisolated(unsafe) static let firstKernel = MLXFast.metalKernel(
        name: "track_qsa_sdpa_2pass_1",
        inputNames: ["queries", "keys", "values", "n_kv", "n_blocks", "scale"],
        outputNames: ["out", "sums", "maxs"],
        source: firstSource, header: header, ensureRowContiguous: false)

    nonisolated(unsafe) static let secondKernel = MLXFast.metalKernel(
        name: "track_qsa_sdpa_2pass_2",
        inputNames: ["partials", "sums", "maxs", "n_blocks"],
        outputNames: ["out"],
        source: secondSource, header: header, ensureRowContiguous: true)

    // MARK: - CPU replica of the first-pass walk (toggle exactness without Metal)

    /// First-pass partials for one query position. `vecLoad` does not change
    /// arithmetic: it is a memory-transaction flag. `kvReuse` pairs heads
    /// 2p, 2p+1 onto one K/V walk; each head still owns its max/sum/o.
    static func cpuFirstPass(
        q: [Float], k: [Float], v: [Float],
        HQ: Int, HK: Int, D: Int, N: Int, blocks: Int, scale: Float,
        kvReuse: Bool
    ) -> (o: [Float], sums: [Float], maxs: [Float]) {
        let gqa = HQ / HK
        let qkpt = D / bn
        var o = [Float](repeating: 0, count: HQ * blocks * D)
        var sums = [Float](repeating: 0, count: HQ * blocks)
        var maxs = [Float](repeating: qsaFiniteMin, count: HQ * blocks)
        let hpt = kvReuse ? 2 : 1
        let pairs = kvReuse ? gqa / 2 : gqa
        for kv in 0 ..< HK {
            for block in 0 ..< blocks {
                for p in 0 ..< pairs {
                    let h0 = gqa * kv + (kvReuse ? 2 * p : p)
                    var accMax = [Float](repeating: qsaFiniteMin, count: hpt)
                    var accSum = [Float](repeating: 0, count: hpt)
                    var accO = [Float](repeating: 0, count: hpt * D)
                    var qh = [Float](repeating: 0, count: hpt * D)
                    for h in 0 ..< hpt {
                        let head = h0 + h
                        for d in 0 ..< D { qh[h * D + d] = scale * q[head * D + d] }
                    }
                    var i = block
                    while i < N {
                        let kBase = (kv * N + i) * D
                        for h in 0 ..< hpt {
                            var lane = [Float](repeating: 0, count: bn)
                            for lid in 0 ..< bn {
                                var s: Float = 0
                                let base = lid * qkpt
                                for j in 0 ..< qkpt {
                                    s += qh[h * D + base + j] * k[kBase + base + j]
                                }
                                lane[lid] = s
                            }
                            var score: Float = 0
                            for lid in 0 ..< bn { score += lane[lid] }
                            let newMax = max(accMax[h], score)
                            let factor = exp(accMax[h] - newMax)
                            let expScore = exp(score - newMax)
                            accMax[h] = newMax
                            accSum[h] = accSum[h] * factor + expScore
                            for d in 0 ..< D {
                                accO[h * D + d] =
                                    accO[h * D + d] * factor + expScore * v[kBase + d]
                            }
                        }
                        i += blocks
                    }
                    for h in 0 ..< hpt {
                        let head = h0 + h
                        let slot = head * blocks + block
                        sums[slot] = accSum[h]
                        maxs[slot] = accMax[h]
                        for d in 0 ..< D { o[slot * D + d] = accO[h * D + d] }
                    }
                }
            }
        }
        return (o, sums, maxs)
    }

    static let qsaFiniteMin: Float = -Float.greatestFiniteMagnitude
}
