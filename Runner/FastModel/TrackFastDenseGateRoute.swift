// Share one dispatch between expert selection and the shared expert's dense gate.
// The gate follows native dot.h's vector accumulators and both reduction stages.

import MLX

enum TrackFastDenseGateRoute {
    private static let kernel = MLXFast.metalKernel(
        name: "track_route_dense_shared_gate",
        inputNames: ["logits", "x", "gate_w"], outputNames: ["idx", "w", "gate"],
        source: source, ensureRowContiguous: true)

    static func apply(logits: MLXArray, x: MLXArray, weight: MLXArray, topK: Int)
        -> (idx: MLXArray, w: MLXArray, gate: MLXArray)?
    {
        guard x.shape == [1, 2560], x.dtype == .bfloat16,
            weight.shape == [1, 2560], weight.dtype == x.dtype,
            x.strides.last == 1, weight.strides.last == 1,
            logits.dtype == .float32, logits.size == 512, topK == 10
        else { return nil }
        let groups = (x.dim(1) + 1023) / 1024 + 1
        let result = kernel(
            [logits.reshaped(1, 512), x, weight],
            template: [("E", 512), ("K", topK), ("T", x.dtype), ("KD", x.dim(1))],
            grid: (32, groups, 1), threadGroup: (32, groups, 1),
            outputShapes: [[1, topK], [1, topK], [1]], outputDTypes: [.uint32, .float32, x.dtype])
        return (result[0], result[1], result[2])
    }

    private static let source = #"""
        constexpr int E_PER = (E + 31) / 32;
        const uint row = threadgroup_position_in_grid.y;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;

        constexpr uint DOT_GROUPS = (KD + 1023) / 1024;
        threadgroup float gate_partials[16];
        if (sg == 0 && lane >= DOT_GROUPS && lane < 16) { gate_partials[lane] = 0; }
        if (sg < DOT_GROUPS) {
            constexpr int VEC = 16 / sizeof(T);
            const int start = (int)sg * 32 * 32 + (int)lane * VEC;
            float4 c = 0.0f;
            for (int i = 0; i < 32; i += VEC) {
                const int ix = start + i * 32;
                if (ix + VEC <= KD) {
                    for (int j = 0; j < VEC; j += 4) {
                        c += float4(*reinterpret_cast<const device metal::vec<T, 4>*>(x + ix + j)) *
                             float4(*reinterpret_cast<const device metal::vec<T, 4>*>(gate_w + ix + j));
                    }
                } else {
                    for (int j = 0; j < VEC; ++j) {
                        const int nidx = ix + j;
                        if (nidx < KD) { c[j & 3] += float(x[nidx]) * float(gate_w[nidx]); }
                    }
                }
            }
            float total = c[0] + c[1] + c[2] + c[3];
            total = simd_sum(total);
            if (lane == 0) { gate_partials[sg] = total; }
        }
        threadgroup float selv[K];
        threadgroup uint seli[K];
        if (sg == DOT_GROUPS) {
        const device float* lr = logits + (size_t)row * (size_t)E;
        // each lane owns E_PER experts: e = lane + 32 * j (strided so a tie at
        // the same value resolves to the lowest index across lanes too)
        float v[E_PER];
        bool taken[E_PER];
        for (int j = 0; j < E_PER; ++j) {
            const int e = (int)lane + 32 * j;
            v[j] = (e < E) ? lr[e] : -INFINITY;
            taken[j] = (e >= E);
        }
        for (int k = 0; k < K; ++k) {
            // lane-local best: largest value, then lowest index
            float bv = -INFINITY; int bj = -1;
            for (int j = 0; j < E_PER; ++j) {
                if (!taken[j] && (v[j] > bv)) { bv = v[j]; bj = j; }
            }
            const float gmax = simd_max(bv);
            const uint cand = (bv == gmax && bj >= 0) ? (uint)(lane + 32 * bj) : 0xffffffffu;
            const uint gidx = simd_min(cand);
            if (lane == 0) { selv[k] = gmax; seli[k] = gidx; }
            if (gidx == (uint)(lane + 32 * bj) && bj >= 0) { taken[bj] = true; }
        }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float total = 0;
            if (lane < 16) { total = simd_sum(gate_partials[lane]); }
            total = lane == 0 ? total + 0.0f : 0.0f;
            total = simd_sum(total);
            if (lane == 0) { gate[0] = static_cast<T>(total); }
        }
        if (sg != DOT_GROUPS) { return; }
        // softmax_single_row over the K selected logits (AccT = float)
        constexpr int N_READS = 4;
        float ld[N_READS];
        for (int i = 0; i < N_READS; i++) {
            const int p = (int)lane * N_READS + i;
            ld[i] = (p < K) ? selv[p] : -INFINITY;
        }
        float maxval = -FLT_MAX;
        for (int i = 0; i < N_READS; i++) { maxval = (maxval < ld[i]) ? ld[i] : maxval; }
        maxval = simd_max(maxval);
        float normalizer = 0;
        for (int i = 0; i < N_READS; i++) {
            float exp_x = fast::exp(ld[i] - maxval);
            ld[i] = exp_x;
            normalizer += exp_x;
        }
        normalizer = simd_sum(normalizer);
        normalizer = 1 / normalizer;
        for (int i = 0; i < N_READS; i++) {
            const int p = (int)lane * N_READS + i;
            if (p < K) {
                w[(size_t)row * K + p] = ld[i] * normalizer;
                idx[(size_t)row * K + p] = seli[p];
            }
        }
        """#
}
