import MLX

/// Stable Top-10 over 512 logits. Each lane owns a fixed 16-leaf tournament;
/// deleting a winner changes only its four ancestors. No floating-point
/// arithmetic is performed on logits before the original softmax epilogue.
enum TrackRouteTournament {
    static func apply(logits: MLXArray, gateType: DType) -> [MLXArray] {
        let rows = logits.size / 512
        return kernel(
            [logits.reshaped(rows, 512)], template: [("T", gateType)],
            grid: (32, rows, 1), threadGroup: (32, 1, 1),
            outputShapes: [[rows, 10], [rows, 10], [rows]],
            outputDTypes: [.uint32, .float32, gateType])
    }

    private static let header = #"""
    struct TrackRouteNode { float value; uint index; };
    METAL_FUNC TrackRouteNode track_route_leaf(float value, uint index) {
      // Match the strict > -INFINITY initialization of the original scan:
      // NaN and -INFINITY never supply an eligible index; +INFINITY may win.
      return value > -INFINITY ? TrackRouteNode{value, index}
                              : TrackRouteNode{-INFINITY, 0xffffffffu};
    }
    METAL_FUNC TrackRouteNode track_route_best(TrackRouteNode a, TrackRouteNode b) {
      return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
    }
    """#

    private static let source: String = {
        var s = #"""
        constexpr int K = 10;
        constexpr int N_READS = 4;
        const uint row = threadgroup_position_in_grid.y;
        const uint lane = thread_index_in_simdgroup;
        const device float* lr = logits + size_t(row) * 512;
        float ld[N_READS];
        uint selected[N_READS];
        for (int i = 0; i < N_READS; ++i) {
            ld[i] = -INFINITY;
            selected[i] = 0xffffffffu;
        }
        """# + "\n"
        // Named scalars and constant child references avoid a runtime-indexed
        // per-thread tree array. The host generates only shader source text.
        for j in 0..<16 {
            s += "TrackRouteNode n\(16 + j) = track_route_leaf(lr[lane + \(32 * j)], lane + \(32 * j));\n"
        }
        for n in stride(from: 15, through: 1, by: -1) {
            s += "TrackRouteNode n\(n) = track_route_best(n\(2 * n), n\(2 * n + 1));\n"
        }
        s += #"""
        for (int k = 0; k < K; ++k) {
            const float gmax = simd_max(n1.value);
            const uint cand = n1.value == gmax ? n1.index : 0xffffffffu;
            const uint gidx = simd_min(cand);
            for (int i = 0; i < N_READS; ++i) {
                if (k == int(lane) * N_READS + i) { ld[i] = gmax; selected[i] = gidx; }
            }
            if (k + 1 < K && gidx != 0xffffffffu && lane == (gidx & 31u)) {
                switch (gidx >> 5) {
        """# + "\n"
        for j in 0..<16 {
            s += "case \(j):\n n\(16 + j) = TrackRouteNode{-INFINITY, 0xffffffffu};\n"
            var n = (16 + j) / 2
            while n > 0 {
                s += " n\(n) = track_route_best(n\(2 * n), n\(2 * n + 1));\n"
                n /= 2
            }
            s += " break;\n"
        }
        s += #"""
                }
            }
        }
        // The promoted softmax_single_row expression and lane ownership.
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
                idx[(size_t)row * K + p] = selected[i];
            }
        }
        // The gate is unused by no-gate callers; define it rather than expose
        // an unwritten output. The existing three-array API remains unchanged.
        if (lane == 0) { gate[row] = T(0); }
        """#
        return s
    }()

    private static let kernel = MLXFast.metalKernel(
        name: "track_moe_route_tournament", inputNames: ["logits"],
        outputNames: ["idx", "w", "gate"], source: source, header: header,
        ensureRowContiguous: true)
}
