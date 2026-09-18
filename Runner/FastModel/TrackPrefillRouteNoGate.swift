import MLX

// MARK: dedicated no-gate route kernel for wide (prefill) windows
//
// `route` carries the shared-expert gate as an optional fused GEMV: five inputs
// (`x` plus the gate weight/scales/biases, all bound to `x` when the gate is
// absent), a third `gate` output, and the qmv helper headers the gate body
// needs. This checkpoint's shared gate is a dense bf16 Linear, so the wide
// path always instantiates `HAS_GATE = false` -- the gate body compiles out,
// but every dispatch still binds four dead inputs, allocates a dead `[R]`
// output, and hashes the full qmv helper header into the library key.
//
// This specialization keeps the top-K walk and the K-row softmax bit-for-bit
// (same lane -> expert stride, same simd_max/simd_min winner rule, same
// `fast::exp` accumulation order) and drops everything the gate pulled in:
// one input, two outputs, no helper header, one simdgroup per row.

extension TrackFastMoEKernels {
    /// logits f32 [R, E] -> idx uint32 [R, K], w f32 [R, K]. No gate.
    static let routeNoGateSource = """
        constexpr int E_PER = (E + 31) / 32;
        const uint row = threadgroup_position_in_grid.y;
        const uint lane = thread_index_in_simdgroup;
        constexpr int N_READS = 4;
        float ld[N_READS];
        uint selected[N_READS];
        for (int i = 0; i < N_READS; ++i) {
            ld[i] = -INFINITY;
            selected[i] = 0xffffffffu;
        }
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
            for (int i = 0; i < N_READS; ++i) {
                if (k == (int)lane * N_READS + i) { ld[i] = gmax; selected[i] = gidx; }
            }
            if (gidx == (uint)(lane + 32 * bj) && bj >= 0) { taken[bj] = true; }
        }
        // softmax_single_row over the K selected logits (AccT = float)
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
        """

    nonisolated(unsafe) static let routeNoGateKernel = MLXFast.metalKernel(
        name: "track_moe_route_nogate",
        inputNames: ["logits"],
        outputNames: ["idx", "w"],
        source: routeNoGateSource, ensureRowContiguous: true)

    /// logits f32 [..., E] -> (idx uint32 [..., K], w f32 [..., K])
    static func routeNoGate(logits: MLXArray, topK: Int) -> (idx: MLXArray, w: MLXArray) {
        precondition(logits.dtype == .float32)
        let E = logits.dim(-1)
        let lead = Array(logits.shape.dropLast())
        let R = lead.reduce(1, *)
        precondition(topK <= 32 && topK <= E && R >= 1)
        let outs = routeNoGateKernel(
            [logits.reshaped(R, E)],
            template: [("E", E), ("K", topK)],
            grid: (32, R, 1), threadGroup: (32, 1, 1),
            outputShapes: [[R, topK], [R, topK]], outputDTypes: [.uint32, .float32])
        return (outs[0].reshaped(lead + [topK]), outs[1].reshaped(lead + [topK]))
    }
}
