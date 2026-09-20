import Foundation
import MLX

/// Serial PLE including input injection and the final stream normalization.
enum TrackPLEWholeDecode {
    static let source = """
        constexpr uint H = 2560;
        constexpr uint W = 4 * H;
        const uint hc = threadgroup_position_in_grid.y;
        const uint lid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint d = lid * 4;
        const uint base = hc * H + d;
        threadgroup float partials[32];  // 128 B total, reused throughout.
        const float eps = as_type<float>((uint)EPS_BITS);

        InT r[4];
        for (uint i = 0; i < 4; ++i) {
            const InT sp = pending[d + i] * inject[hc];
            r[i] = residual[base + i] + sp;
        }
        // Reload key elements after their norm; retain the four injected
        // query values until the final residual addition.
        float acc = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            float k = float(key[base + i]);
            acc += k * k;
        }
        const float ik = metal::precise::rsqrt(
            ple_row_sum(acc, partials, lane, sg) / float(H) + eps);
        acc = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            float q = float(r[i]);
            acc += q * q;
        }
        const float iq = metal::precise::rsqrt(
            ple_row_sum(acc, partials, lane, sg) / float(H) + eps);

        // Keep the original dtype boundaries: round RMS before scale, round
        // product before sum, and use row_reduce_looped's four-element fold.
        InT dot = InT(0);
        for (uint i = 0; i < 4; ++i) {
            InT k = InT(float(key[base + i]) * ik);
            k = k * keyScale[base + i];
            InT q = InT(float(r[i]) * iq);
            q = q * queryScale[base + i];
            InT product = k * q;
            dot = product + dot;
        }
        dot = InT(0) + dot;
        dot = simd_sum(dot);
        if (sg == 0 && lane >= 20) { partials[lane] = 0.0f; }
        if (lane == 0) { partials[sg] = float(dot); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot = simd_sum(InT(partials[lane]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        InT gate = dot / InT(as_type<float>((uint)DIVISOR_BITS));
        InT magnitude = metal::abs(gate);
        magnitude = metal::max(magnitude, InT(1e-6f));
        magnitude = metal::sqrt(magnitude);
        InT direction = InT((gate > InT(0)) - (gate < InT(0)));
        gate = magnitude * direction;
        const InT activation = mlx_sigmoid(gate);

        // Retain four gated values alongside the four injected residuals.
        // There is no private full-row scratch.
        InT g[4];
        acc = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            g[i] = activation * value[d + i];
            float v = float(g[i]);
            acc += v * v;
        }
        const float iv = metal::precise::rsqrt(
            ple_row_sum(acc, partials, lane, sg) / float(H) + eps);
        float finalValues[4];
        acc = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const uint c = base + i;
            InT n = InT(float(g[i]) * iv);
            const InT newest = n * convScale[c];
            nextState[8 * W + c] = newest;
            float cv;
            {
                #pragma clang fp contract(off)
                cv = float(convState[c]) * float(convWeight[c * 4]);
                cv += float(convState[3 * W + c]) * float(convWeight[c * 4 + 1]);
                cv += float(convState[6 * W + c]) * float(convWeight[c * 4 + 2]);
                cv += float(newest) * float(convWeight[c * 4 + 3]);
            }
            const InT convolved = InT(cv);
            const InT activated = mlx_silu(convolved);
            const InT delta = g[i] + activated;
            const InT result = r[i] + delta;
            outStream[c] = result;
            finalValues[i] = float(result);
            acc += finalValues[i] * finalValues[i];
        }
        const float invFinal = metal::precise::rsqrt(
            ple_row_sum(acc, partials, lane, sg) / float(H) + eps);
        for (uint i = 0; i < 4; ++i) {
            const InT normalized = InT(finalValues[i] * invFinal);
            outNorm[base + i] = normalized * finalScale[base + i];
        }
        for (uint t = 0; t < 8; ++t) {
            for (uint i = 0; i < 4; ++i) {
                nextState[t * W + base + i] = convState[(t + 1) * W + base + i];
            }
        }
        """

    private static let kernel = MLXFast.metalKernel(
        name: "track_ple_whole_decode",
        inputNames: ["key", "residual", "pending", "inject", "value", "keyScale",
                     "queryScale", "convScale", "convState", "convWeight", "finalScale"],
        outputNames: ["nextState", "outStream", "outNorm"], source: source,
        header: TrackFastKernels.exactHeader + TrackPLEFusion.header,
        ensureRowContiguous: true)

    static func apply(
        _ p: TrackPLE, key: MLXArray, value: MLXArray, residual: MLXArray,
        pending: MLXArray, inject: MLXArray, convState: MLXArray,
        finalScale: MLXArray, eps: Float
    ) -> (full: MLXArray, stream: MLXArray, normed: MLXArray) {
        let result = kernel(
            [key, residual, pending, inject, value, p.normKeyScale, p.normQueryScale,
             p.normConvScale, convState, p.convW, finalScale],
            template: [("InT", residual.dtype), ("EPS_BITS", Int(eps.bitPattern)),
                       ("DIVISOR_BITS", Int(Foundation.sqrt(Float(2560)).bitPattern))],
            grid: (640, 4, 1), threadGroup: (640, 1, 1),
            outputShapes: [[1, 9, 10240], [1, 1, 10240], [1, 1, 10240]],
            outputDTypes: [residual.dtype, residual.dtype, residual.dtype])
        return (result[0], result[1], result[2])
    }
}
