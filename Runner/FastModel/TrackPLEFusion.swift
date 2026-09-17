// MLXFAST-PLEFUSE3: S=1 PLE fusion. No projection or weight-layout changes.
// Scratch: ONE reused float[32] (128 B) in prepare and no second launch:
// the dilated convolution folds into the prepare epilogue because its taps
// are convState rows 0/3/6 plus the normed row the same threadgroup made.
// This retains the RMS/reduction lane layout and the conv's FP32 tap order;
// token tolerance, not bit equality, applies.
import Foundation
import MLX

enum TrackPLEFusion {
    static let header = """
        // MLXFAST-PLEFUSE2: rms_single_row's four adjacent elements per thread,
        // 640 threads / 20 SIMD groups. Reuse its existing 128-byte buffer.
        METAL_FUNC float ple_row_sum(
            float acc, threadgroup float* partials, uint lane, uint sg) {
            acc = simd_sum(acc);
            if (sg == 0 && lane >= 20) { partials[lane] = 0.0f; }
            if (lane == 0) { partials[sg] = acc; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc = simd_sum(partials[lane]);
            // All readers finish before the next reduction reuses the buffer.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return acc;
        }
        """

    static let prepareSource = """
        // MLXFAST-PLEFUSE3: all three group norms, dot, gate, concat, conv.
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

        // Reload the four key/query elements after their norms instead of
        // keeping both rows live across the reductions.
        float acc = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            float k = float(key[base + i]);
            acc += k * k;
        }
        const float ik = metal::precise::rsqrt(
            ple_row_sum(acc, partials, lane, sg) / float(H) + eps);
        acc = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            float q = float(query[base + i]);
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
            InT q = InT(float(query[base + i]) * iq);
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

        // Only four gated values survive this last reduction (8 B/thread
        // for bf16/f16, 16 B for f32); no private full-row scratch.
        InT g[4];
        acc = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            g[i] = activation * value[d + i];
            float v = float(g[i]);
            acc += v * v;
        }
        const float iv = metal::precise::rsqrt(
            ple_row_sum(acc, partials, lane, sg) / float(H) + eps);
        InT r9[4];
        for (uint i = 0; i < 4; ++i) {
            InT n = InT(float(g[i]) * iv);
            r9[i] = n * convScale[base + i];
            full[9 * W + base + i] = r9[i];
        }
        // Same [old nine rows, new row] layout as concatenated. State staging
        // keeps its existing tail view, including capture/rollback behavior;
        // that view evicts row 0, so only rows 1..8 are written.
        for (uint t = 1; t < 9; ++t) {
            for (uint i = 0; i < 4; ++i) {
                full[t * W + base + i] = convState[t * W + base + i];
            }
        }
        // The dilated conv, folded in: taps are convState rows 0/3/6 and the
        // normed row above. FP32 products in ascending tap order, one InT
        // rounding, silu, and the gated residual -- the same arithmetic the
        // separate convolution kernel ran, with no gated intermediate.
        for (uint i = 0; i < 4; ++i) {
            const uint c = base + i;
            float cacc = 0.0f;
            cacc += float(convState[c]) * float(weight[c * 4]);
            cacc += float(convState[3 * W + c]) * float(weight[c * 4 + 1]);
            cacc += float(convState[6 * W + c]) * float(weight[c * 4 + 2]);
            cacc += float(r9[i]) * float(weight[c * 4 + 3]);
            out[c] = g[i] + mlx_silu(InT(cacc));
        }
        """

    static let prepareKernel = MLXFast.metalKernel(
        name: "track_ple_prepare_fuse3",
        inputNames: [
            "key", "query", "value", "keyScale", "queryScale", "convScale",
            "convState", "weight",
        ],
        outputNames: ["full", "out"], source: prepareSource,
        header: TrackFastKernels.exactHeader + header, ensureRowContiguous: true)

    static func supports(_ p: TrackPLE, stream: MLXArray, hidden: Int, hcCount: Int) -> Bool {
        // Guard the exact geometry; every other path builds the original chain.
        hidden == 2560 && hcCount == 4 && stream.shape == [1, 1, 10240]
            && [.bfloat16, .float16, .float32].contains(stream.dtype)
            && p.dilation == 3 && p.stateLength == 9
            && p.keyProj.rows == 10240 && p.valueProj.rows == 2560
            && p.convW.shape == [10240, 4, 1] && p.convW.dtype == stream.dtype
            && [p.normKeyScale, p.normQueryScale, p.normConvScale].allSatisfy {
                $0.shape == [10240] && $0.dtype == stream.dtype
            }
    }

    static func forward(
        _ p: TrackPLE, embedded: MLXArray, stream: MLXArray, convState: MLXArray, eps: Float
    ) -> (full: MLXArray, output: MLXArray)? {
        // The original two projections stay separate, with unchanged kernels,
        // quantization, tiling, and weight-loading lane ownership.
        let key = p.keyProj.apply(embedded)
        let value = p.valueProj.apply(embedded)
        guard key.shape == [1, 1, 10240], value.shape == [1, 1, 2560],
            key.dtype == stream.dtype, value.dtype == stream.dtype
        else { return nil }
        let r = prepareKernel(
            [key, stream, value, p.normKeyScale, p.normQueryScale, p.normConvScale, convState,
             p.convW],
            template: [("InT", stream.dtype), ("EPS_BITS", Int(eps.bitPattern)),
                       ("DIVISOR_BITS", Int(Foundation.sqrt(Float(2560)).bitPattern))],
            grid: (640, 4, 1), threadGroup: (640, 1, 1),
            outputShapes: [[1, 10, 10240], [1, 1, 10240]],
            outputDTypes: [stream.dtype, stream.dtype])
        return (r[0], r[1])
    }
}
