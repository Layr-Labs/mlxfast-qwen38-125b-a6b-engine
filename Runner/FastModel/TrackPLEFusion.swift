// MLXFAST-PLEFUSE2: S=1 PLE fusion. No projection or weight-layout changes.
// Scratch: ONE reused float[32] (128 B) in prepare, ZERO in convolution.
// This retains the RMS/reduction lane layout and the dilated convolution's
// channel-per-threadgroup layout; token tolerance, not bit equality, applies.
import Foundation
import MLX

enum TrackPLEFusion {
    static let header = """
        // MLXFAST-PLEPAIR: 320 physical threads emulate the original 640-thread
        // reduction geometry. Each SIMD group produces its own partial and the
        // corresponding partial from the logical group ten positions later.
        METAL_FUNC float ple_row_sum_pair(
            float acc0, float acc1, threadgroup float* partials, uint lane, uint sg) {
            acc0 = simd_sum(acc0);
            acc1 = simd_sum(acc1);
            if (sg == 0 && lane >= 20) { partials[lane] = 0.0f; }
            if (lane == 0) {
                partials[sg] = acc0;
                partials[sg + 10] = acc1;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc0 = simd_sum(partials[lane]);
            // All readers finish before the next reduction reuses the buffer.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            return acc0;
        }
        """

    static let prepareSource = """
        // MLXFAST-PLEPAIR: all three group norms, dot, gate, and concat.
        // The two logical owners preserve the 640-thread reduction association.
        constexpr uint H = 2560;
        constexpr uint W = 4 * H;
        const uint hc = threadgroup_position_in_grid.y;
        const uint lid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint d0 = lid * 4;
        const uint d1 = d0 + H / 2;
        const uint base0 = hc * H + d0;
        const uint base1 = hc * H + d1;
        threadgroup float partials[32];  // 128 B total, reused throughout.
        const float eps = as_type<float>((uint)EPS_BITS);

        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const float k0 = float(key[base0 + i]);
            const float k1 = float(key[base1 + i]);
            acc0 += k0 * k0;
            acc1 += k1 * k1;
        }
        const float ik = metal::precise::rsqrt(
            ple_row_sum_pair(acc0, acc1, partials, lane, sg) / float(H) + eps);
        acc0 = 0.0f;
        acc1 = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            const float q0 = float(query[base0 + i]);
            const float q1 = float(query[base1 + i]);
            acc0 += q0 * q0;
            acc1 += q1 * q1;
        }
        const float iq = metal::precise::rsqrt(
            ple_row_sum_pair(acc0, acc1, partials, lane, sg) / float(H) + eps);

        // Keep the original dtype boundaries and four-element logical folds.
        InT dot0 = InT(0);
        InT dot1 = InT(0);
        for (uint i = 0; i < 4; ++i) {
            InT k0 = InT(float(key[base0 + i]) * ik);
            k0 = k0 * keyScale[base0 + i];
            InT q0 = InT(float(query[base0 + i]) * iq);
            q0 = q0 * queryScale[base0 + i];
            InT product0 = k0 * q0;
            dot0 = product0 + dot0;

            InT k1 = InT(float(key[base1 + i]) * ik);
            k1 = k1 * keyScale[base1 + i];
            InT q1 = InT(float(query[base1 + i]) * iq);
            q1 = q1 * queryScale[base1 + i];
            InT product1 = k1 * q1;
            dot1 = product1 + dot1;
        }
        dot0 = InT(0) + dot0;
        dot1 = InT(0) + dot1;
        dot0 = simd_sum(dot0);
        dot1 = simd_sum(dot1);
        if (sg == 0 && lane >= 20) { partials[lane] = 0.0f; }
        if (lane == 0) {
            partials[sg] = float(dot0);
            partials[sg + 10] = float(dot1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot0 = simd_sum(InT(partials[lane]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        InT gate = dot0 / InT(as_type<float>((uint)DIVISOR_BITS));
        InT magnitude = metal::abs(gate);
        magnitude = metal::max(magnitude, InT(1e-6f));
        magnitude = metal::sqrt(magnitude);
        InT direction = InT((gate > InT(0)) - (gate < InT(0)));
        gate = magnitude * direction;
        const InT activation = mlx_sigmoid(gate);

        InT g0[4];
        InT g1[4];
        acc0 = 0.0f;
        acc1 = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            g0[i] = activation * value[d0 + i];
            g1[i] = activation * value[d1 + i];
            gated[base0 + i] = g0[i];
            gated[base1 + i] = g1[i];
            const float v0 = float(g0[i]);
            const float v1 = float(g1[i]);
            acc0 += v0 * v0;
            acc1 += v1 * v1;
        }
        const float iv = metal::precise::rsqrt(
            ple_row_sum_pair(acc0, acc1, partials, lane, sg) / float(H) + eps);
        for (uint i = 0; i < 4; ++i) {
            InT n0 = InT(float(g0[i]) * iv);
            InT n1 = InT(float(g1[i]) * iv);
            full[9 * W + base0 + i] = n0 * convScale[base0 + i];
            full[9 * W + base1 + i] = n1 * convScale[base1 + i];
        }
        // Same [old nine rows, new row] layout as concatenated. State staging
        // keeps its existing tail view, including capture/rollback behavior.
        for (uint t = 0; t < 9; ++t) {
            for (uint i = 0; i < 4; ++i) {
                full[t * W + base0 + i] = convState[t * W + base0 + i];
                full[t * W + base1 + i] = convState[t * W + base1 + i];
            }
        }
        """

    private static func replaceOnce(_ source: String, _ old: String, _ new: String) -> String {
        let pieces = source.components(separatedBy: old)
        precondition(pieces.count == 2, "PLE fused prepare anchor must be unique")
        return pieces[0] + new + pieces[1]
    }

    // The fused prepare keeps the checked source and changes only its final epilogue.
    private static let fusedPrepareSource: String = {
        let old = """
        for (uint i = 0; i < 4; ++i) {
            InT n0 = InT(float(g0[i]) * iv);
            InT n1 = InT(float(g1[i]) * iv);
            full[9 * W + base0 + i] = n0 * convScale[base0 + i];
            full[9 * W + base1 + i] = n1 * convScale[base1 + i];
        }
        """
        let new = """
        for (uint i = 0; i < 4; ++i) {
            const uint c0 = base0 + i;
            const uint c1 = base1 + i;
            InT n0 = InT(float(g0[i]) * iv);
            InT n1 = InT(float(g1[i]) * iv);
            const InT newest0 = n0 * convScale[c0];
            const InT newest1 = n1 * convScale[c1];
            full[9 * W + c0] = newest0;
            full[9 * W + c1] = newest1;

            float conv0 = 0.0f;
            float conv1 = 0.0f;
            {
            #pragma clang fp reassociate(off)
            #pragma clang fp contract(off)
                const float p00 = float(convState[c0]) * float(convW[c0 * 4 + 0]);
                const float p01 = float(convState[3 * W + c0]) * float(convW[c0 * 4 + 1]);
                const float p02 = float(convState[6 * W + c0]) * float(convW[c0 * 4 + 2]);
                const float p03 = float(newest0) * float(convW[c0 * 4 + 3]);
                conv0 = p00;
                conv0 += p01;
                conv0 += p02;
                conv0 += p03;

                const float p10 = float(convState[c1]) * float(convW[c1 * 4 + 0]);
                const float p11 = float(convState[3 * W + c1]) * float(convW[c1 * 4 + 1]);
                const float p12 = float(convState[6 * W + c1]) * float(convW[c1 * 4 + 2]);
                const float p13 = float(newest1) * float(convW[c1 * 4 + 3]);
                conv1 = p10;
                conv1 += p11;
                conv1 += p12;
                conv1 += p13;
            }

            const InT pleDelta0 = g0[i] + mlx_silu(InT(conv0));
            const InT pleDelta1 = g1[i] + mlx_silu(InT(conv1));
            added[c0] = query[c0] + pleDelta0;
            added[c1] = query[c1] + pleDelta1;
        }
        """
        let withoutGated0 = replaceOnce(
            prepareSource, "    gated[base0 + i] = g0[i];\n", "")
        let withoutGated = replaceOnce(
            withoutGated0, "    gated[base1 + i] = g1[i];\n", "")
        return replaceOnce(withoutGated, old, new)
    }()

    static let convolutionSource = """
        // MLXFAST-PLEFUSE2: preserve the S=1 dilated implicit-GEMM dispatch:
        // group=(32,1,4), grid of groups=(1,1,W), channel=group.z.
        // Its small-channel loaders assign taps 0..3 to SIMD 0 lanes 0..3;
        // lane 0 owns the single valid output. Keep those owners, discard
        // the padded matrix work, and require no threadgroup storage.
        constexpr uint W = 10240;
        const uint c = threadgroup_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        if (simdgroup_index_in_threadgroup != 0) { return; }
        float product = 0.0f;
        if (lane < 4) {
            product = float(full[(lane * 3) * W + c]) * float(weight[c * 4 + lane]);
        }
        // FP32 products and chronological FP32 additions, taps 0,1,2,3.
        // Shuffle broadcasts retain the weight-loading lanes. Unlike a
        // padded simdgroup MMA, this has no matrix accumulator or spill array.
        float acc = simd_broadcast(product, 0);
        acc += simd_broadcast(product, 1);
        acc += simd_broadcast(product, 2);
        acc += simd_broadcast(product, 3);
        if (lane == 0) {
            InT convolved = InT(acc);
            InT activated = mlx_silu(convolved);
            out[c] = gated[c] + activated;
        }
        """

    static let prepareKernel = MLXFast.metalKernel(
        name: "track_ple_prepare_fuse2_pair",
        inputNames: ["key", "query", "value", "keyScale", "queryScale", "convScale", "convState"],
        outputNames: ["gated", "full"], source: prepareSource,
        header: TrackFastKernels.exactHeader + header, ensureRowContiguous: true)

    static let convolutionKernel = MLXFast.metalKernel(
        name: "track_ple_convolution_fuse2", inputNames: ["full", "weight", "gated"],
        outputNames: ["out"], source: convolutionSource,
        header: TrackFastKernels.exactHeader, ensureRowContiguous: true)

    static let fusedPrepareKernel = MLXFast.metalKernel(
        name: "track_ple_prepare_conv_fused2_residual_pair",
        inputNames: ["key", "query", "value", "keyScale", "queryScale", "convScale", "convState", "convW"],
        outputNames: ["full", "added"], source: fusedPrepareSource,
        header: TrackFastKernels.exactHeader + header, ensureRowContiguous: true)

    static func supports(_ p: TrackPLE, stream: MLXArray, hidden: Int, hcCount: Int) -> Bool {
        // This predicate runs once per decode token. Spell the three fixed dtype
        // and scale checks directly rather than constructing temporary arrays.
        let dtype = stream.dtype
        return hidden == 2560 && hcCount == 4 && stream.shape == [1, 1, 10240]
            && (dtype == .bfloat16 || dtype == .float16 || dtype == .float32)
            && p.dilation == 3 && p.stateLength == 9
            && p.keyProj.rows == 10240 && p.valueProj.rows == 2560
            && p.convW.shape == [10240, 4, 1] && p.convW.dtype == dtype
            && p.normKeyScale.shape == [10240] && p.normKeyScale.dtype == dtype
            && p.normQueryScale.shape == [10240] && p.normQueryScale.dtype == dtype
            && p.normConvScale.shape == [10240] && p.normConvScale.dtype == dtype
    }

    private static func prepareTemplates(dtype: DType, eps: Float)
        -> [(String, any KernelTemplateArg)]
    {
        [
            ("InT", dtype),
            ("EPS_BITS", Int(eps.bitPattern)),
            ("DIVISOR_BITS", Int(Foundation.sqrt(Float(2560)).bitPattern)),
        ]
    }

    static func forwardProjected(
        _ p: TrackPLE, key: MLXArray, value: MLXArray, stream: MLXArray,
        convState: MLXArray, eps: Float, fusedResidual: Bool
    ) -> (full: MLXArray, output: MLXArray, residualAdded: Bool)? {
        guard key.shape == [1, 1, 10240], value.shape == [1, 1, 2560],
            key.dtype == stream.dtype, value.dtype == stream.dtype
        else { return nil }
        if fusedResidual {
            let r = fusedPrepareKernel(
                [key, stream, value, p.normKeyScale, p.normQueryScale,
                 p.normConvScale, convState, p.convW],
                template: prepareTemplates(dtype: stream.dtype, eps: eps),
                grid: (320, 4, 1), threadGroup: (320, 1, 1),
                outputShapes: [[1, 10, 10240], [1, 1, 10240]],
                outputDTypes: [stream.dtype, stream.dtype])
            return (r[0], r[1], true)
        }
        let r = prepareKernel(
            [key, stream, value, p.normKeyScale, p.normQueryScale, p.normConvScale, convState],
            template: prepareTemplates(dtype: stream.dtype, eps: eps),
            grid: (320, 4, 1), threadGroup: (320, 1, 1),
            outputShapes: [[1, 1, 10240], [1, 10, 10240]],
            outputDTypes: [stream.dtype, stream.dtype])
        let output = convolutionKernel(
            [r[1], p.convW, r[0]], template: [("InT", stream.dtype)],
            grid: (32, 1, 4 * 10240), threadGroup: (32, 1, 4),
            outputShapes: [[1, 1, 10240]], outputDTypes: [stream.dtype])[0]
        return (r[1], output, false)
    }
}
