// MLXFAST-G7: fold `track_moe_route_1` into the gate|up launch.
//
// At decode the routing runs as its own dispatch of ONE threadgroup
// (`TrackFastMoEKernels.route`, `grid: (32, 2, 1)`), immediately in front of
// the gate|up kernel's 1760 threadgroups. Measured on the 48-layer chain that
// stub costs 5.9 us per layer: 3.3 us of its own work, ~1.1 us of barrier, and
// ~1.7 us of pipeline drain from a one-threadgroup kernel standing in front of
// a wide one. Only removing the launch reclaims the last two.
//
// So every gate|up threadgroup walks the router logits itself and derives the
// expert its own slot needs, instead of reading `idx[z]` that the stub wrote.
// The walk is the SAME code as `routeSource`: same lane mapping
// (`e = lane + 32j`), same comparison (`v[j] > bv`, so the lowest j wins a tie
// inside a lane), same `simd_max` on the value and `simd_min` on the index, and
// the same already-taken masking -- only its storage changes from a bool array
// to a bitmask, which is not part of any comparison. Each threadgroup therefore
// reaches the same `selected[]` the stub would have written.
//
// Three details carry the bit-exactness:
//   * a non-writer threadgroup only needs slot `zi`, and round k depends only
//     on rounds < k, so it may leave the loop once k > zi. That early exit does
//     not change `selected[zi]`. It is also most of the win: the naive walk
//     costs 4.3 us, the bounded one 2.3 us.
//   * the softmax over the ten selected logits runs ONCE, on the writer
//     threadgroup (y == 0 && z == 0) in simdgroup 1, character for character as
//     in `routeSource`, with the same `p = lane * 4 + i` lane mapping, so `w`
//     and `idx` are unchanged.
//   * the shared-expert gate GEMV is the same `track_inject_qmv` on simdgroup 0
//     of the same writer threadgroup, so `gate` is unchanged.
//
// The gate|up body below is `gateUpReuseSource` verbatim except that `e` comes
// from the register the walk produced instead of from `idx[z]`.

import Foundation
import MLX
import MLXFast

extension TrackFastMoEKernels {
    /// The router walk, run by every threadgroup, ahead of the gate|up body.
    static func gateUpRoutePrologue(oneSg: Bool) -> String {
        """
        const uint lane7 = thread_index_in_simdgroup;
        const uint sgi7 = simdgroup_index_in_threadgroup;
        const bool writer7 = threadgroup_position_in_grid.y == 0 && threadgroup_position_in_grid.z == 0;
        const int zi = (int)threadgroup_position_in_grid.z;
        if (writer7 && sgi7 == 0) {
            track_inject_qmv<T, GS, BITS, KD, 1, 4>(gwq, gsq, gbq, x, gate, 0u, lane7);
        }
        uint e_sel = 0u;
        \(oneSg ? "threadgroup uint e_tg;" : "")
        if (zi != BR\(oneSg ? " && sgi7 == 0" : "")) {
            constexpr int E_PER = (E + 31) / 32;
            constexpr int N_READS = 4;
            float ld[N_READS];
            uint selected[N_READS];
            for (int i = 0; i < N_READS; ++i) { ld[i] = -INFINITY; selected[i] = 0xffffffffu; }
            const device float* lr = logits;
            float v[E_PER];
            uint taken_m = 0u;
            for (int j = 0; j < E_PER; ++j) {
                const int e = (int)lane7 + 32 * j;
                v[j] = (e < E) ? lr[e] : -INFINITY;
                taken_m |= (e >= E) ? (1u << j) : 0u;
            }
            for (int k = 0; k < K; ++k) {
                if (!writer7 && k > zi) { break; }
                float bv = -INFINITY; int bj = -1;
                for (int j = 0; j < E_PER; ++j) {
                    if (!((taken_m >> j) & 1u) && (v[j] > bv)) { bv = v[j]; bj = j; }
                }
                        const float gmax = simd_max(bv);
                const uint cand = (bv == gmax && bj >= 0) ? (uint)(lane7 + 32 * bj) : 0xffffffffu;
                const uint gidx = simd_min(cand);
                for (int i = 0; i < N_READS; ++i) {
                    if (k == (int)lane7 * N_READS + i) { ld[i] = gmax; selected[i] = gidx; }
                }
                if (gidx == (uint)(lane7 + 32 * bj) && bj >= 0) { taken_m |= (1u << (uint)bj); }
            }
            uint mine = 0xffffffffu;
            for (int i = 0; i < N_READS; ++i) { if (zi == (int)lane7 * N_READS + i) { mine = selected[i]; } }
            e_sel = simd_min(mine);
            if (writer7 && sgi7 == \(oneSg ? 0 : 1)) {
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
                    const int p = (int)lane7 * N_READS + i;
                    if (p < K) { w[p] = ld[i] * normalizer; idx[p] = selected[i]; }
                }
            }
            \(oneSg ? "if (lane7 == 0) { e_tg = e_sel; }" : "")
        }
        \(oneSg ? "threadgroup_barrier(mem_flags::mem_threadgroup); e_sel = e_tg;" : "")

        """
    }

    /// `gateUpReuseSource` with the routed expert taken from the walk's
    /// register instead of the stub's `idx[z]`. Everything else is verbatim.
    /// Four simdgroups per threadgroup: each covers twice as many output rows
    /// as the two-simdgroup form, and only simdgroup 0 runs the walk (the
    /// others wait one barrier for its answer). The walk is identical and
    /// `out_row` only changes WHICH threadgroup computes a row, so no output
    /// element's accumulation order changes.
    static let gateUpRouteSource: String =
        gateUpRoutePrologue(oneSg: true)
        + gateUpReuseSource
        .replacingOccurrences(
            of: "const uint e = shared ? 0u : idx[z];",
            with: "const uint e = shared ? 0u : e_sel;")
        .replacingOccurrences(of: "* (2 * RPS)", with: "* (4 * RPS)")

    nonisolated(unsafe) static let gateUpRouteKernel = MLXFast.metalKernel(
        name: "track_moe_gate_up_route_2row",
        inputNames: [
            "wg", "sg", "bg", "wu", "su", "bu", "wsh", "ssh", "bsh", "x", "xrow",
            "logits", "gwq", "gsq", "gbq",
        ],
        outputNames: ["act", "idx", "w", "gate"],
        source: gateUpRouteSource,
        header: helpersCore + TrackFastKernels.exactHeader
            + TrackFastKernels.mixerHeadHeaderTail + gateUpReuseHelpers,
        ensureRowContiguous: true)

    /// One-token routed gate|up with the routing folded in: returns the same
    /// four arrays `route()` + `gateUpAct()` produce. Returns nil when the call
    /// is not the exact decode shape this kernel was written for, so the caller
    /// keeps the two-launch path.
    static func gateUpRoute(
        wg: MLXArray, sg: MLXArray, bg: MLXArray, wu: MLXArray, su: MLXArray, bu: MLXArray,
        shared: TrackQuantWeight, sharedGate: TrackQuantWeight, x: MLXArray, xrow: MLXArray,
        logits: MLXArray, topK: Int, experts: Int, groupSize: Int, bits: Int
    ) -> (act: MLXArray, idx: MLXArray, w: MLXArray, gate: MLXArray)? {
        let S = x.dim(0), KD = x.dim(1), N = wg.dim(1)
        guard S == 1, x.dtype == .bfloat16, KD == 2560, N == 640,
            groupSize == 32, bits == 4, shared.mode == .affine, shared.rows == 2 * N,
            shared.groupSize == groupSize, shared.bits == bits, shared.biases != nil,
            sharedGate.rows == 1, sharedGate.groupSize == groupSize, sharedGate.bits == bits,
            sharedGate.mode == .affine, sharedGate.biases != nil,
            logits.dtype == .float32, logits.size == experts, experts == 512,
            topK <= 32, topK <= experts, xrow.dtype == .uint32,
            isFast(k: KD, n: N)
        else { return nil }
        let BR = topK
        let rows = gateUpReuseRowsPerSimdgroup
        let inputs: [MLXArray] = [
            wg, sg, bg, wu, su, bu, shared.weight, shared.scales, shared.biases!, x, xrow,
            logits.reshaped(experts), sharedGate.weight, sharedGate.scales, sharedGate.biases!,
        ]
        let template: [(String, any KernelTemplateArg)] = [
            ("T", x.dtype), ("GS", groupSize), ("BITS", bits), ("N", N),
            ("KD", KD), ("BR", BR), ("RPS", rows), ("E", experts), ("K", topK),
        ]
        let outs = gateUpRouteKernel(
            inputs,
            template: template,
            grid: (32, N / rows, BR + 1), threadGroup: (32, 4, 1),
            outputShapes: [[BR + S, N], [S, topK], [S, topK], [S]],
            outputDTypes: [x.dtype, .uint32, .float32, x.dtype])
        return (outs[0], outs[1], outs[2], outs[3])
    }
}
