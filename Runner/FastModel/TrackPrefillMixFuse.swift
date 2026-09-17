import Foundation
import MLX
import MLXFast

extension TrackPrefillMixFuse {
    /// MLXFAST-MIXFUSE: the up projection and the hyper-connection mix in one
    /// launch. The shipped pair writes `w` [S, HC*H] out of the GEMM and reads
    /// every byte of it straight back in `hcMix`; at S = 1024 that is 21 MB
    /// written and 21 MB read per sublayer, for a value no other consumer sees.
    /// Here the GEMM's output tile is rounded to the activation dtype exactly as
    /// the device store would, staged in the weight buffer the K loop has
    /// finished with, and folded in place.
    ///
    /// The fold needs all four streams of a hidden column in one threadgroup,
    /// which is why it runs over the packed `decodeUp` rows: eight consecutive
    /// packed rows are two hidden columns times four streams, so a 64-wide
    /// column tile holds sixteen complete hidden columns.
    static let source = #"""
        alignas(16) threadgroup T Ws[64 * 72];
        track_mix_fused<T, 32, 4, true, 64, 64, 64, 2, 2, 0, HAS_INJ>(
            w, scales, biases, x, normed, inj, y, inject, Ws, K, N, M, HH, LW, HC,
            threadgroup_position_in_grid, thread_index_in_threadgroup,
            simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
        """#

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "track_prefill_mix_fuse",
        inputNames: ["w", "scales", "biases", "x", "normed", "inj"],
        outputNames: ["y", "inject"], source: source,
        header: TrackPrefillIndirect.metalHeader + TrackFastKernels.exactHeader + fuseHeader,
        ensureRowContiguous: true)

    /// `act [S, LW]`, packed up weight `[HC*H, LW]`, `normed [S, HC*H]`,
    /// `inj [S, LW]` -> (`input [S, H]`, `inject [S, HC]`).
    static func apply(
        act: MLXArray, up: TrackQuantWeight, normed: MLXArray, inj: MLXArray,
        hidden H: Int, hcCount: Int, hasInject: Bool
    ) -> (input: MLXArray, inject: MLXArray)? {
        // `K` is the GEMM contraction (the mixer low rank). `injStride` is the
        // row stride of the inject buffer, which `hcMix` reads as its `LW`:
        // with an inject weight it is `hcCount`, otherwise the low rank.
        let S = act.dim(0), K = act.dim(1), N = up.rows
        let injStride = inj.dim(1)
        guard N == hcCount * H, hcCount == 4, H % 2 == 0,
            up.bits == 4, up.groupSize == 32, up.biases != nil, up.mode == .affine,
            act.dtype == .bfloat16, normed.dtype == act.dtype, inj.dtype == act.dtype,
            normed.dim(0) == S, normed.dim(1) == N, inj.dim(0) == S,
            injStride >= hcCount, K % 64 == 0, N % 64 == 0, S >= 64
        else { return nil }
        let o = kernel(
            [up.weight, up.scales, up.biases!, act, normed, inj],
            template: [
                ("T", act.dtype), ("K", K), ("N", N), ("M", S), ("HH", H),
                ("LW", injStride), ("HC", hcCount), ("HAS_INJ", hasInject),
            ],
            grid: ((N / 64) * 32, ((S + 63) / 64) * 2, 2), threadGroup: (32, 2, 2),
            outputShapes: [[S, H], [S, hcCount]], outputDTypes: [act.dtype, act.dtype])
        return (o[0], o[1])
    }
}
