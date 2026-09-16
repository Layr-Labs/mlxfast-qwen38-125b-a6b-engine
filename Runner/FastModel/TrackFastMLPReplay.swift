import Foundation
import MLX

/// The serial MLP half of a layer as one cached MLX graph. All numerical work
/// remains in the same kernels; weights are captured from the immutable model.
enum TrackFastMLPReplay {
    nonisolated(unsafe) static var enabled =
        ProcessInfo.processInfo.environment["TRACK_MLP_REPLAY"] != "0"

    static func make(hc: TrackHC, moe: TrackMoE, hcCount: Int, hidden: Int, eps: Float)
        -> (@Sendable ([MLXArray]) -> [MLXArray])?
    {
        guard case .quant(let down) = hc.down, case .quant(let up) = hc.up,
            down.biases != nil, up.biases != nil else { return nil }
        let inj: TrackQuantWeight?
        if hc.hasInject {
            guard case .quant(let q)? = hc.inject, q.biases != nil else { return nil }
            inj = q
        } else { inj = nil }
        let packedUp = hc.decodeUp
        return compile(shapeless: false) { inputs in
            let n = TrackFastKernels.injectNorm(
                residual: inputs[0], out: inputs[1], inject: inputs[2], scale: hc.normScaleQ,
                hcCount: hcCount, hidden: hidden, eps: eps, tile: false)
            let normed = n.normed.reshaped(1, hcCount * hidden)
            let d = TrackFastMixerKernels.downInject(normed: normed, down: down, inject: inj)
            let u = TrackFastMixerKernels.upMix(
                act: d.act, normed: normed, up: packedUp ?? up, inj: d.inj,
                hcCount: hcCount, hidden: hidden, hasInject: hc.hasInject,
                emitF32: true, packedRows: packedUp != nil)
            let out = TrackQwen4ExpFastModel.moeForwardShared(
                moe, u.input.reshaped(1, 1, hidden),
                inputF32: u.inputF32.reshaped(1, 1, hidden), replay: nil)
            return [n.stream, out, u.inject.reshaped(1, 1, hcCount)]
        }
    }
}
