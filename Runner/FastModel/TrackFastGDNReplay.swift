import Foundation
import MLX

/// Pure serial mixer + GDN graph. Recurrent input and output stay explicit;
/// the caller continues to own and commit the request-state ledger.
enum TrackFastGDNReplay {
    nonisolated(unsafe) static var enabled =
        ProcessInfo.processInfo.environment["TRACK_GDN_REPLAY"] != "0"

    static func make(hc: TrackHC, gdn g: TrackGDN, hcCount: Int, hidden: Int)
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
            let normed = inputs[0].reshaped(1, hcCount * hidden)
            let d = TrackFastMixerKernels.downInject(normed: normed, down: down, inject: inj)
            let u = TrackFastMixerKernels.upMix(
                act: d.act, normed: normed, up: packedUp ?? up, inj: d.inj,
                hcCount: hcCount, hidden: hidden, hasInject: hc.hasInject,
                emitF32: false, packedRows: packedUp != nil)
            let proj = g.proj.apply(u.input.reshaped(1, 1, hidden))
            let gated: MLXArray, conv: MLXArray, ssm: MLXArray
            if let r = TrackFastGDNDecode.apply(
                proj: proj, convState: inputs[1], convW: g.convW,
                negExpALog: g.negExpALog, dtBias: g.dtBias, stateIn: inputs[2], normW: g.normW,
                zOffset: g.zOffset, eps: 1e-6, capture: false, geometry: g.geometry)
            {
                (gated, conv, ssm) = (r.gated, r.convOut, r.stateOut)
            } else {
                let r = TrackFastKernels.gdn(
                    proj: proj, convState: inputs[1], convW: g.convW,
                    negExpALog: g.negExpALog, dtBias: g.dtBias, stateIn: inputs[2],
                    T: 1, capture: false, geometry: g.geometry)
                gated = TrackFastKernels.gatedRMS(
                    y: r.y, proj: proj, w: g.normW, zOffset: g.zOffset, eps: 1e-6)
                (conv, ssm) = (r.convOut, r.stateOut)
            }
            return [g.out.apply(gated), u.inject.reshaped(1, 1, hcCount), conv, ssm]
        }
    }
}
