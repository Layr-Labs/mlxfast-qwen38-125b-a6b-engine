// Reusable graphs for the pure, repeatedly evaluated parts of a decode step.
// The cache contains executable graphs, never token-dependent results. Every
// invocation receives new activations and (for deltanet) explicit state. Cache
// mutation, capture staging, attention offsets, and acceptance stay outside.
import Foundation
import MLX

enum TrackFastGraph {
    static let enabled =
        (ProcessInfo.processInfo.environment["TRACK_COMPILED_BLOCKS"] ?? "1") != "0"

    static func mixer(_ hc: TrackHC, _ normed: MLXArray, hcCount: Int, hidden: Int)
        -> (MLXArray, MLXArray)
    {
        let lo = hc.down.apply(normed)
        let act: MLXArray, inj: MLXArray
        if normed.dim(1) == 1, hc.hasInject,
            case .quant(let iq)? = hc.inject, let ib = iq.biases
        {
            let r = TrackFastKernels.mixerHead(
                lo: lo, normed: normed, w: iq.weight, s: iq.scales, b: ib,
                groupSize: iq.groupSize, bits: iq.bits, width: hc.lowrank)
            act = r.act
            inj = r.inj
        } else {
            inj = hc.inject?.apply(normed) ?? lo
            act = TrackFastKernels.siluHead(lo: lo, width: hc.lowrank)
        }
        return TrackFastKernels.hcMix(
            w: hc.up.apply(act), normed: normed, inj: inj,
            hcCount: hcCount, hidden: hidden, hasInject: hc.hasInject)
    }

    static func recurrent(_ g: TrackGDN, _ args: [MLXArray], capture: Bool) -> [MLXArray] {
        let x = args[0], conv = args[1], state = args[2]
        let proj = g.proj.apply(x)
        let r = TrackFastKernels.gdn(
            proj: proj, convState: conv, convW: g.convW, negExpALog: g.negExpALog,
            dtBias: g.dtBias, stateIn: state, T: x.dim(1), capture: capture,
            geometry: g.geometry)
        let gated = TrackFastKernels.gatedRMS(
            y: r.y, proj: proj, w: g.normW, zOffset: g.zOffset, eps: 1e-6)
        return [g.out.apply(gated), r.convOut, r.stateOut]
    }
}
