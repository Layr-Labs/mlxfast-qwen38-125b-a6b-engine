import Foundation
import MLX

// Positional ABI deliberately keeps the serial decode call site small. The
// immutable tail contains model arrays and geometry, never request results.
struct TrackDecodeMLPReplay {
    let operands: [MLXArray]
    let graph: @Sendable ([MLXArray]) -> [MLXArray]

    @inline(__always)
    func call(_ stream: MLXArray, _ attended: MLXArray, _ inject: MLXArray) -> [MLXArray] {
        graph([stream, attended, inject] + operands)
    }

    static let enabled = ProcessInfo.processInfo.environment["TRACK_DECODE_MLP_REPLAY"] != "0"

    static func make(_ m: TrackMoE, _ hc: TrackHC, hidden H: Int, hcCount C: Int, eps: Float)
        -> TrackDecodeMLPReplay?
    {
        guard enabled, C == 4, H % 128 == 0,
            m.routerW16.dtype == .bfloat16, m.routerW16.shape == [512, H],
            H < 16 * 512, hc.hasInject,
            case .quant(let d) = hc.down, let db = d.biases,
            case .quant(let i)? = hc.inject, let ib = i.biases,
            let u = hc.decodeUp, let ub = u.biases,
            d.bits == 4, i.bits == 4, u.bits == 4,
            d.groupSize == i.groupSize, d.rows % 8 == 0, i.rows == C,
            d.weight.dim(1) * 8 == C * H, i.weight.dim(1) * 8 == C * H,
            u.rows == C * H, u.weight.dim(1) * 8 == d.rows,
            d.rows % 32 == 0, d.rows < 768,
            case .dense(let gate) = m.sharedGate, gate.shape == [1, H],
            let gu = m.sharedGateUp.fused, case .quant(let g) = gu, let gb = g.biases,
            case .quant(let q) = m.sharedDown, let qb = q.biases
        else { return nil }
        // xrowTable can eval on first use, so never call it during tracing.
        let rows = TrackQwen4ExpFastModel.xrowTable(S: 1, K: m.topK)
        let operands = [
            hc.normScaleQ, d.weight, d.scales, db, i.weight, i.scales, ib,
            u.weight, u.scales, ub, m.routerW16, gate, rows,
            m.expertGate.w, m.expertGate.s, m.expertGate.b,
            m.expertUp.w, m.expertUp.s, m.expertUp.b,
            g.weight, g.scales, gb, m.expertDown.w, m.expertDown.s, m.expertDown.b,
            q.weight, q.scales, qb,
        ]
        let graph = compile(shapeless: false) {
            [dg = d.groupSize, dm = d.mode, ig = i.groupSize, im = i.mode,
             ug = u.groupSize, um = u.mode, gg = g.groupSize, gbit = g.bits, gm = g.mode,
             qg = q.groupSize, qbit = q.bits, qm = q.mode,
             eg = m.expertGroupSize, eb = m.expertBits, K = m.topK] p in
            let n = TrackFastKernels.injectNorm(
                residual: p[0], out: p[1], inject: p[2], scale: p[3],
                hcCount: C, hidden: H, eps: eps, tile: false)
            let normed = n.normed.reshaped(1, C * H)
            let down = TrackQuantWeight(weight: p[4], scales: p[5], biases: p[6],
                                        groupSize: dg, bits: 4, mode: dm)
            let inj = TrackQuantWeight(weight: p[7], scales: p[8], biases: p[9],
                                       groupSize: ig, bits: 4, mode: im)
            let up = TrackQuantWeight(weight: p[10], scales: p[11], biases: p[12],
                                      groupSize: ug, bits: 4, mode: um)
            let a = TrackFastMixerKernels.downInject(normed: normed, down: down, inject: inj)
            let mix = TrackFastMixerKernels.upMix(
                act: a.act, normed: normed, up: up, inj: a.inj,
                hcCount: C, hidden: H, hasInject: true, emitF32: true, packedRows: true)
            let x = mix.input.reshaped(1, 1, H)
            let logits = TrackFastMoEKernels.routerGemv(x: mix.inputF32.reshaped(H), w: p[13])
            let r = TrackFastMoEKernels.route(
                logits: logits.reshaped(1, -1), x: mix.input, sharedGate: nil, topK: K)
            let gate = matmul(x, p[14].transposed()).reshaped(1)
            let idx = r.idx.reshaped(K)
            let sharedGU = TrackQuantWeight(weight: p[22], scales: p[23], biases: p[24],
                                            groupSize: gg, bits: gbit, mode: gm)
            let act = TrackFastMoEKernels.gateUpAct(
                wg: p[16], sg: p[17], bg: p[18], wu: p[19], su: p[20], bu: p[21],
                shared: sharedGU, x: mix.input, idx: idx, xrow: p[15], groupSize: eg, bits: eb)
            let sharedDown = TrackQuantWeight(weight: p[28], scales: p[29], biases: p[30],
                                              groupSize: qg, bits: qbit, mode: qm)
            let out = TrackFastMoEKernels.downCombine(
                wd: p[25], sd: p[26], bd: p[27], sharedDown: sharedDown, act: act,
                idx: idx, w: r.w.reshaped(K), gate: gate, topK: K, groupSize: eg, bits: eb)
            return [n.stream, out.reshaped(1, 1, H), mix.inject.reshaped(1, 1, C)]
        }
        return TrackDecodeMLPReplay(operands: operands, graph: graph)
    }
}
