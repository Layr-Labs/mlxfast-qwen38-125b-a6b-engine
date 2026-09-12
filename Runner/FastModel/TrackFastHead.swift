// TrackFastHead.swift -- the MTP head's forward over the fast kernels.
//
// The head is one full-attention decoder layer between two projections and a
// final mixer, driven once per draft step. The fork's module runs it through
// the legacy `Qwen4ExpDecoderLayer` path (~125 launches per step); this runs
// the SAME tensors through the bit-exact fusions of the fast model, with the
// attention cache update and the scaled-dot-product attention left to the
// engine's own `attentionWithCacheUpdate`, exactly as the legacy path calls it.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

final class TrackFastHead {
    let cfg: Qwen4ExpTextConfiguration
    let hcCount: Int
    let hidden: Int
    let eps: Float
    let rotaryDims: Int
    let rotary: Qwen4ExpRotary
    let attentionScale: Float
    let indexerBudget: Int

    let preNormEmbeddingW: MLXArray
    let preNormHiddenW: MLXArray
    let fcEmbedding: TrackProj
    let fcHidden: TrackProj
    let attnHC: TrackHC
    let mlpHC: TrackHC
    let attn: TrackAttn
    let moe: TrackMoE
    let finalMixer: TrackHC
    private let attnMixerReplay: (@Sendable ([MLXArray]) -> [MLXArray])?
    private let mlpMixerReplay: (@Sendable ([MLXArray]) -> [MLXArray])?
    private let finalMixerReplay: (@Sendable ([MLXArray]) -> [MLXArray])?
    private let moePairReplay: (@Sendable ([MLXArray]) -> [MLXArray])?
    private let injectNormReplay: @Sendable ([MLXArray]) -> [MLXArray]

    static let enabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_FAST_HEAD"] ?? "1") != "0"
    }()

    init(_ mtp: TrackQwen4ExpMTPModule, configuration cfg: Qwen4ExpTextConfiguration) {
        self.cfg = cfg
        self.hcCount = cfg.hcCount
        self.hidden = cfg.hiddenSize
        self.eps = cfg.rmsNormEps
        self.rotaryDims = cfg.rotaryDimensions
        self.rotary = Qwen4ExpRotary(dimensions: cfg.rotaryDimensions, base: cfg.ropeTheta)
        self.attentionScale = Foundation.pow(Float(cfg.headDim), -0.5)
        self.indexerBudget = cfg.indexerBudget
        precondition(cfg.rmsNormWeightOffset == 0)
        precondition(mtp.layerCount == 1, "TrackFastHead serves a one-layer head")
        self.preNormEmbeddingW = mtp.preFCNormEmbedding.weight
        self.preNormHiddenW = mtp.preFCNormHidden.weight
        self.fcEmbedding = TrackProj(mtp.fcEmbedding)
        self.fcHidden = TrackProj(mtp.fcHidden)
        let layer = mtp.layers[0]
        precondition(!layer.isLinear && layer.ple == nil)
        self.attnHC = TrackQwen4ExpFastModel.bindHC(layer.trackChild("attn_hyper_connection"), cfg: cfg)
        self.mlpHC = TrackQwen4ExpFastModel.bindHC(layer.trackChild("mlp_hyper_connection"), cfg: cfg)
        self.attn = TrackQwen4ExpFastModel.bindAttn(layer.trackChild("self_attn"), cfg: cfg)
        self.moe = TrackQwen4ExpFastModel.bindMoE(layer.trackChild("mlp"), cfg: cfg)
        self.finalMixer = TrackQwen4ExpFastModel.bindHC(mtp.trackChild("hyper_connection_mixer"), cfg: cfg)
        self.attnMixerReplay = Self.makeMixerReplay(attnHC, cfg: cfg, emitF32: false)
        self.mlpMixerReplay = Self.makeMixerReplay(mlpHC, cfg: cfg, emitF32: true)
        self.finalMixerReplay = Self.makeMixerReplay(finalMixer, cfg: cfg, emitF32: false)
        self.moePairReplay = TrackQwen4ExpFastModel.makeMoEPairReplay(moe)
        self.injectNormReplay = compile(shapeless: false) {
            [hcCount = cfg.hcCount, hidden = cfg.hiddenSize, eps = cfg.rmsNormEps] inputs in
            let r = TrackFastKernels.injectNorm(
                residual: inputs[0], out: inputs[1], inject: inputs[2], scale: inputs[3],
                hcCount: hcCount, hidden: hidden, eps: eps, tile: false)
            return [r.stream, r.normed]
        }
    }

    /// Cache only the two existing opaque kernels over immutable loaded weights.
    /// Head-owned closures also work when the target mixer has no replay bundle.
    private static func makeMixerReplay(
        _ hc: TrackHC, cfg: Qwen4ExpTextConfiguration, emitF32: Bool
    ) -> (@Sendable ([MLXArray]) -> [MLXArray])? {
        guard case .quant(let down) = hc.down, case .quant(let up) = hc.up,
            down.biases != nil, up.biases != nil
        else { return nil }
        let inject: TrackQuantWeight?
        if hc.hasInject {
            guard case .quant(let q)? = hc.inject, q.biases != nil else { return nil }
            inject = q
        } else {
            inject = nil
        }
        return compile(shapeless: false) {
            [down, up, inject, hcCount = cfg.hcCount, hidden = cfg.hiddenSize] inputs in
            let normed = inputs[0]
            let S = normed.dim(1)
            let n2 = normed.reshaped(S, hcCount * hidden)
            let d = TrackFastMixerKernels.downInject(normed: n2, down: down, inject: inject)
            let u = TrackFastMixerKernels.upMix(
                act: d.act, normed: n2, up: up, inj: d.inj, hcCount: hcCount, hidden: hidden,
                hasInject: inject != nil, emitF32: emitF32)
            return [
                u.input.reshaped(1, S, hidden), u.inject.reshaped(1, S, hcCount), u.inputF32,
            ]
        }
    }

    private func injectNorm(
        residual: MLXArray, out: MLXArray?, inject: MLXArray?, scale: MLXArray
    ) -> (stream: MLXArray, normed: MLXArray) {
        if let out, let inject, StreamOrDevice.default.stream === Stream.gpu {
            let r = injectNormReplay([residual, out, inject, scale])
            return (r[0], r[1])
        }
        return TrackFastKernels.injectNorm(
            residual: residual, out: out, inject: inject, scale: scale,
            hcCount: hcCount, hidden: hidden, eps: eps, tile: false)
    }

    private func hcMix(
        _ hc: TrackHC, normed: MLXArray,
        replay: (@Sendable ([MLXArray]) -> [MLXArray])?, emitF32: Bool = false
    ) -> (input: MLXArray, inject: MLXArray, inputF32: MLXArray?) {
        let S = normed.dim(1)
        if normed.dim(0) == 1, S <= 8, case .quant(let dq) = hc.down, case .quant(let uq) = hc.up,
            dq.biases != nil, uq.biases != nil
        {
            var injQ: TrackQuantWeight? = nil
            if hc.hasInject, case .quant(let q)? = hc.inject, q.biases != nil { injQ = q }
            if !hc.hasInject || injQ != nil {
                if let replay, StreamOrDevice.default.stream === Stream.gpu {
                    let r = replay([normed])
                    return (r[0], r[1], emitF32 ? r[2].reshaped(1, S, hidden) : nil)
                }
                let n2 = normed.reshaped(S, hcCount * hidden)
                let d = TrackFastMixerKernels.downInject(normed: n2, down: dq, inject: injQ)
                let u = TrackFastMixerKernels.upMix(
                    act: d.act, normed: n2, up: uq, inj: d.inj, hcCount: hcCount, hidden: hidden,
                    hasInject: hc.hasInject, emitF32: emitF32)
                return (
                    u.input.reshaped(1, S, hidden), u.inject.reshaped(1, S, hcCount),
                    emitF32 ? u.inputF32.reshaped(1, S, hidden) : nil)
            }
        }
        let lo = hc.down.apply(normed)
        let act: MLXArray, inj: MLXArray
        if normed.dim(1) == 1, hc.hasInject, case .quant(let iq)? = hc.inject, let ib = iq.biases {
            let r = TrackFastKernels.mixerHead(
                lo: lo, normed: normed, w: iq.weight, s: iq.scales, b: ib,
                groupSize: iq.groupSize, bits: iq.bits, width: hc.lowrank)
            act = r.act; inj = r.inj
        } else {
            inj = hc.inject?.apply(normed) ?? lo
            act = TrackFastKernels.siluHead(lo: lo, width: hc.lowrank)
        }
        let w = hc.up.apply(act)
        let r = TrackFastKernels.hcMix(
            w: w, normed: normed, inj: inj, hcCount: hcCount, hidden: hidden,
            hasInject: hc.hasInject)
        return (r.input, r.inject, nil)
    }

    /// One head application over `[1, S]` inputs; returns `sample` `[1,S,H]`
    /// and the next multi stream `[1,S,hc*H]`. Nil when the fast path does
    /// not serve the call (context past the indexer budget).
    func forward(
        nextTokenIds ids: MLXArray, multiStream multi: MLXArray, embedTokens: Embedding,
        cache: Qwen4ExpAttentionCache
    ) -> (sample: MLXArray, multi: MLXArray)? {
        let B = ids.dim(0), S = ids.dim(1)
        guard Self.enabled, B == 1, cache.offset + S <= indexerBudget else { return nil }
        let offset = cache.offset

        // embed -> pre-norm -> fc ; multi -> pre-norm (one statistic) -> per-stream fc ; add
        let embedded = fcEmbedding.apply(
            MLXFast.rmsNorm(embedTokens(ids), weight: preNormEmbeddingW, eps: eps))
        var stream = MLXFast.rmsNorm(multi, weight: preNormHiddenW, eps: eps)
            .reshaped(B, S, hcCount, hidden)
        stream = fcHidden.apply(stream)
        stream = embedded[.ellipsis, .newAxis, 0...] + stream
        let hyper = stream.reshaped(B, S, hcCount * hidden)

        // attention block
        var (st, normed) = injectNorm(
            residual: hyper, out: nil, inject: nil, scale: attnHC.normScaleQ)
        let am = hcMix(attnHC, normed: normed, replay: attnMixerReplay)
        let attended = attention(am.input, cache: cache, offset: offset)
        (st, normed) = injectNorm(
            residual: st, out: attended, inject: am.inject, scale: mlpHC.normScaleQ)
        let mm = hcMix(mlpHC, normed: normed, replay: mlpMixerReplay, emitF32: true)
        let moeOut = TrackQwen4ExpFastModel.moeForwardShared(
            moe, mm.input, inputF32: mm.inputF32, replay: moePairReplay)
        let (multiNext, finalNormed) = injectNorm(
            residual: st, out: moeOut, inject: mm.inject, scale: finalMixer.normScaleQ)
        let sample = hcMix(finalMixer, normed: finalNormed, replay: finalMixerReplay).input
        return (sample, multiNext)
    }

    private func attention(_ x: MLXArray, cache: Qwen4ExpAttentionCache, offset: Int) -> MLXArray {
        let B = x.dim(0), S = x.dim(1)
        let heads = cfg.attentionHeads, kvHeads = cfg.kvHeads, d = cfg.headDim
        let qkv = attn.qkv.apply(x)
        let idxStart = 2 * attn.qWidth + 2 * attn.kvWidth
        _ = cache.updateIndexer(keys: qkv[.ellipsis, idxStart ..< (idxStart + cfg.indexerHeadDim)])
        let (c, s) = rotary.cosSin(qwen4ExpPositions(offset: offset, count: S))
        let prep = TrackFastKernels.attnPrep(
            qkv: qkv, qNorm: attn.qNormW, kNorm: attn.kNormW,
            cos: c.asType(x.dtype).reshaped(S, rotaryDims), sin: s.asType(x.dtype).reshaped(S, rotaryDims),
            heads: heads, kvHeads: kvHeads, headDim: d, rotaryDims: rotaryDims, eps: eps)
        let mask = makeAttentionMask(n: S, cache: cache)
        let att = attentionWithCacheUpdate(
            queries: prep.q, keys: prep.k, values: prep.v, cache: cache,
            scale: attentionScale, mask: mask)
        let out = TrackFastKernels.attnGate(att: att, qkv: qkv, gateOffset: attn.qWidth)
        return attn.out.apply(out)
    }
}
