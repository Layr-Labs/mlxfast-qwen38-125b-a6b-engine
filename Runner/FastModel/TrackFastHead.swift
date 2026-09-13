// TrackFastHead.swift -- the MTP head's forward over the fast kernels.
//
// The head is one full-attention decoder layer between two projections and a
// final mixer, driven once per draft step. The fork's module runs it through
// the legacy `Qwen4ExpDecoderLayer` path (~125 launches per step); this runs
// the SAME tensors through the bit-exact fusions of the fast model, with the
// attention cache update and the scaled-dot-product attention left to the
// engine's own `attentionWithCacheUpdate`, exactly as the legacy path calls it.
//
// Inject-norm and the MoE pair use the same compiled replay islands as the
// tower (`shapeless: false`, warmed for S=1...8 at load). `TRACK_HEAD_REPLAY=0`
// keeps the uncompiled kernels.

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
    let injectNormReplay: (@Sendable ([MLXArray]) -> [MLXArray])?
    let moePairReplay: (@Sendable ([MLXArray]) -> [MLXArray])?

    static let enabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_FAST_HEAD"] ?? "1") != "0"
    }()

    /// Kill switch for A/B: `TRACK_HEAD_REPLAY=0` rebuilds inject-norm and the
    /// MoE pair uncompiled on every draft step, matching the pre-island path.
    static let replayEnabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_HEAD_REPLAY"] ?? "1") != "0"
    }()

    /// Decode-window lengths the fused MoE pair and the small inject-norm
    /// kernel serve. First-step backlog after a prefill can be larger (wide
    /// inject-norm, uncompiled MoE); that shape compiles on first use.
    static let replayWindows = 1...8

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
        let moe = TrackQwen4ExpFastModel.bindMoE(
            layer.trackChild("mlp"), cfg: cfg,
            modulePath: "mtp.layers.0.mlp.switch_mlp")
        self.moe = moe
        self.finalMixer = TrackQwen4ExpFastModel.bindHC(mtp.trackChild("hyper_connection_mixer"), cfg: cfg)
        if Self.replayEnabled {
            let inject = TrackQwen4ExpFastModel.makeInjectNormReplay(
                hcCount: cfg.hcCount, hidden: cfg.hiddenSize, eps: cfg.rmsNormEps)
            let pair = TrackQwen4ExpFastModel.makeMoEPairReplay(moe, hcCount: cfg.hcCount)
            self.injectNormReplay = inject
            self.moePairReplay = pair
            self.warmCompiledReplays(injectNormReplay: inject, moePairReplay: pair)
        } else {
            self.injectNormReplay = nil
            self.moePairReplay = nil
        }
    }

    /// Trace each fused-window shape once at load so the first draft step does
    /// not pay compile. No-ops off the GPU stream (the runtime path does too).
    private func warmCompiledReplays(
        injectNormReplay: @Sendable ([MLXArray]) -> [MLXArray],
        moePairReplay: (@Sendable ([MLXArray]) -> [MLXArray])?
    ) {
        guard StreamOrDevice.default.stream === Stream.gpu else { return }
        let scale = attnHC.normScaleQ
        let dtype = scale.dtype
        let W = hcCount * hidden
        for S in Self.replayWindows {
            let residual = MLXArray.zeros([1, S, W], dtype: dtype)
            let out = MLXArray.zeros([1, S, hidden], dtype: dtype)
            let inject = MLXArray.zeros([1, S, hcCount], dtype: dtype)
            _ = injectNormReplay([residual, out, inject, scale])
        }
        if let moePairReplay,
            TrackFastMoEKernels.isFast(k: hidden, n: moe.sharedHidden),
            !TrackFastMoEKernels.isFast(k: moe.sharedHidden, n: hidden),
            case .quant(let guq) = moe.sharedGateUp.fused, let guB = guq.biases,
            case .quant(let dq) = moe.sharedDown, let dB = dq.biases
        {
            let K = moe.topK
            for S in Self.replayWindows {
                let x2 = MLXArray.zeros([S, hidden], dtype: dtype)
                let flatIdx = MLXArray.zeros([S * K], dtype: .uint32)
                let weights = MLXArray.zeros([S * K], dtype: .float32)
                let gate = MLXArray.zeros([S], dtype: dtype)
                let xrow = TrackQwen4ExpFastModel.xrowTable(S: S, K: K)
                _ = moePairReplay([
                    x2, flatIdx, weights, gate, xrow,
                    moe.expertGate.w, moe.expertGate.s, moe.expertGate.b,
                    moe.expertUp.w, moe.expertUp.s, moe.expertUp.b,
                    guq.weight, guq.scales, guB,
                    moe.expertDown.w, moe.expertDown.s, moe.expertDown.b,
                    dq.weight, dq.scales, dB,
                ])
            }
        }
    }

    private func injectNorm(
        residual: MLXArray, out: MLXArray?, inject: MLXArray?, scale: MLXArray
    ) -> (stream: MLXArray, normed: MLXArray) {
        if let injectNormReplay, let out, let inject,
            StreamOrDevice.default.stream === Stream.gpu
        {
            let result = injectNormReplay([residual, out, inject, scale])
            return (result[0], result[1])
        }
        return TrackFastKernels.injectNorm(
            residual: residual, out: out, inject: inject, scale: scale,
            hcCount: hcCount, hidden: hidden, eps: eps, tile: false)
    }

    private func hcMix(_ hc: TrackHC, normed: MLXArray, precomputedInj: MLXArray? = nil)
        -> (MLXArray, MLXArray)
    {
        let S = normed.dim(1)
        if normed.dim(0) == 1, S <= 8, case .quant(let dq) = hc.down, case .quant(let uq) = hc.up,
            dq.biases != nil, uq.biases != nil
        {
            var injQ: TrackQuantWeight? = nil
            if precomputedInj == nil, hc.hasInject, case .quant(let q)? = hc.inject, q.biases != nil
            {
                injQ = q
            }
            if precomputedInj != nil || !hc.hasInject || injQ != nil {
                let n2 = normed.reshaped(S, hcCount * hidden)
                let d = TrackFastMixerKernels.downInject(normed: n2, down: dq, inject: injQ)
                let injIn = precomputedInj?.reshaped(S, hcCount) ?? d.inj
                let u = TrackFastMixerKernels.upMix(
                    act: d.act, normed: n2, up: uq, inj: injIn, hcCount: hcCount, hidden: hidden,
                    hasInject: hc.hasInject)
                return (u.input.reshaped(1, S, hidden), u.inject.reshaped(1, S, hcCount))
            }
        }
        let lo = hc.down.apply(normed)
        let act: MLXArray, inj: MLXArray
        if let precomputedInj {
            inj = precomputedInj
            act = TrackFastKernels.siluHead(lo: lo, width: hc.lowrank)
        } else if normed.dim(1) == 1, hc.hasInject, case .quant(let iq)? = hc.inject, let ib = iq.biases
        {
            let r = TrackFastKernels.mixerHead(
                lo: lo, normed: normed, w: iq.weight, s: iq.scales, b: ib,
                groupSize: iq.groupSize, bits: iq.bits, width: hc.lowrank)
            act = r.act; inj = r.inj
        } else {
            inj = hc.inject?.apply(normed) ?? lo
            act = TrackFastKernels.siluHead(lo: lo, width: hc.lowrank)
        }
        let w = hc.up.apply(act)
        return TrackFastKernels.hcMix(
            w: w, normed: normed, inj: inj, hcCount: hcCount, hidden: hidden,
            hasInject: hc.hasInject)
    }

    private func mixAfterInject(
        hc: TrackHC, residual: MLXArray, out: MLXArray?, inject: MLXArray?,
        scale: MLXArray
    ) -> (stream: MLXArray, input: MLXArray, injectW: MLXArray) {
        if case .quant(let iq)? = hc.inject, iq.biases != nil,
            let fused = TrackQuantPrologue.apply(
                residual: residual, out: out, injectGate: inject, scale: scale,
                weight: iq, hcCount: hcCount, hidden: hidden, eps: eps, tile: false)
        {
            let m = hcMix(hc, normed: fused.normed, precomputedInj: fused.inj)
            return (fused.stream, m.0, m.1)
        }
        let (stream, normed) = injectNorm(
            residual: residual, out: out, inject: inject, scale: scale)
        let m = hcMix(hc, normed: normed)
        return (stream, m.0, m.1)
    }

    /// One head application over `[1, S]` inputs; returns `sample` `[1,S,H]`
    /// and the next multi stream `[1,S,hc*H]`. Nil when the fast path does
    /// not serve the call (context past the indexer budget).
    func forward(
        nextTokenIds ids: MLXArray, multiStream multi: MLXArray, embedTokens: Embedding,
        cache: Qwen4ExpAttentionCache
    ) -> (sample: MLXArray, multi: MLXArray)? {
        let B = ids.dim(0), S = ids.dim(1)
        guard Self.enabled, B == 1, cache.offset + S <= indexerBudget else {
            TrackIndexerTape.syncHead(cache)
            return nil
        }
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
        var (st, input, injectW) = mixAfterInject(
            hc: attnHC, residual: hyper, out: nil, inject: nil, scale: attnHC.normScaleQ)
        let attended = attention(input, cache: cache, offset: offset, residual: st, inject: injectW)
        if attended.dim(-1) == hcCount * hidden {
            (st, input, injectW) = mixAfterInject(
                hc: mlpHC, residual: attended, out: nil, inject: nil, scale: mlpHC.normScaleQ)
        } else {
            (st, input, injectW) = mixAfterInject(
                hc: mlpHC, residual: st, out: attended, inject: injectW, scale: mlpHC.normScaleQ)
        }
        let moeOut = TrackQwen4ExpFastModel.moeForwardShared(
            moe, input, replay: moePairReplay, residual: st, inject: injectW, hcCount: hcCount)
        let (multiNext, finalNormed): (MLXArray, MLXArray)
        if moeOut.dim(-1) == hcCount * hidden {
            (multiNext, finalNormed) = injectNorm(
                residual: moeOut, out: nil, inject: nil, scale: finalMixer.normScaleQ)
        } else {
            (multiNext, finalNormed) = injectNorm(
                residual: st, out: moeOut, inject: injectW, scale: finalMixer.normScaleQ)
        }
        let (sample, _) = hcMix(finalMixer, normed: finalNormed)
        if let buf = TrackIndexerTape.buffer(owner: cache) { asyncEval(buf) }
        return (sample, multiNext)
    }

    private func attention(
        _ x: MLXArray, cache: Qwen4ExpAttentionCache, offset: Int,
        residual: MLXArray? = nil, inject: MLXArray? = nil
    ) -> MLXArray {
        let S = x.dim(1)
        let heads = cfg.attentionHeads, kvHeads = cfg.kvHeads, d = cfg.headDim
        let qkv = attn.qkv.apply(x)
        // Fused qkv carries indexer k only. Indexer q is omitted: this path
        // serves only while the tape is inside the budget, so the keep-mask
        // top-k does not run. Past the budget this forward returns nil.
        let idxStart = 2 * attn.qWidth + 2 * attn.kvWidth
        TrackIndexerTape.appendHead(
            cache: cache,
            keys: TrackContiguity.packedLastAxis(
                qkv[.ellipsis, idxStart ..< (idxStart + cfg.indexerHeadDim)]),
            capacity: indexerBudget)
        let (c, s) = rotary.cosSin(qwen4ExpPositions(offset: offset, count: S))
        let cosT = c.asType(x.dtype).reshaped(S, rotaryDims)
        let sinT = s.asType(x.dtype).reshaped(S, rotaryDims)
        let prep: (q: MLXArray, k: MLXArray, v: MLXArray)
        if TrackKVFuse.enabled, let ring = TrackKVFuse.ring(from: cache),
            TrackKVFuse.canAppend(offset: ring.offset, count: S, cap: ring.cap, wrap: false)
        {
            prep = TrackFastKernels.attnPrepFused(
                qkv: qkv, qNorm: attn.qNormW, kNorm: attn.kNormW, cos: cosT, sin: sinT,
                kCache: ring.keys, vCache: ring.values, writeOffset: ring.offset,
                heads: heads, kvHeads: kvHeads, headDim: d, rotaryDims: rotaryDims, eps: eps)
        } else {
            prep = TrackFastKernels.attnPrep(
                qkv: qkv, qNorm: attn.qNormW, kNorm: attn.kNormW, cos: cosT, sin: sinT,
                heads: heads, kvHeads: kvHeads, headDim: d, rotaryDims: rotaryDims, eps: eps)
        }
        let mask = makeAttentionMask(n: S, cache: cache)
        let att = attentionWithCacheUpdate(
            queries: prep.q, keys: prep.k, values: prep.v, cache: cache,
            scale: attentionScale, mask: mask)
        let out = TrackFastKernels.attnGate(att: att, qkv: qkv, gateOffset: attn.qWidth)
        if let residual, let inject,
            let fused = TrackResidualEpilogue.project(
                attn.out, x: out, residual: residual, inject: inject,
                hcCount: hcCount, hidden: hidden)
        {
            return fused
        }
        return attn.out.apply(out)
    }
}
