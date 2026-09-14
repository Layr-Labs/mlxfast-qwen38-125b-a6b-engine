// Init-time JIT warmup for speculative verify widths S = 1...7 (depth k = 0...6).
// Metal templates and shapeless:false compile islands key on S / T / VPT / BR.
// Dummy activations; resident weights. eval() forces JIT. Dummy arrays drop after.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

enum TrackFastWarmup {
    /// Decode S=1 plus MTP verify widths 1+k for declared depth k = 1...6.
    static let verifyWarmupWidths = 1...7

    private static let lock = NSLock()
    nonisolated(unsafe) private static var towerDone = false
    nonisolated(unsafe) private static var headDone = false

    static func warmTower(_ model: TrackQwen4ExpFastModel) {
        lock.lock(); defer { lock.unlock() }
        guard !towerDone else { return }
        guard StreamOrDevice.default.stream === Stream.gpu else { return }
        guard !model.layers.isEmpty,
            model.hidden % 4 == 0,
            (model.hcCount * model.hidden).isMultiple(of: 512)
        else { return }
        for S in verifyWarmupWidths {
            warmTowerWidth(S, model)
        }
        Memory.clearCache()
        towerDone = true
    }

    static func warmHead(_ head: TrackFastHead, embedTokens: Embedding) {
        lock.lock(); defer { lock.unlock() }
        guard !headDone else { return }
        guard StreamOrDevice.default.stream === Stream.gpu else { return }
        for S in verifyWarmupWidths {
            let cache = Qwen4ExpAttentionCache()
            let ids = MLXArray.zeros([1, S], dtype: .int32)
            let multi = MLXArray.zeros(
                [1, S, head.hcCount * head.hidden], dtype: head.attnHC.normScaleQ.dtype)
            if let out = head.forward(
                nextTokenIds: ids, multiStream: multi, embedTokens: embedTokens, cache: cache)
            {
                eval(out.sample, out.multi)
            }
        }
        Memory.clearCache()
        headDone = true
    }

    private static func warmTowerWidth(_ S: Int, _ model: TrackQwen4ExpFastModel) {
        let dtype = model.layers[0].attnHC.normScaleQ.dtype
        let H = model.hidden
        let hc = model.hcCount
        let W = hc * H
        let residual = MLXArray.zeros([1, S, W], dtype: dtype)
        let blockOut = MLXArray.zeros([1, S, H], dtype: dtype)
        let inject = MLXArray.zeros([1, S, hc], dtype: dtype)
        let scale = model.layers[0].attnHC.normScaleQ
        let replayed = model.injectNormReplay([residual, blockOut, inject, scale])
        eval(replayed)
        // TILE / HAS_INJECT variants used on the first layer and the PLE reseat.
        // S is not a template parameter here; one width compiles the kernel.
        if S == 1 {
            let tiled = TrackFastKernels.injectNorm(
                residual: MLXArray.zeros([1, S, H], dtype: dtype), out: nil, inject: nil,
                scale: scale, hcCount: hc, hidden: H, eps: model.eps, tile: true)
            let plain = TrackFastKernels.injectNorm(
                residual: residual, out: nil, inject: nil, scale: scale,
                hcCount: hc, hidden: H, eps: model.eps, tile: false)
            eval(tiled.stream, tiled.normed, plain.stream, plain.normed)
        }

        let normed = MLXArray.zeros([1, S, W], dtype: dtype)
        mixer(model.layers[0].attnHC, normed: normed, hcCount: hc, hidden: H, emitF32: false)
        mixer(model.layers[0].mlpHC, normed: normed, hcCount: hc, hidden: H, emitF32: true)
        mixer(model.finalMixer, normed: normed, hcCount: hc, hidden: H, emitF32: false)

        let x = MLXArray.zeros([1, S, H], dtype: dtype)
        if let gdn = model.layers.first(where: { $0.gdn != nil })?.gdn,
            gdn.geometry.dk == 128, gdn.geometry.dv == 128, gdn.geometry.convDim.isMultiple(of: 128)
        {
            eval(gdn.proj.apply(x))
            let geo = gdn.geometry
            let proj = MLXArray.zeros([1, S, geo.projWidth], dtype: dtype)
            let convState = MLXArray.zeros(
                [1, geo.convKernel - 1, geo.convDim], dtype: dtype)
            let ssm = MLXArray.zeros([1, geo.hv, geo.dv, geo.dk], dtype: .float32)
            for capture in [false, true] {
                let r = TrackFastKernels.gdn(
                    proj: proj, convState: convState, convW: gdn.convW,
                    negExpALog: gdn.negExpALog, dtBias: gdn.dtBias, stateIn: ssm,
                    T: S, capture: capture, geometry: geo)
                let gated = TrackFastKernels.gatedRMS(
                    y: r.y, proj: proj, w: gdn.normW, zOffset: gdn.zOffset, eps: 1e-6)
                eval(r.y, r.convOut, r.stateOut, gated)
            }
            eval(gdn.out.apply(MLXArray.zeros([1, S, geo.hv * geo.dv], dtype: dtype)))
        }

        if let attn = model.layers.first(where: { $0.attn != nil })?.attn {
            eval(attn.qkv.apply(x))
            let qkv = MLXArray.zeros([1, S, attn.qkv.width], dtype: dtype)
            let (cos, sin) = model.rotary.cosSin(qwen4ExpPositions(offset: 0, count: S))
            let rope = (
                cos: cos.asType(dtype).reshaped(S, model.rotaryDims),
                sin: sin.asType(dtype).reshaped(S, model.rotaryDims)
            )
            let prep = TrackFastKernels.attnPrep(
                qkv: qkv, qNorm: attn.qNormW, kNorm: attn.kNormW, cos: rope.cos, sin: rope.sin,
                heads: model.cfg.attentionHeads, kvHeads: model.cfg.kvHeads,
                headDim: model.cfg.headDim, rotaryDims: model.rotaryDims, eps: model.eps)
            let att = MLXArray.zeros(
                [1, model.cfg.attentionHeads, S, model.cfg.headDim], dtype: dtype)
            let gated = TrackFastKernels.attnGate(att: att, qkv: qkv, gateOffset: attn.qWidth)
            eval(prep.q, prep.k, prep.v, gated)
            eval(attn.out.apply(MLXArray.zeros(
                [1, S, model.cfg.attentionHeads * model.cfg.headDim], dtype: dtype)))
        }

        let inputF32 = MLXArray.zeros([1, S, H], dtype: .float32)
        for layer in model.layers {
            let moeOut = TrackQwen4ExpFastModel.moeForwardShared(
                layer.moe, x, inputF32: inputF32, replay: layer.moePairReplay)
            eval(moeOut)
        }
        eval(model.base.head(x))
    }

    private static func mixer(
        _ hc: TrackHC, normed: MLXArray, hcCount: Int, hidden: Int, emitF32: Bool
    ) {
        let S = normed.dim(1)
        guard case .quant(let dq) = hc.down, case .quant(let uq) = hc.up,
            dq.biases != nil, uq.biases != nil
        else { return }
        var injQ: TrackQuantWeight? = nil
        if hc.hasInject, case .quant(let q)? = hc.inject, q.biases != nil { injQ = q }
        guard !hc.hasInject || injQ != nil else { return }
        let n2 = normed.reshaped(S, hcCount * hidden)
        let d = TrackFastMixerKernels.downInject(normed: n2, down: dq, inject: injQ)
        let u = TrackFastMixerKernels.upMix(
            act: d.act, normed: n2, up: uq, inj: d.inj, hcCount: hcCount, hidden: hidden,
            hasInject: hc.hasInject, emitF32: emitF32)
        eval(d.lo, d.act, d.inj, u.input, u.inject, u.inputF32)
    }
}

extension TrackQwen4ExpFastModel {
    func warmVerifyCaches() {
        TrackFastWarmup.warmTower(self)
    }
}

extension TrackFastHead {
    func warmVerifyCaches(embedTokens: Embedding) {
        TrackFastWarmup.warmHead(self, embedTokens: embedTokens)
    }
}
