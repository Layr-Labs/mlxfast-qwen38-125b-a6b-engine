// TrackFastModel.swift -- the track's fast forward pass over the loaded target.
//
// WHAT THIS IS. `TrackQwen4ExpFastModel` wraps the engine-loaded `Qwen4ExpModel`
// and serves the SAME tensors through a leaner graph. Nothing is re-quantized
// and no weight value changes: projections that read the same input are
// concatenated along their output rows in memory (row order is a layout
// choice, the per-row arithmetic is the same kernel over the same bytes),
// norm scales that the reference multiplies by a power of two are pre-scaled
// (exact in bf16), the bf16 router weight is held once in float32 (the
// reference casts it on every call), and one fused Metal launch per
// gated-deltanet layer replaces the engine's chain of ~20 launches.
//
// The measured decode step on this family is bound by graph construction and
// launch count, not by weight bandwidth (see docs and the fork's
// Qwen4ExpDecodeStepCostTests), so the lever is fewer, heavier launches.
//
// The engine sees a `LanguageModel` that conforms to the same CBv2 protocols
// as `Qwen4ExpModel`; caches and recurrent state keep the engine's layouts so
// prefill, decode and capture-verify interoperate with the trusted code.
// Anything this fast path does not serve (positioned inputs, a QSA context
// past the indexer budget, wider batches) is delegated to the wrapped model.

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

// MARK: - Weight helpers

extension Module {
    func trackChild(_ key: String) -> Module {
        guard let child = children()[unwrapping: key] else {
            preconditionFailure("TrackFastModel: module has no child \(key)")
        }
        return child
    }
    func trackArrays() -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: parameters().flattened())
    }
    func trackArray(_ key: String) -> MLXArray {
        guard let array = trackArrays()[key] else {
            preconditionFailure("TrackFastModel: module has no parameter \(key)")
        }
        return array
    }
}

/// An affine-quantized projection `[N, K]`.
struct TrackQuantWeight {
    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode

    var rows: Int { weight.dim(0) }

    func apply(_ x: MLXArray) -> MLXArray {
        quantizedMM(
            x, weight, scales: scales, biases: biases, transpose: true,
            groupSize: groupSize, bits: bits, mode: mode)
    }

    func compatible(_ other: TrackQuantWeight) -> Bool {
        groupSize == other.groupSize && bits == other.bits && mode == other.mode
            && weight.dim(1) == other.weight.dim(1) && (biases == nil) == (other.biases == nil)
            && weight.dtype == other.weight.dtype
    }

    static func concat(_ parts: [TrackQuantWeight]) -> TrackQuantWeight {
        precondition(!parts.isEmpty)
        let first = parts[0]
        for p in parts.dropFirst() { precondition(first.compatible(p)) }
        let w = concatenated(parts.map(\.weight), axis: 0)
        let s = concatenated(parts.map(\.scales), axis: 0)
        let b = first.biases == nil ? nil : concatenated(parts.map { $0.biases! }, axis: 0)
        return TrackQuantWeight(
            weight: w, scales: s, biases: b, groupSize: first.groupSize, bits: first.bits,
            mode: first.mode)
    }

    func rowsReordered(_ order: [Int32]) -> TrackQuantWeight {
        let idx = MLXArray(order)
        return TrackQuantWeight(
            weight: weight[idx], scales: scales[idx], biases: biases.map { $0[idx] },
            groupSize: groupSize, bits: bits, mode: mode)
    }
}

/// A projection that is either quantized or a dense `[N, K]` weight.
enum TrackProj {
    case quant(TrackQuantWeight)
    case dense(MLXArray)

    init(_ module: Module) {
        if let q = module as? QuantizedLinear {
            self = .quant(
                TrackQuantWeight(
                    weight: q.weight, scales: q.scales, biases: q.biases,
                    groupSize: q.groupSize, bits: q.bits, mode: q.mode))
        } else if let l = module as? Linear {
            precondition(l.bias == nil, "TrackFastModel: biased Linear is not served")
            self = .dense(l.weight)
        } else {
            preconditionFailure("TrackFastModel: unsupported projection \(type(of: module))")
        }
    }

    var rows: Int {
        switch self {
        case .quant(let q): return q.rows
        case .dense(let w): return w.dim(0)
        }
    }

    func apply(_ x: MLXArray) -> MLXArray {
        switch self {
        case .quant(let q): return q.apply(x)
        case .dense(let w): return matmul(x, w.transposed())
        }
    }

    /// Concatenate along output rows when every part is quantized with one
    /// geometry; otherwise nil (the caller keeps the parts separate).
    static func fused(_ parts: [TrackProj]) -> TrackProj? {
        var quants: [TrackQuantWeight] = []
        for p in parts {
            guard case .quant(let q) = p else { return nil }
            if let f = quants.first, !f.compatible(q) { return nil }
            quants.append(q)
        }
        return .quant(TrackQuantWeight.concat(quants))
    }
}

/// Several projections of one input, applied fused when possible.
struct TrackMultiProj {
    let fused: TrackProj?
    let parts: [TrackProj]
    let offsets: [Int]

    init(_ parts: [TrackProj]) {
        self.parts = parts
        self.fused = TrackProj.fused(parts)
        var offs: [Int] = [0]
        for p in parts { offs.append(offs.last! + p.rows) }
        self.offsets = offs
    }

    var width: Int { offsets.last! }

    /// The concatenated output `[..., width]`.
    ///
    /// Row concatenation is bit-exact on the GEMV paths (one row is one
    /// accumulation regardless of N), but the split-K GEMM the wide (prefill)
    /// shapes dispatch chooses its split from N, so wide inputs run the parts
    /// separately and stay exact with the reference.
    func apply(_ x: MLXArray) -> MLXArray {
        if let fused, x.dim(-2) <= 8 { return fused.apply(x) }
        return concatenated(parts.map { $0.apply(x) }, axis: -1)
    }
}

// MARK: - Bound layers

struct TrackHC {
    /// hc_norm scale, pre-divided by hc_count (exact: power of two).
    let normScaleQ: MLXArray
    /// input_mix_weight_down (320 rows: the fast GEMV) and block_inject_weight
    /// (4 rows: the non-fast GEMV) stay SEPARATE launches: concatenating them
    /// would route the 320 rows through the non-fast kernel, whose
    /// accumulation differs from the fast one in rare last-bit cases.
    let down: TrackProj
    let inject: TrackProj?
    let up: TrackProj
    let lowrank: Int
    let hasInject: Bool
}

struct TrackGDN {
    let proj: TrackMultiProj  // qkv | z | b | a
    let convW: MLXArray  // [convDim, KC]
    let negExpALog: MLXArray  // [Hv] float32
    let dtBias: MLXArray  // [Hv]
    let normW: MLXArray  // [Dv]
    let out: TrackProj
    let geometry: TrackFastKernels.GDNGeometry
    let zOffset: Int
    let valueDim: Int
}

struct TrackAttn {
    /// q rows (all heads), gate rows (all heads), k rows, v rows, indexer k rows.
    let qkv: TrackMultiProj
    /// The whole `index_qk_proj` (q and k rows), for wide windows: the GEMM
    /// path's rounding depends on N, so the tape must come from the full
    /// projection to stay exact there.
    let indexerFull: TrackProj
    let indexerQWidth: Int
    let qNormW: MLXArray
    let kNormW: MLXArray
    let indexerK: TrackProj
    let out: TrackProj
    let qWidth: Int
    let kvWidth: Int
}

struct TrackMoE {
    let routerW32: MLXArray  // [E, H] float32
    let switchMLP: SwitchGLU
    /// The routed experts' quantized arrays, for the custom gather path.
    let expertGate: (w: MLXArray, s: MLXArray, b: MLXArray)
    let expertUp: (w: MLXArray, s: MLXArray, b: MLXArray)
    let expertDown: (w: MLXArray, s: MLXArray, b: MLXArray)
    let expertGroupSize: Int
    let expertBits: Int
    let sharedGateUp: TrackMultiProj
    let sharedDown: TrackProj
    let sharedGate: TrackProj
    let topK: Int
    let sharedHidden: Int
}

struct TrackPLE {
    let embedding: Qwen4ExpNGramEmbedding
    let keyProj: TrackProj
    let valueProj: TrackProj
    let normKeyScale: MLXArray
    let normQueryScale: MLXArray
    let normConvScale: MLXArray
    let convW: MLXArray  // [wide, K, 1]
    let dilation: Int
    let stateLength: Int
    let stateLayerIndex: Int
}

struct TrackLayer {
    let index: Int
    let attnHC: TrackHC
    let mlpHC: TrackHC
    let gdn: TrackGDN?
    let attn: TrackAttn?
    let moe: TrackMoE
    let ple: TrackPLE?
}

// MARK: - The model

public final class TrackQwen4ExpFastModel: Module, @unchecked Sendable {

    let base: Qwen4ExpModel
    let cfg: Qwen4ExpTextConfiguration
    let embedTokens: Embedding
    let layers: [TrackLayer]
    let finalMixer: TrackHC
    let hcCount: Int
    let hidden: Int
    let eps: Float
    let rotaryDims: Int
    let rotary: Qwen4ExpRotary
    let indexerBudget: Int
    let attentionScale: Float

    /// Debug taps (tests): when set, every layer's output stream and the block
    /// inputs/outputs are appended here.
    nonisolated(unsafe) static var debugTaps: [(String, MLXArray)]? = nil
    /// Layers per partial dispatch inside a forward (0 = one dispatch per step).
    nonisolated(unsafe) public static var asyncChunk: Int = 6

    /// Kill switch for A/B: `TRACK_FAST_FORWARD=0` routes every forward to the
    /// wrapped model.
    static let enabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_FAST_FORWARD"] ?? "1") != "0"
    }()

    public init(base: Qwen4ExpModel) {
        self.base = base
        let cfg = base.configuration
        self.cfg = cfg
        self.hcCount = cfg.hcCount
        self.hidden = cfg.hiddenSize
        self.eps = cfg.rmsNormEps
        self.rotaryDims = cfg.rotaryDimensions
        self.rotary = Qwen4ExpRotary(dimensions: cfg.rotaryDimensions, base: cfg.ropeTheta)
        self.indexerBudget = cfg.indexerBudget
        self.attentionScale = Foundation.pow(Float(cfg.headDim), -0.5)
        precondition(
            cfg.rmsNormWeightOffset == 0,
            "TrackFastModel: baked norm convention expected (offset 0)")

        guard let embed = base.model.children()[unwrapping: "embed_tokens"] as? Embedding else {
            preconditionFailure("TrackFastModel: tower has no embed_tokens")
        }
        self.embedTokens = embed

        let tower = base.model
        var built: [TrackLayer] = []
        for (index, layer) in tower.layers.enumerated() {
            built.append(Self.bind(layer: layer, index: index, cfg: cfg))
        }
        self.layers = built
        self.finalMixer = Self.bindHC(tower.trackChild("hyper_connection_mixer"), cfg: cfg)
        super.init()
    }

    // MARK: binding

    static func bindHC(_ m: Module, cfg: Qwen4ExpTextConfiguration) -> TrackHC {
        let scale = m.trackChild("hc_norm").trackArray("weight")
        let down = TrackProj(m.trackChild("input_mix_weight_down"))
        let up = TrackProj(m.trackChild("input_mix_weight_up"))
        let inject = m.children()[unwrapping: "block_inject_weight"].map { TrackProj($0) }
        let q = (scale * MLXArray(Float(1) / Float(cfg.hcCount), dtype: scale.dtype))
        return TrackHC(
            normScaleQ: q, down: down, inject: inject, up: up, lowrank: cfg.hcLowrank,
            hasInject: inject != nil)
    }

    static func bindGDN(_ m: Module, cfg: Qwen4ExpTextConfiguration) -> TrackGDN {
        let hk = cfg.linearNumKeyHeads, hv = cfg.linearNumValueHeads
        let dk = cfg.linearKeyHeadDim, dv = cfg.linearValueHeadDim
        let keyDim = hk * dk, valueDim = hv * dv
        let convDim = 2 * keyDim + valueDim
        let qkv = TrackProj(m.trackChild("in_proj_qkv"))
        let z = TrackProj(m.trackChild("in_proj_z"))
        let b = TrackProj(m.trackChild("in_proj_b"))
        let a = TrackProj(m.trackChild("in_proj_a"))
        precondition(qkv.rows == convDim && z.rows == valueDim && b.rows == hv && a.rows == hv)
        let proj = TrackMultiProj([qkv, z, b, a])
        let convRaw = m.trackChild("conv1d").trackArray("weight")  // [C, K, 1]
        let kc = convRaw.dim(1)
        let convW = convRaw.reshaped(convDim, kc)
        let aLog = m.trackArray("A_log")
        let negExpALog = -exp(aLog.asType(.float32))
        let dtBias = m.trackArray("dt_bias")
        let normW = m.trackChild("norm").trackArray("weight")
        let out = TrackProj(m.trackChild("out_proj"))
        let geometry = TrackFastKernels.GDNGeometry(
            projWidth: proj.width, convDim: convDim, convKernel: kc,
            hk: hk, hv: hv, dk: dk, dv: dv,
            bOffset: proj.offsets[2], aOffset: proj.offsets[3])
        return TrackGDN(
            proj: proj, convW: convW, negExpALog: negExpALog, dtBias: dtBias, normW: normW,
            out: out, geometry: geometry, zOffset: proj.offsets[1], valueDim: valueDim)
    }

    static func bindAttn(_ m: Module, cfg: Qwen4ExpTextConfiguration) -> TrackAttn {
        let heads = cfg.attentionHeads, kvHeads = cfg.kvHeads, d = cfg.headDim
        let qProj = TrackProj(m.trackChild("q_proj"))  // rows: per head [q(d) | gate(d)]
        precondition(qProj.rows == heads * d * 2)
        // Reorder rows to [q of every head | gate of every head] so the two
        // halves are contiguous slices of the fused output.
        var order: [Int32] = []
        for h in 0 ..< heads { for i in 0 ..< d { order.append(Int32(h * 2 * d + i)) } }
        for h in 0 ..< heads { for i in 0 ..< d { order.append(Int32(h * 2 * d + d + i)) } }
        let qReordered: TrackProj
        switch qProj {
        case .quant(let q): qReordered = .quant(q.rowsReordered(order))
        case .dense(let w): qReordered = .dense(w[MLXArray(order)])
        }
        let k = TrackProj(m.trackChild("k_proj"))
        let v = TrackProj(m.trackChild("v_proj"))
        precondition(k.rows == kvHeads * d && v.rows == kvHeads * d)
        let qNormW = m.trackChild("q_norm").trackArray("weight")
        let kNormW = m.trackChild("k_norm").trackArray("weight")
        let indexer = m.trackChild("indexer")
        let idxProj = TrackProj(indexer.trackChild("index_qk_proj"))
        let idxSplit = cfg.indexerHeads * cfg.indexerHeadDim
        let idxRows = idxProj.rows
        precondition(idxRows == (cfg.indexerHeads + cfg.indexerKVHeads) * cfg.indexerHeadDim)
        let kOrder = (idxSplit ..< idxRows).map { Int32($0) }
        let indexerK: TrackProj
        switch idxProj {
        case .quant(let q): indexerK = .quant(q.rowsReordered(kOrder))
        case .dense(let w): indexerK = .dense(w[MLXArray(kOrder)])
        }
        let out = TrackProj(m.trackChild("o_proj"))
        let qkv = TrackMultiProj([qReordered, k, v, indexerK])
        return TrackAttn(
            qkv: qkv, indexerFull: idxProj, indexerQWidth: idxSplit,
            qNormW: qNormW, kNormW: kNormW, indexerK: indexerK, out: out,
            qWidth: heads * d, kvWidth: kvHeads * d)
    }

    static func bindMoE(_ m: Module, cfg: Qwen4ExpTextConfiguration) -> TrackMoE {
        let gate = m.trackChild("gate")
        let routerW: MLXArray
        switch TrackProj(gate) {
        case .dense(let w): routerW = w.asType(.float32)
        case .quant(let q):
            routerW = dequantized(
                q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize,
                bits: q.bits, mode: q.mode
            ).asType(.float32)
        }
        guard let switchMLP = m.trackChild("switch_mlp") as? SwitchGLU else {
            preconditionFailure("TrackFastModel: switch_mlp is not a SwitchGLU")
        }
        func expert(_ key: String) -> (w: MLXArray, s: MLXArray, b: MLXArray) {
            let p = switchMLP.trackChild(key)
            return (p.trackArray("weight"), p.trackArray("scales"), p.trackArray("biases"))
        }
        guard let qdown = switchMLP.trackChild("down_proj") as? Quantized else {
            preconditionFailure("TrackFastModel: routed experts are not quantized")
        }
        let shared = m.trackChild("shared_expert")
        let sg = TrackProj(shared.trackChild("gate_proj"))
        let su = TrackProj(shared.trackChild("up_proj"))
        let sd = TrackProj(shared.trackChild("down_proj"))
        let sharedGate = TrackProj(m.trackChild("shared_expert_gate"))
        return TrackMoE(
            routerW32: routerW, switchMLP: switchMLP,
            expertGate: expert("gate_proj"), expertUp: expert("up_proj"), expertDown: expert("down_proj"),
            expertGroupSize: qdown.groupSize, expertBits: qdown.bits,
            sharedGateUp: TrackMultiProj([sg, su]),
            sharedDown: sd, sharedGate: sharedGate, topK: cfg.numExpertsPerTok,
            sharedHidden: sg.rows)
    }

    static func bindPLE(_ ple: Qwen4ExpPLELayer, ordinal: Int, cfg: Qwen4ExpTextConfiguration)
        -> TrackPLE
    {
        TrackPLE(
            embedding: ple.pleEmbedding,
            keyProj: TrackProj(ple.trackChild("key_proj")),
            valueProj: TrackProj(ple.trackChild("value_proj")),
            normKeyScale: ple.trackChild("norm_key").trackArray("weight"),
            normQueryScale: ple.trackChild("norm_query").trackArray("weight"),
            normConvScale: ple.trackChild("norm_conv").trackArray("weight"),
            convW: ple.trackChild("conv1d").trackArray("weight"),
            dilation: cfg.ngramSize,
            stateLength: (cfg.pleConvKernelSize - 1) * cfg.ngramSize,
            stateLayerIndex: cfg.pleStateLayerIndex(ordinal: ordinal))
    }

    static func bind(layer: Qwen4ExpDecoderLayer, index: Int, cfg: Qwen4ExpTextConfiguration)
        -> TrackLayer
    {
        let attnHC = bindHC(layer.trackChild("attn_hyper_connection"), cfg: cfg)
        let mlpHC = bindHC(layer.trackChild("mlp_hyper_connection"), cfg: cfg)
        let moe = bindMoE(layer.trackChild("mlp"), cfg: cfg)
        var gdn: TrackGDN? = nil
        var attn: TrackAttn? = nil
        if layer.isLinear {
            gdn = bindGDN(layer.trackChild("linear_attn"), cfg: cfg)
        } else {
            attn = bindAttn(layer.trackChild("self_attn"), cfg: cfg)
        }
        var ple: TrackPLE? = nil
        if let pleLayer = layer.ple, let ordinal = cfg.pleLayerIndices.firstIndex(of: index) {
            ple = bindPLE(pleLayer, ordinal: ordinal, cfg: cfg)
        }
        return TrackLayer(
            index: index, attnHC: attnHC, mlpHC: mlpHC, gdn: gdn, attn: attn, moe: moe, ple: ple)
    }

    // MARK: forward pieces

    private func groupNorm(_ x: MLXArray, scale: MLXArray) -> MLXArray {
        let shape = x.shape
        let grouped = x.reshaped(shape.dropLast() + [hcCount, hidden])
        return MLXFast.rmsNorm(grouped, weight: MLXArray.mlxNone, eps: eps).reshaped(shape) * scale
    }

    /// Hyper-connection mixer over an already-normalized stream: returns the
    /// block input `[B,S,H]` and the inject weights `[B,S,hc]`.
    private func hcMix(_ hc: TrackHC, normed: MLXArray, tag: String = "") -> (MLXArray, MLXArray) {
        let lo = hc.down.apply(normed)  // [B,S,lowrank]
        let act: MLXArray, inj: MLXArray
        // One-token windows: MLX routes the 4-row inject GEMV to `qmv`, a
        // latency-bound launch; the fused mixer-head kernel carries the same
        // arithmetic. Wider windows route to `qmv_wide`, so they keep MLX's
        // own launch.
        if normed.dim(1) == 1, hc.hasInject, case .quant(let iq)? = hc.inject, let ib = iq.biases {
            let r = TrackFastKernels.mixerHead(
                lo: lo, normed: normed, w: iq.weight, s: iq.scales, b: ib,
                groupSize: iq.groupSize, bits: iq.bits, width: hc.lowrank)
            act = r.act; inj = r.inj
        } else {
            inj = hc.inject?.apply(normed) ?? lo  // [B,S,hc]
            act = TrackFastKernels.siluHead(lo: lo, width: hc.lowrank)
        }
        let w = hc.up.apply(act)  // [B,S,W], pre-sigmoid
        if Self.debugTaps != nil, !tag.isEmpty {
            Self.debugTaps?.append((tag + ".normedQ", normed))
            Self.debugTaps?.append((tag + ".lo", lo))
            Self.debugTaps?.append((tag + ".inj", inj))
            Self.debugTaps?.append((tag + ".act", act))
            Self.debugTaps?.append((tag + ".w", w))
        }
        return TrackFastKernels.hcMix(
            w: w, normed: normed, inj: inj, hcCount: hcCount, hidden: hidden,
            hasInject: hc.hasInject)
    }

    private func gdnForward(
        _ g: TrackGDN, _ x: MLXArray, layerIndex: Int,
        evaluation: CBv2RecurrentStateEvaluation, capture: Bool
    ) -> MLXArray {
        let B = x.dim(0), S = x.dim(1)
        let geo = g.geometry
        let proj = g.proj.apply(x)  // [B,S,PROJ_W]
        let state = evaluation.inputState(modelLayerIndex: layerIndex)
        let convState =
            state?.conv ?? MLXArray.zeros([B, geo.convKernel - 1, geo.convDim], dtype: x.dtype)
        let ssm =
            state?.ssm ?? MLXArray.zeros([B, geo.hv, geo.dv, geo.dk], dtype: .float32)
        let r = TrackFastKernels.gdn(
            proj: proj, convState: convState, convW: g.convW, negExpALog: g.negExpALog,
            dtBias: g.dtBias, stateIn: ssm, T: S, capture: capture, geometry: geo)
        let gated = TrackFastKernels.gatedRMS(
            y: r.y, proj: proj, w: g.normW, zOffset: g.zOffset, eps: 1e-6)
        do {
            if capture {
                try evaluation.stageCaptured(
                    modelLayerIndex: layerIndex, conv: r.convOut, ssm: r.stateOut, positions: S)
            } else {
                try evaluation.stage(modelLayerIndex: layerIndex, conv: r.convOut, ssm: r.stateOut)
            }
        } catch {
            preconditionFailure("TrackFastModel: recurrent stage failed at layer \(layerIndex): \(error)")
        }
        return g.out.apply(gated)
    }

    /// The reference's rope tables for this forward, cast to the activation
    /// dtype exactly as `qwen4ExpRopePartial` does: `[S, rot]` each.
    private func ropeTables(offset: Int, count: Int, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
        let (c, s) = rotary.cosSin(qwen4ExpPositions(offset: offset, count: count))
        return (c.asType(dtype).reshaped(count, rotaryDims), s.asType(dtype).reshaped(count, rotaryDims))
    }

    private func attnForward(
        _ a: TrackAttn, _ x: MLXArray, cache: Qwen4ExpCBv2LayerCache,
        rope: (cos: MLXArray, sin: MLXArray)
    ) -> MLXArray {
        let B = x.dim(0), S = x.dim(1)
        let heads = cfg.attentionHeads, kvHeads = cfg.kvHeads, d = cfg.headDim
        let qkv = a.qkv.apply(x)  // q | gate | k | v | indexer k
        // The indexer tape first: its truncation reads the pre-update offset.
        let idxKeys: MLXArray
        if S <= 8 {
            let idxStart = 2 * a.qWidth + 2 * a.kvWidth
            idxKeys = qkv[.ellipsis, idxStart ..< (idxStart + cfg.indexerHeadDim)]
        } else {
            idxKeys = a.indexerFull.apply(x)[.ellipsis, a.indexerQWidth...]
        }
        _ = cache.updateIndexerTape(keys: idxKeys)

        let prep = TrackFastKernels.attnPrep(
            qkv: qkv, qNorm: a.qNormW, kNorm: a.kNormW, cos: rope.cos, sin: rope.sin,
            heads: heads, kvHeads: kvHeads, headDim: d, rotaryDims: rotaryDims, eps: eps)
        let att = cache.updateAndAttend(
            queries: prep.q, keys: prep.k, values: prep.v,
            scale: attentionScale, sinks: nil, keepMask: nil)  // [B,HQ,S,D]
        let out = TrackFastKernels.attnGate(att: att, qkv: qkv, gateOffset: a.qWidth)
        return a.out.apply(out)
    }

    private func moeForward(_ m: TrackMoE, _ x: MLXArray) -> MLXArray {
        Self.moeForwardShared(m, x)
    }

    static func moeForwardShared(_ m: TrackMoE, _ x: MLXArray) -> MLXArray {
        let logits = matmul(x.asType(.float32), m.routerW32.transposed())
        let idx = argPartition(-logits, kth: m.topK - 1, axis: -1)[.ellipsis, ..<m.topK]
        let weights = softmax(takeAlong(logits, idx, axis: -1), axis: -1, precise: true)
        let routed: MLXArray
        if x.dim(0) == 1 && x.dim(1) <= 8 {
            // Custom gather over MLX's own per-row GEMV arithmetic (bit-exact with
            // SwitchGLU); the op's fixed per-call cost dominates at these shapes.
            let S = x.dim(1), K = m.topK, H = x.dim(2)
            let flatIdx = idx.reshaped(S * K).asType(.uint32)
            let xrow = MLXArray((0 ..< (S * K)).map { UInt32($0 / K) })
            let gu = TrackFastMoEKernels.gateUp(
                wg: m.expertGate.w, sg: m.expertGate.s, bg: m.expertGate.b,
                wu: m.expertUp.w, su: m.expertUp.s, bu: m.expertUp.b,
                x: x.reshaped(S, H), idx: flatIdx, xrow: xrow,
                groupSize: m.expertGroupSize, bits: m.expertBits)
            let act = TrackFastKernels.swiglu2(gate: gu.gate, up: gu.up)
            routed = TrackFastMoEKernels.single(
                w: m.expertDown.w, scales: m.expertDown.s, biases: m.expertDown.b,
                x: act, idx: flatIdx, groupSize: m.expertGroupSize, bits: m.expertBits
            ).reshaped(1, S, K, H)
        } else {
            routed = m.switchMLP(x, idx)  // [B,S,K,H]
        }
        let gu = m.sharedGateUp.apply(x)
        let shared = m.sharedDown.apply(TrackFastKernels.swiglu(gu: gu))
        let gate = m.sharedGate.apply(x)  // [B,S,1]
        // The combine kernel folds the K products in the association MLX's small
        // column reduce uses (verified at float precision for every window size).
        return TrackFastKernels.moeCombine(routed: routed, w: weights, shared: shared, gate: gate)
    }

    private func pleForward(
        _ p: TrackPLE, stream: MLXArray, ids: MLXArray,
        evaluation: CBv2RecurrentStateEvaluation, capture: Bool
    ) -> MLXArray {
        let B = stream.dim(0), S = stream.dim(1)
        precondition(B == 1)
        let wide = hcCount * hidden
        let contextLength = max(1, p.dilation - 1)
        let state = evaluation.inputState(modelLayerIndex: p.stateLayerIndex)
        let previous =
            (state?.ssm
                ?? MLXArray.full(
                    [1, contextLength], values: MLXArray(Int32(cfg.eosTokenId)), dtype: .int32))
            .asType(ids.dtype)
        let convState =
            state?.conv ?? MLXArray.zeros([1, p.stateLength, wide], dtype: stream.dtype)

        let embedded = p.embedding(ids, previousContext: previous).asType(stream.dtype)
        let key = groupNorm(p.keyProj.apply(embedded), scale: p.normKeyScale)
            .reshaped(B, S, hcCount, hidden)
        let value = p.valueProj.apply(embedded)
        let query = groupNorm(stream, scale: p.normQueryScale).reshaped(B, S, hcCount, hidden)
        var gate = (key * query).sum(axis: -1, keepDims: true) / Foundation.sqrt(Float(hidden))
        let floor = MLXArray(Float(1e-6), dtype: gate.dtype)
        gate = MLX.sqrt(maximum(MLX.abs(gate), floor)) * MLX.sign(gate)
        let gated = (sigmoid(gate) * value[.ellipsis, .newAxis, 0...]).reshaped(B, S, wide)
        let normed = groupNorm(gated, scale: p.normConvScale)
        let full = concatenated([convState, normed], axis: 1)  // [1, n+S, wide]
        let convolved = silu(
            conv1d(full, p.convW, stride: 1, padding: 0, dilation: p.dilation, groups: wide))
        let history = concatenated([previous, ids], axis: 1)
        do {
            if capture {
                let n = p.stateLength
                let convStack = asStrided(full, [S, n, wide], strides: [wide, wide, 1], offset: wide)
                let contextStack = asStrided(
                    history.asType(.int32), [S, contextLength], strides: [1, 1], offset: 1)
                try evaluation.stageCaptured(
                    modelLayerIndex: p.stateLayerIndex, conv: convStack, ssm: contextStack,
                    positions: S)
            } else {
                try evaluation.stage(
                    modelLayerIndex: p.stateLayerIndex,
                    conv: full[0..., (-p.stateLength)..., 0...],
                    ssm: history[0..., (-contextLength)...].asType(.int32))
            }
        } catch {
            preconditionFailure("TrackFastModel: PLE stage failed: \(error)")
        }
        return gated + convolved
    }

    /// Both tower streams. `caches` is the compact attention layout.
    ///
    /// The inject of one block and the norm of the next mixer are one launch
    /// (`injectNorm`), so a block's output is carried as `pending` until the
    /// next mixer consumes it.
    func fastStreams(
        _ ids: MLXArray, inputEmbeddings: MLXArray?, caches: [Qwen4ExpCBv2LayerCache],
        recurrentState: [CBv2RecurrentStateEvaluation], offset: Int, capture: Bool
    ) -> (mixed: MLXArray, multi: MLXArray) {
        precondition(recurrentState.count == 1)
        let evaluation = recurrentState[0]
        var residual = inputEmbeddings ?? embedTokens(ids)  // [B,S,H] until tiled
        var tile = true
        var pendingOut: MLXArray? = nil
        var pendingInject: MLXArray? = nil
        var attentionIndex = 0
        var stream = residual
        let ropeTab = ropeTables(offset: offset, count: ids.dim(1), dtype: residual.dtype)

        for layer in layers {
            var normed: MLXArray
            if let ple = layer.ple {
                // Materialize the stream, add the PLE block, then norm.
                (stream, _) = TrackFastKernels.injectNorm(
                    residual: residual, out: pendingOut, inject: pendingInject,
                    scale: layer.attnHC.normScaleQ, hcCount: hcCount, hidden: hidden, eps: eps,
                    tile: tile)
                stream =
                    stream
                    + pleForward(
                        ple, stream: stream, ids: ids, evaluation: evaluation, capture: capture)
                (stream, normed) = TrackFastKernels.injectNorm(
                    residual: stream, out: nil, inject: nil,
                    scale: layer.attnHC.normScaleQ, hcCount: hcCount, hidden: hidden, eps: eps,
                    tile: false)
            } else {
                (stream, normed) = TrackFastKernels.injectNorm(
                    residual: residual, out: pendingOut, inject: pendingInject,
                    scale: layer.attnHC.normScaleQ, hcCount: hcCount, hidden: hidden, eps: eps,
                    tile: tile)
            }
            tile = false
            var (input, injectW) = hcMix(layer.attnHC, normed: normed, tag: "L\(layer.index).attn.hc")
            Self.debugTaps?.append(("L\(layer.index).attn.stream_in", stream))
            Self.debugTaps?.append(("L\(layer.index).attn.input", input))
            let attended: MLXArray
            if let gdn = layer.gdn {
                attended = gdnForward(
                    gdn, input, layerIndex: layer.index, evaluation: evaluation, capture: capture)
            } else {
                let cache = caches[attentionIndex]
                attentionIndex += 1
                attended = attnForward(layer.attn!, input, cache: cache, rope: ropeTab)
            }
            Self.debugTaps?.append(("L\(layer.index).attn.out", attended))
            (stream, normed) = TrackFastKernels.injectNorm(
                residual: stream, out: attended, inject: injectW,
                scale: layer.mlpHC.normScaleQ, hcCount: hcCount, hidden: hidden, eps: eps,
                tile: false)
            Self.debugTaps?.append(("L\(layer.index).mlp.stream_in", stream))
            (input, injectW) = hcMix(layer.mlpHC, normed: normed, tag: "L\(layer.index).mlp.hc")
            Self.debugTaps?.append(("L\(layer.index).mlp.input", input))
            pendingOut = moeForward(layer.moe, input)
            Self.debugTaps?.append(("L\(layer.index).mlp.out", pendingOut!))
            pendingInject = injectW
            residual = stream
            // Dispatch the graph so far: the GPU starts on these layers while the
            // CPU keeps building the rest (the build is otherwise GPU-idle time).
            if Self.asyncChunk > 0, (layer.index + 1) % Self.asyncChunk == 0 {
                asyncEval(stream)
            }
        }
        let (multi, finalNormed) = TrackFastKernels.injectNorm(
            residual: residual, out: pendingOut, inject: pendingInject,
            scale: finalMixer.normScaleQ, hcCount: hcCount, hidden: hidden, eps: eps,
            tile: false)
        let (mixed, _) = hcMix(finalMixer, normed: finalNormed)
        return (mixed, multi)
    }

    // MARK: routing

    private func typedCaches(_ caches: [KVCache]) -> [Qwen4ExpCBv2LayerCache]? {
        var out: [Qwen4ExpCBv2LayerCache] = []
        out.reserveCapacity(caches.count)
        for c in caches {
            guard let t = c as? Qwen4ExpCBv2LayerCache else { return nil }
            out.append(t)
        }
        return out
    }

    /// The fast path serves one unpositioned row whose context stays below
    /// the indexer budget; everything else goes to the wrapped model.
    private func fastPlan(
        tokens: MLXArray, caches: [KVCache], recurrentState: [CBv2RecurrentStateEvaluation],
        positionIds: MLXArray?
    ) -> (caches: [Qwen4ExpCBv2LayerCache], offset: Int)? {
        guard Self.enabled, positionIds == nil, tokens.ndim == 2, tokens.dim(0) == 1,
            recurrentState.count == 1,
            let typed = typedCaches(caches), typed.count == cfg.fullAttentionLayerIndices.count,
            let first = typed.first, first.rows.count == 1
        else { return nil }
        let offset = first.rows[0].absoluteOffset
        let S = tokens.dim(1)
        guard offset + S <= indexerBudget else { return nil }
        return (typed, offset)
    }

    func streamsOrDelegate(
        _ tokens: MLXArray, inputEmbeddings: MLXArray?, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?, capture: Bool
    ) -> (mixed: MLXArray, multi: MLXArray)? {
        guard let plan = fastPlan(
            tokens: tokens, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds)
        else { return nil }
        return fastStreams(
            tokens, inputEmbeddings: inputEmbeddings, caches: plan.caches,
            recurrentState: recurrentState, offset: plan.offset, capture: capture)
    }
}

// MARK: - LanguageModel (delegated)

extension TrackQwen4ExpFastModel: LanguageModel, KVCacheDimensionProvider {
    public var kvHeads: [Int] { base.kvHeads }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        base.sanitize(weights: weights)
    }
    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        base.sanitize(weights: weights, metadata: metadata)
    }
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        try base.prepare(input, cache: cache, windowSize: windowSize)
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        base(inputs, cache: cache)
    }
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        base.newCache(parameters: parameters)
    }
}

// MARK: - CBv2 conformances

extension TrackQwen4ExpFastModel: CBv2PositionAxisProviding {
    public var cbv2PositionAxisCount: Int? { base.cbv2PositionAxisCount }
}

extension TrackQwen4ExpFastModel: CBv2KeepMaskRequiringModel {
    public var cbv2RequiresKeepMask: Bool { base.cbv2RequiresKeepMask }
}

extension TrackQwen4ExpFastModel: CBv2PositionedRecurrentLanguageModelForwardable,
    CBv2PositionedRecurrentEmbeddingForwardable
{
    public var cbv2Capabilities: CBv2ModelCapabilities { base.cbv2Capabilities }
    public var cbv2RecurrentStateSpec: CBv2RecurrentStateSpec { base.cbv2RecurrentStateSpec }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation]
    ) -> MLXArray {
        cbv2Forward(tokens, caches: caches, recurrentState: recurrentState, positionIds: nil)
    }

    public func cbv2Forward(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        if let s = streamsOrDelegate(
            tokens, inputEmbeddings: nil, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, capture: false)
        {
            return base.head(s.mixed)
        }
        return base.cbv2Forward(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
    }

    public var supportsVisionSpanPrefill: Bool { base.supportsVisionSpanPrefill }
    public var supportsCausalVisionPrefill: Bool { base.supportsCausalVisionPrefill }

    public func scaledInputEmbeddings(_ inputs: MLXArray) -> MLXArray {
        base.scaledInputEmbeddings(inputs)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?
    ) -> MLXArray {
        base.embeddingForward(inputs, inputEmbedding: inputEmbedding, cache: cache)
    }

    public func embeddingForward(
        _ inputs: MLXArray, inputEmbedding: MLXArray, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> MLXArray {
        if let s = streamsOrDelegate(
            inputs, inputEmbeddings: inputEmbedding, caches: cache ?? [],
            recurrentState: recurrentState, positionIds: positionIds, capture: false)
        {
            return base.head(s.mixed)
        }
        return base.embeddingForward(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds)
    }
}

extension TrackQwen4ExpFastModel: CBv2RecurrentLanguageModelPrefillForwardable {
    public var cbv2SupportsPackedPrefill: Bool { base.cbv2SupportsPackedPrefill }

    public func cbv2RecurrentPrefill(
        _ inputs: MLXArray, inputEmbedding: MLXArray?, cache: [KVCache]?,
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?,
        requirement: CBv2PrefillRequirement
    ) -> MLXArray {
        if let s = streamsOrDelegate(
            inputs, inputEmbeddings: inputEmbedding, caches: cache ?? [],
            recurrentState: recurrentState, positionIds: positionIds, capture: false)
        {
            switch requirement {
            case .evaluationOnly:
                return s.mixed[0..., -1, 0 ..< 1]
            case .lastPositionLogits:
                return base.head(s.mixed[0..., -1, 0...])
            }
        }
        return base.cbv2RecurrentPrefill(
            inputs, inputEmbedding: inputEmbedding, cache: cache,
            recurrentState: recurrentState, positionIds: positionIds, requirement: requirement)
    }
}

extension TrackQwen4ExpFastModel: CBv2RecurrentMTPForwardable {
    public var cbv2MTPTargetIdentity: ObjectIdentifier { ObjectIdentifier(base) }

    public func cbv2ForwardWithHidden(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        if let s = streamsOrDelegate(
            tokens, inputEmbeddings: nil, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, capture: false)
        {
            return (base.head(s.mixed), s.multi)
        }
        return base.cbv2ForwardWithHidden(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
    }
}

extension TrackQwen4ExpFastModel: CBv2MTPPolicyTopTwoProviding {
    public func cbv2MTPTopTwo(_ logits: MLXArray) -> (ids: MLXArray, values: MLXArray) {
        base.cbv2MTPTopTwo(logits)
    }
}

extension TrackQwen4ExpFastModel: CBv2RecurrentCaptureMTPForwardable {
    public func cbv2ForwardWithHiddenCaptured(
        _ tokens: MLXArray, caches: [KVCache],
        recurrentState: [CBv2RecurrentStateEvaluation], positionIds: MLXArray?
    ) -> (logits: MLXArray, lastHidden: MLXArray) {
        if let s = streamsOrDelegate(
            tokens, inputEmbeddings: nil, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, capture: true)
        {
            return (base.head(s.mixed), s.multi)
        }
        return base.cbv2ForwardWithHiddenCaptured(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
    }
}
