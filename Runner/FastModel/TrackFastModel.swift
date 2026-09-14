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

/// Last-axis slices of a fused `[B, S, W]` row (indexer K, top-k, …) are not
/// MLX row-contiguous when any leading dim is > 1: the parent row pitch is
/// still `W`, so `ensureRowContiguous: true` custom kernels `copy_gpu` the
/// whole tensor, and `concatenated` on the indexer tape gathers every step.
/// Sequence-axis tails of `B == 1` (PLE conv state) stay packed and are left
/// alone. `TRACK_CONTIGUITY=0` restores the strided views.
enum TrackContiguity {
    static let enabled =
        ProcessInfo.processInfo.environment["TRACK_CONTIGUITY"] != "0"

    static func packedLastAxis(_ x: MLXArray) -> MLXArray {
        guard enabled, x.ndim >= 2 else { return x }
        for i in 0 ..< (x.ndim - 1) where x.dim(i) > 1 {
            return x.contiguous()
        }
        return x
    }
}

/// An affine-quantized projection `[N, K]`.

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
    /// The whole `index_qk_proj` (q and k rows). Used when this round's
    /// keep-mask consumer will read the q half, so q and k share the vendor
    /// GEMM (its rounding depends on N).
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
    let routerW16: MLXArray  // [E, H] bf16: the values routerW32 was cast from
    let switchMLP: SwitchGLU
    /// The same three loaded expert projections `switchMLP` itself calls, so
    /// the sorted combine can run them without the closing scatter/unsort.
    /// References to the loaded modules; no new or transformed weights.
    let p12SortedParts: (gate: SwitchLinear, up: SwitchLinear, down: SwitchLinear)?
    /// The routed experts' quantized arrays, for the custom gather path.
    let expertGate: (w: MLXArray, s: MLXArray, b: MLXArray)
    let expertUp: (w: MLXArray, s: MLXArray, b: MLXArray)
    /// True when `expertGate` / `expertUp` share one interleaved buffer.
    let fusedGateUp: Bool
    /// 1 = split, 2 = gate+up slab, 3 = gate+up+down slab. The down gather
    /// reads the same allocation when this is 3.
    let fusedTiles: Int
    let rowsPerDown: Int
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
    /// The same weight as `[wide, K]`, for the fused conv.
    let convW2: MLXArray
    let dilation: Int
    let stateLength: Int
    let stateLayerIndex: Int
    let pleLayerIndex: Int
}

struct TrackLayer {
    let index: Int
    let attnHC: TrackHC
    let mlpHC: TrackHC
    let gdn: TrackGDN?
    let attn: TrackAttn?
    let moe: TrackMoE
    let moePairReplay: (@Sendable ([MLXArray]) -> [MLXArray])?
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
    let injectNormReplay: @Sendable ([MLXArray]) -> [MLXArray]
    private let pleSSMStaging: TrackPLESSMStaging
    let rotaryDims: Int
    let rotary: Qwen4ExpRotary
    let indexerBudget: Int
    let attentionScale: Float
    /// Preallocated GDN/PLE snapshot stacks for MTP capture-verify restore.
    private var gdnStateStore: TrackGDNStateStore?

    /// Debug taps (tests): when set, every layer's output stream and the block
    /// inputs/outputs are appended here.
    nonisolated(unsafe) static var debugTaps: [(String, MLXArray)]? = nil
    /// Layers per partial dispatch inside a forward (0 = one dispatch per step).
    /// `TRACK_ASYNC_CHUNK` at startup: a positive integer, else 1 (`3` restores
    /// the previous default).
    nonisolated(unsafe) public static var asyncChunk: Int = resolvedAsyncChunk(
        ProcessInfo.processInfo.environment["TRACK_ASYNC_CHUNK"])
    /// Layers in the first partial-dispatch chunk (0 = same as asyncChunk):
    /// the first dispatch lands right after the PLE layer.
    nonisolated(unsafe) public static var asyncFirst: Int = 2
    /// Layer count at an optional second dispatch (0 = none).
    nonisolated(unsafe) public static var asyncSecond: Int = 0
    static func resolvedAsyncChunk(_ raw: String?) -> Int {
        raw.flatMap(Int.init).flatMap { $0 > 0 ? $0 : nil } ?? 1
    }

    /// Kill switch for A/B: `TRACK_FAST_FORWARD=0` routes every forward to the
    /// wrapped model.
    static let enabled: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_FAST_FORWARD"] ?? "1") != "0"
    }()

    /// Skip the indexer's q-projection unless this round's top-k keep-mask
    /// will read it. `TRACK_QPROJ_GUARD=0` restores the fused `index_qk_proj`
    /// on every wide window.
    static let qprojGuard: Bool = {
        (ProcessInfo.processInfo.environment["TRACK_QPROJ_GUARD"] ?? "1") != "0"
    }()

    public init(base: Qwen4ExpModel) {
        self.base = base
        let cfg = base.configuration
        self.cfg = cfg
        self.hcCount = cfg.hcCount
        self.hidden = cfg.hiddenSize
        self.eps = cfg.rmsNormEps
        self.injectNormReplay = Self.makeInjectNormReplay(
            hcCount: cfg.hcCount, hidden: cfg.hiddenSize, eps: cfg.rmsNormEps)
        self.pleSSMStaging = TrackPLESSMStaging(contextLength: max(1, cfg.ngramSize - 1))
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

    static func bindMoE(
        _ m: Module, cfg: Qwen4ExpTextConfiguration,
        modulePath: String? = nil
    ) -> TrackMoE {
        let gate = m.trackChild("gate")
        let routerW: MLXArray
        let routerW16: MLXArray
        switch TrackProj(gate) {
        case .dense(let w):
            routerW16 = w
            routerW = w.asType(.float32)
        case .quant(let q):
            routerW16 = dequantized(
                q.weight, scales: q.scales, biases: q.biases, groupSize: q.groupSize,
                bits: q.bits, mode: q.mode)
            routerW = routerW16.asType(.float32)
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
        let slab = modulePath.flatMap { TrackExpertLayout.slab(for: $0) }
        let fusedTiles: Int
        let rowsPerDown: Int
        let fusedGateUp: Bool
        let expertGate: (w: MLXArray, s: MLXArray, b: MLXArray)
        let expertUp: (w: MLXArray, s: MLXArray, b: MLXArray)
        let expertDown: (w: MLXArray, s: MLXArray, b: MLXArray)
        if let slab {
            fusedTiles = 3
            rowsPerDown = TrackExpertLayout.loadedTable?.rowsPerDown ?? 0
            fusedGateUp = true
            let fused = (w: slab.weight, s: slab.scales, b: slab.biases)
            expertGate = fused
            expertUp = fused
            expertDown = fused
        } else {
            fusedGateUp = TrackExpertLayout.enabled && switchMLP.hasFusedGateUp
            fusedTiles = fusedGateUp ? 2 : 1
            rowsPerDown = 0
            if fusedGateUp {
                let fused = expert("gate_up_proj")
                expertGate = fused
                expertUp = fused
            } else {
                expertGate = expert("gate_proj")
                expertUp = expert("up_proj")
            }
            expertDown = expert("down_proj")
        }
        let shared = m.trackChild("shared_expert")
        let sg = TrackProj(shared.trackChild("gate_proj"))
        let su = TrackProj(shared.trackChild("up_proj"))
        let sd = TrackProj(shared.trackChild("down_proj"))
        let sharedGate = TrackProj(m.trackChild("shared_expert_gate"))
        return TrackMoE(
            routerW32: routerW, routerW16: routerW16, switchMLP: switchMLP,
            p12SortedParts: {
                guard fusedTiles == 1,
                    let g = switchMLP.children()[unwrapping: "gate_proj"] as? SwitchLinear,
                    let u = switchMLP.children()[unwrapping: "up_proj"] as? SwitchLinear,
                    let d = switchMLP.children()[unwrapping: "down_proj"] as? SwitchLinear
                else { return nil }
                return (gate: g, up: u, down: d)
            }(),
            expertGate: expertGate, expertUp: expertUp, fusedGateUp: fusedGateUp,
            fusedTiles: fusedTiles, rowsPerDown: rowsPerDown,
            expertDown: expertDown,
            expertGroupSize: qdown.groupSize, expertBits: qdown.bits,
            sharedGateUp: TrackMultiProj([sg, su]),
            sharedDown: sd, sharedGate: sharedGate, topK: cfg.numExpertsPerTok,
            sharedHidden: sg.rows)
    }

    static func bindPLE(_ ple: Qwen4ExpPLELayer, ordinal: Int, cfg: Qwen4ExpTextConfiguration)
        -> TrackPLE
    {
        let convW = ple.trackChild("conv1d").trackArray("weight")
        return TrackPLE(
            embedding: ple.pleEmbedding,
            keyProj: TrackProj(ple.trackChild("key_proj")),
            valueProj: TrackProj(ple.trackChild("value_proj")),
            normKeyScale: ple.trackChild("norm_key").trackArray("weight"),
            normQueryScale: ple.trackChild("norm_query").trackArray("weight"),
            normConvScale: ple.trackChild("norm_conv").trackArray("weight"),
            convW: convW,
            convW2: convW.reshaped(convW.dim(0), convW.dim(1)),
            dilation: cfg.ngramSize,
            stateLength: (cfg.pleConvKernelSize - 1) * cfg.ngramSize,
            stateLayerIndex: cfg.pleStateLayerIndex(ordinal: ordinal),
            pleLayerIndex: ordinal)
    }

    static func bind(layer: Qwen4ExpDecoderLayer, index: Int, cfg: Qwen4ExpTextConfiguration)
        -> TrackLayer
    {
        let attnHC = bindHC(layer.trackChild("attn_hyper_connection"), cfg: cfg)
        let mlpHC = bindHC(layer.trackChild("mlp_hyper_connection"), cfg: cfg)
        let moe = bindMoE(
            layer.trackChild("mlp"), cfg: cfg,
            modulePath: "model.layers.\(index).mlp.switch_mlp")
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
            index: index, attnHC: attnHC, mlpHC: mlpHC, gdn: gdn, attn: attn, moe: moe,
            moePairReplay: makeMoEPairReplay(moe, hcCount: cfg.hcCount), ple: ple)
    }

    // MARK: forward pieces

    private func groupNorm(_ x: MLXArray, scale: MLXArray) -> MLXArray {
        let shape = x.shape
        let grouped = x.reshaped(shape.dropLast() + [hcCount, hidden])
        return MLXFast.rmsNorm(grouped, weight: MLXArray.mlxNone, eps: eps).reshaped(shape) * scale
    }

    /// Hyper-connection mixer over an already-normalized stream: returns the
    /// block input `[B,S,H]` and the inject weights `[B,S,hc]`.
    /// Returns the block input, the inject weights and, when `emitF32`, the
    /// input as float32 (the router's operand) written by the same launch.
    private func hcMix(
        _ hc: TrackHC, normed: MLXArray, tag: String = "", emitF32: Bool = false,
        precomputedInj: MLXArray? = nil
    ) -> (input: MLXArray, inject: MLXArray, inputF32: MLXArray?) {
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
                    hasInject: hc.hasInject, emitF32: emitF32)
                let f32 = emitF32 ? u.inputF32.reshaped(1, S, hidden) : nil
                if Self.debugTaps != nil, !tag.isEmpty {
                    Self.debugTaps?.append((tag + ".normedQ", normed))
                    Self.debugTaps?.append((tag + ".lo", d.lo.reshaped(1, S, -1)))
                    Self.debugTaps?.append((tag + ".inj", injIn.reshaped(1, S, -1)))
                }
                return (u.input.reshaped(1, S, hidden), u.inject.reshaped(1, S, hcCount), f32)
            }
        }
        let lo = hc.down.apply(normed)  // [B,S,lowrank]
        let act: MLXArray, inj: MLXArray
        // One-token windows: MLX routes the 4-row inject GEMV to `qmv`, a
        // latency-bound launch; the fused mixer-head kernel carries the same
        // arithmetic. Wider windows route to `qmv_wide`, so they keep MLX's
        // own launch.
        if let precomputedInj {
            inj = precomputedInj
            if hc.hasInject,
                let fused = TrackPrefillMixerAct.apply(hc.down, x: normed, width: hc.lowrank)
            {
                act = fused
            } else {
                act = TrackFastKernels.siluHead(lo: lo, width: hc.lowrank)
            }
        } else if normed.dim(1) == 1, hc.hasInject, case .quant(let iq)? = hc.inject, let ib = iq.biases
        {
            let r = TrackFastKernels.mixerHead(
                lo: lo, normed: normed, w: iq.weight, s: iq.scales, b: ib,
                groupSize: iq.groupSize, bits: iq.bits, width: hc.lowrank)
            act = r.act; inj = r.inj
        } else {
            inj = hc.inject?.apply(normed) ?? lo  // [B,S,hc]
            if hc.hasInject,
                let fused = TrackPrefillMixerAct.apply(hc.down, x: normed, width: hc.lowrank)
            {
                act = fused
            } else {
                act = TrackFastKernels.siluHead(lo: lo, width: hc.lowrank)
            }
        }
        let w = hc.up.apply(act)  // [B,S,W], pre-sigmoid
        if Self.debugTaps != nil, !tag.isEmpty {
            Self.debugTaps?.append((tag + ".normedQ", normed))
            Self.debugTaps?.append((tag + ".lo", lo))
            Self.debugTaps?.append((tag + ".inj", inj))
            Self.debugTaps?.append((tag + ".act", act))
            Self.debugTaps?.append((tag + ".w", w))
        }
        let r = TrackFastKernels.hcMix(
            w: w, normed: normed, inj: inj, hcCount: hcCount, hidden: hidden,
            hasInject: hc.hasInject)
        return (r.input, r.inject, nil)
    }

    private func gdnForward(
        _ g: TrackGDN, _ x: MLXArray, layerIndex: Int,
        evaluation: CBv2RecurrentStateEvaluation, capture: Bool,
        residual: MLXArray? = nil, inject: MLXArray? = nil
    ) -> MLXArray {
        let B = x.dim(0), S = x.dim(1)
        // A wide window already runs the four input projections as four
        // separate GEMMs (the split-K choice depends on N). `separate` drops
        // only the concatenation that followed them: the prep kernel reads the
        // beta/alpha gates from their own buffers and the gated RMS kernel
        // reads z from its own, so every GEMM, shape and rounding is unchanged.
        let separate =
            TrackP12Prefill.splitGDN && TrackP12Prefill.eligible(x) && !capture
            && g.proj.parts.count == 4
        let geo = separate ? TrackP12Prefill.splitGeometry(g.geometry) : g.geometry
        let prof = TrackFastProfile.prefill != nil && S >= TrackFastProfile.minWindow
        var pt = prof ? CFAbsoluteTimeGetCurrent() : 0
        let split = separate ? g.proj.parts.map { $0.apply(x) } : nil
        let proj = split?[0] ?? g.proj.apply(x)  // [B,S,PROJ_W]
        if prof { TrackFastProfile.tick("gdn.proj", &pt, split ?? [proj]) }
        let state = evaluation.inputState(modelLayerIndex: layerIndex)
        let adopted =
            S <= TrackGDNStateRestore.maxVerifyWidth
            ? gdnStateStore?.adoptionInputs(
                modelLayerIndex: layerIndex, inputConv: state?.conv)
            : nil
        let convState =
            adopted?.convBank
            ?? state?.conv
            ?? MLXArray.zeros([B, geo.convKernel - 1, geo.convDim], dtype: x.dtype)
        let ssm =
            adopted?.ssmBank
            ?? state?.ssm
            ?? MLXArray.zeros([B, geo.hv, geo.dv, geo.dk], dtype: .float32)
        let gated: MLXArray, convOut: MLXArray, stateOut: MLXArray
        if adopted == nil, let fused = TrackFastGDNDecode.apply(
            proj: proj, convState: convState, convW: g.convW, negExpALog: g.negExpALog,
            dtBias: g.dtBias, stateIn: ssm, normW: g.normW, zOffset: g.zOffset,
            eps: 1e-6, capture: capture, geometry: geo)
        {
            (gated, convOut, stateOut) = (fused.gated, fused.convOut, fused.stateOut)
            if prof { TrackFastProfile.tick("gdn.decodeFused", &pt, [gated, stateOut, convOut]) }
        } else {
            let r = TrackFastKernels.gdn(
                proj: proj, convState: convState, convW: g.convW, negExpALog: g.negExpALog,
                dtBias: g.dtBias, stateIn: ssm, T: S, capture: capture, geometry: geo,
                separateBA: split.map { (b: $0[2], a: $0[3]) },
                adoptedSlot: adopted?.flag)
            if prof { TrackFastProfile.tick("gdn.prep+lean", &pt, [r.y, r.stateOut, r.convOut]) }
            gated = TrackFastKernels.gatedRMS(
                y: r.y, proj: split?[1] ?? proj, w: g.normW,
                zOffset: separate ? 0 : g.zOffset, eps: 1e-6)
            (convOut, stateOut) = (r.convOut, r.stateOut)
            if prof { TrackFastProfile.tick("gdn.gatedRMS", &pt, [gated]) }
        }
        do {
            if capture {
                try stageRecurrentCapture(
                    evaluation, modelLayerIndex: layerIndex, conv: convOut, ssm: stateOut,
                    positions: S)
            } else {
                try evaluation.stage(modelLayerIndex: layerIndex, conv: convOut, ssm: stateOut)
            }
        } catch {
            preconditionFailure("TrackFastModel: recurrent stage failed at layer \(layerIndex): \(error)")
        }
        if let residual, let inject,
            let fused = TrackResidualEpilogue.project(
                g.out, x: gated, residual: residual, inject: inject,
                hcCount: hcCount, hidden: hidden)
        {
            if prof { TrackFastProfile.tick("gdn.out", &pt, [fused]) }
            return fused
        }
        let o = g.out.apply(gated)
        if prof { TrackFastProfile.tick("gdn.out", &pt, [o]) }
        return o
    }

    /// The reference's rope tables for this forward, cast to the activation
    /// dtype exactly as `qwen4ExpRopePartial` does: `[S, rot]` each.
    private func ropeTables(offset: Int, count: Int, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
        let (c, s) = rotary.cosSin(qwen4ExpPositions(offset: offset, count: count))
        return (c.asType(dtype).reshaped(count, rotaryDims), s.asType(dtype).reshaped(count, rotaryDims))
    }

    private func attnForward(
        _ a: TrackAttn, _ x: MLXArray, cache: Qwen4ExpCBv2LayerCache,
        rope: (cos: MLXArray, sin: MLXArray),
        residual: MLXArray? = nil, inject: MLXArray? = nil
    ) -> MLXArray {
        let S = x.dim(1)
        let heads = cfg.attentionHeads, kvHeads = cfg.kvHeads, d = cfg.headDim
        let splitInputs: [MLXArray]?
        let qkv: MLXArray
        if TrackP12Prefill.splitAttention && TrackP12Prefill.eligible(x)
            && a.qkv.parts.count == 4
        {
            let parts = a.qkv.parts.prefix(3).map { $0.apply(x) }
            splitInputs = parts
            qkv = parts[0]
        } else {
            splitInputs = nil
            qkv =
                TrackP12Prefill.omitUnusedIndexer && TrackP12Prefill.eligible(x)
                && a.qkv.parts.count == 4
                ? concatenated(a.qkv.parts.prefix(3).map { $0.apply(x) }, axis: -1)
                : a.qkv.apply(x)
        }
        // The indexer tape first: its truncation reads the pre-update offset.
        // Last-axis slices of the fused QKV / index_qk_proj row are packed
        // only when B == S == 1; pack them before the tape concat so a
        // custom kernel (or the next concat) does not gather the parent pitch.
        // Indexer q has one consumer: the top-k keep-mask, which runs only
        // after the tape exceeds the budget. Decode (S <= 8) already omits
        // those rows from the fused qkv. Wide windows skip the q half of
        // `index_qk_proj` unless this round will read it, or the guard is off.
        let idxKeys: MLXArray
        if S <= 8 {
            let idxStart = 2 * a.qWidth + 2 * a.kvWidth
            idxKeys = TrackContiguity.packedLastAxis(
                qkv[.ellipsis, idxStart ..< (idxStart + cfg.indexerHeadDim)])
        } else if !Self.qprojGuard
            || (cache.rows.first?.absoluteOffset ?? 0) + S > indexerBudget
        {
            idxKeys = TrackContiguity.packedLastAxis(
                a.indexerFull.apply(x)[.ellipsis, a.indexerQWidth...])
        } else {
            idxKeys = TrackContiguity.packedLastAxis(a.indexerK.apply(x))
        }
        TrackIndexerTape.appendCBv2(cache: cache, keys: idxKeys, capacity: indexerBudget)

        let prep: (q: MLXArray, k: MLXArray, v: MLXArray)
        let fuseRing: TrackKVFuse.Ring? = {
            guard TrackKVFuse.enabled, let ring = TrackKVFuse.ring(from: cache),
                TrackKVFuse.canAppend(offset: ring.offset, count: S, cap: ring.cap, wrap: false)
            else { return nil }
            return ring
        }()
        if let parts = splitInputs {
            if let ring = fuseRing {
                prep = TrackP12Prefill.attnPrepSplitFused(
                    qGate: parts[0], k: parts[1], v: parts[2],
                    qNorm: a.qNormW, kNorm: a.kNormW, cos: rope.cos, sin: rope.sin,
                    kCache: ring.keys, vCache: ring.values, writeOffset: ring.offset,
                    heads: heads, kvHeads: kvHeads, headDim: d, rotaryDims: rotaryDims, eps: eps)
            } else {
                prep = TrackP12Prefill.attnPrepSplit(
                    qGate: parts[0], k: parts[1], v: parts[2],
                    qNorm: a.qNormW, kNorm: a.kNormW, cos: rope.cos, sin: rope.sin,
                    heads: heads, kvHeads: kvHeads, headDim: d, rotaryDims: rotaryDims, eps: eps)
            }
        } else if let ring = fuseRing {
            prep = TrackFastKernels.attnPrepFused(
                qkv: qkv, qNorm: a.qNormW, kNorm: a.kNormW, cos: rope.cos, sin: rope.sin,
                kCache: ring.keys, vCache: ring.values, writeOffset: ring.offset,
                heads: heads, kvHeads: kvHeads, headDim: d, rotaryDims: rotaryDims, eps: eps)
        } else {
            prep = TrackFastKernels.attnPrep(
                qkv: qkv, qNorm: a.qNormW, kNorm: a.kNormW, cos: rope.cos, sin: rope.sin,
                heads: heads, kvHeads: kvHeads, headDim: d, rotaryDims: rotaryDims, eps: eps)
        }
        if let row = cache.rows.first {
            let kLen = row.absoluteOffset + S
            let vaOK = TrackVATransposes.eligible(
                q: prep.q, kHeads: prep.k.dim(1), kLen: kLen, cacheRows: cache.rows.count)
            let qsaOK = TrackQSALoads.eligible(
                q: prep.q, kHeads: prep.k.dim(1), kLen: kLen, cacheRows: cache.rows.count)
            if vaOK || qsaOK {
                let (cachedK, cachedV) = row.update(keys: prep.k, values: prep.v)
                cache.setRows(cache.rows)
                // VA second-pass transposes (D=256) win when their toggle is on;
                // TRACK_VA_BATCHED_TRANSPOSE=0 restores QSA loads (main-tip).
                let att =
                    vaOK
                    ? TrackVATransposes.attend(
                        q: prep.q, k: cachedK, v: cachedV, scale: attentionScale)
                    : TrackQSALoads.attend(
                        q: prep.q, k: cachedK, v: cachedV, scale: attentionScale)
                let out = TrackFastKernels.attnGate(att: att, qkv: qkv, gateOffset: a.qWidth)
                return a.out.apply(out)
            }
        }
        let att = cache.updateAndAttend(
            queries: prep.q, keys: prep.k, values: prep.v,
            scale: attentionScale, sinks: nil, keepMask: nil)  // [B,HQ,S,D]
        let out = TrackFastKernels.attnGate(att: att, qkv: qkv, gateOffset: a.qWidth)
        if let residual, let inject,
            let fused = TrackResidualEpilogue.project(
                a.out, x: out, residual: residual, inject: inject,
                hcCount: hcCount, hidden: hidden)
        {
            return fused
        }
        return a.out.apply(out)
    }

    /// Compiled inject-norm island. Shape-keyed (`shapeless: false`): each
    /// distinct `[B,S,*]` traces once and is reused. The Swift `S >= 9`
    /// wide-kernel branch is part of the trace, so a new S that crosses that
    /// threshold is a new compiled graph, not a reassociation of the old one.
    static func makeInjectNormReplay(hcCount: Int, hidden: Int, eps: Float)
        -> @Sendable ([MLXArray]) -> [MLXArray]
    {
        compile(shapeless: false) {
            [hcCount, hidden, eps] inputs in
            let result = TrackFastKernels.injectNorm(
                residual: inputs[0], out: inputs[1], inject: inputs[2], scale: inputs[3],
                hcCount: hcCount, hidden: hidden, eps: eps, tile: false)
            return [result.stream, result.normed]
        }
    }

    /// Replay only the two opaque expert launches; routing and all current arrays stay live.
    static func makeMoEPairReplay(_ m: TrackMoE, hcCount: Int) -> (@Sendable ([MLXArray]) -> [MLXArray])? {
        guard let fused = m.sharedGateUp.fused, case .quant(let guq) = fused,
            case .quant(let dq) = m.sharedDown, guq.biases != nil, dq.biases != nil
        else { return nil }
        let fuseResidual = TrackResidualEpilogue.enabled
        return compile(shapeless: false) {
            [groupSize = m.expertGroupSize, bits = m.expertBits, topK = m.topK,
             fusedTiles = m.fusedTiles, rowsPerDown = m.rowsPerDown,
             guGroupSize = guq.groupSize, guBits = guq.bits, guMode = guq.mode,
             downGroupSize = dq.groupSize, downBits = dq.bits, downMode = dq.mode,
             fuseResidual, hcCount] inputs in
            let sharedGU = TrackQuantWeight(
                weight: inputs[11], scales: inputs[12], biases: inputs[13],
                groupSize: guGroupSize, bits: guBits, mode: guMode)
            let act = TrackFastMoEKernels.gateUpAct(
                wg: inputs[5], sg: inputs[6], bg: inputs[7],
                wu: inputs[8], su: inputs[9], bu: inputs[10], shared: sharedGU,
                x: inputs[0], idx: inputs[1], xrow: inputs[4], groupSize: groupSize, bits: bits,
                fusedTiles: fusedTiles)
            let sharedDown = TrackQuantWeight(
                weight: inputs[17], scales: inputs[18], biases: inputs[19],
                groupSize: downGroupSize, bits: downBits, mode: downMode)
            if fuseResidual {
                return [TrackFastMoEKernels.downCombine(
                    wd: inputs[14], sd: inputs[15], bd: inputs[16], sharedDown: sharedDown,
                    act: act, idx: inputs[1], w: inputs[2], gate: inputs[3], topK: topK,
                    groupSize: groupSize, bits: bits,
                    residual: inputs[20], inject: inputs[21], hcCount: hcCount,
                    fusedTiles: fusedTiles, rowsPerDown: rowsPerDown)]
            }
            return [TrackFastMoEKernels.downCombine(
                wd: inputs[14], sd: inputs[15], bd: inputs[16], sharedDown: sharedDown,
                act: act, idx: inputs[1], w: inputs[2], gate: inputs[3], topK: topK,
                groupSize: groupSize, bits: bits,
                fusedTiles: fusedTiles, rowsPerDown: rowsPerDown)]
        }
    }

    /// Row index of each (token, expert) slot, one constant array per window size
    /// (uploading it per step was one host copy per layer).
    nonisolated(unsafe) private static var xrowTables: [Int: MLXArray] = [:]
    private static let xrowLock = NSLock()
    static func xrowTable(S: Int, K: Int) -> MLXArray {
        xrowLock.lock(); defer { xrowLock.unlock() }
        if let t = xrowTables[S * 1024 + K] { return t }
        let t = MLXArray((0 ..< (S * K)).map { UInt32($0 / K) })
        eval(t)
        xrowTables[S * 1024 + K] = t
        return t
    }

    /// Marker-side evaluation of the S<=8 decode branch gate, so a receipt
    /// shows WHICH condition routed the first decode MoE — the narrow decode
    /// path or the wide fall-through. Pure reads; no behavior change.
    static func narrowMoEBranchGate(m: TrackMoE, x: MLXArray) -> String {
        var parts: [String] = []
        parts.append("row0=\(x.dim(0) == 1)")
        parts.append("tok=\(x.dim(1))")
        if case .quant(let guq) = m.sharedGateUp.fused ?? .dense(x) {
            parts.append("guQuant=\(guq.biases != nil),rows=\(guq.rows)")
        } else {
            parts.append("guQuant=false")
        }
        if case .quant(let dq) = m.sharedDown {
            parts.append("dqQuant=\(dq.biases != nil)")
        } else {
            parts.append("dqQuant=false")
        }
        if case .quant(let guq) = m.sharedGateUp.fused ?? .dense(x),
            case .quant(let dq) = m.sharedDown, guq.biases != nil, dq.biases != nil
        {
            let twoKernel = TrackSwiGLUEpilogue.canRunTwoKernel(
                TrackSwiGLUEpilogue.Request(
                    dtype: x.dtype, tokens: x.dim(1), hidden: x.dim(2),
                    intermediate: m.sharedHidden, bits: m.expertBits,
                    groupSize: m.expertGroupSize, sharedRows: guq.rows,
                    idxIsUInt32: true, xrowIsUInt32: true))
            parts.append("twoKernel=\(twoKernel)")
            parts.append(
                "sharedHidden=\(m.sharedHidden),bits=\(m.expertBits),gs=\(m.expertGroupSize),dtype=\(x.dtype)")
        }
        return parts.joined(separator: ",")
    }

    static func moeForwardShared(
        _ m: TrackMoE, _ x: MLXArray, inputF32: MLXArray? = nil,
        replay: (@Sendable ([MLXArray]) -> [MLXArray])?,
        residual: MLXArray? = nil, inject: MLXArray? = nil, hcCount: Int = 4
    ) -> MLXArray {
        let prof = TrackFastProfile.prefill != nil && x.dim(1) >= TrackFastProfile.minWindow
        var pt = prof ? CFAbsoluteTimeGetCurrent() : 0
        TrackBootPhase.markFirstDecodeMoe(
            "s1 moe[0] enter (S=\(x.dim(1)), H=\(x.dim(2)), K=\(m.topK),"
                + " fusedTiles=\(m.fusedTiles), guFused=\(m.fusedGateUp),"
                + " replay=\(replay != nil),"
                + " narrow=\(narrowMoEBranchGate(m: m, x: x)))",
            newCall: true)
        let logits: MLXArray
        if x.dim(0) == 1, x.dim(1) == 1, m.routerW16.dtype == .bfloat16, x.dim(2) % 128 == 0,
            m.routerW16.dim(0) % 16 == 0, x.dim(2) < 16 * m.routerW16.dim(0), m.routerW16.dim(0) < 4096
        {
            // One token: MLX's float gemv arithmetic over the bf16 weight (the
            // reference upcasts it to float32 and reads twice the bytes).
            let xf = (inputF32 ?? x.asType(.float32)).reshaped(x.dim(2))
            logits = TrackFastMoEKernels.routerGemv(x: xf, w: m.routerW16).reshaped(1, 1, -1)
        } else if inputF32 == nil, let wide = TrackPrefillRouter.apply(x: x, w: m.routerW16) {
            logits = wide
        } else {
            logits = matmul(inputF32 ?? x.asType(.float32), m.routerW32.transposed())
        }
        if prof { TrackFastProfile.tick("moe.router", &pt, [logits]) }
        TrackBootPhase.markFirstDecodeMoe("s1 moe[0] router done")
        // Top-k + softmax. Default: one launch (`TRACK_ROUTER_FUSED`,
        // order-fixed simd_shuffle_down ArgMax + softmax_single_row). Off, or
        // a shape outside the warmed decode set (B=1, S<=8, quantized shared
        // expert), falls back to the three-op chain.
        if x.dim(0) == 1, x.dim(1) <= 8,
            case .quant(let guq) = m.sharedGateUp.fused ?? .dense(x),
            case .quant(let dq) = m.sharedDown, guq.biases != nil, dq.biases != nil,
            TrackSwiGLUEpilogue.canRunTwoKernel(
                TrackSwiGLUEpilogue.Request(
                    dtype: x.dtype, tokens: x.dim(1), hidden: x.dim(2),
                    intermediate: m.sharedHidden, bits: m.expertBits,
                    groupSize: m.expertGroupSize, sharedRows: guq.rows,
                    idxIsUInt32: true, xrowIsUInt32: true))
        {
            // The shared-expert gate is a bf16 Linear on this checkpoint (router gates
            // are BF16): MLX's own GEMV keeps it; a quantized one rides in `route`.
            var gateQ: TrackQuantWeight? = nil
            if case .quant(let gq) = m.sharedGate, gq.biases != nil { gateQ = gq }
            // Decode windows: three launches over MLX's own GEMV arithmetic for the
            // window size (per-row `qmv_fast` / `qmv` for the gathered experts, `qmv`
            // or `qmv_wide` for the shared expert): top-k + softmax + shared gate,
            // gate|up + SwiGLU for the routed and the shared expert, down + combine.
            let S = x.dim(1), K = m.topK, H = x.dim(2)
            let x2 = x.reshaped(S, H)
            let logits2 = logits.reshaped(S, -1)
            let idx: MLXArray
            let weights: MLXArray
            let gate: MLXArray
            if TrackFastMoEKernels.fusedRouterEnabled {
                let r = TrackFastMoEKernels.route(
                    logits: logits2, x: x2, sharedGate: gateQ, topK: K)
                idx = r.idx
                weights = r.w
                gate = gateQ != nil ? r.gate : m.sharedGate.apply(x).reshaped(S)
            } else {
                let r = TrackFastMoEKernels.routeChain(logits: logits2, topK: K)
                idx = r.idx
                weights = r.w
                gate = m.sharedGate.apply(x).reshaped(S)
            }
            if prof { TrackFastProfile.tick("moe.route", &pt, [idx, weights, gate]) }
            TrackBootPhase.markFirstDecodeMoe(
                "s1 moe[0] topk done (fusedRouter=\(TrackFastMoEKernels.fusedRouterEnabled))")
            let flatIdx = idx.reshaped(S * K)
            let xrow = Self.xrowTable(S: S, K: K)
            let fuse = residual != nil && inject != nil
                && TrackResidualEpilogue.coversDownProj(k: m.sharedHidden, n: H, tokens: S)
            let res2 = residual?.reshaped(S, hcCount * H)
            let inj2 = inject?.reshaped(S, hcCount)
            if let replay, StreamOrDevice.default.stream === Stream.gpu,
                fuse == TrackResidualEpilogue.enabled
            {
                var args: [MLXArray] = [
                    x2, flatIdx, weights.reshaped(S * K), gate, xrow,
                    m.expertGate.w, m.expertGate.s, m.expertGate.b,
                    m.expertUp.w, m.expertUp.s, m.expertUp.b,
                    guq.weight, guq.scales, guq.biases!,
                    m.expertDown.w, m.expertDown.s, m.expertDown.b,
                    dq.weight, dq.scales, dq.biases!,
                ]
                if fuse, let res2, let inj2 { args.append(contentsOf: [res2, inj2]) }
                TrackBootPhase.markFirstDecodeMoe(
                    "s1 moe[0] replay branch (fuse=\(fuse), args=\(args.count))")
                let y = replay(args)[0]
                TrackBootPhase.markFirstDecodeMoe("s1 moe[0] replay done")
                return fuse ? y.reshaped(1, S, hcCount * H) : y.reshaped(1, S, H)
            }
            let act = TrackFastMoEKernels.gateUpAct(
                wg: m.expertGate.w, sg: m.expertGate.s, bg: m.expertGate.b,
                wu: m.expertUp.w, su: m.expertUp.s, bu: m.expertUp.b, shared: guq,
                x: x2, idx: flatIdx, xrow: xrow, groupSize: m.expertGroupSize, bits: m.expertBits,
                fusedGateUp: m.fusedGateUp, fusedTiles: m.fusedTiles)
            TrackBootPhase.markFirstDecodeMoe("s1 moe[0] direct gu done")
            let down = TrackFastMoEKernels.downCombine(
                wd: m.expertDown.w, sd: m.expertDown.s, bd: m.expertDown.b, sharedDown: dq,
                act: act, idx: flatIdx, w: weights.reshaped(S * K), gate: gate, topK: K,
                groupSize: m.expertGroupSize, bits: m.expertBits,
                residual: res2, inject: inj2, hcCount: hcCount,
                fusedTiles: m.fusedTiles, rowsPerDown: m.rowsPerDown)
            TrackBootPhase.markFirstDecodeMoe("s1 moe[0] direct down done")
            return fuse ? down.reshaped(1, S, hcCount * H) : down.reshaped(1, S, H)
        }
        let idx: MLXArray, weights: MLXArray
        if TrackFastMoEKernels.fusedRouterEnabled, TrackP12Prefill.eligible(x),
            x.dim(2) == 2560, logits.dtype == .float32, logits.dim(-1) == 512,
            m.topK == 10, StreamOrDevice.default.stream === Stream.gpu
        {
            let routed = TrackFastMoEKernels.route(
                logits: logits, x: x, sharedGate: nil, topK: m.topK)
            idx = routed.idx
            weights = routed.w
        } else {
            let chained = TrackFastMoEKernels.routeChain(logits: logits, topK: m.topK)
            idx = chained.idx
            weights = chained.w
        }
        if prof { TrackFastProfile.tick("moe.route", &pt, [idx, weights]) }
        let dispatch: TrackMoEDispatch.Pack?
        if TrackP12Prefill.sortedCombine, TrackP12Prefill.eligible(x),
            m.p12SortedParts != nil, !m.switchMLP.hasFusedGateUp
        {
            dispatch = TrackMoEDispatch.pack(indices: idx, experts: m.routerW32.dim(0))
        } else {
            dispatch = nil
        }
        let sharedAct: MLXArray
        if x.dim(1) > 8, let fusedGU = m.sharedGateUp.fused {
            // MLXFAST-SHAREDFUSE: wide windows run gate|up as ONE N = 1280 GEMM.
            // Both N = 640 and N = 1280 take the plain NAX qmm (no split-K:
            // 32 x 10 = 320 column x row tiles already exceed the split-K
            // threshold), whose column tiles are independent, so every output
            // element is the one the two separate GEMMs produce; the SwiGLU
            // reads the gate and up halves of the concatenation as before.
            sharedAct = TrackFastKernels.swiglu(gu: fusedGU.apply(x))
        } else if TrackP12Prefill.splitShared, TrackP12Prefill.eligible(x),
            m.sharedGateUp.parts.count == 2
        {
            // The same two GEMMs and the same `mlx_silu(gate) * up`, over the
            // two outputs directly instead of over their concatenation.
            let rows = x.dim(0) * x.dim(1)
            let sgate = m.sharedGateUp.parts[0].apply(x)
            let sup = m.sharedGateUp.parts[1].apply(x)
            sharedAct = TrackFastKernels.swiglu2(
                gate: sgate.reshaped(rows, m.sharedHidden),
                up: sup.reshaped(rows, m.sharedHidden)
            ).reshaped(x.dim(0), x.dim(1), m.sharedHidden)
        } else {
            sharedAct = TrackFastKernels.swiglu(gu: m.sharedGateUp.apply(x))
        }
        let shared = m.sharedDown.apply(sharedAct)
        let gate = m.sharedGate.apply(x)  // [B,S,1]
        if prof { TrackFastProfile.tick("moe.shared", &pt, [shared, gate]) }
        // Prefill-only grouped GEMM: sort (token, expert) pairs, gather once
        // per expert group, run the same gather_qmm as the sorted path, scatter
        // via the inverse permutation. Decode (S<=8) returned above.
        if let grouped = TrackGroupedGEMM.apply(
            m, x, indices: idx, weights: weights, shared: shared, gate: gate)
        {
            if prof { TrackFastProfile.tick("moe.grouped+combine", &pt, [grouped]) }
            return grouped
        }
        // Read the routed rows through the permutation the sort already
        // produced instead of materialising a scattered copy of them.
        if let combined = TrackP12Prefill.sortedMoE(
            m, x, indices: idx, weights: weights, shared: shared, gate: gate, dispatch: dispatch)
        {
            if prof { TrackFastProfile.tick("moe.sorted+combine", &pt, [combined]) }
            return combined
        }
        let routed = m.switchMLP(x, idx)  // [B,S,K,H]
        if prof { TrackFastProfile.tick("moe.switchMLP", &pt, [routed]) }
        // The combine kernel folds the K products in the association MLX's small
        // column reduce uses (verified at float precision for every window size).
        return TrackFastKernels.moeCombine(routed: routed, w: weights, shared: shared, gate: gate)
    }

    /// Token ids for the PLE n-gram hash. Serial/chained windows read these
    /// at `fastStreams` entry (host memcpy). MTP verify delays the read until
    /// layer-0 GDN is submitted, and may already hold the 16 gathered rows.
    private struct PLEHostHistory {
        let history: [Int64]
        let contextLength: Int
        let gatheredRows: MLXArray?
    }

    private func hostInt64s(_ a: MLXArray) -> [Int64] {
        switch a.dtype {
        case .int32: return a.asArray(Int32.self).map(Int64.init)
        case .int64: return a.asArray(Int64.self)
        default: return a.asType(.int64).asArray(Int64.self)
        }
    }

    /// Host copy of the PLE context and the fed ids. Decode windows and the
    /// w05 fallback read this before the layer loop. MTP verify reads it
    /// after layer-0 is submitted (`eagerGather` hashes and gathers here so
    /// the 16 rows enter the walk as a host embedding).
    private func prefetchPLEHostHistory(
        ids: MLXArray, evaluation: CBv2RecurrentStateEvaluation, eagerGather: Bool
    ) -> PLEHostHistory? {
        let S = ids.dim(1)
        guard S <= 8, ids.dim(0) == 1,
            let p = layers.first(where: { $0.ple != nil })?.ple,
            let host = p.embedding.rowSourceHolder.source as? Qwen4ExpNGramHostRowSource
        else { return nil }
        let contextLength = max(1, p.dilation - 1)
        let state = evaluation.inputState(modelLayerIndex: p.stateLayerIndex)
        let ctx =
            state?.ssm.map(hostInt64s)
            ?? Array(repeating: Int64(cfg.eosTokenId), count: contextLength)
        let history = ctx + hostInt64s(ids)
        var gathered: MLXArray? = nil
        if eagerGather {
            let gid = p.embedding.hostRowIds(history: [history], newCount: S)
            gathered = host.rows(
                globalIds: gid, shape: [1, S, (cfg.ngramSize - 1) * cfg.headsPerNGram])
        }
        return PLEHostHistory(
            history: history, contextLength: contextLength, gatheredRows: gathered)
    }

    private func pleForward(
        _ p: TrackPLE, stream: MLXArray, ids: MLXArray,
        evaluation: CBv2RecurrentStateEvaluation, capture: Bool,
        hostPrefetch: PLEHostHistory?
    ) -> MLXArray {
        let B = stream.dim(0), S = stream.dim(1)
        precondition(B == 1)
        let wide = hcCount * hidden
        let contextLength = hostPrefetch?.contextLength ?? max(1, p.dilation - 1)
        let state = evaluation.inputState(modelLayerIndex: p.stateLayerIndex)
        // Device-side context (prefill windows and the device row source).
        func devicePrevious() -> MLXArray {
            (state?.ssm
                ?? MLXArray.full(
                    [1, contextLength], values: MLXArray(Int32(cfg.eosTokenId)), dtype: .int32))
                .asType(ids.dtype)
        }
        let convState =
            state?.conv ?? MLXArray.zeros([1, p.stateLength, wide], dtype: stream.dtype)

        let embedded: MLXArray
        // Host history is the same integers `hostRowIds` hashed. Verify
        // rounds gather those 16 rows before this call (layer-0 already
        // submitted); serial gathers here after the entry asArray.
        var hostHistory: [Int64]? = nil
        if let prefetch = hostPrefetch,
            let host = p.embedding.rowSourceHolder.source as? Qwen4ExpNGramHostRowSource
        {
            let rows =
                prefetch.gatheredRows
                ?? host.rows(
                    globalIds: p.embedding.hostRowIds(history: [prefetch.history], newCount: S),
                    shape: [B, S, (cfg.ngramSize - 1) * cfg.headsPerNGram])
            embedded = rows.reshaped(B, S, -1).asType(stream.dtype)
            hostHistory = prefetch.history
        } else if TrackPLEChain.isEnabled,
            let source = p.embedding.rowSourceHolder.source
        {
            let hash = TrackPLEChain.ngramHash(cfg: cfg, pleLayerIndex: p.pleLayerIndex)
            if let gid = TrackPLEChain.fusedRowIds(
                ids: ids, previousContext: devicePrevious(), hash: hash)
            {
                embedded = source.rows(globalIds: gid).reshaped(B, S, -1).asType(stream.dtype)
            } else {
                embedded = p.embedding(ids, previousContext: devicePrevious()).asType(stream.dtype)
            }
        } else {
            embedded = p.embedding(ids, previousContext: devicePrevious()).asType(stream.dtype)
        }
        // TRACK_PLE_CHAIN_FUSE: two unchanged GEMVs + prepare + convolution.
        // Kill switch and guard misses take the three-launch PLE block below.
        // PLEFUSE2 is §5.4-priced and is not selected.
        let full: MLXArray
        let output: MLXArray
        let keyFlat = p.keyProj.apply(embedded)
        let value = p.valueProj.apply(embedded)
        if TrackPLEChain.shouldFuse(
            S: S, hidden: hidden, hcCount: hcCount, dilation: p.dilation,
            stateLength: p.stateLength, dtype: stream.dtype, keyRows: p.keyProj.rows,
            valueRows: p.valueProj.rows, convCols: p.convW2.dim(1)),
            keyFlat.shape.suffix(1) == [wide], value.shape.suffix(1) == [hidden],
            convState.dtype == stream.dtype
        {
            (full, output) = TrackPLEChain.afterProjections(
                keyFlat: keyFlat, stream: stream, value: value,
                kScale: p.normKeyScale, qScale: p.normQueryScale, cScale: p.normConvScale,
                convState: convState, convW: p.convW2, dilation: p.dilation,
                hcCount: hcCount, hidden: hidden, eps: eps)
        } else {
            (full, output) = TrackPLEChain.unfusedAfterProjections(
                keyFlat: keyFlat, stream: stream, value: value,
                kScale: p.normKeyScale, qScale: p.normQueryScale, cScale: p.normConvScale,
                convState: convState, convW: p.convW2, dilation: p.dilation,
                hcCount: hcCount, hidden: hidden, eps: eps)
        }
        do {
            if capture {
                let n = p.stateLength
                let convStack = asStrided(full, [S, n, wide], strides: [wide, wide, 1], offset: wide)
                let contextStack: MLXArray
                if let h = hostHistory {
                    // Row s = the context after consuming window token s.
                    contextStack = pleSSMStaging.writeCapture(history: h, newCount: S)
                } else {
                    let history = concatenated([devicePrevious(), ids], axis: 1)
                    contextStack = asStrided(
                        history.asType(.int32), [S, contextLength], strides: [1, 1], offset: 1)
                }
                try stageRecurrentCapture(
                    evaluation, modelLayerIndex: p.stateLayerIndex, conv: convStack,
                    ssm: contextStack, positions: S)
            } else {
                let ssm: MLXArray
                // Ranked free-run carries hostHistory via PLE prefetch.
                // Skipping writeDecode on S=1 used the device id slice and
                // mismatched free-run tokens at step 2 (Yukon 5fadbfe).
                // writeDecode is MLX subscript-set (crash-free vs the
                // dangling noCopy pointer). No hostHistory: device slice.
                if let h = hostHistory {
                    ssm = pleSSMStaging.writeDecode(h.suffix(contextLength))
                } else {
                    let history = concatenated([devicePrevious(), ids], axis: 1)
                    ssm = history[0..., (-contextLength)...].asType(.int32)
                }
                try evaluation.stage(
                    modelLayerIndex: p.stateLayerIndex,
                    conv: full[0..., (-p.stateLength)..., 0...],
                    ssm: ssm)
            }
        } catch {
            preconditionFailure("TrackFastModel: PLE stage failed: \(error)")
        }
        return output
    }

    private func injectNorm(
        residual: MLXArray, out: MLXArray?, inject: MLXArray?, scale: MLXArray, tile: Bool
    ) -> (stream: MLXArray, normed: MLXArray) {
        // Native replay does not key Swift task-local streams; trace and replay
        // only on the canonical GPU stream. Other contexts retain the raw path.
        if !tile, let out, let inject, StreamOrDevice.default.stream === Stream.gpu {
            let result = injectNormReplay([residual, out, inject, scale])
            return (result[0], result[1])
        }
        return TrackFastKernels.injectNorm(
            residual: residual, out: out, inject: inject, scale: scale,
            hcCount: hcCount, hidden: hidden, eps: eps, tile: tile)
    }

    /// RMS of a stream that already has the block output folded in (W-wide),
    /// otherwise the existing inject + residual add + RMS.
    private func consumeBlockOutput(
        residual: MLXArray, out: MLXArray?, inject: MLXArray?, scale: MLXArray, tile: Bool
    ) -> (stream: MLXArray, normed: MLXArray) {
        if let out, out.dim(-1) == hcCount * hidden {
            return injectNorm(
                residual: out, out: nil, inject: nil, scale: scale, tile: false)
        }
        return injectNorm(
            residual: residual, out: out, inject: inject, scale: scale, tile: tile)
    }

    /// RMSNorm + mixer. Decode inject-path GEMV (N=4, K=10240) fuses the
    /// grouped RMSNorm prologue into the 4-row qmv when the shape matches.
    /// A W-wide residual-epilogue output skips the inject add.
    private func mixAfterInject(
        hc: TrackHC, residual: MLXArray, out: MLXArray?, inject: MLXArray?,
        scale: MLXArray, tile: Bool, tag: String, emitF32: Bool = false
    ) -> (stream: MLXArray, input: MLXArray, injectW: MLXArray, inputF32: MLXArray?) {
        if let out, out.dim(-1) == hcCount * hidden {
            let (stream, normed) = consumeBlockOutput(
                residual: residual, out: out, inject: inject, scale: scale, tile: tile)
            let m = hcMix(hc, normed: normed, tag: tag, emitF32: emitF32)
            return (stream, m.input, m.inject, m.inputF32)
        }
        if case .quant(let iq)? = hc.inject, iq.biases != nil,
            let fused = TrackQuantPrologue.apply(
                residual: residual, out: out, injectGate: inject, scale: scale,
                weight: iq, hcCount: hcCount, hidden: hidden, eps: eps, tile: tile)
        {
            let m = hcMix(
                hc, normed: fused.normed, tag: tag, emitF32: emitF32, precomputedInj: fused.inj)
            return (fused.stream, m.input, m.inject, m.inputF32)
        }
        let (stream, normed) = injectNorm(
            residual: residual, out: out, inject: inject, scale: scale, tile: tile)
        let m = hcMix(hc, normed: normed, tag: tag, emitF32: emitF32)
        return (stream, m.input, m.inject, m.inputF32)
    }

    private func gdnStore() -> TrackGDNStateStore {
        if let gdnStateStore { return gdnStateStore }
        let store = TrackGDNStateStore()
        gdnStateStore = store
        return store
    }

    /// Captured verify: copy into preallocated snapshot slots and restore
    /// at the accepted boundary (view-swap + flag, or w26 copy when
    /// `TRACK_ROLLBACK_FLAG=0`). Width 1, or `TRACK_GDN_STATE_RESTORE=0`,
    /// keep the vendor `stageCaptured` views.
    private func stageRecurrentCapture(
        _ evaluation: CBv2RecurrentStateEvaluation, modelLayerIndex: Int,
        conv: MLXArray, ssm: MLXArray, positions: Int
    ) throws {
        if TrackGDNStateRestore.usesPrefixReplay(positions: positions) {
            let store = gdnStore()
            store.capture(modelLayerIndex: modelLayerIndex, conv: conv, ssm: ssm)
            try store.stagePrefixReplay(
                evaluation, modelLayerIndex: modelLayerIndex, positions: positions)
        } else {
            try evaluation.stageCaptured(
                modelLayerIndex: modelLayerIndex, conv: conv, ssm: ssm, positions: positions)
        }
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
        TrackBootPhase.markFirstDecodeStage("s1 embed+rope done")

        let profiling = TrackFastProfile.prefill != nil && ids.dim(1) >= TrackFastProfile.minWindow
        var profT = profiling ? CFAbsoluteTimeGetCurrent() : 0
        if profiling { TrackFastProfile.windows += 1 }
        // Serial/chained: host copy at entry (w05). MTP verify: async-copy
        // the lazy draft ids, submit layer-0 GDN, then hash+gather so the
        // 16 PLE rows enter the walk as a host embedding.
        let pleLayer = layers.first(where: { $0.ple != nil })
        let delayPLE = TrackPLEVerifyPrefetch.shouldDelay(
            window: ids.dim(1),
            capture: capture,
            seamOK: TrackPLEVerifyPrefetch.seamOK(
                layer0IsGDN: layers.first?.gdn != nil,
                layer0HasPLE: layers.first?.ple != nil,
                pleLayerIndex: pleLayer?.index)
                && (pleLayer?.ple?.embedding.rowSourceHolder.source is Qwen4ExpNGramHostRowSource),
            enabled: TrackPLEVerifyPrefetch.enabled)
        var pleHost: PLEHostHistory? = nil

        func runLayer(_ layer: TrackLayer) {
            var input: MLXArray
            var injectW: MLXArray
            if let ple = layer.ple {
                // Materialize the stream, add the PLE block, then norm.
                (stream, _) = consumeBlockOutput(
                    residual: residual, out: pendingOut, inject: pendingInject,
                    scale: layer.attnHC.normScaleQ,
                    tile: tile)
                stream =
                    stream
                    + pleForward(
                        ple, stream: stream, ids: ids, evaluation: evaluation, capture: capture,
                        hostPrefetch: pleHost)
                let (st, normed) = injectNorm(
                    residual: stream, out: nil, inject: nil,
                    scale: layer.attnHC.normScaleQ,
                    tile: false)
                stream = st
                if profiling { TrackFastProfile.tick("norm+ple", &profT, [stream, normed]) }
                let am = hcMix(layer.attnHC, normed: normed, tag: "L\(layer.index).attn.hc")
                input = am.input
                injectW = am.inject
            } else {
                let am = mixAfterInject(
                    hc: layer.attnHC, residual: residual, out: pendingOut, inject: pendingInject,
                    scale: layer.attnHC.normScaleQ, tile: tile, tag: "L\(layer.index).attn.hc")
                stream = am.stream
                input = am.input
                injectW = am.injectW
                if profiling { TrackFastProfile.tick("norm", &profT, [stream]) }
            }
            tile = false
            if layer.index == 0 {
                TrackBootPhase.markFirstDecodeStage("s1 L0 norm+mix done")
            }
            if profiling { TrackFastProfile.tick("mixer", &profT, [input, injectW]) }
            Self.debugTaps?.append(("L\(layer.index).attn.stream_in", stream))
            Self.debugTaps?.append(("L\(layer.index).attn.input", input))
            let attended: MLXArray
            if let gdn = layer.gdn {
                attended = gdnForward(
                    gdn, input, layerIndex: layer.index, evaluation: evaluation, capture: capture,
                    residual: stream, inject: injectW)
            } else {
                let cache = caches[attentionIndex]
                attentionIndex += 1
                attended = attnForward(
                    layer.attn!, input, cache: cache, rope: ropeTab,
                    residual: stream, inject: injectW)
            }
            Self.debugTaps?.append(("L\(layer.index).attn.out", attended))
            if profiling { TrackFastProfile.tick(layer.gdn != nil ? "gdn" : "attn", &profT, [attended]) }
            if layer.index == 0 {
                TrackBootPhase.markFirstDecodeStage(
                    layer.gdn != nil ? "s1 L0 gdn done" : "s1 L0 attn done")
            }
            let mm = mixAfterInject(
                hc: layer.mlpHC, residual: stream, out: attended, inject: injectW,
                scale: layer.mlpHC.normScaleQ, tile: false, tag: "L\(layer.index).mlp.hc",
                emitF32: true)
            stream = mm.stream
            if profiling { TrackFastProfile.tick("norm", &profT, [stream]) }
            if layer.index == 0 {
                TrackBootPhase.markFirstDecodeStage("s1 L0 mlpnorm done")
            }
            Self.debugTaps?.append(("L\(layer.index).mlp.stream_in", stream))
            input = mm.input
            injectW = mm.injectW
            if profiling { TrackFastProfile.tick("mixer", &profT, [input, injectW]) }
            Self.debugTaps?.append(("L\(layer.index).mlp.input", input))
            pendingOut = Self.moeForwardShared(
                layer.moe, input, inputF32: mm.inputF32, replay: layer.moePairReplay,
                residual: stream, inject: injectW, hcCount: hcCount)
            if profiling { TrackFastProfile.tick("moe", &profT, [pendingOut!]) }
            if layer.index == 0 {
                TrackBootPhase.markFirstDecodeStage("s1 L0 moe done")
            }
            switch layer.index {
            case 1, 11, 23, 35, 47:
                TrackBootPhase.markFirstDecodeStage("s1 L\(layer.index) done")
            default: break
            }
            Self.debugTaps?.append(("L\(layer.index).mlp.out", pendingOut!))
            pendingInject = injectW
            residual = stream
            // Dispatch the graph so far: the GPU starts on these layers while the
            // CPU keeps building the rest (the build is otherwise GPU-idle time).
            if Self.asyncChunk > 0 {
                let n = layer.index + 1
                let first = Self.asyncFirst > 0 ? Self.asyncFirst : Self.asyncChunk
                let second = Self.asyncSecond > first ? Self.asyncSecond : first
                if n == first || n == second || (n > second && (n - second) % Self.asyncChunk == 0) { asyncEval(stream) }
            }
        }

        if delayPLE, !layers.isEmpty {
            TrackPLEVerifyPrefetch.run(
                copyIds: { asyncEval(ids) },
                buildAndSubmitLayer0: {
                    runLayer(layers[0])
                    asyncEval(stream)
                },
                readIdsAndGather: {
                    pleHost = prefetchPLEHostHistory(
                        ids: ids, evaluation: evaluation, eagerGather: true)
                    TrackBootPhase.markFirstDecodeStage("s1 ple host done (delayed)")
                })
            for layer in layers.dropFirst() { runLayer(layer) }
        } else {
            pleHost = prefetchPLEHostHistory(
                ids: ids, evaluation: evaluation, eagerGather: false)
            TrackBootPhase.markFirstDecodeStage("s1 ple host done")
            for layer in layers { runLayer(layer) }
        }
        let (multi, finalNormed) = consumeBlockOutput(
            residual: residual, out: pendingOut, inject: pendingInject,
            scale: finalMixer.normScaleQ,
            tile: false)
        let mixed = hcMix(finalMixer, normed: finalNormed).input
        // Collapse the slice-update chain: these buffers are not in vendor
        // innerState, so the engine loop would not eval them.
        TrackBootPhase.markFirstDecodeStage("s1 final mix done")
        let tapes = TrackIndexerTape.liveBuffers()
        if !tapes.isEmpty { asyncEval(tapes) }
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
        else {
            if let typed = typedCaches(caches) { TrackIndexerTape.syncCBv2(typed) }
            return nil
        }
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
        TrackBootPhase.markForwardBegin(width: tokens.dim(1))
        if let s = streamsOrDelegate(
            tokens, inputEmbeddings: nil, caches: caches, recurrentState: recurrentState,
            positionIds: positionIds, capture: false)
        {
            if tokens.dim(1) > 1 {
                TrackBootPhase.mark("cbv2 forward fast path done (S=\(tokens.dim(1)))")
            }
            TrackBootPhase.markFirstDecodeStage("s1 streams done")
            TrackBootPhase.markFirstDecodeStage("s1 head begin")
            let logits = base.head(s.mixed)
            TrackBootPhase.markFirstDecodeStage("s1 head done")
            return logits
        }
        if tokens.dim(1) > 1 {
            TrackBootPhase.mark("cbv2 forward fell back (S=\(tokens.dim(1)))")
        }
        TrackBootPhase.markFirstDecodeStage("s1 delegate path")
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
        if TrackFastProfile.prefill != nil {
            let plan = fastPlan(tokens: inputs, caches: cache ?? [], recurrentState: recurrentState, positionIds: positionIds)
            print("[profile] cbv2RecurrentPrefill S=\(inputs.dim(1)) posIds=\(positionIds != nil) caches=\(cache?.count ?? -1) fast=\(plan != nil) req=\(requirement)")
        }
        TrackBootPhase.mark("prefill forward begin (S=\(inputs.dim(1)))")
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
        TrackBootPhase.mark("prefill forward fell back to the module path")
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
            // Verify graph is built. asyncChunk already submitted earlier
            // layers, so the GPU is in flight. stagePrefixReplay pre-built
            // the accept-all last-slot copies (host only, not enqueued).
            return (base.head(s.mixed), s.multi)
        }
        return base.cbv2ForwardWithHiddenCaptured(
            tokens, caches: caches, recurrentState: recurrentState, positionIds: positionIds)
    }
}
