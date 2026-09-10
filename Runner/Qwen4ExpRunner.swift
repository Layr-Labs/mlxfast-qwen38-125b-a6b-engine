// Copied from the pinned fork at commit 449f2d01, Libraries/MLXRunners/Qwen4ExpRunner.swift: this is the track's EDITABLE Runner.
// Copyright © 2026 Eigen Labs.
//
// MLXRunners — Qwen 3.8 Flash-Next 125B-A6B (`qwen4_exp`, `qwen4_exp_text`).
//
// A HYBRID trunk like Qwen 3.5: of the 48 layers only the 12 full-attention
// ones own a key-value tape, and `cbv2LayerKinds` is that compact storage
// layout with `modelLayerIndex` mapping each row back to its decoder layer.
// The other 36 layers are gated-deltanet recurrence carried as request-owned
// recurrent state.
//
// Two things separate this family from Qwen 3.5:
//
//   * QSA. Every full-attention layer runs an indexer with a 2048-token
//     budget and emits a keep mask. The mask is not an optimization: without
//     it the model answers differently. The manifest therefore declares
//     `requiresKeepMask`, the family owns its own layer cache (a second
//     per-row indexer tape beside the key-value tape), and the engine refuses
//     to start over a cache provider that cannot apply the mask.
//   * The n-gram PLE table. It is 29.8 GiB and it is never held as model
//     parameters. The caller passes it in through
//     `RunnerLoadOptions.resources` under ``ngramRowSourceResource``, either
//     as the PATH of the n-gram shard directory or as an already built
//     `Qwen4ExpNGramRowSource`; without it a checkpoint that has PLE layers
//     is refused, because the model's forward pass cannot run.
//
// Speculation is the checkpoint's own `mtp.*` head
// (`Qwen4ExpInlineMTPAssistant`), request-stateful across rounds, depth 1...6.
// Paged storage, prefix reuse, compiled decode and packed prefill stay off,
// and only single-stream regimes are declared: the indexer scores one tape
// per call.

import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
// ADDED to the fork's copy: `adoptMTPHead` reads and applies quantization
// geometry, which is MLXNN's `Quantized` and `quantize`.
import MLXNN
// ADDED to the fork's copy: this file used to live INSIDE MLXRunners, so
// the runner boundary -- `Runner`, `RunnerManifest`, `RunnerCheckpoint`,
// `RunnerError`, `RunnerEngineAssembly`, `CBv2SingleRowStepper` -- needed no
// import. Here it is a separate module.
import MLXRunners
import Tokenizers

public final class TrackQwen4ExpRunner: Runner, @unchecked Sendable {

    /// Name under which the caller passes the n-gram row source in
    /// `RunnerLoadOptions.resources`.
    ///
    /// Two value shapes are accepted, and only two:
    ///
    ///   * a path, as a `URL` or a `String`, of the n-gram shard DIRECTORY
    ///     that the offline transform writes. bench-worker passes its
    ///     `--resource` value in this shape. A path to a single file is
    ///     refused by name.
    ///   * an already built `Qwen4ExpNGramRowSource`, for an in-process
    ///     caller that holds one.
    public static let ngramRowSourceResource = "qwen4exp.ngramRowSource"

    public static let manifest = RunnerManifest(
        runnerID: "layr/qwen4exp-125b-a6b",
        modelTypes: ["qwen4_exp", "qwen4_exp_text"],
        engine: CBv2ModelCapabilities(
            supportsPrefixReuse: false,
            supportsPagedKV: false,
            supportsCompiledDecode: false,
            supportsPackedPrefill: false,
            supportsMTP: true,
            supportsCompactRecurrentMTPReplay: false),
        kvBackends: [.contiguous],
        decoders: [
            DecoderDeclaration(
                mode: DecoderID.serial.rawValue, drafter: .none, state: .stateless,
                depth: nil),
            DecoderDeclaration(
                mode: DecoderID.mtp.rawValue, drafter: .embeddedHead,
                state: .requestStateful,
                depth: 1 ... Qwen4ExpInlineMTPAssistant.maximumDepth),
        ],
        regimes: [
            RegimeDeclaration(batch: .single, timing: .freeRun, perStreamTiming: false),
            RegimeDeclaration(batch: .single, timing: .teacherForced, perStreamTiming: false),
        ],
        multimodal: false,
        recurrentLayers: true,
        requiresKeepMask: true)

    public let servingModel: any LanguageModel
    public let tokenizer: any MLXLMCommon.Tokenizer
    public let eosTokenIDs: Set<Int>
    public let layerKinds: [CBv2LayerKind]
    public let loadedDecoders: [DecoderID]
    public let headProvenance: HeadProvenance?
    public let loadedModelType: String

    private let model: Qwen4ExpModel
    private let drafter: (any CBv2MTPDrafter)?
    private let kvBytesCapacity: Int
    private let maxSequenceLength: Int

    private init(
        model: Qwen4ExpModel,
        serving: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        eosTokenIDs: Set<Int>,
        loadedModelType: String,
        drafter: (any CBv2MTPDrafter)?,
        headProvenance: HeadProvenance?,
        kvBytesCapacity: Int,
        maxSequenceLength: Int
    ) {
        self.model = model
        self.servingModel = serving
        self.layerKinds = model.cbv2LayerKinds
        self.tokenizer = tokenizer
        self.eosTokenIDs = eosTokenIDs
        self.loadedModelType = loadedModelType
        self.drafter = drafter
        self.headProvenance = headProvenance
        self.kvBytesCapacity = kvBytesCapacity
        self.maxSequenceLength = maxSequenceLength
        self.loadedDecoders = drafter == nil ? [.serial] : [.serial, .mtp]
    }

    /// Adopt a module the caller already holds. Reads NO tensors: the
    /// embedded head is BOUND to the module in memory, never reloaded, and
    /// the n-gram rows come from the caller's resource.
    ///
    /// The one file read beyond `config.json` is the embedded head's own
    /// shard, hashed for provenance (§12c). That is a checkpoint fact, not a
    /// second copy of the weights.
    public static func adopt(
        model: any LanguageModel,
        tokenizer: any MLXLMCommon.Tokenizer,
        configuration: ModelConfiguration,
        directory: URL,
        options: RunnerLoadOptions
    ) throws -> TrackQwen4ExpRunner {
        // Checkpoint facts FIRST, module second.
        let modelType = try RunnerCheckpoint.modelType(at: directory)
        let eosTokenIDs = RunnerCheckpoint.eosTokenIDs(
            at: directory, tokenizer: tokenizer)

        // The head is a block of the checkpoint, not a separate artifact, so
        // a caller handing one over — a resident module or another directory
        // — is asking for something this family does not serve. Refused
        // BEFORE the module is examined: accepting a drafter and then
        // binding the checkpoint's own would serve something other than what
        // the caller handed in.
        guard options.preloadedDrafter == nil else {
            throw RunnerError.drafterUnavailable(
                "qwen4_exp draws its mtp head from the checkpoint; "
                    + "a preloaded drafter is not served")
        }
        if let drafterDirectory = options.drafterDirectory,
            drafterDirectory.standardizedFileURL != directory.standardizedFileURL
        {
            throw RunnerError.drafterUnavailable(
                "the mtp head is embedded in the checkpoint; "
                    + "\(drafterDirectory.path) is not served")
        }

        guard let model = model as? Qwen4ExpModel else {
            throw RunnerError.unexpectedModel(String(describing: type(of: model)))
        }

        // THE NORM CONVENTION, against the tensors that were actually
        // loaded. `config.json` does not carry it, so the configuration
        // decides it from this family's pinned default and the module fixes
        // it at init — by the time adoption holds a module, the only thing
        // left to do is prove the choice was right. Wrong, it scales every
        // non-gated norm in the tower by about one unit and produces
        // incoherent output while every shape, tensor count and digest still
        // checks out; that is what the box saw as a step-0 parity failure.
        //
        // Reads only resident tensors. See `Qwen4ExpNormConvention`.
        try Self.validateNormConvention(model)

        // The PLE layers read their rows through the injected source. A model
        // that has PLE layers and no source cannot run a forward pass at all,
        // so adoption refuses here rather than at the first token.
        if !model.pleEmbeddings.isEmpty {
            model.install(
                ngramRowSource: try Self.resolveNGramRowSource(
                    options.resources[ngramRowSourceResource], for: model))
        }

        // ADOPT THE HEAD INTO THIS REPOSITORY. The loader builds the fork's
        // head from the checkpoint's `mtp.*` block; adoption rebuilds it as
        // `TrackQwen4ExpMTPModule`, which is editable here, and hands the
        // loaded tensors over. A checkpoint without the `mtp.*` block has no
        // head and is serial only.
        let head = try model.mtp.map {
            try Self.adoptMTPHead($0, configuration: model.configuration)
        }
        // The fork's `mtp` module stays bound to the model: a Module property
        // may only change through `update(modules:)`, and there is nothing to
        // gain from removing it — the track head holds the SAME arrays, not a
        // copy. The drafter below is the only reader of a head; it serves the
        // track head.
        let drafter = head.map { TrackQwen4ExpInlineMTPAssistant(target: model, mtp: $0) }
        // §12c: the one embedded-head rule, in the one shared helper. It
        // hashes the shards carrying the `mtp.*` tensors, not the whole
        // checkpoint.
        let provenance =
            drafter == nil
            ? nil : try RunnerCheckpoint.provenance(ofEmbeddedHeadAt: directory)

        // THE FAST FORWARD. The engine drives `TrackQwen4ExpFastModel`, which
        // serves the SAME loaded tensors through a leaner graph (see
        // FastModel/TrackFastModel.swift). TRACK_FAST_FORWARD=0 serves the
        // fork's module directly, for A/B.
        let serving: any LanguageModel =
            TrackQwen4ExpFastModel.enabled ? TrackQwen4ExpFastModel(base: model) : model

        return TrackQwen4ExpRunner(
            model: model,
            serving: serving,
            tokenizer: tokenizer,
            eosTokenIDs: eosTokenIDs,
            loadedModelType: modelType,
            drafter: drafter,
            headProvenance: provenance,
            kvBytesCapacity: options.kvBytesCapacity,
            maxSequenceLength: options.maxSequenceLength)
    }

    /// The family half of the `--verbose` load summary.
    ///
    /// The RMSNorm offset and its SOURCE lead, because that is the line that
    /// would have named today's defect on the first run: the value alone
    /// does not say whether the checkpoint spoke or whether this family's
    /// default was applied. The validator's own statistic follows, so the
    /// summary states not just what was configured but what the WEIGHTS say.
    public func loadSummary(
        weights: URL, options: RunnerLoadOptions
    ) -> RunnerLoadSummary {
        var summary = genericLoadSummary(weights: weights, options: options)
        let text = model.configuration

        summary.add("rms_norm_weight_offset", text.rmsNormWeightOffset)
        summary.add("rms_norm_weight_offset_source", text.rmsNormWeightOffsetSource.rawValue)
        // What the loaded tensors READ AS, next to what was configured.
        // Reads resident tensors only.
        if let verdict = Qwen4ExpNormConvention.read(
            model: model, expectedOffset: text.rmsNormWeightOffset)
        {
            summary.add("norm_convention_tensors_read", verdict.inspected)
            summary.add("norm_convention_disagreements", verdict.disagreements.count)
            summary.add("norm_convention_implied_offset", verdict.impliedOffset)
            if let first = Qwen4ExpNormConvention.nonGatedNormWeights(model).first {
                let measured = Qwen4ExpNormConvention.measure(first.weight)
                summary.add("norm_convention_sample_path", first.path)
                summary.add(
                    "norm_convention_sample_negative_fraction", measured.negativeFraction)
                summary.add("norm_convention_sample_mean", measured.mean)
            }
        }

        summary.add("rms_norm_eps", text.rmsNormEps)
        summary.add("hidden_size", text.hiddenSize)
        summary.add("hidden_layers", text.hiddenLayers)
        summary.add("attention_heads", text.attentionHeads)
        summary.add("kv_heads", text.kvHeads)
        summary.add("head_dim", text.headDim)
        summary.add("vocabulary_size", text.vocabularySize)
        summary.add("full_attention_interval", text.fullAttentionInterval)
        summary.add("num_experts", text.numExperts)
        summary.add("num_experts_per_tok", text.numExpertsPerTok)
        summary.add("moe_intermediate_size", text.moeIntermediateSize)
        summary.add("tie_word_embeddings", text.tieWordEmbeddings)

        // Synthetic recurrent slots: the PLE short-conv state and the n-gram
        // history ride `modelLayerIndex` values PAST the last real layer, so
        // a reader can tell them from a layer number.
        let synthetic = model.cbv2RecurrentStateSpec.layers
            .map { $0.modelLayerIndex }
            .filter { $0 >= text.hiddenLayers }
            .sorted()
        summary.add(
            "aux_state_synthetic_indices",
            synthetic.map(String.init).joined(separator: ","))
        summary.add("ple_layers", model.pleEmbeddings.count)

        // Tensors, counted off the resident tree — no file is reopened.
        let parameters = model.parameters().flattened()
        summary.add("tensors_bound", parameters.count)
        var quantizationCounts: [String: Int] = [:]
        for (path, _) in parameters where path.hasSuffix(".scales") {
            quantizationCounts["quantized", default: 0] += 1
        }
        summary.add("quantized_projections", quantizationCounts["quantized"] ?? 0)

        summary.add(
            "resources_accepted",
            options.resources[Self.ngramRowSourceResource] == nil
                ? "none" : Self.ngramRowSourceResource)
        return summary
    }

    /// Rebuild the checkpoint's `mtp.*` head as this repository's own module
    /// and move the loaded tensors into it.
    ///
    /// THIS IS THE HEAD RE-QUANTIZATION SEAM, and it is code, not
    /// configuration. `geometry` below is the ONE place that decides what
    /// quantization the served head carries.
    ///
    /// The default is the CHECKPOINT'S OWN geometry: every projection the
    /// loader quantized is quantized here with that projection's group size,
    /// bit width and mode, and the loaded arrays are then bound unchanged. So
    /// the served head is bit-exact with the head the pinned fork builds —
    /// same tensors, same dtypes, same scales and biases — and the default
    /// path changes nothing a run can measure.
    ///
    /// TO RE-QUANTIZE THE HEAD, change the geometry this call selects. A
    /// participant who wants a different group size, bit width or mode
    /// dequantizes the loaded projection and quantizes it again here, at this
    /// call, before the parameters are bound. The head weights are the
    /// checkpoint's and stay the checkpoint's: this seam re-encodes what the
    /// checkpoint carries, it does not replace it and it does not ship a head.
    public static func adoptMTPHead(
        _ loaded: Qwen4ExpMTPModule, configuration: Qwen4ExpTextConfiguration
    ) throws -> TrackQwen4ExpMTPModule {
        let head = TrackQwen4ExpMTPModule(configuration, layerCount: loaded.layerCount)

        // The geometry the served head carries. Read off the loaded head, so
        // the default is the checkpoint's own.
        var geometry: [String: (groupSize: Int, bits: Int, mode: QuantizationMode)] = [:]
        for (path, module) in loaded.leafModules().flattened() {
            guard let quantized = module as? Quantized else { continue }
            geometry[path] = (quantized.groupSize, quantized.bits, quantized.mode)
        }
        quantize(model: head) { path, _ in geometry[path] }

        // Verified in full: a head that took only part of the checkpoint's
        // tensors would still draft, and would draft something else.
        try head.update(parameters: loaded.parameters(), verify: .all)
        return head
    }

    /// Refuse a module whose loaded norm weights contradict the offset it
    /// was configured with.
    ///
    /// Ported from the previous engine's `validateLoadedNormConvention`
    /// (mlxfast-qwen38-125b-a6b-engine-dev
    /// `Sources/MLXFastHarness/Qwen4ExpNormConventionBind.swift:162-230`),
    /// thresholds and gated/non-gated split unchanged.
    static func validateNormConvention(_ model: Qwen4ExpModel) throws {
        let expected = model.configuration.rmsNormWeightOffset
        guard
            let mismatch = Qwen4ExpNormConvention.mismatch(
                model: model, expectedOffset: expected)
        else { return }
        throw RunnerError.normConventionMismatch(
            expectedOffset: expected, observed: mismatch.observed)
    }

    /// Resolve the n-gram row source from the resource the caller gave.
    ///
    /// The runner names the `Qwen4ExpNGramRowSource` seam and one construction
    /// entry point, `Qwen4ExpNGramRowSourceLoader`, and no conformer. A later
    /// conformer is chosen inside the loader, so this stays as it is.
    static func resolveNGramRowSource(
        _ resource: AnyObject?, for model: Qwen4ExpModel
    ) throws -> any Qwen4ExpNGramRowSource {
        if let source = resource as? Qwen4ExpNGramRowSource {
            return source
        }
        if let url = resource as? URL {
            return try Qwen4ExpNGramRowSourceLoader.rowSource(at: url, for: model)
        }
        if let path = resource as? String {
            return try Qwen4ExpNGramRowSourceLoader.rowSource(
                at: URL(fileURLWithPath: path), for: model)
        }
        throw RunnerError.resourceMissing(
            "\(ngramRowSourceResource): this checkpoint has "
                + "\(model.pleEmbeddings.count) PLE layers and the n-gram table "
                + "is never model parameters; pass the n-gram shard directory")
    }

    /// The family's own caches. `newCacheV2` still runs the vending closure
    /// for every layer and then discards what it returns: the QSA indexer
    /// needs `Qwen4ExpCBv2LayerCache`, whose second tape trims with the
    /// key-value tape so an MTP rollback keeps the two in step.
    private func newCaches(
        _ make: (_ layerIndex: Int, _ kind: CBv2LayerKind) throws ->
            any CBv2AttendingLayerCache
    ) throws -> [any CBv2AttendingLayerCache] {
        try model.newCacheV2(makeLayerCache: make)
    }

    public func makeEngine(_ build: EngineBuild) throws -> any CBv2Engine {
        try RunnerEngineAssembly.makeEngine(
            manifest: Self.manifest,
            loadedDecoders: loadedDecoders,
            model: servingModel,
            tokenizer: tokenizer,
            layerKinds: layerKinds,
            newCaches: newCaches,
            mtpDrafter: drafter,
            build: build)
    }

    public func makeStepper() throws -> any TeacherForcedStepper {
        CBv2SingleRowStepper(
            model: servingModel,
            layerKinds: layerKinds,
            newCaches: newCaches,
            kvBytesCapacity: kvBytesCapacity,
            maxLength: maxSequenceLength)
    }
}
