import Foundation
import MLXFastCore
@testable import MLXFastTransform
import Testing

// Transform-side conformance for the Qwen 3.8 125B A6B family: family
// detection against the other families in this tree, the uniform quantization
// contract, and the exact 3,414-tensor inventory.
//
// EVERYTHING HERE IS SYNTHETIC and derived from the validator's own pinned
// geometry rather than from a checkpoint capture, so these tests pin the
// validator's INTERNAL consistency and its rejection behaviour. The
// artifact-facing half -- does the real checkpoint's header set equal this
// inventory? -- is pinned separately against the published index and the
// per-shard headers.

private typealias Geometry = Qwen4ExpCheckpointValidation.PinnedGeometry

// MARK: - Family detection

/// This target names itself at EITHER level -- `qwen4_exp` at the root, or
/// `qwen4_exp_text` inside `text_config` -- and the legacy Gemma families name
/// neither. This pins that, and the ORDER that makes it work: the legacy
/// branch is an unconditional `return .gemma4` for any config carrying a
/// `text_config`, so a discriminator placed after it would never run.
@Test
func qwen4ExpIsDetectedByModelTypeAtEitherLevel() throws {
    let root: [String: Any] = ["model_type": "qwen4_exp", "text_config": [:]]
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: root) == .qwen4Exp)

    let nested: [String: Any] = [
        "model_type": "somethingElse",
        "text_config": ["model_type": "qwen4_exp_text"],
    ]
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: nested) == .qwen4Exp)

    // The legacy dense Gemma family declares neither and still routes there.
    let dense: [String: Any] = [
        "model_type": "gemma4",
        "text_config": ["model_type": "gemma4_text"],
    ]
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: dense) == .gemma4)
}

/// The architecture fallback needs BOTH of its halves. Either alone would
/// misroute: the expert count alone catches any future 512-expert model, and
/// the deltanet schedule alone catches any future hybrid one.
@Test
func qwen4ExpArchitectureFallbackNeedsBothHalves() throws {
    let expertsOnly: [String: Any] = [
        "text_config": ["num_experts": 512, "layer_types": ["full_attention"]]
    ]
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: expertsOnly) == .gemma4)

    let scheduleOnly: [String: Any] = [
        "text_config": ["num_experts": 64, "layer_types": ["linear_attention"]]
    ]
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: scheduleOnly) == .gemma4)

    // Both halves, no model_type: the fallback claims it.
    let both: [String: Any] = [
        "text_config": [
            "num_experts": 512,
            "layer_types": ["linear_attention", "full_attention"],
        ]
    ]
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: both) == .qwen4Exp)

    // Wrong JSON kind is not a near-miss to be coerced: a string "512" leaves
    // the config on the legacy path rather than selecting a family whose
    // validator would then reject it with a confusing inventory error.
    let stringly: [String: Any] = [
        "text_config": ["num_experts": "512", "layer_types": ["linear_attention"]]
    ]
    #expect(try SwiftTransform.detectModelFamily(sourceConfigRoot: stringly) == .gemma4)
}

/// Qwen still wins over both Gemma families: its `text_config` could otherwise
/// be probed for MoE keys it does not carry, but the model-type prefix is
/// checked first and this pins that ordering survives the new branch.
@Test
func qwenStillRoutesAheadOfEitherGemmaFamily() throws {
    #expect(
        try SwiftTransform.detectModelFamily(
            sourceConfigRoot: ["text_config": ["model_type": "qwen3_5_text"]]
        ) == .qwen35
    )
}

// MARK: - The quantization block

/// Build the pinned `quantization` block. UNIFORM on this target: the three
/// scalars ARE the whole block, and a per-tensor entry is refused rather than
/// carried, which the drift test below pins.
private func pinnedQuantizationBlock() -> [String: Any] {
    let block: [String: Any] = [
        "group_size": Geometry.quantizationGroupSize,
        "bits": Geometry.quantizationBits,
        "mode": Geometry.quantizationMode,
    ]
    return block
}

private func pinnedSourceConfig(
    textConfig: [String: Any]? = nil,
    quantization: [String: Any]? = nil,
    quantizationConfig: [String: Any]? = nil
) -> [String: Any] {
    let block = quantization ?? pinnedQuantizationBlock()
    return [
        "model_type": "qwen4_exp",
        "text_config": textConfig ?? [
            "model_type": "qwen4_exp_text",
            "num_experts": Geometry.expertCount,
            "num_hidden_layers": Geometry.layerCount,
            "layer_types": ["linear_attention", "full_attention"],
        ],
        "quantization": block,
        "quantization_config": quantizationConfig ?? block,
    ]
}

@Test
func quantizationSpecParsesTheUniformBlock() throws {
    let spec = try Qwen4ExpCheckpointValidation.quantizationSpec(
        fromConfigRoot: pinnedSourceConfig())
    #expect(spec.groupSize == 32)
    #expect(spec.bits == 4)
    #expect(spec.mode == "affine")
    #expect(
        spec.overrides.count
            == Qwen4ExpCheckpointValidation.expectedQuantizationOverrideCount)
    #expect(spec.overrides.isEmpty)
    // UNIFORM: every path resolves to the same width, including the ones a
    // promoted target would have singled out.
    for path in [
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj",
        "language_model.model.layers.47.self_attn.q_proj",
        "language_model.model.embed_tokens",
        "language_model.mtp.fc_hidden",
    ] {
        #expect(spec.spec(forPath: path).bits == 4)
        #expect(spec.spec(forPath: path).groupSize == 32)
    }
}

@Test
func quantizationSpecRequiresTheDuplicateBlocksToAgree() throws {
    var divergent = pinnedQuantizationBlock()
    divergent["language_model.model.layers.7.mlp.up_proj"] = [
        "group_size": 64, "bits": 4,
    ]
    #expect(throws: (any Error).self) {
        _ = try Qwen4ExpCheckpointValidation.quantizationSpec(
            fromConfigRoot: pinnedSourceConfig(quantizationConfig: divergent))
    }
    // One block alone is accepted -- the transform must not demand a duplicate
    // the checkpoint may legitimately not publish.
    var single = pinnedSourceConfig()
    single.removeValue(forKey: "quantization_config")
    #expect(throws: Never.self) {
        _ = try Qwen4ExpCheckpointValidation.quantizationSpec(
            fromConfigRoot: single)
    }
}

/// This target is UNIFORM, so the whole override surface is a REFUSAL surface:
/// an entry the emitter carried would be a width the runtime never reads.
@Test
func quantizationSpecRejectsEveryShapeOfBlockDrift() throws {
    func rejected(_ mutate: (inout [String: Any]) -> Void) -> Bool {
        var block = pinnedQuantizationBlock()
        mutate(&block)
        do {
            _ = try Qwen4ExpCheckpointValidation.quantizationSpec(
                fromConfigRoot: pinnedSourceConfig(
                    quantization: block, quantizationConfig: block))
            return false
        } catch {
            return true
        }
    }

    // A per-tensor override of ANY shape, which this target does not have.
    #expect(
        rejected {
            $0["language_model.model.layers.3.mlp.switch_mlp.up_proj"] = [
                "group_size": 32, "bits": 8,
            ]
        })
    #expect(
        rejected {
            $0["language_model.model.layers.3.mlp.switch_mlp.up_proj"] = [
                "group_size": 32, "bits": 4,
            ]
        })
    // A non-object value under a tensor path is still an unexpected key.
    #expect(rejected { $0["language_model.model.layers.3.mlp.up_proj"] = 8 })
    // A changed fallback: this checkpoint is affine 4-bit group-32.
    #expect(rejected { $0["bits"] = 8 })
    #expect(rejected { $0["group_size"] = 64 })
    #expect(rejected { $0["mode"] = "nvfp4" })
}

// MARK: - The emitted runtime config

/// THE REGRESSION THIS EXISTS FOR. The `.qwen35` branch of
/// `makeRuntimeConfigData` REBUILDS the emitted `quantization` from a parsed
/// `{group_size, bits, mode}` triple, which is lossless only because the Qwen
/// block has exactly three keys. Copying that shape here would emit a config
/// declaring uniform 4-bit for tensors the shards were written at 8 -- right
/// names, right shapes, wrong numerics, and nothing downstream notices.
///
/// Since 2026-09-12 the tree DOES carry 8-bit tensors: the served head. Its
/// per-module entries are emitted from the pinned geometry, one for each of
/// the head's quantized projections, keyed `mtp.*` the way the runtime walks
/// its module tree. The tower's scalars stay uniform 4-bit.
@Test
func emittedRuntimeConfigCarriesTheUniformQuantizationBlock() throws {
    let data = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: pinnedSourceConfig(), family: .qwen4Exp)
    let emitted = try #require(
        try JSONSerialization.jsonObject(with: data) as? [String: Any])

    // The text tower is flattened to the top level.
    #expect(emitted["text_config"] == nil)
    #expect(emitted["num_experts"] as? Int == Geometry.expertCount)
    #expect(emitted["layer_types"] as? [String] == ["linear_attention", "full_attention"])
    // The duplicate is removed once the two were verified to agree.
    #expect(emitted["quantization_config"] == nil)
    // And nothing else is invented.
    #expect(emitted["vision_config"] == nil)

    let quantization = try #require(emitted["quantization"] as? [String: Any])
    #expect(quantization["group_size"] as? Int == 32)
    #expect(quantization["bits"] as? Int == 4)
    #expect(quantization["mode"] as? String == "affine")
    let overrideKeys = quantization.keys.filter {
        !["group_size", "bits", "mode"].contains($0)
    }
    #expect(
        overrideKeys.count
            == Qwen4ExpCheckpointValidation.expectedMTPQuantizedModuleCount)
    #expect(
        Set(overrideKeys)
            == Set(Qwen4ExpCheckpointValidation.runtimeHeadQuantizationOverrides().keys))
    for key in overrideKeys {
        // Runtime module paths: the checkpoint's `language_model.` prefix is
        // gone, and every entry is the served head's own width.
        #expect(key.hasPrefix("mtp."), "\(key)")
        #expect(!key.hasPrefix("language_model."), "\(key)")
        let entry = try #require(quantization[key] as? [String: Any])
        #expect(entry["bits"] as? Int == Geometry.mtpHeadServedQuantizationBits)
        #expect(entry["group_size"] as? Int == Geometry.quantizationGroupSize)
        #expect(entry["mode"] as? String == Geometry.quantizationMode)
    }
    // The tower is untouched: no entry names a tower module.
    #expect(!overrideKeys.contains { $0.hasPrefix("model.") || $0.hasPrefix("lm_head") })
}

/// The legacy `.gemma4` family emits the projection and tied-head sidecars;
/// `.qwen4Exp` must not, and the transform's sidecar switch is where that is
/// decided. This pins the family's membership of the emit-nothing branch by
/// exercising the one observable consequence available without a checkpoint:
/// the runtime config is the flattened tower and the quantization block, and
/// nothing else.
@Test
func emittedRuntimeConfigCarriesOnlyTheTowerAndTheQuantizationBlock() throws {
    let data = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: pinnedSourceConfig(), family: .qwen4Exp)
    let emitted = try #require(
        try JSONSerialization.jsonObject(with: data) as? [String: Any])
    let textKeys: Set<String> = [
        "model_type", "num_experts", "num_hidden_layers", "layer_types",
    ]
    #expect(
        Set(emitted.keys)
            == textKeys.union([
                "quantization",
                Qwen4ExpCheckpointValidation.rmsNormWeightOffsetKey,
            ]))
}

// MARK: - The norm convention travels with the tree

/// THE REGRESSION THIS EXISTS FOR. No published `config.json` says how the
/// non-gated norm weights are stored. This checkpoint bakes the offset, the
/// reference implementation assumes it does not, and a consumer that reads
/// only the transformed file therefore computes `y * (1 + w)` on weights that
/// already hold `1 + w` -- every non-gated norm in the tower, the mixer, the
/// PLE norms, the indexer layernorms and the MTP head. It is fixed at module
/// construction, so nothing downstream can catch it. The transform writes the
/// key so the tree carries its own convention.
///
/// TOP LEVEL, not inside a `text_config`: the emitted config IS the flattened
/// tower, and a transformed tree carries no `text_config` block at all.
@Test
func emittedRuntimeConfigCarriesThePinnedNormConvention() throws {
    let data = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: pinnedSourceConfig(), family: .qwen4Exp)
    let emitted = try #require(
        try JSONSerialization.jsonObject(with: data) as? [String: Any])

    #expect(emitted["text_config"] == nil)
    let offset = try #require(
        emitted[Qwen4ExpCheckpointValidation.rmsNormWeightOffsetKey] as? NSNumber)
    // Derived from the pin, never a literal, so a repin moves the file too.
    #expect(
        offset.doubleValue
            == Double(MLXFastConstants.rmsNormConvention.weightOffset))
    // And the pin is the baked convention this checkpoint was measured to use.
    #expect(MLXFastConstants.rmsNormConvention == .offsetBaked)
    #expect(offset.doubleValue == 0)
}

/// The emitted config satisfies the verifier's own requirement. A transform
/// that wrote a key the verifier then refused would be a tree nobody could
/// use.
@Test
func theEmittedRuntimeConfigPassesTheNormConventionGate() throws {
    let data = try SwiftTransform.makeRuntimeConfigData(
        sourceConfigRoot: pinnedSourceConfig(), family: .qwen4Exp)
    let emitted = try #require(
        try JSONSerialization.jsonObject(with: data) as? [String: Any])
    try Qwen4ExpCheckpointValidation.validateNormConvention(
        inRuntimeConfigRoot: emitted)
}

/// FAIL CLOSED. A missing key is a refusal, not a default: the runtime's own
/// decode falls back to the zero-centered `1`, so a tree accepted without the
/// key is a silently wrong tower.
@Test
func theNormConventionGateRefusesATreeThatDoesNotDeclareIt() throws {
    let key = Qwen4ExpCheckpointValidation.rmsNormWeightOffsetKey
    let good: [String: Any] = [
        "model_type": "qwen4_exp_text",
        key: 0,
    ]
    try Qwen4ExpCheckpointValidation.validateNormConvention(
        inRuntimeConfigRoot: good)

    // Absent.
    #expect(throws: MLXFastError.self) {
        try Qwen4ExpCheckpointValidation.validateNormConvention(
            inRuntimeConfigRoot: ["model_type": "qwen4_exp_text"])
    }
    // Present but null, which is how a hand-edited config spells "absent".
    #expect(throws: MLXFastError.self) {
        try Qwen4ExpCheckpointValidation.validateNormConvention(
            inRuntimeConfigRoot: ["model_type": "qwen4_exp_text", key: NSNull()])
    }
    // Present with the WRONG convention -- the reference default is exactly
    // the value this checkpoint must not use.
    #expect(throws: MLXFastError.self) {
        try Qwen4ExpCheckpointValidation.validateNormConvention(
            inRuntimeConfigRoot: ["model_type": "qwen4_exp_text", key: 1])
    }
    // And a value that is not a number at all.
    #expect(throws: MLXFastError.self) {
        try Qwen4ExpCheckpointValidation.validateNormConvention(
            inRuntimeConfigRoot: ["model_type": "qwen4_exp_text", key: "0"])
    }
}

@Test
func textTowerSelectionKeepsOnlyTheLanguageModelPrefix() {
    #expect(
        SwiftTransform.isSelectedTextTowerKey(
            "language_model.model.layers.0.mlp.gate_proj.weight",
            family: .qwen4Exp))
    for dropped in [
        "vision_tower.blocks.0.attn.qkv.weight",
        "embed_vision.weight",
        "multi_modal_projector.linear.weight",
    ] {
        #expect(!SwiftTransform.isSelectedTextTowerKey(dropped, family: .qwen4Exp))
    }
}

// MARK: - The exact 1,339-tensor inventory

@Test
func expectedInventoryMatchesThePinnedTensorCounts() {
    let inventory = Qwen4ExpCheckpointValidation.expectedTensorInventory()
    #expect(inventory.count == 3_414)
    #expect(
        inventory.count == Qwen4ExpCheckpointValidation.expectedTensorCount)

    // 13 top level: the embedding's triple, the untied head's triple, and the
    // tower's final hyper-connection mixer. THERE IS NO `model.norm`.
    let topLevel = inventory.keys.filter {
        !$0.contains(".layers.") && !$0.hasPrefix("language_model.mtp.")
    }
    #expect(
        topLevel.count
            == Qwen4ExpCheckpointValidation.expectedTopLevelTensorCount)
    #expect(
        Set(topLevel) == [
            "language_model.model.embed_tokens.weight",
            "language_model.model.embed_tokens.scales",
            "language_model.model.embed_tokens.biases",
            "language_model.lm_head.weight",
            "language_model.lm_head.scales",
            "language_model.lm_head.biases",
            "language_model.model.hyper_connection_mixer.hc_norm.weight",
            "language_model.model.hyper_connection_mixer.input_mix_weight_down.weight",
            "language_model.model.hyper_connection_mixer.input_mix_weight_down.scales",
            "language_model.model.hyper_connection_mixer.input_mix_weight_down.biases",
            "language_model.model.hyper_connection_mixer.input_mix_weight_up.weight",
            "language_model.model.hyper_connection_mixer.input_mix_weight_up.scales",
            "language_model.model.hyper_connection_mixer.input_mix_weight_up.biases",
        ])
    #expect(!inventory.keys.contains("language_model.model.norm.weight"))

    // 36 linear layers and 12 full-attention layers, 61 tensors each. The two
    // kinds differ in WHICH 19 they carry, not how many.
    var linearLayers = 0
    var globalLayers = 0
    for layer in 0..<Geometry.layerCount {
        let prefix = "language_model.model.layers.\(layer)."
        let count = inventory.keys.filter {
            $0.hasPrefix(prefix) && !$0.contains(".ple.")
        }.count
        if Geometry.isFullAttention(layer: layer) {
            globalLayers += 1
            #expect(
                count
                    == Qwen4ExpCheckpointValidation
                    .expectedFullAttentionLayerTensorCount,
                "layer \(layer)")
        } else {
            linearLayers += 1
            #expect(
                count
                    == Qwen4ExpCheckpointValidation
                    .expectedLinearLayerTensorCount,
                "layer \(layer)")
        }
    }
    #expect(globalLayers == 12)
    #expect(linearLayers == 36)
    // The remainder is the per-layer embedding block, its 384-tensor table,
    // and the embedded head.
    #expect(
        Qwen4ExpCheckpointValidation.expectedTopLevelTensorCount
            + linearLayers
            * Qwen4ExpCheckpointValidation.expectedLinearLayerTensorCount
            + globalLayers
            * Qwen4ExpCheckpointValidation
            .expectedFullAttentionLayerTensorCount
            + Qwen4ExpCheckpointValidation.expectedPLETensorCount
            + Qwen4ExpCheckpointValidation.expectedNGramShardTensorCount
            + Qwen4ExpCheckpointValidation.expectedMTPTensorCount
            == Qwen4ExpCheckpointValidation.expectedTensorCount)
}

/// Presence and absence are both facts of this checkpoint, asserted as such
/// rather than as tolerated extras.
@Test
func expectedInventoryEncodesTheUntiedHeadAndTheHybridSchedule() {
    let inventory = Qwen4ExpCheckpointValidation.expectedTensorInventory()
    // The head is UNTIED, so it ships its own triple.
    #expect(inventory["language_model.lm_head.weight"] != nil)
    // The embedded multi-token-prediction head ships NO embedding of its own.
    #expect(inventory["language_model.mtp.embed_tokens.weight"] == nil)
    for layer in 0..<Geometry.layerCount {
        let attention =
            "language_model.model.layers.\(layer).self_attn.q_proj.weight"
        let linear =
            "language_model.model.layers.\(layer).linear_attn.conv1d.weight"
        #expect(
            (inventory[attention] != nil) == Geometry.isFullAttention(layer: layer),
            "layer \(layer)")
        #expect(
            (inventory[linear] != nil) != Geometry.isFullAttention(layer: layer),
            "layer \(layer)")
    }
    // The per-layer embedding block sits on exactly one layer.
    for layer in 0..<Geometry.layerCount {
        let ple = "language_model.model.layers.\(layer).ple.conv1d.weight"
        #expect(
            (inventory[ple] != nil)
                == Geometry.pleLayerIndices.contains(layer), "layer \(layer)")
    }
}

@Test
func expectedInventoryPacksEachPathAtTheUniformWidth() {
    let inventory = Qwen4ExpCheckpointValidation.expectedTensorInventory()
    func shape(_ name: String) -> [Int] { inventory[name]?.shape ?? [] }

    // Uniform 4-bit group-32: hidden 2560 packs to 2560 * 4 / 32 = 320 U32
    // columns, with 2560 / 32 = 80 group columns.
    #expect(shape("language_model.model.embed_tokens.weight") == [248_320, 320])
    #expect(shape("language_model.model.embed_tokens.scales") == [248_320, 80])
    #expect(shape("language_model.lm_head.weight") == [248_320, 320])

    // A full-attention layer. `q_proj` carries the output gate too, so it is
    // 24 * 256 * 2 wide; `k_proj` and `v_proj` are 2 * 256.
    #expect(
        shape("language_model.model.layers.3.self_attn.q_proj.weight")
            == [12_288, 320])
    #expect(
        shape("language_model.model.layers.3.self_attn.k_proj.weight")
            == [512, 320])
    #expect(shape("language_model.model.layers.3.self_attn.q_norm.weight") == [256])
    #expect(
        shape("language_model.model.layers.3.self_attn.indexer.index_qk_proj.weight")
            == [640, 320])
    #expect(
        shape("language_model.model.layers.3.self_attn.indexer.q_layernorm.weight")
            == [128])

    // A linear-attention layer. The fused convolution is 2 * 16 * 128 plus
    // 48 * 128 = 10240 channels, in MLX layout (channels, kernel, 1).
    #expect(
        shape("language_model.model.layers.0.linear_attn.conv1d.weight")
            == [10_240, 4, 1])
    #expect(
        shape("language_model.model.layers.0.linear_attn.in_proj_qkv.weight")
            == [10_240, 320])
    #expect(
        shape("language_model.model.layers.0.linear_attn.in_proj_z.weight")
            == [6_144, 320])
    #expect(shape("language_model.model.layers.0.linear_attn.A_log") == [48])
    #expect(shape("language_model.model.layers.0.linear_attn.norm.weight") == [128])

    // Experts are STACKED: a leading experts axis, never split per expert.
    // The router stays unquantized.
    #expect(
        shape("language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight")
            == [512, 640, 320])
    #expect(
        shape("language_model.model.layers.0.mlp.switch_mlp.down_proj.weight")
            == [512, 2_560, 80])
    #expect(shape("language_model.model.layers.0.mlp.gate.weight") == [512, 2_560])

    // Hyper-connections run at hc_count * hidden = 10240.
    #expect(
        shape("language_model.model.layers.0.attn_hyper_connection.hc_norm.weight")
            == [10_240])
    #expect(
        shape(
            "language_model.model.layers.0.attn_hyper_connection.input_mix_weight_down.weight"
        ) == [320, 1_280])
    #expect(
        shape("language_model.model.layers.0.attn_hyper_connection.block_inject_weight.weight")
            == [4, 1_280])

    // The n-gram table: 160 values a row at 4 bits is 20 U32 columns, with
    // 160 / 32 = 5 group columns. That is the 100 bytes a row the offload
    // design is built on.
    let shard0 =
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_0"
    #expect(shape("\(shard0).weight") == [Geometry.ngramRowsPerShard, 20])
    #expect(shape("\(shard0).scales") == [Geometry.ngramRowsPerShard, 5])
    #expect(Geometry.ngramRowsPerShard == 2_500_012)

    // The embedded head reads the pre-final-mixer stream, so its hidden norm
    // is hc_count * hidden wide while its embedding norm is hidden wide.
    #expect(shape("language_model.mtp.pre_fc_norm_hidden.weight") == [10_240])
    #expect(shape("language_model.mtp.pre_fc_norm_embedding.weight") == [2_560])
    #expect(shape("language_model.mtp.fc_hidden.weight") == [2_560, 320])
}

// MARK: - validateSelectedTensors against a synthetic exact checkpoint

private func gemmaValidationMetadata(
    mutate: (inout [String: Qwen4ExpCheckpointValidation.ExpectedTensorMetadata])
        -> Void = { _ in }
) -> (
    selectedKeys: Set<String>, index: CheckpointIndex,
    headers: [String: SafetensorsHeader]
) {
    var inventory = Qwen4ExpCheckpointValidation.expectedTensorInventory()
    mutate(&inventory)

    let shardName = "model-00001-of-00001.safetensors"
    var weightMap: [String: String] = [:]
    var tensors: [String: SafetensorInfo] = [:]
    for (name, metadata) in inventory {
        weightMap[name] = shardName
        tensors[name] = SafetensorInfo(
            name: name,
            dtype: metadata.dtype,
            shape: metadata.shape,
            dataStart: 0,
            dataEnd: 1
        )
    }
    return (
        Set(inventory.keys),
        CheckpointIndex(raw: ["weight_map": weightMap], weightMap: weightMap),
        [
            shardName: SafetensorsHeader(
                headerLength: 8, metadata: ["format": "pt"], tensors: tensors)
        ]
    )
}

@Test
func transformAcceptsTheExactPublicTextTower() throws {
    let metadata = gemmaValidationMetadata()
    #expect(
        metadata.selectedKeys.count
            == Qwen4ExpCheckpointValidation.expectedTensorCount)
    try Qwen4ExpCheckpointValidation.validateSelectedTensors(
        selectedKeys: metadata.selectedKeys,
        index: metadata.index,
        headers: metadata.headers,
        quantization: try Qwen4ExpCheckpointValidation.quantizationSpec(
            fromConfigRoot: pinnedSourceConfig())
    )
}

@Test
func transformRejectsInventoryAndPackingDrift() throws {
    let quantization = try Qwen4ExpCheckpointValidation.quantizationSpec(
        fromConfigRoot: pinnedSourceConfig())

    func rejected(
        _ mutate: (
            inout [String: Qwen4ExpCheckpointValidation.ExpectedTensorMetadata]
        ) -> Void
    ) -> Bool {
        let metadata = gemmaValidationMetadata(mutate: mutate)
        do {
            try Qwen4ExpCheckpointValidation.validateSelectedTensors(
                selectedKeys: metadata.selectedKeys,
                index: metadata.index,
                headers: metadata.headers,
                quantization: quantization
            )
            return false
        } catch {
            return true
        }
    }

    let quantized = Qwen4ExpCheckpointValidation.ExpectedTensorMetadata.self
    // A packed weight stored at the wrong width. This is the packing check
    // earning its place: the name and the leading axis are right and only the
    // packed column count is wrong, which is what a width mistake looks like.
    #expect(
        rejected {
            $0["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"] =
                quantized.init(dtype: "U32", shape: [512, 640, 640])
        })
    // A LINEAR layer that ships attention projections after all: the layer
    // schedule is a tensor-inventory fact, not only a numerics one.
    #expect(
        rejected {
            for component in ["weight", "scales", "biases"] {
                $0["language_model.model.layers.0.self_attn.q_proj.\(component)"] =
                    quantized.init(
                        dtype: component == "weight" ? "U32" : "BF16",
                        shape: component == "weight"
                            ? [12_288, 320] : [12_288, 80])
            }
        })
    // A full-attention layer that drops the QSA indexer it must ship.
    #expect(
        rejected {
            $0.removeValue(
                forKey:
                    "language_model.model.layers.3.self_attn.indexer.index_qk_proj.weight")
        })
    // The untied output head DROPPED. This target ships one, so its absence
    // is the error -- the direction that inverted with the tower.
    #expect(
        rejected { $0.removeValue(forKey: "language_model.lm_head.weight") })
    // The embedded multi-token-prediction head DROPPED. There is no separate
    // head artifact on this track, so a tree without it has no speculative
    // arm at all.
    #expect(
        rejected {
            $0.removeValue(forKey: "language_model.mtp.fc_hidden.weight")
        })
    // Compressed-tensors and FP8 KV-scale aliases.
    #expect(
        rejected {
            $0["language_model.model.layers.3.self_attn.k_proj.k_scale"] =
                quantized.init(dtype: "BF16", shape: [1])
        })
    // A quantized projection missing its bias companion (affine needs both).
    #expect(
        rejected {
            $0.removeValue(
                forKey: "language_model.model.layers.3.self_attn.q_proj.biases")
        })
    // Wrong dtype on a packed weight.
    #expect(
        rejected {
            $0["language_model.model.layers.3.self_attn.q_proj.weight"] =
                quantized.init(dtype: "BF16", shape: [4_096, 352])
        })
    // Wrong shape on an unquantized norm.
    #expect(
        rejected {
            $0["language_model.model.layers.3.self_attn.q_norm.weight"] =
                quantized.init(dtype: "BF16", shape: [512])
        })
    // A stacked expert tensor flattened into rank 2.
    #expect(
        rejected {
            $0["language_model.model.layers.0.experts.switch_glu.gate_proj.weight"] =
                quantized.init(dtype: "U32", shape: [90_112, 352])
        })
}

// MARK: - The served head (David ruling 2026-09-12)

/// The head the transform SERVES is the publisher's 8-bit conversion of the
/// same 76 tensors. Same names, same scale and bias shapes, packed columns
/// doubled. Both tables come out of one builder, so they cannot drift apart.
@Test
func servedHeadInventoryIsTheEmbeddedHeadAtEightBits() throws {
    let embedded = Qwen4ExpCheckpointValidation.expectedHeadInventory(
        bits: Geometry.quantizationBits)
    let served = Qwen4ExpCheckpointValidation.expectedHeadInventory(
        bits: Geometry.mtpHeadServedQuantizationBits)
    #expect(embedded.count == Qwen4ExpCheckpointValidation.expectedMTPTensorCount)
    #expect(Set(embedded.keys) == Set(served.keys))
    #expect(embedded.keys.allSatisfy { $0.hasPrefix("language_model.mtp.") })

    // The source table's head IS the embedded table, entry for entry.
    let source = Qwen4ExpCheckpointValidation.expectedTensorInventory()
    for (name, metadata) in embedded {
        #expect(source[name] == metadata, "\(name)")
    }

    var packed = 0
    for (name, metadata) in served {
        let embeddedMetadata = try #require(embedded[name])
        #expect(metadata.dtype == embeddedMetadata.dtype, "\(name)")
        if metadata.dtype == "U32" {
            packed += 1
            #expect(metadata.shape.dropLast() == embeddedMetadata.shape.dropLast(), "\(name)")
            #expect(metadata.shape.last == embeddedMetadata.shape.last.map { $0 * 2 }, "\(name)")
        } else {
            #expect(metadata.shape == embeddedMetadata.shape, "\(name)")
        }
    }
    #expect(packed == Qwen4ExpCheckpointValidation.expectedMTPQuantizedModuleCount)
    // Read off the pinned 8-bit shards: hidden 2560 packs to 640 U32 columns.
    #expect(served["language_model.mtp.fc_hidden.weight"]?.shape == [2_560, 640])
    #expect(served["language_model.mtp.fc_hidden.scales"]?.shape == [2_560, 80])
    #expect(
        served["language_model.mtp.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape
            == [512, 640, 640])
    #expect(
        served["language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape
            == [512, 2_560, 160])
    #expect(served["language_model.mtp.layers.0.self_attn.o_proj.weight"]?.shape == [2_560, 1_536])
}

@Test
func servedHeadOverridesNameEveryQuantizedHeadModuleOnce() throws {
    let stems = Qwen4ExpCheckpointValidation.headQuantizedModuleStems()
    #expect(stems.count == Qwen4ExpCheckpointValidation.expectedMTPQuantizedModuleCount)
    #expect(stems == stems.sorted())
    #expect(Set(stems).count == stems.count)
    #expect(stems.contains("language_model.mtp.fc_embedding"))
    #expect(stems.contains("language_model.mtp.layers.0.mlp.switch_mlp.up_proj"))
    #expect(stems.contains("language_model.mtp.layers.0.self_attn.indexer.index_qk_proj"))
    // The router is unquantized and the norms carry no scales: no entry.
    #expect(!stems.contains("language_model.mtp.layers.0.mlp.gate"))
    #expect(!stems.contains("language_model.mtp.pre_fc_norm_hidden"))

    let overrides = Qwen4ExpCheckpointValidation.runtimeHeadQuantizationOverrides()
    #expect(overrides.count == stems.count)
    for stem in stems {
        let key = String(stem.dropFirst("language_model.".count))
        let entry = try #require(overrides[key])
        #expect(entry["bits"] as? Int == 8)
        #expect(entry["group_size"] as? Int == 32)
        #expect(entry["mode"] as? String == "affine")
    }
}

/// The head source's own config: the publisher's 8-bit conversion declares
/// the same two-block shape at 8 bits. Anything else is the wrong artifact.
@Test
func headSourceQuantizationSpecPinsEightBits() throws {
    let eightBit: [String: Any] = ["group_size": 32, "bits": 8, "mode": "affine"]
    let spec = try Qwen4ExpCheckpointValidation.headSourceQuantizationSpec(
        fromConfigRoot: pinnedSourceConfig(quantization: eightBit, quantizationConfig: eightBit))
    #expect(spec.bits == 8)
    #expect(spec.groupSize == 32)
    #expect(spec.overrides.isEmpty)

    // The 4-bit target itself is refused as a head source, and so is a
    // promoted or re-grouped variant.
    #expect(throws: MLXFastError.self) {
        _ = try Qwen4ExpCheckpointValidation.headSourceQuantizationSpec(
            fromConfigRoot: pinnedSourceConfig())
    }
    var promoted = eightBit
    promoted["language_model.mtp.fc_hidden"] = ["group_size": 32, "bits": 4]
    #expect(throws: MLXFastError.self) {
        _ = try Qwen4ExpCheckpointValidation.headSourceQuantizationSpec(
            fromConfigRoot: pinnedSourceConfig(quantization: promoted, quantizationConfig: promoted))
    }
    let regrouped: [String: Any] = ["group_size": 64, "bits": 8, "mode": "affine"]
    #expect(throws: MLXFastError.self) {
        _ = try Qwen4ExpCheckpointValidation.headSourceQuantizationSpec(
            fromConfigRoot: pinnedSourceConfig(quantization: regrouped, quantizationConfig: regrouped))
    }
}

/// Synthetic head-source metadata: the served inventory laid out across two
/// shards the way the pinned 8-bit checkpoint lays it out, with the tower
/// tensors that share those shards alongside. Metadata only; no bytes.
private func headSourceMetadata(
    mutate: (inout [String: Qwen4ExpCheckpointValidation.ExpectedTensorMetadata]) -> Void = { _ in }
) -> (index: CheckpointIndex, headers: [String: SafetensorsHeader]) {
    var inventory = Qwen4ExpCheckpointValidation.expectedHeadInventory(
        bits: Geometry.mtpHeadServedQuantizationBits)
    // A tower tensor the pinned shard 41 also carries; the head validator
    // must look past it.
    inventory["language_model.lm_head.weight"] =
        Qwen4ExpCheckpointValidation.ExpectedTensorMetadata(dtype: "U32", shape: [248_320, 640])
    mutate(&inventory)

    let first = "model-00041-of-00042.safetensors"
    let second = "model-00042-of-00042.safetensors"
    var weightMap: [String: String] = [:]
    var tensors: [String: [String: SafetensorInfo]] = [first: [:], second: [:]]
    for (name, metadata) in inventory {
        let shard = name.contains(".mlp.switch_mlp.") || name.contains("lm_head") ? first : second
        weightMap[name] = shard
        tensors[shard, default: [:]][name] = SafetensorInfo(
            name: name, dtype: metadata.dtype, shape: metadata.shape, dataStart: 0, dataEnd: 1)
    }
    return (
        CheckpointIndex(raw: ["weight_map": weightMap], weightMap: weightMap),
        [
            first: SafetensorsHeader(headerLength: 8, metadata: ["format": "mlx"], tensors: tensors[first, default: [:]]),
            second: SafetensorsHeader(headerLength: 8, metadata: ["format": "mlx"], tensors: tensors[second, default: [:]]),
        ]
    )
}

private func eightBitHeadSpec() throws -> Qwen4ExpTransformQuantizationSpec {
    let eightBit: [String: Any] = ["group_size": 32, "bits": 8, "mode": "affine"]
    return try Qwen4ExpCheckpointValidation.headSourceQuantizationSpec(
        fromConfigRoot: pinnedSourceConfig(quantization: eightBit, quantizationConfig: eightBit))
}

@Test
func headSourceValidationAcceptsTheServedHeadAcrossItsTwoShards() throws {
    let metadata = headSourceMetadata()
    try Qwen4ExpCheckpointValidation.validateHeadSource(
        index: metadata.index, headers: metadata.headers, quantization: try eightBitHeadSpec())
}

@Test
func headSourceValidationRejectsTheWrongHead() throws {
    let spec = try eightBitHeadSpec()
    func rejected(
        _ mutate: (inout [String: Qwen4ExpCheckpointValidation.ExpectedTensorMetadata]) -> Void
    ) -> Bool {
        let metadata = headSourceMetadata(mutate: mutate)
        do {
            try Qwen4ExpCheckpointValidation.validateHeadSource(
                index: metadata.index, headers: metadata.headers, quantization: spec)
            return false
        } catch {
            return true
        }
    }
    typealias Metadata = Qwen4ExpCheckpointValidation.ExpectedTensorMetadata
    // The 4-bit head offered as the served head: right names, right leading
    // dimensions, half the packed columns. Exactly the substitution the
    // width check exists to refuse.
    #expect(
        rejected {
            $0["language_model.mtp.fc_hidden.weight"] = Metadata(dtype: "U32", shape: [2_560, 320])
        })
    // A head tensor missing from the shards.
    #expect(rejected { $0.removeValue(forKey: "language_model.mtp.layers.0.self_attn.k_proj.biases") })
    // A head tensor that is not the head's: an embedding of its own.
    #expect(
        rejected {
            $0["language_model.mtp.embed_tokens.weight"] = Metadata(dtype: "U32", shape: [248_320, 640])
        })
    // Wrong dtype on a norm.
    #expect(
        rejected {
            $0["language_model.mtp.layers.0.self_attn.q_norm.weight"] = Metadata(dtype: "F32", shape: [256])
        })
    // Scales and biases that disagree.
    #expect(
        rejected {
            $0["language_model.mtp.fc_hidden.biases"] = Metadata(dtype: "BF16", shape: [2_560, 40])
        })
    // And the 4-bit target's own spec is not a served-head spec at all.
    let fourBit = try Qwen4ExpCheckpointValidation.quantizationSpec(fromConfigRoot: pinnedSourceConfig())
    let metadata = headSourceMetadata()
    #expect(throws: MLXFastError.self) {
        try Qwen4ExpCheckpointValidation.validateHeadSource(
            index: metadata.index, headers: metadata.headers, quantization: fourBit)
    }
}
