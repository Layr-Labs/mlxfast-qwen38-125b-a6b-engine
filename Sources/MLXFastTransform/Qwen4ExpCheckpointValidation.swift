import CoreFoundation
import Foundation
import MLXFastCore

/// One resolved affine width, as it appears either in the checkpoint's
/// fallback scalars or in one of its per-tensor override objects.
struct Qwen4ExpTransformTensorQuantization: Equatable {
    let groupSize: Int
    let bits: Int
}

/// Quantization expectations parsed from the pinned Qwen 3.8 125B A6B
/// checkpoint's `quantization` / `quantization_config` blocks.
///
/// UNLIKE EVERY OTHER TARGET THIS TRANSFORM CARRIES, the block is not a
/// three-key object. It is affine 4-bit / group-64 PLUS 120 per-tensor
/// overrides promoting four projection families to 8 bits on all 30 layers.
/// The Qwen-era spec type modelled exactly `{group_size, bits, mode}` and
/// REJECTED anything else; ported verbatim it would refuse this checkpoint
/// outright, and relaxed the obvious way (ignore unknown keys) it would accept
/// it while discarding every promotion. So the override table is modelled as
/// data and `spec(forPath:)` is the only way widths are read.
struct Qwen4ExpTransformQuantizationSpec: Equatable {
    /// Applies to any path not named in `overrides`.
    let groupSize: Int
    let bits: Int
    /// Affine for this checkpoint. Overrides carry no mode of their own, so
    /// this applies to fallback and overrides alike.
    let mode: String
    /// Checkpoint tensor path -> width, e.g.
    /// `language_model.model.layers.0.mlp.gate_proj`.
    let overrides: [String: Qwen4ExpTransformTensorQuantization]

    var fallback: Qwen4ExpTransformTensorQuantization {
        Qwen4ExpTransformTensorQuantization(groupSize: groupSize, bits: bits)
    }

    /// Resolve the width for one checkpoint tensor stem.
    func spec(forPath path: String) -> Qwen4ExpTransformTensorQuantization {
        overrides[path] ?? fallback
    }
}

/// Transform-side structural validation of the Qwen 3.8 125B A6B text-tower
/// tensor set.
///
/// The source checkpoint (`mlx-community/gemma-4-26B-A4B-it-qat-4bit`) is
/// already MLX affine-quantized, so the transform passes tensors through
/// unchanged. This pass fails fast -- before the multi-GB copy -- when the set
/// it would copy cannot satisfy the runtime loader
/// (`Qwen4ExpWeightLoader`):
///
/// - the selected `language_model.*` namespace must be EXACTLY the public
///   1,339-tensor inventory, tensor for tensor, at the exact dtype and shape;
/// - every quantized projection stored as packed U32 codes must ship BF16
///   `.scales` AND BF16 `.biases` (affine needs both; the Laguna NVFP4
///   contract forbids `.biases`, which is why the two validators cannot share
///   one rule) with matching leading dimensions;
/// - each packed width must match the group size and bit width the emitted
///   config.json declares FOR THAT PATH -- 4-bit group-64 by default, 8-bit
///   group-64 on the 120 promoted tensors. A validator that used one global
///   width here would accept a 4-bit-packed `mlp.gate_proj` on a checkpoint
///   that declares it at 8 bits, which is precisely the silent-numerics
///   failure the override table exists to prevent;
/// - compressed-tensors aliases, global-scale tensors, and FP8 KV scales are
///   rejected outright;
/// - no `lm_head.*` tensor may appear. `tie_word_embeddings` is true, so the
///   output projection IS the embedding; a checkpoint that ships an untied
///   head is a different artifact;
/// - no `mtp.*` tensor may appear. The Gemma 4 MTP head is a separately pinned
///   artifact and the pinned backbone revision contains none.
///
/// Deliberately independent of shard placement: this pass pins the complete
/// tensor namespace, dtype, and shape, while the public inventory fixture pins
/// placement and header digests.
enum Qwen4ExpCheckpointValidation {
    struct ExpectedTensorMetadata: Equatable {
        let dtype: String
        let shape: [Int]
    }

    /// Frozen geometry of the pinned `gemma4_text` tower. Kept as literals
    /// rather than read from the config under validation: a validator that
    /// derives its expectations from the artifact it is checking cannot detect
    /// a changed artifact. (`MLXFastConstants` mirrors these; they are NOT
    /// referenced from there for the same reason the Qwen and Laguna
    /// validators keep their own copies -- the constants block is what a
    /// repin edits, and this block is what catches a repin nobody meant.)
    enum PinnedGeometry {
        static let vocabSize = 248_320
        static let hiddenSize = 2_560
        static let layerCount = 48
        static let fullAttentionInterval = 4
        static let attentionHeads = 24
        static let keyValueHeads = 2
        static let headDim = 256

        static let expertCount = 512
        static let moeIntermediateSize = 640
        static let sharedExpertIntermediateSize = 640

        static let linearKeyHeads = 16
        static let linearValueHeads = 48
        static let linearKeyHeadDim = 128
        static let linearValueHeadDim = 128
        static let linearConvKernelDim = 4

        static let hcCount = 4
        static let hcLowrank = 320

        static let indexerHeads = 4
        static let indexerKeyValueHeads = 1
        static let indexerHeadDim = 128

        static let ngramSize = 3
        static let headsPerNGram = 8
        static let ngramVocabSizeBase = 20_000_000
        static let ngramVocabDivisor = 128
        static let ngramShardCount = 128
        static let pleEmbedDim = 2_560
        static let pleConvKernelSize = 4
        /// ONE-BASED in the checkpoint config, so this is layer INDEX 1.
        static let pleLayerIds = [2]

        static let mtpLayerCount = 1

        static let quantizationGroupSize = 32
        static let quantizationBits = 4
        static let quantizationMode = "affine"
        /// The width of the head the transform SERVES (David ruling
        /// 2026-09-12): the publisher's 8-bit conversion of the same 76
        /// `language_model.mtp.*` tensors, spliced into the 4-bit tower's
        /// tree. Same group size, same mode; only the packed columns differ.
        static let mtpHeadServedQuantizationBits = 8

        /// The hyper-connection stream width: the residual carries `hcCount`
        /// streams side by side the whole way down the tower.
        static var hyperConnectionWidth: Int { hiddenSize * hcCount }

        /// The gated deltanet's fused convolution width.
        static var linearConvDim: Int {
            linearKeyHeadDim * linearKeyHeads * 2
                + linearValueHeadDim * linearValueHeads
        }

        static func isFullAttention(layer index: Int) -> Bool {
            index % fullAttentionInterval == fullAttentionInterval - 1
        }

        static var pleLayerIndices: [Int] { pleLayerIds.map { $0 - 1 } }

        /// Rows one n-gram shard holds. Derived exactly as the model derives
        /// it: consecutive primes above the base, summed, padded to the
        /// divisor, split into shards.
        static var ngramRowsPerShard: Int {
            let heads = (ngramSize - 1) * headsPerNGram
            var total = 0
            for head in 0 ..< heads {
                total += nthPrimeAfter(ngramVocabSizeBase - 1, count: head + 1)
            }
            let padded = ((total + ngramVocabDivisor - 1) / ngramVocabDivisor)
                * ngramVocabDivisor
            return (padded + ngramShardCount - 1) / ngramShardCount
        }

        static func isPrime(_ value: Int) -> Bool {
            if value < 2 { return false }
            if value % 2 == 0 { return value == 2 }
            var d = 3
            while d * d <= value {
                if value % d == 0 { return false }
                d += 2
            }
            return true
        }

        static func nthPrimeAfter(_ start: Int, count: Int) -> Int {
            var p = start
            for _ in 0 ..< count {
                p += 1
                while !isPrime(p) { p += 1 }
            }
            return p
        }
    }

    /// 3,414 `language_model.*` tensors of the pinned 3,747: 3,335 under
    /// `language_model.model`, 76 under `language_model.mtp`, 3 for the head.
    /// The other 333 are `vision_tower.*` and the transform drops them.
    static let expectedTensorCount = 3_414
    /// Every decoder layer carries 61 tensors, whichever kind it is. The two
    /// kinds differ in WHICH 19 of the 61 they carry, not how many.
    static let expectedLinearLayerTensorCount = 61
    static let expectedFullAttentionLayerTensorCount = 61
    /// `embed_tokens` (3) + the tower's final hyper-connection mixer (7) +
    /// `lm_head` (3).
    static let expectedTopLevelTensorCount = 13
    /// The per-layer embedding block's own tensors, and its table.
    static let expectedPLETensorCount = 13
    static let expectedNGramShardTensorCount = 384
    /// The head embedded under `language_model.mtp.*`.
    static let expectedMTPTensorCount = 76
    /// UNIFORM. This target promotes nothing, so the override table is empty
    /// and a config that carries one is refused rather than half-read.
    static let expectedQuantizationOverrideCount = 0
    /// The head's quantized projections: every `language_model.mtp.*` stem
    /// that ships `.scales`. The EMITTED runtime config carries exactly one
    /// per-module entry for each of them, at the served width.
    static let expectedMTPQuantizedModuleCount = 22

    static let textTowerPrefix = "language_model."

    private static let modelPrefix = "language_model.model"
    private static let layerPrefix = "language_model.model.layers."

    /// Per-layer norms, all `[hidden_size]`. Gemma 4 carries seven, including
    /// the two `_1`/`_2` post-feedforward variants and the `_2`
    /// pre-feedforward one the Gemma 3 layout does not have.
    private static let layerNormSuffixes = [
        "input_layernorm", "post_attention_layernorm",
        "pre_feedforward_layernorm", "pre_feedforward_layernorm_2",
        "post_feedforward_layernorm", "post_feedforward_layernorm_1",
        "post_feedforward_layernorm_2",
    ]

    /// Parses the checkpoint's quantization block(s).
    ///
    /// The pinned artifact publishes the SAME spec twice, as `quantization`
    /// and `quantization_config`; `SwiftTransform.makeRuntimeConfigData`
    /// accepts either form and requires them to agree when both are present,
    /// so this validator applies exactly that policy rather than a stricter
    /// one -- a transform that emits a config must not reject the checkpoint
    /// that config came from.
    static func quantizationSpec(
        fromConfigRoot root: [String: Any]
    ) throws -> Qwen4ExpTransformQuantizationSpec {
        try uniformQuantizationSpec(
            fromConfigRoot: root,
            expectedBits: PinnedGeometry.quantizationBits,
            label: "Qwen 3.8 125B A6B"
        )
    }

    /// Parses the SERVED HEAD source's quantization block(s): the publisher's
    /// 8-bit conversion of the same checkpoint. Same two-block shape, same
    /// agreement rule, uniform, pinned to the served width.
    static func headSourceQuantizationSpec(
        fromConfigRoot root: [String: Any]
    ) throws -> Qwen4ExpTransformQuantizationSpec {
        try uniformQuantizationSpec(
            fromConfigRoot: root,
            expectedBits: PinnedGeometry.mtpHeadServedQuantizationBits,
            label: "Qwen 3.8 125B A6B MTP head source"
        )
    }

    private static func uniformQuantizationSpec(
        fromConfigRoot root: [String: Any],
        expectedBits: Int,
        label: String
    ) throws -> Qwen4ExpTransformQuantizationSpec {
        func parseBlock(
            _ key: String
        ) throws -> Qwen4ExpTransformQuantizationSpec? {
            guard let value = root[key], !(value is NSNull) else {
                return nil
            }
            guard let block = value as? [String: Any] else {
                throw MLXFastError.invalidInput(
                    "\(label) config \(key) must be an object"
                )
            }
            let scalarKeys: Set<String> = ["group_size", "bits", "mode"]
            guard block["group_size"] != nil, block["bits"] != nil,
                  block["mode"] != nil
            else {
                throw MLXFastError.invalidInput(
                    "\(label) config \(key) must explicitly define "
                        + "group_size, bits, and mode"
                )
            }
            let groupSize = try intField("group_size", in: block)
            let bits = try intField("bits", in: block)
            let mode = try stringField("mode", in: block)
            guard mode == PinnedGeometry.quantizationMode,
                  groupSize == PinnedGeometry.quantizationGroupSize,
                  bits == expectedBits
            else {
                throw MLXFastError.invalidInput(
                    "\(label) quantization must be affine \(expectedBits)-bit "
                        + "group_size \(PinnedGeometry.quantizationGroupSize)"
                )
            }

            // UNIFORM. Neither source promotes anything, so every non-scalar
            // key is a per-tensor override this transform would carry into a
            // config the runtime loader then ignores -- the runtime resolves
            // one width for every quantized path of a source. Refuse rather
            // than emit a block whose extra half nothing reads. (The served
            // head's own per-module entries are EMITTED, from the pinned
            // geometry, never read from a source.)
            let overrides: [String: Qwen4ExpTransformTensorQuantization] = [:]
            let extra = Set(block.keys).subtracting(scalarKeys).sorted()
            guard extra.isEmpty else {
                throw MLXFastError.invalidInput(
                    "\(label) config \(key) carries "
                        + "\(extra.count) per-tensor entr(ies) and this target "
                        + "is uniform, first: \(extra[0])"
                )
            }

            return Qwen4ExpTransformQuantizationSpec(
                groupSize: groupSize,
                bits: bits,
                mode: mode,
                overrides: overrides
            )
        }

        let quantization = try parseBlock("quantization")
        let quantizationConfig = try parseBlock("quantization_config")
        if let quantization, let quantizationConfig {
            guard quantization == quantizationConfig else {
                throw MLXFastError.invalidInput(
                    "\(label) config quantization and "
                        + "quantization_config must match exactly"
                )
            }
            return quantization
        }
        guard let spec = quantization ?? quantizationConfig else {
            throw MLXFastError.invalidInput(
                "\(label) config is missing both quantization and "
                    + "quantization_config"
            )
        }
        return spec
    }

    /// Exact metadata contract of the transformed text tower.
    ///
    /// DTYPES are the MLX affine-conversion convention rather than a reading
    /// of the pinned headers: packed codes `U32`, their scale/bias companions
    /// and every unquantized parameter `BF16` (the checkpoint's own `dtype` is
    /// `bfloat16`). Shapes and names ARE read off the pinned artifact. If the
    /// box session finds a different dtype on one of the small unquantized
    /// tensors, that is a repin of this table, not a relaxation of the check.
    ///
    /// This is the SOURCE contract: the 4-bit target as published, its own
    /// 4-bit copy of the embedded head included. The head the transform
    /// SERVES is `expectedHeadInventory(bits:)` at the served width.
    static func expectedTensorInventory() -> [String: ExpectedTensorMetadata] {
        var builder = InventoryBuilder()
        builder.addTower(bits: PinnedGeometry.quantizationBits)
        builder.addHead(bits: PinnedGeometry.quantizationBits)
        precondition(
            builder.inventory.count == expectedTensorCount,
            "Qwen 3.8 125B A6B inventory must contain \(expectedTensorCount) "
                + "tensors, built \(builder.inventory.count)"
        )
        return builder.inventory
    }

    /// The 76 tensors of the embedded head at one affine width. At
    /// `PinnedGeometry.quantizationBits` this is the copy the 4-bit target
    /// carries; at `PinnedGeometry.mtpHeadServedQuantizationBits` it is the
    /// head the transform splices in from the 8-bit source. Same names, same
    /// scale and bias shapes; only the packed columns scale with the width.
    static func expectedHeadInventory(bits: Int) -> [String: ExpectedTensorMetadata] {
        var builder = InventoryBuilder()
        builder.addHead(bits: bits)
        precondition(
            builder.inventory.count == expectedMTPTensorCount,
            "Qwen 3.8 125B A6B head inventory must contain "
                + "\(expectedMTPTensorCount) tensors, built \(builder.inventory.count)"
        )
        return builder.inventory
    }

    /// The head's quantized projections as checkpoint stems: every
    /// `language_model.mtp.*` tensor that ships `.scales`.
    static func headQuantizedModuleStems() -> [String] {
        expectedHeadInventory(bits: PinnedGeometry.mtpHeadServedQuantizationBits).keys
            .filter { $0.hasSuffix(".scales") }
            .map { String($0.dropLast(".scales".count)) }
            .sorted()
    }

    /// The per-module entries the emitted runtime `config.json` carries for
    /// the served head, keyed the way the runtime walks its module tree. The
    /// runtime drops the checkpoint's `language_model.` prefix at sanitize, so
    /// the key for `language_model.mtp.fc_hidden` is `mtp.fc_hidden`. Every
    /// entry is the pinned served geometry; nothing here is read from a file.
    static func runtimeHeadQuantizationOverrides() -> [String: [String: Any]] {
        var overrides: [String: [String: Any]] = [:]
        for stem in headQuantizedModuleStems() {
            precondition(
                stem.hasPrefix(textTowerPrefix),
                "Qwen 3.8 125B A6B head stem \(stem) is outside the text tower")
            let module = String(stem.dropFirst(textTowerPrefix.count))
            overrides[module] = [
                "group_size": PinnedGeometry.quantizationGroupSize,
                "bits": PinnedGeometry.mtpHeadServedQuantizationBits,
                "mode": PinnedGeometry.quantizationMode,
            ]
        }
        precondition(
            overrides.count == expectedMTPQuantizedModuleCount,
            "Qwen 3.8 125B A6B head must carry \(expectedMTPQuantizedModuleCount) "
                + "quantized modules, built \(overrides.count)")
        return overrides
    }

    /// Validate the served head's SOURCE: the shards `--head-source` names
    /// must carry every one of the 76 `language_model.mtp.*` tensors at the
    /// served width, each packed projection with its affine companions. The
    /// other tensors in those shards (the 8-bit tower tensors that share
    /// them, `lm_head` included) are ignored: the transform copies the head
    /// out of them and nothing else.
    static func validateHeadSource(
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader],
        quantization: Qwen4ExpTransformQuantizationSpec
    ) throws {
        guard quantization.bits == PinnedGeometry.mtpHeadServedQuantizationBits,
              quantization.groupSize == PinnedGeometry.quantizationGroupSize,
              quantization.mode == PinnedGeometry.quantizationMode,
              quantization.overrides.isEmpty
        else {
            throw MLXFastError.invalidInput(
                "Qwen 3.8 125B A6B MTP head source must be uniform affine "
                    + "\(PinnedGeometry.mtpHeadServedQuantizationBits)-bit group_size "
                    + "\(PinnedGeometry.quantizationGroupSize)"
            )
        }
        let expected = expectedHeadInventory(bits: quantization.bits)
        let expectedNames = Set(expected.keys)
        let mtpPrefix = "\(textTowerPrefix)mtp."
        let present = Set(index.weightMap.keys.filter { $0.hasPrefix(mtpPrefix) })
        let headerNameList = headers.values.flatMap { $0.tensors.keys }
        guard present == expectedNames,
              expectedNames.isSubset(of: Set(headerNameList)),
              headerNameList.count == Set(headerNameList).count
        else {
            let missing = expectedNames.subtracting(present).sorted()
            let extra = present.subtracting(expectedNames).sorted()
            throw MLXFastError.invalidInput(
                "Qwen 3.8 125B A6B MTP head source must carry exactly the "
                    + "\(expectedMTPTensorCount) embedded head tensors "
                    + "(missing: \(missing.prefix(8).joined(separator: ", ")); "
                    + "extra: \(extra.prefix(8).joined(separator: ", ")))"
            )
        }
        for name in expectedNames.sorted() {
            guard let expectedMetadata = expected[name] else {
                preconditionFailure(
                    "missing expected Qwen 3.8 125B A6B head metadata for \(name)")
            }
            let actual = try tensorInfo(named: name, index: index, headers: headers)
            guard actual.dtype == expectedMetadata.dtype,
                  actual.shape == expectedMetadata.shape
            else {
                throw MLXFastError.invalidInput(
                    "Qwen 3.8 125B A6B MTP head source tensor \(name) dtype/shape "
                        + "\(actual.dtype) \(actual.shape) does not match the served "
                        + "\(quantization.bits)-bit head \(expectedMetadata.dtype) "
                        + "\(expectedMetadata.shape)"
                )
            }
        }
        for stem in headQuantizedModuleStems() {
            try validateAffinePacking(
                stem: stem, index: index, headers: headers, spec: quantization.fallback)
        }
    }

    /// Builds the exact expected inventory. The tower and the head are built
    /// by the same code at a caller-chosen affine width, so the SOURCE table
    /// (tower and head at 4 bits) and the SERVED head table (8 bits) can never
    /// disagree on a name or a leading dimension.
    private struct InventoryBuilder {
        var inventory: [String: ExpectedTensorMetadata] = [:]
        private let hidden = PinnedGeometry.hiddenSize
        private let wide = PinnedGeometry.hyperConnectionWidth
        private let groupSize = PinnedGeometry.quantizationGroupSize

        mutating func add(_ name: String, _ dtype: TensorDType, _ shape: [Int]) {
            precondition(
                inventory[name] == nil,
                "duplicate expected Qwen 3.8 125B A6B tensor \(name)"
            )
            inventory[name] = ExpectedTensorMetadata(
                dtype: dtype.rawValue, shape: shape)
        }

        /// One affine-quantized projection: packed U32 codes plus the BF16
        /// scale and bias companions the affine scheme requires. `leading` is
        /// everything before the contracted axis, so the stacked expert
        /// tensors pass their experts axis through it.
        mutating func addAffine(
            _ stem: String, leading: [Int], inFeatures: Int, bits: Int
        ) {
            precondition(
                inFeatures.isMultiple(of: groupSize)
                    && (inFeatures * bits).isMultiple(of: 32),
                "Qwen 3.8 125B A6B tensor \(stem) is not packable at \(bits) bits"
            )
            add("\(stem).weight", .u32, leading + [inFeatures * bits / 32])
            add("\(stem).scales", .bf16, leading + [inFeatures / groupSize])
            add("\(stem).biases", .bf16, leading + [inFeatures / groupSize])
        }

        /// One hyper-connection mixer. The final mixer of a tower or of the
        /// embedded head has NO inject head.
        mutating func addHyperConnection(_ stem: String, inject: Bool, bits: Int) {
            add("\(stem).hc_norm.weight", .bf16, [wide])
            addAffine(
                "\(stem).input_mix_weight_down",
                leading: [PinnedGeometry.hcLowrank], inFeatures: wide, bits: bits)
            addAffine(
                "\(stem).input_mix_weight_up",
                leading: [wide], inFeatures: PinnedGeometry.hcLowrank, bits: bits)
            if inject {
                addAffine(
                    "\(stem).block_inject_weight",
                    leading: [PinnedGeometry.hcCount], inFeatures: wide, bits: bits)
            }
        }

        /// One decoder layer, of either kind. Both kinds carry the same two
        /// hyper-connection mixers and the same mixture-of-experts block.
        mutating func addDecoderLayer(_ prefix: String, isFullAttention: Bool, bits: Int) {
            let moeIntermediate = PinnedGeometry.moeIntermediateSize
            let sharedIntermediate = PinnedGeometry.sharedExpertIntermediateSize
            let experts = PinnedGeometry.expertCount

            addHyperConnection("\(prefix).attn_hyper_connection", inject: true, bits: bits)
            addHyperConnection("\(prefix).mlp_hyper_connection", inject: true, bits: bits)

            // The router stays in full precision: the checkpoint ships no
            // scales beside it.
            add("\(prefix).mlp.gate.weight", .bf16, [experts, hidden])
            for projection in ["gate_proj", "up_proj"] {
                addAffine(
                    "\(prefix).mlp.shared_expert.\(projection)",
                    leading: [sharedIntermediate], inFeatures: hidden, bits: bits)
                addAffine(
                    "\(prefix).mlp.switch_mlp.\(projection)",
                    leading: [experts, moeIntermediate], inFeatures: hidden, bits: bits)
            }
            addAffine(
                "\(prefix).mlp.shared_expert.down_proj",
                leading: [hidden], inFeatures: sharedIntermediate, bits: bits)
            addAffine(
                "\(prefix).mlp.switch_mlp.down_proj",
                leading: [experts, hidden], inFeatures: moeIntermediate, bits: bits)
            addAffine(
                "\(prefix).mlp.shared_expert_gate",
                leading: [1], inFeatures: hidden, bits: bits)

            if isFullAttention {
                let attentionDim =
                    PinnedGeometry.attentionHeads * PinnedGeometry.headDim
                let kvDim =
                    PinnedGeometry.keyValueHeads * PinnedGeometry.headDim
                let indexerDim =
                    (PinnedGeometry.indexerHeads
                        + PinnedGeometry.indexerKeyValueHeads)
                    * PinnedGeometry.indexerHeadDim
                for norm in ["q_norm", "k_norm"] {
                    add(
                        "\(prefix).self_attn.\(norm).weight", .bf16,
                        [PinnedGeometry.headDim])
                }
                // `q_proj` carries the output gate as well, hence the doubled
                // width.
                addAffine(
                    "\(prefix).self_attn.q_proj",
                    leading: [attentionDim * 2], inFeatures: hidden, bits: bits)
                for projection in ["k_proj", "v_proj"] {
                    addAffine(
                        "\(prefix).self_attn.\(projection)",
                        leading: [kvDim], inFeatures: hidden, bits: bits)
                }
                addAffine(
                    "\(prefix).self_attn.o_proj",
                    leading: [hidden], inFeatures: attentionDim, bits: bits)
                addAffine(
                    "\(prefix).self_attn.indexer.index_qk_proj",
                    leading: [indexerDim], inFeatures: hidden, bits: bits)
                for norm in ["q_layernorm", "k_layernorm"] {
                    add(
                        "\(prefix).self_attn.indexer.\(norm).weight", .bf16,
                        [PinnedGeometry.indexerHeadDim])
                }
            } else {
                let valueDim =
                    PinnedGeometry.linearValueHeadDim
                    * PinnedGeometry.linearValueHeads
                add(
                    "\(prefix).linear_attn.A_log", .bf16,
                    [PinnedGeometry.linearValueHeads])
                add(
                    "\(prefix).linear_attn.dt_bias", .bf16,
                    [PinnedGeometry.linearValueHeads])
                // Depthwise kernel in MLX layout: (channels, kernel, 1).
                add(
                    "\(prefix).linear_attn.conv1d.weight", .bf16,
                    [
                        PinnedGeometry.linearConvDim,
                        PinnedGeometry.linearConvKernelDim, 1,
                    ])
                add(
                    "\(prefix).linear_attn.norm.weight", .bf16,
                    [PinnedGeometry.linearValueHeadDim])
                addAffine(
                    "\(prefix).linear_attn.in_proj_qkv",
                    leading: [PinnedGeometry.linearConvDim], inFeatures: hidden, bits: bits)
                addAffine(
                    "\(prefix).linear_attn.in_proj_z",
                    leading: [valueDim], inFeatures: hidden, bits: bits)
                for projection in ["in_proj_a", "in_proj_b"] {
                    addAffine(
                        "\(prefix).linear_attn.\(projection)",
                        leading: [PinnedGeometry.linearValueHeads],
                        inFeatures: hidden, bits: bits)
                }
                addAffine(
                    "\(prefix).linear_attn.out_proj",
                    leading: [hidden], inFeatures: valueDim, bits: bits)
            }
        }

        /// The tower: the embedding, the untied output head, the final
        /// mixer, the 48 layers, and the per-layer embedding block with its
        /// sharded table.
        mutating func addTower(bits: Int) {
            let vocab = PinnedGeometry.vocabSize
            addAffine(
                "\(Qwen4ExpCheckpointValidation.modelPrefix).embed_tokens", leading: [vocab], inFeatures: hidden, bits: bits)
            addAffine(
                "\(Qwen4ExpCheckpointValidation.textTowerPrefix)lm_head", leading: [vocab], inFeatures: hidden, bits: bits)
            // THERE IS NO `model.norm`: this mixer stands in for it.
            addHyperConnection(
                "\(Qwen4ExpCheckpointValidation.modelPrefix).hyper_connection_mixer", inject: false, bits: bits)

            for layerIndex in 0 ..< PinnedGeometry.layerCount {
                addDecoderLayer(
                    "\(Qwen4ExpCheckpointValidation.layerPrefix)\(layerIndex)",
                    isFullAttention: PinnedGeometry.isFullAttention(layer: layerIndex),
                    bits: bits)
            }

            // The per-layer embedding block and its sharded table.
            let ngramHeads =
                (PinnedGeometry.ngramSize - 1) * PinnedGeometry.headsPerNGram
            let rowDimensions = PinnedGeometry.pleEmbedDim / ngramHeads
            let rowsPerShard = PinnedGeometry.ngramRowsPerShard
            for layerIndex in PinnedGeometry.pleLayerIndices {
                let prefix = "\(Qwen4ExpCheckpointValidation.layerPrefix)\(layerIndex).ple"
                add(
                    "\(prefix).conv1d.weight", .bf16,
                    [wide, PinnedGeometry.pleConvKernelSize, 1])
                for norm in ["norm_conv", "norm_key", "norm_query"] {
                    add("\(prefix).\(norm).weight", .bf16, [wide])
                }
                addAffine(
                    "\(prefix).key_proj", leading: [wide],
                    inFeatures: PinnedGeometry.pleEmbedDim, bits: bits)
                addAffine(
                    "\(prefix).value_proj", leading: [hidden],
                    inFeatures: PinnedGeometry.pleEmbedDim, bits: bits)
                add(
                    "\(prefix).ple_embedding.layer_multipliers", .i64,
                    [PinnedGeometry.ngramSize])
                for buffer in ["ngram_heads_offsets", "ngram_heads_vocab_sizes"] {
                    add("\(prefix).ple_embedding.\(buffer)", .i64, [ngramHeads])
                }
                for shard in 0 ..< PinnedGeometry.ngramShardCount {
                    addAffine(
                        "\(prefix).ple_embedding.ngram_embedding.shard_\(shard)",
                        leading: [rowsPerShard], inFeatures: rowDimensions, bits: bits)
                }
            }
        }

        /// The head EMBEDDED in this checkpoint. It has no embedding table and
        /// no head of its own; it rides the target's.
        mutating func addHead(bits: Int) {
            let mtpPrefix = "\(Qwen4ExpCheckpointValidation.textTowerPrefix)mtp"
            add("\(mtpPrefix).pre_fc_norm_embedding.weight", .bf16, [hidden])
            add("\(mtpPrefix).pre_fc_norm_hidden.weight", .bf16, [wide])
            for projection in ["fc_embedding", "fc_hidden"] {
                addAffine(
                    "\(mtpPrefix).\(projection)", leading: [hidden],
                    inFeatures: hidden, bits: bits)
            }
            addHyperConnection("\(mtpPrefix).hyper_connection_mixer", inject: false, bits: bits)
            for layerIndex in 0 ..< PinnedGeometry.mtpLayerCount {
                addDecoderLayer(
                    "\(mtpPrefix).layers.\(layerIndex)", isFullAttention: true, bits: bits)
            }
        }
    }

    static func validateSelectedTensors(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader],
        quantization: Qwen4ExpTransformQuantizationSpec
    ) throws {
        // Affine quantization legitimately ships `.biases`, so unlike the
        // Laguna NVFP4 contract that suffix is NOT forbidden here.
        let forbiddenSuffixes = [
            ".weight_packed",
            ".input_global_scale",
            ".weight_global_scale",
            ".k_scale",
            ".v_scale",
        ]
        if let forbiddenName = selectedKeys.sorted().first(where: { name in
            forbiddenSuffixes.contains { suffix in name.hasSuffix(suffix) }
        }) {
            throw MLXFastError.invalidInput(
                "Qwen 3.8 125B A6B MLX transform rejects "
                    + "compressed-tensors/global-scale and FP8 KV-scale tensor "
                    + "\(forbiddenName)"
            )
        }
        // The head is UNTIED on this target: `language_model.lm_head.*` is a
        // real triple the transform MUST select, unlike the tied tower this
        // tree was seeded from where selecting one was the error.
        for component in ["weight", "scales", "biases"] {
            let name = "\(textTowerPrefix)lm_head.\(component)"
            guard selectedKeys.contains(name) else {
                throw MLXFastError.invalidInput(
                    "Qwen 3.8 125B A6B has an untied output head and must "
                        + "select \(name)"
                )
            }
        }
        // The multi-token-prediction head is EMBEDDED in this checkpoint, so
        // the transform MUST select it: there is no separate head artifact,
        // and a tree without `language_model.mtp.*` has no speculative arm at
        // all. This is the opposite of the tied-head tower this tree was
        // seeded from, where an mtp tensor meant the wrong checkpoint.
        let mtpSelected = selectedKeys.filter { name in
            name.hasPrefix("\(textTowerPrefix)mtp.")
        }
        guard mtpSelected.count == expectedMTPTensorCount else {
            throw MLXFastError.invalidInput(
                "Qwen 3.8 125B A6B transform selected \(mtpSelected.count) "
                    + "embedded multi-token-prediction tensors, expected "
                    + "\(expectedMTPTensorCount)"
            )
        }

        for name in selectedKeys.sorted() where name.hasSuffix(".weight") {
            let stem = String(name.dropLast(".weight".count))
            let scalesName = "\(stem).scales"
            let biasesName = "\(stem).biases"
            guard selectedKeys.contains(scalesName)
                    || selectedKeys.contains(biasesName)
            else {
                continue
            }
            guard selectedKeys.contains(scalesName),
                  selectedKeys.contains(biasesName)
            else {
                throw MLXFastError.invalidInput(
                    "Qwen 3.8 125B A6B affine projection \(stem) must ship both "
                        + ".scales and .biases"
                )
            }
            // THE PER-PATH LOOKUP, and it is the point of this whole file. A
            // single global width would accept a 4-bit-packed tensor the
            // config declares at 8 bits and vice versa -- right names, right
            // leading dimensions, wrong numerics.
            try validateAffinePacking(
                stem: stem, index: index, headers: headers,
                spec: quantization.spec(forPath: stem))
        }

        try validateExactPublicInventory(
            selectedKeys: selectedKeys,
            index: index,
            headers: headers
        )
    }

    /// One affine projection's packed geometry against the width its path
    /// resolves to: U32 codes, BF16 scale and bias companions of one shape,
    /// and a packed column count that is exactly `in_features * bits / 32`
    /// for the group count the scales carry.
    static func validateAffinePacking(
        stem: String,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader],
        spec pathSpec: Qwen4ExpTransformTensorQuantization
    ) throws {
        let weightInfo = try tensorInfo(
            named: "\(stem).weight", index: index, headers: headers)
        let scalesInfo = try tensorInfo(
            named: "\(stem).scales", index: index, headers: headers)
        let biasesInfo = try tensorInfo(
            named: "\(stem).biases", index: index, headers: headers)
        // Rank is >= 2 rather than == 2: the stacked SwitchGLU expert
        // tensors are rank 3 (experts x out x packed-in).
        guard weightInfo.dtype == TensorDType.u32.rawValue,
              scalesInfo.dtype == TensorDType.bf16.rawValue,
              biasesInfo.dtype == TensorDType.bf16.rawValue,
              weightInfo.shape.count >= 2,
              scalesInfo.shape == biasesInfo.shape,
              scalesInfo.shape.count == weightInfo.shape.count,
              weightInfo.shape.dropLast() == scalesInfo.shape.dropLast(),
              weightInfo.shape.allSatisfy({ $0 > 0 }),
              scalesInfo.shape.allSatisfy({ $0 > 0 })
        else {
            throw MLXFastError.invalidInput(
                "Qwen 3.8 125B A6B affine projection \(stem) has "
                    + "incompatible weight, scale, or bias metadata"
            )
        }

        guard let packedWidth = weightInfo.shape.last,
              let groupCount = scalesInfo.shape.last
        else {
            throw MLXFastError.invalidInput(
                "Qwen 3.8 125B A6B affine projection \(stem) has no packed "
                    + "axis"
            )
        }
        let (inputFeatures, inputOverflow) =
            groupCount.multipliedReportingOverflow(by: pathSpec.groupSize)
        let (packedBits, packedOverflow) =
            packedWidth.multipliedReportingOverflow(by: 32)
        guard !inputOverflow, !packedOverflow else {
            throw MLXFastError.invalidInput(
                "Qwen 3.8 125B A6B projection \(stem) packed width overflows "
                    + "Int"
            )
        }
        let (expectedPackedBits, expectedOverflow) =
            inputFeatures.multipliedReportingOverflow(by: pathSpec.bits)
        guard !expectedOverflow, packedBits == expectedPackedBits else {
            throw MLXFastError.invalidInput(
                "quantized Qwen 3.8 125B A6B projection \(stem) stored width "
                    + "\(packedWidth) does not match config quantization "
                    + "group_size \(pathSpec.groupSize) bits "
                    + "\(pathSpec.bits) for input dimension "
                    + "\(inputFeatures)"
            )
        }
    }

    static func validateExactPublicInventory(
        selectedKeys: Set<String>,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws {
        let expected = expectedTensorInventory()
        let expectedNames = Set(expected.keys)
        // The public checkpoint also carries the `vision_tower.*` and
        // `embed_vision.*` namespaces (358 tensors), which the transform never
        // selects, so -- unlike the Laguna validator -- the index and header
        // name sets are supersets rather than equal. Constrain them where it
        // matters: every selected name must be indexed exactly once and
        // present in exactly one header.
        let headerNameList = headers.values.flatMap { $0.tensors.keys }
        let headerNames = Set(headerNameList)

        guard selectedKeys == expectedNames,
              expectedNames.isSubset(of: Set(index.weightMap.keys)),
              expectedNames.isSubset(of: headerNames),
              headerNameList.count == headerNames.count
        else {
            let missing = expectedNames.subtracting(selectedKeys).sorted()
            let extra = selectedKeys.subtracting(expectedNames).sorted()
            let unindexed = expectedNames
                .subtracting(Set(index.weightMap.keys)).sorted()
            throw MLXFastError.invalidInput(
                "Qwen 3.8 125B A6B checkpoint tensor inventory must match the "
                    + "exact public \(expectedTensorCount)-tensor contract "
                    + "(missing: \(missing.prefix(8).joined(separator: ", ")); "
                    + "extra: \(extra.prefix(8).joined(separator: ", ")); "
                    + "unindexed/duplicate header tensors: "
                    + "\(unindexed.prefix(8).joined(separator: ", ")))"
            )
        }

        for name in expected.keys.sorted() {
            guard let expectedMetadata = expected[name] else {
                preconditionFailure(
                    "missing expected Qwen 3.8 125B A6B metadata for \(name)")
            }
            let actual = try tensorInfo(
                named: name, index: index, headers: headers)
            guard actual.dtype == expectedMetadata.dtype,
                  actual.shape == expectedMetadata.shape
            else {
                throw MLXFastError.invalidInput(
                    "Qwen 3.8 125B A6B tensor \(name) dtype/shape "
                        + "\(actual.dtype) \(actual.shape) does not match "
                        + "exact public metadata \(expectedMetadata.dtype) "
                        + "\(expectedMetadata.shape)"
                )
            }
        }
    }

    private static func tensorInfo(
        named name: String,
        index: CheckpointIndex,
        headers: [String: SafetensorsHeader]
    ) throws -> SafetensorInfo {
        guard let shardName = index.weightMap[name],
              let info = headers[shardName]?.tensors[name]
        else {
            throw MLXFastError.invalidInput(
                "missing validated tensor metadata for \(name)")
        }
        return info
    }

    // MARK: - The norm convention

    /// Runtime `config.json` key carrying the additive offset every non-gated
    /// `Qwen4ExpRMSNorm` applies to its stored weight.
    ///
    /// NO PUBLISHED CHECKPOINT DECLARES IT. The pinned tree bakes the offset
    /// (its non-gated norm tensors hold `1 + w`), the reference implementation
    /// assumes zero-centered weights and computes `y * (1 + w)`, and the two
    /// differ by a whole-tower scale error rather than a numeric detail. The
    /// transform therefore WRITES the key, so a consumer that reads only the
    /// file gets the convention with the weights.
    static let rmsNormWeightOffsetKey = "rms_norm_weight_offset"

    /// Refuse a runtime config that does not declare this track's pinned norm
    /// convention.
    ///
    /// FAIL CLOSED, like the digest checks: a MISSING key is a refusal, not a
    /// default. The runtime's own decode falls back to the zero-centered `1`
    /// when the key is absent, so accepting a tree without it would hand a
    /// silently wrong tower to every consumer that trusts this verifier. The
    /// expected value is derived from `MLXFastConstants.rmsNormConvention`, so
    /// a repin of the convention moves the requirement with it.
    static func validateNormConvention(
        inRuntimeConfigRoot root: [String: Any]
    ) throws {
        let expected = Double(MLXFastConstants.rmsNormConvention.weightOffset)
        guard let value = root[rmsNormWeightOffsetKey], !(value is NSNull) else {
            throw MLXFastError.invalidInput(
                "Qwen 3.8 125B A6B runtime config must declare "
                    + "\(rmsNormWeightOffsetKey) = \(expected); a tree without "
                    + "it is refused, because the runtime would silently fall "
                    + "back to the zero-centered convention this checkpoint "
                    + "does not use. Re-run the transform to write the key."
            )
        }
        guard let number = value as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue == expected
        else {
            throw MLXFastError.invalidInput(
                "Qwen 3.8 125B A6B runtime config \(rmsNormWeightOffsetKey) "
                    + "must be the pinned \(expected), found \(value)"
            )
        }
    }

    private static func intField(
        _ key: String, in object: [String: Any]
    ) throws -> Int {
        guard let number = object[key] as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID(),
              !CFNumberIsFloatType(number),
              let integer = Int(number.stringValue)
        else {
            throw MLXFastError.invalidInput(
                "Qwen 3.8 125B A6B quantization field \(key) must be a finite "
                    + "integer in Int range"
            )
        }
        return integer
    }

    private static func stringField(
        _ key: String, in object: [String: Any]
    ) throws -> String {
        guard let string = object[key] as? String else {
            throw MLXFastError.invalidInput(
                "Qwen 3.8 125B A6B quantization field \(key) must be a string"
            )
        }
        return string
    }
}
