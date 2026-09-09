//
//  Qwen4ExpMTPHeadAdoptionTests.swift
//
//  The head adoption seam, over a TINY head and no checkpoint.
//
//  `TrackQwen4ExpRunner.adoptMTPHead` rebuilds the checkpoint's `mtp.*` head
//  as this repository's own module. Its default must change nothing: the
//  served head has to be bit-exact with the head the pinned fork builds. This
//  suite builds a fork head over a small configuration, quantizes it the way
//  the loader quantizes a checkpoint that carries scales, adopts it, and
//  compares both the parameter tree and the head's output.
//
//  GATED, and the gate is the one the rest of this package uses. The head runs
//  a full-attention layer with the QSA indexer and a mixture-of-experts block,
//  so the comparison is a real MLX forward pass. Hosted CI has no usable Metal
//  runtime for it: the suite runs on a box. See docs/ci-coverage.md.
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import Testing
import TrackRunner

private let mlxRuntimeTestsEnabled =
    ProcessInfo.processInfo.environment["MLXFAST_RUN_MLX_RUNTIME_TESTS"] == "1"

/// A head small enough to build in a test, shaped so every quantized
/// projection has an input width the group size divides.
private func tinyTextConfiguration() throws -> Qwen4ExpTextConfiguration {
    let json = """
        {
          "model_type": "qwen4_exp_text",
          "hidden_size": 64,
          "num_hidden_layers": 4,
          "num_attention_heads": 2,
          "num_key_value_heads": 1,
          "head_dim": 32,
          "vocab_size": 64,
          "rms_norm_eps": 1e-6,
          "rms_norm_weight_offset": 0,
          "full_attention_interval": 4,
          "num_experts": 2,
          "num_experts_per_tok": 1,
          "moe_intermediate_size": 64,
          "shared_expert_intermediate_size": 64,
          "hc_count": 2,
          "hc_lowrank": 64,
          "indexer_n_heads": 1,
          "indexer_kv_heads": 1,
          "indexer_head_dim": 64,
          "indexer_budget": 16,
          "indexer_compress_ratio": 4,
          "ple_layer_ids": [],
          "partial_rotary_factor": 0.25,
          "rope_theta": 10000,
          "max_position_embeddings": 256,
          "tie_word_embeddings": false
        }
        """
    return try JSONDecoder().decode(
        Qwen4ExpTextConfiguration.self, from: Data(json.utf8))
}

/// The group size and bit width this test's "checkpoint" carries. The value
/// is the test's, not the track's: what the test proves is that adoption
/// REPRODUCES whatever geometry the loaded head has.
private let checkpointGroupSize = 32
private let checkpointBits = 4

@Test(.enabled(if: mlxRuntimeTestsEnabled))
func adoptedMTPHeadCarriesTheLoadedHeadsParametersExactly() throws {
    MLXRandom.seed(1234)
    let configuration = try tinyTextConfiguration()

    // The head the pinned fork builds, quantized the way
    // MLXLMCommon/Load.swift quantizes a checkpoint that carries scales.
    let loaded = Qwen4ExpMTPModule(configuration)
    quantize(model: loaded, groupSize: checkpointGroupSize, bits: checkpointBits)

    let adopted = try TrackQwen4ExpRunner.adoptMTPHead(
        loaded, configuration: configuration)

    let expected = loaded.parameters().flattened().sorted { $0.0 < $1.0 }
    let actual = adopted.parameters().flattened().sorted { $0.0 < $1.0 }
    #expect(expected.map(\.0) == actual.map(\.0))
    #expect(!expected.isEmpty)

    for (left, right) in zip(expected, actual) {
        #expect(left.1.shape == right.1.shape, "\(left.0) shape")
        #expect(left.1.dtype == right.1.dtype, "\(left.0) dtype")
        #expect(MLX.all(left.1 .== right.1).item(Bool.self), "\(left.0) values")
    }

    // The geometry travelled with the tensors: every projection the loaded
    // head quantized is quantized in the adopted head, with the same numbers.
    let loadedGeometry = quantizationGeometry(of: loaded)
    let adoptedGeometry = quantizationGeometry(of: adopted)
    #expect(!loadedGeometry.isEmpty)
    #expect(loadedGeometry == adoptedGeometry)
    // The whole head, not only its two front projections: the mixture-of-
    // experts stack and the attention stack are quantized leaves too.
    #expect(loadedGeometry["layers.0.mlp.switch_mlp.gate_proj"] != nil)
    #expect(loadedGeometry["layers.0.self_attn.q_proj"] != nil)
    #expect(loadedGeometry["layers.0.self_attn.indexer.index_qk_proj"] != nil)
}

@Test(.enabled(if: mlxRuntimeTestsEnabled))
func adoptedMTPHeadDraftsTheSameSampleAndMultiStream() throws {
    MLXRandom.seed(1234)
    let configuration = try tinyTextConfiguration()

    let loaded = Qwen4ExpMTPModule(configuration)
    quantize(model: loaded, groupSize: checkpointGroupSize, bits: checkpointBits)
    let adopted = try TrackQwen4ExpRunner.adoptMTPHead(
        loaded, configuration: configuration)

    let embedTokens = Embedding(
        embeddingCount: configuration.vocabularySize,
        dimensions: configuration.hiddenSize)
    let sequence = 4
    let ids = MLXRandom.randInt(
        0 ..< configuration.vocabularySize, [1, sequence]).asType(.int32)
    let multiStream = MLXRandom.normal(
        [1, sequence, configuration.hcCount * configuration.hiddenSize])

    let fromFork = loaded(
        nextTokenIds: ids, multiStream: multiStream, embedTokens: embedTokens,
        cache: loaded.makeCache())
    let fromTrack = adopted(
        nextTokenIds: ids, multiStream: multiStream, embedTokens: embedTokens,
        cache: adopted.makeCache())

    #expect(fromFork.sample.shape == fromTrack.sample.shape)
    #expect(fromFork.multi.shape == fromTrack.multi.shape)
    #expect(MLX.all(fromFork.sample .== fromTrack.sample).item(Bool.self))
    #expect(MLX.all(fromFork.multi .== fromTrack.multi).item(Bool.self))
}

/// Group size, bit width and mode of every quantized leaf, keyed by path.
private func quantizationGeometry(of module: Module) -> [String: String] {
    var geometry: [String: String] = [:]
    for (path, leaf) in module.leafModules().flattened() {
        guard let quantized = leaf as? Quantized else { continue }
        geometry[path] = "\(quantized.groupSize)/\(quantized.bits)/\(quantized.mode)"
    }
    return geometry
}
