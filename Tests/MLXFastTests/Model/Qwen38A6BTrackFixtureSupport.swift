import Foundation
import MLXFastCore
import Testing

/// Fixture locations and pins for the CURRENT track target,
/// `qwen3.8-125b-a6b-mlx-v1`.
///
/// Kept apart from `Qwen4ExpArtifactFixtureSupport.swift`, which still names
/// the Gemma 4 26B A4B checkpoint the retained gemma model code validates
/// against. These two files disagree on purpose while the model port is
/// outstanding (`docs/qwen38-125b-a6b-port-notes.md`).
let qwen38A6BRepository = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"
let qwen38A6BRevision = "327c8a604de613b42f84ba5e6b796c0931e8aa3b"

/// SHA256 of the checkpoint's own `config.json` bytes exactly as published at
/// the pinned revision, fetched 2026-08-27 from
/// `https://huggingface.co/Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP/resolve/327c8a604de613b42f84ba5e6b796c0931e8aa3b/config.json`.
/// This is NOT the digest of the normalized
/// `fixtures/qwen3_8_125b_a6b_config.json` re-render, which differs in
/// whitespace and key order only (2-space indent, keys sorted, trailing
/// newline).
///
/// THIS IS A LAPTOP-SIDE PUBLIC FETCH, NOT A BOX-VERIFIED ARTIFACT. It proves
/// the fixture matches what the pinned revision publishes; it does not prove
/// it matches the byte the ranked box's transform reads.
let qwen38A6BConfigSHA256 =
    "4efd5f41c47e8de9f55c678df4a237cbbc78e4d517dad42a906889f7d94f79a2"

// Tests/MLXFastTests/Model/<this file> -> repository root is four levels up.
private let qwen38A6BRepositoryRoot = URL(fileURLWithPath: #filePath)
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()

let qwen38A6BConfigFixtureURL = qwen38A6BRepositoryRoot
    .appendingPathComponent("fixtures/qwen3_8_125b_a6b_config.json")

let qwen38A6BTrackContractURL = qwen38A6BRepositoryRoot
    .appendingPathComponent("fixtures/qwen3_8_125b_a6b_track.json")

let qwen38A6BTensorInventoryURL = qwen38A6BRepositoryRoot
    .appendingPathComponent("fixtures/qwen3_8_125b_a6b_tensor_inventory.json")

let qwen38A6BReferenceManifestURL = qwen38A6BRepositoryRoot
    .appendingPathComponent("fixtures/reference_qwen3_8_125b_a6b_4bit.sha256")

/// The served MTP head's source (David ruling 2026-09-12): the publisher's
/// 8-bit conversion of the same checkpoint, two shards of it.
let qwen38A6BMTPHeadSourceRepository = "Vontra/Qwen3.8-Flash-Next-MLX-8bit-MTP"
let qwen38A6BMTPHeadSourceRevision = "9c306179562765396e197a8a7a5de1b6b761c41a"

let qwen38A6BMTPHeadManifestURL = qwen38A6BRepositoryRoot
    .appendingPathComponent("fixtures/reference_qwen3_8_125b_a6b_mtp_8bit.sha256")

let qwen38A6BMTPHeadInventoryURL = qwen38A6BRepositoryRoot
    .appendingPathComponent("fixtures/qwen3_8_125b_a6b_mtp_8bit_inventory.json")

func qwen38A6BMTPHeadInventoryObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: qwen38A6BMTPHeadInventoryURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Qwen 3.8 125B A6B MTP head inventory fixture must be a JSON object"
        )
    }
    return object
}

/// The `<sha256> <bytes> <path>` records of a reference manifest, comment
/// and blank lines skipped.
func qwen38A6BManifestRecords(_ url: URL) throws -> [(sha256: String, bytes: Int, path: String)] {
    let text = try String(contentsOf: url, encoding: .utf8)
    var records: [(sha256: String, bytes: Int, path: String)] = []
    for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
        if line.hasPrefix("#") { continue }
        let fields = line.split(separator: " ", omittingEmptySubsequences: true)
        guard fields.count == 3, let bytes = Int(fields[1]) else {
            throw MLXFastError.invalidInput("malformed manifest record: \(line)")
        }
        records.append((String(fields[0]), bytes, String(fields[2])))
    }
    return records
}

func qwen38A6BTrackContractObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: qwen38A6BTrackContractURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Qwen 3.8 125B A6B track contract fixture must be a JSON object"
        )
    }
    return object
}

func qwen38A6BConfigObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: qwen38A6BConfigFixtureURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Qwen 3.8 125B A6B config fixture must be a JSON object"
        )
    }
    return object
}

func qwen38A6BTensorInventoryObject() throws -> [String: Any] {
    guard let object = try JSONSerialization.jsonObject(
        with: try Data(contentsOf: qwen38A6BTensorInventoryURL)
    ) as? [String: Any] else {
        throw MLXFastError.invalidInput(
            "Qwen 3.8 125B A6B tensor inventory fixture must be a JSON object"
        )
    }
    return object
}
