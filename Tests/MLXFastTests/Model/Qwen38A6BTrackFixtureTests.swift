import Foundation
import MLXFastCore
@testable import MLXFastHarness
@testable import MLXFastTransform
import Testing

// Contract tests for `fixtures/qwen3_8_125b_a6b_track.json`.
//
// The track is ARMED (David 2026-08-31, "mlx is fine proceed"). Scoring is
// promoted in its own change: `official_scoring_enabled` is true, the timed
// pool and the hidden oracle carry real {r2_path, sha256, bytes} pins for the
// 8 track goldens, `live_golden` names botany, and a `runner` block registers
// the ranked pipeline. This suite asserts that armed contract and that no
// pending sentinel survives. The atomic bench half sets OFFICIAL_BASELINE_MLX +
// the acceptance bands and co-merges (the flag alone leaves benchd's
// official_baseline()? refusing while the constant is None).
//
// This is a laptop-side JSON-shape test only. It does not exercise benchd's
// Rust contract parsers; those live in the bench repository.

@Suite("Qwen 3.8 125B A6B track contract fixture")
struct Qwen38A6BTrackFixtureTests {

    /// The sentinel every unarmed organizer slot in this fixture carries.
    /// EXACT-MATCH ONLY -- never a prefix check.
    static let pendingOrganizerSentinel = "QWEN38-125B-A6B-MLX-PENDING-ORGANIZER"

    @Test("fixture parses as a JSON object")
    func fixtureParses() throws {
        let object = try qwen38A6BTrackContractObject()
        #expect(object["schema_version"] as? Int == 1)
    }

    @Test("track_id is pinned to the leaderboard / R2-prefix identity")
    func trackIdIsPinned() throws {
        let object = try qwen38A6BTrackContractObject()
        #expect(object["track_id"] as? String == "qwen3.8-125b-a6b-mlx-v1")
    }

    /// TWO NAMES, AND THEY ARE NOT THE SAME NAME.
    ///
    /// David ruling 2026-08-27, relayed by orchestrator: the MLX and CUDA
    /// tracks of this model share ONE benchmarker, so they share ONE bench
    /// release branch and ONE dist channel, `qwen3.8-125b-a6b-v1`. The TRACK
    /// id stays platform-specific and is what names the leaderboard namespace,
    /// the runner label set and the R2 object prefix.
    ///
    /// The failure this guards is a merge in the wrong direction: dropping
    /// `-mlx` from the track id would pool two platforms' results on one
    /// board, and adding it to the channel would split one benchmarker into
    /// two. Both edits look like a tidy-up. So the two spellings are asserted
    /// TOGETHER, in one test, with the reason attached.
    @Test("the bench channel names the project, the track id names the platform")
    func benchChannelAndTrackIdAreDifferentNames() throws {
        let object = try qwen38A6BTrackContractObject()
        #expect(object["track_id"] as? String == "qwen3.8-125b-a6b-mlx-v1")

        let script = try String(
            contentsOfFile: "tools/fetch-benchd.sh", encoding: .utf8)
        #expect(
            script.contains("BRANCH=\"${BENCHD_BRANCH:-qwen3.8-125b-a6b-v1}\""),
            "tools/fetch-benchd.sh must default the channel branch to the PROJECT name")
        #expect(
            !script.contains("BENCHD_BRANCH:-qwen3.8-125b-a6b-mlx-v1"),
            "the channel branch must not be the platform track id")
    }

    @Test("benchmark_name matches the Yukon manifest name")
    func benchmarkNameMatchesManifest() throws {
        let object = try qwen38A6BTrackContractObject()
        #expect(object["benchmark_name"] as? String == "mlxfast-qwen38-125b-a6b")
    }

    @Test("track_id is substring-clean against every retired MTP name")
    func trackIdIsCleanAgainstRetiredNames() throws {
        let object = try qwen38A6BTrackContractObject()
        let trackId = try #require(object["track_id"] as? String)
        let retired = [
            "MLXFAST_MTP_",
            "mtp-ranked",
            "measure-mtp-job",
            "mtp-weights",
            "laguna-xs-2.1-mtp",
            "gemma4-31b-it",
        ]
        for name in retired {
            #expect(!trackId.contains(name), "track_id must not substring-collide with retired name \(name)")
            #expect(!name.contains(trackId), "retired name \(name) must not substring-collide with track_id")
        }
    }

    /// ARMED. `official_scoring_enabled` flips true LAST, in its own change --
    /// this one -- once the baseline pair is captured. The atomic bench half
    /// arms OFFICIAL_BASELINE_MLX; the two co-merge, and the flag alone would
    /// leave benchd's official_baseline()? refusing while the constant is None.
    @Test("official_scoring_enabled is true — the track is armed")
    func officialScoringEnabled() throws {
        let object = try qwen38A6BTrackContractObject()
        #expect(object["official_scoring_enabled"] as? Bool == true)
    }

    /// ARMED runner registration: the ranked pipeline's provider and workflow,
    /// and the single-flight concurrency cap (one ranked run at a time).
    @Test("runner registers the github-actions benchmark.yml workflow, single-flight")
    func runnerRegistersSingleFlightWorkflow() throws {
        let object = try qwen38A6BTrackContractObject()
        let runner = try #require(object["runner"] as? [String: Any])
        #expect(runner["provider"] as? String == "github-actions")
        #expect(runner["workflow"] as? String == "benchmark.yml")
        #expect(runner["maxConcurrentWorkflows"] as? Int == 1)
    }

    /// `live_golden` names the pooled prompt that also serves as the hidden
    /// correctness oracle. It must be one of the pooled goldens (botany).
    @Test("live_golden names a pooled golden (botany)")
    func liveGoldenNamesAPooledGolden() throws {
        let object = try qwen38A6BTrackContractObject()
        let live = try #require(object["live_golden"] as? String)
        #expect(live == "botany")
        let pool = try #require(object["timed_prompt_pool"] as? [[String: Any]])
        let names = pool.compactMap {
            ($0["r2_path"] as? String)?
                .components(separatedBy: "/").last?
                .replacingOccurrences(of: ".golden.json", with: "")
        }
        #expect(names.contains(live))
    }

    /// Once armed, NO pending-organizer sentinel may survive anywhere in the
    /// fixture -- not in a pool entry, not in the hidden oracle, not in a value
    /// a future edit might reintroduce.
    @Test("no pending-organizer sentinel survives in the armed fixture")
    func noPendingSentinelSurvives() throws {
        let object = try qwen38A6BTrackContractObject()
        let survivors = allStrings(in: object).filter { $0 == Self.pendingOrganizerSentinel }
        #expect(survivors.isEmpty, "the pending sentinel must be gone once armed: \(survivors)")
    }

    /// MODE FENCE. This track has exactly ONE speculative arm, the native MTP
    /// head embedded in the pinned target checkpoint. `serial` must be present
    /// because the baseline leg is pinned serial and is validated against this
    /// same list. `dflash` must be ABSENT: the DFlash arm is not part of this
    /// track and declaring it would arm a mode nothing here can run.
    @Test("allowed_modes declares serial and mtp only")
    func allowedModesDeclaresSerialAndMTPOnly() throws {
        let object = try qwen38A6BTrackContractObject()
        let modes = try #require(object["allowed_modes"] as? [String])
        #expect(modes == ["serial", "mtp"])
        #expect(!modes.contains("dflash"))
    }

    @Test("kv_backend is pinned explicitly to contiguous")
    func kvBackendIsContiguous() throws {
        let object = try qwen38A6BTrackContractObject()
        #expect(object["kv_backend"] as? String == "contiguous")
    }

    /// David ruling 2026-08-27, relayed by orchestrator: "Single-stream only".
    /// The scored width is 1, and the batch-8 ContinuousBatchingV2 adaptation
    /// is not pursued. The width is RULED AHEAD OF THE PIN: at the published
    /// channel tip the benchmarker certifies width 8 only, so this fixture is
    /// refused at that certification until the bench lane lands the
    /// single-stream regime. That refusal is fail-closed, and declaring the
    /// ruled shape is what this repository is supposed to do.
    @Test("scored_batch_size is pinned to the ruled single-stream width")
    func scoredBatchSizeIsOne() throws {
        let object = try qwen38A6BTrackContractObject()
        #expect(object["scored_batch_size"] as? Int == 1)
    }

    @Test("scored_exponents equals the ruled certify pair, exact field names")
    func scoredExponentsMatchesRuledPair() throws {
        let object = try qwen38A6BTrackContractObject()
        let exponents = try #require(object["scored_exponents"] as? [String: Any])
        // Field names must match benchd's `DeclaredScoredExponents` struct
        // exactly -- NOT the shorthand "prefill"/"decode" spelling, which
        // `ScoredExponents::certify` would treat as absent.
        let prefill = try #require(exponents["prefill_gain_exponent"] as? Double)
        let decode = try #require(exponents["decode_gain_exponent"] as? Double)
        #expect(prefill == 0.25)
        #expect(decode == 0.75)
        #expect(prefill.bitPattern == Double(0.25).bitPattern)
        #expect(decode.bitPattern == Double(0.75).bitPattern)
        #expect(exponents.count == 2, "scored_exponents must carry exactly the certify pair, no extra keys")
    }

    /// The pool is 8 prompts on the single-stream shape too: the streams run
    /// one at a time and their times are summed, so the pool cardinality is
    /// the SAMPLE COUNT, no longer the cohort width.
    @Test("timed_prompt_pool has exactly 8 slots")
    func timedPromptPoolHasEightSlots() throws {
        let object = try qwen38A6BTrackContractObject()
        let pool = try #require(object["timed_prompt_pool"] as? [[String: Any]])
        #expect(pool.count == 8)
    }

    /// Every pool slot is ARMED with a real pin: a 64-hex lowercase sha256, a
    /// positive byte count, and an r2_path that names one of this track's own
    /// goldens. No pending sentinel may survive; the byte-and-sha pair is what
    /// `tools/ranked-box-preflight.sh verify_pin` holds every staged tape to.
    @Test("every timed_prompt_pool entry is a real {r2_path, sha256, bytes} pin")
    func timedPromptPoolEntriesAreArmedPins() throws {
        let object = try qwen38A6BTrackContractObject()
        let pool = try #require(object["timed_prompt_pool"] as? [[String: Any]])
        for entry in pool {
            let sha = try #require(entry["sha256"] as? String)
            let r2 = try #require(entry["r2_path"] as? String)
            let bytes = try #require(entry["bytes"] as? Int)
            #expect(sha != Self.pendingOrganizerSentinel)
            #expect(isSixtyFourLowercaseHex(sha), "sha256 must be a 64-hex digest, got \(sha)")
            #expect(bytes > 0)
            #expect(r2.hasPrefix("correctness_prompts/qwen3.8-125b-a6b-mlx-v1/"))
            #expect(r2.hasSuffix(".golden.json"))
        }
    }

    @Test("hidden_correctness_golden is a root-level sibling, armed to the live golden")
    func hiddenCorrectnessGoldenIsRootLevelArmedPin() throws {
        let object = try qwen38A6BTrackContractObject()
        // ROOT level -- benchd's `hidden_correctness_golden_pin_from_contract`
        // reads this key directly off the contract root, never a nested wrapper.
        let golden = try #require(object["hidden_correctness_golden"] as? [String: Any])
        let sha = try #require(golden["sha256"] as? String)
        let bytes = try #require(golden["bytes"] as? Int)
        #expect(sha != Self.pendingOrganizerSentinel)
        #expect(isSixtyFourLowercaseHex(sha))
        #expect(bytes > 0)
        #expect(object["hidden_material"] == nil)
        // The hidden oracle is pinned to the live golden (botany): its pin must
        // be byte-identical to that prompt's entry in the timed pool.
        let liveName = try #require(object["live_golden"] as? String)
        let pool = try #require(object["timed_prompt_pool"] as? [[String: Any]])
        let live = try #require(
            pool.first {
                ($0["r2_path"] as? String)?.hasSuffix("/\(liveName).golden.json") == true
            })
        #expect(golden["sha256"] as? String == live["sha256"] as? String)
        #expect(golden["bytes"] as? Int == live["bytes"] as? Int)
    }

    @Test("target reference-model pin is a real 40-hex revision, matching the compiled constants")
    func targetPinIsFortyHexAndMatchesConstants() throws {
        let object = try qwen38A6BTrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let modelId = try #require(target["upstream_model_id"] as? String)
        let revision = try #require(target["upstream_revision"] as? String)
        #expect(modelId == qwen38A6BRepository)
        #expect(revision == qwen38A6BRevision)
        #expect(modelId == MLXFastConstants.referenceModelRepository)
        #expect(revision == MLXFastConstants.referenceModelRevision)
        #expect(isFortyLowercaseHex(revision))
    }

    @Test("target geometry matches the pinned checkpoint's own config.json")
    func targetGeometryMatchesPublishedConfig() throws {
        let object = try qwen38A6BTrackContractObject()
        let target = try #require(object["target"] as? [String: Any])
        let config = try qwen38A6BConfigObject()
        let text = try #require(config["text_config"] as? [String: Any])
        #expect(target["num_hidden_layers"] as? Int == text["num_hidden_layers"] as? Int)
        #expect(target["hidden_size"] as? Int == text["hidden_size"] as? Int)
        #expect(target["vocab_size"] as? Int == text["vocab_size"] as? Int)
        #expect(target["num_attention_heads"] as? Int == text["num_attention_heads"] as? Int)
        #expect(target["num_key_value_heads"] as? Int == text["num_key_value_heads"] as? Int)
        #expect(target["num_experts"] as? Int == text["num_experts"] as? Int)
        #expect(target["num_experts_per_tok"] as? Int == text["num_experts_per_tok"] as? Int)
        #expect(target["moe_intermediate_size"] as? Int == text["moe_intermediate_size"] as? Int)
        #expect(target["full_attention_interval"] as? Int == text["full_attention_interval"] as? Int)
        #expect(target["model_type"] as? String == text["model_type"] as? String)
        #expect(target["model_type"] as? String == MLXFastConstants.requiredGoldenModelType)
        // The declared full-attention indices must BE the config's own schedule.
        let layerTypes = try #require(text["layer_types"] as? [String])
        let derived = layerTypes.enumerated().filter { $0.element == "full_attention" }.map(\.offset)
        #expect(target["full_attention_layer_indices"] as? [Int] == derived)
        #expect(target["linear_attention_layer_count"] as? Int == layerTypes.count - derived.count)
    }

    /// The head is EMBEDDED. This track stages no head weight file: the MTP
    /// tensors ride inside the pinned target checkpoint under
    /// `language_model.mtp.*`, so there is no separate repository, no separate
    /// revision, and nothing for a head stager to fetch.
    @Test("mtp_head declares an embedded head, not a staged artifact")
    func mtpHeadIsEmbedded() throws {
        let object = try qwen38A6BTrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        #expect(head["source"] as? String == "embedded")
        #expect(head["tensor_prefix"] as? String == "language_model.mtp.")
        #expect(head["tensor_count"] as? Int == 76)
        #expect(head["num_hidden_layers"] as? Int == 1)
        #expect(head["use_dedicated_embeddings"] as? Bool == false)
        #expect(head["permitted_draft_depths"] as? [Int] == [1, 2, 3, 4, 5, 6])
        // No staged-artifact keys may appear: those would imply a fetch.
        #expect(head["upstream_model_id"] == nil)
        #expect(head["upstream_revision"] == nil)
        // And no separate assistant declaration survives from the Gemma track.
        #expect(object["assistant"] == nil)
    }

    @Test("the head tensor count agrees with the pinned tensor inventory")
    func mtpHeadTensorCountAgreesWithInventory() throws {
        let object = try qwen38A6BTrackContractObject()
        let head = try #require(object["mtp_head"] as? [String: Any])
        let inventory = try qwen38A6BTensorInventoryObject()
        let summary = try #require(inventory["summary"] as? [String: Any])
        #expect(head["tensor_count"] as? Int == summary["mtp_tensor_count"] as? Int)
    }

    @Test("scoring_semantics records the ruled composite formula")
    func scoringSemanticsRecordsRuledFormula() throws {
        let object = try qwen38A6BTrackContractObject()
        let semantics = try #require(object["scoring_semantics"] as? [String: Any])
        let formula = try #require(semantics["formula"] as? String)
        #expect(formula.contains("0.25"))
        #expect(formula.contains("0.75"))
        #expect(formula.contains("prefill_gain"))
        #expect(formula.contains("decode_gain"))
    }

    /// THE QUOTE IS THIS TRACK'S OWN, AND IT IS PINNED WHOLE.
    ///
    /// The fixture used to carry the Gemma-era ruling, which opened "we want to
    /// score gemma's benchmark" and described a sum over 8 concurrently timed
    /// streams. It was a real David quote, but it named another model and a
    /// regime this track does not run, sitting inside THIS track's contract.
    ///
    /// David re-issued it for this track on 2026-08-27 (relayed by
    /// orchestrator). The replacement is pinned EXACTLY, as one string, because
    /// a verbatim block is worth nothing if it can be paraphrased: a test that
    /// only looked for the exponents would pass on a summary of the quote.
    ///
    /// The no-Gemma scan covers the WHOLE `scoring_semantics` block -- every
    /// key and every nested string, `formula` included -- not just the quote:
    /// a survivor in a sibling field, or under a new key, must red just the same.
    @Test("ruling_verbatim is the 2026-08-27 quote for this track, exactly")
    func rulingVerbatimIsThisTracksOwnQuote() throws {
        let object = try qwen38A6BTrackContractObject()
        let semantics = try #require(object["scoring_semantics"] as? [String: Any])
        let ruling = try #require(semantics["ruling_verbatim"] as? [String])
        #expect(
            ruling == [
                "score the qwen 3.8 125b-a6b tracks (mlx and cuda) single-stream, "
                    + "paired serial vs the built-in mtp, on prefill gains ^ .25 * decode ^ .75"
            ])
        #expect(semantics["ruling_date"] as? String == "2026-08-27")
        let survivors = allStrings(in: semantics).filter { $0.lowercased().contains("gemma") }
        #expect(
            survivors.isEmpty,
            "the Gemma-era ruling must not survive anywhere in scoring_semantics: \(survivors)")
    }
}

/// Every string reachable inside a decoded JSON value -- object keys, string
/// values, and both recursively through nested objects and arrays -- so a whole
/// block can be scanned rather than one field of it.
private func allStrings(in value: Any) -> [String] {
    switch value {
    case let string as String:
        return [string]
    case let array as [Any]:
        return array.flatMap(allStrings(in:))
    case let object as [String: Any]:
        return object.flatMap { [$0.key] + allStrings(in: $0.value) }
    default:
        return []
    }
}

private func isFortyLowercaseHex(_ value: String) -> Bool {
    isLowercaseHex(value, count: 40)
}

private func isSixtyFourLowercaseHex(_ value: String) -> Bool {
    isLowercaseHex(value, count: 64)
}

private func isLowercaseHex(_ value: String, count: Int) -> Bool {
    value.utf8.count == count && value.utf8.allSatisfy { byte in
        (byte >= UInt8(ascii: "0") && byte <= UInt8(ascii: "9"))
            || (byte >= UInt8(ascii: "a") && byte <= UInt8(ascii: "f"))
    }
}

// MARK: - The served MTP head (David ruling 2026-09-12)

@Suite("qwen3.8-125b-a6b served MTP head pins")
struct Qwen38A6BServedHeadFixtureTests {
    @Test("mtp_head.source_checkpoint names the compiled head source, revision and manifest")
    func headSourceMatchesCompiledConstants() throws {
        let contract = try qwen38A6BTrackContractObject()
        let head = try #require(contract["mtp_head"] as? [String: Any])
        let source = try #require(head["source_checkpoint"] as? [String: Any])
        #expect(source["upstream_model_id"] as? String == MLXFastConstants.mtpHeadSourceRepository)
        #expect(source["upstream_model_id"] as? String == qwen38A6BMTPHeadSourceRepository)
        let revision = try #require(source["upstream_revision"] as? String)
        #expect(revision == MLXFastConstants.mtpHeadSourceRevision)
        #expect(revision.count == 40)
        #expect(revision.allSatisfy { "0123456789abcdef".contains($0) })
        #expect(source["manifest_path"] as? String == MLXFastConstants.mtpHeadSourceManifestPath)
        #expect(FileManager.default.fileExists(atPath: qwen38A6BMTPHeadManifestURL.path))
        #expect(source["tensor_count"] as? Int == Qwen4ExpCheckpointValidation.expectedMTPTensorCount)
        let quantization = try #require(source["quantization"] as? [String: Any])
        #expect(
            quantization["bits"] as? Int
                == Qwen4ExpCheckpointValidation.PinnedGeometry.mtpHeadServedQuantizationBits)
        #expect(quantization["group_size"] as? Int == 32)
        #expect(quantization["mode"] as? String == "affine")
        let served = try #require(head["served_quantization"] as? [String: Any])
        #expect(served["bits"] as? Int == 8)
        // The head stays EMBEDDED in the transformed tree; only its source moved.
        #expect(head["source"] as? String == "embedded")
        // And the declaration cap the runner enforces is the raised one.
        #expect(head["max_bytes"] as? Int == Gemma4MTPHeadDeclaration.defaultMaxBytes)
    }

    @Test("the head manifest pins the config and exactly the two head shards")
    func headManifestPinsTheTwoShards() throws {
        let records = try qwen38A6BManifestRecords(qwen38A6BMTPHeadManifestURL)
        #expect(records.count == 3)
        let contract = try qwen38A6BTrackContractObject()
        let head = try #require(contract["mtp_head"] as? [String: Any])
        let source = try #require(head["source_checkpoint"] as? [String: Any])
        let shards = try #require(source["shards"] as? [String])
        #expect(Set(records.map(\.path)) == Set(shards + ["config.json"]))
        for record in records {
            #expect(record.sha256.count == 64, "\(record.path)")
            #expect(record.sha256.allSatisfy { "0123456789abcdef".contains($0) }, "\(record.path)")
            #expect(record.bytes > 0, "\(record.path)")
        }
        // Two shards of 4.6 GB and 1.0 GB, not the publisher's whole 203 GB.
        let shardBytes = records.filter { $0.path.hasSuffix(".safetensors") }.map(\.bytes).reduce(0, +)
        #expect(shardBytes > 5_000_000_000 && shardBytes < 6_000_000_000)
        let headBytes = try #require(source["tensor_bytes"] as? Int)
        #expect(headBytes < shardBytes)
    }

    @Test("the head inventory fixture is the served head table, tensor for tensor")
    func headInventoryMatchesTheServedHeadTable() throws {
        let inventory = try qwen38A6BMTPHeadInventoryObject()
        let source = try #require(inventory["source"] as? [String: Any])
        #expect(source["repository"] as? String == MLXFastConstants.mtpHeadSourceRepository)
        #expect(source["revision"] as? String == MLXFastConstants.mtpHeadSourceRevision)
        let expected = Qwen4ExpCheckpointValidation.expectedHeadInventory(
            bits: Qwen4ExpCheckpointValidation.PinnedGeometry.mtpHeadServedQuantizationBits)
        let tensors = try #require(inventory["mtp_tensors"] as? [[Any]])
        #expect(tensors.count == expected.count)
        var seen = Set<String>()
        for entry in tensors {
            let name = try #require(entry[0] as? String)
            let dtype = try #require(entry[1] as? String)
            let shape = try #require(entry[2] as? [Int])
            let shard = try #require(entry[3] as? Int)
            let metadata = try #require(expected[name], "\(name)")
            #expect(metadata.dtype == dtype, "\(name)")
            #expect(metadata.shape == shape, "\(name)")
            #expect(shard == 1 || shard == 2, "\(name)")
            seen.insert(name)
        }
        #expect(seen == Set(expected.keys))
        let bytes = try #require(inventory["mtp_tensor_bytes"] as? Int)
        let contract = try qwen38A6BTrackContractObject()
        let head = try #require(contract["mtp_head"] as? [String: Any])
        let contractSource = try #require(head["source_checkpoint"] as? [String: Any])
        #expect(contractSource["tensor_bytes"] as? Int == bytes)
        // The spliced shard the transform must reproduce, pinned in both.
        let transformed = try #require(inventory["transformed_shard"] as? [String: Any])
        #expect(transformed["name"] as? String == "model-00022-of-00022.safetensors")
        let sha = try #require(transformed["sha256"] as? String)
        #expect(sha.count == 64)
        let contractShard = try #require(head["transformed_shard"] as? [String: Any])
        #expect(contractShard["sha256"] as? String == sha)
        #expect(contractShard["bytes"] as? Int == transformed["bytes"] as? Int)
    }
}
