public enum MLXFastConstants {
    // Qwen 3.8 125B A6B track identity (`qwen3.8-125b-a6b-mlx-v1`).
    //
    // PORTED 2026-08-27 from the Qwen 3.8 125B A6B MLX engine seed, which is a
    // tree-identical seed of the Gemma 4 26B A4B MLX engine. The pinned target
    // is an MLX 4-bit affine conversion published as
    // `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`. The raw checkpoint declares
    // `model_type` "qwen4_exp" at the top level and "qwen4_exp_text" inside its
    // nested `text_config`. Its speculative head is EMBEDDED in the same
    // checkpoint under `language_model.mtp.*` (76 tensors), so this track
    // stages no separate head artifact.
    //
    // RETIRED-NAME CHECK (AGENTS.md). The retired names are `gemma4-31b-it`,
    // `MLXFAST_MTP_`, `mtp-ranked`, `measure-mtp-job`, `mtp-weights` and
    // `laguna-xs-2.1-mtp`. `qwen3.8-125b-a6b-mlx-v1` is substring-clean against
    // every one of them, and this port introduces no `MLXFAST_MTP_`-prefixed
    // environment name.
    //
    // WHAT THE PIN COSTS, stated instead of discovered. `Golden.swift`
    // validates the provenance block of every golden against
    // repository+revision, and the CHECKED-IN public correctness goldens in
    // correctness_prompts/ are GEMMA captures, retained on David's ruling for
    // reuse of the 1024-token prompts. They are therefore REJECTED against
    // these constants. That is the fail-closed direction -- a Gemma golden must
    // not validate against a Qwen target -- and it is deliberate: the goldens
    // are hardware-generated and must be regenerated on the ranked box. The
    // local public drift gate cannot pass until then, and it should not.
    //
    // THE GEOMETRY BLOCK BELOW IS STILL THE GEMMA TOWER. This port moved the
    // checkpoint IDENTITY only. The frozen geometry, the hybrid cache schedule
    // and their mirror in Sources/MLXFastTransform describe the Gemma 4 26B A4B
    // text tower and must move together with the
    // model port that introduces the `qwen4_exp` tower. Until that lands, the
    // trusted config gate REFUSES the pinned target, which is the fail-closed
    // direction. See docs/qwen38-125b-a6b-port-notes.md.
    public static let referenceModelRepository = "Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP"
    public static let referenceModelRevision = "327c8a604de613b42f84ba5e6b796c0931e8aa3b"
    public static let referenceModelName = "Qwen3.8-Flash-Next-MLX-4bit-MTP"
    public static let defaultReferencePath = "reference_weights/Qwen3.8-Flash-Next-MLX-4bit-MTP"
    public static let defaultReferenceCachePath = ".cache/huggingface/hub/models--Vontra--Qwen3.8-Flash-Next-MLX-4bit-MTP/snapshots/327c8a604de613b42f84ba5e6b796c0931e8aa3b"
    public static let defaultWeightsPath = "weights"
    public static let defaultGoldenPath = "correctness_golden.json"
    public static let defaultPublicCorrectnessPromptPath = "correctness_prompts/public_longcopy_gate_english_1024.txt"
    public static let defaultPublicCorrectnessGoldenPath = "correctness_prompts/public_longcopy_gate_english_1024_256.json"
    public static let defaultPublicLocalSubmitGoldenPath = "correctness_prompts/public_longcopy_gate_english_1024_1024.json"
    public static let defaultScorePath = "score.json"
    public static let defaultLocalIterateScorePath = "score.local-iterate.json"

    // The model identity every golden must declare in its `model_type` key.
    // SINGLE SOURCE for this fork, the way benchd single-sources it as
    // `bench_core::constants::REQUIRED_GOLDEN_MODEL_TYPE`: the loader wrapper
    // in Golden.swift is the only consumer, so no call site gets to spell the
    // literal again and drift from it.
    //
    // This is the pinned target's TEXT-TOWER `model_type`: its `text_config`
    // declares "qwen4_exp_text" and its top level declares "qwen4_exp".
    //
    // Deliberately NOT unified with the frozen-invariant `model_type` check in
    // Sources/MLXFastTransform. That one reads the transformed WEIGHTS
    // config.json; this one reads a GOLDEN document. They are separate
    // contracts and the reference keeps them apart for the same reason.
    public static let requiredGoldenModelType = "qwen4_exp_text"

    // Frozen text-tower geometry of
    // Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP @ 327c8a60, READ OFF the pinned
    // revision's own config.json (`text_config`).
    //
    // This block, the checkpoint validator
    // in Sources/MLXFastTransform and the contract fixture's `target.*`
    // geometry move as ONE SET. A gate holding some fields of one model and
    // some of another rejects every checkpoint and explains none of them.
    // MLXFastCore is trusted and cannot import the editable model target, so
    // this block and `Qwen4ExpEngineConfig.validateFrozenInvariants()` are
    // deliberately duplicated and move in lockstep.
    //
    // THE TOWER IS HYBRID, and that is the fact most of the trusted code reads
    // from here. 48 layers on a FOUR-layer repeat: the LAST layer of each
    // group (index % 4 == 3, i.e. 3, 7, ... 47) is full attention and the
    // other three are gated deltanet linear attention. So 12 layers carry a
    // key-value cache and 36 carry a constant-size recurrent state.
    //
    // THERE IS NO SLIDING WINDOW on this target, unlike the Gemma tower this
    // tree was seeded from. Every one of the 12 attention layers is global.
    // Code that used to reason about a windowed cache now reasons about the
    // linear/full SPLIT instead; `slidingWindow` is gone rather than set to a
    // placeholder, so a stale reader fails to compile instead of reading a
    // number that means nothing here.
    //
    // THERE IS NO FINAL NORM TENSOR. The last hyper-connection mixer stands in
    // for it. Tensor-inventory reasoning that expects `model.norm.weight` is
    // wrong by exactly one tensor.
    public static let vocabSize = 248_320
    public static let hiddenSize = 2_560
    public static let numHiddenLayers = 48
    /// Query-head count of a FULL-ATTENTION layer. The linear layers have no
    /// query heads in this sense; their head counts are the `linear*` pins.
    public static let attentionHeads = 24
    public static let numKeyValueHeads = 2
    public static let headDim = 256
    /// `full_attention_interval`: the LAST layer of each group of four is full
    /// attention. Trusted code that has to reason about the hybrid cache stack
    /// reads the schedule from here instead of repeating the literal.
    public static let fullAttentionInterval = 4
    public static let rmsNormEps = 1e-6

    /// HOW THE CHECKPOINT STORES ITS NON-GATED RMSNORM WEIGHTS.
    ///
    /// The two conventions in circulation compute the same function from
    /// DIFFERENT stored bytes, and nothing in the checkpoint's `config.json`
    /// says which one it holds -- the file carries `rms_norm_eps` and nothing
    /// else about the norms. Get it wrong in either direction and every
    /// non-gated norm in the tower is off by one, which multiplies activations
    /// by roughly 0 (or by roughly 2) and produces incoherent output at every
    /// quantization level.
    ///
    /// PROVENANCE. Reported on ml-explore/mlx-lm pull request 1788: the
    /// OFFICIAL `Qwen/Qwen3.8-Flash-Next` checkpoint stores zero-centered
    /// weights and needs `y * (1 + w)` (Sofille65, 2026-08-26); and, measured
    /// element-wise across the non-gated norm tensors, the Vontra conversion
    /// stores `1 + w` already, so the same code path computes `2 + w` on it
    /// (eauchs, 2026-08-27). The gated deltanet norm stays conventional in
    /// BOTH checkpoints.
    ///
    /// THIS IS A CHECKPOINT FACT, SO IT IS PINNED AND THEN VERIFIED AGAINST
    /// THE LOADED TENSORS. The value below is the declaration;
    /// `validateLoadedNormConvention` measures what the checkpoint actually
    /// holds and refuses by name when the two disagree. That way a wrong pin
    /// stops the worker instead of quietly degrading the model.
    public enum RMSNormConvention: String, Equatable, Sendable {
        /// The checkpoint stores `w`, and the model computes `y * (1 + w)`.
        /// Sampled non-gated norm weights average near ZERO.
        case zeroCentered
        /// The checkpoint stores `1 + w`, and the model computes `y * w`.
        /// Sampled non-gated norm weights average near ONE.
        case offsetBaked

        /// The additive offset the model applies to a stored weight.
        public var weightOffset: Float {
            switch self {
            case .zeroCentered: return 1
            case .offsetBaked: return 0
            }
        }
    }

    /// The PINNED convention of this track's checkpoint: it BAKES THE OFFSET.
    ///
    /// MEASURED ON THE BOX, 2026-08-28, over all 48 layers and the MTP head,
    /// read-only. Every non-gated norm family's weights are strictly positive
    /// and centred near or above 1, so the checkpoint already holds `1 + w`:
    ///
    ///   attention hyper-connection `hc_norm` (48)   mean of means 1.036
    ///   mixture hyper-connection `hc_norm` (48)                   1.374
    ///   final `hyper_connection_mixer.hc_norm` (1)                3.750
    ///   `self_attn.q_norm` / `k_norm` (12 + 12)            1.454 / 1.453
    ///   `indexer.q_layernorm` / `k_layernorm`              0.954 / 0.955
    ///   `ple.norm_conv` / `norm_key` / `norm_query`  0.891 / 0.893 / 0.893
    ///   `mtp.*` hyper-connection norms                       1.28 - 4.79
    ///   `mtp.*` `q_norm` / `k_norm`                                 3.68
    ///   `mtp.*` indexer layernorms                         1.10 / 1.08
    ///   `mtp.pre_fc_norm_embedding` / `_hidden`            0.236 / 0.672
    ///
    /// The GATED deltanet norm (`linear_attn.norm`, 36 tensors, mean 1.024) is
    /// conventional in this checkpoint and in the official one alike, and is
    /// not part of this fact: `Qwen4ExpRMSNormGated` always computes `y * w`.
    ///
    /// The upstream thread on ml-explore/mlx-lm pull request 1788 corroborates
    /// it from the other direction: the OFFICIAL checkpoint is zero-centered
    /// and needs `y * (1 + w)` (Sofille65, 2026-08-26), and the Vontra
    /// conversion was measured element-wise to hold `1 + w` already (eauchs,
    /// 2026-08-27). mlx-lm at the port's source revision computes
    /// `mx.fast.rms_norm(x, 1.0 + self.weight)` unconditionally, so the
    /// REFERENCE double-counts on this checkpoint too -- which is why both
    /// implementations produced incoherent output on real weights.
    public static let rmsNormConvention: RMSNormConvention = .offsetBaked

    /// THE CLASSIFIER IS THE SIGN PATTERN, NOT THE MEAN, and the box numbers
    /// are why. `mtp.pre_fc_norm_embedding` averages 0.236 -- close enough to
    /// zero to fool a mean threshold -- while every one of its entries is
    /// POSITIVE. A zero-centered tensor is scattered around zero and has
    /// roughly half its entries negative; a baked tensor is `1 + w` and
    /// straddles zero only where some `w < -1`.
    ///
    /// THE BOUNDARY IS 0.20, AND THE MEASURED DISTRIBUTION IS WHY. Across all
    /// 157 non-gated language-model norm tensors of the pinned tree, measured
    /// on the box 2026-08-28, the LARGEST negative fraction is 0.0551
    /// (`model.layers.0.mlp_hyper_connection.hc_norm`, mean 0.8905). A
    /// genuinely zero-centered tensor sits near 0.5 -- the pinned tree's own
    /// vision-tower LayerNorm biases, which this check never reads, run
    /// 0.45 to 0.57. So the two populations are separated by roughly an order
    /// of magnitude, and 0.20 sits between them with about 3.6x of headroom
    /// below the zero-centered floor and 3.6x above the observed baked
    /// maximum.
    ///
    /// AN EARLIER 0.05 CEILING WAS WRONG, and the way it was wrong is the
    /// reason this comment carries numbers. It was chosen from family
    /// AVERAGES, and one tensor in 193 exceeded it: the hyper-connection norms
    /// are not a unit-scale gain (the final mixer averages 3.75, one layer-0
    /// tensor reaches -4.94), so a tight ceiling is inherently marginal on
    /// that family. The pinned checkpoint would have been REFUSED at load.
    ///
    /// ONE NUMBER, NOT TWO. A tensor is read as BAKED when FEWER than
    /// `rmsNormZeroCenteredMinNegativeFraction` of its entries are negative
    /// and its mean clears `rmsNormBakedMinMean`; it is read as ZERO-CENTERED
    /// when its negative fraction lands in
    /// `[rmsNormZeroCenteredMinNegativeFraction,
    /// rmsNormZeroCenteredMaxNegativeFraction]`.
    ///
    /// A separate "baked ceiling" constant used to sit here. It was DEAD --
    /// the baked branch reads the zero-centered floor -- so it could be given
    /// any value at all without changing a verdict while its documentation
    /// claimed to be the rule. Deleted rather than wired up: the two rules are
    /// one boundary and there is nothing for a second constant to say.
    ///
    /// The floor is "about half", loosely, because a trained norm is not
    /// symmetric. The rules meet at this one number, so no tensor can fall
    /// between them.
    public static let rmsNormZeroCenteredMinNegativeFraction: Float = 0.20
    public static let rmsNormZeroCenteredMaxNegativeFraction: Float = 0.80

    /// A baked tensor's mean must also clear this floor. It is a second,
    /// weaker check on the same fact: the lowest the box measured is
    /// `mtp.pre_fc_norm_embedding` at 0.236, so the floor sits below that with
    /// room, and it exists only to catch an all-positive tensor that is
    /// nowhere near 1 -- a tensor of near-zeros, which is neither convention
    /// and is what a mis-converted file looks like.
    public static let rmsNormBakedMinMean: Float = 0.05
    public static let maxPositionEmbeddings = 262_144
    public static let tieWordEmbeddings = false

    // Mixture of experts. Every layer carries one, including the layers that
    // also carry the PLE block.
    public static let numExperts = 512
    public static let numExpertsPerToken = 10
    public static let moeIntermediateSize = 640
    public static let sharedExpertIntermediateSize = 640

    // Gated deltanet, on the 36 linear-attention layers.
    public static let linearNumKeyHeads = 16
    public static let linearNumValueHeads = 48
    public static let linearKeyHeadDim = 128
    public static let linearValueHeadDim = 128
    public static let linearConvKernelDim = 4
    public static let outputGateType = "sigmoid"

    // Hyper-connections. The residual stream is `hcCount` parallel streams the
    // whole way down the tower, so the stream width is hcCount * hiddenSize.
    public static let hcCount = 4
    public static let hcLowrank = 320

    // QSA indexer, on each full-attention layer. Below the budget the indexer
    // returns no mask and attention stays plain causal.
    public static let indexerHeads = 4
    public static let indexerKVHeads = 1
    public static let indexerHeadDim = 128
    public static let indexerBudget = 2_048
    public static let indexerCompressRatio = 4

    // Partial rotary: only the first quarter of each head is rotated.
    // `mrope_interleaved` with sections [11, 11, 10] is a MULTIMODAL fact; for
    // a text-only tower every section shares one position per token, which
    // makes interleaved mrope identical to plain rope over the same positions.
    public static let partialRotaryFactor = 0.25
    public static let ropeTheta = 10_000_000.0

    // N-gram / per-layer embedding. `pleLayerIds` is ONE-BASED in the
    // checkpoint config, so [2] means layer INDEX 1.
    public static let ngramSize = 3
    public static let headsPerNGram = 8
    public static let ngramVocabSizeBase = 20_000_000
    public static let makeNGramVocabSizeDivisibleBy = 128
    public static let splitNGramParts = 128
    public static let pleEmbedDim = 2_560
    public static let pleLayerIds = [2]
    public static let pleConvKernelSize = 4

    // Native multi-token prediction, embedded in the target checkpoint.
    public static let mtpNumHiddenLayers = 1
    public static let mtpUseDedicatedEmbeddings = false

    public static let bosTokenId = 248_044
    public static let padTokenId = 248_044
    /// The checkpoint declares a LIST at the root. The first entry is the one
    /// the n-gram hash uses as its segment boundary.
    public static let eosTokenIds = [248_046, 248_044]

    // 1_024 (was 512): David's 2026-08-24 seed-length ruling for the Gemma 4
    // track — "Seed becomes 1024". The decode window is unchanged
    // (`benchmarkDecodeSteps` stays 128); golden shape becomes 1024
    // prompt_tokens + 129 expected_tokens (seed next-token + 128 checked
    // steps). The hidden pool prompts must be re-authored/re-uploaded at 1024
    // tokens and every golden regenerated on the box; see
    // docs/qwen38-125b-a6b-port-notes.md.
    public static let correctnessPromptTokens = 1_024
    // Keep the public gate long enough to catch broad decode regressions while
    // leaving budget for the hidden GPQA behavior checks in the official job.
    public static let correctnessSteps = 64
    public static let correctnessTopLogits = 8
    public static let correctnessLogitTieTolerance = 1e-6
    public static let correctnessMaxAnchorContextTokens = 1_024
    public static let correctnessMaxFreeRunSteps = 256
    public static let correctnessMaxBehaviorPromptTokens = 2_048
    public static let correctnessMaxBehaviorSteps = 128
    public static let correctnessGPQACaseCount = 9
    // Cross-machine greedy decode can drift on hidden GPQA even with pinned
    // Swift/MLX. Semantic GPQA behavior captures a short continuation for the
    // private judge; exact token enforcement stays on the long copy gate and
    // non-semantic behavior fixtures.
    // 128 (was 64; before that 10, DeepSeek-era): the 10->64 history and its
    // calibration runs predate the GPQA prompt-encoding (BOS) fix and
    // measured degenerate no-BOS completions, so they no longer bind. With
    // BOS the reference answers letter-first and then explains; 128 lets the
    // explanation finish for the judge instead of cutting mid-sentence.
    // Generation happens in the untimed gates phase (never the frozen timed
    // window), so the cost is ~10-15s of job wall-clock, not score.
    public static let correctnessGPQAMaxNewTokens = 128
    // Semantic judging uses short hidden GPQA answers as a baseline-calibrated
    // gate for optimizations that preserve the exact prefix but damage answer
    // sense. 9 (was 5): raised to the full fixture together with the GPQA
    // prompt-encoding (BOS) fix -- selection takes the first N budget-valid
    // cases in file order, and the old window of 5 contained only two of the
    // five cases the correctly-prompted reference answers right. Per-case
    // cost is one short untimed generation plus one judge call; the 4 extra
    // cases add roughly a minute to the job.
    // The captured answer is a prefix of the behavior-gate generation, so
    // semanticGPQAMaxNewTokens is only effective up to
    // correctnessGPQAMaxNewTokens (and must stay <= correctnessMaxBehaviorSteps).
    public static let semanticGPQACaseCount = 9
    public static let semanticGPQAMaxNewTokens = 128
    // 7 of 9, set from measurement rather than prediction. The gate compares
    // the candidate against the pinned reference model's own recorded answers
    // (accepted_responses in the hidden fixture), so it is a regression check:
    // an unmodified candidate reproduces the reference on every case by
    // construction, independent of whether those answers are factually right.
    // That is the point of the design -- the reference model is at chance on
    // these questions, so a correctness-based gate could only ever sit on the
    // noise floor (see the 2026-07-27 measurements: 1-4 of 9 correct depending
    // on option order, with a single case carrying the entire margin).
    // Calibration, 2026-07-27, offline against the real gate script:
    //   self-match (unmodified candidate), 27 runs / 243 judgements: 9/9 every
    //     run, zero variance, including the one case whose reference output is
    //     degenerate.
    //   three answers changed to a different option, 8 runs: exactly 6/9 every
    //     run, failing only the changed cases.
    //   answer content preserved but label flipped or tail truncated, 8 runs:
    //     9/9 -- cosmetic near-tie drift is tolerated.
    // So judge nondeterminism costs nothing, each damaged answer costs exactly
    // one case, and a floor of 7 absorbs two independent damaged answers. It
    // also clears the >= 6 needed to reject a submission that hardcodes one
    // fixed letter: the reference selects a spread of letters, so a constant
    // answer matches at most 5 of 9.
    // Earlier floors of 1 were calibrated against pre-BOS-fix runs whose
    // reference never answered at all, so they justified nothing.
    // Keep in sync with the workflow env MLXFAST_SEMANTIC_GPQA_MIN_PASS and
    // run-semantic-gpqa-gate.sh. Regenerating the fixture's accepted_responses
    // (new prompts, token budget, or reference checkpoint) invalidates this
    // calibration -- re-run it.
    public static let semanticGPQAMinPassCount = 7
    // 1_024 (was 512): moves with `correctnessPromptTokens` under the
    // 2026-08-24 seed-length ruling — the timed prefill leg is now 8 x 1024
    // tokens per cohort. Any baseline/calibration value derived at the
    // 512-token prefill window is invalidated by this change and must be
    // re-derived before scoring arms.
    public static let benchmarkPrefillPromptTokens = 1_024
    // Stable public identifier for the private timed-evaluation prompt. The
    // prompt bytes remain operator-provisioned; changing this identifier is a
    // ranking-contract change and forces build-hash-keyed timed oracles to be
    // regenerated.
    public static let benchmarkEvaluationTargetID = "lowsim-prose-qwen38-v1"
    // Offline prompt-lookup susceptibility gate. The analyzer simulates a
    // longest recurrent suffix predictor over these orders. At <= 3% accepted
    // draft tokens, even an idealized zero-overhead predictor is capped near
    // 1.03x before verification and lookup overhead.
    public static let benchmarkNGramSelfSimilarityOrders = [1, 2, 3]
    public static let benchmarkMaxPromptLookupHitRate = 0.03
    // Scored decode is parent-measured wall time for decode setup plus this
    // many checked token steps. Charging setup prevents submitted model code
    // from precomputing future decode tokens in an unscored seed-prefill phase.
    public static let benchmarkDecodeSteps = 128
    // Local iterate charges the same 1024-token seed prefill as the official
    // decode window, so it must use the same denominator to produce a
    // comparable decode seconds-per-token estimate.
    public static let localIterateBenchmarkDecodeSteps = benchmarkDecodeSteps
    // Local submit uses a longer public fixture so the Yukon pre-submit hook
    // exercises one continuous decode trajectory for about ten minutes instead
    // of repeating the short local-iterate correctness window.
    public static let localSubmitBenchmarkDecodeSteps = 1023
    public static let localSubmitBenchmarkRepeats = 1
    // Seed measured decode with the full prompt. A short instruction-prefix
    // seed can free-run differently across Apple Silicon/MLX versions even
    // when teacher-forced correctness agrees, which makes the timed oracle
    // fragile for reasons unrelated to kernel performance.
    // 1_024 (was 512): moves with `correctnessPromptTokens` /
    // `benchmarkPrefillPromptTokens` under the 2026-08-24 seed-length ruling.
    public static let benchmarkDecodeSeedTokens = 1_024
    // Official paired timing runs LAST at workflow level, after correctness,
    // GPQA, and the hidden-material scrub. The on-box wrapper launches the
    // baseline and candidate in fresh worker processes, and each timed prefill
    // starts without an in-process warmup. The calibration below used that
    // same shape, so keep zero warmup and one measured run.
    public static let benchmarkPrefillWarmupRuns = 0
    public static let benchmarkPrefillTimedRuns = 1
    // Acceptance bands (see AcceptanceBand + docs/thermal-variance-investigation.md).
    // Prefill and decode are noisy single measurements, gated against the same-VM
    // paired baseline B (which cancels host-speed differences). Each run's value must
    // land within [B*(1-down), B*(1+up)]; > +up = slowdown/regression (fail),
    // < -down = improvement too large for one submission / lucky reading (fail).
    //
    // Prefill: +/-3% symmetric -- prefill is not a real optimization axis, so it is a
    // health gate (regression and lucky-fast both fail past 3%).
    //
    // Decode: +1% regression / -2.5% gain -- tight on regressions (decode is the primary
    // scored axis), and a single submission's decode gain is capped at 2.5%; larger wins
    // must be CHUNKED across submissions (bounds lucky-measurement inflation and forces
    // incremental, verifiable progress). Decode is the axis the score rewards, but the
    // per-submission step is capped, not the cumulative total across submissions.
    //
    // BAND DERIVATION (gemma4 calibration session 2026-08-25, calibration-20260825T102741Z;
    // flagged for reviewer attention because the methodology, not just the numbers, is new):
    // the band SHAPE is preserved from the qwen-era precedent (symmetric prefill health
    // gate; asymmetric decode with the tight side on regressions), and the magnitudes are
    // re-derived from this session's measured variability. The qwen precedent pinned its
    // tight sides at ~7.7x the session CV (prefill 5% over CV 0.65%; decode +2% over CV
    // 0.26%) and its decode gain cap at ~19x; i.e. bands sat ~8-25x over measured CV.
    // Rule applied here: tight-side band = ceil-to-half-percent of 10x the per-axis
    // session CV (10x sits at the conservative end of that precedent envelope):
    //   prefill: 10 x 0.2712% = 2.712% -> 3.0% both sides (symmetric).
    //            Bounds: < qwen's 5% (never loosened); >= 4x session spread
    //            (4 x 0.581% = 2.324%) so ordinary run-to-run scatter cannot trip it.
    //   decode up: 10 x 0.0937% = 0.937% -> 1.0%. Bounds: < qwen's +2%;
    //            >= 4 x 0.2084% = 0.834%.
    //   decode down: qwen's down:up asymmetry ratio (5/2 = 2.5) preserved:
    //            1.0% x 2.5 = 2.5%. The alternative candidate was keeping the 5%
    //            per-submission chunking cap unchanged (it is partly policy, not pure
    //            variability); the tighter candidate is pinned per operator instruction,
    //            with the alternative recorded in the calibration PR for review.
    public static let prefillBandUpTolerance = 0.03
    public static let prefillBandDownTolerance = 0.03
    public static let decodeBandUpTolerance = 0.01
    public static let decodeBandDownTolerance = 0.025
    // The Poolside Laguna XS 2.1 NVFP4 text tower is ~21.6 GB; 25 GiB keeps ample
    // headroom for shard alignment/padding without approving a second full
    // copy of the model.
    public static let defaultMaxTransformedWeightsBytes = 25 * 1024 * 1024 * 1024
    public static let defaultMaxSubmissionSourceBytes = 256 * 1024 * 1024

    // FREE-RUN / COHORT TOKEN CEILING. The largest `total_tokens` a configured
    // free-run or cohort request may ask a worker for, enforced by
    // `RuntimeWorkerRequestValidation` and `RuntimeWorkerCohortSupport` and used
    // as the per-stream `maxTokensPerStream` (+1 for the seed) the free-run
    // sessions open their engines with. 1,536 is three wraps of a 512-position
    // sliding-window cache, which is what keeps a wrap-seam tail boundary
    // reachable from a configured diagnostic run.
    //
    // Renamed from the old `experimentalDFlash`-prefixed spelling with the
    // DFlash excision: the VALUE is unchanged, and every consumer of it was,
    // and still is, a non-DFlash free-run/cohort path.
    public static let freeRunMaxConfiguredTotalTokens = 1_536

    // MARK: - Qwen 3.8 native-MTP track (qwen3.8-27b-mtp-v1)

    /// Tensors the MTP head carries.
    ///
    /// It lives HERE, in the no-MLX core, because both sides need it and they
    /// cannot share a type: the worker enforces it at load
    /// (`Qwen36MTPHeadAttachment`), and the trusted CLI reports it in the
    /// evidence payload without linking any model code.
    ///
    /// MEASURED 2026-08-14 ON THE 3.8 HEAD; the QWEN38-VERIFY-AT-RELEASE hedge
    /// that stood here is DISCHARGED. The head extracted from the official bf16
    /// base `Qwen/Qwen3.8-27B` @ `1d4bf0f2` carries **15 bf16 tensors**
    /// (849,398,784 tensor bytes in a single 849,400,347-byte
    /// `model.safetensors`).
    ///
    /// WHY IT WAS 31 AND WHY THAT IS NOT A CONTRADICTION. The old count came
    /// from the 3.6 head's own `model.safetensors.index.json`, and that head was
    /// a 4-bit group-64 MLX conversion: its 8 matrices each appear as a
    /// weight/scales/biases TRIPLE (24 entries) alongside 7 norms, which is 31.
    /// The 3.8 head is bf16 and unquantized, so the same 8 matrices are 8
    /// entries: 8 + 7 = 15. The head did not change shape; the count was
    /// counting a quantization layout, not an architecture.
    ///
    /// CONSEQUENCE, stated rather than discovered. `setup-qwen-mtp.sh` still
    /// defaults to the 3.6 head for the local path, and that head has 31
    /// tensors, so the local load now refuses on THIS constant. That is the
    /// same intended refusal the mismatched local pair already carried (a 3.6
    /// head cannot merge onto a 3.8 backbone), arriving one step earlier and
    /// with a clearer message.
    public static let gemma4MTPHeadTensorCount = 15

    /// TRUSTED BOUND on the drafts a single round may actually propose.
    ///
    /// OPERATOR-RATIFIED 2026-08-14. The track no longer pins a draft depth:
    /// depth and per-round schedule are part of the competitive surface, and a
    /// candidate picks its own draft count per round — 0 through this value,
    /// adaptively if it likes. This constant is the ONE bound, and it exists
    /// for exactly one reason: it bounds the verify width a round may ask the
    /// target for, which bounds the memory and the row ledger a single round
    /// can demand. It is deliberately NOT a statement about which depth is
    /// good; the measured envelope (1.74x @ 32 tokens, 1.34x @ 128, ~1.0x @
    /// 512, depth 3 exact-but-slower) is now a competitor's problem, not a
    /// pinned parameter.
    ///
    /// It lives in the no-MLX core because the TRUSTED parent enforces it
    /// (`Gemma4RuntimeMTPDriver.requireStructurallySound`, over the draft count a
    /// round ACTUALLY proposed) while the worker mirrors it as a request bound
    /// through `Qwen36MTPLimits.maxDepth`. The parent's check is the one that
    /// binds: the worker's copy sits in editable model code.
    public static let gemma4MTPMaxDraftDepth = 8

    /// Wire/request spelling of the same bound, kept because the worker
    /// protocol and `Qwen36MTPLimits` were written against this name. It is an
    /// alias, not a second knob — a submission that raised one and not the
    /// other would still be bounded by `gemma4MTPMaxDraftDepth` at the parent.
    public static let gemma4MTPMaxDepth = gemma4MTPMaxDraftDepth

    /// The depth of the TRUE SERIAL CONTROL: 0, meaning MTP OFF.
    ///
    /// Operator design, and it is not a naming choice. Depth 1 is NOT serial: it
    /// is a one-deep speculative decoder that still drafts, still verifies and
    /// still accepts — measured on box 3 at 512 tokens it ran 302 rounds with a
    /// 0.699 accept rate, i.e. an ALREADY-ACCELERATED baseline. Dividing by it
    /// measures depth-2 against one-deep speculation, not against serial decode,
    /// and that is how a 0.875x "speedup" came out of a method the authors
    /// measured at 1.34x @ 128 against true serial.
    ///
    /// Depth 0 therefore drafts nothing and consults the head not at all: one
    /// token per target forward. Depth 1 remains available as a labelled
    /// speculative-depth-1 diagnostic and is never the denominator.
    public static let gemma4MTPSerialControlDepth = 0

    /// Semantic GPQA min-pass for THIS track, derived per NEW-MODEL-BRINGUP 7.4
    /// as `min(observed) - 1` over the four baseline-equivalent ranked
    /// calibration dispatches of 2026-08-13 (31712368539, 31715555814,
    /// 31718615518, 31721547429). Every one of them judged 9/9, so
    /// `min(observed) = 9` and the floor is 8 — one case of error budget for
    /// judge nondeterminism, which is the whole point of the -1.
    ///
    /// TRACK-SCOPED deliberately. The unprefixed `semanticGPQAMinPassCount = 7`
    /// above is the SHARED serial track's calibrated floor; raising the shared
    /// constant would retune a live track that has not been re-derived.
    /// Mirrored into MLXFAST_SEMANTIC_GPQA_MIN_PASS in the Qwen
    /// workflow ONLY, and pinned to it by
    /// `theQwenWorkflowMirrorsTheTrackScopedGPQAFloor`. The shared
    /// `run-semantic-gpqa-gate.sh` default stays 7: it serves both tracks, and
    /// the value that binds is the workflow env, not the script default.
    ///
    /// KNOWN LIMITATION — this floor cannot reject a constant-"A" answerer, and
    /// raising it to 9 would not fix that. Every `answer_key` in the hidden
    /// fixture is "A" while each prompt offers four options, and the reference
    /// model is measurably position-biased toward A (it tracks the correct
    /// option only ~2/9 when option order is rotated). Since
    /// `accepted_responses` was filled from the reference model's own captures,
    /// the gate measures FIDELITY TO THE REFERENCE, not question-answering
    /// accuracy: 8 of the 9 reference captures are "A", so a constant-"A"
    /// answerer scores exactly 8 — precisely this floor. Going to 9 would only
    /// delete the judge-nondeterminism budget while still admitting it. The
    /// real fix is shuffling option order, which is organizer material and is
    /// tracked operator-side, deliberately not done in this repo.
    /// (Do NOT restate the older "the reference selects a spread of letters"
    /// rationale attached to the shared constant above — it is false for this
    /// fixture and this floor does not rest on it.)
    /// QWEN38-VERIFY-AT-RELEASE: 8 is min(observed) - 1 over four QWEN 3.6
    /// ranked calibration dispatches. NEW-MODEL-BRINGUP 7.4 requires that
    /// derivation to be re-run against the 3.8 model -- and the regenerated
    /// GPQA fixture -- before this floor means anything.
    public static let gemma4MTPSemanticGPQAMinPassCount = 8

}
