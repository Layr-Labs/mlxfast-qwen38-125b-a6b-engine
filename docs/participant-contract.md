# Qwen 3.8 125B A6B — participant contract

This document states the terms that bind a submission to track
`qwen3.8-125b-a6b-mlx-v1`.

`benchmark.json` is the Yukon track manifest.
`fixtures/qwen3_8_125b_a6b_track.json` is the track contract fixture. Both files
carry pure configuration: values, paths, commands, and pins. They carry no
prose. This document explains those files. It never overrides them.

## 1. Order of authority

Apply these in order. The higher entry wins.

1. The ranked run on the official runner. It is the authority on any score.
2. `fixtures/qwen3_8_125b_a6b_track.json` and `benchmark.json`.
3. This document.
4. `README.md` and `TASK.md`.

If either configuration file disagrees with this document on a plain value, the
configuration file wins. If either disagrees with the benchmarker about
measurement, the benchmarker wins.

The benchmarker is a prebuilt `benchd` binary resolved from the bench
repository's release channel (the track branch's `dist/`). The channel publishes
`benchd.manifest.json` (`{branch, source_commit, sha256, bytes}`) beside the
binary; `./tools/fetch-benchd.sh` verifies the binary against that manifest,
installs both into `benchd-bin/`, and logs the resolved identity. The harness is
trusted-side: a submission cannot change what measures it. This repository
carries no submodule and no sha pin.

## 2. What the track measures

The track measures Qwen 3.8 125B A6B MLX text-tower inference speed.

You optimize the MLX runner, the offline transform, the batching engine, and
the vendored MLX Metal kernel families that the forward pass dispatches. You
also optimize the speculative-decode arm.

The target model is `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` at revision
`327c8a604de613b42f84ba5e6b796c0931e8aa3b`. It is a sparse MoE model.

| Property | Value |
|---|---|
| Architecture | `qwen4_exp`. The text tower is `qwen4_exp_text`. |
| Hidden layers | 48, on a four-layer repeat |
| Full attention | 12 layers, at index `% 4 == 3`: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 |
| Linear attention | The other 36 layers. They are gated deltanet and carry a constant-size recurrent state. |
| Full-attention heads | 24 query heads, 2 KV heads, head dimension 256 |
| Rotary | Partial 0.25, `rope_theta` 1e7, interleaved mrope sections [11, 11, 10] |
| QSA indexer | 4 heads, 1 KV head, dimension 128, budget 2048, compress 4 |
| Hyper-connections | `hc_count` 4, `hc_lowrank` 320 |
| MoE | 512 routed experts, 10 per token, `moe_intermediate` 640, plus a shared expert of width 640 behind a shared expert gate |
| n-gram / PLE | Layer index 1. `ngram_size` 3, 8 heads per n-gram, 128 split parts: 384 shard tensors plus 3 int64 buffers. The table is offloaded to SSD behind a bounded LRU. |
| Hidden size | 2560 |
| Vocabulary | 248320, embeddings untied |
| Tokens | eos 248046 and 248044; bos and pad 248044 |
| Quantization | Affine, group size 32, 4 bits. Router gates and multimodal weights are BF16. |
| Final norm | There is no `model.norm` tensor. The final `hyper_connection_mixer` stands in for it. |
| Raw tensors | 3747 across 22 shards. Index `total_size` is 113,209,155,128 bytes. |
| Text tower | 3414 tensors. The vision tower is 333 tensors and the loader skips them. |

There is no sliding-window attention on this model.

The speculative-decode arm is the native MTP head. It is embedded in the pinned
target checkpoint under `language_model.mtp.*`: 76 tensors, 1 hidden layer,
hybrid full attention. It has no embedding and no `lm_head` of its own. It
rides the target's `language_model.embed_tokens` and `language_model.lm_head`.

`kv_backend` is pinned `contiguous` on both legs. The benchmarker refuses when
it cannot honour the pinned backend. It does not degrade to another backend.

## 3. What you may edit

`benchmark.json` `editablePaths` is the authority. It lists 71 entries.

The rule behind the list: anything that only **proposes** tokens or computes
the forward pass is editable. Anything that **verifies**, **measures**, or
**ledgers** stays trusted.

The editable surface has four groups.

1. **The head declaration.** `mtp-head.manifest.json`. The declaration file
   only. It carries no weights, and no head weights directory exists. See
   section 4.
2. **The Runner.** `Runner/`.
3. **The offline transform.** `Sources/MLXFastTransform/`.
4. **The vendored MLX Metal kernels.** The 68 files the forward pass
   dispatches: the quantized matmul, the mixture-of-experts gather-GEMM, SDPA
   and steel attention, RoPE, RMSNorm, softmax, sort, reduce, copy,
   elementwise, `arg_reduce`, and gather indexing.

THE RUNNER IS EDITABLE, AND IT LIVES IN `Runner/`. It is the model family's
code: it loads the checkpoint, it declares the manifest, and it builds the
engine and the one-row stepper. `Sources/BenchWorker/` registers it in
`RunnerRegistry` before the engine resolves a runner, so it SHADOWS the fork's
built-in runner for `qwen4_exp` and `qwen4_exp_text`. Keep the manifest as it
is: the runner manifest digest is a benchd conformance input, and a changed
digest fails the conformance check.

THE ENGINE CORE IS NOT AN EDITABLE PATH. It is the `Vendor/mlx-swift-lm`
submodule. A gitlink names a commit, not bytes in this tree, so an editable
entry over it would let a submission move the engine to a commit nothing here
verified. Whether a submission may repoint the gitlink at its own fork commit,
and under what repository allowlist, is not ruled yet.

### 3.1 Optional paths

`optionalEditablePaths` lists `mtp-head.manifest.json`.

A submission archive has REPLACE semantics over `editablePaths`. An absent head
declaration means the embedded head. The overlay therefore skips a missing
optional path instead of failing closed.
Yukon's overlay reads this list from the trusted contract, never from the
submission.

### 3.2 The byte budget

`editableSurfaceByteBudget` caps the editable surface.

| Key | Value |
|---|---|
| `maxTotalBytes` | 3771619 |
| `maxFileBytes` | 524288 |
| `maxGrowthBytes` | 262144 |
| `exemptPathMaxBytes` | 512000000 |
| `exemptPathMaxFileBytes` | 100000000 |

Every editable path is enforced. Nothing is exempt.

`exemptPaths` is **absent** since 2026-08-26. The exemption existed for one
reason: to let head weights ride in a submission outside the source budget. A
submission carries no head weights any more, so there is nothing to exempt.

The two exempt caps stay declared. They cannot bind while `exemptPaths` is
absent. They stay because both enforcers carry the same two numbers as
compiled-in fallbacks, and this manifest is what holds those constants to a
reviewed value. `tools/lint-benchmark-manifest.py` check 3b enforces that
equality.

No head weight file is staged, so this budget never meets one. The head is part
of the pinned target checkpoint, which is outside the editable surface. What
the runner LOADS is bounded instead by the 2 GiB declaration cap in section 4.

### 3.3 What you may not edit

You may not edit anything that verifies, measures, or ledgers. This covers the
trusted harness, the target weights, the transform contract, the tokenizer, the
goldens, the gates, and the timing and telemetry code. `fixtures/` is outside
the editable surface. The scoring step reads the contract from the trusted
checkout for that reason.

### 3.4 The target quantization is frozen

The target model's quantization is frozen as shipped.

A submission must not re-quantize any target weight. It must not re-represent a
target weight. It must not change the numerical format of a target weight. This
holds even when the result passes every correctness gate.

`Sources/MLXFastTransform/` is editable. That does not license a change of
target format. A lossier target substitutes a degraded model. It does not
optimize the accepted one.

The MTP head is a narrow exception, and the exception is re-quantization only.

You may re-quantize the MTP head. You may **not** replace it. You may **not**
upload head weights of your own. Custom head weights are not accepted on this
track.

This is the 2026-08-26 ruling. It replaces the earlier bring-your-own-head
design, under which a participant could declare and ship a head of their own
choosing. That design is retired.

The head is the organizer's pinned weights, because it is part of the pinned
target checkpoint. `fixtures/qwen3_8_125b_a6b_track.json` names the repository
and revision, and `fixtures/reference_qwen3_8_125b_a6b_4bit.sha256` carries the
per-file digests that `./setup.sh` verifies every downloaded byte against. Both
files live in `fixtures/`, which is outside the editable surface.

Three things enforce this, and section 4 states each one:

1. No head weights directory exists and no submission path can hold head
   weights. A submission that carries a weight file is refused.
2. The head declaration accepts `"source": "pinned"` only. `"remote"` and
   `"in_branch"` are refused by name.
3. A re-quantization happens on load, in memory, on the benchmark machine.
   No re-quantized file is made, so there is no artifact to travel in a
   submission. Section 4.4 states the mechanism.

The loader reads a `quantization` block in the shape an MLX conversion writes.
That block selects which modules load quantized and at what geometry. The
accepted parameters are `group_size` (positive, at most 65536), `bits` between
2 and 8, and optional per-layer overrides (at most 8192 entries). A value
outside those bounds is refused by name.

The MTP head loader does not check a declare-versus-carry mismatch. An absent
declaration skips quantization, and packed weights then fail later inside the
weight bind with a shape error. A declaration with no packed tensor quantizes
nothing, silently. That limit is stated here rather than promised away.

The reason for the whole exception is the propose-and-decide split. The head
only proposes tokens. The pinned target model decides every emitted token.

## 4. The embedded MTP head

The track carries one speculative head. It is the organizer's weights, because
it is part of the pinned target checkpoint.

| Item | Value |
|---|---|
| Declaration | `mtp-head.manifest.json` |
| Where the weights are | Inside the pinned target checkpoint, under `language_model.mtp.*` |
| Organizer pin | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP@327c8a60` |

The declaration file is editable. The checkpoint is not.

Nothing stages a head weight file. There is no head stager and no head weights
directory. `./setup.sh` provisions the target checkpoint, and the head arrives
with it.

### 4.1 What you may declare

`"source": "pinned"` is the only accepted source. On this track it means the
head embedded in the pinned target checkpoint. A declaration may also state
`max_bytes` (it may lower the 2 GiB track cap and may not raise it), a `bytes`
count, and an optional `sha256`. It carries no `arm` key; there is one arm, so
there is nothing to select.

`"source": "remote"` is refused by name. `"source": "in_branch"` is refused by
name. Both were accepted before the 2026-08-26 ruling and both meant "load
weights the participant chose". The refusal names the retired source and names
`pinned` as what replaced it.

An absent declaration selects the embedded head. A declaration that is present
but broken is a refusal. The runner never falls back silently.

### 4.2 What you may not do

You may not ship head weights. No path in the editable surface can hold them,
so a submission that carries a weight file is refused before any measurement.
Yukon archives and overlays only `editablePaths`, so the file never reaches
the measured tree, and the benchmarker's own write-divergence gate refuses any
content that differs from the trusted baseline outside the editable surface.

You may not edit the checkpoint's head tensors. The checkpoint is not an
editable path, so any change to it is outside the surface.

### 4.3 What the size cap does and does not do

The 2 GiB declaration cap (`max_bytes` = 2147483648) bounds what the runner
loads.

The size cap is the only gate on the declaration. A declared `sha256` is
optional, and the runner does not verify it against the head bytes. It treats a
wrong digest and an absent digest alike. That is stated here plainly because it
is a real limit, not a detail: **nothing in this repository binds the loaded
head bytes to the organizer's pinned digests at run time.** The harness
computes a head digest that it reports and never compares.

What does bind at run time is relative, not absolute: the benchmarker compares
the candidate workspace against the trusted baseline workspace and refuses any
divergence outside the editable surface. A correctly provisioned baseline is
therefore load-bearing for the whole property.

The head bytes themselves are bound one level up. They are part of the pinned
target checkpoint, and `./setup.sh` verifies every downloaded byte against the
per-file digests in `fixtures/reference_qwen3_8_125b_a6b_4bit.sha256`. Both
legs load the head out of that one verified checkpoint.

### 4.4 How a re-quantization reaches the box

A re-quantization happens ON LOAD, in memory. Nothing on disk changes.

You do not make a re-quantized checkpoint. Your code quantizes the head's
parameters in memory, in the same pass that binds them. The staged bytes are
only read.

#### Where the seam is

The head is in `Runner/`, which is an editable path.
`Runner/Qwen4ExpMTP.swift` holds the head module.
`Runner/Qwen4ExpMTPDrafter.swift` holds the assistant that drives it.

The seam is `TrackQwen4ExpRunner.adoptMTPHead` in `Runner/Qwen4ExpRunner.swift`.
The function builds the head as the track's own module, selects the
quantization geometry, and then binds the tensors the loader read from the
checkpoint.

By default the function selects the CHECKPOINT'S OWN geometry. It reads the
group size, the bit width and the mode of each loaded projection, and it
applies the same values to the head it builds. The served head is therefore
bit-exact with the head the pinned fork builds: the same tensors, the same data
types and the same quantization parameters. The default path changes nothing a
run can measure.

To re-quantize the head, change the geometry that this function selects. Select
a different group size, a different bit width or a different mode, and quantize
the loaded projection again at that point. This is CODE, in an editable path,
and it travels in a submission as code. There is no configuration key, no
environment variable and no manifest key for it.

The policy stays as written. A re-quantization of the pinned head is permitted.
A replacement of the head is not, and head weights of your own are not. The
declaration accepts `"source": "pinned"` only.

The loader reads the checkpoint's own quantization block by default.
Section 3.4 states the bounds the loader accepts: `group_size` positive and at
most 65536, `bits` between 2 and 8, and at most 8192 per-layer overrides. A
value outside those bounds is refused by name. The loader does not check a
declare-versus-carry mismatch; section 3.4 states that limit.

#### Why nothing is written

Two properties follow from the in-memory rule, and both are why this mechanism
is the safe one.

1. The benchmarker compares the candidate workspace against the trusted
   baseline workspace and refuses any change outside the editable surface. That
   comparison reads the disk. A re-quantization on load is not a disk
   operation, so there is nothing for the gate to see.
2. The ranked worker runs under a sandbox profile that denies file writes. Code
   that tried to rewrite a staged head would fail there.

Do not rewrite the head tensors on disk. Do not rewrite them from `setup.sh` or
from the `mlxfast-swift transform` command. Each of those runs before the
workspace comparison, and the benchmarker refuses the change.

#### What changes in the record

The worker reports the digest of the head it loaded. That digest is the digest
of the ORGANIZER's checkpoint bytes, before and after a re-quantization,
because the bytes do not change. The geometry you selected is not visible in
that digest.

#### What this does not permit

The exception is for the HEAD's OWN weights only. The target model's
quantization stays frozen, as section 3.4 states.

**The head is embedded, so the rule is drawn by module path.** David ruling
2026-08-27, relayed by orchestrator: "Exempt mtp.* from the freeze."
`language_model.mtp.*` loads into the target's own module tree, so the
loaded-target check has to say which side of the line each module is on. It
says it this way:

| Module path | Treatment |
|---|---|
| `mtp.*` | EXEMPT. Re-quantize it on load. |
| `mtp.*` naming `embed_tokens` or `lm_head` | REFUSED BY NAME. |
| Everything else, `model.embed_tokens` and `lm_head` included | FROZEN. |

**The shared tensors are the middle row, and they are shared for a real
reason.** The head owns no embedding table and no output projection. It READS
the target's embedding table for its next-token vectors and the target's output
projection for its logits, which is what `use_dedicated_embeddings: false`
means on this checkpoint. Those two tensors decide the TARGET's tokens.
Coarsening one of them is a target re-quantization whatever path it is spelled
under, so a quantized module inside the head subtree that names either one is
refused, and the refusal says which shared tensor it reached.

The target is verified TWICE, and both checks read the loaded model, not the
declaration:

1. At worker startup, immediately after the target is loaded.
2. Again at the top of each window that gets measured, immediately before the
   measured work starts.

The second check exists because the first one alone verifies a model that code
can still change afterwards. Both refuse by name, and a refusal stops the worker
before any measurement.

The head only **proposes** tokens. The organizer-pinned target model decides
every emitted token. The serial control leg always runs the embedded head.

## 5. Scoring

### 5.1 The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

Each component is a gain:

```text
gain = baseline_leg_seconds_per_token / candidate_leg_seconds_per_token
```

The score is serial-anchored. A faster candidate scores above 1.

### 5.1.0 The pair is measured on the box, and no file stores it

**DAVID RULING 2026-09-08: EACH RANKED MACHINE HAS ITS OWN BASELINE.** A ranked
run measures PAIRS OF LEGS. It measures them on the SAME box, in the SAME job,
over the ONE prompt the fixture names in `live_golden`. The fixture's
`official_pairs` sets the count, and it is 2 (David ruling 2026-09-09). Every
pair is the same two legs in the same order:

1. The **serial-control leg**. It runs on the organizer's reference tree, which
   `MLXFAST_BASELINE_WORKSPACE` names. That tree is a build of this repository
   at the commit the fixture names in `baseline_reference_commit`. The leg uses
   no speculation. The measure script passes the serial tape to benchd as
   `--control-golden`, so this leg is verified against the serial tape and the
   candidate leg against the tape recorded at its declared depth.
2. The **candidate leg**. It runs on the submission tree, at its declared draft
   depth.

```text
score = (ref_prefill_spt / cand_prefill_spt) ^ 0.25
      * (ref_decode_spt  / cand_decode_spt ) ^ 0.75
```

The legs run STRICTLY ONE AFTER THE OTHER, and each leg loads the model once.
Per role the per-token times are SUMMED over the pairs, and each gain is the
ratio of those two sums. The floors, the ceiling and the acceptance bands apply
to that aggregate, not to one pair. Every control leg is checked against this
box's own baseline calibration (section 5.1.0.1).

**BOTH SPEEDUP FLOORS ARE 0.95** (David ruling 2026-09-09). A candidate that
regresses prefill or decode by more than 5 percent is refused. The fixture
declares them as `decode_speedup_floor` and `prefill_speedup_floor`, and the
benchmarker enforces the fixture's values. The ceiling stays 5.0.

**NO STORED PAIR EXISTS ANYWHERE.** Not in the scoring constants. Not in the
fixture. Not in a golden. A golden that carries
`benchmark.baseline_prefill_seconds_per_token` or
`benchmark.baseline_decode_seconds_per_token` is REFUSED on the ranked path.
`tools/lint-benchmark-manifest.py` check 5b keeps both fields out of this
repository.

The organizer stages the reference tree on each ranked box.
`tools/stage-baseline-workspace.sh` builds it there from a staged mirror or
bundle. The ranked job verifies the tree. It never fetches or builds it, because
the job holds no credential.

The reference tree carries its own engine, its own Metal library and its own
transformed weights. The candidate cannot move the control leg.

### 5.1.0.1 The per-box calibration is a health band

Each ranked box records what its own serial-control leg costs.
`MLXFAST_BASELINE_CALIBRATION` names the file.

**THE FILE IS A HEALTH BAND. IT IS NEVER THE DENOMINATOR.** The benchmarker
compares the measured control leg against the band. It stops the run by name
when the leg falls outside the band, and it seals no score. A stale calibration
file can stop a run. It can never move a score.

An operator writes the file on the box:

```bash
tools/calibrate-box.sh "<runner name>" /path/to/baseline-calibration.json
```

The command takes the box GPU lock. It then runs the serial-control leg four
times under the full official methodology: the cool gate before each pass, one
resident worker for each pass, and the same live golden the ranked run scores
over. The file it writes carries the values only:

```json
{
  "version": 1,
  "track_id": "qwen3.8-125b-a6b-mlx-v1",
  "box": "<the runner name>",
  "reference_commit": "<the fixture's baseline_reference_commit>",
  "prompt": "botany",
  "passes": 4,
  "prefill_seconds_per_token_mean": 0.0006282488193359375,
  "decode_seconds_per_token_mean": 0.0329116748046875,
  "prefill_cv": 0.004,
  "decode_cv": 0.002,
  "prefill_band_low": 0.95,
  "prefill_band_high": 1.05,
  "decode_band_low": 0.98,
  "decode_band_high": 1.02,
  "captured_at": "2026-09-08T00:00:00Z",
  "benchd_source_commit": "<40 hex>"
}
```

The calibrator writes no file when the coefficient of variation is more than
1 percent on either axis. A box that cannot repeat itself has no band worth
recording.

`tools/ranked-box-preflight.sh` refuses the run before any measurement when:

- either variable is absent;
- the workspace is not a git checkout at `baseline_reference_commit`;
- the workspace has no staged worker, no `mlx.metallib`, no fingerprint sidecar,
  or no transformed weights of its own;
- the calibration file does not parse, or its `version` is not 1;
- its `track_id` is not this track;
- its `box` is not this runner's `RUNNER_NAME`;
- its `reference_commit` is not the fixture's `baseline_reference_commit`;
- its `captured_at` is not later than the reference commit's date;
- any numeric value is not finite and positive;
- a band does not straddle 1 (`low < 1 < high`).

Each refusal names the failing thing.

**THE SCORED SHAPE IS SINGLE-STREAM.** David ruling 2026-08-27, relayed by
orchestrator: this track scores a PAIRED serial-against-MTP comparison, ONE
stream at a time, at `scored_batch_size` 1. The batched cohort adaptation is NOT
pursued. Section 11.4 states why: the batched path cannot run this model.

`aggregate` is a **sum over the pairs**, per role. Each gain is therefore a
RATIO OF SUMS, not a mean of per-pair ratios. The scored ranked run times the
ONE prompt `live_golden` names, on every leg.

The ruling this track is scored under, verbatim, dated 2026-08-27, and carried
in the fixture's `scoring_semantics.ruling_verbatim`: "score the qwen 3.8
125b-a6b tracks (mlx and cuda) single-stream, paired serial vs the built-in
mtp, on prefill gains ^ .25 * decode ^ .75".

`scoring.mode` is `qwen-native-mtp-paired-decode-only`, which is the
benchmarker's own single-stream paired regime name (`benchd`
`overlay::SCORING_MODE`). It names the measurement methodology, not the
formula.

### 5.1.1 Where the prefill window is, and why you cannot move work out of it

**THE VERBS DO NOT CHANGE.** The single-stream pair is still
`free_decode_begin` followed by `free_decode_run`. There is no new message and
no new field. The benchmarker splits its OWN parent clock at the verb boundary:

| Window | From | To |
|---|---|---|
| prefill | `free_decode_begin` sent | the validated `seed_token` comes back |
| decode | there | `free_decode_run(N)` returns |

`elapsed = prefill + decode`, and `seconds_per_token` is unchanged. Both legs
are bracketed identically.

**WHAT THE ENGINE OWES, and it is not optional:**

1. `free_decode_begin` runs the FULL seed prefill -- the golden's
   `decode_seed_tokens`, all 1024 of them, with the requested spec resolved --
   and replies only after that work has COMPLETED. The reply is the seed token
   (the greedy argmax after the whole seed) and the echoed `effective_spec`.
2. `free_decode_run` does NOT prefill and does not re-run any part of the seed.
   It decodes from the state `begin` left.
3. Nothing prefills before `free_decode_begin` arrives.
4. The hello advertises `free_run_decode` only.
5. Units are unchanged.

**MOVING PREFILL WORK INTO THE RUN IS NOT AN OPTIMISATION. IT IS A SCORING
DEFECT.** Deferred seed work does not disappear: it leaves the prefill window
and lands in the decode window. The whole window is unchanged, so `elapsed` and
`seconds_per_token` look identical -- but the composite weights the two windows
0.25 and 0.75, so shrinking prefill and growing decode by the same amount MOVES
THE COMPOSITE, and it moves it against you. The same applies in reverse to a
leg that did decode work early.

This is engine-side and unobservable on the wire, so it is held by tests rather
than by the protocol: `Tests/MLXFastTests/Qwen4ExpPrefillWindowTests.swift`
counts the tokens each verb pushes through the target and checks the
full-attention cache offsets at the verb boundary, on the serial leg AND the
mtp leg. The mtp leg is the sharper case: it feeds tokens it may take back, so
its forward count legitimately exceeds N, and what must hold is that its
offsets land exactly on `seed + N` -- rollback took back the drafts and nothing
else, and never re-prefilled.

### 5.2 The measured window

| Quantity | Value |
|---|---|
| Seed tokens per stream | 1024 |
| Checked decode steps | 128 |
| Golden shape | 1024 `prompt_tokens` and 129 `expected_tokens` |
| Streams per window | 1 |
| Timed prompts per leg | 1 (the fixture's `live_golden`) |
| Prompts in the pinned correctness pool | 8 |
| Prefill tokens per correctness-pool pass | 8 x 1024 |
| Pairs per ranked job | 2 (the fixture's `official_pairs`) |
| Legs per ranked job | 4 (each pair is serial control, then candidate) |

The correctness-pool rows are not the scored timing. The box stages all 8 pinned
prompts and `tools/ranked-box-preflight.sh` verifies all 8 against the fixture
pins. Each timed leg runs the ONE prompt `live_golden` names, and section 5.1.0
states what that timing produces.

The legs run one after the other, in one job. Each leg loads the weights once,
and each leg gets its own worker residency. The unmeasured warm-up prefill pass
stays at 1 pass on MLX, and it applies to every leg in the same way.

`MLXFastConstants.correctnessPromptTokens`, `benchmarkPrefillPromptTokens`, and
`benchmarkDecodeSeedTokens` all equal 1024. `benchmarkDecodeSteps` is 128.

Every timed leg runs on a cool, quiescent box. The ranked job waits for the
machine to go idle before the clock starts, and the benchmarker holds each timed
phase behind the fixed 40 C cool-down gate, which waits up to 900 s. The job refuses to measure at all
when the box has no GPU temperature reader, or when that reader returns a frozen
or implausible value. Only pairs accepted under that gate feed the composite.

### 5.3 The parameters

| Parameter | Value |
|---|---|
| `scoredBatchSize` | 1 |
| `prefillGainExponent` | 0.25 |
| `decodeGainExponent` | 0.75 |
| `pairsPerCohort` | 2 |
| `minPairsPerCohort` | 2 |
| `decodeSpeedupFloor` | 0.95 |
| `prefillSpeedupFloor` | 0.95 |
| `decodeSpeedupCeiling` | 5.0 |
| `kvBackend` | `contiguous` |

The scored run times ONE prompt at a time. There is no sweep and no per-run
choice of width. A width the benchmarker has not certified has no series tag,
and the benchmarker refuses that width rather than run it.

There is NO MEDIAN on this track. Each role's per-token times are summed over
the 2 pairs, and each gain is the ratio of those sums.

### 5.4 Token fidelity

The benchmarker applies a per-stream token-tolerance gate with a **10%
budget**.

This track does not require token-for-token equality with the serial
trajectory. The block-shaped forward pass diverges from the serial forward pass
at near-tie argmaxes. The gate prices that divergence against the 10% budget.
The gate accepts similar output. It does not certify lossless output.

### 5.5 Arming

`fixtures/qwen3_8_125b_a6b_track.json` sets `official_scoring_enabled` to
`true`. That flag is the SINGLE authority on this track's arm state, and it is
load-bearing. The pinned benchmarker reads it from the `--contract` fixture. It
refuses to seal an official scoring artifact while the flag is `false`. It also
refuses while the flag is absent, because it treats an absent flag as unarmed
rather than armed. No submission can publish an official score until that flag
flips.

The benchmarker, not the engine, produces the composite. It computes it from
benchd's own parent-clocked prefill and decode windows, summed over the pairs,
at the certified exponent pair. No
engine-reported value feeds it, and it does not depend on per-stream
instrumentation. Each record seals exactly one of `composite` and
`composite_absent_reason`. A composite is absent only when the record accepted
no pair, or when a window is degenerate, and the reason names which.

`tools/qwen38-125b-a6b-measure-and-score.sh` refuses with a non-zero exit
rather than emit a score. It does not substitute the shared-window
`raw_ratio_of_means` diagnostic for the ruled composite formula. Refuse, not
degrade, is the standing posture for this track. The `kv_backend` check and the
byte-budget check use it too.

The timed prompt pool is armed. All 8 `timed_prompt_pool[]` entries carry an
`r2_path`, a `sha256` and a positive byte count, and so does
`hidden_correctness_golden`. One per-depth oracle is pinned for each draft depth
1 to 6 under `live_golden_speculative`. The fixture also carries
`baseline_reference_commit`, which the reference tree must be at.
`tools/ranked-box-preflight.sh` refuses a contract that still carries the
`QWEN38-125B-A6B-MLX-PENDING-ORGANIZER` sentinel. The sentinel is matched
exactly; it is never a prefix test.

The six per-depth oracles hold the same bytes, and that is expected.
Verification is greedy and lossless, so the emitted token sequence does not
change with the draft depth. Each depth keeps its own file so that each depth
has its own pin slot.

### 5.6 Which goldens you can hold

| Object | Where it lives | Can you have it? |
|---|---|---|
| `correctness_prompts/public_longcopy_gate_english_1024_256.json` and `..._1024_1024.json` | Checked into git | **Yes.** They are already in your clone. See section 11.3. |
| `timed_prompt_pool[]`, 8 tapes | R2, at the `r2_path` keys the fixture pins. The ranked box stages them out of band into `MLXFAST_QWEN38_GOLDEN_DIR`. | **No.** They are organizer material and they are never in git. |
| `live_golden_speculative{}`, 6 per-depth oracles | The same: R2 keys, staged on the box. | **No.** Same material, same handling. |
| `hidden_correctness_golden` | The live golden, pinned by digest only. It is one of the staged files. | **No.** It is the token-fidelity oracle and it stays on the box. |
| The reference tree (`MLXFAST_BASELINE_WORKSPACE`) | Built on the ranked box at `baseline_reference_commit`. | **No.** It is the serial-control leg's engine. Its commit is public: the fixture names it. |

`tools/fetch-goldens.sh` is the organizer-side, pin-verified fetcher for R2
objects. It reads the R2 base from the environment variable
`R2_BUCKET_ENDPOINT` only. That value is secret-tier and is absent from this
repository. The script verifies the byte count first, then the sha256, and
deletes the file on either mismatch. It refuses to fetch anything the contract
declares hidden, and that guard fails closed when it cannot read the contract.

> **NOTE — this repository pins no public golden for that tool to fetch.**
> A participant has nothing to fetch with it today. Whether to publish a public
> local-calibration golden is an organizer decision.

The organizer stages the whole pinned set on a ranked box with the same tool.
`--all` reads the fixture, fetches every tape and every per-depth oracle, and
verifies each one against its `{sha256, bytes}` pin. It signs the requests with
the signer vendored at `tools/download-r2-object.sh`, so it needs R2 credentials
and refuses without them. A file that already matches its pin is left alone, so
the command is safe to re-run:

```bash
R2_BUCKET_ENDPOINT=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
  tools/fetch-goldens.sh --all --out "$MLXFAST_QWEN38_GOLDEN_DIR"
tools/ranked-box-preflight.sh
```

The preflight then verifies the staged directory against the fixture again and
refuses an extra `*.json` in it.

## 6. Running the benchmark

`benchmarkCommand` targets `benchd iterate --mode official` through
`tools/qwen38-125b-a6b-measure-and-score.sh`. That script is trusted-side
tooling. It is not an editable path, so a submission cannot rewrite the
measurement pipeline from inside its own archive.

The wrapped invocation is:

```text
benchd iterate --mode official \
  --engine .build/release/bench-worker \
  --weights <transformed weights> \
  --golden $MLXFAST_QWEN38_GOLDEN_DIR/<live golden>.golden.json \
  --contract fixtures/qwen3_8_125b_a6b_track.json \
  --baseline-workspace $MLXFAST_BASELINE_WORKSPACE \
  --baseline-calibration $MLXFAST_BASELINE_CALIBRATION \
  [--box <this box>] [--mtp-depth N] \
  --score-path score.json
```

`--engine` and `--weights` are RELATIVE to the checkout root. The benchmarker
re-roots both under `--baseline-workspace` to find the serial-control leg's own
worker and its own transformed weights. The script refuses an engine or a
weights directory outside the checkout: such a path has no relative form, and
the control leg would then run the candidate's own binary or read the
candidate's own transform.

`--box` is passed only on a hand run. On a runner the benchmarker reads
`RUNNER_NAME` itself and lets it win over the flag, so passing the flag there
would be argv that cannot matter. A hand run has no `RUNNER_NAME`, and the value
then comes from the calibration file's own `box`.

The measure script refuses by name when `MLXFAST_BASELINE_WORKSPACE` or
`MLXFAST_BASELINE_CALIBRATION` is absent on a real run. `--preflight-only`
requires neither: a participant runs it off the box.

benchd seals `score.json` itself, in the `{score, metrics}` shape the scorer
reads. This script writes no score and converts nothing.

**WHERE THE COMPOSITE LIVES ON THIS TRACK.** A single-stream run has no cohort
record, so benchd seals the composite ONE LEVEL UP: `results.json` carries
`composite` (`{composite_score, composite_speedup_floor,
composite_speedup_floor_met, decode_gain, prefill_gain}`) beside
`composite_scored_exponents`, and exactly one of `composite` /
`composite_absent_reason` is present. The published score is
`composite.composite_score`. benchd's own overlay publishes the same number
with the aggregation discriminator
`shared_window_composite_prefill_decode_gain` and a `single_stream_composite`
block carrying the two gains and the exponent pair.

The decode-only median (`aggregate.raw_decode_speedup_median`) is NOT the score
on this track. It is a different formula -- decode only, no prefill component,
no exponents -- and the emitter refuses a single-stream record that seals no
composite rather than publishing the median in its place. The median is
forwarded in `metrics` as a diagnostic.

Two drift tripwires run at that seam, because benchd reports a floor and an
exponent pair but wires neither to an exit code: the emitter REFUSES a run
whose `composite_speedup_floor_met` is false, and refuses a run whose sealed
`composite_scored_exponents` differ from `benchmark.json`
`scoring.scoredExponents`.

`preSubmitCommand` runs `./tools/qwen38-125b-a6b-measure-and-score.sh --preflight-only`.
That runs the arm gate and, when the live golden is staged, the integrity pin.
It exits without measuring, and it needs no reference tree.

The ranked pipeline is `.github/workflows/benchmark.yml`, which `benchmark.json`
`runner.workflow` names. It triggers on `workflow_dispatch` only. Its hosted
surface-check job gates its ranked job, which runs on the self-hosted labels
`[self-hosted, macOS, qwen3.8-125b-a6b-mlx-v1]` — the third label is the track id.
The ranked job holds no credential. The organizer stages the timed-pool tapes
onto the box out of band, from R2; they are never in the checkout. Before `./setup.sh` runs, `tools/ranked-box-preflight.sh`
verifies each tape against this track's `{sha256, bytes}` pins. One ranked run
occupies
the box at a time. A second dispatch queues rather than cancelling the first.

**ONE RESIDENT WORKER PER LEG, AND THE BENCHMARKER BOOTS IT.** A ranked job has
two legs on two trees, so one resident cannot serve both: the reference leg's
weights are not the candidate's. The benchmarker knows where a leg begins and
ends, so for each leg it calls that leg's OWN copy of `tools/resident-up.sh`:

```text
tools/resident-up.sh --boot --spec <serial|mtp> --draft-len <N> --socket-out <file>
tools/resident-up.sh --stop --socket <path>
```

`--boot` loads that tree's own `.build/release/bench-worker` and that tree's own
`weights/`, waits for a healthy hello, writes the socket path as the first line
of the `--socket-out` file, and exits 0 with the resident still running. A
`<socket>.pid` sidecar and a `<socket>.ready` marker sit beside the socket, and
`--stop` ends the resident and removes all three. Every per-phase
`bench-worker runtime-worker` the benchmarker starts attaches to that leg's
socket instead of loading the checkpoint again.

`--spec` is authoritative. Nothing in the boot reads the tree's
`mtp-head.manifest.json`, so a reference tree that declares a draft depth still
boots a SERIAL control leg when the benchmarker says serial.

`tools/qwen38-125b-a6b-measure-and-score.sh` boots no resident and exports no
socket. What it still owns is the window: it takes the box GPU lock
`/tmp/mtplx-gpu-exclusive.lock` and holds it for the whole measurement, because
a resident holds about 113 GB of unified memory whoever booted it and the box
needs exactly one loader. `tools/resident-up.sh` refuses to boot when nobody
holds that lock, so every per-leg boot happens inside that window.

**A `BENCH_WORKER_RESIDENT_SOCKET` IN THE ENVIRONMENT IS REFUSED.** It names one
already-loaded resident, so both legs would attach to it and the serial-control
leg would run on the candidate's weights. Public run 34230122059 failed exactly
that way, when the measure script still booted one resident from the candidate
tree and exported its socket: the reference leg attached to it and the
benchmarker refused the phase — "resident holds `<candidate>/weights` but this
phase asked for `<baseline-workspace>/weights`". The refusal was right and the
topology was wrong. Both `tools/ranked-box-preflight.sh` and the measure script
now refuse an inherited socket by name.

None of this changes what is scored. It removes repeated loads, not measured
work.

The wrapper form of `tools/resident-up.sh` (`--weights <dir> -- <command>`) is
kept for LOCAL, UNSCORED use. The ranked path has no caller for it.

`setupCommand` is `./tools/fetch-benchd.sh && ./setup.sh`. It chains no head
stager, because there is none. The checked-in `mtp-head.manifest.json` declares
`"source": "pinned"`, and the head arrives inside the target checkpoint that
`./setup.sh` downloads and verifies.

## 7. The pinned artifacts

| Artifact | Identity |
|---|---|
| Target model | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` @ `327c8a604de613b42f84ba5e6b796c0931e8aa3b` |
| Target manifest | `fixtures/reference_qwen3_8_125b_a6b_4bit.sha256` |
| MTP head | Embedded in the target checkpoint under `language_model.mtp.*` |
| Engine fork revision | `449f2d01b39f9088739c98d80a4f8a1b3cfa105e` |

`fixtures/reference_qwen3_8_125b_a6b_4bit.sha256` pins 32 files totalling
113,233,030,116 bytes, of which 22 are safetensors shards. The checkpoint holds
3747 raw tensors; the text tower holds 3414 of them and the skipped vision
tower holds 333. The checkpoint's own index reports a `total_size` of
113,209,155,128 bytes. `Sources/MLXFastCore/Constants.swift` mirrors the same
repository and revision pin.

The model repository is public and downloads without a token. There is no
organizer-hosted mirror for this checkpoint, so
`MLXFAST_REFERENCE_FALLBACK_BASE_URL` is empty by default.

Participants never supply the target weights. Substituting or re-deriving the
target is a failure.

The rectangular cap is `B * (1 + k) <= 8` on M3 and later. Batch size is locked
at 8.

You can select the draft depth. It is not pinned at 1.

Your drafter code sets the depth. That code is an editable path, so the depth is
a free lever. A request above the ceiling is clamped to the ceiling. It is not
refused.

The permitted values are 1 to 6. The engine limits the depth to 6. The embedded
head is served at depth 1 to 6
(`TrackQwen4ExpInlineMTPAssistant.maximumDepth`, in `Runner/`), and the depth
policy table has the same size. The runner's manifest declares depth `[1, 6]`,
and that manifest digest is a benchd conformance input, so the declared ceiling
does not move.

An absent depth does not mean 1. Two layers supply a depth when a request does
not name one. Do not confuse them.

| Request | Result |
|---|---|
| benchd invocation gives no `--mtp-depth` | benchd measures at depth 2 |
| the `mtp` block has no `depth` key | the engine uses its ceiling of 6 |

The `fixed_depth = 1` constant in the track fixture is a darkbloom reference
constant. It records a protocol value that this track inherited. It is not a
limit on your draft depth.

Every run seals the depth that operated. Read `effective_spec` for the depth the
run declared. Read `effective_mean_draft_len` for the draft length that the run
realized. The two can differ: a run can declare depth 2 and realize a mean draft
length near 1.

## 8. Prohibited techniques

A submission that uses any of these fails the static review.

- A cache or memo keyed on a request's input tokens whose only possible hit is
  the harness repeating one identical computation. Bit-identical output does
  not make it legitimate. The benchmark measures single-pass inference. An
  optimization must save work that recurs in single-pass production inference.
- Hardcoded hidden prompts, hidden token identifiers, or answers.
- Timing shortcuts, protocol injection, network access, and filesystem
  exfiltration.
- Any change outside `editablePaths`.

Input-independent caching stays legal. This covers weights, dequantized
tensors, and RoPE or mask tables keyed on shapes and offsets. Within-request KV
reuse stays legal.

Keep every change prompt-independent and model-general. The hidden prompts
differ from the public fixtures.

## 9. Submitting

Use the Yukon CLI for every account operation and every submission operation.
`README.md` holds the commands.

A submission archive packages only `editablePaths`. It rejects generated
artifacts, symlinks, local scores, reference checkpoints, and any source change
outside the editable surface. `yukon submit` does not run a local test first,
and no local run blocks the upload.

The ranked run on the official runner is the gate that ranks a submission.

## 10. License

The pinned checkpoint carries the Qwen Community License 1.0. That is the
license name the checkpoint records, not an SPDX identifier: the contract
fixture records `"spdx": "other"` with `"license_name": "qwen-community-1.0"`.
The terms ship with the checkpoint at its pinned revision.

The checkpoint is a 4-bit MLX conversion of `Qwen/Qwen3.8-Flash-Next`.

This repository distributes no model weights.

## 11. The state of the track

Read this section before you conclude that something is broken.

### 11.1 The bench channel

Section 5.5 states the arm state: the track IS armed.

The bench release branch and dist channel are `qwen3.8-125b-a6b-v1`, which is
the PROJECT name, not this track's id. David ruling 2026-08-27: the MLX and
CUDA tracks of this model share one benchmarker, so they share one channel. The
track id `qwen3.8-125b-a6b-mlx-v1` is unchanged and still names the leaderboard
namespace, the runner labels and the R2 prefix.

THE CHANNEL RESOLVES FROM THE RELEASE BRANCH. `./tools/fetch-benchd.sh` reads
`dist/benchd.manifest.json` at the tip of branch `BENCHD_BRANCH` (default
`qwen3.8-125b-a6b-v1`), checks the manifest's `branch` against that channel, and
installs the binary only when it matches the `sha256` and `bytes` the manifest
names. The script then prints the identity it resolved and keeps the manifest
beside the binary in `benchd-bin/`.

THAT MANIFEST IS THE SOURCE OF TRUTH for which `benchd` measures your run,
and it moves when the organizer republishes dist. This document therefore names
no `{source_commit, sha256, bytes}` triple as current: run the script and read
the identity line.

The channel host is the public bench repository `Layr-Labs/mlxfast-bench`, so
`./tools/fetch-benchd.sh` needs no token. Set `BENCHD_DIST_TOKEN` (or
`GITHUB_TOKEN`) only when you point `BENCHD_DIST_BASE_URL` at a private mirror.
An air-gapped box takes a verified pair through `BENCHD_DIST_LOCAL` instead.

### 11.2 The model port HAS landed

The engine constructs, gates, loads and runs `qwen4_exp_text`. The geometry in
`Sources/MLXFastCore/Constants.swift`, the checkpoint validator in
`Sources/MLXFastTransform` and this contract's `target.*` block are all this
target's, and they move as ONE SET: a gate holding some fields of one model and
some of another rejects every checkpoint and explains none of them.

The model tower and the n-gram row source are in the `Vendor/mlx-swift-lm`
submodule. They are not editable paths in this repository; see section 3. The
Runner and the MTP head are in `Runner/`, which is editable.

What remains is named in 11.4 and 11.5: the batched path refuses, and the
speculative arm is correct but not yet fast.

### 11.3 The checked-in goldens are REGENERATED and they load

The two `correctness_prompts/*.json` goldens were regenerated on 2026-08-28 on
ranked hardware, against the pinned target
(`Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` @ `327c8a604de613b42f84ba5e6b796c0931e8aa3b`),
carrying `model_type` `qwen4_exp_text`. They were double generated -- a fresh
process each, byte-identical before either was pinned -- and they LOAD through
the model-identity loader.

So `./benchmark.sh --local-iterate` reaches a golden, and the local public
drift gate can pass. The PROMPT file is unchanged; only the expected tokens and
the provenance block moved.

These are the PUBLIC goldens. Section 5.5 remains the authority on the arm
state and on the pinned hidden oracle.

**A GOLDEN MUST CARRY `model_provenance`.** The block names the repository and
the revision of the pinned model. `loadQwenGoldenFixture` is the loader that
every golden consumer in `Sources/` calls, and it refuses a golden that carries
no block. The error message names `model_provenance`. Before this rule, the
loader read the block when it was present and skipped the check when it was
absent, so a golden that named no checkpoint passed. The model-agnostic
`loadGoldenFixture` keeps the reference schema, which has no such key.

### 11.4 The batched path REFUSES, and the ruling is single-stream

**(a) The batched cohort path refuses by name.** `makeCohortEngine` throws.
There are TWO blockers and the second is decisive:

1. The QSA sparse attention emits a custom array mask, and the
   ContinuousBatchingV2 path owns the attention call and discards a custom
   mask. A cohort engine would serve DENSE attention under a model trained
   sparse.
2. A ContinuousBatchingV2 layer is full attention or a sliding window. On this
   tower 36 of the 48 layers carry a constant-size RECURRENT state and NO
   key-value tape, so three quarters of the model has no shape in that engine's
   cache bank. This holds at EVERY context length, so no budget or window
   check avoids it.

The engine also stops ADVERTISING the batched capability in its hello, so the
benchmarker refuses at its pre-measurement capability check rather than after
it has sent a batched begin. There is no dense-attention fallback: below the
indexer budget a cohort engine would look correct and would diverge exactly
where the score is measured, so a fallback is worse than a refusal.

**(b) That question is RULED, and the ruling is single-stream.** David ruling
2026-08-27, relayed by orchestrator: this track is scored single-stream, and the
ContinuousBatchingV2 adaptation is not pursued. The fixture therefore pins
`scored_batch_size` 1 and `scoring.mode`
`qwen-native-mtp-paired-decode-only`, and section 5 describes a single-stream
paired series.

**The bench-side dependency is MET.** The benchmarker keeps
`scored_batch_size` 1 on the single-stream regime (it never reaches the cohort
width match), and it seals the composite on the single-stream series at the top
level of the record beside `composite_scored_exponents`
(`prefill_gain_exponent` / `decode_gain_exponent`). `./tools/fetch-benchd.sh`
resolves whatever the channel manifest names (section 11.1). The width
certification does not refuse the shape this fixture declares, and scored runs
happen.

**(c) The mtp arm can now be faster than serial, and whether it is depends on
your drafter.** Its verify runs at the draft depth (see 11.5), so a round pays
one target forward for its whole chain rather than one per committed token. It
still pays the depth head forwards that proposed the chain, and one full-stack
snapshot per round, so an accepted draft is what buys the target forward back.
Making it fast is the point of the track.

**WHAT "CORRECT" MEANS FOR THIS ARM, stated precisely, because an earlier
wording overstated it.** This section used to say the arm is "token-exact
against the serial control". That is a FIXTURE-PROVEN property, not a
pinned-weight one, and the two are not the same claim:

* ON THE FIXTURE, token equality with the serial leg is asserted by test, at
  every depth the envelope permits.
* ON THE PINNED WEIGHTS, the recorded population is five near-tie argmax flips
  among the 1,728 non-row-0 rows compared, with ZERO among the 576 row-0
  samples. So the mtp stream MAY diverge from the serial stream at a near-tie
  row.

Under the ruled semantics that divergence is NOT an error. The verify runs at
the draft depth, and the wide forward is the oracle: a committed token is
correct when it matches what that forward says, not when it matches what a
one-token-at-a-time decode would have said. Section 5.4 is the gate that prices
any resulting difference in emitted tokens, and it already says this track does
not require token-for-token equality with the serial trajectory. The engine's
`docs/qwen38-125b-a6b-port-notes.md` section 5.2.1.4 has the measured
population.

### 11.5 The verify runs at the draft depth

A speculative round verifies its whole draft chain in ONE target forward. The
verify width is the resolved draft depth on the mtp leg; the serial control
still runs one token at a time.

**THE CAP THIS SECTION USED TO DESCRIBE IS GONE (David ruling 2026-08-28).**
It existed because a measurement said a multi-token forward disagreed with the
same tokens fed one at a time, and that measurement named the keep mask as
ruled out. The keep mask WAS the cause: the QSA indexer computed its
complete-block count with true division instead of floor division, so the mask
let a query attend to future keys inside its own partial block, and how many
depended on the segment width. That, a wrong RMSNorm convention for this
checkpoint, and a vendored quantized-gather defect were all fixed, and the
survey was re-run on ranked hardware against the fixed engine.

**WHAT THE WIDE VERIFY RESTS ON.** Not bit-identity -- MLX dispatches a
different kernel at one row than at several, by design, so the logits differ in
their last bits. It rests on ARGMAX AGREEMENT: the wide forward picking the
same tokens.

Two separate pieces of evidence, and they are not interchangeable. On the
FIXTURE, a test asserts that the speculative leg commits the serial leg's
stream token for token at every permitted depth. On the PINNED WEIGHTS, the
re-survey found argmax agreement on every one of its 576 row-0 samples, and
five near-tie flips among the 1,728 non-row-0 rows -- so a wide verify may
commit a token a one-at-a-time decode would not have, at a near-tie. That is
the oracle doing its job, not a defect: see 11.4(c).

**WHAT THIS MEANS FOR YOU.** The arm is no longer strictly more work than
serial for the same output: a round pays one target forward for its whole
chain instead of one per committed token. Whether that becomes a speedup on the
ranked box is a measurement, not a promise, and it depends on the drafter -- an
accepted draft is what buys the forward back.
