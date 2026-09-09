# mlxfast — Qwen 3.8 125B A6B MLX

This repository is the engine for the Qwen 3.8 125B A6B MLX speedup benchmark.
The track identifier is `qwen3.8-125b-a6b-mlx-v1`.

## What this repository is

This repository holds the engine. The engine is the Swift and Metal code that
runs the target model on Apple Silicon. You optimize the engine. You make the
model do the same work in less time.

The benchmarker measures the engine. The benchmarker is a separate program
called `benchd`. It arrives as a verified prebuilt binary. It owns all
timing, all scoring, and all gates. Nothing in this repository measures or
scores anything.

The ranked run runs ONE STREAM AT A TIME (David ruling 2026-08-27:
"Single-stream only"). The ranked run measures your engine and a serial control
engine in the same session, on the same box. The score compares the two. The
control engine is the organizer's reference tree, and its cost is measured in
that same session. No file stores it.

**THE SCORED RUN TIMES ONE PROMPT.** Both legs time the prompt the fixture names
in `live_golden`. The pool of 8 pinned prompts is the CORRECTNESS pool: the box
stages all 8 and the preflight verifies all 8, but the timed leg runs the one.

> **NOTE — the track goldens are not in this repository.**
> The 8 timed-pool tapes and the 6 per-depth oracles are organizer material.
> They are published in R2 at the `r2_path` keys the contract pins. The ranked
> box stages them out of band into the directory its runner service exports as
> `MLXFAST_QWEN38_GOLDEN_DIR`. `tools/ranked-box-preflight.sh` verifies every
> file there against the contract's `{sha256, bytes}` and refuses an extra
> `*.json`. They are never in git, so your clone does not carry them.
>
> The organizer stages them with the signer this repository vendors:
>
> ```bash
> R2_BUCKET_ENDPOINT=... R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... \
>   tools/fetch-goldens.sh --all --out "$MLXFAST_QWEN38_GOLDEN_DIR"
> tools/ranked-box-preflight.sh
> ```

> Official scoring is armed and the ranked box is registered. Section
> [Current status](#current-status) states each fact.
> `docs/qwen38-125b-a6b-port-notes.md` is the engineering record.

### Lineage

This repository descends from `Layr-Labs/mlxfast-qwen-38-27b-mtp-engine`, which
descends from `Layr-Labs/mlxfast-challenge-dev`. Those repositories rank
different models under different rules. Only this track's rules apply here.

A few fixtures and transform validators carry `Qwen 3.6` or `Laguna` in their
names. Those names point at real foreign checkpoints on purpose. They are the
negative controls and the fixture substrate that this track's own gates are
tested against.

## Requirements

- An Apple Silicon Mac.
- Enough unified memory for the target model and its working set. The target
  checkpoint is 113,233,030,116 bytes across 32 pinned files, of which 22 are
  safetensors shards. Add the KV cache and the decode buffers on top of that.
- At least 260 GiB of free disk space before the download starts. Change this
  limit with `MLXFAST_REFERENCE_MIN_FREE_GIB`.
- macOS 14 or later. `Package.swift` sets that platform floor. CI builds on a
  `macos-26` runner, and the Metal toolchain policy in `setup.sh` treats
  macOS 26 as its own case.
- Swift 6, through Xcode or through the Xcode Command Line Tools.
  `Package.swift` declares `swift-tools-version: 6.3`.
- The Xcode Metal Toolchain. `./setup.sh` tries to download it. Some users need
  full Xcode. Install it, open it once, and accept the license with
  `sudo xcodebuild -license accept`.
- CMake. `./setup.sh` installs it through Homebrew when it is missing.
- Git.

You do not need Rust. The benchmarker arrives as a prebuilt binary.
`./tools/fetch-benchd.sh` resolves it from the `qwen3.8-125b-a6b-v1` dist channel and
verifies it against the channel's `benchd.manifest.json`. That channel is the
public bench repository `Layr-Labs/mlxfast-bench`, so the fetch needs no token.

## Quickstart

Run these commands in order. One sentence describes each command.

```bash
git clone <repository-url> mlxfast-qwen38-125b-a6b-engine
```

This command copies the repository to your machine.

```bash
cd mlxfast-qwen38-125b-a6b-engine
```

This command makes the repository your working directory.

```bash
./tools/fetch-benchd.sh
```

This command resolves the benchmarker binary from the dist channel into
`benchd-bin/` and verifies its sha256 and its byte count against the channel's
`benchd.manifest.json` (installed beside the binary).

```bash
./setup.sh
```

This command checks your toolchain, initializes the pinned engine submodule,
builds and stages the Swift binaries and `mlx.metallib`, downloads and verifies
the target model, and transforms it into the `weights/` tree the engine loads.
Set `MLXFAST_WEIGHTS_PATH` to choose another output directory. Setup reruns the
current transform even when the reference download is cached; it reports
success only after the transform produces a fresh output tree. A failed transform
leaves previous weights in place. Run setup while the engine is idle, because
publishing a replacement directory briefly moves the previous tree aside.
The emitted `config.json` declares
`rms_norm_weight_offset`, so the tree carries this checkpoint's RMSNorm
convention to any consumer that reads the file, and transform verification
refuses a tree that does not declare it.

```bash
./tools/local-baseline.sh
```

This command runs correctness and local timing against the checked-in public
golden through `./benchmark.sh --local-iterate`, with the normal cool gate.
It selects the staged worker and writes `score.local-iterate.json`. This is an
unranked measurement: a `null` score is expected without paired ranked scoring;
check the run's exit status, correctness result, and timing metrics. The command
also works when called from outside the checkout.

After `yukon clone`, change to the printed work directory. `yukon setup` runs
the repository's setup command. Once `weights/` is prepared as described above,
run `./tools/local-baseline.sh`. `yukon run` is the ranked entry point for this
challenge and requires organizer-staged goldens, the reference workspace, and
the box calibration. Those assets are unnecessary for this public local baseline.

The helper clears inherited `MLXFAST_BASELINE_WORKSPACE` and
`MLXFAST_BASELINE_CALIBRATION` for its child process, so an operator shell's
ranked settings cannot select paired scoring here.

There is no head-staging step. The MTP head ships inside the pinned target
checkpoint. See [The MTP head is embedded](#the-mtp-head-is-embedded).

`tools/local-baseline.sh --help` lists the optional environment overrides for
the worker, golden, weights, and result path. Relative overrides resolve from
the checkout. Direct `./benchmark.sh` calls still require an explicit
`MLXFAST_CORRECTNESS_GOLDEN_PATH` and `MLXFAST_ENGINE_BIN`; the local helper
supplies their documented defaults.

> **WARNING — do not pass `--golden`, `--weights`, or `--score-path` to
> `./benchmark.sh`.**
> The script rejects these flags. Use the environment variables instead.
> `MLXFAST_WEIGHTS_PATH` defaults to `weights`.

### The two public goldens

| File | Purpose |
|---|---|
| `correctness_prompts/public_longcopy_gate_english_1024_256.json` | The drift tripwire. 1024 prompt tokens and 256 expected tokens. |
| `correctness_prompts/public_longcopy_gate_english_1024_1024.json` | The local-submit golden. 1024 prompt tokens and 1024 expected tokens. |
| `correctness_prompts/public_longcopy_gate_english_1024.txt` | The prompt text the two goldens tokenize. |

> **NOTE — the goldens are regenerated against the pinned target and they
> load.**
> The two `.json` goldens were regenerated on 2026-08-28 on ranked hardware
> against `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`, double generated and
> byte-identical before pinning, and they pass the model-identity loader. So
> `./benchmark.sh --local-iterate` reaches a golden. The prompt file is
> unchanged. The HIDDEN correctness oracle is a separate file and is still the
> pending sentinel.

> **NOTE — the local test needs the n-gram shards.**
> `./benchmark.sh --local-iterate` passes the n-gram shard directory from the
> track fixture (`ngram_shard_dir`) to the worker as the `qwen4exp.ngramRowSource`
> resource. Run the transform first so the shards exist, or attach to a resident
> with `BENCH_WORKER_RESIDENT_SOCKET` (see `tools/resident-up.sh`).

> **NOTE — a golden must name the checkpoint it came from.**
> Each `.json` golden must carry a `model_provenance` block. The block names
> the repository and the revision of the pinned model. The loader refuses a
> golden that carries no block. The error message names `model_provenance`.
> The loader does not skip the check when the block is absent.

## Repository structure

| Path | What it holds | Status |
|---|---|---|
| `Runner/` | The track Runner: the Qwen 3.8 Flash-Next model family code and its manifest. | Editable |
| `Sources/BenchWorker/` | The `bench-worker` shim. It registers the Runner in `Runner/`. | Trusted |
| `Sources/MLXFastTransform/` | The offline transform that writes `weights/`. | Editable |
| `Sources/MLXFastCLI/` | The trusted CLI, `mlxfast-swift`. | Trusted |
| `Sources/MLXFastCore/` | Shared constants and contracts. | Trusted |
| `Sources/MLXFastTrustedHarness/` | The editable-surface budget, the head declaration reader, transform verification, and the metallib fingerprint. | Trusted |
| `Vendor/mlx-swift/` | The pinned MLX fork, a vendored tree. The listed Metal kernel sources are editable. | Mixed |
| `Vendor/mlx-swift-lm/` | The engine fork, a git submodule. It holds the engine core. | Submodule, not editable |
| `fixtures/` | The track contract and the pinned checkpoint manifests. | Trusted |
| `tools/` | Setup, build, lint, and measurement scripts. | Trusted |
| `benchd-bin/` | Where `./tools/fetch-benchd.sh` installs the verified binary. Git ignores it. | Fetched |
| `mtp-head.manifest.json` | The MTP head declaration. It declares; it carries no weights. | Editable, optional |
| `correctness_prompts/` | The public prompt and the two public goldens, for local runs. The track goldens are NOT here: they live in R2 and on the ranked box. | Trusted |
| `weights/` | The transformed weights the engine loads. | Generated |
| `benchmark.json` | The Yukon track manifest. It lists every editable path. | Trusted |

### Starting a new track from this repository

This repository is the template for the next track. Seed a new repository with a
copy of this tree, then run `tools/new-track.sh` in it. The script stamps the new
identity into the manifest, the contract fixture, the checkpoint file list, the
runner label, the engine pin and the docs. It touches no golden: a track's
goldens live in R2 and on its own box, never in git. It never commits. `docs/new-track-repo-procedure.md` holds the full procedure and
the usage line.

### The engine is a submodule

`Vendor/mlx-swift-lm` is a git submodule. It points to one commit of
`Layr-Labs/mlx-swift-lm`. That fork holds the engine core:

* the Qwen 3.8 Flash-Next model;
* the runner boundary and the batching engine;
* the shim body that `Sources/BenchWorker/` builds on.

The submodule is not editable.

The Runner is NOT in the submodule any more. `Runner/` in this repository
holds it, `Sources/BenchWorker/` registers it, and the registry gives a later
registration the claim on the model type, so `Runner/` shadows the fork's
built-in runner. This repository builds its own `bench-worker` from that pair.
The binary keeps its name and its staged location: benchmarking starts
`.build/release/bench-worker`, as before.

Clone the repository with `--recurse-submodules`. If you cloned it without that
option, run `git submodule update --init`.

The pin is `449f2d0`, on branch `feat/qwen38-flash-next-runner`. That branch is not
merged. Re-pin the submodule to the fork's `main` after the branch merges.

`Vendor/mlx-swift` stays a vendored tree, because its Metal kernel sources are
editable. The fork asks for `mlx-swift` from the network. SwiftPM uses this
package's local `Vendor/mlx-swift` instead, because a local path dependency of
the root package wins. SwiftPM prints a "conflicting identity" warning when it
does this. The warning is expected.

The copy is `Layr-Labs/mlx-swift` at `6b0505cc`, which is MLX 0.32.2. It holds
the two nested submodules of that repository as plain files: `Source/Cmlx/mlx`
at `734241bb` and `Source/Cmlx/mlx-c` at `9ff12fab`. `6b0505cc` is the core
that Darkbloom builds the same engine fork against, and the fork prefers a
sibling `../mlx-swift` checkout over the network, so this copy is the core the
fork compiles against. One runner builds against one core. To move the pin,
copy the whole tree again from a fresh checkout of the new commit, with its
submodules, then rebuild the metallib.

The n-gram table is 29.8 GiB, and it is never model parameters. The runner
builds a disk-resident row source from a resource that the caller supplies. The
benchmarker starts the engine with that resource:

```
bench-worker runtime-worker --weights <dir> \
  --resource qwen4exp.ngramRowSource=<shard directory>
```

`fixtures/qwen3_8_125b_a6b_track.json` `ngram_shard_dir` names the directory.
`tools/qwen38-125b-a6b-measure-and-score.sh` forwards it. Without the resource
the runner refuses at load, and it names what is missing.

### The MTP head is embedded

This track has exactly one speculative arm. It is the native MTP head, and the
head ships inside the pinned target checkpoint.

| Property | Value |
|---|---|
| Tensor prefix | `language_model.mtp.*` |
| Tensors | 76 |
| Hidden layers | 1 |
| Attention | Hybrid, full attention |
| Embeddings | None of its own. It rides `language_model.embed_tokens`. |
| Output head | None of its own. It rides `language_model.lm_head`. |

Nothing stages a head weight file. There is no head stager script and no head
weights directory. No submission carries head weights, and the ranked box
stages none.

`./setup.sh` provisions the target checkpoint, and the head arrives with it.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists 71 entries. The
rule behind the list is simple. Code that **proposes** tokens or computes the
forward pass is editable. Code that **verifies**, **measures**, or **ledgers**
stays trusted.

The editable surface has four groups.

1. The head declaration. `mtp-head.manifest.json`. The declaration file only.
2. The Runner. `Runner/`.
3. The offline transform. `Sources/MLXFastTransform/`.
4. The 68 vendored MLX Metal kernel files the forward pass dispatches. These
   are the quantized matmul, the MoE gather-GEMM, SDPA and steel attention,
   RoPE, RMSNorm, softmax, sort, reduce, copy, elementwise, `arg_reduce`, and
   gather indexing.

### The Runner

The Runner in `Runner/` is editable. It is the model family's code: it loads
the checkpoint, it declares the manifest, and it builds the engine and the
one-row stepper. It also holds the MTP head and the assistant that drives it,
in `Runner/Qwen4ExpMTP.swift` and `Runner/Qwen4ExpMTPDrafter.swift`. `Sources/BenchWorker/` registers it in `RunnerRegistry`
before the engine resolves a runner, so it SHADOWS the fork's built-in runner
for `qwen4_exp` and `qwen4_exp_text`. The `Vendor/mlx-swift-lm` submodule is
not editable: it holds the engine core, a gitlink names a commit and not
bytes, and whether a submission may repoint it at its own fork commit is not
ruled yet.

Keep the manifest as it is. The runner manifest digest is a benchd
conformance input, and a changed digest fails the conformance check.

### The MTP head declaration

The MTP head is the organizer's pinned weights, because it is part of the
pinned target checkpoint. You may re-quantize it. You may not replace it, and
you may not upload head weights of your own. Custom head weights are not
accepted on this track.

`mtp-head.manifest.json` is a declaration. It stays editable and optional. It
accepts `"source": "pinned"` only, which on this track means the head embedded
in the pinned target checkpoint. `"source": "remote"` and `"source":
"in_branch"` are refused by name. The declaration carries a 2 GiB cap
(`max_bytes` = 2147483648); a declaration may lower it and may not raise it.

An absent declaration selects the embedded head. That is the normal case. A
declaration that is present but broken is a refusal. The runner never falls
back silently.

The size cap is the only gate on a declaration. A declared `sha256` is
optional, and the runner does not verify it against the head bytes.
`docs/participant-contract.md` section 4.3 states that limit plainly.

A re-quantization happens ON LOAD, in memory. Nothing on disk changes. The head
module and the assistant that drives it are in `Runner/`, which is editable:
`Runner/Qwen4ExpMTP.swift` and `Runner/Qwen4ExpMTPDrafter.swift`. The seam is
`TrackQwen4ExpRunner.adoptMTPHead` in `Runner/Qwen4ExpRunner.swift`, which
selects the quantization geometry of the served head. By default it selects the
checkpoint's own geometry, so the served head is bit-exact with the head the
pinned fork builds. To re-quantize, change the geometry that function selects.
A replacement of the head is still refused, and head weights of your own are
still refused. `docs/participant-contract.md` section 4.4 is the authority.

A head only **proposes** tokens. The pinned target model decides every emitted
token. The serial control leg always runs the embedded head.

### Batch size and draft depth

> **NOTE — the scored batch size is locked. Draft depth is not.**
> The scored batch size is 1. It is not a tunable.
>
> The draft depth is a free lever, and it is not pinned at 1. You declare it
> in `mtp-head.manifest.json`, which is editable:
>
> ```json
> "spec": { "enabled": true, "num_speculative_tokens": 3 }
> ```
>
> Select a depth from 1 to 6 (the contract's `permitted_draft_depths`). With
> no `spec` block, `enabled: false`, or `0`, the run is serial (depth 0). A
> depth outside 1 to 6 is refused before the engine starts, not clamped.
> `tools/spec-declaration.sh describe` prints what your tree declares.
> The ranked run sends the depth to the engine on each request and seals
> the engine's `effective_spec` echo.
>
> A declared depth is scored against the oracle recorded AT THAT DEPTH
> (`fixtures/qwen3_8_125b_a6b_track.json` `live_golden_speculative`, the
> `botany.mtpN.golden.json` tapes). The runner verifies a draft window in one
> target forward, and the multi-row kernels round differently from the
> single-row ones, so a speculative run is not token-identical to the serial
> tape; it is token-identical to its own depth's tape, which the organizer
> records on the pinned runner. Correctness at a depth means matching that tape.
>
> Every run seals what actually ran: `effective_spec` for the declared depth,
> `effective_mean_draft_len` for the realized draft length.

The local modes (`./benchmark.sh --local-iterate` and `--local-submit`) read
the same declaration as the ranked entrypoint. They request that depth for
the timed decode window and refuse a benchmarker that cannot honor it. Before
dispatch, the wrapper prints the requested mode, draft depth, single-stream
batch size, worker SHA-256 and checkout revision. The checkout revision is
not the worker's build revision; benchd records the latter at runtime and
verifies the worker's `effective_spec` echo.

The public local fixture still checks teacher-forced correctness. Selecting
MTP locally does not add the organizer's per-depth oracle or produce a ranked
score. Compare timings from the same effective mode and depth.

The rectangular cap is `B * (1 + k) <= 8` on M3 and later.

### The byte budget

`benchmark.json` `editableSurfaceByteBudget` caps the enforced editable
surface.

| Key | Value |
|---|---|
| `maxTotalBytes` | 3771619 |
| `maxFileBytes` | 524288 |
| `maxGrowthBytes` | 262144 |
| `exemptPathMaxBytes` | 512000000 |
| `exemptPathMaxFileBytes` | 100000000 |

Every editable path is enforced. Nothing is exempt.

`exemptPaths` is absent. The exemption existed to let head weights ride in a
submission outside the source budget. A submission carries no head weights, so
there is nothing to exempt. The two exempt caps stay declared because both
enforcers carry the same numbers as compiled-in fallbacks and this manifest is
what holds them to a reviewed value.

### The target quantization is frozen

The target model's quantization is frozen as shipped. Do not re-quantize a
target weight. Do not re-represent one. Do not change the numerical format of
one. This holds even when the result passes every correctness gate.

`Sources/MLXFastTransform/` is editable. That does not license a change of
target format. A lossier target substitutes a degraded model instead of
optimizing the accepted one.

The MTP head is a narrow exception, and the exception is re-quantization only.
You may re-quantize the head. You may not replace it. The head stays within its
2 GiB declaration cap. The head only proposes tokens, and the pinned target
decides every emitted token.

### What you must not change

- Everything in `Sources/` that `editablePaths` does not list.
- `Package.swift` and `Package.resolved`. The dependency graph is frozen.
- Everything in `Vendor/` that `editablePaths` does not list.
- `fixtures/`, `benchmark.json`, the scripts, the tests, and the
  documents.
- `weights/`, the reference checkpoints, the scores, and the goldens.

Do not hardcode hidden prompts. Do not hardcode hidden token identifiers. Do
not use timing shortcuts, protocol injection, network access, or filesystem
exfiltration.

Do not add a cache keyed on a request's input tokens whose only possible hit is
the harness repeating one identical computation. The benchmark measures
single-pass inference. Input-independent caches stay legal. These are weights,
dequantized tensors, and RoPE or mask tables keyed on shapes and offsets.
Within-request KV reuse also stays legal.

## Local testing vs the ranked run

The local test and the ranked run are different by design. Read this section
before you tune.

The local test runs a **single stream**. It uses a public golden. It prints a
single-stream estimate.

### What each local mode checks

Both local modes run one fused checked-timing pass. The pass teacher-forces the
golden's expected tokens and times the wall clock. It judges correctness from
that same pass. A mismatch is reported as a teacher-forced token mismatch.

A correctness failure does not discard the timing. The benchmarker reruns the
timing phase in a mismatch-tolerant form. It then reports the correctness
failure together with real timing numbers.

| Mode | Decode steps | Expected tokens the golden must hold | Cool gate |
|---|---|---|---|
| `--local-iterate` | 128 | 129 | On, because `./benchmark.sh` always passes `--cool-gate` |
| `--local-submit` | 1023 | 1024 | On |

The gate is on because `./benchmark.sh` arms it. Driving the Swift CLI directly
skips it and times a hot GPU. Use `./benchmark.sh`. See AGENTS.md, "The
cool-down gate".

The two public goldens differ in length for this reason. Use the 256-token
golden for `--local-iterate`. Use the 1024-token golden for `--local-submit`.

> **NOTE — both local modes check correctness and speed.**
> Neither local mode is a speed-only signal. Both apply the teacher-forced
> check. Neither one runs the ranked gates.

The ranked run runs **one stream at a time**. David ruling 2026-08-27:
"Single-stream only". It measures 2 pairs over the one prompt the fixture names
in `live_golden`, and it sums each role's per-token times over the pairs.

> **WARNING — the batched cohort path refuses BY NAME.**
> Two reasons, and the second is decisive. The QSA sparse attention emits a
> custom array mask, and the ContinuousBatchingV2 path discards a custom mask
> by contract. And 36 of the 48 layers carry recurrent state with no
> key-value tape, so three quarters of the model has no shape in that engine's
> cache bank at any context length. The engine no longer advertises the
> batched capability, so the benchmarker refuses before it spends box time.
> There is no dense-attention fallback. Section 11.4 of
> `docs/participant-contract.md` holds the detail.

> **WARNING — a local score is directional, not predictive.**
> Treat a local score as a smoke signal for speed and correctness. Do not treat
> it as a prediction of the ranked composite. The ranked M5 run is the
> authority.

Local testing stays single-stream, and so does the ranked run.

## Scoring and gates

### The formula

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
```

Each component is a gain:

```text
gain = baseline_leg_seconds_per_token / candidate_leg_seconds_per_token
```

The score is serial-anchored. A faster candidate scores above 1.

### The pair is measured, not stored

A ranked run measures PAIRS OF LEGS. It measures them on the SAME box, in the
SAME job, over the ONE prompt the fixture names in `live_golden`. The fixture's
`official_pairs` sets the count, and it is 2 (David ruling 2026-09-09). Every
pair is the same two legs in the same order:

1. The **serial-control leg**. It runs on the organizer's reference tree. That
   tree is a build of this repository at the commit the fixture names in
   `baseline_reference_commit`. This leg uses no speculation. Its tokens are
   checked against the serial tape (`<live_golden>.golden.json`), never against
   a per-depth tape.
2. The **candidate leg**. It runs on your tree, at the draft depth you declare.

The legs run strictly one after the other, and each leg loads the model once.
Per role the per-token times are summed over the pairs, and the score is the
ratio of those sums. Every control leg is checked against this box's baseline
calibration. All the numbers come from the same machine, minutes apart.

**NO FILE HOLDS A BASELINE PAIR.** The scoring constants hold none. The fixture
holds none. The goldens hold none. A golden that carries
`benchmark.baseline_prefill_seconds_per_token` or
`benchmark.baseline_decode_seconds_per_token` is refused on the ranked path.
`tools/lint-benchmark-manifest.py` keeps the two fields out of the tree.

The organizer stages the reference tree on each ranked box. The ranked job
verifies that tree. It does not fetch it and it does not build it.

### Each box has its own calibration

Each ranked box records what its own serial-control leg costs. The record is a
file. `MLXFAST_BASELINE_CALIBRATION` names it.

**THE FILE IS A HEALTH BAND. IT IS NEVER A DENOMINATOR.** The benchmarker
compares the measured control leg against the band. The run stops by name when
the leg falls outside the band. A stale calibration file can stop a run. It can
never move a score.

An operator writes the file on the box:

```bash
tools/calibrate-box.sh "<runner name>" /path/to/baseline-calibration.json
```

The command takes the box GPU lock. It then runs the serial-control leg four
times under the full official methodology: the cool gate before each pass, one
resident worker for each pass, and the same live golden the ranked run scores
over. It writes the mean, the coefficient of variation and the band for prefill
and for decode. It writes no file when the coefficient of variation is more
than 1 percent on either axis. A box that cannot repeat itself has no band.

The `box` value in the file must equal the runner name. The `reference_commit`
value must equal the fixture's `baseline_reference_commit`.
`tools/ranked-box-preflight.sh` refuses the run when either differs.

### The measured window

| Quantity | Value |
|---|---|
| Seed tokens per stream | 1024 |
| Checked decode steps | 128 |
| Golden shape | 1024 prompt tokens and 129 expected tokens |
| Streams per window | 1 |
| Timed prompts per leg | 1 (the fixture's `live_golden`) |
| Prompts in the pinned correctness pool | 8 |
| Prefill tokens per correctness-pool pass | 8 x 1024 |
| Pairs per ranked job | 2 (the fixture's `official_pairs`) |
| Legs per ranked job | 4 (each pair is serial control, then candidate) |

The correctness-pool rows are not the scored timing. The box stages all 8 pinned
prompts and the preflight verifies all 8. Each timed leg runs the one prompt
`live_golden` names.

The legs run one after the other. Each leg loads the weights once. The
unmeasured warm-up prefill pass stays at 1 pass, and it applies to every leg in
the same way.

The serial-control leg runs entirely inside the reference tree. It uses that
tree's worker, that tree's Metal library and that tree's own transformed
weights. Nothing you change can move it.

### One resident worker per LEG, booted by the benchmarker

A ranked job has 2 pairs, so 4 legs on two trees. The weights load ONCE per leg.

The benchmarker starts `bench-worker runtime-worker` once for each phase: the
warmup, the timed prefill, the timed decode and the correctness pass. Each start
used to load the whole checkpoint again. It no longer does: a resident owns the
weights for the leg, and each per-phase worker attaches to it.

**THE RESIDENT BELONGS TO THE LEG, NOT TO THE WINDOW.** The two roles run
different trees with different weights, so one resident cannot serve both. The
benchmarker knows where a leg begins and ends, so the benchmarker boots it. For
each leg it calls that leg's OWN copy of `tools/resident-up.sh`:

```bash
tools/resident-up.sh --boot --spec <serial|mtp> --draft-len <N> --socket-out <file>
tools/resident-up.sh --stop --socket <path>
```

`--boot` loads that tree's own `.build/release/bench-worker` and that tree's own
`weights/`, waits for a healthy hello, writes the socket path as the first line
of the `--socket-out` file, and exits 0 with the resident still running. A
`<socket>.pid` sidecar and a `<socket>.ready` marker sit beside the socket.
`--stop` ends that resident and removes all three files. A second `--stop` is a
no-op.

`--spec` is authoritative. Nothing in the boot reads the tree's
`mtp-head.manifest.json`, so a reference tree that declares a draft depth still
boots a serial control leg when the benchmarker says serial.

`tools/qwen38-125b-a6b-measure-and-score.sh` boots nothing. It takes the box GPU
lock `/tmp/mtplx-gpu-exclusive.lock` and holds it for the whole measurement,
because a resident holds about 113 GB of unified memory whoever booted it, and
the box needs exactly one loader. `tools/resident-up.sh` refuses to boot when
nobody holds that lock, so every per-leg boot happens inside that window.

| Name | What it is |
|---|---|
| `BENCH_WORKER_RESIDENT_SOCKET` | A leg's resident Unix socket. A per-phase worker attaches to it instead of loading. The measure script never sets it, and REFUSES when the environment does. |
| `RESIDENT_IDENTITY_FILE` | A JSON file that records the resident, its socket, its declared leg and its hello. |

> **WARNING — do not set `BENCH_WORKER_RESIDENT_SOCKET` on a ranked box.**
> It names one already-loaded resident. Both legs would attach to it, so the
> serial-control leg would run on the candidate's weights. Public run
> 34230122059 failed this way when the measure script exported one. The measure
> script and `tools/ranked-box-preflight.sh` both refuse it now.

The wrapper form of `tools/resident-up.sh` stays for local, unscored use:

```bash
tools/resident-up.sh --weights <dir> [--ngram <dir>] -- <command>
```

It boots one resident, exports `BENCH_WORKER_RESIDENT_SOCKET` to the command,
and halts the resident when the command ends. The ranked path does not use it.

### The parameters

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

The scored width is fixed. A width the benchmarker has not certified has no
series tag, and the benchmarker refuses that width rather than run it.

**BOTH FLOORS ARE 0.95** (David ruling 2026-09-09). A candidate that regresses
prefill or decode by more than 5 percent is refused. The floors and the ceiling
apply to the aggregate over the 2 pairs, not to one pair. The fixture declares
them as `decode_speedup_floor` and `prefill_speedup_floor`, and the benchmarker
enforces the fixture's values.

There is no median. Each role's per-token times are summed over the pairs, and
each gain is the ratio of those sums.

`kvBackend` is pinned `contiguous` on both legs. The benchmarker refuses when
it cannot honour the pinned backend. It does not degrade to another backend.

### Token fidelity

The benchmarker applies a per-stream token-tolerance gate with a **10% budget**.

> **WARNING — the gate accepts similar output, not identical output.**
> This track does not require your output to match the serial trajectory token
> for token. The block-shaped forward pass diverges from the serial forward
> pass at near-tie argmaxes. The gate prices that divergence against the 10%
> budget. Do not read the gate as lossless.

A near-tie argmax can diverge across Apple Silicon generations, even for
correct code. Before you treat a local failure as your own regression, check
whether an unmodified `main` fails at the same token position on your machine.

### Current status

Five statements are true right now.

1. `fixtures/qwen3_8_125b_a6b_track.json` sets `official_scoring_enabled` to
   `true`. That flag is the single authority on the arm state, and the
   benchmarker enforces it.
2. The timed prompt pool and the hidden correctness oracle are pinned. All 8
   `timed_prompt_pool[]` entries and `hidden_correctness_golden` carry a sha256
   and a byte count. `live_golden` names the prompt the timed leg runs. One
   per-depth oracle is pinned for each draft depth 1 to 6.
3. A runner advertises the ranked label set
   `[self-hosted, macOS, qwen3.8-125b-a6b-mlx-v1]`. The box stages the goldens
   and the benchmarker pair, and the preflight verifies every staged file
   against the fixture pins before any measurement. The box also stages the
   reference tree and its own calibration file. It exports them as
   `MLXFAST_BASELINE_WORKSPACE` and `MLXFAST_BASELINE_CALIBRATION`. The ranked
   job refuses when either is absent. `tools/stage-baseline-workspace.sh`
   builds the reference tree, and `tools/calibrate-box.sh` writes the
   calibration file.
4. Scoring is single-stream and paired. `scored_batch_size` is `1`. The
   candidate leg runs one stream with the MTP head at the declared depth. The
   serial-control leg runs on the organizer's reference tree at the fixture's
   `baseline_reference_commit`. The score is the live ratio. No golden carries
   a baseline pair.
5. The bench channel `qwen3.8-125b-a6b-v1` resolves from its release branch.
   `./tools/fetch-benchd.sh` reads the channel's `dist/benchd.manifest.json`
   on branch `BENCHD_BRANCH` (default `qwen3.8-125b-a6b-v1`), verifies the
   binary against the `sha256` and `bytes` that manifest names, and prints the
   identity it resolved. THE MANIFEST IS THE SOURCE OF TRUTH for which
   `benchd` measures a run, and this document pins nothing: read the identity
   off the script, not off this page.

The model port has landed: the engine constructs, gates, loads and runs
`qwen4_exp_text`, and `Sources/MLXFastCore/Constants.swift` carries this
target's geometry. `docs/qwen38-125b-a6b-port-notes.md` holds the detail.

## Submitting

Use the Yukon CLI for every account operation and every submission operation.

```bash
export PATH="${HOME}/.local/bin:${PATH}"
```

This command puts `yukon` on your path.

```bash
yukon login <api-key> --api <url>
```

This command authenticates you.

```bash
yukon clone <benchmark-id-or-name>
```

This command clones the benchmark repository.

```bash
yukon submit --model "<exact model name>" --note-file submission-note.md
```

This command uploads your editable-path archive.

```bash
yukon submissions
```

This command lists your submissions.

A submission archive replaces the editable paths. It rejects generated
artifacts, symlinks, local scores, reference checkpoints, and any source change
outside the editable surface. `yukon submit` does not run a local test first.
No local run blocks the upload. Run the local test yourself before you submit.

## The pinned artifacts

| Artifact | Identity |
|---|---|
| Target model | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` @ `327c8a604de613b42f84ba5e6b796c0931e8aa3b` |
| Target manifest | `fixtures/reference_qwen3_8_125b_a6b_4bit.sha256` (32 records, 113,233,030,116 bytes) |
| MTP head | Embedded in the target checkpoint under `language_model.mtp.*` |
| Benchmarker | branch `qwen3.8-125b-a6b-v1` — the PROJECT channel, shared with the CUDA track — resolved at run time from that branch's dist channel (`benchd.manifest.json` is the authority for the bytes; `tools/fetch-benchd.sh` enforces it and logs the resolved `source_commit`/sha256) |
| Track id | `qwen3.8-125b-a6b-mlx-v1` — the PLATFORM name: leaderboard namespace, runner labels, R2 prefix. NOT the bench branch. |
| Engine fork revision | `449f2d01b39f9088739c98d80a4f8a1b3cfa105e` |

The model repository is public. It downloads without a token. There is no
organizer-hosted mirror for this checkpoint, so
`MLXFAST_REFERENCE_FALLBACK_BASE_URL` is empty by default.

### The target model

| Property | Value |
|---|---|
| Architecture | `qwen4_exp`. The text tower is `qwen4_exp_text`. |
| Hidden layers | 48 |
| Attention pattern | A four-layer repeat. Index `% 4 == 3` is full attention. |
| Full-attention layers | 12, at indices 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, and 47 |
| Linear-attention layers | 36. They are gated deltanet layers with a constant-size recurrent state. |
| Full attention | 24 query heads, 2 KV heads, head dimension 256 |
| Rotary | Partial 0.25, `rope_theta` 1e7, interleaved mrope sections [11, 11, 10] |
| QSA indexer | 4 heads, 1 KV head, dimension 128, budget 2048, compress 4 |
| Hyper-connections | `hc_count` 4, `hc_lowrank` 320 |
| Routed experts | 512 |
| Experts per token | 10 |
| MoE intermediate size | 640. A shared expert of the same width sits behind a shared expert gate. |
| n-gram / PLE | Layer index 1. `ngram_size` 3, 8 heads per n-gram, 128 split parts, so 384 shard tensors plus 3 int64 buffers. The table is offloaded to SSD behind a bounded LRU. |
| Hidden size | 2560 |
| Vocabulary | 248320 |
| Embeddings | Untied |
| Tokens | eos 248046 and 248044; bos and pad 248044 |
| Quantization | Affine, group size 32, 4 bits. Router gates and multimodal weights are BF16. |
| Final norm | There is no `model.norm` tensor. The final `hyper_connection_mixer` stands in for it. |
| Raw tensors | 3747 across 22 shards, index `total_size` 113,209,155,128 bytes |
| Text tower tensors | 3414 |
| Vision tower | 333 tensors. The loader skips them. |

There is no sliding-window attention on this model.

## Building after an edit

Two build forms matter, because the vendored MLX package builds in JIT mode.

Kernel families with an `mlx-generated/*.cpp` twin compile at runtime from the
C++ source strings inside those files. For these families the twin is the
runtime-effective source. Edit the twin. Keep the readable `.metal` and `.h`
pair in step with it.

RoPE, RMSNorm, the SDPA vector kernel, and `arg_reduce` load ahead of time from
`mlx.metallib`. After you edit one of those sources, run
`tools/build-mlx-metallib.sh`. `./setup.sh` runs that script for you.

`_nax` names are the M5-generation kernel variants. The ranked runner selects
them. Tune the `_nax` twin as well as the plain one.

```bash
swift build -c release --force-resolved-versions
```

This command builds the trusted CLI into `.build/release`.

```bash
tools/build-bench-worker.sh
```

After initial setup, this command rebuilds the release track worker and Metal
library, checks that the worker links the editable `TrackRunner`, and stages
the set where benchd resolves it. It downloads and loads no model weights.
It records the source/toolchain key, worker SHA-256, Metal SHA-256, and Metal
fingerprint-sidecar SHA-256 in `.build/release/bench-worker.build.json`.

Run `tools/build-bench-worker.sh --check` before reusing a staged build. It
refuses missing provenance, source changes (including new files under
`Runner/`), or changed staged artifacts. The record detects local stale builds;
it is not a signed attestation or a substitute for benchd's runtime checks.

The equivalent manual worker build selects this package's unique product:

```bash
swift build -c release --force-resolved-versions --scratch-path .build-worker \
  --product track-bench-worker
```

This command builds `.build-worker/release/track-bench-worker`, which registers
the editable `Runner/` and compiles against `Vendor/mlx-swift`. The dependency
also exports a product named `bench-worker`; building that product omits this
repository's `Runner/` edits.

```bash
tools/stage-bench-worker.sh
```

After `tools/build-mlx-metallib.sh`, this command copies `track-bench-worker`
to `.build/release/bench-worker`, with `mlx.metallib` and its fingerprint
sidecar beside it. Direct staging copies existing bytes and clears the
combined build command's provenance record; it does not prove freshness.

> **WARNING — always pass `--force-resolved-versions`.**
> The dependency graph is frozen. A bare `swift build` or `swift test` can
> rewrite `Package.resolved`. `./setup.sh` then refuses to run. Restore the
> file with `git checkout -- Package.resolved`.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and on every push to
`main`. It runs repository hygiene checks on `ubuntu-latest`. It runs
`swift build --build-tests` and `swift test` on a hosted `macos-26` runner. It
treats first-party warnings as errors.

CI is advisory. No status check is required. A red run blocks neither a merge
nor a dispatch. `docs/ci-coverage.md` holds the detail.

CI never measures and never scores. CI holds no secret, downloads no weights,
and runs no GPU test. The GPU tests and the checkpoint tests are box-only.
`docs/ci-coverage.md` lists them, and CI fails when that list drifts.

`.github/workflows/benchmark.yml` is the ranked pipeline. It triggers on
`workflow_dispatch` only. It holds no secret. Its ranked job runs on
`[self-hosted, macOS, qwen3.8-125b-a6b-mlx-v1]`. Before it measures, it verifies
every box-staged asset against the contract's `{sha256, bytes}` pins. It
publishes no score until the runner is registered, the box is staged, and the
benchmarker emits a composite. Each of those gaps gives a non-zero exit and no
artifact.

## Where to get help

| Question | Authority |
|---|---|
| What the track measures, path by path | `benchmark.json` |
| Pins, the timed pool, scoring values | `fixtures/qwen3_8_125b_a6b_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| The engineering log for this port | `docs/qwen38-125b-a6b-port-notes.md` |
| What CI covers | `docs/ci-coverage.md` |
| Agent and contributor guidance | `AGENTS.md` |

> **NOTE — the order of authority.**
> The ranked M5 run is the authority on any score. The contract fixture
> `fixtures/qwen3_8_125b_a6b_track.json` wins over this document. This document
> only explains; it never overrides. If either disagrees with the benchmarker
> about measurement, the benchmarker wins.

## License and attribution

This repository's harness code is licensed per [LICENSE](LICENSE). The pinned
checkpoint carries the Qwen Community License 1.0. The terms ship with the
checkpoint at its pinned revision. This repository distributes no model
weights. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) holds the full
third-party attribution.
