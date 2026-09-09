# The task — Qwen 3.8 125B A6B MLX

Make the Qwen 3.8 125B A6B text tower run faster on Apple Silicon.

The ranked track is `qwen3.8-125b-a6b-mlx-v1`. Read [README.md](README.md) for the
setup steps and the repository structure. Read
[`docs/participant-contract.md`](docs/participant-contract.md) for the reasons
behind the rules.

## What you optimize

You optimize the engine. The engine is the MLX runner, the offline transform,
and the vendored MLX Metal kernels that the forward pass dispatches. You also
optimize the batching engine and the speculative-decode arm.

The target model is `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`. It is a sparse
MoE model. The text tower is 48 layers on a four-layer repeat, with 512 routed
experts, 10 experts per token, and untied embeddings. Twelve layers use full
attention, at the last index of each group. The other 36 are gated deltanet
linear attention. There is no sliding window on this model.

The MTP head proposes tokens. The target model decides every emitted token. The
head ships inside the pinned target checkpoint under `language_model.mtp.*`.

## What you may change

`benchmark.json` `editablePaths` is the authority. It lists 71 entries in four
groups.

| Group | Paths |
|---|---|
| The head declaration | `mtp-head.manifest.json` (the declaration file only) |
| The Runner | `Runner/` |
| The offline transform | `Sources/MLXFastTransform/` |
| The vendored kernels | The 68 MLX Metal files the forward pass dispatches |

The Runner in `Runner/` is editable. `Sources/BenchWorker/` registers it
before the engine resolves a runner, so it shadows the fork's built-in runner.
`Runner/` also holds the MTP head and the assistant that drives it.
The engine core is the `Vendor/mlx-swift-lm` submodule and it is not editable:
a gitlink names a commit, not bytes.

The rule behind the list is simple. Code that **proposes** tokens or computes
the forward pass is editable. Code that **verifies**, **measures**, or
**ledgers** stays trusted.

The MTP head is the organizer's pinned weights, because it is part of the
pinned target checkpoint. You may re-quantize it. You may not replace it, and
you may not upload head weights of your own.

No submission carries a head weight file. Nothing stages one.
`mtp-head.manifest.json` stays editable and optional, and it accepts
`"source": "pinned"` only, which on this track means the head embedded in the
pinned target checkpoint. `"source": "remote"` and `"source": "in_branch"` are
refused by name.

The head has a 2 GiB declaration cap. The size cap is the only gate on a
declaration. A declared `sha256` is optional, and the runner does not verify it.

A re-quantization happens ON LOAD, in memory. Nothing on disk changes, and no
artifact travels in a submission.

The head module and the assistant that drives it are in `Runner/`, which is
editable: `Runner/Qwen4ExpMTP.swift` and `Runner/Qwen4ExpMTPDrafter.swift`. The
seam is `TrackQwen4ExpRunner.adoptMTPHead` in `Runner/Qwen4ExpRunner.swift`,
which selects the quantization geometry of the served head. By default it
selects the checkpoint's own geometry, so the served head is bit-exact with the
head the pinned fork builds. To re-quantize, change the geometry that function
selects. A replacement of the head is still refused, and head weights of your
own are still refused. `docs/participant-contract.md` section 4.4 is the
authority.

> **WARNING — the target quantization is frozen.**
> Do not re-quantize any target weight. Do not re-represent one. Do not change
> the numerical format of one. This holds even when the result passes every
> correctness gate. An editable transform does not license the change. The MTP
> head is a narrow exception, and the exception is re-quantization only. You may
> re-quantize the head within its 2 GiB declaration cap. You may not replace it.

> **NOTE — the scored batch size is locked. Draft depth is not.**
> The scored batch size is 1. It is not a tunable.
>
> The draft depth is a free lever, set from your own drafter code, which is
> editable. It is not pinned at 1.
>
> Select a depth from 1 to 6. The non-editable engine clamps at 6, and benchd
> measures at depth 2 when the invocation names no depth.
>
> Each run seals `effective_spec` and `effective_mean_draft_len`, so the depth
> that ran and the draft length it realized are both visible afterwards.

You may not change anything that verifies, measures, or ledgers. This covers the
trusted harness, the target weights, the transform contract, the tokenizer, the
goldens, the gates, and the timing code.

## How to run it

```bash
./tools/fetch-benchd.sh
```

This command resolves and verifies the pinned benchmarker binary.

```bash
./setup.sh
```

This command builds the Swift binaries and downloads the target model. The MTP
head arrives inside that checkpoint, so there is no head-staging step.

```bash
.build/release/mlxfast-swift transform \
  --reference reference_weights/Qwen3.8-Flash-Next-MLX-4bit-MTP \
  --output weights
```

This command writes the `weights/` tree that the engine loads.

```bash
MLXFAST_ENGINE_BIN=.build/release/bench-worker \
MLXFAST_CORRECTNESS_GOLDEN_PATH=correctness_prompts/public_longcopy_gate_english_1024_256.json \
  ./benchmark.sh --local-iterate
```

This command runs the local test against the checked-in public golden.

## How it scores

```text
composite = prefill_gain ^ 0.25 * decode_gain ^ 0.75
gain      = baseline_aggregate / candidate_aggregate
```

The score is serial-anchored. A faster candidate scores above 1.

`aggregate` is the per-stream sum. Add each stream's own elapsed time
together. Do this for prefill and for decode separately, on both legs.

The ranked run measures one stream at a time over a 1024-token seed and a
128-step decode window. It runs 4 pairs. The floor is 0.90. The ceiling is
5.0. The KV backend is pinned `contiguous`.

The benchmarker applies a per-stream token-tolerance gate with a 10% budget.

> **WARNING — the gate accepts similar output, not identical output.**
> This track does not require token-for-token equality with the serial
> trajectory. The gate prices divergence against the 10% budget.

## The current state

> **NOTE — the track is NOT armed.**
> `fixtures/qwen3_8_125b_a6b_track.json` sets `official_scoring_enabled` to
> `false`, and the benchmarker refuses to seal an official scoring artifact
> while it is. The timed prompt pool and the hidden correctness oracle carry the
> pending sentinel `QWEN38-125B-A6B-MLX-PENDING-ORGANIZER`. No ORGANIZER
> goldens exist for this track, no runner advertises the ranked label, and the
> bench release branch and its dist channel are not published.

> **NOTE — the engine is pinned to an unmerged fork branch.**
> `Vendor/mlx-swift-lm` is a git submodule at `449f2d0`, on branch
> `feat/qwen38-flash-next-runner`. Re-pin it to the fork's `main` after that
> branch merges. The batched cohort path needs a ContinuousBatchingV2
> adaptation, because the QSA sparse attention emits a custom array mask that
> the CBv2 path discards by contract. `docs/qwen38-125b-a6b-port-notes.md` holds the
> detail.

The repositories stay private until launch.

## Local runs are directional

The local test and the ranked run use different machines, and only the ranked
run measures the paired legs under the scored gates. Treat a local score as a
smoke signal, not as a prediction. The ranked M5 run is the authority.

## Authorities

| Question | File |
|---|---|
| Editable paths, commands, scoring values | `benchmark.json` |
| Pins, the timed pool, scoring semantics | `fixtures/qwen3_8_125b_a6b_track.json` |
| Why the manifest says what it says | `docs/participant-contract.md` |
| What a measured run executes | the channel benchmarker (`tools/fetch-benchd.sh`, verified against the dist `benchd.manifest.json`) |

Where this document and the contract fixture disagree, the fixture wins. Where
either disagrees with the benchmarker about measurement, the benchmarker wins.
