# Qwen 3.8 125B A6B port notes — DRAFT

**DRAFT.** This document describes work that is part done. Each section says
what is landed and what is not. Do not read an entry as a statement that the
engine runs this model today. It does not.

This repository was seeded from the Gemma 4 26B A4B MLX engine. The seed commit
is tree-identical to that engine's `main`. This document is the engineering
record of the conversion to the `qwen3.8-125b-a6b-mlx-v1` track.

---

## 1. Track identity

One string serves three roles: the bench release branch, the track id, and the
R2 key prefix.

| Role | Value |
|---|---|
| Track id / release branch / R2 prefix | `qwen3.8-125b-a6b-mlx-v1` |
| Leaderboard namespace | `qwen3.8-125b-a6b-mlx-v1` |
| Static review track id | `qwen3.8-125b-a6b-mlx-v1` |
| Yukon benchmark name | `mlxfast-qwen38-125b-a6b` |
| Engine repository | `Layr-Labs/mlxfast-qwen38-125b-a6b-engine` |
| Contract fixture | `fixtures/qwen3_8_125b_a6b_track.json` |
| Pending sentinel | `QWEN38-125B-A6B-MLX-PENDING-ORGANIZER` |

The sentinel is matched EXACTLY. It is never a prefix test. Every organizer slot
that has no value yet carries the sentinel, and a test asserts that it does.

The track id is substring-clean against every retired name
(`gemma4-31b-it`, `MLXFAST_MTP_`, `mtp-ranked`, `measure-mtp-job`,
`mtp-weights`, `laguna-xs-2.1-mtp`). The port adds no `MLXFAST_MTP_`
environment name. The engine's own environment prefix is `MLXFAST_QWEN38_`.

## 2. The pinned target

| Item | Value |
|---|---|
| Repository | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` |
| Revision | `327c8a604de613b42f84ba5e6b796c0931e8aa3b` |
| Base model | `Qwen/Qwen3.8-Flash-Next`, quantized |
| License | Qwen Community License 1.0 |
| Shards | 22 |
| Tensors | 3747 |
| Index `total_size` | 113,209,155,128 bytes |
| Download manifest | `fixtures/reference_qwen3_8_125b_a6b_4bit.sha256` |
| Manifest records | 32, totalling 113,233,030,116 bytes |

The manifest pins the load-bearing files only. It does not pin
`.gitattributes`, `LICENSE` or `qwen-logo.png`. None of the three is engine
loaded. This follows the convention of the manifests it replaces.

The LFS-backed entries (the 22 shards and `tokenizer.json`) are pinned by the
LFS object's own sha256. The other entries were fetched and hashed directly. No
weight bytes were downloaded to produce this manifest.

`./setup.sh` needs 260 GiB of free space by default. The snapshot is about
105.5 GiB. The floor keeps the same ratio of snapshot to floor that the Gemma
default used.

### 2.1 Model facts

The text tower is 48 layers on a four-layer repeat. The last layer of each group
(`index % 4 == 3`, that is layers 3, 7, ... 47) is full attention. The other
three are gated deltanet linear attention. So 12 layers carry a KV cache and 36
carry a constant-size recurrent state. There is no sliding-window attention on
this model.

| Item | Value |
|---|---|
| Architecture | `qwen4_exp`; text tower `qwen4_exp_text` |
| Hidden size | 2560 |
| Full attention | 24 query heads, 2 KV heads, head_dim 256 |
| Rotary | partial 0.25, `rope_theta` 1e7, interleaved mrope sections [11, 11, 10] |
| QSA indexer | 4 heads, 1 KV head, dim 128, budget 2048, compress 4 |
| Hyper-connections | `hc_count` 4, `hc_lowrank` 320 |
| MoE | 512 routed experts, top-10, `moe_intermediate` 640 |
| Shared expert | width 640, behind `shared_expert_gate` |
| n-gram / PLE | `ngram_size` 3, 8 heads per n-gram, 128 split parts, on layer index 1 |
| Vocabulary | 248,320, embeddings UNTIED |
| Tokens | eos [248046, 248044], bos and pad 248044 |
| Quantization | 4-bit affine, group 32. Router gates and multimodal are BF16 |

Three facts correct earlier statements of the checkpoint's shape.

1. The n-gram shard tensors number **384**, not 386. They are 128 shards times
   `{weight, scales, biases}` on layer index **1** only. `ple_layer_ids` reads
   `[2]` because it is 1-based. Three more `ple_embedding` tensors are the
   int64 buffers `layer_multipliers`, `ngram_heads_vocab_sizes` and
   `ngram_heads_offsets`.
2. `lm_head` is named `language_model.lm_head.*`, not top-level `lm_head.*`.
   It is 3 tensors.
3. There is no `model.norm` tensor. The final `hyper_connection_mixer` stands
   in for it.

The tensor split is: `language_model.model` 3335, `language_model.mtp` 76,
`language_model.lm_head` 3, `vision_tower` 333. The text tower is therefore 3414
tensors. The vision tower is present in the checkpoint and is SKIPPED at load.

## 3. The speculative arm

This track has exactly ONE speculative arm: the native MTP head.

The head is EMBEDDED in the pinned target checkpoint, under the tensor prefix
`language_model.mtp.*`. It is 76 tensors and it lives entirely in shard 22. It
is 1 hidden layer, hybrid, full attention, with submodules `fc_embedding`,
`fc_hidden` and `hyper_connection_mixer`. It has no embedding and no `lm_head`
of its own: it rides the target's `language_model.embed_tokens` and
`language_model.lm_head`.

So this track stages no head weight file. There is no organizer head artifact to
fetch. `mtp-head/` no longer exists as a directory. No submission carries head
weights.

`mtp-head.manifest.json` survives as the DECLARATION surface only. It stays an
editable and optional editable path. It accepts `"source": "pinned"` only, which
on this track means "the head embedded in the pinned target checkpoint". It
carries no `arm` key.

Permitted draft depths are 1 to 6. The engine clamps at 6. David ruled the cap
on 2026-09-04 (engine repo 351801c): "cap the mtp at 6 for this challenge on
both mlx and cuda, make sure its usable". The CUDA side carries the same cap.

### 3.1 The DFlash arm is removed

RULED. David ruling 2026-08-27, relayed by orchestrator, verbatim: "Remove it,
MTP-only". `allowed_modes` is `serial` and `mtp`. The DFlash arm is deleted, not
disabled. That covers the drafter model, its loader, its wire verbs, its free-run
session, its head directory and declaration, its stager, its fixtures, its
contract document, its tests, and the arm-selection channel that chose between
the two arms. With one arm there is nothing to select.

`Laguna` and `Qwen 3.6` material is NOT DFlash material and is retained.
`LagunaConfig` and `LagunaCheckpointValidation` are the substrate the generic
transform tests run on. `Qwen35CheckpointValidation` and
`fixtures/qwen3_6_27b_*` are a live NEGATIVE CONTROL: they build a config the
trusted-config gate must REJECT. Renaming or deleting either would make the name
lie about what it holds.

### 3.2 What the sweep left for this change

The arm removal took the sources but left the tests that named the arm, so the
test target did not compile. That is repaired here: the tests whose subject went
with the arm are deleted, and the ones whose intent survives now use the
reserved `dspark` mode. `dflash` stays in the fail-closed lists as a RETIRED
name that must never route.

The submission-security suite floor drops from 287 to 259, in both places that
state it, in the same change that deletes the assertions. A floor only ever
drops beside the assertions it was guarding.

## 4. benchd resolves from its release channel

There is no `benchd.pin` in this repository and there must never be one again
(David ruling 2026-08-27).

`tools/fetch-benchd.sh` resolves `benchd` from the bench repository's dist
channel at `refs/heads/qwen3.8-125b-a6b-v1/dist/`. It fetches
`benchd.manifest.json` FIRST and verifies `{branch, sha256, bytes}` against
it. Nothing unverified is installed or returned. `BENCHD_REFRESH=1` discards the
installed pair and re-resolves.

### 4.1 The branch is the PROJECT, the track id is the PLATFORM

David ruling 2026-08-27, relayed by orchestrator. The MLX track and the CUDA
track of this model SHARE one bench release branch and one dist channel,
because they share one benchmarker. So two names exist and they are not
interchangeable:

| Name | Value | What it names |
|---|---|---|
| Bench release branch and dist channel | `qwen3.8-125b-a6b-v1` | the PROJECT. One benchmarker for both platforms. |
| Track id | `qwen3.8-125b-a6b-mlx-v1` | the PLATFORM. Leaderboard namespace, runner label set, R2 object prefix, static-review track id. |

Only `tools/fetch-benchd.sh` carries the PROJECT name: the `BENCHD_BRANCH`
default and, through it, the manifest's `branch` check. Everything else in this
repository keeps the TRACK id, unchanged. A change that pushed the project name
into the leaderboard namespace or the R2 prefix would merge the two platforms'
results, so the two names are stated here together to make that hard to do by
accident.

An earlier ruling (2026-08-27, verbatim "From gemma4-26b-a4b-mlx-v1 tip") cut
`qwen3.8-125b-a6b-mlx-v1` from the Gemma release-branch tip; that branch is
merged at `a7be295b`. The shared branch supersedes it as the CHANNEL. The
platform branch is not the channel any more.

### 4.2 The channel host

The default `BENCHD_DIST_BASE_URL` is the bench repository
(`Layr-Labs/mlxfast-bench`). It is the organizer's repository: participants
cannot write to it. An air-gapped box takes a verified pair through
`BENCHD_DIST_LOCAL` instead.

### 4.3 The shared channel resolves from the RELEASE BRANCH

**THE PAIR IS PUBLISHED AND IT VERIFIES.** The release branch
`qwen3.8-125b-a6b-v1` carries `dist/benchd.manifest.json`, and the default
network resolve, with no override, installs the pair it names.

THE MANIFEST AT THE CHANNEL TIP IS THE SOURCE OF TRUTH, and it MOVES: a bench
fix is published by republishing dist, with no commit in this repository (see
the header of `tools/fetch-benchd.sh`). So no `{source_commit, sha256, bytes}`
triple is written down here as current. To read the identity, run the script
and read the line it prints:

```
./tools/fetch-benchd.sh >/dev/null
```

```
fetch-benchd.sh: resolving the qwen3.8-125b-a6b-v1 channel tip (https://raw.githubusercontent.com/Layr-Labs/mlxfast-bench/refs/heads/qwen3.8-125b-a6b-v1/dist)
fetch-benchd.sh: benchd identity: branch=qwen3.8-125b-a6b-v1 source_commit=8439d6fe... sha256=2ee40b21ed08ea906cb3954f635a1719d1294fed46718b0b286afcf39ea5db76 bytes=2612000
```

That second line is an EXAMPLE of the shape, from a laptop resolve on
2026-08-29, not a pin (the real line prints the full 40-character
`source_commit`; it is abbreviated above). Expect different values; a
difference is the channel moving, which is the design.

The resolve is the same verification path a ranked box takes: the manifest is
loaded, its `branch` is checked against the expected channel, and the binary is
verified against `{sha256, bytes}` before anything is installed. The identity
line is also written to `benchd-bin/benchd.manifest.json` beside the binary,
so the harness a run used can be read off the box afterwards.

Before the merge, the pair lived only on the pull request's branch, and the
release branch tip carried a manifest inherited from the branch it was cut
from; the guard refused that tip by name as the wrong channel. That refusal was
the guard working: it is exactly the check that stops a box running a
benchmarker from a channel nobody meant to serve.

No lane path is hardcoded anywhere in this repository. `BENCHD_BRANCH` names
the release channel and nothing else.

The submission guards still forbid the `benchd.pin` and `benchd-bin` spellings,
so a submission can neither plant a pin nor repoint the fetch.

## 5. The engine, and what refuses

> **SUPERSEDED by section 13.** This section records the PRIVATE engine that
> this repository carried until the Darkbloom CBv2 re-base. That engine is
> deleted. The findings below are still true about the model, and section 13
> says where each surface lives now. Read section 13 first.

Read this section before you conclude that something is broken.

### 5.1 The engine drives this target

The port is END TO END. The engine constructs, gates, loads and runs
`qwen4_exp_text`:

| Surface | State |
|---|---|
| `Sources/MLXFastCore/Constants.swift` | this target's geometry. `slidingWindow` and `finalLogitSoftcapping` are DELETED, not set to a placeholder: this tower has neither, so a stale reader fails to compile. |
| `Sources/MLXFastCore/Qwen4ExpConfigKeys.swift` | the runtime `config.json` key set, shared by the participant loader and the trusted gate. |
| `Sources/MLXFastTransform/Qwen4ExpCheckpointValidation.swift` | the offline transform's own inventory and quantization contract. |
| the engine fork's `Qwen4ExpRunner` | builds the model, installs the disk-resident n-gram table, pins the hybrid cache stack, and vends the engine and the stepper. |
| the trusted pinned-configuration gate | this target's contract, in lockstep with the loader. |

QUANTIZATION IS UNIFORM: affine, 4 bits, group size 32, with no per-tensor
overrides. Both the config parser and the transform REFUSE a block that carries
one, because the loader resolves one width for every quantized path and an
override it ignored would quantize tensors at the wrong width with the right
names and the right shapes. The frozen-target pins say the same thing from the
trusted side: with an empty override table, the gate's override half becomes a
statement that NOTHING may carry another width.

THE N-GRAM SHARDS ARE NEVER READ. The loader skips them while reading the
shards rather than dropping them after, because a tensor dropped after the read
has already cost its bytes -- 29.80 GiB of them.

### 5.2 What refuses, and why

ONE refusal is load-bearing. It is named, it fails closed, and it is where a
ruling would land.

**The batched cohort path refuses.** `Gemma4Runtime.makeCohortEngine` throws by
name. There are TWO blockers, and the second is the decisive one:

1. Every full-attention layer runs a QSA indexer. Above the indexer's budget it
   emits a per-query boolean KEEP MASK, and the ContinuousBatchingV2 path owns
   the attention call and DISCARDS a custom array mask. A cohort engine would
   serve dense attention under a model trained sparse.
2. A `CBv2LayerKind` is either full attention or a sliding window. This tower
   has 36 layers that carry a constant-size RECURRENT state and no key-value
   tape at all, so three quarters of the model has no shape in the v2 cache
   bank. This one holds at EVERY context length, so no budget check rescues it.

A fallback would be worse than a refusal: below the indexer budget the model IS
plain causal, so a cohort engine would look correct on short contexts and
diverge exactly where the score is measured.

THE ENGINE ALSO STOPS ADVERTISING IT. The hello no longer carries
`batched_free_run_decode` or `max_batch_size`, so the benchmarker refuses at its
pre-measurement capability check instead of after it has already sent a batched
begin. `RuntimeWorkerCohortTests` and `EmitWireFixtureTests` pin the withheld
capabilities, and the wire fixture digest was repinned with them.

THAT QUESTION IS NOW RULED. David 2026-08-27, relayed by orchestrator:
"Single-stream only". This track scores a paired serial-against-MTP
single-stream series over the pinned pool at `scored_batch_size` 1, and the
ContinuousBatchingV2 adaptation is NOT pursued. Section 9.2 records what that
changed here and what it leaves to the bench lane.

### 5.2.1 The speculative arm

**The MTP arm RUNS.** It is not a refusal any more. The head ships inside the
pinned checkpoint as 76 tensors under `language_model.mtp.*`; the worker
advertises `mtp` only when those tensors actually bound, and the arm drafts,
verifies, accepts and rolls back.

#### The round

One round, at draft depth `k` from the envelope:

1. DRAFT. `Qwen4ExpModel.mtpStep` runs `k` times, seeded by the target's
   pre-final-mixer stream at the last consumed position and chaining on its
   own output.
2. VERIFY. The target is fed the pending token plus all but the last draft, so
   `k` fed tokens verify `k` drafts: the prediction after fed token `i` is the
   target's own greedy argmax for draft `i + 1`.
3. ACCEPT. The longest prefix where a draft equals the target's argmax. At the
   first disagreement the TARGET's token is committed, not the draft's. This
   is what makes the arm change speed and not output.
4. ROLL BACK to the accepted boundary.

#### Rollback is a snapshot, not a trim

On the 12 full-attention layers a rewind is a trim: the key-value tape is
append-only and `offset` says how much of it is real, and the QSA indexer's own
key tape is sliced with it.

On the 36 gated-deltanet layers it is NOT. Their state is a RECURRENCE that
each forward overwrites, so there is no row to drop and no offset to move back
-- the state after the accepted prefix is simply not in the cache any more.

So the whole stack is snapshotted BEFORE the verify, and a rollback restores
that snapshot and replays the accepted tokens.
`Qwen4ExpSpeculativeRollbackTests` pins that the restored stack is
BIT-IDENTICAL to the snapshot -- every recurrent state, every short-convolution
state, the n-gram history, every offset and the indexer tape -- and that a
rolled-back stack decodes like one that never advanced.

#### What the arm cannot do yet, and WHY THE OLD REASON WAS WRONG

The verify feeds ONE TOKEN AT A TIME
(`Qwen4ExpFreeRunSession.verifySegmentWidth == 1`). Raising that is the whole
optimisation.

**THE MEASUREMENT THAT JUSTIFIED THE CAP WAS TAKEN ON DEFECTIVE CODE.** It said
a multi-token forward is not bit-identical to the same tokens fed one at a
time: on the fixture, the largest logit difference was about 0.13 with the QSA
indexer active and about 0.001 without it, with the argmax flipping in 7 of 40
trials at width 3. It also recorded "NOT the keep mask" as ruled out.

That conclusion was wrong, and the ruling-out was wrong with it. The keep mask
WAS the cause -- section 5.2.1.1 -- and the earlier check missed it because it
compared the mask column by column at one position rather than asking whether
the mask ever reaches past its own query. On the fixed code the same survey
shows ZERO argmax disagreement in 40 trials at width 2 and at width 3, and the
residual logit difference falls from 0.13 to 0.001, which is float
accumulation order: MLX dispatches a different kernel at M = 1 than at M > 1,
and the MIT reference shows the same residual against itself.

**THE CAP IS LIFTED (David ruling 2026-08-28).** The measurement this
repository used to hold was taken on defective code and is VOID -- both its
numbers and its conclusions, including the "row 0 differs between widths"
finding that started this investigation. The re-survey on fixed code replaced
it: section 5.2.1.4 has the numbers, and the ACCEPTANCE BASIS they are read
against is ruled there. The mtp leg now verifies at its resolved depth; see the
subsection below for what the lift changed and what it caught.

WHAT THE CAP COST WHILE IT WAS IN PLACE, and what removing it changes. At
width 1 every committed token paid one target forward, PLUS the draft-depth
head forwards that proposed it, PLUS one full-stack snapshot per round -- so
the arm was strictly more work than serial for the same output. A round now
verifies its whole draft chain in one target forward, which is where the
speedup has to come from. Whether it arrives is a box measurement, not a
claim this document makes.

### 5.2.1.1 A REAL DEFECT, FOUND AND FIXED: the QSA keep mask reached forward

**What was wrong.** `Qwen4ExpQSAIndexer` computed the count of COMPLETE key
blocks behind a query as

```
maximum(qPos + 1, 0) / compressRatio
```

`/` on an MLX array is TRUE division: it promotes `int32` to float and keeps
the remainder. The MIT mlx-lm reference (`@c961f839`) computes `//`, an integer
count. Both uses of that count then went wrong:

* `visible` compares block ids against it, so `0.25` instead of `0` admitted
  the INCOMPLETE block the query sits in. The keep mask then let query row `r`
  attend to keys `r + 1` up to the end of that block -- FUTURE tokens -- and
  the top-k chose among a shifted candidate set.
* `ownStart` scales the count back up, so `1.0` instead of `0` pushed the "own
  partial block" tail PAST the query and dropped keys the query must always
  keep.

**How it was found.** Not by reading. The port was compared against the
reference NUMERICALLY, on SHARED RANDOM WEIGHTS: the reference builds a model
at a reduced-real config, writes its own weights and its own logits, and this
repository loads the same weights and compares. Single-token forwards agreed to
1.9e-7. The full prefill disagreed by 0.28 -- and only on rows at or past the
indexer budget. Raising the budget so the indexer went inert made the
disagreement vanish. That localized it to the indexer in one step.

**The measurement, before and after.** Same weights, same input, S = 24:

| Quantity | Before | After |
|---|---|---|
| Keep-mask cells where the two implementations disagree | 52 of 576 | 0 |
| Attention layer output vs the reference | 0.133 | 0.0 |
| Whole model, worst row vs the reference | 0.281 | 0.000394 |
| Fixture: worst row move when ONE later token changes | 0.083 | 0.0 |

The residual 3.9e-4 is fp32 accumulation order and is present with the indexer
inert too.

**What it did and did not affect.** The indexer is inert at or below its budget
(2048 tokens), and the scored window is 1024 seed plus 128 decode. So the
SCORED path never reached the defect. Every context past 2048 tokens did.

**The reference is correct here.** This was our own deviation from it, so there
is nothing to report upstream for this item.

**What it does NOT explain.** The pinned-weights width survey shows divergence
in its "indexer off" arm as well (offset 1024, below the budget), where this
mask is never built. That arm remains open, and so does the separate finding
that this engine and the reference disagree on the pinned checkpoint even on
the single-token path.

`Tests/MLXFastTests/Qwen4ExpQSAIndexerCausalityTests.swift` holds it: the mask
never keeps a key past its query's own column, at six tape lengths; no row
moves when a later token changes, quantized and not; and the block count stays
integral.

THE FIRST TWO WERE RED ON THE PREVIOUS LINE and are GREEN on this one -- the
mask test at all six tape lengths, with the leaking query and key pairs
enumerated. THE THIRD WAS NOT, and saying otherwise would overstate the
evidence: it pins what `floorDivide` means, and `floorDivide` always meant
that. It is there so a later edit cannot reintroduce true division by claiming
the two are the same.

The whole-tower perturbation test pins the default device to the processor.
Its quantized arm was flaky in the full serialized suite -- several unrelated
rows moving by the IDENTICAL amount, which is two prefills computed by
different arithmetic rather than a causal leak, because a leak moves each row
by its own amount. Other suites here run inside a device override, so which
device this test inherited depended on what ran before it, and quantized
matmul does not sum in the same order on both. Pinning it makes both halves of
a bit-identical comparison the same computation. The pin names the GRAPHICS
processor, because this tower cannot run anywhere else: the gated deltanet is a
custom Metal kernel and MLX refuses it on the processor.

### 5.2.1.2 THE NORM CONVENTION: this checkpoint bakes the offset

**The fact.** Two conventions for RMSNorm weights are in circulation and they
compute the same function from DIFFERENT stored bytes:

| Convention | The file holds | The model must compute |
|---|---|---|
| zero-centered | `w` | `y * (1 + w)` |
| offset baked | `1 + w` | `y * w` |

Nothing in the published `config.json` says which one a checkpoint holds. It
carries `rms_norm_eps` and nothing else about the norms. Apply the wrong one
and every non-gated norm in the tower is off by a whole unit of scale: the
model multiplies activations by roughly zero, or by roughly two, and produces
incoherent output at every quantization level while every shape, every digest
and every byte count still checks out.

**This track's checkpoint BAKES THE OFFSET.** Measured on the box on
2026-08-28, read-only, over all 48 layers and the MTP head. Every non-gated
norm family is strictly positive and centred at or near one; the numbers are
recorded as facts in `fixtures/qwen3_8_125b_a6b_norm_convention.json`, family
by family, with the measurement's provenance. The GATED deltanet norm
(`linear_attn.norm`, 36 tensors, mean 1.024) is conventional in this checkpoint
and in the official one alike and is NOT part of the fact.

**Corroboration, from the other direction.** The thread on ml-explore/mlx-lm
pull request 1788 reports that the OFFICIAL `Qwen/Qwen3.8-Flash-Next`
checkpoint is zero-centered and needs `y * (1 + w)` (Sofille65, 2026-08-26),
and that the Vontra conversion was measured element-wise to hold `1 + w`
already (eauchs, 2026-08-27). mlx-lm at this port's source revision computes
`mx.fast.rms_norm(x, 1.0 + self.weight)` unconditionally, so THE REFERENCE
DOUBLE-COUNTS ON THIS CHECKPOINT TOO. That is why the box parity capture found
both implementations incoherent on real weights, and why their disagreement
looked like a causality problem: two differently-broken towers.

**What this port does.** `Qwen4ExpRMSNorm` takes a `weightOffset`, one or zero,
from the configuration. At offset zero it multiplies by the stored weight
VERBATIM -- no arithmetic of its own, so no rounding of its own. The engine
sets the offset from the pinned fact
(`MLXFastConstants.rmsNormConvention`), and because that assignment lives in
`Sources/MLXFastModel` -- an editable path -- the fact is verified again in
trusted scope.

**The verification, and why it reads the sign pattern.**
`validateLoadedNormConvention` (`Sources/MLXFastHarness`, outside every
editable path) classifies EVERY non-gated norm tensor the loaded model holds
and requires them all to agree, then refuses BY NAME -- naming the tensor, its
negative fraction, its mean and the rule that fired -- when the measurement
disagrees with the pin, or when a live module applies an offset the pin does
not. It classifies on the FRACTION OF NEGATIVE ENTRIES, not the mean: a
zero-centered tensor scatters around zero and has roughly half its entries
negative, while a baked tensor straddles zero only where some `w < -1`. The
means alone would not do -- `mtp.pre_fc_norm_embedding` averages 0.236, near
enough to zero to fool a threshold, with every entry positive.

**THE BOUNDARY IS 0.20, AND THE FIRST ATTEMPT AT IT WAS WRONG.** A 0.05
ceiling, chosen from family AVERAGES, would have REFUSED the pinned tree at
load: one tensor of 193, `model.layers.0.mlp_hyper_connection.hc_norm`, is
5.51% negative with mean 0.8905, which landed in the gap between the two bands
and read as neither. The hyper-connection norms are not a unit-scale gain --
the final mixer averages 3.75 and one layer-0 tensor reaches -4.94 -- so a
tight ceiling is inherently marginal on that family.

The measured separation is what sets the number. Across all 157 non-gated
language-model norm tensors the largest negative fraction is 0.0551; a
genuinely zero-centered tensor sits near half. 0.20 lies between them with
about 3.6x of room on each side, and the two rules meet at that one number so
there is no gap left to fall into.
`fixtures/qwen3_8_125b_a6b_norm_convention.json` records the per-family maxima
and minima the boundary was drawn from, and a test replays every family at its
WORST point and fails on the laptop if a threshold ever moves far enough to
refuse the pinned tree again.

**WHAT THE FIXTURE HOLDS, and what it deliberately does not.** It carries
values only -- dates, counts, dtypes, flags, family names and the two numbers
per family. The sentences that describe the measurement live here instead:

* The measurement was taken on the box on 2026-08-28, read-only, on the
  processor, over every norm tensor of the pinned tree: 193 in the language
  model and 110 in the vision tower.
* Each family's record is its WORST point -- the highest negative fraction any
  of its tensors reached, paired with the lowest mean any of them reached.
  Those two numbers need not come from the same tensor, and that is
  deliberate: a threshold has to clear the worst of both.
* The vision families are recorded and NEVER classified. Their numbers are the
  reason, not an input: `.bias` tensors are legitimately zero-centered
  (negative fraction 0.45 to 0.57) while `.weight` tensors run 0.00 to 0.31,
  crossing the boundary in both directions --
  `vision_tower.blocks.*.norm2.weight` sits at 0.1997, three ten-thousandths
  below it.

The fixture carries no pointer back to this document. A reader who needs the
prose is reading the port notes already.

**EVERY TENSOR, NOT A SAMPLE.** The first revision read the first 24 in path
order, which is both weaker and no cheaper: these are BF16 vectors of at most
10,240 entries and the whole set is one pass. A wrong convention is a property
of the FILE, so one disagreeing tensor is the signal, and unanimity is what
lets the check say "this tree is not what the pin says" rather than "the
tensors I happened to look at were fine".

**ONE LOAD ENTRY POINT.** `loadPinnedTarget` builds the weight cache and runs
the classifier, and the pinned-weights survey uses it. The cache already
applied the convention -- it sets the offset from the trusted constant before
constructing the model -- so the survey's arithmetic was never in doubt; what
it could not do was SAY SO from its own output, because the classifier ran only
in the worker's load path. The survey's report now carries the convention, the
offset READ BACK off a live module, and how many tensors agreed, so a box
result can be checked for that before it is read.

**THE VISION TOWER IS NEVER CLASSIFIED.** This is a text-only port and those
tensors are dropped at load, but the stronger reason is that its LayerNorms are
a different normalizer with a different convention, and the box numbers show
they would poison a unanimity rule: the `.bias` tensors are legitimately
zero-centered (0.45 to 0.57 negative) while the `.weight` tensors run from 0.00
to 0.31, crossing the boundary in both directions -- `blocks.*.norm2.weight`
sits at 0.1997, three ten-thousandths below it. Those numbers are recorded in
the fixture as the reason, not as an input.

`Tests/MLXFastTests/Qwen4ExpNormConventionTests.swift` pins the pin against the
fixture, the baked path against a reference computation bit for bit, the two
paths against each other on corresponding weights, the classifier on the
shapes the box measured (including the low-mean all-positive one and an
ambiguous one that must refuse), and both refusal directions with the MTP
head's norms in the sample.

**WHAT THIS DOES NOT CLAIM.** It has not been shown to fix generation: the
box has not re-run on this code. It removes a definite, measured defect from
the tower's arithmetic. Section 12 is the plan that settles the rest.

#### Norm convention travels with the tree

**The gap.** The pin and the load-time check above protect THIS engine. They
do not protect a different consumer. The transform copied `text_config`
through and never wrote the convention, so the transformed tree said nothing
about how its norm weights are stored. A fork that reads only `config.json`
took the reference default, which is the zero-centered `1`, and computed
`y * (1 + w)` on weights that already hold `1 + w`.

**The evidence.** The static differential of 2026-09-05 ranked this first of
every difference between the two paths, and the box confirmed it: with the key
present at the top level of the transformed tree, step-0 teacher-forced parity
returned to argmax 6184. Without it the model preferred a newline and the top
eight tokens were flat and low, which is what a wrong tower looks like and not
what a wrong tie-break looks like.

**What the transform does now.** The `.qwen4Exp` branch of
`makeRuntimeConfigData` writes `rms_norm_weight_offset` into the emitted
config. The value comes from `MLXFastConstants.rmsNormConvention.weightOffset`
and is never a literal, so a repin of the convention moves the emitted file
with it. The key goes at the TOP LEVEL, because the emitted config IS the
flattened `text_config` and a transformed tree carries no `text_config` block.
That is also where the runtime decodes it.

**What refuses.** `TransformVerifier` reads the tree's own `config.json` and
refuses a tree of this family that does not declare the pinned value. It
refuses BEFORE it regenerates and compares bytes, so the message names the key
instead of reporting an opaque config mismatch. A missing key is a refusal and
never a default, which is the same posture as the digest checks: the runtime's
own decode would fall back to the convention this checkpoint does not use.
Other families carry no such knob and pass through untouched.

**Belt and braces, not the only guard.** The fork also carries a family
default and a load-time validator. This change makes the TREE self-describing,
so a consumer that has neither still computes the right norms.

**A TREE TRANSFORMED BEFORE THIS CHANGE DOES NOT CARRY THE KEY.** It must be
transformed again. The fork's family default covers it until then.

### 5.2.1.3 THE QUANTIZED MIXTURE GATHER WAS NOT A FUNCTION OF ITS INPUTS

**The symptom.** Repeated prefills of the same tokens through one quantized
fixture model, a fresh cache each time: run 1 clean, run 2 differing by 1.30,
run 3 entirely NaN. The unquantized fixture was bit-identical across the same
three runs. A model is a pure function of its inputs, so this was a defect and
not a numerics question.

**The cause.** `mlx_gather_qmm` with `sorted_indices = true`. `SwitchGLU` sorts
its expert indices once they number 64 or more (`projectExperts`,
`doSort = indices.size >= 64`) and passes that hint down to the gather. The
threshold is exactly where the defect starts:

| Indices | Sorted hint | Repeated calls |
|---|---|---|
| 62 | off | bit-stable |
| 64 | on | diverge on the second call |
| 128 | on | diverge, then NaN |

**The bisect that got there**, each step a repeated-call comparison:

* the quantized model diverges; the UNQUANTIZED model does not; and
  bfloat16-but-not-quantized does not, which separates the dtype from the
  quantization;
* by component: dense quantized linear, attention with the QSA indexer, gated
  deltanet and the hyper-connection mixer are all stable. Only the MIXTURE
  diverges;
* inside the mixture: the router's logits and its top-k indices are stable, the
  shared expert is stable, and the gather diverges with the indices held FIXED,
  so it is not the routing;
* the same gather on bfloat16 weights is stable, and at one token (under the
  sort threshold) the quantized gather is stable.

**The vendor reproducer** has no model in it: one quantized `[E, out, in]`
stack, one activation, one index vector, called repeatedly.
`gatherQuantizedMM(sortedIndices: false)` is bit-stable on sorted and unsorted
indices alike; `sortedIndices: true` diverges; the NON-quantized
`gatherMM(sortedIndices: true)` is bit-stable. So it is neither the sorting nor
the gather, but the quantized kernel's sorted path. Environment: Apple M5 (10
GPU cores), macOS 26.6.2 (25G83), vendored MLX 0.32.0, mlx-swift `df1fdc5f`,
mlx-swift-lm `ed55bee`.

**THE HINT-ON PATH IS WRONG FROM THE FIRST CALL, not merely unstable across
calls.** The first description of this defect said the quantized model was "not
a function of its inputs", which is true and was how it was found -- but it
understates it, and the understatement matters. Repeatability and correctness
are different properties, and a reader could have concluded that one hint-on
result was as good as another.

Measured against GROUND TRUTH -- dequantize the packed weights and run the
ordinary dense `gather_mm` on them, so the only remaining difference is 4-bit
quantization error:

| Compared against the dense dequantized gather | max abs delta |
|---|---|
| hint OFF | 0.203 |
| hint ON, on its FIRST evaluation | 24.0 |

Two orders of magnitude apart, and the hint-on number is taken before any
repetition can be blamed for it. So the sorted path does not merely drift: it
returns a wrong answer immediately, and then a different wrong answer on the
next call.

**The fix** is one line in `SwitchLayers.swift`: the quantized gather is called
with the hint off. The hint is only a hint -- it permits a faster kernel, and
the arithmetic is the same either way. The test does not take that on trust: it
asserts the hint-off result against the dense dequantized ground truth, so the
suite says the workaround computes the RIGHT numbers rather than merely
repeatable ones. The sort itself is KEPT: it still helps locality, and it is
the flag that is broken, not the order.

**WHAT IT REACHED, and this is the part that matters.** The mixture runs on
every layer of this tower, and the checkpoint ships its experts quantized
(`switch_mlp.{gate,up,down}_proj.{weight,scales,biases}` in the pinned
inventory), so `quantizeUniformly` builds `QuantizedSwitchLinear` and the real
load path calls the same kernel. At the scored geometry a prefill of 1024
tokens at top-10 produces **10,240 indices**, far over the threshold, so THE
SCORED PREFILL WENT THROUGH THE BROKEN KERNEL ON EVERY LAYER. A width-1 decode
step produces 10 indices and never did.

That asymmetry explains a box observation that had looked contradictory: the
single-token determinism controls (narrow rerun, snapshot against fresh) were
all zero while prefills were wrong. They were measuring the one width that
never entered the broken path.

**CORRECTION 2026-08-28: that last claim was INFERRED FROM THE THRESHOLD, not
demonstrated at production dims.** A later reproducer run found the kernel
clean and bit-stable at the ENGINE-EXACT sorted call shape (the `gatherSort`
layout) across 64 to 8,192 indices at production dims, while the same binary
still reproduces the defect at the BATCHED call shape and at the small fixture
dims -- so "every scored prefill went through the broken kernel" overstates
what was measured. The question is moot for this track, because the fix
predates every baseline and every golden; the reproducer is the probe to re-run
on any geometry or vintage change.

**WHAT IS NOT ESTABLISHED FROM HERE.** Whether the box's silicon exhibits the
same defect. It was found on an M5 laptop; the box is different hardware, and
its recorded re-prefill control was zero, which is evidence it may NOT be
affected. The check is cheap and needs no weights: run
`Qwen4ExpQuantizedGatherDeterminismTests` there and read the reported
`gather_qmm sorted_indices=true` line. The fix is correct either way --
withholding a hint cannot change an answer -- so it does not wait on that
result.

**WHEN TO REMOVE IT.** When the vendored MLX is advanced, run that suite by
hand and read the same line. If it reports stability, restore the hint and
measure the throughput the workaround costs on every mixture layer.

### 5.2.1.4 THE RE-SURVEY ON FIXED CODE: what it settled and what it did not

The width survey and the reference parity capture were re-run on the box on
2026-08-28 against merged code at `1b940ae2` -- the QSA keep-mask fix, the norm
convention and the quantized-gather workaround all in. Artifacts:
`box-results/resurvey-2/`.

**The load proved itself.** The survey's report carries the four fields the
re-survey plan says to read first: `norm_convention: offsetBaked`,
`norm_weight_offset_applied: 0`, `norm_convention_verified: true`,
`norm_tensors_classified: 157`. Every non-gated norm tensor agreed with the
pin, so the run is not resting on an assumption about which arithmetic it used.

**Same-width prefills are bit-identical on real weights.** Row `k` of a full
prefill equals row `k` of a shorter prefill ending at `k`, to zero, on both
comparisons taken. The narrow-rerun determinism control is also zero. The
quantized nondeterminism is gone from the real path, not only from the fixture.

**Argmax agreement.** Across the width survey's 576 row-0 samples -- 96
positions, three widths, two indexer arms -- there are ZERO row-0 argmax flips.
That is the row a speculative verify commits from.

Stated exactly, because the difference matters for what the cap lift can
assume: the survey compares 1,728 rows in total, and **five of them flip**, all
on rows at or after the first, at near-tie margins:

| arm | width | row | max abs delta | narrow top-2 margin |
|---|---|---|---|---|
| indexer off | 4 | 3 | 3.06 | 0.394 |
| indexer off | 3 | 1 | 3.06 | 0.394 |
| indexer off | 4 | 1 | 3.06 | 0.394 |
| indexer on | 3 | 2 | 0.595 | 0.108 |
| indexer on | 4 | 2 | 0.595 | 0.108 |

Those are three distinct events at two prompt positions, each appearing at more
than one width. A verify wider than one token reads rows beyond row 0, so this
is the population that governs a cap lift, and it is not empty.

**Cross-width magnitudes.** Typical differences are of the 1e-3 class with
tails: worst row-0 delta 1.13 (width 2, indexer off), 2.84 (widths 3 and 4,
indexer off), 1.06 (width 2, on) and 6.75 (widths 3 and 4, on). The worst
any-row delta is 6.77. These are the numbers a tolerance-based criterion would
have to price; the argmax columns are what actually decides tokens.

**Engine against reference, on the pinned weights.** Maximum absolute logit
difference 2.12 over the compared rows, with ARGMAX AGREEMENT EVERYWHERE and
top-5 overlap 5 of 5 on all but one row (4 of 5). The post-prefill hidden state
agrees at cosine **0.9972** -- against 0.16 before the fixes. Both greedy
32-token continuations are **identical, 32 of 32 tokens**, and both are
readable English.

That last fact is the one that closes the original investigation: two
independent implementations, on the same checkpoint, now produce the same
coherent text. The `engine plain loop` and `engine serial leg` continuations
also match each other, so the session path and the plain loop agree.

**THE EARLIER MEASUREMENTS REMAIN VOID.** The first width survey and the first
parity capture were taken on code with three defects in it. Nothing in them is
salvageable as evidence -- not their numbers and not their conclusions,
including the "row 0 differs between widths" finding that started the whole
investigation. Cite this section, not those.

#### The acceptance basis -- RULED 2026-08-28

The re-survey plan's numeric criteria were written before any of this was
understood, and read against them several rows FAIL: engine causality at 1e-5
(cross-width rows land 0.07 to 1.10) and engine-versus-reference at 0.05 (rows
land 0.95 to 2.12). Reporting that as a failure would have been misleading, and
quietly widening the tolerances to make it pass would have been worse. David
ruled the basis instead:

| Criterion | Why it is the right kind of question |
|---|---|
| Argmax agreement | Tokens are what the model emits and what the score is computed from. A logit delta that does not move an argmax has changed nothing observable. |
| Same-width bit-identity | Within one width the computation is the same computation, so any difference is a defect. This is the criterion that catches the class of bug already found three times. |
| Greedy coherence | The only check that would have caught all three defects at once, and the one a human can read. |
| Hidden-state cosine | A single scalar over 2,560 dimensions that degrades gracefully, unlike a maximum over logits. |

Those four DECIDE. Absolute logit deltas are DIAGNOSTICS: they are what a
kernel-selection difference legitimately produces, and MLX dispatches different
kernels at different widths by design.

**A DIAGNOSTIC IS NEVER SILENTLY WIDENED.** The failure mode this rules out is
the one that nearly happened here -- a run misses a numeric bar, and the bar
moves. If a delta grows, that is a finding to report and explain, not a number
to adjust. A tolerance may change only as a stated, dated decision with the
measurement that motivated it, in the open.

**THE CROSS-CHECK.** mlx-lm, patched to compute `y * w` for the non-gated norms
so it reads this checkpoint's convention, is a SAME-CHECKPOINT cross-check: a
second implementation over the same weights, which is what caught the norm
defect and what confirmed the fixes. It is a cross-check and not an oracle --
it shares this port's upstream lineage, so a defect inherited from that lineage
would appear in both.

**THE ORACLE IS THE CUDA RUNNER ON test-spark.** Independent implementation,
independent hardware. When the two disagree, that is the comparison that
settles which one is wrong.

#### The width cap is LIFTED (David ruling 2026-08-28)

David ruled: "Yes -- lift the cap". The mtp leg now verifies at its resolved
draft depth; the serial leg still runs one token at a time.

`verifySegmentWidth` is no longer a constant. It is a per-session value -- 1 on
the serial leg, the resolved depth on the mtp leg -- and its doc comment states
the ORACLE-THROUGH-WIDE-PATH semantics: a wide verify commits what the wide
forward says, and the claim that makes it sound is ARGMAX AGREEMENT, not
bit-identity.

TWO TESTS DID NOT MOVE, deliberately. The rollback bit-identity assertion is
independent of verify width. The leg-identity test already required the
speculative leg to commit the serial leg's stream token for token at every
depth the envelope allows -- which is exactly the property the lift rests on --
so leaving it untouched is what makes it able to catch the lift going wrong.

THAT TEST IS FIXTURE SCOPE, and the distinction matters when it is quoted. It
proves token equality with the serial leg on the weight-free fixture; it does
not extend that to the pinned checkpoint, where the recorded population is five
near-tie argmax flips among 1,728 non-row-0 rows (5.2.1.4). Under the ruled
oracle-through-the-wide-path semantics a committed token is correct when it
matches the WIDE forward, so such a divergence is not an error -- but "the mtp
leg is token-exact against serial" is not a claim the pinned weights support,
and the participant contract's 11.4(c) states it that way.

**IT DID CATCH IT.** The first application of the lift crashed that test with
`Index out of range`. The accept loop indexed its predictions by the running
ACCEPT COUNT rather than by the segment's own position:

```
let index = accepted + offset      // wrong at any width above 1
```

At width 1 the two are the same number and the loop body runs once, so the bug
was invisible for as long as the cap was in place. At width 3 the index skips
an entry on the second offset and runs off the end on the third. It is
`start + offset` now, where `start` is the segment's first position in the feed.

That is worth stating plainly: the cap was not only costing the arm its
speedup, it was hiding a defect in the code the lift turns on. Nothing else in
the round needed changing -- the loop and `feedSegmented` were already
width-general.

VALIDATED ON THE FIXTURE: rollback bit-identity, wide-versus-narrow token
agreement, LEG IDENTITY AT EVERY DEPTH, both-legs-stop-at-the-same-token, the
head tape invariant and the serial-leg executor identity all pass with the lift
in place.

ONE THING TO MEASURE ON THE BOX, not assumed here: with a wide verify, a
rejected round leaves the target ahead of the accepted boundary, so the
restore-and-replay branch runs on nearly every round instead of rarely. That is
a second forward per round and it eats into the speedup the lift exists to buy.
### 5.2.1.5 THE CACHE-POSITION GATE REFUSED EVERY DECODE STEP

**The symptom.** Golden regeneration on the box, at engine `1b940ae2`, died at
its FIRST decode step:

```
runtime worker correctness failed: Qwen 3.8 position offset 1024
does not match cache offset 0 at layer 0
```

**The cause.** `gemma4Logits` calls `verifyQwenCachePosition` before every
forward, and that check required EVERY layer cache's offset to equal the
caller's position -- "lockstep", written when this tower was Gemma, whose two
cache classes both count positions. On Qwen 3.8, 36 of the 48 layers hold a
gated-deltanet RECURRENCE: a fixed-size state each forward overwrites, with no
key-value tape and therefore no position to count. Their offset reads zero
forever, which this port had already written down elsewhere --
`Qwen4ExpFreeRunSession.targetAttentionOffsets` says the recurrent layers "read
0 forever" and reads offsets only from the attention caches.

Layer 0 is a linear layer. So the first forward at any position past zero
refused, on the first layer it looked at.

**What it took down.** Every `gemma4Logits` path with `positionOffset > 0`:

* golden generation (`mlxfast-swift generate-golden`); and
* the teacher-forced correctness path -- the SCORED correctness leg.

**What it did not take down, which is why it hid.** The timed free-run path
does not call `gemma4Logits`: `Qwen4ExpFreeRunSession` drives the model
directly. So the width survey, the parity capture and the benchmark serial leg
all worked, and the failure looked like a golden-tooling problem rather than a
gate that refuses the scored correctness leg.

**Why the tests missed it.** The gate's unit tests build SYNTHETIC caches in
which every layer advances. One of them asserted the real stack is REJECTED --
"The Qwen tower's recurrent caches legitimately stayed at 0; Gemma's sliding
caches never do" -- so the defect was written down as intended behaviour and
tested for. That test is now the acceptance case.

**The fix.** Lockstep binds on the 12 FULL-ATTENTION layers, identified by the
pinned schedule (`fullAttentionInterval`) rather than by a cache class, so a
vendored rename cannot quietly turn it off. The recurrent layers keep their
topology check -- unbounded, no window -- and are not asked for a position they
do not have. A desynced attention cache still refuses BY NAME, now naming the
layer as a full-attention one.

Both copies of the check are updated identically. They live in two compilation
units on purpose -- only the worker's is competitor-reachable -- and until now
only `Gemma4RuntimePreflight.swift` had a parity test, so this pair could drift
silently. `bothCopiesOfTheCheckAreIdentical` compares the function's own text
in the two files.

**Trusted-scope and harnessHash.** The trusted twin is under `Sources`, which
is a `harnessHashRoots` entry, so this edit changes `harnessHash()` -- as does
any source change. Nothing needs re-pinning: the hash is computed at run time
over the roster and stamped into a sealed score, not compared against a stored
constant. The change is inside the trusted boundary and does not move it: the
check still lives in trusted scope, still reads the pinned schedule from
`MLXFastConstants`, and still refuses by name.

**The tests.** RED first, and with the production message: driving the real
decode loop through a prefill and four steps on a 48-layer fixture reproduced
`does not match cache offset 0 at layer 0` exactly. The suite covers the greedy
decode loop, the teacher-forced loop, a desynced attention cache on the real
stack, and the twin parity. The fixture is built at the PINNED layer count
because the gate requires one cache per layer -- an 8-layer fixture would have
been refused for a different reason and proved nothing.

### 5.2.2 The free-run session is native

Because the v2 engine cannot drive this tower, the timed free-run path is
`Qwen4ExpFreeRunSession`: one whole-prompt forward, then one `[1, 1]` forward
per round, through `model.newCache`.

That is the SAME implementation the teacher-forced correctness verbs run, which
settles the leg-implementation-identity question by construction rather than by
a fix. Both legs of a paired measurement and the pinned reference tapes are now
computed one way, so a token-exactness gate over the triple compares like with
like. `RuntimeWorkerFreeRunLegIdentityTests` pins it.

### 5.3 The goldens

**REGENERATED ON THE PINNED TARGET, 2026-08-28.** The two checked-in public
goldens carried the Gemma expected tokens and the Gemma `model_provenance`, so
every loader refused them and the tests asserted the refusal. They have been
regenerated on ranked hardware and the expectations are inverted: both now
LOAD, through `loadQwenGoldenFixture`, which pins the model identity on top of
the provenance check.

| Fact | Value |
|---|---|
| Engine commit | `4a1e000212859ceb13913d2f067cdc8d2a1ae32d` (merged main) |
| Target | `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` @ `327c8a604de613b42f84ba5e6b796c0931e8aa3b` |
| `model_type` | `qwen4_exp_text` |
| Generation | A and B, a fresh `generate-golden` process each, asserted byte-identical before either was pinned |
| Prompt file | `correctness_prompts/public_longcopy_gate_english_1024.txt`, sha256 `606968b5ee8b8c763057aab66cfac6f048e8bfac8a769a7330f3d7c5c9f0e290`, UNCHANGED |

| Golden | sha256 | bytes | steps |
|---|---|---|---|
| `public_longcopy_gate_english_1024_256.json` | `385af04443b1cbc1656ae9fcb3c403171ee925cc0ac557b96973f2629ff1261d` | 17,910 | 256 |
| `public_longcopy_gate_english_1024_1024.json` | `bc22a71eaffd4b51be696123ee6c1e636f76d81ba44c30faf844108194d9fe71` | 28,498 | 1,024 |

**THIS IS WHERE GOLDEN PROVENANCE IS RECORDED.** The generator also emits a
`provenance.json` beside the files; it is deliberately NOT checked in. This
table is the record, and `GoldenTests.checkedInPublicCorrectnessGoldenIsValid`
is the enforcement -- it pins both digests and both byte counts, loads each
golden at its own step count, and asserts the 256-step expected tokens are a
prefix of the 1024-step ones, which is what catches two files that came from
different runs.

**THE NEGATIVE CONTROL SURVIVED THE INVERSION.**
`aGemmaGoldenIsRefusedAgainstTheQwenTarget` keeps the property the old
assertions were protecting: a golden whose provenance names the Gemma
checkpoint is refused, and the MODEL-AGNOSTIC loader refuses it on the
provenance block, so no `model_type` edit talks one past. Without that, the
inversion would have removed the only thing standing between a Gemma golden and
a Qwen run.

**ABSENCE OF `model_provenance` IS NOW A REFUSAL.** The loader used to read
the provenance block when it was present and skip the repository and revision
check when it was absent, so a golden that named no checkpoint loaded with the
pin check never run. `loadQwenGoldenFixture` now requires the block, and the
refusal names the missing field. The requirement is opt-in on the wrapper, the
same way `requiredModelType` is, so the model-agnostic `loadGoldenFixture`
still accepts a reference-corpus golden that carries `model_type` and no
provenance. Every golden consumer in `Sources/` calls the wrapper, so no run
loads a golden without the check.
`GoldenTests.aGoldenWithoutModelProvenanceIsRefusedByName` is the enforcement:
it strips the block from the checked-in golden and asserts the refusal names
`model_provenance`, and it asserts the model-agnostic loader still accepts the
same file, which is what shows the requirement is held in one place.

**WHAT THIS UNBLOCKS.** `./benchmark.sh --local-iterate` reaches a golden for
the first time on this track: the correctness gate now has expected tokens
generated by this engine against this checkpoint, rather than a file every
loader refused.

**WHAT IT DOES NOT.** The HIDDEN correctness oracle in the track fixture stays
the `QWEN38-125B-A6B-MLX-PENDING-ORGANIZER` sentinel -- these are the PUBLIC
goldens, and section 5.4 still governs the arm state. The frozen-window
calibration record in `Sources/MLXFastCore/Constants.swift`
still names the OLD golden's digest and byte
count, and is deliberately left alone: it is a record of which artifacts a past
measurement actually used, and rewriting it to match today's files would
falsify it. Those Gemma-era baseline constants describe a run against the old
golden, which is one more reason the local-mode score estimate is an estimate.

### 5.4 The track is not armed

`fixtures/qwen3_8_125b_a6b_track.json` sets `official_scoring_enabled` to false.
The timed prompt pool is 8 sentinel slots. The hidden correctness oracle is a
sentinel. No runner advertises the
`[self-hosted, macOS, qwen3.8-125b-a6b-mlx-v1]` label set.
`official_scoring_enabled` flips true LAST, in its own change, after one clean
scored window.

### 5.4.1 The embedded head and the frozen target: RULED

Two rules meet on this track.

The TARGET QUANTIZATION IS FROZEN: `validateLoadedTargetQuantization` walks the
loaded model's quantized leaf modules and refuses any that does not carry the
pinned geometry. That is what stops a participant substituting a lossier target
for the accepted one.

A PARTICIPANT MAY RE-QUANTIZE THE HEAD, on load, in memory. On the tower this
tree was seeded from that was safe: the head was a SEPARATE checkpoint in its
own module tree, and the freeze did not reach it. Here the head is EMBEDDED --
`language_model.mtp.*` loads with the target, into the same module tree -- so a
whole-tree freeze reaches it.

THE RULING. David 2026-08-27, relayed by orchestrator, verbatim: "Exempt mtp.*
from the freeze". The head's own weights are re-quantizable on load like any
other head. The SHARED tensors stay frozen: the head owns no embedding table
and no output projection, it reads the target's, and those decide the TARGET's
tokens. A quantized module inside the head subtree that names `embed_tokens` or
`lm_head` is refused BY NAME, and the refusal says which shared tensor it
reached.

The rule is one total function over the module path,
`qwen4ExpHeadRequantExemption` in
`Sources/MLXFastHarness/Gemma4TargetQuantizationBind.swift`, which is trusted
scope. It is spelled there rather than read from the model so that an editable
model file cannot widen the exemption by renaming a subtree.
`Tests/MLXFastTests/Qwen4ExpEmbeddedHeadRequantTests.swift` pins BOTH halves --
a coarsened `mtp.fc_hidden` is accepted, a coarsened `model.embed_tokens` or
`lm_head` is refused by name -- plus the rule itself, driven directly, because
the abuse case (a shared tensor spelled inside the head) does not exist in any
module tree a test can build today, and refusing it if one ever appears is the
rule's whole job.

### 5.4.2 The gated suite must run serialized

With `MLXFAST_RUN_MLX_RUNTIME_TESTS=1` the suite HANGS intermittently under the
default parallel runner: the run stops mid-test and makes no further progress,
with the process alive and almost no processor time used. Observed twice.
`swift test --no-parallel` with the same variable passes 444 tests in about
5.6 seconds, every time.

The likely cause is shared process-global state across concurrently running
suites -- the width probe's router recorder is one known example, and several
suites now switch the default device -- but the cause is NOT yet proven, so
this is recorded as an OBSERVATION and not as a diagnosis.

Continuous integration is not affected today: it sets
`MLXFAST_RUN_MLX_RUNTIME_TESTS` to `0`. A BOX run is affected, so a box session
must pass `--no-parallel`.

### 5.5 The editable-surface byte budget

`maxTotalBytes` is RE-DERIVED for this tree, by the same method the previous
value used: the ENFORCED at-rest editable surface plus a stated margin of
exactly 1 MiB (1,048,576 B = four times `maxGrowthBytes`).

    3,150,302 (at rest) + 1,048,576 (margin) = 4,198,878

Old value 4,404,587, new value 4,198,878. NOTHING IS EXEMPT on this track, so
the at-rest figure and the enforced figure are the same number and there is no
exempt-byte double count to get wrong. The old value would still have PASSED the
margin check, which is why it is re-derived rather than left: carrying it leaves
1,254,286 bytes of unexplained slack a submission could fill.

The number appears in five places and they move together: `benchmark.json`, the
Swift enforcer's `defaultMaxTotalBytes`, the shell enforcer's fallback in
`.github/scripts/submission-static-review-checks.sh`, the participant contract
and README tables, and the manifest test's mirror.

The cap was raised again on 2026-08-30, from 4,204,273 to 9,447,153 (David
ruling: add 5 MiB, 5,242,880 bytes). The enforced at-rest surface on main had
reached 3,156,707 bytes, leaving 1,047,566 bytes of margin -- under the stated
1 MiB minimum, so `enforcedSurfaceStaysUnderTotalCap` failed on main and on
every submission PR. The margin at the raise is 6,290,446 bytes. This is the
dev-repo companion to public PR #53, which applied the same +5 MiB raise on the
same 1-MiB-margin tripwire. All five declaration sites moved in the same commit;
the 1 MiB minimum-margin tripwire is unchanged.

### 5.6 Reference parity: how to re-run it

The cross-implementation comparison that found the QSA defect is a laptop
procedure, and it is worth repeating for any future port question. It vendors
NO reference code; it only runs the reference and compares numbers.

1. Make a scratch virtual environment and install MLX.
2. Clone `ml-explore/mlx-lm` by SHA (this port's source is `c961f839`) into the
   scratch area and install it there.
3. Build a model at a REDUCED-REAL config -- real head dims, head counts,
   expert count and indexer parameters, reduced depth, vocabulary and expert
   width -- with random weights. Write its parameters and its own logits as
   safetensors.
4. Load the SAME parameters into `Qwen4ExpModel` through `sanitize`, run the
   same token ids, and compare row by row.

Two properties make this sharp. A single-token forward agrees to fp32 noise
when the port is faithful, so any disagreement there is a projection or a norm.
A multi-token forward that disagrees ONLY on some rows names the component: the
QSA defect showed up exactly on rows at or past the indexer budget, and raising
the budget made it vanish.

The parameter TREES can also be diffed on their own, without running anything:
identical names and shapes rule out a mapping error in one step. On the pinned
geometry the two trees match exactly, except for the 128 n-gram shard tensors
this port deliberately keeps off the parameter tree (section 5.1).

## 6. Sentinels this port left in place

Each of these waits for organizer material that does not exist yet.

| Location | Waiting for |
|---|---|
| `fixtures/qwen3_8_125b_a6b_track.json` `timed_prompt_pool[].r2_path` | the 8 timed pool prompts, uploaded to R2 under the track prefix |
| `fixtures/qwen3_8_125b_a6b_track.json` `timed_prompt_pool[].sha256` | the same, identified by sha256 |
| `fixtures/qwen3_8_125b_a6b_track.json` `hidden_correctness_golden` | the hidden oracle's sha256 and byte count |

The three inventory sentinels this port started with are CLOSED. A safetensors
header sits at the start of its file and its length is the file's first eight
bytes, so each header can be read with two bounded range requests. That is
metadata, not weight bytes, and it needs no download of the 105 GiB tree. All
22 headers were read that way and the fixture now carries:

* `shard_headers` — per shard, the header length and the sha256 of the header
  bytes;
* `summary.dtype_counts` — BF16 2724, U32 1020, I64 3, which sums to the pinned
  tensor count of 3747;
* `summary.tensor_shapes` — a digest over every `[name, dtype, shape]` triple,
  with its canonicalization stated beside it.

The headers also CONFIRM the counts the fixture derives from the index: 3335
under `language_model.model`, 76 under `language_model.mtp`, 3 under
`language_model.lm_head`, 333 under `vision_tower`, and 384 n-gram shard
tensors. They confirm the n-gram row geometry as well: each shard weight is
`U32 [2500012, 20]` and each scale and bias is `BF16 [2500012, 5]`, which is
160 values a row at 4 bits with group 32, so 100 bytes a row and 29.80 GiB for
the table.

## 7. Fixtures

| Fixture | State |
|---|---|
| `fixtures/qwen3_8_125b_a6b_config.json` | LANDED. The published `config.json` re-rendered for JSON formatting only: 2-space indent, sorted keys, trailing newline. This is a laptop-side public fetch, not a box-verified artifact. |
| `fixtures/qwen3_8_125b_a6b_tensor_inventory.json` | PART LANDED. Index-derived facts are real; shard-header facts carry the sentinel. |
| `fixtures/reference_qwen3_8_125b_a6b_4bit.sha256` | LANDED. 32 records, 113,233,030,116 bytes. |
| `fixtures/qwen3_8_125b_a6b_track.json` | LANDED, UNARMED. |
| `fixtures/gemma4_26b_a4b_config.json` | RETAINED. The Gemma model code still in the tree validates against it. It moves when that code moves. |
| `fixtures/qwen3_6_27b_*` | RETAINED. Live negative control. |
| `fixtures/poolside_laguna_xs_2_1_nvfp4_*` | RETAINED. Generic transform-test substrate. |

## 8. The vendored model port and its licence basis

The vendored tower is a Swift port of the MIT-licensed mlx-lm reference
implementation (ml-explore/mlx-lm pull request 1788, `mlx_lm/models/qwen4_exp.py`
at head `c961f839`). The native multi-token-prediction head follows the
Apache-2.0 vLLM reference (vllm-project/vllm pull request 53896,
`vllm/models/qwen4_exp/nvidia/mtp.py` at head `2a4cd640`), which is the only
permissively licensed description of that head. No source under a copyleft
licence was read or ported. `THIRD_PARTY_NOTICES.md` records both.

| File | Holds |
|---|---|
| `Vendor/.../MLXLLM/Models/Qwen4Exp.swift` | configuration, decoder layer, tower, model, sanitize |
| `Vendor/.../MLXLLM/Models/Qwen4ExpText.swift` | norms, partial rotary, QSA indexer, attention, gated deltanet, mixture of experts, hyper-connections |
| `Vendor/.../MLXLLM/Models/Qwen4ExpNGram.swift` | the n-gram hash, the PLE layer, the row-source seam |
| `Vendor/.../MLXLLM/Models/Qwen4ExpMTP.swift` | the head embedded under `mtp.*` |
| `Vendor/.../MLXLMCommon/Qwen4ExpCaches.swift` | the attention cache that carries the QSA indexer tape, and the four-slot linear-layer cache |
| `Sources/MLXFastModel/Qwen4ExpNGramTable.swift` | the disk-resident n-gram table and its bounded row cache |

Two facts about the tower are easy to get wrong and are pinned by tests:

* There is NO final `norm` tensor. The last hyper-connection mixer stands in for
  it.
* The MTP head reads the hyper-connection stream BEFORE that mixer, which is
  `hc_count * hidden` wide, not the collapsed hidden state.

The n-gram table is never a model parameter. It stays on the solid-state disk
and reaches the model through an injected row source, with a bounded row cache
that cannot change a value. `docs/ngram-cache-design.md` is the contract, and
its contract section is engine-neutral because the CUDA engine follows it too.

## 9. The scoring shape

### 9.1 The ruled constants and where they are cited

`benchmark.json` `scoring.*` and the track fixture carry values only, with no
prose fields (David ruling 2026-08-24). The citation for each value lives in
`docs/participant-contract.md` section 5 and in the header of
`tools/lint-benchmark-manifest.py`, which pins the whole block and refuses a
drift.

`pairsPerCohort` and `minPairsPerCohort` are 2 (David 2026-09-09: "move to 2
pairs on both mlx and cuda for now"; "1 pair is not sufficient"). The count
benchd enforces is the track fixture's `official_pairs`: benchd reads it from
`--contract` and refuses a ranked run whose fixture does not declare it. Each
pair is one serial-control leg on the reference tree and one candidate leg at
the declared depth; per role the per-token times are summed over the pairs and
the score is the ratio of the sums. `benchmark.json` carries the same number
for readers, and the lint fails when the two disagree.

The speedup floors are 0.95 on BOTH axes (David 2026-09-09): a candidate that
regresses prefill or decode by more than 5 % is refused. They are track settings
too, and the enforced values live in the fixture as `decode_speedup_floor` and
`prefill_speedup_floor`; `benchmark.json` carries the same two numbers and the
lint cross-checks them. The ceiling stays 5.0. Before 2026-09-09 this repository
declared a decode floor of 0.90 and no prefill floor at all. Both were authoring
errors: 0.90 is the `qwen3.8-27b-mtp-v1` free-run constant, and the Gemma-era
Swift score constants this repository still carried at the time had read 0.95
throughout. Those Swift constants are now DELETED (section 9.1.1): the fixture
is the only place a floor is stated.

### 9.1.1 No Swift scoring path, and no stored baseline pair

This engine computes NO score. benchd measures, scores and seals; the engine
reports raw timings. The tree used to carry a second, unused scoring path from
the Gemma era: `Sources/MLXFastCore/Score.swift` (the `BenchmarkScore` estimate,
the `ScorePayload`/`ScoreMetrics` score.json writer and its diagnostic
coarsening) plus the constants it read --
`officialBaselinePrefillSecondsPerToken`, `officialBaselineDecodeSecondsPerToken`,
`scorePrefillWeight`, `scoreDecodeWeight`, `scorePrefillSpeedupFloor`,
`scoreDecodeSpeedupFloor` and `publicDiagnosticSignificantFigures` -- and the
`BenchmarkGolden.resolvedBaseline*` accessors that fell back to the stored pair.

NOTHING CALLED ANY OF IT. The CLI verbs are `transform`, `verify-transform`,
`attach-benchmark-oracle`, `analyze-ngram-similarity`, `checkpoint-shards` and
`mtp-verify`; `tools/*.sh` and the workflows call benchd for every score. The
only callers were the path's own tests. All of it is DELETED. The two stored
baseline seconds-per-token were the last file-held baseline PAIR in the tree,
which `docs/participant-contract.md` section 5.1.0 says must not exist: a ranked
run measures its own control leg on the box (David ruling 2026-09-08).

NO STORED TIMING VALUE REMAINS. The last two,
`gemma4MTPOfficialBaselineDecodeSecondsPerToken` and
`gemma4MTPOfficialBaselinePrefillSecondsPerToken`, are DELETED (2026-09-09,
David ruling). They were `qwen3.8-27b-mtp-v1` calibration records -- ANOTHER
track -- carried in this tree; the decode one was even labelled SCORED and
described as "the serial denominator of this track's paired ratio". A grep over
`Sources/`, `Runner/`, `Tests/` and `tools/` found each one's only occurrence to
be its own declaration, so no test asserted either value and nothing else moved
with them. `Sources/MLXFastCore/Constants.swift` now holds no seconds-per-token
at all: a ranked run measures its own control leg on the box, and section 5.1.0
of `docs/participant-contract.md` says a stored baseline must not exist.

### 9.2 Single-stream only (RULED)

David ruling 2026-08-27, relayed by orchestrator: "Single-stream only". The
scored value is a PAIRED serial-against-MTP comparison, run ONE STREAM AT A
TIME, at `scored_batch_size` 1. The batched ContinuousBatchingV2 adaptation is
NOT pursued. Section 5.2 states why the batched path cannot drive this tower.

SUPERSEDED 2026-09-08. The sentence that stood here said each of the 8 pinned
prompts runs in its own window and the 8 elapsed times are summed. That was the
pool-timing shape. The ranked run has been the paired per-box shape since David's
2026-09-08 ruling: it times the ONE prompt `live_golden` names, over
`official_pairs` pairs, and it sums each role's per-token times over the pairs.
The pinned 8-prompt pool is the CORRECTNESS pool.

THE QUOTE IN THE FIXTURE IS NOW THIS TRACK'S OWN. It used to be the Gemma-era
ruling, which opened "we want to score gemma's benchmark" and described a sum
over 8 concurrently timed streams -- a real David quote naming another model
and a regime this track does not run, sitting inside this track's contract.
David re-issued it on 2026-08-27 (relayed by orchestrator), and
`scoring_semantics.ruling_verbatim` now carries exactly one string: "score the
qwen 3.8 125b-a6b tracks (mlx and cuda) single-stream, paired serial vs the
built-in mtp, on prefill gains ^ .25 * decode ^ .75". The fixture carries
`ruling_date` beside it, and the fixture test pins the whole string rather than
a phrase from it.

WHAT CHANGED IN THIS REPOSITORY:

| Place | Value |
|---|---|
| `fixtures/qwen3_8_125b_a6b_track.json` | `scored_batch_size` 8 to 1 |
| `benchmark.json` | `scoredBatchSize` 8 to 1 |
| `benchmark.json` | `scoring.mode` to `qwen-native-mtp-paired-decode-only` |
| `tools/lint-benchmark-manifest.py` | the pinned expectation block |
| `docs/participant-contract.md`, `README.md` | sections 5 and 11.4, and the scoring tables |
| `Tests/.../Qwen38A6BTrackFixtureTests.swift` | the width test |

The mode string is NOT invented here. `qwen-native-mtp-paired-decode-only` is
benchd's own single-stream paired regime name: `overlay::SCORING_MODE`, the
same string as `measure_job::MEASURE_JOB_MODE`, read off the bench repository at
`a7be295b`.

WHAT THIS LEFT TO THE BENCH LANE, as of 2026-08-27. At the channel tip of that
day, `ScoredBatchPoint::certify` certified B = 8 and nothing else, so a fixture
declaring `scored_batch_size` 1 was refused at width certification, and the
composite was computed on the batched cohort regime only. This repository
declared a shape the published benchmarker refused. That was deliberate and
fail-closed. It is history: the lane below has since merged, and scored runs
happen.

THE BENCH LANE HAS MERGED, and this is verified, not assumed. At the release
branch tip `56a9821a` (pull request 217 on the development bench repository,
merged) `effective_candidate_regime` reads
`None | Some(SCORED_BATCH_SIZE_SINGLE_STREAM) => Ok(spec_regime)`: width 1
keeps the single-stream regime and never reaches the cohort width match. The
same file carries `scored_exponents` with the field names
`prefill_gain_exponent` / `decode_gain_exponent`, which is exactly what this
fixture declares. That tip is a CITATION for the behaviour, read when this was
written; what the channel serves is whatever its manifest names at resolve time
(section 4.3), which is that lane or later.

### 9.2.1 The score emitter now publishes the composite

`.github/scripts/emit-qwen38-125b-a6b-score.sh` converts benchd's
`results.json` into the `{score, metrics}` shape the scorer takes. Its
single-stream branch used to publish `aggregate.raw_decode_speedup_median`, the
even-n median of the per-prompt DECODE ratios. That is not the ruled formula:
it has no prefill component and no exponents.

The branch now publishes `composite.composite_score`. A single-stream run has
no cohort record, so benchd seals the composite at the TOP LEVEL of the record
-- the same `CompositeCohortScore` object the cohort series hangs on
`per_cohort[]` -- beside `composite_scored_exponents`, with exactly one of
`composite` / `composite_absent_reason` present. benchd's overlay publishes the
same number under the aggregation discriminator
`shared_window_composite_prefill_decode_gain` with a `single_stream_composite`
block. Field names read off the merged bench release tip `56a9821a` (pull request
217; `crates/benchd/src/measure_job.rs`, `crates/benchd/src/overlay.rs`),
not invented here.

A single-stream record that seals NO composite is REFUSED, not scored by the
old rule. That is the regression that mattered: publishing the decode median
for a record that predates the composite would silently score a different
formula under the ruled name. The median is still forwarded in `metrics` as a
diagnostic.

Two drift tripwires run there, because benchd reports both values and wires
neither to an exit code: a run whose `composite_speedup_floor_met` is false is
refused, and so is a run whose sealed `composite_scored_exponents` differ from
`benchmark.json` `scoring.scoredExponents`.
`tools/test-qwen38-125b-a6b-score-emitter.sh` pins the composite score, the
no-composite refusal, the below-floor refusal and the exponent-drift refusal,
and that the decode median is never published as the score.

### 9.3 The prefill window, and what the engine owes it (RULED)

David ruling 2026-08-27: the composite is
`prefill_gain^0.25 * decode_gain^0.75` on this single-stream track. The bench
side adds a prefill window to the single-stream free-run verbs and certifies
the exponents on the B = 1 point. The ENGINE half is the subject here.

NO NEW MESSAGE AND NO NEW FIELD. The verbs stay `free_decode_begin` +
`free_decode_run`. benchd splits its own parent clock at the verb boundary: the
prefill window is begin-sent to validated `seed_token`, the decode window is
from there to `free_decode_run(N)` returning, `elapsed` is their sum, and
`seconds_per_token` is unchanged. The captured engine-wire fixture
(`ENGINE_WIRE_V1_SHA256`) is therefore UNCHANGED by this work, which the wire
fixture test confirms.

WHAT THAT MAKES CHEATABLE, and it looks like an optimisation. Deferring seed
work out of `begin` and into `run` does not remove the work: it moves it from
the prefill window into the decode window. `elapsed` and `seconds_per_token`
are identical, so no timing gate notices -- but the exponents are 0.25 and
0.75, so the composite MOVES. Nothing on the wire can detect it, so the engine
obligation is held by tests.

WHERE THE ENGINE MEETS IT:

| Obligation | Where |
|---|---|
| `begin` runs the FULL seed prefill and replies only after it completes | `Qwen4ExpFreeRunSession.init` -- one forward over the whole seed, greedy argmax at the last position, and an explicit `eval` of BOTH cache stacks before it returns |
| `run` never re-feeds the seed | `Qwen4ExpFreeRunSession.run` starts from `pendingToken` and the stored stream; a serial round feeds one token, an mtp round feeds the pending token plus accepted drafts |
| the mtp leg does not re-prefill on rollback | snapshot and replay touch DRAFTED tokens only; the full-attention offsets land exactly on `seed + N` |
| nothing prefills before `begin` arrives | the constructor warmup is prompt-independent (constant beginning-of-sequence tokens, throwaway cache) and runs BEFORE the protocol hello, outside every window benchd clocks |
| the hello advertises `free_run_decode` only | `runtimeWorkerAdvertisedCapabilities`, already true since the cohort refusal |

WHY `eval` IS THERE. MLX is lazy. The greedy argmax forces the logits and
everything they depend on, but the caches the seed forward WROTE are separate
arrays, and an unevaluated one is paid for by whoever touches it next -- which
is the decode window. The session therefore forces the target cache state and
the head cache state before `begin` returns. On an already-evaluated graph it
is a no-op; when it is not a no-op, it is exactly the work that belongs on that
side of the boundary.

HOW IT IS PINNED. `Tests/MLXFastTests/Qwen4ExpPrefillWindowTests.swift` counts
every token pushed through the target (`targetTokensFed`) and reads the
full-attention cache offsets (`targetAttentionOffsets`), on the serial leg AND
the mtp leg:

* after open: fed equals the seed length, offsets equal the seed length;
* after `run(N)`: offsets equal `seed + N` on both legs; the serial leg fed
  exactly `seed + N`, and the mtp leg fed at least that -- its rejected-draft
  forwards are real work and are counted, while the offsets are what refuse a
  seed re-feed;
* a `run(0)` changes neither, so the decode window opens on the prefill
  window's closing state.

The offsets are read from the 12 FULL-ATTENTION layers only. The 36
gated-deltanet layers hold a recurrence with no tape and no position to report;
`targetTokensFed` is what covers them.

NEGATIVE CONTROL RUN. With the opener changed to defer the last 4 seed tokens
into the run window, the suite fails on the fed count and on both offset
checks; restored, it passes. The tests are not vacuous.

THE MECHANISM PIN IS A SOURCE TEST, deliberately. MLX exposes no way to ask an
array whether it has been evaluated, so no runtime assertion can tell "the
prefill finished inside begin" from "the graph was built inside begin and paid
for later". The counters pin WHICH tokens each verb consumes; one source-text
assertion pins that the seed forward is forced before begin returns.

## 10. The unified MLX and CUDA user experience is flagged for later

David, 2026-08-28: the unified experience -- a `cuda.fast` / `mlx.fast` router
toggle with both lines on one chart -- is designed and deferred. It does not
gate engine work. What stays held is the board, the tracks and the chart.

The score emitter's aggregation label was held under the earlier reading of
that ruling and is RELEASED. It landed here: on a composite-scored
single-stream record the emitter used to forward benchd's
`aggregate.scoring_aggregation`, which reads
`median_of_per_prompt_raw_serial_relative_speedup` -- the DECODE-ONLY median's
name, beside a score that is a weighted product of two gains. Nothing was
scored wrongly, because the board keys on the series and the composite, but the
published blob told a reader the wrong thing about its own number. It now
forwards `shared_window_composite_prefill_decode_gain`, which is benchd's
`AGGREGATION_COHORT_COMPOSITE` (`crates/benchd/src/overlay.rs`), and keeps
benchd's own aggregate name under `decode_median_aggregation` so the diagnostic
beside it stays labelled correctly too.

## 11. Follow-ups this port has not closed

* **REPEATED FORWARDS OF A QUANTIZED FIXTURE MODEL GO NaN.** Found 2026-08-28
  while chasing a flaky test. On the small fixture, with the SAME tokens and no
  perturbation, three prefills through a freshly quantized model give: run 1
  clean, run 2 clean but differing from run 1 by 1.30, run 3 ENTIRELY NaN
  (5,120 of 5,120 logits). The same three prefills on the UNQUANTIZED fixture
  are bit-identical with no NaN. A fresh cache is built for each call, so the
  corruption is in the model or in what MLX holds for it, not in the cache.
  The vendored `GatedDelta.swift` already carries an upstream note about a
  Metal kernel recompile between turns triggering a use-after-free when a
  quantized cache is reused, which is the first place to look.
  SCOPE IS UNKNOWN AND MATTERS: the ranked engine loads an already-quantized
  checkpoint by a different path and its narrow decode produced plausible
  tokens on the box, so it is not identically broken there -- but a survey that
  performs hundreds of forwards on a quantized model is exactly what this would
  poison. It cost the QSA causality suite its quantized arm (that arm was
  comparing two prefills that already disagreed by more than the effect it was
  looking for).

* The pinned-weights width survey diverges in its "indexer off" arm, below the
  indexer budget, where the keep mask is never built. Section 5.2.1.1 fixed the
  arm above the budget only.
* The box parity capture found this engine and the reference disagreeing on the
  pinned checkpoint on the SINGLE-TOKEN path, with a post-prefill hidden cosine
  of 0.16, while they agree to 1.9e-7 on shared random weights. The norm
  convention (5.2.1.2) is a shared defect and does not explain a DISAGREEMENT
  between them; that one is still open.
* `Tests/MLXFastTests/Model/Qwen4ExpArtifactFixtureSupport.swift` still names
  the Gemma repository and revision in its prose while the fixture bytes it
  pins are this track's `config.json`. Stale text, no behaviour.

## 12. Re-measuring on the box

`docs/box-resurvey-plan.md` is the procedure. It re-runs the width survey and
the reference parity comparison on the fixed engine, with the chat-template
prompt shape and the norm convention matched on both sides, and states the pass
criteria so the run is mechanical.

IT HAS BEEN RUN, on 2026-08-28 against `1b940ae2`. Section 5.2.1.4 records what
it found. Its numeric criteria were the wrong question and David ruled the
acceptance basis on the same day; the plan's pass criteria now state the ruled
basis, with the absolute deltas kept as diagnostics.

## 13. The Darkbloom CBv2 shape

This section replaces the private engine that sections 5 and 8 describe.

### 13.1 What changed

The repository held its own inference engine and its own runtime worker. Both
are deleted. The engine is now the `Layr-Labs/mlx-swift-lm` fork, and this
repository holds that fork as a git submodule at `Vendor/mlx-swift-lm`.

Three parts of the fork do the work:

| Part | What it does |
|---|---|
| `Libraries/MLXLLM/Models/Qwen4Exp*.swift` | the Qwen 3.8 Flash-Next model, the QSA indexer, the embedded MTP head, and the disk-resident n-gram table. |
| `Libraries/MLXRunners/Qwen4ExpRunner.swift` | loads the checkpoint once. It vends the CBv2 engine and a one-row teacher-forced stepper over the same model. |
| `Executables/bench-worker` | the Engine Protocol v1 server. One binary serves every model family. The checkpoint's `model_type` selects the runner. |

The Darkbloom runner contract is the contract these parts obey.

### 13.2 The pin, and why it is a branch

The submodule points at `449f2d0`, on branch `feat/qwen38-flash-next-runner`. That
branch is not merged. It is the only head that carries all four pieces the
track needs together: the model port, the scaffold with `--resource` parsing,
the runner, and the n-gram table reader.

Re-pin the submodule to the fork's `main` after the scaffold, the round-audit
journal and the runner merge.

### 13.3 The two Vendor entries are different things

`Vendor/mlx-swift` stays a VENDORED TREE. Its Metal kernel sources are the
track's optimization surface, so its files must be editable.

`Vendor/mlx-swift-lm` is a SUBMODULE. A gitlink names a commit, not bytes in
this tree, so it is not an editable path.

The fork declares its own `mlx-swift` dependency as a floating branch URL.
SwiftPM resolves the root package's local `Vendor/mlx-swift` ahead of it,
because a local path dependency of the root package wins over a remote
dependency with the same package identity. Every target in the graph therefore
builds against `Vendor/mlx-swift`. SwiftPM prints a "conflicting identity"
warning when it applies that override, and it states that a future SwiftPM
version will make the condition an error. The stable fix is a fork change: the
fork must use `../mlx-swift` when that directory is present.

### 13.4 How the benchmarker starts the engine

benchd starts the engine as `<engine> runtime-worker --weights <dir>`. The
engine is the staged `bench-worker` binary at `.build/release/bench-worker`.
`tools/stage-bench-worker.sh` puts the binary and its `mlx.metallib` there as a
pair, because Metal loads the library from the directory of the running binary.

The n-gram table is 29.8 GiB and it is never model parameters. The runner
builds a disk-resident row source from a named resource:

```
--resource qwen4exp.ngramRowSource=<shard directory>
```

`fixtures/qwen3_8_125b_a6b_track.json` `ngram_shard_dir` names the directory.
`tools/qwen38-125b-a6b-measure-and-score.sh` reads it from the fixture and
forwards it. Without the resource the runner refuses at load and names it.

### 13.5 What this repository still owns

| Path | What it does |
|---|---|
| `setup.sh` | downloads and verifies the target checkpoint, builds the CLI and the engine, builds `mlx.metallib`, and stages the engine pair. |
| `Sources/MLXFastTransform` | the offline transform and its checkpoint validation. |
| `Sources/MLXFastCLI` | the trusted CLI: `transform`, `verify-transform`, `checkpoint-shards`, `analyze-ngram-similarity`, and the two golden attach verbs. |
| `Sources/MLXFastTrustedHarness` | the editable-surface byte budget, the head declaration reader, transform verification, and the `mlx.metallib` fingerprint. |
| `tools/build-mlx-metallib.sh` | builds `mlx.metallib` from `Vendor/mlx-swift` with `MLX_METAL_JIT` off. |
| `benchmark.json`, `fixtures/` | the track manifest and the track contract. |

### 13.6 What is not proven yet

* The pinned benchd builds the engine spawn argv itself. It has no flag that
  forwards `--resource`, and it always adds `--speculative-protocol v1.1`,
  which the fork's `bench-worker` does not accept. Both need a change outside
  this repository before a scored run can start.
* `benchd correctness --manifest <path>` does not exist in the pinned benchd,
  so the runner manifest is not checked against the wire yet.

### 13.7 ONE RESIDENT bench-worker PER WINDOW

**WHAT THE RESIDENT IS.** `bench-worker resident` is a second verb on the same
staged binary the benchmarker starts. It loads the checkpoint once, binds a
Unix socket, and serves every session that connects to it. It is the OWNER of
the weights for one benchmark window.

**WHY.** benchd starts `bench-worker runtime-worker` once per phase -- the
warmup, the timed prefill, the timed decode, the correctness pass, and again
for each leg. In process, each start loads the whole 113 GB checkpoint. David
ruled on 2026-08-30 that the weights load ONCE per window. The resident is how
this track keeps that rule: one load, and each per-phase worker attaches.

**ONE PER LEG, NOT ONE PER WINDOW -- AND benchd BOOTS IT.** A paired job has
two legs on two trees with two weight directories, so one resident cannot serve
both. Public run 34230122059 proved it: the measure script booted ONE resident
from the CANDIDATE tree and exported `BENCH_WORKER_RESIDENT_SOCKET` into benchd,
the reference leg attached to it, and benchd refused the phase --
"resident holds `<candidate>/weights` but this phase asked for
`<baseline-workspace>/weights`". The refusal was right. The topology was wrong.

Only benchd knows where a leg begins and ends, so benchd boots each leg's
resident, from that leg's own tree:

```text
<GPU lock holder = ./tools/qwen38-125b-a6b-measure-and-score.sh>
  -> benchd iterate --mode official
     -> <baseline workspace>/tools/resident-up.sh --boot --spec serial --draft-len 0 --socket-out F
        -> bench-worker resident        (the reference tree's engine + weights)
     -> bench-worker runtime-worker     (attaches to leg 1, loads nothing)
     -> <baseline workspace>/tools/resident-up.sh --stop --socket <F line 1>
     -> ./tools/resident-up.sh --boot --spec mtp --draft-len N --socket-out G
        -> bench-worker resident        (the candidate tree's engine + weights)
     -> bench-worker runtime-worker     (attaches to leg 2, loads nothing)
     -> ./tools/resident-up.sh --stop --socket <G line 1>
```

`--boot` writes the socket path as the first line of `--socket-out`, plus a
`<socket>.pid` sidecar and a `<socket>.ready` marker, and exits 0 with the
resident RUNNING. `--stop` ends it and removes all three; a second `--stop` is a
no-op. `--spec` is authoritative: nothing in the boot reads the tree's
`mtp-head.manifest.json`, so a reference tree that declares a draft depth still
boots a serial control leg. The boot ignores `MLXFAST_ENGINE_BIN` and
`MLXFAST_WEIGHTS_PATH` for the same reason -- the ranked job exports those for
the candidate, and a leg is defined by its tree.

The measure script boots nothing and exports no socket. An inherited
`BENCH_WORKER_RESIDENT_SOCKET` is refused by name, there and in
`tools/ranked-box-preflight.sh`.

**THE GPU LOCK.** A resident holds about 113 GB of unified memory whoever booted
it, so the window must be exclusive. The measure script takes
`/tmp/mtplx-gpu-exclusive.lock` FIRST and holds it for the whole measurement:
the lock holder is the outermost process and becomes the script again through
`execv`, so the lock lives exactly as long as the run. `tools/resident-up.sh`
never takes the lock -- a lock it took would end when it exits, which is not
the window -- and it REFUSES to boot when nobody holds it, so every per-leg boot
benchd makes is inside that window. A wait of `MLXFAST_GPU_LOCK_TIMEOUT_S`
seconds (1800 by default) ends in a refusal, never in a measurement beside
another owner.

**ONE RESIDENT PER BOX, CHECKED ACROSS TREES.** Each tree's boot writes its
pidfile under its own tree, so neither can see the other's. `--boot` therefore
also scans the box for a live `bench-worker resident` and refuses when it finds
one: that is what catches leg 2 booting before leg 1 was stopped, which would
double-load 113 GB.

**THE NAMES.**

| Name | What it is |
|---|---|
| `BENCH_WORKER_RESIDENT_SOCKET` | A leg's resident Unix socket. The ranked path never sets it and REFUSES when the environment does. |
| `RESIDENT_IDENTITY_FILE` | JSON: the resident, its pid, its socket, its declared leg, its hello. |
| `RESIDENT_UP_LOCK_PATH` | The GPU lock file. Default `/tmp/mtplx-gpu-exclusive.lock`. |
| `MLXFAST_GPU_LOCK_TIMEOUT_S` | Ceiling on the wait for that lock. Default 1800. |
| `MLXFAST_GPU_WINDOW_HELD` | Set to `1` by the measure script on the inner run, inside the lock it took. Never set by hand. |

**WHAT benchd DOES WITH THE SOCKET.** benchd boots each leg's resident, reads the
socket from the `--socket-out` file, and passes
`BENCH_WORKER_RESIDENT_SOCKET` to the per-phase workers of that leg only. A
channel binary that does not yet boot per leg starts per-phase workers that each
load the weights themselves; the run measures what it measured before, and the
only cost is repeated loads.

**THE WRAPPER FORM SURVIVES, FOR LOCAL USE.**
`tools/resident-up.sh --weights <dir> -- <command>` boots one resident, exports
the socket to the command and halts it afterwards. After this change it has no
caller in this repository: its users are a participant's hand run and anything
that wants the socket exported around a command, and `tools/benchmark.sh`
honours the variable it exports.

**THE STAGED SET.** `tools/stage-bench-worker.sh` copies the worker,
`mlx.metallib` AND `mlx.metallib.fingerprint` into `.build/release`. The
harness reads the fingerprint record at `<metallib path>.fingerprint`, which is
the STAGED metallib on a ranked run, so a metallib staged without its sidecar
fails that check with "no fingerprint record" and an official run treats that
as fatal. A metallib that arrives with no sidecar is a refusal in the stager,
not a skip.
