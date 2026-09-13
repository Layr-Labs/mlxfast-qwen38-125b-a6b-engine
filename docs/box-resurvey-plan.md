# Box re-survey plan (DRAFT)

This document tells an operator how to re-measure the Qwen 3.8 125B A6B MLX
engine on the box. It is an operations document. It does not define any part of
the participant flow.

RUN ONCE ALREADY, on 2026-08-28 against `1b940ae2`; engine port notes 5.2.1.4
records the outcome and the acceptance basis David ruled the same day. This
document is the procedure for repeating it.

Two measurements in this repository were taken on defective code and are VOID:

* the wide-versus-narrow width survey, and
* the parity comparison against the mlx-lm reference.

Two defects have landed since. The QSA keep mask reached forward
(port notes 5.2.1.1). The norm convention was wrong for this checkpoint
(port notes 5.2.1.2). Both change what the tower computes, so both
measurements must be taken again.

Read the whole document before you start. Every step is read-only except the
engine build.

## 1. What you need

| Item | Value |
|---|---|
| Engine | the MERGED main branch, after the indexer fix and the norm-convention fix |
| Weights | the pinned tree, `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP` at revision `327c8a604de613b42f84ba5e6b796c0931e8aa3b` |
| MTP head source | the two 8-bit shards beside the weights, `Vontra/Qwen3.8-Flash-Next-MLX-8bit-MTP` at revision `9c306179562765396e197a8a7a5de1b6b761c41a`, verified against `fixtures/reference_qwen3_8_125b_a6b_mtp_8bit.sha256` |
| Reference | mlx-lm at `c961f8399a23e962837495ec403ea7c5e0e4e848` |
| Prompt | the 1024-token public correctness prompt, wrapped as section 4 says |

Do not run this on a branch. Run it on merged code, so the numbers describe
what the track ships.

## 2. The lock

The box runs one job at a time. Take the lock before you load a model, and
release it when you stop, including when a step fails.

1. Take the lock.
2. Unload the resident model.
3. Run the steps below.
4. Reload the resident model.
5. Release the lock.

Record the start time and the end time. If a step fails, stop, release the
lock, and report. Do not retry a failed measurement with a changed setting in
the same session.

## 3. Step A: the width survey

Run `Qwen4ExpWidthDivergencePinnedWeightsTests`, both arms, unchanged:

```
MLXFAST_RUN_MLX_RUNTIME_TESTS=1 \
MLXFAST_WIDTH_DIVERGENCE_WEIGHTS_PATH=<transformed weights tree> \
MLXFAST_WIDTH_DIVERGENCE_OUT=<result.json> \
swift test --no-parallel --filter Qwen4ExpWidthDivergencePinnedWeightsTests
```

The "indexer off" arm seeds 1024-token prompts. The "indexer on" arm seeds
prompts past the 2048-token budget. Both arms run in one invocation.

Keep `result.json` and the console log. They are the evidence.

The report records the norm convention the run used: `norm_convention`,
`norm_weight_offset_applied` (0 on this checkpoint, because it bakes the
offset), `norm_convention_verified`, and `norm_tensors_classified`. Check those
four fields FIRST. The survey loads through the same entry point the worker
uses, so a run that reached the end has already had every non-gated norm tensor
classified and agreed with the pin; if the fields are missing, the result came
from an older engine and is not comparable.

### Pass criteria for step A

THE BASIS IS RULED (David, 2026-08-28). These four DECIDE:

| Check | Criterion |
|---|---|
| Argmax | Row 0's argmax agrees across all widths at every position |
| Same-width bit-identity | Two forwards of the same width from the same state agree to zero |
| Greedy coherence | See step B: the continuation is readable English |
| Hidden-state cosine | See step B: at or above 0.99 |

Absolute logit deltas are DIAGNOSTICS. Record them -- the worst per width and
arm, and the worst row-0 value -- and do not treat them as a gate. Different
widths dispatch different kernels by design, so a nonzero delta is expected;
what would be a defect is an argmax moving with it.

**REPORT ROWS BEYOND ROW 0 SEPARATELY.** Row 0 is what a serial decode commits,
but a wide verify reads the rows after it too, so a flip there is not
cosmetic. The 2026-08-28 run found five, all at near-tie margins, out of 1,728
rows compared. State the count, the margins and the widths.

A row-0 argmax flip means a keep mask reaches forward or another
width-dependent mask exists. Report the failing rows and stop.

**NEVER WIDEN A DIAGNOSTIC SILENTLY.** If a delta grows, that is a finding to
report and explain. A tolerance may change only as a stated, dated decision
with the measurement that motivated it.

## 4. Step B: the reference parity comparison

### 4.1 The prompt shape

The earlier capture fed 1054 raw prose tokens with no wrapping. This model is
an instruction-tuned chat model, so a bare continuation is a weak test. Use the
chat template.

Build the prompt with the tokenizer's own template and
`add_special_tokens=False`, because the template already supplies every special
token:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<PROMPT><|im_end|>
<|im_start|>assistant
<think>

</think>

```

The trailing block is the template's generation prompt with thinking disabled.
The tokenizer sets `add_bos_token` to false and defines no beginning-of-sequence
token, so nothing is prepended.

Record the token ids. BOTH sides must consume the identical id list. Compare
the two lists before you compare any logits; if they differ, stop.

### 4.2 Match the norm convention on the reference side

The ENGINE side needs nothing done: it loads through `loadPinnedTarget`, which
applies the pinned convention and classifies every non-gated norm tensor before
handing the model over. The reference side is what has to be corrected.

The reference computes `mx.fast.rms_norm(x, 1.0 + weight)` for its non-gated
norms. This checkpoint already stores `1 + w`. Left alone, the reference
computes `2 + w` and its numbers are meaningless.

Correct it in the parity script, in ONE of these two ways. Do not do both.

1. **Subtract one at load.** In the script's `sanitize` wrapper, subtract 1.0
   from every NON-GATED norm weight before the reference sees it.
2. **Patch the module.** Replace the reference's non-gated `RMSNorm` forward
   with `mx.fast.rms_norm(x, self.weight)`.

Option 2 is preferred: it is exact, while subtracting one and adding it back
rounds in bfloat16.

The NON-GATED families, which are the ones to correct:

```
model.layers.*.attn_hyper_connection.hc_norm
model.layers.*.mlp_hyper_connection.hc_norm
model.hyper_connection_mixer.hc_norm
model.layers.*.self_attn.q_norm
model.layers.*.self_attn.k_norm
model.layers.*.self_attn.indexer.q_layernorm
model.layers.*.self_attn.indexer.k_layernorm
model.layers.*.ple.norm_conv
model.layers.*.ple.norm_key
model.layers.*.ple.norm_query
mtp.*  (every norm above, under the head's prefix)
mtp.pre_fc_norm_embedding
mtp.pre_fc_norm_hidden
```

The GATED family is `model.layers.*.linear_attn.norm`. LEAVE IT ALONE. It is
conventional in this checkpoint, and correcting it would introduce the error
this step exists to remove.

State in the report which of the two options you used.

### 4.3 What to capture

From the identical token ids, on both sides:

1. a single-token forward from the same cache state, logits for one row;
2. a full prefill, logits for the last four rows;
3. a 32-token greedy continuation, decoded to text;
4. the post-prefill hidden state before the output projection.

### Pass criteria for step B

The same ruled basis:

| Check | Criterion |
|---|---|
| Engine same-width bit-identity | Row k of a prefill equals row k of a shorter prefill ending at k, to ZERO |
| Argmax, engine against reference | The same token on every compared row |
| Hidden-state agreement | Cosine similarity at or above 0.99 |
| Coherence | The 32-token greedy continuation is readable English on BOTH sides |

Absolute logit deltas between the two implementations are DIAGNOSTICS. Record
the maximum and the median per row. They will not be small: on 2026-08-28 they
ran to 2.12 with the argmax agreeing everywhere and the hidden state at cosine
0.9972.

The coherence check is the one that matters most. Two implementations agreeing
on incoherent output would mean a shared defect remains -- which is exactly
what the first capture found, before the three defects behind it were fixed.

**THE REFERENCE IS A CROSS-CHECK, NOT AN ORACLE.** mlx-lm patched to compute
`y * w` for the non-gated norms reads this checkpoint's convention and is a
second implementation over the same weights, which is what found the norm
defect. It shares this port's upstream lineage, so a defect inherited from that
lineage would appear in both. **The oracle is the CUDA runner on test-spark**:
independent implementation, independent hardware. When the two disagree, that
is the comparison that settles it.

## 5. Reporting

Report, in this order:

1. the engine commit, the weights revision, and the reference revision;
2. which norm-convention option you used on the reference side;
3. the token ids both sides consumed, and that they matched;
4. the step A table and the step B table, filled in;
5. the two 32-token continuations, as text;
6. the lock window, and any step that failed.

Attach `result.json` from step A and the captured logits from step B.

## 6. Step C: the coherence check on its own

If step B's parity criteria pass but the continuations are not readable, stop
and report. Two implementations agreeing on incoherent output means a defect
they SHARE is still present, and the parity comparison cannot see it. Say so
plainly rather than reporting step B as a pass.

If step B's continuations ARE readable on both sides, that is the first
evidence this checkpoint has ever generated correctly through this engine.
Record the exact text.

## 7. What this plan does not do

It does not change the verify width, arm the track, or touch any participant
surface. It measures.
