# Moshirag Runtime Port and Delayed-Evidence Training

## Decision

Voryn retains PersonaPlex as the live duplex audio model. Moshirag is the reference
implementation for **causal streaming evidence conditioning**, not a replacement
voice model. Its useful contract is a small `T x D` sequence that changes the
generator input once per speech-token step. The locally inspected checkpoint uses
`D=4096`, ARC4-derived conditioning, a learned `3072 -> 2048 -> 4096` bridge, and
a 12.5 Hz evidence rate. The local PersonaPlex source lacks that branch, so the
fork carries a reviewed, fail-closed compatibility patch.

This document supersedes any implication that `evidence.update` metadata alone
creates semantic control. It does not. The evidence adapter, patched forward pass,
native delayed-code training, source identity, and runtime cancellation semantics
are all required before promotion.

## Runtime Contract

The semantic service submits a typed control revision first. A late source may then
submit only bounded evidence:

```json
{
  "type": "evidence.update",
  "protocolVersion": 2,
  "callId": "CA...",
  "revision": 43,
  "supportsControlRevision": 42,
  "contextHash": "sha256:...",
  "availability": "ready",
  "provenance": {"source": "shipment_tool", "record": "replacement-queued"},
  "allowedClaims": ["replacement_queued", "carrier_scan_pending"]
}
```

The envelope cannot contain canonical reply text, prompts, target labels, or a
verbatim response. `revision=43` cancels output generated from revision 42. A
subsequent typed control update authorizes the next answer; the model snapshots the
accepted control prefix and evidence stream together at that next turn boundary.
No revision can rewrite queued or emitted audio.

## Native Model Port

`personaplex-setup/moshirag_streaming_sum.patch` changes only these interfaces:

1. `LMModel.forward_codes` and `forward_embeddings` accept an optional additive
   `streaming_sum` with exactly the same `[batch, frames, hidden]` shape as the
   input embeddings.
2. `LMGen` owns a CUDA `condition_streaming_sum` row and a separate queued
   `pending_streaming_sums` sequence for each batch slot.
3. `update_streaming_sum_tensors` accepts immutable `[T, hidden]` streams.
4. The generator consumes one row immediately before each live `step`; exhaustion
   explicitly zeroes the condition rather than carrying stale evidence forward.

The patch is applied only to an isolated source copy:

```bash
CUDA_VISIBLE_DEVICES=0 \
  /srv/personaplex_workspace/robit-man-personaplex/personaplex-setup/apply_moshirag_streaming_sum_patch.sh \
  /srv/personaplex_workspace/upstream-personaplex/moshi
```

Do not patch a source tree currently serving calls. After application, capture a
fresh source fingerprint with `training.native_source.moshi_source_fingerprint` and
bind it to the evidence-adapter checkpoint. Any source/model/checkpoint mismatch
must keep runtime evidence in `evidence_deferred`.

## Training Contract

### Stage A: semantic prefix

Freeze PersonaPlex. Train `SemanticPrefixAdapter` only using the native delayed
duplex codes and agent-only loss. Prefix examples include state, typed control, and
turn timing, but never the intended reply. Promotion requires held-out semantic
adherence gains without a regression in no-control behavior, voice quality, first
audio latency, or interruption behavior.

### Stage B: delayed evidence stream

Freeze both PersonaPlex and the accepted Stage-A prefix. Train only
`EvidenceStreamAdapter`, a compact transformer/cross-attention encoder with a
conservative learned gate. It maps typed evidence tokens to 16 hidden rows by
default. `EvidenceStreamTrainer` applies those rows after the virtual control
prefix at the response boundary through `forward_with_semantic_prefix_and_evidence`.
That function fails closed until the native source has `streaming_sum` support.

Every Stage-B record joins one `ControlTrainingFrame` and one
`EvidenceTrainingFrame`. The evidence frame names the prior supporting control
revision and pre-evidence state hash; its target plan is a strictly later revision
whose context hash equals `postEvidenceStateHash`. Each evidence source must arrive
before the target agent audio begins. The target transcript and target code tokens
are labels only and are prohibited in the evidence envelope.

### Counterfactual requirement

Each `counterfactual.groupId` requires at least two branches sharing caller duplex
context and base plan but differing in one material late fact. Examples include:

- `refund_pending` versus `refund_issued`
- `replacement_queued` versus `carrier_scan_pending`
- `policy_allows_escalation` versus `policy_requires_supervisor_review`
- `identity_verified` versus `identity_not_verified`
- `caller_still_present` versus `caller_barged_in`

The expected agent target must materially differ; a data auditor and semantic judge
must reject branches that preserve the same reply meaning. This blocks the adapter
from learning topic correlation instead of evidence following.

## Corpus V7 Requirements

V7 is the first eligible evidence-training corpus. It requires all of the following:

- GPU-only generative synthesis on CUDA devices 0, 1, and 2; no CPU inference and
  no use of GPU 3.
- Diverse multi-turn, two-speaker phone calls across cooperative, skeptical,
  resistant, safety-sensitive, clarification, repair, interruption, handoff, and
  non-call conversational trajectories.
- Natural openings and endings, with model-selected terminal action/tool labels;
  no deterministic signoff heuristic or repeated stock introductions.
- Actual barge-in overlap, a captured cancellation time, and a recovery turn for
  each interruption trajectory.
- Chatterbox Turbo as the corpus renderer. Whisper ASR, word timing, WER, loudness,
  clipping, and telephony codec checks are admission gates; failures quarantine the
  record rather than being silently repaired.
- Provenance-approved clone references only. Miso may be separately auditioned for
  expressive-renderer research, but it is not a duplex control-model renderer.
- Evidence records for ready, failed, and expired availability. Only `ready`
  records are candidates for evidence-conditioned loss; failed/expired examples
  train safe waiting and no-invention behavior in the control stage.

## Evaluation and Promotion

Measure, per held-out counterfactual pair:

1. Correct fact/tool-result incorporation and forbidden-claim avoidance.
2. Difference sensitivity: changing only the evidence must change the relevant
   response decision while unrelated wording remains natural.
3. Control abstention when evidence is expired, failed, missing, or stale.
4. Barge-in cancellation, generation-ID invalidation, and fresh-revision recovery.
5. First audio latency, continuous-audio gap distribution, codec integrity, ASR
   intelligibility, and speaker/voice preservation.
6. A no-control regression suite proving that control dropout preserves native
   PersonaPlex conversational behavior.

Promotion needs predeclared threshold improvement on semantic adherence and no
regression beyond the threshold on latency, agent speech quality, or barge-in. A
strict wording request still routes to the validated strict renderer: learned
speech-to-speech conditioning is semantic guidance, not a deterministic text
guarantee.
