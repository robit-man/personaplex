# MoshiRAG Upstream Adoption for PersonaPlex Semantic Control

Status: accepted architecture direction

Date: 2026-07-19

## 1. Decision

Use the official MoshiRAG implementation as the architectural baseline for
text-to-speech-model conditioning. Do not replace PersonaPlex with the released
Moshika-RAG model, and do not continue treating the local experimental
`streaming_sum` adapter as an independently designed final mechanism.

The target system is:

1. PersonaPlex remains the full-duplex speech and voice/persona model.
2. MoshiRAG's ARC-4 reference encoder, projection, streamed additive injection,
   and production scheduler are ported or reused where architecture-compatible.
3. A second fixed-width turn-boundary control prefix carries immediate goals,
   policy constraints, stance, and action authorization before the first output
   token of the turn.
4. A typed control envelope supplies revision, acknowledgement, snapshot, stale
   rejection, and cancellation semantics outside the model.
5. New PersonaPlex weights are trained to causally follow the combined control
   prefix and reference stream while preserving full-duplex behavior and voice.

This is an adoption-and-extension strategy, not a wholesale model replacement.

## 2. Pinned upstream evidence

The implementation comparison is pinned to:

- MoshiRAG source commit:
  `8c6dfc101b7871baa428424bcdc583b74fb561d9`
- PyTorch model revision:
  `kyutai/moshika-rag-pytorch-bf16@7135a6e3c46abb66c2cd95cb04cbfcbe8376f83d`
- Candle model revision:
  `kyutai/moshika-rag-candle-bf16@26c8294761455c0aaafbadc6772c890fdc11f68f`
- Paper: MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech
  Language Models, arXiv:2604.12928v3.

Primary sources:

- https://github.com/kyutai-labs/moshi-rag
- https://huggingface.co/collections/kyutai/moshirag-release
- https://huggingface.co/kyutai/moshika-rag-pytorch-bf16
- https://huggingface.co/kyutai/moshika-rag-candle-bf16
- https://arxiv.org/abs/2604.12928

The pinned source checkout lives at:

`/srv/voxrn_cache/personaplex/source/moshi-rag-official`

## 3. What the release proves

MoshiRAG demonstrates that a compact, full-duplex speech LM can accept textual
information as a learned model input while continuing a live conversation. Its
released configuration uses:

- A 4096-dimensional, 32-layer temporal model operating at 12.5 Hz.
- An ARC encoder compressed by a factor of four.
- A learned projection from ARC's 3072-dimensional output through a configured
  bridge to the model's 4096-dimensional input space.
- `streaming_sum` fusion, consuming one reference embedding per model frame.
- A separate reference encoder service that sends an encoded tensor to the live
  LM process.
- A learned retrieval trigger token and asynchronous retrieval scheduling.
- Reference dropout and simulated retrieval timing during training.

The official runtime queues a `[T, dim]` tensor per active slot, consumes one row
for each executing frame, and adds that row directly to the normal text and
audio input embeddings before the temporal transformer. This is a real trained
model input, not prompt text written by a sidecar LLM.

The paper's ablations are important:

- Insertive reference injection was more accurate than additive injection.
- Additive injection was selected because insertive injection expands the
  sequence and harms long-conversation efficiency.
- ARC-4 additive injection substantially outperformed T5 additive injection and
  ARC-8 additive injection in the reported evaluation.
- Timing matters: a compressed reference must reach the model before the key
  response content is generated.

These findings invalidate an unbounded, arbitrary-amplitude custom residual and
support a compressed, timed, architecture-native conditioning stream.

## 4. What the release does not provide

The release does not satisfy the VoxRN target by itself:

- The released model is Moshika-RAG, not PersonaPlex.
- Its card describes one female synthetic voice rather than PersonaPlex's
  selectable voice/persona conditioning.
- Its semantic input is an untyped retrieved reference string.
- It has no `control_revision`, acknowledged revision, immutable generation
  snapshot, stale-update rejection, generation ID, or call-level cancellation
  protocol.
- It can inject retrieval results during an utterance. Policy-sensitive VoxRN
  control must instead be snapshotted at a safe turn boundary.
- It is optimized primarily for factual retrieval, not goals, constraints,
  caller posture, tool authorization, resistance, correction, handoff, or
  model-driven call completion.
- It does not make exact wording deterministic.
- The public repository contains inference and evaluation code, but not the
  training pipeline used for the released checkpoint.

Consequently, using the checkpoint directly would lose the PersonaPlex product
requirements and still leave the control protocol unimplemented.

## 5. Correct model architecture

### 5.1 Runtime control envelope

Keep operational metadata out of the semantic embedding payload:

```json
{
  "schema": "voxrn.personaplex.control.v1",
  "call_id": "CA...",
  "revision": 42,
  "parent_revision": 41,
  "effective_from": "next_agent_turn",
  "generation_id": "generation-uuid",
  "payload_sha256": "...",
  "expires_at": "...",
  "payload": {
    "intent": "resolve_delivery_issue",
    "known_facts": [
      "replacement shipped July 14",
      "tracking is awaiting carrier scan"
    ],
    "caller_posture": "skeptical",
    "next_goal": "acknowledge delay and offer escalation options",
    "constraints": [
      "do not invent a delivery date",
      "do not repeat the greeting"
    ],
    "tool_results": [
      "shipment status is replacement queued"
    ],
    "style": {
      "warmth": 0.7,
      "assertiveness": 0.35,
      "brevity": 0.55
    }
  }
}
```

`call_id`, revisions, hashes, expiration, and generation IDs are runtime control
metadata. They must not consume model conditioning capacity. Only canonicalized
semantic payload fields are encoded for the model.

### 5.2 Two-path silent conditioning

Use two complementary conditioning paths.

#### Immediate control prefix

Encode a compact canonical summary of:

- intent
- next goal
- required action
- prohibited actions or claims
- caller posture
- style controls
- end-call or handoff authorization

Project it to a small fixed number of virtual embeddings and apply it at the
agent-turn boundary before output generation. This ensures that urgent control
affects the first generated token rather than arriving after several streamed
reference frames.

The prefix must be bounded, zero-safe, and dropped on a controlled fraction of
training examples. It must not be recomputed for every 80 ms frame.

#### Detailed reference stream

Encode longer facts, tool results, explanations, and retrieved context with the
official ARC-4-compatible reference encoder and projection. Consume the
resulting embeddings using MoshiRAG's one-row-per-frame `streaming_sum`
scheduler.

This preserves long-context efficiency and provides enough bandwidth for
detailed factual grounding.

#### Why both are required

MoshiRAG's own ablation shows that insertive information is easier for the model
to use, while additive streaming is more efficient. A short bounded prefix plus
a compressed additive stream captures that tradeoff:

- Prefix: immediate behavioral and policy steering.
- Stream: detailed mutable facts and tool/reference content.

This is preferable to a large arbitrary per-layer K/V adapter for the first
implementation because it stays close to released, demonstrated Moshi paths and
has explicit timing behavior.

### 5.3 Turn snapshot and cancellation

At an agent-turn boundary:

1. Select the highest valid acknowledged control revision.
2. Canonicalize and hash its semantic payload.
3. Resolve or fetch both conditioning tensors once.
4. Bind them to an immutable generation snapshot.
5. Start generation with that snapshot and generation ID.

If a caller barges in:

1. Invalidate the active generation ID.
2. Stop forwarding generated audio to Twilio immediately.
3. Clear queued reference rows for that generation.
4. Update ASR and state using the interruption.
5. Accept a newer control revision.
6. Start the next agent turn from a fresh snapshot.

A late conditioner response may populate a cache, but it may not mutate an
already snapshotted policy-sensitive generation.

## 6. Training reality and scale

The official result was not produced with a small adapter experiment. The paper
reports approximately:

- 1,901,376 synthetic conversations.
- 47,770 hours of training audio.
- 100,000 training updates.
- All Moshi parameters trainable except the reference encoder.

Our current 393 counterfactual pairs are useful for:

- proving that control input changes model likelihood in the right direction;
- catching target leakage;
- testing null and stale controls;
- validating exact branch contrast;
- running deterministic regression tests.

They are not sufficient evidence for broad 95 percent generated-audio
reliability. A 1,000-conversation dataset is still a pilot. The proposed
50-by-20-by-10 cascade yields 10,000 leaf conversations, not 1,000, and should
be treated as the minimum first curriculum rather than the final scale target.

The training program must therefore use progressive evidence gates.

### Stage 0: architecture and transfer probe

- Compare raw PersonaPlex and Moshika-RAG tensor names and shapes.
- Identify directly reusable ARC conditioner, bridge, trigger, and fuser weights.
- Measure the effect of adding the upstream condition path at exact zero.
- Require bitwise or tolerance-bounded parity for null control.

### Stage 1: conditioner-only causal proof

- Freeze PersonaPlex.
- Train only the immediate prefix projector, ARC bridge or compatible adapter,
  gates, and newly introduced embeddings.
- Use the exact counterfactual pairs and agent-only text/audio loss.
- Require matched control to beat counterfactual, stale, and null controls.

This stage proves wiring. It is not a production checkpoint.

### Stage 2: selective PersonaPlex adaptation

- Unfreeze selected upper temporal layers or use LoRA on those layers.
- Retain replay examples without control and with sparse control.
- Train on the full accepted 10,000-leaf cascade plus counterfactual variants.
- Include timing perturbation, reference dropout, stale revisions, interruption
  cancellation, correction, and tool-result reversals.

### Stage 3: broad supervised adaptation

- Expand beyond 10,000 conversations if held-out category reliability or
  generated-audio reliability plateaus.
- Consider whole-model adaptation only with enough replay data and measured
  voice/full-duplex preservation.
- Keep the reference encoder frozen initially, matching MoshiRAG.
- Do not promote a run solely because teacher-forced loss converges.

### Stage 4: generated-audio qualification

- Generate complete duplex audio from held-out calls.
- Decode the actual model output, not target or teacher-forced audio.
- Run CUDA ASR over every evaluated agent response.
- Judge semantic requirements from ASR text and tool/end-call events.
- Measure voice similarity, codec integrity, TTFAT, interruption cutoff, overlap,
  recovery, and end-call behavior.

## 7. Dataset contract changes

Every controlled agent turn must contain:

- Prior duplex audio context.
- The typed semantic payload available before the response.
- Envelope revision and timing metadata.
- The immutable revision selected for the target turn.
- Agent transcript and native audio/code tokens as labels only.
- Control prefix availability time.
- Reference-stream availability time and row schedule.
- Barge-in onset, cancellation cutoff, and invalidated generation ID where
  applicable.
- Required concepts, prohibited concepts, expected action, and terminal-state
  labels for evaluation only.

The target response must never be copied into control input.

At least one counterfactual sibling should change a material state field while
keeping prior caller context close or identical. Examples include:

- refund pending versus refund issued;
- identity verified versus identity unverified;
- action authorized versus policy forbidden;
- caller cooperative versus caller newly resistant;
- tool success versus tool timeout;
- continue conversation versus handoff;
- continue conversation versus end-call authorized;
- old fact versus corrected fact after interruption.

The changed semantic state must require a materially different response. Merely
paraphrasing the same target is not a valid counterfactual.

## 8. Evaluation matrix and promotion gate

Every candidate must be compared against:

1. Raw PersonaPlex with no semantic conditioner.
2. Official Moshika-RAG on retrieval/factual tasks.
3. PersonaPlex with the immediate prefix only.
4. PersonaPlex with the detailed ARC stream only.
5. PersonaPlex with both paths.

Report, by held-out category and overall:

- required-fact recall;
- prohibited-fact violation rate;
- correct next-action rate;
- counterfactual branch discrimination;
- stale-revision rejection;
- null-control behavioral parity;
- interruption acknowledgement and recovery;
- cancellation audio tail duration;
- model-driven end-call correctness;
- generated-audio ASR intelligibility;
- speaker/voice preservation;
- TTFAT and meaningful-content delay;
- duplex benchmark regression.

The 95 percent claim applies only when all of the following are true on held-out
generated audio:

- The aggregate semantic success rate is at least 95 percent.
- Every critical policy category is at least 95 percent.
- Confidence intervals are reported with enough trials to be meaningful.
- There is no target leakage or template overlap across splits.
- Stale and counterfactual controls materially reduce the matching score.
- Voice, latency, and full-duplex regression budgets pass.
- Exact-language cases are routed to the strict renderer and are excluded from
  claims of deterministic PersonaPlex wording.

## 9. Implementation map

Existing local components remain useful:

- `training/native_training.py`: exact changed-token counterfactual masks.
- `training/causal_trainer.py`: matched/null/stale/counterfactual objectives.
- `tools/canonicalize_control_v4_pairs.py`: raw-model-bound corpus validation.
- `tools/train_semantic_control_v4.py`: distributed proof trainer and metrics.
- `tools/launch_semantic_control_v4.py`: CUDA lane launch and resource controls.
- `training/control_stream.py`: provisional mechanism retained only as a
  comparison baseline until the upstream path replaces it.

Required next implementation units:

- `training/moshirag_conditioning.py`: pinned ARC-4 and projection compatibility.
- `training/turn_control_prefix.py`: bounded fixed-width immediate prefix.
- `training/control_snapshot.py`: immutable revision-to-generation binding.
- `runtime/control_protocol.py`: typed envelope validation and stale rejection.
- `runtime/control_scheduler.py`: safe boundary application and pending tensor
  lifecycle.
- `runtime/generation_cancel.py`: generation invalidation and queue clearing.
- `tools/import_moshirag_release.py`: deterministic source/checkpoint import with
  revisions and hashes.
- `tools/compare_moshi_personaplex_weights.py`: key/shape compatibility report.
- `tools/train_semantic_control_v5.py`: two-path staged training.
- `tools/evaluate_semantic_control_v5.py`: generated-audio qualification.

## 10. Claims boundary

The work remains useful and potentially novel, but the claim must be precise.

Not novel:

- Streaming additive text embeddings into a full-duplex Moshi model.
- An asynchronous external text back end.
- ARC-4 reference compression.

Distinct target contribution:

- Porting learned MoshiRAG-grade conditioning into PersonaPlex while retaining
  voice and role control.
- Combining immediate turn-boundary control with a detailed streamed reference.
- A typed, revisioned, acknowledged, cancellable call-state protocol.
- Counterfactual training for goals, policy, tools, posture, repair, handoff, and
  call completion rather than factual RAG alone.
- End-to-end Twilio barge-in semantics and strict-renderer routing.
- Generated-audio evidence for semantic reliability rather than prompt-level or
  teacher-forced claims.

The released MoshiRAG model is therefore a strong upstream foundation. It does
not eliminate the need for PersonaPlex-specific weight training, corpus
construction, runtime integration, or qualification.
