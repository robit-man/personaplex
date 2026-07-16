# PersonaPlex: 1,000 Synthetic Calls and Learned Semantic Control

## Objective

Produce a certified corpus of 1,000 two-party synthetic phone conversations and use it to train and evaluate a PersonaPlex control adapter. The adapter must make a typed, mutable semantic control frame a learned causal input to next-turn speech-token generation.

This is not a prompt-wrapper project. PersonaPlex remains the fast full-duplex audio model. A deeper call-state/LLM/tool layer decides what should happen; the learned control path steers PersonaPlex at safe turn boundaries. Exact wording is handled by a separate strict renderer when it is required.

## Non-negotiable invariants

- The control frame is available before an agent target turn. Target text and target audio are labels only and must never be copied into the control input.
- Every branch is replayable from an immutable, versioned snapshot. Shared counterfactual prefix turns have byte-identical audio, timing, transcript, ASR result, and control state.
- Only agent target audio/token positions contribute speech loss. Caller audio is duplex context, never an agent imitation target.
- A control revision is acknowledged before it governs a response. A newer revision cannot silently mutate an already-running generation.
- Caller barge-in cancels queued outbound audio immediately. The next eligible response uses the newest acknowledged revision.
- Chatterbox Turbo is the current corpus renderer. Each admitted utterance is passed through ASR, alignment, codec, and semantic evaluation. Audio that fails does not enter training.
- Quality and semantic acceptance use actual ASR, timing measurements, typed-state checks, and independent model inference. They are not rescued by pattern matching or fabricated labels.
- No repeated scripted greetings, boilerplate placeholders, deterministic signoff templates, or presumed company names. Terminal behavior is model-selected and expressed as an explicit `end_call` action only when the semantic state supports it.
- Only CUDA devices `0`, `1`, and `2` are used by the corpus and training lanes. GPU `3` and unrelated services remain untouched.

## Deliverables

| ID | Deliverable | Definition of done |
| --- | --- | --- |
| D1 | 1,000-call raw corpus | 500 counterfactual groups, two branches per group, durable manifests and replay evidence. |
| D2 | Certified corpus | Every training-eligible call has passed audio/ASR/timing/semantic/provenance gates and has a certificate. |
| D3 | Native PersonaPlex training set | Delayed duplex code streams, per-turn control frames, target masks, interruption metadata, and train/validation/test splits. |
| D4 | Learned control adapter | Frozen-base prefix/K-V adapter with field-aware control encoder, revision-aware inference API, and checkpoint artifacts. |
| D5 | Evaluation suite | Semantic, counterfactual, interruption, latency, voice, codec, and safety results per checkpoint. |
| D6 | Runtime control plane | Typed update protocol, acknowledged revision cache, cancellation semantics, and Twilio streaming integration. |
| D7 | End-to-end evidence | Controlled mock-call and Twilio-compatible tests proving the newest valid state governs the next spoken turn. |

## 1. Corpus contract

### 1.1 Unit of production

One counterfactual group creates two complete conversations:

1. A shared prefix containing the same caller history, duplex timing, facts, and previously acknowledged control revisions.
2. A branch-point state change: tool result, policy constraint, caller posture, new fact, interruption, or goal change.
3. A `available` branch and a `constrained` branch whose agent response must materially differ for the stated semantic reason.

The target branch audio is newly generated. Shared prefix audio is replay context only and is explicitly marked ineligible for target loss in secondary branches. This prevents duplicated labels from overweighting greetings or early context.

### 1.2 Required turn record

Each turn stores the following immutable data:

```json
{
  "conversation_id": "v7cf-...",
  "counterfactual_group_id": "...",
  "branch": "available|constrained",
  "turn_index": 7,
  "speaker": "caller|agent",
  "duplex_audio": {
    "path": "...",
    "codec": "wav_pcm_then_phone_codec",
    "sha256": "...",
    "started_ms": 0,
    "ended_ms": 0,
    "overlap_ms": 0,
    "cutoff_ms": null
  },
  "target": {
    "training_eligible": true,
    "transcript_label": "...",
    "speech_codes_path": null
  },
  "asr": {
    "transcript": "...",
    "word_timing": [],
    "wer": 0.0,
    "confidence": 0.0
  },
  "control": {
    "schema_version": "1.0",
    "revision": 42,
    "effective_from": "next_agent_turn",
    "acknowledged": true,
    "frame": {}
  },
  "generation": {
    "generation_id": "...",
    "cancelled": false,
    "control_snapshot_hash": "..."
  }
}
```

The corpus writer hashes the ordered control snapshot and replay payload. The trainer rejects a target turn if its pre-turn control revision is unacknowledged, stale, absent, or inconsistent with the replay chain.

### 1.3 Typed control frame

The frame is compact, declarative, and versioned. It describes state, not the desired sentence.

```json
{
  "schema_version": "1.0",
  "call_id": "CA...",
  "revision": 42,
  "effective_from": "next_agent_turn",
  "intent": "resolve_delivery_issue",
  "known_facts": [
    {"key": "replacement_status", "value": "queued", "source": "shipment_tool"},
    {"key": "carrier_scan", "value": "awaiting", "source": "shipment_tool"}
  ],
  "commitments": [],
  "uncertainty": ["carrier pickup time is unknown"],
  "caller_posture": "skeptical",
  "next_goal": "acknowledge delay and offer supported escalation paths",
  "constraints": ["do_not_invent_delivery_date", "do_not_repeat_greeting"],
  "tool_result_refs": ["shipment:replacement-queued"],
  "style": {"warmth": 0.7, "assertiveness": 0.35, "brevity": 0.55},
  "terminal": {"eligible": false, "reason": null}
}
```

Allowed field values come from typed taxonomies and source-backed facts. The planner may choose novel content, but cannot inject target-response text as a hidden field.

## 2. Coverage plan for 1,000 calls

### 2.1 Quotas

The 500 groups are allocated before generation and tracked by metadata. A group is regenerated until it passes certification or a documented source/tool failure makes it impossible.

| Dimension | Required coverage |
| --- | --- |
| Conversation length | 3-5, 6-10, 11-18, and 19-30 turn bands; include both short resolutions and longer repairs. |
| Topics | Service/support, deliveries, billing, account access, travel, healthcare administration without diagnosis, education, polling, community issues, utilities, retail, insurance administration, workplace coordination, casual/social, technical troubleshooting, surveys, handoff/escalation, and mixed-topic calls. |
| Caller posture | Cooperative, curious, hurried, skeptical, guarded, frustrated, resistant, conditional, confused, silent/hesitant, and recovering after misunderstanding. |
| Agent behavior | Clarify, verify, explain limits, propose options, negotiate next step, correct itself, apologize, safely refuse, hand off, summarize, and close naturally. |
| Control perturbation | Tool-result change, policy change, fact correction, commitment change, caller-posture change, interruption invalidation, and terminal-state change. |
| Duplex timing | Natural gaps, interruptions, partial overlaps, barge-in cutoffs, response repair, backchannels, pauses, and delayed tool-result turns. |
| Voices | Provenance-approved reference voices with balanced use across role, accent, rate, timbre, expressiveness, and phone-channel conditions. |

No topic, voice pair, opening pattern, closing pattern, or control perturbation may dominate the admitted set. Coverage is measured from structured metadata and reviewed before each training export.

### 2.2 Conversation realism rules

- The caller opens on an actual topic or concrete circumstance, not a generic contact ritual.
- Planner-generated entities are resolved to natural names, dates, amounts, and organizations before rendering. Placeholder language is rejected.
- Call completion is a semantic state, not a phrase detector. The planner may decide `end_call` only after it has satisfied or safely terminated the active goal; the simulator ends from that action.
- Interrupted agent audio includes an actual truncation point. The recovery turn must react to the caller’s new words and newest control revision.
- Counterfactual branches use identical pre-pivot context and must diverge in a way supported by their distinct tool/policy/fact state.

## 3. Production pipeline

### Phase A: Generate paired conversations

**Inputs:** allocated group specification, approved voice references, scenario seed, initial state tree, renderer configuration.

**Process:**

1. Generate the primary branch with its rolling caller ASR, call state tree, typed control revisions, and model-driven terminal decisions.
2. Capture the immutable replay snapshot at the pivot.
3. Replay the prefix exactly for the alternate branch, issue the different control revision, and generate only post-pivot speech.
4. Render each utterance with Chatterbox Turbo, preserve raw timing, and construct duplex audio with genuine overlap/cutoff metadata.
5. Run Whisper ASR and word alignment on every rendered target and caller turn.
6. Run a pre-admission semantic audit over the actual state and a post-ASR semantic audit over what was actually heard.

**Gate A:** both branches have complete manifests; shared prefix equality is exact; all audio renders; no unacknowledged revision; no target-audio duplication from replay context.

### Phase B: Independent certification

**Inputs:** complete paired manifest from Phase A.

**Checks:**

- ASR availability, WER, confidence, word timing, silence, clipping, sample rate, phone-codec round trip, and channel integrity.
- Independent semantic verification of actual ASR text against facts, policy, tool results, caller posture, and branch-specific goal.
- Counterfactual verification that the changed state caused a material difference, not merely a stylistic paraphrase.
- Provenance check for all reference voices and renderer configuration.
- Exact replay hash check and target-loss eligibility check.

**Gate B:** a group is certified only if both conversations pass. Partial pairs, weak audio, unsupported claims, stale-control turns, or incoherent branches are quarantined and scheduled for fresh generation.

### Phase C: Corpus assembly

**Inputs:** certificates only, never raw manifests.

**Process:**

1. Build train/validation/test split at counterfactual-group level so replay prefixes and branch siblings cannot leak across splits.
2. Convert eligible agent audio to PersonaPlex-native delayed code streams.
3. Attach prior duplex code context, compact control frame, revision ID, branch metadata, interruption/cutoff mask, and agent-only loss mask.
4. Run a leakage audit: an independent model receives the frame but not target text and must not be able to reconstruct a target sentence from hidden instruction content.
5. Publish immutable dataset index, coverage report, certificates, manifests, and checksums.

**Gate C:** 1,000 certified calls; every coverage quota met; counterfactual groups are split-safe; no invalid target mask or control-frame leak; held-out test set is frozen.

## 4. Learned control adapter

### 4.1 Architecture

```text
typed control frame
  -> field-aware control tokenizer/encoder
  -> K learned virtual tokens and selected-layer K/V prefixes
  -> learned per-layer gates
  -> frozen PersonaPlex speech transformer + duplex audio context
  -> next agent speech-code logits
```

The frame encoder represents fields separately: intent, facts/source, commitments, uncertainty, posture, goal, constraints, tool references, style, and terminal state. A field-presence mask distinguishes an absent fact from a negative fact. The adapter is cached once per acknowledged control revision on GPU.

The base PersonaPlex model remains frozen for the first stage. Training starts with the control encoder, projection, K/V prefix parameters, layer gates, and a small revision embedding only. Control dropout keeps the audio model stable when a call has sparse guidance.

### 4.2 Losses

- **Primary:** agent-only next speech-code cross-entropy over the native delayed duplex sequence.
- **Counterfactual contrast:** paired branch loss that rewards different code distributions where the control state differs and preserves shared-prefix behavior where it does not.
- **Control adherence auxiliary:** predicts structured fact/goal/constraint satisfaction from generated-turn representations; it never supplies target text.
- **Interruption/cancellation auxiliary:** predicts whether generation should continue, backchannel, yield, or be invalidated after a barge-in.
- **Regularization:** control-dropout consistency, gate sparsity, and base-audio preservation on control-free turns.

The adapter must not be rewarded for merely repeating the caller or for emitting generic closing language. Those failures are measured explicitly in held-out data.

### 4.3 Training stages

| Stage | Trainable modules | Data | Exit gate |
| --- | --- | --- | --- |
| T0 | None | Corpus integrity/codec smoke set | Native streams, masks, and revisions load deterministically. |
| T1 | Control encoder + prefix/K-V + gates | Certified corpus | Improves held-out semantic adherence without degrading audio/control-free turns. |
| T2 | Optional selected upper PersonaPlex layers | Only if T1 plateaus | Improvement exceeds regression threshold on all audio/latency/interrupt tests. |
| T3 | Runtime calibration | Held-out live-style mock calls | Revision acknowledgement and cancellation work under streaming timing. |

Training is memory-aware and distributed over CUDA `0,1,2` only. The scheduler samples current GPU utilization and leaves headroom for active services. It runs a real short epoch/forward-backward/optimizer/checkpoint cycle before any long job, then resumes from durable checkpoints.

## 5. Runtime control protocol

### 5.1 Control update lifecycle

```text
caller audio
 -> streaming ASR + turn detector
 -> call-state tree and tool results
 -> semantic LLM produces control.update revision N
 -> schema validation and policy validation
 -> control encoder caches prefix_N on GPU
 -> control.ack revision N
 -> next agent-turn snapshot {generation_id, N, prefix_N, duplex_context}
 -> PersonaPlex speech generation
 -> Twilio outbound audio
```

`control.update` is accepted only if its revision is greater than the acknowledged revision and it validates against the state schema. The session stores an immutable generation snapshot; it does not recompute guidance every 20 ms.

### 5.2 Barge-in and stale work

1. Caller speech during outbound agent audio creates a cancellation event.
2. The scheduler invalidates that `generation_id`, flushes queued outbound audio frames, and records the cutoff timestamp.
3. ASR/tool/policy changes can produce revision `N+1` while cancellation is occurring.
4. The next full response may begin only from an acknowledged latest revision. If state is not ready, PersonaPlex may wait or issue a safe backchannel, not a policy-sensitive stale answer.

### 5.3 Strict wording route

When compliance requires exact wording, the call-state layer routes to a separately validated strict renderer. PersonaPlex is not falsely represented as deterministic text realization. The control adapter remains the normal expressive, low-latency route.

## 6. Evaluation and release gates

### 6.1 Offline evaluation

Every checkpoint is evaluated on unseen counterfactual groups:

- Semantic fact incorporation and non-invention.
- Goal/constraint adherence and caller-posture appropriateness.
- Counterfactual sensitivity: same audio context plus different valid control state produces correspondingly different speech.
- Control invariance: irrelevant control change does not destabilize a response.
- Barge-in cancellation, recovery relevance, and stale-audio suppression.
- First-audio latency, steady streaming latency, real-time factor, codec quality, and voice preservation.
- ASR intelligibility, timing alignment, repetition, generic greeting/closing rate, and hallucination rate.

The report compares: base PersonaPlex, prompt-only sidecar baseline, frozen-base adapter, and any limited-adaptation checkpoint. The prompt-only baseline is required to prove that learned conditioning adds value.

### 6.2 End-to-end evaluation

Use emulated Twilio bidirectional streams first, then controlled live-style mock calls. Test network jitter, delayed ASR, dropped media frames, tool latency, control revision races, barge-in, hangup, codec round trips, and terminal action delivery.

**Release gate:** the next spoken agent turn reflects the newest acknowledged valid semantic frame in the majority of held-out causal tests, stale audio never continues after a confirmed barge-in, and no quality/latency regression exceeds the predefined budget. Exact claims require strict-renderer routing.

## 7. Execution order and live TODO

### Now: corpus production and certification

- [x] Define 500 paired counterfactual specifications for 1,000 calls.
- [x] Establish three isolated CUDA 0/1/2 generation lanes and durable per-lane progress.
- [x] Enforce Chatterbox Turbo render plus Whisper/ASR, timing, replay, and semantic gates.
- [x] Certify initial paired calls and quarantine failures rather than relaxing acceptance.
- [ ] Continue paired generation until all 500 groups are either certified or regenerated with documented cause.
- [ ] Run independent certification continuously over completed manifests and maintain a certified-only dataset index.
- [ ] Publish coverage/quota dashboard and regenerate underrepresented categories before corpus freeze.
- [ ] Freeze train/validation/test group split and produce native PersonaPlex code-stream shards.

### Next: adapter training

- [ ] Implement/control-check field-aware frame encoder and K/V prefix adapter in the PersonaPlex fork.
- [ ] Add revision embeddings, control dropout, agent-only masks, counterfactual contrastive loss, and interruption auxiliary.
- [ ] Run memory-aware multi-GPU T0 smoke epoch on certified shards.
- [ ] Run T1 frozen-base checkpoints with every-checkpoint held-out evaluation.
- [ ] Decide T2 only from measured T1 plateau and regression budget, not intuition.

### Then: runtime integration

- [ ] Add `control.update`, `control.ack`, generation snapshot, and cancellation events to the PersonaPlex streaming server.
- [ ] Cache the encoded prefix by acknowledged revision and inject it once per agent turn.
- [ ] Wire Twilio media cancellation and latest-state handoff.
- [ ] Implement strict-renderer escalation path for exact-language turns.
- [ ] Complete emulated and controlled live-call evaluations before exposing the voice option broadly.

## 8. Failure policy

- **Bad audio or ASR:** quarantine the whole branch; regenerate with a new seed/voice pairing. Never lower thresholds merely to increase count.
- **Semantic incoherence:** regenerate the scenario branch with corrected state; do not rewrite transcript after rendering.
- **Repeated pattern or placeholder:** reject via semantic review and regenerate from a fresh topic/entity plan.
- **Counterfactual collapse:** regenerate both branches from the same prefix with stronger, source-backed state divergence.
- **GPU/service contention:** stop only duplicate or owned stale workers; preserve durable progress and resume from it. Do not use GPU 3 or disrupt unrelated services.
- **Adapter fails semantic evaluation:** retain the frozen base, inspect held-out evidence, change the training objective/data balance, and retrain. Do not claim runtime control from metadata alone.

## Completion definition

This project is complete only when the certified 1,000-call corpus has passed its coverage and integrity gates; the PersonaPlex fork consumes control as a trained forward-pass input; the runtime honors revision acknowledgement and cancellation; and held-out end-to-end calls demonstrate that mutable semantic state causally changes subsequent speech without breaking latency, full-duplex behavior, intelligibility, or voice quality.
