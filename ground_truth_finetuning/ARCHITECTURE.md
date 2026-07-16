# Architecture: Semantic Control for a Native Audio Model

## 1. Objective

The architecture must make semantic guidance causally available to the speech model while preserving full-duplex speech behavior. The controller cannot merely display a text answer after PersonaPlex speaks. It must produce a versioned plan that reaches the audio model at an acknowledged turn boundary, and the system must measure whether later audio follows that plan.

## 2. Three planes

### 2.1 Audio plane

```
Twilio Media Stream <-> codec bridge <-> PersonaPlex server <-> PersonaPlex LM
```

Responsibilities:

- Convert Twilio 8 kHz mu-law frames to the model's expected PCM/Opus representation and convert generated speech back to Twilio transport audio.
- Preserve media timestamps, sequence ordering, clear/mark semantics, barge-in behavior, and backpressure.
- Keep a short jitter buffer with bounded latency.
- Receive control messages but never invent business state.
- Emit applied-revision acknowledgements, media timing, interruption state, and audio-plane diagnostics.

### 2.2 Semantic plane

```
caller audio -> streaming ASR -> dialogue state -> deeper LLM -> typed ControlPlan
```

Responsibilities:

- Maintain the authoritative per-call transcript, entity ledger, policy state, and dialogue goal.
- Generate a typed plan from the caller turn and current state.
- Decide whether a response must be strict or may be expressive.
- Reject stale plans and unsafe plan transitions.
- Keep semantic data separate from raw audio, with data retention and access controls defined in `DATA_AND_GOVERNANCE.md`.

The deeper LLM is not in the per-frame audio loop. It operates in parallel while the caller speaks and prepares a plan before the boundary. This is the mechanism that allows an LLM-level context window without blocking audio-frame generation.

### 2.3 Control plane

```
semantic plan -> versioned control update -> bridge -> server boundary queue -> LM prefill -> ack/telemetry
```

Responsibilities:

- Bind each update to `call_id`, `turn_id`, `revision`, and `context_hash`.
- Deliver updates atomically and idempotently.
- Apply only the newest valid revision at a safe boundary.
- Acknowledge the applied revision with model and timing metadata.
- Route strict plans to deterministic TTS when exact wording is required.
- Expose enough telemetry to distinguish rejected, superseded, queued, applied, and stale updates.

## 3. Runtime modes

| Mode | Audio owner | Semantic guarantee | Intended use |
| --- | --- | --- | --- |
| `strict` | Deterministic renderer | Canonical wording is exact after renderer/ASR validation. | Regulated statements, numbers, commitments, user-requested verbatim responses. |
| `expressive` | PersonaPlex | Plan-level compliance is measured, never presumed exact. | Natural greeting, backchanneling, low-risk conversational flow. |
| `safe_fallback` | Deterministic renderer or transfer | No PersonaPlex output is emitted. | Missing acknowledgement, stale plan, policy failure, media failure, or uncertain state. |

An expressive result may be evaluated against a canonical response, but it must not be advertised as a deterministic rendering of that response.

## 4. Control data model

A `ControlPlan` is structured, bounded, and serializable. It is not a free-form prompt blob.

```json
{
  "schemaVersion": 1,
  "callId": "CA...",
  "turnId": 17,
  "revision": 42,
  "contextHash": "sha256:...",
  "mode": "expressive",
  "intent": "reschedule_appointment",
  "dialogueAct": "confirm_availability",
  "entities": {
    "requested_day": "Thursday",
    "time_window": "afternoon"
  },
  "constraints": {
    "required_facts": ["availability_unknown"],
    "forbidden_claims": ["appointment_confirmed"],
    "must_ask": ["preferred_time"],
    "must_not_request": ["payment_card"]
  },
  "delivery": {
    "language": "en-US",
    "register": "calm_professional",
    "assertiveness": 0.45,
    "interruptibility": "yield_on_caller_speech",
    "max_duration_ms": 6500
  },
  "expiryMs": 7000
}
```

`canonicalText` is deliberately absent. When strict mode is selected, canonical text travels on a separate protected rendering contract and is never used as a semantic-prefix feature during training.

Validation rules:

- `revision` strictly increases for a call.
- `contextHash` is the hash of the normalized dialogue state consumed by the planner.
- `expiryMs` is bounded and expires relative to receipt time.
- All categorical fields use a versioned enumeration.
- Entity values are canonicalized, typed, and redacted when prohibited by policy.
- A plan cannot request strict mode without a valid canonical rendering contract.

## 5. Semantic-prefix adapter

### 5.1 Design

The initial trainable mechanism is a small adapter ahead of a frozen PersonaPlex language model:

```
ControlPlan -> deterministic serialization -> existing text tokenizer/embedding
            -> plan encoder + projection -> P learned prefix frames
            -> PersonaPlex delayed temporal path -> agent text/audio logits
```

The adapter is the sole learned path for new semantics in stage 1. It must emit `P` prefix-frame embeddings in the exact hidden dimension expected by the base model. Its serialization uses fixed field order, typed delimiters, bounded value vocabularies, and an explicit schema version. Unknown fields fail validation instead of silently altering prompts.

The prefix is prefilled at a turn boundary, after the user context is represented and before agent generation begins. The server must not reset the entire conversation merely to install a new plan. The prior conversation state remains causal context, while the prefix scopes the subsequent agent response.

### 5.2 Why a prefix adapter first

- It preserves the pretrained full-duplex model and reduces catastrophic degradation risk.
- It creates a narrow, inspectable control interface.
- It can be ablated against a no-plan baseline and a shuffled-plan control.
- It avoids pretending that an external LLM text overlay changes PersonaPlex speech generation.

### 5.3 Training inputs and targets

For each training item:

- Input context: system prompt, consented voice prompt, prior caller/agent streams, and a `ControlPlan` with no canonical response text.
- Labels: verified agent text stream and aligned agent audio code streams for the next agent turn.
- Target mask: agent text and agent audio only.
- Context-only streams: caller audio and prior history.

The model configuration determines codebook count and stream offsets at export and train time. PersonaPlex/Moshi implementations commonly use a text stream at index zero and separate agent/user audio blocks, but the exporter must query the loaded model configuration rather than encode static indices.

### 5.4 Causal and delay correctness

PersonaPlex/Moshi uses delayed code generation. The adapter training path must call the same delay, `forward_codes`, depformer, and undelay mechanics as native training. A separate shortcut that feeds unshifted audio codes is invalid. The exporter records code layout and delay configuration in each run manifest.

## 6. Server integration contract

The server fork requires an explicit adapter interface rather than ad-hoc prompt replacement:

```python
class SemanticPrefixProvider(Protocol):
    def validate(self, update: ControlUpdate) -> None: ...
    def build_prefix(self, plan: ControlPlan, model: LMModel) -> Tensor: ...
    def prefill_at_boundary(self, state: GenerationState, prefix: Tensor) -> GenerationState: ...
```

Required properties:

- No model-side access to unvalidated JSON.
- Prefill is bounded by a maximum prefix-frame count and a deadline.
- Prefix state is per-call and never leaks across calls.
- The applied revision, context hash, model version, and prefix latency are emitted in `control_ack`.
- A failure returns a structured negative acknowledgement and selects `safe_fallback`.

The existing fork's wire protocol may temporarily carry role-guidance text, but that is a transport experiment only. It is not the semantic-prefix architecture until the server calls a real prefix provider and reports a verified applied revision.

## 7. Context window and state mutation

The semantic plane owns the long-lived state window. It may revise goals, entities, policies, or delivery settings while a call is active. To make mutation safe:

1. Normalize the current state and calculate `contextHash`.
2. Produce `revision + 1` from that state.
3. Send the update while the caller is still speaking when possible.
4. At the next boundary, apply only if the update is current, unexpired, policy-valid, and acknowledged.
5. Record a new state snapshot after the agent turn.

An already emitted audio frame cannot be retroactively changed. A caller interruption invalidates queued agent audio and triggers a fresh planner turn. The controller never claims otherwise.

## 8. Failure and rollback design

- No acknowledgement before deadline: suppress expressive output and use strict/fallback response.
- Context hash mismatch: reject update, replan from current state.
- Revision gap or duplicate: accept only idempotent duplicate acknowledgements; otherwise reject.
- Prefix compute failure: record `prefix_build_failed`, do not fall through to stale guidance.
- ASR uncertainty above threshold: request clarification or transfer; do not synthesize guessed entities.
- Media transport failure: emit no hidden retry audio; end or transfer under the call policy.

## 9. Explicit non-architecture

The following are not acceptable substitutes for this design:

- An LLM call after PersonaPlex has generated speech.
- Resetting all model history on a generic quiet chunk.
- Passing canonical target wording into the training control feature.
- Hardcoding seventeen codebooks because one checkpoint used that configuration.
- Declaring control because a websocket accepted a message without an applied-revision acknowledgement.

## 10. ControlTrainingFrame and rolling semantic state

The training unit is a `ControlTrainingFrame`, not an appended prompt and not a
canonical response. A frame contains a hash-linked immutable snapshot of the
rolling call tree, the semantic agents that contributed it, the typed plan for
the forthcoming agent turn, expiry, and the expected turn-taking behavior.

The call tree contains only bounded semantic fields: phase, objective,
commitments, unresolved items, policy boundaries, compliance/resistance posture,
and repair posture. It is updated by a state reducer after every audible event.
Task, policy, knowledge, and safety agents submit typed patches to that reducer;
the reducer emits the next state revision. The runtime validates the base hash,
new hash, revision, and expiry before deriving the plan prefix.

`ControlTrainingFrame` serializes state and plan in a fixed typed order. Target
wording, canonical responses, reply fields, and verbatim fields are rejected at
schema validation. The adapter sees the frame before an agent turn; the target
text/audio remains a label only.

## 11. Full-duplex interruption and recovery

Training examples preserve two independently aligned tracks. An agent may be
interrupted only when its control policy permits yielding. The timeline records
the barge-in point, cancellation latency, audible end of agent media, and the
recovery expectation. Audio queued after the audible end is discarded from the
agent target mask. The caller's interjection is conditioned on the audible prefix
only, and the next agent turn receives a new post-interruption frame.

This makes a barge-in causal: it is neither a textual label pasted onto a
sequential transcript nor a target that asks the model to learn unheard speech.
Offline synthesis may initially use separately rendered mono clips, but the
exporter must materialize the declared duplex timeline before Mimi encoding.
