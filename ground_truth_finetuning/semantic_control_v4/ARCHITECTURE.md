# Semantic Control v4 Architecture

## 1. Decision

PersonaPlex remains the low-latency full-duplex speech body. A deeper semantic
service maintains the call-state tree and emits typed revisions. The model fork
gains one native temporal conditioning stream, trained to make those revisions
causally useful during text and speech-token generation.

The first evidence-backed implementation uses MoshiRAG-style additive temporal
injection. It does not add an unverified per-layer K/V mechanism before the
native stream design has been measured. Limited upper-temporal LoRA is the
planned capacity expansion when the frozen-base adapter reaches a causal
adherence ceiling.

## 2. End-to-end planes

```text
Audio plane
  Twilio 8 kHz mu-law
    -> paced decoder/resampler
    -> Mimi caller stream
    -> PersonaPlex temporal + depth transformers
    -> Mimi agent stream
    -> paced mu-law egress

Semantic plane
  caller audio
    -> parallel streaming ASR and turn observations
    -> immutable call events
    -> state reducer
    -> Nemotron reasoning, tools, policy, and safety agents
    -> typed control revision

Control plane
  typed control revision
    -> schema and freshness validation
    -> field-aware encoding on CUDA
    -> immutable control-stream cache entry
    -> snapshot at next agent boundary
    -> native temporal streaming-sum injection
```

The semantic plane may run more slowly than the 12.5 Hz audio plane. It must
produce a valid next-turn frame before policy-sensitive speech begins. If it
does not, the audio plane may wait or emit only a specifically allowed
backchannel.

## 3. Control frame v2

The frame is compact, typed, target-free, and declarative. Required categories
are:

| Category | Meaning |
| --- | --- |
| Identity | Call, target turn, revision, base/current state hashes. |
| Freshness | Effective boundary, predecessor revision, expiry, generation policy. |
| State | Intent, phase, facts, uncertainty, commitments, unresolved items. |
| Caller | Posture, resistance, compliance, affect when confidently observed. |
| Goal | Next semantic goal and intended dialogue act. |
| Obligations | Required facts, questions, actions, and forbidden claims/requests. |
| Evidence | Bounded tool/result records with source, status, revision, and claims. |
| Style | Language, register, warmth, assertiveness, brevity, rate, pauses. |
| Duplex | Yield policy, expected overlap, recovery state, response duration. |
| Termination | End authorization, terminal action, handoff state. |
| Lineage | Counterfactual group, branch, changed field, and provenance hashes. |

No frame may contain `targetText`, `canonicalText`, `response`, `reply`,
`verbatim`, target audio, or a paraphrase intentionally constructed from the
label. Strict text lives only in the separate strict-render contract.

## 4. Field-aware encoding

The control-v3 serializer flattens most values into underscore atoms and feeds
them into a newly initialized full-size embedding table. V4 replaces both
choices.

For each semantic field, the encoder emits a sequence of records:

```text
token_id, field_id, value_type_id, source_id, revision_relation_id, position
```

Text values retain ordinary spaces and punctuation after control-character
escaping. Field names use a closed vocabulary. Unknown extension fields map to
`extension` and retain a stable hashed sub-id; they do not create unbounded
embedding tables.

Critical fields have reserved token budgets in this order:

1. Obligations and forbidden behavior.
2. New tool, policy, correction, and interruption deltas.
3. Current facts, uncertainty, and commitments.
4. Goal, dialogue act, and termination state.
5. Caller posture and duplex behavior.
6. Short audible context.
7. Style and noncritical extensions.

Truncation may remove only lower-priority tail content. It cannot remove all
tokens for any populated critical field. The encoder records per-field token
counts and truncation flags in the run artifact.

## 5. Learned temporal control stream

`SemanticControlStreamAdapter` receives frozen PersonaPlex text embeddings for
the lexical token ids. Reusing the base embedding gives unseen names, facts,
and ordinary language a meaningful starting representation. The adapter adds
trainable field, value-type, source, revision-relation, and position embeddings.

The default small model is:

```text
PersonaPlex text embedding: 4096, frozen
input projection: 4096 -> 1024
field/type/source/revision embeddings: 1024
context encoder: 4 transformer layers, 16 heads
compression queries: 48 x 1024
cross-attention pooling: encoded fields -> 48 temporal rows
output projection: 1024 -> 4096
learned per-row gate initialized conservatively
```

The 48 rows span 3.84 seconds at 12.5 Hz. Their effect persists through the
temporal transformer's causal cache after direct injection ends. Stream length
is configurable and checkpoint-bound; it is never silently changed at runtime.

A learned null stream is used for control dropout and safe sparse-state
training. It is not used to bypass missing required control at runtime.

## 6. Native model injection

At a valid boundary, the scheduler queues one immutable `[T, 4096]` stream for
the call's batch slot. Immediately before each native `LMGen.step`, one row is
added to the ordinary sum of agent text, agent audio, and caller audio
embeddings. Exhaustion writes an explicit zero row.

This mirrors MoshiRAG's published additive reference path and preserves:

- Native delayed codebook semantics.
- Full-duplex caller and agent streams.
- The temporal transformer's streaming cache.
- PersonaPlex voice prompting.
- The depth transformer's native speech-token generation.

The condition is applied to real temporal steps. Unlike control-v3, it does not
advance the temporal cache through silent virtual prefix frames.

## 7. Limited model adaptation

Stage 1 freezes PersonaPlex and trains only the control stream adapter. Stage 2
is enabled only if held-out pair accuracy or generated semantic adherence
plateaus below contract.

Stage 2 adds low-rank adapters to explicitly enumerated upper temporal
transformer projections. It does not adapt Mimi, voice-prompt handling, the
caller stream embeddings, or the depth transformer initially. Text-stream
counterfactual preference loss is used because published full-duplex preference
work reports instability when audio-token probabilities enter DPO directly.

Each LoRA checkpoint records:

- Exact module paths and source fingerprint.
- Rank, alpha, dropout, and trainable parameter count.
- Base, control-adapter, dataset, and reference-policy hashes.
- Frozen parameter audit.

If upper-temporal LoRA damages timing, voice, no-control behavior, or
interruption handling, the candidate is rejected rather than compensated with
runtime heuristics.

## 8. Runtime state machine

One call owns:

```text
last_seen_revision
last_acknowledged_revision
pending_control_entry
active_control_snapshot
generation_id
cancellation_token
outbound_media_queue
terminal_action_state
```

The state transitions are:

```text
control.update -> validate -> encode/cache -> queued
queued + matching boundary -> snapshot/queue stream -> applied
newer revision -> supersede pending/active -> invalidate generation id
barge-in -> invalidate generation id -> clear media -> cancel condition rows
expired/mismatched frame -> fail closed
terminal model action -> end-call tool -> stop future generation
```

The egress queue carries `generation_id` with every media packet. It checks
validity immediately before network write. Clearing the queue and invalidating
the id happen before semantic replanning.

## 9. Mid-utterance updates

V4 does not pretend already encoded audio can be revised. A new update during
agent speech has one of three typed policies:

| Policy | Runtime action |
| --- | --- |
| `cancel_and_replan` | Stop media, invalidate generation, apply at next boundary. |
| `defer_to_next_turn` | Finish current non-sensitive utterance, then apply. |
| `strict_interrupt` | Stop expressive route and invoke validated strict rendering. |

The default for caller barge-in, corrected facts, revoked permission, and
policy change is `cancel_and_replan`.

## 10. Termination

Natural signoff remains model-driven. The control frame exposes termination
state and allowed action, and the model is trained on varied natural endings.
The model's private text/action stream emits a typed terminal token or action.
The scheduler invokes the end-call tool once, drains only already valid media,
and rejects repeated terminal actions. There is no phrase-list or deterministic
farewell detector in the semantic gate.

## 11. Strict rendering boundary

The semantic authority selects strict rendering when wording must be exact.
PersonaPlex may provide a natural lead-in only when explicitly allowed, then its
egress is paused while the strict renderer emits validated audio. The strict
route has independent ASR, entity, and codec gates and carries the same
generation id and cancellation semantics.
