# Semantic-control architecture

## Guarantees and non-guarantees

PersonaPlex provides a low-latency conversational audio layer. It does not
provide an exact-wording guarantee when conditioned by a natural-language
prompt. The system must not label guided audio as LLM-authoritative speech.

`strict` mode produces the semantic authority's canonical `target_text` with
deterministic streaming TTS. `expressive` mode lets PersonaPlex realize a
semantic plan, checks its output with ASR, and falls back to strict mode when
required content is missing or prohibited content is detected.

## Planes

### Audio plane

`Twilio bidirectional stream -> mu-law bridge -> Opus -> PersonaPlex -> Opus
-> mu-law -> Twilio`

It owns packet timing, media timestamps, interruption detection, and the
initial PersonaPlex WebSocket handshake. No semantic controller may reset or
prefill a model while the caller or agent is speaking.

### Semantic plane

Streaming caller ASR is stabilized into a final caller turn. A text LLM reads
that turn, current call state, retrieved data, tool results, and policy. It
emits a `SemanticPlan`, not an untyped prompt:

- intent and verified facts;
- required entities and allowed/prohibited claims;
- spoken style;
- canonical `target_text` when exact wording matters;
- monotonic revision, call id, context hash, and target boundary.

The semantic plane is authoritative for tools and regulated, transactional,
or factual content.

### Render and arbitration plane

The controller queues each `control.update`, rejects stale/context-mismatched
updates, and applies a valid update only after the next completed caller turn.
It returns `control.ack` with the applied revision and turn. The server adapter
must make the acknowledgement observable in the call timeline.

In expressive mode the renderer sends a compact semantic plan to PersonaPlex
before it begins the following response. It then compares incremental ASR to
the plan. Missing required entities, a forbidden claim, tool mismatch, or a
timeout terminates PersonaPlex egress and starts strict TTS from the canonical
text. The fallback must be audible before the system asserts completion.

## Wire contract

`control.update` is UTF-8 JSON inside a control frame. It has this shape:

```json
{
  "type": "control.update",
  "call_id": "CA...",
  "revision": 18,
  "apply_after_turn_id": 7,
  "base_context_hash": "sha256:...",
  "context_hash": "sha256:...",
  "mode": "strict",
  "plan": {
    "intent": "reschedule appointment",
    "facts": ["Thursday availability is not yet confirmed"],
    "required_entities": {"day": "Thursday"},
    "allowed_claims": ["ask for preferred Thursday time"],
    "prohibited_claims": ["do not confirm a booking"],
    "target_text": "I can help with Thursday. What time works best?",
    "style": "brief and helpful"
  }
}
```

The matching `control.ack` contains `call_id`, `revision`, `applied`, `reason`,
and `turn_id`. A queued acknowledgement is not permission to render. Only an
`applied: true` acknowledgement proves the active revision.

## State and cancellation

The context hash covers the final caller turn, retained history, tool result
versions, and policy revision. New caller speech cancels pending speculative
work. Tool results that arrive after the plan was created require a new
revision. The server must retain revision, decision, boundary time, ASR result,
and fallback reason for every spoken response.

## Deployment boundary

The audio server extension is installed against a pinned upstream PersonaPlex
revision. It must consume control frames separately from audio frames and
return an acknowledgement. It must not use an arbitrary low-energy audio chunk
as an implicit safe reset point. The Voryn bridge owns PSTN conversion; the
server extension owns model state; the semantic service owns policy and tools.

## Native prefix application

The controlled runtime is pinned to upstream PersonaPlex commit
`3428dfd95309a7f3c84fd93259ded0f810d1ff91`. At a valid caller boundary it
does not replace PersonaPlex's text prompt or reset its duplex state. Instead:

1. `control.update` carries a V2 `ControlTrainingFrame`, a matching state
   revision/context hash, and an absolute expiry time.
2. The server validates the target-wording-free frame, serializes it with the
   same `PlanSerializer` used by training, and caches the adapter's `K` GPU
   virtual embeddings by `frameHash`.
3. At `control.boundary`, it verifies the current state hash, feeds those `K`
   embeddings through `LMModel.forward_embeddings` one causal frame at a time,
   discards all prefix outputs, and preserves the existing streaming cache.
4. Only then does it emit `control.ack status=applied` and permit media tagged
   with the new generation ID. A new boundary or caller barge-in invalidates
   queued media and the active generation ID.

The extension uses binary message `0x04` for control input and `0x05` for
acknowledgements. This is a causal model input path. It remains experimental
until the CUDA-native prefix harness and full Twilio emulation gate pass.
