# Runtime Semantic Control Protocol

## 1. Purpose

The protocol connects a semantic controller to PersonaPlex without falsely treating network receipt as model conditioning. It provides a narrow, auditable control path and a strict fallback when native audio generation cannot meet a semantic guarantee.

## 2. Message envelope

All messages use a versioned JSON body over the bridge control channel. The envelope is canonicalized before hashing and signing/logging.

```json
{
  "type": "control.update",
  "protocolVersion": 1,
  "callId": "CA...",
  "turnId": 17,
  "revision": 42,
  "contextHash": "sha256:...",
  "sentAtMs": 1784030000000,
  "expiresAtMs": 1784030007000,
  "plan": { "...": "ControlPlan defined in ARCHITECTURE.md" }
}
```

The bridge returns exactly one terminal status for each tuple `(callId, revision)`:

```json
{
  "type": "control.ack",
  "protocolVersion": 1,
  "callId": "CA...",
  "turnId": 17,
  "revision": 42,
  "contextHash": "sha256:...",
  "status": "applied",
  "boundaryId": "...",
  "appliedAtMs": 1784030000123,
  "modelRevision": "...",
  "prefixLatencyMs": 12
}
```

Allowed statuses: `queued`, `applied`, `superseded`, `rejected`, `expired`, `prefix_build_failed`, `context_mismatch`, `safe_fallback`. A `queued` status is informative but not permission to emit controlled audio. Only `applied` activates a revision.

## 3. State machine

```
created -> sent -> received -> validated -> queued -> boundary_seen -> applied -> acknowledged
                      |             |             |
                      v             v             v
                   rejected      superseded     fallback
```

Rules:

- `revision` is monotonically increasing within a call.
- The server keeps only the newest valid pending revision for a given future boundary.
- Duplicate transmission of an already terminal update returns the original terminal status.
- Any mismatch in `callId`, `turnId`, context hash, expiry, policy, or schema rejects the update.
- A caller barge-in cancels uncommitted agent media and invalidates queued plans derived from obsolete state.
- The controller sends an explicit boundary marker when it has finalized a caller turn; audio-plane VAD may also produce a boundary candidate, but policy decides whether it is safe.

## 4. Boundary application

At a valid boundary the server:

1. Confirms that the update is still latest, valid, unexpired, and context-consistent.
2. Materializes the semantic prefix through the registered adapter.
3. Prefills only the scoped per-call generation state.
4. Records the applied revision and timing.
5. Emits `control.ack status=applied`.
6. Allows agent audio generation only after the result is available.

If prefix construction misses the latency deadline, fails validation, or cannot prove applied state, the audio plane must not emit a response falsely attributed to that plan.

## 5. Strict response contract

For strict mode, the semantic plane supplies a separate rendering contract:

```json
{
  "callId": "CA...",
  "turnId": 17,
  "contextHash": "sha256:...",
  "canonicalText": "I can check Thursday afternoon availability. What time works best?",
  "renderer": {"engine": "approved_engine", "voice": "consented_voice_id"},
  "constraints": {"no_paraphrase": true, "max_duration_ms": 6500}
}
```

The strict renderer is independently tested by ASR and text normalization. The contract is held in protected call state and must not be serialized into the expressive training plan.

## 6. Transport behavior

Twilio media control has transport-specific consequences:

- Outgoing audio must be paced to media timestamps; burst writes are not realistic call behavior.
- A barge-in clears buffered agent audio and cancels the active generation/request.
- Mark events associate sent media with control revision and support end-to-end latency measurement.
- Codec conversion uses an explicit test vector for mu-law, sample rate, channel count, and clipping behavior.
- The bridge provides bounded queues and drops stale media rather than speaking a stale response late.

## 7. Current implementation truth

The fork now contains an executable, pinned-upstream overlay:
`personaplex_control.controlled_server` targets upstream PersonaPlex commit
`3428dfd95309a7f3c84fd93259ded0f810d1ff91`. It accepts V2 typed control
messages on binary kind `0x04`, emits acknowledgements on kind `0x05`, and
uses `SemanticPrefixProvider` to encode a `ControlTrainingFrame` once on GPU,
cache it by `frameHash`, and prefill the live `LMModel.forward_embeddings`
stream at a confirmed caller boundary. Prefix outputs are discarded; they are
not rendered as audio or injected as a system prompt.

This implementation is deliberately **unverified** until
`evaluation/runtime_prefix_harness.py` passes on CUDA with a trained adapter,
the actual pinned Moshi checkpoint, and a batch-certified V2 frame. Until that
artifact exists, the runtime must remain experimental and no deployment may
claim applied semantic control.

The harness is the next gate. It fails unless `queued` is followed by
`applied`, where `applied` is emitted only after direct transformer prefill and
includes the adapter revision plus build/prefill timing.

## 8. Observability

For every control attempt, record structured fields:

- call/session pseudonym; turn; revision; context hash; plan schema version;
- receipt, validation, queue, boundary, prefix, acknowledgement, and first-audio timestamps;
- terminal status/reason; model and adapter revision; mode; renderer revision;
- ASR confidence and caller interruption event; and
- safe-fallback activation.

Do not log raw transcripts, phone numbers, sensitive entity values, canonical text, or voice data in broad operational logs. Store protected evidence separately with access control.

## 9. Operational policy

- Missing `control_ack`: strict/fallback only.
- Stale plan: replan; never emit it after the caller changes topic.
- Guardrail rejection: record typed reason, then use policy-approved fallback.
- Codec/stream failure: stop media and transfer/end per call policy.
- Observability failure: the call may continue only in a safe non-controlled mode; do not claim compliance that cannot be measured.

## 10. Rolling state and multi-agent authority

The semantic plane maintains a per-call `SemanticState` tree. State reducer,
task, policy, knowledge, and safety agents may propose bounded patches; only the
state reducer emits a monotonic revision with a base-state hash and new-state
hash. A `ControlTrainingFrame` is derived from that revision and contains the
typed plan to apply at the next agent boundary. Raw agent prompts, target wording,
and unbounded tool output are not accepted on the audio-plane control channel.

On caller barge-in, the bridge immediately clears unsent agent media, records the
audible cutoff, invalidates any pending frame based on the prior state, and emits
an interruption event to the reducer. The following response requires a fresh
state revision and `control.ack status=applied`; a stale plan cannot resume after
the caller has changed the conversational context.
