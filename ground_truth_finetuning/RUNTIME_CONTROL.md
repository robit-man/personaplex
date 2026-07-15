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

The repository currently contains an experimental websocket bridge and PersonaPlex server overlay for versioned control messages. It is not yet an established semantic-prefix runtime. A direct in-container control test sent an update, boundary, and silence prefill but timed out waiting for `control_ack`. Therefore no component may report the revision as applied or describe the current path as proven semantic control.

The next runtime task, after the adapter and harness exist, is an isolated protocol test that records every transition above and fails on a missing terminal acknowledgement. Only then should the Twilio emulation harness exercise it.

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
