# Causal Dataset and Synthesis Contract

## 1. Dataset purpose

The corpus teaches a conditional relationship:

```text
same audible history + different current state/control -> different valid speech
```

It is not enough to associate a topic with a typical reply. Every material
control axis needs causal pairs or larger counterfactual families that isolate
the changed state and require a meaningful target difference.

## 2. Planning cascade

The generative planning lattice has three levels:

| Level | Default count | Output |
| --- | ---: | --- |
| Topic cards | 50 | Broad domains and interaction modes, no dialogue. |
| Scenario contracts per topic | 20 | Concrete participants, state, uncertainty, tools, and outcome space. |
| Trajectory leaves per scenario | 10 | Length, posture arc, timing, control pivots, voices, and endings. |

The full lattice contains 10,000 candidate leaves. Audio generation selects a
stratified, coverage-optimized subset. The default release batch renders 500
two-branch groups, or 1,000 complete conversations. Increasing the render count
does not require changing the planning hierarchy.

All three levels are generated through typed JSON contracts with model
inference. Local regexes or hand-authored scheduling templates may not fill
semantic fields after an inference failure. Failed contracts are repaired by a
model using validation errors and the original inputs.

## 3. Topic coverage

The 50 topic cards collectively cover:

- Casual friendship, family-safe social exchange, hobbies, games, arts, and
  community activity.
- Education, tutoring, research discussion, scientific reasoning, and
  information seeking.
- Technology support, software collaboration, engineering, and operations.
- Travel planning, local navigation, housing, transport, and accessibility.
- Commerce, billing, delivery, inventory, subscriptions, and returns.
- Scheduling, reservations, event coordination, and availability.
- Health-adjacent administration without diagnosis, legal-adjacent intake
  without legal advice, and financial-adjacent support without personalized
  investment advice.
- Polling, surveys, interviews, feedback, negotiation, debate, and consensus.
- Creative ideation, critique, storytelling, media, food, sports, nature, and
  everyday problem solving.
- Safety refusal, verification boundaries, handoff, correction, recovery, and
  graceful inability.

No single domain may exceed 10% of rendered conversations. Scheduling and
customer-service openings together may not exceed 15%. Corpus-level lexical and
embedding diversity reports enforce this independently of topic labels.

## 4. Interaction coverage

Every batch includes cooperative, conditionally cooperative, skeptical,
resistant, confused, impatient, emotionally subdued, enthusiastic, and
adversarial-but-safe caller trajectories. Required phenomena include:

- Clarification and ambiguity resolution.
- Fact correction and self-correction.
- Tool success, empty result, delay, contradiction, failure, and expiry.
- Policy permission, revocation, escalation, and allowed alternative.
- Commitment creation, confirmation, revision, and cancellation.
- Topic shift, return to prior topic, and irrelevant aside.
- Natural disagreement, persuasion without manipulation, and refusal.
- Interruption before speech, early speech, mid-claim, and near completion.
- Backchannel, pause hold, false start, overlap, and repair.
- Handoff, terminal completion, caller-led ending, and agent-led ending.

Openings, closings, names, organizations, values, dates, and places are generated
as concrete synthetic entities. Literal placeholders such as "company name",
"customer name", or "insert date" are forbidden.

## 5. Counterfactual family contract

One group contains:

```text
group_id
shared_prefix_records
pivot_target_ordinal
base_state_hash
branch A control delta + evidence + target continuation
branch B control delta + evidence + target continuation
```

The shared prefix is generated and rendered once. Every branch references or
hardlinks the exact prefix audio and timeline records. Before the pivot, all of
these must match byte-for-byte or value-for-value:

- Per-turn audio SHA-256.
- Transcript and independent ASR result.
- Start, end, audible end, and overlap timing.
- Voice references.
- Base state hash and event history.
- Duplex channel assignment and codec configuration.

At the pivot, exactly one declared causal field changes. Compound real-world
updates are allowed only in a separate `multi_delta` track and never count as
single-axis counterfactual evidence.

## 6. Required causal axes

| Axis | Example branch difference |
| --- | --- |
| Tool result | `refund_pending` vs `refund_issued`. |
| Availability | known option vs no verified option. |
| Policy | escalation allowed vs supervisor approval required. |
| Identity | verified vs unresolved. |
| Caller correction | old destination vs corrected destination. |
| Commitment | tentative preference vs confirmed commitment. |
| Caller posture | cooperative vs skeptical after same fact. |
| Goal | clarify one value vs proceed with next action. |
| Evidence | ready vs failed vs expired. |
| Interruption | uninterrupted completion vs barge-in revision. |
| Termination | continue task vs end authorized. |
| Safety | ordinary compliance vs bounded refusal/alternative. |

The target branch difference is independently adjudicated before TTS. Both
targets must be individually natural and valid; one branch may not be an
intentionally nonsensical negative.

## 7. Per-turn record

Each eligible target includes:

```json
{
  "conversation_id": "...",
  "target_turn_id": 4,
  "duplex_prefix": {
    "audio_sha256": "sha256:...",
    "native_code_sha256": "sha256:...",
    "boundary_frame": 197
  },
  "control": {
    "frame": {},
    "frame_sha256": "sha256:...",
    "available_before_agent_ms": 450
  },
  "counterfactual": {
    "group_id": "...",
    "branch_id": "available",
    "changed_field": "tool_result.status",
    "base_state_hash": "sha256:..."
  },
  "labels": {
    "agent_text_sha256": "sha256:...",
    "agent_audio_sha256": "sha256:...",
    "agent_only_native_mask_sha256": "sha256:..."
  }
}
```

The training manifest does not contain plaintext target wording. Plaintext
labels live in a separately permissioned lineage artifact for auditing and ASR
alignment. The model input loader never joins them into control tokens.

## 8. Timing synthesis

Timing is sampled from approved real-call distributions by interaction type,
then constrained by conversational semantics. It is not a fixed silence pad.

Required outputs include:

- Caller and agent start/end times.
- Audible cutoff separate from rendered end.
- Word timings from independent ASR.
- Planned gap or overlap and observed gap or overlap.
- Barge-in detection, clear, and last-valid-media timestamps.
- Evidence/control available time and target boundary time.

For counterfactual pivots, pre-pivot timing is shared exactly. Post-pivot timing
may differ when the control delta legitimately changes hesitation, interruption,
brevity, or ending behavior.

## 9. Audio generation and authenticity

Chatterbox Turbo is the default synthesis renderer. Every utterance uses a
provenance-approved 5-10 second voice reference and CUDA inference. Voice pairs
are distinct unless a specific same-speaker test is declared.

Whisper validates rendered speech through transcript, word timing, confidence,
duration, and WER. The admission policy is pragmatic:

- Severe mismatch, empty speech, malformed timing, clipping, or wrong language
  is rejected and regenerated.
- Marginal WER is reviewed with semantic/entity checks rather than rejected by
  one global threshold alone.
- A local patch may rerender only the failed turn. It must preserve the shared
  prefix and all accepted neighboring turns.
- Semantic-control certification remains a separate model-inference judgment.

No CPU model fallback is allowed. Host CPU work may assemble files, calculate
hashes, and run ordinary orchestration.

## 10. Certification stages

The dataset passes these gates in order:

1. Schema and prohibited-label audit.
2. Voice provenance and license audit.
3. Conversation completeness and model-driven terminal action audit.
4. Counterfactual shared-prefix and one-delta audit.
5. Independent semantic branch-validity audit.
6. Audio decode, channel, sample-rate, clipping, and duration audit.
7. Whisper transcript, word timing, and intelligibility audit.
8. Native Mimi encoding and exact stream-layout audit.
9. Agent-only target-mask audit.
10. Group/topic/voice-isolated split audit.
11. Corpus coverage, duplication, opening, and closing diversity audit.
12. Signed immutable certificate binding every manifest and model artifact.

Any repair creates a new artifact hash and re-runs downstream gates. Passing
neighbor turns are retained; an entire conversation is regenerated only when
its causal state or shared prefix is invalid.

## 11. Existing V8 migration

The existing V8 corpus is migrated into v4 pair indexes without rerendering:

- Preserve nested `counterfactual` metadata through native manifests.
- Identify the unique pivot where branch, target turn, and base state match but
  post-update state differs.
- Verify replay records have identical prefix audio/text/timing.
- Store explicit partner example ids in a pair index.
- Keep all group members in one split.
- Quarantine incomplete groups from pair loss while retaining independently
  valid turns for ordinary SFT.

Observed migration inventory before v4 implementation:

```text
complete certified source groups: 439
byte-identical replay prefixes: 439
native causal pivot pairs after quarantine: 396
train pairs: 313
validation pairs: 46
test pairs: 37
```
