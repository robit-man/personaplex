# Causal Coverage Overhaul

Status: active ground truth

This document records the empirical failure of the first ARC4 causal corpus and
defines the replacement data, training, and evaluation contract. A large corpus
must not be admitted merely because it has many topics, voices, or turns. It must
repeatedly teach reusable causal relationships between a typed control update and
the next speech-token trajectory.

## 1. Measured failure

The reproducible teacher-forced held-out trajectory is:

| Checkpoint | Trainable block | Held-out directions | Held-out full pairs | Train-sample directions | Train-sample full pairs |
|---|---|---:|---:|---:|---:|
| current v9 baseline | control adapter | 18/92 | 3/46 | not used | not used |
| step 150 | control adapter | 30/92 | 4/46 | 43/92 | 12/46 |
| step 175 | upper temporal LoRA | 31/92 | 5/46 | 45/92 | 15/46 |
| step 200 | upper temporal LoRA | 32/92 | 4/46 | 50/92 | 17/46 |

Continued optimization improves sampled training pairs while held-out full-pair
sensitivity remains approximately 9-11 percent. This is not convergence and is
not evidence of live semantic controllability.

The structural certificate for the 393-pair corpus found:

- 92.62 percent composite interventions.
- 3.82 percent of pairs belong to a changed-path signature repeated across at
  least three distinct premises.
- 393 groups and exactly one two-branch pair per group.
- 100 scenario premises and 12 semantic axes, but only three values each for
  caller posture, compliance posture, and resistance posture.
- Barge-in and recovery flags are common, but all serialized event types are
  `completed_turn`; the flags are usually entangled with unrelated fact and
  delivery changes.
- Held-out full-pair passes are zero for consent, delivery, handoff,
  interruption, policy, preference, safety, and identity-verification axes at
  step 175.

The failure is therefore structural. Most branch pairs simultaneously alter
semantic facts, delivery style, and turn-taking. Nearly every resulting
changed-path signature is a one-off. The model can memorize composite training
signatures but cannot learn a reusable semantic operation.

The machine-readable current-corpus report is:

`/srv/voxrn_cache/personaplex/training/arc4-native-controlv3-causal-v2/causal_coverage_certificate.v1.json`

## 2. Non-negotiable causal unit

Every training unit MUST be a counterfactual sibling group sharing:

- The same conversation prefix and native delayed duplex code streams up to the
  intervention boundary.
- The same speaker identities and voice references.
- The same target-turn boundary.
- One immutable, acknowledged control snapshot per branch.
- A target response transcript/audio/code stream used only as a label.
- A typed declaration of the one intervention family being trained.

Target wording, target transcript fragments, target audio embeddings, opaque
branch names, and post-response facts MUST NOT enter the control input.

Each group SHOULD contain at least four causally meaningful siblings rather than
only `available` and `constrained`:

1. A verified positive state or permitted action.
2. A verified negative state or prohibited action.
3. An uncertain, unavailable, or not-yet-verified state requiring a safe wait,
   clarification, or bounded backchannel.
4. A superseding revision, correction, tool result, or interruption that makes
   the previously planned response stale.

Pair rows may be derived from the sibling group, but all derived pairs retain the
group identifier and remain in one split.

## 3. Orthogonal intervention families

### 3.1 Semantic intervention

Change only typed semantic state such as facts, tool results, policy boundaries,
availability, uncertainty, next goal, caller posture, or commitments.

Hold fixed:

- Delivery register, rate, assertiveness, warmth, pause density, and duration.
- Turn-taking policy and actual duplex timing.
- Speaker references and codec path.
- Shared acoustic/text prefix.

The target semantic realization must materially differ while preserving the
fixed delivery contract.

### 3.2 Delivery intervention

Change only typed delivery controls. Hold semantic claims, next action, tool
state, policy, timing events, and speaker identity fixed. Targets should be
semantically equivalent but differ in the requested delivery dimension.

### 3.3 Turn-taking intervention

Change only the actual interaction trajectory: barge-in onset, cancellation
cutoff, yield timing, recovery boundary, or safe backchannel requirement. Hold
semantic state and delivery settings fixed unless a later, separately versioned
semantic update is introduced.

An interruption example is valid only when the duplex audio/code stream contains
the overlap, outgoing-audio cutoff, and recovery turn. A label stating that an
interruption occurred is insufficient.

### 3.4 Composite trajectories

Real calls contain compound changes, so a bounded minority of groups may contain
multi-family trajectories. They are admitted only after each component operator
has repeated single-family support. Composite examples cannot dominate adapter
training and cannot satisfy single-family coverage quotas.

## 4. Repeated causal operators

Topic diversity and causal diversity are separate axes. Each typed operation
must recur across many unrelated premises, voices, turn positions, and lexical
realizations. Examples include:

- `tool_result: pending -> issued`
- `availability: open -> unavailable`
- `policy: escalation allowed -> escalation prohibited`
- `evidence: unverified -> verified`
- `identity: insufficient -> verified`
- `caller posture: cooperative -> skeptical`
- `revision: current -> superseded`
- `generation: active -> cancelled by barge-in`

The operator identity is structural metadata for balancing and certification; it
is not an opaque shortcut token passed to the model. The model receives concrete
typed values and their revision, not branch labels.

Each operator signature must be represented across enough distinct premises to
make topic/template memorization an invalid solution. The minimum support and
maximum composite fraction are explicit dataset-contract parameters consumed by
the certificate, not hidden constants in code.

## 5. Cascade contract

The generative cascade remains useful but its levels have separate duties:

1. `50` top-level seed domains establish broad subject distribution.
2. Up to `20` scenario contracts per seed establish distinct participants,
   stakes, goals, environments, and conversational forms.
3. Up to `10` trajectory leaves per scenario establish varied lengths, voices,
   opening positions, cooperation/resistance paths, and endings.
4. Each selected trajectory contains multiple typed control pivots.
5. Each pivot expands into an orthogonal counterfactual sibling group.

The full `50 x 20 x 10` plan is a 10,000-leaf candidate space. A 1,000-call run
uses balanced, deterministic sampling from that space; it must not simply take
the first 1,000 leaves. Seed material is replaceable, but the causal coverage
contract remains invariant.

Generation agents receive explicit contracts for:

- Natural concrete entities and values; no `company name`, `customer name`, or
  similar spoken placeholders.
- Diverse openings and endings; no repeated introductions or goodbye loops.
- Model-driven completion and end-call tool intent, not deterministic sign-off
  text.
- Realistic pause, overlap, interruption, cancellation, and recovery timing.
- Multiple turn lengths and conversation lengths.
- Cooperation, conditional compliance, skepticism, resistance, refusal,
  clarification, correction, escalation, handoff, and casual discussion.
- Typed control revisions available before each target response.
- No target-response leakage into generation inputs used by PersonaPlex.

## 6. Split design

All siblings and all pair derivations from a group remain in one split. Evaluation
must include separate suites:

- Unseen groups with familiar operators and topic families.
- Unseen topic families with familiar operators.
- Unseen lexical/entity realizations.
- Unseen operator compositions built from trained atomic operators.
- Revision, stale-control, interruption, and cancellation stress suites.

The primary promotion metric is full-group causal correctness, not mean margin.
Both directions of every required contrast must pass. Reporting must keep
held-out and train-sample metrics in explicit namespaces.

## 7. Admission gates

The structural gate MUST verify without inspecting target text:

- Pair/group identifiers are unique and groups do not cross splits.
- Every expected semantic axis meets pair and distinct-premise quotas.
- Changed paths belong to declared schema families.
- Repeated signature support meets the contract-defined fraction.
- Composite intervention fraction stays below the contract-defined maximum.
- Barge-in and recovery quotas are met with actual timing artifacts.
- Branch labels and target text were not serialized into the model input.
- Voice references have acceptable provenance and rights metadata.

The audio gate then verifies:

- Chatterbox Turbo or another explicitly approved renderer produced the audio.
- Whisper or an approved ASR model measures transcript agreement, word timing,
  cutoff integrity, and recovery alignment.
- Severe ASR failures are rejected; marginal failures may be locally repaired
  without regenerating an otherwise valid conversation.
- Codec, channel, sample-rate, clipping, silence, overlap, and duration checks
  pass.

ASR is an admission measurement, not a CPU fallback for generation.

## 8. Training redesign

Training remains staged:

1. Freeze PersonaPlex and train the typed control encoder/adapter on atomic
   semantic interventions.
2. Train bounded temporal or per-layer conditioning only after held-out atomic
   generalization improves.
3. Introduce delivery-only and turn-taking-only operators with family-specific
   losses.
4. Introduce a bounded composite curriculum.
5. Use control dropout, null-control, stale-control, and superseding-revision
   negatives.

Each batch should include multiple sibling groups and same-operator examples from
different topics. Objectives must include:

- Agent-only native speech/text token likelihood.
- Full-sibling minimum-margin coverage.
- Within-prefix counterfactual ranking.
- Cross-topic same-operator consistency.
- Null/stale-control rejection.
- Semantic invariance for delivery-only contrasts.
- Semantic-state invariance for turn-taking-only contrasts until a new revision
  arrives.

Training promotion requires held-out improvement. Rising train sensitivity with
flat held-out full-group sensitivity is an automatic stop condition.

## 9. Runtime contract

The runtime control path remains:

```text
ASR / tools / policy / call-state reducer
  -> typed control.update revision N
  -> validate and reject stale revisions
  -> encode and cache learned control representation N
  -> snapshot revision N at agent-turn start
  -> PersonaPlex speech-token generation conditioned on duplex context + N
```

On barge-in, outgoing audio and its generation identifier are cancelled. A later
revision cannot retroactively alter emitted audio. The next generation snapshots
the newest acknowledged revision. If no valid current control exists, the model
may wait or emit a validated safe backchannel; it may not make policy-sensitive
claims from stale state.

Exact wording remains a separate guarantee and routes to the strict renderer.

## 10. Reliability claim

No teacher-forced score is a live reliability claim. Public promotion requires:

- Generated native duplex evaluation.
- Semantic/factual/tool-result adherence.
- Revision and stale-control behavior.
- Actual interruption cancellation and recovery.
- First-audio latency and sustained streaming timing.
- Voice preservation and codec quality.
- Live Twilio transport behavior.

The final test plan must predeclare sample sizes and use confidence intervals.
The aggregate and safety-critical strata must meet the predeclared 95-percent
reliability criterion; post-hoc selection of favorable topics or axes is not
allowed.
