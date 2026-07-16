# Diverse Controlled Conversation Synthesis Cascade Contract

**Status:** Normative implementation contract

**Audience:** An agent or engineering team building, extending, operating, or auditing
the diverse synthetic corpus pipeline for semantically controllable PersonaPlex.

**Authority:** This contract extends the requirements in `ARCHITECTURE.md`,
`TRAINING.md`, `RUNTIME_CONTROL.md`, `CERTIFICATION.md`, and `COVERAGE_SPEC.md`.
Where a conflict exists, the safety, causal-control, and promotion invariants in those
documents win.

## 1. Objective

Build a reproducible generator that takes a small, mutable set of seed ideas and
expands them into a large, auditable lattice of realistic duplex conversations. The
generator must select a balanced subset of that lattice, materialize counterfactual
branches, render consented/provenance-approved voices, and produce native-training
inputs in which a typed semantic/evidence state *causally precedes* each supervised
agent response.

The desired operator interaction is intentionally simple:

```text
Build a diverse controlled corpus from these seed ideas, these topic constraints,
these coverage targets, and these approved voices.
```

The implementation behind that request is not a single prompt. It is a staged,
versioned cascade with independent generation, audio, semantic, lineage, codec, and
training gates.

The corpus trains a hybrid system with separate responsibilities:

```text
caller audio -> ASR / state tree -> semantic controller / tools
                                    -> typed revisioned control + delayed evidence
                                    -> PersonaPlex control/evidence adapter
                                    -> next agent speech tokens -> live audio
```

PersonaPlex remains the low-latency conversational body. The semantic controller owns
facts, uncertainty, tool results, policy limits, goals, and updates. A strict renderer
remains the only route for exact-wording guarantees.

## 2. Non-negotiable invariants

1. **No target leakage.** The canonical target response, target transcript, and target
   audio are labels. They must not appear in a control plan, evidence frame, scenario
   contract, planner prompt, or adapter input available before target generation.
2. **Causal state.** Each control revision and evidence update is available before the
   corresponding agent target turn. Future facts, future turns, and post-call outcomes
   are forbidden from its input.
3. **Typed control, not prompt theater.** The final native adapter consumes structured
   control and delayed-evidence representations. A sidecar natural-language prompt or
   an external LLM writing each utterance is not semantic control.
4. **Counterfactual pairs are atomic.** A pair shares context through a selected pivot,
   changes one legitimate state/evidence factor, and must produce meaningfully different
   valid post-pivot behavior. A lone valid branch is not a trainable pair.
5. **Model-only semantic judgment.** No regular expression, keyword list, or local
   heuristic may decide semantic fidelity, naturalness, policy adherence, or whether a
   response incorporates control. Deterministic schema, hash, timing, split, and audio
   structure checks are permitted; substantive decisions require independent model
   inference and evidence.
6. **Authentic audio evidence.** Chatterbox Turbo is the synthetic renderer for this
   corpus. Every rendered supervised turn must pass Whisper ASR, WER, confidence,
   word-level timing, waveform/codec, and independent semantic scrutiny before source
   promotion.
7. **Real duplex behavior.** The corpus includes normal turns, overlap, barge-in,
   cancellation cutoffs, recovery, clarification, resistance, refusal, handoff, and
   model-driven completion. An `interruption` label without audible timing and a cropped
   target is insufficient.
8. **No fabricated provenance.** Voice references need explicit cloning consent or a
   recorded public-domain dedication, source URL, license, content hash, and immutable
   reference ID. Do not commit reference audio or raw generated calls.
9. **Training promotion is separate.** A source-corpus certificate is not a native-code
   certificate, a trained checkpoint, or a deployment approval.

## 3. The corrected cascade and scale model

The cascade creates a **candidate lattice**, not an instruction to render every leaf.

```text
50 topic families
  x 20 scenario contracts per family
  x 10 trajectory seeds per scenario
  = 10,000 candidate trajectory units

Select 500 units under quotas
  x 2 causal counterfactual branches each
  = 1,000 source conversations / 500 trainable counterfactual groups
```

The unselected candidate units are retained as a content-addressed curriculum pool.
They support later gap filling, evaluation-only scenarios, failure replacement, and new
training rounds without reusing the same introductions, closings, voices, topics, or
control patterns.

Do **not** treat the arithmetic as a requirement to render 10,000 conversations during
the first run. Rendering, ASR, semantic auditing, and native encoding are expensive;
the lattice exists to make selection deliberate and reproducible.

## 4. Input contract: `DiverseCorpusRequestV1`

An operator or parent agent starts a run with a bounded, versioned request. Free-form
seed ideas are allowed, but they are input material, not verified facts or target text.

```json
{
  "schema": "personaplex.diverse-corpus-request.v1",
  "requestId": "uuid",
  "seedRevision": "sha256:...",
  "seedIdeas": [
    "public-transit rider feedback",
    "technical project handoff",
    "neighbors repairing a misunderstanding"
  ],
  "topicConstraints": {
    "include": ["casual", "service", "technical", "interview"],
    "exclude": ["identity verification", "political persuasion", "unconsented impersonation"]
  },
  "coverageTarget": {
    "candidateTopics": 50,
    "scenariosPerTopic": 20,
    "trajectorySeedsPerScenario": 10,
    "selectedCounterfactualGroups": 500,
    "branchesPerGroup": 2
  },
  "allowedVoicesManifest": "sha256:...",
  "renderer": "voicebox_chatterbox_turbo",
  "asr": "whisper",
  "allowedPhysicalCudaDevices": [0, 1, 2],
  "requestedAdditions": ["more technical repair", "more brief casual exchanges"],
  "prohibitedContentPolicyRevision": "policy-revision-id"
}
```

The request is invalid if it asks for an unsupported renderer, an unapproved voice,
physical CUDA device `3`, non-paired final coverage, target-text injection, or content
outside the program's governance limits.

`seedRevision`, plan hashes, source code revision, approved-voice manifest hash,
model IDs, inference configuration, and schema revisions must be persisted in the run
card. An agent may append new seed ideas only by creating a new immutable request
revision. It must never silently mutate a plan already being rendered or certified.

## 5. Layer A: topic-family compiler

### Purpose

Create 50 broad, mutually useful topic families. A topic is a **domain of interaction**,
not a single named entity, organization, script, or fact claim. This prevents collapse
into appointment scheduling, introductions, or call-center language.

### Required output: `TopicCardV1`

```json
{
  "topicId": "technical_maker_support",
  "seedRevision": "sha256:...",
  "domain": "technical maker support",
  "interactionModes": ["technical_troubleshooting", "collaborative_planning"],
  "registerRange": ["casual", "precise", "reflective"],
  "safeStakes": ["diagnosis", "tradeoff", "repair", "clarification"],
  "forbiddenPatterns": ["identity collection", "scripted company greeting"],
  "candidateScenarioBudget": 20,
  "diversityTags": ["practical", "uncertainty", "evidence-seeking"]
}
```

### Rules

- Generate topic cards with a planner model under a raw JSON contract and independently
  validate the output schema.
- Maintain a taxonomy ledger with the current 50 topic IDs, input seed provenance,
  interaction modes, and coverage counts.
- Use deterministic quota selection for coverage bookkeeping only. It is not a semantic
  quality judge.
- Reuse of a topic family is expected; repeated exact topics, openings, and
  introductions are not.

### Existing anchors

- Voryn's existing broad domain/mode inventory is in
  `voryn/lib/syntheticConversations.js` (`CONVERSATION_MODES` and
  `CONTROL_TRAJECTORIES`).
- The current starter taxonomy is in
  `voryn/scripts/plan-personaplex-1000-corpus.js` (`TOPIC_FAMILIES`).
- The fine-tuning coverage requirements are in `COVERAGE_SPEC.md`.

## 6. Layer B: scenario-contract compiler

### Purpose

For each topic card, produce 20 scenario contracts. A scenario provides enough stable
context to make a dialogue coherent without deciding what either speaker will say.

### Required output: `ScenarioContractV1`

```json
{
  "scenarioId": "technical_maker_support_network_diagnosis_07",
  "topicId": "technical_maker_support",
  "mode": "technical_troubleshooting",
  "premise": "A home network becomes unreliable after a recent equipment change.",
  "participants": [
    {"role": "caller", "knowledge": "observations and local constraints"},
    {"role": "agent", "knowledge": "safe diagnostic process, not hidden device state"}
  ],
  "startingState": {
    "knownFacts": ["intermittent dropouts began after a change"],
    "uncertainty": ["root cause", "which device is responsible"],
    "policyConstraints": ["do_not_claim_remote_verification"]
  },
  "interactionOpportunity": ["clarification", "evidence_request", "repair"],
  "allowedToolClasses": ["diagnostic_observation"],
  "disallowedClaims": ["personal identity", "account credentials"],
  "scenarioOutcomeSpace": ["safe next test", "bounded handoff", "uncertainty retained"],
  "requiredControlPhenomena": ["tool_availability_change", "uncertainty_boundary"]
}
```

### Rules

- A scenario must be self-contained, safe, and capable of supporting more than one
  natural trajectory.
- Never put a canonical response, a desired exact phrase, or fabricated private data in
  this contract.
- Treat named values in a scenario as invented non-identifying props unless a governed
  tool/evidence source proves them. Never generate realistic identifiers, contact data,
  credentials, or impersonation cues.
- Require scenario diversity across domains, social stakes, interaction modes, register,
  uncertainty, policy route, and tool/evidence class.

## 7. Layer C: trajectory-seed compiler

### Purpose

Create 10 distinct full-conversation blueprints for each scenario. A trajectory seed is
not a transcript. It specifies the interaction shape that the turn-level realizer must
honor while producing novel wording.

### Required output: `TrajectorySeedV1`

```json
{
  "trajectoryId": "technical_maker_support_network_diagnosis_07:t04",
  "scenarioId": "technical_maker_support_network_diagnosis_07",
  "conversationLength": {"targetTurns": 8, "min": 6, "max": 12},
  "pace": "natural_conversational",
  "openingStyle": "continuation",
  "closingStyle": "next_step",
  "voicePairPolicy": "distinct_approved_references",
  "interactionArc": [
    "initial_observation",
    "clarifying_question",
    "resistance_to_complexity",
    "simplified_test",
    "evidence_update",
    "recovery_or_handoff",
    "model_decides_terminal_action"
  ],
  "duplexEvents": [
    {"kind": "barge_in", "afterTurnOrdinal": 3, "cutoffRangeMs": [600, 1800]},
    {"kind": "repair", "afterTurnOrdinal": 4}
  ],
  "postureArc": ["neutral", "skeptical", "conditional_compliance", "resolved_or_deferred"],
  "counterfactualPivotOrdinal": 5,
  "controlPhenomena": ["tool_result", "freshness", "uncertainty", "next_goal"]
}
```

### Required variety

Across the candidate lattice and final sample, deliberately cover:

- brief, balanced, and extended calls;
- casual affiliation, interviews, technical work, care, community, creative review,
  planning, service recovery, conflict repair, learning, research, and administration;
- cooperation, conditional compliance, skepticism, resistance, clarification, refusal,
  correction, escalation, handoff, recovery, and model-driven completion;
- rapid reciprocal, normal conversational, reflective, interview-probe, collaborative,
  and repair/restatement pacing;
- no-greeting continuations, direct requests, callbacks, corrections, handoffs, and
  time-sensitive updates;
- diverse approved voice pairs and no repeated voice-pair/introduction/closing pattern
  beyond an explicit quota;
- natural interruptions with audible overlap and cancellation, not just a metadata flag.

The trajectory compiler must prevent a common degenerate result: every call beginning
with a generic introduction and ending with repeated reciprocal goodbyes. Completion is
an LLM-selected private `end_call` action grounded in the state, not a deterministic
sign-off detector.

### Existing anchors

- Timing, interruption, private `end_call`, ASR, rendering, and turn loop:
  `voryn/lib/agentVsAgentSim.js`.
- Typed synthesis settings, control trajectories, control-frame generation, and
  Chatterbox selection: `voryn/lib/syntheticConversations.js`.
- Existing opening/closing sets and initial 1,000-call plan:
  `voryn/scripts/plan-personaplex-1000-corpus.js`.

## 8. Layer D: counterfactual control and Moshirag evidence compiler

### Purpose

Turn each selected trajectory seed into a pair of causal branches. This is the bridge
between ordinary conversational diversity and a model that can follow a mutable control
plane.

### Pair construction

1. Materialize common pre-pivot dialogue, state, and duplex context.
2. Select one pivot at which a legitimate new state/evidence update arrives.
3. Produce two branches that vary **exactly one** defined factor.
4. Generate post-pivot target turns independently for each branch.
5. Require an independent semantic auditor to establish that the change mattered and
   that both outputs remain grounded in their respective states.

Allowed one-field pivot changes include:

- tool status: `pending` versus `ready`, or `ready` versus `failed`;
- a verified fact changing or expiring;
- policy availability changing;
- caller posture changing after an interruption;
- a handoff/ownership change;
- a new uncertainty boundary;
- cancellation of a stale planned output after barge-in.

Never manufacture a contrast by embedding different desired wording in the control
frame. The target response differs because the allowed facts, goals, constraints, or
freshness differ.

### Required control frame: `ControlTrainingFrame`

The frame follows `schemas/control_training_frame.schema.json` and
`training/contracts.py`. It must contain a compact declarative state, revision,
effective boundary, context hash, known facts, uncertainty, policy constraints, tool
result references, caller posture, next goal, and style controls. It must reject all
canonical-response fields.

Example shape:

```json
{
  "callId": "synthetic-call-id",
  "revision": 12,
  "effectiveFrom": "next_agent_turn",
  "contextHash": "sha256:...",
  "intent": "resolve_delivery_issue",
  "knownFacts": ["replacement shipped July 14"],
  "uncertainty": ["carrier scan has not arrived"],
  "constraints": ["do_not_invent_delivery_date", "do_not_repeat_greeting"],
  "toolResultRefs": ["shipment:replacement-queued"],
  "callerPosture": "skeptical",
  "nextGoal": "acknowledge delay and offer escalation options",
  "style": {"warmth": 0.7, "assertiveness": 0.35, "brevity": 0.55}
}
```

### Required delayed-evidence frame: `EvidenceTrainingFrame`

Every selected target turn also carries an evidence frame following
`schemas/evidence_training_frame.schema.json`. It records the event that produced the
evidence, the supporting control revision, availability and expiry times, provenance,
allowed claims, context hash, branch identity, and the target-turn boundary. The later
native encoder maps this representation to the Moshirag-style delayed evidence stream;
it is not merely logged metadata.

The intended learned path is:

```text
typed control/evidence frame
  -> frozen text/control encoder
  -> delayed streaming evidence representation
  -> trainable prefix / evidence adapter
  -> PersonaPlex transformer conditioning
  -> next agent speech tokens
```

Moshirag alignment requirements:

- Evidence is available before the generated target turn and consumed causally.
- The encoder output is aligned to native delayed duplex code time, not repeated across
  audio frames by a heuristic.
- Control/evidence revision changes take effect only at an acknowledged turn boundary.
- If a barge-in supersedes a response, queued media and its generation ID are cancelled;
  the next response consumes the newer revision.
- Train with control/evidence dropout so normal conversation remains viable with sparse
  state.
- Include no-control, stale-control, update-rejected, policy-change, tool-result,
  interruption, and correction cases in both training and held-out evaluation.

### Existing anchors

- V8 paired-plan derivation: `voryn/scripts/derive-personaplex-v7-counterfactual-plan.js`.
- V8 lane generation and cross-lane accepted-pair lookup:
  `voryn/scripts/run-personaplex-v7-paired-lane.js`.
- Voryn certification and pair auditing:
  `voryn/scripts/certify-personaplex-synthetic-dataset.js` and
  `voryn/scripts/certify-personaplex-v7-paired-queue.js`.
- Runtime revision/evidence semantics: `personaplex_control/runtime.py` and
  `personaplex_control/controlled_server.py`.
- Moshirag reconciliation and native conditioning plan:
  `MOSHI_RESEARCH_AND_CONTROL_RECONCILIATION.md`,
  `MOSHIRAG_RUNTIME_PORT_AND_TRAINING.md`, and
  `SEMANTIC_CONTROL_CONVERGENCE_PLAN.md`.
- Evidence schema, adapter, and training path:
  `schemas/evidence_training_frame.schema.json`,
  `training/evidence_conditioning.py`, and `tools/train_evidence_stream.py`.

## 9. Layer E: turn-level dialogue realization

### Purpose

Generate actual spoken turns from the scenario, trajectory, audible prior context, and
the current typed frame. The realizer writes target labels, but those labels remain
separate from inputs used for later native training.

### Required behavior

- Use the configured dialogue model with reasoning disabled and a strict raw-JSON
  response contract containing `action`, spoken `text`, and non-spoken
  `internal_reason`.
- Generate one turn at a time. Recompute conversation state after each caller turn,
  typed control update, tool result, or interruption.
- Give a caller barge-in only the audible prefix of the interrupted agent turn.
- Treat pending tool state, expired evidence, and unacknowledged control as reasons to
  wait or give a safe backchannel, not permission to use stale sensitive state.
- Let the model choose `end_call` when the trajectory is naturally complete. Never add
  deterministic goodbye text or deterministic termination based only on a phrase.
- Prohibit generic placeholders such as `company name`, `agent name`, or bracketed
  fill-ins. A model-only repair attempt may replace malformed dialogue; unresolved
  output fails the candidate.

### Existing anchors

- Voryn dialogue orchestration and model provider path:
  `voryn/lib/syntheticConversations.js` and
  `voryn/lib/enterpriseInferenceProviders.js`.
- Turn simulator and live-like timing model: `voryn/lib/agentVsAgentSim.js`.

## 10. Layer F: audio, ASR, and authentic timing

### Required renderer path

1. Assign two distinct approved voice-reference IDs from the immutable reference bank.
2. Render each turn with `voicebox_chatterbox_turbo`.
3. Persist audio hash, renderer configuration, reference ID, and duration.
4. Run Whisper ASR on every rendered supervised turn.
5. Persist transcript, WER, confidence, segments, and word timings.
6. Derive actual audible start/end, overlap, cutoff, and cancellation timing from the
   rendered artifacts, not only planning intent.
7. Reject failed audio before semantic certification.

Current calibrated source thresholds are WER at most `0.25`, confidence at least
`0.45`, and nonempty word-level Whisper alignment. They may change only through a
versioned held-out error analysis; throughput alone is not a reason to relax them.

The reference bank must be immutable during a run. Its entries are loaded and verified
by `voryn/lib/voiceReferenceBank.js`; current audit/manifest tooling is in
`voryn/scripts/audit-personaplex-voice-references.js` and
`voryn/scripts/build-certified-personaplex-reference-manifest.js`.

## 11. Layer G: independent certification and retry rules

### Certification sequence

1. **Structural admission:** schema, hashes, required fields, legal voice reference,
   audio existence, timing shape, and branch lineage are checked deterministically.
2. **Audio admission:** Whisper WER/confidence/alignment and waveform/codec checks pass.
3. **Semantic turn audit:** an independent model judges control use, factual grounding,
   policy adherence, naturalness, placeholder absence, and semantic ASR fidelity.
4. **Counterfactual-pair audit:** an independent model confirms both branches share the
   correct pivot context, differ at the specified state/evidence factor, and materially
   diverge after the pivot without target leakage.
5. **Corpus audit:** accepted records receive immutable lineage, group-isolated split
   assignment, source manifest, and certificate.

### Failure handling

- A true semantic failure, negation inversion, placeholder, timing violation, missing
  word alignment, source-provenance failure, or branch mismatch rejects the candidate.
- A transport failure or malformed structured model response may use bounded
  **model-only** JSON repair/retry. It is not converted to a pass without an auditor.
- Exhausted retries leave the candidate unresolved/rejected and select a replacement
  candidate from the lattice. Do not weaken a gate to fill a quota.
- A failed pair receives a typed repair packet. A control agent may regenerate only the
  earliest invalid **post-pivot** suffix from its preserved causal snapshot. Pre-pivot
  defects require pair replacement because the exact shared prefix is no longer valid.
- A bounded repair budget prevents one scenario from consuming unbounded GPU time. Once
  exhausted, the coordinator selects a replacement trajectory from the candidate lattice
  and retains the failure packet for audit and future generator improvement.
- Keep failure reason, model/version, revision chain, and replacement lineage in the
  run ledger. Do not retain raw sensitive tool payloads on the audio plane.

The current Voryn independent source certifier is
`voryn/scripts/certify-personaplex-synthetic-dataset.js`. Its queue wrapper only
requeues transport-unavailable certificates; it does not reclassify substantive
semantic failures as passes.

## 12. Layer H: selection, diversity, and split policy

The selection coordinator chooses `500` candidate trajectory units from the 10,000-unit
lattice. Selection is deterministic from the request/plan hashes and a documented
quota table. It must balance at least:

- topic family and scenario;
- interaction mode and trajectory arc;
- call length and pace;
- voice pair and reference reuse;
- opening and terminal style;
- interruption/cancellation/recovery type;
- control source, update type, evidence availability, and counterfactual pivot field;
- caller posture and agent delivery style;
- policy-constrained versus ordinary resolution paths.

Assign train/validation/test partitions by **counterfactual group**, never individual
turn or branch. Topic, voice-pair, and scenario holdouts must be represented in the
evaluation design so an adapter cannot pass by memorizing a stock dialogue pattern.

The coordinator may use deterministic identifiers and quotas to prevent duplicate
allocation. It must not use lexical similarity or rules as a substitute for model-based
semantic diversity or certification.

## 13. Layer I: export, native encoding, and training handoff

Only corpus-certified Voryn records proceed to the native pipeline.

```text
certified paired source records
  -> controlled duplex export
  -> timeline / lineage validation
  -> native Mimi/Moshi code encoding
  -> tensor-level certification
  -> frozen-base semantic-prefix training
  -> delayed-evidence adapter training
  -> held-out counterfactual and live-like evaluation
```

Required implementation anchors:

- `tools/export_controlled_duplex_dataset.py`
- `tools/validate_controlled_duplex_dataset.py`
- `tools/export_v7_evidence_frames.py`
- `tools/encode_controlled_native_adapter_tensors.py`
- `tools/certify_controlled_native_corpus.py`
- `tools/run_controlled_native_pipeline.py`
- `tools/train_semantic_prefix.py`
- `tools/train_evidence_stream.py`

The native pipeline must discover and record the actual base-model stream layout,
delays, codec contract, and target masks. It is forbidden to stretch text labels over
frames or assume a fixed codebook count. Training starts with the PersonaPlex base
frozen and learns only the compact control/prefix adapter. Evidence-stream adaptation
is a second gated stage, not an unverified simultaneous modification of the base model.

## 14. Agent orchestration protocol

An implementation agent should delegate by artifact boundary, not by vague roles:

| Agent stage | Input | Required artifact | Cannot do |
| --- | --- | --- | --- |
| Taxonomy compiler | request + seeds | 50 `TopicCardV1` records | write dialogue targets |
| Scenario compiler | topic card | 20 `ScenarioContractV1` records/topic | invent target labels |
| Trajectory compiler | scenario contract | 10 `TrajectorySeedV1` records/scenario | certify semantics |
| Pair compiler | selected seed | two causal branch specs + pivot | reuse a target label |
| Dialogue realizer | branch state + audible history | provisional turn records | certify itself |
| Audio worker | provisional text + approved ref | audio/ASR/timing evidence | alter semantic state |
| Semantic auditor | candidate + prior state + evidence | independent verdict | generate replacement dialogue |
| Corpus certifier | audited branch pair | source certificate | encode native tensors |
| Native pipeline | certified source | tensor certificate + train manifest | promote a checkpoint |
| Evaluator | checkpoint + held-out suite | metrics/report | self-approve deployment |

Every handoff must use a schema-versioned file or message with content hashes. A
sub-agent may request bounded model-only repair from its predecessor, but it may not
silently rewrite an artifact owned by another stage.

## 15. Implementation sequence for the current codebase

1. Preserve the existing V8 paired plan as the current generation baseline; do not mix
   older V5 pilot records into its final corpus.
2. Introduce `DiverseCorpusRequestV1`, `TopicCardV1`, `ScenarioContractV1`, and
   `TrajectorySeedV1` schemas and a content-addressed plan ledger.
3. Refactor the current fixed topic list in
   `voryn/scripts/plan-personaplex-1000-corpus.js` into the three-layer cascade while
   retaining its existing safety and coverage vocabulary as seed material.
4. Teach `voryn/scripts/derive-personaplex-v7-counterfactual-plan.js` to select exactly
   500 quota-balanced trajectory units and emit two branches per selected unit.
   The interim bridge is `tools/compile_diverse_cascade_voryn_plan.py`: it model-compiles
   selected pair specifications to the existing V8 lane-plan shape while pinning branch
   lineage, approved voice pairs, pivot identity, and evidence-update references.
5. Keep `voryn/lib/syntheticConversations.js` as the turn-level realizer, but require
   it to persist scenario/trajectory IDs, control revision chain, evidence availability,
   pair pivot, and lineage hashes on every record.
6. Retain the current Chatterbox/Whisper gates and model-only repair semantics. Add
   aggregate calibration reports; do not lower gates without held-out evidence.
7. Expand source certification to require the new cascade lineage and pair-pivot proof.
8. Export the certified `ControlTrainingFrame` and `EvidenceTrainingFrame` objects into
   the controlled native dataset and make them mandatory for adapter batches.
9. Train/evaluate the frozen prefix adapter before enabling evidence injection, then
   train/evaluate the delayed evidence adapter with no-control and stale-control
   ablations.
10. Update this contract, the TODO ledger, and run cards whenever schemas, gates,
    available models, or artifact locations change.

## 16. Completion criteria

The phrase "construct a diverse controlled corpus" is complete only when all of the
following are true:

- A request revision deterministically produces a 10,000-unit candidate lattice with
  the required topic, scenario, and trajectory contracts.
- The coordinator selects 500 quota-balanced units and produces exactly 1,000 valid
  counterfactual branch conversations.
- Every promoted turn has approved voice provenance, Chatterbox audio, Whisper evidence,
  auditable timing, typed pre-turn control, causal delayed evidence, and no target leak.
- Every promoted counterfactual group has both branches, one defined pivot, and an
  independent semantic proof of material post-pivot difference.
- The corpus has immutable group-isolated train/validation/test assignment, collision
  and diversity reports, and a passed source certificate.
- Native encoding produces a `certified_for_adapter_training` tensor certificate for the
  exact model/codec revision.
- A frozen-base prefix checkpoint and then an evidence-stream checkpoint pass held-out
  semantic adherence, state freshness, interruption, latency, codec, and preservation
  gates against no-control and shuffled-control baselines.
- The runtime applies only acknowledged revisions at turn boundaries and cancels stale
  output after barge-in. Exact wording still routes to the strict renderer.

Until those conditions are met, the system is an in-progress corpus/training framework,
not a production semantically controllable PersonaPlex deployment.
