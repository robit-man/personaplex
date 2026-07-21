# Semantic Control v5 Architecture and Data Contract

Status: normative architecture; implementation state is recorded per section
Scope: authentic planning through live PersonaPlex promotion

## 1. Objective

PersonaPlex remains the low-latency, full-duplex conversational body. A deeper
semantic system maintains mutable call state and emits typed, versioned control
updates. The trained PersonaPlex receiver must causally use those updates while
preserving native speech timing, voice, overlap, and turn-taking.

The control frame is a model input. It is not a sidecar prompt writer, a target
response, or metadata stored only for observability.

```text
caller audio
  -> streaming ASR, tools, policy, and state reducer
  -> typed control.update revision N
  -> target-free control encoder
  -> native 12.5 Hz streaming-sum condition
  -> PersonaPlex temporal/text receiver
  -> agent speech tokens and live audio
```

## 2. End-to-end artifact flow

```text
50 bound seed domains
  -> 1 topic card per seed
  -> 1 joint authentic 20-slot diversification blueprint per topic
  -> 1 full scenario expansion bound to each blueprint slot
  -> whole-topic independent clustered-findings scrutiny
  -> targeted repair of only rejected blueprint slots
  -> 10 compact authentic trajectory candidates per scenario
  -> balanced selection of 250 primary + 250 reserve candidates
  -> full typed expansion of those 500 candidates
  -> four causal siblings per active primary group
  -> schema-7 Voryn plans with immutable typed control programs
  -> render one exact duplex prefix and replay it across four suffixes
  -> certify real timing, ASR, audio, control, and model-selected end_call
  -> immutable post-render branch artifacts
  -> precodec and native delayed-duplex tensors
  -> leakage-component train/validation/test splits
  -> native agent-only SFT + 4x4 listwise causal training
  -> generated native evaluation
  -> revision/cancellation runtime integration
  -> live Twilio evaluation
```

## 3. Authentic schema-constrained planning

Implementation state: implemented and exercised by the current topic/scenario
generation run.

`training/diverse_cascade.py` sends a strict JSON Schema through the
OpenAI-compatible `response_format` field. Semantic content comes from model
inference. Deterministic code supplies only identities, ordinals, required
operator assignments, balancing, hashes, and validation.

The parser accepts one JSON object. It has no prose extraction, regex salvage,
field coercion, placeholder substitution, or heuristic semantic repair. A
reachable endpoint returning malformed protocol or semantic output is treated
as authoritative failure. Endpoint failover is limited to transport and
explicitly retriable HTTP failures.

When an artifact fails structural validation, only that assigned artifact is
regenerated. A retry receives the prior validation error as a typed repair
constraint, but no rejected content is admitted and no target response is
invented by deterministic code.

This follows the official Ollama structured-output interface, which supports a
JSON Schema in `format` and the OpenAI-compatible API through
`response_format`. See [REFERENCES.md](REFERENCES.md).

## 4. Atomic persistence and resumability

Implementation state: implemented and live-exercised.

`tools/build_diverse_synthesis_cascade.py` stores every accepted model artifact
under:

```text
<run-root>/.stage_checkpoints/topics/<content-addressed-id>.json
<run-root>/.stage_checkpoints/scenarios/<content-addressed-id>.json
<run-root>/.stage_checkpoints/trajectories/<content-addressed-id>.json
```

Checkpoint identities and configured unique fields are immutable. Reusing an
identity with different content fails closed. `--resume` reloads accepted
artifacts and requests only missing identities. A stage-level interruption must
not discard already accepted artifacts.

Process state, PID values, checkpoint counts, and GPU utilization are operational
telemetry, not architecture truth. They must be timestamped when reported and
must be refreshed from the host before being described as current. A stage is
complete only when its canonical artifact, expected cardinality, run manifest,
and content hashes agree. A running process does not establish completion, and a
momentarily absent process does not invalidate durable checkpoints.

The unit of planner retry is one topic, scenario, trajectory candidate, or
group specification. The unit of audio retry is one four-sibling group suffix
set anchored to its prefix source. Broad regeneration is not an acceptable
substitute for localized repair.

## 5. The corrected 50 x 20 x 10 cascade

### 5.1 Topic layer

Implementation state: complete for the current pilot.

The content-addressed v2 catalog has 50 distinct seed domains. Every topic card
copies only the assigned seed identity and three factorized causal affordances:
semantic, delivery, and turn-taking. Topic cards contain no dialogue target.

### 5.2 Scenario layer

Implementation state: the initial 1,000-scenario corpus is structurally complete
but quarantined and not training eligible.

Each topic produces 20 concise scenario contracts, for 1,000 scenario premises.
The contracts define participants, starting state, uncertainty, stakes, tool
classes, policy boundaries, outcome space, and required control phenomena. They
must not contain canonical agent wording.

The 20 structural diversifiers cover cooperation, skepticism, resistance,
correction, ambiguity, confirming and disconfirming tools, policy change,
superseding evidence, interruption, repair, refusal, handoff, conditional
compliance, casual exchange, technical explanation, service recovery,
multi-step decisions, accessibility, and model-selected completion.

Structural validity did not establish semantic diversity. The independent Qwen
v4 clustered-findings audit rejected `918/1000`, or `91.8%`, primarily for
semantic mode collapse. Its immutable report is:

```text
/srv/voxrn_cache/personaplex/training/cascade-v5-pilot-20260720/scenario_stage_rejection.v1.json
reportId: sha256:c02f53487d795b213ad87078f1ea133f912b149bbdbc7886a7d20af4dc9755c1
```

The entire initial scenario corpus is quarantined. The 82 individually
non-rejected rows are not extracted as a favorable subset because their
distribution was generated inside a failed topic-level process. The 55 blind
repair candidates are discarded because unbound repair does not restore the
missing joint diversity contract.

### 5.3 Replacement 20-slot scenario blueprint

Implementation state: architecture specified; implementation in progress;
replacement inference and certification pending.

The replacement unit is one complete topic, not one independent scenario. Its
pipeline is:

1. Generate one joint authentic diversification blueprint containing exactly 20
   contrastive slots for the topic.
2. Bind every slot to a required immutable `scenarioSlotId` and blueprint hash.
3. Expand each slot independently into one full scenario while holding its slot
   assignment fixed through strict schema constants.
4. Submit all 20 expanded scenarios together to an independent whole-topic
   clustered-findings audit.
5. Repair only specifically rejected slots against the original blueprint and
   the full accepted sibling set.
6. Repeat whole-topic scrutiny after repair; never certify a repaired slot in
   isolation.

The joint blueprint must make contrast explicit before verbose expansion. Each
slot declares a distinct interaction mode, material premise, participant
relationship, conversational objective, uncertainty or evidence shape, causal
opportunity, outcome route, and duplex behavior. It contains no target dialogue
and no full scenario prose.

Per-slot expansion receives only the topic contract, complete blueprint,
assigned slot, immutable identities, and scenario schema. It may elaborate the
assigned semantics but may not substitute a more familiar scheduling, support,
verification, or transactional template.

Whole-topic scrutiny is an admission gate, not a repair generator. It compares
all 20 rows for semantic clusters, repeated event skeletons, lexical disguise,
duplicate causal opportunities, and collapsed outcome routes. Its findings must
name rejected slot IDs and reasons. Targeted repair receives those findings and
regenerates only the named slots while preserving every accepted slot and the
original blueprint identities.

### 5.4 Compact ten-way trajectory candidate fan-out

Implementation state: required but not implemented.

The efficient v5 design requires one schema-constrained inference per scenario
that returns exactly ten compact candidate descriptors. Every descriptor must
be authentic model output and must include enough typed state to support
selection without containing target dialogue. The compact descriptor includes:

| Dimension | Required content |
| --- | --- |
| Identity | Scenario-bound candidate ordinal and immutable lineage. |
| Causal operator | One intervention family, axis, changed path, and typed transition. |
| Interaction | Posture transition, evidence source, outcome route, and duplex event type. |
| Shape | Length band, turn count band, style, opening position, and termination opportunity. |
| Timing intent | Planned update, interruption, cancellation, overlap, and recovery events. |
| Safety | Tool boundaries, prohibited claims, and strict-renderer requirement. |

The current `plan_trajectories()` implementation instead performs ten
sequential full trajectory-generation calls per scenario. Review of the first
replacement proposal found two additional blockers:

- Its positional array schema uses `prefixItems`, which the active structured
  output path does not support.
- Its proposed response can require roughly 12,000 output tokens while the
  active planner contract allows roughly 4,000.

The corrected Stage A response is an object with ten schema-required candidate
ID properties rather than a positional `prefixItems` array. Every property holds
a genuinely compact descriptor that fits the aggregate 4,000-token contract.
Verbose typed state, timing schedules, and scenario prose belong only in Stage B
after balanced selection. The redesign must be measured against the actual
serialized schema and token budget before inference starts.

### 5.5 Selection and full expansion

Implementation state: selector implemented; two-phase expansion pending.

The 10,000 compact candidates are the selection universe. The deterministic
typed balancing algorithm selects exactly 250 primary and 250 reserve groups
across causal axis, intervention family, posture, evidence source, duplex event,
outcome route, length, style, topic, and scenario.

Only those 500 selected descriptors are fully expanded into complete typed
trajectory contracts. Reserve contracts are fully planned so a typed rejection
can replace a primary group without inventing a new distribution after the run
starts. The 250 active primary groups produce 1,000 rendered conversations at
four siblings per group.

## 6. Factorized four-role causal groups

Implementation state: implemented and focused-tested; scale materialization is
pending.

Every active group contains exactly one sibling of each role:

| Role | Required causal meaning |
| --- | --- |
| `verified_positive` | Evidence or policy permits the intended fact or action. |
| `verified_negative` | Evidence or policy contradicts or prohibits it. |
| `uncertain` | The model must wait, clarify, hedge, or use a bounded backchannel. |
| `superseded` | A newer revision, tool result, or interruption invalidates the prior plan. |

The group holds the following fixed through the causal pivot:

- Exact caller and agent duplex prefix.
- Native delayed code streams.
- Speaker and voice references.
- Target-turn boundary.
- Template and lineage identifiers.
- Non-intervened semantic, delivery, and turn-taking fields.

One declared family changes per atomic group. Composite trajectories may be
introduced only after each atomic operator has repeated cross-premise support.
Sibling names and operator IDs are balancing metadata and are not shortcuts
serialized into the model input.

The target text, target audio, target hashes, and post-response facts are labels.
They never enter `commonContext`, the control frame, or the control tensor.

## 7. Compiler and unchanged control consumption

Implementation state: implemented and focused-tested.

`tools/compile_diverse_cascade_voryn_plan.py` emits schema-7 render plans and a
content-addressed shared-prefix sidecar. Each plan contains:

- Immutable group, role, premise, template, lineage, and voice-pair identity.
- A target-free `commonContext` and verified `commonContextHash`.
- One typed control event for every planned agent target.
- Positive control revisions and content hashes.
- A strict-before-target pivot binding.
- Actual interruption, cancellation, recovery, and termination intentions.
- A target-free `postRenderBridge` and `renderPlanId`.

Voryn must consume the compiler's typed control frames unchanged. It may append
observed timing, ASR, codec, and render evidence. It may not rewrite control
meaning, infer missing target wording, or silently synthesize a replacement
frame.

## 8. Render once and replay the exact duplex prefix

Implementation state: implemented and focused-tested in Voryn.

Voryn `scripts/run-personaplex-synthetic-lane.js` assigns a complete four-role
group to one physical CUDA lane. Voryn
`lib/personaplexSyntheticGroupLane.js` performs this transaction:

1. Render the first sibling through the pre-pivot boundary.
2. Capture the complete replay snapshot, including ordered records and call state.
3. Deep-freeze and fingerprint the snapshot.
4. Replay the exact snapshot as context-only data for the other three siblings.
5. Generate only each sibling's causally distinct suffix.
6. Verify exact shared-prefix identity and unchanged replay fingerprint.
7. Verify complete branch lineage and one final model-emitted `end_call` action.
8. Atomically commit the group bundle and progress only after all four pass.

Textual equality alone is insufficient. The shared-prefix claim concerns the
actual ordered duplex context supplied to generation. Failed groups remain
unresolved and are retried or replaced as complete causal units.

## 9. Immutable post-render identity chain

Implementation state: implemented and focused-tested; corpus-scale execution is
pending.

`tools/export_controlled_duplex_dataset.py --compiled-plan` joins a certified
Voryn timeline to its schema-7 plan and emits exactly one
`personaplex.voryn-branch-artifact.v5` record per sibling pivot.

The exporter verifies:

- The observed control frame hash and revision match the immutable plan.
- The control availability frame is strictly less than the native pivot frame.
- The common context and shared-prefix identities are unchanged.
- Actual timing, cancellation, overlap, recovery, and model-selected termination
  come from the certified timeline.
- Target wording is absent from the branch artifact.

`planRecordId` hashes the complete immutable post-render branch artifact. It is
propagated through exported examples, precodec provenance,
`control_labels.jsonl`, native materialization, and trainer inputs. A missing or
mismatched identifier fails closed.

## 10. Native materialization and leakage-safe packing

Implementation state: implemented and focused-tested; v5 artifacts have not yet
been materialized at scale.

The native contract is fixed at 24 kHz, 12.5 Hz, and 80 ms per temporal frame.
`tools/materialize_native_causal_groups_v5.py` stores one shared native prefix
per group plus four branch suffixes, masks, control streams, and target-audio
labels. It does not splice mismatched sidecars or recompute Mimi during the
materialization transaction.

The trainer-ready outputs are:

```text
native_causal_groups_v5.jsonl
rerender_rejections_v5.jsonl
materialization_report_v5.json
native_moshirag_groups_v2.jsonl
native_moshirag_test_v2.jsonl
native_moshirag_all_splits_v2.jsonl
native_moshirag_dataset_v2.json
```

`training/causal_group_pack.py` builds union-find leakage components across
group lineage, premise/template identity, operator identity, and voice pair.
Whole connected components enter exactly one of train, validation, or test.
All four siblings remain together. The packer emits:

```text
common_inputs.jsonl
listwise_groups.jsonl
pairwise_diagnostics.jsonl
leakage_components.jsonl
causal_coverage_certificate.json
manifest.json
```

Pairwise rows are diagnostics. The four-role listwise group is the training and
promotion unit.

`manifest.json` is also the mandatory trainer-admission artifact. Its
content-addressed `manifestId` binds every pack output, including the coverage
certificate, plus the immutable source-group inputs. Its `trainerBinding`
binds the exact trainer dataset contract, trainer group manifest, model
contract and revision, and the canonical `groupId/componentId/split`
projection. The packer refuses to write this manifest when the trainer
projection differs from the leakage-component split.

## 11. Native training objective

Implementation state: full-rank CUDA/NCCL trainer implemented and smoke-tested;
v5 training has not started.

The first v5 run freezes ARC, Mimi, audio-code embeddings, the depth
transformer, audio heads, and the existing voice-prompt path. It trains the
complete PersonaPlex temporal and text receiver plus the native control
conditioner at full rank.

Each group supplies one shared prefix and four branch suffixes. The objective
contains:

- Agent-only native text and audible agent-audio likelihood.
- A 4 by 4 matched-versus-counterfactual listwise matrix.
- A pre-response control-state probe.
- Null-control and control-dropout behavior.
- Stale and wrong-branch negatives.

Caller audio and prior-agent audio remain context. They are never supervised as
current agent targets. Control must be available strictly before the first
supervised target frame.

The trainer uses CUDA, NCCL, and FSDP only. CPU model fallback and CPU parameter
offload are forbidden. Machine capacity is discovered at runtime. Physical
devices `0,1,2` are an explicit run policy, not a hardcoded estimate of memory
capacity. Host-memory admission uses `/proc/meminfo` and throttles only above the
configured 80 percent used-memory ceiling.

Before GPU admission or `torchrun`, the trainer requires the certified pack
manifest and recomputes the complete hash chain. It verifies source files,
pack outputs, manifest identity, certificate status and coverage, component
membership, exact split assignments, trainer dataset/group hashes, and the
model-contract hash and revision. The same proof is recorded in run and
checkpoint contracts and must match on resume. Any absent, rejected, changed,
or semantically mismatched artifact fails closed before an epoch can start.

Checkpoints at steps 100, 125, and 150 report train and validation namespaces
separately. The test split is retained for final evaluation and is not used for
checkpoint selection. A 95 percent teacher-forced group/probe gate is only a
training diagnostic.

## 12. Runtime revision and cancellation contract

Implementation state: architecture specified; complete v5 checkpoint/runtime
integration and live proof are pending.

At every valid `control.update`:

1. Validate schema, call identity, and monotonic revision.
2. Reject stale or duplicate revisions.
3. Encode and cache the target-free control representation once.
4. Acknowledge the accepted revision.
5. Snapshot that representation at the next agent-turn boundary.
6. Bind `generation_id` to the acknowledged `control_revision`.

On barge-in or a superseding revision:

1. Invalidate the active generation ID.
2. Stop outgoing audio and clear queued media.
3. Preserve already emitted audio as immutable history.
4. Reduce the new caller/tool/policy state.
5. Start the next response only from the newest acknowledged revision.

The model does not switch semantic guidance halfway through already emitted
audio. If no valid current frame exists, it may wait or emit a separately
validated safe backchannel. It may not make a policy-sensitive claim from stale
state.

## 13. Exact wording fallback

Implementation state: encoded in the v5 request and planning contract; runtime
routing and live validation remain pending.

Semantic control governs facts, goals, posture, constraints, next action, and
delivery. It cannot guarantee exact wording. Any path requiring legally,
contractually, or safety-mandated language must route to the validated strict
renderer. Evaluation must score that route as a routing decision, not as proof
that PersonaPlex generated exact text.

## 14. Promotion gates

Promotion requires all gates in order:

| Gate | Required evidence |
| --- | --- |
| Structural | Schema validity, complete four-role groups, repeated operators, no target leakage, immutable lineage, and leakage-disjoint splits. |
| Audio | Approved voice provenance, Chatterbox Turbo render, Whisper timing/transcript checks, codec/channel/rate checks, overlap and cutoff integrity. |
| Teacher-forced | Held-out full-group direction and probe metrics at preregistered checkpoints. Diagnostic only. |
| Generated native | Free-running semantic adherence, fact/tool incorporation, stale-control rejection, revision correction, cancellation, recovery, voice, codec, and latency. |
| Live Twilio | Real transport, bidirectional streaming, barge-in, queue cancellation, turn latency, tool state, termination, and strict-renderer routing. |
| Statistical | Predeclared sample sizes and confidence intervals; aggregate and safety-critical strata satisfy the 95 percent criterion. |

No gate may be inferred from a later file name, process exit code, or favorable
subset. Failures remain visible and are repaired at the smallest causal unit.
