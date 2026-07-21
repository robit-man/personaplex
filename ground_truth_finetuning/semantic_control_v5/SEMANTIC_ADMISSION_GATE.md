# Semantic Admission Gate

## Purpose

This document is the source of truth for semantic admission of Stage-T taxonomy anchors and
Stage-E scenario contracts. Structural validators remain deterministic. Semantic rejection is
model-driven, typed, independently verified, source-bound, checkpointed, and free of regex or
lexical admission heuristics.

The gate exists to prevent two opposite failures:

- Weak scrutiny admits mode collapse, placeholders, leaked target dialogue, unsafe instructions,
  incoherent state, or semantically duplicate trajectories.
- A noisy open-ended judge repeatedly rejects valid material and causes unbounded repair loops.

## Model roles

The production topology uses distinct model identities:

| Role | Model | CUDA lane |
|---|---|---|
| Taxonomy, blueprint, and scenario generator | `open-agents-qwen36-35b:latest` | 0, 1, 2 |
| Targeted taxonomy repair planner | `huihui_ai/Qwen3.6-abliterated:27b` | dynamically assigned CUDA lane |
| Primary semantic proposer and blueprint judge | `robit/ornith-vision:35b` | 0 and 1 |
| Secondary taxonomy semantic proposer | `robit/ornith:35b` | 2 |
| Evidence-bound final verifier | `gemma3:27b` | 1 |

Reasoning is disabled for every model call. Every response uses strict JSON Schema. Model identity,
endpoint binding, prompt source hash, response schema, source artifact hashes, and protocol version
participate in checkpoint identity.

`robit/ornith-vision:35b` is intentionally a proposer rather than the sole final judge. Live
calibration showed useful recall but unacceptable open-audit precision. Gemma is only asked to
confirm one exact typed claim at a time; it never performs a second open audit.

## Taxonomy admission path

The Stage-T path is:

```text
Qwen raw 20-anchor taxonomy
  -> deterministic schema and identity validation
  -> Ornith Vision typed findings
  -> Ornith 35B typed findings
  -> exact union of {code, scenarioIds}
  -> one Gemma verification per source-bound claim
  -> confirmed finding clusters only
  -> targeted Qwen27 repair of eligible IDs
  -> immutable admitted taxonomy checkpoint
  -> taxonomy-bound Stage P blueprint schema
  -> blueprint repairs that may change only non-taxonomy niche fields
  -> repeat full-set proposal and verification
  -> immutable taxonomy admission checkpoint
  -> Stage P may begin
```

Supported finding codes are:

- `mode_submode_mismatch`
- `field_role_misuse`
- `semantic_duplicate_template_collapse`
- `implausible_anchor`

Proposer rationales are never sent to the verifier. The verifier receives only the exact typed
claim, the topic card, the affected source anchor bindings, and the fixed definition of that code.
It returns only `confirmed: true|false`.

### Evidence-local checkpoint identity

A claim-verification checkpoint is keyed by the exact affected anchor evidence, not the complete
20-anchor view. Changing an unrelated anchor therefore cannot flip a previous verification result.
Duplicate claims bind all and only the IDs named by that claim.

### Monotonic repair lineage

After a repair cycle, unchanged anchors that were cleared by the prior judgment are byte-frozen and
ineligible for later repair. The full set is still presented to proposers so a changed anchor can be
compared with immutable peers. If a duplicate claim includes one changed and one immutable anchor,
only the changed anchor is eligible for repair.

This prevents the observed oscillation where a single open judge cleared an anchor in one cycle and
reopened the identical bytes in the next cycle.

### Canonical repair contract

Repair responses use explicit canonical properties:

- `submode`
- `participantRelationship`
- `setting`
- `centralResource`
- `centralTension`
- `changedFields`

Every repaired anchor regenerates all five semantic fields. `changedFields` must name all five fields
exactly once, and host validation checks the declaration against the actual parent delta. Repaired
submodes may not copy the parent or an immutable anchor. Compact `u/r/s/c/n` wire names are retained
for raw generation but prohibited from the repair interface because they caused model role confusion.

The host never patches semantic text. Invalid repair output is rejected and retried against the same
typed finding and immutable parent.

## Scenario admission path

The Stage-E scenario path uses the same principle:

```text
two independent semantic proposers
  -> exact union of typed {code, scenarioIds} claims
  -> no proposer rationale passed forward
  -> one source-bound final verification per claim
  -> only confirmed claims materialize rejection
```

The final verifier does not audit every scenario again. A scenario no proposer claimed cannot be
rejected by verifier hallucination. Pair claims are verified against exactly two scenario-plus-
blueprint records. Single claims are verified against one same-ID admitted blueprint and scenario.

The production scenario topology currently uses Qwen35 and Ornith Vision as proposers and Gemma as
the final evidence-bound verifier.

## Runtime configuration

The versioned template is:

`ground_truth_finetuning/config/personaplex-runtime.env.example`

The active host source of truth is:

`/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env`

Required variables are:

```bash
PERSONAPLEX_CASCADE_PLANNER_ENDPOINT
PERSONAPLEX_CASCADE_PLANNER_MODEL
PERSONAPLEX_TAXONOMY_REPAIR_ENDPOINT
PERSONAPLEX_TAXONOMY_REPAIR_MODEL
PERSONAPLEX_BLUEPRINT_JUDGE_ENDPOINT
PERSONAPLEX_BLUEPRINT_JUDGE_MODEL
PERSONAPLEX_TAXONOMY_SECONDARY_JUDGE_ENDPOINT
PERSONAPLEX_TAXONOMY_SECONDARY_JUDGE_MODEL
PERSONAPLEX_TAXONOMY_VERIFIER_ENDPOINT
PERSONAPLEX_TAXONOMY_VERIFIER_MODEL
PERSONAPLEX_TAXONOMY_VERIFIER_WORKERS
```

Do not hard-code physical GPU totals, free-memory quantities, or alternate ports in the pipeline.
Worker services own CUDA affinity. Endpoint values come from the environment source of truth.

## Production invocation

```bash
set -a
source /srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env
set +a

python3 -m ground_truth_finetuning.tools.build_scenarios_from_blueprints_v5 \
  --request "$REQUEST_JSON" \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --stage all \
  --planner-endpoint "$PERSONAPLEX_CASCADE_PLANNER_ENDPOINT" \
  --planner-model "$PERSONAPLEX_CASCADE_PLANNER_MODEL" \
  --judge-endpoint "$PERSONAPLEX_BLUEPRINT_JUDGE_ENDPOINT" \
  --judge-model "$PERSONAPLEX_BLUEPRINT_JUDGE_MODEL" \
  --secondary-taxonomy-judge-endpoint "$PERSONAPLEX_TAXONOMY_SECONDARY_JUDGE_ENDPOINT" \
  --secondary-taxonomy-judge-model "$PERSONAPLEX_TAXONOMY_SECONDARY_JUDGE_MODEL" \
  --taxonomy-verifier-endpoint "$PERSONAPLEX_TAXONOMY_VERIFIER_ENDPOINT" \
  --taxonomy-verifier-model "$PERSONAPLEX_TAXONOMY_VERIFIER_MODEL" \
  --taxonomy-verifier-workers "$PERSONAPLEX_TAXONOMY_VERIFIER_WORKERS" \
  --max-workers 3 \
  --judge-workers 3 \
  --max-attempts 6 \
  --max-repair-cycles 6 \
  --resume
```

## Immutable artifacts

Taxonomy lineage is rooted under:

```text
.scenario_blueprint_v5/checkpoints/taxonomy/
.scenario_blueprint_v5/checkpoints/taxonomy_judgments/
.scenario_blueprint_v5/checkpoints/taxonomy_repairs/
.scenario_blueprint_v5/checkpoints/taxonomy_verifications/
.scenario_blueprint_v5/checkpoints/taxonomy_admissions/
```

Never edit or replace a checkpoint. A protocol, source, binding, schema, prompt, or input change must
produce a new content-addressed identity. Resume may reuse only a checkpoint whose body hash and all
bound identities validate.

## Admission criteria

Stage P is prohibited until every one of the 50 topic taxonomies has an admission checkpoint under
the current protocol and model binding. Stage E is prohibited until every Stage-P blueprint set is
admitted. Compact fanout is prohibited until all 1,000 scenario contracts pass structural validation
and the calibrated typed-claim semantic gate.

No threshold may be loosened merely to increase throughput. A high rejection rate must be classified
as generation failure, model-protocol failure, verifier noise, transport failure, or true content
failure, then fixed at that source.

## Live calibration record

The July 20, 2026 production calibration established:

- Scenario controls for prescribed target dialogue, explicit unsafe action, and exact duplication
  were proposed and independently confirmed with the expected typed code.
- The legacy scenario corpus produced 29 proposer claims; the final verifier retained four known
  mode mismatches and filtered duplicate/template noise.
- In the first stable taxonomy sample, 41 of 106 proposer claims were confirmed, filtering 61 percent
  of proposer noise.
- The first nine admitted topic taxonomies required zero to two targeted repairs; no admitted topic
  required more than two cycles after evidence-local monotonic lineage was enabled.

These observations validate the architecture, not the final dataset. The 50-topic, 1,000-scenario,
rendering, packing, training, and held-out 95 percent reliability gates remain mandatory.
