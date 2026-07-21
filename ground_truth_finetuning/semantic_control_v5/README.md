# PersonaPlex Semantic Control v5 Ground Truth

Status: active implementation and execution contract
Last reconciled: 2026-07-20
Promotion state: scenario corpus quarantined; not training eligible, not trained,
not generated-audio validated, not live validated

## 1. Authority

This directory is the current source of truth for the v5 causal-data, native
training, generated evaluation, and live-promotion path. The v4 documents remain
historical evidence and explain why the previous composite two-branch corpus did
not generalize. They do not override the v5 contracts in this directory.

When sources disagree, use this order:

1. Immutable generated artifacts and their content hashes.
2. Executable schemas and validators.
3. This v5 ground-truth directory.
4. Historical v4 and repository-root planning documents.

No teacher-forced metric, unit test, schema pass, or successful synthetic render
is by itself evidence of 95 percent live semantic-control reliability.

## 2. Status vocabulary

| Status | Meaning |
| --- | --- |
| `implemented` | The production-path code and contract exist. |
| `focused-tested` | A focused test or smoke path has exercised the stated invariant. |
| `generating` | Durable artifacts exist and the stage remains incomplete. |
| `observed` | Timestamped operational telemetry that may already be stale. |
| `quarantined` | Structurally materialized data failed an independent admission gate and cannot proceed to training. |
| `pending` | Required implementation or empirical evidence does not yet exist. |
| `blocked` | A named defect prevents the next stage. |

## 3. Timestamped operational observations

Counts, PIDs, GPU utilization, and process state are ephemeral observations. They
are never ground truth and must never be described as current without a fresh
live query. A PID proves activity only at the instant it was observed. Stage
completion requires the canonical artifact, expected cardinality, run manifest,
and bound hashes.

Earlier PID and partial-count observations are retained only as operational
history. The authoritative completed-stage evidence is now the canonical 50
topic and 1,000 scenario corpus plus its immutable independent audit. Process
state remains ephemeral and must still be queried with
[RUNBOOK.md](RUNBOOK.md#2-authoritative-live-operational-checks).

The independent Qwen v4 clustered-findings audit rejected `918/1000` scenarios,
or `91.8%`, primarily for semantic mode collapse. The complete scenario corpus
is quarantined and is not training eligible. A set of 55 blind repair candidates
was also discarded and must not be reincorporated.

Immutable rejection evidence:

```text
/srv/voxrn_cache/personaplex/training/cascade-v5-pilot-20260720/scenario_stage_rejection.v1.json
sha256:c02f53487d795b213ad87078f1ea133f912b149bbdbc7886a7d20af4dc9755c1
```

The following table separates durable implementation state from historical
telemetry:

| Stage | State | Durable evidence |
| --- | --- | --- |
| Topic planning | structurally complete | `50/50` topics exist in the completed run. This does not certify scenario diversity. |
| Initial scenario planning | structurally complete, quarantined | `1000/1000` scenarios exist, but the independent audit rejected `918/1000` for primarily semantic mode collapse. |
| Initial scenario repair | discarded | All 55 blind repair candidates were discarded; blind local repair is not an accepted recovery architecture. |
| Replacement scenario blueprint | designed, implementation in progress | Joint authentic 20-slot diversification per topic, followed by slot-bound expansion, whole-topic independent scrutiny, and targeted slot repair. |
| Compact trajectory fan-out | blocked | The desired one-call, ten-candidate compact fan-out is not implemented. Current code generates ten full trajectory objects sequentially per scenario. |
| Trajectory candidates | pending behind architecture gate | Do not launch the inefficient full trajectory stage as the final architecture. Query artifacts rather than relying on a recorded count. |
| Balanced selection | implemented, not executed | Selector supports `250` primary plus `250` reserve groups. |
| Four-role group planning | implemented, focused-tested, not executed at scale | Roles are `verified_positive`, `verified_negative`, `uncertain`, and `superseded`. |
| Voryn shared-prefix rendering | implemented, focused-tested, not executed on v5 corpus | One complete group is assigned to one lane; one exact pre-pivot duplex snapshot is replayed. |
| Post-render v5 bridge | implemented, focused-tested, not executed on v5 corpus | Immutable `planRecordId` and branch-artifact propagation are present. |
| Native materialization and packing | implemented, focused-tested, not executed on v5 corpus | Shared-prefix storage, strict timing, leakage-component splits, and the content-addressed trainer binding are present. |
| Full-rank native training | implemented, smoke-tested, not run on v5 corpus | CUDA/NCCL path, pre-`torchrun` certified-pack admission, and checkpoints exist; no v5 convergence result exists. |
| Generated native evaluation | pending | No free-running v5 checkpoint has passed the generated duplex gate. |
| Live Twilio evaluation | pending | No v5 checkpoint has passed the live 95 percent gate. |
| Public release | pending | Release is forbidden until generated and live gates pass. |

The compact fan-out discrepancy is deliberate documentation, not a wording
error. Review also found that the positional candidate schema depends on
unsupported `prefixItems` and that the proposed output contract requires roughly
12,000 tokens while the active planner budget is roughly 4,000. The redesign
uses required candidate-ID object properties and a genuinely compact Stage A.
The efficient target architecture is specified in
[ARCHITECTURE.md](ARCHITECTURE.md), and its implementation is a hard prerequisite
in [TODO.md](TODO.md).

## 4. Ground-truth documents

| Document | Purpose |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | End-to-end data, control, rendering, lineage, training, and runtime contracts. |
| [RUNBOOK.md](RUNBOOK.md) | Exact repository entry points, artifact names, and safe execution sequence. |
| [TODO.md](TODO.md) | Exhaustive implementation and evidence checklist with explicit promotion blockers. |
| [REFERENCES.md](REFERENCES.md) | Primary-source evidence and the boundary between literature and local design decisions. |

## 5. Canonical implementation anchors

| Concern | Source anchor |
| --- | --- |
| Seed, request, and artifact schemas | `schemas/diverse_seed_library.v2.schema.json`, `schemas/diverse_corpus_request.v2.schema.json`, `schemas/diverse_cascade_artifacts.v2.schema.json` |
| Bound 50-seed catalog | `seed_catalogs/personaplex_diverse_seed_library.v2.json` |
| v5 run request | `requests/personaplex_diverse_50x20x10.control-v5.json` |
| Authentic structured planning | `training/diverse_cascade.py` |
| Per-artifact persistence | `tools/build_diverse_synthesis_cascade.py` |
| Pre-generation cardinality and provenance | `tools/materialize_diverse_synthesis_cascade.py` |
| Schema-7 render plan and typed controls | `tools/compile_diverse_cascade_voryn_plan.py` |
| Four-sibling lane scheduler | Voryn `scripts/run-personaplex-synthetic-lane.js` |
| Shared-prefix capture and replay | Voryn `lib/personaplexSyntheticGroupLane.js` |
| Certified duplex export and v5 branch artifact | `tools/export_controlled_duplex_dataset.py` |
| Target-free precodec projection | `tools/prepare_controlled_native_adapter_dataset.py` |
| Native tensor encoding and certification | `tools/encode_controlled_native_adapter_tensors.py`, `tools/certify_controlled_native_corpus.py` |
| Native shared-prefix group materialization | `tools/materialize_native_causal_groups_v5.py` |
| Leakage-component packing | `training/causal_group_pack.py`, `tools/pack_native_causal_groups.py` |
| Native MoshiRAG receiver | `training/native_moshirag_control.py`, `training/native_training.py` |
| CUDA/NCCL full-rank trainer | `tools/train_native_moshirag_control.py` |

## 6. Non-negotiable claim boundary

The project may claim that the architecture is implemented only where the
status table says `implemented`. It may claim model controllability only after
all of the following are true:

1. Group-disjoint generated evaluation passes on withheld causal operators,
   topics, lexical realizations, voices, and revision trajectories.
2. Every scored target is free-running native PersonaPlex output, not a
   teacher-forced target comparison.
3. Actual barge-in cancels queued audio and the recovery response uses the
   newest acknowledged control revision.
4. Semantic, factual, tool-result, timing, voice, codec, and latency strata meet
   the preregistered thresholds.
5. The independent live Twilio suite meets the preregistered 95 percent
   reliability criterion with its required confidence bound.

Exact mandated wording is never delegated to probabilistic semantic control.
It routes to the separately validated strict renderer.
