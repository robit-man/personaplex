# Semantic Control v4 Traceability

| Requirement | Design artifact | Planned implementation | Required evidence |
| --- | --- | --- | --- |
| `SC-001` Typed mutable control | `ARCHITECTURE.md` sections 3 and 8 | control schema/contracts/runtime | schema and state-machine tests |
| `SC-002` Trained speech-model input | `ARCHITECTURE.md` sections 4-6 | field encoder and temporal stream adapter | native forward sensitivity |
| `SC-003` No target leakage | `DATASET_CONTRACT.md` sections 7 and 10 | exporter/certifier | negative leakage fixtures |
| `SC-004` Causal pair learning | `TRAINING_CONTRACT.md` sections 3 and 5 | pair index, sampler, ranking loss | held-out pair accuracy |
| `SC-005` Stale revision cancellation | `ARCHITECTURE.md` sections 8-9 | runtime session and egress queue | protocol/barge-in tests |
| `SC-006` Exact wording isolation | `RELIABILITY_CONTRACT.md` section 1 | strict router | strict route E2E test |
| `SC-007` Natural full duplex | `EVALUATION_CONTRACT.md` sections 4 and 7 | generated duplex harness | pause/turn/backchannel/interruption report |
| `SC-008` Model-driven termination | `ARCHITECTURE.md` section 10 | terminal action head/path | no-loop and end-tool tests |
| `SC-009` CUDA-only 0/1/2 | `TRAINING_CONTRACT.md` section 9 | dynamic GPU admission | telemetry and process mapping |
| `SC-010` Dynamic host memory guard | `TRAINING_CONTRACT.md` section 9 | resource controller | >80% pause/resume test |
| `SC-011` 95% generated reliability | `RELIABILITY_CONTRACT.md` | reliability evaluator | signed 1,000-trial report |
| `SC-012` Research provenance | `RESEARCH_SYNTHESIS.md` | evidence register | GRADE assessment and citations |
| `SC-013` Reproducible release | all contracts | run cards/model card/scripts | hash-bound public artifacts |

## Baseline evidence links

| Evidence | Location |
| --- | --- |
| Certified control-v3 native root | `/srv/voxrn_cache/personaplex-transition/prepared/v7-p1000v5-controlv3-20260718T212644Z` |
| Completed training attempt | `06_training_execution/attempt-20260718T221058Z/training` under that root |
| Existing source plan | `/srv/personaplex_workspace/ground_truth_runs/personaplex-1000-plan.v8-counterfactual-diverse-v6.jsonl` |
| Existing private dataset | `cudabenchmarktest/personaplex-v7-semantic-control-synthetic-controlv3` |
| MoshiRAG source snapshot | `/srv/personaplex_workspace/moshi-rag-research` at `8c6dfc1` |
| Canonical v4 corpus and pair certificates | `/srv/voxrn_cache/personaplex-transition/v4-causal-redesign-20260719/02_canonical_native` |
| Single-GPU forward/backward smoke | `/srv/voxrn_cache/personaplex-transition/v4-causal-redesign-20260719/03_smoke-cropped-20260720T004327Z` |
| Three-A100 transport and training smoke | `/srv/voxrn_cache/personaplex-transition/v4-causal-redesign-20260719/03_smoke-ddp-transportfix-20260720T010259Z` |

## Current implementation evidence

| Requirement | Current state | Evidence |
| --- | --- | --- |
| `SC-001` | `structurally_validated` | Typed revision/runtime tests and controlled server integration. |
| `SC-002` | `teacher_forced_validated` | Three-A100 causal forward/backward smoke and checkpoint deltas. |
| `SC-003` | `structurally_validated` | Pair certification and target-leak negative tests. |
| `SC-004` | `teacher_forced_validated` | 396 exact native causal pairs; convergence run active. |
| `SC-005` | `structurally_validated` | Revision, cancellation, and stream-clear tests. |
| `SC-007` | `implemented` | Paced duplex/codec harness exists; candidate trial pending. |
| `SC-009` | `structurally_validated` | Physical GPUs 0/1/2 observed in resource telemetry. |
| `SC-010` | `structurally_validated` | Capacity-relative admission and `/proc/meminfo` guard. |
| `SC-011` | `planned` | Generated-audio release set has not run. |
| `SC-013` | `implemented` | Hash-bound contracts/checkpoints exist; public release pending. |

## Evidence-state vocabulary

Every row in future traceability updates uses one status:

```text
planned
implemented
structurally_validated
teacher_forced_validated
generated_audio_validated
live_equivalent_validated
release_passed
failed
```

`implemented` is never presented as `validated`. A requirement reaches
`release_passed` only when its evidence artifact is immutable and included in
the final run manifest.
