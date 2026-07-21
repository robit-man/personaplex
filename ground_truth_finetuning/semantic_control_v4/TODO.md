# Semantic Control v4 Exhaustive Execution Ledger

Legend: `[ ]` not started, `[~]` active, `[x]` evidenced complete, `[!]` failed
and requires a new iteration.

## A. Ground truth and baseline

- [x] `A-001` Record the control-v3 run as baseline, not release candidate.
- [x] `A-002` Record control-v3 context-ablation failure.
- [x] `A-003` Audit existing counterfactual lineage in source records.
- [x] `A-004` Count complete source groups and native pivot pairs.
- [x] `A-005` Verify source replay prefixes by audio hash, text, and timing.
- [x] `A-006` Persist the pair audit as a signed JSON artifact.
- [ ] `A-007` Capture frozen no-control generated-audio baseline.
- [ ] `A-008` Capture frozen role-prompt generated-audio baseline.
- [ ] `A-009` Capture baseline duplex timing and interruption metrics.
- [x] `A-010` Freeze gate thresholds before candidate evaluation.

## B. Research and architecture

- [x] `B-001` Inspect PersonaPlex architecture and synthetic-data method.
- [x] `B-002` Inspect Moshi delayed multistream and inner-monologue design.
- [x] `B-003` Inspect MoshiRAG paper and released streaming-sum source.
- [x] `B-004` Inspect spoken-dialogue DPO-LN design and audio-loss instability.
- [x] `B-005` Inspect multi-faceted duplex GRPO reward design.
- [x] `B-006` Inspect Full-Duplex-Bench and v3 disfluency/tool requirements.
- [x] `B-007` Write source-backed architecture synthesis.
- [ ] `B-008` Complete per-source GRADE evidence sidecars.
- [ ] `B-009` Record architecture impact analysis.
- [x] `B-010` Accept ADR for temporal control stream.
- [ ] `B-011` Freeze semantic control v2 schema.
- [x] `B-012` Freeze model/checkpoint compatibility contract.

## C. Schema and encoding

- [ ] `C-001` Add semantic control frame v2 JSON schema.
- [ ] `C-002` Add backward migration from control frame v1.
- [ ] `C-003` Add closed field-id vocabulary.
- [ ] `C-004` Add value-type, source, and revision-relation vocabularies.
- [x] `C-005` Preserve natural lexical values during serialization.
- [x] `C-006` Reserve token budgets for critical fields.
- [x] `C-007` Emit field/type/source arrays aligned to token ids.
- [ ] `C-008` Record per-field truncation diagnostics.
- [ ] `C-009` Reject all target/canonical response aliases recursively.
- [x] `C-010` Add schema and encoding golden fixtures.
- [x] `C-011` Add target-leak negative fixtures.
- [ ] `C-012` Add truncation and unknown-extension fixtures.

## D. Pair indexing and certification

- [x] `D-001` Build explicit native counterfactual pair index.
- [x] `D-002` Match group, target turn, base state, and branch ids.
- [x] `D-003` Require divergent current state and target label hashes.
- [x] `D-004` Bind partner example ids in both directions.
- [x] `D-005` Verify all pair members share one split.
- [x] `D-006` Verify replay prefix audio/text/timing lineage.
- [x] `D-007` Verify exactly one declared causal field changed.
- [ ] `D-008` Quarantine incomplete groups from pair loss only.
- [ ] `D-009` Retain independently valid incomplete turns for ordinary SFT.
- [x] `D-010` Emit pair counts by axis, topic, voice, and split.
- [x] `D-011` Add pair-index certificate and hashes.
- [x] `D-012` Validate existing V8 migration end to end.

## E. Model adapter

- [x] `E-001` Implement `SemanticControlStreamAdapter`.
- [x] `E-002` Reuse frozen PersonaPlex text embeddings.
- [x] `E-003` Add trainable field/type/source/revision embeddings.
- [x] `E-004` Add compact contextual encoder.
- [x] `E-005` Add learned temporal compression queries.
- [x] `E-006` Add 4096-dimensional output projection.
- [x] `E-007` Add conservative per-row gates.
- [x] `E-008` Add an exact-zero null stream.
- [ ] `E-009` Add stream norm telemetry.
- [x] `E-010` Add checkpoint-bound architecture config.
- [x] `E-011` Add state-dict compatibility validation.
- [x] `E-012` Add adapter shape, mask, and dropout tests.

## F. Native forward and runtime

- [ ] `F-001` Generalize `streaming_sum` naming to `control_stream` aliases.
- [ ] `F-002` Preserve backward evidence-stream checkpoint support.
- [x] `F-003` Apply control rows only to real native temporal steps.
- [x] `F-004` Zero condition after stream exhaustion.
- [x] `F-005` Add per-slot queue replacement with revision identity.
- [x] `F-006` Add atomic queue cancellation.
- [x] `F-007` Add `SemanticControlStreamProvider`.
- [x] `F-008` Cache by frame hash, adapter hash, and model hash.
- [x] `F-009` Snapshot at matching turn boundary only.
- [x] `F-010` Reject stale, expired, and context-mismatched revisions.
- [x] `F-011` Invalidate generation before clearing barge-in media.
- [x] `F-012` Check generation id immediately before network egress.
- [ ] `F-013` Add strict-route transition.
- [ ] `F-014` Add typed model terminal action and idempotent end-call action.
- [ ] `F-015` Add concurrent-call state isolation test.
- [ ] `F-016` Add runtime CUDA/control encode latency telemetry.

## G. Causal training objective

- [x] `G-001` Add per-example normalized text NLL.
- [x] `G-002` Add matched SFT with agent-only text/audio loss.
- [x] `G-003` Add both-direction sibling-control margin loss.
- [x] `G-004` Add current-vs-stale margin loss.
- [x] `G-005` Add current-vs-null margin loss.
- [x] `G-006` Add bounded gate regularization.
- [ ] `G-007` Add structured whole/evidence/style dropout.
- [ ] `G-008` Keep critical obligations immune to field dropout.
- [ ] `G-009` Add pair-aware sampler and length buckets.
- [x] `G-010` Keep pair members on one rank and step.
- [ ] `G-011` Add gradient accumulation and mixed precision.
- [x] `G-012` Add distributed metric reduction.
- [ ] `G-013` Add exact trainable/frozen parameter audit.
- [ ] `G-014` Add immutable run contract and checkpoint hashes.

## H. Checkpoint evaluation

- [x] `H-001` Evaluate all four target/control pair combinations.
- [x] `H-002` Report direction and whole-pair accuracy.
- [ ] `H-003` Report margins by changed field.
- [ ] `H-004` Report stale/null/random control discrimination.
- [ ] `H-005` Report critical field ablations.
- [ ] `H-006` Report no-control regression.
- [ ] `H-007` Report control encode latency and gate norms.
- [ ] `H-008` Select checkpoints by causal metrics before matched loss.
- [ ] `H-009` Emit machine decision `advance/iterate/reject`.
- [x] `H-010` Prevent automatic promotion on teacher-forced evidence alone.

## I. Stage-2 temporal adaptation

- [ ] `I-001` Enumerate upper temporal projection paths from pinned source.
- [ ] `I-002` Implement source-bound LoRA insertion.
- [ ] `I-003` Freeze Mimi, depth transformer, voice path, and caller embeddings.
- [ ] `I-004` Implement frozen Stage-1 reference policy.
- [ ] `I-005` Implement text-only DPO-LN/APO objective.
- [ ] `I-006` Add KL/reference regularization.
- [ ] `I-007` Mix ordinary no-control replay examples.
- [ ] `I-008` Audit timing, voice, and interruption regressions.
- [ ] `I-009` Reject unstable audio-token preference optimization.
- [ ] `I-010` Produce Stage-2 ablation report.

## J. Semantic evaluator

- [x] `J-001` Replace substring scoring as a promotion gate.
- [x] `J-002` Implement raw-JSON typed semantic judge.
- [x] `J-003` Require per-obligation verdicts and transcript spans.
- [ ] `J-004` Add independent fallback judge on allowed CUDA.
- [x] `J-005` Treat persistent judge outage as trial failure.
- [x] `J-006` Add blind counterfactual pair judge.
- [ ] `J-007` Add judge calibration set and human labels.
- [ ] `J-008` Report judge agreement and confidence calibration.
- [x] `J-009` Prevent target transcript and branch-name leakage to judge.
- [x] `J-010` Add Wilson confidence interval implementation.

## K. Generated-audio harness

- [ ] `K-001` Stream native duplex prefix at 12.5 Hz.
- [ ] `K-002` Apply current control at exact boundary.
- [ ] `K-003` Generate free-running PersonaPlex audio and text.
- [ ] `K-004` Capture control-row consumption and generation ids.
- [ ] `K-005` Decode native audio to waveform.
- [ ] `K-006` Perform 8 kHz mu-law round trip.
- [ ] `K-007` Run independent Whisper ASR with word timing.
- [ ] `K-008` Run typed semantic and pair judges.
- [ ] `K-009` Exercise multiple paired sampling seeds.
- [ ] `K-010` Add silence, clipping, codec, and voice checks.
- [ ] `K-011` Add model-terminal and timeout handling.
- [ ] `K-012` Archive complete trial manifests.

## L. Duplex and Twilio failure modes

- [ ] `L-001` Pause-hold scenario.
- [ ] `L-002` Smooth-turn scenario.
- [ ] `L-003` Backchannel timing scenario.
- [ ] `L-004` Early and mid-speech barge-in scenarios.
- [ ] `L-005` False-start scenario.
- [ ] `L-006` Caller self-correction scenario.
- [ ] `L-007` Tool delay and expiry scenario.
- [ ] `L-008` Stale control arrival scenario.
- [ ] `L-009` Lost acknowledgement scenario.
- [ ] `L-010` Packet delay, loss, duplicate, and reorder scenarios.
- [ ] `L-011` Websocket disconnect/reconnect scenario.
- [ ] `L-012` Server restart scenario.
- [ ] `L-013` Strict-render failure scenario.
- [ ] `L-014` No-invalid-generation-egress assertion.
- [ ] `L-015` End-call idempotency and no-goodbye-loop assertion.

## M. Next synthesis batch

- [ ] `M-001` Freeze 50 topic cards.
- [ ] `M-002` Generate and validate 20 scenarios per topic.
- [ ] `M-003` Generate and validate 10 trajectory leaves per scenario.
- [ ] `M-004` Compute coverage/novelty embeddings for 10,000 leaves.
- [ ] `M-005` Select 500 balanced causal groups for 1,000 conversations.
- [ ] `M-006` Balance all required causal axes.
- [ ] `M-007` Balance interaction posture and length.
- [ ] `M-008` Balance voice pairs and provenance sources.
- [ ] `M-009` Render each shared prefix once.
- [ ] `M-010` Branch only at the declared control pivot.
- [ ] `M-011` Use Chatterbox Turbo on CUDA 0/1/2 only.
- [ ] `M-012` Run turn-local Whisper/authenticity gates.
- [ ] `M-013` Patch failed turns without rerendering accepted prefixes.
- [ ] `M-014` Run independent semantic and pair certification.
- [ ] `M-015` Require model-driven terminal action.
- [ ] `M-016` Certify native tensors and group-isolated splits.
- [ ] `M-017` Publish dataset only after certificate and release review.

## N. Resource and automation

- [x] `N-001` Discover CUDA inventory and UUID mapping dynamically.
- [x] `N-002` Enforce physical allowlist 0/1/2.
- [x] `N-003` Discover VRAM capacity/free memory dynamically.
- [ ] `N-004` Estimate per-stage residency from artifacts and calibration.
- [x] `N-005` Refuse CPU model load/offload/fallback.
- [x] `N-006` Pause new tasks only above 80% host memory utilization.
- [x] `N-007` Resume automatically below threshold.
- [x] `N-008` Sample live `nvidia-smi` activity during startup and work.
- [ ] `N-009` Fail startup when expected GPU residency never appears.
- [x] `N-010` Preserve unrelated GPU services and re-admit dynamically.
- [x] `N-013` Probe NCCL transport on the admitted GPU set and record the selected mode.
- [ ] `N-011` Persist service/source-of-truth endpoint routing.
- [ ] `N-012` Add watchdog without unbounded process/log memory.

## O. Execution ladder

- [x] `O-001` Run schema/unit tests.
- [x] `O-002` Run one real pair forward/backward.
- [ ] `O-003` Overfit 20 pairs to near-perfect discrimination.
- [~] `O-004` Run existing V8 adapter-only Stage 1.
- [ ] `O-005` Evaluate all existing held-out causal pairs.
- [ ] `O-006` Generate held-out speech for selected checkpoints.
- [ ] `O-007` Diagnose failures by causal axis.
- [ ] `O-008` Launch next 1,000-conversation synthesis batch.
- [ ] `O-009` Train combined certified corpus.
- [ ] `O-010` Decide whether Stage 2 is required.
- [ ] `O-011` Run Stage 2 when required.
- [ ] `O-012` Build generated preference set.
- [ ] `O-013` Run Stage 3 alignment.
- [ ] `O-014` Freeze final 1,000-trial gate.
- [ ] `O-015` Achieve overall point rate >=0.97 and Wilson lower bound >=0.95.
- [ ] `O-016` Achieve pair sensitivity >=0.95.
- [ ] `O-017` Demonstrate zero stale/policy-sensitive emissions.
- [ ] `O-018` Run live-equivalent Twilio suite.
- [ ] `O-019` Run bounded live-infrastructure synthetic-call suite.
- [ ] `O-020` Publish only the exact passed checkpoint and evidence.

## P. Product follow-through

- [ ] `P-001` Expose PersonaPlex in agent voice selection.
- [ ] `P-002` Mark completed tool-selection tab green on progression.
- [ ] `P-003` Make required call tools preselected but removable.
- [ ] `P-004` Warn when removal breaks the selected call architecture.
- [ ] `P-005` Use day/night CSS variables for tool cards and selected state.
- [ ] `P-006` Keep setup variables at top and gate tool progression.
- [ ] `P-007` Validate product selection reaches the controlled runtime.
