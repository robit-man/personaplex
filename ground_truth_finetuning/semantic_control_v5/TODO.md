# Semantic Control v5 Exhaustive Checklist

Status: active
Completion rule: every required item is checked and linked to immutable evidence

Legend:

| Mark | Meaning |
| --- | --- |
| `[x]` | Implemented with named evidence. |
| `[~]` | In progress or partially evidenced. |
| `[ ]` | Pending. |
| `[!]` | Current blocker or invalid claim. |

## A. Ground truth and provenance

- [x] `DOC-01` Preserve v4 as historical failure evidence rather than rewriting it as success.
- [x] `DOC-02` Establish `semantic_control_v5/README.md` as the current status authority.
- [x] `DOC-03` Record the evidence/decision boundary and primary sources in `REFERENCES.md`.
- [x] `DOC-04` Record exact code and artifact anchors in `ARCHITECTURE.md` and `RUNBOOK.md`.
- [x] `DOC-05` State that teacher-forced gates are not generated or live reliability.
- [x] `DOC-06` State that exact wording routes to a strict renderer.
- [x] `DOC-07` Classify PIDs, counts, process state, and GPU utilization as timestamped observations rather than ground truth.
- [ ] `DOC-08` Bind the final public model card to immutable dataset, checkpoint, code, and evaluation hashes.
- [x] `DOC-09` Add authoritative live process, checkpoint, canonical-artifact, hash, and sampled GPU inspection commands.

## B. Schemas and request binding

- [x] `SCHEMA-01` Add `schemas/diverse_seed_library.v2.schema.json`.
- [x] `SCHEMA-02` Add `schemas/diverse_corpus_request.v2.schema.json`.
- [x] `SCHEMA-03` Add `schemas/diverse_cascade_artifacts.v2.schema.json`.
- [x] `SCHEMA-04` Bind exactly 50 seed domains in `seed_catalogs/personaplex_diverse_seed_library.v2.json`.
- [x] `SCHEMA-05` Bind the catalog content hash in `requests/personaplex_diverse_50x20x10.control-v5.json`.
- [x] `SCHEMA-06` Bind `50 x 20 x 10`, 250 primary, 250 reserve, and four siblings.
- [x] `SCHEMA-07` Bind physical CUDA devices `0,1,2`, dynamic hardware discovery, and no CPU model fallback.
- [x] `SCHEMA-08` Bind Chatterbox Turbo rendering and Whisper admission measurement.
- [x] `SCHEMA-09` Bind native temporal streaming-sum conditioning and strict-before-target timing.
- [x] `SCHEMA-10` Bind model-selected `end_call` and strict-renderer exact-wording fallback.
- [~] `SCHEMA-11` Redesign compact candidate and full-expansion artifact definitions for required candidate-ID object properties.
- [ ] `SCHEMA-12` Add immutable linkage from each full expansion to its compact candidate hash.
- [!] `SCHEMA-13` Remove unsupported `prefixItems` from the compact fan-out response contract.
- [ ] `SCHEMA-14` Prove the complete Stage A schema is accepted by the active structured-output serving path.

## C. Authentic structured planning

- [x] `PLAN-01` Send strict JSON Schema through OpenAI-compatible `response_format`.
- [x] `PLAN-02` Parse raw JSON only and prohibit regex, prose extraction, field coercion, and heuristic repair.
- [x] `PLAN-03` Disable planner reasoning in the request path.
- [x] `PLAN-04` Restrict endpoint failover to transport and explicitly retriable HTTP failures.
- [x] `PLAN-05` Retry only the structurally invalid assigned artifact.
- [x] `PLAN-06` Persist immutable per-artifact checkpoints and resume only missing identities.
- [!] `PLAN-07` Replace ten sequential full trajectory calls per scenario with one authentic ten-candidate compact fan-out.
- [~] `PLAN-08` Define a strict compact-candidate object with ten required candidate-ID properties and no `prefixItems` dependency.
- [ ] `PLAN-09` Persist every compact candidate independently before stage aggregation.
- [ ] `PLAN-10` Ensure compact candidates contain no target dialogue, target hashes, or canonical agent responses.
- [ ] `PLAN-11` Select 250 primary and 250 reserve compact candidates before full expansion.
- [ ] `PLAN-12` Fully expand only the selected 500 and bind each expansion to its compact candidate hash.
- [ ] `PLAN-13` Prove failed expansion retries do not mutate selection rank or distribution.
- [ ] `PLAN-14` Prove reserve replacement consumes the next preselected typed reserve rather than replanning distribution.
- [!] `PLAN-15` Replace the rejected roughly 12,000-token Stage A response design with a genuinely compact response fitting the roughly 4,000-token planner output contract.
- [ ] `PLAN-16` Measure serialized worst-case Stage A output and retain explicit headroom before launching inference.

## D. Topic and scenario generation

- [x] `CASCADE-01` Generate and checkpoint 50 topic cards, one per bound seed.
- [x] `CASCADE-02` Structurally generate all 1,000 initial scenario contracts.
- [x] `CASCADE-03` Complete the initial resumable scenario run; historical process PIDs no longer determine stage state.
- [x] `CASCADE-04` Materialize all 1,000 initial scenario checkpoints and the canonical stage artifact.
- [x] `CASCADE-05` Run independent Qwen v4 whole-corpus clustered-findings scrutiny.
- [!] `CASCADE-06` Quarantine the initial corpus after `918/1000`, or `91.8%`, were rejected primarily for semantic mode collapse.
- [!] `CASCADE-07` Treat broad semantic scenario coverage as failed despite structural cardinality success.
- [!] `CASCADE-08` Treat the initial corpus as not training eligible and prohibit favorable-subset extraction.
- [ ] `CASCADE-09` Generate 10,000 compact candidates with zero missing scenario ordinals.
- [ ] `CASCADE-10` Validate uniqueness across premise, causal transition, style, outcome, and duplex-event dimensions.
- [x] `CASCADE-11` Preserve immutable rejection report `scenario_stage_rejection.v1.json` with report ID `sha256:c02f53487d795b213ad87078f1ea133f912b149bbdbc7886a7d20af4dc9755c1`.
- [x] `CASCADE-12` Discard all 55 blind repair candidates.
- [x] `CASCADE-13` Specify one joint authentic 20-slot diversification blueprint per topic.
- [~] `CASCADE-14` Implement required immutable blueprint slot IDs, contrastive slot schema, and blueprint hashing.
- [ ] `CASCADE-15` Run replacement blueprint inference for all 50 topics under a new run identity.
- [ ] `CASCADE-16` Expand every blueprint slot into one bound full scenario without semantic substitution.
- [ ] `CASCADE-17` Run independent whole-topic scrutiny over each complete 20-scenario set.
- [ ] `CASCADE-18` Emit typed clustered findings keyed to rejected blueprint slot IDs.
- [ ] `CASCADE-19` Repair only rejected slots while preserving the blueprint and accepted sibling rows.
- [ ] `CASCADE-20` Re-audit every repaired whole-topic set rather than certifying repairs individually.
- [ ] `CASCADE-21` Certify all 50 complete topic sets before trajectory generation.
- [ ] `CASCADE-22` Publish an immutable replacement scenario certificate with counts, findings, model identity, prompts, schema hashes, and report ID.

## E. Operational observation discipline

- [x] `OPS-01` Timestamp every process, checkpoint, and GPU observation.
- [x] `OPS-02` Query generator PIDs with full stage and output-root command context before reporting activity.
- [x] `OPS-03` Count unique immutable checkpoint JSON files directly from each stage directory.
- [x] `OPS-04` Distinguish partial checkpoint count from canonical JSONL stage completion.
- [x] `OPS-05` Hash canonical stage artifacts and run identity files when assessing completion.
- [x] `OPS-06` Sample GPU utilization repeatedly rather than inferring inactivity from one quiet instant.
- [x] `OPS-07` Inspect GPU compute processes separately because the cascade orchestrator may not hold VRAM.
- [ ] `OPS-08` Keep active-run monitoring reports outside normative architecture claims.
- [ ] `OPS-09` Mark a stage complete only after expected cardinality, canonical artifact, manifest, and hashes agree.

## F. Selection and factorized causal groups

- [x] `GROUP-01` Implement deterministic typed balancing over all eligible candidates.
- [x] `GROUP-02` Implement 250 primary and 250 reserve slots.
- [x] `GROUP-03` Implement typed deterministic reserve replacement.
- [x] `GROUP-04` Require exactly four roles: positive, negative, uncertain, and superseded.
- [x] `GROUP-05` Require one changed intervention family per atomic group.
- [x] `GROUP-06` Require shared prefix, voice pair, pivot, template, and lineage identity.
- [x] `GROUP-07` Require wrong-branch, stale-revision, and null-control negatives.
- [x] `GROUP-08` Keep target text/audio exclusively on the label side.
- [ ] `GROUP-09` Execute balanced selection on the complete 10,000-candidate lattice.
- [ ] `GROUP-10` Fully expand all 500 selected candidate groups.
- [ ] `GROUP-11` Materialize four sibling specifications for the active 250 primary groups.
- [ ] `GROUP-12` Certify repeated operator support across enough unrelated premises in every split.
- [ ] `GROUP-13` Bound composite groups to the preregistered minority after atomic coverage passes.

## G. Compiler and voice provenance

- [x] `COMPILE-01` Emit schema-7 render plans from selected v5 groups.
- [x] `COMPILE-02` Preserve full target-free `commonContext` and verify `commonContextHash`.
- [x] `COMPILE-03` Emit typed control frames with positive revisions and frame hashes.
- [x] `COMPILE-04` Emit strict-before-target pivot bindings.
- [x] `COMPILE-05` Emit content-addressed shared-prefix sidecars.
- [x] `COMPILE-06` Emit target-free `postRenderBridge` and `renderPlanId`.
- [x] `COMPILE-07` Prohibit target response leakage in plans and bridges.
- [ ] `COMPILE-08` Rebuild and certify the approved voice-reference manifest at the request-bound location.
- [ ] `COMPILE-09` Verify rights, provenance, duration, speech presence, speaker identity, and audio hashes for every voice reference.
- [ ] `COMPILE-10` Compile all active primary groups and required reserve replacements.
- [ ] `COMPILE-11` Verify exactly 1,000 active branch plans and 250 shared-prefix records before rendering.

## H. Voryn render-once shared-prefix transaction

- [x] `RENDER-01` Assign each complete four-sibling group to one lane.
- [x] `RENDER-02` Enforce physical CUDA lane mapping `0`, `1`, or `2`.
- [x] `RENDER-03` Capture the first sibling's exact pre-pivot duplex snapshot.
- [x] `RENDER-04` Deep-freeze, fingerprint, and replay the snapshot for the other three siblings.
- [x] `RENDER-05` Mark replayed prefix records context-only and ineligible for target loss.
- [x] `RENDER-06` Reject divergent prefixes, incomplete groups, invalid lineage, and early or non-model termination.
- [x] `RENDER-07` Commit group bundle and progress atomically only after all four siblings pass.
- [x] `RENDER-08` Consume compiler-provided typed control frames unchanged.
- [ ] `RENDER-09` Run the complete v5 primary render across CUDA devices `0,1,2`.
- [ ] `RENDER-10` Record real overlap, cutoff, barge-in, cancellation, and recovery timing in certified timelines.
- [ ] `RENDER-11` Run Whisper transcript and word-timing admission on every rendered target.
- [ ] `RENDER-12` Repair only failed suffixes or replace the complete typed group; never splice unrelated prefixes.
- [ ] `RENDER-13` Keep total rejection below the operational target without loosening severe audio or semantic gates.
- [ ] `RENDER-14` Confirm every accepted conversation ends with one model-selected `end_call`, not deterministic goodbye text.

## I. Post-render bridge and native lineage

- [x] `BRIDGE-01` Finalize one `personaplex.voryn-branch-artifact.v5` per certified sibling pivot.
- [x] `BRIDGE-02` Match observed frame hash and revision to the immutable render plan.
- [x] `BRIDGE-03` Require control availability frame strictly before native pivot frame.
- [x] `BRIDGE-04` Derive timing, interruption, cancellation, recovery, and termination from certified events.
- [x] `BRIDGE-05` Hash the complete immutable artifact as `planRecordId`.
- [x] `BRIDGE-06` Propagate `planRecordId` through examples, precodec provenance, labels, and native inputs.
- [x] `BRIDGE-07` Reject target transcript leakage into common context or control inputs.
- [ ] `BRIDGE-08` Execute the exporter against all certified v5 group bundles.
- [ ] `BRIDGE-09` Verify exactly four artifacts per admitted group and no orphan plan or timeline.
- [ ] `BRIDGE-10` Publish an immutable lineage report from request hash through native tensor hashes.

## J. Native encoding, materialization, and splits

- [x] `NATIVE-01` Fix the native frame contract at 24 kHz, 12.5 Hz, and 80 ms.
- [x] `NATIVE-02` Encode target-free controls separately from target labels.
- [x] `NATIVE-03` Supervise only current audible agent text/audio frames.
- [x] `NATIVE-04` Store one native shared prefix per group plus four suffix/mask/control records.
- [x] `NATIVE-05` Reject sidecar, alignment, cancellation, and model-contract mismatches.
- [x] `NATIVE-06` Build union-find leakage components across lineage, template, operator, and voice pair.
- [x] `NATIVE-07` Keep each component and all four siblings in one split.
- [x] `NATIVE-08` Emit immutable common-input, listwise, diagnostics, component, certificate, and manifest artifacts.
- [ ] `NATIVE-09` Encode all accepted v5 audio on CUDA devices `0,1,2` without CPU fallback.
- [ ] `NATIVE-10` Certify the complete encoded native corpus.
- [ ] `NATIVE-11` Materialize all native v5 groups with zero unexplained rejections.
- [ ] `NATIVE-12` Resolve every typed rerender rejection at source and rerun only the affected group.
- [ ] `NATIVE-13` Pack train, validation, and test with repeated operator coverage in every required split.
- [~] `NATIVE-14` Freeze and hash the final trainer dataset contract and manifests; mandatory hash-chain admission is implemented, final v5 artifacts remain pending.

## K. Full-rank native training

- [x] `TRAIN-01` Implement CUDA/NCCL-only admission and forbid CPU model fallback/offload.
- [x] `TRAIN-02` Discover GPU and host capacity at runtime.
- [x] `TRAIN-03` Throttle host memory only above the configured 80 percent used-memory limit.
- [x] `TRAIN-04` Implement FSDP over the full temporal/text receiver and native conditioner.
- [x] `TRAIN-05` Keep ARC, Mimi, audio embeddings, depth transformer, audio heads, and voice path frozen for the first run.
- [x] `TRAIN-06` Implement agent-only native text/audio likelihood.
- [x] `TRAIN-07` Implement the four-by-four listwise causal objective.
- [x] `TRAIN-08` Implement pre-response state probing, control dropout, null control, stale control, and wrong-branch negatives.
- [x] `TRAIN-09` Keep train and validation telemetry separate and retain test for final evaluation.
- [x] `TRAIN-10` Implement complete immutable checkpoints at steps 100, 125, and 150.
- [x] `TRAIN-11` Enforce a mandatory pre-`torchrun` certified-pack gate binding source data, leakage/coverage certificate, split assignment, trainer manifests, and model contract.
- [ ] `TRAIN-12` Inspect steps 100, 125, and 150 for held-out full-group progress and mode collapse.
- [ ] `TRAIN-13` Stop or redesign if train sensitivity rises while held-out full-group performance stays flat.
- [ ] `TRAIN-14` Select a checkpoint without consulting test outcomes.
- [ ] `TRAIN-15` Run the retained test split once after checkpoint selection.
- [!] `TRAIN-16` Do not label a teacher-forced 95 percent gate as generated or live convergence.

## L. Generated native evaluation

- [ ] `EVAL-01` Implement a v5 free-running native generation harness bound to an exact checkpoint hash.
- [ ] `EVAL-02` Evaluate all four roles from byte-identical held-out prefixes.
- [ ] `EVAL-03` Score semantic adherence without using target text as model input.
- [ ] `EVAL-04` Score factual and tool-result incorporation separately from conversational quality.
- [ ] `EVAL-05` Score policy constraints, uncertainty, correction, and superseding revisions.
- [ ] `EVAL-06` Score stale-control rejection and newest-revision acknowledgement.
- [ ] `EVAL-07` Execute real interruption, queued-audio cancellation, cutoff, and recovery.
- [ ] `EVAL-08` Measure first-audio latency, frame cadence, sustained real-time factor, and queue depth.
- [ ] `EVAL-09` Measure Whisper agreement, intelligibility, clipping, silence, codec validity, and channel integrity.
- [ ] `EVAL-10` Measure voice preservation and expressive naturalness without rewarding semantic errors.
- [ ] `EVAL-11` Include unseen topics, voices, entities, lexical forms, operators, and operator compositions.
- [ ] `EVAL-12` Include adversarial near-neighbor frames that differ by one material field.
- [ ] `EVAL-13` Include null, stale, late, malformed, duplicate, and out-of-order control updates.
- [ ] `EVAL-14` Use independent adjudication and calibrated human review for ambiguous outcomes.
- [ ] `EVAL-15` Predeclare sample sizes, stratum thresholds, and confidence interval method.
- [ ] `EVAL-16` Require the aggregate and every safety-critical stratum to meet the 95 percent criterion.

## M. Runtime control and live Twilio evaluation

- [ ] `LIVE-01` Bind every call session to monotonic `control_revision`, acknowledged revision, immutable generation snapshot, generation ID, and cancellation token.
- [ ] `LIVE-02` Reject stale, duplicate, malformed, wrong-call, and unavailable control frames.
- [ ] `LIVE-03` Encode each accepted control revision once and cache it on GPU.
- [ ] `LIVE-04` Snapshot the acknowledged control representation only at safe agent-turn boundaries.
- [ ] `LIVE-05` Cancel generated audio and invalidate generation ID immediately on barge-in.
- [ ] `LIVE-06` Clear stale queued Twilio media before the recovery response.
- [ ] `LIVE-07` Ensure the recovery response uses the newest acknowledged revision.
- [ ] `LIVE-08` Permit only wait or validated safe backchannel behavior when no valid current frame exists.
- [ ] `LIVE-09` Route exact-wording requirements to the separately validated strict renderer.
- [ ] `LIVE-10` Score strict-renderer routing separately from PersonaPlex semantic realization.
- [ ] `LIVE-11` Run preregistered live Twilio trials across the required semantic and duplex strata.
- [ ] `LIVE-12` Meet the 95 percent live reliability criterion and confidence bound without post-hoc subset selection.

## N. Release

- [ ] `RELEASE-01` Freeze code, request, catalog, schemas, voice manifest, dataset, split, checkpoint, and evaluation hashes.
- [ ] `RELEASE-02` Publish dataset provenance, exclusions, known limitations, and license metadata.
- [ ] `RELEASE-03` Publish checkpoint architecture, resource requirements, runtime protocol, and strict-renderer boundary.
- [ ] `RELEASE-04` Publish generated and live evaluation methods with failures, denominators, and confidence intervals.
- [ ] `RELEASE-05` Publish deployment scripts that use future-proof `/srv/voxrn_cache` resource roots.
- [ ] `RELEASE-06` Publish only after every required generated and live gate is complete.
