# PersonaPlex Ground-Truth Completion Ledger

This ledger is the execution authority for the steerable PersonaPlex program.
`ARCHITECTURE.md`, `RUNTIME_CONTROL.md`, `TRAINING.md`, and the versioned schemas
define requirements; this file tracks implementation, verification, and promotion.

## Status vocabulary

- `[x]` Implemented and verified with the required artifact.
- `[~]` Implemented locally but not yet verified to the required gate.
- `[ ]` Not implemented.
- `[!]` Known defect, conflict, or missing prerequisite. It blocks promotion.
- `[N/A]` Explicitly not applicable, with a recorded reason.

No `[~]` item may be described as working in production. Every completed item must
link to or name a reproducible artifact under `reports/`, `evaluation/`, or a
versioned run directory. Raw calls, secrets, reference audio, and sensitive state
must never be committed as evidence.

## MoshiRAG-first critical path

### Delayed-evidence control implementation ledger (2026-07-15)

- [x] Establish the Moshirag reference contract locally: `T x 4096` streaming-sum evidence rows, consumed one per generation step.
- [x] Confirm the current upstream PersonaPlex source lacks `condition_streaming_sum`, pending per-slot evidence, and update/apply APIs.
- [x] Add a target-wording-free delayed-evidence training schema with revision, context, timing, provenance, and counterfactual identity constraints.
- [x] Add a trainable evidence-stream adapter and native agent-only loss path that fails closed without the maintained upstream conditioning patch.
- [x] Add a reproducible upstream patch/bootstrap artifact; it is not applied to any live deployment automatically.
- [ ] Add Voryn V7 synthesis/export examples with aligned `EvidenceTrainingFrame` records, two valid counterfactual branches per group, and no target-wording leakage.
- [ ] Validate V7 corpus structure, temporal causality, duplex code alignment, and counterfactual pairing before codec/ASR admission.
- [ ] Apply the maintained patch to an isolated CUDA-only PersonaPlex source environment and fingerprint the resulting source/model contract.
- [ ] Train prefix stage on an adequately sized certified split; freeze and promotion-gate the accepted checkpoint.
- [ ] Train evidence-stream stage against delayed V7 frames with the prefix frozen; reject regressions to turn-taking, voice quality, or base no-control behavior.
- [ ] Run held-out factual/tool-result, policy-change, and interruption-recovery counterfactual evaluation with semantic judges and real-time latency measurements.
- [ ] Only after promotion: enable `evidence.update` beyond `evidence_deferred` in the controlled server, gated by the trained checkpoint/model/source fingerprints.

- [x] Research reconciliation now distinguishes the primary MoshiRAG semantic
  conditioning reference from MisoTTS expressive-rendering research.
  Proof: `MOSHI_RESEARCH_AND_CONTROL_RECONCILIATION.md` and
  `SEMANTIC_CONTROL_CONVERGENCE_PLAN.md`.
- [~] Official `kyutai/moshika-rag-pytorch-bf16` artifact download is staged at
  `/srv/voxrn_cache/moshi-rag/kyutai-moshika-rag-pytorch-bf16`.
  The released configuration is inspected: frozen ARC4 text compression,
  `3072 -> 2048 -> 4096` bridge, 12.5 Hz streaming-sum fuser, and 20% reference
  dropout. Required proof: completed content-addressed artifact manifest and
  CUDA load contract; staging/config inspection alone is not validation.
- [~] `evidence.update` V2 schema and controlled-server transport now validate
  provenance, supporting control revision, context hash, TTL, allowed claims,
  availability, and terminal acknowledgement. It cancels stale output and is
  deliberately deferred until a trained evidence encoder exists.
  Proof: `personaplex_control/runtime.py`, `controlled_server.py`; structural
  harness accepted a valid update, rejected a prompt field, and emitted
  `superseded` plus `evidence_deferred`.
  Required remaining proof: complete accepted/rejected protocol matrix and a
  native CUDA evidence-adapter harness.
- [ ] Persist evidence receipt and expiry in the per-call event log without
  retaining raw sensitive tool payloads on the audio plane.
  Required proof: redacted trace and retention-policy test.
- [ ] Add evidence availability timelines and one-field state/evidence
  counterfactuals to certified synthesis plans.
  Required proof: lineage-locked counterfactual-pair report.
- [ ] Train a frozen-base control-prefix adapter before enabling learned evidence
  injection.
  Required proof: held-out prefix evaluation with a non-empty train/validation/
  test split.
- [ ] Add a separately trainable evidence encoder only after the prefix adapter
  passes semantic adherence, freshness, and preservation gates.
  Required proof: native CUDA evidence harness proves causal effect without
  stale-media emission.
- [ ] Build the custom Nemotron state/control compiler corpus from the same
  event logs, excluding PersonaPlex target utterance labels.
  Required proof: held-out typed-frame SFT/evaluation suite.
- [ ] Keep MisoTTS out of certified training data until CUDA-only A/B evaluation
  versus Chatterbox Turbo covers ASR, word timing, codec behavior, watermark,
  provenance, and license compliance.
  Required proof: renderer decision record; Miso is not a duplex replacement.

## Current factual status corrections

- [~] One six-turn V6 GPU0 conversation passed inline and independent batch
  semantic certification, including terminal action metadata.
  Proof: `/srv/voxrn_cache/personaplex-lanes/gpu0/datasets/synthesize/synth_20260716010700161_066c360f.certified.jsonl`.
  It is a pilot only and insufficient for train/validation/test training.
- [x] The Voryn certified exporter and precodec preparation path replace the
  older flattened exporter for the V6 pilot.
  Proof: `/srv/personaplex_workspace/ground_truth_runs/voryn-verified-v6-export-20260716` and
  `/srv/personaplex_workspace/ground_truth_runs/voryn-verified-v6-precodec-20260716`.
- [x] The distilled `student_best.pt` native model contract was CUDA-loaded on
  GPU 0 with actual stream layout and delays recorded.
  Proof: `/srv/personaplex_workspace/ground_truth_runs/personaplex-student-best.cuda0.contract.v2.json`.

## Gate 0: Program invariants and evidence discipline

- [x] The architecture, runtime protocol, and training program are anchored in
  versioned ground-truth documents.
  Proof: `ARCHITECTURE.md`, `RUNTIME_CONTROL.md`, `TRAINING.md`.
- [x] Control plans prohibit canonical response fields.
  Proof: `training/contracts.py::ControlPlan.from_mapping`.
- [~] `ControlTrainingFrame` schema, contracts, and deterministic serializer
  exist and reject canonical target-text field names.
  Required proof: unit test with accepted frame plus canonical-text rejection.
- [ ] Add a machine-readable requirements-to-test matrix.
  Required proof: `reports/requirements_matrix.json` covering every checklist ID.
- [ ] Add immutable run IDs, base-model revision, code revision, data-root hash,
  and environment fingerprint to every exporter, training, and evaluation report.
  Required proof: one complete dry-run run card.
- [ ] Add a redaction/no-secret scan for manifests, logs, and reports.
  Required proof: scan report with zero high-severity findings.
- [ ] Define retention and access controls for raw reference audio, generated
  training audio, semantic state, and tool evidence.
  Required proof: approved storage-policy record.

## Gate 1: Approved voices and Chatterbox Turbo audio plane

- [x] A provenance-gated reference-bank format accepts only explicit cloning
  consent or documented public-domain dedication, with source/license/hash.
  Proof: `lib/voiceReferenceBank.js` and `/srv/voxrn_cache/chatterbox-reference-bank/manifest.json`.
- [x] 48 normalized 5-9 second LibriVox public-domain reference clips were built
  under `/srv/voxrn_cache/chatterbox-reference-bank`.
  Proof: reference-bank manifest and SHA-256 entries.
- [x] Two direct Chatterbox Turbo clone renders were played locally and passed
  Whisper calibration: WER `0.00` and `0.20`.
  Proof: local calibration event log from 2026-07-15.
- [~] Chatterbox is wired as `voicebox_chatterbox_turbo` for synthetic turns.
  Required proof: complete admitted ground-truth call using distinct bank voices.
- [~] Voicebox render timeout is propagated to synthetic turns with a 180-second
  cold-render allowance.
  Required proof: timeout regression test and one cold-start + warm-start report.
- [ ] Verify clone similarity/diversity without identifying real people beyond
  approved reference IDs.
  Required proof: consent-scoped speaker-embedding aggregate report.
- [ ] Verify telephony codec path: 24 kHz render -> 8 kHz PCM -> mu-law ->
  decoded waveform, with clipping and level thresholds.
  Required proof: codec test-vector report and WAV hashes.
- [ ] Verify noise, packet-loss, jitter, and packet-reordering robustness on
  generated audio before Twilio promotion.
  Required proof: emulation report with pass/fail thresholds.
- [ ] Define and enforce synthetic-turn loudness, silence, duration, and DC-offset
  thresholds.
  Required proof: audio-quality validator report.

## Gate 2: Synthetic conversation semantics and diversity

- [~] Implement the agent-operable diverse synthesis cascade: a versioned request
  expands into 50 topic cards, 20 scenario contracts per topic, 10 trajectory seeds
  per scenario, deterministic quota-selected groups, and model-generated causal pair
  specifications without target-label leakage.
  Proof: `DIVERSE_SYNTHESIS_CASCADE_CONTRACT.md`,
  `schemas/diverse_corpus_request.schema.json`,
  `schemas/diverse_cascade_artifacts.schema.json`,
  `tools/build_diverse_synthesis_cascade.py`, and
  `tools/validate_diverse_synthesis_cascade.py`, and
  `tools/compile_diverse_cascade_voryn_plan.py`.
  Required remaining proof: a full 10,000-unit planning run, structural validation
  report, Voryn bridge export, and independent source certification of the selected
  500 counterfactual groups.

- [x] The 1,000-conversation plan allocates 48 approved voices, 1,128 possible
  unordered pairs, varied topics, openings, closings, coverage profiles, and seeds.
  Proof: `/srv/personaplex_workspace/ground_truth_runs/personaplex-1000-plan.jsonl`.
- [x] The generator defaults to free `nvidia/nemotron-3-nano-30b-a3b:free` with
  reasoning disabled and a bounded spoken-output contract.
  Proof: `lib/syntheticConversations.js`; direct provider call returned a spoken reply.
- [x] Generator settings include assertiveness, skepticism, compliance,
  resistance, interruption, recovery, hesitation, and pace dimensions.
  Proof: `normalizeSynthesisSettings` and `buildRunDynamics`.
- [~] Generator emits a rolling call-state representation and per-target
  `ControlTrainingFrame` before the target response.
  Required proof: schema-validated admitted run with monotonic revision/state hash chain.
- [~] Generator gives a caller barge-in only the audible prefix of the prior agent
  response through `historyOverride`.
  Required proof: deterministic interruption fixture showing prefix truncation.
- [~] Generator records logical and audible end times, overlap, cancellation
  latency, and recovery expectation in `voxrn.duplex-timeline.v1`.
  Required proof: timeline validator plus materialized audio exhibiting the cutoff.
- [~] Generator rejects WER, confidence, missing word alignment, missing audio,
  and invalid audible timing.
  Required proof: pass/fail fixtures for every rejection reason.
- [x] Dataset persistence is rooted under `/srv/voxrn_cache`; the synthesis
  subtree is writable by the local generator.
  Proof: corrected ownership for `/srv/voxrn_cache/datasets/synthesize`.
- [~] One six-turn V6 ground-truth pilot has passed inline and batch semantic
  certification. Three-GPU profile coverage, corpus-level diversity, and held-out
  split gates are still incomplete.
  Exit criterion: one certified call per coverage profile on each allowed A100
  before scale-up.
- [ ] Add anti-duplication ledgers for normalized openings, closings, intents,
  recovery patterns, and voice-pair reuse across every shard.
  Required proof: corpus-level collision report.
- [ ] Add diversity quotas and held-out splits for domain, interaction trajectory,
  voice pair, opening/closing, interruption type, and policy route.
  Required proof: deterministic split manifest and quota report.
- [ ] Add adversarial but safe scenarios: refusal, uncertainty, verification,
  correction, escalation, handoff, and policy-constrained alternatives.
  Required proof: scenario taxonomy with no prohibited sensitive-data generation.
- [ ] Add counterfactual state/control pairs that vary exactly one control field and
  regenerate valid target audio/text rather than reusing labels.
  Required proof: counterfactual-pair validator report.

## Gate 3: Persistent three-A100 generation service

- [x] GPU policy is restricted to A100 devices `0`, `1`, and `2`; GPU `3` is
  excluded from Voicebox, Whisper, generation, and training commands.
  Proof: lane launch contract and command environment.
- [ ] Add a persistent local lane supervisor outside the agent command lifecycle.
  The current background shell workers are reclaimed before writing logs.
  Required proof: `systemd --user`, tmux, or equivalent service status for all
  three lane workers, with restart policy and per-lane logs.
- [ ] Create one isolated lane per A100 with distinct Voicebox port, SQLite data
  directory, profile map, generated-audio directory, and log path.
  Required proof: lane manifest showing GPU/port/data-root uniqueness.
- [ ] Share only immutable model/cache paths across lanes and verify concurrent
  Hugging Face cache safety.
  Required proof: three-lane cold-start report with no cache corruption.
- [ ] Add GPU admission checks for free memory, active external workload, thermal
  state, and per-lane concurrency of one active Chatterbox job.
  Required proof: admission snapshot embedded in each lane run card.
- [ ] Add bounded retry/cancellation behavior that marks Voicebox jobs terminal
  rather than leaving `generating` or orphaned rows.
  Required proof: injected backend-restart test report.
- [ ] Add a shard coordinator that atomically leases plan rows, records attempt
  history, and never duplicates a certified conversation.
  Required proof: coordinator SQLite/JSONL lease ledger and duplicate scan.
- [ ] Run one certified four-turn candidate on each A100 before increasing shard
  size.
  Required proof: three manifests plus ASR/timing/control-frame reports.
- [ ] Increase only after the pilot gate: 3 -> 15 -> 150 -> 1,000 calls, with a
  fresh quality/diversity report at each promotion.
  Required proof: four promotion reports and immutable corpus manifests.

## Gate 4: Duplex timeline materialization and corpus certification

- [ ] Implement a timeline validator for monotonic event order, valid overlaps,
  audible cutoffs, cancellation latency, and causal recovery turns.
  Required proof: valid/invalid fixture suite.
- [ ] Materialize `voxrn.duplex-timeline.v1` into separate caller/agent tracks at
  24 kHz, preserving declared offsets and overlap.
  Required proof: deterministic mixer output hashes and channel-layout report.
- [ ] Crop agent target audio at barge-in audible cutoff before Mimi encoding.
  Required proof: waveform/length assertion showing no queued suffix remains.
- [ ] Keep prior agent audio as context only and target only the current agent
  utterance in the loss mask.
  Required proof: target-mask visualizer and unit test.
- [ ] Convert and audit telephony-compatible source audio before 24 kHz duplex
  materialization; reject clipped or non-finite samples.
  Required proof: codec/loudness validation report.
- [ ] Require word-level Whisper alignment for all supervised agent text and
  preserve caller alignment as context metadata.
  Required proof: alignment coverage report.
- [ ] Create a source-to-export lineage record tying audio SHA-256, control-frame
  hash, plan hash, timeline hash, and split assignment together.
  Required proof: lineage verifier output.
- [x] `export_controlled_duplex_dataset.py` exports certified Voryn records
  without placing canonical target text into the control input. The old flattened
  exporter remains deprecated and must not be used for this programme.
  Proof: V6 export run under
  `/srv/personaplex_workspace/ground_truth_runs/voryn-verified-v6-export-20260716`.
- [ ] Update `validate_dataset.py` to validate ControlTrainingFrame, timeline,
  state-hash chain, control-frame hash, split isolation, and no-text-leakage.
  Required proof: complete valid corpus and negative fixtures.
- [ ] Produce a corpus certificate only after every item passes provenance,
  audio/ASR, timeline, semantic, codec, and split gates.
  Required proof: signed/root-hashed certificate with rejection counts.

## Gate 5: Native PersonaPlex semantic-prefix training

- [x] Native delayed-code forward path and agent-only loss primitives exist.
  Proof: `training/native_training.py` and `training/trainer.py`.
- [~] Control-frame contract and serializer are wired into the train entrypoint.
  Required proof: tokenizer/adapter batch test using a certified frame.
- [ ] Add tests for serializer determinism, field ordering, bounded length,
  forbidden-field rejection, and state-hash/plan-hash mismatch rejection.
  Required proof: pytest report.
- [ ] Implement batch construction from materialized duplex timelines and native
  Mimi/Moshi code streams; discover stream layout from the loaded model.
  Required proof: encoded batch inspection report for the actual checkpoint.
- [ ] Verify delay/undelay, `forward_codes`, depformer layout, and target masking
  against the chosen PersonaPlex checkpoint before any optimizer step.
  Required proof: native-code equivalence test.
- [x] The compatible distilled PersonaPlex hybrid checkpoint contract is pinned,
  SHA-256 recorded, and CUDA-loaded with actual stream layout and delay values.
  Proof: `/srv/personaplex_workspace/ground_truth_runs/personaplex-student-best.cuda0.contract.v2.json`.
- [ ] Run frozen-base adapter smoke training on one short certified shard using
  memory-aware BF16 settings.
  Required proof: run card, loss components, gradient checks, samples.
- [ ] Run a three-A100 distributed adapter epoch only after the smoke gate passes.
  Required proof: DDP run card, rank metrics, memory/utilization report.
- [ ] Add plan-adherence auxiliary/contrastive loss without canonical wording.
  Required proof: counterfactual evaluation showing one-field sensitivity.
- [ ] Add baseline-preservation/KL evaluation for neutral frames.
  Required proof: frozen-base regression report.
- [ ] Define checkpoint promotion thresholds for semantic adherence, latency,
  naturalness, interruption, state freshness, and no-hallucinated commitments.
  Required proof: machine-readable acceptance policy.
- [ ] Consider bounded LoRA/top-layer adaptation only after frozen adapter meets
  the promotion gate.
  Required proof: documented ablation and rollback decision.

## Gate 6: PersonaPlex server control-plane integration

- [~] Implemented `SemanticPrefixProvider` against upstream PersonaPlex commit
  `3428dfd95309a7f3c84fd93259ded0f810d1ff91`: validate, GPU-cache, direct
  transformer prefill, and post-prefill acknowledgement.
  Required proof: `evaluation/runtime_prefix_harness.py` on CUDA with a trained
  adapter and a batch-certified V2 frame.
- [~] Bound V2 `ControlTrainingFrame` transport validation to the runtime
  protocol; the audio plane rejects raw prompts and canonical target wording.
  Required proof: accepted typed update plus forbidden-field, stale, duplicate,
  mismatch, expiry, and malformed-wire matrix.
- [ ] Implement state-reducer ownership and bounded patches from task, policy,
  knowledge, and safety agents.
  Required proof: revision/hash-chain test with concurrent patch proposals.
- [~] Runtime session implements monotonic revisions, immutable cached prefixes,
  expiry, duplicate identity acknowledgement, context matching, and generation
  IDs.
  Required proof: duplicate/stale/mismatch/expiry protocol matrix.
- [~] Updates apply only at explicit caller-turn boundaries via direct streaming
  transformer prefill; no model-history reset occurs.
  Required proof: causal-state preservation test against the pinned model.
- [~] Caller barge-in invalidates pending/active revisions and all unsent media
  generation IDs.
  Required proof: injected barge-in trace with no stale post-cutoff media.
- [~] Runtime emits `queued`, `applied`, `superseded`, `rejected`, `expired`,
  `context_mismatch`, `prefix_build_failed`, and `safe_fallback` statuses.
  Required proof: complete terminal-status harness report.
- [ ] Measure prefix encoding/prefill deadline and fail closed to safe fallback.
  Required proof: latency-percentile report and forced-deadline fixture.
- [ ] Version adapter, serializer, state schema, and base model in every control
  acknowledgement and call trace.
  Required proof: protocol trace sample.
- [!] Existing bridge overlay accepted experimental control messages but did not
  prove a real prefix application acknowledgement.
  Exit criterion: protocol harness receives `applied` only after actual prefill.

## Gate 7: Twilio bidirectional audio plane and strict mode

- [ ] Implement/verify Twilio bidirectional stream -> mu-law/PCM bridge ->
  PersonaPlex duplex stream -> paced outbound media path.
  Required proof: deterministic media-loopback test vector.
- [ ] Implement stream timestamps, marks, bounded buffering, and stale-media
  dropping tied to control revisions.
  Required proof: media timing trace.
- [ ] Implement VAD/ASR partial/final events and caller-turn boundary policy.
  Required proof: timing suite with silence, overlap, and noisy caller fixtures.
- [ ] Implement barge-in clear/cancel semantics across Twilio buffer, PersonaPlex
  generation, and semantic state reducer.
  Required proof: end-to-end interruption capture.
- [ ] Add degraded-mode policy for ASR uncertainty, semantic timeout, codec error,
  stream disconnect, and observability failure.
  Required proof: failure-injection report.
- [ ] Route exact-language requirements to a separately validated strict renderer;
  never label expressive PersonaPlex output as verbatim-guaranteed.
  Required proof: routing and ASR comparison report.
- [ ] Add per-call isolation for state, prefix cache, voice reference, tool data,
  and media queues.
  Required proof: concurrent-call cross-talk test.
- [ ] Validate telephone realism: first-audio latency, p50/p95 turn latency,
  interruption stop latency, jitter tolerance, and audible clipping.
  Required proof: emulated-call KPI report.

## Gate 8: Evaluation, promotion, and product readiness

- [ ] Build fixed held-out evaluation suites for semantic adherence, required
  questions, forbidden claims, entity recall, uncertainty, refusal, handoff,
  interruption, recovery, and strict route correctness.
  Required proof: versioned suite manifest.
- [ ] Evaluate counterfactual control sensitivity with identical conversation/audio
  context and one changed frame field.
  Required proof: blinded scorer report.
- [ ] Evaluate voice consistency/diversity only within approved consent scope.
  Required proof: aggregate, non-identifying report.
- [ ] Evaluate latency and quality against the frozen PersonaPlex baseline.
  Required proof: baseline/candidate comparison card.
- [ ] Run Twilio failure-mode emulation after the in-process control harness and
  duplex exporter both pass.
  Required proof: scenario trace bundle and pass/fail table.
- [ ] Add dashboards for applied revision rate, stale-plan rejection, fallback
  rate, barge-in stop latency, ASR/WER distribution, codec failures, and queue age.
  Required proof: dashboard schema and sample aggregate data.
- [ ] Verify PersonaPlex agent selection and configuration are visible in the
  agent-creation flow without misrepresenting the runtime as semantically proven.
  Required proof: UI route screenshot/test plus capability-state copy review.
- [ ] Publish run cards, corpus cards, model cards, limitations, deployment
  checklist, rollback plan, and operational runbook before production exposure.
  Required proof: release packet with all required links.

## Immediate ordered execution queue

1. Replace ephemeral lane background processes with a persistent three-A100 local
   supervisor and capture its health evidence.
2. Complete one four-turn candidate on each lane and fix every strict admission
   failure before increasing volume.
3. Add synthetic control-frame/timeline validators and deterministic fixtures.
4. Replace the legacy exporter with a Voryn v3 duplex timeline exporter.
5. Certify a small, diverse corpus shard and inspect native codec/model layout.
6. Run frozen-adapter smoke training, then the three-A100 epoch only after the
   smoke report passes.
7. Implement and prove actual prefix prefill/acknowledgement in the server fork.
8. Exercise Twilio media/control emulation only after the preceding local gates pass.
# 2026-07-15 local evidence and newly reconciled blockers

- [x] `E-001` Three isolated Chatterbox Turbo workers were exercised on CUDA devices `0`, `1`, and `2` only; each rendered caller and agent audio, ran Whisper, and played local previews. GPU `3` was not selected.
- [x] `E-002` Voryn v3 output was inspected: approved reference provenance, Whisper word timings, typed `ControlTrainingFrame`, `frameHash`, state/context hashes, `audibleEndedAtMs`, and per-conversation duplex timeline sidecars are present.
- [x] `E-003` The strict generation gate rejected two pilot conversations for ASR WER/confidence failure rather than silently admitting them.
- [!] `B-001` The only admitted high-interruption pilot contains a terminal agent truncation but no actual following caller overlap or recovery agent turn. It is diagnostic evidence only, not trainable full-duplex coverage.
- [!] `B-002` That pilot's terminal label claims completion while its control frame forbids an unverified completion claim and requests confirmation. Semantic-plan adherence must be an explicit admission gate.
- [~] `I-001` Added `tools/export_controlled_duplex_dataset.py`: strict native 24 kHz stereo exporter, target-label/control separation, interruption/recovery admission checks, and diagnostic-only incomplete mode.
- [~] `I-002` Added `tools/validate_controlled_duplex_dataset.py`: verifies 24 kHz two-channel materialization and blocks label/canonical-response leakage into control input.
- [ ] `N-001` Add generation-time semantic-plan adherence validation before a target turn can be marked `training.eligible`.
- [ ] `N-002` Change synthetic interruption scheduling so high-interruption scenarios require an agent target turn followed by an overlapping caller barge-in and an explicit recovery target turn; terminal truncation is invalid.
- [ ] `N-003` Run the exporter in strict mode on one repaired high-interruption conversation and promote its evidence only after export validation passes with zero diagnostics.
- [~] `I-003` Replaced topic-only corpus planning with a versioned executable coverage assignment: dialogue act, interaction class, speech style, turn pattern, dynamic band, required state fields, update source, and barge-in/recovery requirement.
- [~] `I-004` Generator now persists the broader state tree, forces non-terminal overlap/recovery for assigned rows, and rejects control/label contradictions such as unverified completion claims.
- [ ] `N-004` Generate a fresh version-2 plan and verify every taxonomy bucket and control-source quota in its summary before starting the 1,000-conversation workers.
- [ ] `N-005` Run one high-interruption pilot through generation, strict duplex export, and independent validation; it must have zero diagnostic examples before it is counted as training material.
- [~] `I-005` Added bounded seed-varied regeneration per coverage assignment. A rejected attempt remains logged and excluded; the assignment is satisfied only by an admitted strict run.
- [~] `I-006` WER canonicalization now treats equivalent clock/number forms such as `three p.m.`, `3 PM`, and `3pm` consistently without hiding lexical substitutions.
- [~] `I-007` Lane wrapper now locks each GPU lane, reclaims only a verified stale private Voicebox listener, respects row-level turn counts, and passes bounded retry configuration to production workers.
- [~] `I-008` Planner constraints and multiword entity values are normalized into symbolic control atoms before entering a frame, preventing natural target wording from being supplied to the prefix adapter.
- [~] `I-009` Required-question admission now recognizes natural interrogative syntax, not only a text punctuation mark, so it measures spoken behavior rather than formatter choice.
- [x] `E-004` Certified local high-interruption pilot: Voryn admitted 8/8 turns and 4/4 expressive target frames; strict duplex export admitted four examples with zero diagnostics; independent validator passed 24 kHz stereo audio, crop/timing, and no label leakage.
- [x] `N-003` Strict exporter was run on a repaired high-interruption conversation and passed independent validation with zero diagnostic examples.
- [x] `N-005` One high-interruption pilot completed generation, strict export, and independent validation. It is the current certified integration fixture, not a claim that the 1,000-row corpus is complete.
- [~] `I-010` Full-plan allocator now crosses dialogue act with interruption pattern and rotates speech/control assignments within every act block, preventing hidden category correlation.
- [x] `E-005` Verified the 1,000-row version-2 plan: all 1,000 rows require target control, 200 require real barge-in/recovery, each dialogue act appears 50 times, each interaction class and speech style 100 times, each turn pattern 200 times, and all directed reference pairs are unique.
- [x] `N-004` Generated and quota-validated `personaplex-1000-plan.v2.jsonl` before worker launch.
- [~] `I-011` Extended the PersonaPlex typed control contract to accept the production semantic-source vocabulary: ASR finalizer, tool result, interruption controller, handoff router, and timer.
- [~] `I-012` Added the controlled-native pre-codec bridge: directed-pair split assignment, no target wording in its manifest, cropped target-word alignments, and explicit control-frame validation.
- [~] `I-013` Added the controlled native encoder and corpus certifier. Their loss mask excludes every caller stream and all prior/future agent context outside the current audible target turn.
- [x] `N-006` Re-exported the certified barge/recovery fixture through the controlled-native pre-codec bridge. Verified all four records: public-domain directed-pair provenance, hash-only target labels, pair-stable split assignment, 24 kHz duplex integrity, and one audible barge crop.
- [x] `N-007` Restored `student_best.pt` under the pinned CUDA 12.1 runtime, bound the 17-stream delayed layout plus deterministic Moshi-source fingerprint, encoded four duplex targets with the matching Mimi/SentencePiece artifacts, and tensor-certified them. Independent audit confirmed zero caller, prior-agent, future-agent, or out-of-window text loss bits.
- [~] `N-008` Replaced the narrow v2 service-call allocation with the research-anchored v3 plan: 100 topic seeds across 20 families and ten context lenses, unique 6-20 turn briefs, broad conversational forms, and an 85% non-service minimum. Regenerate and verify the v3 quotas before materialization on GPU lanes 0, 1, and 2. Each admitted target must retain strict ASR, semantic-control, duplex timing, provenance, and overlap/recovery evidence; failed scenarios are requeued rather than silently skipped.
- [~] `N-009` Added a per-target seeded control-event program and bidirectional caller-ASR capture contract. Validate that every admitted v3 target records an applied boundary mutation, hash-only target label, and independent semantic-realization pass before adapter encoding.

## 2026-07-15 V7 paired-control execution update

- [!] `B-004` Two artifacts named as V7 paired groups passed audio/ASR/semantic
  certification, but inspected output remains
  `voxrn.synthetic-conversation.v3` with `counterfactualGroupId=null`,
  `branch=null`, and `control.evidence=null`. They are valid audio pilots but
  are not paired-control corpus evidence and must not count toward the 1,000
  call target or adapter training. Inspection source:
  `/srv/voxrn_cache/personaplex-lanes/gpu0/datasets/synthesize/personaplex-v7-paired-v7cf-p1000v4-0001-1784171258195.certified.jsonl`
  and
  `/srv/voxrn_cache/personaplex-lanes/gpu1/datasets/synthesize/personaplex-v7-paired-v7cf-p1000v4-0002-1784171650418.certified.jsonl`.
- [~] `B-005` The V7 manifest writer and certificate queue were corrected
  locally: V7 records now require V4 schema, stable group/branch lineage,
  replay-branch restamping, pivot-aligned evidence, and certificate-side V7
  contract validation before generic promotion. Required proof: a fresh V4
  paired artifact that passes the new queue and evidence exporter end to end.
- [~] `I-014` V7 paired generator is intended to replay shared context byte-identically,
  excludes replayed secondary-prefix targets from loss eligibility, captures an
  acknowledged pre-turn typed control revision, and performs a post-ASR
  semantic realization check. Required proof: a fresh schema-valid V7 artifact
  with group/branch/evidence lineage plus a V7 lineage/counterfactual report
  across a quota-complete certified shard.
- [~] `I-016` Added `export_v7_evidence_frames.py`, which validates typed
  `ControlTrainingFrame`/`EvidenceTrainingFrame` alignment, exact replay
  prefix, paired branch lineage, audio/timeline presence, target-label
  separation, and split isolation before emitting delayed-evidence examples.
  Required proof: a successful V4 paired export with zero rejected groups.
- [~] `I-017` V7 lane completion is now certification-aware: structurally
  admitted groups remain pending while the independent queue audits them; a
  deleted/rejected group causes a bounded systemd restart and fresh generation,
  while a lane exits cleanly only after every assigned group is certified.
  Required proof: one accepted V4 pair and one forced rejection/requeue trace
  through the durable supervisor.
- [x] `I-015` Dedicated 35B semantic inference lanes now exist for physical
  CUDA 0 and CUDA 2, with a local ChatML compatibility adapter on CUDA 2;
  CUDA 1 retains the proven shared 35B lane. Required proof: persistent service
  status, physical GPU binding, and per-lane run cards for the complete run.
  Proof: `/srv/personaplex_workspace/ground_truth_runs/personaplex-v7-lane-supervisor-status-20260715.json`.
- [!] `B-003` Raw V7 admissions and structurally valid manifests are not corpus
  certificates. No native adapter training may consume them until the paired
  independent certificate, split/lineage validation, and coverage quota gates
  pass.
- [x] `N-010` Replaced the detached V7 lane handoff shells with a
  `systemd --user` three-lane supervisor. It must wait for the present owned
  workers to exit, preserve their progress files, bind physical CUDA 0/1/2,
  use distinct Voicebox/data/log roots, restart failed owned workers with a
  bounded backoff, and emit a durable lane status report.
  Proof: enabled `personaplex-v7-lane@0.service`,
  `personaplex-v7-lane@1.service`, and `personaplex-v7-lane@2.service` were
  active on 2026-07-15 and waiting behind their respective existing workers.
  Post-handoff proof: `/srv/personaplex_workspace/ground_truth_runs/personaplex-v7-lane-supervisor-status-20260715.json`.
- [~] `N-011` Added enabled `systemd --user` V7 certifier instances for lanes
  0, 1, and 2. They process one completed pair per pass against the lane's
  semantic endpoint, preserve failures for regeneration, and atomically promote
  only paired certificates into the corpus index.
  Local proof: `personaplex-v7-certifier@0.service`,
  `personaplex-v7-certifier@1.service`, and
  `personaplex-v7-certifier@2.service` were active on 2026-07-15.
  Required remaining proof: durable certificate index with accepted/rejected
  counts, branch-pair IDs, coverage, and lineage hashes.
- [~] `N-012` Corrected V4 V7 workers are active under the persistent three-lane
  supervisor and the paired certifier is active. Generate and certify the remaining V7 groups until the 500 paired
  groups / 1,000 conversations meet the coverage allocation in
  `EXECUTION_PLAN_1000_CALLS_AND_SEMANTIC_CONTROL.md`; regenerate failed
  assignments rather than lowering semantic, ASR, timing, or provenance gates.
- [ ] `N-013` Materialize certified V7 records into evidence-frame and native
  delayed-duplex shards, then prove split isolation and target-label exclusion
  before the frozen-base prefix smoke epoch.

# Ralph execution contract: complete semantic-control implementation

**Operator mandate:** Continue this program across process failure, context compaction, and worker restart. A failure is a defect to diagnose and repair, not a stopping point. Never mark an item complete from intent, logs alone, or an unverified claim. Each completed item must cite a reproducible artifact path, command, revision, and measured result. Do not deploy or push unless explicitly requested. Never use CUDA GPU 3; use GPUs 0, 1, and 2 only and respect other services.

## 0. Program continuity and evidence discipline
- [ ] `R-000` Maintain this file as the durable source of truth; append real artifacts and reasons for any retry or rejection.
- [ ] `R-001` At the beginning of every resumed worker/session, read the last incomplete gate and resume it; do not restart completed gates without a stated invalidation reason.
- [ ] `R-002` Record every owned process with command, PID, GPU assignment, resource root, start time, and expected artifact.
- [ ] `R-003` Detect an owned worker exit, model load failure, empty GPU execution, corrupt artifact, failed evaluation, or stale lock; repair and restart the affected stage automatically.
- [ ] `R-004` Keep all generated artifacts immutable after certification; create a new revision rather than editing an admitted source artifact.
- [ ] `R-005` Use only explicit schema/provenance contracts and actual model inference for semantic decisions; do not use regex, keyword, or placeholder heuristics as a semantic acceptance substitute.
- [ ] `R-006` Keep training labels out of all control, evidence, and retrieval inputs; verify this mechanically before export and training.
- [ ] `R-007` Pin source repository revision, renderer revision, model revision, model checksum, prompt/template revision, seed, and environment manifest for every run.
- [ ] `R-008` Keep a rejected-artifact ledger containing exact contract failure, certifier output, source path, and retry disposition.
- [ ] `R-009` Preserve a held-out split that cannot be reintroduced into synthesis prompt selection, adapter training, threshold tuning, or checkpoint selection.
- [ ] `R-010` Require an independent certifier process to validate generated work; the generator cannot self-admit its own artifacts.

## 1. Rights, voices, rendering, and source provenance
- [ ] `D-001` Inventory every source voice reference with source URL, license, attribution requirement, consent/provenance status, speaker identifier, language/accent tags, and checksum.
- [ ] `D-002` Exclude unlicensed, ambiguous, celebrity, scraped, non-consenting, or provenance-incomplete voice material from cloning and training.
- [ ] `D-003` Curate approved short reference windows with clean speech, no music, no overlapping speakers, no clipping, and sufficient phonetic/style diversity.
- [ ] `D-004` Maintain disjoint voice-reference, training, validation, and evaluation speaker selections where the experiment requires unseen-speaker evaluation.
- [ ] `D-005` Use Chatterbox Turbo as the corpus renderer unless a separately certified renderer experiment is explicitly enrolled; do not silently mix renderers.
- [ ] `D-006` Run the LuxTTS versus Chatterbox Turbo comparison as a labeled, non-training audition experiment with matched text, references, codec path, ASR, timing, and listener artifacts.
- [ ] `D-007` Admit an alternate renderer only after provenance, ASR, timing, codec, speaker-similarity, and human review results are recorded; otherwise keep it isolated.
- [ ] `D-008` Capture raw render, normalized PCM, target phone codec, and round-trip decoded waveform for every admitted utterance.
- [ ] `D-009` Validate sample rate, channel count, duration, finite samples, peak/clip behavior, loudness range, leading/trailing silence, and codec decodability from waveforms.
- [ ] `D-010` Use a real ASR model for transcription authenticity and alignment; retain model/version, decode configuration, confidence, word timestamps, and WER/CER against the target label.
- [ ] `D-011` Reject or quarantine renderer samples that fail empirical ASR, timing, codec, or audio-integrity gates; tune gates only against an audited held-out calibration set.
- [ ] `D-012` Store playback-ready artifacts for operator review without treating subjective review as a replacement for certification.

## 2. Synthetic-call ontology and diversity plan
- [ ] `S-001` Define a versioned conversation-profile schema covering domain, topic, relationship, channel, caller objective, agent role, language/register, stakes, policy constraints, tools, and terminal disposition.
- [ ] `S-002` Define a versioned participant schema covering approved voice reference, speech rate, prosody, confidence, warmth, assertiveness, compliance, resistance, uncertainty, interruption tendency, repair style, and accessibility/codec conditions.
- [ ] `S-003` Define an exhaustive topic taxonomy spanning support, delivery, billing, account recovery, health-adjacent non-diagnostic triage, insurance-adjacent coverage questions, travel, education, polling, commerce, casual conversation, complaints, retention, surveys, technical troubleshooting, scheduling, employment-adjacent intake, community services, and handoff/escalation.
- [ ] `S-004` Sample topic, role, posture, interaction trajectory, voice pairing, opening style, closing style, tool state, and control-plane event independently enough to prevent mode collapse.
- [ ] `S-005` Prohibit default greeting/opening/outro templates and placeholder utterances; generate concrete names, organizations, facts, dates, and local details consistent with the selected scenario.
- [ ] `S-006` Detect duplicate or near-duplicate conversations, openings, closing sequences, semantic plans, and speaker-pair/topic combinations using embedding/model evaluation and provenance keys, not lexical-only matching.
- [ ] `S-007` Maintain explicit quota ledgers for topics, voice pairings, roles, languages/registers, interaction postures, interruption modes, control events, and terminal dispositions.
- [ ] `S-008` Generate cooperative, conditionally cooperative, skeptical, resistant, hostile-but-safe, confused, refusing, clarifying, correcting, recovering, escalating, handing-off, and casually conversational trajectories.
- [ ] `S-009` Generate natural discussion lengths and turn counts ranging from brief resolution through long multi-fact interactions; prevent one-turn exchanges from dominating.
- [ ] `S-010` Generate realistic interruptions, overlap, silence, hesitation, acknowledgement, repair, restart, and barge-in sequences as timed audio events, not static labels.
- [ ] `S-011` Generate natural endings through model intent and task resolution, with the semantic plane producing an end-call action when appropriate; do not force deterministic goodbye matching.
- [ ] `S-012` Include failed/partial tool outcomes, changed facts, contradictory caller claims, unavailable actions, policy restrictions, recovery, and corrective apology cases.
- [ ] `S-013` Produce exactly 1,000 eligible paired conversations as 500 counterfactual groups with two branches each, unless a recorded plan revision changes this target.
- [ ] `S-014` Do not count a generated conversation toward the 1,000 target until its full audio/timing/control/provenance bundle independently certifies.

## 3. V4/V7 paired counterfactual data contract
- [ ] `C-001` Reject all legacy V3 records from training, validation, counting, encoding, and checkpoint selection; retain them only as clearly labeled non-training pilots.
- [ ] `C-002` Require `voxrn.synthetic-conversation.v4` schema on every eligible paired artifact.
- [ ] `C-003` Require a stable `counterfactualGroupId`, exactly two distinct branch IDs, and immutable lineage metadata on every eligible artifact.
- [ ] `C-004` Require an exactly shared pre-pivot context in both branches, with matching caller/agent duplex history and matching control history before the pivot.
- [ ] `C-005` Require one explicit pivot target ordinal and one material control/evidence/fact difference that changes the legitimate next response after the pivot.
- [ ] `C-006` Require branch-specific post-pivot trajectories to remain plausible under the same preceding audio context and to differ because of the declared causal change.
- [ ] `C-007` Require replay-context prefixes to be marked non-eligible and ensure they cannot be counted as independent conversations or training examples.
- [ ] `C-008` Require every evidence item to carry matching group ID, branch ID, pivot ordinal, evidence revision, availability time, provenance, and immutable payload hash.
- [ ] `C-009` Verify that no target transcript or target speech-token content is copied into a control frame, evidence frame, retrieval payload, or tool-result field.
- [ ] `C-010` Validate branch pairing, exact prefix equality, pivot uniqueness, evidence lineage, target-label non-leakage, and audio/timeline availability in a standalone contract validator.
- [ ] `C-011` Emit a per-group V7 contract sidecar and require it before generic semantic/audio certification.
- [ ] `C-012` Make the certifier delete/requeue invalid groups rather than allowing a stale progress marker to mask a rejection.
- [ ] `C-013` Prove rejection/retry behavior with an intentionally malformed fixture and retain the certifier trace.
- [ ] `C-014` Produce a rolling index containing all eligible groups, quota coverage, certification state, rejection reason, artifact checksums, and split assignment.

## 4. Conversation generation and timed duplex construction
- [ ] `G-001` Generate a structured call tree before rendering: caller intent, known facts, uncertainty, commitments, policy constraints, tools, posture, goals, and terminal conditions.
- [ ] `G-002` Generate a rolling typed control frame before each target agent turn without copying the planned target wording.
- [ ] `G-003` Generate tool result references and materialized tool facts independently of response labels, including success, delayed, partial, unavailable, and contradictory outcomes.
- [ ] `G-004` Generate counterfactual variants that preserve context but change one valid tool/policy/fact/posture/evidence condition and require materially different agent behavior.
- [ ] `G-005` Generate caller behavior that can revise the state tree through new facts, resistance, compliance, questions, interruption, correction, or refusal.
- [ ] `G-006` Ensure the agent response naturally incorporates current evidence/control rather than restating a generic template.
- [ ] `G-007` Construct speaker timelines from word/phoneme timing, rendered waveform duration, VAD/turn data, and selected interaction profile.
- [ ] `G-008` Model realistic conversational gaps, overlap, latency, partial starts, interruption cutoffs, and repair starts using measured timing distributions from approved call examples.
- [ ] `G-009` Materialize actual cancellation cutoffs by truncating outgoing agent audio at the barge-in point and recording generated, sent, cancelled, and audible durations.
- [ ] `G-010` Include recovery responses after cancellation that use the updated revision and acknowledge the caller’s new contribution naturally.
- [ ] `G-011` Record raw participant audio, mixed duplex audio, agent-only/caller-only stems, timeline events, transcript labels, ASR output, and codec variants.
- [ ] `G-012` Validate all generated names, businesses, dates, quantities, tracking values, and identifiers for internal consistency within a call without literal placeholder tokens.
- [ ] `G-013` Maintain a no-template-collapse report based on model-assisted semantic clustering and distributional coverage across openings, closings, goals, and response acts.
- [ ] `G-014` Generate in parallel only through GPU-aware queue lanes with per-lane resource roots, model endpoints, retry budgets, and artifact isolation.

## 5. Semantic control and MoshiRAG evidence specification
- [ ] `M-001` Write the versioned compact control-frame schema used at runtime and training: call ID, revision, effective boundary, intent, facts, uncertainty, commitments, policy constraints, tool references, caller posture, next goal, terminal action, and continuous style controls.
- [ ] `M-002` Define typed evidence objects with source class, source revision, availability time, reliability, validity window, payload hash, retrieval provenance, and relation to the call-state tree.
- [ ] `M-003` Distinguish durable facts, transient caller claims, tool results, policy constraints, retrieval evidence, and speculative hypotheses; do not collapse them into an untyped prompt string.
- [ ] `M-004` Define MoshiRAG-inspired delayed-evidence semantics: evidence arrives after prior context, is encoded separately, becomes effective only at an acknowledged boundary, and can alter the next response causally.
- [ ] `M-005` Build a content-addressed evidence store and retrieve only evidence authorized by call ID, policy, validity, and revision boundaries.
- [ ] `M-006` Encode evidence/control once per accepted revision and cache the GPU representation; forbid per-20-ms re-encoding.
- [ ] `M-007` Define control merge, supersession, revocation, and uncertainty behavior for multiple evidence updates.
- [ ] `M-008` Reject stale, malformed, unauthorized, future-effective, or label-leaking control updates with typed machine-readable reasons.
- [ ] `M-009` Define safe sparse-control behavior: natural low-risk backchannel/wait behavior is permitted; policy-sensitive substantive speech may not be generated from stale state.
- [ ] `M-010` Define exact-wording escalation behavior: route exact scripts to a separately validated strict renderer and never claim deterministic wording from speech-to-speech conditioning.
- [ ] `M-011` Create counterfactual control/evidence fixtures where the same audio context has different legitimate updated facts, tool state, policies, or caller posture.
- [ ] `M-012` Score whether generated next-turn content changes in the correct direction because of the new evidence, not merely because topic keywords are present.

## 6. PersonaPlex upstream model adaptation
- [ ] `A-001` Identify the exact PersonaPlex fork revision, streaming-generator entry point, transformer forward path, code-token representation, audio context representation, and decoding scheduler used by the target runtime.
- [ ] `A-002` Add a compact control/evidence encoder that consumes the structured versioned frame, not an appended natural-language system prompt.
- [ ] `A-003` Add learned virtual control-prefix tokens or per-layer K/V prefixes produced by that encoder.
- [ ] `A-004` Add a learned gate for control injection so absent, weak, or irrelevant control does not damage native turn-taking, voice identity, or audio quality.
- [ ] `A-005` Inject control into selected transformer layers or all layers with configuration captured in the experiment manifest.
- [ ] `A-006` Keep the PersonaPlex base frozen for the first stage; train only the control/evidence encoder, prefix/K-V adapter, gates, and explicitly declared lightweight heads.
- [ ] `A-007` Add control dropout, evidence dropout, stale-control negative examples, and null-control examples to preserve conversational behavior when state is sparse.
- [ ] `A-008` Add branch, revision, group, pivot, and effective-boundary alignment assertions in the native training input path.
- [ ] `A-009` Train only agent target audio/code-token loss for controlled generations; caller/replay context is conditioning, never a target loss.
- [ ] `A-010` Verify gradients flow into every adapter component and do not flow into frozen base parameters during the frozen stage.
- [ ] `A-011` Verify model save/load preserves adapter weights, schema version, tokenizer/codec compatibility, and inference configuration.
- [ ] `A-012` Maintain an isolated upstream patch series with tests/harnesses so the fork can be rebased without carrying unrelated Voryn application changes.
- [ ] `A-013` Assess any limited base adaptation only after the frozen adapter passes all non-regression gates; retain a frozen-adapter baseline checkpoint.
- [ ] `A-014` Reject a patch that merely accepts a WebSocket `guidance` field without changing the actual forward pass and generated speech-token logits.

## 7. Revisioned real-time runtime and audio plane
- [ ] `P-001` Add per-call `control_revision`, acknowledged-control revision, immutable generation snapshot, generation ID, cancellation token, and terminal state to the session model.
- [ ] `P-002` Implement typed `control.update` intake from the semantic/state service with schema/version/authentication validation.
- [ ] `P-003` Encode and GPU-cache a candidate control representation when a revision arrives; acknowledge only after successful validation and encoding.
- [ ] `P-004` Reject revisions lower than or equal to the latest acknowledged revision and record the reason without altering active generation.
- [ ] `P-005` Snapshot the latest acknowledged control representation exactly at an agent turn boundary along with the current duplex context and generation ID.
- [ ] `P-006` Pass that snapshot into the real PersonaPlex forward/generator path and log model-side revision consumption.
- [ ] `P-007` Do not recompute/change control in the middle of a 20-ms output chunk or retroactively claim already-sent audio used a later revision.
- [ ] `P-008` Detect caller barge-in from real streaming audio/turn state, immediately invalidate the active generation ID, halt queued media, and record cancellation latency.
- [ ] `P-009` Ensure next agent turn observes only the newest acknowledged revision, latest duplex context, and fresh generation ID.
- [ ] `P-010` Handle tool results, policy updates, ASR corrections, retrieval evidence, and state-tree mutations as revision-producing inputs.
- [ ] `P-011` Preserve full-duplex streaming behavior over the Twilio audio plane: incoming mu-law/Opus decode, sample-rate/codec bridge, PersonaPlex streaming decode, outbound framing, and sequence correctness.
- [ ] `P-012` Add backpressure, jitter, gap, out-of-order frame, reconnect, decode failure, render failure, and cancellation recovery handling.
- [ ] `P-013` Implement model-driven terminal intent/state propagation to the call-control layer; a valid end action must stop media and invoke the appropriate call-end tool path.
- [ ] `P-014` Prohibit endless sign-off loops through semantic terminal-state training/evaluation, not deterministic goodbye pattern matching.
- [ ] `P-015` Emit per-call causal traces linking evidence revision, acknowledged adapter cache key, generation snapshot, emitted media span, cancellation, and terminal disposition.

## 8. Evidence export, native encoding, and dataset admission
- [ ] `E-001` Export only certified V4/V7 paired artifacts into `EvidenceTrainingFrame` records.
- [ ] `E-002` Validate control-frame mapping, evidence-frame mapping, revision monotonicity, evidence availability before agent start, and post-evidence context hash alignment during export.
- [ ] `E-003` Require exactly one declared plan/target relation per training frame and reject ambiguous/multi-target alignments.
- [ ] `E-004` Include recent delayed duplex code context, agent target code/audio labels, control frame, evidence references, timing/cutoff fields, group/branch/pivot metadata, and split assignment.
- [ ] `E-005` Strip target transcript/audio/code content from all control/evidence inputs and test the exporter against deliberate leak fixtures.
- [ ] `E-006` Encode native PersonaPlex-compatible tensors/shards with codec/tokenizer revision, shape, dtype, integrity hash, frame count, and source-frame manifest.
- [ ] `E-007` Validate random and boundary shard samples by decoding/inspecting correspondence to their manifest/timeline without mutating source data.
- [ ] `E-008` Generate per-split manifests with stable IDs and prevent train/validation/test group leakage across counterfactual branches.
- [ ] `E-009` Verify each class of control event, evidence source, posture, interruption, and terminal state has sufficient train and held-out coverage.
- [ ] `E-010` Publish a data-card-style artifact documenting generation process, renderer, sources, exclusions, coverage, certification gates, known limitations, and intended use.

## 9. CUDA resource-aware staged training
- [ ] `T-001` Inspect current GPU 0/1/2 memory/compute use and reserve only safe capacity; never launch training on GPU 3 or evict unrelated services.
- [ ] `T-002` Record GPU topology, CUDA/PyTorch/driver versions, process allocation, distributed backend, precision mode, and effective batch/accumulation configuration.
- [ ] `T-003` Run a memory-aware single-step dry run using a certified shard and fail on incorrect device placement, missing gradients, non-finite loss, or base-weight mutation.
- [ ] `T-004` Run the frozen semantic-prefix stage with distributed GPU 0/1/2 workers, checkpointing, restart-safe state, deterministic seeds, and explicit per-rank logs.
- [ ] `T-005` Evaluate a baseline/no-control condition, null-control condition, current-control condition, and counterfactual-control condition at every checkpoint interval.
- [ ] `T-006` Track agent-only code/audio loss, semantic adherence, factual/tool incorporation, constraint violations, control sensitivity, counterfactual separation, voice preservation, latency, interruption behavior, and codec quality.
- [ ] `T-007` Stop/rollback a candidate checkpoint that regresses native turn-taking, first-audio latency, speaker quality, duplex behavior, or interruption recovery beyond predeclared tolerances.
- [ ] `T-008` Select the frozen-stage checkpoint only from held-out metrics and reproduce it from manifest/checkpoint state.
- [ ] `T-009` Run the MoshiRAG/evidence-conditioning stage initialized from the selected frozen semantic-prefix checkpoint.
- [ ] `T-010` Train with delayed-evidence and revision-boundary examples, including no-evidence, irrelevant-evidence, stale-evidence, revoked-evidence, and materially changed-evidence counterfactuals.
- [ ] `T-011` Test a limited base-adaptation stage only if frozen-stage evidence shows a specific remaining failure; compare against frozen baseline on identical held-out suites.
- [ ] `T-012` Save optimizer/scaler/RNG/data-cursor state and all adapter/control configs so interrupted runs resume exactly without reusing completed data incorrectly.
- [ ] `T-013` Generate checkpoint cards describing corpus revision, adapter architecture, trainable parameter count, resource use, metrics, risks, and promotion decision.

## 10. Evaluation and adversarial validation
- [ ] `V-001` Build a held-out causal-control suite where prior audio is identical and only one valid control/evidence update differs between paired branches.
- [ ] `V-002` Measure whether the next generated speech reacts correctly to updated intent, facts, policy, tool results, caller posture, and next-goal constraints using model-based and human-audited scoring.
- [ ] `V-003` Measure factual incorporation and non-invention against the structured state/evidence source, including unavailable/uncertain facts.
- [ ] `V-004` Measure stale-control rejection, acknowledgement ordering, effective-boundary behavior, and immutable snapshot correctness.
- [ ] `V-005` Measure first-control-encode latency, first-audio latency, median/p95 chunk cadence, cancellation latency, recovery latency, and end-to-end media delay.
- [ ] `V-006` Measure codec quality after the actual Twilio-compatible codec bridge, including decode integrity, clipping, loudness, ASR stability, and artifact incidence.
- [ ] `V-007` Measure voice/expressive preservation against the unmodified PersonaPlex baseline using approved speaker/quality evaluation methods.
- [ ] `V-008` Measure duplex behavior using real overlap/barge-in fixtures, including exact audible cutoff and correct updated recovery turn.
- [ ] `V-009` Measure terminal behavior using model-generated terminal state/tool invocation and test resistance to endless polite-signoff loops without regex termination logic.
- [ ] `V-010` Evaluate hostile, skeptical, resistant, correction, handoff, partial tool failure, constraint conflict, and rapid-evidence-update cases.
- [ ] `V-011` Run ablations: no adapter, control only, evidence only, control plus evidence, control dropout, and selected injection-layer/gate configurations.
- [ ] `V-012` Require the final candidate to beat the baseline on causal adherence and factual incorporation while staying within predeclared quality/latency/duplex tolerances.
- [ ] `V-013` Store raw outputs, scores, evaluator versions, prompts/fixtures, audio artifacts, and failure taxonomy for every promotion evaluation.

## 11. Twilio-compatible end-to-end emulation
- [ ] `W-001` Implement/use a deterministic local Twilio Media Streams emulator that exercises inbound/outbound sequencing, mu-law/Opus conversion, jitter, silence, packet loss, duplicate/out-of-order events, reconnect, and stop/mark semantics.
- [ ] `W-002` Feed realistic generated caller audio through the emulator and capture agent output as it would be transmitted, not a direct offline decoder shortcut.
- [ ] `W-003` Run streaming ASR and turn detection in parallel with the audio path; feed resulting state/evidence updates through the normal typed control protocol.
- [ ] `W-004` Verify model control updates are acknowledged before the next generation snapshot and stale updates cannot alter already-sent media.
- [ ] `W-005` Verify barge-in cancels queued/generated media, emits the correct trace, updates the state tree, and produces a semantically updated recovery response.
- [ ] `W-006` Verify terminal state triggers the modeled call-end action/tool path and leaves no residual media/signoff loop.
- [ ] `W-007` Run representative end-to-end scenarios across the full trajectory taxonomy and retain playback artifacts for human review.
- [ ] `W-008` Do not use live caller data or mutate live call state during emulator evaluation.

## 12. Product integration and interface completion
- [ ] `I-001` Register PersonaPlex as a selectable live conversational voice option in AI Agents create-agent flow with accurate capability/latency/semantic-control disclosure.
- [ ] `I-002` Route PersonaPlex selection to the streaming audio plane rather than fragmented webhook/TTS orchestration.
- [ ] `I-003` Prevent unsupported exact-wording/policy-sensitive modes from presenting PersonaPlex as deterministic; route those to the strict renderer path.
- [ ] `I-004` Fix create-agent progress state so completed tool selection visibly marks the relevant top tab green when progressing to voice selection.
- [ ] `I-005` Preselect required call tools while allowing removal; show a clear, state-specific warning if removal makes the chosen agent configuration unsupported.
- [ ] `I-006` Move mandatory setup variables to the top of the workspace, highlight them gently until complete, and block tool progression until required values exist.
- [ ] `I-007` Apply day/night-mode CSS variables to tool module and selected-tool background/border states, matching adjacent component behavior.
- [ ] `I-008` Add deterministic integration tests or browser harness cases for PersonaPlex selection, required-variable gating, tool-tab progression, and theme state.

## 13. Promotion, documentation, and release readiness
- [ ] `L-001` Update PersonaPlex fork documentation with the actual adapter architecture, typed control protocol, revision/cancellation semantics, corpus contract, training stages, metrics, and known non-guarantees.
- [ ] `L-002` Update model/dataset documentation for the `cudabenchmarktest` Hugging Face artifacts only after the corresponding files/checkpoints/cards actually exist; include license/provenance and intended-use limitations.
- [ ] `L-003` Produce an architecture decision record comparing learned prefix/K-V control with prompt-side guidance, strict rendering, and any relevant MoshiRAG/Moshi-style mechanisms.
- [ ] `L-004` Produce a reproducible bootstrap/resource-location document for `/srv/voxrn_cache` and all cache/environment variables without embedding secrets.
- [ ] `L-005` Require final artifact inventory: source revisions, patches, model/checkpoint hashes, data manifests, evaluator results, resource manifest, runtime traces, and rollback baseline.
- [ ] `L-006` Require independent review of all promotion evidence; unresolved regression, provenance issue, missing causal trace, or missing reproducibility artifact blocks promotion.
- [ ] `L-007` Only after all gates pass and explicit operator approval, prepare a clean commit/push/release plan. No automatic deployment is authorized by this ledger.

## 2026-07-16 implementation evidence log

- [~] `C-001` through `C-014`: Legacy V3 sidecars are now terminal/auditable and skipped by the V7 certifier. Fresh generator records are asserted V4 before bundle write. Evidence: `scripts/run-personaplex-v7-paired-lane.js`, `scripts/certify-personaplex-v7-paired-queue.js`; fresh certification remains in progress.
- [~] `N-012`: GPU 0/1/2 lane lifecycle repaired to one group -> independent certification -> service restart, with `SYNTHESIS_MAX_ASR_WER=0.25`. Evidence: `/srv/personaplex_workspace/robit-man-personaplex/ground_truth_finetuning/tools/run-v7-lane-service.sh` and `~/.config/systemd/user/personaplex-v7-lane@.service`.
- [~] `A-002` through `A-005`: Runtime now stages a typed `evidence.update`, binds it only to a later aligned `control.update.evidenceFrame`, encodes a learned CUDA evidence stream, and queues it into the patched native generator at the acknowledged boundary. Source: `personaplex_control/runtime.py`; compile evidence recorded in session, functional checkpoint proof pending.
- [~] `P-003` through `P-009`: Evidence arrival invalidates active generation; only the next immutable control snapshot can queue its stream; barge-in clears queued native evidence and media eligibility. Source: `personaplex_control/runtime.py`; controlled-server checkpoint deployment and emulator proof pending.
- [~] `A-012`: Applied the isolated Moshirag streaming-sum patch to `/srv/personaplex_workspace/personaplex-distilled-patch/moshi` at upstream `3428dfd95309a7f3c84fd93259ded0f810d1ff91`. Patched source contract: `/srv/personaplex_workspace/ground_truth_runs/personaplex-student-best.moshirag.cuda2.contract.v3.json`, source SHA-256 `sha256:c578958da236b4b4f24458005297983d360d9e868655ae608100feac0458651e`.
- [~] `T-009` through `T-013`: Added `tools/train_evidence_stream.py`; it freezes the base/control adapter, requires certified source/tensors, trains only `EvidenceStreamAdapter`, saves source/control-checkpoint-bound artifacts, and records held-out wrong-evidence sensitivity. No training run has started because no V4/V7 pair has passed every independent certificate yet.
- [~] `E-001` through `E-009`: V4 evidence is carried through controlled duplex export, precodec preparation, and native encoding. Counterfactual groups use group-stable splits. Source: `export_controlled_duplex_dataset.py`, `prepare_controlled_native_adapter_dataset.py`, and `encode_controlled_native_adapter_tensors.py`; first certified materialization remains pending.
- [~] `V-001` through `V-013`: Structural V4 pair collection now selects exactly one evidence-aligned pivot target from each branch and invalidates stale generic certificates via `counterfactualPairingRevision=v4-lineage-pivot-v2`. The first audited V4 group was correctly rejected for an ASR-distorted response and a control-ignoring terminal response; it is not eligible for training.
- [~] `G-001` through `G-014`: A malformed model control JSON no longer dereferences a null turn context. It fails closed and retries without render/output. Source: `lib/syntheticConversations.js`.

### 2026-07-16 execution evidence: certified V4 throughput and runtime control

- [x] Removed `naturalRealization` from materializer requests and persisted control/evidence context. Target labels are not supplied to control conditioning.
- [x] Made synthetic reply and control materialization retry paths model-only typed retries; malformed output is never locally parsed, inferred, or admitted.
- [x] Removed local transport normalization from typed control/evidence admission. Any malformed typed envelope receives a separate model-only repair request; failed repair remains rejected.
- [~] Routed semantic planning, typed control materialization, caller authenticity, and final certification to a separately hosted local 35B endpoint after the cloud Nemotron session quota was exhausted. Dialogue generation stays on a different local 35B process for every lane; this preserves an independent model process but is not a cloud-model fallback.
- [x] Kept Chatterbox rendering and Whisper ASR gates enabled; current admission limits are ASR confidence >= 0.45 and WER <= 0.25.
- [x] Moved pre-render semantic work to the independent final batch certifier. Candidate bundles are non-trainable until every caller/target turn and counterfactual pivot pair pass its cross-lane local semantic audit.
- [x] Decoupled candidate generation from certification waiting. Each GPU lane advances after structural bundle creation while a parallel lane certifier promotes or rejects independently.
- [x] Added V4 lane unresolved-group quarantine so exhausted groups do not deadlock the full 1000-conversation plan.
- [x] Added runtime evidence harness coverage for `control N -> evidence N+1 -> successor control -> native streaming-sum -> barge-in cancellation`.
- [x] Added controlled-server evidence checkpoint compatibility gates for semantic-adapter hash, model revision, stream length, and pinned Moshi weight hash.
- [x] Observed and archived independent V4 bundle certificates after parallel certification. `v7cf-p1000v4-0004` passed bundle certification with 16 promoted records but remains export-quarantined by `B-006`; fresh `v7cf-p1000v4-0023` passed with 40 promoted records. Both certificates report two accepted conversations and one accepted counterfactual group.
- [ ] Export only certificates carrying `counterfactualPairingRevision=v4-lineage-pivot-v2`, accepted conversations, accepted counterfactual groups, and promoted records.
- [ ] Run CUDA 0/1/2 DDP stage-one semantic-prefix training from the certified native tensor corpus, then run checkpoint semantic/control evaluations.
- [ ] Run CUDA 0/1/2 delayed-evidence adapter training against the accepted stage-one checkpoint, then run the native evidence/barge-in runtime harness.

### 2026-07-16 V4 lineage export defect and containment

- [!] `B-006` A certificate for `v7cf-p1000v4-0004` reported two accepted
  conversations and one accepted group, but strict duplex export admitted only
  the `available` branch. The `constrained` branch's enclosing record used a
  branch-local conversation ID while its newly materialized control/evidence
  frames retained the replay source conversation ID. The initial V4 evidence
  exporter compared control to evidence but not either frame to the enclosing
  record, so it incorrectly emitted two delayed-evidence examples. All outputs
  from that pre-fix evidence export are diagnostic only and excluded from
  encoding/training/counts.
- [x] `I-018` Added cross-export identity admission: V4 evidence export rejects
  a record/control/evidence conversation-ID mismatch, and shared-prefix replay
  context is explicitly quarantined from target admission while remaining
  subject to generic audio/timeline/provenance checks. The synthetic continuation
  materializer now uses the branch-local ID for post-pivot control/evidence
  frames. Proof: `ground_truth_finetuning/tests/test_v4_export_contract.py`,
  `python3 -m unittest ground_truth_finetuning.tests.test_v4_export_contract`
  (3 passing); rejected source run
  `/srv/personaplex_workspace/ground_truth_runs/v4-pair-0004-evidence-identity-rejection-20260716-0601`.
- [ ] `N-014` Regenerate and independently certify a fresh V4 pair with the
  repaired materializer. Require both branch-local strict duplex export and
  evidence export to accept the same two pivot examples before pre-codec/native
  tensor preparation; do not restart, count, or promote the existing source
  bundle as repaired evidence.
- [!] `B-007` The configured `nemotron-3-ultra:cloud` semantic endpoints returned
  a provider session-usage-limit error. Ground-truth lanes failed closed at the
  caller-authenticity judgment before audio render, so no missing-render/ASR
  fallback may be interpreted as an admitted sample.
- [x] `I-019` The dialogue plane is explicitly local and CUDA-resident: GPU 0
  uses `personaplex-control-ornith:35b`; GPUs 1 and 2 use
  `robit/ornith:35b`. After the cloud Nemotron quota failure, semantic planning,
  control materialization, caller authenticity, and final certification were
  rerouted cross-lane to a separate local 35B endpoint for each dialogue lane.
  The fresh `0023` certificate proves the route is live; it does not make the
  cloud endpoint available or weaken any semantic gate.
- [x] `I-020` Typed control/evidence transport accepts raw JSON only. Fenced,
  prose, or mixed content is passed to a separate typed model-repair inference
  and is rejected if that repair does not return raw JSON; no regex extraction,
  field synthesis, semantic inference, or audio fallback is permitted.
  Proof: `node --check lib/syntheticConversations.js lib/agentVsAgentSim.js`
  and `tests/unit/agent-vs-agent-sim.test.js`.
- [x] `I-021` V7 lane progress no longer silently skips exhausted assignments.
  It stores each bounded failure in a rejection ledger, rotates the
  least-recently attempted unresolved group ahead of fresh work, and applies a
  deterministic seed offset per regeneration ordinal to both branches. A
  successful group clears only its active unresolved marker. Proof:
  `node --check scripts/run-personaplex-v7-paired-lane.js`; live progress
  recorded `requeue_seed_varied` entries on lanes 0/1/2 with regeneration
  ordinal 1. This establishes requeue execution, not data promotion.

### 2026-07-16 current execution checkpoint

- [x] `N-015` Corrected dialogue routing: only spoken caller/target generation
  receives `SYNTHESIZE_DIALOGUE_INFERENCE_*`; independent control, ASR
  authenticity, and certification remain on `SYNTHESIZE_INFERENCE_*`.
- [x] `N-016` Verified resident local models through Ollama process state:
  CUDA 0 `personaplex-control-ornith:35b`, CUDA 1 and CUDA 2
  `robit/ornith:35b`. No PersonaPlex synthesis worker is assigned to GPU 3.
- [x] `C-015` Current raw bundle-level certificate inventory contains five V4
  accepted pairs / 112 promoted records: `0004` (16), `0006` (28), `0008`
  (12), `0011` (16), and `0023` (40). `0004` remains export-quarantined by
  `B-006`; the four fresh export candidates total 96 promoted records. The
  strict aggregate precodec pass currently has train/test coverage but no
  validation group, so it is not eligible for encoding or training.
- [x] `N-017` The evidence materializer retries malformed typed transport through
  a separate model-only repair request and rejects unrepaired output. The V5
  plan is now live on CUDA lanes 0, 1, and 2, and every repaired record remains
  subject to its normal independent certificate. First proof:
  `v7cf-p1000v5-0003` accepted both branches, one counterfactual group, and 24
  promoted records at `2026-07-16T07:18:19Z`.
- [x] `N-018` Bound native-codec resources are checksum-verified: Mimi codec
  `/srv/voxrn_cache/personaplex/models/auxiliary/tokenizer-e351c8d8-checkpoint125.safetensors`
  is `sha256:09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50`;
  text tokenizer is `sha256:78d4336533ddc26f9acf7250d7fb83492152196c6ea4212c841df76933f18d2d`.
  The separate Moshi checkpoint is `sha256:d584e4476b4feff1a0b0b452e9fe14696dd1319b09663997db4ebd4288933058` and is never substituted for Mimi.

### 2026-07-16 fresh V4 pair, codec provenance, and non-promotion evidence

- [x] `N-014` Regenerated `v7cf-p1000v4-0023` after the branch-identity repair.
  Its independent local semantic certificate accepted two conversations, one
  V4 `v4-lineage-pivot-v2` counterfactual group, and 40 promoted records.
  The fresh certified-only input exported exactly two evidence examples and two
  strict duplex examples, with zero rejections/diagnostics; the independent
  duplex validator passed. Evidence roots:
  `/srv/personaplex_workspace/ground_truth_runs/v4-pair-0023-certified-20260716-0618-evidence`
  and `...-duplex`. The pre-fix `0004` source remains diagnostic only.
- [!] `B-008` The historical native contract bound the frozen LM and source but
  not Mimi or SentencePiece bytes. A first fresh encoding attempt correctly
  rejected `/srv/voxrn_cache/moshi-rag/.../model.safetensors`: it is a Moshika
  LM state dict, not a Mimi state dict. No tensor was emitted by that attempt.
- [x] `I-022` Native contracts now pin LM, Moshi source, Mimi, and SentencePiece
  SHA-256 values. Both encoders verify Mimi/tokenizer bytes; the tensor
  certifier records one codec identity; semantic and evidence trainers reject
  a certificate or current tokenizer that differs. Proof:
  `ground_truth_finetuning/tests/test_native_codec_contract.py` and seven
  passing focused contract/export tests.
- [~] `N-018` The fresh pair encoded under
  `native-model-contract.v4.mimi-bound.json` using only physical CUDA 2. It
  produced two 17-stream `torch.int64` native code tensors with agent-only
  masks, exact Mimi/tokenizer provenance, and a 12.5 Hz codec rate. Its tensor
  certificate reports `insufficient_split_coverage` (2 train, 0 validation,
  0 test), not `certified_for_adapter_training`; no semantic-prefix or evidence
  DDP process/checkpoint has been started. More independently certified groups
  must populate all group-isolated splits first.
- [x] `I-023` Added a direct runtime-session regression using a typed,
  target-wording-free control frame: revision 1 queues and applies only at a
  matching caller boundary; barge-in invalidates that generation, calls evidence
  cancellation, and prevents the superseded revision from becoming active
  again; revision 2 can then apply. Proof:
  `ground_truth_finetuning/tests/test_runtime_control_session.py`.
  The CUDA `runtime_prefix_harness.py` remains blocked on a genuine trained
  control/evidence checkpoint and is not represented as passed.
- [x] `I-024` Hardened inline semantic verification without semantic heuristics:
  each typed judgment uses three raw-JSON model attempts on its primary
  CUDA-resident independent route, then a bounded second CUDA model route.
  It remains fail-closed when both routes cannot return a typed judgment.
  Lane 0 falls back GPU 2 -> GPU 1; lane 1 GPU 0 -> GPU 2; lane 2 GPU 0 ->
  GPU 1. All routes stay within CUDA devices 0, 1, and 2.
- [x] `I-025` Corrected Chatterbox/Whisper remediation: every failed WER,
  confidence, or word-timing gate now obtains a distinct model-generated
  rewrite before rerendering, rather than rerendering identical failed text.
  Raw JSON is required for planner and audio-rewrite transport; no fence
  stripping or local semantic recovery is used. Admission limits remain WER
  <= 0.25, confidence >= 0.45, and valid word-level alignment.
- [x] `I-026` Relaxed only terminal surface-form restrictions that forced
  robotic closings. Terminal action, factual/status preservation, grounded
  conclusion, and no-generic-farewell rules remain model-verified; the model
  may now use natural language rather than a fixed declarative template.
- [x] `C-016` V5 provenance has its first independently certified paired bundle:
  `/srv/voxrn_cache/personaplex-lanes/gpu2/datasets/synthesize/personaplex-v7-paired-v7cf-p1000v5-0003-1784186177471.certified.certificate.json`
  reports two accepted conversations, one accepted counterfactual group, 24
  promoted records, zero rejected counterfactual conversations, and
  `v4-lineage-pivot-v2` pairing. This is V5-only evidence and must not be
  mixed with historical V4 smoke records; it is not yet an export or training
  corpus because group-isolated train/validation/test coverage is incomplete.
- [~] `I-027` Target-label recovery now treats the newest typed boundary update
  as a causal speech requirement: evidence must be applied before a commitment,
  limits before options, corrections before topic changes, and required
  questions before follow-up logistics. It is generated and judged only by the
  independent CUDA model paths, never by a local semantic heuristic. GPU 0 is
  running this revision against `v7cf-p1000v5-0001`; promotion evidence is
  pending its independent certificate.
- [x] `I-028` The final independent semantic certifier now applies the same
  model-only raw-JSON transport repair contract as the generation path. A
  certificate with only `semantic_auditor_unavailable` or
  `counterfactual_auditor_unavailable` transport failures is re-audited; any
  substantive semantic failure remains fail-closed and is requeued. Repair is
  routed to a separate CUDA-resident model for every lane.
- [x] `C-017` Clean V8/V9 provenance is now independently certified. The
  immutable plan
  `/srv/personaplex_workspace/ground_truth_runs/personaplex-1000-plan.v8-counterfactual-diverse-v6.jsonl`
  has 1,000 calls / 500 exact pairs with `available -> ready` and
  `constrained -> failed`. Its first promoted bundle,
  `v8cf-p1000v5-0001`, passed two conversation audits and one counterfactual
  audit at `2026-07-16T07:45:23Z`, promoting all 12 records. The earlier
  transport-only zero-promotion certificate was re-audited, not accepted by
  fiat. This is proof of the corrected end-to-end generation/certification
  path, not sufficient train/validation/test coverage for adapter training.
- [x] `I-029` Corpus throughput now uses six logical synthesis workers without
  loading a fourth model or using GPU 3: workers `0/3` share physical CUDA 0,
  `1/4` CUDA 1, and `2/5` CUDA 2. Each worker has an isolated dataset/progress
  root, while a global accepted-certificate scan skips groups already certified
  elsewhere. Startup evidence records the physical `CUDA_VISIBLE_DEVICES` value
  for every worker; the first scaled checkpoint reached four accepted paired
  groups and 76 promoted V8 records. This increases request-level concurrency
  only; it does not weaken the independent semantic, counterfactual, Whisper,
  or word-alignment gates.
