# PersonaPlex ground-truth iteration 001

## Execution record

- External loop: `8bb7cef2-8fe0-482a-a201-8f7d42cc6aa4`, iteration 1.
- Internal Ralph loop: `ralph-execute-the-ground-truth-perso-mrn3ag7uf9ry`.
- GPU policy observed: synthesis workers use physical CUDA devices 0, 1, and 2 only.
- The V7 synthesis and independent certifier services are active for all three
  allowed GPUs. This is an observation, not a throughput or promotion claim.

## Newly tested certified candidate

Input certificate:
`/srv/voxrn_cache/personaplex-lanes/gpu0/datasets/synthesize/personaplex-v7-paired-v7cf-p1000v4-0004-1784180693503.certified.certificate.json`

- Certificate fields: `acceptedConversations=2`,
  `acceptedCounterfactualGroups=1`,
  `counterfactualPairingRevision=v4-lineage-pivot-v2`.
- The pre-fix evidence exporter created two examples with no rejected groups and
  `targetTextInControlOrEvidence=false`:
  `/srv/personaplex_workspace/ground_truth_runs/v4-pair-0004-evidence-20260716-0557`.
  That output is diagnostic only: the exporter had not compared frame identity
  to the enclosing turn.
- Strict duplex export materialized only one admitted example and rejected the
  constrained branch; its independent audio/no-leakage validator passed only
  for that partial one-branch output:
  `/srv/personaplex_workspace/ground_truth_runs/v4-pair-0004-duplex-20260716-0557`.

## Blocking defect

The constrained branch is rejected as `turn_1_v4_target_evidence_lineage_invalid`.
Its record `conversationId` differs from the `conversationId` carried by its
control/evidence frames, which retain the primary branch identity. The evidence
exporter formerly validated control-to-evidence alignment but did not validate
either frame against the enclosing record identity; it therefore accepted an
inconsistent pair. The repair now rejects the source bundle at evidence export
with that exact reason, recorded in
`/srv/personaplex_workspace/ground_truth_runs/v4-pair-0004-evidence-identity-rejection-20260716-0601`.
The partial duplex result is diagnostic only and must not be used for tensor
preparation, training, checkpoint selection, or a corpus count.

## Next gate

Repair the paired replay/materialization identity contract, add a regression
fixture making both exporters agree on a valid and an invalid pair, regenerate
the affected group through independent certification, and require a two-branch
strict duplex export before native encoding.

## Local repair verification

- `lib/syntheticConversations.js` now builds post-pivot control/evidence frames
  with the branch-local conversation ID rather than the replay source ID.
- V4 evidence export now requires the record, control frame, and evidence frame
  to have the same conversation identity.
- Shared replay turns stay subject to generic audio/timeline/provenance checks,
  but are explicitly quarantined from target-label admission.
- `python3 -m unittest ground_truth_finetuning.tests.test_v4_export_contract`
  passed three regression tests. No live worker was restarted; the corrected
  code requires a separately observed, newly generated and independently
  certified V4 pair before promotion.

## Semantic-service availability repair

- The configured cloud semantic model returned a provider session-usage-limit
  error for a minimal structured request. Lanes correctly failed closed before
  rendering; the resulting missing-render/ASR messages were consequences of
  that rejection, not quality-gate bypasses.
- The resident `personaplex-control-ornith:35b` and `robit/ornith:35b` local
  endpoints each passed the equivalent JSON-only probe. Lane routing now sends
  semantic planning/certification to an independent resident local 35B endpoint
  rather than the branch's dialogue endpoint whenever possible.
- The three lane processes naturally reloaded the new environment between
  batches. The sleeping certifier services were restarted to load the same
  audited route. No model was downloaded, no GPU outside 0/1/2 was selected,
  and no candidate has been promoted on this basis alone.

## Strict local-model transport repair

- Fresh local semantic runs revealed fenced, otherwise complete JSON from the
  typed control materializer and reply-envelope normalizer. Both are now
  accepted only when the *entire* response is one JSON code fence; prose and
  partial extraction still fail closed. This is syntax normalization, not a
  semantic rule or response rewrite.
- Verification: `node --check lib/syntheticConversations.js`,
  `node --check lib/agentVsAgentSim.js`, and
  `npx vitest run tests/unit/agent-vs-agent-sim.test.js` (2 passing).
- Existing workers retain their loaded code until their current bounded attempt
  ends and the supervisor starts the next one. A newly certified pair remains
  required before any export/training claim.
