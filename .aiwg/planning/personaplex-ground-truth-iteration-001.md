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
- Strict evidence export created two examples with no rejected groups and
  `targetTextInControlOrEvidence=false`:
  `/srv/personaplex_workspace/ground_truth_runs/v4-pair-0004-evidence-20260716-0557`.
- Strict duplex export materialized only one admitted example and rejected the
  constrained branch; its independent audio/no-leakage validator passed only
  for that partial one-branch output:
  `/srv/personaplex_workspace/ground_truth_runs/v4-pair-0004-duplex-20260716-0557`.

## Blocking defect

The constrained branch is rejected as `turn_1_v4_target_evidence_lineage_invalid`.
Its record `conversationId` differs from the `conversationId` carried by its
control/evidence frames, which retain the primary branch identity. The evidence
exporter currently validates control-to-evidence alignment but does not validate
either frame against the enclosing record identity; it therefore accepted an
inconsistent pair. The partial duplex result is diagnostic only and must not be
used for tensor preparation, training, checkpoint selection, or a corpus count.

## Next gate

Repair the paired replay/materialization identity contract, add a regression
fixture making both exporters agree on a valid and an invalid pair, regenerate
the affected group through independent certification, and require a two-branch
strict duplex export before native encoding.
