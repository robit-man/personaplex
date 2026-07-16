# Public Progress Snapshot

This file records only aggregate, non-sensitive program status. Raw calls, generated
audio, reference clips, semantic state, tool evidence, model weights, credentials, and
cache paths are deliberately excluded from Git.

## 2026-07-16: V8 source-corpus calibration

The Voryn-backed V8 counterfactual generator is running six logical workers across
physical CUDA devices `0`, `1`, and `2` only. Each accepted group contains two
counterfactual branches that differ at a defined state/control pivot; a group is not
eligible for adapter training unless both branches are independently certified.

Aggregate snapshot at publication:

| Measure | Value |
| --- | ---: |
| independent certification files | 40 |
| promoted source records | 644 |
| accepted counterfactual groups | 28 |
| accepted conversations | 66 |
| WER median / p95 / maximum | 0.000 / 0.154 / 0.250 |
| ASR confidence median / minimum | 0.756 / 0.458 |

The source target is 500 accepted counterfactual groups (1,000 conversations), with
group-isolated train/validation/test splits. This snapshot is therefore **not** a
training-ready corpus and does not authorize an optimizer run.

## Gate calibration decision

The current audio thresholds are retained:

- Word error rate must be at most `0.25`.
- ASR confidence must be at least `0.45`.
- Word-level timing must be present in Whisper segment evidence.
- Every target turn must pass an independent semantic/control audit.
- Both branches of a counterfactual group must be present and valid.

The observed rejections justify these gates. The independent auditor rejected material
semantic changes caused by rendering/ASR, including negation inversions such as
`can't` becoming `can`, unsupported certainty, and placeholder-like corruption. Those
are training-invalid labels, not acceptable variance. Transport failures such as an
unavailable semantic auditor are retried with model-only structured-output repair and
are never promoted without a completed audit. A missing or mismatched counterfactual
branch is a lineage failure, not an audio-quality failure.

No threshold is loosened solely to increase throughput. Recalibration requires a
versioned held-out error analysis showing that a proposed change improves recall
without admitting a semantic, provenance, timing, or codec defect.

## Publication boundary

The `voxrn_synthesis/` package documents and launches the external Voryn synthesis
runtime. It is intentionally a bridge: the data plane remains in the Voryn checkout,
while this repository owns contracts, certification, native encoding, training, and
evaluation. This prevents raw operational data and deployment credentials from being
copied into the training suite.
