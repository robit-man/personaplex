# Semantic Control v4 Empirical Findings

Status: execution evidence through 2026-07-19. This document records local
observations separately from published claims in `RESEARCH_SYNTHESIS.md`.

## 1. Control-v3 is not a semantic-control release candidate

The completed 12,000-step v3 adapter learned coarse plan sensitivity, but
removing mutable textual context changed held-out loss by only about `0.0009`.
Its fresh 32k by 4096 embedding table, atomized text values, matched-only
objective, and prefetched virtual prefix do not establish causal fact or tool
following.

Evidence:

- `/srv/voxrn_cache/personaplex-transition/prepared/v7-p1000v5-controlv3-20260718T212644Z/06_training_execution/attempt-20260718T221058Z/training`

## 2. Replayed source audio did not imply native tensor identity

The source corpus contained 396 metadata-level candidate causal pairs after
quality quarantine. Only 52 initially had equal `prefix_at` boundaries, and
only 15 had byte-identical native tensors through the pivot. The branch audio
content was replayed, but independently generated timelines introduced small
boundary drift before native encoding. Training those rows as counterfactuals
would allow audible-history differences to explain the label and invalidate
the causal claim.

The repair does not loosen the identity gate. It selects one provenance-bound
donor prefix per pair, splices each branch's own post-pivot suffix, shifts the
target mask, re-encodes boundary metadata, and certifies exact tensor identity
through the pivot.

Certified result:

- 396 exact causal pairs: 313 train, 46 validation, 37 test.
- 1,088 examples: 792 pair members plus 296 stale-control negatives.
- `/srv/voxrn_cache/personaplex-transition/v4-causal-redesign-20260719/02_canonical_native/corpus_certificate.json`
- `/srv/voxrn_cache/personaplex-transition/v4-causal-redesign-20260719/02_canonical_native/pair_certificate.json`

Future synthesis must render and encode a shared prefix once, branch only at
the declared pivot, and fail certification if prefix tensors differ.

## 3. Full-history backward was unnecessary and exhausted VRAM

An uncropped single-GPU backward replicated several control variants over a
multi-minute call and exhausted an A100 near 73 GiB. The causal target only
depends on recent audible history, the control boundary, and the supervised
target. `PairData` now derives frame counts from the codec contract and keeps a
configurable recent-history window plus target tail. The caller remains context
only and target loss remains agent-only.

The cropped single-GPU smoke completed forward, activation-recomputed backward,
optimizer update, checkpoint evaluation, and final checkpoint emission:

- `/srv/voxrn_cache/personaplex-transition/v4-causal-redesign-20260719/03_smoke-cropped-20260720T004327Z`

## 4. Distributed startup failure was an NCCL transport failure

All three 7B base models loaded in 11-13 seconds after model loading was moved
before NCCL initialization. A later 220 MiB DDP synchronization still hung with
all A100s reporting 100% utilization. A standalone collective probe reproduced
the failure outside PersonaPlex. `nvidia-smi topo -m` reports a mixed NVLink,
PCIe host-bridge, and same-NUMA topology, and CUDA reports peer access, so static
topology assumptions were insufficient.

Measured on the admitted GPU order:

- Native NCCL: bounded probe timed out after 12.55 seconds.
- P2P-disabled shared-memory NCCL: 220 MiB broadcast about 40 ms and all-reduce
  about 67 ms in the direct diagnostic.
- Production capacity-relative preflight: shared-memory mode passed in about
  3.98 seconds including process startup.

The launcher now runs a capacity-relative collective preflight on the exact
admitted GPU set. It uses native NCCL only if it passes; otherwise it kills the
entire bounded probe process group, tests shared-memory transport, records the
decision, and refuses training if neither path passes. This is live discovery,
not a server-specific port, GPU-memory, or topology magic number.

Evidence:

- `/srv/voxrn_cache/personaplex-transition/v4-causal-redesign-20260719/03_smoke-ddp-transportfix-20260720T010259Z/transport_preflight.json`

## 5. Three-A100 causal training is operational

The distributed smoke completed transport admission, model loading, checkpoint
resume support, baseline evaluation, matched and counterfactual forward passes,
activation-recomputed backward, gradient reduction, optimizer update, and
checkpoint/final artifact emission on physical CUDA devices 0, 1, and 2.

Observed envelope:

- Maximum host-memory ratio: `0.3978`, below the dynamic `0.80` limit.
- Peak aggregate GPU memory observed by physical index: 58,815 MiB on GPU 0,
  39,363 MiB on GPU 1, and 37,795 MiB on GPU 2, including unrelated services.
- All selected GPUs reached 100% compute utilization during work.
- Step 1 reduced held-out mean null-minus-own text NLL from `-5.0625` to
  `-3.5859`; one step is deliberately insufficient for pair convergence.

Evidence:

- `/srv/voxrn_cache/personaplex-transition/v4-causal-redesign-20260719/03_smoke-ddp-transportfix-20260720T010259Z`

## 6. What is and is not validated

Validated now:

- Exact causal native-prefix construction and certification.
- CUDA-only single- and three-GPU Stage-1 execution.
- Dynamic host-memory admission and sampled NVIDIA telemetry.
- Trainable temporal control-stream influence in the native PersonaPlex forward
  path.
- Typed runtime revision acknowledgement, boundary snapshots, stale rejection,
  and cancellation plumbing at structural/unit-test scope.

Not yet validated:

- Converged held-out causal discrimination.
- Free-running spoken fact, goal, style, and action adherence.
- At least 1,000 first-attempt generated-audio trials and 250 causal pairs.
- Wilson lower bound at or above 95% across the preregistered release set.
- Zero stale/policy-critical emissions in paced duplex/Twilio failure trials.

No checkpoint may be described as 95% reliable until those generated-audio and
live-equivalent artifacts exist.
