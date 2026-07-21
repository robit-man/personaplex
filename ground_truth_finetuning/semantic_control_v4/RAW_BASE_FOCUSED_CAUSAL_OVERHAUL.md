# PersonaPlex raw-base focused causal overhaul

Status: implementation and checkpoint validation in progress. This document is
ground truth for the July 2026 raw-base transition. It does not certify a model
for release and does not replace generated-audio evaluation.

## Objective

Train a mutable, typed semantic control frame as a causal input to native
PersonaPlex speech-token generation. The control frame is encoded before an
agent turn, cached by revision, and injected as a bounded temporal residual on
real delayed-duplex frames. Target text and audio remain labels only.

The release target remains at least 95% generated-audio semantic reliability
with a statistically valid lower confidence bound. Teacher-forced likelihood
diagnostics are necessary but cannot satisfy that target.

## Immutable raw-base identity

- Upstream repository: `nvidia/personaplex-7b-v1`
- Upstream revision: `fdaf4090a61cb315c138a1faee287ffd6c716309`
- Local root: `/srv/voxrn_cache/huggingface/nvidia/personaplex-7b-v1/fdaf4090a61cb315c138a1faee287ffd6c716309`
- Raw model bytes: `16742874000`
- Raw model SHA-256: `db1290db583cdaa6cb4de444ed279e0b586ca2a372b41434b07a7461c8c0e2f4`
- Mimi SHA-256: `09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50`
- SPM SHA-256: `78d4336533ddc26f9acf7250d7fb83492152196c6ea4212c841df76933f18d2d`
- Voice archive SHA-256: `8564e9ca7a06ca723b07c3a77c623f0faa5937d04b2647b3a727b06c5ca0b7bb`
- Native contract: `/srv/voxrn_cache/personaplex/contracts/personaplex-7b-v1-fdaf4090-control-v4.mimi-bound.json`

These identities match NVIDIA's upstream LFS metadata. The old
`student_best.pt` has SHA-256
`8dde8925f94bbc2c41664b4195ffa4d849775e81c678a47a67a290f6c3344932`
and is a separate historical derivative. It must not be described as raw
PersonaPlex or used as an implicit fallback for this training line.

## Raw-bound causal corpus

Canonical root:

`/srv/voxrn_cache/personaplex-transition/v4-causal-redesign-20260719/11_canonical_native_raw_exactdiff`

Certified counts:

- Train pairs: 310
- Validation pairs: 46
- Test pairs: 37
- Total causal pairs: 393
- Manifest rows: 1079, including stale-control negatives
- Exact-label-contrast rejections: 3 train pairs
- Split-leakage groups: 0

The source code tokens are Mimi/SPM-bound. Rebinding from the historical LM
lineage to the verified raw LM does not alter any native tensor bytes. The
canonicalizer validates codec hashes against the selected model contract,
records both source and output model revisions, and certifies the resulting
manifest hash.

## Exact branch-difference supervision

Whole-response text NLL diluted the few words causally changed by a control
frame. After three full exposures, the previous broad run still produced zero
strict held-out pairs and mean whole-response separation of only `+0.0002`.

The corrected objective constructs a loss-only mask using exact dynamic
programming over the longest common subsequence of each branch's supervised
agent text tokens:

1. Read only target-label token positions from branch A and branch B.
2. Compute an exact LCS. No regex, semantic heuristic, fuzzy matcher, or judge
   participates.
3. Mark tokens outside the shared subsequence independently in both branches.
4. Require at least one branch-distinct token in both directions.
5. Apply the focused causal ranking loss only to those marked tokens.
6. Continue full matched text plus agent-audio loss over the complete target.
7. Keep target labels and masks out of serialized control input and runtime
   messages.

The final canonicalizer rejects a pair before artifact emission if either
direction lacks exact target contrast. Three of 396 prior pairs failed this
test; they are recorded in the new certificate rather than silently skipped.

## Adapter architecture

Architecture revision: `lexical-attention-rms-bounded-v4`.

The adapter receives frozen PersonaPlex text embeddings plus typed channels for
field, value kind, source, revision, and position. A trainable transformer and
cross-attention compressor produce one residual row per controlled native
frame. Cross-attention weights also pool the original frozen lexical embeddings
into a direct semantic path, preserving pretrained lexical geometry for unseen
facts.

Both learned and lexical paths use zero-centered residual gates. At fresh
initialization, all control streams are exactly zero. Own, sibling, null, and
stale likelihoods therefore match exactly and base PersonaPlex behavior is
preserved.

Input-dependent gate adjustments are independently bounded for learned and
lexical paths. The combined residual is then RMS-capped relative to the actual
frozen lexical embedding RMS for that model and control. This is model-relative
normalization, not an absolute hidden-state magic number.

Every checkpoint records these dimensionless architecture limits in
`adapter_config`. Incompatible older checkpoints fail strict configuration or
state loading.

## Empirical failure sequence

### Whole-response-only objective

- A 21-pair overfit reached 20/21 strict train pairs, proving adapter capacity.
- The 313-pair run remained at 0/32 strict held-out pairs through step 420.
- Root cause: sparse causal differences were averaged across the entire reply.

### Unbounded lexical passthrough

- Focused train separation reached `+0.099` within three steps, proving that the
  direct lexical route carried causal information.
- Baseline null-minus-own NLL was approximately `-4`, proving the initial
  residual damaged the raw base.

### Zero static gate without context bounding

- Fresh step-zero deltas were exactly `0.0`, validating no-op initialization.
- By step 21, the reported static gate was only `0.000417`, but controlled NLL
  had degraded by roughly five points.
- Direct measurement showed context adjustment mean `0.2458`, effective gate
  mean `0.2372`, control-stream RMS `0.3184`, native embedding RMS `0.02772`,
  and a control/native RMS ratio of `11.49x`.
- Root cause: input-dependent context gates escaped the static-gate telemetry.

### Corrected bounded architecture

- Effective gates and post-cap stream/lexical RMS ratio are now first-class
  training telemetry.
- `control_stream_rms` is derived from the actual conditioned streams used for
  the objective.
- Null/base preservation is weighted independently from focused causal
  discrimination.

## Required evaluation ladder

1. Unit contracts pass for typed encoding, exact target contrast, no-op
   initialization, null-zero behavior, temporal alignment, RMS cap, and release
   statistics.
2. Three-GPU CUDA smoke loads the raw safetensors on physical GPUs 0, 1, and 2,
   executes forward/backward/optimizer/checkpoint evaluation, and remains below
   the dynamic host-memory limit.
3. A bounded 21-pair run must improve focused train separation without material
   null/base degradation.
4. A 21-pair overfit must reach at least 95% strict train-pair discrimination.
5. Full training must show monotonic held-out improvement across independently
   generated pair groups; train memorization alone is insufficient.
6. The selected checkpoint must pass stale-revision, null-control, cancellation,
   and interruption state tests.
7. A live controlled server must consume typed revisioned updates at turn
   boundaries and reject stale revisions.
8. Held-out paired calls must share identical caller audio and differ only in
   control state.
9. Generated speech must be decoded and evaluated with CUDA ASR, audio/codec
   checks, typed semantic inference, factual support, forbidden-claim avoidance,
   next-goal advancement, interruption recovery, and pair discrimination.
10. First-audio latency, barge-in cutoff latency, stale-audio cancellation, and
    Twilio framing must meet the runtime contract.
11. Final reliability uses first attempts, includes inference/judge failures in
    the denominator, covers at least 1000 trials and 250 causal pairs, and
    requires the configured Wilson lower bound at or above 95%.
12. Only a checkpoint passing generated-audio gates can be exported or described
    as semantically controllable.

## Training resource contract

- CUDA only; no model or ASR CPU fallback.
- Physical training GPU allow-list: 0, 1, 2.
- GPU admission is discovered dynamically from total/free memory and live
  utilization.
- Host-memory throttling begins only at the configured ratio of discovered
  physical RAM; current policy is 80%.
- NCCL transport is probed before launch. Native P2P is preferred, with measured
  shared-memory transport selected only when native collectives fail.
- Model weights, corpora, caches, runs, and publication staging remain under
  `/srv`.

## Publication policy

The raw mirror target is
`cudabenchmarktest/personaplex-7b-v1-raw-mirror`. It must contain the verified
upstream bytes, immutable revision, checksums, original model card, NVIDIA
license copy and required notice, and Moshi/CC-BY attribution. Semantic-control
checkpoints belong in a separate repository and must not be published as a
successful model until the generated-audio release gate passes.

## Research anchors

- PersonaPlex: https://arxiv.org/abs/2602.06053
- Moshi: https://arxiv.org/abs/2410.00037
- MoshiRAG: https://arxiv.org/abs/2604.12928
- Preference alignment for spoken dialogue: https://arxiv.org/abs/2506.21463
- Multi-faceted spoken-dialogue alignment: https://arxiv.org/abs/2606.11167

The implementation draws from MoshiRAG's causal streaming-conditioning
principle, but the typed revision protocol, exact pair objective, no-op residual
initialization, and model-relative RMS gate are local design decisions that must
be validated empirically.
