# PersonaPlex-Native MoshiRAG Continued Pretraining

**Status:** Architecture, training, runtime, and promotion specification
**Date:** 2026-07-20
**Scope:** The next semantic-control generalization experiment after the ordered-prefix and upper-temporal-LoRA runs

## 1. Decision

The next trainable system is a PersonaPlex-native MoshiRAG conditioner with full-rank adaptation of the complete temporal and text receiver.

The required design is:

- Keep PersonaPlex as the base model and preserve its voice and full-duplex behavior.
- Encode target-free canonical control text with frozen ARC-4.
- Deliver both an eight-row turn-boundary burst and the detailed ARC-4 sequence through native `streaming_sum` addition on real 12.5 Hz temporal frames.
- Train all 32 temporal-transformer layers, the PersonaPlex text embedding and text head, and the new conditioner projections at full rank.
- Initially freeze ARC, Mimi, the audio-code embeddings, the depth transformer, audio heads, and the existing voice-prompt path.
- Train with agent text/audio SFT, listwise matched-versus-counterfactual discrimination, an auxiliary pre-response control-state objective, and PersonaPlex replay.
- Preserve constant-size per-frame inference. Do not add temporal-layer text cross-attention or advance the temporal cache with synthetic silent prefix frames.

This is not a claim of generated-audio success. Existing local semantic-control results are teacher-forced diagnostics. This document does not assert a 95% result or any equivalent production-quality result.

## 2. Evidence and inference boundary

This document uses three labels.

| Label | Meaning |
|---|---|
| **EVIDENCE** | Directly established by an official paper/repository or by a named local source artifact. |
| **INFERENCE** | An engineering conclusion drawn from the evidence but not directly demonstrated by the source. |
| **DECISION** | A project requirement for the next run. A decision can be evidence-informed without being an upstream MoshiRAG prescription. |

The distinction applies section by section. Local run metrics establish behavior only for the recorded checkpoints and evaluation harness. They do not establish free-running speech behavior.

## 3. Primary-source reconciliation

### 3.1 Official MoshiRAG architecture

**EVIDENCE:** Official MoshiRAG uses a frozen ARC text encoder, ARC compression by four, a trainable projection to the 4096-dimensional Moshi temporal space, and additive streaming conditioning at 12.5 Hz. The released configuration uses `streaming_sum` and sets transformer cross-attention to false. See the [official MoshiRAG paper](https://arxiv.org/html/2604.12928v1), [official MoshiRAG repository](https://github.com/kyutai-labs/moshi-rag), [pinned local configuration](/srv/voxrn_cache/personaplex/source/moshi-rag-official/configs/moshirag.json#L60), [ARC conditioner implementation](/srv/voxrn_cache/personaplex/source/moshi-rag-official/moshi/moshi/conditioners/arc_encoder.py#L440), and [condition fuser](/srv/voxrn_cache/personaplex/source/moshi-rag-official/moshi/moshi/conditioners/base.py#L465).

**EVIDENCE:** The pinned implementation's ARC bridge is two bias-free linear projections, 3072 to 2048 to 4096. ARC runs with gradients disabled when `finetune` is false. Natural text is tokenized by ARC; it is not reduced to a freshly initialized table of atomized control IDs. See the [bridge and conditioner source](/srv/voxrn_cache/personaplex/source/moshi-rag-official/moshi/moshi/conditioners/arc_encoder.py#L390).

**EVIDENCE:** MoshiRAG trains the model broadly while freezing the reference encoder. The paper reports reference dropout, randomized retrieval timing, at least a one-second buffer before grounded content in its synthetic timing construction, approximately 1.9 million conversations, and approximately 47,770 hours. The official paper reports that insertive ARC conditioning outperformed additive ARC conditioning, but additive conditioning was selected for long-conversation efficiency. Text cross-attention did not converge as well as insertive conditioning in that experiment and was omitted.

**EVIDENCE:** The PyTorch and Rust paths consume one condition row per executed temporal frame. Exhausted slots become zero. The condition does not require a growing reference KV cache. See the [PyTorch condition state](/srv/voxrn_cache/personaplex/source/moshi-rag-official/moshi/moshi/models/lm.py#L729) and [Rust production generator](/srv/voxrn_cache/personaplex/source/moshi-rag-official/rust/moshi-core/src/lm_generate_multistream.rs#L268).

**EVIDENCE:** MoshiRAG separates reference encoding from the real-time generator and performs retrieval/encoding asynchronously with cancellation and timeout handling. See the [conditioner service](/srv/voxrn_cache/personaplex/source/moshi-rag-official/moshi/moshi/server_conditioner.py#L173) and [RAG manager](/srv/voxrn_cache/personaplex/source/moshi-rag-official/moshi/moshi/inference_utils/rag_manager.py#L44).

**INFERENCE:** The additive interface is not the main reason the local semantic controller is weak. Official MoshiRAG makes a simple interface useful by adapting the receiving model broadly. Adding more complexity to a mostly frozen side adapter is less evidence-aligned than teaching the complete temporal/text receiver to consume the native stream.

### 3.2 Official PersonaPlex architecture

**EVIDENCE:** PersonaPlex learns role and behavior conditioning from lexical text prompts paired with voice prompts and full-duplex conversations. Its text prompt is represented in the model's normal text-token history rather than by a new semantic-control vocabulary. See the [official PersonaPlex paper](https://arxiv.org/abs/2602.06053), [official NVIDIA PersonaPlex page](https://research.nvidia.com/labs/adlr/personaplex/), and [official PersonaPlex repository](https://github.com/NVIDIA/personaplex).

**EVIDENCE:** The baseline PersonaPlex LM sums text and audio embeddings before the temporal transformer and has no general MoshiRAG condition-provider branch. See the [local PersonaPlex LM](/srv/voxrn_cache/personaplex/source/moshi/moshi/models/lm.py#L435).

**INFERENCE:** Existing lexical representations should remain part of the trainable receiving path. A freshly initialized semantic-ID embedding plus a small frozen-base adapter should not be the only route through which the model must discover the meaning of control fields.

### 3.3 Imported checkpoint and compatibility evidence

**EVIDENCE:** The imported Moshika base and MoshiRAG checkpoints have 343 shape-compatible tensors in common with PersonaPlex. All 343 changed numerically in the MoshiRAG checkpoint. The changed set includes all 192 temporal-transformer tensors, the text embedding, the text head, and Moshika audio/depth components. Twelve depth tensors are incompatible with PersonaPlex because PersonaPlex uses different depth dimensions/codebook structure. See the [numeric comparison](/srv/voxrn_cache/personaplex/compatibility/moshirag-personaplex-v1.numeric.json), [structural compatibility report](/srv/voxrn_cache/personaplex/compatibility/moshirag-personaplex-v1.json), and [verified release import](/srv/voxrn_cache/personaplex/imports/moshirag-release-v1.full.json).

**EVIDENCE:** The current Moshika RAG task-vector import is restricted to the temporal transformer. It transfers 192 tensors containing approximately 6.577 billion parameters but does not transfer the changed text path or incompatible depth path. The local patched PersonaPlex model adds the conditioner branch over an otherwise untransformed NVIDIA checkpoint. See the [patched model contract](/srv/voxrn_cache/personaplex/contracts/personaplex-7b-v1-fdaf4090-control-v4.mimi-bound.json).

**INFERENCE:** Shape compatibility is not semantic portability. The Moshika temporal task vector may be retained as an initialization ablation, but it must not be treated as the primary learned PersonaPlex control solution.

### 3.4 Local failure evidence

**EVIDENCE:** The current semantic prefix is a newly initialized 32K-by-4096 embedding, a small encoder/query pooling path, and a gate while the base model remains frozen. See the [semantic-prefix implementation](/srv/personaplex_workspace/robit-man-personaplex/ground_truth_finetuning/training/semantic_prefix.py#L12).

**EVIDENCE:** The upper temporal LoRA trains only four temporal layers and only attention output and MLP projections. It does not train attention Q/K/V. Its recorded trainable size is approximately 1.606 million parameters. See the [temporal-LoRA implementation](/srv/personaplex_workspace/robit-man-personaplex/ground_truth_finetuning/training/temporal_lora.py#L99) and [upper-LoRA run contract](/srv/voxrn_cache/personaplex/training/runs/arc4-v10-upper-lora-step25-to50-20260720T100208Z/run_contract.json#L87).

**EVIDENCE:** In the recorded teacher-forced evaluations, the ordered-prefix run passed 6 of 46 strict pairs, the upper-LoRA continuation passed 2 of 46, the earlier two-path run passed 0 of 46, and the broader layer-adapted diagnostic also passed 0 strict pairs. See the [ordered-prefix contract](/srv/voxrn_cache/personaplex/training/runs/arc4-ordered-prefix-sanitized-v9-gate050-resume25-to50-20260720T093544Z/run_contract.json), [two-path contract](/srv/voxrn_cache/personaplex/training/runs/arc4-two-path-ragdelta-v8-smoke25-20260720/run_contract.json), and [layer-adapted diagnostic](/srv/voxrn_cache/personaplex/training/runs/arc4-causal-layer-adapted-v7-20260720/curve_diagnostic.json#L141).

**INFERENCE:** These results justify a capacity and objective change. They do not isolate LoRA as the only causal variable, and they do not prove that the bounded burst is useless. The next experiment keeps the latency-safe interface and changes the receiver's trainability and causal supervision.

## 4. Required architecture

### 4.1 Condition representation

**DECISION:** Every semantic-control payload must have two representations:

| Representation | Purpose | Runtime form |
|---|---|---|
| Typed envelope | Revision, validity, cancellation, deadline, field provenance, and audit semantics | Runtime metadata, never inferred from generated text |
| Canonical target-free text | Learned semantic content | Frozen ARC-4 input |

The canonical text must use ordinary lexical field labels and values. It must not contain the desired agent answer, a paraphrase of the answer, future transcript text, or generated-audio targets. Mutable values must remain explicit natural text. Hashes and categorical IDs may accompany the envelope for identity, but they must not replace lexical content.

### 4.2 Frozen ARC-4 path

For canonical text `z`, the detailed condition is:

```text
R = stop_gradient(ARC4(z))                    # [T_ref, 3072]
D = P_ref(R)                                  # [T_ref, 4096]
P_ref: Linear(3072, 2048) -> Linear(2048, 4096)
```

**DECISION:** ARC-4 weights and tokenization remain pinned to the imported release revision. `P_ref`, learned in-sequence padding, and the conditioner normalization are trainable. No nonlinear activation is inserted between the two bridge projections in the upstream-parity path.

### 4.3 Turn-boundary burst

The turn-boundary lane exists to make the complete control state available during the first agent frames. It is not a replacement for the detailed stream.

```text
B = BoundaryResampler(R)                      # [8, 4096]
C_turn[t] = B[t] for 0 <= t < 8, else 0
C_ref[t]  = D[t] for 0 <= t < min(T_ref, 96), else 0
C[t]      = C_turn[t] + C_ref[t]
```

**DECISION:** `BoundaryResampler` uses eight learned queries over frozen ARC rows, a 1024-dimensional trainable bottleneck, one eight-head cross-attention pooling block, one 1024-to-4096 output projection, and no temporal-LM cross-attention. The resampler is full-rank trainable. It may be initialized from the current ordered-prefix adapter where tensor meaning and shape match, but no such import is required.

**DECISION:** The burst lasts exactly eight real temporal frames, or 640 ms at 12.5 Hz. The detailed stream is bounded to 96 rows, or 7.68 seconds. Longer canonical text must be rejected or deterministically reduced by the data contract before ARC encoding; runtime truncation may not silently drop fields.

**DECISION:** There is no global sigmoid gate initialized near zero. Each lane has a trainable scalar initialized to `1.0`, and its pre-addition RMS is logged. A hard clamp is permitted only as a documented numerical-safety intervention after observed instability, not as the default semantic bottleneck.

**INFERENCE:** The burst pooling block is a project design, not an official MoshiRAG component. Its role is bounded early availability. The official evidence for semantic integration resides primarily in the broadly trained receiver and the detailed ARC stream.

### 4.4 Native `streaming_sum` fusion

At executed temporal frame `t`:

```text
x[t] = audio_embedding[t] + text_embedding[t] + C[t]
h[t] = TemporalTransformer(x[t], kv_cache)
```

**DECISION:** A condition row is consumed only when a real temporal frame is executed. Preparing, attaching, cancelling, or exhausting a condition must not advance the temporal position, create a fake audio frame, or modify the self-attention KV cache by itself.

**DECISION:** The fuser operates at 12.5 Hz, one row every 80 ms. If a stream is exhausted, absent, expired, or cancelled, its contribution is exact zero. Learned padding is allowed only inside an accepted encoded sequence; an absent condition must preserve zero-condition parity.

**DECISION:** No per-layer condition cross-attention is part of this run. No condition tokens are inserted into the temporal sequence. These alternatives require separate ablations and cannot be mixed into the primary run.

## 5. Exact full-rank train/freeze matrix

Any parameter not explicitly marked **TRAIN** below is frozen. The launch contract must enumerate every named parameter, its element count, dtype, and state. A run with an unclassified parameter is invalid.

| Component | State | Required scope |
|---|---|---|
| Temporal transformer blocks 0 through 31 | **TRAIN, full rank** | All self-attention Q, K, V, output projections; all MLP/gating input and output projections; all block norms; all biases or learned scales present in the implementation |
| Temporal input/final normalization associated with the 32-layer stack | **TRAIN, full rank** | Every learned normalization parameter on the temporal path |
| PersonaPlex text embedding | **TRAIN, full rank** | Complete embedding table used by the temporal LM |
| PersonaPlex text output head | **TRAIN, full rank** | Complete text-logit projection and associated learned normalization/bias |
| ARC detailed bridge `P_ref` | **TRAIN, full rank** | 3072-to-2048 and 2048-to-4096 projections |
| Boundary resampler | **TRAIN, full rank** | Learned queries, 3072-to-1024 input projection, one eight-head pooling block, FFN/norms, and 1024-to-4096 output projection |
| Conditioner lane scales | **TRAIN** | One scalar for turn burst and one for detailed reference, both initialized to `1.0` |
| Conditioner in-sequence padding | **TRAIN** | Only padding used within an accepted ARC sequence |
| Auxiliary control-state heads | **TRAIN, training only** | Field-value contrastive projections and ready/stale/null classifier; excluded from inference checkpoints |
| ARC-4 token embedding and all ARC encoder layers | **FREEZE** | Execute under `no_grad`, or consume provenance-locked precomputed ARC features |
| Mimi encoder, decoder, quantizer, and codebooks | **FREEZE** | Prefer precomputed Mimi codes for training |
| PersonaPlex audio-code input embeddings | **FREEZE** | All codebook embeddings on the temporal input path |
| Depth transformer/Depformer | **FREEZE initially** | All depth attention, MLP, norms, projections, and embeddings |
| Audio-code output heads | **FREEZE initially** | Every codebook output projection |
| Existing PersonaPlex voice-prompt path | **FREEZE initially** | Speaker/voice conditioning parameters and audio prompt representation |
| Existing first-speaker conditioner, if present | **FREEZE initially** | Preserve session initialization behavior |
| Moshika RAG temporal task vector | **OFF in primary run** | Allowed only as a separately named initialization ablation |
| Upper-layer LoRA modules | **ABSENT** | Do not stack LoRA on the full-rank primary run |
| Runtime condition scheduler/fuser | **NO PARAMETERS** | Deterministic queueing and addition only |

**DECISION:** Full rank means optimizer ownership of the original parameters. Training only LoRA, only the upper four layers, only adapter gates, or only attention output/MLP projections does not satisfy this specification.

**INFERENCE:** Freezing depth/audio for the first run is a PersonaPlex-preservation compromise. Official MoshiRAG trained all model parameters except the reference encoder, but the imported Moshika depth path is not shape-compatible with PersonaPlex and cannot be transferred as if it were equivalent.

## 6. Three-A100 memory-aware training plan

### 6.1 Memory premise

**EVIDENCE:** The temporal transformer alone contains approximately 6.577 billion trainable parameters in the imported comparison. A conventional mixed-precision AdamW training copy can require on the order of 16 bytes per trainable parameter across model parameters, gradients, FP32 master parameters, and two FP32 optimizer moments before implementation-specific savings. The temporal path therefore cannot be safely replicated with optimizer state on each of three A100s.

**DECISION:** Memory pressure must be addressed by sharding, checkpointing, shorter windows, accumulation, or CPU offload. It must not be addressed by silently reducing the trainable matrix to upper-layer LoRA.

### 6.2 Input preprocessing

**DECISION:** Mimi codes and ARC-4 embeddings are precomputed into immutable training shards whenever augmentation does not alter their source text/audio. Every shard records:

- PersonaPlex tokenizer revision.
- Mimi revision and codebook contract.
- ARC repository commit and checkpoint SHA-256.
- Canonical payload hash.
- Source audio hash.
- Temporal alignment and revision-arrival schedule.

ARC and Mimi therefore consume no persistent training-GPU memory in the normal path. Online recomputation is permitted only for a diagnostic run and must use the same pinned revisions.

### 6.3 Preferred FSDP profile

Use this profile when all three A100s expose at least 75 GiB each.

| Setting | Required value |
|---|---|
| World size | 3 |
| Precision | BF16 parameters/compute, FP32 optimizer moments |
| Sharding | FSDP `FULL_SHARD` for parameters, gradients, and optimizer state |
| Auto-wrap | One FSDP unit per temporal block; separate units for frozen depth blocks and trainable text/conditioner modules |
| Original parameters | `use_orig_params=True` or equivalent mixed frozen/trainable support |
| Activation checkpointing | Every temporal block and every depth block traversed by audio loss |
| Attention kernel | Memory-efficient SDPA or Flash Attention supported by the pinned environment |
| Microbatch | One conversation window per GPU |
| Initial window | At most 375 temporal frames, 30 seconds at 12.5 Hz |
| Gradient accumulation | 16 microsteps, effective 48 windows per optimizer update |
| Sequence packing | Off for the first run; causal sibling and revision boundaries must remain explicit |
| Gradient clipping | Global norm `1.0` after unsharding-aware reduction |
| CPU offload | Off initially |
| All-gather control | Limit concurrent all-gathers; use backward prefetch only if measured memory permits |

The optimizer starts with three parameter groups:

| Group | Parameters | Initial learning rate | Weight decay |
|---|---|---:|---:|
| Receiver | Full temporal transformer | `1.0e-5` | `0.1` excluding norms/biases |
| Lexical path | Text embedding and text head | `5.0e-6` | `0.1` excluding norms/biases |
| Conditioner | ARC bridge, boundary resampler, scales, state heads | `1.0e-4` | `0.01` excluding queries/scales/norms/biases |

Use AdamW with betas `(0.9, 0.95)`, epsilon `1e-8`, a 2% linear warmup, and cosine decay to 10% of each group's initial learning rate. These values are **DECISIONS** for the initial run, not published MoshiRAG hyperparameters.

### 6.4 ZeRO-3 constrained-memory profile

Use this profile when the A100s expose approximately 40 GiB each or when the preferred profile exceeds the measured memory envelope.

| Setting | Required value |
|---|---|
| World size | 3 |
| Sharding | DeepSpeed ZeRO stage 3 or an equivalent fully sharded implementation |
| Optimizer offload | CPU offload enabled with pinned memory |
| Parameter offload | Off first; enable for frozen depth/audio modules before reducing trainable scope |
| Precision | BF16 forward/backward, FP32 optimizer state |
| Microbatch | One window per GPU |
| Initial window | 240 temporal frames, 19.2 seconds |
| Gradient accumulation | 32 microsteps, effective 96 windows per optimizer update |
| Activation checkpointing | Every temporal and traversed depth block |
| Prefetch buckets | Sized from measured free memory, never copied from an unrelated model configuration |
| ARC/Mimi residency | None; use precomputed features/codes |

If the constrained profile still fails, apply remedies in this order:

1. Reduce temporal window length while preserving complete causal spans.
2. Increase gradient accumulation to preserve effective batch size.
3. Offload frozen depth/audio parameters.
4. Offload trainable parameters with ZeRO-3/NVMe only if host bandwidth is measured adequate.
5. Reduce evaluation batch concurrency.
6. Stop and revise the training infrastructure if full receiver ownership still cannot fit.

Upper-layer LoRA is not the fallback for an out-of-memory error in this experiment.

### 6.5 Distributed checkpoint format

**DECISION:** Write sharded checkpoints without gathering the full model on GPU. A checkpoint is resumable only if it contains:

- Model shards and a parameter-name manifest.
- Optimizer and scheduler shards.
- Global optimizer step and consumed temporal-frame count.
- RNG state for every rank.
- Sampler/data-cursor state.
- Train/freeze matrix and hash.
- Base, ARC, Mimi, tokenizer, source-code, and data-manifest revisions.
- Objective weights and timing curriculum state.
- FSDP/ZeRO configuration and world size.
- Eval results tied to the exact checkpoint hash.

Save a resumable sharded checkpoint every 500 optimizer updates. Run the causal validation set at the same interval. Produce a CPU-consolidated model only for a checkpoint that passes the structural and teacher-forced gates.

## 7. Data contract and curriculum

### 7.1 Causal sibling groups

Every controlled training item belongs to a sibling group with the same audible history through the causal boundary.

| Sibling | Required difference |
|---|---|
| Matched | Correct current control revision and compatible target continuation |
| Single-field swap | Exactly one material field changed and a correspondingly changed target continuation |
| Multi-field swap | Two or more material fields changed |
| Stale | Superseded revision paired with the current audible history |
| Null | No semantic-control payload |
| Late | Correct payload arrives after its declared attachment deadline |
| Paraphrase | Same semantics expressed with different lexical serialization |

All siblings must be assigned to the same data split. Splits must also isolate scenario templates, entity combinations, and source conversations so that surface-form memorization cannot create a false causal result.

**DECISION:** The first full-rank run requires at least 10,000 leaf conversations with material sibling coverage. This is a minimum experiment corpus, not parity with MoshiRAG's reported scale and not a production sufficiency claim.

### 7.2 Timing curriculum

Training examples must separate lead, causally affected body, and tail regions.

**DECISION:** Use the following starting timing distribution:

| Condition | Share of controlled examples |
|---|---:|
| Complete burst and detailed stream ready at turn start | 60% |
| Burst ready, detailed stream attached before affected body | 20% |
| Detailed stream misses deadline or is cancelled | 10% |
| Stale revision presented and rejected | 10% |

Apply independent 20% conditioner dropout during training. Dropout produces an exact-zero absent stream, not a different textual value. Randomize the ready-to-body gap, with at least one second in examples intended to teach asynchronous pre-grounding behavior. Never train the model to rely on a late reference after its declared deadline.

### 7.3 SFT objective

Mask control text, system prompts, and user-only regions from the agent-generation loss. Optimize agent text and audio targets:

```text
L_sft = L_agent_text + 0.02 * L_agent_nonsemantic_audio
```

The `0.02` non-semantic audio weight follows the published PersonaPlex training description. Causally affected text spans receive a `2.0` token weight inside `L_agent_text`; unchanged context retains weight `1.0`.

### 7.4 Listwise counterfactual objective

For audible history `h`, target continuation `y_j`, matching control `c_j`, and incompatible sibling controls `c_k`, define a focused score as negative token NLL over the causally affected span:

```text
s(h, c_k, y_j) = -NLL_affected(y_j | h, c_k)

L_listwise(j) = -log(
    exp(s(h, c_j, y_j) / tau) /
    sum_k exp(s(h, c_k, y_j) / tau)
)
```

**DECISION:** Set `tau=0.1`. Apply the objective bidirectionally across every material sibling target, not only matched target versus one wrong condition. Stale, null, and late siblings participate when their valid behavior differs from the matched branch.

**INFERENCE:** This objective is not reported by MoshiRAG. It is required here because independent hinge terms can improve average loss while leaving most strict sibling pairs unresolved.

### 7.5 Pre-response control-state objective

Attach training-only heads to the temporal hidden state immediately before the first causally affected target token.

The heads must predict:

- Ready, stale, null, cancelled, or late status.
- Active revision identity within the sibling group.
- Each material field value by contrastive matching to its frozen ARC field-value embedding.
- Whether the detailed stream was attached before its deadline.

Use in-batch and sibling values as negatives. Do not require a fixed global classifier for open-vocabulary values. Remove all state heads from runtime/exported inference checkpoints.

**INFERENCE:** The state objective is a diagnostic forcing function. It tests whether control information reaches the receiver before generation without adding inference latency.

### 7.6 Combined loss and batch mixture

For controlled examples:

```text
L_controlled = L_sft + 0.5 * L_listwise + 0.1 * L_state
```

Use this optimizer-step mixture:

| Batch source | Share |
|---|---:|
| Material causal sibling groups | 50% |
| Ordinary PersonaPlex replay without semantic control | 30% |
| Delayed, cancelled, stale, null, and interruption-focused examples | 20% |

Replay uses `L_sft` only and an exact-zero condition. No-control replay is mandatory because the temporal and text paths are being adapted at full rank.

## 8. Runtime revision, attachment, and cancellation semantics

### 8.1 Required identity

Every asynchronous encode request and condition buffer carries:

```text
call_id
turn_id
generation_id
control_revision
payload_hash
created_at
valid_from
expires_at
attach_deadline
```

`control_revision` is monotonic within its control namespace. `payload_hash` identifies immutable canonical content. A revision's content may never be mutated in place.

### 8.2 State machine

```text
EMPTY -> ENCODING -> READY -> ATTACHED -> EXHAUSTED
             |          |         |
             +----------+---------+-> CANCELLED
             +----------------------> EXPIRED
```

**DECISION:** An ARC result may transition to `READY` only when all request identity fields still match the pending slot. Attachment is atomic at a temporal-frame boundary. Rows begin at the first real frame after attachment; they are never backdated.

### 8.3 Revision rules

- A generation pins one control revision before its first controlled agent frame.
- A newer revision cannot replace an already attached revision in that generation.
- A newer revision cancels pending encoding for an older unattached revision.
- A late encoder result may remain in the content-addressed cache, but it cannot attach automatically to a different generation.
- Cache reuse requires a fresh validity check against revision, expiry, turn, and attach deadline.
- Stale, expired, or identity-mismatched results are discarded from the pending slot without mutating generated state.

### 8.4 Cancellation rules

- User interruption, turn cancellation, superseding revision, call termination, and timeout each cancel the pending or attached buffer for that generation.
- Cancellation sets all future condition rows to exact zero and invalidates pending callbacks.
- Cancellation cannot retract rows already consumed or audio already emitted.
- Async workers must check cancellation before expensive encoding, before publishing a result, and before atomic attachment.
- Callback completion after cancellation is observable telemetry, not permission to attach.

### 8.5 Deadline and safe behavior

The eight-row burst must be ready before controlled generation begins. If a required burst is unavailable, the runtime may wait, ask a noncommittal clarification, or emit a safe backchannel. It must not emit content whose correctness depends on the missing control.

The detailed stream may attach asynchronously before `attach_deadline`. If it misses the deadline, the current generation receives zero detailed rows. A later generation may reuse the cached embedding only after a new eligibility check.

### 8.6 Queue and latency contract

- Maximum detailed rows per revision: 96.
- Maximum turn-burst rows per revision: 8.
- Maximum attached revision per generation: one.
- Per-frame operation: dequeue at most two 4096-dimensional rows, add them, and advance their cursors.
- ARC encoding: once per payload hash, outside the 80 ms generation loop.
- Temporal KV growth from condition: zero.
- Synthetic temporal frames from condition: zero.

**EVIDENCE:** The local ARC smoke test produced finite, nonidentical 4096-dimensional embeddings and observed approximately 136 ms first-use and 28 ms subsequent encoding in that environment. See the [local conditioner smoke report](/srv/voxrn_cache/personaplex/eval/moshirag-conditioner-smoke/report.json#L14).

**INFERENCE:** These measurements support asynchronous precomputation but are not a service-level objective. Production latency must be measured under the deployed hardware, concurrency, cache, and payload distributions.

## 9. Checkpoint and evaluation gates

No checkpoint is selected by aggregate training loss alone. Promotion is lexicographic: satisfy preservation gates first, then maximize the worst material-field causal result, then use aggregate listwise margin as a tie-breaker.

### 9.1 Gate A: structural and provenance

A checkpoint fails immediately if any item is false:

- The train/freeze manifest exactly matches Section 5.
- ARC, Mimi, tokenizer, base model, source, and dataset revisions are recorded.
- No target or future transcript text appears in a control payload.
- Sibling examples remain in one split.
- Absent, cancelled, and exhausted conditions produce exact-zero fuser rows.
- No condition operation advances the temporal cache without a real frame.
- State heads are excluded from inference export.
- The checkpoint can resume from its sharded state with the same data cursor.

### 9.2 Gate B: step-zero parity

Before optimization, compare the assembled model with zero condition against the pinned PersonaPlex base on the replay set.

Required result:

- Text logits match within the declared BF16 numerical tolerance.
- Audio logits match within the declared BF16 numerical tolerance.
- Generated-condition queues remain empty.
- Voice-prompt and first-speaker setup take the existing code path.

The exact tolerance and kernel determinism report must be stored with the run contract. A structural mismatch must be repaired before training.

### 9.3 Gate C: teacher-forced causal validation

Use a held-out set with at least 100 material sibling comparisons per field family. The checkpoint must satisfy all of the following:

- Overall strict matched-over-incompatible pair pass rate is at least two thirds.
- No material field family has a strict pair pass rate below `0.55`.
- The median focused matched-minus-incompatible log-likelihood margin is positive in every field family.
- A stratified 90% bootstrap lower bound for the overall focused margin is above zero.
- Ready/stale/null/cancelled/late state classification balanced accuracy is at least `0.80`.
- Field-value contrastive retrieval is above its within-batch chance rate in every material field family.
- Zero-condition replay text NLL regresses by no more than 2% relative to the step-zero model.
- Zero-condition replay audio NLL regresses by no more than 2% relative to the step-zero model.

These are promotion thresholds for this experiment, not claimed achieved results.

### 9.4 Gate D: free-running generated-audio evaluation

Teacher-forced success does not prove generated speech behavior. A checkpoint passing Gate C must undergo free-running evaluation with matched sibling prompts and identical audio prefixes.

Required measurements:

- Required fact/action adherence.
- Prohibited fact/action avoidance.
- Matched versus single-field-swapped branch separation.
- Stale, null, cancelled, and late-reference behavior.
- Transcript and audio agreement.
- Speaker similarity and voice-prompt retention.
- Naturalness/intelligibility under the pinned evaluator.
- Time to first audible token.
- Full-duplex overlap, interruption, and barge-in behavior.
- Condition-attach overhead with precomputed ARC embeddings.

Promotion requires at least two thirds strict branch success overall, no material field family below `0.55`, no more than a 3 percentage-point regression on the pinned no-control duplex suite, and no more than 20 ms p90 time-to-first-audible-token overhead when the required burst is precomputed. Human review and automated scoring must both be retained with the checkpoint contract.

This gate is a future requirement. No current checkpoint is represented here as having passed it.

### 9.5 Gate E: runtime race and cancellation tests

The runtime candidate must pass deterministic tests for:

- Cancellation before ARC starts.
- Cancellation during ARC encoding.
- Cancellation after `READY` but before attachment.
- Cancellation after partial consumption.
- Newer revision arriving during older-revision encoding.
- Encoder callback arriving after generation termination.
- Expiry before attachment.
- Detailed stream missing its attachment deadline.
- Cache hit with an ineligible stale revision.
- Queue exhaustion producing exact zeros.

No race test may attach a row whose complete identity tuple differs from the active generation tuple.

## 10. Selective depth/audio unfreezing escalation

Depth/audio unfreezing is not automatic and is not triggered by aggregate loss alone.

### 10.1 Entry criteria

An escalation may begin only after a temporal-plus-text checkpoint passes Gates A through C and one of these conditions is reproduced at two consecutive evaluation checkpoints:

- Generated text follows the correct control branch, but generated audio/transcript behavior trails strict text branch success by at least 10 percentage points.
- Focused text NLL and state-head objectives have converged, while focused audio-code NLL remains flat across three evaluations and free-running audio omits or corrupts branch-specific content.
- Controlled listening behavior fails on interruption or user-audio distinctions despite correct textual control-state decoding, and attribution points to frozen audio input/depth capacity rather than stale timing or data leakage.

The attribution report must include matched hidden-state probes, text logits, audio-code logits, generated transcripts, revision timing, and replay comparisons. A conditioner or timing defect must be fixed before unfreezing depth/audio.

### 10.2 Escalation order

Each row is a separate run. Do not combine steps without measuring the preceding one.

| Stage | Newly trainable components | Components that remain frozen |
|---|---|---|
| D1 | Audio-code output heads and depth input/output projections | Depth attention/MLP blocks, audio input embeddings, Mimi, voice-prompt path |
| D2 | D1 plus the top two depth-transformer blocks, full rank | Remaining depth blocks, audio input embeddings, Mimi, voice-prompt path |
| D3 | Complete depth transformer and audio-code heads, full rank | Audio input embeddings, Mimi, voice-prompt path |
| A1 | D3 plus audio-code input embeddings | Mimi and any separately parameterized speaker/voice encoder |

Every escalation keeps the full temporal/text matrix trainable and reduces the new component's learning rate to one quarter of the temporal receiver learning rate. Increase no-control PersonaPlex replay from 30% to 50% for D2 and later.

### 10.3 Escalation stop conditions

Stop and reject the escalation checkpoint if any condition occurs:

- Causal strict-pair validation falls below the Gate C requirement.
- Zero-condition text or audio NLL regression exceeds 2%.
- Pinned speaker similarity decreases by more than `0.03` in the evaluator's normalized score.
- No-control duplex/interruption success decreases by more than 3 percentage points.
- The newly unfrozen path fails to improve the attributed audio-control gap after two evaluation intervals.

Mimi remains frozen throughout this program. Mimi unfreezing requires a separate codec-quality hypothesis, reconstruction evaluation, and architecture decision; semantic-control underfitting is not sufficient justification.

## 11. Required ablations

The primary result is compared against these separately named runs:

| Run | Difference from primary |
|---|---|
| Frozen receiver | Conditioner trains, temporal/text receiver frozen |
| All-layer LoRA | LoRA spans all 32 temporal layers including Q/K/V/output/MLP, text path frozen |
| Detailed only | Remove the eight-row boundary burst |
| Burst only | Remove the detailed ARC stream after frame 8 |
| Task-vector initialization | Initialize compatible temporal tensors with the Moshika RAG task vector |
| No listwise loss | Keep SFT and state objective only |
| No state head | Keep SFT and listwise objective only |

The primary run must not be replaced by an ablation. Each ablation uses the same data splits, timing schedules, optimizer-update budget, and evaluation gates.

## 12. Source index and provenance

### Official sources

- [MoshiRAG official repository](https://github.com/kyutai-labs/moshi-rag)
- [MoshiRAG paper](https://arxiv.org/abs/2604.12928)
- [Moshi official repository](https://github.com/kyutai-labs/moshi)
- [Moshi paper](https://arxiv.org/abs/2410.00037)
- [PersonaPlex official repository](https://github.com/NVIDIA/personaplex)
- [PersonaPlex paper](https://arxiv.org/abs/2602.06053)
- [NVIDIA PersonaPlex research page](https://research.nvidia.com/labs/adlr/personaplex/)

### Pinned local upstream sources

- PersonaPlex checkout: commit `3428dfd95309a7f3c84fd93259ded0f810d1ff91` at [local PersonaPlex source](/srv/voxrn_cache/personaplex/source/moshi/moshi/models/lm.py#L435)
- MoshiRAG checkout: commit `8c6dfc101b7871baa428424bcdc583b74fb561d9` at [local MoshiRAG configuration](/srv/voxrn_cache/personaplex/source/moshi-rag-official/configs/moshirag.json#L60)
- [ARC encoder and bridge](/srv/voxrn_cache/personaplex/source/moshi-rag-official/moshi/moshi/conditioners/arc_encoder.py#L390)
- [Condition fuser](/srv/voxrn_cache/personaplex/source/moshi-rag-official/moshi/moshi/conditioners/base.py#L368)
- [PyTorch temporal condition consumption](/srv/voxrn_cache/personaplex/source/moshi-rag-official/moshi/moshi/models/lm.py#L729)
- [Rust temporal condition consumption](/srv/voxrn_cache/personaplex/source/moshi-rag-official/rust/moshi-core/src/lm_generate_multistream.rs#L268)
- [Async RAG lifecycle](/srv/voxrn_cache/personaplex/source/moshi-rag-official/moshi/moshi/inference_utils/rag_manager.py#L44)

### Local contracts, findings, and run evidence

- [MoshiRAG release import report](/srv/voxrn_cache/personaplex/imports/moshirag-release-v1.full.json)
- [MoshiRAG-to-PersonaPlex compatibility](/srv/voxrn_cache/personaplex/compatibility/moshirag-personaplex-v1.json)
- [Numeric checkpoint comparison](/srv/voxrn_cache/personaplex/compatibility/moshirag-personaplex-v1.numeric.json)
- [Patched PersonaPlex model contract](/srv/voxrn_cache/personaplex/contracts/personaplex-7b-v1-fdaf4090-control-v4.mimi-bound.json)
- [Local MoshiRAG adoption analysis](/srv/personaplex_workspace/robit-man-personaplex/ground_truth_finetuning/semantic_control_v4/MOSHIRAG_UPSTREAM_ADOPTION.md#L56)
- [Current semantic-control architecture](/srv/personaplex_workspace/robit-man-personaplex/ground_truth_finetuning/semantic_control_v4/ARCHITECTURE.md#L103)
- [Current empirical findings](/srv/personaplex_workspace/robit-man-personaplex/ground_truth_finetuning/semantic_control_v4/EMPIRICAL_FINDINGS.md#L8)
- [Ordered-prefix run contract](/srv/voxrn_cache/personaplex/training/runs/arc4-ordered-prefix-sanitized-v9-gate050-resume25-to50-20260720T093544Z/run_contract.json)
- [Upper-LoRA run contract](/srv/voxrn_cache/personaplex/training/runs/arc4-v10-upper-lora-step25-to50-20260720T100208Z/run_contract.json)
- [Layer-adapted diagnostic](/srv/voxrn_cache/personaplex/training/runs/arc4-causal-layer-adapted-v7-20260720/curve_diagnostic.json#L141)
- [ARC conditioner smoke report](/srv/voxrn_cache/personaplex/eval/moshirag-conditioner-smoke/report.json#L14)

The local label `ground_truth_finetuning` is workspace provenance, not an official public PersonaPlex branch. The external architectural claims in this document are grounded in the official repositories and papers listed above.
