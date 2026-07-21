# Semantic Control v4 Training Contract

## 1. Optimization target

The model must prefer the speech continuation licensed by the current control
frame over a continuation licensed by a stale or counterfactual sibling frame,
while retaining PersonaPlex voice quality and duplex behavior.

Standard next-token loss alone is insufficient because both branches are fluent
and topic-compatible. The objective must expose matched and mismatched causal
relationships directly.

## 2. Stage 0: immutable baseline

Record before training:

- Base model, Mimi, tokenizer, source, and patch SHA-256 values.
- Control-v3 adapter and metric artifacts.
- Frozen PersonaPlex no-control and role-prompt generated-audio results.
- Current GPU topology, free memory, active compute processes, and host memory.
- Exact train/validation/test group counts and coverage.

No stage may overwrite another run directory. Every attempt has an immutable
run contract and append-only metrics.

## 3. Stage 1: adapter-only causal SFT

Freeze every PersonaPlex parameter, including the native text embedding. Train
only `SemanticControlStreamAdapter`.

For each ordinary target:

```text
L_sft = L_text_agent + 0.02 * L_audio_agent
```

Loss is computed only on the current agent's text and audio streams. Caller
audio and non-target agent turns are context only.

For each causal pair `(A, B)`:

```text
N_AA = normalized text NLL of target A under control A
N_AB = normalized text NLL of target A under control B
N_BB = normalized text NLL of target B under control B
N_BA = normalized text NLL of target B under control A

L_pair = relu(margin + N_AA - N_AB)
       + relu(margin + N_BB - N_BA)
```

The total objective is:

```text
L = L_sft
  + lambda_pair * L_pair
  + lambda_stale * L_stale
  + lambda_null * L_null
  + lambda_gate * L_gate_regularization
```

`L_stale` uses the prior control revision from the same conversation as a hard
negative. `L_null` requires control-sensitive targets to prefer the current
frame over a learned dropped-control representation. Gate regularization keeps
the injected norm bounded relative to native temporal embeddings.

Pair ranking uses text-token likelihood only. Audio likelihood remains in
matched SFT for voice and acoustic continuity.

## 4. Control and field dropout

Dropout is structured and recorded:

| Dropout | Default | Purpose |
| --- | ---: | --- |
| Whole control | 0.10 | Preserve sparse/no-control conversational behavior. |
| Noncritical context | 0.15 | Prevent dependence on verbose transcript history. |
| Style fields | 0.10 | Separate meaning from delivery. |
| Evidence record | 0.20 | Match MoshiRAG robustness and teach abstention. |
| Critical obligation | 0.00 | Never erase the supervised causal requirement. |

Failed and expired evidence are real typed values, not evidence dropout. They
train no-invention behavior.

## 5. Batching

The loader has two explicit batch kinds:

- `sft`: independently certified targets bucketed by native length and boundary.
- `causal_pair`: two pivot targets with the same group, turn, and base state,
  distinct branches, distinct current state, and one declared changed field.

Pair members always reside on the same rank and optimization step. Random DDP
sharding cannot separate them. Validation/test groups never enter the training
sampler.

The initial implementation may use microbatch one pair per GPU with gradient
accumulation. Effective batch size and per-kind proportions are run-contract
fields.

## 6. Checkpoint evaluation during Stage 1

Every checkpoint reports:

- Matched agent text/audio loss.
- Sibling-control text loss.
- Pair-direction accuracy and whole-pair accuracy.
- Margin distribution by changed field.
- Current-vs-stale and current-vs-null accuracy.
- Text-context, tool-result, obligation, style, and termination ablations.
- Control stream norm, gate values, and encode latency.
- No-control baseline loss regression.

Checkpoint selection is lexicographic:

1. Zero validation integrity failures.
2. Highest whole-pair causal accuracy.
3. Highest stale/null discrimination.
4. Lowest matched validation loss within the top causal band.
5. Lowest encode latency and no-control regression.

Training loss alone never selects a checkpoint.

## 7. Stage 2: upper-temporal low-rank adaptation

Stage 2 starts only when Stage 1 is stable and generated-audio adherence remains
below contract. Freeze the accepted control adapter as the reference, then add
LoRA to an explicit set of upper temporal transformer projections.

Use length-normalized DPO or anchored preference optimization over the model's
text stream:

```text
context = shared duplex prefix + current control stream
chosen = current branch target
rejected = counterfactual sibling or stale-revision target
reference = frozen Stage-1 policy
```

The depth transformer and Mimi remain frozen in the first Stage-2 experiment.
Audio-token DPO is prohibited until an ablation demonstrates stability and
benefit, because published full-duplex preference work found joint text/audio
probability optimization unstable.

Stage 2 must retain a KL/reference term and periodically replay ordinary
no-control duplex examples.

## 8. Stage 3: generated-outcome alignment

Teacher-forced discrimination can still fail at free generation. Stage 3 uses
generated completions from the current checkpoint and constructs preference
groups from independent rewards:

- Semantic adherence to the active control frame.
- No forbidden or stale claim.
- Correct terminal/tool action.
- Contextual relevance and naturalness.
- Pause handling, turn taking, backchanneling, and interruption response.
- Voice/intelligibility preservation.

Timing and semantic rewards are normalized separately before combination.
Timing-only optimization is prohibited because it can reduce response quality.
The initial method is offline DPO/APO over accepted/rejected generated pairs.
GRPO is optional only after the offline path is reproducible.

## 9. Distributed and resource contract

Training may use physical CUDA devices `0`, `1`, and `2` only. Device discovery
uses live NVML/`nvidia-smi` telemetry and the configured allowlist. It does not
hard-code VRAM capacity, host RAM capacity, or free-memory thresholds.

Admission rules:

- Discover total/free VRAM and active compute utilization per allowed GPU.
- Estimate model, optimizer, activation, and safety reserve from artifact size
  plus a calibrated multiplier stored in the run contract.
- Select only GPUs that currently satisfy the dynamic estimate.
- Refuse CPU model loading or CPU offload.
- Delay new work only when host memory utilization exceeds 80%, then resume
  automatically when it recovers.
- Never allocate GPU 3.
- Sample live GPU activity during startup and training; fail if no allowed GPU
  shows the expected model residency and compute activity.

Use NCCL distributed training. A process records rank, physical UUID, visible
device mapping, model residency, peak VRAM, and throughput.

## 10. Smoke-to-scale progression

1. Contract/schema tests with synthetic tensors.
2. One real causal pair forward/backward on one available A100.
3. Twenty-pair overfit test; require near-perfect pair discrimination.
4. Full existing V8 Stage-1 run on all admitted GPUs.
5. Generated-audio validation on held-out pairs.
6. Failure-driven next synthesis batch.
7. Stage-2 LoRA only if generated adherence plateaus.
8. Generated-outcome alignment.
9. Frozen 1,000-trial gate and live-equivalent run.

Each stage produces a machine-readable decision: `advance`, `iterate_data`,
`iterate_model`, or `reject`. Automatic progression occurs only on `advance`.

## 11. Stop and rollback rules

Reject a checkpoint when any of these occurs:

- Pair accuracy does not beat shuffled/null controls.
- Validation improves while test causal axes collapse.
- Generated facts copy stale or sibling controls.
- No-control or voice behavior regresses beyond budget.
- Interruption cancellation emits invalidated media.
- Adapter norm or gate saturates without semantic gain.
- Host memory or forbidden-device policy is violated.
- Source/model/dataset hashes differ from the run contract.
