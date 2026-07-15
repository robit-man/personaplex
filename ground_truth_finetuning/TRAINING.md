# Training Program

## 1. Preconditions

Do not start a training run until the corpus validator in `DATA_AND_GOVERNANCE.md` passes and the baseline report in `EVALUATION.md` is published. The historical 3,000-example token-distillation artifacts and flat-loss SFT logs are not a baseline for this program because they do not prove aligned native audio control.

Every run must record:

- Git commit for code and exact base-model revision.
- Dataset manifest root hash and split manifest hash.
- Complete model configuration discovered at runtime.
- Accelerator, precision, batch construction, optimizer, and random seeds.
- Loss curves, validation metrics, audio samples from fixed IDs, and failure samples.
- Checkpoint lineage and license/consent compatibility.

## 2. Stage 0: frozen baseline

Run the unmodified PersonaPlex base model through the fixed scenario suite. The baseline must capture:

- First-audio latency and total response latency.
- ASR transcript of generated agent audio.
- Semantic-plan adherence, entity recall, forbidden-claim rate, and question coverage.
- Interruption stop latency and recovery behavior.
- Consent-scoped voice/prosody measures.

This establishes both the benefit target and the regression budget. A baseline that only says the model emitted a waveform is insufficient.

## 3. Stage 1: semantic-prefix adapter

### 3.1 Trainable components

Freeze the PersonaPlex LM, audio tokenizer/codec, text tokenizer, and voice-prompt conditioning path. Train only:

- Typed-plan serializer embedding path.
- Small plan encoder.
- Projection to `P` prefix-frame embeddings.
- Optional layer normalization and bounded gating parameters.

Use a small `P` initially and report its latency/memory impact. The adapter should be independently checkpointable and removable at inference.

### 3.2 Batch construction

Each batch preserves native conversation causality:

```
[system role + consented voice prompt] [prior dialogue context] [typed plan prefix] -> next agent turn
```

Inputs include caller audio history, agent audio/text history, and the plan. Labels contain only next-agent text/audio streams. Padding masks, delayed code masks, and target masks must remain distinct.

Do not train one isolated agent utterance as though it had caller history if the runtime needs interruption and multi-turn behavior.

### 3.3 Objective

Use the model's native delayed generation path (`delay`, `forward_codes`, depformer, and `undelay`) and compute loss only where the explicit target mask permits it:

```
L = L_agent_text + lambda_audio * L_agent_audio + lambda_control * L_plan_consistency
```

Initial `lambda_audio` follows the official PersonaPlex training convention of a small nonsemantic-audio weighting (reported as 0.02 in the preprint), then is tuned only by a documented ablation. It must not silently inherit the legacy `0.5 * audio_loss` heuristic.

`L_plan_consistency` is initially an auxiliary classifier or contrastive objective over plan constraints, not a hidden copy of canonical response text. Examples include intent, required-question class, required entity, forbidden-claim class, delivery bucket, and strict/expressive routing.

Every batch must log loss components separately. A single combined scalar hides failures such as text compliance improving while speech collapses.

### 3.4 Negative and counterfactual examples

At least one controlled counterfactual family is required for each important plan field. Hold conversation context and voice prompt constant while varying one of:

- required question;
- required entity;
- forbidden claim;
- dialogue act;
- assertiveness bucket;
- interruption policy; or
- strict/expressive route.

Counterfactual labels require valid rewritten audio/text targets. They cannot reuse one target response after changing its semantics.

## 4. Stage 2: limited adaptation, optional

Only after Stage 1 meets its promotion gate, evaluate unfreezing a tightly bounded set of top LM blocks or low-rank adapters. This stage exists to improve plan responsiveness, not to erase the base model's conversational timing.

Requirements:

- Compare frozen and adapted candidates on the same held-out scenarios.
- Keep a KL or teacher-preservation term against frozen baseline behavior for neutral control plans.
- Measure degradation in full-duplex interruption and voice metrics.
- Roll back if latency, turn-taking, or safety regress beyond `EVALUATION.md` budgets.

Full-model fine-tuning is out of scope unless a separate decision records capacity, corpus scale, data rights, and catastrophic-forgetting mitigations.

## 5. Stage 3: preference and constraint optimization, optional

After supervised semantic control is measurable, collect preference labels only from authorized evaluators. Optimize pairwise choice or reward models for plan compliance, naturalness, interruption response, and prosody. Never optimize a generic preference score without hard semantic and safety constraints; doing so commonly rewards fluent but fabricated answers.

Strict-mode wording remains a renderer property. Do not use preference optimization to claim exact native speech wording.

## 6. Checkpoint acceptance

A checkpoint is promotable to runtime experiment only when:

- It has a complete run card and reproducibility bundle.
- It beats the frozen baseline on predeclared held-out plan-adherence metrics.
- It does not exceed the latency/memory budget.
- It passes leakage checks and no-caller-target checks.
- It passes fixed adversarial and interruption scenarios.
- It has no unresolved provenance or consent issue.

A file named `best`, a lower training loss, BF16 storage, or a low-bit label does not establish any of these properties.

## 7. Fine-tuning suite layout

The implementation to be added under this repository should follow this contract:

```
ground_truth_finetuning/
  schemas/                 # Versioned JSON schemas for plan, manifest, report
  tools/                   # Export, validate, tokenize, split, and report CLIs
  datasets/                # Ignored manifests/pointers; never raw restricted data
  training/                # Adapter model, native loss, trainer, configurations
  evaluation/              # Offline/replay/emulated-call harnesses and scorers
  reports/                 # Versioned aggregate reports, not raw recordings
```

The docs exist before this suite so all code can be reviewed against fixed acceptance criteria. Secrets, raw calls, raw consent records, and unlicensed checkpoints must be ignored and excluded from commits.

## 8. Stop conditions

Immediately stop a run and mark it invalid if:

- A leakage validator detects canonical response text in model control inputs.
- Text alignment or codebook configuration cannot be verified.
- The model loses turn-taking behavior or produces persistent unintended audio.
- A data-rights, consent, or redaction defect is discovered.
- The candidate only improves on training data or on scenarios used to hand-tune prompts.
