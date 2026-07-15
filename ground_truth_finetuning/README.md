# PersonaPlex Ground-Truth Fine-Tuning Program

## Status and authority

This directory is the normative design for the `robit-man/personaplex` control-plane and fine-tuning work. Implementation, model cards, experiment reports, and deployment runbooks must conform to it. A claim is not promotable merely because a component starts, produces audio, or completes an optimization run.

The target product is a controllable live-call agent with three independently testable properties:

1. **Expressive live conversation.** PersonaPlex remains responsible for low-latency, full-duplex speech-to-speech interaction and voice-prompt-conditioned timbre.
2. **Semantic governance.** An external ASR and deeper LLM maintain a per-call state window and produce a typed, auditable next-turn plan.
3. **Wording guarantees when required.** A deterministic text-to-speech renderer owns exact wording. PersonaPlex cannot be represented as an exact-wording renderer until it passes the semantic-control evaluation gates in this directory.

The current fork does **not** meet that target. Its historical hybrid artifacts are useful only as evidence of earlier experiments, not as deployable model assets. In particular, a post-hoc text overlay, a token-only distillation checkpoint, an HTTP health response, or a single intelligible audio response is not semantic control.

## Non-negotiable invariants

- A model receives only consented, licensed, traceable speech and text data.
- Every conversation has a source manifest, immutable split assignment, and chain of custody for raw and derived artifacts.
- A controller plan never contains the canonical target response as an input feature during training. That would leak the answer and invalidate control metrics.
- Agent-text and agent-audio targets are explicitly masked. Caller audio is context, never a prediction target.
- Text alignment is verified from the actual PersonaPlex tokenizer/code stream. Repeating or heuristically stretching text tokens across audio frames is prohibited.
- The audio codebook layout is inspected from the loaded model/configuration. It is never assumed to be a fixed count.
- The system applies a control revision only at an acknowledged safe boundary. An unacknowledged update is not considered active.
- Strict exact-wording requests use deterministic TTS until native PersonaPlex compliance is independently demonstrated.
- Promotion relies on held-out measurements, failure cases, and an auditable report, not subjective listening alone.

## Scope

This program covers the training and runtime work required to turn PersonaPlex into a meaningfully controllable expressive audio plane. It does not authorize deceptive voice cloning, unconsented speaker simulation, political persuasion, impersonation, or production deployment of an unvalidated model.

## Document index

| Document | Purpose |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Three-plane design, model interface, stream semantics, and boundary protocol. |
| [DATA_AND_GOVERNANCE.md](DATA_AND_GOVERNANCE.md) | Provenance, consent, manifests, dataset splits, and artifact validity. |
| [TRAINING.md](TRAINING.md) | Staged training suite, losses, checkpoints, and stop conditions. |
| [RUNTIME_CONTROL.md](RUNTIME_CONTROL.md) | Versioned control protocol, call-state semantics, strict fallback, and operational safety. |
| [VOICE_AND_PROSODY.md](VOICE_AND_PROSODY.md) | Voice-prompt strategy, prosody workflow, and consented teacher integrations. |
| [EVALUATION.md](EVALUATION.md) | Offline, replay, emulated-Twilio, and live promotion gates. |
| [CERTIFICATION.md](CERTIFICATION.md) | Fail-closed corpus certification gates and certificate semantics. |
| [RESEARCH_LOG.md](RESEARCH_LOG.md) | Dated evidence, references, verified observations, and open investigations. |
| [REFERENCES.md](REFERENCES.md) | Claim-level primary-source register and applicability limits. |
| [DECISIONS.md](DECISIONS.md) | Architecture decisions and explicitly rejected shortcuts. |

## Delivery sequence

1. Build the reproducible corpus exporter and validator. No training occurs before a validation report passes.
2. Establish a frozen baseline using the same call scenarios and metrics as the candidate model.
3. Train a small semantic-prefix adapter against the validated corpus; keep the base PersonaPlex weights frozen.
4. Add the server-side prefix-prefill/control-boundary implementation and prove acknowledgement ordering in isolated replay.
5. Evaluate semantic compliance, latency, interruptions, voice, and fallback behavior against the gates in `EVALUATION.md`.
6. Only after the adapter is successful, consider limited unfreezing, preference optimization, or broader model changes.
7. Publish model cards and weights only with the resulting evidence, limitations, provenance disclosure, and exact commit identifiers.

## Promotion gates

A stage can advance only when all of the following are true:

- The code, config, environment lock, data manifest hashes, and metrics report are versioned.
- The stage's holdout split was not selected, hand-edited, or used to tune the candidate.
- The corpus has a `certified_for_adapter_training` certificate from the tensor-level validator; `passed_precodec_only` is not trainable status.
- Required semantic, latency, interruption, and safety metrics meet their documented threshold.
- The fallback behavior is tested and remains available.
- Any deviation has an approved decision entry in `DECISIONS.md`.

## Terms

- **Canonical response:** The target wording and target speech selected by the semantic controller or labeler.
- **Control plan:** A typed, bounded representation of intent, constraints, entities, policy, dialogue state, and allowed delivery style. It excludes the canonical response text when used as a training input.
- **Revision:** A monotonically increasing per-call update to the control plan, bound to a call and conversation-state hash.
- **Boundary:** A point at which the audio plane may safely consume a revision without retroactively changing audio already committed to the transport.
- **Strict mode:** A deterministic text-to-speech path with exact canonical wording.
- **Expressive mode:** A PersonaPlex audio-plane path. Its output is constrained and evaluated, but exact wording is not assumed.
