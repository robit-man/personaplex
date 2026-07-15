# Architecture Decisions

## ADR-001: Maintain three separate planes

**Decision:** Use audio, semantic, and control planes rather than forcing a deeper text LLM into the per-frame speech loop.

**Reason:** The audio model needs low-latency causal generation; the semantic model needs a broader, mutable dialogue state. A versioned boundary protocol makes their interaction inspectable.

**Consequence:** Exact wording and expressive speech are separate modes. More components must be observed, but their failure modes are explicit.

## ADR-002: Start with a frozen-base semantic-prefix adapter

**Decision:** Train a small prefix adapter before unfreezing native PersonaPlex weights.

**Reason:** It provides a causal model-side path for typed plans while reducing catastrophic forgetting and permitting clear ablations.

**Consequence:** Early capability is bounded by the frozen base. If it fails the plan-sensitivity gates, limited adaptation is evaluated only with a documented report.

## ADR-003: Keep canonical wording out of control-training inputs

**Decision:** Do not include canonical response text in the serialized plan seen by the adapter.

**Reason:** Including it turns the problem into response copying and invalidates evidence that the model understands plan fields.

**Consequence:** Annotation is more demanding: plan and response labels must be separate and counterfactual labels require valid rewritten targets.

## ADR-004: Strict renderer owns exact text

**Decision:** Route exact wording, numbers, commitments, and other high-risk responses through deterministic TTS with ASR validation.

**Reason:** Native speech-to-speech models may paraphrase even when behaviorally guided. The current PersonaPlex experiments do not demonstrate exact semantic control.

**Consequence:** Strict responses can sound less naturally conversational, but truthfulness and auditability prevail. PersonaPlex remains the expressive path.

## ADR-005: Runtime configuration determines stream layout

**Decision:** Exporters and trainers inspect model configuration and assert codebook layout at runtime.

**Reason:** Moshi/PersonaPlex configurations can use different audio-codebook counts. A fixed index map risks silently supervising the wrong stream.

**Consequence:** Dataset exports are tied to model revisions and require configuration hashes.

## ADR-006: No generic quiet-chunk reset

**Decision:** Do not reset all generation state merely because a short quiet chunk is observed.

**Reason:** Silence is ambiguous in full-duplex conversation and a reset destroys causal state, creating artificial apparent control.

**Consequence:** The server applies a revision only at an explicit/validated boundary and logs the decision.

## ADR-007: Acknowledgement is the control truth source

**Decision:** A received websocket update is not considered applied until the server emits a terminal acknowledgement with revision and context hash.

**Reason:** The present direct test timed out waiting for acknowledgement. Without it, transport success cannot prove model conditioning.

**Consequence:** Missing acknowledgement fails closed into strict/fallback mode and blocks promotion.

## ADR-008: Historical hybrid and distillation artifacts are non-authoritative

**Decision:** Preserve history for audit but do not build production architecture or model claims on it.

**Reason:** The historical overlay was post-hoc, the adjacent server was not a native runtime, SFT loss did not demonstrate convergence, and the claimed NF4 artifact was BF16.

**Consequence:** New model cards state these limitations and new training artifacts must meet the ground-truth program requirements.

## ADR-009: Voice identity is a governed input

**Decision:** Use only consented, scoped voice prompts and maintain a revocation path.

**Reason:** Voice prompts are powerful identity conditioning data and may be personally identifying.

**Consequence:** Public availability is not sufficient authorization; all derived prompts and codecs have lineage.

## ADR-010: Treat Qwen3-TTS and TADA as bounded research dependencies

**Decision:** Evaluate Qwen3-TTS as a consented reference/strict-render candidate and TADA as a prosody/strict-render candidate, without assuming latent compatibility or unverified style guarantees.

**Reason:** Their public designs address useful adjacent capabilities but are not drop-in native PersonaPlex control mechanisms.

**Consequence:** Each integration needs its own model card, license review, benchmark, and fallback.
