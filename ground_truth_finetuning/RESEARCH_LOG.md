# Research Log and Evidence Register

This log separates published facts, local observations, and hypotheses. Entries must be updated when an experiment changes their status.

## 2026-07-15: Official model facts

### PersonaPlex

- **Source:** NVIDIA [PersonaPlex repository](https://github.com/NVIDIA/personaplex) and [PersonaPlex preprint](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf).
- **Published design:** PersonaPlex uses concurrent user audio, agent text, and agent audio streams. A system prompt combines role text and an audio voice prompt.
- **Training implication:** Audio and text are joint outputs in a native speech model. A text response generated after the model has spoken is not part of its inference path.
- **Loss implication:** The preprint describes masked system-prompt loss and relatively small nonsemantic-audio weighting (reported as 0.02) with a padded-text weighting. Stage 1 follows this as a starting point and uses an explicit ablation rather than historical local heuristics.
- **Published evaluation scope:** The preprint reports 350 customer-service evaluation questions in Service-Duplex-Bench in addition to 400 Full-Duplex-Bench questions. The local suite therefore needs broad held-out role and conversation-dynamics coverage rather than a few hand-selected calls.
- **Limit:** Role prompts guide behavior; they are not an exact-wording guarantee.

### Moshi

- **Source:** [Moshi paper](https://arxiv.org/abs/2410.00037).
- **Published design:** A full-duplex spoken dialogue model jointly models semantic and acoustic streams and targets low practical latency.
- **Published latency:** Moshi reports 160 ms theoretical and 200 ms practical latency. This is a research-system reference, not a production Twilio latency promise; bridge, ASR, planning, prefix prefill, codec conversion, and network measurements are all reported independently.
- **Implementation implication:** Code delays, stream assignment, and temporal causality are model semantics, not preprocessing details.

### Qwen3-TTS

- **Source:** [official Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS).
- **Published capability:** The project describes Base voice cloning from a short reference and fine-tuning, plus separate CustomVoice/VoiceDesign modes with style/instruction-related capabilities and streaming support.
- **Design use:** Candidate consented reference/teacher generator and candidate strict renderer research path.
- **Limit:** Base cloning is not treated as proof of arbitrary natural-language prosody control. Identity conditioning and delivery control are evaluated separately.

### TADA

- **Source:** [official TADA repository](https://github.com/HumeAI/tada).
- **Published capability:** Text and acoustic representation are aligned 1:1 with dynamic duration/prosody support.
- **Design use:** Candidate prosody teacher or deterministic rendering research path.
- **Limit:** Its representation is not PersonaPlex's representation. No cross-model latent interchange is assumed.

## 2026-07-15: Local source inspection

### Native training interface

- **Observed source:** PersonaPlex/Moshi `LMModel.forward_train` operates over codes, invokes code delay/undelay behavior, and returns agent audio and text logits/masks.
- **Observed source:** Native generation prepares a text stream plus separate agent and user audio stream groups. In a typical PersonaPlex configuration, agent audio begins after the text codebook and user audio follows agent audio.
- **Conclusion:** Training must query the active model/configuration for exact layout. A hardcoded `17` codebook layout is invalid as a general exporter assumption.

### Prompt preparation

- **Observed source:** Runtime system-prompt preparation sequences voice prompt, silence, text prompt, and silence.
- **Conclusion:** Voice prompting is a real native conditioning route. Semantic control needs its own causal prefix mechanism rather than repurposing voice prompt state blindly at each turn.

## 2026-07-15: Historical artifact audit

### Historical hybrid path

- **Observed:** The historical `hybrid` agent routed a PersonaPlex-produced textual artifact to an external Ollama/LLM overlay. The adjacent `personaplex-setup/server.py` did not load the actual Moshi/PersonaPlex runtime and called a method absent from the historical `HybridAgent` implementation.
- **Conclusion:** It is not a functional semantic speech-to-speech server and must not be migrated as the new runtime.

### Historical SFT and distillation

- **Observed:** The historical SFT procedure generated teacher sequences from prompts rather than using verified caller/agent native audio pairs and typed plan labels. Its logged loss was flat at `10.375` for twenty epochs.
- **Observed:** The historical distillation artifacts were token-logit experiments on roughly 3,000 items. A published file named `student_best.pt` is stored as BF16, despite an NF4-distilled repository label.
- **Conclusion:** Neither artifact demonstrates controlled native audio behavior, fine-tuning convergence, NF4 quantization, or deployable semantic control. They are excluded from promotion baselines.

## 2026-07-15: Audio-plane observations

- **Observed test:** A Supertonic TTS caller input asking to reschedule an appointment to Thursday afternoon was transcribed exactly by the available host-side ASR test.
- **Observed test:** The PersonaPlex audio bridge produced an approximately 18.2-second response transcribed as "Sure." in one no-guidance run.
- **Observed test:** Earlier guidance experiments produced generic/incomplete responses, including a response transcribed as "Hello. Thank you for calling." and another as "Hello this is."
- **Conclusion:** Audio transport and basic intelligible output were observed. These samples do not demonstrate semantic plan adherence, exact wording, or robust conversation behavior.

## 2026-07-15: Control-protocol observation

- **Observed test:** A direct in-container bridge test sent a matching-context update, explicit boundary, and silence prefill. It timed out while waiting for a `control_ack`.
- **Conclusion:** The current experimental control overlay has not proven that a revision reaches and changes the model state. It must be treated as unacknowledged and non-production.
- **Next investigation:** Build a deterministic isolated protocol harness with complete state-transition tracing, then fix only the measured missing transition. Do not use live Twilio calls to infer server-internal control state.

## Open research questions

1. What exact internal state/prefill hook can add a scoped semantic prefix without corrupting the full-duplex cache?
2. What minimum adapter size and prefix-frame budget produces counterfactual plan sensitivity within live latency limits?
3. How should current cached text/audio state, planner state, and VAD boundary be reconciled after overlapping speech?
4. Which consented TTS tool best provides strict exact wording and approved voice quality under call latency budgets?
5. Can a constrained adapter meet useful entity/question compliance without unfreezing PersonaPlex blocks?

No answer is assumed until the corresponding experiment and report exist.
