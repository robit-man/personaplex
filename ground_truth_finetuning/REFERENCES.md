# Primary-Source Reference Register

This register records the evidence behind design claims. It prevents a cited model capability from becoming an unsupported product claim.

## R-001: PersonaPlex paper and source

- **Primary sources:** [PersonaPlex preprint](https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf), [NVIDIA implementation](https://github.com/NVIDIA/personaplex).
- **Verified claims:** PersonaPlex is a full-duplex speech-to-speech model based on Moshi. It receives user audio, agent text, and agent audio; its hybrid system prompt combines textual role conditioning and an audio voice prompt. The paper describes a text segment with silent agent audio, a voice segment with padded agent text, masked system-prompt loss, `0.02` nonsemantic-audio weighting, and `0.3` padded-text weighting.
- **Applies to this design:** The need to use native delayed streams, preserve hybrid voice conditioning, supervise both agent text and audio, and start Stage 1 with the paper-aligned loss balance.
- **Does not establish:** That a runtime role prompt can be changed at arbitrary turn boundaries, that generated wording is exact, that an external LLM context is consumed causally, or that any local checkpoint is equivalent to NVIDIA's training.

## R-002: Moshi foundation model

- **Primary source:** [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037).
- **Verified claims:** Moshi separately models its own speech and user speech in parallel streams and predicts time-aligned text before audio tokens in its hierarchical generation path. The paper reports 160 ms theoretical and 200 ms practical latency.
- **Applies to this design:** The semantic-prefix adapter must be inserted through the model's causal, delayed temporal path. It supports treating text and audio targets as coupled but separately measurable.
- **Does not establish:** A Twilio latency target, business-semantic correctness, or a safe state mutation API.

## R-003: PersonaPlex operational documentation

- **Primary source:** [NVIDIA PersonaPlex README](https://github.com/NVIDIA/personaplex).
- **Verified claims:** The released project exposes a live Moshi server, supports text prompts and supplied voice prompts, documents offline streaming evaluation, and provides packaged voice embeddings. It describes customer-service and casual-role prompting examples.
- **Applies to this design:** A reproducible frozen baseline can use upstream offline evaluation and documented voice-prompt inputs as controlled research fixtures, subject to their model license.
- **Does not establish:** Production eligibility, customer-data rights, external control acknowledgement, or a trained control adapter.

## R-004: Qwen3-TTS

- **Primary source:** [Qwen3-TTS repository](https://github.com/QwenLM/Qwen3-TTS).
- **Verified claims:** The project documents streaming generation, a Base model capable of short-reference voice cloning and fine-tuning, and separately released CustomVoice/VoiceDesign variants that document instruction-driven style/voice capabilities. It documents tokenization and model-specific loading paths.
- **Applies to this design:** Qwen3-TTS is a candidate evaluated component for consented reference-clip generation, canonical strict rendering, or carefully scoped voice/prosody research.
- **Does not establish:** That Base cloning reliably follows arbitrary style text, that its latent representation can be mixed with PersonaPlex, that it meets Voryn call latency, or that its license covers any particular data source.

## R-005: TADA

- **Primary source:** [TADA repository](https://github.com/HumeAI/tada).
- **Verified claims:** TADA documents a one-to-one text/acoustic alignment, dynamic per-token speech duration/prosody, dual-stream generation, and separate encoder/prompt caching.
- **Applies to this design:** It motivates a research path for prosody teachers or a strict renderer with clear text-to-speech alignment.
- **Does not establish:** PersonaPlex compatibility, full-duplex interruption behavior, a voice-cloning permission grant, or semantic control of the live audio model.

## R-006: Local runtime source inspection

- **Evidence location:** Exact versioned source tree used by the local PersonaPlex image, inspected on 2026-07-15; details summarized in `RESEARCH_LOG.md`.
- **Verified claims:** The implementation's native train path processes codec codes through delay/temporal/depth generation and returns separate agent text/audio logits/masks. Exact stream offsets depend on the loaded configuration.
- **Applies to this design:** Exporter validation, mask construction, runtime model-layout assertions, and the ban on fixed codebook layout.
- **Does not establish:** That the local custom control overlay applies a prefix or produces control acknowledgement. Its direct control test timed out, which is recorded as a blocking failure.

## Citation discipline

- Cite a primary source adjacent to every model-specific claim in reports and model cards.
- Label all local results with exact run, model, dataset, and scenario identifiers.
- Use "observed" for a local measurement, "published" for a source-backed statement, and "hypothesis" for a planned mechanism.
- A result cannot move from hypothesis to observed until its artefacts and evaluator output are versioned.
