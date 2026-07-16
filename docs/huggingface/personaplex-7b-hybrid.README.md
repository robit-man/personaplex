---
license: other
license_name: nvidia-open-model
base_model: nvidia/personaplex-7b-v1
tags:
- speech
- voice
- full-duplex
- research
- semantic-control
---

# PersonaPlex hybrid research artifacts

## Status: not a deployable semantic-control model

This repository contains historical experiment artifacts plus an experimental
native control overlay. The historical helper generates a text overlay after
PersonaPlex has spoken; it is **not** a speech-to-speech semantic-control path
and must not be used as one.

The fork now contains `personaplex_control.controlled_server`, pinned to
upstream PersonaPlex `3428dfd95309a7f3c84fd93259ded0f810d1ff91`. It accepts a
typed `ControlTrainingFrame`, encodes it through a trainable adapter, and
prefills virtual embeddings into the live Moshi transformer at a caller-turn
boundary. No trained control adapter or CUDA harness report has been released
yet. Therefore this remains a research implementation, not a deployable
semantic-control model.

Do not use this artifact for regulated, transactional, or factual live calls.

## What is required for a real hybrid deployment

A production system needs a caller-ASR plus LLM semantic authority, a
versioned turn-boundary control protocol, and a learned prefix/K-V conditioning
branch in PersonaPlex itself. The semantic authority must emit immutable,
hash-bound frames; the audio server must cache the frame prefix on GPU and
acknowledge it only after transformer prefill; caller barge-in must invalidate
the generation ID and unsent media. Strict requests must render the canonical
LLM text with deterministic TTS. PersonaPlex may be used as an expressive
guided renderer only with ASR semantic checks and a strict-TTS fallback.
Protocol and evaluation requirements are documented at
https://github.com/robit-man/personaplex/tree/main/docs.

## Historical distillation result

The associated 3,000-sample five-epoch log measures teacher-token loss. It is
not a measurement of semantic adherence, audio quality, latency, ASR, or
full-duplex behavior. It must not be interpreted as a hybrid-agent benchmark.

## License and provenance

The upstream PersonaPlex model and derivative artifacts remain subject to the
NVIDIA Open Model License and all source-data/voice permissions. A future
release must include exact source revision, data manifest, checkpoint hash, and
end-to-end evaluation report.
