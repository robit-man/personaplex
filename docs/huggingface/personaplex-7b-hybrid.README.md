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

This repository contains historical experiment artifacts. It does **not**
include an executable PersonaPlex server fork or weights that have been shown
to make an external LLM's response become PersonaPlex's spoken audio. The
historical helper generates a text overlay after PersonaPlex has spoken; it is
not a speech-to-speech semantic control path.

Do not use this artifact for regulated, transactional, or factual live calls.

## What is required for a real hybrid deployment

A production system needs a caller-ASR plus LLM semantic authority, a
turn-boundary control protocol, and an arbitration layer. Strict requests must
render the canonical LLM text with deterministic TTS. PersonaPlex may be used
as an expressive guided renderer only with ASR semantic checks and a strict-TTS
fallback. Protocol and evaluation requirements are documented at
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
