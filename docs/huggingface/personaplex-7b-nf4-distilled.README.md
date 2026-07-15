---
license: other
license_name: nvidia-open-model
base_model: nvidia/personaplex-7b-v1
tags:
- speech
- voice
- research
- distillation
---

# PersonaPlex distilled research checkpoint

## Status and naming

`student_best.pt` is a BF16 PyTorch state dict. It is **not** an NF4 inference
artifact merely because the teacher or an earlier experiment used NF4 weights.
This card therefore does not claim an NF4 runtime footprint or a deployable
semantic-control capability.

## Evidence available

The included training log records five epochs over 3,000 samples with a
teacher-token objective. That result alone does not establish voice quality,
full-duplex latency, ASR, role adherence, grounding, tool correctness, or
semantic control. No production recommendation is made.

## Required before a promoted release

A promoted checkpoint must publish a reproducible base revision, data manifest
and licenses, training configuration, checkpoint hash, supported runtime,
quantization method, and ASR-grounded end-to-end evaluation. It must clearly
separate exact-text strict rendering from expressive guided rendering.

## License

The upstream PersonaPlex model and any derivative checkpoint remain subject to
the NVIDIA Open Model License and applicable audio/voice-data permissions.
