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

The checkpoint is a base-model research artifact, not a control adapter. A
semantically steerable runtime requires a separately trained prefix adapter
that receives a target-wording-free `ControlTrainingFrame` and conditions the
native delayed duplex transformer before the next agent speech tokens are
generated. Loading this checkpoint behind an external text-interception LLM
does not meet that requirement.

## Evidence available

The included training log records five epochs over 3,000 samples with a
teacher-token objective. That result alone does not establish voice quality,
full-duplex latency, ASR, role adherence, grounding, tool correctness, or
semantic control. No production recommendation is made.

## Required before a promoted release

A promoted checkpoint must publish a reproducible base revision, data manifest
and licenses, training configuration, checkpoint hash, supported runtime,
quantization method, trained-adapter contract, and ASR-grounded end-to-end
evaluation. It must report semantic adherence, stale-control handling,
interruption cancellation, first-audio latency, and voice preservation, and
must clearly separate exact-text strict rendering from expressive guided
rendering.

## License

The upstream PersonaPlex model and any derivative checkpoint remain subject to
the NVIDIA Open Model License and applicable audio/voice-data permissions.
