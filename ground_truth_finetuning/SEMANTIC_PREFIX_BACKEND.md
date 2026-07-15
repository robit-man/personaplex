# Typed semantic-prefix backend

The semantic-prefix adapter is a distinct research stage from the caller-safe
upstream LoRA stage. It learns a bounded embedding prefix from the typed control
plan and inserts it at the measured onset of the next agent response. The base
PersonaPlex LM remains frozen.

## Exact data path

1. Voryn writes a plan without canonical response text, a canonical label held
   separately, SHA-verified audio, Whisper word timings, and measured turn gap.
2. The stereo exporter writes agent-left/caller-right WAV plus target-only word
   alignments using the pinned upstream Moshi-Finetune format.
3. `inspect_native_model_contract.py` loads the intended base weights and records
   the actual codebook count, delays, padding IDs, and weights hash. It must be
   regenerated for every base revision.
4. `encode_native_adapter_tensors.py` uses that contract, the matching Mimi
   checkpoint, and a pinned port of the upstream main-speaker `Interleaver`
   algorithm to produce `[17, T]` native codes. The port is required because the
   NVIDIA PersonaPlex source intentionally omits the optional Kyutai conditioning
   module imported by the full upstream package; the tool verifies the referenced
   upstream commit before encoding.
   The target mask contains only text stream `0` and agent audio `1..8`; caller
   audio `9..16` is always context.
5. `certify_corpus.py` is the gate before adapter training. It rejects missing
   alignment, unpinned delays, bad hashes, and any caller target bit.

The adapter then receives `canonical_json(plan.as_wire_dict())`, never the
canonical response. Its loss is agent text plus agent audio only. Prefix insertion
uses gradient checkpointing by default because frozen base weights still require
activation gradients to learn the prefix.

`launch_semantic_prefix.py` makes a fresh GPU admission decision before it
invokes distributed `torchrun`. Each DDP rank owns one frozen base-model replica,
so every selected GPU must independently meet the full usable-memory threshold;
the launcher will not spread a single replica across busy devices. The run stores
the admission report, native contract, corpus certificate, metrics, and adapter
checkpoint together.

## Scope and proof obligation

This architecture can condition subsequent agent behavior at a turn boundary. It
cannot prove that arbitrary live plans force exact wording. The deterministic
renderer remains responsible for exact text. A checkpoint can be promoted only
after the control-protocol harness shows plan-sensitive behavior without degraded
ASR, timing, interruption handling, or safety evaluation.
