# Pinned Moshi-Finetune backend

Stage 1 uses the upstream [Moshi-Finetune](https://github.com/kyutai-labs/moshi-finetune)
codebase at commit `2acc879fe7c48f885a18f6cc9548bccb2674d87b`, not an
invented text/audio interleaver. The upstream `Interleaver` converts timestamped
main-speaker words into the native 12.5 Hz Inner-Monologue text stream and the
stereo waveform into parallel Mimi streams.

## Voryn to upstream data contract

`export_moshi_finetune_dataset.py` accepts only a Voryn pre-codec manifest with:

- SHA-verified caller and target audio.
- Whisper word timestamps and an exact normalized match between target Whisper
  transcript and canonical response.
- Measured caller-end to agent-start gap, preserved as silence in both channels.
- An expressive control plan with canonical text stored outside the plan.

It writes 24 kHz WAV files where left is agent output and right is caller input,
with sibling `.json` alignment files containing only `SPEAKER_MAIN` target words.
The control plan remains in `control_labels.jsonl`; it is not leaked into a text
label or runtime prompt.

## Caller-loss overlay

Stock Moshi-Finetune optimizes all depformer audio streams. That is unsuitable
for the Voryn stage because caller audio is context, not a desired output. The
staging tool copies the pinned source into each run directory and patches its
train and evaluation loss masks. The overlay reads `stream_layout.json` and
supervises only agent depformer streams `0..7` (global codebooks `1..8`). It
throws if caller streams `8..15` (global `9..16`) become targets.

No upstream checkout is modified in place. The staged backend records its exact
source revision and patched file list in `PERSONAPLEX_OVERLAY.json`.

## Resource admission and launch

`training.gpu_admission` accepts a GPU only when its utilization is at or below
the configured ceiling and free VRAM minus a retained reserve meets the requested
minimum. The launcher writes this report before staging or executing anything.
Defaults are intentionally conservative: `44 GiB` usable after an `8 GiB`
reserve and `25%` maximum current utilization.

The launcher uses `torchrun`, upstream FSDP, LoRA rank `64`, gradient
checkpointing, `batch_size=1`, and a short `12 s` window by default. It does not
start training unless `--execute` is supplied after an admitted GPU report.

This stage adapts the speech model. It does not establish that arbitrary live
control updates produce an exact desired sentence; that remains the semantic
prefix/control-plane evaluation gate.
