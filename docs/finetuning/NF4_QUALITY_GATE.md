# Direct-NF4 Release Quality Gate

The direct-NF4 Jetson path is blocked by default because the existing handoff fixture proves that transport can work while model output remains unintelligible. A packed checkpoint is not deployable merely because it loads or emits audio.

`scripts/start_nf4_server.sh` requires a report at `PERSONAPLEX_NF4_QUALITY_REPORT` unless `PERSONAPLEX_NF4_QUALITY_GATE=off` is set explicitly for debugging.

The report schema is `personaplex.nf4-quality-report.v1` and must attest all of the following:

- The BF16 reference path passed the same fixture.
- Mimi encode/decode passes Whisper WER `<= 0.12` for the known input artifact.
- The runtime uses direct packed weights and CUDA only.
- NF4 output has a non-empty Whisper transcript, is independently certified as relevant to the input, and is not repetitive.
- The first 20 streamed frames pass the BF16 parity comparison.

The canonical negative fixture and observed failure are retained in `NF4_AUDIO_QUALITY_HANDOFF_2026-07-16.md`. A report must be generated from a fresh BF16/NF4 A/B run; hand-written reports are not evidence of quality.
