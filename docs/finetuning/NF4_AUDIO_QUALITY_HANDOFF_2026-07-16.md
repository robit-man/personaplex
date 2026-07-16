# PersonaPlex NF4 Audio Quality Handoff

Date: 2026-07-16

This note is for the fine-tuning/runtime agent investigating why the Jetson NF4 deployment produces unintelligible audio even though the transport path is now working.

## Artifacts

- Synthetic input prompt WAV: [artifacts/personaplex_tts_input_24k.wav](artifacts/personaplex_tts_input_24k.wav)
- PersonaPlex generated reply WAV: [artifacts/personaplex_reply_tunnel_live.wav](artifacts/personaplex_reply_tunnel_live.wav)

Checksums:

```text
7e6d67264b5a8718c143e394c6f6605e2dc7e4ce74ce084d10cc96e41e4d5537  artifacts/personaplex_tts_input_24k.wav
8aea0d457e65a9a433b69bbec8fbb75a0d0e86c5e56bdaea1738b9d9a2dd079f  artifacts/personaplex_reply_tunnel_live.wav
```

## Test Setup

Runtime:

- Device: Jetson edge hardware
- Server path: `./scripts/deploy_nf4_cloudflared.sh`
- Model: `models/cudabenchmarktest/personaplex-7b-nf4/model-nf4.safetensors`
- Runtime mode: direct packed NF4, `PERSONAPLEX_NF4_DTYPE=fp16`
- Mimi/tokenizer files from `models/cudabenchmarktest/personaplex-7b-nf4`
- Voice prompt: default `OverBarn.pt`
- `jetson_clocks`: ON during final validation
- `nvp model`: MAXN

Transport status:

- The default prompt state is cached at server startup.
- Local WebSocket handshake measured around `0.026s`.
- Cloudflare WebSocket handshake measured around `0.4s` to `0.7s`.
- Cloudflare self-testing from the same Jetson is not a clean external-browser test, but synthetic audio sent over the tunnel did receive generated PersonaPlex audio frames.

Input prompt text used for TTS:

```text
Hello PersonaPlex. Please say hello back and tell me you can hear this synthetic voice.
```

Input audio generation:

- TTS engine: `espeak-ng`
- Resampled to: 24 kHz mono WAV
- Duration: about `6.70s`

## Observed Result

The server generated audio frames and wrote a valid WAV:

```text
file: docs/finetuning/artifacts/personaplex_reply_tunnel_live.wav
duration: 7.76s
format: 16-bit mono PCM
sample rate: 24000 Hz
rms: about -33.6 dBFS
peak: 7733
```

Synthetic tunnel run metrics:

```text
handshake_after: 0.425s
first_audio_at: 0.425s
first_text_at: 1.764s
audio_msgs: 100
audio_bytes: 25335
reply_seconds: 7.76
```

This proves the server is emitting reply audio, but the audio is not a meaningful English reply.

## Whisper Validation

`faster_whisper` was run locally on the generated reply WAV.

`tiny.en` result:

```text
detected_language: en
probability: 1.000
transcript: <empty>
```

`base.en` result:

```text
[0.00 -> 7.00] Come on, come on, come on, come on.
TRANSCRIPT: Come on, come on, come on, come on.
```

The user also confirmed by listening that the output was not legible English.

## Server Text Tokens

The server-side text channel during the same run emitted nonsensical tokens:

```text
wartime Add wartime pull wartime pullet again' againoo photosynthesis Hinduhan botheret lever wartime$ lever Hinduhan reset revival nations Jeanne Hindu Jeanne lever Belt reset Hinduoo Hindu again again nations Hindu wartime rep belt rep4$ wartime Sl again sl reset high again reset high h again again revival Jeanne again h nations belt4' reset again high revival belt resetoo western4 again again again than reset affair revival reckon belt reset 4 high h
```

The bad text channel and bad decoded audio agree: this is not just an audio playback issue.

## Interpretation

The transport layer is no longer the primary blocker:

- `/api/chat` accepts default browser-style connections without required query params.
- Default prompt priming is cached and no longer delays browser Connect by about 55 seconds.
- The server emits audio and text frames after receiving synthetic mic audio.

The remaining failure is model/runtime output quality:

- Generated text is semantically unstable and repetitive.
- Generated audio is present but not intelligible as the requested answer.
- Whisper only detects a repetitive phrase from the generated audio, not the requested response.

This may be a fine-tuning issue, an NF4 runtime numerical issue, or a prompt/state restoration mismatch. It should not be treated as a Cloudflare-only issue.

## Recommended Next Checks

1. Run the exact input artifact through a known-good BF16/reference PersonaPlex path.
   - If BF16 produces a good reply, focus on NF4 runtime/kernel/numerics.
   - If BF16 also produces bad output, focus on checkpoint/fine-tuning data and prompt conditioning.

2. Compare direct NF4 logits/tokens against BF16 for the same cached prompt state and the same first 20 audio frames.
   - Look for divergence immediately after prompt restore.
   - Record text token IDs, acoustic token IDs, and sampled probabilities.

3. Verify the default prompt-state cache restore.
   - The cache copies LM `cache`, `provided`, `condition_streaming_sum`, `pending_streaming_sums`, and `offset`.
   - If additional hidden state exists outside `LMGen._streaming_state`, include it or disable prompt caching for quality A/B.

4. Validate Mimi encode/decode independently.
   - Feed `personaplex_tts_input_24k.wav` through Mimi encode/decode without LM generation.
   - Confirm the reconstructed input remains intelligible.

5. Add an ASR-based regression test.
   - Input: `artifacts/personaplex_tts_input_24k.wav`
   - Expected behavior: non-empty English response related to hearing the synthetic voice.
   - Failure condition: empty Whisper transcript, repetitive transcript, or low semantic overlap.

6. Investigate NF4 runtime numerics.
   - Current NF4 CUDA linear kernel is simple row-wise dequant matmul, not a fused/tensor-core int4 GEMM.
   - CUDA graph mode currently crashes with illegal memory access in the NF4 attention path.
   - Keep `NO_CUDA_GRAPH=1` until graph-safety is fixed.

## Current Code State

Relevant commits already pushed:

```text
8740e66c fix: cache default prompt state for fast connect
f4afbebe fix: keep cloudflared chat sessions alive
0fa554b1 fix: make direct NF4 deploy self-contained
```

The current committed audio artifact is a negative-quality example: it proves generation happens, and it gives the fine-tuning agent an exact bad-output sample to reproduce.
