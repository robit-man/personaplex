# Evaluation protocol

## End-to-end test path

Run every scenario through the production media representation:

`canonical TTS caller audio -> mu-law Twilio frames -> bridge -> PersonaPlex
or strict renderer -> mu-law output -> ASR -> scorer`

Retain only synthetic or consented material. Capture packet timestamps, first
audio time, output duration, ASR transcript, active control revision, tool
decision, and fallback reason.

## Required scenario matrix

- greetings, factual lookup, appointment scheduling, tool confirmation;
- required entity and forbidden claim tests;
- stale control update, context mismatch, and cancelled speculative update;
- caller interruption at early/middle/late agent speech;
- upstream close, dropped media, delayed control acknowledgement, and ASR
  partial/final disagreement;
- no-guidance, expressive guidance, and strict canonical response controls.

## Metrics and gates

Report P50/P95 first-audio latency, interruption-stop latency, ASR word error
rate on synthetic controls, required-entity recall, forbidden-claim rate,
factual agreement, tool-confirmation accuracy, fallback rate, and stale
revision rate. Include scenario-level failures, not only aggregates.

Strict mode requires exact or normalized-text agreement with `target_text`.
Expressive mode is evaluated for semantic agreement, never exact wording. A
model that emits unrelated ASR text, even if it returns audio, fails the gate.
