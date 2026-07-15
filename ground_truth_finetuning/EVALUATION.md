# Evaluation and Promotion Gates

## 1. Evaluation principles

Every candidate is compared against a frozen baseline on fixed, versioned scenarios. Generated audio is transcribed with an independent ASR path, then scored against structured plan constraints. Human listening is supplementary and never substitutes for semantic, timing, or safety checks.

Metrics are reported by scenario, language, voice-prompt scope, mode, and failure category. Aggregate averages must not hide a serious subset regression.

## 2. Scenario suite

The suite includes licensed or synthetic-with-lineage conversations spanning:

- greetings, backchannels, clarification, confirmation, and closing;
- entities such as dates, times, phone-like numeric strings, amounts, names, and addresses with protected synthetic values;
- unknown availability and explicit forbidden-claim situations;
- multi-turn correction, topic shift, and stale-plan rejection;
- callers who interrupt during first syllable, mid-sentence, and after media buffering;
- ASR uncertainty, silence, overlapping speech, packet delay, reordered frames, duplicate frames, and codec clipping;
- strict wording, expressive wording, safe fallback, transfer, and policy refusal;
- delivery settings including assertiveness, pace, pauses, and emphasis;
- approved voice prompts and neutral fallback voices.

A scenario's source, generated timing distribution, expected plan constraints, and gold strict text are versioned. Test scenarios never feed training until copied into a new dataset/version with a new holdout suite.

## 3. Core semantic metrics

For expressive mode, score the generated-ASR transcript against the active plan:

- **Intent adherence:** predicted dialogue act matches plan.
- **Required-fact recall:** every required fact is represented correctly.
- **Required-question coverage:** required question or equivalent slot request is present.
- **Entity accuracy:** normalized values match the current entity ledger.
- **Forbidden-claim rate:** any prohibited claim is a failure.
- **Contradiction rate:** output conflicts with known state or policy.
- **Stale revision rate:** output reflects a superseded plan.
- **Plan sensitivity:** counterfactual plan changes result in the intended output change.

For strict mode:

- **Normalized exact match:** ASR output matches canonical text after documented normalization.
- **Number and entity exactness:** no substitutions, omissions, or additions.
- **Renderer failure rate:** all unavailable/invalid renderings select safe fallback.

ASR scores are calibrated against held-out human transcripts. Samples below ASR confidence threshold are manually adjudicated and remain separately reported.

## 4. Audio and timing metrics

Measure from transport events, not wall-clock guesses:

- Caller-end to plan-ready latency.
- Plan-ready to control acknowledgement latency.
- Boundary to first-agent-audio P50/P95/P99.
- First-agent-audio to complete response duration.
- Control revision application rate and stale/superseded rate.
- Barge-in detection to media-clear latency.
- Barge-in to stopped-agent-audio latency.
- Packet-loss/reordering recovery rate.
- Codec round-trip intelligibility and clipping rate.

Set initial promotion thresholds only after Stage 0 measures a realistic baseline. A candidate must improve semantic metrics without violating the explicitly adopted latency and interruption regression budgets.

## 5. Voice and prosody metrics

Using only consented evaluation pairs:

- Independent-ASR word error rate and named-entity accuracy.
- Speaker similarity distribution against held-out reference utterances.
- Pitch, energy, duration, pause, and speaking-rate deviation by requested delivery bucket.
- Human A/B preference with disclosed synthetic audio and randomized blinded order.
- Voice identity leakage and unintended caller-voice imitation checks.

No voice metric is a release override for prohibited data use or semantic failure.

## 6. Required ablations

Every Stage 1 report includes:

| Candidate | Why it exists |
| --- | --- |
| Frozen base, no plan | Baseline behavior and latency. |
| Frozen base, role guidance only | Measures current prompt-level behavior. |
| Adapter with valid plan | Proposed control mechanism. |
| Adapter with shuffled plan | Detects memorization or no actual plan sensitivity. |
| Adapter with one plan field ablated | Attributes gains to individual controls. |
| Strict renderer | Exact-wording operational baseline. |

Any result that does not outperform shuffled-plan control is not evidence of semantic conditioning.

## 7. Emulated Twilio end-to-end test

The harness must use the same codec boundaries as production:

```
validated TTS/caller WAV -> 8 kHz mu-law framed media -> bridge -> audio plane
  -> paced outbound mu-law -> decoded WAV -> independent ASR + event scorer
```

It emulates media marks, clears, jitter, delayed packets, dropped packets, duplicate packets, caller barge-in, server restart, bridge disconnect, plan update before/after boundary, acknowledgement loss, and strict-renderer failure. Timing is derived from collected approved call distributions where available and is reported as synthetic when it is not.

For each run, archive an event timeline with audio hashes, not raw sensitive audio by default. A control test is a failure if no terminal acknowledgement arrives; it must not be reported as a successful guided response.

## 8. Live-call canary gates

Live Twilio tests are permitted only with authorized test numbers, synthetic content, approved voices, and explicit monitoring. They require:

- The corpus's tensor-level certificate status is `certified_for_adapter_training` for any checkpoint being exercised.
- All offline and emulated-Twilio required cases pass.
- A rollback to safe fallback is tested.
- Administrative access and observability are confirmed.
- No real customer data is introduced.
- Call recording/retention complies with applicable consent and law.

Canary reports identify the exact deployed image, bridge revision, adapter checkpoint, configuration hash, and test scenario IDs.

## 9. Minimum report template

```text
candidate:
base model revision:
adapter revision:
dataset/split manifest hashes:
scenario-suite revision:
mode:
semantic metrics by scenario:
strict exact-match metrics:
latency percentiles:
interruption metrics:
voice/prosody metrics:
control acknowledgement distribution:
fallback activations:
known failures:
promotion decision and approver:
```

## 10. Current evidence boundary

Existing evidence shows that the audio bridge can produce intelligible PersonaPlex output in a limited test, and that a Supertonic caller-TTS input transcribed accurately under a host ASR test. It does not establish semantic reliability: prior prompt-guided PersonaPlex outputs were generic or incomplete, and the current revision-acknowledgement path timed out in an in-container direct test. The program begins by treating both as baseline observations, not release evidence.
