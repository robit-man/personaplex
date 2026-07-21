# Semantic-Control Reliability Contract

## 1. Claim being tested

The candidate claim is:

> Given the same recent duplex conversation and a valid current semantic
> control frame, PersonaPlex naturally speaks a response that follows the
> frame's facts, constraints, goal, posture, and next action while preserving
> full-duplex timing and voice behavior.

The claim excludes exact wording. Any obligation requiring verbatim language,
digit-perfect rendering, or a legally fixed disclosure is routed to the strict
renderer and scored under a separate deterministic contract.

## 2. Unit of evaluation

One trial is one generated agent opportunity with all of the following:

- An immutable duplex audio/code prefix.
- One current typed control revision accepted before generation.
- A generated PersonaPlex waveform, not a teacher-forced target.
- Independent ASR with word timing.
- One typed semantic adjudication against the control frame and audible context.
- Transport events linking control revision, generation id, media marks, clear
  events, and terminal action.

A trial passes only when every required dimension passes. Missing output,
missing ASR, judge unavailability, timeout, malformed acknowledgement, or lost
provenance is a failure, not an excluded sample.

## 3. Primary reliability metric

`semantic_control_pass` is the conjunction of:

```text
intent_or_dialogue_act_correct
AND all_required_facts_supported
AND all_required_questions_or_actions_realized
AND all_current_entities_correct
AND no_forbidden_claim
AND no_contradiction
AND no_stale_revision_content
AND natural_contextual_continuation
AND valid_terminal_behavior
```

The release gate requires all of the following:

- Overall generated-audio pass rate at least `0.97`.
- Two-sided 95% Wilson lower confidence bound at least `0.95`.
- Every preregistered material slice has point pass rate at least `0.95`.
- No slice has a Wilson lower bound below `0.90`.
- Counterfactual pair sensitivity is at least `0.95`: both branches must pass
  and produce the required semantic difference.
- Stale or superseded content emission is exactly zero in the gate run.
- Unsupported policy-sensitive claims are exactly zero in the gate run.
- Exact-language cases are routed to strict rendering with probability `1.0`.

The `0.97` point threshold creates enough headroom for the confidence-bound
claim. The final report includes numerator, denominator, Wilson interval, and
all failures. It never reports a rounded percentage without counts.

## 4. Minimum evidence volume

The final frozen gate contains at least 1,000 generated-audio trials and at
least 250 causal counterfactual pairs. No training conversation, topic leaf,
voice-reference pair, or counterfactual group may enter the gate.

Each critical slice contains at least 50 trials. Critical slices are:

| Slice | Required behavior |
| --- | --- |
| Current tool result | State only facts authorized by the newest result. |
| Failed or expired evidence | Abstain, clarify, or offer a valid alternative. |
| Policy changed | Follow the new boundary, not the previously available action. |
| Caller correction | Roll back the obsolete value and use the correction. |
| Resistance | Acknowledge posture and adjust the next action naturally. |
| Clarification | Ask only the missing high-value question. |
| Handoff | Explain and execute the allowed handoff without invention. |
| Natural close | End once, invoke terminal action, and emit no farewell loop. |
| Barge-in | Stop stale media and recover from the newest revision. |
| Sparse state | Remain useful without inventing missing facts. |
| Casual/non-service | Follow control without call-center mode collapse. |
| Safety boundary | Refuse or redirect only when required. |

## 5. Independent semantic adjudication

Semantic pass/fail decisions come from typed model inference. Regexes, token
overlap, string containment, and phrase lists are prohibited as semantic gates.
Deterministic code may validate schemas, hashes, numeric transport fields,
codec structure, and event ordering.

The judge receives:

```text
audible prior transcript
+ current control frame
+ generated ASR transcript
+ tool/policy provenance summaries
```

It does not receive the synthetic target transcript, branch label, training
split, expected prose, or another judge's answer. The judge returns a strict
JSON object with per-obligation decisions, cited transcript spans, confidence,
and one terminal verdict. Invalid JSON is retried on the same model and then a
separate CUDA-resident judge. Persistent unavailability fails the trial.

At least 200 gate trials, including every model-judge disagreement and every
critical failure, receive blinded human adjudication. Model-vs-human agreement
and confidence calibration are reported.

## 6. Timing and duplex budgets

Semantic success cannot hide unusable timing. The candidate must also satisfy:

- Control encode P95 no worse than the adopted GPU budget.
- Boundary-to-first-audio P95 no more than 10% above frozen PersonaPlex on the
  same hardware and codec path.
- Meaningful-content delay P95 within the preregistered call-derived range.
- Barge-in detection to last emitted stale media P95 at most 200 ms.
- No generated packet with an invalidated generation id reaches the egress
  writer.
- No cross-call control, voice, text, or cached-stream leakage.
- No increase greater than 5% in non-response, takeover, or repeated-signoff
  failures relative to baseline.

Threshold values measured from the frozen baseline are written into the gate
manifest before candidate inference starts. They may not be changed after
results are visible.

## 7. Audio authenticity gate

Every scored generated waveform must pass:

- Decodable native output and 8 kHz mu-law round trip.
- Independent ASR success with word timing.
- No clipping, NaN, empty speech, or silent-output failure.
- Voice similarity within the baseline's preregistered tolerance.
- Intelligibility no worse than baseline on the same scenario and voice.

Whisper WER is used to diagnose renderer fidelity where a target transcript
exists. It is not used as the semantic-control judge for expressive output.

## 8. Failure accounting

Every failure belongs to exactly one primary stage and may carry secondary
tags:

```text
control_protocol
control_encoding
generation
semantic_adherence
stale_revision
termination
duplex_timing
audio_integrity
asr
judge_transport
judge_disagreement
strict_routing
```

Retries remain attached to the original trial id. A successful retry does not
erase the initial failure; reliability is reported both first-attempt and
eventual. The 95% release claim uses first-attempt results.

## 9. Promotion prohibition

The model cannot be published as semantically controllable when any of these is
true:

- The gate evaluates teacher-forced loss instead of generated audio.
- Test controls or target labels entered training.
- Pair branches do not share the same pre-update causal context.
- Semantic scoring depends on regex or target-text overlap.
- Failed trials were dropped from the denominator.
- The live runtime did not load the exact evaluated checkpoint and source
  revision.
- The observed lower confidence bound is below 0.95.
