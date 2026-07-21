# Generated-Audio and Live Reliability Evaluation

## 1. Evaluation layers

Evaluation advances through four layers. A later layer cannot repair a failure
in an earlier layer.

| Layer | Question |
| --- | --- |
| Structural | Are contracts, lineage, masks, hashes, and revisions valid? |
| Teacher-forced | Does the model assign higher probability to the target under the correct control? |
| Generated duplex | Does free-running PersonaPlex speak the controlled behavior in native audio? |
| Live-equivalent | Do revision, timing, cancellation, codec, and egress semantics hold end to end? |

## 2. Structural suite

The suite validates:

- Schema versions and forbidden target keys.
- Monotonic revision and context-hash transitions.
- One changed field per causal pair.
- Exact shared-prefix audio/text/timing lineage.
- Group-isolated splits.
- Native stream layout and agent-only masks.
- Model, Mimi, tokenizer, source, adapter, and evaluator hashes.
- No plaintext target joined to encoded control input.

## 3. Teacher-forced causal suite

For every held-out pair, evaluate all four combinations:

```text
target A | control A
target A | control B
target B | control B
target B | control A
```

Report per-direction and whole-pair accuracy, normalized text NLL margins, audio
loss, and changed-field slices. Also evaluate current-vs-stale, current-vs-null,
field ablations, and current-vs-random controls.

A whole pair passes only when both matched controls beat their sibling controls.
This metric is diagnostic and cannot satisfy the release claim by itself.

## 4. Generated duplex suite

For each trial:

1. Reset the native streaming state and selected approved voice prompt.
2. Stream the preregistered caller/agent prefix at native frame cadence.
3. Submit the typed control update and require a matching acknowledgement.
4. Apply the immutable control stream at the declared boundary.
5. Continue caller audio or silence according to the scenario.
6. Capture agent audio/text tokens, timestamps, generation ids, and control-row
   consumption.
7. Stop only on the model terminal action, scenario timeout, or harness failure.
8. Decode native audio, perform a mu-law round trip, and run independent ASR.
9. Submit the ASR transcript and frame to the typed semantic judge.
10. Archive all hashes, metrics, and terminal status.

Counterfactual branches reuse the same prefix bytes and voice prompt. Random
sampling seeds are paired and reported. Multiple seeds test stochastic
reliability rather than one favorable completion.

## 5. Semantic judge output

The judge returns:

```json
{
  "schemaVersion": 1,
  "trialId": "...",
  "intentCorrect": true,
  "requiredFacts": [{"fact": "...", "status": "supported", "span": "..."}],
  "requiredQuestions": [{"question": "...", "status": "realized", "span": "..."}],
  "entityAccuracy": true,
  "forbiddenClaims": [],
  "contradictions": [],
  "staleStateUse": false,
  "naturalContinuation": true,
  "terminalBehavior": "continue",
  "confidence": 0.94,
  "pass": true
}
```

The transport accepts raw JSON only. The judge's own rationale is bounded and
cannot substitute for populated obligation arrays. A second independent judge
adjudicates invalid, low-confidence, and disagreement cases.

## 6. Counterfactual semantic sensitivity

The pair judge sees the shared context and both independently judged outputs,
but not branch names. It verifies:

- Each output is valid under its own frame.
- Each output would be invalid or materially inferior under the sibling frame.
- The semantic difference tracks the declared changed field.
- Unrelated facts and conversational persona remain stable.

Surface wording difference alone is not sensitivity. Two paraphrases that make
the same decision fail the pair.

## 7. Duplex dynamics

Use Full-Duplex-Bench-style axes:

- Pause handling.
- Smooth turn taking.
- Backchannel timing and takeover rate.
- User interruption and post-interruption relevance.

Add production-specific axes:

- False start and caller self-correction.
- Tool latency and meaningful-content delay.
- Control update during silence, caller speech, and agent speech.
- Packet jitter, loss, duplication, and reordering.
- Media clear acknowledgement and stale packet rejection.
- Natural model-selected termination.

Timing comes from transport events and audio activity, never transcript phrase
heuristics.

## 8. Twilio-equivalent path

```text
24 kHz caller WAV
  -> resample to 8 kHz
  -> mu-law encode
  -> 20 ms paced Twilio media frames
  -> Voryn/PersonaPlex bridge
  -> native Mimi/PersonaPlex stream
  -> outbound audio
  -> 8 kHz mu-law media frames and marks
  -> decode to WAV
  -> independent ASR and semantic judge
```

Failure modes include disconnect, reconnect, duplicate start, delayed mark,
lost acknowledgement, control expiry, stale update, barge-in, backend timeout,
strict-render failure, and server restart. Every scenario has a deterministic
transport oracle even though spoken wording remains generative.

## 9. Live call gate

After offline and emulated gates pass, run a bounded live-infrastructure suite
against synthetic callers. It uses the exact deployed checkpoint and revision,
real websocket/Twilio-compatible pacing, and approved synthetic content. The
suite contains no human personal data and does not place unsolicited calls.

The live report binds:

- Deployment image/source digest.
- Model and adapter hashes.
- Control compiler revision.
- Call ids and event-log hashes.
- Input/output audio hashes.
- Per-call semantic and timing verdicts.

Live success must reproduce the offline causal effect. A server that only
acknowledges control metadata but generates unchanged speech fails.

## 10. Statistical report

The final evaluator computes Wilson intervals with code, not manual arithmetic.
It reports overall and per-slice counts, first-attempt and eventual results,
judge agreement, failure taxonomy, and paired bootstrap intervals for baseline
comparisons.

The frozen gate manifest is signed before inference and includes trial ids,
scenario hashes, voice hashes, sampling seeds, expected control obligations,
and thresholds. Results cannot alter membership or thresholds.
