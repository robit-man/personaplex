# Public release gate: semantically controlled PersonaPlex

No PersonaPlex semantic-prefix checkpoint becomes public merely because an
optimizer run exits successfully. Promotion requires the following immutable
release bundle and every gate below to pass against the exact checkpoint.

## Required artifacts

- `run_contract.json`, native model contract, control-v3 transition state, and
  native tensor certificate, with matching hashes.
- Training metrics at step zero and every selected checkpoint, including
  control loss, plan sensitivity, text-context sensitivity, terminal-control
  coverage, and agent-only loss-mask verification.
- Held-out counterfactual evaluation in which caller audio history is held
  comparable while facts, policy constraints, tool results, posture, goal, or
  terminal authorization differ. The resulting agent realization must change
  in the required semantic direction without leaking target label text into the
  control input.
- Duplex streaming evaluation covering barge-in cancellation, stale control
  revision rejection, new-revision acknowledgement on the next agent turn,
  natural recovery, and model-selected end-call behavior.
- Speech evaluation with codec round-trip checks, Whisper transcription,
  timing/alignment, voice-preservation review, and first-audio latency. Exact
  wording claims are prohibited; workflows requiring exact language must use
  the separately validated strict renderer.
- A license and provenance inventory for every source and cloning reference.
  Synthetic data, private evaluation prompts, phone metadata, credentials, and
  unlicensed references are never public artifacts.

## Promotion rules

1. The checkpoint must improve semantic-control adherence over step zero on a
   held-out set. A lower loss alone is not sufficient.
2. Text-context and terminal controls must be demonstrably effective; zero
   sensitivity, missing terminal examples, stale-revision behavior, or target
   text leakage is a hard release failure.
3. Natural turn-taking, interruption recovery, intelligibility, and latency
   must remain within the release thresholds recorded in the evaluation report.
   Regressions require remediation or a rejected release, not threshold edits.
4. The runtime must load the trained control adapter in the actual PersonaPlex
   forward path: typed frame -> cached control encoder -> learned prefix or
   per-layer K/V conditioning -> next agent speech tokens. A sidecar prompt,
   LLM-written utterance, or metadata-only WebSocket field does not qualify.
5. An independent live-call evaluation must verify revision snapshots and
   barge-in cancellation under Twilio-compatible mu-law streaming. It may guide
   behavior naturally but must not claim deterministic exact speech.

## Public Hugging Face bundle after approval

- Adapter weights, architecture/configuration, compatible base-model revision,
  tokenizer/Mimi requirements, hashes, and license terms.
- A detailed model card: purpose, non-goals, control-frame schema, training
  stages, data composition and exclusions, evaluation results, latency/hardware
  profile, failure modes, safety boundaries, and provenance.
- Reproducible deployment instructions linking the public GitHub fork: CUDA
  requirements, model-cache environment variables, quantized/NF4 runtime
  limitations, Twilio bridge setup, stream codecs, semantic-control protocol,
  GPU admission policy, and rollback instructions.
- Public scripts must fail closed on incompatible source/model hashes and must
  never contain secrets, caller recordings, private prompts, or deployment
  endpoints.

The initial dataset and intermediate checkpoints remain private until this
release bundle is complete and a human reviews the full evaluation report.
