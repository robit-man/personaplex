# Voryn Controlled-Synthesis Bridge

This directory is the reproducible interface between the PersonaPlex fine-tuning suite
and Voryn's synthetic duplex-call generator. It does not contain audio, voice
references, API keys, Twilio credentials, source call data, model weights, or a corpus.

## What Voryn must provide

The external Voryn checkout supplies the generator and independent source-certifier:

- `scripts/plan-personaplex-1000-corpus.js` creates a topic-, timing-, voice-, and
  trajectory-diverse 1,000-conversation plan.
- `scripts/derive-personaplex-v7-counterfactual-plan.js` creates exactly paired
  available/constrained branches and binds each pair to a causal pivot.
- `scripts/run-personaplex-v7-paired-lane.js` generates a duplex conversation with
  control frames, timing/cancellation evidence, Chatterbox Turbo audio, and Whisper
  evidence.
- `scripts/certify-personaplex-synthetic-dataset.js` independently audits every turn
  and counterfactual pair.
- `scripts/certify-personaplex-v7-paired-queue.js` promotes only fully certified
  pairs and makes failed groups eligible for regeneration.

The generator has strict invariants:

- The target response is a label and must never appear in the pre-turn control frame.
- Typed control updates are revisioned and available before the target turn.
- All semantic repairs and audits use model inference; structural parsing is not a
  semantic admission shortcut.
- Chatterbox Turbo output is ASR-scored and word-aligned before source admission.
- A model emits the private `end_call` action. There is no deterministic goodbye rule.
- A barge-in cancels queued output and produces timing/cutoff evidence for the next
  recovery turn.

## Required environment

Set `VORYN_CHECKOUT` to a checkout containing the scripts above. The Voryn runtime
owns its inference, Voicebox/Chatterbox, Whisper, reference-bank, and artifact-root
configuration. Never put those credentials in this repository.

Only physical CUDA devices `0`, `1`, and `2` are allowed. A logical lane count above
three may multiplex one process per physical GPU, but each process must expose exactly
one allowed device through `CUDA_VISIBLE_DEVICES`.

## Launching a lane

The service environment must provide the V8 plan path, lane identity, resource root,
and independent-certifier configuration. These wrappers only validate the external
runtime contract and dispatch to Voryn:

```bash
export VORYN_CHECKOUT=/path/to/voryn
export CUDA_VISIBLE_DEVICES=0
export SYNTHESIS_LANE_INDEX=0
export SYNTHESIS_LANE_COUNT=6
export SYNTHESIS_PLAN_PATH=/srv/personaplex_workspace/ground_truth_runs/personaplex-1000-plan.v8-counterfactual-diverse-v6.jsonl
./ground_truth_finetuning/voxrn_synthesis/run_v8_lane.sh
./ground_truth_finetuning/voxrn_synthesis/run_v8_certifier.sh
```

Certification output is source evidence only. After Voryn completes a corpus-level
certificate, use `tools/run_controlled_native_pipeline.py` for export, native Mimi
encoding, tensor certification, and resource-admitted adapter training.

## Gate calibration report

`quality_report.py` is an aggregate-only reporting utility. It reads certified JSONL
files and reports WER, confidence, audio coverage, and word-alignment coverage. It
never admits or rejects an example and cannot replace the semantic or tensor
certifiers.

```bash
python3 ground_truth_finetuning/voxrn_synthesis/quality_report.py \
  --input-root /srv/voxrn_cache/personaplex-lanes \
  --output /tmp/personaplex-v8-source-quality.json
```
