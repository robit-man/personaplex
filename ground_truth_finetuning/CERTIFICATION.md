# Corpus Certification

## 1. Certificate meanings

The program uses narrow certificate statuses. They must never be collapsed into a generic "validated" label.

| Status | Meaning | Does not mean |
| --- | --- | --- |
| `failed` | One or more required checks failed. | The data is safe to train after manual inspection. |
| `passed_precodec_only` | Synthetic source, plan separation, audio integrity, Whisper quality, and alignment evidence passed. | PersonaPlex code streams or masks are valid. |
| `certified_for_adapter_training` | Source and tensor artifacts passed all required checks for the named base-model revision. | A checkpoint trained on it is good, deployable, or semantically compliant. |
| `evaluation_promoted` | A named checkpoint passed the evaluation gates. | It can bypass runtime safeguards or strict renderer routing. |

There is no self-certification. The source generator cannot mark its own records trainable, and the trainer cannot mark its own output deployable.

## 2. Gate A: synthetic-source scrutiny

Run Voryn's source audit on a `/synthesize` v2 export. It must validate every rendered turn, not a selected subset.

```bash
node scripts/audit-personaplex-synthesis.js /path/to/synth.jsonl /path/to/precodec-certificate.json
```

The audit requires:

- `ground_truth` tier, rendered audio, and an audio SHA-256 matching the persisted file;
- nonempty Whisper transcript, WER at or below the record threshold, and nonempty Whisper timing segments;
- target control plan produced before generation, without a canonical-response field;
- canonical response stored separately and exactly matching the generated target text;
- a model-produced planner result rather than a generic fallback plan; and
- expressive mode only. Strict requests are renderer fixtures, not PersonaPlex adapter labels.

A passed source audit emits `passed_precodec_only`. It is a provenance and quality result, not a trainability result.

The Voryn corpus-preparation command requires that exact matching certificate and rejects an export without it:

```bash
node scripts/prepare-personaplex-corpus.js \
  /path/to/synth.jsonl \
  /path/to/precodec-certificate.json \
  /path/to/personaplex-ground-truth-corpus
```

## 3. Gate B: native-code scrutiny

Encode only Gate-A-passed examples with the exact pinned PersonaPlex model. The exporter must write the artifact contract in `DATA_AND_GOVERNANCE.md` and preserve raw audio hashes.

The tensor certifier then loads each artifact rather than accepting an exporter claim:

```bash
python -m ground_truth_finetuning.tools.certify_corpus \
  --manifest /path/to/examples.jsonl \
  --artifact-root /path/to/codec-artifacts \
  --source-audio-root /path/to/synthesize-root \
  --certificate /path/to/corpus-certificate.json
```

It verifies:

- plan schema and canonical-response separation;
- source caller/agent audio files and hashes;
- identical `[K, T]` codes and boolean target-mask tensor shapes;
- explicit and complete text/agent/caller codebook partition;
- no caller or unknown-stream target bits, plus nonempty agent text and agent-audio supervision;
- code and mask SHA-256 values;
- a text-alignment document tied to the exact codes hash and base-model revision; and
- a per-item WER threshold that is respected by both caller context and agent output.

The certificate includes the manifest hash, model revisions, count, per-item failures, and tool version. Any content or artifact change invalidates it.

## 4. Gate C: adapter and runtime scrutiny

After Gate B, training produces a separate run report. The checkpoint advances only if it beats shuffled-plan and no-plan ablations on the held-out suite. It then must pass:

- control-boundary acknowledgement and revision/hash ordering;
- ASR-scored semantic plan adherence on held-out audio;
- full-duplex interruption, codec, and Twilio-emulation tests; and
- strict fallback activation for missing acknowledgement or stale control.

No certificate permits an unacknowledged control overlay to emit an allegedly controlled expressive response.

## 5. Review and revocation

Certificates are append-only reports. A source-data revocation, model revision change, serializer/schema change, codec change, target-mask change, or discovered audit defect invalidates dependent certificates. Reissue from the earliest affected gate; do not edit a prior passing report in place.
