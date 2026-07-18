# Controlled native PersonaPlex training pipeline

This is the only permitted path from Voryn synthetic conversations to the
semantic-prefix optimizer. It is deliberately split into independently
verifiable stages.

1. Generate Voryn v3 *candidates* with Chatterbox Turbo, Whisper transcript and
   word-timing evidence, typed control frames, and actual overlap/recovery where
   assigned. Candidate generation keeps expensive model critics off the audio
   critical path; it must never mark a target turn trainable.
2. Run Voryn batch certification. It must independently audit every caller turn
   for authentic non-identifying conversational speech and every target turn for
   causal realization of its materialized control frame. A conversation is
   promoted only when every turn passes. Promoted target records carry
   `semanticAdherence.verificationStatus == "batch_certified"`; promoted caller
   records carry `authenticity.status == "batch_certified"`.
3. Run `export_controlled_duplex_dataset.py` on promoted JSONL only. It creates the 24 kHz duplex
   timeline, crops interrupted agent renders at `audibleEndedAtMs`, and refuses
   missing promotion certificates, recovery, or control-label leakage.
4. Run `prepare_controlled_native_adapter_dataset.py`. It produces a pair-stable
   train/validation/test pre-codec manifest. The output has no target wording:
   it contains only a target-label hash and a cropped target-word alignment sidecar.
5. Inspect the exact compatible base model, Mimi checkpoint, and SentencePiece
   tokenizer with `inspect_native_model_contract.py`. A model contract and
   matching hashes are required. The contract also binds a deterministic hash of
   the Moshi Python source actually used for encoding/training; runtime checks do
   not invoke Git or trust checkout metadata. No approximate or NF4-only
   substitute may enter native prefix training.
6. Run `encode_controlled_native_adapter_tensors.py`. It applies the native
   Mimi/Moshi delayed-code layout and writes a target mask that supervises only
   the current agent text and audible agent codec frames. Caller streams and all
   prior-agent context are always false in the loss mask.
7. Run `certify_controlled_native_corpus.py`; only
   `certified_for_adapter_training` can be passed to `train_semantic_prefix.py`.

When CUDA capacity is shared, step 6 may be partitioned into deterministic
modulo shards. Each shard must use the same immutable inputs and a distinct
`--shard-index`; run `merge_controlled_native_tensor_shards.py` only after all
shards complete. The merger validates a complete, disjoint ID set against the
pre-codec manifest before certification can see the unified manifest.

```bash
python3 ground_truth_finetuning/tools/prepare_controlled_native_adapter_dataset.py \
  --export-root /srv/voxrn_cache/personaplex/exports/controlled-duplex \
  --output-root /srv/voxrn_cache/personaplex/precodec/controlled-v1

python3 ground_truth_finetuning/tools/encode_controlled_native_adapter_tensors.py \
  --manifest /srv/voxrn_cache/personaplex/precodec/controlled-v1/precodec_manifest.jsonl \
  --precodec-root /srv/voxrn_cache/personaplex/precodec/controlled-v1 \
  --artifact-root /srv/voxrn_cache/personaplex/tensors/controlled-v1 \
  --moshi-source-root /srv/voxrn_cache/personaplex/source/moshi \
  --mimi-path /srv/voxrn_cache/models/COMPATIBLE_MIMI.pt \
  --tokenizer-path /srv/voxrn_cache/models/COMPATIBLE_TOKENIZER.model \
  --model-contract /srv/voxrn_cache/personaplex/contracts/COMPATIBLE_BASE.json \
  --device cuda:0

# Optional shared-host form: launch one command per currently admitted CUDA
# device with the same --shard-count, then merge only after every shard exits 0.
python3 ground_truth_finetuning/tools/encode_controlled_native_adapter_tensors.py \
  --manifest /srv/voxrn_cache/personaplex/precodec/controlled-v1/precodec_manifest.jsonl \
  --precodec-root /srv/voxrn_cache/personaplex/precodec/controlled-v1 \
  --artifact-root /srv/voxrn_cache/personaplex/tensors/controlled-v1 \
  --moshi-source-root /srv/voxrn_cache/personaplex/source/moshi \
  --mimi-path /srv/voxrn_cache/models/COMPATIBLE_MIMI.pt \
  --tokenizer-path /srv/voxrn_cache/models/COMPATIBLE_TOKENIZER.model \
  --model-contract /srv/voxrn_cache/personaplex/contracts/COMPATIBLE_BASE.json \
  --device cuda:1 --shard-index 1 --shard-count 3

python3 ground_truth_finetuning/tools/merge_controlled_native_tensor_shards.py \
  --source-manifest /srv/voxrn_cache/personaplex/precodec/controlled-v1/precodec_manifest.jsonl \
  --artifact-root /srv/voxrn_cache/personaplex/tensors/controlled-v1 \
  --shard-count 3

python3 ground_truth_finetuning/tools/certify_controlled_native_corpus.py \
  --manifest /srv/voxrn_cache/personaplex/tensors/controlled-v1/encoded_examples.jsonl \
  --artifact-root /srv/voxrn_cache/personaplex/tensors/controlled-v1 \
  --precodec-root /srv/voxrn_cache/personaplex/precodec/controlled-v1 \
  --certificate /srv/voxrn_cache/personaplex/tensors/controlled-v1/certificate.json
```

The strict renderer is a separate evaluation/runtime path. It must not be
encoded as if expressive PersonaPlex output guaranteed its wording.
