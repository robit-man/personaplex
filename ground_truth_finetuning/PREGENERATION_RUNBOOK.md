# Canonical 50 x 20 x 10 pre-generation runbook

The first corpus stage is a planning-only candidate lattice. It creates exactly:

- `50` generated topic cards
- `20` generated scenario contracts per topic, `1,000` total
- `10` generated trajectory leaves per scenario, `10,000` total
- `500` quota-balanced selected counterfactual groups
- `2` branches per group, `1,000` Voryn plan entries

No call audio, transcript, target response, or training material is created by pre-generation. The Voryn renderer, Whisper validation, semantic certification, and tensor pipeline remain separate mandatory stages.

## Start a new immutable run

Create a request revision whose `allowedVoicesManifest` equals the canonical JSON hash of the approved voice-manifest file. Do not reuse the placeholder hash in the example request.

```bash
./tools/run_diverse_synthesis_pre_generation.sh \
  --request ground_truth_finetuning/requests/personaplex_diverse_50x20x10.control-v4.json \
  --output-root /srv/personaplex_workspace/ground_truth_runs/cascade-control-v4 \
  --voice-manifest /srv/voxrn_cache/chatterbox-reference-bank/manifest.json \
  --voryn-plan /srv/personaplex_workspace/ground_truth_runs/personaplex-cascade-control-v4.v8.jsonl \
  --max-workers 3
```

The materializer writes `pre_generation_manifest.json` only after the exact fanout, unique lineage IDs, branch cardinality, request hash, voice provenance hash, and Voryn-plan hash all pass.

## Promote deliberately

The active plan is configured by `PERSONAPLEX_SYNTHESIS_PLAN_PATH` in `/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env`. The materializer refuses to overwrite the active plan. After human review of the pre-generation manifest, update that one setting and restart the renderer, lanes, and certifiers. A plan remains planning-only until Voryn independently renders and certifies it.

## Canonical 50x20x10 production inputs

Use [`personaplex_diverse_seed_library.v1.json`](seed_catalogs/personaplex_diverse_seed_library.v1.json) and [`personaplex_diverse_50x20x10.control-v4.json`](requests/personaplex_diverse_50x20x10.control-v4.json), not the illustrative request or the older v1 production request. The v4 request is hash-bound to the 48-reference Chatterbox manifest at `/srv/voxrn_cache/chatterbox-reference-bank/manifest.json` and requires a typed control frame before every agent target, wrong-branch/stale/null negatives, mutable revisions, interruption invalidation, and model-selected termination. Regenerate the request hash when either source material changes; do not hand-edit a hash or substitute an unapproved voice manifest.

## Causal audio construction rule

For each selected two-branch group, render and encode the duplex prefix exactly
once. Both branches must reference that immutable native prefix and may diverge
only at the declared control pivot. Independently rendering two nominally equal
prefixes is prohibited: it creates timing drift that can leak branch identity.
The pair certifier must compare native tensors through `prefix_at` exactly and
must quarantine a pair rather than weakening the identity check.

The downstream renderer may patch a failed post-pivot turn or suffix in place,
but it must never rerender an already admitted shared prefix. Admission remains
first-attempt based for reliability accounting even when repair artifacts are
retained for training provenance.
