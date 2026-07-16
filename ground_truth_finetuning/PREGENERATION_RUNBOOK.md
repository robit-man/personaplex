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
  --request /srv/personaplex_workspace/ground_truth_runs/cascade-request-r2.json \
  --output-root /srv/personaplex_workspace/ground_truth_runs/cascade-r2 \
  --voice-manifest /srv/voxrn_cache/chatterbox-reference-bank/manifest.json \
  --voryn-plan /srv/personaplex_workspace/ground_truth_runs/personaplex-cascade-r2.v8.jsonl \
  --max-workers 3
```

The materializer writes `pre_generation_manifest.json` only after the exact fanout, unique lineage IDs, branch cardinality, request hash, voice provenance hash, and Voryn-plan hash all pass.

## Promote deliberately

The active plan is configured by `PERSONAPLEX_SYNTHESIS_PLAN_PATH` in `/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env`. The materializer refuses to overwrite the active plan. After human review of the pre-generation manifest, update that one setting and restart the renderer, lanes, and certifiers. A plan remains planning-only until Voryn independently renders and certifies it.
