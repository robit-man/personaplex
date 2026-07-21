# Semantic Control v5 Execution Runbook

Status: executable sequence with explicit stop gates
Repository: `/srv/personaplex_workspace/robit-man-personaplex`

## 1. Working variables

Use the runtime service contract for planner endpoints. Do not copy service
ports into this document and do not pass secrets on a command line.

```bash
cd /srv/personaplex_workspace/robit-man-personaplex

REPO="$PWD"
REQUEST="$REPO/ground_truth_finetuning/requests/personaplex_diverse_50x20x10.control-v5.json"
RUN_ROOT="/srv/voxrn_cache/personaplex/training/cascade-v5-pilot-20260720"
VOICE_MANIFEST="/srv/voxrn_cache/personaplex/voices/approved-manifest.json"
VORYN_PLAN="$RUN_ROOT/voryn-plan.v5.jsonl"
EXPORT_ROOT="/srv/voxrn_cache/personaplex/exports/control-v5"
PRECODEC_ROOT="/srv/voxrn_cache/personaplex/precodec/control-v5"
NATIVE_ROOT="/srv/voxrn_cache/personaplex/tensors/control-v5"
GROUP_ROOT="/srv/voxrn_cache/personaplex/native-groups/control-v5"
PACK_ROOT="/srv/voxrn_cache/personaplex/packs/control-v5"
TRAIN_ROOT="/srv/voxrn_cache/personaplex/training/native-moshirag-control-v5"
SCENARIO_REJECTION="$RUN_ROOT/scenario_stage_rejection.v1.json"
SCENARIO_REJECTION_ID="sha256:c02f53487d795b213ad87078f1ea133f912b149bbdbc7886a7d20af4dc9755c1"

export PERSONAPLEX_CASCADE_PLANNER_MODEL="robit/ornith:35b"
# PERSONAPLEX_CASCADE_PLANNER_ENDPOINT is supplied by the runtime source of truth.
# It may contain comma-separated OpenAI-compatible endpoints.
```

The request binds planning to three lanes, reasoning disabled, Chatterbox Turbo,
Whisper, physical CUDA devices `0,1,2`, dynamic hardware discovery, no CPU model
fallback, and a host used-memory ceiling of 80 percent.

## 2. Authoritative live operational checks

Run these commands immediately before reporting process state, checkpoint count,
or GPU activity. Prefix every observation with the emitted timestamp. Do not
copy a prior PID or count forward as current status.

### 2.1 Generator process identity

```bash
date --iso-8601=seconds

mapfile -t CASCADE_PIDS < <(pgrep -f '[b]uild_diverse_synthesis_cascade.py' || true)
if ((${#CASCADE_PIDS[@]})); then
  ps -p "$(IFS=,; echo "${CASCADE_PIDS[*]}")" \
    -o pid=,ppid=,lstart=,etimes=,stat=,%cpu=,%mem=,args=
else
  printf '%s\n' 'no cascade builder observed at this timestamp'
fi
```

The command line should identify the request, output root, stage, and resume
mode. Planner credentials must come from the environment and must never be
passed as CLI arguments or copied into reports.

### 2.2 Durable checkpoint and canonical-artifact counts

```bash
date --iso-8601=seconds

for stage in topics scenarios trajectories; do
  count="$(find "$RUN_ROOT/.stage_checkpoints/$stage" \
    -maxdepth 1 -type f -name '*.json' -printf '%f\n' 2>/dev/null \
    | LC_ALL=C sort -u | wc -l)"
  printf '%s=%s\n' "$stage" "$count"
done

for artifact in topic_cards.jsonl scenario_contracts.jsonl trajectory_seeds.jsonl; do
  path="$RUN_ROOT/$artifact"
  if [[ -f "$path" ]]; then
    printf '%s canonical_rows=%s sha256=' "$artifact" "$(wc -l < "$path")"
    sha256sum "$path" | awk '{print $1}'
  else
    printf '%s canonical_artifact=absent\n' "$artifact"
  fi
done

for identity in request.json seed_catalog.json run_manifest.json; do
  path="$RUN_ROOT/$identity"
  [[ -f "$path" ]] && sha256sum "$path"
done
```

Expected final planning counts are `50` topics, `1000` scenarios, and `10000`
compact trajectory candidates. Checkpoint counts show durable partial work.
Canonical JSONL cardinality plus the run manifest establishes stage completion.
Historical observations are recorded in [README.md](README.md), but they are not
a substitute for this query.

### 2.3 Authoritative scenario quarantine check

The completed initial stage is structurally present but rejected. Verify the
immutable report identity before using any downstream command:

```bash
date --iso-8601=seconds
test -f "$SCENARIO_REJECTION"
ACTUAL_REPORT_ID="$(jq -er '.reportId' "$SCENARIO_REJECTION")"
printf 'scenario_rejection_report=%s\nexpected_report_id=%s\nactual_report_id=%s\n' \
  "$SCENARIO_REJECTION" "$SCENARIO_REJECTION_ID" "$ACTUAL_REPORT_ID"
test "$ACTUAL_REPORT_ID" = "$SCENARIO_REJECTION_ID"

jq '{reportId, status, counts, rejectionRate, primaryFindings}' \
  "$SCENARIO_REJECTION"
```

This report quarantines all 1,000 initial scenarios from training. Do not select
the 82 non-rejected rows as a post-hoc favorable subset. Do not admit the 55
blind repair candidates.

### 2.4 Sampled GPU activity on physical devices 0, 1, and 2

One instantaneous utilization sample can land between inference calls. Sample a
window and inspect both device activity and the compute-process table.

```bash
for sample in 1 2 3 4 5 6; do
  date --iso-8601=seconds
  nvidia-smi -i 0,1,2 \
    --query-gpu=index,uuid,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
    --format=csv,noheader,nounits
  sleep 10
done

date --iso-8601=seconds
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
  --format=csv,noheader,nounits
```

The Python cascade PID is an orchestrator and may not appear as the process
holding VRAM. Correlate its live command line and checkpoint growth with the
configured planner worker processes and sampled activity on all three admitted
GPUs. Do not infer inactivity from one quiet sample.

### 2.5 Timestamped growth check

Use two observations to establish that a long-running stage is making progress:

```bash
for observation in 1 2; do
  date --iso-8601=seconds
  find "$RUN_ROOT/.stage_checkpoints/scenarios" \
    -maxdepth 1 -type f -name '*.json' -printf '%f\n' 2>/dev/null \
    | LC_ALL=C sort -u | wc -l
  pgrep -af '[b]uild_diverse_synthesis_cascade.py' || true
  (( observation == 1 )) && sleep 60
done
```

An unchanged count over one minute is not automatically failure because a
single authentic inference or retry can exceed that interval. Use process state,
logs, and a longer sampled window before classifying a stall.

## 3. Topic and scenario planning

Every generative command uses authentic model inference with strict JSON Schema
output. `--resume` reuses immutable per-artifact checkpoints.

```bash
python3 ground_truth_finetuning/tools/build_diverse_synthesis_cascade.py \
  --request "$REQUEST" \
  --output-root "$RUN_ROOT" \
  --stage topics \
  --planner-model "$PERSONAPLEX_CASCADE_PLANNER_MODEL" \
  --max-workers 3 \
  --resume

python3 ground_truth_finetuning/tools/build_diverse_synthesis_cascade.py \
  --request "$REQUEST" \
  --output-root "$RUN_ROOT" \
  --stage scenarios \
  --planner-model "$PERSONAPLEX_CASCADE_PLANNER_MODEL" \
  --max-workers 3 \
  --resume
```

Stop if the canonical stage file or checkpoint count disagrees with the request.
Do not delete accepted checkpoints to resolve one malformed artifact.

The initial `$RUN_ROOT` has completed these structural commands and is now an
immutable quarantined evidence root. Do not resume it into a replacement corpus,
overwrite its scenarios, or use it as input to trajectory selection. The
replacement blueprint run must use a new run identity and output root.

## 4. Replacement scenario-generation gate

No accepted production command exists yet for the replacement blueprint path.
Implementation must first provide four explicit stages:

```text
topic_blueprints
slot_expansion
whole_topic_scrutiny
targeted_slot_repair
```

The stages must enforce these boundaries:

| Stage | Required behavior |
| --- | --- |
| `topic_blueprints` | One authentic inference returns exactly 20 compact, contrastive, required-ID slots for one topic. |
| `slot_expansion` | One full scenario is generated per slot with topic, slot ID, and blueprint hash fixed by schema. |
| `whole_topic_scrutiny` | An independent model reviews all 20 expanded rows together and emits typed clustered findings. |
| `targeted_slot_repair` | Only rejected slot IDs are regenerated; the original blueprint and accepted rows remain immutable. |

Certification occurs only after a fresh whole-topic audit accepts the complete
20-row set. Individual schema validity, a successful expansion call, or a
targeted repair is not certification.

## 5. Hard stop before trajectory generation

Do not launch the current `--stage trajectories` implementation for the final
v5 corpus until the replacement scenario corpus is independently certified and
TODO items `PLAN-07` through `PLAN-16` are complete. The current implementation
makes ten full sequential trajectory calls per scenario. A reviewed replacement
also failed because `prefixItems` is unsupported and a roughly 12,000-token
response cannot fit a roughly 4,000-token output contract. The accepted
architecture is:

```text
one scenario
  -> one schema-constrained inference
  -> one object with ten required candidate-ID properties
  -> exactly ten genuinely compact authentic candidate descriptors
  -> atomic checkpoint per descriptor

10,000 compact descriptors
  -> typed balanced selection
  -> 250 primary + 250 reserve
  -> full expansion of only those 500
```

After that implementation lands, the intended stage command remains:

```bash
python3 ground_truth_finetuning/tools/build_diverse_synthesis_cascade.py \
  --request "$REQUEST" \
  --output-root "$RUN_ROOT" \
  --stage trajectories \
  --planner-model "$PERSONAPLEX_CASCADE_PLANNER_MODEL" \
  --max-workers 3 \
  --resume
```

## 6. Selection, expansion, groups, and reserve refill

The corrected implementation must select before full expansion. The existing
selector and typed reserve replacement are the source anchors.

```bash
python3 ground_truth_finetuning/tools/build_diverse_synthesis_cascade.py \
  --request "$REQUEST" \
  --output-root "$RUN_ROOT" \
  --stage selection \
  --max-workers 3 \
  --resume

python3 ground_truth_finetuning/tools/build_diverse_synthesis_cascade.py \
  --request "$REQUEST" \
  --output-root "$RUN_ROOT" \
  --stage pairs \
  --planner-model "$PERSONAPLEX_CASCADE_PLANNER_MODEL" \
  --max-workers 3 \
  --resume
```

For typed rejected groups:

```bash
python3 ground_truth_finetuning/tools/build_diverse_synthesis_cascade.py \
  --request "$REQUEST" \
  --output-root "$RUN_ROOT" \
  --stage refill \
  --rejected-groups "$RUN_ROOT/rejected_groups.jsonl" \
  --resume
```

Required canonical artifacts are:

```text
topic_cards.jsonl
scenario_contracts.jsonl
trajectory_seeds.jsonl
primary_trajectories.jsonl
reserve_trajectories.jsonl
selected_trajectories.jsonl
counterfactual_pair_specs.jsonl
run_manifest.json
```

## 7. Compile immutable schema-7 Voryn plans

The approved voice manifest must be provenance-complete and its hash must match
the request binding before compilation.

```bash
python3 ground_truth_finetuning/tools/compile_diverse_cascade_voryn_plan.py \
  --cascade-root "$RUN_ROOT" \
  --voice-manifest "$VOICE_MANIFEST" \
  --output "$VORYN_PLAN" \
  --planner-model "$PERSONAPLEX_CASCADE_PLANNER_MODEL" \
  --max-workers 3 \
  --resume
```

The compiler also writes
`$RUN_ROOT/voryn-plan.v5.shared-prefixes.jsonl`. Every schema-7 plan must contain
`postRenderBridge`, `renderPlanId`, a complete typed `controlProgram`, and one
final `endCall` authorization.

`tools/materialize_diverse_synthesis_cascade.py` is the cardinality and
pre-generation provenance gate:

```bash
python3 ground_truth_finetuning/tools/materialize_diverse_synthesis_cascade.py \
  --request "$REQUEST" \
  --output-root "$RUN_ROOT" \
  --voice-manifest "$VOICE_MANIFEST" \
  --voryn-plan "$VORYN_PLAN" \
  --planner-model "$PERSONAPLEX_CASCADE_PLANNER_MODEL" \
  --max-workers 3 \
  --resume
```

Do not use `--allow-live-plan-replacement` for a frozen training run.

## 8. Render four-sibling groups in Voryn

Run from the Voryn repository. The group scheduler verifies physical lane
mapping and commits progress only after all four siblings pass.

```bash
cd /home/roko/Documents/Projects/github_repos/vox_docker_eval/voryn

GROUP_COUNT="$(jq -r '.counterfactual.groupId' "$VORYN_PLAN" | sort -u | wc -l)"
MAX_GROUPS_PER_LANE="$(( (GROUP_COUNT + 2) / 3 ))"

for lane in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$lane" \
  VOICEBOX_CUDA_VISIBLE_DEVICES="$lane" \
  SYNTHESIS_LANE_INDEX="$lane" \
  SYNTHESIS_LANE_COUNT=3 \
  SYNTHESIS_MAX_GROUPS="$MAX_GROUPS_PER_LANE" \
  SYNTHESIS_PLAN_PATH="$VORYN_PLAN" \
  SYNTHESIS_PROGRESS_NAMESPACE="control-v5" \
  VOXRN_RESOURCE_ROOT="/srv/voxrn_cache" \
  node scripts/run-personaplex-synthetic-lane.js &
done
wait
```

Do not override `SYNTHESIS_TURNS`. Do not admit a group missing an exact replay
snapshot, all four roles, certified suffixes, or final model-emitted `end_call`.

## 9. Export certified duplex examples and branch artifacts

`CERTIFIED_VORYN_ROOT` must point only to independently promoted Voryn v4
conversation JSONL and timeline sidecars.

```bash
cd "$REPO"
CERTIFIED_VORYN_ROOT="/srv/voxrn_cache/datasets/synthesize"

python3 ground_truth_finetuning/tools/export_controlled_duplex_dataset.py \
  "$CERTIFIED_VORYN_ROOT" \
  --output-dir "$EXPORT_ROOT" \
  --compiled-plan "$VORYN_PLAN" \
  --sample-rate 24000
```

Do not use `--allow-incomplete` with v5 branch-artifact finalization. Required
outputs include `examples.jsonl`, `branch_artifacts.v5.jsonl`, `rejections.jsonl`,
and `manifest.json`.

## 10. Precodec, native encoding, and certification

```bash
python3 ground_truth_finetuning/tools/prepare_controlled_native_adapter_dataset.py \
  --export-root "$EXPORT_ROOT" \
  --output-root "$PRECODEC_ROOT"
```

Encode one deterministic shard on each admitted physical CUDA device. Replace
the model paths only with artifacts bound by the inspected model contract.

```bash
for shard in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$shard" \
  python3 ground_truth_finetuning/tools/encode_controlled_native_adapter_tensors.py \
    --manifest "$PRECODEC_ROOT/precodec_manifest.jsonl" \
    --precodec-root "$PRECODEC_ROOT" \
    --artifact-root "$NATIVE_ROOT" \
    --moshi-source-root "/srv/voxrn_cache/personaplex/source/moshi-contract-c578958da236" \
    --mimi-path "/srv/voxrn_cache/huggingface/nvidia/personaplex-7b-v1/fdaf4090a61cb315c138a1faee287ffd6c716309/tokenizer-e351c8d8-checkpoint125.safetensors" \
    --tokenizer-path "/srv/voxrn_cache/huggingface/nvidia/personaplex-7b-v1/fdaf4090a61cb315c138a1faee287ffd6c716309/tokenizer_spm_32k_3.model" \
    --model-contract "/srv/voxrn_cache/personaplex/contracts/personaplex-7b-v1-fdaf4090-control-v4.mimi-bound.json" \
    --device cuda:0 \
    --shard-index "$shard" \
    --shard-count 3 &
done
wait

python3 ground_truth_finetuning/tools/merge_controlled_native_tensor_shards.py \
  --source-manifest "$PRECODEC_ROOT/precodec_manifest.jsonl" \
  --artifact-root "$NATIVE_ROOT" \
  --shard-count 3

python3 ground_truth_finetuning/tools/certify_controlled_native_corpus.py \
  --manifest "$NATIVE_ROOT/encoded_examples.jsonl" \
  --artifact-root "$NATIVE_ROOT" \
  --precodec-root "$PRECODEC_ROOT" \
  --certificate "$NATIVE_ROOT/certificate.json"
```

No CPU encoding fallback is permitted for the production run.

## 11. Materialize native four-role groups

```bash
python3 ground_truth_finetuning/tools/materialize_native_causal_groups_v5.py \
  --compiled-plan "$EXPORT_ROOT/branch_artifacts.v5.jsonl" \
  --precodec-manifest "$PRECODEC_ROOT/precodec_manifest.jsonl" \
  --control-labels "$PRECODEC_ROOT/control_labels.jsonl" \
  --native-manifest "$NATIVE_ROOT/encoded_examples.jsonl" \
  --certificate "$NATIVE_ROOT/certificate.json" \
  --precodec-root "$PRECODEC_ROOT" \
  --native-root "$NATIVE_ROOT" \
  --output-root "$GROUP_ROOT"
```

The materializer writes one shared prefix per group and four branch suffixes.
Any item in `rerender_rejections_v5.jsonl` remains a source repair task; it must
not be patched by splicing unrelated native sidecars.

## 12. Build leakage-component splits

```bash
python3 ground_truth_finetuning/tools/pack_native_causal_groups.py \
  --input "$GROUP_ROOT/native_causal_groups_v5.jsonl" \
  --output-dir "$PACK_ROOT" \
  --trainer-data-contract "$GROUP_ROOT/native_moshirag_dataset_v2.json" \
  --trainer-group-manifest "$GROUP_ROOT/native_moshirag_groups_v2.jsonl" \
  --model-contract "/srv/voxrn_cache/personaplex/contracts/personaplex-7b-v1-fdaf4090-control-v4.mimi-bound.json" \
  --split-seed personaplex-native-causal-groups-v1 \
  --train-ratio 0.8 \
  --validation-ratio 0.1 \
  --test-ratio 0.1 \
  --minimum-distinct-premises 2 \
  --required-coverage-splits train validation test
```

The immutable pack must report `certified` and must keep every leakage component
inside one split. Packing also fails unless the trainer manifest has the exact
same group/component/split projection and its dataset contract binds the exact
group-manifest and model-contract revision.

## 13. Full-rank native training

Use the trainer-ready shared-prefix manifest emitted by the v5 materializer. The
test-only manifest is not used for checkpoint selection.

```bash
python3 ground_truth_finetuning/tools/train_native_moshirag_control.py \
  --data-contract "$GROUP_ROOT/native_moshirag_dataset_v2.json" \
  --group-manifest "$GROUP_ROOT/native_moshirag_groups_v2.jsonl" \
  --data-root "$GROUP_ROOT" \
  --model-contract "/srv/voxrn_cache/personaplex/contracts/personaplex-7b-v1-fdaf4090-control-v4.mimi-bound.json" \
  --certified-pack-manifest "$PACK_ROOT/manifest.json" \
  --moshi-source-root "/srv/voxrn_cache/personaplex/source/moshi-contract-c578958da236" \
  --moshi-path "/srv/voxrn_cache/huggingface/nvidia/personaplex-7b-v1/fdaf4090a61cb315c138a1faee287ffd6c716309/model.safetensors" \
  --run-dir "$TRAIN_ROOT" \
  --max-steps 150 \
  --workers 3 \
  --allowed-physical-gpus 0,1,2 \
  --host-memory-limit 0.8 \
  --gate-min-group-pass-rate 0.95 \
  --gate-min-probe-accuracy 0.95
```

The certified pack manifest is mandatory. Before host/GPU admission and before
launching `torchrun`, the trainer verifies the manifest self-hash, every source
and pack-output hash, the certified coverage document, leakage-component
membership, the exact group/component/split projection, and the bound dataset,
group-manifest, and model-contract hashes. A mismatch exits with status `2` and
no training worker is launched. The proof is repeated by workers and retained
in run/checkpoint metadata so resume cannot switch packs.

The trainer writes checkpoint directories at steps 100, 125, and 150, plus
separate train and validation metrics. A status of
`teacher_forced_gate_passed` leaves `generatedDuplexGate` and `liveCallGate` at
`pending`.

## 14. Generated and live gates

No authoritative command exists yet for the complete v5 generated-native and
live-Twilio promotion suites. Do not improvise a success claim from older v4
evaluators. TODO items `EVAL-01` through `LIVE-12` define the required work.

Promotion is complete only when the generated and live harnesses bind the exact
checkpoint hash, dataset manifest, split certificate, runtime revision protocol,
audio artifacts, adjudications, and confidence intervals.
