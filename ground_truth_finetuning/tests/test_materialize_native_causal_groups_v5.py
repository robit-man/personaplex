from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import struct
import wave

import pytest
import torch

from ground_truth_finetuning.tools.materialize_native_causal_groups_v5 import (
    GROUPS_FILENAME,
    REJECTIONS_FILENAME,
    TRAINER_ALL_SPLITS_FILENAME,
    TRAINER_CONTRACT_FILENAME,
    TRAINER_DATASET_SCHEMA,
    TRAINER_GROUP_SCHEMA,
    TRAINER_MANIFEST_FILENAME,
    TRAINER_TEST_FILENAME,
    content_hash,
    main,
    sha256_file,
    sha256_text,
)
from ground_truth_finetuning.training.causal_group_pack import (
    CAUSAL_GROUP_ROLES,
    normalize_causal_group,
)
from ground_truth_finetuning.training.contracts import validate_control_frame_mapping
from ground_truth_finetuning.tools.train_native_moshirag_control import (
    NativeDatasetContract,
    NativeTensorLoader,
    hash_file as trainer_hash_file,
    load_group_manifest,
    load_native_group,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def control_frame(role: str, revision: int) -> dict:
    state_hash = "sha256:" + sha256(f"state-{role}".encode()).hexdigest()
    raw = {
        "schemaVersion": 1,
        "frameId": f"frame-{role}",
        "conversationId": "conversation-v5-fixture",
        "targetTurnId": 2,
        "stateRevision": revision,
        "baseStateHash": "sha256:" + "a" * 64,
        "stateHash": state_hash,
        "semanticSources": ["state_reducer", "tool_result"],
        "state": {"evidencePosture": role, "nextGoal": "respond to the current evidence state"},
        "update": {"applyAt": "next_agent_turn_boundary", "expiresAtMs": 2000},
        "turnTaking": {"expectedBargeIn": role == "verified_negative"},
        "plan": {
            "schemaVersion": 1,
            "callId": "conversation-v5-fixture",
            "turnId": 2,
            "revision": revision,
            "contextHash": state_hash,
            "mode": "expressive",
            "intent": "respond according to verified evidence",
            "dialogueAct": "inform",
            "entities": {},
            "constraints": {
                "required_facts": [],
                "forbidden_claims": [],
                "must_ask": [],
                "must_not_request": [],
            },
            "delivery": {
                "language": "en-US",
                "register": "neutral",
                "assertiveness": 0.5,
                "interruptibility": "yield_on_caller_speech",
                "max_duration_ms": 2000,
                "emphasis_targets": [],
            },
            "expiryMs": 2000,
        },
    }
    return json.loads(json.dumps(validate_control_frame_mapping(raw).as_wire_dict()))


def write_duplex(path: Path, role_index: int, *, prefix_mismatch: bool = False) -> None:
    samples: list[int] = []
    for index in range(15_360):
        shared = index < 5_760
        left = index + 10 if shared else 1000 + role_index * 100 + index
        right = index + 20 if shared else 2000 + role_index * 100 + index
        if prefix_mismatch and index == 12:
            left += 7
        samples.extend((left, right))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def actual_events(role: str) -> dict:
    if role == "verified_negative":
        return {
            "bargeIn": {"occurred": True, "cutoffFrame": 5},
            "cancellation": {"generationId": "generation-negative", "cancelled": True, "atFrame": 5},
            "endCall": {"decision": "continue", "decisionFrame": 4, "toolCallFrame": None},
        }
    if role == "superseded":
        return {
            "bargeIn": {"occurred": False, "cutoffFrame": None},
            "cancellation": {"generationId": "generation-superseded", "cancelled": False, "atFrame": None},
            "endCall": {"decision": "end_call", "decisionFrame": 4, "toolCallFrame": 5},
        }
    return {
        "bargeIn": {"occurred": False, "cutoffFrame": None},
        "cancellation": {"generationId": f"generation-{role}", "cancelled": False, "atFrame": None},
        "endCall": {"decision": "continue", "decisionFrame": 4, "toolCallFrame": None},
    }


def build_fixture(
    root: Path, *, prefix_code_mismatch: str | None = None,
    prefix_audio_mismatch: str | None = None, late_control: str | None = None,
    leak_target_hash: str | None = None, caller_mask: str | None = None,
) -> dict[str, Path]:
    precodec_root = root / "precodec"
    native_root = root / "native"
    (precodec_root / "audio").mkdir(parents=True)
    (native_root / "tensors").mkdir(parents=True)
    (native_root / "alignments").mkdir(parents=True)
    common_context = {
        "callerState": "The caller has asked for a current evidence-based status.",
        "pivotTargetOrdinal": 2,
    }
    common_context_hash = content_hash(common_context)
    shared_prefix_id = content_hash({"groupId": "group-v5-fixture", "context": common_context_hash})
    voice_pair = {"id": "voice-caller->voice-agent", "caller": "voice-caller", "agent": "voice-agent"}
    transcripts = {
        "verified_positive": "The evidence is verified and ready.",
        "verified_negative": "The evidence is verified but unavailable.",
        "uncertain": "The evidence remains uncertain for now.",
        "superseded": "The previous evidence has been superseded.",
    }
    plans: list[dict] = []
    precodec_rows: list[dict] = []
    native_rows: list[dict] = []
    labels: list[dict] = []

    for role_index, role in enumerate(CAUSAL_GROUP_ROLES, start=1):
        transcript = transcripts[role]
        transcript_hash = sha256_text(transcript)
        frame = control_frame(role, role_index)
        if leak_target_hash == role:
            frame = deepcopy(frame)
            frame["state"]["opaqueDigest"] = transcript_hash
            frame = json.loads(
                json.dumps(validate_control_frame_mapping(frame).as_wire_dict())
            )
        frame_hash = validate_control_frame_mapping(frame).frame_hash
        plan_payload = {
            "schema": "personaplex.voryn-branch-artifact.v5",
            "groupId": "group-v5-fixture",
            "siblingRole": role,
            "sharedPrefixId": shared_prefix_id,
            "commonContextHash": common_context_hash,
            "commonContext": common_context,
            "premiseId": "premise-v5-fixture",
            "templateId": "template-v5-fixture",
            "lineageIdentifiers": ["topic-v5-fixture", "trajectory-v5-fixture"],
            "controlOperator": {
                "id": "operator-evidence-status",
                "family": "semantic",
                "changedPaths": ["state.evidenceStatus"],
            },
            "voicePair": voice_pair,
            "controlBinding": {
                "frameHash": frame_hash,
                "revision": role_index,
                "availableFrame": 3 if late_control == role else 2,
            },
            "nativePivotFrame": 3,
            "actualEvents": actual_events(role),
            "sourceExportExampleId": f"source-export-{role}",
        }
        plan = {**plan_payload, "planRecordId": content_hash(plan_payload)}
        plans.append(plan)
        example_id = "sha256:" + sha256(
            f"source-export-{role}|{frame_hash}".encode("utf-8")
        ).hexdigest()
        stem = example_id.removeprefix("sha256:")

        codes = torch.tensor(
            [
                [0, 10, 11, 100 + role_index, 110 + role_index, 120 + role_index, 0, 0],
                [20, 21, 22, 200 + role_index, 210 + role_index, 220 + role_index, 0, 0],
                [30, 31, 32, 300 + role_index, 310 + role_index, 320 + role_index, 0, 0],
            ],
            dtype=torch.long,
        )
        if prefix_code_mismatch == role:
            codes[1, 1] += 1
        mask = torch.zeros_like(codes, dtype=torch.bool)
        mask[0, 3:6] = True
        mask[1, 3:6] = True
        if caller_mask == role:
            mask[2, 4] = True
        codes_path = native_root / "tensors" / f"{stem}.pt"
        mask_path = native_root / "tensors" / f"{stem}.mask.pt"
        torch.save({"codes": codes}, codes_path)
        torch.save({"target_mask": mask}, mask_path)
        alignment = {
            "verified": True,
            "codes_sha256": sha256_file(codes_path),
            "target_label_sha256": transcript_hash,
            "target_frames": [3, 6],
            "frame_rate": 12.5,
        }
        alignment_path = native_root / "alignments" / f"{stem}.json"
        alignment_path.write_text(json.dumps(alignment, sort_keys=True), encoding="utf-8")
        wav_path = precodec_root / "audio" / f"{stem}.wav"
        write_duplex(
            wav_path,
            role_index,
            prefix_mismatch=prefix_audio_mismatch == role,
        )
        source = {
            "schema": "personaplex.controlled-native-precodec.v1",
            "example_id": example_id,
            "split": "train",
            "duplex": {
                "path": f"audio/{stem}.wav",
                "sha256": sha256_file(wav_path),
                "sample_rate": 24_000,
                "channels": {"agent": 0, "caller": 1},
            },
            "target": {"start_ms": 240, "end_ms": 480, "rendered_end_ms": 480},
            "control": {"frame": frame, "frame_hash": frame_hash},
            "evidence": None,
            "counterfactual": {
                "groupId": "group-v5-fixture",
                "siblingRole": role,
                "pivotTargetOrdinal": 2,
            },
            "quality": {"accepted": True, "wer": 0.0, "maxAsrWer": 0.12},
            "labels": {"agent_text_sha256": transcript_hash, "alignment_path": f"audio/{stem}.json"},
            "provenance": {
                "plan_record_id": plan["planRecordId"],
                "source_export_example_id": f"source-export-{role}",
                "scenario_key": f"fixture-{role}",
                "voice_pair": {
                    "caller": {"id": "voice-caller", "sha256": "sha256:" + "c" * 64},
                    "agent": {"id": "voice-agent", "sha256": "sha256:" + "d" * 64},
                },
            },
        }
        precodec_rows.append(source)
        native = deepcopy(source)
        control_stream = torch.full((4, 4), float(role_index), dtype=torch.float32)
        control_path = native_root / "tensors" / f"{stem}.control.pt"
        torch.save({"control_stream": control_stream}, control_path)
        native["model_encoding"] = {
            "model_revision": "personaplex-test-revision",
            "codebook_layout": {"text": [0], "agent_audio": [1], "caller_audio": [2]},
            "delay_config_sha256": "sha256:" + "e" * 64,
            "codec": {
                "mimi_weights_sha256": "sha256:" + "f" * 64,
                "tokenizer_sha256": "sha256:" + "1" * 64,
                "frame_rate_hz": 12.5,
            },
            "codes_path": f"tensors/{stem}.pt",
            "codes_sha256": sha256_file(codes_path),
            "target_mask_path": f"tensors/{stem}.mask.pt",
            "target_mask_sha256": sha256_file(mask_path),
            "text_alignment_path": f"alignments/{stem}.json",
            "text_alignment_sha256": sha256_file(alignment_path),
            "prefix_at": 3,
            "native_control": {
                "schema": "personaplex.native-moshirag-control.v1",
                "stream_path": f"tensors/{stem}.control.pt",
                "stream_key": "control_stream",
                "stream_sha256": sha256_file(control_path),
                "control_frame_hash": frame_hash,
                "control_revision": role_index,
                "acknowledged_control_revision": role_index,
                "control_active_frame": 2,
                "retrieval_buffer_frames": 0,
                "probe_frame_index": 2,
                "probe_targets": {"evidence_status": role_index - 1},
                "probe_slot_cardinalities": {"evidence_status": 4},
                "padding_token_id": 0,
            },
        }
        native_rows.append(native)
        labels.append(
            {
                "example_id": example_id,
                "control_frame": frame,
                "control_frame_hash": frame_hash,
                "evidence_frame": None,
                "evidence_frame_hash": None,
                "target_transcript": transcript,
                "target_label_sha256": transcript_hash,
            }
        )

    base_records = list(zip(plans, precodec_rows, native_rows, labels))
    plans = []
    precodec_rows = []
    native_rows = []
    labels = []
    for group_variant in ("alpha", "beta", "gamma"):
        group_id = f"group-v5-{group_variant}"
        context = {
            "callerState": "The caller has asked for a current evidence-based status.",
            "pivotTargetOrdinal": 2,
            "scenarioVariant": group_variant,
        }
        context_hash = content_hash(context)
        prefix_id = content_hash({"groupId": group_id, "context": context_hash})
        caller_voice = f"voice-caller-{group_variant}"
        agent_voice = f"voice-agent-{group_variant}"
        for base_plan, base_precodec, base_native, base_label in base_records:
            role = base_plan["siblingRole"]
            source_export_id = f"source-export-{group_variant}-{role}"
            payload = deepcopy(base_plan)
            payload.pop("planRecordId")
            payload.update(
                {
                    "groupId": group_id,
                    "sharedPrefixId": prefix_id,
                    "commonContextHash": context_hash,
                    "commonContext": context,
                    "premiseId": f"premise-{group_variant}",
                    "templateId": f"template-{group_variant}",
                    "lineageIdentifiers": [
                        f"topic-{group_variant}", f"trajectory-{group_variant}"
                    ],
                    "controlOperator": {
                        "id": f"operator-evidence-status-{group_variant}",
                        "family": "semantic",
                        "changedPaths": ["state.evidenceStatus"],
                    },
                    "voicePair": {
                        "id": f"{caller_voice}->{agent_voice}",
                        "caller": caller_voice,
                        "agent": agent_voice,
                    },
                    "sourceExportExampleId": source_export_id,
                }
            )
            plan = {**payload, "planRecordId": content_hash(payload)}
            example_id = "sha256:" + sha256(
                f"{source_export_id}|{base_label['control_frame_hash']}".encode("utf-8")
            ).hexdigest()
            precodec = deepcopy(base_precodec)
            precodec["example_id"] = example_id
            precodec["counterfactual"]["groupId"] = group_id
            precodec["provenance"].update(
                {
                    "plan_record_id": plan["planRecordId"],
                    "source_export_example_id": source_export_id,
                    "scenario_key": f"{group_variant}-{role}",
                    "voice_pair": {
                        "caller": {
                            "id": caller_voice, "sha256": "sha256:" + "c" * 64
                        },
                        "agent": {
                            "id": agent_voice, "sha256": "sha256:" + "d" * 64
                        },
                    },
                }
            )
            native = deepcopy(base_native)
            native.update(deepcopy(precodec))
            native["model_encoding"] = deepcopy(base_native["model_encoding"])
            label = deepcopy(base_label)
            label["example_id"] = example_id
            plans.append(plan)
            precodec_rows.append(precodec)
            native_rows.append(native)
            labels.append(label)

    plan_path = root / "compiled_v5.jsonl"
    precodec_manifest = precodec_root / "precodec_manifest.jsonl"
    controls_path = precodec_root / "control_labels.jsonl"
    native_manifest = native_root / "encoded_examples.jsonl"
    write_jsonl(plan_path, plans)
    write_jsonl(precodec_manifest, precodec_rows)
    write_jsonl(controls_path, labels)
    write_jsonl(native_manifest, native_rows)
    certificate = {
        "schema_version": 2,
        "kind": "personaplex-corpus-certificate",
        "status": "certified_for_adapter_training",
        "manifest_sha256": sha256_file(native_manifest),
        "artifact_root": str(native_root.resolve()),
        "items": len(native_rows),
        "failed_items": 0,
        "caller_stream_supervision": "forbidden",
        "model_revisions": ["personaplex-test-revision"],
        "codec_artifacts": [
            {
                "mimi_weights_sha256": "sha256:" + "f" * 64,
                "tokenizer_sha256": "sha256:" + "1" * 64,
            }
        ],
    }
    certificate_path = native_root / "certificate.json"
    certificate_path.write_text(json.dumps(certificate, sort_keys=True), encoding="utf-8")
    return {
        "compiled": plan_path,
        "precodec_manifest": precodec_manifest,
        "controls": controls_path,
        "native_manifest": native_manifest,
        "certificate": certificate_path,
        "precodec_root": precodec_root,
        "native_root": native_root,
        "output": root / "output",
    }


def run(paths: dict[str, Path]) -> int:
    return main(
        [
            "--compiled-plan", str(paths["compiled"]),
            "--precodec-manifest", str(paths["precodec_manifest"]),
            "--control-labels", str(paths["controls"]),
            "--native-manifest", str(paths["native_manifest"]),
            "--certificate", str(paths["certificate"]),
            "--precodec-root", str(paths["precodec_root"]),
            "--native-root", str(paths["native_root"]),
            "--output-root", str(paths["output"]),
        ]
    )


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_materializes_one_packable_group_and_preserves_runtime_events(tmp_path: Path) -> None:
    paths = build_fixture(tmp_path)
    assert run(paths) == 0
    groups = rows(paths["output"] / GROUPS_FILENAME)
    assert len(groups) == 3
    group = groups[0]
    normalized = normalize_causal_group(group)
    assert group["schema"] == "personaplex.native-causal-group.v5"
    assert normalized["commonInput"]["nativePivotFrame"] == 3
    assert len(group["siblings"]) == 4
    assert {item["role"] for item in group["siblings"]} == set(CAUSAL_GROUP_ROLES)
    assert sum(1 for _ in (paths["output"] / "artifacts" / "common").glob("*.codes.pt")) == 3
    negative = next(item for item in group["siblings"] if item["role"] == "verified_negative")
    assert negative["target"]["actualEvents"]["bargeIn"]["cutoffFrame"] == 5
    assert negative["target"]["actualEvents"]["cancellation"]["generationId"] == "generation-negative"
    superseded = next(item for item in group["siblings"] if item["role"] == "superseded")
    assert superseded["target"]["actualEvents"]["endCall"] == {
        "decision": "end_call", "decisionFrame": 4, "toolCallFrame": 5
    }
    suffix = torch.load(
        paths["output"] / negative["target"]["nativeSuffixCodes"]["path"],
        map_location="cpu", weights_only=True,
    )["codes"]
    assert tuple(suffix.shape) == (3, 2)
    assert rows(paths["output"] / REJECTIONS_FILENAME) == []

    trainer_manifest = paths["output"] / TRAINER_MANIFEST_FILENAME
    trainer_contract_value = json.loads(
        (paths["output"] / TRAINER_CONTRACT_FILENAME).read_text()
    )
    assert trainer_contract_value["schema"] == TRAINER_DATASET_SCHEMA
    contract = NativeDatasetContract.from_mapping(
        trainer_contract_value,
        manifest_sha256=trainer_hash_file(trainer_manifest),
        model_revision="personaplex-test-revision",
    )
    training_groups = load_group_manifest(
        trainer_manifest, data_root=paths["output"], contract=contract
    )
    assert {item.split for item in training_groups} == {"train", "validation", "test"}
    for item in training_groups:
        loaded = load_native_group(
            item, loader=NativeTensorLoader(paths["output"]), contract=contract
        )
        assert len(loaded.siblings) == 4
    test_rows = rows(paths["output"] / TRAINER_TEST_FILENAME)
    assert len(test_rows) == 1
    assert test_rows[0]["schema"] == TRAINER_GROUP_SCHEMA
    assert test_rows[0]["split"] == "test"
    all_rows = rows(paths["output"] / TRAINER_ALL_SPLITS_FILENAME)
    assert {item["split"] for item in all_rows} == {"train", "validation", "test"}


@pytest.mark.parametrize(
    ("fixture_kwargs", "reason"),
    [
        ({"prefix_code_mismatch": "uncertain"}, "shared_prefix_codes_mismatch"),
        ({"prefix_audio_mismatch": "uncertain"}, "shared_prefix_audio_mismatch"),
        ({"late_control": "uncertain"}, "control_timing_invalid"),
        ({"leak_target_hash": "uncertain"}, "target_leakage"),
        ({"caller_mask": "uncertain"}, "caller_supervision_forbidden"),
    ],
)
def test_rejects_unsafe_group_without_splicing(
    tmp_path: Path, fixture_kwargs: dict, reason: str
) -> None:
    paths = build_fixture(tmp_path, **fixture_kwargs)
    assert run(paths) == 2
    assert rows(paths["output"] / GROUPS_FILENAME) == []
    rejections = rows(paths["output"] / REJECTIONS_FILENAME)
    assert len(rejections) == 3
    assert rejections[0]["status"] == "rerender_required"
    assert {item["reasonCode"] for item in rejections} == {reason}
    assert rejections[0]["repairPolicy"] == "rerender_entire_four_sibling_group_from_one_shared_prefix"
    assert not (paths["output"] / "artifacts").exists()
