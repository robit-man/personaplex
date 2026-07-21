from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import struct
import wave

import pytest

from ground_truth_finetuning.tools.export_controlled_duplex_dataset import (
    BRANCH_ARTIFACT_SCHEMA,
    CAUSAL_GROUP_ROLES,
    RENDER_BINDING_SCHEMA,
    content_hash,
    finalize_v5_branch_artifact,
    validate_complete_v5_artifacts,
)
from ground_truth_finetuning.tools.prepare_controlled_native_adapter_dataset import prepare
from ground_truth_finetuning.training.contracts import validate_control_frame_mapping


def control_frame(role: str, revision: int) -> dict:
    state_hash = "sha256:" + sha256(f"state-{role}".encode()).hexdigest()
    raw = {
        "schemaVersion": 1,
        "frameId": f"frame-{role}",
        "conversationId": f"conversation-{role}",
        "targetTurnId": 4,
        "stateRevision": revision,
        "baseStateHash": "sha256:" + "a" * 64,
        "stateHash": state_hash,
        "semanticSources": ["state_reducer", "tool_result"],
        "state": {"evidencePosture": role, "nextGoal": "respond to current evidence"},
        "update": {"applyAt": "next_agent_turn_boundary", "expiresAtMs": 2400},
        "turnTaking": {"expectedBargeIn": role == "verified_negative"},
        "plan": {
            "schemaVersion": 1,
            "callId": f"conversation-{role}",
            "turnId": 4,
            "revision": revision,
            "contextHash": state_hash,
            "mode": "expressive",
            "intent": "respond according to verified evidence",
            "dialogueAct": "inform",
            "entities": {},
            "constraints": {
                "required_facts": [], "forbidden_claims": [],
                "must_ask": [], "must_not_request": [],
            },
            "delivery": {
                "language": "en-US", "register": "neutral", "assertiveness": 0.5,
                "interruptibility": "yield_on_caller_speech", "max_duration_ms": 2400,
                "emphasis_targets": [],
            },
            "expiryMs": 2400,
        },
    }
    return json.loads(json.dumps(validate_control_frame_mapping(raw).as_wire_dict()))


def render_plan(role: str, frame: dict, common_context: dict | None = None) -> dict:
    common = common_context or {
        "requestId": "request-v5",
        "scenario": {"scenarioId": "scenario-v5", "premise": "Evidence changes safely."},
        "trajectory": {"trajectoryId": "trajectory-v5"},
        "groupId": "group-v5",
        "interventionFamily": "semantic",
        "typedPivot": {"field": "state.evidence.status", "from": "pending", "to": "changed"},
    }
    bridge = {
        "schema": RENDER_BINDING_SCHEMA,
        "groupId": "group-v5",
        "siblingRole": role,
        "scenarioKey": f"group-v5-{role}",
        "sharedPrefixId": "sha256:" + "b" * 64,
        "commonContextHash": content_hash(common),
        "commonContext": common,
        "premiseId": "scenario-v5",
        "templateId": "sha256:" + "c" * 64,
        "lineageIdentifiers": ["request-v5", "topic-v5", "scenario-v5", "trajectory-v5", "group-v5"],
        "controlOperator": {
            "id": "sha256:" + "d" * 64,
            "family": "semantic",
            "changedPaths": ["state.evidence.status"],
        },
        "voicePair": {
            "id": "sha256:" + "e" * 64,
            "caller": "voice-caller",
            "agent": "voice-agent",
        },
        "pivotTargetOrdinal": 2,
        "pivotControlBinding": {
            "frameHash": validate_control_frame_mapping(frame).frame_hash,
            "revision": frame["stateRevision"],
            "availableBeforeTarget": True,
        },
        "nativeFrameContract": {
            "sampleRateHz": 24_000,
            "frameRateHz": 12.5,
            "frameDurationMs": 80,
            "samplesPerFrame": 1_920,
        },
    }
    return {
        "schemaVersion": 7,
        "scenarioKey": bridge["scenarioKey"],
        "counterfactual": {"groupId": "group-v5", "branchId": role},
        "postRenderBridge": bridge,
        "renderPlanId": content_hash(bridge),
    }


def records_and_timeline(role: str, frame: dict, *, previous_end: int = 400) -> tuple[list[dict], dict]:
    conversation_id = f"conversation-{role}"
    frame_hash = validate_control_frame_mapping(frame).frame_hash
    target_text = "The verified evidence changes what we can safely do next."
    records = [
        {
            "conversationId": conversation_id, "runIndex": 0, "turnIndex": 0,
            "speaker": "caller", "counterfactualGroupId": "group-v5",
            "counterfactualBranchId": role, "scenarioKey": f"group-v5-{role}",
            "voiceReference": {"id": "voice-caller"}, "audioPath": "turn-0.wav",
            "timing": {"startedAtMs": 80, "audibleEndedAtMs": previous_end, "endedAtMs": previous_end,
                       "turnTaking": {"eventType": "completed_turn"}},
        },
        {
            "conversationId": conversation_id, "runIndex": 0, "turnIndex": 1,
            "speaker": "target", "counterfactualGroupId": "group-v5",
            "counterfactualBranchId": role, "counterfactualPivotTargetOrdinal": 2,
            "scenarioKey": f"group-v5-{role}", "voiceReference": {"id": "voice-agent"},
            "audioPath": "turn-1.wav", "text": "A shared prefix response.",
            "timing": {"startedAtMs": 480, "audibleEndedAtMs": 640, "endedAtMs": 640,
                       "turnTaking": {"eventType": "completed_turn", "nextBargeIn": None}},
        },
        {
            "conversationId": conversation_id, "runIndex": 0, "turnIndex": 2,
            "speaker": "caller", "counterfactualGroupId": "group-v5",
            "counterfactualBranchId": role, "scenarioKey": f"group-v5-{role}",
            "voiceReference": {"id": "voice-caller"}, "audioPath": "turn-2.wav",
            "timing": {"startedAtMs": 680, "audibleEndedAtMs": previous_end + 300, "endedAtMs": previous_end + 300,
                       "turnTaking": {"eventType": "completed_turn"}},
        },
        {
            "conversationId": conversation_id, "runIndex": 0, "turnIndex": 3,
            "speaker": "target", "counterfactualGroupId": "group-v5",
            "counterfactualBranchId": role, "counterfactualPivotTargetOrdinal": 2,
            "scenarioKey": f"group-v5-{role}", "voiceReference": {"id": "voice-agent"},
            "audioPath": "turn-3.wav", "text": target_text,
            "timing": {"startedAtMs": 800, "audibleEndedAtMs": 1040, "endedAtMs": 1040,
                       "turnTaking": {"eventType": "completed_turn", "nextBargeIn": None}},
            "control": {"frame": frame, "frameHash": frame_hash, "evidenceHash": None,
                        "targetLabelSha256": "sha256:" + sha256(target_text.encode()).hexdigest()},
            "generation": {"replyAction": "end_call" if role == "superseded" else "speak"},
        },
    ]
    timeline = {
        "schema": "voxrn.duplex-timeline.v1",
        "conversationId": conversation_id,
        "channels": {"agent": "target", "caller": "caller"},
        "events": [
            {"turnIndex": item["turnIndex"], "speaker": item["speaker"],
             "audioPath": item["audioPath"], "timing": deepcopy(item["timing"]),
             "controlFrameHash": (item.get("control") or {}).get("frameHash")}
            for item in records
        ],
    }
    return records, timeline


def finalize(tmp_path: Path, role: str, *, previous_end: int = 400, common_context: dict | None = None):
    frame = control_frame(role, CAUSAL_GROUP_ROLES.index(role) + 1)
    plan = render_plan(role, frame, common_context)
    records, timeline = records_and_timeline(role, frame, previous_end=previous_end)
    timeline_path = tmp_path / f"{role}.timeline.json"
    timeline_path.write_text(json.dumps(timeline, sort_keys=True), encoding="utf-8")
    artifact, target = finalize_v5_branch_artifact(
        plan, records, timeline, timeline_path,
        f"{target_conversation(records)}:target:3",
    )
    return plan, artifact, target


def target_conversation(records: list[dict]) -> str:
    return str(records[0]["conversationId"])


def write_stereo(path: Path) -> None:
    samples = [0] * (24_000 * 2 * 2)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(struct.pack("<" + "h" * len(samples), *samples))


def test_four_roles_finalize_with_content_identity_and_native_timing(tmp_path: Path) -> None:
    plans = {}
    artifacts = []
    for role in CAUSAL_GROUP_ROLES:
        plan, artifact, _target = finalize(tmp_path, role)
        plans[("group-v5", role)] = plan
        artifacts.append(artifact)
        assert artifact["schema"] == BRANCH_ARTIFACT_SCHEMA
        assert artifact["planRecordId"] == content_hash({
            key: value for key, value in artifact.items() if key != "planRecordId"
        })
        assert artifact["nativePivotFrame"] == 10
        assert artifact["controlBinding"]["availableFrame"] == 9
        assert artifact["controlBinding"]["availableFrame"] < artifact["nativePivotFrame"]
    validate_complete_v5_artifacts(plans, artifacts)
    with pytest.raises(ValueError, match="exactly one pivot"):
        validate_complete_v5_artifacts(plans, artifacts[:-1])


def test_rejects_late_control_and_target_leakage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strictly before"):
        finalize(tmp_path, "verified_positive", previous_end=800)
    leaked = {
        "requestId": "request-v5",
        "scenario": {"instruction": "The verified evidence changes what we can safely do next."},
        "trajectory": {"trajectoryId": "trajectory-v5"},
        "groupId": "group-v5",
        "interventionFamily": "semantic",
        "typedPivot": {"field": "state.evidence.status", "from": "pending", "to": "changed"},
    }
    with pytest.raises(ValueError, match="target text leaked"):
        finalize(tmp_path, "verified_positive", common_context=leaked)


def test_plan_record_id_propagates_while_transcript_stays_outside_control(tmp_path: Path) -> None:
    _plan, artifact, target = finalize(tmp_path, "verified_positive")
    export_root = tmp_path / "export"
    output_root = tmp_path / "precodec"
    (export_root / "audio").mkdir(parents=True)
    (output_root / "audio").mkdir(parents=True)
    audio = export_root / "audio" / "duplex.wav"
    write_stereo(audio)
    label = target["text"]
    frame = target["control"]["frame"]
    frame_hash = target["control"]["frameHash"]
    example = {
        "schema": "personaplex.controlled-duplex.example.v1",
        "exampleId": artifact["sourceExportExampleId"],
        "duplexAudio": {
            "path": "audio/duplex.wav",
            "sha256": "sha256:" + sha256(audio.read_bytes()).hexdigest(),
        },
        "target": {"audibleStartMs": 800, "audibleEndMs": 1040, "renderedEndMs": 1040},
        "controlFrame": frame,
        "controlFrameHash": frame_hash,
        "evidenceFrame": None,
        "evidenceFrameHash": None,
        "counterfactual": {
            "groupId": "group-v5", "branchId": "verified_positive",
            "siblingRole": "verified_positive", "pivotTargetOrdinal": 2,
        },
        "labels": {
            "agentText": label,
            "asr": {"segments": [{"words": [
                {"word": "The", "start": 0.0, "end": 0.08},
                {"word": "evidence", "start": 0.09, "end": 0.2},
            ]}]},
        },
        "quality": {"accepted": True},
        "provenance": {
            "voicePair": {"caller": {"id": "voice-caller"}, "agent": {"id": "voice-agent"}},
            "scenarioKey": "group-v5-verified_positive",
            "planRecordId": artifact["planRecordId"],
            "renderPlanId": artifact["renderEvidence"]["renderPlanId"],
            "branchArtifactSchema": BRANCH_ARTIFACT_SCHEMA,
        },
    }
    manifest, labels = prepare(example, export_root, output_root)
    assert manifest["provenance"]["plan_record_id"] == artifact["planRecordId"]
    assert labels["plan_record_id"] == artifact["planRecordId"]
    assert labels["target_transcript"] == label
    assert label not in json.dumps(manifest["control"], sort_keys=True)
    assert label not in json.dumps(manifest, sort_keys=True)
