#!/usr/bin/env python3
"""Export certified Voryn V3/V4 conversations as native duplex PersonaPlex examples.

This exporter deliberately keeps the silent control frame separate from the agent
text/audio label.  A conversation is training-admitted only when it has valid
provenance/audio/ASR data and, when it claims barge-in coverage, contains a real
caller overlap followed by a recovery agent turn.  ``--allow-incomplete`` is a
diagnostic-only mode: it materializes audio for inspection but never writes those
examples to ``examples.jsonl``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA = "personaplex.controlled-duplex.example.v1"
BRANCH_ARTIFACT_SCHEMA = "personaplex.voryn-branch-artifact.v5"
RENDER_BINDING_SCHEMA = "personaplex.voryn-render-plan-binding.v5"
CAUSAL_GROUP_ROLES = (
    "verified_positive",
    "verified_negative",
    "uncertain",
    "superseded",
)
SAMPLE_RATE = 24_000
FRAME_RATE_HZ = 12.5
FRAME_DURATION_MS = 80
SAMPLES_PER_FRAME = 1_920


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", value):
        raise ValueError(f"{field} must be a canonical SHA-256 URI")
    return value


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def assert_target_free_payload(payload: Mapping[str, Any], target_text: str, target_hash: str | None = None) -> None:
    forbidden_keys = {
        "agenttext", "agentheardtext", "targettranscript", "targetlabel",
        "targetlabelsha256", "canonicalresponse", "canonical_response", "heardtext",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).casefold() in forbidden_keys:
                    raise ValueError(f"post-render bridge contains forbidden target-label field {key}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    serialised = canonical(payload)
    normalized_label = normalise(target_text)
    if len(normalized_label) >= 16 and normalized_label in normalise(serialised):
        raise ValueError("target text leaked into post-render bridge payload")
    if target_hash is not None and require_sha256(target_hash, "target label hash")[7:] in serialised.casefold():
        raise ValueError("target-label hash leaked into post-render bridge payload")


def validate_render_plan(plan: Mapping[str, Any]) -> tuple[str, str]:
    if plan.get("schemaVersion") != 7:
        raise ValueError("v5 post-render finalization requires a schemaVersion 7 Voryn plan")
    bridge = plan.get("postRenderBridge")
    if not isinstance(bridge, Mapping) or bridge.get("schema") != RENDER_BINDING_SCHEMA:
        raise ValueError("schema-7 plan lacks a typed v5 postRenderBridge binding")
    render_plan_id = require_sha256(plan.get("renderPlanId"), "renderPlanId")
    if content_hash(bridge) != render_plan_id:
        raise ValueError("renderPlanId does not bind the target-free postRenderBridge payload")
    group_id = bridge.get("groupId")
    role = bridge.get("siblingRole")
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("postRenderBridge.groupId is invalid")
    if role not in CAUSAL_GROUP_ROLES:
        raise ValueError(f"unsupported v5 sibling role {role!r}")
    counterfactual = plan.get("counterfactual") or {}
    if counterfactual.get("groupId") != group_id or counterfactual.get("branchId") != role:
        raise ValueError("postRenderBridge lineage differs from schema-7 counterfactual lineage")
    common_context = bridge.get("commonContext")
    if not isinstance(common_context, Mapping):
        raise ValueError("postRenderBridge lacks complete commonContext")
    if content_hash(common_context) != require_sha256(
        bridge.get("commonContextHash"), "commonContextHash"
    ):
        raise ValueError("postRenderBridge commonContextHash is stale")
    frame_contract = bridge.get("nativeFrameContract")
    if frame_contract != {
        "sampleRateHz": SAMPLE_RATE,
        "frameRateHz": FRAME_RATE_HZ,
        "frameDurationMs": FRAME_DURATION_MS,
        "samplesPerFrame": SAMPLES_PER_FRAME,
    }:
        raise ValueError("postRenderBridge does not pin the native 24 kHz/12.5 Hz contract")
    return group_id, role


def read_render_plans(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    plans: dict[tuple[str, str], dict[str, Any]] = {}
    groups: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            plan = json.loads(line)
            if not isinstance(plan, dict):
                raise ValueError(f"{path}:{line_number}: render plan must be an object")
            group_id, role = validate_render_plan(plan)
            key = (group_id, role)
            if key in plans:
                raise ValueError(f"duplicate schema-7 render plan for {group_id}/{role}")
            plans[key] = plan
            groups[group_id].add(role)
    if not plans:
        raise ValueError("compiled schema-7 plan contains no v5 render bindings")
    expected = set(CAUSAL_GROUP_ROLES)
    incomplete = {group: sorted(expected - roles) for group, roles in groups.items() if roles != expected}
    if incomplete:
        raise ValueError(f"compiled schema-7 plan has incomplete four-role groups: {incomplete}")
    return plans


def _integer_ms(value: Any, field: str) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative millisecond timestamp")
    rounded = int(round(float(value)))
    if abs(float(value) - rounded) > 1e-6:
        raise ValueError(f"{field} must resolve to an integer millisecond")
    return rounded


def floor_native_frame(milliseconds: Any, field: str) -> int:
    return _integer_ms(milliseconds, field) // FRAME_DURATION_MS


def ceil_native_frame(milliseconds: Any, field: str) -> int:
    return math.ceil(_integer_ms(milliseconds, field) / FRAME_DURATION_MS)


def voice_reference_id(value: Any, field: str) -> str:
    identifier = value.get("id") if isinstance(value, Mapping) else value
    if not isinstance(identifier, str) or not identifier:
        raise ValueError(f"{field} lacks an approved voice-reference id")
    return identifier


def select_pivot_target(records: list[dict[str, Any]], pivot_ordinal: int) -> dict[str, Any]:
    targets = [record for record in sorted(records, key=lambda item: item["turnIndex"]) if record.get("speaker") == "target"]
    if not isinstance(pivot_ordinal, int) or isinstance(pivot_ordinal, bool) or pivot_ordinal < 1:
        raise ValueError("pivotTargetOrdinal must be a positive integer")
    if len(targets) < pivot_ordinal:
        raise ValueError("certified branch does not contain its causal pivot target")
    target = targets[pivot_ordinal - 1]
    if (target.get("replay") or {}).get("role") == "shared_prefix_context_only":
        raise ValueError("causal pivot resolved to quarantined replay context")
    return target


def _timeline_event(
    timeline: Mapping[str, Any], conversation_id: str, record: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if timeline.get("schema") != "voxrn.duplex-timeline.v1":
        raise ValueError("unsupported or missing certified duplex timeline schema")
    if timeline.get("conversationId") != conversation_id:
        raise ValueError("duplex timeline conversation identity differs")
    events = timeline.get("events")
    if not isinstance(events, list) or not all(isinstance(item, dict) for item in events):
        raise ValueError("duplex timeline events are malformed")
    matches = [item for item in events if item.get("turnIndex") == record.get("turnIndex")]
    if len(matches) != 1:
        raise ValueError("duplex timeline must contain exactly one causal pivot event")
    event = matches[0]
    if event.get("speaker") != "target" or event.get("audioPath") != record.get("audioPath"):
        raise ValueError("causal pivot timeline event differs from the certified record")
    if canonical(event.get("timing")) != canonical(record.get("timing")):
        raise ValueError("causal pivot timing differs between certified record and timeline")
    return event, sorted(events, key=lambda item: int(item.get("turnIndex", -1)))


def _actual_events(
    record: Mapping[str, Any], events: list[dict[str, Any]], pivot: int
) -> dict[str, Any]:
    timing = record.get("timing") or {}
    started = _integer_ms(timing.get("startedAtMs"), "target.startedAtMs")
    audible_end = _integer_ms(timing.get("audibleEndedAtMs"), "target.audibleEndedAtMs")
    rendered_end = _integer_ms(timing.get("endedAtMs"), "target.endedAtMs")
    if not started < audible_end <= rendered_end:
        raise ValueError("certified target timing is not monotonic")
    turn_taking = timing.get("turnTaking") or {}
    next_barge = turn_taking.get("nextBargeIn")
    was_cancelled = audible_end < rendered_end
    if was_cancelled != isinstance(next_barge, Mapping):
        raise ValueError("audible cutoff and nextBargeIn evidence disagree")
    generation = record.get("generation") or {}
    generation_id = generation.get("generationId") or generation.get("id")
    if not isinstance(generation_id, str) or not generation_id:
        generation_id = "voryn-generation:" + content_hash({
            "conversationId": record.get("conversationId"),
            "runIndex": record.get("runIndex"),
            "turnIndex": record.get("turnIndex"),
        }).removeprefix("sha256:")
    if was_cancelled:
        barge_at = _integer_ms(next_barge.get("bargeInAtMs"), "nextBargeIn.bargeInAtMs")
        following_callers = [
            event for event in events
            if int(event.get("turnIndex", -1)) > int(record["turnIndex"])
            and event.get("speaker") == "caller"
        ]
        if not following_callers:
            raise ValueError("barge-in target has no following caller timeline event")
        caller_timing = following_callers[0].get("timing") or {}
        caller_turn_taking = caller_timing.get("turnTaking") or {}
        if (
            caller_turn_taking.get("eventType") != "caller_barge_in"
            or caller_turn_taking.get("interruptedTurnIndex") != record.get("turnIndex")
            or _integer_ms(caller_timing.get("startedAtMs"), "barge-in caller.startedAtMs") != barge_at
        ):
            raise ValueError("caller timeline does not authenticate the planned barge-in")
        cutoff_frame = floor_native_frame(barge_at, "barge-in timestamp")
        cancellation_frame = ceil_native_frame(audible_end, "cancellation timestamp")
        total_frames = ceil_native_frame(
            max(_integer_ms((event.get("timing") or {}).get("audibleEndedAtMs"), "timeline audible end") for event in events),
            "timeline duration",
        )
        if cutoff_frame < pivot or cancellation_frame < cutoff_frame or cancellation_frame >= total_frames:
            raise ValueError("barge-in/cancellation frames are outside the actual native timeline")
        barge = {"occurred": True, "cutoffFrame": cutoff_frame}
        cancellation = {
            "generationId": generation_id,
            "cancelled": True,
            "atFrame": cancellation_frame,
        }
    else:
        barge = {"occurred": False, "cutoffFrame": None}
        cancellation = {
            "generationId": generation_id,
            "cancelled": False,
            "atFrame": None,
        }
    action = generation.get("replyAction")
    if action not in {"speak", "end_call"}:
        raise ValueError("certified target lacks an authentic speak/end_call generation action")
    if action == "end_call":
        end_call = {
            "decision": "end_call",
            "decisionFrame": pivot,
            "toolCallFrame": max(pivot, ceil_native_frame(audible_end, "end-call timestamp") - 1),
        }
    else:
        end_call = {"decision": "continue", "decisionFrame": pivot, "toolCallFrame": None}
    return {"bargeIn": barge, "cancellation": cancellation, "endCall": end_call}


def finalize_v5_branch_artifact(
    plan: Mapping[str, Any], records: list[dict[str, Any]], timeline: Mapping[str, Any],
    timeline_path: Path, source_export_example_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    group_id, role = validate_render_plan(plan)
    bridge = plan["postRenderBridge"]
    conversation_ids = {record.get("conversationId") for record in records}
    groups = {record.get("counterfactualGroupId") for record in records}
    roles = {record.get("counterfactualBranchId") for record in records}
    scenarios = {record.get("scenarioKey") for record in records}
    if len(conversation_ids) != 1 or groups != {group_id} or roles != {role}:
        raise ValueError("certified records do not share the render plan's branch lineage")
    if scenarios != {bridge["scenarioKey"]}:
        raise ValueError("certified records do not share the render plan's scenarioKey")
    pivot_record = select_pivot_target(records, bridge["pivotTargetOrdinal"])
    _event, ordered_events = _timeline_event(timeline, str(pivot_record["conversationId"]), pivot_record)
    timing = pivot_record.get("timing") or {}
    pivot_frame = floor_native_frame(timing.get("startedAtMs"), "pivot target startedAtMs")
    if pivot_frame < 1:
        raise ValueError("causal pivot must have at least one native frame of prior context")
    previous_events = [
        event for event in ordered_events
        if int(event.get("turnIndex", -1)) < int(pivot_record["turnIndex"])
    ]
    available_frame = 0 if not previous_events else ceil_native_frame(
        (previous_events[-1].get("timing") or {}).get("audibleEndedAtMs"),
        "previous audible end",
    )
    if available_frame >= pivot_frame:
        raise ValueError("control availability is not strictly before the causal pivot target")
    expected_binding = bridge.get("pivotControlBinding") or {}
    control = pivot_record.get("control") or {}
    frame = control.get("frame") or {}
    frame_hash = require_sha256(control.get("frameHash"), "pivot control frameHash")
    revision = frame.get("stateRevision")
    if (
        frame_hash != expected_binding.get("frameHash")
        or revision != expected_binding.get("revision")
        or expected_binding.get("availableBeforeTarget") is not True
    ):
        raise ValueError("certified pivot control does not match the immutable render plan")
    voice_pair = bridge.get("voicePair") or {}
    actual_caller = next(
        (voice_reference_id(record.get("voiceReference"), "caller voice") for record in records if record.get("speaker") == "caller"),
        None,
    )
    actual_agent = next(
        (voice_reference_id(record.get("voiceReference"), "agent voice") for record in records if record.get("speaker") == "target"),
        None,
    )
    if actual_caller != voice_pair.get("caller") or actual_agent != voice_pair.get("agent"):
        raise ValueError("certified branch voice pair differs from the immutable group voice pair")
    control_binding = {
        "frameHash": frame_hash,
        "revision": revision,
        "availableFrame": available_frame,
    }
    evidence_hash = control.get("evidenceHash")
    if evidence_hash is not None:
        control_binding["evidenceFrameHash"] = require_sha256(
            evidence_hash, "pivot evidenceFrameHash"
        )
    payload = {
        "schema": BRANCH_ARTIFACT_SCHEMA,
        "groupId": group_id,
        "siblingRole": role,
        "sharedPrefixId": bridge["sharedPrefixId"],
        "commonContextHash": bridge["commonContextHash"],
        "commonContext": deepcopy(bridge["commonContext"]),
        "premiseId": bridge["premiseId"],
        "templateId": bridge["templateId"],
        "lineageIdentifiers": deepcopy(bridge["lineageIdentifiers"]),
        "controlOperator": deepcopy(bridge["controlOperator"]),
        "voicePair": deepcopy(bridge["voicePair"]),
        "controlBinding": control_binding,
        "nativeFrameContract": deepcopy(bridge["nativeFrameContract"]),
        "nativePivotFrame": pivot_frame,
        "actualEvents": _actual_events(pivot_record, ordered_events, pivot_frame),
        "sourceExportExampleId": source_export_example_id,
        "renderEvidence": {
            "renderPlanId": plan["renderPlanId"],
            "conversationId": pivot_record["conversationId"],
            "targetTurnIndex": pivot_record["turnIndex"],
            "timelineSchema": timeline["schema"],
            "timelineSha256": "sha256:" + hashlib.sha256(timeline_path.read_bytes()).hexdigest(),
        },
    }
    target_hash = (control.get("targetLabelSha256") or content_hash(str(pivot_record.get("text") or "")))
    assert_target_free_payload(payload, str(pivot_record.get("text") or ""), target_hash)
    artifact = {**payload, "planRecordId": content_hash(payload)}
    return artifact, pivot_record


def validate_complete_v5_artifacts(
    plans: Mapping[tuple[str, str], Mapping[str, Any]], artifacts: list[dict[str, Any]]
) -> None:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for artifact in artifacts:
        if artifact.get("schema") != BRANCH_ARTIFACT_SCHEMA:
            raise ValueError("post-render output contains an unsupported branch artifact")
        plan_id = require_sha256(artifact.get("planRecordId"), "planRecordId")
        payload = {key: value for key, value in artifact.items() if key != "planRecordId"}
        if content_hash(payload) != plan_id:
            raise ValueError("planRecordId does not bind its immutable branch artifact")
        key = (artifact.get("groupId"), artifact.get("siblingRole"))
        if key in by_key:
            raise ValueError(f"multiple certified pivot artifacts were emitted for {key}")
        if key not in plans:
            raise ValueError(f"post-render artifact has no schema-7 render plan: {key}")
        if artifact.get("renderEvidence", {}).get("renderPlanId") != plans[key].get("renderPlanId"):
            raise ValueError("post-render artifact binds the wrong schema-7 render plan")
        by_key[key] = artifact
    if set(by_key) != set(plans):
        missing = sorted(set(plans) - set(by_key))
        raise ValueError(f"post-render finalization did not emit exactly one pivot per plan: {missing}")
    groups: dict[str, set[str]] = defaultdict(set)
    for group_id, role in by_key:
        groups[str(group_id)].add(str(role))
    if any(roles != set(CAUSAL_GROUP_ROLES) for roles in groups.values()):
        raise ValueError("post-render branch artifacts do not form exact four-role groups")


def run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr[-2000:]}"
        )


def audio_probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels,sample_rate,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"no audio stream: {path}")
    return streams[0]


def jsonl_inputs(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file() and path.suffix == ".jsonl":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.jsonl"))


def read_conversations(inputs: Iterable[Path]) -> dict[str, list[dict[str, Any]]]:
    conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in jsonl_inputs(inputs):
        with source.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("schema") not in {"voxrn.synthetic-conversation.v3", "voxrn.synthetic-conversation.v4"}:
                    continue
                if not record.get("conversationId"):
                    raise ValueError(f"{source}:{number}: missing conversationId")
                record["_source_path"] = source
                conversations[record["conversationId"]].append(record)
    return conversations


def find_timeline(records: list[dict[str, Any]]) -> tuple[Path | None, dict[str, Any] | None]:
    source = records[0]["_source_path"]
    timeline_ref = records[0].get("duplexTimelinePath")
    if not timeline_ref:
        return None, None
    path = source.parent / timeline_ref
    if not path.is_file():
        return path, None
    return path, json.loads(path.read_text(encoding="utf-8"))


def target_control_issues(record: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    control = record.get("control") or {}
    frame = control.get("frame") or {}
    plan = control.get("plan") or {}
    quality = record.get("quality") or {}
    if record.get("speaker") != "target":
        return issues
    replay = record.get("replay") or {}
    if replay.get("role") == "shared_prefix_context_only":
        training = record.get("training") or {}
        if training.get("eligible") or "shared_prefix_replay_context_only" not in training.get("exclusionReasons", []):
            issues.append("shared_prefix_replay_not_quarantined")
        # The shared prefix is duplex conditioning context. Generic audio,
        # timeline, and provenance checks still run in conversation_issues(),
        # but it cannot be required to be a branch-local target label.
        return issues
    if not record.get("training", {}).get("eligible"):
        issues.append("target_not_marked_training_eligible")
    if record.get("semanticAdherence", {}).get("verificationStatus") != "batch_certified":
        issues.append("target_semantic_certificate_missing")
    if not quality.get("accepted"):
        issues.append("target_quality_not_accepted")
    if not frame or not plan:
        issues.append("missing_control_frame_or_plan")
        return issues
    if frame.get("stateHash") != plan.get("contextHash"):
        issues.append("control_plan_context_hash_mismatch")
    if not frame.get("frameId") or not frame.get("stateRevision"):
        issues.append("control_frame_identity_missing")
    evidence = control.get("evidence")
    if record.get("schema") == "voxrn.synthetic-conversation.v4" and evidence is not None:
        if not isinstance(evidence, dict) or (
            evidence.get("conversationId") != record.get("conversationId")
            or evidence.get("counterfactual", {}).get("groupId") != record.get("counterfactualGroupId")
            or evidence.get("counterfactual", {}).get("branchId") != record.get("counterfactualBranchId")
            or evidence.get("plan", {}).get("revision") != plan.get("revision")
        ):
            issues.append("v4_target_evidence_lineage_invalid")
    serialised = canonical({"frame": frame, "evidence": evidence})
    label = normalise(record.get("text", ""))
    if label and len(label) >= 16 and label in normalise(serialised):
        issues.append("target_text_leaked_into_control_frame")
    if "canonicalResponse" in serialised or "canonical_response" in serialised:
        issues.append("canonical_response_leaked_into_control_frame")
    return issues


def target_training_issues(record: dict[str, Any]) -> list[str]:
    """Return label-local exclusions without invalidating causal neighbour turns."""
    issues = target_control_issues(record)
    if record.get("speaker") != "target":
        return issues
    if (record.get("replay") or {}).get("role") == "shared_prefix_context_only":
        return issues
    # V4 records retain counterfactual provenance even when a target has no
    # late-evidence frame. Primary semantic-prefix training needs every
    # certified target turn; the evidence-stream stage selects the smaller
    # evidence-bearing subset separately.
    return sorted(set(issues))


def interruption_issues(records: list[dict[str, Any]]) -> list[str]:
    """Require real audio overlap and a recovery response when coverage says barge-in."""
    issues: list[str] = []
    requires_barge = max(
        (int(record.get("characteristics", {}).get("interruption", 0)) for record in records),
        default=0,
    ) >= 70
    if not requires_barge:
        return issues
    ordered = sorted(records, key=lambda item: item["turnIndex"])
    observed = False
    recovered = False
    for index, record in enumerate(ordered):
        if record.get("speaker") != "target":
            continue
        timing = record.get("timing") or {}
        audible_end = timing.get("audibleEndedAtMs")
        rendered_end = timing.get("endedAtMs")
        if not isinstance(audible_end, (int, float)) or not isinstance(rendered_end, (int, float)):
            continue
        if audible_end >= rendered_end:
            continue
        following = ordered[index + 1 :]
        caller = next((item for item in following if item.get("speaker") == "caller"), None)
        if not caller:
            continue
        caller_start = (caller.get("timing") or {}).get("startedAtMs")
        if not isinstance(caller_start, (int, float)) or caller_start >= rendered_end:
            continue
        observed = True
        recovery = next((item for item in following if item.get("speaker") == "target"), None)
        if recovery and (recovery.get("control") or {}).get("frame", {}).get("turnTaking", {}).get("recoveryExpected"):
            recovered = True
    if not observed:
        issues.append("claimed_barge_in_without_real_caller_overlap")
    elif not recovered:
        issues.append("barge_in_without_recovery_agent_turn")
    return issues


def conversation_issues(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None]:
    issues: list[str] = []
    ordered = sorted(records, key=lambda item: item["turnIndex"])
    seen_turns: set[int] = set()
    for record in ordered:
        turn = record.get("turnIndex")
        if not isinstance(turn, int) or turn in seen_turns:
            issues.append("duplicate_or_invalid_turn_index")
        seen_turns.add(turn)
        quality = record.get("quality") or {}
        if not quality.get("accepted"):
            issues.append(f"turn_{turn}_quality_not_accepted")
        audio_ref = record.get("audioPath")
        audio_path = record["_source_path"].parent / str(audio_ref or "")
        if not audio_ref or not audio_path.is_file():
            issues.append(f"turn_{turn}_audio_missing")
        if record.get("speaker") == "caller" and record.get("authenticity", {}).get("status") != "batch_certified":
            issues.append(f"turn_{turn}_caller_authenticity_certificate_missing")
    timeline_path, timeline = find_timeline(ordered)
    if not timeline:
        issues.append("duplex_timeline_missing")
    else:
        events = {event.get("turnIndex"): event for event in timeline.get("events", [])}
        if timeline.get("conversationId") != ordered[0].get("conversationId"):
            issues.append("duplex_timeline_conversation_mismatch")
        for record in ordered:
            event = events.get(record["turnIndex"])
            if not event:
                issues.append(f"turn_{record['turnIndex']}_missing_from_timeline")
            elif event.get("audioPath") != record.get("audioPath"):
                issues.append(f"turn_{record['turnIndex']}_timeline_audio_mismatch")
    issues.extend(interruption_issues(ordered))
    return ordered, sorted(set(issues)), timeline


def materialize_duplex(records: list[dict[str, Any]], output: Path, sample_rate: int) -> dict[str, Any]:
    total_ms = max(int((record.get("timing") or {}).get("audibleEndedAtMs", 0)) for record in records)
    if total_ms <= 0:
        raise ValueError("non-positive duplex duration")
    total_seconds = total_ms / 1000
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="personaplex-duplex-") as temp_name:
        temp = Path(temp_name)
        channel_files: dict[str, list[Path]] = {"target": [], "caller": []}
        for record in records:
            timing = record["timing"]
            audible_ms = int(timing.get("audibleSpeechMs", timing.get("speechMs", 0)))
            start_ms = int(timing.get("startedAtMs", 0))
            if audible_ms <= 0:
                raise ValueError(f"turn {record['turnIndex']} has no audible speech")
            source = record["_source_path"].parent / record["audioPath"]
            clip = temp / f"turn-{record['turnIndex']}.wav"
            run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-af",
                    (
                        f"atrim=duration={audible_ms / 1000:.3f},aresample={sample_rate},"
                        f"asetpts=PTS-STARTPTS,adelay={start_ms}:all=1,"
                        f"apad=whole_dur={total_seconds:.3f},atrim=duration={total_seconds:.3f}"
                    ),
                    "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(clip),
                ]
            )
            channel_files[record["speaker"]].append(clip)

        mixes: dict[str, Path] = {}
        for speaker, clips in channel_files.items():
            mix = temp / f"{speaker}.wav"
            if clips:
                run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        *sum((["-i", str(clip)] for clip in clips), []),
                        "-filter_complex", f"amix=inputs={len(clips)}:normalize=0",
                        "-t", f"{total_seconds:.3f}", "-ac", "1", "-ar", str(sample_rate),
                        "-c:a", "pcm_s16le", str(mix),
                    ]
                )
            else:
                run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", f"anullsrc=r={sample_rate}:cl=mono", "-t", f"{total_seconds:.3f}",
                        "-c:a", "pcm_s16le", str(mix),
                    ]
                )
            mixes[speaker] = mix
        run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(mixes["target"]), "-i", str(mixes["caller"]),
                "-filter_complex", "[0:a][1:a]join=inputs=2:channel_layout=stereo[duplex]",
                "-map", "[duplex]", "-t", f"{total_seconds:.3f}", "-ar", str(sample_rate),
                "-ac", "2", "-c:a", "pcm_s16le", str(output),
            ]
        )
    probe = audio_probe(output)
    return {
        "path": str(output),
        "sha256": "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(),
        "durationMs": total_ms,
        "sampleRate": int(probe["sample_rate"]),
        "channels": int(probe["channels"]),
    }


def target_example(
    record: dict[str, Any], duplex: dict[str, Any], root: Path,
    records: list[dict[str, Any]], branch_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    control = record["control"]
    frame = control["frame"]
    source_audio = record["_source_path"].parent / record["audioPath"]
    timing = record["timing"]
    caller_reference = next((item.get("voiceReference") for item in records if item.get("speaker") == "caller"), None)
    agent_reference = next((item.get("voiceReference") for item in records if item.get("speaker") == "target"), None)
    return {
        "schema": SCHEMA,
        "exampleId": f"{record['conversationId']}:target:{record['turnIndex']}",
        "conversationId": record["conversationId"],
        "targetTurnIndex": record["turnIndex"],
        "duplexAudio": {
            "path": str(Path(duplex["path"]).relative_to(root)),
            "sha256": duplex["sha256"],
            "sampleRate": duplex["sampleRate"],
            "channels": {"agent": 0, "caller": 1},
            "durationMs": duplex["durationMs"],
        },
        "target": {
            "audibleStartMs": int(timing["startedAtMs"]),
            "audibleEndMs": int(timing["audibleEndedAtMs"]),
            "renderedEndMs": int(timing["endedAtMs"]),
            "sourceAudioSha256": record.get("audioSha256"),
            "sourceAudioPath": str(source_audio),
        },
        "controlFrame": frame,
        "controlFrameHash": control.get("frameHash"),
        "evidenceFrame": control.get("evidence"),
        "evidenceFrameHash": control.get("evidenceHash"),
        "counterfactual": {
            "groupId": record.get("counterfactualGroupId"),
            "branchId": record.get("counterfactualBranchId"),
            "siblingRole": record.get("counterfactualBranchId"),
            "pivotTargetOrdinal": record.get("counterfactualPivotTargetOrdinal"),
        } if record.get("schema") == "voxrn.synthetic-conversation.v4" else None,
        "labels": {
            "agentText": record["text"],
            "agentHeardText": record.get("heardText"),
            "asr": record.get("asr"),
        },
        "quality": record.get("quality"),
        "provenance": {
            "datasetId": record["datasetId"],
            "scenarioKey": record.get("scenarioKey"),
            "voiceReference": record.get("voiceReference"),
            "voicePair": {
                "caller": caller_reference,
                "agent": agent_reference,
            },
            "sourceManifest": str(record["_source_path"]),
            **({
                "planRecordId": branch_artifact["planRecordId"],
                "renderPlanId": branch_artifact["renderEvidence"]["renderPlanId"],
                "branchArtifactSchema": branch_artifact["schema"],
            } if branch_artifact is not None else {}),
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Certified Voryn V3/V4 JSONL file(s) or directory roots")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument(
        "--compiled-plan", type=Path,
        help="Schema-7 v5 render plans used to finalize one certified causal-pivot artifact per branch",
    )
    parser.add_argument("--allow-incomplete", action="store_true", help="materialize rejected conversations for diagnostic playback only")
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    if args.sample_rate != 24000:
        raise ValueError("native PersonaPlex training export is fixed at 24000 Hz")
    if args.compiled_plan is not None and args.allow_incomplete:
        raise ValueError("v5 branch-artifact finalization cannot run in diagnostic incomplete mode")

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    conversations = read_conversations(args.inputs)
    render_plans = read_render_plans(args.compiled_plan.resolve()) if args.compiled_plan else {}
    examples: list[dict[str, Any]] = []
    branch_artifacts: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    admitted_conversations = 0
    quarantined_target_turns = 0
    for conversation_id, raw_records in sorted(conversations.items()):
        records, conversation_level_issues, timeline = conversation_issues(raw_records)
        if conversation_level_issues:
            rejected.append({
                "scope": "conversation", "conversationId": conversation_id,
                "issues": conversation_level_issues,
            })
        if conversation_level_issues and not args.allow_incomplete:
            continue
        eligible_targets = []
        for record in records:
            if record.get("speaker") != "target":
                continue
            if (record.get("replay") or {}).get("role") == "shared_prefix_context_only":
                continue
            target_issues = target_training_issues(record)
            if target_issues:
                quarantined_target_turns += 1
                rejected.append({
                    "scope": "target_turn", "conversationId": conversation_id,
                    "turnIndex": record.get("turnIndex"), "issues": target_issues,
                })
                continue
            eligible_targets.append(record)
        render_plan = None
        if render_plans:
            lineage = {
                (record.get("counterfactualGroupId"), record.get("counterfactualBranchId"))
                for record in records
            }
            if len(lineage) != 1:
                raise ValueError(f"{conversation_id}: certified conversation has ambiguous v5 lineage")
            plan_key = next(iter(lineage))
            render_plan = render_plans.get(plan_key)
            if render_plan is None:
                raise ValueError(f"{conversation_id}: no schema-7 render plan for {plan_key}")
            pivot_record = select_pivot_target(
                records, render_plan["postRenderBridge"]["pivotTargetOrdinal"]
            )
            if pivot_record not in eligible_targets:
                raise ValueError(f"{conversation_id}: causal pivot target was not training-admitted")
            eligible_targets = [pivot_record]
        if not eligible_targets and not args.allow_incomplete:
            rejected.append({
                "scope": "conversation", "conversationId": conversation_id,
                "issues": ["no_training_eligible_target_turns"],
            })
            continue
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", conversation_id)
        duplex = materialize_duplex(records, root / "audio" / f"{safe_name}.wav", args.sample_rate)
        branch_artifact = None
        if render_plan is not None:
            timeline_path, _ = find_timeline(records)
            if timeline_path is None or timeline is None:
                raise ValueError(f"{conversation_id}: certified timeline is unavailable")
            source_example_id = f"{eligible_targets[0]['conversationId']}:target:{eligible_targets[0]['turnIndex']}"
            branch_artifact, finalized_target = finalize_v5_branch_artifact(
                render_plan, records, timeline, timeline_path, source_example_id
            )
            if finalized_target is not eligible_targets[0]:
                raise ValueError(f"{conversation_id}: finalizer selected a different causal pivot")
        target_rows = [target_example(
            record, duplex, root, records, branch_artifact=branch_artifact
        ) for record in eligible_targets]
        if not conversation_level_issues and target_rows:
            admitted_conversations += 1
            examples.extend(target_rows)
            if branch_artifact is not None:
                branch_artifacts.append(branch_artifact)
        else:
            diagnostics.extend([{
                **row, "trainingAdmitted": False,
                "rejectionIssues": conversation_level_issues or ["no_training_eligible_target_turns"],
            } for row in target_rows])

    if render_plans:
        validate_complete_v5_artifacts(render_plans, branch_artifacts)
    write_jsonl(root / "examples.jsonl", examples)
    write_jsonl(root / "branch_artifacts.v5.jsonl", sorted(
        branch_artifacts, key=lambda item: (item["groupId"], CAUSAL_GROUP_ROLES.index(item["siblingRole"]))
    ))
    write_jsonl(root / "diagnostic_examples.jsonl", diagnostics)
    write_jsonl(root / "rejections.jsonl", rejected)
    manifest = {
        "schema": "personaplex.controlled-duplex-export.v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sampleRate": args.sample_rate,
        "sourceConversationCount": len(conversations),
        "admittedConversationCount": admitted_conversations,
        "admittedExampleCount": len(examples),
        "v5BranchArtifactCount": len(branch_artifacts),
        "diagnosticExampleCount": len(diagnostics),
        "rejectedConversationCount": len(rejected),
        "quarantinedTargetTurnCount": quarantined_target_turns,
        "strict": not args.allow_incomplete,
        "files": {
            "examples": "examples.jsonl",
            "v5BranchArtifacts": "branch_artifacts.v5.jsonl",
            "diagnosticExamples": "diagnostic_examples.jsonl",
            "rejections": "rejections.jsonl",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"export failed: {error}", file=sys.stderr)
        raise
