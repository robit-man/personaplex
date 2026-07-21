#!/usr/bin/env python3
"""Compile selected causal groups into target-free Voryn lane-plan entries.

The compiler preserves the complete causal contract, emits one validated typed
ControlTrainingFrame per agent target, and writes one render-once shared-prefix plan
per group. It supports both legacy available/constrained pairs and v5 four-sibling
groups without treating a planner prompt as semantic conditioning.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.contracts import (  # noqa: E402
    ContractError,
    validate_control_frame_mapping,
)
from ground_truth_finetuning.training.diverse_cascade import (  # noqa: E402
    CascadeError,
    JsonOnlyPlanner,
    PlannerConfig,
    assert_no_target_leak,
    canonical_json,
    content_hash,
    load_json,
    load_jsonl,
    request_requires_typed_trajectories,
    request_sibling_roles,
    validate_pair_spec,
    validate_request,
    validate_scenario_contract,
    validate_topic_card,
    validate_trajectory_seed,
    validate_v4_pair_spec,
    validate_v4_trajectory_seed,
    write_json,
    write_jsonl,
)


V8_CORPUS = "personaplex-synthetic-counterfactual-v9-cascade"
V5_RENDER_BINDING_SCHEMA = "personaplex.voryn-render-plan-binding.v5"
TEMPLATE_FIELDS = {
    "branchId", "topic", "topicFamily", "topicSeedId", "contextLens",
    "conversationMode", "lengthProfile", "turnCadence", "responseLengthProfile",
    "openingStyle", "closingStyle", "coverageProfile", "turns", "coverage",
    "controlProgram", "dynamics", "branchInstruction",
}
CONTROL_FIELDS = {
    "id", "targetOrdinal", "source", "kind", "facts", "commitments",
    "uncertainty", "policyConstraints", "toolResult", "nextGoal", "guidance",
    "semanticContext", "endCall", "requiredFacts", "forbiddenClaims", "mustAsk",
    "updateReason", "expectedEffects",
}
COMMON_TEMPLATE_FIELDS = (
    "topic", "topicFamily", "topicSeedId", "contextLens", "conversationMode",
    "lengthProfile", "turnCadence", "responseLengthProfile", "openingStyle",
    "coverageProfile", "turns",
)
EVENT_ORDINAL_FIELDS = (
    "targetOrdinal", "agentTargetOrdinal", "targetTurnId", "agentTurnOrdinal",
)


def stable_integer(value: Any) -> int:
    return int(sha256(canonical_json(value).encode("utf-8")).hexdigest()[:8], 16)


def approved_voice_ids(manifest_path: Path) -> list[str]:
    manifest = load_json(manifest_path)
    references = manifest.get("references")
    if not isinstance(references, list):
        raise CascadeError("Voice manifest must contain references")
    voice_ids = sorted({
        str(item.get("id") or "")
        for item in references
        if isinstance(item, dict) and item.get("id")
    })
    if len(voice_ids) < 2:
        raise CascadeError("V8 plan compilation requires at least two approved voice IDs")
    return voice_ids


def assign_voice_pair(
    request: dict[str, Any], group_id: str, voice_ids: list[str]
) -> tuple[str, str]:
    seed = stable_integer({
        "seedRevision": request["seedRevision"],
        "groupId": group_id,
        "voiceManifest": request["allowedVoicesManifest"],
    })
    caller_index = seed % len(voice_ids)
    target_index = (seed // len(voice_ids) + 1) % len(voice_ids)
    if target_index == caller_index:
        target_index = (target_index + 1) % len(voice_ids)
    return voice_ids[caller_index], voice_ids[target_index]


def build_common_context(
    request: dict[str, Any],
    selection: dict[str, Any],
    scenario: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    """Reproduce the target-free context whose hash selects a causal group."""
    context = {
        "requestId": request["requestId"],
        "scenario": deepcopy(scenario),
        "trajectory": deepcopy(trajectory),
        "groupId": selection["groupId"],
    }
    if len(request_sibling_roles(request)) == 4:
        context["interventionFamily"] = trajectory["interventionFamily"]
        context["typedPivot"] = deepcopy(trajectory["typedPivot"])
    assert_no_target_leak(context)
    return context


def require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CascadeError(f"V8 template {field} must be nonempty text")
    return value


def require_string_array(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise CascadeError(f"V8 template {field} must be a string array")
    return value


def normalized_group_branches(pair: dict[str, Any]) -> list[dict[str, Any]]:
    values = pair.get("branches", pair.get("siblings"))
    if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
        raise CascadeError("Causal group branches/siblings must be an object array")
    normalized: list[dict[str, Any]] = []
    for value in values:
        branch = deepcopy(value)
        role = branch.get("branchId", branch.get("siblingRole"))
        if not isinstance(role, str) or not role:
            raise CascadeError("Every causal sibling requires branchId or siblingRole")
        branch["branchId"] = role
        branch.pop("siblingRole", None)
        normalized.append(branch)
    if len({item["branchId"] for item in normalized}) != len(normalized):
        raise CascadeError("Causal sibling roles must be unique")
    return normalized


def validate_control_program(
    program: Any, branch_id: str, turns: int
) -> list[dict[str, Any]]:
    if not isinstance(program, list) or not program:
        raise CascadeError("V8 template controlProgram must be nonempty")
    maximum_ordinal = max(1, turns // 2)
    ordinals: set[int] = set()
    validated: list[dict[str, Any]] = []
    for raw_item in program:
        if not isinstance(raw_item, dict):
            raise CascadeError("V8 controlProgram entries must be objects")
        item = deepcopy(raw_item)
        missing = CONTROL_FIELDS - set(item)
        if missing:
            raise CascadeError(f"V8 controlProgram is missing {sorted(missing)}")
        ordinal = item["targetOrdinal"]
        if not isinstance(ordinal, int) or not 1 <= ordinal <= maximum_ordinal:
            raise CascadeError("V8 control targetOrdinal is outside the agent-turn range")
        if ordinal in ordinals:
            raise CascadeError("V8 control program has duplicate target ordinals")
        ordinals.add(ordinal)
        for key in ("id", "source", "kind", "nextGoal", "guidance", "updateReason"):
            require_text(item[key], f"controlProgram.{key}")
        for key in (
            "facts", "commitments", "uncertainty", "policyConstraints",
            "requiredFacts", "forbiddenClaims", "mustAsk", "expectedEffects",
        ):
            require_string_array(item[key], f"controlProgram.{key}")
        if not isinstance(item["semanticContext"], dict) or not isinstance(item["endCall"], bool):
            raise CascadeError("V8 control semanticContext/endCall are invalid")
        if item["toolResult"] is not None and not isinstance(item["toolResult"], dict):
            raise CascadeError("V8 control toolResult must be object or null")
        assert_no_target_leak(item)
        validated.append(item)
    expected_ordinals = set(range(1, maximum_ordinal + 1))
    if ordinals != expected_ordinals:
        raise CascadeError("V8 controlProgram must contain exactly one entry for every agent target")
    end_call_entries = [item for item in validated if item["endCall"]]
    if len(end_call_entries) != 1:
        raise CascadeError("V8 control program must have exactly one model-selected end_call stage")
    if end_call_entries[0]["targetOrdinal"] != maximum_ordinal:
        raise CascadeError("V8 end_call stage must be the final target ordinal")
    return sorted(validated, key=lambda item: item["targetOrdinal"])


def validate_template(template: dict[str, Any], branch_id: str) -> dict[str, Any]:
    normalized = deepcopy(template)
    declared_role = normalized.get("branchId", normalized.get("siblingRole"))
    if declared_role != branch_id:
        raise CascadeError("V8 template role does not match causal sibling")
    normalized["branchId"] = branch_id
    normalized.pop("siblingRole", None)
    missing = TEMPLATE_FIELDS - set(normalized)
    if missing:
        raise CascadeError(f"V8 template is missing {sorted(missing)}")
    for field in COMMON_TEMPLATE_FIELDS[:-1] + ("closingStyle", "branchInstruction"):
        require_text(normalized[field], field)
    turns = normalized["turns"]
    if not isinstance(turns, int) or not 4 <= turns <= 48 or turns % 2:
        raise CascadeError("V8 template turns must be an even integer in [4, 48]")
    if not isinstance(normalized["coverage"], dict) or not isinstance(normalized["dynamics"], dict):
        raise CascadeError("V8 template coverage and dynamics must be objects")
    for key in (
        "intent", "trajectory", "interactionClass", "speechStyle", "turnPattern",
        "nextGoal",
    ):
        require_text(normalized["coverage"].get(key), f"coverage.{key}")
    for key in ("controlSources", "requiredStateFields"):
        require_string_array(normalized["coverage"].get(key), f"coverage.{key}")
    if normalized["coverage"].get("requireControlForAllTargets") is not True:
        raise CascadeError("coverage.requireControlForAllTargets must be true")
    for key in (
        "assertiveness", "skepticism", "compliance", "resistance", "recovery",
        "hesitation", "pace", "interruption",
    ):
        value = normalized["dynamics"].get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 100:
            raise CascadeError(f"dynamics.{key} must be an integer in [0, 100]")
    normalized["controlProgram"] = validate_control_program(
        normalized["controlProgram"], branch_id, turns
    )
    assert_no_target_leak(normalized)
    return normalized


def _bind_authoritative_shared_prefix(
    templates: list[dict[str, Any]], roles: tuple[str, ...], pivot_ordinal: int
) -> list[dict[str, Any]]:
    by_role = {template["branchId"]: deepcopy(template) for template in templates}
    leader = by_role[roles[0]]
    maximum_ordinal = leader["turns"] // 2
    if not 1 <= pivot_ordinal <= maximum_ordinal:
        raise CascadeError("Causal pivot lies outside the compiled agent targets")
    for template in templates[1:]:
        for field in COMMON_TEMPLATE_FIELDS:
            if canonical_json(template[field]) != canonical_json(leader[field]):
                raise CascadeError(f"Sibling templates disagree on shared field {field}")
    shared_controls = {
        item["targetOrdinal"]: deepcopy(item)
        for item in leader["controlProgram"]
        if item["targetOrdinal"] < pivot_ordinal
    }
    result: list[dict[str, Any]] = []
    for role in roles:
        template = by_role[role]
        controls = {
            item["targetOrdinal"]: deepcopy(item)
            for item in template["controlProgram"]
        }
        controls.update(deepcopy(shared_controls))
        template["controlProgram"] = [controls[index] for index in sorted(controls)]
        result.append(validate_template(template, role))
    return result


def create_templates(
    planner: JsonOnlyPlanner,
    request: dict[str, Any],
    topic: dict[str, Any],
    scenario: dict[str, Any],
    trajectory: dict[str, Any],
    pair: dict[str, Any],
) -> list[dict[str, Any]]:
    roles = request_sibling_roles(request)
    branches = {item["branchId"]: item for item in normalized_group_branches(pair)}
    if set(branches) != set(roles):
        raise CascadeError("Causal group roles do not match request sibling roles")
    branch_requirements = [
        {
            "siblingRole": role,
            "controlDelta": branches[role]["controlDelta"],
            "evidenceUpdate": branches[role]["evidenceUpdate"],
            "availabilityTiming": branches[role].get("availabilityTiming"),
            "negativeControls": branches[role].get("negativeControls", []),
            "semanticAssertions": branches[role].get("semanticAssertions", []),
        }
        for role in roles
    ]
    prompt = {
        "task": "Compile target-free Voryn control-plan templates for every causal sibling.",
        "requestId": request["requestId"],
        "topicCard": topic,
        "scenarioContract": scenario,
        "trajectorySeed": trajectory,
        "pairSpec": pair,
        "requiredTopLevelKey": "planTemplates",
        "requiredSiblingRoles": list(roles),
        "requiredTemplateFields": sorted(TEMPLATE_FIELDS),
        "requiredControlProgramFields": sorted(CONTROL_FIELDS),
        "siblingContracts": branch_requirements,
        "requirements": [
            f"Return exactly {len(roles)} templates, one for each required sibling role.",
            "Use branchId or siblingRole to identify each template.",
            "All templates must share scenario, opening, cadence, and every pre-pivot control entry; only post-pivot controls may diverge.",
            "Every agent target must have one controlProgram entry carrying facts, uncertainty, constraints, next goal, and natural behavioral guidance.",
            "Apply the sibling's typed controlDelta only when its declared availability boundary is reached.",
            "Include exactly one endCall=true entry at the final target. It is a private model-selected tool action, never a deterministic goodbye phrase.",
            "Use non-identifying invented circumstances only. No placeholders, identity claims, contact data, credentials, or company scripts.",
            "Never include desired wording, dialogue, canonical response text, target transcript, target audio, or semantic-certification claims.",
        ],
    }
    response = planner.call(
        "You compile target-free Voryn causal planning templates. Reason silently and return raw JSON only. "
        "No markdown, dialogue transcript, target wording, target audio, or target labels.",
        canonical_json(prompt),
    )
    raw_templates = response.get("planTemplates") if set(response) == {"planTemplates"} else None
    if not isinstance(raw_templates, list) or len(raw_templates) != len(roles):
        raise CascadeError(f"Plan compiler must return exactly {len(roles)} planTemplates")
    templates: list[dict[str, Any]] = []
    for raw_template in raw_templates:
        if not isinstance(raw_template, dict):
            raise CascadeError("Plan templates must be objects")
        role = raw_template.get("branchId", raw_template.get("siblingRole"))
        if not isinstance(role, str):
            raise CascadeError("Every plan template requires branchId or siblingRole")
        templates.append(validate_template(raw_template, role))
    if len({item["branchId"] for item in templates}) != len(roles) or {
        item["branchId"] for item in templates
    } != set(roles):
        raise CascadeError("Plan templates do not contain the exact request sibling roles")
    return _bind_authoritative_shared_prefix(templates, roles, pair["pivotOrdinal"])


def _event_target_ordinal(event: dict[str, Any]) -> int | None:
    for field in EVENT_ORDINAL_FIELDS:
        value = event.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _events_for_target(trajectory: dict[str, Any], target_ordinal: int) -> list[dict[str, Any]]:
    events = trajectory.get("duplexEvents", [])
    if not isinstance(events, list) or not all(isinstance(item, dict) for item in events):
        raise CascadeError("Trajectory duplexEvents must be an object array")
    selected = [
        deepcopy(event)
        for event in events
        if _event_target_ordinal(event) in {None, target_ordinal}
    ]
    if len(selected) > 32:
        raise CascadeError("A target cannot bind more than 32 duplex events")
    return selected


def _revision_schedule_entry(
    trajectory: dict[str, Any], target_ordinal: int
) -> dict[str, Any] | None:
    schedule = trajectory.get("controlRevisionSchedule", [])
    if not isinstance(schedule, list):
        raise CascadeError("controlRevisionSchedule must be an array")
    exact = [
        item for item in schedule
        if isinstance(item, dict) and item.get("targetOrdinal") == target_ordinal
    ]
    if len(exact) > 1:
        raise CascadeError("controlRevisionSchedule repeats a target ordinal")
    return deepcopy(exact[0]) if exact else None


def _control_revision(
    branch: dict[str, Any], trajectory: dict[str, Any], target_ordinal: int,
    pivot_ordinal: int,
) -> tuple[int, dict[str, Any] | None]:
    schedule_entry = _revision_schedule_entry(trajectory, target_ordinal)
    revision: int | None = None
    if schedule_entry is not None:
        for field in ("controlRevision", "stateRevision", "revision"):
            candidate = schedule_entry.get(field)
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                revision = candidate
                break
    if target_ordinal >= pivot_ordinal:
        timing = branch.get("availabilityTiming", {})
        if isinstance(timing, dict):
            candidate = timing.get("controlRevision")
            if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                revision = max(revision or 0, candidate)
    return revision or target_ordinal, schedule_entry


def _semantic_sources(
    control: dict[str, Any], target_events: list[dict[str, Any]]
) -> list[str]:
    sources = ["state_reducer"]
    if control["policyConstraints"] or control["forbiddenClaims"]:
        sources.append("policy_agent")
    if control["toolResult"] is not None:
        sources.append("tool_result")
    if target_events:
        sources.append("interruption_controller")
    return sources


def _typed_control_entry(
    *,
    scenario_key: str,
    pair: dict[str, Any],
    branch: dict[str, Any],
    trajectory: dict[str, Any],
    template: dict[str, Any],
    control: dict[str, Any],
    base_state_hash: str,
) -> tuple[dict[str, Any], str]:
    target_ordinal = control["targetOrdinal"]
    pivot_ordinal = pair["pivotOrdinal"]
    delta = branch["controlDelta"]
    delta_effective = target_ordinal >= pivot_ordinal
    active_value = deepcopy(delta["to"] if delta_effective else delta["from"])
    revision, schedule_entry = _control_revision(
        branch, trajectory, target_ordinal, pivot_ordinal
    )
    target_events = _events_for_target(trajectory, target_ordinal)
    causal_state: dict[str, Any] = {
        "intent": template["coverage"]["intent"],
        "nextGoal": control["nextGoal"],
        "facts": deepcopy(control["facts"]),
        "commitments": deepcopy(control["commitments"]),
        "uncertainty": deepcopy(control["uncertainty"]),
        "policyConstraints": deepcopy(control["policyConstraints"]),
        "semanticContext": deepcopy(control["semanticContext"]),
        "guidance": control["guidance"],
        "endCallAuthorized": control["endCall"],
        "causalControl": {
            "axis": trajectory.get("causalAxis", "legacy_pair"),
            "interventionFamily": pair.get(
                "interventionFamily", trajectory.get("interventionFamily", "legacy_pair")
            ),
            "field": delta["field"],
            "activeValue": active_value,
            "effectiveAtThisTarget": delta_effective,
        },
    }
    if control["toolResult"] is not None:
        causal_state["toolResult"] = deepcopy(control["toolResult"])
    if delta_effective:
        causal_state["evidenceUpdate"] = deepcopy(branch["evidenceUpdate"])
    state_hash = content_hash({
        "baseStateHash": base_state_hash,
        "stateRevision": revision,
        "state": causal_state,
    })
    expiry_ms = 30000
    constraints = {
        "required_facts": deepcopy(control["requiredFacts"]),
        "forbidden_claims": deepcopy(control["forbiddenClaims"]),
        "must_ask": deepcopy(control["mustAsk"]),
        "must_not_request": [],
    }
    frame_mapping = {
        "schemaVersion": 1,
        "frameId": f"{scenario_key}-target-{target_ordinal}-revision-{revision}",
        "conversationId": scenario_key,
        "targetTurnId": target_ordinal,
        "stateRevision": revision,
        "baseStateHash": base_state_hash,
        "stateHash": state_hash,
        "semanticSources": _semantic_sources(control, target_events),
        "state": causal_state,
        "update": {
            "applyAt": "next_agent_turn_boundary",
            "expiresAtMs": expiry_ms,
            "reason": control["updateReason"],
            "availabilityTiming": deepcopy(branch.get("availabilityTiming", {})),
            "revisionScheduleEntry": schedule_entry,
            "controlDeltaHash": content_hash(delta),
        },
        "turnTaking": {
            "callerMayInterrupt": True,
            "interruptionPolicy": "yield_on_caller_speech",
            "duplexEvents": target_events,
        },
        "plan": {
            "schemaVersion": 1,
            "callId": scenario_key,
            "turnId": target_ordinal,
            "revision": revision,
            "contextHash": state_hash,
            "mode": "expressive",
            "intent": template["coverage"]["intent"],
            "dialogueAct": control["kind"],
            "entities": {
                "controlField": delta["field"],
                "controlValue": canonical_json(active_value),
            },
            "constraints": constraints,
            "delivery": {
                "language": "en-US",
                "register": template["coverage"]["speechStyle"],
                "assertiveness": template["dynamics"]["assertiveness"] / 100.0,
                "interruptibility": "yield_on_caller_speech",
                "max_duration_ms": expiry_ms,
                "speaking_rate_bucket": template["turnCadence"],
                "pause_density_bucket": template["responseLengthProfile"],
                "emphasis_targets": deepcopy(control["expectedEffects"][:8]),
            },
            "expiryMs": expiry_ms,
        },
    }
    try:
        frame = validate_control_frame_mapping(frame_mapping)
    except ContractError as error:
        raise CascadeError(
            f"Typed control frame failed for {scenario_key} target {target_ordinal}: {error}"
        ) from error
    # Dataclass wire helpers retain tuples in memory; canonical JSON normalizes
    # them to the arrays required by the mapping/schema contract.
    frame_wire = json.loads(canonical_json(frame.as_wire_dict()))
    validate_control_frame_mapping(frame_wire)
    assert_no_target_leak(frame_wire)
    compiled = deepcopy(control)
    compiled.update({
        "controlRevision": revision,
        "controlDeltaHash": content_hash(delta),
        "controlDeltaEffective": delta_effective,
        "controlFrame": frame_wire,
        "controlFrameHash": frame.frame_hash,
    })
    return compiled, state_hash


def materialize_typed_control_program(
    scenario_key: str,
    pair: dict[str, Any],
    branch: dict[str, Any],
    trajectory: dict[str, Any],
    template: dict[str, Any],
    shared_prefix: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    pivot_ordinal = pair["pivotOrdinal"]
    result: list[dict[str, Any]] = []
    previous_state_hash = pair["commonContextHash"]
    if shared_prefix is not None:
        shared_controls = deepcopy(shared_prefix.get("controlProgram"))
        if not isinstance(shared_controls, list):
            raise CascadeError("Shared-prefix plan lacks its typed control program")
        expected_prefix_ordinals = list(range(1, pivot_ordinal))
        actual_prefix_ordinals = [item.get("targetOrdinal") for item in shared_controls]
        if actual_prefix_ordinals != expected_prefix_ordinals:
            raise CascadeError(
                "Shared-prefix controls must cover every target ordinal strictly before the pivot"
            )
        result.extend(shared_controls)
        if shared_controls:
            terminal_frame = shared_controls[-1].get("controlFrame")
            if not isinstance(terminal_frame, dict) or not isinstance(
                terminal_frame.get("stateHash"), str
            ):
                raise CascadeError("Shared-prefix terminal control frame lacks stateHash")
            previous_state_hash = terminal_frame["stateHash"]
    branch_controls = [
        control for control in template["controlProgram"]
        if shared_prefix is None or control["targetOrdinal"] >= pivot_ordinal
    ]
    for control in branch_controls:
        compiled, previous_state_hash = _typed_control_entry(
            scenario_key=scenario_key,
            pair=pair,
            branch=branch,
            trajectory=trajectory,
            template=template,
            control=control,
            base_state_hash=previous_state_hash,
        )
        result.append(compiled)
    return result


def build_shared_prefix_plan(
    request: dict[str, Any],
    selection: dict[str, Any],
    trajectory: dict[str, Any],
    pair: dict[str, Any],
    templates: list[dict[str, Any]],
    *,
    scenario: dict[str, Any] | None = None,
) -> dict[str, Any]:
    roles = request_sibling_roles(request)
    by_role = {template["branchId"]: template for template in templates}
    leader = by_role[roles[0]]
    common_context = None
    if request.get("strategyVersion") == "semantic-control-v5":
        if scenario is None:
            raise CascadeError("V5 shared-prefix planning requires the complete scenario")
        common_context = build_common_context(request, selection, scenario, trajectory)
        if content_hash(common_context) != pair["commonContextHash"]:
            raise CascadeError("V5 compiler commonContext does not bind commonContextHash")
        prefix_id = content_hash({
            "schema": "personaplex.shared-prefix-identity.v1",
            "requestId": request["requestId"],
            "groupId": selection["groupId"],
            "commonContextHash": pair["commonContextHash"],
            "pivotTargetOrdinal": pair["pivotOrdinal"],
        })
    else:
        prefix_id = f"shared-prefix-{selection['groupId']}"
    prefix_seed = stable_integer({
        "request": request["requestId"],
        "group": selection["groupId"],
        "commonContextHash": pair["commonContextHash"],
    })
    pivot_ordinal = pair["pivotOrdinal"]
    prefix_controls = [
        deepcopy(item) for item in leader["controlProgram"]
        if item["targetOrdinal"] < pivot_ordinal
    ]
    branch = normalized_group_branches(pair)[0]
    shared_template = deepcopy(leader)
    shared_template["controlProgram"] = prefix_controls
    typed_controls: list[dict[str, Any]] = []
    previous_state_hash = pair["commonContextHash"]
    for control in prefix_controls:
        compiled, previous_state_hash = _typed_control_entry(
            scenario_key=prefix_id,
            pair=pair,
            branch=branch,
            trajectory=trajectory,
            template=shared_template,
            control=control,
            base_state_hash=previous_state_hash,
        )
        typed_controls.append(compiled)
    prefix_events = [
        deepcopy(event) for event in trajectory.get("duplexEvents", [])
        if _event_target_ordinal(event) is None
        or int(_event_target_ordinal(event)) <= pivot_ordinal
    ]
    artifact = {
        "schema": "personaplex.shared-prefix-plan.v1",
        "sharedPrefixId": prefix_id,
        "groupId": selection["groupId"],
        "sharedPrefixSeed": prefix_seed,
        "commonContextHash": pair["commonContextHash"],
        "sharedPrefixPolicy": pair.get(
            "sharedPrefixPolicy", "shared_plan_identical_before_pivot"
        ),
        "renderOnce": True,
        "nativeCodeReuseRequired": pair.get("sharedPrefixPolicy")
        == "native_code_identical_through_pivot",
        "pivotTargetOrdinal": pivot_ordinal,
        "reuseThroughBoundaryBeforeTargetOrdinal": pivot_ordinal,
        "commonTemplate": {
            field: deepcopy(leader[field]) for field in COMMON_TEMPLATE_FIELDS
        },
        "controlProgram": typed_controls,
        "duplexEvents": prefix_events,
    }
    if common_context is not None:
        artifact["commonContext"] = common_context
    assert_no_target_leak(artifact)
    artifact["sharedPrefixHash"] = content_hash(artifact)
    return artifact


def _shared_prefix_ref(shared_prefix: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": shared_prefix["schema"],
        "sharedPrefixId": shared_prefix["sharedPrefixId"],
        "sharedPrefixSeed": shared_prefix["sharedPrefixSeed"],
        "sharedPrefixHash": shared_prefix["sharedPrefixHash"],
        "renderOnce": True,
        "nativeCodeReuseRequired": shared_prefix["nativeCodeReuseRequired"],
        "pivotTargetOrdinal": shared_prefix["pivotTargetOrdinal"],
    }


def materialize_v8_plan(
    request: dict[str, Any],
    selection: dict[str, Any],
    trajectory: dict[str, Any],
    pair: dict[str, Any],
    template: dict[str, Any],
    caller_voice_id: str,
    target_voice_id: str,
    ordinal: int,
    shared_prefix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    branch_id = template["branchId"]
    branches = normalized_group_branches(pair)
    branch = next((item for item in branches if item["branchId"] == branch_id), None)
    if branch is None:
        raise CascadeError(f"Template role {branch_id} is absent from causal group")
    if shared_prefix is None:
        shared_prefix = build_shared_prefix_plan(
            request, selection, trajectory, pair, [template]
        )
    scenario_key = f"{selection['groupId']}-{branch_id}"
    typed_control_program = materialize_typed_control_program(
        scenario_key, pair, branch, trajectory, template, shared_prefix
    )
    evidence_status = branch["evidenceUpdate"].get("status")
    availability = evidence_status if isinstance(evidence_status, str) else (
        "ready" if branch_id == "available" else "failed"
        if branch_id == "constrained" else branch_id
    )
    prefix_ref = _shared_prefix_ref(shared_prefix)
    plan = {
        "schemaVersion": 7,
        "corpus": V8_CORPUS,
        "conversationOrdinal": ordinal,
        "scenarioKey": scenario_key,
        "topic": template["topic"],
        "topicFamily": template["topicFamily"],
        "topicSeedId": template["topicSeedId"],
        "contextLens": template["contextLens"],
        "conversationMode": template["conversationMode"],
        "lengthProfile": template["lengthProfile"],
        "turnCadence": template["turnCadence"],
        "responseLengthProfile": template["responseLengthProfile"],
        "callerVoiceReferenceId": caller_voice_id,
        "targetVoiceReferenceId": target_voice_id,
        "openingStyle": template["openingStyle"],
        "closingStyle": template["closingStyle"],
        "coverageProfile": template["coverageProfile"],
        "turns": template["turns"],
        "coverage": deepcopy(template["coverage"]),
        "controlProgram": typed_control_program,
        "dynamics": deepcopy(template["dynamics"]),
        "seed": stable_integer({
            "request": request["requestId"],
            "group": selection["groupId"],
            "branch": branch_id,
        }),
        "sharedPrefixRef": prefix_ref,
        "counterfactual": {
            "groupId": selection["groupId"],
            "branchId": branch_id,
            "siblingRole": branch_id,
            "changedField": branch["controlDelta"]["field"],
            "controlDelta": deepcopy(branch["controlDelta"]),
            "pivotTargetOrdinal": pair["pivotOrdinal"],
            "branchInstruction": template["branchInstruction"],
            "availability": availability,
        },
        "cascade": {
            "schema": "personaplex.diverse-cascade-voryn-bridge.v2",
            "requestId": request["requestId"],
            "topicId": selection["topicId"],
            "scenarioId": selection["scenarioId"],
            "trajectoryId": selection["trajectoryId"],
            "pairSpecHash": content_hash(pair),
            "selectionHash": selection["selectionHash"],
            "commonContextHash": pair["commonContextHash"],
            "sharedPrefixPolicy": pair.get("sharedPrefixPolicy"),
            "sharedPrefixRef": prefix_ref,
            "interventionFamily": pair.get(
                "interventionFamily", trajectory.get("interventionFamily")
            ),
            "typedPivot": deepcopy(pair.get("typedPivot", trajectory.get("typedPivot"))),
            "controlDelta": deepcopy(branch["controlDelta"]),
            "evidenceUpdate": deepcopy(branch["evidenceUpdate"]),
            "availabilityTiming": deepcopy(branch.get("availabilityTiming")),
            "controlRevisionSchedule": deepcopy(
                trajectory.get("controlRevisionSchedule", [])
            ),
            "duplexEvents": deepcopy(trajectory.get("duplexEvents", [])),
            "negativeControls": deepcopy(branch.get("negativeControls", [])),
            "semanticAssertions": deepcopy(branch.get("semanticAssertions", [])),
            "terminationContract": deepcopy(trajectory.get("terminationContract")),
        },
    }
    if request.get("strategyVersion") == "semantic-control-v5":
        common_context = shared_prefix.get("commonContext")
        if not isinstance(common_context, dict):
            raise CascadeError("V5 render plan lacks its complete commonContext")
        if content_hash(common_context) != pair["commonContextHash"]:
            raise CascadeError("V5 render-plan commonContext hash is stale")
        pivot_control = next(
            (
                item for item in typed_control_program
                if item["targetOrdinal"] == pair["pivotOrdinal"]
            ),
            None,
        )
        if pivot_control is None:
            raise CascadeError("V5 render plan lacks a pivot control frame")
        operator_field = branch["controlDelta"]["field"]
        operator_family = pair.get(
            "interventionFamily", trajectory.get("interventionFamily")
        )
        control_operator = {
            "id": content_hash({
                "family": operator_family,
                "changedPaths": [operator_field],
            }),
            "family": operator_family,
            "changedPaths": [operator_field],
        }
        voice_pair = {
            "id": content_hash({
                "caller": caller_voice_id,
                "agent": target_voice_id,
            }),
            "caller": caller_voice_id,
            "agent": target_voice_id,
        }
        group_template_id = content_hash({
            "schema": "personaplex.causal-group-template.v1",
            "requestId": request["requestId"],
            "groupId": selection["groupId"],
            "topicId": selection["topicId"],
            "scenarioId": selection["scenarioId"],
            "trajectoryId": selection["trajectoryId"],
            "commonTemplate": shared_prefix["commonTemplate"],
        })
        bridge = {
            "schema": V5_RENDER_BINDING_SCHEMA,
            "groupId": selection["groupId"],
            "siblingRole": branch_id,
            "scenarioKey": scenario_key,
            "sharedPrefixId": shared_prefix["sharedPrefixId"],
            "commonContextHash": pair["commonContextHash"],
            "commonContext": deepcopy(common_context),
            "premiseId": selection["scenarioId"],
            "templateId": group_template_id,
            "lineageIdentifiers": [
                request["requestId"],
                selection["topicId"],
                selection["scenarioId"],
                selection["trajectoryId"],
                selection["groupId"],
            ],
            "controlOperator": control_operator,
            "voicePair": voice_pair,
            "pivotTargetOrdinal": pair["pivotOrdinal"],
            "pivotControlBinding": {
                "frameHash": pivot_control["controlFrameHash"],
                "revision": pivot_control["controlRevision"],
                "availableBeforeTarget": True,
            },
            "nativeFrameContract": {
                "sampleRateHz": 24_000,
                "frameRateHz": 12.5,
                "frameDurationMs": 80,
                "samplesPerFrame": 1_920,
            },
        }
        assert_no_target_leak(bridge)
        plan["postRenderBridge"] = bridge
        plan["renderPlanId"] = content_hash(bridge)
    assert_no_target_leak(plan)
    return plan


def shared_prefix_output_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.shared-prefixes.jsonl")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cascade-root", required=True, type=Path)
    parser.add_argument("--voice-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--planner-endpoint",
        default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_ENDPOINT", ""),
    )
    parser.add_argument(
        "--planner-model",
        default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_MODEL", ""),
    )
    parser.add_argument(
        "--planner-api-key",
        default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_API_KEY", ""),
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 64:
        raise CascadeError("max-workers must be in [1, 64]")
    root = args.cascade_root.resolve()
    request = load_json(root / "request.json")
    validate_request(request)
    roles = request_sibling_roles(request)
    require_typed = request_requires_typed_trajectories(request)
    topics = load_jsonl(root / "topic_cards.jsonl")
    scenarios = load_jsonl(root / "scenario_contracts.jsonl")
    trajectories = load_jsonl(root / "trajectory_seeds.jsonl")
    selection = load_jsonl(root / "selected_trajectories.jsonl")
    pairs = load_jsonl(root / "counterfactual_pair_specs.jsonl")
    if not all((topics, scenarios, trajectories, selection, pairs)):
        raise CascadeError(
            "Cascade root must contain complete topics, scenarios, trajectories, selection, and groups"
        )
    topic_by_id = {item["topicId"]: item for item in topics}
    scenario_by_id = {item["scenarioId"]: item for item in scenarios}
    trajectory_by_id = {item["trajectoryId"]: item for item in trajectories}
    pair_by_group = {item["groupId"]: item for item in pairs}
    for topic in topics:
        validate_topic_card(topic, request["seedRevision"])
    for scenario in scenarios:
        validate_scenario_contract(scenario, set(topic_by_id))
    for trajectory in trajectories:
        validate_trajectory_seed(
            trajectory, set(scenario_by_id), require_typed=require_typed
        )
        if request.get("strategyVersion") == "semantic-control-v4":
            validate_v4_trajectory_seed(trajectory)
    for pair in pairs:
        trajectory = trajectory_by_id.get(pair.get("trajectoryId"))
        validate_pair_spec(
            pair,
            set(trajectory_by_id),
            request=request,
            trajectory=trajectory,
        )
        if request.get("strategyVersion") == "semantic-control-v4":
            validate_v4_pair_spec(pair)
    if {row["groupId"] for row in selection} != set(pair_by_group):
        raise CascadeError("Selected group IDs and group specification IDs must match exactly")
    voice_ids = approved_voice_ids(args.voice_manifest)
    planner = JsonOnlyPlanner(
        PlannerConfig(args.planner_endpoint, args.planner_model, args.planner_api_key)
    )
    existing = load_jsonl(args.output) if args.resume else []
    prefix_path = shared_prefix_output_path(args.output)
    existing_prefixes = load_jsonl(prefix_path) if args.resume else []
    expected_role_set = set(roles)
    existing_by_group: dict[str, set[str]] = {}
    for row in existing:
        counterfactual = row.get("counterfactual", {})
        existing_by_group.setdefault(str(counterfactual.get("groupId")), set()).add(
            str(counterfactual.get("branchId"))
        )
    for group_id, present_roles in existing_by_group.items():
        if present_roles != expected_role_set:
            raise CascadeError(
                f"Resume output has an incomplete sibling group {group_id}: {sorted(present_roles)}"
            )
    existing_prefix_groups = {row.get("groupId") for row in existing_prefixes}
    if set(existing_by_group) != existing_prefix_groups:
        raise CascadeError("Resume output and shared-prefix sidecar cover different groups")
    work = [row for row in selection if row["groupId"] not in existing_by_group]

    def compile_group(row: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        pair = pair_by_group[row["groupId"]]
        trajectory = trajectory_by_id[row["trajectoryId"]]
        scenario = scenario_by_id[row["scenarioId"]]
        topic = topic_by_id[row["topicId"]]
        templates = create_templates(
            planner, request, topic, scenario, trajectory, pair
        )
        shared_prefix = build_shared_prefix_plan(
            request, row, trajectory, pair, templates, scenario=scenario
        )
        caller_voice_id, target_voice_id = assign_voice_pair(
            request, row["groupId"], voice_ids
        )
        plans = [
            materialize_v8_plan(
                request,
                row,
                trajectory,
                pair,
                template,
                caller_voice_id,
                target_voice_id,
                0,
                shared_prefix,
            )
            for template in templates
        ]
        return plans, shared_prefix

    generated: list[dict[str, Any]] = []
    generated_prefixes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(compile_group, row): row for row in work}
        for future in as_completed(futures):
            try:
                plans, shared_prefix = future.result()
                generated.extend(plans)
                generated_prefixes.append(shared_prefix)
            except Exception as error:
                group_id = futures[future]["groupId"]
                raise CascadeError(
                    f"Voryn plan compilation failed for {group_id}: {error}"
                ) from error
    plans = existing + generated
    prefixes = existing_prefixes + generated_prefixes
    actual_by_group: dict[str, set[str]] = {}
    for row in plans:
        group_id = row["counterfactual"]["groupId"]
        role = row["counterfactual"]["branchId"]
        if role in actual_by_group.setdefault(group_id, set()):
            raise CascadeError("Voryn plan output contains duplicate group/sibling entries")
        actual_by_group[group_id].add(role)
    if set(actual_by_group) != {row["groupId"] for row in selection}:
        raise CascadeError("Voryn plan output does not cover every selected group")
    if any(group_roles != expected_role_set for group_roles in actual_by_group.values()):
        raise CascadeError("Voryn plan output does not contain every requested sibling role")
    expected = len(selection) * len(roles)
    if len(plans) != expected:
        raise CascadeError(f"Voryn plan has {len(plans)} siblings; expected {expected}")
    if len(prefixes) != len(selection) or {
        prefix["groupId"] for prefix in prefixes
    } != {row["groupId"] for row in selection}:
        raise CascadeError("Shared-prefix sidecar must contain exactly one plan per group")
    role_order = {role: index for index, role in enumerate(roles)}
    plans.sort(key=lambda row: (
        row["counterfactual"]["groupId"],
        role_order[row["counterfactual"]["branchId"]],
    ))
    prefixes.sort(key=lambda row: row["groupId"])
    for ordinal, plan in enumerate(plans, start=1):
        plan["conversationOrdinal"] = ordinal
    write_jsonl(args.output, plans)
    write_jsonl(prefix_path, prefixes)
    write_json(args.output.with_suffix(".manifest.json"), {
        "schema": "personaplex.diverse-cascade-voryn-plan-manifest.v2",
        "requestId": request["requestId"],
        "requestHash": content_hash(request),
        "planHash": content_hash(plans),
        "sharedPrefixPlanHash": content_hash(prefixes),
        "sharedPrefixArtifact": prefix_path.name,
        "conversations": len(plans),
        "counterfactualGroups": len(selection),
        "siblingsPerGroup": len(roles),
        "renderer": request["renderer"],
        "asr": request["asr"],
        "admission": "voryn_plan_only_not_source_certified",
    })
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CascadeError as error:
        print(f"cascade Voryn bridge error: {error}", file=sys.stderr)
        raise SystemExit(2)
