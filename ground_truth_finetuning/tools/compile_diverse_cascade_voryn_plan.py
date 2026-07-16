#!/usr/bin/env python3
"""Compile selected cascade pairs into Voryn V8 lane-plan entries.

The model creates target-free conversation/control blueprints. This bridge pins branch
lineage, causal pivot, approved voice pair, and V8 transport fields deterministically;
Voryn later realizes dialogue and performs independent audio/semantic certification.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.diverse_cascade import (  # noqa: E402
    CascadeError,
    JsonOnlyPlanner,
    PlannerConfig,
    assert_no_target_leak,
    canonical_json,
    content_hash,
    load_json,
    load_jsonl,
    validate_pair_spec,
    validate_request,
    validate_scenario_contract,
    validate_topic_card,
    validate_trajectory_seed,
    write_json,
    write_jsonl,
)


V8_CORPUS = "personaplex-synthetic-counterfactual-v9-cascade"
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


def stable_integer(value: Any) -> int:
    return int(sha256(canonical_json(value).encode("utf-8")).hexdigest()[:8], 16)


def approved_voice_ids(manifest_path: Path) -> list[str]:
    manifest = load_json(manifest_path)
    references = manifest.get("references")
    if not isinstance(references, list):
        raise CascadeError("Voice manifest must contain references")
    voice_ids = sorted({str(item.get("id") or "") for item in references if isinstance(item, dict) and item.get("id")})
    if len(voice_ids) < 2:
        raise CascadeError("V8 plan compilation requires at least two approved voice IDs")
    return voice_ids


def assign_voice_pair(request: dict[str, Any], group_id: str, voice_ids: list[str]) -> tuple[str, str]:
    seed = stable_integer({"seedRevision": request["seedRevision"], "groupId": group_id, "voiceManifest": request["allowedVoicesManifest"]})
    caller_index = seed % len(voice_ids)
    target_index = (seed // len(voice_ids) + 1) % len(voice_ids)
    if target_index == caller_index:
        target_index = (target_index + 1) % len(voice_ids)
    return voice_ids[caller_index], voice_ids[target_index]


def require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CascadeError(f"V8 template {field} must be nonempty text")
    return value


def require_string_array(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise CascadeError(f"V8 template {field} must be a string array")
    return value


def validate_control_program(program: Any, branch_id: str, turns: int) -> list[dict[str, Any]]:
    if not isinstance(program, list) or not program:
        raise CascadeError("V8 template controlProgram must be nonempty")
    ordinals: set[int] = set()
    validated: list[dict[str, Any]] = []
    for item in program:
        if not isinstance(item, dict):
            raise CascadeError("V8 controlProgram entries must be objects")
        missing = CONTROL_FIELDS - set(item)
        if missing:
            raise CascadeError(f"V8 controlProgram is missing {sorted(missing)}")
        ordinal = item["targetOrdinal"]
        if not isinstance(ordinal, int) or ordinal < 1 or ordinal > max(1, turns // 2):
            raise CascadeError("V8 control targetOrdinal is outside the agent-turn range")
        if ordinal in ordinals:
            raise CascadeError("V8 control program has duplicate target ordinals")
        ordinals.add(ordinal)
        for key in ("id", "source", "kind", "nextGoal", "guidance", "updateReason"):
            require_text(item[key], f"controlProgram.{key}")
        for key in ("facts", "commitments", "uncertainty", "policyConstraints", "requiredFacts", "forbiddenClaims", "mustAsk", "expectedEffects"):
            require_string_array(item[key], f"controlProgram.{key}")
        if not isinstance(item["semanticContext"], dict) or not isinstance(item["endCall"], bool):
            raise CascadeError("V8 control semanticContext/endCall are invalid")
        if item["toolResult"] is not None and not isinstance(item["toolResult"], dict):
            raise CascadeError("V8 control toolResult must be object or null")
        if "canonical" in canonical_json(item).lower() or "target text" in canonical_json(item).lower():
            raise CascadeError("V8 control program includes prohibited target-label language")
        validated.append(item)
    if not any(bool(item["endCall"]) for item in validated):
        raise CascadeError("V8 control program must include one model-selected end_call stage")
    if sum(bool(item["endCall"]) for item in validated) != 1:
        raise CascadeError("V8 control program must have exactly one end_call stage")
    return sorted(validated, key=lambda item: item["targetOrdinal"])


def validate_template(template: dict[str, Any], branch_id: str) -> dict[str, Any]:
    missing = TEMPLATE_FIELDS - set(template)
    if missing:
        raise CascadeError(f"V8 template is missing {sorted(missing)}")
    if template["branchId"] != branch_id:
        raise CascadeError("V8 template branchId does not match causal pair")
    for field in (
        "topic", "topicFamily", "topicSeedId", "contextLens", "conversationMode",
        "lengthProfile", "turnCadence", "responseLengthProfile", "openingStyle",
        "closingStyle", "coverageProfile", "branchInstruction",
    ):
        require_text(template[field], field)
    turns = template["turns"]
    if not isinstance(turns, int) or not 4 <= turns <= 48 or turns % 2:
        raise CascadeError("V8 template turns must be an even integer in [4, 48]")
    if not isinstance(template["coverage"], dict) or not isinstance(template["dynamics"], dict):
        raise CascadeError("V8 template coverage and dynamics must be objects")
    for key in ("intent", "trajectory", "interactionClass", "speechStyle", "turnPattern", "nextGoal"):
        require_text(template["coverage"].get(key), f"coverage.{key}")
    for key in ("controlSources", "requiredStateFields"):
        require_string_array(template["coverage"].get(key), f"coverage.{key}")
    if not isinstance(template["coverage"].get("requireControlForAllTargets"), bool):
        raise CascadeError("coverage.requireControlForAllTargets must be boolean")
    for key in ("assertiveness", "skepticism", "compliance", "resistance", "recovery", "hesitation", "pace", "interruption"):
        value = template["dynamics"].get(key)
        if not isinstance(value, int) or not 0 <= value <= 100:
            raise CascadeError(f"dynamics.{key} must be an integer in [0, 100]")
    template["controlProgram"] = validate_control_program(template["controlProgram"], branch_id, turns)
    assert_no_target_leak(template)
    return template


def create_templates(planner: JsonOnlyPlanner, request: dict[str, Any], topic: dict[str, Any], scenario: dict[str, Any], trajectory: dict[str, Any], pair: dict[str, Any]) -> list[dict[str, Any]]:
    branches = {item["branchId"]: item for item in pair["branches"]}
    prompt = {
        "task": "Compile two target-free Voryn V8 control-plan templates from one selected causal pair.",
        "requestId": request["requestId"],
        "topicCard": topic,
        "scenarioContract": scenario,
        "trajectorySeed": trajectory,
        "pairSpec": pair,
        "requiredTopLevelKey": "planTemplates",
        "requiredTemplateFields": sorted(TEMPLATE_FIELDS),
        "requiredControlProgramFields": sorted(CONTROL_FIELDS),
        "requirements": [
            "Return exactly two templates, branchId available and constrained.",
            "Both templates share scenario and trajectory context through the pair pivot, then apply only their branch control/evidence change.",
            "Control guidance must describe facts, uncertainty, constraints, next goals, and natural behavior, never a desired exact utterance.",
            "Every controlProgram must include exactly one final endCall=true stage. It is a private model action, not a deterministic goodbye phrase.",
            "Use non-identifying invented circumstances only. No placeholders, identity claims, contact data, credentials, or company scripts.",
            "Do not include canonical response text, target audio, transcript labels, or semantic-certification claims.",
            f"Available branch delta: {branches['available']['controlDelta']}; evidence: {branches['available']['evidenceUpdate']}.",
            f"Constrained branch delta: {branches['constrained']['controlDelta']}; evidence: {branches['constrained']['evidenceUpdate']}.",
        ],
    }
    response = planner.call(
        "You compile target-free Voryn V8 planning templates. Reason silently and return raw JSON only. "
        "No markdown, no dialogue transcript, and no target labels.",
        canonical_json(prompt),
    )
    if set(response) != {"planTemplates"} or not isinstance(response["planTemplates"], list) or len(response["planTemplates"]) != 2:
        raise CascadeError("Plan compiler must return exactly two planTemplates")
    templates = [validate_template(item, str(item.get("branchId") or "")) for item in response["planTemplates"] if isinstance(item, dict)]
    if {item["branchId"] for item in templates} != {"available", "constrained"}:
        raise CascadeError("Plan templates must contain available and constrained branches")
    return templates


def materialize_v8_plan(request: dict[str, Any], selection: dict[str, Any], trajectory: dict[str, Any], pair: dict[str, Any], template: dict[str, Any], caller_voice_id: str, target_voice_id: str, ordinal: int) -> dict[str, Any]:
    branch_id = template["branchId"]
    branch = next(item for item in pair["branches"] if item["branchId"] == branch_id)
    availability = "ready" if branch_id == "available" else "failed"
    changed_field = branch["controlDelta"]["field"]
    return {
        "schemaVersion": 7,
        "corpus": V8_CORPUS,
        "conversationOrdinal": ordinal,
        "scenarioKey": f"{selection['groupId']}-{branch_id}",
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
        "coverage": template["coverage"],
        "controlProgram": template["controlProgram"],
        "dynamics": template["dynamics"],
        "seed": stable_integer({"request": request["requestId"], "group": selection["groupId"], "branch": branch_id}),
        "counterfactual": {
            "groupId": selection["groupId"],
            "branchId": branch_id,
            "changedField": changed_field,
            "pivotTargetOrdinal": pair["pivotOrdinal"],
            "branchInstruction": template["branchInstruction"],
            "availability": availability,
        },
        "cascade": {
            "schema": "personaplex.diverse-cascade-voryn-bridge.v1",
            "requestId": request["requestId"],
            "topicId": selection["topicId"],
            "scenarioId": selection["scenarioId"],
            "trajectoryId": selection["trajectoryId"],
            "pairSpecHash": content_hash(pair),
            "selectionHash": selection["selectionHash"],
            "evidenceUpdate": branch["evidenceUpdate"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cascade-root", required=True, type=Path)
    parser.add_argument("--voice-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--planner-endpoint", default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_ENDPOINT", ""))
    parser.add_argument("--planner-model", default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_MODEL", ""))
    parser.add_argument("--planner-api-key", default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_API_KEY", ""))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 64:
        raise CascadeError("max-workers must be in [1, 64]")
    root = args.cascade_root.resolve()
    request = load_json(root / "request.json")
    validate_request(request)
    topics = load_jsonl(root / "topic_cards.jsonl")
    scenarios = load_jsonl(root / "scenario_contracts.jsonl")
    trajectories = load_jsonl(root / "trajectory_seeds.jsonl")
    selection = load_jsonl(root / "selected_trajectories.jsonl")
    pairs = load_jsonl(root / "counterfactual_pair_specs.jsonl")
    if not all((topics, scenarios, trajectories, selection, pairs)):
        raise CascadeError("Cascade root must contain complete topics, scenarios, trajectories, selection, and pairs")
    topic_by_id = {item["topicId"]: item for item in topics}
    scenario_by_id = {item["scenarioId"]: item for item in scenarios}
    trajectory_by_id = {item["trajectoryId"]: item for item in trajectories}
    pair_by_group = {item["groupId"]: item for item in pairs}
    for topic in topics:
        validate_topic_card(topic, request["seedRevision"])
    for scenario in scenarios:
        validate_scenario_contract(scenario, set(topic_by_id))
    for trajectory in trajectories:
        validate_trajectory_seed(trajectory, set(scenario_by_id))
    for pair in pairs:
        validate_pair_spec(pair, set(trajectory_by_id))
    if {row["groupId"] for row in selection} != set(pair_by_group):
        raise CascadeError("Selected group IDs and pair specification group IDs must match exactly")
    voice_ids = approved_voice_ids(args.voice_manifest)
    planner = JsonOnlyPlanner(PlannerConfig(args.planner_endpoint, args.planner_model, args.planner_api_key))
    existing = load_jsonl(args.output) if args.resume else []
    existing_groups = {row.get("counterfactual", {}).get("groupId") for row in existing}
    work = [row for row in selection if row["groupId"] not in existing_groups]

    def compile_group(row: dict[str, Any]) -> list[dict[str, Any]]:
        pair = pair_by_group[row["groupId"]]
        trajectory = trajectory_by_id[row["trajectoryId"]]
        scenario = scenario_by_id[row["scenarioId"]]
        topic = topic_by_id[row["topicId"]]
        templates = create_templates(planner, request, topic, scenario, trajectory, pair)
        caller_voice_id, target_voice_id = assign_voice_pair(request, row["groupId"], voice_ids)
        return [materialize_v8_plan(request, row, trajectory, pair, template, caller_voice_id, target_voice_id, 0) for template in templates]

    generated: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(compile_group, row): row for row in work}
        for future in as_completed(futures):
            try:
                generated.extend(future.result())
            except Exception as error:
                raise CascadeError(f"Voryn plan compilation failed for {futures[future]['groupId']}: {error}") from error
    plans = existing + generated
    if len({(row["counterfactual"]["groupId"], row["counterfactual"]["branchId"]) for row in plans}) != len(plans):
        raise CascadeError("Voryn plan output contains duplicate group/branch entries")
    expected = len(selection) * 2
    if len(plans) != expected:
        raise CascadeError(f"Voryn plan has {len(plans)} branches; expected {expected}")
    plans.sort(key=lambda row: (row["counterfactual"]["groupId"], row["counterfactual"]["branchId"]))
    for ordinal, plan in enumerate(plans, start=1):
        plan["conversationOrdinal"] = ordinal
    write_jsonl(args.output, plans)
    write_json(args.output.with_suffix(".manifest.json"), {
        "schema": "personaplex.diverse-cascade-voryn-plan-manifest.v1",
        "requestId": request["requestId"],
        "requestHash": content_hash(request),
        "planHash": content_hash(plans),
        "conversations": len(plans),
        "counterfactualGroups": len(selection),
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
