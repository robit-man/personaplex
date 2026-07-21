#!/usr/bin/env python3
"""Materialize a request-defined PersonaPlex planning lattice and Voryn plan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.diverse_cascade import (
    CascadeError,
    PlannerConfig,
    cascade_config_hash,
    collect_schema_hashes,
    content_hash,
    load_bound_seed_catalog,
    load_json,
    load_jsonl,
    planner_config_hash,
    request_selection_counts,
    request_sibling_roles,
    validate_request,
    write_json,
)

LIVE_PLAN = Path("/srv/personaplex_workspace/ground_truth_runs/personaplex-1000-plan.v8-counterfactual-diverse-v6.jsonl")


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if result.returncode:
        raise CascadeError(f"Cascade command failed ({result.returncode}): {' '.join(command)}")


def count_unique(rows: list[dict[str, Any]], key: str) -> int:
    return len({row.get(key) for row in rows})


def requested_counts(request: dict[str, Any]) -> dict[str, int]:
    coverage = request["coverageTarget"]
    primary, reserve = request_selection_counts(request)
    siblings = len(request_sibling_roles(request))
    return {
        "topics": coverage["candidateTopics"],
        "scenarios": coverage["candidateTopics"] * coverage["scenariosPerTopic"],
        "trajectoryLeaves": coverage["candidateTopics"] * coverage["scenariosPerTopic"] * coverage["trajectorySeedsPerScenario"],
        "primaryGroups": primary,
        "reserveGroups": reserve,
        "counterfactualGroups": primary,
        "vorynBranches": primary * siblings,
    }


def validate_materialization(root: Path, output_plan: Path, request: dict[str, Any]) -> dict[str, Any]:
    topics = load_jsonl(root / "topic_cards.jsonl")
    scenarios = load_jsonl(root / "scenario_contracts.jsonl")
    trajectories = load_jsonl(root / "trajectory_seeds.jsonl")
    primary_path = root / "primary_trajectories.jsonl"
    primary = load_jsonl(primary_path) if primary_path.exists() else load_jsonl(root / "selected_trajectories.jsonl")
    reserves = load_jsonl(root / "reserve_trajectories.jsonl")
    selection = load_jsonl(root / "selected_trajectories.jsonl")
    groups = load_jsonl(root / "counterfactual_pair_specs.jsonl")
    plans = load_jsonl(output_plan)
    expected = requested_counts(request)
    counts = {
        "topics": len(topics),
        "scenarios": len(scenarios),
        "trajectoryLeaves": len(trajectories),
        "primaryGroups": len(primary),
        "reserveGroups": len(reserves),
        "counterfactualGroups": len(groups),
        "vorynBranches": len(plans),
    }
    if counts != expected:
        raise CascadeError(f"Cascade cardinality mismatch: got {counts!r}, expected {expected!r}")
    if len(selection) != expected["primaryGroups"]:
        raise CascadeError("Active selection does not contain the request-defined primary group count")
    for rows, key, label in (
        (topics, "topicId", "Topic cards"),
        (scenarios, "scenarioId", "Scenario contracts"),
        (trajectories, "trajectoryId", "Trajectory leaves"),
        (primary + reserves, "groupId", "Primary/reserve selections"),
        (selection, "groupId", "Active selections"),
        (groups, "groupId", "Counterfactual groups"),
    ):
        if count_unique(rows, key) != len(rows):
            raise CascadeError(f"{label} are not unique")
    active_group_ids = {row["groupId"] for row in selection}
    if {row["groupId"] for row in groups} != active_group_ids:
        raise CascadeError("Counterfactual groups do not exactly match the active selection")
    expected_roles = set(request_sibling_roles(request))
    group_branches: dict[str, set[str]] = {}
    for row in plans:
        counterfactual = row.get("counterfactual")
        if not isinstance(counterfactual, dict):
            raise CascadeError("Voryn branch lacks typed counterfactual metadata")
        group_id = counterfactual.get("groupId")
        branch_id = counterfactual.get("branchId", counterfactual.get("siblingRole"))
        group_branches.setdefault(group_id, set()).add(branch_id)
    if set(group_branches) != active_group_ids or any(branches != expected_roles for branches in group_branches.values()):
        raise CascadeError("Voryn plan does not contain the request-defined sibling roles for every active group")
    return {
        "counts": counts,
        "hashes": {
            "topics": content_hash(topics),
            "scenarios": content_hash(scenarios),
            "trajectoryLeaves": content_hash(trajectories),
            "primarySelection": content_hash(primary),
            "reserveSelection": content_hash(reserves),
            "activeSelection": content_hash(selection),
            "counterfactualGroups": content_hash(groups),
            "vorynPlan": content_hash(plans),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--voice-manifest", required=True, type=Path)
    parser.add_argument("--voryn-plan", required=True, type=Path)
    parser.add_argument("--planner-endpoint", default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_ENDPOINT", ""))
    parser.add_argument("--planner-model", default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_MODEL", ""))
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--rejected-groups", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-live-plan-replacement", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 64:
        raise CascadeError("max-workers must be in [1, 64]")
    if not args.planner_endpoint or not args.planner_model:
        raise CascadeError("planner-endpoint and planner-model are required")

    request_path = args.request.resolve()
    root = args.output_root.resolve()
    voice_manifest = args.voice_manifest.resolve()
    output_plan = args.voryn_plan.resolve()
    request = load_json(request_path)
    validate_request(request)
    _, catalog_hash = load_bound_seed_catalog(request, request_path, REPOSITORY_ROOT)
    if content_hash(load_json(voice_manifest)) != request["allowedVoicesManifest"]:
        raise CascadeError("Voice manifest hash does not match request.allowedVoicesManifest")
    if output_plan == LIVE_PLAN and not args.allow_live_plan_replacement:
        raise CascadeError("Refusing to overwrite the active V8 plan; write a new plan and promote it explicitly after review")
    if output_plan.exists() and not args.resume:
        raise CascadeError("Voryn plan output already exists; use --resume or choose a new output path")

    planner_args = [
        "--planner-endpoint", args.planner_endpoint,
        "--planner-model", args.planner_model,
        "--max-workers", str(args.max_workers),
    ]
    if args.resume:
        planner_args.append("--resume")
    build_args = list(planner_args)
    if args.rejected_groups is not None:
        build_args.extend(["--rejected-groups", str(args.rejected_groups.resolve())])
    build = REPOSITORY_ROOT / "ground_truth_finetuning" / "tools" / "build_diverse_synthesis_cascade.py"
    validate = REPOSITORY_ROOT / "ground_truth_finetuning" / "tools" / "validate_diverse_synthesis_cascade.py"
    compile_plan = REPOSITORY_ROOT / "ground_truth_finetuning" / "tools" / "compile_diverse_cascade_voryn_plan.py"
    run([sys.executable, str(build), "--request", str(request_path), "--output-root", str(root), "--stage", "all", *build_args])
    run([sys.executable, str(validate), "--input-root", str(root), "--report", str(root / "cascade_validation.json"), "--require-pairs"])
    run([sys.executable, str(compile_plan), "--cascade-root", str(root), "--voice-manifest", str(voice_manifest), "--output", str(output_plan), *planner_args])

    result = validate_materialization(root, output_plan, request)
    schema_hashes = collect_schema_hashes(REPOSITORY_ROOT)
    planner_config = PlannerConfig(args.planner_endpoint, args.planner_model, "")
    manifest = {
        "schema": "personaplex.diverse-cascade-pre-generation-manifest.v2",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "requestId": request["requestId"],
        "requestHash": content_hash(request),
        "catalogHash": catalog_hash,
        "plannerHash": planner_config_hash(planner_config),
        "configHash": cascade_config_hash(request, args.max_workers),
        "schemaHashes": schema_hashes,
        "schemaHash": content_hash(schema_hashes),
        "requestedCounts": requested_counts(request),
        "voiceManifest": str(voice_manifest),
        "voiceManifestHash": request["allowedVoicesManifest"],
        "cascadeRoot": str(root),
        "vorynPlan": str(output_plan),
        **result,
        "admission": "planning_complete_not_audio_rendered_not_source_certified",
        "promotion": "requires explicit runtime-contract plan-path change after review",
    }
    write_json(root / "pre_generation_manifest.json", manifest)
    print(manifest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CascadeError as error:
        print(f"cascade pre-generation error: {error}", file=sys.stderr)
        raise SystemExit(2)
