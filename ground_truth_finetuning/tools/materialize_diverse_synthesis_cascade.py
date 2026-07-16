#!/usr/bin/env python3
"""Materialize the canonical 50x20x10 PersonaPlex planning lattice.

This creates planning-only artifacts and a Voryn branch plan. It never renders audio,
creates target dialogue, modifies the active synthesis plan, or certifies source data.
"""

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
    content_hash,
    load_json,
    load_jsonl,
    validate_request,
    write_json,
)

STANDARD_FANOUT = {
    "candidateTopics": 50,
    "scenariosPerTopic": 20,
    "trajectorySeedsPerScenario": 10,
    "selectedCounterfactualGroups": 500,
    "branchesPerGroup": 2,
}
LIVE_PLAN = Path("/srv/personaplex_workspace/ground_truth_runs/personaplex-1000-plan.v8-counterfactual-diverse-v6.jsonl")


def run(command: list[str]) -> None:
    result = subprocess.run(command, cwd=REPOSITORY_ROOT, check=False)
    if result.returncode:
        raise CascadeError(f"Cascade command failed ({result.returncode}): {' '.join(command)}")


def require_standard_fanout(request: dict[str, Any]) -> None:
    coverage = request["coverageTarget"]
    if coverage != STANDARD_FANOUT:
        raise CascadeError(
            "This materializer is pinned to the canonical 50 topics x 20 scenarios x 10 trajectories "
            "candidate lattice, selected as 500 paired groups / 1,000 branches. "
            f"Received {coverage!r}"
        )


def count_unique(rows: list[dict[str, Any]], key: str) -> int:
    return len({row.get(key) for row in rows})


def validate_materialization(root: Path, output_plan: Path, request: dict[str, Any]) -> dict[str, Any]:
    topics = load_jsonl(root / "topic_cards.jsonl")
    scenarios = load_jsonl(root / "scenario_contracts.jsonl")
    trajectories = load_jsonl(root / "trajectory_seeds.jsonl")
    selection = load_jsonl(root / "selected_trajectories.jsonl")
    pairs = load_jsonl(root / "counterfactual_pair_specs.jsonl")
    plans = load_jsonl(output_plan)
    expected = STANDARD_FANOUT
    counts = {
        "topics": len(topics),
        "scenarios": len(scenarios),
        "trajectoryLeaves": len(trajectories),
        "selectedGroups": len(selection),
        "counterfactualPairs": len(pairs),
        "vorynBranches": len(plans),
    }
    required_counts = {
        "topics": expected["candidateTopics"],
        "scenarios": expected["candidateTopics"] * expected["scenariosPerTopic"],
        "trajectoryLeaves": expected["candidateTopics"] * expected["scenariosPerTopic"] * expected["trajectorySeedsPerScenario"],
        "selectedGroups": expected["selectedCounterfactualGroups"],
        "counterfactualPairs": expected["selectedCounterfactualGroups"],
        "vorynBranches": expected["selectedCounterfactualGroups"] * expected["branchesPerGroup"],
    }
    if counts != required_counts:
        raise CascadeError(f"Cascade cardinality mismatch: got {counts!r}, expected {required_counts!r}")
    if count_unique(topics, "topicId") != counts["topics"]:
        raise CascadeError("Topic cards are not unique")
    if count_unique(scenarios, "scenarioId") != counts["scenarios"]:
        raise CascadeError("Scenario contracts are not unique")
    if count_unique(trajectories, "trajectoryId") != counts["trajectoryLeaves"]:
        raise CascadeError("Trajectory leaves are not unique")
    if count_unique(selection, "groupId") != counts["selectedGroups"]:
        raise CascadeError("Selected counterfactual groups are not unique")
    group_branches = {(row.get("counterfactual") or {}).get("groupId"): set() for row in plans}
    for row in plans:
        counterfactual = row.get("counterfactual") or {}
        group_branches.setdefault(counterfactual.get("groupId"), set()).add(counterfactual.get("branchId"))
    if len(group_branches) != expected["selectedCounterfactualGroups"] or any(branches != {"available", "constrained"} for branches in group_branches.values()):
        raise CascadeError("Voryn plan does not contain exactly one available/constrained pair per selected group")
    return {
        "counts": counts,
        "hashes": {
            "topics": content_hash(topics),
            "scenarios": content_hash(scenarios),
            "trajectoryLeaves": content_hash(trajectories),
            "selection": content_hash(selection),
            "counterfactualPairs": content_hash(pairs),
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
    require_standard_fanout(request)
    if content_hash(load_json(voice_manifest)) != request["allowedVoicesManifest"]:
        raise CascadeError("Voice manifest hash does not match request.allowedVoicesManifest")
    if output_plan == LIVE_PLAN and not args.allow_live_plan_replacement:
        raise CascadeError("Refusing to overwrite the active V8 plan; write a new plan and promote it explicitly after review")
    if output_plan.exists() and not args.resume:
        raise CascadeError("Voryn plan output already exists; use --resume or choose a new output path")

    planner_args = ["--planner-endpoint", args.planner_endpoint, "--planner-model", args.planner_model, "--max-workers", str(args.max_workers)]
    if args.resume:
        planner_args.append("--resume")
    build = REPOSITORY_ROOT / "ground_truth_finetuning" / "tools" / "build_diverse_synthesis_cascade.py"
    validate = REPOSITORY_ROOT / "ground_truth_finetuning" / "tools" / "validate_diverse_synthesis_cascade.py"
    compile_plan = REPOSITORY_ROOT / "ground_truth_finetuning" / "tools" / "compile_diverse_cascade_voryn_plan.py"
    run([sys.executable, str(build), "--request", str(request_path), "--output-root", str(root), "--stage", "all", *planner_args])
    run([sys.executable, str(validate), "--input-root", str(root), "--report", str(root / "cascade_validation.json"), "--require-pairs"])
    run([sys.executable, str(compile_plan), "--cascade-root", str(root), "--voice-manifest", str(voice_manifest), "--output", str(output_plan), *planner_args])

    result = validate_materialization(root, output_plan, request)
    manifest = {
        "schema": "personaplex.diverse-cascade-pre-generation-manifest.v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "requestId": request["requestId"],
        "requestHash": content_hash(request),
        "standardFanout": STANDARD_FANOUT,
        "planner": {"endpoint": args.planner_endpoint, "model": args.planner_model, "maxWorkers": args.max_workers},
        "voiceManifest": str(voice_manifest),
        "voiceManifestHash": request["allowedVoicesManifest"],
        "cascadeRoot": str(root),
        "vorynPlan": str(output_plan),
        **result,
        "admission": "planning_complete_not_audio_rendered_not_source_certified",
        "promotion": "requires explicit runtime-contract plan-path change after review; never replace a live plan implicitly",
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
