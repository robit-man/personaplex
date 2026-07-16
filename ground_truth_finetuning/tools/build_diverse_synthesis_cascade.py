#!/usr/bin/env python3
"""Build one stage of the agent-operable diverse controlled-synthesis cascade."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.diverse_cascade import (
    CascadeError,
    JsonOnlyPlanner,
    PlannerConfig,
    load_json,
    load_jsonl,
    parallel_map,
    plan_pair,
    plan_scenarios,
    plan_topics,
    plan_trajectories,
    select_trajectories,
    validate_request,
    write_jsonl,
    write_run_manifest,
)


def require_existing(path: Path, label: str) -> list[dict]:
    rows = load_jsonl(path)
    if not rows:
        raise CascadeError(f"{label} is required before this stage: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("topics", "scenarios", "trajectories", "selection", "pairs", "all"))
    parser.add_argument("--planner-endpoint", default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_ENDPOINT", ""))
    parser.add_argument("--planner-model", default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_MODEL", ""))
    parser.add_argument("--planner-api-key", default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_API_KEY", ""))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 64:
        raise CascadeError("max-workers must be in [1, 64]")

    request = load_json(args.request)
    validate_request(request)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    request_copy = root / "request.json"
    if request_copy.exists() and not args.resume:
        raise CascadeError("Output root already has a request; use --resume or choose a new root")
    if not request_copy.exists():
        request_copy.write_text(args.request.read_text(encoding="utf-8"), encoding="utf-8")

    paths = {
        "topics": root / "topic_cards.jsonl",
        "scenarios": root / "scenario_contracts.jsonl",
        "trajectories": root / "trajectory_seeds.jsonl",
        "selection": root / "selected_trajectories.jsonl",
        "pairs": root / "counterfactual_pair_specs.jsonl",
    }
    artifacts = {name: load_jsonl(path) for name, path in paths.items()}
    needs_planner = args.stage in {"topics", "scenarios", "trajectories", "pairs", "all"}
    planner = JsonOnlyPlanner(PlannerConfig(args.planner_endpoint, args.planner_model, args.planner_api_key)) if needs_planner else None

    if args.stage in {"topics", "all"}:
        if artifacts["topics"]:
            if not args.resume:
                raise CascadeError("topic_cards.jsonl already exists; use --resume")
        else:
            artifacts["topics"] = plan_topics(planner, request)
            write_jsonl(paths["topics"], artifacts["topics"])
            write_run_manifest(root, request, "topics", artifacts)

    if args.stage in {"scenarios", "all"}:
        topics = require_existing(paths["topics"], "Topic cards")
        if artifacts["scenarios"]:
            if not args.resume:
                raise CascadeError("scenario_contracts.jsonl already exists; use --resume")
        else:
            generated = parallel_map(topics, lambda topic: plan_scenarios(planner, topic, request), args.max_workers)
            artifacts["scenarios"] = sorted((row for batch in generated for row in batch), key=lambda row: row["scenarioId"])
            write_jsonl(paths["scenarios"], artifacts["scenarios"])
            write_run_manifest(root, request, "scenarios", artifacts)

    if args.stage in {"trajectories", "all"}:
        scenarios = require_existing(paths["scenarios"], "Scenario contracts")
        if artifacts["trajectories"]:
            if not args.resume:
                raise CascadeError("trajectory_seeds.jsonl already exists; use --resume")
        else:
            generated = parallel_map(scenarios, lambda scenario: plan_trajectories(planner, scenario, request), args.max_workers)
            artifacts["trajectories"] = sorted((row for batch in generated for row in batch), key=lambda row: row["trajectoryId"])
            write_jsonl(paths["trajectories"], artifacts["trajectories"])
            write_run_manifest(root, request, "trajectories", artifacts)

    if args.stage in {"selection", "all"}:
        topics = require_existing(paths["topics"], "Topic cards")
        scenarios = require_existing(paths["scenarios"], "Scenario contracts")
        trajectories = require_existing(paths["trajectories"], "Trajectory seeds")
        if artifacts["selection"]:
            if not args.resume:
                raise CascadeError("selected_trajectories.jsonl already exists; use --resume")
        else:
            artifacts["selection"] = select_trajectories(request, topics, scenarios, trajectories)
            write_jsonl(paths["selection"], artifacts["selection"])
            write_run_manifest(root, request, "selection", artifacts)

    if args.stage in {"pairs", "all"}:
        scenarios = require_existing(paths["scenarios"], "Scenario contracts")
        trajectories = require_existing(paths["trajectories"], "Trajectory seeds")
        selection = require_existing(paths["selection"], "Selected trajectories")
        if artifacts["pairs"]:
            if not args.resume:
                raise CascadeError("counterfactual_pair_specs.jsonl already exists; use --resume")
        else:
            scenario_by_id = {item["scenarioId"]: item for item in scenarios}
            trajectory_by_id = {item["trajectoryId"]: item for item in trajectories}
            generated = parallel_map(
                selection,
                lambda row: plan_pair(planner, request, row, scenario_by_id[row["scenarioId"]], trajectory_by_id[row["trajectoryId"]]),
                args.max_workers,
            )
            artifacts["pairs"] = sorted(generated, key=lambda row: row["groupId"])
            write_jsonl(paths["pairs"], artifacts["pairs"])
            write_run_manifest(root, request, "pairs", artifacts)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CascadeError as error:
        print(f"cascade error: {error}", file=sys.stderr)
        raise SystemExit(2)
