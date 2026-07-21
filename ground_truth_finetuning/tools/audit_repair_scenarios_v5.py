#!/usr/bin/env python3
"""Audit or selectively repair completed PersonaPlex v5 scenario contracts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.diverse_cascade import (
    JsonOnlyPlanner,
    PlannerConfig,
    load_jsonl,
)
from ground_truth_finetuning.training.scenario_scrutiny import (
    AuthenticScenarioJudge,
    ScenarioScrutinyError,
    scrutinize_scenarios,
)
from ground_truth_finetuning.training.scenario_adjudication_v5 import (
    AdjudicatedScenarioJudge,
    AuthenticDecomposedScenarioAdjudicator,
)


def _env_api_key(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "")
        if value:
            return value
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument(
        "--judge-endpoint",
        default=os.environ.get("PERSONAPLEX_SCENARIO_JUDGE_ENDPOINT", ""),
        help="Comma-separated independent OpenAI-compatible chat-completions endpoints.",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("PERSONAPLEX_SCENARIO_JUDGE_MODEL", ""),
    )
    parser.add_argument(
        "--judge-api-key",
        default=_env_api_key("PERSONAPLEX_SCENARIO_JUDGE_API_KEY", "OPENROUTER_API_KEY"),
    )
    parser.add_argument("--judge-timeout-seconds", type=int, default=180)
    parser.add_argument("--judge-max-tokens", type=int, default=4096)
    parser.add_argument("--judge-max-attempts", type=int, default=6)
    parser.add_argument(
        "--blueprint-sets",
        type=Path,
        help="Exact admitted blueprint-set JSONL; defaults to the canonical file under --root.",
    )
    parser.add_argument(
        "--secondary-judge-endpoint",
        default=os.environ.get("PERSONAPLEX_SCENARIO_SECONDARY_JUDGE_ENDPOINT", ""),
    )
    parser.add_argument(
        "--secondary-judge-model",
        default=os.environ.get("PERSONAPLEX_SCENARIO_SECONDARY_JUDGE_MODEL", ""),
    )
    parser.add_argument(
        "--secondary-judge-api-key",
        default=_env_api_key(
            "PERSONAPLEX_SCENARIO_SECONDARY_JUDGE_API_KEY", "OPENROUTER_API_KEY"
        ),
    )
    parser.add_argument(
        "--adjudicator-endpoint",
        default=os.environ.get("PERSONAPLEX_SCENARIO_ADJUDICATOR_ENDPOINT", ""),
    )
    parser.add_argument(
        "--adjudicator-model",
        default=os.environ.get("PERSONAPLEX_SCENARIO_ADJUDICATOR_MODEL", ""),
    )
    parser.add_argument(
        "--adjudicator-api-key",
        default=os.environ.get("PERSONAPLEX_SCENARIO_ADJUDICATOR_API_KEY", ""),
    )
    parser.add_argument(
        "--planner-endpoint",
        default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_ENDPOINT", ""),
        help="Comma-separated authentic generation endpoints used only by --repair.",
    )
    parser.add_argument(
        "--planner-model",
        default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_MODEL", ""),
    )
    parser.add_argument(
        "--planner-api-key",
        default=_env_api_key("PERSONAPLEX_CASCADE_PLANNER_API_KEY", "OPENROUTER_API_KEY"),
    )
    parser.add_argument("--planner-timeout-seconds", type=int, default=180)
    parser.add_argument("--planner-max-tokens", type=int, default=4096)
    parser.add_argument("--planner-max-attempts", type=int, default=6)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-repair-rounds", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--repair", action="store_true")
    mode.add_argument("--dry-audit", action="store_true")
    args = parser.parse_args()

    semantic_judge_values = (
        args.judge_endpoint,
        args.judge_model,
        args.secondary_judge_endpoint,
        args.secondary_judge_model,
        args.adjudicator_endpoint,
        args.adjudicator_model,
    )
    if not all(semantic_judge_values):
        parser.error(
            "primary, secondary, and adjudicator endpoints/models are all required; "
            "single-model semantic admission is disabled"
        )
    if args.repair and (not args.planner_endpoint or not args.planner_model):
        parser.error("--repair requires --planner-endpoint and --planner-model")

    blueprint_sets_path = args.blueprint_sets
    if blueprint_sets_path is None:
        candidates = (
            args.root / "scenario_blueprint_sets.jsonl",
            args.root / "scenario_blueprint_sets.final.smoke.jsonl",
        )
        blueprint_sets_path = next((path for path in candidates if path.is_file()), None)
    if blueprint_sets_path is None or not blueprint_sets_path.is_file():
        parser.error("an admitted --blueprint-sets JSONL is required for semantic adjudication")
    blueprint_sets = load_jsonl(blueprint_sets_path)

    try:
        primary_judge = AuthenticScenarioJudge(PlannerConfig(
            endpoint=args.judge_endpoint,
            model=args.judge_model,
            api_key=args.judge_api_key,
            timeout_seconds=args.judge_timeout_seconds,
            max_tokens=args.judge_max_tokens,
            temperature=0.0,
        ), max_attempts=args.judge_max_attempts)
        secondary_judge = AuthenticScenarioJudge(PlannerConfig(
            endpoint=args.secondary_judge_endpoint,
            model=args.secondary_judge_model,
            api_key=args.secondary_judge_api_key,
            timeout_seconds=args.judge_timeout_seconds,
            max_tokens=args.judge_max_tokens,
            temperature=0.0,
        ), max_attempts=args.judge_max_attempts)
        adjudicator = AuthenticDecomposedScenarioAdjudicator(PlannerConfig(
            endpoint=args.adjudicator_endpoint,
            model=args.adjudicator_model,
            api_key=args.adjudicator_api_key,
            timeout_seconds=args.judge_timeout_seconds,
            max_tokens=args.judge_max_tokens,
            temperature=0.0,
        ),
            checkpoint_root=args.root / ".scenario_scrutiny" / "adjudicator_subcalls",
            max_attempts=args.judge_max_attempts,
        )
        judge = AdjudicatedScenarioJudge(
            primary_judge,
            secondary_judge,
            adjudicator,
            trace_root=args.root / ".scenario_scrutiny" / "adjudication",
            blueprint_sets=blueprint_sets,
        )
        planner = None
        if args.repair:
            planner = JsonOnlyPlanner(PlannerConfig(
                endpoint=args.planner_endpoint,
                model=args.planner_model,
                api_key=args.planner_api_key,
                timeout_seconds=args.planner_timeout_seconds,
                max_tokens=args.planner_max_tokens,
            ))
        report = scrutinize_scenarios(
            args.root,
            judge,
            request_path=args.request,
            planner=planner,
            repair=args.repair,
            dry_audit=args.dry_audit or not args.repair,
            resume=args.resume,
            max_workers=args.max_workers,
            max_repair_rounds=args.max_repair_rounds,
            repair_max_attempts=args.planner_max_attempts,
        )
    except ScenarioScrutinyError as error:
        print(f"scenario scrutiny failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps({
        "status": report["status"],
        "reportId": report["reportId"],
        "reportPath": report["reportPath"],
        "topicCount": report["topicCount"],
        "scenarioCount": report["scenarioCount"],
        "repairTransactions": len(report["repairTransactions"]),
    }, sort_keys=True))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
