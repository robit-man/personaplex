#!/usr/bin/env python3
"""Build the efficient semantic-control-v5 trajectory candidate/selection/expansion stages."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.efficient_v5_fanout import (  # noqa: E402
    CANDIDATES_FILENAME,
    PRIMARY_FILENAME,
    RESERVE_FILENAME,
    RoundRobinJsonSchemaPlanner,
    FanoutError,
    expand_selected_candidates,
    generate_compact_candidates,
    read_json,
    read_jsonl,
    select_compact_candidates,
    validate_v5_inputs,
    write_combined_manifest,
)


def bind_input(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FanoutError(f"required input is missing: {source}")
    if destination.exists():
        if destination.resolve() != source and destination.read_bytes() != source.read_bytes():
            raise FanoutError(f"output root contains a conflicting {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--stage", required=True,
        choices=("candidates", "selection", "expand", "all"),
    )
    parser.add_argument(
        "--planner-endpoint",
        default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_ENDPOINT", ""),
        help="Comma-separated three-lane OpenAI-compatible chat-completion endpoints",
    )
    parser.add_argument(
        "--planner-model",
        default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_MODEL", ""),
    )
    parser.add_argument(
        "--planner-api-key",
        default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_API_KEY", ""),
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.max_workers <= 3:
        raise FanoutError("--max-workers must be in [1,3]")
    if not 1 <= args.max_attempts <= 12:
        raise FanoutError("--max-attempts must be in [1,12]")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    bind_input(args.request, output_root / "request.json")
    bind_input(input_root / "topic_cards.jsonl", output_root / "topic_cards.jsonl")
    bind_input(input_root / "scenario_contracts.jsonl", output_root / "scenario_contracts.jsonl")
    request = read_json(output_root / "request.json")
    topics = read_jsonl(output_root / "topic_cards.jsonl")
    scenarios = read_jsonl(output_root / "scenario_contracts.jsonl")
    validate_v5_inputs(request, topics, scenarios)

    planner = None
    if args.stage in {"candidates", "expand", "all"}:
        endpoints = [item for item in args.planner_endpoint.split(",") if item.strip()]
        model = args.planner_model or str((request.get("planner") or {}).get("model") or "")
        planner = RoundRobinJsonSchemaPlanner(endpoints, model, args.planner_api_key)

    if args.stage in {"candidates", "all"}:
        if not args.resume and (
            (output_root / CANDIDATES_FILENAME).exists()
            or any((output_root / ".efficient_v5_checkpoints" / "candidates").glob("*.json"))
        ):
            raise FanoutError("candidate outputs/checkpoints already exist; use --resume")
        generate_compact_candidates(
            request=request, topics=topics, scenarios=scenarios,
            output_root=output_root, planner=planner,
            max_workers=args.max_workers, max_attempts=args.max_attempts,
        )

    if args.stage in {"selection", "all"}:
        candidates = read_jsonl(output_root / CANDIDATES_FILENAME)
        if not args.resume and any(
            (output_root / name).exists()
            for name in (PRIMARY_FILENAME, RESERVE_FILENAME, "selected_trajectories.jsonl")
        ):
            raise FanoutError("selection outputs already exist; use --resume")
        select_compact_candidates(
            request=request, topics=topics, scenarios=scenarios,
            candidates=candidates, output_root=output_root,
        )

    if args.stage in {"expand", "all"}:
        candidates = read_jsonl(output_root / CANDIDATES_FILENAME)
        primary = read_jsonl(output_root / PRIMARY_FILENAME)
        reserve = read_jsonl(output_root / RESERVE_FILENAME)
        if not args.resume and (output_root / "trajectory_seeds.jsonl").exists():
            raise FanoutError("trajectory_seeds.jsonl already exists; use --resume")
        expand_selected_candidates(
            request=request, topics=topics, scenarios=scenarios,
            candidates=candidates, primary=primary, reserve=reserve,
            output_root=output_root, planner=planner,
            max_workers=args.max_workers, max_attempts=args.max_attempts,
        )

    write_combined_manifest(output_root, request)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FanoutError as error:
        print(f"efficient v5 fan-out failed: {error}", file=sys.stderr)
        raise SystemExit(2)
