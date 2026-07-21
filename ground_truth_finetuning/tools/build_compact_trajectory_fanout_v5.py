#!/usr/bin/env python3
"""Build the production compact semantic-control-v5 trajectory fan-out."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.compact_trajectory_fanout import (  # noqa: E402
    CANDIDATES_FILENAME,
    CHECKPOINT_ROOT,
    MAX_PROTOCOL_ATTEMPTS,
    MAX_STAGE_ATTEMPTS,
    PRIMARY_FILENAME,
    PRODUCTION_SCENARIO_COUNT,
    RESERVE_FILENAME,
    SELECTED_FILENAME,
    TRAJECTORIES_FILENAME,
    FanoutError,
    ThreeEndpointJsonSchemaClient,
    expand_selected_candidates,
    generate_compact_candidates,
    read_json,
    read_jsonl,
    select_compact_candidates,
    validate_v5_inputs,
    write_combined_manifest,
)
from ground_truth_finetuning.training.strict_schema_transport import (  # noqa: E402
    DEFAULT_RETRY_BASE_SECONDS,
    DEFAULT_RETRY_MAX_SECONDS,
)


def _generative_api_key() -> str:
    return os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get(
        "PERSONAPLEX_CASCADE_PLANNER_API_KEY", ""
    )


def bind_input(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FanoutError(f"required input is missing: {source}")
    if destination.exists():
        if destination.read_bytes() != source.read_bytes():
            raise FanoutError(f"output root contains conflicting {destination.name}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())


def _assert_fresh(output_root: Path, stage: str) -> None:
    stage_paths = {
        "a": [
            output_root / CANDIDATES_FILENAME,
            output_root / CHECKPOINT_ROOT / "stage_a_candidates",
            output_root / CHECKPOINT_ROOT / "stage_a_scenarios",
        ],
        "b": [
            output_root / PRIMARY_FILENAME,
            output_root / RESERVE_FILENAME,
            output_root / SELECTED_FILENAME,
        ],
        "c": [
            output_root / TRAJECTORIES_FILENAME,
            output_root / CHECKPOINT_ROOT / "stage_c_expansions",
        ],
    }
    if any(path.exists() for path in stage_paths[stage]):
        raise FanoutError(f"Stage {stage.upper()} outputs already exist; use --resume")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("a", "b", "c", "all", "candidates", "selection", "expand"),
    )
    parser.add_argument(
        "--planner-endpoints",
        "--planner-endpoint",
        dest="planner_endpoints",
        default=os.environ.get(
            "PERSONAPLEX_CASCADE_PLANNER_ENDPOINTS",
            os.environ.get("PERSONAPLEX_CASCADE_PLANNER_ENDPOINT", ""),
        ),
        help="One to three comma-separated OpenAI-compatible chat-completion endpoints",
    )
    parser.add_argument(
        "--planner-model",
        default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_MODEL", ""),
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument(
        "--protocol-attempts",
        type=int,
        default=int(os.environ.get("PERSONAPLEX_GENERATIVE_TRANSPORT_ATTEMPTS", "6")),
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=float(
            os.environ.get(
                "PERSONAPLEX_GENERATIVE_RETRY_BASE_SECONDS",
                str(DEFAULT_RETRY_BASE_SECONDS),
            )
        ),
    )
    parser.add_argument(
        "--retry-max-seconds",
        type=float,
        default=float(
            os.environ.get(
                "PERSONAPLEX_GENERATIVE_RETRY_MAX_SECONDS",
                str(DEFAULT_RETRY_MAX_SECONDS),
            )
        ),
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if not 1 <= args.max_workers <= 3:
        raise FanoutError("--max-workers must be in [1,3]")
    if not 1 <= args.max_attempts <= MAX_STAGE_ATTEMPTS:
        raise FanoutError(f"--max-attempts must be in [1,{MAX_STAGE_ATTEMPTS}]")
    if not 2 <= args.protocol_attempts <= MAX_PROTOCOL_ATTEMPTS:
        raise FanoutError(
            f"--protocol-attempts must be in [2,{MAX_PROTOCOL_ATTEMPTS}]"
        )
    stage = {"candidates": "a", "selection": "b", "expand": "c"}.get(
        args.stage, args.stage
    )
    stages = ("a", "b", "c") if stage == "all" else (stage,)

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    bind_input(args.request, output_root / "request.json")
    bind_input(input_root / "topic_cards.jsonl", output_root / "topic_cards.jsonl")
    bind_input(
        input_root / "scenario_contracts.jsonl", output_root / "scenario_contracts.jsonl"
    )
    request = read_json(output_root / "request.json")
    topics = read_jsonl(output_root / "topic_cards.jsonl")
    scenarios = read_jsonl(output_root / "scenario_contracts.jsonl")
    validate_v5_inputs(
        request,
        topics,
        scenarios,
        required_scenario_count=PRODUCTION_SCENARIO_COUNT,
        require_production_counts=True,
    )

    if not args.resume:
        for selected_stage in stages:
            _assert_fresh(output_root, selected_stage)

    planner = None
    if "a" in stages or "c" in stages:
        model = args.planner_model or str((request.get("planner") or {}).get("model") or "")
        planner = ThreeEndpointJsonSchemaClient(
            args.planner_endpoints,
            model,
            _generative_api_key(),
            protocol_attempts=args.protocol_attempts,
            retry_base_seconds=args.retry_base_seconds,
            retry_max_seconds=args.retry_max_seconds,
        )

    if "a" in stages:
        generate_compact_candidates(
            request=request,
            topics=topics,
            scenarios=scenarios,
            output_root=output_root,
            planner=planner,
            max_workers=args.max_workers,
            max_attempts=args.max_attempts,
            required_scenario_count=PRODUCTION_SCENARIO_COUNT,
        )

    if "b" in stages:
        candidates = read_jsonl(output_root / CANDIDATES_FILENAME)
        select_compact_candidates(
            request=request,
            topics=topics,
            scenarios=scenarios,
            candidates=candidates,
            output_root=output_root,
        )

    if "c" in stages:
        candidates = read_jsonl(output_root / CANDIDATES_FILENAME)
        primary = read_jsonl(output_root / PRIMARY_FILENAME)
        reserve = read_jsonl(output_root / RESERVE_FILENAME)
        expand_selected_candidates(
            request=request,
            topics=topics,
            scenarios=scenarios,
            candidates=candidates,
            primary=primary,
            reserve=reserve,
            output_root=output_root,
            planner=planner,
            max_workers=args.max_workers,
            max_attempts=args.max_attempts,
        )

    manifest = write_combined_manifest(output_root, request)
    print(manifest["manifestHash"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FanoutError as error:
        print(f"compact trajectory fan-out failed: {error}", file=sys.stderr)
        raise SystemExit(2)
