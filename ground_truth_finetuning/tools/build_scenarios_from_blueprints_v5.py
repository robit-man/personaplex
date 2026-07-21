#!/usr/bin/env python3
"""Build joint v5 scenario blueprints and immutable bound scenario expansions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.scenario_blueprint_v5 import (  # noqa: E402
    AdjudicatedWholeBlueprintJudge,
    AuthenticWholeBlueprintJudge,
    MAX_WORKERS,
    ModelTransportUnavailable,
    ScenarioBlueprintError,
    ThreeEndpointStrictSchemaPlanner,
    generate_scenarios,
    load_blueprint_sets,
    load_blueprint_scrutinies,
    prepare_output_root,
)
from ground_truth_finetuning.training.scenario_blueprint_repair_v5 import (  # noqa: E402
    admit_blueprints,
)
from ground_truth_finetuning.training.scenario_taxonomy_admission_v5 import (  # noqa: E402
    AdjudicatedTaxonomyJudge,
    AuthenticTaxonomyJudge,
)
from ground_truth_finetuning.training.strict_schema_transport import (  # noqa: E402
    DEFAULT_RETRY_BASE_SECONDS,
    DEFAULT_RETRY_MAX_SECONDS,
    DEFAULT_TRANSPORT_ATTEMPTS,
    OPENROUTER_NEMOTRON_ULTRA_MODEL,
    is_openrouter_endpoint,
    normalize_chat_completion_endpoints,
)


def _env_api_key(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "")
        if value:
            return value
    return ""


def _parse_endpoints(value: str, label: str, *, local_only: bool = False) -> tuple[str, ...]:
    try:
        endpoints = normalize_chat_completion_endpoints(value)
    except ValueError as error:
        raise ScenarioBlueprintError(f"{label}: {error}") from error
    if local_only and any(is_openrouter_endpoint(endpoint) for endpoint in endpoints):
        raise ScenarioBlueprintError(f"{label} must remain on independent local endpoints")
    return endpoints


LOCAL_NEMOTRON_SUPER_MODEL = "nemotron-3-super:120b"
LOCAL_NEMOTRON_SECONDARY_MODEL = "nemotron-3-nano:30b"


def _env_enabled(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ScenarioBlueprintError(f"{name} must be a boolean")


def _local_fallback_options(
    prefix: str,
    *,
    expected_model: str,
    bind_fallback: bool = False,
) -> dict[str, object]:
    endpoint_value = os.environ.get(f"{prefix}_ENDPOINT", "").strip()
    model = os.environ.get(f"{prefix}_MODEL", "").strip()
    if not endpoint_value and not model:
        return {}
    if not endpoint_value or not model:
        raise ScenarioBlueprintError(
            f"{prefix}_ENDPOINT and {prefix}_MODEL must be configured together"
        )
    if model != expected_model:
        raise ScenarioBlueprintError(f"{prefix}_MODEL must be {expected_model}")
    return {
        "fallback_endpoints": _parse_endpoints(
            endpoint_value, f"{prefix} endpoints", local_only=True
        ),
        "fallback_model": model,
        "prefer_fallback": _env_enabled(f"{prefix}_PREFER", default=True),
        "bind_fallback": bind_fallback,
    }


def _large_proposer_config() -> tuple[tuple[str, ...], str]:
    endpoint_value = os.environ.get("PERSONAPLEX_LARGE_PROPOSER_ENDPOINT", "")
    model = os.environ.get("PERSONAPLEX_LARGE_PROPOSER_MODEL", "").strip()
    if not endpoint_value.strip():
        raise ScenarioBlueprintError(
            "PERSONAPLEX_LARGE_PROPOSER_ENDPOINT is required for blueprint adjudication"
        )
    if not model:
        raise ScenarioBlueprintError(
            "PERSONAPLEX_LARGE_PROPOSER_MODEL is required for blueprint adjudication"
        )
    endpoints = _parse_endpoints(endpoint_value, "large proposer endpoints")
    if any(not is_openrouter_endpoint(endpoint) for endpoint in endpoints):
        raise ScenarioBlueprintError(
            "large proposer endpoints must be credential-free OpenRouter URLs"
        )
    if model != OPENROUTER_NEMOTRON_ULTRA_MODEL:
        raise ScenarioBlueprintError(
            "large proposer model must be " + OPENROUTER_NEMOTRON_ULTRA_MODEL
        )
    return endpoints, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("blueprints", "scenarios", "all"))
    parser.add_argument(
        "--planner-endpoint",
        default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_ENDPOINT", ""),
        help="One to three comma-separated OpenAI-compatible chat-completion endpoints",
    )
    parser.add_argument(
        "--planner-model",
        default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_MODEL", ""),
    )
    parser.add_argument(
        "--taxonomy-repair-endpoint",
        default=os.environ.get("PERSONAPLEX_TAXONOMY_REPAIR_ENDPOINT", ""),
        help="One to three OpenAI-compatible endpoints for targeted taxonomy repair",
    )
    parser.add_argument(
        "--taxonomy-repair-model",
        default=os.environ.get("PERSONAPLEX_TAXONOMY_REPAIR_MODEL", ""),
    )
    parser.add_argument(
        "--judge-endpoint",
        default=os.environ.get("PERSONAPLEX_BLUEPRINT_JUDGE_ENDPOINT", ""),
        help="One to three independent strict-schema judge endpoints",
    )
    parser.add_argument(
        "--judge-model",
        default=os.environ.get("PERSONAPLEX_BLUEPRINT_JUDGE_MODEL", ""),
    )
    parser.add_argument(
        "--taxonomy-verifier-endpoint",
        default=os.environ.get("PERSONAPLEX_TAXONOMY_VERIFIER_ENDPOINT", ""),
    )
    parser.add_argument(
        "--taxonomy-verifier-model",
        default=os.environ.get("PERSONAPLEX_TAXONOMY_VERIFIER_MODEL", ""),
    )
    parser.add_argument(
        "--transport-attempts",
        type=int,
        default=int(
            os.environ.get(
                "PERSONAPLEX_GENERATIVE_TRANSPORT_ATTEMPTS",
                str(DEFAULT_TRANSPORT_ATTEMPTS),
            )
        ),
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
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--judge-workers", type=int, default=1)
    parser.add_argument(
        "--taxonomy-verifier-workers",
        type=int,
        default=int(os.environ.get("PERSONAPLEX_TAXONOMY_VERIFIER_WORKERS", "1")),
    )
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--max-repair-cycles", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault(
        "PERSONAPLEX_MODEL_TRACE_ROOT",
        str(args.output_root / ".scenario_blueprint_v5" / "model_attempts"),
    )

    if not 1 <= args.max_workers <= MAX_WORKERS:
        raise ScenarioBlueprintError("--max-workers must be in [1,3]")
    if not 1 <= args.max_attempts <= 12:
        raise ScenarioBlueprintError("--max-attempts must be in [1,12]")
    if not 1 <= args.judge_workers <= MAX_WORKERS:
        raise ScenarioBlueprintError("--judge-workers must be in [1,3]")
    if not 1 <= args.taxonomy_verifier_workers <= MAX_WORKERS:
        raise ScenarioBlueprintError("--taxonomy-verifier-workers must be in [1,3]")
    if not 0 <= args.max_repair_cycles <= 12:
        raise ScenarioBlueprintError("--max-repair-cycles must be in [0,12]")

    request, topics = prepare_output_root(
        request_path=args.request,
        input_root=args.input_root,
        output_root=args.output_root,
        resume=args.resume,
    )
    endpoints = _parse_endpoints(args.planner_endpoint, "planner endpoints")
    configured_model = str((request.get("planner") or {}).get("model") or "")
    transport_options = {
        "transport_attempts": args.transport_attempts,
        "retry_base_seconds": args.retry_base_seconds,
        "retry_max_seconds": args.retry_max_seconds,
    }
    local_super_runtime = _local_fallback_options(
        "PERSONAPLEX_LOCAL_SUPER",
        expected_model=LOCAL_NEMOTRON_SUPER_MODEL,
    )
    local_super_judge_runtime = _local_fallback_options(
        "PERSONAPLEX_LOCAL_SUPER",
        expected_model=LOCAL_NEMOTRON_SUPER_MODEL,
        bind_fallback=True,
    )
    local_secondary_judge_runtime = _local_fallback_options(
        "PERSONAPLEX_LOCAL_SECONDARY",
        expected_model=LOCAL_NEMOTRON_SECONDARY_MODEL,
        bind_fallback=True,
    )
    generative_api_key = _env_api_key("OPENROUTER_API_KEY")
    planner = ThreeEndpointStrictSchemaPlanner(
        endpoints,
        args.planner_model or configured_model,
        generative_api_key,
        **transport_options,
        **local_super_runtime,
    )
    taxonomy_repair_endpoints = (
        _parse_endpoints(args.taxonomy_repair_endpoint, "taxonomy repair endpoints")
        if args.taxonomy_repair_endpoint.strip()
        else ()
    )
    taxonomy_repair_planner = None
    if taxonomy_repair_endpoints:
        taxonomy_repair_planner = ThreeEndpointStrictSchemaPlanner(
            taxonomy_repair_endpoints,
            args.taxonomy_repair_model or args.planner_model or configured_model,
            _env_api_key("OPENROUTER_API_KEY"),
            **transport_options,
            **local_super_runtime,
        )
    blueprint_sets = None
    blueprint_scrutinies = None
    if args.stage in {"blueprints", "all"}:
        judge_endpoints = _parse_endpoints(
            args.judge_endpoint, "Nemotron Super proposer endpoints"
        )
        nemotron_super_proposer_model = ThreeEndpointStrictSchemaPlanner(
            judge_endpoints,
            args.judge_model,
            _env_api_key("PERSONAPLEX_BLUEPRINT_JUDGE_API_KEY", "OPENROUTER_API_KEY"),
            temperature=0.0,
            **transport_options,
            **local_super_judge_runtime,
        )
        large_proposer_endpoints, large_proposer_model_name = (
            _large_proposer_config()
        )
        large_proposer_model = ThreeEndpointStrictSchemaPlanner(
            large_proposer_endpoints,
            large_proposer_model_name,
            _env_api_key("OPENROUTER_API_KEY"),
            temperature=0.0,
            **transport_options,
            **local_secondary_judge_runtime,
        )
        gemma_verifier_model = ThreeEndpointStrictSchemaPlanner(
            _parse_endpoints(
                args.taxonomy_verifier_endpoint,
                "Gemma verifier endpoints",
                local_only=True,
            ),
            args.taxonomy_verifier_model,
            _env_api_key("PERSONAPLEX_TAXONOMY_VERIFIER_API_KEY"),
            temperature=0.0,
            **transport_options,
        )
        judge = AdjudicatedWholeBlueprintJudge(
            AuthenticWholeBlueprintJudge(nemotron_super_proposer_model),
            AuthenticWholeBlueprintJudge(large_proposer_model),
            gemma_verifier_model,
            checkpoint_root=(
                args.output_root
                / ".scenario_blueprint_v5"
                / "checkpoints"
                / "blueprint_verifications"
            ),
            max_workers=args.taxonomy_verifier_workers,
        )
        taxonomy_judge = AdjudicatedTaxonomyJudge(
            AuthenticTaxonomyJudge(nemotron_super_proposer_model),
            AuthenticTaxonomyJudge(large_proposer_model),
            gemma_verifier_model,
            quality_model=nemotron_super_proposer_model,
            quality_witness_model=nemotron_super_proposer_model,
            checkpoint_root=(
                args.output_root
                / ".scenario_blueprint_v5"
                / "checkpoints"
                / "taxonomy_verifications"
            ),
            max_workers=args.taxonomy_verifier_workers,
            quality_workers=args.judge_workers,
        )
        blueprint_sets, blueprint_scrutinies = admit_blueprints(
            request=request,
            topics=topics,
            output_root=args.output_root,
            planner=planner,
            taxonomy_repair_planner=taxonomy_repair_planner,
            taxonomy_judge=taxonomy_judge,
            judge=judge,
            max_workers=args.max_workers,
            judge_workers=args.judge_workers,
            max_attempts=args.max_attempts,
            max_repair_cycles=args.max_repair_cycles,
            resume=args.resume,
        )
    if args.stage in {"scenarios", "all"}:
        if blueprint_sets is None:
            blueprint_sets = load_blueprint_sets(args.output_root, request, topics)
        if blueprint_scrutinies is None:
            blueprint_scrutinies = load_blueprint_scrutinies(
                args.output_root, request, topics, blueprint_sets
            )
        generate_scenarios(
            request=request,
            topics=topics,
            blueprint_sets=blueprint_sets,
            blueprint_scrutinies=blueprint_scrutinies,
            output_root=args.output_root,
            planner=planner,
            max_workers=args.max_workers,
            max_attempts=args.max_attempts,
            resume=args.resume,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModelTransportUnavailable as error:
        print(f"scenario blueprint v5 transport unavailable: {error}", file=sys.stderr)
        raise SystemExit(75)
    except ScenarioBlueprintError as error:
        print(f"scenario blueprint v5 failed: {error}", file=sys.stderr)
        raise SystemExit(2)
