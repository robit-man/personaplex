#!/usr/bin/env python3
"""Build one stage of the agent-operable diverse controlled-synthesis cascade."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from threading import Lock

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.diverse_cascade import (
    CascadeError,
    JsonOnlyPlanner,
    PlannerConfig,
    content_hash,
    collect_schema_hashes,
    load_bound_seed_catalog,
    load_json,
    load_jsonl,
    parallel_map,
    plan_pair,
    plan_scenarios,
    plan_topics,
    plan_trajectories,
    prepare_run_identity,
    refill_selection,
    request_requires_typed_trajectories,
    request_selection_counts,
    select_trajectories,
    validate_request,
    validate_unique_causal_signatures,
    validate_unique_scenario_premises,
    write_jsonl,
    write_json,
    write_run_manifest,
)


def require_existing(path: Path, label: str) -> list[dict]:
    rows = load_jsonl(path)
    if not rows:
        raise CascadeError(f"{label} is required before this stage: {path}")
    return rows


def reject_legacy_v5_trajectory_stage(request: dict, stage: str) -> None:
    """Prevent the retired sequential planner from handling v5 trajectories."""

    if request.get("strategyVersion") == "semantic-control-v5" and stage in {
        "trajectories",
        "all",
    }:
        raise CascadeError(
            "semantic-control-v5 trajectories require "
            "ground_truth_finetuning/tools/build_compact_trajectory_fanout_v5.py"
        )


class ArtifactCheckpoint:
    """Thread-safe immutable per-artifact checkpoints for resumable model stages."""

    def __init__(
        self,
        root: Path,
        stage: str,
        identity_field: str,
        existing_records: list[dict] | None = None,
        unique_fields: tuple[str, ...] = (),
    ) -> None:
        self.directory = root / ".stage_checkpoints" / stage
        self.identity_field = identity_field
        self.unique_fields = unique_fields
        self._lock = Lock()
        self._records: dict[str, dict] = {}
        self._unique_values: dict[str, dict[object, str]] = {field: {} for field in unique_fields}
        for record in existing_records or []:
            self._register(record)
        if self.directory.exists():
            for path in sorted(self.directory.glob("*.json")):
                self._register(load_json(path))

    def _identity(self, record: dict) -> str:
        value = record.get(self.identity_field)
        if not isinstance(value, str) or not value:
            raise CascadeError(f"Checkpoint record lacks {self.identity_field}")
        return value

    def _register(self, record: dict) -> bool:
        identity = self._identity(record)
        prior = self._records.get(identity)
        if prior is not None:
            if prior != record:
                raise CascadeError(f"Checkpoint identity {identity} has conflicting immutable content")
            return False
        for field, seen in self._unique_values.items():
            value = record.get(field)
            if value is None:
                raise CascadeError(f"Checkpoint record lacks unique field {field}")
            prior_identity = seen.get(value)
            if prior_identity is not None:
                raise CascadeError(
                    f"Checkpoint {field} duplicates {prior_identity}; regenerate {identity}"
                )
        self._records[identity] = record
        for field, seen in self._unique_values.items():
            seen[record[field]] = identity
        return True

    def admit(self, record: dict) -> None:
        with self._lock:
            if not self._register(record):
                return
            artifact_hash = content_hash({
                "identityField": self.identity_field,
                "identity": record[self.identity_field],
            })[7:]
            write_json(self.directory / f"{artifact_hash}.json", record)

    def rows(self) -> list[dict]:
        with self._lock:
            return [self._records[key] for key in sorted(self._records)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("topics", "scenarios", "trajectories", "selection", "refill", "pairs", "all"))
    parser.add_argument("--planner-endpoint", default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_ENDPOINT", ""))
    parser.add_argument("--planner-model", default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_MODEL", ""))
    parser.add_argument("--planner-api-key", default=os.environ.get("PERSONAPLEX_CASCADE_PLANNER_API_KEY", ""))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--rejected-groups", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1 or args.max_workers > 64:
        raise CascadeError("max-workers must be in [1, 64]")

    request_path = args.request.resolve()
    request = load_json(request_path)
    validate_request(request)
    reject_legacy_v5_trajectory_stage(request, args.stage)
    catalog, catalog_hash = load_bound_seed_catalog(request, request_path, REPOSITORY_ROOT)
    root = args.output_root.resolve()
    prepare_run_identity(root, request, catalog, catalog_hash, args.resume)
    schema_hashes = collect_schema_hashes(REPOSITORY_ROOT)
    paths = {
        "topics": root / "topic_cards.jsonl",
        "scenarios": root / "scenario_contracts.jsonl",
        "trajectories": root / "trajectory_seeds.jsonl",
        "primary": root / "primary_trajectories.jsonl",
        "reserve": root / "reserve_trajectories.jsonl",
        "selection": root / "selected_trajectories.jsonl",
        "rejections": root / "rejected_groups.jsonl",
        "refill": root / "refill_selection.jsonl",
        "pairs": root / "counterfactual_pair_specs.jsonl",
    }
    artifacts = {name: load_jsonl(path) for name, path in paths.items()}
    if not artifacts["primary"] and artifacts["selection"] and not artifacts["refill"]:
        artifacts["primary"] = list(artifacts["selection"])

    needs_planner = args.stage in {"topics", "scenarios", "trajectories", "pairs", "all"}
    planner_config = PlannerConfig(args.planner_endpoint, args.planner_model, args.planner_api_key) if needs_planner else None
    planner = JsonOnlyPlanner(planner_config) if planner_config is not None else None

    def manifest(stage: str) -> None:
        write_run_manifest(
            root,
            request,
            stage,
            artifacts,
            planner_config=planner_config,
            max_workers=args.max_workers,
            catalog_hash=catalog_hash,
            schema_hashes=schema_hashes,
        )

    if args.stage in {"topics", "all"}:
        if artifacts["topics"] and not args.resume:
            raise CascadeError("topic_cards.jsonl already exists; use --resume")
        checkpoint = ArtifactCheckpoint(root, "topics", "topicId", artifacts["topics"])
        artifacts["topics"] = plan_topics(
            planner,
            request,
            catalog,
            args.max_workers,
            existing_records=checkpoint.rows(),
            on_record=checkpoint.admit,
        )
        write_jsonl(paths["topics"], artifacts["topics"])
        manifest("topics")

    if args.stage in {"scenarios", "all"}:
        topics = require_existing(paths["topics"], "Topic cards")
        if artifacts["scenarios"] and not args.resume:
            raise CascadeError("scenario_contracts.jsonl already exists; use --resume")
        checkpoint = ArtifactCheckpoint(
            root,
            "scenarios",
            "scenarioId",
            artifacts["scenarios"],
            unique_fields=("premise",),
        )
        existing_by_topic: dict[str, list[dict]] = {}
        for row in checkpoint.rows():
            existing_by_topic.setdefault(row["topicId"], []).append(row)
        generated = parallel_map(
            topics,
            lambda topic: plan_scenarios(
                planner,
                topic,
                request,
                existing_records=existing_by_topic.get(topic["topicId"], []),
                on_record=checkpoint.admit,
            ),
            args.max_workers,
        )
        artifacts["scenarios"] = sorted((row for batch in generated for row in batch), key=lambda row: row["scenarioId"])
        validate_unique_scenario_premises(artifacts["scenarios"])
        write_jsonl(paths["scenarios"], artifacts["scenarios"])
        manifest("scenarios")

    if args.stage in {"trajectories", "all"}:
        scenarios = require_existing(paths["scenarios"], "Scenario contracts")
        if artifacts["trajectories"] and not args.resume:
            raise CascadeError("trajectory_seeds.jsonl already exists; use --resume")
        checkpoint = ArtifactCheckpoint(root, "trajectories", "trajectoryId", artifacts["trajectories"])
        existing_by_scenario: dict[str, list[dict]] = {}
        for row in checkpoint.rows():
            existing_by_scenario.setdefault(row["scenarioId"], []).append(row)
        generated = parallel_map(
            scenarios,
            lambda scenario: plan_trajectories(
                planner,
                scenario,
                request,
                existing_records=existing_by_scenario.get(scenario["scenarioId"], []),
                on_record=checkpoint.admit,
            ),
            args.max_workers,
        )
        artifacts["trajectories"] = sorted((row for batch in generated for row in batch), key=lambda row: row["trajectoryId"])
        validate_unique_causal_signatures(
            artifacts["trajectories"],
            require_typed=request_requires_typed_trajectories(request),
        )
        write_jsonl(paths["trajectories"], artifacts["trajectories"])
        manifest("trajectories")

    if args.stage in {"selection", "all"}:
        topics = require_existing(paths["topics"], "Topic cards")
        scenarios = require_existing(paths["scenarios"], "Scenario contracts")
        trajectories = require_existing(paths["trajectories"], "Trajectory seeds")
        if artifacts["primary"] or artifacts["reserve"] or artifacts["selection"]:
            if not args.resume:
                raise CascadeError("Selection artifacts already exist; use --resume")
        else:
            selected = select_trajectories(request, topics, scenarios, trajectories)
            artifacts["primary"] = [row for row in selected if row["selectionTier"] == "primary"]
            artifacts["reserve"] = [row for row in selected if row["selectionTier"] == "reserve"]
            artifacts["selection"] = list(artifacts["primary"])
            write_jsonl(paths["primary"], artifacts["primary"])
            write_jsonl(paths["reserve"], artifacts["reserve"])
            write_jsonl(paths["selection"], artifacts["selection"])
            manifest("selection")

    if args.stage == "refill" or (args.stage == "all" and args.rejected_groups is not None):
        if artifacts["pairs"]:
            raise CascadeError("Refill must run before counterfactual groups are generated")
        primary = require_existing(paths["primary"], "Primary selections")
        _, reserve_count = request_selection_counts(request)
        reserves = load_jsonl(paths["reserve"])
        if len(reserves) != reserve_count:
            raise CascadeError("Reserve selections do not match the request")
        if args.rejected_groups is None:
            raise CascadeError("--rejected-groups is required for the refill stage")
        rejected = load_jsonl(args.rejected_groups.resolve())
        if not rejected:
            raise CascadeError("Refill requires at least one typed rejected-group record")
        artifacts["rejections"] = rejected
        artifacts["refill"] = refill_selection(request, primary, reserves, rejected)
        artifacts["selection"] = list(artifacts["refill"])
        write_jsonl(paths["rejections"], rejected)
        write_jsonl(paths["refill"], artifacts["refill"])
        write_jsonl(paths["selection"], artifacts["selection"])
        manifest("refill")

    if args.stage in {"pairs", "all"}:
        scenarios = require_existing(paths["scenarios"], "Scenario contracts")
        trajectories = require_existing(paths["trajectories"], "Trajectory seeds")
        selection = require_existing(paths["selection"], "Active selected trajectories")
        if artifacts["pairs"] and not args.resume:
            raise CascadeError("counterfactual_pair_specs.jsonl already exists; use --resume")
        checkpoint = ArtifactCheckpoint(root, "pairs", "groupId", artifacts["pairs"])
        scenario_by_id = {item["scenarioId"]: item for item in scenarios}
        trajectory_by_id = {item["trajectoryId"]: item for item in trajectories}
        completed_ids = {row["groupId"] for row in checkpoint.rows()}
        pending = [row for row in selection if row["groupId"] not in completed_ids]
        generated = parallel_map(
            pending,
            lambda row: plan_pair(planner, request, row, scenario_by_id[row["scenarioId"]], trajectory_by_id[row["trajectoryId"]]),
            args.max_workers,
            on_result=checkpoint.admit,
        )
        if generated:
            completed_ids.update(row["groupId"] for row in generated)
        artifacts["pairs"] = checkpoint.rows()
        expected_ids = {row["groupId"] for row in selection}
        if completed_ids != expected_ids:
            raise CascadeError("Counterfactual group checkpoints do not exactly cover active selection")
        write_jsonl(paths["pairs"], artifacts["pairs"])
        manifest("pairs")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CascadeError as error:
        print(f"cascade error: {error}", file=sys.stderr)
        raise SystemExit(2)
