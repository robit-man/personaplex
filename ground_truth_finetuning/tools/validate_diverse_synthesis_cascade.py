#!/usr/bin/env python3
"""Validate structural, causal, and lineage invariants of a cascade planning run."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.diverse_cascade import (
    BALANCE_AXES,
    CascadeError,
    cascade_config_hash,
    catalog_seed_ids,
    collect_schema_hashes,
    content_hash,
    is_content_hash,
    load_bound_seed_catalog,
    load_json,
    load_jsonl,
    refill_selection,
    request_requires_typed_trajectories,
    request_selection_counts,
    validate_pair_spec,
    validate_request,
    validate_scenario_contract,
    validate_topic_bindings,
    validate_topic_card,
    validate_trajectory_seed,
    validate_unique_causal_signatures,
    validate_unique_scenario_premises,
    validate_v4_pair_spec,
    validate_v4_trajectory_seed,
    write_json,
)


def unique(records: list[dict], key: str, label: str) -> None:
    values = [record.get(key) for record in records]
    if len(values) != len(set(values)):
        raise CascadeError(f"{label} has duplicate {key} values")


def validate_selection_rows(
    rows: list[dict],
    trajectory_ids: set[str],
    expected_tier: str | None,
    require_typed: bool,
) -> None:
    unique(rows, "groupId", "Selection")
    unique(rows, "trajectoryId", "Selection")
    for item in rows:
        if item.get("schema") != "personaplex.selected-trajectory.v1" or item.get("trajectoryId") not in trajectory_ids:
            raise CascadeError("Selection references an invalid trajectory")
        if expected_tier is not None and item.get("selectionTier") != expected_tier:
            raise CascadeError(f"Selection row must have {expected_tier} tier")
        if require_typed:
            dimensions = item.get("balanceDimensions")
            rationale = item.get("selectionRationale")
            if not isinstance(dimensions, dict) or set(dimensions) != set(BALANCE_AXES):
                raise CascadeError("Typed selection lacks all balance dimensions")
            if not all(isinstance(value, list) and value for value in dimensions.values()):
                raise CascadeError("Typed selection balance dimensions must be nonempty arrays")
            if not isinstance(rationale, dict) or rationale.get("algorithm") != "typed-balanced-all-leaves-v1":
                raise CascadeError("Typed selection lacks deterministic rationale")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--require-pairs", action="store_true")
    args = parser.parse_args()
    root = args.input_root.resolve()
    request_path = root / "request.json"
    request = load_json(request_path)
    validate_request(request)
    catalog, catalog_hash = load_bound_seed_catalog(request, request_path, REPOSITORY_ROOT)
    topics = load_jsonl(root / "topic_cards.jsonl")
    scenarios = load_jsonl(root / "scenario_contracts.jsonl")
    trajectories = load_jsonl(root / "trajectory_seeds.jsonl")
    primary_path = root / "primary_trajectories.jsonl"
    primary = load_jsonl(primary_path) if primary_path.exists() else load_jsonl(root / "selected_trajectories.jsonl")
    reserves = load_jsonl(root / "reserve_trajectories.jsonl")
    selection = load_jsonl(root / "selected_trajectories.jsonl")
    rejected = load_jsonl(root / "rejected_groups.jsonl")
    pairs = load_jsonl(root / "counterfactual_pair_specs.jsonl")
    coverage = request["coverageTarget"]
    primary_target, reserve_target = request_selection_counts(request)
    require_typed = request_requires_typed_trajectories(request)

    if len(topics) != coverage["candidateTopics"]:
        raise CascadeError("Topic count does not match request")
    unique(topics, "topicId", "Topic cards")
    source_seed_ids = catalog_seed_ids(catalog) if catalog is not None else None
    for topic in topics:
        validate_topic_card(topic, request["seedRevision"], source_seed_ids)
    if catalog is not None:
        validate_topic_bindings(topics, catalog)
    topic_ids = {topic["topicId"] for topic in topics}

    unique(scenarios, "scenarioId", "Scenario contracts")
    for scenario in scenarios:
        validate_scenario_contract(scenario, topic_ids)
    validate_unique_scenario_premises(scenarios)
    for topic_id in topic_ids:
        if sum(item["topicId"] == topic_id for item in scenarios) != coverage["scenariosPerTopic"]:
            raise CascadeError(f"Topic {topic_id} has the wrong scenario count")
    scenario_ids = {scenario["scenarioId"] for scenario in scenarios}

    unique(trajectories, "trajectoryId", "Trajectory seeds")
    for trajectory in trajectories:
        validate_trajectory_seed(trajectory, scenario_ids, require_typed=require_typed)
        if request.get("strategyVersion") == "semantic-control-v4":
            validate_v4_trajectory_seed(trajectory)
    validate_unique_causal_signatures(trajectories, require_typed=require_typed)
    for scenario_id in scenario_ids:
        if sum(item["scenarioId"] == scenario_id for item in trajectories) != coverage["trajectorySeedsPerScenario"]:
            raise CascadeError(f"Scenario {scenario_id} has the wrong trajectory count")
    trajectory_ids = {trajectory["trajectoryId"] for trajectory in trajectories}

    if len(primary) != primary_target or len(reserves) != reserve_target or len(selection) != primary_target:
        raise CascadeError("Primary, reserve, or active selection count does not match request")
    validate_selection_rows(primary, trajectory_ids, "primary" if require_typed or reserve_target else None, require_typed)
    validate_selection_rows(reserves, trajectory_ids, "reserve", require_typed)
    validate_selection_rows(selection, trajectory_ids, None, require_typed)
    if {row["trajectoryId"] for row in primary} & {row["trajectoryId"] for row in reserves}:
        raise CascadeError("Primary and reserve selections overlap")
    all_initial_groups = {row["groupId"] for row in primary + reserves}
    if not {row["groupId"] for row in selection}.issubset(all_initial_groups):
        raise CascadeError("Active selection is not drawn from primary/reserve groups")
    if rejected:
        expected_active = refill_selection(request, primary, reserves, rejected)
        if [row["groupId"] for row in expected_active] != [row["groupId"] for row in selection]:
            raise CascadeError("Active selection does not match deterministic typed refill")

    if args.require_pairs:
        if len(pairs) != primary_target:
            raise CascadeError("Counterfactual group count does not match active primary count")
        unique(pairs, "groupId", "Counterfactual groups")
        trajectory_by_id = {trajectory["trajectoryId"]: trajectory for trajectory in trajectories}
        for pair in pairs:
            trajectory = trajectory_by_id.get(pair.get("trajectoryId"))
            validate_pair_spec(pair, trajectory_ids, request=request, trajectory=trajectory)
            if request.get("strategyVersion") == "semantic-control-v4":
                validate_v4_pair_spec(pair)
        selected_group_ids = {item["groupId"] for item in selection}
        if {pair["groupId"] for pair in pairs} != selected_group_ids:
            raise CascadeError("Counterfactual group IDs do not exactly match active selection IDs")

    schema_hashes = collect_schema_hashes(REPOSITORY_ROOT)
    run_manifest_path = root / "run_manifest.json"
    if run_manifest_path.exists():
        manifest = load_json(run_manifest_path)
        expected_hashes = {
            "requestHash": content_hash(request),
            "catalogHash": catalog_hash,
            "configHash": cascade_config_hash(request, manifest.get("artifacts") and None),
            "schemaHash": content_hash(schema_hashes),
        }
        if manifest.get("requestHash") != expected_hashes["requestHash"] or manifest.get("catalogHash") != catalog_hash:
            raise CascadeError("Run manifest request/catalog lineage differs from validated inputs")
        for field in ("plannerHash", "configHash", "schemaHash"):
            if not is_content_hash(manifest.get(field)):
                raise CascadeError(f"Run manifest does not persist a valid {field}")

    report = {
        "schema": "personaplex.diverse-cascade-validation-report.v2",
        "requestId": request["requestId"],
        "requestHash": content_hash(request),
        "catalogHash": catalog_hash,
        "configHash": cascade_config_hash(request),
        "schemaHashes": schema_hashes,
        "schemaHash": content_hash(schema_hashes),
        "counts": {
            "topics": len(topics),
            "scenarios": len(scenarios),
            "trajectories": len(trajectories),
            "primaryGroups": len(primary),
            "reserveGroups": len(reserves),
            "activeGroups": len(selection),
            "counterfactualGroups": len(pairs),
        },
        "selectionRationaleHash": content_hash([row.get("selectionRationale") for row in primary + reserves]),
        "status": "structural_pass_not_semantically_certified",
        "requiresIndependentVorynCertification": True,
    }
    write_json(args.report, report)
    print(report)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CascadeError as error:
        print(f"cascade validation error: {error}", file=sys.stderr)
        raise SystemExit(2)
