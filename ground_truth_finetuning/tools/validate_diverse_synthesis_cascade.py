#!/usr/bin/env python3
"""Validate structural, causal, and lineage invariants of a cascade planning run.

This validator intentionally does not judge semantic naturalness or certify audio. Those
are independent Voryn source-certification responsibilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.diverse_cascade import (
    CascadeError,
    content_hash,
    load_json,
    load_jsonl,
    validate_pair_spec,
    validate_request,
    validate_scenario_contract,
    validate_topic_card,
    validate_trajectory_seed,
    write_json,
)


def unique(records: list[dict], key: str, label: str) -> None:
    values = [record.get(key) for record in records]
    if len(values) != len(set(values)):
        raise CascadeError(f"{label} has duplicate {key} values")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--require-pairs", action="store_true")
    args = parser.parse_args()
    root = args.input_root.resolve()
    request = load_json(root / "request.json")
    validate_request(request)
    topics = load_jsonl(root / "topic_cards.jsonl")
    scenarios = load_jsonl(root / "scenario_contracts.jsonl")
    trajectories = load_jsonl(root / "trajectory_seeds.jsonl")
    selection = load_jsonl(root / "selected_trajectories.jsonl")
    pairs = load_jsonl(root / "counterfactual_pair_specs.jsonl")
    if len(topics) != request["coverageTarget"]["candidateTopics"]:
        raise CascadeError("Topic count does not match request")
    unique(topics, "topicId", "Topic cards")
    for topic in topics:
        validate_topic_card(topic, request["seedRevision"])
    topic_ids = {topic["topicId"] for topic in topics}
    unique(scenarios, "scenarioId", "Scenario contracts")
    for scenario in scenarios:
        validate_scenario_contract(scenario, topic_ids)
    for topic_id in topic_ids:
        if sum(item["topicId"] == topic_id for item in scenarios) != request["coverageTarget"]["scenariosPerTopic"]:
            raise CascadeError(f"Topic {topic_id} has the wrong scenario count")
    scenario_ids = {scenario["scenarioId"] for scenario in scenarios}
    unique(trajectories, "trajectoryId", "Trajectory seeds")
    for trajectory in trajectories:
        validate_trajectory_seed(trajectory, scenario_ids)
    for scenario_id in scenario_ids:
        if sum(item["scenarioId"] == scenario_id for item in trajectories) != request["coverageTarget"]["trajectorySeedsPerScenario"]:
            raise CascadeError(f"Scenario {scenario_id} has the wrong trajectory count")
    trajectory_ids = {trajectory["trajectoryId"] for trajectory in trajectories}
    expected_selected = request["coverageTarget"]["selectedCounterfactualGroups"]
    if selection and len(selection) != expected_selected:
        raise CascadeError("Selection count does not match request")
    unique(selection, "groupId", "Selected trajectories")
    unique(selection, "trajectoryId", "Selected trajectories")
    for item in selection:
        if item.get("schema") != "personaplex.selected-trajectory.v1" or item.get("trajectoryId") not in trajectory_ids:
            raise CascadeError("Selection references an invalid trajectory")
    if args.require_pairs:
        if len(pairs) != expected_selected:
            raise CascadeError("Pair count does not match selected group count")
        unique(pairs, "groupId", "Counterfactual pairs")
        for pair in pairs:
            validate_pair_spec(pair, trajectory_ids)
        selected_group_ids = {item["groupId"] for item in selection}
        if {pair["groupId"] for pair in pairs} != selected_group_ids:
            raise CascadeError("Pair group IDs do not exactly match selected group IDs")
    report = {
        "schema": "personaplex.diverse-cascade-validation-report.v1",
        "requestId": request["requestId"],
        "requestHash": content_hash(request),
        "counts": {
            "topics": len(topics),
            "scenarios": len(scenarios),
            "trajectories": len(trajectories),
            "selectedGroups": len(selection),
            "pairs": len(pairs),
        },
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
