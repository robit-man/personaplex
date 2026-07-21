from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.tools.materialize_diverse_synthesis_cascade import validate_materialization
from ground_truth_finetuning.tools.build_diverse_synthesis_cascade import ArtifactCheckpoint
from ground_truth_finetuning.training import diverse_cascade
from ground_truth_finetuning.training.diverse_cascade import (
    CascadeError,
    content_hash,
    load_bound_seed_catalog,
    plan_pair,
    plan_topics,
    prepare_run_identity,
    refill_selection,
    select_trajectories,
    validate_unique_causal_signatures,
    validate_unique_scenario_premises,
    write_json,
    write_jsonl,
)


SIBLING_ROLES = ["verified_positive", "verified_negative", "uncertain", "superseded"]


def make_catalog(count: int = 1) -> dict:
    return {
        "schema": "personaplex.diverse-seed-library.v1",
        "libraryId": "focused-catalog",
        "seeds": [
            {"id": f"S{index + 1:02d}", "title": f"Seed {index + 1}", "focus": f"Distinct focus {index + 1}"}
            for index in range(count)
        ],
    }


def make_request(
    *,
    catalog: dict | None = None,
    topics: int = 1,
    scenarios: int = 1,
    leaves: int = 10,
    primary: int = 1,
    reserve: int = 0,
    siblings: int = 4,
) -> dict:
    catalog = catalog or make_catalog(topics)
    catalog_hash = content_hash(catalog)
    coverage = {
        "candidateTopics": topics,
        "scenariosPerTopic": scenarios,
        "trajectorySeedsPerScenario": leaves,
        "primaryGroups": primary,
        "reserveGroups": reserve,
        "selectedCounterfactualGroups": primary,
        "branchesPerGroup": siblings,
    }
    request = {
        "schema": "personaplex.diverse-corpus-request.v1",
        "requestId": "focused-root-cascade",
        "seedRevision": catalog_hash,
        "seedCatalog": "catalog.json",
        "seedCatalogHash": catalog_hash,
        "seedIdeas": ["legacy-compatible seed idea"],
        "coverageTarget": coverage,
        "allowedVoicesManifest": "sha256:" + "b" * 64,
        "renderer": "voicebox_chatterbox_turbo",
        "asr": "whisper",
        "allowedPhysicalCudaDevices": [0, 1, 2],
        "prohibitedContentPolicyRevision": "safe-v1",
    }
    if siblings == 4:
        request["causalGroupContract"] = {
            "siblingRoles": SIBLING_ROLES,
            "interventionFamilies": ["semantic", "delivery", "turn_taking"],
        }
    return request


def make_topic(index: int = 0) -> dict:
    return {
        "schema": "personaplex.topic-card.v1",
        "topicId": f"topic_{index:02d}",
        "sourceSeedId": f"S{index + 1:02d}",
        "seedRevision": "unused-in-selection",
        "domain": f"Topic domain {index}",
        "interactionModes": ["service"],
        "registerRange": ["neutral"],
        "safeStakes": ["low"],
        "forbiddenPatterns": ["target dialogue"],
        "diversityTags": ["focused"],
    }


def make_scenario(topic_id: str = "topic_00", scenario_id: str = "scenario_00", premise: str | None = None) -> dict:
    return {
        "schema": "personaplex.scenario-contract.v1",
        "scenarioId": scenario_id,
        "topicId": topic_id,
        "mode": "service",
        "premise": premise or f"A concrete and structurally distinct premise for {scenario_id}.",
        "participants": [
            {"role": "caller", "knowledge": "typed caller facts"},
            {"role": "agent", "knowledge": "typed policy facts"},
        ],
        "startingState": {
            "knownFacts": ["one fact"],
            "uncertainty": ["one uncertainty"],
            "policyConstraints": ["one boundary"],
        },
        "interactionOpportunity": ["clarify"],
        "allowedToolClasses": ["lookup"],
        "disallowedClaims": ["unsupported claim"],
        "scenarioOutcomeSpace": ["resolved", "handoff"],
        "requiredControlPhenomena": ["revision"],
    }


def make_trajectory(
    index: int,
    *,
    scenario_id: str = "scenario_00",
    variant: str | None = None,
) -> dict:
    variant = variant or f"v{index}"
    return {
        "schema": "personaplex.trajectory-seed.v1",
        "trajectoryId": f"trajectory_{index:02d}",
        "scenarioId": scenario_id,
        "conversationLength": {"targetTurns": 8 + 2 * (index % 2), "min": 6, "max": 12},
        "pace": f"pace_{variant}",
        "openingStyle": f"opening_{variant}",
        "closingStyle": f"closing_{variant}",
        "voicePairPolicy": "distinct_approved_references",
        "interactionArc": ["clarification", variant],
        "duplexEvents": [{"eventType": f"event_{variant}"}],
        "postureArc": ["cooperative", f"posture_{variant}"],
        "counterfactualPivotOrdinal": 3,
        "controlPhenomena": ["typed_revision"],
        "causalAxis": f"axis_{variant}",
        "interventionFamily": f"family_{variant}",
        "typedPivot": {"field": "evidence.status", "from": "pending", "to": f"state_{variant}"},
        "postureTransition": {"from": "cooperative", "to": f"posture_{variant}"},
        "evidenceSource": f"source_{variant}",
        "outcomeRoute": f"route_{variant}",
    }


class StaticPlanner:
    def __init__(self, response: dict):
        self.response = response

    def call(self, _system: str, _user: str) -> dict:
        return deepcopy(self.response)


class SequencePlanner:
    def __init__(self, responses: list[dict | Exception]):
        self.responses = list(responses)
        self.calls = 0

    def call(self, _system: str, _user: str) -> dict:
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return deepcopy(response)


def test_artifact_retry_includes_malformed_protocol_response() -> None:
    planner = SequencePlanner([
        CascadeError("Planner returned non-JSON content; no text recovery is permitted"),
        {"items": [{"id": "valid"}]},
    ])
    result = diverse_cascade._validated_model_one(
        planner,
        "items",
        {"task": "test"},
        lambda candidate: None,
    )
    assert result == {"id": "valid"}
    assert planner.calls == 2


def test_artifact_checkpoint_is_immutable_unique_and_resumable(tmp_path) -> None:
    checkpoint = ArtifactCheckpoint(
        tmp_path,
        "scenarios",
        "scenarioId",
        unique_fields=("premise",),
    )
    first = {"scenarioId": "scenario_one", "premise": "A unique premise."}
    checkpoint.admit(first)
    resumed = ArtifactCheckpoint(
        tmp_path,
        "scenarios",
        "scenarioId",
        unique_fields=("premise",),
    )
    assert resumed.rows() == [first]
    with pytest.raises(CascadeError, match="conflicting immutable content"):
        resumed.admit({"scenarioId": "scenario_one", "premise": "Changed content."})
    with pytest.raises(CascadeError, match="duplicates scenario_one"):
        resumed.admit({"scenarioId": "scenario_two", "premise": "A unique premise."})


def test_catalog_hash_and_one_topic_per_source_binding(tmp_path) -> None:
    catalog = make_catalog(2)
    request = make_request(catalog=catalog, topics=2, leaves=1, primary=2)
    write_json(tmp_path / "catalog.json", catalog)
    request_path = tmp_path / "request.json"
    write_json(request_path, request)
    loaded, loaded_hash = load_bound_seed_catalog(request, request_path)
    assert loaded == catalog
    assert loaded_hash == request["seedCatalogHash"]

    cards = []
    for index, seed in enumerate(catalog["seeds"]):
        card = make_topic(index)
        card["seedRevision"] = request["seedRevision"]
        card["sourceSeedId"] = seed["id"]
        cards.append(card)
    planned = plan_topics(StaticPlanner({"topicCards": cards}), request, catalog)
    assert {row["sourceSeedId"] for row in planned} == {"S01", "S02"}

    mismatched = deepcopy(request)
    mismatched["seedCatalogHash"] = "sha256:" + "0" * 64
    with pytest.raises(CascadeError, match="catalog hash"):
        load_bound_seed_catalog(mismatched, request_path)


def test_checked_in_v5_request_and_catalog_pass_runtime_validation() -> None:
    request_path = (
        REPOSITORY_ROOT
        / "ground_truth_finetuning"
        / "requests"
        / "personaplex_diverse_50x20x10.control-v5.json"
    )
    request = diverse_cascade.load_json(request_path)
    diverse_cascade.validate_request(request)
    catalog, catalog_hash = load_bound_seed_catalog(
        request,
        request_path,
        REPOSITORY_ROOT,
    )
    assert catalog is not None
    assert catalog_hash == request["seedCatalogHash"] == request["seedRevision"]
    assert len(catalog["seeds"]) == request["coverageTarget"]["candidateTopics"] == 50
    assert request["coverageTarget"]["primaryGroups"] * len(
        diverse_cascade.request_sibling_roles(request)
    ) == 1000


def test_duplicate_premise_and_typed_transition_are_rejected() -> None:
    premise = "The exact same concrete premise must not be admitted twice."
    with pytest.raises(CascadeError, match="exact-duplicate-free"):
        validate_unique_scenario_premises([
            make_scenario(scenario_id="scenario_00", premise=premise),
            make_scenario(scenario_id="scenario_01", premise=premise),
        ])
    first = make_trajectory(0)
    duplicate = deepcopy(first)
    duplicate["trajectoryId"] = "trajectory_99"
    with pytest.raises(CascadeError, match="Duplicate causal state-transition signature"):
        validate_unique_causal_signatures([first, duplicate], require_typed=True)


def test_typed_causal_axis_does_not_imply_legacy_v4_envelope() -> None:
    diverse_cascade.validate_trajectory_seed(
        make_trajectory(0),
        {"scenario_00"},
        require_typed=True,
    )


def test_tenth_leaf_is_eligible_for_deterministic_selection(monkeypatch) -> None:
    request = make_request(primary=1, leaves=10)
    topics = [make_topic()]
    scenarios = [make_scenario()]
    trajectories = [make_trajectory(index) for index in range(10)]

    def rank(_request: dict, value: object) -> str:
        if isinstance(value, dict) and value.get("trajectoryId") == "trajectory_09":
            return "0"
        return "f" + content_hash(value)

    monkeypatch.setattr(diverse_cascade, "_stable_rank", rank)
    selected = select_trajectories(request, topics, scenarios, trajectories)
    assert selected[0]["trajectoryId"] == "trajectory_09"


def test_selection_balances_all_typed_dimensions(monkeypatch) -> None:
    request = make_request(primary=2, leaves=4)
    topics = [make_topic()]
    scenarios = [make_scenario()]
    trajectories = [
        make_trajectory(0, variant="a"),
        make_trajectory(1, variant="a2"),
        make_trajectory(2, variant="b"),
        make_trajectory(3, variant="b2"),
    ]
    for trajectory in trajectories[1:2]:
        for field in ("causalAxis", "interventionFamily", "postureTransition", "evidenceSource", "duplexEvents", "outcomeRoute", "conversationLength", "pace", "openingStyle", "closingStyle"):
            trajectory[field] = deepcopy(trajectories[0][field])

    def rank(_request: dict, value: object) -> str:
        if isinstance(value, dict):
            return str(value.get("trajectoryId", value.get("topicId", "z")))
        return content_hash(value)

    monkeypatch.setattr(diverse_cascade, "_stable_rank", rank)
    selected = select_trajectories(request, topics, scenarios, trajectories)
    dimensions = [row["balanceDimensions"] for row in selected]
    assert set(dimensions[0]) == set(diverse_cascade.BALANCE_AXES)
    for axis in diverse_cascade.BALANCE_AXES:
        assert dimensions[0][axis] != dimensions[1][axis]
    assert all(row["selectionRationale"]["algorithm"] == "typed-balanced-all-leaves-v1" for row in selected)


def test_request_defined_four_sibling_group_uses_one_family_and_typed_pivot() -> None:
    request = make_request(primary=1, leaves=1)
    scenario = make_scenario()
    trajectory = make_trajectory(0, variant="semantic")
    selection = {"groupId": "cascade-focused-root-cascade-0001", "trajectoryId": trajectory["trajectoryId"]}
    common_context = {
        "requestId": request["requestId"],
        "scenario": scenario,
        "trajectory": trajectory,
        "groupId": selection["groupId"],
        "interventionFamily": trajectory["interventionFamily"],
        "typedPivot": trajectory["typedPivot"],
    }
    branch_states = {
        "verified_positive": "verified",
        "verified_negative": "rejected",
        "uncertain": "pending",
        "superseded": "stale",
    }
    group = {
        "schema": "personaplex.counterfactual-sibling-group-spec.v1",
        "groupId": selection["groupId"],
        "trajectoryId": trajectory["trajectoryId"],
        "pivotOrdinal": trajectory["counterfactualPivotOrdinal"],
        "commonContextHash": content_hash(common_context),
        "interventionFamily": trajectory["interventionFamily"],
        "typedPivot": trajectory["typedPivot"],
        "branches": [
            {
                "branchId": role,
                "controlDelta": {"field": "evidence.status", "from": "pending", "to": state},
                "evidenceUpdate": {"source": "typed_fixture", "status": state},
            }
            for role, state in branch_states.items()
        ],
    }
    planned = plan_pair(StaticPlanner({"groupSpecs": [group]}), request, selection, scenario, trajectory)
    assert planned["interventionFamily"] == trajectory["interventionFamily"]
    assert {branch["branchId"] for branch in planned["branches"]} == set(SIBLING_ROLES)


def test_typed_rejection_promotes_reserve_in_deterministic_order() -> None:
    request = make_request(primary=2, reserve=2, leaves=4)
    selected = select_trajectories(
        request,
        [make_topic()],
        [make_scenario()],
        [make_trajectory(index) for index in range(4)],
    )
    primary = [row for row in selected if row["selectionTier"] == "primary"]
    reserves = [row for row in selected if row["selectionTier"] == "reserve"]
    rejected = [{
        "schema": "personaplex.rejected-causal-group.v1",
        "groupId": primary[0]["groupId"],
        "stage": "source_validation",
        "reasonCode": "typed_control_mismatch",
    }]
    active = refill_selection(request, list(reversed(primary)), list(reversed(reserves)), rejected)
    promoted = next(row for row in active if row["materializationRole"] == "refill")
    assert promoted["groupId"] == min(reserves, key=lambda row: row["tierOrdinal"])["groupId"]
    assert promoted["replacesGroupId"] == primary[0]["groupId"]


def test_resume_rejects_request_and_catalog_hash_changes(tmp_path) -> None:
    catalog = make_catalog()
    request = make_request(catalog=catalog)
    prepare_run_identity(tmp_path, request, catalog, content_hash(catalog), resume=False)
    changed = deepcopy(request)
    changed["requestId"] = "different-root-cascade"
    with pytest.raises(CascadeError, match="request hash differs"):
        prepare_run_identity(tmp_path, changed, catalog, content_hash(catalog), resume=True)
    changed_catalog = deepcopy(catalog)
    changed_catalog["libraryId"] = "changed-catalog"
    with pytest.raises(CascadeError, match="catalog"):
        prepare_run_identity(tmp_path, request, changed_catalog, content_hash(changed_catalog), resume=True)


def test_materializer_uses_request_defined_primary_reserve_and_sibling_counts(tmp_path) -> None:
    request = make_request(leaves=2, primary=1, reserve=1)
    root = tmp_path / "cascade"
    output = tmp_path / "plan.jsonl"
    group_id = "cascade-focused-root-cascade-0001"
    write_jsonl(root / "topic_cards.jsonl", [{"topicId": "topic_00"}])
    write_jsonl(root / "scenario_contracts.jsonl", [{"scenarioId": "scenario_00"}])
    write_jsonl(root / "trajectory_seeds.jsonl", [{"trajectoryId": "trajectory_00"}, {"trajectoryId": "trajectory_01"}])
    write_jsonl(root / "primary_trajectories.jsonl", [{"groupId": group_id, "trajectoryId": "trajectory_00"}])
    write_jsonl(root / "reserve_trajectories.jsonl", [{"groupId": "cascade-focused-root-cascade-0002", "trajectoryId": "trajectory_01"}])
    write_jsonl(root / "selected_trajectories.jsonl", [{"groupId": group_id, "trajectoryId": "trajectory_00"}])
    write_jsonl(root / "counterfactual_pair_specs.jsonl", [{"groupId": group_id}])
    write_jsonl(output, [
        {"counterfactual": {"groupId": group_id, "branchId": role}}
        for role in SIBLING_ROLES
    ])
    result = validate_materialization(root, output, request)
    assert result["counts"]["primaryGroups"] == 1
    assert result["counts"]["reserveGroups"] == 1
    assert result["counts"]["vorynBranches"] == 4
