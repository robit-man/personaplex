from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCHEMAS = ROOT / "ground_truth_finetuning" / "schemas"
CATALOG = ROOT / "ground_truth_finetuning" / "seed_catalogs" / "personaplex_diverse_seed_library.v2.json"
REQUEST = ROOT / "ground_truth_finetuning" / "requests" / "personaplex_diverse_50x20x10.control-v5.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def content_digest(path: Path) -> str:
    payload = json.dumps(load(path), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + sha256(payload.encode("utf-8")).hexdigest()


def validator(name: str) -> jsonschema.Draft202012Validator:
    schema = load(SCHEMAS / name)
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


def test_v2_catalog_is_bound_unique_and_factorized() -> None:
    catalog = load(CATALOG)
    validator("diverse_seed_library.v2.schema.json").validate(catalog)
    assert len(catalog["seeds"]) == 50
    assert len({seed["id"] for seed in catalog["seeds"]}) == 50
    assert len({seed["title"] for seed in catalog["seeds"]}) == 50
    for seed in catalog["seeds"]:
        assert {item["family"] for item in seed["causalAffordances"]} == {
            "semantic",
            "delivery",
            "turn_taking",
        }


def test_v5_request_selects_one_thousand_from_ten_thousand_candidates() -> None:
    request = load(REQUEST)
    validator("diverse_corpus_request.v2.schema.json").validate(request)
    coverage = request["coverageTarget"]
    assert coverage["candidateTopics"] * coverage["scenariosPerTopic"] * coverage["trajectorySeedsPerScenario"] == 10_000
    assert coverage["primaryGroups"] == coverage["selectedCounterfactualGroups"]
    assert coverage["primaryGroups"] * coverage["branchesPerGroup"] == 1_000
    assert coverage["reserveGroups"] == 250
    assert request["seedRevision"] == request["seedCatalogHash"] == content_digest(CATALOG)
    assert request["allowedPhysicalCudaDevices"] == [0, 1, 2]
    assert request["resourcePolicy"]["hardwareDiscovery"] == "runtime"
    assert request["resourcePolicy"]["cpuModelFallback"] is False
    assert request["planner"]["reasoning"] is False


def test_v5_request_rejects_legacy_branch_count() -> None:
    request = load(REQUEST)
    request["coverageTarget"]["branchesPerGroup"] = 2
    with pytest.raises(jsonschema.ValidationError):
        validator("diverse_corpus_request.v2.schema.json").validate(request)


def trajectory_fixture() -> dict:
    return {
        "schema": "personaplex.trajectory-seed.v2",
        "trajectoryId": "trajectory_semantic_001",
        "scenarioId": "scenario_semantic_001",
        "conversationLength": {"targetTurns": 8, "min": 6, "max": 10},
        "pace": "natural variable pace",
        "openingStyle": "contextual continuation",
        "closingStyle": "model selected concise close",
        "voicePairPolicy": "distinct_approved_references",
        "interactionArc": ["clarify", "resolve", "close"],
        "duplexEvents": [{"eventType": "barge_in", "targetOrdinal": 3, "offsetMs": 420, "overlapMs": 180, "cancelOutgoingAudio": True, "invalidateGeneration": True}],
        "postureArc": ["skeptical", "conditionally cooperative"],
        "counterfactualPivotOrdinal": 3,
        "controlPhenomena": ["mutable revision", "barge-in recovery"],
        "causalAxis": "evidence status",
        "interventionFamily": "semantic",
        "typedPivot": {"field": "state.evidence.status", "from": "uncertain", "to": "verified"},
        "postureTransition": {"from": "skeptical", "to": "conditionally cooperative"},
        "evidenceSource": "tool_result",
        "outcomeRoute": "offer supported options",
        "semanticStateArc": [{"state": "awaiting evidence"}, {"state": "evidence received"}],
        "controlRevisionSchedule": [{"controlRevision": 2, "targetOrdinal": 3, "availableBeforeTarget": True, "source": "tool_result"}],
        "terminationContract": {"decisionSource": "model", "action": "end_call_tool", "deterministicPhrase": False},
        "negativeControlCoverage": ["paired_wrong_branch", "stale_revision", "null_control"],
    }


def test_artifact_schema_has_a_strict_root_and_rejects_target_fields() -> None:
    artifacts = validator("diverse_cascade_artifacts.v2.schema.json")
    trajectory = trajectory_fixture()
    artifacts.validate(trajectory)
    leaked = deepcopy(trajectory)
    leaked["targetText"] = "The answer the agent must say."
    with pytest.raises(jsonschema.ValidationError):
        artifacts.validate(leaked)


def test_four_sibling_group_requires_typed_control_values() -> None:
    roles = ["verified_positive", "verified_negative", "uncertain", "superseded"]
    group = {
        "schema": "personaplex.counterfactual-sibling-group-spec.v2",
        "groupId": "causal-group-semantic-001",
        "trajectoryId": "trajectory_semantic_001",
        "pivotOrdinal": 3,
        "commonContextHash": "sha256:" + "b" * 64,
        "sharedPrefixPolicy": "native_code_identical_through_pivot",
        "interventionFamily": "semantic",
        "typedPivot": {"field": "state.evidence.status", "from": "uncertain", "to": "verified"},
        "branches": [
            {
                "branchId": role,
                "controlDelta": {"field": "state.evidence.status", "from": "uncertain", "to": role},
                "controlValue": role,
                "evidenceUpdate": {"status": role},
                "availabilityTiming": {"effectiveFrom": "next_agent_turn", "availableBeforeTarget": True, "controlRevision": index + 2},
                "negativeControls": ["paired_wrong_branch", "stale_revision", "null_control"],
                "semanticAssertions": ["response meaning must follow the active evidence status"],
            }
            for index, role in enumerate(roles)
        ],
    }
    artifacts = validator("diverse_cascade_artifacts.v2.schema.json")
    artifacts.validate(group)
    del group["branches"][0]["controlValue"]
    with pytest.raises(jsonschema.ValidationError):
        artifacts.validate(group)
