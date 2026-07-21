from pathlib import Path

import pytest

from ground_truth_finetuning.training.scenario_adjudication_v5 import (
    AdjudicatedScenarioJudge,
    AuthenticDecomposedScenarioAdjudicator,
    _adjudicator_response_schema,
    _materialize_adjudication,
    finding_claims_from_decisions,
)
from ground_truth_finetuning.training.diverse_cascade import PlannerConfig
from ground_truth_finetuning.training.scenario_scrutiny import (
    DIMENSION_KEYS,
    ScenarioScrutinyError,
)


def _decision(topic_id: str, ids: list[str], rejected: set[str]) -> dict:
    status = "fail" if rejected else "pass"
    findings = [{
        "code": "incoherent_known_facts",
        "rationale": "Known facts materially contradict.",
        "relatedScenarioIds": [],
    }]
    return {
        "topicId": topic_id,
        "groupDecision": "reject" if rejected else "pass",
        "groupRationale": "Bound semantic decision.",
        "dimensionVerdicts": {
            key: {"status": status if key == "statePolicyOutcomeCoherence" else "pass", "rationale": "Bound."}
            for key in DIMENSION_KEYS
        },
        "accepted": [
            {"scenarioId": scenario_id, "rationale": "Accepted."}
            for scenario_id in ids if scenario_id not in rejected
        ],
        "rejected": [
            {"scenarioId": scenario_id, "findings": findings}
            for scenario_id in ids if scenario_id in rejected
        ],
    }


class _Judge:
    def __init__(self, model: str, rejected: set[str]):
        self.model = model
        self.rejected = rejected
        self.calls = 0

    def binding(self) -> dict:
        return {"model": self.model, "reasoning": {"enabled": False}}

    def audit_topic(self, topic: dict, scenarios: list[dict]) -> dict:
        self.calls += 1
        return _decision(topic["topicId"], [row["scenarioId"] for row in scenarios], self.rejected)


class _Adjudicator:
    def __init__(self, model: str, rejected: set[str]):
        self.model = model
        self.rejected = rejected
        self.calls = 0
        self.candidates: list[str] = []
        self.claims: list[dict] = []
        self.error: Exception | None = None

    def binding(self) -> dict:
        return {"model": self.model, "reasoning": {"enabled": False}}

    def adjudicate(self, topic, scenarios, source_blueprints, finding_claims):
        self.calls += 1
        self.candidates = [row["scenarioId"] for row in scenarios]
        self.claims = list(finding_claims)
        if self.error is not None:
            raise self.error
        ids = [row["scenarioId"] for row in scenarios]
        return _decision(topic["topicId"], ids, self.rejected)


def _group() -> tuple[dict, list[dict]]:
    topic = {"topicId": "topic_test"}
    scenarios = [
        {
            "scenarioId": f"scenario_topic_test_{index:02d}",
            "topicId": "topic_test",
            "premise": f"Distinct premise {index}.",
            "startingState": {
                "knownFacts": [f"Known fact {index}."],
                "uncertainty": [f"Open question {index}."],
                "policyConstraints": [f"Constraint {index}."],
            },
            "interactionOpportunity": ["barge_in_repair"],
            "allowedToolClasses": ["state_lookup"],
            "disallowedClaims": [f"Unsupported claim {index}."],
            "scenarioOutcomeSpace": [f"verified_positive: outcome {index}."],
            "requiredControlPhenomena": ["control_operator: evidence_status_revision"],
        }
        for index in range(1, 21)
    ]
    return topic, scenarios


def _blueprint_sets(scenarios: list[dict]) -> list[dict]:
    return [{
        "topicId": "topic_test",
        "blueprints": {
            row["scenarioId"]: {
                "interactionMode": "mode_test",
                "submode": f"submode {index}",
                "participantRelationship": "peer relationship",
                "setting": f"setting {index}",
                "centralResource": f"resource {index}",
                "centralTension": f"tension {index}",
                "evidencePivot": f"evidence {index}",
                "causalMechanism": f"mechanism {index}",
                "controlOperator": "evidence_status_revision",
                "duplexOpportunity": "interruptible clarification",
            }
            for index, row in enumerate(scenarios, start=1)
        },
    }]


def _bind_modes(scenarios: list[dict]) -> None:
    for row in scenarios:
        row["mode"] = "mode_test"


def test_all_proposed_rejections_require_third_model_adjudication(tmp_path: Path) -> None:
    topic, scenarios = _group()
    _bind_modes(scenarios)
    first, second, third = scenarios[0]["scenarioId"], scenarios[1]["scenarioId"], scenarios[2]["scenarioId"]
    primary = _Judge("primary-model", {first, second})
    secondary = _Judge("secondary-model", {second, third})
    adjudicator = _Adjudicator("arbiter-model", {third})
    judge = AdjudicatedScenarioJudge(
        primary, secondary, adjudicator, trace_root=tmp_path,
        blueprint_sets=_blueprint_sets(scenarios),
    )

    decision = judge.audit_topic(topic, scenarios)

    assert adjudicator.candidates == [row["scenarioId"] for row in scenarios]
    assert [item["scenarioId"] for item in decision["rejected"]] == [third]


def test_clean_proposers_still_require_blind_full_group_adjudication(tmp_path: Path) -> None:
    topic, scenarios = _group()
    _bind_modes(scenarios)
    primary = _Judge("primary-model", set())
    secondary = _Judge("secondary-model", set())
    adjudicator = _Adjudicator("arbiter-model", set())
    judge = AdjudicatedScenarioJudge(
        primary, secondary, adjudicator, trace_root=tmp_path,
        blueprint_sets=_blueprint_sets(scenarios),
    )

    assert judge.audit_topic(topic, scenarios)["groupDecision"] == "pass"
    assert adjudicator.calls == 1
    assert adjudicator.candidates == [row["scenarioId"] for row in scenarios]


def test_immutable_trace_resumes_without_recalling_models(tmp_path: Path) -> None:
    topic, scenarios = _group()
    _bind_modes(scenarios)
    candidate = scenarios[0]["scenarioId"]
    primary = _Judge("primary-model", {candidate})
    secondary = _Judge("secondary-model", set())
    adjudicator = _Adjudicator("arbiter-model", set())
    judge = AdjudicatedScenarioJudge(
        primary, secondary, adjudicator, trace_root=tmp_path,
        blueprint_sets=_blueprint_sets(scenarios),
    )

    first = judge.audit_topic(topic, scenarios)
    second = judge.audit_topic(topic, scenarios)

    assert first == second
    assert primary.calls == secondary.calls == adjudicator.calls == 1


def test_failed_arbiter_retry_reuses_immutable_proposals(tmp_path: Path) -> None:
    topic, scenarios = _group()
    _bind_modes(scenarios)
    candidate = scenarios[0]["scenarioId"]
    primary = _Judge("primary-model", {candidate})
    secondary = _Judge("secondary-model", set())
    adjudicator = _Adjudicator("arbiter-model", set())
    adjudicator.error = RuntimeError("temporary arbiter failure")
    judge = AdjudicatedScenarioJudge(
        primary, secondary, adjudicator, trace_root=tmp_path,
        blueprint_sets=_blueprint_sets(scenarios),
    )

    with pytest.raises(RuntimeError, match="temporary"):
        judge.audit_topic(topic, scenarios)
    adjudicator.error = None
    assert judge.audit_topic(topic, scenarios)["groupDecision"] == "pass"
    assert primary.calls == secondary.calls == 1
    assert adjudicator.calls == 2


def test_scenario_mode_must_match_same_id_source_blueprint(tmp_path: Path) -> None:
    topic, scenarios = _group()
    _bind_modes(scenarios)
    scenarios[0]["mode"] = "wrong_mode"
    judge = AdjudicatedScenarioJudge(
        _Judge("primary-model", set()),
        _Judge("secondary-model", set()),
        _Adjudicator("arbiter-model", set()),
        trace_root=tmp_path,
        blueprint_sets=_blueprint_sets(scenarios),
    )
    with pytest.raises(ScenarioScrutinyError, match="source blueprint"):
        judge.audit_topic(topic, scenarios)


def test_all_three_models_must_be_distinct(tmp_path: Path) -> None:
    _topic, scenarios = _group()
    _bind_modes(scenarios)
    with pytest.raises(ScenarioScrutinyError, match="distinct"):
        AdjudicatedScenarioJudge(
            _Judge("same-model", set()),
            _Judge("same-model", set()),
            _Adjudicator("arbiter-model", set()),
            trace_root=tmp_path,
            blueprint_sets=_blueprint_sets(scenarios),
        )


def test_arbiter_wire_contract_contains_findings_only() -> None:
    candidate_ids = ["scenario_topic_test_01", "scenario_topic_test_02"]
    schema = _adjudicator_response_schema(candidate_ids)
    assert schema["required"] == candidate_ids
    assert set(schema["properties"]) == set(candidate_ids)
    for scenario_id in candidate_ids:
        assert schema["properties"][scenario_id]["required"] == ["findings"]


def test_typed_findings_solely_materialize_consistent_verdicts() -> None:
    topic, scenarios = _group()
    ids = [row["scenarioId"] for row in scenarios]
    candidate_ids = ids[:2]
    decision = _materialize_adjudication(
        {
            candidate_ids[0]: {"findings": [{
                "code": "semantic_near_duplicate",
                "rationale": "The concrete causal trajectories are interchangeable.",
                "relatedScenarioIds": [candidate_ids[1]],
            }]},
            candidate_ids[1]: {"findings": []},
        },
        topic["topicId"],
        ids,
        candidate_ids,
    )
    assert decision["groupDecision"] == "reject"
    assert decision["dimensionVerdicts"]["semanticDiversity"]["status"] == "fail"
    assert len(decision["rejected"]) == 2


def test_finding_claims_are_typed_union_without_proposer_rationales() -> None:
    topic, scenarios = _group()
    ids = [row["scenarioId"] for row in scenarios]
    first = _decision(topic["topicId"], ids, {ids[0], ids[1]})
    first["rejected"][0]["findings"] = [{
        "code": "semantic_near_duplicate", "rationale": "Bound pair.",
        "relatedScenarioIds": [ids[1]],
    }]
    first["rejected"][1]["findings"] = [{
        "code": "semantic_near_duplicate", "rationale": "Bound pair.",
        "relatedScenarioIds": [ids[0]],
    }]
    second = _decision(topic["topicId"], ids, {ids[1], ids[2]})
    second["rejected"][0]["findings"] = [{
        "code": "semantic_near_duplicate", "rationale": "Bound pair.",
        "relatedScenarioIds": [ids[2]],
    }]
    second["rejected"][1]["findings"] = [{
        "code": "semantic_near_duplicate", "rationale": "Bound pair.",
        "relatedScenarioIds": [ids[1]],
    }]
    assert finding_claims_from_decisions([first, second], ids) == [
        {"code": "semantic_near_duplicate", "scenarioIds": [ids[0], ids[1]]},
        {"code": "semantic_near_duplicate", "scenarioIds": [ids[1], ids[2]]},
    ]


class _ClaimPlanner:
    def __init__(self, confirmed: bool):
        self.confirmed = confirmed
        self.prompts: list[dict] = []

    def call(self, _system: str, prompt: str, _schema: dict) -> dict:
        import json

        parsed = json.loads(prompt)
        self.prompts.append(parsed)
        code = parsed["proposedFinding"]["code"]
        return {
            "findings": ([{"code": code, "rationale": "Exact claim confirmed."}]
                         if self.confirmed else [])
        }


def _source_blueprints(scenarios: list[dict]) -> list[dict]:
    blueprint_set = _blueprint_sets(scenarios)[0]
    return [
        {"scenarioId": row["scenarioId"], "topicId": "topic_test",
         **blueprint_set["blueprints"][row["scenarioId"]]}
        for row in scenarios
    ]


def _claim_verifier(tmp_path: Path, confirmed: bool):
    verifier = AuthenticDecomposedScenarioAdjudicator(
        PlannerConfig(endpoint="http://unused.invalid/v1/chat/completions",
                      model="arbiter-model", api_key="", temperature=0.0),
        checkpoint_root=tmp_path,
    )
    planner = _ClaimPlanner(confirmed)
    verifier._planner = planner
    return verifier, planner


def test_empty_proposer_union_performs_no_open_audit(tmp_path: Path) -> None:
    topic, scenarios = _group()
    _bind_modes(scenarios)
    verifier, planner = _claim_verifier(tmp_path, True)

    decision = verifier.adjudicate(topic, scenarios, _source_blueprints(scenarios), [])

    assert decision["groupDecision"] == "pass"
    assert planner.prompts == []


def test_only_exact_typed_claim_is_verified_without_proposer_rationale(tmp_path: Path) -> None:
    topic, scenarios = _group()
    _bind_modes(scenarios)
    verifier, planner = _claim_verifier(tmp_path, True)
    claim = {"code": "target_dialogue_like_content", "scenarioIds": [scenarios[0]["scenarioId"]]}

    decision = verifier.adjudicate(
        topic, scenarios, _source_blueprints(scenarios), [claim]
    )

    assert [row["scenarioId"] for row in decision["rejected"]] == [scenarios[0]["scenarioId"]]
    assert len(planner.prompts) == 1
    assert planner.prompts[0]["proposedFinding"] == claim
    assert "rationale" not in planner.prompts[0]["proposedFinding"]


def test_unconfirmed_typed_claim_cannot_reject(tmp_path: Path) -> None:
    topic, scenarios = _group()
    _bind_modes(scenarios)
    verifier, planner = _claim_verifier(tmp_path, False)
    claim = {"code": "unsafe_content", "scenarioIds": [scenarios[0]["scenarioId"]]}

    decision = verifier.adjudicate(
        topic, scenarios, _source_blueprints(scenarios), [claim]
    )

    assert decision["groupDecision"] == "pass"
    assert len(planner.prompts) == 1


def test_pair_finding_requires_exactly_two_source_bound_ids(tmp_path: Path) -> None:
    topic, scenarios = _group()
    _bind_modes(scenarios)
    verifier, _planner = _claim_verifier(tmp_path, True)

    with pytest.raises(ScenarioScrutinyError, match="source-bound"):
        verifier.adjudicate(
            topic,
            scenarios,
            _source_blueprints(scenarios),
            [{"code": "semantic_near_duplicate", "scenarioIds": [scenarios[0]["scenarioId"]]}],
        )
