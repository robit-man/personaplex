from __future__ import annotations

from ground_truth_finetuning.tools.evaluate_nemotron_context_contract import (
    ATOMIC_QUALITY_FINDING_CODES,
    atomic_payload_for,
    atomic_response_schema,
    cases,
    context_for,
    response_schema,
    semantic_match,
)


def test_context_cases_have_causal_id_rebinding_and_clean_controls() -> None:
    by_id = {case["caseId"]: case for case in cases()}
    first = by_id["typed_quality_defects"]["requiredClaims"]
    second = by_id["counterfactual_id_rebinding"]["requiredClaims"]
    assert set(first) == set(second)
    assert all(first[code] != second[code] for code in first)
    assert by_id["clean_pass"]["requiredClaims"] == {}


def test_semantic_match_requires_expected_claims_and_protects_clean_ids() -> None:
    case = next(item for item in cases() if item["caseId"] == "typed_quality_defects")
    exact = {
        "findingClusters": [
            {"code": code, "scenarioIds": ids}
            for code, ids in case["requiredClaims"].items()
        ]
    }
    assert semantic_match(exact, case) == (True, [])
    missing = {"findingClusters": exact["findingClusters"][:-1]}
    assert semantic_match(missing, case)[0] is False
    false_positive = {
        "findingClusters": [
            *exact["findingClusters"],
            {"code": "implausible_anchor", "scenarioIds": case["cleanIds"]},
        ]
    }
    assert semantic_match(false_positive, case)[0] is False


def test_ab_schema_accepts_the_expected_typed_contract() -> None:
    ids = sorted(cases()[0]["bindings"])
    schema = response_schema(ids)
    assert schema["properties"]["findingClusters"]["maxItems"] >= 3


def test_atomic_contract_is_one_code_source_bound_and_reasoning_off() -> None:
    case = cases()[1]
    ids = sorted(case["bindings"])
    schema = atomic_response_schema()
    payload = atomic_payload_for(
        model="nemotron-3-super:120b",
        code="incomplete_or_malformed_field",
        scenario_id=ids[2],
        context=context_for(case),
    )
    assert schema["required"] == ["confirmed"]
    assert payload["reasoning_effort"] == "none"
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["temperature"] == 0.0
    assert "incomplete_or_malformed_field" in payload["messages"][1]["content"]
    assert "otherFindingCodesAreOutOfScope" in payload["messages"][1]["content"]
    assert ids[2] in payload["messages"][1]["content"]
    assert set(ATOMIC_QUALITY_FINDING_CODES) == {
        "language_or_encoding_corruption",
        "incomplete_or_malformed_field",
        "unnatural_or_placeholder_content",
    }
