from __future__ import annotations

from pathlib import Path

import pytest

from ground_truth_finetuning.tests.test_scenario_blueprint_v5 import (
    joint_blueprints,
    topic,
)
from ground_truth_finetuning.training.scenario_blueprint_v5 import (
    AdjudicatedWholeBlueprintJudge,
    AuthenticWholeBlueprintJudge,
    BLUEPRINT_FINDING_CARDINALITY_HASH,
    BLUEPRINT_FINDING_DEFINITIONS,
    BLUEPRINT_JUDGE_SOURCE_HASH,
    BLUEPRINT_PROPOSER_PROMPT_HASH,
    BLUEPRINT_VERIFIER_PROMPT_HASH,
    InvalidModelOutput,
    ScenarioBlueprintError,
    blueprint_judge_protocol_hash,
    canonical_json,
    make_blueprint_set,
    scenario_ids_for_topic,
    validate_blueprint_proposal,
)


class RecordingStrictModel:
    def __init__(self, model: str, output) -> None:
        self.model = model
        self.output = output
        self.calls: list[dict] = []

    def binding(self) -> dict:
        return {
            "protocol": "focused_fake_strict_schema",
            "model": self.model,
            "reasoning": {"enabled": False},
            "responseFormat": "strict_json_schema",
        }

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        value = self.output(kwargs) if callable(self.output) else self.output
        return value, {
            "model": self.model,
            "finishReason": "stop",
            "usage": {"completion_tokens": 1},
        }


def _adjudicator(
    tmp_path: Path,
    primary_output: dict,
    secondary_output: dict,
    confirmed_codes: set[str],
):
    primary_model = RecordingStrictModel(
        "robit/ornith-vision:35b", primary_output
    )
    secondary_model = RecordingStrictModel("robit/ornith:35b", secondary_output)
    verifier_model = RecordingStrictModel(
        "gemma3:27b",
        lambda call: {
            "confirmed": call["context"]["proposedFinding"]["code"]
            in confirmed_codes
        },
    )
    judge = AdjudicatedWholeBlueprintJudge(
        AuthenticWholeBlueprintJudge(primary_model),
        AuthenticWholeBlueprintJudge(secondary_model),
        verifier_model,
        checkpoint_root=tmp_path / "claims",
        max_workers=1,
    )
    return judge, primary_model, secondary_model, verifier_model


def test_adjudication_unions_claims_and_verifies_only_local_evidence(
    tmp_path: Path,
) -> None:
    card = topic()
    blueprint_set = make_blueprint_set(card, joint_blueprints())
    ids = blueprint_set["scenarioIds"]
    duplicate = {
        "code": "semantic_near_duplicate_cluster",
        "scenarioIds": [ids[1], ids[0]],
    }
    primary = {
        "findings": [
            duplicate,
            {"code": "incoherent_niche", "scenarioIds": [ids[2]]},
        ]
    }
    secondary = {
        "findings": [
            {
                "code": "semantic_near_duplicate_cluster",
                "scenarioIds": [ids[0], ids[1]],
            },
            {"code": "field_role_collapse", "scenarioIds": [ids[3]]},
        ]
    }
    judge, primary_model, secondary_model, verifier_model = _adjudicator(
        tmp_path,
        primary,
        secondary,
        {"semantic_near_duplicate_cluster", "field_role_collapse"},
    )

    decision, metadata = judge.judge_topic(card, blueprint_set)

    assert decision["groupDecision"] == "reject"
    assert decision["findings"] == sorted(
        [
            {
                "code": "semantic_near_duplicate_cluster",
                "scenarioIds": [ids[0], ids[1]],
                "rationale": BLUEPRINT_FINDING_DEFINITIONS[
                    "semantic_near_duplicate_cluster"
                ],
            },
            {
                "code": "field_role_collapse",
                "scenarioIds": [ids[3]],
                "rationale": BLUEPRINT_FINDING_DEFINITIONS["field_role_collapse"],
            },
        ],
        key=canonical_json,
    )
    assert metadata["proposedClaimCount"] == 3
    assert metadata["confirmedClaimCount"] == 2
    assert len(verifier_model.calls) == 3

    for proposer_call in primary_model.calls + secondary_model.calls:
        assert proposer_call["schema"]["required"] == ["findings"]
        assert "rationale" not in canonical_json(proposer_call["schema"])
        assert proposer_call["context"]["typedFindingContract"][
            "proposerRationaleAllowed"
        ] is False
    for verifier_call in verifier_model.calls:
        context = verifier_call["context"]
        claim_ids = context["proposedFinding"]["scenarioIds"]
        evidence = context["sourceBoundBlueprintEvidence"]
        assert evidence["scenarioIds"] == claim_ids
        assert set(evidence["blueprints"]) == set(claim_ids)
        assert "jointCompactBlueprint" not in context

    binding = judge.binding()
    assert binding["roleModelBindings"]["primaryProposer"]["model"] == (
        "robit/ornith-vision:35b"
    )
    assert binding["roleModelBindings"]["secondaryProposer"]["model"] == (
        "robit/ornith:35b"
    )
    assert binding["roleModelBindings"]["evidenceBoundVerifier"]["model"] == (
        "gemma3:27b"
    )
    assert binding["protocolHash"] == blueprint_judge_protocol_hash()
    assert binding["sourceHash"] == BLUEPRINT_JUDGE_SOURCE_HASH
    assert binding["cardinalityHash"] == BLUEPRINT_FINDING_CARDINALITY_HASH
    assert binding["promptHashes"] == {
        "primaryProposer": BLUEPRINT_PROPOSER_PROMPT_HASH,
        "secondaryProposer": BLUEPRINT_PROPOSER_PROMPT_HASH,
        "evidenceBoundVerifier": BLUEPRINT_VERIFIER_PROMPT_HASH,
    }

    checkpoint_count = len(list((tmp_path / "claims").rglob("*.json")))
    assert checkpoint_count == 3
    judge.judge_topic(card, blueprint_set)
    assert len(verifier_model.calls) == 3


@pytest.mark.parametrize(
    "finding",
    [
        {
            "code": "semantic_near_duplicate_cluster",
            "scenarioIds": ["scenario_topic_alpha_01"],
        },
        {
            "code": "incoherent_niche",
            "scenarioIds": [
                "scenario_topic_alpha_01",
                "scenario_topic_alpha_02",
            ],
        },
        {
            "code": "field_role_collapse",
            "scenarioIds": ["scenario_stale_01"],
        },
        {
            "code": "target_field_leakage",
            "scenarioIds": ["scenario_topic_alpha_01"],
            "rationale": "Proposer prose is forbidden.",
        },
    ],
)
def test_typed_proposal_rejects_wrong_ids_prose_and_code_cardinality(
    finding: dict,
) -> None:
    with pytest.raises(InvalidModelOutput, match="violates strict schema"):
        validate_blueprint_proposal(
            {"findings": [finding]}, scenario_ids_for_topic("topic_alpha")
        )


def test_adjudicator_requires_three_distinct_role_model_bindings(
    tmp_path: Path,
) -> None:
    primary = AuthenticWholeBlueprintJudge(
        RecordingStrictModel("robit/ornith:35b", {"findings": []})
    )
    secondary = AuthenticWholeBlueprintJudge(
        RecordingStrictModel("robit/ornith:35b", {"findings": []})
    )
    verifier = RecordingStrictModel("gemma3:27b", {"confirmed": False})

    with pytest.raises(ScenarioBlueprintError, match="distinct model bindings"):
        AdjudicatedWholeBlueprintJudge(
            primary,
            secondary,
            verifier,
            checkpoint_root=tmp_path / "claims",
        )
