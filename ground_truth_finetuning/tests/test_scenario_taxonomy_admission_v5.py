from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ground_truth_finetuning.tests.test_scenario_blueprint_v5 import (
    ScriptedPlanner,
    encode_blueprint_response,
    joint_blueprints,
    request,
    taxonomy_anchors,
    taxonomy_wire,
    topic,
)
from ground_truth_finetuning.training.scenario_blueprint_v5 import (
    TAXONOMY_FIELDS,
    TAXONOMY_WIRE_KEYS,
    ScenarioBlueprintError,
    canonical_json,
    generate_topic_blueprints,
    scenario_ids_for_topic,
)
from ground_truth_finetuning.training.scenario_taxonomy_admission_v5 import (
    ATOMIC_QUALITY_TAXONOMY_FINDING_CODES,
    AdjudicatedTaxonomyJudge,
    MIN_REPAIRED_ANCHOR_FIELDS,
    TAXONOMY_FINDING_CODES,
    TAXONOMY_JUDGE_SOURCE_HASH,
    TAXONOMY_REPAIR_SOURCE_HASH,
    _full_set_taxonomy_judgment,
    _repair_ids_for_current_judgment,
    admit_topic_taxonomy,
    build_taxonomy_judge_response_schema,
    build_taxonomy_judge_view,
    repair_ids_from_taxonomy_judgment,
    taxonomy_admission_protocol_hash,
    validate_taxonomy_judgment,
    validate_taxonomy_proposal,
)


class ScriptedTaxonomyJudge:
    def __init__(self, decisions=(), events: list[str] | None = None) -> None:
        self.decisions = list(decisions)
        self.calls: list[dict] = []
        self.events = events

    def binding(self) -> dict:
        return {
            "protocol": "independent_fake_taxonomy_judge",
            "modelBinding": {
                "protocol": "fake_strict_schema",
                "model": "independent-taxonomy-judge-model",
                "reasoning": {"enabled": False},
                "responseFormat": "strict_json_schema",
            },
        }

    def judge_taxonomy(self, topic_card, judge_view, *, retry_feedback=None):
        if self.events is not None:
            self.events.append("judge")
        self.calls.append(deepcopy(judge_view))
        if not self.decisions:
            raise AssertionError("unexpected taxonomy judge call")
        decision = self.decisions.pop(0)
        if isinstance(decision, BaseException):
            raise decision
        return decision, {
            "model": "independent-taxonomy-judge-model",
            "finishReason": "stop",
        }


class EventPlanner(ScriptedPlanner):
    def __init__(self, outputs, events: list[str]) -> None:
        super().__init__(outputs)
        self.events = events

    def generate(self, **kwargs):
        self.events.append(kwargs["name"])
        return super().generate(**kwargs)


class SameBindingTaxonomyJudge(ScriptedTaxonomyJudge):
    def binding(self) -> dict:
        return {"protocol": "same", "modelBinding": ScriptedPlanner([]).binding()}


class NamedTaxonomyJudge(ScriptedTaxonomyJudge):
    def __init__(self, model: str, decisions=()) -> None:
        super().__init__(decisions)
        self.model = model

    def binding(self) -> dict:
        return {
            "protocol": "named_fake_taxonomy_judge",
            "modelBinding": {
                "protocol": "fake_strict_schema",
                "model": self.model,
                "reasoning": {"enabled": False},
                "responseFormat": "strict_json_schema",
            },
        }


class LowUsageScriptedPlanner(ScriptedPlanner):
    def generate(self, **kwargs):
        value, metadata = super().generate(**kwargs)
        metadata["usage"]["completion_tokens"] = 1
        return value, metadata


def taxonomy_repair_wire(
    card: dict, repair_ids: tuple[str, ...], *, changed_fields: int
) -> dict:
    wire = taxonomy_anchors(card["topicId"])
    declared_fields = [
        "submode",
        "participantRelationship",
        "setting",
        "centralResource",
        "centralTension",
    ][:changed_fields]
    for index, scenario_id in enumerate(repair_ids):
        wire[scenario_id]["submode"] = (
            f"repaired submode {index:02d}"
        )
        if changed_fields >= 2:
            wire[scenario_id]["participantRelationship"] = (
                f"repaired relationship {index:02d}"
            )
        if changed_fields >= 3:
            wire[scenario_id]["setting"] = (
                f"repaired setting {index:02d}"
            )
        if changed_fields >= 4:
            wire[scenario_id]["centralResource"] = (
                f"repaired resource {index:02d}"
            )
        if changed_fields >= 5:
            wire[scenario_id]["centralTension"] = (
                f"repaired tension {index:02d}"
            )
        wire[scenario_id]["changedFields"] = declared_fields
    return {scenario_id: wire[scenario_id] for scenario_id in repair_ids}


def test_judge_view_binds_modes_and_typed_clusters_are_the_only_signal() -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    view = build_taxonomy_judge_view(card, taxonomy_anchors(card["topicId"]))

    assert view["scenarioIds"] == list(ids)
    for ordinal, scenario_id in enumerate(ids):
        binding = view["scenarioBindings"][scenario_id]
        assert set(binding) == {"interactionMode", *TAXONOMY_FIELDS}
        assert binding["interactionMode"] == card["interactionModes"][
            ordinal % len(card["interactionModes"])
        ]
        assert {
            field: binding[field] for field in TAXONOMY_FIELDS
        } == taxonomy_anchors(card["topicId"])[scenario_id]

    findings = {
        "findingClusters": [
            {"code": "mode_submode_mismatch", "scenarioIds": [ids[0]]},
            {"code": "field_role_misuse", "scenarioIds": [ids[1]]},
            {
                "code": "semantic_duplicate_template_collapse",
                "scenarioIds": [ids[3], ids[2]],
            },
            {"code": "implausible_anchor", "scenarioIds": [ids[4]]},
        ]
    }
    normalized = validate_taxonomy_judgment(findings, ids)
    assert {
        finding["code"] for finding in normalized["findingClusters"]
    }.issubset(set(TAXONOMY_FINDING_CODES))
    assert repair_ids_from_taxonomy_judgment(normalized, ids) == ids[:5]
    assert build_taxonomy_judge_response_schema(ids)["required"] == [
        "findingClusters"
    ]

    with pytest.raises(ScenarioBlueprintError, match="violates strict schema"):
        validate_taxonomy_judgment(
            {"findingClusters": [], "groupDecision": "pass"}, ids
        )


def test_two_proposer_claim_union_requires_independent_source_bound_confirmation(
    tmp_path: Path,
) -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    view = build_taxonomy_judge_view(card, taxonomy_anchors(card["topicId"]))
    duplicate = {
        "code": "semantic_duplicate_template_collapse",
        "scenarioIds": [ids[1], ids[2]],
    }
    primary = NamedTaxonomyJudge("primary-taxonomy-model", [{
        "findingClusters": [
            {"code": "mode_submode_mismatch", "scenarioIds": [ids[0]]},
            duplicate,
        ]
    }])
    secondary = NamedTaxonomyJudge("secondary-taxonomy-model", [{
        "findingClusters": [
            duplicate,
            {"code": "field_role_misuse", "scenarioIds": [ids[3]]},
        ]
    }])
    verifier = LowUsageScriptedPlanner([
        lambda call: {"confirmed": call["context"]["proposedFinding"]["code"] != "field_role_misuse"},
        lambda call: {"confirmed": call["context"]["proposedFinding"]["code"] != "field_role_misuse"},
        lambda call: {"confirmed": call["context"]["proposedFinding"]["code"] != "field_role_misuse"},
    ])
    judge = AdjudicatedTaxonomyJudge(
        primary,
        secondary,
        verifier,
        checkpoint_root=tmp_path / "claims",
        max_workers=1,
    )

    decision, metadata = judge.judge_taxonomy(card, view)

    assert decision == {"findingClusters": [
        {"code": "mode_submode_mismatch", "scenarioIds": [ids[0]]},
        duplicate,
    ]}
    assert metadata["proposedClaimCount"] == 3
    assert metadata["confirmedClaimCount"] == 2
    assert len(verifier.calls) == 3
    assert all(
        set(call["context"]["proposedFinding"]) == {"code", "scenarioIds"}
        for call in verifier.calls
    )


def test_atomic_quality_is_checked_per_scenario_then_independently_confirmed(
    tmp_path: Path,
) -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    view = build_taxonomy_judge_view(card, taxonomy_anchors(card["topicId"]))
    target_id = ids[4]
    target_code = "unnatural_or_placeholder_content"
    primary = NamedTaxonomyJudge("primary-taxonomy-model", [{"findingClusters": []}])
    secondary = NamedTaxonomyJudge("secondary-taxonomy-model", [{"findingClusters": []}])

    def quality_decision(call):
        context = call["context"]
        return {
            "confirmed": (
                context["scenarioId"] == target_id
                and context["proposedFindingCode"] == target_code
            )
        }

    quality_call_count = len(ids) * len(ATOMIC_QUALITY_TAXONOMY_FINDING_CODES)
    quality = LowUsageScriptedPlanner(
        [quality_decision for _ in range(quality_call_count)]
    )
    verifier = LowUsageScriptedPlanner([{"confirmed": True}])
    judge = AdjudicatedTaxonomyJudge(
        primary,
        secondary,
        verifier,
        quality_model=quality,
        checkpoint_root=tmp_path / "claims",
        max_workers=1,
        quality_workers=3,
    )

    decision, metadata = judge.judge_taxonomy(card, view)

    assert decision == {
        "findingClusters": [{"code": target_code, "scenarioIds": [target_id]}]
    }
    assert len(quality.calls) == quality_call_count
    assert len(verifier.calls) == 1
    assert metadata["atomicQualityCheckCount"] == quality_call_count
    assert metadata["atomicQualityConfirmedCount"] == 1
    assert len(metadata["atomicQualityCheckpointHashes"]) == quality_call_count


def test_weaker_relational_claim_requires_focused_strong_model_scrutiny(
    tmp_path: Path,
) -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    view = build_taxonomy_judge_view(card, taxonomy_anchors(card["topicId"]))
    claim = {
        "code": "semantic_duplicate_template_collapse",
        "scenarioIds": [ids[1], ids[2]],
    }
    primary = NamedTaxonomyJudge("primary-taxonomy-model", [{"findingClusters": []}])
    secondary = NamedTaxonomyJudge(
        "secondary-taxonomy-model", [{"findingClusters": [claim]}]
    )
    quality_call_count = len(ids) * len(ATOMIC_QUALITY_TAXONOMY_FINDING_CODES)
    quality = LowUsageScriptedPlanner(
        [lambda _call: {"confirmed": False} for _ in range(quality_call_count + 1)]
    )
    verifier = LowUsageScriptedPlanner([])
    judge = AdjudicatedTaxonomyJudge(
        primary,
        secondary,
        verifier,
        quality_model=quality,
        checkpoint_root=tmp_path / "claims",
        max_workers=1,
        quality_workers=3,
    )

    decision, metadata = judge.judge_taxonomy(card, view)

    assert decision == {"findingClusters": []}
    assert len(quality.calls) == quality_call_count + 1
    assert len(verifier.calls) == 0
    assert metadata["relationalProposedClaimCount"] == 1
    assert metadata["relationalScrutinyConfirmedCount"] == 0
    assert len(metadata["relationalScrutinyCheckpointHashes"]) == 1
    assert metadata["proposedClaimCount"] == 0


def test_singleton_duplicate_proposal_reaches_scrutiny_but_never_repair(
    tmp_path: Path,
) -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    view = build_taxonomy_judge_view(card, taxonomy_anchors(card["topicId"]))
    malformed_claim = {
        "code": "semantic_duplicate_template_collapse",
        "scenarioIds": [ids[7]],
    }
    assert validate_taxonomy_proposal(
        {"findingClusters": [malformed_claim]}, ids
    ) == {"findingClusters": [malformed_claim]}
    with pytest.raises(ScenarioBlueprintError, match="at least two IDs"):
        validate_taxonomy_judgment({"findingClusters": [malformed_claim]}, ids)

    primary = NamedTaxonomyJudge("primary-taxonomy-model", [{"findingClusters": []}])
    secondary = NamedTaxonomyJudge(
        "secondary-taxonomy-model", [{"findingClusters": [malformed_claim]}]
    )
    quality_call_count = len(ids) * len(ATOMIC_QUALITY_TAXONOMY_FINDING_CODES)
    quality = LowUsageScriptedPlanner(
        [lambda _call: {"confirmed": False} for _ in range(quality_call_count + 1)]
    )
    verifier = LowUsageScriptedPlanner([])
    judge = AdjudicatedTaxonomyJudge(
        primary,
        secondary,
        verifier,
        quality_model=quality,
        checkpoint_root=tmp_path / "claims",
        max_workers=1,
        quality_workers=3,
    )

    decision, metadata = judge.judge_taxonomy(card, view)

    assert decision == {"findingClusters": []}
    assert metadata["relationalProposedClaimCount"] == 1
    assert metadata["relationalScrutinyConfirmedCount"] == 0
    assert metadata["proposedClaimCount"] == 0
    assert len(verifier.calls) == 0


def test_atomic_quality_detection_enters_repair_despite_witness_false_negative(
    tmp_path: Path,
) -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    view = build_taxonomy_judge_view(card, taxonomy_anchors(card["topicId"]))
    target_id = ids[6]
    target_code = "incomplete_or_malformed_field"
    primary = NamedTaxonomyJudge("primary-taxonomy-model", [{"findingClusters": []}])
    secondary = NamedTaxonomyJudge("secondary-taxonomy-model", [{"findingClusters": []}])

    def quality_decision(call):
        context = call["context"]
        return {
            "confirmed": (
                context["scenarioId"] == target_id
                and context["proposedFindingCode"] == target_code
            )
        }

    quality_call_count = len(ids) * len(ATOMIC_QUALITY_TAXONOMY_FINDING_CODES)
    quality = LowUsageScriptedPlanner(
        [quality_decision for _ in range(quality_call_count)]
    )
    witness = LowUsageScriptedPlanner([{"confirmed": False}])
    verifier = LowUsageScriptedPlanner([{"confirmed": False}])
    judge = AdjudicatedTaxonomyJudge(
        primary,
        secondary,
        verifier,
        quality_model=quality,
        quality_witness_model=witness,
        checkpoint_root=tmp_path / "claims",
        max_workers=1,
        quality_workers=3,
    )

    decision, metadata = judge.judge_taxonomy(card, view)

    assert decision == {
        "findingClusters": [{"code": target_code, "scenarioIds": [target_id]}]
    }
    assert len(witness.calls) == 1
    assert len(verifier.calls) == 1
    assert metadata["atomicWitnessCheckCount"] == 1
    assert metadata["atomicWitnessConfirmedCount"] == 0
    assert metadata["independentVerifierConfirmedCount"] == 0
    assert metadata["confirmedClaimCount"] == 1


def test_repair_lineage_never_suppresses_newly_exposed_findings() -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    source_checkpoint = {"repairScenarioIds": [ids[0]]}
    judgment = {"findingClusters": [
        {"code": "field_role_misuse", "scenarioIds": [ids[0]]},
        {"code": "mode_submode_mismatch", "scenarioIds": [ids[1]]},
        {
            "code": "semantic_duplicate_template_collapse",
            "scenarioIds": [ids[0], ids[2]],
        },
    ]}

    normalized = _full_set_taxonomy_judgment(judgment, ids)

    assert normalized == {"findingClusters": [
        {"code": "field_role_misuse", "scenarioIds": [ids[0]]},
        {"code": "mode_submode_mismatch", "scenarioIds": [ids[1]]},
        {
            "code": "semantic_duplicate_template_collapse",
            "scenarioIds": [ids[0], ids[2]],
        },
    ]}
    assert _repair_ids_for_current_judgment(normalized, ids) == ids[:3]


def test_targeted_repair_preserves_bytes_reruns_full_set_and_resumes(
    tmp_path: Path,
) -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    repair_ids = ids[:2]
    finding = {
        "findingClusters": [
            {
                "code": "semantic_duplicate_template_collapse",
                "scenarioIds": list(repair_ids),
            }
        ]
    }
    events: list[str] = []
    planner = EventPlanner(
        [
            taxonomy_wire(card["topicId"]),
            taxonomy_repair_wire(card, repair_ids, changed_fields=1),
            taxonomy_repair_wire(card, repair_ids, changed_fields=len(TAXONOMY_FIELDS)),
        ],
        events,
    )
    judge = ScriptedTaxonomyJudge([finding, {"findingClusters": []}], events)

    admitted = admit_topic_taxonomy(
        request=request(),
        topic=card,
        output_root=tmp_path,
        planner=planner,
        judge=judge,
        max_attempts=2,
        max_repair_cycles=1,
    )

    raw = taxonomy_anchors(card["topicId"])
    for scenario_id in ids:
        before = canonical_json(raw[scenario_id]).encode("utf-8")
        after = canonical_json(admitted["taxonomyAnchors"][scenario_id]).encode("utf-8")
        if scenario_id in repair_ids:
            changed = [
                field
                for field in TAXONOMY_FIELDS
                if raw[scenario_id][field]
                != admitted["taxonomyAnchors"][scenario_id][field]
            ]
            assert len(changed) >= MIN_REPAIRED_ANCHOR_FIELDS
        else:
            assert after == before
    assert events == [
        "personaplex_scenario_taxonomy_v5",
        "judge",
        "personaplex_scenario_taxonomy_repair_v5",
        "personaplex_scenario_taxonomy_repair_v5",
        "judge",
    ]
    assert len(judge.calls) == 2
    assert all(call["scenarioIds"] == list(ids) for call in judge.calls)
    assert admitted["admittedSourceType"] == "repair"
    assert len(admitted["judgmentCheckpointHashes"]) == 2
    assert len(admitted["repairCheckpointHashes"]) == 1
    assert admitted["taxonomyJudgeSourceHash"] == TAXONOMY_JUDGE_SOURCE_HASH
    assert admitted["taxonomyRepairSourceHash"] == TAXONOMY_REPAIR_SOURCE_HASH

    no_call_planner = ScriptedPlanner([])
    no_call_judge = ScriptedTaxonomyJudge([])
    resumed = admit_topic_taxonomy(
        request=request(),
        topic=card,
        output_root=tmp_path,
        planner=no_call_planner,
        judge=no_call_judge,
        max_attempts=2,
        max_repair_cycles=1,
        resume=True,
    )
    assert resumed == admitted
    assert no_call_planner.calls == []
    assert no_call_judge.calls == []


def test_stage_p_waits_for_admission_and_binds_protocol_model_source_hashes(
    tmp_path: Path,
) -> None:
    card = topic()
    events: list[str] = []
    planner = EventPlanner(
        [
            taxonomy_wire(card["topicId"]),
            encode_blueprint_response(
                joint_blueprints(card["topicId"]),
                card,
                taxonomy_anchors=taxonomy_anchors(card["topicId"]),
            ),
        ],
        events,
    )
    judge = ScriptedTaxonomyJudge([{"findingClusters": []}], events)

    checkpoint = generate_topic_blueprints(
        request=request(),
        topic=card,
        output_root=tmp_path,
        planner=planner,
        taxonomy_judge=judge,
        max_attempts=1,
    )

    assert events == [
        "personaplex_scenario_taxonomy_v5",
        "judge",
        "personaplex_scenario_blueprints_v5",
    ]
    assert checkpoint["taxonomyAdmissionProtocolHash"] == (
        taxonomy_admission_protocol_hash()
    )
    assert checkpoint["taxonomyAdmissionCheckpointHash"].startswith("sha256:")
    assert checkpoint["taxonomyPlannerBindingHash"].startswith("sha256:")
    assert checkpoint["taxonomyJudgeModelBindingHash"].startswith("sha256:")
    assert checkpoint["taxonomySourceCheckpointHash"].startswith("sha256:")
    assert checkpoint["taxonomyJudgeSourceHash"] == TAXONOMY_JUDGE_SOURCE_HASH
    assert checkpoint["taxonomyRepairSourceHash"] == TAXONOMY_REPAIR_SOURCE_HASH
    stage_p_context = planner.calls[-1]["context"]
    assert stage_p_context["boundBranchTaxonomyAdmissionHash"] == (
        checkpoint["taxonomyAdmissionCheckpointHash"]
    )

    unused = ScriptedPlanner([])
    with pytest.raises(ScenarioBlueprintError, match="independently judged"):
        generate_topic_blueprints(
            request=request(),
            topic=topic("topic_unadmitted"),
            output_root=tmp_path / "unadmitted",
            planner=unused,
            max_attempts=1,
        )
    assert unused.calls == []


def test_taxonomy_judge_binding_must_be_independent_before_stage_t_call(
    tmp_path: Path,
) -> None:
    planner = ScriptedPlanner([taxonomy_wire()])
    judge = SameBindingTaxonomyJudge([{"findingClusters": []}])

    with pytest.raises(ScenarioBlueprintError, match="independently bound"):
        admit_topic_taxonomy(
            request=request(),
            topic=topic(),
            output_root=tmp_path,
            planner=planner,
            judge=judge,
            max_attempts=1,
        )
    assert planner.calls == []
    assert judge.calls == []
