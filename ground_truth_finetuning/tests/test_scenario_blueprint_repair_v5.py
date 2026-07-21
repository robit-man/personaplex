from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from ground_truth_finetuning.tests.test_scenario_blueprint_v5 import (
    PassingTaxonomyJudge,
    ScriptedJudge,
    ScriptedPlanner,
    joint_blueprints,
    passing_judgment,
    request,
    taxonomy_anchors,
    topic,
)
from ground_truth_finetuning.training.scenario_blueprint_repair_v5 import (
    admit_blueprints,
    build_blueprint_repair_response_schema,
    repair_ids_from_scrutiny,
    repair_topic_blueprints,
)
from ground_truth_finetuning.training.scenario_blueprint_v5 import (
    BLUEPRINT_SETS_FILENAME,
    BLUEPRINT_SCRUTINY_FILENAME,
    ScenarioBlueprintError,
    encode_blueprint_response,
    generate_topic_blueprint_scrutiny,
    make_blueprint_set,
    scenario_ids_for_topic,
)
from ground_truth_finetuning.training.scenario_taxonomy_admission_v5 import (
    TAXONOMY_FIELDS,
)


class FunctionalCascadePlanner:
    def __init__(self) -> None:
        self.calls = []

    def binding(self) -> dict:
        return {
            "protocol": "functional_fake_strict_schema",
            "model": "functional-cascade-model",
            "reasoning": {"enabled": False},
            "responseFormat": "strict_json_schema",
        }

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        card = kwargs["context"]["topicCard"]
        topic_id = card["topicId"]
        if kwargs["name"] == "personaplex_scenario_taxonomy_v5":
            from ground_truth_finetuning.tests.test_scenario_blueprint_v5 import taxonomy_wire
            value = taxonomy_wire(topic_id)
        elif kwargs["name"] == "personaplex_scenario_blueprints_v5":
            value = encode_blueprint_response(
                joint_blueprints(topic_id),
                card,
                taxonomy_anchors=taxonomy_anchors(topic_id),
            )
        else:
            raise AssertionError(f"unexpected functional planner call: {kwargs['name']}")
        return value, {"finishReason": "stop", "usage": {"completion_tokens": 512}}


def rejecting_scrutiny(tmp_path: Path, card: dict, blueprint_set: dict) -> dict:
    ids = scenario_ids_for_topic(card["topicId"])
    decision = passing_judgment()
    decision["findings"] = [
        {
            "code": "semantic_near_duplicate_cluster",
            "scenarioIds": list(ids[:2]),
            "rationale": "The two niches share the same semantic situation and causal resolution.",
        }
    ]
    return generate_topic_blueprint_scrutiny(
        request=request(),
        topic=card,
        blueprint_set=blueprint_set,
        output_root=tmp_path,
        judge=ScriptedJudge([decision]),
        max_attempts=1,
    )


def repair_response(card: dict, blueprint_set: dict, repair_ids: tuple[str, ...]) -> dict:
    repaired = deepcopy(blueprint_set["blueprints"])
    replacements = (
        ("repair evidence alpha", "repair mechanism alpha"),
        ("repair evidence beta", "repair mechanism beta"),
    )
    fields = ("evidencePivot", "causalMechanism")
    for scenario_id, values in zip(repair_ids, replacements, strict=True):
        for field, value in zip(fields, values, strict=True):
            repaired[scenario_id][field] = value
        repaired[scenario_id]["fourSiblingAffordance"] = {
            "verifiedPositive": f"positive repair {scenario_id[-2:]}",
            "verifiedNegative": f"negative repair {scenario_id[-2:]}",
            "uncertain": f"uncertain repair {scenario_id[-2:]}",
            "superseded": f"superseded repair {scenario_id[-2:]}",
        }
    encoded = encode_blueprint_response(
        repaired, card, taxonomy_anchors=taxonomy_anchors(card["topicId"])
    )
    return {scenario_id: encoded[scenario_id] for scenario_id in repair_ids}


def test_repair_schema_contains_only_authentic_finding_ids_without_prefix_items(tmp_path: Path) -> None:
    card = topic()
    blueprint_set = make_blueprint_set(card, joint_blueprints())
    scrutiny = rejecting_scrutiny(tmp_path, card, blueprint_set)
    repair_ids = repair_ids_from_scrutiny(blueprint_set, scrutiny)
    schema = build_blueprint_repair_response_schema(card, blueprint_set, repair_ids)

    assert schema["required"] == list(repair_ids)
    assert set(schema["properties"]) == set(repair_ids)
    assert "prefixItems" not in str(schema)


def test_repair_changes_only_rejected_slots_and_persists_lineage(tmp_path: Path) -> None:
    card = topic()
    blueprint_set = make_blueprint_set(card, joint_blueprints())
    scrutiny = rejecting_scrutiny(tmp_path, card, blueprint_set)
    repair_ids = repair_ids_from_scrutiny(blueprint_set, scrutiny)
    planner = ScriptedPlanner([repair_response(card, blueprint_set, repair_ids)])

    checkpoint = repair_topic_blueprints(
        request=request(), topic=card, blueprint_set=blueprint_set,
        blueprint_scrutiny=scrutiny, output_root=tmp_path, planner=planner,
        repair_cycle=1, max_attempts=1,
    )
    repaired = checkpoint["repairedBlueprintSet"]
    for scenario_id in blueprint_set["scenarioIds"]:
        if scenario_id in repair_ids:
            assert repaired["blueprints"][scenario_id] != blueprint_set["blueprints"][scenario_id]
            for field in TAXONOMY_FIELDS:
                assert (
                    repaired["blueprints"][scenario_id][field]
                    == blueprint_set["blueprints"][scenario_id][field]
                )
        else:
            assert repaired["blueprints"][scenario_id] == blueprint_set["blueprints"][scenario_id]
    assert checkpoint["sourceScrutinyHash"] == scrutiny["checkpointHash"]
    assert checkpoint["parentJointBlueprintHash"] == blueprint_set["jointBlueprintHash"]


def test_repair_resume_makes_no_model_call(tmp_path: Path) -> None:
    card = topic()
    blueprint_set = make_blueprint_set(card, joint_blueprints())
    scrutiny = rejecting_scrutiny(tmp_path, card, blueprint_set)
    repair_ids = repair_ids_from_scrutiny(blueprint_set, scrutiny)
    first = repair_topic_blueprints(
        request=request(), topic=card, blueprint_set=blueprint_set,
        blueprint_scrutiny=scrutiny, output_root=tmp_path,
        planner=ScriptedPlanner([repair_response(card, blueprint_set, repair_ids)]),
        repair_cycle=1, max_attempts=1,
    )
    no_call = ScriptedPlanner([])
    resumed = repair_topic_blueprints(
        request=request(), topic=card, blueprint_set=blueprint_set,
        blueprint_scrutiny=scrutiny, output_root=tmp_path, planner=no_call,
        repair_cycle=1, max_attempts=1, resume=True,
    )
    assert resumed == first
    assert no_call.calls == []


def test_repair_refuses_passing_scrutiny(tmp_path: Path) -> None:
    card = topic()
    blueprint_set = make_blueprint_set(card, joint_blueprints())
    passing = generate_topic_blueprint_scrutiny(
        request=request(), topic=card, blueprint_set=blueprint_set,
        output_root=tmp_path, judge=ScriptedJudge(), max_attempts=1,
    )
    with pytest.raises(ScenarioBlueprintError, match="rejecting scrutiny"):
        repair_topic_blueprints(
            request=request(), topic=card, blueprint_set=blueprint_set,
            blueprint_scrutiny=passing, output_root=tmp_path,
            planner=ScriptedPlanner([]), repair_cycle=1, max_attempts=1,
        )


def test_admit_blueprints_writes_canonical_outputs_only_after_exact_50_topic_pass(
    tmp_path: Path,
) -> None:
    topics = [topic(f"topic_{index:02d}") for index in range(50)]
    planner = FunctionalCascadePlanner()
    final_sets, final_scrutinies = admit_blueprints(
        request=request(),
        topics=topics,
        output_root=tmp_path,
        planner=planner,
        taxonomy_judge=PassingTaxonomyJudge(),
        judge=ScriptedJudge(),
        max_workers=3,
        judge_workers=1,
        max_attempts=1,
        max_repair_cycles=1,
    )

    assert len(final_sets) == len(final_scrutinies) == 50
    assert all(item["decision"]["groupDecision"] == "pass" for item in final_scrutinies)
    assert len((tmp_path / BLUEPRINT_SETS_FILENAME).read_text().splitlines()) == 50
    assert len((tmp_path / BLUEPRINT_SCRUTINY_FILENAME).read_text().splitlines()) == 50
