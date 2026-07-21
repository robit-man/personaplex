from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from threading import Lock

import pytest

from ground_truth_finetuning.training.scenario_blueprint_v5 import (
    BLUEPRINT_JUDGE_DIMENSIONS,
    BLUEPRINT_INITIAL_OUTPUT_TOKENS,
    BLUEPRINT_MAX_OUTPUT_TOKENS,
    BLUEPRINTS_PER_TOPIC,
    CONTROL_OPERATORS,
    TAXONOMY_FIELDS,
    TAXONOMY_MAX_OUTPUT_TOKENS,
    TAXONOMY_WIRE_KEYS,
    InvalidModelOutput,
    ScenarioBlueprintError,
    TruncatedModelOutput,
    build_blueprint_response_schema,
    build_expansion_response_schema,
    build_taxonomy_response_schema,
    build_scenario_blueprint_binding_manifest,
    build_scenario_blueprint_bindings,
    canonical_json,
    content_hash,
    decode_blueprint_response,
    decode_taxonomy_response,
    encode_blueprint_response,
    expand_blueprint_slot,
    generate_topic_blueprints,
    generate_topic_blueprint_scrutiny,
    make_blueprint_set,
    scenario_ids_for_topic,
    typed_niche_signature,
    validate_blueprint_response,
    validate_blueprint_judgment,
    validate_blueprint_scrutiny,
    validate_canonical_scenarios,
    validate_corpus_shape,
    validate_expansion_response,
    validate_taxonomy_anchors,
)


def topic(topic_id: str = "topic_alpha") -> dict:
    return {
        "schema": "personaplex.topic-card.v2",
        "topicId": topic_id,
        "domain": f"Broad test domain {topic_id}",
        "interactionModes": ["collaborative", "investigative", "reflective"],
        "safeStakes": ["low", "moderate"],
    }


def request() -> dict:
    return {
        "schema": "personaplex.diverse-corpus-request.v2",
        "strategyVersion": "semantic-control-v5",
        "coverageTarget": {"candidateTopics": 50, "scenariosPerTopic": 20},
        "topicConstraints": {"exclude": ["target dialogue"]},
        "semanticControl": {"targetLeakageProhibited": True},
    }


def niche(index: int, ids: tuple[str, ...]) -> dict:
    return {
        "participantRelationship": f"peer relation {index:02d}",
        "setting": f"shared setting {index:02d}",
        "interactionMode": ("collaborative", "investigative", "reflective")[index % 3],
        "submode": f"submode {index:02d}",
        "centralResource": f"resource {index:02d}",
        "centralTension": f"tension {index:02d}",
        "evidencePivot": f"evidence pivot {index:02d}",
        "controlOperator": CONTROL_OPERATORS[index % len(CONTROL_OPERATORS)],
        "causalMechanism": f"causal mechanism {index:02d}",
        "stakesProfile": ("low", "moderate")[index % 2],
        "outcomeTopology": (
            "evidence_confirmed",
            "evidence_disconfirmed",
            "uncertain_pending",
            "superseded_redirect",
            "constraint_limited",
            "mutual_repair",
        )[index % 6],
        "fourSiblingAffordance": {
            "verifiedPositive": f"positive route {index:02d}",
            "verifiedNegative": f"negative route {index:02d}",
            "uncertain": f"uncertain route {index:02d}",
            "superseded": f"superseded route {index:02d}",
        },
        "duplexOpportunity": (
            "barge_in_repair",
            "brief_overlap",
            "backchannel_then_resume",
            "cancel_and_restart",
            "completed_turn_handoff",
            "clarification_pause",
        )[(index * 5) % 6],
        "semanticDistinctnessFrom": {
            "siblingIds": [ids[(index + 1) % len(ids)]],
            "distinction": f"decisive contrast {index:02d}",
        },
    }


def joint_blueprints(topic_id: str = "topic_alpha") -> dict:
    ids = scenario_ids_for_topic(topic_id)
    return {scenario_id: niche(index, ids) for index, scenario_id in enumerate(ids)}


def taxonomy_anchors(topic_id: str = "topic_alpha") -> dict:
    blueprints = joint_blueprints(topic_id)
    return {
        scenario_id: {field: blueprint[field] for field in TAXONOMY_FIELDS}
        for scenario_id, blueprint in blueprints.items()
    }


def taxonomy_wire(topic_id: str = "topic_alpha") -> dict:
    return {
        scenario_id: {
            TAXONOMY_WIRE_KEYS[field]: anchor[field] for field in TAXONOMY_FIELDS
        }
        for scenario_id, anchor in taxonomy_anchors(topic_id).items()
    }


def scenario_contract(topic_id: str, scenario_id: str, marker: str | None = None) -> dict:
    unique = marker or scenario_id
    return {
        "schema": "personaplex.scenario-contract.v2",
        "scenarioId": scenario_id,
        "topicId": topic_id,
        "mode": "evidence-guided collaboration",
        "premise": f"A compact but fully specified premise for {unique} changes after bounded evidence arrives.",
        "participants": [
            {"role": "caller", "knowledge": f"Caller knows initial state {unique}."},
            {"role": "agent", "knowledge": f"Agent observes bounded evidence {unique}."},
        ],
        "startingState": {
            "knownFacts": [f"One fact is known for {unique}."],
            "uncertainty": [f"One state remains uncertain for {unique}."],
            "policyConstraints": ["Do not claim an unobserved result."],
        },
        "interactionOpportunity": ["Use the evidence update to choose the next safe action."],
        "allowedToolClasses": ["read_only_lookup"],
        "disallowedClaims": ["Do not invent verification."],
        "scenarioOutcomeSpace": [
            "Verified positive route.",
            "Verified negative route.",
            "Uncertain route.",
            "Superseded route.",
        ],
        "requiredControlPhenomena": ["A typed evidence revision changes the next action."],
    }


def expansion_response(topic_id: str, scenario_id: str, blueprint_set: dict) -> dict:
    scenario = scenario_contract(topic_id, scenario_id)
    scenario["mode"] = blueprint_set["blueprints"][scenario_id]["interactionMode"]
    scenario.pop("scenarioOutcomeSpace")
    scenario.pop("requiredControlPhenomena")
    return {
        "scenarioId": scenario_id,
        "topicId": topic_id,
        "blueprintHash": blueprint_set["blueprintHashes"][scenario_id],
        "jointBlueprintHash": blueprint_set["jointBlueprintHash"],
        "scenarioContract": scenario,
    }


class ScriptedPlanner:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls: list[dict] = []
        self._lock = Lock()

    def binding(self) -> dict:
        return {
            "protocol": "fake_strict_schema",
            "model": "authentic-test-model",
            "reasoning": {"enabled": False},
            "responseFormat": "strict_json_schema",
        }

    def generate(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
            if not self.outputs:
                raise AssertionError("unexpected model call")
            output = self.outputs.pop(0)
        if isinstance(output, BaseException):
            raise output
        if callable(output):
            output = output(kwargs)
        return output, {"finishReason": "stop", "usage": {"completion_tokens": 1200}}


def passing_judgment() -> dict:
    return {
        "groupDecision": "pass",
        "groupRationale": "All twenty niches are distinct and support four causal sibling states.",
        "dimensions": {
            dimension: {"status": "pass", "rationale": f"{dimension} passes independently."}
            for dimension in BLUEPRINT_JUDGE_DIMENSIONS
        },
        "findings": [],
    }


class ScriptedJudge:
    def __init__(self, decisions=None):
        self.decisions = list(decisions or [])
        self.calls: list[tuple[dict, dict]] = []

    def binding(self) -> dict:
        return {
            "protocol": "independent_fake_whole_blueprint_judge",
            "modelBinding": {
                "model": "independent-judge-model",
                "endpoint": "http://judge-lane/v1/chat/completions",
                "reasoning": {"enabled": False},
            },
        }

    def judge_topic(self, topic_card: dict, blueprint_set: dict):
        self.calls.append((topic_card, blueprint_set))
        decision = self.decisions.pop(0) if self.decisions else passing_judgment()
        return decision, {"model": "independent-judge-model", "finishReason": "stop"}


class PassingTaxonomyJudge:
    def __init__(self) -> None:
        self.calls: list[tuple[dict, dict]] = []

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
        self.calls.append((topic_card, judge_view))
        return {"findingClusters": []}, {
            "model": "independent-taxonomy-judge-model",
            "finishReason": "stop",
        }


def passed_scrutiny(tmp_path: Path, card: dict, blueprint_set: dict) -> dict:
    return generate_topic_blueprint_scrutiny(
        request=request(),
        topic=card,
        blueprint_set=blueprint_set,
        output_root=tmp_path,
        judge=ScriptedJudge(),
        max_attempts=1,
    )


def recursively_has_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(recursively_has_key(child, key) for child in value.values())
    if isinstance(value, list):
        return any(recursively_has_key(child, key) for child in value)
    return False


def test_stage_p_schema_is_required_twenty_key_object_without_prefix_items() -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    schema = build_blueprint_response_schema(card)

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(ids)
    assert set(schema["properties"]) == set(ids)
    assert len(schema["properties"]) == BLUEPRINTS_PER_TOPIC
    assert not recursively_has_key(schema, "prefixItems")
    assert all(
        property_schema["additionalProperties"] is False
        and set(property_schema["required"]) == set(property_schema["properties"])
        for property_schema in schema["properties"].values()
    )


def test_taxonomy_wire_is_canonical_and_old_aliases_are_ineligible() -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    schema = build_taxonomy_response_schema(card)
    anchor_schema = schema["properties"][ids[0]]

    assert TAXONOMY_WIRE_KEYS == {field: field for field in TAXONOMY_FIELDS}
    assert anchor_schema["required"] == list(TAXONOMY_FIELDS)
    assert set(anchor_schema["properties"]) == set(TAXONOMY_FIELDS)
    assert all(
        "maxLength" not in anchor_schema["properties"][field]
        for field in TAXONOMY_FIELDS
    )
    assert all(
        anchor_schema["properties"][field]["minLength"] == 3
        for field in TAXONOMY_FIELDS
    )

    natural_wire = deepcopy(taxonomy_wire(card["topicId"]))
    natural_wire[ids[0]]["centralTension"] = (
        "One participant needs a detailed explanation while the other needs enough time "
        "to verify the evidence carefully before committing to a shared decision."
    )
    assert decode_taxonomy_response(natural_wire, card)[ids[0]][
        "centralTension"
    ].endswith(".")

    old_alias_wire = deepcopy(taxonomy_wire(card["topicId"]))
    old_alias_wire[ids[0]] = {
        "u": "legacy submode",
        "r": "legacy relationship",
        "s": "legacy setting",
        "c": "legacy resource",
        "t": "legacy tension",
    }
    with pytest.raises(InvalidModelOutput, match="violates strict schema"):
        decode_taxonomy_response(old_alias_wire, card)


def test_taxonomy_structural_gates_reject_shifted_roles_and_duplicate_resources() -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])

    repeated_resource = deepcopy(taxonomy_anchors(card["topicId"]))
    repeated_resource[ids[1]]["centralResource"] = repeated_resource[ids[0]][
        "centralResource"
    ]
    with pytest.raises(InvalidModelOutput, match="centralResource repeats exactly"):
        validate_taxonomy_anchors(repeated_resource, card)

    shifted_mode = deepcopy(taxonomy_anchors(card["topicId"]))
    shifted_mode[ids[0]]["centralResource"] = card["interactionModes"][0]
    with pytest.raises(InvalidModelOutput, match="interaction-mode label"):
        validate_taxonomy_anchors(shifted_mode, card)

    collapsed_roles = deepcopy(taxonomy_anchors(card["topicId"]))
    collapsed_roles[ids[0]]["participantRelationship"] = collapsed_roles[ids[0]][
        "centralResource"
    ]
    with pytest.raises(InvalidModelOutput, match="collapse distinct roles"):
        validate_taxonomy_anchors(collapsed_roles, card)


def test_taxonomy_complete_phrases_are_not_clipped_to_legacy_limits() -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    anchors = deepcopy(taxonomy_anchors(card["topicId"]))
    anchors[ids[0]].update({
        "submode": "collaborative reconstruction of a disputed maintenance history",
        "participantRelationship": "a tenant and building manager negotiating responsibilities after prior repairs",
        "setting": "a scheduled video conference with the maintenance record visible to both parties",
        "centralResource": "the signed maintenance inspection report and its dated photographic appendix",
        "centralTension": "the report confirms damage but leaves responsibility uncertain because two repair visits occurred between inspection and discovery",
    })

    assert decode_taxonomy_response(anchors, card) == anchors

    blueprints = deepcopy(joint_blueprints(card["topicId"]))
    blueprints[ids[0]].update(anchors[ids[0]])
    encoded = encode_blueprint_response(
        blueprints, card, taxonomy_anchors=anchors
    )
    assert all(
        TAXONOMY_WIRE_KEYS[field] not in encoded[ids[0]]
        for field in TAXONOMY_FIELDS
    )
    assert decode_blueprint_response(
        encoded, card, taxonomy_anchors=anchors
    ) == blueprints


def test_stage_p_response_budget_and_tight_fields_are_sent_to_authentic_call(tmp_path: Path) -> None:
    card = topic()
    planner = ScriptedPlanner(
        [
            taxonomy_wire(card["topicId"]),
            encode_blueprint_response(
                joint_blueprints(),
                card,
                taxonomy_anchors=taxonomy_anchors(card["topicId"]),
            ),
        ]
    )
    checkpoint = generate_topic_blueprints(
        request=request(),
        topic=card,
        output_root=tmp_path,
        planner=planner,
        taxonomy_judge=PassingTaxonomyJudge(),
        max_attempts=1,
    )

    call = planner.calls[1]
    assert planner.calls[0]["max_output_tokens"] == TAXONOMY_MAX_OUTPUT_TOKENS == 8192
    assert planner.calls[0]["context"]["assignedInteractionModeByScenarioId"] == {
        scenario_id: card["interactionModes"][ordinal % len(card["interactionModes"])]
        for ordinal, scenario_id in enumerate(scenario_ids_for_topic(card["topicId"]))
    }
    assert call["max_output_tokens"] == BLUEPRINT_INITIAL_OUTPUT_TOKENS == 4096
    assert BLUEPRINT_MAX_OUTPUT_TOKENS == 12288
    assert call["context"]["outputContract"]["maxLiveOutputTokens"] == 12288
    assert call["schema"]["required"] == list(scenario_ids_for_topic(card["topicId"]))
    max_lengths = []

    def collect(value):
        if isinstance(value, dict):
            if "maxLength" in value:
                max_lengths.append(value["maxLength"])
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(call["schema"])
    assert not max_lengths
    first_wire_niche = call["schema"]["properties"][scenario_ids_for_topic(card["topicId"])[0]]
    assert max(len(field) for field in first_wire_niche["properties"]) == 1
    assert checkpoint["blueprintSet"]["jointBlueprintHash"] == content_hash(joint_blueprints())


def test_stage_p_wire_encoding_is_lossless_and_materially_smaller() -> None:
    card = topic()
    canonical = joint_blueprints()
    encoded = encode_blueprint_response(canonical, card)

    assert decode_blueprint_response(encoded, card) == canonical
    assert len(canonical_json(encoded)) < len(canonical_json(canonical)) * 0.72


def test_schema_bound_coverage_rejects_cross_slot_niche_substitution() -> None:
    card = topic()
    value = joint_blueprints()
    ids = scenario_ids_for_topic(card["topicId"])
    duplicate = deepcopy(value[ids[0]])
    duplicate["semanticDistinctnessFrom"] = deepcopy(value[ids[1]]["semanticDistinctnessFrom"])
    value[ids[1]] = duplicate

    with pytest.raises(InvalidModelOutput, match="violates strict schema"):
        validate_blueprint_response(value, card)


def test_mirrored_pairs_fail_the_three_field_typed_divergence_floor() -> None:
    card = topic()
    value = joint_blueprints()
    ids = scenario_ids_for_topic(card["topicId"])
    target_id = ids[6]
    mirrored = deepcopy(value[ids[0]])
    mirrored["evidencePivot"] = "one changed evidence detail"
    for field in (
        "interactionMode",
        "controlOperator",
        "stakesProfile",
        "outcomeTopology",
        "duplexOpportunity",
    ):
        mirrored[field] = value[target_id][field]
    mirrored["semanticDistinctnessFrom"] = {
        "siblingIds": [ids[0]],
        "distinction": "only the evidence detail changes",
    }
    value[target_id] = mirrored

    with pytest.raises(InvalidModelOutput, match="differs on only"):
        validate_blueprint_response(value, card)


def test_judge_findings_are_the_sole_normalized_admission_signal() -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    contradictory = passing_judgment()
    contradictory["findings"] = [
        {
            "code": "semantic_near_duplicate_cluster",
            "scenarioIds": list(ids[:2]),
            "rationale": "The pair is semantically equivalent despite different surface details.",
        }
    ]

    normalized = validate_blueprint_judgment(contradictory, ids)
    assert normalized["groupDecision"] == "reject"
    assert normalized["dimensions"]["semanticDiversity"]["status"] == "fail"


def test_target_field_leakage_is_rejected_without_text_repair() -> None:
    card = topic()
    value = joint_blueprints()
    first_id = scenario_ids_for_topic(card["topicId"])[0]
    value[first_id]["targetText"] = "forbidden"

    with pytest.raises(InvalidModelOutput):
        validate_blueprint_response(value, card)


def test_malformed_and_truncated_stage_p_outputs_retry_the_exact_topic(tmp_path: Path) -> None:
    card = topic()
    planner = ScriptedPlanner(
        [
            taxonomy_wire(card["topicId"]),
            {"malformed": {}},
            TruncatedModelOutput("finish_reason=length"),
            encode_blueprint_response(
                joint_blueprints(),
                card,
                taxonomy_anchors=taxonomy_anchors(card["topicId"]),
            ),
        ]
    )
    checkpoint = generate_topic_blueprints(
        request=request(),
        topic=card,
        output_root=tmp_path,
        planner=planner,
        taxonomy_judge=PassingTaxonomyJudge(),
        max_attempts=3,
    )

    assert len(planner.calls) == 4
    assert {call["context"]["topicCard"]["topicId"] for call in planner.calls} == {card["topicId"]}
    assert "retryFeedback" not in planner.calls[1]["context"]
    assert "malformed" in planner.calls[2]["context"]["retryFeedback"]["previousDefect"]
    assert "TruncatedModelOutput" in planner.calls[3]["context"]["retryFeedback"]["previousDefect"]
    assert [call["max_output_tokens"] for call in planner.calls[1:]] == [4096, 4096, 8192]
    assert checkpoint["topicId"] == card["topicId"]


def test_taxonomy_retry_revises_prior_object_with_exact_structural_prohibitions(
    tmp_path: Path,
) -> None:
    card = topic()
    ids = scenario_ids_for_topic(card["topicId"])
    rejected = deepcopy(taxonomy_wire(card["topicId"]))
    duplicate_value = rejected[ids[0]]["submode"]
    rejected[ids[1]]["submode"] = duplicate_value
    planner = ScriptedPlanner([
        rejected,
        taxonomy_wire(card["topicId"]),
        encode_blueprint_response(
            joint_blueprints(),
            card,
            taxonomy_anchors=taxonomy_anchors(card["topicId"]),
        ),
    ])

    generate_topic_blueprints(
        request=request(),
        topic=card,
        output_root=tmp_path,
        planner=planner,
        taxonomy_judge=PassingTaxonomyJudge(),
        max_attempts=2,
    )

    revision_context = planner.calls[1]["context"]
    assert revision_context["previousRejectedTaxonomy"] == rejected
    assert revision_context["retryFeedback"]["previousRejectedTaxonomyHash"] == (
        content_hash(rejected)
    )
    assert revision_context["structuralRevisionContract"] == {
        "mustReturnAllScenarioIds": True,
        "identicalResponseForbidden": True,
        "forbiddenExactValuesByScenarioId": {
            ids[1]: {"submode": [duplicate_value]}
        },
    }


def test_per_topic_and_per_scenario_resume_make_no_model_call(tmp_path: Path) -> None:
    card = topic()
    blueprints = joint_blueprints()
    first_planner = ScriptedPlanner(
        [
            taxonomy_wire(card["topicId"]),
            encode_blueprint_response(
                blueprints,
                card,
                taxonomy_anchors=taxonomy_anchors(card["topicId"]),
            ),
        ]
    )
    first = generate_topic_blueprints(
        request=request(), topic=card, output_root=tmp_path, planner=first_planner,
        taxonomy_judge=PassingTaxonomyJudge(),
    )
    no_topic_call = ScriptedPlanner([])
    resumed = generate_topic_blueprints(
        request=request(),
        topic=card,
        output_root=tmp_path,
        planner=no_topic_call,
        taxonomy_judge=PassingTaxonomyJudge(),
        resume=True,
    )
    assert resumed == first
    assert no_topic_call.calls == []

    blueprint_set = first["blueprintSet"]
    scrutiny = passed_scrutiny(tmp_path, card, blueprint_set)
    scenario_id = blueprint_set["scenarioIds"][0]
    first_expand = ScriptedPlanner([expansion_response(card["topicId"], scenario_id, blueprint_set)])
    scenario_checkpoint = expand_blueprint_slot(
        request=request(),
        topic=card,
        blueprint_set=blueprint_set,
        blueprint_scrutiny=scrutiny,
        scenario_id=scenario_id,
        output_root=tmp_path,
        planner=first_expand,
    )
    no_scenario_call = ScriptedPlanner([])
    scenario_resumed = expand_blueprint_slot(
        request=request(),
        topic=card,
        blueprint_set=blueprint_set,
        blueprint_scrutiny=scrutiny,
        scenario_id=scenario_id,
        output_root=tmp_path,
        planner=no_scenario_call,
        resume=True,
    )
    assert scenario_resumed == scenario_checkpoint
    assert no_scenario_call.calls == []


def test_expansion_schema_prompt_and_response_are_hash_bound(tmp_path: Path) -> None:
    card = topic()
    blueprint_set = make_blueprint_set(card, joint_blueprints())
    scenario_id = blueprint_set["scenarioIds"][4]
    expected_blueprint_hash = blueprint_set["blueprintHashes"][scenario_id]
    expected_joint_hash = blueprint_set["jointBlueprintHash"]
    schema = build_expansion_response_schema(
        scenario_id, card["topicId"], expected_blueprint_hash, expected_joint_hash
    )
    assert schema["properties"]["scenarioId"] == {"const": scenario_id}
    assert schema["properties"]["topicId"] == {"const": card["topicId"]}
    assert schema["properties"]["blueprintHash"] == {"const": expected_blueprint_hash}
    assert schema["properties"]["jointBlueprintHash"] == {"const": expected_joint_hash}

    bad = expansion_response(card["topicId"], scenario_id, blueprint_set)
    bad["blueprintHash"] = "sha256:" + "0" * 64
    with pytest.raises(InvalidModelOutput):
        validate_expansion_response(
            bad, scenario_id, card["topicId"], expected_blueprint_hash, expected_joint_hash
        )

    planner = ScriptedPlanner([expansion_response(card["topicId"], scenario_id, blueprint_set)])
    scrutiny = passed_scrutiny(tmp_path, card, blueprint_set)
    checkpoint = expand_blueprint_slot(
        request=request(),
        topic=card,
        blueprint_set=blueprint_set,
        blueprint_scrutiny=scrutiny,
        scenario_id=scenario_id,
        output_root=tmp_path,
        planner=planner,
        max_attempts=1,
    )
    context = planner.calls[0]["context"]
    assert context["jointCompactBlueprint"] == blueprint_set
    assert context["assignedBlueprint"] == blueprint_set["blueprints"][scenario_id]
    assert checkpoint["blueprintHash"] == expected_blueprint_hash
    assert checkpoint["jointBlueprintHash"] == expected_joint_hash


def test_stage_e_materializes_control_and_four_routes_from_admitted_blueprint() -> None:
    card = topic()
    blueprint_set = make_blueprint_set(card, joint_blueprints())
    scenario_id = blueprint_set["scenarioIds"][0]
    blueprint = blueprint_set["blueprints"][scenario_id]
    wire = expansion_response(card["topicId"], scenario_id, blueprint_set)

    assert "scenarioOutcomeSpace" not in wire["scenarioContract"]
    assert "requiredControlPhenomena" not in wire["scenarioContract"]
    normalized = validate_expansion_response(
        wire,
        scenario_id,
        card["topicId"],
        blueprint_set["blueprintHashes"][scenario_id],
        blueprint_set["jointBlueprintHash"],
        blueprint,
    )
    assert normalized["scenarioContract"]["scenarioOutcomeSpace"] == [
        f"verified_positive: {blueprint['fourSiblingAffordance']['verifiedPositive']}",
        f"verified_negative: {blueprint['fourSiblingAffordance']['verifiedNegative']}",
        f"uncertain: {blueprint['fourSiblingAffordance']['uncertain']}",
        f"superseded: {blueprint['fourSiblingAffordance']['superseded']}",
    ]
    assert normalized["scenarioContract"]["requiredControlPhenomena"][0] == (
        f"control_operator: {blueprint['controlOperator']}"
    )


def test_control_operator_repetition_is_legal_when_full_signatures_are_unique() -> None:
    card = topic()
    value = joint_blueprints()

    validated = validate_blueprint_response(value, card)
    assert len({typed_niche_signature(item) for item in validated.values()}) == 20
    assert len({item["controlOperator"] for item in validated.values()}) < 20
    assert {item["controlOperator"] for item in validated.values()} == set(CONTROL_OPERATORS)


def test_independent_whole_blueprint_judge_is_injectable_and_gates_expansion(
    tmp_path: Path,
) -> None:
    card = topic()
    blueprint_set = make_blueprint_set(card, joint_blueprints())
    judge = ScriptedJudge()
    scrutiny = generate_topic_blueprint_scrutiny(
        request=request(),
        topic=card,
        blueprint_set=blueprint_set,
        output_root=tmp_path,
        judge=judge,
        max_attempts=1,
    )
    assert len(judge.calls) == 1
    assert judge.calls[0][1] == blueprint_set
    validate_blueprint_scrutiny(scrutiny, request(), card, blueprint_set)

    scenario_id = blueprint_set["scenarioIds"][0]
    planner = ScriptedPlanner([expansion_response(card["topicId"], scenario_id, blueprint_set)])
    with pytest.raises(ScenarioBlueprintError, match="passing independent"):
        expand_blueprint_slot(
            request=request(),
            topic=card,
            blueprint_set=blueprint_set,
            blueprint_scrutiny=None,
            scenario_id=scenario_id,
            output_root=tmp_path,
            planner=planner,
        )
    assert planner.calls == []

    mixed = make_blueprint_set(topic("topic_beta"), joint_blueprints("topic_beta"))
    with pytest.raises(ScenarioBlueprintError, match="binding mismatch"):
        expand_blueprint_slot(
            request=request(),
            topic=topic("topic_beta"),
            blueprint_set=mixed,
            blueprint_scrutiny=scrutiny,
            scenario_id=mixed["scenarioIds"][0],
            output_root=tmp_path,
            planner=planner,
        )
    assert planner.calls == []


def test_rejecting_clustered_judge_artifact_blocks_stage_e(tmp_path: Path) -> None:
    card = topic()
    blueprint_set = make_blueprint_set(card, joint_blueprints())
    ids = blueprint_set["scenarioIds"]
    decision = passing_judgment()
    decision["groupDecision"] = "reject"
    decision["groupRationale"] = "One semantic near-duplicate cluster requires replanning."
    decision["dimensions"]["semanticDiversity"] = {
        "status": "fail",
        "rationale": "Two slots collapse onto the same scenario niche.",
    }
    decision["findings"] = [
        {
            "code": "semantic_near_duplicate_cluster",
            "scenarioIds": ids[:2],
            "rationale": "The pair differs only lexically.",
        }
    ]
    scrutiny = generate_topic_blueprint_scrutiny(
        request=request(),
        topic=card,
        blueprint_set=blueprint_set,
        output_root=tmp_path,
        judge=ScriptedJudge([decision]),
        max_attempts=1,
    )
    planner = ScriptedPlanner([])
    with pytest.raises(ScenarioBlueprintError, match="refused non-passing"):
        expand_blueprint_slot(
            request=request(),
            topic=card,
            blueprint_set=blueprint_set,
            blueprint_scrutiny=scrutiny,
            scenario_id=ids[0],
            output_root=tmp_path,
            planner=planner,
        )
    assert planner.calls == []


def test_exact_50x20_cardinality_and_global_exact_premise_uniqueness() -> None:
    topics = [topic(f"topic_{index:02d}") for index in range(50)]
    scenarios = [
        scenario_contract(card["topicId"], scenario_id)
        for card in topics
        for scenario_id in scenario_ids_for_topic(card["topicId"])
    ]
    validate_corpus_shape(request(), topics)
    validate_canonical_scenarios(request(), topics, scenarios)

    with pytest.raises(ScenarioBlueprintError, match="exactly 1000"):
        validate_canonical_scenarios(request(), topics, scenarios[:-1])

    collided = deepcopy(scenarios)
    collided[-1]["premise"] = collided[0]["premise"]
    with pytest.raises(ScenarioBlueprintError, match="exact-duplicate-free"):
        validate_canonical_scenarios(request(), topics, collided)


def test_canonical_sidecar_is_exact_1_to_1_and_rejects_stale_blueprint_lineage(
    tmp_path: Path,
) -> None:
    topics = [topic(f"topic_{index:02d}") for index in range(50)]
    blueprint_sets = [
        make_blueprint_set(card, joint_blueprints(card["topicId"])) for card in topics
    ]
    scrutinies = [
        passed_scrutiny(tmp_path, card, blueprint_set)
        for card, blueprint_set in zip(topics, blueprint_sets, strict=True)
    ]
    scenarios = [
        scenario_contract(card["topicId"], scenario_id)
        for card in topics
        for scenario_id in scenario_ids_for_topic(card["topicId"])
    ]
    scenario_by_id = {item["scenarioId"]: item for item in scenarios}
    scrutiny_by_topic = {item["topicId"]: item for item in scrutinies}
    expansion_binding = {
        "protocol": "fake_strict_schema",
        "model": "expansion-model",
        "reasoning": {"enabled": False},
    }
    checkpoints = []
    for blueprint_set in blueprint_sets:
        topic_id = blueprint_set["topicId"]
        for scenario_id in blueprint_set["scenarioIds"]:
            scenario = scenario_by_id[scenario_id]
            body = {
                "scenarioId": scenario_id,
                "topicId": topic_id,
                "blueprintHash": blueprint_set["blueprintHashes"][scenario_id],
                "jointBlueprintHash": blueprint_set["jointBlueprintHash"],
                "blueprintScrutinyHash": scrutiny_by_topic[topic_id]["checkpointHash"],
                "plannerBinding": expansion_binding,
                "plannerBindingHash": content_hash(expansion_binding),
                "scenarioContract": scenario,
                "scenarioContractHash": content_hash(scenario),
            }
            checkpoint = dict(body)
            checkpoint["checkpointHash"] = content_hash(body)
            checkpoints.append(checkpoint)

    bindings = build_scenario_blueprint_bindings(
        request=request(),
        topics=topics,
        blueprint_sets=blueprint_sets,
        blueprint_scrutinies=scrutinies,
        expansion_checkpoints=checkpoints,
        scenarios=scenarios,
    )
    assert len(bindings) == len(scenarios) == 1000
    assert {item["scenarioId"] for item in bindings} == {
        item["scenarioId"] for item in scenarios
    }
    assert all(
        {
            "blueprintProfileHash",
            "expansionCheckpointHash",
            "plannerBinding",
            "finalScenarioHash",
        }.issubset(item)
        for item in bindings
    )
    manifest = build_scenario_blueprint_binding_manifest(
        request=request(),
        topics=topics,
        blueprint_sets=blueprint_sets,
        blueprint_scrutinies=scrutinies,
        scenarios=scenarios,
        bindings=bindings,
    )
    assert manifest["scenarioCount"] == manifest["bindingCount"] == 1000

    stale = deepcopy(checkpoints)
    stale[0]["blueprintHash"] = stale[1]["blueprintHash"]
    stale_body = dict(stale[0])
    stale_body.pop("checkpointHash")
    stale[0]["checkpointHash"] = content_hash(stale_body)
    with pytest.raises(ScenarioBlueprintError, match="blueprint hash mismatch"):
        build_scenario_blueprint_bindings(
            request=request(),
            topics=topics,
            blueprint_sets=blueprint_sets,
            blueprint_scrutinies=scrutinies,
            expansion_checkpoints=stale,
            scenarios=scenarios,
        )


def test_corpus_shape_rejects_non_fifty_topic_input() -> None:
    with pytest.raises(ScenarioBlueprintError, match="exactly fifty"):
        validate_corpus_shape(request(), [topic()])
