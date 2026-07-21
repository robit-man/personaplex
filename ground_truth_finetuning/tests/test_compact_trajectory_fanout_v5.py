from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import urllib.error

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.compact_trajectory_fanout import (
    CANDIDATES_FILENAME,
    CAUSAL_SIBLING_ROLES,
    CHECKPOINT_ROOT,
    COMPACT_RESPONSE_NAME,
    CUDA_DEVICES,
    DUPLEX_PROFILES,
    EXPANSION_RESPONSE_NAME,
    INTERACTION_MECHANISMS,
    MAX_PROTOCOL_ATTEMPTS,
    MAX_COMPACT_RESPONSE_BYTES,
    MAX_STAGE_ATTEMPTS,
    PRIMARY_FILENAME,
    PRODUCTION_CANDIDATE_COUNT,
    PRODUCTION_EXPANSION_COUNT,
    PRODUCTION_PRIMARY_COUNT,
    PRODUCTION_RESERVE_COUNT,
    PRODUCTION_SCENARIO_COUNT,
    PRODUCTION_SCENARIOS_PER_TOPIC,
    PRODUCTION_TOPIC_COUNT,
    POSTURE_TRANSITIONS,
    RESERVE_FILENAME,
    SELECTED_FILENAME,
    STYLE_PROFILES,
    STATE_TRANSITIONS,
    STAGE_A_MAX_OUTPUT_TOKENS,
    STAGE_A_OUTPUT_TOKEN_HEADROOM,
    TARGET_DIALOGUE_FIELDS,
    TRAJECTORIES_FILENAME,
    FanoutError,
    ThreeEndpointJsonSchemaClient,
    _selection_hash,
    _validate_selection_rows,
    candidate_lineages,
    canonical_json,
    compact_response_schema,
    compact_lineage,
    compact_wire_candidate,
    content_hash,
    expand_selected_candidates,
    full_trajectory_response_schema,
    generate_compact_candidates,
    measure_compact_response_bound,
    read_jsonl,
    select_compact_candidates,
)
from ground_truth_finetuning.tools.build_diverse_synthesis_cascade import (
    reject_legacy_v5_trajectory_stage,
)
from ground_truth_finetuning.training.diverse_cascade import CascadeError


REQUEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "requests"
    / "personaplex_diverse_50x20x10.control-v5.json"
)


def request_fixture(
    *, scenario_count: int, primary: int, reserve: int, topic_count: int = 1
) -> dict:
    value = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    value["requestId"] = f"compact-focused-{scenario_count}-{primary}-{reserve}"
    value["coverageTarget"].update(
        {
            "candidateTopics": topic_count,
            "scenariosPerTopic": scenario_count // topic_count,
            "trajectorySeedsPerScenario": 10,
            "primaryGroups": primary,
            "reserveGroups": reserve,
            "selectedCounterfactualGroups": primary,
            "branchesPerGroup": 4,
        }
    )
    return value


def topic_fixture(
    request: dict, topic_id: str = "topic_compact", source_seed_id: str = "S01"
) -> dict:
    return {
        "schema": "personaplex.topic-card.v2",
        "topicId": topic_id,
        "sourceSeedId": source_seed_id,
        "seedRevision": request["seedRevision"],
        "domain": "community problem solving",
        "interactionModes": ["cooperative", "skeptical", "conditional"],
        "registerRange": ["casual", "neutral", "formal"],
        "safeStakes": ["planning", "clarification", "handoff"],
        "forbiddenPatterns": ["target dialogue", "placeholder identity"],
        "diversityTags": ["community", "technical", "creative"],
        "causalAffordances": [
            {
                "family": "semantic",
                "operatorId": "evidence_status_transition",
                "changedPath": "state.evidence.status",
            },
            {
                "family": "delivery",
                "operatorId": "next_goal_transition",
                "changedPath": "plan.nextGoal",
            },
            {
                "family": "turn_taking",
                "operatorId": "interruption_transition",
                "changedPath": "turnTaking.eventType",
            },
        ],
    }


def scenario_fixture(index: int, topic_id: str = "topic_compact") -> dict:
    return {
        "schema": "personaplex.scenario-contract.v2",
        "scenarioId": f"scenario_compact_{index:03d}",
        "topicId": topic_id,
        "mode": "collaborative_resolution",
        "premise": (
            f"Two neighbors in scenario {index} compare safe ways to coordinate shared "
            "workshop hours without assuming either person's availability."
        ),
        "participants": [
            {
                "role": "caller",
                "knowledge": "knows personal constraints and previous attempts",
            },
            {
                "role": "agent",
                "knowledge": "knows facts introduced through rolling semantic state",
            },
        ],
        "startingState": {
            "knownFacts": [f"shared space is used in scenario {index}"],
            "uncertainty": ["the preferred time window is unresolved"],
            "policyConstraints": ["do not invent agreement or availability"],
        },
        "interactionOpportunity": ["clarify constraints", "compare options"],
        "allowedToolClasses": ["calendar_lookup", "note_capture"],
        "disallowedClaims": ["guaranteed agreement", "invented schedule"],
        "scenarioOutcomeSpace": [
            "conditional_agreement",
            "bounded_clarification",
            "documented_handoff",
        ],
        "requiredControlPhenomena": [
            "tool update",
            "caller posture change",
            "interruption recovery",
        ],
    }


def compact_card(binding: dict, topic: dict, scenario: dict) -> dict:
    slot = binding["candidateOrdinal"]
    target_turns = (8, 10, 14)[slot % 3]
    length_band = ("short", "medium", "long")[slot % 3]
    agent_targets = target_turns // 2
    pivot = 2 if agent_targets == 4 else 2 + slot % (agent_targets - 2)
    operator = topic["causalAffordances"][slot % len(topic["causalAffordances"])]
    duplex_profile = tuple(DUPLEX_PROFILES)[slot % len(DUPLEX_PROFILES)]
    scenario_token = content_hash(scenario)[7:15]
    sources = (
        "asr_fact",
        "tool_result",
        "policy_decision",
        "caller_posture_change",
        "interruption",
        "handoff",
        "scenario_state",
        "state_reducer",
    )
    axes = (
        "fact_change",
        "tool_result_change",
        "policy_constraint_change",
        "caller_posture_change",
        "next_goal_change",
        "interruption_invalidation",
        "uncertainty_resolution",
        "handoff_route_change",
        "termination_authorization_change",
    )
    return {
        "schema": "personaplex.compact-trajectory-leaf.v5",
        "candidateId": binding["candidateId"],
        "trajectoryId": binding["trajectoryId"],
        "candidateOrdinal": slot,
        "lineage": compact_lineage(binding),
        "stateTransition": {
            "kind": STATE_TRANSITIONS[slot % len(STATE_TRANSITIONS)],
            "fromState": f"s{scenario_token}_p{slot}",
            "toState": f"s{scenario_token}_r{slot}",
            "tag": f"{scenario_token}_{slot}",
        },
        "postureTransition": {
            "from": tuple(POSTURE_TRANSITIONS.values())[
                slot % len(POSTURE_TRANSITIONS)
            ][0],
            "to": tuple(POSTURE_TRANSITIONS.values())[
                slot % len(POSTURE_TRANSITIONS)
            ][1],
        },
        "evidenceSource": sources[slot % len(sources)],
        "duplexProfile": duplex_profile,
        "duplexEventTypes": list(DUPLEX_PROFILES[duplex_profile]),
        "outcomeRoute": scenario["scenarioOutcomeSpace"][slot % 3],
        "conversationLength": {
            "targetTurns": target_turns,
            "lengthBand": length_band,
        },
        "styleProfile": STYLE_PROFILES[slot % len(STYLE_PROFILES)],
        "interactionMechanism": INTERACTION_MECHANISMS[
            slot % len(INTERACTION_MECHANISMS)
        ],
        "stakes": topic["safeStakes"][slot % len(topic["safeStakes"])],
        "causalAxis": axes[slot % len(axes)],
        "interventionFamily": operator["family"],
        "causalOperator": {
            "operatorId": operator["operatorId"],
            "family": operator["family"],
            "changedPath": operator["changedPath"],
        },
        "typedPivot": {
            "field": operator["changedPath"],
            "from": f"s{scenario_token}_pending_{slot}",
            "to": f"s{scenario_token}_resolved_{slot}",
        },
        "counterfactualPivotOrdinal": pivot,
        "controlPhenomena": [
            scenario["requiredControlPhenomena"][
                slot % len(scenario["requiredControlPhenomena"])
            ]
        ],
        "modelAdmission": "admit",
    }


def expanded_trajectory(candidate: dict) -> dict:
    turns = candidate["conversationLength"]["targetTurns"] // 2
    pivot = candidate["counterfactualPivotOrdinal"]
    revisions = [20 + ordinal for ordinal in range(1, turns + 1)]
    barge_target = min(2, turns - 1)
    events = [
        {
            "eventType": "barge_in",
            "targetOrdinal": barge_target,
            "offsetMs": 420,
            "overlapMs": 140,
            "cancelOutgoingAudio": False,
            "invalidateGeneration": False,
        },
        {
            "eventType": "cancelled_generation",
            "targetOrdinal": barge_target,
            "offsetMs": 560,
            "overlapMs": 140,
            "cancelOutgoingAudio": True,
            "invalidateGeneration": True,
        },
    ]
    recovery_type = (
        "recovery"
        if "recovery" in candidate["duplexEventTypes"]
        else "repair_after_barge_in"
    )
    events.append(
        {
            "eventType": recovery_type,
            "targetOrdinal": barge_target + 1,
            "offsetMs": 190,
            "overlapMs": 0,
            "cancelOutgoingAudio": False,
            "invalidateGeneration": False,
        }
    )
    for event_type in candidate["duplexEventTypes"]:
        if event_type in {"barge_in", "cancelled_generation", recovery_type}:
            continue
        events.append(
            {
                "eventType": event_type,
                "targetOrdinal": 1,
                "offsetMs": 110,
                "overlapMs": 40 if event_type == "brief_overlap" else 0,
                "cancelOutgoingAudio": False,
                "invalidateGeneration": False,
            }
        )
    states = []
    schedule = []
    for ordinal, revision in enumerate(revisions, start=1):
        active = candidate["typedPivot"]["from"] if ordinal < pivot else candidate["typedPivot"]["to"]
        source = candidate["evidenceSource"]
        states.append(
            {
                "targetOrdinal": ordinal,
                "phase": f"phase_{candidate['candidateOrdinal']}_{ordinal}",
                "availableBeforeTarget": True,
                "controlRevision": revision,
                "knownFacts": [f"bounded fact {candidate['candidateId']} {ordinal}"],
                "uncertainty": [f"remaining uncertainty {candidate['candidateId']} {ordinal}"],
                "policyConstraints": ["do not invent an agreement"],
                "commitments": [f"bounded commitment {candidate['candidateId']} {ordinal}"],
                "callerPosture": (
                    candidate["postureTransition"]["from"]
                    if ordinal < pivot
                    else candidate["postureTransition"]["to"]
                ),
                "nextGoal": f"advance the declared state for target {ordinal}",
                "evidence": {
                    "source": source,
                    "status": f"observed_{ordinal}",
                    "facts": [f"evidence state {candidate['candidateId']} {ordinal}"],
                },
                "toolResult": {
                    "source": "fixture_state_source",
                    "status": f"bounded_{ordinal}",
                    "facts": [f"tool state {candidate['candidateId']} {ordinal}"],
                },
                "causalState": {
                    "candidateHash": candidate["candidateHash"],
                    "operatorId": candidate["causalOperator"]["operatorId"],
                    "changedPath": candidate["typedPivot"]["field"],
                    "from": candidate["typedPivot"]["from"],
                    "to": candidate["typedPivot"]["to"],
                    "activeValue": active,
                },
                "revisionReason": f"new state is available before target {ordinal}",
            }
        )
        schedule.append(
            {
                "controlRevision": revision,
                "targetOrdinal": ordinal,
                "availableBeforeTarget": True,
                "source": source,
            }
        )
    length = candidate["conversationLength"]
    return {
        "schema": "personaplex.trajectory-seed.v2",
        "trajectoryId": candidate["trajectoryId"],
        "scenarioId": candidate["lineage"]["scenarioId"],
        "conversationLength": {
            "targetTurns": length["targetTurns"],
            "min": max(4, length["targetTurns"] - 2),
            "max": length["targetTurns"] + 2,
        },
        "pace": f"pace derived authentically for {candidate['styleProfile']}",
        "openingStyle": f"opening shaped by {candidate['interactionMechanism']}",
        "closingStyle": f"closing bounded by {candidate['outcomeRoute']}",
        "voicePairPolicy": "distinct_approved_references",
        "interactionArc": [
            f"establish {candidate['stakes']}",
            f"apply {candidate['interactionMechanism']}",
            f"resolve through {candidate['outcomeRoute']}",
        ],
        "duplexEvents": events,
        "postureArc": [
            candidate["postureTransition"]["from"],
            candidate["postureTransition"]["to"],
        ],
        "counterfactualPivotOrdinal": candidate["counterfactualPivotOrdinal"],
        "controlPhenomena": candidate["controlPhenomena"],
        "causalAxis": candidate["causalAxis"],
        "interventionFamily": candidate["interventionFamily"],
        "typedPivot": candidate["typedPivot"],
        "postureTransition": candidate["postureTransition"],
        "evidenceSource": candidate["evidenceSource"],
        "outcomeRoute": candidate["outcomeRoute"],
        "semanticStateArc": states,
        "controlRevisionSchedule": schedule,
        "terminationContract": {
            "decisionSource": "model",
            "action": "end_call_tool",
            "deterministicPhrase": False,
        },
        "negativeControlCoverage": [
            "paired_wrong_branch",
            "stale_revision",
            "null_control",
        ],
    }


class FakePlanner:
    def __init__(
        self,
        *,
        malformed_compact_once: bool = False,
        metadata_overrides: dict | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self.malformed_compact_once = malformed_compact_once
        self.metadata_overrides = metadata_overrides or {}

    def generate(self, *, name, schema, instructions, context, max_output_tokens):
        self.calls.append(
            {
                "name": name,
                "context": deepcopy(context),
                "schema": deepcopy(schema),
                "maxOutputTokens": max_output_tokens,
            }
        )
        if name == COMPACT_RESPONSE_NAME:
            cards = {
                binding["candidateId"]: compact_wire_candidate(
                    compact_card(binding, context["topicCard"], context["scenarioContract"])
                )
                for binding in context["candidateBindings"]
            }
            if self.malformed_compact_once:
                self.malformed_compact_once = False
                first_id = context["candidateBindings"][0]["candidateId"]
                cards[first_id]["t"] = "trajectory_wrong_identity"
            value = {"candidates": cards}
        else:
            candidate = context["compactCandidate"]
            value = {
                "premiseState": {
                    "situation": (
                        f"Selected candidate {candidate['candidateId']} expands its compact "
                        "state transition without target dialogue."
                    ),
                    "knownFacts": [f"bound candidate {candidate['candidateId']}"],
                    "uncertainty": [f"bounded uncertainty for {candidate['outcomeRoute']}"],
                    "policyConstraints": ["do not invent unavailable evidence"],
                },
                "trajectory": expanded_trajectory(candidate),
            }
        metadata = {
            "endpoint": (
                f"http://fake-cuda-lane-{(len(self.calls) - 1) % len(CUDA_DEVICES)}"
                "/v1/chat/completions"
            ),
            "model": "authentic-fake",
            "protocolAttempt": 1,
            "responseHash": content_hash(value),
            "responseFormat": "json_schema_strict",
            "responseName": name,
            "responseSchemaHash": content_hash(schema),
            "reasoningDisabled": True,
            "accelerator": "cuda",
            "cudaDevice": (len(self.calls) - 1) % len(CUDA_DEVICES),
            "finishReason": "stop",
            "usage": {},
        }
        metadata.update(deepcopy(self.metadata_overrides))
        return value, metadata


def all_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from all_keys(child)


@pytest.fixture(scope="module")
def production_stage_a(tmp_path_factory):
    output_root = tmp_path_factory.mktemp("compact-production-stage-a")
    request = request_fixture(
        scenario_count=1_000,
        primary=250,
        reserve=250,
        topic_count=50,
    )
    topics = [
        topic_fixture(
            request,
            topic_id=f"topic_compact_{topic_index:02d}",
            source_seed_id=f"S{topic_index + 1:02d}",
        )
        for topic_index in range(50)
    ]
    scenarios = [
        scenario_fixture(
            topic_index * 20 + scenario_index,
            topic_id=topics[topic_index]["topicId"],
        )
        for topic_index in range(50)
        for scenario_index in range(20)
    ]
    planner = FakePlanner()
    candidates, manifest = generate_compact_candidates(
        request=request,
        topics=topics,
        scenarios=scenarios,
        output_root=output_root,
        planner=planner,
        max_workers=3,
        required_scenario_count=1_000,
    )
    return {
        "outputRoot": output_root,
        "request": request,
        "topics": topics,
        "scenarios": scenarios,
        "planner": planner,
        "candidates": candidates,
        "manifest": manifest,
    }


def test_compact_schema_has_required_identity_map_and_bounded_response(
    tmp_path: Path,
) -> None:
    request = request_fixture(scenario_count=1, primary=1, reserve=0)
    topic = topic_fixture(request)
    scenario = scenario_fixture(0)
    schema = compact_response_schema(request, topic, scenario)
    assert "prefixItems" not in set(all_keys(schema))
    candidate_schema = schema["properties"]["candidates"]
    lineages = candidate_lineages(request, topic, scenario)
    candidate_ids = [lineage["candidateId"] for lineage in lineages]
    assert candidate_schema["type"] == "object"
    assert candidate_schema["additionalProperties"] is False
    assert candidate_schema["required"] == candidate_ids
    assert set(candidate_schema["properties"]) == set(candidate_ids)
    assert all(
        set(candidate_schema["properties"][candidate_id]["properties"]) == {"x"}
        for candidate_id in candidate_ids
    )
    response = {
        "candidates": {
                lineage["candidateId"]: compact_wire_candidate(
                    compact_card(lineage, topic, scenario)
                )
            for lineage in lineages
        }
    }
    assert len(canonical_json(response).encode("utf-8")) <= 8_192
    budget = measure_compact_response_bound(request, topic, scenario)
    assert budget["worstCaseSerializedBytes"] <= MAX_COMPACT_RESPONSE_BYTES
    assert budget["requestedOutputTokenLimit"] == STAGE_A_MAX_OUTPUT_TOKENS
    assert budget["reservedOutputTokens"] == STAGE_A_OUTPUT_TOKEN_HEADROOM == 1_024

    oversized = deepcopy(scenario)
    oversized["scenarioOutcomeSpace"] = ["x" * 1_000]
    planner = FakePlanner()
    with pytest.raises(FanoutError, match="worst-case serialized response"):
        generate_compact_candidates(
            request=request,
            topics=[topic],
            scenarios=[oversized],
            output_root=tmp_path / "oversized-budget",
            planner=planner,
            max_workers=1,
            required_scenario_count=1,
            require_production_counts=False,
        )
    assert planner.calls == []


def test_production_shape_cuda_endpoints_and_retry_bounds(tmp_path: Path) -> None:
    request = request_fixture(scenario_count=1, primary=1, reserve=0)
    topic = topic_fixture(request)
    scenario = scenario_fixture(0)
    planner = FakePlanner()
    with pytest.raises(FanoutError, match="50 topics x 20 scenarios x 10 leaves"):
        generate_compact_candidates(
            request=request,
            topics=[topic],
            scenarios=[scenario],
            output_root=tmp_path / "non-production",
            planner=planner,
            max_workers=1,
            required_scenario_count=1,
        )
    assert planner.calls == []
    with pytest.raises(FanoutError, match="max_attempts"):
        generate_compact_candidates(
            request=request,
            topics=[topic],
            scenarios=[scenario],
            output_root=tmp_path / "unbounded-stage",
            planner=planner,
            max_workers=1,
            max_attempts=MAX_STAGE_ATTEMPTS + 1,
            required_scenario_count=1,
            require_production_counts=False,
        )
    endpoints = (
        "http://lane0/v1/chat/completions",
        "http://lane1/v1/chat/completions",
        "http://lane2/v1/chat/completions",
    )
    with pytest.raises(FanoutError, match="protocol_attempts"):
        ThreeEndpointJsonSchemaClient(
            endpoints,
            "model",
            protocol_attempts=MAX_PROTOCOL_ATTEMPTS + 1,
        )
    with pytest.raises(FanoutError, match=r"HTTP\(S\)"):
        ThreeEndpointJsonSchemaClient(
            ("file:///tmp/cpu", endpoints[1], endpoints[2]),
            "model",
        )


def test_rejects_unattested_reasoning_and_target_bearing_input(tmp_path: Path) -> None:
    request = request_fixture(scenario_count=1, primary=1, reserve=0)
    topic = topic_fixture(request)
    scenario = scenario_fixture(0)
    planner = FakePlanner(metadata_overrides={"reasoningDisabled": False})
    with pytest.raises(FanoutError, match="reasoning_not_disabled"):
        generate_compact_candidates(
            request=request,
            topics=[topic],
            scenarios=[scenario],
            output_root=tmp_path / "reasoning",
            planner=planner,
            max_workers=1,
            max_attempts=1,
            required_scenario_count=1,
            require_production_counts=False,
        )
    assert read_jsonl(tmp_path / "reasoning" / "trajectory_candidate_audit.jsonl")[0][
        "failureCode"
    ] == "reasoning_not_disabled"
    leaked_scenario = deepcopy(scenario)
    leaked_scenario["startingState"]["targetDialogue"] = "forbidden target wording"
    target_planner = FakePlanner()
    with pytest.raises(FanoutError, match="target-bearing field targetDialogue"):
        generate_compact_candidates(
            request=request,
            topics=[topic],
            scenarios=[leaked_scenario],
            output_root=tmp_path / "target-leak",
            planner=target_planner,
            max_workers=1,
            required_scenario_count=1,
            require_production_counts=False,
        )
    assert target_planner.calls == []
    extra_field_scenario = deepcopy(scenario)
    extra_field_scenario["plannerNotes"] = "declarative but undeclared state"
    exact_planner = FakePlanner()
    with pytest.raises(FanoutError, match="fields are not exact"):
        generate_compact_candidates(
            request=request,
            topics=[topic],
            scenarios=[extra_field_scenario],
            output_root=tmp_path / "extra-field",
            planner=exact_planner,
            max_workers=1,
            required_scenario_count=1,
            require_production_counts=False,
        )
    assert exact_planner.calls == []


def test_stage_a_10k_cardinality_identity_target_freedom_and_resume(
    production_stage_a,
) -> None:
    assert (
        PRODUCTION_TOPIC_COUNT,
        PRODUCTION_SCENARIOS_PER_TOPIC,
        PRODUCTION_SCENARIO_COUNT,
        PRODUCTION_CANDIDATE_COUNT,
        PRODUCTION_PRIMARY_COUNT,
        PRODUCTION_RESERVE_COUNT,
        PRODUCTION_EXPANSION_COUNT,
    ) == (50, 20, 1_000, 10_000, 250, 250, 500)
    request = production_stage_a["request"]
    topics = production_stage_a["topics"]
    scenarios = production_stage_a["scenarios"]
    planner = production_stage_a["planner"]
    candidates = production_stage_a["candidates"]
    manifest = production_stage_a["manifest"]
    output_root = production_stage_a["outputRoot"]
    assert len(candidates) == 10_000
    assert manifest["candidateCount"] == 10_000
    assert manifest["leavesPerScenario"] == 10
    assert len(list((output_root / CHECKPOINT_ROOT / "stage_a_scenarios").glob("*.json"))) == 1_000
    assert len(list((output_root / CHECKPOINT_ROOT / "stage_a_candidates").glob("*.json"))) == 10_000
    assert manifest["checkpointMode"] == "immutable-content-addressed-per-candidate"
    assert manifest["candidateCheckpointCount"] == 10_000
    assert manifest["causalSiblingRoles"] == list(CAUSAL_SIBLING_ROLES)
    assert manifest["outputBudget"]["reservedOutputTokens"] == 1_024
    topic_by_id = {topic["topicId"]: topic for topic in topics}
    for scenario in (scenarios[0], scenarios[-1]):
        rows = [
            item
            for item in candidates
            if item["lineage"]["scenarioId"] == scenario["scenarioId"]
        ]
        assert len(rows) == 10
        expected = candidate_lineages(request, topic_by_id[scenario["topicId"]], scenario)
        assert [item["lineage"] for item in rows] == [compact_lineage(item) for item in expected]
        assert [item["candidateId"] for item in rows] == [item["candidateId"] for item in expected]
    assert not (set(all_keys(candidates)) & TARGET_DIALOGUE_FIELDS)
    assert all(
        not (set(all_keys(call["context"])) & TARGET_DIALOGUE_FIELDS)
        for call in planner.calls
    )
    calls_before_resume = len(planner.calls)
    resumed, _ = generate_compact_candidates(
        request=request,
        topics=topics,
        scenarios=scenarios,
        output_root=output_root,
        planner=planner,
        max_workers=3,
        required_scenario_count=1_000,
    )
    assert resumed == candidates
    assert len(planner.calls) == calls_before_resume


def test_stage_a_retries_whole_malformed_identity_response(tmp_path: Path) -> None:
    request = request_fixture(scenario_count=1, primary=1, reserve=0)
    topic = topic_fixture(request)
    scenario = scenario_fixture(0)
    planner = FakePlanner(malformed_compact_once=True)
    candidates, _ = generate_compact_candidates(
        request=request,
        topics=[topic],
        scenarios=[scenario],
        output_root=tmp_path,
        planner=planner,
        max_workers=1,
        max_attempts=2,
        required_scenario_count=1,
        require_production_counts=False,
    )
    compact_calls = [call for call in planner.calls if call["name"] == COMPACT_RESPONSE_NAME]
    assert len(compact_calls) == 2
    assert all(len(call["context"]["candidateBindings"]) == 10 for call in compact_calls)
    assert all(call["maxOutputTokens"] <= 4_096 for call in compact_calls)
    assert len(candidates) == 10
    response_paths = list((tmp_path / CHECKPOINT_ROOT / "stage_a_scenarios").glob("*.json"))
    checkpoint_paths = list((tmp_path / CHECKPOINT_ROOT / "stage_a_candidates").glob("*.json"))
    assert len(response_paths) == 1
    assert len(checkpoint_paths) == 10
    assert all(path.stat().st_mode & 0o222 == 0 for path in checkpoint_paths)
    checkpoint_paths[0].unlink()
    calls_before_partial_resume = len(planner.calls)
    resumed, _ = generate_compact_candidates(
        request=request,
        topics=[topic],
        scenarios=[scenario],
        output_root=tmp_path,
        planner=planner,
        max_workers=1,
        required_scenario_count=1,
        require_production_counts=False,
    )
    assert resumed == candidates
    assert len(planner.calls) == calls_before_partial_resume
    assert len(list((tmp_path / CHECKPOINT_ROOT / "stage_a_candidates").glob("*.json"))) == 10
    with pytest.raises(FanoutError, match="max_workers"):
        generate_compact_candidates(
            request=request,
            topics=[topic],
            scenarios=[scenario],
            output_root=tmp_path / "bad-workers",
            planner=planner,
            max_workers=4,
            required_scenario_count=1,
            require_production_counts=False,
        )
    response_paths[0].chmod(0o600)
    with pytest.raises(FanoutError, match="read-only regular file"):
        generate_compact_candidates(
            request=request,
            topics=[topic],
            scenarios=[scenario],
            output_root=tmp_path,
            planner=planner,
            max_workers=1,
            required_scenario_count=1,
            require_production_counts=False,
        )


def test_typed_disjoint_250_plus_250_and_selected_only_expansion_resume(
    production_stage_a,
) -> None:
    request = production_stage_a["request"]
    topics = production_stage_a["topics"]
    scenarios = production_stage_a["scenarios"]
    planner = production_stage_a["planner"]
    candidates = production_stage_a["candidates"]
    output_root = production_stage_a["outputRoot"]
    calls_before_selection = len(planner.calls)
    primary, reserve, selection_manifest = select_compact_candidates(
        request=request,
        topics=topics,
        scenarios=scenarios,
        candidates=candidates,
        output_root=output_root,
    )
    assert len(planner.calls) == calls_before_selection
    assert len(primary) == len(reserve) == 250
    assert selection_manifest["naturalLanguageScoring"] is False
    assert selection_manifest["causalSiblingRoles"] == list(CAUSAL_SIBLING_ROLES)
    assert {row["trajectoryId"] for row in primary}.isdisjoint(
        row["trajectoryId"] for row in reserve
    )
    assert {row["scenarioId"] for row in primary}.isdisjoint(
        row["scenarioId"] for row in reserve
    )
    assert len(read_jsonl(output_root / SELECTED_FILENAME)) == 250
    candidate_by_trajectory = {item["trajectoryId"]: item for item in candidates}
    tampered_primary = deepcopy(primary)
    tampered = tampered_primary[0]
    candidate = candidate_by_trajectory[tampered["trajectoryId"]]
    tampered["balanceDimensions"] = deepcopy(tampered["balanceDimensions"])
    tampered["balanceDimensions"]["style"] = ["forged_style"]
    tampered["selectionHash"] = _selection_hash(
        request,
        tampered["groupId"],
        tampered["selectionTier"],
        candidate,
        tampered["balanceDimensions"],
    )
    with pytest.raises(FanoutError, match="balanceDimensions"):
        _validate_selection_rows(
            request,
            tampered_primary,
            reserve,
            candidate_by_trajectory,
        )
    expansion_schema = full_trajectory_response_schema(candidate)
    assert "maxLength" not in set(all_keys(expansion_schema))
    trajectory_properties = expansion_schema["properties"]["trajectory"]["properties"]
    state_properties = trajectory_properties["semanticStateArc"]["items"]["properties"]
    schedule_properties = trajectory_properties["controlRevisionSchedule"]["items"][
        "properties"
    ]
    assert state_properties["evidence"]["properties"]["source"] == {
        "const": candidate["evidenceSource"]
    }
    assert schedule_properties["source"] == {"const": candidate["evidenceSource"]}
    assert trajectory_properties["postureArc"] == {
        "const": [
            candidate["postureTransition"]["from"],
            candidate["postureTransition"]["to"],
        ]
    }
    trajectories, expansion_manifest = expand_selected_candidates(
        request=request,
        topics=topics,
        scenarios=scenarios,
        candidates=candidates,
        primary=primary,
        reserve=reserve,
        output_root=output_root,
        planner=planner,
        max_workers=3,
    )
    expansion_calls = [
        call for call in planner.calls if call["name"] == EXPANSION_RESPONSE_NAME
    ]
    selected_ids = {row["trajectoryId"] for row in primary + reserve}
    expanded_ids = {
        call["context"]["compactCandidate"]["trajectoryId"] for call in expansion_calls
    }
    assert expanded_ids == selected_ids
    assert len(expansion_calls) == 500
    assert len(trajectories) == 500
    assert expansion_manifest["trajectoryCount"] == 500
    assert expansion_manifest["causalSiblingRoles"] == list(CAUSAL_SIBLING_ROLES)
    assert len(read_jsonl(output_root / TRAJECTORIES_FILENAME)) == 500
    assert len(read_jsonl(output_root / PRIMARY_FILENAME)) == 250
    assert len(read_jsonl(output_root / RESERVE_FILENAME)) == 250
    expansion_checkpoints = list(
        (output_root / CHECKPOINT_ROOT / "stage_c_expansions").glob("*.json")
    )
    assert expansion_checkpoints[0].stat().st_mode & 0o222 == 0
    calls_before_resume = len(planner.calls)
    resumed, _ = expand_selected_candidates(
        request=request,
        topics=topics,
        scenarios=scenarios,
        candidates=candidates,
        primary=primary,
        reserve=reserve,
        output_root=output_root,
        planner=planner,
        max_workers=3,
    )
    assert resumed == trajectories
    assert len(planner.calls) == calls_before_resume


def test_legacy_builder_rejects_sequential_v5_trajectory_planning() -> None:
    request = request_fixture(scenario_count=1, primary=1, reserve=0)
    with pytest.raises(CascadeError, match="build_compact_trajectory_fanout_v5.py"):
        reject_legacy_v5_trajectory_stage(request, "trajectories")
    reject_legacy_v5_trajectory_stage(request, "selection")


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


def test_three_endpoint_client_retries_malformed_protocol_with_reasoning_disabled(
    monkeypatch,
) -> None:
    calls: list[str] = []
    payloads: list[dict] = []

    def urlopen(request, timeout):
        calls.append(request.full_url)
        payloads.append(json.loads(request.data))
        if len(calls) == 1:
            truncated_content = json.dumps({"ok": True})
            return FakeResponse(
                json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "length",
                                "message": {"content": truncated_content},
                            }
                        ],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                    }
                ).encode()
            )
        content = json.dumps({"ok": True})
        return FakeResponse(
            json.dumps(
                {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": content}}
                    ],
                    "usage": {
                        "prompt_tokens": 101,
                        "completion_tokens": 4,
                        "total_tokens": 105,
                    },
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = ThreeEndpointJsonSchemaClient(
        "http://lane0/v1/chat/completions,http://lane1/v1/chat/completions,http://lane2/v1/chat/completions",
        "robit/ornith:35b",
        protocol_attempts=3,
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok"],
        "properties": {"ok": {"const": True}},
    }
    value, metadata = client.generate(
        name="focused_protocol",
        schema=schema,
        instructions="return schema",
        context={"task": "focused"},
        max_output_tokens=50,
    )
    assert value == {"ok": True}
    assert metadata["protocolAttempt"] == 2
    assert metadata["finishReason"] == "stop"
    assert metadata["usage"]["total_tokens"] == 105
    assert metadata["accelerator"] == "cuda"
    assert metadata["cudaDevice"] == 1
    assert metadata["responseName"] == "focused_protocol"
    assert metadata["responseSchemaHash"] == content_hash(schema)
    assert calls == [
        "http://lane0/v1/chat/completions",
        "http://lane1/v1/chat/completions",
    ]
    assert all(payload["reasoning"] == {"enabled": False} for payload in payloads)
    assert all(payload["response_format"]["type"] == "json_schema" for payload in payloads)
    assert all(payload["response_format"]["json_schema"]["strict"] is True for payload in payloads)
    assert all("seed" not in payload for payload in payloads)
    two_lane_client = ThreeEndpointJsonSchemaClient(
        "http://one/v1/chat/completions,http://two/v1/chat/completions",
        "model",
    )
    assert two_lane_client.endpoints == (
        "http://one/v1/chat/completions",
        "http://two/v1/chat/completions",
    )
