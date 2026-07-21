from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import urllib.error

from ground_truth_finetuning.training.efficient_v5_fanout import (
    CANDIDATE_AUDIT_FILENAME,
    CANDIDATE_MANIFEST_FILENAME,
    CANDIDATES_FILENAME,
    COMBINED_MANIFEST_FILENAME,
    EXPANSION_MANIFEST_FILENAME,
    PRIMARY_FILENAME,
    RESERVE_FILENAME,
    SELECTED_FILENAME,
    TRAJECTORIES_FILENAME,
    RoundRobinJsonSchemaPlanner,
    content_hash,
    expand_selected_candidates,
    generate_compact_candidates,
    read_jsonl,
    select_compact_candidates,
    validate_expanded_trajectory,
    write_combined_manifest,
)


REQUEST_PATH = Path(__file__).resolve().parents[1] / "requests" / "personaplex_diverse_50x20x10.control-v5.json"


def request_fixture() -> dict:
    value = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    value["requestId"] = "efficient-v5-focused"
    value["coverageTarget"].update({
        "candidateTopics": 1,
        "scenariosPerTopic": 1,
        "trajectorySeedsPerScenario": 10,
        "primaryGroups": 2,
        "reserveGroups": 2,
        "selectedCounterfactualGroups": 2,
        "branchesPerGroup": 4,
    })
    return value


def topic_fixture(request: dict) -> dict:
    return {
        "schema": "personaplex.topic-card.v2",
        "topicId": "topic_focused",
        "sourceSeedId": "S01",
        "seedRevision": request["seedRevision"],
        "domain": "community problem solving",
        "interactionModes": ["cooperative", "skeptical", "conditional"],
        "registerRange": ["casual", "neutral", "formal"],
        "safeStakes": ["planning", "clarification", "handoff"],
        "forbiddenPatterns": ["target dialogue", "placeholder identity"],
        "diversityTags": ["community", "technical", "creative"],
        "causalAffordances": [
            {"family": "semantic", "operatorId": "evidence_status_transition", "changedPath": "state.evidence.status"},
            {"family": "delivery", "operatorId": "next_goal_transition", "changedPath": "plan.nextGoal"},
            {"family": "turn_taking", "operatorId": "interruption_transition", "changedPath": "turnTaking.eventType"},
        ],
    }


def scenario_fixture() -> dict:
    return {
        "schema": "personaplex.scenario-contract.v2",
        "scenarioId": "scenario_focused",
        "topicId": "topic_focused",
        "mode": "collaborative_resolution",
        "premise": "Two neighbors are comparing several safe ways to coordinate shared workshop hours without assuming the other person's availability.",
        "participants": [
            {"role": "caller", "knowledge": "knows their constraints and what has already been attempted"},
            {"role": "agent", "knowledge": "knows only facts introduced through the rolling semantic state"},
        ],
        "startingState": {
            "knownFacts": ["shared space is used by both neighbors"],
            "uncertainty": ["the preferred time window is unresolved"],
            "policyConstraints": ["do not invent agreement or availability"],
        },
        "interactionOpportunity": ["clarify constraints", "compare options"],
        "allowedToolClasses": ["calendar_lookup", "note_capture"],
        "disallowedClaims": ["guaranteed agreement", "invented schedule"],
        "scenarioOutcomeSpace": ["conditional agreement", "clarification", "handoff"],
        "requiredControlPhenomena": ["tool update", "caller posture change", "interruption recovery"],
    }


def compact_leaf(slot: int, *, variant: int | None = None) -> dict:
    value = slot if variant is None else variant
    target_turns = 8 + (value % 4) * 2
    agent_targets = target_turns // 2
    pivot = 2 + value % max(1, agent_targets - 2)
    barge_target = min(agent_targets - 1, pivot + (value % 2))
    extra_types = ("completed_turn", "brief_overlap", "backchannel")
    return {
        "slot": slot,
        "premise": f"Trajectory {value} explores a distinct evidence-dependent workshop coordination constraint with concrete uncertainty and a reversible outcome path.",
        "interactionArc": [f"surface constraint {value}", f"test evidence {value}", f"revise option {value}"],
        "postureArc": [f"guarded-{value}", f"conditional-{value}", f"resolved-{value}"],
        "postureTransition": {"from": f"guarded-{value}", "to": f"conditional-{value}"},
        "evidenceSource": (
            "asr_fact", "tool_result", "policy_decision", "caller_posture_change",
            "interruption", "handoff", "scenario_state", "state_reducer",
        )[value % 8],
        "evidenceEvolution": [f"uncertain evidence {value}", f"bounded update {value}", f"confirmed state {value}"],
        "duplexEvents": [
            {"eventType": "barge_in", "targetOrdinal": barge_target, "offsetMs": 420 + value * 7, "overlapMs": 120 + value, "cancelOutgoingAudio": False, "invalidateGeneration": False},
            {"eventType": "cancelled_generation", "targetOrdinal": barge_target, "offsetMs": 560 + value * 7, "overlapMs": 120 + value, "cancelOutgoingAudio": True, "invalidateGeneration": True},
            {"eventType": "recovery", "targetOrdinal": min(agent_targets, barge_target + 1), "offsetMs": 180 + value * 3, "overlapMs": 0, "cancelOutgoingAudio": False, "invalidateGeneration": False},
            {"eventType": extra_types[value % 3], "targetOrdinal": 1 + value % agent_targets, "offsetMs": 90 + value * 5, "overlapMs": value * 11, "cancelOutgoingAudio": False, "invalidateGeneration": False},
        ],
        "outcomeRoute": f"bounded_outcome_route_{value}",
        "conversationLength": {"targetTurns": target_turns, "min": target_turns - 1, "max": target_turns + 2},
        "style": {
            "pace": f"measured_{value}",
            "openingStyle": f"in_media_res_{value}",
            "closingStyle": f"state_grounded_{value}",
            "register": f"situational_{value}",
            "turnCadence": f"adaptive_{value}",
        },
        "pivot": {"from": f"pending-{value}", "to": f"resolved-{value}", "targetOrdinal": pivot},
        "controlPhenomena": [f"typed revision {value}", f"interruption invalidation {value}", f"state-grounded recovery {value}"],
    }


def expanded_trajectory(candidate: dict, *, invalidate_schedule: bool = False) -> dict:
    turns = candidate["conversationLength"]["targetTurns"] // 2
    pivot = candidate["counterfactualPivotOrdinal"]
    revisions = [10 + ordinal for ordinal in range(1, turns + 1)]
    schedule = [
        {
            "controlRevision": revision,
            "targetOrdinal": ordinal,
            "availableBeforeTarget": not (invalidate_schedule and ordinal == 1),
            "source": candidate["evidenceSource"],
        }
        for ordinal, revision in enumerate(revisions, start=1)
    ]
    states = []
    for ordinal, revision in enumerate(revisions, start=1):
        active = candidate["typedPivot"]["from"] if ordinal < pivot else candidate["typedPivot"]["to"]
        states.append({
            "targetOrdinal": ordinal,
            "phase": f"phase-{ordinal}",
            "availableBeforeTarget": True,
            "controlRevision": revision,
            "knownFacts": [f"bounded fact {ordinal}"],
            "uncertainty": [f"remaining uncertainty {ordinal}"],
            "policyConstraints": ["do not invent agreement"],
            "commitments": [f"bounded commitment state {ordinal}"],
            "callerPosture": candidate["postureArc"][min(ordinal - 1, len(candidate["postureArc"]) - 1)],
            "nextGoal": f"advance the evidence-grounded outcome at target {ordinal}",
            "evidence": {"source": candidate["evidenceSource"], "status": f"state-{ordinal}", "facts": [f"evidence fact {ordinal}"]},
            "toolResult": {"source": "bounded_fixture_tool", "status": f"observed-{ordinal}", "facts": [f"tool fact {ordinal}"]},
            "causalState": {
                "causalIdentity": candidate["causalIdentity"],
                "operatorId": candidate["operatorAssignment"]["operatorId"],
                "changedPath": candidate["typedPivot"]["field"],
                "from": candidate["typedPivot"]["from"],
                "to": candidate["typedPivot"]["to"],
                "activeValue": active,
            },
            "revisionReason": f"new typed state became available before target {ordinal}",
        })
    return {
        "schema": "personaplex.trajectory-seed.v2",
        "trajectoryId": candidate["trajectoryId"],
        "scenarioId": candidate["scenarioId"],
        "conversationLength": candidate["conversationLength"],
        "pace": candidate["style"]["pace"],
        "openingStyle": candidate["style"]["openingStyle"],
        "closingStyle": candidate["style"]["closingStyle"],
        "voicePairPolicy": "distinct_approved_references",
        "interactionArc": candidate["interactionArc"],
        "duplexEvents": candidate["duplexEvents"],
        "postureArc": candidate["postureArc"],
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
        "terminationContract": {"decisionSource": "model", "action": "end_call_tool", "deterministicPhrase": False},
        "negativeControlCoverage": ["paired_wrong_branch", "stale_revision", "null_control"],
    }


class FakePlanner:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.failed_expansion: set[str] = set()

    def generate(self, *, name, schema, instructions, context, max_output_tokens):
        self.calls.append({"name": name, "context": deepcopy(context), "schema": deepcopy(schema)})
        if name == "personaplex_compact_trajectories_v5":
            slots = [item["slot"] for item in context["slots"]]
            leaves = []
            for slot in slots:
                if len(slots) == 10 and slot == 9:
                    leaf = compact_leaf(8)
                    leaf["slot"] = 9
                else:
                    leaf = compact_leaf(slot, variant=slot + (100 if len(slots) < 10 else 0))
                leaves.append(leaf)
            value = {"candidates": leaves}
        else:
            candidate = context["candidate"]
            invalid = candidate["trajectoryId"] not in self.failed_expansion
            self.failed_expansion.add(candidate["trajectoryId"])
            value = {"trajectory": expanded_trajectory(candidate, invalidate_schedule=invalid)}
        return value, {
            "endpoint": "fake://planner",
            "model": "focused-fake",
            "responseHash": content_hash(value),
            "responseFormat": "json_schema_strict",
        }


def test_end_to_end_fanout_repairs_only_failed_units_and_resumes(tmp_path: Path) -> None:
    request = request_fixture()
    topics = [topic_fixture(request)]
    scenarios = [scenario_fixture()]
    planner = FakePlanner()
    candidates, candidate_manifest = generate_compact_candidates(
        request=request,
        topics=topics,
        scenarios=scenarios,
        output_root=tmp_path,
        planner=planner,
        max_workers=1,
        max_attempts=3,
    )
    candidate_calls = [call for call in planner.calls if call["name"] == "personaplex_compact_trajectories_v5"]
    assert [[item["slot"] for item in call["context"]["slots"]] for call in candidate_calls] == [
        list(range(10)), [9]
    ]
    assert len(candidates) == 10
    assert candidate_manifest["candidateCount"] == 10
    assert len(list((tmp_path / ".efficient_v5_checkpoints" / "candidates").glob("*.json"))) == 10
    calls_before_resume = len(planner.calls)
    resumed, _manifest = generate_compact_candidates(
        request=request,
        topics=topics,
        scenarios=scenarios,
        output_root=tmp_path,
        planner=planner,
        max_workers=1,
        max_attempts=3,
    )
    assert resumed == candidates
    assert len(planner.calls) == calls_before_resume

    primary, reserve, selection_manifest = select_compact_candidates(
        request=request,
        topics=topics,
        scenarios=scenarios,
        candidates=candidates,
        output_root=tmp_path,
    )
    assert len(primary) == len(reserve) == 2
    assert {row["trajectoryId"] for row in primary}.isdisjoint(
        row["trajectoryId"] for row in reserve
    )
    assert selection_manifest["primaryCount"] == 2
    trajectories, expansion_manifest = expand_selected_candidates(
        request=request,
        topics=topics,
        scenarios=scenarios,
        candidates=candidates,
        primary=primary,
        reserve=reserve,
        output_root=tmp_path,
        planner=planner,
        max_workers=1,
        max_attempts=3,
    )
    expansion_calls = [call for call in planner.calls if call["name"] == "personaplex_full_trajectory_v2"]
    assert len(expansion_calls) == 8
    assert len(trajectories) == 4
    candidate_by_id = {row["trajectoryId"]: row for row in candidates}
    for trajectory in trajectories:
        validate_expanded_trajectory(
            trajectory, candidate_by_id[trajectory["trajectoryId"]], {"scenario_focused"}
        )
        assert len(trajectory["semanticStateArc"]) == trajectory["conversationLength"]["targetTurns"] // 2
        assert all(item["availableBeforeTarget"] for item in trajectory["controlRevisionSchedule"])
    assert expansion_manifest["trajectoryCount"] == 4
    calls_before_expansion_resume = len(planner.calls)
    resumed_trajectories, _ = expand_selected_candidates(
        request=request,
        topics=topics,
        scenarios=scenarios,
        candidates=candidates,
        primary=primary,
        reserve=reserve,
        output_root=tmp_path,
        planner=planner,
        max_workers=1,
        max_attempts=3,
    )
    assert resumed_trajectories == trajectories
    assert len(planner.calls) == calls_before_expansion_resume

    combined = write_combined_manifest(tmp_path, request)
    assert combined["files"][CANDIDATES_FILENAME].startswith("sha256:")
    for name in (
        CANDIDATES_FILENAME,
        CANDIDATE_AUDIT_FILENAME,
        CANDIDATE_MANIFEST_FILENAME,
        PRIMARY_FILENAME,
        RESERVE_FILENAME,
        SELECTED_FILENAME,
        TRAJECTORIES_FILENAME,
        EXPANSION_MANIFEST_FILENAME,
        COMBINED_MANIFEST_FILENAME,
    ):
        assert (tmp_path / name).is_file()
    assert len(read_jsonl(tmp_path / TRAJECTORIES_FILENAME)) == 4


class FakeResponse:
    def __init__(self, value: dict):
        content = json.dumps(value)
        self.body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.body


def test_three_endpoint_round_robin_and_transport_only_failover(monkeypatch) -> None:
    calls: list[str] = []
    payloads: list[dict] = []
    fail_first = True

    def urlopen(request, timeout):
        nonlocal fail_first
        calls.append(request.full_url)
        payloads.append(json.loads(request.data))
        if fail_first:
            fail_first = False
            raise urllib.error.URLError("temporary lane failure")
        return FakeResponse({"ok": True})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    planner = RoundRobinJsonSchemaPlanner(
        ["http://lane0/v1/chat/completions", "http://lane1/v1/chat/completions", "http://lane2/v1/chat/completions"],
        "robit/ornith:35b",
    )
    schema = {
        "type": "object", "additionalProperties": False,
        "required": ["ok"], "properties": {"ok": {"const": True}},
    }
    for _ in range(3):
        value, metadata = planner.generate(
            name="focused_schema",
            schema=schema,
            instructions="return schema",
            context={"task": "focused"},
            max_output_tokens=50,
        )
        assert value == {"ok": True}
        assert metadata["responseFormat"] == "json_schema_strict"
    assert calls == [
        "http://lane0/v1/chat/completions",
        "http://lane1/v1/chat/completions",
        "http://lane1/v1/chat/completions",
        "http://lane2/v1/chat/completions",
    ]
    assert all(payload["response_format"]["type"] == "json_schema" for payload in payloads)
    assert all(payload["response_format"]["json_schema"]["strict"] is True for payload in payloads)
    assert all(payload["reasoning"] is False for payload in payloads)
