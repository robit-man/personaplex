from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import urllib.error

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.tools.compile_diverse_cascade_voryn_plan import (
    build_common_context,
    build_shared_prefix_plan,
    create_templates,
    materialize_v8_plan,
    validate_control_program,
)
from ground_truth_finetuning.training.contracts import validate_control_frame_mapping
from ground_truth_finetuning.training.diverse_cascade import (
    CascadeError,
    JsonOnlyPlanner,
    PlannerConfig,
    content_hash,
    planner_config_hash,
    request_sibling_roles,
    validate_pair_spec,
)


ROLES = (
    "verified_positive",
    "verified_negative",
    "uncertain",
    "superseded",
)


class StaticPlanner:
    def __init__(self, templates: list[dict]):
        self.templates = templates

    def call(self, _system: str, _user: str) -> dict:
        return {"planTemplates": deepcopy(self.templates)}


def request(siblings: int = 4) -> dict:
    result = {
        "schema": "personaplex.diverse-corpus-request.v2",
        "requestId": "compiler-v5-focused",
        "seedRevision": "sha256:" + "1" * 64,
        "seedIdeas": ["safe diverse interactions"],
        "coverageTarget": {
            "candidateTopics": 1,
            "scenariosPerTopic": 1,
            "trajectorySeedsPerScenario": 1,
            "primaryGroups": 1,
            "reserveGroups": 0,
            "siblingsPerGroup": siblings,
        },
        "allowedVoicesManifest": "sha256:" + "2" * 64,
        "renderer": "voicebox_chatterbox_turbo",
        "asr": "whisper",
        "allowedPhysicalCudaDevices": [0, 1, 2],
        "prohibitedContentPolicyRevision": "safe-v1",
        "strategyVersion": "semantic-control-v5",
    }
    if siblings == 4:
        result["causalGroupContract"] = {"siblingRoles": list(ROLES)}
    return result


def trajectory() -> dict:
    return {
        "schema": "personaplex.trajectory-seed.v1",
        "trajectoryId": "trajectory-v5",
        "scenarioId": "scenario-v5",
        "conversationLength": {"targetTurns": 8, "min": 8, "max": 12},
        "pace": "natural",
        "openingStyle": "in_media_res",
        "closingStyle": "model_selected",
        "voicePairPolicy": "distinct_approved_references",
        "interactionArc": ["clarify", "resolve"],
        "duplexEvents": [
            {"eventType": "barge_in", "targetOrdinal": 2, "bargeInAtMs": 640, "cutoffAtMs": 700},
            {"eventType": "recovery", "targetOrdinal": 3, "resumeAfterMs": 180},
        ],
        "postureArc": ["skeptical", "conditional"],
        "counterfactualPivotOrdinal": 2,
        "controlPhenomena": ["typed_revision", "barge_in_recovery"],
        "causalAxis": "evidence_status",
        "interventionFamily": "semantic",
        "typedPivot": {"field": "evidence.status", "from": "pending", "to": "updated"},
        "postureTransition": {"from": "skeptical", "to": "conditional"},
        "evidenceSource": "tool_result",
        "outcomeRoute": "resolve_or_handoff",
        "controlRevisionSchedule": [
            {"targetOrdinal": index, "controlRevision": 10 + index, "availableBeforeTarget": True, "source": "state_reducer"}
            for index in range(1, 5)
        ],
        "terminationContract": {
            "decisionSource": "model",
            "action": "end_call_tool",
            "deterministicPhrase": False,
        },
    }


def pair(use_siblings_alias: bool = True) -> dict:
    states = {
        "verified_positive": "verified",
        "verified_negative": "rejected",
        "uncertain": "pending_review",
        "superseded": "superseded",
    }
    members = [
        {
            "siblingRole" if use_siblings_alias else "branchId": role,
            "controlDelta": {"field": "evidence.status", "from": "pending", "to": state},
            "controlValue": state,
            "evidenceUpdate": {"source": "tool_result", "status": state},
            "availabilityTiming": {"availableBeforeTarget": True, "controlRevision": 20 + index},
            "negativeControls": ["paired_wrong_branch", "stale_revision", "null_control"],
            "semanticAssertions": [f"use the {state} state without inventing facts"],
        }
        for index, (role, state) in enumerate(states.items())
    ]
    common_context = build_common_context(
        request(), selection(), {"scenarioId": "scenario-v5"}, trajectory()
    )
    result = {
        "schema": "personaplex.counterfactual-sibling-group-spec.v2",
        "groupId": "compiler-v5-group-0001",
        "trajectoryId": "trajectory-v5",
        "pivotOrdinal": 2,
        "commonContextHash": content_hash(common_context),
        "interventionFamily": "semantic",
        "typedPivot": {"field": "evidence.status", "from": "pending", "to": "updated"},
        "sharedPrefixPolicy": "native_code_identical_through_pivot",
    }
    result["siblings" if use_siblings_alias else "branches"] = members
    return result


def control(role: str, ordinal: int) -> dict:
    return {
        "id": f"control-{role}-{ordinal}",
        "targetOrdinal": ordinal,
        "source": "state_reducer",
        "kind": "inform" if ordinal < 4 else "complete",
        "facts": ["a bounded fact"],
        "commitments": [],
        "uncertainty": ["one unresolved detail"],
        "policyConstraints": ["do not invent evidence"],
        "toolResult": {"status": "bounded"} if ordinal >= 2 else None,
        "nextGoal": "advance the safe interaction naturally",
        "guidance": "respond naturally from the current typed state without prescribed wording",
        "semanticContext": {"callerPosture": "skeptical"},
        "endCall": ordinal == 4,
        "requiredFacts": ["a bounded fact"],
        "forbiddenClaims": ["unsupported certainty"],
        "mustAsk": [],
        "updateReason": "state revision available",
        "expectedEffects": ["clear"],
    }


def template(role: str, *, role_alias: bool = True) -> dict:
    result = {
        "topic": "Evidence changes during a natural conversation",
        "topicFamily": "general_service",
        "topicSeedId": "seed-v5",
        "contextLens": "bounded scenario context",
        "conversationMode": "full_duplex",
        "lengthProfile": "medium",
        "turnCadence": "natural",
        "responseLengthProfile": "concise",
        "openingStyle": "in_media_res",
        "closingStyle": "model_selected",
        "coverageProfile": "typed_causal",
        "turns": 8,
        "coverage": {
            "intent": "resolve the evidence-dependent issue",
            "trajectory": "clarify then resolve",
            "interactionClass": "conditional cooperation",
            "speechStyle": "neutral",
            "turnPattern": "duplex",
            "nextGoal": "reach the state-supported outcome",
            "controlSources": ["state_reducer", "tool_result"],
            "requiredStateFields": ["evidence.status"],
            "requireControlForAllTargets": True,
        },
        "controlProgram": [control(role, ordinal) for ordinal in range(1, 5)],
        "dynamics": {
            "assertiveness": 45,
            "skepticism": 60,
            "compliance": 40,
            "resistance": 35,
            "recovery": 70,
            "hesitation": 20,
            "pace": 55,
            "interruption": 65,
        },
        "branchInstruction": f"follow the typed {role} state",
    }
    result["siblingRole" if role_alias else "branchId"] = role
    return result


def selection() -> dict:
    return {
        "groupId": "compiler-v5-group-0001",
        "topicId": "topic-v5",
        "scenarioId": "scenario-v5",
        "trajectoryId": "trajectory-v5",
        "selectionHash": "sha256:" + "4" * 64,
    }


def test_v5_compiles_all_siblings_with_typed_frames_and_one_shared_prefix() -> None:
    req = request()
    traj = trajectory()
    group = pair()
    validate_pair_spec(group, {traj["trajectoryId"]}, request=req, trajectory=traj)
    templates = create_templates(
        StaticPlanner([template(role) for role in ROLES]),
        req,
        {"topicId": "topic-v5"},
        {"scenarioId": "scenario-v5"},
        traj,
        group,
    )
    assert {item["branchId"] for item in templates} == set(ROLES)
    shared = build_shared_prefix_plan(
        req, selection(), traj, group, templates,
        scenario={"scenarioId": "scenario-v5"},
    )
    shared_without_hash = deepcopy(shared)
    shared_hash = shared_without_hash.pop("sharedPrefixHash")
    assert shared_hash == content_hash(shared_without_hash)
    plans = [
        materialize_v8_plan(
            req, selection(), traj, group, item, "caller", "agent", 0, shared
        )
        for item in templates
    ]
    assert len(plans) == len(request_sibling_roles(req)) == 4
    assert {row["counterfactual"]["siblingRole"] for row in plans} == set(ROLES)
    assert len({row["sharedPrefixRef"]["sharedPrefixHash"] for row in plans}) == 1
    assert all(row["sharedPrefixRef"]["renderOnce"] for row in plans)
    assert shared["sharedPrefixId"].startswith("sha256:")
    assert shared["commonContextHash"] == content_hash(shared["commonContext"])
    shared_prefix_controls = shared["controlProgram"]
    assert [item["targetOrdinal"] for item in shared_prefix_controls] == [1]
    shared_terminal_state_hash = shared_prefix_controls[-1]["controlFrame"]["stateHash"]
    sibling_prefixes = [
        [item for item in row["controlProgram"] if item["targetOrdinal"] < group["pivotOrdinal"]]
        for row in plans
    ]
    assert all(prefix == shared_prefix_controls for prefix in sibling_prefixes)
    assert len({
        (
            prefix[0]["controlFrameHash"],
            prefix[0]["controlRevision"],
            json.dumps(prefix[0]["controlFrame"], sort_keys=True),
        )
        for prefix in sibling_prefixes
    }) == 1
    pivot_controls = [
        next(item for item in row["controlProgram"] if item["targetOrdinal"] == group["pivotOrdinal"])
        for row in plans
    ]
    assert all(
        item["controlFrame"]["baseStateHash"] == shared_terminal_state_hash
        for item in pivot_controls
    )
    assert len({item["controlFrameHash"] for item in pivot_controls}) == 4
    assert all(
        prefix[0]["controlFrame"]["conversationId"] == shared["sharedPrefixId"]
        for prefix in sibling_prefixes
    )
    assert all(
        item["controlFrame"]["conversationId"] == row["scenarioKey"]
        for row, item in zip(plans, pivot_controls)
    )
    for row in plans:
        cascade = row["cascade"]
        assert cascade["commonContextHash"] == group["commonContextHash"]
        assert cascade["sharedPrefixPolicy"] == "native_code_identical_through_pivot"
        assert cascade["controlDelta"] == row["counterfactual"]["controlDelta"]
        assert cascade["controlRevisionSchedule"] == traj["controlRevisionSchedule"]
        assert cascade["duplexEvents"] == traj["duplexEvents"]
        assert cascade["terminationContract"] == traj["terminationContract"]
        assert cascade["availabilityTiming"]["availableBeforeTarget"] is True
        assert cascade["negativeControls"]
        assert cascade["semanticAssertions"]
        bridge = row["postRenderBridge"]
        assert row["renderPlanId"] == content_hash(bridge)
        assert bridge["commonContext"] == shared["commonContext"]
        assert bridge["commonContextHash"] == content_hash(bridge["commonContext"])
        assert bridge["pivotTargetOrdinal"] == 2
        assert bridge["pivotControlBinding"]["frameHash"] == row["controlProgram"][1]["controlFrameHash"]
        assert bridge["pivotControlBinding"]["revision"] == row["controlProgram"][1]["controlRevision"]
        assert "agentText" not in json.dumps(bridge)
        assert len(row["controlProgram"]) == 4
        for entry in row["controlProgram"]:
            frame = validate_control_frame_mapping(entry["controlFrame"])
            assert frame.frame_hash == entry["controlFrameHash"]
            assert frame.state_revision == entry["controlRevision"]
            assert "branchId" not in frame.state
            assert "siblingRole" not in frame.state
            assert set(frame.state["causalControl"]) == {
                "activeValue", "axis", "effectiveAtThisTarget", "field",
                "interventionFamily",
            }
        pivot_frame = row["controlProgram"][1]["controlFrame"]
        assert pivot_frame["turnTaking"]["duplexEvents"][0]["bargeInAtMs"] == 640
        assert pivot_frame["turnTaking"]["duplexEvents"][0]["cutoffAtMs"] == 700
        assert row["controlProgram"][-1]["endCall"] is True


def test_end_call_must_be_unique_and_at_final_target() -> None:
    program = [control("available", ordinal) for ordinal in range(1, 5)]
    program[-1]["endCall"] = False
    program[1]["endCall"] = True
    with pytest.raises(CascadeError, match="final target ordinal"):
        validate_control_program(program, "available", 8)
    program[1]["endCall"] = False
    program.pop(2)
    with pytest.raises(CascadeError, match="every agent target"):
        validate_control_program(program, "available", 8)


def test_legacy_two_branch_templates_remain_supported() -> None:
    req = request(siblings=2)
    req.pop("strategyVersion")
    legacy_pair = pair(use_siblings_alias=False)
    legacy_pair["schema"] = "personaplex.counterfactual-pair-spec.v1"
    legacy_pair["branches"] = [
        {
            "branchId": role,
            "controlDelta": {"field": "tool.status", "from": "pending", "to": state},
            "evidenceUpdate": {"status": state},
        }
        for role, state in (("available", "ready"), ("constrained", "failed"))
    ]
    legacy_pair.pop("siblings", None)
    legacy_pair.pop("interventionFamily")
    legacy_pair.pop("typedPivot")
    legacy_pair.pop("sharedPrefixPolicy")
    legacy_templates = [template(role, role_alias=False) for role in ("available", "constrained")]
    compiled = create_templates(
        StaticPlanner(legacy_templates), req, {}, {}, trajectory(), legacy_pair
    )
    assert [item["branchId"] for item in compiled] == ["available", "constrained"]


class FakeResponse:
    def __init__(self, content: str):
        self.body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.body


def test_planner_round_robin_transport_failover_and_normalized_hash(monkeypatch) -> None:
    calls: list[str] = []
    failed_once = False

    def urlopen(req, timeout):
        nonlocal failed_once
        calls.append(req.full_url)
        if not failed_once:
            failed_once = True
            raise urllib.error.URLError("temporarily unavailable")
        return FakeResponse('{"ok":true}')

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    config = PlannerConfig(
        " http://127.0.0.1:12086/v1/chat/completions/, http://127.0.0.1:12084/v1/chat/completions, http://127.0.0.1:12086/v1/chat/completions ",
        "model",
        "",
    )
    planner = JsonOnlyPlanner(config)
    assert planner.call("system", "user") == {"ok": True}
    assert calls == [
        "http://127.0.0.1:12086/v1/chat/completions",
        "http://127.0.0.1:12084/v1/chat/completions",
    ]
    calls.clear()
    assert planner.call("system", "user") == {"ok": True}
    assert calls == ["http://127.0.0.1:12084/v1/chat/completions"]
    normalized = PlannerConfig(
        "http://127.0.0.1:12086/v1/chat/completions,http://127.0.0.1:12084/v1/chat/completions",
        "model",
        "",
    )
    assert planner_config_hash(config) == planner_config_hash(normalized)


def test_planner_never_fails_over_on_malformed_semantic_output(monkeypatch) -> None:
    calls: list[str] = []

    def urlopen(req, timeout):
        calls.append(req.full_url)
        return FakeResponse("not-json")

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    planner = JsonOnlyPlanner(PlannerConfig("http://one.test,http://two.test", "model", ""))
    with pytest.raises(CascadeError, match="non-JSON content"):
        planner.call("system", "user")
    assert calls == ["http://one.test"]
