from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ground_truth_finetuning.training.scenario_scrutiny import (
    AuthenticScenarioJudge,
    DIMENSION_KEYS,
    ScenarioScrutinyError,
    scrutinize_scenarios,
)
from ground_truth_finetuning.training.diverse_cascade import PlannerConfig


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def topic(topic_id: str) -> dict:
    return {
        "schema": "personaplex.topic-card.v2",
        "topicId": topic_id,
        "domain": f"Broad domain for {topic_id}",
    }


def scenario(topic_id: str, ordinal: int, marker: str = "original") -> dict:
    scenario_id = f"scenario_{topic_id}_{ordinal:02d}"
    return {
        "schema": "personaplex.scenario-contract.v2",
        "scenarioId": scenario_id,
        "topicId": topic_id,
        "mode": f"distinct interaction mode {ordinal}",
        "premise": (
            f"A fully specified natural interaction premise {ordinal} in {topic_id} with "
            f"independent facts, uncertainty, and a consequential update; version {marker}."
        ),
        "participants": [
            {"role": "caller", "knowledge": f"Caller knowledge state {ordinal}."},
            {"role": "agent", "knowledge": f"Agent knowledge boundary {ordinal}."},
        ],
        "startingState": {
            "knownFacts": [f"Observed fact {ordinal} is available."],
            "uncertainty": [f"Evidence status {ordinal} is not yet resolved."],
            "policyConstraints": [f"Do not claim an outcome before evidence update {ordinal}."],
        },
        "interactionOpportunity": [f"Resolve or preserve uncertainty for case {ordinal}."],
        "allowedToolClasses": ["read_only_status_lookup"],
        "disallowedClaims": [f"Do not invent verification state {ordinal}."],
        "scenarioOutcomeSpace": [
            f"Verified-positive route {ordinal}.",
            f"Verified-negative route {ordinal}.",
            f"Uncertain route {ordinal}.",
            f"Superseded route {ordinal}.",
        ],
        "requiredControlPhenomena": [
            "A typed evidence revision causally changes the next goal across four sibling states."
        ],
    }


def fixture_root(tmp_path: Path, topic_ids: tuple[str, ...] = ("topic_alpha",)) -> tuple[Path, dict[str, dict]]:
    root = tmp_path / "run"
    topics = [topic(topic_id) for topic_id in topic_ids]
    rows = [scenario(topic_id, ordinal) for topic_id in topic_ids for ordinal in range(20)]
    by_id = {row["scenarioId"]: row for row in rows}
    write_json(root / "request.json", {
        "schema": "personaplex.diverse-corpus-request.v2",
        "strategyVersion": "semantic-control-v5",
        "coverageTarget": {"scenariosPerTopic": 20},
    })
    write_json(root / "run_manifest.json", {
        "schema": "personaplex.diverse-cascade-run.v2",
        "runId": "immutable-fixture-run",
    })
    write_jsonl(root / "topic_cards.jsonl", topics)
    write_jsonl(root / "scenario_contracts.jsonl", rows)
    for index, row in enumerate(rows):
        write_json(root / ".stage_checkpoints" / "scenarios" / f"checkpoint-{index:03d}.json", row)
    return root, by_id


def judge_result(topic_id: str, ids: list[str], rejected: set[str]) -> dict:
    failed = bool(rejected)
    return {
        "topicId": topic_id,
        "groupDecision": "reject" if failed else "pass",
        "groupRationale": "One contract requires authentic regeneration." if failed else "The complete group passes scrutiny.",
        "dimensionVerdicts": {
            key: {
                "status": "fail" if failed and key == "semanticDiversity" else "pass",
                "rationale": "A semantic collision is present." if failed and key == "semanticDiversity" else "This dimension passes.",
            }
            for key in DIMENSION_KEYS
        },
        "accepted": [
            {"scenarioId": scenario_id, "rationale": "Distinct, coherent, safe, and causally useful."}
            for scenario_id in ids if scenario_id not in rejected
        ],
        "rejected": [
            {
                "scenarioId": scenario_id,
                "findings": [{
                    "code": "semantic_near_duplicate",
                    "rationale": "Its meaning collapses onto another contract in this topic group.",
                    "relatedScenarioIds": [ids[0]] if scenario_id != ids[0] else [ids[1]],
                }],
            }
            for scenario_id in ids if scenario_id in rejected
        ],
    }


def model_judge_result(topic_id: str, ids: list[str], rejected: set[str]) -> dict:
    normalized = judge_result(topic_id, ids, rejected)
    accepted_by_id = {item["scenarioId"]: item for item in normalized["accepted"]}
    rejected_by_id = {item["scenarioId"]: item for item in normalized["rejected"]}
    return {
        "topicId": normalized["topicId"],
        "groupDecision": normalized["groupDecision"],
        "groupRationale": normalized["groupRationale"],
        "dimensionVerdicts": normalized["dimensionVerdicts"],
        "findings": [
            {
                "code": rejected_by_id[scenario_id]["findings"][0]["code"],
                "rationale": rejected_by_id[scenario_id]["findings"][0]["rationale"],
                "scenarioIds": [scenario_id],
            }
            for scenario_id in ids if scenario_id in rejected
        ],
    }


class FakeJudge:
    def __init__(self, reject_until_regenerated: dict[str, set[str]]):
        self.reject_until_regenerated = reject_until_regenerated
        self.calls: Counter[str] = Counter()

    def binding(self) -> dict:
        return {
            "protocol": "fake_openai_chat_completions",
            "model": "independent-fake-judge",
            "reasoning": {"enabled": False},
            "responseFormat": "strict_json_schema",
        }

    def audit_topic(self, topic_card: dict, scenarios: list[dict]) -> dict:
        topic_id = topic_card["topicId"]
        self.calls[topic_id] += 1
        rejected = {
            row["scenarioId"]
            for row in scenarios
            if row["scenarioId"] in self.reject_until_regenerated.get(topic_id, set())
            and not row["premise"].endswith("version regenerated.")
        }
        return judge_result(topic_id, [row["scenarioId"] for row in scenarios], rejected)


class FakePlanner:
    endpoints = ("http://planner-0/v1/chat/completions", "http://planner-1/v1/chat/completions")

    class Config:
        model = "authentic-fake-planner"

    config = Config()


class TargetedRepairer:
    def __init__(self, originals: dict[str, dict]):
        self.originals = originals
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def binding(self) -> dict:
        return {
            "protocol": "fake_openai_chat_completions",
            "model": "authentic-fake-repairer",
            "reasoning": {"enabled": False},
            "responseFormat": "strict_json_schema",
        }

    def repair_one(self, topic_card, original, admitted, rejected_context, judge_rejection):
        topic_id = topic_card["topicId"]
        self.calls.append((topic_id, tuple(sorted(row["scenarioId"] for row in admitted))))
        replacement = deepcopy(self.originals[original["scenarioId"]])
        replacement["premise"] = replacement["premise"].replace(
            "version original.", "version regenerated."
        )
        replacement["mode"] = f"regenerated interaction mode {original['scenarioId']}"
        replacement["requiredControlPhenomena"] = [
            f"regenerated causal control for {original['scenarioId']}"
        ]
        return replacement


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def checkpoint_for(root: Path, scenario_id: str) -> Path:
    matches = []
    for path in (root / ".stage_checkpoints" / "scenarios").glob("*.json"):
        if json.loads(path.read_text())["scenarioId"] == scenario_id:
            matches.append(path)
    assert len(matches) == 1
    return matches[0]


def test_targeted_repair_quarantines_one_id_and_preserves_all_admitted_checkpoints(tmp_path: Path) -> None:
    root, originals = fixture_root(tmp_path)
    rejected_id = "scenario_topic_alpha_07"
    admitted_id = "scenario_topic_alpha_08"
    admitted_checkpoint = checkpoint_for(root, admitted_id)
    admitted_bytes = admitted_checkpoint.read_bytes()
    rejected_checkpoint = checkpoint_for(root, rejected_id)
    rejected_bytes = rejected_checkpoint.read_bytes()
    run_identity_bytes = (root / "run_manifest.json").read_bytes()
    judge = FakeJudge({"topic_alpha": {rejected_id}})
    plan_path = TargetedRepairer(originals)

    report = scrutinize_scenarios(
        root,
        judge,
        planner=FakePlanner(),
        repair=True,
        max_workers=3,
        max_repair_rounds=2,
        scenario_repairer=plan_path,
    )

    assert report["status"] == "pass"
    assert len(plan_path.calls) == 1
    assert plan_path.calls[0][0] == "topic_alpha"
    assert len(plan_path.calls[0][1]) == 19
    assert rejected_id not in plan_path.calls[0][1]
    assert admitted_checkpoint.read_bytes() == admitted_bytes
    assert (root / "run_manifest.json").read_bytes() == run_identity_bytes
    repaired = {row["scenarioId"]: row for row in read_jsonl(root / "scenario_contracts.jsonl")}
    assert repaired[rejected_id]["premise"].endswith("version regenerated.")
    assert repaired[admitted_id] == originals[admitted_id]
    transaction = next((root / ".scenario_scrutiny" / "quarantine").iterdir())
    assert (transaction / "original" / f"{rejected_id}.json").read_bytes() == rejected_bytes
    assert sorted(path.name for path in (transaction / "original").iterdir()) == [f"{rejected_id}.json"]


def test_unchanged_topic_is_not_rejudged_or_regenerated_during_repair(tmp_path: Path) -> None:
    root, originals = fixture_root(tmp_path, ("topic_alpha", "topic_beta"))
    rejected_id = "scenario_topic_alpha_03"
    judge = FakeJudge({"topic_alpha": {rejected_id}})
    plan_path = TargetedRepairer(originals)

    report = scrutinize_scenarios(
        root,
        judge,
        planner=FakePlanner(),
        repair=True,
        max_workers=3,
        max_repair_rounds=2,
        scenario_repairer=plan_path,
    )

    assert report["status"] == "pass"
    assert judge.calls == Counter({"topic_alpha": 2, "topic_beta": 1})
    assert [call[0] for call in plan_path.calls] == ["topic_alpha"]
    assert report["repairTransactions"][0]["replacedScenarioIds"] == [rejected_id]


def test_dry_audit_emits_typed_rejection_without_touching_stage(tmp_path: Path) -> None:
    root, _ = fixture_root(tmp_path)
    rejected_id = "scenario_topic_alpha_11"
    stage_before = (root / "scenario_contracts.jsonl").read_bytes()
    checkpoint = checkpoint_for(root, rejected_id)
    checkpoint_before = checkpoint.read_bytes()

    report = scrutinize_scenarios(
        root,
        FakeJudge({"topic_alpha": {rejected_id}}),
        dry_audit=True,
        max_workers=3,
    )

    assert report["status"] == "rejected"
    assert report["rounds"][0]["rejectedScenarioIds"] == [rejected_id]
    assert report["rounds"][0]["topicAudits"][0]["rejected"][0]["findings"][0]["code"] == "semantic_near_duplicate"
    assert (root / "scenario_contracts.jsonl").read_bytes() == stage_before
    assert checkpoint.read_bytes() == checkpoint_before
    assert not (root / ".scenario_scrutiny" / "quarantine").exists()


def test_refuses_incomplete_topic_before_any_judge_inference(tmp_path: Path) -> None:
    root, _ = fixture_root(tmp_path)
    rows = read_jsonl(root / "scenario_contracts.jsonl")[:-1]
    write_jsonl(root / "scenario_contracts.jsonl", rows)
    judge = FakeJudge({})

    with pytest.raises(ScenarioScrutinyError, match="exactly 20"):
        scrutinize_scenarios(root, judge, dry_audit=True)

    assert judge.calls == Counter()


def test_rejects_judge_result_that_does_not_partition_all_twenty_ids(tmp_path: Path) -> None:
    root, _ = fixture_root(tmp_path)

    class MalformedJudge(FakeJudge):
        def audit_topic(self, topic_card: dict, scenarios: list[dict]) -> dict:
            result = judge_result(topic_card["topicId"], [row["scenarioId"] for row in scenarios], set())
            result["accepted"].pop()
            return result

    with pytest.raises(ScenarioScrutinyError, match="classify each scenario ID exactly once"):
        scrutinize_scenarios(root, MalformedJudge({}), dry_audit=True)


def test_authentic_judge_retries_protocol_failures_for_the_same_topic(monkeypatch) -> None:
    rows = [scenario("topic_alpha", ordinal) for ordinal in range(20)]
    valid = judge_result("topic_alpha", [row["scenarioId"] for row in rows], set())
    model_result = model_judge_result("topic_alpha", [row["scenarioId"] for row in rows], set())

    class FlakyPlanner:
        calls = 0

        def __init__(self, config):
            self.config = config

        def call(self, system, prompt, response_schema):
            self.calls += 1
            if self.calls < 3:
                raise ValueError("malformed structured response")
            return model_result

    monkeypatch.setattr(
        "ground_truth_finetuning.training.scenario_scrutiny.JsonOnlyPlanner",
        FlakyPlanner,
    )
    judge = AuthenticScenarioJudge(PlannerConfig(
        endpoint="http://judge.invalid/v1/chat/completions",
        model="judge-model",
        api_key="",
        temperature=0.0,
    ), max_attempts=3)

    result = judge.audit_topic(topic("topic_alpha"), rows)
    assert result["topicId"] == valid["topicId"]
    assert result["groupDecision"] == valid["groupDecision"]
    assert [item["scenarioId"] for item in result["accepted"]] == [
        item["scenarioId"] for item in valid["accepted"]
    ]
    assert result["rejected"] == []
    assert judge._planner.calls == 3
