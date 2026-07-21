"""Inference-authentic scrutiny and targeted repair for v5 scenario contracts.

Semantic admission decisions are made only by an independent model judge.  This
module deliberately contains no lexical normalizer, regular expression, semantic
fallback, or deterministic scenario generator.  Local logic is limited to strict
schema validation, immutable identity binding, and transactional replacement of
the exact records rejected by the judge.
"""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
import json
import os
import tempfile

from jsonschema import Draft202012Validator

from ground_truth_finetuning.training.diverse_cascade import (
    CascadeError,
    JsonOnlyPlanner,
    PlannerConfig,
    _v5_response_schema,
    canonical_json,
    content_hash,
    load_json,
    load_jsonl,
    normalize_planner_endpoints,
    validate_scenario_contract,
    validate_unique_scenario_premises,
)


SCENARIOS_PER_TOPIC = 20
AUDIT_ROOT_NAME = ".scenario_scrutiny"
AUDIT_SCHEMA = "personaplex.scenario-topic-audit.v1"
REPORT_SCHEMA = "personaplex.scenario-scrutiny-report.v1"
TRANSACTION_SCHEMA = "personaplex.scenario-repair-transaction.v1"

FINDING_CODES = (
    "semantic_near_duplicate",
    "scheduling_template_collapse",
    "mode_blueprint_mismatch",
    "placeholder_or_identity_company_leakage",
    "unsafe_content",
    "target_dialogue_like_content",
    "weak_four_sibling_causal_affordance",
    "incoherent_known_facts",
    "incoherent_uncertainty",
    "incoherent_policy_constraints",
    "incoherent_outcome_space",
)

DIMENSION_KEYS = (
    "semanticDiversity",
    "modeAndTemplateDiversity",
    "identityAndCompanySafety",
    "contentAndTargetLeakageSafety",
    "fourSiblingCausalAffordance",
    "statePolicyOutcomeCoherence",
)

JUDGE_RESPONSE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "topicId",
        "groupDecision",
        "groupRationale",
        "dimensionVerdicts",
        "accepted",
        "rejected",
    ],
    "properties": {
        "topicId": {"type": "string", "minLength": 1},
        "groupDecision": {"enum": ["pass", "reject"]},
        "groupRationale": {
            "type": "string",
            "minLength": 1,
        },
        "dimensionVerdicts": {
            "type": "object",
            "additionalProperties": False,
            "required": list(DIMENSION_KEYS),
            "properties": {
                key: {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["status", "rationale"],
                    "properties": {
                        "status": {"enum": ["pass", "fail"]},
                        "rationale": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                }
                for key in DIMENSION_KEYS
            },
        },
        "accepted": {
            "type": "array",
            "minItems": 0,
            "maxItems": SCENARIOS_PER_TOPIC,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["scenarioId", "rationale"],
                "properties": {
                    "scenarioId": {"type": "string", "minLength": 1},
                    "rationale": {
                        "type": "string",
                        "minLength": 1,
                    },
                },
            },
        },
        "rejected": {
            "type": "array",
            "minItems": 0,
            "maxItems": SCENARIOS_PER_TOPIC,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["scenarioId", "findings"],
                "properties": {
                    "scenarioId": {"type": "string", "minLength": 1},
                    "findings": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": len(FINDING_CODES),
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["code", "rationale", "relatedScenarioIds"],
                            "properties": {
                                "code": {"enum": list(FINDING_CODES)},
                                "rationale": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                                "relatedScenarioIds": {
                                    "type": "array",
                                    "minItems": 0,
                                    "maxItems": SCENARIOS_PER_TOPIC,
                                    "uniqueItems": True,
                                    "items": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}


def _judge_model_response_schema(scenario_ids: list[str]) -> dict[str, Any]:
    if len(scenario_ids) != SCENARIOS_PER_TOPIC or len(set(scenario_ids)) != SCENARIOS_PER_TOPIC:
        raise ScenarioScrutinyError("Judge model schema requires exactly 20 unique scenario IDs")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "topicId", "groupDecision", "groupRationale", "dimensionVerdicts", "findings",
        ],
        "properties": {
            "topicId": {"const": ""},
            "groupDecision": deepcopy(JUDGE_RESPONSE_SCHEMA["properties"]["groupDecision"]),
            "groupRationale": deepcopy(JUDGE_RESPONSE_SCHEMA["properties"]["groupRationale"]),
            "dimensionVerdicts": deepcopy(JUDGE_RESPONSE_SCHEMA["properties"]["dimensionVerdicts"]),
            "findings": {
                "type": "array",
                "minItems": 0,
                "maxItems": SCENARIOS_PER_TOPIC,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "rationale", "scenarioIds"],
                    "properties": {
                        "code": {"enum": list(FINDING_CODES)},
                        "rationale": {
                            "type": "string",
                            "minLength": 1,
                        },
                        "scenarioIds": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": SCENARIOS_PER_TOPIC,
                            "uniqueItems": True,
                            "items": {"enum": scenario_ids},
                        },
                    },
                },
            },
        },
    }


def _normalize_model_judge_result(
    result: dict[str, Any],
    topic_id: str,
    scenario_ids: list[str],
) -> dict[str, Any]:
    schema = _judge_model_response_schema(scenario_ids)
    schema["properties"]["topicId"] = {"const": topic_id}
    errors = sorted(Draft202012Validator(schema).iter_errors(result), key=lambda error: list(error.path))
    if errors:
        raise ScenarioScrutinyError(f"Judge model response violates bound schema: {errors[0].message}")
    rejected_ids = {
        scenario_id
        for finding in result["findings"]
        for scenario_id in finding["scenarioIds"]
    }
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        if scenario_id not in rejected_ids:
            accepted.append({
                "scenarioId": scenario_id,
                "rationale": "Accepted by the independent bound scenario decision.",
            })
        else:
            applicable = [finding for finding in result["findings"] if scenario_id in finding["scenarioIds"]]
            if not applicable:
                raise ScenarioScrutinyError("Every rejected judge decision requires a finding cluster")
            rejected.append({
                "scenarioId": scenario_id,
                "findings": [
                    {
                        "code": finding["code"],
                        "rationale": finding["rationale"],
                        "relatedScenarioIds": [
                            related for related in finding["scenarioIds"] if related != scenario_id
                        ],
                    }
                    for finding in applicable
                ],
            })
    normalized = {
        "topicId": result["topicId"],
        "groupDecision": result["groupDecision"],
        "groupRationale": result["groupRationale"],
        "dimensionVerdicts": result["dimensionVerdicts"],
        "accepted": accepted,
        "rejected": rejected,
    }
    validate_judge_result(normalized, topic_id, scenario_ids)
    return normalized

JUDGE_SYSTEM = """You are the independent admission judge for PersonaPlex v5 scenario contracts.
Reasoning mode is disabled. Return only the strict JSON-Schema response; expose one-sentence concise
decision rationales, never hidden reasoning or replacement prose. Judge the twenty records together by
meaning, not exact-string overlap. Reject only IDs that require regeneration, but reject enough
records to remove a group-level collapse. Do not rewrite, normalize, complete, or suggest scenario
text. Express semantic evidence once in compact finding clusters. Membership in any finding cluster
is the sole rejection decision; every bound scenario absent from all clusters is accepted. Scenario
contracts are planning inputs, never target dialogue."""


class ScenarioScrutinyError(RuntimeError):
    """Raised when scrutiny cannot preserve its strict contracts."""


class ScenarioJudge(Protocol):
    def binding(self) -> dict[str, Any]: ...

    def audit_topic(
        self,
        topic: dict[str, Any],
        scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class ScenarioRepairer(Protocol):
    def binding(self) -> dict[str, Any]: ...

    def repair_one(
        self,
        topic: dict[str, Any],
        original: dict[str, Any],
        admitted: list[dict[str, Any]],
        rejected_context: list[dict[str, Any]],
        judge_rejection: dict[str, Any],
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ScrutinyPaths:
    root: Path
    request: Path
    run_manifest: Path
    topics: Path
    scenarios: Path
    scenario_checkpoints: Path
    audit_root: Path

    @classmethod
    def from_root(cls, root: Path, request: Path | None = None) -> "ScrutinyPaths":
        resolved = root.resolve()
        return cls(
            root=resolved,
            request=(request.resolve() if request is not None else resolved / "request.json"),
            run_manifest=resolved / "run_manifest.json",
            topics=resolved / "topic_cards.jsonl",
            scenarios=resolved / "scenario_contracts.jsonl",
            scenario_checkpoints=resolved / ".stage_checkpoints" / "scenarios",
            audit_root=resolved / AUDIT_ROOT_NAME,
        )


class AuthenticScenarioJudge:
    """Independent OpenAI-compatible judge with strict schema output."""

    def __init__(self, config: PlannerConfig, *, max_attempts: int = 6):
        if config.temperature != 0.0:
            raise ScenarioScrutinyError("Scenario judge temperature must be exactly 0.0")
        if max_attempts < 1:
            raise ScenarioScrutinyError("Scenario judge max_attempts must be positive")
        self._planner = JsonOnlyPlanner(config)
        self._max_attempts = max_attempts
        self._binding = {
            "protocol": "openai_chat_completions",
            "endpoints": list(normalize_planner_endpoints(config.endpoint)),
            "model": config.model,
            "reasoning": {"enabled": False},
            "temperature": 0.0,
            "maxTokens": config.max_tokens,
            "maxAttempts": max_attempts,
            "responseFormat": "strict_json_schema",
            "responseSchemaHash": content_hash(
                _judge_model_response_schema([f"bound-slot-{index:02d}" for index in range(20)])
            ),
            "responseSchemaVersion": "personaplex.scenario-topic-judge-clustered-findings.v4",
        }

    def binding(self) -> dict[str, Any]:
        return json.loads(canonical_json(self._binding))

    def audit_topic(
        self,
        topic: dict[str, Any],
        scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = {
            "task": "Audit one complete v5 topic group for training admission.",
            "topicCard": topic,
            "scenarioContracts": [_scenario_audit_view(row) for row in scenarios],
            "groupCardinality": SCENARIOS_PER_TOPIC,
            "criteria": {
                "semanticNearDuplicates": (
                    "Detect paraphrased or structurally equivalent premises and trajectories, "
                    "not merely exact strings."
                ),
                "modeCollapse": (
                    "Reject narrow scheduling, appointment, support-ticket, or reusable-template "
                    "collapse when the topic supports materially broader interactions."
                ),
                "identitySafety": (
                    "Reject unresolved placeholders, fabricated identity/contact/company slots, "
                    "impersonation assumptions, or company-script leakage."
                ),
                "contentSafety": (
                    "Reject unsafe planning content and any premise/fact/policy/outcome that reads "
                    "like target agent dialogue or dictates the spoken answer."
                ),
                "causalAffordance": (
                    "Each scenario must support a shared-prefix four-sibling intervention where "
                    "verified-positive, verified-negative, uncertain, and superseded control states "
                    "can license materially different next behavior without embedding target speech."
                ),
                "coherence": (
                    "Known facts, uncertainty, policy constraints, disallowed claims, allowed tools, "
                    "interaction opportunities, and outcome space must be mutually coherent."
                ),
            },
            "decisionRules": [
                "A pass requires zero rejected IDs and every dimension verdict passing.",
                "A reject requires at least one rejected ID and at least one failed dimension.",
                "Finding-cluster membership is rejection; every supplied ID absent from all findings is accepted.",
                "Group semantically related rejected IDs into one finding instead of repeating prose per ID.",
                "Do not provide repaired scenario text.",
            ],
        }
        scenario_ids = _scenario_ids(scenarios)
        response_schema = _judge_model_response_schema(scenario_ids)
        response_schema["properties"]["topicId"] = {"const": topic["topicId"]}
        failures: list[str] = []
        for attempt in range(1, self._max_attempts + 1):
            try:
                result = self._planner.call(
                    JUDGE_SYSTEM,
                    canonical_json(prompt),
                    response_schema,
                )
                return _normalize_model_judge_result(result, topic["topicId"], scenario_ids)
            except Exception as error:
                failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
        raise ScenarioScrutinyError(
            f"Scenario judge exhausted {self._max_attempts} protocol attempts for "
            f"{topic['topicId']}: {' | '.join(failures)}"
        )


def _scenario_diversity_view(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(scenario[key])
        for key in (
            "scenarioId",
            "mode",
            "premise",
            "interactionOpportunity",
            "requiredControlPhenomena",
            "scenarioOutcomeSpace",
        )
    }


def _scenario_audit_view(scenario: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(scenario[key])
        for key in (
            "scenarioId",
            "topicId",
            "mode",
            "premise",
            "startingState",
            "interactionOpportunity",
            "allowedToolClasses",
            "disallowedClaims",
            "scenarioOutcomeSpace",
            "requiredControlPhenomena",
        )
    }


class AuthenticScenarioRepairer:
    """Regenerate one rejected ID with full judge and peer-diversity context."""

    def __init__(self, planner: JsonOnlyPlanner, *, max_attempts: int = 6):
        if max_attempts < 1:
            raise ScenarioScrutinyError("Scenario repair max_attempts must be positive")
        self._planner = planner
        self._max_attempts = max_attempts
        self._binding = {
            "protocol": "openai_chat_completions",
            "endpoints": list(planner.endpoints),
            "model": planner.config.model,
            "reasoning": {"enabled": False},
            "maxAttempts": max_attempts,
            "responseFormat": "strict_json_schema",
            "repairMode": "judge_feedback_conditioned_full_rejected_cluster_v2",
        }

    def binding(self) -> dict[str, Any]:
        return json.loads(canonical_json(self._binding))

    def repair_one(
        self,
        topic: dict[str, Any],
        original: dict[str, Any],
        admitted: list[dict[str, Any]],
        rejected_context: list[dict[str, Any]],
        judge_rejection: dict[str, Any],
    ) -> dict[str, Any]:
        scenario_id = original["scenarioId"]
        topic_id = topic["topicId"]
        schema = _v5_response_schema(
            "scenarios",
            1,
            "scenarioContract",
            {"scenarioId": scenario_id, "topicId": topic_id},
        )
        prompt = {
            "task": "Regenerate exactly one rejected scenario contract without changing its identity.",
            "topicCard": topic,
            "assignedIdentity": {"scenarioId": scenario_id, "topicId": topic_id},
            "rejectedOriginal": original,
            "independentJudgeRejection": judge_rejection,
            "admittedAndAlreadyRepairedPeerViews": [
                _scenario_diversity_view(row)
                for row in sorted(admitted, key=lambda row: row["scenarioId"])
            ],
            "rejectedClusterViewsToAvoid": [
                _scenario_diversity_view(row)
                for row in sorted(rejected_context, key=lambda row: row["scenarioId"])
            ],
            "forbiddenRejectedModeLabels": sorted({row["mode"] for row in rejected_context}),
            "requirements": [
                "Return exactly the assigned scenario ID and topic ID.",
                "Resolve every independent judge finding rather than paraphrasing the rejected original.",
                "Do not reuse a mode label or control-phenomena signature from the rejected cluster.",
                "Use a materially different interaction mechanism, participant relationship, evidence transition, stakes, outcome topology, and conversational mode from related and admitted peers.",
                "Support verified-positive, verified-negative, uncertain, and superseded next-state interventions that can require materially different behavior.",
                "Keep all fields concise, natural, internally coherent, and target-free.",
                "Do not write target dialogue, desired wording, placeholders, company scripts, contact data, or deterministic greetings and sign-offs.",
            ],
        }
        failures: list[str] = []
        peers = admitted + rejected_context
        peer_premises = {row["premise"] for row in peers}
        rejected_modes = {row["mode"] for row in rejected_context}
        peer_control_signatures = {
            canonical_json(row["requiredControlPhenomena"])
            for row in peers
        }
        for attempt in range(1, self._max_attempts + 1):
            try:
                result = self._planner.call(
                    "You repair one PersonaPlex scenario contract from independent semantic audit evidence. "
                    "Reasoning is disabled. Return only strict schema JSON and never target dialogue.",
                    canonical_json(prompt),
                    schema,
                )
                rows = result.get("scenarios")
                if not isinstance(rows, list) or len(rows) != 1:
                    raise ScenarioScrutinyError("Repair response must contain exactly one scenario")
                candidate = rows[0]
                validate_scenario_contract(candidate, {topic_id})
                if candidate["scenarioId"] != scenario_id or candidate["topicId"] != topic_id:
                    raise ScenarioScrutinyError("Repair response changed its assigned identity")
                if candidate["premise"] in peer_premises:
                    raise ScenarioScrutinyError("Repair response duplicated a peer premise")
                if candidate["mode"] in rejected_modes:
                    raise ScenarioScrutinyError("Repair response reused a rejected-cluster mode")
                if canonical_json(candidate["requiredControlPhenomena"]) in peer_control_signatures:
                    raise ScenarioScrutinyError("Repair response reused a peer control signature")
                return candidate
            except Exception as error:
                failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
        raise ScenarioScrutinyError(
            f"Scenario repair exhausted {self._max_attempts} protocol attempts for "
            f"{scenario_id}: {' | '.join(failures)}"
        )


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except FileNotFoundError as error:
        raise ScenarioScrutinyError(f"Required scrutiny input is missing: {path}") from error


def _scenario_ids(scenarios: list[dict[str, Any]]) -> list[str]:
    return [scenario["scenarioId"] for scenario in scenarios]


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _write_immutable_json(path: Path, value: dict[str, Any]) -> None:
    data = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != data:
            raise ScenarioScrutinyError(f"Immutable artifact conflicts with existing content: {path}")
        return
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _load_required_json(path: Path) -> dict[str, Any]:
    try:
        value = load_json(path)
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ScenarioScrutinyError(f"Cannot load required JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ScenarioScrutinyError(f"Required JSON artifact must be an object: {path}")
    return value


def validate_judge_result(result: dict[str, Any], topic_id: str, scenario_ids: list[str]) -> None:
    errors = sorted(Draft202012Validator(JUDGE_RESPONSE_SCHEMA).iter_errors(result), key=lambda error: list(error.path))
    if errors:
        raise ScenarioScrutinyError(f"Judge response violates strict schema: {errors[0].message}")
    if result["topicId"] != topic_id:
        raise ScenarioScrutinyError("Judge response topicId does not match the audited topic")
    accepted_ids = [item["scenarioId"] for item in result["accepted"]]
    rejected_ids = [item["scenarioId"] for item in result["rejected"]]
    classified = accepted_ids + rejected_ids
    expected = set(scenario_ids)
    if len(scenario_ids) != SCENARIOS_PER_TOPIC or len(expected) != SCENARIOS_PER_TOPIC:
        raise ScenarioScrutinyError("Judge input must contain exactly 20 unique scenario IDs")
    if len(classified) != SCENARIOS_PER_TOPIC or len(set(classified)) != SCENARIOS_PER_TOPIC:
        raise ScenarioScrutinyError("Judge must classify each scenario ID exactly once")
    if set(classified) != expected:
        raise ScenarioScrutinyError("Judge accepted/rejected IDs do not exactly match the audited group")
    for rejected in result["rejected"]:
        for finding in rejected["findings"]:
            if not set(finding["relatedScenarioIds"]).issubset(expected):
                raise ScenarioScrutinyError("Judge finding references a scenario outside the audited group")
    failed_dimensions = [
        key for key, verdict in result["dimensionVerdicts"].items()
        if verdict["status"] == "fail"
    ]
    if result["groupDecision"] == "pass":
        if rejected_ids or failed_dimensions:
            raise ScenarioScrutinyError("A passing judge decision cannot contain rejections or failed dimensions")
    elif not rejected_ids or not failed_dimensions:
        raise ScenarioScrutinyError("A rejecting judge decision requires rejected IDs and a failed dimension")


def _validate_completed_stage(
    paths: ScrutinyPaths,
    request: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    if request.get("schema") != "personaplex.diverse-corpus-request.v2":
        raise ScenarioScrutinyError("Scenario scrutiny requires a v2 corpus request")
    if request.get("strategyVersion") != "semantic-control-v5":
        raise ScenarioScrutinyError("Scenario scrutiny requires semantic-control-v5")
    coverage = request.get("coverageTarget")
    if not isinstance(coverage, dict) or coverage.get("scenariosPerTopic") != SCENARIOS_PER_TOPIC:
        raise ScenarioScrutinyError("The bound request must require exactly 20 scenarios per topic")
    topics = load_jsonl(paths.topics)
    scenarios = load_jsonl(paths.scenarios)
    if not topics or not scenarios:
        raise ScenarioScrutinyError("Completed topic_cards.jsonl and scenario_contracts.jsonl are required")
    topic_by_id: dict[str, dict[str, Any]] = {}
    for topic in topics:
        if topic.get("schema") != "personaplex.topic-card.v2":
            raise ScenarioScrutinyError("Every audited topic card must use personaplex.topic-card.v2")
        topic_id = topic.get("topicId")
        if not isinstance(topic_id, str) or not topic_id or topic_id in topic_by_id:
            raise ScenarioScrutinyError("Topic cards must have unique nonempty topicId values")
        topic_by_id[topic_id] = topic
    grouped: dict[str, list[dict[str, Any]]] = {topic_id: [] for topic_id in topic_by_id}
    seen_scenarios: set[str] = set()
    for scenario in scenarios:
        try:
            validate_scenario_contract(scenario, set(topic_by_id))
        except CascadeError as error:
            raise ScenarioScrutinyError(f"Invalid v5 scenario contract: {error}") from error
        scenario_id = scenario["scenarioId"]
        if scenario_id in seen_scenarios:
            raise ScenarioScrutinyError(f"Duplicate scenarioId in completed stage: {scenario_id}")
        seen_scenarios.add(scenario_id)
        grouped[scenario["topicId"]].append(scenario)
    for topic_id, rows in grouped.items():
        if len(rows) != SCENARIOS_PER_TOPIC:
            raise ScenarioScrutinyError(
                f"Topic {topic_id} has {len(rows)} scenarios; exactly {SCENARIOS_PER_TOPIC} are required"
            )
        rows.sort(key=lambda row: row["scenarioId"])
    try:
        validate_unique_scenario_premises(scenarios)
    except CascadeError as error:
        raise ScenarioScrutinyError(f"Completed scenario stage is structurally non-unique: {error}") from error
    return topics, topic_by_id, grouped


class AuditStore:
    def __init__(
        self,
        paths: ScrutinyPaths,
        run_identity_hash: str,
        request_hash: str,
        topic_cards_hash: str,
        judge_binding: dict[str, Any],
        resume: bool,
    ) -> None:
        self.paths = paths
        self.run_identity_hash = run_identity_hash
        self.request_hash = request_hash
        self.topic_cards_hash = topic_cards_hash
        self.judge_binding = judge_binding
        self.judge_config_hash = content_hash(judge_binding)
        self.resume = resume
        self._memory: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def _key(self, topic: dict[str, Any], scenarios: list[dict[str, Any]]) -> str:
        return content_hash({
            "runIdentityHash": self.run_identity_hash,
            "requestHash": self.request_hash,
            "topicCard": topic,
            "scenarioContracts": scenarios,
            "judgeConfigHash": self.judge_config_hash,
        })

    def audit(
        self,
        judge: ScenarioJudge,
        topic: dict[str, Any],
        scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]:
        group_hash = content_hash(scenarios)
        key = self._key(topic, scenarios)
        with self._lock:
            cached = self._memory.get(key)
        if cached is not None:
            return cached
        path = self.paths.audit_root / "audits" / topic["topicId"] / f"{key[7:]}.json"
        if path.exists():
            if not self.resume:
                raise ScenarioScrutinyError(f"Audit checkpoint exists; use --resume: {path}")
            checkpoint = _load_required_json(path)
            self._validate_checkpoint(checkpoint, topic, scenarios, group_hash, key)
        else:
            result = judge.audit_topic(topic, scenarios)
            validate_judge_result(result, topic["topicId"], _scenario_ids(scenarios))
            body = {
                "schema": AUDIT_SCHEMA,
                "auditKey": key,
                "runIdentityHash": self.run_identity_hash,
                "requestHash": self.request_hash,
                "topicCardsHash": self.topic_cards_hash,
                "topicId": topic["topicId"],
                "topicCardHash": content_hash(topic),
                "scenarioGroupHash": group_hash,
                "scenarioIds": _scenario_ids(scenarios),
                "judgeConfigHash": self.judge_config_hash,
                "judge": self.judge_binding,
                "decision": result,
            }
            checkpoint = dict(body)
            checkpoint["auditId"] = content_hash(body)
            _write_immutable_json(path, checkpoint)
        with self._lock:
            prior = self._memory.setdefault(key, checkpoint)
            if prior != checkpoint:
                raise ScenarioScrutinyError("Concurrent audit checkpoint content diverged")
        return checkpoint

    def _validate_checkpoint(
        self,
        checkpoint: dict[str, Any],
        topic: dict[str, Any],
        scenarios: list[dict[str, Any]],
        group_hash: str,
        key: str,
    ) -> None:
        required = {
            "schema", "auditKey", "runIdentityHash", "requestHash", "topicCardsHash",
            "topicId", "topicCardHash", "scenarioGroupHash", "scenarioIds",
            "judgeConfigHash", "judge", "decision", "auditId",
        }
        if set(checkpoint) != required:
            raise ScenarioScrutinyError("Immutable audit checkpoint has an invalid field set")
        body = dict(checkpoint)
        audit_id = body.pop("auditId")
        expected = {
            "schema": AUDIT_SCHEMA,
            "auditKey": key,
            "runIdentityHash": self.run_identity_hash,
            "requestHash": self.request_hash,
            "topicCardsHash": self.topic_cards_hash,
            "topicId": topic["topicId"],
            "topicCardHash": content_hash(topic),
            "scenarioGroupHash": group_hash,
            "scenarioIds": _scenario_ids(scenarios),
            "judgeConfigHash": self.judge_config_hash,
            "judge": self.judge_binding,
        }
        for field, value in expected.items():
            if body.get(field) != value:
                raise ScenarioScrutinyError(f"Immutable audit checkpoint binding mismatch: {field}")
        if audit_id != content_hash(body):
            raise ScenarioScrutinyError("Immutable audit checkpoint hash is invalid")
        validate_judge_result(body["decision"], topic["topicId"], _scenario_ids(scenarios))


def _parallel_topic_audits(
    store: AuditStore,
    judge: ScenarioJudge,
    topic_by_id: dict[str, dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    max_workers: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {
        executor.submit(store.audit, judge, topic_by_id[topic_id], grouped[topic_id]): topic_id
        for topic_id in sorted(topic_by_id)
    }
    try:
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except BaseException as error:
                for pending in futures:
                    pending.cancel()
                raise ScenarioScrutinyError(
                    f"Scenario judge failed for topic {futures[future]}: {error}"
                ) from error
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return sorted(results, key=lambda checkpoint: checkpoint["topicId"])


def _checkpoint_files_by_id(paths: ScrutinyPaths) -> dict[str, Path]:
    if not paths.scenario_checkpoints.is_dir():
        raise ScenarioScrutinyError(
            f"Scenario checkpoint directory is required for targeted repair: {paths.scenario_checkpoints}"
        )
    indexed: dict[str, Path] = {}
    for path in sorted(paths.scenario_checkpoints.glob("*.json")):
        record = _load_required_json(path)
        scenario_id = record.get("scenarioId")
        if not isinstance(scenario_id, str) or not scenario_id:
            raise ScenarioScrutinyError(f"Scenario checkpoint lacks scenarioId: {path}")
        if scenario_id in indexed:
            raise ScenarioScrutinyError(f"Multiple checkpoint files bind scenarioId {scenario_id}")
        indexed[scenario_id] = path
    return indexed


class RepairCandidateStore:
    def __init__(
        self,
        paths: ScrutinyPaths,
        run_identity_hash: str,
        request_hash: str,
        repairer: ScenarioRepairer,
        resume: bool,
    ) -> None:
        self.paths = paths
        self.run_identity_hash = run_identity_hash
        self.request_hash = request_hash
        self.repairer = repairer
        self.repairer_binding = repairer.binding()
        self.repairer_config_hash = content_hash(self.repairer_binding)
        self.resume = resume

    def repair(
        self,
        topic: dict[str, Any],
        original: dict[str, Any],
        admitted: list[dict[str, Any]],
        rejected_context: list[dict[str, Any]],
        judge_rejection: dict[str, Any],
    ) -> dict[str, Any]:
        admitted = sorted(admitted, key=lambda row: row["scenarioId"])
        binding = {
            "runIdentityHash": self.run_identity_hash,
            "requestHash": self.request_hash,
            "topicId": topic["topicId"],
            "topicCardHash": content_hash(topic),
            "scenarioId": original["scenarioId"],
            "rejectedOriginalHash": content_hash(original),
            "admittedPeerContextHash": content_hash(admitted),
            "rejectedClusterContextHash": content_hash(rejected_context),
            "judgeRejection": judge_rejection,
            "repairerConfigHash": self.repairer_config_hash,
        }
        repair_key = content_hash(binding)
        path = (
            self.paths.audit_root
            / "repair_candidates"
            / repair_key[7:9]
            / f"{repair_key[7:]}.json"
        )
        if path.exists():
            if not self.resume:
                raise ScenarioScrutinyError(f"Repair checkpoint exists; use --resume: {path}")
            checkpoint = _load_required_json(path)
            expected_fields = {
                "schema", "repairKey", "binding", "repairer", "candidate", "candidateId",
            }
            if set(checkpoint) != expected_fields:
                raise ScenarioScrutinyError("Immutable repair checkpoint has an invalid field set")
            if checkpoint["schema"] != "personaplex.scenario-repair-candidate.v1":
                raise ScenarioScrutinyError("Immutable repair checkpoint has an invalid schema")
            if checkpoint["repairKey"] != repair_key or checkpoint["binding"] != binding:
                raise ScenarioScrutinyError("Immutable repair checkpoint binding is stale")
            if checkpoint["repairer"] != self.repairer_binding:
                raise ScenarioScrutinyError("Immutable repair checkpoint repairer binding is stale")
            candidate = checkpoint["candidate"]
            candidate_body = dict(checkpoint)
            candidate_id = candidate_body.pop("candidateId")
            if candidate_id != content_hash(candidate_body):
                raise ScenarioScrutinyError("Immutable repair checkpoint hash is invalid")
        else:
            candidate = self.repairer.repair_one(
                topic,
                original,
                admitted,
                rejected_context,
                judge_rejection,
            )
            body = {
                "schema": "personaplex.scenario-repair-candidate.v1",
                "repairKey": repair_key,
                "binding": binding,
                "repairer": self.repairer_binding,
                "candidate": candidate,
            }
            checkpoint = dict(body)
            checkpoint["candidateId"] = content_hash(body)
            _write_immutable_json(path, checkpoint)
        validate_scenario_contract(candidate, {topic["topicId"]})
        if candidate["scenarioId"] != original["scenarioId"]:
            raise ScenarioScrutinyError("Repair checkpoint changed its assigned scenarioId")
        if candidate["topicId"] != topic["topicId"]:
            raise ScenarioScrutinyError("Repair checkpoint changed its assigned topicId")
        if candidate["premise"] in {row["premise"] for row in admitted + rejected_context}:
            raise ScenarioScrutinyError("Repair checkpoint duplicates a peer premise")
        if candidate["mode"] in {row["mode"] for row in rejected_context}:
            raise ScenarioScrutinyError("Repair checkpoint reuses a rejected-cluster mode")
        if canonical_json(candidate["requiredControlPhenomena"]) in {
            canonical_json(row["requiredControlPhenomena"])
            for row in admitted + rejected_context
        }:
            raise ScenarioScrutinyError("Repair checkpoint reuses a peer control signature")
        return candidate


def _regenerate_rejected_topics(
    repair_store: RepairCandidateStore,
    topic_by_id: dict[str, dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    rejected_by_topic: dict[str, set[str]],
    rejection_bindings: dict[str, dict[str, Any]],
    max_workers: int,
) -> dict[str, dict[str, Any]]:
    def regenerate(topic_id: str) -> dict[str, dict[str, Any]]:
        rejected_ids = rejected_by_topic[topic_id]
        original_by_id = {row["scenarioId"]: row for row in grouped[topic_id]}
        rejected_context = [original_by_id[scenario_id] for scenario_id in sorted(rejected_ids)]
        admitted = [
            scenario for scenario in grouped[topic_id]
            if scenario["scenarioId"] not in rejected_ids
        ]
        generated_by_id: dict[str, dict[str, Any]] = {}
        for scenario_id in sorted(rejected_ids):
            candidate = repair_store.repair(
                topic_by_id[topic_id],
                original_by_id[scenario_id],
                admitted,
                rejected_context,
                rejection_bindings[scenario_id],
            )
            generated_by_id[scenario_id] = candidate
            admitted.append(candidate)
        if set(generated_by_id) != rejected_ids:
            raise ScenarioScrutinyError("Authentic repair did not return the exact rejected ID set")
        return generated_by_id

    replacements: dict[str, dict[str, Any]] = {}
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {
        executor.submit(regenerate, topic_id): topic_id
        for topic_id in sorted(rejected_by_topic)
    }
    try:
        for future in as_completed(futures):
            try:
                batch = future.result()
            except BaseException as error:
                for pending in futures:
                    pending.cancel()
                raise ScenarioScrutinyError(
                    f"Targeted scenario regeneration failed for topic {futures[future]}: {error}"
                ) from error
            overlap = set(replacements).intersection(batch)
            if overlap:
                raise ScenarioScrutinyError(f"Targeted regeneration duplicated IDs: {sorted(overlap)}")
            replacements.update(batch)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    expected_ids = set().union(*rejected_by_topic.values())
    if set(replacements) != expected_ids:
        raise ScenarioScrutinyError("Targeted regeneration did not return the exact rejected ID set")
    return replacements


def _prepare_and_commit_transaction(
    paths: ScrutinyPaths,
    run_identity_hash: str,
    current_rows: list[dict[str, Any]],
    replacements: dict[str, dict[str, Any]],
    rejection_bindings: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checkpoint_files = _checkpoint_files_by_id(paths)
    if not set(replacements).issubset(checkpoint_files):
        missing = sorted(set(replacements).difference(checkpoint_files))
        raise ScenarioScrutinyError(f"Rejected scenario checkpoints are missing: {missing}")
    current_by_id = {row["scenarioId"]: row for row in current_rows}
    if not set(replacements).issubset(current_by_id):
        raise ScenarioScrutinyError("Rejected IDs are absent from scenario_contracts.jsonl")
    final_by_id = dict(current_by_id)
    final_by_id.update(replacements)
    final_rows = [final_by_id[scenario_id] for scenario_id in sorted(final_by_id)]
    validate_unique_scenario_premises(final_rows)

    records: list[dict[str, Any]] = []
    original_bytes: dict[str, bytes] = {}
    regenerated_bytes: dict[str, bytes] = {}
    for scenario_id in sorted(replacements):
        checkpoint_path = checkpoint_files[scenario_id]
        relative = checkpoint_path.relative_to(paths.root).as_posix()
        original = checkpoint_path.read_bytes()
        regenerated = _json_bytes(replacements[scenario_id])
        original_bytes[scenario_id] = original
        regenerated_bytes[scenario_id] = regenerated
        records.append({
            "scenarioId": scenario_id,
            "topicId": replacements[scenario_id]["topicId"],
            "checkpointRelativePath": relative,
            "originalSha256": _sha256_bytes(original),
            "regeneratedSha256": _sha256_bytes(regenerated),
            "judgeRejection": rejection_bindings[scenario_id],
        })
    final_stage_bytes = _jsonl_bytes(final_rows)
    body = {
        "schema": TRANSACTION_SCHEMA,
        "runIdentityHash": run_identity_hash,
        "sourceScenarioContractsHash": _sha256_file(paths.scenarios),
        "finalScenarioContractsHash": _sha256_bytes(final_stage_bytes),
        "records": records,
    }
    transaction = dict(body)
    transaction["transactionId"] = content_hash(body)
    transaction_id = transaction["transactionId"]
    quarantine_root = paths.audit_root / "quarantine"
    final_directory = quarantine_root / transaction_id[7:]
    if final_directory.exists():
        raise ScenarioScrutinyError(f"Repair transaction already exists: {final_directory}")
    quarantine_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepared-", dir=quarantine_root))
    try:
        for scenario_id in sorted(replacements):
            _atomic_write_bytes(temporary / "original" / f"{scenario_id}.json", original_bytes[scenario_id])
            _atomic_write_bytes(temporary / "regenerated" / f"{scenario_id}.json", regenerated_bytes[scenario_id])
        _atomic_write_bytes(temporary / "scenario_contracts.jsonl", final_stage_bytes)
        _write_immutable_json(temporary / "manifest.json", transaction)
        os.rename(temporary, final_directory)
    finally:
        if temporary.exists():
            for child in sorted(temporary.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            temporary.rmdir()

    for record in records:
        scenario_id = record["scenarioId"]
        target = paths.root / record["checkpointRelativePath"]
        _atomic_write_bytes(target, regenerated_bytes[scenario_id])
    _atomic_write_bytes(paths.scenarios, final_stage_bytes)
    commit = {
        "schema": "personaplex.scenario-repair-commit.v1",
        "transactionId": transaction_id,
        "runIdentityHash": run_identity_hash,
        "finalScenarioContractsHash": _sha256_bytes(final_stage_bytes),
        "replacedScenarioIds": sorted(replacements),
    }
    commit["commitId"] = content_hash(commit)
    _write_immutable_json(final_directory / "commit.json", commit)
    return final_rows, transaction


def _recover_pending_transactions(paths: ScrutinyPaths, run_identity_hash: str, resume: bool) -> None:
    quarantine_root = paths.audit_root / "quarantine"
    if not quarantine_root.exists():
        return
    for directory in sorted(path for path in quarantine_root.iterdir() if path.is_dir()):
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists() or (directory / "commit.json").exists():
            continue
        if not resume:
            raise ScenarioScrutinyError(
                f"Prepared repair transaction requires --resume: {directory.name}"
            )
        manifest = _load_required_json(manifest_path)
        if manifest.get("schema") != TRANSACTION_SCHEMA or manifest.get("runIdentityHash") != run_identity_hash:
            raise ScenarioScrutinyError("Pending repair transaction is not bound to the active run")
        for record in manifest.get("records", []):
            relative = record.get("checkpointRelativePath")
            scenario_id = record.get("scenarioId")
            expected_relative = f".stage_checkpoints/scenarios/{Path(str(relative)).name}"
            if relative != expected_relative or not isinstance(scenario_id, str):
                raise ScenarioScrutinyError("Pending repair transaction contains an unsafe checkpoint path")
            regenerated = directory / "regenerated" / f"{scenario_id}.json"
            data = regenerated.read_bytes()
            if _sha256_bytes(data) != record.get("regeneratedSha256"):
                raise ScenarioScrutinyError("Pending regenerated checkpoint hash mismatch")
            _atomic_write_bytes(paths.root / relative, data)
        stage_bytes = (directory / "scenario_contracts.jsonl").read_bytes()
        if _sha256_bytes(stage_bytes) != manifest.get("finalScenarioContractsHash"):
            raise ScenarioScrutinyError("Pending scenario stage hash mismatch")
        _atomic_write_bytes(paths.scenarios, stage_bytes)
        commit = {
            "schema": "personaplex.scenario-repair-commit.v1",
            "transactionId": manifest["transactionId"],
            "runIdentityHash": run_identity_hash,
            "finalScenarioContractsHash": manifest["finalScenarioContractsHash"],
            "replacedScenarioIds": sorted(record["scenarioId"] for record in manifest["records"]),
        }
        commit["commitId"] = content_hash(commit)
        _write_immutable_json(directory / "commit.json", commit)


def _write_report(paths: ScrutinyPaths, body: dict[str, Any]) -> dict[str, Any]:
    report = dict(body)
    report["reportId"] = content_hash(body)
    relative = Path("reports") / f"{report['reportId'][7:]}.json"
    path = paths.audit_root / relative
    _write_immutable_json(path, report)
    _atomic_write_bytes(
        paths.audit_root / "latest_report.json",
        _json_bytes({
            "schema": "personaplex.scenario-scrutiny-report-pointer.v1",
            "reportId": report["reportId"],
            "path": relative.as_posix(),
        }),
    )
    result = dict(report)
    result["reportPath"] = path.as_posix()
    return result


def scrutinize_scenarios(
    root: Path,
    judge: ScenarioJudge,
    *,
    request_path: Path | None = None,
    planner: JsonOnlyPlanner | None = None,
    repair: bool = False,
    dry_audit: bool = False,
    resume: bool = False,
    max_workers: int = 3,
    max_repair_rounds: int = 3,
    repair_max_attempts: int = 6,
    scenario_repairer: ScenarioRepairer | None = None,
) -> dict[str, Any]:
    if max_workers < 1 or max_workers > 3:
        raise ScenarioScrutinyError("max_workers must be in [1, 3]")
    if max_repair_rounds < 1:
        raise ScenarioScrutinyError("max_repair_rounds must be positive")
    if repair_max_attempts < 1:
        raise ScenarioScrutinyError("repair_max_attempts must be positive")
    if repair and dry_audit:
        raise ScenarioScrutinyError("repair and dry_audit are mutually exclusive")
    if repair and planner is None and scenario_repairer is None:
        raise ScenarioScrutinyError("Targeted repair requires an authentic scenario planner")

    paths = ScrutinyPaths.from_root(root, request_path)
    request = _load_required_json(paths.request)
    _load_required_json(paths.run_manifest)
    run_identity_hash = _sha256_file(paths.run_manifest)
    request_hash = _sha256_file(paths.request)
    topic_cards_hash = _sha256_file(paths.topics)
    initial_scenarios_hash = _sha256_file(paths.scenarios)
    identity_snapshot = paths.run_manifest.read_bytes()
    _recover_pending_transactions(paths, run_identity_hash, resume)
    topics, topic_by_id, grouped = _validate_completed_stage(paths, request)

    judge_binding = judge.binding()
    if not isinstance(judge_binding, dict) or judge_binding.get("reasoning") != {"enabled": False}:
        raise ScenarioScrutinyError("Judge binding must explicitly disable reasoning")
    store = AuditStore(
        paths,
        run_identity_hash,
        request_hash,
        topic_cards_hash,
        judge_binding,
        resume,
    )
    active_repairer = scenario_repairer
    if repair and active_repairer is None:
        active_repairer = AuthenticScenarioRepairer(
            planner,
            max_attempts=repair_max_attempts,
        )
    repair_store = (
        RepairCandidateStore(
            paths,
            run_identity_hash,
            request_hash,
            active_repairer,
            resume,
        )
        if active_repairer is not None else None
    )
    rounds: list[dict[str, Any]] = []
    transactions: list[dict[str, Any]] = []
    current_rows = [row for topic_id in sorted(grouped) for row in grouped[topic_id]]
    status = "rejected"

    for round_index in range(max_repair_rounds + 1):
        audits = _parallel_topic_audits(store, judge, topic_by_id, grouped, max_workers)
        rejected_by_topic: dict[str, set[str]] = {}
        rejection_bindings: dict[str, dict[str, Any]] = {}
        for audit in audits:
            rejected = audit["decision"]["rejected"]
            if rejected:
                rejected_by_topic[audit["topicId"]] = {item["scenarioId"] for item in rejected}
                for decision in rejected:
                    rejection_bindings[decision["scenarioId"]] = {
                        "auditId": audit["auditId"],
                        "findings": decision["findings"],
                    }
        rounds.append({
            "round": round_index,
            "scenarioContractsHash": _sha256_file(paths.scenarios),
            "topicAudits": [
                {
                    "topicId": audit["topicId"],
                    "auditId": audit["auditId"],
                    "decision": audit["decision"]["groupDecision"],
                    "acceptedScenarioIds": [item["scenarioId"] for item in audit["decision"]["accepted"]],
                    "rejected": audit["decision"]["rejected"],
                }
                for audit in audits
            ],
            "rejectedScenarioIds": sorted(rejection_bindings),
        })
        if not rejected_by_topic:
            status = "pass"
            break
        if not repair or dry_audit:
            status = "rejected"
            break
        if round_index >= max_repair_rounds:
            status = "repair_failed"
            break

        if repair_store is None:
            raise ScenarioScrutinyError("Targeted repair store is unavailable")
        replacements = _regenerate_rejected_topics(
            repair_store,
            topic_by_id,
            grouped,
            rejected_by_topic,
            rejection_bindings,
            max_workers,
        )
        candidate_by_id = {row["scenarioId"]: row for row in current_rows}
        candidate_by_id.update(replacements)
        candidate_rows = [candidate_by_id[scenario_id] for scenario_id in sorted(candidate_by_id)]
        validate_unique_scenario_premises(candidate_rows)
        current_rows, transaction = _prepare_and_commit_transaction(
            paths,
            run_identity_hash,
            current_rows,
            replacements,
            rejection_bindings,
        )
        transactions.append({
            "transactionId": transaction["transactionId"],
            "sourceScenarioContractsHash": transaction["sourceScenarioContractsHash"],
            "finalScenarioContractsHash": transaction["finalScenarioContractsHash"],
            "replacedScenarioIds": sorted(replacements),
        })
        _, topic_by_id, grouped = _validate_completed_stage(paths, request)

    if paths.run_manifest.read_bytes() != identity_snapshot:
        raise ScenarioScrutinyError("Run identity changed during scenario scrutiny")
    final_scenarios_hash = _sha256_file(paths.scenarios)
    final_ids = sorted(row["scenarioId"] for rows in grouped.values() for row in rows)
    report_body = {
        "schema": REPORT_SCHEMA,
        "status": status,
        "mode": "repair" if repair else "dry_audit",
        "runIdentityHash": run_identity_hash,
        "requestHash": request_hash,
        "topicCardsHash": topic_cards_hash,
        "initialScenarioContractsHash": initial_scenarios_hash,
        "finalScenarioContractsHash": final_scenarios_hash,
        "judgeConfigHash": store.judge_config_hash,
        "plannerConfigHash": (
            content_hash({
                "endpoints": list(planner.endpoints),
                "model": planner.config.model,
                "reasoning": {"enabled": False},
            })
            if planner is not None else None
        ),
        "repairerConfigHash": (
            repair_store.repairer_config_hash if repair_store is not None else None
        ),
        "topicCount": len(topics),
        "scenarioCount": len(final_ids),
        "finalScenarioIds": final_ids,
        "rounds": rounds,
        "repairTransactions": transactions,
    }
    return _write_report(paths, report_body)
