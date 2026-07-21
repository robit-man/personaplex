"""Efficient, resumable semantic-control-v5 trajectory fan-out.

Stage A asks the model for ten compact semantic leaves per scenario in one
JSON-Schema-constrained call. Stage B projects those immutable leaves through
the existing typed balance selector. Stage C expands only primary and reserve
selections into the full trajectory-seed.v2 contract. Invalid model output is
never rewritten: only the missing leaf or selected item is requested again.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
from typing import Any, Mapping, Protocol, Sequence
import urllib.error
import urllib.request

from jsonschema import Draft202012Validator

from ground_truth_finetuning.training.diverse_cascade import (
    CascadeError,
    assert_no_target_leak,
    canonical_json,
    content_hash,
    request_selection_counts,
    select_trajectories,
    validate_request,
    validate_scenario_contract,
    validate_topic_card,
    validate_trajectory_seed,
    validate_unique_causal_signatures,
)


CANDIDATE_SCHEMA = "personaplex.compact-trajectory-candidate.v5"
CANDIDATE_MANIFEST_SCHEMA = "personaplex.compact-trajectory-candidate-manifest.v1"
EXPANSION_CHECKPOINT_SCHEMA = "personaplex.trajectory-expansion-checkpoint.v1"
FANOUT_MANIFEST_SCHEMA = "personaplex.efficient-v5-fanout-manifest.v1"
LEAVES_PER_SCENARIO = 10
NEGATIVE_CONTROLS = ["paired_wrong_branch", "stale_revision", "null_control"]
TERMINATION_CONTRACT = {
    "decisionSource": "model",
    "action": "end_call_tool",
    "deterministicPhrase": False,
}
CONTROL_SOURCES = (
    "asr_fact",
    "tool_result",
    "policy_decision",
    "caller_posture_change",
    "interruption",
    "handoff",
    "scenario_state",
    "state_reducer",
)
EVENT_TYPES = (
    "completed_turn",
    "barge_in",
    "cancelled_generation",
    "repair_after_barge_in",
    "recovery",
    "brief_overlap",
    "backchannel",
)
FORBIDDEN_PLANNING_KEYS = {
    "agenttext",
    "targettext",
    "targettranscript",
    "canonicalresponse",
    "canonical_response",
    "heardtext",
    "utterance",
    "dialogue",
    "reply",
    "script",
}
MODE_COLLAPSE_LIMITS = {
    "evidenceSource": 3,
    "outcomeRoute": 2,
    "postureTransition": 2,
    "duplexProfile": 2,
    "style": 2,
}

CANDIDATES_FILENAME = "trajectory_candidates.jsonl"
CANDIDATE_AUDIT_FILENAME = "trajectory_candidate_audit.jsonl"
CANDIDATE_MANIFEST_FILENAME = "trajectory_candidate_manifest.json"
PRIMARY_FILENAME = "primary_trajectories.jsonl"
RESERVE_FILENAME = "reserve_trajectories.jsonl"
SELECTED_FILENAME = "selected_trajectories.jsonl"
SELECTION_AUDIT_FILENAME = "trajectory_selection_audit.jsonl"
SELECTION_MANIFEST_FILENAME = "trajectory_selection_manifest.json"
TRAJECTORIES_FILENAME = "trajectory_seeds.jsonl"
EXPANSION_AUDIT_FILENAME = "trajectory_expansion_audit.jsonl"
EXPANSION_MANIFEST_FILENAME = "trajectory_expansion_manifest.json"
COMBINED_MANIFEST_FILENAME = "efficient_v5_fanout_manifest.json"


class FanoutError(CascadeError):
    pass


class LeafRejected(FanoutError):
    pass


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FanoutError(f"{path} must contain one JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise FanoutError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    return rows


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    _atomic_bytes(path, payload)


def _normalized(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(value.casefold().split())
    if isinstance(value, Mapping):
        return {str(key): _normalized(child) for key, child in sorted(value.items())}
    if isinstance(value, list):
        return [_normalized(child) for child in value]
    return value


def _assert_target_free(value: Any) -> None:
    assert_no_target_leak(value)

    def walk(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key).casefold() in FORBIDDEN_PLANNING_KEYS:
                    raise LeafRejected(f"target-dialogue field {key!r} is forbidden")
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)


def _json_scalar_schema() -> dict[str, Any]:
    return {"type": ["string", "number", "integer", "boolean", "null"]}


def _string_list_schema(min_items: int = 1, max_items: int = 12) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": min_items,
        "maxItems": max_items,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 2},
    }


def _duplex_event_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "eventType", "targetOrdinal", "offsetMs", "overlapMs",
            "cancelOutgoingAudio", "invalidateGeneration",
        ],
        "properties": {
            "eventType": {"enum": list(EVENT_TYPES)},
            "targetOrdinal": {"type": "integer", "minimum": 1, "maximum": 47},
            "offsetMs": {"type": "integer", "minimum": -2000, "maximum": 30000},
            "overlapMs": {"type": "integer", "minimum": 0, "maximum": 5000},
            "cancelOutgoingAudio": {"type": "boolean"},
            "invalidateGeneration": {"type": "boolean"},
        },
    }


def deterministic_candidate_assignments(
    request: Mapping[str, Any], topic: Mapping[str, Any], scenario: Mapping[str, Any]
) -> list[dict[str, Any]]:
    affordances = topic.get("causalAffordances")
    axes = (request.get("semanticControl") or {}).get("requiredCausalAxes")
    if not isinstance(affordances, list) or len(affordances) < 3:
        raise FanoutError(f"{topic.get('topicId')}: at least three causal affordances are required")
    if not isinstance(axes, list) or not axes:
        raise FanoutError("v5 request lacks required causal axes")
    rotation = int(
        sha256(f"{request['requestId']}|{scenario['scenarioId']}".encode()).hexdigest()[:8], 16
    )
    assignments: list[dict[str, Any]] = []
    for slot in range(LEAVES_PER_SCENARIO):
        affordance = affordances[(rotation + slot) % len(affordances)]
        axis = axes[(rotation + slot) % len(axes)]
        identity = {
            "requestId": request["requestId"],
            "scenarioId": scenario["scenarioId"],
            "slot": slot,
        }
        candidate_id = "trajectory_" + content_hash(identity)[7:39]
        assignments.append({
            "slot": slot,
            "candidateId": candidate_id,
            "trajectoryId": candidate_id,
            "causalAxis": axis,
            "interventionFamily": affordance["family"],
            "operatorId": affordance["operatorId"],
            "changedPath": affordance["changedPath"],
        })
    return assignments


def compact_response_schema(
    assignments: Sequence[Mapping[str, Any]], allowed_sources: Sequence[str]
) -> dict[str, Any]:
    item_properties = {
        "premise": {"type": "string", "minLength": 40},
        "interactionArc": _string_list_schema(3, 9),
        "postureArc": _string_list_schema(2, 7),
        "postureTransition": {
            "type": "object", "additionalProperties": False,
            "required": ["from", "to"],
            "properties": {
                "from": {"type": "string", "minLength": 2},
                "to": {"type": "string", "minLength": 2},
            },
        },
        "evidenceSource": {"enum": list(allowed_sources)},
        "evidenceEvolution": _string_list_schema(2, 8),
        "duplexEvents": {
            "type": "array", "minItems": 3, "maxItems": 12,
            "items": _duplex_event_schema(),
        },
        "outcomeRoute": {"type": "string", "minLength": 3},
        "conversationLength": {
            "type": "object", "additionalProperties": False,
            "required": ["targetTurns", "min", "max"],
            "properties": {
                "targetTurns": {"type": "integer", "minimum": 6, "maximum": 18},
                "min": {"type": "integer", "minimum": 4, "maximum": 18},
                "max": {"type": "integer", "minimum": 5, "maximum": 24},
            },
        },
        "style": {
            "type": "object", "additionalProperties": False,
            "required": ["pace", "openingStyle", "closingStyle", "register", "turnCadence"],
            "properties": {
                "pace": {"type": "string", "minLength": 3},
                "openingStyle": {"type": "string", "minLength": 3},
                "closingStyle": {"type": "string", "minLength": 3},
                "register": {"type": "string", "minLength": 3},
                "turnCadence": {"type": "string", "minLength": 3},
            },
        },
        "pivot": {
            "type": "object", "additionalProperties": False,
            "required": ["from", "to", "targetOrdinal"],
            "properties": {
                "from": _json_scalar_schema(),
                "to": _json_scalar_schema(),
                "targetOrdinal": {"type": "integer", "minimum": 2, "maximum": 17},
            },
        },
        "controlPhenomena": _string_list_schema(3, 12),
    }
    prefix_items = []
    for assignment in assignments:
        properties = {"slot": {"const": assignment["slot"]}, **deepcopy(item_properties)}
        prefix_items.append({
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": properties,
        })
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "array",
                "minItems": len(assignments),
                "maxItems": len(assignments),
                "prefixItems": prefix_items,
                "items": False,
            }
        },
    }


class SchemaModel(Protocol):
    def generate(
        self, *, name: str, schema: Mapping[str, Any], instructions: str,
        context: Mapping[str, Any], max_output_tokens: int,
    ) -> tuple[Any, Mapping[str, Any]]:
        ...


class RoundRobinJsonSchemaPlanner:
    """Three-lane OpenAI-compatible planner with transport-only failover."""

    def __init__(
        self, endpoints: Sequence[str], model: str, api_key: str = "", timeout_seconds: int = 240
    ) -> None:
        normalized = tuple(dict.fromkeys(endpoint.strip().rstrip("/") for endpoint in endpoints if endpoint.strip()))
        if len(normalized) != 3:
            raise FanoutError("efficient v5 fan-out requires exactly three distinct planner endpoints")
        if not model.strip():
            raise FanoutError("planner model is required")
        self.endpoints = normalized
        self.model = model.strip()
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._lock = Lock()
        self._next = 0

    def _rotation(self) -> tuple[str, ...]:
        with self._lock:
            start = self._next
            self._next = (self._next + 1) % len(self.endpoints)
        return self.endpoints[start:] + self.endpoints[:start]

    def generate(
        self, *, name: str, schema: Mapping[str, Any], instructions: str,
        context: Mapping[str, Any], max_output_tokens: int,
    ) -> tuple[Any, Mapping[str, Any]]:
        payload = {
            "model": self.model,
            "stream": False,
            "reasoning": False,
            "temperature": 0.8,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": instructions},
                {"role": "user", "content": canonical_json(context)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "strict": True, "schema": schema},
            },
        }
        body = canonical_json(payload).encode("utf-8")
        transport_errors: list[str] = []
        for endpoint in self._rotation():
            headers = {"content-type": "application/json"}
            if self.api_key:
                headers["authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    envelope = json.loads(response.read())
            except urllib.error.HTTPError as error:
                raise FanoutError(
                    f"planner endpoint {endpoint} rejected the constrained request with HTTP {error.code}"
                ) from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                transport_errors.append(f"{endpoint}: {error}")
                continue
            try:
                content = envelope["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise FanoutError("planner response lacks choices[0].message.content") from error
            if not isinstance(content, str):
                raise FanoutError("planner constrained content must be a JSON string")
            try:
                value = json.loads(content)
            except json.JSONDecodeError as error:
                raise FanoutError("planner returned non-JSON constrained content") from error
            return value, {
                "endpoint": endpoint,
                "model": self.model,
                "responseHash": content_hash(value),
                "responseFormat": "json_schema_strict",
            }
        raise FanoutError("all planner transports failed: " + "; ".join(transport_errors))


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, value: Mapping[str, Any]) -> None:
        record = dict(value)
        record["auditRecordHash"] = content_hash(record)
        payload = (canonical_json(record) + "\n").encode("utf-8")
        with self._lock:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _validate_with_schema(value: Any, schema: Mapping[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return [error.message for error in sorted(validator.iter_errors(value), key=lambda item: list(item.path))]


def _validate_duplex_events(events: Sequence[Mapping[str, Any]], target_turns: int) -> None:
    if any(event["targetOrdinal"] > target_turns for event in events):
        raise LeafRejected("duplex event targets a turn outside conversationLength")
    barges = [event for event in events if event["eventType"] == "barge_in"]
    cancellations = [event for event in events if event["eventType"] == "cancelled_generation"]
    recoveries = [
        event for event in events
        if event["eventType"] in {"repair_after_barge_in", "recovery"}
    ]
    if not barges or not cancellations or not recoveries:
        raise LeafRejected("duplex plan requires barge_in, cancelled_generation, and recovery")
    for barge in barges:
        matching_cancel = next(
            (
                event for event in cancellations
                if event["targetOrdinal"] == barge["targetOrdinal"]
                and event["offsetMs"] >= barge["offsetMs"]
                and event["cancelOutgoingAudio"] is True
                and event["invalidateGeneration"] is True
            ),
            None,
        )
        matching_recovery = next(
            (event for event in recoveries if event["targetOrdinal"] > barge["targetOrdinal"]),
            None,
        )
        if matching_cancel is None or matching_recovery is None:
            raise LeafRejected("barge-in lacks an ordered authentic cancellation/recovery path")


def _candidate_semantics(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _normalized(raw[key])
        for key in (
            "premise", "interactionArc", "postureArc", "postureTransition",
            "evidenceSource", "evidenceEvolution", "duplexEvents", "outcomeRoute",
            "conversationLength", "style", "pivot", "controlPhenomena",
        )
    }


def build_candidate(
    raw: Mapping[str, Any], assignment: Mapping[str, Any],
    scenario: Mapping[str, Any], topic: Mapping[str, Any], item_schema: Mapping[str, Any],
) -> dict[str, Any]:
    errors = _validate_with_schema(raw, item_schema)
    if errors:
        raise LeafRejected("JSON Schema: " + "; ".join(errors[:4]))
    _assert_target_free(raw)
    length = raw["conversationLength"]
    if not length["min"] <= length["targetTurns"] <= length["max"]:
        raise LeafRejected("conversationLength must satisfy min <= targetTurns <= max")
    pivot = raw["pivot"]
    agent_targets = max(1, length["targetTurns"] // 2)
    if not 2 <= pivot["targetOrdinal"] < agent_targets:
        raise LeafRejected("causal pivot must precede at least one recovery target")
    if canonical_json(pivot["from"]) == canonical_json(pivot["to"]):
        raise LeafRejected("typed pivot from/to states must differ")
    if _normalized(raw["premise"]) == _normalized(scenario["premise"]):
        raise LeafRejected("compact premise must be a distinct trajectory, not a scenario copy")
    _validate_duplex_events(raw["duplexEvents"], agent_targets)
    typed_pivot = {
        "field": assignment["changedPath"],
        "from": deepcopy(pivot["from"]),
        "to": deepcopy(pivot["to"]),
    }
    causal_identity = content_hash({
        "candidateId": assignment["candidateId"],
        "scenarioId": scenario["scenarioId"],
        "causalAxis": assignment["causalAxis"],
        "interventionFamily": assignment["interventionFamily"],
        "operatorId": assignment["operatorId"],
        "typedPivot": typed_pivot,
        "pivotTargetOrdinal": pivot["targetOrdinal"],
    })
    candidate: dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "candidateId": assignment["candidateId"],
        "trajectoryId": assignment["trajectoryId"],
        "candidateOrdinal": assignment["slot"],
        "scenarioId": scenario["scenarioId"],
        "topicId": topic["topicId"],
        "operatorAssignment": {
            "operatorId": assignment["operatorId"],
            "family": assignment["interventionFamily"],
            "changedPath": assignment["changedPath"],
        },
        "causalAxis": assignment["causalAxis"],
        "interventionFamily": assignment["interventionFamily"],
        "typedPivot": typed_pivot,
        "counterfactualPivotOrdinal": pivot["targetOrdinal"],
        "premise": raw["premise"].strip(),
        "interactionArc": deepcopy(raw["interactionArc"]),
        "postureArc": deepcopy(raw["postureArc"]),
        "postureTransition": deepcopy(raw["postureTransition"]),
        "evidenceSource": raw["evidenceSource"],
        "evidenceEvolution": deepcopy(raw["evidenceEvolution"]),
        "duplexEvents": deepcopy(raw["duplexEvents"]),
        "outcomeRoute": raw["outcomeRoute"],
        "conversationLength": deepcopy(raw["conversationLength"]),
        "style": deepcopy(raw["style"]),
        "controlPhenomena": deepcopy(raw["controlPhenomena"]),
        "semanticFingerprint": content_hash(_candidate_semantics(raw)),
        "causalIdentity": causal_identity,
    }
    _assert_target_free(candidate)
    candidate["candidateHash"] = content_hash(candidate)
    return candidate


def _mode_values(candidate: Mapping[str, Any]) -> dict[str, str]:
    return {
        "evidenceSource": canonical_json(_normalized(candidate["evidenceSource"])),
        "outcomeRoute": canonical_json(_normalized(candidate["outcomeRoute"])),
        "postureTransition": content_hash(_normalized(candidate["postureTransition"])),
        "duplexProfile": content_hash([
            {
                "eventType": event["eventType"],
                "targetOrdinal": event["targetOrdinal"],
                "cancelOutgoingAudio": event["cancelOutgoingAudio"],
                "invalidateGeneration": event["invalidateGeneration"],
            }
            for event in candidate["duplexEvents"]
        ]),
        "style": content_hash(_normalized(candidate["style"])),
    }


class CandidateCheckpointStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._premises: dict[str, str] = {}
        self._fingerprints: dict[str, str] = {}
        self._mode_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
            lambda: {axis: Counter() for axis in MODE_COLLAPSE_LIMITS}
        )
        for path in sorted(self.directory.glob("*.json")):
            value = read_json(path)
            self._register(value, write=False)

    def _register(self, candidate: dict[str, Any], *, write: bool) -> bool:
        candidate_id = candidate.get("candidateId")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise FanoutError("candidate checkpoint lacks candidateId")
        immutable = {key: value for key, value in candidate.items() if key != "candidateHash"}
        if candidate.get("candidateHash") != content_hash(immutable):
            raise FanoutError(f"{candidate_id}: candidateHash is stale")
        prior = self._records.get(candidate_id)
        if prior is not None:
            if prior != candidate:
                raise FanoutError(f"{candidate_id}: immutable checkpoint content conflicts")
            return False
        premise = canonical_json(_normalized(candidate.get("premise")))
        fingerprint = candidate.get("semanticFingerprint")
        if premise in self._premises:
            raise LeafRejected(f"premise duplicates {self._premises[premise]}")
        if not isinstance(fingerprint, str) or fingerprint in self._fingerprints:
            raise LeafRejected(f"semantic trajectory duplicates {self._fingerprints.get(str(fingerprint), 'another leaf')}")
        scenario_id = str(candidate.get("scenarioId"))
        mode_values = _mode_values(candidate)
        for axis, value in mode_values.items():
            if self._mode_counts[scenario_id][axis][value] >= MODE_COLLAPSE_LIMITS[axis]:
                raise LeafRejected(f"mode-collapse budget exceeded for {axis}")
        if write:
            checkpoint = self.directory / f"{content_hash({'candidateId': candidate_id})[7:]}.json"
            write_json_atomic(checkpoint, candidate)
        self._records[candidate_id] = deepcopy(candidate)
        self._premises[premise] = candidate_id
        self._fingerprints[str(fingerprint)] = candidate_id
        for axis, value in mode_values.items():
            self._mode_counts[scenario_id][axis][value] += 1
        return True

    def admit(self, candidate: dict[str, Any]) -> bool:
        with self._lock:
            return self._register(candidate, write=True)

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(self._records[key]) for key in sorted(self._records)]

    def for_scenario(self, scenario_id: str) -> list[dict[str, Any]]:
        return [row for row in self.rows() if row["scenarioId"] == scenario_id]


def validate_v5_inputs(
    request: dict[str, Any], topics: list[dict[str, Any]], scenarios: list[dict[str, Any]]
) -> None:
    validate_request(request)
    if request.get("strategyVersion") != "semantic-control-v5":
        raise FanoutError("efficient trajectory fan-out requires semantic-control-v5")
    coverage = request["coverageTarget"]
    if coverage.get("trajectorySeedsPerScenario") != LEAVES_PER_SCENARIO:
        raise FanoutError("efficient v5 fan-out is pinned to exactly ten leaves per scenario")
    if len(topics) != coverage["candidateTopics"]:
        raise FanoutError("completed topic_cards.jsonl count differs from request")
    if len(scenarios) != len(topics) * coverage["scenariosPerTopic"]:
        raise FanoutError("completed scenario_contracts.jsonl count differs from request")
    topic_ids = {topic.get("topicId") for topic in topics}
    if len(topic_ids) != len(topics):
        raise FanoutError("topic IDs are missing or duplicated")
    for topic in topics:
        validate_topic_card(topic, request["seedRevision"])
    scenario_ids = {scenario.get("scenarioId") for scenario in scenarios}
    if len(scenario_ids) != len(scenarios):
        raise FanoutError("scenario IDs are missing or duplicated")
    for scenario in scenarios:
        validate_scenario_contract(scenario, topic_ids)
    for topic_id in topic_ids:
        if sum(scenario["topicId"] == topic_id for scenario in scenarios) != coverage["scenariosPerTopic"]:
            raise FanoutError(f"{topic_id}: scenario fan-out is incomplete")


def _allowed_sources(request: Mapping[str, Any]) -> tuple[str, ...]:
    requested = (request.get("requiredControlCoverage") or {}).get("stateSources") or []
    return tuple(dict.fromkeys([*requested, "scenario_state", "state_reducer"]))


def _scenario_prompt_context(
    request: Mapping[str, Any], topic: Mapping[str, Any], scenario: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]], accepted: Sequence[Mapping[str, Any]],
    prior_rejections: Mapping[int, str],
) -> dict[str, Any]:
    return {
        "task": "Generate compact target-free semantic trajectory candidates for only the requested deterministic slots.",
        "requestId": request["requestId"],
        "topic": topic,
        "scenario": scenario,
        "slots": [dict(item) for item in assignments],
        "acceptedLeaves": [
            {
                "slot": row["candidateOrdinal"],
                "premise": row["premise"],
                "semanticFingerprint": row["semanticFingerprint"],
                "evidenceSource": row["evidenceSource"],
                "outcomeRoute": row["outcomeRoute"],
                "postureTransition": row["postureTransition"],
                "style": row["style"],
            }
            for row in accepted
        ],
        "priorRejections": {str(key): value for key, value in sorted(prior_rejections.items())},
        "requirements": [
            "All premise, arc, posture, evidence evolution, duplex timing, outcome, length, and style content must be newly model-generated.",
            "Do not emit IDs or operator fields; deterministic identity and operator assignment are supplied outside model output.",
            "Describe behavior and state only. Never emit dialogue, a target response, quoted speech, scripts, transcripts, or canonical wording.",
            "Every leaf must include an actual barge-in, cancellation, and later recovery event with ordered millisecond timing.",
            "Make all requested leaves materially different across premise, posture, evidence source, duplex profile, outcome, length, and style.",
        ],
    }


def generate_compact_candidates(
    *, request: dict[str, Any], topics: list[dict[str, Any]], scenarios: list[dict[str, Any]],
    output_root: Path, planner: SchemaModel, max_workers: int = 3, max_attempts: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_v5_inputs(request, topics, scenarios)
    if not 1 <= max_workers <= 3 or max_attempts < 1:
        raise FanoutError("candidate fan-out requires 1-3 workers and at least one attempt")
    topic_by_id = {topic["topicId"]: topic for topic in topics}
    store = CandidateCheckpointStore(output_root / ".efficient_v5_checkpoints" / "candidates")
    audit = AuditLog(output_root / CANDIDATE_AUDIT_FILENAME)
    allowed_sources = _allowed_sources(request)

    def generate_scenario(scenario: dict[str, Any]) -> None:
        topic = topic_by_id[scenario["topicId"]]
        assignments = deterministic_candidate_assignments(request, topic, scenario)
        assignment_by_slot = {item["slot"]: item for item in assignments}
        prior_rejections: dict[int, str] = {}
        for attempt in range(1, max_attempts + 1):
            accepted = store.for_scenario(scenario["scenarioId"])
            accepted_slots = {row["candidateOrdinal"] for row in accepted}
            missing = [assignment for assignment in assignments if assignment["slot"] not in accepted_slots]
            if not missing:
                return
            schema = compact_response_schema(missing, allowed_sources)
            context = _scenario_prompt_context(
                request, topic, scenario, missing, accepted, prior_rejections
            )
            response, metadata = planner.generate(
                name="personaplex_compact_trajectories_v5",
                schema=schema,
                instructions=(
                    "You generate diverse semantic-control training plans. Return only the strict JSON-Schema object. "
                    "Never produce dialogue or target wording and never imitate a missing field with a placeholder."
                ),
                context=context,
                max_output_tokens=8_000,
            )
            rejected: dict[int, str] = {}
            accepted_now: list[int] = []
            items = response.get("candidates") if isinstance(response, Mapping) and set(response) == {"candidates"} else None
            raw_by_slot: dict[int, list[Any]] = defaultdict(list)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, Mapping) and isinstance(item.get("slot"), int):
                        raw_by_slot[int(item["slot"])].append(item)
            for position, assignment in enumerate(missing):
                slot = assignment["slot"]
                matches = raw_by_slot.get(slot, [])
                if len(matches) != 1:
                    rejected[slot] = "model response did not contain exactly one object for the requested slot"
                    continue
                try:
                    candidate = build_candidate(
                        matches[0], assignment, scenario, topic,
                        schema["properties"]["candidates"]["prefixItems"][position],
                    )
                    store.admit(candidate)
                    accepted_now.append(slot)
                except (CascadeError, LeafRejected, ValueError, TypeError, KeyError) as error:
                    rejected[slot] = str(error)
            audit.append({
                "schema": "personaplex.compact-trajectory-attempt.v1",
                "scenarioId": scenario["scenarioId"],
                "attempt": attempt,
                "requestedSlots": [item["slot"] for item in missing],
                "acceptedSlots": accepted_now,
                "rejectedSlots": {str(key): value for key, value in sorted(rejected.items())},
                "planner": dict(metadata),
            })
            prior_rejections = rejected
        remaining = sorted(
            set(range(LEAVES_PER_SCENARIO))
            - {row["candidateOrdinal"] for row in store.for_scenario(scenario["scenarioId"])}
        )
        if remaining:
            raise FanoutError(
                f"{scenario['scenarioId']}: unresolved compact leaves after {max_attempts} attempts: {remaining}"
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_scenario, scenario): scenario for scenario in scenarios}
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    rows = store.rows()
    expected_ids = {
        assignment["candidateId"]
        for scenario in scenarios
        for assignment in deterministic_candidate_assignments(request, topic_by_id[scenario["topicId"]], scenario)
    }
    if len(rows) != len(scenarios) * LEAVES_PER_SCENARIO or {row["candidateId"] for row in rows} != expected_ids:
        raise FanoutError("candidate checkpoints do not exactly cover the 10-leaf scenario fan-out")
    for scenario in scenarios:
        if len([row for row in rows if row["scenarioId"] == scenario["scenarioId"]]) != LEAVES_PER_SCENARIO:
            raise FanoutError(f"{scenario['scenarioId']}: candidate cardinality is not ten")
    if len({row["semanticFingerprint"] for row in rows}) != len(rows):
        raise FanoutError("candidate corpus contains duplicate semantic fingerprints")
    rows.sort(key=lambda row: (row["scenarioId"], row["candidateOrdinal"]))
    candidate_path = output_root / CANDIDATES_FILENAME
    write_jsonl_atomic(candidate_path, rows)
    manifest_payload = {
        "schema": CANDIDATE_MANIFEST_SCHEMA,
        "requestHash": content_hash(request),
        "topicSetHash": content_hash(topics),
        "scenarioSetHash": content_hash(scenarios),
        "candidateCount": len(rows),
        "scenarios": len(scenarios),
        "leavesPerScenario": LEAVES_PER_SCENARIO,
        "candidateSetHash": content_hash(rows),
        "files": {
            CANDIDATES_FILENAME: hash_file(candidate_path),
            CANDIDATE_AUDIT_FILENAME: hash_file(output_root / CANDIDATE_AUDIT_FILENAME),
        },
    }
    manifest = {**manifest_payload, "manifestHash": content_hash(manifest_payload)}
    write_json_atomic(output_root / CANDIDATE_MANIFEST_FILENAME, manifest)
    return rows, manifest


def candidate_projection(candidate: Mapping[str, Any]) -> dict[str, Any]:
    agent_targets = max(1, candidate["conversationLength"]["targetTurns"] // 2)
    return {
        "schema": "personaplex.trajectory-seed.v2",
        "trajectoryId": candidate["trajectoryId"],
        "scenarioId": candidate["scenarioId"],
        "conversationLength": deepcopy(candidate["conversationLength"]),
        "pace": candidate["style"]["pace"],
        "openingStyle": candidate["style"]["openingStyle"],
        "closingStyle": candidate["style"]["closingStyle"],
        "voicePairPolicy": "distinct_approved_references",
        "interactionArc": deepcopy(candidate["interactionArc"]),
        "duplexEvents": deepcopy(candidate["duplexEvents"]),
        "postureArc": deepcopy(candidate["postureArc"]),
        "counterfactualPivotOrdinal": candidate["counterfactualPivotOrdinal"],
        "controlPhenomena": deepcopy(candidate["controlPhenomena"]),
        "causalAxis": candidate["causalAxis"],
        "interventionFamily": candidate["interventionFamily"],
        "typedPivot": deepcopy(candidate["typedPivot"]),
        "postureTransition": deepcopy(candidate["postureTransition"]),
        "evidenceSource": candidate["evidenceSource"],
        "outcomeRoute": candidate["outcomeRoute"],
        "semanticStateArc": [{"candidateProjectionHash": candidate["candidateHash"]}],
        "controlRevisionSchedule": [
            {
                "controlRevision": ordinal,
                "targetOrdinal": ordinal,
                "availableBeforeTarget": True,
                "source": candidate["evidenceSource"],
            }
            for ordinal in range(1, agent_targets + 1)
        ],
        "terminationContract": deepcopy(TERMINATION_CONTRACT),
        "negativeControlCoverage": list(NEGATIVE_CONTROLS),
    }


def select_compact_candidates(
    *, request: dict[str, Any], topics: list[dict[str, Any]], scenarios: list[dict[str, Any]],
    candidates: list[dict[str, Any]], output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expected = len(scenarios) * LEAVES_PER_SCENARIO
    if len(candidates) != expected or len({row["candidateId"] for row in candidates}) != expected:
        raise FanoutError("selection requires the complete unique compact candidate corpus")
    projections = [candidate_projection(candidate) for candidate in candidates]
    validate_unique_causal_signatures(projections, require_typed=True)
    selected = select_trajectories(request, topics, scenarios, projections)
    primary_count, reserve_count = request_selection_counts(request)
    primary = [row for row in selected if row["selectionTier"] == "primary"]
    reserve = [row for row in selected if row["selectionTier"] == "reserve"]
    if len(primary) != primary_count or len(reserve) != reserve_count:
        raise FanoutError("typed selector did not return the exact primary/reserve contract")
    primary_ids = {row["trajectoryId"] for row in primary}
    reserve_ids = {row["trajectoryId"] for row in reserve}
    candidate_ids = {row["trajectoryId"] for row in candidates}
    if primary_ids & reserve_ids or not (primary_ids | reserve_ids).issubset(candidate_ids):
        raise FanoutError("selection is overlapping or references a non-candidate trajectory")
    write_jsonl_atomic(output_root / PRIMARY_FILENAME, primary)
    write_jsonl_atomic(output_root / RESERVE_FILENAME, reserve)
    write_jsonl_atomic(output_root / SELECTED_FILENAME, primary)
    candidate_by_id = {row["trajectoryId"]: row for row in candidates}
    selection_audit = [
        {
            "schema": "personaplex.compact-trajectory-selection-audit.v1",
            "trajectoryId": row["trajectoryId"],
            "groupId": row["groupId"],
            "selectionTier": row["selectionTier"],
            "candidateHash": candidate_by_id[row["trajectoryId"]]["candidateHash"],
            "causalIdentity": candidate_by_id[row["trajectoryId"]]["causalIdentity"],
            "selectionHash": row["selectionHash"],
            "balanceDimensions": row["balanceDimensions"],
        }
        for row in selected
    ]
    write_jsonl_atomic(output_root / SELECTION_AUDIT_FILENAME, selection_audit)
    manifest_payload = {
        "schema": "personaplex.compact-trajectory-selection-manifest.v1",
        "requestHash": content_hash(request),
        "candidateSetHash": content_hash(candidates),
        "algorithm": request["selectionPolicy"]["algorithm"],
        "primaryCount": len(primary),
        "reserveCount": len(reserve),
        "selectedCandidateSetHash": content_hash(sorted(primary_ids | reserve_ids)),
        "files": {
            name: hash_file(output_root / name)
            for name in (PRIMARY_FILENAME, RESERVE_FILENAME, SELECTED_FILENAME, SELECTION_AUDIT_FILENAME)
        },
    }
    manifest = {**manifest_payload, "manifestHash": content_hash(manifest_payload)}
    write_json_atomic(output_root / SELECTION_MANIFEST_FILENAME, manifest)
    return primary, reserve, manifest


def full_trajectory_response_schema(candidate: Mapping[str, Any]) -> dict[str, Any]:
    agent_targets = max(1, candidate["conversationLength"]["targetTurns"] // 2)
    pivot = candidate["counterfactualPivotOrdinal"]
    typed_pivot = candidate["typedPivot"]
    state_items = []
    revision_items = []
    for ordinal in range(1, agent_targets + 1):
        active_value = typed_pivot["from"] if ordinal < pivot else typed_pivot["to"]
        state_properties = {
            "targetOrdinal": {"const": ordinal},
            "phase": {"type": "string", "minLength": 2},
            "availableBeforeTarget": {"const": True},
            "controlRevision": {"type": "integer", "minimum": 1},
            "knownFacts": _string_list_schema(1, 12),
            "uncertainty": _string_list_schema(1, 10),
            "policyConstraints": _string_list_schema(1, 10),
            "commitments": _string_list_schema(1, 8),
            "callerPosture": {"type": "string", "minLength": 2},
            "nextGoal": {"type": "string", "minLength": 4},
            "evidence": {
                "type": "object", "additionalProperties": False,
                "required": ["source", "status", "facts"],
                "properties": {
                    "source": {"enum": list(CONTROL_SOURCES)},
                    "status": {"type": "string", "minLength": 2},
                    "facts": _string_list_schema(1, 10),
                },
            },
            "toolResult": {
                "type": "object", "additionalProperties": False,
                "required": ["source", "status", "facts"],
                "properties": {
                    "source": {"type": "string", "minLength": 2},
                    "status": {"type": "string", "minLength": 2},
                    "facts": _string_list_schema(1, 10),
                },
            },
            "causalState": {
                "type": "object", "additionalProperties": False,
                "required": ["causalIdentity", "operatorId", "changedPath", "from", "to", "activeValue"],
                "properties": {
                    "causalIdentity": {"const": candidate["causalIdentity"]},
                    "operatorId": {"const": candidate["operatorAssignment"]["operatorId"]},
                    "changedPath": {"const": typed_pivot["field"]},
                    "from": {"const": typed_pivot["from"]},
                    "to": {"const": typed_pivot["to"]},
                    "activeValue": {"const": active_value},
                },
            },
            "revisionReason": {"type": "string", "minLength": 4},
        }
        state_items.append({
            "type": "object", "additionalProperties": False,
            "required": list(state_properties), "properties": state_properties,
        })
        revision_items.append({
            "type": "object", "additionalProperties": False,
            "required": ["controlRevision", "targetOrdinal", "availableBeforeTarget", "source"],
            "properties": {
                "controlRevision": {"type": "integer", "minimum": 1},
                "targetOrdinal": {"const": ordinal},
                "availableBeforeTarget": {"const": True},
                "source": {"enum": list(CONTROL_SOURCES)},
            },
        })
    constants = {
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
        "terminationContract": TERMINATION_CONTRACT,
        "negativeControlCoverage": NEGATIVE_CONTROLS,
    }
    properties: dict[str, Any] = {key: {"const": value} for key, value in constants.items()}
    properties["semanticStateArc"] = {
        "type": "array", "minItems": agent_targets, "maxItems": agent_targets,
        "prefixItems": state_items, "items": False,
    }
    properties["controlRevisionSchedule"] = {
        "type": "array", "minItems": agent_targets, "maxItems": agent_targets,
        "prefixItems": revision_items, "items": False,
    }
    trajectory_schema = {
        "type": "object", "additionalProperties": False,
        "required": list(properties), "properties": properties,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object", "additionalProperties": False,
        "required": ["trajectory"],
        "properties": {"trajectory": trajectory_schema},
    }


def validate_expanded_trajectory(
    trajectory: dict[str, Any], candidate: Mapping[str, Any], known_scenarios: set[str]
) -> None:
    schema = full_trajectory_response_schema(candidate)
    errors = _validate_with_schema({"trajectory": trajectory}, schema)
    if errors:
        raise LeafRejected("JSON Schema: " + "; ".join(errors[:6]))
    _assert_target_free(trajectory)
    validate_trajectory_seed(trajectory, known_scenarios, require_typed=True)
    agent_targets = max(1, trajectory["conversationLength"]["targetTurns"] // 2)
    expected_ordinals = list(range(1, agent_targets + 1))
    schedule = trajectory["controlRevisionSchedule"]
    states = trajectory["semanticStateArc"]
    if [item["targetOrdinal"] for item in schedule] != expected_ordinals:
        raise LeafRejected("controlRevisionSchedule must cover every target in order")
    if [item["targetOrdinal"] for item in states] != expected_ordinals:
        raise LeafRejected("semanticStateArc must cover every target in order")
    revisions = [item["controlRevision"] for item in schedule]
    if any(right <= left for left, right in zip(revisions, revisions[1:])):
        raise LeafRejected("control revisions must increase strictly")
    if [item["controlRevision"] for item in states] != revisions:
        raise LeafRejected("semantic states and revision schedule disagree")
    if any(item["availableBeforeTarget"] is not True for item in states + schedule):
        raise LeafRejected("every control revision must be available strictly before its target")
    _validate_duplex_events(trajectory["duplexEvents"], agent_targets)
    if trajectory["terminationContract"] != TERMINATION_CONTRACT:
        raise LeafRejected("termination must remain model-selected end_call_tool")
    if trajectory["negativeControlCoverage"] != NEGATIVE_CONTROLS:
        raise LeafRejected("negative-control identity constants changed")


class ExpansionCheckpointStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._records: dict[str, dict[str, Any]] = {}
        for path in sorted(self.directory.glob("*.json")):
            self._register(read_json(path), write=False)

    def _register(self, wrapper: dict[str, Any], *, write: bool) -> bool:
        if wrapper.get("schema") != EXPANSION_CHECKPOINT_SCHEMA:
            raise FanoutError("unsupported expansion checkpoint schema")
        record_hash = wrapper.get("recordHash")
        immutable = {key: value for key, value in wrapper.items() if key != "recordHash"}
        if record_hash != content_hash(immutable):
            raise FanoutError("expansion checkpoint recordHash is stale")
        trajectory_id = wrapper.get("trajectoryId")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise FanoutError("expansion checkpoint lacks trajectoryId")
        prior = self._records.get(trajectory_id)
        if prior is not None:
            if prior != wrapper:
                raise FanoutError(f"{trajectory_id}: expansion checkpoint conflicts")
            return False
        if write:
            path = self.directory / f"{content_hash({'trajectoryId': trajectory_id})[7:]}.json"
            write_json_atomic(path, wrapper)
        self._records[trajectory_id] = deepcopy(wrapper)
        return True

    def admit(self, candidate: Mapping[str, Any], trajectory: dict[str, Any]) -> bool:
        payload = {
            "schema": EXPANSION_CHECKPOINT_SCHEMA,
            "trajectoryId": candidate["trajectoryId"],
            "candidateHash": candidate["candidateHash"],
            "causalIdentity": candidate["causalIdentity"],
            "trajectoryHash": content_hash(trajectory),
            "trajectory": trajectory,
        }
        wrapper = {**payload, "recordHash": content_hash(payload)}
        with self._lock:
            return self._register(wrapper, write=True)

    def get(self, trajectory_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._records.get(trajectory_id)
            return deepcopy(value) if value is not None else None

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(self._records[key]) for key in sorted(self._records)]


def _expansion_prompt_context(
    request: Mapping[str, Any], topic: Mapping[str, Any], scenario: Mapping[str, Any],
    candidate: Mapping[str, Any], prior_error: str | None,
) -> dict[str, Any]:
    return {
        "task": "Expand this selected compact candidate into its complete typed semantic-control trajectory.",
        "requestId": request["requestId"],
        "topic": topic,
        "scenario": scenario,
        "candidate": candidate,
        "priorValidationError": prior_error,
        "requirements": [
            "Preserve every JSON-Schema const exactly; these are immutable causal and balance identities.",
            "Populate a target-free semanticStateArc and strictly increasing controlRevisionSchedule for every agent target.",
            "Every semantic state is available before its response and changes causalState.activeValue exactly at the pivot.",
            "Describe facts, uncertainty, policy, commitments, caller posture, tool state, and next goal without writing dialogue.",
            "The explicit barge-in, cancellation, invalidation, recovery timing is real planned duplex structure, not a label.",
            "Termination remains a model-selected end_call tool action without deterministic sign-off wording.",
        ],
    }


def expand_selected_candidates(
    *, request: dict[str, Any], topics: list[dict[str, Any]], scenarios: list[dict[str, Any]],
    candidates: list[dict[str, Any]], primary: list[dict[str, Any]], reserve: list[dict[str, Any]],
    output_root: Path, planner: SchemaModel, max_workers: int = 3, max_attempts: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 1 <= max_workers <= 3 or max_attempts < 1:
        raise FanoutError("expansion requires 1-3 workers and at least one attempt")
    candidate_by_id = {row["trajectoryId"]: row for row in candidates}
    selected_ids = [row["trajectoryId"] for row in primary + reserve]
    if len(selected_ids) != len(set(selected_ids)) or any(item not in candidate_by_id for item in selected_ids):
        raise FanoutError("selected primary/reserve candidates overlap or are missing")
    primary_count, reserve_count = request_selection_counts(request)
    if len(primary) != primary_count or len(reserve) != reserve_count:
        raise FanoutError("expansion requires the exact request primary/reserve cardinality")
    topic_by_id = {row["topicId"]: row for row in topics}
    scenario_by_id = {row["scenarioId"]: row for row in scenarios}
    known_scenarios = set(scenario_by_id)
    store = ExpansionCheckpointStore(output_root / ".efficient_v5_checkpoints" / "expanded")
    audit = AuditLog(output_root / EXPANSION_AUDIT_FILENAME)

    def expand_one(trajectory_id: str) -> None:
        candidate = candidate_by_id[trajectory_id]
        existing = store.get(trajectory_id)
        if existing is not None:
            if existing["candidateHash"] != candidate["candidateHash"]:
                raise FanoutError(f"{trajectory_id}: selected candidate changed after expansion")
            validate_expanded_trajectory(existing["trajectory"], candidate, known_scenarios)
            return
        scenario = scenario_by_id[candidate["scenarioId"]]
        topic = topic_by_id[candidate["topicId"]]
        prior_error = None
        schema = full_trajectory_response_schema(candidate)
        for attempt in range(1, max_attempts + 1):
            response, metadata = planner.generate(
                name="personaplex_full_trajectory_v2",
                schema=schema,
                instructions=(
                    "You expand one selected semantic trajectory into strict target-free JSON. "
                    "Return only the JSON-Schema object. Do not write any caller or agent dialogue."
                ),
                context=_expansion_prompt_context(
                    request, topic, scenario, candidate, prior_error
                ),
                max_output_tokens=10_000,
            )
            trajectory = response.get("trajectory") if isinstance(response, Mapping) and set(response) == {"trajectory"} else None
            try:
                if not isinstance(trajectory, dict):
                    raise LeafRejected("model response lacks exactly one trajectory object")
                validate_expanded_trajectory(trajectory, candidate, known_scenarios)
                store.admit(candidate, trajectory)
                audit.append({
                    "schema": "personaplex.trajectory-expansion-attempt.v1",
                    "trajectoryId": trajectory_id,
                    "attempt": attempt,
                    "accepted": True,
                    "validationError": None,
                    "planner": dict(metadata),
                })
                return
            except (CascadeError, LeafRejected, ValueError, TypeError, KeyError) as error:
                prior_error = str(error)
                audit.append({
                    "schema": "personaplex.trajectory-expansion-attempt.v1",
                    "trajectoryId": trajectory_id,
                    "attempt": attempt,
                    "accepted": False,
                    "validationError": prior_error,
                    "planner": dict(metadata),
                })
        raise FanoutError(
            f"{trajectory_id}: full trajectory remained invalid after {max_attempts} attempts: {prior_error}"
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(expand_one, trajectory_id): trajectory_id for trajectory_id in selected_ids}
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    wrappers = store.rows()
    selected_set = set(selected_ids)
    selected_wrappers = [row for row in wrappers if row["trajectoryId"] in selected_set]
    if len(selected_wrappers) != len(selected_ids) or {row["trajectoryId"] for row in selected_wrappers} != selected_set:
        raise FanoutError("expansion checkpoints do not exactly cover primary plus reserve selections")
    trajectories = [row["trajectory"] for row in selected_wrappers]
    trajectories.sort(key=lambda row: row["trajectoryId"])
    for trajectory in trajectories:
        validate_expanded_trajectory(
            trajectory, candidate_by_id[trajectory["trajectoryId"]], known_scenarios
        )
    validate_unique_causal_signatures(trajectories, require_typed=True)
    trajectory_path = output_root / TRAJECTORIES_FILENAME
    write_jsonl_atomic(trajectory_path, trajectories)
    manifest_payload = {
        "schema": "personaplex.trajectory-expansion-manifest.v1",
        "requestHash": content_hash(request),
        "selectedCandidateSetHash": content_hash(sorted(selected_set)),
        "trajectoryCount": len(trajectories),
        "primaryCount": len(primary),
        "reserveCount": len(reserve),
        "trajectorySetHash": content_hash(trajectories),
        "candidateBindings": {
            row["trajectoryId"]: {
                "candidateHash": row["candidateHash"],
                "causalIdentity": row["causalIdentity"],
                "trajectoryHash": row["trajectoryHash"],
            }
            for row in selected_wrappers
        },
        "files": {
            TRAJECTORIES_FILENAME: hash_file(trajectory_path),
            EXPANSION_AUDIT_FILENAME: hash_file(output_root / EXPANSION_AUDIT_FILENAME),
        },
    }
    manifest = {**manifest_payload, "manifestHash": content_hash(manifest_payload)}
    write_json_atomic(output_root / EXPANSION_MANIFEST_FILENAME, manifest)
    return trajectories, manifest


def write_combined_manifest(output_root: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    tracked = (
        CANDIDATES_FILENAME,
        CANDIDATE_AUDIT_FILENAME,
        CANDIDATE_MANIFEST_FILENAME,
        PRIMARY_FILENAME,
        RESERVE_FILENAME,
        SELECTED_FILENAME,
        SELECTION_AUDIT_FILENAME,
        SELECTION_MANIFEST_FILENAME,
        TRAJECTORIES_FILENAME,
        EXPANSION_AUDIT_FILENAME,
        EXPANSION_MANIFEST_FILENAME,
    )
    files = {
        name: hash_file(output_root / name)
        for name in tracked
        if (output_root / name).is_file()
    }
    payload = {
        "schema": FANOUT_MANIFEST_SCHEMA,
        "requestHash": content_hash(request),
        "architecture": "compact-10-per-scenario_then-balanced-500_then-full-expand",
        "files": files,
    }
    manifest = {**payload, "manifestHash": content_hash(payload)}
    write_json_atomic(output_root / COMBINED_MANIFEST_FILENAME, manifest)
    return manifest
