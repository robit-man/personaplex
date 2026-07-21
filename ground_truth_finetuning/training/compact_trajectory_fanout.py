"""Compact, resumable semantic-control-v5 trajectory fan-out.

Stage A performs one authentic, reasoning-disabled, strict JSON-Schema model
call per scenario.  The model returns all ten compact leaves and chooses every
semantic field.  Host code binds lineage, validates structure and exact
equality constraints, and rejects malformed responses wholesale; it never
generates or repairs semantic content.

Stage B is model-independent.  It balances only declared typed metadata while
enforcing one selected group per scenario across primary and reserve tiers.

Stage C performs authentic strict-schema expansion for primary plus reserve
cards only.  Scenario and expansion checkpoints are immutable and addressed by
the hash of their content.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import tempfile
from threading import Lock
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import urllib.error
import urllib.parse
import urllib.request

from jsonschema import Draft202012Validator

from ground_truth_finetuning.training.diverse_cascade import (
    CascadeError,
    canonical_json,
    content_hash,
    request_sibling_roles,
    request_selection_counts,
    validate_request,
    validate_scenario_contract,
    validate_topic_card,
    validate_trajectory_seed,
)
from ground_truth_finetuning.training.strict_schema_transport import (
    DEFAULT_RETRY_BASE_SECONDS,
    DEFAULT_RETRY_MAX_SECONDS,
    RETRIABLE_HTTP_STATUS,
    build_schema_transport_projection,
    canonical_retry_context,
    canonical_validation_defect,
    classify_provider_error,
    is_openrouter_endpoint,
    normalize_chat_completion_endpoints,
    retry_after_seconds,
    retry_delay_seconds,
    validate_retry_settings,
)


PRODUCTION_TOPIC_COUNT = 50
PRODUCTION_SCENARIOS_PER_TOPIC = 20
PRODUCTION_SCENARIO_COUNT = 1_000
LEAVES_PER_SCENARIO = 10
PRODUCTION_CANDIDATE_COUNT = 10_000
PRODUCTION_PRIMARY_COUNT = 250
PRODUCTION_RESERVE_COUNT = 250
PRODUCTION_EXPANSION_COUNT = 500

CUDA_DEVICES = (0, 1, 2)
MAX_STAGE_ATTEMPTS = 4
MAX_PROTOCOL_ATTEMPTS = 6
MAX_ENDPOINT_TIMEOUT_SECONDS = 600
MAX_SCHEMA_OUTPUT_TOKENS = 4_096
STAGE_A_MAX_OUTPUT_TOKENS = 3_072
STAGE_A_OUTPUT_TOKEN_HEADROOM = MAX_SCHEMA_OUTPUT_TOKENS - STAGE_A_MAX_OUTPUT_TOKENS
MAX_COMPACT_RESPONSE_BYTES = 8_192
MAX_MODEL_RESPONSE_BYTES = 1_048_576

COMPACT_CARD_SCHEMA = "personaplex.compact-trajectory-leaf.v5"
COMPACT_RESPONSE_NAME = "personaplex_compact_trajectory_fanout_v5"
EXPANSION_RESPONSE_NAME = "personaplex_full_trajectory_seed_v2"
SCENARIO_CHECKPOINT_SCHEMA = "personaplex.compact-trajectory-scenario-checkpoint.v1"
CANDIDATE_CHECKPOINT_SCHEMA = "personaplex.compact-trajectory-candidate-checkpoint.v1"
EXPANSION_CHECKPOINT_SCHEMA = "personaplex.compact-trajectory-expansion-checkpoint.v1"

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
COMBINED_MANIFEST_FILENAME = "compact_trajectory_fanout_v5_manifest.json"

CHECKPOINT_ROOT = ".compact_trajectory_fanout_v5"
CAUSAL_SIBLING_ROLES = (
    "verified_positive",
    "verified_negative",
    "uncertain",
    "superseded",
)
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
LENGTH_BANDS = ("short", "medium", "long")
STYLE_PROFILES = (
    "measured_clarification",
    "brisk_coordination",
    "deliberate_evidence_review",
    "adaptive_repair",
    "calm_boundary_setting",
    "concise_handoff",
)
DUPLEX_PROFILES: dict[str, tuple[str, ...]] = {
    "barge_cancel_recover": ("barge_in", "cancelled_generation", "recovery"),
    "barge_cancel_repair": ("barge_in", "cancelled_generation", "repair_after_barge_in"),
    "barge_cancel_recover_overlap": (
        "barge_in", "brief_overlap", "cancelled_generation", "recovery",
    ),
    "barge_cancel_recover_backchannel": (
        "backchannel", "barge_in", "cancelled_generation", "recovery",
    ),
    "turn_then_barge_cancel_recover": (
        "completed_turn", "barge_in", "cancelled_generation", "recovery",
    ),
}
POSTURE_STATES = (
    "cooperative", "guarded", "skeptical", "frustrated",
    "uncertain", "conditional", "reassured", "resolved",
)
STATE_TRANSITIONS = (
    "unknown_to_verified", "pending_to_available", "allowed_to_blocked",
    "blocked_to_allowed", "uncertain_to_bounded", "stale_to_current",
    "active_to_superseded", "local_to_handoff",
)
STATE_TRANSITION_VALUES: dict[str, tuple[str, str]] = {
    "unknown_to_verified": ("unknown", "verified"),
    "pending_to_available": ("pending", "available"),
    "allowed_to_blocked": ("allowed", "blocked"),
    "blocked_to_allowed": ("blocked", "allowed"),
    "uncertain_to_bounded": ("uncertain", "bounded"),
    "stale_to_current": ("stale", "current"),
    "active_to_superseded": ("active", "superseded"),
    "local_to_handoff": ("local", "handoff"),
}
POSTURE_TRANSITIONS: dict[str, tuple[str, str]] = {
    "guarded_to_conditional": ("guarded", "conditional"),
    "skeptical_to_reassured": ("skeptical", "reassured"),
    "frustrated_to_cooperative": ("frustrated", "cooperative"),
    "uncertain_to_resolved": ("uncertain", "resolved"),
    "cooperative_to_guarded": ("cooperative", "guarded"),
    "conditional_to_skeptical": ("conditional", "skeptical"),
}
LENGTH_PIVOT_PROFILES: dict[str, tuple[str, int, int]] = {
    "s8p2": ("short", 8, 2),
    "m10p2": ("medium", 10, 2),
    "m10p3": ("medium", 10, 3),
    "m12p3": ("medium", 12, 3),
    "m12p4": ("medium", 12, 4),
    "l14p2": ("long", 14, 2),
    "l14p4": ("long", 14, 4),
    "l14p5": ("long", 14, 5),
    "l16p5": ("long", 16, 5),
    "l18p6": ("long", 18, 6),
}
INTERACTION_MECHANISMS = (
    "clarify_then_verify", "verify_then_offer_options", "constraint_then_replan",
    "interruption_then_recover", "uncertainty_then_bound", "policy_then_redirect",
    "evidence_then_handoff", "revision_then_confirm",
)
BALANCE_AXES = (
    "causalAxis",
    "interventionFamily",
    "postureTransition",
    "evidenceSource",
    "duplexEventType",
    "outcomeRoute",
    "conversationLength",
    "style",
)
TARGET_DIALOGUE_FIELDS = frozenset(
    {
        "agentText",
        "callerText",
        "targetText",
        "targetDialogue",
        "targetTranscript",
        "targetAudio",
        "targetHash",
        "canonicalResponse",
        "dialogue",
        "transcript",
        "utterance",
        "reply",
        "script",
        "agent_text",
        "caller_text",
        "target_text",
        "target_dialogue",
        "target_transcript",
        "target_audio",
        "target_hash",
        "canonical_response",
        "groundTruth",
        "ground_truth",
        "expectedResponse",
        "expected_response",
        "referenceResponse",
        "reference_response",
    }
)

TOPIC_CARD_V2_FIELDS = frozenset(
    {
        "schema",
        "topicId",
        "sourceSeedId",
        "seedRevision",
        "domain",
        "interactionModes",
        "registerRange",
        "safeStakes",
        "forbiddenPatterns",
        "diversityTags",
        "causalAffordances",
    }
)
SCENARIO_CONTRACT_V2_FIELDS = frozenset(
    {
        "schema",
        "scenarioId",
        "topicId",
        "mode",
        "premise",
        "participants",
        "startingState",
        "interactionOpportunity",
        "allowedToolClasses",
        "disallowedClaims",
        "scenarioOutcomeSpace",
        "requiredControlPhenomena",
    }
)
COMPACT_CANDIDATE_FIELDS = frozenset(
    {
        "schema",
        "candidateId",
        "trajectoryId",
        "candidateOrdinal",
        "lineage",
        "causalAxis",
        "interventionFamily",
        "causalOperator",
        "typedPivot",
        "counterfactualPivotOrdinal",
        "stateTransition",
        "postureTransition",
        "evidenceSource",
        "duplexProfile",
        "duplexEventTypes",
        "outcomeRoute",
        "conversationLength",
        "styleProfile",
        "interactionMechanism",
        "stakes",
        "controlPhenomena",
        "modelAdmission",
    }
)
COMPACT_DERIVATION_FIELDS = frozenset({"semanticFingerprint", "candidateHash"})

PLANNER_METADATA_FIELDS = frozenset(
    {
        "endpoint",
        "model",
        "protocolAttempt",
        "responseHash",
        "responseFormat",
        "responseName",
        "responseSchemaHash",
        "reasoningDisabled",
        "accelerator",
        "cudaDevice",
        "finishReason",
        "schemaTransport",
        "usage",
    }
)
USAGE_METADATA_FIELDS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
    }
)


class FanoutError(CascadeError):
    """Base failure for compact fan-out."""


class RetryableModelOutput(FanoutError):
    """A model/protocol result that must be retried without host repair."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class ProtocolError(RetryableModelOutput):
    """Malformed or unavailable JSON-Schema model protocol."""


class AdmissionError(RetryableModelOutput):
    """Structurally valid JSON that fails exact structural admission."""


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
            raise FanoutError(f"{path}:{line_number}: expected a JSON object")
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


def _assert_immutable_checkpoint_file(path: Path) -> None:
    file_stat = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_mode & 0o222:
        raise FanoutError(f"immutable checkpoint must be a read-only regular file: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _immutable_bytes(path: Path, payload: bytes) -> None:
    """Atomically create an immutable object, never replacing existing bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            _assert_immutable_checkpoint_file(path)
            if path.read_bytes() != payload:
                raise FanoutError(f"immutable checkpoint collision at {path}")
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        _fsync_directory(path.parent)


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _atomic_bytes(
        path,
        "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8"),
    )


class AttemptJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        value = dict(record)
        value["auditRecordHash"] = content_hash(value)
        payload = (canonical_json(value) + "\n").encode("utf-8")
        with self._lock:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


def _schema_errors(value: Any, schema: Mapping[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(value),
        key=lambda error: tuple(str(component) for component in error.absolute_path),
    )
    sanitized: list[str] = []
    for error in errors:
        path = "$"
        for component in error.absolute_path:
            path += f"[{component}]" if isinstance(component, int) else f".{component}"
        sanitized.append(f"{path}:{error.validator}")
    return sanitized


def _string_list_schema(min_items: int, max_items: int) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": min_items,
        "maxItems": max_items,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 2},
    }


def _json_scalar_schema() -> dict[str, Any]:
    return {"type": ["string", "number", "integer", "boolean", "null"]}


def _rotated_values(values: Sequence[str], salt: str) -> list[str]:
    ordered = list(dict.fromkeys(values))
    if not ordered:
        return []
    offset = int(sha256(salt.encode("utf-8")).hexdigest()[:8], 16) % len(ordered)
    return ordered[offset:] + ordered[:offset]


def _duplex_event_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "eventType",
            "targetOrdinal",
            "offsetMs",
            "overlapMs",
            "cancelOutgoingAudio",
            "invalidateGeneration",
        ],
        "properties": {
            "eventType": {"enum": list(EVENT_TYPES)},
            "targetOrdinal": {"type": "integer", "minimum": 1, "maximum": 24},
            "offsetMs": {"type": "integer", "minimum": -2_000, "maximum": 30_000},
            "overlapMs": {"type": "integer", "minimum": 0, "maximum": 5_000},
            "cancelOutgoingAudio": {"type": "boolean"},
            "invalidateGeneration": {"type": "boolean"},
        },
    }


class SchemaModel(Protocol):
    def generate(
        self,
        *,
        name: str,
        schema: Mapping[str, Any],
        instructions: str,
        context: Mapping[str, Any],
        max_output_tokens: int,
    ) -> tuple[Any, Mapping[str, Any]]:
        ...


def _normalize_cuda_endpoint(value: str) -> str:
    try:
        return normalize_chat_completion_endpoints((value,))[0]
    except ValueError as error:
        raise FanoutError(str(error)) from error


def _sanitize_usage(value: Any, *, strict: bool) -> dict[str, int]:
    if value is None and not strict:
        return {}
    if not isinstance(value, Mapping):
        if strict:
            raise ProtocolError("malformed_usage", "usage must be an object")
        return {}
    if strict and set(value) - USAGE_METADATA_FIELDS:
        raise ProtocolError("malformed_usage", "usage contains undeclared fields")
    sanitized: dict[str, int] = {}
    for key in USAGE_METADATA_FIELDS:
        item = value.get(key)
        if item is None:
            continue
        if type(item) is not int or item < 0:
            if strict:
                raise ProtocolError("malformed_usage", "token counts must be non-negative integers")
            continue
        sanitized[key] = item
    return sanitized


def parse_three_endpoints(endpoints: str | Sequence[str]) -> tuple[str, ...]:
    try:
        return normalize_chat_completion_endpoints(endpoints)
    except ValueError as error:
        raise FanoutError(str(error)) from error


class ThreeEndpointJsonSchemaClient:
    """Strict-schema client for one to three local or OpenRouter lanes."""

    def __init__(
        self,
        endpoints: str | Sequence[str],
        model: str,
        api_key: str = "",
        timeout_seconds: int = 240,
        protocol_attempts: int = 6,
        retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
        retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.endpoints = parse_three_endpoints(endpoints)
        if not isinstance(model, str) or not model.strip():
            raise FanoutError("planner model is required")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= MAX_ENDPOINT_TIMEOUT_SECONDS:
            raise FanoutError(
                f"timeout_seconds must be in [1,{MAX_ENDPOINT_TIMEOUT_SECONDS}]"
            )
        if (
            type(protocol_attempts) is not int
            or not 2 <= protocol_attempts <= MAX_PROTOCOL_ATTEMPTS
        ):
            raise FanoutError(
                f"protocol_attempts must be in [2,{MAX_PROTOCOL_ATTEMPTS}]"
            )
        try:
            _, retry_base_seconds, retry_max_seconds = validate_retry_settings(
                protocol_attempts, retry_base_seconds, retry_max_seconds
            )
        except ValueError as error:
            raise FanoutError(str(error)) from error
        self.model = model.strip()
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.protocol_attempts = protocol_attempts
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self._sleep = sleep
        self._provenance_by_endpoint = {
            endpoint: (
                ("remote", None)
                if is_openrouter_endpoint(endpoint)
                else ("cuda", CUDA_DEVICES[index])
            )
            for index, endpoint in enumerate(self.endpoints)
        }
        self._lock = Lock()
        self._next = 0

    def _rotation(self) -> tuple[str, ...]:
        with self._lock:
            start = self._next
            self._next = (self._next + 1) % len(self.endpoints)
        endpoints = self.endpoints
        return endpoints[start:] + endpoints[:start]

    def generate(
        self,
        *,
        name: str,
        schema: Mapping[str, Any],
        instructions: str,
        context: Mapping[str, Any],
        max_output_tokens: int,
    ) -> tuple[Any, Mapping[str, Any]]:
        if (
            type(max_output_tokens) is not int
            or not 1 <= max_output_tokens <= MAX_SCHEMA_OUTPUT_TOKENS
        ):
            raise FanoutError(
                f"max_output_tokens must be in [1,{MAX_SCHEMA_OUTPUT_TOKENS}]"
            )
        rotation = self._rotation()
        failures: list[str] = []
        request_context: Mapping[str, Any] = context
        for attempt in range(1, self.protocol_attempts + 1):
            endpoint = rotation[(attempt - 1) % len(rotation)]
            projection = build_schema_transport_projection(endpoint, self.model, schema)
            payload = {
                "model": self.model,
                "stream": False,
                "reasoning": {"enabled": False},
                "temperature": 0.9,
                "max_tokens": max_output_tokens,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": canonical_json(request_context)},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "strict": True,
                        "schema": projection.transport_schema,
                    },
                },
            }
            body = canonical_json(payload).encode("utf-8")
            headers = {"content-type": "application/json"}
            if self.api_key:
                headers["authorization"] = f"Bearer {self.api_key}"
            request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    response_headers = getattr(response, "headers", None)
                    response_body = response.read(MAX_MODEL_RESPONSE_BYTES + 1)
                if len(response_body) > MAX_MODEL_RESPONSE_BYTES:
                    raise ValueError("response byte limit exceeded")
                envelope = json.loads(response_body)
                if not isinstance(envelope, Mapping):
                    raise TypeError("response envelope is not an object")
                provider_error = classify_provider_error(envelope)
                if provider_error is not None:
                    if not provider_error.retryable:
                        raise ProtocolError(
                            "provider_rejected",
                            provider_error.classification,
                        )
                    failures.append(
                        f"attempt {attempt} endpoint {endpoint}: "
                        f"{provider_error.classification}"
                    )
                    if attempt < self.protocol_attempts:
                        requested_delay = (
                            retry_after_seconds(response_headers)
                            if provider_error.code in {429, 503}
                            else None
                        )
                        self._sleep(
                            retry_delay_seconds(
                                attempt,
                                self.retry_base_seconds,
                                self.retry_max_seconds,
                                requested_delay,
                            )
                        )
                    continue
                choices = envelope["choices"]
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError("exactly one response choice is required")
                choice = choices[0]
                if not isinstance(choice, Mapping):
                    raise TypeError("response choice is not an object")
                finish_reason = choice.get("finish_reason")
                if finish_reason != "stop":
                    raise ValueError("finish_reason is not stop")
                message = choice["message"]
                if not isinstance(message, Mapping):
                    raise TypeError("response message is not an object")
                if any(
                    message.get(field) not in (None, "", [], {})
                    for field in ("reasoning", "reasoning_content", "analysis", "refusal")
                ):
                    raise ValueError("reasoning or refusal content was returned")
                content = message["content"]
                if not isinstance(content, str):
                    raise TypeError("message content is not a string")
                value = json.loads(content)
                errors = _schema_errors(value, schema)
                if errors:
                    canonical_errors = list(
                        Draft202012Validator(schema).iter_errors(value)
                    )
                    defect = canonical_validation_defect(canonical_errors)
                    if attempt < self.protocol_attempts:
                        request_context = canonical_retry_context(
                            context, defect, attempt + 1
                        )
                        failures.append(
                            f"attempt {attempt} endpoint {endpoint}: "
                            f"canonical_schema_rejected {defect}"
                        )
                        continue
                    raise ProtocolError("canonical_schema_exhausted", defect)
            except urllib.error.HTTPError as error:
                if error.code not in RETRIABLE_HTTP_STATUS:
                    raise ProtocolError(
                        "http_rejected", f"endpoint {endpoint}: HTTP {error.code}"
                    ) from error
                failures.append(f"attempt {attempt} endpoint {endpoint}: HTTP {error.code}")
                if attempt < self.protocol_attempts:
                    requested_delay = (
                        retry_after_seconds(error.headers)
                        if error.code in {429, 503}
                        else None
                    )
                    self._sleep(
                        retry_delay_seconds(
                            attempt,
                            self.retry_base_seconds,
                            self.retry_max_seconds,
                            requested_delay,
                        )
                    )
                continue
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                failures.append(f"attempt {attempt} endpoint {endpoint}: transport {type(error).__name__}")
                if attempt < self.protocol_attempts:
                    self._sleep(
                        retry_delay_seconds(
                            attempt,
                            self.retry_base_seconds,
                            self.retry_max_seconds,
                        )
                    )
                continue
            except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
                failures.append(
                    f"attempt {attempt} endpoint {endpoint}: malformed protocol "
                    f"{type(error).__name__}"
                )
                continue
            accelerator, cuda_device = self._provenance_by_endpoint[endpoint]
            return value, {
                "endpoint": endpoint,
                "model": self.model,
                "protocolAttempt": attempt,
                "responseHash": content_hash(value),
                "responseFormat": "json_schema_strict",
                "responseName": name,
                "responseSchemaHash": content_hash(schema),
                "reasoningDisabled": True,
                "accelerator": accelerator,
                "cudaDevice": cuda_device,
                "finishReason": finish_reason,
                "schemaTransport": projection.binding,
                "usage": _sanitize_usage(envelope.get("usage"), strict=False),
            }
        raise ProtocolError("protocol_exhausted", "; ".join(failures))


RoundRobinJsonSchemaPlanner = ThreeEndpointJsonSchemaClient


def _assert_no_target_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in TARGET_DIALOGUE_FIELDS:
                raise AdmissionError("target_field", f"forbidden target-bearing field {key}")
            _assert_no_target_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_target_fields(child)


def _validated_planner_metadata(
    metadata: Any,
    *,
    value: Any,
    schema: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ProtocolError("malformed_metadata", "planner metadata must be an object")
    _assert_no_target_fields(metadata)
    metadata_fields = set(metadata)
    if metadata_fields not in (
        PLANNER_METADATA_FIELDS,
        PLANNER_METADATA_FIELDS - {"schemaTransport"},
    ):
        raise ProtocolError("malformed_metadata", "planner metadata fields are not exact")
    try:
        endpoint = _normalize_cuda_endpoint(metadata["endpoint"])
    except FanoutError as error:
        raise ProtocolError("invalid_endpoint", "planner endpoint is not a safe HTTP lane") from error
    if not isinstance(metadata["model"], str) or not metadata["model"].strip():
        raise ProtocolError("missing_model", "planner model identity is required")
    protocol_attempt = metadata["protocolAttempt"]
    if (
        type(protocol_attempt) is not int
        or not 1 <= protocol_attempt <= MAX_PROTOCOL_ATTEMPTS
    ):
        raise ProtocolError("unbounded_protocol_attempt", "protocol attempt is outside the bound")
    if metadata["responseHash"] != content_hash(value):
        raise ProtocolError("response_hash_mismatch", "planner response hash is stale")
    if metadata["responseFormat"] != "json_schema_strict":
        raise ProtocolError("unconstrained_response", "strict JSON Schema was not attested")
    if metadata["responseName"] != name:
        raise ProtocolError("response_name_mismatch", "schema response name is stale")
    if metadata["responseSchemaHash"] != content_hash(schema):
        raise ProtocolError("response_schema_hash_mismatch", "response schema hash is stale")
    expected_schema_transport = build_schema_transport_projection(
        endpoint, metadata["model"].strip(), schema
    ).binding
    if (
        "schemaTransport" in metadata
        and metadata["schemaTransport"] != expected_schema_transport
    ):
        raise ProtocolError(
            "schema_transport_mismatch",
            "canonical and projected schema binding is stale",
        )
    if metadata["reasoningDisabled"] is not True:
        raise ProtocolError("reasoning_not_disabled", "reasoning-disabled inference is required")
    if is_openrouter_endpoint(endpoint):
        if metadata["accelerator"] != "remote" or metadata["cudaDevice"] is not None:
            raise ProtocolError("invalid_backend", "OpenRouter must use remote provenance")
    elif (
        metadata["accelerator"] != "cuda"
        or type(metadata["cudaDevice"]) is not int
        or metadata["cudaDevice"] not in CUDA_DEVICES
    ):
        raise ProtocolError("non_cuda_backend", "local CUDA device provenance is required")
    if metadata["finishReason"] != "stop":
        raise ProtocolError("incomplete_response", "finishReason must be stop")
    usage = _sanitize_usage(metadata["usage"], strict=True)
    return {
        "endpoint": endpoint,
        "model": metadata["model"].strip(),
        "protocolAttempt": protocol_attempt,
        "responseHash": metadata["responseHash"],
        "responseFormat": "json_schema_strict",
        "responseName": name,
        "responseSchemaHash": metadata["responseSchemaHash"],
        "reasoningDisabled": True,
        "accelerator": "cuda",
        "cudaDevice": metadata["cudaDevice"],
        "finishReason": "stop",
        "schemaTransport": expected_schema_transport,
        "usage": usage,
    }


def _allowed_sources(request: Mapping[str, Any]) -> tuple[str, ...]:
    requested = (request.get("requiredControlCoverage") or {}).get("stateSources") or []
    values = tuple(dict.fromkeys([*requested, "scenario_state", "state_reducer"]))
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise FanoutError("required control state sources must be non-empty strings")
    return values


def _causal_axes(request: Mapping[str, Any]) -> tuple[str, ...]:
    values = tuple(dict.fromkeys((request.get("semanticControl") or {}).get("requiredCausalAxes") or []))
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise FanoutError("semantic-control-v5 request lacks declared causal axes")
    return values


def _operators(topic: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    raw = topic.get("causalAffordances")
    if not isinstance(raw, list) or not raw:
        raise FanoutError(f"{topic.get('topicId')}: causalAffordances are required")
    operators: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise FanoutError(f"{topic.get('topicId')}: malformed causal affordance")
        operator = {
            "operatorId": item.get("operatorId"),
            "family": item.get("family"),
            "changedPath": item.get("changedPath"),
        }
        if any(not isinstance(value, str) or not value for value in operator.values()):
            raise FanoutError(f"{topic.get('topicId')}: malformed causal affordance")
        operators.append(operator)  # type: ignore[arg-type]
    if len({canonical_json(item) for item in operators}) != len(operators):
        raise FanoutError(f"{topic.get('topicId')}: duplicate causal affordances")
    return tuple(operators)


def validate_v5_inputs(
    request: dict[str, Any],
    topics: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    *,
    required_scenario_count: int | None = None,
    require_production_counts: bool = False,
) -> None:
    _assert_no_target_fields({"request": request, "topics": topics, "scenarios": scenarios})
    validate_request(request)
    if request.get("strategyVersion") != "semantic-control-v5":
        raise FanoutError("compact trajectory fan-out requires semantic-control-v5")
    if request_sibling_roles(request) != CAUSAL_SIBLING_ROLES:
        raise FanoutError("compact trajectory fan-out requires the exact four-role causal contract")
    coverage = request["coverageTarget"]
    expected_scenarios = coverage["candidateTopics"] * coverage["scenariosPerTopic"]
    if coverage.get("trajectorySeedsPerScenario") != LEAVES_PER_SCENARIO:
        raise FanoutError("trajectorySeedsPerScenario must be exactly ten")
    if len(topics) != coverage["candidateTopics"]:
        raise FanoutError("topic_cards.jsonl cardinality differs from the request")
    if len(scenarios) != expected_scenarios:
        raise FanoutError("scenario_contracts.jsonl cardinality differs from the request")
    if required_scenario_count is not None and len(scenarios) != required_scenario_count:
        raise FanoutError(f"exactly {required_scenario_count} scenario-contract.v2 inputs are required")
    primary_count, reserve_count = request_selection_counts(request)
    if require_production_counts:
        expected = (
            PRODUCTION_TOPIC_COUNT,
            PRODUCTION_SCENARIOS_PER_TOPIC,
            LEAVES_PER_SCENARIO,
            PRODUCTION_SCENARIO_COUNT,
            PRODUCTION_PRIMARY_COUNT,
            PRODUCTION_RESERVE_COUNT,
        )
        observed = (
            coverage["candidateTopics"],
            coverage["scenariosPerTopic"],
            coverage["trajectorySeedsPerScenario"],
            len(scenarios),
            primary_count,
            reserve_count,
        )
        if observed != expected:
            raise FanoutError(
                "production fan-out requires exactly 50 topics x 20 scenarios x 10 leaves, "
                "with 250 primary and 250 reserve"
            )
    topic_ids: set[str] = set()
    for topic in topics:
        validate_topic_card(topic, request["seedRevision"])
        topic_id = topic.get("topicId")
        if not isinstance(topic_id, str) or topic_id in topic_ids:
            raise FanoutError("topic IDs must be present and unique")
        topic_ids.add(topic_id)
        _operators(topic)
    scenario_ids: set[str] = set()
    per_topic: Counter[str] = Counter()
    for topic in topics:
        if set(topic) != TOPIC_CARD_V2_FIELDS:
            raise FanoutError(f"{topic.get('topicId')}: topic-card.v2 fields are not exact")
        for affordance in topic["causalAffordances"]:
            if not isinstance(affordance, Mapping) or set(affordance) != {
                "family", "operatorId", "changedPath",
            }:
                raise FanoutError(f"{topic.get('topicId')}: causal affordance fields are not exact")
    for scenario in scenarios:
        if scenario.get("schema") != "personaplex.scenario-contract.v2":
            raise FanoutError("all inputs must be scenario-contract.v2")
        if set(scenario) != SCENARIO_CONTRACT_V2_FIELDS:
            raise FanoutError(f"{scenario.get('scenarioId')}: scenario-contract.v2 fields are not exact")
        if any(
            not isinstance(participant, Mapping)
            or set(participant) != {"role", "knowledge"}
            for participant in scenario["participants"]
        ):
            raise FanoutError(f"{scenario.get('scenarioId')}: participant fields are not exact")
        if set(scenario["startingState"]) != {
            "knownFacts", "uncertainty", "policyConstraints",
        }:
            raise FanoutError(f"{scenario.get('scenarioId')}: startingState fields are not exact")
        validate_scenario_contract(scenario, topic_ids)
        scenario_id = scenario.get("scenarioId")
        if not isinstance(scenario_id, str) or scenario_id in scenario_ids:
            raise FanoutError("scenario IDs must be present and unique")
        scenario_ids.add(scenario_id)
        per_topic[scenario["topicId"]] += 1
    for topic_id in topic_ids:
        if per_topic[topic_id] != coverage["scenariosPerTopic"]:
            raise FanoutError(f"{topic_id}: scenario coverage differs from the request")


def candidate_lineage(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    scenario: Mapping[str, Any],
    ordinal: int,
) -> dict[str, Any]:
    if not 0 <= ordinal < LEAVES_PER_SCENARIO:
        raise FanoutError("candidate ordinal is outside [0,9]")
    identity_material = {
        "contract": COMPACT_CARD_SCHEMA,
        "requestId": request["requestId"],
        "requestHash": content_hash(request),
        "topicId": topic["topicId"],
        "topicHash": content_hash(topic),
        "scenarioId": scenario["scenarioId"],
        "scenarioHash": content_hash(scenario),
        "candidateOrdinal": ordinal,
    }
    digest = sha256(canonical_json(identity_material).encode("utf-8")).hexdigest()[:32]
    return {
        "requestId": request["requestId"],
        "topicId": topic["topicId"],
        "scenarioId": scenario["scenarioId"],
        "candidateOrdinal": ordinal,
        "candidateId": f"compact_{digest}",
        "trajectoryId": f"trajectory_{digest}",
    }


def candidate_lineages(
    request: Mapping[str, Any], topic: Mapping[str, Any], scenario: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        candidate_lineage(request, topic, scenario, ordinal)
        for ordinal in range(LEAVES_PER_SCENARIO)
    ]


def compact_lineage(binding: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: binding[key]
        for key in ("requestId", "topicId", "scenarioId", "candidateOrdinal")
    }


def compact_response_schema(
    request: Mapping[str, Any], topic: Mapping[str, Any], scenario: Mapping[str, Any]
) -> dict[str, Any]:
    operators = _operators(topic)
    sources = _allowed_sources(request)
    axes = _causal_axes(request)
    outcomes = tuple(dict.fromkeys(scenario.get("scenarioOutcomeSpace") or []))
    families = tuple(dict.fromkeys((request.get("causalGroupContract") or {}).get("interventionFamilies") or []))
    if not outcomes or any(not isinstance(value, str) or not value for value in outcomes):
        raise FanoutError(f"{scenario.get('scenarioId')}: scenarioOutcomeSpace is required")
    if not families:
        families = tuple(dict.fromkeys(item["family"] for item in operators))
    operator_families = {item["family"] for item in operators}
    if not operator_families.issubset(set(families)):
        raise FanoutError(f"{topic.get('topicId')}: operator family is outside the request contract")

    stakes = tuple(dict.fromkeys(topic.get("safeStakes") or []))
    phenomena = tuple(dict.fromkeys(scenario.get("requiredControlPhenomena") or []))
    if not stakes or not phenomena:
        raise FanoutError("compact descriptors require declared stakes and control phenomena")
    semantic_properties: dict[str, Any] = {
        "a": {"enum": _rotated_values(axes, f"{scenario['scenarioId']}|axis")},
        "o": {
            "enum": _rotated_values(
                [item["operatorId"] for item in operators],
                f"{scenario['scenarioId']}|operator",
            )
        },
        "q": {
            "enum": _rotated_values(
                STATE_TRANSITIONS, f"{scenario['scenarioId']}|transition"
            )
        },
        "h": {
            "type": "string",
            "minLength": 3,
            "maxLength": 16,
            "pattern": "^[a-z0-9_:-]{3,16}$",
        },
        "u": {
            "enum": _rotated_values(
                tuple(POSTURE_TRANSITIONS), f"{scenario['scenarioId']}|posture"
            )
        },
        "e": {"enum": _rotated_values(sources, f"{scenario['scenarioId']}|evidence")},
        "d": {
            "enum": _rotated_values(
                tuple(DUPLEX_PROFILES), f"{scenario['scenarioId']}|duplex"
            )
        },
        "r": {"enum": _rotated_values(outcomes, f"{scenario['scenarioId']}|outcome")},
        "z": {
            "enum": _rotated_values(
                tuple(LENGTH_PIVOT_PROFILES), f"{scenario['scenarioId']}|length"
            )
        },
        "y": {
            "enum": _rotated_values(
                STYLE_PROFILES, f"{scenario['scenarioId']}|style"
            )
        },
        "i": {
            "enum": _rotated_values(
                INTERACTION_MECHANISMS, f"{scenario['scenarioId']}|interaction"
            )
        },
        "k": {"enum": _rotated_values(stakes, f"{scenario['scenarioId']}|stakes")},
        "c": {
            "enum": _rotated_values(phenomena, f"{scenario['scenarioId']}|phenomenon")
        },
        "m": {"const": "admit"},
    }
    descriptor_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(semantic_properties),
        "properties": semantic_properties,
    }
    candidate_properties: dict[str, Any] = {}
    candidate_ids: list[str] = []
    for lineage in candidate_lineages(request, topic, scenario):
        properties = {"x": {"$ref": "#/$defs/compactDescriptor"}}
        candidate_id = lineage["candidateId"]
        candidate_ids.append(candidate_id)
        candidate_properties[candidate_id] = {
            "type": "object",
            "additionalProperties": False,
            "required": list(properties),
            "properties": properties,
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$defs": {"compactDescriptor": descriptor_schema},
        "type": "object",
        "additionalProperties": False,
        "required": ["candidates"],
        "properties": {
            "candidates": {
                "type": "object",
                "additionalProperties": False,
                "required": candidate_ids,
                "properties": candidate_properties,
            }
        },
    }


def _maximal_schema_value(
    schema: Mapping[str, Any], root_schema: Mapping[str, Any]
) -> Any:
    reference = schema.get("$ref")
    if reference is not None:
        if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
            raise FanoutError("Stage A budget measurement supports local $defs references only")
        definition = root_schema.get("$defs", {}).get(reference.removeprefix("#/$defs/"))
        if not isinstance(definition, Mapping):
            raise FanoutError("Stage A response schema contains an unresolved definition")
        return _maximal_schema_value(definition, root_schema)
    if "const" in schema:
        return deepcopy(schema["const"])
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return deepcopy(
            max(enum, key=lambda item: len(canonical_json(item).encode("utf-8")))
        )
    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise FanoutError("Stage A response objects must declare required properties")
        return {
            key: _maximal_schema_value(properties[key], root_schema)
            for key in required
        }
    if schema_type == "string":
        maximum = schema.get("maxLength")
        if type(maximum) is not int or maximum < 0:
            raise FanoutError("Stage A strings must have a finite maxLength")
        return "z" * maximum
    raise FanoutError("Stage A response schema contains an unbounded value")


def measure_compact_response_bound(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure the exact maximum canonical Stage A response before inference."""

    schema = compact_response_schema(request, topic, scenario)
    witness = _maximal_schema_value(schema, schema)
    serialized_bytes = len(canonical_json(witness).encode("utf-8"))
    if serialized_bytes > MAX_COMPACT_RESPONSE_BYTES:
        raise FanoutError(
            f"{scenario.get('scenarioId')}: Stage A worst-case serialized response "
            f"is {serialized_bytes} bytes, above compact ceiling {MAX_COMPACT_RESPONSE_BYTES}"
        )
    return {
        "worstCaseSerializedBytes": serialized_bytes,
        "maxSerializedBytes": MAX_COMPACT_RESPONSE_BYTES,
        "plannerOutputTokenLimit": MAX_SCHEMA_OUTPUT_TOKENS,
        "requestedOutputTokenLimit": STAGE_A_MAX_OUTPUT_TOKENS,
        "reservedOutputTokens": STAGE_A_OUTPUT_TOKEN_HEADROOM,
        "worstCaseResponseHash": content_hash(witness),
    }


def normalize_compact_response(
    response: Mapping[str, Any],
    lineages: Sequence[Mapping[str, Any]],
    operators: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    values = response.get("candidates")
    if not isinstance(values, Mapping):
        raise ProtocolError("candidate_map", "candidates must be the required identity-keyed object")
    try:
        normalized = []
        for lineage in lineages:
            wire = values[lineage["candidateId"]]
            descriptor = wire["x"]
            duplex_profile = descriptor["d"]
            pivot_from, pivot_to = STATE_TRANSITION_VALUES[descriptor["q"]]
            posture_from, posture_to = POSTURE_TRANSITIONS[descriptor["u"]]
            length_band, target_turns, pivot_ordinal = LENGTH_PIVOT_PROFILES[
                descriptor["z"]
            ]
            operator = next(
                item for item in operators if item["operatorId"] == descriptor["o"]
            )
            normalized.append({
                "schema": COMPACT_CARD_SCHEMA,
                "candidateId": lineage["candidateId"],
                "trajectoryId": lineage["trajectoryId"],
                "candidateOrdinal": lineage["candidateOrdinal"],
                "lineage": compact_lineage(lineage),
                "causalAxis": descriptor["a"],
                "interventionFamily": operator["family"],
                "causalOperator": {
                    "operatorId": descriptor["o"],
                    "family": operator["family"],
                    "changedPath": operator["changedPath"],
                },
                "typedPivot": {
                    "field": operator["changedPath"],
                    "from": pivot_from,
                    "to": pivot_to,
                },
                "counterfactualPivotOrdinal": pivot_ordinal,
                "stateTransition": {
                    "kind": descriptor["q"],
                    "fromState": pivot_from,
                    "toState": pivot_to,
                    "tag": descriptor["h"],
                },
                "postureTransition": {
                    "from": posture_from,
                    "to": posture_to,
                },
                "evidenceSource": descriptor["e"],
                "duplexProfile": duplex_profile,
                "duplexEventTypes": list(DUPLEX_PROFILES[duplex_profile]),
                "outcomeRoute": descriptor["r"],
                "conversationLength": {
                    "lengthBand": length_band,
                    "targetTurns": target_turns,
                },
                "styleProfile": descriptor["y"],
                "interactionMechanism": descriptor["i"],
                "stakes": descriptor["k"],
                "controlPhenomena": [descriptor["c"]],
                "modelAdmission": descriptor["m"],
            })
        return normalized
    except (KeyError, TypeError, ValueError) as error:
        raise ProtocolError("candidate_map", "candidate identity map is incomplete") from error


def compact_wire_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    raw = _strip_candidate_derivations(candidate)
    posture_code = next(
        code
        for code, values in POSTURE_TRANSITIONS.items()
        if values
        == (raw["postureTransition"]["from"], raw["postureTransition"]["to"])
    )
    length_code = next(
        code
        for code, values in LENGTH_PIVOT_PROFILES.items()
        if values
        == (
            raw["conversationLength"]["lengthBand"],
            raw["conversationLength"]["targetTurns"],
            raw["counterfactualPivotOrdinal"],
        )
    )
    return {
        "x": {
            "a": raw["causalAxis"],
            "o": raw["causalOperator"]["operatorId"],
            "q": raw["stateTransition"]["kind"],
            "h": raw["stateTransition"]["tag"],
            "u": posture_code,
            "e": raw["evidenceSource"],
            "d": raw["duplexProfile"],
            "r": raw["outcomeRoute"],
            "z": length_code,
            "y": raw["styleProfile"],
            "i": raw["interactionMechanism"],
            "k": raw["stakes"],
            "c": raw["controlPhenomena"][0],
            "m": raw["modelAdmission"],
        },
    }


def _length_band(target_turns: int) -> str:
    if target_turns <= 8:
        return "short"
    if target_turns <= 12:
        return "medium"
    return "long"


def _strip_candidate_derivations(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in candidate.items()
        if key not in {"semanticFingerprint", "candidateHash"}
    }


def _candidate_scenario_id(candidate: Mapping[str, Any]) -> str:
    lineage = candidate.get("lineage")
    if not isinstance(lineage, Mapping) or not isinstance(lineage.get("scenarioId"), str):
        raise FanoutError("compact candidate lacks bound lineage.scenarioId")
    return lineage["scenarioId"]


def _semantic_payload(candidate: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "schema",
        "candidateId",
        "trajectoryId",
        "candidateOrdinal",
        "lineage",
        "semanticFingerprint",
        "candidateHash",
    }
    return {key: deepcopy(value) for key, value in candidate.items() if key not in excluded}


def _typed_signature(candidate: Mapping[str, Any]) -> str:
    return content_hash(
        {
            "causalAxis": candidate["causalAxis"],
            "interventionFamily": candidate["interventionFamily"],
            "causalOperator": candidate["causalOperator"],
            "typedPivot": candidate["typedPivot"],
            "stateTransition": {
                "kind": candidate["stateTransition"]["kind"],
                "fromState": candidate["stateTransition"]["fromState"],
                "toState": candidate["stateTransition"]["toState"],
            },
            "postureTransition": candidate["postureTransition"],
            "evidenceSource": candidate["evidenceSource"],
            "duplexProfile": candidate["duplexProfile"],
            "duplexEventTypes": sorted(candidate["duplexEventTypes"]),
            "outcomeRoute": candidate["outcomeRoute"],
            "conversationLength": candidate["conversationLength"]["lengthBand"],
            "style": candidate["styleProfile"],
            "interactionMechanism": candidate["interactionMechanism"],
            "stakes": candidate["stakes"],
        }
    )


def _validate_duplex_events(events: Sequence[Mapping[str, Any]], agent_targets: int) -> None:
    if any(event["targetOrdinal"] > agent_targets for event in events):
        raise AdmissionError("duplex_target", "event targets a nonexistent agent turn")
    barges = [event for event in events if event["eventType"] == "barge_in"]
    cancellations = [event for event in events if event["eventType"] == "cancelled_generation"]
    recoveries = [
        event
        for event in events
        if event["eventType"] in {"repair_after_barge_in", "recovery"}
    ]
    if not barges or not cancellations or not recoveries:
        raise AdmissionError(
            "duplex_shape",
            "barge_in, cancelled_generation, and later recovery are all required",
        )
    for barge in barges:
        cancellation = next(
            (
                event
                for event in cancellations
                if event["targetOrdinal"] == barge["targetOrdinal"]
                and event["offsetMs"] >= barge["offsetMs"]
                and event["cancelOutgoingAudio"] is True
                and event["invalidateGeneration"] is True
            ),
            None,
        )
        recovery = next(
            (event for event in recoveries if event["targetOrdinal"] > barge["targetOrdinal"]),
            None,
        )
        if cancellation is None or recovery is None:
            raise AdmissionError("duplex_order", "barge-in lacks bound cancellation and recovery")


def _validate_candidate(
    candidate: Mapping[str, Any],
    operators: Sequence[Mapping[str, str]],
) -> None:
    fields = set(candidate)
    if not COMPACT_CANDIDATE_FIELDS.issubset(fields):
        raise AdmissionError("candidate_shape", "compact candidate fields are incomplete")
    if fields - COMPACT_CANDIDATE_FIELDS - COMPACT_DERIVATION_FIELDS:
        raise AdmissionError("candidate_shape", "compact candidate contains undeclared fields")
    if bool(fields & COMPACT_DERIVATION_FIELDS) and not COMPACT_DERIVATION_FIELDS.issubset(fields):
        raise AdmissionError("candidate_shape", "compact candidate derivation fields are incomplete")
    raw = _strip_candidate_derivations(candidate)
    _assert_no_target_fields(raw)
    selected_operator = candidate["causalOperator"]
    if canonical_json(selected_operator) not in {canonical_json(item) for item in operators}:
        raise AdmissionError("operator_binding", "causal operator is not one declared affordance")
    if candidate["interventionFamily"] != selected_operator["family"]:
        raise AdmissionError("operator_binding", "intervention family disagrees with causal operator")
    if candidate["typedPivot"]["field"] != selected_operator["changedPath"]:
        raise AdmissionError("operator_binding", "typed pivot path disagrees with causal operator")
    if canonical_json(candidate["typedPivot"]["from"]) == canonical_json(candidate["typedPivot"]["to"]):
        raise AdmissionError("pivot_noop", "typed pivot from and to values must differ")
    if candidate["stateTransition"]["fromState"] == candidate["stateTransition"]["toState"]:
        raise AdmissionError("state_noop", "state transition must change")
    if candidate["postureTransition"]["from"] == candidate["postureTransition"]["to"]:
        raise AdmissionError("posture_noop", "posture transition must change")
    length = candidate["conversationLength"]
    if length["lengthBand"] != _length_band(length["targetTurns"]):
        raise AdmissionError("length_band", "declared length band disagrees with targetTurns")
    agent_targets = length["targetTurns"] // 2
    pivot = candidate["counterfactualPivotOrdinal"]
    if not 2 <= pivot < agent_targets:
        raise AdmissionError("pivot_position", "pivot must precede at least one recovery target")
    expected_types = set(DUPLEX_PROFILES[candidate["duplexProfile"]])
    if set(candidate["duplexEventTypes"]) != expected_types:
        raise AdmissionError("duplex_declaration", "event types disagree with duplex profile")
    expected_semantic_hash = content_hash(_semantic_payload(candidate))
    if "semanticFingerprint" in candidate and candidate["semanticFingerprint"] != expected_semantic_hash:
        raise FanoutError(f"{candidate.get('candidateId')}: stale semanticFingerprint")
    immutable = {key: deepcopy(value) for key, value in candidate.items() if key != "candidateHash"}
    if "candidateHash" in candidate and candidate["candidateHash"] != content_hash(immutable):
        raise FanoutError(f"{candidate.get('candidateId')}: stale candidateHash")


def _validate_candidate_set(
    candidates: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    scenario: Mapping[str, Any],
) -> None:
    if len(candidates) != LEAVES_PER_SCENARIO:
        raise AdmissionError("candidate_cardinality", "exactly ten compact leaves are required")
    ordered = sorted(candidates, key=lambda item: item.get("candidateOrdinal", -1))
    schema = compact_response_schema(request, topic, scenario)
    errors = _schema_errors(
        {
            "candidates": {
                item["candidateId"]: compact_wire_candidate(item)
                for item in ordered
            }
        },
        schema,
    )
    if errors:
        raise AdmissionError("response_schema", "; ".join(errors[:5]))
    operators = _operators(topic)
    for ordinal, candidate in enumerate(ordered):
        _validate_candidate(
            candidate,
            operators,
        )
    candidate_ids = [item["candidateId"] for item in ordered]
    trajectory_ids = [item["trajectoryId"] for item in ordered]
    fingerprints = [content_hash(_semantic_payload(item)) for item in ordered]
    signatures = [_typed_signature(item) for item in ordered]
    if len(set(candidate_ids)) != 10 or len(set(trajectory_ids)) != 10:
        raise AdmissionError("identity_collapse", "candidate and trajectory IDs must be unique")
    if len(set(fingerprints)) != 10:
        raise AdmissionError("semantic_exact_duplicate", "compact semantic payloads must be unique")
    if len(set(signatures)) != 10:
        raise AdmissionError("typed_shape_collapse", "all ten declared typed shapes must be unique")


def _derive_candidates(raw_candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    admitted: list[dict[str, Any]] = []
    for raw in raw_candidates:
        candidate = deepcopy(dict(raw))
        candidate["semanticFingerprint"] = content_hash(_semantic_payload(candidate))
        candidate["candidateHash"] = content_hash(candidate)
        admitted.append(candidate)
    return admitted


def _validate_production_typed_coverage(
    request: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> None:
    if len(candidates) != PRODUCTION_CANDIDATE_COUNT:
        return
    required = {
        "causalAxis": set(_causal_axes(request)),
        "interventionFamily": set(
            (request.get("causalGroupContract") or {}).get("interventionFamilies") or []
        ),
        "evidenceSource": set(
            (request.get("requiredControlCoverage") or {}).get("stateSources") or []
        ),
        "duplexProfile": set(DUPLEX_PROFILES),
        "conversationLength": set(LENGTH_BANDS),
        "style": set(STYLE_PROFILES),
    }
    observed = {
        "causalAxis": {item["causalAxis"] for item in candidates},
        "interventionFamily": {item["interventionFamily"] for item in candidates},
        "evidenceSource": {item["evidenceSource"] for item in candidates},
        "duplexProfile": {item["duplexProfile"] for item in candidates},
        "conversationLength": {
            item["conversationLength"]["lengthBand"] for item in candidates
        },
        "style": {item["styleProfile"] for item in candidates},
    }
    for axis, required_values in required.items():
        missing = required_values - observed[axis]
        if missing:
            raise FanoutError(
                f"production compact corpus lacks declared {axis} values: {sorted(missing)}"
            )


def _input_binding(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    scenario: Mapping[str, Any],
    response_schema: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "stage": "compact-trajectory-fanout-v5-stage-a",
        "requestHash": content_hash(request),
        "topicHash": content_hash(topic),
        "scenarioHash": content_hash(scenario),
        "responseSchemaHash": content_hash(response_schema),
    }
    return {**value, "inputBindingHash": content_hash(value)}


class ScenarioCheckpointStore:
    def __init__(
        self,
        directory: Path,
        expected: Mapping[str, Mapping[str, Any]],
    ) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.expected = expected
        self._lock = Lock()
        self._records: dict[str, dict[str, Any]] = {}
        self._fingerprints: dict[str, str] = {}
        for path in sorted(self.directory.glob("*.json")):
            _assert_immutable_checkpoint_file(path)
            wrapper = read_json(path)
            self._register(wrapper, source_path=path, write=False)

    def _register(
        self,
        wrapper: dict[str, Any],
        *,
        source_path: Path | None = None,
        write: bool,
    ) -> bool:
        if wrapper.get("schema") != SCENARIO_CHECKPOINT_SCHEMA:
            raise FanoutError("unsupported Stage A checkpoint schema")
        checkpoint_hash = wrapper.get("checkpointHash")
        payload = {key: deepcopy(value) for key, value in wrapper.items() if key != "checkpointHash"}
        if checkpoint_hash != content_hash(payload):
            raise FanoutError("Stage A checkpoint hash is stale")
        if source_path is not None and source_path.name != f"{checkpoint_hash[7:]}.json":
            raise FanoutError(f"Stage A checkpoint path is not content-addressed: {source_path}")
        scenario_id = wrapper.get("scenarioId")
        expected = self.expected.get(str(scenario_id))
        if expected is None:
            raise FanoutError(f"checkpoint references unknown scenario {scenario_id}")
        if wrapper.get("inputBinding") != expected["inputBinding"]:
            raise FanoutError(f"{scenario_id}: checkpoint input binding changed")
        prior = self._records.get(str(scenario_id))
        if prior is not None:
            if prior != wrapper:
                raise FanoutError(f"{scenario_id}: conflicting immutable scenario checkpoints")
            return False
        candidates = wrapper.get("candidates")
        if not isinstance(candidates, list):
            raise FanoutError(f"{scenario_id}: checkpoint candidates are malformed")
        _validate_candidate_set(
            candidates,
            expected["request"],
            expected["topic"],
            expected["scenario"],
        )
        if wrapper.get("candidateSetHash") != content_hash(candidates):
            raise FanoutError(f"{scenario_id}: candidateSetHash is stale")
        model_response = {
            "candidates": {
                item["candidateId"]: compact_wire_candidate(item)
                for item in sorted(candidates, key=lambda value: value["candidateOrdinal"])
            }
        }
        _validated_planner_metadata(
            wrapper.get("planner"),
            value=model_response,
            schema=expected["schema"],
            name=COMPACT_RESPONSE_NAME,
        )
        local_fingerprints = [item["semanticFingerprint"] for item in candidates]
        for fingerprint in local_fingerprints:
            owner = self._fingerprints.get(fingerprint)
            if owner is not None:
                raise AdmissionError(
                    "global_exact_duplicate",
                    f"{scenario_id} duplicates compact semantics already admitted for {owner}",
                )
        if write:
            path = self.directory / f"{checkpoint_hash[7:]}.json"
            _immutable_bytes(
                path,
                (json.dumps(wrapper, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        self._records[str(scenario_id)] = deepcopy(wrapper)
        for fingerprint in local_fingerprints:
            self._fingerprints[fingerprint] = str(scenario_id)
        return True

    def get(self, scenario_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._records.get(scenario_id)
            return deepcopy(value) if value is not None else None

    def admit(
        self,
        scenario_id: str,
        input_binding: Mapping[str, Any],
        candidates: list[dict[str, Any]],
        planner_metadata: Mapping[str, Any],
    ) -> bool:
        payload = {
            "schema": SCENARIO_CHECKPOINT_SCHEMA,
            "scenarioId": scenario_id,
            "inputBinding": deepcopy(dict(input_binding)),
            "candidateSetHash": content_hash(candidates),
            "candidateIds": [item["candidateId"] for item in candidates],
            "candidates": deepcopy(candidates),
            "planner": deepcopy(dict(planner_metadata)),
        }
        wrapper = {**payload, "checkpointHash": content_hash(payload)}
        with self._lock:
            return self._register(wrapper, write=True)

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            candidates = [
                deepcopy(candidate)
                for scenario_id in sorted(self._records)
                for candidate in self._records[scenario_id]["candidates"]
            ]
        return sorted(
            candidates,
            key=lambda item: (_candidate_scenario_id(item), item["candidateOrdinal"]),
        )

    def receipts(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(self._records[key]) for key in sorted(self._records)]


class CandidateCheckpointStore:
    """Immutable candidate identities derived from authenticated ten-way receipts."""

    def __init__(
        self,
        directory: Path,
        expected: Mapping[str, Mapping[str, Any]],
        receipts: Sequence[Mapping[str, Any]],
    ) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.expected = expected
        self._lock = Lock()
        self._receipts: dict[str, dict[str, Any]] = {}
        self._records: dict[str, dict[str, Any]] = {}
        for receipt in receipts:
            self._bind_receipt(dict(receipt))
        for path in sorted(self.directory.glob("*.json")):
            _assert_immutable_checkpoint_file(path)
            self._register(read_json(path), source_path=path, write=False)

    def _bind_receipt(self, receipt: dict[str, Any]) -> None:
        scenario_id = receipt.get("scenarioId")
        if not isinstance(scenario_id, str):
            raise FanoutError("Stage A receipt lacks scenarioId")
        prior = self._receipts.get(scenario_id)
        if prior is not None and prior != receipt:
            raise FanoutError(f"{scenario_id}: conflicting immutable Stage A receipts")
        self._receipts[scenario_id] = deepcopy(receipt)

    def bind_receipt(self, receipt: Mapping[str, Any]) -> None:
        with self._lock:
            self._bind_receipt(dict(receipt))

    def _register(
        self,
        wrapper: dict[str, Any],
        *,
        source_path: Path | None = None,
        write: bool,
    ) -> bool:
        required_fields = {
            "schema",
            "candidateId",
            "trajectoryId",
            "scenarioId",
            "candidateHash",
            "candidateSetHash",
            "inputBinding",
            "responseCheckpointHash",
            "plannerResponseHash",
            "candidate",
            "checkpointHash",
        }
        if set(wrapper) != required_fields:
            raise FanoutError("Stage A candidate checkpoint fields are not exact")
        if wrapper.get("schema") != CANDIDATE_CHECKPOINT_SCHEMA:
            raise FanoutError("unsupported Stage A candidate checkpoint schema")
        checkpoint_hash = wrapper.get("checkpointHash")
        payload = {
            key: deepcopy(value) for key, value in wrapper.items() if key != "checkpointHash"
        }
        if checkpoint_hash != content_hash(payload):
            raise FanoutError("Stage A candidate checkpoint hash is stale")
        if source_path is not None and source_path.name != f"{checkpoint_hash[7:]}.json":
            raise FanoutError(
                f"Stage A candidate checkpoint path is not content-addressed: {source_path}"
            )
        candidate_id = wrapper.get("candidateId")
        expected = self.expected.get(str(candidate_id))
        if expected is None:
            raise FanoutError(f"candidate checkpoint references unknown identity {candidate_id}")
        scenario_id = expected["scenarioId"]
        receipt = self._receipts.get(scenario_id)
        if receipt is None:
            raise FanoutError(f"{candidate_id}: candidate checkpoint lacks its Stage A receipt")
        candidates = receipt.get("candidates")
        receipt_candidate = next(
            (
                candidate
                for candidate in candidates
                if isinstance(candidate, Mapping)
                and candidate.get("candidateId") == candidate_id
            ),
            None,
        ) if isinstance(candidates, list) else None
        candidate = wrapper.get("candidate")
        if not isinstance(candidate, dict) or candidate != receipt_candidate:
            raise FanoutError(f"{candidate_id}: candidate checkpoint differs from its receipt")
        if wrapper.get("scenarioId") != scenario_id:
            raise FanoutError(f"{candidate_id}: candidate scenario binding is stale")
        if wrapper.get("trajectoryId") != candidate.get("trajectoryId"):
            raise FanoutError(f"{candidate_id}: candidate trajectory binding is stale")
        if wrapper.get("candidateHash") != candidate.get("candidateHash"):
            raise FanoutError(f"{candidate_id}: candidate hash binding is stale")
        if wrapper.get("candidateSetHash") != receipt.get("candidateSetHash"):
            raise FanoutError(f"{candidate_id}: candidate set binding is stale")
        if wrapper.get("inputBinding") != expected["inputBinding"]:
            raise FanoutError(f"{candidate_id}: candidate input binding is stale")
        if wrapper.get("responseCheckpointHash") != receipt.get("checkpointHash"):
            raise FanoutError(f"{candidate_id}: candidate receipt binding is stale")
        if wrapper.get("plannerResponseHash") != receipt.get("planner", {}).get("responseHash"):
            raise FanoutError(f"{candidate_id}: candidate model-response binding is stale")
        _assert_no_target_fields(wrapper)
        prior = self._records.get(str(candidate_id))
        if prior is not None:
            if prior != wrapper:
                raise FanoutError(f"{candidate_id}: conflicting immutable candidate checkpoints")
            return False
        if write:
            path = self.directory / f"{checkpoint_hash[7:]}.json"
            _immutable_bytes(
                path,
                (json.dumps(wrapper, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        self._records[str(candidate_id)] = deepcopy(wrapper)
        return True

    def admit(
        self,
        receipt: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> bool:
        self.bind_receipt(receipt)
        candidate_value = deepcopy(dict(candidate))
        candidate_id = candidate_value["candidateId"]
        expected = self.expected.get(candidate_id)
        if expected is None:
            raise FanoutError(f"candidate receipt contains unknown identity {candidate_id}")
        payload = {
            "schema": CANDIDATE_CHECKPOINT_SCHEMA,
            "candidateId": candidate_id,
            "trajectoryId": candidate_value["trajectoryId"],
            "scenarioId": expected["scenarioId"],
            "candidateHash": candidate_value["candidateHash"],
            "candidateSetHash": receipt["candidateSetHash"],
            "inputBinding": deepcopy(expected["inputBinding"]),
            "responseCheckpointHash": receipt["checkpointHash"],
            "plannerResponseHash": receipt["planner"]["responseHash"],
            "candidate": candidate_value,
        }
        wrapper = {**payload, "checkpointHash": content_hash(payload)}
        with self._lock:
            return self._register(wrapper, write=True)

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = [deepcopy(wrapper["candidate"]) for wrapper in self._records.values()]
        return sorted(
            rows,
            key=lambda item: (_candidate_scenario_id(item), item["candidateOrdinal"]),
        )


def _invoke_model(
    planner: SchemaModel,
    *,
    name: str,
    schema: Mapping[str, Any],
    instructions: str,
    context: Mapping[str, Any],
    max_output_tokens: int,
) -> tuple[Any, dict[str, Any]]:
    _assert_no_target_fields(context)
    result = planner.generate(
        name=name,
        schema=schema,
        instructions=instructions,
        context=context,
        max_output_tokens=max_output_tokens,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise ProtocolError("malformed_return", "planner must return (value, metadata)")
    value, metadata = result
    _assert_no_target_fields(value)
    errors = _schema_errors(value, schema)
    if errors:
        raise ProtocolError("schema_mismatch", "; ".join(errors[:5]))
    return value, _validated_planner_metadata(
        metadata,
        value=value,
        schema=schema,
        name=name,
    )


def _candidate_prompt_context(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    scenario: Mapping[str, Any],
    lineages: Sequence[Mapping[str, Any]],
    output_budget: Mapping[str, Any],
    prior_failure_code: str | None,
) -> dict[str, Any]:
    return {
        "task": "Generate ten authentic compact target-free trajectory leaf cards in one response.",
        "requestId": request["requestId"],
        "topicCard": topic,
        "scenarioContract": scenario,
        "candidateBindings": [deepcopy(dict(item)) for item in lineages],
        "outputBudget": deepcopy(dict(output_budget)),
        "declaredDomains": {
            "causalAxes": list(_causal_axes(request)),
            "causalOperators": [dict(item) for item in _operators(topic)],
            "evidenceSources": list(_allowed_sources(request)),
            "outcomeRoutes": list(dict.fromkeys(scenario["scenarioOutcomeSpace"])),
            "lengthBands": list(LENGTH_BANDS),
            "styleProfiles": list(STYLE_PROFILES),
        },
        "priorFailureCode": prior_failure_code,
        "requirements": [
            "Choose every semantic field authentically; identity constants are the only host assignments.",
            "Return all ten cards. Never return a partial repair batch.",
            "Use declarative state and behavior only, with no caller or agent dialogue, transcript, target wording, target hashes, or canonical responses.",
            "Every card must contain a real barge-in, bound cancellation/invalidation, and later recovery plan.",
            "Use materially distinct typed shapes and semantic state plans across all ten cards.",
            "Give all ten cards distinct joint typed shapes across the declared axes and profiles; changing only the short state tag is not sufficient.",
            "Set modelAdmission only after checking target freedom, semantic distinctness, and causal coherence.",
            "Keep every compact string within the schema bounds; the measured response ceiling is mandatory.",
        ],
    }


def generate_compact_candidates(
    *,
    request: dict[str, Any],
    topics: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    output_root: Path,
    planner: SchemaModel,
    max_workers: int = 3,
    max_attempts: int = 4,
    required_scenario_count: int | None = None,
    require_production_counts: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_v5_inputs(
        request,
        topics,
        scenarios,
        required_scenario_count=required_scenario_count,
        require_production_counts=require_production_counts,
    )
    if type(max_workers) is not int or not 1 <= max_workers <= len(CUDA_DEVICES):
        raise FanoutError("Stage A max_workers must be in [1,3]")
    if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_STAGE_ATTEMPTS:
        raise FanoutError(f"Stage A max_attempts must be in [1,{MAX_STAGE_ATTEMPTS}]")
    output_root.mkdir(parents=True, exist_ok=True)
    topic_by_id = {topic["topicId"]: topic for topic in topics}
    expected: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        topic = topic_by_id[scenario["topicId"]]
        schema = compact_response_schema(request, topic, scenario)
        expected[scenario["scenarioId"]] = {
            "request": request,
            "topic": topic,
            "scenario": scenario,
            "schema": schema,
            "inputBinding": _input_binding(request, topic, scenario, schema),
            "outputBudget": measure_compact_response_bound(request, topic, scenario),
        }
    response_store = ScenarioCheckpointStore(
        output_root / CHECKPOINT_ROOT / "stage_a_scenarios", expected
    )
    expected_candidates = {
        lineage["candidateId"]: {
            "scenarioId": scenario["scenarioId"],
            "inputBinding": expected[scenario["scenarioId"]]["inputBinding"],
        }
        for scenario in scenarios
        for lineage in candidate_lineages(
            request, topic_by_id[scenario["topicId"]], scenario
        )
    }
    candidate_store = CandidateCheckpointStore(
        output_root / CHECKPOINT_ROOT / "stage_a_candidates",
        expected_candidates,
        response_store.receipts(),
    )
    for receipt in response_store.receipts():
        for candidate in receipt["candidates"]:
            candidate_store.admit(receipt, candidate)
    journal = AttemptJournal(output_root / CANDIDATE_AUDIT_FILENAME)

    def generate_one(scenario: dict[str, Any]) -> None:
        scenario_id = scenario["scenarioId"]
        if response_store.get(scenario_id) is not None:
            return
        bound = expected[scenario_id]
        topic = bound["topic"]
        schema = bound["schema"]
        lineages = candidate_lineages(request, topic, scenario)
        prior_failure_code: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response, metadata = _invoke_model(
                    planner,
                    name=COMPACT_RESPONSE_NAME,
                    schema=schema,
                    instructions=(
                        "Generate compact semantic-control plans, never dialogue. Return only the strict "
                        "JSON-Schema object. Reasoning is disabled by the caller; do not emit analysis."
                    ),
                    context=_candidate_prompt_context(
                        request,
                        topic,
                        scenario,
                        lineages,
                        bound["outputBudget"],
                        prior_failure_code,
                    ),
                    max_output_tokens=STAGE_A_MAX_OUTPUT_TOKENS,
                )
                response_bytes = len(canonical_json(response).encode("utf-8"))
                if response_bytes > bound["outputBudget"]["worstCaseSerializedBytes"]:
                    raise ProtocolError(
                        "response_size_bound",
                        "strict Stage A response exceeded its measured schema maximum",
                    )
                reported_tokens = max(
                    metadata["usage"].get("completion_tokens", 0),
                    metadata["usage"].get("output_tokens", 0),
                )
                if reported_tokens > STAGE_A_MAX_OUTPUT_TOKENS:
                    raise ProtocolError(
                        "response_token_bound",
                        "Stage A completion exceeded the requested output-token limit",
                    )
                raw_candidates = normalize_compact_response(
                    response, lineages, _operators(topic)
                )
                candidates = _derive_candidates(raw_candidates)
                _validate_candidate_set(candidates, request, topic, scenario)
                response_store.admit(
                    scenario_id,
                    bound["inputBinding"],
                    candidates,
                    metadata,
                )
                receipt = response_store.get(scenario_id)
                if receipt is None:
                    raise FanoutError(f"{scenario_id}: Stage A receipt was not persisted")
                for candidate in candidates:
                    candidate_store.admit(receipt, candidate)
                journal.append(
                    {
                        "schema": "personaplex.compact-trajectory-attempt.v2",
                        "scenarioId": scenario_id,
                        "attempt": attempt,
                        "accepted": True,
                        "failureCode": None,
                        "responseHash": metadata.get("responseHash"),
                    }
                )
                return
            except RetryableModelOutput as error:
                prior_failure_code = error.code
                journal.append(
                    {
                        "schema": "personaplex.compact-trajectory-attempt.v2",
                        "scenarioId": scenario_id,
                        "attempt": attempt,
                        "accepted": False,
                        "failureCode": error.code,
                    }
                )
        raise FanoutError(
            f"{scenario_id}: no complete ten-card response admitted after {max_attempts} attempts "
            f"(last failure: {prior_failure_code})"
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_one, scenario) for scenario in scenarios]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    rows = candidate_store.rows()
    expected_count = len(scenarios) * LEAVES_PER_SCENARIO
    expected_ids = {
        lineage["candidateId"]
        for scenario in scenarios
        for lineage in candidate_lineages(request, topic_by_id[scenario["topicId"]], scenario)
    }
    if len(rows) != expected_count or {row["candidateId"] for row in rows} != expected_ids:
        raise FanoutError("Stage A checkpoints do not exactly cover every bound candidate identity")
    per_scenario = Counter(_candidate_scenario_id(row) for row in rows)
    if any(per_scenario[scenario["scenarioId"]] != 10 for scenario in scenarios):
        raise FanoutError("Stage A must contain exactly ten candidates per scenario")
    if len({row["semanticFingerprint"] for row in rows}) != expected_count:
        raise FanoutError("Stage A contains exact duplicate semantic payloads")
    _validate_production_typed_coverage(request, rows)
    candidates_path = output_root / CANDIDATES_FILENAME
    write_jsonl_atomic(candidates_path, rows)
    manifest_payload = {
        "schema": "personaplex.compact-trajectory-candidate-manifest.v2",
        "requestHash": content_hash(request),
        "topicSetHash": content_hash(topics),
        "scenarioSetHash": content_hash(scenarios),
        "scenarioCount": len(scenarios),
        "leavesPerScenario": LEAVES_PER_SCENARIO,
        "candidateCount": len(rows),
        "candidateSetHash": content_hash(rows),
        "checkpointMode": "immutable-content-addressed-per-candidate",
        "candidateCheckpointCount": len(rows),
        "responseReceiptMode": "immutable-content-addressed-per-scenario",
        "causalSiblingRoles": list(CAUSAL_SIBLING_ROLES),
        "outputBudget": {
            "maxWorstCaseSerializedBytes": max(
                bound["outputBudget"]["worstCaseSerializedBytes"]
                for bound in expected.values()
            ),
            "maxSerializedBytes": MAX_COMPACT_RESPONSE_BYTES,
            "plannerOutputTokenLimit": MAX_SCHEMA_OUTPUT_TOKENS,
            "requestedOutputTokenLimit": STAGE_A_MAX_OUTPUT_TOKENS,
            "reservedOutputTokens": STAGE_A_OUTPUT_TOKEN_HEADROOM,
        },
        "files": {
            CANDIDATES_FILENAME: hash_file(candidates_path),
            CANDIDATE_AUDIT_FILENAME: hash_file(output_root / CANDIDATE_AUDIT_FILENAME),
        },
    }
    manifest = {**manifest_payload, "manifestHash": content_hash(manifest_payload)}
    write_json_atomic(output_root / CANDIDATE_MANIFEST_FILENAME, manifest)
    return rows, manifest


def _validate_complete_candidate_corpus(
    request: dict[str, Any],
    topics: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    require_production_counts: bool = True,
) -> None:
    validate_v5_inputs(
        request,
        topics,
        scenarios,
        require_production_counts=require_production_counts,
    )
    expected_count = len(scenarios) * LEAVES_PER_SCENARIO
    if len(candidates) != expected_count:
        raise FanoutError("selection requires the complete ten-leaf candidate corpus")
    if len({item.get("candidateId") for item in candidates}) != expected_count:
        raise FanoutError("candidate IDs must be globally unique")
    if len({item.get("trajectoryId") for item in candidates}) != expected_count:
        raise FanoutError("trajectory IDs must be globally unique")
    topic_by_id = {item["topicId"]: item for item in topics}
    scenario_by_id = {item["scenarioId"]: item for item in scenarios}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        scenario = scenario_by_id.get(_candidate_scenario_id(candidate))
        if scenario is None:
            raise FanoutError("candidate references an unknown scenario")
        if candidate.get("lineage", {}).get("topicId") != scenario["topicId"]:
            raise FanoutError("candidate topic/scenario lineage is inconsistent")
        grouped[scenario["scenarioId"]].append(candidate)
    for scenario in scenarios:
        rows = grouped.get(scenario["scenarioId"], [])
        _validate_candidate_set(
            rows,
            request,
            topic_by_id[scenario["topicId"]],
            scenario,
        )
    if len({item["semanticFingerprint"] for item in candidates}) != expected_count:
        raise FanoutError("candidate corpus contains exact duplicate semantic payloads")
    _validate_production_typed_coverage(request, candidates)


def candidate_balance_dimensions(candidate: Mapping[str, Any]) -> dict[str, list[str]]:
    """Project declared typed metadata only; natural-language fields are excluded."""

    return {
        "causalAxis": [str(candidate["causalAxis"])],
        "interventionFamily": [str(candidate["interventionFamily"])],
        "postureTransition": [canonical_json(candidate["postureTransition"])],
        "evidenceSource": [str(candidate["evidenceSource"])],
        "duplexEventType": sorted(str(value) for value in candidate["duplexEventTypes"]),
        "outcomeRoute": [str(candidate["outcomeRoute"])],
        "conversationLength": [str(candidate["conversationLength"]["lengthBand"])],
        "style": [str(candidate["styleProfile"])],
    }


def _stable_identity_rank(request: Mapping[str, Any], identity: str) -> str:
    material = {
        "seedRevision": request["seedRevision"],
        "requestHash": content_hash(request),
        "identity": identity,
    }
    return sha256(canonical_json(material).encode("utf-8")).hexdigest()


def _topic_quotas(
    request: Mapping[str, Any], topic_ids: Sequence[str], target: int
) -> dict[str, int]:
    if target < len(topic_ids):
        raise FanoutError("each selection tier must cover every topic")
    base, remainder = divmod(target, len(topic_ids))
    order = sorted(topic_ids, key=lambda topic_id: _stable_identity_rank(request, topic_id))
    return {
        topic_id: base + (1 if index < remainder else 0)
        for index, topic_id in enumerate(order)
    }


def _selection_hash(
    request: Mapping[str, Any],
    group_id: str,
    tier: str,
    candidate: Mapping[str, Any],
    dimensions: Mapping[str, Any],
) -> str:
    return content_hash(
        {
            "requestHash": content_hash(request),
            "groupId": group_id,
            "selectionTier": tier,
            "candidateId": candidate["candidateId"],
            "candidateHash": candidate["candidateHash"],
            "trajectoryId": candidate["trajectoryId"],
            "scenarioId": _candidate_scenario_id(candidate),
            "dimensions": dimensions,
        }
    )


def select_compact_candidates(
    *,
    request: dict[str, Any],
    topics: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _validate_complete_candidate_corpus(request, topics, scenarios, candidates)
    primary_target, reserve_target = request_selection_counts(request)
    if primary_target + reserve_target > len(scenarios):
        raise FanoutError(
            "group-disjoint selection needs at least one distinct scenario per selected group"
        )
    policy = request.get("selectionPolicy") or {}
    if policy.get("allLeavesEligible") is not True:
        raise FanoutError("selectionPolicy.allLeavesEligible must be true")
    requested_axes = tuple(policy.get("balanceAxes") or BALANCE_AXES)
    if set(requested_axes) != set(BALANCE_AXES):
        raise FanoutError("selection must use the complete declared typed balance-axis contract")
    topic_by_id = {item["topicId"]: item for item in topics}
    scenario_by_id = {item["scenarioId"]: item for item in scenarios}
    candidates_by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    dimensions_by_id: dict[str, dict[str, list[str]]] = {}
    for candidate in candidates:
        topic_id = scenario_by_id[_candidate_scenario_id(candidate)]["topicId"]
        candidates_by_topic[topic_id].append(candidate)
        dimensions_by_id[candidate["candidateId"]] = candidate_balance_dimensions(candidate)

    selected_scenarios: set[str] = set()
    all_selected: list[dict[str, Any]] = []
    total_target = primary_target + reserve_target
    width = max(4, len(str(total_target)))

    def choose_tier(tier: str, target: int) -> list[dict[str, Any]]:
        quotas = _topic_quotas(request, sorted(topic_by_id), target)
        counts: Counter[tuple[str, str]] = Counter()
        tier_rows: list[dict[str, Any]] = []
        remaining = dict(quotas)
        topic_order = sorted(topic_by_id, key=lambda item: _stable_identity_rank(request, item))

        def score(candidate: Mapping[str, Any]) -> tuple[int, int, str]:
            dimensions = dimensions_by_id[candidate["candidateId"]]
            costs = [
                sum(counts[(axis, value)] for value in dimensions[axis])
                for axis in BALANCE_AXES
            ]
            return (
                max(costs),
                sum(costs),
                _stable_identity_rank(request, str(candidate["candidateId"])),
            )

        while any(value > 0 for value in remaining.values()):
            progressed = False
            for topic_id in topic_order:
                if remaining[topic_id] <= 0:
                    continue
                pool = [
                    candidate
                    for candidate in candidates_by_topic[topic_id]
                    if _candidate_scenario_id(candidate) not in selected_scenarios
                ]
                if not pool:
                    raise FanoutError(
                        f"{topic_id}: exhausted distinct scenario groups before satisfying {tier} quota"
                    )
                chosen = min(pool, key=score)
                dimensions = dimensions_by_id[chosen["candidateId"]]
                cost_by_axis = {
                    axis: sum(counts[(axis, value)] for value in dimensions[axis])
                    for axis in BALANCE_AXES
                }
                ordinal = len(all_selected) + len(tier_rows) + 1
                tier_ordinal = len(tier_rows) + 1
                group_id = f"cascade-{request['requestId']}-{ordinal:0{width}d}"
                rank = _stable_identity_rank(request, chosen["candidateId"])
                row = {
                    "schema": "personaplex.selected-trajectory.v1",
                    "groupId": group_id,
                    "topicId": topic_id,
                    "scenarioId": _candidate_scenario_id(chosen),
                    "trajectoryId": chosen["trajectoryId"],
                    "sourceSeedId": topic_by_id[topic_id].get("sourceSeedId"),
                    "selectionTier": tier,
                    "selectionOrdinal": ordinal,
                    "tierOrdinal": tier_ordinal,
                    "selectionSeedRevision": request["seedRevision"],
                    "balanceDimensions": dimensions,
                    "selectionRationale": {
                        "algorithm": "typed-balanced-group-disjoint-v2",
                        "candidatePoolSize": len(candidates),
                        "eligibleAtDecision": len(pool),
                        "coverageGroup": {
                            "kind": "scenario",
                            "scenarioId": _candidate_scenario_id(chosen),
                        },
                        "balanceCostByAxis": cost_by_axis,
                        "maxBalanceCost": max(cost_by_axis.values()),
                        "aggregateBalanceCost": sum(cost_by_axis.values()),
                        "deterministicIdentityRank": rank,
                        "naturalLanguageScoring": False,
                    },
                }
                row["selectionHash"] = _selection_hash(
                    request, group_id, tier, chosen, dimensions
                )
                tier_rows.append(row)
                selected_scenarios.add(_candidate_scenario_id(chosen))
                for axis, values in dimensions.items():
                    for value in values:
                        counts[(axis, value)] += 1
                remaining[topic_id] -= 1
                progressed = True
            if not progressed:
                raise FanoutError(f"selection made no progress for {tier}")
        return tier_rows

    primary = choose_tier("primary", primary_target)
    all_selected.extend(primary)
    reserve = choose_tier("reserve", reserve_target)
    all_selected.extend(reserve)
    if len(primary) != primary_target or len(reserve) != reserve_target:
        raise FanoutError("selection did not satisfy exact primary/reserve cardinality")
    primary_ids = {row["trajectoryId"] for row in primary}
    reserve_ids = {row["trajectoryId"] for row in reserve}
    primary_scenarios = {row["scenarioId"] for row in primary}
    reserve_scenarios = {row["scenarioId"] for row in reserve}
    if primary_ids & reserve_ids or primary_scenarios & reserve_scenarios:
        raise FanoutError("primary and reserve coverage groups must be disjoint")
    if len({row["groupId"] for row in all_selected}) != len(all_selected):
        raise FanoutError("selection group IDs must be unique")

    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_root / PRIMARY_FILENAME, primary)
    write_jsonl_atomic(output_root / RESERVE_FILENAME, reserve)
    # The downstream compiler consumes active primary groups from this name.
    write_jsonl_atomic(output_root / SELECTED_FILENAME, primary)
    candidate_by_trajectory = {item["trajectoryId"]: item for item in candidates}
    audit_rows = [
        {
            "schema": "personaplex.compact-trajectory-selection-audit.v2",
            "groupId": row["groupId"],
            "selectionTier": row["selectionTier"],
            "scenarioId": row["scenarioId"],
            "trajectoryId": row["trajectoryId"],
            "candidateId": candidate_by_trajectory[row["trajectoryId"]]["candidateId"],
            "candidateHash": candidate_by_trajectory[row["trajectoryId"]]["candidateHash"],
            "selectionHash": row["selectionHash"],
            "balanceDimensions": row["balanceDimensions"],
        }
        for row in all_selected
    ]
    write_jsonl_atomic(output_root / SELECTION_AUDIT_FILENAME, audit_rows)
    manifest_payload = {
        "schema": "personaplex.compact-trajectory-selection-manifest.v2",
        "requestHash": content_hash(request),
        "candidateSetHash": content_hash(candidates),
        "algorithm": "typed-balanced-group-disjoint-v2",
        "naturalLanguageScoring": False,
        "causalSiblingRoles": list(CAUSAL_SIBLING_ROLES),
        "coverageGroup": "scenario",
        "primaryCount": len(primary),
        "reserveCount": len(reserve),
        "primaryScenarioSetHash": content_hash(sorted(primary_scenarios)),
        "reserveScenarioSetHash": content_hash(sorted(reserve_scenarios)),
        "selectedCandidateSetHash": content_hash(sorted(primary_ids | reserve_ids)),
        "files": {
            name: hash_file(output_root / name)
            for name in (
                PRIMARY_FILENAME,
                RESERVE_FILENAME,
                SELECTED_FILENAME,
                SELECTION_AUDIT_FILENAME,
            )
        },
    }
    manifest = {**manifest_payload, "manifestHash": content_hash(manifest_payload)}
    write_json_atomic(output_root / SELECTION_MANIFEST_FILENAME, manifest)
    return primary, reserve, manifest


def _trajectory_constants(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "personaplex.trajectory-seed.v2",
        "trajectoryId": candidate["trajectoryId"],
        "scenarioId": _candidate_scenario_id(candidate),
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


def full_trajectory_response_schema(candidate: Mapping[str, Any]) -> dict[str, Any]:
    constants = _trajectory_constants(candidate)
    target_turns = candidate["conversationLength"]["targetTurns"]
    agent_targets = target_turns // 2
    pivot = constants["counterfactualPivotOrdinal"]
    typed_pivot = constants["typedPivot"]
    state_properties = {
        "targetOrdinal": {"type": "integer", "minimum": 1, "maximum": agent_targets},
        "phase": {"type": "string", "minLength": 2},
        "availableBeforeTarget": {"const": True},
        "controlRevision": {"type": "integer", "minimum": 1},
        "knownFacts": _string_list_schema(1, 10),
        "uncertainty": _string_list_schema(1, 8),
        "policyConstraints": _string_list_schema(1, 8),
        "commitments": _string_list_schema(1, 8),
        "callerPosture": {
            "enum": [
                candidate["postureTransition"]["from"],
                candidate["postureTransition"]["to"],
            ]
        },
        "nextGoal": {"type": "string", "minLength": 4},
        "evidence": {
            "type": "object",
            "additionalProperties": False,
            "required": ["source", "status", "facts"],
            "properties": {
                "source": {"const": candidate["evidenceSource"]},
                "status": {"type": "string", "minLength": 2},
                "facts": _string_list_schema(1, 8),
            },
        },
        "toolResult": {
            "type": "object",
            "additionalProperties": False,
            "required": ["source", "status", "facts"],
            "properties": {
                "source": {"type": "string", "minLength": 2},
                "status": {"type": "string", "minLength": 2},
                "facts": _string_list_schema(1, 8),
            },
        },
        "causalState": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "candidateHash", "operatorId", "changedPath", "from", "to", "activeValue",
            ],
            "properties": {
                "candidateHash": {"const": candidate["candidateHash"]},
                "operatorId": {"const": candidate["causalOperator"]["operatorId"]},
                "changedPath": {"const": typed_pivot["field"]},
                "from": {"const": typed_pivot["from"]},
                "to": {"const": typed_pivot["to"]},
                "activeValue": {"enum": [typed_pivot["from"], typed_pivot["to"]]},
            },
        },
        "revisionReason": {"type": "string", "minLength": 4},
    }
    state_item = {
        "type": "object",
        "additionalProperties": False,
        "required": list(state_properties),
        "properties": state_properties,
    }
    schedule_properties = {
        "controlRevision": {"type": "integer", "minimum": 1},
        "targetOrdinal": {"type": "integer", "minimum": 1, "maximum": agent_targets},
        "availableBeforeTarget": {"const": True},
        "source": {"const": candidate["evidenceSource"]},
    }
    schedule_item = {
        "type": "object",
        "additionalProperties": False,
        "required": list(schedule_properties),
        "properties": schedule_properties,
    }
    properties: dict[str, Any] = {
        key: {"const": deepcopy(value)} for key, value in constants.items()
    }
    properties.update(
        {
            "conversationLength": {
                "type": "object",
                "additionalProperties": False,
                "required": ["targetTurns", "min", "max"],
                "properties": {
                    "targetTurns": {"const": target_turns},
                    "min": {"type": "integer", "minimum": 4, "maximum": target_turns},
                    "max": {"type": "integer", "minimum": target_turns, "maximum": 24},
                },
            },
            "pace": {"type": "string", "minLength": 3},
            "openingStyle": {"type": "string", "minLength": 3},
            "closingStyle": {"type": "string", "minLength": 3},
            "voicePairPolicy": {"const": "distinct_approved_references"},
            "interactionArc": _string_list_schema(3, 8),
            "duplexEvents": {
                "type": "array", "minItems": 3, "maxItems": 10,
                "items": _duplex_event_schema(),
            },
            "postureArc": {
                "const": [
                    candidate["postureTransition"]["from"],
                    candidate["postureTransition"]["to"],
                ]
            },
        }
    )
    properties["semanticStateArc"] = {
        "type": "array",
        "minItems": agent_targets,
        "maxItems": agent_targets,
        "items": state_item,
    }
    properties["controlRevisionSchedule"] = {
        "type": "array",
        "minItems": agent_targets,
        "maxItems": agent_targets,
        "items": schedule_item,
    }
    trajectory_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["premiseState", "trajectory"],
        "properties": {
            "premiseState": {
                "type": "object",
                "additionalProperties": False,
                "required": ["situation", "knownFacts", "uncertainty", "policyConstraints"],
                "properties": {
                    "situation": {"type": "string", "minLength": 24},
                    "knownFacts": _string_list_schema(1, 8),
                    "uncertainty": _string_list_schema(1, 8),
                    "policyConstraints": _string_list_schema(1, 8),
                },
            },
            "trajectory": trajectory_schema,
        },
    }


def validate_expanded_trajectory(
    trajectory: dict[str, Any],
    candidate: Mapping[str, Any],
    known_scenarios: set[str],
) -> None:
    trajectory_schema = full_trajectory_response_schema(candidate)["properties"]["trajectory"]
    errors = _schema_errors(trajectory, trajectory_schema)
    if errors:
        raise AdmissionError("full_schema", "; ".join(errors[:6]))
    _assert_no_target_fields(trajectory)
    validate_trajectory_seed(trajectory, known_scenarios, require_typed=True)
    agent_targets = trajectory["conversationLength"]["targetTurns"] // 2
    expected_ordinals = list(range(1, agent_targets + 1))
    states = trajectory["semanticStateArc"]
    schedule = trajectory["controlRevisionSchedule"]
    if [item["targetOrdinal"] for item in states] != expected_ordinals:
        raise AdmissionError("state_ordinals", "semanticStateArc must cover targets in order")
    if [item["targetOrdinal"] for item in schedule] != expected_ordinals:
        raise AdmissionError("schedule_ordinals", "controlRevisionSchedule must cover targets in order")
    revisions = [item["controlRevision"] for item in schedule]
    if any(right <= left for left, right in zip(revisions, revisions[1:])):
        raise AdmissionError("revision_order", "control revisions must increase strictly")
    if [item["controlRevision"] for item in states] != revisions:
        raise AdmissionError("revision_binding", "state and schedule revisions disagree")
    if any(
        state["evidence"]["source"] != revision["source"]
        for state, revision in zip(states, schedule)
    ):
        raise AdmissionError("source_binding", "state and schedule evidence sources disagree")
    if any(
        state["evidence"]["source"] != candidate["evidenceSource"]
        for state in states
    ):
        raise AdmissionError("source_binding", "expanded evidence source changed from compact card")
    pivot = candidate["counterfactualPivotOrdinal"]
    for state in states:
        expected_active = (
            candidate["typedPivot"]["from"]
            if state["targetOrdinal"] < pivot
            else candidate["typedPivot"]["to"]
        )
        if state["causalState"]["activeValue"] != expected_active:
            raise AdmissionError("pivot_binding", "causal activeValue changes outside the pivot")
        expected_posture = (
            candidate["postureTransition"]["from"]
            if state["targetOrdinal"] < pivot
            else candidate["postureTransition"]["to"]
        )
        if state["callerPosture"] != expected_posture:
            raise AdmissionError("posture_binding", "caller posture changes outside the pivot")
    if {event["eventType"] for event in trajectory["duplexEvents"]} != set(
        candidate["duplexEventTypes"]
    ):
        raise AdmissionError("duplex_profile", "expanded events disagree with compact profile")
    _validate_duplex_events(trajectory["duplexEvents"], agent_targets)


def _validate_selection_rows(
    request: Mapping[str, Any],
    primary: Sequence[Mapping[str, Any]],
    reserve: Sequence[Mapping[str, Any]],
    candidate_by_trajectory: Mapping[str, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    primary_target, reserve_target = request_selection_counts(dict(request))
    if len(primary) != primary_target or len(reserve) != reserve_target:
        raise FanoutError("Stage C requires exact primary and reserve cardinality")
    rows = list(primary) + list(reserve)
    _assert_no_target_fields(rows)
    if len({row.get("trajectoryId") for row in rows}) != len(rows):
        raise FanoutError("primary and reserve trajectory IDs overlap")
    if len({row.get("groupId") for row in rows}) != len(rows):
        raise FanoutError("primary and reserve group IDs overlap")
    if {row.get("scenarioId") for row in primary} & {row.get("scenarioId") for row in reserve}:
        raise FanoutError("primary and reserve scenario coverage groups overlap")
    width = max(4, len(str(len(rows))))
    selection_ordinal = 0
    for expected_tier, tier_rows in (("primary", primary), ("reserve", reserve)):
        for tier_ordinal, row in enumerate(tier_rows, start=1):
            selection_ordinal += 1
            if row.get("schema") != "personaplex.selected-trajectory.v1":
                raise FanoutError("selection row schema is malformed")
            if row.get("selectionTier") != expected_tier:
                raise FanoutError(f"{expected_tier} rows must declare the {expected_tier} tier")
            if row.get("selectionOrdinal") != selection_ordinal:
                raise FanoutError("selection ordinals must be exact and ordered")
            if row.get("tierOrdinal") != tier_ordinal:
                raise FanoutError("selection tier ordinals must be exact and ordered")
            expected_group_id = (
                f"cascade-{request['requestId']}-{selection_ordinal:0{width}d}"
            )
            if row.get("groupId") != expected_group_id:
                raise FanoutError("selection groupId is not bound to its exact ordinal")
            if row.get("selectionSeedRevision") != request["seedRevision"]:
                raise FanoutError("selection seed revision is stale")
    for row in rows:
        candidate = candidate_by_trajectory.get(str(row.get("trajectoryId")))
        if candidate is None:
            raise FanoutError("selection references a non-candidate trajectory")
        tier = row.get("selectionTier")
        if tier not in {"primary", "reserve"}:
            raise FanoutError("selection tier is malformed")
        if row.get("scenarioId") != _candidate_scenario_id(candidate):
            raise FanoutError("selection scenario binding disagrees with compact candidate")
        if row.get("topicId") != candidate["lineage"]["topicId"]:
            raise FanoutError("selection topic binding disagrees with compact candidate")
        expected_dimensions = candidate_balance_dimensions(candidate)
        if row.get("balanceDimensions") != expected_dimensions:
            raise FanoutError("selection balanceDimensions disagree with compact candidate")
        rationale = row.get("selectionRationale")
        if (
            not isinstance(rationale, Mapping)
            or rationale.get("algorithm") != "typed-balanced-group-disjoint-v2"
            or rationale.get("naturalLanguageScoring") is not False
            or rationale.get("coverageGroup")
            != {"kind": "scenario", "scenarioId": _candidate_scenario_id(candidate)}
        ):
            raise FanoutError("selection rationale is not typed and scenario-bound")
        expected_hash = _selection_hash(
            request,
            row["groupId"],
            tier,
            candidate,
            expected_dimensions,
        )
        if row.get("selectionHash") != expected_hash:
            raise FanoutError("selectionHash is stale")
    return rows


class ExpansionCheckpointStore:
    def __init__(
        self,
        directory: Path,
        request: Mapping[str, Any],
        selected_rows: Sequence[Mapping[str, Any]],
        candidates: Mapping[str, Mapping[str, Any]],
        known_scenarios: set[str],
    ) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.request = request
        self.selected = {row["trajectoryId"]: row for row in selected_rows}
        self.candidates = candidates
        self.known_scenarios = known_scenarios
        self._records: dict[str, dict[str, Any]] = {}
        self._semantic_hashes: dict[str, str] = {}
        self._lock = Lock()
        for path in sorted(self.directory.glob("*.json")):
            _assert_immutable_checkpoint_file(path)
            self._register(read_json(path), source_path=path, write=False)

    def _register(
        self,
        wrapper: dict[str, Any],
        *,
        source_path: Path | None = None,
        write: bool,
    ) -> bool:
        if wrapper.get("schema") != EXPANSION_CHECKPOINT_SCHEMA:
            raise FanoutError("unsupported Stage C checkpoint schema")
        checkpoint_hash = wrapper.get("checkpointHash")
        payload = {key: deepcopy(value) for key, value in wrapper.items() if key != "checkpointHash"}
        if checkpoint_hash != content_hash(payload):
            raise FanoutError("Stage C checkpoint hash is stale")
        if source_path is not None and source_path.name != f"{checkpoint_hash[7:]}.json":
            raise FanoutError(f"Stage C checkpoint path is not content-addressed: {source_path}")
        trajectory_id = wrapper.get("trajectoryId")
        selection = self.selected.get(str(trajectory_id))
        candidate = self.candidates.get(str(trajectory_id))
        if selection is None or candidate is None:
            raise FanoutError("expansion checkpoint is not in the current selected-only set")
        if wrapper.get("candidateHash") != candidate["candidateHash"]:
            raise FanoutError(f"{trajectory_id}: compact candidate changed after expansion")
        if wrapper.get("selectionHash") != selection["selectionHash"]:
            raise FanoutError(f"{trajectory_id}: selection changed after expansion")
        trajectory = wrapper.get("trajectory")
        premise_state = wrapper.get("premiseState")
        if not isinstance(trajectory, dict) or not isinstance(premise_state, dict):
            raise FanoutError(f"{trajectory_id}: checkpoint expansion is malformed")
        response_errors = _schema_errors(
            {"premiseState": premise_state, "trajectory": trajectory},
            full_trajectory_response_schema(candidate),
        )
        if response_errors:
            raise FanoutError(f"{trajectory_id}: checkpoint expansion schema is stale")
        model_response = {"premiseState": premise_state, "trajectory": trajectory}
        _assert_no_target_fields(model_response)
        _validated_planner_metadata(
            wrapper.get("planner"),
            value=model_response,
            schema=full_trajectory_response_schema(candidate),
            name=EXPANSION_RESPONSE_NAME,
        )
        if wrapper.get("trajectoryHash") != content_hash(trajectory):
            raise FanoutError(f"{trajectory_id}: trajectoryHash is stale")
        semantic_hash = content_hash(
            {
                "premiseState": premise_state,
                "semanticStateArc": trajectory.get("semanticStateArc"),
                "controlRevisionSchedule": trajectory.get("controlRevisionSchedule"),
            }
        )
        if wrapper.get("expansionSemanticHash") != semantic_hash:
            raise FanoutError(f"{trajectory_id}: expansionSemanticHash is stale")
        prior_owner = self._semantic_hashes.get(semantic_hash)
        if prior_owner is not None and prior_owner != trajectory_id:
            raise AdmissionError(
                "expansion_exact_duplicate",
                f"{trajectory_id} duplicates the semantic expansion for {prior_owner}",
            )
        validate_expanded_trajectory(trajectory, candidate, self.known_scenarios)
        prior = self._records.get(str(trajectory_id))
        if prior is not None:
            if prior != wrapper:
                raise FanoutError(f"{trajectory_id}: conflicting immutable expansion checkpoints")
            return False
        if write:
            path = self.directory / f"{checkpoint_hash[7:]}.json"
            _immutable_bytes(
                path,
                (json.dumps(wrapper, indent=2, sort_keys=True) + "\n").encode("utf-8"),
            )
        self._records[str(trajectory_id)] = deepcopy(wrapper)
        self._semantic_hashes[semantic_hash] = str(trajectory_id)
        return True

    def get(self, trajectory_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._records.get(trajectory_id)
            return deepcopy(value) if value is not None else None

    def admit(
        self,
        candidate: Mapping[str, Any],
        selection: Mapping[str, Any],
        premise_state: dict[str, Any],
        trajectory: dict[str, Any],
        planner_metadata: Mapping[str, Any],
    ) -> bool:
        semantic_hash = content_hash(
            {
                "premiseState": premise_state,
                "semanticStateArc": trajectory["semanticStateArc"],
                "controlRevisionSchedule": trajectory["controlRevisionSchedule"],
            }
        )
        payload = {
            "schema": EXPANSION_CHECKPOINT_SCHEMA,
            "trajectoryId": candidate["trajectoryId"],
            "candidateId": candidate["candidateId"],
            "candidateHash": candidate["candidateHash"],
            "selectionHash": selection["selectionHash"],
            "trajectoryHash": content_hash(trajectory),
            "expansionSemanticHash": semantic_hash,
            "premiseState": deepcopy(premise_state),
            "trajectory": deepcopy(trajectory),
            "planner": deepcopy(dict(planner_metadata)),
        }
        wrapper = {**payload, "checkpointHash": content_hash(payload)}
        with self._lock:
            return self._register(wrapper, write=True)

    def rows(self) -> list[dict[str, Any]]:
        with self._lock:
            return [deepcopy(self._records[key]) for key in sorted(self._records)]


def _expansion_prompt_context(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    scenario: Mapping[str, Any],
    candidate: Mapping[str, Any],
    selection: Mapping[str, Any],
    prior_failure_code: str | None,
) -> dict[str, Any]:
    return {
        "task": "Authentically expand one selected compact card into trajectory-seed.v2.",
        "requestId": request["requestId"],
        "topicCard": topic,
        "scenarioContract": scenario,
        "compactCandidate": candidate,
        "selectionBinding": {
            "groupId": selection["groupId"],
            "selectionTier": selection["selectionTier"],
            "selectionHash": selection["selectionHash"],
        },
        "priorFailureCode": prior_failure_code,
        "requirements": [
            "Preserve every schema constant exactly; they bind the selected compact card.",
            "Generate all semantic-state and revision content authentically, without host templates or repair.",
            "Every control state must exist strictly before its target and use a strictly increasing revision.",
            "Use declarative facts, uncertainty, policy, commitments, posture, evidence, and goals only.",
            "Never emit dialogue, transcript, target wording, target hashes, or a canonical response.",
            "Change causalState.activeValue at the declared pivot and preserve real cancellation/recovery structure.",
        ],
    }


def expand_selected_candidates(
    *,
    request: dict[str, Any],
    topics: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    primary: list[dict[str, Any]],
    reserve: list[dict[str, Any]],
    output_root: Path,
    planner: SchemaModel,
    max_workers: int = 3,
    max_attempts: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if type(max_workers) is not int or not 1 <= max_workers <= len(CUDA_DEVICES):
        raise FanoutError("Stage C max_workers must be in [1,3]")
    if type(max_attempts) is not int or not 1 <= max_attempts <= MAX_STAGE_ATTEMPTS:
        raise FanoutError(f"Stage C max_attempts must be in [1,{MAX_STAGE_ATTEMPTS}]")
    _validate_complete_candidate_corpus(request, topics, scenarios, candidates)
    candidate_by_trajectory = {item["trajectoryId"]: item for item in candidates}
    selected_rows = _validate_selection_rows(
        request, primary, reserve, candidate_by_trajectory
    )
    selected_ids = {row["trajectoryId"] for row in selected_rows}
    if len(selected_ids) != len(selected_rows):
        raise FanoutError("Stage C selected set is not unique")
    selection_by_trajectory = {row["trajectoryId"]: row for row in selected_rows}
    topic_by_id = {item["topicId"]: item for item in topics}
    scenario_by_id = {item["scenarioId"]: item for item in scenarios}
    known_scenarios = set(scenario_by_id)
    output_root.mkdir(parents=True, exist_ok=True)
    store = ExpansionCheckpointStore(
        output_root / CHECKPOINT_ROOT / "stage_c_expansions",
        request,
        selected_rows,
        {key: candidate_by_trajectory[key] for key in selected_ids},
        known_scenarios,
    )
    journal = AttemptJournal(output_root / EXPANSION_AUDIT_FILENAME)

    def expand_one(trajectory_id: str) -> None:
        if store.get(trajectory_id) is not None:
            return
        candidate = candidate_by_trajectory[trajectory_id]
        selection = selection_by_trajectory[trajectory_id]
        scenario = scenario_by_id[_candidate_scenario_id(candidate)]
        topic = topic_by_id[scenario["topicId"]]
        schema = full_trajectory_response_schema(candidate)
        prior_failure_code: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response, metadata = _invoke_model(
                    planner,
                    name=EXPANSION_RESPONSE_NAME,
                    schema=schema,
                    instructions=(
                        "Expand one selected compact semantic-control card. Return only the strict "
                        "trajectory-seed.v2 JSON-Schema object, without dialogue or analysis."
                    ),
                    context=_expansion_prompt_context(
                        request,
                        topic,
                        scenario,
                        candidate,
                        selection,
                        prior_failure_code,
                    ),
                    max_output_tokens=MAX_SCHEMA_OUTPUT_TOKENS,
                )
                premise_state = response["premiseState"]
                trajectory = response["trajectory"]
                validate_expanded_trajectory(trajectory, candidate, known_scenarios)
                store.admit(candidate, selection, premise_state, trajectory, metadata)
                journal.append(
                    {
                        "schema": "personaplex.trajectory-expansion-attempt.v2",
                        "trajectoryId": trajectory_id,
                        "attempt": attempt,
                        "accepted": True,
                        "failureCode": None,
                        "responseHash": metadata.get("responseHash"),
                    }
                )
                return
            except RetryableModelOutput as error:
                prior_failure_code = error.code
                journal.append(
                    {
                        "schema": "personaplex.trajectory-expansion-attempt.v2",
                        "trajectoryId": trajectory_id,
                        "attempt": attempt,
                        "accepted": False,
                        "failureCode": error.code,
                    }
                )
        raise FanoutError(
            f"{trajectory_id}: full expansion not admitted after {max_attempts} attempts "
            f"(last failure: {prior_failure_code})"
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(expand_one, trajectory_id) for trajectory_id in sorted(selected_ids)]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    wrappers = store.rows()
    if len(wrappers) != len(selected_ids) or {row["trajectoryId"] for row in wrappers} != selected_ids:
        raise FanoutError("Stage C checkpoints do not exactly cover the selected-only set")
    trajectories = sorted(
        (row["trajectory"] for row in wrappers), key=lambda item: item["trajectoryId"]
    )
    if len({item["trajectoryId"] for item in trajectories}) != len(trajectories):
        raise FanoutError("expanded trajectory IDs must be unique")
    trajectory_path = output_root / TRAJECTORIES_FILENAME
    write_jsonl_atomic(trajectory_path, trajectories)
    manifest_payload = {
        "schema": "personaplex.trajectory-expansion-manifest.v2",
        "requestHash": content_hash(request),
        "selectedCandidateSetHash": content_hash(sorted(selected_ids)),
        "trajectoryCount": len(trajectories),
        "primaryCount": len(primary),
        "reserveCount": len(reserve),
        "checkpointMode": "immutable-content-addressed-per-selected-leaf",
        "causalSiblingRoles": list(CAUSAL_SIBLING_ROLES),
        "trajectorySetHash": content_hash(trajectories),
        "candidateBindings": {
            wrapper["trajectoryId"]: {
                "candidateId": wrapper["candidateId"],
                "candidateHash": wrapper["candidateHash"],
                "selectionHash": wrapper["selectionHash"],
                "trajectoryHash": wrapper["trajectoryHash"],
            }
            for wrapper in wrappers
        },
        "files": {
            TRAJECTORIES_FILENAME: hash_file(trajectory_path),
            EXPANSION_AUDIT_FILENAME: hash_file(output_root / EXPANSION_AUDIT_FILENAME),
        },
    }
    manifest = {**manifest_payload, "manifestHash": content_hash(manifest_payload)}
    write_json_atomic(output_root / EXPANSION_MANIFEST_FILENAME, manifest)
    return trajectories, manifest


def write_combined_manifest(
    output_root: Path, request: Mapping[str, Any]
) -> dict[str, Any]:
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
    payload = {
        "schema": "personaplex.compact-trajectory-fanout-manifest.v1",
        "requestHash": content_hash(request),
        "architecture": "authentic-compact-10_then-typed-disjoint-250-plus-250_then-selected-only-expand",
        "causalSiblingRoles": list(CAUSAL_SIBLING_ROLES),
        "files": {
            name: hash_file(output_root / name)
            for name in tracked
            if (output_root / name).is_file()
        },
    }
    manifest = {**payload, "manifestHash": content_hash(payload)}
    write_json_atomic(output_root / COMBINED_MANIFEST_FILENAME, manifest)
    return manifest
