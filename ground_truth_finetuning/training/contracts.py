"""Typed contracts shared by corpus, trainer, and runtime integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Any, Mapping


_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MODES = {"expressive", "strict", "safe_fallback"}
_INTERRUPTIBILITY = {"yield_on_caller_speech", "complete_if_uninterrupted"}


class ContractError(ValueError):
    """A control or dataset contract violates a non-negotiable invariant."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_uri(value: Any) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{key} must be a non-empty string")
    return value


def _string_list(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ContractError(f"{key} must be a list of non-empty strings")
    return tuple(value)


@dataclass(frozen=True)
class Delivery:
    language: str
    register: str
    assertiveness: float
    interruptibility: str
    max_duration_ms: int
    speaking_rate_bucket: str = "normal"
    pause_density_bucket: str = "moderate"
    emphasis_targets: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Delivery":
        assertiveness = value.get("assertiveness")
        max_duration_ms = value.get("max_duration_ms")
        interruptibility = _required_string(value, "interruptibility")
        if not isinstance(assertiveness, (int, float)) or not 0 <= float(assertiveness) <= 1:
            raise ContractError("delivery.assertiveness must be between 0 and 1")
        if not isinstance(max_duration_ms, int) or not 250 <= max_duration_ms <= 30000:
            raise ContractError("delivery.max_duration_ms must be an integer between 250 and 30000")
        if interruptibility not in _INTERRUPTIBILITY:
            raise ContractError("delivery.interruptibility is unsupported")
        return cls(
            language=_required_string(value, "language"),
            register=_required_string(value, "register"),
            assertiveness=float(assertiveness),
            interruptibility=interruptibility,
            max_duration_ms=max_duration_ms,
            speaking_rate_bucket=str(value.get("speaking_rate_bucket", "normal")),
            pause_density_bucket=str(value.get("pause_density_bucket", "moderate")),
            emphasis_targets=_string_list(value, "emphasis_targets"),
        )


@dataclass(frozen=True)
class Constraints:
    required_facts: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    must_ask: tuple[str, ...]
    must_not_request: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Constraints":
        return cls(
            required_facts=_string_list(value, "required_facts"),
            forbidden_claims=_string_list(value, "forbidden_claims"),
            must_ask=_string_list(value, "must_ask"),
            must_not_request=_string_list(value, "must_not_request"),
        )


@dataclass(frozen=True)
class ControlPlan:
    schema_version: int
    call_id: str
    turn_id: int
    revision: int
    context_hash: str
    mode: str
    intent: str
    dialogue_act: str
    entities: dict[str, str]
    constraints: Constraints
    delivery: Delivery
    expiry_ms: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ControlPlan":
        forbidden = {"canonicalText", "canonical_text", "targetText", "target_text", "response"}
        present = forbidden.intersection(value)
        if present:
            raise ContractError(f"plan must not contain canonical-response fields: {sorted(present)}")
        schema_version = value.get("schemaVersion", value.get("schema_version"))
        if schema_version != 1:
            raise ContractError("only control plan schema version 1 is supported")
        turn_id = value.get("turnId", value.get("turn_id"))
        revision = value.get("revision")
        expiry_ms = value.get("expiryMs", value.get("expiry_ms"))
        if not isinstance(turn_id, int) or turn_id < 0:
            raise ContractError("turnId must be a non-negative integer")
        if not isinstance(revision, int) or revision < 0:
            raise ContractError("revision must be a non-negative integer")
        if not isinstance(expiry_ms, int) or not 250 <= expiry_ms <= 30000:
            raise ContractError("expiryMs must be between 250 and 30000")
        context_hash = _required_string(value, "contextHash")
        if not _HASH_RE.match(context_hash):
            raise ContractError("contextHash must be sha256:<64 lowercase hex characters>")
        mode = _required_string(value, "mode")
        if mode not in _MODES:
            raise ContractError(f"mode must be one of {sorted(_MODES)}")
        entities = value.get("entities", {})
        if not isinstance(entities, dict) or not all(
            isinstance(key, str) and key and isinstance(item, str) and item
            for key, item in entities.items()
        ):
            raise ContractError("entities must map non-empty strings to non-empty strings")
        constraints = value.get("constraints")
        delivery = value.get("delivery")
        if not isinstance(constraints, Mapping) or not isinstance(delivery, Mapping):
            raise ContractError("constraints and delivery must be mappings")
        return cls(
            schema_version=schema_version,
            call_id=_required_string(value, "callId"),
            turn_id=turn_id,
            revision=revision,
            context_hash=context_hash,
            mode=mode,
            intent=_required_string(value, "intent"),
            dialogue_act=_required_string(value, "dialogueAct"),
            entities=dict(sorted(entities.items())),
            constraints=Constraints.from_mapping(constraints),
            delivery=Delivery.from_mapping(delivery),
            expiry_ms=expiry_ms,
        )

    def as_wire_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "callId": self.call_id,
            "turnId": self.turn_id,
            "revision": self.revision,
            "contextHash": self.context_hash,
            "mode": self.mode,
            "intent": self.intent,
            "dialogueAct": self.dialogue_act,
            "entities": self.entities,
            "constraints": asdict(self.constraints),
            "delivery": asdict(self.delivery),
            "expiryMs": self.expiry_ms,
        }

    @property
    def plan_hash(self) -> str:
        return sha256_uri(self.as_wire_dict())


def validate_plan_mapping(value: Mapping[str, Any]) -> ControlPlan:
    return ControlPlan.from_mapping(value)


def assert_monotonic_revision(previous: ControlPlan | None, candidate: ControlPlan) -> None:
    if previous is None:
        return
    if previous.call_id != candidate.call_id:
        raise ContractError("cannot compare revisions from different calls")
    if candidate.revision <= previous.revision:
        raise ContractError("control revision must strictly increase")
