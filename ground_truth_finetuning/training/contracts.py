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


def _stream_index_list(mapping: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ContractError(f"{key} must be a non-empty list of stream indices")
    if not all(isinstance(item, int) and item >= 0 for item in value):
        raise ContractError(f"{key} must contain non-negative integer stream indices")
    if len(set(value)) != len(value):
        raise ContractError(f"{key} must not contain duplicate stream indices")
    return tuple(value)


@dataclass(frozen=True)
class StreamLayout:
    """The exact global-codebook ownership for a loaded duplex PersonaPlex LM.

    PersonaPlex's 17 streams are not interchangeable.  The canonical Moshi layout
    has one text stream, eight agent-output Mimi streams, and eight caller-input
    Mimi streams.  Training must name those groups explicitly because ``dep_q``
    spans both audio directions in the native model.
    """

    text_stream_indices: tuple[int, ...]
    agent_audio_stream_indices: tuple[int, ...]
    caller_audio_stream_indices: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StreamLayout":
        layout = cls(
            text_stream_indices=_stream_index_list(value, "text_stream_indices"),
            agent_audio_stream_indices=_stream_index_list(value, "agent_audio_stream_indices"),
            caller_audio_stream_indices=_stream_index_list(value, "caller_audio_stream_indices"),
        )
        layout.validate_static()
        return layout

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "text_stream_indices": list(self.text_stream_indices),
            "agent_audio_stream_indices": list(self.agent_audio_stream_indices),
            "caller_audio_stream_indices": list(self.caller_audio_stream_indices),
        }

    def validate_static(self) -> None:
        all_indices = (
            self.text_stream_indices
            + self.agent_audio_stream_indices
            + self.caller_audio_stream_indices
        )
        if len(set(all_indices)) != len(all_indices):
            raise ContractError("stream-layout groups must be disjoint")
        if len(self.text_stream_indices) != 1:
            raise ContractError("current native training supports exactly one text stream")

    def validate_for_model(self, lm_model: object) -> None:
        self.validate_static()
        expected = set(range(int(lm_model.num_codebooks)))
        actual = set(
            self.text_stream_indices
            + self.agent_audio_stream_indices
            + self.caller_audio_stream_indices
        )
        if actual != expected:
            raise ContractError(
                "stream-layout must account for every loaded model codebook exactly once"
            )
        if self.text_stream_indices != (0,):
            raise ContractError("native delayed path currently requires the text stream at index 0")
        audio_start = int(lm_model.audio_offset)
        audio_end = audio_start + int(lm_model.dep_q)
        for stream_index in self.agent_audio_stream_indices + self.caller_audio_stream_indices:
            if not audio_start <= stream_index < audio_end:
                raise ContractError("audio stream lies outside the native depformer range")

    def agent_audio_output_indices(self, lm_model: object) -> tuple[int, ...]:
        self.validate_for_model(lm_model)
        return tuple(index - int(lm_model.audio_offset) for index in self.agent_audio_stream_indices)


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
