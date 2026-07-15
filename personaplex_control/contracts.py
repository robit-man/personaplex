"""Validated, transport-neutral semantic-control messages.

These objects carry semantic constraints rather than a raw prompt. A server can
reject stale guidance and acknowledge exactly which revision was applied at a
turn boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any


class ControlMode(str, Enum):
    EXPRESSIVE = "expressive"
    STRICT = "strict"


def _require_text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be non-empty text up to {maximum} characters")
    return value.strip()


def _text_list(value: object, name: str, maximum_items: int = 32) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise ValueError(f"{name} must be a list of at most {maximum_items} strings")
    return tuple(_require_text(item, name, 512) for item in value)


@dataclass(frozen=True)
class SemanticPlan:
    """Authoritative intent and constraints for one forthcoming response."""

    intent: str
    facts: tuple[str, ...] = ()
    required_entities: Mapping[str, str] = field(default_factory=dict)
    allowed_claims: tuple[str, ...] = ()
    prohibited_claims: tuple[str, ...] = ()
    target_text: str | None = None
    style: str = "natural, concise, spoken"

    def __post_init__(self) -> None:
        object.__setattr__(self, "intent", _require_text(self.intent, "intent", 512))
        object.__setattr__(self, "facts", tuple(_require_text(item, "facts", 1024) for item in self.facts))
        object.__setattr__(self, "allowed_claims", tuple(_require_text(item, "allowed_claims", 1024) for item in self.allowed_claims))
        object.__setattr__(self, "prohibited_claims", tuple(_require_text(item, "prohibited_claims", 1024) for item in self.prohibited_claims))
        object.__setattr__(self, "style", _require_text(self.style, "style", 512))
        if self.target_text is not None:
            object.__setattr__(self, "target_text", _require_text(self.target_text, "target_text", 4000))
        if not isinstance(self.required_entities, Mapping) or len(self.required_entities) > 32:
            raise ValueError("required_entities must contain at most 32 string pairs")
        entities = {
            _require_text(key, "required_entities key", 128): _require_text(value, "required_entities value", 512)
            for key, value in self.required_entities.items()
        }
        object.__setattr__(self, "required_entities", entities)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SemanticPlan":
        return cls(
            intent=value.get("intent"),
            facts=_text_list(value.get("facts", []), "facts"),
            required_entities=value.get("required_entities", {}),
            allowed_claims=_text_list(value.get("allowed_claims", []), "allowed_claims"),
            prohibited_claims=_text_list(value.get("prohibited_claims", []), "prohibited_claims"),
            target_text=value.get("target_text"),
            style=value.get("style", "natural, concise, spoken"),
        )


@dataclass(frozen=True)
class ControlUpdate:
    """A monotonic semantic-plan revision sent by the semantic authority."""

    call_id: str
    revision: int
    apply_after_turn_id: int
    base_context_hash: str
    context_hash: str
    mode: ControlMode
    plan: SemanticPlan

    def __post_init__(self) -> None:
        object.__setattr__(self, "call_id", _require_text(self.call_id, "call_id", 256))
        object.__setattr__(self, "base_context_hash", _require_text(self.base_context_hash, "base_context_hash", 256))
        object.__setattr__(self, "context_hash", _require_text(self.context_hash, "context_hash", 256))
        if not isinstance(self.revision, int) or self.revision < 1:
            raise ValueError("revision must be a positive integer")
        if not isinstance(self.apply_after_turn_id, int) or self.apply_after_turn_id < 0:
            raise ValueError("apply_after_turn_id must be a non-negative integer")
        if not isinstance(self.mode, ControlMode):
            object.__setattr__(self, "mode", ControlMode(self.mode))
        if not isinstance(self.plan, SemanticPlan):
            raise ValueError("plan must be a SemanticPlan")
        if self.mode is ControlMode.STRICT and not self.plan.target_text:
            raise ValueError("strict mode requires canonical target_text")

    @classmethod
    def from_wire(cls, payload: bytes | str) -> "ControlUpdate":
        raw = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
        if raw.get("type") != "control.update":
            raise ValueError("expected control.update message")
        return cls(
            call_id=raw.get("call_id"),
            revision=raw.get("revision"),
            apply_after_turn_id=raw.get("apply_after_turn_id"),
            base_context_hash=raw.get("base_context_hash"),
            context_hash=raw.get("context_hash"),
            mode=ControlMode(raw.get("mode")),
            plan=SemanticPlan.from_dict(raw.get("plan", {})),
        )

    def to_wire(self) -> str:
        data = asdict(self)
        data["type"] = "control.update"
        data["mode"] = self.mode.value
        return json.dumps(data, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class ControlAck:
    """Server acknowledgement; applied is true only after a turn boundary."""

    call_id: str
    revision: int
    applied: bool
    reason: str
    turn_id: int | None = None

    def to_wire(self) -> str:
        data = asdict(self)
        data["type"] = "control.ack"
        return json.dumps(data, separators=(",", ":"), sort_keys=True)
