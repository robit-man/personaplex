"""Field-aware, target-free encoding for semantic control frames.

The v3 serializer flattened most values into identifier-like atoms.  This module
keeps lexical values in natural language so the frozen PersonaPlex text embedding
table supplies useful semantic grounding.  Field, value-kind, semantic-source,
and revision features remain explicit trainable channels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Protocol, Sequence

import torch
from torch import Tensor

from .contracts import ControlTrainingFrame, EvidenceTrainingFrame, ContractError


class TextTokenEncoder(Protocol):
    def encode(self, text: str) -> Sequence[int]: ...


FIELD_NAMES = (
    "padding",
    "header",
    "mode",
    "intent",
    "dialogue_act",
    "next_goal",
    "guidance",
    "phase",
    "posture",
    "fact",
    "commitment",
    "uncertainty",
    "unresolved",
    "policy",
    "tool_result",
    "semantic_binding",
    "topic",
    "audible_context",
    "entity",
    "required_fact",
    "forbidden_claim",
    "must_ask",
    "must_not_request",
    "delivery",
    "turn_taking",
    "update",
    "evidence",
    "termination",
    "other_state",
)
VALUE_KIND_NAMES = (
    "padding",
    "metadata",
    "natural_text",
    "fact",
    "constraint",
    "question",
    "entity",
    "style",
    "number",
    "boolean",
    "prior_speech",
    "tool",
    "evidence",
)
SOURCE_NAMES = (
    "state_reducer",
    "asr_finalizer",
    "policy_agent",
    "task_agent",
    "tool_result",
    "interruption_controller",
    "handoff_router",
    "timer",
    "knowledge_agent",
    "safety_agent",
)
FIELD_TO_ID = {name: index for index, name in enumerate(FIELD_NAMES)}
VALUE_KIND_TO_ID = {name: index for index, name in enumerate(VALUE_KIND_NAMES)}
SOURCE_TO_BIT = {name: index for index, name in enumerate(SOURCE_NAMES)}
REVISION_BUCKET_COUNT = 18
CURRENT_REVISION_BUCKET = 9


def _clean_text(value: Any) -> str:
    raw = str(value)
    cleaned = "".join(" " if ord(char) < 32 or char in "<>" else char for char in raw)
    return " ".join(cleaned.split())


def _human_key(value: str) -> str:
    return " ".join(value.replace("_", " ").replace("-", " ").split())


def _source_mask(names: Iterable[str]) -> int:
    mask = 0
    for name in names:
        bit = SOURCE_TO_BIT.get(name)
        if bit is not None:
            mask |= 1 << bit
    return mask


@dataclass(frozen=True)
class ControlSegment:
    field: str
    value_kind: str
    source_mask: int
    text: str
    priority: int
    token_cap: int = 96


@dataclass(frozen=True)
class EncodedControl:
    token_ids: tuple[int, ...]
    field_ids: tuple[int, ...]
    value_kind_ids: tuple[int, ...]
    source_masks: tuple[int, ...]
    revision_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        lengths = {
            len(self.token_ids),
            len(self.field_ids),
            len(self.value_kind_ids),
            len(self.source_masks),
            len(self.revision_ids),
        }
        if lengths != {len(self.token_ids)} or not self.token_ids:
            raise ValueError("encoded control channels must be non-empty and aligned")


@dataclass(frozen=True)
class ControlTensorBatch:
    token_ids: Tensor
    attention_mask: Tensor
    field_ids: Tensor
    value_kind_ids: Tensor
    source_masks: Tensor
    revision_ids: Tensor
    control_present: Tensor


def pad_encoded_controls(
    controls: Sequence[EncodedControl],
    *,
    device: torch.device,
    present: Sequence[bool] | None = None,
) -> ControlTensorBatch:
    if not controls:
        raise ValueError("at least one encoded control is required")
    if present is None:
        present = [True] * len(controls)
    if len(present) != len(controls):
        raise ValueError("present flags must align with controls")
    width = max(len(item.token_ids) for item in controls)
    shape = (len(controls), width)
    token_ids = torch.zeros(shape, dtype=torch.long, device=device)
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    field_ids = torch.zeros(shape, dtype=torch.long, device=device)
    value_kind_ids = torch.zeros(shape, dtype=torch.long, device=device)
    source_masks = torch.zeros(shape, dtype=torch.long, device=device)
    revision_ids = torch.zeros(shape, dtype=torch.long, device=device)
    for row, item in enumerate(controls):
        length = len(item.token_ids)
        token_ids[row, :length] = torch.tensor(item.token_ids, dtype=torch.long, device=device)
        mask[row, :length] = True
        field_ids[row, :length] = torch.tensor(item.field_ids, dtype=torch.long, device=device)
        value_kind_ids[row, :length] = torch.tensor(item.value_kind_ids, dtype=torch.long, device=device)
        source_masks[row, :length] = torch.tensor(item.source_masks, dtype=torch.long, device=device)
        revision_ids[row, :length] = torch.tensor(item.revision_ids, dtype=torch.long, device=device)
    return ControlTensorBatch(
        token_ids=token_ids,
        attention_mask=mask,
        field_ids=field_ids,
        value_kind_ids=value_kind_ids,
        source_masks=source_masks,
        revision_ids=revision_ids,
        control_present=torch.tensor(present, dtype=torch.bool, device=device),
    )


class FieldAwareControlSerializer:
    """Serialize semantic state as natural lexical segments plus typed channels."""

    version = 5

    _STATE_FIELDS = {
        "intent": ("intent", "natural_text", "task_agent", 5),
        "nextGoal": ("next_goal", "natural_text", "task_agent", 4),
        "activeControlGuidance": ("guidance", "natural_text", "task_agent", 3),
        "phase": ("phase", "metadata", "state_reducer", 12),
        "callerPosture": ("posture", "metadata", "state_reducer", 7),
        "compliancePosture": ("posture", "metadata", "state_reducer", 7),
        "resistancePosture": ("posture", "metadata", "state_reducer", 7),
        "recoveryPending": ("turn_taking", "boolean", "interruption_controller", 8),
        "recoveryStyle": ("turn_taking", "style", "interruption_controller", 8),
        "endCallAuthorized": ("termination", "boolean", "state_reducer", 2),
        "topic": ("topic", "natural_text", "task_agent", 15),
    }
    _LIST_FIELDS = {
        "facts": ("fact", "fact", "state_reducer", 4),
        "commitments": ("commitment", "fact", "state_reducer", 5),
        "uncertainty": ("uncertainty", "fact", "state_reducer", 5),
        "unresolved": ("unresolved", "fact", "state_reducer", 6),
        "policyBoundaries": ("policy", "constraint", "policy_agent", 3),
        "policyConstraints": ("policy", "constraint", "policy_agent", 3),
    }

    @staticmethod
    def _revision_id(frame_revision: int, expected_revision: int | None) -> int:
        expected = frame_revision if expected_revision is None else expected_revision
        delta = max(-8, min(8, frame_revision - expected))
        return delta + CURRENT_REVISION_BUCKET

    @staticmethod
    def _segment(
        field: str,
        kind: str,
        source: str | Iterable[str],
        label: str,
        value: Any,
        priority: int,
        token_cap: int = 96,
    ) -> ControlSegment | None:
        text = _clean_text(value)
        if not text:
            return None
        sources = [source] if isinstance(source, str) else list(source)
        return ControlSegment(
            field=field,
            value_kind=kind,
            source_mask=_source_mask(sources),
            text=f"{label}: {text}{'' if text.endswith(('.', '?', '!')) else '.'}",
            priority=priority,
            token_cap=token_cap,
        )

    def _state_segments(self, frame: ControlTrainingFrame) -> list[ControlSegment]:
        segments: list[ControlSegment] = []
        handled = set(self._STATE_FIELDS) | set(self._LIST_FIELDS) | {
            "textContext",
            "toolResults",
            "semanticBindings",
        }
        for key, spec in self._STATE_FIELDS.items():
            if key in frame.state:
                segment = self._segment(*spec[:3], _human_key(key), frame.state[key], spec[3])
                if segment is not None:
                    segments.append(segment)
        for key, spec in self._LIST_FIELDS.items():
            values = frame.state.get(key, [])
            if isinstance(values, list):
                for value in values[:32]:
                    segment = self._segment(*spec[:3], _human_key(key), value, spec[3], 72)
                    if segment is not None:
                        segments.append(segment)
        bindings = frame.state.get("semanticBindings")
        if isinstance(bindings, dict):
            for key, value in sorted(bindings.items()):
                segment = self._segment(
                    "semantic_binding",
                    "fact",
                    "state_reducer",
                    f"semantic binding {_human_key(key)}",
                    value,
                    4,
                    96,
                )
                if segment is not None:
                    segments.append(segment)
        tools = frame.state.get("toolResults")
        if isinstance(tools, list):
            for tool in tools[:16]:
                if isinstance(tool, dict):
                    natural = "; ".join(
                        f"{_human_key(str(key))} is {_clean_text(value)}"
                        for key, value in sorted(tool.items())
                        if key not in {"id", "revision"}
                    )
                else:
                    natural = _clean_text(tool)
                segment = self._segment("tool_result", "tool", "tool_result", "verified tool result", natural, 2, 128)
                if segment is not None:
                    segments.append(segment)
        context = frame.state.get("textContext")
        if isinstance(context, dict) and isinstance(context.get("turns"), list):
            for turn in context["turns"][-8:]:
                if not isinstance(turn, dict):
                    continue
                speaker = _clean_text(turn.get("speaker", "unknown"))
                source = _clean_text(turn.get("source", "unknown"))
                text = _clean_text(turn.get("text", ""))
                if text:
                    segments.append(
                        ControlSegment(
                            field="audible_context",
                            value_kind="prior_speech",
                            source_mask=_source_mask(["asr_finalizer"]),
                            text=f"Previously audible {speaker} speech from {source}: {text}",
                            priority=6,
                            token_cap=128,
                        )
                    )
        for key, value in sorted(frame.state.items()):
            if key in handled:
                continue
            segment = self._segment(
                "other_state",
                "natural_text",
                "state_reducer",
                f"state {_human_key(key)}",
                value,
                18,
                64,
            )
            if segment is not None:
                segments.append(segment)
        return segments

    def segments(
        self,
        frame: ControlTrainingFrame,
        evidence: EvidenceTrainingFrame | None = None,
    ) -> list[ControlSegment]:
        global_sources = frame.semantic_sources
        plan = frame.plan
        segments = [
            ControlSegment("header", "metadata", _source_mask(global_sources), "Semantic control for the next agent turn.", 0, 24),
            self._segment("mode", "metadata", global_sources, "rendering mode", plan.mode, 1, 24),
            self._segment("intent", "natural_text", "task_agent", "plan intent", plan.intent, 2, 96),
            self._segment("dialogue_act", "metadata", "task_agent", "dialogue act", plan.dialogue_act, 4, 32),
            self._segment("update", "metadata", global_sources, "update reason", frame.update.get("reason", "unknown"), 9, 48),
        ]
        result = [segment for segment in segments if segment is not None]
        result.extend(self._state_segments(frame))
        for key, value in plan.entities.items():
            segment = self._segment("entity", "entity", "task_agent", f"known entity {_human_key(key)}", value, 3, 64)
            if segment is not None:
                result.append(segment)
        constraint_groups = (
            ("required_fact", "fact", "required fact", plan.constraints.required_facts, 2),
            ("forbidden_claim", "constraint", "forbidden claim", plan.constraints.forbidden_claims, 1),
            ("must_ask", "question", "question to ask", plan.constraints.must_ask, 3),
            ("must_not_request", "constraint", "request that is forbidden", plan.constraints.must_not_request, 1),
        )
        for field, kind, label, values, priority in constraint_groups:
            for value in values:
                segment = self._segment(field, kind, "policy_agent", label, value, priority, 72)
                if segment is not None:
                    result.append(segment)
        delivery = plan.delivery
        delivery_values = {
            "language": delivery.language,
            "register": delivery.register,
            "assertiveness": f"{delivery.assertiveness:.2f}",
            "speaking rate": delivery.speaking_rate_bucket,
            "pause density": delivery.pause_density_bucket,
            "interruptibility": delivery.interruptibility,
            "maximum duration milliseconds": delivery.max_duration_ms,
            "emphasis targets": ", ".join(delivery.emphasis_targets) if delivery.emphasis_targets else "none",
        }
        for key, value in delivery_values.items():
            segment = self._segment("delivery", "style", "task_agent", key, value, 10, 40)
            if segment is not None:
                result.append(segment)
        for key, value in sorted(frame.turn_taking.items()):
            segment = self._segment(
                "turn_taking",
                "metadata",
                "interruption_controller",
                f"turn taking {_human_key(key)}",
                value,
                5,
                48,
            )
            if segment is not None:
                result.append(segment)
        if evidence is not None:
            for claim in evidence.allowed_claims:
                segment = self._segment("evidence", "evidence", "tool_result", "available evidence claim", claim, 1, 96)
                if segment is not None:
                    result.append(segment)
            for key, value in sorted(evidence.provenance.items()):
                segment = self._segment(
                    "evidence",
                    "evidence",
                    "knowledge_agent",
                    f"evidence {_human_key(key)}",
                    value,
                    7,
                    64,
                )
                if segment is not None:
                    result.append(segment)
            result.append(
                ControlSegment(
                    "evidence",
                    "metadata",
                    _source_mask(["tool_result"]),
                    f"Evidence availability: {_clean_text(evidence.availability)}.",
                    1,
                    24,
                )
            )
        return sorted(result, key=lambda item: item.priority)

    def encode(
        self,
        frame: ControlTrainingFrame,
        tokenizer: TextTokenEncoder,
        text_cardinality: int,
        *,
        evidence: EvidenceTrainingFrame | None = None,
        expected_revision: int | None = None,
        max_tokens: int = 512,
    ) -> EncodedControl:
        if max_tokens < 32:
            raise ContractError("field-aware control encoding requires at least 32 tokens")
        revision_id = self._revision_id(frame.state_revision, expected_revision)
        token_ids: list[int] = []
        field_ids: list[int] = []
        kind_ids: list[int] = []
        source_masks: list[int] = []
        revision_ids: list[int] = []
        for segment in self.segments(frame, evidence):
            remaining = max_tokens - len(token_ids)
            if remaining <= 0:
                break
            encoded = [int(token) for token in tokenizer.encode(segment.text)]
            if any(token < 0 or token >= text_cardinality for token in encoded):
                raise ContractError("control tokenizer emitted an ID outside the PersonaPlex text vocabulary")
            if not encoded:
                continue
            if len(encoded) > max_tokens:
                if segment.priority <= 4:
                    raise ContractError(
                        f"complete {segment.field} control segment exceeds the {max_tokens}-token "
                        "control context; split it upstream at a natural semantic boundary"
                    )
                continue
            if len(encoded) > remaining:
                continue
            token_ids.extend(encoded)
            field_ids.extend([FIELD_TO_ID[segment.field]] * len(encoded))
            kind_ids.extend([VALUE_KIND_TO_ID[segment.value_kind]] * len(encoded))
            source_masks.extend([segment.source_mask] * len(encoded))
            revision_ids.extend([revision_id] * len(encoded))
        if not token_ids:
            raise ContractError("field-aware control frame encoded to no tokens")
        return EncodedControl(
            tuple(token_ids),
            tuple(field_ids),
            tuple(kind_ids),
            tuple(source_masks),
            tuple(revision_ids),
        )


def revision_delta_from_id(revision_id: int) -> int:
    if not 1 <= revision_id < REVISION_BUCKET_COUNT:
        raise ValueError("revision ID is outside the encoded range")
    return revision_id - CURRENT_REVISION_BUCKET
