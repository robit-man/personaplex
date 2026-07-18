"""Deterministic serialization of typed plans for an existing text tokenizer."""

from __future__ import annotations

import re
from typing import Any, Protocol, Sequence

from .contracts import (
    ControlPlan,
    ControlTrainingFrame,
    EvidenceTrainingFrame,
    ContractError,
    assert_evidence_control_alignment,
)


class TextTokenEncoder(Protocol):
    def encode(self, text: str) -> Sequence[int]: ...


def _atom(value: str) -> str:
    return re.sub(r"[^a-z0-9_.:/-]+", "_", value.strip().lower())


class PlanSerializer:
    """Serializes bounded fields in a stable order; never serializes target wording."""

    version = 3
    _FRAME_SEGMENT_BUDGETS = (
        ("header", 48),
        ("plan", 176),
        ("state", 104),
        ("context", 152),
        ("turn", 32),
    )

    def render(self, plan: ControlPlan) -> str:
        fields: list[str] = [
            f"<control:v{self.version}>",
            f"<mode:{_atom(plan.mode)}>",
            f"<intent:{_atom(plan.intent)}>",
            f"<act:{_atom(plan.dialogue_act)}>",
        ]
        for key, value in sorted(plan.entities.items()):
            fields.append(f"<entity:{_atom(key)}={_atom(value)}>")
        for fact in plan.constraints.required_facts:
            fields.append(f"<require:{_atom(fact)}>")
        for claim in plan.constraints.forbidden_claims:
            fields.append(f"<forbid:{_atom(claim)}>")
        for question in plan.constraints.must_ask:
            fields.append(f"<ask:{_atom(question)}>")
        for request in plan.constraints.must_not_request:
            fields.append(f"<no_request:{_atom(request)}>")
        delivery = plan.delivery
        fields.extend(
            [
                f"<language:{_atom(delivery.language)}>",
                f"<register:{_atom(delivery.register)}>",
                f"<assertive:{delivery.assertiveness:.2f}>",
                f"<rate:{_atom(delivery.speaking_rate_bucket)}>",
                f"<pauses:{_atom(delivery.pause_density_bucket)}>",
                f"<interrupt:{_atom(delivery.interruptibility)}>",
                f"<duration_ms:{delivery.max_duration_ms}>",
            ]
        )
        for target in delivery.emphasis_targets:
            fields.append(f"<emphasis:{_atom(target)}>")
        fields.append("<control:end>")
        return " ".join(fields)

    def encode(self, plan: ControlPlan, tokenizer: TextTokenEncoder, text_cardinality: int) -> list[int]:
        tokens = [int(token) for token in tokenizer.encode(self.render(plan))]
        if not tokens:
            raise ContractError("plan tokenizer returned no tokens")
        if any(token < 0 or token >= text_cardinality for token in tokens):
            raise ContractError("plan tokenizer emitted a token outside the loaded model text vocabulary")
        return tokens

    def render_frame(self, frame: ControlTrainingFrame, *, include_text_context: bool = True) -> str:
        """Serialize mutable state plus a typed plan in a bounded, fixed order."""
        fields = [
            "<control-frame:v2>",
            f"<state-revision:{frame.state_revision}>",
            f"<apply-at:{_atom(str(frame.update['applyAt']))}>",
            f"<update-reason:{_atom(str(frame.update.get('reason', 'unknown')))}>",
        ]
        fields.extend(f"<source:{_atom(source)}>" for source in frame.semantic_sources)
        state = dict(frame.state)
        text_context = state.pop("textContext", None)
        for path, value in self._flatten_state(state):
            fields.append(f"<state:{_atom(path)}={_atom(value)}>")
        if include_text_context:
            fields.extend(self._render_text_context(text_context))
        for path, value in self._flatten_state(frame.turn_taking, prefix="turn"): 
            fields.append(f"<state:{_atom(path)}={_atom(value)}>")
        fields.append(self.render(frame.plan))
        fields.append("<control-frame:end>")
        return " ".join(fields)

    def _encode_text(self, value: str, tokenizer: TextTokenEncoder, text_cardinality: int) -> list[int]:
        tokens = [int(token) for token in tokenizer.encode(value)]
        if not tokens:
            raise ContractError("control-frame tokenizer returned no tokens")
        if any(token < 0 or token >= text_cardinality for token in tokens):
            raise ContractError("control-frame tokenizer emitted a token outside the loaded model text vocabulary")
        return tokens

    def _budgeted_frame_segments(self, frame: ControlTrainingFrame, *, include_text_context: bool) -> dict[str, str]:
        state = dict(frame.state)
        text_context = state.pop("textContext", None)
        end_call_authorized = bool(state.pop("endCallAuthorized", False))
        header = [
            f"<control-frame:v{self.version}>",
            f"<state-revision:{frame.state_revision}>",
            f"<apply-at:{_atom(str(frame.update['applyAt']))}>",
            f"<update-reason:{_atom(str(frame.update.get('reason', 'unknown')))}>",
            f"<end-call-authorized:{int(end_call_authorized)}>",
        ]
        header.extend(f"<source:{_atom(source)}>" for source in frame.semantic_sources)
        state_fields = [f"<state:{_atom(path)}={_atom(value)}>" for path, value in self._flatten_state(state)]
        turn_fields = [f"<state:{_atom(path)}={_atom(value)}>" for path, value in self._flatten_state(frame.turn_taking, prefix="turn")]
        return {
            "header": " ".join(header),
            "plan": " ".join([self.render(frame.plan), "<control-frame:end>"]),
            "state": " ".join(state_fields),
            "context": " ".join(self._render_text_context(text_context, max_turns=2, max_text_chars=320)) if include_text_context else "",
            "turn": " ".join(turn_fields),
        }

    def encode_frame(
        self,
        frame: ControlTrainingFrame,
        tokenizer: TextTokenEncoder,
        text_cardinality: int,
        *,
        include_text_context: bool = True,
        max_tokens: int | None = None,
    ) -> list[int]:
        if max_tokens is None:
            return self._encode_text(
                self.render_frame(frame, include_text_context=include_text_context),
                tokenizer,
                text_cardinality,
            )
        if max_tokens < 1:
            raise ContractError("control-frame max_tokens must be positive")
        segments = self._budgeted_frame_segments(frame, include_text_context=include_text_context)
        total_budget = sum(budget for _, budget in self._FRAME_SEGMENT_BUDGETS)
        tokens: list[int] = []
        for name, nominal_budget in self._FRAME_SEGMENT_BUDGETS:
            remaining = max_tokens - len(tokens)
            if remaining < 1:
                break
            budget = nominal_budget if max_tokens >= total_budget else max(1, round(nominal_budget * max_tokens / total_budget))
            value = segments[name]
            if value:
                tokens.extend(self._encode_text(value, tokenizer, text_cardinality)[:min(budget, remaining)])
        return tokens

    def render_evidence(self, frame: EvidenceTrainingFrame, control: ControlTrainingFrame) -> str:
        """Serialize bounded late evidence without serializing target wording."""
        assert_evidence_control_alignment(control, frame)
        fields = [
            "<evidence-frame:v1>",
            f"<availability:{_atom(frame.availability)}>",
            f"<evidence-revision:{frame.evidence_revision}>",
            f"<supports-control:{frame.supports_control_revision}>",
            f"<counterfactual-group:{_atom(str(frame.counterfactual['groupId']))}>",
            f"<counterfactual-branch:{_atom(str(frame.counterfactual['branchId']))}>",
            f"<counterfactual-change:{_atom(str(frame.counterfactual['changedField']))}>",
        ]
        for path, value in self._flatten_state(frame.provenance, prefix="provenance"):
            fields.append(f"<evidence:{_atom(path)}={_atom(value)}>")
        for claim in frame.allowed_claims:
            fields.append(f"<allowed-claim:{_atom(claim)}>")
        fields.append("<evidence-frame:end>")
        return " ".join(fields)

    def encode_evidence(
        self,
        frame: EvidenceTrainingFrame,
        control: ControlTrainingFrame,
        tokenizer: TextTokenEncoder,
        text_cardinality: int,
    ) -> list[int]:
        tokens = [int(token) for token in tokenizer.encode(self.render_evidence(frame, control))]
        if not tokens:
            raise ContractError("evidence tokenizer returned no tokens")
        if any(token < 0 or token >= text_cardinality for token in tokens):
            raise ContractError("evidence tokenizer emitted a token outside the loaded model text vocabulary")
        return tokens

    @staticmethod
    def _render_text_context(value: Any, *, max_turns: int = 6, max_text_chars: int = 480) -> list[str]:
        if not isinstance(value, dict):
            return []
        turns = value.get("turns")
        if not isinstance(turns, list):
            return []
        fields = ["<audible-context:begin>"]
        for item in turns[-max_turns:]:
            if not isinstance(item, dict):
                continue
            speaker = _atom(str(item.get("speaker", "unknown")))
            source = _atom(str(item.get("source", "unknown")))
            raw_text = str(item.get("text", ""))[:max_text_chars]
            raw_text = re.sub(r"[<>\x00-\x1f]+", " ", raw_text).strip()
            if raw_text:
                fields.extend([f"<audible-turn:{speaker}:{source}>", raw_text, "<audible-turn:end>"])
        fields.append("<audible-context:end>")
        return fields

    @staticmethod
    def _flatten_state(value: Any, prefix: str = "state") -> list[tuple[str, str]]:
        if isinstance(value, dict):
            items: list[tuple[str, str]] = []
            for key in sorted(value):
                items.extend(PlanSerializer._flatten_state(value[key], f"{prefix}.{key}"))
            return items
        if isinstance(value, list):
            return [(f"{prefix}.{index}", str(item)) for index, item in enumerate(value[:32])]
        return [(prefix, str(value))]
