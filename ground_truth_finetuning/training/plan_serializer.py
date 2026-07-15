"""Deterministic serialization of typed plans for an existing text tokenizer."""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from .contracts import ControlPlan, ContractError


class TextTokenEncoder(Protocol):
    def encode(self, text: str) -> Sequence[int]: ...


def _atom(value: str) -> str:
    return re.sub(r"[^a-z0-9_.:/-]+", "_", value.strip().lower())


class PlanSerializer:
    """Serializes bounded fields in a stable order; never serializes target wording."""

    version = 1

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
