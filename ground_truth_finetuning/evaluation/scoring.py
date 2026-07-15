"""Deterministic plan-versus-ASR scoring for held-out runs."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from ..training.contracts import ControlPlan


def normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


@dataclass(frozen=True)
class PlanScore:
    required_fact_recall: float
    required_question_coverage: float
    entity_recall: float
    forbidden_claims: tuple[str, ...]
    exact_match: bool | None


def _recall(required: Iterable[str], actual: str) -> float:
    required_values = [normalize_text(value) for value in required]
    if not required_values:
        return 1.0
    return sum(value in actual for value in required_values) / len(required_values)


def score_transcript(plan: ControlPlan, transcript: str, canonical_text: str | None = None) -> PlanScore:
    actual = normalize_text(transcript)
    forbidden = tuple(
        value for value in plan.constraints.forbidden_claims if normalize_text(value) in actual
    )
    return PlanScore(
        required_fact_recall=_recall(plan.constraints.required_facts, actual),
        required_question_coverage=_recall(plan.constraints.must_ask, actual),
        entity_recall=_recall(plan.entities.values(), actual),
        forbidden_claims=forbidden,
        exact_match=(normalize_text(canonical_text) == actual) if canonical_text is not None else None,
    )
