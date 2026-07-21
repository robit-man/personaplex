"""Statistical release gate for generated-audio semantic-control trials."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping


REQUIRED_JUDGMENT_FLAGS = (
    "semantic_adherence",
    "required_facts_supported",
    "forbidden_claims_avoided",
    "required_question_or_action",
    "next_goal_advanced",
    "caller_posture_respected",
    "style_adherence",
    "natural_conversational_response",
)


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total < 1 or not 0 <= successes <= total:
        return (0.0, 0.0)
    point = successes / total
    denominator = 1 + z * z / total
    centre = point + z * z / (2 * total)
    spread = z * math.sqrt((point * (1 - point) + z * z / (4 * total)) / total)
    return (
        max(0.0, (centre - spread) / denominator),
        min(1.0, (centre + spread) / denominator),
    )


@dataclass(frozen=True)
class TrialOutcome:
    trial_id: str
    passed: bool
    judge_available: bool
    stale_emissions: int
    unsupported_policy_claims: int
    pair_id: str | None
    branch_id: str | None
    pair_discrimination_pass: bool | None
    slices: dict[str, str]
    failure_reasons: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrialOutcome":
        trial_id = value.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError("trial_id is required")
        judgment = value.get("judgment")
        judge_available = isinstance(judgment, Mapping) and judgment.get("status") == "ok"
        failures: list[str] = []
        if not judge_available:
            failures.append("semantic_judge_unavailable")
            judgment = {}
        for flag in REQUIRED_JUDGMENT_FLAGS:
            if judgment.get(flag) is not True:
                failures.append(flag)
        if judgment.get("unsupported_policy_sensitive_claim") is True:
            failures.append("unsupported_policy_sensitive_claim")
        if judgment.get("stale_control_used") is True:
            failures.append("stale_control_used")
        audio = value.get("audio_checks")
        if not isinstance(audio, Mapping) or audio.get("admitted") is not True:
            failures.append("audio_not_admitted")
        transport = value.get("transport_checks")
        if not isinstance(transport, Mapping) or transport.get("passed") is not True:
            failures.append("transport_failed")
        stale_emissions = int(value.get("stale_emissions", 0))
        unsupported = int(value.get("unsupported_policy_claims", 0))
        if stale_emissions:
            failures.append("stale_media_emitted")
        if unsupported:
            failures.append("unsupported_policy_claim")
        slices = value.get("slices", {})
        if not isinstance(slices, Mapping):
            raise ValueError(f"{trial_id}: slices must be an object")
        return cls(
            trial_id=trial_id,
            passed=not failures,
            judge_available=judge_available,
            stale_emissions=stale_emissions,
            unsupported_policy_claims=unsupported,
            pair_id=str(value["pair_id"]) if value.get("pair_id") else None,
            branch_id=str(value["branch_id"]) if value.get("branch_id") else None,
            pair_discrimination_pass=value.get("pair_discrimination_pass")
            if isinstance(value.get("pair_discrimination_pass"), bool)
            else None,
            slices={str(key): str(item) for key, item in slices.items()},
            failure_reasons=tuple(failures),
        )


def _rate(successes: int, total: int) -> dict[str, Any]:
    lower, upper = wilson_interval(successes, total)
    return {
        "successes": successes,
        "total": total,
        "point": successes / total if total else 0.0,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
    }


def evaluate_release_gate(
    values: Iterable[Mapping[str, Any]],
    *,
    minimum_trials: int = 1000,
    minimum_pairs: int = 250,
    minimum_slice_trials: int = 20,
    overall_point_target: float = 0.97,
    overall_lower_target: float = 0.95,
    slice_point_target: float = 0.95,
    slice_lower_target: float = 0.90,
) -> dict[str, Any]:
    outcomes = [TrialOutcome.from_mapping(value) for value in values]
    overall = _rate(sum(item.passed for item in outcomes), len(outcomes))
    slice_members: dict[tuple[str, str], list[TrialOutcome]] = {}
    for outcome in outcomes:
        for dimension, value in outcome.slices.items():
            slice_members.setdefault((dimension, value), []).append(outcome)
    slices: dict[str, dict[str, Any]] = {}
    failing_slices: list[str] = []
    undercovered_slices: list[str] = []
    for (dimension, value), members in sorted(slice_members.items()):
        key = f"{dimension}={value}"
        result = _rate(sum(item.passed for item in members), len(members))
        result["material"] = len(members) >= minimum_slice_trials
        slices[key] = result
        if not result["material"]:
            undercovered_slices.append(key)
        elif result["point"] < slice_point_target or result["wilson_95_lower"] < slice_lower_target:
            failing_slices.append(key)
    pair_members: dict[str, list[TrialOutcome]] = {}
    for outcome in outcomes:
        if outcome.pair_id:
            pair_members.setdefault(outcome.pair_id, []).append(outcome)
    pair_passes = 0
    eligible_pairs = 0
    for members in pair_members.values():
        branches = {member.branch_id for member in members}
        if len(branches) < 2:
            continue
        eligible_pairs += 1
        pair_passes += int(
            all(member.passed for member in members)
            and all(member.pair_discrimination_pass is True for member in members)
        )
    pair_rate = _rate(pair_passes, eligible_pairs)
    zero_faults = (
        sum(item.stale_emissions for item in outcomes) == 0
        and sum(item.unsupported_policy_claims for item in outcomes) == 0
    )
    failures: list[str] = []
    if len(outcomes) < minimum_trials:
        failures.append("insufficient_generated_audio_trials")
    if overall["point"] < overall_point_target:
        failures.append("overall_point_below_target")
    if overall["wilson_95_lower"] < overall_lower_target:
        failures.append("overall_wilson_lower_below_target")
    if failing_slices:
        failures.append("material_slice_below_target")
    if eligible_pairs < minimum_pairs:
        failures.append("insufficient_causal_pairs")
    if pair_rate["point"] < 0.95:
        failures.append("causal_pair_sensitivity_below_target")
    if not zero_faults:
        failures.append("zero_tolerance_safety_or_staleness_fault")
    return {
        "schema_version": 4,
        "kind": "personaplex-generated-audio-semantic-reliability",
        "status": "passed" if not failures else "failed",
        "overall": overall,
        "causal_pairs": pair_rate,
        "slices": slices,
        "failing_slices": failing_slices,
        "undercovered_slices": undercovered_slices,
        "zero_faults": zero_faults,
        "judge_failures": sum(not item.judge_available for item in outcomes),
        "first_attempt_denominator": len(outcomes),
        "failure_reasons": failures,
        "thresholds": {
            "minimum_trials": minimum_trials,
            "minimum_pairs": minimum_pairs,
            "minimum_slice_trials": minimum_slice_trials,
            "overall_point": overall_point_target,
            "overall_wilson_lower": overall_lower_target,
            "slice_point": slice_point_target,
            "slice_wilson_lower": slice_lower_target,
            "causal_pair_point": 0.95,
        },
    }
