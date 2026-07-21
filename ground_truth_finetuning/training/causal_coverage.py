"""Structural coverage certification for semantic-control counterfactual pairs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


LINEAGE_PATHS = frozenset({"state.semanticBindings.branchId"})
SEMANTIC_PATH_ROOTS = (
    "state.activeControlGuidance",
    "state.callerPosture",
    "state.commitments",
    "state.compliancePosture",
    "state.endCallAuthorized",
    "state.facts",
    "state.intent",
    "state.nextGoal",
    "state.policyBoundaries",
    "state.policyConstraints",
    "state.recoveryPending",
    "state.recoveryStyle",
    "state.resistancePosture",
    "state.semanticBindings.concreteUpdate",
    "state.toolResults",
    "state.uncertainty",
    "state.unresolved",
    "plan.constraints",
    "plan.dialogueAct",
    "plan.entities",
    "plan.intent",
    "plan.mode",
)
DELIVERY_PATH_ROOTS = ("plan.delivery",)
TURN_TAKING_PATH_ROOTS = ("turnTaking",)


def _matches_root(path: str, root: str) -> bool:
    return path == root or path.startswith(root + ".")


def classify_changed_path(path: str) -> str:
    """Classify a schema path without inspecting natural-language values."""

    if path in LINEAGE_PATHS:
        return "lineage"
    if any(_matches_root(path, root) for root in SEMANTIC_PATH_ROOTS):
        return "semantic"
    if any(_matches_root(path, root) for root in DELIVERY_PATH_ROOTS):
        return "delivery"
    if any(_matches_root(path, root) for root in TURN_TAKING_PATH_ROOTS):
        return "turn_taking"
    return "unknown"


def intervention_kind(changed_paths: Iterable[str]) -> str:
    families = {
        classify_changed_path(path)
        for path in changed_paths
        if classify_changed_path(path) != "lineage"
    }
    if not families:
        return "lineage_only"
    if "unknown" in families:
        return "unknown"
    if len(families) == 1:
        return f"{next(iter(families))}_only"
    return "composite"


@dataclass(frozen=True)
class CausalCoverageThresholds:
    expected_axes: tuple[str, ...]
    required_splits: tuple[str, ...]
    min_pairs_per_axis: int
    min_distinct_premises_per_axis: int
    min_signature_support: int
    min_supported_pair_fraction: float
    max_composite_fraction: float
    min_barge_in_pairs: int
    min_recovery_pairs: int

    def __post_init__(self) -> None:
        if not self.expected_axes:
            raise ValueError("expected_axes must not be empty")
        if not self.required_splits:
            raise ValueError("required_splits must not be empty")
        for name in (
            "min_pairs_per_axis",
            "min_distinct_premises_per_axis",
            "min_signature_support",
            "min_barge_in_pairs",
            "min_recovery_pairs",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("min_supported_pair_fraction", "max_composite_fraction"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def _member_frame(
    pair: Mapping[str, Any],
    member_name: str,
    examples: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    member = pair.get(member_name)
    if not isinstance(member, Mapping):
        raise ValueError(f"pair lacks {member_name}")
    example_id = str(member.get("example_id") or "")
    if example_id not in examples:
        raise ValueError(f"pair member references unknown example: {example_id}")
    control = examples[example_id].get("control")
    if not isinstance(control, Mapping) or not isinstance(control.get("frame"), Mapping):
        raise ValueError(f"example lacks typed control frame: {example_id}")
    return control["frame"]


def _frame_metadata(frame: Mapping[str, Any]) -> dict[str, Any]:
    state = frame.get("state") if isinstance(frame.get("state"), Mapping) else {}
    bindings = (
        state.get("semanticBindings")
        if isinstance(state.get("semanticBindings"), Mapping)
        else {}
    )
    turn_taking = (
        frame.get("turnTaking")
        if isinstance(frame.get("turnTaking"), Mapping)
        else {}
    )
    return {
        "axis": str(bindings.get("counterfactualAxis") or ""),
        "premise": str(bindings.get("scenarioPremise") or ""),
        "caller_posture": str(state.get("callerPosture") or ""),
        "compliance_posture": str(state.get("compliancePosture") or ""),
        "resistance_posture": str(state.get("resistancePosture") or ""),
        "expected_barge_in": bool(turn_taking.get("expectedBargeIn")),
        "recovery_expected": bool(turn_taking.get("recoveryExpected")),
    }


def build_causal_coverage_report(
    pairs: Sequence[Mapping[str, Any]],
    examples: Mapping[str, Mapping[str, Any]],
    thresholds: CausalCoverageThresholds,
) -> dict[str, Any]:
    """Build a target-text-free structural certificate for a causal pair corpus."""

    reasons: list[str] = []
    seen_pair_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    axis_counts: Counter[str] = Counter()
    axis_premises: dict[str, set[str]] = defaultdict(set)
    signature_counts: Counter[tuple[str, str, tuple[str, ...]]] = Counter()
    signature_premises: dict[tuple[str, str, tuple[str, ...]], set[str]] = defaultdict(set)
    pair_signatures: list[tuple[str, str, tuple[str, ...]]] = []
    barge_in_pairs = 0
    recovery_pairs = 0
    posture_counts: dict[str, Counter[str]] = {
        "caller": Counter(),
        "compliance": Counter(),
        "resistance": Counter(),
    }

    for index, pair in enumerate(pairs):
        pair_id = str(pair.get("pair_id") or "")
        if not pair_id:
            reasons.append(f"pair at index {index} lacks pair_id")
            continue
        if pair_id in seen_pair_ids:
            reasons.append(f"duplicate pair_id: {pair_id}")
            continue
        seen_pair_ids.add(pair_id)
        split = str(pair.get("split") or "")
        group_id = str(pair.get("group_id") or "")
        if not split or not group_id:
            reasons.append(f"pair {pair_id} lacks split or group_id")
            continue
        split_counts[split] += 1
        group_splits[group_id].add(split)

        try:
            frame_a = _member_frame(pair, "member_a", examples)
            frame_b = _member_frame(pair, "member_b", examples)
        except ValueError as exc:
            reasons.append(str(exc))
            continue
        metadata_a = _frame_metadata(frame_a)
        metadata_b = _frame_metadata(frame_b)
        axis = metadata_a["axis"]
        premise = metadata_a["premise"]
        if not axis or axis != metadata_b["axis"]:
            reasons.append(f"pair {pair_id} has missing or mismatched counterfactual axis")
            continue
        if not premise or premise != metadata_b["premise"]:
            reasons.append(f"pair {pair_id} has missing or mismatched scenario premise")
            continue

        changed_paths = tuple(sorted(str(path) for path in pair.get("changed_paths") or ()))
        kind = intervention_kind(changed_paths)
        kind_counts[kind] += 1
        axis_counts[axis] += 1
        axis_premises[axis].add(premise)
        non_lineage_paths = tuple(
            path for path in changed_paths if classify_changed_path(path) != "lineage"
        )
        signature = (axis, kind, non_lineage_paths)
        signature_counts[signature] += 1
        signature_premises[signature].add(premise)
        pair_signatures.append(signature)

        barge_in_pairs += int(
            metadata_a["expected_barge_in"] or metadata_b["expected_barge_in"]
        )
        recovery_pairs += int(
            metadata_a["recovery_expected"] or metadata_b["recovery_expected"]
        )
        posture_counts["caller"][metadata_a["caller_posture"]] += 1
        posture_counts["compliance"][metadata_a["compliance_posture"]] += 1
        posture_counts["resistance"][metadata_a["resistance_posture"]] += 1

    leaking_groups = sorted(
        group for group, splits in group_splits.items() if len(splits) > 1
    )
    if leaking_groups:
        reasons.append(f"counterfactual groups cross splits: {len(leaking_groups)}")
    missing_splits = sorted(set(thresholds.required_splits).difference(split_counts))
    if missing_splits:
        reasons.append(f"missing required splits: {missing_splits}")

    axis_records: dict[str, Any] = {}
    for axis in thresholds.expected_axes:
        count = axis_counts[axis]
        premise_count = len(axis_premises[axis])
        axis_records[axis] = {
            "pairs": count,
            "distinctPremises": premise_count,
        }
        if count < thresholds.min_pairs_per_axis:
            reasons.append(
                f"axis {axis} has {count} pairs; requires {thresholds.min_pairs_per_axis}"
            )
        if premise_count < thresholds.min_distinct_premises_per_axis:
            reasons.append(
                f"axis {axis} has {premise_count} premises; requires "
                f"{thresholds.min_distinct_premises_per_axis}"
            )
    unexpected_axes = sorted(set(axis_counts).difference(thresholds.expected_axes))
    if unexpected_axes:
        reasons.append(f"unexpected counterfactual axes: {unexpected_axes}")

    pair_count = len(pair_signatures)
    supported_pairs = sum(
        1
        for signature in pair_signatures
        if signature_counts[signature] >= thresholds.min_signature_support
        and len(signature_premises[signature]) >= thresholds.min_signature_support
    )
    supported_fraction = supported_pairs / pair_count if pair_count else 0.0
    if supported_fraction < thresholds.min_supported_pair_fraction:
        reasons.append(
            f"supported pair fraction {supported_fraction:.6f} is below "
            f"{thresholds.min_supported_pair_fraction:.6f}"
        )
    composite_fraction = kind_counts["composite"] / pair_count if pair_count else 0.0
    if composite_fraction > thresholds.max_composite_fraction:
        reasons.append(
            f"composite intervention fraction {composite_fraction:.6f} exceeds "
            f"{thresholds.max_composite_fraction:.6f}"
        )
    if kind_counts["unknown"]:
        reasons.append(f"unknown changed paths in {kind_counts['unknown']} pairs")
    if kind_counts["lineage_only"]:
        reasons.append(f"lineage-only interventions in {kind_counts['lineage_only']} pairs")
    if barge_in_pairs < thresholds.min_barge_in_pairs:
        reasons.append(
            f"barge-in pairs {barge_in_pairs} below {thresholds.min_barge_in_pairs}"
        )
    if recovery_pairs < thresholds.min_recovery_pairs:
        reasons.append(
            f"recovery pairs {recovery_pairs} below {thresholds.min_recovery_pairs}"
        )

    signature_records = [
        {
            "axis": axis,
            "kind": kind,
            "changedPaths": list(paths),
            "pairs": signature_counts[(axis, kind, paths)],
            "distinctPremises": len(signature_premises[(axis, kind, paths)]),
        }
        for axis, kind, paths in sorted(signature_counts)
    ]
    return {
        "schema": "personaplex.causal-coverage-certificate.v1",
        "status": "certified" if not reasons else "rejected",
        "pairCount": pair_count,
        "splitCounts": dict(sorted(split_counts.items())),
        "groupCount": len(group_splits),
        "groupSplitLeakageCount": len(leaking_groups),
        "interventionKinds": dict(sorted(kind_counts.items())),
        "compositeFraction": composite_fraction,
        "supportedPairs": supported_pairs,
        "supportedPairFraction": supported_fraction,
        "bargeInPairs": barge_in_pairs,
        "recoveryPairs": recovery_pairs,
        "axes": axis_records,
        "postures": {
            name: dict(sorted(counts.items())) for name, counts in posture_counts.items()
        },
        "signatures": signature_records,
        "thresholds": {
            "expectedAxes": list(thresholds.expected_axes),
            "requiredSplits": list(thresholds.required_splits),
            "minPairsPerAxis": thresholds.min_pairs_per_axis,
            "minDistinctPremisesPerAxis": thresholds.min_distinct_premises_per_axis,
            "minSignatureSupport": thresholds.min_signature_support,
            "minSupportedPairFraction": thresholds.min_supported_pair_fraction,
            "maxCompositeFraction": thresholds.max_composite_fraction,
            "minBargeInPairs": thresholds.min_barge_in_pairs,
            "minRecoveryPairs": thresholds.min_recovery_pairs,
        },
        "reasons": reasons,
    }
