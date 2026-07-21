#!/usr/bin/env python3
"""Certify causal differences inside family-specific ARC-4 frame budgets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.causal_coverage import classify_changed_path
from ground_truth_finetuning.training.contracts import (
    validate_control_frame_mapping,
    validate_evidence_frame_mapping,
)
from personaplex_control.moshirag_reference import (
    ARC4_REFERENCE_REVISION,
    render_arc4_reference,
    render_arc4_reference_fields,
)


FIELD_BY_FAMILY = {
    "semantic": "decision",
    "delivery": "delivery",
    "turn_taking": "context",
}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render_fields(row: Mapping[str, Any]) -> dict[str, str]:
    control = validate_control_frame_mapping(row["control"]["frame"])
    evidence_value = (row.get("evidence") or {}).get("frame")
    evidence = (
        validate_evidence_frame_mapping(evidence_value)
        if isinstance(evidence_value, dict)
        else None
    )
    return render_arc4_reference_fields(control, evidence)


def render_envelope(row: Mapping[str, Any]) -> str:
    control = validate_control_frame_mapping(row["control"]["frame"])
    evidence_value = (row.get("evidence") or {}).get("frame")
    evidence = (
        validate_evidence_frame_mapping(evidence_value)
        if isinstance(evidence_value, dict)
        else None
    )
    return render_arc4_reference(control, evidence)


def first_difference(left: Sequence[int], right: Sequence[int]) -> int | None:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def changed_families(changed_paths: Sequence[str]) -> tuple[str, ...]:
    families = {
        classify_changed_path(str(path))
        for path in changed_paths
        if classify_changed_path(str(path)) != "lineage"
    }
    return tuple(sorted(families))


def field_budget_result(
    left_text: str,
    right_text: str,
    tokenizer: Any,
    *,
    arc_frames: int,
    head_frames: int,
) -> dict[str, Any]:
    left_ids = tokenizer.encode(left_text, add_special_tokens=True)
    right_ids = tokenizer.encode(right_text, add_special_tokens=True)
    first = first_difference(left_ids, right_ids)
    reasons: list[str] = []
    if left_text == right_text or first is None:
        reasons.append("references_identical")
    elif first >= arc_frames:
        reasons.append("first_difference_outside_arc_budget")
    elif first >= head_frames:
        reasons.append("first_difference_outside_causal_head")
    return {
        "firstDifferentToken": first,
        "tokenLengths": [len(left_ids), len(right_ids)],
        "arcPrefixesIdentical": left_ids[:arc_frames] == right_ids[:arc_frames],
        "reasons": reasons,
    }


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arc-frames", type=int, default=96)
    parser.add_argument("--causal-head-fraction", type=float, default=0.25)
    parser.add_argument("--split", action="append")
    parser.add_argument(
        "--legacy-envelope",
        action="store_true",
        help="Check the old monolithic envelope instead of typed field slots.",
    )
    args = parser.parse_args()
    if args.arc_frames < 1 or not 0 < args.causal_head_fraction <= 1:
        raise SystemExit("arc-frames and causal-head-fraction must be positive")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(f"tokenizer dependency is required: {exc}") from exc

    manifest = args.manifest.resolve()
    pair_index = args.pair_index.resolve()
    rows = {str(row["example_id"]): row for row in load_jsonl(manifest)}
    selected_splits = set(args.split or [])
    pairs = [
        pair
        for pair in load_jsonl(pair_index)
        if not selected_splits or str(pair.get("split")) in selected_splits
    ]
    if not pairs:
        raise SystemExit("no causal pairs matched the requested splits")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True
    )
    head_frames = max(1, round(args.arc_frames * args.causal_head_fraction))
    details: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    differences_by_field: dict[str, list[int]] = defaultdict(list)

    for pair in pairs:
        member_rows = []
        for key in ("member_a", "member_b"):
            example_id = str(pair[key]["example_id"])
            if example_id not in rows:
                raise ValueError(f"pair member is absent from manifest: {example_id}")
            member_rows.append(rows[example_id])

        pair_reasons: list[str] = []
        checks: dict[str, Any] = {}
        if args.legacy_envelope:
            result = field_budget_result(
                render_envelope(member_rows[0]),
                render_envelope(member_rows[1]),
                tokenizer,
                arc_frames=args.arc_frames,
                head_frames=head_frames,
            )
            checks["envelope"] = result
        else:
            families = changed_families(pair.get("changed_paths") or ())
            unsupported = [family for family in families if family not in FIELD_BY_FAMILY]
            if unsupported or not families:
                pair_reasons.append(
                    "unmapped_changed_path_families:"
                    + ",".join(unsupported or ("none",))
                )
            fields = [render_fields(member_rows[0]), render_fields(member_rows[1])]
            for family in families:
                field = FIELD_BY_FAMILY.get(family)
                if field is None:
                    continue
                result = field_budget_result(
                    fields[0][field],
                    fields[1][field],
                    tokenizer,
                    arc_frames=args.arc_frames,
                    head_frames=head_frames,
                )
                checks[field] = result

        for field, result in checks.items():
            first = result["firstDifferentToken"]
            if first is not None:
                differences_by_field[field].append(first)
            pair_reasons.extend(f"{field}:{reason}" for reason in result["reasons"])
        detail = {
            "pairId": pair["pair_id"],
            "split": pair["split"],
            "changedFamilies": list(changed_families(pair.get("changed_paths") or ())),
            "fieldChecks": checks,
            "reasons": pair_reasons,
        }
        details.append(detail)
        if pair_reasons:
            failures.append(detail)

    def difference_summary(values: list[int]) -> dict[str, int | None]:
        return {
            "minimum": min(values) if values else None,
            "median": percentile(values, 0.5) if values else None,
            "p95": percentile(values, 0.95) if values else None,
            "maximum": max(values) if values else None,
        }

    certificate = {
        "schema": "personaplex.arc4-causal-budget-certificate.v2",
        "status": "certified" if not failures else "rejected",
        "manifest": str(manifest),
        "manifestSha256": sha256_file(manifest),
        "pairIndex": str(pair_index),
        "pairIndexSha256": sha256_file(pair_index),
        "tokenizer": str(args.tokenizer.resolve()),
        "referenceRevision": ARC4_REFERENCE_REVISION,
        "familyAware": not args.legacy_envelope,
        "arcFramesPerField": args.arc_frames,
        "causalHeadFramesPerField": head_frames,
        "pairs": len(pairs),
        "failures": len(failures),
        "firstDifferenceByField": {
            field: difference_summary(values)
            for field, values in sorted(differences_by_field.items())
        },
        "failureDetails": failures,
        "details": details,
        "targetTextPassedToSerializer": False,
        "opaqueBranchIdPassedToSerializer": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: certificate[key]
                for key in (
                    "status",
                    "pairs",
                    "failures",
                    "familyAware",
                    "firstDifferenceByField",
                )
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
