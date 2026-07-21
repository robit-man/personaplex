#!/usr/bin/env python3
"""Emit a fail-closed structural coverage certificate for ARC4 causal pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.causal_coverage import (
    CausalCoverageThresholds,
    build_causal_coverage_report,
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-axis", action="append", required=True)
    parser.add_argument("--required-split", action="append", required=True)
    parser.add_argument("--min-pairs-per-axis", type=int, required=True)
    parser.add_argument("--min-distinct-premises-per-axis", type=int, required=True)
    parser.add_argument("--min-signature-support", type=int, required=True)
    parser.add_argument("--min-supported-pair-fraction", type=float, required=True)
    parser.add_argument("--max-composite-fraction", type=float, required=True)
    parser.add_argument("--min-barge-in-pairs", type=int, required=True)
    parser.add_argument("--min-recovery-pairs", type=int, required=True)
    parser.add_argument("--allow-rejected", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pair_path = args.pair_index.resolve()
    manifest_path = args.manifest.resolve()
    pairs = load_jsonl(pair_path)
    example_rows = load_jsonl(manifest_path)
    examples = {str(row["example_id"]): row for row in example_rows}
    if len(examples) != len(example_rows):
        raise SystemExit("manifest contains duplicate example_id values")
    thresholds = CausalCoverageThresholds(
        expected_axes=tuple(dict.fromkeys(args.expected_axis)),
        required_splits=tuple(dict.fromkeys(args.required_split)),
        min_pairs_per_axis=args.min_pairs_per_axis,
        min_distinct_premises_per_axis=args.min_distinct_premises_per_axis,
        min_signature_support=args.min_signature_support,
        min_supported_pair_fraction=args.min_supported_pair_fraction,
        max_composite_fraction=args.max_composite_fraction,
        min_barge_in_pairs=args.min_barge_in_pairs,
        min_recovery_pairs=args.min_recovery_pairs,
    )
    report = build_causal_coverage_report(pairs, examples, thresholds)
    report["pairIndex"] = str(pair_path)
    report["pairIndexSha256"] = hash_file(pair_path)
    report["manifest"] = str(manifest_path)
    report["manifestSha256"] = hash_file(manifest_path)
    report["targetTextInspected"] = False
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({key: report[key] for key in (
        "status", "pairCount", "groupCount", "supportedPairFraction",
        "compositeFraction", "bargeInPairs", "recoveryPairs", "reasons",
    )}, sort_keys=True))
    return 0 if report["status"] == "certified" or args.allow_rejected else 2


if __name__ == "__main__":
    raise SystemExit(main())
