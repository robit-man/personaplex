#!/usr/bin/env python3
"""Build a hash-bound hard-pair replay curriculum from complete ARC evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.hard_pair_sampling import HARD_PAIR_CURRICULUM_SCHEMA


IGNORED_LINEAGE_PATHS = {
    "state.semanticBindings.branchId",
    "state.semanticBindings.concreteUpdate",
}


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--whole-margin", type=float, default=0.08)
    parser.add_argument("--focused-margin", type=float, default=0.30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.step < 0 or args.whole_margin <= 0.0 or args.focused_margin <= 0.0:
        raise SystemExit("step and margins must be positive")
    train_pairs = [row for row in load_jsonl(args.pair_index) if row.get("split") == "train"]
    by_id = {str(row["pair_id"]): row for row in train_pairs}
    events = [
        row for row in load_jsonl(args.metrics)
        if row.get("event") == "checkpoint" and int(row.get("step", -1)) == args.step
    ]
    if len(events) != 1:
        raise SystemExit("metrics must contain exactly one requested checkpoint event")
    details = events[0].get("train_evaluation", {}).get("details")
    if not isinstance(details, list):
        raise SystemExit("checkpoint lacks train-evaluation details")
    measured = {str(row.get("pair_id")): row for row in details if isinstance(row, dict)}
    if set(measured) != set(by_id):
        raise SystemExit("train evaluation does not exactly cover the certified train pairs")
    axis_counts = Counter(
        path
        for pair in train_pairs
        for path in pair.get("changed_paths", [])
        if path not in IGNORED_LINEAGE_PATHS
    )
    raw_rows: list[dict[str, Any]] = []
    for pair_id, pair in sorted(by_id.items()):
        detail = measured[pair_id]
        whole = [float(value) for value in detail["whole_deltas"]]
        focused = [float(value) for value in detail["focused_deltas"]]
        deficits = [
            max(0.0, (args.whole_margin - whole[index]) / args.whole_margin)
            + max(0.0, (args.focused_margin - focused[index]) / args.focused_margin)
            for index in range(2)
        ]
        difficulty = 0.5 * (sum(deficits) / len(deficits)) + 0.5 * max(deficits)
        axes = [
            path for path in pair.get("changed_paths", [])
            if path not in IGNORED_LINEAGE_PATHS and axis_counts[path] > 0
        ]
        rarity = (
            sum(math.sqrt(len(train_pairs) / axis_counts[path]) for path in axes) / len(axes)
            if axes else 1.0
        )
        raw_rows.append(
            {
                "pairId": pair_id,
                "difficulty": difficulty,
                "axisRarity": min(3.0, rarity),
                "rawWeight": (0.25 + difficulty) * min(3.0, rarity),
                "changedPaths": axes,
            }
        )
    raw_mean = sum(row["rawWeight"] for row in raw_rows) / len(raw_rows)
    for row in raw_rows:
        row["weight"] = min(5.0, max(0.1, row.pop("rawWeight") / raw_mean))
    normalized_mean = sum(row["weight"] for row in raw_rows) / len(raw_rows)
    for row in raw_rows:
        row["weight"] /= normalized_mean
    output = {
        "schema": HARD_PAIR_CURRICULUM_SCHEMA,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "pairIndex": str(args.pair_index.resolve()),
        "pairIndexSha256": file_hash(args.pair_index.resolve()),
        "metrics": str(args.metrics.resolve()),
        "metricsSha256": file_hash(args.metrics.resolve()),
        "sourceStep": args.step,
        "thresholds": {"whole": args.whole_margin, "focused": args.focused_margin},
        "method": "normalized_margin_deficit_times_typed_axis_rarity_v1",
        "targetTextUsed": False,
        "pairs": raw_rows,
        "statistics": {
            "count": len(raw_rows),
            "minimumWeight": min(row["weight"] for row in raw_rows),
            "maximumWeight": max(row["weight"] for row in raw_rows),
            "meanWeight": sum(row["weight"] for row in raw_rows) / len(raw_rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": "built", "output": str(args.output), **output["statistics"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
