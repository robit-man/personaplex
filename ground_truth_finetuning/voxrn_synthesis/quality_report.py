#!/usr/bin/env python3
"""Aggregate-only calibration report for independently certified Voryn source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def metric(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "maximum": max(values) if values else None,
    }


def has_word_alignment(record: dict[str, Any]) -> bool:
    segments = record.get("asr", {}).get("segments", [])
    return any(isinstance(segment, dict) and segment.get("words") for segment in segments)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    files = sorted(args.input_root.rglob("*v8cf*.certified.jsonl"))
    records: list[dict[str, Any]] = []
    for file_path in files:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))

    wer = [float(item["asr"]["wer"]) for item in records if isinstance(item.get("asr", {}).get("wer"), (int, float))]
    confidence = [float(item["asr"]["confidence"]) for item in records if isinstance(item.get("asr", {}).get("confidence"), (int, float))]
    report = {
        "schema": "personaplex.voryn-source-quality-report.v1",
        "files": [str(path) for path in files],
        "records": len(records),
        "audio_coverage": sum(bool(item.get("audioPath")) for item in records),
        "word_alignment_coverage": sum(has_word_alignment(item) for item in records),
        "wer": metric(wer),
        "asr_confidence": metric(confidence),
        "admission_note": "Aggregate report only. Use the independent Voryn source certifier and native tensor certifier for promotion.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
