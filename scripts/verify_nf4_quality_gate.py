#!/usr/bin/env python3
"""Fail closed unless a direct-NF4 release report passed the BF16 quality contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"[personaplex-nf4-quality] ERROR: {message}")


def required_bool(value: object, label: str) -> None:
    if value is not True:
        fail(f"{label} must be true")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if not args.report.is_file():
        fail(f"quality report not found: {args.report}")

    try:
        report = json.loads(args.report.read_text())
    except (OSError, json.JSONDecodeError) as error:
        fail(f"quality report is unreadable: {error}")

    if report.get("schema") != "personaplex.nf4-quality-report.v1":
        fail("unsupported quality report schema")
    required_bool(report.get("bf16ReferencePassed"), "bf16ReferencePassed")

    mimi = report.get("mimiRoundTrip") or {}
    required_bool(mimi.get("passed"), "mimiRoundTrip.passed")
    if not isinstance(mimi.get("wer"), (int, float)) or float(mimi["wer"]) > 0.12:
        fail("mimiRoundTrip.wer must be <= 0.12")

    nf4 = report.get("nf4") or {}
    required_bool(nf4.get("directPackedWeights"), "nf4.directPackedWeights")
    required_bool(nf4.get("cudaOnly"), "nf4.cudaOnly")
    required_bool(nf4.get("nonEmptyWhisperTranscript"), "nf4.nonEmptyWhisperTranscript")
    required_bool(nf4.get("semanticResponseAccepted"), "nf4.semanticResponseAccepted")
    if nf4.get("repetitiveOutput") is not False:
        fail("nf4.repetitiveOutput must be false")

    parity = report.get("firstTwentyFrames") or {}
    required_bool(parity.get("bf16ParityPassed"), "firstTwentyFrames.bf16ParityPassed")
    if int(parity.get("comparedFrames", 0)) < 20:
        fail("firstTwentyFrames.comparedFrames must be at least 20")

    print(json.dumps({"ok": True, "report": str(args.report)}))


if __name__ == "__main__":
    main()
