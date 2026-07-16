#!/usr/bin/env python3
"""Validate a PersonaPlex controlled-duplex export without accepting diagnostics."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels,sample_rate,duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)["streams"][0]


def rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--allow-diagnostics", action="store_true")
    args = parser.parse_args()
    root = args.export_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    examples = rows(root / "examples.jsonl")
    diagnostics = rows(root / "diagnostic_examples.jsonl")
    for example in examples:
        if example.get("schema") != "personaplex.controlled-duplex.example.v1":
            failures.append(f"{example.get('exampleId')}: schema")
            continue
        frame = example.get("controlFrame") or {}
        label = normalise(example.get("labels", {}).get("agentText", ""))
        serialised_frame = json.dumps(frame, sort_keys=True, separators=(",", ":"))
        if not frame.get("frameId") or not frame.get("stateRevision"):
            failures.append(f"{example.get('exampleId')}: invalid control frame")
        if "canonicalResponse" in serialised_frame or (label and len(label) >= 16 and label in normalise(serialised_frame)):
            failures.append(f"{example.get('exampleId')}: label leaked into control input")
        path = root / example["duplexAudio"]["path"]
        try:
            audio = probe(path)
            if int(audio["channels"]) != 2 or int(audio["sample_rate"]) != 24000:
                failures.append(f"{example.get('exampleId')}: duplex audio must be 24 kHz stereo")
        except Exception as error:
            failures.append(f"{example.get('exampleId')}: audio invalid: {error}")
    if diagnostics and not args.allow_diagnostics:
        failures.append(f"diagnostic examples present: {len(diagnostics)}")
    report = {
        "schema": "personaplex.controlled-duplex-validation.v1",
        "admittedExamples": len(examples),
        "diagnosticExamples": len(diagnostics),
        "manifestAdmittedExamples": manifest.get("admittedExampleCount"),
        "failures": failures,
        "passed": not failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
