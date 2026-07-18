#!/usr/bin/env python3
"""Certify a completed control-v3 tensor root and activate isolated training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


TOOLS = Path(__file__).resolve().parent


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise SystemExit(f"required {label} is missing or empty: {resolved}")
    return resolved


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--allowed-gpus", default="0,1,2")
    parser.add_argument("--train-service", default="")
    args = parser.parse_args()

    prepared_root = args.prepared_root.expanduser().resolve()
    precodec_root = prepared_root / "02_precodec"
    artifact_root = prepared_root / "03_native_tensors"
    manifest = require_file(artifact_root / "encoded_examples.jsonl", "merged encoded manifest")
    require_file(artifact_root / "encoding_report.json", "encoding report")
    require_file(precodec_root / "precodec_manifest.jsonl", "pre-codec manifest")
    contract = require_file(args.model_contract, "model contract")
    source_root = args.moshi_source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise SystemExit(f"required Moshi source root is missing: {source_root}")
    moshi_path = require_file(args.moshi_path, "Moshi weights")
    tokenizer_path = require_file(args.tokenizer_path, "tokenizer")
    try:
        allowed_gpus = [int(item) for item in args.allowed_gpus.split(",") if item.strip()]
    except ValueError as error:
        raise SystemExit("allowed-gpus must be a comma-separated integer list") from error
    if not allowed_gpus or any(gpu < 0 for gpu in allowed_gpus) or len(set(allowed_gpus)) != len(allowed_gpus):
        raise SystemExit("allowed-gpus must be unique non-negative GPU indices")

    certificate = prepared_root / "04_certificate" / "controlled_native_certificate.json"
    command = [
        sys.executable, str(TOOLS / "certify_controlled_native_corpus.py"),
        "--manifest", str(manifest), "--artifact-root", str(artifact_root),
        "--precodec-root", str(precodec_root), "--certificate", str(certificate),
    ]
    subprocess.run(command, check=True)
    certificate_data = json.loads(require_file(certificate, "native tensor certificate").read_text(encoding="utf-8"))
    if certificate_data.get("status") != "certified_for_adapter_training":
        raise SystemExit("native tensor certificate did not authorize adapter training")

    state = {
        "schema": "personaplex.controlv3-transition-state.v1",
        "status": "prepared",
        "preparedAt": datetime.now(timezone.utc).isoformat(),
        "outputRoot": str(prepared_root),
        "encodedManifest": str(manifest),
        "artifactRoot": str(artifact_root),
        "certificate": str(certificate),
        "modelContract": str(contract),
        "moshiSourceRoot": str(source_root),
        "moshiPath": str(moshi_path),
        "tokenizerPath": str(tokenizer_path),
        "allowedGpus": allowed_gpus,
    }
    state_path = args.state.expanduser().resolve()
    write_json(state_path, state)
    if args.train_service:
        subprocess.run(["systemctl", "--user", "start", "--no-block", args.train_service], check=True)
    print(json.dumps({"status": state["status"], "state": str(state_path), "certificate": str(certificate)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
