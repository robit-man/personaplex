#!/usr/bin/env python3
"""Fail-closed Voryn-to-PersonaPlex controlled native training pipeline.

The pipeline never trains on raw JSONL. It exports admitted duplex calls, creates
label-separated pre-codec data, encodes matching native streams, writes a tensor
certificate, and only then asks the resource-admitted launcher to run epochs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "ground_truth_finetuning" / "tools"


def run(command: Sequence[str]) -> None:
    print(json.dumps({"command": list(command)}), flush=True)
    completed = subprocess.run(list(command))
    if completed.returncode:
        raise SystemExit(completed.returncode)


def require_new(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"refusing existing pipeline output: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voryn-input", action="append", type=Path, required=True, help="Admitted Voryn dataset root or JSONL; repeatable")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--mimi-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--allow-gpu", action="append", type=int, default=None)
    parser.add_argument("--encode-device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-examples", type=int, default=64)
    parser.add_argument("--min-free-gib", type=float, default=44.0)
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    parser.add_argument("--max-utilization-pct", type=int, default=25)
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="stop after the certified native tensor corpus is written")
    args = parser.parse_args()
    if args.execute_training and args.prepare_only:
        raise SystemExit("--execute-training and --prepare-only are mutually exclusive")
    if args.world_size < 1 or args.max_steps < 1 or args.checkpoint_every < 1 or args.eval_examples < 1:
        raise SystemExit("world-size, max-steps, checkpoint-every, and eval-examples must be positive")
    root = args.output_root.resolve()
    require_new(root)
    for required in [args.moshi_source_root, args.moshi_path, args.mimi_path, args.tokenizer_path, args.model_contract]:
        if not required.exists():
            raise SystemExit(f"required native artifact is missing: {required}")

    export_root = root / "01_export"
    precodec_root = root / "02_precodec"
    artifact_root = root / "03_native_tensors"
    certificate = root / "04_certificate" / "controlled_native_certificate.json"
    training_root = root / "05_training"
    root.mkdir(parents=True)
    run([sys.executable, str(TOOLS / "export_controlled_duplex_dataset.py"), *[str(path.resolve()) for path in args.voryn_input], "--output-dir", str(export_root)])
    run([sys.executable, str(TOOLS / "prepare_controlled_native_adapter_dataset.py"), "--export-root", str(export_root), "--output-root", str(precodec_root)])
    run([
        sys.executable, str(TOOLS / "encode_controlled_native_adapter_tensors.py"),
        "--manifest", str(precodec_root / "precodec_manifest.jsonl"),
        "--precodec-root", str(precodec_root), "--artifact-root", str(artifact_root),
        "--moshi-source-root", str(args.moshi_source_root.resolve()), "--mimi-path", str(args.mimi_path.resolve()),
        "--tokenizer-path", str(args.tokenizer_path.resolve()), "--model-contract", str(args.model_contract.resolve()),
        "--device", args.encode_device,
    ])
    encoded_manifest = artifact_root / "encoded_examples.jsonl"
    run([
        sys.executable, str(TOOLS / "certify_controlled_native_corpus.py"),
        "--manifest", str(encoded_manifest), "--artifact-root", str(artifact_root),
        "--precodec-root", str(precodec_root), "--certificate", str(certificate),
    ])
    certificate_data = json.loads(certificate.read_text(encoding="utf-8"))
    if certificate_data.get("status") != "certified_for_adapter_training":
        raise SystemExit("native tensor certificate did not authorize adapter training")
    if args.prepare_only:
        (root / "pipeline_summary.json").write_text(json.dumps({
            "status": "prepared_for_training",
            "certificate": str(certificate), "encoded_manifest": str(encoded_manifest),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    launch = [
        sys.executable, str(TOOLS / "launch_semantic_prefix.py"),
        "--manifest", str(encoded_manifest), "--artifact-root", str(artifact_root), "--certificate", str(certificate),
        "--model-contract", str(args.model_contract.resolve()), "--moshi-source-root", str(args.moshi_source_root.resolve()),
        "--moshi-path", str(args.moshi_path.resolve()), "--tokenizer-path", str(args.tokenizer_path.resolve()),
        "--run-root", str(training_root), "--world-size", str(args.world_size), "--max-steps", str(args.max_steps),
        "--checkpoint-every", str(args.checkpoint_every), "--eval-examples", str(args.eval_examples),
        "--min-free-gib", str(args.min_free_gib), "--reserve-gib", str(args.reserve_gib),
        "--max-utilization-pct", str(args.max_utilization_pct),
    ]
    for gpu in args.allow_gpu or []:
        launch.extend(["--allow-gpu", str(gpu)])
    if args.execute_training:
        launch.append("--execute")
    run(launch)
    (root / "pipeline_summary.json").write_text(json.dumps({
        "status": "training_executed" if args.execute_training else "training_staged",
        "certificate": str(certificate), "encoded_manifest": str(encoded_manifest), "training_root": str(training_root),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
