"""Admit GPUs, stage the caller-safe upstream backend, and launch LoRA training."""

from __future__ import annotations

import sys
from pathlib import Path
GTFT_TOOL_ROOT = Path(__file__).resolve().parents[2]
if str(GTFT_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(GTFT_TOOL_ROOT))

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from ground_truth_finetuning.training.gpu_admission import admit_gpus
from ground_truth_finetuning.tools.export_moshi_finetune_dataset import UPSTREAM_REVISION
from ground_truth_finetuning.tools.stage_moshi_finetune_backend import main as stage_backend


def read_certificate(path: Path) -> dict:
    certificate = json.loads(path.read_text())
    if certificate.get("kind") != "personaplex-upstream-lora-dataset-certificate":
        raise ValueError("unexpected dataset certificate kind")
    if certificate.get("status") != "certified_for_upstream_agent_only_lora":
        raise ValueError("dataset has not passed upstream agent-only certification")
    if certificate.get("upstream", {}).get("revision") != UPSTREAM_REVISION:
        raise ValueError("dataset was exported for a different upstream revision")
    return certificate


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-certificate", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--mimi-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--min-free-gib", type=float, default=44.0)
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    parser.add_argument("--max-utilization-pct", type=int, default=25)
    parser.add_argument("--allow-gpu", action="append", type=int, default=None)
    parser.add_argument("--duration-sec", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise SystemExit(f"refusing existing run root: {run_root}")
    certificate = read_certificate(args.dataset_certificate.resolve())
    required = [args.moshi_path, args.mimi_path, args.tokenizer_path]
    if args.config_path is not None:
        required.append(args.config_path)
    for path in required:
        if not path.is_file():
            raise SystemExit(f"required model artifact is missing: {path}")
    report = admit_gpus(
        world_size=args.world_size,
        min_free_gib=args.min_free_gib,
        reserve_gib=args.reserve_gib,
        max_utilization_pct=args.max_utilization_pct,
        allowed_indices=args.allow_gpu,
    )
    run_root.mkdir(parents=True)
    write_json(run_root / "gpu_admission.json", report)
    if report["status"] != "admitted":
        print(json.dumps({"status": "refused", "reason": report.get("refusal")}))
        return 2
    layout_path = dataset_root / "stream_layout.json"
    if not layout_path.is_file():
        raise SystemExit("dataset stream_layout.json is missing")
    backend = run_root / "backend"
    old_argv = sys.argv
    try:
        sys.argv = [
            "stage_moshi_finetune_backend",
            "--upstream-root", str(args.upstream_root.resolve()),
            "--destination", str(backend),
            "--stream-layout", str(layout_path),
        ]
        stage_backend()
    finally:
        sys.argv = old_argv
    config = {
        "data": {
            "train_data": str((dataset_root / "train.jsonl").resolve()),
            "eval_data": str((dataset_root / "validation.jsonl").resolve()),
            "shuffle": True,
        },
        "moshi_paths": {
            "hf_repo_id": None,
            "moshi_path": str(args.moshi_path.resolve()),
            "mimi_path": str(args.mimi_path.resolve()),
            "tokenizer_path": str(args.tokenizer_path.resolve()),
            "config_path": str(args.config_path.resolve()) if args.config_path else None,
        },
        "full_finetuning": False,
        "lora": {"enable": True, "rank": 64, "scaling": 2.0, "ft_embed": False},
        "first_codebook_weight_multiplier": 25.0,
        "text_padding_weight": 0.5,
        "duration_sec": args.duration_sec,
        "batch_size": args.batch_size,
        "num_microbatches": args.microbatches,
        "max_steps": args.max_steps,
        "gradient_checkpointing": True,
        "optim": {"lr": 2e-6, "weight_decay": 0.1, "pct_start": 0.05},
        "seed": 20260715,
        "log_freq": 1,
        "eval_freq": 0,
        "do_eval": False,
        "do_ckpt": True,
        "ckpt_freq": max(1, args.max_steps),
        "save_adapters": True,
        "run_dir": str((run_root / "training").resolve()),
    }
    config_path = run_root / "training.yaml"
    write_json(config_path, config)
    run_manifest = {
        "schema_version": 1,
        "kind": "personaplex-upstream-lora-run",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_certificate": str(args.dataset_certificate.resolve()),
        "dataset_certificate_status": certificate["status"],
        "gpu_admission": report,
        "backend": str(backend),
        "config": str(config_path),
        "execute": args.execute,
    }
    write_json(run_root / "run_manifest.json", run_manifest)
    command = ["torchrun", "--standalone", "--nproc-per-node", str(args.world_size), "-m", "train", str(config_path)]
    if not args.execute:
        print(json.dumps({"status": "staged", "command": command}))
        return 0
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in report["selected_gpu_indices"])
    environment["PERSONAPLEX_STREAM_LAYOUT_PATH"] = str(layout_path)
    completed = subprocess.run(command, cwd=backend, env=environment)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
