"""Resource-admitted launcher for distributed semantic-prefix training."""

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

from ground_truth_finetuning.training.gpu_admission import admit_gpus


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--min-free-gib", type=float, default=44.0)
    parser.add_argument("--reserve-gib", type=float, default=8.0)
    parser.add_argument("--max-utilization-pct", type=int, default=25)
    parser.add_argument("--allow-gpu", action="append", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.max_steps < 1 or args.checkpoint_every < 1 or args.eval_examples < 1:
        raise SystemExit("max-steps, checkpoint-every, and eval-examples must be positive")
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise SystemExit(f"refusing existing run root: {run_root}")
    required = [args.manifest, args.certificate, args.model_contract, args.moshi_path, args.tokenizer_path]
    for path in required:
        if not path.is_file():
            raise SystemExit(f"required artifact is missing: {path}")
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
    command = [
        "torchrun", "--standalone", "--nproc-per-node", str(args.world_size), "-m",
        "ground_truth_finetuning.tools.train_semantic_prefix",
        "--manifest", str(args.manifest.resolve()),
        "--artifact-root", str(args.artifact_root.resolve()),
        "--certificate", str(args.certificate.resolve()),
        "--model-contract", str(args.model_contract.resolve()),
        "--moshi-source-root", str(args.moshi_source_root.resolve()),
        "--moshi-path", str(args.moshi_path.resolve()),
        "--tokenizer-path", str(args.tokenizer_path.resolve()),
        "--run-dir", str((run_root / "training").resolve()),
        "--max-steps", str(args.max_steps),
        "--checkpoint-every", str(args.checkpoint_every),
        "--eval-examples", str(args.eval_examples),
    ]
    run_manifest = {
        "schema_version": 1,
        "kind": "personaplex-semantic-prefix-launch",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "gpu_admission": report,
        "execute": args.execute,
    }
    write_json(run_root / "launch.json", run_manifest)
    if not args.execute:
        print(json.dumps({"status": "staged", "command": command}))
        return 0
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in report["selected_gpu_indices"])
    return subprocess.run(command, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
