#!/usr/bin/env python3
"""Dynamically resource-admitted launcher and monitor for v4 control training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.gpu_admission import (
    admit_gpus_by_ratio,
    host_memory_snapshot,
    query_gpus,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_transport_probe(
    *, environment: dict[str, str], world_size: int, timeout_seconds: float
) -> dict:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(world_size),
        "-m",
        "ground_truth_finetuning.tools.probe_nccl_transport",
    ]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
    return {
        "status": "timeout" if timed_out else ("passed" if process.returncode == 0 else "failed"),
        "return_code": process.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "output": output[-8000:],
    }


def select_nccl_transport(
    *, environment: dict[str, str], world_size: int, timeout_seconds: float
) -> tuple[dict, dict[str, str]]:
    attempts = []
    for mode, overrides in (
        ("native", {}),
        ("shared_memory", {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "0"}),
    ):
        candidate = environment.copy()
        candidate.update(overrides)
        result = run_transport_probe(
            environment=candidate,
            world_size=world_size,
            timeout_seconds=timeout_seconds,
        )
        result["mode"] = mode
        attempts.append(result)
        if result["status"] == "passed":
            return {"status": "passed", "selected_mode": mode, "attempts": attempts}, overrides
    return {"status": "failed", "selected_mode": None, "attempts": attempts}, {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest",
        "artifact-root",
        "certificate",
        "pair-index",
        "pair-certificate",
        "model-contract",
        "moshi-source-root",
        "moshi-path",
        "tokenizer-path",
        "run-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--min-world-size", type=int, default=1)
    parser.add_argument("--allow-gpu", action="append", type=int, default=[0, 1, 2])
    parser.add_argument("--min-usable-ratio", type=float, default=0.30)
    parser.add_argument("--reserve-ratio", type=float, default=0.10)
    parser.add_argument("--max-utilization-pct", type=int, default=85)
    parser.add_argument("--host-memory-limit", type=float, default=0.80)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--matched-weight", type=float, default=1.0)
    parser.add_argument("--causal-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-margin", type=float, default=0.08)
    parser.add_argument("--focused-counterfactual-margin", type=float, default=0.30)
    parser.add_argument("--null-weight", type=float, default=0.25)
    parser.add_argument("--stale-weight", type=float, default=0.25)
    parser.add_argument("--train-pair-limit", type=int, default=0)
    parser.add_argument("--eval-train-pairs", type=int, default=0)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=4)
    parser.add_argument("--eval-pairs", type=int, default=16)
    parser.add_argument("--sample-interval-seconds", type=float, default=30.0)
    parser.add_argument("--history-seconds", type=float, default=20.0)
    parser.add_argument("--target-tail-seconds", type=float, default=0.64)
    parser.add_argument("--max-context-gate-adjustment", type=float, default=0.05)
    parser.add_argument("--max-stream-to-lexical-rms-ratio", type=float, default=0.25)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--transport-probe-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise SystemExit(f"refusing existing run root: {run_root}")
    if not 1 <= args.min_world_size <= args.world_size:
        raise SystemExit("min-world-size must be between one and world-size")
    host = host_memory_snapshot()
    if float(host["used_ratio"]) >= args.host_memory_limit:
        raise SystemExit(
            f"host memory is already {float(host['used_ratio']):.1%}; limit is {args.host_memory_limit:.1%}"
        )
    allowed = sorted(set(args.allow_gpu))
    report = admit_gpus_by_ratio(
        world_size=args.world_size,
        min_usable_ratio=args.min_usable_ratio,
        reserve_ratio=args.reserve_ratio,
        max_utilization_pct=args.max_utilization_pct,
        allowed_indices=allowed,
    )
    selected = report["selected_gpu_indices"]
    effective_world_size = args.world_size
    if report["status"] != "admitted" and len(selected) >= args.min_world_size:
        effective_world_size = len(selected)
        report = admit_gpus_by_ratio(
            world_size=effective_world_size,
            min_usable_ratio=args.min_usable_ratio,
            reserve_ratio=args.reserve_ratio,
            max_utilization_pct=args.max_utilization_pct,
            allowed_indices=allowed,
        )
        selected = report["selected_gpu_indices"]
        report["degraded_from_world_size"] = args.world_size
    run_root.mkdir(parents=True)
    write_json(run_root / "gpu_admission.json", report)
    if report["status"] != "admitted":
        print(json.dumps({"status": "refused", "reason": report.get("refusal")}))
        return 2
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(effective_world_size),
        "-m",
        "ground_truth_finetuning.tools.train_semantic_control_v4",
    ]
    for name in (
        "manifest",
        "artifact_root",
        "certificate",
        "pair_index",
        "pair_certificate",
        "model_contract",
        "moshi_source_root",
        "moshi_path",
        "tokenizer_path",
    ):
        command.extend([f"--{name.replace('_', '-')}", str(getattr(args, name).resolve())])
    command.extend(
        [
            "--run-dir",
            str((run_root / "training").resolve()),
            "--max-steps",
            str(args.max_steps),
            "--learning-rate",
            str(args.learning_rate),
            "--matched-weight",
            str(args.matched_weight),
            "--causal-weight",
            str(args.causal_weight),
            "--counterfactual-margin",
            str(args.counterfactual_margin),
            "--focused-counterfactual-margin",
            str(args.focused_counterfactual_margin),
            "--null-weight",
            str(args.null_weight),
            "--stale-weight",
            str(args.stale_weight),
            "--train-pair-limit",
            str(args.train_pair_limit),
            "--eval-train-pairs",
            str(args.eval_train_pairs),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--eval-pairs",
            str(args.eval_pairs),
            "--host-memory-limit",
            str(args.host_memory_limit),
            "--history-seconds",
            str(args.history_seconds),
            "--target-tail-seconds",
            str(args.target_tail_seconds),
            "--max-context-gate-adjustment",
            str(args.max_context_gate_adjustment),
            "--max-stream-to-lexical-rms-ratio",
            str(args.max_stream_to_lexical_rms_ratio),
        ]
    )
    if args.resume_checkpoint is not None:
        command.extend(["--resume-checkpoint", str(args.resume_checkpoint.resolve())])
    if args.reset_optimizer_on_resume:
        command.append("--reset-optimizer-on-resume")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in selected)
    environment["NO_TORCH_COMPILE"] = "1"
    environment["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    transport = {"status": "not_run", "selected_mode": None, "attempts": []}
    if args.execute and effective_world_size > 1:
        transport, transport_overrides = select_nccl_transport(
            environment=environment,
            world_size=effective_world_size,
            timeout_seconds=args.transport_probe_timeout_seconds,
        )
        environment.update(transport_overrides)
        write_json(run_root / "transport_preflight.json", transport)
    launch = {
        "schema_version": 4,
        "kind": "personaplex-semantic-control-stream-launch",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "selected_physical_gpus": selected,
        "effective_world_size": effective_world_size,
        "host_memory_at_admission": host,
        "transport_preflight": transport,
        "execute": args.execute,
    }
    write_json(run_root / "launch.json", launch)
    if not args.execute:
        print(json.dumps({"status": "staged", "command": command}))
        return 0
    if transport["status"] != "passed":
        print(json.dumps({"status": "refused", "reason": "no NCCL transport passed preflight"}))
        return 2
    process = subprocess.Popen(command, cwd=str(ROOT), env=environment)
    samples = run_root / "resource_samples.jsonl"
    while process.poll() is None:
        sample = {
            "at": datetime.now(timezone.utc).isoformat(),
            "host": host_memory_snapshot(),
            "gpus": [
                {
                    "index": gpu.index,
                    "uuid": gpu.uuid,
                    "memory_total_mib": gpu.memory_total_mib,
                    "memory_used_mib": gpu.memory_used_mib,
                    "utilization_pct": gpu.utilization_pct,
                }
                for gpu in query_gpus()
                if gpu.index in selected
            ],
        }
        with samples.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        time.sleep(args.sample_interval_seconds)
    return int(process.returncode or 0)


if __name__ == "__main__":
    raise SystemExit(main())
