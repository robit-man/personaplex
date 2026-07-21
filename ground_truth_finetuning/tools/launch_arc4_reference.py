#!/usr/bin/env python3
"""Discover resources, certify NCCL routing, and launch ARC-4 training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.gpu_admission import admit_gpus  # noqa: E402
from ground_truth_finetuning.training.arc4_two_path import TWO_PATH_ARC4_ARCHITECTURE  # noqa: E402


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def host_memory_used_fraction() -> float:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        name, raw, *_rest = line.split()
        values[name.rstrip(":")] = int(raw)
    return 1.0 - values["MemAvailable"] / values["MemTotal"]


def collective_environment(route: str) -> dict[str, str]:
    values = {
        "NCCL_IB_DISABLE": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    }
    if route == "shm":
        values.update({"NCCL_P2P_DISABLE": "1", "NCCL_CUMEM_ENABLE": "0"})
    return values


def probe_collective(
    python: Path,
    gpu_indices: list[int],
    route: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(collective_environment(route))
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_indices))
    command = [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={len(gpu_indices)}",
        str(ROOT / "ground_truth_finetuning/tools/probe_nccl_collective.py"),
    ]
    try:
        result = subprocess.run(
            command,
            env=environment,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        return {
            "route": route,
            "passed": result.returncode == 0,
            "returnCode": result.returncode,
            "stdoutTail": result.stdout[-2000:],
            "stderrTail": result.stderr[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "route": route,
            "passed": False,
            "timeout": True,
            "stdoutTail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderrTail": (exc.stderr or "")[-4000:] if isinstance(exc.stderr, str) else "",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--pair-certificate", type=Path, required=True)
    parser.add_argument("--packed-pair-certificate", type=Path, required=True)
    parser.add_argument("--field-slot-certificate", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--arc4-root", type=Path, required=True)
    parser.add_argument("--decision-manifest", type=Path)
    parser.add_argument("--decision-certificate", type=Path)
    parser.add_argument("--decision-arc4-root", type=Path)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-optimizer-state", action="store_true")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--allow-gpu", action="append", type=int)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--eval-pairs", type=int, default=32)
    parser.add_argument("--context-frames", type=int, default=64)
    parser.add_argument("--post-target-tail-frames", type=int, default=16)
    parser.add_argument("--adapter-rank", type=int, default=256)
    parser.add_argument("--adapter-architecture", default="arc4-field-layerwise-adapted-v5")
    parser.add_argument("--initial-gate", type=float, default=0.25)
    parser.add_argument("--max-stream-rms", type=float, default=0.35)
    parser.add_argument("--output-stream-frames", type=int, default=128)
    parser.add_argument("--layer-control-count", type=int, default=8)
    parser.add_argument("--layer-initial-gate", type=float, default=0.05)
    parser.add_argument("--max-layer-rms", type=float, default=0.15)
    parser.add_argument("--layer-adaptation-rank", type=int, default=128)
    parser.add_argument("--layer-adaptation-initial-gate", type=float, default=0.10)
    parser.add_argument("--max-layer-adaptation-rms", type=float, default=0.10)
    parser.add_argument("--two-path-decision-frames", type=int, default=48)
    parser.add_argument("--two-path-stream-frames", type=int, default=96)
    parser.add_argument("--two-path-prefix-tokens", type=int, default=8)
    parser.add_argument("--two-path-rank", type=int, default=64)
    parser.add_argument("--two-path-attention-heads", type=int, default=8)
    parser.add_argument("--two-path-prefix-gate", type=float, default=0.05)
    parser.add_argument("--two-path-max-prefix-rms", type=float, default=0.08)
    parser.add_argument("--two-path-stream-scale", type=float, default=1.0)
    parser.add_argument("--two-path-max-stream-residual-rms", type=float, default=0.0)
    parser.add_argument("--task-vector-base", type=Path)
    parser.add_argument("--task-vector-rag", type=Path)
    parser.add_argument("--task-vector-import-report", type=Path)
    parser.add_argument("--task-vector-temporal-alpha", type=float, default=0.0)
    parser.add_argument("--temporal-lora-layers", type=int, default=0)
    parser.add_argument("--temporal-lora-rank", type=int, default=8)
    parser.add_argument("--temporal-lora-alpha", type=float, default=16.0)
    parser.add_argument("--temporal-lora-learning-rate", type=float)
    parser.add_argument("--freeze-control-adapter", action="store_true")
    parser.add_argument("--freeze-temporal-lora", action="store_true")
    parser.add_argument("--counterfactual-margin", type=float, default=0.08)
    parser.add_argument("--focused-counterfactual-margin", type=float, default=0.30)
    parser.add_argument("--causal-weight", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--matched-weight", type=float, default=1.0)
    parser.add_argument("--whole-causal-weight", type=float, default=1.0)
    parser.add_argument("--focused-causal-weight", type=float, default=1.0)
    parser.add_argument("--null-weight", type=float, default=0.25)
    parser.add_argument("--stale-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--contrastive-whole-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-focused-weight", type=float, default=2.0)
    parser.add_argument("--coverage-surrogate-weight", type=float, default=0.0)
    parser.add_argument("--coverage-surrogate-temperature", type=float, default=0.25)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--hard-curriculum", type=Path)
    parser.add_argument("--hard-replay-ratio", type=float, default=0.0)
    parser.add_argument("--train-eval-pairs", type=int, default=32)
    parser.add_argument("--pair-worst-direction-weight", type=float, default=0.0)
    parser.add_argument("--pair-worst-temperature", type=float, default=0.05)
    parser.add_argument("--max-host-memory-fraction", type=float, default=0.80)
    parser.add_argument("--weight-residency-multiplier", type=float, default=1.35)
    parser.add_argument("--max-existing-gpu-utilization-pct", type=int, default=25)
    parser.add_argument("--collective-route", choices=("auto", "p2p", "shm"), default="auto")
    parser.add_argument("--collective-probe-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    required_files = [
        args.manifest,
        args.certificate,
        args.pair_index,
        args.pair_certificate,
        args.packed_pair_certificate,
        args.field_slot_certificate,
        args.model_contract,
        args.moshi_path,
        args.python,
    ]
    if any(not path.resolve().is_file() for path in required_files):
        raise SystemExit("one or more required ARC-4 launch files are missing")
    if args.adapter_architecture == TWO_PATH_ARC4_ARCHITECTURE:
        two_path_files = (
            args.decision_manifest,
            args.decision_certificate,
            args.task_vector_base,
            args.task_vector_rag,
            args.task_vector_import_report,
        )
        if any(path is None or not path.resolve().is_file() for path in two_path_files):
            raise SystemExit("two-path ARC launch files are incomplete")
        if args.decision_arc4_root is None or not args.decision_arc4_root.resolve().is_dir():
            raise SystemExit("two-path decision ARC root is missing")
    if args.temporal_lora_layers < 0:
        raise SystemExit("temporal-lora-layers cannot be negative")
    if args.temporal_lora_layers and args.adapter_architecture != TWO_PATH_ARC4_ARCHITECTURE:
        raise SystemExit("temporal LoRA requires two-path ARC conditioning")
    if args.freeze_control_adapter and not args.temporal_lora_layers:
        raise SystemExit("freezing the control adapter requires temporal LoRA")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.resolve().is_file():
        raise SystemExit("ARC-4 resume checkpoint is missing")
    if args.hard_curriculum is not None and not args.hard_curriculum.resolve().is_file():
        raise SystemExit("ARC-4 hard curriculum is missing")
    if args.run_dir.exists():
        raise SystemExit(f"refusing existing run directory: {args.run_dir}")
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    if certificate.get("status") != "certified_for_arc4_adapter_training":
        raise SystemExit("ARC-4 certificate does not authorize training")
    host_used = host_memory_used_fraction()
    if host_used >= args.max_host_memory_fraction:
        raise SystemExit(
            f"host memory admission refused at {host_used:.3f} >= {args.max_host_memory_fraction:.3f}"
        )

    gib = 1024**3
    resident_weight_gib = (
        args.moshi_path.stat().st_size
    ) / gib
    derived_free_gib = resident_weight_gib * args.weight_residency_multiplier
    report = admit_gpus(
        world_size=args.world_size,
        min_free_gib=derived_free_gib,
        reserve_gib=None,
        reserve_ratio=0.0,
        max_utilization_pct=args.max_existing_gpu_utilization_pct,
        allowed_indices=args.allow_gpu,
    )
    if report.get("status") != "admitted":
        print(json.dumps({"status": "refused", "gpuAdmission": report}, sort_keys=True))
        return 2
    selected = [int(value) for value in report["selected_gpu_indices"]]

    probes: list[dict[str, Any]] = []
    if args.collective_route == "auto":
        probes.append(probe_collective(args.python.absolute(), selected, "p2p", args.collective_probe_timeout_seconds))
        route = "p2p" if probes[-1]["passed"] else "shm"
        if route == "shm":
            probes.append(probe_collective(args.python.absolute(), selected, "shm", args.collective_probe_timeout_seconds))
    else:
        route = args.collective_route
        probes.append(probe_collective(args.python.absolute(), selected, route, args.collective_probe_timeout_seconds))
    if not probes[-1]["passed"]:
        raise SystemExit(f"NCCL collective admission failed for route {route}")

    trainer = ROOT / "ground_truth_finetuning/tools/train_arc4_causal.py"
    command = [
        str(args.python.absolute()), "-m", "torch.distributed.run", "--standalone",
        f"--nproc-per-node={args.world_size}", str(trainer),
        "--manifest", str(args.manifest.resolve()),
        "--certificate", str(args.certificate.resolve()),
        "--pair-index", str(args.pair_index.resolve()),
        "--pair-certificate", str(args.pair_certificate.resolve()),
        "--packed-pair-certificate", str(args.packed_pair_certificate.resolve()),
        "--field-slot-certificate", str(args.field_slot_certificate.resolve()),
        "--artifact-root", str(args.artifact_root.resolve()),
        "--arc4-root", str(args.arc4_root.resolve()),
        "--model-contract", str(args.model_contract.resolve()),
        "--moshi-source-root", str(args.moshi_source_root.resolve()),
        "--moshi-path", str(args.moshi_path.resolve()),
        "--run-dir", str(args.run_dir.resolve()),
        "--max-steps", str(args.max_steps),
        "--checkpoint-every", str(args.checkpoint_every),
        "--eval-pairs", str(args.eval_pairs),
        "--context-frames", str(args.context_frames),
        "--post-target-tail-frames", str(args.post_target_tail_frames),
        "--adapter-rank", str(args.adapter_rank),
        "--adapter-architecture", str(args.adapter_architecture),
        "--initial-gate", str(args.initial_gate),
        "--max-stream-rms", str(args.max_stream_rms),
        "--output-stream-frames", str(args.output_stream_frames),
        "--layer-control-count", str(args.layer_control_count),
        "--layer-initial-gate", str(args.layer_initial_gate),
        "--max-layer-rms", str(args.max_layer_rms),
        "--layer-adaptation-rank", str(args.layer_adaptation_rank),
        "--layer-adaptation-initial-gate", str(args.layer_adaptation_initial_gate),
        "--max-layer-adaptation-rms", str(args.max_layer_adaptation_rms),
        "--two-path-decision-frames", str(args.two_path_decision_frames),
        "--two-path-stream-frames", str(args.two_path_stream_frames),
        "--two-path-prefix-tokens", str(args.two_path_prefix_tokens),
        "--two-path-rank", str(args.two_path_rank),
        "--two-path-attention-heads", str(args.two_path_attention_heads),
        "--two-path-prefix-gate", str(args.two_path_prefix_gate),
        "--two-path-max-prefix-rms", str(args.two_path_max_prefix_rms),
        "--two-path-stream-scale", str(args.two_path_stream_scale),
        "--two-path-max-stream-residual-rms", str(args.two_path_max_stream_residual_rms),
        "--temporal-lora-layers", str(args.temporal_lora_layers),
        "--temporal-lora-rank", str(args.temporal_lora_rank),
        "--temporal-lora-alpha", str(args.temporal_lora_alpha),
        "--counterfactual-margin", str(args.counterfactual_margin),
        "--focused-counterfactual-margin", str(args.focused_counterfactual_margin),
        "--causal-weight", str(args.causal_weight),
        "--learning-rate", str(args.learning_rate),
        "--matched-weight", str(args.matched_weight),
        "--whole-causal-weight", str(args.whole_causal_weight),
        "--focused-causal-weight", str(args.focused_causal_weight),
        "--null-weight", str(args.null_weight),
        "--stale-weight", str(args.stale_weight),
        "--contrastive-weight", str(args.contrastive_weight),
        "--contrastive-temperature", str(args.contrastive_temperature),
        "--contrastive-whole-weight", str(args.contrastive_whole_weight),
        "--contrastive-focused-weight", str(args.contrastive_focused_weight),
        "--coverage-surrogate-weight", str(args.coverage_surrogate_weight),
        "--coverage-surrogate-temperature", str(args.coverage_surrogate_temperature),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--shuffle-seed", str(args.shuffle_seed),
        "--hard-replay-ratio", str(args.hard_replay_ratio),
        "--train-eval-pairs", str(args.train_eval_pairs),
        "--pair-worst-direction-weight", str(args.pair_worst_direction_weight),
        "--pair-worst-temperature", str(args.pair_worst_temperature),
        "--max-host-memory-fraction", str(args.max_host_memory_fraction),
    ]
    if args.adapter_architecture == TWO_PATH_ARC4_ARCHITECTURE:
        assert args.decision_manifest is not None
        assert args.decision_certificate is not None
        assert args.decision_arc4_root is not None
        assert args.task_vector_base is not None
        assert args.task_vector_rag is not None
        assert args.task_vector_import_report is not None
        command.extend(
            (
                "--decision-manifest", str(args.decision_manifest.resolve()),
                "--decision-certificate", str(args.decision_certificate.resolve()),
                "--decision-arc4-root", str(args.decision_arc4_root.resolve()),
                "--task-vector-base", str(args.task_vector_base.resolve()),
                "--task-vector-rag", str(args.task_vector_rag.resolve()),
                "--task-vector-import-report", str(args.task_vector_import_report.resolve()),
                "--task-vector-temporal-alpha", str(args.task_vector_temporal_alpha),
            )
        )
    if args.resume_checkpoint is not None:
        command.extend(("--resume-checkpoint", str(args.resume_checkpoint.resolve())))
    if args.resume_optimizer_state:
        command.append("--resume-optimizer-state")
    if args.temporal_lora_learning_rate is not None:
        command.extend(
            ("--temporal-lora-learning-rate", str(args.temporal_lora_learning_rate))
        )
    if args.freeze_control_adapter:
        command.append("--freeze-control-adapter")
    if args.freeze_temporal_lora:
        command.append("--freeze-temporal-lora")
    if args.hard_curriculum is not None:
        command.extend(("--hard-curriculum", str(args.hard_curriculum.resolve())))
    if args.evaluation_only:
        command.append("--evaluation-only")
    launch_record = {
        "schema": "personaplex.arc4-training-launch.v1",
        "execute": args.execute,
        "hostMemoryUsedFraction": host_used,
        "derivedMinimumFreeGiB": derived_free_gib,
        "gpuAdmission": report,
        "collectiveRoute": route,
        "collectiveProbes": probes,
        "command": command,
    }
    record_path = args.run_dir.parent / f"{args.run_dir.name}.launch.json"
    write_json(record_path, launch_record)
    if not args.execute:
        print(json.dumps({"status": "staged", "launchRecord": str(record_path), "route": route}))
        return 0
    environment = os.environ.copy()
    environment.update(collective_environment(route))
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected))
    return subprocess.run(command, cwd=ROOT, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
