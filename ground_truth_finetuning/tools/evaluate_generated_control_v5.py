#!/usr/bin/env python3
"""Run the held-out, free-running generated semantic-control v5 evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.evaluation.generated_control_eval_v5 import (
    ALLOWED_PHYSICAL_CUDA_DEVICES,
    CudaAdmission,
    EvaluationConfig,
    EvaluationContractError,
    GeneratedControlEvaluationHarness,
    HostRamAdmission,
    sha256_path,
)


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationContractError(f"cannot load JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationContractError(f"{path} must contain one JSON object")
    return value


def load_jsonl_bytes(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EvaluationContractError(f"cannot load split JSONL {path}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationContractError(
                f"{path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise EvaluationContractError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    if not rows:
        raise EvaluationContractError(f"held-out split is empty: {path}")
    return rows, payload


def load_factory(specification: str):
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise EvaluationContractError("adapter factory must use module.path:callable syntax")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise EvaluationContractError(
            f"cannot import adapter factory {specification}: {exc}"
        ) from exc
    if not callable(factory):
        raise EvaluationContractError(f"adapter factory {specification} is not callable")
    return factory


def build_adapters(
    specification: str, *, checkpoint: Path, checkpoint_sha256: str
) -> tuple[Any, Any, Any]:
    factory = load_factory(specification)
    context = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "physical_cuda_devices": list(ALLOWED_PHYSICAL_CUDA_DEVICES),
        "cpu_model_fallback": False,
        "require_cuda_attestation": True,
        "generation_mode": "free_running",
    }
    adapters = factory(context)
    if isinstance(adapters, Mapping):
        generator = adapters.get("generator")
        asr = adapters.get("asr")
        judge = adapters.get("judge")
    elif isinstance(adapters, (tuple, list)) and len(adapters) == 3:
        generator, asr, judge = adapters
    else:
        raise EvaluationContractError(
            "adapter factory must return {generator, asr, judge} or a three-item tuple"
        )
    if generator is None or asr is None or judge is None:
        raise EvaluationContractError("adapter factory omitted generator, ASR, or judge")
    return generator, asr, judge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-sha256", required=True)
    parser.add_argument("--split", type=Path, required=True, help="Held-out four-sibling JSONL")
    parser.add_argument("--split-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--adapter-factory", required=True, help="module.path:callable")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--minimum-free-cuda-gib", type=float, default=0.001)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        checkpoint_hash = sha256_path(args.checkpoint)
        dataset_hash = sha256_path(args.dataset)
        split_hash = sha256_path(args.split)
        for label, observed, expected in (
            ("checkpoint", checkpoint_hash, args.checkpoint_sha256),
            ("dataset", dataset_hash, args.dataset_sha256),
            ("split", split_hash, args.split_sha256),
        ):
            if observed != expected:
                raise EvaluationContractError(
                    f"{label} hash mismatch: expected {expected}, observed {observed}"
                )
        preregistration = load_json(args.preregistration)
        config = EvaluationConfig.from_mapping(
            preregistration,
            checkpoint_sha256=checkpoint_hash,
            dataset_sha256=dataset_hash,
            split_sha256=split_hash,
        )
        rows, split_bytes = load_jsonl_bytes(args.split)
        dataset_bytes = args.dataset.read_bytes() if args.dataset.is_file() else None
        if dataset_bytes is None:
            raise EvaluationContractError(
                "--dataset must be one exact manifest file; directory hashes cannot bind row bytes"
            )
        generator, asr, judge = build_adapters(
            args.adapter_factory,
            checkpoint=args.checkpoint,
            checkpoint_sha256=checkpoint_hash,
        )
        cuda = CudaAdmission(
            minimum_free_memory_bytes=max(
                1, int(args.minimum_free_cuda_gib * 1024 * 1024 * 1024)
            )
        )
        host = HostRamAdmission(maximum_used_fraction=config.host_ram_used_limit)
        report = GeneratedControlEvaluationHarness(
            config=config,
            generator=generator,
            asr=asr,
            judge=judge,
            output_dir=args.output_dir,
            device_admission=cuda,
            host_ram_admission=host,
            dataset_root=args.dataset_root or args.split.parent,
            resume=not args.no_resume,
        ).run(rows, dataset_bytes=dataset_bytes, split_bytes=split_bytes)
    except (EvaluationContractError, RuntimeError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 2
    summary = report["summary"]
    print(
        json.dumps(
            {
                "status": summary["status"],
                "promotion_eligible": summary["promotion_eligible"],
                "overall": summary["overall"],
                "summary_id": summary["summary_id"],
                "manifest_id": report["manifest"]["manifest_id"],
                "failure_reasons": summary["failure_reasons"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["promotion_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
