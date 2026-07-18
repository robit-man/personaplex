"""Fail-closed GPU admission for PersonaPlex training runs.

The launcher never assumes that a large GPU is available just because it exists.
It measures utilization and free memory immediately before launch and emits an
immutable report that is retained with the training artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Iterable


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    memory_total_mib: int
    memory_used_mib: int
    utilization_pct: int

    @property
    def free_gib(self) -> float:
        return (self.memory_total_mib - self.memory_used_mib) / 1024


class GPUAdmissionError(RuntimeError):
    """No safe GPU set exists for the requested run."""


def query_gpus() -> list[GPU]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    gpus = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise GPUAdmissionError(f"unexpected nvidia-smi row: {line!r}")
        gpus.append(
            GPU(
                index=int(fields[0]),
                uuid=fields[1],
                name=fields[2],
                memory_total_mib=int(fields[3]),
                memory_used_mib=int(fields[4]),
                utilization_pct=int(fields[5]),
            )
        )
    if not gpus:
        raise GPUAdmissionError("nvidia-smi returned no GPUs")
    return gpus


def admit_gpus(
    *,
    world_size: int,
    min_free_gib: float,
    reserve_gib: float | None,
    reserve_ratio: float,
    max_utilization_pct: int,
    allowed_indices: Iterable[int] | None = None,
) -> dict:
    if world_size < 1:
        raise ValueError("world_size must be at least one")
    if min_free_gib <= 0 or (reserve_gib is not None and reserve_gib < 0):
        raise ValueError("memory thresholds must be non-negative and min_free_gib positive")
    if not 0 <= reserve_ratio < 1:
        raise ValueError("reserve_ratio must be between zero and one")
    if not 0 <= max_utilization_pct <= 100:
        raise ValueError("max_utilization_pct must be between 0 and 100")
    allowed = set(allowed_indices) if allowed_indices is not None else None
    candidates = []
    decisions = []
    for gpu in query_gpus():
        effective_reserve_gib = max(reserve_gib or 0.0, gpu.memory_total_mib / 1024 * reserve_ratio)
        after_reserve = gpu.free_gib - effective_reserve_gib
        reasons = []
        if allowed is not None and gpu.index not in allowed:
            reasons.append("not_in_allowlist")
        if gpu.utilization_pct > max_utilization_pct:
            reasons.append("utilization_above_limit")
        if after_reserve < min_free_gib:
            reasons.append("insufficient_free_memory_after_reserve")
        decision = {
            **asdict(gpu),
            "free_gib": round(gpu.free_gib, 2),
            "reserve_gib": round(effective_reserve_gib, 2),
            "free_gib_after_reserve": round(after_reserve, 2),
            "accepted": not reasons,
            "reasons": reasons,
        }
        decisions.append(decision)
        if not reasons:
            candidates.append(gpu)
    candidates.sort(key=lambda gpu: (gpu.utilization_pct, -gpu.free_gib, gpu.index))
    selected = candidates[:world_size]
    report = {
        "schema_version": 1,
        "kind": "personaplex-gpu-admission",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "request": {
            "world_size": world_size,
            "min_free_gib": min_free_gib,
            "reserve_gib": reserve_gib,
            "reserve_ratio": reserve_ratio,
            "max_utilization_pct": max_utilization_pct,
            "allowed_indices": sorted(allowed) if allowed is not None else None,
        },
        "gpus": decisions,
        "selected_gpu_indices": [gpu.index for gpu in selected],
        "selected_gpu_uuids": [gpu.uuid for gpu in selected],
        "status": "admitted" if len(selected) == world_size else "refused",
    }
    if report["status"] != "admitted":
        report["refusal"] = (
            f"requested {world_size} GPU(s), but only {len(selected)} satisfied the "
            "utilization and reserved-memory limits"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--min-free-gib", type=float, default=44.0)
    parser.add_argument("--reserve-gib", type=float)
    parser.add_argument("--reserve-ratio", type=float, default=0.10)
    parser.add_argument("--max-utilization-pct", type=int, default=25)
    parser.add_argument("--allow-gpu", action="append", type=int, default=None)
    args = parser.parse_args()
    try:
        report = admit_gpus(
            world_size=args.world_size,
            min_free_gib=args.min_free_gib,
            reserve_gib=args.reserve_gib,
            reserve_ratio=args.reserve_ratio,
            max_utilization_pct=args.max_utilization_pct,
            allowed_indices=args.allow_gpu,
        )
    except (OSError, subprocess.CalledProcessError, ValueError, GPUAdmissionError) as exc:
        report = {
            "schema_version": 1,
            "kind": "personaplex-gpu-admission",
            "status": "refused",
            "error": str(exc),
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "selected_gpu_indices": report.get("selected_gpu_indices", [])}))
    return 0 if report["status"] == "admitted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
