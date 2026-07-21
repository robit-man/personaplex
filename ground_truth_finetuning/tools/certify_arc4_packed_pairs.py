#!/usr/bin/env python3
"""Certify branch distinction in global ARC-4 frames from joined tensors."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personaplex_control.arc4_packing import (
    ARC4_PACKING_REVISION,
    ARC4_SUPPORTED_PACKING_REVISIONS,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[round((len(values) - 1) * fraction)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--arc4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-frames", type=int, default=2)
    parser.add_argument(
        "--packing-revision",
        choices=ARC4_SUPPORTED_PACKING_REVISIONS,
        default=ARC4_PACKING_REVISION,
    )
    args = parser.parse_args()
    if args.global_frames < 1:
        raise SystemExit("global-frames must be positive")

    manifest = args.manifest.resolve()
    pair_index = args.pair_index.resolve()
    arc4_root = args.arc4_root.resolve()
    rows = {str(row["example_id"]): row for row in load_jsonl(manifest)}
    pairs = load_jsonl(pair_index)
    required_ids = {
        str(pair[key]["example_id"])
        for pair in pairs
        for key in ("member_a", "member_b")
    }
    requests: dict[Path, list[tuple[str, str]]] = {}
    for example_id in required_ids:
        row = rows.get(example_id)
        if row is None:
            raise ValueError(f"pair member is absent from manifest: {example_id}")
        binding = row.get("arc4_reference") or {}
        if binding.get("packing_revision") != args.packing_revision:
            raise ValueError(f"{example_id}: stale ARC packing revision")
        shard = (arc4_root / str(binding["shard_path"])).resolve()
        if arc4_root not in shard.parents or not shard.is_file():
            raise ValueError(f"{example_id}: ARC shard escapes or is absent")
        requests.setdefault(shard, []).append((example_id, str(binding["tensor_key"])))

    global_values: dict[str, torch.Tensor] = {}
    for shard, items in requests.items():
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for example_id, tensor_key in items:
                if tensor_key not in available:
                    raise ValueError(f"{example_id}: tensor key is absent from shard")
                tensor = handle.get_tensor(tensor_key)
                if tensor.ndim != 2 or tensor.shape[0] < args.global_frames:
                    raise ValueError(f"{example_id}: ARC tensor lacks global frames")
                value = tensor[: args.global_frames].to(dtype=torch.float32).contiguous()
                if not torch.isfinite(value).all():
                    raise ValueError(f"{example_id}: ARC global frames are non-finite")
                global_values[example_id] = value

    details = []
    failures = []
    relative_deltas = []
    for pair in pairs:
        left_id = str(pair["member_a"]["example_id"])
        right_id = str(pair["member_b"]["example_id"])
        left = global_values[left_id]
        right = global_values[right_id]
        delta_rms = float(torch.mean((left - right).square()).sqrt().item())
        scale = max(
            float(torch.mean(left.square()).sqrt().item()),
            float(torch.mean(right.square()).sqrt().item()),
            torch.finfo(torch.float32).tiny,
        )
        relative = delta_rms / scale
        failure = not math.isfinite(relative) or torch.equal(left, right)
        detail = {
            "pairId": pair["pair_id"],
            "split": pair["split"],
            "globalFramesIdentical": bool(torch.equal(left, right)),
            "relativeRmsDelta": relative,
        }
        details.append(detail)
        relative_deltas.append(relative)
        if failure:
            failures.append(detail)

    certificate = {
        "schema": "personaplex.arc4-packed-pair-certificate.v1",
        "status": "certified" if not failures else "rejected",
        "packingRevision": args.packing_revision,
        "manifest": str(manifest),
        "manifestSha256": file_hash(manifest),
        "pairIndex": str(pair_index),
        "pairIndexSha256": file_hash(pair_index),
        "arc4Root": str(arc4_root),
        "globalFrames": args.global_frames,
        "pairs": len(pairs),
        "failures": len(failures),
        "relativeRmsDelta": {
            "minimum": min(relative_deltas),
            "p05": percentile(relative_deltas, 0.05),
            "median": percentile(relative_deltas, 0.5),
            "maximum": max(relative_deltas),
        },
        "failureDetails": failures,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: certificate[key] for key in ("status", "pairs", "failures", "relativeRmsDelta")}, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
