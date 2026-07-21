#!/usr/bin/env python3
"""Certify causal branch distinction in every field-slotted ARC-4 pair."""

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

from personaplex_control.arc4_packing import (  # noqa: E402
    ARC4_FIELD_ORDER,
    ARC4_FIELD_SLOTS_PACKING_REVISION,
    field_frame_allocation,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[round((len(values) - 1) * fraction)]


def summarize(values: list[tuple[bool, float]]) -> dict[str, float | int]:
    deltas = [value for _identical, value in values]
    identical = sum(is_identical for is_identical, _value in values)
    return {
        "distinctPairs": len(values) - identical,
        "identicalPairs": identical,
        "minimum": min(deltas),
        "p05": percentile(deltas, 0.05),
        "median": percentile(deltas, 0.50),
        "p95": percentile(deltas, 0.95),
        "maximum": max(deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--arc4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=96)
    args = parser.parse_args()

    allocations = field_frame_allocation(args.frames)
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
        if binding.get("packing_revision") != ARC4_FIELD_SLOTS_PACKING_REVISION:
            raise ValueError(f"{example_id}: stale field-slot packing revision")
        if tuple(binding.get("tensor_shape") or ()) != (args.frames, 4096):
            raise ValueError(f"{example_id}: unexpected field-slot tensor shape")
        shard = (arc4_root / str(binding["shard_path"])).resolve()
        if arc4_root not in shard.parents or not shard.is_file():
            raise ValueError(f"{example_id}: ARC shard escapes or is absent")
        requests.setdefault(shard, []).append((example_id, str(binding["tensor_key"])))

    tensors: dict[str, torch.Tensor] = {}
    for shard, items in requests.items():
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for example_id, tensor_key in items:
                if tensor_key not in available:
                    raise ValueError(f"{example_id}: tensor key is absent from shard")
                tensor = handle.get_tensor(tensor_key).to(dtype=torch.float32)
                if tuple(tensor.shape) != (args.frames, 4096) or not torch.isfinite(tensor).all():
                    raise ValueError(f"{example_id}: invalid field-slot tensor")
                tensors[example_id] = tensor.contiguous()

    bounds: dict[str, tuple[int, int]] = {}
    start = 0
    for name, frames in zip(ARC4_FIELD_ORDER, allocations):
        bounds[name] = (start, start + frames)
        start += frames
    summaries: dict[str, list[tuple[bool, float]]] = {
        name: [] for name in ARC4_FIELD_ORDER
    }
    details = []
    failures = []
    for pair in pairs:
        left = tensors[str(pair["member_a"]["example_id"])]
        right = tensors[str(pair["member_b"]["example_id"])]
        slots: dict[str, dict[str, float | bool]] = {}
        for name, (begin, end) in bounds.items():
            left_slot = left[begin:end]
            right_slot = right[begin:end]
            identical = bool(torch.equal(left_slot, right_slot))
            delta = float(torch.mean((left_slot - right_slot).square()).sqrt().item())
            scale = max(
                float(torch.mean(left_slot.square()).sqrt().item()),
                float(torch.mean(right_slot.square()).sqrt().item()),
                torch.finfo(torch.float32).tiny,
            )
            relative = delta / scale
            if not math.isfinite(relative):
                raise ValueError(f"{pair['pair_id']}: non-finite {name} delta")
            summaries[name].append((identical, relative))
            slots[name] = {"identical": identical, "relativeRmsDelta": relative}
        detail = {"pairId": pair["pair_id"], "split": pair["split"], "slots": slots}
        details.append(detail)
        if slots["decision"]["identical"]:
            failures.append(detail)

    certificate = {
        "schema": "personaplex.arc4-field-slot-certificate.v1",
        "status": "certified" if not failures else "rejected",
        "packingRevision": ARC4_FIELD_SLOTS_PACKING_REVISION,
        "joinedManifest": str(manifest),
        "joinedManifestSha256": file_hash(manifest),
        "pairIndex": str(pair_index),
        "pairIndexSha256": file_hash(pair_index),
        "arc4Root": str(arc4_root),
        "frames": args.frames,
        "allocation": dict(zip(ARC4_FIELD_ORDER, allocations)),
        "pairs": len(pairs),
        "failures": len(failures),
        "summary": {name: summarize(values) for name, values in summaries.items()},
        "failureDetails": failures,
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": certificate["status"],
                "pairs": certificate["pairs"],
                "failures": certificate["failures"],
                "summary": certificate["summary"],
            },
            sort_keys=True,
        )
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
