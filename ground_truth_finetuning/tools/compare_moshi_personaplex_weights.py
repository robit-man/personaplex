#!/usr/bin/env python3
"""Compare safetensors structures and optional Moshika-to-MoshiRAG deltas."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from safetensors import safe_open


PERSONAPLEX_REVISION = "fdaf4090a61cb315c138a1faee287ffd6c716309"
MOSHIKA_BASE_REVISION = "a49141e28b3d9c947cf9aa5314431e1b11cbd2f5"
MOSHIKA_RAG_REVISION = "7135a6e3c46abb66c2cd95cb04cbfcbe8376f83d"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def resource_root(value: str | None) -> Path:
    configured = (
        value
        or os.environ.get("PERSONAPLEX_SHARED_CACHE_ROOT")
        or os.environ.get("VOXRN_SHARED_CACHE_ROOT")
        or os.environ.get("VOXRN_RESOURCE_ROOT")
        or "/srv/voxrn_cache"
    )
    return Path(configured).expanduser().resolve()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def tensor_scope(key: str) -> str:
    parts = key.split(".")
    if "layers" in parts:
        index = parts.index("layers")
        if len(parts) > index + 1 and parts[index + 1].isdigit():
            return ".".join(parts[: index + 1] + ["*"])
    if len(parts) <= 3:
        return key
    return ".".join(parts[:3])


def read_header(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    tensors: dict[str, dict[str, Any]] = {}
    metadata: dict[str, str] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            view = handle.get_slice(key)
            tensors[key] = {
                "shape": list(view.get_shape()),
                "dtype": str(view.get_dtype()),
            }
    scopes = Counter(tensor_scope(key) for key in tensors)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "metadata": metadata,
        "tensor_count": len(tensors),
        "scopes": dict(sorted(scopes.items())),
        "tensors": tensors,
    }


def same_spec(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["shape"] == right["shape"] and left["dtype"] == right["dtype"]


def compare_headers(
    left_name: str,
    left: dict[str, Any],
    right_name: str,
    right: dict[str, Any],
) -> dict[str, Any]:
    left_tensors = left["tensors"]
    right_tensors = right["tensors"]
    left_keys = set(left_tensors)
    right_keys = set(right_tensors)
    common = sorted(left_keys & right_keys)
    compatible = [
        key
        for key in common
        if same_spec(left_tensors[key], right_tensors[key])
    ]
    mismatched = [
        {
            "key": key,
            left_name: left_tensors[key],
            right_name: right_tensors[key],
        }
        for key in common
        if not same_spec(left_tensors[key], right_tensors[key])
    ]
    return {
        "left": left_name,
        "right": right_name,
        "left_only_count": len(left_keys - right_keys),
        "right_only_count": len(right_keys - left_keys),
        "common_count": len(common),
        "compatible_count": len(compatible),
        "shape_or_dtype_mismatch_count": len(mismatched),
        "left_only": sorted(left_keys - right_keys),
        "right_only": sorted(right_keys - left_keys),
        "shape_or_dtype_mismatches": mismatched,
    }


def keys_matching_prefixes(
    keys: Iterable[str],
    prefixes: list[str],
) -> list[str]:
    if not prefixes:
        return sorted(keys)
    return sorted(
        key for key in keys if any(key.startswith(prefix) for prefix in prefixes)
    )


def numeric_delta(
    base_path: Path,
    rag_path: Path,
    specs: dict[str, dict[str, Any]],
    keys: list[str],
    max_tensors: int,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for --numeric-delta") from exc

    selected = keys[:max_tensors] if max_tensors else keys
    scope_stats: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "tensor_count": 0,
            "changed_tensor_count": 0,
            "numel": 0,
            "base_sum_squares": 0.0,
            "delta_sum_squares": 0.0,
            "max_abs_delta": 0.0,
        }
    )
    top: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    with (
        safe_open(str(base_path), framework="pt", device="cpu") as base,
        safe_open(str(rag_path), framework="pt", device="cpu") as rag,
        torch.no_grad(),
    ):
        for index, key in enumerate(selected, start=1):
            dtype = specs[key]["dtype"].upper()
            if not any(token in dtype for token in ("F16", "F32", "F64", "BF16")):
                skipped.append({"key": key, "reason": f"non-floating dtype {dtype}"})
                continue

            base_tensor = base.get_tensor(key).float()
            rag_tensor = rag.get_tensor(key).float()
            if base_tensor.shape != rag_tensor.shape:
                skipped.append({"key": key, "reason": "shape mismatch"})
                del base_tensor, rag_tensor
                continue

            rag_tensor.sub_(base_tensor)
            numel = rag_tensor.numel()
            base_sum_squares = float(base_tensor.square().sum().item())
            delta_sum_squares = float(rag_tensor.square().sum().item())
            max_abs_delta = float(rag_tensor.abs().max().item()) if numel else 0.0
            base_rms = math.sqrt(base_sum_squares / max(1, numel))
            delta_rms = math.sqrt(delta_sum_squares / max(1, numel))
            relative_rms = delta_rms / max(base_rms, 1e-12)

            scope = tensor_scope(key)
            aggregate = scope_stats[scope]
            aggregate["tensor_count"] += 1
            aggregate["changed_tensor_count"] += int(max_abs_delta > 0.0)
            aggregate["numel"] += numel
            aggregate["base_sum_squares"] += base_sum_squares
            aggregate["delta_sum_squares"] += delta_sum_squares
            aggregate["max_abs_delta"] = max(
                float(aggregate["max_abs_delta"]),
                max_abs_delta,
            )
            top.append(
                {
                    "key": key,
                    "numel": numel,
                    "base_rms": base_rms,
                    "delta_rms": delta_rms,
                    "relative_rms": relative_rms,
                    "max_abs_delta": max_abs_delta,
                }
            )
            del base_tensor, rag_tensor
            if index % 100 == 0:
                print(
                    json.dumps(
                        {
                            "event": "numeric_delta_progress",
                            "processed": index,
                            "total": len(selected),
                        }
                    ),
                    flush=True,
                )

    finalized_scopes: dict[str, dict[str, Any]] = {}
    for scope, values in sorted(scope_stats.items()):
        numel = int(values["numel"])
        base_rms = math.sqrt(float(values["base_sum_squares"]) / max(1, numel))
        delta_rms = math.sqrt(float(values["delta_sum_squares"]) / max(1, numel))
        finalized_scopes[scope] = {
            "tensor_count": int(values["tensor_count"]),
            "changed_tensor_count": int(values["changed_tensor_count"]),
            "numel": numel,
            "base_rms": base_rms,
            "delta_rms": delta_rms,
            "relative_rms": delta_rms / max(base_rms, 1e-12),
            "max_abs_delta": float(values["max_abs_delta"]),
        }

    top.sort(key=lambda item: item["relative_rms"], reverse=True)
    return {
        "selected_tensor_count": len(selected),
        "evaluated_tensor_count": sum(
            int(scope["tensor_count"]) for scope in finalized_scopes.values()
        ),
        "truncated": bool(max_tensors and len(keys) > max_tensors),
        "prefixes": [],
        "scopes": finalized_scopes,
        "top_relative_rms": top[:100],
        "skipped": skipped,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare raw PersonaPlex, Moshika base, and MoshiRAG safetensors "
            "without loading model weights by default."
        )
    )
    parser.add_argument("--resource-root")
    parser.add_argument("--personaplex", type=Path)
    parser.add_argument("--moshika-base", type=Path)
    parser.add_argument("--moshirag", type=Path)
    parser.add_argument("--arc4", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--numeric-delta",
        action="store_true",
        help="Measure Moshika-base to MoshiRAG tensor deltas one tensor at a time.",
    )
    parser.add_argument(
        "--numeric-prefix",
        action="append",
        default=[],
        help="Restrict numeric deltas to key prefixes; repeat as needed.",
    )
    parser.add_argument(
        "--max-numeric-tensors",
        type=int,
        default=0,
        help="Limit numeric tensors after prefix filtering; zero means all.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = resource_root(args.resource_root)
    personaplex_path = (
        args.personaplex.expanduser().resolve()
        if args.personaplex
        else root
        / "huggingface"
        / "nvidia"
        / "personaplex-7b-v1"
        / PERSONAPLEX_REVISION
        / "model.safetensors"
    )
    base_path = (
        args.moshika_base.expanduser().resolve()
        if args.moshika_base
        else root
        / "huggingface"
        / "kyutai"
        / "moshika-pytorch-bf16"
        / MOSHIKA_BASE_REVISION
        / "model.safetensors"
    )
    rag_path = (
        args.moshirag.expanduser().resolve()
        if args.moshirag
        else root
        / "huggingface"
        / "kyutai"
        / "moshika-rag-pytorch-bf16"
        / MOSHIKA_RAG_REVISION
        / "model.safetensors"
    )
    arc_path = args.arc4.expanduser().resolve() if args.arc4 else None
    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else root
        / "personaplex"
        / "compatibility"
        / "moshirag-personaplex-v1.json"
    )
    if args.max_numeric_tensors < 0:
        raise ValueError("--max-numeric-tensors cannot be negative")

    print(json.dumps({"event": "read_header", "model": "personaplex"}), flush=True)
    personaplex = read_header(personaplex_path)
    print(json.dumps({"event": "read_header", "model": "moshika_base"}), flush=True)
    base = read_header(base_path)
    print(json.dumps({"event": "read_header", "model": "moshirag"}), flush=True)
    rag = read_header(rag_path)
    arc = read_header(arc_path) if arc_path and arc_path.is_file() else None

    base_rag = compare_headers("moshika_base", base, "moshirag", rag)
    base_personaplex = compare_headers(
        "moshika_base", base, "personaplex", personaplex
    )
    rag_personaplex = compare_headers(
        "moshirag", rag, "personaplex", personaplex
    )

    base_tensors = base["tensors"]
    rag_tensors = rag["tensors"]
    personaplex_tensors = personaplex["tensors"]
    common_backbone = sorted(set(base_tensors) & set(rag_tensors))
    rag_delta_applicable = [
        key
        for key in common_backbone
        if key in personaplex_tensors
        and same_spec(base_tensors[key], rag_tensors[key])
        and same_spec(rag_tensors[key], personaplex_tensors[key])
    ]
    rag_only = sorted(set(rag_tensors) - set(base_tensors))
    conditioner_candidates = [
        key
        for key in rag_only
        if any(
            marker in key.lower()
            for marker in ("condition", "reference", "arc", "rag")
        )
    ]

    payload: dict[str, Any] = {
        "schema": "personaplex.moshirag-weight-compatibility.v1",
        "created_at": utc_now(),
        "resource_root": str(root),
        "models": {
            "personaplex": {
                key: value for key, value in personaplex.items() if key != "tensors"
            },
            "moshika_base": {
                key: value for key, value in base.items() if key != "tensors"
            },
            "moshirag": {
                key: value for key, value in rag.items() if key != "tensors"
            },
            "arc4": (
                {key: value for key, value in arc.items() if key != "tensors"}
                if arc
                else None
            ),
        },
        "comparisons": {
            "moshika_base_to_moshirag": base_rag,
            "moshika_base_to_personaplex": base_personaplex,
            "moshirag_to_personaplex": rag_personaplex,
        },
        "transfer": {
            "rag_only_tensor_count": len(rag_only),
            "rag_only_tensors": rag_only,
            "conditioner_candidate_count": len(conditioner_candidates),
            "conditioner_candidates": conditioner_candidates,
            "common_backbone_tensor_count": len(common_backbone),
            "rag_delta_applicable_to_personaplex_count": len(
                rag_delta_applicable
            ),
            "rag_delta_applicable_to_personaplex": rag_delta_applicable,
        },
    }

    if args.numeric_delta:
        numeric_keys = [
            key
            for key in keys_matching_prefixes(
                rag_delta_applicable,
                args.numeric_prefix,
            )
            if same_spec(base_tensors[key], rag_tensors[key])
        ]
        numeric = numeric_delta(
            base_path,
            rag_path,
            base_tensors,
            numeric_keys,
            args.max_numeric_tensors,
        )
        numeric["prefixes"] = args.numeric_prefix
        payload["numeric_base_to_rag_delta"] = numeric

    atomic_json(output_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "rag_only": len(rag_only),
                "conditioner_candidates": len(conditioner_candidates),
                "rag_delta_applicable_to_personaplex": len(
                    rag_delta_applicable
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
