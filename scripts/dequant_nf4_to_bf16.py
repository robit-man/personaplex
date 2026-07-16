#!/usr/bin/env python3
"""Convert PersonaPlex NF4 safetensors to a bf16 safetensors cache."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time

import torch
from safetensors import safe_open


META_SUFFIXES = (".__scales__", ".__shape__", ".__numel__")
GROUP_SIZE = 64
BF16_BYTES = 2


def is_cache_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
    except Exception:
        return False
    if not keys:
        return False
    return not any(key.endswith(META_SUFFIXES) for key in keys)


def dequant_nf4_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: torch.Tensor,
    numel: torch.Tensor,
    device: str,
) -> torch.Tensor:
    packed = packed.to(device=device)
    scales = scales.to(device=device, dtype=torch.float32)
    numel_int = int(numel.item())
    orig_shape = [int(dim) for dim in shape.tolist() if int(dim) > 0]

    lo = (packed & 0x0F).to(torch.int8) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int8) - 8
    unpacked = torch.empty(packed.numel() * 2, dtype=torch.float32, device=device)
    unpacked[0::2] = lo.float()
    unpacked[1::2] = hi.float()

    n_groups = scales.numel()
    groups = unpacked[: n_groups * GROUP_SIZE].reshape(n_groups, GROUP_SIZE)
    deq = (groups * scales.unsqueeze(1)).reshape(-1)[:numel_int]
    return deq.reshape(orig_shape).to(torch.bfloat16).cpu()


def tensor_shape(handle, name: str) -> list[int]:
    if hasattr(handle, "get_slice"):
        return [int(dim) for dim in handle.get_slice(name).get_shape()]
    return [int(dim) for dim in handle.get_tensor(name).shape]


def product(values: list[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


def build_header(input_path: Path) -> tuple[bytes, list[str], int]:
    entries: list[str] = []
    header: dict[str, dict[str, object]] = {}
    offset = 0

    with safe_open(input_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for name in handle.keys():
            if name.endswith(META_SUFFIXES):
                continue

            if f"{name}.__scales__" in keys:
                shape = [
                    int(dim)
                    for dim in handle.get_tensor(f"{name}.__shape__").tolist()
                    if int(dim) > 0
                ]
            else:
                shape = tensor_shape(handle, name)

            nbytes = product(shape) * BF16_BYTES
            header[name] = {
                "dtype": "BF16",
                "shape": shape,
                "data_offsets": [offset, offset + nbytes],
            }
            offset += nbytes
            entries.append(name)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - ((8 + len(header_bytes)) % 8)) % 8
    header_bytes += b" " * padding
    return header_bytes, entries, offset


def write_tensor(output, tensor: torch.Tensor) -> int:
    tensor = tensor.contiguous()
    raw = tensor.view(torch.uint8).numpy()
    output.write(memoryview(raw))
    return int(raw.nbytes)


def convert(input_path: Path, output_path: Path, device: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    processed: set[str] = set()

    started = time.time()
    header_bytes, entries, total_data_bytes = build_header(input_path)
    total = len(entries)

    with tmp_path.open("wb") as output:
        output.write(struct.pack("<Q", len(header_bytes)))
        output.write(header_bytes)

        with safe_open(input_path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            for name in entries:
                scales_key = f"{name}.__scales__"
                if scales_key in keys:
                    tensor = dequant_nf4_tensor(
                        handle.get_tensor(name),
                        handle.get_tensor(scales_key),
                        handle.get_tensor(f"{name}.__shape__"),
                        handle.get_tensor(f"{name}.__numel__"),
                        device,
                    )
                else:
                    tensor = handle.get_tensor(name).to(torch.bfloat16).cpu()

                write_tensor(output, tensor)
                processed.add(name)
                del tensor

                if len(processed) == 1 or len(processed) % 25 == 0 or len(processed) == total:
                    elapsed = time.time() - started
                    print(f"dequantized and wrote {len(processed)}/{total} tensors in {elapsed:.1f}s", flush=True)

        expected_size = 8 + len(header_bytes) + total_data_bytes
        actual_size = output.tell()
        if actual_size != expected_size:
            raise RuntimeError(f"wrote {actual_size} bytes, expected {expected_size}")

    os.replace(tmp_path, output_path)
    size_gb = output_path.stat().st_size / 1024**3
    print(f"wrote {output_path} ({size_gb:.2f} GiB)", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--device", "-d", default="cpu")
    parser.add_argument("--check", action="store_true", help="only validate the output cache")
    args = parser.parse_args()

    if args.check:
        return 0 if is_cache_valid(args.output) else 1

    if args.input is None:
        print("--input is required unless --check is used", file=sys.stderr)
        return 1
    if not args.input.exists():
        print(f"missing input: {args.input}", file=sys.stderr)
        return 1
    if is_cache_valid(args.output) and args.output.stat().st_mtime > args.input.stat().st_mtime:
        print(f"cached: {args.output} is up to date", flush=True)
        return 0

    convert(args.input, args.output, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
