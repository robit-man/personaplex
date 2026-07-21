#!/usr/bin/env python3
"""Exercise NCCL collectives on a capacity-relative tensor before training."""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    total_bytes = torch.cuda.get_device_properties(local_rank).total_memory
    # Scale communication volume to each device instead of assuming a model or
    # accelerator size. The divisor keeps the probe below one percent of VRAM.
    elements = max(1, total_bytes // (4 * 512))
    tensor = torch.full(
        (elements,), float(rank), device=torch.device("cuda", local_rank), dtype=torch.float32
    )
    dist.barrier()
    started = time.monotonic()
    dist.broadcast(tensor, src=0)
    torch.cuda.synchronize()
    broadcast_seconds = time.monotonic() - started
    tensor.add_(rank)
    started = time.monotonic()
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    all_reduce_seconds = time.monotonic() - started
    expected = float(sum(range(world_size)))
    if not torch.isclose(tensor[0], torch.tensor(expected, device=tensor.device)):
        raise RuntimeError(f"collective integrity failure: expected {expected}, got {tensor[0].item()}")
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "world_size": world_size,
                    "bytes_per_rank": elements * 4,
                    "broadcast_seconds": broadcast_seconds,
                    "all_reduce_seconds": all_reduce_seconds,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
