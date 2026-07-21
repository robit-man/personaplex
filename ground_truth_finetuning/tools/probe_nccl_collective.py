#!/usr/bin/env python3
"""Torchrun target that verifies DDP broadcast and gradient all-reduce."""

from __future__ import annotations

import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=20))
    device = torch.device("cuda", local_rank)
    model = torch.nn.Linear(1024, 1024, bias=False, device=device)
    wrapped = DistributedDataParallel(model, device_ids=[local_rank])
    value = torch.ones(8, 1024, device=device) * (rank + 1)
    wrapped(value).sum().backward()
    check = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(check)
    expected = float(dist.get_world_size() * (dist.get_world_size() + 1) // 2)
    if float(check) != expected or model.weight.grad is None or not torch.isfinite(model.weight.grad).all():
        raise RuntimeError("NCCL collective probe produced invalid values")
    if rank == 0:
        print(json.dumps({"status": "passed", "worldSize": dist.get_world_size()}), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
