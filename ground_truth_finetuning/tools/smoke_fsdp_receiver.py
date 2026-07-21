#!/usr/bin/env python3
"""Three-rank CUDA smoke test for receiver sharding and DCP checkpointing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch import nn

from ground_truth_finetuning.training.fsdp_receiver import (
    clip_sharded_grad_norm,
    load_receiver_checkpoint,
    save_receiver_checkpoint,
    shard_full_rank_temporal_text_receiver,
)


class Layer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def forward(self, value, **_kwargs):
        return torch.tanh(self.linear(value))


class Transformer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer(hidden), Layer(hidden)])

    def forward(self, value, **kwargs):
        for layer in self.layers:
            value = layer(value, **kwargs)
        return value


class TinyPersonaPlex(nn.Module):
    def __init__(self, hidden: int = 16, vocabulary: int = 32) -> None:
        super().__init__()
        self.transformer = Transformer(hidden)
        self.text_emb = nn.Embedding(vocabulary, hidden)
        self.text_linear = nn.Linear(hidden, vocabulary)
        self.out_norm = nn.LayerNorm(hidden)
        self.emb = nn.ModuleList([nn.Embedding(vocabulary, hidden)])
        self.depformer = nn.Linear(hidden, hidden)

    def forward(self, text_tokens, audio_tokens, condition):
        value = self.text_emb(text_tokens) + self.emb[0](audio_tokens) + condition
        value = self.transformer(value)
        value = self.out_norm(value)
        return self.text_linear(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(7)
    model = TinyPersonaPlex().to(device=device, dtype=torch.bfloat16)
    bundle = shard_full_rank_temporal_text_receiver(model, device=device)
    optimizer = torch.optim.AdamW(bundle.trainable_parameters, lr=1e-3)
    text = torch.arange(8, device=device).reshape(2, 4)
    audio = torch.flip(text, dims=(1,))
    condition = torch.randn(2, 4, 16, device=device, dtype=torch.bfloat16)
    target = torch.remainder(text + 1, 32)
    loss = nn.functional.cross_entropy(model(text, audio, condition).float().reshape(-1, 32), target.reshape(-1))
    loss.backward()
    norm = clip_sharded_grad_norm(bundle.trainable_parameters, max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    save_receiver_checkpoint(
        model,
        optimizer,
        bundle,
        args.checkpoint_dir,
        {"schema": "personaplex.fsdp-receiver-smoke.v1", "worldSize": dist.get_world_size()},
    )
    load_receiver_checkpoint(model, optimizer, bundle, args.checkpoint_dir)
    frozen_has_grad = any(parameter.grad is not None for parameter in model.emb.parameters())
    report = torch.tensor(
        [float(loss.detach()), float(norm.detach()), float(frozen_has_grad)],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(report)
    report /= dist.get_world_size()
    if dist.get_rank() == 0:
        print(json.dumps({
            "status": "passed" if report[2].item() == 0.0 else "failed",
            "worldSize": dist.get_world_size(),
            "meanLoss": report[0].item(),
            "meanGradientNorm": report[1].item(),
            "frozenAudioGradientFraction": report[2].item(),
            "trainableParameters": bundle.trainable_parameter_count,
            "cpuOffload": bundle.cpu_offload,
        }, sort_keys=True))
    dist.destroy_process_group()
    return 0 if report[2].item() == 0.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
