"""FSDP sharding for full-rank PersonaPlex temporal/text receiver adaptation."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from typing import Iterator

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch import Tensor, nn


@dataclass(frozen=True)
class FSDPReceiverBundle:
    roots: tuple[FSDP, ...]
    trainable_parameters: tuple[nn.Parameter, ...]
    trainable_parameter_count: int
    frozen_parameter_count: int
    sharding: str = "full_shard_temporal_layers_and_text_modules"
    cpu_offload: bool = False

    @contextmanager
    def no_sync(self) -> Iterator[None]:
        with ExitStack() as stack:
            for root in self.roots:
                stack.enter_context(root.no_sync())
            yield


def _wrap(
    module: nn.Module,
    *,
    device: torch.device,
    auto_wrap_policy=None,
) -> FSDP:
    return FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
        limit_all_gathers=True,
        forward_prefetch=False,
    )


def shard_full_rank_temporal_text_receiver(
    lm_model: nn.Module,
    *,
    device: torch.device,
) -> FSDPReceiverBundle:
    """Freeze audio/depth modules and FULL_SHARD the explicit receiver modules."""

    if not dist.is_initialized() or dist.get_world_size() < 2:
        raise RuntimeError("full-rank receiver sharding requires distributed execution")
    if device.type != "cuda":
        raise RuntimeError("full-rank receiver training is CUDA-only")
    required = ("transformer", "text_emb", "text_linear")
    missing = [name for name in required if not isinstance(getattr(lm_model, name, None), nn.Module)]
    if missing:
        raise ValueError(f"PersonaPlex model lacks receiver modules: {missing}")
    for parameter in lm_model.parameters():
        parameter.requires_grad_(False)
    receiver_modules = [lm_model.transformer, lm_model.text_emb, lm_model.text_linear]
    if isinstance(getattr(lm_model, "out_norm", None), nn.Module):
        receiver_modules.append(lm_model.out_norm)
    for module in receiver_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    layers = getattr(lm_model.transformer, "layers", None)
    policy = None
    if layers is not None and len(layers):
        policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={type(layers[0])},
        )
    lm_model.transformer = _wrap(
        lm_model.transformer,
        device=device,
        auto_wrap_policy=policy,
    )
    lm_model.text_emb = _wrap(lm_model.text_emb, device=device)
    lm_model.text_linear = _wrap(lm_model.text_linear, device=device)
    roots: list[FSDP] = [lm_model.transformer, lm_model.text_emb, lm_model.text_linear]
    if isinstance(getattr(lm_model, "out_norm", None), nn.Module):
        lm_model.out_norm = _wrap(lm_model.out_norm, device=device)
        roots.append(lm_model.out_norm)

    seen: set[int] = set()
    trainable: list[nn.Parameter] = []
    for root in roots:
        for parameter in root.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                seen.add(id(parameter))
                trainable.append(parameter)
    if not trainable:
        raise RuntimeError("FSDP receiver has no trainable parameters")
    frozen_count = sum(
        parameter.numel() for parameter in lm_model.parameters() if not parameter.requires_grad
    )
    return FSDPReceiverBundle(
        roots=tuple(roots),
        trainable_parameters=tuple(trainable),
        trainable_parameter_count=sum(parameter.numel() for parameter in trainable),
        frozen_parameter_count=frozen_count,
    )


def clip_sharded_grad_norm(
    parameters: tuple[nn.Parameter, ...],
    *,
    max_norm: float,
) -> Tensor:
    """Clip one global norm across disjoint FULL_SHARD parameter shards."""

    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    device = next((parameter.device for parameter in parameters if parameter.grad is not None), None)
    if device is None:
        return torch.zeros((), device=torch.device("cuda"))
    local_squared = torch.zeros((), device=device, dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            local_squared += parameter.grad.detach().double().square().sum()
    dist.all_reduce(local_squared, op=dist.ReduceOp.SUM)
    norm = local_squared.sqrt()
    scale = torch.clamp(norm.new_tensor(max_norm) / (norm + 1e-12), max=1.0)
    if bool((scale < 1.0).item()):
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale.to(dtype=parameter.grad.dtype))
    return norm


def save_receiver_checkpoint(
    lm_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    bundle: FSDPReceiverBundle,
    checkpoint_dir: Path,
    metadata: dict,
) -> None:
    """Save sharded receiver/optimizer state without gathering onto host RAM."""

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    submodules = set(bundle.roots)
    state = {
        "model": get_model_state_dict(lm_model, submodules=submodules),
        "optimizer": get_optimizer_state_dict(
            lm_model,
            optimizer,
            submodules=submodules,
        ),
    }
    dcp.save(state, checkpoint_id=checkpoint_dir / "shards")
    if dist.get_rank() == 0:
        temporary = checkpoint_dir / "metadata.json.tmp"
        temporary.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        temporary.replace(checkpoint_dir / "metadata.json")
    dist.barrier()


def load_receiver_checkpoint(
    lm_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    bundle: FSDPReceiverBundle,
    checkpoint_dir: Path,
) -> None:
    submodules = set(bundle.roots)
    model_state = get_model_state_dict(lm_model, submodules=submodules)
    optimizer_state = get_optimizer_state_dict(
        lm_model,
        optimizer,
        submodules=submodules,
    )
    state = {"model": model_state, "optimizer": optimizer_state}
    dcp.load(state, checkpoint_id=checkpoint_dir / "shards")
    partial_options = StateDictOptions(strict=False)
    set_model_state_dict(lm_model, state["model"], options=partial_options)
    set_optimizer_state_dict(
        lm_model,
        optimizer,
        state["optimizer"],
        options=partial_options,
    )
    dist.barrier()
