"""Bounded LoRA for selected upper PersonaPlex temporal-transformer layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TEMPORAL_LORA_ARCHITECTURE = "upper-temporal-linear-lora-v1"


@dataclass(frozen=True)
class TemporalLoRAConfig:
    layer_count: int = 4
    rank: int = 8
    alpha: float = 16.0
    architecture_revision: str = TEMPORAL_LORA_ARCHITECTURE

    def __post_init__(self) -> None:
        if self.layer_count < 1 or self.rank < 1:
            raise ValueError("temporal LoRA dimensions must be positive")
        if self.alpha <= 0.0:
            raise ValueError("temporal LoRA alpha must be positive")
        if self.architecture_revision != TEMPORAL_LORA_ARCHITECTURE:
            raise ValueError("unsupported temporal LoRA architecture")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class FrozenLoRALinear(nn.Module):
    """A frozen linear plus a zero-initialized, fp32 low-rank residual."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRA base must be torch.nn.Linear")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        device = base.weight.device
        self.down = nn.Parameter(
            torch.empty(rank, base.in_features, dtype=torch.float32, device=device)
        )
        self.up = nn.Parameter(
            torch.zeros(base.out_features, rank, dtype=torch.float32, device=device)
        )
        self.scale = float(alpha) / float(rank)
        nn.init.kaiming_uniform_(self.down, a=5**0.5)

    def forward(self, value: Tensor) -> Tensor:
        frozen = self.base(value)
        residual = F.linear(F.linear(value.float(), self.down), self.up)
        return frozen + (self.scale * residual).to(dtype=frozen.dtype)


@dataclass
class TemporalLoRAInstallation:
    config: TemporalLoRAConfig
    layer_indices: tuple[int, ...]
    modules: dict[str, FrozenLoRALinear]

    def parameters(self) -> Iterable[nn.Parameter]:
        for module in self.modules.values():
            yield module.down
            yield module.up

    def state_dict(self) -> dict[str, Tensor]:
        output: dict[str, Tensor] = {}
        for name, module in self.modules.items():
            output[f"{name}.down"] = module.down.detach()
            output[f"{name}.up"] = module.up.detach()
        return output

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        expected = {
            f"{name}.{side}"
            for name in self.modules
            for side in ("down", "up")
        }
        if set(state) != expected:
            missing = sorted(expected.difference(state))
            extra = sorted(set(state).difference(expected))
            raise ValueError(f"temporal LoRA state mismatch missing={missing} extra={extra}")
        with torch.no_grad():
            for name, module in self.modules.items():
                for side in ("down", "up"):
                    target = getattr(module, side)
                    value = state[f"{name}.{side}"]
                    if tuple(value.shape) != tuple(target.shape):
                        raise ValueError(f"temporal LoRA shape mismatch: {name}.{side}")
                    target.copy_(value.to(device=target.device, dtype=target.dtype))


def install_temporal_lora(
    lm_model: object,
    config: TemporalLoRAConfig,
) -> TemporalLoRAInstallation:
    for parameter in lm_model.parameters():
        parameter.requires_grad_(False)
    layers = lm_model.transformer.layers
    if config.layer_count > len(layers):
        raise ValueError("temporal LoRA layer count exceeds transformer depth")
    indices = tuple(range(len(layers) - config.layer_count, len(layers)))
    modules: dict[str, FrozenLoRALinear] = {}
    for index in indices:
        layer = layers[index]
        targets = (
            ("self_attn.out_proj", layer.self_attn, "out_proj"),
            ("gating.linear_in", layer.gating, "linear_in"),
            ("gating.linear_out", layer.gating, "linear_out"),
        )
        for suffix, parent, attribute in targets:
            base = getattr(parent, attribute)
            wrapper = FrozenLoRALinear(base, rank=config.rank, alpha=config.alpha)
            setattr(parent, attribute, wrapper)
            modules[f"transformer.layers.{index}.{suffix}"] = wrapper
    return TemporalLoRAInstallation(config=config, layer_indices=indices, modules=modules)
