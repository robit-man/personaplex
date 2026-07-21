import torch
from torch import nn

from ground_truth_finetuning.training.temporal_lora import (
    FrozenLoRALinear,
    TemporalLoRAConfig,
    install_temporal_lora,
)


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out_proj = nn.Linear(8, 8, bias=False)


class _Gating(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_in = nn.Linear(8, 12, bias=False)
        self.linear_out = nn.Linear(6, 8, bias=False)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.gating = _Gating()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList([_Layer() for _ in range(4)])
        self.depformer = nn.Linear(8, 8, bias=False)


def test_temporal_lora_wraps_only_selected_upper_projections() -> None:
    model = _Model()
    depformer = model.depformer
    lower = model.transformer.layers[1].self_attn.out_proj
    value = torch.randn(2, 3, 8)
    expected = model.transformer.layers[3].self_attn.out_proj(value)
    installed = install_temporal_lora(
        model,
        TemporalLoRAConfig(layer_count=2, rank=2, alpha=4.0),
    )
    assert installed.layer_indices == (2, 3)
    assert model.transformer.layers[1].self_attn.out_proj is lower
    assert model.depformer is depformer
    assert isinstance(model.transformer.layers[3].self_attn.out_proj, FrozenLoRALinear)
    torch.testing.assert_close(
        model.transformer.layers[3].self_attn.out_proj(value),
        expected,
    )
    assert all(not parameter.requires_grad for parameter in lower.parameters())
    assert sum(parameter.numel() for parameter in installed.parameters()) == 200


def test_temporal_lora_state_round_trip_changes_output() -> None:
    first = _Model()
    installed = install_temporal_lora(
        first,
        TemporalLoRAConfig(layer_count=1, rank=2, alpha=2.0),
    )
    with torch.no_grad():
        for module in installed.modules.values():
            module.up.fill_(0.1)
    state = installed.state_dict()
    second = _Model()
    second.load_state_dict(first.state_dict(), strict=False)
    restored = install_temporal_lora(
        second,
        TemporalLoRAConfig(layer_count=1, rank=2, alpha=2.0),
    )
    restored.load_state_dict(state)
    for name in state:
        torch.testing.assert_close(restored.state_dict()[name], state[name])
