import torch

from ground_truth_finetuning.training.arc4_conditioning import (
    Arc4InjectionConfig,
    GatedArc4InjectionAdapter,
    clip_optimizer_gradients,
    margin_adjusted_control_contrastive_loss,
    joint_margin_coverage_surrogate,
    pad_arc4_references,
    reduce_pair_direction_losses,
)


def test_pad_arc4_references_preserves_rows_and_zero_pads():
    first = torch.ones(1, 2, 4)
    second = torch.full((1, 3, 4), 2.0)
    result = pad_arc4_references([first, second])
    assert result.shape == (2, 3, 4)
    assert torch.equal(result[0, :2], first[0])
    assert torch.count_nonzero(result[0, 2]) == 0
    assert torch.equal(result[1], second[0])


def test_arc4_adapter_null_is_exact_and_stream_is_rms_bounded():
    config = Arc4InjectionConfig(hidden_size=8, rank=4, initial_gate=0.10, max_stream_rms=0.05)
    adapter = GatedArc4InjectionAdapter(config)
    null = adapter(torch.zeros(2, 3, 8))
    assert torch.count_nonzero(null) == 0
    stream = adapter(torch.randn(2, 3, 8) * 100)
    rms = stream.square().mean(dim=(1, 2)).sqrt()
    assert torch.all(rms <= 0.050001)


def test_pair_direction_reduction_preserves_legacy_mean_at_zero_weight():
    values = torch.tensor([0.2, 0.8], requires_grad=True)
    reduced = reduce_pair_direction_losses(
        values,
        worst_direction_weight=0.0,
        temperature=0.05,
    )
    torch.testing.assert_close(reduced, values.mean())


def test_pair_direction_reduction_prioritizes_the_worse_direction():
    values = torch.tensor([0.2, 0.8], requires_grad=True)
    reduced = reduce_pair_direction_losses(
        values,
        worst_direction_weight=1.0,
        temperature=0.05,
    )
    reduced.backward()
    assert values.grad is not None
    assert values.grad[1] > values.grad[0]
    assert values.mean() < reduced < values.max()


def test_optimizer_gradient_clipping_includes_every_parameter_group():
    adapter = torch.nn.Parameter(torch.tensor(0.0))
    temporal_lora = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD(
        [
            {"params": [adapter], "lr": 0.1},
            {"params": [temporal_lora], "lr": 0.1},
        ]
    )
    adapter.grad = torch.tensor(3.0)
    temporal_lora.grad = torch.tensor(4.0)
    norm = clip_optimizer_gradients(optimizer, 1.0)
    torch.testing.assert_close(norm, torch.tensor(5.0))
    torch.testing.assert_close(adapter.grad, torch.tensor(0.6), atol=1e-5, rtol=0)
    torch.testing.assert_close(temporal_lora.grad, torch.tensor(0.8), atol=1e-5, rtol=0)


def test_margin_adjusted_contrastive_prefers_correct_control_ranking():
    own = torch.tensor(1.0)
    bad = margin_adjusted_control_contrastive_loss(
        own,
        [torch.tensor(1.01), torch.tensor(1.02)],
        [0.08, 0.03],
        temperature=0.1,
    )
    good = margin_adjusted_control_contrastive_loss(
        own,
        [torch.tensor(1.50), torch.tensor(1.40)],
        [0.08, 0.03],
        temperature=0.1,
    )
    assert good < bad


def test_joint_margin_coverage_surrogate_tracks_exact_pass_boundary():
    passing = joint_margin_coverage_surrogate(
        torch.tensor([0.09, 0.10]),
        torch.tensor([0.31, 0.40]),
        whole_margin=0.08,
        focused_margin=0.30,
        temperature=0.25,
        worst_direction_weight=1.0,
        worst_direction_temperature=0.05,
    )
    failing = joint_margin_coverage_surrogate(
        torch.tensor([0.07, 0.02]),
        torch.tensor([0.29, 0.10]),
        whole_margin=0.08,
        focused_margin=0.30,
        temperature=0.25,
        worst_direction_weight=1.0,
        worst_direction_temperature=0.05,
    )
    assert passing < failing
