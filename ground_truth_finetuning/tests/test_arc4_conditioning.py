import torch

from ground_truth_finetuning.training.arc4_conditioning import (
    Arc4ConditioningBundle,
    Arc4InjectionConfig,
    FIELD_PERSISTENT_ARC4_ARCHITECTURE,
    GatedArc4InjectionAdapter,
    LAYERWISE_PERSISTENT_ARC4_ARCHITECTURE,
    LAYERWISE_ADAPTED_ARC4_ARCHITECTURE,
)


def test_arc4_adapter_starts_as_small_gated_identity() -> None:
    config = Arc4InjectionConfig(hidden_size=8, rank=2, initial_gate=0.02)
    adapter = GatedArc4InjectionAdapter(config)
    reference = torch.randn(2, 3, 8)
    output = adapter(reference)
    torch.testing.assert_close(output, reference * 0.02, atol=1e-6, rtol=1e-5)


def test_arc4_control_dropout_preserves_trainable_graph() -> None:
    adapter = GatedArc4InjectionAdapter(
        Arc4InjectionConfig(hidden_size=8, rank=2, initial_gate=0.02)
    )
    output = adapter(torch.randn(1, 2, 8), drop_condition=True)
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert adapter.gate_logit.grad is not None


def test_field_persistent_arc4_exposes_all_fields_from_first_frame() -> None:
    config = Arc4InjectionConfig(
        hidden_size=8,
        rank=2,
        initial_gate=0.25,
        architecture_revision=FIELD_PERSISTENT_ARC4_ARCHITECTURE,
        field_frames=(2, 2, 1, 1),
        output_frames=9,
    )
    adapter = GatedArc4InjectionAdapter(config)
    first = torch.randn(1, 6, 8)
    second = first.clone()
    second[:, -1, 0] += 3.0
    first_output = adapter(first)
    second_output = adapter(second)
    assert first_output.shape == (1, 9, 8)
    torch.testing.assert_close(first_output[:, :1], first_output[:, -1:])
    assert not torch.allclose(first_output[:, 0], second_output[:, 0])


def test_field_persistent_arc4_rejects_wrong_input_budget() -> None:
    adapter = GatedArc4InjectionAdapter(
        Arc4InjectionConfig(
            hidden_size=8,
            rank=2,
            architecture_revision=FIELD_PERSISTENT_ARC4_ARCHITECTURE,
            field_frames=(2, 2, 1, 1),
        )
    )
    try:
        adapter(torch.randn(1, 5, 8))
    except ValueError as error:
        assert "requires 6 input frames" in str(error)
    else:
        raise AssertionError("wrong field-slot budget was accepted")


def test_layerwise_arc4_emits_versioned_persistent_bundle() -> None:
    adapter = GatedArc4InjectionAdapter(
        Arc4InjectionConfig(
            hidden_size=8,
            rank=2,
            architecture_revision=LAYERWISE_PERSISTENT_ARC4_ARCHITECTURE,
            field_frames=(2, 2, 1, 1),
            output_frames=7,
            layer_indices=(2, 3),
        )
    )
    output = adapter(torch.randn(2, 6, 8))
    assert isinstance(output, Arc4ConditioningBundle)
    assert output.input_stream.shape == (2, 7, 8)
    assert output.layer_streams.shape == (2, 2, 7, 8)
    assert output.layer_indices == (2, 3)
    torch.testing.assert_close(output.input_stream[:, :1], output.input_stream[:, -1:])
    torch.testing.assert_close(output.layer_streams[:, :, :1], output.layer_streams[:, :, -1:])


def test_layerwise_adapted_arc4_exports_bounded_upper_layer_weights() -> None:
    adapter = GatedArc4InjectionAdapter(
        Arc4InjectionConfig(
            hidden_size=8,
            rank=2,
            architecture_revision=LAYERWISE_ADAPTED_ARC4_ARCHITECTURE,
            field_frames=(2, 2, 1, 1),
            output_frames=7,
            layer_indices=(2, 3),
            layer_adaptation_rank=3,
        )
    )
    output = adapter(torch.randn(1, 6, 8))
    assert isinstance(output, Arc4ConditioningBundle)
    assert output.layer_adapter_down is not None
    assert output.layer_adapter_up is not None
    assert output.layer_adapter_gates is not None
    assert output.layer_adapter_down.shape == (2, 3, 8)
    assert output.layer_adapter_up.shape == (2, 8, 3)
    assert output.layer_adapter_gates.shape == (2,)
    assert output.max_layer_adaptation_rms == 0.10
    assert torch.count_nonzero(output.layer_adapter_up) == 0
