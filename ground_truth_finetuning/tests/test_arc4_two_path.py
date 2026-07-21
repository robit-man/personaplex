import torch
import torch.nn.functional as F

from ground_truth_finetuning.training.arc4_two_path import (
    Arc4TwoPathAdapter,
    Arc4TwoPathBundle,
    Arc4TwoPathConfig,
)


def config() -> Arc4TwoPathConfig:
    return Arc4TwoPathConfig(
        hidden_size=8,
        decision_frames=4,
        stream_frames=6,
        prefix_tokens=3,
        rank=4,
        attention_heads=2,
    )


def test_two_path_arc_emits_prefix_and_ordered_stream() -> None:
    adapter = Arc4TwoPathAdapter(config())
    reference = torch.randn(2, 10, 8)
    output = adapter(reference)
    assert isinstance(output, Arc4TwoPathBundle)
    assert output.prefix_embeddings.shape == (2, 3, 8)
    assert output.evidence_stream.shape == (2, 6, 8)
    torch.testing.assert_close(output.evidence_stream, reference[:, 4:])
    assert not any(name.startswith("stream_") for name, _ in adapter.named_parameters())


def test_two_path_arc_starts_from_ordered_segment_reduction() -> None:
    adapter = Arc4TwoPathAdapter(config())
    reference = torch.randn(2, 10, 8)
    output = adapter(reference)
    pooled = F.adaptive_avg_pool1d(reference[:, :4].transpose(1, 2), 3).transpose(1, 2)
    expected = adapter._bound(adapter.prefix_gate * pooled, adapter.config.max_prefix_rms)
    torch.testing.assert_close(output.prefix_embeddings, expected)


def test_two_path_arc_null_control_is_exact() -> None:
    adapter = Arc4TwoPathAdapter(config())
    output = adapter(torch.zeros(2, 10, 8))
    assert torch.count_nonzero(output.prefix_embeddings) == 0
    assert torch.count_nonzero(output.evidence_stream) == 0


def test_two_path_arc_decision_changes_immediate_prefix() -> None:
    adapter = Arc4TwoPathAdapter(config())
    first = torch.randn(1, 10, 8)
    second = first.clone()
    second[:, 0, 0] += 5.0
    first_output = adapter(first)
    second_output = adapter(second)
    assert not torch.allclose(first_output.prefix_embeddings, second_output.prefix_embeddings)
    torch.testing.assert_close(first_output.evidence_stream, second_output.evidence_stream)
