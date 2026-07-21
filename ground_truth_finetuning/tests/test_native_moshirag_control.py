import torch
from torch import nn

from ground_truth_finetuning.training.native_moshirag_control import (
    PreResponseControlStateProbe,
    StreamingConditionSnapshot,
    align_streaming_sum,
    listwise_causal_loss,
    select_full_rank_temporal_text_parameters,
    strict_listwise_group_pass,
)


def snapshot(**overrides) -> StreamingConditionSnapshot:
    values = {
        "generation_id": "generation-1",
        "control_revision": 4,
        "acknowledged_revision": 4,
        "available_at_frame": 2,
        "active_from_frame": 3,
        "retrieval_buffer_frames": 1,
        "cancel_at_frame": None,
    }
    values.update(overrides)
    return StreamingConditionSnapshot(**values)


def test_streaming_sum_is_aligned_zero_padded_and_cancelled() -> None:
    reference = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    timeline = align_streaming_sum(
        reference,
        sequence_frames=8,
        snapshot=snapshot(cancel_at_frame=6),
    )
    assert torch.equal(timeline[:4], torch.zeros(4, 2))
    assert torch.equal(timeline[4:6], reference[:2])
    assert torch.equal(timeline[6:], torch.zeros(2, 2))


def test_streaming_sum_rejects_unacknowledged_revision() -> None:
    try:
        snapshot(acknowledged_revision=3)
    except ValueError as error:
        assert "acknowledged" in str(error)
    else:
        raise AssertionError("unacknowledged revision was accepted")


def test_listwise_loss_and_strict_group_metric_reward_diagonal() -> None:
    good = torch.tensor(
        [[0.1, 1.0, 1.2], [1.1, 0.2, 1.3], [1.0, 1.4, 0.1]],
        requires_grad=True,
    )
    bad = good.detach().clone()
    bad[0, 1] = 0.0
    assert listwise_causal_loss(good, temperature=0.2) < listwise_causal_loss(
        bad, temperature=0.2
    )
    assert strict_listwise_group_pass(good.detach(), minimum_margin=0.5)
    assert not strict_listwise_group_pass(bad, minimum_margin=0.5)
    listwise_causal_loss(good, temperature=0.2).backward()
    assert good.grad is not None


def test_probe_reads_pre_response_frame_and_trains_all_slots() -> None:
    probe = PreResponseControlStateProbe(4, {"policy": 3, "status": 2})
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    loss = probe.loss(
        hidden,
        torch.tensor([2, 3]),
        {"policy": torch.tensor([1, 2]), "status": torch.tensor([0, 1])},
    )
    loss.backward()
    assert hidden.grad is not None


class FakePersonaPlex(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.text_emb = nn.Embedding(8, 4)
        self.text_linear = nn.Linear(4, 8)
        self.out_norm = nn.LayerNorm(4)
        self.emb = nn.ModuleList([nn.Embedding(8, 4)])
        self.depformer = nn.Linear(4, 4)


def test_full_rank_selection_excludes_audio_and_depth_modules() -> None:
    model = FakePersonaPlex()
    selection = select_full_rank_temporal_text_parameters(model)
    selected_ids = {id(parameter) for parameter in selection.parameters}
    assert selected_ids.isdisjoint(id(parameter) for parameter in model.emb.parameters())
    assert selected_ids.isdisjoint(id(parameter) for parameter in model.depformer.parameters())
    assert any(name.startswith("transformer.") for name in selection.parameter_names)
    assert any(name.startswith("text_emb.") for name in selection.parameter_names)
    assert any(name.startswith("text_linear.") for name in selection.parameter_names)
