import torch

from personaplex_control.arc4_packing import (
    ARC4_FIELD_SLOTS_PACKING_REVISION,
    ARC4_FIELD_ORDER,
    ARC4_GLOBAL_FIRST_PACKING_REVISION,
    ARC4_SUPPORTED_PACKING_REVISIONS,
    field_frame_allocation,
    pack_arc4_fields,
    pack_arc4_stream,
)
from personaplex_control.moshirag_reference import render_arc4_reference_envelope
from personaplex_control.moshirag_conditioner_server import (
    _find_named_mapping,
    _strip_best_prefix,
)


def test_finds_nested_multi_arc_encoder_config() -> None:
    config = {
        "condition_provider": {
            "conditioners": {
                "reference_with_time": {
                    "multi_arc_encoder": {"compression": [-4], "dim": 3072}
                }
            }
        }
    }
    assert _find_named_mapping(config, "multi_arc_encoder") == {
        "compression": [-4],
        "dim": 3072,
    }


def test_checkpoint_prefix_is_normalized_without_unrelated_tensors() -> None:
    state = {
        "module.embedder.weight": "embedder",
        "module.output_proj.weight": "projection",
        "other.weight": "unrelated",
    }
    selected = _strip_best_prefix(
        state,
        {"embedder.weight", "output_proj.weight"},
        ("", "module."),
    )
    assert selected == {
        "embedder.weight": "embedder",
        "output_proj.weight": "projection",
    }


def test_global_semantics_are_packed_before_temporal_bins() -> None:
    encoded = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [99.0, 99.0]]])
    mask = torch.tensor([[True, True, True, False]])
    packed = pack_arc4_stream(encoded, mask, 4)
    assert packed.shape == (1, 4, 2)
    assert torch.equal(packed[0, 0], encoded[0, 2])
    assert torch.equal(packed[0, 1], encoded[0, :3].mean(dim=0))
    assert not torch.any(packed == 99.0)


def test_fields_receive_stable_ranges() -> None:
    encoded = torch.arange(4 * 6 * 2, dtype=torch.float32).reshape(4, 6, 2)
    mask = torch.ones((4, 6), dtype=torch.bool)
    packed = pack_arc4_fields(encoded, mask, ARC4_FIELD_ORDER, 96)
    assert packed.shape == (1, 96, 2)
    assert field_frame_allocation(96) == (48, 24, 12, 12)
    assert torch.equal(packed[0, 0], encoded[0, -1])
    assert torch.equal(packed[0, 48], encoded[1, -1])


def test_both_packing_contracts_are_explicit_and_reference_is_canonical() -> None:
    assert ARC4_SUPPORTED_PACKING_REVISIONS == (
        ARC4_FIELD_SLOTS_PACKING_REVISION,
        ARC4_GLOBAL_FIRST_PACKING_REVISION,
    )
    fields = {name: f"{name}-value" for name in ARC4_FIELD_ORDER}
    rendered = render_arc4_reference_envelope(fields)
    assert rendered.startswith('{"v":"personaplex-semantic-reference-v5-no-lineage"')
    assert rendered == render_arc4_reference_envelope(fields)
