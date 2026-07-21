"""Source-of-truth temporal packing for bounded ARC-4 control streams."""

from __future__ import annotations

import torch
import torch.nn.functional as F


ARC4_FIELD_SLOTS_PACKING_REVISION = "arc4-field-slots-v2"
ARC4_GLOBAL_FIRST_PACKING_REVISION = "arc4-global-first-v2"
ARC4_SUPPORTED_PACKING_REVISIONS = (
    ARC4_FIELD_SLOTS_PACKING_REVISION,
    ARC4_GLOBAL_FIRST_PACKING_REVISION,
)
# Compatibility alias for the runtime's primary field-slotted path.
ARC4_PACKING_REVISION = ARC4_FIELD_SLOTS_PACKING_REVISION
ARC4_FIELD_ORDER = ("decision", "state", "delivery", "context")
ARC4_FIELD_WEIGHTS = (8, 4, 2, 2)


def pack_arc4_stream(
    encoded: torch.Tensor,
    mask: torch.Tensor,
    output_frames: int,
) -> torch.Tensor:
    """Pack every valid ARC state into fixed frames with global semantics first.

    ARC's final causal state has observed the complete reference and is placed in
    frame zero.  A masked global mean follows, then adaptive temporal bins retain
    local ordering.  This prevents a bounded PersonaPlex stream from silently
    discarding semantic changes near the end of a control reference.
    """

    if encoded.ndim != 3 or mask.ndim != 2 or encoded.shape[:2] != mask.shape:
        raise ValueError("ARC encoded/mask shapes must be [batch, frames, hidden] and [batch, frames]")
    if output_frames < 1:
        raise ValueError("output_frames must be positive")
    packed = []
    for batch_index in range(encoded.shape[0]):
        valid = encoded[batch_index][mask[batch_index].to(dtype=torch.bool)]
        if valid.shape[0] < 1:
            raise ValueError("ARC stream contains no valid frames")
        if output_frames == 1:
            value = valid[-1:]
        else:
            global_frames = [valid[-1], valid.mean(dim=0)]
            remaining = output_frames - len(global_frames)
            if remaining > 0:
                temporal = F.adaptive_avg_pool1d(
                    valid.transpose(0, 1).unsqueeze(0), remaining
                ).squeeze(0).transpose(0, 1)
                value = torch.cat((torch.stack(global_frames), temporal), dim=0)
            else:
                value = torch.stack(global_frames[:output_frames])
        packed.append(value)
    return torch.stack(packed).contiguous()


def field_frame_allocation(output_frames: int) -> tuple[int, ...]:
    quantum = sum(ARC4_FIELD_WEIGHTS)
    if output_frames < quantum or output_frames % quantum:
        raise ValueError(f"field-slotted ARC frames must be a positive multiple of {quantum}")
    scale = output_frames // quantum
    return tuple(weight * scale for weight in ARC4_FIELD_WEIGHTS)


def pack_arc4_fields(
    encoded: torch.Tensor,
    mask: torch.Tensor,
    field_names: tuple[str, ...],
    output_frames: int,
) -> torch.Tensor:
    """Pack independently encoded semantic fields into stable frame ranges."""

    if field_names != ARC4_FIELD_ORDER:
        raise ValueError("ARC field names/order do not match the versioned packing contract")
    if encoded.shape[0] != len(ARC4_FIELD_ORDER):
        raise ValueError("ARC field batch does not match the versioned field count")
    allocations = field_frame_allocation(output_frames)
    values = [
        pack_arc4_stream(encoded[index : index + 1], mask[index : index + 1], frames)
        for index, frames in enumerate(allocations)
    ]
    return torch.cat(values, dim=1).contiguous()
