"""PersonaPlex-native MoshiRAG conditioning and listwise training primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


NATIVE_MOSHIRAG_CONTROL_SCHEMA = "personaplex.native-moshirag-control.v1"


@dataclass(frozen=True)
class StreamingConditionSnapshot:
    generation_id: str
    control_revision: int
    acknowledged_revision: int
    available_at_frame: int
    active_from_frame: int
    retrieval_buffer_frames: int = 0
    cancel_at_frame: int | None = None

    def __post_init__(self) -> None:
        if not self.generation_id:
            raise ValueError("generation_id must not be empty")
        if self.control_revision < 0 or self.acknowledged_revision < 0:
            raise ValueError("control revisions must be non-negative")
        if self.control_revision != self.acknowledged_revision:
            raise ValueError("generation must snapshot an acknowledged control revision")
        for name in (
            "available_at_frame",
            "active_from_frame",
            "retrieval_buffer_frames",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.cancel_at_frame is not None and self.cancel_at_frame < 0:
            raise ValueError("cancel_at_frame must be non-negative")

    @property
    def condition_start_frame(self) -> int:
        return max(self.available_at_frame, self.active_from_frame) + self.retrieval_buffer_frames


def align_streaming_sum(
    reference: Tensor,
    *,
    sequence_frames: int,
    snapshot: StreamingConditionSnapshot,
    drop_control: bool = False,
) -> Tensor:
    """Align one immutable ARC stream to real temporal frames.

    The function never repeats the last ARC row. Frames before availability,
    after stream exhaustion, and at/after cancellation are explicit zeros.
    """

    if sequence_frames < 1:
        raise ValueError("sequence_frames must be positive")
    if reference.ndim == 3:
        if reference.shape[0] != 1:
            raise ValueError("batched ARC reference must have batch size one")
        reference = reference[0]
    if reference.ndim != 2 or reference.shape[0] < 1 or reference.shape[1] < 1:
        raise ValueError("ARC reference must have shape [frames, hidden]")
    if not torch.isfinite(reference).all():
        raise ValueError("ARC reference contains non-finite values")
    timeline = reference.new_zeros((sequence_frames, reference.shape[1]))
    if drop_control:
        return timeline
    start = snapshot.condition_start_frame
    stop = sequence_frames
    if snapshot.cancel_at_frame is not None:
        stop = min(stop, snapshot.cancel_at_frame)
    if start >= stop:
        return timeline
    count = min(reference.shape[0], stop - start)
    timeline[start : start + count] = reference[:count]
    return timeline


def native_streaming_sum_forward(
    lm_model: nn.Module,
    codes: Tensor,
    condition_timeline: Tensor,
) -> tuple[Tensor, Tensor]:
    """Run PersonaPlex temporal/text prediction with native input addition."""

    embeddings = lm_model.embed_codes(codes)
    if condition_timeline.ndim == 2:
        condition_timeline = condition_timeline.unsqueeze(0)
    if condition_timeline.ndim != 3:
        raise ValueError("condition timeline must have shape [batch, frames, hidden]")
    if condition_timeline.shape != embeddings.shape:
        raise ValueError(
            f"condition shape {tuple(condition_timeline.shape)} does not match "
            f"embeddings {tuple(embeddings.shape)}"
        )
    conditioned = embeddings + condition_timeline.to(
        device=embeddings.device,
        dtype=embeddings.dtype,
    )
    return lm_model.forward_embeddings(conditioned)


def listwise_causal_loss(nll_matrix: Tensor, *, temperature: float) -> Tensor:
    """Rank every sibling's matched control over all incompatible siblings."""

    if nll_matrix.ndim != 2 or nll_matrix.shape[0] != nll_matrix.shape[1]:
        raise ValueError("listwise NLL matrix must be square")
    if nll_matrix.shape[0] < 2:
        raise ValueError("listwise objective requires at least two siblings")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not torch.isfinite(nll_matrix).all():
        raise ValueError("listwise NLL matrix contains non-finite values")
    targets = torch.arange(nll_matrix.shape[0], device=nll_matrix.device)
    logits = -nll_matrix / temperature
    target_to_control = F.cross_entropy(logits, targets)
    control_to_target = F.cross_entropy(logits.transpose(0, 1), targets)
    return 0.5 * (target_to_control + control_to_target)


def strict_listwise_group_pass(
    nll_matrix: Tensor,
    *,
    minimum_margin: float,
) -> bool:
    """Require every diagonal match to beat every sibling in both directions."""

    if minimum_margin < 0:
        raise ValueError("minimum_margin must be non-negative")
    if nll_matrix.ndim != 2 or nll_matrix.shape[0] != nll_matrix.shape[1]:
        raise ValueError("listwise NLL matrix must be square")
    diagonal = nll_matrix.diagonal()
    size = nll_matrix.shape[0]
    for index in range(size):
        for other in range(size):
            if index == other:
                continue
            if not bool((nll_matrix[index, other] - diagonal[index] >= minimum_margin).item()):
                return False
            if not bool((nll_matrix[other, index] - diagonal[index] >= minimum_margin).item()):
                return False
    return True


class PreResponseControlStateProbe(nn.Module):
    """Auxiliary typed-state head used only during continued pretraining."""

    def __init__(self, hidden_size: int, slot_cardinalities: Mapping[str, int]) -> None:
        super().__init__()
        if hidden_size < 1 or not slot_cardinalities:
            raise ValueError("probe requires a hidden size and at least one slot")
        if any(cardinality < 2 for cardinality in slot_cardinalities.values()):
            raise ValueError("every probe slot must have at least two values")
        self.slot_names = tuple(sorted(slot_cardinalities))
        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_size, slot_cardinalities[name])
                for name in self.slot_names
            }
        )

    def forward(self, hidden: Tensor, frame_indices: Tensor) -> dict[str, Tensor]:
        if hidden.ndim != 3:
            raise ValueError("hidden state must have shape [batch, frames, hidden]")
        if frame_indices.ndim != 1 or frame_indices.shape[0] != hidden.shape[0]:
            raise ValueError("frame_indices must have one entry per batch row")
        if bool(((frame_indices < 0) | (frame_indices >= hidden.shape[1])).any().item()):
            raise ValueError("probe frame index is outside the sequence")
        rows = hidden[
            torch.arange(hidden.shape[0], device=hidden.device),
            frame_indices,
        ]
        return {name: self.heads[name](rows) for name in self.slot_names}

    def loss(
        self,
        hidden: Tensor,
        frame_indices: Tensor,
        targets: Mapping[str, Tensor],
    ) -> Tensor:
        if set(targets) != set(self.slot_names):
            raise ValueError("probe targets do not match configured slots")
        logits = self(hidden, frame_indices)
        return torch.stack(
            [F.cross_entropy(logits[name], targets[name]) for name in self.slot_names]
        ).mean()


@dataclass(frozen=True)
class FullRankParameterSelection:
    parameters: tuple[nn.Parameter, ...]
    parameter_names: tuple[str, ...]
    parameter_count: int


def select_full_rank_temporal_text_parameters(lm_model: nn.Module) -> FullRankParameterSelection:
    """Select the explicit Moshi temporal/text receiver while excluding audio/depth."""

    required_modules = {
        "transformer": getattr(lm_model, "transformer", None),
        "text_emb": getattr(lm_model, "text_emb", None),
        "text_linear": getattr(lm_model, "text_linear", None),
    }
    missing = sorted(name for name, module in required_modules.items() if not isinstance(module, nn.Module))
    if missing:
        raise ValueError(f"PersonaPlex model lacks trainable modules: {missing}")
    out_norm = getattr(lm_model, "out_norm", None)
    if isinstance(out_norm, nn.Module):
        required_modules["out_norm"] = out_norm

    selected_by_id: dict[int, tuple[str, nn.Parameter]] = {}
    for module_name, module in required_modules.items():
        for name, parameter in module.named_parameters():
            selected_by_id[id(parameter)] = (f"{module_name}.{name}", parameter)
    selected = tuple(selected_by_id[key] for key in sorted(selected_by_id))
    if not selected:
        raise ValueError("full-rank temporal/text selection is empty")
    return FullRankParameterSelection(
        parameters=tuple(parameter for _name, parameter in selected),
        parameter_names=tuple(name for name, _parameter in selected),
        parameter_count=sum(parameter.numel() for _name, parameter in selected),
    )
