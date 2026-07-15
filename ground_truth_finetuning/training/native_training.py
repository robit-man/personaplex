"""Adapter training through PersonaPlex's native delayed-code path.

This module deliberately uses the loaded LM's runtime codebook/delay configuration.
It does not assume a fixed PersonaPlex codebook count or stream offset.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn.functional as F

from .contracts import StreamLayout


@dataclass
class NativePrefixOutput:
    audio_logits: Tensor
    audio_mask: Tensor
    text_logits: Tensor
    text_mask: Tensor


@dataclass
class LossBreakdown:
    total: Tensor
    text: Tensor
    audio: Tensor
    text_tokens: int
    audio_tokens: int


def _require_uniform_prefix_position(prefix_at: Tensor | int, batch_size: int, sequence_length: int) -> int:
    if isinstance(prefix_at, Tensor):
        if prefix_at.numel() != batch_size or not torch.all(prefix_at == prefix_at.reshape(-1)[0]):
            raise ValueError("batch must be bucketed to one shared prefix_at position")
        prefix_at = int(prefix_at.reshape(-1)[0].item())
    if not isinstance(prefix_at, int) or prefix_at < 1 or prefix_at >= sequence_length:
        raise ValueError("prefix_at must be inside the delayed input sequence")
    return prefix_at


def _drop_inserted_prefix(sequence: Tensor, prefix_at: int, prefix_frames: int) -> Tensor:
    return torch.cat(
        [sequence[:, :prefix_at], sequence[:, prefix_at + prefix_frames :]], dim=1
    )


def forward_with_semantic_prefix(
    lm_model: object,
    codes: Tensor,
    prefix_embeddings: Tensor,
    prefix_at: Tensor | int,
) -> NativePrefixOutput:
    """Runs native training logic with a prefix inserted before the next agent turn.

    `codes` is [B, K, T] and follows the loaded model layout. The explicit target
    mask is handled by `agent_only_loss`; this function only preserves native delay
    and undelay semantics.
    """
    if codes.ndim != 3:
        raise ValueError("codes must have shape [batch, codebooks, frames]")
    batch, codebooks, frames = codes.shape
    if codebooks != lm_model.num_codebooks:
        raise ValueError(f"codebooks={codebooks} does not match loaded model={lm_model.num_codebooks}")
    if prefix_embeddings.ndim != 3 or prefix_embeddings.shape[0] != batch:
        raise ValueError("prefix_embeddings must have shape [batch, prefix_frames, hidden]")
    prefix_at_int = _require_uniform_prefix_position(prefix_at, batch, frames)
    if prefix_embeddings.shape[-1] != lm_model.embed_codes(codes[:, :, :1]).shape[-1]:
        raise ValueError("prefix hidden size does not match loaded PersonaPlex embeddings")

    # Imported lazily so this suite is usable for schema tooling without Moshi installed.
    from moshi.models.lm import _delay_sequence, _undelay_sequence

    initial = lm_model._get_initial_token().expand(batch, -1, -1)
    delayed = _delay_sequence(lm_model.delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)
    model_inputs = delayed[:, :, :-1]
    target_codes = delayed[:, :, 1:]
    base_embeddings = lm_model.embed_codes(model_inputs)
    injected = torch.cat(
        [
            base_embeddings[:, :prefix_at_int],
            prefix_embeddings.to(dtype=base_embeddings.dtype),
            base_embeddings[:, prefix_at_int:],
        ],
        dim=1,
    )
    transformer_all, text_all = lm_model.forward_embeddings(injected)
    transformer_out = _drop_inserted_prefix(transformer_all, prefix_at_int, prefix_embeddings.shape[1])
    text_logits = _drop_inserted_prefix(text_all, prefix_at_int, prefix_embeddings.shape[1])
    audio_logits = lm_model.forward_depformer_training(target_codes, transformer_out)
    audio_logits, audio_mask = _undelay_sequence(
        lm_model.delays[lm_model.audio_offset : lm_model.audio_offset + lm_model.dep_q],
        audio_logits,
        fill_value=float("nan"),
    )
    audio_mask &= target_codes[:, lm_model.audio_offset : lm_model.audio_offset + lm_model.dep_q] != lm_model.zero_token_id
    text_logits, text_mask = _undelay_sequence(lm_model.delays[:1], text_logits, fill_value=float("nan"))
    text_mask &= target_codes[:, :1] != lm_model.zero_token_id
    return NativePrefixOutput(audio_logits, audio_mask, text_logits, text_mask)


def _masked_cross_entropy(logits: Tensor, targets: Tensor, valid: Tensor) -> tuple[Tensor, int]:
    valid = valid.bool()
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits[valid], targets[valid]), count


def agent_only_loss(
    lm_model: object,
    output: NativePrefixOutput,
    codes: Tensor,
    agent_target_mask: Tensor,
    stream_layout: StreamLayout,
    *,
    audio_weight: float = 0.02,
) -> LossBreakdown:
    """Computes loss only for named agent streams, never caller audio.

    ``lm_model.dep_q`` covers every audio stream consumed by the depformer.  In
    duplex PersonaPlex that includes the caller's eight input streams, so slicing
    ``audio_offset:audio_offset + dep_q`` would silently optimize on caller audio.
    """
    if agent_target_mask.shape != codes.shape or agent_target_mask.dtype != torch.bool:
        raise ValueError("agent_target_mask must be bool and match codes [B, K, T]")
    stream_layout.validate_for_model(lm_model)
    if codes.shape[1] != lm_model.num_codebooks:
        raise ValueError("codes do not match the loaded model stream count")

    text_index = stream_layout.text_stream_indices[0]
    allowed_target_indices = stream_layout.text_stream_indices + stream_layout.agent_audio_stream_indices
    forbidden_target_indices = sorted(set(range(codes.shape[1])) - set(allowed_target_indices))
    if forbidden_target_indices and agent_target_mask[:, forbidden_target_indices].any():
        raise ValueError("caller or unknown stream is marked as an optimization target")

    agent_global_indices = torch.tensor(
        stream_layout.agent_audio_stream_indices, device=codes.device, dtype=torch.long
    )
    agent_output_indices = torch.tensor(
        stream_layout.agent_audio_output_indices(lm_model),
        device=output.audio_logits.device,
        dtype=torch.long,
    )
    text_valid = output.text_mask[:, :1] & agent_target_mask[:, text_index : text_index + 1]
    audio_logits = output.audio_logits.index_select(1, agent_output_indices)
    audio_mask = output.audio_mask.index_select(1, agent_output_indices)
    audio_targets = codes.index_select(1, agent_global_indices)
    audio_targets_mask = agent_target_mask.index_select(1, agent_global_indices)
    audio_valid = audio_mask & audio_targets_mask
    text_loss, text_tokens = _masked_cross_entropy(
        output.text_logits[:, 0], codes[:, text_index], text_valid[:, 0]
    )
    audio_loss, audio_tokens = _masked_cross_entropy(
        audio_logits,
        audio_targets,
        audio_valid,
    )
    return LossBreakdown(
        total=text_loss + audio_weight * audio_loss,
        text=text_loss,
        audio=audio_loss,
        text_tokens=text_tokens,
        audio_tokens=audio_tokens,
    )
