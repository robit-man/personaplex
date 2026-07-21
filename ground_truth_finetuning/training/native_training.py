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
class NativeStreamingOutput(NativePrefixOutput):
    transformer_hidden: Tensor


@dataclass
class LossBreakdown:
    total: Tensor
    text: Tensor
    audio: Tensor
    text_tokens: int
    audio_tokens: int


@dataclass
class PerExampleLossBreakdown:
    total: Tensor
    text: Tensor
    audio: Tensor
    text_tokens: Tensor
    audio_tokens: Tensor


@dataclass(frozen=True)
class ExactTextContrast:
    """Exact loss-only token differences for a certified counterfactual pair."""

    mask_a: Tensor
    mask_b: Tensor
    shared_tokens: int
    changed_tokens_a: int
    changed_tokens_b: int


def _require_uniform_prefix_position(prefix_at: Tensor | int, batch_size: int, sequence_length: int) -> int:
    if isinstance(prefix_at, Tensor):
        if prefix_at.numel() != batch_size or not torch.all(prefix_at == prefix_at.reshape(-1)[0]):
            raise ValueError("batch must be bucketed to one shared prefix_at position")
        prefix_at = int(prefix_at.reshape(-1)[0].item())
    if not isinstance(prefix_at, int) or prefix_at < 1 or prefix_at >= sequence_length:
        raise ValueError("prefix_at must be inside the delayed input sequence")
    return prefix_at


def _drop_inserted_prefix(sequence: Tensor, prefix_at: int, prefix_frames: int, *, time_dimension: int) -> Tensor:
    if sequence.shape[time_dimension] < prefix_at + prefix_frames:
        raise ValueError("native output is shorter than the inserted semantic prefix")
    before = [slice(None)] * sequence.ndim
    after = [slice(None)] * sequence.ndim
    before[time_dimension] = slice(None, prefix_at)
    after[time_dimension] = slice(prefix_at + prefix_frames, None)
    return torch.cat(
        [sequence[tuple(before)], sequence[tuple(after)]], dim=time_dimension
    )


def forward_with_semantic_prefix(
    lm_model: object,
    codes: Tensor,
    prefix_embeddings: Tensor,
    prefix_at: Tensor | int,
    *,
    activation_checkpointing: bool = True,
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
    if activation_checkpointing:
        from torch.utils.checkpoint import checkpoint

        transformer_all, text_all = checkpoint(
            lambda embeddings: lm_model.forward_embeddings(embeddings),
            injected,
            use_reentrant=False,
        )
    else:
        transformer_all, text_all = lm_model.forward_embeddings(injected)
    transformer_out = _drop_inserted_prefix(
        transformer_all, prefix_at_int, prefix_embeddings.shape[1], time_dimension=1
    )
    text_logits = _drop_inserted_prefix(
        text_all, prefix_at_int, prefix_embeddings.shape[1], time_dimension=2
    )
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


def forward_with_semantic_prefix_and_evidence(
    lm_model: object,
    codes: Tensor,
    prefix_embeddings: Tensor,
    evidence_stream: Tensor,
    prefix_at: Tensor | int,
    *,
    activation_checkpointing: bool = True,
) -> NativePrefixOutput:
    """Train causal delayed evidence against the same native agent-only targets.

    ``evidence_stream`` is an adapter output, never target text.  Its rows are
    added to the model input from the post-prefix response boundary onward,
    matching the patched runtime's one-row-per-live-step streaming-sum contract.
    The method deliberately fails closed until the maintained upstream patch is
    installed; silently treating evidence as a prompt or a prefix would invalidate
    the training/runtime equivalence.
    """
    if evidence_stream.ndim != 3 or evidence_stream.shape[0] != codes.shape[0]:
        raise ValueError("evidence_stream must have shape [batch, evidence_frames, hidden]")
    if evidence_stream.shape[1] < 1:
        raise ValueError("evidence_stream must contain at least one frame")
    batch, codebooks, frames = codes.shape
    if codebooks != lm_model.num_codebooks:
        raise ValueError("codes do not match loaded PersonaPlex codebooks")
    prefix_at_int = _require_uniform_prefix_position(prefix_at, batch, frames)
    if prefix_embeddings.ndim != 3 or prefix_embeddings.shape[0] != batch:
        raise ValueError("prefix_embeddings must have shape [batch, prefix_frames, hidden]")
    from moshi.models.lm import _delay_sequence, _undelay_sequence

    initial = lm_model._get_initial_token().expand(batch, -1, -1)
    delayed = _delay_sequence(lm_model.delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)
    model_inputs = delayed[:, :, :-1]
    target_codes = delayed[:, :, 1:]
    base_embeddings = lm_model.embed_codes(model_inputs)
    if prefix_embeddings.shape[-1] != base_embeddings.shape[-1] or evidence_stream.shape[-1] != base_embeddings.shape[-1]:
        raise ValueError("control prefix and evidence stream must match PersonaPlex hidden size")
    injected = torch.cat(
        [
            base_embeddings[:, :prefix_at_int],
            prefix_embeddings.to(dtype=base_embeddings.dtype),
            base_embeddings[:, prefix_at_int:],
        ],
        dim=1,
    )
    streaming_sum = torch.zeros_like(injected)
    start = prefix_at_int + prefix_embeddings.shape[1]
    usable = min(evidence_stream.shape[1], streaming_sum.shape[1] - start)
    if usable < 1:
        raise ValueError("no post-boundary native frames remain for delayed evidence")
    streaming_sum[:, start : start + usable] = evidence_stream[:, :usable].to(dtype=injected.dtype)
    try:
        if activation_checkpointing:
            from torch.utils.checkpoint import checkpoint

            transformer_all, text_all = checkpoint(
                lambda embeddings, condition: lm_model.forward_embeddings(embeddings, streaming_sum=condition),
                injected,
                streaming_sum,
                use_reentrant=False,
            )
        else:
            transformer_all, text_all = lm_model.forward_embeddings(injected, streaming_sum=streaming_sum)
    except TypeError as exc:
        raise RuntimeError(
            "native PersonaPlex source lacks streaming_sum support; apply the maintained Moshirag compatibility patch"
        ) from exc
    transformer_out = _drop_inserted_prefix(
        transformer_all, prefix_at_int, prefix_embeddings.shape[1], time_dimension=1
    )
    text_logits = _drop_inserted_prefix(
        text_all, prefix_at_int, prefix_embeddings.shape[1], time_dimension=2
    )
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


def align_control_stream(control_stream: Tensor, start_positions: Tensor, total_frames: int) -> Tensor:
    """Align compressed control rows to real native frames without virtual prefill."""
    if control_stream.ndim != 3:
        raise ValueError("control_stream must be [batch, control_frames, hidden]")
    if start_positions.ndim != 1 or start_positions.shape[0] != control_stream.shape[0]:
        raise ValueError("start_positions must contain one index per control stream")
    if total_frames < 1:
        raise ValueError("total_frames must be positive")
    if torch.any(start_positions < 1) or torch.any(start_positions >= total_frames):
        raise ValueError("control start position must lie inside native model inputs")
    positions = torch.arange(total_frames, device=control_stream.device).unsqueeze(0)
    relative = positions - start_positions.unsqueeze(1)
    valid = (relative >= 0) & (relative < control_stream.shape[1])
    indices = relative.clamp(min=0, max=control_stream.shape[1] - 1)
    aligned = control_stream.gather(
        1, indices.unsqueeze(-1).expand(-1, -1, control_stream.shape[-1])
    )
    return aligned * valid.unsqueeze(-1).to(aligned.dtype)


def forward_with_control_stream(
    lm_model: object,
    codes: Tensor,
    control_stream: Tensor,
    control_at: Tensor | int,
    *,
    layer_control_streams: Tensor | None = None,
    layer_indices: tuple[int, ...] = (),
    layer_adapter_down: Tensor | None = None,
    layer_adapter_up: Tensor | None = None,
    layer_adapter_gates: Tensor | None = None,
    max_layer_adaptation_rms: float | None = None,
    activation_checkpointing: bool = True,
) -> NativePrefixOutput:
    """Condition native delayed-code training on a MoshiRAG-style temporal stream."""
    if codes.ndim != 3:
        raise ValueError("codes must have shape [batch, codebooks, frames]")
    batch, codebooks, frames = codes.shape
    if codebooks != lm_model.num_codebooks:
        raise ValueError("codes do not match the loaded PersonaPlex codebook count")
    if control_stream.ndim != 3 or control_stream.shape[0] != batch:
        raise ValueError("control_stream must be [batch, control_frames, hidden]")
    if isinstance(control_at, int):
        starts = torch.full((batch,), control_at, device=codes.device, dtype=torch.long)
    else:
        starts = control_at.reshape(-1).to(device=codes.device, dtype=torch.long)
    from moshi.models.lm import _delay_sequence, _undelay_sequence

    initial = lm_model._get_initial_token().expand(batch, -1, -1)
    delayed = _delay_sequence(lm_model.delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)
    model_inputs = delayed[:, :, :-1]
    target_codes = delayed[:, :, 1:]
    base_embeddings = lm_model.embed_codes(model_inputs)
    if control_stream.shape[-1] != base_embeddings.shape[-1]:
        raise ValueError("control stream width does not match PersonaPlex hidden size")
    streaming_sum = align_control_stream(
        control_stream.to(device=base_embeddings.device, dtype=base_embeddings.dtype),
        starts,
        base_embeddings.shape[1],
    )
    aligned_layers = None
    if layer_control_streams is not None:
        if (
            layer_control_streams.ndim != 4
            or layer_control_streams.shape[0] != batch
            or layer_control_streams.shape[-1] != base_embeddings.shape[-1]
            or layer_control_streams.shape[1] != len(layer_indices)
        ):
            raise ValueError(
                "layer control streams must be [batch, selected_layers, frames, hidden]"
            )
        selected_layers = layer_control_streams.shape[1]
        flat_layers = layer_control_streams.reshape(
            batch * selected_layers,
            layer_control_streams.shape[2],
            layer_control_streams.shape[3],
        )
        aligned_layers = align_control_stream(
            flat_layers.to(device=base_embeddings.device, dtype=base_embeddings.dtype),
            starts.repeat_interleave(selected_layers),
            base_embeddings.shape[1],
        ).reshape(
            batch,
            selected_layers,
            base_embeddings.shape[1],
            base_embeddings.shape[-1],
        )
    adapter_tensors = (layer_adapter_down, layer_adapter_up, layer_adapter_gates)
    has_layer_adapter = any(value is not None for value in adapter_tensors)
    if has_layer_adapter and not all(value is not None for value in adapter_tensors):
        raise ValueError("layer adaptation requires down, up, and gate tensors together")
    if has_layer_adapter and max_layer_adaptation_rms is None:
        raise ValueError("layer adaptation requires a bounded RMS limit")
    try:
        if activation_checkpointing:
            from torch.utils.checkpoint import checkpoint

            if aligned_layers is None:
                transformer_out, text_logits = checkpoint(
                    lambda embeddings, condition: lm_model.forward_embeddings(
                        embeddings, streaming_sum=condition
                    ),
                    base_embeddings,
                    streaming_sum,
                    use_reentrant=False,
                )
            elif not has_layer_adapter:
                transformer_out, text_logits = checkpoint(
                    lambda embeddings, condition, layers: lm_model.forward_embeddings(
                        embeddings,
                        streaming_sum=condition,
                        layerwise_sum=layers,
                        layer_indices=layer_indices,
                    ),
                    base_embeddings,
                    streaming_sum,
                    aligned_layers,
                    use_reentrant=False,
                )
            else:
                transformer_out, text_logits = checkpoint(
                    lambda embeddings, condition, layers, down, up, gates: lm_model.forward_embeddings(
                        embeddings,
                        streaming_sum=condition,
                        layerwise_sum=layers,
                        layer_indices=layer_indices,
                        layer_adapter_down=down,
                        layer_adapter_up=up,
                        layer_adapter_gates=gates,
                        max_layer_adaptation_rms=max_layer_adaptation_rms,
                    ),
                    base_embeddings,
                    streaming_sum,
                    aligned_layers,
                    layer_adapter_down,
                    layer_adapter_up,
                    layer_adapter_gates,
                    use_reentrant=False,
                )
        else:
            transformer_out, text_logits = lm_model.forward_embeddings(
                base_embeddings,
                streaming_sum=streaming_sum,
                layerwise_sum=aligned_layers,
                layer_indices=layer_indices,
                layer_adapter_down=layer_adapter_down,
                layer_adapter_up=layer_adapter_up,
                layer_adapter_gates=layer_adapter_gates,
                max_layer_adaptation_rms=max_layer_adaptation_rms,
            )
    except TypeError as exc:
        raise RuntimeError(
            "native PersonaPlex source lacks streaming_sum support; apply the maintained MoshiRAG patch"
        ) from exc
    audio_logits = lm_model.forward_depformer_training(target_codes, transformer_out)
    audio_logits, audio_mask = _undelay_sequence(
        lm_model.delays[lm_model.audio_offset : lm_model.audio_offset + lm_model.dep_q],
        audio_logits,
        fill_value=float("nan"),
    )
    audio_mask &= (
        target_codes[:, lm_model.audio_offset : lm_model.audio_offset + lm_model.dep_q]
        != lm_model.zero_token_id
    )
    text_logits, text_mask = _undelay_sequence(
        lm_model.delays[:1], text_logits, fill_value=float("nan")
    )
    text_mask &= target_codes[:, :1] != lm_model.zero_token_id
    return NativePrefixOutput(audio_logits, audio_mask, text_logits, text_mask)


def forward_with_native_streaming_sum(
    lm_model: object,
    codes: Tensor,
    reference_stream: Tensor,
    condition_start_frames: Tensor | int,
    *,
    cancel_at_frames: Tensor | None = None,
    control_dropout_mask: Tensor | None = None,
    activation_checkpointing: bool = True,
) -> NativeStreamingOutput:
    """Run native delayed duplex training with one immutable ARC stream per row."""

    if codes.ndim != 3:
        raise ValueError("codes must have shape [batch, codebooks, frames]")
    batch, codebooks, _frames = codes.shape
    if codebooks != lm_model.num_codebooks:
        raise ValueError("codes do not match loaded PersonaPlex codebooks")
    if reference_stream.ndim != 3 or reference_stream.shape[0] != batch:
        raise ValueError("reference_stream must have shape [batch, frames, hidden]")
    if reference_stream.shape[1] < 1:
        raise ValueError("reference_stream must contain at least one frame")
    from moshi.models.lm import _delay_sequence, _undelay_sequence

    initial = lm_model._get_initial_token().expand(batch, -1, -1)
    delayed = _delay_sequence(lm_model.delays, codes, initial)
    delayed = torch.cat([initial, delayed], dim=2)
    model_inputs = delayed[:, :, :-1]
    target_codes = delayed[:, :, 1:]
    base_embeddings = lm_model.embed_codes(model_inputs)
    if reference_stream.shape[-1] != base_embeddings.shape[-1]:
        raise ValueError("ARC stream hidden size does not match PersonaPlex")
    if isinstance(condition_start_frames, int):
        starts = torch.full(
            (batch,), condition_start_frames, device=codes.device, dtype=torch.long
        )
    else:
        starts = condition_start_frames.to(device=codes.device, dtype=torch.long).reshape(-1)
    if starts.shape[0] != batch or bool(((starts < 0) | (starts > base_embeddings.shape[1])).any().item()):
        raise ValueError("condition_start_frames must contain one valid frame per row")
    if cancel_at_frames is None:
        cancels = torch.full((batch,), -1, device=codes.device, dtype=torch.long)
    else:
        cancels = cancel_at_frames.to(device=codes.device, dtype=torch.long).reshape(-1)
        if cancels.shape[0] != batch or bool((cancels < -1).any().item()):
            raise ValueError("cancel_at_frames must contain -1 or one frame per row")
    if control_dropout_mask is None:
        dropped = torch.zeros(batch, device=codes.device, dtype=torch.bool)
    else:
        dropped = control_dropout_mask.to(device=codes.device, dtype=torch.bool).reshape(-1)
        if dropped.shape[0] != batch:
            raise ValueError("control_dropout_mask must contain one value per row")

    streaming_sum = torch.zeros_like(base_embeddings)
    for row in range(batch):
        if bool(dropped[row].item()):
            continue
        start = int(starts[row].item())
        stop = base_embeddings.shape[1]
        cancel = int(cancels[row].item())
        if cancel >= 0:
            stop = min(stop, cancel)
        if start >= stop:
            continue
        usable = min(reference_stream.shape[1], stop - start)
        streaming_sum[row, start : start + usable] = reference_stream[row, :usable].to(
            dtype=base_embeddings.dtype
        )
    try:
        if activation_checkpointing:
            from torch.utils.checkpoint import checkpoint

            transformer_out, text_logits = checkpoint(
                lambda embeddings, condition: lm_model.forward_embeddings(
                    embeddings, streaming_sum=condition
                ),
                base_embeddings,
                streaming_sum,
                use_reentrant=False,
            )
        else:
            transformer_out, text_logits = lm_model.forward_embeddings(
                base_embeddings, streaming_sum=streaming_sum
            )
    except TypeError as exc:
        raise RuntimeError(
            "native PersonaPlex source lacks streaming_sum support; apply the maintained MoshiRAG patch"
        ) from exc
    audio_logits = lm_model.forward_depformer_training(target_codes, transformer_out)
    audio_logits, audio_mask = _undelay_sequence(
        lm_model.delays[lm_model.audio_offset : lm_model.audio_offset + lm_model.dep_q],
        audio_logits,
        fill_value=float("nan"),
    )
    audio_mask &= (
        target_codes[:, lm_model.audio_offset : lm_model.audio_offset + lm_model.dep_q]
        != lm_model.zero_token_id
    )
    text_logits, text_mask = _undelay_sequence(
        lm_model.delays[:1], text_logits, fill_value=float("nan")
    )
    text_mask &= target_codes[:, :1] != lm_model.zero_token_id
    return NativeStreamingOutput(
        audio_logits=audio_logits,
        audio_mask=audio_mask,
        text_logits=text_logits,
        text_mask=text_mask,
        transformer_hidden=transformer_out,
    )


def _masked_cross_entropy(logits: Tensor, targets: Tensor, valid: Tensor) -> tuple[Tensor, int]:
    valid = valid.bool()
    count = int(valid.sum().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits[valid].float(), targets[valid]), count


def exact_text_contrast_masks(
    codes_a: Tensor,
    target_mask_a: Tensor,
    codes_b: Tensor,
    target_mask_b: Tensor,
    *,
    text_stream_index: int,
    zero_token_id: int,
) -> ExactTextContrast:
    """Mark target text tokens outside an exact longest common subsequence.

    The target labels are used only to construct a loss mask. They are never
    serialized into the semantic control frame or provided to the adapter. The
    dynamic program is exact and deterministic; it has no lexical rules, regexes,
    semantic judge, or approximate sequence matcher.
    """
    if codes_a.ndim != 2 or codes_b.ndim != 2:
        raise ValueError("paired codes must be [codebooks, frames]")
    if target_mask_a.shape != codes_a.shape or target_mask_b.shape != codes_b.shape:
        raise ValueError("paired target masks must match their code tensors")
    if target_mask_a.dtype != torch.bool or target_mask_b.dtype != torch.bool:
        raise ValueError("paired target masks must be bool")
    if not 0 <= text_stream_index < codes_a.shape[0] or not 0 <= text_stream_index < codes_b.shape[0]:
        raise ValueError("text stream index is outside paired code tensors")

    positions_a = torch.where(
        target_mask_a[text_stream_index]
        & codes_a[text_stream_index].ne(zero_token_id)
    )[0]
    positions_b = torch.where(
        target_mask_b[text_stream_index]
        & codes_b[text_stream_index].ne(zero_token_id)
    )[0]
    tokens_a = [int(value) for value in codes_a[text_stream_index, positions_a].tolist()]
    tokens_b = [int(value) for value in codes_b[text_stream_index, positions_b].tolist()]
    if not tokens_a or not tokens_b:
        raise ValueError("counterfactual pair has no supervised text tokens")

    # Exact LCS lengths. Target turns are short, so the quadratic table is small
    # and preferable to an approximate or heuristic alignment.
    lengths = [[0] * (len(tokens_b) + 1) for _ in range(len(tokens_a) + 1)]
    for i, token_a in enumerate(tokens_a, start=1):
        previous = lengths[i - 1]
        current = lengths[i]
        for j, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])

    shared_a: set[int] = set()
    shared_b: set[int] = set()
    i, j = len(tokens_a), len(tokens_b)
    while i and j:
        if tokens_a[i - 1] == tokens_b[j - 1]:
            shared_a.add(i - 1)
            shared_b.add(j - 1)
            i -= 1
            j -= 1
        elif lengths[i - 1][j] >= lengths[i][j - 1]:
            i -= 1
        else:
            j -= 1

    mask_a = torch.zeros_like(target_mask_a, dtype=torch.bool)
    mask_b = torch.zeros_like(target_mask_b, dtype=torch.bool)
    for index, position in enumerate(positions_a.tolist()):
        if index not in shared_a:
            mask_a[text_stream_index, int(position)] = True
    for index, position in enumerate(positions_b.tolist()):
        if index not in shared_b:
            mask_b[text_stream_index, int(position)] = True
    changed_a = int(mask_a.sum().item())
    changed_b = int(mask_b.sum().item())
    if changed_a < 1 or changed_b < 1:
        raise ValueError(
            "counterfactual pair must contain branch-distinct target text in both directions"
        )
    return ExactTextContrast(
        mask_a=mask_a,
        mask_b=mask_b,
        shared_tokens=len(shared_a),
        changed_tokens_a=changed_a,
        changed_tokens_b=changed_b,
    )


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


def agent_only_loss_per_example(
    lm_model: object,
    output: NativePrefixOutput,
    codes: Tensor,
    agent_target_mask: Tensor,
    stream_layout: StreamLayout,
    *,
    audio_weight: float = 0.02,
) -> PerExampleLossBreakdown:
    """Return independent losses so matched and cross-control NLL can be compared."""
    if agent_target_mask.shape != codes.shape or agent_target_mask.dtype != torch.bool:
        raise ValueError("agent_target_mask must be bool and match codes")
    stream_layout.validate_for_model(lm_model)
    text_index = stream_layout.text_stream_indices[0]
    forbidden = sorted(
        set(range(codes.shape[1]))
        - set(stream_layout.text_stream_indices + stream_layout.agent_audio_stream_indices)
    )
    if forbidden and agent_target_mask[:, forbidden].any():
        raise ValueError("caller or unknown streams cannot be optimization targets")
    agent_global = torch.tensor(
        stream_layout.agent_audio_stream_indices, device=codes.device, dtype=torch.long
    )
    agent_output = torch.tensor(
        stream_layout.agent_audio_output_indices(lm_model),
        device=output.audio_logits.device,
        dtype=torch.long,
    )
    text_valid = output.text_mask[:, 0] & agent_target_mask[:, text_index]
    audio_logits = output.audio_logits.index_select(1, agent_output)
    audio_valid = output.audio_mask.index_select(1, agent_output)
    audio_valid &= agent_target_mask.index_select(1, agent_global)
    audio_targets = codes.index_select(1, agent_global)
    text_losses: list[Tensor] = []
    audio_losses: list[Tensor] = []
    text_counts: list[int] = []
    audio_counts: list[int] = []
    for row in range(codes.shape[0]):
        row_text_valid = text_valid[row]
        row_audio_valid = audio_valid[row]
        text_count = int(row_text_valid.sum().item())
        audio_count = int(row_audio_valid.sum().item())
        text_loss = (
            F.cross_entropy(
                output.text_logits[row, 0][row_text_valid].float(),
                codes[row, text_index][row_text_valid],
            )
            if text_count
            else output.text_logits[row].sum() * 0.0
        )
        audio_loss = (
            F.cross_entropy(audio_logits[row][row_audio_valid], audio_targets[row][row_audio_valid])
            if audio_count
            else audio_logits[row].sum() * 0.0
        )
        text_losses.append(text_loss)
        audio_losses.append(audio_loss)
        text_counts.append(text_count)
        audio_counts.append(audio_count)
    text = torch.stack(text_losses)
    audio = torch.stack(audio_losses)
    return PerExampleLossBreakdown(
        total=text + audio_weight * audio,
        text=text,
        audio=audio,
        text_tokens=torch.tensor(text_counts, device=text.device, dtype=torch.long),
        audio_tokens=torch.tensor(audio_counts, device=audio.device, dtype=torch.long),
    )
