"""Learned, GPU-only delayed-evidence conditioning for PersonaPlex.

This is deliberately not a prompt adapter.  It maps bounded structured evidence
tokens to a short hidden-state stream and queues that stream through the patched
generator's ``update_streaming_sum_tensors`` API.  A live session snapshots the
result at a turn boundary; no evidence row can rewrite audio already emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - import is environment dependent
    raise RuntimeError("EvidenceStreamAdapter requires PyTorch") from exc


class StreamingConditioningError(RuntimeError):
    """The native source cannot guarantee a faithful evidence-conditioning path."""


class EvidenceStreamAdapter(nn.Module):
    """Map structured evidence token IDs to bounded causal streaming-sum rows."""

    def __init__(
        self,
        *,
        text_cardinality: int,
        hidden_size: int,
        stream_frames: int = 16,
        encoder_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if stream_frames < 1 or stream_frames > 128:
            raise ValueError("stream_frames must be between 1 and 128")
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self.stream_frames = stream_frames
        self.token_embedding = nn.Embedding(text_cardinality, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=attention_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=encoder_layers)
        self.stream_queries = nn.Parameter(torch.empty(stream_frames, hidden_size))
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, attention_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        # A conservative initial gate preserves the frozen conversational base.
        self.gate = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.stream_queries, mean=0.0, std=0.02)

    def forward(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if token_ids.ndim != 2 or attention_mask.shape != token_ids.shape:
            raise ValueError("token_ids and attention_mask must both be [batch, sequence]")
        if not torch.all(attention_mask.any(dim=1)):
            raise ValueError("every evidence example must contain at least one token")
        key_padding = ~attention_mask.bool()
        memory = self.encoder(self.token_embedding(token_ids), src_key_padding_mask=key_padding)
        queries = self.stream_queries.unsqueeze(0).expand(token_ids.shape[0], -1, -1)
        stream, _ = self.cross_attention(queries, memory, memory, key_padding_mask=key_padding, need_weights=False)
        return self.output_norm(stream) * torch.sigmoid(self.gate)


@dataclass(frozen=True)
class EvidenceStreamSnapshot:
    """Immutable, GPU-resident evidence representation for one next-turn generation."""

    revision: int
    context_hash: str
    tensor: Tensor


class MoshiStreamingSumBridge:
    """Strict bridge to the maintained Moshirag-compatible LMGen interface."""

    def __init__(self, lm_gen: object) -> None:
        self.lm_gen = lm_gen
        self._require_supported()

    def _require_supported(self) -> None:
        missing = [
            name
            for name in ("update_streaming_sum_tensors", "apply_pending_streaming_sum_condition")
            if not callable(getattr(self.lm_gen, name, None))
        ]
        if missing:
            raise StreamingConditioningError(
                "native PersonaPlex lacks Moshirag streaming-sum support; "
                "apply personaplex-setup/moshirag_streaming_sum.patch first: "
                + ", ".join(missing)
            )

    @property
    def hidden_size(self) -> int:
        dim = getattr(getattr(self.lm_gen, "lm_model", None), "dim", None)
        if not isinstance(dim, int) or dim < 1:
            raise StreamingConditioningError("loaded PersonaPlex LM does not expose a positive hidden dimension")
        return dim

    @property
    def device(self) -> torch.device:
        device = getattr(getattr(self.lm_gen, "lm_model", None), "device", None)
        resolved = torch.device(device)
        if resolved.type != "cuda":
            raise StreamingConditioningError("evidence conditioning is CUDA-only; CPU inference is prohibited")
        return resolved

    def queue(self, streams: Sequence[Tensor | None]) -> None:
        """Queue one immutable stream per native batch slot for a future turn."""
        state = getattr(self.lm_gen, "_streaming_state", None)
        if state is None:
            raise StreamingConditioningError("queue evidence only inside an active LMGen streaming session")
        batch_size = int(state.cache.shape[0])
        if len(streams) != batch_size:
            raise StreamingConditioningError(f"expected {batch_size} evidence streams, received {len(streams)}")
        normalized: list[Tensor | None] = []
        for stream in streams:
            if stream is None:
                normalized.append(None)
                continue
            if stream.ndim != 2 or stream.shape[0] < 1 or stream.shape[1] != self.hidden_size:
                raise StreamingConditioningError("evidence stream must be [frames>=1, PersonaPlex hidden]")
            if stream.device.type != "cuda":
                raise StreamingConditioningError("evidence stream must already reside on CUDA")
            normalized.append(stream.detach().to(device=self.device, dtype=getattr(self.lm_gen.lm_model, "dtype", stream.dtype)))
        self.lm_gen.update_streaming_sum_tensors(normalized)

    def cancel(self, batch_size: int) -> None:
        """Consume zero rows after a barge-in, preventing stale evidence reuse."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        clear = getattr(self.lm_gen, "clear_streaming_sum_tensors", None)
        if callable(clear):
            clear()
            return
        self.lm_gen.update_streaming_sum_tensors([None] * batch_size)
