"""Trainable temporal semantic-control stream using frozen PersonaPlex lexical embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .control_encoding import (
    FIELD_NAMES,
    REVISION_BUCKET_COUNT,
    SOURCE_NAMES,
    VALUE_KIND_NAMES,
    ControlTensorBatch,
)


@dataclass(frozen=True)
class ControlStreamConfig:
    control_dim: int = 1024
    encoder_layers: int = 4
    attention_heads: int = 8
    feedforward_multiplier: int = 3
    stream_frames: int = 48
    max_tokens: int = 512
    dropout: float = 0.10
    architecture_revision: str = "lexical-attention-rms-bounded-v4"
    max_context_gate_adjustment: float = 0.05
    max_stream_to_lexical_rms_ratio: float = 0.25

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ControlStreamConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown control-stream configuration fields: {sorted(unknown)}")
        return cls(**{key: value[key] for key in known if key in value})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticControlStreamAdapter(nn.Module):
    """Compress typed lexical control into rows added to real duplex frames."""

    def __init__(self, *, lm_hidden_size: int, config: ControlStreamConfig | None = None) -> None:
        super().__init__()
        self.config = config or ControlStreamConfig()
        cfg = self.config
        if cfg.control_dim % cfg.attention_heads:
            raise ValueError("control_dim must be divisible by attention_heads")
        if cfg.architecture_revision != "lexical-attention-rms-bounded-v4":
            raise ValueError("unsupported semantic-control adapter architecture revision")
        if not 0 < cfg.max_context_gate_adjustment <= 1:
            raise ValueError("max_context_gate_adjustment must be in (0, 1]")
        if not 0 < cfg.max_stream_to_lexical_rms_ratio <= 1:
            raise ValueError("max_stream_to_lexical_rms_ratio must be in (0, 1]")
        if not 1 <= cfg.stream_frames <= 256 or not 32 <= cfg.max_tokens <= 2048:
            raise ValueError("control stream_frames or max_tokens is outside the supported range")
        self.lm_hidden_size = lm_hidden_size
        self.text_projection = nn.Linear(lm_hidden_size, cfg.control_dim, bias=False)
        self.field_embedding = nn.Embedding(len(FIELD_NAMES), cfg.control_dim, padding_idx=0)
        self.value_kind_embedding = nn.Embedding(len(VALUE_KIND_NAMES), cfg.control_dim, padding_idx=0)
        self.source_embedding = nn.Embedding(len(SOURCE_NAMES), cfg.control_dim)
        self.revision_embedding = nn.Embedding(REVISION_BUCKET_COUNT, cfg.control_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(cfg.max_tokens, cfg.control_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.control_dim,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.control_dim * cfg.feedforward_multiplier,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.encoder_layers)
        self.compression_queries = nn.Parameter(torch.empty(cfg.stream_frames, cfg.control_dim))
        self.cross_attention = nn.MultiheadAttention(
            cfg.control_dim,
            cfg.attention_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(cfg.control_dim, lm_hidden_size, bias=False)
        self.output_norm = nn.LayerNorm(lm_hidden_size)
        self.lexical_output_norm = nn.LayerNorm(lm_hidden_size)
        # Zero-centered residual gates make an untrained adapter an exact no-op.
        # This preserves the frozen base at initialization while retaining a
        # nonzero derivative so causal supervision can grow either path.
        self.gate_bias = nn.Parameter(torch.zeros(cfg.stream_frames))
        self.lexical_gate_bias = nn.Parameter(torch.zeros(cfg.stream_frames))
        self.context_gate = nn.Sequential(
            nn.LayerNorm(cfg.control_dim),
            nn.Linear(cfg.control_dim, cfg.stream_frames * 2),
        )
        self.stream_dropout = nn.Dropout(cfg.dropout)
        nn.init.normal_(self.compression_queries, mean=0.0, std=0.02)
        nn.init.zeros_(self.context_gate[-1].weight)
        nn.init.zeros_(self.context_gate[-1].bias)
        self._last_effective_gate_abs_mean: Tensor | None = None
        self._last_stream_to_lexical_rms_ratio: Tensor | None = None

    def _source_features(self, masks: Tensor, dtype: torch.dtype) -> Tensor:
        bits = torch.arange(len(SOURCE_NAMES), device=masks.device, dtype=torch.long)
        active = ((masks.unsqueeze(-1) >> bits) & 1).to(dtype=dtype)
        return active @ self.source_embedding.weight.to(dtype=dtype)

    def forward(self, lexical_embeddings: Tensor, batch: ControlTensorBatch) -> Tensor:
        if lexical_embeddings.ndim != 3:
            raise ValueError("lexical_embeddings must be [batch, tokens, PersonaPlex hidden]")
        if lexical_embeddings.shape[:2] != batch.token_ids.shape:
            raise ValueError("lexical embeddings and typed control channels are not aligned")
        if lexical_embeddings.shape[-1] != self.lm_hidden_size:
            raise ValueError("lexical embedding width does not match the configured PersonaPlex model")
        if batch.token_ids.shape[1] > self.config.max_tokens:
            raise ValueError("control sequence exceeds adapter max_tokens")
        if not torch.all(batch.attention_mask.any(dim=1)):
            raise ValueError("every control row must contain at least one lexical token")
        parameter_dtype = self.text_projection.weight.dtype
        lexical = lexical_embeddings.detach().to(dtype=parameter_dtype)
        positions = torch.arange(batch.token_ids.shape[1], device=batch.token_ids.device)
        hidden = self.text_projection(lexical)
        hidden = hidden + self.field_embedding(batch.field_ids)
        hidden = hidden + self.value_kind_embedding(batch.value_kind_ids)
        hidden = hidden + self.revision_embedding(batch.revision_ids)
        hidden = hidden + self.position_embedding(positions).unsqueeze(0)
        hidden = hidden + self._source_features(batch.source_masks, hidden.dtype)
        key_padding = ~batch.attention_mask.bool()
        memory = self.encoder(hidden, src_key_padding_mask=key_padding)
        queries = self.compression_queries.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        compressed, lexical_attention = self.cross_attention(
            queries,
            memory,
            memory,
            key_padding_mask=key_padding,
            need_weights=True,
            average_attn_weights=True,
        )
        if lexical_attention.ndim != 3:
            raise RuntimeError("control cross-attention did not return per-frame lexical weights")
        lexical_stream = torch.bmm(
            lexical_attention.to(dtype=lexical.dtype), lexical
        )
        weights = batch.attention_mask.to(memory.dtype).unsqueeze(-1)
        pooled = (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        context_adjustment = self.context_gate(pooled).reshape(
            hidden.shape[0], 2, self.config.stream_frames
        )
        context_adjustment = self.config.max_context_gate_adjustment * torch.tanh(
            context_adjustment
        )
        learned_gates = torch.tanh(
            self.gate_bias.unsqueeze(0) + context_adjustment[:, 0]
        ).unsqueeze(-1)
        lexical_gates = torch.tanh(
            self.lexical_gate_bias.unsqueeze(0) + context_adjustment[:, 1]
        ).unsqueeze(-1)
        stream = (
            self.output_norm(self.output_projection(compressed)) * learned_gates
            + self.lexical_output_norm(lexical_stream) * lexical_gates
        )
        lexical_weights = batch.attention_mask.to(lexical.dtype).unsqueeze(-1)
        lexical_count = lexical_weights.sum(dim=(1, 2)).clamp_min(1.0)
        lexical_rms = (
            (lexical.float().square() * lexical_weights.float()).sum(dim=(1, 2))
            / (lexical_count.float() * lexical.shape[-1])
        ).sqrt()
        epsilon = torch.finfo(torch.float32).eps
        stream_power = stream.float().square().mean(dim=(1, 2))
        # The adapter is intentionally an exact zero at initialization. Clamp
        # power before sqrt so the zero point has a finite, zero derivative.
        stream_rms = stream_power.clamp_min(epsilon).sqrt()
        maximum_stream_rms = (
            lexical_rms * self.config.max_stream_to_lexical_rms_ratio
        )
        rms_scale = torch.minimum(
            torch.ones_like(stream_rms),
            maximum_stream_rms / stream_rms,
        )
        stream = stream * rms_scale.to(stream.dtype).view(-1, 1, 1)
        post_cap_rms = stream.detach().float().square().mean(dim=(1, 2)).sqrt()
        self._last_effective_gate_abs_mean = torch.stack(
            (learned_gates.abs().mean(), lexical_gates.abs().mean())
        ).mean().detach()
        self._last_stream_to_lexical_rms_ratio = (
            post_cap_rms / lexical_rms.clamp_min(epsilon)
        ).mean().detach()
        stream = self.stream_dropout(stream)
        present = batch.control_present.to(stream.dtype).view(-1, 1, 1)
        return stream * present

    def mean_gate(self) -> Tensor:
        return torch.stack(
            (
                torch.tanh(self.gate_bias).abs().mean(),
                torch.tanh(self.lexical_gate_bias).abs().mean(),
            )
        ).mean()

    def last_effective_gate(self) -> Tensor:
        if self._last_effective_gate_abs_mean is None:
            return self.mean_gate()
        return self._last_effective_gate_abs_mean

    def last_stream_to_lexical_rms_ratio(self) -> Tensor:
        if self._last_stream_to_lexical_rms_ratio is None:
            return self.mean_gate() * 0.0
        return self._last_stream_to_lexical_rms_ratio
