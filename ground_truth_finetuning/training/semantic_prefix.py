"""Small trainable semantic-prefix adapter; PersonaPlex base weights remain frozen."""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:  # pragma: no cover - import is environment dependent
    raise RuntimeError("SemanticPrefixAdapter requires PyTorch") from exc


class SemanticPrefixAdapter(nn.Module):
    """Maps plan-token IDs to bounded prefix embeddings in LM hidden space."""

    def __init__(
        self,
        *,
        text_cardinality: int,
        hidden_size: int,
        prefix_frames: int,
        encoder_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if prefix_frames < 1 or prefix_frames > 256:
            raise ValueError("prefix_frames must be between 1 and 256")
        if hidden_size % attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        self.prefix_frames = prefix_frames
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
        self.prefix_queries = nn.Parameter(torch.empty(prefix_frames, hidden_size))
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, attention_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.gate = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.prefix_queries, mean=0.0, std=0.02)

    def forward(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if token_ids.ndim != 2 or attention_mask.shape != token_ids.shape:
            raise ValueError("token_ids and attention_mask must both be [batch, sequence]")
        if not torch.any(attention_mask):
            raise ValueError("each batch must contain at least one plan token")
        key_padding = ~attention_mask.bool()
        memory = self.encoder(self.token_embedding(token_ids), src_key_padding_mask=key_padding)
        queries = self.prefix_queries.unsqueeze(0).expand(token_ids.shape[0], -1, -1)
        prefix, _ = self.cross_attention(queries, memory, memory, key_padding_mask=key_padding, need_weights=False)
        return self.output_norm(prefix) * torch.sigmoid(self.gate)
