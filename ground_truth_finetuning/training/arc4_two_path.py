"""Two-path ARC conditioning: immediate decision prefix plus detailed reference stream."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TWO_PATH_ARC4_ARCHITECTURE = "arc4-ordered-prefix-frozen-stream-v9"


@dataclass(frozen=True)
class Arc4TwoPathConfig:
    hidden_size: int = 4096
    decision_frames: int = 48
    stream_frames: int = 96
    prefix_tokens: int = 8
    rank: int = 64
    attention_heads: int = 8
    initial_prefix_gate: float = 0.05
    max_prefix_rms: float = 0.15
    initial_stream_scale: float = 1.0
    max_stream_residual_rms: float = 0.10
    architecture_revision: str = TWO_PATH_ARC4_ARCHITECTURE

    def __post_init__(self) -> None:
        dimensions = (
            self.hidden_size,
            self.decision_frames,
            self.stream_frames,
            self.prefix_tokens,
            self.rank,
            self.attention_heads,
        )
        if min(dimensions) < 1:
            raise ValueError("two-path ARC dimensions must be positive")
        if not 0.0 < self.initial_prefix_gate < 1.0:
            raise ValueError("initial_prefix_gate must be in (0,1)")
        if not 0.0 < self.max_prefix_rms <= 1.0:
            raise ValueError("max_prefix_rms must be in (0,1]")
        if self.initial_stream_scale != 1.0:
            raise ValueError("v9 keeps the pretrained ARC stream at an exact unit scale")
        if not 0.0 <= self.max_stream_residual_rms <= 1.0:
            raise ValueError("max_stream_residual_rms must be in [0,1]")
        if self.architecture_revision != TWO_PATH_ARC4_ARCHITECTURE:
            raise ValueError("unsupported two-path ARC architecture revision")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class Arc4TwoPathBundle:
    prefix_embeddings: Tensor
    evidence_stream: Tensor


class Arc4TwoPathAdapter(nn.Module):
    """Preserve pretrained ARC evidence and learn a compact immediate prefix.

    V8 trained projections over both paths and memorized corpus identifiers in a
    single pass. V9 leaves the MoshiRAG-compatible temporal stream immutable.
    The decision slot is reduced in ordered contiguous segments, then receives a
    zero-initialized low-rank residual. This starts from a deterministic mapping,
    preserves field order, and sharply limits memorization capacity.
    """

    def __init__(self, config: Arc4TwoPathConfig) -> None:
        super().__init__()
        self.config = config
        self.decision_norm = nn.LayerNorm(
            config.hidden_size,
            elementwise_affine=False,
        )
        self.prefix_down = nn.Linear(config.hidden_size, config.rank, bias=False)
        self.prefix_up = nn.Linear(config.rank, config.hidden_size, bias=False)
        self.prefix_gate_logit = nn.Parameter(
            torch.logit(torch.tensor(config.initial_prefix_gate, dtype=torch.float32))
        )
        nn.init.xavier_uniform_(self.prefix_down.weight)
        nn.init.zeros_(self.prefix_up.weight)

    @property
    def prefix_gate(self) -> Tensor:
        return torch.sigmoid(self.prefix_gate_logit)

    @property
    def gate(self) -> Tensor:
        """Shared trainer metric contract: v8 reports its immediate-prefix gate."""
        return self.prefix_gate

    @staticmethod
    def _bound(value: Tensor, maximum_rms: float) -> Tensor:
        epsilon = torch.finfo(value.dtype).eps
        rms = (value.square().mean(dim=(1, 2), keepdim=True) + epsilon).sqrt()
        scale = torch.minimum(
            torch.ones_like(rms),
            torch.as_tensor(maximum_rms, device=value.device, dtype=value.dtype) / rms,
        )
        return value * scale

    def _prefix(self, decision: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool1d(
            decision.transpose(1, 2),
            self.config.prefix_tokens,
        ).transpose(1, 2)
        normalized = self.decision_norm(pooled)
        residual = self.prefix_up(F.silu(self.prefix_down(normalized)))
        prefix = self.prefix_gate.to(pooled.dtype) * (pooled + residual)
        return self._bound(prefix, self.config.max_prefix_rms)

    def forward(
        self,
        reference: Tensor,
        *,
        drop_condition: bool = False,
    ) -> Arc4TwoPathBundle:
        expected = self.config.decision_frames + self.config.stream_frames
        if (
            reference.ndim != 3
            or reference.shape[1] != expected
            or reference.shape[2] != self.config.hidden_size
        ):
            raise ValueError(
                "two-path ARC reference must be "
                f"[batch,{expected},{self.config.hidden_size}]"
            )
        output_dtype = reference.dtype
        working = reference.to(dtype=self.prefix_down.weight.dtype)
        decision = working[:, : self.config.decision_frames]
        detailed = reference[:, self.config.decision_frames :]
        prefix = self._prefix(decision).to(dtype=output_dtype)
        stream = detailed
        if drop_condition:
            prefix = prefix * 0.0
            stream = stream * 0.0
        return Arc4TwoPathBundle(prefix_embeddings=prefix, evidence_stream=stream)
