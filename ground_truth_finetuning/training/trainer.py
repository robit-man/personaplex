"""One optimization step for a frozen PersonaPlex LM plus semantic-prefix adapter."""

from __future__ import annotations

from typing import Mapping

import torch

from .contracts import StreamLayout
from .native_training import LossBreakdown, agent_only_loss, forward_with_semantic_prefix


class SemanticPrefixTrainer:
    def __init__(
        self,
        lm_model: object,
        adapter: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        stream_layout: StreamLayout,
        *,
        activation_checkpointing: bool = True,
    ) -> None:
        self.lm_model = lm_model
        self.adapter = adapter
        self.optimizer = optimizer
        stream_layout.validate_for_model(lm_model)
        self.stream_layout = stream_layout
        self.activation_checkpointing = activation_checkpointing
        for parameter in lm_model.parameters():
            parameter.requires_grad_(False)
        lm_model.eval()

    def step(self, batch: Mapping[str, torch.Tensor], *, audio_weight: float = 0.02) -> LossBreakdown:
        required = {"plan_token_ids", "plan_attention_mask", "codes", "agent_target_mask", "prefix_at"}
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"batch missing required fields: {sorted(missing)}")
        self.optimizer.zero_grad(set_to_none=True)
        prefix = self.adapter(batch["plan_token_ids"], batch["plan_attention_mask"])
        output = forward_with_semantic_prefix(
            self.lm_model,
            batch["codes"],
            prefix,
            batch["prefix_at"],
            activation_checkpointing=self.activation_checkpointing,
        )
        losses = agent_only_loss(
            self.lm_model,
            output,
            batch["codes"],
            batch["agent_target_mask"],
            self.stream_layout,
            audio_weight=audio_weight,
        )
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
        self.optimizer.step()
        return losses
