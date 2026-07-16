"""One optimization step for a frozen PersonaPlex LM plus semantic-prefix adapter."""

from __future__ import annotations

from typing import Mapping

import torch

from .contracts import StreamLayout
from .native_training import (
    LossBreakdown,
    agent_only_loss,
    forward_with_semantic_prefix,
    forward_with_semantic_prefix_and_evidence,
)


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
        required = {"codes", "agent_target_mask", "prefix_at"}
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"batch missing required fields: {sorted(missing)}")
        token_ids = batch.get("control_token_ids", batch.get("plan_token_ids"))
        attention_mask = batch.get("control_attention_mask", batch.get("plan_attention_mask"))
        if token_ids is None or attention_mask is None:
            raise ValueError("batch requires control_token_ids/control_attention_mask")
        self.optimizer.zero_grad(set_to_none=True)
        prefix = self.adapter(token_ids, attention_mask)
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

    @torch.no_grad()
    def evaluate(self, batch: Mapping[str, torch.Tensor], *, audio_weight: float = 0.02) -> LossBreakdown:
        """Teacher-forced held-out loss without mutating adapter or optimizer state."""
        required = {"codes", "agent_target_mask", "prefix_at"}
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"batch missing required fields: {sorted(missing)}")
        token_ids = batch.get("control_token_ids", batch.get("plan_token_ids"))
        attention_mask = batch.get("control_attention_mask", batch.get("plan_attention_mask"))
        if token_ids is None or attention_mask is None:
            raise ValueError("batch requires control_token_ids/control_attention_mask")
        prefix = self.adapter(token_ids, attention_mask)
        output = forward_with_semantic_prefix(
            self.lm_model,
            batch["codes"],
            prefix,
            batch["prefix_at"],
            activation_checkpointing=False,
        )
        return agent_only_loss(
            self.lm_model,
            output,
            batch["codes"],
            batch["agent_target_mask"],
            self.stream_layout,
            audio_weight=audio_weight,
        )


class EvidenceStreamTrainer:
    """Second-stage trainer for a frozen base/prefix and trainable evidence stream.

    Prefix training must converge first.  This stage never updates base PersonaPlex
    parameters or the already accepted control-prefix adapter, which isolates
    semantic evidence adherence from conversational timing and voice quality.
    """

    def __init__(
        self,
        lm_model: object,
        control_adapter: torch.nn.Module,
        evidence_adapter: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        stream_layout: StreamLayout,
        *,
        activation_checkpointing: bool = True,
    ) -> None:
        self.lm_model = lm_model
        self.control_adapter = control_adapter
        self.evidence_adapter = evidence_adapter
        self.optimizer = optimizer
        stream_layout.validate_for_model(lm_model)
        self.stream_layout = stream_layout
        self.activation_checkpointing = activation_checkpointing
        for parameter in lm_model.parameters():
            parameter.requires_grad_(False)
        for parameter in control_adapter.parameters():
            parameter.requires_grad_(False)
        lm_model.eval()
        control_adapter.eval()

    @staticmethod
    def _tokens(batch: Mapping[str, torch.Tensor], prefix: str) -> tuple[torch.Tensor, torch.Tensor]:
        token_ids = batch.get(f"{prefix}_token_ids")
        attention_mask = batch.get(f"{prefix}_attention_mask")
        if token_ids is None or attention_mask is None:
            raise ValueError(f"batch requires {prefix}_token_ids/{prefix}_attention_mask")
        return token_ids, attention_mask

    def _forward(self, batch: Mapping[str, torch.Tensor], *, activation_checkpointing: bool) -> LossBreakdown:
        required = {"codes", "agent_target_mask", "prefix_at"}
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"batch missing required fields: {sorted(missing)}")
        control_ids, control_mask = self._tokens(batch, "control")
        evidence_ids, evidence_mask = self._tokens(batch, "evidence")
        with torch.no_grad():
            prefix = self.control_adapter(control_ids, control_mask)
        evidence_stream = self.evidence_adapter(evidence_ids, evidence_mask)
        output = forward_with_semantic_prefix_and_evidence(
            self.lm_model,
            batch["codes"],
            prefix,
            evidence_stream,
            batch["prefix_at"],
            activation_checkpointing=activation_checkpointing,
        )
        return agent_only_loss(
            self.lm_model,
            output,
            batch["codes"],
            batch["agent_target_mask"],
            self.stream_layout,
        )

    def step(self, batch: Mapping[str, torch.Tensor], *, audio_weight: float = 0.02) -> LossBreakdown:
        del audio_weight  # kept for the same public call shape as SemanticPrefixTrainer
        self.optimizer.zero_grad(set_to_none=True)
        losses = self._forward(batch, activation_checkpointing=self.activation_checkpointing)
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(self.evidence_adapter.parameters(), 1.0)
        self.optimizer.step()
        return losses

    @torch.no_grad()
    def evaluate(self, batch: Mapping[str, torch.Tensor], *, audio_weight: float = 0.02) -> LossBreakdown:
        del audio_weight
        return self._forward(batch, activation_checkpointing=False)
