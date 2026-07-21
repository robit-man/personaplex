"""Causal pair objectives for the PersonaPlex semantic control stream."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .contracts import StreamLayout
from .control_encoding import EncodedControl, pad_encoded_controls
from .native_training import agent_only_loss_per_example, forward_with_control_stream


@dataclass
class CausalControlLoss:
    total: Tensor
    matched_sft: Tensor
    counterfactual_margin: Tensor
    null_margin: Tensor
    stale_margin: Tensor
    stream_regularization: Tensor
    a_own_text: Tensor
    a_cross_text: Tensor
    b_own_text: Tensor
    b_cross_text: Tensor
    a_own_focused_text: Tensor
    a_cross_focused_text: Tensor
    b_own_focused_text: Tensor
    b_cross_focused_text: Tensor
    a_null_text: Tensor
    b_null_text: Tensor
    a_stale_text: Tensor | None
    b_stale_text: Tensor | None


class CausalControlTrainer:
    """Freeze PersonaPlex and train control with same-context causal negatives."""

    def __init__(
        self,
        lm_model: object,
        adapter: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        stream_layout: StreamLayout,
        *,
        activation_checkpointing: bool = True,
        audio_weight: float = 0.02,
        counterfactual_margin: float = 0.08,
        focused_counterfactual_margin: float = 0.30,
        null_margin: float = 0.03,
        stale_margin: float = 0.03,
        matched_weight: float = 1.0,
        causal_weight: float = 1.0,
        null_weight: float = 0.25,
        stale_weight: float = 0.25,
        stream_regularization_weight: float = 1e-5,
    ) -> None:
        self.lm_model = lm_model
        self.adapter = adapter
        self.optimizer = optimizer
        self.stream_layout = stream_layout
        self.activation_checkpointing = activation_checkpointing
        self.audio_weight = audio_weight
        self.counterfactual_margin_value = counterfactual_margin
        self.focused_counterfactual_margin_value = focused_counterfactual_margin
        self.null_margin_value = null_margin
        self.stale_margin_value = stale_margin
        self.matched_weight = matched_weight
        self.causal_weight = causal_weight
        self.null_weight = null_weight
        self.stale_weight = stale_weight
        self.stream_regularization_weight = stream_regularization_weight
        stream_layout.validate_for_model(lm_model)
        for parameter in lm_model.parameters():
            parameter.requires_grad_(False)
        lm_model.eval()

    def _variant_losses(
        self,
        example: Mapping[str, Tensor],
        controls: Sequence[EncodedControl],
        present: Sequence[bool],
        *,
        activation_checkpointing: bool,
    ):
        device = example["codes"].device
        typed = pad_encoded_controls(controls, device=device, present=present)
        with torch.no_grad():
            lexical = self.lm_model.text_emb(typed.token_ids)
        if lexical.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                streams = self.adapter(lexical, typed)
        else:
            streams = self.adapter(lexical, typed)
        variants = len(controls)
        codes = example["codes"].repeat(variants, 1, 1)
        target_mask = example["agent_target_mask"].repeat(variants, 1, 1)
        focused_target_mask = example.get("contrast_target_mask")
        if not isinstance(focused_target_mask, Tensor):
            raise ValueError("paired example lacks an exact contrast_target_mask")
        focused_target_mask = focused_target_mask.repeat(variants, 1, 1)
        prefix_at = example["prefix_at"].reshape(1).repeat(variants)
        output = forward_with_control_stream(
            self.lm_model,
            codes,
            streams,
            prefix_at,
            activation_checkpointing=activation_checkpointing,
        )
        losses = agent_only_loss_per_example(
            self.lm_model,
            output,
            codes,
            target_mask,
            self.stream_layout,
            audio_weight=self.audio_weight,
        )
        focused_losses = agent_only_loss_per_example(
            self.lm_model,
            output,
            codes,
            focused_target_mask,
            self.stream_layout,
            audio_weight=0.0,
        )
        if torch.any(focused_losses.text_tokens < 1):
            raise ValueError("exact counterfactual focus contains no valid text token")
        return losses, focused_losses, streams

    def _compute(
        self,
        example_a: Mapping[str, Tensor],
        example_b: Mapping[str, Tensor],
        control_a: EncodedControl,
        control_b: EncodedControl,
        *,
        stale_a: EncodedControl | None,
        stale_b: EncodedControl | None,
        activation_checkpointing: bool,
    ) -> CausalControlLoss:
        controls_a = [control_a, control_b, control_a]
        controls_b = [control_b, control_a, control_b]
        present_a = [True, True, False]
        present_b = [True, True, False]
        if stale_a is not None:
            controls_a.append(stale_a)
            present_a.append(True)
        if stale_b is not None:
            controls_b.append(stale_b)
            present_b.append(True)
        losses_a, focused_a, streams_a = self._variant_losses(
            example_a, controls_a, present_a, activation_checkpointing=activation_checkpointing
        )
        losses_b, focused_b, streams_b = self._variant_losses(
            example_b, controls_b, present_b, activation_checkpointing=activation_checkpointing
        )
        matched = torch.stack((losses_a.total[0], losses_b.total[0])).mean()
        causal_terms = torch.stack(
            (
                torch.relu(
                    self.focused_counterfactual_margin_value
                    + focused_a.text[0]
                    - focused_a.text[1]
                ),
                torch.relu(
                    self.focused_counterfactual_margin_value
                    + focused_b.text[0]
                    - focused_b.text[1]
                ),
            )
        )
        null_terms = torch.stack(
            (
                torch.relu(self.null_margin_value + losses_a.text[0] - losses_a.text[2]),
                torch.relu(self.null_margin_value + losses_b.text[0] - losses_b.text[2]),
            )
        )
        stale_terms: list[Tensor] = []
        a_stale = losses_a.text[3] if stale_a is not None else None
        b_stale = losses_b.text[3] if stale_b is not None else None
        if a_stale is not None:
            stale_terms.append(torch.relu(self.stale_margin_value + losses_a.text[0] - a_stale))
        if b_stale is not None:
            stale_terms.append(torch.relu(self.stale_margin_value + losses_b.text[0] - b_stale))
        stale_loss = (
            torch.stack(stale_terms).mean()
            if stale_terms
            else matched.detach() * 0.0
        )
        stream_regularization = torch.stack(
            (streams_a[0].float().square().mean(), streams_b[0].float().square().mean())
        ).mean()
        total = (
            self.matched_weight * matched
            + self.causal_weight * causal_terms.mean()
            + self.null_weight * null_terms.mean()
            + self.stale_weight * stale_loss
            + self.stream_regularization_weight * stream_regularization
        )
        return CausalControlLoss(
            total=total,
            matched_sft=matched,
            counterfactual_margin=causal_terms.mean(),
            null_margin=null_terms.mean(),
            stale_margin=stale_loss,
            stream_regularization=stream_regularization,
            a_own_text=losses_a.text[0],
            a_cross_text=losses_a.text[1],
            b_own_text=losses_b.text[0],
            b_cross_text=losses_b.text[1],
            a_own_focused_text=focused_a.text[0],
            a_cross_focused_text=focused_a.text[1],
            b_own_focused_text=focused_b.text[0],
            b_cross_focused_text=focused_b.text[1],
            a_null_text=losses_a.text[2],
            b_null_text=losses_b.text[2],
            a_stale_text=a_stale,
            b_stale_text=b_stale,
        )

    def step_pair(
        self,
        example_a: Mapping[str, Tensor],
        example_b: Mapping[str, Tensor],
        control_a: EncodedControl,
        control_b: EncodedControl,
        *,
        stale_a: EncodedControl | None = None,
        stale_b: EncodedControl | None = None,
    ) -> CausalControlLoss:
        self.optimizer.zero_grad(set_to_none=True)
        result = self._compute(
            example_a,
            example_b,
            control_a,
            control_b,
            stale_a=stale_a,
            stale_b=stale_b,
            activation_checkpointing=self.activation_checkpointing,
        )
        if not bool(torch.isfinite(result.total).item()):
            raise FloatingPointError("semantic-control objective became non-finite")
        result.total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.adapter.parameters(), 1.0, error_if_nonfinite=True
        )
        self.optimizer.step()
        return result

    @torch.no_grad()
    def evaluate_pair(
        self,
        example_a: Mapping[str, Tensor],
        example_b: Mapping[str, Tensor],
        control_a: EncodedControl,
        control_b: EncodedControl,
        *,
        stale_a: EncodedControl | None = None,
        stale_b: EncodedControl | None = None,
    ) -> CausalControlLoss:
        return self._compute(
            example_a,
            example_b,
            control_a,
            control_b,
            stale_a=stale_a,
            stale_b=stale_b,
            activation_checkpointing=False,
        )
