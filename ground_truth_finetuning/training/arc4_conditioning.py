"""Trainable gated bridge from fixed ARC-4 references into PersonaPlex."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch
from torch import Tensor, nn

from .contracts import StreamLayout
from .arc4_two_path import Arc4TwoPathBundle
from .native_training import (
    LossBreakdown,
    agent_only_loss,
    agent_only_loss_per_example,
    forward_with_semantic_prefix_and_evidence,
    forward_with_control_stream,
)


LEGACY_ARC4_ARCHITECTURE = "arc4-residual-rms-bounded-v2"
FIELD_PERSISTENT_ARC4_ARCHITECTURE = "arc4-field-persistent-rms-v3"
LAYERWISE_PERSISTENT_ARC4_ARCHITECTURE = "arc4-field-layerwise-persistent-v4"
LAYERWISE_ADAPTED_ARC4_ARCHITECTURE = "arc4-field-layerwise-adapted-v5"
DEFAULT_ARC4_FIELD_FRAMES = (48, 24, 12, 12)
DEFAULT_ARC4_FIELD_WEIGHTS = (8.0, 4.0, 2.0, 2.0)


@dataclass(frozen=True)
class Arc4InjectionConfig:
    hidden_size: int = 4096
    rank: int = 256
    initial_gate: float = 0.10
    max_stream_rms: float = 0.25
    architecture_revision: str = LEGACY_ARC4_ARCHITECTURE
    output_frames: int = 96
    field_frames: tuple[int, int, int, int] = DEFAULT_ARC4_FIELD_FRAMES
    field_weights: tuple[float, float, float, float] = DEFAULT_ARC4_FIELD_WEIGHTS
    layer_indices: tuple[int, ...] = ()
    layer_initial_gate: float = 0.05
    max_layer_rms: float = 0.15
    layer_adaptation_rank: int = 128
    layer_adaptation_initial_gate: float = 0.10
    max_layer_adaptation_rms: float = 0.10

    def __post_init__(self) -> None:
        if self.hidden_size < 1 or self.rank < 1:
            raise ValueError("ARC-4 adapter dimensions must be positive")
        if not 0.0 < self.initial_gate < 1.0:
            raise ValueError("ARC-4 initial gate must be between zero and one")
        if not 0.0 < self.max_stream_rms <= 1.0:
            raise ValueError("ARC-4 max_stream_rms must be in (0, 1]")
        if self.architecture_revision not in {
            LEGACY_ARC4_ARCHITECTURE,
            FIELD_PERSISTENT_ARC4_ARCHITECTURE,
            LAYERWISE_PERSISTENT_ARC4_ARCHITECTURE,
            LAYERWISE_ADAPTED_ARC4_ARCHITECTURE,
        }:
            raise ValueError("unsupported ARC-4 adapter architecture revision")
        if self.output_frames < 1:
            raise ValueError("ARC-4 output_frames must be positive")
        if len(self.field_frames) != 4 or any(value < 1 for value in self.field_frames):
            raise ValueError("ARC-4 field_frames must contain four positive slots")
        if len(self.field_weights) != 4 or any(value <= 0 for value in self.field_weights):
            raise ValueError("ARC-4 field_weights must contain four positive values")
        if not 0.0 < self.layer_initial_gate < 1.0:
            raise ValueError("ARC-4 layer_initial_gate must be between zero and one")
        if not 0.0 < self.max_layer_rms <= 1.0:
            raise ValueError("ARC-4 max_layer_rms must be in (0, 1]")
        if self.architecture_revision in {
            LAYERWISE_PERSISTENT_ARC4_ARCHITECTURE,
            LAYERWISE_ADAPTED_ARC4_ARCHITECTURE,
        }:
            if not self.layer_indices:
                raise ValueError("layerwise ARC-4 requires selected layer indices")
            if tuple(sorted(set(self.layer_indices))) != self.layer_indices:
                raise ValueError("ARC-4 layer indices must be unique and increasing")
            if self.layer_indices[0] < 0:
                raise ValueError("ARC-4 layer indices must be non-negative")
        if self.layer_adaptation_rank < 1:
            raise ValueError("ARC-4 layer_adaptation_rank must be positive")
        if not 0.0 < self.layer_adaptation_initial_gate < 1.0:
            raise ValueError("ARC-4 layer adaptation gate must be between zero and one")
        if not 0.0 < self.max_layer_adaptation_rms <= 1.0:
            raise ValueError("ARC-4 max_layer_adaptation_rms must be in (0, 1]")

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        if self.architecture_revision == LEGACY_ARC4_ARCHITECTURE:
            value.pop("output_frames")
            value.pop("field_frames")
            value.pop("field_weights")
            value.pop("layer_indices")
            value.pop("layer_initial_gate")
            value.pop("max_layer_rms")
            value.pop("layer_adaptation_rank")
            value.pop("layer_adaptation_initial_gate")
            value.pop("max_layer_adaptation_rms")
        elif self.architecture_revision == FIELD_PERSISTENT_ARC4_ARCHITECTURE:
            value.pop("layer_indices")
            value.pop("layer_initial_gate")
            value.pop("max_layer_rms")
            value.pop("layer_adaptation_rank")
            value.pop("layer_adaptation_initial_gate")
            value.pop("max_layer_adaptation_rms")
        elif self.architecture_revision == LAYERWISE_PERSISTENT_ARC4_ARCHITECTURE:
            value.pop("layer_adaptation_rank")
            value.pop("layer_adaptation_initial_gate")
            value.pop("max_layer_adaptation_rms")
        return value


@dataclass(frozen=True)
class Arc4ConditioningBundle:
    input_stream: Tensor
    layer_streams: Tensor
    layer_indices: tuple[int, ...]
    layer_adapter_down: Tensor | None = None
    layer_adapter_up: Tensor | None = None
    layer_adapter_gates: Tensor | None = None
    max_layer_adaptation_rms: float | None = None


class GatedArc4InjectionAdapter(nn.Module):
    """Versioned RMS-bounded map from Moshika ARC space to PersonaPlex space."""

    def __init__(self, config: Arc4InjectionConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = nn.LayerNorm(config.hidden_size)
        self.down = nn.Linear(config.hidden_size, config.rank, bias=False)
        self.up = nn.Linear(config.rank, config.hidden_size, bias=False)
        if config.architecture_revision in {
            FIELD_PERSISTENT_ARC4_ARCHITECTURE,
            LAYERWISE_PERSISTENT_ARC4_ARCHITECTURE,
            LAYERWISE_ADAPTED_ARC4_ARCHITECTURE,
        }:
            self.mix = nn.Linear(config.rank * len(config.field_frames), config.rank, bias=False)
            initial_weights = torch.tensor(config.field_weights, dtype=torch.float32)
            self.field_logits = nn.Parameter(initial_weights.log())
            nn.init.xavier_uniform_(self.mix.weight)
        if config.architecture_revision in {
            LAYERWISE_PERSISTENT_ARC4_ARCHITECTURE,
            LAYERWISE_ADAPTED_ARC4_ARCHITECTURE,
        }:
            self.layer_up = nn.ModuleList(
                nn.Linear(config.rank, config.hidden_size, bias=False)
                for _ in config.layer_indices
            )
            layer_gate = torch.logit(
                torch.tensor(config.layer_initial_gate, dtype=torch.float32)
            )
            self.layer_gate_logits = nn.Parameter(
                layer_gate.repeat(len(config.layer_indices))
            )
            for projection in self.layer_up:
                nn.init.zeros_(projection.weight)
        if config.architecture_revision == LAYERWISE_ADAPTED_ARC4_ARCHITECTURE:
            self.hidden_adapter_down = nn.ModuleList(
                nn.Linear(config.hidden_size, config.layer_adaptation_rank, bias=False)
                for _ in config.layer_indices
            )
            self.hidden_adapter_up = nn.ModuleList(
                nn.Linear(config.layer_adaptation_rank, config.hidden_size, bias=False)
                for _ in config.layer_indices
            )
            adaptation_gate = torch.logit(
                torch.tensor(config.layer_adaptation_initial_gate, dtype=torch.float32)
            )
            self.hidden_adapter_gate_logits = nn.Parameter(
                adaptation_gate.repeat(len(config.layer_indices))
            )
            for down_projection, up_projection in zip(
                self.hidden_adapter_down, self.hidden_adapter_up
            ):
                nn.init.xavier_uniform_(down_projection.weight)
                nn.init.zeros_(up_projection.weight)
        gate_logit = torch.logit(torch.tensor(config.initial_gate, dtype=torch.float32))
        self.gate_logit = nn.Parameter(gate_logit)
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    @property
    def gate(self) -> Tensor:
        return torch.sigmoid(self.gate_logit)

    def _bound_to(self, conditioned: Tensor, maximum_rms: float) -> Tensor:
        rms = conditioned.square().mean(dim=(1, 2), keepdim=True).clamp_min(
            torch.finfo(conditioned.dtype).eps
        ).sqrt()
        scale = torch.minimum(
            torch.ones_like(rms),
            torch.as_tensor(
                maximum_rms,
                device=conditioned.device,
                dtype=conditioned.dtype,
            ) / rms,
        )
        return conditioned * scale

    def _bound(self, conditioned: Tensor) -> Tensor:
        return self._bound_to(conditioned, self.config.max_stream_rms)

    def _legacy_forward(self, working: Tensor) -> Tensor:
        residual = self.up(torch.nn.functional.silu(self.down(self.norm(working))))
        return self._bound(self.gate.to(working.dtype) * (working + residual))

    def _field_summary(self, working: Tensor) -> tuple[Tensor, Tensor]:
        expected_frames = sum(self.config.field_frames)
        if working.shape[1] != expected_frames:
            raise ValueError(
                f"field-persistent ARC-4 requires {expected_frames} input frames, "
                f"received {working.shape[1]}"
            )
        normalized = self.norm(working)
        compressed = torch.nn.functional.silu(self.down(normalized))
        field_means: list[Tensor] = []
        compressed_means: list[Tensor] = []
        start = 0
        for frames in self.config.field_frames:
            end = start + frames
            field_means.append(normalized[:, start:end].mean(dim=1))
            compressed_means.append(compressed[:, start:end].mean(dim=1))
            start = end
        weights = torch.softmax(self.field_logits, dim=0).to(working.dtype)
        direct = (
            torch.stack(field_means, dim=1) * weights.view(1, -1, 1)
        ).sum(dim=1)
        latent = torch.nn.functional.silu(
            self.mix(torch.cat(compressed_means, dim=-1))
        )
        return direct, latent

    def _field_persistent_forward(self, working: Tensor) -> Tensor:
        direct, latent = self._field_summary(working)
        residual = self.up(latent)
        persistent = self.gate.to(working.dtype) * (direct + residual)
        persistent = self._bound(persistent.unsqueeze(1))
        return persistent.expand(-1, self.config.output_frames, -1)

    def _layerwise_persistent_forward(self, working: Tensor) -> Arc4ConditioningBundle:
        direct, latent = self._field_summary(working)
        input_residual = self.up(latent)
        input_persistent = self.gate.to(working.dtype) * (direct + input_residual)
        input_persistent = self._bound(input_persistent.unsqueeze(1))
        input_stream = input_persistent.expand(-1, self.config.output_frames, -1)

        layer_residuals = torch.stack(
            [projection(latent) for projection in self.layer_up], dim=1
        )
        layer_gates = torch.sigmoid(self.layer_gate_logits).to(working.dtype).view(1, -1, 1)
        layer_persistent = layer_gates * (direct.unsqueeze(1) + layer_residuals)
        batch, layers, hidden = layer_persistent.shape
        bounded_layers = self._bound_to(
            layer_persistent.reshape(batch * layers, 1, hidden),
            self.config.max_layer_rms,
        ).reshape(batch, layers, 1, hidden)
        layer_streams = bounded_layers.expand(
            -1, -1, self.config.output_frames, -1
        )
        bundle = Arc4ConditioningBundle(
            input_stream=input_stream,
            layer_streams=layer_streams,
            layer_indices=self.config.layer_indices,
        )
        if self.config.architecture_revision != LAYERWISE_ADAPTED_ARC4_ARCHITECTURE:
            return bundle
        return Arc4ConditioningBundle(
            input_stream=bundle.input_stream,
            layer_streams=bundle.layer_streams,
            layer_indices=bundle.layer_indices,
            layer_adapter_down=torch.stack(
                [projection.weight for projection in self.hidden_adapter_down], dim=0
            ),
            layer_adapter_up=torch.stack(
                [projection.weight for projection in self.hidden_adapter_up], dim=0
            ),
            layer_adapter_gates=self.hidden_adapter_gate_logits,
            max_layer_adaptation_rms=self.config.max_layer_adaptation_rms,
        )

    def forward(
        self, reference: Tensor, *, drop_condition: bool = False
    ) -> Tensor | Arc4ConditioningBundle:
        if reference.ndim != 3 or reference.shape[-1] != self.config.hidden_size:
            raise ValueError("ARC-4 reference must be [batch, frames, hidden]")
        output_dtype = reference.dtype
        working = reference.to(dtype=self.down.weight.dtype)
        if self.config.architecture_revision in {
            LAYERWISE_PERSISTENT_ARC4_ARCHITECTURE,
            LAYERWISE_ADAPTED_ARC4_ARCHITECTURE,
        }:
            bundle = self._layerwise_persistent_forward(working)
            if drop_condition:
                return Arc4ConditioningBundle(
                    input_stream=bundle.input_stream * 0.0,
                    layer_streams=bundle.layer_streams * 0.0,
                    layer_indices=bundle.layer_indices,
                    layer_adapter_down=bundle.layer_adapter_down,
                    layer_adapter_up=bundle.layer_adapter_up,
                    layer_adapter_gates=bundle.layer_adapter_gates,
                    max_layer_adaptation_rms=bundle.max_layer_adaptation_rms,
                )
            return Arc4ConditioningBundle(
                input_stream=bundle.input_stream.to(dtype=output_dtype),
                layer_streams=bundle.layer_streams.to(dtype=output_dtype),
                layer_indices=bundle.layer_indices,
                layer_adapter_down=bundle.layer_adapter_down,
                layer_adapter_up=bundle.layer_adapter_up,
                layer_adapter_gates=bundle.layer_adapter_gates,
                max_layer_adaptation_rms=bundle.max_layer_adaptation_rms,
            )
        if self.config.architecture_revision == FIELD_PERSISTENT_ARC4_ARCHITECTURE:
            conditioned = self._field_persistent_forward(working)
        else:
            conditioned = self._legacy_forward(working)
        conditioned = conditioned.to(dtype=output_dtype)
        return conditioned * 0.0 if drop_condition else conditioned


def pad_arc4_references(references: Sequence[Tensor]) -> Tensor:
    """Pad variable-length certified ARC streams without changing valid rows."""

    if not references:
        raise ValueError("at least one ARC-4 reference is required")
    hidden = references[0].shape[-1]
    if any(
        value.ndim != 3
        or value.shape[0] != 1
        or value.shape[1] < 1
        or value.shape[-1] != hidden
        for value in references
    ):
        raise ValueError("ARC-4 references must be [1, frames, hidden] with a shared width")
    frames = max(value.shape[1] for value in references)
    output = references[0].new_zeros((len(references), frames, hidden))
    for index, value in enumerate(references):
        output[index, : value.shape[1]] = value[0]
    return output


@dataclass
class Arc4CausalLoss:
    total: Tensor
    matched_sft: Tensor
    counterfactual_margin: Tensor
    focused_counterfactual_margin: Tensor
    contrastive_control: Tensor
    coverage_surrogate: Tensor
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


def reduce_pair_direction_losses(
    values: Tensor,
    *,
    worst_direction_weight: float,
    temperature: float,
) -> Tensor:
    """Blend the pair mean with a smooth maximum so neither direction can hide."""
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("pair direction losses must be a non-empty vector")
    if not 0.0 <= worst_direction_weight <= 1.0:
        raise ValueError("worst_direction_weight must be in [0,1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    mean = values.mean()
    normalizer = values.new_tensor(float(values.numel())).log()
    smooth_worst = temperature * (
        torch.logsumexp(values / temperature, dim=0) - normalizer
    )
    return torch.lerp(mean, smooth_worst, worst_direction_weight)


def margin_adjusted_control_contrastive_loss(
    own: Tensor,
    negatives: Sequence[Tensor],
    margins: Sequence[float],
    *,
    temperature: float,
) -> Tensor:
    """Classify the matched control against wrong, null, and stale variants."""
    if own.ndim != 0 or not negatives or len(negatives) != len(margins):
        raise ValueError("contrastive control losses require aligned scalar variants")
    if temperature <= 0.0 or any(margin < 0.0 for margin in margins):
        raise ValueError("contrastive temperature and margins are invalid")
    adjusted = [own]
    adjusted.extend(
        negative - own.new_tensor(margin)
        for negative, margin in zip(negatives, margins, strict=True)
    )
    logits = -torch.stack(adjusted) / temperature
    return torch.nn.functional.cross_entropy(
        logits.unsqueeze(0),
        torch.zeros(1, dtype=torch.long, device=own.device),
    )


def joint_margin_coverage_surrogate(
    whole_deltas: Tensor,
    focused_deltas: Tensor,
    *,
    whole_margin: float,
    focused_margin: float,
    temperature: float,
    worst_direction_weight: float,
    worst_direction_temperature: float,
) -> Tensor:
    """Bounded differentiable proxy for the exact two-margin pass predicate."""
    if whole_deltas.shape != focused_deltas.shape or whole_deltas.ndim != 1:
        raise ValueError("coverage deltas must be aligned vectors")
    if whole_margin <= 0.0 or focused_margin <= 0.0 or temperature <= 0.0:
        raise ValueError("coverage margins and temperature must be positive")
    normalized = torch.minimum(
        whole_deltas / whole_margin,
        focused_deltas / focused_margin,
    )
    failure_probability = torch.sigmoid((1.0 - normalized) / temperature)
    return reduce_pair_direction_losses(
        failure_probability,
        worst_direction_weight=worst_direction_weight,
        temperature=worst_direction_temperature,
    )


def clip_optimizer_gradients(
    optimizer: torch.optim.Optimizer,
    max_norm: float = 1.0,
) -> Tensor:
    """Clip every optimized parameter, including model-side control adapters."""
    if max_norm <= 0.0:
        raise ValueError("gradient max norm must be positive")
    parameters: list[nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            identity = id(parameter)
            if identity not in seen and parameter.grad is not None:
                seen.add(identity)
                parameters.append(parameter)
    if not parameters:
        return torch.tensor(0.0)
    return torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm,
        error_if_nonfinite=True,
    )


class Arc4CausalTrainer:
    """Train ARC-4 as the sole causal control stream on exact duplex pairs."""

    def __init__(
        self,
        lm_model: object,
        adapter: nn.Module,
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
        causal_weight: float = 2.0,
        whole_causal_weight: float = 1.0,
        focused_causal_weight: float = 1.0,
        null_weight: float = 0.25,
        stale_weight: float = 0.25,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 0.10,
        contrastive_whole_weight: float = 1.0,
        contrastive_focused_weight: float = 2.0,
        coverage_surrogate_weight: float = 0.0,
        coverage_surrogate_temperature: float = 0.25,
        stream_regularization_weight: float = 1e-5,
        pair_worst_direction_weight: float = 0.0,
        pair_worst_temperature: float = 0.05,
        trainable_model_parameters: Sequence[nn.Parameter] = (),
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
        self.whole_causal_weight = whole_causal_weight
        self.focused_causal_weight = focused_causal_weight
        self.null_weight = null_weight
        self.stale_weight = stale_weight
        if contrastive_weight < 0.0:
            raise ValueError("contrastive weight cannot be negative")
        if contrastive_temperature <= 0.0:
            raise ValueError("contrastive temperature must be positive")
        if contrastive_whole_weight < 0.0 or contrastive_focused_weight <= 0.0:
            raise ValueError("contrastive branch weights are invalid")
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_whole_weight = contrastive_whole_weight
        self.contrastive_focused_weight = contrastive_focused_weight
        if coverage_surrogate_weight < 0.0 or coverage_surrogate_temperature <= 0.0:
            raise ValueError("coverage surrogate settings are invalid")
        self.coverage_surrogate_weight = coverage_surrogate_weight
        self.coverage_surrogate_temperature = coverage_surrogate_temperature
        self.stream_regularization_weight = stream_regularization_weight
        if not 0.0 <= pair_worst_direction_weight <= 1.0:
            raise ValueError("pair_worst_direction_weight must be in [0,1]")
        if pair_worst_temperature <= 0.0:
            raise ValueError("pair_worst_temperature must be positive")
        self.pair_worst_direction_weight = pair_worst_direction_weight
        self.pair_worst_temperature = pair_worst_temperature
        stream_layout.validate_for_model(lm_model)
        self.trainable_model_parameters = tuple(trainable_model_parameters)
        for parameter in lm_model.parameters():
            parameter.requires_grad_(False)
        for parameter in self.trainable_model_parameters:
            parameter.requires_grad_(True)
        lm_model.eval()

    def _variant_losses(
        self,
        example: Mapping[str, Tensor],
        references: Sequence[Tensor],
        present: Sequence[bool],
        *,
        activation_checkpointing: bool,
    ):
        if len(references) != len(present):
            raise ValueError("ARC reference/presence variants are not aligned")
        raw = pad_arc4_references(references)
        conditioned = self.adapter(raw)
        prefix_embeddings: Tensor | None = None
        if isinstance(conditioned, Arc4TwoPathBundle):
            prefix_embeddings = conditioned.prefix_embeddings
            streams = conditioned.evidence_stream
            layer_streams = None
            layer_indices = ()
            layer_adapter_down = None
            layer_adapter_up = None
            layer_adapter_gates = None
            max_layer_adaptation_rms = None
        elif isinstance(conditioned, Arc4ConditioningBundle):
            streams = conditioned.input_stream
            layer_streams: Tensor | None = conditioned.layer_streams
            layer_indices = conditioned.layer_indices
            layer_adapter_down = conditioned.layer_adapter_down
            layer_adapter_up = conditioned.layer_adapter_up
            layer_adapter_gates = conditioned.layer_adapter_gates
            max_layer_adaptation_rms = conditioned.max_layer_adaptation_rms
        else:
            streams = conditioned
            layer_streams = None
            layer_indices = ()
            layer_adapter_down = None
            layer_adapter_up = None
            layer_adapter_gates = None
            max_layer_adaptation_rms = None
        presence = torch.tensor(
            present, device=streams.device, dtype=streams.dtype
        ).view(-1, 1, 1)
        streams = streams * presence
        if prefix_embeddings is not None:
            prefix_embeddings = prefix_embeddings * presence
        if layer_streams is not None:
            layer_streams = layer_streams * presence.unsqueeze(1)
        variants = len(references)
        codes = example["codes"].repeat(variants, 1, 1)
        target_mask = example["agent_target_mask"].repeat(variants, 1, 1)
        focused = example.get("contrast_target_mask")
        if not isinstance(focused, Tensor):
            raise ValueError("paired ARC example lacks contrast_target_mask")
        focused = focused.repeat(variants, 1, 1)
        starts = example["prefix_at"].reshape(1).repeat(variants)
        if prefix_embeddings is not None:
            output = forward_with_semantic_prefix_and_evidence(
                self.lm_model,
                codes,
                prefix_embeddings,
                streams,
                starts,
                activation_checkpointing=activation_checkpointing,
            )
        else:
            output = forward_with_control_stream(
                self.lm_model,
                codes,
                streams,
                starts,
                layer_control_streams=layer_streams,
                layer_indices=layer_indices,
                layer_adapter_down=layer_adapter_down,
                layer_adapter_up=layer_adapter_up,
                layer_adapter_gates=layer_adapter_gates,
                max_layer_adaptation_rms=max_layer_adaptation_rms,
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
            focused,
            self.stream_layout,
            audio_weight=0.0,
        )
        if torch.any(focused_losses.text_tokens < 1):
            raise ValueError("exact ARC counterfactual focus has no valid text token")
        return losses, focused_losses, streams, layer_streams

    def _compute(
        self,
        example_a: Mapping[str, Tensor],
        example_b: Mapping[str, Tensor],
        reference_a: Tensor,
        reference_b: Tensor,
        *,
        stale_a: Tensor | None,
        stale_b: Tensor | None,
        activation_checkpointing: bool,
    ) -> Arc4CausalLoss:
        refs_a = [reference_a, reference_b, reference_a]
        refs_b = [reference_b, reference_a, reference_b]
        present_a = [True, True, False]
        present_b = [True, True, False]
        if stale_a is not None:
            refs_a.append(stale_a)
            present_a.append(True)
        if stale_b is not None:
            refs_b.append(stale_b)
            present_b.append(True)
        losses_a, focused_a, streams_a, layer_streams_a = self._variant_losses(
            example_a,
            refs_a,
            present_a,
            activation_checkpointing=activation_checkpointing,
        )
        losses_b, focused_b, streams_b, layer_streams_b = self._variant_losses(
            example_b,
            refs_b,
            present_b,
            activation_checkpointing=activation_checkpointing,
        )
        matched = torch.stack((losses_a.total[0], losses_b.total[0])).mean()
        whole_terms = torch.stack(
            (
                torch.relu(self.counterfactual_margin_value + losses_a.text[0] - losses_a.text[1]),
                torch.relu(self.counterfactual_margin_value + losses_b.text[0] - losses_b.text[1]),
            )
        )
        focused_terms = torch.stack(
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
        stale_loss = torch.stack(stale_terms).mean() if stale_terms else matched.detach() * 0.0
        contrastive_whole_terms = []
        contrastive_focused_terms = []
        for losses, focused, stale_present in (
            (losses_a, focused_a, stale_a is not None),
            (losses_b, focused_b, stale_b is not None),
        ):
            whole_negatives = [losses.text[1], losses.text[2]]
            focused_negatives = [focused.text[1], focused.text[2]]
            margins = [self.counterfactual_margin_value, self.null_margin_value]
            if stale_present:
                whole_negatives.append(losses.text[3])
                focused_negatives.append(focused.text[3])
                margins.append(self.stale_margin_value)
            contrastive_whole_terms.append(
                margin_adjusted_control_contrastive_loss(
                    losses.text[0],
                    whole_negatives,
                    margins,
                    temperature=self.contrastive_temperature,
                )
            )
            contrastive_focused_terms.append(
                margin_adjusted_control_contrastive_loss(
                    focused.text[0],
                    focused_negatives,
                    margins,
                    temperature=self.contrastive_temperature,
                )
            )
        contrastive_control = (
            self.contrastive_whole_weight * torch.stack(contrastive_whole_terms).mean()
            + self.contrastive_focused_weight * torch.stack(contrastive_focused_terms).mean()
        )
        coverage_surrogate = joint_margin_coverage_surrogate(
            torch.stack(
                (losses_a.text[1] - losses_a.text[0], losses_b.text[1] - losses_b.text[0])
            ),
            torch.stack(
                (focused_a.text[1] - focused_a.text[0], focused_b.text[1] - focused_b.text[0])
            ),
            whole_margin=self.counterfactual_margin_value,
            focused_margin=self.focused_counterfactual_margin_value,
            temperature=self.coverage_surrogate_temperature,
            worst_direction_weight=self.pair_worst_direction_weight,
            worst_direction_temperature=self.pair_worst_temperature,
        )
        regularization = torch.stack(
            (streams_a[0].float().square().mean(), streams_b[0].float().square().mean())
        ).mean()
        if layer_streams_a is not None and layer_streams_b is not None:
            regularization = torch.stack(
                (
                    regularization,
                    layer_streams_a[0].float().square().mean(),
                    layer_streams_b[0].float().square().mean(),
                )
            ).mean()
        whole_causal_loss = reduce_pair_direction_losses(
            whole_terms,
            worst_direction_weight=self.pair_worst_direction_weight,
            temperature=self.pair_worst_temperature,
        )
        focused_causal_loss = reduce_pair_direction_losses(
            focused_terms,
            worst_direction_weight=self.pair_worst_direction_weight,
            temperature=self.pair_worst_temperature,
        )
        total = (
            self.matched_weight * matched
            + self.causal_weight
            * (
                self.whole_causal_weight * whole_causal_loss
                + self.focused_causal_weight * focused_causal_loss
            )
            + self.null_weight * null_terms.mean()
            + self.stale_weight * stale_loss
            + self.contrastive_weight * contrastive_control
            + self.coverage_surrogate_weight * coverage_surrogate
            + self.stream_regularization_weight * regularization
        )
        return Arc4CausalLoss(
            total=total,
            matched_sft=matched,
            counterfactual_margin=whole_causal_loss,
            focused_counterfactual_margin=focused_causal_loss,
            contrastive_control=contrastive_control,
            coverage_surrogate=coverage_surrogate,
            null_margin=null_terms.mean(),
            stale_margin=stale_loss,
            stream_regularization=regularization,
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

    def backward_pair(
        self,
        *args,
        loss_scale: float = 1.0,
        **kwargs,
    ) -> Arc4CausalLoss:
        if loss_scale <= 0.0:
            raise ValueError("loss_scale must be positive")
        result = self._compute(
            *args,
            **kwargs,
            activation_checkpointing=self.activation_checkpointing,
        )
        if not bool(torch.isfinite(result.total).item()):
            raise FloatingPointError("ARC-4 causal objective became non-finite")
        (result.total * loss_scale).backward()
        return result

    def apply_gradients(self) -> Tensor:
        gradient_norm = clip_optimizer_gradients(self.optimizer, 1.0)
        self.optimizer.step()
        return gradient_norm

    def step_pair(self, *args, **kwargs) -> Arc4CausalLoss:
        self.optimizer.zero_grad(set_to_none=True)
        result = self.backward_pair(*args, **kwargs)
        self.apply_gradients()
        return result

    @torch.no_grad()
    def evaluate_pair(self, *args, **kwargs) -> Arc4CausalLoss:
        return self._compute(*args, **kwargs, activation_checkpointing=False)


@dataclass(frozen=True)
class Arc4StepMetrics:
    total: float
    matched: float
    text: float
    audio: float
    wrong_reference: float | None
    ranking: float
    ranking_active: bool
    gate: float


class Arc4ReferenceTrainer:
    """Frozen-base trainer with optional matched-vs-wrong ARC ranking updates."""

    def __init__(
        self,
        lm_model: object,
        control_adapter: nn.Module,
        arc4_adapter: nn.Module,
        optimizer: torch.optim.Optimizer,
        stream_layout: StreamLayout,
        *,
        activation_checkpointing: bool = True,
    ) -> None:
        self.lm_model = lm_model
        self.control_adapter = control_adapter
        self.arc4_adapter = arc4_adapter
        self.optimizer = optimizer
        self.stream_layout = stream_layout
        self.activation_checkpointing = activation_checkpointing
        stream_layout.validate_for_model(lm_model)
        for parameter in lm_model.parameters():
            parameter.requires_grad_(False)
        for parameter in control_adapter.parameters():
            parameter.requires_grad_(False)
        lm_model.eval()
        control_adapter.eval()

    def _loss(
        self,
        batch: Mapping[str, Tensor],
        reference: Tensor,
        *,
        activation_checkpointing: bool,
        drop_condition: bool = False,
    ) -> LossBreakdown:
        with torch.no_grad():
            prefix = self.control_adapter(
                batch["control_token_ids"],
                batch["control_attention_mask"],
            )
        stream = self.arc4_adapter(reference, drop_condition=drop_condition)
        output = forward_with_semantic_prefix_and_evidence(
            self.lm_model,
            batch["codes"],
            prefix,
            stream,
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

    def step(
        self,
        batch: Mapping[str, Tensor],
        reference: Tensor,
        *,
        wrong_reference: Tensor | None = None,
        ranking_margin: float = 0.05,
        ranking_weight: float = 0.25,
        drop_condition: bool = False,
    ) -> Arc4StepMetrics:
        self.optimizer.zero_grad(set_to_none=True)
        wrong_value: float | None = None
        ranking_active = False
        if wrong_reference is not None and not drop_condition:
            with torch.no_grad():
                wrong_probe = self._loss(
                    batch,
                    wrong_reference,
                    activation_checkpointing=False,
                )
                wrong_value = float(wrong_probe.total)
        matched = self._loss(
            batch,
            reference,
            activation_checkpointing=self.activation_checkpointing,
            drop_condition=drop_condition,
        )
        matched_value = float(matched.total.detach())
        ranking = 0.0
        if wrong_value is not None:
            ranking = max(0.0, ranking_margin + matched_value - wrong_value)
            ranking_active = ranking > 0.0
        matched_scale = 1.0 + (ranking_weight if ranking_active else 0.0)
        (matched.total * matched_scale).backward()
        if wrong_reference is not None:
            wrong = self._loss(
                batch,
                wrong_reference,
                activation_checkpointing=self.activation_checkpointing,
            )
            wrong_scale = -ranking_weight if ranking_active else 0.0
            (wrong_scale * wrong.total).backward()
        torch.nn.utils.clip_grad_norm_(self.arc4_adapter.parameters(), 1.0)
        self.optimizer.step()
        module = self.arc4_adapter.module if hasattr(self.arc4_adapter, "module") else self.arc4_adapter
        gate = float(module.gate.detach())
        return Arc4StepMetrics(
            total=matched_value + ranking_weight * ranking,
            matched=matched_value,
            text=float(matched.text.detach()),
            audio=float(matched.audio.detach()),
            wrong_reference=wrong_value,
            ranking=ranking,
            ranking_active=ranking_active,
            gate=gate,
        )

    @torch.no_grad()
    def evaluate(
        self,
        batch: Mapping[str, Tensor],
        reference: Tensor,
    ) -> LossBreakdown:
        return self._loss(batch, reference, activation_checkpointing=False)
