"""Revision-safe client/provider for upstream MoshiRAG ARC-4 references."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import load

from ground_truth_finetuning.training.contracts import (
    ControlTrainingFrame,
    EvidenceTrainingFrame,
    assert_evidence_control_alignment,
)
from ground_truth_finetuning.training.evidence_conditioning import (
    MoshiStreamingSumBridge,
    StreamingConditioningError,
)
from ground_truth_finetuning.training.arc4_conditioning import (
    Arc4InjectionConfig,
    FIELD_PERSISTENT_ARC4_ARCHITECTURE,
    GatedArc4InjectionAdapter,
)
from ground_truth_finetuning.training.plan_serializer import PlanSerializer
from personaplex_control.arc4_packing import (
    ARC4_FIELD_ORDER,
    ARC4_PACKING_REVISION,
    ARC4_SUPPORTED_PACKING_REVISIONS,
)


ARC4_REFERENCE_REVISION = "personaplex-semantic-reference-v6-budget-first-no-lineage"


def render_arc4_reference_envelope(fields: Mapping[str, str]) -> str:
    if tuple(fields) != ARC4_FIELD_ORDER:
        raise ValueError("ARC-4 fields do not match the versioned field order")
    return json.dumps(
        {"v": ARC4_REFERENCE_REVISION, "fields": dict(fields)},
        ensure_ascii=True,
        separators=(",", ":"),
    )


class Arc4ConditionerError(RuntimeError):
    """The external ARC-4 service failed closed or violated its contract."""


@dataclass(frozen=True)
class Arc4ReferenceCacheEntry:
    evidence_hash: str
    reference_hash: str
    conditioner_revision: str
    stream: torch.Tensor
    build_ms: float


def render_arc4_reference_fields(
    control: ControlTrainingFrame,
    evidence: EvidenceTrainingFrame | None,
) -> dict[str, str]:
    """Build a causal-first, target-free ARC-4 input.

    ARC-4 is frame bounded.  The branch-defining semantic update must therefore
    precede shared plan/style/history fields.  Counterfactual branch identifiers
    are intentionally omitted: they are corpus lineage, not behavior the live
    model may use as a shortcut.
    """

    state = dict(control.state)
    context = state.get("textContext")
    recent_turns = []
    if isinstance(context, dict) and isinstance(context.get("turns"), list):
        recent_turns = [
            {
                "speaker": item.get("speaker"),
                "text": item.get("text"),
                "turn": item.get("turn"),
            }
            for item in context["turns"][-2:]
            if isinstance(item, dict)
        ]
    plan = control.plan
    bindings = state.get("semanticBindings")
    if not isinstance(bindings, dict):
        bindings = {}
    # Keep short, typed, live-available discriminators before free-form text.
    # The ARC-4 decision slot is latency bounded; placing a long natural-language
    # update first can hide the material state difference beyond that budget.
    intervention_family = bindings.get("interventionFamily")
    control_value = bindings.get("controlValue")
    causal: dict[str, Any] = {
        "evidenceStatus": evidence.availability if evidence is not None else None,
        "activeValue": (
            control_value
            if intervention_family in (None, "semantic")
            else None
        ),
        "update": bindings.get("concreteUpdate"),
        "axis": bindings.get("counterfactualAxis"),
        "kind": bindings.get("controlKind"),
        "revision": control.state_revision,
    }
    if evidence is not None:
        assert_evidence_control_alignment(control, evidence)
        # Operational record identifiers are corpus lineage, not semantic
        # evidence. Passing them to ARC lets a small adapter memorize examples
        # instead of learning from the natural-language claims.
        evidence_source = {
            key: evidence.provenance[key]
            for key in ("source", "confidence")
            if key in evidence.provenance
        }
        causal["evidence"] = {
            "status": evidence.availability,
            "claims": list(evidence.allowed_claims),
            "source": evidence_source,
        }
    causal.update(
        {
            "goal": state.get("nextGoal"),
            "guidance": state.get("activeControlGuidance"),
            "intent": plan.intent,
            "act": plan.dialogue_act,
            "required": list(plan.constraints.required_facts),
            "forbidden": list(plan.constraints.forbidden_claims),
            "ask": list(plan.constraints.must_ask),
            "doNotRequest": list(plan.constraints.must_not_request),
        }
    )
    reference: dict[str, Any] = {
        "v": "personaplex-semantic-reference-v4-budget-first",
        "causal": causal,
        "revision": control.state_revision,
        "delivery": {
            "activeValue": (
                control_value if intervention_family == "delivery" else None
            ),
            "register": plan.delivery.register,
            "assertiveness": plan.delivery.assertiveness,
            "rate": plan.delivery.speaking_rate_bucket,
            "pauses": plan.delivery.pause_density_bucket,
            "interruptibility": plan.delivery.interruptibility,
            "maxMs": plan.delivery.max_duration_ms,
        },
        "state": {
            "intent": state.get("intent"),
            "callerPosture": state.get("callerPosture"),
            "compliancePosture": state.get("compliancePosture"),
            "resistancePosture": state.get("resistancePosture"),
            # The legacy facts list mixes audible facts with reducer/event IDs.
            # Concrete updates and allowed claims above are the typed semantic
            # facts until the state contract exposes a lineage-free facts field.
            "commitments": list(state.get("commitments", []))[-4:],
            "uncertainty": list(state.get("uncertainty", []))[-4:],
            "policy": list(state.get("policyConstraints", []))[-6:],
            "tools": list(state.get("toolResults", []))[-4:],
            "endCallAuthorized": bool(state.get("endCallAuthorized", False)),
        },
        "turnTaking": dict(control.turn_taking),
        "recentAudibleContext": recent_turns,
    }
    values = {
        "decision": reference["causal"],
        "state": reference["state"],
        "delivery": reference["delivery"],
        "context": {
            "activeValue": (
                control_value if intervention_family == "turn_taking" else None
            ),
            "turnTaking": reference["turnTaking"],
            "recentAudibleContext": reference["recentAudibleContext"],
        },
    }
    return {
        name: json.dumps(values[name], ensure_ascii=True, separators=(",", ":"))
        for name in ARC4_FIELD_ORDER
    }


def render_arc4_reference(
    control: ControlTrainingFrame,
    evidence: EvidenceTrainingFrame | None,
) -> str:
    fields = render_arc4_reference_fields(control, evidence)
    return render_arc4_reference_envelope(fields)


class Arc4ConditionerClient:
    """Small bounded client for the separately pinned CUDA conditioner."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 2.0,
        expected_packing_revision: str = ARC4_PACKING_REVISION,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("conditioner timeout must be positive")
        if expected_packing_revision not in ARC4_SUPPORTED_PACKING_REVISIONS:
            raise ValueError("unsupported ARC-4 packing revision")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        health = self._json_request("GET", "/health")
        if health.get("status") != "ready" or health.get("cpu_fallback") is not False:
            raise Arc4ConditionerError("ARC-4 conditioner is not CUDA-ready")
        self.revision = str(health.get("conditioner_revision", ""))
        self.packing_revision = str(health.get("packing_revision", ""))
        self.field_order = tuple(health.get("field_order", ()))
        self.output_dim = int(health.get("output_dim", 0))
        self.frame_rate_hz = float(health.get("frame_rate_hz", 0.0))
        if not self.revision or self.output_dim < 1 or self.frame_rate_hz <= 0:
            raise Arc4ConditionerError("ARC-4 conditioner health contract is incomplete")
        if self.packing_revision != expected_packing_revision:
            raise Arc4ConditionerError("ARC-4 conditioner packing contract mismatch")
        if self.field_order != ARC4_FIELD_ORDER:
            raise Arc4ConditionerError("ARC-4 conditioner field-slot contract mismatch")

    def _request(self, method: str, path: str, body: bytes | None = None) -> tuple[bytes, Any]:
        headers = {"accept": "application/json"}
        if body is not None:
            headers["content-type"] = "application/json"
            headers["accept"] = "application/x-safetensors"
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read(), response.headers
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise Arc4ConditionerError(f"ARC-4 request failed: {exc}") from exc

    def _json_request(self, method: str, path: str) -> dict[str, Any]:
        raw, _headers = self._request(method, path)
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise Arc4ConditionerError("ARC-4 service returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise Arc4ConditionerError("ARC-4 service returned a non-object response")
        return payload

    def encode_fields(self, fields: Mapping[str, str], *, max_frames: int) -> torch.Tensor:
        if tuple(fields) != ARC4_FIELD_ORDER:
            raise ValueError("ARC-4 fields do not match the versioned field order")
        payload: dict[str, Any] = {
            "fields": [{"name": name, "text": fields[name]} for name in ARC4_FIELD_ORDER],
            "reference": render_arc4_reference_envelope(fields),
            "packing": self.packing_revision,
        }
        if max_frames is not None:
            if max_frames < 1:
                raise ValueError("max_frames must be positive")
            payload["max_frames"] = max_frames
        body = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        raw, headers = self._request("POST", "/embed", body)
        response_revision = headers.get("X-PersonaPlex-Conditioner-Revision")
        if response_revision != self.revision:
            raise Arc4ConditionerError("conditioner revision changed during the call")
        if headers.get("X-PersonaPlex-Packing-Revision") != self.packing_revision:
            raise Arc4ConditionerError("conditioner packing revision changed during the call")
        try:
            tensors = load(raw)
        except Exception as exc:
            raise Arc4ConditionerError("conditioner returned invalid safetensors") from exc
        tensor = tensors.get("tensor")
        if tensor is None:
            raise Arc4ConditionerError("conditioner response lacks tensor")
        if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[-1] != self.output_dim:
            raise Arc4ConditionerError(f"invalid ARC-4 shape {tuple(tensor.shape)}")
        if tensor.shape[1] < 1 or not torch.isfinite(tensor).all():
            raise Arc4ConditionerError("ARC-4 stream is empty or non-finite")
        return tensor


class Arc4ReferenceProvider:
    """Queue immutable ARC-4 streams through PersonaPlex streaming-sum input."""

    always_condition = True
    unified_evidence = False
    conditioning_mode = "virtual_prefix_v3_plus_arc4_reference_v1"

    def __init__(
        self,
        *,
        lm_gen: Any,
        conditioner_url: str,
        adapter_checkpoint: Path,
        expected_model_revision: str,
        expected_control_adapter_sha256: str | None,
        primary_control: bool = False,
        timeout_seconds: float = 2.0,
        max_reference_frames: int = 64,
        max_cached_frames: int = 64,
    ) -> None:
        if max_reference_frames < 1 or max_cached_frames < 1:
            raise ValueError("ARC-4 frame/cache limits must be positive")
        self.lm_gen = lm_gen
        self.client = Arc4ConditionerClient(
            conditioner_url,
            timeout_seconds=timeout_seconds,
        )
        self.max_reference_frames = max_reference_frames
        self.max_cached_frames = max_cached_frames
        self.primary_control = primary_control
        if primary_control:
            self.always_condition = False
            self.unified_evidence = True
            self.conditioning_mode = "arc4_primary_persistent_stream_v5_field_slots"
        self.bridge = MoshiStreamingSumBridge(lm_gen)
        self._cache: dict[str, Arc4ReferenceCacheEntry] = {}
        hidden_size = int(self.lm_gen.lm_model.dim)
        if self.client.output_dim != hidden_size:
            raise Arc4ConditionerError(
                f"ARC-4 output {self.client.output_dim} != PersonaPlex hidden size {hidden_size}"
            )
        try:
            payload = torch.load(adapter_checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(adapter_checkpoint, map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("arc4_adapter_state_dict"), dict):
            raise Arc4ConditionerError("ARC-4 runtime checkpoint lacks adapter state")
        if payload.get("conditioner_revision") != self.client.revision:
            raise Arc4ConditionerError("ARC-4 checkpoint conditioner revision mismatch")
        if payload.get("model_revision") != expected_model_revision:
            raise Arc4ConditionerError("ARC-4 checkpoint PersonaPlex model revision mismatch")
        if primary_control:
            if payload.get("schema") != "personaplex.arc4-causal-control.v5":
                raise Arc4ConditionerError("ARC-4 primary checkpoint schema mismatch")
            if payload.get("conditioning_mode") != self.conditioning_mode:
                raise Arc4ConditionerError("ARC-4 primary conditioning mode mismatch")
            if payload.get("legacy_prefix_mode") != "disabled":
                raise Arc4ConditionerError("ARC-4 primary checkpoint retained legacy prefix leakage")
        else:
            expected_control = str(expected_control_adapter_sha256 or "").removeprefix("sha256:")
            actual_control = str(payload.get("control_adapter_checkpoint_sha256", "")).removeprefix("sha256:")
            if not expected_control or actual_control != expected_control:
                raise Arc4ConditionerError("ARC-4 checkpoint immediate-prefix dependency mismatch")
        raw_config = payload.get("arc4_adapter_config")
        if not isinstance(raw_config, dict):
            raise Arc4ConditionerError("ARC-4 runtime checkpoint lacks adapter config")
        config = Arc4InjectionConfig(**raw_config)
        if config.hidden_size != hidden_size:
            raise Arc4ConditionerError("ARC-4 adapter hidden size mismatch")
        if primary_control:
            if config.architecture_revision != FIELD_PERSISTENT_ARC4_ARCHITECTURE:
                raise Arc4ConditionerError("ARC-4 primary checkpoint is not field-persistent")
            self.max_reference_frames = sum(config.field_frames)
        self.adapter = GatedArc4InjectionAdapter(config).to(device=self.device)
        self.adapter.load_state_dict(payload["arc4_adapter_state_dict"], strict=True)
        self.adapter.eval()
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(False)
        checkpoint_digest = hashlib.sha256(adapter_checkpoint.read_bytes()).hexdigest()
        self.adapter_version = f"arc4:{self.client.revision}:sha256:{checkpoint_digest}"

    @property
    def device(self) -> torch.device:
        device = next(self.lm_gen.lm_model.parameters()).device
        if device.type != "cuda":
            raise StreamingConditioningError("ARC-4 PersonaPlex runtime is CUDA-only")
        return device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.lm_gen.lm_model.parameters()).dtype

    @torch.inference_mode()
    def build(self, first, second=None, *, expected_revision: int | None = None) -> Arc4ReferenceCacheEntry:
        if self.primary_control:
            control, evidence = first, second
            if expected_revision is not None and control.state_revision != expected_revision:
                raise Arc4ConditionerError("ARC-4 control revision changed before encoding")
        else:
            evidence, control = first, second
        if not isinstance(control, ControlTrainingFrame):
            raise Arc4ConditionerError("ARC-4 build requires a typed control frame")
        if evidence is not None and not isinstance(evidence, EvidenceTrainingFrame):
            raise Arc4ConditionerError("ARC-4 build received invalid evidence")
        reference = render_arc4_reference(control, evidence)
        reference_fields = render_arc4_reference_fields(control, evidence)
        reference_hash = "sha256:" + hashlib.sha256(reference.encode("utf-8")).hexdigest()
        cache_key = self.client.revision + ":" + reference_hash
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        begin = time.perf_counter()
        stream = self.client.encode_fields(reference_fields, max_frames=self.max_reference_frames)
        stream = stream.to(device=self.device, dtype=self.dtype, non_blocking=True).contiguous()
        stream = self.adapter(stream).to(dtype=self.dtype).contiguous()
        torch.cuda.synchronize(self.device)
        evidence_hash = evidence.evidence_hash if evidence is not None else control.frame_hash
        entry = Arc4ReferenceCacheEntry(
            evidence_hash=evidence_hash,
            reference_hash=reference_hash,
            conditioner_revision=self.client.revision,
            stream=stream,
            build_ms=(time.perf_counter() - begin) * 1000.0,
        )
        self._cache[cache_key] = entry
        while len(self._cache) > self.max_cached_frames:
            self._cache.pop(next(iter(self._cache)))
        return entry

    @torch.inference_mode()
    def queue(self, entry: Arc4ReferenceCacheEntry) -> None:
        self.bridge.queue([entry.stream[0]])

    @torch.inference_mode()
    def apply(self, entry: Arc4ReferenceCacheEntry) -> float:
        begin = time.perf_counter()
        self.queue(entry)
        return (time.perf_counter() - begin) * 1000.0

    @torch.inference_mode()
    def cancel(self) -> None:
        self.bridge.cancel(1)
