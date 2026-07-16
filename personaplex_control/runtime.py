"""Native semantic-prefix control for the pinned PersonaPlex/Moshi runtime.

This module deliberately consumes the same ``ControlTrainingFrame`` used by the
native trainer.  It never forwards a natural-language system prompt or target
wording to PersonaPlex.  A valid update is encoded once on the GPU, then its
virtual embedding frames are inserted into the *existing* streaming transformer
state at an acknowledged caller-turn boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Mapping

import torch

from ground_truth_finetuning.training.contracts import (
    ContractError,
    ControlTrainingFrame,
    EvidenceTrainingFrame,
    assert_evidence_control_alignment,
    validate_control_frame_mapping,
    validate_evidence_frame_mapping,
)
from ground_truth_finetuning.training.evidence_conditioning import (
    MoshiStreamingSumBridge,
    StreamingConditioningError,
)
from ground_truth_finetuning.training.plan_serializer import PlanSerializer


CONTROL_MESSAGE_IN = 0x04
CONTROL_MESSAGE_OUT = 0x05
PROTOCOL_VERSION = 2
_FORBIDDEN_WIRE_KEYS = {
    "canonicaltext",
    "canonical_text",
    "targettext",
    "target_text",
    "response",
    "reply",
    "verbatim",
    "prompt",
    "systemprompt",
    "system_prompt",
}


class ControlProtocolError(ValueError):
    """A control message cannot be admitted to the audio plane."""


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ControlProtocolError(f"{name} must be non-empty text")
    return value


def _contains_forbidden_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _FORBIDDEN_WIRE_KEYS or _contains_forbidden_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


@dataclass(frozen=True)
class RuntimeControlUpdate:
    """A versioned, target-wording-free control envelope for one next turn."""

    call_id: str
    revision: int
    context_hash: str
    expires_at_unix_ms: int
    frame: ControlTrainingFrame
    evidence_frame: EvidenceTrainingFrame | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeControlUpdate":
        if value.get("type") != "control.update":
            raise ControlProtocolError("expected control.update")
        if value.get("protocolVersion") != PROTOCOL_VERSION:
            raise ControlProtocolError(f"protocolVersion must be {PROTOCOL_VERSION}")
        if _contains_forbidden_key({key: item for key, item in value.items() if key != "frame"}):
            raise ControlProtocolError("control envelope contains a forbidden wording/prompt field")
        raw_frame = value.get("frame")
        if not isinstance(raw_frame, Mapping):
            raise ControlProtocolError("control.update requires a typed frame object")
        try:
            frame = validate_control_frame_mapping(raw_frame)
        except ContractError as exc:
            raise ControlProtocolError(str(exc)) from exc
        raw_evidence = value.get("evidenceFrame")
        evidence_frame: EvidenceTrainingFrame | None = None
        if raw_evidence is not None:
            if not isinstance(raw_evidence, Mapping):
                raise ControlProtocolError("evidenceFrame must be a typed evidence frame object")
            try:
                evidence_frame = validate_evidence_frame_mapping(raw_evidence)
                assert_evidence_control_alignment(frame, evidence_frame)
            except ContractError as exc:
                raise ControlProtocolError(str(exc)) from exc
            if evidence_frame.plan.plan_hash != frame.plan.plan_hash:
                raise ControlProtocolError("evidenceFrame.plan must match the active control frame plan")
        call_id = _required_text(value.get("callId"), "callId")
        context_hash = _required_text(value.get("contextHash"), "contextHash")
        revision = value.get("revision")
        expires_at_unix_ms = value.get("expiresAtUnixMs")
        if not isinstance(revision, int) or revision < 1:
            raise ControlProtocolError("revision must be a positive integer")
        if not isinstance(expires_at_unix_ms, int) or expires_at_unix_ms <= 0:
            raise ControlProtocolError("expiresAtUnixMs must be a positive Unix timestamp in milliseconds")
        if call_id != frame.conversation_id:
            raise ControlProtocolError("callId must equal frame.conversationId")
        if revision != frame.state_revision:
            raise ControlProtocolError("revision must equal frame.stateRevision")
        if context_hash != frame.state_hash:
            raise ControlProtocolError("contextHash must equal frame.stateHash")
        if evidence_frame is not None and evidence_frame.conversation_id != call_id:
            raise ControlProtocolError("evidenceFrame conversationId must equal callId")
        return cls(call_id, revision, context_hash, expires_at_unix_ms, frame, evidence_frame)

    @classmethod
    def from_wire(cls, payload: str | bytes) -> "RuntimeControlUpdate":
        try:
            raw = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControlProtocolError("control.update must be UTF-8 JSON") from exc
        if not isinstance(raw, Mapping):
            raise ControlProtocolError("control.update must be an object")
        return cls.from_mapping(raw)


@dataclass(frozen=True)
class RuntimeEvidenceUpdate:
    """Bounded late evidence for the next valid semantic-control revision.

    Accepted evidence has no direct rendering path.  Until the separately
    trained evidence adapter is installed, it is acknowledged as deferred after
    invalidating stale audio.  The following Nemotron control revision must
    explicitly authorize any resulting agent speech.
    """

    call_id: str
    revision: int
    supports_control_revision: int
    context_hash: str
    expires_at_unix_ms: int
    evidence_id: str
    provenance: Mapping[str, Any]
    allowed_claims: tuple[str, ...]
    availability: str

    @property
    def evidence_hash(self) -> str:
        canonical = json.dumps(
            {
                "callId": self.call_id,
                "revision": self.revision,
                "supportsControlRevision": self.supports_control_revision,
                "contextHash": self.context_hash,
                "expiresAtUnixMs": self.expires_at_unix_ms,
                "evidenceId": self.evidence_id,
                "provenance": self.provenance,
                "allowedClaims": self.allowed_claims,
                "availability": self.availability,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(canonical).hexdigest()}"

    def matches_frame(self, frame: EvidenceTrainingFrame) -> bool:
        """Bind a staged source event to the later, trainable evidence frame.

        The transport event cannot contain a future control plan. The semantic
        authority stages it first, then submits a later control update whose
        typed evidence frame proves the same source fact was incorporated.
        """

        return (
            frame.conversation_id == self.call_id
            and frame.evidence_id == self.evidence_id
            and frame.evidence_revision == self.revision
            and frame.supports_control_revision == self.supports_control_revision
            and frame.context_hash == self.context_hash
            and frame.availability == self.availability
            and frame.provenance == dict(self.provenance)
            and frame.allowed_claims == self.allowed_claims
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeEvidenceUpdate":
        if value.get("type") != "evidence.update":
            raise ControlProtocolError("expected evidence.update")
        if value.get("protocolVersion") != PROTOCOL_VERSION:
            raise ControlProtocolError(f"protocolVersion must be {PROTOCOL_VERSION}")
        if _contains_forbidden_key(value):
            raise ControlProtocolError("evidence envelope contains a forbidden wording/prompt field")
        call_id = _required_text(value.get("callId"), "callId")
        context_hash = _required_text(value.get("contextHash"), "contextHash")
        evidence_id = _required_text(value.get("evidenceId"), "evidenceId")
        revision = value.get("revision")
        supports_control_revision = value.get("supportsControlRevision")
        expires_at_unix_ms = value.get("expiresAtUnixMs")
        provenance = value.get("provenance")
        allowed_claims = value.get("allowedClaims")
        availability = value.get("availability")
        if not isinstance(revision, int) or revision < 1:
            raise ControlProtocolError("revision must be a positive integer")
        if not isinstance(supports_control_revision, int) or supports_control_revision < 1:
            raise ControlProtocolError("supportsControlRevision must be a positive integer")
        if not isinstance(expires_at_unix_ms, int) or expires_at_unix_ms <= 0:
            raise ControlProtocolError("expiresAtUnixMs must be a positive Unix timestamp in milliseconds")
        if len(evidence_id) > 256:
            raise ControlProtocolError("evidenceId exceeds the bounded wire limit")
        if availability not in {"ready", "failed", "expired"}:
            raise ControlProtocolError("availability must be ready, failed, or expired")
        if not isinstance(provenance, Mapping) or not provenance or len(provenance) > 16:
            raise ControlProtocolError("provenance must contain 1-16 scalar fields")
        if any(
            not isinstance(key, str)
            or not isinstance(item, (str, int, float, bool, type(None)))
            or (isinstance(item, str) and len(item) > 512)
            for key, item in provenance.items()
        ):
            raise ControlProtocolError("provenance must contain bounded scalar fields")
        if not isinstance(allowed_claims, list) or len(allowed_claims) > 16:
            raise ControlProtocolError("allowedClaims must contain at most 16 bounded claims")
        if availability == "ready" and not allowed_claims:
            raise ControlProtocolError("ready evidence requires at least one allowed claim")
        if any(not isinstance(claim, str) or not claim or len(claim) > 512 for claim in allowed_claims):
            raise ControlProtocolError("allowedClaims must contain bounded strings")
        return cls(
            call_id=call_id,
            revision=revision,
            supports_control_revision=supports_control_revision,
            context_hash=context_hash,
            expires_at_unix_ms=expires_at_unix_ms,
            evidence_id=evidence_id,
            provenance=dict(provenance),
            allowed_claims=tuple(allowed_claims),
            availability=availability,
        )

    @classmethod
    def from_wire(cls, payload: str | bytes) -> "RuntimeEvidenceUpdate":
        try:
            raw = json.loads(payload.decode("utf-8") if isinstance(payload, bytes) else payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ControlProtocolError("evidence.update must be UTF-8 JSON") from exc
        if not isinstance(raw, Mapping):
            raise ControlProtocolError("evidence.update must be an object")
        return cls.from_mapping(raw)


@dataclass(frozen=True)
class ControlAck:
    """Terminal or informational control status safe for broad operational logs."""

    call_id: str
    revision: int
    context_hash: str
    status: str
    reason: str
    turn_id: int | None = None
    generation_id: int | None = None
    frame_hash: str | None = None
    evidence_hash: str | None = None
    adapter_version: str | None = None
    prefix_build_ms: float | None = None
    prefix_prefill_ms: float | None = None

    def as_wire_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": "control.ack",
            "protocolVersion": PROTOCOL_VERSION,
            "callId": self.call_id,
            "revision": self.revision,
            "contextHash": self.context_hash,
            "status": self.status,
            "reason": self.reason,
        }
        optional = {
            "turnId": self.turn_id,
            "generationId": self.generation_id,
            "frameHash": self.frame_hash,
            "evidenceHash": self.evidence_hash,
            "adapterVersion": self.adapter_version,
            "prefixBuildMs": self.prefix_build_ms,
            "prefixPrefillMs": self.prefix_prefill_ms,
        }
        data.update({key: value for key, value in optional.items() if value is not None})
        return data

    def to_wire(self) -> bytes:
        return json.dumps(self.as_wire_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class PrefixCacheEntry:
    frame_hash: str
    prefix: torch.Tensor
    build_ms: float


@dataclass(frozen=True)
class EvidenceCacheEntry:
    evidence_hash: str
    stream: torch.Tensor
    build_ms: float


@dataclass(frozen=True)
class PendingGeneration:
    update: RuntimeControlUpdate
    control_prefix: PrefixCacheEntry
    evidence_stream: EvidenceCacheEntry | None


class SemanticPrefixProvider:
    """Builds and injects cached semantic virtual tokens into a live LMGen.

    ``LMModel.forward_embeddings`` advances only the streaming transformer.  The
    provider intentionally does not call ``LMGen.step`` for prefix frames, so no
    speech/text code token is sampled, cached in delayed-code state, or emitted.
    The following real audio-code step therefore conditions on the prefix while
    retaining the existing duplex history.
    """

    def __init__(
        self,
        *,
        lm_gen: Any,
        adapter: torch.nn.Module,
        tokenizer: Any,
        adapter_version: str,
        max_plan_tokens: int = 512,
        max_cached_frames: int = 32,
    ) -> None:
        if max_plan_tokens < 1 or max_cached_frames < 1:
            raise ValueError("prefix limits must be positive")
        self.lm_gen = lm_gen
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.adapter_version = adapter_version
        self.max_plan_tokens = max_plan_tokens
        self.max_cached_frames = max_cached_frames
        self.serializer = PlanSerializer()
        self._cache: dict[str, PrefixCacheEntry] = {}

    @property
    def device(self) -> torch.device:
        return next(self.lm_gen.lm_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.lm_gen.lm_model.parameters()).dtype

    def _elapsed_ms(self, begin: float) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - begin) * 1000.0

    @torch.inference_mode()
    def build(self, frame: ControlTrainingFrame) -> PrefixCacheEntry:
        cached = self._cache.get(frame.frame_hash)
        if cached is not None:
            return cached
        token_ids = self.serializer.encode_frame(
            frame,
            self.tokenizer,
            int(self.lm_gen.lm_model.text_card),
        )[: self.max_plan_tokens]
        if not token_ids:
            raise ControlProtocolError("control frame encoded to no tokenizer IDs")
        start = time.perf_counter()
        ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).unsqueeze(0)
        mask = torch.ones_like(ids, dtype=torch.bool)
        prefix = self.adapter(ids, mask)
        if prefix.ndim != 3 or prefix.shape[0] != 1 or prefix.shape[-1] != int(self.lm_gen.lm_model.dim):
            raise ControlProtocolError("adapter emitted an invalid control prefix shape")
        entry = PrefixCacheEntry(
            frame_hash=frame.frame_hash,
            prefix=prefix.to(device=self.device, dtype=self.dtype).contiguous(),
            build_ms=self._elapsed_ms(start),
        )
        self._cache[frame.frame_hash] = entry
        while len(self._cache) > self.max_cached_frames:
            self._cache.pop(next(iter(self._cache)))
        return entry

    @torch.inference_mode()
    def prefill(self, entry: PrefixCacheEntry) -> float:
        if not self.lm_gen.lm_model.is_streaming:
            raise ControlProtocolError("PersonaPlex LM is not in streaming mode")
        start = time.perf_counter()
        for index in range(entry.prefix.shape[1]):
            # One virtual frame at a time preserves the server's streaming
            # transformer's causal cache semantics and avoids a media emission.
            self.lm_gen.lm_model.forward_embeddings(entry.prefix[:, index : index + 1])
        return self._elapsed_ms(start)


class EvidenceStreamProvider:
    """GPU-cached MoshiRAG-style evidence streams for an acknowledged turn.

    Evidence is encoded only after the semantic authority produces a later
    control revision that explicitly aligns it to the next target turn. The
    resulting stream is queued into the patched native generator, where one row
    is added to each real code-step embedding. It never becomes a prompt or a
    generated code token by itself.
    """

    def __init__(
        self,
        *,
        lm_gen: Any,
        adapter: torch.nn.Module,
        tokenizer: Any,
        adapter_version: str,
        max_evidence_tokens: int = 256,
        max_cached_frames: int = 32,
    ) -> None:
        if max_evidence_tokens < 1 or max_cached_frames < 1:
            raise ValueError("evidence limits must be positive")
        self.lm_gen = lm_gen
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.adapter_version = adapter_version
        self.max_evidence_tokens = max_evidence_tokens
        self.max_cached_frames = max_cached_frames
        self.serializer = PlanSerializer()
        self.bridge = MoshiStreamingSumBridge(lm_gen)
        self._cache: dict[str, EvidenceCacheEntry] = {}

    @property
    def device(self) -> torch.device:
        return next(self.lm_gen.lm_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.lm_gen.lm_model.parameters()).dtype

    def _elapsed_ms(self, begin: float) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return (time.perf_counter() - begin) * 1000.0

    @torch.inference_mode()
    def build(self, evidence: EvidenceTrainingFrame, control: ControlTrainingFrame) -> EvidenceCacheEntry:
        assert_evidence_control_alignment(control, evidence)
        cached = self._cache.get(evidence.evidence_hash)
        if cached is not None:
            return cached
        token_ids = self.serializer.encode_evidence(
            evidence,
            control,
            self.tokenizer,
            int(self.lm_gen.lm_model.text_card),
        )[: self.max_evidence_tokens]
        if not token_ids:
            raise ControlProtocolError("evidence frame encoded to no tokenizer IDs")
        start = time.perf_counter()
        ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).unsqueeze(0)
        mask = torch.ones_like(ids, dtype=torch.bool)
        stream = self.adapter(ids, mask)
        if stream.ndim != 3 or stream.shape[0] != 1 or stream.shape[-1] != int(self.lm_gen.lm_model.dim):
            raise ControlProtocolError("evidence adapter emitted an invalid streaming-sum shape")
        entry = EvidenceCacheEntry(
            evidence_hash=evidence.evidence_hash,
            stream=stream.to(device=self.device, dtype=self.dtype).contiguous(),
            build_ms=self._elapsed_ms(start),
        )
        self._cache[evidence.evidence_hash] = entry
        while len(self._cache) > self.max_cached_frames:
            self._cache.pop(next(iter(self._cache)))
        return entry

    @torch.inference_mode()
    def queue(self, entry: EvidenceCacheEntry) -> None:
        self.bridge.queue([entry.stream[0]])

    @torch.inference_mode()
    def cancel(self) -> None:
        self.bridge.cancel(1)


class RuntimeControlSession:
    """Per-call revision, prefix, acknowledgement, and cancellation state."""

    def __init__(
        self,
        *,
        call_id: str,
        prefix_provider: SemanticPrefixProvider,
        evidence_provider: EvidenceStreamProvider | None = None,
        allow_uncontrolled_audio: bool = False,
        prefill_deadline_ms: float = 120.0,
    ) -> None:
        if prefill_deadline_ms <= 0:
            raise ValueError("prefill_deadline_ms must be positive")
        self.call_id = _required_text(call_id, "call_id")
        self.prefix_provider = prefix_provider
        self.evidence_provider = evidence_provider
        self.allow_uncontrolled_audio = allow_uncontrolled_audio
        self.prefill_deadline_ms = prefill_deadline_ms
        self.last_seen_revision = 0
        self.pending: PendingGeneration | None = None
        self.active: RuntimeControlUpdate | None = None
        self.staged_evidence: RuntimeEvidenceUpdate | None = None
        self.generation_id = 0
        self.render_allowed = allow_uncontrolled_audio
        self._statuses: dict[int, ControlAck] = {}

    def _ack(
        self,
        update: RuntimeControlUpdate | RuntimeEvidenceUpdate,
        status: str,
        reason: str,
        *,
        turn_id: int | None = None,
        prefill_ms: float | None = None,
        build_ms: float | None = None,
    ) -> ControlAck:
        if isinstance(update, RuntimeControlUpdate):
            artifact = {"frame_hash": update.frame.frame_hash}
            if update.evidence_frame is not None:
                artifact["evidence_hash"] = update.evidence_frame.evidence_hash
        else:
            artifact = {"evidence_hash": update.evidence_hash}
        return ControlAck(
            call_id=update.call_id,
            revision=update.revision,
            context_hash=update.context_hash,
            status=status,
            reason=reason,
            turn_id=turn_id,
            generation_id=self.generation_id,
            adapter_version=self.prefix_provider.adapter_version,
            prefix_build_ms=build_ms,
            prefix_prefill_ms=prefill_ms,
            **artifact,
        )

    def submit(self, update: RuntimeControlUpdate, *, now_unix_ms: int | None = None) -> list[ControlAck]:
        now_unix_ms = now_unix_ms if now_unix_ms is not None else int(time.time() * 1000)
        if update.call_id != self.call_id:
            return [self._ack(update, "rejected", "call_id_mismatch")]
        existing = self._statuses.get(update.revision)
        if existing is not None:
            if existing.context_hash == update.context_hash and existing.frame_hash == update.frame.frame_hash:
                return [existing]
            return [self._ack(update, "rejected", "revision_reused_with_different_frame")]
        if update.revision <= self.last_seen_revision:
            return [self._ack(update, "rejected", "stale_revision")]
        self.last_seen_revision = update.revision
        if update.expires_at_unix_ms <= now_unix_ms:
            ack = self._ack(update, "expired", "control_update_expired")
            self._statuses[update.revision] = ack
            return [ack]
        acknowledgements: list[ControlAck] = []
        if self.pending is not None:
            prior = self.pending.update
            prior_ack = self._ack(prior, "superseded", "newer_control_update_received")
            self._statuses[prior.revision] = prior_ack
            acknowledgements.append(prior_ack)
            self.pending = None
        try:
            entry = self.prefix_provider.build(update.frame)
        except (ControlProtocolError, RuntimeError, ValueError) as exc:
            ack = self._ack(update, "prefix_build_failed", str(exc))
            self._statuses[update.revision] = ack
            acknowledgements.append(ack)
            return acknowledgements
        evidence_entry: EvidenceCacheEntry | None = None
        if update.evidence_frame is not None:
            staged = self.staged_evidence
            if staged is None:
                ack = self._ack(update, "rejected", "evidence_frame_has_no_staged_source_event")
                self._statuses[update.revision] = ack
                acknowledgements.append(ack)
                return acknowledgements
            if staged.expires_at_unix_ms <= now_unix_ms:
                ack = self._ack(update, "expired", "staged_evidence_expired_before_control_binding")
                self._statuses[update.revision] = ack
                acknowledgements.append(ack)
                return acknowledgements
            if not staged.matches_frame(update.evidence_frame):
                ack = self._ack(update, "rejected", "evidence_frame_does_not_match_staged_source_event")
                self._statuses[update.revision] = ack
                acknowledgements.append(ack)
                return acknowledgements
            if self.evidence_provider is None:
                ack = self._ack(update, "rejected", "evidence_adapter_not_installed")
                self._statuses[update.revision] = ack
                acknowledgements.append(ack)
                return acknowledgements
            try:
                evidence_entry = self.evidence_provider.build(update.evidence_frame, update.frame)
            except (ControlProtocolError, StreamingConditioningError, RuntimeError, ValueError) as exc:
                ack = self._ack(update, "prefix_build_failed", str(exc))
                self._statuses[update.revision] = ack
                acknowledgements.append(ack)
                return acknowledgements
            self.staged_evidence = None
        elif self.staged_evidence is not None:
            ack = self._ack(update, "rejected", "fresh_staged_evidence_must_be_bound_to_control_revision")
            self._statuses[update.revision] = ack
            acknowledgements.append(ack)
            return acknowledgements
        self.pending = PendingGeneration(update, entry, evidence_entry)
        queued = self._ack(update, "queued", "prefix_cached_waiting_for_caller_turn_boundary", build_ms=entry.build_ms)
        self._statuses[update.revision] = queued
        acknowledgements.append(queued)
        return acknowledgements

    def submit_evidence(self, update: RuntimeEvidenceUpdate, *, now_unix_ms: int | None = None) -> list[ControlAck]:
        """Validate late evidence, cancel stale output, and defer learned use."""

        now_unix_ms = now_unix_ms if now_unix_ms is not None else int(time.time() * 1000)
        if update.call_id != self.call_id:
            return [self._ack(update, "rejected", "call_id_mismatch")]
        existing = self._statuses.get(update.revision)
        if existing is not None:
            if existing.context_hash == update.context_hash and existing.evidence_hash == update.evidence_hash:
                return [existing]
            return [self._ack(update, "rejected", "revision_reused_with_different_evidence")]
        if update.revision <= self.last_seen_revision:
            return [self._ack(update, "rejected", "stale_revision")]
        supporting = self.pending.update if self.pending is not None else self.active
        if supporting is None or supporting.revision != update.supports_control_revision:
            return [self._ack(update, "rejected", "supporting_control_revision_not_current")]
        if supporting.context_hash != update.context_hash:
            return [self._ack(update, "context_mismatch", "evidence_context_hash_does_not_match_control")]
        self.last_seen_revision = update.revision
        if update.expires_at_unix_ms <= now_unix_ms or update.availability == "expired":
            acknowledgement = self._ack(update, "expired", "evidence_update_expired")
            self._statuses[update.revision] = acknowledgement
            return [acknowledgement]
        acknowledgements: list[ControlAck] = []
        self.generation_id += 1
        self.render_allowed = False
        if self.pending is not None:
            prior = self.pending.update
            self.pending = None
            acknowledgement = self._ack(prior, "superseded", "evidence_update_received")
            self._statuses[prior.revision] = acknowledgement
            acknowledgements.append(acknowledgement)
        if self.active is not None:
            prior = self.active
            self.active = None
            acknowledgement = self._ack(prior, "superseded", "evidence_update_received")
            self._statuses[prior.revision] = acknowledgement
            acknowledgements.append(acknowledgement)
        self.staged_evidence = update
        reason = "evidence_staged_waiting_for_bound_control_revision"
        if update.availability == "failed":
            reason = "evidence_staged_backend_reported_failure"
        acknowledgement = self._ack(update, "evidence_staged", reason)
        self._statuses[update.revision] = acknowledgement
        acknowledgements.append(acknowledgement)
        return acknowledgements

    def apply_boundary(
        self,
        *,
        call_id: str,
        turn_id: int,
        context_hash: str,
        now_unix_ms: int | None = None,
    ) -> ControlAck:
        now_unix_ms = now_unix_ms if now_unix_ms is not None else int(time.time() * 1000)
        if call_id != self.call_id:
            return ControlAck(call_id, 0, context_hash, "rejected", "call_id_mismatch", turn_id=turn_id)
        if not isinstance(turn_id, int) or turn_id < 0:
            return ControlAck(call_id, 0, context_hash, "rejected", "invalid_caller_turn_id", turn_id=turn_id)
        if self.pending is None:
            self.generation_id += 1
            self.render_allowed = False
            return ControlAck(call_id, self.last_seen_revision, context_hash, "safe_fallback", "missing_fresh_control_update", turn_id=turn_id, generation_id=self.generation_id)
        pending = self.pending
        update = pending.update
        entry = pending.control_prefix
        self.pending = None
        if update.expires_at_unix_ms <= now_unix_ms:
            self.generation_id += 1
            self.render_allowed = False
            ack = self._ack(update, "expired", "control_update_expired_before_boundary", turn_id=turn_id, build_ms=entry.build_ms)
            self._statuses[update.revision] = ack
            return ack
        if update.context_hash != context_hash:
            self.generation_id += 1
            self.render_allowed = False
            ack = self._ack(update, "context_mismatch", "boundary_context_hash_does_not_match_frame", turn_id=turn_id, build_ms=entry.build_ms)
            self._statuses[update.revision] = ack
            return ack
        try:
            prefill_ms = self.prefix_provider.prefill(entry)
            if pending.evidence_stream is not None:
                if self.evidence_provider is None:
                    raise ControlProtocolError("evidence adapter disappeared before generation boundary")
                self.evidence_provider.queue(pending.evidence_stream)
            if prefill_ms > self.prefill_deadline_ms:
                raise ControlProtocolError(f"prefix_prefill_deadline_exceeded:{prefill_ms:.2f}ms")
        except (ControlProtocolError, RuntimeError, ValueError) as exc:
            self.generation_id += 1
            self.render_allowed = False
            ack = self._ack(update, "prefix_build_failed", str(exc), turn_id=turn_id, build_ms=entry.build_ms)
            self._statuses[update.revision] = ack
            return ack
        self.active = update
        self.generation_id += 1
        self.render_allowed = True
        ack = self._ack(update, "applied", "prefix_prefilled_at_caller_turn_boundary", turn_id=turn_id, prefill_ms=prefill_ms, build_ms=entry.build_ms)
        self._statuses[update.revision] = ack
        return ack

    def caller_barge_in(self, *, reason: str = "caller_barge_in") -> list[ControlAck]:
        acknowledgements: list[ControlAck] = []
        self.generation_id += 1
        self.render_allowed = False
        if self.evidence_provider is not None:
            try:
                self.evidence_provider.cancel()
            except (StreamingConditioningError, RuntimeError, ValueError):
                # Emission is already invalidated above. The next native step
                # receives a zero stream even if its queue reset cannot run.
                pass
        if self.pending is not None:
            update = self.pending.update
            self.pending = None
            ack = self._ack(update, "superseded", reason)
            self._statuses[update.revision] = ack
            acknowledgements.append(ack)
        if self.active is not None:
            update = self.active
            self.active = None
            ack = self._ack(update, "superseded", reason)
            self._statuses[update.revision] = ack
            acknowledgements.append(ack)
        return acknowledgements

    def may_emit(self, generation_id: int) -> bool:
        return self.render_allowed and generation_id == self.generation_id
