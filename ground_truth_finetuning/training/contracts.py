"""Typed contracts shared by corpus, trainer, and runtime integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import re
from typing import Any, Mapping


_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MODES = {"expressive", "strict", "safe_fallback"}
_INTERRUPTIBILITY = {"yield_on_caller_speech", "complete_if_uninterrupted"}
_SEMANTIC_SOURCES = {
    "state_reducer", "asr_finalizer", "policy_agent", "task_agent", "tool_result",
    "interruption_controller", "handoff_router", "timer", "knowledge_agent", "safety_agent",
}
_FORBIDDEN_CONTROL_KEYS = {"canonicaltext", "canonical_text", "targettext", "target_text", "response", "reply", "verbatim"}


class ContractError(ValueError):
    """A control or dataset contract violates a non-negotiable invariant."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def sha256_uri(value: Any) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{key} must be a non-empty string")
    return value


def _string_list(mapping: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = mapping.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ContractError(f"{key} must be a list of non-empty strings")
    return tuple(value)


def _stream_index_list(mapping: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value:
        raise ContractError(f"{key} must be a non-empty list of stream indices")
    if not all(isinstance(item, int) and item >= 0 for item in value):
        raise ContractError(f"{key} must contain non-negative integer stream indices")
    if len(set(value)) != len(value):
        raise ContractError(f"{key} must not contain duplicate stream indices")
    return tuple(value)


@dataclass(frozen=True)
class StreamLayout:
    """The exact global-codebook ownership for a loaded duplex PersonaPlex LM.

    PersonaPlex's 17 streams are not interchangeable.  The canonical Moshi layout
    has one text stream, eight agent-output Mimi streams, and eight caller-input
    Mimi streams.  Training must name those groups explicitly because ``dep_q``
    spans both audio directions in the native model.
    """

    text_stream_indices: tuple[int, ...]
    agent_audio_stream_indices: tuple[int, ...]
    caller_audio_stream_indices: tuple[int, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StreamLayout":
        layout = cls(
            text_stream_indices=_stream_index_list(value, "text_stream_indices"),
            agent_audio_stream_indices=_stream_index_list(value, "agent_audio_stream_indices"),
            caller_audio_stream_indices=_stream_index_list(value, "caller_audio_stream_indices"),
        )
        layout.validate_static()
        return layout

    def as_dict(self) -> dict[str, list[int]]:
        return {
            "text_stream_indices": list(self.text_stream_indices),
            "agent_audio_stream_indices": list(self.agent_audio_stream_indices),
            "caller_audio_stream_indices": list(self.caller_audio_stream_indices),
        }

    def validate_static(self) -> None:
        all_indices = (
            self.text_stream_indices
            + self.agent_audio_stream_indices
            + self.caller_audio_stream_indices
        )
        if len(set(all_indices)) != len(all_indices):
            raise ContractError("stream-layout groups must be disjoint")
        if len(self.text_stream_indices) != 1:
            raise ContractError("current native training supports exactly one text stream")

    def validate_for_model(self, lm_model: object) -> None:
        self.validate_static()
        expected = set(range(int(lm_model.num_codebooks)))
        actual = set(
            self.text_stream_indices
            + self.agent_audio_stream_indices
            + self.caller_audio_stream_indices
        )
        if actual != expected:
            raise ContractError(
                "stream-layout must account for every loaded model codebook exactly once"
            )
        if self.text_stream_indices != (0,):
            raise ContractError("native delayed path currently requires the text stream at index 0")
        audio_start = int(lm_model.audio_offset)
        audio_end = audio_start + int(lm_model.dep_q)
        for stream_index in self.agent_audio_stream_indices + self.caller_audio_stream_indices:
            if not audio_start <= stream_index < audio_end:
                raise ContractError("audio stream lies outside the native depformer range")

    def agent_audio_output_indices(self, lm_model: object) -> tuple[int, ...]:
        self.validate_for_model(lm_model)
        return tuple(index - int(lm_model.audio_offset) for index in self.agent_audio_stream_indices)


@dataclass(frozen=True)
class Delivery:
    language: str
    register: str
    assertiveness: float
    interruptibility: str
    max_duration_ms: int
    speaking_rate_bucket: str = "normal"
    pause_density_bucket: str = "moderate"
    emphasis_targets: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Delivery":
        assertiveness = value.get("assertiveness")
        max_duration_ms = value.get("max_duration_ms")
        interruptibility = _required_string(value, "interruptibility")
        if not isinstance(assertiveness, (int, float)) or not 0 <= float(assertiveness) <= 1:
            raise ContractError("delivery.assertiveness must be between 0 and 1")
        if not isinstance(max_duration_ms, int) or not 250 <= max_duration_ms <= 30000:
            raise ContractError("delivery.max_duration_ms must be an integer between 250 and 30000")
        if interruptibility not in _INTERRUPTIBILITY:
            raise ContractError("delivery.interruptibility is unsupported")
        return cls(
            language=_required_string(value, "language"),
            register=_required_string(value, "register"),
            assertiveness=float(assertiveness),
            interruptibility=interruptibility,
            max_duration_ms=max_duration_ms,
            speaking_rate_bucket=str(value.get("speaking_rate_bucket", "normal")),
            pause_density_bucket=str(value.get("pause_density_bucket", "moderate")),
            emphasis_targets=_string_list(value, "emphasis_targets"),
        )


@dataclass(frozen=True)
class Constraints:
    required_facts: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    must_ask: tuple[str, ...]
    must_not_request: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "Constraints":
        return cls(
            required_facts=_string_list(value, "required_facts"),
            forbidden_claims=_string_list(value, "forbidden_claims"),
            must_ask=_string_list(value, "must_ask"),
            must_not_request=_string_list(value, "must_not_request"),
        )


@dataclass(frozen=True)
class ControlPlan:
    schema_version: int
    call_id: str
    turn_id: int
    revision: int
    context_hash: str
    mode: str
    intent: str
    dialogue_act: str
    entities: dict[str, str]
    constraints: Constraints
    delivery: Delivery
    expiry_ms: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ControlPlan":
        forbidden = {"canonicalText", "canonical_text", "targetText", "target_text", "response"}
        present = forbidden.intersection(value)
        if present:
            raise ContractError(f"plan must not contain canonical-response fields: {sorted(present)}")
        schema_version = value.get("schemaVersion", value.get("schema_version"))
        if schema_version != 1:
            raise ContractError("only control plan schema version 1 is supported")
        turn_id = value.get("turnId", value.get("turn_id"))
        revision = value.get("revision")
        expiry_ms = value.get("expiryMs", value.get("expiry_ms"))
        if not isinstance(turn_id, int) or turn_id < 0:
            raise ContractError("turnId must be a non-negative integer")
        if not isinstance(revision, int) or revision < 0:
            raise ContractError("revision must be a non-negative integer")
        if not isinstance(expiry_ms, int) or not 250 <= expiry_ms <= 30000:
            raise ContractError("expiryMs must be between 250 and 30000")
        context_hash = _required_string(value, "contextHash")
        if not _HASH_RE.match(context_hash):
            raise ContractError("contextHash must be sha256:<64 lowercase hex characters>")
        mode = _required_string(value, "mode")
        if mode not in _MODES:
            raise ContractError(f"mode must be one of {sorted(_MODES)}")
        entities = value.get("entities", {})
        if not isinstance(entities, dict) or not all(
            isinstance(key, str) and key and isinstance(item, str) and item
            for key, item in entities.items()
        ):
            raise ContractError("entities must map non-empty strings to non-empty strings")
        constraints = value.get("constraints")
        delivery = value.get("delivery")
        if not isinstance(constraints, Mapping) or not isinstance(delivery, Mapping):
            raise ContractError("constraints and delivery must be mappings")
        return cls(
            schema_version=schema_version,
            call_id=_required_string(value, "callId"),
            turn_id=turn_id,
            revision=revision,
            context_hash=context_hash,
            mode=mode,
            intent=_required_string(value, "intent"),
            dialogue_act=_required_string(value, "dialogueAct"),
            entities=dict(sorted(entities.items())),
            constraints=Constraints.from_mapping(constraints),
            delivery=Delivery.from_mapping(delivery),
            expiry_ms=expiry_ms,
        )

    def as_wire_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "callId": self.call_id,
            "turnId": self.turn_id,
            "revision": self.revision,
            "contextHash": self.context_hash,
            "mode": self.mode,
            "intent": self.intent,
            "dialogueAct": self.dialogue_act,
            "entities": self.entities,
            "constraints": asdict(self.constraints),
            "delivery": asdict(self.delivery),
            "expiryMs": self.expiry_ms,
        }

    @property
    def plan_hash(self) -> str:
        return sha256_uri(self.as_wire_dict())


def validate_plan_mapping(value: Mapping[str, Any]) -> ControlPlan:
    return ControlPlan.from_mapping(value)


def assert_monotonic_revision(previous: ControlPlan | None, candidate: ControlPlan) -> None:
    if previous is None:
        return
    if previous.call_id != candidate.call_id:
        raise ContractError("cannot compare revisions from different calls")
    if candidate.revision <= previous.revision:
        raise ContractError("control revision must strictly increase")


def _bounded_control_value(value: Any, path: str, *, depth: int = 0) -> Any:
    if depth > 5:
        raise ContractError(f"{path} is nested too deeply")
    if isinstance(value, str):
        if not value.strip() or len(value) > 512:
            raise ContractError(f"{path} must be non-empty text up to 512 characters")
        return value
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise ContractError(f"{path} must be finite")
        return value
    if isinstance(value, list):
        if len(value) > 32:
            raise ContractError(f"{path} has too many items")
        return [_bounded_control_value(item, f"{path}[{index}]", depth=depth + 1) for index, item in enumerate(value)]
    if isinstance(value, Mapping):
        if len(value) > 32:
            raise ContractError(f"{path} has too many fields")
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 96:
                raise ContractError(f"{path} has an invalid field name")
            if key.casefold() in _FORBIDDEN_CONTROL_KEYS:
                raise ContractError(f"{path}.{key} is a forbidden canonical-response field")
            normalized[key] = _bounded_control_value(item, f"{path}.{key}", depth=depth + 1)
        return dict(sorted(normalized.items()))
    raise ContractError(f"{path} has unsupported value type")


@dataclass(frozen=True)
class ControlTrainingFrame:
    """Causal, target-wording-free semantic context for one agent turn."""

    schema_version: int
    frame_id: str
    conversation_id: str
    target_turn_id: int
    state_revision: int
    base_state_hash: str
    state_hash: str
    semantic_sources: tuple[str, ...]
    state: dict[str, Any]
    update: dict[str, Any]
    turn_taking: dict[str, Any]
    plan: ControlPlan

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ControlTrainingFrame":
        if value.get("schemaVersion", value.get("schema_version")) != 1:
            raise ContractError("only control training-frame schema version 1 is supported")
        plan_value = value.get("plan")
        if not isinstance(plan_value, Mapping):
            raise ContractError("control frame requires an inline typed plan")
        plan = ControlPlan.from_mapping(plan_value)
        state_hash = _required_string(value, "stateHash")
        base_state_hash = _required_string(value, "baseStateHash")
        if not _HASH_RE.match(state_hash) or not _HASH_RE.match(base_state_hash):
            raise ContractError("control-frame state hashes must be sha256 URIs")
        if plan.context_hash != state_hash:
            raise ContractError("plan.contextHash must equal control-frame stateHash")
        target_turn_id = value.get("targetTurnId")
        state_revision = value.get("stateRevision")
        if not isinstance(target_turn_id, int) or target_turn_id < 0:
            raise ContractError("targetTurnId must be a non-negative integer")
        if target_turn_id != plan.turn_id:
            raise ContractError("targetTurnId must equal plan.turnId")
        if not isinstance(state_revision, int) or state_revision < 1:
            raise ContractError("stateRevision must be a positive integer")
        sources = value.get("semanticSources")
        if not isinstance(sources, list) or not sources or not all(isinstance(item, str) and item in _SEMANTIC_SOURCES for item in sources):
            raise ContractError("semanticSources must contain known semantic agents")
        if len(set(sources)) != len(sources):
            raise ContractError("semanticSources must not repeat an agent")
        state = _bounded_control_value(value.get("state"), "state")
        update = _bounded_control_value(value.get("update"), "update")
        turn_taking = _bounded_control_value(value.get("turnTaking"), "turnTaking")
        if not isinstance(state, dict) or not isinstance(update, dict) or not isinstance(turn_taking, dict):
            raise ContractError("state, update, and turnTaking must be objects")
        if update.get("applyAt") != "next_agent_turn_boundary":
            raise ContractError("control frame must apply at the next agent turn boundary")
        if not isinstance(update.get("expiresAtMs"), int) or update["expiresAtMs"] != plan.expiry_ms:
            raise ContractError("control-frame expiry must match the typed plan")
        return cls(
            schema_version=1,
            frame_id=_required_string(value, "frameId"),
            conversation_id=_required_string(value, "conversationId"),
            target_turn_id=target_turn_id,
            state_revision=state_revision,
            base_state_hash=base_state_hash,
            state_hash=state_hash,
            semantic_sources=tuple(sources),
            state=state,
            update=update,
            turn_taking=turn_taking,
            plan=plan,
        )

    def as_wire_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "frameId": self.frame_id,
            "conversationId": self.conversation_id,
            "targetTurnId": self.target_turn_id,
            "stateRevision": self.state_revision,
            "baseStateHash": self.base_state_hash,
            "stateHash": self.state_hash,
            "semanticSources": list(self.semantic_sources),
            "state": self.state,
            "update": self.update,
            "turnTaking": self.turn_taking,
            "plan": self.plan.as_wire_dict(),
        }

    @property
    def frame_hash(self) -> str:
        return sha256_uri(self.as_wire_dict())


def validate_control_frame_mapping(value: Mapping[str, Any]) -> ControlTrainingFrame:
    return ControlTrainingFrame.from_mapping(value)


_EVIDENCE_AVAILABILITY = {"ready", "failed", "expired"}


@dataclass(frozen=True)
class EvidenceTrainingFrame:
    """Causal, target-wording-free delayed evidence aligned to one agent turn.

    The control plan remains the authority for behavior.  This frame represents
    a late tool, policy, retrieval, or ASR-derived fact that was available
    before the target agent turn.  It is intentionally unsuitable for carrying
    a canonical reply: the trainable evidence adapter only receives bounded
    facts, provenance, and timing, while target audio/code tokens remain labels.
    """

    schema_version: int
    evidence_id: str
    conversation_id: str
    target_turn_id: int
    evidence_revision: int
    supports_control_revision: int
    context_hash: str
    post_evidence_state_hash: str
    availability: str
    provenance: dict[str, Any]
    allowed_claims: tuple[str, ...]
    timing: dict[str, Any]
    counterfactual: dict[str, Any]
    plan: ControlPlan

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvidenceTrainingFrame":
        normalized = _bounded_control_value(value, "evidence")
        if not isinstance(normalized, dict):  # defensive; the helper already guarantees this
            raise ContractError("evidence frame must be an object")
        if normalized.get("schemaVersion", normalized.get("schema_version")) != 1:
            raise ContractError("only evidence training-frame schema version 1 is supported")
        plan_value = normalized.get("plan")
        if not isinstance(plan_value, Mapping):
            raise ContractError("evidence frame requires an inline typed plan")
        plan = ControlPlan.from_mapping(plan_value)
        target_turn_id = normalized.get("targetTurnId")
        evidence_revision = normalized.get("evidenceRevision")
        supports_control_revision = normalized.get("supportsControlRevision")
        if not isinstance(target_turn_id, int) or target_turn_id < 0:
            raise ContractError("evidence targetTurnId must be a non-negative integer")
        if target_turn_id != plan.turn_id:
            raise ContractError("evidence targetTurnId must equal plan.turnId")
        if not isinstance(supports_control_revision, int) or supports_control_revision < 0:
            raise ContractError("supportsControlRevision must be a non-negative integer")
        if not isinstance(evidence_revision, int) or evidence_revision <= supports_control_revision:
            raise ContractError("evidenceRevision must be greater than the supporting control revision")
        if plan.revision <= evidence_revision:
            raise ContractError("inline target plan must be a later revision than delayed evidence")
        context_hash = _required_string(normalized, "contextHash")
        post_evidence_state_hash = _required_string(normalized, "postEvidenceStateHash")
        if not _HASH_RE.match(context_hash) or not _HASH_RE.match(post_evidence_state_hash):
            raise ContractError("evidence state hashes must be sha256 URIs")
        if post_evidence_state_hash != plan.context_hash:
            raise ContractError("postEvidenceStateHash must equal the inline target plan contextHash")
        availability = _required_string(normalized, "availability")
        if availability not in _EVIDENCE_AVAILABILITY:
            raise ContractError(f"evidence availability must be one of {sorted(_EVIDENCE_AVAILABILITY)}")
        provenance = normalized.get("provenance")
        timing = normalized.get("timing")
        counterfactual = normalized.get("counterfactual")
        if not isinstance(provenance, dict) or not provenance:
            raise ContractError("evidence provenance must be a non-empty object")
        if not isinstance(timing, dict) or not isinstance(counterfactual, dict):
            raise ContractError("evidence timing and counterfactual must be objects")
        if timing.get("applyAt") != "next_agent_turn_boundary":
            raise ContractError("evidence must apply at the next agent turn boundary")
        available_at = timing.get("evidenceAvailableAtMs")
        agent_start = timing.get("agentStartAtMs")
        if not isinstance(available_at, int) or not isinstance(agent_start, int) or available_at < 0 or agent_start < available_at:
            raise ContractError("evidence timing must place availability before its agent target")
        if not isinstance(counterfactual.get("groupId"), str) or not counterfactual["groupId"]:
            raise ContractError("evidence counterfactual.groupId must be non-empty")
        if not isinstance(counterfactual.get("branchId"), str) or not counterfactual["branchId"]:
            raise ContractError("evidence counterfactual.branchId must be non-empty")
        if not isinstance(counterfactual.get("changedField"), str) or not counterfactual["changedField"]:
            raise ContractError("evidence counterfactual.changedField must be non-empty")
        allowed_claims = _string_list(normalized, "allowedClaims")
        if availability == "ready" and not allowed_claims:
            raise ContractError("ready evidence requires at least one allowed claim")
        return cls(
            schema_version=1,
            evidence_id=_required_string(normalized, "evidenceId"),
            conversation_id=_required_string(normalized, "conversationId"),
            target_turn_id=target_turn_id,
            evidence_revision=evidence_revision,
            supports_control_revision=supports_control_revision,
            context_hash=context_hash,
            post_evidence_state_hash=post_evidence_state_hash,
            availability=availability,
            provenance=provenance,
            allowed_claims=allowed_claims,
            timing=timing,
            counterfactual=counterfactual,
            plan=plan,
        )

    def as_wire_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "evidenceId": self.evidence_id,
            "conversationId": self.conversation_id,
            "targetTurnId": self.target_turn_id,
            "evidenceRevision": self.evidence_revision,
            "supportsControlRevision": self.supports_control_revision,
            "contextHash": self.context_hash,
            "postEvidenceStateHash": self.post_evidence_state_hash,
            "availability": self.availability,
            "provenance": self.provenance,
            "allowedClaims": list(self.allowed_claims),
            "timing": self.timing,
            "counterfactual": self.counterfactual,
            "plan": self.plan.as_wire_dict(),
        }

    @property
    def evidence_hash(self) -> str:
        return sha256_uri(self.as_wire_dict())


def validate_evidence_frame_mapping(value: Mapping[str, Any]) -> EvidenceTrainingFrame:
    return EvidenceTrainingFrame.from_mapping(value)


def assert_evidence_control_alignment(control: ControlTrainingFrame, evidence: EvidenceTrainingFrame) -> None:
    """Reject cross-turn, cross-context, or unapproved delayed-evidence joins."""
    if control.conversation_id != evidence.conversation_id:
        raise ContractError("evidence and control frame belong to different conversations")
    if control.target_turn_id != evidence.target_turn_id:
        raise ContractError("evidence and control frame target different turns")
    if control.plan.revision <= evidence.evidence_revision:
        raise ContractError("target control revision does not follow delayed evidence")
    if control.state_hash != evidence.post_evidence_state_hash:
        raise ContractError("evidence post-update state differs from the active control state")
