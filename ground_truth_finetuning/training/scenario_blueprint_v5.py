"""Joint scenario-blueprint and bound expansion pipeline for PersonaPlex v5.

Stage P creates all twenty semantic niches for one topic in one constrained
model call. Stage E expands one immutable niche per constrained model call.
Host code assigns identities, validates structural contracts, and persists
content-addressed checkpoints; it never synthesizes or repairs semantic fields.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import BoundedSemaphore, Lock
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
import tempfile
import urllib.error
import urllib.request

from jsonschema import Draft202012Validator

from ground_truth_finetuning.training.strict_schema_transport import (
    DEFAULT_RETRY_BASE_SECONDS,
    DEFAULT_RETRY_MAX_SECONDS,
    DEFAULT_TRANSPORT_ATTEMPTS,
    RETRIABLE_HTTP_STATUS,
    append_canonical_schema_contract,
    build_schema_transport_projection,
    canonical_retry_context,
    canonical_validation_defect,
    classify_provider_error,
    is_openrouter_endpoint,
    normalize_chat_completion_endpoints,
    retry_after_seconds,
    retry_delay_seconds,
    schema_capability_profile,
    validate_retry_settings,
)


TOPICS_PER_CORPUS = 50
BLUEPRINTS_PER_TOPIC = 20
MAX_WORKERS = 3
BLUEPRINT_INITIAL_OUTPUT_TOKENS = 4096
BLUEPRINT_MAX_OUTPUT_TOKENS = 12288
SCENARIO_MAX_OUTPUT_TOKENS = 4096
BLUEPRINT_JUDGE_MAX_OUTPUT_TOKENS = 2048
BLUEPRINT_CLAIM_VERIFIER_MAX_OUTPUT_TOKENS = 64
TAXONOMY_MAX_OUTPUT_TOKENS = 8192
MAX_LIVE_OUTPUT_TOKENS = max(TAXONOMY_MAX_OUTPUT_TOKENS, BLUEPRINT_MAX_OUTPUT_TOKENS)
BLUEPRINT_PROTOCOL_VERSION = (
    "joint-blueprint-adaptive-feedback-v7-natural-boundaries-adaptive-token-budget"
)
TAXONOMY_PROTOCOL_VERSION = "twenty-branch-anchor-taxonomy-v5-semantic-boundaries"
BLUEPRINT_JUDGE_PROTOCOL_VERSION = (
    "typed-whole-blueprint-findings-v5-model-identity-adjudication"
)
SCENARIO_EXPANSION_PROTOCOL_VERSION = "host-materialized-control-routes-v3"
TYPED_DIVERGENCE_DENOMINATOR = 5
BLUEPRINT_MAX_PROPOSER_CLAIMS = BLUEPRINTS_PER_TOPIC
BLUEPRINT_MAX_FINAL_FINDINGS = BLUEPRINTS_PER_TOPIC * 2

BLUEPRINT_SET_SCHEMA = "personaplex.scenario-blueprint-set.v5"
TAXONOMY_CHECKPOINT_SCHEMA = "personaplex.scenario-taxonomy-checkpoint.v9"
BLUEPRINT_CHECKPOINT_SCHEMA = "personaplex.scenario-blueprint-checkpoint.v6"
EXPANSION_CHECKPOINT_SCHEMA = "personaplex.scenario-expansion-checkpoint.v5"
BLUEPRINT_SCRUTINY_SCHEMA = "personaplex.scenario-blueprint-scrutiny.v5"
BLUEPRINT_CLAIM_VERIFICATION_CHECKPOINT_SCHEMA = (
    "personaplex.scenario-blueprint-claim-verification.v1"
)
BLUEPRINT_BINDING_SCHEMA = "personaplex.scenario-blueprint-binding.v5"
BLUEPRINT_BINDING_MANIFEST_SCHEMA = "personaplex.scenario-blueprint-binding-manifest.v5"
BLUEPRINT_SETS_FILENAME = "scenario_blueprint_sets.jsonl"
BLUEPRINT_SCRUTINY_FILENAME = "scenario_blueprint_scrutiny.jsonl"
SCENARIO_CONTRACTS_FILENAME = "scenario_contracts.jsonl"
SCENARIO_BINDINGS_FILENAME = "scenario_blueprint_bindings.jsonl"
SCENARIO_BINDINGS_MANIFEST_FILENAME = "scenario_blueprint_bindings.manifest.json"
CHECKPOINT_ROOT_NAME = ".scenario_blueprint_v5"

BLUEPRINT_JUDGE_DIMENSIONS = (
    "semanticDiversity",
    "causalMechanismDiversity",
    "topicModeCoverage",
    "fourSiblingAffordance",
    "fieldRoleCoherence",
)

BLUEPRINT_FINDING_CODES = (
    "semantic_near_duplicate_cluster",
    "causal_mechanism_collapse",
    "topic_mode_collapse",
    "insufficient_four_sibling_affordance",
    "incoherent_niche",
    "field_role_collapse",
    "target_field_leakage",
)

BLUEPRINT_FINDING_DIMENSIONS = {
    "semantic_near_duplicate_cluster": ("semanticDiversity",),
    "causal_mechanism_collapse": ("causalMechanismDiversity",),
    "topic_mode_collapse": ("topicModeCoverage",),
    "insufficient_four_sibling_affordance": ("fourSiblingAffordance",),
    "incoherent_niche": ("semanticDiversity", "fourSiblingAffordance"),
    "field_role_collapse": ("fieldRoleCoherence",),
    "target_field_leakage": BLUEPRINT_JUDGE_DIMENSIONS,
}

BLUEPRINT_FINDING_CARDINALITIES = {
    "semantic_near_duplicate_cluster": {"minimum": 2, "maximum": BLUEPRINTS_PER_TOPIC},
    "causal_mechanism_collapse": {"minimum": 2, "maximum": BLUEPRINTS_PER_TOPIC},
    "topic_mode_collapse": {"minimum": 2, "maximum": BLUEPRINTS_PER_TOPIC},
    "insufficient_four_sibling_affordance": {"minimum": 1, "maximum": 1},
    "incoherent_niche": {"minimum": 1, "maximum": 1},
    "field_role_collapse": {"minimum": 1, "maximum": 1},
    "target_field_leakage": {"minimum": 1, "maximum": 1},
}

BLUEPRINT_FINDING_DEFINITIONS = {
    "semantic_near_duplicate_cluster": (
        "The implicated niches are materially interchangeable across their complete semantic "
        "signatures; shared vocabulary or structure alone is insufficient."
    ),
    "causal_mechanism_collapse": (
        "The implicated niches collapse onto the same causal change pattern without materially "
        "distinct resources, tensions, evidence pivots, stakes, or outcomes."
    ),
    "topic_mode_collapse": (
        "The implicated niches collectively fail to provide materially distinct interaction "
        "submodes for the topic; surface wording differences do not establish coverage."
    ),
    "insufficient_four_sibling_affordance": (
        "The niche cannot support all four causally distinct verified-positive, "
        "verified-negative, uncertain, and superseded next-state routes."
    ),
    "incoherent_niche": (
        "The niche fields do not compose into one plausible topic-bound situation with a usable "
        "causal path to distinct outcomes."
    ),
    "field_role_collapse": (
        "At least one field materially fails its declared role, such as a non-resource resource, "
        "non-event evidence pivot, or relationship-shaped mechanism."
    ),
    "target_field_leakage": (
        "The niche contains dialogue, response wording, target text or audio, or another "
        "prohibited target-bearing field rather than declarative semantics."
    ),
}

BLUEPRINT_PROPOSER_SYSTEM = """You are one independent whole-blueprint semantic finding
proposer. Reasoning mode is disabled. Audit the complete bound twenty-niche blueprint and return
only exact typed claims with code and scenarioIds from the strict schema. Return no rationale,
status, score, dimension verdict, repair, dialogue, target response, or unbound ID. Shared words,
structure, or transferable control operators alone are not defects. An empty findings array is the
sole no-claim response."""

BLUEPRINT_CLAIM_VERIFIER_SYSTEM = """You are the final independent verifier for exactly one
typed whole-blueprint semantic finding. Reasoning mode is disabled. The proposed claim is untrusted
and has no proposer rationale. Evaluate only its exact code and exact source-bound scenario IDs
against the supplied topic and evidence-local blueprints. Return confirmed=true only when that
evidence directly satisfies the supplied fixed code definition. Do not perform an open audit,
substitute another finding, inspect unbound siblings, repair content, use lexical matching, write
dialogue, or infer missing facts."""

NICHE_SIGNATURE_FIELDS = (
    "participantRelationship",
    "setting",
    "interactionMode",
    "submode",
    "centralResource",
    "centralTension",
    "evidencePivot",
    "controlOperator",
    "causalMechanism",
    "stakesProfile",
    "outcomeTopology",
    "fourSiblingAffordance",
    "duplexOpportunity",
)

NICHE_REQUIRED_FIELDS = NICHE_SIGNATURE_FIELDS + ("semanticDistinctnessFrom",)
TAXONOMY_FIELDS = (
    "submode",
    "participantRelationship",
    "setting",
    "centralResource",
    "centralTension",
)
TAXONOMY_WIRE_KEYS = {
    field: field for field in TAXONOMY_FIELDS
}
EXACT_UNIQUE_NICHE_FIELDS = (
    "submode",
)
EXACT_UNIQUE_NICHE_FIELD_GROUPS = (
    ("centralResource", "centralTension"),
    ("evidencePivot", "causalMechanism"),
    ("centralResource", "fourSiblingAffordance"),
)

# Stage P must fit twenty complete niches inside the live 4096-token response
# ceiling. Repeating the descriptive field names twenty times consumed a large
# fraction of that budget, so the model-facing schema uses compact aliases.
# These aliases are a transport encoding only: checkpoints and all downstream
# consumers receive the losslessly decoded canonical field names.
NICHE_WIRE_KEYS = {
    "participantRelationship": "r",
    "setting": "s",
    "interactionMode": "m",
    "submode": "u",
    "centralResource": "c",
    "centralTension": "t",
    "evidencePivot": "e",
    "controlOperator": "q",
    "causalMechanism": "k",
    "stakesProfile": "p",
    "outcomeTopology": "o",
    "fourSiblingAffordance": "f",
    "duplexOpportunity": "x",
    "semanticDistinctnessFrom": "d",
}
DISTINCTNESS_WIRE_KEYS = {"siblingIds": "i", "distinction": "v"}
FOUR_SIBLING_STATES = (
    "verifiedPositive",
    "verifiedNegative",
    "uncertain",
    "superseded",
)
AFFORDANCE_WIRE_KEYS = {
    "verifiedPositive": "p",
    "verifiedNegative": "n",
    "uncertain": "u",
    "superseded": "s",
}

OUTCOME_TOPOLOGIES = (
    "evidence_confirmed",
    "evidence_disconfirmed",
    "uncertain_pending",
    "superseded_redirect",
    "constraint_limited",
    "mutual_repair",
)

DUPLEX_OPPORTUNITIES = (
    "barge_in_repair",
    "brief_overlap",
    "backchannel_then_resume",
    "cancel_and_restart",
    "completed_turn_handoff",
    "clarification_pause",
)

# These are the mutable semantic-plane causes the adapter must learn to realize
# differently from otherwise similar duplex context. They intentionally recur
# across topics; the free causalMechanism field supplies topic-specific form.
CONTROL_OPERATORS = (
    "evidence_status_revision",
    "tool_result_revision",
    "policy_constraint_revision",
    "caller_posture_revision",
    "commitment_state_revision",
    "interruption_cancellation_revision",
    "superseding_context_revision",
)
CONTROL_OPERATOR_WIRE_CODES = {
    operator: chr(ord("a") + index) for index, operator in enumerate(CONTROL_OPERATORS)
}
CONTROL_OPERATOR_BY_WIRE_CODE = {
    code: operator for operator, code in CONTROL_OPERATOR_WIRE_CODES.items()
}

DEFAULT_STAKES_PROFILES = (
    "low",
    "moderate",
    "time_sensitive_low_risk",
    "relationship_sensitive",
    "resource_constrained",
)

FORBIDDEN_TARGET_FIELD_NAMES = frozenset(
    {
        "agentresponse",
        "canonicalresponse",
        "dialogue",
        "expectedresponse",
        "reply",
        "script",
        "spokentext",
        "targetaudio",
        "targethash",
        "targethashes",
        "targettext",
        "targettranscript",
        "utterance",
    }
)


class ScenarioBlueprintError(RuntimeError):
    """Base error for the v5 scenario-blueprint pipeline."""


class ModelTransportUnavailable(ScenarioBlueprintError):
    """Every configured physical model route was temporarily unavailable."""


class InvalidModelOutput(ScenarioBlueprintError):
    """A model response failed its assigned structural contract."""


class TruncatedModelOutput(InvalidModelOutput):
    """The endpoint reported finish_reason=length."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def content_hash(value: Any) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


BLUEPRINT_PROPOSER_PROMPT_HASH = content_hash(BLUEPRINT_PROPOSER_SYSTEM)
BLUEPRINT_VERIFIER_PROMPT_HASH = content_hash(BLUEPRINT_CLAIM_VERIFIER_SYSTEM)
BLUEPRINT_FINDING_CARDINALITY_HASH = content_hash(BLUEPRINT_FINDING_CARDINALITIES)
BLUEPRINT_JUDGE_SOURCE_HASH = content_hash(
    {
        "proposerSystem": BLUEPRINT_PROPOSER_SYSTEM,
        "claimVerifierSystem": BLUEPRINT_CLAIM_VERIFIER_SYSTEM,
        "findingDefinitions": BLUEPRINT_FINDING_DEFINITIONS,
        "findingCardinalities": BLUEPRINT_FINDING_CARDINALITIES,
        "evidencePolicy": "topic-card-plus-exact-claimed-blueprints-only",
        "finalRationalePolicy": "fixed-definition-for-verifier-confirmed-code-only",
    }
)


def blueprint_judge_protocol_hash() -> str:
    return content_hash(
        {
            "protocolVersion": BLUEPRINT_JUDGE_PROTOCOL_VERSION,
            "sourceHash": BLUEPRINT_JUDGE_SOURCE_HASH,
            "proposerPromptHash": BLUEPRINT_PROPOSER_PROMPT_HASH,
            "verifierPromptHash": BLUEPRINT_VERIFIER_PROMPT_HASH,
            "cardinalityHash": BLUEPRINT_FINDING_CARDINALITY_HASH,
            "verificationCheckpointSchema": (
                BLUEPRINT_CLAIM_VERIFICATION_CHECKPOINT_SCHEMA
            ),
        }
    )


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScenarioBlueprintError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ScenarioBlueprintError(f"{path} must contain one JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ScenarioBlueprintError(f"cannot read JSONL {path}: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ScenarioBlueprintError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ScenarioBlueprintError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _identifier(value: Any, label: str, maximum: int = 160) -> str:
    if not isinstance(value, str) or not 3 <= len(value) <= maximum:
        raise ScenarioBlueprintError(f"{label} must be a bounded lowercase identifier")
    allowed = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_:-")
    if value[0] not in "abcdefghijklmnopqrstuvwxyz" or any(character not in allowed for character in value):
        raise ScenarioBlueprintError(f"{label} must be a bounded lowercase identifier")
    return value


def _unique_nonempty_strings(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(set(value)) != len(value)
    ):
        raise ScenarioBlueprintError(f"{label} must be a unique nonempty string array")
    return value


def _normalized_field_name(value: Any) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def assert_no_target_fields(value: Any, path: str = "root") -> None:
    """Reject target-bearing field names without inspecting semantic text."""

    if isinstance(value, dict):
        for key, child in value.items():
            if _normalized_field_name(key) in FORBIDDEN_TARGET_FIELD_NAMES:
                raise InvalidModelOutput(f"target field {key!r} is forbidden at {path}")
            assert_no_target_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_target_fields(child, f"{path}[{index}]")


def scenario_ids_for_topic(topic_id: str) -> tuple[str, ...]:
    _identifier(topic_id, "topicId", 120)
    result = tuple(
        f"scenario_{topic_id}_{ordinal:02d}"
        for ordinal in range(1, BLUEPRINTS_PER_TOPIC + 1)
    )
    for scenario_id in result:
        _identifier(scenario_id, "scenarioId")
    return result


def _topic_enums(topic: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    interaction_modes = _unique_nonempty_strings(
        topic.get("interactionModes"), "topic.interactionModes"
    )
    safe_stakes_value = topic.get("safeStakes")
    safe_stakes = (
        _unique_nonempty_strings(safe_stakes_value, "topic.safeStakes")
        if safe_stakes_value is not None
        else list(DEFAULT_STAKES_PROFILES)
    )
    return interaction_modes, safe_stakes


def _semantic_text(minimum: int = 3) -> dict[str, Any]:
    return {"type": "string", "minLength": minimum}


def _compact_text(_maximum: int) -> dict[str, Any]:
    """Compatibility wrapper: semantic prose is token-bounded, never character-clipped."""

    return _semantic_text()


def build_taxonomy_response_schema(
    topic: Mapping[str, Any], scenario_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """Build the canonical twenty-branch taxonomy schema used before Stage P."""

    topic_id = _identifier(topic.get("topicId"), "topic.topicId", 120)
    expected_ids = tuple(scenario_ids or scenario_ids_for_topic(topic_id))
    if len(expected_ids) != BLUEPRINTS_PER_TOPIC or len(set(expected_ids)) != len(expected_ids):
        raise ScenarioBlueprintError("taxonomy requires exactly twenty unique scenario IDs")
    properties = {
        scenario_id: {
            "type": "object",
            "additionalProperties": False,
            "required": [TAXONOMY_WIRE_KEYS[field] for field in TAXONOMY_FIELDS],
            "properties": {
                TAXONOMY_WIRE_KEYS[field]: _semantic_text()
                for field in TAXONOMY_FIELDS
            },
        }
        for scenario_id in expected_ids
    }
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(expected_ids),
        "properties": properties,
    }
    Draft202012Validator.check_schema(schema)
    if "prefixItems" in canonical_json(schema):
        raise ScenarioBlueprintError("taxonomy schema must never use prefixItems")
    return schema


def validate_taxonomy_anchors(
    value: Any, topic: Mapping[str, Any], scenario_ids: Sequence[str] | None = None
) -> dict[str, dict[str, str]]:
    """Validate canonical taxonomy anchors without semantic host synthesis."""

    topic_id = _identifier(topic.get("topicId"), "topic.topicId", 120)
    expected_ids = tuple(scenario_ids or scenario_ids_for_topic(topic_id))
    if not isinstance(value, dict) or set(value) != set(expected_ids):
        raise InvalidModelOutput("canonical taxonomy must bind the exact twenty scenario IDs")
    submode_owners: dict[str, str] = {}
    resource_owners: dict[str, str] = {}
    anchor_owners: dict[str, str] = {}
    forbidden_resources = {
        item
        for item in [topic.get("domain"), *(topic.get("interactionModes") or [])]
        if isinstance(item, str) and item
    }
    for scenario_id in expected_ids:
        anchor = value[scenario_id]
        if not isinstance(anchor, dict) or set(anchor) != set(TAXONOMY_FIELDS):
            raise InvalidModelOutput(f"canonical taxonomy anchor is malformed for {scenario_id}")
        for field in TAXONOMY_FIELDS:
            field_value = anchor[field]
            if (
                not isinstance(field_value, str)
                or len(field_value) < 3
            ):
                raise InvalidModelOutput(
                    f"canonical taxonomy field {field} is invalid for {scenario_id}"
                )
        if len(set(anchor.values())) != len(TAXONOMY_FIELDS):
            raise InvalidModelOutput(
                f"taxonomy fields collapse distinct roles inside {scenario_id}"
            )
        submode = anchor["submode"]
        if submode in submode_owners:
            raise InvalidModelOutput(
                f"taxonomy submode repeats exactly for {submode_owners[submode]} and {scenario_id}"
            )
        submode_owners[submode] = scenario_id
        resource = anchor["centralResource"]
        if resource in forbidden_resources:
            raise InvalidModelOutput(
                f"taxonomy centralResource repeats a topic or interaction-mode label for {scenario_id}"
            )
        if resource in resource_owners:
            raise InvalidModelOutput(
                "taxonomy centralResource repeats exactly for "
                f"{resource_owners[resource]} and {scenario_id}"
            )
        resource_owners[resource] = scenario_id
        signature = canonical_json([anchor[field] for field in TAXONOMY_FIELDS])
        if signature in anchor_owners:
            raise InvalidModelOutput(
                f"taxonomy anchor repeats exactly for {anchor_owners[signature]} and {scenario_id}"
            )
        anchor_owners[signature] = scenario_id
    return value


def decode_taxonomy_response(
    value: Any, topic: Mapping[str, Any], scenario_ids: Sequence[str] | None = None
) -> dict[str, dict[str, str]]:
    topic_id = _identifier(topic.get("topicId"), "topic.topicId", 120)
    expected_ids = tuple(scenario_ids or scenario_ids_for_topic(topic_id))
    _raise_schema_errors(
        value, build_taxonomy_response_schema(topic, expected_ids), "taxonomy wire response"
    )
    assert isinstance(value, dict)
    decoded = {
        scenario_id: {
            field: value[scenario_id][TAXONOMY_WIRE_KEYS[field]]
            for field in TAXONOMY_FIELDS
        }
        for scenario_id in expected_ids
    }
    return validate_taxonomy_anchors(decoded, topic, expected_ids)


def _taxonomy_structural_revision_contract(
    value: Any,
    topic: Mapping[str, Any],
    scenario_ids: Sequence[str],
) -> dict[str, Any]:
    """Describe exact structural conflicts in a rejected model response."""

    if not isinstance(value, dict) or set(value) != set(scenario_ids):
        return {}
    forbidden: dict[str, dict[str, list[str]]] = {}

    def forbid(scenario_id: str, field: str, field_value: Any) -> None:
        if not isinstance(field_value, str) or not field_value:
            return
        fields = forbidden.setdefault(scenario_id, {})
        values = fields.setdefault(field, [])
        if field_value not in values:
            values.append(field_value)

    for field in ("submode", "centralResource"):
        owner_by_value: dict[str, str] = {}
        for scenario_id in scenario_ids:
            anchor = value.get(scenario_id)
            if not isinstance(anchor, dict):
                continue
            field_value = anchor.get(TAXONOMY_WIRE_KEYS[field])
            if not isinstance(field_value, str):
                continue
            if field_value in owner_by_value:
                forbid(scenario_id, field, field_value)
            else:
                owner_by_value[field_value] = scenario_id

    forbidden_resources = {
        item
        for item in [topic.get("domain"), *(topic.get("interactionModes") or [])]
        if isinstance(item, str) and item
    }
    for scenario_id in scenario_ids:
        anchor = value.get(scenario_id)
        if not isinstance(anchor, dict):
            continue
        canonical_values = {
            field: anchor.get(TAXONOMY_WIRE_KEYS[field]) for field in TAXONOMY_FIELDS
        }
        resource = canonical_values["centralResource"]
        if resource in forbidden_resources:
            forbid(scenario_id, "centralResource", resource)
        owners: dict[str, str] = {}
        for field in TAXONOMY_FIELDS:
            field_value = canonical_values[field]
            if not isinstance(field_value, str):
                continue
            if field_value in owners:
                forbid(scenario_id, field, field_value)
            else:
                owners[field_value] = field

    return {
        "mustReturnAllScenarioIds": True,
        "identicalResponseForbidden": True,
        "forbiddenExactValuesByScenarioId": forbidden,
    }


def _coverage_assignment(
    ordinal: int, interaction_modes: Sequence[str], safe_stakes: Sequence[str]
) -> dict[str, str]:
    """Return the deterministic typed-coverage cell for one zero-based slot."""

    return {
        "interactionMode": interaction_modes[ordinal % len(interaction_modes)],
        "stakesProfile": safe_stakes[ordinal % len(safe_stakes)],
        "outcomeTopology": OUTCOME_TOPOLOGIES[ordinal % len(OUTCOME_TOPOLOGIES)],
        "duplexOpportunity": DUPLEX_OPPORTUNITIES[
            (ordinal * (len(DUPLEX_OPPORTUNITIES) - 1)) % len(DUPLEX_OPPORTUNITIES)
        ],
        "controlOperator": CONTROL_OPERATORS[ordinal % len(CONTROL_OPERATORS)],
    }


def _blueprint_response_schema(
    topic: Mapping[str, Any],
    scenario_ids: Sequence[str] | None,
    *,
    compact_wire: bool,
    taxonomy_anchors: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    topic_id = _identifier(topic.get("topicId"), "topic.topicId", 120)
    expected_ids = tuple(scenario_ids or scenario_ids_for_topic(topic_id))
    if len(expected_ids) != BLUEPRINTS_PER_TOPIC or len(set(expected_ids)) != len(expected_ids):
        raise ScenarioBlueprintError("Stage P requires exactly twenty unique scenario IDs")
    for scenario_id in expected_ids:
        _identifier(scenario_id, "scenarioId")
    interaction_modes, safe_stakes = _topic_enums(topic)
    bound_taxonomy = (
        validate_taxonomy_anchors(dict(taxonomy_anchors), topic, expected_ids)
        if taxonomy_anchors is not None
        else None
    )

    properties: dict[str, Any] = {}
    for ordinal, scenario_id in enumerate(expected_ids):
        sibling_ids = [candidate for candidate in expected_ids if candidate != scenario_id]
        coverage = _coverage_assignment(ordinal, interaction_modes, safe_stakes)
        anchor = bound_taxonomy[scenario_id] if bound_taxonomy is not None else None
        canonical_properties = {
            "participantRelationship": (
                {"const": anchor["participantRelationship"]}
                if anchor
                else _semantic_text()
            ),
            "setting": (
                {"const": anchor["setting"]}
                if anchor
                else _semantic_text()
            ),
            "interactionMode": {"const": coverage["interactionMode"]},
            "submode": (
                {"const": anchor["submode"]}
                if anchor
                else _semantic_text()
            ),
            "centralResource": (
                {"const": anchor["centralResource"]}
                if anchor
                else _semantic_text()
            ),
            "centralTension": (
                {"const": anchor["centralTension"]}
                if anchor
                else _semantic_text()
            ),
            "evidencePivot": _compact_text(36),
            "controlOperator": {"const": coverage["controlOperator"]},
            "causalMechanism": _compact_text(36),
            "stakesProfile": {"const": coverage["stakesProfile"]},
            "outcomeTopology": {"const": coverage["outcomeTopology"]},
            "fourSiblingAffordance": {
                "type": "object",
                "additionalProperties": False,
                "required": list(FOUR_SIBLING_STATES),
                "properties": {
                    state: _compact_text(32) for state in FOUR_SIBLING_STATES
                },
            },
            "duplexOpportunity": {"const": coverage["duplexOpportunity"]},
            "semanticDistinctnessFrom": {
                "type": "object",
                "additionalProperties": False,
                "required": ["siblingIds", "distinction"],
                "properties": {
                    "siblingIds": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 2,
                        "uniqueItems": True,
                        "items": {"type": "string", "enum": sibling_ids},
                    },
                    "distinction": _compact_text(36),
                },
            },
        }
        if compact_wire:
            wire_properties = {
                NICHE_WIRE_KEYS[field]: field_schema
                for field, field_schema in canonical_properties.items()
                if field != "semanticDistinctnessFrom"
                and not (bound_taxonomy is not None and field in TAXONOMY_FIELDS)
            }
            wire_properties[NICHE_WIRE_KEYS["controlOperator"]] = {
                "const": CONTROL_OPERATOR_WIRE_CODES[coverage["controlOperator"]]
            }
            affordance = canonical_properties["fourSiblingAffordance"]
            wire_properties[NICHE_WIRE_KEYS["fourSiblingAffordance"]] = {
                **affordance,
                "required": list(AFFORDANCE_WIRE_KEYS.values()),
                "properties": {
                    AFFORDANCE_WIRE_KEYS[state]: affordance["properties"][state]
                    for state in FOUR_SIBLING_STATES
                },
            }
            distinctness = canonical_properties["semanticDistinctnessFrom"]
            wire_properties[NICHE_WIRE_KEYS["semanticDistinctnessFrom"]] = {
                **distinctness,
                "required": list(DISTINCTNESS_WIRE_KEYS.values()),
                "properties": {
                    DISTINCTNESS_WIRE_KEYS[field]: field_schema
                    for field, field_schema in distinctness["properties"].items()
                },
            }
            required = [
                NICHE_WIRE_KEYS[field]
                for field in NICHE_REQUIRED_FIELDS
                if not (bound_taxonomy is not None and field in TAXONOMY_FIELDS)
            ]
            niche_properties = wire_properties
        else:
            required = list(NICHE_REQUIRED_FIELDS)
            niche_properties = canonical_properties
        properties[scenario_id] = {
            "type": "object",
            "additionalProperties": False,
            "required": required,
            "properties": niche_properties,
        }
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(expected_ids),
        "properties": properties,
    }
    Draft202012Validator.check_schema(schema)
    if "prefixItems" in canonical_json(schema):
        raise ScenarioBlueprintError("Stage P schema must never use prefixItems")
    return schema


def build_blueprint_response_schema(
    topic: Mapping[str, Any],
    scenario_ids: Sequence[str] | None = None,
    taxonomy_anchors: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Build the compact strict Stage P wire schema with twenty ID-bound properties."""

    return _blueprint_response_schema(
        topic, scenario_ids, compact_wire=True, taxonomy_anchors=taxonomy_anchors
    )


def _canonical_blueprint_response_schema(
    topic: Mapping[str, Any], scenario_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    return _blueprint_response_schema(topic, scenario_ids, compact_wire=False)


def typed_niche_signature(niche: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact typed semantic signature, excluding comparison prose."""

    return tuple(canonical_json(niche.get(field)) for field in NICHE_SIGNATURE_FIELDS)


def _raise_schema_errors(value: Any, schema: Mapping[str, Any], label: str) -> None:
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.absolute_path) or "root"
        raise InvalidModelOutput(f"{label} violates strict schema at {location}: {first.message}")


def validate_blueprint_response(
    value: Any, topic: Mapping[str, Any], scenario_ids: Sequence[str] | None = None
) -> dict[str, dict[str, Any]]:
    topic_id = _identifier(topic.get("topicId"), "topic.topicId", 120)
    expected_ids = tuple(scenario_ids or scenario_ids_for_topic(topic_id))
    schema = _canonical_blueprint_response_schema(topic, expected_ids)
    _raise_schema_errors(value, schema, "canonical Stage P response")
    assert_no_target_fields(value)
    assert isinstance(value, dict)
    signatures: dict[tuple[str, ...], str] = {}
    signature_by_id: dict[str, tuple[str, ...]] = {}
    for scenario_id in expected_ids:
        niche = value[scenario_id]
        signature = typed_niche_signature(niche)
        if signature in signatures:
            raise InvalidModelOutput(
                f"typed niche signature repeats for {signatures[signature]} and {scenario_id}"
            )
        signatures[signature] = scenario_id
        signature_by_id[scenario_id] = signature
    minimum_differences = (
        len(NICHE_SIGNATURE_FIELDS) + TYPED_DIVERGENCE_DENOMINATOR - 1
    ) // TYPED_DIVERGENCE_DENOMINATOR
    for left_index, left_id in enumerate(expected_ids):
        for right_id in expected_ids[left_index + 1 :]:
            differences = sum(
                left_value != right_value
                for left_value, right_value in zip(
                    signature_by_id[left_id], signature_by_id[right_id], strict=True
                )
            )
            if differences < minimum_differences:
                raise InvalidModelOutput(
                    f"typed niche pair {left_id} and {right_id} differs on only "
                    f"{differences}/{len(NICHE_SIGNATURE_FIELDS)} fields; "
                    f"at least {minimum_differences} are required"
                )
    for field in EXACT_UNIQUE_NICHE_FIELDS:
        owners: dict[str, str] = {}
        for scenario_id in expected_ids:
            field_value = canonical_json(value[scenario_id][field])
            if field_value in owners:
                raise InvalidModelOutput(
                    f"compact niche field {field} repeats exactly for "
                    f"{owners[field_value]} and {scenario_id}"
                )
            owners[field_value] = scenario_id
    for fields in EXACT_UNIQUE_NICHE_FIELD_GROUPS:
        owners: dict[str, str] = {}
        for scenario_id in expected_ids:
            group_value = canonical_json(
                [value[scenario_id][field] for field in fields]
            )
            if group_value in owners:
                raise InvalidModelOutput(
                    f"compact niche anchor {','.join(fields)} repeats exactly for "
                    f"{owners[group_value]} and {scenario_id}"
                )
            owners[group_value] = scenario_id
    return value


def encode_blueprint_response(
    value: Any,
    topic: Mapping[str, Any],
    scenario_ids: Sequence[str] | None = None,
    taxonomy_anchors: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Encode canonical niches into the compact Stage P transport representation."""

    topic_id = _identifier(topic.get("topicId"), "topic.topicId", 120)
    expected_ids = tuple(scenario_ids or scenario_ids_for_topic(topic_id))
    canonical = validate_blueprint_response(value, topic, expected_ids)
    encoded: dict[str, dict[str, Any]] = {}
    for scenario_id in expected_ids:
        niche = canonical[scenario_id]
        wire = {
            NICHE_WIRE_KEYS[field]: niche[field]
            for field in NICHE_SIGNATURE_FIELDS
            if not (taxonomy_anchors is not None and field in TAXONOMY_FIELDS)
        }
        wire[NICHE_WIRE_KEYS["controlOperator"]] = CONTROL_OPERATOR_WIRE_CODES[
            niche["controlOperator"]
        ]
        wire[NICHE_WIRE_KEYS["fourSiblingAffordance"]] = {
            AFFORDANCE_WIRE_KEYS[state]: niche["fourSiblingAffordance"][state]
            for state in FOUR_SIBLING_STATES
        }
        distinctness = niche["semanticDistinctnessFrom"]
        wire[NICHE_WIRE_KEYS["semanticDistinctnessFrom"]] = {
            DISTINCTNESS_WIRE_KEYS[field]: distinctness[field]
            for field in ("siblingIds", "distinction")
        }
        encoded[scenario_id] = wire
    _raise_schema_errors(
        encoded,
        build_blueprint_response_schema(topic, expected_ids, taxonomy_anchors),
        "encoded Stage P response",
    )
    return encoded


def decode_blueprint_response(
    value: Any,
    topic: Mapping[str, Any],
    scenario_ids: Sequence[str] | None = None,
    taxonomy_anchors: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate and losslessly decode the compact Stage P transport representation."""

    topic_id = _identifier(topic.get("topicId"), "topic.topicId", 120)
    expected_ids = tuple(scenario_ids or scenario_ids_for_topic(topic_id))
    _raise_schema_errors(
        value,
        build_blueprint_response_schema(topic, expected_ids, taxonomy_anchors),
        "wire Stage P response",
    )
    assert_no_target_fields(value)
    assert isinstance(value, dict)
    decoded: dict[str, dict[str, Any]] = {}
    for scenario_id in expected_ids:
        wire = value[scenario_id]
        niche = {
            field: wire[NICHE_WIRE_KEYS[field]]
            for field in NICHE_SIGNATURE_FIELDS
            if not (taxonomy_anchors is not None and field in TAXONOMY_FIELDS)
        }
        if taxonomy_anchors is not None:
            for field in TAXONOMY_FIELDS:
                niche[field] = taxonomy_anchors[scenario_id][field]
        niche["controlOperator"] = CONTROL_OPERATOR_BY_WIRE_CODE[
            wire[NICHE_WIRE_KEYS["controlOperator"]]
        ]
        affordance = wire[NICHE_WIRE_KEYS["fourSiblingAffordance"]]
        niche["fourSiblingAffordance"] = {
            state: affordance[AFFORDANCE_WIRE_KEYS[state]]
            for state in FOUR_SIBLING_STATES
        }
        distinctness = wire[NICHE_WIRE_KEYS["semanticDistinctnessFrom"]]
        niche["semanticDistinctnessFrom"] = {
            field: distinctness[DISTINCTNESS_WIRE_KEYS[field]]
            for field in ("siblingIds", "distinction")
        }
        decoded[scenario_id] = niche
    return validate_blueprint_response(decoded, topic, expected_ids)


def _string_list_schema(max_items: int = 5, max_length: int = 240) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": max_items,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1},
    }


def build_scenario_contract_schema(
    scenario_id: str,
    topic_id: str,
    blueprint: Mapping[str, Any] | None = None,
    *,
    wire: bool = False,
) -> dict[str, Any]:
    _identifier(scenario_id, "scenarioId")
    _identifier(topic_id, "topicId", 120)
    string_list = _string_list_schema()
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema",
            "scenarioId",
            "topicId",
            "mode",
            "premise",
            "participants",
            "startingState",
            "interactionOpportunity",
            "allowedToolClasses",
            "disallowedClaims",
            "scenarioOutcomeSpace",
            "requiredControlPhenomena",
        ],
        "properties": {
            "schema": {"const": "personaplex.scenario-contract.v2"},
            "scenarioId": {"const": scenario_id},
            "topicId": {"const": topic_id},
            "mode": (
                {"const": blueprint["interactionMode"]}
                if blueprint is not None
                else _semantic_text()
            ),
            "premise": _semantic_text(24),
            "participants": {
                "type": "array",
                "minItems": 2,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["role", "knowledge"],
                    "properties": {
                        "role": _semantic_text(2),
                        "knowledge": _semantic_text(),
                    },
                },
            },
            "startingState": {
                "type": "object",
                "additionalProperties": False,
                "required": ["knownFacts", "uncertainty", "policyConstraints"],
                "properties": {
                    "knownFacts": string_list,
                    "uncertainty": string_list,
                    "policyConstraints": string_list,
                },
            },
            "interactionOpportunity": string_list,
            "allowedToolClasses": _string_list_schema(max_items=4, max_length=120),
            "disallowedClaims": string_list,
            "scenarioOutcomeSpace": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "uniqueItems": True,
                "items": _semantic_text(),
            },
            "requiredControlPhenomena": _string_list_schema(max_items=5),
        },
    }
    if wire and blueprint is not None:
        schema["required"] = [
            field
            for field in schema["required"]
            if field not in {"scenarioOutcomeSpace", "requiredControlPhenomena"}
        ]
        schema["properties"].pop("scenarioOutcomeSpace")
        schema["properties"].pop("requiredControlPhenomena")
    return schema


def build_expansion_response_schema(
    scenario_id: str,
    topic_id: str,
    blueprint_hash: str,
    joint_blueprint_hash: str,
    blueprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "scenarioId",
            "topicId",
            "blueprintHash",
            "jointBlueprintHash",
            "scenarioContract",
        ],
        "properties": {
            "scenarioId": {"const": scenario_id},
            "topicId": {"const": topic_id},
            "blueprintHash": {"const": blueprint_hash},
            "jointBlueprintHash": {"const": joint_blueprint_hash},
            "scenarioContract": build_scenario_contract_schema(
                scenario_id, topic_id, blueprint, wire=blueprint is not None
            ),
        },
    }
    Draft202012Validator.check_schema(schema)
    return schema


def validate_expansion_response(
    value: Any,
    scenario_id: str,
    topic_id: str,
    blueprint_hash: str,
    joint_blueprint_hash: str,
    blueprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    schema = build_expansion_response_schema(
        scenario_id, topic_id, blueprint_hash, joint_blueprint_hash, blueprint
    )
    _raise_schema_errors(value, schema, "Stage E response")
    assert_no_target_fields(value)
    assert isinstance(value, dict)
    normalized = json.loads(canonical_json(value))
    if blueprint is not None:
        affordance = blueprint["fourSiblingAffordance"]
        normalized["scenarioContract"]["scenarioOutcomeSpace"] = [
            f"verified_positive: {affordance['verifiedPositive']}",
            f"verified_negative: {affordance['verifiedNegative']}",
            f"uncertain: {affordance['uncertain']}",
            f"superseded: {affordance['superseded']}",
        ]
        normalized["scenarioContract"]["requiredControlPhenomena"] = [
            f"control_operator: {blueprint['controlOperator']}",
            f"evidence_pivot: {blueprint['evidencePivot']}",
            f"causal_mechanism: {blueprint['causalMechanism']}",
            f"duplex_opportunity: {blueprint['duplexOpportunity']}",
        ]
        _raise_schema_errors(
            normalized["scenarioContract"],
            build_scenario_contract_schema(scenario_id, topic_id, blueprint),
            "materialized scenario contract",
        )
        assert_no_target_fields(normalized)
    return normalized


class StrictSchemaModel(Protocol):
    def generate(
        self,
        *,
        name: str,
        schema: Mapping[str, Any],
        instructions: str,
        context: Mapping[str, Any],
        max_output_tokens: int,
    ) -> tuple[Any, Mapping[str, Any]]:
        ...


@dataclass(frozen=True)
class PlannerBinding:
    endpoints: tuple[str, ...]
    model: str
    timeout_seconds: int = 240
    temperature: float = 0.85
    transport_attempts: int = DEFAULT_TRANSPORT_ATTEMPTS
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS
    retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS


MODEL_ATTEMPT_TRACE_SCHEMA = "personaplex.strict-schema-model-attempt.v1"


def _model_attempt_topic_id(context: Mapping[str, Any]) -> str:
    topic_card = context.get("topicCard")
    if isinstance(topic_card, Mapping) and isinstance(topic_card.get("topicId"), str):
        return str(topic_card["topicId"])
    topic_id = context.get("topicId")
    return str(topic_id) if isinstance(topic_id, str) and topic_id else "unbound"


def _write_model_attempt_trace(
    *,
    name: str,
    context: Mapping[str, Any],
    route_name: str,
    endpoint: str,
    logical_model: str,
    actual_model: str,
    attempt: int,
    finish_reason: Any,
    projection_binding: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    usage: Any,
    status: str,
    response: Any,
    defect: str | None = None,
) -> None:
    """Persist synthetic model output without prompts, credentials, or headers."""

    root_value = os.environ.get("PERSONAPLEX_MODEL_TRACE_ROOT", "").strip()
    if not root_value:
        return
    normalized_response = json.loads(canonical_json(response))
    body: dict[str, Any] = {
        "schema": MODEL_ATTEMPT_TRACE_SCHEMA,
        "name": name,
        "topicId": _model_attempt_topic_id(context),
        "status": status,
        "route": {
            "kind": route_name,
            "endpoint": endpoint,
            "logicalModel": logical_model,
            "actualModel": actual_model,
        },
        "transportAttempt": attempt,
        "finishReason": finish_reason,
        "requestBinding": {
            "promptHash": content_hash(messages),
            "canonicalSchemaHash": projection_binding["canonicalSchemaHash"],
            "transportSchemaHash": projection_binding["transportSchemaHash"],
            "profileHash": projection_binding["profile"]["profileHash"],
        },
        "response": normalized_response,
        "responseHash": content_hash(normalized_response),
        "usage": usage if isinstance(usage, Mapping) else {},
    }
    if defect:
        body["defect"] = defect
    trace_hash = content_hash(body)
    record = dict(body)
    record["traceHash"] = trace_hash
    path = (
        Path(root_value)
        / name
        / body["topicId"]
        / f"{trace_hash[7:]}.json"
    )
    if path.exists():
        if read_json(path) != record:
            raise ScenarioBlueprintError(f"model-attempt trace identity mismatch: {path}")
    else:
        _write_immutable_json(path, record)
    print(
        canonical_json(
            {
                "event": "strict_schema_model_attempt",
                "name": name,
                "topicId": body["topicId"],
                "status": status,
                "route": route_name,
                "actualModel": actual_model,
                "attempt": attempt,
                "traceHash": trace_hash,
            }
        ),
        flush=True,
    )


class ThreeEndpointStrictSchemaPlanner:
    """OpenAI-compatible strict-schema client with one to three lanes."""

    _RETRIABLE_HTTP_STATUS = RETRIABLE_HTTP_STATUS

    def __init__(
        self,
        endpoints: Sequence[str],
        model: str,
        api_key: str = "",
        *,
        timeout_seconds: int = 240,
        temperature: float = 0.85,
        allow_fewer_endpoints: bool = False,
        transport_attempts: int = DEFAULT_TRANSPORT_ATTEMPTS,
        retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
        retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
        fallback_endpoints: Sequence[str] = (),
        fallback_model: str = "",
        fallback_api_key: str = "",
        prefer_fallback: bool = False,
        bind_fallback: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        del allow_fewer_endpoints
        fallback_endpoint_values = tuple(fallback_endpoints)
        fallback_model_value = fallback_model.strip()
        if bool(fallback_endpoint_values) != bool(fallback_model_value):
            raise ScenarioBlueprintError(
                "fallback endpoints and fallback model must be configured together"
            )
        try:
            normalized = normalize_chat_completion_endpoints(endpoints)
            normalized_fallback = (
                normalize_chat_completion_endpoints(fallback_endpoint_values)
                if fallback_endpoint_values
                else ()
            )
            retry_settings = validate_retry_settings(
                transport_attempts, retry_base_seconds, retry_max_seconds
            )
        except ValueError as error:
            raise ScenarioBlueprintError(str(error)) from error
        if not model.strip():
            raise ScenarioBlueprintError("planner model is required")
        self.config = PlannerBinding(
            normalized,
            model.strip(),
            timeout_seconds,
            temperature,
            *retry_settings,
        )
        self.api_key = api_key
        self.fallback_config = (
            PlannerBinding(
                normalized_fallback,
                fallback_model_value,
                timeout_seconds,
                temperature,
                *retry_settings,
            )
            if normalized_fallback
            else None
        )
        self.fallback_api_key = fallback_api_key
        self._prefer_fallback = bool(prefer_fallback and self.fallback_config)
        self._bind_fallback = bool(bind_fallback and self.fallback_config)
        self._sleep = sleep
        self._lock = Lock()
        self._next_endpoint = {"primary": 0, "fallback": 0}

    def binding(self) -> dict[str, Any]:
        profiles = [
            schema_capability_profile(endpoint, self.config.model)
            for endpoint in self.config.endpoints
        ]
        if all(profile.response_format_supported for profile in profiles):
            response_format = "strict_json_schema"
        elif all(not profile.response_format_supported for profile in profiles):
            response_format = "omitted"
        else:
            response_format = "profile_bound"
        binding = {
            "protocol": "openai_chat_completions",
            "endpoints": list(self.config.endpoints),
            "model": self.config.model,
            "reasoning": {"enabled": False},
            "responseFormat": response_format,
            "temperature": self.config.temperature,
            "schemaTransportProfiles": [
                {"endpoint": endpoint, **profile.binding()}
                for endpoint, profile in zip(self.config.endpoints, profiles)
            ],
            "transportRetry": {
                "attempts": self.config.transport_attempts,
                "baseSeconds": self.config.retry_base_seconds,
                "maxSeconds": self.config.retry_max_seconds,
                "retryAfterStatuses": sorted(self._RETRIABLE_HTTP_STATUS),
            },
        }
        if self._bind_fallback and self.fallback_config is not None:
            fallback_profiles = [
                schema_capability_profile(endpoint, self.fallback_config.model)
                for endpoint in self.fallback_config.endpoints
            ]
            binding["runtimeFallback"] = {
                "protocol": "openai_chat_completions",
                "endpoints": list(self.fallback_config.endpoints),
                "model": self.fallback_config.model,
                "preferred": self._prefer_fallback,
                "reasoning": {"enabled": False},
                "schemaTransportProfiles": [
                    {"endpoint": endpoint, **profile.binding()}
                    for endpoint, profile in zip(
                        self.fallback_config.endpoints, fallback_profiles
                    )
                ],
            }
        return binding

    def _rotation(self, route_name: str, endpoints: tuple[str, ...]) -> tuple[str, ...]:
        with self._lock:
            start = self._next_endpoint[route_name]
            self._next_endpoint[route_name] = (start + 1) % len(endpoints)
        return endpoints[start:] + endpoints[:start]

    def generate(
        self,
        *,
        name: str,
        schema: Mapping[str, Any],
        instructions: str,
        context: Mapping[str, Any],
        max_output_tokens: int,
    ) -> tuple[Any, Mapping[str, Any]]:
        if not 1 <= max_output_tokens <= MAX_LIVE_OUTPUT_TOKENS:
            raise ScenarioBlueprintError(
                f"live output-token budget must be in [1,{MAX_LIVE_OUTPUT_TOKENS}]"
            )
        Draft202012Validator.check_schema(dict(schema))
        transport_errors: list[str] = []
        routes: list[tuple[str, PlannerBinding, str]] = [
            ("primary", self.config, self.api_key)
        ]
        if self.fallback_config is not None:
            fallback_route = (
                "fallback",
                self.fallback_config,
                self.fallback_api_key,
            )
            routes = [fallback_route, *routes] if self._prefer_fallback else [*routes, fallback_route]

        for route_name, route_config, route_api_key in routes:
            rotation = self._rotation(route_name, route_config.endpoints)
            request_context: Mapping[str, Any] = context
            for attempt in range(1, route_config.transport_attempts + 1):
                endpoint = rotation[(attempt - 1) % len(rotation)]
                profile = schema_capability_profile(endpoint, route_config.model)
                projection = build_schema_transport_projection(
                    endpoint, route_config.model, schema
                )
                prompt_instructions = (
                    append_canonical_schema_contract(instructions, schema)
                    if profile.canonical_schema_in_prompt
                    else instructions
                )
                messages = [
                    {"role": "system", "content": prompt_instructions},
                    {"role": "user", "content": canonical_json(request_context)},
                ]
                payload = {
                    "model": route_config.model,
                    "stream": False,
                    "reasoning": {"enabled": False},
                    "temperature": route_config.temperature,
                    "max_tokens": max_output_tokens,
                    "messages": messages,
                }
                if not is_openrouter_endpoint(endpoint):
                    payload["reasoning"] = {"effort": "none"}
                    payload["reasoning_effort"] = "none"
                if profile.response_format_supported:
                    payload["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": name,
                            "strict": True,
                            "schema": projection.transport_schema,
                        },
                    }
                encoded = canonical_json(payload).encode("utf-8")
                headers = {"content-type": "application/json"}
                if route_api_key:
                    headers["authorization"] = f"Bearer {route_api_key}"
                request = urllib.request.Request(
                    endpoint, data=encoded, headers=headers, method="POST"
                )
                response_headers = None
                try:
                    with urllib.request.urlopen(
                        request, timeout=route_config.timeout_seconds
                    ) as response:
                        response_headers = getattr(response, "headers", None)
                        envelope = json.loads(response.read())
                except urllib.error.HTTPError as error:
                    if error.code in self._RETRIABLE_HTTP_STATUS:
                        reset = error.headers.get("x-ratelimit-reset") if error.headers else None
                        reset_suffix = f" reset={reset}" if reset else ""
                        transport_errors.append(
                            f"route {route_name} attempt {attempt} endpoint {endpoint}: "
                            f"HTTP {error.code}{reset_suffix}"
                        )
                        if attempt < route_config.transport_attempts:
                            requested_delay = retry_after_seconds(error.headers)
                            self._sleep(
                                retry_delay_seconds(
                                    attempt,
                                    route_config.retry_base_seconds,
                                    route_config.retry_max_seconds,
                                    requested_delay,
                                )
                            )
                        continue
                    raise ScenarioBlueprintError(
                        f"planner endpoint rejected the strict request with HTTP {error.code}"
                    ) from error
                except (
                    urllib.error.URLError,
                    TimeoutError,
                    ConnectionError,
                    OSError,
                ) as error:
                    transport_errors.append(
                        f"route {route_name} attempt {attempt} endpoint {endpoint}: "
                        f"{type(error).__name__}"
                    )
                    if attempt < route_config.transport_attempts:
                        self._sleep(
                            retry_delay_seconds(
                                attempt,
                                route_config.retry_base_seconds,
                                route_config.retry_max_seconds,
                            )
                        )
                    continue
                provider_error = classify_provider_error(envelope)
                if provider_error is not None:
                    if not provider_error.retryable:
                        raise ScenarioBlueprintError(
                            "planner provider returned a non-retriable error envelope "
                            f"({provider_error.classification})"
                        )
                    transport_errors.append(
                        f"route {route_name} attempt {attempt} endpoint {endpoint}: "
                        f"{provider_error.classification}"
                    )
                    if attempt < route_config.transport_attempts:
                        requested_delay = retry_after_seconds(response_headers)
                        self._sleep(
                            retry_delay_seconds(
                                attempt,
                                route_config.retry_base_seconds,
                                route_config.retry_max_seconds,
                                requested_delay,
                            )
                        )
                    continue
                try:
                    choice = envelope["choices"][0]
                    finish_reason = choice.get("finish_reason")
                    content = choice["message"]["content"]
                except (KeyError, IndexError, TypeError) as error:
                    raise InvalidModelOutput(
                        "planner response lacks choices[0].message.content"
                    ) from error
                if finish_reason == "length":
                    _write_model_attempt_trace(
                        name=name,
                        context=context,
                        route_name=route_name,
                        endpoint=endpoint,
                        logical_model=self.config.model,
                        actual_model=route_config.model,
                        attempt=attempt,
                        finish_reason=finish_reason,
                        projection_binding=projection.binding,
                        messages=messages,
                        usage=envelope.get("usage", {}),
                        status="transport_truncated",
                        response=content,
                    )
                    raise TruncatedModelOutput(
                        f"planner output was truncated at its {max_output_tokens}-token live limit"
                    )
                if not isinstance(content, str):
                    raise InvalidModelOutput(
                        "planner strict content must be a raw JSON string"
                    )
                try:
                    value = json.loads(content)
                except json.JSONDecodeError as error:
                    _write_model_attempt_trace(
                        name=name,
                        context=context,
                        route_name=route_name,
                        endpoint=endpoint,
                        logical_model=self.config.model,
                        actual_model=route_config.model,
                        attempt=attempt,
                        finish_reason=finish_reason,
                        projection_binding=projection.binding,
                        messages=messages,
                        usage=envelope.get("usage", {}),
                        status="malformed_json",
                        response=content,
                        defect=f"{error.msg} at character {error.pos}",
                    )
                    raise InvalidModelOutput(
                        "planner returned malformed strict JSON; prose recovery is forbidden"
                    ) from error
                canonical_errors = list(
                    Draft202012Validator(schema).iter_errors(value)
                )
                if canonical_errors:
                    defect = canonical_validation_defect(canonical_errors)
                    _write_model_attempt_trace(
                        name=name,
                        context=context,
                        route_name=route_name,
                        endpoint=endpoint,
                        logical_model=self.config.model,
                        actual_model=route_config.model,
                        attempt=attempt,
                        finish_reason=finish_reason,
                        projection_binding=projection.binding,
                        messages=messages,
                        usage=envelope.get("usage", {}),
                        status="canonical_schema_rejected",
                        response=value,
                        defect=defect,
                    )
                    if attempt < route_config.transport_attempts:
                        request_context = canonical_retry_context(
                            context, defect, attempt + 1
                        )
                        transport_errors.append(
                            f"route {route_name} attempt {attempt} endpoint {endpoint}: "
                            f"canonical_schema_rejected {defect}"
                        )
                        continue
                    raise InvalidModelOutput(
                        "planner output violates the canonical response schema: "
                        + defect
                    )
                metadata = {
                    "endpoint": endpoint,
                    "model": route_config.model,
                    "transportAttempt": attempt,
                    "transportRoute": {
                        "kind": route_name,
                        "fallback": route_name == "fallback",
                        "logicalModel": self.config.model,
                        "actualModel": route_config.model,
                        "logicalEndpointsHash": content_hash(
                            list(self.config.endpoints)
                        ),
                    },
                    "finishReason": finish_reason,
                    "responseHash": content_hash(value),
                    "schemaTransport": projection.binding,
                    "requestBinding": {
                        "profileHash": projection.binding["profile"]["profileHash"],
                        "modelHash": content_hash(route_config.model),
                        "schemaHash": projection.binding["canonicalSchemaHash"],
                        "promptHash": content_hash(messages),
                    },
                    "usage": envelope.get("usage", {}),
                }
                _write_model_attempt_trace(
                    name=name,
                    context=context,
                    route_name=route_name,
                    endpoint=endpoint,
                    logical_model=self.config.model,
                    actual_model=route_config.model,
                    attempt=attempt,
                    finish_reason=finish_reason,
                    projection_binding=projection.binding,
                    messages=messages,
                    usage=envelope.get("usage", {}),
                    status="canonical_schema_accepted",
                    response=value,
                )
                return value, metadata
        raise ModelTransportUnavailable(
            "all physical planner routes unavailable: " + "; ".join(transport_errors)
        )


def _model_binding(model: StrictSchemaModel) -> dict[str, Any]:
    binding_method = getattr(model, "binding", None)
    if callable(binding_method):
        value = binding_method()
        if not isinstance(value, Mapping):
            raise ScenarioBlueprintError("planner binding() must return an object")
        return json.loads(canonical_json(dict(value)))
    return {
        "protocol": "strict_schema_model",
        "model": str(getattr(model, "model", type(model).__name__)),
        "reasoning": {"enabled": False},
        "responseFormat": "strict_json_schema",
    }


def _call_model(
    model: StrictSchemaModel,
    *,
    name: str,
    schema: Mapping[str, Any],
    instructions: str,
    context: Mapping[str, Any],
    max_output_tokens: int,
) -> tuple[Any, dict[str, Any]]:
    result = model.generate(
        name=name,
        schema=schema,
        instructions=instructions,
        context=context,
        max_output_tokens=max_output_tokens,
    )
    if isinstance(result, tuple) and len(result) == 2:
        value, metadata = result
    else:
        value, metadata = result, {}
    if not isinstance(metadata, Mapping):
        raise InvalidModelOutput("planner metadata must be an object")
    copied_metadata = json.loads(canonical_json(dict(metadata)))
    endpoint = copied_metadata.get("endpoint")
    model_name = copied_metadata.get("model")
    if isinstance(endpoint, str) and isinstance(model_name, str):
        try:
            expected_schema_transport = build_schema_transport_projection(
                endpoint, model_name, schema
            ).binding
        except ValueError:
            expected_schema_transport = None
        if expected_schema_transport is not None:
            supplied_schema_transport = copied_metadata.get("schemaTransport")
            if (
                supplied_schema_transport is not None
                and supplied_schema_transport != expected_schema_transport
            ):
                raise InvalidModelOutput("planner schema-transport binding mismatch")
            copied_metadata["schemaTransport"] = expected_schema_transport
    usage = copied_metadata.get("usage")
    if isinstance(usage, dict):
        completion_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
        if isinstance(completion_tokens, int) and completion_tokens > max_output_tokens:
            raise InvalidModelOutput("planner usage exceeds the assigned live output-token budget")
    return value, copied_metadata


TAXONOMY_SYSTEM = """You design exactly twenty materially different branches beneath one broad topic.
Return only the strict JSON object with the canonical properties submode, participantRelationship,
setting, centralResource, and centralTension. Cover the breadth of the topic rather than producing
mirrored pairs or lexical variants. Each scenario ID has an immutable assignedInteractionMode in the
input; its submode must materially realize that exact mode. participantRelationship must identify both participant roles and
their social or organizational relationship, never an object, document, service, or event. A setting is
an environment, not an event. centralResource must be a concise concrete noun phrase naming the object,
document, information, service, entitlement, or decision at issue; it must never be quoted dialogue, an
apology, a speech act, a target response, the broad topic, or an interaction-mode label. centralTension
must state the specific conflict, tradeoff, uncertainty, or incompatible goals around that resource.
Every field must be a complete phrase, every submode and centralResource must be exact-unique, and every
five-field anchor must be distinct. Do not emit dialogue, responses, control frames, target text/audio,
names, contact data, credentials, or placeholders. There is no character or word ceiling on a field.
Close each JSON string only at a natural semantic boundary after its phrase or clause is complete;
use terminal punctuation where natural and never clip a word or thought to shorten output."""


def _taxonomy_context(
    request: Mapping[str, Any], topic: Mapping[str, Any], scenario_ids: Sequence[str]
) -> dict[str, Any]:
    interaction_modes, safe_stakes = _topic_enums(topic)
    assigned_modes = {
        scenario_id: _coverage_assignment(ordinal, interaction_modes, safe_stakes)[
            "interactionMode"
        ]
        for ordinal, scenario_id in enumerate(scenario_ids)
    }
    return {
        "task": "Jointly decompose one broad topic into exactly twenty distinct scenario branches.",
        "topicCard": dict(topic),
        "scenarioIds": list(scenario_ids),
        "assignedInteractionModeByScenarioId": assigned_modes,
        "corpusConstraints": {
            "topicConstraints": request.get("topicConstraints", {}),
            "requiredControlCoverage": request.get("requiredControlCoverage", {}),
        },
        "diversityContract": [
            "Each submode must materially realize its scenario ID's immutable assignedInteractionMode.",
            "Use the full breadth of the topic card, including materially different relationships and settings.",
            "Do not create mirrored pairs or variants that differ only in an object name.",
            "Every submode is exact-unique and every five-field anchor represents a distinct branch.",
        ],
        "fieldDefinitions": {
            "submode": "A concrete problem/activity subclass, never an interaction-mode label.",
            "participantRelationship": "Both participant roles and their social/organizational relationship; never an object, document, service, or event.",
            "setting": "Only the physical, institutional, or communication environment.",
            "centralResource": "A concise concrete noun phrase naming the object, document, information, service, entitlement, or decision at issue; never dialogue, an apology, a speech act, a target response, the topic, a mode, or a relationship.",
            "centralTension": "The specific conflict, tradeoff, uncertainty, or incompatible goals around that resource.",
        },
        "outputContract": {
            "strictJsonSchema": True,
            "oneJointCall": True,
            "maxLiveOutputTokens": TAXONOMY_MAX_OUTPUT_TOKENS,
            "targetFieldsForbidden": True,
        },
    }


def _taxonomy_stage_key(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    response_schema: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
) -> str:
    return content_hash(
        {
            "stage": "scenario_taxonomy_v5",
            "protocolVersion": TAXONOMY_PROTOCOL_VERSION,
            "systemHash": content_hash(TAXONOMY_SYSTEM),
            "requestHash": content_hash(request),
            "topicCardHash": content_hash(topic),
            "responseSchemaHash": content_hash(response_schema),
            "plannerBindingHash": content_hash(planner_binding),
        }
    )


def _validate_taxonomy_checkpoint(
    checkpoint: Any,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    response_schema: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ScenarioBlueprintError("taxonomy checkpoint must be an object")
    required = {
        "schema", "stageKey", "protocolVersion", "requestHash", "topicId",
        "topicCardHash", "responseSchemaHash", "plannerBinding", "plannerBindingHash",
        "modelCall", "taxonomyAnchors", "taxonomyAnchorsHash", "checkpointHash",
    }
    if set(checkpoint) != required:
        raise ScenarioBlueprintError("taxonomy checkpoint has an invalid field set")
    expected = {
        "schema": TAXONOMY_CHECKPOINT_SCHEMA,
        "stageKey": _taxonomy_stage_key(request, topic, response_schema, planner_binding),
        "protocolVersion": TAXONOMY_PROTOCOL_VERSION,
        "requestHash": content_hash(request),
        "topicId": topic["topicId"],
        "topicCardHash": content_hash(topic),
        "responseSchemaHash": content_hash(response_schema),
        "plannerBinding": dict(planner_binding),
        "plannerBindingHash": content_hash(planner_binding),
    }
    for field, expected_value in expected.items():
        if checkpoint.get(field) != expected_value:
            raise ScenarioBlueprintError(f"taxonomy checkpoint binding mismatch: {field}")
    anchors = validate_taxonomy_anchors(checkpoint.get("taxonomyAnchors"), topic)
    if checkpoint.get("taxonomyAnchorsHash") != content_hash(anchors):
        raise ScenarioBlueprintError("taxonomy anchor hash mismatch")
    if not isinstance(checkpoint.get("modelCall"), dict):
        raise ScenarioBlueprintError("taxonomy modelCall must be an object")
    _checkpoint_body_hash(checkpoint)
    return checkpoint


def generate_topic_taxonomy(
    *,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    output_root: Path,
    planner: StrictSchemaModel,
    max_attempts: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    """Create or exactly resume the compact twenty-branch taxonomy checkpoint."""

    if not 1 <= max_attempts <= 12:
        raise ScenarioBlueprintError("max_attempts must be in [1,12]")
    topic_id = _identifier(topic.get("topicId"), "topic.topicId", 120)
    scenario_ids = scenario_ids_for_topic(topic_id)
    response_schema = build_taxonomy_response_schema(topic, scenario_ids)
    planner_binding = _model_binding(planner)
    stage_key = _taxonomy_stage_key(request, topic, response_schema, planner_binding)
    path = _checkpoint_path(Path(output_root), "taxonomy", topic_id, stage_key)
    if path.exists():
        if not resume:
            raise ScenarioBlueprintError(f"taxonomy checkpoint exists; use --resume: {path}")
        return _validate_taxonomy_checkpoint(
            read_json(path), request, topic, response_schema, planner_binding
        )
    context = _taxonomy_context(request, topic, scenario_ids)
    failures: list[str] = []
    previous_rejected_taxonomy: Any = None
    for attempt in range(1, max_attempts + 1):
        attempt_context = dict(context)
        if failures:
            attempt_context["retryFeedback"] = {
                "attempt": attempt,
                "previousDefect": failures[-1],
                "directive": (
                    "Revise the supplied previousRejectedTaxonomy, return all twenty IDs, "
                    "obey every exact structural prohibition, and never return it unchanged."
                ),
            }
            if previous_rejected_taxonomy is not None:
                attempt_context["previousRejectedTaxonomy"] = previous_rejected_taxonomy
                attempt_context["structuralRevisionContract"] = (
                    _taxonomy_structural_revision_contract(
                        previous_rejected_taxonomy, topic, scenario_ids
                    )
                )
                attempt_context["retryFeedback"]["previousRejectedTaxonomyHash"] = (
                    content_hash(previous_rejected_taxonomy)
                )
        response: Any = None
        try:
            response, metadata = _call_model(
                planner,
                name="personaplex_scenario_taxonomy_v5",
                schema=response_schema,
                instructions=TAXONOMY_SYSTEM,
                context=attempt_context,
                max_output_tokens=TAXONOMY_MAX_OUTPUT_TOKENS,
            )
            anchors = decode_taxonomy_response(response, topic, scenario_ids)
            body = {
                "schema": TAXONOMY_CHECKPOINT_SCHEMA,
                "stageKey": stage_key,
                "protocolVersion": TAXONOMY_PROTOCOL_VERSION,
                "requestHash": content_hash(request),
                "topicId": topic_id,
                "topicCardHash": content_hash(topic),
                "responseSchemaHash": content_hash(response_schema),
                "plannerBinding": planner_binding,
                "plannerBindingHash": content_hash(planner_binding),
                "modelCall": metadata,
                "taxonomyAnchors": anchors,
                "taxonomyAnchorsHash": content_hash(anchors),
            }
            checkpoint = dict(body)
            checkpoint["checkpointHash"] = content_hash(body)
            _write_immutable_json(path, checkpoint)
            return checkpoint
        except ModelTransportUnavailable:
            raise
        except Exception as error:
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            if response is not None:
                previous_rejected_taxonomy = response
    raise ScenarioBlueprintError(
        f"taxonomy exhausted {max_attempts} attempts for {topic_id}: " + " | ".join(failures)
    )


STAGE_P_SYSTEM = """You create compact semantic scenario niches for a controlled conversational-audio corpus.
Return only the strict JSON object. Jointly design all twenty niches before filling fields so they
are materially distinct in relationship, setting, resource, tension, evidence, causality, stakes,
outcome topology, four-sibling affordance, and duplex behavior. Treat every schema-fixed coverage
value as an immutable design mandate, especially interaction mode and control-revision operator.
Do not emit dialogue, utterances,
scripts, canonical responses, target text/audio, names, contact data, credentials, or placeholders.
Never organize the set into mirrored pairs or contrast pairs: every pair must differ on at least three
of the typed niche fields even when one is named as the other's nearest sibling. Do not certify quality
Every submode must be unique, and the schema-provided compound anchor groups must be exact-duplicate-free
across all twenty slots. Do not certify quality and do not
ask the host to infer, generate, or repair semantic content. evidencePivot must be a mutable fact, tool,
policy, posture, commitment, or interruption event. causalMechanism must explain how that event changes
the next action and must not copy a participant relationship. The four sibling routes must be materially
different consequences, not restatements of the central tension."""


STAGE_E_SYSTEM = """You expand one assigned compact niche into one PersonaPlex scenario contract.
Return only the strict JSON object. Use the assigned niche, not a sibling niche, while consulting the
complete joint blueprint to preserve semantic separation. Supply context and causal affordances but
no dialogue, utterance, script, canonical response, target transcript, target text, or target audio.
The mode must equal the assigned blueprint interactionMode. The premise and participants must concretely
realize the assigned taxonomy anchors without placeholders. scenarioOutcomeSpace must contain exactly four
materially different routes in this order: verified-positive, verified-negative, uncertain, superseded.
Each route must causally change the appropriate next behavior. requiredControlPhenomena must identify how
the assigned controlOperator changes that behavior. Do not alter or reinterpret any bound identity or hash.
Natural-language fields have no character ceiling: finish every phrase, sentence, and list item at a
natural semantic boundary before closing its JSON string."""


def _blueprint_context(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    scenario_ids: Sequence[str],
    taxonomy_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task": "Jointly create exactly twenty compact, unique scenario-niche blueprints.",
        "topicCard": dict(topic),
        "scenarioIds": list(scenario_ids),
        "boundBranchTaxonomy": taxonomy_checkpoint["taxonomyAnchors"],
        "boundBranchTaxonomyHash": taxonomy_checkpoint["taxonomyAnchorsHash"],
        "boundBranchTaxonomyAdmissionHash": taxonomy_checkpoint["checkpointHash"],
        "boundBranchTaxonomyLineageHash": taxonomy_checkpoint["lineageHash"],
        "corpusConstraints": {
            "topicConstraints": request.get("topicConstraints", {}),
            "semanticControl": request.get("semanticControl", {}),
            "requiredControlCoverage": request.get("requiredControlCoverage", {}),
        },
        "semanticDistinctnessContract": (
            "Every niche must differ materially from all nineteen siblings on at least three of the typed "
            "fields. Do not create mirrored or minimally varied pairs. In semanticDistinctnessFrom, name "
            "the one or two nearest sibling IDs and state the compact decisive distinction; even those "
            "nearest siblings must satisfy the same three-field divergence floor."
        ),
        "coverageLatticeContract": (
            "The strict schema assigns each slot an interaction mode, control-revision operator, stakes "
            "profile, outcome topology, and duplex behavior. Realize every assigned cell materially; do not "
            "collapse different cells into one generic resolution template."
        ),
        "uniqueAnchorContract": list(EXACT_UNIQUE_NICHE_FIELDS),
        "uniqueCompoundAnchorContract": [
            list(fields) for fields in EXACT_UNIQUE_NICHE_FIELD_GROUPS
        ],
        "identityContract": "The response object keys are the complete immutable scenario-ID set.",
        "outputContract": {
            "strictJsonSchema": True,
            "oneJointCall": True,
            "initialLiveOutputTokens": BLUEPRINT_INITIAL_OUTPUT_TOKENS,
            "maxLiveOutputTokens": BLUEPRINT_MAX_OUTPUT_TOKENS,
            "tokenBudgetEscalation": "double_only_after_authenticated_finish_reason_length",
            "targetFieldsForbidden": True,
        },
    }


def make_blueprint_set(
    topic: Mapping[str, Any],
    blueprints: Mapping[str, Mapping[str, Any]],
    planner_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    topic_id = _identifier(topic.get("topicId"), "topic.topicId", 120)
    expected_ids = scenario_ids_for_topic(topic_id)
    validated = validate_blueprint_response(dict(blueprints), topic, expected_ids)
    joint_hash = content_hash(validated)
    bound_planner = json.loads(
        canonical_json(dict(planner_binding or {"protocol": "fixture_strict_schema_model"}))
    )
    return {
        "schema": BLUEPRINT_SET_SCHEMA,
        "topicId": topic_id,
        "topicCardHash": content_hash(topic),
        "scenarioIds": list(expected_ids),
        "jointBlueprintHash": joint_hash,
        "blueprintPlannerBinding": bound_planner,
        "blueprintPlannerBindingHash": content_hash(bound_planner),
        "blueprintHashes": {
            scenario_id: content_hash(validated[scenario_id]) for scenario_id in expected_ids
        },
        "blueprints": validated,
    }


def validate_blueprint_set(value: Any, topic: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScenarioBlueprintError("blueprint set must be an object")
    required = {
        "schema",
        "topicId",
        "topicCardHash",
        "scenarioIds",
        "jointBlueprintHash",
        "blueprintPlannerBinding",
        "blueprintPlannerBindingHash",
        "blueprintHashes",
        "blueprints",
    }
    if set(value) != required:
        raise ScenarioBlueprintError("blueprint set has an invalid field set")
    topic_id = _identifier(topic.get("topicId"), "topic.topicId", 120)
    expected_ids = scenario_ids_for_topic(topic_id)
    if value["schema"] != BLUEPRINT_SET_SCHEMA or value["topicId"] != topic_id:
        raise ScenarioBlueprintError("blueprint set identity binding mismatch")
    if value["topicCardHash"] != content_hash(topic):
        raise ScenarioBlueprintError("blueprint set topic hash mismatch")
    if value["scenarioIds"] != list(expected_ids):
        raise ScenarioBlueprintError("blueprint set scenario-ID binding mismatch")
    blueprints = validate_blueprint_response(value["blueprints"], topic, expected_ids)
    if value["jointBlueprintHash"] != content_hash(blueprints):
        raise ScenarioBlueprintError("joint blueprint hash mismatch")
    if not isinstance(value["blueprintPlannerBinding"], dict):
        raise ScenarioBlueprintError("blueprint planner binding must be an object")
    if value["blueprintPlannerBindingHash"] != content_hash(value["blueprintPlannerBinding"]):
        raise ScenarioBlueprintError("blueprint planner binding hash mismatch")
    expected_hashes = {
        scenario_id: content_hash(blueprints[scenario_id]) for scenario_id in expected_ids
    }
    if value["blueprintHashes"] != expected_hashes:
        raise ScenarioBlueprintError("per-scenario blueprint hash mismatch")
    return value


def _checkpoint_path(output_root: Path, stage: str, identity: str, stage_key: str) -> Path:
    return (
        output_root
        / CHECKPOINT_ROOT_NAME
        / "checkpoints"
        / stage
        / identity
        / f"{stage_key[7:]}.json"
    )


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ScenarioBlueprintError(f"immutable artifact collision at {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ScenarioBlueprintError(f"immutable artifact collision at {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def _write_immutable_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, resume: bool) -> None:
    payload = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    if path.exists() and not resume:
        raise ScenarioBlueprintError(f"canonical artifact exists; use --resume: {path}")
    _write_immutable_bytes(path, payload)


def _checkpoint_body_hash(checkpoint: Mapping[str, Any]) -> str:
    body = dict(checkpoint)
    supplied = body.pop("checkpointHash", None)
    expected = content_hash(body)
    if supplied != expected:
        raise ScenarioBlueprintError("immutable checkpoint content hash is invalid")
    return expected


def _topic_stage_key(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    schema: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
    taxonomy_checkpoint: Mapping[str, Any],
) -> str:
    return content_hash(
        {
            "stage": "scenario_blueprints_v5",
            "protocolVersion": BLUEPRINT_PROTOCOL_VERSION,
            "stageSystemHash": content_hash(STAGE_P_SYSTEM),
            "typedDivergenceDenominator": TYPED_DIVERGENCE_DENOMINATOR,
            "taxonomyAdmissionCheckpointHash": taxonomy_checkpoint["checkpointHash"],
            "taxonomyAdmissionProtocolHash": taxonomy_checkpoint["protocolHash"],
            "taxonomyPlannerBindingHash": taxonomy_checkpoint["taxonomyPlannerBindingHash"],
            "taxonomyJudgeModelBindingHash": taxonomy_checkpoint[
                "taxonomyJudgeModelBindingHash"
            ],
            "taxonomySourceCheckpointHash": taxonomy_checkpoint[
                "admittedSourceCheckpointHash"
            ],
            "taxonomyJudgeSourceHash": taxonomy_checkpoint["taxonomyJudgeSourceHash"],
            "taxonomyRepairSourceHash": taxonomy_checkpoint["taxonomyRepairSourceHash"],
            "taxonomyAnchorsHash": taxonomy_checkpoint["taxonomyAnchorsHash"],
            "requestHash": content_hash(request),
            "topicCardHash": content_hash(topic),
            "responseSchemaHash": content_hash(schema),
            "plannerBindingHash": content_hash(planner_binding),
        }
    )


def _validate_blueprint_checkpoint(
    checkpoint: Any,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    schema: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
    taxonomy_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ScenarioBlueprintError("blueprint checkpoint must be an object")
    required = {
        "schema",
        "stageKey",
        "requestHash",
        "topicId",
        "topicCardHash",
        "responseSchemaHash",
        "plannerBindingHash",
        "plannerBinding",
        "taxonomyAdmissionCheckpointHash",
        "taxonomyAdmissionProtocolHash",
        "taxonomyPlannerBindingHash",
        "taxonomyJudgeModelBindingHash",
        "taxonomySourceCheckpointHash",
        "taxonomyJudgeSourceHash",
        "taxonomyRepairSourceHash",
        "taxonomyAnchorsHash",
        "modelCall",
        "blueprintSet",
        "checkpointHash",
    }
    if set(checkpoint) != required:
        raise ScenarioBlueprintError("blueprint checkpoint has an invalid field set")
    stage_key = _topic_stage_key(
        request, topic, schema, planner_binding, taxonomy_checkpoint
    )
    expected = {
        "schema": BLUEPRINT_CHECKPOINT_SCHEMA,
        "stageKey": stage_key,
        "requestHash": content_hash(request),
        "topicId": topic["topicId"],
        "topicCardHash": content_hash(topic),
        "responseSchemaHash": content_hash(schema),
        "plannerBindingHash": content_hash(planner_binding),
        "plannerBinding": dict(planner_binding),
        "taxonomyAdmissionCheckpointHash": taxonomy_checkpoint["checkpointHash"],
        "taxonomyAdmissionProtocolHash": taxonomy_checkpoint["protocolHash"],
        "taxonomyPlannerBindingHash": taxonomy_checkpoint["taxonomyPlannerBindingHash"],
        "taxonomyJudgeModelBindingHash": taxonomy_checkpoint[
            "taxonomyJudgeModelBindingHash"
        ],
        "taxonomySourceCheckpointHash": taxonomy_checkpoint[
            "admittedSourceCheckpointHash"
        ],
        "taxonomyJudgeSourceHash": taxonomy_checkpoint["taxonomyJudgeSourceHash"],
        "taxonomyRepairSourceHash": taxonomy_checkpoint["taxonomyRepairSourceHash"],
        "taxonomyAnchorsHash": taxonomy_checkpoint["taxonomyAnchorsHash"],
    }
    for field, expected_value in expected.items():
        if checkpoint.get(field) != expected_value:
            raise ScenarioBlueprintError(f"blueprint checkpoint binding mismatch: {field}")
    if not isinstance(checkpoint["modelCall"], dict):
        raise ScenarioBlueprintError("blueprint checkpoint modelCall must be an object")
    validate_blueprint_set(checkpoint["blueprintSet"], topic)
    _checkpoint_body_hash(checkpoint)
    return checkpoint


def generate_topic_blueprints(
    *,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    output_root: Path,
    planner: StrictSchemaModel,
    taxonomy_repair_planner: StrictSchemaModel | None = None,
    taxonomy_admission: Mapping[str, Any] | None = None,
    taxonomy_judge: Any | None = None,
    max_attempts: int = 4,
    max_taxonomy_repair_cycles: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    """Create or exactly resume one immutable joint Stage P checkpoint."""

    if not 1 <= max_attempts <= 12:
        raise ScenarioBlueprintError("max_attempts must be in [1,12]")
    topic_id = _identifier(topic.get("topicId"), "topic.topicId", 120)
    scenario_ids = scenario_ids_for_topic(topic_id)
    planner_binding = _model_binding(planner)
    from ground_truth_finetuning.training.scenario_taxonomy_admission_v5 import (
        admit_topic_taxonomy,
        validate_taxonomy_admission_checkpoint,
    )

    if taxonomy_admission is not None and taxonomy_judge is not None:
        raise ScenarioBlueprintError(
            "provide either a taxonomy admission checkpoint or a taxonomy judge, not both"
        )
    if taxonomy_admission is None:
        if taxonomy_judge is None:
            raise ScenarioBlueprintError(
                "Stage P requires an independently judged taxonomy admission"
            )
        taxonomy_checkpoint = admit_topic_taxonomy(
            request=request,
            topic=topic,
            output_root=output_root,
            planner=planner,
            repair_planner=taxonomy_repair_planner,
            judge=taxonomy_judge,
            max_attempts=max_attempts,
            max_repair_cycles=max_taxonomy_repair_cycles,
            resume=resume,
        )
    else:
        taxonomy_checkpoint = validate_taxonomy_admission_checkpoint(
            taxonomy_admission, request, topic, planner_binding
        )
    taxonomy_anchors = taxonomy_checkpoint["taxonomyAnchors"]
    schema = build_blueprint_response_schema(topic, scenario_ids, taxonomy_anchors)
    stage_key = _topic_stage_key(
        request, topic, schema, planner_binding, taxonomy_checkpoint
    )
    path = _checkpoint_path(Path(output_root), "blueprints", topic_id, stage_key)
    if path.exists():
        if not resume:
            raise ScenarioBlueprintError(f"blueprint checkpoint exists; use --resume: {path}")
        return _validate_blueprint_checkpoint(
            read_json(path), request, topic, schema, planner_binding, taxonomy_checkpoint
        )

    context = _blueprint_context(request, topic, scenario_ids, taxonomy_checkpoint)
    failures: list[str] = []
    output_token_budget = BLUEPRINT_INITIAL_OUTPUT_TOKENS
    for attempt in range(1, max_attempts + 1):
        attempt_context = dict(context)
        if failures:
            attempt_context["retryFeedback"] = {
                "attempt": attempt,
                "previousDefect": failures[-1],
                "directive": (
                    "Regenerate the complete joint set while specifically correcting this defect; "
                    "do not repeat it or relax any schema-bound coverage cell."
                ),
            }
        try:
            response, metadata = _call_model(
                planner,
                name="personaplex_scenario_blueprints_v5",
                schema=schema,
                instructions=STAGE_P_SYSTEM,
                context=attempt_context,
                max_output_tokens=output_token_budget,
            )
            blueprints = decode_blueprint_response(
                response, topic, scenario_ids, taxonomy_anchors
            )
            blueprint_set = make_blueprint_set(topic, blueprints, planner_binding)
            body = {
                "schema": BLUEPRINT_CHECKPOINT_SCHEMA,
                "stageKey": stage_key,
                "requestHash": content_hash(request),
                "topicId": topic_id,
                "topicCardHash": content_hash(topic),
                "responseSchemaHash": content_hash(schema),
                "plannerBindingHash": content_hash(planner_binding),
                "plannerBinding": planner_binding,
                "taxonomyAdmissionCheckpointHash": taxonomy_checkpoint["checkpointHash"],
                "taxonomyAdmissionProtocolHash": taxonomy_checkpoint["protocolHash"],
                "taxonomyPlannerBindingHash": taxonomy_checkpoint[
                    "taxonomyPlannerBindingHash"
                ],
                "taxonomyJudgeModelBindingHash": taxonomy_checkpoint[
                    "taxonomyJudgeModelBindingHash"
                ],
                "taxonomySourceCheckpointHash": taxonomy_checkpoint[
                    "admittedSourceCheckpointHash"
                ],
                "taxonomyJudgeSourceHash": taxonomy_checkpoint[
                    "taxonomyJudgeSourceHash"
                ],
                "taxonomyRepairSourceHash": taxonomy_checkpoint[
                    "taxonomyRepairSourceHash"
                ],
                "taxonomyAnchorsHash": taxonomy_checkpoint["taxonomyAnchorsHash"],
                "modelCall": metadata,
                "blueprintSet": blueprint_set,
            }
            checkpoint = dict(body)
            checkpoint["checkpointHash"] = content_hash(body)
            _write_immutable_json(path, checkpoint)
            return checkpoint
        except ModelTransportUnavailable:
            raise
        except TruncatedModelOutput as error:
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
            output_token_budget = min(
                BLUEPRINT_MAX_OUTPUT_TOKENS,
                max(output_token_budget + 1, output_token_budget * 2),
            )
        except Exception as error:
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
    raise ScenarioBlueprintError(
        f"Stage P exhausted {max_attempts} exact attempts for {topic_id}: " + " | ".join(failures)
    )


def _validate_worker_count(max_workers: int) -> None:
    if not 1 <= max_workers <= MAX_WORKERS:
        raise ScenarioBlueprintError("max_workers must be in [1,3]")


def _parallel(
    items: Sequence[Any],
    worker: Callable[[Any], dict[str, Any]],
    *,
    max_workers: int,
    identity: Callable[[Any], str],
) -> list[dict[str, Any]]:
    _validate_worker_count(max_workers)
    results: list[dict[str, Any]] = []
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(worker, item): identity(item) for item in items}
    try:
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except BaseException as error:
                for pending in futures:
                    pending.cancel()
                raise ScenarioBlueprintError(f"stage failed for {futures[future]}: {error}") from error
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return results


def validate_corpus_shape(request: Mapping[str, Any], topics: Sequence[Mapping[str, Any]]) -> None:
    coverage = request.get("coverageTarget")
    if not isinstance(coverage, Mapping):
        raise ScenarioBlueprintError("request.coverageTarget is required")
    if coverage.get("candidateTopics") != TOPICS_PER_CORPUS:
        raise ScenarioBlueprintError("v5 blueprint corpus requires candidateTopics=50")
    if coverage.get("scenariosPerTopic") != BLUEPRINTS_PER_TOPIC:
        raise ScenarioBlueprintError("v5 blueprint corpus requires scenariosPerTopic=20")
    if len(topics) != TOPICS_PER_CORPUS:
        raise ScenarioBlueprintError("v5 blueprint corpus requires exactly fifty topic cards")
    topic_ids = [_identifier(topic.get("topicId"), "topic.topicId", 120) for topic in topics]
    if len(set(topic_ids)) != len(topic_ids):
        raise ScenarioBlueprintError("topic cards must have unique topicId values")
    for topic in topics:
        _topic_enums(topic)


def generate_blueprints(
    *,
    request: Mapping[str, Any],
    topics: Sequence[Mapping[str, Any]],
    output_root: Path,
    planner: StrictSchemaModel,
    taxonomy_judge: Any | None = None,
    max_workers: int = MAX_WORKERS,
    max_attempts: int = 4,
    max_taxonomy_repair_cycles: int = 4,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run Stage P for all fifty topics and emit a complete immutable set JSONL."""

    validate_corpus_shape(request, topics)
    ordered_topics = sorted(topics, key=lambda item: str(item["topicId"]))
    checkpoints = _parallel(
        ordered_topics,
        lambda topic: generate_topic_blueprints(
            request=request,
            topic=topic,
            output_root=Path(output_root),
            planner=planner,
            taxonomy_judge=taxonomy_judge,
            max_attempts=max_attempts,
            max_taxonomy_repair_cycles=max_taxonomy_repair_cycles,
            resume=resume,
        ),
        max_workers=max_workers,
        identity=lambda topic: str(topic["topicId"]),
    )
    sets = sorted(
        (checkpoint["blueprintSet"] for checkpoint in checkpoints),
        key=lambda item: item["topicId"],
    )
    topic_by_id = {str(topic["topicId"]): topic for topic in topics}
    if len(sets) != TOPICS_PER_CORPUS:
        raise ScenarioBlueprintError("Stage P did not produce exactly fifty blueprint sets")
    for blueprint_set in sets:
        validate_blueprint_set(blueprint_set, topic_by_id[blueprint_set["topicId"]])
    _write_immutable_jsonl(
        Path(output_root) / BLUEPRINT_SETS_FILENAME, sets, resume=resume
    )
    return sets


def load_blueprint_sets(
    output_root: Path,
    request: Mapping[str, Any],
    topics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    validate_corpus_shape(request, topics)
    path = Path(output_root) / BLUEPRINT_SETS_FILENAME
    if not path.is_file():
        raise ScenarioBlueprintError("complete Stage P artifact is required before Stage E")
    rows = read_jsonl(path)
    if len(rows) != TOPICS_PER_CORPUS:
        raise ScenarioBlueprintError("Stage P artifact must contain exactly fifty lines")
    topic_by_id = {str(topic["topicId"]): topic for topic in topics}
    if set(topic_by_id) != {row.get("topicId") for row in rows}:
        raise ScenarioBlueprintError("Stage P topic cardinality/binding mismatch")
    for row in rows:
        validate_blueprint_set(row, topic_by_id[row["topicId"]])
    return sorted(rows, key=lambda item: item["topicId"])


def _blueprint_claim_schema_variants(
    scenario_ids: Sequence[str], *, include_rationale: bool
) -> list[dict[str, Any]]:
    ids = list(scenario_ids)
    variants: list[dict[str, Any]] = []
    for code in BLUEPRINT_FINDING_CODES:
        cardinality = BLUEPRINT_FINDING_CARDINALITIES[code]
        required = ["code", "scenarioIds"]
        properties: dict[str, Any] = {
            "code": {"const": code},
            "scenarioIds": {
                "type": "array",
                "minItems": cardinality["minimum"],
                "maxItems": cardinality["maximum"],
                "uniqueItems": True,
                "items": {"type": "string", "enum": ids},
            },
        }
        if include_rationale:
            required.append("rationale")
            properties["rationale"] = {
                "type": "string",
                "minLength": 3,
            }
        variants.append(
            {
                "type": "object",
                "additionalProperties": False,
                "required": required,
                "properties": properties,
            }
        )
    return variants


def build_blueprint_proposer_response_schema(
    scenario_ids: Sequence[str],
) -> dict[str, Any]:
    ids = tuple(scenario_ids)
    if len(ids) != BLUEPRINTS_PER_TOPIC or len(set(ids)) != len(ids):
        raise ScenarioBlueprintError(
            "blueprint proposal requires exactly twenty unique scenario IDs"
        )
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["findings"],
        "properties": {
            "findings": {
                "type": "array",
                "maxItems": BLUEPRINT_MAX_PROPOSER_CLAIMS,
                "uniqueItems": True,
                "items": {
                    "oneOf": _blueprint_claim_schema_variants(
                        ids, include_rationale=False
                    )
                },
            }
        },
    }
    Draft202012Validator.check_schema(schema)
    return schema


def validate_blueprint_proposal(
    value: Any, scenario_ids: Sequence[str]
) -> dict[str, Any]:
    ids = tuple(scenario_ids)
    _raise_schema_errors(
        value,
        build_blueprint_proposer_response_schema(ids),
        "whole-blueprint typed proposal",
    )
    assert_no_target_fields(value)
    assert isinstance(value, dict)
    order = {scenario_id: ordinal for ordinal, scenario_id in enumerate(ids)}
    claims: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for claim in value["findings"]:
        claim_ids = tuple(sorted(claim["scenarioIds"], key=order.__getitem__))
        key = (claim["code"], claim_ids)
        if key in seen:
            raise InvalidModelOutput(
                "whole-blueprint proposal repeats a typed finding claim"
            )
        seen.add(key)
        claims.append({"code": claim["code"], "scenarioIds": list(claim_ids)})
    claims.sort(key=canonical_json)
    return {"findings": claims}


def build_blueprint_judge_response_schema(scenario_ids: Sequence[str]) -> dict[str, Any]:
    ids = tuple(scenario_ids)
    if len(ids) != BLUEPRINTS_PER_TOPIC or len(set(ids)) != len(ids):
        raise ScenarioBlueprintError("blueprint scrutiny requires exactly twenty unique scenario IDs")
    verdict = {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "rationale"],
        "properties": {
            "status": {"enum": ["pass", "fail"]},
            "rationale": _semantic_text(),
        },
    }
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["groupDecision", "groupRationale", "dimensions", "findings"],
        "properties": {
            "groupDecision": {"enum": ["pass", "reject"]},
            "groupRationale": _semantic_text(),
            "dimensions": {
                "type": "object",
                "additionalProperties": False,
                "required": list(BLUEPRINT_JUDGE_DIMENSIONS),
                "properties": {dimension: verdict for dimension in BLUEPRINT_JUDGE_DIMENSIONS},
            },
            "findings": {
                "type": "array",
                "maxItems": BLUEPRINT_MAX_FINAL_FINDINGS,
                "uniqueItems": True,
                "items": {
                    "oneOf": _blueprint_claim_schema_variants(
                        ids, include_rationale=True
                    )
                },
            },
        },
    }
    Draft202012Validator.check_schema(schema)
    return schema


def validate_blueprint_judgment(
    value: Any, scenario_ids: Sequence[str]
) -> dict[str, Any]:
    schema = build_blueprint_judge_response_schema(scenario_ids)
    _raise_schema_errors(value, schema, "whole-blueprint judgment")
    assert_no_target_fields(value)
    assert isinstance(value, dict)
    order = {scenario_id: ordinal for ordinal, scenario_id in enumerate(scenario_ids)}
    normalized = json.loads(canonical_json(value))
    cluster_keys: set[tuple[str, tuple[str, ...]]] = set()
    for finding in normalized["findings"]:
        finding_ids = tuple(
            sorted(finding["scenarioIds"], key=order.__getitem__)
        )
        key = (finding["code"], finding_ids)
        if key in cluster_keys:
            raise InvalidModelOutput("whole-blueprint judgment repeats a finding cluster")
        cluster_keys.add(key)
        finding["scenarioIds"] = list(finding_ids)
    normalized["findings"].sort(key=canonical_json)
    # Finding clusters are the sole admission signal. Models frequently emit a
    # semantically useful finding while leaving a redundant top-level enum at
    # "pass"; retrying that authentic judgment loses evidence and cannot improve
    # the blueprint. Normalize redundant status fields from the findings instead.
    normalized["groupDecision"] = "reject" if normalized["findings"] else "pass"
    failed_dimensions = {
        dimension
        for finding in normalized["findings"]
        for dimension in BLUEPRINT_FINDING_DIMENSIONS[finding["code"]]
    }
    for dimension in BLUEPRINT_JUDGE_DIMENSIONS:
        normalized["dimensions"][dimension]["status"] = (
            "fail" if dimension in failed_dimensions else "pass"
        )
    return normalized


def _verified_blueprint_judgment(
    confirmed_claims: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    claims = [json.loads(canonical_json(dict(claim))) for claim in confirmed_claims]
    claims.sort(key=canonical_json)
    codes = sorted({claim["code"] for claim in claims})
    failed_dimensions = {
        dimension
        for claim in claims
        for dimension in BLUEPRINT_FINDING_DIMENSIONS[claim["code"]]
    }
    if codes:
        group_rationale = "Verifier-confirmed finding codes: " + ", ".join(codes)
    else:
        group_rationale = (
            "No typed whole-blueprint claim was confirmed by the evidence-bound verifier."
        )
    dimensions: dict[str, dict[str, str]] = {}
    for dimension in BLUEPRINT_JUDGE_DIMENSIONS:
        dimension_codes = sorted(
            code
            for code in codes
            if dimension in BLUEPRINT_FINDING_DIMENSIONS[code]
        )
        dimensions[dimension] = {
            "status": "fail" if dimension_codes else "pass",
            "rationale": (
                "Verifier-confirmed finding codes: " + ", ".join(dimension_codes)
                if dimension_codes
                else f"No verifier-confirmed finding code maps to {dimension}."
            ),
        }
    return {
        "groupDecision": "reject" if claims else "pass",
        "groupRationale": group_rationale,
        "dimensions": dimensions,
        "findings": [
            {
                "code": claim["code"],
                "scenarioIds": claim["scenarioIds"],
                "rationale": BLUEPRINT_FINDING_DEFINITIONS[claim["code"]],
            }
            for claim in claims
        ],
    }


class WholeBlueprintJudge(Protocol):
    def binding(self) -> Mapping[str, Any]:
        ...

    def judge_topic(
        self, topic: Mapping[str, Any], blueprint_set: Mapping[str, Any]
    ) -> Any:
        ...


class AuthenticWholeBlueprintJudge:
    """One independent strict-schema proposer for a complete topic blueprint."""

    def __init__(self, model: StrictSchemaModel) -> None:
        self.model = model

    def binding(self) -> dict[str, Any]:
        return {
            "protocol": "independent_whole_blueprint_typed_proposer_v1",
            "protocolVersion": BLUEPRINT_JUDGE_PROTOCOL_VERSION,
            "protocolHash": blueprint_judge_protocol_hash(),
            "sourceHash": BLUEPRINT_JUDGE_SOURCE_HASH,
            "proposerPromptHash": BLUEPRINT_PROPOSER_PROMPT_HASH,
            "cardinalityHash": BLUEPRINT_FINDING_CARDINALITY_HASH,
            "modelBinding": _model_binding(self.model),
            "reasoning": {"enabled": False},
            "responseFormat": "strict_json_schema",
            "maxOutputTokens": BLUEPRINT_JUDGE_MAX_OUTPUT_TOKENS,
        }

    def judge_topic(
        self, topic: Mapping[str, Any], blueprint_set: Mapping[str, Any]
    ) -> tuple[Any, Mapping[str, Any]]:
        scenario_ids = blueprint_set["scenarioIds"]
        schema = build_blueprint_proposer_response_schema(scenario_ids)
        context = {
            "task": "Propose exact typed findings for one complete twenty-slot blueprint.",
            "topicCard": dict(topic),
            "jointCompactBlueprint": dict(blueprint_set),
            "typedFindingContract": {
                "soleSemanticRejectionSignal": "findings",
                "codes": list(BLUEPRINT_FINDING_CODES),
                "definitions": dict(BLUEPRINT_FINDING_DEFINITIONS),
                "scenarioIdCardinalities": dict(BLUEPRINT_FINDING_CARDINALITIES),
                "exactSourceIdsRequired": True,
                "proposerRationaleAllowed": False,
            },
        }
        return _call_model(
            self.model,
            name="personaplex_whole_blueprint_finding_proposer_v5",
            schema=schema,
            instructions=BLUEPRINT_PROPOSER_SYSTEM,
            context=context,
            max_output_tokens=BLUEPRINT_JUDGE_MAX_OUTPUT_TOKENS,
        )


class AdjudicatedWholeBlueprintJudge:
    """Two typed proposers plus immutable evidence-local claim verification."""

    _VERIFICATION_SCHEMA = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["confirmed"],
        "properties": {"confirmed": {"type": "boolean"}},
    }

    def __init__(
        self,
        primary: WholeBlueprintJudge,
        secondary: WholeBlueprintJudge,
        verifier: StrictSchemaModel,
        *,
        checkpoint_root: Path,
        max_workers: int = MAX_WORKERS,
    ) -> None:
        if not 1 <= max_workers <= MAX_WORKERS:
            raise ScenarioBlueprintError(
                "whole-blueprint claim verifier workers must be in [1,3]"
            )
        self.primary = primary
        self.secondary = secondary
        self.verifier = verifier
        self.checkpoint_root = Path(checkpoint_root)
        self.max_workers = max_workers
        self._verifier_slots = BoundedSemaphore(max_workers)

        primary_binding = _judge_binding(primary)
        secondary_binding = _judge_binding(secondary)
        verifier_binding = _model_binding(verifier)
        role_model_bindings = {
            "primaryProposer": dict(_judge_model_binding(primary_binding)),
            "secondaryProposer": dict(_judge_model_binding(secondary_binding)),
            "evidenceBoundVerifier": verifier_binding,
        }
        role_model_binding_hashes = {
            role: content_hash(binding)
            for role, binding in role_model_bindings.items()
        }
        if len({_judge_model_identity(binding) for binding in role_model_bindings.values()}) != 3:
            raise ScenarioBlueprintError(
                "whole-blueprint proposers and verifier must have distinct model bindings"
            )
        self._binding = {
            "protocol": "two_proposer_source_bound_whole_blueprint_claim_verification_v1",
            "protocolVersion": BLUEPRINT_JUDGE_PROTOCOL_VERSION,
            "protocolHash": blueprint_judge_protocol_hash(),
            "sourceHash": BLUEPRINT_JUDGE_SOURCE_HASH,
            "promptHashes": {
                "primaryProposer": BLUEPRINT_PROPOSER_PROMPT_HASH,
                "secondaryProposer": BLUEPRINT_PROPOSER_PROMPT_HASH,
                "evidenceBoundVerifier": BLUEPRINT_VERIFIER_PROMPT_HASH,
            },
            "cardinalityHash": BLUEPRINT_FINDING_CARDINALITY_HASH,
            "modelBinding": {
                "protocol": "composite_whole_blueprint_adjudication",
                "roleModelBindingHashes": role_model_binding_hashes,
                "reasoning": {"enabled": False},
                "responseFormat": "strict_json_schema",
            },
            "roleModelBindings": role_model_bindings,
            "memberModelBindings": list(role_model_bindings.values()),
            "primaryProposer": primary_binding,
            "secondaryProposer": secondary_binding,
            "evidenceBoundVerifier": verifier_binding,
            "reasoning": {"enabled": False},
            "responseFormat": "strict_json_schema",
        }

    def binding(self) -> dict[str, Any]:
        return json.loads(canonical_json(self._binding))

    def _verification_path(
        self,
        topic: Mapping[str, Any],
        claim: Mapping[str, Any],
        evidence: Mapping[str, Any],
        proposal_schema_hash: str,
    ) -> tuple[str, Path]:
        key = content_hash(
            {
                "protocolVersion": BLUEPRINT_JUDGE_PROTOCOL_VERSION,
                "protocolHash": blueprint_judge_protocol_hash(),
                "sourceHash": BLUEPRINT_JUDGE_SOURCE_HASH,
                "proposerPromptHash": BLUEPRINT_PROPOSER_PROMPT_HASH,
                "verifierPromptHash": BLUEPRINT_VERIFIER_PROMPT_HASH,
                "cardinalityHash": BLUEPRINT_FINDING_CARDINALITY_HASH,
                "proposalResponseSchemaHash": proposal_schema_hash,
                "verificationResponseSchema": self._VERIFICATION_SCHEMA,
                "topicCard": topic,
                "claim": claim,
                "sourceBoundEvidence": evidence,
                "verifierRole": "evidenceBoundVerifier",
                "verifierBinding": _model_binding(self.verifier),
            }
        )
        return (
            key,
            self.checkpoint_root / str(topic["topicId"]) / f"{key[7:]}.json",
        )

    def _verify_claim(
        self,
        topic: Mapping[str, Any],
        blueprint_set: Mapping[str, Any],
        claim: Mapping[str, Any],
    ) -> dict[str, Any]:
        evidence = {
            "scenarioIds": list(claim["scenarioIds"]),
            "blueprints": {
                scenario_id: {
                    "blueprintHash": blueprint_set["blueprintHashes"][scenario_id],
                    "blueprint": blueprint_set["blueprints"][scenario_id],
                }
                for scenario_id in claim["scenarioIds"]
            },
        }
        proposal_schema_hash = content_hash(
            build_blueprint_proposer_response_schema(blueprint_set["scenarioIds"])
        )
        verification_schema_hash = content_hash(self._VERIFICATION_SCHEMA)
        verifier_binding = _model_binding(self.verifier)
        key, path = self._verification_path(
            topic, claim, evidence, proposal_schema_hash
        )
        expected = {
            "schema": BLUEPRINT_CLAIM_VERIFICATION_CHECKPOINT_SCHEMA,
            "verificationKey": key,
            "protocolVersion": BLUEPRINT_JUDGE_PROTOCOL_VERSION,
            "protocolHash": blueprint_judge_protocol_hash(),
            "sourceHash": BLUEPRINT_JUDGE_SOURCE_HASH,
            "proposerPromptHash": BLUEPRINT_PROPOSER_PROMPT_HASH,
            "verifierPromptHash": BLUEPRINT_VERIFIER_PROMPT_HASH,
            "cardinalityHash": BLUEPRINT_FINDING_CARDINALITY_HASH,
            "proposalResponseSchemaHash": proposal_schema_hash,
            "verificationResponseSchemaHash": verification_schema_hash,
            "topicId": topic["topicId"],
            "topicCardHash": content_hash(topic),
            "sourceBoundScenarioIds": list(claim["scenarioIds"]),
            "sourceBoundEvidenceHash": content_hash(evidence),
            "claim": dict(claim),
            "claimHash": content_hash(claim),
            "verifierRole": "evidenceBoundVerifier",
            "verifierBinding": verifier_binding,
            "verifierBindingHash": content_hash(verifier_binding),
        }
        required = set(expected) | {"modelCall", "confirmed", "checkpointHash"}
        if path.exists():
            checkpoint = read_json(path)
            if set(checkpoint) != required:
                raise ScenarioBlueprintError(
                    f"whole-blueprint claim-verification checkpoint field mismatch: {path}"
                )
            body = dict(checkpoint)
            checkpoint_hash = body.pop("checkpointHash", None)
            if checkpoint_hash != content_hash(body):
                raise ScenarioBlueprintError(
                    f"whole-blueprint claim-verification checkpoint hash mismatch: {path}"
                )
            for field, expected_value in expected.items():
                if checkpoint.get(field) != expected_value:
                    raise ScenarioBlueprintError(
                        "whole-blueprint claim-verification checkpoint identity "
                        f"mismatch: {field}"
                    )
            if not isinstance(checkpoint.get("modelCall"), dict) or type(
                checkpoint.get("confirmed")
            ) is not bool:
                raise ScenarioBlueprintError(
                    f"whole-blueprint claim-verification checkpoint is malformed: {path}"
                )
            return checkpoint

        with self._verifier_slots:
            response, metadata = _call_model(
                self.verifier,
                name="personaplex_whole_blueprint_claim_verifier_v5",
                schema=self._VERIFICATION_SCHEMA,
                instructions=BLUEPRINT_CLAIM_VERIFIER_SYSTEM,
                context={
                    "task": "Verify exactly one untrusted typed whole-blueprint finding.",
                    "topicCard": dict(topic),
                    "proposedFinding": dict(claim),
                    "definition": BLUEPRINT_FINDING_DEFINITIONS[claim["code"]],
                    "requiredScenarioIdCardinality": dict(
                        BLUEPRINT_FINDING_CARDINALITIES[claim["code"]]
                    ),
                    "sourceBoundBlueprintEvidence": evidence,
                    "rules": [
                        "Return only confirmed for the proposed code and exact IDs.",
                        "Do not inspect or infer evidence from an unbound sibling.",
                        "Do not search for or return any unproposed finding.",
                    ],
                },
                max_output_tokens=BLUEPRINT_CLAIM_VERIFIER_MAX_OUTPUT_TOKENS,
            )
        _raise_schema_errors(
            response,
            self._VERIFICATION_SCHEMA,
            "whole-blueprint claim verification",
        )
        body = {
            **expected,
            "modelCall": metadata,
            "confirmed": bool(response["confirmed"]),
        }
        checkpoint = dict(body)
        checkpoint["checkpointHash"] = content_hash(body)
        _write_immutable_json(path, checkpoint)
        return checkpoint

    def judge_topic(
        self, topic: Mapping[str, Any], blueprint_set: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        validate_blueprint_set(blueprint_set, topic)
        with ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="whole-blueprint-proposer"
        ) as pool:
            primary_future = pool.submit(
                _call_judge, self.primary, topic, blueprint_set
            )
            secondary_future = pool.submit(
                _call_judge, self.secondary, topic, blueprint_set
            )
            primary_raw, primary_metadata = primary_future.result()
            secondary_raw, secondary_metadata = secondary_future.result()

        scenario_ids = blueprint_set["scenarioIds"]
        primary = validate_blueprint_proposal(primary_raw, scenario_ids)
        secondary = validate_blueprint_proposal(secondary_raw, scenario_ids)
        claims = {
            canonical_json(claim): claim
            for proposal in (primary, secondary)
            for claim in proposal["findings"]
        }
        ordered_claims = [claims[key] for key in sorted(claims)]
        checkpoints: list[dict[str, Any]] = []
        if ordered_claims:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(ordered_claims)),
                thread_name_prefix="whole-blueprint-claim-verifier",
            ) as pool:
                checkpoints = list(
                    pool.map(
                        lambda claim: self._verify_claim(
                            topic, blueprint_set, claim
                        ),
                        ordered_claims,
                    )
                )
        confirmed = [
            dict(checkpoint["claim"])
            for checkpoint in checkpoints
            if checkpoint["confirmed"]
        ]
        confirmed.sort(key=canonical_json)
        decision = validate_blueprint_judgment(
            _verified_blueprint_judgment(confirmed), scenario_ids
        )
        return decision, {
            "protocol": self._binding["protocol"],
            "protocolVersion": BLUEPRINT_JUDGE_PROTOCOL_VERSION,
            "protocolHash": blueprint_judge_protocol_hash(),
            "sourceHash": BLUEPRINT_JUDGE_SOURCE_HASH,
            "proposerPromptHash": BLUEPRINT_PROPOSER_PROMPT_HASH,
            "verifierPromptHash": BLUEPRINT_VERIFIER_PROMPT_HASH,
            "cardinalityHash": BLUEPRINT_FINDING_CARDINALITY_HASH,
            "proposalResponseSchemaHash": content_hash(
                build_blueprint_proposer_response_schema(scenario_ids)
            ),
            "verificationResponseSchemaHash": content_hash(
                self._VERIFICATION_SCHEMA
            ),
            "primaryModelCall": primary_metadata,
            "secondaryModelCall": secondary_metadata,
            "primaryFindings": primary["findings"],
            "secondaryFindings": secondary["findings"],
            "proposedClaimCount": len(ordered_claims),
            "confirmedClaimCount": len(confirmed),
            "verificationCheckpointHashes": [
                checkpoint["checkpointHash"] for checkpoint in checkpoints
            ],
        }


def _judge_binding(judge: WholeBlueprintJudge) -> dict[str, Any]:
    value = judge.binding()
    if not isinstance(value, Mapping):
        raise ScenarioBlueprintError("whole-blueprint judge binding must be an object")
    return json.loads(canonical_json(dict(value)))


def _judge_model_binding(binding: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = binding.get("modelBinding")
    return nested if isinstance(nested, Mapping) else binding


def _judge_model_identity(binding: Mapping[str, Any]) -> str:
    protocol = binding.get("protocol")
    model = binding.get("model")
    if isinstance(protocol, str) and protocol and isinstance(model, str) and model:
        return canonical_json({"protocol": protocol, "model": model})
    return content_hash(binding)


def _judge_member_model_bindings(
    binding: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    role_bindings = binding.get("roleModelBindings")
    if isinstance(role_bindings, Mapping):
        required_roles = (
            "primaryProposer",
            "secondaryProposer",
            "evidenceBoundVerifier",
        )
        if set(role_bindings) != set(required_roles) or any(
            not isinstance(role_bindings.get(role), Mapping)
            for role in required_roles
        ):
            raise ScenarioBlueprintError(
                "whole-blueprint adjudicator role bindings are malformed"
            )
        return [role_bindings[role] for role in required_roles]
    members = binding.get("memberModelBindings")
    if isinstance(members, list):
        if not members or any(not isinstance(member, Mapping) for member in members):
            raise ScenarioBlueprintError(
                "whole-blueprint adjudicator member bindings are malformed"
            )
        return list(members)
    return [_judge_model_binding(binding)]


def _call_judge(
    judge: WholeBlueprintJudge,
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    result = judge.judge_topic(topic, blueprint_set)
    if isinstance(result, tuple) and len(result) == 2:
        decision, metadata = result
    else:
        decision, metadata = result, {}
    if not isinstance(metadata, Mapping):
        raise InvalidModelOutput("whole-blueprint judge metadata must be an object")
    return decision, json.loads(canonical_json(dict(metadata)))


def _scrutiny_stage_key(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    response_schema: Mapping[str, Any],
    judge_binding: Mapping[str, Any],
) -> str:
    return content_hash(
        {
            "stage": "whole_blueprint_scrutiny_v5",
            "protocolVersion": BLUEPRINT_JUDGE_PROTOCOL_VERSION,
            "requestHash": content_hash(request),
            "topicCardHash": content_hash(topic),
            "blueprintSetHash": content_hash(blueprint_set),
            "jointBlueprintHash": blueprint_set["jointBlueprintHash"],
            "responseSchemaHash": content_hash(response_schema),
            "judgeBindingHash": content_hash(judge_binding),
        }
    )


def validate_blueprint_scrutiny(
    checkpoint: Any,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    *,
    require_pass: bool = True,
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ScenarioBlueprintError("whole-blueprint scrutiny artifact must be an object")
    required = {
        "schema",
        "stageKey",
        "requestHash",
        "topicId",
        "topicCardHash",
        "blueprintSetHash",
        "jointBlueprintHash",
        "blueprintPlannerBinding",
        "judgeBinding",
        "judgeBindingHash",
        "responseSchemaHash",
        "modelCall",
        "decision",
        "checkpointHash",
    }
    if set(checkpoint) != required:
        raise ScenarioBlueprintError("whole-blueprint scrutiny artifact has an invalid field set")
    validate_blueprint_set(blueprint_set, topic)
    scenario_ids = blueprint_set["scenarioIds"]
    response_schema = build_blueprint_judge_response_schema(scenario_ids)
    expected = {
        "schema": BLUEPRINT_SCRUTINY_SCHEMA,
        "requestHash": content_hash(request),
        "topicId": topic["topicId"],
        "topicCardHash": content_hash(topic),
        "blueprintSetHash": content_hash(blueprint_set),
        "jointBlueprintHash": blueprint_set["jointBlueprintHash"],
        "blueprintPlannerBinding": blueprint_set["blueprintPlannerBinding"],
        "judgeBindingHash": content_hash(checkpoint.get("judgeBinding")),
        "responseSchemaHash": content_hash(response_schema),
    }
    for field, expected_value in expected.items():
        if checkpoint.get(field) != expected_value:
            raise ScenarioBlueprintError(f"whole-blueprint scrutiny binding mismatch: {field}")
    if checkpoint.get("stageKey") != _scrutiny_stage_key(
        request, topic, blueprint_set, response_schema, checkpoint["judgeBinding"]
    ):
        raise ScenarioBlueprintError("whole-blueprint scrutiny stage key mismatch")
    if not isinstance(checkpoint.get("modelCall"), dict):
        raise ScenarioBlueprintError("whole-blueprint scrutiny modelCall must be an object")
    decision = validate_blueprint_judgment(checkpoint["decision"], scenario_ids)
    _checkpoint_body_hash(checkpoint)
    if require_pass and decision["groupDecision"] != "pass":
        raise ScenarioBlueprintError(
            f"Stage E refused non-passing blueprint for {topic['topicId']}"
        )
    return checkpoint


def generate_topic_blueprint_scrutiny(
    *,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    output_root: Path,
    judge: WholeBlueprintJudge,
    max_attempts: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    if not 1 <= max_attempts <= 12:
        raise ScenarioBlueprintError("max_attempts must be in [1,12]")
    validate_blueprint_set(blueprint_set, topic)
    judge_binding = _judge_binding(judge)
    planner_identity = _judge_model_identity(blueprint_set["blueprintPlannerBinding"])
    judge_identities = [
        _judge_model_identity(binding)
        for binding in _judge_member_model_bindings(judge_binding)
    ]
    if len(judge_identities) == 1 and judge_identities[0] == planner_identity:
        raise ScenarioBlueprintError(
            "whole-blueprint judge must be independent from the Stage P planner"
        )
    if len(judge_identities) >= 3 and (
        judge_identities[-1] == planner_identity
        or all(identity == planner_identity for identity in judge_identities[:-1])
    ):
        raise ScenarioBlueprintError(
            "whole-blueprint adjudication requires an independent proposer and final verifier"
        )
    response_schema = build_blueprint_judge_response_schema(blueprint_set["scenarioIds"])
    stage_key = _scrutiny_stage_key(
        request, topic, blueprint_set, response_schema, judge_binding
    )
    path = _checkpoint_path(
        Path(output_root), "blueprint_scrutiny", str(topic["topicId"]), stage_key
    )
    if path.exists():
        if not resume:
            raise ScenarioBlueprintError(f"blueprint scrutiny exists; use --resume: {path}")
        return validate_blueprint_scrutiny(
            read_json(path), request, topic, blueprint_set, require_pass=False
        )
    failures: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            raw_decision, metadata = _call_judge(judge, topic, blueprint_set)
            decision = validate_blueprint_judgment(
                raw_decision, blueprint_set["scenarioIds"]
            )
            body = {
                "schema": BLUEPRINT_SCRUTINY_SCHEMA,
                "stageKey": stage_key,
                "requestHash": content_hash(request),
                "topicId": topic["topicId"],
                "topicCardHash": content_hash(topic),
                "blueprintSetHash": content_hash(blueprint_set),
                "jointBlueprintHash": blueprint_set["jointBlueprintHash"],
                "blueprintPlannerBinding": blueprint_set["blueprintPlannerBinding"],
                "judgeBinding": judge_binding,
                "judgeBindingHash": content_hash(judge_binding),
                "responseSchemaHash": content_hash(response_schema),
                "modelCall": metadata,
                "decision": decision,
            }
            checkpoint = dict(body)
            checkpoint["checkpointHash"] = content_hash(body)
            _write_immutable_json(path, checkpoint)
            return checkpoint
        except ModelTransportUnavailable:
            raise
        except Exception as error:
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
    raise ScenarioBlueprintError(
        f"blueprint scrutiny exhausted {max_attempts} exact attempts for {topic['topicId']}: "
        + " | ".join(failures)
    )


def scrutinize_blueprints(
    *,
    request: Mapping[str, Any],
    topics: Sequence[Mapping[str, Any]],
    blueprint_sets: Sequence[Mapping[str, Any]],
    output_root: Path,
    judge: WholeBlueprintJudge,
    max_workers: int = MAX_WORKERS,
    max_attempts: int = 4,
    resume: bool = False,
) -> list[dict[str, Any]]:
    validate_corpus_shape(request, topics)
    topic_by_id = {str(topic["topicId"]): topic for topic in topics}
    set_by_topic = {str(item["topicId"]): item for item in blueprint_sets}
    if len(blueprint_sets) != TOPICS_PER_CORPUS or set(topic_by_id) != set(set_by_topic):
        raise ScenarioBlueprintError("blueprint scrutiny requires the exact fifty-topic set")
    assignments = [(topic_by_id[topic_id], set_by_topic[topic_id]) for topic_id in sorted(topic_by_id)]
    checkpoints = _parallel(
        assignments,
        lambda assignment: generate_topic_blueprint_scrutiny(
            request=request,
            topic=assignment[0],
            blueprint_set=assignment[1],
            output_root=Path(output_root),
            judge=judge,
            max_attempts=max_attempts,
            resume=resume,
        ),
        max_workers=max_workers,
        identity=lambda assignment: str(assignment[0]["topicId"]),
    )
    checkpoints.sort(key=lambda item: item["topicId"])
    for checkpoint in checkpoints:
        topic_id = checkpoint["topicId"]
        validate_blueprint_scrutiny(
            checkpoint, request, topic_by_id[topic_id], set_by_topic[topic_id], require_pass=True
        )
    _write_immutable_jsonl(
        Path(output_root) / BLUEPRINT_SCRUTINY_FILENAME, checkpoints, resume=resume
    )
    return checkpoints


def load_blueprint_scrutinies(
    output_root: Path,
    request: Mapping[str, Any],
    topics: Sequence[Mapping[str, Any]],
    blueprint_sets: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    path = Path(output_root) / BLUEPRINT_SCRUTINY_FILENAME
    if not path.is_file():
        raise ScenarioBlueprintError("independent whole-blueprint scrutiny is required before Stage E")
    rows = read_jsonl(path)
    topic_by_id = {str(topic["topicId"]): topic for topic in topics}
    set_by_topic = {str(item["topicId"]): item for item in blueprint_sets}
    if len(rows) != TOPICS_PER_CORPUS or {row.get("topicId") for row in rows} != set(topic_by_id):
        raise ScenarioBlueprintError("whole-blueprint scrutiny cardinality/binding mismatch")
    for row in rows:
        topic_id = row["topicId"]
        validate_blueprint_scrutiny(
            row, request, topic_by_id[topic_id], set_by_topic[topic_id], require_pass=True
        )
    return sorted(rows, key=lambda item: item["topicId"])


def _expansion_context(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    blueprint_scrutiny: Mapping[str, Any],
    scenario_id: str,
) -> dict[str, Any]:
    return {
        "task": "Expand exactly one assigned niche into personaplex.scenario-contract.v2.",
        "topicCard": dict(topic),
        "jointCompactBlueprint": dict(blueprint_set),
        "assignedScenarioId": scenario_id,
        "assignedBlueprintHash": blueprint_set["blueprintHashes"][scenario_id],
        "assignedJointBlueprintHash": blueprint_set["jointBlueprintHash"],
        "assignedBlueprint": blueprint_set["blueprints"][scenario_id],
        "independentBlueprintAdmission": {
            "scrutinyHash": blueprint_scrutiny["checkpointHash"],
            "judgeBinding": blueprint_scrutiny["judgeBinding"],
            "decision": blueprint_scrutiny["decision"],
        },
        "corpusConstraints": {
            "topicConstraints": request.get("topicConstraints", {}),
            "semanticControl": request.get("semanticControl", {}),
            "requiredControlCoverage": request.get("requiredControlCoverage", {}),
        },
        "expansionContract": [
            "Expand only assignedBlueprint; do not substitute or blend any sibling niche.",
            "Preserve four materially different verified-positive, verified-negative, uncertain, and superseded routes.",
            "Provide contextual facts, uncertainty, constraints, and opportunities without target dialogue.",
            "Echo every assigned identity and content hash exactly.",
        ],
    }


def _scenario_stage_key(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    blueprint_scrutiny: Mapping[str, Any],
    scenario_id: str,
    response_schema: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
) -> str:
    return content_hash(
        {
            "stage": "scenario_expansion_v5",
            "protocolVersion": SCENARIO_EXPANSION_PROTOCOL_VERSION,
            "stageSystemHash": content_hash(STAGE_E_SYSTEM),
            "requestHash": content_hash(request),
            "topicCardHash": content_hash(topic),
            "scenarioId": scenario_id,
            "blueprintHash": blueprint_set["blueprintHashes"][scenario_id],
            "jointBlueprintHash": blueprint_set["jointBlueprintHash"],
            "blueprintScrutinyHash": blueprint_scrutiny["checkpointHash"],
            "responseSchemaHash": content_hash(response_schema),
            "plannerBindingHash": content_hash(planner_binding),
        }
    )


def _validate_expansion_checkpoint(
    checkpoint: Any,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    blueprint_scrutiny: Mapping[str, Any],
    scenario_id: str,
    response_schema: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ScenarioBlueprintError("scenario checkpoint must be an object")
    required = {
        "schema",
        "stageKey",
        "requestHash",
        "topicId",
        "topicCardHash",
        "scenarioId",
        "blueprintHash",
        "jointBlueprintHash",
        "blueprintScrutinyHash",
        "responseSchemaHash",
        "plannerBindingHash",
        "plannerBinding",
        "modelCall",
        "scenarioContract",
        "scenarioContractHash",
        "checkpointHash",
    }
    if set(checkpoint) != required:
        raise ScenarioBlueprintError("scenario checkpoint has an invalid field set")
    expected = {
        "schema": EXPANSION_CHECKPOINT_SCHEMA,
        "stageKey": _scenario_stage_key(
            request,
            topic,
            blueprint_set,
            blueprint_scrutiny,
            scenario_id,
            response_schema,
            planner_binding,
        ),
        "requestHash": content_hash(request),
        "topicId": topic["topicId"],
        "topicCardHash": content_hash(topic),
        "scenarioId": scenario_id,
        "blueprintHash": blueprint_set["blueprintHashes"][scenario_id],
        "jointBlueprintHash": blueprint_set["jointBlueprintHash"],
        "blueprintScrutinyHash": blueprint_scrutiny["checkpointHash"],
        "responseSchemaHash": content_hash(response_schema),
        "plannerBindingHash": content_hash(planner_binding),
        "plannerBinding": dict(planner_binding),
    }
    for field, expected_value in expected.items():
        if checkpoint.get(field) != expected_value:
            raise ScenarioBlueprintError(f"scenario checkpoint binding mismatch: {field}")
    scenario = checkpoint.get("scenarioContract")
    _raise_schema_errors(
        scenario,
        build_scenario_contract_schema(
            scenario_id,
            str(topic["topicId"]),
            blueprint_set["blueprints"][scenario_id],
        ),
        "scenario contract",
    )
    assert_no_target_fields(scenario)
    if checkpoint.get("scenarioContractHash") != content_hash(scenario):
        raise ScenarioBlueprintError("scenario contract hash mismatch")
    if not isinstance(checkpoint.get("modelCall"), dict):
        raise ScenarioBlueprintError("scenario checkpoint modelCall must be an object")
    _checkpoint_body_hash(checkpoint)
    return checkpoint


class _PremiseRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._owners: dict[str, str] = {}

    def add_existing(self, scenario_id: str, premise: str) -> None:
        with self._lock:
            owner = self._owners.get(premise)
            if owner is not None and owner != scenario_id:
                raise ScenarioBlueprintError(
                    f"exact premise collision in resumed checkpoints: {owner} and {scenario_id}"
                )
            self._owners[premise] = scenario_id

    def persist(
        self,
        scenario_id: str,
        premise: str,
        writer: Callable[[], None],
    ) -> None:
        with self._lock:
            owner = self._owners.get(premise)
            if owner is not None and owner != scenario_id:
                raise InvalidModelOutput(
                    f"exact premise collision with already admitted scenario {owner}"
                )
            writer()
            self._owners[premise] = scenario_id


def _expansion_parameters(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    blueprint_scrutiny: Mapping[str, Any],
    scenario_id: str,
    planner: StrictSchemaModel,
) -> tuple[dict[str, Any], dict[str, Any], str, Path]:
    validate_blueprint_set(blueprint_set, topic)
    validate_blueprint_scrutiny(
        blueprint_scrutiny, request, topic, blueprint_set, require_pass=True
    )
    if scenario_id not in blueprint_set["scenarioIds"]:
        raise ScenarioBlueprintError("assigned scenario ID is absent from the joint blueprint")
    blueprint_hash = blueprint_set["blueprintHashes"][scenario_id]
    joint_hash = blueprint_set["jointBlueprintHash"]
    response_schema = build_expansion_response_schema(
        scenario_id,
        str(topic["topicId"]),
        blueprint_hash,
        joint_hash,
        blueprint_set["blueprints"][scenario_id],
    )
    planner_binding = _model_binding(planner)
    stage_key = _scenario_stage_key(
        request,
        topic,
        blueprint_set,
        blueprint_scrutiny,
        scenario_id,
        response_schema,
        planner_binding,
    )
    return response_schema, planner_binding, stage_key, _checkpoint_path(
        Path("."), "scenarios", scenario_id, stage_key
    )


def expand_blueprint_slot(
    *,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    blueprint_scrutiny: Mapping[str, Any] | None,
    scenario_id: str,
    output_root: Path,
    planner: StrictSchemaModel,
    max_attempts: int = 4,
    resume: bool = False,
    premise_registry: _PremiseRegistry | None = None,
) -> dict[str, Any]:
    """Create or exactly resume one immutable Stage E scenario checkpoint."""

    if not 1 <= max_attempts <= 12:
        raise ScenarioBlueprintError("max_attempts must be in [1,12]")
    validate_blueprint_set(blueprint_set, topic)
    if blueprint_scrutiny is None:
        raise ScenarioBlueprintError(
            "Stage E requires a passing independent whole-blueprint scrutiny artifact"
        )
    validate_blueprint_scrutiny(
        blueprint_scrutiny, request, topic, blueprint_set, require_pass=True
    )
    if scenario_id not in blueprint_set["scenarioIds"]:
        raise ScenarioBlueprintError("assigned scenario ID is absent from the joint blueprint")
    topic_id = str(topic["topicId"])
    blueprint_hash = blueprint_set["blueprintHashes"][scenario_id]
    joint_hash = blueprint_set["jointBlueprintHash"]
    response_schema = build_expansion_response_schema(
        scenario_id,
        topic_id,
        blueprint_hash,
        joint_hash,
        blueprint_set["blueprints"][scenario_id],
    )
    planner_binding = _model_binding(planner)
    stage_key = _scenario_stage_key(
        request,
        topic,
        blueprint_set,
        blueprint_scrutiny,
        scenario_id,
        response_schema,
        planner_binding,
    )
    path = _checkpoint_path(Path(output_root), "scenarios", scenario_id, stage_key)
    registry = premise_registry or _PremiseRegistry()
    if path.exists():
        if not resume:
            raise ScenarioBlueprintError(f"scenario checkpoint exists; use --resume: {path}")
        checkpoint = _validate_expansion_checkpoint(
            read_json(path),
            request,
            topic,
            blueprint_set,
            blueprint_scrutiny,
            scenario_id,
            response_schema,
            planner_binding,
        )
        registry.add_existing(scenario_id, checkpoint["scenarioContract"]["premise"])
        return checkpoint

    context = _expansion_context(
        request, topic, blueprint_set, blueprint_scrutiny, scenario_id
    )
    failures: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            response, metadata = _call_model(
                planner,
                name="personaplex_scenario_expansion_v5",
                schema=response_schema,
                instructions=STAGE_E_SYSTEM,
                context=context,
                max_output_tokens=SCENARIO_MAX_OUTPUT_TOKENS,
            )
            validated = validate_expansion_response(
                response,
                scenario_id,
                topic_id,
                blueprint_hash,
                joint_hash,
                blueprint_set["blueprints"][scenario_id],
            )
            scenario = validated["scenarioContract"]
            body = {
                "schema": EXPANSION_CHECKPOINT_SCHEMA,
                "stageKey": stage_key,
                "requestHash": content_hash(request),
                "topicId": topic_id,
                "topicCardHash": content_hash(topic),
                "scenarioId": scenario_id,
                "blueprintHash": blueprint_hash,
                "jointBlueprintHash": joint_hash,
                "blueprintScrutinyHash": blueprint_scrutiny["checkpointHash"],
                "responseSchemaHash": content_hash(response_schema),
                "plannerBindingHash": content_hash(planner_binding),
                "plannerBinding": planner_binding,
                "modelCall": metadata,
                "scenarioContract": scenario,
                "scenarioContractHash": content_hash(scenario),
            }
            checkpoint = dict(body)
            checkpoint["checkpointHash"] = content_hash(body)
            registry.persist(
                scenario_id,
                scenario["premise"],
                lambda: _write_immutable_json(path, checkpoint),
            )
            return checkpoint
        except ModelTransportUnavailable:
            raise
        except Exception as error:
            failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
    raise ScenarioBlueprintError(
        f"Stage E exhausted {max_attempts} exact attempts for {scenario_id}: "
        + " | ".join(failures)
    )


def validate_canonical_scenarios(
    request: Mapping[str, Any],
    topics: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
) -> None:
    validate_corpus_shape(request, topics)
    topic_by_id = {str(topic["topicId"]): topic for topic in topics}
    expected_ids = {
        scenario_id
        for topic_id in topic_by_id
        for scenario_id in scenario_ids_for_topic(topic_id)
    }
    if len(scenarios) != TOPICS_PER_CORPUS * BLUEPRINTS_PER_TOPIC:
        raise ScenarioBlueprintError("canonical scenario stage requires exactly 1000 contracts")
    actual_ids = [scenario.get("scenarioId") for scenario in scenarios]
    if set(actual_ids) != expected_ids or len(set(actual_ids)) != len(actual_ids):
        raise ScenarioBlueprintError("canonical scenario IDs do not bind the exact 50x20 lattice")
    premises: dict[str, str] = {}
    for scenario in scenarios:
        scenario_id = str(scenario["scenarioId"])
        topic_id = str(scenario.get("topicId"))
        if topic_id not in topic_by_id or scenario_id not in scenario_ids_for_topic(topic_id):
            raise ScenarioBlueprintError(f"scenario lineage is invalid for {scenario_id}")
        _raise_schema_errors(
            scenario,
            build_scenario_contract_schema(scenario_id, topic_id),
            "canonical scenario",
        )
        assert_no_target_fields(scenario)
        premise = str(scenario["premise"])
        if premise in premises:
            raise ScenarioBlueprintError(
                f"canonical premises must be exact-duplicate-free: {premises[premise]} and {scenario_id}"
            )
        premises[premise] = scenario_id


def build_scenario_blueprint_bindings(
    *,
    request: Mapping[str, Any],
    topics: Sequence[Mapping[str, Any]],
    blueprint_sets: Sequence[Mapping[str, Any]],
    blueprint_scrutinies: Sequence[Mapping[str, Any]],
    expansion_checkpoints: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the exact 1:1 canonical lineage sidecar without changing v2 contracts."""

    validate_canonical_scenarios(request, topics, scenarios)
    topic_by_id = {str(topic["topicId"]): topic for topic in topics}
    set_by_topic = {str(item["topicId"]): item for item in blueprint_sets}
    scrutiny_by_topic = {str(item["topicId"]): item for item in blueprint_scrutinies}
    if (
        len(set_by_topic) != TOPICS_PER_CORPUS
        or set(set_by_topic) != set(topic_by_id)
        or len(scrutiny_by_topic) != TOPICS_PER_CORPUS
        or set(scrutiny_by_topic) != set(topic_by_id)
    ):
        raise ScenarioBlueprintError("binding assembly requires exact topic blueprint/scrutiny coverage")
    for topic_id in sorted(topic_by_id):
        validate_blueprint_set(set_by_topic[topic_id], topic_by_id[topic_id])
        validate_blueprint_scrutiny(
            scrutiny_by_topic[topic_id],
            request,
            topic_by_id[topic_id],
            set_by_topic[topic_id],
            require_pass=True,
        )
    _validate_checkpoint_lineage(
        expansion_checkpoints, blueprint_sets, blueprint_scrutinies
    )
    checkpoint_by_id = {
        str(checkpoint["scenarioId"]): checkpoint for checkpoint in expansion_checkpoints
    }
    scenario_by_id = {str(scenario["scenarioId"]): scenario for scenario in scenarios}
    if set(checkpoint_by_id) != set(scenario_by_id):
        raise ScenarioBlueprintError("expansion checkpoints and final scenarios are not exact 1:1")

    bindings: list[dict[str, Any]] = []
    for scenario_id in sorted(scenario_by_id):
        scenario = scenario_by_id[scenario_id]
        topic_id = str(scenario["topicId"])
        blueprint_set = set_by_topic[topic_id]
        scrutiny = scrutiny_by_topic[topic_id]
        checkpoint = checkpoint_by_id[scenario_id]
        if checkpoint["scenarioContract"] != scenario:
            raise ScenarioBlueprintError(
                f"mixed final scenario/checkpoint content for {scenario_id}"
            )
        scenario_hash = content_hash(scenario)
        if checkpoint["scenarioContractHash"] != scenario_hash:
            raise ScenarioBlueprintError(f"stale final scenario hash for {scenario_id}")
        slot = blueprint_set["scenarioIds"].index(scenario_id) + 1
        planner_binding = {
            "blueprint": blueprint_set["blueprintPlannerBinding"],
            "scrutiny": scrutiny["judgeBinding"],
            "expansion": checkpoint["plannerBinding"],
        }
        body = {
            "schema": BLUEPRINT_BINDING_SCHEMA,
            "scenarioId": scenario_id,
            "topicId": topic_id,
            "blueprintSlot": slot,
            "blueprintProfileHash": blueprint_set["blueprintHashes"][scenario_id],
            "jointBlueprintHash": blueprint_set["jointBlueprintHash"],
            "blueprintScrutinyHash": scrutiny["checkpointHash"],
            "expansionCheckpointHash": checkpoint["checkpointHash"],
            "plannerBinding": planner_binding,
            "plannerBindingHash": content_hash(planner_binding),
            "finalScenarioHash": scenario_hash,
        }
        row = dict(body)
        row["bindingHash"] = content_hash(body)
        bindings.append(row)
    validate_scenario_blueprint_bindings(
        request=request,
        topics=topics,
        blueprint_sets=blueprint_sets,
        blueprint_scrutinies=blueprint_scrutinies,
        expansion_checkpoints=expansion_checkpoints,
        scenarios=scenarios,
        bindings=bindings,
    )
    return bindings


def validate_scenario_blueprint_bindings(
    *,
    request: Mapping[str, Any],
    topics: Sequence[Mapping[str, Any]],
    blueprint_sets: Sequence[Mapping[str, Any]],
    blueprint_scrutinies: Sequence[Mapping[str, Any]],
    expansion_checkpoints: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
) -> None:
    validate_canonical_scenarios(request, topics, scenarios)
    if len(bindings) != TOPICS_PER_CORPUS * BLUEPRINTS_PER_TOPIC:
        raise ScenarioBlueprintError("blueprint binding sidecar must contain exactly 1000 rows")
    set_by_topic = {str(item["topicId"]): item for item in blueprint_sets}
    scrutiny_by_topic = {str(item["topicId"]): item for item in blueprint_scrutinies}
    checkpoint_by_id = {
        str(checkpoint["scenarioId"]): checkpoint for checkpoint in expansion_checkpoints
    }
    scenario_by_id = {str(scenario["scenarioId"]): scenario for scenario in scenarios}
    binding_by_id = {str(binding.get("scenarioId")): binding for binding in bindings}
    if (
        len(binding_by_id) != len(bindings)
        or set(binding_by_id) != set(scenario_by_id)
        or set(checkpoint_by_id) != set(scenario_by_id)
    ):
        raise ScenarioBlueprintError("scenario, checkpoint, and binding identities are not exact 1:1")
    required = {
        "schema",
        "scenarioId",
        "topicId",
        "blueprintSlot",
        "blueprintProfileHash",
        "jointBlueprintHash",
        "blueprintScrutinyHash",
        "expansionCheckpointHash",
        "plannerBinding",
        "plannerBindingHash",
        "finalScenarioHash",
        "bindingHash",
    }
    for scenario_id, binding in binding_by_id.items():
        if set(binding) != required or binding.get("schema") != BLUEPRINT_BINDING_SCHEMA:
            raise ScenarioBlueprintError(f"invalid binding field set for {scenario_id}")
        scenario = scenario_by_id[scenario_id]
        topic_id = str(scenario["topicId"])
        blueprint_set = set_by_topic.get(topic_id)
        scrutiny = scrutiny_by_topic.get(topic_id)
        checkpoint = checkpoint_by_id[scenario_id]
        if blueprint_set is None or scrutiny is None:
            raise ScenarioBlueprintError(f"binding has unknown topic lineage for {scenario_id}")
        expected_slot = blueprint_set["scenarioIds"].index(scenario_id) + 1
        expected_planner_binding = {
            "blueprint": blueprint_set["blueprintPlannerBinding"],
            "scrutiny": scrutiny["judgeBinding"],
            "expansion": checkpoint["plannerBinding"],
        }
        expected = {
            "scenarioId": scenario_id,
            "topicId": topic_id,
            "blueprintSlot": expected_slot,
            "blueprintProfileHash": blueprint_set["blueprintHashes"][scenario_id],
            "jointBlueprintHash": blueprint_set["jointBlueprintHash"],
            "blueprintScrutinyHash": scrutiny["checkpointHash"],
            "expansionCheckpointHash": checkpoint["checkpointHash"],
            "plannerBinding": expected_planner_binding,
            "plannerBindingHash": content_hash(expected_planner_binding),
            "finalScenarioHash": content_hash(scenario),
        }
        for field, expected_value in expected.items():
            if binding.get(field) != expected_value:
                raise ScenarioBlueprintError(
                    f"stale or mixed blueprint binding for {scenario_id}: {field}"
                )
        body = dict(binding)
        supplied_hash = body.pop("bindingHash")
        if supplied_hash != content_hash(body):
            raise ScenarioBlueprintError(f"binding content hash is invalid for {scenario_id}")


def build_scenario_blueprint_binding_manifest(
    *,
    request: Mapping[str, Any],
    topics: Sequence[Mapping[str, Any]],
    blueprint_sets: Sequence[Mapping[str, Any]],
    blueprint_scrutinies: Sequence[Mapping[str, Any]],
    scenarios: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(bindings) != TOPICS_PER_CORPUS * BLUEPRINTS_PER_TOPIC:
        raise ScenarioBlueprintError("cannot manifest an incomplete blueprint binding sidecar")
    ordered_topics = sorted(topics, key=lambda item: str(item["topicId"]))
    ordered_sets = sorted(blueprint_sets, key=lambda item: str(item["topicId"]))
    ordered_scrutinies = sorted(
        blueprint_scrutinies, key=lambda item: str(item["topicId"])
    )
    ordered_scenarios = sorted(scenarios, key=lambda item: str(item["scenarioId"]))
    ordered_bindings = sorted(bindings, key=lambda item: str(item["scenarioId"]))
    body = {
        "schema": BLUEPRINT_BINDING_MANIFEST_SCHEMA,
        "requestHash": content_hash(request),
        "topicCount": TOPICS_PER_CORPUS,
        "scenarioCount": TOPICS_PER_CORPUS * BLUEPRINTS_PER_TOPIC,
        "bindingCount": len(ordered_bindings),
        "topicCardsHash": content_hash(ordered_topics),
        "blueprintSetsHash": content_hash(ordered_sets),
        "blueprintScrutiniesHash": content_hash(ordered_scrutinies),
        "scenarioContractsHash": content_hash(ordered_scenarios),
        "scenarioBlueprintBindingsHash": content_hash(ordered_bindings),
    }
    manifest = dict(body)
    manifest["manifestHash"] = content_hash(body)
    return manifest


def _validate_checkpoint_lineage(
    checkpoints: Sequence[Mapping[str, Any]],
    blueprint_sets: Sequence[Mapping[str, Any]],
    blueprint_scrutinies: Sequence[Mapping[str, Any]],
) -> None:
    set_by_topic = {str(item["topicId"]): item for item in blueprint_sets}
    scrutiny_by_topic = {str(item["topicId"]): item for item in blueprint_scrutinies}
    if len(checkpoints) != TOPICS_PER_CORPUS * BLUEPRINTS_PER_TOPIC:
        raise ScenarioBlueprintError("Stage E checkpoint cardinality is not exactly 1000")
    seen: set[str] = set()
    for checkpoint in checkpoints:
        scenario_id = str(checkpoint.get("scenarioId"))
        topic_id = str(checkpoint.get("topicId"))
        blueprint_set = set_by_topic.get(topic_id)
        if blueprint_set is None or scenario_id not in blueprint_set["scenarioIds"]:
            raise ScenarioBlueprintError(f"scenario checkpoint has unbound lineage: {scenario_id}")
        if checkpoint.get("blueprintHash") != blueprint_set["blueprintHashes"][scenario_id]:
            raise ScenarioBlueprintError(f"scenario checkpoint blueprint hash mismatch: {scenario_id}")
        if checkpoint.get("jointBlueprintHash") != blueprint_set["jointBlueprintHash"]:
            raise ScenarioBlueprintError(f"scenario checkpoint joint hash mismatch: {scenario_id}")
        scrutiny = scrutiny_by_topic.get(topic_id)
        if scrutiny is None or checkpoint.get("blueprintScrutinyHash") != scrutiny.get("checkpointHash"):
            raise ScenarioBlueprintError(f"scenario checkpoint scrutiny hash mismatch: {scenario_id}")
        if checkpoint.get("plannerBindingHash") != content_hash(checkpoint.get("plannerBinding")):
            raise ScenarioBlueprintError(f"scenario checkpoint planner binding mismatch: {scenario_id}")
        if checkpoint.get("scenarioContractHash") != content_hash(checkpoint.get("scenarioContract")):
            raise ScenarioBlueprintError(f"scenario checkpoint final scenario hash mismatch: {scenario_id}")
        _checkpoint_body_hash(checkpoint)
        if scenario_id in seen:
            raise ScenarioBlueprintError(f"duplicate scenario checkpoint identity: {scenario_id}")
        seen.add(scenario_id)


def generate_scenarios(
    *,
    request: Mapping[str, Any],
    topics: Sequence[Mapping[str, Any]],
    blueprint_sets: Sequence[Mapping[str, Any]],
    blueprint_scrutinies: Sequence[Mapping[str, Any]] | None = None,
    output_root: Path,
    planner: StrictSchemaModel,
    max_workers: int = MAX_WORKERS,
    max_attempts: int = 4,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Run Stage E and emit canonical JSONL only after all 1000 slots pass."""

    validate_corpus_shape(request, topics)
    _validate_worker_count(max_workers)
    topic_by_id = {str(topic["topicId"]): topic for topic in topics}
    set_by_topic = {str(item["topicId"]): item for item in blueprint_sets}
    if set(topic_by_id) != set(set_by_topic) or len(blueprint_sets) != TOPICS_PER_CORPUS:
        raise ScenarioBlueprintError("Stage E requires exactly one bound blueprint set per topic")
    for topic_id, blueprint_set in set_by_topic.items():
        validate_blueprint_set(blueprint_set, topic_by_id[topic_id])
    scrutinies = list(
        blueprint_scrutinies
        if blueprint_scrutinies is not None
        else load_blueprint_scrutinies(output_root, request, topics, blueprint_sets)
    )
    scrutiny_by_topic = {str(item["topicId"]): item for item in scrutinies}
    if len(scrutinies) != TOPICS_PER_CORPUS or set(scrutiny_by_topic) != set(topic_by_id):
        raise ScenarioBlueprintError("Stage E requires exactly one passing scrutiny per topic")
    for topic_id, scrutiny in scrutiny_by_topic.items():
        validate_blueprint_scrutiny(
            scrutiny,
            request,
            topic_by_id[topic_id],
            set_by_topic[topic_id],
            require_pass=True,
        )

    assignments = [
        (
            topic_by_id[topic_id],
            set_by_topic[topic_id],
            scrutiny_by_topic[topic_id],
            scenario_id,
        )
        for topic_id in sorted(topic_by_id)
        for scenario_id in set_by_topic[topic_id]["scenarioIds"]
    ]
    planner_binding = _model_binding(planner)
    registry = _PremiseRegistry()
    completed: list[dict[str, Any]] = []
    pending: list[
        tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], str]
    ] = []

    for topic, blueprint_set, blueprint_scrutiny, scenario_id in assignments:
        blueprint_hash = blueprint_set["blueprintHashes"][scenario_id]
        joint_hash = blueprint_set["jointBlueprintHash"]
        response_schema = build_expansion_response_schema(
            scenario_id,
            str(topic["topicId"]),
            blueprint_hash,
            joint_hash,
            blueprint_set["blueprints"][scenario_id],
        )
        stage_key = _scenario_stage_key(
            request,
            topic,
            blueprint_set,
            blueprint_scrutiny,
            scenario_id,
            response_schema,
            planner_binding,
        )
        path = _checkpoint_path(Path(output_root), "scenarios", scenario_id, stage_key)
        if path.exists():
            if not resume:
                raise ScenarioBlueprintError(f"scenario checkpoint exists; use --resume: {path}")
            checkpoint = _validate_expansion_checkpoint(
                read_json(path),
                request,
                topic,
                blueprint_set,
                blueprint_scrutiny,
                scenario_id,
                response_schema,
                planner_binding,
            )
            registry.add_existing(scenario_id, checkpoint["scenarioContract"]["premise"])
            completed.append(checkpoint)
        else:
            pending.append((topic, blueprint_set, blueprint_scrutiny, scenario_id))

    generated = _parallel(
        pending,
        lambda assignment: expand_blueprint_slot(
            request=request,
            topic=assignment[0],
            blueprint_set=assignment[1],
            blueprint_scrutiny=assignment[2],
            scenario_id=assignment[3],
            output_root=Path(output_root),
            planner=planner,
            max_attempts=max_attempts,
            resume=False,
            premise_registry=registry,
        ),
        max_workers=max_workers,
        identity=lambda assignment: assignment[3],
    )
    checkpoints = completed + generated
    _validate_checkpoint_lineage(checkpoints, blueprint_sets, scrutinies)
    scenarios = sorted(
        (checkpoint["scenarioContract"] for checkpoint in checkpoints),
        key=lambda scenario: scenario["scenarioId"],
    )
    validate_canonical_scenarios(request, topics, scenarios)
    bindings = build_scenario_blueprint_bindings(
        request=request,
        topics=topics,
        blueprint_sets=blueprint_sets,
        blueprint_scrutinies=scrutinies,
        expansion_checkpoints=checkpoints,
        scenarios=scenarios,
    )
    manifest = build_scenario_blueprint_binding_manifest(
        request=request,
        topics=topics,
        blueprint_sets=blueprint_sets,
        blueprint_scrutinies=scrutinies,
        scenarios=scenarios,
        bindings=bindings,
    )
    _write_immutable_jsonl(
        Path(output_root) / SCENARIO_BINDINGS_FILENAME, bindings, resume=resume
    )
    _write_immutable_jsonl(
        Path(output_root) / SCENARIO_CONTRACTS_FILENAME, scenarios, resume=resume
    )
    manifest_path = Path(output_root) / SCENARIO_BINDINGS_MANIFEST_FILENAME
    if manifest_path.exists() and not resume:
        raise ScenarioBlueprintError(f"canonical artifact exists; use --resume: {manifest_path}")
    _write_immutable_json(manifest_path, manifest)
    return scenarios


def quarantine_artifact(path: Path, output_root: Path) -> Path:
    """Move one prior output to a content-addressed quarantine without overwrite."""

    path = Path(path)
    if not path.is_file():
        raise ScenarioBlueprintError(f"cannot quarantine missing artifact: {path}")
    digest = file_hash(path)[7:]
    destination = Path(output_root) / "quarantine" / f"{path.name}.{digest}.prior"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != path.read_bytes():
            raise ScenarioBlueprintError(f"quarantine hash collision at {destination}")
        path.unlink()
    else:
        os.replace(path, destination)
    return destination


def prepare_output_root(
    *,
    request_path: Path,
    input_root: Path,
    output_root: Path,
    resume: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Snapshot immutable inputs into a disjoint root and preserve prior outputs."""

    request_path = Path(request_path).resolve()
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()
    if input_root == output_root:
        raise ScenarioBlueprintError("output root must be disjoint from the input root")
    topics_path = input_root / "topic_cards.jsonl"
    if not request_path.is_file() or not topics_path.is_file():
        raise ScenarioBlueprintError("request and input topic_cards.jsonl are required")
    request_bytes = request_path.read_bytes()
    topics_bytes = topics_path.read_bytes()
    request = read_json(request_path)
    topics = read_jsonl(topics_path)
    validate_corpus_shape(request, topics)
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_root = output_root / CHECKPOINT_ROOT_NAME / "checkpoints"
    if not resume and checkpoint_root.exists() and any(checkpoint_root.rglob("*.json")):
        raise ScenarioBlueprintError("output root has immutable checkpoints; use --resume or a new root")
    bound_names = (
        "request.json",
        "topic_cards.jsonl",
        BLUEPRINT_SETS_FILENAME,
        BLUEPRINT_SCRUTINY_FILENAME,
        SCENARIO_CONTRACTS_FILENAME,
        SCENARIO_BINDINGS_FILENAME,
        SCENARIO_BINDINGS_MANIFEST_FILENAME,
    )
    if not resume:
        for name in bound_names:
            candidate = output_root / name
            if candidate.is_file():
                quarantine_artifact(candidate, output_root)
    _write_immutable_bytes(output_root / "request.json", request_bytes)
    _write_immutable_bytes(output_root / "topic_cards.jsonl", topics_bytes)
    return request, topics


# Explicit aliases for callers that describe the stages rather than implementation details.
scenario_blueprint_response_schema = build_blueprint_response_schema
scenario_expansion_response_schema = build_expansion_response_schema
run_blueprint_stage = generate_blueprints
run_scenario_stage = generate_scenarios
