"""Provider-aware strict JSON-Schema transport primitives.

The canonical schema is always the host admission contract.  A provider
capability profile may produce a narrower transport grammar, but the profile,
canonical hash, projected hash, and removed-keyword counts stay bound together.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence
import urllib.parse


OPENROUTER_NEMOTRON_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_NEMOTRON_ULTRA_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
DEFAULT_TRANSPORT_ATTEMPTS = 4
MAX_TRANSPORT_ATTEMPTS = 12
DEFAULT_RETRY_BASE_SECONDS = 1.0
DEFAULT_RETRY_MAX_SECONDS = 30.0
MAX_RETRY_MAX_SECONDS = 300.0
RETRIABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
FULL_CANONICAL_SCHEMA_RETRY_DIRECTIVE = (
    "Regenerate the complete response from scratch under every original structural "
    "assignment. Satisfy the full canonical JSON Schema, including uniqueness and "
    "exclusion constraints omitted only from the provider transport grammar. Do not "
    "patch, reuse, or reinterpret an immutable assigned field."
)
MINIFIED_JSON_ONLY_CONTRACT = (
    "Return exactly one minified JSON value and no other content. Do not include "
    "Markdown, code fences, comments, prose, or text before or after the JSON. "
    "The JSON must validate against this exact canonical JSON Schema:"
)

_SCHEMA_MAP_KEYWORDS = frozenset(
    {
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
)
_SCHEMA_ARRAY_KEYWORDS = frozenset(
    {"allOf", "anyOf", "oneOf", "prefixItems"}
)
_SCHEMA_SINGLE_KEYWORDS = frozenset(
    {
        "additionalItems",
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def content_hash(value: Any) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def append_canonical_schema_contract(
    instructions: str,
    canonical_schema: Mapping[str, Any],
) -> str:
    """Append the deterministic raw-JSON contract without projecting the schema."""

    return (
        instructions
        + "\n\n"
        + MINIFIED_JSON_ONLY_CONTRACT
        + "\n"
        + canonical_json(dict(canonical_schema))
    )


@dataclass(frozen=True)
class ProviderSchemaCapabilityProfile:
    """An explicit, evidence-backed provider grammar capability profile."""

    name: str
    provider: str
    models: tuple[str, ...]
    unsupported_keywords: tuple[str, ...]
    response_format_supported: bool = True
    canonical_schema_in_prompt: bool = False

    def binding(self) -> dict[str, Any]:
        body = {
            "name": self.name,
            "provider": self.provider,
            "models": list(self.models),
            "unsupportedKeywords": list(self.unsupported_keywords),
        }
        if not self.response_format_supported:
            body["responseFormat"] = "omitted"
            body["schemaDelivery"] = "exact_canonical_schema_in_prompt"
        return {**body, "profileHash": content_hash(body)}


CANONICAL_SCHEMA_PROFILE = ProviderSchemaCapabilityProfile(
    name="canonical_json_schema_2020_12",
    provider="openai_compatible",
    models=(),
    unsupported_keywords=(),
)
OPENROUTER_NEMOTRON_SCHEMA_PROFILE = ProviderSchemaCapabilityProfile(
    name="openrouter_nemotron_free_grammar_v1",
    provider="openrouter_nvidia",
    models=(OPENROUTER_NEMOTRON_MODEL,),
    unsupported_keywords=("not", "uniqueItems"),
)
OPENROUTER_NEMOTRON_ULTRA_SCHEMA_PROFILE = ProviderSchemaCapabilityProfile(
    name="openrouter_nemotron_ultra_550b_prompt_schema_v1",
    provider="openrouter_nvidia",
    models=(OPENROUTER_NEMOTRON_ULTRA_MODEL,),
    unsupported_keywords=(),
    response_format_supported=False,
    canonical_schema_in_prompt=True,
)


@dataclass(frozen=True)
class SchemaTransportProjection:
    transport_schema: dict[str, Any]
    binding: dict[str, Any]


@dataclass(frozen=True)
class ProviderEnvelopeError:
    code: int | None
    retryable: bool

    @property
    def classification(self) -> str:
        suffix = str(self.code) if self.code is not None else "unknown"
        return f"provider_error_{suffix}"


def normalize_chat_completion_endpoints(
    endpoints: str | Sequence[str],
) -> tuple[str, ...]:
    values = endpoints.split(",") if isinstance(endpoints, str) else list(endpoints)
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            continue
        endpoint = value.strip().rstrip("/")
        try:
            parsed = urllib.parse.urlsplit(endpoint)
        except ValueError as error:
            raise ValueError("planner endpoint URL is malformed") from error
        if (
            parsed.scheme.lower() not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or not parsed.path.rstrip("/").endswith("/v1/chat/completions")
        ):
            raise ValueError(
                "planner endpoints must be credential-free HTTP(S) "
                "/v1/chat/completions URLs"
            )
        if endpoint not in normalized:
            normalized.append(endpoint)
    if not 1 <= len(normalized) <= 3:
        raise ValueError("planning requires one to three distinct endpoints")
    return tuple(normalized)


def is_openrouter_endpoint(endpoint: str) -> bool:
    try:
        hostname = (urllib.parse.urlsplit(endpoint).hostname or "").lower()
    except ValueError:
        return False
    return hostname == "openrouter.ai" or hostname.endswith(".openrouter.ai")


def schema_capability_profile(
    endpoint: str, model: str
) -> ProviderSchemaCapabilityProfile:
    if (
        is_openrouter_endpoint(endpoint)
        and model.strip() == OPENROUTER_NEMOTRON_ULTRA_MODEL
    ):
        return OPENROUTER_NEMOTRON_ULTRA_SCHEMA_PROFILE
    if is_openrouter_endpoint(endpoint) and model.strip() == OPENROUTER_NEMOTRON_MODEL:
        return OPENROUTER_NEMOTRON_SCHEMA_PROFILE
    return CANONICAL_SCHEMA_PROFILE


def _project_schema_node(
    value: Any,
    profile: ProviderSchemaCapabilityProfile,
    removed: dict[str, int],
) -> Any:
    if isinstance(value, bool):
        return value
    if not isinstance(value, Mapping):
        return deepcopy(value)
    unsupported = frozenset(profile.unsupported_keywords)
    projected: dict[str, Any] = {}
    for key, child in value.items():
        if key in unsupported:
            removed[key] = removed.get(key, 0) + 1
            continue
        if key in _SCHEMA_MAP_KEYWORDS and isinstance(child, Mapping):
            projected[key] = {
                name: _project_schema_node(schema, profile, removed)
                for name, schema in child.items()
            }
        elif key in _SCHEMA_ARRAY_KEYWORDS and isinstance(child, list):
            projected[key] = [
                _project_schema_node(schema, profile, removed) for schema in child
            ]
        elif key in _SCHEMA_SINGLE_KEYWORDS:
            if isinstance(child, list):
                projected[key] = [
                    _project_schema_node(schema, profile, removed) for schema in child
                ]
            else:
                projected[key] = _project_schema_node(child, profile, removed)
        elif key == "dependencies" and isinstance(child, Mapping):
            projected[key] = {
                name: (
                    _project_schema_node(dependency, profile, removed)
                    if isinstance(dependency, (Mapping, bool))
                    else deepcopy(dependency)
                )
                for name, dependency in child.items()
            }
        else:
            projected[key] = deepcopy(child)
    return projected


def build_schema_transport_projection(
    endpoint: str,
    model: str,
    canonical_schema: Mapping[str, Any],
) -> SchemaTransportProjection:
    profile = schema_capability_profile(endpoint, model)
    canonical_copy = json.loads(canonical_json(dict(canonical_schema)))
    removed: dict[str, int] = {}
    transport_schema = _project_schema_node(canonical_copy, profile, removed)
    if not isinstance(transport_schema, dict):
        raise ValueError("response schema projection must remain an object")
    binding = {
        "profile": profile.binding(),
        "canonicalSchemaHash": content_hash(canonical_copy),
        "transportSchemaHash": content_hash(transport_schema),
        "removedKeywordCounts": dict(sorted(removed.items())),
    }
    return SchemaTransportProjection(transport_schema, binding)


def _structural_path(parts: Sequence[Any]) -> str:
    path = "$"
    for part in parts:
        if type(part) is int:
            path += f"[{part}]"
        elif isinstance(part, str) and part.replace("_", "a").isalnum():
            path += f".{part}"
        else:
            path += f"[{canonical_json(str(part))}]"
    return path


def canonical_validation_defect(errors: Sequence[Any]) -> str:
    """Describe schema failures exactly without including generated values."""

    ordered = sorted(
        errors,
        key=lambda error: (
            tuple(str(item) for item in error.absolute_path),
            tuple(str(item) for item in error.absolute_schema_path),
        ),
    )
    defects: list[str] = []
    for error in ordered[:8]:
        defects.append(
            "instancePath="
            + _structural_path(tuple(error.absolute_path))
            + ";schemaPath="
            + _structural_path(tuple(error.absolute_schema_path))
            + ";keyword="
            + str(error.validator)
            + ";constraintHash="
            + content_hash(error.validator_value)
        )
    return " | ".join(defects)


def canonical_retry_context(
    context: Mapping[str, Any],
    defect: str,
    next_attempt: int,
) -> dict[str, Any]:
    retry_context = deepcopy(dict(context))
    feedback: dict[str, Any] = {
        "attempt": next_attempt,
        "canonicalSchemaDefect": defect,
        "directive": FULL_CANONICAL_SCHEMA_RETRY_DIRECTIVE,
    }
    prior_feedback = context.get("retryFeedback")
    if isinstance(prior_feedback, Mapping):
        feedback["priorStageFeedback"] = deepcopy(dict(prior_feedback))
    retry_context["retryFeedback"] = feedback
    return retry_context


def classify_provider_error(envelope: Any) -> ProviderEnvelopeError | None:
    if not isinstance(envelope, Mapping) or "error" not in envelope:
        return None
    error = envelope.get("error")
    raw_code = error.get("code") if isinstance(error, Mapping) else None
    code: int | None = None
    if type(raw_code) is int:
        code = raw_code
    elif isinstance(raw_code, str) and raw_code.strip().isdigit():
        code = int(raw_code.strip())
    return ProviderEnvelopeError(
        code=code,
        retryable=code is None or code in RETRIABLE_HTTP_STATUS,
    )


def retry_after_seconds(
    headers: Any,
    *,
    now: datetime | None = None,
) -> float | None:
    if headers is None or not hasattr(headers, "get"):
        return None
    raw_value = headers.get("Retry-After")
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        reference = now or datetime.now(timezone.utc)
        seconds = (retry_at - reference).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def validate_retry_settings(
    attempts: int,
    base_seconds: float,
    max_seconds: float,
) -> tuple[int, float, float]:
    if type(attempts) is not int or not 1 <= attempts <= MAX_TRANSPORT_ATTEMPTS:
        raise ValueError(
            f"transport attempts must be in [1,{MAX_TRANSPORT_ATTEMPTS}]"
        )
    if isinstance(base_seconds, bool) or not isinstance(base_seconds, (int, float)):
        raise ValueError("retry base seconds must be numeric")
    if isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float)):
        raise ValueError("retry max seconds must be numeric")
    base = float(base_seconds)
    maximum = float(max_seconds)
    if not math.isfinite(base) or not 0.0 <= base <= MAX_RETRY_MAX_SECONDS:
        raise ValueError(
            f"retry base seconds must be in [0,{MAX_RETRY_MAX_SECONDS:g}]"
        )
    if (
        not math.isfinite(maximum)
        or not base <= maximum <= MAX_RETRY_MAX_SECONDS
    ):
        raise ValueError(
            f"retry max seconds must be in [{base:g},{MAX_RETRY_MAX_SECONDS:g}]"
        )
    return attempts, base, maximum


def retry_delay_seconds(
    failure_attempt: int,
    base_seconds: float,
    max_seconds: float,
    retry_after: float | None = None,
) -> float:
    exponential = base_seconds * (2 ** max(0, failure_attempt - 1))
    requested = retry_after if retry_after is not None else 0.0
    return min(max_seconds, max(exponential, requested))
