from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Lock
import urllib.error

import pytest

from ground_truth_finetuning.tools.build_scenarios_from_blueprints_v5 import (
    _large_proposer_config,
    _parse_endpoints,
)
from ground_truth_finetuning.training.compact_trajectory_fanout import (
    ProtocolError,
    ThreeEndpointJsonSchemaClient,
)
from ground_truth_finetuning.training.scenario_blueprint_v5 import (
    InvalidModelOutput,
    ModelTransportUnavailable,
    ScenarioBlueprintError,
    ThreeEndpointStrictSchemaPlanner,
)
from ground_truth_finetuning.training.strict_schema_transport import (
    FULL_CANONICAL_SCHEMA_RETRY_DIRECTIVE,
    MINIFIED_JSON_ONLY_CONTRACT,
    OPENROUTER_NEMOTRON_MODEL,
    OPENROUTER_NEMOTRON_ULTRA_MODEL,
    build_schema_transport_projection,
    canonical_json,
    content_hash,
)


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class FakeResponse:
    def __init__(self, envelope: dict, headers: dict[str, str] | None = None):
        self.body = json.dumps(envelope).encode("utf-8")
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, limit: int | None = None) -> bytes:
        return self.body if limit is None else self.body[:limit]


def success_envelope(value: dict) -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(value)},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    }


def constrained_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["blocked", "tags", "not"],
        "properties": {
            "blocked": {"type": "boolean"},
            "tags": {
                "type": "array",
                "items": {"type": "integer"},
                "uniqueItems": True,
            },
            "not": {"type": "string"},
        },
        "not": {
            "required": ["blocked"],
            "properties": {"blocked": {"const": True}},
        },
    }


def test_nemotron_projection_is_profile_bound_and_canonical_host_rejects(
    monkeypatch,
) -> None:
    payloads: list[dict] = []

    def urlopen(request, timeout):
        payloads.append(json.loads(request.data))
        return FakeResponse(
            success_envelope({"blocked": True, "tags": [1, 1], "not": "property"})
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    schema = constrained_schema()
    projection = build_schema_transport_projection(
        OPENROUTER_ENDPOINT, OPENROUTER_NEMOTRON_MODEL, schema
    )
    assert projection.binding["removedKeywordCounts"] == {"not": 1, "uniqueItems": 1}
    assert projection.binding["canonicalSchemaHash"] != projection.binding["transportSchemaHash"]
    assert "not" not in projection.transport_schema
    assert "not" in projection.transport_schema["properties"]
    assert "uniqueItems" not in projection.transport_schema["properties"]["tags"]

    planner = ThreeEndpointStrictSchemaPlanner(
        [OPENROUTER_ENDPOINT],
        OPENROUTER_NEMOTRON_MODEL,
        transport_attempts=2,
        retry_base_seconds=0,
        retry_max_seconds=0,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(InvalidModelOutput, match="canonical response schema"):
        planner.generate(
            name="projected_but_canonical",
            schema=schema,
            instructions="return schema",
            context={"task": "focused"},
            max_output_tokens=64,
        )
    sent_schema = payloads[0]["response_format"]["json_schema"]["schema"]
    assert sent_schema == projection.transport_schema
    assert len(payloads) == 2
    retry_context = json.loads(payloads[1]["messages"][1]["content"])
    assert retry_context["task"] == "focused"
    feedback = retry_context["retryFeedback"]
    assert feedback["directive"] == FULL_CANONICAL_SCHEMA_RETRY_DIRECTIVE
    assert "keyword=not" in feedback["canonicalSchemaDefect"]
    assert "keyword=uniqueItems" in feedback["canonicalSchemaDefect"]
    assert "blocked=True" not in feedback["canonicalSchemaDefect"]


def test_ultra_omits_response_format_and_host_validates_exact_schema(
    monkeypatch,
) -> None:
    payloads: list[dict] = []
    responses = iter(({"tags": [1, 1]}, {"tags": [1, 2]}))

    def urlopen(request, timeout):
        payloads.append(json.loads(request.data))
        return FakeResponse(success_envelope(next(responses)))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["tags"],
        "properties": {
            "tags": {
                "type": "array",
                "items": {"type": "integer"},
                "uniqueItems": True,
            }
        },
    }
    planner = ThreeEndpointStrictSchemaPlanner(
        [OPENROUTER_ENDPOINT],
        OPENROUTER_NEMOTRON_ULTRA_MODEL,
        "unit-test-secret",
        transport_attempts=2,
        retry_base_seconds=0,
        retry_max_seconds=0,
        sleep=lambda _seconds: None,
    )
    value, metadata = planner.generate(
        name="ultra_prompt_schema",
        schema=schema,
        instructions="Return the adjudication result.",
        context={"task": "private-input"},
        max_output_tokens=64,
    )

    assert value == {"tags": [1, 2]}
    assert len(payloads) == 2
    assert all("response_format" not in payload for payload in payloads)
    assert all(payload["reasoning"] == {"enabled": False} for payload in payloads)
    expected_system_prompt = (
        "Return the adjudication result.\n\n"
        + MINIFIED_JSON_ONLY_CONTRACT
        + "\n"
        + canonical_json(schema)
    )
    assert all(
        payload["messages"][0]["content"] == expected_system_prompt
        for payload in payloads
    )
    retry_context = json.loads(payloads[1]["messages"][1]["content"])
    assert "keyword=uniqueItems" in retry_context["retryFeedback"][
        "canonicalSchemaDefect"
    ]
    assert planner.binding()["responseFormat"] == "omitted"
    assert metadata["transportAttempt"] == 2
    assert metadata["schemaTransport"]["transportSchemaHash"] == content_hash(schema)
    assert metadata["requestBinding"] == {
        "profileHash": metadata["schemaTransport"]["profile"]["profileHash"],
        "modelHash": content_hash(OPENROUTER_NEMOTRON_ULTRA_MODEL),
        "schemaHash": content_hash(schema),
        "promptHash": content_hash(payloads[1]["messages"]),
    }
    metadata_text = canonical_json(metadata)
    assert "unit-test-secret" not in metadata_text
    assert "private-input" not in metadata_text
    assert expected_system_prompt not in metadata_text


def test_http_200_resource_exhausted_502_envelope_uses_bounded_retry_after(
    monkeypatch,
) -> None:
    calls = 0
    delays: list[float] = []

    def urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            return FakeResponse(
                {
                    "error": {
                        "code": 502,
                        "message": "ResourceExhausted detail must not enter diagnostics",
                    }
                },
                {"Retry-After": "20"},
            )
        return FakeResponse(success_envelope({"ok": True}))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    planner = ThreeEndpointStrictSchemaPlanner(
        [OPENROUTER_ENDPOINT],
        OPENROUTER_NEMOTRON_ULTRA_MODEL,
        transport_attempts=2,
        retry_base_seconds=0.5,
        retry_max_seconds=5,
        sleep=delays.append,
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok"],
        "properties": {"ok": {"const": True}},
    }
    value, metadata = planner.generate(
        name="provider_retry",
        schema=schema,
        instructions="return schema",
        context={"task": "focused"},
        max_output_tokens=32,
    )
    assert value == {"ok": True}
    assert calls == 2
    assert delays == [5.0]
    assert metadata["transportAttempt"] == 2
    assert metadata["schemaTransport"]["profile"]["name"] == (
        "openrouter_nemotron_ultra_550b_prompt_schema_v1"
    )
    assert "ResourceExhausted" not in canonical_json(metadata)


@pytest.mark.parametrize("status", [429, 503])
def test_retry_after_is_honored_for_http_throttling(monkeypatch, status: int) -> None:
    calls = 0
    delays: list[float] = []

    def urlopen(request, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                status,
                "retryable",
                {"Retry-After": "3"},
                None,
            )
        return FakeResponse(success_envelope({"ok": True}))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    planner = ThreeEndpointStrictSchemaPlanner(
        [OPENROUTER_ENDPOINT],
        OPENROUTER_NEMOTRON_MODEL,
        transport_attempts=2,
        retry_base_seconds=0.25,
        retry_max_seconds=10,
        sleep=delays.append,
    )
    planner.generate(
        name="retry_after",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"const": True}},
        },
        instructions="return schema",
        context={"task": "focused"},
        max_output_tokens=32,
    )
    assert calls == 2
    assert delays == [3.0]


def test_local_fallback_preserves_logical_binding_and_records_physical_route(
    monkeypatch,
) -> None:
    local_endpoint = "http://127.0.0.1:11434/v1/chat/completions"
    local_model = "nemotron-3-super:120b"
    payloads: list[tuple[str, dict]] = []

    def urlopen(request, timeout):
        payload = json.loads(request.data)
        payloads.append((request.full_url, payload))
        if request.full_url == OPENROUTER_ENDPOINT:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "daily quota",
                {"x-ratelimit-reset": "1784678400000"},
                None,
            )
        return FakeResponse(success_envelope({"ok": True}))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    planner = ThreeEndpointStrictSchemaPlanner(
        [OPENROUTER_ENDPOINT],
        OPENROUTER_NEMOTRON_MODEL,
        transport_attempts=1,
        fallback_endpoints=[local_endpoint],
        fallback_model=local_model,
    )
    value, metadata = planner.generate(
        name="local_fallback",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["ok"],
            "properties": {"ok": {"const": True}},
        },
        instructions="return schema",
        context={"task": "focused"},
        max_output_tokens=32,
    )

    assert value == {"ok": True}
    assert [endpoint for endpoint, _payload in payloads] == [
        OPENROUTER_ENDPOINT,
        local_endpoint,
    ]
    assert payloads[1][1]["model"] == local_model
    assert payloads[1][1]["reasoning"] == {"effort": "none"}
    assert payloads[1][1]["reasoning_effort"] == "none"
    assert "runtimeFallback" not in planner.binding()
    assert metadata["model"] == local_model
    assert metadata["transportRoute"] == {
        "kind": "fallback",
        "fallback": True,
        "logicalModel": OPENROUTER_NEMOTRON_MODEL,
        "actualModel": local_model,
        "logicalEndpointsHash": content_hash([OPENROUTER_ENDPOINT]),
    }


def test_transport_exhaustion_is_not_a_semantic_model_failure(monkeypatch) -> None:
    def urlopen(request, timeout):
        raise urllib.error.HTTPError(
            request.full_url, 429, "daily quota", {}, None
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    planner = ThreeEndpointStrictSchemaPlanner(
        [OPENROUTER_ENDPOINT],
        OPENROUTER_NEMOTRON_MODEL,
        transport_attempts=1,
    )
    with pytest.raises(ModelTransportUnavailable, match="physical planner routes"):
        planner.generate(
            name="unavailable",
            schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["ok"],
                "properties": {"ok": {"const": True}},
            },
            instructions="return schema",
            context={"task": "focused"},
            max_output_tokens=32,
        )


def test_one_endpoint_serves_three_concurrent_workers_with_exact_binding(monkeypatch) -> None:
    payloads: list[dict] = []
    lock = Lock()

    def urlopen(request, timeout):
        with lock:
            payloads.append(json.loads(request.data))
        return FakeResponse(success_envelope({"ok": True}))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    planner = ThreeEndpointStrictSchemaPlanner(
        [OPENROUTER_ENDPOINT],
        OPENROUTER_NEMOTRON_MODEL,
        transport_attempts=1,
    )
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok"],
        "properties": {"ok": {"const": True}},
    }

    def generate(_index: int):
        return planner.generate(
            name="single_lane_concurrency",
            schema=schema,
            instructions="return schema",
            context={"task": "focused"},
            max_output_tokens=32,
        )[0]

    with ThreadPoolExecutor(max_workers=3) as executor:
        assert list(executor.map(generate, range(3))) == [{"ok": True}] * 3
    assert len(payloads) == 3
    assert {payload["model"] for payload in payloads} == {OPENROUTER_NEMOTRON_MODEL}
    assert all(payload["reasoning"] == {"enabled": False} for payload in payloads)
    assert all(
        payload["response_format"]["json_schema"]["strict"] is True
        for payload in payloads
    )


def test_compact_fanout_projects_wire_schema_but_rejects_canonical_violation(
    monkeypatch,
) -> None:
    payloads: list[dict] = []

    def urlopen(request, timeout):
        payloads.append(json.loads(request.data))
        return FakeResponse(
            success_envelope({"blocked": True, "tags": [2, 2], "not": "property"})
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    client = ThreeEndpointJsonSchemaClient(
        OPENROUTER_ENDPOINT,
        OPENROUTER_NEMOTRON_MODEL,
        protocol_attempts=2,
        retry_base_seconds=0,
        retry_max_seconds=0,
        sleep=lambda _seconds: None,
    )
    with pytest.raises(ProtocolError, match="canonical_schema_exhausted"):
        client.generate(
            name="compact_projection",
            schema=constrained_schema(),
            instructions="return schema",
            context={"task": "focused"},
            max_output_tokens=64,
        )
    assert len(payloads) == 2
    assert all(
        "uniqueItems"
        not in payload["response_format"]["json_schema"]["schema"]["properties"]["tags"]
        for payload in payloads
    )
    retry_context = json.loads(payloads[1]["messages"][1]["content"])
    assert retry_context["retryFeedback"]["directive"] == (
        FULL_CANONICAL_SCHEMA_RETRY_DIRECTIVE
    )
    assert "keyword=not" in retry_context["retryFeedback"]["canonicalSchemaDefect"]


def test_adjudicator_guard_and_runtime_example_bindings() -> None:
    with pytest.raises(ScenarioBlueprintError, match="independent local"):
        _parse_endpoints(OPENROUTER_ENDPOINT, "Gemma verifier", local_only=True)

    env_path = (
        Path(__file__).resolve().parents[1]
        / "config"
        / "personaplex-runtime.env.example"
    )
    values = {
        key: value
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#") and "=" in line
        for key, value in [line.split("=", 1)]
    }
    assert values["OPENROUTER_API_KEY"] == ""
    assert values["PERSONAPLEX_CASCADE_PLANNER_ENDPOINT"] == OPENROUTER_ENDPOINT
    assert values["PERSONAPLEX_CASCADE_PLANNER_MODEL"] == OPENROUTER_NEMOTRON_MODEL
    assert values["PERSONAPLEX_TAXONOMY_REPAIR_ENDPOINT"] == OPENROUTER_ENDPOINT
    assert values["PERSONAPLEX_TAXONOMY_REPAIR_MODEL"] == OPENROUTER_NEMOTRON_MODEL
    assert values["PERSONAPLEX_LOCAL_SUPER_ENDPOINT"].startswith("http://127.0.0.1")
    assert values["PERSONAPLEX_LOCAL_SUPER_MODEL"] == "nemotron-3-super:120b"
    assert values["PERSONAPLEX_LOCAL_SUPER_PREFER"] == "1"
    assert values["PERSONAPLEX_LOCAL_SECONDARY_ENDPOINT"].startswith(
        "http://127.0.0.1"
    )
    assert values["PERSONAPLEX_LOCAL_SECONDARY_MODEL"] == "nemotron-3-nano:30b"
    assert values["PERSONAPLEX_BLUEPRINT_JUDGE_ENDPOINT"] == OPENROUTER_ENDPOINT
    assert values["PERSONAPLEX_BLUEPRINT_JUDGE_MODEL"] == OPENROUTER_NEMOTRON_MODEL
    assert values["PERSONAPLEX_LARGE_PROPOSER_ENDPOINT"] == OPENROUTER_ENDPOINT
    assert values["PERSONAPLEX_LARGE_PROPOSER_MODEL"] == (
        OPENROUTER_NEMOTRON_ULTRA_MODEL
    )
    assert values["PERSONAPLEX_TAXONOMY_VERIFIER_ENDPOINT"].startswith(
        "http://127.0.0.1"
    )
    assert values["PERSONAPLEX_SCENARIO_JUDGE_ENDPOINT"] == OPENROUTER_ENDPOINT
    assert values["PERSONAPLEX_SCENARIO_JUDGE_MODEL"] == OPENROUTER_NEMOTRON_MODEL
    assert values["PERSONAPLEX_SCENARIO_SECONDARY_JUDGE_ENDPOINT"] == OPENROUTER_ENDPOINT
    assert values["PERSONAPLEX_SCENARIO_SECONDARY_JUDGE_MODEL"] == (
        OPENROUTER_NEMOTRON_ULTRA_MODEL
    )
    assert values["PERSONAPLEX_SCENARIO_ADJUDICATOR_ENDPOINT"].startswith(
        "http://127.0.0.1"
    )

    tools_root = Path(__file__).resolve().parents[1] / "tools"
    lane_service = (tools_root / "run-v7-lane-service.sh").read_text(encoding="utf-8")
    certifier_service = (tools_root / "run-v7-certifier-service.sh").read_text(
        encoding="utf-8"
    )
    assert 'source "$OPENROUTER_ENV"' in lane_service
    assert "SYNTHESIZE_INFERENCE_PROVIDER=openrouter" in lane_service
    assert 'SYNTHESIZE_INFERENCE_MODEL="$generative_model"' in lane_service
    assert 'SYNTHESIZE_DIALOGUE_INFERENCE_MODEL="$generative_model"' in lane_service
    assert 'SYNTHESIZE_INFERENCE_MODEL="$PERSONAPLEX_CONTROL_MODEL"' not in lane_service
    assert 'SYNTHESIZE_CERTIFIER_MODEL="$verifier_model"' in certifier_service
    assert 'SYNTHESIZE_CERTIFIER_MODEL="$PERSONAPLEX_CONTROL_MODEL"' not in certifier_service


def test_large_proposer_requires_dedicated_exact_env_binding(monkeypatch) -> None:
    monkeypatch.setenv("PERSONAPLEX_CASCADE_PLANNER_ENDPOINT", OPENROUTER_ENDPOINT)
    monkeypatch.setenv("PERSONAPLEX_CASCADE_PLANNER_MODEL", OPENROUTER_NEMOTRON_MODEL)
    monkeypatch.delenv("PERSONAPLEX_LARGE_PROPOSER_ENDPOINT", raising=False)
    monkeypatch.delenv("PERSONAPLEX_LARGE_PROPOSER_MODEL", raising=False)
    with pytest.raises(ScenarioBlueprintError, match="LARGE_PROPOSER_ENDPOINT"):
        _large_proposer_config()

    monkeypatch.setenv("PERSONAPLEX_LARGE_PROPOSER_ENDPOINT", OPENROUTER_ENDPOINT)
    monkeypatch.setenv("PERSONAPLEX_LARGE_PROPOSER_MODEL", OPENROUTER_NEMOTRON_MODEL)
    with pytest.raises(ScenarioBlueprintError, match="large proposer model must be"):
        _large_proposer_config()

    monkeypatch.setenv(
        "PERSONAPLEX_LARGE_PROPOSER_MODEL", OPENROUTER_NEMOTRON_ULTRA_MODEL
    )
    assert _large_proposer_config() == (
        (OPENROUTER_ENDPOINT,),
        OPENROUTER_NEMOTRON_ULTRA_MODEL,
    )
