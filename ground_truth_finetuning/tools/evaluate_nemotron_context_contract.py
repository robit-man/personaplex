#!/usr/bin/env python3
"""A/B test Nemotron context following through schema output and tool calls."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence
import urllib.request

from jsonschema import Draft202012Validator

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from ground_truth_finetuning.training.scenario_taxonomy_admission_v5 import (
    TAXONOMY_FINDING_CODES,
    TAXONOMY_FINDING_DEFINITIONS,
    TAXONOMY_JUDGE_SYSTEM,
)


REPORT_SCHEMA = "personaplex.nemotron-context-conformance.v1"
SUBMIT_TOOL_NAME = "submit_taxonomy_judgment"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434/v1/chat/completions"
DEFAULT_NATIVE_ENDPOINT = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "nemotron-3-super:120b"
ATOMIC_QUALITY_FINDING_CODES = (
    "language_or_encoding_corruption",
    "incomplete_or_malformed_field",
    "unnatural_or_placeholder_content",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def content_hash(value: Any) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def response_schema(scenario_ids: Sequence[str]) -> dict[str, Any]:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["findingClusters"],
        "properties": {
            "findingClusters": {
                "type": "array",
                "maxItems": len(TAXONOMY_FINDING_CODES),
                "uniqueItems": True,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "scenarioIds"],
                    "properties": {
                        "code": {"type": "string", "enum": list(TAXONOMY_FINDING_CODES)},
                        "scenarioIds": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": len(scenario_ids),
                            "uniqueItems": True,
                            "items": {"type": "string", "enum": list(scenario_ids)},
                        },
                    },
                },
            }
        },
    }
    Draft202012Validator.check_schema(schema)
    return schema


def atomic_response_schema() -> dict[str, Any]:
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["confirmed"],
        "properties": {"confirmed": {"type": "boolean"}},
    }
    Draft202012Validator.check_schema(schema)
    return schema


def clean_bindings() -> dict[str, dict[str, str]]:
    return {
        "scenario_context_01": {
            "interactionMode": "information_seeking",
            "submode": "asking how late the neighborhood bus runs",
            "participantRelationship": "a new resident asking a longtime neighbor for local transit guidance",
            "setting": "an apartment lobby beside the posted neighborhood map",
            "centralResource": "evening bus timetable",
            "centralTension": "the printed schedule may be outdated, so the resident needs a dependable way home after a late shift",
        },
        "scenario_context_02": {
            "interactionMode": "resource_negotiation",
            "submode": "requesting brief access outside standard storage hours",
            "participantRelationship": "a tenant negotiating an exception with the manager of the building storage room",
            "setting": "the building office shortly before the storage room closes",
            "centralResource": "after-hours storage access window",
            "centralTension": "the tenant needs work equipment tonight while the manager must preserve the building's access policy",
        },
        "scenario_context_03": {
            "interactionMode": "troubleshooting_support",
            "submode": "interpreting a dishwasher drain error",
            "participantRelationship": "a homeowner asking an appliance support technician for troubleshooting guidance",
            "setting": "a kitchen beside a stopped dishwasher displaying error E24",
            "centralResource": "dishwasher drain diagnostic procedure",
            "centralTension": "the homeowner wants to try a safe reset without risking a leak or masking a blocked drain",
        },
        "scenario_context_04": {
            "interactionMode": "scheduling_coordination",
            "submode": "narrowing a utility activation visit window",
            "participantRelationship": "a new tenant coordinating service access with a municipal utility dispatcher",
            "setting": "an unfurnished apartment during the tenant's move-in afternoon",
            "centralResource": "electric service activation appointment",
            "centralTension": "the utility offers only a broad visit window, but the tenant must attend a lease handoff elsewhere that afternoon",
        },
    }


def dirty_bindings() -> dict[str, dict[str, str]]:
    bindings = clean_bindings()
    bindings["scenario_context_02"]["setting"] = (
        "the building office near the storage room on a 工作日夜晚"
    )
    bindings["scenario_context_03"]["centralTension"] = (
        "the homeowner wants to try a safe reset without risking"
    )
    bindings["scenario_context_04"]["participantRelationship"] = (
        "the customer speaking with the COMPANY NAME representative"
    )
    return bindings


def counterfactual_bindings() -> dict[str, dict[str, str]]:
    bindings = clean_bindings()
    bindings["scenario_context_01"]["setting"] = (
        "an apartment lobby beside the map during 晚间通勤"
    )
    bindings["scenario_context_02"]["centralTension"] = (
        "the tenant needs work equipment tonight while"
    )
    bindings["scenario_context_03"]["participantRelationship"] = (
        "a homeowner speaking with the COMPANY NAME support person"
    )
    return bindings


def cases() -> list[dict[str, Any]]:
    return [
        {
            "caseId": "clean_pass",
            "bindings": clean_bindings(),
            "requiredClaims": {},
            "cleanIds": sorted(clean_bindings()),
        },
        {
            "caseId": "typed_quality_defects",
            "bindings": dirty_bindings(),
            "requiredClaims": {
                "language_or_encoding_corruption": ["scenario_context_02"],
                "incomplete_or_malformed_field": ["scenario_context_03"],
                "unnatural_or_placeholder_content": ["scenario_context_04"],
            },
            "cleanIds": ["scenario_context_01"],
        },
        {
            "caseId": "counterfactual_id_rebinding",
            "bindings": counterfactual_bindings(),
            "requiredClaims": {
                "language_or_encoding_corruption": ["scenario_context_01"],
                "incomplete_or_malformed_field": ["scenario_context_02"],
                "unnatural_or_placeholder_content": ["scenario_context_03"],
            },
            "cleanIds": ["scenario_context_04"],
        },
    ]


def context_for(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "task": "Independently admit or reject this complete bound Stage-T taxonomy test view.",
        "topicCard": {
            "topicId": "topic_context_contract",
            "domain": "moving and settling in",
            "interactionModes": [
                "information_seeking",
                "resource_negotiation",
                "troubleshooting_support",
                "scheduling_coordination",
            ],
            "safeStakes": ["minor delay", "temporary inconvenience"],
        },
        "boundTaxonomyJudgeView": {
            "topicId": "topic_context_contract",
            "requiredLanguage": "English",
            "qualityContract": {
                "completeGrammaticalFields": True,
                "naturalConversationSetup": True,
                "placeholdersAndMetaTextForbidden": True,
                "mixedLanguageAndEncodingCorruptionForbidden": True,
            },
            "scenarioIds": sorted(case["bindings"]),
            "scenarioBindings": deepcopy(case["bindings"]),
        },
        "typedFindingContract": {
            "soleSemanticRejectionSignal": "findingClusters",
            "codes": list(TAXONOMY_FINDING_CODES),
            "definitions": dict(TAXONOMY_FINDING_DEFINITIONS),
            "exactImplicatedIdsRequired": True,
            "oneClusterPerCode": True,
        },
    }


def post_json(endpoint: str, payload: Mapping[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=canonical_json(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise ValueError("Ollama response envelope is not an object")
    return value


def parse_schema_response(envelope: Mapping[str, Any]) -> Any:
    return json.loads(envelope["choices"][0]["message"]["content"])


def parse_tool_response(envelope: Mapping[str, Any]) -> Any:
    message = envelope["choices"][0]["message"]
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("model did not return exactly one tool call")
    function = calls[0].get("function", {})
    if function.get("name") != SUBMIT_TOOL_NAME:
        raise ValueError("model called the wrong tool")
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        return json.loads(arguments)
    if isinstance(arguments, dict):
        return arguments
    raise ValueError("tool arguments are neither an object nor a JSON string")


def _tool_arguments(function: Mapping[str, Any]) -> dict[str, Any]:
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("typed tool arguments are neither an object nor a JSON string")
    return arguments


def typed_finding_tools(scenario_ids: Sequence[str]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for code in TAXONOMY_FINDING_CODES:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"report_{code}",
                    "description": TAXONOMY_FINDING_DEFINITIONS[code],
                    "parameters": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["scenarioIds"],
                        "properties": {
                            "scenarioIds": {
                                "type": "array",
                                "minItems": 1,
                                "uniqueItems": True,
                                "items": {"type": "string", "enum": list(scenario_ids)},
                            }
                        },
                    },
                },
            }
        )
    tools.append(
        {
            "type": "function",
            "function": {
                "name": "approve_taxonomy",
                "description": (
                    "Approve the complete taxonomy only when no typed finding applies to any "
                    "scenario. Never call this with a report tool."
                ),
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {},
                },
            },
        }
    )
    return tools


def parse_native_typed_tools(envelope: Mapping[str, Any]) -> dict[str, Any]:
    message = envelope.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("native Ollama response lacks message")
    calls = message.get("tool_calls")
    if not isinstance(calls, list) or not calls:
        raise ValueError("native model returned no typed terminal tool")
    clusters: list[dict[str, Any]] = []
    seen_codes: set[str] = set()
    approved = False
    for call in calls:
        function = call.get("function") if isinstance(call, Mapping) else None
        if not isinstance(function, Mapping):
            raise ValueError("native tool call lacks function")
        name = function.get("name")
        arguments = _tool_arguments(function)
        if name == "approve_taxonomy":
            if arguments:
                raise ValueError("approve_taxonomy arguments must be empty")
            approved = True
            continue
        if not isinstance(name, str) or not name.startswith("report_"):
            raise ValueError(f"unknown typed terminal tool: {name}")
        code = name.removeprefix("report_")
        if code not in TAXONOMY_FINDING_CODES:
            raise ValueError(f"unknown finding tool code: {code}")
        if code in seen_codes:
            raise ValueError(f"finding tool called more than once: {code}")
        seen_codes.add(code)
        clusters.append({"code": code, "scenarioIds": arguments.get("scenarioIds")})
    if approved and clusters:
        raise ValueError("approve_taxonomy cannot accompany finding tools")
    if not approved and not clusters:
        raise ValueError("typed tool response contains no terminal decision")
    clusters.sort(key=canonical_json)
    return {"findingClusters": clusters}


def payload_for(
    *, model: str, mode: str, schema: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    system = TAXONOMY_JUDGE_SYSTEM
    if mode == "typed_tools_native":
        scenario_ids = context["boundTaxonomyJudgeView"]["scenarioIds"]
        return {
            "model": model,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0, "num_predict": 512},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        system
                        + " Reasoning and narration are disabled. Return no prose. If the full set "
                        "passes, call approve_taxonomy exactly once. Otherwise call each applicable "
                        "report_<finding_code> tool exactly once, in parallel, with all and only the "
                        "affected scenario IDs. Never call approve_taxonomy with a report tool."
                    ),
                },
                {"role": "user", "content": canonical_json(context)},
            ],
            "tools": typed_finding_tools(scenario_ids),
        }
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 1536,
        "reasoning": {"effort": "none"},
        "reasoning_effort": "none",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": canonical_json(context)},
        ],
    }
    if mode == "schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "personaplex_context_contract",
                "strict": True,
                "schema": schema,
            },
        }
    elif mode == "tool":
        payload["messages"][0]["content"] += (
            " Call submit_taxonomy_judgment exactly once with the complete typed decision. "
            "Do not answer in assistant content and do not call any other function."
        )
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": SUBMIT_TOOL_NAME,
                    "description": (
                        "Submit the final Stage-T taxonomy finding clusters after inspecting the "
                        "entire source-bound view. This is the only accepted terminal action."
                    ),
                    "parameters": schema,
                },
            }
        ]
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return payload


def atomic_payload_for(
    *,
    model: str,
    code: str,
    scenario_id: str,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    schema = atomic_response_schema()
    atomic_context = {
        "task": "Verify one exact typed finding for one exact candidate scenario.",
        "topicCard": context["topicCard"],
        "boundTaxonomyJudgeView": context["boundTaxonomyJudgeView"],
        "findingCode": code,
        "findingDefinition": TAXONOMY_FINDING_DEFINITIONS[code],
        "candidateScenarioId": scenario_id,
        "candidateEvidence": context["boundTaxonomyJudgeView"]["scenarioBindings"][
            scenario_id
        ],
        "decisionContract": {
            "confirmedTrueOnlyWhenCandidateDirectlySatisfiesExactDefinition": True,
            "confirmedFalseOtherwise": True,
            "otherFindingCodesAreOutOfScope": True,
            "sourceBoundEvidenceOnly": True,
        },
    }
    return {
        "model": model,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 256,
        "reasoning": {"effort": "none"},
        "reasoning_effort": "none",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are one atomic Stage-T source-bound claim verifier. Reasoning and "
                    "narration are disabled. Evaluate only whether candidateScenarioId directly "
                    f"satisfies findingCode={code}. The normative definition is: "
                    f"{TAXONOMY_FINDING_DEFINITIONS[code]} Return confirmed=true only when the "
                    "candidate evidence directly satisfies that exact definition; otherwise return "
                    "confirmed=false. The complete view is supplied only when the definition needs "
                    "cross-scenario comparison. Do not evaluate other codes, repair content, use "
                    "lexical shortcuts, or emit prose."
                ),
            },
            {"role": "user", "content": canonical_json(atomic_context)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "personaplex_atomic_claim_verdict",
                "strict": True,
                "schema": schema,
            },
        },
    }


def claim_map(value: Mapping[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for cluster in value["findingClusters"]:
        result.setdefault(cluster["code"], set()).update(cluster["scenarioIds"])
    return result


def semantic_match(value: Mapping[str, Any], case: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    actual = claim_map(value)
    required = {code: set(ids) for code, ids in case["requiredClaims"].items()}
    if not required and actual:
        failures.append("clean case produced findings")
    for code, ids in required.items():
        missing = ids.difference(actual.get(code, set()))
        if missing:
            failures.append(f"{code} missed IDs {sorted(missing)}")
    clean_ids = set(case["cleanIds"])
    accused_clean = (
        sorted(clean_ids.intersection(set().union(*actual.values()))) if actual else []
    )
    if accused_clean:
        failures.append(f"clean IDs were accused {accused_clean}")
    return not failures, failures


def evaluate_atomic_once(
    *, endpoint: str, model: str, case: Mapping[str, Any], timeout: int
) -> dict[str, Any]:
    mode = "atomic_schema"
    ids = sorted(case["bindings"])
    context = context_for(case)
    payloads = {
        (code, scenario_id): atomic_payload_for(
            model=model,
            code=code,
            scenario_id=scenario_id,
            context=context,
        )
        for code in ATOMIC_QUALITY_FINDING_CODES
        for scenario_id in ids
    }
    started = time.monotonic()
    record: dict[str, Any] = {
        "caseId": case["caseId"],
        "mode": mode,
        "requestHash": content_hash(
            {
                f"{code}:{scenario_id}": content_hash(payload)
                for (code, scenario_id), payload in payloads.items()
            }
        ),
    }

    def call_one(code: str, scenario_id: str) -> dict[str, Any]:
        envelope = post_json(endpoint, payloads[(code, scenario_id)], timeout)
        message = envelope["choices"][0]["message"]
        value = json.loads(message["content"])
        errors = list(Draft202012Validator(atomic_response_schema()).iter_errors(value))
        if errors:
            raise ValueError(
                f"{code}:{scenario_id} schema failure: "
                + " | ".join(error.message for error in errors)
            )
        return {
            "code": code,
            "scenarioId": scenario_id,
            "confirmed": bool(value["confirmed"]),
            "assistantMessage": message,
            "usage": envelope.get("usage", {}),
            "responseHash": content_hash(value),
        }

    try:
        subcalls: dict[tuple[str, str], dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="atomic-context") as pool:
            futures = {
                pool.submit(call_one, code, scenario_id): (code, scenario_id)
                for code in ATOMIC_QUALITY_FINDING_CODES
                for scenario_id in ids
            }
            for future in as_completed(futures):
                key = futures[future]
                subcalls[key] = future.result()
        clusters = [
            {
                "code": code,
                "scenarioIds": [
                    scenario_id
                    for scenario_id in ids
                    if subcalls[(code, scenario_id)]["confirmed"]
                ],
            }
            for code in ATOMIC_QUALITY_FINDING_CODES
            if any(subcalls[(code, scenario_id)]["confirmed"] for scenario_id in ids)
        ]
        clusters.sort(key=canonical_json)
        value = {"findingClusters": clusters}
        full_errors = list(
            Draft202012Validator(response_schema(ids)).iter_errors(value)
        )
        if full_errors:
            raise ValueError(
                "merged atomic schema failure: "
                + " | ".join(error.message for error in full_errors)
            )
        semantic_pass, failures = semantic_match(value, case)
        reasoning_off = all(
            not bool(item["assistantMessage"].get("thinking"))
            and not bool(item["assistantMessage"].get("reasoning"))
            for item in subcalls.values()
        )
        record.update(
            {
                "latencySeconds": round(time.monotonic() - started, 6),
                "transportPass": True,
                "parsePass": True,
                "schemaPass": True,
                "semanticPass": semantic_pass,
                "reasoningOffPass": reasoning_off,
                "failures": failures,
                "value": value,
                "responseHash": content_hash(value),
                "atomicCalls": [
                    subcalls[(code, scenario_id)]
                    for code in ATOMIC_QUALITY_FINDING_CODES
                    for scenario_id in ids
                ],
                "usage": {
                    "prompt_tokens": sum(
                        int(item["usage"].get("prompt_tokens", 0))
                        for item in subcalls.values()
                    ),
                    "completion_tokens": sum(
                        int(item["usage"].get("completion_tokens", 0))
                        for item in subcalls.values()
                    ),
                },
            }
        )
    except Exception as error:
        record.update(
            {
                "latencySeconds": round(time.monotonic() - started, 6),
                "transportPass": False,
                "parsePass": False,
                "schemaPass": False,
                "semanticPass": False,
                "reasoningOffPass": False,
                "failures": [f"{type(error).__name__}: {error}"],
            }
        )
    return record


def evaluate_once(
    *,
    endpoint: str,
    native_endpoint: str,
    model: str,
    mode: str,
    case: Mapping[str, Any],
    timeout: int,
) -> dict[str, Any]:
    if mode == "atomic_schema":
        return evaluate_atomic_once(
            endpoint=endpoint,
            model=model,
            case=case,
            timeout=timeout,
        )
    ids = sorted(case["bindings"])
    schema = response_schema(ids)
    payload = payload_for(model=model, mode=mode, schema=schema, context=context_for(case))
    started = time.monotonic()
    record: dict[str, Any] = {
        "caseId": case["caseId"],
        "mode": mode,
        "requestHash": content_hash(payload),
    }
    try:
        request_endpoint = native_endpoint if mode == "typed_tools_native" else endpoint
        envelope = post_json(request_endpoint, payload, timeout)
        record["latencySeconds"] = round(time.monotonic() - started, 6)
        record["transportPass"] = True
        message = (
            envelope["message"]
            if mode == "typed_tools_native"
            else envelope["choices"][0]["message"]
        )
        record["assistantMessage"] = message
        if mode == "schema":
            value = parse_schema_response(envelope)
        elif mode == "tool":
            value = parse_tool_response(envelope)
        else:
            value = parse_native_typed_tools(envelope)
        record["parsePass"] = True
        errors = sorted(
            Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.path)
        )
        record["schemaPass"] = not errors
        if errors:
            record["failures"] = [error.message for error in errors]
            record["semanticPass"] = False
        else:
            semantic_pass, failures = semantic_match(value, case)
            record["semanticPass"] = semantic_pass
            record["failures"] = failures
        record["value"] = value
        record["responseHash"] = content_hash(value)
        record["usage"] = envelope.get("usage", {})
        explicit_reasoning = message.get("thinking") or message.get("reasoning")
        narrative_content = message.get("content")
        record["reasoningOffPass"] = not bool(explicit_reasoning) and (
            mode == "schema"
            or not isinstance(narrative_content, str)
            or not narrative_content.strip()
        )
    except Exception as error:
        record["latencySeconds"] = round(time.monotonic() - started, 6)
        record.setdefault("transportPass", False)
        record.setdefault("parsePass", False)
        record.setdefault("schemaPass", False)
        record.setdefault("semanticPass", False)
        record.setdefault("reasoningOffPass", False)
        record["failures"] = [f"{type(error).__name__}: {error}"]
    return record


def summarize(records: Sequence[Mapping[str, Any]], mode: str) -> dict[str, Any]:
    selected = [record for record in records if record["mode"] == mode]
    total = len(selected)
    return {
        "attempts": total,
        "transportPassRate": sum(bool(row["transportPass"]) for row in selected) / total,
        "parsePassRate": sum(bool(row["parsePass"]) for row in selected) / total,
        "schemaPassRate": sum(bool(row["schemaPass"]) for row in selected) / total,
        "semanticPassRate": sum(bool(row["semanticPass"]) for row in selected) / total,
        "reasoningOffPassRate": sum(bool(row["reasoningOffPass"]) for row in selected) / total,
        "contractPassRate": sum(
            bool(row["semanticPass"] and row["reasoningOffPass"]) for row in selected
        ) / total,
        "meanLatencySeconds": round(
            sum(float(row["latencySeconds"]) for row in selected) / total, 6
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--native-endpoint", default=DEFAULT_NATIVE_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 100:
        parser.error("--repetitions must be in [1,100]")

    modes = ("schema", "atomic_schema")
    records: list[dict[str, Any]] = []
    for repetition in range(1, args.repetitions + 1):
        for case in cases():
            for mode in modes:
                record = evaluate_once(
                    endpoint=args.endpoint,
                    native_endpoint=args.native_endpoint,
                    model=args.model,
                    mode=mode,
                    case=case,
                    timeout=args.timeout,
                )
                record["repetition"] = repetition
                records.append(record)
                print(
                    canonical_json(
                        {
                            "event": "nemotron_context_conformance",
                            "caseId": record["caseId"],
                            "mode": mode,
                            "repetition": repetition,
                            "semanticPass": record["semanticPass"],
                            "latencySeconds": record["latencySeconds"],
                        }
                    ),
                    flush=True,
                )

    summaries = {mode: summarize(records, mode) for mode in modes}
    schema_score = summaries["schema"]["contractPassRate"]
    challenger_modes = tuple(mode for mode in modes if mode != "schema")
    best_tool = max(
        challenger_modes,
        key=lambda mode: (
            summaries[mode]["contractPassRate"],
            -summaries[mode]["meanLatencySeconds"],
        ),
    )
    best_tool_score = summaries[best_tool]["contractPassRate"]
    if best_tool_score >= 0.95 and best_tool_score > schema_score:
        selected_transport = best_tool
    elif schema_score >= 0.95:
        selected_transport = "schema"
    else:
        selected_transport = "none"
    body = {
        "schema": REPORT_SCHEMA,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "endpoint": args.endpoint,
        "model": args.model,
        "reasoning": {"enabled": False},
        "repetitions": args.repetitions,
        "caseContractHash": content_hash(cases()),
        "findingDefinitionsHash": content_hash(TAXONOMY_FINDING_DEFINITIONS),
        "records": records,
        "summaries": summaries,
        "selection": {
            "transport": selected_transport,
            "policy": (
                "tools must reach 0.95 contract adherence and strictly outperform schema; "
                "otherwise schema must reach 0.95 or no transport is admitted"
            ),
        },
    }
    report = dict(body)
    report["reportHash"] = content_hash(body)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if selected_transport != "none" else 2


if __name__ == "__main__":
    raise SystemExit(main())
