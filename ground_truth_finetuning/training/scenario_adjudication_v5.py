"""Three-model semantic adjudication for PersonaPlex v5 scenario admission.

Deterministic validators reject malformed structure before this layer. Semantic
rejection is never entrusted to one model: two independent judges propose a
candidate union and a third, distinct model makes the final bound decision.
"""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Protocol
import json

from jsonschema import Draft202012Validator

from ground_truth_finetuning.training.diverse_cascade import (
    JsonOnlyPlanner,
    PlannerConfig,
    canonical_json,
    content_hash,
    normalize_planner_endpoints,
)
from ground_truth_finetuning.training.scenario_scrutiny import (
    JUDGE_SYSTEM,
    DIMENSION_KEYS,
    FINDING_CODES,
    ScenarioJudge,
    ScenarioScrutinyError,
    _normalize_model_judge_result,
    _scenario_audit_view,
    _scenario_ids,
    validate_judge_result,
)


ADJUDICATION_PROTOCOL_VERSION = (
    "personaplex.scenario-semantic-adjudication.v9-natural-semantic-boundaries"
)
ADJUDICATION_TRACE_SCHEMA = "personaplex.scenario-semantic-adjudication-trace.v2"
PROPOSAL_TRACE_SCHEMA = "personaplex.scenario-semantic-proposals.v1"
PROPOSAL_PROTOCOL_VERSION = "personaplex.scenario-semantic-proposals.v1"

ADJUDICATOR_SYSTEM = JUDGE_SYSTEM + """
You are the final semantic adjudicator, distinct from two proposing judges whose findings are hidden
from you to prevent anchoring. Reassess every candidate against the actual twenty-scenario group.
Reject a candidate only when its supplied contract itself has a material semantic defect. Shared
domain vocabulary, required four-route topology, required typed control vocabulary, or a broad
interaction family are not duplicate evidence. A semantic duplicate requires materially
interchangeable participants, concrete resource, decision pressure, mutable evidence, causal state
transition, and licensed next behavior. Broad abstractions such as "requires verification",
"requires compliance", "resource conflict", "equipment issue", or "must choose a next action" are
invalid duplicate evidence. A pair is materially distinct when its resource, evidence event, or
licensed next behavior differs. Scheduling collapse requires the same operative temporal activity
and decision trajectory, not merely any timing constraint. A mode mismatch requires the premise's
operative activity to contradict its bound mode, not merely mention an adjacent activity. Emit
findings only for the bound candidate IDs; an empty finding set clears every candidate. Do not vote,
repair, rewrite, or invent scenario content.
"""

FINDING_DIMENSIONS = {
    "semantic_near_duplicate": "semanticDiversity",
    "scheduling_template_collapse": "modeAndTemplateDiversity",
    "mode_blueprint_mismatch": "modeAndTemplateDiversity",
    "placeholder_or_identity_company_leakage": "identityAndCompanySafety",
    "unsafe_content": "contentAndTargetLeakageSafety",
    "target_dialogue_like_content": "contentAndTargetLeakageSafety",
    "weak_four_sibling_causal_affordance": "fourSiblingCausalAffordance",
    "incoherent_known_facts": "statePolicyOutcomeCoherence",
    "incoherent_uncertainty": "statePolicyOutcomeCoherence",
    "incoherent_policy_constraints": "statePolicyOutcomeCoherence",
    "incoherent_outcome_space": "statePolicyOutcomeCoherence",
}

BLUEPRINT_AUDIT_FIELDS = (
    "interactionMode",
    "submode",
    "participantRelationship",
    "setting",
    "centralResource",
    "centralTension",
    "evidencePivot",
    "causalMechanism",
    "controlOperator",
    "duplexOpportunity",
)


def _adjudicator_response_schema(candidate_ids: list[str]) -> dict[str, Any]:
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise ScenarioScrutinyError("Adjudicator candidates must be nonempty and unique")
    finding = {
        "type": "object",
        "additionalProperties": False,
        "required": ["code", "rationale", "relatedScenarioIds"],
        "properties": {
            "code": {"enum": list(FINDING_CODES)},
            "rationale": {
                "type": "string",
                "minLength": 1,
            },
            "relatedScenarioIds": {
                "type": "array",
                "minItems": 0,
                "maxItems": 3,
                "uniqueItems": True,
                "items": {"enum": candidate_ids},
            },
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": candidate_ids,
        "properties": {
            scenario_id: {
                "type": "object",
                "additionalProperties": False,
                "required": ["findings"],
                "properties": {
                    "findings": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 2,
                        "uniqueItems": True,
                        "items": finding,
                    }
                },
            }
            for scenario_id in candidate_ids
        },
    }


def _materialize_adjudication(
    raw: Any,
    topic_id: str,
    scenario_ids: list[str],
    candidate_ids: list[str],
) -> dict[str, Any]:
    schema = _adjudicator_response_schema(candidate_ids)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(raw), key=lambda error: list(error.path)
    )
    if errors:
        raise ScenarioScrutinyError(
            f"Adjudicator findings violate bound schema: {errors[0].message}"
        )
    clustered_findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    pair_codes = {"semantic_near_duplicate", "scheduling_template_collapse"}
    for scenario_id in candidate_ids:
        for finding in raw[scenario_id]["findings"]:
            related = [
                value for value in finding["relatedScenarioIds"] if value != scenario_id
            ]
            if finding["code"] in pair_codes and not related:
                raise ScenarioScrutinyError(
                    f"{finding['code']} for {scenario_id} requires concrete related IDs"
                )
            member_ids = sorted(
                {scenario_id, *related}, key=candidate_ids.index
            )
            cluster = {
                "code": finding["code"],
                "rationale": finding["rationale"],
                "scenarioIds": member_ids,
            }
            identity = canonical_json({
                "code": cluster["code"],
                "scenarioIds": cluster["scenarioIds"],
            })
            if identity not in seen:
                seen.add(identity)
                clustered_findings.append(cluster)
    findings = clustered_findings
    failed_dimensions = {FINDING_DIMENSIONS[finding["code"]] for finding in findings}
    result = {
        "topicId": topic_id,
        "groupDecision": "reject" if findings else "pass",
        "groupRationale": (
            "One or more bound candidates have independently confirmed semantic findings."
            if findings
            else "The blind adjudicator confirmed no semantic finding for any candidate."
        ),
        "dimensionVerdicts": {
            dimension: {
                "status": "fail" if dimension in failed_dimensions else "pass",
                "rationale": (
                    "At least one typed adjudicator finding is bound to this dimension."
                    if dimension in failed_dimensions
                    else "No typed adjudicator finding is bound to this dimension."
                ),
            }
            for dimension in DIMENSION_KEYS
        },
        "findings": findings,
    }
    return _normalize_model_judge_result(result, topic_id, scenario_ids)


class ScenarioAdjudicator(Protocol):
    def binding(self) -> dict[str, Any]: ...

    def adjudicate(
        self,
        topic: dict[str, Any],
        scenarios: list[dict[str, Any]],
        source_blueprints: list[dict[str, Any]],
        finding_claims: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


def _model_name(binding: dict[str, Any], label: str) -> str:
    model = binding.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ScenarioScrutinyError(f"{label} binding must expose a non-empty model identity")
    return model.strip()


def _rejected_ids(decision: dict[str, Any]) -> set[str]:
    return {str(item["scenarioId"]) for item in decision["rejected"]}


def _write_immutable_trace(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json(value) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(payload)
    except FileExistsError:
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ScenarioScrutinyError(f"Cannot read adjudication trace: {path}") from error
        if existing != payload:
            raise ScenarioScrutinyError(f"Immutable adjudication trace diverged: {path}")


class AuthenticScenarioAdjudicator:
    """Strict-schema third-model adjudicator with reasoning disabled."""

    def __init__(self, config: PlannerConfig, *, max_attempts: int = 6):
        if max_attempts < 1:
            raise ScenarioScrutinyError("Adjudicator max_attempts must be positive")
        self._planner = JsonOnlyPlanner(config)
        self._max_attempts = max_attempts
        self._binding = {
            "provider": "openai-compatible",
            "endpoints": list(normalize_planner_endpoints(config.endpoint)),
            "model": config.model,
            "reasoning": {"enabled": False},
            "temperature": 0.0,
            "maxTokens": config.max_tokens,
            "maxAttempts": max_attempts,
            "responseFormat": "strict_json_schema",
            "protocolVersion": ADJUDICATION_PROTOCOL_VERSION,
        }

    def binding(self) -> dict[str, Any]:
        return json.loads(canonical_json(self._binding))

    def adjudicate(
        self,
        topic: dict[str, Any],
        scenarios: list[dict[str, Any]],
        source_blueprints: list[dict[str, Any]],
        candidate_ids: list[str],
    ) -> dict[str, Any]:
        scenario_ids = _scenario_ids(scenarios)
        if not candidate_ids:
            raise ScenarioScrutinyError("Adjudicator must not be called with an empty candidate set")
        if not set(candidate_ids).issubset(scenario_ids):
            raise ScenarioScrutinyError("Adjudication candidates are not bound to the topic group")
        schema = _adjudicator_response_schema(candidate_ids)
        prompt = {
            "task": "Adjudicate the complete semantic-rejection candidate union.",
            "topicCard": topic,
            "scenarioContracts": [_scenario_audit_view(row) for row in scenarios],
            "sourceBlueprints": source_blueprints,
            "candidateScenarioIds": candidate_ids,
            "rules": [
                "No proposer findings or rationales are supplied; decide from source contracts only.",
                "Compare every contract with its same-ID source blueprint before judging collapse.",
                "Reassess every candidate from its source contract and full peer group.",
                "Only candidateScenarioIds may appear in findings.",
                "Return one required property per candidate ID; its findings array is independent.",
                "An empty per-ID findings array clears that ID.",
                "Duplicate/template findings must name one to three concrete related IDs.",
                "Do not issue one blanket group finding or copy one rationale across the group.",
                "Finding membership is the sole final rejection signal.",
                "Do not reject required shared control or four-route schema structure.",
                "Do not use broad process abstractions as evidence of semantic duplication.",
                "Do not emit replacement content.",
            ],
        }
        failures: list[str] = []
        for attempt in range(1, self._max_attempts + 1):
            try:
                attempt_prompt = deepcopy(prompt)
                if failures:
                    attempt_prompt["retryFeedback"] = {
                        "attempt": attempt,
                        "previousProtocolDefect": failures[-1],
                        "directive": (
                            "Return the complete exact per-ID object again and correct this "
                            "protocol defect without changing or omitting any bound ID."
                        ),
                    }
                raw = self._planner.call(
                    ADJUDICATOR_SYSTEM,
                    canonical_json(attempt_prompt),
                    schema,
                )
                decision = _materialize_adjudication(
                    raw, topic["topicId"], scenario_ids, candidate_ids
                )
                if not _rejected_ids(decision).issubset(candidate_ids):
                    raise ScenarioScrutinyError("Adjudicator rejected an ID outside the candidate union")
                return decision
            except Exception as error:
                failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
        raise ScenarioScrutinyError(
            f"Scenario adjudicator exhausted {self._max_attempts} attempts for "
            f"{topic['topicId']}: {' | '.join(failures)}"
        )


DECOMPOSED_ADJUDICATION_PROTOCOL_VERSION = (
    "personaplex.scenario-semantic-adjudication.v8-evidence-bound-claim-subcalls"
)
DECOMPOSED_SUBCALL_SCHEMA = "personaplex.scenario-adjudicator-claim-subcall.v2"
DECOMPOSED_ADJUDICATOR_SYSTEM = """You are an independent PersonaPlex finding verifier.
Reasoning mode is disabled. Return only the strict findings-only JSON object. The proposed typed
finding is untrusted and proposer rationale is deliberately hidden. Evaluate only that exact code
against the supplied source-bound evidence. Return the same finding code only when the evidence
directly proves its definition; return an empty findings array otherwise. Never perform an open
audit, substitute another finding, infer defects from absent peers, emit replacement content, or
write dialogue. Shared required schema structure and typed control vocabulary are not defects.
Treat disallowedClaims as future prohibitions, startingState.uncertainty as intentional unresolved
state, and barge-in/overlap/cutoff/repair as conversational-audio timing."""
PAIR_FINDING_CODES = (
    "semantic_near_duplicate",
    "scheduling_template_collapse",
)
SINGLE_FINDING_CODES = tuple(
    code for code in FINDING_CODES if code not in PAIR_FINDING_CODES
)

FINDING_VERIFICATION_RULES = {
    "semantic_near_duplicate": (
        "Confirm only when the two scenario-plus-blueprint records are materially interchangeable "
        "after identifiers are ignored: participants, resource, evidence pivot, decision pressure, "
        "causal transition, and licensed next behavior must all be equivalent."
    ),
    "scheduling_template_collapse": (
        "Confirm only when both records repeat the same operative temporal activity and the same "
        "decision trajectory; generic timing, waiting, or coordination language is insufficient."
    ),
    "mode_blueprint_mismatch": (
        "Confirm only when the operative premise contradicts the same-ID sourceBlueprint "
        "interactionMode. Adjacent activities and duplex opportunities do not establish mismatch."
    ),
    "placeholder_or_identity_company_leakage": (
        "Confirm only when unresolved placeholder text, invented identity/company scripting, or "
        "unlicensed identity/contact content appears in the scenario input."
    ),
    "unsafe_content": (
        "Confirm only when the input requires actionable harmful, illegal, or unauthorized "
        "instructions or licenses an unsafe action; merely discussing a safety topic is insufficient."
    ),
    "target_dialogue_like_content": (
        "Confirm when any input field prescribes, quotes, or leaks exact words that the generated "
        "agent must say. Declarative facts, goals, constraints, and outcome labels are not dialogue."
    ),
    "weak_four_sibling_causal_affordance": (
        "Confirm only when the four supplied outcome routes fail to license materially different "
        "next behavior or causal state transitions. Shared route schema is required and not a defect."
    ),
    "incoherent_known_facts": (
        "Confirm only when two simultaneously known facts materially contradict each other."
    ),
    "incoherent_uncertainty": (
        "Confirm only when an uncertainty is already resolved by a simultaneously known fact or "
        "cannot affect the conversation state."
    ),
    "incoherent_policy_constraints": (
        "Confirm only when policy constraints materially contradict one another or require behavior "
        "that the same scenario forbids."
    ),
    "incoherent_outcome_space": (
        "Confirm only when outcome routes contradict their labels or cannot follow from the stated "
        "evidence/control transitions."
    ),
}


def _findings_only_schema(allowed_codes: tuple[str, ...]) -> dict[str, Any]:
    if not allowed_codes or any(code not in FINDING_CODES for code in allowed_codes):
        raise ScenarioScrutinyError("Findings-only schema has invalid allowed codes")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["findings"],
        "properties": {
            "findings": {
                "type": "array",
                "minItems": 0,
                "maxItems": min(3, len(allowed_codes)),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "rationale"],
                    "properties": {
                        "code": {"enum": list(allowed_codes)},
                        "rationale": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                },
            },
        },
    }


def _validate_findings_only(
    raw: Any, allowed_codes: tuple[str, ...]
) -> list[dict[str, str]]:
    schema = _findings_only_schema(allowed_codes)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(raw), key=lambda error: list(error.path)
    )
    if errors:
        raise ScenarioScrutinyError(
            f"Adjudicator subcall violates strict findings schema: {errors[0].message}"
        )
    normalized: list[dict[str, str]] = []
    seen_codes: set[str] = set()
    for finding in raw["findings"]:
        if finding["code"] not in seen_codes:
            seen_codes.add(finding["code"])
            normalized.append(deepcopy(finding))
    return normalized


def _materialize_cluster_decision(
    topic_id: str,
    scenario_ids: list[str],
    clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    failed_dimensions = {FINDING_DIMENSIONS[item["code"]] for item in clusters}
    rejected: list[dict[str, Any]] = []
    accepted: list[dict[str, str]] = []
    for scenario_id in scenario_ids:
        by_code: dict[str, dict[str, Any]] = {}
        for cluster in clusters:
            if scenario_id not in cluster["scenarioIds"]:
                continue
            code = cluster["code"]
            related = [value for value in cluster["scenarioIds"] if value != scenario_id]
            existing = by_code.get(code)
            if existing is None:
                by_code[code] = {
                    "code": code,
                    "rationale": cluster["rationale"],
                    "relatedScenarioIds": related,
                }
            else:
                existing["relatedScenarioIds"] = [
                    value for value in scenario_ids
                    if value in set(existing["relatedScenarioIds"]) | set(related)
                ]
        if by_code:
            rejected.append({
                "scenarioId": scenario_id,
                "findings": [by_code[code] for code in FINDING_CODES if code in by_code],
            })
        else:
            accepted.append({
                "scenarioId": scenario_id,
                "rationale": "No decomposed typed finding was confirmed for this scenario.",
            })
    normalized = {
        "topicId": topic_id,
        "groupDecision": "reject" if clusters else "pass",
        "groupRationale": (
            "Bound decomposed subcalls confirmed one or more semantic findings."
            if clusters
            else "Bound decomposed subcalls confirmed no semantic findings."
        ),
        "dimensionVerdicts": {
            dimension: {
                "status": "fail" if dimension in failed_dimensions else "pass",
                "rationale": (
                    "At least one typed subcall finding is bound to this dimension."
                    if dimension in failed_dimensions
                    else "No typed subcall finding is bound to this dimension."
                ),
            }
            for dimension in DIMENSION_KEYS
        },
        "accepted": accepted,
        "rejected": rejected,
    }
    validate_judge_result(normalized, topic_id, scenario_ids)
    return normalized


def finding_claims_from_decisions(
    decisions: list[dict[str, Any]], scenario_ids: list[str]
) -> list[dict[str, Any]]:
    order = {scenario_id: index for index, scenario_id in enumerate(scenario_ids)}
    claims: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        validate_judge_result(decision, decision["topicId"], scenario_ids)
        for rejected in decision["rejected"]:
            left = rejected["scenarioId"]
            for finding in rejected["findings"]:
                code = finding["code"]
                if code in PAIR_FINDING_CODES:
                    related = [
                        right for right in finding["relatedScenarioIds"]
                        if right != left and right in order
                    ]
                    if not related:
                        raise ScenarioScrutinyError(
                            f"Proposed {code} for {left} lacks a source-bound peer"
                        )
                    members = [
                        list(sorted((left, right), key=order.__getitem__))
                        for right in related
                    ]
                else:
                    members = [[left]]
                for member_ids in members:
                    claim = {"code": code, "scenarioIds": member_ids}
                    claims[canonical_json(claim)] = claim
    return sorted(
        claims.values(),
        key=lambda claim: (
            order[claim["scenarioIds"][0]],
            order[claim["scenarioIds"][-1]],
            FINDING_CODES.index(claim["code"]),
        ),
    )


class AuthenticDecomposedScenarioAdjudicator:
    """Blueprint-grounded verifier for exact typed proposer claims."""

    def __init__(
        self,
        config: PlannerConfig,
        *,
        checkpoint_root: Path,
        max_attempts: int = 6,
    ) -> None:
        if max_attempts < 1:
            raise ScenarioScrutinyError("Adjudicator max_attempts must be positive")
        self._planner = JsonOnlyPlanner(config)
        self._checkpoint_root = Path(checkpoint_root)
        self._max_attempts = max_attempts
        self._binding = {
            "provider": "openai-compatible",
            "endpoints": list(normalize_planner_endpoints(config.endpoint)),
            "model": config.model,
            "reasoning": {"enabled": False},
            "temperature": 0.0,
            "maxTokens": config.max_tokens,
            "maxAttempts": max_attempts,
            "responseFormat": "strict_json_schema",
            "protocolVersion": DECOMPOSED_ADJUDICATION_PROTOCOL_VERSION,
            "decomposition": "one_source_bound_subcall_per_typed_proposer_claim",
        }

    def binding(self) -> dict[str, Any]:
        return json.loads(canonical_json(self._binding))

    def _call(
        self,
        *,
        kind: str,
        prompt: dict[str, Any],
        allowed_codes: tuple[str, ...],
    ) -> list[dict[str, str]]:
        schema = _findings_only_schema(allowed_codes)
        key = content_hash({
            "protocolVersion": DECOMPOSED_ADJUDICATION_PROTOCOL_VERSION,
            "kind": kind,
            "binding": self._binding,
            "prompt": prompt,
            "responseSchema": schema,
        })
        path = self._checkpoint_root / kind / f"{key[7:]}.json"
        if path.exists():
            try:
                checkpoint = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ScenarioScrutinyError(f"Invalid adjudicator subcall: {path}") from error
            body = dict(checkpoint)
            checkpoint_id = body.pop("checkpointId", None)
            if checkpoint_id != content_hash(body) or body.get("subcallKey") != key:
                raise ScenarioScrutinyError(f"Adjudicator subcall identity mismatch: {path}")
            return _validate_findings_only(body.get("response"), allowed_codes)

        failures: list[str] = []
        for attempt in range(1, self._max_attempts + 1):
            attempt_prompt = deepcopy(prompt)
            if failures:
                attempt_prompt["retryFeedback"] = {
                    "attempt": attempt,
                    "previousProtocolDefect": failures[-1],
                    "directive": "Return the complete findings-only object and correct this defect.",
                }
            try:
                raw = self._planner.call(
                    DECOMPOSED_ADJUDICATOR_SYSTEM,
                    canonical_json(attempt_prompt),
                    schema,
                )
                findings = _validate_findings_only(raw, allowed_codes)
                body = {
                    "schema": DECOMPOSED_SUBCALL_SCHEMA,
                    "protocolVersion": DECOMPOSED_ADJUDICATION_PROTOCOL_VERSION,
                    "subcallKey": key,
                    "kind": kind,
                    "bindingHash": content_hash(self._binding),
                    "promptHash": content_hash(prompt),
                    "responseSchemaHash": content_hash(schema),
                    "response": {"findings": findings},
                    "responseHash": content_hash(findings),
                }
                checkpoint = dict(body)
                checkpoint["checkpointId"] = content_hash(body)
                _write_immutable_trace(path, checkpoint)
                return findings
            except Exception as error:
                failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
        raise ScenarioScrutinyError(
            f"Adjudicator {kind} subcall exhausted attempts: {' | '.join(failures)}"
        )

    def adjudicate(
        self,
        topic: dict[str, Any],
        scenarios: list[dict[str, Any]],
        source_blueprints: list[dict[str, Any]],
        finding_claims: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scenario_ids = _scenario_ids(scenarios)
        if len(source_blueprints) != len(scenarios):
            raise ScenarioScrutinyError("Scenario/blueprint adjudication cardinality mismatch")
        scenario_by_id = {row["scenarioId"]: row for row in scenarios}
        blueprint_by_id = {row["scenarioId"]: row for row in source_blueprints}
        if set(scenario_by_id) != set(blueprint_by_id):
            raise ScenarioScrutinyError("Scenario/blueprint adjudication identity mismatch")

        order = {scenario_id: index for index, scenario_id in enumerate(scenario_ids)}
        normalized_claims: dict[str, dict[str, Any]] = {}
        for raw_claim in finding_claims:
            if not isinstance(raw_claim, dict) or set(raw_claim) != {"code", "scenarioIds"}:
                raise ScenarioScrutinyError("Adjudicator finding claim is malformed")
            code = raw_claim["code"]
            member_ids = raw_claim["scenarioIds"]
            if code not in FINDING_CODES or not isinstance(member_ids, list):
                raise ScenarioScrutinyError("Adjudicator finding claim is not typed")
            expected_size = 2 if code in PAIR_FINDING_CODES else 1
            if (
                len(member_ids) != expected_size
                or len(set(member_ids)) != expected_size
                or any(value not in order for value in member_ids)
            ):
                raise ScenarioScrutinyError("Adjudicator finding claim is not source-bound")
            normalized_ids = sorted(member_ids, key=order.__getitem__)
            claim = {"code": code, "scenarioIds": normalized_ids}
            normalized_claims[canonical_json(claim)] = claim

        clusters: list[dict[str, Any]] = []
        for claim in sorted(
            normalized_claims.values(),
            key=lambda value: (
                order[value["scenarioIds"][0]],
                order[value["scenarioIds"][-1]],
                FINDING_CODES.index(value["code"]),
            ),
        ):
            code = claim["code"]
            member_ids = claim["scenarioIds"]
            evidence: dict[str, Any]
            if len(member_ids) == 1:
                scenario_id = member_ids[0]
                evidence = {
                    "sourceBlueprint": blueprint_by_id[scenario_id],
                    "scenarioContract": _scenario_audit_view(scenario_by_id[scenario_id]),
                }
                kind = "single-claim"
            else:
                left, right = member_ids
                evidence = {
                    "left": {
                        "sourceBlueprint": blueprint_by_id[left],
                        "scenarioContract": _scenario_audit_view(scenario_by_id[left]),
                    },
                    "right": {
                        "sourceBlueprint": blueprint_by_id[right],
                        "scenarioContract": _scenario_audit_view(scenario_by_id[right]),
                    },
                }
                kind = "pair-claim"
            findings = self._call(
                kind=kind,
                prompt={
                    "task": "Verify exactly one untrusted typed finding against source evidence.",
                    "proposedFinding": claim,
                    "definition": FINDING_VERIFICATION_RULES[code],
                    "evidence": evidence,
                    "rules": [
                        "The proposer rationale is hidden and must not be reconstructed.",
                        "Return the proposed code only when its exact definition is directly proved.",
                        "Return an empty findings array for weak, adjacent, or different defects.",
                        "Do not perform an open audit or return any unproposed code.",
                    ],
                },
                allowed_codes=(code,),
            )
            clusters.extend({
                "code": finding["code"],
                "rationale": finding["rationale"],
                "scenarioIds": member_ids,
            } for finding in findings)

        unique: list[dict[str, Any]] = []
        seen: set[str] = set()
        for cluster in clusters:
            identity = canonical_json({
                "code": cluster["code"], "scenarioIds": cluster["scenarioIds"]
            })
            if identity not in seen:
                seen.add(identity)
                unique.append(cluster)
        return _materialize_cluster_decision(topic["topicId"], scenario_ids, unique)


class AdjudicatedScenarioJudge:
    """ScenarioJudge compatible two-proposer plus one-arbiter admission gate."""

    def __init__(
        self,
        primary: ScenarioJudge,
        secondary: ScenarioJudge,
        adjudicator: ScenarioAdjudicator,
        *,
        trace_root: Path,
        blueprint_sets: list[dict[str, Any]],
    ) -> None:
        self._primary = primary
        self._secondary = secondary
        self._adjudicator = adjudicator
        self._trace_root = Path(trace_root)
        if not isinstance(blueprint_sets, list) or not blueprint_sets:
            raise ScenarioScrutinyError("Adjudication requires nonempty source blueprint sets")
        self._blueprint_sets = deepcopy(blueprint_sets)
        self._blueprints_by_id: dict[str, dict[str, Any]] = {}
        for blueprint_set in self._blueprint_sets:
            if not isinstance(blueprint_set, dict) or not isinstance(
                blueprint_set.get("blueprints"), dict
            ):
                raise ScenarioScrutinyError("Source blueprint set is malformed")
            topic_id = blueprint_set.get("topicId")
            for scenario_id, blueprint in blueprint_set["blueprints"].items():
                if scenario_id in self._blueprints_by_id or not isinstance(blueprint, dict):
                    raise ScenarioScrutinyError("Source blueprint IDs must be unique objects")
                missing = [field for field in BLUEPRINT_AUDIT_FIELDS if field not in blueprint]
                if missing:
                    raise ScenarioScrutinyError(
                        f"Source blueprint {scenario_id} lacks audit fields: {missing}"
                    )
                self._blueprints_by_id[str(scenario_id)] = {
                    "scenarioId": str(scenario_id),
                    "topicId": topic_id,
                    **{field: deepcopy(blueprint[field]) for field in BLUEPRINT_AUDIT_FIELDS},
                }
        primary_binding = primary.binding()
        secondary_binding = secondary.binding()
        adjudicator_binding = adjudicator.binding()
        bindings = (primary_binding, secondary_binding, adjudicator_binding)
        if any(binding.get("reasoning") != {"enabled": False} for binding in bindings):
            raise ScenarioScrutinyError("Every semantic judge must explicitly disable reasoning")
        models = {
            _model_name(primary_binding, "primary"),
            _model_name(secondary_binding, "secondary"),
            _model_name(adjudicator_binding, "adjudicator"),
        }
        if len(models) != 3:
            raise ScenarioScrutinyError("Primary, secondary, and adjudicator models must be distinct")
        self._binding = {
            "strategy": "two_proposer_typed_union_then_source_bound_third_model_claim_verification",
            "protocolVersion": ADJUDICATION_PROTOCOL_VERSION,
            "reasoning": {"enabled": False},
            "primary": primary_binding,
            "secondary": secondary_binding,
            "adjudicator": adjudicator_binding,
            "sourceBlueprintSetsHash": content_hash(self._blueprint_sets),
        }

    def binding(self) -> dict[str, Any]:
        return json.loads(canonical_json(self._binding))

    def _trace_path(
        self,
        topic: dict[str, Any],
        scenarios: list[dict[str, Any]],
    ) -> tuple[str, Path]:
        key = content_hash({
            "protocolVersion": ADJUDICATION_PROTOCOL_VERSION,
            "topicCard": topic,
            "scenarioContracts": scenarios,
            "binding": self._binding,
        })
        return key, self._trace_root / str(topic["topicId"]) / f"{key[7:]}.json"

    def _load_or_create_proposals(
        self,
        topic: dict[str, Any],
        scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scenario_ids = _scenario_ids(scenarios)
        primary_binding = self._primary.binding()
        secondary_binding = self._secondary.binding()
        proposal_key = content_hash({
            "protocolVersion": PROPOSAL_PROTOCOL_VERSION,
            "topicCard": topic,
            "scenarioContracts": scenarios,
            "primaryBinding": primary_binding,
            "secondaryBinding": secondary_binding,
        })
        path = (
            self._trace_root
            / str(topic["topicId"])
            / "proposals"
            / f"{proposal_key[7:]}.json"
        )
        if path.exists():
            try:
                trace = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ScenarioScrutinyError(f"Invalid semantic-proposal trace: {path}") from error
            body = dict(trace)
            proposal_id = body.pop("proposalId", None)
            if proposal_id != content_hash(body) or body.get("proposalKey") != proposal_key:
                raise ScenarioScrutinyError(f"Semantic-proposal trace identity mismatch: {path}")
            primary = body.get("primaryDecision")
            secondary = body.get("secondaryDecision")
            validate_judge_result(primary, topic["topicId"], scenario_ids)
            validate_judge_result(secondary, topic["topicId"], scenario_ids)
            return trace

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="scenario-proposer") as pool:
            primary_future = pool.submit(self._primary.audit_topic, topic, scenarios)
            secondary_future = pool.submit(self._secondary.audit_topic, topic, scenarios)
            primary = primary_future.result()
            secondary = secondary_future.result()
        validate_judge_result(primary, topic["topicId"], scenario_ids)
        validate_judge_result(secondary, topic["topicId"], scenario_ids)
        candidate_ids = sorted(_rejected_ids(primary) | _rejected_ids(secondary))
        body = {
            "schema": PROPOSAL_TRACE_SCHEMA,
            "protocolVersion": PROPOSAL_PROTOCOL_VERSION,
            "proposalKey": proposal_key,
            "topicId": topic["topicId"],
            "topicCardHash": content_hash(topic),
            "scenarioGroupHash": content_hash(scenarios),
            "scenarioIds": scenario_ids,
            "primaryBinding": primary_binding,
            "primaryBindingHash": content_hash(primary_binding),
            "secondaryBinding": secondary_binding,
            "secondaryBindingHash": content_hash(secondary_binding),
            "primaryDecision": primary,
            "secondaryDecision": secondary,
            "candidateScenarioIds": candidate_ids,
        }
        trace = dict(body)
        trace["proposalId"] = content_hash(body)
        _write_immutable_trace(path, trace)
        return trace

    def audit_topic(
        self,
        topic: dict[str, Any],
        scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]:
        scenario_ids = _scenario_ids(scenarios)
        source_blueprints: list[dict[str, Any]] = []
        for scenario in scenarios:
            scenario_id = scenario["scenarioId"]
            blueprint = self._blueprints_by_id.get(scenario_id)
            if blueprint is None or blueprint["topicId"] != topic["topicId"]:
                raise ScenarioScrutinyError(
                    f"Missing same-topic source blueprint for {scenario_id}"
                )
            if scenario.get("mode") != blueprint["interactionMode"]:
                raise ScenarioScrutinyError(
                    f"Scenario mode is not bound to its source blueprint: {scenario_id}"
                )
            source_blueprints.append(deepcopy(blueprint))
        key, path = self._trace_path(topic, scenarios)
        if path.exists():
            try:
                trace = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ScenarioScrutinyError(f"Invalid adjudication trace: {path}") from error
            body = dict(trace)
            trace_id = body.pop("traceId", None)
            if trace_id != content_hash(body) or body.get("adjudicationKey") != key:
                raise ScenarioScrutinyError(f"Adjudication trace identity mismatch: {path}")
            decision = body.get("finalDecision")
            validate_judge_result(decision, topic["topicId"], scenario_ids)
            return deepcopy(decision)

        proposals = self._load_or_create_proposals(topic, scenarios)
        primary = proposals["primaryDecision"]
        secondary = proposals["secondaryDecision"]
        proposer_candidate_ids = list(proposals["candidateScenarioIds"])
        finding_claims = finding_claims_from_decisions(
            [primary, secondary], scenario_ids
        )
        candidate_ids = sorted(
            {scenario_id for claim in finding_claims for scenario_id in claim["scenarioIds"]},
            key=scenario_ids.index,
        )
        final = self._adjudicator.adjudicate(
            topic,
            scenarios,
            source_blueprints,
            finding_claims,
        )
        validate_judge_result(final, topic["topicId"], scenario_ids)
        if not _rejected_ids(final).issubset(candidate_ids):
            raise ScenarioScrutinyError("Final adjudication escaped the proposer candidate union")
        body = {
            "schema": ADJUDICATION_TRACE_SCHEMA,
            "protocolVersion": ADJUDICATION_PROTOCOL_VERSION,
            "adjudicationKey": key,
            "topicId": topic["topicId"],
            "topicCardHash": content_hash(topic),
            "scenarioGroupHash": content_hash(scenarios),
            "scenarioIds": scenario_ids,
            "compositeBinding": self._binding,
            "compositeBindingHash": content_hash(self._binding),
            "proposalId": proposals["proposalId"],
            "primaryDecision": primary,
            "secondaryDecision": secondary,
            "proposerCandidateScenarioIds": proposer_candidate_ids,
            "candidateScenarioIds": candidate_ids,
            "findingClaims": finding_claims,
            "adjudicatorInvoked": True,
            "finalDecision": final,
        }
        trace = dict(body)
        trace["traceId"] = content_hash(body)
        _write_immutable_trace(path, trace)
        return deepcopy(final)
