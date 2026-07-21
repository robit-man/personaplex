"""Independent admission boundary for PersonaPlex v5 Stage-T taxonomies.

Stage T proposes semantic anchors. This module presents those anchors to an
independently bound model judge, repairs only IDs named by typed findings, and
persists the raw/judgment/repair/admission lineage before Stage P may consume
the anchors. Host code validates structure and identity but never synthesizes
or lexically judges semantic content.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import BoundedSemaphore
from typing import Any, Mapping, Protocol, Sequence

from jsonschema import Draft202012Validator

from ground_truth_finetuning.training.scenario_blueprint_v5 import (
    BLUEPRINTS_PER_TOPIC,
    TAXONOMY_FIELDS,
    TAXONOMY_MAX_OUTPUT_TOKENS,
    TAXONOMY_WIRE_KEYS,
    InvalidModelOutput,
    ModelTransportUnavailable,
    ScenarioBlueprintError,
    StrictSchemaModel,
    _call_model,
    _checkpoint_body_hash,
    _checkpoint_path,
    _coverage_assignment,
    _model_binding,
    _raise_schema_errors,
    _topic_enums,
    _write_immutable_json,
    assert_no_target_fields,
    build_taxonomy_response_schema,
    canonical_json,
    content_hash,
    generate_topic_taxonomy,
    read_json,
    scenario_ids_for_topic,
    validate_taxonomy_anchors,
)


TAXONOMY_ADMISSION_PROTOCOL_VERSION = "upstream-taxonomy-admission-v14-semantic-boundaries"
TAXONOMY_JUDGE_PROTOCOL_VERSION = "typed-taxonomy-findings-v11-authoritative-atomic-repair"
TAXONOMY_REPAIR_PROTOCOL_VERSION = "typed-taxonomy-targeted-repair-v8-semantic-boundaries"

TAXONOMY_JUDGE_VIEW_SCHEMA = "personaplex.scenario-taxonomy-judge-view.v6"
TAXONOMY_JUDGMENT_CHECKPOINT_SCHEMA = (
    "personaplex.scenario-taxonomy-judgment-checkpoint.v14"
)
TAXONOMY_REPAIR_CHECKPOINT_SCHEMA = (
    "personaplex.scenario-taxonomy-repair-checkpoint.v9"
)
TAXONOMY_ADMISSION_CHECKPOINT_SCHEMA = (
    "personaplex.scenario-taxonomy-admission-checkpoint.v14"
)

TAXONOMY_JUDGE_MAX_OUTPUT_TOKENS = 1536
MIN_REPAIRED_ANCHOR_FIELDS = 2

RELATIONAL_TAXONOMY_FINDING_CODES = (
    "mode_submode_mismatch",
    "field_role_misuse",
    "semantic_duplicate_template_collapse",
    "implausible_anchor",
)

ATOMIC_QUALITY_TAXONOMY_FINDING_CODES = (
    "language_or_encoding_corruption",
    "incomplete_or_malformed_field",
    "unnatural_or_placeholder_content",
)

TAXONOMY_FINDING_CODES = (
    *RELATIONAL_TAXONOMY_FINDING_CODES,
    *ATOMIC_QUALITY_TAXONOMY_FINDING_CODES,
)

TAXONOMY_FINDING_DEFINITIONS = {
    "mode_submode_mismatch": (
        "Confirm only if the concrete submode and its semantic anchor fields fail to instantiate "
        "the exact assigned interactionMode. Different wording or a more specific activity is not "
        "a mismatch when it genuinely realizes that mode."
    ),
    "field_role_misuse": (
        "Confirm only if a value materially fails its declared role: participantRelationship must "
        "be a relationship, setting a place/context, centralResource a concrete object/information/"
        "service, and centralTension a conflict or decision pressure. Creative specificity passes."
    ),
    "semantic_duplicate_template_collapse": (
        "Confirm only if every implicated anchor is materially interchangeable across assigned mode, "
        "submode, relationship, setting, resource, and tension. Shared domain vocabulary or structure "
        "alone is insufficient."
    ),
    "implausible_anchor": (
        "Confirm only if the complete anchor cannot plausibly occur within the topic and safe-stakes "
        "boundary. Unusual but coherent scenarios pass."
    ),
    "language_or_encoding_corruption": (
        "Confirm when any taxonomy field is not fluent English throughout, contains an unexplained "
        "fragment from another language or script, contains mojibake, or splices malformed tokens. "
        "Ordinary established proper nouns and loanwords are not defects."
    ),
    "incomplete_or_malformed_field": (
        "Confirm when any field ends mid-word or mid-thought, has a dangling conjunction or missing "
        "semantic head, or is otherwise not a complete grammatical phrase with a recoverable meaning. "
        "Concise but complete noun phrases pass."
    ),
    "unnatural_or_placeholder_content": (
        "Confirm when an anchor contains a placeholder, meta-instruction, generated-field label, "
        "template residue, or wording too malformed or bureaucratically synthetic to describe a "
        "plausible real conversation setup. A literal unresolved slot such as COMPANY NAME, "
        "[NAME], [COMPANY], TBD, or an equivalent placeholder is always a confirmed defect even "
        "when the surrounding relationship remains understandable. Specific natural professional "
        "terminology passes."
    ),
}

TAXONOMY_CLAIM_VERIFIER_SYSTEM = """You are the final independent verifier for one typed Stage-T
taxonomy finding. Reasoning mode is disabled. The proposed finding is untrusted and carries no
proposer rationale. Evaluate only its exact code and exact source-bound scenario IDs against the
supplied topic and anchor bindings. Return confirmed=true only when the evidence directly satisfies
the supplied definition; otherwise return false. Do not perform an open audit, substitute another
finding, repair content, use lexical matching, write dialogue, or infer missing facts."""

TAXONOMY_JUDGE_SYSTEM = """You are an independent Stage-T taxonomy admission judge.
Judge the complete twenty-ID set in the supplied bound judge view. Return only typed finding
clusters from the strict schema. Check whether each submode materially realizes its assigned
interactionMode, whether relationship/setting/resource/tension values perform their declared
field roles, whether anchors are semantic duplicates or one collapsed template, and whether an
anchor is plausible for the topic. Judge only the supplied relational finding codes; local language,
completeness, and placeholder quality are evaluated independently one scenario and one code at a
time. Group every affected ID under one cluster per finding code. A semantic duplicate/template
collapse necessarily requires at least two materially interchangeable scenario IDs; never use that
code for one scenario. Shared vocabulary alone is not a defect. Do not repair or generate anchors,
apply lexical rules, write dialogue, or emit target responses. An empty finding array is the sole
pass signal; every rejection must be represented by a typed code and exact IDs."""

TAXONOMY_ATOMIC_QUALITY_SYSTEM = """You are an independent Stage-T taxonomy quality verifier.
Reasoning mode is disabled. Evaluate exactly one source-bound scenario against exactly one supplied
quality finding definition. Return confirmed=true only when that exact defect is directly present.
Otherwise return false. Do not audit for other defects, compare other scenarios, repair content,
apply lexical matching, generate text, or include rationale. The strict boolean object is the only
valid response."""

TAXONOMY_RELATIONAL_SCRUTINY_SYSTEM = """You are the focused Stage-T relational claim
scrutinizer. Reasoning mode is disabled. A weaker set-level proposer supplied one untrusted typed
claim. Evaluate only that exact code, exact source-bound scenario IDs, and supplied definition.
Return confirmed=true only when every required condition is directly demonstrated by the bound
evidence. For semantic_duplicate_template_collapse, all implicated anchors must be materially
interchangeable across their assigned mode, submode, relationship, setting, resource, and tension;
shared mode, domain, vocabulary, or sentence structure is insufficient. Return false when the claim
is overstated, partially supported, or contradicted by material distinctions. Do not search for other
defects, repair content, generate text, or include rationale. Return only the strict boolean object."""

TAXONOMY_ATOMIC_WITNESS_SYSTEM = """You are a fresh source-bound witness for one Stage-T taxonomy
quality claim first detected in a separate model pass. Reasoning mode is disabled. Evaluate exactly the
supplied code, scenario ID, definition, and source-bound anchor evidence. Return confirmed=true only
when that exact quality defect is directly present; otherwise return false. Do not defer to the first
pass, audit for other defects, repair content, generate text, or include rationale. Return only the
strict boolean object."""

TAXONOMY_REPAIR_SYSTEM = """Repair only the Stage-T taxonomy IDs required by the strict schema.
Return only the canonical strict JSON object. Resolve every supplied typed finding against the full
immutable twenty-ID judge view. Regenerate every taxonomy field in every repaired anchor while
preserving the deterministic assigned interactionMode, and leave every unlisted anchor absent.
Use the explicit canonical property names submode, participantRelationship, setting, centralResource,
and centralTension. For every returned anchor, changedFields must contain those five names exactly
once; every generated value must differ from its listed forbidden parent value and the declaration
must match the actual delta.
participantRelationship must identify both participant roles and their relationship. centralResource
must be a concise concrete noun phrase, never dialogue, an apology, speech act, target response, topic,
or interaction mode. Every field must be a complete phrase. Create topic-valid semantic anchors; do
not use lexical substitutions, templates, dialogue, utterances, scripts, target responses, target
text/audio, or host-generated placeholders. Every value must be fluent natural English with no
foreign-script fragments, encoding corruption, dangling clauses, missing words, or cut-off tokens.
Read each returned field as a standalone phrase before returning it; if it is not complete and
natural, rewrite it rather than closing valid JSON around malformed prose. No field has a character
or word ceiling. centralTension must be a complete conflict clause, not a clipped prefix. Close every
JSON string only after reaching a natural semantic boundary; use terminal punctuation where natural.
The overall output-token budget bounds the response, never an individual string boundary."""

TAXONOMY_PROPOSER_PROTOCOL_VERSION = "taxonomy-relational-proposer-v1"
TAXONOMY_ATOMIC_QUALITY_PROTOCOL_VERSION = "taxonomy-atomic-quality-v1"
TAXONOMY_ATOMIC_WITNESS_PROTOCOL_VERSION = "taxonomy-atomic-quality-witness-v1"
TAXONOMY_RELATIONAL_SCRUTINY_PROTOCOL_VERSION = "taxonomy-relational-scrutiny-v1"
TAXONOMY_CLAIM_VERIFIER_PROTOCOL_VERSION = "taxonomy-claim-verifier-v1"

TAXONOMY_PROPOSER_SOURCE_HASH = content_hash({
    "system": TAXONOMY_JUDGE_SYSTEM,
    "definitions": {
        code: TAXONOMY_FINDING_DEFINITIONS[code]
        for code in RELATIONAL_TAXONOMY_FINDING_CODES
    },
})
TAXONOMY_ATOMIC_QUALITY_SOURCE_HASH = content_hash({
    "system": TAXONOMY_ATOMIC_QUALITY_SYSTEM,
    "definitions": {
        code: TAXONOMY_FINDING_DEFINITIONS[code]
        for code in ATOMIC_QUALITY_TAXONOMY_FINDING_CODES
    },
    "unit": "one_source_bound_scenario_by_one_finding_code",
})
TAXONOMY_ATOMIC_WITNESS_SOURCE_HASH = content_hash({
    "system": TAXONOMY_ATOMIC_WITNESS_SYSTEM,
    "definitions": {
        code: TAXONOMY_FINDING_DEFINITIONS[code]
        for code in ATOMIC_QUALITY_TAXONOMY_FINDING_CODES
    },
    "unit": "one_source_bound_positive_claim",
    "role": "fresh_strong_model_second_pass",
})
TAXONOMY_RELATIONAL_SCRUTINY_SOURCE_HASH = content_hash({
    "system": TAXONOMY_RELATIONAL_SCRUTINY_SYSTEM,
    "definitions": {
        code: TAXONOMY_FINDING_DEFINITIONS[code]
        for code in RELATIONAL_TAXONOMY_FINDING_CODES
    },
    "unit": "one_source_bound_proposed_claim",
})
TAXONOMY_CLAIM_VERIFIER_SOURCE_HASH = content_hash({
    "system": TAXONOMY_CLAIM_VERIFIER_SYSTEM,
    "definitions": TAXONOMY_FINDING_DEFINITIONS,
    "unit": "one_source_bound_confirmed_claim",
})
TAXONOMY_JUDGE_SOURCE_HASH = content_hash({
    "proposer": {
        "protocol": TAXONOMY_PROPOSER_PROTOCOL_VERSION,
        "sourceHash": TAXONOMY_PROPOSER_SOURCE_HASH,
    },
    "atomicQuality": {
        "protocol": TAXONOMY_ATOMIC_QUALITY_PROTOCOL_VERSION,
        "sourceHash": TAXONOMY_ATOMIC_QUALITY_SOURCE_HASH,
    },
    "atomicQualityWitness": {
        "protocol": TAXONOMY_ATOMIC_WITNESS_PROTOCOL_VERSION,
        "sourceHash": TAXONOMY_ATOMIC_WITNESS_SOURCE_HASH,
    },
    "relationalScrutiny": {
        "protocol": TAXONOMY_RELATIONAL_SCRUTINY_PROTOCOL_VERSION,
        "sourceHash": TAXONOMY_RELATIONAL_SCRUTINY_SOURCE_HASH,
    },
    "claimVerifier": {
        "protocol": TAXONOMY_CLAIM_VERIFIER_PROTOCOL_VERSION,
        "sourceHash": TAXONOMY_CLAIM_VERIFIER_SOURCE_HASH,
    },
    "findingPartition": {
        "relational": list(RELATIONAL_TAXONOMY_FINDING_CODES),
        "atomicQuality": list(ATOMIC_QUALITY_TAXONOMY_FINDING_CODES),
    },
    "repairLineagePolicy": "every-cycle-rejudges-full-set-and-repairs-all-current-confirmed-findings",
})
TAXONOMY_REPAIR_SOURCE_HASH = content_hash({
    "system": TAXONOMY_REPAIR_SYSTEM,
    "fieldBoundaryPolicy": "natural_semantic_completion",
    "hardCharacterOrWordCeilings": False,
})


def taxonomy_admission_protocol_hash() -> str:
    return content_hash(
        {
            "admissionProtocolVersion": TAXONOMY_ADMISSION_PROTOCOL_VERSION,
            "judgeProtocolVersion": TAXONOMY_JUDGE_PROTOCOL_VERSION,
            "repairProtocolVersion": TAXONOMY_REPAIR_PROTOCOL_VERSION,
            "judgeSourceHash": TAXONOMY_JUDGE_SOURCE_HASH,
            "repairSourceHash": TAXONOMY_REPAIR_SOURCE_HASH,
            "findingCodes": list(TAXONOMY_FINDING_CODES),
            "minimumChangedFields": MIN_REPAIRED_ANCHOR_FIELDS,
        }
    )


def build_taxonomy_judge_view(
    topic: Mapping[str, Any], taxonomy_anchors: Mapping[str, Mapping[str, str]]
) -> dict[str, Any]:
    """Bind every anchor to the host's deterministic Stage-P interaction mode."""

    topic_id = str(topic.get("topicId"))
    scenario_ids = scenario_ids_for_topic(topic_id)
    anchors = validate_taxonomy_anchors(dict(taxonomy_anchors), topic, scenario_ids)
    interaction_modes, safe_stakes = _topic_enums(topic)
    bindings: dict[str, dict[str, str]] = {}
    for ordinal, scenario_id in enumerate(scenario_ids):
        coverage = _coverage_assignment(ordinal, interaction_modes, safe_stakes)
        bindings[scenario_id] = {
            "interactionMode": coverage["interactionMode"],
            **{field: anchors[scenario_id][field] for field in TAXONOMY_FIELDS},
        }
    return {
        "schema": TAXONOMY_JUDGE_VIEW_SCHEMA,
        "topicId": topic_id,
        "requiredLanguage": "English",
        "qualityContract": {
            "completeGrammaticalFields": True,
            "naturalConversationSetup": True,
            "placeholdersAndMetaTextForbidden": True,
            "mixedLanguageAndEncodingCorruptionForbidden": True,
        },
        "topicCardHash": content_hash(topic),
        "taxonomyAnchorsHash": content_hash(anchors),
        "scenarioIds": list(scenario_ids),
        "scenarioBindings": bindings,
    }


def build_taxonomy_judge_response_schema(
    scenario_ids: Sequence[str],
    finding_codes: Sequence[str] = TAXONOMY_FINDING_CODES,
) -> dict[str, Any]:
    ids = tuple(scenario_ids)
    codes = tuple(finding_codes)
    if len(ids) != BLUEPRINTS_PER_TOPIC or len(set(ids)) != len(ids):
        raise ScenarioBlueprintError(
            "taxonomy judgment requires exactly twenty unique scenario IDs"
        )
    if not codes or len(set(codes)) != len(codes) or not set(codes).issubset(
        TAXONOMY_FINDING_CODES
    ):
        raise ScenarioBlueprintError("taxonomy judgment finding-code scope is invalid")
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["findingClusters"],
        "properties": {
            "findingClusters": {
                "type": "array",
                "maxItems": len(codes),
                "uniqueItems": True,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "scenarioIds"],
                    "properties": {
                        "code": {"enum": list(codes)},
                        "scenarioIds": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": BLUEPRINTS_PER_TOPIC,
                            "uniqueItems": True,
                            "items": {"type": "string", "enum": list(ids)},
                        },
                    },
                },
            }
        },
    }
    Draft202012Validator.check_schema(schema)
    return schema


def _normalize_taxonomy_judgment(
    value: Any,
    scenario_ids: Sequence[str],
    *,
    require_actionable_duplicates: bool,
) -> dict[str, Any]:
    """Normalize typed clusters while keeping proposal and action boundaries distinct."""

    ids = tuple(scenario_ids)
    _raise_schema_errors(
        value,
        build_taxonomy_judge_response_schema(ids),
        "taxonomy admission judgment",
    )
    assert_no_target_fields(value)
    assert isinstance(value, dict)
    order = {scenario_id: ordinal for ordinal, scenario_id in enumerate(ids)}
    clusters: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    seen_codes: set[str] = set()
    for finding in value["findingClusters"]:
        finding_ids = tuple(sorted(finding["scenarioIds"], key=order.__getitem__))
        if (
            require_actionable_duplicates
            and
            finding["code"] == "semantic_duplicate_template_collapse"
            and len(finding_ids) < 2
        ):
            raise InvalidModelOutput(
                "semantic duplicate/template collapse must implicate at least two IDs"
            )
        key = (finding["code"], finding_ids)
        if key in seen:
            raise InvalidModelOutput("taxonomy judgment repeats a typed finding cluster")
        if finding["code"] in seen_codes:
            raise InvalidModelOutput(
                "taxonomy judgment must emit at most one cluster per finding code"
            )
        seen.add(key)
        seen_codes.add(finding["code"])
        clusters.append({"code": finding["code"], "scenarioIds": list(finding_ids)})
    clusters.sort(key=lambda finding: canonical_json(finding))
    return {"findingClusters": clusters}


def validate_taxonomy_judgment(
    value: Any, scenario_ids: Sequence[str]
) -> dict[str, Any]:
    """Validate the actionable merged finding set consumed by repair/admission."""

    return _normalize_taxonomy_judgment(
        value, scenario_ids, require_actionable_duplicates=True
    )


def validate_taxonomy_proposal(
    value: Any, scenario_ids: Sequence[str]
) -> dict[str, Any]:
    """Preserve untrusted typed proposals so focused inference can reject them."""

    return _normalize_taxonomy_judgment(
        value, scenario_ids, require_actionable_duplicates=False
    )


def repair_ids_from_taxonomy_judgment(
    judgment: Mapping[str, Any], scenario_ids: Sequence[str]
) -> tuple[str, ...]:
    normalized = validate_taxonomy_judgment(judgment, scenario_ids)
    implicated = {
        scenario_id
        for finding in normalized["findingClusters"]
        for scenario_id in finding["scenarioIds"]
    }
    ordered = tuple(scenario_id for scenario_id in scenario_ids if scenario_id in implicated)
    if not ordered:
        raise ScenarioBlueprintError("taxonomy repair requires typed finding clusters")
    return ordered


def _full_set_taxonomy_judgment(
    judgment: Mapping[str, Any], scenario_ids: Sequence[str]
) -> dict[str, Any]:
    """Normalize every current finding; repair lineage never suppresses new defects."""

    return validate_taxonomy_judgment(judgment, scenario_ids)


def _repair_ids_for_current_judgment(
    judgment: Mapping[str, Any], scenario_ids: Sequence[str]
) -> tuple[str, ...]:
    """Repair all IDs implicated by the current full-set judgment."""

    return repair_ids_from_taxonomy_judgment(judgment, scenario_ids)


class TaxonomyJudge(Protocol):
    def binding(self) -> Mapping[str, Any]:
        ...

    def judge_taxonomy(
        self,
        topic: Mapping[str, Any],
        judge_view: Mapping[str, Any],
        *,
        retry_feedback: Mapping[str, Any] | None = None,
    ) -> Any:
        ...


class AuthenticTaxonomyJudge:
    """Strict-schema model judge that receives no Stage-T planner metadata."""

    def __init__(self, model: StrictSchemaModel) -> None:
        self.model = model

    def binding(self) -> dict[str, Any]:
        return {
            "protocol": "independent_taxonomy_strict_schema_v1",
            "modelBinding": _model_binding(self.model),
            "findingCodes": list(RELATIONAL_TAXONOMY_FINDING_CODES),
            "reasoning": {"enabled": False},
            "responseFormat": "strict_json_schema",
            "maxOutputTokens": TAXONOMY_JUDGE_MAX_OUTPUT_TOKENS,
        }

    def judge_taxonomy(
        self,
        topic: Mapping[str, Any],
        judge_view: Mapping[str, Any],
        *,
        retry_feedback: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Mapping[str, Any]]:
        context: dict[str, Any] = {
            "task": "Independently admit or reject one complete raw Stage-T taxonomy.",
            "topicCard": dict(topic),
            "boundTaxonomyJudgeView": dict(judge_view),
            "typedFindingContract": {
                "soleSemanticRejectionSignal": "findingClusters",
                "codes": list(RELATIONAL_TAXONOMY_FINDING_CODES),
                "definitions": {
                    code: TAXONOMY_FINDING_DEFINITIONS[code]
                    for code in RELATIONAL_TAXONOMY_FINDING_CODES
                },
                "exactImplicatedIdsRequired": True,
                "oneClusterPerCode": True,
            },
        }
        if retry_feedback is not None:
            context["retryFeedback"] = dict(retry_feedback)
        return _call_model(
            self.model,
            name="personaplex_scenario_taxonomy_judge_v5",
            schema=build_taxonomy_judge_response_schema(
                judge_view["scenarioIds"], RELATIONAL_TAXONOMY_FINDING_CODES
            ),
            instructions=TAXONOMY_JUDGE_SYSTEM,
            context=context,
            max_output_tokens=TAXONOMY_JUDGE_MAX_OUTPUT_TOKENS,
        )


class AdjudicatedTaxonomyJudge:
    """Two independent proposers plus source-bound per-claim verification."""

    _VERIFICATION_SCHEMA = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["confirmed"],
        "properties": {"confirmed": {"type": "boolean"}},
    }

    def __init__(
        self,
        primary: TaxonomyJudge,
        secondary: TaxonomyJudge,
        verifier: StrictSchemaModel,
        *,
        quality_model: StrictSchemaModel | None = None,
        quality_witness_model: StrictSchemaModel | None = None,
        checkpoint_root: Path,
        max_workers: int = 3,
        quality_workers: int = 1,
    ) -> None:
        if not 1 <= max_workers <= 3:
            raise ScenarioBlueprintError("taxonomy claim verifier workers must be in [1,3]")
        if not 1 <= quality_workers <= 12:
            raise ScenarioBlueprintError("taxonomy quality workers must be in [1,12]")
        self.primary = primary
        self.secondary = secondary
        self.verifier = verifier
        self.quality_model = quality_model
        self.quality_witness_model = quality_witness_model
        if quality_witness_model is not None and quality_model is None:
            raise ScenarioBlueprintError(
                "taxonomy quality witness requires an atomic quality model"
            )
        self.checkpoint_root = Path(checkpoint_root)
        self.max_workers = max_workers
        self.quality_workers = quality_workers
        self._verifier_slots = BoundedSemaphore(max_workers)
        primary_binding = _taxonomy_judge_binding(primary)
        secondary_binding = _taxonomy_judge_binding(secondary)
        verifier_binding = _model_binding(verifier)
        member_bindings = [
            dict(_taxonomy_judge_model_binding(primary_binding)),
            dict(_taxonomy_judge_model_binding(secondary_binding)),
            verifier_binding,
        ]
        if len({_model_binding_identity(binding) for binding in member_bindings}) != 3:
            raise ScenarioBlueprintError(
                "taxonomy proposers and claim verifier must have distinct model bindings"
            )
        quality_binding = (
            _model_binding(quality_model) if quality_model is not None else None
        )
        quality_witness_binding = (
            _model_binding(quality_witness_model)
            if quality_witness_model is not None
            else None
        )
        self._binding = {
            "protocol": (
                "two_proposer_source_bound_taxonomy_claim_verification_v2_atomic_quality"
                if quality_binding is not None
                else "two_proposer_source_bound_taxonomy_claim_verification_v1"
            ),
            "modelBinding": {
                "protocol": "composite_taxonomy_admission",
                "memberBindingHashes": [content_hash(value) for value in member_bindings],
                "reasoning": {"enabled": False},
                "responseFormat": "strict_json_schema",
            },
            "memberModelBindings": member_bindings,
            "primary": primary_binding,
            "secondary": secondary_binding,
            "verifier": verifier_binding,
            "atomicQuality": {
                "enabled": quality_binding is not None,
                "modelBinding": quality_binding,
                "protocolVersion": TAXONOMY_ATOMIC_QUALITY_PROTOCOL_VERSION,
                "sourceHash": TAXONOMY_ATOMIC_QUALITY_SOURCE_HASH,
                "findingCodes": list(ATOMIC_QUALITY_TAXONOMY_FINDING_CODES),
                "workers": quality_workers,
                "unit": "one_source_bound_scenario_by_one_finding_code",
            },
            "atomicQualityWitness": {
                "enabled": quality_witness_binding is not None,
                "modelBinding": quality_witness_binding,
                "protocolVersion": TAXONOMY_ATOMIC_WITNESS_PROTOCOL_VERSION,
                "sourceHash": TAXONOMY_ATOMIC_WITNESS_SOURCE_HASH,
                "workers": quality_workers,
                "unit": "one_source_bound_positive_claim",
                "decisionPolicy": (
                    "detector_positive_enters_targeted_repair_witnesses_are_audit_only"
                ),
            },
            "relationalScrutiny": {
                "enabled": quality_binding is not None,
                "modelBinding": quality_binding,
                "protocolVersion": TAXONOMY_RELATIONAL_SCRUTINY_PROTOCOL_VERSION,
                "sourceHash": TAXONOMY_RELATIONAL_SCRUTINY_SOURCE_HASH,
                "findingCodes": list(RELATIONAL_TAXONOMY_FINDING_CODES),
                "workers": quality_workers,
                "unit": "one_source_bound_proposed_claim",
                "requiredBeforeIndependentVerification": True,
            },
            "reasoning": {"enabled": False},
            "responseFormat": "strict_json_schema",
            "verificationProtocolVersion": TAXONOMY_CLAIM_VERIFIER_PROTOCOL_VERSION,
            "verificationSourceHash": TAXONOMY_CLAIM_VERIFIER_SOURCE_HASH,
        }

    def binding(self) -> dict[str, Any]:
        return json.loads(canonical_json(self._binding))

    def _verification_path(
        self,
        topic: Mapping[str, Any],
        claim: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> tuple[str, Path]:
        key = content_hash({
            "protocolVersion": TAXONOMY_CLAIM_VERIFIER_PROTOCOL_VERSION,
            "verifierSourceHash": TAXONOMY_CLAIM_VERIFIER_SOURCE_HASH,
            "topicCard": topic,
            "claim": claim,
            "sourceBoundEvidence": evidence,
            "verifierBinding": _model_binding(self.verifier),
            "responseSchema": self._VERIFICATION_SCHEMA,
        })
        return key, self.checkpoint_root / str(topic["topicId"]) / f"{key[7:]}.json"

    def _verify_claim(
        self,
        topic: Mapping[str, Any],
        judge_view: Mapping[str, Any],
        claim: Mapping[str, Any],
    ) -> dict[str, Any]:
        evidence = {
            scenario_id: judge_view["scenarioBindings"][scenario_id]
            for scenario_id in claim["scenarioIds"]
        }
        key, path = self._verification_path(topic, claim, evidence)
        if path.exists():
            checkpoint = read_json(path)
            body = dict(checkpoint)
            checkpoint_hash = body.pop("checkpointHash", None)
            if checkpoint_hash != content_hash(body) or body.get("verificationKey") != key:
                raise ScenarioBlueprintError(
                    f"taxonomy claim-verification checkpoint identity mismatch: {path}"
                )
            return checkpoint
        with self._verifier_slots:
            response, metadata = _call_model(
                self.verifier,
                name="personaplex_scenario_taxonomy_claim_verifier_v5",
                schema=self._VERIFICATION_SCHEMA,
                instructions=TAXONOMY_CLAIM_VERIFIER_SYSTEM,
                context={
                    "task": "Verify exactly one untrusted typed taxonomy finding.",
                    "topicCard": dict(topic),
                    "proposedFinding": dict(claim),
                    "definition": TAXONOMY_FINDING_DEFINITIONS[claim["code"]],
                    "sourceBoundAnchorEvidence": evidence,
                    "rules": [
                        "Return only confirmed for the proposed code and IDs.",
                        "Different wording is not a semantic defect.",
                        "Do not search for or return any unproposed finding.",
                    ],
                },
                max_output_tokens=64,
            )
        _raise_schema_errors(
            response, self._VERIFICATION_SCHEMA, "taxonomy claim verification"
        )
        body = {
            "schema": "personaplex.scenario-taxonomy-claim-verification.v2",
            "verificationKey": key,
            "protocolVersion": TAXONOMY_CLAIM_VERIFIER_PROTOCOL_VERSION,
            "verifierSourceHash": TAXONOMY_CLAIM_VERIFIER_SOURCE_HASH,
            "topicId": topic["topicId"],
            "sourceBoundEvidenceHash": content_hash(evidence),
            "claim": dict(claim),
            "claimHash": content_hash(claim),
            "verifierBinding": _model_binding(self.verifier),
            "modelCall": metadata,
            "confirmed": bool(response["confirmed"]),
        }
        checkpoint = dict(body)
        checkpoint["checkpointHash"] = content_hash(body)
        _write_immutable_json(path, checkpoint)
        return checkpoint

    def _atomic_quality_path(
        self,
        topic: Mapping[str, Any],
        scenario_id: str,
        code: str,
        evidence: Mapping[str, Any],
    ) -> tuple[str, Path]:
        if self.quality_model is None:
            raise ScenarioBlueprintError("atomic taxonomy quality model is not configured")
        key = content_hash({
            "protocolVersion": TAXONOMY_ATOMIC_QUALITY_PROTOCOL_VERSION,
            "qualitySourceHash": TAXONOMY_ATOMIC_QUALITY_SOURCE_HASH,
            "topicCard": topic,
            "scenarioId": scenario_id,
            "findingCode": code,
            "findingDefinition": TAXONOMY_FINDING_DEFINITIONS[code],
            "sourceBoundEvidence": evidence,
            "qualityModelBinding": _model_binding(self.quality_model),
            "responseSchema": self._VERIFICATION_SCHEMA,
        })
        return (
            key,
            self.checkpoint_root
            / "atomic_quality"
            / str(topic["topicId"])
            / f"{key[7:]}.json",
        )

    def _check_atomic_quality(
        self,
        topic: Mapping[str, Any],
        judge_view: Mapping[str, Any],
        task: tuple[str, str],
    ) -> dict[str, Any]:
        if self.quality_model is None:
            raise ScenarioBlueprintError("atomic taxonomy quality model is not configured")
        code, scenario_id = task
        evidence = dict(judge_view["scenarioBindings"][scenario_id])
        key, path = self._atomic_quality_path(topic, scenario_id, code, evidence)
        if path.exists():
            checkpoint = read_json(path)
            body = dict(checkpoint)
            checkpoint_hash = body.pop("checkpointHash", None)
            if checkpoint_hash != content_hash(body) or body.get("qualityKey") != key:
                raise ScenarioBlueprintError(
                    f"taxonomy atomic-quality checkpoint identity mismatch: {path}"
                )
            return checkpoint
        response, metadata = _call_model(
            self.quality_model,
            name="personaplex_scenario_taxonomy_atomic_quality_v5",
            schema=self._VERIFICATION_SCHEMA,
            instructions=TAXONOMY_ATOMIC_QUALITY_SYSTEM,
            context={
                "task": "Verify exactly one taxonomy quality obligation.",
                "topicCard": dict(topic),
                "scenarioId": scenario_id,
                "sourceBoundAnchorEvidence": evidence,
                "proposedFindingCode": code,
                "definition": TAXONOMY_FINDING_DEFINITIONS[code],
                "rules": [
                    "Return only whether this exact code is confirmed for this exact scenario.",
                    "Do not inspect or report another finding code.",
                    "Do not repair or generate content.",
                ],
            },
            max_output_tokens=32,
        )
        _raise_schema_errors(
            response, self._VERIFICATION_SCHEMA, "taxonomy atomic quality verification"
        )
        body = {
            "schema": "personaplex.scenario-taxonomy-atomic-quality.v1",
            "qualityKey": key,
            "protocolVersion": TAXONOMY_ATOMIC_QUALITY_PROTOCOL_VERSION,
            "qualitySourceHash": TAXONOMY_ATOMIC_QUALITY_SOURCE_HASH,
            "topicId": topic["topicId"],
            "scenarioId": scenario_id,
            "findingCode": code,
            "sourceBoundEvidenceHash": content_hash(evidence),
            "qualityModelBinding": _model_binding(self.quality_model),
            "modelCall": metadata,
            "confirmed": bool(response["confirmed"]),
        }
        checkpoint = dict(body)
        checkpoint["checkpointHash"] = content_hash(body)
        _write_immutable_json(path, checkpoint)
        return checkpoint

    def _atomic_quality_checks(
        self,
        topic: Mapping[str, Any],
        judge_view: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if self.quality_model is None:
            return []
        tasks = [
            (code, scenario_id)
            for code in ATOMIC_QUALITY_TAXONOMY_FINDING_CODES
            for scenario_id in judge_view["scenarioIds"]
        ]
        with ThreadPoolExecutor(
            max_workers=min(self.quality_workers, len(tasks)),
            thread_name_prefix="taxonomy-atomic-quality",
        ) as pool:
            return list(pool.map(
                lambda task: self._check_atomic_quality(topic, judge_view, task),
                tasks,
            ))

    def _relational_scrutiny_path(
        self,
        topic: Mapping[str, Any],
        claim: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> tuple[str, Path]:
        if self.quality_model is None:
            raise ScenarioBlueprintError("relational scrutiny model is not configured")
        key = content_hash({
            "protocolVersion": TAXONOMY_RELATIONAL_SCRUTINY_PROTOCOL_VERSION,
            "scrutinySourceHash": TAXONOMY_RELATIONAL_SCRUTINY_SOURCE_HASH,
            "topicCard": topic,
            "claim": claim,
            "findingDefinition": TAXONOMY_FINDING_DEFINITIONS[claim["code"]],
            "sourceBoundEvidence": evidence,
            "scrutinyModelBinding": _model_binding(self.quality_model),
            "responseSchema": self._VERIFICATION_SCHEMA,
        })
        return (
            key,
            self.checkpoint_root
            / "relational_scrutiny"
            / str(topic["topicId"])
            / f"{key[7:]}.json",
        )

    def _scrutinize_relational_claim(
        self,
        topic: Mapping[str, Any],
        judge_view: Mapping[str, Any],
        claim: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.quality_model is None:
            raise ScenarioBlueprintError("relational scrutiny model is not configured")
        evidence = {
            scenario_id: judge_view["scenarioBindings"][scenario_id]
            for scenario_id in claim["scenarioIds"]
        }
        key, path = self._relational_scrutiny_path(topic, claim, evidence)
        if path.exists():
            checkpoint = read_json(path)
            body = dict(checkpoint)
            checkpoint_hash = body.pop("checkpointHash", None)
            if checkpoint_hash != content_hash(body) or body.get("scrutinyKey") != key:
                raise ScenarioBlueprintError(
                    f"taxonomy relational-scrutiny checkpoint identity mismatch: {path}"
                )
            return checkpoint
        response, metadata = _call_model(
            self.quality_model,
            name="personaplex_scenario_taxonomy_relational_scrutiny_v5",
            schema=self._VERIFICATION_SCHEMA,
            instructions=TAXONOMY_RELATIONAL_SCRUTINY_SYSTEM,
            context={
                "task": "Scrutinize exactly one untrusted relational taxonomy claim.",
                "topicCard": dict(topic),
                "proposedFinding": dict(claim),
                "definition": TAXONOMY_FINDING_DEFINITIONS[claim["code"]],
                "sourceBoundAnchorEvidence": evidence,
                "rules": [
                    "Return only whether this exact claim is fully demonstrated.",
                    "Shared interaction mode or domain vocabulary alone is insufficient.",
                    "Do not search for another finding or repair content.",
                ],
            },
            max_output_tokens=32,
        )
        _raise_schema_errors(
            response, self._VERIFICATION_SCHEMA, "taxonomy relational scrutiny"
        )
        body = {
            "schema": "personaplex.scenario-taxonomy-relational-scrutiny.v1",
            "scrutinyKey": key,
            "protocolVersion": TAXONOMY_RELATIONAL_SCRUTINY_PROTOCOL_VERSION,
            "scrutinySourceHash": TAXONOMY_RELATIONAL_SCRUTINY_SOURCE_HASH,
            "topicId": topic["topicId"],
            "claim": dict(claim),
            "claimHash": content_hash(claim),
            "sourceBoundEvidenceHash": content_hash(evidence),
            "scrutinyModelBinding": _model_binding(self.quality_model),
            "modelCall": metadata,
            "confirmed": bool(response["confirmed"]),
        }
        checkpoint = dict(body)
        checkpoint["checkpointHash"] = content_hash(body)
        _write_immutable_json(path, checkpoint)
        return checkpoint

    def _scrutinize_relational_claims(
        self,
        topic: Mapping[str, Any],
        judge_view: Mapping[str, Any],
        claims: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.quality_model is None or not claims:
            return []
        with ThreadPoolExecutor(
            max_workers=min(self.quality_workers, len(claims)),
            thread_name_prefix="taxonomy-relational-scrutiny",
        ) as pool:
            return list(pool.map(
                lambda claim: self._scrutinize_relational_claim(
                    topic, judge_view, claim
                ),
                claims,
            ))

    def _atomic_witness_path(
        self,
        topic: Mapping[str, Any],
        claim: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> tuple[str, Path]:
        if self.quality_witness_model is None:
            raise ScenarioBlueprintError("atomic quality witness is not configured")
        key = content_hash({
            "protocolVersion": TAXONOMY_ATOMIC_WITNESS_PROTOCOL_VERSION,
            "witnessSourceHash": TAXONOMY_ATOMIC_WITNESS_SOURCE_HASH,
            "topicCard": topic,
            "claim": claim,
            "findingDefinition": TAXONOMY_FINDING_DEFINITIONS[claim["code"]],
            "sourceBoundEvidence": evidence,
            "witnessModelBinding": _model_binding(self.quality_witness_model),
            "responseSchema": self._VERIFICATION_SCHEMA,
        })
        return (
            key,
            self.checkpoint_root
            / "atomic_quality_witness"
            / str(topic["topicId"])
            / f"{key[7:]}.json",
        )

    def _verify_atomic_witness(
        self,
        topic: Mapping[str, Any],
        judge_view: Mapping[str, Any],
        claim: Mapping[str, Any],
    ) -> dict[str, Any]:
        if self.quality_witness_model is None:
            raise ScenarioBlueprintError("atomic quality witness is not configured")
        evidence = {
            scenario_id: judge_view["scenarioBindings"][scenario_id]
            for scenario_id in claim["scenarioIds"]
        }
        key, path = self._atomic_witness_path(topic, claim, evidence)
        if path.exists():
            checkpoint = read_json(path)
            body = dict(checkpoint)
            checkpoint_hash = body.pop("checkpointHash", None)
            if checkpoint_hash != content_hash(body) or body.get("witnessKey") != key:
                raise ScenarioBlueprintError(
                    f"taxonomy atomic-witness checkpoint identity mismatch: {path}"
                )
            return checkpoint
        response, metadata = _call_model(
            self.quality_witness_model,
            name="personaplex_scenario_taxonomy_atomic_witness_v5",
            schema=self._VERIFICATION_SCHEMA,
            instructions=TAXONOMY_ATOMIC_WITNESS_SYSTEM,
            context={
                "task": "Independently verify one positive taxonomy quality claim.",
                "topicCard": dict(topic),
                "proposedFinding": dict(claim),
                "definition": TAXONOMY_FINDING_DEFINITIONS[claim["code"]],
                "sourceBoundAnchorEvidence": evidence,
                "rules": [
                    "Return only whether this exact code is present for this exact ID.",
                    "Do not trust or repeat another model's decision.",
                    "Do not audit another code or repair content.",
                ],
            },
            max_output_tokens=32,
        )
        _raise_schema_errors(
            response, self._VERIFICATION_SCHEMA, "taxonomy atomic quality witness"
        )
        body = {
            "schema": "personaplex.scenario-taxonomy-atomic-quality-witness.v1",
            "witnessKey": key,
            "protocolVersion": TAXONOMY_ATOMIC_WITNESS_PROTOCOL_VERSION,
            "witnessSourceHash": TAXONOMY_ATOMIC_WITNESS_SOURCE_HASH,
            "topicId": topic["topicId"],
            "claim": dict(claim),
            "claimHash": content_hash(claim),
            "sourceBoundEvidenceHash": content_hash(evidence),
            "witnessModelBinding": _model_binding(self.quality_witness_model),
            "modelCall": metadata,
            "confirmed": bool(response["confirmed"]),
        }
        checkpoint = dict(body)
        checkpoint["checkpointHash"] = content_hash(body)
        _write_immutable_json(path, checkpoint)
        return checkpoint

    def _atomic_witness_checks(
        self,
        topic: Mapping[str, Any],
        judge_view: Mapping[str, Any],
        claims: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if self.quality_witness_model is None or not claims:
            return []
        with ThreadPoolExecutor(
            max_workers=min(self.quality_workers, len(claims)),
            thread_name_prefix="taxonomy-atomic-witness",
        ) as pool:
            return list(pool.map(
                lambda claim: self._verify_atomic_witness(topic, judge_view, claim),
                claims,
            ))

    def judge_taxonomy(
        self,
        topic: Mapping[str, Any],
        judge_view: Mapping[str, Any],
        *,
        retry_feedback: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="taxonomy-proposer") as pool:
            primary_future = pool.submit(
                _call_taxonomy_judge,
                self.primary,
                topic,
                judge_view,
                retry_feedback,
            )
            secondary_future = pool.submit(
                _call_taxonomy_judge,
                self.secondary,
                topic,
                judge_view,
                retry_feedback,
            )
            primary_raw, primary_metadata = primary_future.result()
            secondary_raw, secondary_metadata = secondary_future.result()
        scenario_ids = judge_view["scenarioIds"]
        primary = validate_taxonomy_proposal(primary_raw, scenario_ids)
        secondary = validate_taxonomy_proposal(secondary_raw, scenario_ids)
        unsupported = [
            claim
            for decision in (primary, secondary)
            for claim in decision["findingClusters"]
            if claim["code"] not in RELATIONAL_TAXONOMY_FINDING_CODES
        ]
        if unsupported:
            raise InvalidModelOutput(
                "set-level taxonomy proposers emitted atomic quality findings"
            )
        quality_checkpoints = self._atomic_quality_checks(topic, judge_view)
        atomic_claims = [
            {
                "code": checkpoint["findingCode"],
                "scenarioIds": [checkpoint["scenarioId"]],
            }
            for checkpoint in quality_checkpoints
            if checkpoint["confirmed"]
        ]
        atomic_witness_checkpoints = self._atomic_witness_checks(
            topic, judge_view, atomic_claims
        )
        relational_claims = {
            canonical_json(claim): claim
            for claim in primary["findingClusters"] + secondary["findingClusters"]
        }
        ordered_relational_claims = [
            relational_claims[key] for key in sorted(relational_claims)
        ]
        scrutiny_checkpoints = self._scrutinize_relational_claims(
            topic, judge_view, ordered_relational_claims
        )
        if self.quality_model is None:
            scrutinized_relational_claims = ordered_relational_claims
        else:
            scrutinized_relational_claims = [
                dict(checkpoint["claim"])
                for checkpoint in scrutiny_checkpoints
                if checkpoint["confirmed"]
            ]
        claims = {
            canonical_json(claim): claim
            for claim in scrutinized_relational_claims + atomic_claims
        }
        ordered_claims = [claims[key] for key in sorted(claims)]
        checkpoints: list[dict[str, Any]] = []
        if ordered_claims:
            with ThreadPoolExecutor(
                max_workers=min(self.max_workers, len(ordered_claims)),
                thread_name_prefix="taxonomy-claim-verifier",
            ) as pool:
                checkpoints = list(pool.map(
                    lambda claim: self._verify_claim(topic, judge_view, claim),
                    ordered_claims,
                ))
        independently_confirmed_keys = {
            canonical_json(checkpoint["claim"])
            for checkpoint in checkpoints
            if checkpoint["confirmed"]
        }
        witness_confirmed_keys = {
            canonical_json(checkpoint["claim"])
            for checkpoint in atomic_witness_checkpoints
            if checkpoint["confirmed"]
        }
        actionable_claims = {
            canonical_json(claim): claim
            for claim in scrutinized_relational_claims
            if canonical_json(claim) in independently_confirmed_keys
        }
        actionable_claims.update({
            canonical_json(claim): claim
            for claim in atomic_claims
        })
        confirmed_claims = list(actionable_claims.values())
        confirmed_ids_by_code: dict[str, set[str]] = {}
        for claim in confirmed_claims:
            confirmed_ids_by_code.setdefault(claim["code"], set()).update(
                claim["scenarioIds"]
            )
        scenario_order = {
            scenario_id: ordinal for ordinal, scenario_id in enumerate(scenario_ids)
        }
        confirmed = [
            {
                "code": code,
                "scenarioIds": sorted(ids, key=scenario_order.__getitem__),
            }
            for code, ids in confirmed_ids_by_code.items()
        ]
        confirmed.sort(key=canonical_json)
        return {"findingClusters": confirmed}, {
            "protocol": self._binding["protocol"],
            "primaryModelCall": primary_metadata,
            "secondaryModelCall": secondary_metadata,
            "primaryFindingClusters": primary["findingClusters"],
            "secondaryFindingClusters": secondary["findingClusters"],
            "atomicQualityCheckCount": len(quality_checkpoints),
            "atomicQualityConfirmedCount": len(atomic_claims),
            "atomicQualityCheckpointHashes": [
                checkpoint["checkpointHash"] for checkpoint in quality_checkpoints
            ],
            "atomicWitnessCheckCount": len(atomic_witness_checkpoints),
            "atomicWitnessConfirmedCount": len(witness_confirmed_keys),
            "atomicWitnessCheckpointHashes": [
                checkpoint["checkpointHash"]
                for checkpoint in atomic_witness_checkpoints
            ],
            "relationalProposedClaimCount": len(ordered_relational_claims),
            "relationalScrutinyConfirmedCount": len(
                scrutinized_relational_claims
            ),
            "relationalScrutinyCheckpointHashes": [
                checkpoint["checkpointHash"] for checkpoint in scrutiny_checkpoints
            ],
            "proposedClaimCount": len(ordered_claims),
            "independentVerifierConfirmedCount": len(
                independently_confirmed_keys
            ),
            "confirmedClaimCount": len(confirmed),
            "verificationCheckpointHashes": [
                checkpoint["checkpointHash"] for checkpoint in checkpoints
            ],
        }


def _taxonomy_judge_binding(judge: TaxonomyJudge) -> dict[str, Any]:
    value = judge.binding()
    if not isinstance(value, Mapping):
        raise ScenarioBlueprintError("taxonomy judge binding must be an object")
    return json.loads(canonical_json(dict(value)))


def _taxonomy_judge_model_binding(
    judge_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    nested = judge_binding.get("modelBinding")
    return nested if isinstance(nested, Mapping) else judge_binding


def _model_binding_identity(binding: Mapping[str, Any]) -> str:
    protocol = binding.get("protocol")
    model = binding.get("model")
    if isinstance(protocol, str) and protocol and isinstance(model, str) and model:
        return canonical_json({"protocol": protocol, "model": model})
    return content_hash(binding)


def _ensure_independent_bindings(
    planner_binding: Mapping[str, Any], judge_binding: Mapping[str, Any]
) -> None:
    raw_members = judge_binding.get("memberModelBindings")
    judge_models = (
        list(raw_members)
        if isinstance(raw_members, list)
        else [_taxonomy_judge_model_binding(judge_binding)]
    )
    planner_identity = _model_binding_identity(planner_binding)
    judge_identities = [
        _model_binding_identity(binding)
        for binding in judge_models
        if isinstance(binding, Mapping)
    ]
    if len(judge_identities) == 1 and judge_identities[0] == planner_identity:
        raise ScenarioBlueprintError(
            "taxonomy judge must be independently bound from the Stage-T/repair planner"
        )
    if len(judge_identities) >= 3 and (
        judge_identities[-1] == planner_identity
        or all(identity == planner_identity for identity in judge_identities[:-1])
    ):
        raise ScenarioBlueprintError(
            "taxonomy adjudication requires an independent proposer and final verifier"
        )


def _call_taxonomy_judge(
    judge: TaxonomyJudge,
    topic: Mapping[str, Any],
    judge_view: Mapping[str, Any],
    retry_feedback: Mapping[str, Any] | None,
) -> tuple[Any, dict[str, Any]]:
    result = judge.judge_taxonomy(
        topic, judge_view, retry_feedback=retry_feedback
    )
    if isinstance(result, tuple) and len(result) == 2:
        judgment, metadata = result
    else:
        judgment, metadata = result, {}
    if not isinstance(metadata, Mapping):
        raise InvalidModelOutput("taxonomy judge metadata must be an object")
    return judgment, json.loads(canonical_json(dict(metadata)))


def _judgment_stage_key(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    source_type: str,
    judgment_cycle: int,
    judge_view: Mapping[str, Any],
    response_schema: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
    judge_binding: Mapping[str, Any],
    raw_checkpoint_hash: str,
) -> str:
    return content_hash(
        {
            "stage": "scenario_taxonomy_judgment_v5",
            "protocolVersion": TAXONOMY_JUDGE_PROTOCOL_VERSION,
            "judgeSourceHash": TAXONOMY_JUDGE_SOURCE_HASH,
            "requestHash": content_hash(request),
            "topicCardHash": content_hash(topic),
            "rawTaxonomyCheckpointHash": raw_checkpoint_hash,
            "sourceTaxonomyCheckpointHash": source_checkpoint["checkpointHash"],
            "sourceTaxonomyAnchorsHash": source_checkpoint["taxonomyAnchorsHash"],
            "sourceType": source_type,
            "judgmentCycle": judgment_cycle,
            "judgeViewHash": content_hash(judge_view),
            "responseSchemaHash": content_hash(response_schema),
            "plannerBindingHash": content_hash(planner_binding),
            "judgeBindingHash": content_hash(judge_binding),
        }
    )


def _validate_judgment_checkpoint(
    checkpoint: Any,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    source_type: str,
    judgment_cycle: int,
    planner_binding: Mapping[str, Any],
    judge_binding: Mapping[str, Any],
    raw_checkpoint_hash: str,
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ScenarioBlueprintError("taxonomy judgment checkpoint must be an object")
    required = {
        "schema",
        "stageKey",
        "protocolVersion",
        "judgeSourceHash",
        "judgmentCycle",
        "requestHash",
        "topicId",
        "topicCardHash",
        "rawTaxonomyCheckpointHash",
        "sourceTaxonomyCheckpointHash",
        "sourceTaxonomyAnchorsHash",
        "sourceType",
        "taxonomyPlannerBinding",
        "taxonomyPlannerBindingHash",
        "taxonomyJudgeBinding",
        "taxonomyJudgeBindingHash",
        "taxonomyJudgeModelBindingHash",
        "judgeView",
        "judgeViewHash",
        "responseSchemaHash",
        "modelCall",
        "findingClusters",
        "findingClustersHash",
        "checkpointHash",
    }
    if set(checkpoint) != required:
        raise ScenarioBlueprintError("taxonomy judgment checkpoint has an invalid field set")
    judge_view = build_taxonomy_judge_view(topic, source_checkpoint["taxonomyAnchors"])
    response_schema = build_taxonomy_judge_response_schema(judge_view["scenarioIds"])
    expected = {
        "schema": TAXONOMY_JUDGMENT_CHECKPOINT_SCHEMA,
        "stageKey": _judgment_stage_key(
            request,
            topic,
            source_checkpoint,
            source_type,
            judgment_cycle,
            judge_view,
            response_schema,
            planner_binding,
            judge_binding,
            raw_checkpoint_hash,
        ),
        "protocolVersion": TAXONOMY_JUDGE_PROTOCOL_VERSION,
        "judgeSourceHash": TAXONOMY_JUDGE_SOURCE_HASH,
        "judgmentCycle": judgment_cycle,
        "requestHash": content_hash(request),
        "topicId": topic["topicId"],
        "topicCardHash": content_hash(topic),
        "rawTaxonomyCheckpointHash": raw_checkpoint_hash,
        "sourceTaxonomyCheckpointHash": source_checkpoint["checkpointHash"],
        "sourceTaxonomyAnchorsHash": source_checkpoint["taxonomyAnchorsHash"],
        "sourceType": source_type,
        "taxonomyPlannerBinding": dict(planner_binding),
        "taxonomyPlannerBindingHash": content_hash(planner_binding),
        "taxonomyJudgeBinding": dict(judge_binding),
        "taxonomyJudgeBindingHash": content_hash(judge_binding),
        "taxonomyJudgeModelBindingHash": content_hash(
            _taxonomy_judge_model_binding(judge_binding)
        ),
        "judgeView": judge_view,
        "judgeViewHash": content_hash(judge_view),
        "responseSchemaHash": content_hash(response_schema),
    }
    for field, expected_value in expected.items():
        if checkpoint.get(field) != expected_value:
            raise ScenarioBlueprintError(
                f"taxonomy judgment checkpoint binding mismatch: {field}"
            )
    normalized = _full_set_taxonomy_judgment(
        {"findingClusters": checkpoint.get("findingClusters")},
        judge_view["scenarioIds"],
    )["findingClusters"]
    if checkpoint.get("findingClusters") != normalized:
        raise ScenarioBlueprintError("taxonomy finding clusters are not normalized")
    if checkpoint.get("findingClustersHash") != content_hash(normalized):
        raise ScenarioBlueprintError("taxonomy finding-cluster hash mismatch")
    if not isinstance(checkpoint.get("modelCall"), dict):
        raise ScenarioBlueprintError("taxonomy judgment modelCall must be an object")
    _checkpoint_body_hash(checkpoint)
    return checkpoint


def generate_taxonomy_judgment(
    *,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    source_type: str,
    judgment_cycle: int,
    raw_checkpoint_hash: str,
    output_root: Path,
    planner_binding: Mapping[str, Any],
    judge: TaxonomyJudge,
    max_attempts: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    if source_type not in {"raw", "repair"}:
        raise ScenarioBlueprintError("taxonomy judgment source must be raw or repair")
    if not 0 <= judgment_cycle <= 12 or not 1 <= max_attempts <= 12:
        raise ScenarioBlueprintError("judgment_cycle and max_attempts must be bounded by 12")
    validate_taxonomy_anchors(source_checkpoint.get("taxonomyAnchors"), topic)
    judge_binding = _taxonomy_judge_binding(judge)
    _ensure_independent_bindings(planner_binding, judge_binding)
    judge_view = build_taxonomy_judge_view(topic, source_checkpoint["taxonomyAnchors"])
    response_schema = build_taxonomy_judge_response_schema(judge_view["scenarioIds"])
    stage_key = _judgment_stage_key(
        request,
        topic,
        source_checkpoint,
        source_type,
        judgment_cycle,
        judge_view,
        response_schema,
        planner_binding,
        judge_binding,
        raw_checkpoint_hash,
    )
    path = _checkpoint_path(
        Path(output_root),
        "taxonomy_judgments",
        f"{topic['topicId']}_c{judgment_cycle:02d}",
        stage_key,
    )
    if path.exists():
        if not resume:
            raise ScenarioBlueprintError(
                f"taxonomy judgment exists; use --resume: {path}"
            )
        return _validate_judgment_checkpoint(
            read_json(path),
            request,
            topic,
            source_checkpoint,
            source_type,
            judgment_cycle,
            planner_binding,
            judge_binding,
            raw_checkpoint_hash,
        )

    failures: list[str] = []
    for attempt in range(1, max_attempts + 1):
        retry_feedback = None
        if failures:
            retry_feedback = {
                "attempt": attempt,
                "previousProtocolDefect": failures[-1],
                "directive": "Rejudge the same complete bound view using only typed clusters.",
            }
        try:
            response, metadata = _call_taxonomy_judge(
                judge, topic, judge_view, retry_feedback
            )
            clusters = _full_set_taxonomy_judgment(
                response,
                judge_view["scenarioIds"],
            )["findingClusters"]
            body = {
                "schema": TAXONOMY_JUDGMENT_CHECKPOINT_SCHEMA,
                "stageKey": stage_key,
                "protocolVersion": TAXONOMY_JUDGE_PROTOCOL_VERSION,
                "judgeSourceHash": TAXONOMY_JUDGE_SOURCE_HASH,
                "judgmentCycle": judgment_cycle,
                "requestHash": content_hash(request),
                "topicId": topic["topicId"],
                "topicCardHash": content_hash(topic),
                "rawTaxonomyCheckpointHash": raw_checkpoint_hash,
                "sourceTaxonomyCheckpointHash": source_checkpoint["checkpointHash"],
                "sourceTaxonomyAnchorsHash": source_checkpoint["taxonomyAnchorsHash"],
                "sourceType": source_type,
                "taxonomyPlannerBinding": dict(planner_binding),
                "taxonomyPlannerBindingHash": content_hash(planner_binding),
                "taxonomyJudgeBinding": judge_binding,
                "taxonomyJudgeBindingHash": content_hash(judge_binding),
                "taxonomyJudgeModelBindingHash": content_hash(
                    _taxonomy_judge_model_binding(judge_binding)
                ),
                "judgeView": judge_view,
                "judgeViewHash": content_hash(judge_view),
                "responseSchemaHash": content_hash(response_schema),
                "modelCall": metadata,
                "findingClusters": clusters,
                "findingClustersHash": content_hash(clusters),
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
        f"taxonomy judgment exhausted {max_attempts} attempts for {topic['topicId']}: "
        + " | ".join(failures)
    )


def build_taxonomy_repair_response_schema(
    topic: Mapping[str, Any],
    repair_ids: Sequence[str],
    parent_anchors: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    scenario_ids = scenario_ids_for_topic(str(topic.get("topicId")))
    ordered_repair_ids = tuple(repair_ids)
    if (
        not ordered_repair_ids
        or len(set(ordered_repair_ids)) != len(ordered_repair_ids)
        or any(scenario_id not in scenario_ids for scenario_id in ordered_repair_ids)
    ):
        raise ScenarioBlueprintError(
            "taxonomy repair IDs must be a nonempty unique scenario-ID subset"
        )
    full_schema = build_taxonomy_response_schema(topic, scenario_ids)
    parent = validate_taxonomy_anchors(dict(parent_anchors), topic, scenario_ids)
    repair_id_set = set(ordered_repair_ids)
    immutable_submodes = [
        parent[scenario_id]["submode"]
        for scenario_id in scenario_ids
        if scenario_id not in repair_id_set
    ]
    repair_properties: dict[str, Any] = {}
    for scenario_id in ordered_repair_ids:
        compact_schema = full_schema["properties"][scenario_id]
        anchor_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [*TAXONOMY_FIELDS, "changedFields"],
            "properties": {},
        }
        for field in TAXONOMY_FIELDS:
            wire_key = TAXONOMY_WIRE_KEYS[field]
            forbidden_values = [parent[scenario_id][field]]
            if field == "submode":
                forbidden_values.extend(immutable_submodes)
            field_schema = json.loads(
                canonical_json(compact_schema["properties"][wire_key])
            )
            field_schema["not"] = {
                "enum": list(dict.fromkeys(forbidden_values))
            }
            anchor_schema["properties"][field] = field_schema
        anchor_schema["properties"]["changedFields"] = {
            "type": "array",
            "minItems": len(TAXONOMY_FIELDS),
            "maxItems": len(TAXONOMY_FIELDS),
            "uniqueItems": True,
            "items": {"enum": list(TAXONOMY_FIELDS)},
        }
        repair_properties[scenario_id] = anchor_schema
    schema = {
        "$schema": full_schema["$schema"],
        "type": "object",
        "additionalProperties": False,
        "required": list(ordered_repair_ids),
        "properties": repair_properties,
    }
    Draft202012Validator.check_schema(schema)
    if "prefixItems" in canonical_json(schema):
        raise ScenarioBlueprintError("taxonomy repair schema must never use prefixItems")
    return schema


def _repair_delta(
    before: Mapping[str, Mapping[str, str]],
    after: Mapping[str, Mapping[str, str]],
    scenario_ids: Sequence[str],
    repair_ids: Sequence[str],
) -> dict[str, list[str]]:
    repair_id_set = set(repair_ids)
    changed: dict[str, list[str]] = {}
    for scenario_id in scenario_ids:
        before_bytes = canonical_json(before[scenario_id]).encode("utf-8")
        after_bytes = canonical_json(after[scenario_id]).encode("utf-8")
        if scenario_id not in repair_id_set:
            if after_bytes != before_bytes:
                raise InvalidModelOutput(
                    f"taxonomy repair altered accepted anchor bytes: {scenario_id}"
                )
            continue
        fields = [
            field
            for field in TAXONOMY_FIELDS
            if before[scenario_id][field] != after[scenario_id][field]
        ]
        if len(fields) < MIN_REPAIRED_ANCHOR_FIELDS:
            raise InvalidModelOutput(
                f"taxonomy repair for {scenario_id} must structurally change at least "
                f"{MIN_REPAIRED_ANCHOR_FIELDS} fields"
            )
        changed[scenario_id] = fields
    return changed


def merge_taxonomy_repair_response(
    response: Any,
    topic: Mapping[str, Any],
    taxonomy_anchors: Mapping[str, Mapping[str, str]],
    repair_ids: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Merge only schema-required IDs, then validate the complete set anew."""

    scenario_ids = scenario_ids_for_topic(str(topic.get("topicId")))
    before = validate_taxonomy_anchors(dict(taxonomy_anchors), topic, scenario_ids)
    ordered_repair_ids = tuple(repair_ids)
    response_schema = build_taxonomy_repair_response_schema(
        topic, ordered_repair_ids, before
    )
    _raise_schema_errors(response, response_schema, "taxonomy repair response")
    assert_no_target_fields(response)
    assert isinstance(response, dict)
    repair_id_set = set(ordered_repair_ids)
    merged: dict[str, dict[str, str]] = {}
    for scenario_id in scenario_ids:
        if scenario_id not in repair_id_set:
            merged[scenario_id] = before[scenario_id]
            continue
        merged[scenario_id] = {
            field: response[scenario_id][field]
            for field in TAXONOMY_FIELDS
        }
    validated = validate_taxonomy_anchors(merged, topic, scenario_ids)
    changed = _repair_delta(before, validated, scenario_ids, ordered_repair_ids)
    for scenario_id in ordered_repair_ids:
        declared = response[scenario_id]["changedFields"]
        if set(declared) != set(changed[scenario_id]):
            raise InvalidModelOutput(
                f"taxonomy repair changedFields declaration mismatch for {scenario_id}"
            )
    return validated


def _repair_stage_key(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    source_judgment: Mapping[str, Any],
    repair_ids: Sequence[str],
    response_schema: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
    raw_checkpoint_hash: str,
    repair_cycle: int,
) -> str:
    return content_hash(
        {
            "stage": "scenario_taxonomy_repair_v5",
            "protocolVersion": TAXONOMY_REPAIR_PROTOCOL_VERSION,
            "repairSourceHash": TAXONOMY_REPAIR_SOURCE_HASH,
            "requestHash": content_hash(request),
            "topicCardHash": content_hash(topic),
            "rawTaxonomyCheckpointHash": raw_checkpoint_hash,
            "parentTaxonomyCheckpointHash": source_checkpoint["checkpointHash"],
            "parentTaxonomyAnchorsHash": source_checkpoint["taxonomyAnchorsHash"],
            "sourceJudgmentCheckpointHash": source_judgment["checkpointHash"],
            "repairCycle": repair_cycle,
            "repairScenarioIds": list(repair_ids),
            "responseSchemaHash": content_hash(response_schema),
            "plannerBindingHash": content_hash(planner_binding),
        }
    )


def _validate_repair_checkpoint(
    checkpoint: Any,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    source_judgment: Mapping[str, Any],
    repair_ids: Sequence[str],
    response_schema: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
    raw_checkpoint_hash: str,
    repair_cycle: int,
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ScenarioBlueprintError("taxonomy repair checkpoint must be an object")
    required = {
        "schema",
        "stageKey",
        "protocolVersion",
        "repairSourceHash",
        "repairCycle",
        "requestHash",
        "topicId",
        "topicCardHash",
        "rawTaxonomyCheckpointHash",
        "parentTaxonomyCheckpointHash",
        "parentTaxonomyAnchorsHash",
        "sourceJudgmentCheckpointHash",
        "repairScenarioIds",
        "acceptedScenarioIds",
        "acceptedAnchorJsonHashes",
        "responseSchemaHash",
        "repairPlannerBinding",
        "repairPlannerBindingHash",
        "modelCall",
        "changedFieldsByScenarioId",
        "taxonomyAnchors",
        "taxonomyAnchorsHash",
        "checkpointHash",
    }
    if set(checkpoint) != required:
        raise ScenarioBlueprintError("taxonomy repair checkpoint has an invalid field set")
    scenario_ids = scenario_ids_for_topic(str(topic.get("topicId")))
    repair_id_set = set(repair_ids)
    accepted_ids = [
        scenario_id for scenario_id in scenario_ids if scenario_id not in repair_id_set
    ]
    before = validate_taxonomy_anchors(source_checkpoint["taxonomyAnchors"], topic)
    repaired = validate_taxonomy_anchors(checkpoint.get("taxonomyAnchors"), topic)
    changed = _repair_delta(before, repaired, scenario_ids, repair_ids)
    expected = {
        "schema": TAXONOMY_REPAIR_CHECKPOINT_SCHEMA,
        "stageKey": _repair_stage_key(
            request,
            topic,
            source_checkpoint,
            source_judgment,
            repair_ids,
            response_schema,
            planner_binding,
            raw_checkpoint_hash,
            repair_cycle,
        ),
        "protocolVersion": TAXONOMY_REPAIR_PROTOCOL_VERSION,
        "repairSourceHash": TAXONOMY_REPAIR_SOURCE_HASH,
        "repairCycle": repair_cycle,
        "requestHash": content_hash(request),
        "topicId": topic["topicId"],
        "topicCardHash": content_hash(topic),
        "rawTaxonomyCheckpointHash": raw_checkpoint_hash,
        "parentTaxonomyCheckpointHash": source_checkpoint["checkpointHash"],
        "parentTaxonomyAnchorsHash": source_checkpoint["taxonomyAnchorsHash"],
        "sourceJudgmentCheckpointHash": source_judgment["checkpointHash"],
        "repairScenarioIds": list(repair_ids),
        "acceptedScenarioIds": accepted_ids,
        "acceptedAnchorJsonHashes": {
            scenario_id: content_hash(before[scenario_id]) for scenario_id in accepted_ids
        },
        "responseSchemaHash": content_hash(response_schema),
        "repairPlannerBinding": dict(planner_binding),
        "repairPlannerBindingHash": content_hash(planner_binding),
        "changedFieldsByScenarioId": changed,
        "taxonomyAnchorsHash": content_hash(repaired),
    }
    for field, expected_value in expected.items():
        if checkpoint.get(field) != expected_value:
            raise ScenarioBlueprintError(
                f"taxonomy repair checkpoint binding mismatch: {field}"
            )
    if not isinstance(checkpoint.get("modelCall"), dict):
        raise ScenarioBlueprintError("taxonomy repair modelCall must be an object")
    _checkpoint_body_hash(checkpoint)
    return checkpoint


def repair_topic_taxonomy(
    *,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    source_judgment: Mapping[str, Any],
    raw_checkpoint_hash: str,
    output_root: Path,
    planner: StrictSchemaModel,
    repair_cycle: int,
    max_attempts: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    if not 1 <= repair_cycle <= 12 or not 1 <= max_attempts <= 12:
        raise ScenarioBlueprintError("repair_cycle and max_attempts must be in [1,12]")
    scenario_ids = scenario_ids_for_topic(str(topic.get("topicId")))
    before = validate_taxonomy_anchors(source_checkpoint.get("taxonomyAnchors"), topic)
    normalized_judgment = validate_taxonomy_judgment(
        {"findingClusters": source_judgment.get("findingClusters")}, scenario_ids
    )
    repair_ids = _repair_ids_for_current_judgment(
        normalized_judgment, scenario_ids
    )
    response_schema = build_taxonomy_repair_response_schema(topic, repair_ids, before)
    planner_binding = _model_binding(planner)
    stage_key = _repair_stage_key(
        request,
        topic,
        source_checkpoint,
        source_judgment,
        repair_ids,
        response_schema,
        planner_binding,
        raw_checkpoint_hash,
        repair_cycle,
    )
    path = _checkpoint_path(
        Path(output_root),
        "taxonomy_repairs",
        f"{topic['topicId']}_r{repair_cycle:02d}",
        stage_key,
    )
    if path.exists():
        if not resume:
            raise ScenarioBlueprintError(f"taxonomy repair exists; use --resume: {path}")
        return _validate_repair_checkpoint(
            read_json(path),
            request,
            topic,
            source_checkpoint,
            source_judgment,
            repair_ids,
            response_schema,
            planner_binding,
            raw_checkpoint_hash,
            repair_cycle,
        )

    repair_id_set = set(repair_ids)
    judge_view = build_taxonomy_judge_view(topic, before)
    context = {
        "task": "Repair exactly the IDs implicated by typed Stage-T finding clusters.",
        "topicCard": dict(topic),
        "parentTaxonomyJudgeView": judge_view,
        "typedFindingClusters": normalized_judgment["findingClusters"],
        "repairScenarioIds": list(repair_ids),
        "immutableScenarioIds": [
            scenario_id for scenario_id in scenario_ids if scenario_id not in repair_id_set
        ],
        "repairFieldNames": list(TAXONOMY_FIELDS),
        "forbiddenParentValuesByScenarioId": {
            scenario_id: before[scenario_id] for scenario_id in repair_ids
        },
        "forbiddenImmutableSubmodes": [
            before[scenario_id]["submode"]
            for scenario_id in scenario_ids
            if scenario_id not in repair_id_set
        ],
        "structuralRepairContract": {
            "minimumChangedFieldsPerRepairedAnchor": MIN_REPAIRED_ANCHOR_FIELDS,
            "requiredChangedFieldsPerRepairedAnchor": list(TAXONOMY_FIELDS),
            "everyTaxonomyFieldMustDifferFromParent": True,
            "submodeMustDifferFromEveryImmutableAnchor": True,
            "changedFieldsProperty": "changedFields",
            "changedFieldsCanonicalNames": list(TAXONOMY_FIELDS),
            "declaredFieldsMustExactlyMatchActualDelta": True,
            "fullSetValidationAndUniquenessRerun": True,
            "acceptedAnchorJsonMustRemainByteIdentical": True,
            "hardCharacterOrWordCeilings": False,
            "naturalSemanticBoundaryRequired": True,
            "terminalPunctuationWhereNatural": True,
            "completeThoughtRequiredBeforeClosingEachString": True,
        },
    }
    failures: list[str] = []
    for attempt in range(1, max_attempts + 1):
        attempt_context = dict(context)
        if failures:
            attempt_context["retryFeedback"] = {
                "attempt": attempt,
                "previousStructuralDefect": failures[-1],
                "directive": (
                    "Regenerate all five canonical fields for every same implicated ID. "
                    "Do not copy any listed forbidden parent value or immutable submode."
                ),
            }
        try:
            response, metadata = _call_model(
                planner,
                name="personaplex_scenario_taxonomy_repair_v5",
                schema=response_schema,
                instructions=TAXONOMY_REPAIR_SYSTEM,
                context=attempt_context,
                max_output_tokens=TAXONOMY_MAX_OUTPUT_TOKENS,
            )
            repaired = merge_taxonomy_repair_response(
                response, topic, before, repair_ids
            )
            changed = _repair_delta(before, repaired, scenario_ids, repair_ids)
            accepted_ids = [
                scenario_id
                for scenario_id in scenario_ids
                if scenario_id not in repair_id_set
            ]
            body = {
                "schema": TAXONOMY_REPAIR_CHECKPOINT_SCHEMA,
                "stageKey": stage_key,
                "protocolVersion": TAXONOMY_REPAIR_PROTOCOL_VERSION,
                "repairSourceHash": TAXONOMY_REPAIR_SOURCE_HASH,
                "repairCycle": repair_cycle,
                "requestHash": content_hash(request),
                "topicId": topic["topicId"],
                "topicCardHash": content_hash(topic),
                "rawTaxonomyCheckpointHash": raw_checkpoint_hash,
                "parentTaxonomyCheckpointHash": source_checkpoint["checkpointHash"],
                "parentTaxonomyAnchorsHash": source_checkpoint["taxonomyAnchorsHash"],
                "sourceJudgmentCheckpointHash": source_judgment["checkpointHash"],
                "repairScenarioIds": list(repair_ids),
                "acceptedScenarioIds": accepted_ids,
                "acceptedAnchorJsonHashes": {
                    scenario_id: content_hash(before[scenario_id])
                    for scenario_id in accepted_ids
                },
                "responseSchemaHash": content_hash(response_schema),
                "repairPlannerBinding": planner_binding,
                "repairPlannerBindingHash": content_hash(planner_binding),
                "modelCall": metadata,
                "changedFieldsByScenarioId": changed,
                "taxonomyAnchors": repaired,
                "taxonomyAnchorsHash": content_hash(repaired),
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
        f"taxonomy repair exhausted {max_attempts} attempts for {topic['topicId']} "
        f"cycle {repair_cycle}: " + " | ".join(failures)
    )


def _admission_stage_key(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    raw_checkpoint_hash: str,
    source_checkpoint: Mapping[str, Any],
    final_judgment: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
    judge_binding: Mapping[str, Any],
    judgment_hashes: Sequence[str],
    repair_hashes: Sequence[str],
) -> str:
    return content_hash(
        {
            "stage": "scenario_taxonomy_admission_v5",
            "protocolVersion": TAXONOMY_ADMISSION_PROTOCOL_VERSION,
            "protocolHash": taxonomy_admission_protocol_hash(),
            "requestHash": content_hash(request),
            "topicCardHash": content_hash(topic),
            "rawTaxonomyCheckpointHash": raw_checkpoint_hash,
            "admittedSourceCheckpointHash": source_checkpoint["checkpointHash"],
            "admittedTaxonomyAnchorsHash": source_checkpoint["taxonomyAnchorsHash"],
            "finalJudgmentCheckpointHash": final_judgment["checkpointHash"],
            "plannerBindingHash": content_hash(planner_binding),
            "judgeBindingHash": content_hash(judge_binding),
            "judgeModelBindingHash": content_hash(
                _taxonomy_judge_model_binding(judge_binding)
            ),
            "judgeSourceHash": TAXONOMY_JUDGE_SOURCE_HASH,
            "repairSourceHash": TAXONOMY_REPAIR_SOURCE_HASH,
            "judgmentCheckpointHashes": list(judgment_hashes),
            "repairCheckpointHashes": list(repair_hashes),
        }
    )


def validate_taxonomy_admission_checkpoint(
    checkpoint: Any,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    planner_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the self-contained admission proof consumed by Stage P."""

    if not isinstance(checkpoint, dict):
        raise ScenarioBlueprintError("taxonomy admission checkpoint must be an object")
    required = {
        "schema",
        "stageKey",
        "protocolVersion",
        "protocolHash",
        "requestHash",
        "topicId",
        "topicCardHash",
        "rawTaxonomyCheckpointHash",
        "admittedSourceCheckpointHash",
        "admittedSourceType",
        "taxonomyAnchors",
        "taxonomyAnchorsHash",
        "taxonomyPlannerBinding",
        "taxonomyPlannerBindingHash",
        "taxonomyJudgeBinding",
        "taxonomyJudgeBindingHash",
        "taxonomyJudgeModelBindingHash",
        "taxonomyJudgeSourceHash",
        "taxonomyRepairSourceHash",
        "finalJudgmentCheckpointHash",
        "finalFindingClusters",
        "judgmentCheckpointHashes",
        "repairCheckpointHashes",
        "lineageHash",
        "checkpointHash",
    }
    if set(checkpoint) != required:
        raise ScenarioBlueprintError("taxonomy admission checkpoint has an invalid field set")
    anchors = validate_taxonomy_anchors(checkpoint.get("taxonomyAnchors"), topic)
    bound_planner = checkpoint.get("taxonomyPlannerBinding")
    bound_judge = checkpoint.get("taxonomyJudgeBinding")
    if not isinstance(bound_planner, Mapping) or not isinstance(bound_judge, Mapping):
        raise ScenarioBlueprintError("taxonomy admission model bindings must be objects")
    if planner_binding is not None and dict(bound_planner) != dict(planner_binding):
        raise ScenarioBlueprintError("taxonomy admission planner binding mismatch")
    _ensure_independent_bindings(bound_planner, bound_judge)
    judgment_hashes = checkpoint.get("judgmentCheckpointHashes")
    repair_hashes = checkpoint.get("repairCheckpointHashes")
    if (
        not isinstance(judgment_hashes, list)
        or not judgment_hashes
        or any(not isinstance(value, str) for value in judgment_hashes)
        or not isinstance(repair_hashes, list)
        or any(not isinstance(value, str) for value in repair_hashes)
        or len(judgment_hashes) != len(repair_hashes) + 1
    ):
        raise ScenarioBlueprintError("taxonomy admission lineage cardinality is invalid")
    if checkpoint.get("finalFindingClusters") != []:
        raise ScenarioBlueprintError("taxonomy admission requires an empty final typed finding set")
    if judgment_hashes[-1] != checkpoint.get("finalJudgmentCheckpointHash"):
        raise ScenarioBlueprintError("taxonomy admission final judgment lineage mismatch")
    if repair_hashes:
        expected_source_type = "repair"
        expected_source_hash = repair_hashes[-1]
    else:
        expected_source_type = "raw"
        expected_source_hash = checkpoint.get("rawTaxonomyCheckpointHash")
    lineage = {
        "rawTaxonomyCheckpointHash": checkpoint.get("rawTaxonomyCheckpointHash"),
        "judgmentCheckpointHashes": judgment_hashes,
        "repairCheckpointHashes": repair_hashes,
    }
    expected = {
        "schema": TAXONOMY_ADMISSION_CHECKPOINT_SCHEMA,
        "protocolVersion": TAXONOMY_ADMISSION_PROTOCOL_VERSION,
        "protocolHash": taxonomy_admission_protocol_hash(),
        "requestHash": content_hash(request),
        "topicId": topic["topicId"],
        "topicCardHash": content_hash(topic),
        "admittedSourceCheckpointHash": expected_source_hash,
        "admittedSourceType": expected_source_type,
        "taxonomyAnchorsHash": content_hash(anchors),
        "taxonomyPlannerBindingHash": content_hash(bound_planner),
        "taxonomyJudgeBindingHash": content_hash(bound_judge),
        "taxonomyJudgeModelBindingHash": content_hash(
            _taxonomy_judge_model_binding(bound_judge)
        ),
        "taxonomyJudgeSourceHash": TAXONOMY_JUDGE_SOURCE_HASH,
        "taxonomyRepairSourceHash": TAXONOMY_REPAIR_SOURCE_HASH,
        "lineageHash": content_hash(lineage),
    }
    source_proxy = {
        "checkpointHash": checkpoint.get("admittedSourceCheckpointHash"),
        "taxonomyAnchorsHash": checkpoint.get("taxonomyAnchorsHash"),
    }
    final_judgment_proxy = {
        "checkpointHash": checkpoint.get("finalJudgmentCheckpointHash")
    }
    expected["stageKey"] = _admission_stage_key(
        request,
        topic,
        str(checkpoint.get("rawTaxonomyCheckpointHash")),
        source_proxy,
        final_judgment_proxy,
        bound_planner,
        bound_judge,
        judgment_hashes,
        repair_hashes,
    )
    for field, expected_value in expected.items():
        if checkpoint.get(field) != expected_value:
            raise ScenarioBlueprintError(
                f"taxonomy admission checkpoint binding mismatch: {field}"
            )
    _checkpoint_body_hash(checkpoint)
    return checkpoint


def _persist_taxonomy_admission(
    *,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    raw_checkpoint: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    source_type: str,
    final_judgment: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
    judge_binding: Mapping[str, Any],
    judgment_hashes: Sequence[str],
    repair_hashes: Sequence[str],
    output_root: Path,
    resume: bool,
) -> dict[str, Any]:
    if final_judgment["findingClusters"]:
        raise ScenarioBlueprintError("cannot admit a taxonomy with typed finding clusters")
    stage_key = _admission_stage_key(
        request,
        topic,
        raw_checkpoint["checkpointHash"],
        source_checkpoint,
        final_judgment,
        planner_binding,
        judge_binding,
        judgment_hashes,
        repair_hashes,
    )
    path = _checkpoint_path(
        Path(output_root), "taxonomy_admissions", str(topic["topicId"]), stage_key
    )
    if path.exists():
        if not resume:
            raise ScenarioBlueprintError(
                f"taxonomy admission exists; use --resume: {path}"
            )
        return validate_taxonomy_admission_checkpoint(
            read_json(path), request, topic, planner_binding
        )
    lineage = {
        "rawTaxonomyCheckpointHash": raw_checkpoint["checkpointHash"],
        "judgmentCheckpointHashes": list(judgment_hashes),
        "repairCheckpointHashes": list(repair_hashes),
    }
    body = {
        "schema": TAXONOMY_ADMISSION_CHECKPOINT_SCHEMA,
        "stageKey": stage_key,
        "protocolVersion": TAXONOMY_ADMISSION_PROTOCOL_VERSION,
        "protocolHash": taxonomy_admission_protocol_hash(),
        "requestHash": content_hash(request),
        "topicId": topic["topicId"],
        "topicCardHash": content_hash(topic),
        "rawTaxonomyCheckpointHash": raw_checkpoint["checkpointHash"],
        "admittedSourceCheckpointHash": source_checkpoint["checkpointHash"],
        "admittedSourceType": source_type,
        "taxonomyAnchors": source_checkpoint["taxonomyAnchors"],
        "taxonomyAnchorsHash": source_checkpoint["taxonomyAnchorsHash"],
        "taxonomyPlannerBinding": dict(planner_binding),
        "taxonomyPlannerBindingHash": content_hash(planner_binding),
        "taxonomyJudgeBinding": dict(judge_binding),
        "taxonomyJudgeBindingHash": content_hash(judge_binding),
        "taxonomyJudgeModelBindingHash": content_hash(
            _taxonomy_judge_model_binding(judge_binding)
        ),
        "taxonomyJudgeSourceHash": TAXONOMY_JUDGE_SOURCE_HASH,
        "taxonomyRepairSourceHash": TAXONOMY_REPAIR_SOURCE_HASH,
        "finalJudgmentCheckpointHash": final_judgment["checkpointHash"],
        "finalFindingClusters": [],
        "judgmentCheckpointHashes": list(judgment_hashes),
        "repairCheckpointHashes": list(repair_hashes),
        "lineageHash": content_hash(lineage),
    }
    checkpoint = dict(body)
    checkpoint["checkpointHash"] = content_hash(body)
    _write_immutable_json(path, checkpoint)
    return checkpoint


def admit_topic_taxonomy(
    *,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    output_root: Path,
    planner: StrictSchemaModel,
    repair_planner: StrictSchemaModel | None = None,
    judge: TaxonomyJudge,
    max_attempts: int = 4,
    max_repair_cycles: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    """Generate, independently judge, repair, and admit one Stage-T taxonomy."""

    if not 1 <= max_attempts <= 12 or not 0 <= max_repair_cycles <= 12:
        raise ScenarioBlueprintError(
            "max_attempts must be in [1,12] and taxonomy repair cycles in [0,12]"
        )
    planner_binding = _model_binding(planner)
    active_repair_planner = repair_planner or planner
    repair_planner_binding = _model_binding(active_repair_planner)
    judge_binding = _taxonomy_judge_binding(judge)
    _ensure_independent_bindings(planner_binding, judge_binding)
    _ensure_independent_bindings(repair_planner_binding, judge_binding)
    raw_checkpoint = generate_topic_taxonomy(
        request=request,
        topic=topic,
        output_root=Path(output_root),
        planner=planner,
        max_attempts=max_attempts,
        resume=resume,
    )
    if raw_checkpoint.get("plannerBindingHash") != content_hash(planner_binding):
        raise ScenarioBlueprintError("raw taxonomy planner binding mismatch")

    source_checkpoint = raw_checkpoint
    source_type = "raw"
    judgment_hashes: list[str] = []
    repair_hashes: list[str] = []
    for judgment_cycle in range(0, max_repair_cycles + 1):
        judgment = generate_taxonomy_judgment(
            request=request,
            topic=topic,
            source_checkpoint=source_checkpoint,
            source_type=source_type,
            judgment_cycle=judgment_cycle,
            raw_checkpoint_hash=raw_checkpoint["checkpointHash"],
            output_root=Path(output_root),
            planner_binding=planner_binding,
            judge=judge,
            max_attempts=max_attempts,
            resume=resume,
        )
        judgment_hashes.append(judgment["checkpointHash"])
        if not judgment["findingClusters"]:
            return _persist_taxonomy_admission(
                request=request,
                topic=topic,
                raw_checkpoint=raw_checkpoint,
                source_checkpoint=source_checkpoint,
                source_type=source_type,
                final_judgment=judgment,
                planner_binding=planner_binding,
                judge_binding=judge_binding,
                judgment_hashes=judgment_hashes,
                repair_hashes=repair_hashes,
                output_root=Path(output_root),
                resume=resume,
            )
        if judgment_cycle == max_repair_cycles:
            break
        source_checkpoint = repair_topic_taxonomy(
            request=request,
            topic=topic,
            source_checkpoint=source_checkpoint,
            source_judgment=judgment,
            raw_checkpoint_hash=raw_checkpoint["checkpointHash"],
            output_root=Path(output_root),
            planner=active_repair_planner,
            repair_cycle=judgment_cycle + 1,
            max_attempts=max_attempts,
            resume=resume,
        )
        repair_hashes.append(source_checkpoint["checkpointHash"])
        source_type = "repair"
    raise ScenarioBlueprintError(
        f"taxonomy admission exhausted {max_repair_cycles} repair cycles for "
        f"{topic['topicId']}"
    )
