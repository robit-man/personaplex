"""Immutable judge-driven repair of rejected PersonaPlex v5 blueprint slots."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from pathlib import Path
import json

from jsonschema import Draft202012Validator

from ground_truth_finetuning.training.scenario_blueprint_v5 import (
    BLUEPRINT_SETS_FILENAME,
    BLUEPRINT_SCRUTINY_FILENAME,
    BLUEPRINT_MAX_OUTPUT_TOKENS,
    BLUEPRINTS_PER_TOPIC,
    MAX_WORKERS,
    InvalidModelOutput,
    ModelTransportUnavailable,
    ScenarioBlueprintError,
    StrictSchemaModel,
    _call_model,
    _checkpoint_body_hash,
    _checkpoint_path,
    _model_binding,
    _parallel,
    _raise_schema_errors,
    _write_immutable_json,
    _write_immutable_jsonl,
    build_blueprint_response_schema,
    canonical_json,
    content_hash,
    decode_blueprint_response,
    encode_blueprint_response,
    generate_topic_blueprints,
    generate_topic_blueprint_scrutiny,
    make_blueprint_set,
    read_json,
    scenario_ids_for_topic,
    validate_corpus_shape,
    validate_blueprint_scrutiny,
    validate_blueprint_set,
)
from ground_truth_finetuning.training.scenario_taxonomy_admission_v5 import (
    TAXONOMY_FIELDS,
    admit_topic_taxonomy,
    validate_taxonomy_admission_checkpoint,
    validate_taxonomy_anchors,
)


BLUEPRINT_REPAIR_PROTOCOL_VERSION = "finding-cluster-adaptive-repair-v3-taxonomy-bound"
BLUEPRINT_REPAIR_CHECKPOINT_SCHEMA = "personaplex.scenario-blueprint-repair-checkpoint.v5"

REPAIR_SYSTEM = """You repair only judge-rejected slots in a joint PersonaPlex scenario blueprint.
Return only the strict compact JSON object and only the required scenario-ID properties. Preserve every
schema-fixed coverage cell. Replace the rejected niches with materially different topic-valid scenarios
that resolve every cited finding while remaining distinct from all immutable slots. Do not emit dialogue,
utterances, scripts, canonical responses, target text/audio, names, contact data, credentials, or
placeholders. Never modify or restate an immutable slot."""


def _bound_taxonomy_anchors(
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    taxonomy_anchors: Mapping[str, Any] | None,
) -> dict[str, dict[str, str]]:
    """Resolve certified anchors, or safely pin every field to the parent set."""

    candidate = taxonomy_anchors
    if candidate is None:
        candidate = {
            scenario_id: {
                field: blueprint_set["blueprints"][scenario_id][field]
                for field in TAXONOMY_FIELDS
            }
            for scenario_id in blueprint_set["scenarioIds"]
        }
    return validate_taxonomy_anchors(candidate, topic)


def repair_ids_from_scrutiny(
    blueprint_set: Mapping[str, Any], blueprint_scrutiny: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return the exact ordered union of IDs named by authentic finding clusters."""

    scenario_ids = tuple(blueprint_set["scenarioIds"])
    decision = blueprint_scrutiny.get("decision")
    if not isinstance(decision, Mapping) or decision.get("groupDecision") != "reject":
        raise ScenarioBlueprintError("blueprint repair requires a rejecting scrutiny decision")
    implicated = {
        scenario_id
        for finding in decision.get("findings", [])
        for scenario_id in finding.get("scenarioIds", [])
    }
    unknown = implicated.difference(scenario_ids)
    if unknown:
        raise ScenarioBlueprintError(f"blueprint repair findings contain unknown IDs: {sorted(unknown)}")
    ordered = tuple(scenario_id for scenario_id in scenario_ids if scenario_id in implicated)
    if not ordered:
        raise ScenarioBlueprintError("rejecting scrutiny contains no repairable scenario IDs")
    return ordered


def build_blueprint_repair_response_schema(
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    repair_ids: Sequence[str],
    taxonomy_anchors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project the complete compact schema onto exactly the rejected slot IDs."""

    validate_blueprint_set(blueprint_set, topic)
    ordered_repair_ids = tuple(repair_ids)
    scenario_ids = tuple(blueprint_set["scenarioIds"])
    if (
        not ordered_repair_ids
        or len(set(ordered_repair_ids)) != len(ordered_repair_ids)
        or any(scenario_id not in scenario_ids for scenario_id in ordered_repair_ids)
    ):
        raise ScenarioBlueprintError("repair IDs must be a nonempty unique subset of the joint blueprint")
    bound_anchors = _bound_taxonomy_anchors(topic, blueprint_set, taxonomy_anchors)
    full_schema = build_blueprint_response_schema(topic, scenario_ids, bound_anchors)
    schema = {
        "$schema": full_schema["$schema"],
        "type": "object",
        "additionalProperties": False,
        "required": list(ordered_repair_ids),
        "properties": {
            scenario_id: full_schema["properties"][scenario_id]
            for scenario_id in ordered_repair_ids
        },
    }
    Draft202012Validator.check_schema(schema)
    if "prefixItems" in canonical_json(schema):
        raise ScenarioBlueprintError("blueprint repair schema must never use prefixItems")
    return schema


def merge_blueprint_repair_response(
    response: Any,
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    repair_ids: Sequence[str],
    taxonomy_anchors: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Decode rejected slots and merge them without changing any accepted slot."""

    ordered_repair_ids = tuple(repair_ids)
    bound_anchors = _bound_taxonomy_anchors(topic, blueprint_set, taxonomy_anchors)
    schema = build_blueprint_repair_response_schema(
        topic, blueprint_set, ordered_repair_ids, bound_anchors
    )
    _raise_schema_errors(response, schema, "blueprint repair response")
    if not isinstance(response, dict):
        raise InvalidModelOutput("blueprint repair response must be an object")
    complete_wire = encode_blueprint_response(
        blueprint_set["blueprints"], topic, taxonomy_anchors=bound_anchors
    )
    for scenario_id in ordered_repair_ids:
        complete_wire[scenario_id] = response[scenario_id]
    merged = decode_blueprint_response(
        complete_wire, topic, taxonomy_anchors=bound_anchors
    )
    repair_id_set = set(ordered_repair_ids)
    for scenario_id in blueprint_set["scenarioIds"]:
        before = blueprint_set["blueprints"][scenario_id]
        after = merged[scenario_id]
        if scenario_id not in repair_id_set and after != before:
            raise InvalidModelOutput(f"repair altered immutable slot {scenario_id}")
        if scenario_id in repair_id_set and after == before:
            raise InvalidModelOutput(f"repair left rejected slot unchanged: {scenario_id}")
        if scenario_id in repair_id_set:
            for field in TAXONOMY_FIELDS:
                if after[field] != bound_anchors[scenario_id][field]:
                    raise InvalidModelOutput(
                        f"repair changed admitted taxonomy field {scenario_id}.{field}"
                    )
    return merged


def _repair_stage_key(
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    blueprint_scrutiny: Mapping[str, Any],
    repair_ids: Sequence[str],
    response_schema: Mapping[str, Any],
    taxonomy_admission_checkpoint_hash: str | None,
    taxonomy_anchors: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
    repair_cycle: int,
) -> str:
    return content_hash(
        {
            "stage": "scenario_blueprint_repair_v5",
            "protocolVersion": BLUEPRINT_REPAIR_PROTOCOL_VERSION,
            "systemHash": content_hash(REPAIR_SYSTEM),
            "repairCycle": repair_cycle,
            "requestHash": content_hash(request),
            "topicCardHash": content_hash(topic),
            "parentBlueprintSetHash": content_hash(blueprint_set),
            "sourceScrutinyHash": blueprint_scrutiny["checkpointHash"],
            "repairIds": list(repair_ids),
            "responseSchemaHash": content_hash(response_schema),
            "taxonomyAdmissionCheckpointHash": taxonomy_admission_checkpoint_hash,
            "admittedTaxonomyAnchorsHash": content_hash(taxonomy_anchors),
            "plannerBindingHash": content_hash(planner_binding),
        }
    )


def _validate_repair_checkpoint(
    checkpoint: Any,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    blueprint_scrutiny: Mapping[str, Any],
    repair_ids: Sequence[str],
    response_schema: Mapping[str, Any],
    taxonomy_admission_checkpoint_hash: str | None,
    taxonomy_anchors: Mapping[str, Any],
    planner_binding: Mapping[str, Any],
    repair_cycle: int,
) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ScenarioBlueprintError("blueprint repair checkpoint must be an object")
    required = {
        "schema", "stageKey", "protocolVersion", "repairCycle", "requestHash",
        "topicId", "topicCardHash", "repairScenarioIds", "parentBlueprintSetHash",
        "parentJointBlueprintHash", "sourceScrutinyHash", "responseSchemaHash",
        "taxonomyAdmissionCheckpointHash", "admittedTaxonomyAnchorsHash",
        "repairPlannerBinding", "repairPlannerBindingHash", "modelCall",
        "repairedBlueprintSet", "repairedBlueprintSetHash", "checkpointHash",
    }
    if set(checkpoint) != required:
        raise ScenarioBlueprintError("blueprint repair checkpoint has an invalid field set")
    expected_stage_key = _repair_stage_key(
        request, topic, blueprint_set, blueprint_scrutiny, repair_ids,
        response_schema, taxonomy_admission_checkpoint_hash, taxonomy_anchors,
        planner_binding, repair_cycle
    )
    expected = {
        "schema": BLUEPRINT_REPAIR_CHECKPOINT_SCHEMA,
        "stageKey": expected_stage_key,
        "protocolVersion": BLUEPRINT_REPAIR_PROTOCOL_VERSION,
        "repairCycle": repair_cycle,
        "requestHash": content_hash(request),
        "topicId": topic["topicId"],
        "topicCardHash": content_hash(topic),
        "repairScenarioIds": list(repair_ids),
        "parentBlueprintSetHash": content_hash(blueprint_set),
        "parentJointBlueprintHash": blueprint_set["jointBlueprintHash"],
        "sourceScrutinyHash": blueprint_scrutiny["checkpointHash"],
        "responseSchemaHash": content_hash(response_schema),
        "taxonomyAdmissionCheckpointHash": taxonomy_admission_checkpoint_hash,
        "admittedTaxonomyAnchorsHash": content_hash(taxonomy_anchors),
        "repairPlannerBinding": dict(planner_binding),
        "repairPlannerBindingHash": content_hash(planner_binding),
    }
    for field, expected_value in expected.items():
        if checkpoint.get(field) != expected_value:
            raise ScenarioBlueprintError(f"blueprint repair binding mismatch: {field}")
    repaired_set = validate_blueprint_set(checkpoint.get("repairedBlueprintSet"), topic)
    if checkpoint.get("repairedBlueprintSetHash") != content_hash(repaired_set):
        raise ScenarioBlueprintError("repaired blueprint set hash mismatch")
    if repaired_set["jointBlueprintHash"] == blueprint_set["jointBlueprintHash"]:
        raise ScenarioBlueprintError("blueprint repair did not change the joint blueprint")
    repair_id_set = set(repair_ids)
    for scenario_id in blueprint_set["scenarioIds"]:
        before = blueprint_set["blueprints"][scenario_id]
        after = repaired_set["blueprints"][scenario_id]
        if scenario_id not in repair_id_set and before != after:
            raise ScenarioBlueprintError(f"repaired checkpoint changed accepted slot {scenario_id}")
        if scenario_id in repair_id_set and before == after:
            raise ScenarioBlueprintError(f"repaired checkpoint retained rejected slot {scenario_id}")
        if scenario_id in repair_id_set:
            for field in TAXONOMY_FIELDS:
                if after[field] != taxonomy_anchors[scenario_id][field]:
                    raise ScenarioBlueprintError(
                        f"repaired checkpoint changed admitted taxonomy field "
                        f"{scenario_id}.{field}"
                    )
    if not isinstance(checkpoint.get("modelCall"), dict):
        raise ScenarioBlueprintError("blueprint repair modelCall must be an object")
    _checkpoint_body_hash(checkpoint)
    return checkpoint


def repair_topic_blueprints(
    *,
    request: Mapping[str, Any],
    topic: Mapping[str, Any],
    blueprint_set: Mapping[str, Any],
    blueprint_scrutiny: Mapping[str, Any],
    taxonomy_admission: Mapping[str, Any] | None = None,
    output_root: Path,
    planner: StrictSchemaModel,
    repair_cycle: int,
    max_attempts: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    """Repair exactly one authentic finding-cluster union and persist its lineage."""

    if not 1 <= repair_cycle <= 12 or not 1 <= max_attempts <= 12:
        raise ScenarioBlueprintError("repair_cycle and max_attempts must be in [1,12]")
    validate_blueprint_set(blueprint_set, topic)
    validate_blueprint_scrutiny(
        blueprint_scrutiny, request, topic, blueprint_set, require_pass=False
    )
    planner_binding = _model_binding(planner)
    taxonomy_admission_checkpoint_hash = None
    if taxonomy_admission is None:
        taxonomy_anchors = _bound_taxonomy_anchors(topic, blueprint_set, None)
    else:
        admitted = validate_taxonomy_admission_checkpoint(
            taxonomy_admission, request, topic, planner_binding
        )
        taxonomy_admission_checkpoint_hash = admitted["checkpointHash"]
        taxonomy_anchors = _bound_taxonomy_anchors(
            topic, blueprint_set, admitted["taxonomyAnchors"]
        )
    repair_ids = repair_ids_from_scrutiny(blueprint_set, blueprint_scrutiny)
    response_schema = build_blueprint_repair_response_schema(
        topic, blueprint_set, repair_ids, taxonomy_anchors
    )
    stage_key = _repair_stage_key(
        request, topic, blueprint_set, blueprint_scrutiny, repair_ids,
        response_schema, taxonomy_admission_checkpoint_hash, taxonomy_anchors,
        planner_binding, repair_cycle
    )
    path = _checkpoint_path(
        Path(output_root), "blueprint_repairs", f"{topic['topicId']}_r{repair_cycle:02d}", stage_key
    )
    if path.exists():
        if not resume:
            raise ScenarioBlueprintError(f"blueprint repair exists; use --resume: {path}")
        return _validate_repair_checkpoint(
            read_json(path), request, topic, blueprint_set, blueprint_scrutiny,
            repair_ids, response_schema, taxonomy_admission_checkpoint_hash,
            taxonomy_anchors, planner_binding, repair_cycle
        )

    repair_id_set = set(repair_ids)
    context = {
        "task": "Repair exactly the scenario IDs named by the independent finding clusters.",
        "topicCard": dict(topic),
        "parentJointCompactBlueprint": dict(blueprint_set),
        "independentJudgment": blueprint_scrutiny["decision"],
        "admittedTaxonomyAnchors": {
            scenario_id: taxonomy_anchors[scenario_id] for scenario_id in repair_ids
        },
        "repairScenarioIds": list(repair_ids),
        "immutableScenarioIds": [
            scenario_id for scenario_id in blueprint_set["scenarioIds"]
            if scenario_id not in repair_id_set
        ],
        "repairContract": [
            "Output only repairScenarioIds; accepted slots are immutable context.",
            "Resolve every cited semantic cluster, mode collapse, and causal-collapse finding.",
            "Preserve each repaired slot's schema-fixed coverage lattice cell.",
            "Preserve all five admitted taxonomy fields exactly.",
            "Keep target dialogue, target text, and target audio absent.",
        ],
    }
    failures: list[str] = []
    for attempt in range(1, max_attempts + 1):
        attempt_context = dict(context)
        if failures:
            attempt_context["retryFeedback"] = {
                "attempt": attempt,
                "previousDefect": failures[-1],
                "directive": (
                    "Regenerate only the rejected slots and specifically correct this defect while "
                    "leaving all immutable slots untouched."
                ),
            }
        try:
            response, metadata = _call_model(
                planner,
                name="personaplex_scenario_blueprint_repair_v5",
                schema=response_schema,
                instructions=REPAIR_SYSTEM,
                context=attempt_context,
                max_output_tokens=BLUEPRINT_MAX_OUTPUT_TOKENS,
            )
            merged = merge_blueprint_repair_response(
                response, topic, blueprint_set, repair_ids, taxonomy_anchors
            )
            repaired_set = make_blueprint_set(topic, merged, planner_binding)
            body = {
                "schema": BLUEPRINT_REPAIR_CHECKPOINT_SCHEMA,
                "stageKey": stage_key,
                "protocolVersion": BLUEPRINT_REPAIR_PROTOCOL_VERSION,
                "repairCycle": repair_cycle,
                "requestHash": content_hash(request),
                "topicId": topic["topicId"],
                "topicCardHash": content_hash(topic),
                "repairScenarioIds": list(repair_ids),
                "parentBlueprintSetHash": content_hash(blueprint_set),
                "parentJointBlueprintHash": blueprint_set["jointBlueprintHash"],
                "sourceScrutinyHash": blueprint_scrutiny["checkpointHash"],
                "responseSchemaHash": content_hash(response_schema),
                "taxonomyAdmissionCheckpointHash": taxonomy_admission_checkpoint_hash,
                "admittedTaxonomyAnchorsHash": content_hash(taxonomy_anchors),
                "repairPlannerBinding": planner_binding,
                "repairPlannerBindingHash": content_hash(planner_binding),
                "modelCall": metadata,
                "repairedBlueprintSet": repaired_set,
                "repairedBlueprintSetHash": content_hash(repaired_set),
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
        f"blueprint repair exhausted {max_attempts} attempts for {topic['topicId']} "
        f"cycle {repair_cycle}: " + " | ".join(failures)
    )


def admit_blueprints(
    *,
    request: Mapping[str, Any],
    topics: Sequence[Mapping[str, Any]],
    output_root: Path,
    planner: StrictSchemaModel,
    taxonomy_repair_planner: StrictSchemaModel | None = None,
    taxonomy_judge: Any,
    judge: Any,
    max_workers: int = MAX_WORKERS,
    judge_workers: int = 1,
    max_attempts: int = 4,
    max_repair_cycles: int = 4,
    resume: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the staged 50-topic generate/judge/repair admission transaction."""

    validate_corpus_shape(request, topics)
    if not 1 <= max_workers <= MAX_WORKERS or not 1 <= judge_workers <= MAX_WORKERS:
        raise ScenarioBlueprintError("planner and judge workers must be in [1,3]")
    if not 1 <= max_attempts <= 12 or not 0 <= max_repair_cycles <= 12:
        raise ScenarioBlueprintError("max_attempts must be in [1,12] and repair cycles in [0,12]")
    output_root = Path(output_root)
    ordered_topics = sorted(topics, key=lambda item: str(item["topicId"]))
    topic_by_id = {str(topic["topicId"]): topic for topic in ordered_topics}

    taxonomy_admissions = _parallel(
        ordered_topics,
        lambda topic: admit_topic_taxonomy(
            request=request,
            topic=topic,
            output_root=output_root,
            planner=planner,
            repair_planner=taxonomy_repair_planner,
            judge=taxonomy_judge,
            max_attempts=max_attempts,
            max_repair_cycles=max_repair_cycles,
            resume=resume,
        ),
        max_workers=judge_workers,
        identity=lambda topic: str(topic["topicId"]),
    )
    taxonomy_admission_by_topic = {
        str(checkpoint["topicId"]): checkpoint for checkpoint in taxonomy_admissions
    }

    base_checkpoints = _parallel(
        ordered_topics,
        lambda topic: generate_topic_blueprints(
            request=request,
            topic=topic,
            output_root=output_root,
            planner=planner,
            taxonomy_admission=taxonomy_admission_by_topic[str(topic["topicId"])],
            max_attempts=max_attempts,
            resume=resume,
        ),
        max_workers=max_workers,
        identity=lambda topic: str(topic["topicId"]),
    )
    set_by_topic = {
        str(checkpoint["topicId"]): checkpoint["blueprintSet"]
        for checkpoint in base_checkpoints
    }
    _write_immutable_jsonl(
        output_root / "scenario_blueprint_sets.cycle00.jsonl",
        [set_by_topic[topic_id] for topic_id in sorted(set_by_topic)],
        resume=resume,
    )

    scrutiny_by_topic: dict[str, dict[str, Any]] = {}

    def judge_topics(topic_ids: Sequence[str], cycle: int) -> None:
        judged = _parallel(
            list(topic_ids),
            lambda topic_id: generate_topic_blueprint_scrutiny(
                request=request,
                topic=topic_by_id[topic_id],
                blueprint_set=set_by_topic[topic_id],
                output_root=output_root,
                judge=judge,
                max_attempts=max_attempts,
                resume=resume,
            ),
            max_workers=judge_workers,
            identity=lambda topic_id: topic_id,
        )
        for checkpoint in judged:
            scrutiny_by_topic[str(checkpoint["topicId"])] = checkpoint
        if set(scrutiny_by_topic) != set(topic_by_id):
            missing = sorted(set(topic_by_id).difference(scrutiny_by_topic))
            if cycle == 0 or any(topic_id in topic_ids for topic_id in missing):
                raise ScenarioBlueprintError(
                    f"blueprint scrutiny cycle {cycle} is incomplete: {missing}"
                )
        _write_immutable_jsonl(
            output_root / f"scenario_blueprint_scrutiny.cycle{cycle:02d}.jsonl",
            [scrutiny_by_topic[topic_id] for topic_id in sorted(scrutiny_by_topic)],
            resume=resume,
        )

    judge_topics(tuple(sorted(topic_by_id)), 0)
    for cycle in range(1, max_repair_cycles + 1):
        rejected = tuple(
            topic_id
            for topic_id in sorted(topic_by_id)
            if scrutiny_by_topic[topic_id]["decision"]["groupDecision"] != "pass"
        )
        if not rejected:
            break
        assignments = [
            (
                topic_id,
                topic_by_id[topic_id],
                set_by_topic[topic_id],
                scrutiny_by_topic[topic_id],
            )
            for topic_id in rejected
        ]
        repaired = _parallel(
            assignments,
            lambda assignment: repair_topic_blueprints(
                request=request,
                topic=assignment[1],
                blueprint_set=assignment[2],
                blueprint_scrutiny=assignment[3],
                taxonomy_admission=taxonomy_admission_by_topic[assignment[0]],
                output_root=output_root,
                planner=planner,
                repair_cycle=cycle,
                max_attempts=max_attempts,
                resume=resume,
            ),
            max_workers=max_workers,
            identity=lambda assignment: assignment[0],
        )
        for checkpoint in repaired:
            set_by_topic[str(checkpoint["topicId"])] = checkpoint["repairedBlueprintSet"]
        _write_immutable_jsonl(
            output_root / f"scenario_blueprint_sets.cycle{cycle:02d}.jsonl",
            [set_by_topic[topic_id] for topic_id in sorted(set_by_topic)],
            resume=resume,
        )
        judge_topics(rejected, cycle)

    rejected = [
        topic_id
        for topic_id in sorted(topic_by_id)
        if scrutiny_by_topic[topic_id]["decision"]["groupDecision"] != "pass"
    ]
    if rejected:
        raise ScenarioBlueprintError(
            f"blueprint admission exhausted {max_repair_cycles} repair cycles: {rejected}"
        )
    final_sets = [set_by_topic[topic_id] for topic_id in sorted(set_by_topic)]
    final_scrutinies = [
        scrutiny_by_topic[topic_id] for topic_id in sorted(scrutiny_by_topic)
    ]
    _write_immutable_jsonl(
        output_root / BLUEPRINT_SETS_FILENAME, final_sets, resume=resume
    )
    _write_immutable_jsonl(
        output_root / BLUEPRINT_SCRUTINY_FILENAME, final_scrutinies, resume=resume
    )
    return final_sets, final_scrutinies
