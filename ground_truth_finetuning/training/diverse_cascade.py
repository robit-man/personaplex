"""Versioned, resumable planning cascade for controlled PersonaPlex source data.

The cascade creates typed planning artifacts only. It never creates target dialogue,
target audio, or a semantic certificate.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable
import json
import urllib.error
import urllib.request


class CascadeError(RuntimeError):
    """Raised when a cascade artifact cannot safely advance."""


FORBIDDEN_TARGET_KEYS = frozenset({
    "canonical_response", "canonicalresponse", "target_text", "targettext",
    "target_audio", "targetaudio", "spoken_text", "spokentext",
    "expected_response", "expectedresponse", "agent_response", "agentresponse",
})
ROOT_SIBLING_ROLES = (
    "verified_positive",
    "verified_negative",
    "uncertain",
    "superseded",
)
LEGACY_SIBLING_ROLES = ("available", "constrained")
BALANCE_AXES = (
    "causalAxis",
    "interventionFamily",
    "postureTransition",
    "evidenceSource",
    "duplexEventType",
    "outcomeRoute",
    "conversationLength",
    "style",
)
SUPPORTED_REQUEST_SCHEMAS = {
    "personaplex.diverse-corpus-request.v1",
    "personaplex.diverse-corpus-request.v2",
}
SUPPORTED_GROUP_SCHEMAS = {
    "personaplex.counterfactual-pair-spec.v1",
    "personaplex.counterfactual-sibling-group-spec.v1",
    "personaplex.counterfactual-sibling-group-spec.v2",
    "personaplex.counterfactual-group-spec.v1",
}
SUPPORTED_REJECTION_SCHEMAS = {
    "personaplex.rejected-group.v1",
    "personaplex.rejected-causal-group.v1",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any) -> str:
    return f"sha256:{sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


def is_content_hash(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(character in "0123456789abcdef" for character in value[7:])


def load_json(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CascadeError(f"Cannot load JSON {path}: {error}") from error
    if not isinstance(parsed, dict):
        raise CascadeError(f"{path} must contain one JSON object")
    return parsed


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CascadeError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise CascadeError(f"Cannot load JSONL {path}: {error}") from error
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text("".join(f"{canonical_json(record)}\n" for record in records), encoding="utf-8")
    temporary.replace(path)


def require_keys(value: dict[str, Any], required: set[str], label: str) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise CascadeError(f"{label} is missing required fields: {', '.join(missing)}")


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CascadeError(f"{label} must be nonempty text")
    return value


def require_identifier(value: Any, label: str) -> str:
    identifier = str(value or "")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789_:-")
    if not 3 <= len(identifier) <= 200 or identifier[0] not in "abcdefghijklmnopqrstuvwxyz" or any(
        character not in allowed for character in identifier
    ):
        raise CascadeError(f"{label} must be a lowercase structured identifier")
    return identifier


def require_string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise CascadeError(f"{label} must be a nonempty string array")
    return value


def assert_no_target_leak(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_").replace(" ", "_")
            if normalized in FORBIDDEN_TARGET_KEYS:
                raise CascadeError(f"Target-label field {key!r} is forbidden at {path}")
            assert_no_target_leak(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            assert_no_target_leak(child, f"{path}[{index}]")


def _configuration_objects(request: dict[str, Any]) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    for key in ("selection", "selectionTarget", "selectionPlan"):
        value = request.get(key)
        if isinstance(value, dict):
            objects.append(value)
    coverage = request.get("coverageTarget")
    if isinstance(coverage, dict):
        objects.append(coverage)
    return objects


def _configured_integer(request: dict[str, Any], names: tuple[str, ...], default: int | None = None) -> int:
    for source in _configuration_objects(request):
        for name in names:
            if name in source:
                value = source[name]
                if not isinstance(value, int) or isinstance(value, bool):
                    raise CascadeError(f"{name} must be an integer")
                return value
    if default is None:
        raise CascadeError(f"Request must define one of {', '.join(names)}")
    return default


def request_selection_counts(request: dict[str, Any]) -> tuple[int, int]:
    primary = _configured_integer(
        request,
        ("primaryGroups", "primaryGroupCount", "primaryCounterfactualGroups", "selectedCounterfactualGroups", "selectedGroups"),
    )
    reserve = _configured_integer(
        request,
        ("reserveGroups", "reserveGroupCount", "reserveCounterfactualGroups"),
        default=0,
    )
    return primary, reserve


def _declared_sibling_roles(request: dict[str, Any]) -> list[str] | None:
    for key in ("causalGroupContract", "counterfactualGroup", "siblingGroupContract", "causalGroup"):
        contract = request.get(key)
        if isinstance(contract, dict):
            roles = contract.get("siblingRoles")
            if roles is None:
                roles = contract.get("siblings")
            if roles is not None:
                if not isinstance(roles, list) or not roles:
                    raise CascadeError(f"{key}.siblingRoles must be a nonempty array")
                normalized = [item.get("role") if isinstance(item, dict) else item for item in roles]
                if not all(isinstance(item, str) and item for item in normalized):
                    raise CascadeError(f"{key}.siblingRoles must contain typed role names")
                return normalized
    roles = request.get("siblingRoles")
    if roles is not None:
        return require_string_list(roles, "siblingRoles")
    return None


def request_sibling_roles(request: dict[str, Any]) -> tuple[str, ...]:
    coverage = request.get("coverageTarget")
    if not isinstance(coverage, dict):
        raise CascadeError("coverageTarget must be an object")
    declared = _declared_sibling_roles(request)
    count = coverage.get("siblingsPerGroup", coverage.get("branchesPerGroup"))
    if count is None and declared is not None:
        count = len(declared)
    if not isinstance(count, int) or isinstance(count, bool):
        raise CascadeError("coverageTarget must define integer branchesPerGroup or siblingsPerGroup")
    if count == 2:
        roles = tuple(declared or LEGACY_SIBLING_ROLES)
        if set(roles) != set(LEGACY_SIBLING_ROLES) or len(roles) != 2:
            raise CascadeError("Two-branch groups must declare available and constrained")
        return roles
    if count == 4:
        if declared is None:
            raise CascadeError("Four-sibling groups require request-defined siblingRoles")
        roles = tuple(declared)
        if set(roles) != set(ROOT_SIBLING_ROLES) or len(roles) != 4:
            raise CascadeError("Four-sibling groups must declare verified_positive, verified_negative, uncertain, and superseded")
        return roles
    raise CascadeError("Only legacy two-branch or typed four-sibling causal groups are supported")


def request_requires_typed_trajectories(request: dict[str, Any]) -> bool:
    return len(request_sibling_roles(request)) == 4


def declared_catalog_hash(request: dict[str, Any]) -> str | None:
    if "seedCatalog" not in request:
        return None
    declared = request.get("seedCatalogHash", request.get("seedRevision"))
    if not is_content_hash(declared):
        raise CascadeError("seedCatalogHash (or legacy seedRevision) must be a SHA-256 content hash")
    return str(declared)


def validate_request(request: dict[str, Any]) -> None:
    require_keys(request, {
        "schema", "requestId", "seedRevision", "coverageTarget",
        "allowedVoicesManifest", "renderer", "asr", "allowedPhysicalCudaDevices",
        "prohibitedContentPolicyRevision",
    }, "DiverseCorpusRequest")
    if request["schema"] not in SUPPORTED_REQUEST_SCHEMAS:
        raise CascadeError("Unsupported request schema")
    strategy = request.get("strategyVersion")
    if strategy not in {None, "semantic-control-v4", "semantic-control-v5"}:
        raise CascadeError("Unsupported semantic-control strategyVersion")
    if request["schema"] == "personaplex.diverse-corpus-request.v2" and strategy != "semantic-control-v5":
        raise CascadeError("DiverseCorpusRequestV2 requires semantic-control-v5")
    if strategy == "semantic-control-v5" and request["schema"] != "personaplex.diverse-corpus-request.v2":
        raise CascadeError("semantic-control-v5 requires DiverseCorpusRequestV2")
    if not isinstance(request["requestId"], str) or len(request["requestId"].strip()) < 8:
        raise CascadeError("requestId must be a nonempty stable identifier")
    for field in ("seedRevision", "allowedVoicesManifest"):
        if not is_content_hash(request[field]):
            raise CascadeError(f"{field} must be a SHA-256 content hash")
    if "seedCatalogHash" in request and not is_content_hash(request["seedCatalogHash"]):
        raise CascadeError("seedCatalogHash must be a SHA-256 content hash")
    if "seedCatalogHash" in request and request["seedCatalogHash"] != request["seedRevision"]:
        raise CascadeError("seedRevision and seedCatalogHash must identify the same canonical catalog")
    if "seedCatalog" in request:
        catalog = request["seedCatalog"]
        if not isinstance(catalog, (str, dict)) or (isinstance(catalog, str) and not catalog.strip()):
            raise CascadeError("seedCatalog must be a path or an inline catalog object")
        declared_catalog_hash(request)
    elif "seedIdeas" not in request:
        raise CascadeError("Request must bind a seedCatalog or provide legacy seedIdeas")
    if "seedIdeas" in request:
        require_string_list(request["seedIdeas"], "seedIdeas")
    coverage = request["coverageTarget"]
    if not isinstance(coverage, dict):
        raise CascadeError("coverageTarget must be an object")
    require_keys(coverage, {"candidateTopics", "scenariosPerTopic", "trajectorySeedsPerScenario"}, "coverageTarget")
    lattice_counts = [coverage[name] for name in ("candidateTopics", "scenariosPerTopic", "trajectorySeedsPerScenario")]
    if not all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in lattice_counts):
        raise CascadeError("coverage lattice counts must be positive integers")
    primary, reserve = request_selection_counts(request)
    if primary <= 0 or reserve < 0:
        raise CascadeError("Primary groups must be positive and reserve groups cannot be negative")
    roles = request_sibling_roles(request)
    selected = coverage.get("selectedCounterfactualGroups")
    if selected is not None and selected != primary:
        raise CascadeError("selectedCounterfactualGroups must equal primaryGroups")
    candidate_count = coverage["candidateTopics"] * coverage["scenariosPerTopic"] * coverage["trajectorySeedsPerScenario"]
    if primary + reserve > candidate_count:
        raise CascadeError("Primary and reserve groups exceed candidate lattice size")
    if request["renderer"] != "voicebox_chatterbox_turbo" or request["asr"] != "whisper":
        raise CascadeError("This programme is pinned to Chatterbox Turbo and Whisper")
    if request["allowedPhysicalCudaDevices"] != [0, 1, 2]:
        raise CascadeError("Only physical CUDA devices 0, 1, and 2 are permitted")
    if strategy == "semantic-control-v4":
        semantic = request.get("semanticControl")
        if not isinstance(semantic, dict):
            raise CascadeError("semantic-control-v4 requires semanticControl")
        require_keys(semantic, {
            "conditioning", "controlAvailability", "mutableRevisions",
            "targetLeakageProhibited", "requiredCausalAxes", "negativeControls",
            "termination", "sharedPrefixPolicy",
        }, "semanticControl")
        if semantic["conditioning"] != "gated_temporal_streaming_sum":
            raise CascadeError("v4 conditioning must use gated temporal streaming sum")
        if semantic["controlAvailability"] != "before_every_agent_target":
            raise CascadeError("v4 control must be available before every agent target")
        if semantic["mutableRevisions"] is not True or semantic["targetLeakageProhibited"] is not True:
            raise CascadeError("v4 requires mutable revisions and target-leak prevention")
        if semantic["termination"] != "model_selected_end_call_tool":
            raise CascadeError("v4 termination must be model-selected")
        if semantic["sharedPrefixPolicy"] != "native_code_identical_through_pivot":
            raise CascadeError("v4 groups require native-code-identical prefixes")
        require_string_list(semantic["requiredCausalAxes"], "semanticControl.requiredCausalAxes")
        require_string_list(semantic["negativeControls"], "semanticControl.negativeControls")
    if strategy == "semantic-control-v5":
        if roles != ROOT_SIBLING_ROLES:
            raise CascadeError("semantic-control-v5 requires the ordered four-role causal contract")
        semantic = request.get("semanticControl")
        if not isinstance(semantic, dict):
            raise CascadeError("semantic-control-v5 requires semanticControl")
        require_keys(semantic, {
            "conditioning", "controlAvailability", "mutableRevisions",
            "targetLeakageProhibited", "requiredCausalAxes", "negativeControls",
            "termination", "sharedPrefixPolicy", "exactWordingFallback",
        }, "semanticControl")
        expected = {
            "conditioning": "native_temporal_streaming_sum",
            "controlAvailability": "strictly_before_every_agent_target",
            "mutableRevisions": True,
            "targetLeakageProhibited": True,
            "termination": "model_selected_end_call_tool",
            "sharedPrefixPolicy": "native_code_identical_through_pivot",
            "exactWordingFallback": "validated_strict_renderer",
        }
        mismatches = {key: value for key, value in expected.items() if semantic.get(key) != value}
        if mismatches:
            raise CascadeError(f"semantic-control-v5 contract mismatch: {mismatches}")
        if tuple(semantic.get("negativeControls", ())) != (
            "paired_wrong_branch", "stale_revision", "null_control",
        ):
            raise CascadeError("semantic-control-v5 requires the three ordered hard negatives")
        require_string_list(semantic["requiredCausalAxes"], "semanticControl.requiredCausalAxes")
        planner = request.get("planner")
        if not isinstance(planner, dict) or planner.get("reasoning") is not False or planner.get("endpointSource") != "personaplex_runtime_contract":
            raise CascadeError("semantic-control-v5 planner must disable reasoning and use the runtime endpoint contract")
        resources = request.get("resourcePolicy")
        if (
            not isinstance(resources, dict)
            or resources.get("hardwareDiscovery") != "runtime"
            or resources.get("cpuModelFallback") is not False
            or resources.get("hostMemoryMetric") != "proc_meminfo_MemAvailable"
            or not isinstance(resources.get("maxHostMemoryUsedFraction"), (int, float))
            or not 0 < float(resources["maxHostMemoryUsedFraction"]) <= 0.8
        ):
            raise CascadeError("semantic-control-v5 requires dynamic host-memory admission and no CPU model fallback")
    assert_no_target_leak(request)


def validate_seed_catalog(catalog: dict[str, Any], request: dict[str, Any]) -> str:
    require_keys(catalog, {"schema", "libraryId", "seeds"}, "SeedCatalog")
    if catalog["schema"] not in {
        "personaplex.diverse-seed-library.v1",
        "personaplex.diverse-seed-catalog.v1",
        "personaplex.diverse-seed-library.v2",
    }:
        raise CascadeError("Unsupported seed catalog schema")
    require_text(catalog["libraryId"], "SeedCatalog.libraryId")
    seeds = catalog["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise CascadeError("SeedCatalog.seeds must be a nonempty array")
    seed_ids: list[str] = []
    for index, seed in enumerate(seeds):
        if not isinstance(seed, dict):
            raise CascadeError(f"SeedCatalog.seeds[{index}] must be an object")
        require_keys(seed, {"id", "title", "focus"}, f"SeedCatalog.seeds[{index}]")
        seed_ids.append(require_text(seed["id"], f"SeedCatalog.seeds[{index}].id"))
        require_text(seed["title"], f"SeedCatalog.seeds[{index}].title")
        require_text(seed["focus"], f"SeedCatalog.seeds[{index}].focus")
        if catalog["schema"] == "personaplex.diverse-seed-library.v2":
            affordances = seed.get("causalAffordances")
            if not isinstance(affordances, list) or len(affordances) != 3:
                raise CascadeError("Every v2 seed requires exactly three causalAffordances")
            families = set()
            for affordance in affordances:
                if not isinstance(affordance, dict):
                    raise CascadeError("Seed causalAffordances must be typed objects")
                family = require_text(affordance.get("family"), "SeedCatalog.causalAffordance.family")
                require_text(affordance.get("operatorId"), "SeedCatalog.causalAffordance.operatorId")
                require_text(affordance.get("changedPath"), "SeedCatalog.causalAffordance.changedPath")
                values = affordance.get("siblingValues")
                if not isinstance(values, dict) or set(values) != set(ROOT_SIBLING_ROLES):
                    raise CascadeError("Seed causal affordance must define all four siblingValues")
                for role in ROOT_SIBLING_ROLES:
                    require_text(values[role], f"SeedCatalog.causalAffordance.siblingValues.{role}")
                families.add(family)
            if families != {"semantic", "delivery", "turn_taking"}:
                raise CascadeError("Every v2 seed must factor semantic, delivery, and turn_taking operators")
    if len(seed_ids) != len(set(seed_ids)):
        raise CascadeError("Seed catalog contains duplicate seed IDs")
    if request["coverageTarget"]["candidateTopics"] != len(seeds):
        raise CascadeError("candidateTopics must equal the request-bound seed catalog size")
    actual_hash = content_hash(catalog)
    expected_hash = declared_catalog_hash(request)
    if expected_hash is None or actual_hash != expected_hash:
        raise CascadeError("Seed catalog hash does not match the request-bound catalog hash")
    assert_no_target_leak(catalog)
    return actual_hash


def load_bound_seed_catalog(
    request: dict[str, Any],
    request_path: Path,
    repository_root: Path | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    reference = request.get("seedCatalog")
    if reference is None:
        return None, None
    if isinstance(reference, dict):
        catalog = reference
    else:
        raw = Path(reference)
        candidates = [raw] if raw.is_absolute() else [request_path.parent / raw]
        if repository_root is not None and not raw.is_absolute():
            candidates.append(repository_root / raw)
        if not raw.is_absolute():
            candidates.append(Path.cwd() / raw)
        catalog_path = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if catalog_path is None:
            raise CascadeError(f"Cannot resolve request-bound seed catalog {reference!r}")
        catalog = load_json(catalog_path)
    return catalog, validate_seed_catalog(catalog, request)


def prepare_run_identity(
    output_root: Path,
    request: dict[str, Any],
    catalog: dict[str, Any] | None,
    catalog_hash: str | None,
    resume: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    request_copy = output_root / "request.json"
    incoming_request_hash = content_hash(request)
    if request_copy.exists():
        stored_request_hash = content_hash(load_json(request_copy))
        if stored_request_hash != incoming_request_hash:
            raise CascadeError("Cannot resume: request hash differs from the stored cascade request")
        if not resume:
            raise CascadeError("Output root already has a request; use --resume or choose a new root")
    else:
        write_json(request_copy, request)
    catalog_copy = output_root / "seed_catalog.json"
    if catalog is not None:
        if catalog_hash != content_hash(catalog):
            raise CascadeError("Cannot initialize cascade with an unbound seed catalog")
        if catalog_copy.exists() and content_hash(load_json(catalog_copy)) != catalog_hash:
            raise CascadeError("Cannot resume: seed catalog hash differs from the stored catalog")
        if not catalog_copy.exists():
            write_json(catalog_copy, catalog)
    elif catalog_copy.exists():
        raise CascadeError("Cannot resume: stored cascade has a catalog but the request does not")
    manifest_path = output_root / "run_manifest.json"
    if resume and manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest.get("requestHash") != incoming_request_hash:
            raise CascadeError("Cannot resume: run manifest request hash differs")
        stored_catalog_hash = manifest.get("catalogHash", manifest.get("seedCatalogHash"))
        if stored_catalog_hash != catalog_hash:
            raise CascadeError("Cannot resume: run manifest catalog hash differs")


def catalog_seed_ids(catalog: dict[str, Any]) -> set[str]:
    return {str(seed["id"]) for seed in catalog["seeds"]}


def validate_topic_card(card: dict[str, Any], seed_revision: str, source_seed_ids: set[str] | None = None) -> None:
    require_keys(card, {
        "schema", "topicId", "seedRevision", "domain", "interactionModes",
        "registerRange", "safeStakes", "forbiddenPatterns", "diversityTags",
    }, "TopicCardV1")
    if card["schema"] not in {"personaplex.topic-card.v1", "personaplex.topic-card.v2"}:
        raise CascadeError("Unsupported topic-card schema")
    require_identifier(card["topicId"], "topicId")
    if card["seedRevision"] != seed_revision:
        raise CascadeError("Topic card seedRevision does not match request")
    if source_seed_ids is not None:
        require_keys(card, {"sourceSeedId"}, "TopicCardV1")
        if card["sourceSeedId"] not in source_seed_ids:
            raise CascadeError("Topic card sourceSeedId is not in the request-bound catalog")
    elif "sourceSeedId" in card:
        require_text(card["sourceSeedId"], "TopicCardV1.sourceSeedId")
    if not isinstance(card["domain"], str) or len(card["domain"].strip()) < 3:
        raise CascadeError("Topic domain must be meaningful text")
    for field in ("interactionModes", "registerRange", "safeStakes", "forbiddenPatterns", "diversityTags"):
        require_string_list(card[field], f"TopicCardV1.{field}")
    if card["schema"] == "personaplex.topic-card.v2":
        affordances = card.get("causalAffordances")
        if not isinstance(affordances, list) or len(affordances) < 3:
            raise CascadeError("TopicCardV2 requires causalAffordances")
        if {item.get("family") for item in affordances if isinstance(item, dict)} != {"semantic", "delivery", "turn_taking"}:
            raise CascadeError("TopicCardV2 must retain all three causal families")
        for item in affordances:
            if not isinstance(item, dict):
                raise CascadeError("TopicCardV2 causalAffordances must be objects")
            if set(item) != {"family", "operatorId", "changedPath"}:
                raise CascadeError("TopicCardV2 causalAffordances permit only family, operatorId, and changedPath")
            require_text(item.get("operatorId"), "TopicCardV2.causalAffordance.operatorId")
            require_text(item.get("changedPath"), "TopicCardV2.causalAffordance.changedPath")
    assert_no_target_leak(card)


def validate_topic_bindings(topics: list[dict[str, Any]], catalog: dict[str, Any]) -> None:
    expected = catalog_seed_ids(catalog)
    actual = [topic.get("sourceSeedId") for topic in topics]
    if len(actual) != len(expected) or set(actual) != expected or len(actual) != len(set(actual)):
        raise CascadeError("Topic cards must bind exactly one topic to every sourceSeedId")


def validate_scenario_contract(contract: dict[str, Any], known_topics: set[str]) -> None:
    require_keys(contract, {
        "schema", "scenarioId", "topicId", "mode", "premise", "participants",
        "startingState", "interactionOpportunity", "allowedToolClasses",
        "disallowedClaims", "scenarioOutcomeSpace", "requiredControlPhenomena",
    }, "ScenarioContractV1")
    if contract["schema"] not in {"personaplex.scenario-contract.v1", "personaplex.scenario-contract.v2"}:
        raise CascadeError("Unsupported scenario-contract schema")
    require_identifier(contract["scenarioId"], "scenarioId")
    if contract["topicId"] not in known_topics:
        raise CascadeError("Scenario references an unknown topic")
    require_text(contract["mode"], "ScenarioContractV1.mode")
    if not isinstance(contract["premise"], str) or len(contract["premise"].strip()) < 12:
        raise CascadeError("Scenario premise is too short")
    participants = contract["participants"]
    if not isinstance(participants, list) or len(participants) < 2:
        raise CascadeError("Scenario requires at least caller and agent participants")
    for participant in participants:
        if not isinstance(participant, dict) or not isinstance(participant.get("role"), str) or not isinstance(participant.get("knowledge"), str):
            raise CascadeError("Every scenario participant requires role and knowledge")
    state = contract["startingState"]
    if not isinstance(state, dict):
        raise CascadeError("Scenario startingState must be an object")
    for field in ("knownFacts", "uncertainty", "policyConstraints"):
        require_string_list(state.get(field), f"ScenarioContractV1.startingState.{field}")
    for field in ("interactionOpportunity", "allowedToolClasses", "disallowedClaims", "scenarioOutcomeSpace", "requiredControlPhenomena"):
        require_string_list(contract[field], f"ScenarioContractV1.{field}")
    assert_no_target_leak(contract)


def validate_unique_scenario_premises(scenarios: list[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for scenario in scenarios:
        premise = scenario.get("premise")
        if not isinstance(premise, str):
            raise CascadeError("Scenario premise must be typed text before duplicate validation")
        if premise in seen:
            raise CascadeError(f"Scenario premises must be exact-duplicate-free: {seen[premise]} and {scenario.get('scenarioId')}")
        seen[premise] = str(scenario.get("scenarioId"))


def _transition_parts(pivot: dict[str, Any], label: str) -> tuple[str, Any, Any]:
    field = pivot.get("field", pivot.get("statePath"))
    if not isinstance(field, str) or not field.strip():
        raise CascadeError(f"{label} requires a typed field or statePath")
    before_present = next((key for key in ("from", "before", "priorState") if key in pivot), None)
    after_present = next((key for key in ("to", "after", "nextState") if key in pivot), None)
    if before_present is None or after_present is None:
        raise CascadeError(f"{label} requires typed before/from and after/to states")
    before = pivot[before_present]
    after = pivot[after_present]
    if canonical_json(before) == canonical_json(after):
        raise CascadeError(f"{label} must change state")
    return field, before, after


def typed_pivot(seed: dict[str, Any]) -> dict[str, Any] | None:
    pivot = seed.get("typedPivot", seed.get("causalStateTransition"))
    if pivot is None:
        return None
    if not isinstance(pivot, dict):
        raise CascadeError("Typed causal pivot must be an object")
    _transition_parts(pivot, "typedPivot")
    return pivot


def _validate_typed_trajectory(seed: dict[str, Any]) -> None:
    require_text(seed.get("causalAxis"), "TrajectorySeed.causalAxis")
    require_text(seed.get("interventionFamily"), "TrajectorySeed.interventionFamily")
    pivot = typed_pivot(seed)
    if pivot is None:
        raise CascadeError("Typed trajectory requires typedPivot or causalStateTransition")
    posture = seed.get("postureTransition")
    if not isinstance(posture, dict) or "from" not in posture or "to" not in posture:
        raise CascadeError("Typed trajectory postureTransition requires from and to")
    if canonical_json(posture["from"]) == canonical_json(posture["to"]):
        raise CascadeError("Typed postureTransition must change posture")
    require_text(seed.get("evidenceSource"), "TrajectorySeed.evidenceSource")
    require_text(seed.get("outcomeRoute"), "TrajectorySeed.outcomeRoute")
    events = seed.get("duplexEvents")
    if not isinstance(events, list) or not events:
        raise CascadeError("Typed trajectory requires at least one duplex event")
    for event in events:
        if not isinstance(event, dict):
            raise CascadeError("Typed duplex events must be objects")
        require_text(event.get("eventType"), "TrajectorySeed.duplexEvents.eventType")


def validate_trajectory_seed(seed: dict[str, Any], known_scenarios: set[str], require_typed: bool = False) -> None:
    require_keys(seed, {
        "schema", "trajectoryId", "scenarioId", "conversationLength", "pace",
        "openingStyle", "closingStyle", "voicePairPolicy", "interactionArc",
        "duplexEvents", "postureArc", "counterfactualPivotOrdinal", "controlPhenomena",
    }, "TrajectorySeedV1")
    if seed["schema"] not in {"personaplex.trajectory-seed.v1", "personaplex.trajectory-seed.v2"}:
        raise CascadeError("Unsupported trajectory-seed schema")
    require_identifier(seed["trajectoryId"], "trajectoryId")
    if seed["scenarioId"] not in known_scenarios:
        raise CascadeError("Trajectory references an unknown scenario")
    length = seed["conversationLength"]
    if not isinstance(length, dict) or not all(isinstance(length.get(key), int) for key in ("targetTurns", "min", "max")):
        raise CascadeError("Trajectory conversationLength is invalid")
    if not (4 <= length["min"] <= length["targetTurns"] <= length["max"] <= 48):
        raise CascadeError("Trajectory conversationLength bounds are invalid")
    pivot_ordinal = seed["counterfactualPivotOrdinal"]
    if not isinstance(pivot_ordinal, int) or not 1 <= pivot_ordinal < length["targetTurns"]:
        raise CascadeError("Counterfactual pivot must occur before the final planned turn")
    if seed["voicePairPolicy"] != "distinct_approved_references":
        raise CascadeError("Trajectory must require distinct approved voice references")
    for field in ("pace", "openingStyle", "closingStyle"):
        require_text(seed[field], f"TrajectorySeedV1.{field}")
    for field in ("interactionArc", "postureArc", "controlPhenomena"):
        require_string_list(seed[field], f"TrajectorySeedV1.{field}")
    if not isinstance(seed["duplexEvents"], list):
        raise CascadeError("Trajectory duplexEvents must be an array")
    v4_fields = {
        "semanticStateArc", "controlRevisionSchedule", "terminationContract",
        "negativeControlCoverage", "causalAxis",
    }
    # causalAxis is part of the typed causal contract as well as the older v4
    # envelope.  It cannot, by itself, select v4 validation.
    v4_envelope_fields = v4_fields - {"causalAxis"}
    if v4_envelope_fields.intersection(seed):
        missing = v4_fields - set(seed)
        if missing:
            raise CascadeError(f"V4 trajectory is missing fields: {sorted(missing)}")
        if not isinstance(seed["semanticStateArc"], list) or not seed["semanticStateArc"]:
            raise CascadeError("V4 semanticStateArc must be nonempty")
        if not isinstance(seed["controlRevisionSchedule"], list) or not seed["controlRevisionSchedule"]:
            raise CascadeError("V4 controlRevisionSchedule must be nonempty")
        for revision in seed["controlRevisionSchedule"]:
            if (
                not isinstance(revision, dict)
                or not isinstance(revision.get("targetOrdinal"), int)
                or revision.get("availableBeforeTarget") is not True
                or not isinstance(revision.get("source"), str)
            ):
                raise CascadeError("Every v4 revision must be sourced and available before its target")
        termination = seed["terminationContract"]
        if (
            not isinstance(termination, dict)
            or termination.get("decisionSource") != "model"
            or termination.get("action") != "end_call_tool"
            or termination.get("deterministicPhrase") is not False
        ):
            raise CascadeError("V4 termination must be a model-selected end_call tool action")
        require_string_list(seed["negativeControlCoverage"], "TrajectorySeedV1.negativeControlCoverage")
        require_text(seed["causalAxis"], "TrajectorySeedV1.causalAxis")
    if seed["schema"] == "personaplex.trajectory-seed.v2":
        validate_v4_trajectory_seed(seed)
        if seed["interventionFamily"] not in {"semantic", "delivery", "turn_taking"}:
            raise CascadeError("TrajectorySeedV2 interventionFamily is unsupported")
        if tuple(seed["negativeControlCoverage"]) != (
            "paired_wrong_branch", "stale_revision", "null_control",
        ):
            raise CascadeError("TrajectorySeedV2 requires the ordered hard-negative contract")
        for event in seed["duplexEvents"]:
            if (
                not isinstance(event.get("targetOrdinal"), int)
                or not isinstance(event.get("offsetMs"), int)
            ):
                raise CascadeError("TrajectorySeedV2 duplex events require targetOrdinal and offsetMs")
        for revision in seed["controlRevisionSchedule"]:
            if not isinstance(revision.get("controlRevision"), int) or revision["controlRevision"] < 1:
                raise CascadeError("TrajectorySeedV2 revisions require a positive controlRevision")
        expected_targets = list(range(1, seed["conversationLength"]["targetTurns"] // 2 + 1))
        actual_targets = [revision["targetOrdinal"] for revision in seed["controlRevisionSchedule"]]
        if actual_targets != expected_targets:
            raise CascadeError("TrajectorySeedV2 requires one ordered control revision before every agent target")
        revision_ids = [revision["controlRevision"] for revision in seed["controlRevisionSchedule"]]
        if revision_ids != sorted(set(revision_ids)):
            raise CascadeError("TrajectorySeedV2 controlRevision values must be unique and increasing")
    root_fields = {"typedPivot", "causalStateTransition", "interventionFamily", "postureTransition", "evidenceSource", "outcomeRoute"}
    if require_typed or root_fields.intersection(seed):
        _validate_typed_trajectory(seed)
    assert_no_target_leak(seed)


def validate_v4_trajectory_seed(seed: dict[str, Any]) -> None:
    required = {
        "semanticStateArc", "controlRevisionSchedule", "terminationContract",
        "negativeControlCoverage", "causalAxis",
    }
    missing = required - set(seed)
    if missing:
        raise CascadeError(f"semantic-control-v4 trajectory is missing fields: {sorted(missing)}")


def causal_transition_signature(seed: dict[str, Any]) -> str | None:
    pivot = typed_pivot(seed)
    if pivot is None:
        return None
    field, before, after = _transition_parts(pivot, "typedPivot")
    return canonical_json({
        "scenarioId": seed.get("scenarioId"),
        "causalAxis": seed.get("causalAxis"),
        "interventionFamily": seed.get("interventionFamily"),
        "field": field,
        "from": before,
        "to": after,
    })


def validate_unique_causal_signatures(trajectories: list[dict[str, Any]], require_typed: bool = False) -> None:
    seen: dict[str, str] = {}
    for trajectory in trajectories:
        signature = causal_transition_signature(trajectory)
        if signature is None:
            if require_typed:
                raise CascadeError("Every trajectory requires a typed causal state-transition signature")
            continue
        if signature in seen:
            raise CascadeError(
                f"Duplicate causal state-transition signature: {seen[signature]} and {trajectory.get('trajectoryId')}"
            )
        seen[signature] = str(trajectory.get("trajectoryId"))


def _branches(group: dict[str, Any]) -> list[dict[str, Any]]:
    values = group.get("branches", group.get("siblings"))
    if not isinstance(values, list) or not all(isinstance(item, dict) for item in values):
        raise CascadeError("Counterfactual group branches/siblings must be an object array")
    return values


def _branch_role(branch: dict[str, Any]) -> str:
    role = branch.get("branchId", branch.get("siblingRole"))
    return str(role or "")


def validate_pair_spec(
    pair: dict[str, Any],
    known_trajectories: set[str],
    request: dict[str, Any] | None = None,
    trajectory: dict[str, Any] | None = None,
) -> None:
    require_keys(pair, {"schema", "groupId", "trajectoryId", "pivotOrdinal", "commonContextHash"}, "CounterfactualGroupSpec")
    if pair["schema"] not in SUPPORTED_GROUP_SCHEMAS:
        raise CascadeError("Unsupported counterfactual-group schema")
    if (
        request is not None
        and request.get("strategyVersion") == "semantic-control-v5"
        and pair["schema"] != "personaplex.counterfactual-sibling-group-spec.v2"
    ):
        raise CascadeError("semantic-control-v5 requires the v2 sibling-group schema")
    if pair["trajectoryId"] not in known_trajectories:
        raise CascadeError("Group references an unknown trajectory")
    if not isinstance(pair["groupId"], str) or len(pair["groupId"]) < 8:
        raise CascadeError("Counterfactual groupId is invalid")
    if not is_content_hash(pair["commonContextHash"]):
        raise CascadeError("Group commonContextHash must be content-addressed")
    branches = _branches(pair)
    roles = tuple(_branch_role(branch) for branch in branches)
    if len(branches) == 2:
        if set(roles) != set(LEGACY_SIBLING_ROLES):
            raise CascadeError("Pair branch IDs must be available and constrained")
    elif len(branches) == 4:
        expected = request_sibling_roles(request) if request is not None else ROOT_SIBLING_ROLES
        if set(roles) != set(expected) or len(set(roles)) != 4:
            raise CascadeError("Four-sibling group roles do not match the request")
        family = require_text(pair.get("interventionFamily"), "CounterfactualGroup.interventionFamily")
        pivot = pair.get("typedPivot")
        if not isinstance(pivot, dict):
            raise CascadeError("Four-sibling group requires a typedPivot")
        pivot_field, pivot_from, _ = _transition_parts(pivot, "CounterfactualGroup.typedPivot")
        if trajectory is not None:
            if family != trajectory.get("interventionFamily"):
                raise CascadeError("Group interventionFamily does not match its trajectory")
            trajectory_pivot = typed_pivot(trajectory)
            if trajectory_pivot is None or canonical_json(trajectory_pivot) != canonical_json(pivot):
                raise CascadeError("Group typedPivot does not match its trajectory")
    else:
        raise CascadeError("Counterfactual group must contain exactly two or four siblings")
    deltas: list[dict[str, Any]] = []
    for branch in branches:
        delta = branch.get("controlDelta")
        if not isinstance(delta, dict) or not isinstance(delta.get("field"), str) or "from" not in delta or "to" not in delta:
            raise CascadeError("Every sibling requires a typed controlDelta")
        if not isinstance(branch.get("evidenceUpdate"), dict):
            raise CascadeError("Every sibling requires a typed evidenceUpdate")
        if len(branches) == 4 and "interventionFamily" in branch and branch["interventionFamily"] != pair["interventionFamily"]:
            raise CascadeError("A sibling cannot declare a different intervention family")
        deltas.append(delta)
    if any(delta["field"] != deltas[0]["field"] or canonical_json(delta["from"]) != canonical_json(deltas[0]["from"]) for delta in deltas[1:]):
        raise CascadeError("All siblings must vary one common typed control field from one common state")
    if len({canonical_json(delta["to"]) for delta in deltas}) != len(deltas):
        raise CascadeError("Counterfactual siblings must have distinct target states")
    if len(branches) == 4:
        if deltas[0]["field"] != pivot_field or canonical_json(deltas[0]["from"]) != canonical_json(pivot_from):
            raise CascadeError("Sibling control deltas must bind the declared typedPivot")
        if request is not None and request.get("strategyVersion") == "semantic-control-v5":
            for branch, delta in zip(branches, deltas):
                if "controlValue" not in branch or canonical_json(branch["controlValue"]) != canonical_json(delta["to"]):
                    raise CascadeError("Every v5 sibling controlValue must equal its typed controlDelta.to state")
    if "sharedPrefixPolicy" in pair or any("availabilityTiming" in branch for branch in branches):
        if pair.get("sharedPrefixPolicy") != "native_code_identical_through_pivot":
            raise CascadeError("V4 group sharedPrefixPolicy is invalid")
        for branch in branches:
            timing = branch.get("availabilityTiming")
            if (
                not isinstance(timing, dict)
                or timing.get("availableBeforeTarget") is not True
                or not isinstance(timing.get("controlRevision"), int)
            ):
                raise CascadeError("V4 siblings require causal control availability timing")
            require_string_list(branch.get("negativeControls"), "CounterfactualGroup.negativeControls")
            require_string_list(branch.get("semanticAssertions"), "CounterfactualGroup.semanticAssertions")
    assert_no_target_leak(pair)


def validate_v4_pair_spec(pair: dict[str, Any]) -> None:
    if pair.get("sharedPrefixPolicy") != "native_code_identical_through_pivot":
        raise CascadeError("semantic-control-v4 group lacks the native shared-prefix policy")
    branches = _branches(pair)
    if len(branches) not in {2, 4}:
        raise CascadeError("semantic-control-v4 group requires two or four siblings")
    for branch in branches:
        timing = branch.get("availabilityTiming")
        if (
            not isinstance(timing, dict)
            or timing.get("availableBeforeTarget") is not True
            or not isinstance(timing.get("controlRevision"), int)
            or timing["controlRevision"] < 1
        ):
            raise CascadeError("semantic-control-v4 sibling lacks causal availability timing")
        negative = set(require_string_list(branch.get("negativeControls"), "negativeControls"))
        if not {"paired_wrong_branch", "stale_revision", "null_control"}.issubset(negative):
            raise CascadeError("semantic-control-v4 sibling lacks required hard negatives")
        require_string_list(branch.get("semanticAssertions"), "semanticAssertions")


@dataclass(frozen=True)
class PlannerConfig:
    endpoint: str
    model: str
    api_key: str
    timeout_seconds: int = 180
    max_tokens: int = 4096
    temperature: float = 0.85


def normalize_planner_endpoints(value: str) -> tuple[str, ...]:
    """Return ordered, de-duplicated OpenAI-compatible endpoint URLs."""

    if not isinstance(value, str):
        raise CascadeError("Planner endpoint must be text")
    endpoints: list[str] = []
    for candidate in value.split(","):
        endpoint = candidate.strip().rstrip("/")
        if not endpoint:
            continue
        if not endpoint.startswith(("http://", "https://")):
            raise CascadeError("Planner endpoints must use http:// or https://")
        if endpoint not in endpoints:
            endpoints.append(endpoint)
    if not endpoints:
        raise CascadeError("At least one planner endpoint is required")
    return tuple(endpoints)


class JsonOnlyPlanner:
    """OpenAI-compatible planner with no parsing fallback or semantic shortcut."""

    _RETRIABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

    def __init__(self, config: PlannerConfig):
        if not config.model:
            raise CascadeError("Planner endpoint and model are required for generative stages")
        self.config = config
        self.endpoints = normalize_planner_endpoints(config.endpoint)
        self._endpoint_lock = Lock()
        self._next_endpoint = 0

    def _endpoint_order(self) -> tuple[str, ...]:
        with self._endpoint_lock:
            start = self._next_endpoint % len(self.endpoints)
            self._next_endpoint += 1
        return self.endpoints[start:] + self.endpoints[:start]

    def call(
        self,
        system: str,
        user: str,
        response_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response_format: dict[str, Any]
        if response_schema is None:
            response_format = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "personaplex_cascade_artifact",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "response_format": response_format,
            "reasoning": {"enabled": False},
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        encoded_payload = json.dumps(payload).encode("utf-8")
        endpoints = self._endpoint_order()
        for attempt, endpoint in enumerate(endpoints, start=1):
            request = urllib.request.Request(
                endpoint,
                data=encoded_payload,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                    response_body = response.read().decode("utf-8")
            except urllib.error.HTTPError as error:
                if error.code in self._RETRIABLE_HTTP_STATUS and attempt < len(endpoints):
                    continue
                raise CascadeError(f"Planner inference failed with HTTP {error.code}") from error
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
                if attempt < len(endpoints):
                    continue
                raise CascadeError(f"Planner inference transport failed after {attempt} endpoint attempt(s): {error}") from error

            # A reachable endpoint that returns malformed protocol or semantic output is
            # authoritative. Never hide that model defect by retrying another endpoint.
            try:
                envelope = json.loads(response_body)
            except json.JSONDecodeError as error:
                raise CascadeError("Planner endpoint returned a malformed JSON envelope") from error
            try:
                choice = envelope["choices"][0]
                content = choice["message"]["content"]
            except (KeyError, IndexError, TypeError) as error:
                raise CascadeError("Planner response lacks choices[0].message.content") from error
            if choice.get("finish_reason") == "length":
                raise CascadeError("Planner output was truncated at the configured generation limit")
            if not isinstance(content, str):
                raise CascadeError("Planner response content must be a raw JSON string")
            try:
                result = json.loads(content)
            except json.JSONDecodeError as error:
                raise CascadeError("Planner returned non-JSON content; no text recovery is permitted") from error
            if not isinstance(result, dict):
                raise CascadeError("Planner result must be a JSON object")
            return result
        raise CascadeError("Planner inference exhausted all configured endpoints")


PLANNER_SYSTEM = """You are a planning component for a controlled conversational-audio training corpus.
Reason silently. Return one raw JSON object only, with exactly the requested top-level key.
Do not include markdown, prose, canonical target responses, target transcripts, target audio,
spoken dialogue, names, contact information, credentials, or placeholder text. Do not make a
semantic certification claim; create planning artifacts only."""

SCENARIO_DIVERSIFIERS = (
    "cooperative caller with a straightforward but non-scheduling goal",
    "skeptical caller who needs evidence before conditional agreement",
    "resistant caller who challenges one unsupported assumption",
    "caller who corrects a material fact midway through the exchange",
    "ambiguous request that requires clarification before action",
    "tool result that confirms the initially expected state",
    "tool result that disproves the initially expected state",
    "policy boundary that removes a previously plausible option",
    "newer evidence that supersedes a stale plan",
    "brief interruption that invalidates queued agent audio",
    "longer barge-in followed by acknowledgement and repair",
    "refusal that must be respected without persuasion",
    "handoff that becomes necessary only after new evidence",
    "conditional compliance with one explicit unresolved concern",
    "low-stakes casual exchange with a natural topic shift",
    "technical explanation calibrated to uncertainty",
    "service recovery after an earlier misunderstanding",
    "multi-step decision with one tool-dependent branch",
    "accessibility preference that changes delivery but not facts",
    "completed objective followed by a model-selected end-call action",
)

TRAJECTORY_DIVERSIFIERS = (
    "short cooperative exchange with concise turns",
    "skeptical medium exchange with one clarification",
    "resistant longer exchange with de-escalation",
    "caller correction followed by explicit repair",
    "conditional compliance after a tool result",
    "respectful refusal without repeated persuasion",
    "handoff after an unavailable resolution path",
    "barge-in that cancels outgoing audio and changes the next response",
    "brief natural overlap and recovery without losing the goal",
    "open-ended conversational close selected by the model and end_call tool",
)

CAUSAL_OPERATOR_ROTATION = (
    {"interventionFamily": "semantic", "causalAxis": "evidence_status", "changedPath": "state.evidence.status"},
    {"interventionFamily": "delivery", "causalAxis": "next_goal_route", "changedPath": "plan.nextGoal"},
    {"interventionFamily": "turn_taking", "causalAxis": "interruption_recovery", "changedPath": "turnTaking.eventType"},
)


def _set_schema_const(
    definition: dict[str, Any],
    definitions: dict[str, Any],
    dotted_path: str,
    value: Any,
) -> None:
    node: dict[str, Any] = definition
    parts = dotted_path.split(".")
    for index, part in enumerate(parts):
        while "$ref" in node:
            reference = node["$ref"]
            prefix = "#/$defs/"
            if not isinstance(reference, str) or not reference.startswith(prefix):
                raise CascadeError(f"Unsupported schema reference while binding {dotted_path}")
            resolved = definitions.get(reference.removeprefix(prefix))
            if not isinstance(resolved, dict):
                raise CascadeError(f"Unresolved schema reference while binding {dotted_path}")
            node = resolved
        properties = node.get("properties")
        if not isinstance(properties, dict) or part not in properties:
            raise CascadeError(f"Schema property {dotted_path} cannot be bound")
        child = properties[part]
        if not isinstance(child, dict):
            raise CascadeError(f"Schema property {dotted_path} is not typed")
        if index == len(parts) - 1:
            child["const"] = value
        else:
            node = child


def _v5_response_schema(
    expected_key: str,
    expected_count: int,
    definition_name: str,
    schema_constants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema_path = Path(__file__).resolve().parents[1] / "schemas" / "diverse_cascade_artifacts.v2.schema.json"
    artifact_schema = load_json(schema_path)
    definitions = artifact_schema.get("$defs")
    if not isinstance(definitions, dict) or definition_name not in definitions:
        raise CascadeError(f"Missing v2 artifact schema definition {definition_name}")
    definition = definitions[definition_name]
    if not isinstance(definition, dict):
        raise CascadeError(f"Invalid v2 artifact schema definition {definition_name}")
    if definition_name in {"scenarioContract", "trajectorySeed"}:
        string_list = definitions.get("stringList")
        if not isinstance(string_list, dict) or not isinstance(string_list.get("items"), dict):
            raise CascadeError("V2 artifact schema lacks its semantic string-list definition")
        string_list["maxItems"] = 4
    for dotted_path, value in (schema_constants or {}).items():
        _set_schema_const(definition, definitions, dotted_path, value)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [expected_key],
        "properties": {
            expected_key: {
                "type": "array",
                "minItems": expected_count,
                "maxItems": expected_count,
                "items": {"$ref": f"#/$defs/{definition_name}"},
            },
        },
        "$defs": definitions,
    }


def _model_list(
    planner: JsonOnlyPlanner,
    expected_key: str,
    user_prompt: str,
    expected_count: int,
    *,
    schema_definition: str | None = None,
    schema_constants: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    response_schema = (
        _v5_response_schema(
            expected_key,
            expected_count,
            schema_definition,
            schema_constants,
        )
        if schema_definition is not None
        else None
    )
    response = (
        planner.call(PLANNER_SYSTEM, user_prompt)
        if response_schema is None
        else planner.call(PLANNER_SYSTEM, user_prompt, response_schema)
    )
    if set(response) != {expected_key} or not isinstance(response[expected_key], list):
        raise CascadeError(f"Planner must return only a {expected_key} array")
    values = response[expected_key]
    if len(values) != expected_count or not all(isinstance(value, dict) for value in values):
        raise CascadeError(f"Planner returned {len(values)} {expected_key}; expected {expected_count}")
    return values


def _validated_model_one(
    planner: JsonOnlyPlanner,
    expected_key: str,
    prompt: dict[str, Any],
    validator: Callable[[dict[str, Any]], None],
    *,
    attempts: int = 6,
    schema_definition: str | None = None,
    schema_constants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Regenerate only one structurally invalid artifact, never a completed batch."""

    last_error: CascadeError | None = None
    for attempt in range(1, attempts + 1):
        request_prompt = dict(prompt)
        request_prompt["generationAttempt"] = attempt
        request_prompt["responseRule"] = (
            f"Return one JSON object with exactly one key named {expected_key}; "
            "its value must be a one-item array containing the artifact."
        )
        if last_error is not None:
            request_prompt["repairConstraint"] = {
                "priorValidationFailure": str(last_error),
                "regenerateFreshArtifact": True,
            }
        try:
            candidate = _model_list(
                planner,
                expected_key,
                canonical_json(request_prompt),
                1,
                schema_definition=schema_definition,
                schema_constants=schema_constants,
            )[0]
            validator(candidate)
            return candidate
        except CascadeError as error:
            last_error = error
    raise CascadeError(
        f"Planner failed {expected_key} structural validation after {attempts} attempts: {last_error}"
    )


def plan_topics(
    planner: JsonOnlyPlanner,
    request: dict[str, Any],
    seed_catalog: dict[str, Any] | None = None,
    max_workers: int = 1,
    existing_records: list[dict[str, Any]] | None = None,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    count = request["coverageTarget"]["candidateTopics"]
    source_seeds = seed_catalog["seeds"] if seed_catalog is not None else None
    v5 = request.get("strategyVersion") == "semantic-control-v5"
    required_fields = [
        "schema", "topicId", "seedRevision", "domain", "interactionModes",
        "registerRange", "safeStakes", "forbiddenPatterns", "diversityTags",
    ]
    requirements = [
        "Cover varied safe interaction modes without target dialogue.",
        "Avoid repetitive scheduling, company greetings, identity collection, political persuasion, and target dialogue.",
        "Use unique lowercase structured topicId values and copy seedRevision exactly.",
    ]
    if source_seeds is not None:
        required_fields.append("sourceSeedId")
        requirements.append("Create exactly one topic for every supplied source seed and copy each sourceSeedId exactly once.")
    if v5:
        required_fields.append("causalAffordances")
        requirements.append("Copy each source seed's three causal affordances as family, operatorId, and changedPath only.")
    prompt = {
        "task": "Generate diverse broad topic cards, not dialogue.",
        "requiredTopLevelKey": "topicCards",
        "responseShape": {"topicCards": ["exactly the requested number of topic-card objects"]},
        "requestedCount": count,
        "seedRevision": request["seedRevision"],
        "sourceSeeds": source_seeds,
        "legacySeedIdeas": request.get("seedIdeas"),
        "topicConstraints": request.get("topicConstraints", {}),
        "requiredSchema": "personaplex.topic-card.v2" if v5 else "personaplex.topic-card.v1",
        "requiredFields": required_fields,
        "requirements": requirements,
    }
    source_ids = catalog_seed_ids(seed_catalog) if seed_catalog is not None else None
    existing = list(existing_records or [])
    for card in existing:
        validate_topic_card(card, request["seedRevision"], source_ids)
    if v5 and source_seeds is not None:
        def generate_topic(source_seed: dict[str, Any]) -> dict[str, Any]:
            expected_id = f"topic_{str(source_seed['id']).casefold()}"
            item_prompt = dict(prompt)
            item_prompt.update({
                "requestedCount": 1,
                "sourceSeeds": [source_seed],
                "requiredTopicId": expected_id,
                "legacySeedIdeas": None,
            })

            def validate(card: dict[str, Any]) -> None:
                validate_topic_card(card, request["seedRevision"], source_ids)
                if card.get("sourceSeedId") != source_seed["id"] or card.get("topicId") != expected_id:
                    raise CascadeError("Topic card identity does not match its assigned source seed")

            return _validated_model_one(
                planner,
                "topicCards",
                item_prompt,
                validate,
                schema_definition="topicCard",
                schema_constants={
                    "topicId": expected_id,
                    "sourceSeedId": source_seed["id"],
                    "seedRevision": request["seedRevision"],
                    "causalAffordances": [
                        {
                            "family": affordance["family"],
                            "operatorId": affordance["operatorId"],
                            "changedPath": affordance["changedPath"],
                        }
                        for affordance in source_seed["causalAffordances"]
                    ],
                },
            )

        expected_ids = {f"topic_{str(seed['id']).casefold()}" for seed in source_seeds}
        existing_ids = {card["topicId"] for card in existing}
        if not existing_ids.issubset(expected_ids):
            raise CascadeError("Checkpointed topic cards contain identities outside the bound seed catalog")
        pending_seeds = [
            seed for seed in source_seeds
            if f"topic_{str(seed['id']).casefold()}" not in existing_ids
        ]
        generated = parallel_map(
            pending_seeds,
            generate_topic,
            max_workers,
            on_result=on_record,
        )
        cards = existing + generated
    else:
        if existing:
            if len(existing) != count:
                raise CascadeError("Legacy topic generation cannot resume from a partial batch")
            cards = existing
        else:
            cards = _model_list(planner, "topicCards", canonical_json(prompt), count)
            if on_record is not None:
                for card in cards:
                    on_record(card)
    for card in cards:
        validate_topic_card(card, request["seedRevision"], source_ids)
    if len({card["topicId"] for card in cards}) != len(cards):
        raise CascadeError("Planner generated duplicate topic IDs")
    if seed_catalog is not None:
        validate_topic_bindings(cards, seed_catalog)
    return sorted(cards, key=lambda card: card["topicId"])


def plan_scenarios(
    planner: JsonOnlyPlanner,
    topic: dict[str, Any],
    request: dict[str, Any],
    existing_records: list[dict[str, Any]] | None = None,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    count = request["coverageTarget"]["scenariosPerTopic"]
    prompt = {
        "task": "Generate scenario contracts that supply context but no dialogue targets.",
        "requiredTopLevelKey": "scenarioContracts",
        "responseShape": {"scenarioContracts": ["exactly the requested number of scenario-contract objects"]},
        "topicCard": topic,
        "requestedCount": count,
        "requiredSchema": "personaplex.scenario-contract.v2" if request.get("strategyVersion") == "semantic-control-v5" else "personaplex.scenario-contract.v1",
        "requiredFields": [
            "schema", "scenarioId", "topicId", "mode", "premise", "participants",
            "startingState", "interactionOpportunity", "allowedToolClasses",
            "disallowedClaims", "scenarioOutcomeSpace", "requiredControlPhenomena",
        ],
        "requirements": [
            "Every scenario must be safe, non-identifying, and support multiple interaction arcs.",
            "Include facts, uncertainty, and policy boundaries without canonical responses.",
            "Every premise must be exactly distinct from every other supplied premise.",
            "Set topicId to the supplied topic and use unique lowercase structured scenarioId values.",
            "Keep every field concise: premise at most three sentences and list entries at most one sentence.",
        ],
    }
    existing = list(existing_records or [])
    for scenario in existing:
        validate_scenario_contract(scenario, {topic["topicId"]})
    if request.get("strategyVersion") == "semantic-control-v5":
        scenarios = list(existing)
        expected_ids = {
            f"scenario_{topic['topicId']}_{index + 1:02d}"
            for index in range(count)
        }
        existing_ids = {scenario["scenarioId"] for scenario in existing}
        if not existing_ids.issubset(expected_ids):
            raise CascadeError("Checkpointed scenarios contain identities outside the topic cascade")
        prior_premises = [scenario["premise"] for scenario in existing]
        for index in range(count):
            expected_id = f"scenario_{topic['topicId']}_{index + 1:02d}"
            if expected_id in existing_ids:
                continue
            item_prompt = dict(prompt)
            item_prompt.update({
                "requestedCount": 1,
                "scenarioOrdinal": index + 1,
                "requiredScenarioId": expected_id,
                "structuralDiversifier": SCENARIO_DIVERSIFIERS[index % len(SCENARIO_DIVERSIFIERS)],
                "priorPremisesThatMustNotBeRepeated": prior_premises,
            })

            def validate(candidate: dict[str, Any]) -> None:
                validate_scenario_contract(candidate, {topic["topicId"]})
                if candidate.get("scenarioId") != expected_id:
                    raise CascadeError("Scenario identity does not match its assigned cascade ordinal")
                if candidate.get("premise") in prior_premises:
                    raise CascadeError("Scenario premise repeats a previously admitted premise")
                if on_record is not None:
                    on_record(candidate)

            scenario = _validated_model_one(
                planner,
                "scenarioContracts",
                item_prompt,
                validate,
                schema_definition="scenarioContract",
                schema_constants={
                    "scenarioId": expected_id,
                    "topicId": topic["topicId"],
                },
            )
            scenarios.append(scenario)
            prior_premises.append(scenario["premise"])
    else:
        if existing:
            if len(existing) != count:
                raise CascadeError("Legacy scenario generation cannot resume from a partial batch")
            scenarios = existing
        else:
            scenarios = _model_list(planner, "scenarioContracts", canonical_json(prompt), count)
            if on_record is not None:
                for scenario in scenarios:
                    on_record(scenario)
    for scenario in scenarios:
        validate_scenario_contract(scenario, {topic["topicId"]})
    if len({scenario["scenarioId"] for scenario in scenarios}) != len(scenarios):
        raise CascadeError(f"Planner generated duplicate scenario IDs for {topic['topicId']}")
    validate_unique_scenario_premises(scenarios)
    return sorted(scenarios, key=lambda scenario: scenario["scenarioId"])


def plan_trajectories(
    planner: JsonOnlyPlanner,
    scenario: dict[str, Any],
    request: dict[str, Any],
    existing_records: list[dict[str, Any]] | None = None,
    on_record: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    count = request["coverageTarget"]["trajectorySeedsPerScenario"]
    required_fields = [
        "schema", "trajectoryId", "scenarioId", "conversationLength", "pace",
        "openingStyle", "closingStyle", "voicePairPolicy", "interactionArc",
        "duplexEvents", "postureArc", "counterfactualPivotOrdinal", "controlPhenomena",
    ]
    requirements = [
        "Create genuinely distinct call shapes and model-driven endings.",
        "Include varied cooperation, resistance, repair, clarification, refusal, handoff, and recovery.",
        "Include typed duplex event plans across the set.",
        "Each seed requires distinct approved voice references and a pre-final counterfactual pivot.",
        "Set scenarioId exactly and use unique lowercase structured trajectoryId values.",
    ]
    strategy = request.get("strategyVersion")
    if strategy in {"semantic-control-v4", "semantic-control-v5"}:
        required_fields.extend([
            "semanticStateArc", "controlRevisionSchedule", "terminationContract",
            "negativeControlCoverage", "causalAxis",
        ])
        requirements.extend([
            "For every planned agent target, create a typed sourced revision available before that target.",
            "Every revision has a positive controlRevision; do not invent cryptographic hashes.",
            "negativeControlCoverage includes paired_wrong_branch, stale_revision, and null_control.",
            "terminationContract uses model-selected end_call_tool with no deterministic phrase.",
        ])
    require_typed = request_requires_typed_trajectories(request)
    if require_typed:
        required_fields.extend([
            "typedPivot", "interventionFamily", "postureTransition", "evidenceSource", "outcomeRoute",
        ])
        requirements.extend([
            "Declare one interventionFamily and one typedPivot with field, from, and to states.",
            "Use typed postureTransition, evidenceSource, duplex eventType, and outcomeRoute values.",
            "Every causal state-transition signature in this scenario must be distinct.",
        ])
    prompt = {
        "task": "Generate trajectory seeds, not transcripts or target dialogue.",
        "requiredTopLevelKey": "trajectorySeeds",
        "responseShape": {"trajectorySeeds": ["exactly the requested number of trajectory-seed objects"]},
        "scenarioContract": scenario,
        "requestedCount": count,
        "requiredSchema": "personaplex.trajectory-seed.v2" if strategy == "semantic-control-v5" else "personaplex.trajectory-seed.v1",
        "semanticControl": request.get("semanticControl"),
        "causalGroupContract": request.get("causalGroupContract"),
        "requiredFields": list(dict.fromkeys(required_fields)),
        "requirements": requirements,
    }
    existing = list(existing_records or [])
    for seed in existing:
        validate_trajectory_seed(seed, {scenario["scenarioId"]}, require_typed=require_typed)
        if strategy in {"semantic-control-v4", "semantic-control-v5"}:
            validate_v4_trajectory_seed(seed)
    if strategy == "semantic-control-v5":
        seeds = list(existing)
        expected_ids = {
            f"trajectory_{scenario['scenarioId']}_{index + 1:02d}"
            for index in range(count)
        }
        existing_ids = {seed["trajectoryId"] for seed in existing}
        if not existing_ids.issubset(expected_ids):
            raise CascadeError("Checkpointed trajectories contain identities outside the scenario cascade")
        prior_pivots = [dict(typed_pivot(seed) or {}) for seed in existing]
        for index in range(count):
            expected_id = f"trajectory_{scenario['scenarioId']}_{index + 1:02d}"
            if expected_id in existing_ids:
                continue
            operator = CAUSAL_OPERATOR_ROTATION[index % len(CAUSAL_OPERATOR_ROTATION)]
            item_prompt = dict(prompt)
            item_prompt.update({
                "requestedCount": 1,
                "trajectoryOrdinal": index + 1,
                "requiredTrajectoryId": expected_id,
                "trajectoryDiversifier": TRAJECTORY_DIVERSIFIERS[index % len(TRAJECTORY_DIVERSIFIERS)],
                "requiredCausalOperator": operator,
                "priorTypedPivotsThatMustNotBeRepeated": prior_pivots,
                "timingRequirement": "Use millisecond offsets and actual barge-in cancellation events, not labels that merely say interruption.",
            })

            def validate(candidate: dict[str, Any]) -> None:
                validate_trajectory_seed(
                    candidate,
                    {scenario["scenarioId"]},
                    require_typed=True,
                )
                candidate_pivot = typed_pivot(candidate)
                if candidate.get("trajectoryId") != expected_id:
                    raise CascadeError("Trajectory identity does not match its assigned cascade ordinal")
                if candidate.get("interventionFamily") != operator["interventionFamily"]:
                    raise CascadeError("Trajectory interventionFamily does not match its assigned causal operator")
                if candidate.get("causalAxis") != operator["causalAxis"]:
                    raise CascadeError("Trajectory causalAxis does not match its assigned causal operator")
                if candidate_pivot is None or candidate_pivot.get("field") != operator["changedPath"]:
                    raise CascadeError("Trajectory typedPivot field does not match its assigned causal operator")
                if any(canonical_json(candidate_pivot) == canonical_json(prior) for prior in prior_pivots):
                    raise CascadeError("Trajectory typedPivot repeats a prior trajectory in this scenario")

            seed = _validated_model_one(
                planner,
                "trajectorySeeds",
                item_prompt,
                validate,
                schema_definition="trajectorySeed",
                schema_constants={
                    "trajectoryId": expected_id,
                    "scenarioId": scenario["scenarioId"],
                    "interventionFamily": operator["interventionFamily"],
                    "causalAxis": operator["causalAxis"],
                    "typedPivot.field": operator["changedPath"],
                },
            )
            seeds.append(seed)
            prior_pivots.append(dict(typed_pivot(seed) or {}))
            if on_record is not None:
                on_record(seed)
    else:
        if existing:
            if len(existing) != count:
                raise CascadeError("Legacy trajectory generation cannot resume from a partial batch")
            seeds = existing
        else:
            seeds = _model_list(planner, "trajectorySeeds", canonical_json(prompt), count)
            if on_record is not None:
                for seed in seeds:
                    on_record(seed)
    for seed in seeds:
        validate_trajectory_seed(seed, {scenario["scenarioId"]}, require_typed=require_typed)
        if strategy in {"semantic-control-v4", "semantic-control-v5"}:
            validate_v4_trajectory_seed(seed)
    if len({seed["trajectoryId"] for seed in seeds}) != len(seeds):
        raise CascadeError(f"Planner generated duplicate trajectory IDs for {scenario['scenarioId']}")
    validate_unique_causal_signatures(seeds, require_typed=require_typed)
    return sorted(seeds, key=lambda seed: seed["trajectoryId"])


def _stable_rank(request: dict[str, Any], value: Any) -> str:
    salt = request.get("seedCatalogHash", request["seedRevision"])
    return sha256(f"{salt}:{content_hash(request)}:{canonical_json(value)}".encode("utf-8")).hexdigest()


def _dimension_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return canonical_json(value)


def trajectory_balance_dimensions(trajectory: dict[str, Any]) -> dict[str, list[str]]:
    length = trajectory.get("conversationLength")
    if isinstance(length, dict):
        length_value = length.get("profile", length.get("band", length.get("targetTurns")))
    else:
        length_value = length
    style_value = trajectory.get("style", trajectory.get("styleProfile"))
    if style_value is None:
        style_value = {
            "opening": trajectory.get("openingStyle"),
            "closing": trajectory.get("closingStyle"),
            "pace": trajectory.get("pace"),
        }
    events = trajectory.get("duplexEvents")
    event_types: set[str] = set()
    if isinstance(events, list):
        for event in events:
            if isinstance(event, dict) and isinstance(event.get("eventType"), str):
                event_types.add(event["eventType"])
            elif isinstance(event, str):
                event_types.add(event)
    if not event_types:
        event_types.add("none")
    dimensions = {
        "causalAxis": [trajectory.get("causalAxis", "legacy_unspecified")],
        "interventionFamily": [trajectory.get("interventionFamily", "legacy_unspecified")],
        "postureTransition": [trajectory.get("postureTransition", trajectory.get("postureArc", "legacy_unspecified"))],
        "evidenceSource": [trajectory.get("evidenceSource", "legacy_unspecified")],
        "duplexEventType": sorted(event_types),
        "outcomeRoute": [trajectory.get("outcomeRoute", "legacy_unspecified")],
        "conversationLength": [length_value],
        "style": [style_value],
    }
    return {
        axis: [_dimension_value(value) for value in values]
        for axis, values in dimensions.items()
    }


def select_trajectories(
    request: dict[str, Any],
    topics: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    trajectories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary_target, reserve_target = request_selection_counts(request)
    target = primary_target + reserve_target
    if primary_target < len(topics):
        raise CascadeError("Primary groups must allocate at least one group per topic")
    scenario_by_id = {item["scenarioId"]: item for item in scenarios}
    topic_by_id = {item["topicId"]: item for item in topics}
    if len(scenario_by_id) != len(scenarios) or len(topic_by_id) != len(topics):
        raise CascadeError("Selection inputs contain duplicate topic or scenario IDs")
    candidate_by_id: dict[str, dict[str, Any]] = {}
    candidate_topic: dict[str, str] = {}
    for trajectory in trajectories:
        trajectory_id = trajectory.get("trajectoryId")
        scenario = scenario_by_id.get(trajectory.get("scenarioId"))
        if not isinstance(trajectory_id, str) or scenario is None or scenario.get("topicId") not in topic_by_id:
            raise CascadeError("Selection candidate has invalid trajectory/scenario/topic lineage")
        if trajectory_id in candidate_by_id:
            raise CascadeError("Selection candidate trajectory IDs must be unique")
        candidate_by_id[trajectory_id] = trajectory
        candidate_topic[trajectory_id] = scenario["topicId"]
    if target > len(candidate_by_id):
        raise CascadeError("Selection target exceeds available trajectory leaves")

    remaining = dict(candidate_by_id)
    dimension_counts: dict[tuple[str, str], int] = {}
    topic_counts: dict[str, int] = {}
    scenario_counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    tier_counts = {"primary": 0, "reserve": 0}

    def score(candidate: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, int], dict[str, list[str]]]:
        dimensions = trajectory_balance_dimensions(candidate)
        costs = {
            axis: min(dimension_counts.get((axis, value), 0) for value in dimensions[axis])
            for axis in BALANCE_AXES
        }
        topic_id = candidate_topic[candidate["trajectoryId"]]
        ranking = (
            max(costs.values()),
            sum(costs.values()),
            scenario_counts.get(candidate["scenarioId"], 0),
            topic_counts.get(topic_id, 0),
            _stable_rank(request, candidate),
        )
        return ranking, costs, dimensions

    def choose(candidates: list[dict[str, Any]]) -> None:
        if not candidates:
            raise CascadeError("Selection exhausted before satisfying request counts")
        chosen = min(candidates, key=lambda candidate: score(candidate)[0])
        _, costs, dimensions = score(chosen)
        topic_id = candidate_topic[chosen["trajectoryId"]]
        tier = "primary" if len(selected) < primary_target else "reserve"
        tier_counts[tier] += 1
        ordinal = len(selected) + 1
        width = max(4, len(str(target)))
        group_id = f"cascade-{request['requestId']}-{ordinal:0{width}d}"
        topic = topic_by_id[topic_id]
        rank = _stable_rank(request, chosen)
        record = {
            "schema": "personaplex.selected-trajectory.v1",
            "groupId": group_id,
            "topicId": topic_id,
            "scenarioId": chosen["scenarioId"],
            "trajectoryId": chosen["trajectoryId"],
            "sourceSeedId": topic.get("sourceSeedId"),
            "selectionTier": tier,
            "selectionOrdinal": ordinal,
            "tierOrdinal": tier_counts[tier],
            "selectionSeedRevision": request["seedRevision"],
            "balanceDimensions": dimensions,
            "selectionRationale": {
                "algorithm": "typed-balanced-all-leaves-v1",
                "candidatePoolSize": len(candidate_by_id),
                "eligibleAtDecision": len(remaining),
                "balanceCostByAxis": costs,
                "maxBalanceCost": max(costs.values()),
                "aggregateBalanceCost": sum(costs.values()),
                "deterministicRank": rank,
            },
        }
        record["selectionHash"] = content_hash({
            "requestHash": content_hash(request),
            "groupId": group_id,
            "trajectoryId": chosen["trajectoryId"],
            "tier": tier,
            "dimensions": dimensions,
            "rank": rank,
        })
        selected.append(record)
        for axis, values in dimensions.items():
            for value in values:
                dimension_counts[(axis, value)] = dimension_counts.get((axis, value), 0) + 1
        topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
        scenario_counts[chosen["scenarioId"]] = scenario_counts.get(chosen["scenarioId"], 0) + 1
        del remaining[chosen["trajectoryId"]]

    topic_order = sorted(topic_by_id, key=lambda topic_id: _stable_rank(request, {"topicId": topic_id}))
    for topic_id in topic_order:
        choose([candidate for trajectory_id, candidate in remaining.items() if candidate_topic[trajectory_id] == topic_id])
    while len(selected) < target:
        choose(list(remaining.values()))
    if len({row["trajectoryId"] for row in selected}) != target:
        raise CascadeError("Selection must contain the requested number of unique trajectory leaves")
    return selected


def _typed_rejection(record: dict[str, Any]) -> tuple[str, str, str]:
    if record.get("schema") not in SUPPORTED_REJECTION_SCHEMAS:
        raise CascadeError("Rejected-group record has an unsupported schema")
    group_id = require_text(record.get("groupId"), "RejectedGroup.groupId")
    nested = record.get("rejection")
    if nested is not None:
        if not isinstance(nested, dict):
            raise CascadeError("RejectedGroup.rejection must be a typed object")
        stage = nested.get("stage")
        code = nested.get("code")
    else:
        stage = record.get("stage")
        code = record.get("reasonCode")
    require_identifier(stage, "RejectedGroup.stage")
    require_identifier(code, "RejectedGroup.reasonCode")
    assert_no_target_leak(record)
    return group_id, str(stage), str(code)


def validate_rejected_group_records(records: list[dict[str, Any]], known_group_ids: set[str]) -> None:
    seen: set[str] = set()
    for record in records:
        group_id, _, _ = _typed_rejection(record)
        if group_id not in known_group_ids:
            raise CascadeError(f"Rejected-group record references unknown group {group_id}")
        if group_id in seen:
            raise CascadeError(f"Rejected-group records contain duplicate group {group_id}")
        seen.add(group_id)


def refill_selection(
    request: dict[str, Any],
    primary: list[dict[str, Any]],
    reserves: list[dict[str, Any]],
    rejected_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    primary_target, reserve_target = request_selection_counts(request)
    if len(primary) != primary_target or len(reserves) != reserve_target:
        raise CascadeError("Primary/reserve selection counts do not match the request")
    all_rows = primary + reserves
    group_ids = [row.get("groupId") for row in all_rows]
    if len(group_ids) != len(set(group_ids)):
        raise CascadeError("Primary and reserve selections must have unique group IDs")
    validate_rejected_group_records(rejected_records, set(group_ids))
    rejected_by_id = {record["groupId"]: record for record in rejected_records}
    ordered_primary = sorted(primary, key=lambda row: (row.get("tierOrdinal", row.get("selectionOrdinal", 0)), row["groupId"]))
    eligible_reserves = sorted(
        (row for row in reserves if row["groupId"] not in rejected_by_id),
        key=lambda row: (row.get("tierOrdinal", row.get("selectionOrdinal", 0)), row["groupId"]),
    )
    reserve_index = 0
    active: list[dict[str, Any]] = []
    rejection_set_hash = content_hash(rejected_records)
    for slot, row in enumerate(ordered_primary, start=1):
        if row["groupId"] not in rejected_by_id:
            materialized = dict(row)
            materialized["materializationRole"] = "primary"
        else:
            if reserve_index >= len(eligible_reserves):
                raise CascadeError("Typed rejections exceed the available deterministic reserve pool")
            materialized = dict(eligible_reserves[reserve_index])
            reserve_index += 1
            materialized["materializationRole"] = "refill"
            materialized["replacesGroupId"] = row["groupId"]
            materialized["rejectionRecordHash"] = content_hash(rejected_by_id[row["groupId"]])
        materialized["activeSelectionOrdinal"] = slot
        materialized["refillRationale"] = {
            "algorithm": "typed-rejection-reserve-order-v1",
            "rejectionSetHash": rejection_set_hash,
            "requestHash": content_hash(request),
        }
        materialized["activeSelectionHash"] = content_hash({
            "selectionHash": materialized.get("selectionHash"),
            "slot": slot,
            "role": materialized["materializationRole"],
            "replaces": materialized.get("replacesGroupId"),
            "rejectionSetHash": rejection_set_hash,
        })
        active.append(materialized)
    return active


def plan_pair(
    planner: JsonOnlyPlanner,
    request: dict[str, Any],
    selection: dict[str, Any],
    scenario: dict[str, Any],
    trajectory: dict[str, Any],
) -> dict[str, Any]:
    roles = request_sibling_roles(request)
    four_siblings = len(roles) == 4
    pivot = typed_pivot(trajectory) if four_siblings else None
    family = trajectory.get("interventionFamily") if four_siblings else None
    if four_siblings:
        if pivot is None:
            raise CascadeError("Four-sibling planning requires a trajectory typedPivot")
        require_text(family, "TrajectorySeed.interventionFamily")
    common_context = {
        "requestId": request["requestId"],
        "scenario": scenario,
        "trajectory": trajectory,
        "groupId": selection["groupId"],
    }
    if four_siblings:
        common_context["interventionFamily"] = family
        common_context["typedPivot"] = pivot
    common_hash = content_hash(common_context)
    requirements = [
        f"Return exactly these sibling roles: {list(roles)}.",
        "Every sibling uses one common controlDelta.field and from state, with a distinct to state.",
        "Use the supplied groupId and trajectoryId exactly.",
        f"Set commonContextHash exactly to {common_hash}.",
        "Do not include target wording, transcripts, audio, or semantic-certification claims.",
    ]
    if four_siblings:
        requirements.extend([
            f"Declare interventionFamily exactly as {family}.",
            f"Copy typedPivot exactly as {canonical_json(pivot)}.",
            "Represent verified positive, verified negative, uncertain, and superseded as structural sibling roles.",
        ])
    strategy = request.get("strategyVersion")
    if strategy in {"semantic-control-v4", "semantic-control-v5"}:
        requirements.extend([
            "Set sharedPrefixPolicy to native_code_identical_through_pivot.",
            "Each sibling requires availabilityTiming with availableBeforeTarget true and a positive controlRevision.",
            "Each sibling requires controlValue exactly equal to its controlDelta.to value.",
            "Each sibling requires paired_wrong_branch, stale_revision, and null_control negative controls.",
            "Each sibling requires target-free semanticAssertions.",
        ])
    schema = (
        "personaplex.counterfactual-sibling-group-spec.v2"
        if four_siblings and strategy == "semantic-control-v5"
        else "personaplex.counterfactual-sibling-group-spec.v1"
        if four_siblings
        else "personaplex.counterfactual-pair-spec.v1"
    )
    required_fields = ["schema", "groupId", "trajectoryId", "pivotOrdinal", "commonContextHash", "branches"]
    if four_siblings:
        required_fields.extend(["interventionFamily", "typedPivot"])
    prompt = {
        "task": "Create one typed causal counterfactual sibling group without dialogue text.",
        "requiredTopLevelKey": "groupSpecs" if four_siblings else "pairSpecs",
        "responseShape": {"groupSpecs" if four_siblings else "pairSpecs": ["exactly one group object"]},
        "selection": selection,
        "scenarioContract": scenario,
        "trajectorySeed": trajectory,
        "requiredSchema": schema,
        "requiredFields": required_fields,
        "siblingRoles": roles,
        "semanticControl": request.get("semanticControl"),
        "requirements": requirements,
    }
    key = "groupSpecs" if four_siblings else "pairSpecs"
    pair = _model_list(
        planner,
        key,
        canonical_json(prompt),
        1,
        schema_definition="siblingGroup" if strategy == "semantic-control-v5" else None,
    )[0]
    validate_pair_spec(pair, {trajectory["trajectoryId"]}, request=request, trajectory=trajectory)
    if strategy in {"semantic-control-v4", "semantic-control-v5"}:
        validate_v4_pair_spec(pair)
    if pair["groupId"] != selection["groupId"] or pair["trajectoryId"] != trajectory["trajectoryId"]:
        raise CascadeError("Counterfactual group identity does not match selected trajectory")
    if pair["commonContextHash"] != common_hash:
        raise CascadeError("Counterfactual commonContextHash does not bind its common context")
    if pair["pivotOrdinal"] != trajectory["counterfactualPivotOrdinal"]:
        raise CascadeError("Counterfactual pivotOrdinal does not match trajectory")
    return pair


def parallel_map(
    items: list[Any],
    worker: Callable[[Any], list[dict[str, Any]] | dict[str, Any]],
    max_workers: int,
    on_result: Callable[[Any], None] | None = None,
) -> list[Any]:
    results: list[Any] = []
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {executor.submit(worker, item): item for item in items}
    try:
        for future in as_completed(futures):
            try:
                result = future.result()
            except BaseException as error:
                for pending in futures:
                    pending.cancel()
                source = futures[future]
                raise CascadeError(f"Cascade stage failed for {source}: {error}") from error
            if on_result is not None:
                on_result(result)
            results.append(result)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return results


def collect_schema_hashes(repository_root: Path) -> dict[str, str]:
    schema_root = repository_root / "ground_truth_finetuning" / "schemas"
    hashes: dict[str, str] = {}
    for path in sorted(schema_root.glob("*.schema.json")):
        hashes[path.name] = content_hash(load_json(path))
    if not hashes:
        raise CascadeError(f"No cascade schemas found under {schema_root}")
    return hashes


def cascade_config_hash(request: dict[str, Any], max_workers: int | None = None) -> str:
    return content_hash({
        "strategyVersion": request.get("strategyVersion"),
        "coverageTarget": request.get("coverageTarget"),
        "selection": request.get("selection", request.get("selectionTarget", request.get("selectionPlan"))),
        "causalGroupContract": request.get("causalGroupContract", request.get("counterfactualGroup")),
        "semanticControl": request.get("semanticControl"),
        "requiredControlCoverage": request.get("requiredControlCoverage"),
        "maxWorkers": max_workers,
    })


def planner_config_hash(config: PlannerConfig | None) -> str:
    if config is None:
        return content_hash({"planner": "not_required"})
    return content_hash({
        "endpoints": list(normalize_planner_endpoints(config.endpoint)),
        "model": config.model,
        "timeoutSeconds": config.timeout_seconds,
        "maxTokens": config.max_tokens,
        "temperature": config.temperature,
    })


def write_run_manifest(
    output_root: Path,
    request: dict[str, Any],
    stage: str,
    artifacts: dict[str, list[dict[str, Any]]],
    *,
    planner_config: PlannerConfig | None = None,
    max_workers: int | None = None,
    catalog_hash: str | None = None,
    schema_hashes: dict[str, str] | None = None,
) -> None:
    schemas = schema_hashes or {}
    rationale = [
        row["selectionRationale"]
        for name in ("primary", "reserve", "selection")
        for row in artifacts.get(name, [])
        if isinstance(row.get("selectionRationale"), dict)
    ]
    manifest = {
        "schema": "personaplex.diverse-cascade-run-manifest.v2",
        "requestId": request["requestId"],
        "requestHash": content_hash(request),
        "catalogHash": catalog_hash,
        "seedCatalogHash": catalog_hash,
        "plannerHash": planner_config_hash(planner_config),
        "configHash": cascade_config_hash(request, max_workers),
        "schemaHashes": schemas,
        "schemaHash": content_hash(schemas),
        "completedStage": stage,
        "artifacts": {
            name: {"count": len(rows), "hash": content_hash(rows)}
            for name, rows in artifacts.items()
        },
        "selectionRationaleHash": content_hash(rationale),
        "admission": "planning_only_not_source_certified",
    }
    write_json(output_root / "run_manifest.json", manifest)
