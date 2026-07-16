"""Versioned, resumable planning cascade for controlled PersonaPlex source data.

This module intentionally plans context and control only. It never creates target
dialogue, target audio, or a semantic certificate. Voryn remains the turn/audio
realizer and independent source certifier.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable
import json
import re
import urllib.error
import urllib.request


class CascadeError(RuntimeError):
    """Raised when a cascade artifact cannot safely advance."""


ID_PATTERN = re.compile(r"^[a-z][a-z0-9_:-]{2,199}$")
FORBIDDEN_TARGET_KEYS = frozenset({
    "canonical_response", "canonicalresponse", "target_text", "targettext",
    "target_audio", "targetaudio", "spoken_text", "spokentext",
    "expected_response", "expectedresponse", "agent_response", "agentresponse",
})


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any) -> str:
    return f"sha256:{sha256(canonical_json(value).encode('utf-8')).hexdigest()}"


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


def require_identifier(value: Any, label: str) -> str:
    identifier = str(value or "")
    if not ID_PATTERN.fullmatch(identifier):
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


def validate_request(request: dict[str, Any]) -> None:
    require_keys(request, {
        "schema", "requestId", "seedRevision", "seedIdeas", "coverageTarget",
        "allowedVoicesManifest", "renderer", "asr", "allowedPhysicalCudaDevices",
        "prohibitedContentPolicyRevision",
    }, "DiverseCorpusRequestV1")
    if request["schema"] != "personaplex.diverse-corpus-request.v1":
        raise CascadeError("Unsupported request schema")
    if not isinstance(request["requestId"], str) or len(request["requestId"].strip()) < 8:
        raise CascadeError("requestId must be a nonempty stable identifier")
    for field in ("seedRevision", "allowedVoicesManifest"):
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", str(request[field])):
            raise CascadeError(f"{field} must be a SHA-256 content hash")
    require_string_list(request["seedIdeas"], "seedIdeas")
    coverage = request["coverageTarget"]
    if not isinstance(coverage, dict):
        raise CascadeError("coverageTarget must be an object")
    require_keys(coverage, {
        "candidateTopics", "scenariosPerTopic", "trajectorySeedsPerScenario",
        "selectedCounterfactualGroups", "branchesPerGroup",
    }, "coverageTarget")
    counts = [coverage[name] for name in (
        "candidateTopics", "scenariosPerTopic", "trajectorySeedsPerScenario",
        "selectedCounterfactualGroups",
    )]
    if not all(isinstance(item, int) and item > 0 for item in counts):
        raise CascadeError("coverage counts must be positive integers")
    if coverage["branchesPerGroup"] != 2:
        raise CascadeError("Only two-branch causal counterfactual groups are supported")
    candidate_count = coverage["candidateTopics"] * coverage["scenariosPerTopic"] * coverage["trajectorySeedsPerScenario"]
    if coverage["selectedCounterfactualGroups"] > candidate_count:
        raise CascadeError("selectedCounterfactualGroups exceeds candidate lattice size")
    if request["renderer"] != "voicebox_chatterbox_turbo" or request["asr"] != "whisper":
        raise CascadeError("This programme is pinned to Chatterbox Turbo and Whisper")
    if request["allowedPhysicalCudaDevices"] != [0, 1, 2]:
        raise CascadeError("Only physical CUDA devices 0, 1, and 2 are permitted")
    assert_no_target_leak(request)


def validate_topic_card(card: dict[str, Any], seed_revision: str) -> None:
    require_keys(card, {"schema", "topicId", "seedRevision", "domain", "interactionModes", "registerRange", "safeStakes", "forbiddenPatterns", "diversityTags"}, "TopicCardV1")
    if card["schema"] != "personaplex.topic-card.v1":
        raise CascadeError("Unsupported topic-card schema")
    require_identifier(card["topicId"], "topicId")
    if card["seedRevision"] != seed_revision:
        raise CascadeError("Topic card seedRevision does not match request")
    if not isinstance(card["domain"], str) or len(card["domain"].strip()) < 3:
        raise CascadeError("Topic domain must be meaningful text")
    for field in ("interactionModes", "registerRange", "safeStakes", "forbiddenPatterns", "diversityTags"):
        require_string_list(card[field], f"TopicCardV1.{field}")
    assert_no_target_leak(card)


def validate_scenario_contract(contract: dict[str, Any], known_topics: set[str]) -> None:
    require_keys(contract, {"schema", "scenarioId", "topicId", "mode", "premise", "participants", "startingState", "interactionOpportunity", "allowedToolClasses", "disallowedClaims", "scenarioOutcomeSpace", "requiredControlPhenomena"}, "ScenarioContractV1")
    if contract["schema"] != "personaplex.scenario-contract.v1":
        raise CascadeError("Unsupported scenario-contract schema")
    require_identifier(contract["scenarioId"], "scenarioId")
    if contract["topicId"] not in known_topics:
        raise CascadeError("Scenario references an unknown topic")
    if not isinstance(contract["mode"], str) or not contract["mode"].strip():
        raise CascadeError("Scenario mode is required")
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


def validate_trajectory_seed(seed: dict[str, Any], known_scenarios: set[str]) -> None:
    require_keys(seed, {"schema", "trajectoryId", "scenarioId", "conversationLength", "pace", "openingStyle", "closingStyle", "voicePairPolicy", "interactionArc", "duplexEvents", "postureArc", "counterfactualPivotOrdinal", "controlPhenomena"}, "TrajectorySeedV1")
    if seed["schema"] != "personaplex.trajectory-seed.v1":
        raise CascadeError("Unsupported trajectory-seed schema")
    require_identifier(seed["trajectoryId"], "trajectoryId")
    if seed["scenarioId"] not in known_scenarios:
        raise CascadeError("Trajectory references an unknown scenario")
    length = seed["conversationLength"]
    if not isinstance(length, dict) or not all(isinstance(length.get(key), int) for key in ("targetTurns", "min", "max")):
        raise CascadeError("Trajectory conversationLength is invalid")
    if not (4 <= length["min"] <= length["targetTurns"] <= length["max"] <= 48):
        raise CascadeError("Trajectory conversationLength bounds are invalid")
    pivot = seed["counterfactualPivotOrdinal"]
    if not isinstance(pivot, int) or not 1 <= pivot < length["targetTurns"]:
        raise CascadeError("Counterfactual pivot must occur before the final planned turn")
    if seed["voicePairPolicy"] != "distinct_approved_references":
        raise CascadeError("Trajectory must require distinct approved voice references")
    for field in ("pace", "openingStyle", "closingStyle"):
        if not isinstance(seed[field], str) or not seed[field].strip():
            raise CascadeError(f"Trajectory {field} is required")
    for field in ("interactionArc", "postureArc", "controlPhenomena"):
        require_string_list(seed[field], f"TrajectorySeedV1.{field}")
    if not isinstance(seed["duplexEvents"], list):
        raise CascadeError("Trajectory duplexEvents must be an array")
    assert_no_target_leak(seed)


def validate_pair_spec(pair: dict[str, Any], known_trajectories: set[str]) -> None:
    require_keys(pair, {"schema", "groupId", "trajectoryId", "pivotOrdinal", "commonContextHash", "branches"}, "CounterfactualPairSpecV1")
    if pair["schema"] != "personaplex.counterfactual-pair-spec.v1":
        raise CascadeError("Unsupported counterfactual-pair schema")
    if pair["trajectoryId"] not in known_trajectories:
        raise CascadeError("Pair references an unknown trajectory")
    if not isinstance(pair["groupId"], str) or len(pair["groupId"]) < 8:
        raise CascadeError("Pair groupId is invalid")
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", str(pair["commonContextHash"])):
        raise CascadeError("Pair commonContextHash must be content-addressed")
    branches = pair["branches"]
    if not isinstance(branches, list) or len(branches) != 2:
        raise CascadeError("Pair must contain exactly two branches")
    branch_ids = {branch.get("branchId") for branch in branches if isinstance(branch, dict)}
    if branch_ids != {"available", "constrained"}:
        raise CascadeError("Pair branch IDs must be available and constrained")
    deltas = []
    for branch in branches:
        delta = branch.get("controlDelta") if isinstance(branch, dict) else None
        if not isinstance(delta, dict) or not isinstance(delta.get("field"), str) or "from" not in delta or "to" not in delta:
            raise CascadeError("Every pair branch requires a typed control delta")
        if not isinstance(branch.get("evidenceUpdate"), dict):
            raise CascadeError("Every pair branch requires an evidence update")
        deltas.append(delta)
    if deltas[0]["field"] != deltas[1]["field"] or canonical_json(deltas[0]["from"]) != canonical_json(deltas[1]["from"]):
        raise CascadeError("Pair branches must vary exactly one common control field")
    if canonical_json(deltas[0]["to"]) == canonical_json(deltas[1]["to"]):
        raise CascadeError("Counterfactual branches must have materially different target states")
    assert_no_target_leak(pair)


@dataclass(frozen=True)
class PlannerConfig:
    endpoint: str
    model: str
    api_key: str
    timeout_seconds: int = 180
    max_tokens: int = 4096
    temperature: float = 0.85


class JsonOnlyPlanner:
    """OpenAI-compatible planner with no parsing fallback or semantic shortcut."""

    def __init__(self, config: PlannerConfig):
        if not config.endpoint or not config.model:
            raise CascadeError("Planner endpoint and model are required for generative stages")
        self.config = config

    def call(self, system: str, user: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
            "reasoning": {"enabled": False},
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                envelope = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as error:
            raise CascadeError(f"Planner inference failed: {error}") from error
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise CascadeError("Planner response lacks choices[0].message.content") from error
        if not isinstance(content, str):
            raise CascadeError("Planner response content must be a raw JSON string")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as error:
            raise CascadeError("Planner returned non-JSON content; no text/regex recovery is permitted") from error
        if not isinstance(result, dict):
            raise CascadeError("Planner result must be a JSON object")
        return result


PLANNER_SYSTEM = """You are a planning component for a controlled conversational-audio training corpus.
Reason silently. Return one raw JSON object only, with exactly the requested top-level key.
Do not include markdown, prose, canonical target responses, target transcripts, target audio,
spoken dialogue, names, contact information, credentials, or placeholder text. Do not make a
semantic certification claim; create planning artifacts only."""


def _model_list(planner: JsonOnlyPlanner, expected_key: str, user_prompt: str, expected_count: int) -> list[dict[str, Any]]:
    response = planner.call(PLANNER_SYSTEM, user_prompt)
    if set(response) != {expected_key} or not isinstance(response[expected_key], list):
        raise CascadeError(f"Planner must return only a {expected_key} array")
    values = response[expected_key]
    if len(values) != expected_count or not all(isinstance(value, dict) for value in values):
        raise CascadeError(f"Planner returned {len(values)} {expected_key}; expected {expected_count}")
    return values


def plan_topics(planner: JsonOnlyPlanner, request: dict[str, Any]) -> list[dict[str, Any]]:
    count = request["coverageTarget"]["candidateTopics"]
    prompt = {
        "task": "Generate diverse broad topic cards, not dialogue.",
        "requestedCount": count,
        "seedRevision": request["seedRevision"],
        "seedIdeas": request["seedIdeas"],
        "topicConstraints": request.get("topicConstraints", {}),
        "requiredSchema": "personaplex.topic-card.v1",
        "requiredFields": ["schema", "topicId", "seedRevision", "domain", "interactionModes", "registerRange", "safeStakes", "forbiddenPatterns", "diversityTags"],
        "requirements": [
            "Cover social, service, technical, research, learning, community, creative, care, and practical interaction modes where permitted.",
            "Avoid repetitive scheduling, company greetings, identity collection, political persuasion, and target dialogue.",
            "Use unique lowercase underscore topicId values and copy the supplied seedRevision exactly.",
        ],
    }
    cards = _model_list(planner, "topicCards", canonical_json(prompt), count)
    for card in cards:
        validate_topic_card(card, request["seedRevision"])
    if len({card["topicId"] for card in cards}) != len(cards):
        raise CascadeError("Planner generated duplicate topic IDs")
    return sorted(cards, key=lambda card: card["topicId"])


def plan_scenarios(planner: JsonOnlyPlanner, topic: dict[str, Any], request: dict[str, Any]) -> list[dict[str, Any]]:
    count = request["coverageTarget"]["scenariosPerTopic"]
    prompt = {
        "task": "Generate scenario contracts that supply context but no dialogue targets.",
        "topicCard": topic,
        "requestedCount": count,
        "requiredSchema": "personaplex.scenario-contract.v1",
        "requiredFields": ["schema", "scenarioId", "topicId", "mode", "premise", "participants", "startingState", "interactionOpportunity", "allowedToolClasses", "disallowedClaims", "scenarioOutcomeSpace", "requiredControlPhenomena"],
        "requirements": [
            "Every scenario must be safe, non-identifying, and able to support multiple interaction arcs.",
            "Include facts, uncertainty, and policy boundaries without writing any canonical response.",
            "Vary stakes, evidence needs, resistance, clarification, handoff, recovery, and control phenomena.",
            "Set topicId exactly to the supplied topic card topicId and use unique lowercase structured scenarioId values.",
        ],
    }
    scenarios = _model_list(planner, "scenarioContracts", canonical_json(prompt), count)
    for scenario in scenarios:
        validate_scenario_contract(scenario, {topic["topicId"]})
    if len({scenario["scenarioId"] for scenario in scenarios}) != len(scenarios):
        raise CascadeError(f"Planner generated duplicate scenario IDs for {topic['topicId']}")
    return sorted(scenarios, key=lambda scenario: scenario["scenarioId"])


def plan_trajectories(planner: JsonOnlyPlanner, scenario: dict[str, Any], request: dict[str, Any]) -> list[dict[str, Any]]:
    count = request["coverageTarget"]["trajectorySeedsPerScenario"]
    prompt = {
        "task": "Generate trajectory seeds, not transcripts or target dialogue.",
        "scenarioContract": scenario,
        "requestedCount": count,
        "requiredSchema": "personaplex.trajectory-seed.v1",
        "requiredFields": ["schema", "trajectoryId", "scenarioId", "conversationLength", "pace", "openingStyle", "closingStyle", "voicePairPolicy", "interactionArc", "duplexEvents", "postureArc", "counterfactualPivotOrdinal", "controlPhenomena"],
        "requirements": [
            "Create genuinely distinct call shapes: brief/balanced/extended, varied pacing, varied openings, and model-driven endings.",
            "Include diverse cooperation, conditional compliance, skepticism, resistance, repair, clarification, refusal, handoff, and recovery where appropriate.",
            "Include real duplex event plans with a barge-in/cutoff/recovery subset across the set.",
            "Each seed requires distinct approved voice references and a pre-final counterfactual pivot.",
            "Set scenarioId exactly to the supplied scenario and use unique lowercase structured trajectoryId values.",
        ],
    }
    seeds = _model_list(planner, "trajectorySeeds", canonical_json(prompt), count)
    for seed in seeds:
        validate_trajectory_seed(seed, {scenario["scenarioId"]})
    if len({seed["trajectoryId"] for seed in seeds}) != len(seeds):
        raise CascadeError(f"Planner generated duplicate trajectory IDs for {scenario['scenarioId']}")
    return sorted(seeds, key=lambda seed: seed["trajectoryId"])


def _stable_rank(request: dict[str, Any], value: Any) -> str:
    return sha256(f"{request['seedRevision']}:{canonical_json(value)}".encode("utf-8")).hexdigest()


def select_trajectories(request: dict[str, Any], topics: list[dict[str, Any]], scenarios: list[dict[str, Any]], trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target = request["coverageTarget"]["selectedCounterfactualGroups"]
    scenario_by_id = {item["scenarioId"]: item for item in scenarios}
    by_topic: dict[str, list[dict[str, Any]]] = {item["topicId"]: [] for item in topics}
    for trajectory in trajectories:
        topic_id = scenario_by_id[trajectory["scenarioId"]]["topicId"]
        by_topic[topic_id].append(trajectory)
    topic_ids = sorted(by_topic, key=lambda item: _stable_rank(request, {"topicId": item}))
    if target < len(topic_ids):
        raise CascadeError("selectedCounterfactualGroups must allocate at least one group per topic")
    base, remainder = divmod(target, len(topic_ids))
    used_dimensions: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    for topic_index, topic_id in enumerate(topic_ids):
        quota = base + (1 if topic_index < remainder else 0)
        candidates_by_scenario: dict[str, list[dict[str, Any]]] = {}
        for candidate in by_topic[topic_id]:
            candidates_by_scenario.setdefault(candidate["scenarioId"], []).append(candidate)
        if quota > len(candidates_by_scenario):
            raise CascadeError(f"Topic {topic_id} lacks distinct scenario coverage for quota {quota}")
        remaining = {key: sorted(value, key=lambda item: _stable_rank(request, item)) for key, value in candidates_by_scenario.items()}
        for ordinal in range(quota):
            candidates = [items[0] for items in remaining.values() if items]
            if not candidates:
                raise CascadeError(f"Topic {topic_id} exhausted before its selection quota")
            def score(candidate: dict[str, Any]) -> tuple[int, str]:
                dimensions = [
                    f"pace:{candidate['pace']}",
                    f"opening:{candidate['openingStyle']}",
                    f"closing:{candidate['closingStyle']}",
                    *[f"control:{item}" for item in candidate["controlPhenomena"]],
                ]
                return (sum(used_dimensions.get(item, 0) for item in dimensions), _stable_rank(request, candidate))
            chosen = min(candidates, key=score)
            for dimension in [
                f"pace:{chosen['pace']}", f"opening:{chosen['openingStyle']}", f"closing:{chosen['closingStyle']}",
                *[f"control:{item}" for item in chosen["controlPhenomena"]],
            ]:
                used_dimensions[dimension] = used_dimensions.get(dimension, 0) + 1
            group_id = f"cascade-{request['requestId']}-{len(selected) + 1:04d}"
            selected.append({
                "schema": "personaplex.selected-trajectory.v1",
                "groupId": group_id,
                "topicId": topic_id,
                "scenarioId": chosen["scenarioId"],
                "trajectoryId": chosen["trajectoryId"],
                "selectionOrdinal": ordinal + 1,
                "selectionSeedRevision": request["seedRevision"],
                "selectionHash": content_hash({"request": request["requestId"], "groupId": group_id, "trajectoryId": chosen["trajectoryId"]}),
            })
            del remaining[chosen["scenarioId"]]
    if len(selected) != target or len({row["trajectoryId"] for row in selected}) != target:
        raise CascadeError("Selection must contain exactly the requested number of unique trajectories")
    return selected


def plan_pair(planner: JsonOnlyPlanner, request: dict[str, Any], selection: dict[str, Any], scenario: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    common_context = {
        "requestId": request["requestId"],
        "scenario": scenario,
        "trajectory": trajectory,
        "groupId": selection["groupId"],
    }
    prompt = {
        "task": "Create a causal two-branch counterfactual specification without dialogue text.",
        "selection": selection,
        "scenarioContract": scenario,
        "trajectorySeed": trajectory,
        "requiredSchema": "personaplex.counterfactual-pair-spec.v1",
        "requiredFields": ["schema", "groupId", "trajectoryId", "pivotOrdinal", "commonContextHash", "branches"],
        "requirements": [
            "Return exactly available and constrained branches.",
            "Both branches must share the same one changed controlDelta.field and from value, with different to values.",
            "The changed field must be a valid state/evidence difference supported by the scenario and trajectory.",
            "Use the supplied groupId and trajectoryId exactly.",
            f"Set commonContextHash exactly to {content_hash(common_context)}.",
            "Do not include target wording, transcripts, audio, or a claim that either branch is already semantically certified.",
        ],
    }
    values = _model_list(planner, "pairSpecs", canonical_json(prompt), 1)
    pair = values[0]
    validate_pair_spec(pair, {trajectory["trajectoryId"]})
    if pair["groupId"] != selection["groupId"] or pair["trajectoryId"] != trajectory["trajectoryId"]:
        raise CascadeError("Pair identity does not match selected trajectory")
    if pair["commonContextHash"] != content_hash(common_context):
        raise CascadeError("Pair commonContextHash does not bind its common context")
    if pair["pivotOrdinal"] != trajectory["counterfactualPivotOrdinal"]:
        raise CascadeError("Pair pivotOrdinal does not match trajectory")
    return pair


def parallel_map(items: list[Any], worker: Callable[[Any], list[dict[str, Any]] | dict[str, Any]], max_workers: int) -> list[Any]:
    results: list[Any] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for future in as_completed(futures):
            try:
                result = future.result()
            except Exception as error:
                source = futures[future]
                raise CascadeError(f"Cascade stage failed for {source}: {error}") from error
            results.append(result)
    return results


def write_run_manifest(output_root: Path, request: dict[str, Any], stage: str, artifacts: dict[str, list[dict[str, Any]]]) -> None:
    manifest = {
        "schema": "personaplex.diverse-cascade-run-manifest.v1",
        "requestId": request["requestId"],
        "requestHash": content_hash(request),
        "completedStage": stage,
        "artifacts": {
            name: {"count": len(rows), "hash": content_hash(rows)}
            for name, rows in artifacts.items()
        },
        "admission": "planning_only_not_source_certified",
    }
    write_json(output_root / "run_manifest.json", manifest)

