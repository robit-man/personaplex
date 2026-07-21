"""Leakage-safe packing for native PersonaPlex causal sibling groups.

The packer treats the four sibling responses as one listwise training unit.  Shared
duplex audio and context are canonicalized once, while controls and target labels
remain branch-local.  Splits are assigned only after transitive leakage components
have been built from lineage, template, operator, and voice-pair identifiers.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CAUSAL_GROUP_ROLES = (
    "verified_positive",
    "verified_negative",
    "uncertain",
    "superseded",
)
PACK_SCHEMA = "personaplex.native-causal-group-pack.v1"
CERTIFICATE_SCHEMA = "personaplex.native-causal-group-coverage-certificate.v2"
MANIFEST_SCHEMA = "personaplex.native-causal-group-pack-manifest.v2"
TRAINER_BINDING_SCHEMA = "personaplex.native-moshirag-certified-pack-binding.v1"
TRAINER_DATASET_SCHEMA = "personaplex.native-moshirag-dataset.v2-shared-prefix"
TRAINER_GROUP_SCHEMA = "personaplex.native-moshirag-group.v2-shared-prefix"

_FORBIDDEN_CONTROL_KEYS = frozenset(
    {
        "agentresponse",
        "canonicalresponse",
        "expectedresponse",
        "labels",
        "spokentext",
        "targetaudio",
        "targetlabel",
        "targettext",
        "targettranscript",
    }
)
_LINEAGE_ID_KEYS = frozenset(
    {
        "lineageid",
        "parentid",
        "premiseid",
        "scenarioid",
        "seedid",
        "sourceseedid",
        "topicid",
        "trajectoryid",
    }
)
_AUDIO_LOCATION_KEYS = frozenset(
    {"file", "filepath", "path", "sourcepath", "uri", "url"}
)


class CausalGroupPackError(ValueError):
    """Raised when a causal group cannot be packed without leakage ambiguity."""


@dataclass(frozen=True)
class PackConfig:
    """Deterministic split and structural-coverage policy."""

    split_ratios: tuple[tuple[str, float], ...] = (
        ("train", 0.8),
        ("validation", 0.1),
        ("test", 0.1),
    )
    split_seed: str = "personaplex-native-causal-groups-v1"
    required_coverage_splits: tuple[str, ...] = ("train", "validation", "test")
    minimum_distinct_premises: int = 2

    def __post_init__(self) -> None:
        names = tuple(name for name, _ in self.split_ratios)
        if names != ("train", "validation", "test"):
            raise CausalGroupPackError(
                "split_ratios must define train, validation, and test in that order"
            )
        if any(not isinstance(ratio, (int, float)) or ratio <= 0 for _, ratio in self.split_ratios):
            raise CausalGroupPackError("all split ratios must be positive")
        if abs(sum(ratio for _, ratio in self.split_ratios) - 1.0) > 1e-9:
            raise CausalGroupPackError("split ratios must sum to one")
        if not self.split_seed:
            raise CausalGroupPackError("split_seed must not be empty")
        if not self.required_coverage_splits:
            raise CausalGroupPackError("required_coverage_splits must not be empty")
        unknown = set(self.required_coverage_splits).difference(names)
        if unknown:
            raise CausalGroupPackError(f"unknown required coverage splits: {sorted(unknown)}")
        if self.minimum_distinct_premises < 2:
            raise CausalGroupPackError("minimum_distinct_premises must be at least two")


@dataclass(frozen=True)
class PackResult:
    common_inputs: tuple[dict[str, Any], ...]
    listwise_groups: tuple[dict[str, Any], ...]
    pairwise_diagnostics: tuple[dict[str, Any], ...]
    leakage_components: tuple[dict[str, Any], ...]
    certificate: dict[str, Any]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _word_sequence(value: str) -> tuple[str, ...]:
    words: list[str] = []
    current: list[str] = []
    for character in value.casefold():
        if character.isalnum():
            current.append(character)
        elif current:
            words.append("".join(current))
            current = []
    if current:
        words.append("".join(current))
    return tuple(words)


def _contains_words(container: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(container):
        return False
    width = len(needle)
    return any(container[index : index + width] == needle for index in range(len(container) - width + 1))


def _walk_strings(value: Any, path: str = "control") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _normalized_key(key) in _FORBIDDEN_CONTROL_KEYS:
                raise CausalGroupPackError(f"target-label field {key!r} is forbidden at {path}")
            yield from _walk_strings(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def assert_no_target_text_leak(target_text: str, *model_inputs: Any) -> None:
    """Reject exact target wording or its digest anywhere in model-visible inputs."""

    target_words = _word_sequence(target_text)
    if not target_words:
        raise CausalGroupPackError("target text must contain lexical content")
    target_hashes = {
        sha256(target_text.encode("utf-8")).hexdigest(),
        sha256(" ".join(target_words).encode("utf-8")).hexdigest(),
    }
    for model_input in model_inputs:
        for path, value in _walk_strings(model_input):
            words = _word_sequence(value)
            if _contains_words(words, target_words):
                raise CausalGroupPackError(f"target text leaked into model input at {path}")
            lowered = value.casefold()
            if any(digest in lowered for digest in target_hashes):
                raise CausalGroupPackError(f"target-text digest leaked into model input at {path}")


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CausalGroupPackError(f"{label} must be an object")
    return value


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CausalGroupPackError(f"{label} must be nonempty text")
    return value.strip()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(character in "0123456789abcdef" for character in value[7:])


def _audio_identity_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _audio_identity_value(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
            if _normalized_key(key) not in _AUDIO_LOCATION_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_audio_identity_value(child) for child in value]
    return value


def _audio_hashes(value: Any) -> list[str]:
    hashes: list[str] = []
    if isinstance(value, Mapping):
        for child in value.values():
            hashes.extend(_audio_hashes(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            hashes.extend(_audio_hashes(child))
    elif _is_sha256(value):
        hashes.append(value)
    return hashes


def _audio_identity(value: Any, label: str) -> Any:
    if isinstance(value, str) and _is_sha256(value):
        return {"sha256": value}
    if not isinstance(value, (Mapping, list, tuple)):
        raise CausalGroupPackError(f"{label} must be a content-addressed audio reference")
    if not _audio_hashes(value):
        raise CausalGroupPackError(f"{label} lacks a sha256 content identifier")
    return _audio_identity_value(value)


def _value_from(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


def _branch_role(branch: Mapping[str, Any]) -> str:
    return str(_value_from(branch, ("role", "siblingRole", "branchId")) or "")


def _group_branches(group: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = _value_from(group, ("siblings", "branches"))
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise CausalGroupPackError("causal group must contain a siblings or branches array")
    return list(raw)


def _common_candidate(container: Mapping[str, Any]) -> tuple[Any, Any] | None:
    nested = _value_from(container, ("commonInput", "common", "sharedInput"))
    if isinstance(nested, Mapping):
        audio = _value_from(
            nested,
            ("audio", "commonAudio", "sharedAudio", "priorDuplexAudio", "sharedPrefixAudio"),
        )
        context = _value_from(
            nested,
            ("context", "commonContext", "sharedContext", "priorContext", "sharedPrefixContext"),
        )
    else:
        audio = _value_from(
            container,
            ("commonAudio", "sharedAudio", "priorDuplexAudio", "sharedPrefixAudio"),
        )
        context = _value_from(
            container,
            ("commonContext", "sharedContext", "priorContext", "sharedPrefixContext"),
        )
    if audio is None and context is None:
        return None
    if audio is None or context is None:
        raise CausalGroupPackError("common input must provide both audio and context")
    if not isinstance(context, Mapping):
        raise CausalGroupPackError("common context must be an object")
    return audio, context


def _canonical_common(
    group_id: str,
    group: Mapping[str, Any],
    branches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = [candidate for candidate in [_common_candidate(group)] if candidate is not None]
    candidates.extend(
        candidate for branch in branches if (candidate := _common_candidate(branch)) is not None
    )
    if not candidates:
        raise CausalGroupPackError(f"group {group_id} lacks common audio/context")
    identities = [
        {
            "audio": _audio_identity(audio, f"group {group_id} common audio"),
            "context": context,
        }
        for audio, context in candidates
    ]
    hashes = {content_hash(identity) for identity in identities}
    if len(hashes) != 1:
        raise CausalGroupPackError(f"group {group_id} siblings do not share identical common input")
    audio, context = candidates[0]
    return {
        "commonInputId": next(iter(hashes)),
        "audio": deepcopy(audio),
        "context": deepcopy(context),
    }


_NATIVE_PIVOT_KEYS = (
    "nativePivotFrame",
    "pivotNativeFrame",
    "pivotFrame",
    "memberAtFrame",
    "member_at",
)
_DONOR_PIVOT_KEYS = ("donorAtFrame", "donor_at")


def _native_frame(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CausalGroupPackError(f"{label} must be a non-negative native frame index")
    return value


def _branch_native_pivot(group_id: str, role: str, branch: Mapping[str, Any]) -> int:
    """Resolve one actual pivot and reject stale donor/member sidecars."""

    alignment = branch.get("alignment") if isinstance(branch.get("alignment"), Mapping) else {}
    candidates: list[tuple[str, Any]] = []
    for key in _NATIVE_PIVOT_KEYS:
        if key in branch:
            candidates.append((key, branch[key]))
        if key in alignment:
            candidates.append((f"alignment.{key}", alignment[key]))
    if not candidates:
        raise CausalGroupPackError(
            f"group {group_id} sibling {role} lacks an explicit native pivot frame"
        )
    resolved = {
        _native_frame(value, f"group {group_id} sibling {role} {path}")
        for path, value in candidates
    }
    if len(resolved) != 1:
        raise CausalGroupPackError(
            f"group {group_id} sibling {role} has conflicting native pivot frames"
        )
    pivot = next(iter(resolved))
    donor_candidates: list[tuple[str, Any]] = []
    for key in _DONOR_PIVOT_KEYS:
        if key in branch:
            donor_candidates.append((key, branch[key]))
        if key in alignment:
            donor_candidates.append((f"alignment.{key}", alignment[key]))
    for path, value in donor_candidates:
        donor = _native_frame(value, f"group {group_id} sibling {role} {path}")
        if donor != pivot:
            raise CausalGroupPackError(
                f"group {group_id} sibling {role} donor/member pivot mismatch; "
                "v5 forbids unshifted alignment reuse"
            )
    return pivot


def _certified_group_pivot(
    group_id: str,
    group: Mapping[str, Any],
    branches_by_role: Mapping[str, Mapping[str, Any]],
) -> int:
    pivots = {
        role: _branch_native_pivot(group_id, role, branches_by_role[role])
        for role in CAUSAL_GROUP_ROLES
    }
    if len(set(pivots.values())) != 1:
        raise CausalGroupPackError(
            f"group {group_id} siblings have different native pivot frames; "
            "shared-prefix canonicalization is prohibited"
        )
    pivot = next(iter(pivots.values()))
    declared = _value_from(group, ("nativePivotFrame", "pivotNativeFrame"))
    if declared is not None and _native_frame(declared, f"group {group_id} nativePivotFrame") != pivot:
        raise CausalGroupPackError(f"group {group_id} declared native pivot differs from siblings")
    return pivot


def _extract_control(branch: Mapping[str, Any], group_id: str, role: str) -> dict[str, Any]:
    explicit = branch.get("controlInput")
    if isinstance(explicit, Mapping):
        return deepcopy(dict(explicit))
    if isinstance(branch.get("controlFrame"), Mapping):
        control: dict[str, Any] = {"controlFrame": deepcopy(branch["controlFrame"])}
        if isinstance(branch.get("evidenceFrame"), Mapping):
            control["evidenceFrame"] = deepcopy(branch["evidenceFrame"])
        return control
    if isinstance(branch.get("control"), Mapping):
        return deepcopy(dict(branch["control"]))
    raise CausalGroupPackError(f"group {group_id} sibling {role} lacks typed control input")


def _extract_target(branch: Mapping[str, Any], group_id: str, role: str) -> dict[str, Any]:
    target = deepcopy(dict(branch.get("target"))) if isinstance(branch.get("target"), Mapping) else {}
    labels = branch.get("labels") if isinstance(branch.get("labels"), Mapping) else {}
    text_values = [
        value
        for value in (
            target.get("text"),
            target.get("transcript"),
            branch.get("targetText"),
            labels.get("agentText"),
        )
        if isinstance(value, str) and value.strip()
    ]
    if not text_values:
        raise CausalGroupPackError(f"group {group_id} sibling {role} lacks target text")
    if len({" ".join(_word_sequence(value)) for value in text_values}) != 1:
        raise CausalGroupPackError(f"group {group_id} sibling {role} has conflicting target text")
    target_text = text_values[0].strip()
    audio = _value_from(target, ("audio", "targetAudio", "sourceAudio"))
    if audio is None:
        audio = _value_from(branch, ("targetAudio", "agentAudio", "responseAudio"))
    if audio is None and any(
        key in target for key in ("sha256", "sourceAudioSha256", "audioSha256")
    ):
        audio = target
    if audio is None:
        raise CausalGroupPackError(f"group {group_id} sibling {role} lacks target audio")
    _audio_identity(audio, f"group {group_id} sibling {role} target audio")
    target["text"] = target_text
    target["audio"] = deepcopy(audio)
    if labels:
        target["labels"] = deepcopy(dict(labels))
    return target


def _collect_lineage_identifiers(group: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = group.get("lineageIdentifiers")
    values: set[str] = set()
    if isinstance(explicit, list):
        values.update(_require_text(value, "lineage identifier") for value in explicit)
    elif explicit is not None:
        raise CausalGroupPackError("lineageIdentifiers must be an array")

    def visit(value: Any) -> None:
        if not isinstance(value, Mapping):
            return
        for key, child in value.items():
            normalized = _normalized_key(key)
            if normalized in _LINEAGE_ID_KEYS and isinstance(child, (str, int)):
                values.add(str(child))
            elif isinstance(child, Mapping):
                visit(child)

    visit(group.get("lineage"))
    visit(group.get("cascade"))
    visit(group)
    return tuple(sorted(value for value in values if value))


def _nested_text(mapping: Mapping[str, Any], key: str, names: Sequence[str]) -> str | None:
    nested = mapping.get(key)
    if not isinstance(nested, Mapping):
        return None
    value = _value_from(nested, names)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _consistent_branch_value(branches: Sequence[Mapping[str, Any]], names: Sequence[str]) -> str | None:
    values = {
        str(value).strip()
        for branch in branches
        if (value := _value_from(branch, names)) is not None and str(value).strip()
    }
    if len(values) > 1:
        raise CausalGroupPackError(f"siblings disagree on {names[0]}")
    return next(iter(values), None)


def _group_metadata(
    group_id: str,
    group: Mapping[str, Any],
    branches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lineage = _collect_lineage_identifiers(group)
    if not lineage:
        raise CausalGroupPackError(f"group {group_id} lacks lineage identifiers")

    premise_id = _value_from(group, ("premiseId", "scenarioId"))
    if premise_id is None and isinstance(group.get("lineage"), Mapping):
        premise_id = _value_from(group["lineage"], ("premiseId", "scenarioId"))
    if premise_id is None and isinstance(group.get("cascade"), Mapping):
        premise_id = _value_from(group["cascade"], ("premiseId", "scenarioId"))
    premise_id = _require_text(premise_id, f"group {group_id} premiseId")

    template_id = _value_from(group, ("templateId", "generationTemplateId"))
    template_id = template_id or _nested_text(group, "template", ("id", "templateId"))
    template_id = template_id or _consistent_branch_value(
        branches, ("templateId", "generationTemplateId")
    )
    template_id = _require_text(template_id, f"group {group_id} templateId")

    operator = group.get("controlOperator") if isinstance(group.get("controlOperator"), Mapping) else {}
    family = _value_from(operator, ("family", "interventionFamily"))
    family = family or _value_from(group, ("interventionFamily", "operatorFamily"))
    family = _require_text(family, f"group {group_id} control-operator family")
    changed_paths = _value_from(operator, ("changedPaths", "paths"))
    changed_paths = changed_paths or group.get("changedPaths")
    if changed_paths is None and isinstance(group.get("typedPivot"), Mapping):
        changed_paths = [group["typedPivot"].get("field")]
    if changed_paths is None:
        changed_paths = sorted(
            {
                str(delta["field"])
                for branch in branches
                if isinstance((delta := branch.get("controlDelta")), Mapping)
                and isinstance(delta.get("field"), str)
                and delta["field"]
            }
        )
    if not isinstance(changed_paths, (list, tuple)) or not changed_paths or not all(
        isinstance(path, str) and path.strip() for path in changed_paths
    ):
        raise CausalGroupPackError(f"group {group_id} lacks changed paths")
    normalized_paths = tuple(sorted(set(path.strip() for path in changed_paths)))
    operator_id = _value_from(operator, ("id", "operatorId", "controlOperatorId"))
    operator_id = operator_id or _value_from(group, ("controlOperatorId", "operatorId"))
    operator_id_source = "explicit"
    if operator_id is None and isinstance(group.get("typedPivot"), Mapping):
        operator_id = content_hash(
            {
                "family": family,
                "changedPaths": normalized_paths,
                "typedPivot": group["typedPivot"],
            }
        )
        operator_id_source = "content_derived"
    operator_id = _require_text(operator_id, f"group {group_id} controlOperatorId")

    voice_pair = group.get("voicePair") if isinstance(group.get("voicePair"), Mapping) else {}
    caller_voice = _value_from(voice_pair, ("caller", "callerVoiceId", "callerVoiceReferenceId"))
    agent_voice = _value_from(voice_pair, ("agent", "target", "agentVoiceId", "targetVoiceReferenceId"))
    caller_voice = caller_voice or _value_from(
        group, ("callerVoiceId", "callerVoiceReferenceId")
    )
    agent_voice = agent_voice or _value_from(
        group, ("agentVoiceId", "targetVoiceReferenceId")
    )
    caller_voice = caller_voice or _consistent_branch_value(
        branches, ("callerVoiceId", "callerVoiceReferenceId")
    )
    agent_voice = agent_voice or _consistent_branch_value(
        branches, ("agentVoiceId", "targetVoiceReferenceId")
    )
    caller_voice = _require_text(caller_voice, f"group {group_id} caller voice")
    agent_voice = _require_text(agent_voice, f"group {group_id} agent voice")
    if caller_voice == agent_voice:
        raise CausalGroupPackError(f"group {group_id} must use distinct caller and agent voices")
    voice_pair_id = _value_from(voice_pair, ("id", "voicePairId"))
    voice_pair_id = voice_pair_id or group.get("voicePairId")
    voice_pair_id = str(voice_pair_id or f"{caller_voice}->{agent_voice}")

    return {
        "premiseId": premise_id,
        "lineageIdentifiers": list(lineage),
        "templateId": template_id,
        "controlOperator": {
            "id": operator_id,
            "identifierSource": operator_id_source,
            "family": family,
            "changedPaths": list(normalized_paths),
        },
        "voicePair": {
            "id": voice_pair_id,
            "caller": caller_voice,
            "agent": agent_voice,
        },
    }


def normalize_causal_group(group: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one grouped v5 artifact without assigning a split."""

    if not isinstance(group, Mapping):
        raise CausalGroupPackError("causal group artifact must be an object")
    group_id = _require_text(_value_from(group, ("groupId", "group_id")), "groupId")
    branches = _group_branches(group)
    if len(branches) != len(CAUSAL_GROUP_ROLES):
        raise CausalGroupPackError(f"group {group_id} must contain exactly four siblings")
    by_role: dict[str, Mapping[str, Any]] = {}
    for branch in branches:
        role = _branch_role(branch)
        if role in by_role:
            raise CausalGroupPackError(f"group {group_id} duplicates sibling role {role!r}")
        by_role[role] = branch
    if set(by_role) != set(CAUSAL_GROUP_ROLES):
        raise CausalGroupPackError(
            f"group {group_id} sibling roles must be exactly {list(CAUSAL_GROUP_ROLES)}"
        )

    common = _canonical_common(group_id, group, branches)
    native_pivot_frame = _certified_group_pivot(group_id, group, by_role)
    common["nativePivotFrame"] = native_pivot_frame
    common["pivotPolicy"] = "identical_native_pivot_frame_required"
    common["commonInputId"] = content_hash(
        {
            "audio": _audio_identity(common["audio"], f"group {group_id} common audio"),
            "context": common["context"],
            "nativePivotFrame": native_pivot_frame,
            "pivotPolicy": common["pivotPolicy"],
        }
    )
    metadata = _group_metadata(group_id, group, branches)
    siblings: list[dict[str, Any]] = []
    control_hashes: set[str] = set()
    example_ids: set[str] = set()
    for role in CAUSAL_GROUP_ROLES:
        branch = by_role[role]
        control = _extract_control(branch, group_id, role)
        target = _extract_target(branch, group_id, role)
        assert_no_target_text_leak(target["text"], common["context"], common["audio"], control)
        control_hash = content_hash(control)
        if control_hash in control_hashes:
            raise CausalGroupPackError(f"group {group_id} has identical sibling controls")
        control_hashes.add(control_hash)
        example_id = str(_value_from(branch, ("exampleId", "example_id")) or f"{group_id}:{role}")
        if example_id in example_ids:
            raise CausalGroupPackError(f"group {group_id} duplicates exampleId {example_id}")
        example_ids.add(example_id)
        siblings.append(
            {
                "role": role,
                "exampleId": example_id,
                "controlInput": control,
                "controlInputHash": control_hash,
                "targetLabel": target,
                "targetLabelHash": content_hash(target),
            }
        )

    return {
        "groupId": group_id,
        "sourceSchema": str(group.get("schema") or "unspecified"),
        **metadata,
        "commonInput": common,
        "siblings": siblings,
    }


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        low, high = sorted((left_root, right_root))
        self.parent[high] = low


def _leakage_keys(group: Mapping[str, Any]) -> tuple[str, ...]:
    keys = {f"lineage:{value}" for value in group["lineageIdentifiers"]}
    keys.add(f"template:{group['templateId']}")
    keys.add(f"control-operator:{group['controlOperator']['id']}")
    voice_pair = group["voicePair"]
    keys.add(f"voice-pair-id:{voice_pair['id']}")
    keys.add(f"voice-pair:{voice_pair['caller']}->{voice_pair['agent']}")
    return tuple(sorted(keys))


def build_leakage_components(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return transitive components induced by every protected identifier."""

    group_ids = [str(group["groupId"]) for group in groups]
    if len(group_ids) != len(set(group_ids)):
        raise CausalGroupPackError("groupId values must be unique")
    union_find = _UnionFind(group_ids)
    owner_by_key: dict[str, str] = {}
    keys_by_group: dict[str, tuple[str, ...]] = {}
    for group in sorted(groups, key=lambda item: str(item["groupId"])):
        group_id = str(group["groupId"])
        keys = _leakage_keys(group)
        keys_by_group[group_id] = keys
        for key in keys:
            owner = owner_by_key.setdefault(key, group_id)
            union_find.union(group_id, owner)
    members_by_root: dict[str, list[str]] = {}
    for group_id in group_ids:
        members_by_root.setdefault(union_find.find(group_id), []).append(group_id)
    components: list[dict[str, Any]] = []
    for members in members_by_root.values():
        ordered_members = sorted(members)
        leakage_keys = sorted({key for member in ordered_members for key in keys_by_group[member]})
        component_digest = sha256(canonical_json(ordered_members).encode("utf-8")).hexdigest()[:24]
        components.append(
            {
                "componentId": f"component-{component_digest}",
                "groupIds": ordered_members,
                "groupCount": len(ordered_members),
                "leakageKeys": leakage_keys,
            }
        )
    return sorted(components, key=lambda item: item["componentId"])


def assign_component_splits(
    components: Sequence[Mapping[str, Any]], config: PackConfig
) -> list[dict[str, Any]]:
    """Assign whole components while balancing group counts deterministically."""

    if not components:
        raise CausalGroupPackError("no leakage components were produced")
    total_groups = sum(int(component["groupCount"]) for component in components)
    targets = {name: total_groups * ratio for name, ratio in config.split_ratios}
    assigned = {name: 0 for name, _ in config.split_ratios}
    split_order = [name for name, _ in config.split_ratios]

    def stable_rank(component: Mapping[str, Any]) -> str:
        material = f"{config.split_seed}\0{component['componentId']}".encode("utf-8")
        return sha256(material).hexdigest()

    ordered = sorted(
        components,
        key=lambda item: (-int(item["groupCount"]), stable_rank(item), str(item["componentId"])),
    )
    output: list[dict[str, Any]] = []
    for index, component in enumerate(ordered):
        size = int(component["groupCount"])
        if index < len(split_order):
            split = split_order[index]
        else:
            choices: list[tuple[float, str, str]] = []
            for candidate in split_order:
                projected = dict(assigned)
                projected[candidate] += size
                cost = sum(
                    ((projected[name] - targets[name]) ** 2) / max(targets[name], 1.0)
                    for name in split_order
                )
                tie = sha256(
                    f"{config.split_seed}\0{component['componentId']}\0{candidate}".encode("utf-8")
                ).hexdigest()
                choices.append((cost, tie, candidate))
            split = min(choices)[2]
        assigned[split] += size
        output.append({**deepcopy(dict(component)), "split": split})
    return sorted(output, key=lambda item: item["componentId"])


def _coverage_certificate(
    groups: Sequence[Mapping[str, Any]],
    components: Sequence[Mapping[str, Any]],
    split_by_group: Mapping[str, str],
    config: PackConfig,
) -> dict[str, Any]:
    family_coverage: dict[str, dict[str, set[str]]] = {}
    path_coverage: dict[tuple[str, ...], dict[str, set[str]]] = {}
    for group in groups:
        group_id = str(group["groupId"])
        split = split_by_group[group_id]
        premise = str(group["premiseId"])
        family = str(group["controlOperator"]["family"])
        paths = tuple(group["controlOperator"]["changedPaths"])
        family_bucket = family_coverage.setdefault(
            family, {"groups": set(), "premises": set(), "splits": set()}
        )
        path_bucket = path_coverage.setdefault(
            paths, {"groups": set(), "premises": set(), "splits": set()}
        )
        for bucket in (family_bucket, path_bucket):
            bucket["groups"].add(group_id)
            bucket["premises"].add(premise)
            bucket["splits"].add(split)

    required_splits = set(config.required_coverage_splits)
    reasons: list[str] = []

    def record(label: str, key: Any, bucket: Mapping[str, set[str]]) -> dict[str, Any]:
        missing = sorted(required_splits.difference(bucket["splits"]))
        premise_count = len(bucket["premises"])
        accepted = premise_count >= config.minimum_distinct_premises and not missing
        if premise_count < config.minimum_distinct_premises:
            reasons.append(
                f"{label} {key} has {premise_count} distinct premises; "
                f"requires {config.minimum_distinct_premises}"
            )
        if missing:
            reasons.append(f"{label} {key} is absent from splits {missing}")
        return {
            "groups": len(bucket["groups"]),
            "distinctPremises": premise_count,
            "premiseIds": sorted(bucket["premises"]),
            "splits": sorted(bucket["splits"]),
            "missingRequiredSplits": missing,
            "accepted": accepted,
        }

    families = {
        family: record("operator family", family, family_coverage[family])
        for family in sorted(family_coverage)
    }
    changed_path_signatures = [
        {
            "signature": content_hash({"changedPaths": list(paths)}),
            "changedPaths": list(paths),
            **record("changed-path signature", list(paths), path_coverage[paths]),
        }
        for paths in sorted(path_coverage)
    ]
    split_counts = {
        split: sum(1 for group in groups if split_by_group[str(group["groupId"])] == split)
        for split, _ in config.split_ratios
    }
    for split in config.required_coverage_splits:
        if split_counts[split] == 0:
            reasons.append(f"required split {split} is empty")
    component_splits = {
        str(component["componentId"]): str(component["split"]) for component in components
    }
    component_by_group = {
        str(group_id): str(component["componentId"])
        for component in components
        for group_id in component["groupIds"]
    }
    split_assignments = [
        {
            "groupId": str(group["groupId"]),
            "componentId": component_by_group[str(group["groupId"])],
            "split": split_by_group[str(group["groupId"])],
        }
        for group in sorted(groups, key=lambda item: str(item["groupId"]))
    ]
    reasons = sorted(set(reasons))
    return {
        "schema": CERTIFICATE_SCHEMA,
        "status": "certified" if not reasons else "rejected",
        "groupCount": len(groups),
        "siblingCount": len(groups) * len(CAUSAL_GROUP_ROLES),
        "componentCount": len(components),
        "splitCounts": split_counts,
        "componentSplits": component_splits,
        "splitAssignmentHash": content_hash(split_assignments),
        "requiredSiblingRoles": list(CAUSAL_GROUP_ROLES),
        "coveragePolicy": {
            "requiredSplits": list(config.required_coverage_splits),
            "minimumDistinctPremises": config.minimum_distinct_premises,
        },
        "operatorFamilies": families,
        "changedPathSignatures": changed_path_signatures,
        "nativePivotAlignment": {
            "policy": "identical_native_pivot_frame_required",
            "certifiedGroups": len(groups),
            "shiftedGroups": 0,
        },
        "reasons": reasons,
    }


def pack_causal_groups(
    artifacts: Sequence[Mapping[str, Any]], config: PackConfig | None = None
) -> PackResult:
    """Normalize, component-split, index, and structurally certify v5 groups."""

    config = config or PackConfig()
    normalized = [normalize_causal_group(group) for group in artifacts]
    if not normalized:
        raise CausalGroupPackError("no causal groups were supplied")
    normalized.sort(key=lambda item: item["groupId"])
    if len({group["groupId"] for group in normalized}) != len(normalized):
        raise CausalGroupPackError("groupId values must be unique")

    bare_components = build_leakage_components(normalized)
    components = assign_component_splits(bare_components, config)
    component_by_group: dict[str, str] = {}
    split_by_group: dict[str, str] = {}
    for component in components:
        for group_id in component["groupIds"]:
            component_by_group[group_id] = component["componentId"]
            split_by_group[group_id] = component["split"]

    common_inputs: list[dict[str, Any]] = []
    listwise_groups: list[dict[str, Any]] = []
    pairwise: list[dict[str, Any]] = []
    for group in normalized:
        group_id = group["groupId"]
        common = group["commonInput"]
        common_inputs.append(
            {
                "schema": f"{PACK_SCHEMA}.common-input",
                "groupId": group_id,
                "commonInputId": common["commonInputId"],
                "audio": common["audio"],
                "context": common["context"],
                "nativePivotFrame": common["nativePivotFrame"],
                "pivotPolicy": common["pivotPolicy"],
            }
        )
        siblings = deepcopy(group["siblings"])
        listwise_groups.append(
            {
                "schema": f"{PACK_SCHEMA}.listwise-index",
                "groupId": group_id,
                "split": split_by_group[group_id],
                "componentId": component_by_group[group_id],
                "premiseId": group["premiseId"],
                "lineageIdentifiers": group["lineageIdentifiers"],
                "templateId": group["templateId"],
                "controlOperator": group["controlOperator"],
                "voicePair": group["voicePair"],
                "commonInputRef": {
                    "groupId": group_id,
                    "commonInputId": common["commonInputId"],
                    "nativePivotFrame": common["nativePivotFrame"],
                },
                "siblings": siblings,
            }
        )
        sibling_by_role = {sibling["role"]: sibling for sibling in siblings}
        for left_role, right_role in combinations(CAUSAL_GROUP_ROLES, 2):
            left = sibling_by_role[left_role]
            right = sibling_by_role[right_role]
            edge_identity = {
                "groupId": group_id,
                "leftRole": left_role,
                "rightRole": right_role,
            }
            pairwise.append(
                {
                    "schema": f"{PACK_SCHEMA}.pairwise-diagnostic",
                    "diagnosticOnly": True,
                    "edgeId": content_hash(edge_identity),
                    "groupId": group_id,
                    "split": split_by_group[group_id],
                    "componentId": component_by_group[group_id],
                    "changedPaths": group["controlOperator"]["changedPaths"],
                    "memberA": {"role": left_role, "exampleId": left["exampleId"]},
                    "memberB": {"role": right_role, "exampleId": right["exampleId"]},
                }
            )

    certificate = _coverage_certificate(normalized, components, split_by_group, config)
    return PackResult(
        common_inputs=tuple(common_inputs),
        listwise_groups=tuple(listwise_groups),
        pairwise_diagnostics=tuple(pairwise),
        leakage_components=tuple(deepcopy(components)),
        certificate=certificate,
    )


def parse_artifact_bytes(path: Path, payload: bytes) -> list[dict[str, Any]]:
    """Parse JSONL, a JSON array, one group object, or a ``groups`` wrapper."""

    try:
        text = payload.decode("utf-8")
        if path.suffix.casefold() == ".jsonl":
            values = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                values = parsed
            elif isinstance(parsed, Mapping) and isinstance(parsed.get("groups"), list):
                values = parsed["groups"]
            else:
                values = [parsed]
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CausalGroupPackError(f"cannot parse {path}: {error}") from error
    if not values or not all(isinstance(value, Mapping) for value in values):
        raise CausalGroupPackError(f"{path} must contain one or more group objects")
    return [dict(value) for value in values]


def load_group_artifacts(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read each input once and bind parsed records to immutable byte hashes."""

    if not paths:
        raise CausalGroupPackError("at least one input path is required")
    records: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path in seen_paths:
            raise CausalGroupPackError(f"duplicate input path: {path}")
        seen_paths.add(path)
        if not path.is_file():
            raise CausalGroupPackError(f"input is not a file: {path}")
        payload = path.read_bytes()
        parsed = parse_artifact_bytes(path, payload)
        records.extend(parsed)
        inputs.append(
            {
                "path": str(path),
                "sha256": sha256_bytes(payload),
                "sizeBytes": len(payload),
                "groupRecords": len(parsed),
            }
        )
    return records, inputs


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


def jsonl_bytes(values: Iterable[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json(value) + "\n" for value in values).encode("utf-8")


def split_assignment_records(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Return the canonical trainer-visible group/component/split projection."""

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for group in sorted(groups, key=lambda item: str(item.get("groupId", ""))):
        group_id = _require_text(group.get("groupId"), "packed groupId")
        if group_id in seen:
            raise CausalGroupPackError(f"duplicate packed groupId: {group_id}")
        seen.add(group_id)
        component_id = _require_text(
            group.get("componentId"), f"packed group {group_id} componentId"
        )
        split = _require_text(group.get("split"), f"packed group {group_id} split")
        if split not in {"train", "validation", "test"}:
            raise CausalGroupPackError(f"packed group {group_id} has unsupported split {split!r}")
        records.append(
            {"groupId": group_id, "componentId": component_id, "split": split}
        )
    return records


def _read_bound_file(path: Path, label: str) -> tuple[Path, bytes]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise CausalGroupPackError(f"{label} is not a file: {resolved}")
    try:
        return resolved, resolved.read_bytes()
    except OSError as error:
        raise CausalGroupPackError(f"cannot read {label}: {resolved}") from error


def _artifact_descriptor(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_bytes(payload),
        "sizeBytes": len(payload),
    }


def _json_object(payload: bytes, path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CausalGroupPackError(f"cannot parse {label}: {path}") from error
    if not isinstance(value, Mapping):
        raise CausalGroupPackError(f"{label} must be a JSON object: {path}")
    return value


def _trainer_group_assignments(
    payload: bytes, path: Path
) -> tuple[list[dict[str, str]], int]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CausalGroupPackError(f"trainer group manifest is not UTF-8: {path}") from error
    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise CausalGroupPackError(
                f"trainer group manifest line {line_number} is not valid JSON"
            ) from error
        if not isinstance(value, Mapping):
            raise CausalGroupPackError(
                f"trainer group manifest line {line_number} must be an object"
            )
        if value.get("schema") != TRAINER_GROUP_SCHEMA:
            raise CausalGroupPackError(
                f"trainer group manifest line {line_number} has unsupported schema"
            )
        group_id = _require_text(
            value.get("group_id"), f"trainer group manifest line {line_number} group_id"
        )
        if group_id in seen:
            raise CausalGroupPackError(f"duplicate trainer group_id: {group_id}")
        seen.add(group_id)
        component_id = _require_text(
            value.get("leakage_component_id"),
            f"trainer group manifest line {line_number} leakage_component_id",
        )
        split = _require_text(
            value.get("split"), f"trainer group manifest line {line_number} split"
        )
        if split not in {"train", "validation", "test"}:
            raise CausalGroupPackError(
                f"trainer group manifest line {line_number} has unsupported split {split!r}"
            )
        records.append(
            {"groupId": group_id, "componentId": component_id, "split": split}
        )
    if not records:
        raise CausalGroupPackError("trainer group manifest is empty")
    return sorted(records, key=lambda item: item["groupId"]), len(records)


def prepare_trainer_binding(
    result: PackResult,
    inputs: Sequence[Mapping[str, Any]],
    *,
    trainer_data_contract: Path,
    trainer_group_manifest: Path,
    model_contract: Path,
) -> dict[str, Any]:
    """Bind one exact trainer projection and model contract to a pack result."""

    data_path, data_payload = _read_bound_file(trainer_data_contract, "trainer data contract")
    groups_path, groups_payload = _read_bound_file(
        trainer_group_manifest, "trainer group manifest"
    )
    model_path, model_payload = _read_bound_file(model_contract, "model contract")
    data = _json_object(data_payload, data_path, "trainer data contract")
    model = _json_object(model_payload, model_path, "model contract")
    if data.get("schema") != TRAINER_DATASET_SCHEMA:
        raise CausalGroupPackError("trainer data contract has unsupported schema")
    if data.get("status") != "certified_for_native_moshirag_full_rank_training":
        raise CausalGroupPackError("trainer data contract is not certified for native training")
    groups_descriptor = _artifact_descriptor(groups_path, groups_payload)
    if data.get("manifest_sha256") != groups_descriptor["sha256"]:
        raise CausalGroupPackError("trainer data contract does not bind the trainer group manifest")
    model_revision = _require_text(model.get("model_revision"), "model contract model_revision")
    if data.get("model_revision") != model_revision:
        raise CausalGroupPackError("trainer data contract and model contract revisions differ")

    packed_assignments = split_assignment_records(result.listwise_groups)
    trainer_assignments, trainer_group_count = _trainer_group_assignments(
        groups_payload, groups_path
    )
    if trainer_assignments != packed_assignments:
        packed_by_group = {item["groupId"]: item for item in packed_assignments}
        trainer_by_group = {item["groupId"]: item for item in trainer_assignments}
        missing = sorted(set(packed_by_group) - set(trainer_by_group))
        extra = sorted(set(trainer_by_group) - set(packed_by_group))
        changed = sorted(
            group_id
            for group_id in set(packed_by_group) & set(trainer_by_group)
            if packed_by_group[group_id] != trainer_by_group[group_id]
        )
        raise CausalGroupPackError(
            "trainer group split assignment does not match packed leakage components: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    split_assignment_hash = content_hash(packed_assignments)
    if result.certificate.get("splitAssignmentHash") != split_assignment_hash:
        raise CausalGroupPackError("coverage certificate does not bind the packed split assignment")
    source_inputs = sorted((dict(item) for item in inputs), key=lambda item: item["path"])
    return {
        "schema": TRAINER_BINDING_SCHEMA,
        "sourceGroupInputsHash": content_hash(source_inputs),
        "datasetContract": _artifact_descriptor(data_path, data_payload),
        "groupManifest": {
            **groups_descriptor,
            "groupRecords": trainer_group_count,
        },
        "modelContract": _artifact_descriptor(model_path, model_payload),
        "modelRevision": model_revision,
        "splitAssignmentHash": split_assignment_hash,
    }


def write_immutable(path: Path, payload: bytes) -> None:
    """Create a file once; an identical rerun is allowed but never rewrites it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
    except FileExistsError:
        if path.read_bytes() != payload:
            raise CausalGroupPackError(f"immutable artifact already exists with different content: {path}")


def write_pack_result(
    output_root: Path,
    result: PackResult,
    inputs: Sequence[Mapping[str, Any]],
    config: PackConfig,
    trainer_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Write deterministic pack artifacts and a trainer-bound immutable manifest."""

    root = output_root.expanduser().resolve()
    artifacts: list[tuple[str, bytes]] = [
        ("common_inputs.jsonl", jsonl_bytes(result.common_inputs)),
        ("listwise_groups.jsonl", jsonl_bytes(result.listwise_groups)),
        ("pairwise_diagnostics.jsonl", jsonl_bytes(result.pairwise_diagnostics)),
        ("leakage_components.jsonl", jsonl_bytes(result.leakage_components)),
        ("causal_coverage_certificate.json", json_bytes(result.certificate)),
    ]
    output_descriptors: list[dict[str, Any]] = []
    for name, payload in artifacts:
        write_immutable(root / name, payload)
        output_descriptors.append(
            {"path": name, "sha256": sha256_bytes(payload), "sizeBytes": len(payload)}
        )
    if trainer_binding.get("schema") != TRAINER_BINDING_SCHEMA:
        raise CausalGroupPackError("trainer binding has unsupported schema")
    if trainer_binding.get("splitAssignmentHash") != result.certificate.get(
        "splitAssignmentHash"
    ):
        raise CausalGroupPackError("trainer binding split assignment is stale")
    certificate_descriptor = next(
        item for item in output_descriptors if item["path"] == "causal_coverage_certificate.json"
    )
    bound_trainer = {
        **deepcopy(dict(trainer_binding)),
        "coverageCertificateSha256": certificate_descriptor["sha256"],
    }
    manifest_base = {
        "schema": MANIFEST_SCHEMA,
        "status": result.certificate["status"],
        "inputs": sorted((dict(item) for item in inputs), key=lambda item: item["path"]),
        "configuration": {
            "splitRatios": {name: ratio for name, ratio in config.split_ratios},
            "splitSeed": config.split_seed,
            "requiredCoverageSplits": list(config.required_coverage_splits),
            "minimumDistinctPremises": config.minimum_distinct_premises,
        },
        "counts": {
            "groups": len(result.listwise_groups),
            "siblings": sum(len(group["siblings"]) for group in result.listwise_groups),
            "components": len(result.leakage_components),
            "pairwiseDiagnostics": len(result.pairwise_diagnostics),
        },
        "outputs": output_descriptors,
        "trainerBinding": bound_trainer,
    }
    manifest = {**manifest_base, "manifestId": content_hash(manifest_base)}
    write_immutable(root / "manifest.json", json_bytes(manifest))
    return manifest
