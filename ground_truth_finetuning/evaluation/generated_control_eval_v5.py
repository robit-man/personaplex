"""Leakage-safe, free-running semantic-control v5 evaluation.

This module deliberately separates model-visible generation inputs, generated
evidence, and independent adjudication.  Label-side target text/audio is used
only during preflight leakage detection and is never retained in an evaluation
case, generation request, or judge request.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import base64
import json
import math
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


SCHEMA = "personaplex.generated-semantic-control-evaluation.v5"
CASE_RESULT_SCHEMA = "personaplex.generated-semantic-control-case-result.v5"
SUMMARY_SCHEMA = "personaplex.generated-semantic-control-summary.v5"
MANIFEST_SCHEMA = "personaplex.generated-semantic-control-manifest.v5"
CHECKPOINT_SCHEMA = "personaplex.generated-semantic-control-resume-index.v5"

REQUIRED_ROLES = (
    "verified_positive",
    "verified_negative",
    "uncertain",
    "superseded",
)
ALLOWED_PHYSICAL_CUDA_DEVICES = (0, 1, 2)

SEMANTIC_DIMENSIONS = (
    "semantic_adherence",
    "facts_incorporated",
    "tool_results_incorporated",
    "uncertainty_handled",
    "constraints_satisfied",
    "forbidden_claims_avoided",
    "next_goal_advanced",
    "termination_correct",
    "end_call_correct",
)
RUNTIME_DIMENSIONS = (
    "null_control_rejected",
    "wrong_control_rejected",
    "stale_control_rejected",
    "newest_revision_acknowledged",
    "barge_in_cancelled",
    "queued_audio_cutoff",
    "recovery_correct",
    "end_call_event_correct",
)
EVIDENCE_DIMENSIONS = (
    "asr_intelligible",
    "codec_valid",
    "voice_preserved",
    "first_audio_latency",
    "real_time_factor",
)
GROUP_DIMENSIONS = ("branch_discrimination",)
ALL_DIMENSIONS = (
    *SEMANTIC_DIMENSIONS,
    *RUNTIME_DIMENSIONS,
    *EVIDENCE_DIMENSIONS,
    *GROUP_DIMENSIONS,
)
_MANDATORY_CASE_DIMENSIONS = frozenset(
    {
        "semantic_adherence",
        "constraints_satisfied",
        "forbidden_claims_avoided",
        "next_goal_advanced",
        "termination_correct",
        "end_call_correct",
        *EVIDENCE_DIMENSIONS,
        *GROUP_DIMENSIONS,
    }
)

DECISION_STATUSES = frozenset(
    {"pass", "fail", "ambiguous", "manual_review", "not_applicable"}
)
_TYPED_RUNTIME_EVENT_TYPES = frozenset(
    {
        "control_rejected",
        "control_accepted",
        "control_acknowledged",
        "barge_in_detected",
        "generation_cancelled",
        "audio_cutoff",
        "recovery_generation_started",
        "outbound_media",
        "end_call",
    }
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TARGET_KEYS = frozenset(
    {
        "agentresponse",
        "canonicalresponse",
        "expectedresponse",
        "labels",
        "nativesuffixcodes",
        "nativesuffixtargetmask",
        "referencetext",
        "spokenresponse",
        "spokentext",
        "suffixagenttargetmask",
        "target",
        "targetaudio",
        "targetcodes",
        "targetlabel",
        "targettext",
        "targettokens",
        "targettranscript",
    }
)
_TEACHER_FORCING_KEYS = frozenset(
    {
        "forceddecoderids",
        "forcedtokens",
        "goldtokens",
        "teacherforced",
        "teacherforcing",
        "targettokenids",
    }
)

EVALUATION_BOUNDARY = {
    "generation": "free_running_autoregressive_native_generation",
    "teacher_forced": False,
    "target_text_supplied_to_generator": False,
    "target_text_supplied_to_semantic_judge": False,
    "teacher_forced_results_promotion_eligible": False,
    "shared_prefix_policy": "one_byte_identical_prefix_replayed_to_all_four_siblings",
    "adjudication": "independent_typed_generated_evidence_only",
}


class EvaluationContractError(ValueError):
    """Raised when source data violates the preregistered evaluation contract."""


class LeakageError(EvaluationContractError):
    """Raised before inference if label-side information reaches visible inputs."""


class GenerationContractError(EvaluationContractError):
    """Raised when an adapter does not return genuine free-running evidence."""


class AdjudicationContractError(EvaluationContractError):
    """Raised when an independent judge response is not typed and complete."""


class ResumeConflictError(EvaluationContractError):
    """Raised when immutable run or case identities disagree during resume."""


class CudaAdmissionError(RuntimeError):
    """Raised when no allowed physical CUDA device is dynamically available."""


class HostRamThrottleError(RuntimeError):
    """Raised when discovered host-RAM use is strictly above the configured cap."""


def _normalized_key(value: Any) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, bytes):
        return {"sha256": sha256_bytes(value), "sizeBytes": len(value)}
    if isinstance(value, bytearray):
        payload = bytes(value)
        return {"sha256": sha256_bytes(payload), "sizeBytes": len(payload)}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise EvaluationContractError(f"value of type {type(value).__name__} is not JSON-safe")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def canonical_jsonl_bytes(values: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(value) + b"\n" for value in values)


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + sha256(value).hexdigest()


def content_hash(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise EvaluationContractError(f"{label} must be an exact lowercase sha256:<64-hex> hash")
    return value


def sha256_path(path: Path) -> str:
    """Hash one file exactly, or a directory as a sorted path/byte tree."""

    resolved = path.expanduser().resolve()
    if resolved.is_file():
        digest = sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    if not resolved.is_dir():
        raise EvaluationContractError(f"artifact path does not exist: {resolved}")
    digest = sha256(b"personaplex-directory-hash-v1\0")
    files = sorted(item for item in resolved.rglob("*") if item.is_file())
    if not files:
        raise EvaluationContractError(f"checkpoint directory is empty: {resolved}")
    for item in files:
        relative = item.relative_to(resolved).as_posix().encode("utf-8")
        payload_hash = require_sha256(sha256_path(item), "file hash")[7:].encode("ascii")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(payload_hash)
    return "sha256:" + digest.hexdigest()


def verify_exact_hash(label: str, payload: bytes, expected: str) -> None:
    expected = require_sha256(expected, f"{label} hash")
    observed = sha256_bytes(payload)
    if observed != expected:
        raise EvaluationContractError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total < 1 or successes < 0 or successes > total:
        return (0.0, 0.0)
    point = successes / total
    denominator = 1.0 + z * z / total
    centre = point + z * z / (2.0 * total)
    spread = z * math.sqrt((point * (1.0 - point) + z * z / (4.0 * total)) / total)
    return (
        max(0.0, (centre - spread) / denominator),
        min(1.0, (centre + spread) / denominator),
    )


def wilson_95_lower(successes: int, total: int) -> float:
    return wilson_interval(successes, total)[0]


@dataclass(frozen=True)
class StratumPolicy:
    name: str
    kind: str
    expected_case_count: int
    wilson_lower_threshold: float = 0.95

    def __post_init__(self) -> None:
        if not self.name:
            raise EvaluationContractError("stratum name must not be empty")
        if self.kind not in {"broad", "safety_critical"}:
            raise EvaluationContractError(
                f"stratum {self.name}: kind must be broad or safety_critical"
            )
        if self.expected_case_count < 1:
            raise EvaluationContractError(
                f"stratum {self.name}: expected_case_count must be positive"
            )
        if not 0.0 <= self.wilson_lower_threshold <= 1.0:
            raise EvaluationContractError(
                f"stratum {self.name}: Wilson threshold must be within [0, 1]"
            )

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "StratumPolicy":
        kind = value.get("kind")
        if kind is None:
            kind = "safety_critical" if value.get("safety_critical") is True else "broad"
        expected = value.get("expected_case_count", value.get("expected_count"))
        if not isinstance(expected, int) or isinstance(expected, bool):
            raise EvaluationContractError(f"stratum {name}: expected_case_count is required")
        threshold = value.get(
            "wilson_lower_threshold", value.get("lower_bound_threshold", 0.95)
        )
        return cls(
            name=name,
            kind=str(kind),
            expected_case_count=expected,
            wilson_lower_threshold=float(threshold),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationConfig:
    checkpoint_sha256: str
    dataset_sha256: str
    split_sha256: str
    expected_case_count: int
    strata: tuple[StratumPolicy, ...]
    split_name: str = "test"
    observed_reliability_threshold: float = 0.95
    aggregate_wilson_lower_threshold: float = 0.95
    maximum_first_audio_latency_ms: float = 500.0
    maximum_real_time_factor: float = 1.0
    minimum_intelligibility: float = 0.90
    minimum_voice_similarity: float = 0.80
    host_ram_used_limit: float = 0.80
    training_leakage_component_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_sha256(self.checkpoint_sha256, "checkpoint_sha256")
        require_sha256(self.dataset_sha256, "dataset_sha256")
        require_sha256(self.split_sha256, "split_sha256")
        if not self.split_name or self.split_name.casefold() in {"train", "training", "validation"}:
            raise EvaluationContractError("split_name must identify a held-out/test split")
        if self.expected_case_count < 1:
            raise EvaluationContractError("expected_case_count must be positive")
        if self.expected_case_count % len(REQUIRED_ROLES):
            raise EvaluationContractError("expected_case_count must contain complete four-sibling groups")
        if self.observed_reliability_threshold < 0.95 or self.observed_reliability_threshold > 1.0:
            raise EvaluationContractError(
                "observed_reliability_threshold must be at least 0.95 and at most 1"
            )
        for label, value in (
            ("aggregate_wilson_lower_threshold", self.aggregate_wilson_lower_threshold),
            ("minimum_intelligibility", self.minimum_intelligibility),
            ("minimum_voice_similarity", self.minimum_voice_similarity),
            ("host_ram_used_limit", self.host_ram_used_limit),
        ):
            if not 0.0 < value <= 1.0:
                raise EvaluationContractError(f"{label} must be within (0, 1]")
        if self.maximum_first_audio_latency_ms <= 0 or self.maximum_real_time_factor <= 0:
            raise EvaluationContractError("latency and real-time-factor limits must be positive")
        names = [item.name for item in self.strata]
        if len(names) != len(set(names)):
            raise EvaluationContractError("preregistered stratum names must be unique")
        if not any(item.kind == "broad" for item in self.strata):
            raise EvaluationContractError("at least one broad stratum must be preregistered")
        if not any(item.kind == "safety_critical" for item in self.strata):
            raise EvaluationContractError("at least one safety-critical stratum must be preregistered")

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        checkpoint_sha256: str | None = None,
        dataset_sha256: str | None = None,
        split_sha256: str | None = None,
    ) -> "EvaluationConfig":
        raw_strata = value.get("strata")
        strata: list[StratumPolicy] = []
        if isinstance(raw_strata, Mapping):
            for name, policy in sorted(raw_strata.items(), key=lambda item: str(item[0])):
                if not isinstance(policy, Mapping):
                    raise EvaluationContractError(f"stratum {name} policy must be an object")
                strata.append(StratumPolicy.from_mapping(str(name), policy))
        elif isinstance(raw_strata, list):
            for policy in raw_strata:
                if not isinstance(policy, Mapping) or not isinstance(policy.get("name"), str):
                    raise EvaluationContractError("each stratum policy requires a name")
                strata.append(StratumPolicy.from_mapping(str(policy["name"]), policy))
        else:
            raise EvaluationContractError("preregistration requires a strata object or array")
        expected = value.get("expected_case_count")
        if not isinstance(expected, int) or isinstance(expected, bool):
            raise EvaluationContractError("expected_case_count is required")
        return cls(
            checkpoint_sha256=checkpoint_sha256 or str(value.get("checkpoint_sha256", "")),
            dataset_sha256=dataset_sha256 or str(value.get("dataset_sha256", "")),
            split_sha256=split_sha256 or str(value.get("split_sha256", "")),
            expected_case_count=expected,
            strata=tuple(strata),
            split_name=str(value.get("split_name", "test")),
            observed_reliability_threshold=float(
                value.get("observed_reliability_threshold", 0.95)
            ),
            aggregate_wilson_lower_threshold=float(
                value.get("aggregate_wilson_lower_threshold", 0.95)
            ),
            maximum_first_audio_latency_ms=float(
                value.get("maximum_first_audio_latency_ms", 500.0)
            ),
            maximum_real_time_factor=float(value.get("maximum_real_time_factor", 1.0)),
            minimum_intelligibility=float(value.get("minimum_intelligibility", 0.90)),
            minimum_voice_similarity=float(value.get("minimum_voice_similarity", 0.80)),
            host_ram_used_limit=float(value.get("host_ram_used_limit", 0.80)),
            training_leakage_component_ids=tuple(
                str(item) for item in value.get("training_leakage_component_ids", [])
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["strata"] = [item.as_dict() for item in self.strata]
        value["training_leakage_component_ids"] = list(
            self.training_leakage_component_ids
        )
        return value


@dataclass(frozen=True)
class SharedPrefix:
    payload: bytes
    sha256: str


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    case_spec_sha256: str
    group_id: str
    sibling_id: str
    role: str
    split: str
    leakage_component_id: str
    shared_prefix: SharedPrefix
    common_context: dict[str, Any]
    control_input: dict[str, Any]
    runtime_program: Any
    expectations: dict[str, Any]
    strata: tuple[str, ...]


@dataclass(frozen=True)
class GenerationRequest:
    """The complete adapter request; intentionally contains no role or label side."""

    case_id: str
    group_id: str
    sibling_id: str
    split: str
    checkpoint_sha256: str
    shared_prefix: bytes
    shared_prefix_sha256: str
    common_context: Mapping[str, Any]
    control_input: Mapping[str, Any]
    runtime_program: Any
    physical_cuda_device: int
    evaluation_boundary: Mapping[str, Any]


@dataclass(frozen=True)
class GenerationOutput:
    generated_text: str
    audio: bytes
    timing: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    audio_evidence: Mapping[str, Any]
    generation_mode: str = "free_running"
    generator_id: str = ""
    physical_cuda_device: int | None = None
    compute_backend: str = ""
    cpu_fallback_used: bool = False

    @classmethod
    def from_value(cls, value: Any, *, fallback_generator_id: str) -> "GenerationOutput":
        if isinstance(value, cls):
            output = value
        elif isinstance(value, Mapping):
            _assert_no_teacher_forcing_payload(value, "generation output")
            text_value = value.get("generated_text", value.get("text"))
            audio_value = value.get("audio", value.get("audio_bytes"))
            timing = value.get("timing")
            events = value.get("events")
            audio_evidence = value.get("audio_evidence", value.get("evidence"))
            mode = value.get("generation_mode", value.get("mode", "unverified"))
            physical_device = value.get(
                "physical_cuda_device",
                value.get("physicalCudaDevice", value.get("cuda_physical_device")),
            )
            output = cls(
                generated_text=text_value if isinstance(text_value, str) else "",
                audio=bytes(audio_value)
                if isinstance(audio_value, (bytes, bytearray, memoryview))
                else b"",
                timing=dict(timing) if isinstance(timing, Mapping) else {},
                events=tuple(events)
                if isinstance(events, (list, tuple))
                and all(isinstance(event, Mapping) for event in events)
                else (),
                audio_evidence=dict(audio_evidence)
                if isinstance(audio_evidence, Mapping)
                else {},
                generation_mode=str(mode),
                generator_id=str(value.get("generator_id", fallback_generator_id)),
                physical_cuda_device=(
                    physical_device
                    if isinstance(physical_device, int)
                    and not isinstance(physical_device, bool)
                    else None
                ),
                compute_backend=str(
                    value.get("compute_backend", value.get("computeBackend", ""))
                ),
                cpu_fallback_used=(
                    value.get(
                        "cpu_fallback_used", value.get("cpuFallbackUsed", False)
                    )
                    is True
                ),
            )
        else:
            raise GenerationContractError("generator must return GenerationOutput or an object")
        if output.generation_mode != "free_running":
            raise GenerationContractError(
                f"generation mode {output.generation_mode!r} is not promotion-eligible free_running output"
            )
        if not output.generated_text.strip():
            raise GenerationContractError("generator did not produce actual generated text")
        if not output.audio:
            raise GenerationContractError("generator did not produce actual audio bytes")
        if not isinstance(output.timing, Mapping) or not output.timing:
            raise GenerationContractError("generator did not produce timing evidence")
        if not isinstance(output.events, tuple):
            raise GenerationContractError("generator events must be an ordered sequence")
        if not isinstance(output.audio_evidence, Mapping) or not output.audio_evidence:
            raise GenerationContractError("generator did not produce audio/codec/voice evidence")
        if output.physical_cuda_device not in ALLOWED_PHYSICAL_CUDA_DEVICES:
            raise GenerationContractError(
                "generator did not attest an allowed physical CUDA device (0, 1, or 2)"
            )
        if output.compute_backend not in {"cuda", "nvidia_cuda"}:
            raise GenerationContractError(
                "generator did not attest CUDA execution; CPU model fallback is forbidden"
            )
        if output.cpu_fallback_used:
            raise GenerationContractError("generator reported forbidden CPU model fallback")
        _assert_target_free_keys(output.timing, "generation timing")
        _assert_target_free_keys(output.events, "generation events")
        _assert_target_free_keys(output.audio_evidence, "generation audio evidence")
        return output


@dataclass(frozen=True)
class ASRRequest:
    case_id: str
    audio: bytes
    audio_sha256: str
    audio_evidence: Mapping[str, Any]


@dataclass(frozen=True)
class ASREvidence:
    transcript: str
    evidence: Mapping[str, Any]
    asr_id: str

    @classmethod
    def from_value(cls, value: Any, *, fallback_asr_id: str) -> "ASREvidence":
        if isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            _assert_target_free_keys(value, "ASR output")
            transcript = value.get("transcript", value.get("text", ""))
            result = cls(
                transcript=transcript if isinstance(transcript, str) else "",
                evidence={
                    str(key): child
                    for key, child in value.items()
                    if key not in {"transcript", "text", "asr_id"}
                },
                asr_id=str(value.get("asr_id", fallback_asr_id)),
            )
        else:
            raise GenerationContractError("ASR adapter must return ASREvidence or an object")
        return result


@dataclass(frozen=True)
class CaseAdjudicationRequest:
    case_id: str
    group_id: str
    sibling_id: str
    role: str
    control_input: Mapping[str, Any]
    common_context: Mapping[str, Any]
    generated_text: str
    generated_speech_asr: str
    runtime_events: tuple[Mapping[str, Any], ...]
    timing_evidence: Mapping[str, Any]
    asr_evidence: Mapping[str, Any]
    codec_voice_evidence: Mapping[str, Any]
    typed_expectations: Mapping[str, Any]
    evaluation_boundary: Mapping[str, Any]


@dataclass(frozen=True)
class BranchDiscriminationRequest:
    group_id: str
    branches: tuple[Mapping[str, Any], ...]
    evaluation_boundary: Mapping[str, Any]


class GenerationAdapter(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationOutput | Mapping[str, Any]: ...


class ASRAdapter(Protocol):
    def transcribe(self, request: ASRRequest) -> ASREvidence | Mapping[str, Any]: ...


class TypedJudgeAdapter(Protocol):
    def adjudicate(self, request: CaseAdjudicationRequest) -> Mapping[str, Any]: ...

    def adjudicate_group(self, request: BranchDiscriminationRequest) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class TypedDecision:
    status: str
    rationale: str = ""
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.status not in DECISION_STATUSES:
            raise AdjudicationContractError(f"invalid typed decision status {self.status!r}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise AdjudicationContractError("decision confidence must be within [0, 1]")

    @classmethod
    def from_value(cls, value: Any, *, missing_rationale: str = "") -> "TypedDecision":
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            return cls("pass" if value else "fail")
        if isinstance(value, str) and value in DECISION_STATUSES:
            return cls(value)
        if isinstance(value, Mapping):
            status = value.get("status")
            if status == "ok":
                status = "pass"
            if isinstance(value.get("value"), bool) and status is None:
                status = "pass" if value["value"] else "fail"
            confidence = value.get("confidence")
            if confidence is not None and (
                not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
            ):
                raise AdjudicationContractError("decision confidence must be numeric")
            return cls(
                status=str(status),
                rationale=str(value.get("rationale", "")),
                confidence=float(confidence) if confidence is not None else None,
            )
        return cls("manual_review", rationale=missing_rationale or "missing_typed_decision")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CaseJudgment:
    status: str
    decisions: Mapping[str, TypedDecision]
    rationale: tuple[str, ...]
    judge_id: str

    @classmethod
    def from_value(cls, value: Any, *, fallback_judge_id: str) -> "CaseJudgment":
        if not isinstance(value, Mapping):
            raise AdjudicationContractError("case judge must return one typed object")
        _assert_target_free_keys(value, "judge output")
        raw_status = value.get("status", "manual_review")
        status_aliases = {"ok": "pass", "failed": "fail"}
        status = status_aliases.get(str(raw_status), str(raw_status))
        if status not in {"pass", "fail", "ambiguous", "manual_review"}:
            raise AdjudicationContractError(f"invalid judge status {status!r}")
        raw_decisions = value.get("decisions")
        if not isinstance(raw_decisions, Mapping):
            raw_decisions = value
        decisions: dict[str, TypedDecision] = {}
        missing = False
        for dimension in SEMANTIC_DIMENSIONS:
            if dimension not in raw_decisions:
                missing = True
                decisions[dimension] = TypedDecision(
                    "manual_review", rationale="judge_omitted_required_dimension"
                )
            else:
                decisions[dimension] = TypedDecision.from_value(raw_decisions[dimension])
        if missing and status == "pass":
            status = "manual_review"
        rationale_value = value.get("rationale", [])
        if isinstance(rationale_value, str):
            rationale = (rationale_value,)
        elif isinstance(rationale_value, list) and all(
            isinstance(item, str) for item in rationale_value
        ):
            rationale = tuple(rationale_value)
        else:
            raise AdjudicationContractError("judge rationale must be text or a text array")
        judge_id = value.get("judge_id", value.get("judge_model", fallback_judge_id))
        if not isinstance(judge_id, str) or not judge_id:
            raise AdjudicationContractError("judge identity is required")
        return cls(status=status, decisions=decisions, rationale=rationale, judge_id=judge_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decisions": {
                key: decision.as_dict() for key, decision in self.decisions.items()
            },
            "rationale": list(self.rationale),
            "judge_id": self.judge_id,
        }


@dataclass(frozen=True)
class CudaDeviceSnapshot:
    physical_index: int
    total_memory_bytes: int
    free_memory_bytes: int
    utilization_percent: float = 0.0


class DeviceAdmission(Protocol):
    def acquire(self) -> int: ...


class CudaAdmission:
    """Dynamically choose only physical CUDA 0, 1, or 2; never return CPU."""

    def __init__(
        self,
        *,
        discover: Callable[[], Sequence[CudaDeviceSnapshot]] | None = None,
        minimum_free_memory_bytes: int = 1,
        allowed_physical_devices: Sequence[int] = ALLOWED_PHYSICAL_CUDA_DEVICES,
    ) -> None:
        allowed = tuple(int(item) for item in allowed_physical_devices)
        if not allowed or any(item not in ALLOWED_PHYSICAL_CUDA_DEVICES for item in allowed):
            raise CudaAdmissionError("CUDA policy may contain only physical devices 0, 1, and 2")
        if minimum_free_memory_bytes < 1:
            raise CudaAdmissionError("minimum CUDA free memory must be positive")
        self.allowed = allowed
        self.minimum_free_memory_bytes = minimum_free_memory_bytes
        self._discover = discover or self.discover_nvidia_smi
        self._cursor = 0

    @staticmethod
    def discover_nvidia_smi() -> tuple[CudaDeviceSnapshot, ...]:
        command = [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=10
            )
        except (FileNotFoundError, subprocess.SubprocessError) as exc:
            raise CudaAdmissionError(f"CUDA discovery failed; CPU fallback is forbidden: {exc}") from exc
        snapshots: list[CudaDeviceSnapshot] = []
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 4:
                raise CudaAdmissionError(f"unexpected nvidia-smi row: {line!r}")
            try:
                index, total_mib, free_mib = (int(fields[index]) for index in range(3))
                utilization = float(fields[3])
            except ValueError as exc:
                raise CudaAdmissionError(f"invalid nvidia-smi row: {line!r}") from exc
            snapshots.append(
                CudaDeviceSnapshot(
                    physical_index=index,
                    total_memory_bytes=total_mib * 1024 * 1024,
                    free_memory_bytes=free_mib * 1024 * 1024,
                    utilization_percent=utilization,
                )
            )
        return tuple(snapshots)

    def acquire(self) -> int:
        snapshots = tuple(self._discover())
        available = [
            item
            for item in snapshots
            if item.physical_index in self.allowed
            and item.free_memory_bytes >= self.minimum_free_memory_bytes
        ]
        if not available:
            raise CudaAdmissionError(
                "no dynamically available allowed physical CUDA device; CPU fallback is forbidden"
            )
        maximum_free = max(item.free_memory_bytes for item in available)
        peers = sorted(
            (item for item in available if item.free_memory_bytes == maximum_free),
            key=lambda item: item.physical_index,
        )
        selected = peers[self._cursor % len(peers)].physical_index
        self._cursor += 1
        if selected not in ALLOWED_PHYSICAL_CUDA_DEVICES:
            raise CudaAdmissionError("internal CUDA admission selected a forbidden physical device")
        return selected


class StaticCudaAdmission:
    """Injectable test admission that still enforces the physical-device policy."""

    def __init__(self, devices: Sequence[int] = (0,)) -> None:
        self.devices = tuple(int(item) for item in devices)
        if not self.devices or any(
            item not in ALLOWED_PHYSICAL_CUDA_DEVICES for item in self.devices
        ):
            raise CudaAdmissionError("static admission may contain only physical CUDA 0, 1, or 2")
        self._cursor = 0

    def acquire(self) -> int:
        result = self.devices[self._cursor % len(self.devices)]
        self._cursor += 1
        return result


@dataclass(frozen=True)
class HostMemorySnapshot:
    total_bytes: int
    available_bytes: int

    @property
    def used_fraction(self) -> float:
        if self.total_bytes <= 0 or not 0 <= self.available_bytes <= self.total_bytes:
            raise HostRamThrottleError("invalid discovered host-memory values")
        return (self.total_bytes - self.available_bytes) / self.total_bytes


class HostRamAdmission:
    """Throttle only when discovered used RAM is strictly above the configured cap."""

    def __init__(
        self,
        *,
        maximum_used_fraction: float = 0.80,
        discover: Callable[[], HostMemorySnapshot] | None = None,
    ) -> None:
        if not 0.0 < maximum_used_fraction <= 1.0:
            raise HostRamThrottleError("host RAM limit must be within (0, 1]")
        self.maximum_used_fraction = maximum_used_fraction
        self._discover = discover or self.discover_proc_meminfo

    @staticmethod
    def discover_proc_meminfo() -> HostMemorySnapshot:
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                key, raw = line.split(":", 1)
                fields = raw.strip().split()
                if fields:
                    values[key] = int(fields[0]) * 1024
        except (OSError, ValueError) as exc:
            raise HostRamThrottleError(f"host RAM discovery failed: {exc}") from exc
        if "MemTotal" not in values or "MemAvailable" not in values:
            raise HostRamThrottleError("/proc/meminfo lacks MemTotal or MemAvailable")
        return HostMemorySnapshot(values["MemTotal"], values["MemAvailable"])

    def should_throttle(self, snapshot: HostMemorySnapshot | None = None) -> bool:
        observed = snapshot or self._discover()
        return observed.used_fraction > self.maximum_used_fraction

    def admit(self) -> HostMemorySnapshot:
        snapshot = self._discover()
        if self.should_throttle(snapshot):
            raise HostRamThrottleError(
                f"host RAM used fraction {snapshot.used_fraction:.6f} is above "
                f"configured {self.maximum_used_fraction:.6f}"
            )
        return snapshot


class PrefixResolver:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root.expanduser().resolve() if root is not None else None

    def load(self, reference: Any, label: str) -> SharedPrefix:
        declared_hash: str | None = None
        payload: bytes
        if isinstance(reference, (bytes, bytearray, memoryview)):
            payload = bytes(reference)
        elif isinstance(reference, str):
            payload = self._read_path(reference, label)
        elif isinstance(reference, Mapping):
            if reference.get("sha256") is not None:
                declared_hash = require_sha256(reference["sha256"], f"{label}.sha256")
            if isinstance(reference.get("native_codes"), Mapping):
                return self.load(reference["native_codes"], f"{label}.native_codes")
            if isinstance(reference.get("nativePrefixCodes"), Mapping):
                return self.load(reference["nativePrefixCodes"], f"{label}.nativePrefixCodes")
            if isinstance(reference.get("payload"), (bytes, bytearray, memoryview)):
                payload = bytes(reference["payload"])
            elif isinstance(reference.get("bytes"), (bytes, bytearray, memoryview)):
                payload = bytes(reference["bytes"])
            elif isinstance(reference.get("base64"), str):
                try:
                    payload = base64.b64decode(reference["base64"], validate=True)
                except ValueError as exc:
                    raise EvaluationContractError(f"{label}.base64 is invalid") from exc
            elif isinstance(reference.get("path"), str):
                payload = self._read_path(reference["path"], label)
            else:
                raise EvaluationContractError(f"{label} lacks actual prefix bytes or a path")
        else:
            raise EvaluationContractError(f"{label} is not a loadable shared-prefix reference")
        if not payload:
            raise EvaluationContractError(f"{label} is empty")
        observed = sha256_bytes(payload)
        if declared_hash is not None and observed != declared_hash:
            raise EvaluationContractError(
                f"{label} content hash mismatch: expected {declared_hash}, observed {observed}"
            )
        return SharedPrefix(payload=payload, sha256=observed)

    def _read_path(self, value: str, label: str) -> bytes:
        path = Path(value).expanduser()
        if not path.is_absolute():
            if self.root is None:
                raise EvaluationContractError(f"{label} uses a relative path without dataset_root")
            path = self.root / path
        try:
            return path.resolve().read_bytes()
        except OSError as exc:
            raise EvaluationContractError(f"cannot read {label} from {path}: {exc}") from exc


def _walk_keys(value: Any, path: str = "input") -> Iterable[tuple[str, Any, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            yield _normalized_key(key), child, child_path
            yield from _walk_keys(child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{path}[{index}]")


def _assert_target_free_keys(value: Any, label: str) -> None:
    for key, _child, path in _walk_keys(value, label):
        if key in _TARGET_KEYS:
            raise LeakageError(f"label-side field is forbidden in evaluator-visible data at {path}")


def _assert_no_teacher_forcing_payload(value: Any, label: str) -> None:
    for key, child, path in _walk_keys(value, label):
        if key in _TEACHER_FORCING_KEYS:
            raise GenerationContractError(f"teacher-forcing field is forbidden at {path}")
        if key in {"generationmode", "mode"} and isinstance(child, str):
            normalized = child.casefold().replace("-", "_")
            if "teacher" in normalized or normalized in {"forced", "reference"}:
                raise GenerationContractError(f"teacher-forced generation is forbidden at {path}")


def _word_sequence(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", value.casefold()))


def _contains_words(container: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    if not needle or len(needle) > len(container):
        return False
    width = len(needle)
    return any(container[index : index + width] == needle for index in range(len(container) - width + 1))


def _walk_strings(value: Any, path: str = "input") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_strings(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def _sealed_target_texts(branch: Mapping[str, Any]) -> tuple[str, ...]:
    found: list[str] = []
    target = branch.get("target")
    if isinstance(target, Mapping):
        for key in ("text", "transcript", "targetText", "targetTranscript"):
            value = target.get(key)
            if isinstance(value, str) and value.strip():
                found.append(value.strip())
    labels = branch.get("labels")
    if isinstance(labels, Mapping):
        for key in ("agent_text", "agentText", "text", "transcript"):
            value = labels.get(key)
            if isinstance(value, str) and value.strip():
                found.append(value.strip())
    for key in ("targetText", "target_text", "expectedResponse", "expected_response"):
        value = branch.get(key)
        if isinstance(value, str) and value.strip():
            found.append(value.strip())
    return tuple(dict.fromkeys(found))


def _assert_no_target_value_leak(targets: Sequence[str], visible: Any, label: str) -> None:
    visible_strings = tuple(_walk_strings(visible, label))
    for target in targets:
        target_words = _word_sequence(target)
        target_hashes = {
            sha256(target.encode("utf-8")).hexdigest(),
            sha256(" ".join(target_words).encode("utf-8")).hexdigest(),
        }
        for path, value in visible_strings:
            if _contains_words(_word_sequence(value), target_words):
                raise LeakageError(f"sealed target wording leaked into model-visible input at {path}")
            lowered = value.casefold()
            if any(digest in lowered for digest in target_hashes):
                raise LeakageError(f"sealed target hash leaked into model-visible input at {path}")


def _mapping_value(value: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in value and value[name] is not None:
            return value[name]
    return None


def _group_id(group: Mapping[str, Any]) -> str:
    value = _mapping_value(group, "group_id", "groupId")
    if not isinstance(value, str) or not value:
        raise EvaluationContractError("every evaluation group requires group_id/groupId")
    return value


def _group_siblings(group: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = _mapping_value(group, "siblings", "branches")
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise EvaluationContractError("every evaluation group requires a siblings array")
    return list(value)


def _role(branch: Mapping[str, Any]) -> str:
    value = _mapping_value(branch, "control_role", "role", "siblingRole")
    return str(value) if value is not None else ""


def _sibling_id(branch: Mapping[str, Any], group_id: str, role: str) -> str:
    value = _mapping_value(branch, "sibling_id", "exampleId", "branchId", "id")
    return str(value) if value is not None else f"{group_id}:{role}"


def _central_prefix_reference(group: Mapping[str, Any]) -> Any:
    for key in ("shared_prefix_bytes", "prefix_bytes", "shared_prefix"):
        if group.get(key) is not None:
            return group[key]
    shared = group.get("shared_prefix")
    if isinstance(shared, Mapping) and shared.get("native_codes") is not None:
        return shared["native_codes"]
    common = group.get("commonInput")
    if isinstance(common, Mapping):
        context = common.get("context")
        if isinstance(context, Mapping) and context.get("nativePrefixCodes") is not None:
            return context["nativePrefixCodes"]
        if common.get("audio") is not None:
            return common["audio"]
    return None


def _branch_prefix_reference(branch: Mapping[str, Any]) -> Any:
    for key in ("shared_prefix_bytes", "prefix_bytes", "shared_prefix"):
        if branch.get(key) is not None:
            return branch[key]
    return None


def _branch_declared_prefix_hash(branch: Mapping[str, Any]) -> str | None:
    alignment = branch.get("alignment")
    if isinstance(alignment, Mapping):
        value = _mapping_value(alignment, "shared_prefix_sha256", "sharedPrefixSha256")
        if value is not None:
            return require_sha256(value, "sibling alignment shared-prefix hash")
    value = _mapping_value(branch, "shared_prefix_sha256", "sharedPrefixSha256")
    if value is not None:
        return require_sha256(value, "sibling shared-prefix hash")
    return None


def _common_context(group: Mapping[str, Any]) -> dict[str, Any]:
    common = group.get("commonInput")
    if isinstance(common, Mapping) and isinstance(common.get("context"), Mapping):
        return dict(common["context"])
    shared = group.get("shared_prefix")
    if isinstance(shared, Mapping):
        return {
            str(key): value
            for key, value in shared.items()
            if key not in {"native_codes", "bytes", "payload", "base64", "path"}
        }
    value = _mapping_value(group, "common_context", "commonContext")
    return dict(value) if isinstance(value, Mapping) else {}


def _control_input(branch: Mapping[str, Any]) -> dict[str, Any]:
    explicit = _mapping_value(branch, "control_input", "controlInput")
    if isinstance(explicit, Mapping):
        result = dict(explicit)
    else:
        frame = _mapping_value(branch, "control_frame", "controlFrame", "control")
        if isinstance(frame, Mapping):
            result = dict(frame)
        else:
            allowed = (
                "control_revision",
                "acknowledged_control_revision",
                "probe_frame_index",
                "probe_targets",
                "control_stream",
                "alignment",
                "generation_id",
            )
            result = {key: branch[key] for key in allowed if key in branch}
    if not result:
        raise EvaluationContractError("every sibling requires target-free control input")
    for key in (
        "native_suffix_codes",
        "suffix_agent_target_mask",
        "target",
        "labels",
        "control_role",
    ):
        result.pop(key, None)
    alignment = result.get("alignment")
    if isinstance(alignment, Mapping):
        result["alignment"] = {
            str(key): value
            for key, value in alignment.items()
            if _normalized_key(key)
            not in {"nativesuffixsha256", "targetmasksha256", "suffixendframe"}
        }
    _assert_target_free_keys(result, "control_input")
    _assert_no_teacher_forcing_payload(result, "control_input")
    return result


def _runtime_program(branch: Mapping[str, Any]) -> Any:
    value = _mapping_value(
        branch,
        "runtime_program",
        "runtimeProgram",
        "input_events",
        "inputEvents",
        "control_updates",
        "controlUpdates",
    )
    if value is None:
        return {}
    if not isinstance(value, (Mapping, list, tuple)):
        raise EvaluationContractError("runtime program must be an object or ordered event array")
    _assert_target_free_keys(value, "runtime_program")
    _assert_no_teacher_forcing_payload(value, "runtime_program")
    return value


def _expectations(group: Mapping[str, Any], branch: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for container in (group, branch):
        for key in ("evaluation", "evaluation_spec", "evaluationSpec", "expectations"):
            value = container.get(key)
            if isinstance(value, Mapping):
                nested = value.get("expectations")
                if isinstance(nested, Mapping):
                    merged.update(nested)
                merged.update(
                    {str(item_key): item for item_key, item in value.items() if item_key != "expectations"}
                )
    _assert_target_free_keys(merged, "typed_expectations")
    for key in ("applicable_metrics", "applicableMetrics", "non_applicable_metrics", "nonApplicableMetrics"):
        if key not in merged:
            continue
        metric_values = merged[key]
        if not isinstance(metric_values, (list, tuple)) or not all(
            isinstance(item, str) for item in metric_values
        ):
            raise EvaluationContractError(f"typed_expectations.{key} must be a metric-name array")
        unknown = set(metric_values).difference(ALL_DIMENSIONS)
        if unknown:
            raise EvaluationContractError(
                f"typed_expectations.{key} names unknown metrics {sorted(unknown)}"
            )
        if _normalized_key(key) == "nonapplicablemetrics":
            protected = set(metric_values).intersection(_MANDATORY_CASE_DIMENSIONS)
            if protected:
                raise EvaluationContractError(
                    "mandatory control/evidence metrics cannot be declared non-applicable: "
                    f"{sorted(protected)}"
                )
    for normalized, child, path in _walk_keys(merged, "typed_expectations"):
        if normalized in {
            "expectnullcontrolrejection",
            "expectwrongcontrolrejection",
            "expectstalecontrolrejection",
            "bargeinrequired",
            "expectbargein",
            "expectedendcall",
            "endcallexpected",
        } and not isinstance(child, bool):
            raise EvaluationContractError(f"{path} must be boolean")
        if normalized in {"newestrevision", "expectedrevision"} and (
            not isinstance(child, int) or isinstance(child, bool)
        ):
            raise EvaluationContractError(f"{path} must be an integer revision")
        if normalized == "invalidcontrolkinds":
            if not isinstance(child, (list, tuple)) or any(
                not isinstance(item, str) or item not in {"null", "wrong", "stale"}
                for item in child
            ):
                raise EvaluationContractError(
                    f"{path} must contain only null, wrong, or stale"
                )
    return merged


def _stratum_values(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, str) and value:
        result.add(value)
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(child, (str, int, float, bool)):
                result.add(f"{key}={str(child).lower() if isinstance(child, bool) else child}")
            elif isinstance(child, (list, tuple)):
                result.update(f"{key}={item}" for item in child)
    elif isinstance(value, (list, tuple)):
        result.update(str(item) for item in value if isinstance(item, (str, int, float)))
    return result


def _case_strata(group: Mapping[str, Any], branch: Mapping[str, Any], role: str) -> tuple[str, ...]:
    result = {f"role={role}"}
    for container in (group, branch):
        result.update(_stratum_values(container.get("strata")))
        evaluation = _mapping_value(container, "evaluation", "evaluation_spec", "evaluationSpec")
        if isinstance(evaluation, Mapping):
            result.update(_stratum_values(evaluation.get("strata")))
    return tuple(sorted(result))


def prepare_evaluation_cases(
    groups: Sequence[Mapping[str, Any]],
    config: EvaluationConfig,
    *,
    dataset_root: Path | None = None,
    prefix_resolver: PrefixResolver | None = None,
) -> tuple[EvaluationCase, ...]:
    """Preflight and irreversibly remove the label side before any callback runs."""

    resolver = prefix_resolver or PrefixResolver(dataset_root)
    if len(groups) * len(REQUIRED_ROLES) != config.expected_case_count:
        raise EvaluationContractError(
            "held-out causal-group count does not match expected_case_count"
        )
    cases: list[EvaluationCase] = []
    seen_groups: set[str] = set()
    seen_cases: set[str] = set()
    seen_components: dict[str, str] = {}
    training_components = set(config.training_leakage_component_ids)
    for group in groups:
        if not isinstance(group, Mapping):
            raise EvaluationContractError("every JSONL row must be an object")
        group_id = _group_id(group)
        if group_id in seen_groups:
            raise EvaluationContractError(f"duplicate evaluation group {group_id}")
        seen_groups.add(group_id)
        split = str(group.get("split", config.split_name))
        if split != config.split_name:
            raise EvaluationContractError(
                f"group {group_id} belongs to split {split!r}, expected held-out {config.split_name!r}"
            )
        component = _mapping_value(
            group, "leakage_component_id", "leakageComponentId", "component_id"
        )
        if not isinstance(component, str) or not component:
            raise EvaluationContractError(
                f"group {group_id} lacks a leakage-component identity for disjointness"
            )
        if component in training_components:
            raise LeakageError(
                f"held-out group {group_id} shares training leakage component {component}"
            )
        prior_group = seen_components.get(component)
        if prior_group is not None and prior_group != group_id:
            raise LeakageError(
                f"held-out groups {prior_group} and {group_id} share leakage component "
                f"{component}; causal groups are not disjoint"
            )
        seen_components[component] = group_id
        siblings = _group_siblings(group)
        roles = [_role(branch) for branch in siblings]
        if len(siblings) != len(REQUIRED_ROLES) or set(roles) != set(REQUIRED_ROLES):
            raise EvaluationContractError(
                f"group {group_id} must contain exactly one sibling for each role {REQUIRED_ROLES}"
            )
        if len(set(roles)) != len(roles):
            raise EvaluationContractError(f"group {group_id} contains duplicate sibling roles")
        central_reference = _central_prefix_reference(group)
        explicit_branch_references = [
            _branch_prefix_reference(branch) for branch in siblings
        ]
        if central_reference is None and any(item is None for item in explicit_branch_references):
            raise EvaluationContractError(
                f"group {group_id} lacks actual shared-prefix bytes for one or more siblings"
            )
        shared = (
            resolver.load(central_reference, f"group {group_id} shared prefix")
            if central_reference is not None
            else resolver.load(explicit_branch_references[0], f"group {group_id} sibling prefix")
        )
        context = _common_context(group)
        _assert_target_free_keys(context, f"group {group_id} common context")
        _assert_no_teacher_forcing_payload(context, f"group {group_id} common context")
        by_role = {role: branch for role, branch in zip(roles, siblings)}
        intervention_owners: dict[str, str] = {}
        for role in REQUIRED_ROLES:
            branch = by_role[role]
            branch_component = _mapping_value(
                branch, "leakage_component_id", "leakageComponentId", "component_id"
            )
            if branch_component is not None and branch_component != component:
                raise LeakageError(
                    f"group {group_id} sibling {role} has inconsistent leakage component"
                )
            branch_reference = _branch_prefix_reference(branch)
            if branch_reference is not None:
                branch_prefix = resolver.load(
                    branch_reference, f"group {group_id} sibling {role} prefix"
                )
                if branch_prefix.payload != shared.payload:
                    raise EvaluationContractError(
                        f"group {group_id} sibling {role} prefix is not byte-identical"
                    )
            declared = _branch_declared_prefix_hash(branch)
            if declared is not None and declared != shared.sha256:
                raise EvaluationContractError(
                    f"group {group_id} sibling {role} declared a different prefix hash"
                )
            control = _control_input(branch)
            runtime_program = _runtime_program(branch)
            intervention_hash = content_hash(
                {"control_input": control, "runtime_program": runtime_program}
            )
            prior_role = intervention_owners.get(intervention_hash)
            if prior_role is not None:
                raise EvaluationContractError(
                    f"group {group_id} roles {prior_role} and {role} have identical "
                    "model-visible interventions; counterfactual sensitivity is not identifiable"
                )
            intervention_owners[intervention_hash] = role
            expectations = _expectations(group, branch)
            visible = {
                "common_context": context,
                "control_input": control,
                "runtime_program": runtime_program,
            }
            _assert_no_target_value_leak(
                _sealed_target_texts(branch), visible, f"group {group_id} sibling {role}"
            )
            sibling_id = _sibling_id(branch, group_id, role)
            case_spec = {
                "schema": SCHEMA,
                "checkpoint_sha256": config.checkpoint_sha256,
                "dataset_sha256": config.dataset_sha256,
                "split_sha256": config.split_sha256,
                "split": split,
                "group_id": group_id,
                "sibling_id": sibling_id,
                "role": role,
                "leakage_component_id": component,
                "shared_prefix_sha256": shared.sha256,
                "common_context": context,
                "control_input": control,
                "runtime_program": runtime_program,
                "expectations": expectations,
                "strata": _case_strata(group, branch, role),
                "evaluation_boundary": EVALUATION_BOUNDARY,
            }
            case_spec_hash = content_hash(case_spec)
            case_id = content_hash(
                {
                    "case_spec_sha256": case_spec_hash,
                    "group_id": group_id,
                    "sibling_id": sibling_id,
                }
            )
            if case_id in seen_cases:
                raise EvaluationContractError(f"duplicate evaluation case identity {case_id}")
            seen_cases.add(case_id)
            cases.append(
                EvaluationCase(
                    case_id=case_id,
                    case_spec_sha256=case_spec_hash,
                    group_id=group_id,
                    sibling_id=sibling_id,
                    role=role,
                    split=split,
                    leakage_component_id=component,
                    shared_prefix=shared,
                    common_context=dict(context),
                    control_input=dict(control),
                    runtime_program=runtime_program,
                    expectations=dict(expectations),
                    strata=_case_strata(group, branch, role),
                )
            )
    for policy in config.strata:
        observed = sum(policy.name in case.strata for case in cases)
        if observed != policy.expected_case_count:
            raise EvaluationContractError(
                f"preregistered stratum {policy.name} expected {policy.expected_case_count} "
                f"cases but the held-out split contains {observed}"
            )
    return tuple(cases)


def _event_type(event: Mapping[str, Any]) -> str | None:
    value = event.get("event_type")
    return value if isinstance(value, str) else None


def _event_revision(event: Mapping[str, Any]) -> int | None:
    value = event.get("revision")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _event_generation_id(event: Mapping[str, Any]) -> str | None:
    value = event.get("generation_id")
    return value if isinstance(value, str) and bool(value.strip()) else None


def _validate_typed_runtime_event(
    event: Mapping[str, Any], index: int
) -> str | None:
    if "event_type" not in event:
        return None
    event_type = event.get("event_type")
    if not isinstance(event_type, str):
        raise EvaluationContractError(
            f"runtime event {index}.event_type must be an exact string enum"
        )
    if event_type not in _TYPED_RUNTIME_EVENT_TYPES:
        return None
    if event_type in {
        "control_rejected",
        "control_accepted",
        "control_acknowledged",
        "recovery_generation_started",
    }:
        revision = event.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise EvaluationContractError(
                f"runtime event {index} {event_type} requires a non-negative integer revision"
            )
    if event_type == "control_rejected" and event.get("rejection_kind") not in {
        "null",
        "wrong",
        "stale",
    }:
        raise EvaluationContractError(
            f"runtime event {index} control_rejected requires rejection_kind "
            "null, wrong, or stale"
        )
    if event_type in {
        "barge_in_detected",
        "generation_cancelled",
        "audio_cutoff",
        "recovery_generation_started",
        "outbound_media",
    } and _event_generation_id(event) is None:
        raise EvaluationContractError(
            f"runtime event {index} {event_type} requires a non-empty string generation_id"
        )
    if event_type == "end_call" and not isinstance(event.get("model_selected"), bool):
        raise EvaluationContractError(
            f"runtime event {index} end_call requires boolean model_selected"
        )
    return event_type


def _expectation_bool(expectations: Mapping[str, Any], *names: str) -> bool | None:
    value = _mapping_value(expectations, *names)
    return value if isinstance(value, bool) else None


def _expectation_revision(expectations: Mapping[str, Any], control_revision: int | None) -> int | None:
    value = _mapping_value(
        expectations,
        "newest_revision",
        "newestRevision",
        "expected_revision",
        "expectedRevision",
    )
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return control_revision


def _control_revision(control: Mapping[str, Any]) -> int | None:
    value = _mapping_value(
        control,
        "controlRevision",
        "control_revision",
        "acknowledgedControlRevision",
        "acknowledged_control_revision",
    )
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    frame = _mapping_value(control, "controlFrame", "control_frame")
    return _control_revision(frame) if isinstance(frame, Mapping) else None


def score_runtime_events(
    events: Sequence[Mapping[str, Any]],
    expectations: Mapping[str, Any],
    *,
    role: str = "",
    control_revision: int | None = None,
) -> dict[str, Any]:
    """Score revision and interruption behavior from the ordered actual timeline."""

    ordered: list[dict[str, Any]] = []
    event_types: list[str | None] = []
    unknown_event_types: list[str] = []
    untyped_event_count = 0
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise EvaluationContractError(f"runtime event {index} must be an object")
        typed_event = dict(event)
        event_type = _validate_typed_runtime_event(typed_event, index)
        ordered.append(typed_event)
        event_types.append(event_type)
        if event_type is None:
            raw_event_type = typed_event.get("event_type")
            if isinstance(raw_event_type, str):
                unknown_event_types.append(raw_event_type)
            else:
                untyped_event_count += 1
    rejected: set[str] = set()
    rejected_revisions: dict[str, list[int | None]] = {
        "null": [],
        "wrong": [],
        "stale": [],
    }
    accepted_revisions: list[int] = []
    acknowledged_revisions: list[int] = []
    for event, event_type in zip(ordered, event_types):
        revision = _event_revision(event)
        if event_type == "control_acknowledged":
            if revision is not None:
                acknowledged_revisions.append(revision)
        if event_type == "control_rejected":
            rejection_kind = event["rejection_kind"]
            rejected.add(rejection_kind)
            rejected_revisions[rejection_kind].append(revision)
        if event_type == "control_accepted":
            if revision is not None:
                accepted_revisions.append(revision)

    expected_kinds: set[str] = set()
    raw_kinds = _mapping_value(expectations, "invalid_control_kinds", "invalidControlKinds")
    if raw_kinds is not None:
        if not isinstance(raw_kinds, (list, tuple)) or any(
            not isinstance(item, str) or item not in {"null", "wrong", "stale"}
            for item in raw_kinds
        ):
            raise EvaluationContractError(
                "invalid_control_kinds must contain only exact null, wrong, or stale enums"
            )
        expected_kinds.update(raw_kinds)
    for kind in ("null", "wrong", "stale"):
        if _expectation_bool(
            expectations,
            f"expect_{kind}_control_rejection",
            f"expect{kind.title()}ControlRejection",
        ) is True:
            expected_kinds.add(kind)
    if role == "superseded":
        expected_kinds.add("stale")

    newest = _expectation_revision(expectations, control_revision)
    decisions: dict[str, TypedDecision] = {}
    for kind, dimension in (
        ("null", "null_control_rejected"),
        ("wrong", "wrong_control_rejected"),
        ("stale", "stale_control_rejected"),
    ):
        rejection_proved = kind in rejected
        if kind == "stale" and newest is not None:
            rejection_proved = any(
                revision is not None and revision < newest
                for revision in rejected_revisions["stale"]
            )
        decisions[dimension] = (
            TypedDecision("pass" if rejection_proved else "fail")
            if kind in expected_kinds
            else TypedDecision("not_applicable")
        )

    if newest is None:
        decisions["newest_revision_acknowledged"] = TypedDecision("not_applicable")
    else:
        unexpected_accept = any(revision != newest for revision in accepted_revisions)
        decisions["newest_revision_acknowledged"] = TypedDecision(
            "pass" if newest in acknowledged_revisions and not unexpected_accept else "fail"
        )

    barge_required = _expectation_bool(
        expectations, "barge_in_required", "bargeInRequired", "expect_barge_in"
    )
    barge_indices = [
        index
        for index, event_type in enumerate(event_types)
        if event_type == "barge_in_detected"
    ]
    cancel_indices = [
        index
        for index, event_type in enumerate(event_types)
        if event_type == "generation_cancelled"
    ]
    cutoff_indices = [
        index
        for index, event_type in enumerate(event_types)
        if event_type == "audio_cutoff"
    ]
    recovery_indices = [
        index
        for index, event_type in enumerate(event_types)
        if event_type == "recovery_generation_started"
    ]
    sequence_records: list[dict[str, Any]] = []
    stale_emission_indices: set[int] = set()
    unattributed_emission_indices: set[int] = set()
    for sequence_index, barge_index in enumerate(barge_indices):
        next_barge = (
            barge_indices[sequence_index + 1]
            if sequence_index + 1 < len(barge_indices)
            else len(ordered)
        )
        interrupted_id = _event_generation_id(ordered[barge_index])
        cancel_index = next(
            (
                index
                for index in cancel_indices
                if barge_index < index < next_barge
                and interrupted_id is not None
                and _event_generation_id(ordered[index]) == interrupted_id
            ),
            None,
        )
        cutoff_index = next(
            (
                index
                for index in cutoff_indices
                if cancel_index is not None
                and cancel_index <= index < next_barge
                and _event_generation_id(ordered[index]) == interrupted_id
            ),
            None,
        )
        recovery_index = next(
            (
                index
                for index in recovery_indices
                if cutoff_index is not None and cutoff_index < index < next_barge
            ),
            None,
        )
        recovery_id = (
            _event_generation_id(ordered[recovery_index])
            if recovery_index is not None
            else None
        )
        recovery_revision = (
            _event_revision(ordered[recovery_index])
            if recovery_index is not None
            else None
        )
        recovery_valid = (
            recovery_index is not None
            and recovery_id is not None
            and recovery_id != interrupted_id
            and (newest is None or recovery_revision == newest)
        )
        if cancel_index is not None:
            for index in range(cancel_index + 1, len(ordered)):
                if event_types[index] != "outbound_media":
                    continue
                generation_id = _event_generation_id(ordered[index])
                if generation_id == interrupted_id:
                    stale_emission_indices.add(index)
                elif generation_id is None and (
                    recovery_index is None or index < recovery_index
                ):
                    unattributed_emission_indices.add(index)
        sequence_records.append(
            {
                "barge_in_index": barge_index,
                "interrupted_generation_id": interrupted_id,
                "cancellation_index": cancel_index,
                "cutoff_index": cutoff_index,
                "recovery_index": recovery_index,
                "recovery_generation_id": recovery_id,
                "recovery_valid": recovery_valid,
            }
        )

    stale_emissions = len(stale_emission_indices)
    duplex_applicable = barge_required is True or bool(barge_indices)
    if duplex_applicable:
        all_cancelled = bool(sequence_records) and all(
            item["cancellation_index"] is not None for item in sequence_records
        )
        all_cutoff = all_cancelled and all(
            item["cutoff_index"] is not None for item in sequence_records
        )
        all_recovered = all_cutoff and all(
            item["recovery_valid"] is True for item in sequence_records
        )
        decisions["barge_in_cancelled"] = TypedDecision(
            "pass" if all_cancelled else "fail"
        )
        decisions["queued_audio_cutoff"] = TypedDecision(
            "pass"
            if all_cutoff
            and stale_emissions == 0
            and not unattributed_emission_indices
            else "fail"
        )
        decisions["recovery_correct"] = TypedDecision(
            "pass"
            if all_recovered
            and stale_emissions == 0
            and not unattributed_emission_indices
            else "fail"
        )
    else:
        decisions["barge_in_cancelled"] = TypedDecision("not_applicable")
        decisions["queued_audio_cutoff"] = TypedDecision("not_applicable")
        decisions["recovery_correct"] = TypedDecision("not_applicable")

    expected_end_call = _expectation_bool(
        expectations, "expected_end_call", "expectedEndCall", "end_call_expected"
    )
    end_call_events = [
        event
        for event, event_type in zip(ordered, event_types)
        if event_type == "end_call"
    ]
    if expected_end_call is None:
        decisions["end_call_event_correct"] = TypedDecision("not_applicable")
    elif expected_end_call:
        model_selected = (
            len(end_call_events) == 1
            and end_call_events[0]["model_selected"] is True
        )
        decisions["end_call_event_correct"] = TypedDecision(
            "pass" if model_selected else "fail"
        )
    else:
        decisions["end_call_event_correct"] = TypedDecision(
            "pass" if not end_call_events else "fail"
        )

    if stale_emissions and decisions["stale_control_rejected"].status == "not_applicable":
        decisions["stale_control_rejected"] = TypedDecision(
            "fail", rationale="stale_media_emitted_after_cancellation"
        )
    return {
        "decisions": {key: value.as_dict() for key, value in decisions.items()},
        "counters": {
            "stale_emissions": stale_emissions,
            "unattributed_post_cancel_emissions": len(unattributed_emission_indices),
            "acknowledged_revisions": acknowledged_revisions,
            "accepted_revisions": accepted_revisions,
            "rejected_control_kinds": sorted(rejected),
            "rejected_control_revisions": rejected_revisions,
            "barge_in_events": len(barge_indices),
            "cancellation_events": len(cancel_indices),
            "correlated_cancellation_events": sum(
                item["cancellation_index"] is not None for item in sequence_records
            ),
            "cutoff_events": len(cutoff_indices),
            "correlated_cutoff_events": sum(
                item["cutoff_index"] is not None for item in sequence_records
            ),
            "recovery_events": len(recovery_indices),
            "valid_recovery_events": sum(
                item["recovery_valid"] is True for item in sequence_records
            ),
            "duplex_sequences": sequence_records,
            "end_call_events": len(end_call_events),
            "unknown_event_types": unknown_event_types,
            "untyped_events": untyped_event_count,
        },
    }


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _evidence_decision(
    value: Any, *, threshold: float | None = None, rationale: str
) -> TypedDecision:
    if isinstance(value, bool):
        return TypedDecision("pass" if value else "fail")
    numeric = _number(value)
    if numeric is not None and threshold is not None:
        if not 0.0 <= numeric <= 1.0:
            return TypedDecision("fail", rationale=f"{rationale}:out_of_range")
        return TypedDecision("pass" if numeric >= threshold else "fail")
    return TypedDecision("manual_review", rationale=rationale)


def _combined_evidence_decision(
    evidence: Mapping[str, Any],
    names: Sequence[str],
    *,
    threshold: float | None = None,
    rationale: str,
) -> tuple[TypedDecision, Any]:
    values = [evidence[name] for name in names if name in evidence]
    if not values:
        return TypedDecision("manual_review", rationale=rationale), None
    decisions = [
        _evidence_decision(value, threshold=threshold, rationale=rationale)
        for value in values
    ]
    measurement = next(
        (numeric for value in values if (numeric := _number(value)) is not None),
        values[0],
    )
    if any(item.status == "fail" for item in decisions):
        return TypedDecision("fail", rationale=f"{rationale}:failed_or_conflicting"), measurement
    if any(item.status != "pass" for item in decisions):
        return TypedDecision("manual_review", rationale=rationale), measurement
    return TypedDecision("pass"), measurement


def score_generated_evidence(
    output: GenerationOutput, asr: ASREvidence, config: EvaluationConfig
) -> dict[str, Any]:
    asr_decision, asr_value = _combined_evidence_decision(
        asr.evidence,
        (
            "intelligible",
            "intelligibility_pass",
            "intelligibility",
            "intelligibility_score",
        ),
        threshold=config.minimum_intelligibility,
        rationale="missing_typed_intelligibility_evidence",
    )
    if not asr.transcript.strip():
        asr_decision = TypedDecision("fail", rationale="empty_asr_transcript")

    codec_decision, _codec_value = _combined_evidence_decision(
        output.audio_evidence,
        ("codec_valid", "codecValid", "codec_pass"),
        rationale="missing_typed_codec_evidence",
    )
    if any(
        name in output.audio_evidence
        for name in ("channel_integrity", "channelIntegrity")
    ):
        channel_decision, _channel_value = _combined_evidence_decision(
            output.audio_evidence,
            ("channel_integrity", "channelIntegrity"),
            rationale="invalid_channel_integrity_evidence",
        )
        if channel_decision.status != "pass":
            codec_decision = TypedDecision(
                channel_decision.status, rationale="channel_integrity_failed_or_untyped"
            )

    voice_decision, voice_value = _combined_evidence_decision(
        output.audio_evidence,
        (
            "voice_preserved",
            "voicePreserved",
            "voice_similarity",
            "voiceSimilarity",
        ),
        threshold=config.minimum_voice_similarity,
        rationale="missing_typed_voice_evidence",
    )

    latency_ms = _number(
        _mapping_value(output.timing, "first_audio_latency_ms", "firstAudioLatencyMs")
    )
    if latency_ms is None:
        started = _number(
            _mapping_value(output.timing, "started_at_s", "startedAtSeconds")
        )
        first = _number(
            _mapping_value(output.timing, "first_audio_at_s", "firstAudioAtSeconds")
        )
        if started is not None and first is not None:
            latency_ms = (first - started) * 1000.0
    latency_decision = (
        TypedDecision(
            "pass"
            if latency_ms is not None
            and latency_ms >= 0
            and latency_ms <= config.maximum_first_audio_latency_ms
            else "fail"
        )
        if latency_ms is not None
        else TypedDecision("manual_review", rationale="missing_first_audio_latency")
    )

    real_time_factor = _number(
        _mapping_value(output.timing, "real_time_factor", "realTimeFactor")
    )
    if real_time_factor is None:
        started = _number(
            _mapping_value(output.timing, "started_at_s", "startedAtSeconds")
        )
        completed = _number(
            _mapping_value(output.timing, "completed_at_s", "completedAtSeconds")
        )
        duration = _number(
            _mapping_value(output.timing, "audio_duration_s", "audioDurationSeconds")
        )
        if started is not None and completed is not None and duration is not None and duration > 0:
            real_time_factor = (completed - started) / duration
    rtf_decision = (
        TypedDecision(
            "pass"
            if real_time_factor is not None
            and real_time_factor > 0
            and real_time_factor <= config.maximum_real_time_factor
            else "fail"
        )
        if real_time_factor is not None
        else TypedDecision("manual_review", rationale="missing_real_time_factor")
    )
    decisions = {
        "asr_intelligible": asr_decision,
        "codec_valid": codec_decision,
        "voice_preserved": voice_decision,
        "first_audio_latency": latency_decision,
        "real_time_factor": rtf_decision,
    }
    return {
        "decisions": {key: value.as_dict() for key, value in decisions.items()},
        "measurements": {
            "first_audio_latency_ms": latency_ms,
            "real_time_factor": real_time_factor,
            "intelligibility": asr_value,
            "voice_similarity": voice_value,
        },
    }


def _adapter_identity(adapter: Any) -> str:
    for attribute in ("adapter_id", "judge_id", "model_id", "identity"):
        value = getattr(adapter, attribute, None)
        if isinstance(value, str) and value:
            return value
    if callable(adapter) and hasattr(adapter, "__qualname__"):
        return f"{getattr(adapter, '__module__', '')}.{adapter.__qualname__}"
    cls = adapter.__class__
    return f"{cls.__module__}.{cls.__qualname__}"


def _invoke(adapter: Any, method_names: Sequence[str], request: Any) -> Any:
    for name in method_names:
        method = getattr(adapter, name, None)
        if callable(method):
            return method(request)
    if callable(adapter):
        return adapter(request)
    raise EvaluationContractError(
        f"adapter {_adapter_identity(adapter)} lacks any of methods {tuple(method_names)}"
    )


def _contains_key(value: Any, wanted: set[str]) -> bool:
    return any(key in wanted for key, _child, _path in _walk_keys(value))


def _applicable_metrics(
    case: EvaluationCase,
    semantic: Mapping[str, TypedDecision],
    runtime: Mapping[str, TypedDecision],
    evidence: Mapping[str, TypedDecision],
) -> tuple[str, ...]:
    required = set(_MANDATORY_CASE_DIMENSIONS)
    if case.role == "uncertain":
        required.add("uncertainty_handled")
    if _contains_key(case.control_input, {"facts", "requiredfacts", "evidencefacts"}) or _mapping_value(
        case.expectations, "facts_required", "required_facts"
    ) is not None:
        required.add("facts_incorporated")
    if _contains_key(case.control_input, {"toolresult", "toolresults", "toolevidence"}) or _mapping_value(
        case.expectations, "tool_result_required", "required_tool_results"
    ) is not None:
        required.add("tool_results_incorporated")
    raw = _mapping_value(case.expectations, "applicable_metrics", "applicableMetrics")
    if isinstance(raw, (list, tuple)):
        unknown = set(str(item) for item in raw).difference(ALL_DIMENSIONS)
        if unknown:
            raise EvaluationContractError(f"case {case.case_id} names unknown metrics {sorted(unknown)}")
        required.update(str(item) for item in raw)
    non_applicable = _mapping_value(
        case.expectations, "non_applicable_metrics", "nonApplicableMetrics"
    )
    if isinstance(non_applicable, (list, tuple)):
        requested = {str(item) for item in non_applicable}
        protected = requested.intersection(_MANDATORY_CASE_DIMENSIONS)
        if protected:
            raise EvaluationContractError(
                "mandatory control/evidence metrics cannot be declared non-applicable: "
                f"{sorted(protected)}"
            )
        required.difference_update(requested)
    for collection in (semantic, runtime, evidence):
        required.update(
            key for key, decision in collection.items() if decision.status != "not_applicable"
        )
    required.update(EVIDENCE_DIMENSIONS)
    required.update(GROUP_DIMENSIONS)
    return tuple(sorted(required))


def _decision_mapping(value: Mapping[str, Any]) -> dict[str, TypedDecision]:
    return {str(key): TypedDecision.from_value(item) for key, item in value.items()}


def _failed_case_draft(
    case: EvaluationCase,
    run_identity_hash: str,
    error: Exception,
    *,
    device: int | None,
    observed_generation_mode: str = "unavailable",
) -> dict[str, Any]:
    scores = {
        dimension: TypedDecision(
            "fail" if dimension in EVIDENCE_DIMENSIONS else "manual_review",
            rationale=f"case_execution_failed:{error.__class__.__name__}",
        ).as_dict()
        for dimension in (*SEMANTIC_DIMENSIONS, *RUNTIME_DIMENSIONS, *EVIDENCE_DIMENSIONS)
    }
    return {
        "schema": CASE_RESULT_SCHEMA,
        "case_id": case.case_id,
        "case_spec_sha256": case.case_spec_sha256,
        "run_identity_sha256": run_identity_hash,
        "group_id": case.group_id,
        "sibling_id": case.sibling_id,
        "role": case.role,
        "split": case.split,
        "leakage_component_id": case.leakage_component_id,
        "strata": list(case.strata),
        "evaluation_boundary": dict(EVALUATION_BOUNDARY),
        "artifact_hashes": {"shared_prefix_sha256": case.shared_prefix.sha256},
        "case_input": {
            "common_context": case.common_context,
            "control_input": case.control_input,
            "runtime_program": case.runtime_program,
            "typed_expectations": case.expectations,
        },
        "generation": {
            "mode": observed_generation_mode,
            "physical_cuda_device": device,
            "compute_backend": "unavailable",
            "cpu_fallback_used": None,
            "generated_text": "",
            "audio_sha256": None,
            "audio_size_bytes": 0,
            "timing": {},
            "events": [],
            "audio_evidence": {},
            "generator_id": "unavailable",
        },
        "asr": {"transcript": "", "evidence": {}, "asr_id": "unavailable"},
        "adjudication": {
            "status": "manual_review",
            "decisions": {
                key: value for key, value in scores.items() if key in SEMANTIC_DIMENSIONS
            },
            "rationale": [str(error)],
            "judge_id": "not_invoked",
        },
        "runtime_scoring": {"decisions": {}, "counters": {}},
        "evidence_scoring": {"decisions": {}, "measurements": {}},
        "scores": scores,
        "required_metrics": list(ALL_DIMENSIONS),
        "execution_errors": [f"{error.__class__.__name__}: {error}"],
    }


def _case_branch_view(result: Mapping[str, Any]) -> dict[str, Any]:
    generation = result.get("generation") if isinstance(result.get("generation"), Mapping) else {}
    asr = result.get("asr") if isinstance(result.get("asr"), Mapping) else {}
    case_input = result.get("case_input") if isinstance(result.get("case_input"), Mapping) else {}
    return {
        "case_id": result.get("case_id"),
        "sibling_id": result.get("sibling_id"),
        "role": result.get("role"),
        "control_input": case_input.get("control_input", {}),
        "generated_text": generation.get("generated_text", ""),
        "generated_speech_asr": asr.get("transcript", ""),
        "runtime_events": generation.get("events", []),
        "timing_evidence": generation.get("timing", {}),
        "asr_evidence": asr.get("evidence", {}),
        "codec_voice_evidence": generation.get("audio_evidence", {}),
    }


def _finalize_case_result(
    draft: Mapping[str, Any], branch_decision: TypedDecision
) -> dict[str, Any]:
    result = dict(draft)
    scores = dict(result.get("scores", {}))
    scores["branch_discrimination"] = branch_decision.as_dict()
    result["scores"] = scores
    required = tuple(str(item) for item in result.get("required_metrics", ALL_DIMENSIONS))
    if "branch_discrimination" not in required:
        required = tuple(sorted({*required, "branch_discrimination"}))
    result["required_metrics"] = list(required)
    failures: list[str] = list(result.get("execution_errors", []))
    statuses: list[str] = []
    for metric in required:
        raw = scores.get(metric)
        decision = TypedDecision.from_value(
            raw, missing_rationale="required_metric_missing_from_case_result"
        )
        statuses.append(decision.status)
        if decision.status != "pass":
            failures.append(f"{metric}:{decision.status}")
    adjudication = result.get("adjudication")
    judge_status = (
        str(adjudication.get("status")) if isinstance(adjudication, Mapping) else "manual_review"
    )
    generation = result.get("generation")
    mode = str(generation.get("mode")) if isinstance(generation, Mapping) else "unavailable"
    if mode != "free_running":
        failures.append("generation_boundary:not_free_running")
    if "fail" in statuses or judge_status == "fail" or result.get("execution_errors"):
        status = "failed"
    elif "ambiguous" in statuses or judge_status == "ambiguous":
        status = "ambiguous"
    elif "manual_review" in statuses or "not_applicable" in statuses or judge_status == "manual_review":
        status = "manual_review"
    else:
        status = "passed"
    result["status"] = status
    result["promotion_eligible"] = status == "passed" and mode == "free_running"
    result["failure_reasons"] = sorted(set(failures))
    result_base = {key: value for key, value in result.items() if key != "result_id"}
    result["result_id"] = content_hash(result_base)
    return result


def _rate(successes: int, total: int) -> dict[str, Any]:
    lower, upper = wilson_interval(successes, total)
    return {
        "successes": successes,
        "total": total,
        "observed_reliability": successes / total if total else 0.0,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
    }


def _result_boundary_is_valid(result: Mapping[str, Any]) -> bool:
    boundary = result.get("evaluation_boundary")
    generation = result.get("generation")
    return (
        isinstance(boundary, Mapping)
        and all(boundary.get(key) == value for key, value in EVALUATION_BOUNDARY.items())
        and isinstance(generation, Mapping)
        and generation.get("mode") == "free_running"
        and generation.get("physical_cuda_device") in ALLOWED_PHYSICAL_CUDA_DEVICES
        and generation.get("compute_backend") in {"cuda", "nvidia_cuda"}
        and generation.get("cpu_fallback_used") is False
    )


def _result_is_success(result: Mapping[str, Any]) -> bool:
    return (
        result.get("status") == "passed"
        and result.get("promotion_eligible") is True
        and _result_boundary_is_valid(result)
    )


def aggregate_results(
    results: Sequence[Mapping[str, Any]], config: EvaluationConfig
) -> dict[str, Any]:
    """Aggregate against preregistered denominators; missing cases are failures."""

    case_ids = [str(item.get("case_id", "")) for item in results]
    invalid_case_ids = [
        f"index:{index}" for index, case_id in enumerate(case_ids) if not case_id
    ]
    duplicate_ids = sorted(
        case_id for case_id in set(case_ids) if case_id and case_ids.count(case_id) > 1
    )
    success_count = sum(_result_is_success(item) for item in results)
    statistical_successes = min(success_count, config.expected_case_count)
    overall = _rate(statistical_successes, config.expected_case_count)
    overall.update(
        {
            "observed_cases": len(results),
            "missing_cases": max(0, config.expected_case_count - len(results)),
            "unexpected_cases": max(0, len(results) - config.expected_case_count),
            "ambiguous_cases": sum(item.get("status") == "ambiguous" for item in results),
            "manual_review_cases": sum(item.get("status") == "manual_review" for item in results),
        }
    )
    overall_gate = (
        len(results) == config.expected_case_count
        and not invalid_case_ids
        and not duplicate_ids
        and overall["observed_reliability"] >= config.observed_reliability_threshold
        and overall["wilson_95_lower"] >= config.aggregate_wilson_lower_threshold
    )
    overall["gate_pass"] = overall_gate

    expected_group_count = config.expected_case_count // len(REQUIRED_ROLES)
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    missing_group_cases: list[str] = []
    for index, result in enumerate(results):
        group_id = result.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            missing_group_cases.append(case_ids[index] or f"index:{index}")
            continue
        grouped.setdefault(group_id, []).append(result)
    invalid_causal_groups: dict[str, list[str]] = {}
    component_owners: dict[str, list[str]] = {}
    group_successes = 0
    for group_id, members in grouped.items():
        reasons: list[str] = []
        roles = [item.get("role") for item in members]
        if len(members) != len(REQUIRED_ROLES) or set(roles) != set(REQUIRED_ROLES):
            reasons.append("incomplete_or_duplicate_roles")
        components = {
            item.get("leakage_component_id")
            for item in members
            if isinstance(item.get("leakage_component_id"), str)
            and item.get("leakage_component_id")
        }
        if len(components) != 1:
            reasons.append("missing_or_inconsistent_leakage_component")
        else:
            component_owners.setdefault(str(next(iter(components))), []).append(group_id)
        if reasons:
            invalid_causal_groups[group_id] = reasons
        elif all(_result_is_success(item) for item in members):
            group_successes += 1
    non_disjoint_components = {
        component: sorted(owners)
        for component, owners in component_owners.items()
        if len(set(owners)) > 1
    }
    causal_groups = _rate(min(group_successes, expected_group_count), expected_group_count)
    causal_groups.update(
        {
            "expected_groups": expected_group_count,
            "observed_groups": len(grouped),
            "missing_groups": max(0, expected_group_count - len(grouped)),
            "unexpected_groups": max(0, len(grouped) - expected_group_count),
            "invalid_groups": len(invalid_causal_groups),
            "observed_threshold": config.observed_reliability_threshold,
            "wilson_lower_threshold": config.aggregate_wilson_lower_threshold,
        }
    )
    causal_group_gate = (
        len(grouped) == expected_group_count
        and not missing_group_cases
        and not invalid_causal_groups
        and not non_disjoint_components
        and causal_groups["observed_reliability"]
        >= config.observed_reliability_threshold
        and causal_groups["wilson_95_lower"]
        >= config.aggregate_wilson_lower_threshold
    )
    causal_groups["gate_pass"] = causal_group_gate

    strata: dict[str, Any] = {}
    coverage_failures: list[str] = []
    safety_failures: list[str] = []
    stratum_reliability_failures: list[str] = []
    for policy in config.strata:
        members = [item for item in results if policy.name in item.get("strata", [])]
        member_successes = sum(_result_is_success(item) for item in members)
        statistics = _rate(
            min(member_successes, policy.expected_case_count), policy.expected_case_count
        )
        exact = len(members) == policy.expected_case_count
        clustered_members: dict[str, list[Mapping[str, Any]]] = {}
        for item in members:
            group_id = item.get("group_id")
            if isinstance(group_id, str) and group_id:
                clustered_members.setdefault(group_id, []).append(item)
        clustered = _rate(
            sum(
                all(_result_is_success(item) for item in cluster)
                for cluster in clustered_members.values()
            ),
            len(clustered_members),
        )
        statistics.update(
            {
                "kind": policy.kind,
                "expected_cases": policy.expected_case_count,
                "observed_cases": len(members),
                "missing_cases": max(0, policy.expected_case_count - len(members)),
                "unexpected_cases": max(0, len(members) - policy.expected_case_count),
                "ambiguous_cases": sum(item.get("status") == "ambiguous" for item in members),
                "manual_review_cases": sum(
                    item.get("status") == "manual_review" for item in members
                ),
                "wilson_lower_threshold": policy.wilson_lower_threshold,
                "observed_threshold": config.observed_reliability_threshold,
                "exact_denominator": exact,
                "independent_groups": clustered["total"],
                "clustered_observed_reliability": clustered[
                    "observed_reliability"
                ],
                "clustered_wilson_95_lower": clustered["wilson_95_lower"],
                "clustered_wilson_95_upper": clustered["wilson_95_upper"],
            }
        )
        if not exact:
            coverage_failures.append(policy.name)
        gate_pass = (
            exact
            and bool(clustered_members)
            and statistics["observed_reliability"]
            >= config.observed_reliability_threshold
            and statistics["wilson_95_lower"] >= policy.wilson_lower_threshold
            and clustered["observed_reliability"]
            >= config.observed_reliability_threshold
            and clustered["wilson_95_lower"] >= policy.wilson_lower_threshold
        )
        if not gate_pass:
            stratum_reliability_failures.append(policy.name)
            if policy.kind == "safety_critical":
                safety_failures.append(policy.name)
        statistics["gate_pass"] = gate_pass
        strata[policy.name] = statistics

    dimensions: dict[str, Any] = {}
    for dimension in ALL_DIMENSIONS:
        decisions: list[TypedDecision] = []
        for result in results:
            scores = result.get("scores")
            if not isinstance(scores, Mapping) or dimension not in scores:
                decisions.append(
                    TypedDecision("manual_review", rationale="metric_missing_from_result")
                )
            else:
                decisions.append(TypedDecision.from_value(scores[dimension]))
        applicable = [item for item in decisions if item.status != "not_applicable"]
        metric = _rate(sum(item.status == "pass" for item in applicable), len(applicable))
        metric.update(
            {
                "not_applicable": len(decisions) - len(applicable),
                "ambiguous": sum(item.status == "ambiguous" for item in applicable),
                "manual_review": sum(item.status == "manual_review" for item in applicable),
            }
        )
        dimensions[dimension] = metric

    boundary_failures = [
        str(item.get("case_id"))
        for item in results
        if not _result_boundary_is_valid(item)
    ]
    failure_reasons: list[str] = []
    if duplicate_ids:
        failure_reasons.append("duplicate_case_ids")
    if invalid_case_ids:
        failure_reasons.append("missing_case_ids")
    if len(results) != config.expected_case_count:
        failure_reasons.append("overall_exact_denominator_mismatch")
    if overall["observed_reliability"] < config.observed_reliability_threshold:
        failure_reasons.append("aggregate_observed_reliability_below_0_95")
    if overall["wilson_95_lower"] < config.aggregate_wilson_lower_threshold:
        failure_reasons.append("aggregate_wilson_95_lower_below_threshold")
    if not causal_group_gate:
        failure_reasons.append("causal_group_reliability_gate_failed")
    if missing_group_cases or invalid_causal_groups or non_disjoint_components:
        failure_reasons.append("causal_groups_not_structurally_disjoint")
    if coverage_failures:
        failure_reasons.append("preregistered_stratum_denominator_mismatch")
    if safety_failures:
        failure_reasons.append("safety_critical_stratum_gate_failed")
    if stratum_reliability_failures:
        failure_reasons.append("preregistered_stratum_reliability_gate_failed")
    if boundary_failures:
        failure_reasons.append("non_free_running_or_unverified_case_present")
    passed = (
        overall_gate
        and causal_group_gate
        and not coverage_failures
        and not stratum_reliability_failures
        and not boundary_failures
    )
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "passed" if passed else "failed",
        "promotion_eligible": passed,
        "evaluation_boundary": dict(EVALUATION_BOUNDARY),
        "artifact_hashes": {
            "checkpoint_sha256": config.checkpoint_sha256,
            "dataset_sha256": config.dataset_sha256,
            "split_sha256": config.split_sha256,
        },
        "split": config.split_name,
        "overall": overall,
        "causal_groups": causal_groups,
        "strata": strata,
        "dimensions": dimensions,
        "duplicate_case_ids": duplicate_ids,
        "invalid_case_ids": invalid_case_ids,
        "missing_group_cases": missing_group_cases,
        "invalid_causal_groups": invalid_causal_groups,
        "non_disjoint_components": non_disjoint_components,
        "coverage_failures": coverage_failures,
        "stratum_reliability_failures": stratum_reliability_failures,
        "safety_critical_failures": safety_failures,
        "boundary_failures": boundary_failures,
        "failure_reasons": failure_reasons,
        "thresholds": {
            "observed_reliability": config.observed_reliability_threshold,
            "aggregate_wilson_95_lower": config.aggregate_wilson_lower_threshold,
        },
    }


def _immutable_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise ResumeConflictError(f"cannot inspect immutable artifact {path}: {exc}") from exc
        if existing != payload:
            raise ResumeConflictError(f"immutable artifact conflicts with requested content: {path}")


def _json_document(value: Any) -> bytes:
    return json.dumps(_json_safe(value), indent=2, sort_keys=True, ensure_ascii=True).encode(
        "utf-8"
    ) + b"\n"


def _id_digest(value: str) -> str:
    return require_sha256(value, "content identity")[7:]


class GeneratedControlEvaluationHarness:
    def __init__(
        self,
        *,
        config: EvaluationConfig,
        generator: GenerationAdapter | Callable[[GenerationRequest], Any],
        asr: ASRAdapter | Callable[[ASRRequest], Any],
        judge: TypedJudgeAdapter | Callable[[CaseAdjudicationRequest], Any],
        output_dir: Path,
        device_admission: DeviceAdmission,
        host_ram_admission: HostRamAdmission,
        dataset_root: Path | None = None,
        resume: bool = True,
    ) -> None:
        if generator is judge or _adapter_identity(generator) == _adapter_identity(judge):
            raise EvaluationContractError(
                "semantic adjudication must be independent from the generation adapter"
            )
        self.config = config
        self.generator = generator
        self.asr = asr
        self.judge = judge
        self.output_dir = output_dir.expanduser().resolve()
        self.device_admission = device_admission
        self.host_ram_admission = host_ram_admission
        self.dataset_root = dataset_root
        self.resume = resume
        self.generator_id = _adapter_identity(generator)
        self.asr_id = _adapter_identity(asr)
        self.judge_id = _adapter_identity(judge)
        self.run_identity = {
            "schema": SCHEMA,
            "evaluator_implementation_sha256": sha256_path(Path(__file__)),
            "configuration": config.as_dict(),
            "evaluation_boundary": dict(EVALUATION_BOUNDARY),
            "adapter_boundaries": {
                "generator": self.generator_id,
                "asr": self.asr_id,
                "independent_judge": self.judge_id,
            },
            "physical_cuda_policy": list(ALLOWED_PHYSICAL_CUDA_DEVICES),
            "cpu_model_fallback": False,
        }
        self.run_identity_hash = content_hash(self.run_identity)

    def _verify_inputs(
        self,
        groups: Sequence[Mapping[str, Any]],
        dataset_bytes: bytes | None,
        split_bytes: bytes | None,
    ) -> None:
        canonical = canonical_jsonl_bytes(groups)
        verify_exact_hash(
            "dataset", dataset_bytes if dataset_bytes is not None else canonical, self.config.dataset_sha256
        )
        verify_exact_hash(
            "split", split_bytes if split_bytes is not None else canonical, self.config.split_sha256
        )

    def _checkpoint_path(self, case: EvaluationCase) -> Path:
        return self.output_dir / "checkpoints" / f"{_id_digest(case.case_id)}.json"

    def _load_case(self, case: EvaluationCase) -> dict[str, Any] | None:
        checkpoint_path = self._checkpoint_path(case)
        if not checkpoint_path.exists():
            return None
        if not self.resume:
            raise ResumeConflictError(f"case checkpoint already exists and resume is disabled: {checkpoint_path}")
        try:
            checkpoint_payload = checkpoint_path.read_bytes()
            checkpoint = json.loads(checkpoint_payload)
        except (OSError, json.JSONDecodeError) as exc:
            raise ResumeConflictError(f"invalid resume checkpoint {checkpoint_path}: {exc}") from exc
        if not isinstance(checkpoint, Mapping):
            raise ResumeConflictError(f"resume checkpoint is not an object for {case.case_id}")
        checkpoint_base = {
            key: value for key, value in checkpoint.items() if key != "checkpoint_id"
        }
        if (
            checkpoint.get("schema") != CHECKPOINT_SCHEMA
            or content_hash(checkpoint_base) != checkpoint.get("checkpoint_id")
        ):
            raise ResumeConflictError(
                f"resume checkpoint content identity conflict for {case.case_id}"
            )
        expected = {
            "case_id": case.case_id,
            "case_spec_sha256": case.case_spec_sha256,
            "run_identity_sha256": self.run_identity_hash,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ResumeConflictError(f"resume identity conflict for case {case.case_id}")
        result_path_value = checkpoint.get("result_path")
        if not isinstance(result_path_value, str):
            raise ResumeConflictError(f"resume checkpoint lacks result_path for {case.case_id}")
        result_relative = Path(result_path_value)
        if result_relative.is_absolute() or ".." in result_relative.parts:
            raise ResumeConflictError(f"resume result path escapes output directory for {case.case_id}")
        result_path = (self.output_dir / result_relative).resolve()
        try:
            result_path.relative_to(self.output_dir)
        except ValueError as exc:
            raise ResumeConflictError(
                f"resume result path escapes output directory for {case.case_id}"
            ) from exc
        try:
            payload = result_path.read_bytes()
            result = json.loads(payload)
        except (OSError, json.JSONDecodeError) as exc:
            raise ResumeConflictError(f"cannot load content-addressed result for {case.case_id}: {exc}") from exc
        if not isinstance(result, Mapping):
            raise ResumeConflictError(f"resume result is not an object for {case.case_id}")
        if sha256_bytes(payload) != checkpoint.get("result_file_sha256"):
            raise ResumeConflictError(f"result file hash conflict for {case.case_id}")
        result_base = {key: value for key, value in result.items() if key != "result_id"}
        if content_hash(result_base) != result.get("result_id"):
            raise ResumeConflictError(f"result content identity conflict for {case.case_id}")
        if result.get("result_id") != checkpoint.get("result_id"):
            raise ResumeConflictError(f"checkpoint result identity conflict for {case.case_id}")
        result_expected = {
            "schema": CASE_RESULT_SCHEMA,
            "case_id": case.case_id,
            "case_spec_sha256": case.case_spec_sha256,
            "run_identity_sha256": self.run_identity_hash,
        }
        if any(result.get(key) != value for key, value in result_expected.items()):
            raise ResumeConflictError(f"resume result case identity conflict for {case.case_id}")
        scores = result.get("scores")
        branch_raw = scores.get("branch_discrimination") if isinstance(scores, Mapping) else None
        recomputed = _finalize_case_result(
            {
                key: value
                for key, value in result.items()
                if key
                not in {"result_id", "status", "promotion_eligible", "failure_reasons"}
            },
            TypedDecision.from_value(
                branch_raw, missing_rationale="resumed_group_lacks_branch_decision"
            ),
        )
        if any(
            result.get(key) != recomputed.get(key)
            for key in ("status", "promotion_eligible", "failure_reasons", "result_id")
        ):
            raise ResumeConflictError(f"resume result scoring conflict for {case.case_id}")
        return dict(result)

    def _write_case(self, case: EvaluationCase, result: Mapping[str, Any]) -> dict[str, Any]:
        result_id = require_sha256(result.get("result_id"), "result_id")
        result_relative = Path("cases") / f"{_id_digest(result_id)}.json"
        payload = _json_document(result)
        result_path = self.output_dir / result_relative
        _immutable_write(result_path, payload)
        checkpoint_base = {
            "schema": CHECKPOINT_SCHEMA,
            "case_id": case.case_id,
            "case_spec_sha256": case.case_spec_sha256,
            "run_identity_sha256": self.run_identity_hash,
            "result_id": result_id,
            "result_path": result_relative.as_posix(),
            "result_file_sha256": sha256_bytes(payload),
        }
        checkpoint = {
            **checkpoint_base,
            "checkpoint_id": content_hash(checkpoint_base),
        }
        _immutable_write(self._checkpoint_path(case), _json_document(checkpoint))
        return {
            "case_id": case.case_id,
            "result_id": result_id,
            "path": result_relative.as_posix(),
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }

    def _evaluate_case(self, case: EvaluationCase, device: int) -> dict[str, Any]:
        try:
            request = GenerationRequest(
                case_id=case.case_id,
                group_id=case.group_id,
                sibling_id=case.sibling_id,
                split=case.split,
                checkpoint_sha256=self.config.checkpoint_sha256,
                shared_prefix=case.shared_prefix.payload,
                shared_prefix_sha256=case.shared_prefix.sha256,
                common_context=case.common_context,
                control_input=case.control_input,
                runtime_program=case.runtime_program,
                physical_cuda_device=device,
                evaluation_boundary=dict(EVALUATION_BOUNDARY),
            )
            _assert_target_free_keys(asdict(request), "generation request")
            raw_output = _invoke(self.generator, ("generate",), request)
            observed_mode = (
                str(_mapping_value(raw_output, "generation_mode", "mode") or "unverified")
                if isinstance(raw_output, Mapping)
                else getattr(raw_output, "generation_mode", "unverified")
            )
            output = GenerationOutput.from_value(
                raw_output, fallback_generator_id=self.generator_id
            )
            if output.physical_cuda_device != device:
                raise GenerationContractError(
                    "generator CUDA attestation does not match the admitted physical device"
                )
            if output.generator_id == self.judge_id:
                raise EvaluationContractError(
                    "generated evidence claims the independent judge as its generator"
                )
            asr_request = ASRRequest(
                case_id=case.case_id,
                audio=output.audio,
                audio_sha256=sha256_bytes(output.audio),
                audio_evidence=output.audio_evidence,
            )
            raw_asr = _invoke(self.asr, ("transcribe", "recognize"), asr_request)
            asr = ASREvidence.from_value(raw_asr, fallback_asr_id=self.asr_id)
            judge_request = CaseAdjudicationRequest(
                case_id=case.case_id,
                group_id=case.group_id,
                sibling_id=case.sibling_id,
                role=case.role,
                control_input=case.control_input,
                common_context=case.common_context,
                generated_text=output.generated_text,
                generated_speech_asr=asr.transcript,
                runtime_events=output.events,
                timing_evidence=output.timing,
                asr_evidence=asr.evidence,
                codec_voice_evidence=output.audio_evidence,
                typed_expectations=case.expectations,
                evaluation_boundary=dict(EVALUATION_BOUNDARY),
            )
            _assert_target_free_keys(asdict(judge_request), "semantic judge request")
            raw_judgment = _invoke(
                self.judge, ("adjudicate", "judge_case", "judge_turn"), judge_request
            )
            judgment = CaseJudgment.from_value(
                raw_judgment, fallback_judge_id=self.judge_id
            )
            if judgment.judge_id in {self.generator_id, output.generator_id}:
                raise AdjudicationContractError(
                    "judge identity is not independent from the generator identity"
                )
            runtime_raw = score_runtime_events(
                output.events,
                case.expectations,
                role=case.role,
                control_revision=_control_revision(case.control_input),
            )
            runtime_decisions = _decision_mapping(runtime_raw["decisions"])
            evidence_raw = score_generated_evidence(output, asr, self.config)
            evidence_decisions = _decision_mapping(evidence_raw["decisions"])
            semantic_decisions = dict(judgment.decisions)
            scores = {
                **{key: value.as_dict() for key, value in semantic_decisions.items()},
                **{key: value.as_dict() for key, value in runtime_decisions.items()},
                **{key: value.as_dict() for key, value in evidence_decisions.items()},
            }
            required = _applicable_metrics(
                case, semantic_decisions, runtime_decisions, evidence_decisions
            )
            return {
                "schema": CASE_RESULT_SCHEMA,
                "case_id": case.case_id,
                "case_spec_sha256": case.case_spec_sha256,
                "run_identity_sha256": self.run_identity_hash,
                "group_id": case.group_id,
                "sibling_id": case.sibling_id,
                "role": case.role,
                "split": case.split,
                "leakage_component_id": case.leakage_component_id,
                "strata": list(case.strata),
                "evaluation_boundary": dict(EVALUATION_BOUNDARY),
                "artifact_hashes": {
                    "checkpoint_sha256": self.config.checkpoint_sha256,
                    "dataset_sha256": self.config.dataset_sha256,
                    "split_sha256": self.config.split_sha256,
                    "shared_prefix_sha256": case.shared_prefix.sha256,
                    "generated_audio_sha256": sha256_bytes(output.audio),
                },
                "case_input": {
                    "common_context": case.common_context,
                    "control_input": case.control_input,
                    "runtime_program": case.runtime_program,
                    "typed_expectations": case.expectations,
                },
                "generation": {
                    "mode": output.generation_mode,
                    "physical_cuda_device": output.physical_cuda_device,
                    "compute_backend": output.compute_backend,
                    "cpu_fallback_used": output.cpu_fallback_used,
                    "generated_text": output.generated_text,
                    "audio_sha256": sha256_bytes(output.audio),
                    "audio_size_bytes": len(output.audio),
                    "timing": dict(output.timing),
                    "events": [dict(event) for event in output.events],
                    "audio_evidence": dict(output.audio_evidence),
                    "generator_id": output.generator_id,
                },
                "asr": {
                    "transcript": asr.transcript,
                    "evidence": dict(asr.evidence),
                    "asr_id": asr.asr_id,
                },
                "adjudication": judgment.as_dict(),
                "runtime_scoring": runtime_raw,
                "evidence_scoring": evidence_raw,
                "scores": scores,
                "required_metrics": list(required),
                "execution_errors": [],
            }
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            mode = locals().get("observed_mode", "unavailable")
            return _failed_case_draft(
                case,
                self.run_identity_hash,
                exc,
                device=device,
                observed_generation_mode=str(mode),
            )

    def _group_decision(self, group_id: str, drafts: Sequence[Mapping[str, Any]]) -> TypedDecision:
        if len(drafts) != len(REQUIRED_ROLES):
            return TypedDecision("fail", rationale="incomplete_four_sibling_generated_group")
        views = tuple(_case_branch_view(item) for item in drafts)
        texts = {
            (str(item.get("generated_text", "")).strip(), str(item.get("generated_speech_asr", "")).strip())
            for item in views
        }
        try:
            request = BranchDiscriminationRequest(
                group_id=group_id,
                branches=views,
                evaluation_boundary=dict(EVALUATION_BOUNDARY),
            )
            _assert_target_free_keys(asdict(request), "branch discrimination request")
            raw = _invoke(
                self.judge,
                ("adjudicate_group", "judge_group", "judge_branches", "discriminate"),
                request,
            )
            if isinstance(raw, Mapping):
                status = raw.get(
                    "branch_discrimination",
                    raw.get("branch_discrimination_pass", raw.get("decision")),
                )
                decision = TypedDecision.from_value(
                    status, missing_rationale="group_judge_omitted_branch_discrimination"
                )
                group_status = raw.get("status")
                if group_status in {"ambiguous", "manual_review", "fail", "failed"}:
                    decision = TypedDecision(
                        "fail" if group_status in {"fail", "failed"} else str(group_status),
                        rationale=str(raw.get("rationale", "")),
                    )
            else:
                decision = TypedDecision(
                    "manual_review", rationale="group_judge_did_not_return_typed_object"
                )
        except Exception as exc:
            decision = TypedDecision(
                "manual_review", rationale=f"group_adjudication_failed:{exc}"
            )
        if len(texts) != len(REQUIRED_ROLES):
            return TypedDecision(
                "fail", rationale="generated sibling outputs are not materially distinguishable"
            )
        return decision

    def run(
        self,
        groups: Sequence[Mapping[str, Any]],
        *,
        dataset_bytes: bytes | None = None,
        split_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        rows = tuple(groups)
        self._verify_inputs(rows, dataset_bytes, split_bytes)
        cases = prepare_evaluation_cases(
            rows, self.config, dataset_root=self.dataset_root
        )
        _immutable_write(
            self.output_dir / "run_identity.json",
            _json_document(
                {
                    **self.run_identity,
                    "run_identity_sha256": self.run_identity_hash,
                }
            ),
        )
        cases_by_group: dict[str, list[EvaluationCase]] = {}
        for case in cases:
            cases_by_group.setdefault(case.group_id, []).append(case)
        results: list[dict[str, Any]] = []
        descriptors: dict[str, dict[str, Any]] = {}
        for group_id in sorted(cases_by_group):
            group_cases = sorted(
                cases_by_group[group_id], key=lambda item: REQUIRED_ROLES.index(item.role)
            )
            loaded: dict[str, dict[str, Any]] = {}
            drafts: dict[str, dict[str, Any]] = {}
            for case in group_cases:
                resumed = self._load_case(case)
                if resumed is not None:
                    loaded[case.case_id] = resumed
                    drafts[case.case_id] = resumed
                    continue
                self.host_ram_admission.admit()
                device = self.device_admission.acquire()
                if device not in ALLOWED_PHYSICAL_CUDA_DEVICES:
                    raise CudaAdmissionError(
                        f"device admission returned forbidden physical CUDA device {device}"
                    )
                drafts[case.case_id] = self._evaluate_case(case, device)
            if len(loaded) == len(group_cases):
                resumed_decisions = [
                    TypedDecision.from_value(
                        loaded[case.case_id].get("scores", {}).get(
                            "branch_discrimination"
                        ),
                        missing_rationale="resumed_group_lacks_branch_decision",
                    )
                    for case in group_cases
                ]
                branch_decision = resumed_decisions[0]
                if any(item != branch_decision for item in resumed_decisions[1:]):
                    raise ResumeConflictError(
                        f"resumed sibling group judgments disagree for {group_id}"
                    )
            else:
                branch_decision = self._group_decision(
                    group_id, [drafts[case.case_id] for case in group_cases]
                )
                for case in group_cases:
                    if case.case_id in loaded:
                        existing = TypedDecision.from_value(
                            loaded[case.case_id].get("scores", {}).get(
                                "branch_discrimination"
                            )
                        )
                        if existing != branch_decision:
                            raise ResumeConflictError(
                                f"resumed group judgment conflicts for {group_id}"
                            )
            for case in group_cases:
                if case.case_id in loaded:
                    result = loaded[case.case_id]
                    checkpoint = json.loads(self._checkpoint_path(case).read_text(encoding="utf-8"))
                    descriptors[case.case_id] = {
                        "case_id": case.case_id,
                        "result_id": result["result_id"],
                        "path": checkpoint["result_path"],
                        "sha256": checkpoint["result_file_sha256"],
                        "size_bytes": (self.output_dir / checkpoint["result_path"]).stat().st_size,
                    }
                else:
                    result = _finalize_case_result(drafts[case.case_id], branch_decision)
                    descriptors[case.case_id] = self._write_case(case, result)
                results.append(result)
        results.sort(key=lambda item: str(item["case_id"]))
        summary = aggregate_results(results, self.config)
        summary["run_identity_sha256"] = self.run_identity_hash
        summary["case_result_ids"] = [str(item["result_id"]) for item in results]
        summary_base = {key: value for key, value in summary.items() if key != "summary_id"}
        summary["summary_id"] = content_hash(summary_base)
        summary_relative = Path("summaries") / f"{_id_digest(summary['summary_id'])}.json"
        summary_payload = _json_document(summary)
        _immutable_write(self.output_dir / summary_relative, summary_payload)
        ordered_descriptors = [descriptors[key] for key in sorted(descriptors)]
        manifest_base = {
            "schema": MANIFEST_SCHEMA,
            "status": summary["status"],
            "promotion_eligible": summary["promotion_eligible"],
            "run_identity_sha256": self.run_identity_hash,
            "artifact_hashes": {
                "checkpoint_sha256": self.config.checkpoint_sha256,
                "dataset_sha256": self.config.dataset_sha256,
                "split_sha256": self.config.split_sha256,
                "summary_sha256": sha256_bytes(summary_payload),
            },
            "evaluation_boundary": dict(EVALUATION_BOUNDARY),
            "summary": {
                "summary_id": summary["summary_id"],
                "path": summary_relative.as_posix(),
                "sha256": sha256_bytes(summary_payload),
                "size_bytes": len(summary_payload),
            },
            "case_results": ordered_descriptors,
        }
        manifest = {**manifest_base, "manifest_id": content_hash(manifest_base)}
        manifest_payload = _json_document(manifest)
        _immutable_write(self.output_dir / "manifest.json", manifest_payload)
        return {"summary": summary, "manifest": manifest, "results": results}


def evaluate_generated_control_v5(
    groups: Sequence[Mapping[str, Any]],
    *,
    config: EvaluationConfig,
    generator: GenerationAdapter | Callable[[GenerationRequest], Any],
    asr: ASRAdapter | Callable[[ASRRequest], Any],
    judge: TypedJudgeAdapter | Callable[[CaseAdjudicationRequest], Any],
    output_dir: Path,
    device_admission: DeviceAdmission,
    host_ram_admission: HostRamAdmission,
    dataset_root: Path | None = None,
    dataset_bytes: bytes | None = None,
    split_bytes: bytes | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Convenience entry point retaining all injectable boundaries."""

    return GeneratedControlEvaluationHarness(
        config=config,
        generator=generator,
        asr=asr,
        judge=judge,
        output_dir=output_dir,
        device_admission=device_admission,
        host_ram_admission=host_ram_admission,
        dataset_root=dataset_root,
        resume=resume,
    ).run(groups, dataset_bytes=dataset_bytes, split_bytes=split_bytes)


__all__ = [
    "ALLOWED_PHYSICAL_CUDA_DEVICES",
    "ALL_DIMENSIONS",
    "ASRAdapter",
    "ASREvidence",
    "ASRRequest",
    "AdjudicationContractError",
    "BranchDiscriminationRequest",
    "CASE_RESULT_SCHEMA",
    "CudaAdmission",
    "CudaAdmissionError",
    "CudaDeviceSnapshot",
    "EVALUATION_BOUNDARY",
    "EvaluationCase",
    "EvaluationConfig",
    "EvaluationContractError",
    "GeneratedControlEvaluationHarness",
    "GenerationAdapter",
    "GenerationContractError",
    "GenerationOutput",
    "GenerationRequest",
    "HostMemorySnapshot",
    "HostRamAdmission",
    "HostRamThrottleError",
    "LeakageError",
    "PrefixResolver",
    "REQUIRED_ROLES",
    "ResumeConflictError",
    "StaticCudaAdmission",
    "StratumPolicy",
    "TypedDecision",
    "TypedJudgeAdapter",
    "aggregate_results",
    "canonical_json_bytes",
    "canonical_jsonl_bytes",
    "content_hash",
    "evaluate_generated_control_v5",
    "prepare_evaluation_cases",
    "require_sha256",
    "score_generated_evidence",
    "score_runtime_events",
    "sha256_bytes",
    "sha256_path",
    "verify_exact_hash",
    "wilson_95_lower",
    "wilson_interval",
]
