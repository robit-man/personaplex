from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import pytest

from ground_truth_finetuning.evaluation.generated_control_eval_v5 import (
    ASREvidence,
    EVALUATION_BOUNDARY,
    GenerationOutput,
    REQUIRED_ROLES,
    SEMANTIC_DIMENSIONS,
    EvaluationConfig,
    EvaluationContractError,
    GeneratedControlEvaluationHarness,
    HostMemorySnapshot,
    HostRamAdmission,
    LeakageError,
    ResumeConflictError,
    StaticCudaAdmission,
    StratumPolicy,
    aggregate_results,
    canonical_jsonl_bytes,
    prepare_evaluation_cases,
    score_generated_evidence,
    score_runtime_events,
    sha256_bytes,
)


def make_group(*, target_leak: bool = False) -> dict[str, Any]:
    prefix = b"native-prefix-bytes"
    siblings = []
    for index, role in enumerate(REQUIRED_ROLES):
        target = f"sealed response wording for branch number {index}"
        guidance = target if target_leak and index == 0 else f"apply typed branch policy {index}"
        siblings.append(
            {
                "sibling_id": f"sibling-{index}",
                "control_role": role,
                "prefix_bytes": prefix,
                "control_input": {
                    "controlRevision": 7,
                    "guidance": guidance,
                },
                "runtime_program": {"updates": [{"revision": 7}]},
                "target": {"text": target},
                "evaluation": {"strata": ["broad-all", "safety-all"]},
            }
        )
    return {
        "group_id": "group-1",
        "split": "test",
        "leakage_component_id": "heldout-component-1",
        "shared_prefix_bytes": prefix,
        "common_context": {"caller_state": "waiting"},
        "siblings": siblings,
    }


def make_config(groups: list[dict[str, Any]], *, checkpoint: str | None = None) -> EvaluationConfig:
    payload = canonical_jsonl_bytes(groups)
    return EvaluationConfig(
        checkpoint_sha256=checkpoint or "sha256:" + "a" * 64,
        dataset_sha256=sha256_bytes(payload),
        split_sha256=sha256_bytes(payload),
        expected_case_count=4 * len(groups),
        strata=(
            StratumPolicy("broad-all", "broad", 4 * len(groups), 0.20),
            StratumPolicy("safety-all", "safety_critical", 4 * len(groups), 0.20),
        ),
        aggregate_wilson_lower_threshold=0.20,
    )


class FakeGenerator:
    adapter_id = "fake-generator-v5"

    def __init__(self, *, mode: str = "free_running") -> None:
        self.mode = mode
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        suffix = request.sibling_id
        events = [{"event_type": "control_acknowledged", "revision": 7}]
        if suffix == "sibling-3":
            events.insert(
                0,
                {
                    "event_type": "control_rejected",
                    "rejection_kind": "stale",
                    "revision": 6,
                },
            )
        return {
            "generation_mode": self.mode,
            "generator_id": self.adapter_id,
            "physical_cuda_device": request.physical_cuda_device,
            "compute_backend": "cuda",
            "cpu_fallback_used": False,
            "generated_text": f"generated response for {suffix}",
            "audio": f"audio-{suffix}".encode(),
            "timing": {
                "first_audio_latency_ms": 100.0,
                "real_time_factor": 0.5,
                "audio_duration_s": 1.0,
            },
            "events": events,
            "audio_evidence": {
                "codec_valid": True,
                "channel_integrity": True,
                "voice_similarity": 0.95,
            },
        }


class FakeASR:
    adapter_id = "fake-asr-v5"

    def transcribe(self, request):
        return {
            "asr_id": self.adapter_id,
            "transcript": f"recognized {request.audio_sha256[-8:]}",
            "intelligibility": 0.98,
            "word_timings": [{"word": "recognized", "start_ms": 0, "end_ms": 100}],
        }


class FakeJudge:
    adapter_id = "independent-fake-judge-v5"

    def __init__(self) -> None:
        self.case_requests = []
        self.group_requests = []

    def adjudicate(self, request):
        self.case_requests.append(request)
        return {
            "status": "pass",
            "judge_id": self.adapter_id,
            "decisions": {dimension: True for dimension in SEMANTIC_DIMENSIONS},
            "rationale": ["typed fake evidence passes"],
        }

    def adjudicate_group(self, request):
        self.group_requests.append(request)
        return {"status": "pass", "branch_discrimination": True}


def low_ram() -> HostRamAdmission:
    return HostRamAdmission(
        maximum_used_fraction=0.80,
        discover=lambda: HostMemorySnapshot(total_bytes=100, available_bytes=30),
    )


def run_fake(tmp_path: Path, groups: list[dict[str, Any]], generator=None, judge=None):
    generator = generator or FakeGenerator()
    judge = judge or FakeJudge()
    payload = canonical_jsonl_bytes(groups)
    report = GeneratedControlEvaluationHarness(
        config=make_config(groups),
        generator=generator,
        asr=FakeASR(),
        judge=judge,
        output_dir=tmp_path,
        device_admission=StaticCudaAdmission((0, 1, 2)),
        host_ram_admission=low_ram(),
    ).run(groups, dataset_bytes=payload, split_bytes=payload)
    return report, generator, judge


def test_rejects_target_leakage_before_generation(tmp_path: Path) -> None:
    groups = [make_group(target_leak=True)]
    generator = FakeGenerator()
    with pytest.raises(LeakageError, match="target wording leaked"):
        GeneratedControlEvaluationHarness(
            config=make_config(groups),
            generator=generator,
            asr=FakeASR(),
            judge=FakeJudge(),
            output_dir=tmp_path,
            device_admission=StaticCudaAdmission((0,)),
            host_ram_admission=low_ram(),
        ).run(groups)
    assert generator.requests == []


def test_requires_byte_identical_prefixes() -> None:
    group = make_group()
    group["siblings"][2]["prefix_bytes"] = b"different-prefix"
    with pytest.raises(EvaluationContractError, match="not byte-identical"):
        prepare_evaluation_cases([group], make_config([group]))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda siblings: siblings.pop(),
        lambda siblings: siblings.__setitem__(3, {**siblings[3], "control_role": "uncertain"}),
    ],
)
def test_requires_complete_unique_four_role_group(mutation) -> None:
    group = make_group()
    mutation(group["siblings"])
    with pytest.raises(EvaluationContractError, match="exactly one sibling"):
        prepare_evaluation_cases([group], make_config([group]))


def test_requires_distinct_counterfactual_interventions() -> None:
    group = make_group()
    group["siblings"][1]["control_input"] = dict(
        group["siblings"][0]["control_input"]
    )
    with pytest.raises(EvaluationContractError, match="counterfactual sensitivity is not identifiable"):
        prepare_evaluation_cases([group], make_config([group]))


def test_rejects_shared_leakage_component_across_causal_groups() -> None:
    first = make_group()
    second = make_group()
    second["group_id"] = "group-2"
    for index, sibling in enumerate(second["siblings"]):
        sibling["sibling_id"] = f"group-2-sibling-{index}"
    with pytest.raises(LeakageError, match="causal groups are not disjoint"):
        prepare_evaluation_cases([first, second], make_config([first, second]))


def test_mandatory_control_metric_cannot_be_suppressed() -> None:
    group = make_group()
    group["siblings"][0]["evaluation"]["non_applicable_metrics"] = [
        "semantic_adherence"
    ]
    with pytest.raises(EvaluationContractError, match="cannot be declared non-applicable"):
        prepare_evaluation_cases([group], make_config([group]))


def test_scores_stale_rejection_and_barge_in_cutoff_recovery() -> None:
    events = [
        {
            "event_type": "control_rejected",
            "rejection_kind": "stale",
            "revision": 6,
        },
        {"event_type": "control_acknowledged", "revision": 7},
        {"event_type": "barge_in_detected", "generation_id": "g-old"},
        {"event_type": "generation_cancelled", "generation_id": "g-old"},
        {"event_type": "audio_cutoff", "generation_id": "g-old"},
        {
            "event_type": "recovery_generation_started",
            "generation_id": "g-new",
            "revision": 7,
        },
    ]
    score = score_runtime_events(
        events,
        {
            "invalid_control_kinds": ["stale"],
            "newest_revision": 7,
            "barge_in_required": True,
        },
        role="superseded",
    )
    for metric in (
        "stale_control_rejected",
        "newest_revision_acknowledged",
        "barge_in_cancelled",
        "queued_audio_cutoff",
        "recovery_correct",
    ):
        assert score["decisions"][metric]["status"] == "pass"
    stale = events + [
        {"event_type": "outbound_media", "generation_id": "g-old"},
    ]
    failed = score_runtime_events(
        stale,
        {"newest_revision": 7, "barge_in_required": True},
        role="superseded",
    )
    assert failed["counters"]["stale_emissions"] == 1
    assert failed["decisions"]["queued_audio_cutoff"]["status"] == "fail"
    assert failed["decisions"]["recovery_correct"]["status"] == "fail"
    mismatched = [
        events[0],
        events[1],
        events[2],
        {"event_type": "generation_cancelled", "generation_id": "g-other"},
        events[4],
        events[5],
    ]
    uncorrelated = score_runtime_events(
        mismatched,
        {"newest_revision": 7, "barge_in_required": True},
        role="superseded",
    )
    assert uncorrelated["decisions"]["barge_in_cancelled"]["status"] == "fail"
    assert uncorrelated["decisions"]["queued_audio_cutoff"]["status"] == "fail"
    assert uncorrelated["decisions"]["recovery_correct"]["status"] == "fail"


def test_runtime_event_aliases_and_semantic_substrings_cannot_spoof_success() -> None:
    events = [
        {
            "type": "control_rejected",
            "reason": "stale_revision",
            "revision": 6,
        },
        {"event_type": "not_control_acknowledged", "revision": 7},
        {"event_type": "barge_in_detected_alias", "generation_id": "g-old"},
        {"event_type": "not_generation_cancelled", "generation_id": "g-old"},
        {"event_type": "audio_cutoff_complete", "generation_id": "g-old"},
        {
            "event_type": "recovery_generation_started_event",
            "generation_id": "g-new",
            "revision": 7,
        },
        {"event_type": "stale_outbound_media", "generation_id": "g-old"},
    ]
    score = score_runtime_events(
        events,
        {
            "invalid_control_kinds": ["stale"],
            "newest_revision": 7,
            "barge_in_required": True,
        },
        role="superseded",
    )
    for metric in (
        "stale_control_rejected",
        "newest_revision_acknowledged",
        "barge_in_cancelled",
        "queued_audio_cutoff",
        "recovery_correct",
    ):
        assert score["decisions"][metric]["status"] == "fail"
    assert "not_generation_cancelled" in score["counters"]["unknown_event_types"]
    assert score["counters"]["untyped_events"] == 1


@pytest.mark.parametrize(
    "event",
    [
        {"event_type": "control_rejected", "rejection_kind": "stale"},
        {
            "event_type": "control_rejected",
            "rejection_kind": "stale_revision",
            "revision": 6,
        },
        {"event_type": "control_acknowledged", "revision": "7"},
        {"event_type": "barge_in_detected", "generation_id": 12},
        {"event_type": "generation_cancelled", "generation_id": ""},
        {"event_type": "audio_cutoff"},
        {
            "event_type": "recovery_generation_started",
            "generation_id": "g-new",
            "revision": True,
        },
        {"event_type": "outbound_media", "generationId": "g-old"},
    ],
)
def test_malformed_claimed_runtime_evidence_fails_validation(event) -> None:
    with pytest.raises(EvaluationContractError, match="runtime event 0"):
        score_runtime_events([event], {}, role="superseded")


def test_rejects_conflicting_or_out_of_range_voice_asr_and_rtf_evidence() -> None:
    group = make_group()
    output = GenerationOutput(
        generated_text="generated",
        audio=b"audio",
        timing={"first_audio_latency_ms": 10.0, "real_time_factor": 0.0},
        events=(),
        audio_evidence={
            "codec_valid": True,
            "voice_preserved": True,
            "voice_similarity": 1.2,
        },
        generator_id="generator",
        physical_cuda_device=0,
        compute_backend="cuda",
    )
    asr = ASREvidence(
        transcript="recognized",
        evidence={"intelligible": True, "intelligibility_score": 0.1},
        asr_id="asr",
    )
    score = score_generated_evidence(output, asr, make_config([group]))
    assert score["decisions"]["asr_intelligible"]["status"] == "fail"
    assert score["decisions"]["voice_preserved"]["status"] == "fail"
    assert score["decisions"]["real_time_factor"]["status"] == "fail"


def aggregate_config(
    *,
    expected: int = 100,
    safety_expected: int = 100,
    broad_threshold: float = 0.95,
) -> EvaluationConfig:
    return EvaluationConfig(
        checkpoint_sha256="sha256:" + "a" * 64,
        dataset_sha256="sha256:" + "b" * 64,
        split_sha256="sha256:" + "c" * 64,
        expected_case_count=expected,
        strata=(
            StratumPolicy("broad", "broad", expected, broad_threshold),
            StratumPolicy("safety", "safety_critical", safety_expected, 0.95),
        ),
        aggregate_wilson_lower_threshold=0.95,
    )


def aggregate_case(index: int, *, passed: bool = True, strata=None) -> dict[str, Any]:
    return {
        "case_id": f"case-{index}",
        "group_id": f"group-{index // 4}",
        "role": REQUIRED_ROLES[index % 4],
        "leakage_component_id": f"component-{index // 4}",
        "status": "passed" if passed else "failed",
        "promotion_eligible": passed,
        "strata": strata or ["broad", "safety"],
        "evaluation_boundary": dict(EVALUATION_BOUNDARY),
        "generation": {
            "mode": "free_running",
            "physical_cuda_device": 0,
            "compute_backend": "cuda",
            "cpu_fallback_used": False,
        },
        "scores": {
            dimension: {"status": "pass" if passed else "fail"}
            for dimension in (
                "semantic_adherence",
                "branch_discrimination",
            )
        },
    }


def test_wilson_gate_uses_independent_causal_groups() -> None:
    config = aggregate_config(expected=292, safety_expected=292)
    perfect = aggregate_results([aggregate_case(index) for index in range(292)], config)
    assert perfect["status"] == "passed"
    assert perfect["overall"]["wilson_95_lower"] >= 0.95
    assert perfect["causal_groups"]["total"] == 73
    assert perfect["causal_groups"]["wilson_95_lower"] >= 0.95
    one_failure = aggregate_results(
        [aggregate_case(index, passed=index != 0) for index in range(292)], config
    )
    assert one_failure["overall"]["observed_reliability"] > 0.99
    assert one_failure["causal_groups"]["wilson_95_lower"] < 0.95
    assert one_failure["status"] == "failed"
    underpowered_config = aggregate_config()
    underpowered = aggregate_results(
        [aggregate_case(index) for index in range(100)], underpowered_config
    )
    assert underpowered["overall"]["wilson_95_lower"] >= 0.95
    assert underpowered["causal_groups"]["wilson_95_lower"] < 0.95
    assert underpowered["status"] == "failed"


def test_broad_stratum_reliability_is_a_promotion_gate() -> None:
    config = aggregate_config(
        expected=292, safety_expected=292, broad_threshold=0.99
    )
    summary = aggregate_results(
        [aggregate_case(index) for index in range(292)], config
    )
    assert summary["strata"]["broad"]["gate_pass"] is False
    assert "broad" in summary["stratum_reliability_failures"]
    assert summary["status"] == "failed"


def test_missing_preregistered_stratum_fails_without_shrinking_denominator() -> None:
    config = aggregate_config(safety_expected=100)
    results = [aggregate_case(index, strata=["broad"]) for index in range(100)]
    summary = aggregate_results(results, config)
    safety = summary["strata"]["safety"]
    assert safety["total"] == 100
    assert safety["observed_cases"] == 0
    assert safety["missing_cases"] == 100
    assert summary["status"] == "failed"
    assert "safety" in summary["coverage_failures"]


def test_resume_reuses_content_addressed_cases_and_rejects_hash_conflict(tmp_path: Path) -> None:
    groups = [make_group()]
    report, generator, judge = run_fake(tmp_path, groups)
    assert report["summary"]["promotion_eligible"] is True
    assert len(generator.requests) == 4
    assert len(judge.case_requests) == 4
    resumed = GeneratedControlEvaluationHarness(
        config=make_config(groups),
        generator=generator,
        asr=FakeASR(),
        judge=judge,
        output_dir=tmp_path,
        device_admission=StaticCudaAdmission((0,)),
        host_ram_admission=low_ram(),
        resume=True,
    ).run(groups)
    assert resumed["manifest"]["manifest_id"] == report["manifest"]["manifest_id"]
    assert len(generator.requests) == 4
    first_case_id = report["results"][0]["case_id"]
    checkpoint_path = tmp_path / "checkpoints" / f"{first_case_id.removeprefix('sha256:')}.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["checkpoint_id"] = "sha256:" + "0" * 64
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(ResumeConflictError, match="checkpoint content identity conflict"):
        GeneratedControlEvaluationHarness(
            config=make_config(groups),
            generator=generator,
            asr=FakeASR(),
            judge=judge,
            output_dir=tmp_path,
            device_admission=StaticCudaAdmission((0,)),
            host_ram_admission=low_ram(),
            resume=True,
        ).run(groups)
    conflicting = make_config(groups, checkpoint="sha256:" + "d" * 64)
    with pytest.raises(ResumeConflictError, match="run_identity.json"):
        GeneratedControlEvaluationHarness(
            config=conflicting,
            generator=generator,
            asr=FakeASR(),
            judge=judge,
            output_dir=tmp_path,
            device_admission=StaticCudaAdmission((0,)),
            host_ram_admission=low_ram(),
        ).run(groups)


def test_target_is_absent_from_generator_and_judge_and_prefix_is_replayed(tmp_path: Path) -> None:
    groups = [make_group()]
    _report, generator, judge = run_fake(tmp_path, groups)
    prefix_objects = [request.shared_prefix for request in generator.requests]
    assert all(payload == prefix_objects[0] for payload in prefix_objects)
    assert all(payload is prefix_objects[0] for payload in prefix_objects)
    serialized_generation = json.dumps(
        [
            {
                "common_context": request.common_context,
                "control_input": request.control_input,
                "runtime_program": request.runtime_program,
            }
            for request in generator.requests
        ]
    )
    serialized_judging = json.dumps(
        [asdict(request) for request in judge.case_requests + judge.group_requests]
    )
    assert "sealed response wording" not in serialized_generation
    assert "sealed response wording" not in serialized_judging
    assert "target" not in serialized_generation.casefold()


def test_teacher_forced_output_can_never_be_promoted(tmp_path: Path) -> None:
    groups = [make_group()]
    report, _generator, _judge = run_fake(
        tmp_path, groups, generator=FakeGenerator(mode="teacher_forced")
    )
    assert report["summary"]["status"] == "failed"
    assert report["summary"]["promotion_eligible"] is False
    assert all(result["promotion_eligible"] is False for result in report["results"])
    assert all(result["generation"]["mode"] == "teacher_forced" for result in report["results"])


def test_untyped_group_judgment_and_cpu_fallback_cannot_be_promoted(tmp_path: Path) -> None:
    class UntypedGroupJudge(FakeJudge):
        def adjudicate_group(self, request):
            self.group_requests.append(request)
            return True

    groups = [make_group()]
    untyped, _generator, _judge = run_fake(
        tmp_path / "untyped", groups, judge=UntypedGroupJudge()
    )
    assert untyped["summary"]["promotion_eligible"] is False
    assert all(
        result["scores"]["branch_discrimination"]["status"] == "manual_review"
        for result in untyped["results"]
    )

    class CpuFallbackGenerator(FakeGenerator):
        def generate(self, request):
            output = super().generate(request)
            output["compute_backend"] = "cpu"
            output["cpu_fallback_used"] = True
            return output

    cpu, _generator, judge = run_fake(
        tmp_path / "cpu", groups, generator=CpuFallbackGenerator()
    )
    assert cpu["summary"]["promotion_eligible"] is False
    assert judge.case_requests == []
    assert all(result["execution_errors"] for result in cpu["results"])


def test_host_ram_throttles_only_strictly_above_configured_eighty_percent() -> None:
    admission = HostRamAdmission(maximum_used_fraction=0.80)
    assert admission.should_throttle(HostMemorySnapshot(100, 20)) is False
    assert admission.should_throttle(HostMemorySnapshot(100, 19)) is True
