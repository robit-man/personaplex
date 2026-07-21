#!/usr/bin/env python3
"""Materialize certified Voryn v5 branches as native four-sibling groups.

This bridge is deliberately downstream of Mimi encoding.  It verifies and
splits existing native tensors; it never synthesizes audio, invokes Mimi, or
repairs independently rendered prefixes.  Any ambiguous identity, timing, or
prefix alignment is emitted as a typed rerender rejection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence
import wave

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from ground_truth_finetuning.training.causal_group_pack import (  # noqa: E402
    CAUSAL_GROUP_ROLES,
    CausalGroupPackError,
    PackConfig,
    assign_component_splits,
    assert_no_target_text_leak,
    build_leakage_components,
    canonical_json,
    content_hash,
    normalize_causal_group,
)
from ground_truth_finetuning.training.contracts import (  # noqa: E402
    ContractError,
    assert_evidence_control_alignment,
    sha256_uri,
    validate_control_frame_mapping,
    validate_evidence_frame_mapping,
)
from ground_truth_finetuning.training.native_moshirag_control import (  # noqa: E402
    NATIVE_MOSHIRAG_CONTROL_SCHEMA,
)


TOOL_VERSION = "gtft-native-causal-group-v5-materializer-v1"
PLAN_SCHEMA = "personaplex.voryn-branch-artifact.v5"
GROUP_SCHEMA = "personaplex.native-causal-group.v5"
REJECTION_SCHEMA = "personaplex.native-causal-group-rerender-rejection.v1"
REPORT_SCHEMA = "personaplex.native-causal-group-materialization-report.v1"
GROUPS_FILENAME = "native_causal_groups_v5.jsonl"
REJECTIONS_FILENAME = "rerender_rejections_v5.jsonl"
REPORT_FILENAME = "materialization_report_v5.json"
TRAINER_DATASET_SCHEMA = "personaplex.native-moshirag-dataset.v2-shared-prefix"
TRAINER_GROUP_SCHEMA = "personaplex.native-moshirag-group.v2-shared-prefix"
TRAINER_SHARED_PREFIX_SCHEMA = "personaplex.native-shared-prefix.v1"
TRAINER_ALIGNMENT_SCHEMA = "personaplex.native-branch-window-alignment.v1"
TRAINER_MANIFEST_FILENAME = "native_moshirag_groups_v2.jsonl"
TRAINER_TEST_FILENAME = "native_moshirag_test_v2.jsonl"
TRAINER_ALL_SPLITS_FILENAME = "native_moshirag_all_splits_v2.jsonl"
TRAINER_CONTRACT_FILENAME = "native_moshirag_dataset_v2.json"
FRAME_DURATION_MS = 80


class BridgeError(ValueError):
    """A global source or invocation contract is invalid."""


class RerenderRequired(BridgeError):
    """One causal group cannot be represented without fabricating alignment."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class WavePayload:
    channels: int
    sample_width: int
    sample_rate: int
    compression_type: str
    compression_name: str
    frame_count: int
    frames: bytes


@dataclass(frozen=True)
class NativeBranch:
    plan: dict[str, Any]
    source: dict[str, Any]
    label: dict[str, Any]
    codes: torch.Tensor
    target_mask: torch.Tensor
    alignment: dict[str, Any]
    wave: WavePayload
    model_contract: dict[str, Any]
    pivot: int
    target_end: int
    first_supervised: int
    control_input: dict[str, Any]
    control_stream: torch.Tensor
    native_control: dict[str, Any]
    target_text: str
    target_text_hash: str
    actual_events: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_text(value: str) -> str:
    return "sha256:" + sha256(value.encode("utf-8")).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def require_sha256(value: Any, label: str) -> str:
    if not is_sha256(value):
        raise RerenderRequired("immutable_identity_invalid", f"{label} must be a sha256 URI")
    return str(value)


def require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RerenderRequired("source_contract_invalid", f"{label} must be nonempty text")
    return value.strip()


def require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RerenderRequired(
            "source_contract_invalid", f"{label} must be an integer >= {minimum}"
        )
    return value


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BridgeError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise BridgeError(f"{path} must contain one JSON object")
    return value


def read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise BridgeError(f"cannot read {label} {path}: {error}") from error
    if not values or not all(isinstance(value, dict) for value in values):
        raise BridgeError(f"{label} must contain one or more JSON objects")
    return values


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(canonical_json(dict(row)) + "\n" for row in rows),
        encoding="utf-8",
    )


def index_unique(
    rows: Sequence[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise BridgeError(f"{label} row lacks {key}")
        if value in output:
            raise BridgeError(f"{label} duplicates {key} {value}")
        output[value] = row
    return output


def resolve_under(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise RerenderRequired("artifact_integrity_failed", f"{label} path is missing")
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if path != root and root not in path.parents:
        raise RerenderRequired("artifact_integrity_failed", f"{label} escapes its artifact root")
    if not path.is_file():
        raise RerenderRequired("artifact_integrity_failed", f"{label} does not exist")
    return path


def load_tensor(path: Path, name: str) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    except (OSError, RuntimeError) as error:
        raise RerenderRequired(
            "artifact_integrity_failed", f"cannot load {name} tensor: {error}"
        ) from error
    tensor = value.get(name) if isinstance(value, dict) else value
    if not isinstance(tensor, torch.Tensor):
        raise RerenderRequired(
            "artifact_integrity_failed", f"{path.name} does not contain {name}"
        )
    return tensor.detach().cpu().contiguous()


def load_wave(path: Path) -> WavePayload:
    try:
        with wave.open(str(path), "rb") as handle:
            parameters = handle.getparams()
            frames = handle.readframes(parameters.nframes)
    except (OSError, wave.Error) as error:
        raise RerenderRequired(
            "artifact_integrity_failed", f"cannot read duplex WAV: {error}"
        ) from error
    if parameters.comptype != "NONE":
        raise RerenderRequired(
            "artifact_integrity_failed", "duplex WAV must use uncompressed PCM"
        )
    return WavePayload(
        channels=parameters.nchannels,
        sample_width=parameters.sampwidth,
        sample_rate=parameters.framerate,
        compression_type=parameters.comptype,
        compression_name=parameters.compname,
        frame_count=parameters.nframes,
        frames=frames,
    )


def write_wave(path: Path, payload: WavePayload, frames: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(payload.channels)
        handle.setsampwidth(payload.sample_width)
        handle.setframerate(payload.sample_rate)
        handle.setcomptype(payload.compression_type, payload.compression_name)
        handle.writeframes(frames)


def save_tensor(path: Path, name: str, value: torch.Tensor) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({name: value.contiguous()}, path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "tensorSha256": tensor_hash(value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def validate_certificate(
    certificate: Mapping[str, Any], native_manifest: Path, native_root: Path, row_count: int
) -> None:
    failures: list[str] = []
    if certificate.get("kind") != "personaplex-corpus-certificate":
        failures.append("unsupported certificate kind")
    if certificate.get("status") != "certified_for_adapter_training":
        failures.append("source is not certified_for_adapter_training")
    if certificate.get("failed_items") != 0:
        failures.append("certificate contains failed items")
    if certificate.get("caller_stream_supervision") != "forbidden":
        failures.append("certificate does not forbid caller supervision")
    if certificate.get("manifest_sha256") != sha256_file(native_manifest):
        failures.append("certificate does not bind the native manifest bytes")
    if certificate.get("items") != row_count:
        failures.append("certificate item count differs from native manifest")
    artifact_root = certificate.get("artifact_root")
    if not isinstance(artifact_root, str) or Path(artifact_root).resolve() != native_root:
        failures.append("certificate artifact root differs from the supplied native root")
    if failures:
        raise BridgeError("source certificate rejected: " + "; ".join(failures))


def validate_plan_record(plan: dict[str, Any]) -> tuple[str, str]:
    if plan.get("schema") != PLAN_SCHEMA:
        raise RerenderRequired("plan_contract_invalid", f"plan schema must be {PLAN_SCHEMA}")
    plan_id = require_sha256(plan.get("planRecordId"), "planRecordId")
    immutable_payload = {key: value for key, value in plan.items() if key != "planRecordId"}
    if content_hash(immutable_payload) != plan_id:
        raise RerenderRequired(
            "immutable_identity_invalid", "planRecordId does not bind the canonical plan payload"
        )
    group_id = require_text(plan.get("groupId"), "groupId")
    role = require_text(plan.get("siblingRole"), "siblingRole")
    if role not in CAUSAL_GROUP_ROLES:
        raise RerenderRequired("group_role_contract_failed", f"unsupported sibling role {role}")
    return group_id, role


def extract_plan_record_id(source: Mapping[str, Any]) -> str:
    provenance = source.get("provenance")
    if not isinstance(provenance, Mapping):
        raise RerenderRequired("immutable_join_failed", "pre-codec row lacks provenance")
    return require_sha256(provenance.get("plan_record_id"), "provenance.plan_record_id")


def target_transcript(label: Mapping[str, Any]) -> tuple[str, str]:
    text = label.get("target_transcript")
    if not isinstance(text, str) or not text.strip():
        raise RerenderRequired(
            "target_label_contract_invalid",
            "v5 control label must carry target_transcript outside the control input",
        )
    text = text.strip()
    digest = require_sha256(label.get("target_label_sha256"), "target_label_sha256")
    if sha256_text(text) != digest:
        raise RerenderRequired(
            "target_label_contract_invalid", "target transcript does not match its immutable hash"
        )
    return text, digest


def model_contract(encoding: Mapping[str, Any], codebooks: int) -> dict[str, Any]:
    codec = encoding.get("codec")
    layout = encoding.get("codebook_layout")
    if not isinstance(codec, Mapping) or not isinstance(layout, Mapping):
        raise RerenderRequired(
            "model_contract_mismatch", "native encoding lacks codec or stream-layout contract"
        )
    contract = {
        "modelRevision": require_text(encoding.get("model_revision"), "model_revision"),
        "delayConfigSha256": require_sha256(
            encoding.get("delay_config_sha256"), "delay_config_sha256"
        ),
        "mimiWeightsSha256": require_sha256(
            codec.get("mimi_weights_sha256"), "mimi_weights_sha256"
        ),
        "tokenizerSha256": require_sha256(codec.get("tokenizer_sha256"), "tokenizer_sha256"),
        "frameRateHz": codec.get("frame_rate_hz"),
        "codebookLayout": deepcopy(dict(layout)),
    }
    frame_rate = contract["frameRateHz"]
    if not isinstance(frame_rate, (int, float)) or isinstance(frame_rate, bool) or frame_rate <= 0:
        raise RerenderRequired("model_contract_mismatch", "native frame rate is invalid")
    groups: dict[str, list[int]] = {}
    all_indices: list[int] = []
    for name in ("text", "agent_audio", "caller_audio"):
        values = layout.get(name)
        if not isinstance(values, list) or not values or not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in values
        ):
            raise RerenderRequired("model_contract_mismatch", f"invalid {name} stream indices")
        groups[name] = list(values)
        all_indices.extend(values)
    if len(all_indices) != len(set(all_indices)) or set(all_indices) != set(range(codebooks)):
        raise RerenderRequired(
            "model_contract_mismatch", "stream layout is not a disjoint complete partition"
        )
    contract["contractHash"] = content_hash(contract)
    return contract


def validate_actual_events(value: Any, frames: int, pivot: int) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"bargeIn", "cancellation", "endCall"}:
        raise RerenderRequired(
            "runtime_event_invalid",
            "actualEvents must contain exactly bargeIn, cancellation, and endCall",
        )
    barge = value["bargeIn"]
    cancellation = value["cancellation"]
    end_call = value["endCall"]
    if not isinstance(barge, Mapping) or set(barge) != {"occurred", "cutoffFrame"}:
        raise RerenderRequired("runtime_event_invalid", "bargeIn event is not typed")
    occurred = barge.get("occurred")
    cutoff = barge.get("cutoffFrame")
    if not isinstance(occurred, bool):
        raise RerenderRequired("runtime_event_invalid", "bargeIn.occurred must be boolean")
    if occurred:
        cutoff = require_int(cutoff, "bargeIn.cutoffFrame", minimum=pivot)
        if cutoff >= frames:
            raise RerenderRequired("runtime_event_invalid", "barge-in cutoff is outside native audio")
    elif cutoff is not None:
        raise RerenderRequired("runtime_event_invalid", "non-barge-in event cannot carry a cutoff")

    if not isinstance(cancellation, Mapping) or set(cancellation) != {
        "generationId", "cancelled", "atFrame"
    }:
        raise RerenderRequired("runtime_event_invalid", "cancellation event is not typed")
    generation_id = require_text(cancellation.get("generationId"), "cancellation.generationId")
    cancelled = cancellation.get("cancelled")
    at_frame = cancellation.get("atFrame")
    if not isinstance(cancelled, bool):
        raise RerenderRequired("runtime_event_invalid", "cancellation.cancelled must be boolean")
    if cancelled:
        at_frame = require_int(at_frame, "cancellation.atFrame", minimum=pivot)
        if at_frame >= frames:
            raise RerenderRequired("runtime_event_invalid", "cancellation frame is outside native audio")
    elif at_frame is not None:
        raise RerenderRequired("runtime_event_invalid", "uncancelled generation cannot carry atFrame")
    if occurred and (not cancelled or int(at_frame) < int(cutoff)):
        raise RerenderRequired(
            "runtime_event_invalid", "barge-in must cancel the active generation at or after cutoff"
        )

    if not isinstance(end_call, Mapping) or set(end_call) != {
        "decision", "decisionFrame", "toolCallFrame"
    }:
        raise RerenderRequired("runtime_event_invalid", "endCall event is not typed")
    decision = end_call.get("decision")
    if decision not in {"continue", "end_call"}:
        raise RerenderRequired("runtime_event_invalid", "endCall.decision is invalid")
    decision_frame = require_int(end_call.get("decisionFrame"), "endCall.decisionFrame")
    if decision_frame >= frames:
        raise RerenderRequired("runtime_event_invalid", "end-call decision is outside native audio")
    tool_frame = end_call.get("toolCallFrame")
    if decision == "end_call":
        tool_frame = require_int(tool_frame, "endCall.toolCallFrame", minimum=decision_frame)
        if tool_frame >= frames:
            raise RerenderRequired("runtime_event_invalid", "end-call tool event is outside native audio")
    elif tool_frame is not None:
        raise RerenderRequired("runtime_event_invalid", "continue decision cannot carry toolCallFrame")
    return deepcopy(dict(value))


def validate_control(
    plan: Mapping[str, Any], label: Mapping[str, Any], source: Mapping[str, Any],
    common_context: Mapping[str, Any], first_supervised: int, target_text: str,
    target_hash: str,
) -> dict[str, Any]:
    binding = plan.get("controlBinding")
    if not isinstance(binding, Mapping) or set(binding).difference(
        {"frameHash", "revision", "availableFrame", "evidenceFrameHash"}
    ):
        raise RerenderRequired("control_binding_invalid", "controlBinding is missing or untyped")
    if not {"frameHash", "revision", "availableFrame"}.issubset(binding):
        raise RerenderRequired("control_binding_invalid", "controlBinding lacks required fields")
    frame_value = label.get("control_frame")
    if not isinstance(frame_value, Mapping):
        raise RerenderRequired("control_binding_invalid", "control label lacks control_frame")
    try:
        frame = validate_control_frame_mapping(frame_value)
    except ContractError as error:
        raise RerenderRequired("control_binding_invalid", str(error)) from error
    frame_hash = frame.frame_hash
    if any(
        value != frame_hash
        for value in (
            label.get("control_frame_hash"),
            binding.get("frameHash"),
            (source.get("control") or {}).get("frame_hash"),
        )
    ):
        raise RerenderRequired(
            "immutable_join_failed", "control frame hashes disagree across plan, label, and source"
        )
    if canonical_json((source.get("control") or {}).get("frame")) != canonical_json(
        frame.as_wire_dict()
    ):
        raise RerenderRequired(
            "immutable_join_failed", "native source control frame differs from certified label"
        )
    revision = require_int(binding.get("revision"), "controlBinding.revision", minimum=1)
    if revision != frame.state_revision or revision != frame.plan.revision:
        raise RerenderRequired(
            "control_binding_invalid", "control revision differs from typed frame revisions"
        )
    available = require_int(binding.get("availableFrame"), "controlBinding.availableFrame")
    if available >= first_supervised:
        raise RerenderRequired(
            "control_timing_invalid",
            "controlAvailableFrame must be strictly before the first supervised target frame",
        )
    serialized = canonical_json(frame.as_wire_dict()).casefold()
    if target_hash[7:] in serialized:
        raise RerenderRequired("target_leakage", "target transcript hash leaked into control input")
    try:
        assert_no_target_text_leak(target_text, common_context, frame.as_wire_dict())
    except CausalGroupPackError as error:
        raise RerenderRequired("target_leakage", str(error)) from error
    output: dict[str, Any] = {
        "controlFrame": frame.as_wire_dict(),
        "controlFrameHash": frame_hash,
        "controlRevision": revision,
        "controlAvailableFrame": available,
    }
    evidence_value = label.get("evidence_frame")
    if evidence_value is not None:
        if not isinstance(evidence_value, Mapping):
            raise RerenderRequired("control_binding_invalid", "evidence frame is not an object")
        try:
            evidence = validate_evidence_frame_mapping(evidence_value)
            assert_evidence_control_alignment(frame, evidence)
        except ContractError as error:
            raise RerenderRequired("control_binding_invalid", str(error)) from error
        evidence_wire = evidence.as_wire_dict()
        evidence_hash = sha256_uri(evidence_wire)
        if label.get("evidence_frame_hash") != evidence_hash:
            raise RerenderRequired("immutable_join_failed", "evidence frame hash disagrees")
        if binding.get("evidenceFrameHash") not in {None, evidence_hash}:
            raise RerenderRequired("immutable_join_failed", "plan evidence hash disagrees")
        output["evidenceFrame"] = evidence_wire
        output["evidenceFrameHash"] = evidence_hash
        try:
            assert_no_target_text_leak(target_text, evidence_wire)
        except CausalGroupPackError as error:
            raise RerenderRequired("target_leakage", str(error)) from error
    return output


def load_native_control(
    encoding: Mapping[str, Any], native_root: Path, frame_hash: str, revision: int,
    first_supervised: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    value = encoding.get("native_control")
    required = {
        "schema", "stream_path", "stream_key", "stream_sha256", "control_frame_hash",
        "control_revision", "acknowledged_control_revision", "control_active_frame",
        "retrieval_buffer_frames", "probe_frame_index", "probe_targets",
        "probe_slot_cardinalities", "padding_token_id",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise RerenderRequired(
            "native_control_stream_invalid",
            "model_encoding.native_control must contain the exact v2 trainer binding fields",
        )
    if value.get("schema") != NATIVE_MOSHIRAG_CONTROL_SCHEMA:
        raise RerenderRequired("native_control_stream_invalid", "native control schema is invalid")
    path = resolve_under(native_root, value.get("stream_path"), "native control stream")
    if sha256_file(path) != value.get("stream_sha256"):
        raise RerenderRequired(
            "artifact_integrity_failed", "native control-stream file hash differs from manifest"
        )
    key = require_text(value.get("stream_key"), "native_control.stream_key")
    stream = load_tensor(path, key)
    if stream.ndim != 2 or stream.shape[0] < 1 or not stream.is_floating_point():
        raise RerenderRequired(
            "native_control_stream_invalid", "control stream must be floating [frames, hidden]"
        )
    if not torch.isfinite(stream).all().item():
        raise RerenderRequired("native_control_stream_invalid", "control stream contains non-finite values")
    if value.get("control_frame_hash") != frame_hash:
        raise RerenderRequired(
            "immutable_join_failed", "native control stream binds a different control frame"
        )
    if value.get("control_revision") != revision or value.get("acknowledged_control_revision") != revision:
        raise RerenderRequired(
            "native_control_stream_invalid", "native control revision is stale or unacknowledged"
        )
    active = require_int(value.get("control_active_frame"), "control_active_frame")
    buffer_frames = require_int(value.get("retrieval_buffer_frames"), "retrieval_buffer_frames")
    probe_frame = require_int(value.get("probe_frame_index"), "probe_frame_index")
    if active + buffer_frames > first_supervised:
        raise RerenderRequired(
            "control_timing_invalid", "buffered native control becomes active after supervision"
        )
    if not active + buffer_frames <= probe_frame <= first_supervised:
        raise RerenderRequired(
            "control_timing_invalid", "probe frame is not control-visible before response"
        )
    targets = value.get("probe_targets")
    cardinalities = value.get("probe_slot_cardinalities")
    if (
        not isinstance(targets, Mapping)
        or not targets
        or not isinstance(cardinalities, Mapping)
        or set(targets) != set(cardinalities)
    ):
        raise RerenderRequired("native_control_stream_invalid", "probe slots are incomplete")
    for slot, target in targets.items():
        cardinality = cardinalities[slot]
        if (
            not isinstance(slot, str)
            or not slot
            or not isinstance(target, int)
            or isinstance(target, bool)
            or not isinstance(cardinality, int)
            or isinstance(cardinality, bool)
            or cardinality < 2
            or not 0 <= target < cardinality
        ):
            raise RerenderRequired("native_control_stream_invalid", "probe target/cardinality is invalid")
    padding = value.get("padding_token_id")
    if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
        raise RerenderRequired("native_control_stream_invalid", "padding_token_id is invalid")
    metadata = deepcopy(dict(value))
    metadata["control_hidden_size"] = int(stream.shape[1])
    return stream, metadata


def load_native_branch(
    plan: dict[str, Any], source: dict[str, Any], label: dict[str, Any],
    native_root: Path, precodec_root: Path,
) -> NativeBranch:
    plan_id = require_sha256(plan.get("planRecordId"), "planRecordId")
    if extract_plan_record_id(source) != plan_id:
        raise RerenderRequired("immutable_join_failed", "plan and pre-codec planRecordId differ")
    example_id = require_sha256(source.get("example_id"), "example_id")
    if label.get("example_id") != example_id:
        raise RerenderRequired("immutable_join_failed", "control label example_id differs")
    source_export_id = (source.get("provenance") or {}).get("source_export_example_id")
    source_export_id = require_text(source_export_id, "source_export_example_id")
    declared_source_id = plan.get("sourceExportExampleId")
    if declared_source_id is not None and declared_source_id != source_export_id:
        raise RerenderRequired("immutable_join_failed", "plan source-export ID differs")
    control_hash = require_sha256(label.get("control_frame_hash"), "control_frame_hash")
    expected_example_id = "sha256:" + sha256(
        f"{source_export_id}|{control_hash}".encode("utf-8")
    ).hexdigest()
    if example_id != expected_example_id:
        raise RerenderRequired(
            "immutable_join_failed", "native example_id does not bind source-export/control IDs"
        )

    group_id = require_text(plan.get("groupId"), "groupId")
    role = require_text(plan.get("siblingRole"), "siblingRole")
    counterfactual = source.get("counterfactual")
    if not isinstance(counterfactual, Mapping):
        raise RerenderRequired("immutable_join_failed", "source lacks counterfactual identity")
    source_role = counterfactual.get("siblingRole", counterfactual.get("branchId"))
    if counterfactual.get("groupId") != group_id or source_role != role:
        raise RerenderRequired(
            "immutable_join_failed", "source group/role differs from compiled plan"
        )

    encoding = source.get("model_encoding")
    if not isinstance(encoding, Mapping):
        raise RerenderRequired("artifact_integrity_failed", "source lacks model_encoding")
    codes_path = resolve_under(native_root, encoding.get("codes_path"), "codes")
    mask_path = resolve_under(native_root, encoding.get("target_mask_path"), "target mask")
    alignment_path = resolve_under(
        native_root, encoding.get("text_alignment_path"), "text alignment"
    )
    for path, expected, label_name in (
        (codes_path, encoding.get("codes_sha256"), "codes"),
        (mask_path, encoding.get("target_mask_sha256"), "target mask"),
        (alignment_path, encoding.get("text_alignment_sha256"), "text alignment"),
    ):
        if sha256_file(path) != expected:
            raise RerenderRequired(
                "artifact_integrity_failed", f"{label_name} file hash differs from manifest"
            )
    codes = load_tensor(codes_path, "codes")
    target_mask = load_tensor(mask_path, "target_mask")
    if codes.ndim != 2 or target_mask.shape != codes.shape or target_mask.dtype != torch.bool:
        raise RerenderRequired(
            "artifact_integrity_failed", "codes and bool target mask must share [K,T] shape"
        )
    contract = model_contract(encoding, int(codes.shape[0]))
    layout = contract["codebookLayout"]
    caller_indices = list(layout["caller_audio"])
    text_indices = list(layout["text"])
    agent_indices = list(layout["agent_audio"])
    if target_mask[caller_indices].any().item():
        raise RerenderRequired(
            "caller_supervision_forbidden", "caller audio stream contains target-mask bits"
        )
    if not target_mask[text_indices].any().item() or not target_mask[agent_indices].any().item():
        raise RerenderRequired(
            "target_label_contract_invalid", "agent text/audio supervision is incomplete"
        )
    supervised = target_mask.any(dim=0).nonzero().flatten()
    if supervised.numel() == 0:
        raise RerenderRequired("target_label_contract_invalid", "target mask is empty")
    first_supervised = int(supervised[0].item())

    try:
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RerenderRequired("artifact_integrity_failed", f"invalid alignment: {error}") from error
    if not isinstance(alignment, dict) or alignment.get("verified") is not True:
        raise RerenderRequired("artifact_integrity_failed", "alignment is not verified")
    if alignment.get("codes_sha256") != encoding.get("codes_sha256"):
        raise RerenderRequired("artifact_integrity_failed", "alignment codes hash is stale")
    target_frames = alignment.get("target_frames")
    if (
        not isinstance(target_frames, list)
        or len(target_frames) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in target_frames)
    ):
        raise RerenderRequired("native_pivot_mismatch", "alignment target_frames are invalid")
    pivot, target_end = target_frames
    if not 0 < pivot < target_end <= codes.shape[1]:
        raise RerenderRequired("native_pivot_mismatch", "alignment target range is invalid")
    declared_pivots = (plan.get("nativePivotFrame"), encoding.get("prefix_at"), pivot)
    if any(value != pivot for value in declared_pivots):
        raise RerenderRequired(
            "native_pivot_mismatch", "plan, encoding, and alignment native pivots disagree"
        )
    if target_mask[:, :pivot].any().item() or target_mask[:, target_end:].any().item():
        raise RerenderRequired(
            "native_pivot_mismatch", "target mask escapes its certified target-frame interval"
        )
    text, text_hash = target_transcript(label)
    if alignment.get("target_label_sha256") != text_hash:
        raise RerenderRequired("immutable_join_failed", "alignment target-label hash is stale")
    if (source.get("labels") or {}).get("agent_text_sha256") != text_hash:
        raise RerenderRequired("immutable_join_failed", "source target-label hash differs")

    common_context = plan.get("commonContext")
    if not isinstance(common_context, Mapping):
        raise RerenderRequired("shared_context_invalid", "plan lacks commonContext")
    context_hash = require_sha256(plan.get("commonContextHash"), "commonContextHash")
    if content_hash(common_context) != context_hash:
        raise RerenderRequired(
            "shared_context_invalid", "commonContextHash does not bind commonContext"
        )
    control_input = validate_control(
        plan, label, source, common_context, first_supervised, text, text_hash
    )
    control_stream, native_control = load_native_control(
        encoding,
        native_root,
        str(control_input["controlFrameHash"]),
        int(control_input["controlRevision"]),
        first_supervised,
    )
    if int(control_input["controlAvailableFrame"]) > int(native_control["control_active_frame"]):
        raise RerenderRequired(
            "control_timing_invalid", "control cannot become active before it is available"
        )

    duplex = source.get("duplex")
    if not isinstance(duplex, Mapping):
        raise RerenderRequired("artifact_integrity_failed", "source lacks duplex audio")
    wav_path = resolve_under(precodec_root, duplex.get("path"), "duplex audio")
    if sha256_file(wav_path) != duplex.get("sha256"):
        raise RerenderRequired("artifact_integrity_failed", "duplex audio hash differs")
    wav = load_wave(wav_path)
    if wav.sample_rate != duplex.get("sample_rate") or wav.channels != 2 or wav.sample_width != 2:
        raise RerenderRequired(
            "artifact_integrity_failed", "duplex WAV must match certified 16-bit stereo metadata"
        )
    events = validate_actual_events(plan.get("actualEvents"), int(codes.shape[1]), pivot)
    return NativeBranch(
        plan=plan,
        source=source,
        label=label,
        codes=codes,
        target_mask=target_mask,
        alignment=alignment,
        wave=wav,
        model_contract=contract,
        pivot=pivot,
        target_end=target_end,
        first_supervised=first_supervised,
        control_input=control_input,
        control_stream=control_stream,
        native_control=native_control,
        target_text=text,
        target_text_hash=text_hash,
        actual_events=events,
    )


def common_plan_value(branches: Sequence[NativeBranch], key: str) -> Any:
    values = {canonical_json(branch.plan.get(key)) for branch in branches}
    if len(values) != 1:
        raise RerenderRequired("shared_group_contract_mismatch", f"siblings disagree on {key}")
    return deepcopy(branches[0].plan.get(key))


def common_audio_prefix(branches: Sequence[NativeBranch], pivot: int) -> tuple[bytes, int]:
    reference = branches[0].wave
    frame_rate = float(branches[0].model_contract["frameRateHz"])
    samples_per_native = reference.sample_rate / frame_rate
    rounded = round(samples_per_native)
    if abs(samples_per_native - rounded) > 1e-9:
        raise RerenderRequired(
            "model_contract_mismatch", "native/audio frame ratio is not integral"
        )
    prefix_samples = pivot * int(rounded)
    if prefix_samples <= 0 or prefix_samples > reference.frame_count:
        raise RerenderRequired("native_pivot_mismatch", "native pivot exceeds duplex audio")
    width = reference.channels * reference.sample_width
    prefix = reference.frames[: prefix_samples * width]
    signature = (
        reference.channels,
        reference.sample_width,
        reference.sample_rate,
        reference.compression_type,
    )
    for branch in branches[1:]:
        candidate = branch.wave
        if (
            candidate.channels,
            candidate.sample_width,
            candidate.sample_rate,
            candidate.compression_type,
        ) != signature:
            raise RerenderRequired(
                "shared_prefix_audio_mismatch", "sibling WAV contracts differ"
            )
        if candidate.frames[: prefix_samples * width] != prefix:
            raise RerenderRequired(
                "shared_prefix_audio_mismatch",
                "pre-pivot PCM differs; all four siblings must be rerendered from one prefix",
            )
    return prefix, prefix_samples


def target_audio_bytes(branch: NativeBranch) -> tuple[bytes, tuple[int, int]]:
    target = branch.source.get("target")
    if not isinstance(target, Mapping):
        raise RerenderRequired("target_label_contract_invalid", "source lacks target timing")
    start_ms = require_int(target.get("start_ms"), "target.start_ms")
    end_ms = require_int(target.get("end_ms"), "target.end_ms", minimum=start_ms + 1)
    start = round(start_ms * branch.wave.sample_rate / 1000)
    end = round(end_ms * branch.wave.sample_rate / 1000)
    cancellation = branch.actual_events["cancellation"]
    if cancellation["cancelled"]:
        cutoff_seconds = int(cancellation["atFrame"]) / float(
            branch.model_contract["frameRateHz"]
        )
        end = min(end, round(cutoff_seconds * branch.wave.sample_rate))
    if not 0 <= start < end <= branch.wave.frame_count:
        raise RerenderRequired("target_label_contract_invalid", "target audio bounds are invalid")
    width = branch.wave.channels * branch.wave.sample_width
    return branch.wave.frames[start * width : end * width], (start, end)


def validate_group_contract(branches: Sequence[NativeBranch]) -> tuple[int, dict[str, Any], bytes, int]:
    if len(branches) != len(CAUSAL_GROUP_ROLES):
        raise RerenderRequired("group_role_contract_failed", "group must have exactly four siblings")
    roles = [branch.plan["siblingRole"] for branch in branches]
    if set(roles) != set(CAUSAL_GROUP_ROLES) or len(set(roles)) != len(roles):
        raise RerenderRequired(
            "group_role_contract_failed", "group sibling roles are incomplete or duplicated"
        )
    for key in (
        "groupId", "sharedPrefixId", "commonContextHash", "commonContext", "premiseId",
        "templateId", "lineageIdentifiers", "controlOperator", "voicePair",
    ):
        common_plan_value(branches, key)
    require_sha256(branches[0].plan.get("sharedPrefixId"), "sharedPrefixId")
    voice_pair = branches[0].plan.get("voicePair")
    if not isinstance(voice_pair, Mapping) or set(voice_pair) != {"id", "caller", "agent"}:
        raise RerenderRequired("voice_contract_mismatch", "voicePair is not typed")
    if not all(isinstance(voice_pair.get(key), str) and voice_pair[key] for key in voice_pair):
        raise RerenderRequired("voice_contract_mismatch", "voicePair identifiers are invalid")
    if voice_pair["caller"] == voice_pair["agent"]:
        raise RerenderRequired("voice_contract_mismatch", "caller and agent voices must differ")

    pivots = {branch.pivot for branch in branches}
    contracts = {canonical_json(branch.model_contract) for branch in branches}
    control_shapes = {int(branch.control_stream.shape[1]) for branch in branches}
    probe_cardinalities = {
        canonical_json(branch.native_control["probe_slot_cardinalities"])
        for branch in branches
    }
    padding_tokens = {int(branch.native_control["padding_token_id"]) for branch in branches}
    if len(pivots) != 1:
        raise RerenderRequired("native_pivot_mismatch", "siblings have different native pivots")
    if len(contracts) != 1:
        raise RerenderRequired(
            "model_contract_mismatch", "siblings do not share model/Mimi/tokenizer contract"
        )
    if len(control_shapes) != 1 or len(probe_cardinalities) != 1 or len(padding_tokens) != 1:
        raise RerenderRequired(
            "native_control_stream_invalid",
            "siblings disagree on control width, probe cardinalities, or padding token",
        )
    if abs(float(branches[0].model_contract["frameRateHz"]) - 1000.0 / FRAME_DURATION_MS) > 1e-9:
        raise RerenderRequired(
            "model_contract_mismatch", "trainer-ready PersonaPlex codes must use 80 ms frames"
        )
    pivot = next(iter(pivots))
    reference_prefix = branches[0].codes[:, :pivot]
    for branch in branches[1:]:
        candidate = branch.codes[:, :pivot]
        if candidate.shape != reference_prefix.shape or candidate.dtype != reference_prefix.dtype:
            raise RerenderRequired(
                "shared_prefix_codes_mismatch", "native prefix tensor contracts differ"
            )
        if not torch.equal(candidate, reference_prefix):
            raise RerenderRequired(
                "shared_prefix_codes_mismatch",
                "pre-pivot native codes differ; rerender all siblings from one prefix",
            )
    source_voice_pairs = {
        canonical_json((branch.source.get("provenance") or {}).get("voice_pair"))
        for branch in branches
    }
    if len(source_voice_pairs) != 1:
        raise RerenderRequired("voice_contract_mismatch", "source voice-pair provenance differs")
    source_voice_pair = (branches[0].source.get("provenance") or {}).get("voice_pair")
    if not isinstance(source_voice_pair, Mapping):
        raise RerenderRequired("voice_contract_mismatch", "source voice-pair provenance is missing")
    source_caller = source_voice_pair.get("caller")
    source_agent = source_voice_pair.get("agent")
    if not isinstance(source_caller, Mapping) or not isinstance(source_agent, Mapping):
        raise RerenderRequired("voice_contract_mismatch", "source voice references are untyped")
    if source_caller.get("id") != voice_pair["caller"] or source_agent.get("id") != voice_pair["agent"]:
        raise RerenderRequired("voice_contract_mismatch", "plan and source voice IDs differ")
    prefix_audio, prefix_samples = common_audio_prefix(branches, pivot)
    return pivot, branches[0].model_contract, prefix_audio, prefix_samples


def relative_reference(reference: dict[str, Any], root: Path) -> dict[str, Any]:
    output = deepcopy(reference)
    output["path"] = str(Path(output["path"]).relative_to(root))
    return output


def materialize_group(
    plans: Sequence[dict[str, Any]], sources_by_plan: Mapping[str, dict[str, Any]],
    labels_by_id: Mapping[str, dict[str, Any]], native_root: Path, precodec_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    native_branches: list[NativeBranch] = []
    for plan in sorted(plans, key=lambda value: str(value.get("siblingRole"))):
        plan_id = require_sha256(plan.get("planRecordId"), "planRecordId")
        source = sources_by_plan.get(plan_id)
        if source is None:
            raise RerenderRequired(
                "immutable_join_failed", f"no certified native source preserves planRecordId {plan_id}"
            )
        example_id = require_sha256(source.get("example_id"), "example_id")
        label = labels_by_id.get(example_id)
        if label is None:
            raise RerenderRequired(
                "immutable_join_failed", f"no control label exists for example {example_id}"
            )
        native_branches.append(
            load_native_branch(plan, source, label, native_root, precodec_root)
        )
    native_branches.sort(key=lambda branch: CAUSAL_GROUP_ROLES.index(branch.plan["siblingRole"]))
    pivot, contract, prefix_audio, prefix_samples = validate_group_contract(native_branches)
    group_id = native_branches[0].plan["groupId"]
    stem = sha256(group_id.encode("utf-8")).hexdigest()

    common_codes_path = output_root / "artifacts" / "common" / f"{stem}.prefix.codes.pt"
    common_audio_path = output_root / "artifacts" / "common" / f"{stem}.prefix.wav"
    common_codes = save_tensor(
        common_codes_path, "codes", native_branches[0].codes[:, :pivot]
    )
    write_wave(common_audio_path, native_branches[0].wave, prefix_audio)
    common_audio = {
        "path": str(common_audio_path.relative_to(output_root)),
        "sha256": sha256_file(common_audio_path),
        "sampleRateHz": native_branches[0].wave.sample_rate,
        "channels": native_branches[0].wave.channels,
        "sampleWidthBytes": native_branches[0].wave.sample_width,
        "sampleFrames": prefix_samples,
    }
    common_codes = relative_reference(common_codes, output_root)

    sibling_rows: list[dict[str, Any]] = []
    for branch in native_branches:
        example_id = str(branch.source["example_id"])
        example_stem = example_id.removeprefix("sha256:")
        suffix_codes_path = output_root / "artifacts" / "branches" / f"{example_stem}.suffix.codes.pt"
        suffix_mask_path = output_root / "artifacts" / "branches" / f"{example_stem}.suffix.mask.pt"
        control_stream_path = output_root / "artifacts" / "branches" / f"{example_stem}.control.pt"
        target_audio_path = output_root / "artifacts" / "branches" / f"{example_stem}.target.wav"
        cancellation = branch.actual_events["cancellation"]
        suffix_end = int(cancellation["atFrame"]) if cancellation["cancelled"] else branch.target_end
        if not branch.first_supervised < suffix_end <= branch.codes.shape[1]:
            raise RerenderRequired(
                "runtime_event_invalid", "branch cutoff/end does not leave a supervised response"
            )
        suffix_codes = relative_reference(
            save_tensor(suffix_codes_path, "codes", branch.codes[:, pivot:suffix_end]), output_root
        )
        suffix_mask = relative_reference(
            save_tensor(
                suffix_mask_path, "target_mask", branch.target_mask[:, pivot:suffix_end]
            ),
            output_root,
        )
        control_stream = relative_reference(
            save_tensor(control_stream_path, "control_stream", branch.control_stream), output_root
        )
        target_audio, sample_bounds = target_audio_bytes(branch)
        write_wave(target_audio_path, branch.wave, target_audio)
        audio_reference = {
            "path": str(target_audio_path.relative_to(output_root)),
            "sha256": sha256_file(target_audio_path),
            "sampleRateHz": branch.wave.sample_rate,
            "channels": branch.wave.channels,
            "sampleWidthBytes": branch.wave.sample_width,
            "sampleFrames": sample_bounds[1] - sample_bounds[0],
            "sourceSampleRange": list(sample_bounds),
        }
        sibling_rows.append(
            {
                "role": branch.plan["siblingRole"],
                "exampleId": example_id,
                "nativePivotFrame": pivot,
                "alignment": {
                    "memberAtFrame": pivot,
                    "donorAtFrame": pivot,
                    "firstSupervisedFrame": branch.first_supervised,
                    "targetFrameRange": [pivot, branch.target_end],
                    "frameRateHz": contract["frameRateHz"],
                    "prefixInterval": "half_open_[0,nativePivotFrame)",
                },
                "controlInput": {
                    **branch.control_input,
                    "nativeControl": {
                        "stream": control_stream,
                        "schema": branch.native_control["schema"],
                        "acknowledgedControlRevision": branch.native_control[
                            "acknowledged_control_revision"
                        ],
                        "controlActiveFrame": branch.native_control["control_active_frame"],
                        "retrievalBufferFrames": branch.native_control["retrieval_buffer_frames"],
                        "probeFrameIndex": branch.native_control["probe_frame_index"],
                        "probeTargets": branch.native_control["probe_targets"],
                        "probeSlotCardinalities": branch.native_control[
                            "probe_slot_cardinalities"
                        ],
                        "paddingTokenId": branch.native_control["padding_token_id"],
                    },
                },
                "target": {
                    "text": branch.target_text,
                    "textSha256": branch.target_text_hash,
                    "audio": audio_reference,
                    "nativeSuffixCodes": suffix_codes,
                    "nativeSuffixTargetMask": suffix_mask,
                    "nativeFrameOffset": pivot,
                    "nativeFrameRange": [pivot, suffix_end],
                    "actualEvents": branch.actual_events,
                    "sourceArtifacts": {
                        "codesSha256": branch.source["model_encoding"]["codes_sha256"],
                        "targetMaskSha256": branch.source["model_encoding"]["target_mask_sha256"],
                        "alignmentSha256": branch.source["model_encoding"]["text_alignment_sha256"],
                        "duplexAudioSha256": branch.source["duplex"]["sha256"],
                    },
                },
            }
        )

    plan = native_branches[0].plan
    common_context = {
        "sharedPrefixId": plan["sharedPrefixId"],
        "commonContextHash": plan["commonContextHash"],
        "semanticContext": deepcopy(plan["commonContext"]),
        "nativePrefixCodes": common_codes,
        "modelContract": contract,
        "nativePivotFrame": pivot,
        "prefixInterval": "half_open_[0,nativePivotFrame)",
    }
    group = {
        "schema": GROUP_SCHEMA,
        "groupId": group_id,
        "sharedPrefixId": plan["sharedPrefixId"],
        "commonContextHash": plan["commonContextHash"],
        "premiseId": plan["premiseId"],
        "templateId": plan["templateId"],
        "lineageIdentifiers": deepcopy(plan["lineageIdentifiers"]),
        "controlOperator": deepcopy(plan["controlOperator"]),
        "voicePair": deepcopy(plan["voicePair"]),
        "modelContract": contract,
        "nativePivotFrame": pivot,
        "commonInput": {"audio": common_audio, "context": common_context},
        "siblings": sibling_rows,
        "materialization": {
            "toolVersion": TOOL_VERSION,
            "policy": "verified_existing_native_codes_only_no_mimi_recompute",
            "sharedPrefixPolicy": "byte_identical_native_codes_and_pcm_before_pivot",
            "planRecordIds": [branch.plan["planRecordId"] for branch in native_branches],
        },
    }
    try:
        normalize_causal_group(group)
    except CausalGroupPackError as error:
        raise RerenderRequired("causal_group_pack_rejected", str(error)) from error
    return group


def tensor_reference(value: Mapping[str, Any], key: str) -> dict[str, str]:
    return {"path": str(value["path"]), "key": key, "sha256": str(value["sha256"])}


def emit_trainer_ready_dataset(
    groups: Sequence[Mapping[str, Any]], output_root: Path
) -> dict[str, Any]:
    """Deterministically project packer-compatible groups into the native trainer contract.

    The certified manifest includes all three group-disjoint splits.  Test rows are also
    retained in a separate immutable artifact so evaluation can enforce holdout handling.
    """

    if not groups:
        raise RerenderRequired("trainer_projection_empty", "no accepted groups can be projected")
    normalized = [normalize_causal_group(group) for group in groups]
    components = assign_component_splits(
        build_leakage_components(normalized),
        PackConfig(required_coverage_splits=("train", "validation", "test")),
    )
    component_by_group: dict[str, tuple[str, str]] = {}
    for component in components:
        for group_id in component["groupIds"]:
            component_by_group[str(group_id)] = (
                str(component["componentId"]), str(component["split"])
            )
    projected: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    for group, packed in sorted(
        zip(groups, normalized), key=lambda item: str(item[0]["groupId"])
    ):
        group_id = str(group["groupId"])
        component_id, split = component_by_group[group_id]
        prefix = group["commonInput"]["context"]["nativePrefixCodes"]
        pivot = int(group["nativePivotFrame"])
        siblings: list[dict[str, Any]] = []
        for sibling in group["siblings"]:
            control = sibling["controlInput"]
            native = control["nativeControl"]
            target = sibling["target"]
            events = target["actualEvents"]
            cancellation = events["cancellation"]
            suffix = target["nativeSuffixCodes"]
            mask = target["nativeSuffixTargetMask"]
            start, end = target["nativeFrameRange"]
            cutoff = int(cancellation["atFrame"]) if cancellation["cancelled"] else None
            revision = int(control["controlRevision"])
            siblings.append(
                {
                    "sibling_id": str(sibling["exampleId"]),
                    "control_role": str(sibling["role"]),
                    "generation_id": str(cancellation["generationId"]),
                    "control_revision": revision,
                    "acknowledged_control_revision": int(
                        native["acknowledgedControlRevision"]
                    ),
                    "probe_frame_index": int(native["probeFrameIndex"]),
                    "probe_targets": deepcopy(native["probeTargets"]),
                    "native_suffix_codes": tensor_reference(suffix, "codes"),
                    "suffix_agent_target_mask": tensor_reference(mask, "target_mask"),
                    "control_stream": tensor_reference(native["stream"], "control_stream"),
                    "alignment": {
                        "schema": TRAINER_ALIGNMENT_SCHEMA,
                        "alignment_revision": revision,
                        "shared_prefix_sha256": str(prefix["sha256"]),
                        "native_suffix_sha256": str(suffix["sha256"]),
                        "target_mask_sha256": str(mask["sha256"]),
                        "member_at_frame": pivot,
                        "donor_at_frame": pivot,
                        "suffix_start_frame": int(start),
                        "suffix_end_frame": int(end),
                        "control_available_frame": int(control["controlAvailableFrame"]),
                        "control_active_frame": int(native["controlActiveFrame"]),
                        "retrieval_buffer_frames": int(native["retrievalBufferFrames"]),
                        "first_supervised_agent_frame": int(
                            sibling["alignment"]["firstSupervisedFrame"]
                        ),
                        "cutoff_frame": cutoff,
                        "cutoff_revision": revision if cutoff is not None else None,
                        "cutoff_generation_id": (
                            str(cancellation["generationId"]) if cutoff is not None else None
                        ),
                    },
                }
            )
        projected.append(
            {
                "schema": TRAINER_GROUP_SCHEMA,
                "group_id": group_id,
                "leakage_component_id": component_id,
                "split": split,
                "shared_prefix": {
                    "schema": TRAINER_SHARED_PREFIX_SCHEMA,
                    "common_input_id": str(packed["commonInput"]["commonInputId"]),
                    "native_pivot_frame": pivot,
                    "window_start_frame": 0,
                    "window_end_frame": pivot,
                    "native_codes": tensor_reference(prefix, "codes"),
                },
                "siblings": siblings,
            }
        )
        model = group["modelContract"]
        first_native = group["siblings"][0]["controlInput"]["nativeControl"]
        contracts.append(
            {
                "model_revision": model["modelRevision"],
                "num_codebooks": len(
                    model["codebookLayout"]["text"]
                    + model["codebookLayout"]["agent_audio"]
                    + model["codebookLayout"]["caller_audio"]
                ),
                "control_hidden_size": int(first_native["stream"]["shape"][1]),
                "padding_token_id": int(first_native["paddingTokenId"]),
                "stream_layout": {
                    "text_stream_indices": model["codebookLayout"]["text"],
                    "agent_audio_stream_indices": model["codebookLayout"]["agent_audio"],
                    "caller_audio_stream_indices": model["codebookLayout"]["caller_audio"],
                },
                "probe_slot_cardinalities": first_native["probeSlotCardinalities"],
            }
        )
    if len({canonical_json(contract) for contract in contracts}) != 1:
        raise RerenderRequired(
            "trainer_contract_mismatch", "accepted groups do not share one trainer contract"
        )
    split_counts = Counter(row["split"] for row in projected)
    missing = [split for split in ("train", "validation", "test") if not split_counts[split]]
    if missing:
        raise RerenderRequired(
            "trainer_projection_insufficient_splits",
            f"leakage-safe trainer projection lacks splits {missing}",
        )
    test_rows = [row for row in projected if row["split"] == "test"]
    training_path = output_root / TRAINER_MANIFEST_FILENAME
    test_path = output_root / TRAINER_TEST_FILENAME
    all_path = output_root / TRAINER_ALL_SPLITS_FILENAME
    write_jsonl(training_path, projected)
    write_jsonl(test_path, test_rows)
    write_jsonl(all_path, projected)
    base = contracts[0]
    dataset_contract = {
        "schema": TRAINER_DATASET_SCHEMA,
        "status": "certified_for_native_moshirag_full_rank_training",
        "manifest_sha256": sha256_file(training_path),
        "model_revision": base["model_revision"],
        "native_control_schema": NATIVE_MOSHIRAG_CONTROL_SCHEMA,
        "sibling_count": 4,
        "sibling_roles": list(CAUSAL_GROUP_ROLES),
        "frame_duration_ms": FRAME_DURATION_MS,
        "num_codebooks": base["num_codebooks"],
        "control_hidden_size": base["control_hidden_size"],
        "padding_token_id": base["padding_token_id"],
        "stream_layout": base["stream_layout"],
        "probe_slot_cardinalities": base["probe_slot_cardinalities"],
        "split_policy": "group_and_leakage_component_disjoint",
        "packing": "one_shared_native_prefix_plus_branch_native_suffix",
        "provenance": {
            "all_splits_manifest": {
                "path": TRAINER_ALL_SPLITS_FILENAME, "sha256": sha256_file(all_path)
            },
            "test_manifest": {"path": TRAINER_TEST_FILENAME, "sha256": sha256_file(test_path)},
            "test_selection_policy": "never_used_for_checkpoint_selection",
        },
    }
    contract_path = output_root / TRAINER_CONTRACT_FILENAME
    contract_path.write_text(
        json.dumps(dataset_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "contract": {"path": TRAINER_CONTRACT_FILENAME, "sha256": sha256_file(contract_path)},
        "trainerManifest": {
            "path": TRAINER_MANIFEST_FILENAME, "sha256": sha256_file(training_path)
        },
        "testManifest": {"path": TRAINER_TEST_FILENAME, "sha256": sha256_file(test_path)},
        "allSplitsManifest": {
            "path": TRAINER_ALL_SPLITS_FILENAME, "sha256": sha256_file(all_path)
        },
        "splitCounts": dict(sorted(split_counts.items())),
    }


def rejection_record(
    group_id: str | None, plans: Sequence[Mapping[str, Any]], error: RerenderRequired,
    input_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    record = {
        "schema": REJECTION_SCHEMA,
        "status": "rerender_required",
        "groupId": group_id,
        "reasonCode": error.code,
        "detail": error.detail,
        "planRecordIds": sorted(
            value for plan in plans if is_sha256(value := plan.get("planRecordId"))
        ),
        "siblingRoles": sorted(
            str(value) for plan in plans if (value := plan.get("siblingRole")) is not None
        ),
        "inputHashes": deepcopy(dict(input_hashes)),
        "repairPolicy": "rerender_entire_four_sibling_group_from_one_shared_prefix",
    }
    record["rejectionId"] = content_hash(record)
    return record


def build_sources_by_plan(
    precodec_rows: Sequence[dict[str, Any]], native_by_id: Mapping[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for precodec in precodec_rows:
        example_id = precodec.get("example_id")
        if not isinstance(example_id, str) or example_id not in native_by_id:
            raise BridgeError(f"pre-codec example {example_id!r} lacks a native encoded row")
        native = native_by_id[example_id]
        source_projection = {key: value for key, value in native.items() if key != "model_encoding"}
        if source_projection != precodec:
            raise BridgeError(
                f"native example {example_id} is not an immutable extension of pre-codec manifest"
            )
        try:
            plan_id = extract_plan_record_id(native)
        except RerenderRequired as error:
            raise BridgeError(f"native example {example_id}: {error.detail}") from error
        if plan_id in output:
            raise BridgeError(f"multiple native examples preserve planRecordId {plan_id}")
        output[plan_id] = native
    return output


def materialize(
    *, compiled_plan: Path, precodec_manifest: Path, control_labels: Path,
    native_manifest: Path, certificate_path: Path, precodec_root: Path,
    native_root: Path, output_root: Path, overwrite: bool = False,
) -> dict[str, Any]:
    paths = {
        "compiledPlan": compiled_plan.resolve(),
        "precodecManifest": precodec_manifest.resolve(),
        "controlLabels": control_labels.resolve(),
        "nativeManifest": native_manifest.resolve(),
        "certificate": certificate_path.resolve(),
    }
    input_hashes = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    plans = read_jsonl(paths["compiledPlan"], "compiled v5 plan")
    precodec_rows = read_jsonl(paths["precodecManifest"], "pre-codec manifest")
    label_rows = read_jsonl(paths["controlLabels"], "control labels")
    native_rows = read_jsonl(paths["nativeManifest"], "native manifest")
    certificate = read_json(paths["certificate"])
    validate_certificate(certificate, paths["nativeManifest"], native_root.resolve(), len(native_rows))
    native_by_id = index_unique(native_rows, "example_id", "native manifest")
    labels_by_id = index_unique(label_rows, "example_id", "control labels")
    if set(labels_by_id) != set(native_by_id):
        raise BridgeError("control-label and native example ID sets differ")
    sources_by_plan = build_sources_by_plan(precodec_rows, native_by_id)

    plans_by_group: dict[str, list[dict[str, Any]]] = {}
    invalid_plan_rejections: list[dict[str, Any]] = []
    seen_plan_ids: set[str] = set()
    for plan in plans:
        group_hint = plan.get("groupId") if isinstance(plan.get("groupId"), str) else None
        try:
            group_id, _role = validate_plan_record(plan)
            plan_id = str(plan["planRecordId"])
            if plan_id in seen_plan_ids:
                raise RerenderRequired("immutable_identity_invalid", "duplicate planRecordId")
            seen_plan_ids.add(plan_id)
            plans_by_group.setdefault(group_id, []).append(plan)
        except RerenderRequired as error:
            invalid_plan_rejections.append(
                rejection_record(group_hint, [plan], error, input_hashes)
            )

    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise BridgeError(f"refusing non-empty output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent))
    accepted: list[dict[str, Any]] = []
    rejections = list(invalid_plan_rejections)
    try:
        for group_id, group_plans in sorted(plans_by_group.items()):
            roles = [plan.get("siblingRole") for plan in group_plans]
            if len(group_plans) != 4 or set(roles) != set(CAUSAL_GROUP_ROLES):
                error = RerenderRequired(
                    "group_role_contract_failed",
                    "compiled group must contain exactly one of each four v5 sibling roles",
                )
                rejections.append(rejection_record(group_id, group_plans, error, input_hashes))
                continue
            try:
                accepted.append(
                    materialize_group(
                        group_plans, sources_by_plan, labels_by_id,
                        native_root.resolve(), precodec_root.resolve(), stage,
                    )
                )
            except RerenderRequired as error:
                rejections.append(rejection_record(group_id, group_plans, error, input_hashes))

        accepted.sort(key=lambda value: str(value["groupId"]))
        trainer_outputs: dict[str, Any] | None = None
        if accepted:
            try:
                trainer_outputs = emit_trainer_ready_dataset(accepted, stage)
            except RerenderRequired as error:
                rejections.append(rejection_record(None, plans, error, input_hashes))
        rejections.sort(key=lambda value: (str(value.get("groupId")), value["rejectionId"]))
        groups_path = stage / GROUPS_FILENAME
        rejections_path = stage / REJECTIONS_FILENAME
        write_jsonl(groups_path, accepted)
        write_jsonl(rejections_path, rejections)
        reason_counts = Counter(str(item["reasonCode"]) for item in rejections)
        status = (
            "certified"
            if accepted and not rejections
            else "partial_rerender_required"
            if accepted
            else "rerender_required"
        )
        report: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "toolVersion": TOOL_VERSION,
            "status": status,
            "inputs": input_hashes,
            "sourceCertificateSha256": input_hashes["certificate"]["sha256"],
            "counts": {
                "compiledPlanRecords": len(plans),
                "acceptedGroups": len(accepted),
                "acceptedSiblings": len(accepted) * len(CAUSAL_GROUP_ROLES),
                "rejectedGroups": len(rejections),
            },
            "rejectionReasons": dict(sorted(reason_counts.items())),
            "outputs": {
                "groups": {"path": GROUPS_FILENAME, "sha256": sha256_file(groups_path)},
                "rejections": {
                    "path": REJECTIONS_FILENAME,
                    "sha256": sha256_file(rejections_path),
                },
                "nativeMoshiRag": trainer_outputs,
            },
            "invariants": {
                "mimiRecomputed": False,
                "gpuWorkPerformed": False,
                "misalignedSidecarsSpliced": False,
                "callerTargetMaskAllowed": False,
                "controlAvailableStrictlyBeforeSupervision": True,
                "sharedPrefixStoredOncePerGroup": True,
            },
        }
        report["reportId"] = content_hash(report)
        (stage / REPORT_FILENAME).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(stage, output_root)
        return report
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--compiled-plan", required=True, type=Path)
    value.add_argument("--precodec-manifest", required=True, type=Path)
    value.add_argument("--control-labels", required=True, type=Path)
    value.add_argument("--native-manifest", required=True, type=Path)
    value.add_argument("--certificate", required=True, type=Path)
    value.add_argument("--precodec-root", required=True, type=Path)
    value.add_argument("--native-root", required=True, type=Path)
    value.add_argument("--output-root", required=True, type=Path)
    value.add_argument("--overwrite", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = materialize(
            compiled_plan=args.compiled_plan,
            precodec_manifest=args.precodec_manifest,
            control_labels=args.control_labels,
            native_manifest=args.native_manifest,
            certificate_path=args.certificate,
            precodec_root=args.precodec_root,
            native_root=args.native_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
    except (BridgeError, OSError) as error:
        print(canonical_json({"status": "rejected", "error": str(error)}), file=__import__("sys").stderr)
        return 2
    print(canonical_json({"status": report["status"], **report["counts"]}))
    return 0 if report["status"] == "certified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
