#!/usr/bin/env python3
"""Full-rank PersonaPlex receiver training with native MoshiRAG control streams.

The input is deliberately group-native rather than a loose example manifest. Each
JSONL row is one group-disjoint four-sibling causal intervention. Native delayed
duplex codes, agent-only masks, ARC control streams, availability/cancellation
timing, and typed probe labels are mandatory and content-addressed. No transcript
or target text is accepted as a control input.

Run directly to perform ratio-based GPU admission and launch torchrun, or invoke
the script under torchrun with CUDA_VISIBLE_DEVICES restricted to physical GPUs
0, 1, and 2. The 7B receiver is checkpointed with distributed checkpoint shards;
it is never gathered onto host RAM.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from ground_truth_finetuning.training.contracts import StreamLayout
from ground_truth_finetuning.training.causal_group_pack import (
    CERTIFICATE_SCHEMA,
    MANIFEST_SCHEMA,
    PACK_SCHEMA,
    TRAINER_BINDING_SCHEMA,
    content_hash,
)
from ground_truth_finetuning.training.fsdp_receiver import (
    FSDPReceiverBundle,
    clip_sharded_grad_norm,
    load_receiver_checkpoint,
    save_receiver_checkpoint,
    shard_full_rank_temporal_text_receiver,
)
from ground_truth_finetuning.training.gpu_admission import admit_gpus_by_ratio
from ground_truth_finetuning.training.native_moshirag_control import (
    NATIVE_MOSHIRAG_CONTROL_SCHEMA,
    PreResponseControlStateProbe,
    StreamingConditionSnapshot,
    listwise_causal_loss,
    select_full_rank_temporal_text_parameters,
    strict_listwise_group_pass,
)
from ground_truth_finetuning.training.native_source import require_moshi_source_contract
from ground_truth_finetuning.training.native_training import (
    agent_only_loss_per_example,
    forward_with_native_streaming_sum,
)


DATASET_SCHEMA = "personaplex.native-moshirag-dataset.v2-shared-prefix"
GROUP_SCHEMA = "personaplex.native-moshirag-group.v2-shared-prefix"
SHARED_PREFIX_SCHEMA = "personaplex.native-shared-prefix.v1"
BRANCH_ALIGNMENT_SCHEMA = "personaplex.native-branch-window-alignment.v1"
CHECKPOINT_SCHEMA = "personaplex.native-moshirag-full-rank-checkpoint.v1"
CHECKPOINT_GATES = (100, 125, 150)
FRAME_DURATION_MS = 80
SIBLING_COUNT = 4
PHYSICAL_GPU_CEILING = frozenset({0, 1, 2})
MAX_HOST_MEMORY_RATIO = 0.80
SPLITS = frozenset({"train", "validation", "test"})
PACK_OUTPUT_FILENAMES = frozenset(
    {
        "common_inputs.jsonl",
        "listwise_groups.jsonl",
        "pairwise_diagnostics.jsonl",
        "leakage_components.jsonl",
        "causal_coverage_certificate.json",
    }
)


class TrainerContractError(ValueError):
    """A source artifact or runtime violates the native training contract."""


def _required_keys(
    value: Mapping[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    context: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise TrainerContractError(f"{context} is missing fields: {missing}")
    if unknown:
        raise TrainerContractError(f"{context} has unsupported fields: {unknown}")


def _nonempty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TrainerContractError(f"{context} must be a non-empty string")
    return value


def _sha256_uri(value: Any, context: str) -> str:
    text = _nonempty_string(value, context)
    prefix = "sha256:"
    digest = text[len(prefix) :] if text.startswith(prefix) else ""
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise TrainerContractError(f"{context} must be a lowercase sha256 URI")
    return text


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class TensorReference:
    path: str
    key: str
    sha256: str

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> "TensorReference":
        if not isinstance(value, Mapping):
            raise TrainerContractError(f"{context} must be an object")
        _required_keys(
            value,
            required=("path", "key", "sha256"),
            context=context,
        )
        path = _nonempty_string(value["path"], f"{context}.path")
        if Path(path).is_absolute():
            raise TrainerContractError(f"{context}.path must be relative to data-root")
        return cls(
            path=path,
            key=_nonempty_string(value["key"], f"{context}.key"),
            sha256=_sha256_uri(value["sha256"], f"{context}.sha256"),
        )


@dataclass(frozen=True)
class SharedPrefixSpec:
    common_input_id: str
    native_pivot_frame: int
    window_start_frame: int
    window_end_frame: int
    native_codes: TensorReference

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> "SharedPrefixSpec":
        if not isinstance(value, Mapping):
            raise TrainerContractError(f"{context} must be an object")
        _required_keys(
            value,
            required=(
                "schema",
                "common_input_id",
                "native_pivot_frame",
                "window_start_frame",
                "window_end_frame",
                "native_codes",
            ),
            context=context,
        )
        if value["schema"] != SHARED_PREFIX_SCHEMA:
            raise TrainerContractError(f"{context} has unsupported schema")
        frames: dict[str, int] = {}
        for name in ("native_pivot_frame", "window_start_frame", "window_end_frame"):
            item = value[name]
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise TrainerContractError(f"{context}.{name} must be a non-negative integer")
            frames[name] = item
        if frames["window_start_frame"] != 0:
            raise TrainerContractError(f"{context} must begin at native frame zero")
        if frames["window_end_frame"] != frames["native_pivot_frame"]:
            raise TrainerContractError(f"{context} window end must equal its native pivot")
        return cls(
            common_input_id=_nonempty_string(
                value["common_input_id"], f"{context}.common_input_id"
            ),
            native_pivot_frame=frames["native_pivot_frame"],
            window_start_frame=frames["window_start_frame"],
            window_end_frame=frames["window_end_frame"],
            native_codes=TensorReference.from_mapping(
                value["native_codes"], f"{context}.native_codes"
            ),
        )


@dataclass(frozen=True)
class BranchWindowAlignment:
    alignment_revision: int
    shared_prefix_sha256: str
    native_suffix_sha256: str
    target_mask_sha256: str
    member_at_frame: int
    donor_at_frame: int
    suffix_start_frame: int
    suffix_end_frame: int
    control_available_frame: int
    control_active_frame: int
    retrieval_buffer_frames: int
    first_supervised_agent_frame: int
    cutoff_frame: int | None
    cutoff_revision: int | None
    cutoff_generation_id: str | None

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> "BranchWindowAlignment":
        if not isinstance(value, Mapping):
            raise TrainerContractError(f"{context} must be an object")
        _required_keys(
            value,
            required=(
                "schema",
                "alignment_revision",
                "shared_prefix_sha256",
                "native_suffix_sha256",
                "target_mask_sha256",
                "member_at_frame",
                "donor_at_frame",
                "suffix_start_frame",
                "suffix_end_frame",
                "control_available_frame",
                "control_active_frame",
                "retrieval_buffer_frames",
                "first_supervised_agent_frame",
                "cutoff_frame",
                "cutoff_revision",
                "cutoff_generation_id",
            ),
            context=context,
        )
        if value["schema"] != BRANCH_ALIGNMENT_SCHEMA:
            raise TrainerContractError(f"{context} has unsupported schema")
        integers: dict[str, int] = {}
        for name in (
            "alignment_revision",
            "member_at_frame",
            "donor_at_frame",
            "suffix_start_frame",
            "suffix_end_frame",
            "control_available_frame",
            "control_active_frame",
            "retrieval_buffer_frames",
            "first_supervised_agent_frame",
        ):
            item = value[name]
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise TrainerContractError(f"{context}.{name} must be a non-negative integer")
            integers[name] = item
        cutoff = value["cutoff_frame"]
        cutoff_revision = value["cutoff_revision"]
        cutoff_generation_id = value["cutoff_generation_id"]
        if cutoff is None:
            if cutoff_revision is not None or cutoff_generation_id is not None:
                raise TrainerContractError(f"{context} has stale cutoff metadata without a cutoff")
        else:
            if not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff < 0:
                raise TrainerContractError(f"{context}.cutoff_frame must be non-negative")
            if (
                not isinstance(cutoff_revision, int)
                or isinstance(cutoff_revision, bool)
                or cutoff_revision < 0
            ):
                raise TrainerContractError(f"{context}.cutoff_revision is invalid")
            cutoff_generation_id = _nonempty_string(
                cutoff_generation_id, f"{context}.cutoff_generation_id"
            )
        return cls(
            alignment_revision=integers["alignment_revision"],
            shared_prefix_sha256=_sha256_uri(
                value["shared_prefix_sha256"], f"{context}.shared_prefix_sha256"
            ),
            native_suffix_sha256=_sha256_uri(
                value["native_suffix_sha256"], f"{context}.native_suffix_sha256"
            ),
            target_mask_sha256=_sha256_uri(
                value["target_mask_sha256"], f"{context}.target_mask_sha256"
            ),
            member_at_frame=integers["member_at_frame"],
            donor_at_frame=integers["donor_at_frame"],
            suffix_start_frame=integers["suffix_start_frame"],
            suffix_end_frame=integers["suffix_end_frame"],
            control_available_frame=integers["control_available_frame"],
            control_active_frame=integers["control_active_frame"],
            retrieval_buffer_frames=integers["retrieval_buffer_frames"],
            first_supervised_agent_frame=integers["first_supervised_agent_frame"],
            cutoff_frame=cutoff,
            cutoff_revision=cutoff_revision,
            cutoff_generation_id=cutoff_generation_id,
        )


@dataclass(frozen=True)
class SiblingSpec:
    sibling_id: str
    control_role: str
    generation_id: str
    control_revision: int
    acknowledged_control_revision: int
    probe_frame_index: int
    probe_targets: dict[str, int]
    native_suffix_codes: TensorReference
    suffix_agent_target_mask: TensorReference
    control_stream: TensorReference
    alignment: BranchWindowAlignment

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> "SiblingSpec":
        if not isinstance(value, Mapping):
            raise TrainerContractError(f"{context} must be an object")
        _required_keys(
            value,
            required=(
                "sibling_id",
                "control_role",
                "generation_id",
                "control_revision",
                "acknowledged_control_revision",
                "probe_frame_index",
                "probe_targets",
                "native_suffix_codes",
                "suffix_agent_target_mask",
                "control_stream",
                "alignment",
            ),
            context=context,
        )
        integer_fields = (
            "control_revision",
            "acknowledged_control_revision",
            "probe_frame_index",
        )
        parsed_integers: dict[str, int] = {}
        for name in integer_fields:
            item = value[name]
            if not isinstance(item, int) or isinstance(item, bool) or item < 0:
                raise TrainerContractError(f"{context}.{name} must be a non-negative integer")
            parsed_integers[name] = item
        raw_targets = value["probe_targets"]
        if not isinstance(raw_targets, Mapping) or not raw_targets:
            raise TrainerContractError(f"{context}.probe_targets must be a non-empty object")
        probe_targets: dict[str, int] = {}
        for name, target in raw_targets.items():
            slot = _nonempty_string(name, f"{context}.probe_targets key")
            if not isinstance(target, int) or isinstance(target, bool) or target < 0:
                raise TrainerContractError(
                    f"{context}.probe_targets.{slot} must be a non-negative integer"
                )
            probe_targets[slot] = target
        return cls(
            sibling_id=_nonempty_string(value["sibling_id"], f"{context}.sibling_id"),
            control_role=_nonempty_string(value["control_role"], f"{context}.control_role"),
            generation_id=_nonempty_string(value["generation_id"], f"{context}.generation_id"),
            control_revision=parsed_integers["control_revision"],
            acknowledged_control_revision=parsed_integers[
                "acknowledged_control_revision"
            ],
            probe_frame_index=parsed_integers["probe_frame_index"],
            probe_targets=probe_targets,
            native_suffix_codes=TensorReference.from_mapping(
                value["native_suffix_codes"], f"{context}.native_suffix_codes"
            ),
            suffix_agent_target_mask=TensorReference.from_mapping(
                value["suffix_agent_target_mask"], f"{context}.suffix_agent_target_mask"
            ),
            control_stream=TensorReference.from_mapping(
                value["control_stream"], f"{context}.control_stream"
            ),
            alignment=BranchWindowAlignment.from_mapping(
                value["alignment"], f"{context}.alignment"
            ),
        )

    def snapshot(self) -> StreamingConditionSnapshot:
        return StreamingConditionSnapshot(
            generation_id=self.generation_id,
            control_revision=self.control_revision,
            acknowledged_revision=self.acknowledged_control_revision,
            available_at_frame=self.alignment.control_available_frame,
            active_from_frame=self.alignment.control_active_frame,
            retrieval_buffer_frames=self.alignment.retrieval_buffer_frames,
            cancel_at_frame=self.alignment.cutoff_frame,
        )


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    leakage_component_id: str
    split: str
    shared_prefix: SharedPrefixSpec
    siblings: tuple[SiblingSpec, ...]


@dataclass(frozen=True)
class NativeDatasetContract:
    manifest_sha256: str
    model_revision: str
    sibling_roles: tuple[str, ...]
    num_codebooks: int
    control_hidden_size: int
    padding_token_id: int
    stream_layout: StreamLayout
    probe_slot_cardinalities: dict[str, int]

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        manifest_sha256: str,
        model_revision: str,
    ) -> "NativeDatasetContract":
        if not isinstance(value, Mapping):
            raise TrainerContractError("dataset contract must be an object")
        _required_keys(
            value,
            required=(
                "schema",
                "status",
                "manifest_sha256",
                "model_revision",
                "native_control_schema",
                "sibling_count",
                "sibling_roles",
                "frame_duration_ms",
                "num_codebooks",
                "control_hidden_size",
                "padding_token_id",
                "stream_layout",
                "probe_slot_cardinalities",
                "split_policy",
                "packing",
            ),
            optional=("created_at", "provenance"),
            context="dataset contract",
        )
        expected = {
            "schema": DATASET_SCHEMA,
            "status": "certified_for_native_moshirag_full_rank_training",
            "manifest_sha256": manifest_sha256,
            "model_revision": model_revision,
            "native_control_schema": NATIVE_MOSHIRAG_CONTROL_SCHEMA,
            "sibling_count": SIBLING_COUNT,
            "frame_duration_ms": FRAME_DURATION_MS,
            "split_policy": "group_and_leakage_component_disjoint",
            "packing": "one_shared_native_prefix_plus_branch_native_suffix",
        }
        mismatches = {
            key: (expected_value, value.get(key))
            for key, expected_value in expected.items()
            if value.get(key) != expected_value
        }
        if mismatches:
            raise TrainerContractError(f"dataset contract mismatch: {mismatches}")
        roles = value["sibling_roles"]
        if (
            not isinstance(roles, list)
            or len(roles) != SIBLING_COUNT
            or not all(isinstance(role, str) and role for role in roles)
            or len(set(roles)) != SIBLING_COUNT
        ):
            raise TrainerContractError("dataset contract must define four unique sibling roles")
        num_codebooks = value["num_codebooks"]
        hidden = value["control_hidden_size"]
        padding_token_id = value["padding_token_id"]
        if not isinstance(num_codebooks, int) or isinstance(num_codebooks, bool) or num_codebooks < 2:
            raise TrainerContractError("num_codebooks must be an integer of at least two")
        if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden < 1:
            raise TrainerContractError("control_hidden_size must be positive")
        if (
            not isinstance(padding_token_id, int)
            or isinstance(padding_token_id, bool)
            or padding_token_id < 0
        ):
            raise TrainerContractError("padding_token_id must be a non-negative integer")
        raw_slots = value["probe_slot_cardinalities"]
        if not isinstance(raw_slots, Mapping) or not raw_slots:
            raise TrainerContractError("probe_slot_cardinalities must be non-empty")
        slots: dict[str, int] = {}
        for name, cardinality in raw_slots.items():
            slot = _nonempty_string(name, "probe slot name")
            if (
                not isinstance(cardinality, int)
                or isinstance(cardinality, bool)
                or cardinality < 2
            ):
                raise TrainerContractError(f"probe slot {slot!r} must have at least two values")
            slots[slot] = cardinality
        if not isinstance(value["stream_layout"], Mapping):
            raise TrainerContractError("stream_layout must be an object")
        return cls(
            manifest_sha256=manifest_sha256,
            model_revision=model_revision,
            sibling_roles=tuple(roles),
            num_codebooks=num_codebooks,
            control_hidden_size=hidden,
            padding_token_id=padding_token_id,
            stream_layout=StreamLayout.from_mapping(value["stream_layout"]),
            probe_slot_cardinalities=slots,
        )


@dataclass(frozen=True)
class CertifiedPackProof:
    manifest_id: str
    manifest_sha256: str
    coverage_certificate_sha256: str
    split_assignment_hash: str
    source_group_inputs_hash: str
    dataset_contract_sha256: str
    group_manifest_sha256: str
    model_contract_sha256: str
    model_revision: str

    def to_mapping(self) -> dict[str, str]:
        return {
            "manifestId": self.manifest_id,
            "manifestSha256": self.manifest_sha256,
            "coverageCertificateSha256": self.coverage_certificate_sha256,
            "splitAssignmentHash": self.split_assignment_hash,
            "sourceGroupInputsHash": self.source_group_inputs_hash,
            "datasetContractSha256": self.dataset_contract_sha256,
            "groupManifestSha256": self.group_manifest_sha256,
            "modelContractSha256": self.model_contract_sha256,
            "modelRevision": self.model_revision,
        }


def _parse_group(value: Any, contract: NativeDatasetContract, line_number: int) -> GroupSpec:
    context = f"manifest line {line_number}"
    if not isinstance(value, Mapping):
        raise TrainerContractError(f"{context} must be an object")
    _required_keys(
        value,
        required=(
            "schema",
            "group_id",
            "leakage_component_id",
            "split",
            "shared_prefix",
            "siblings",
        ),
        context=context,
    )
    if value["schema"] != GROUP_SCHEMA:
        raise TrainerContractError(f"{context} has unsupported schema")
    split = value["split"]
    if split not in SPLITS:
        raise TrainerContractError(f"{context}.split must be train, validation, or test")
    raw_siblings = value["siblings"]
    if not isinstance(raw_siblings, list) or len(raw_siblings) != SIBLING_COUNT:
        raise TrainerContractError(f"{context} must contain exactly four siblings")
    siblings = tuple(
        SiblingSpec.from_mapping(item, f"{context}.siblings[{index}]")
        for index, item in enumerate(raw_siblings)
    )
    ids = [sibling.sibling_id for sibling in siblings]
    roles = [sibling.control_role for sibling in siblings]
    if len(set(ids)) != SIBLING_COUNT:
        raise TrainerContractError(f"{context} contains duplicate sibling IDs")
    if set(roles) != set(contract.sibling_roles):
        raise TrainerContractError(
            f"{context} sibling roles do not match the certified dataset contract"
        )
    return GroupSpec(
        group_id=_nonempty_string(value["group_id"], f"{context}.group_id"),
        leakage_component_id=_nonempty_string(
            value["leakage_component_id"], f"{context}.leakage_component_id"
        ),
        split=split,
        shared_prefix=SharedPrefixSpec.from_mapping(
            value["shared_prefix"], f"{context}.shared_prefix"
        ),
        siblings=siblings,
    )


def _resolve_tensor_path(data_root: Path, reference: TensorReference) -> Path:
    root = data_root.resolve()
    path = (root / reference.path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TrainerContractError(f"tensor path escapes data-root: {reference.path}") from exc
    if not path.is_file():
        raise TrainerContractError(f"required native tensor is absent: {path}")
    return path


def load_group_manifest(
    manifest_path: Path,
    *,
    data_root: Path,
    contract: NativeDatasetContract,
) -> tuple[GroupSpec, ...]:
    groups: list[GroupSpec] = []
    seen_groups: set[str] = set()
    seen_siblings: set[str] = set()
    component_splits: dict[str, str] = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainerContractError(
                    f"manifest line {line_number} is not valid JSON"
                ) from exc
            group = _parse_group(raw, contract, line_number)
            if group.group_id in seen_groups:
                raise TrainerContractError(f"duplicate causal group ID: {group.group_id}")
            seen_groups.add(group.group_id)
            prior_split = component_splits.setdefault(group.leakage_component_id, group.split)
            if prior_split != group.split:
                raise TrainerContractError(
                    f"leakage component {group.leakage_component_id!r} crosses dataset splits"
                )
            _resolve_tensor_path(data_root, group.shared_prefix.native_codes)
            for sibling in group.siblings:
                if sibling.sibling_id in seen_siblings:
                    raise TrainerContractError(f"duplicate sibling ID: {sibling.sibling_id}")
                seen_siblings.add(sibling.sibling_id)
                for reference in (
                    sibling.native_suffix_codes,
                    sibling.suffix_agent_target_mask,
                    sibling.control_stream,
                ):
                    _resolve_tensor_path(data_root, reference)
            groups.append(group)
    if not groups:
        raise TrainerContractError("native group manifest is empty")
    counts = {split: sum(group.split == split for group in groups) for split in SPLITS}
    if any(not counts[split] for split in SPLITS):
        raise TrainerContractError("manifest requires non-empty train, validation, and test groups")
    return tuple(groups)


def _artifact_descriptor(
    value: Any,
    context: str,
    *,
    group_records: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TrainerContractError(f"{context} must be an object")
    required = ("path", "sha256", "sizeBytes") + (("groupRecords",) if group_records else ())
    _required_keys(value, required=required, context=context)
    size = value["sizeBytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise TrainerContractError(f"{context}.sizeBytes must be a non-negative integer")
    descriptor = {
        "path": _nonempty_string(value["path"], f"{context}.path"),
        "sha256": _sha256_uri(value["sha256"], f"{context}.sha256"),
        "sizeBytes": size,
    }
    if group_records:
        records = value["groupRecords"]
        if not isinstance(records, int) or isinstance(records, bool) or records < 1:
            raise TrainerContractError(f"{context}.groupRecords must be positive")
        descriptor["groupRecords"] = records
    return descriptor


def _verify_descriptor_file(
    descriptor: Mapping[str, Any], path: Path, context: str
) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise TrainerContractError(f"{context} is absent: {resolved}")
    size = resolved.stat().st_size
    if size != descriptor["sizeBytes"]:
        raise TrainerContractError(
            f"{context} size mismatch: expected {descriptor['sizeBytes']}, got {size}"
        )
    digest = hash_file(resolved)
    if digest != descriptor["sha256"]:
        raise TrainerContractError(
            f"{context} hash mismatch: expected {descriptor['sha256']}, got {digest}"
        )


def _verify_bound_argument(
    descriptor: Mapping[str, Any], actual_path: Path, context: str
) -> None:
    expected_path = Path(str(descriptor["path"])).expanduser().resolve()
    resolved = actual_path.expanduser().resolve()
    if resolved != expected_path:
        raise TrainerContractError(
            f"{context} path mismatch: pack binds {expected_path}, trainer received {resolved}"
        )
    _verify_descriptor_file(descriptor, resolved, context)


def _load_pack_assignments(
    path: Path, contract: NativeDatasetContract
) -> list[dict[str, str]]:
    assignments: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainerContractError(
                    f"packed listwise index line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise TrainerContractError(
                    f"packed listwise index line {line_number} must be an object"
                )
            _required_keys(
                value,
                required=(
                    "schema",
                    "groupId",
                    "split",
                    "componentId",
                    "premiseId",
                    "lineageIdentifiers",
                    "templateId",
                    "controlOperator",
                    "voicePair",
                    "commonInputRef",
                    "siblings",
                ),
                context=f"packed listwise index line {line_number}",
            )
            if value["schema"] != f"{PACK_SCHEMA}.listwise-index":
                raise TrainerContractError(
                    f"packed listwise index line {line_number} has unsupported schema"
                )
            group_id = _nonempty_string(
                value["groupId"], f"packed listwise index line {line_number}.groupId"
            )
            if group_id in seen:
                raise TrainerContractError(f"packed listwise index duplicates group {group_id}")
            seen.add(group_id)
            split = value["split"]
            if split not in SPLITS:
                raise TrainerContractError(f"packed group {group_id} has unsupported split")
            component_id = _nonempty_string(
                value["componentId"], f"packed group {group_id}.componentId"
            )
            siblings = value["siblings"]
            roles = (
                [item.get("role") for item in siblings]
                if isinstance(siblings, list) and all(isinstance(item, Mapping) for item in siblings)
                else []
            )
            if roles != list(contract.sibling_roles):
                raise TrainerContractError(
                    f"packed group {group_id} sibling roles differ from the dataset contract"
                )
            assignments.append(
                {"groupId": group_id, "componentId": component_id, "split": str(split)}
            )
    if not assignments:
        raise TrainerContractError("packed listwise index is empty")
    return sorted(assignments, key=lambda item: item["groupId"])


def _load_trainer_assignments(
    path: Path, contract: NativeDatasetContract
) -> list[dict[str, str]]:
    assignments: list[dict[str, str]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainerContractError(
                    f"trainer group manifest line {line_number} is not valid JSON"
                ) from exc
            group = _parse_group(value, contract, line_number)
            if group.group_id in seen:
                raise TrainerContractError(f"duplicate causal group ID: {group.group_id}")
            seen.add(group.group_id)
            assignments.append(
                {
                    "groupId": group.group_id,
                    "componentId": group.leakage_component_id,
                    "split": group.split,
                }
            )
    if not assignments:
        raise TrainerContractError("native group manifest is empty")
    return sorted(assignments, key=lambda item: item["groupId"])


def _load_component_splits(
    path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    component_splits: dict[str, str] = {}
    group_components: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainerContractError(
                    f"leakage component line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, Mapping):
                raise TrainerContractError(f"leakage component line {line_number} must be an object")
            _required_keys(
                value,
                required=("componentId", "groupIds", "groupCount", "leakageKeys", "split"),
                context=f"leakage component line {line_number}",
            )
            component_id = _nonempty_string(
                value["componentId"], f"leakage component line {line_number}.componentId"
            )
            if component_id in component_splits:
                raise TrainerContractError(f"duplicate leakage component {component_id}")
            split = value["split"]
            if split not in SPLITS:
                raise TrainerContractError(f"leakage component {component_id} has unsupported split")
            group_ids = value["groupIds"]
            if (
                not isinstance(group_ids, list)
                or not group_ids
                or not all(isinstance(item, str) and item for item in group_ids)
                or len(group_ids) != len(set(group_ids))
                or value["groupCount"] != len(group_ids)
            ):
                raise TrainerContractError(f"leakage component {component_id} has invalid members")
            component_splits[component_id] = str(split)
            for group_id in group_ids:
                if group_id in group_components:
                    raise TrainerContractError(f"packed group {group_id} belongs to multiple components")
                group_components[group_id] = component_id
    if not component_splits:
        raise TrainerContractError("leakage component index is empty")
    return component_splits, group_components


def verify_certified_pack(
    pack_manifest_path: Path,
    *,
    data_contract_path: Path,
    group_manifest_path: Path,
    model_contract_path: Path,
) -> CertifiedPackProof:
    """Verify the complete certified-pack hash chain before distributed launch."""

    manifest_path = pack_manifest_path.expanduser().resolve()
    manifest = _load_json(manifest_path, "certified pack manifest")
    _required_keys(
        manifest,
        required=(
            "schema",
            "status",
            "inputs",
            "configuration",
            "counts",
            "outputs",
            "trainerBinding",
            "manifestId",
        ),
        context="certified pack manifest",
    )
    if manifest["schema"] != MANIFEST_SCHEMA:
        raise TrainerContractError("certified pack manifest has unsupported schema")
    if manifest["status"] != "certified":
        raise TrainerContractError("causal coverage pack is not certified")
    manifest_id = _sha256_uri(manifest["manifestId"], "certified pack manifest.manifestId")
    manifest_base = dict(manifest)
    del manifest_base["manifestId"]
    if content_hash(manifest_base) != manifest_id:
        raise TrainerContractError("certified pack manifestId does not match its content")

    raw_inputs = manifest["inputs"]
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise TrainerContractError("certified pack manifest inputs must be non-empty")
    source_inputs: list[dict[str, Any]] = []
    source_paths: set[Path] = set()
    for index, value in enumerate(raw_inputs):
        descriptor = _artifact_descriptor(
            value, f"certified pack manifest.inputs[{index}]", group_records=True
        )
        path = Path(descriptor["path"]).expanduser().resolve()
        if not Path(descriptor["path"]).is_absolute():
            raise TrainerContractError("certified pack source input paths must be absolute")
        if path in source_paths:
            raise TrainerContractError(f"certified pack duplicates source input {path}")
        source_paths.add(path)
        _verify_descriptor_file(descriptor, path, f"certified pack source input {path}")
        source_inputs.append(descriptor)
    source_inputs.sort(key=lambda item: item["path"])
    if source_inputs != raw_inputs:
        raise TrainerContractError("certified pack source inputs are not canonically ordered")

    raw_outputs = manifest["outputs"]
    if not isinstance(raw_outputs, list):
        raise TrainerContractError("certified pack outputs must be an array")
    outputs: dict[str, dict[str, Any]] = {}
    pack_root = manifest_path.parent
    for index, value in enumerate(raw_outputs):
        descriptor = _artifact_descriptor(value, f"certified pack outputs[{index}]")
        name = descriptor["path"]
        if name in outputs:
            raise TrainerContractError(f"certified pack duplicates output {name}")
        if Path(name).is_absolute() or Path(name).name != name:
            raise TrainerContractError(f"certified pack output path is not local: {name}")
        output_path = (pack_root / name).resolve()
        try:
            output_path.relative_to(pack_root.resolve())
        except ValueError as exc:
            raise TrainerContractError(f"certified pack output escapes pack root: {name}") from exc
        _verify_descriptor_file(descriptor, output_path, f"certified pack output {name}")
        outputs[name] = descriptor
    if set(outputs) != PACK_OUTPUT_FILENAMES:
        raise TrainerContractError(
            f"certified pack outputs differ from the required set: {sorted(set(outputs) ^ PACK_OUTPUT_FILENAMES)}"
        )

    binding = manifest["trainerBinding"]
    if not isinstance(binding, Mapping):
        raise TrainerContractError("certified pack trainerBinding must be an object")
    _required_keys(
        binding,
        required=(
            "schema",
            "sourceGroupInputsHash",
            "datasetContract",
            "groupManifest",
            "modelContract",
            "modelRevision",
            "splitAssignmentHash",
            "coverageCertificateSha256",
        ),
        context="certified pack trainerBinding",
    )
    if binding["schema"] != TRAINER_BINDING_SCHEMA:
        raise TrainerContractError("certified pack trainer binding has unsupported schema")
    source_group_inputs_hash = _sha256_uri(
        binding["sourceGroupInputsHash"], "trainerBinding.sourceGroupInputsHash"
    )
    if source_group_inputs_hash != content_hash(source_inputs):
        raise TrainerContractError("trainer binding does not match the source group inputs")
    split_assignment_hash = _sha256_uri(
        binding["splitAssignmentHash"], "trainerBinding.splitAssignmentHash"
    )
    coverage_hash = _sha256_uri(
        binding["coverageCertificateSha256"],
        "trainerBinding.coverageCertificateSha256",
    )
    if coverage_hash != outputs["causal_coverage_certificate.json"]["sha256"]:
        raise TrainerContractError("trainer binding does not match the coverage certificate")

    data_descriptor = _artifact_descriptor(
        binding["datasetContract"], "trainerBinding.datasetContract"
    )
    group_descriptor = _artifact_descriptor(
        binding["groupManifest"], "trainerBinding.groupManifest", group_records=True
    )
    model_descriptor = _artifact_descriptor(
        binding["modelContract"], "trainerBinding.modelContract"
    )
    _verify_bound_argument(data_descriptor, data_contract_path, "bound dataset contract")
    _verify_bound_argument(group_descriptor, group_manifest_path, "bound trainer group manifest")
    _verify_bound_argument(model_descriptor, model_contract_path, "bound model contract")
    model_contract = _load_json(model_contract_path.resolve(), "model contract")
    model_revision = _nonempty_string(
        model_contract.get("model_revision"), "model_contract.model_revision"
    )
    if binding["modelRevision"] != model_revision:
        raise TrainerContractError("trainer binding and model contract revisions differ")
    dataset_contract = NativeDatasetContract.from_mapping(
        _load_json(data_contract_path.resolve(), "dataset contract"),
        manifest_sha256=group_descriptor["sha256"],
        model_revision=model_revision,
    )

    assignments = _load_pack_assignments(
        pack_root / "listwise_groups.jsonl", dataset_contract
    )
    assignment_hash = content_hash(assignments)
    if assignment_hash != split_assignment_hash:
        raise TrainerContractError("trainer binding split assignment hash is stale")
    trainer_assignments = _load_trainer_assignments(
        group_manifest_path.resolve(), dataset_contract
    )
    if len(trainer_assignments) != group_descriptor["groupRecords"]:
        raise TrainerContractError("bound trainer group record count is stale")
    if trainer_assignments != assignments:
        raise TrainerContractError(
            "trainer group split assignment does not match the certified leakage pack"
        )

    component_splits, group_components = _load_component_splits(
        pack_root / "leakage_components.jsonl"
    )
    assignment_components = {item["groupId"]: item["componentId"] for item in assignments}
    if group_components != assignment_components:
        raise TrainerContractError("leakage component membership differs from the listwise index")
    if any(component_splits[item["componentId"]] != item["split"] for item in assignments):
        raise TrainerContractError("a certified leakage component crosses splits")

    certificate = _load_json(
        pack_root / "causal_coverage_certificate.json", "causal coverage certificate"
    )
    _required_keys(
        certificate,
        required=(
            "schema",
            "status",
            "groupCount",
            "siblingCount",
            "componentCount",
            "splitCounts",
            "componentSplits",
            "splitAssignmentHash",
            "requiredSiblingRoles",
            "coveragePolicy",
            "operatorFamilies",
            "changedPathSignatures",
            "nativePivotAlignment",
            "reasons",
        ),
        context="causal coverage certificate",
    )
    if certificate["schema"] != CERTIFICATE_SCHEMA or certificate["status"] != "certified":
        raise TrainerContractError("causal coverage certificate is not certified under the required schema")
    if certificate["reasons"] != []:
        raise TrainerContractError("certified causal coverage certificate contains rejection reasons")
    split_counts = {split: sum(item["split"] == split for item in assignments) for split in SPLITS}
    expected_certificate = {
        "groupCount": len(assignments),
        "siblingCount": len(assignments) * SIBLING_COUNT,
        "componentCount": len(component_splits),
        "splitCounts": split_counts,
        "componentSplits": component_splits,
        "splitAssignmentHash": assignment_hash,
        "requiredSiblingRoles": list(dataset_contract.sibling_roles),
    }
    mismatches = {
        key: (expected, certificate.get(key))
        for key, expected in expected_certificate.items()
        if certificate.get(key) != expected
    }
    if mismatches:
        raise TrainerContractError(f"causal coverage certificate mismatch: {mismatches}")
    policy = certificate["coveragePolicy"]
    if (
        not isinstance(policy, Mapping)
        or set(policy.get("requiredSplits", [])) != SPLITS
        or len(policy.get("requiredSplits", [])) != len(SPLITS)
        or not isinstance(policy.get("minimumDistinctPremises"), int)
        or isinstance(policy.get("minimumDistinctPremises"), bool)
        or policy["minimumDistinctPremises"] < 2
    ):
        raise TrainerContractError("causal coverage certificate does not cover all trainer splits")
    families = certificate["operatorFamilies"]
    paths = certificate["changedPathSignatures"]
    coverage_records = (
        list(families.values()) if isinstance(families, Mapping) else []
    ) + (paths if isinstance(paths, list) else [])
    if not coverage_records or any(
        not isinstance(record, Mapping)
        or record.get("accepted") is not True
        or record.get("missingRequiredSplits") != []
        for record in coverage_records
    ):
        raise TrainerContractError("causal operator coverage records are not fully accepted")

    counts = manifest["counts"]
    expected_counts = {
        "groups": len(assignments),
        "siblings": len(assignments) * SIBLING_COUNT,
        "components": len(component_splits),
        "pairwiseDiagnostics": len(assignments) * 6,
    }
    if not isinstance(counts, Mapping) or dict(counts) != expected_counts:
        raise TrainerContractError("certified pack manifest counts are stale")
    configuration = manifest["configuration"]
    if (
        not isinstance(configuration, Mapping)
        or set(configuration.get("requiredCoverageSplits", [])) != SPLITS
        or configuration.get("minimumDistinctPremises")
        != policy["minimumDistinctPremises"]
    ):
        raise TrainerContractError("certified pack configuration is weaker than the trainer gate")

    return CertifiedPackProof(
        manifest_id=manifest_id,
        manifest_sha256=hash_file(manifest_path),
        coverage_certificate_sha256=coverage_hash,
        split_assignment_hash=assignment_hash,
        source_group_inputs_hash=source_group_inputs_hash,
        dataset_contract_sha256=data_descriptor["sha256"],
        group_manifest_sha256=group_descriptor["sha256"],
        model_contract_sha256=model_descriptor["sha256"],
        model_revision=model_revision,
    )


class NativeTensorLoader:
    """Content-addressed tensor loader with no persistent host tensor cache."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.resolve()
        self._verified: dict[Path, str] = {}

    def load(self, reference: TensorReference) -> Tensor:
        path = _resolve_tensor_path(self.data_root, reference)
        prior = self._verified.get(path)
        if prior is not None and prior != reference.sha256:
            raise TrainerContractError(f"conflicting hashes declared for {path}")
        if prior is None:
            actual = hash_file(path)
            if actual != reference.sha256:
                raise TrainerContractError(
                    f"native tensor hash mismatch for {path}: expected {reference.sha256}, got {actual}"
                )
            self._verified[path] = actual
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            payload: Any = load_file(path, device="cpu")
        else:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping) or reference.key not in payload:
            raise TrainerContractError(f"{path} lacks required tensor key {reference.key!r}")
        tensor = payload[reference.key]
        if not isinstance(tensor, Tensor):
            raise TrainerContractError(f"{path}:{reference.key} is not a tensor")
        return tensor


@dataclass(frozen=True)
class LoadedSibling:
    spec: SiblingSpec
    codes: Tensor
    agent_target_mask: Tensor
    control_stream: Tensor
    condition_start_frame: int


@dataclass(frozen=True)
class LoadedGroup:
    spec: GroupSpec
    siblings: tuple[LoadedSibling, ...]
    padding_token_id: int


def load_native_group(
    group: GroupSpec,
    *,
    loader: NativeTensorLoader,
    contract: NativeDatasetContract,
) -> LoadedGroup:
    loaded: list[LoadedSibling] = []
    allowed_targets = set(
        contract.stream_layout.text_stream_indices
        + contract.stream_layout.agent_audio_stream_indices
    )
    forbidden_targets = sorted(set(range(contract.num_codebooks)) - allowed_targets)
    shared_prefix = loader.load(group.shared_prefix.native_codes)
    if shared_prefix.dtype != torch.long or shared_prefix.ndim != 2:
        raise TrainerContractError(
            f"{group.group_id} shared prefix must be int64 [codebooks, frames]"
        )
    if (
        shared_prefix.shape[0] != contract.num_codebooks
        or shared_prefix.shape[1] != group.shared_prefix.native_pivot_frame
    ):
        raise TrainerContractError(
            f"{group.group_id} shared prefix tensor does not terminate at its certified pivot"
        )
    if bool((shared_prefix < 0).any().item()):
        raise TrainerContractError(f"{group.group_id} shared prefix contains negative codes")
    for sibling in group.siblings:
        suffix = loader.load(sibling.native_suffix_codes)
        suffix_mask = loader.load(sibling.suffix_agent_target_mask)
        control = loader.load(sibling.control_stream)
        if suffix.dtype != torch.long or suffix.ndim != 2:
            raise TrainerContractError(
                f"{sibling.sibling_id} native suffix must be int64 [codebooks, frames]"
            )
        if suffix.shape[0] != contract.num_codebooks or suffix.shape[1] < 1:
            raise TrainerContractError(f"{sibling.sibling_id} native suffix shape is invalid")
        if bool((suffix < 0).any().item()):
            raise TrainerContractError(f"{sibling.sibling_id} native suffix contains negatives")
        if suffix_mask.dtype != torch.bool or suffix_mask.shape != suffix.shape:
            raise TrainerContractError(
                f"{sibling.sibling_id} suffix target mask must be bool and match its suffix"
            )
        if forbidden_targets and bool(suffix_mask[forbidden_targets].any().item()):
            raise TrainerContractError(
                f"{sibling.sibling_id} marks caller or unknown streams as targets"
            )
        if control.ndim != 2 or control.shape[0] < 1 or control.shape[1] != contract.control_hidden_size:
            raise TrainerContractError(
                f"{sibling.sibling_id} control stream must be [frames, {contract.control_hidden_size}]"
            )
        if not control.is_floating_point() or not bool(torch.isfinite(control).all().item()):
            raise TrainerContractError(
                f"{sibling.sibling_id} control stream must contain finite floating-point values"
            )
        alignment = sibling.alignment
        pivot = group.shared_prefix.native_pivot_frame
        if alignment.alignment_revision != sibling.control_revision:
            raise TrainerContractError(
                f"{sibling.sibling_id} alignment revision is stale relative to control"
            )
        if alignment.shared_prefix_sha256 != group.shared_prefix.native_codes.sha256:
            raise TrainerContractError(f"{sibling.sibling_id} alignment binds a stale shared prefix")
        if alignment.native_suffix_sha256 != sibling.native_suffix_codes.sha256:
            raise TrainerContractError(f"{sibling.sibling_id} alignment binds a stale native suffix")
        if alignment.target_mask_sha256 != sibling.suffix_agent_target_mask.sha256:
            raise TrainerContractError(f"{sibling.sibling_id} alignment binds a stale target mask")
        if {
            alignment.member_at_frame,
            alignment.donor_at_frame,
            alignment.suffix_start_frame,
        } != {pivot}:
            raise TrainerContractError(
                f"{sibling.sibling_id} donor/member/suffix alignment is stale at the shared pivot"
            )
        expected_end = pivot + int(suffix.shape[1])
        if alignment.suffix_end_frame != expected_end:
            raise TrainerContractError(
                f"{sibling.sibling_id} suffix end metadata is stale relative to tensor length"
            )
        codes = torch.cat([shared_prefix, suffix], dim=1)
        mask = torch.cat(
            [
                torch.zeros(
                    contract.num_codebooks,
                    pivot,
                    dtype=torch.bool,
                ),
                suffix_mask,
            ],
            dim=1,
        )
        try:
            snapshot = sibling.snapshot()
        except ValueError as exc:
            raise TrainerContractError(f"{sibling.sibling_id}: {exc}") from exc
        start = snapshot.condition_start_frame
        target_positions = torch.where(mask[list(sorted(allowed_targets))].any(dim=0))[0]
        if target_positions.numel() == 0:
            raise TrainerContractError(f"{sibling.sibling_id} has no agent-only targets")
        first_target = int(target_positions[0].item())
        if alignment.first_supervised_agent_frame != first_target:
            raise TrainerContractError(
                f"{sibling.sibling_id} first-target alignment metadata is stale"
            )
        if alignment.control_available_frame >= first_target:
            raise TrainerContractError(
                f"{sibling.sibling_id} control_available_frame must be strictly before the first target"
            )
        if start > first_target:
            raise TrainerContractError(
                f"{sibling.sibling_id} buffered control becomes active after the first target"
            )
        if not start <= sibling.probe_frame_index < first_target:
            raise TrainerContractError(
                f"{sibling.sibling_id} probe frame must be control-visible before the first response"
            )
        if snapshot.cancel_at_frame is not None:
            cancel = snapshot.cancel_at_frame
            if (
                sibling.alignment.cutoff_revision != sibling.control_revision
                or sibling.alignment.cutoff_generation_id != sibling.generation_id
            ):
                raise TrainerContractError(
                    f"{sibling.sibling_id} cutoff metadata is stale relative to generation/control"
                )
            if not first_target < cancel <= codes.shape[1]:
                raise TrainerContractError(
                    f"{sibling.sibling_id} cancellation lies outside its active native sequence"
                )
            if cancel != alignment.suffix_end_frame:
                raise TrainerContractError(
                    f"{sibling.sibling_id} suffix was not cropped exactly at its cutoff"
                )
            if cancel < codes.shape[1] and bool(mask[:, cancel:].any().item()):
                raise TrainerContractError(
                    f"{sibling.sibling_id} supervises agent output after cancellation"
                )
        if set(sibling.probe_targets) != set(contract.probe_slot_cardinalities):
            raise TrainerContractError(
                f"{sibling.sibling_id} probe labels do not match the certified slots"
            )
        for slot, target in sibling.probe_targets.items():
            if target >= contract.probe_slot_cardinalities[slot]:
                raise TrainerContractError(
                    f"{sibling.sibling_id} probe target {slot}={target} is out of range"
                )
        loaded.append(
            LoadedSibling(
                spec=sibling,
                codes=codes,
                agent_target_mask=mask,
                control_stream=control,
                condition_start_frame=start,
            )
        )
    return LoadedGroup(
        spec=group,
        siblings=tuple(loaded),
        padding_token_id=contract.padding_token_id,
    )


@dataclass(frozen=True)
class NativeGroupBatch:
    matrix_codes: Tensor
    matrix_masks: Tensor
    matrix_controls: Tensor
    matrix_starts: Tensor
    matrix_cancels: Tensor
    matched_codes: Tensor
    matched_masks: Tensor
    matched_controls: Tensor
    matched_starts: Tensor
    matched_cancels: Tensor
    diagonal_indices: Tensor
    probe_frames: Tensor
    probe_targets: dict[str, Tensor]


def build_group_batch(group: LoadedGroup, device: torch.device) -> NativeGroupBatch:
    if len(group.siblings) != SIBLING_COUNT:
        raise TrainerContractError("listwise training requires exactly four loaded siblings")
    controls = group.siblings
    max_control_frames = max(int(item.control_stream.shape[0]) for item in controls)
    hidden = int(controls[0].control_stream.shape[1])
    padded_controls = torch.zeros(
        SIBLING_COUNT,
        max_control_frames,
        hidden,
        dtype=torch.float32,
    )
    for index, item in enumerate(controls):
        padded_controls[index, : item.control_stream.shape[0]] = item.control_stream.float()
    max_native_frames = max(int(item.codes.shape[1]) for item in controls)
    codes = torch.full(
        (SIBLING_COUNT, controls[0].codes.shape[0], max_native_frames),
        group.padding_token_id,
        dtype=torch.long,
    )
    masks = torch.zeros_like(codes, dtype=torch.bool)
    for index, item in enumerate(controls):
        codes[index, :, : item.codes.shape[1]] = item.codes
        masks[index, :, : item.agent_target_mask.shape[1]] = item.agent_target_mask
    starts = torch.tensor(
        [item.condition_start_frame for item in controls], dtype=torch.long
    )
    cancels = torch.tensor(
        [
            item.spec.alignment.cutoff_frame
            if item.spec.alignment.cutoff_frame is not None
            else -1
            for item in controls
        ],
        dtype=torch.long,
    )
    target_indices = torch.arange(SIBLING_COUNT).repeat_interleave(SIBLING_COUNT)
    control_indices = torch.arange(SIBLING_COUNT).repeat(SIBLING_COUNT)
    diagonal = torch.arange(SIBLING_COUNT) * (SIBLING_COUNT + 1)
    return NativeGroupBatch(
        matrix_codes=codes.index_select(0, target_indices).to(device=device, non_blocking=False),
        matrix_masks=masks.index_select(0, target_indices).to(device=device, non_blocking=False),
        matrix_controls=padded_controls.index_select(0, control_indices).to(
            device=device, non_blocking=False
        ),
        matrix_starts=starts.index_select(0, control_indices).to(device=device),
        matrix_cancels=cancels.index_select(0, control_indices).to(device=device),
        matched_codes=codes.to(device=device, non_blocking=False),
        matched_masks=masks.to(device=device, non_blocking=False),
        matched_controls=padded_controls.to(device=device, non_blocking=False),
        matched_starts=starts.to(device=device),
        matched_cancels=cancels.to(device=device),
        diagonal_indices=diagonal.to(device=device),
        probe_frames=torch.tensor(
            [item.spec.probe_frame_index for item in controls],
            dtype=torch.long,
            device=device,
        ),
        probe_targets={
            slot: torch.tensor(
                [item.spec.probe_targets[slot] for item in controls],
                dtype=torch.long,
                device=device,
            )
            for slot in sorted(controls[0].spec.probe_targets)
        },
    )


@dataclass(frozen=True)
class ObjectiveWeights:
    matched: float
    listwise: float
    probe: float
    dropout: float

    def validate(self) -> None:
        if self.matched <= 0 or min(self.listwise, self.probe, self.dropout) < 0:
            raise TrainerContractError("objective weights must be non-negative and matched positive")


@dataclass
class GroupObjective:
    total: Tensor
    matched: Tensor
    listwise: Tensor
    probe: Tensor
    dropout: Tensor
    nll_matrix: Tensor
    matched_text: Tensor
    matched_audio: Tensor
    text_tokens: Tensor
    audio_tokens: Tensor
    probe_correct: Tensor
    probe_count: Tensor
    dropped_count: Tensor


def compose_causal_objective(
    nll_matrix: Tensor,
    probe_loss: Tensor,
    dropout_loss: Tensor,
    *,
    weights: ObjectiveWeights,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    weights.validate()
    matched = nll_matrix.diagonal().mean()
    listwise = listwise_causal_loss(nll_matrix, temperature=temperature)
    total = (
        weights.matched * matched
        + weights.listwise * listwise
        + weights.probe * probe_loss
        + weights.dropout * dropout_loss
    )
    return total, matched, listwise


def deterministic_dropout_mask(
    sibling_ids: Sequence[str],
    *,
    probability: float,
    seed: int,
    step: int,
    rank: int,
    micro_step: int,
    device: torch.device,
) -> Tensor:
    if not 0 <= probability < 1:
        raise TrainerContractError("control dropout must be in [0, 1)")
    values = []
    for sibling_id in sibling_ids:
        payload = f"{seed}:{step}:{rank}:{micro_step}:{sibling_id}".encode("utf-8")
        sample = int.from_bytes(sha256(payload).digest()[:8], "big") / float(1 << 64)
        values.append(sample < probability)
    return torch.tensor(values, dtype=torch.bool, device=device)


def _probe_loss_and_accuracy(
    probe: nn.Module,
    hidden: Tensor,
    frame_indices: Tensor,
    targets: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor]:
    logits = probe(hidden, frame_indices)
    if set(logits) != set(targets):
        raise TrainerContractError("probe outputs do not match typed targets")
    losses = []
    correct = hidden.new_zeros((), dtype=torch.float32)
    count = hidden.new_zeros((), dtype=torch.float32)
    for name in sorted(targets):
        losses.append(nn.functional.cross_entropy(logits[name].float(), targets[name]))
        correct += logits[name].argmax(dim=-1).eq(targets[name]).float().sum()
        count += float(targets[name].numel())
    return torch.stack(losses).mean(), correct, count


def forward_group_objective(
    lm_model: nn.Module,
    probe: nn.Module,
    batch: NativeGroupBatch,
    stream_layout: StreamLayout,
    *,
    objective_weights: ObjectiveWeights,
    audio_weight: float,
    listwise_temperature: float,
    dropout_mask: Tensor,
    activation_checkpointing: bool,
) -> GroupObjective:
    matrix_output = forward_with_native_streaming_sum(
        lm_model,
        batch.matrix_codes,
        batch.matrix_controls,
        batch.matrix_starts,
        cancel_at_frames=batch.matrix_cancels,
        activation_checkpointing=activation_checkpointing,
    )
    matrix_losses = agent_only_loss_per_example(
        lm_model,
        matrix_output,
        batch.matrix_codes,
        batch.matrix_masks,
        stream_layout,
        audio_weight=audio_weight,
    )
    nll_matrix = matrix_losses.total.reshape(SIBLING_COUNT, SIBLING_COUNT)
    diagonal_hidden = matrix_output.transformer_hidden.index_select(
        0, batch.diagonal_indices
    )
    probe_loss, probe_correct, probe_count = _probe_loss_and_accuracy(
        probe, diagonal_hidden, batch.probe_frames, batch.probe_targets
    )
    if dropout_mask.shape != (SIBLING_COUNT,) or dropout_mask.dtype != torch.bool:
        raise TrainerContractError("dropout mask must contain one bool per sibling")
    dropout_output = forward_with_native_streaming_sum(
        lm_model,
        batch.matched_codes,
        batch.matched_controls,
        batch.matched_starts,
        cancel_at_frames=batch.matched_cancels,
        control_dropout_mask=dropout_mask,
        activation_checkpointing=activation_checkpointing,
    )
    dropout_losses = agent_only_loss_per_example(
        lm_model,
        dropout_output,
        batch.matched_codes,
        batch.matched_masks,
        stream_layout,
        audio_weight=audio_weight,
    ).total
    dropped_count = dropout_mask.float().sum()
    dropout_loss = (
        (dropout_losses * dropout_mask.float()).sum() / dropped_count.clamp_min(1.0)
        if bool(dropout_mask.any().item())
        else dropout_losses.sum() * 0.0
    )
    total, matched, listwise = compose_causal_objective(
        nll_matrix,
        probe_loss,
        dropout_loss,
        weights=objective_weights,
        temperature=listwise_temperature,
    )
    matrix_text = matrix_losses.text.reshape(SIBLING_COUNT, SIBLING_COUNT)
    matrix_audio = matrix_losses.audio.reshape(SIBLING_COUNT, SIBLING_COUNT)
    matrix_text_tokens = matrix_losses.text_tokens.reshape(SIBLING_COUNT, SIBLING_COUNT)
    matrix_audio_tokens = matrix_losses.audio_tokens.reshape(SIBLING_COUNT, SIBLING_COUNT)
    return GroupObjective(
        total=total,
        matched=matched,
        listwise=listwise,
        probe=probe_loss,
        dropout=dropout_loss,
        nll_matrix=nll_matrix,
        matched_text=matrix_text.diagonal().mean(),
        matched_audio=matrix_audio.diagonal().mean(),
        text_tokens=matrix_text_tokens.diagonal().sum(),
        audio_tokens=matrix_audio_tokens.diagonal().sum(),
        probe_correct=probe_correct,
        probe_count=probe_count,
        dropped_count=dropped_count,
    )


def parse_meminfo(text: str) -> dict[str, float | int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.strip().split()
        if not fields:
            continue
        try:
            amount = int(fields[0])
        except ValueError as exc:
            raise TrainerContractError(f"invalid /proc/meminfo value for {key}") from exc
        unit = fields[1] if len(fields) > 1 else "B"
        multiplier = 1024 if unit == "kB" else 1
        values[key] = amount * multiplier
    if values.get("MemTotal", 0) <= 0 or "MemAvailable" not in values:
        raise TrainerContractError("/proc/meminfo lacks MemTotal or MemAvailable")
    total = values["MemTotal"]
    available = values["MemAvailable"]
    if not 0 <= available <= total:
        raise TrainerContractError("/proc/meminfo reports impossible available memory")
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_ratio": (total - available) / total,
    }


def host_memory_snapshot(path: Path = Path("/proc/meminfo")) -> dict[str, float | int]:
    return parse_meminfo(path.read_text(encoding="ascii"))


def wait_for_host_memory(
    *,
    limit: float,
    poll_seconds: float,
    timeout_seconds: float,
    snapshot_reader=host_memory_snapshot,
) -> dict[str, float | int]:
    if not 0 < limit <= MAX_HOST_MEMORY_RATIO:
        raise TrainerContractError("host memory limit must be in (0, 0.80]")
    if poll_seconds <= 0 or timeout_seconds <= 0:
        raise TrainerContractError("host memory poll and timeout must be positive")
    started = time.monotonic()
    while True:
        snapshot = snapshot_reader()
        if float(snapshot["used_ratio"]) <= limit:
            return snapshot
        if time.monotonic() - started >= timeout_seconds:
            raise TrainerContractError(
                f"host RAM remained at or above {limit:.0%} for {timeout_seconds:.0f}s"
            )
        time.sleep(poll_seconds)


def collective_wait_for_host_memory(
    *,
    device: torch.device,
    limit: float,
    poll_seconds: float,
    timeout_seconds: float,
) -> float:
    if not 0 < limit <= MAX_HOST_MEMORY_RATIO:
        raise TrainerContractError("host memory limit must be in (0, 0.80]")
    started = time.monotonic()
    while True:
        state = torch.zeros(2, dtype=torch.float64, device=device)
        if dist.get_rank() == 0:
            ratio = float(host_memory_snapshot()["used_ratio"])
            state[0] = ratio
            state[1] = float(
                ratio <= limit or time.monotonic() - started >= timeout_seconds
            )
        dist.broadcast(state, src=0)
        ratio = float(state[0].item())
        if ratio <= limit:
            return ratio
        if bool(state[1].item()):
            raise TrainerContractError(
                f"host RAM remained at or above {limit:.0%} for {timeout_seconds:.0f}s"
            )
        time.sleep(poll_seconds)


def parse_allowed_physical_gpus(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise TrainerContractError("allowed physical GPUs must be comma-separated integers") from exc
    if not parsed or len(set(parsed)) != len(parsed):
        raise TrainerContractError("allowed physical GPUs must be unique and non-empty")
    if not set(parsed).issubset(PHYSICAL_GPU_CEILING):
        raise TrainerContractError("this trainer is restricted to physical CUDA GPUs 0,1,2")
    return parsed


def visible_physical_gpus(
    environ: Mapping[str, str],
    *,
    cuda_device_count: int,
) -> tuple[int, ...]:
    raw = environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return tuple(range(cuda_device_count))
    try:
        visible = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise TrainerContractError(
            "CUDA_VISIBLE_DEVICES must contain physical integer indices, not UUID aliases"
        ) from exc
    if len(visible) != cuda_device_count or len(set(visible)) != len(visible):
        raise TrainerContractError("CUDA_VISIBLE_DEVICES does not match discovered CUDA ordinals")
    return visible


def validate_worker_devices(
    *,
    environ: Mapping[str, str],
    cuda_device_count: int,
    world_size: int,
    local_rank: int,
    allowed_physical_gpus: Sequence[int],
) -> int:
    if world_size < 2:
        raise TrainerContractError("full-rank FSDP training requires at least two torchrun ranks")
    visible = visible_physical_gpus(environ, cuda_device_count=cuda_device_count)
    if len(visible) != world_size:
        raise TrainerContractError(
            "CUDA_VISIBLE_DEVICES must expose exactly one admitted GPU per torchrun rank"
        )
    if local_rank < 0 or local_rank >= world_size:
        raise TrainerContractError("LOCAL_RANK lies outside WORLD_SIZE")
    if not set(visible).issubset(set(allowed_physical_gpus)):
        raise TrainerContractError("torchrun exposes a physical GPU outside the allowlist")
    if not set(visible).issubset(PHYSICAL_GPU_CEILING):
        raise TrainerContractError("torchrun exposed a physical GPU outside 0,1,2")
    return visible[local_rank]


def _broadcast_rank_zero(value_factory) -> Any:
    payload: list[Any] = [None]
    if dist.get_rank() == 0:
        try:
            payload[0] = {"ok": True, "value": value_factory()}
        except Exception as exc:  # propagated identically so peers do not hang
            payload[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(payload, src=0)
    result = payload[0]
    if not isinstance(result, Mapping) or not result.get("ok"):
        error = result.get("error", "rank-zero operation failed") if isinstance(result, Mapping) else result
        raise TrainerContractError(str(error))
    return result["value"]


def preflight_dataset(
    groups: Sequence[GroupSpec],
    *,
    loader: NativeTensorLoader,
    contract: NativeDatasetContract,
) -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    error: str | None = None
    try:
        for index in range(rank, len(groups), world_size):
            load_native_group(groups[index], loader=loader, contract=contract)
    except Exception as exc:
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    errors: list[str | None] = [None] * world_size
    dist.all_gather_object(errors, error)
    failures = [item for item in errors if item]
    if failures:
        raise TrainerContractError("dataset tensor preflight failed: " + " | ".join(failures))


def deterministic_group(
    groups: Sequence[GroupSpec],
    *,
    seed: int,
    global_sample_index: int,
) -> GroupSpec:
    if not groups:
        raise TrainerContractError("cannot sample an empty training split")
    epoch, offset = divmod(global_sample_index, len(groups))
    ordered = sorted(
        groups,
        key=lambda group: sha256(f"{seed}:{epoch}:{group.group_id}".encode("utf-8")).digest(),
    )
    return ordered[offset]


def _select_evaluation_groups(
    groups: Sequence[GroupSpec], *, seed: int, limit: int, namespace: str
) -> tuple[GroupSpec, ...]:
    if limit < 1:
        raise TrainerContractError("evaluation group limit must be positive")
    ordered = sorted(
        groups,
        key=lambda group: sha256(
            f"eval:{namespace}:{seed}:{group.group_id}".encode("utf-8")
        ).digest(),
    )
    return tuple(ordered[: min(limit, len(ordered))])


def evaluate_groups(
    lm_model: nn.Module,
    probe: nn.Module,
    groups: Sequence[GroupSpec],
    *,
    namespace: str,
    loader: NativeTensorLoader,
    contract: NativeDatasetContract,
    device: torch.device,
    audio_weight: float,
    listwise_temperature: float,
    strict_margin: float,
    seed: int,
    limit: int,
) -> dict[str, Any]:
    selected = _select_evaluation_groups(
        groups, seed=seed, limit=limit, namespace=namespace
    )
    if not selected:
        raise TrainerContractError(f"{namespace} evaluation split is empty")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    rounds = math.ceil(len(selected) / world_size)
    totals = torch.zeros(11, dtype=torch.float64, device=device)
    was_training = lm_model.training
    lm_model.eval()
    probe.eval()
    no_dropout_weights = ObjectiveWeights(1.0, 1.0, 1.0, 0.0)
    with torch.no_grad():
        for round_index in range(rounds):
            selected_index = round_index * world_size + rank
            include = selected_index < len(selected)
            group = selected[selected_index] if include else selected[0]
            loaded = load_native_group(group, loader=loader, contract=contract)
            batch = build_group_batch(loaded, device)
            result = forward_group_objective(
                lm_model,
                probe,
                batch,
                contract.stream_layout,
                objective_weights=no_dropout_weights,
                audio_weight=audio_weight,
                listwise_temperature=listwise_temperature,
                dropout_mask=torch.zeros(SIBLING_COUNT, dtype=torch.bool, device=device),
                activation_checkpointing=False,
            )
            if include:
                totals += torch.tensor(
                    [
                        1.0,
                        float(result.matched.detach()),
                        float(result.listwise.detach()),
                        float(result.probe.detach()),
                        float(
                            strict_listwise_group_pass(
                                result.nll_matrix.detach(), minimum_margin=strict_margin
                            )
                        ),
                        float(result.probe_correct.detach()),
                        float(result.probe_count.detach()),
                        float(result.matched_text.detach()),
                        float(result.matched_audio.detach()),
                        float(result.text_tokens.detach()),
                        float(result.audio_tokens.detach()),
                    ],
                    dtype=torch.float64,
                    device=device,
                )
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if was_training:
        lm_model.train()
        probe.train()
    count = int(totals[0].item())
    if count != len(selected):
        raise TrainerContractError(
            f"{namespace} evaluation reduced {count} groups, expected {len(selected)}"
        )
    return {
        "scope": namespace,
        "groups": count,
        "matchedNll": totals[1].item() / count,
        "listwiseLoss": totals[2].item() / count,
        "probeLoss": totals[3].item() / count,
        "strictGroupPasses": int(totals[4].item()),
        "strictGroupPassRate": totals[4].item() / count,
        "probeAccuracy": totals[5].item() / max(totals[6].item(), 1.0),
        "textNll": totals[7].item() / count,
        "audioNll": totals[8].item() / count,
        "textTokens": int(totals[9].item()),
        "audioTokens": int(totals[10].item()),
        "evaluationMode": "teacher_forced_group_disjoint_native_delayed_duplex",
    }


def checkpoint_summary_record(
    *,
    step: int,
    checkpoint: str,
    heldout: Mapping[str, Any],
    train: Mapping[str, Any],
    minimum_group_pass_rate: float,
    minimum_probe_accuracy: float,
) -> dict[str, Any]:
    gate = {
        "minimumHeldoutStrictGroupPassRate": minimum_group_pass_rate,
        "minimumHeldoutProbeAccuracy": minimum_probe_accuracy,
        "strictGroupPassRatePassed": heldout["strictGroupPassRate"]
        >= minimum_group_pass_rate,
        "probeAccuracyPassed": heldout["probeAccuracy"] >= minimum_probe_accuracy,
    }
    gate["passed"] = gate["strictGroupPassRatePassed"] and gate["probeAccuracyPassed"]
    return {
        "event": "checkpoint_summary",
        "step": step,
        "checkpoint": checkpoint,
        "heldout": dict(heldout),
        "train": dict(train),
        "generalizationGap": {
            "strictGroupPassRate": train["strictGroupPassRate"]
            - heldout["strictGroupPassRate"],
            "probeAccuracy": train["probeAccuracy"] - heldout["probeAccuracy"],
        },
        "teacherForcedGate": gate,
        "promotionScope": "teacher-forced native causal gate; generated duplex and live-call gates remain mandatory",
    }


def compact_step_record(
    *,
    step: int,
    reduced: Sequence[float],
    world_size: int,
    duration_seconds: float,
    host_ram_used_ratio: float,
) -> dict[str, Any]:
    if len(reduced) != 11 or world_size < 1:
        raise TrainerContractError("invalid compact step telemetry payload")
    scale = 1.0 / world_size
    return {
        "event": "step",
        "step": step,
        "loss": reduced[0] * scale,
        "matched": reduced[1] * scale,
        "listwise": reduced[2] * scale,
        "probe": reduced[3] * scale,
        "dropout": reduced[4] * scale,
        "textNll": reduced[5] * scale,
        "audioNll": reduced[6] * scale,
        "textTokens": int(reduced[7]),
        "audioTokens": int(reduced[8]),
        "droppedControls": int(reduced[9]),
        "globalShardedGradNorm": reduced[10] * scale,
        "seconds": duration_seconds,
        "hostRamUsedRatio": host_ram_used_ratio,
    }


def _append_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _materialize_adamw_state(optimizer: torch.optim.AdamW) -> None:
    """Create tensor placeholders so DCP can restore an AdamW state dictionary."""

    for group in optimizer.param_groups:
        capturable = bool(group.get("capturable", False))
        for parameter in group["params"]:
            state = optimizer.state[parameter]
            if state:
                continue
            step_device = parameter.device if capturable else torch.device("cpu")
            state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)
            if bool(group.get("amsgrad", False)):
                state["max_exp_avg_sq"] = torch.zeros_like(parameter)


def save_probe_checkpoint(
    probe: DistributedDataParallel,
    optimizer: torch.optim.AdamW,
    checkpoint_dir: Path,
) -> None:
    state = {
        "probe": probe.module.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    dcp.save(state, checkpoint_id=checkpoint_dir / "probe_shards")
    dist.barrier()


def load_probe_checkpoint(
    probe: DistributedDataParallel,
    optimizer: torch.optim.AdamW,
    checkpoint_dir: Path,
) -> None:
    _materialize_adamw_state(optimizer)
    state = {
        "probe": probe.module.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    dcp.load(state, checkpoint_id=checkpoint_dir / "probe_shards")
    probe.module.load_state_dict(state["probe"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    dist.barrier()


def _model_hidden_size(lm_model: nn.Module) -> int:
    text_linear = getattr(lm_model, "text_linear", None)
    hidden = getattr(text_linear, "in_features", None)
    if not isinstance(hidden, int) or hidden < 1:
        raise TrainerContractError("cannot determine PersonaPlex temporal hidden size")
    return hidden


def _assert_full_rank_receiver_selection(lm_model: nn.Module) -> dict[str, Any]:
    selection = select_full_rank_temporal_text_parameters(lm_model)
    transformer = getattr(lm_model, "transformer", None)
    layers = getattr(transformer, "layers", None)
    if layers is None or len(layers) < 1:
        raise TrainerContractError("PersonaPlex temporal transformer has no discoverable layers")
    selected_ids = {id(parameter) for parameter in selection.parameters}
    for layer_index, layer in enumerate(layers):
        layer_parameters = tuple(layer.parameters())
        if not layer_parameters or any(id(parameter) not in selected_ids for parameter in layer_parameters):
            raise TrainerContractError(
                f"temporal layer {layer_index} is not fully selected for full-rank training"
            )
    return {
        "temporalLayers": len(layers),
        "parameterCount": selection.parameter_count,
        "parameterNames": list(selection.parameter_names),
    }


def _assert_post_shard_freeze_contract(
    lm_model: nn.Module,
    bundle: FSDPReceiverBundle,
    expected_parameter_count: int,
) -> None:
    if bundle.cpu_offload:
        raise TrainerContractError("receiver FSDP unexpectedly enabled CPU offload")
    if bundle.trainable_parameter_count != expected_parameter_count:
        raise TrainerContractError("FSDP receiver selection changed the full-rank parameter set")
    allowed_roots = {"transformer", "text_emb", "text_linear", "out_norm"}
    unexpected = []
    for name, parameter in lm_model.named_parameters():
        root = name.split(".", 1)[0]
        if parameter.requires_grad and root not in allowed_roots:
            unexpected.append(name)
    if unexpected:
        raise TrainerContractError(
            f"Mimi/audio/depth/voice parameters became trainable: {unexpected[:8]}"
        )


def _training_fingerprint(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "learningRate": args.learning_rate,
        "probeLearningRate": args.probe_learning_rate,
        "weightDecay": args.weight_decay,
        "audioWeight": args.audio_weight,
        "matchedWeight": args.matched_weight,
        "listwiseWeight": args.listwise_weight,
        "probeWeight": args.probe_weight,
        "dropoutWeight": args.dropout_weight,
        "controlDropout": args.control_dropout,
        "listwiseTemperature": args.listwise_temperature,
        "strictMargin": args.strict_margin,
        "gradientAccumulation": args.gradient_accumulation,
        "seed": args.seed,
        "frameDurationMs": FRAME_DURATION_MS,
        "siblingCount": SIBLING_COUNT,
    }


def _checkpoint_metadata(
    *,
    args: argparse.Namespace,
    step: int,
    contract: NativeDatasetContract,
    model_contract: Mapping[str, Any],
    groups: Sequence[GroupSpec],
    receiver: Mapping[str, Any],
    physical_gpus: Sequence[int],
    certified_pack: CertifiedPackProof,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "step": step,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "manifestSha256": contract.manifest_sha256,
        "datasetContractSha256": hash_file(args.data_contract.resolve()),
        "modelRevision": contract.model_revision,
        "modelWeightsSha256": model_contract["moshi_weights_sha256"],
        "certifiedPack": certified_pack.to_mapping(),
        "worldSize": dist.get_world_size(),
        "physicalGpus": list(physical_gpus),
        "cpuOffload": False,
        "receiver": dict(receiver),
        "training": _training_fingerprint(args),
        "groups": {
            "train": sum(group.split == "train" for group in groups),
            "heldout": sum(group.split == "validation" for group in groups),
            "test": sum(group.split == "test" for group in groups),
        },
        "checkpointGates": list(CHECKPOINT_GATES),
    }


def _validate_resume_metadata(
    metadata: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    contract: NativeDatasetContract,
    model_contract: Mapping[str, Any],
    certified_pack: CertifiedPackProof,
) -> int:
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "manifestSha256": contract.manifest_sha256,
        "datasetContractSha256": hash_file(args.data_contract.resolve()),
        "modelRevision": contract.model_revision,
        "modelWeightsSha256": model_contract["moshi_weights_sha256"],
        "certifiedPack": certified_pack.to_mapping(),
        "worldSize": dist.get_world_size(),
        "training": _training_fingerprint(args),
    }
    mismatches = {
        key: (expected_value, metadata.get(key))
        for key, expected_value in expected.items()
        if metadata.get(key) != expected_value
    }
    if mismatches:
        raise TrainerContractError(f"resume checkpoint contract mismatch: {mismatches}")
    step = metadata.get("step")
    if not isinstance(step, int) or step < 1 or step >= args.max_steps:
        raise TrainerContractError("resume step must be positive and below max-steps")
    return step


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainerContractError(f"cannot read {context}: {path}") from exc
    if not isinstance(value, dict):
        raise TrainerContractError(f"{context} must be a JSON object")
    return value


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_steps < max(CHECKPOINT_GATES):
        raise TrainerContractError("max-steps must reach mandatory gates 100, 125, and 150")
    if args.workers < 2 or args.workers > len(PHYSICAL_GPU_CEILING):
        raise TrainerContractError("workers must be two or three")
    if args.gradient_accumulation < 1 or min(args.eval_groups, args.train_eval_groups) < 1:
        raise TrainerContractError("accumulation and evaluation group counts must be positive")
    positive = (
        args.learning_rate,
        args.probe_learning_rate,
        args.audio_weight,
        args.listwise_temperature,
        args.max_grad_norm,
        args.host_memory_poll_seconds,
        args.host_memory_timeout_seconds,
    )
    if min(positive) <= 0:
        raise TrainerContractError("learning, loss, clipping, and admission values must be positive")
    if not 0 <= args.control_dropout < 1:
        raise TrainerContractError("control-dropout must be in [0, 1)")
    if not 0 < args.host_memory_limit <= MAX_HOST_MEMORY_RATIO:
        raise TrainerContractError("host-memory-limit must be in (0, 0.80]")
    if not 0 < args.gpu_min_usable_ratio <= 1:
        raise TrainerContractError("gpu-min-usable-ratio must be in (0, 1]")
    if not 0 <= args.gpu_reserve_ratio < 1:
        raise TrainerContractError("gpu-reserve-ratio must be in [0, 1)")
    if args.gpu_min_usable_ratio + args.gpu_reserve_ratio > 1:
        raise TrainerContractError("GPU usable and reserve ratios cannot exceed one")
    if not 0 <= args.gpu_max_utilization_pct <= 100:
        raise TrainerContractError("gpu-max-utilization-pct must be in [0, 100]")
    if not 0 <= args.strict_margin:
        raise TrainerContractError("strict-margin must be non-negative")
    for threshold in (args.gate_min_group_pass_rate, args.gate_min_probe_accuracy):
        if not 0 <= threshold <= 1:
            raise TrainerContractError("checkpoint gate thresholds must be in [0, 1]")
    ObjectiveWeights(
        args.matched_weight,
        args.listwise_weight,
        args.probe_weight,
        args.dropout_weight,
    ).validate()
    allowed = parse_allowed_physical_gpus(args.allowed_physical_gpus)
    if args.workers > len(allowed):
        raise TrainerContractError("workers exceeds the physical GPU allowlist")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-contract", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--certified-pack-manifest", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--allowed-physical-gpus", default="0,1,2")
    parser.add_argument("--gpu-min-usable-ratio", type=float, default=0.55)
    parser.add_argument("--gpu-reserve-ratio", type=float, default=0.10)
    parser.add_argument("--gpu-max-utilization-pct", type=int, default=30)
    parser.add_argument("--host-memory-limit", type=float, default=MAX_HOST_MEMORY_RATIO)
    parser.add_argument("--host-memory-poll-seconds", type=float, default=5.0)
    parser.add_argument("--host-memory-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--audio-weight", type=float, default=0.02)
    parser.add_argument("--matched-weight", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=0.25)
    parser.add_argument("--probe-weight", type=float, default=0.10)
    parser.add_argument("--dropout-weight", type=float, default=0.10)
    parser.add_argument("--control-dropout", type=float, default=0.10)
    parser.add_argument("--listwise-temperature", type=float, default=0.20)
    parser.add_argument("--strict-margin", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--eval-groups", type=int, default=32)
    parser.add_argument("--train-eval-groups", type=int, default=32)
    parser.add_argument("--gate-min-group-pass-rate", type=float, default=0.95)
    parser.add_argument("--gate-min-probe-accuracy", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--activation-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require-final-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--distributed-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--admission-report", type=Path, help=argparse.SUPPRESS)
    return parser


def launch_distributed(args: argparse.Namespace, argv: Sequence[str]) -> int:
    allowed = parse_allowed_physical_gpus(args.allowed_physical_gpus)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    wait_for_host_memory(
        limit=args.host_memory_limit,
        poll_seconds=args.host_memory_poll_seconds,
        timeout_seconds=args.host_memory_timeout_seconds,
    )
    report = admit_gpus_by_ratio(
        world_size=args.workers,
        min_usable_ratio=args.gpu_min_usable_ratio,
        reserve_ratio=args.gpu_reserve_ratio,
        max_utilization_pct=args.gpu_max_utilization_pct,
        allowed_indices=allowed,
    )
    if report.get("status") != "admitted":
        raise TrainerContractError(str(report.get("refusal") or "GPU admission refused"))
    selected = report.get("selected_gpu_indices")
    if not isinstance(selected, list) or len(selected) != args.workers:
        raise TrainerContractError("GPU admission returned an invalid selection")
    if not set(selected).issubset(PHYSICAL_GPU_CEILING):
        raise TrainerContractError("GPU admission escaped the physical 0,1,2 ceiling")
    report_path = args.run_dir / "gpu_admission.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in selected)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={args.workers}",
        str(Path(__file__).resolve()),
        *argv,
        "--distributed-worker",
        "--admission-report",
        str(report_path),
    ]
    return subprocess.run(command, env=environment, check=False).returncode


def _worker_admission(
    args: argparse.Namespace,
    *,
    visible: Sequence[int],
) -> dict[str, Any]:
    allowed = parse_allowed_physical_gpus(args.allowed_physical_gpus)

    def load_or_create() -> dict[str, Any]:
        if args.admission_report is not None:
            report = _load_json(args.admission_report.resolve(), "GPU admission report")
        else:
            report = admit_gpus_by_ratio(
                world_size=dist.get_world_size(),
                min_usable_ratio=args.gpu_min_usable_ratio,
                reserve_ratio=args.gpu_reserve_ratio,
                max_utilization_pct=args.gpu_max_utilization_pct,
                allowed_indices=allowed,
            )
            args.run_dir.mkdir(parents=True, exist_ok=True)
            (args.run_dir / "gpu_admission.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return report

    report = _broadcast_rank_zero(load_or_create)
    selected = report.get("selected_gpu_indices") if isinstance(report, Mapping) else None
    if report.get("status") != "admitted" or not isinstance(selected, list):
        raise TrainerContractError(str(report.get("refusal") or "GPU admission refused"))
    if set(selected) != set(visible) or len(selected) != len(visible):
        raise TrainerContractError(
            "admitted physical GPUs do not match torchrun CUDA_VISIBLE_DEVICES"
        )
    return dict(report)


def run_worker(
    args: argparse.Namespace, certified_pack: CertifiedPackProof | None = None
) -> int:
    if certified_pack is None:
        certified_pack = verify_certified_pack(
            args.certified_pack_manifest,
            data_contract_path=args.data_contract,
            group_manifest_path=args.group_manifest,
            model_contract_path=args.model_contract,
        )
    if not torch.cuda.is_available() or not torch.distributed.is_nccl_available():
        raise TrainerContractError("native full-rank training requires CUDA and NCCL; CPU fallback is forbidden")
    if not torch.cuda.is_bf16_supported():
        raise TrainerContractError("admitted CUDA device does not support native bfloat16 training")
    required_environment = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    if any(name not in os.environ for name in required_environment):
        raise TrainerContractError("worker must be launched by torchrun")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    allowed = parse_allowed_physical_gpus(args.allowed_physical_gpus)
    physical_gpu = validate_worker_devices(
        environ=os.environ,
        cuda_device_count=torch.cuda.device_count(),
        world_size=world_size,
        local_rank=local_rank,
        allowed_physical_gpus=allowed,
    )
    if world_size != args.workers:
        raise TrainerContractError("torchrun WORLD_SIZE does not match --workers")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    final_gate_passed = False
    try:
        visible = visible_physical_gpus(
            os.environ, cuda_device_count=torch.cuda.device_count()
        )
        admission = _worker_admission(args, visible=visible)
        physical_by_rank: list[int | None] = [None] * world_size
        dist.all_gather_object(physical_by_rank, physical_gpu)
        if tuple(int(item) for item in physical_by_rank) != tuple(visible):
            raise TrainerContractError("rank-to-physical-GPU mapping is inconsistent")
        host_ratio = collective_wait_for_host_memory(
            device=device,
            limit=args.host_memory_limit,
            poll_seconds=args.host_memory_poll_seconds,
            timeout_seconds=args.host_memory_timeout_seconds,
        )
        model_contract = _load_json(args.model_contract.resolve(), "model contract")
        manifest_path = args.group_manifest.resolve()
        if not manifest_path.is_file():
            raise TrainerContractError(f"group manifest is absent: {manifest_path}")
        manifest_hash = hash_file(manifest_path)
        if manifest_hash != certified_pack.group_manifest_sha256:
            raise TrainerContractError("trainer group manifest changed after certified-pack admission")
        model_revision = _nonempty_string(
            model_contract.get("model_revision"), "model_contract.model_revision"
        )
        dataset_contract = NativeDatasetContract.from_mapping(
            _load_json(args.data_contract.resolve(), "dataset contract"),
            manifest_sha256=manifest_hash,
            model_revision=model_revision,
        )
        model_layout = StreamLayout.from_mapping(model_contract.get("stream_layout", {}))
        if model_layout != dataset_contract.stream_layout:
            raise TrainerContractError("dataset and model stream layouts differ")
        groups = load_group_manifest(
            manifest_path,
            data_root=args.data_root.resolve(),
            contract=dataset_contract,
        )
        train_groups = tuple(group for group in groups if group.split == "train")
        heldout_groups = tuple(group for group in groups if group.split == "validation")
        if len(train_groups) < world_size:
            raise TrainerContractError("training split has fewer groups than torchrun ranks")
        tensor_loader = NativeTensorLoader(args.data_root.resolve())
        preflight_dataset(groups, loader=tensor_loader, contract=dataset_contract)
        integrity = _broadcast_rank_zero(
            lambda: {
                "modelWeightsSha256": hash_file(args.moshi_path.resolve()),
                "source": require_moshi_source_contract(
                    args.moshi_source_root.resolve(), model_contract
                ),
            }
        )
        if integrity["modelWeightsSha256"] != model_contract.get("moshi_weights_sha256"):
            raise TrainerContractError("PersonaPlex weights do not match the model contract")
        sys.path.insert(0, str(args.moshi_source_root.resolve()))
        from moshi.models.loaders import get_moshi_lm

        collective_wait_for_host_memory(
            device=device,
            limit=args.host_memory_limit,
            poll_seconds=args.host_memory_poll_seconds,
            timeout_seconds=args.host_memory_timeout_seconds,
        )
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        lm_model = get_moshi_lm(
            args.moshi_path.resolve(), device=device, dtype=torch.bfloat16
        )
        if any(parameter.device.type != "cuda" for parameter in lm_model.parameters()):
            raise TrainerContractError("PersonaPlex model used a CPU parameter fallback")
        if "streaming_sum" not in inspect.signature(lm_model.forward_embeddings).parameters:
            raise TrainerContractError(
                "native PersonaPlex source lacks forward_embeddings(streaming_sum=...)"
            )
        dataset_contract.stream_layout.validate_for_model(lm_model)
        if int(lm_model.num_codebooks) != dataset_contract.num_codebooks:
            raise TrainerContractError("dataset codebook count differs from PersonaPlex")
        if int(lm_model.zero_token_id) != dataset_contract.padding_token_id:
            raise TrainerContractError("dataset padding token differs from PersonaPlex zero token")
        hidden_size = _model_hidden_size(lm_model)
        if hidden_size != dataset_contract.control_hidden_size:
            raise TrainerContractError("ARC control width differs from PersonaPlex hidden size")
        receiver_record = _assert_full_rank_receiver_selection(lm_model)
        bundle = shard_full_rank_temporal_text_receiver(lm_model, device=device)
        _assert_post_shard_freeze_contract(
            lm_model, bundle, int(receiver_record["parameterCount"])
        )
        receiver_optimizer = torch.optim.AdamW(
            bundle.trainable_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        probe_module = PreResponseControlStateProbe(
            hidden_size, dataset_contract.probe_slot_cardinalities
        ).to(device=device, dtype=torch.bfloat16)
        probe = DistributedDataParallel(
            probe_module,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )
        probe_optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=args.probe_learning_rate,
            weight_decay=args.weight_decay,
        )
        start_step = 0
        if args.resume_checkpoint is not None:
            checkpoint_dir = args.resume_checkpoint.resolve()
            if not (checkpoint_dir / "COMPLETE").is_file():
                raise TrainerContractError("resume checkpoint is incomplete")
            metadata = _broadcast_rank_zero(
                lambda: _load_json(checkpoint_dir / "metadata.json", "checkpoint metadata")
            )
            start_step = _validate_resume_metadata(
                metadata,
                args=args,
                contract=dataset_contract,
                model_contract=model_contract,
                certified_pack=certified_pack,
            )
            load_receiver_checkpoint(
                lm_model, receiver_optimizer, bundle, checkpoint_dir
            )
            load_probe_checkpoint(probe, probe_optimizer, checkpoint_dir)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = args.run_dir / "metrics.jsonl"
        if rank == 0:
            run_contract = {
                "schema": "personaplex.native-moshirag-full-rank-run.v1",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "manifest": str(manifest_path),
                "manifestSha256": manifest_hash,
                "dataContract": str(args.data_contract.resolve()),
                "dataContractSha256": hash_file(args.data_contract.resolve()),
                "modelContract": str(args.model_contract.resolve()),
                "modelRevision": model_revision,
                "certifiedPack": certified_pack.to_mapping(),
                "moshiSource": integrity["source"],
                "gpuAdmission": admission,
                "physicalGpus": list(visible),
                "hostMemoryLimit": args.host_memory_limit,
                "hostMemoryUsedRatioAtStart": host_ratio,
                "receiver": receiver_record,
                "frozen": ["mimi", "audio_embeddings", "depformer", "voice_conditioning"],
                "trainable": [
                    "all_temporal_transformer_layers",
                    "text_embedding",
                    "text_head",
                    "output_norm",
                    "training_only_state_probe",
                ],
                "training": _training_fingerprint(args),
                "checkpointGates": list(CHECKPOINT_GATES),
                "resume": str(args.resume_checkpoint.resolve()) if args.resume_checkpoint else None,
            }
            (args.run_dir / "run_contract.json").write_text(
                json.dumps(run_contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        dist.barrier()
        objective_weights = ObjectiveWeights(
            args.matched_weight,
            args.listwise_weight,
            args.probe_weight,
            args.dropout_weight,
        )

        def checkpoint(step: int) -> bool:
            checkpoint_dir = args.run_dir / f"checkpoint-step-{step:06d}"
            existence = _broadcast_rank_zero(
                lambda: {
                    "exists": checkpoint_dir.exists(),
                    "path": str(checkpoint_dir),
                }
            )
            if existence["exists"]:
                raise TrainerContractError(
                    f"refusing to overwrite checkpoint: {checkpoint_dir}"
                )
            metadata = _checkpoint_metadata(
                args=args,
                step=step,
                contract=dataset_contract,
                model_contract=model_contract,
                groups=groups,
                receiver=receiver_record,
                physical_gpus=visible,
                certified_pack=certified_pack,
            )
            save_receiver_checkpoint(
                lm_model,
                receiver_optimizer,
                bundle,
                checkpoint_dir,
                metadata,
            )
            save_probe_checkpoint(probe, probe_optimizer, checkpoint_dir)
            heldout = evaluate_groups(
                lm_model,
                probe.module,
                heldout_groups,
                namespace="heldout",
                loader=tensor_loader,
                contract=dataset_contract,
                device=device,
                audio_weight=args.audio_weight,
                listwise_temperature=args.listwise_temperature,
                strict_margin=args.strict_margin,
                seed=args.seed,
                limit=args.eval_groups,
            )
            train_evaluation = evaluate_groups(
                lm_model,
                probe.module,
                train_groups,
                namespace="train",
                loader=tensor_loader,
                contract=dataset_contract,
                device=device,
                audio_weight=args.audio_weight,
                listwise_temperature=args.listwise_temperature,
                strict_margin=args.strict_margin,
                seed=args.seed,
                limit=args.train_eval_groups,
            )
            summary = checkpoint_summary_record(
                step=step,
                checkpoint=checkpoint_dir.name,
                heldout=heldout,
                train=train_evaluation,
                minimum_group_pass_rate=args.gate_min_group_pass_rate,
                minimum_probe_accuracy=args.gate_min_probe_accuracy,
            )
            if rank == 0:
                (checkpoint_dir / "evaluation.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                (checkpoint_dir / "COMPLETE").write_text("complete\n", encoding="ascii")
                _append_json(metrics_path, summary)
                print(json.dumps(summary, sort_keys=True, separators=(",", ":")), flush=True)
            dist.barrier()
            return bool(summary["teacherForcedGate"]["passed"])

        checkpoint_steps = set(CHECKPOINT_GATES) | {args.max_steps}
        lm_model.train()
        probe.train()
        for step in range(start_step + 1, args.max_steps + 1):
            step_started = time.monotonic()
            host_ratio = collective_wait_for_host_memory(
                device=device,
                limit=args.host_memory_limit,
                poll_seconds=args.host_memory_poll_seconds,
                timeout_seconds=args.host_memory_timeout_seconds,
            )
            receiver_optimizer.zero_grad(set_to_none=True)
            probe_optimizer.zero_grad(set_to_none=True)
            local = torch.zeros(10, dtype=torch.float64, device=device)
            for micro_step in range(args.gradient_accumulation):
                sample_number = (
                    ((step - 1) * args.gradient_accumulation + micro_step) * world_size
                    + rank
                )
                group = deterministic_group(
                    train_groups,
                    seed=args.seed,
                    global_sample_index=sample_number,
                )
                loaded = load_native_group(
                    group, loader=tensor_loader, contract=dataset_contract
                )
                batch = build_group_batch(loaded, device)
                dropout_mask = deterministic_dropout_mask(
                    [sibling.spec.sibling_id for sibling in loaded.siblings],
                    probability=args.control_dropout,
                    seed=args.seed,
                    step=step,
                    rank=rank,
                    micro_step=micro_step,
                    device=device,
                )
                synchronize = micro_step + 1 == args.gradient_accumulation
                with ExitStack() as stack:
                    if not synchronize:
                        stack.enter_context(bundle.no_sync())
                        stack.enter_context(probe.no_sync())
                    result = forward_group_objective(
                        lm_model,
                        probe,
                        batch,
                        dataset_contract.stream_layout,
                        objective_weights=objective_weights,
                        audio_weight=args.audio_weight,
                        listwise_temperature=args.listwise_temperature,
                        dropout_mask=dropout_mask,
                        activation_checkpointing=args.activation_checkpointing,
                    )
                    (result.total / args.gradient_accumulation).backward()
                local += torch.stack(
                    [
                        result.total.detach().double(),
                        result.matched.detach().double(),
                        result.listwise.detach().double(),
                        result.probe.detach().double(),
                        result.dropout.detach().double(),
                        result.matched_text.detach().double(),
                        result.matched_audio.detach().double(),
                        result.text_tokens.detach().double(),
                        result.audio_tokens.detach().double(),
                        result.dropped_count.detach().double(),
                    ]
                )
            local[:7] /= args.gradient_accumulation
            global_norm = clip_sharded_grad_norm(
                bundle.trainable_parameters, max_norm=args.max_grad_norm
            )
            nn.utils.clip_grad_norm_(probe.parameters(), args.max_grad_norm)
            receiver_optimizer.step()
            probe_optimizer.step()
            reduced = torch.cat([local, global_norm.detach().double().reshape(1)])
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
            duration = torch.tensor(
                time.monotonic() - step_started, dtype=torch.float64, device=device
            )
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            if rank == 0:
                record = compact_step_record(
                    step=step,
                    reduced=reduced.tolist(),
                    world_size=world_size,
                    duration_seconds=float(duration.item()),
                    host_ram_used_ratio=host_ratio,
                )
                _append_json(metrics_path, record)
                print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)
            if step in checkpoint_steps:
                final_gate_passed = checkpoint(step)
        if rank == 0:
            status = {
                "schema": "personaplex.native-moshirag-training-status.v1",
                "status": "teacher_forced_gate_passed" if final_gate_passed else "teacher_forced_gate_failed",
                "step": args.max_steps,
                "generatedDuplexGate": "pending",
                "liveCallGate": "pending",
            }
            (args.run_dir / "status.json").write_text(
                json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        dist.barrier()
        return 0 if final_gate_passed or not args.require_final_gate else 3
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Sequence[str] | None = None) -> int:
    actual_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(actual_argv)
    try:
        _validate_args(args)
        certified_pack = verify_certified_pack(
            args.certified_pack_manifest,
            data_contract_path=args.data_contract,
            group_manifest_path=args.group_manifest,
            model_contract_path=args.model_contract,
        )
        under_torchrun = all(name in os.environ for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
        if not args.distributed_worker and not under_torchrun:
            return launch_distributed(args, actual_argv)
        return run_worker(args, certified_pack)
    except TrainerContractError as exc:
        print(f"native PersonaPlex trainer refused to run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
