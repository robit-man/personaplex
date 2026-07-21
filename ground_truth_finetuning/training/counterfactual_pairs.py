"""Certified same-prefix causal pair indexing for semantic-control training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .contracts import ControlTrainingFrame, validate_control_frame_mapping


@dataclass(frozen=True)
class PairMember:
    example_id: str
    manifest_index: int
    branch_id: str
    state_hash: str
    frame_hash: str
    stale_example_id: str | None
    prefix_at: int


@dataclass(frozen=True)
class CausalPair:
    pair_id: str
    split: str
    group_id: str
    pivot_target_ordinal: int
    target_turn_id: int
    base_state_hash: str
    prefix_at: int
    shared_prefix_sha256: str
    changed_paths: tuple[str, ...]
    member_a: PairMember
    member_b: PairMember

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema"] = "personaplex.causal-control-pair.v4"
        value["changed_paths"] = list(self.changed_paths)
        return value


def _load_tensor(path: Path, name: str) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    tensor = value.get(name) if isinstance(value, dict) else value
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{path} does not contain tensor {name!r}")
    return tensor


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return f"sha256:{digest.hexdigest()}"


def _changed_paths(left: Any, right: Any, path: str = "") -> list[str]:
    if type(left) is not type(right):
        return [path or "root"]
    if isinstance(left, dict):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(_changed_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list):
        if left == right:
            return []
        return [path or "root"]
    return [] if left == right else [path or "root"]


def _semantic_view(frame: ControlTrainingFrame) -> dict[str, Any]:
    value = frame.as_wire_dict()
    for key in ("frameId", "conversationId", "stateHash"):
        value.pop(key, None)
    plan = value.get("plan", {})
    for key in ("callId", "contextHash"):
        plan.pop(key, None)
    return value


def build_causal_pairs(
    records: list[dict[str, Any]],
    *,
    artifact_root: Path,
    verify_prefix_tensors: bool = True,
) -> tuple[list[CausalPair], dict[str, Any]]:
    indexed: list[tuple[int, dict[str, Any], ControlTrainingFrame]] = []
    group_splits: dict[str, set[str]] = {}
    by_conversation: dict[str, list[tuple[int, dict[str, Any], ControlTrainingFrame]]] = {}
    for index, row in enumerate(records):
        counterfactual = row.get("counterfactual")
        raw_frame = row.get("control", {}).get("frame")
        if not isinstance(counterfactual, dict) or not isinstance(raw_frame, dict):
            continue
        frame = validate_control_frame_mapping(raw_frame)
        indexed.append((index, row, frame))
        group_id = str(counterfactual.get("groupId", ""))
        group_splits.setdefault(group_id, set()).add(str(row.get("split", "")))
        by_conversation.setdefault(frame.conversation_id, []).append((index, row, frame))
    leaked = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) != 1}
    if leaked:
        raise ValueError(f"counterfactual groups cross dataset splits: {leaked}")
    stale_by_id: dict[str, str | None] = {}
    for conversation_rows in by_conversation.values():
        ordered = sorted(conversation_rows, key=lambda item: item[2].state_revision)
        prior: str | None = None
        for _, row, _ in ordered:
            example_id = str(row.get("example_id"))
            stale_by_id[example_id] = prior
            prior = example_id
    grouped: dict[tuple[str, int, int, str], list[tuple[int, dict[str, Any], ControlTrainingFrame]]] = {}
    for item in indexed:
        _, row, frame = item
        cf = row["counterfactual"]
        key = (
            str(cf.get("groupId", "")),
            int(cf.get("pivotTargetOrdinal", -1)),
            frame.target_turn_id,
            frame.base_state_hash,
        )
        grouped.setdefault(key, []).append(item)
    pairs: list[CausalPair] = []
    rejection_counts: dict[str, int] = {}

    def reject(reason: str) -> None:
        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

    for key, candidates in sorted(grouped.items()):
        for left, right in itertools.combinations(candidates, 2):
            left_index, left_row, left_frame = left
            right_index, right_row, right_frame = right
            left_cf = left_row["counterfactual"]
            right_cf = right_row["counterfactual"]
            if left_cf.get("branchId") == right_cf.get("branchId"):
                reject("same_branch")
                continue
            if left_row.get("split") != right_row.get("split"):
                raise ValueError(f"pair {key[0]} crosses splits")
            if left_frame.state_hash == right_frame.state_hash:
                reject("state_not_divergent")
                continue
            left_encoding = left_row.get("model_encoding", {})
            right_encoding = right_row.get("model_encoding", {})
            left_at = int(left_encoding.get("prefix_at", -1))
            right_at = int(right_encoding.get("prefix_at", -1))
            if left_at < 1 or right_at < 1:
                reject("invalid_prefix_position")
                continue
            if verify_prefix_tensors and left_at != right_at:
                reject("prefix_position_mismatch")
                continue
            prefix_hash = "not-verified"
            if verify_prefix_tensors:
                left_codes = _load_tensor(artifact_root / left_encoding["codes_path"], "codes")
                right_codes = _load_tensor(artifact_root / right_encoding["codes_path"], "codes")
                if left_codes.ndim != 2 or right_codes.ndim != 2:
                    reject("invalid_code_shape")
                    continue
                left_prefix = left_codes[:, :left_at]
                right_prefix = right_codes[:, :right_at]
                if left_prefix.shape != right_prefix.shape or not torch.equal(left_prefix, right_prefix):
                    reject("native_prefix_not_identical")
                    continue
                prefix_hash = _tensor_hash(left_prefix)
            changes = tuple(
                path
                for path in _changed_paths(_semantic_view(left_frame), _semantic_view(right_frame))
                if path not in {"stateRevision", "plan.revision"}
            )
            if not changes:
                reject("no_semantic_change")
                continue
            left_id = str(left_row["example_id"])
            right_id = str(right_row["example_id"])
            identity = {
                "group": key[0],
                "pivot": key[1],
                "turn": key[2],
                "base": key[3],
                "members": sorted([left_id, right_id]),
            }
            pair_id = "sha256:" + sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            members = sorted(
                [
                    PairMember(
                        left_id,
                        left_index,
                        str(left_cf["branchId"]),
                        left_frame.state_hash,
                        left_frame.frame_hash,
                        stale_by_id.get(left_id),
                        left_at,
                    ),
                    PairMember(
                        right_id,
                        right_index,
                        str(right_cf["branchId"]),
                        right_frame.state_hash,
                        right_frame.frame_hash,
                        stale_by_id.get(right_id),
                        right_at,
                    ),
                ],
                key=lambda member: member.branch_id,
            )
            pairs.append(
                CausalPair(
                    pair_id=pair_id,
                    split=str(left_row["split"]),
                    group_id=key[0],
                    pivot_target_ordinal=key[1],
                    target_turn_id=key[2],
                    base_state_hash=key[3],
                    prefix_at=left_at if left_at == right_at else -1,
                    shared_prefix_sha256=prefix_hash,
                    changed_paths=changes,
                    member_a=members[0],
                    member_b=members[1],
                )
            )
    pair_ids = [pair.pair_id for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("causal pair index contains duplicate pair IDs")
    report = {
        "schema_version": 1,
        "kind": "personaplex-causal-control-pair-index-certificate",
        "status": (
            "certified_for_causal_control_training"
            if pairs and verify_prefix_tensors
            else "candidate_for_prefix_canonicalization"
            if pairs
            else "failed"
        ),
        "source_records": len(records),
        "counterfactual_groups": len(group_splits),
        "pairs": len(pairs),
        "pairs_by_split": {
            split: sum(pair.split == split for pair in pairs)
            for split in ("train", "validation", "test")
        },
        "shared_prefix_tensor_verification": verify_prefix_tensors,
        "split_leakage_groups": 0,
        "rejections": rejection_counts,
    }
    return pairs, report
