#!/usr/bin/env python3
"""Precompute target-free ARC-4 streams for certified PersonaPlex target turns."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ground_truth_finetuning.training.contracts import (
    ContractError,
    EvidenceTrainingFrame,
    validate_control_frame_mapping,
    validate_evidence_frame_mapping,
)
from personaplex_control.moshirag_reference import (
    Arc4ConditionerClient,
    Arc4ConditionerError,
    render_arc4_reference,
    render_arc4_reference_fields,
)
from personaplex_control.arc4_packing import (
    ARC4_PACKING_REVISION,
    ARC4_SUPPORTED_PACKING_REVISIONS,
)


@dataclass(frozen=True)
class WorkItem:
    item_id: str
    source_path: Path
    source_line: int
    row: Mapping[str, Any]
    reference_text: str
    reference_fields: Mapping[str, str]
    reference_hash: str


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _host_memory_used_fraction() -> float:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="ascii") as handle:
        for line in handle:
            name, raw, *_rest = line.split()
            values[name.rstrip(":")] = int(raw)
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    if total <= 0:
        raise RuntimeError("cannot discover host memory from /proc/meminfo")
    return 1.0 - (available / total)


def _wait_for_host_memory(limit: float, poll_seconds: float) -> None:
    while _host_memory_used_fraction() >= limit:
        time.sleep(poll_seconds)


def _resolve_sources(patterns: list[str]) -> list[Path]:
    paths = sorted({Path(value) for pattern in patterns for value in glob.glob(pattern)})
    paths = [path for path in paths if path.is_file() and path.stat().st_size > 0]
    if not paths:
        raise SystemExit("no non-empty certified JSONL sources matched")
    return paths


def _extract_evidence(raw: Any) -> EvidenceTrainingFrame | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ContractError("control.evidence must be an object or null")
    candidate = raw.get("frame", raw)
    if not isinstance(candidate, Mapping):
        raise ContractError("control.evidence.frame must be an object")
    return validate_evidence_frame_mapping(candidate)


def _iter_work(sources: list[Path]) -> Iterator[WorkItem]:
    seen: set[str] = set()
    for source in sources:
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("speaker") not in {"target", "agent"}:
                    continue
                if row.get("quality", {}).get("accepted") is not True:
                    continue
                if row.get("training", {}).get("eligible") is not True:
                    continue
                control = row.get("control")
                if not isinstance(control, Mapping) or not isinstance(control.get("frame"), Mapping):
                    raise ContractError(
                        f"eligible target lacks typed control.frame at {source}:{line_number}"
                    )
                frame = validate_control_frame_mapping(control["frame"])
                evidence = _extract_evidence(control.get("evidence"))
                reference = render_arc4_reference(frame, evidence)
                reference_fields = render_arc4_reference_fields(frame, evidence)
                reference_hash = "sha256:" + hashlib.sha256(reference.encode("utf-8")).hexdigest()
                conversation_id = str(row.get("conversationId", ""))
                turn_index = int(row.get("turnIndex", -1))
                audio_hash = str(row.get("audioSha256", ""))
                item_id = "sha256:" + hashlib.sha256(
                    f"{conversation_id}\0{turn_index}\0{audio_hash}\0{reference_hash}".encode("utf-8")
                ).hexdigest()
                if item_id in seen:
                    continue
                seen.add(item_id)
                yield WorkItem(
                    item_id=item_id,
                    source_path=source,
                    source_line=line_number,
                    row=row,
                    reference_text=reference,
                    reference_fields=reference_fields,
                    reference_hash=reference_hash,
                )


def _load_completed(index_path: Path) -> set[str]:
    if not index_path.exists():
        return set()
    completed: set[str] = set()
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(str(json.loads(line)["itemId"]))
    return completed


def _encode_with_retry(
    client: Arc4ConditionerClient,
    fields: Mapping[str, str],
    attempts: int,
    max_frames: int,
) -> torch.Tensor:
    for attempt in range(1, attempts + 1):
        try:
            return client.encode_fields(fields, max_frames=max_frames)
        except Arc4ConditionerError:
            if attempt == attempts:
                raise
            time.sleep(min(2.0, 0.2 * (2 ** (attempt - 1))))
    raise AssertionError("unreachable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glob", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditioner-url", required=True)
    parser.add_argument(
        "--packing-revision",
        choices=ARC4_SUPPORTED_PACKING_REVISIONS,
        default=ARC4_PACKING_REVISION,
    )
    parser.add_argument("--conditioner-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--request-attempts", type=int, default=3)
    parser.add_argument("--shard-items", type=int, default=256)
    parser.add_argument("--max-reference-frames", type=int, default=64)
    parser.add_argument("--max-host-memory-fraction", type=float, default=0.80)
    parser.add_argument("--memory-poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.max_host_memory_fraction < 1.0:
        raise SystemExit("--max-host-memory-fraction must be between zero and one")
    if args.shard_items < 1 or args.max_reference_frames < 1 or args.request_attempts < 1:
        raise SystemExit("shard, frame, and attempt limits must be positive")

    sources = _resolve_sources(args.input_glob)
    output = args.output_dir
    if output.exists() and not args.resume:
        raise SystemExit(f"output exists; use --resume or a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / "shards"
    shard_dir.mkdir(exist_ok=True)
    index_path = output / "index.jsonl"
    failure_path = output / "failures.jsonl"
    completed = _load_completed(index_path) if args.resume else set()
    client = Arc4ConditionerClient(
        args.conditioner_url,
        timeout_seconds=args.conditioner_timeout_seconds,
        expected_packing_revision=args.packing_revision,
    )

    existing_shards = sorted(shard_dir.glob("arc4-*.safetensors"))
    shard_number = len(existing_shards)
    shard_tensors: dict[str, torch.Tensor] = {}
    shard_records: list[dict[str, Any]] = []
    selected = encoded_count = skipped_count = failure_count = 0
    counterfactual_groups: set[str] = set()
    started = time.time()

    def flush() -> None:
        nonlocal shard_number, shard_tensors, shard_records
        if not shard_tensors:
            return
        shard_name = f"arc4-{shard_number:05d}.safetensors"
        final_path = shard_dir / shard_name
        temporary = final_path.with_suffix(".safetensors.tmp")
        save_file(shard_tensors, temporary)
        os.replace(temporary, final_path)
        shard_hash = _sha256_file(final_path)
        with index_path.open("a", encoding="utf-8") as index:
            for record in shard_records:
                record["shard"] = str(Path("shards") / shard_name)
                record["shardSha256"] = shard_hash
                index.write(json.dumps(record, sort_keys=True) + "\n")
            index.flush()
            os.fsync(index.fileno())
        shard_number += 1
        shard_tensors = {}
        shard_records = []

    try:
        for item in _iter_work(sources):
            if args.max_items is not None and selected >= args.max_items:
                break
            selected += 1
            if item.item_id in completed:
                skipped_count += 1
                continue
            _wait_for_host_memory(args.max_host_memory_fraction, args.memory_poll_seconds)
            try:
                tensor = _encode_with_retry(
                    client,
                    item.reference_fields,
                    args.request_attempts,
                    args.max_reference_frames,
                )[:, : args.max_reference_frames]
                if tensor.shape[1] < 1 or tensor.shape[-1] != client.output_dim:
                    raise Arc4ConditionerError(f"invalid bounded ARC tensor {tuple(tensor.shape)}")
                tensor = tensor.squeeze(0).to(dtype=torch.bfloat16, device="cpu").contiguous()
                tensor_key = f"reference_{encoded_count:08d}"
                row = item.row
                target_text_hash = "sha256:" + hashlib.sha256(
                    str(row.get("text", "")).encode("utf-8")
                ).hexdigest()
                record = {
                    "schema": "personaplex.arc4-reference-index.v1",
                    "itemId": item.item_id,
                    "conversationId": row.get("conversationId"),
                    "turnIndex": row.get("turnIndex"),
                    "counterfactualGroupId": row.get("counterfactualGroupId"),
                    "counterfactualBranchId": row.get("counterfactualBranchId"),
                    "sourcePath": str(item.source_path),
                    "sourceLine": item.source_line,
                    "audioPath": row.get("audioPath"),
                    "audioSha256": row.get("audioSha256"),
                    "duplexTimelinePath": row.get("duplexTimelinePath"),
                    "frameHash": row["control"].get("frameHash"),
                    "evidenceHash": row["control"].get("evidenceHash"),
                    "planHash": row["control"].get("planHash"),
                    "referenceHash": item.reference_hash,
                    "conditionerRevision": client.revision,
                    "packingRevision": client.packing_revision,
                    "targetTextSha256": target_text_hash,
                    "targetTextPassedToConditioner": False,
                    "tensorKey": tensor_key,
                    "tensorShape": list(tensor.shape),
                    "tensorDtype": str(tensor.dtype).removeprefix("torch."),
                    "timing": row.get("timing"),
                }
                group = row.get("counterfactualGroupId")
                if group:
                    counterfactual_groups.add(str(group))
                shard_tensors[tensor_key] = tensor
                shard_records.append(record)
                encoded_count += 1
                if len(shard_tensors) >= args.shard_items:
                    flush()
            except (Arc4ConditionerError, ContractError, RuntimeError, ValueError) as exc:
                failure_count += 1
                with failure_path.open("a", encoding="utf-8") as failures:
                    failures.write(
                        json.dumps(
                            {
                                "itemId": item.item_id,
                                "sourcePath": str(item.source_path),
                                "sourceLine": item.source_line,
                                "errorType": type(exc).__name__,
                                "error": str(exc),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
        flush()
    except BaseException:
        flush()
        raise

    source_records = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sources
    ]
    manifest = {
        "schema": "personaplex.arc4-reference-corpus.v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "complete": failure_count == 0,
        "conditioner": {
            "url": args.conditioner_url,
            "revision": client.revision,
            "packingRevision": client.packing_revision,
            "outputDim": client.output_dim,
            "frameRateHz": client.frame_rate_hz,
        },
        "selection": {
            "certifiedTargetTurnsOnly": True,
            "trainingEligibleOnly": True,
            "targetTextPassedToConditioner": False,
            "selected": selected,
            "encoded": encoded_count,
            "resumed": skipped_count,
            "failed": failure_count,
            "counterfactualGroups": len(counterfactual_groups),
        },
        "resources": {
            "hostMemoryLimitFraction": args.max_host_memory_fraction,
            "cpuModelFallback": False,
        },
        "elapsedSeconds": time.time() - started,
        "sources": source_records,
    }
    _atomic_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "manifest": str(output / "manifest.json"),
                "complete": manifest["complete"],
                "conditionerRevision": client.revision,
                "selection": manifest["selection"],
                "elapsedSeconds": manifest["elapsedSeconds"],
                "sourceFiles": len(source_records),
            },
            indent=2,
        )
    )
    if failure_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
