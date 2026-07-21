#!/usr/bin/env python3
"""Merge independently encoded ARC-4 reference roots without rewriting tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CORPUS_SCHEMA = "personaplex.arc4-reference-corpus.v1"
INDEX_SCHEMA = "personaplex.arc4-reference-index.v1"
MERGE_SCHEMA = "personaplex.arc4-reference-merge.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge disjoint ARC-4 precompute lanes into one certified root."
    )
    parser.add_argument(
        "--input-root",
        action="append",
        required=True,
        help="Completed precompute root. Repeat once per lane.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_sha256(value: Any) -> str:
    text = str(value or "")
    if text.startswith("sha256:"):
        text = text[7:]
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text.lower()):
        raise ValueError(f"invalid sha256 value: {value!r}")
    return text.lower()


def relative_shard(value: Any) -> Path:
    shard = Path(str(value or ""))
    if shard.is_absolute() or ".." in shard.parts or len(shard.parts) < 2:
        raise ValueError(f"unsafe shard path: {value!r}")
    if shard.parts[0] != "shards":
        raise ValueError(f"shard must be rooted under shards/: {value!r}")
    return shard


def conditioner_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    conditioner = manifest.get("conditioner")
    if not isinstance(conditioner, dict):
        raise ValueError("manifest is missing conditioner contract")
    return {
        "revision": conditioner.get("revision"),
        "packingRevision": conditioner.get("packingRevision"),
        "outputDim": conditioner.get("outputDim"),
        "frameRateHz": conditioner.get("frameRateHz"),
    }


def validate_manifest(root: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"unsupported manifest schema in {root}")
    if manifest.get("complete") is not True:
        raise ValueError(f"incomplete precompute lane: {root}")
    resources = manifest.get("resources") or {}
    if resources.get("cpuModelFallback") is not False:
        raise ValueError(f"CPU model fallback was not explicitly disabled: {root}")
    selection = manifest.get("selection") or {}
    selected = int(selection.get("selected", -1))
    encoded = int(selection.get("encoded", -2))
    failed = int(selection.get("failed", -1))
    if selected < 1 or selected != encoded or failed != 0:
        raise ValueError(
            f"lane is not lossless: {root} selected={selected} encoded={encoded} failed={failed}"
        )
    if selection.get("targetTextPassedToConditioner") is not False:
        raise ValueError(f"target-text leakage is not explicitly false: {root}")
    if selection.get("certifiedTargetTurnsOnly") is not True:
        raise ValueError(f"uncertified turns may be present: {root}")
    if selection.get("trainingEligibleOnly") is not True:
        raise ValueError(f"ineligible turns may be present: {root}")


def hardlink_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> int:
    args = parse_args()
    roots = sorted({Path(value).resolve() for value in args.input_root}, key=str)
    if len(roots) < 2:
        raise ValueError("at least two distinct --input-root values are required")

    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace existing output: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "shards").mkdir(parents=True)

    manifests: list[dict[str, Any]] = []
    expected_conditioner: dict[str, Any] | None = None
    expected_tensor_contract: tuple[str, tuple[int, ...]] | None = None
    seen_items: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    merged_sources: list[dict[str, Any]] = []
    lane_reports: list[dict[str, Any]] = []
    total_records = 0
    link_modes: set[str] = set()

    index_path = temporary / "index.jsonl"
    try:
        with index_path.open("w", encoding="utf-8") as output_index:
            for lane_number, root in enumerate(roots):
                manifest_path = root / "manifest.json"
                source_index = root / "index.jsonl"
                if not manifest_path.is_file() or not source_index.is_file():
                    raise FileNotFoundError(f"lane lacks manifest.json or index.jsonl: {root}")
                manifest = read_json(manifest_path)
                validate_manifest(root, manifest)
                manifests.append(manifest)

                contract = conditioner_contract(manifest)
                if expected_conditioner is None:
                    expected_conditioner = contract
                elif contract != expected_conditioner:
                    raise ValueError(
                        f"conditioner contract mismatch: expected={expected_conditioner} got={contract}"
                    )

                for source in manifest.get("sources") or []:
                    if not isinstance(source, dict):
                        raise ValueError(f"invalid source descriptor in {manifest_path}")
                    key = (str(source.get("path")), normalized_sha256(source.get("sha256")))
                    if key in seen_sources:
                        raise ValueError(f"duplicate source across lanes: {key[0]}")
                    seen_sources.add(key)
                    merged_sources.append(source)

                destination_lane = temporary / "shards" / f"lane-{lane_number:02d}"
                destination_lane.mkdir()
                verified_shards: dict[Path, str] = {}
                lane_records = 0

                with source_index.open("r", encoding="utf-8") as input_index:
                    for line_number, raw_line in enumerate(input_index, 1):
                        if not raw_line.strip():
                            continue
                        record = json.loads(raw_line)
                        if not isinstance(record, dict) or record.get("schema") != INDEX_SCHEMA:
                            raise ValueError(f"invalid index record: {source_index}:{line_number}")
                        if record.get("targetTextPassedToConditioner") is not False:
                            raise ValueError(f"target leakage: {source_index}:{line_number}")
                        if record.get("conditionerRevision") != contract["revision"]:
                            raise ValueError(f"conditioner revision mismatch: {source_index}:{line_number}")
                        if record.get("packingRevision") != contract["packingRevision"]:
                            raise ValueError(f"packing revision mismatch: {source_index}:{line_number}")

                        item_id = str(record.get("itemId") or "")
                        if not item_id or item_id in seen_items:
                            raise ValueError(f"missing or duplicate itemId: {source_index}:{line_number}")
                        seen_items.add(item_id)

                        shape = tuple(int(value) for value in record.get("tensorShape") or [])
                        tensor_contract = (str(record.get("tensorDtype")), shape)
                        if expected_tensor_contract is None:
                            expected_tensor_contract = tensor_contract
                        elif tensor_contract != expected_tensor_contract:
                            raise ValueError(
                                f"tensor contract mismatch: expected={expected_tensor_contract} got={tensor_contract}"
                            )
                        if not shape or shape[-1] != int(contract["outputDim"]):
                            raise ValueError(f"tensor width does not match conditioner: {source_index}:{line_number}")

                        source_shard_relative = relative_shard(record.get("shard"))
                        source_shard = (root / source_shard_relative).resolve()
                        if not source_shard.is_file() or root not in source_shard.parents:
                            raise FileNotFoundError(f"missing or escaped shard: {source_shard}")
                        declared_hash = normalized_sha256(record.get("shardSha256"))
                        actual_hash = verified_shards.get(source_shard)
                        if actual_hash is None:
                            actual_hash = sha256_file(source_shard)
                            if actual_hash != declared_hash:
                                raise ValueError(f"shard hash mismatch: {source_shard}")
                            destination_shard = destination_lane / source_shard.name
                            link_modes.add(hardlink_or_copy(source_shard, destination_shard))
                            verified_shards[source_shard] = actual_hash
                        elif actual_hash != declared_hash:
                            raise ValueError(f"inconsistent shard hash in index: {source_shard}")

                        record["shard"] = str(
                            Path("shards") / destination_lane.name / source_shard.name
                        )
                        output_index.write(
                            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                        )
                        lane_records += 1

                expected_records = int(manifest["selection"]["encoded"])
                if lane_records != expected_records:
                    raise ValueError(
                        f"lane index count mismatch: {root} expected={expected_records} got={lane_records}"
                    )
                total_records += lane_records
                lane_reports.append(
                    {
                        "root": str(root),
                        "manifestSha256": sha256_file(manifest_path),
                        "indexSha256": sha256_file(source_index),
                        "records": lane_records,
                        "shards": len(verified_shards),
                    }
                )
            output_index.flush()
            os.fsync(output_index.fileno())

        assert expected_conditioner is not None
        assert expected_tensor_contract is not None
        merged_selection = {
            "certifiedTargetTurnsOnly": True,
            "counterfactualGroups": sum(
                int(manifest["selection"].get("counterfactualGroups", 0))
                for manifest in manifests
            ),
            "encoded": total_records,
            "failed": 0,
            "resumed": sum(
                int(manifest["selection"].get("resumed", 0)) for manifest in manifests
            ),
            "selected": total_records,
            "targetTextPassedToConditioner": False,
            "trainingEligibleOnly": True,
        }
        merged_manifest = {
            "complete": True,
            "conditioner": {
                **expected_conditioner,
                "url": "merged://offline-field-slot-precompute",
            },
            "createdAt": datetime.now(timezone.utc).isoformat(),
            "elapsedSeconds": max(float(item.get("elapsedSeconds", 0.0)) for item in manifests),
            "merge": {
                "schema": MERGE_SCHEMA,
                "laneCount": len(roots),
                "lanes": lane_reports,
                "indexSha256": sha256_file(index_path),
                "linkModes": sorted(link_modes),
                "tensorContract": {
                    "dtype": expected_tensor_contract[0],
                    "shape": list(expected_tensor_contract[1]),
                },
            },
            "resources": {
                "cpuModelFallback": False,
                "hostMemoryLimitFraction": max(
                    float((item.get("resources") or {}).get("hostMemoryLimitFraction", 0.0))
                    for item in manifests
                ),
            },
            "schema": CORPUS_SCHEMA,
            "selection": merged_selection,
            "sources": sorted(merged_sources, key=lambda item: str(item.get("path"))),
        }
        manifest_output = temporary / "manifest.json"
        manifest_output.write_text(
            json.dumps(merged_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "complete": True,
                "output": str(output),
                "lanes": len(roots),
                "records": total_records,
                "sources": len(merged_sources),
                "conditioner": expected_conditioner,
                "tensorContract": {
                    "dtype": expected_tensor_contract[0],
                    "shape": list(expected_tensor_contract[1]),
                },
                "linkModes": sorted(link_modes),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
