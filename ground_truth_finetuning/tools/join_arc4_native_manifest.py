#!/usr/bin/env python3
"""Join certified native targets to precomputed ARC-4 streams by frame hash."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personaplex_control.arc4_packing import (
    ARC4_PACKING_REVISION,
    ARC4_SUPPORTED_PACKING_REVISIONS,
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty JSONL objects: {path}")
    return rows


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def select_arc4(
    native: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(candidates) == 1:
        return candidates[0]
    evidence = native.get("evidence")
    evidence_hash = evidence.get("frame_hash") if isinstance(evidence, Mapping) else None
    narrowed = [row for row in candidates if row.get("evidenceHash") == evidence_hash]
    if len(narrowed) == 1:
        return narrowed[0]
    raise ValueError(
        f"ambiguous ARC-4 join for frame {native.get('control', {}).get('frame_hash')}: "
        f"{len(candidates)} candidates, {len(narrowed)} evidence matches"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--native-certificate", type=Path, required=True)
    parser.add_argument("--arc4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--packing-revision",
        choices=ARC4_SUPPORTED_PACKING_REVISIONS,
        default=ARC4_PACKING_REVISION,
    )
    args = parser.parse_args()

    native_manifest = args.native_manifest.resolve()
    native_certificate_path = args.native_certificate.resolve()
    arc4_root = args.arc4_root.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise SystemExit(f"refusing existing joined output: {output}")

    native_certificate = load_json(native_certificate_path)
    if native_certificate.get("status") != "certified_for_adapter_training":
        raise SystemExit("native certificate does not authorize adapter training")
    if native_certificate.get("manifest_sha256") != hash_file(native_manifest):
        raise SystemExit("native certificate does not match native manifest")
    arc_manifest_path = arc4_root / "manifest.json"
    arc_index_path = arc4_root / "index.jsonl"
    arc_manifest = load_json(arc_manifest_path)
    if arc_manifest.get("complete") is not True:
        raise SystemExit("ARC-4 corpus is incomplete")
    if arc_manifest.get("selection", {}).get("targetTextPassedToConditioner") is not False:
        raise SystemExit("ARC-4 corpus does not certify target-label separation")
    if arc_manifest.get("conditioner", {}).get("packingRevision") != args.packing_revision:
        raise SystemExit("ARC-4 corpus packing revision is stale or missing")

    native_rows = load_jsonl(native_manifest)
    arc_rows = load_jsonl(arc_index_path)
    by_frame: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in arc_rows:
        if row.get("conditionerRevision") != arc_manifest.get("conditioner", {}).get("revision"):
            raise SystemExit("ARC-4 index mixes conditioner revisions")
        if row.get("packingRevision") != args.packing_revision:
            raise SystemExit("ARC-4 index mixes or omits packing revisions")
        if row.get("targetTextPassedToConditioner") is not False:
            raise SystemExit("ARC-4 index row lacks target-label separation")
        frame_hash = row.get("frameHash")
        if not isinstance(frame_hash, str) or not frame_hash:
            raise SystemExit("ARC-4 index row lacks frameHash")
        by_frame[frame_hash].append(row)

    shard_hashes: dict[Path, str] = {}
    shard_keys: dict[Path, set[str]] = {}
    joined: list[dict[str, Any]] = []
    missing: list[str] = []
    splits: collections.Counter[str] = collections.Counter()
    groups: dict[str, set[str]] = collections.defaultdict(set)
    for native in native_rows:
        control = native.get("control")
        frame_hash = control.get("frame_hash") if isinstance(control, Mapping) else None
        candidates = by_frame.get(str(frame_hash), [])
        if not candidates:
            missing.append(str(native.get("example_id")))
            continue
        arc = select_arc4(native, candidates)
        shard = (arc4_root / str(arc["shard"])).resolve()
        if arc4_root not in shard.parents or not shard.is_file():
            raise SystemExit(f"ARC-4 shard escapes root or is missing: {shard}")
        expected_hash = "sha256:" + str(arc["shardSha256"]).removeprefix("sha256:")
        if shard not in shard_hashes:
            shard_hashes[shard] = hash_file(shard)
        actual_hash = shard_hashes[shard]
        if actual_hash != expected_hash:
            raise SystemExit(f"ARC-4 shard hash mismatch: {shard}")
        if shard not in shard_keys:
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                shard_keys[shard] = set(handle.keys())
        if arc["tensorKey"] not in shard_keys[shard]:
            raise SystemExit(f"ARC-4 tensor key is absent: {arc['tensorKey']}")
        value = dict(native)
        value["arc4_reference"] = {
            "schema": "personaplex.arc4-reference-binding.v1",
            "conditioner_revision": arc["conditionerRevision"],
            "packing_revision": arc["packingRevision"],
            "reference_hash": arc["referenceHash"],
            "frame_hash": arc["frameHash"],
            "evidence_hash": arc.get("evidenceHash"),
            "shard_path": str(shard.relative_to(arc4_root)),
            "shard_sha256": actual_hash,
            "tensor_key": arc["tensorKey"],
            "tensor_shape": arc["tensorShape"],
            "tensor_dtype": arc["tensorDtype"],
            "target_text_passed_to_conditioner": False,
        }
        joined.append(value)
        splits[str(native.get("split"))] += 1
        counterfactual = native.get("counterfactual")
        if isinstance(counterfactual, Mapping) and counterfactual.get("groupId"):
            groups[str(counterfactual["groupId"])].add(str(counterfactual.get("branchId")))

    if missing:
        raise SystemExit(
            f"ARC-4 join coverage is incomplete: {len(missing)}/{len(native_rows)} missing; "
            f"first={missing[:8]}"
        )
    if len(joined) != len(native_rows) or len({row["example_id"] for row in joined}) != len(joined):
        raise SystemExit("joined examples are missing or duplicated")
    if set(splits) != {"train", "validation", "test"} or min(splits.values()) < 1:
        raise SystemExit(f"joined split coverage is invalid: {dict(splits)}")

    output.mkdir(parents=True)
    joined_manifest = output / "arc4_native_examples.jsonl"
    with joined_manifest.open("w", encoding="utf-8") as handle:
        for row in joined:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    certificate = {
        "schema": "personaplex.arc4-native-certificate.v1",
        "kind": "personaplex-arc4-native-corpus-certificate",
        "status": "certified_for_arc4_adapter_training",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "joinedManifest": str(joined_manifest),
        "joinedManifestSha256": hash_file(joined_manifest),
        "nativeManifest": str(native_manifest),
        "nativeManifestSha256": hash_file(native_manifest),
        "nativeCertificate": str(native_certificate_path),
        "nativeCertificateSha256": hash_file(native_certificate_path),
        "arc4Manifest": str(arc_manifest_path),
        "arc4ManifestSha256": hash_file(arc_manifest_path),
        "arc4Index": str(arc_index_path),
        "arc4IndexSha256": hash_file(arc_index_path),
        "arc4Root": str(arc4_root),
        "conditionerRevision": arc_manifest["conditioner"]["revision"],
        "packingRevision": args.packing_revision,
        "examples": len(joined),
        "splits": dict(sorted(splits.items())),
        "counterfactualGroups": len(groups),
        "multiBranchCounterfactualGroups": sum(len(branches) >= 2 for branches in groups.values()),
        "joinKey": "control.frame_hash",
        "targetTextPassedToConditioner": False,
        "callerStreamSupervision": "forbidden",
        "verifiedShards": len(shard_hashes),
    }
    certificate_path = output / "certificate.json"
    atomic_json(certificate_path, certificate)
    print(
        json.dumps(
            {
                "status": certificate["status"],
                "examples": len(joined),
                "splits": dict(splits),
                "counterfactualGroups": len(groups),
                "certificate": str(certificate_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
