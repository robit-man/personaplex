#!/usr/bin/env python3
"""Merge complete, disjoint native tensor shards into one certified input manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ground_truth_finetuning.tools.encode_native_adapter_tensors import hash_file


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if args.shard_count < 1:
        raise SystemExit("shard-count must be positive")
    source_rows = read_jsonl(args.source_manifest.resolve())
    source_ids = {row.get("example_id") for row in source_rows}
    if None in source_ids or len(source_ids) != len(source_rows):
        raise SystemExit("source manifest must contain unique example_id values")
    artifact_root = args.artifact_root.resolve()
    encoded_rows: list[dict] = []
    for index in range(args.shard_count):
        shard = artifact_root / f"encoded_examples.shard-{index:02d}-of-{args.shard_count:02d}.jsonl"
        if not shard.is_file():
            raise SystemExit(f"missing tensor shard: {shard}")
        encoded_rows.extend(read_jsonl(shard))
    encoded_ids = [row.get("example_id") for row in encoded_rows]
    if None in encoded_ids or len(encoded_ids) != len(set(encoded_ids)):
        raise SystemExit("tensor shards contain a missing or duplicate example_id")
    if set(encoded_ids) != source_ids:
        missing = len(source_ids.difference(encoded_ids))
        foreign = len(set(encoded_ids).difference(source_ids))
        raise SystemExit(f"tensor shards do not match source manifest: missing={missing}, foreign={foreign}")
    merged = artifact_root / "encoded_examples.jsonl"
    merged.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sorted(encoded_rows, key=lambda row: row["example_id"])),
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "kind": "personaplex-controlled-native-tensor-shard-merge",
        "status": "encoded_pending_tensor_certification",
        "items": len(encoded_rows),
        "shard_count": args.shard_count,
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": hash_file(args.source_manifest.resolve()),
        "manifest": str(merged),
        "manifest_sha256": hash_file(merged),
    }
    (artifact_root / "encoding_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
