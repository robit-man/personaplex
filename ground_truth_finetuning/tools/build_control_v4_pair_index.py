#!/usr/bin/env python3
"""Build and certify exact-shared-prefix causal pairs from a native manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.counterfactual_pairs import build_causal_pairs


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--skip-prefix-tensor-verification", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    pairs, certificate = build_causal_pairs(
        records,
        artifact_root=args.artifact_root.resolve(),
        verify_prefix_tensors=not args.skip_prefix_tensor_verification,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(pair.as_dict(), sort_keys=True) + "\n" for pair in pairs),
        encoding="utf-8",
    )
    certificate.update(
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest),
            "manifest_sha256": hash_file(manifest),
            "pair_index": str(args.output.resolve()),
            "pair_index_sha256": hash_file(args.output.resolve()),
        }
    )
    args.certificate.parent.mkdir(parents=True, exist_ok=True)
    args.certificate.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: certificate[key] for key in ("status", "pairs", "pairs_by_split")}, sort_keys=True))
    return 0 if certificate["status"] in {
        "certified_for_causal_control_training",
        "candidate_for_prefix_canonicalization",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
