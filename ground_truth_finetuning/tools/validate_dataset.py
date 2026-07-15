"""Validate manifest lineage, split isolation, plans, and encoded target masks."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ground_truth_finetuning.training.contracts import ContractError, validate_plan_mapping


REQUIRED_PROVENANCE = {"kind", "source_sha256", "license_id", "consent_id", "redaction_version"}


def _fail(errors: list[str], line: int, message: str) -> None:
    errors.append(f"line {line}: {message}")


def validate_manifest(path: Path) -> list[str]:
    errors: list[str] = []
    lineages: dict[str, set[str]] = defaultdict(set)
    with path.open() as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                item: dict[str, Any] = json.loads(raw)
            except json.JSONDecodeError as exc:
                _fail(errors, line_number, f"invalid JSON: {exc.msg}")
                continue
            for key in ("example_id", "dataset_version", "split", "conversation_lineage_id", "provenance", "semantics"):
                if key not in item:
                    _fail(errors, line_number, f"missing {key}")
            provenance = item.get("provenance", {})
            if not isinstance(provenance, dict):
                _fail(errors, line_number, "provenance must be an object")
            elif missing := REQUIRED_PROVENANCE.difference(provenance):
                _fail(errors, line_number, f"provenance missing {sorted(missing)}")
            lineage = item.get("conversation_lineage_id")
            split = item.get("split")
            if isinstance(lineage, str) and isinstance(split, str):
                lineages[lineage].add(split)
            semantics = item.get("semantics", {})
            plan = semantics.get("plan") if isinstance(semantics, dict) else None
            if not isinstance(plan, dict):
                _fail(errors, line_number, "semantics.plan must be an inline validated control plan")
            else:
                try:
                    validate_plan_mapping(plan)
                except ContractError as exc:
                    _fail(errors, line_number, str(exc))
            encoding = item.get("model_encoding")
            if encoding is not None:
                required = {"model_revision", "codebook_layout", "delay_config_sha256", "codes_sha256", "text_alignment_sha256", "target_mask_sha256"}
                if not isinstance(encoding, dict) or required.difference(encoding):
                    _fail(errors, line_number, "encoded item is missing model layout/alignment/mask evidence")
    for lineage, splits in lineages.items():
        if len(splits) > 1:
            errors.append(f"lineage {lineage!r} crosses immutable splits: {sorted(splits)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    errors = validate_manifest(args.manifest)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
