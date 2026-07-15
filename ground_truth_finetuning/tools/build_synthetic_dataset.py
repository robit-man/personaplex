"""Build schema-valid synthetic control conversations with measured timing distributions.

This tool writes plans and canonical labels only. It never pretends synthetic text is
native audio training data; audio encoding occurs later through a validated exporter.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random
from statistics import median
from typing import Any

from ground_truth_finetuning.training.contracts import canonical_json, sha256_uri, validate_plan_mapping


TIMING_FIELDS = (
    "caller_end_to_plan_ready_ms",
    "plan_ready_to_boundary_ms",
    "boundary_to_first_audio_ms",
    "response_duration_ms",
)


def _fit_lognormal(values: list[float]) -> tuple[float, float]:
    if len(values) < 20 or any(value <= 0 for value in values):
        raise ValueError("each timing field needs at least 20 positive sanitized samples")
    med = median(values)
    p95 = sorted(values)[max(0, math.ceil(len(values) * 0.95) - 1)]
    sigma = max(0.01, math.log(p95 / med) / 1.6448536269514722)
    return math.log(med), sigma


def _load_timing_models(path: Path) -> dict[str, tuple[float, float]]:
    samples: dict[str, list[float]] = {field: [] for field in TIMING_FIELDS}
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        item = json.loads(raw)
        for field in TIMING_FIELDS:
            value = item.get(field)
            if isinstance(value, (int, float)):
                samples[field].append(float(value))
    return {field: _fit_lognormal(values) for field, values in samples.items()}


def _sample_timing(models: dict[str, tuple[float, float]], rng: random.Random) -> dict[str, int]:
    return {field: max(1, round(rng.lognormvariate(mu, sigma))) for field, (mu, sigma) in models.items()}


def _template_record(template: dict[str, Any], index: int, timing: dict[str, int]) -> dict[str, Any]:
    plan = dict(template["plan"])
    plan["callId"] = f"synthetic-{index:08d}"
    plan["turnId"] = 1
    plan["revision"] = 1
    plan["contextHash"] = sha256_uri({"template": template["id"], "index": index, "caller": template["caller"]})
    plan = validate_plan_mapping(plan).as_wire_dict()
    canonical = template["canonical_text"]
    lineage = f"synthetic:{template['id']}:{index}"
    item = {
        "example_id": sha256_uri({"lineage": lineage, "plan": plan, "canonical": canonical}),
        "dataset_version": "gtft-synthetic-v1",
        "split": "train",
        "conversation_lineage_id": lineage,
        "turn_index": 1,
        "provenance": {
            "kind": "synthetic_controlled",
            "source_sha256": "sha256:" + sha256(canonical_json(template).encode()).hexdigest(),
            "license_id": "internal-synthetic-template-v1",
            "consent_id": "not_applicable_synthetic",
            "redaction_version": "synthetic-no-pii-v1",
        },
        "caller_text": template["caller"],
        "semantics": {
            "plan": plan,
            "canonical_response": canonical,
            "canonical_response_sha256": sha256_uri(canonical),
            "annotation_status": "template_reviewed",
        },
        "timing": timing,
        "synthetic": True,
    }
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, required=True)
    parser.add_argument("--timing-reference", type=Path, required=True, help="sanitized JSONL with timing fields only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    templates = json.loads(args.templates.read_text())
    if not isinstance(templates, list) or not templates:
        raise SystemExit("templates must be a non-empty JSON list")
    models = _load_timing_models(args.timing_reference)
    rng = random.Random(args.seed)
    records = [_template_record(rng.choice(templates), index, _sample_timing(models, rng)) for index in range(args.count)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(canonical_json(record) + "\n" for record in records))
    print(json.dumps({"output": str(args.output), "records": len(records), "timing_source": str(args.timing_reference)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
