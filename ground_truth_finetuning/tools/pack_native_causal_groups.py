#!/usr/bin/env python3
"""Pack v5 four-sibling artifacts into leakage-safe native training indexes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.causal_group_pack import (
    CausalGroupPackError,
    PackConfig,
    load_group_artifacts,
    pack_causal_groups,
    prepare_trainer_binding,
    write_pack_result,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Grouped v5 JSON/JSONL artifact; repeat for multiple immutable inputs.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trainer-data-contract", required=True, type=Path)
    parser.add_argument("--trainer-group-manifest", required=True, type=Path)
    parser.add_argument("--model-contract", required=True, type=Path)
    parser.add_argument("--split-seed", default="personaplex-native-causal-groups-v1")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--minimum-distinct-premises", type=int, default=2)
    parser.add_argument(
        "--required-coverage-splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation", "test"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = PackConfig(
        split_ratios=(
            ("train", args.train_ratio),
            ("validation", args.validation_ratio),
            ("test", args.test_ratio),
        ),
        split_seed=args.split_seed,
        required_coverage_splits=tuple(args.required_coverage_splits),
        minimum_distinct_premises=args.minimum_distinct_premises,
    )
    try:
        artifacts, inputs = load_group_artifacts(args.input)
        result = pack_causal_groups(artifacts, config)
        trainer_binding = prepare_trainer_binding(
            result,
            inputs,
            trainer_data_contract=args.trainer_data_contract,
            trainer_group_manifest=args.trainer_group_manifest,
            model_contract=args.model_contract,
        )
        manifest = write_pack_result(
            args.output_dir, result, inputs, config, trainer_binding
        )
    except CausalGroupPackError as error:
        print(json.dumps({"status": "rejected", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifestId": manifest["manifestId"],
                **manifest["counts"],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["status"] == "certified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
