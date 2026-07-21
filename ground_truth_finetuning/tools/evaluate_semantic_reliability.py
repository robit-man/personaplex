#!/usr/bin/env python3
"""Judge generated audio transcripts and apply the v4 statistical release gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.evaluation.reliability import evaluate_release_gate
from ground_truth_finetuning.evaluation.semantic_judge import (
    SemanticJudgeError,
    TypedSemanticJudge,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adjudicated-trials", type=Path)
    parser.add_argument("--judge-endpoint")
    parser.add_argument("--judge-model")
    parser.add_argument("--minimum-trials", type=int, default=1000)
    parser.add_argument("--minimum-pairs", type=int, default=250)
    parser.add_argument("--minimum-slice-trials", type=int, default=20)
    args = parser.parse_args()
    trials = load_jsonl(args.trials.resolve())
    judge = None
    if args.judge_endpoint or args.judge_model:
        if not args.judge_endpoint or not args.judge_model:
            raise SystemExit("judge-endpoint and judge-model are required together")
        judge = TypedSemanticJudge(endpoint=args.judge_endpoint, model=args.judge_model)
    for trial in trials:
        if isinstance(trial.get("judgment"), dict) and trial["judgment"].get("status") == "ok":
            continue
        if judge is None:
            trial["judgment"] = {
                "status": "failed",
                "reason": "typed_semantic_judge_not_configured",
            }
            continue
        try:
            trial["judgment"] = judge.judge_turn(
                control_frame=trial["control_frame"],
                asr_transcript=str(trial["asr_transcript"]),
                runtime_events=trial.get("runtime_events", {}),
            ).as_dict()
        except (SemanticJudgeError, KeyError, TypeError, ValueError) as exc:
            trial["judgment"] = {"status": "failed", "reason": str(exc)}
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        if trial.get("pair_id"):
            by_pair.setdefault(str(trial["pair_id"]), []).append(trial)
    for members in by_pair.values():
        if len({member.get("branch_id") for member in members}) < 2:
            continue
        pair_pass = False
        if judge is not None:
            ordered = sorted(members, key=lambda member: str(member.get("branch_id")))
            try:
                pair_pass = judge.judge_pair(
                    frame_a=ordered[0]["control_frame"],
                    transcript_a=str(ordered[0]["asr_transcript"]),
                    frame_b=ordered[1]["control_frame"],
                    transcript_b=str(ordered[1]["asr_transcript"]),
                )
            except (SemanticJudgeError, KeyError, TypeError, ValueError):
                pair_pass = False
        elif all(isinstance(member.get("pair_discrimination_pass"), bool) for member in members):
            pair_pass = all(member["pair_discrimination_pass"] for member in members)
        for member in members:
            member["pair_discrimination_pass"] = pair_pass
    report = evaluate_release_gate(
        trials,
        minimum_trials=args.minimum_trials,
        minimum_pairs=args.minimum_pairs,
        minimum_slice_trials=args.minimum_slice_trials,
    )
    report["evaluated_at"] = datetime.now(timezone.utc).isoformat()
    report["trials_path"] = str(args.trials.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.adjudicated_trials is not None:
        args.adjudicated_trials.parent.mkdir(parents=True, exist_ok=True)
        args.adjudicated_trials.write_text(
            "".join(json.dumps(trial, sort_keys=True) + "\n" for trial in trials),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "overall": report["overall"],
                "causal_pairs": report["causal_pairs"],
                "failure_reasons": report["failure_reasons"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
