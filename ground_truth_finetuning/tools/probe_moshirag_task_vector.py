#!/usr/bin/env python3
"""Probe a Moshika-RAG task vector on PersonaPlex without saving hybrid weights."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.arc4_conditioning import Arc4CausalTrainer  # noqa: E402
from ground_truth_finetuning.training.contracts import StreamLayout  # noqa: E402
from ground_truth_finetuning.training.moshirag_task_vector import (  # noqa: E402
    apply_task_vector_target,
    candidate_targets,
)
from ground_truth_finetuning.tools.train_arc4_causal import (  # noqa: E402
    PairData,
    evaluate_pairs,
    load_certified_pairs,
    load_jsonl,
)


class DirectArcStream(nn.Module):
    """Expose certified ARC rows unchanged apart from an explicit scalar ablation."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(1.0), persistent=False)

    def set_scale(self, value: float) -> None:
        self.scale.fill_(value)

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        return reference * self.scale.to(device=reference.device, dtype=reference.dtype)


def compact_evaluation(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"details"}
    }


def score(value: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(value["pair_passes"]),
        float(value["direction_passes"]),
        float(value["mean_cross_minus_own_focused_text_nll"]),
        float(value["mean_cross_minus_own_text_nll"]),
    )


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona-model", type=Path, required=True)
    parser.add_argument("--moshika-base", type=Path, required=True)
    parser.add_argument("--moshika-rag", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--pair-certificate", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--arc4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coarse-pairs", type=int, default=12)
    parser.add_argument("--full-pairs", type=int, default=46)
    parser.add_argument("--full-candidates", type=int, default=2)
    parser.add_argument("--context-frames", type=int, default=64)
    parser.add_argument("--post-target-tail-frames", type=int, default=16)
    args = parser.parse_args()
    if min(args.coarse_pairs, args.full_pairs, args.full_candidates, args.context_frames) < 1:
        raise SystemExit("probe sizes must be positive")
    if args.post_target_tail_frames < 0:
        raise SystemExit("post-target tail frames must be non-negative")
    required = (
        args.persona_model,
        args.moshika_base,
        args.moshika_rag,
        args.model_contract,
        args.manifest,
        args.certificate,
        args.pair_index,
        args.pair_certificate,
    )
    if any(not value.resolve().is_file() for value in required):
        raise SystemExit("one or more task-vector probe inputs are missing")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("task-vector probe is CUDA-only and requires one visible GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    sys.path.insert(0, str(args.moshi_source_root.resolve()))
    from moshi.models.loaders import get_moshi_lm

    contract = json.loads(args.model_contract.read_text(encoding="utf-8"))
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    lm = get_moshi_lm(args.persona_model.resolve(), device=device, dtype=torch.bfloat16)
    if any(parameter.device.type != "cuda" for parameter in lm.parameters()):
        raise SystemExit("PersonaPlex model was not loaded wholly on CUDA")
    layout = StreamLayout.from_mapping(contract["stream_layout"])
    layout.validate_for_model(lm)
    pairs = load_certified_pairs(args.pair_index.resolve(), args.pair_certificate.resolve(), certificate)
    heldout = [value for value in pairs if value.get("split") == "validation"]
    if len(heldout) < args.full_pairs:
        raise SystemExit("validation split is smaller than the requested full probe")
    rows = load_jsonl(args.manifest.resolve())
    data = PairData(
        rows,
        artifact_root=args.artifact_root.resolve(),
        arc4_root=args.arc4_root.resolve(),
        device=device,
        hidden_size=int(lm.dim),
        text_stream_index=layout.text_stream_indices[0],
        zero_token_id=int(lm.zero_token_id),
        context_frames=args.context_frames,
        post_target_tail_frames=args.post_target_tail_frames,
    )
    adapter = DirectArcStream().to(device=device)
    evaluator = Arc4CausalTrainer(
        lm,
        adapter,
        None,  # Evaluation-only probe: no optimizer or gradients are constructed.
        layout,
        activation_checkpointing=False,
        audio_weight=0.0,
        counterfactual_margin=0.08,
        focused_counterfactual_margin=0.30,
        null_weight=0.0,
        stale_weight=0.0,
    )
    candidates = [
        {"name": "persona_null", "scope": "none", "alpha": 0.0, "streamScale": 0.0},
        {"name": "persona_arc", "scope": "none", "alpha": 0.0, "streamScale": 1.0},
        *[
            {
                "name": f"hybrid_temporal_{alpha:g}",
                "scope": "temporal",
                "alpha": alpha,
                "streamScale": 1.0,
            }
            for alpha in (0.25, 0.5, 1.0)
        ],
        *[
            {
                "name": f"hybrid_temporal_text_{alpha:g}",
                "scope": "temporal_text",
                "alpha": alpha,
                "streamScale": 1.0,
            }
            for alpha in (0.25, 0.5, 1.0)
        ],
    ]
    current = {"temporal": 0.0, "text": 0.0}
    coarse_results: list[dict[str, Any]] = []
    with torch.inference_mode(), safe_open(
        args.moshika_base.resolve(), framework="pt", device=0
    ) as base_file, safe_open(args.moshika_rag.resolve(), framework="pt", device=0) as rag_file:
        for candidate in candidates:
            mutation = apply_task_vector_target(
                lm,
                base_file,
                rag_file,
                current,
                candidate_targets(candidate["scope"], candidate["alpha"]),
            )
            adapter.set_scale(candidate["streamScale"])
            evaluation = evaluate_pairs(
                evaluator,
                heldout,
                data,
                max_pairs=args.coarse_pairs,
            )
            coarse_results.append(
                {**candidate, "mutation": mutation, "evaluation": compact_evaluation(evaluation)}
            )
            print(json.dumps({"phase": "coarse", "candidate": candidate["name"], **compact_evaluation(evaluation)}, sort_keys=True), flush=True)
        eligible = [value for value in coarse_results if value["name"] != "persona_null"]
        selected = sorted(eligible, key=lambda value: score(value["evaluation"]), reverse=True)[
            : args.full_candidates
        ]
        full_results: list[dict[str, Any]] = []
        for candidate in selected:
            mutation = apply_task_vector_target(
                lm,
                base_file,
                rag_file,
                current,
                candidate_targets(candidate["scope"], candidate["alpha"]),
            )
            adapter.set_scale(candidate["streamScale"])
            evaluation = evaluate_pairs(
                evaluator,
                heldout,
                data,
                max_pairs=args.full_pairs,
            )
            full_results.append(
                {**candidate, "mutation": mutation, "evaluation": evaluation}
            )
            print(json.dumps({"phase": "full", "candidate": candidate["name"], **compact_evaluation(evaluation)}, sort_keys=True), flush=True)
    report = {
        "schema": "personaplex.moshirag-task-vector-probe.v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "cudaDevice": torch.cuda.get_device_name(device),
        "personaModel": str(args.persona_model.resolve()),
        "moshikaBase": str(args.moshika_base.resolve()),
        "moshikaRag": str(args.moshika_rag.resolve()),
        "arc4Root": str(args.arc4_root.resolve()),
        "arc4ConditionerRevision": certificate.get("conditionerRevision"),
        "coarsePairs": args.coarse_pairs,
        "fullPairs": args.full_pairs,
        "selectionRule": "lexicographic pair passes, direction passes, focused mean, whole mean",
        "audioParameterMutation": "forbidden",
        "coarseResults": coarse_results,
        "fullResults": full_results,
        "promotionScope": "teacher-forced transfer probe only; no hybrid checkpoint was saved",
    }
    write_json(args.output.resolve(), report)
    print(json.dumps({"status": "complete", "artifact": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
