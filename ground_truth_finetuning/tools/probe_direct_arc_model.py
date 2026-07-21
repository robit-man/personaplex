#!/usr/bin/env python3
"""Evaluate one CUDA speech LM against certified causal pairs using direct ARC rows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.arc4_conditioning import Arc4CausalTrainer  # noqa: E402
from ground_truth_finetuning.training.contracts import StreamLayout  # noqa: E402
from ground_truth_finetuning.tools.probe_moshirag_task_vector import (  # noqa: E402
    DirectArcStream,
    write_json,
)
from ground_truth_finetuning.tools.train_arc4_causal import (  # noqa: E402
    PairData,
    evaluate_pairs,
    load_certified_pairs,
    load_jsonl,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lm-config", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--stream-layout-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--pair-certificate", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--arc4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=46)
    parser.add_argument("--context-frames", type=int, default=64)
    parser.add_argument("--post-target-tail-frames", type=int, default=16)
    args = parser.parse_args()
    if min(args.pairs, args.context_frames) < 1 or args.post_target_tail_frames < 0:
        raise SystemExit("direct ARC probe sizes are invalid")
    required = (
        args.model,
        args.lm_config,
        args.stream_layout_contract,
        args.manifest,
        args.certificate,
        args.pair_index,
        args.pair_certificate,
    )
    if any(not value.resolve().is_file() for value in required):
        raise SystemExit("one or more direct ARC probe inputs are missing")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("direct ARC probe is CUDA-only and requires one visible GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    sys.path.insert(0, str(args.moshi_source_root.resolve()))
    from moshi.models.loaders import get_moshi_lm

    lm_config = json.loads(args.lm_config.read_text(encoding="utf-8"))
    lm_config.pop("conditioners", None)
    lm_config.pop("fuser", None)
    lm_config.pop("model_id", None)
    lm_config["cross_attention"] = False
    lm = get_moshi_lm(
        args.model.resolve(),
        lm_kwargs=lm_config,
        device=device,
        dtype=torch.bfloat16,
    )
    if not hasattr(lm, "embed_codes"):
        def embed_codes(model, sequence):
            value = None
            for codebook_index in range(model.num_audio_codebooks):
                embedded = model.emb[codebook_index](
                    sequence[:, codebook_index + model.audio_offset]
                )
                value = embedded if value is None else value + embedded
            text = model.text_emb(sequence[:, 0])
            return text if value is None else value + text

        lm.embed_codes = MethodType(embed_codes, lm)
    if not hasattr(lm, "forward_embeddings"):
        def forward_embeddings(model, embeddings, streaming_sum=None, **kwargs):
            unsupported = {
                key: value
                for key, value in kwargs.items()
                if value not in (None, (), False)
            }
            if unsupported:
                raise ValueError(f"official direct probe received unsupported layer controls: {unsupported.keys()}")
            if streaming_sum is not None:
                if streaming_sum.shape != embeddings.shape:
                    raise ValueError("streaming_sum must match offline embedding sequence")
                embeddings = embeddings + streaming_sum.to(
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                )
            return model.forward_text(sequence_emb=embeddings)

        lm.forward_embeddings = MethodType(forward_embeddings, lm)
    if any(parameter.device.type != "cuda" for parameter in lm.parameters()):
        raise SystemExit("speech LM was not loaded wholly on CUDA")
    stream_contract = json.loads(args.stream_layout_contract.read_text(encoding="utf-8"))
    layout = StreamLayout.from_mapping(stream_contract["stream_layout"])
    layout.validate_for_model(lm)
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    pairs = load_certified_pairs(args.pair_index.resolve(), args.pair_certificate.resolve(), certificate)
    heldout = [value for value in pairs if value.get("split") == "validation"]
    if len(heldout) < args.pairs:
        raise SystemExit("validation split is smaller than requested direct ARC probe")
    data = PairData(
        load_jsonl(args.manifest.resolve()),
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
        None,
        layout,
        activation_checkpointing=False,
        audio_weight=0.0,
        counterfactual_margin=0.08,
        focused_counterfactual_margin=0.30,
        null_weight=0.0,
        stale_weight=0.0,
    )
    with torch.inference_mode():
        evaluation = evaluate_pairs(evaluator, heldout, data, max_pairs=args.pairs)
    report = {
        "schema": "personaplex.direct-arc-model-probe.v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "cudaDevice": torch.cuda.get_device_name(device),
        "model": str(args.model.resolve()),
        "lmConfig": str(args.lm_config.resolve()),
        "arc4Root": str(args.arc4_root.resolve()),
        "arc4ConditionerRevision": certificate.get("conditionerRevision"),
        "evaluation": evaluation,
        "promotionScope": "teacher-forced architecture/data alignment probe only",
    }
    write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": "complete",
                "artifact": str(args.output.resolve()),
                **{key: value for key, value in evaluation.items() if key != "details"},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
