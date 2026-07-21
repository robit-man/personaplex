#!/usr/bin/env python3
"""Train ARC-4 as PersonaPlex's primary control stream on certified causal pairs."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import lru_cache
import json
import os
from pathlib import Path
import random
import sys
from datetime import datetime, timezone
from typing import Any, Mapping

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.evaluation.reliability import wilson_interval  # noqa: E402
from ground_truth_finetuning.training.arc4_conditioning import (  # noqa: E402
    Arc4CausalTrainer,
    Arc4ConditioningBundle,
    Arc4InjectionConfig,
    FIELD_PERSISTENT_ARC4_ARCHITECTURE,
    GatedArc4InjectionAdapter,
    LAYERWISE_PERSISTENT_ARC4_ARCHITECTURE,
    LAYERWISE_ADAPTED_ARC4_ARCHITECTURE,
)
from ground_truth_finetuning.training.contracts import StreamLayout  # noqa: E402
from ground_truth_finetuning.training.arc4_two_path import (  # noqa: E402
    Arc4TwoPathAdapter,
    Arc4TwoPathConfig,
    TWO_PATH_ARC4_ARCHITECTURE,
)
from ground_truth_finetuning.training.moshirag_task_vector import (  # noqa: E402
    apply_task_vector_target,
    candidate_targets,
)
from ground_truth_finetuning.training.temporal_lora import (  # noqa: E402
    TemporalLoRAConfig,
    install_temporal_lora,
)
from ground_truth_finetuning.training.hard_pair_sampling import (  # noqa: E402
    HardPairCurriculum,
)
from ground_truth_finetuning.training.native_source import require_moshi_source_contract  # noqa: E402
from ground_truth_finetuning.training.native_training import exact_text_contrast_masks  # noqa: E402
from personaplex_control.arc4_packing import ARC4_PACKING_REVISION  # noqa: E402
from ground_truth_finetuning.tools.train_arc4_reference import (  # noqa: E402
    Arc4TensorReader,
    crop_native_turn,
    hash_file,
    load_jsonl,
    load_tensor,
    rank_info,
    rank_log,
    wait_for_host_memory,
)


TWO_PATH_STREAM_PACKING_REVISION = "arc4-global-first-v2"


class PairData:
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        artifact_root: Path,
        arc4_root: Path,
        device: torch.device,
        hidden_size: int,
        text_stream_index: int,
        zero_token_id: int,
        context_frames: int,
        post_target_tail_frames: int,
    ) -> None:
        self.rows = {str(row["example_id"]): row for row in rows}
        if len(self.rows) != len(rows):
            raise ValueError("ARC-4 causal manifest contains duplicate example IDs")
        self.artifact_root = artifact_root
        self.reader = Arc4TensorReader(arc4_root, device, hidden_size)
        self.device = device
        self.text_stream_index = text_stream_index
        self.zero_token_id = zero_token_id
        self.context_frames = context_frames
        self.post_target_tail_frames = post_target_tail_frames

    @lru_cache(maxsize=160)
    def _cpu_example(self, example_id: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[example_id]
        encoding = row["model_encoding"]
        codes = load_tensor(self.artifact_root / str(encoding["codes_path"]), "codes")
        mask = load_tensor(self.artifact_root / str(encoding["target_mask_path"]), "target_mask")
        return crop_native_turn(
            codes,
            mask,
            int(encoding["prefix_at"]),
            context_frames=self.context_frames,
            post_target_tail_frames=self.post_target_tail_frames,
        )

    def pair_examples(
        self, example_a: str, example_b: str
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        codes_a, mask_a, at_a = self._cpu_example(example_a)
        codes_b, mask_b, at_b = self._cpu_example(example_b)
        if at_a != at_b or not torch.equal(codes_a[:, :at_a], codes_b[:, :at_b]):
            raise ValueError("pair certificate violated exact shared-prefix invariance")
        contrast = exact_text_contrast_masks(
            codes_a,
            mask_a,
            codes_b,
            mask_b,
            text_stream_index=self.text_stream_index,
            zero_token_id=self.zero_token_id,
        )

        def move(codes, mask, at, focused):
            return {
                "codes": codes.unsqueeze(0).to(self.device, non_blocking=True),
                "agent_target_mask": mask.unsqueeze(0).to(self.device, non_blocking=True),
                "contrast_target_mask": focused.unsqueeze(0).to(self.device, non_blocking=True),
                "prefix_at": torch.tensor([at], device=self.device, dtype=torch.long),
            }

        return move(codes_a, mask_a, at_a, contrast.mask_a), move(
            codes_b, mask_b, at_b, contrast.mask_b
        )

    def reference(self, example_id: str) -> torch.Tensor:
        return self.reader.load(self.rows[example_id])


class TwoPathPairData:
    """Join a decision-field reference and a detailed global stream by example ID."""

    def __init__(self, detailed: PairData, decision: PairData, decision_frames: int) -> None:
        if set(detailed.rows) != set(decision.rows):
            raise ValueError("two-path ARC manifests do not contain identical example IDs")
        for example_id, detailed_row in detailed.rows.items():
            decision_row = decision.rows[example_id]
            detailed_hash = detailed_row.get("control", {}).get("frame_hash")
            decision_hash = decision_row.get("control", {}).get("frame_hash")
            if detailed_hash != decision_hash:
                raise ValueError("two-path ARC manifests disagree on control frame identity")
        self.detailed = detailed
        self.decision = decision
        self.decision_frames = decision_frames
        self.rows = detailed.rows

    def pair_examples(
        self, example_a: str, example_b: str
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return self.detailed.pair_examples(example_a, example_b)

    def reference(self, example_id: str) -> torch.Tensor:
        decision = self.decision.reference(example_id)
        detailed = self.detailed.reference(example_id)
        if decision.shape[1] < self.decision_frames:
            raise ValueError("two-path decision reference is shorter than its certified slot")
        return torch.cat((decision[:, : self.decision_frames], detailed), dim=1)


def load_certified_pairs(
    pair_index: Path,
    pair_certificate: Path,
    native_certificate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    certificate = json.loads(pair_certificate.read_text(encoding="utf-8"))
    if certificate.get("status") != "certified_for_causal_control_training":
        raise ValueError("pair certificate does not authorize causal training")
    if certificate.get("pair_index_sha256") != hash_file(pair_index):
        raise ValueError("pair index hash does not match its certificate")
    if certificate.get("manifest_sha256") != native_certificate.get("nativeManifestSha256"):
        raise ValueError("pair certificate and ARC join bind different native manifests")
    pairs = load_jsonl(pair_index)
    if len(pairs) != int(certificate.get("pairs", -1)):
        raise ValueError("pair certificate count mismatch")
    return pairs


def pair_members(pair: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    left, right = pair.get("member_a"), pair.get("member_b")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ValueError("causal pair lacks member_a/member_b")
    return left, right


def evaluate_pairs(
    trainer: Arc4CausalTrainer,
    pairs: list[dict[str, Any]],
    data: PairData,
    *,
    max_pairs: int,
) -> dict[str, Any]:
    selected = pairs[:max_pairs]
    if not selected:
        raise ValueError("ARC checkpoint evaluation requires held-out pairs")
    pair_passes = 0
    direction_passes = 0
    null_passes = 0
    stale_passes = 0
    stale_total = 0
    whole_deltas: list[float] = []
    focused_deltas: list[float] = []
    details: list[dict[str, Any]] = []
    was_training = trainer.adapter.training
    trainer.adapter.eval()
    try:
        for pair in selected:
            left, right = pair_members(pair)
            left_id, right_id = str(left["example_id"]), str(right["example_id"])
            example_left, example_right = data.pair_examples(left_id, right_id)
            stale_left = str(left.get("stale_example_id") or "")
            stale_right = str(right.get("stale_example_id") or "")
            result = trainer.evaluate_pair(
                example_left,
                example_right,
                data.reference(left_id),
                data.reference(right_id),
                stale_a=data.reference(stale_left) if stale_left in data.rows else None,
                stale_b=data.reference(stale_right) if stale_right in data.rows else None,
            )
            whole = [
                float((result.a_cross_text - result.a_own_text).cpu()),
                float((result.b_cross_text - result.b_own_text).cpu()),
            ]
            focused = [
                float((result.a_cross_focused_text - result.a_own_focused_text).cpu()),
                float((result.b_cross_focused_text - result.b_own_focused_text).cpu()),
            ]
            branch_passes = [
                whole[index] >= trainer.counterfactual_margin_value
                and focused[index] >= trainer.focused_counterfactual_margin_value
                for index in range(2)
            ]
            pair_passes += int(all(branch_passes))
            direction_passes += sum(branch_passes)
            whole_deltas.extend(whole)
            focused_deltas.extend(focused)
            nulls = [
                float((result.a_null_text - result.a_own_text).cpu()),
                float((result.b_null_text - result.b_own_text).cpu()),
            ]
            null_passes += sum(value >= trainer.null_margin_value for value in nulls)
            stales: list[float] = []
            if result.a_stale_text is not None:
                stales.append(float((result.a_stale_text - result.a_own_text).cpu()))
            if result.b_stale_text is not None:
                stales.append(float((result.b_stale_text - result.b_own_text).cpu()))
            stale_total += len(stales)
            stale_passes += sum(value >= trainer.stale_margin_value for value in stales)
            details.append(
                {
                    "pair_id": pair["pair_id"],
                    "whole_deltas": whole,
                    "focused_deltas": focused,
                    "null_deltas": nulls,
                    "stale_deltas": stales,
                    "pair_pass": all(branch_passes),
                }
            )
    finally:
        trainer.adapter.train(was_training)
    lower, _ = wilson_interval(pair_passes, len(selected))
    return {
        "kind": "personaplex-arc4-causal-checkpoint-evaluation-v2",
        "pairs": len(selected),
        "pair_passes": pair_passes,
        "pair_sensitivity": pair_passes / len(selected),
        "pair_sensitivity_wilson_lower": lower,
        "direction_passes": direction_passes,
        "direction_total": len(selected) * 2,
        "mean_cross_minus_own_text_nll": sum(whole_deltas) / len(whole_deltas),
        "mean_cross_minus_own_focused_text_nll": sum(focused_deltas) / len(focused_deltas),
        "null_margin_passes": null_passes,
        "null_margin_total": len(selected) * 2,
        "stale_margin_passes": stale_passes,
        "stale_margin_total": stale_total,
        "details": details,
        "promotion_scope": "teacher-forced causal diagnostic only; generated duplex gate remains mandatory",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--pair-certificate", type=Path, required=True)
    parser.add_argument("--packed-pair-certificate", type=Path, required=True)
    parser.add_argument("--field-slot-certificate", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--arc4-root", type=Path, required=True)
    parser.add_argument("--decision-manifest", type=Path)
    parser.add_argument("--decision-certificate", type=Path)
    parser.add_argument("--decision-arc4-root", type=Path)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-optimizer-state", action="store_true")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-pairs", type=int, default=32)
    parser.add_argument("--context-frames", type=int, default=64)
    parser.add_argument("--post-target-tail-frames", type=int, default=16)
    parser.add_argument("--adapter-rank", type=int, default=256)
    parser.add_argument("--adapter-architecture", default=LAYERWISE_ADAPTED_ARC4_ARCHITECTURE)
    parser.add_argument("--initial-gate", type=float, default=0.25)
    parser.add_argument("--max-stream-rms", type=float, default=0.35)
    parser.add_argument("--output-stream-frames", type=int, default=128)
    parser.add_argument("--layer-control-count", type=int, default=8)
    parser.add_argument("--layer-initial-gate", type=float, default=0.05)
    parser.add_argument("--max-layer-rms", type=float, default=0.15)
    parser.add_argument("--layer-adaptation-rank", type=int, default=128)
    parser.add_argument("--layer-adaptation-initial-gate", type=float, default=0.10)
    parser.add_argument("--max-layer-adaptation-rms", type=float, default=0.10)
    parser.add_argument("--two-path-decision-frames", type=int, default=48)
    parser.add_argument("--two-path-stream-frames", type=int, default=96)
    parser.add_argument("--two-path-prefix-tokens", type=int, default=8)
    parser.add_argument("--two-path-rank", type=int, default=256)
    parser.add_argument("--two-path-attention-heads", type=int, default=8)
    parser.add_argument("--two-path-prefix-gate", type=float, default=0.05)
    parser.add_argument("--two-path-max-prefix-rms", type=float, default=0.15)
    parser.add_argument("--two-path-stream-scale", type=float, default=1.0)
    parser.add_argument("--two-path-max-stream-residual-rms", type=float, default=0.10)
    parser.add_argument("--task-vector-base", type=Path)
    parser.add_argument("--task-vector-rag", type=Path)
    parser.add_argument("--task-vector-import-report", type=Path)
    parser.add_argument("--task-vector-temporal-alpha", type=float, default=0.0)
    parser.add_argument("--temporal-lora-layers", type=int, default=0)
    parser.add_argument("--temporal-lora-rank", type=int, default=8)
    parser.add_argument("--temporal-lora-alpha", type=float, default=16.0)
    parser.add_argument("--temporal-lora-learning-rate", type=float)
    parser.add_argument("--freeze-control-adapter", action="store_true")
    parser.add_argument("--freeze-temporal-lora", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--counterfactual-margin", type=float, default=0.08)
    parser.add_argument("--focused-counterfactual-margin", type=float, default=0.30)
    parser.add_argument("--null-margin", type=float, default=0.03)
    parser.add_argument("--stale-margin", type=float, default=0.03)
    parser.add_argument("--causal-weight", type=float, default=2.0)
    parser.add_argument("--matched-weight", type=float, default=1.0)
    parser.add_argument("--whole-causal-weight", type=float, default=1.0)
    parser.add_argument("--focused-causal-weight", type=float, default=1.0)
    parser.add_argument("--null-weight", type=float, default=0.25)
    parser.add_argument("--stale-weight", type=float, default=0.25)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--contrastive-temperature", type=float, default=0.10)
    parser.add_argument("--contrastive-whole-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-focused-weight", type=float, default=2.0)
    parser.add_argument("--coverage-surrogate-weight", type=float, default=0.0)
    parser.add_argument("--coverage-surrogate-temperature", type=float, default=0.25)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("--hard-curriculum", type=Path)
    parser.add_argument("--hard-replay-ratio", type=float, default=0.0)
    parser.add_argument("--train-eval-pairs", type=int, default=32)
    parser.add_argument("--pair-worst-direction-weight", type=float, default=0.0)
    parser.add_argument("--pair-worst-temperature", type=float, default=0.05)
    parser.add_argument("--max-host-memory-fraction", type=float, default=0.80)
    parser.add_argument("--memory-poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    two_path_mode = args.adapter_architecture == TWO_PATH_ARC4_ARCHITECTURE
    if min(
        args.max_steps,
        args.checkpoint_every,
        args.eval_pairs,
        args.context_frames,
        args.adapter_rank,
        args.output_stream_frames,
        args.layer_control_count,
        args.layer_adaptation_rank,
        args.gradient_accumulation_steps,
        args.train_eval_pairs,
        args.two_path_decision_frames,
        args.two_path_stream_frames,
        args.two_path_prefix_tokens,
        args.two_path_rank,
        args.two_path_attention_heads,
    ) < 1:
        raise SystemExit("ARC-4 causal step/size arguments must be positive")
    if args.post_target_tail_frames < 0:
        raise SystemExit("post-target tail frames must be non-negative")
    if not 0 < args.max_host_memory_fraction < 1:
        raise SystemExit("host memory limit must be in (0,1)")
    if not 0.0 <= args.pair_worst_direction_weight <= 1.0:
        raise SystemExit("pair-worst-direction-weight must be in [0,1]")
    if args.pair_worst_temperature <= 0.0:
        raise SystemExit("pair-worst-temperature must be positive")
    if args.hard_replay_ratio < 0.0:
        raise SystemExit("hard-replay-ratio cannot be negative")
    if (args.hard_curriculum is None) != (args.hard_replay_ratio == 0.0):
        raise SystemExit("hard curriculum and a positive replay ratio must be supplied together")
    two_path_files = (
        args.decision_manifest,
        args.decision_certificate,
        args.decision_arc4_root,
        args.task_vector_base,
        args.task_vector_rag,
        args.task_vector_import_report,
    )
    if two_path_mode and (
        any(value is None for value in two_path_files)
        or any(not value.resolve().exists() for value in two_path_files if value is not None)
    ):
        raise SystemExit("two-path ARC training requires every decision/task-vector input")
    if two_path_mode and not 0.0 < args.task_vector_temporal_alpha <= 1.0:
        raise SystemExit("two-path temporal task-vector alpha must be in (0,1]")
    objective_weights = (
        args.causal_weight,
        args.matched_weight,
        args.whole_causal_weight,
        args.focused_causal_weight,
        args.null_weight,
        args.stale_weight,
        args.contrastive_weight,
        args.contrastive_whole_weight,
        args.contrastive_focused_weight,
        args.coverage_surrogate_weight,
    )
    if any(value < 0 for value in objective_weights) or args.focused_causal_weight <= 0:
        raise SystemExit("objective weights must be non-negative and focused causal weight must be positive")
    if args.contrastive_temperature <= 0.0 or args.contrastive_focused_weight <= 0.0:
        raise SystemExit("contrastive temperature/focused weight must be positive")
    if args.coverage_surrogate_temperature <= 0.0:
        raise SystemExit("coverage surrogate temperature must be positive")

    rank, local_rank, world_size = rank_info()
    rank_log(rank, "process_started", localRank=local_rank, worldSize=world_size)
    if not torch.cuda.is_available():
        raise SystemExit("ARC-4 causal training is CUDA-only")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    native_certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    manifest = args.manifest.resolve()
    if native_certificate.get("status") != "certified_for_arc4_adapter_training":
        raise SystemExit("ARC join certificate does not authorize training")
    if native_certificate.get("joinedManifestSha256") != hash_file(manifest):
        raise SystemExit("ARC join certificate does not match manifest")
    primary_packing_revision = (
        TWO_PATH_STREAM_PACKING_REVISION if two_path_mode else ARC4_PACKING_REVISION
    )
    if native_certificate.get("packingRevision") != primary_packing_revision:
        raise SystemExit("ARC join certificate has a stale packing contract")
    packed_certificate_path = args.packed_pair_certificate.resolve()
    packed_certificate = json.loads(packed_certificate_path.read_text(encoding="utf-8"))
    if packed_certificate.get("status") != "certified":
        raise SystemExit("packed ARC pair certificate does not authorize training")
    if packed_certificate.get("packingRevision") != primary_packing_revision:
        raise SystemExit("packed ARC pair certificate has a stale packing contract")
    if packed_certificate.get("manifestSha256") != hash_file(manifest):
        raise SystemExit("packed ARC pair certificate does not match the joined manifest")
    if packed_certificate.get("pairIndexSha256") != hash_file(args.pair_index.resolve()):
        raise SystemExit("packed ARC pair certificate does not match the pair index")
    field_certificate_path = args.field_slot_certificate.resolve()
    field_certificate = json.loads(field_certificate_path.read_text(encoding="utf-8"))
    if field_certificate.get("schema") != "personaplex.arc4-field-slot-certificate.v1":
        raise SystemExit("field-slot certificate schema is unsupported")
    if field_certificate.get("status") != "certified":
        raise SystemExit("field-slot certificate does not authorize training")
    if field_certificate.get("packingRevision") != ARC4_PACKING_REVISION:
        raise SystemExit("field-slot certificate has a stale packing contract")
    decision_manifest = (
        args.decision_manifest.resolve() if two_path_mode and args.decision_manifest else manifest
    )
    if field_certificate.get("joinedManifestSha256") != hash_file(decision_manifest):
        raise SystemExit("field-slot certificate does not match the joined manifest")
    if field_certificate.get("pairIndexSha256") != hash_file(args.pair_index.resolve()):
        raise SystemExit("field-slot certificate does not match the pair index")
    pairs = load_certified_pairs(
        args.pair_index.resolve(), args.pair_certificate.resolve(), native_certificate
    )
    if int(field_certificate.get("pairs", -1)) != len(pairs):
        raise SystemExit("field-slot certificate pair count mismatch")
    decision_certificate = None
    if two_path_mode:
        assert args.decision_certificate is not None
        assert args.decision_arc4_root is not None
        decision_certificate = json.loads(args.decision_certificate.read_text(encoding="utf-8"))
        if decision_certificate.get("status") != "certified_for_arc4_adapter_training":
            raise SystemExit("two-path decision certificate does not authorize training")
        if decision_certificate.get("packingRevision") != ARC4_PACKING_REVISION:
            raise SystemExit("two-path decision certificate has a stale packing revision")
        if decision_certificate.get("joinedManifestSha256") != hash_file(decision_manifest):
            raise SystemExit("two-path decision certificate does not match its manifest")
        if Path(decision_certificate.get("arc4Root", "")).resolve() != args.decision_arc4_root.resolve():
            raise SystemExit("two-path decision certificate does not match its ARC root")
    train_pairs = [pair for pair in pairs if pair.get("split") == "train"]
    heldout_pairs = [pair for pair in pairs if pair.get("split") == "validation"]
    if len(train_pairs) < world_size or not heldout_pairs:
        raise SystemExit("certified causal pair index lacks train/validation pairs")
    hard_curriculum = None
    hard_curriculum_record = None
    if args.hard_curriculum is not None:
        hard_path = args.hard_curriculum.resolve()
        document = json.loads(hard_path.read_text(encoding="utf-8"))
        if document.get("pairIndexSha256") != hash_file(args.pair_index.resolve()):
            raise SystemExit("hard curriculum pair-index hash mismatch")
        hard_curriculum = HardPairCurriculum.from_document(
            document,
            expected_pair_ids=tuple(str(pair["pair_id"]) for pair in train_pairs),
            replay_ratio=args.hard_replay_ratio,
            seed=args.shuffle_seed,
        )
        hard_curriculum_record = {
            "path": str(hard_path),
            "sha256": hash_file(hard_path),
            "sourceStep": document.get("sourceStep"),
            "method": document.get("method"),
            "replayRatio": args.hard_replay_ratio,
            "cycleSize": hard_curriculum.cycle_size,
        }

    contract = json.loads(args.model_contract.read_text(encoding="utf-8"))
    require_moshi_source_contract(args.moshi_source_root.resolve(), contract)
    model_hash = hash_file(args.moshi_path.resolve())
    if contract.get("moshi_weights_sha256") != model_hash:
        raise SystemExit("Moshi weights do not match model contract")
    os.environ.setdefault("NO_TORCH_COMPILE", "1")
    os.environ.setdefault("NO_CUDA_GRAPH", "1")
    sys.path.insert(0, str(args.moshi_source_root.resolve()))
    from moshi.models.loaders import get_moshi_lm

    if world_size > 1:
        dist.init_process_group(
            "nccl",
            device_id=torch.device("cuda", local_rank),
        )
    lm = get_moshi_lm(args.moshi_path.resolve(), device=device, dtype=torch.bfloat16)
    task_vector_record = None
    if two_path_mode:
        assert args.task_vector_base is not None
        assert args.task_vector_rag is not None
        assert args.task_vector_import_report is not None
        import_report = json.loads(args.task_vector_import_report.read_text(encoding="utf-8"))
        if import_report.get("complete") is not True:
            raise SystemExit("MoshiRAG import report is incomplete")
        verified_files = {
            str(Path(item["path"]).resolve()): item
            for artifact in import_report.get("artifacts", [])
            for item in artifact.get("files", [])
            if item.get("status") == "verified"
        }
        base_path = args.task_vector_base.resolve()
        rag_path = args.task_vector_rag.resolve()
        if str(base_path) not in verified_files or str(rag_path) not in verified_files:
            raise SystemExit("task-vector weights are not bound by the verified import report")
        with safe_open(base_path, framework="pt", device=local_rank) as base_file, safe_open(
            rag_path, framework="pt", device=local_rank
        ) as rag_file:
            mutation = apply_task_vector_target(
                lm,
                base_file,
                rag_file,
                {"temporal": 0.0, "text": 0.0},
                candidate_targets("temporal", args.task_vector_temporal_alpha),
            )
        task_vector_record = {
            "scope": "temporal_transformer_only",
            "alpha": args.task_vector_temporal_alpha,
            "base": str(base_path),
            "baseSha256": "sha256:" + verified_files[str(base_path)]["actual_sha256"],
            "rag": str(rag_path),
            "ragSha256": "sha256:" + verified_files[str(rag_path)]["actual_sha256"],
            "importReport": str(args.task_vector_import_report.resolve()),
            "importReportSha256": hash_file(args.task_vector_import_report.resolve()),
            **mutation,
            "audioParameterMutation": "forbidden",
        }
    layout = StreamLayout.from_mapping(contract["stream_layout"])
    layout.validate_for_model(lm)
    if args.temporal_lora_layers < 0:
        raise SystemExit("temporal-lora-layers cannot be negative")
    if args.temporal_lora_layers and not two_path_mode:
        raise SystemExit("temporal LoRA requires two-path ARC conditioning")
    if args.freeze_control_adapter and not args.temporal_lora_layers:
        raise SystemExit("freezing the control adapter requires temporal LoRA")
    if args.freeze_temporal_lora and not args.temporal_lora_layers:
        raise SystemExit("freezing temporal LoRA requires installed temporal LoRA")
    if args.freeze_control_adapter and args.freeze_temporal_lora:
        raise SystemExit("control adapter and temporal LoRA cannot both be frozen")
    if args.temporal_lora_learning_rate is not None and args.temporal_lora_learning_rate <= 0:
        raise SystemExit("temporal LoRA learning rate must be positive")
    temporal_lora_config = (
        TemporalLoRAConfig(
            layer_count=args.temporal_lora_layers,
            rank=args.temporal_lora_rank,
            alpha=args.temporal_lora_alpha,
        )
        if args.temporal_lora_layers
        else None
    )
    temporal_lora = (
        install_temporal_lora(lm, temporal_lora_config)
        if temporal_lora_config is not None
        else None
    )
    installed_temporal_lora_parameters = (
        tuple(temporal_lora.parameters())
        if temporal_lora is not None
        else ()
    )
    temporal_lora_parameters = (
        () if args.freeze_temporal_lora else installed_temporal_lora_parameters
    )
    limited_adaptation = temporal_lora is not None
    layer_indices: tuple[int, ...] = ()
    if args.adapter_architecture in {
        LAYERWISE_PERSISTENT_ARC4_ARCHITECTURE,
        LAYERWISE_ADAPTED_ARC4_ARCHITECTURE,
    }:
        total_layers = len(lm.transformer.layers)
        if args.layer_control_count > total_layers:
            raise SystemExit("layer-control-count exceeds PersonaPlex transformer depth")
        layer_indices = tuple(range(total_layers - args.layer_control_count, total_layers))
    if two_path_mode:
        config: Arc4InjectionConfig | Arc4TwoPathConfig = Arc4TwoPathConfig(
            hidden_size=int(lm.dim),
            decision_frames=args.two_path_decision_frames,
            stream_frames=args.two_path_stream_frames,
            prefix_tokens=args.two_path_prefix_tokens,
            rank=args.two_path_rank,
            attention_heads=args.two_path_attention_heads,
            initial_prefix_gate=args.two_path_prefix_gate,
            max_prefix_rms=args.two_path_max_prefix_rms,
            initial_stream_scale=args.two_path_stream_scale,
            max_stream_residual_rms=args.two_path_max_stream_residual_rms,
        )
        adapter: torch.nn.Module = Arc4TwoPathAdapter(config).to(device=device)
    else:
        config = Arc4InjectionConfig(
            hidden_size=int(lm.dim),
            rank=args.adapter_rank,
            initial_gate=args.initial_gate,
            max_stream_rms=args.max_stream_rms,
            architecture_revision=args.adapter_architecture,
            output_frames=args.output_stream_frames,
            layer_indices=layer_indices,
            layer_initial_gate=args.layer_initial_gate,
            max_layer_rms=args.max_layer_rms,
            layer_adaptation_rank=args.layer_adaptation_rank,
            layer_adaptation_initial_gate=args.layer_adaptation_initial_gate,
            max_layer_adaptation_rms=args.max_layer_adaptation_rms,
        )
        adapter = GatedArc4InjectionAdapter(config).to(device=device)
    checkpoint_schema = (
        "personaplex.arc4-causal-control.v10"
        if limited_adaptation
        else (
            "personaplex.arc4-causal-control.v9"
            if two_path_mode
            else "personaplex.arc4-causal-control.v7"
        )
    )
    conditioning_mode = (
        "arc4_ordered_prefix_frozen_stream_upper_temporal_lora_v10"
        if limited_adaptation
        else (
            "arc4_ordered_prefix_frozen_global_stream_v9"
            if two_path_mode
            else "arc4_primary_layerwise_adapted_v7_field_slots"
        )
    )
    legacy_prefix_mode = "ordered_segment_prefix_v9" if two_path_mode else "disabled"
    resume_payload = None
    resume_step = 0
    resume_mode = None
    resume_parent_schema = None
    if args.resume_checkpoint is not None:
        resume_path = args.resume_checkpoint.resolve()
        try:
            resume_payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        except TypeError:
            resume_payload = torch.load(resume_path, map_location="cpu")
        resume_parent_schema = (
            str(resume_payload.get("schema"))
            if isinstance(resume_payload, dict)
            else None
        )
        v9_warm_start = (
            limited_adaptation
            and resume_parent_schema == "personaplex.arc4-causal-control.v9"
        )
        resume_mode = (
            "v9_adapter_to_v10_zero_initialized_temporal_lora"
            if v9_warm_start
            else "exact_checkpoint_continuation"
        )
        expected_resume = {
            "schema": (
                "personaplex.arc4-causal-control.v9"
                if v9_warm_start
                else checkpoint_schema
            ),
            "conditioning_mode": (
                "arc4_ordered_prefix_frozen_global_stream_v9"
                if v9_warm_start
                else conditioning_mode
            ),
            "legacy_prefix_mode": legacy_prefix_mode,
            "model_revision": contract["model_revision"],
            "conditioner_revision": native_certificate["conditionerRevision"],
            "packing_revision": primary_packing_revision,
            "manifest_sha256": hash_file(manifest),
            "pair_index_sha256": hash_file(args.pair_index.resolve()),
        }
        if two_path_mode:
            expected_resume["decision_manifest_sha256"] = hash_file(decision_manifest)
        if not isinstance(resume_payload, dict) or any(
            resume_payload.get(key) != value for key, value in expected_resume.items()
        ):
            raise SystemExit("resume checkpoint does not match the causal training contract")
        if two_path_mode and resume_payload.get("task_vector") != task_vector_record:
            raise SystemExit("resume checkpoint task vector mismatch")
        if resume_payload.get("arc4_adapter_config") != config.as_dict():
            raise SystemExit("resume checkpoint adapter configuration mismatch")
        state = resume_payload.get("arc4_adapter_state_dict")
        if not isinstance(state, dict):
            raise SystemExit("resume checkpoint lacks adapter state")
        adapter.load_state_dict(state, strict=True)
        if temporal_lora is not None and not v9_warm_start:
            if resume_payload.get("temporal_lora_config") != temporal_lora_config.as_dict():
                raise SystemExit("resume checkpoint temporal LoRA configuration mismatch")
            if tuple(resume_payload.get("temporal_lora_layer_indices") or ()) != temporal_lora.layer_indices:
                raise SystemExit("resume checkpoint temporal LoRA layer selection mismatch")
            temporal_lora_state = resume_payload.get("temporal_lora_state_dict")
            if not isinstance(temporal_lora_state, dict):
                raise SystemExit("resume checkpoint lacks temporal LoRA state")
            temporal_lora.load_state_dict(temporal_lora_state)
        resume_step = int(resume_payload.get("step", 0))
        if resume_step < 1 or args.max_steps <= resume_step:
            raise SystemExit("max-steps must exceed the positive resume checkpoint step")
    if args.freeze_control_adapter:
        for parameter in adapter.parameters():
            parameter.requires_grad_(False)
    adapter_parameters = tuple(
        parameter for parameter in adapter.parameters() if parameter.requires_grad
    )
    if world_size > 1 and adapter_parameters:
        adapter = DistributedDataParallel(adapter, device_ids=[local_rank], output_device=local_rank)
    optimizer_groups: list[dict[str, Any]] = []
    if adapter_parameters:
        optimizer_groups.append(
            {"params": adapter_parameters, "lr": args.learning_rate, "name": "control_adapter"}
        )
    if temporal_lora_parameters:
        optimizer_groups.append(
            {
                "params": temporal_lora_parameters,
                "lr": (
                    args.temporal_lora_learning_rate
                    if args.temporal_lora_learning_rate is not None
                    else args.learning_rate
                ),
                "name": "upper_temporal_lora",
            }
        )
    if not optimizer_groups:
        raise SystemExit("causal training has no trainable parameters")
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.01)
    optimizer_resume_mode = "fresh"
    if args.resume_optimizer_state:
        if resume_payload is None:
            raise SystemExit("optimizer resume requires a checkpoint")
        optimizer_state = resume_payload.get("optimizer_state_dict")
        if not isinstance(optimizer_state, dict):
            raise SystemExit("resume checkpoint lacks optimizer state")
        expected_objective = {
            "matched": args.matched_weight,
            "causal": args.causal_weight,
            "whole": args.whole_causal_weight,
            "focused": args.focused_causal_weight,
            "null": args.null_weight,
            "stale": args.stale_weight,
            "contrastive": args.contrastive_weight,
            "contrastive_temperature": args.contrastive_temperature,
            "contrastive_whole": args.contrastive_whole_weight,
            "contrastive_focused": args.contrastive_focused_weight,
            "coverage_surrogate": args.coverage_surrogate_weight,
            "coverage_surrogate_temperature": args.coverage_surrogate_temperature,
            "pair_worst_direction": args.pair_worst_direction_weight,
            "pair_worst_temperature": args.pair_worst_temperature,
        }
        if resume_payload.get("objective_weights") != expected_objective:
            raise SystemExit("optimizer resume objective mismatch")
        optimizer.load_state_dict(optimizer_state)
        optimizer_resume_mode = "restored_exact"
    trainer = Arc4CausalTrainer(
        lm,
        adapter,
        optimizer,
        layout,
        counterfactual_margin=args.counterfactual_margin,
        focused_counterfactual_margin=args.focused_counterfactual_margin,
        null_margin=args.null_margin,
        stale_margin=args.stale_margin,
        causal_weight=args.causal_weight,
        matched_weight=args.matched_weight,
        whole_causal_weight=args.whole_causal_weight,
        focused_causal_weight=args.focused_causal_weight,
        null_weight=args.null_weight,
        stale_weight=args.stale_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_whole_weight=args.contrastive_whole_weight,
        contrastive_focused_weight=args.contrastive_focused_weight,
        coverage_surrogate_weight=args.coverage_surrogate_weight,
        coverage_surrogate_temperature=args.coverage_surrogate_temperature,
        pair_worst_direction_weight=args.pair_worst_direction_weight,
        pair_worst_temperature=args.pair_worst_temperature,
        trainable_model_parameters=temporal_lora_parameters,
    )
    module = adapter.module if isinstance(adapter, DistributedDataParallel) else adapter
    evaluator = Arc4CausalTrainer(
        lm,
        module,
        optimizer,
        layout,
        activation_checkpointing=False,
        counterfactual_margin=args.counterfactual_margin,
        focused_counterfactual_margin=args.focused_counterfactual_margin,
        null_margin=args.null_margin,
        stale_margin=args.stale_margin,
        causal_weight=args.causal_weight,
        matched_weight=args.matched_weight,
        whole_causal_weight=args.whole_causal_weight,
        focused_causal_weight=args.focused_causal_weight,
        null_weight=args.null_weight,
        stale_weight=args.stale_weight,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_whole_weight=args.contrastive_whole_weight,
        contrastive_focused_weight=args.contrastive_focused_weight,
        coverage_surrogate_weight=args.coverage_surrogate_weight,
        coverage_surrogate_temperature=args.coverage_surrogate_temperature,
        pair_worst_direction_weight=args.pair_worst_direction_weight,
        pair_worst_temperature=args.pair_worst_temperature,
        trainable_model_parameters=temporal_lora_parameters,
    )
    rows = load_jsonl(manifest)
    detailed_data = PairData(
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
    if two_path_mode:
        assert args.decision_arc4_root is not None
        decision_data = PairData(
            load_jsonl(decision_manifest),
            artifact_root=args.artifact_root.resolve(),
            arc4_root=args.decision_arc4_root.resolve(),
            device=device,
            hidden_size=int(lm.dim),
            text_stream_index=layout.text_stream_indices[0],
            zero_token_id=int(lm.zero_token_id),
            context_frames=args.context_frames,
            post_target_tail_frames=args.post_target_tail_frames,
        )
        data: PairData | TwoPathPairData = TwoPathPairData(
            detailed_data,
            decision_data,
            args.two_path_decision_frames,
        )
    else:
        data = detailed_data
    effective_pairs_per_update = world_size * args.gradient_accumulation_steps
    resume_samples_seen = 0
    if resume_payload is not None:
        prior_schedule = resume_payload.get("optimization_schedule") or {}
        prior_effective = int(
            prior_schedule.get(
                "effective_pairs_per_optimizer_update",
                effective_pairs_per_update,
            )
        )
        resume_samples_seen = int(
            resume_payload.get(
                "samples_seen",
                resume_step * prior_effective,
            )
        )
        if prior_effective < 1 or resume_samples_seen < 0:
            raise SystemExit("resume checkpoint has an invalid sample schedule")
    hard_curriculum_origin_samples = 0
    if hard_curriculum is not None:
        prior_hard = (
            resume_payload.get("hard_curriculum")
            if isinstance(resume_payload, dict)
            else None
        )
        if (
            isinstance(prior_hard, dict)
            and prior_hard.get("sha256") == hard_curriculum_record["sha256"]
            and float(prior_hard.get("replayRatio", -1.0)) == args.hard_replay_ratio
        ):
            hard_curriculum_origin_samples = int(
                prior_hard.get("originSamplesSeen", 0)
            )
        else:
            hard_curriculum_origin_samples = resume_samples_seen
        if hard_curriculum_origin_samples < 0 or hard_curriculum_origin_samples > resume_samples_seen:
            raise SystemExit("hard curriculum has an invalid sample origin")
        hard_curriculum_record["originSamplesSeen"] = hard_curriculum_origin_samples
    shuffled_pair_orders: dict[int, list[int]] = {}
    train_pairs_by_id = {str(pair["pair_id"]): pair for pair in train_pairs}
    train_evaluation_pairs = list(train_pairs)
    random.Random(args.shuffle_seed + 1_000_003).shuffle(train_evaluation_pairs)

    def pair_for_micro_step(update_offset: int, micro_step: int) -> Mapping[str, Any]:
        global_sample = (
            resume_samples_seen
            + update_offset * effective_pairs_per_update
            + rank * args.gradient_accumulation_steps
            + micro_step
        )
        if hard_curriculum is not None:
            return train_pairs_by_id[
                hard_curriculum.pair_id_for_sample(
                    global_sample - hard_curriculum_origin_samples
                )
            ]
        epoch, offset = divmod(global_sample, len(train_pairs))
        order = shuffled_pair_orders.get(epoch)
        if order is None:
            order = list(range(len(train_pairs)))
            random.Random(args.shuffle_seed + epoch).shuffle(order)
            shuffled_pair_orders[epoch] = order
        return train_pairs[order[offset]]

    if rank == 0:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        (args.run_dir / "run_contract.json").write_text(
            json.dumps(
                {
                    "schema": (
                        "personaplex.arc4-causal-training-run.v10"
                        if limited_adaptation
                        else (
                            "personaplex.arc4-causal-training-run.v9"
                            if two_path_mode
                            else "personaplex.arc4-causal-training-run.v7"
                        )
                    ),
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "manifest": str(manifest),
                    "manifestSha256": hash_file(manifest),
                    "pairIndex": str(args.pair_index.resolve()),
                    "pairIndexSha256": hash_file(args.pair_index.resolve()),
                    "pairCertificate": str(args.pair_certificate.resolve()),
                    "packedPairCertificate": str(packed_certificate_path),
                    "packedPairCertificateSha256": hash_file(packed_certificate_path),
                    "fieldSlotCertificate": str(field_certificate_path),
                    "fieldSlotCertificateSha256": hash_file(field_certificate_path),
                    "decisionManifest": str(decision_manifest) if two_path_mode else None,
                    "decisionManifestSha256": hash_file(decision_manifest) if two_path_mode else None,
                    "decisionCertificate": (
                        str(args.decision_certificate.resolve())
                        if two_path_mode and args.decision_certificate
                        else None
                    ),
                    "decisionArc4Root": (
                        str(args.decision_arc4_root.resolve())
                        if two_path_mode and args.decision_arc4_root
                        else None
                    ),
                    "modelRevision": contract["model_revision"],
                    "modelSha256": model_hash,
                    "conditionerRevision": native_certificate["conditionerRevision"],
                    "packingRevision": primary_packing_revision,
                    "decisionPackingRevision": ARC4_PACKING_REVISION if two_path_mode else None,
                    "adapterConfig": config.as_dict(),
                    "controlAdapterTrainable": not args.freeze_control_adapter,
                    "taskVector": task_vector_record,
                    "temporalLora": (
                        {
                            "config": temporal_lora_config.as_dict(),
                            "layerIndices": list(temporal_lora.layer_indices),
                            "trainableParameters": sum(
                                parameter.numel()
                                for parameter in temporal_lora_parameters
                            ),
                            "installedParameters": sum(
                                parameter.numel()
                                for parameter in installed_temporal_lora_parameters
                            ),
                            "trainable": not args.freeze_temporal_lora,
                            "audioDepformerMutation": "forbidden",
                        }
                        if temporal_lora is not None
                        else None
                    ),
                    "worldSize": world_size,
                    "legacyPrefixMode": legacy_prefix_mode,
                    "callerStreamSupervision": "forbidden",
                    "targetTextPassedToConditioner": False,
                    "hostMemoryThrottleFraction": args.max_host_memory_fraction,
                    "optimizationSchedule": {
                        "gradientAccumulationStepsPerRank": args.gradient_accumulation_steps,
                        "effectivePairsPerOptimizerUpdate": (
                            effective_pairs_per_update
                        ),
                        "resumeSamplesSeen": resume_samples_seen,
                        "shuffleSeed": args.shuffle_seed,
                        "sampling": (
                            "deterministic_uniform_plus_weighted_hard_replay"
                            if hard_curriculum is not None
                            else "deterministic_epoch_shuffle_without_replacement"
                        ),
                        "controlAdapterLearningRate": (
                            args.learning_rate if adapter_parameters else None
                        ),
                        "temporalLoraLearningRate": (
                            args.temporal_lora_learning_rate
                            if args.temporal_lora_learning_rate is not None
                            else (args.learning_rate if temporal_lora_parameters else None)
                        ),
                    },
                    "hardCurriculum": hard_curriculum_record,
                    "objectiveWeights": {
                        "matched": args.matched_weight,
                        "causal": args.causal_weight,
                        "whole": args.whole_causal_weight,
                        "focused": args.focused_causal_weight,
                        "null": args.null_weight,
                        "stale": args.stale_weight,
                        "contrastive": args.contrastive_weight,
                        "contrastiveTemperature": args.contrastive_temperature,
                        "contrastiveWhole": args.contrastive_whole_weight,
                        "contrastiveFocused": args.contrastive_focused_weight,
                        "coverageSurrogate": args.coverage_surrogate_weight,
                        "coverageSurrogateTemperature": args.coverage_surrogate_temperature,
                        "pairWorstDirection": args.pair_worst_direction_weight,
                        "pairWorstTemperature": args.pair_worst_temperature,
                    },
                    "evaluation": {
                        "heldoutPairs": min(args.eval_pairs, len(heldout_pairs)),
                        "trainPairs": min(args.train_eval_pairs, len(train_evaluation_pairs)),
                        "trainSampling": "deterministic_shuffle",
                        "evaluationOnly": args.evaluation_only,
                    },
                    "resume": {
                        "checkpoint": str(args.resume_checkpoint.resolve()),
                        "checkpointSha256": hash_file(args.resume_checkpoint.resolve()),
                        "step": resume_step,
                        "parentSchema": resume_parent_schema,
                        "mode": resume_mode,
                        "optimizerState": optimizer_resume_mode,
                    } if args.resume_checkpoint is not None else None,
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
    if world_size > 1:
        dist.barrier()
    metrics_path = args.run_dir / "metrics.jsonl"

    def checkpoint(step: int) -> None:
        path = args.run_dir / f"arc4_causal_step_{step:06d}.pt"
        torch.save(
            {
                "schema": checkpoint_schema,
                "conditioning_mode": conditioning_mode,
                "legacy_prefix_mode": legacy_prefix_mode,
                "arc4_adapter_state_dict": module.state_dict(),
                "arc4_adapter_config": config.as_dict(),
                "control_adapter_trainable": not args.freeze_control_adapter,
                "model_revision": contract["model_revision"],
                "conditioner_revision": native_certificate["conditionerRevision"],
                "packing_revision": primary_packing_revision,
                "decision_packing_revision": ARC4_PACKING_REVISION if two_path_mode else None,
                "manifest_sha256": hash_file(manifest),
                "decision_manifest_sha256": hash_file(decision_manifest) if two_path_mode else None,
                "pair_index_sha256": hash_file(args.pair_index.resolve()),
                "field_slot_certificate_sha256": hash_file(field_certificate_path),
                "task_vector": task_vector_record,
                "temporal_lora_config": (
                    temporal_lora_config.as_dict()
                    if temporal_lora_config is not None
                    else None
                ),
                "temporal_lora_state_dict": (
                    temporal_lora.state_dict()
                    if temporal_lora is not None
                    else None
                ),
                "temporal_lora_layer_indices": (
                    temporal_lora.layer_indices
                    if temporal_lora is not None
                    else ()
                ),
                "objective_weights": {
                    "matched": args.matched_weight,
                    "causal": args.causal_weight,
                    "whole": args.whole_causal_weight,
                    "focused": args.focused_causal_weight,
                    "null": args.null_weight,
                    "stale": args.stale_weight,
                    "contrastive": args.contrastive_weight,
                    "contrastive_temperature": args.contrastive_temperature,
                    "contrastive_whole": args.contrastive_whole_weight,
                    "contrastive_focused": args.contrastive_focused_weight,
                    "coverage_surrogate": args.coverage_surrogate_weight,
                    "coverage_surrogate_temperature": args.coverage_surrogate_temperature,
                    "pair_worst_direction": args.pair_worst_direction_weight,
                    "pair_worst_temperature": args.pair_worst_temperature,
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "optimization_schedule": {
                    "gradient_accumulation_steps_per_rank": args.gradient_accumulation_steps,
                    "effective_pairs_per_optimizer_update": (
                        effective_pairs_per_update
                    ),
                    "shuffle_seed": args.shuffle_seed,
                    "sampling": (
                        "deterministic_uniform_plus_weighted_hard_replay"
                        if hard_curriculum is not None
                        else "deterministic_epoch_shuffle_without_replacement"
                    ),
                    "control_adapter_learning_rate": (
                        args.learning_rate if adapter_parameters else None
                    ),
                    "temporal_lora_learning_rate": (
                        args.temporal_lora_learning_rate
                        if args.temporal_lora_learning_rate is not None
                        else (args.learning_rate if temporal_lora_parameters else None)
                    ),
                },
                "hard_curriculum": hard_curriculum_record,
                "samples_seen": (
                    resume_samples_seen
                    + (step - resume_step) * effective_pairs_per_update
                ),
                "resume_parent_sha256": (
                    hash_file(args.resume_checkpoint.resolve())
                    if args.resume_checkpoint is not None
                    else None
                ),
                "resume_parent_schema": resume_parent_schema,
                "resume_mode": resume_mode,
                "step": step,
            },
            path,
        )
        evaluation = evaluate_pairs(
            evaluator,
            heldout_pairs,
            data,
            max_pairs=min(args.eval_pairs, len(heldout_pairs)),
        )
        train_evaluation = evaluate_pairs(
            evaluator,
            train_evaluation_pairs,
            data,
            max_pairs=min(args.train_eval_pairs, len(train_evaluation_pairs)),
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event": "checkpoint",
                        "step": step,
                        "checkpoint": path.name,
                        **evaluation,
                        "train_evaluation": train_evaluation,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            summary_keys = (
                "direction_passes",
                "direction_total",
                "pair_passes",
                "pairs",
                "pair_sensitivity",
                "pair_sensitivity_wilson_lower",
                "mean_cross_minus_own_text_nll",
                "mean_cross_minus_own_focused_text_nll",
                "null_margin_passes",
                "null_margin_total",
                "stale_margin_passes",
                "stale_margin_total",
            )
            handle.write(
                json.dumps(
                    {
                        "event": "checkpoint_summary",
                        "step": step,
                        "checkpoint": path.name,
                        "heldout": {
                            key: evaluation.get(key) for key in summary_keys
                        },
                        "train": {
                            key: train_evaluation.get(key) for key in summary_keys
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    if rank == 0:
        checkpoint(resume_step)
    if world_size > 1:
        dist.barrier()
    if args.evaluation_only:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
        return 0
    adapter.train()
    try:
        for step in range(resume_step, args.max_steps):
            wait_for_host_memory(
                rank=rank,
                world_size=world_size,
                device=device,
                limit=args.max_host_memory_fraction,
                poll_seconds=args.memory_poll_seconds,
            )
            trainer.optimizer.zero_grad(set_to_none=True)
            values = torch.zeros(13, dtype=torch.float64, device=device)
            update_offset = step - resume_step
            for micro_step in range(args.gradient_accumulation_steps):
                pair = pair_for_micro_step(update_offset, micro_step)
                left, right = pair_members(pair)
                left_id, right_id = str(left["example_id"]), str(right["example_id"])
                example_left, example_right = data.pair_examples(left_id, right_id)
                stale_left = str(left.get("stale_example_id") or "")
                stale_right = str(right.get("stale_example_id") or "")
                synchronize = micro_step + 1 == args.gradient_accumulation_steps
                sync_context = (
                    nullcontext()
                    if synchronize or not isinstance(adapter, DistributedDataParallel)
                    else adapter.no_sync()
                )
                with sync_context:
                    result = trainer.backward_pair(
                        example_left,
                        example_right,
                        data.reference(left_id),
                        data.reference(right_id),
                        stale_a=data.reference(stale_left) if stale_left in data.rows else None,
                        stale_b=data.reference(stale_right) if stale_right in data.rows else None,
                        loss_scale=1.0 / args.gradient_accumulation_steps,
                    )
                values += torch.stack(
                    [
                        result.total.detach().double(),
                        result.matched_sft.detach().double(),
                        result.counterfactual_margin.detach().double(),
                        result.focused_counterfactual_margin.detach().double(),
                        result.contrastive_control.detach().double(),
                        result.coverage_surrogate.detach().double(),
                        result.null_margin.detach().double(),
                        result.stale_margin.detach().double(),
                        (result.a_cross_text - result.a_own_text).detach().double(),
                        (result.b_cross_text - result.b_own_text).detach().double(),
                        (result.a_cross_focused_text - result.a_own_focused_text).detach().double(),
                        (result.b_cross_focused_text - result.b_own_focused_text).detach().double(),
                        module.gate.detach().double(),
                    ]
                )
            if world_size > 1:
                for parameter in temporal_lora_parameters:
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad)
                        parameter.grad.div_(world_size)
            gradient_norm = trainer.apply_gradients().detach().double().to(device)
            if world_size > 1:
                dist.all_reduce(gradient_norm)
                gradient_norm /= world_size
            values /= args.gradient_accumulation_steps
            if world_size > 1:
                dist.all_reduce(values)
                values /= world_size
            completed = step + 1
            if rank == 0:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "step": completed,
                                "loss": float(values[0]),
                                "matched_sft": float(values[1]),
                                "counterfactual_margin_loss": float(values[2]),
                                "focused_counterfactual_margin_loss": float(values[3]),
                                "contrastive_control_loss": float(values[4]),
                                "coverage_surrogate_loss": float(values[5]),
                                "null_margin_loss": float(values[6]),
                                "stale_margin_loss": float(values[7]),
                                "a_cross_minus_own_text_nll": float(values[8]),
                                "b_cross_minus_own_text_nll": float(values[9]),
                                "a_cross_minus_own_focused_text_nll": float(values[10]),
                                "b_cross_minus_own_focused_text_nll": float(values[11]),
                                "gate": float(values[12]),
                                "gradient_norm": float(gradient_norm),
                            },
                            sort_keys=True,
                        ) + "\n"
                    )
            if completed % args.checkpoint_every == 0 or completed == args.max_steps:
                if world_size > 1:
                    dist.barrier()
                if rank == 0:
                    checkpoint(completed)
                if world_size > 1:
                    dist.barrier()
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
