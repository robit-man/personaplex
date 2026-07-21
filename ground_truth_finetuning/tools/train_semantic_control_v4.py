#!/usr/bin/env python3
"""Distributed Stage-1 training for the PersonaPlex v4 semantic control stream."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.causal_trainer import CausalControlTrainer
from ground_truth_finetuning.training.contracts import (
    StreamLayout,
    validate_control_frame_mapping,
    validate_evidence_frame_mapping,
)
from ground_truth_finetuning.training.control_encoding import FieldAwareControlSerializer
from ground_truth_finetuning.training.control_stream import (
    ControlStreamConfig,
    SemanticControlStreamAdapter,
)
from ground_truth_finetuning.training.gpu_admission import host_memory_snapshot
from ground_truth_finetuning.training.native_source import require_moshi_source_contract
from ground_truth_finetuning.training.native_training import exact_text_contrast_masks


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_tensor(path: Path, name: str) -> torch.Tensor:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    tensor = value.get(name) if isinstance(value, dict) else value
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{path} does not contain tensor {name!r}")
    return tensor


def rank_info() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def require_source_certificate(path: Path, manifest: Path, contract: dict[str, Any]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != "personaplex-corpus-certificate":
        raise ValueError("unexpected native tensor certificate kind")
    if value.get("status") != "certified_for_adapter_training":
        raise ValueError("native tensor corpus is not certified for adapter training")
    if value.get("manifest_sha256") != hash_file(manifest):
        raise ValueError("native tensor certificate does not match manifest")
    if set(value.get("model_revisions", [])) != {contract["model_revision"]}:
        raise ValueError("native tensor certificate model revision mismatch")


def require_pair_certificate(
    path: Path,
    pair_index: Path,
    manifest: Path,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "certified_for_causal_control_training":
        raise ValueError("causal pair index is not certified")
    if value.get("pair_index_sha256") != hash_file(pair_index):
        raise ValueError("causal pair certificate does not match pair-index bytes")
    if value.get("manifest_sha256") != hash_file(manifest):
        raise ValueError("causal pair certificate does not match native manifest")
    if value.get("split_leakage_groups") != 0:
        raise ValueError("causal pair index contains split leakage")
    return value


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total < 1:
        return 0.0
    point = successes / total
    denominator = 1 + z * z / total
    centre = point + z * z / (2 * total)
    spread = z * math.sqrt((point * (1 - point) + z * z / (4 * total)) / total)
    return max(0.0, (centre - spread) / denominator)


def wait_for_host_memory(*, limit: float, rank: int, event_path: Path) -> None:
    while True:
        snapshot = host_memory_snapshot()
        if float(snapshot["used_ratio"]) < limit:
            return
        if rank == 0:
            with event_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "at": datetime.now(timezone.utc).isoformat(),
                            "event": "host_memory_throttle",
                            "limit": limit,
                            **snapshot,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        time.sleep(5)


def write_rank_stage(run_root: Path, *, rank: int, stage: str) -> None:
    path = run_root / f"startup_rank_{rank}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "at": datetime.now(timezone.utc).isoformat(),
                    "rank": rank,
                    "stage": stage,
                },
                sort_keys=True,
            )
            + "\n"
        )


def shared_integrity(
    *,
    run_root: Path,
    rank: int,
    moshi_path: Path,
    tokenizer_path: Path,
    timeout_seconds: int = 900,
) -> tuple[str, str]:
    path = run_root / "startup_integrity.json"
    if rank == 0:
        value = {
            "moshi_weights_sha256": hash_file(moshi_path),
            "tokenizer_sha256": hash_file(tokenizer_path),
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    else:
        deadline = time.monotonic() + timeout_seconds
        while not path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("rank-zero integrity result was not published")
            time.sleep(0.2)
    value = json.loads(path.read_text(encoding="utf-8"))
    return str(value["moshi_weights_sha256"]), str(value["tokenizer_sha256"])


class PairData:
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        artifact_root: Path,
        serializer: FieldAwareControlSerializer,
        tokenizer: Any,
        text_cardinality: int,
        max_tokens: int,
        history_seconds: float,
        target_tail_seconds: float,
        text_stream_index: int,
        zero_token_id: int,
        device: torch.device,
    ) -> None:
        self.rows = {str(row["example_id"]): row for row in records}
        self.artifact_root = artifact_root
        self.serializer = serializer
        self.tokenizer = tokenizer
        self.text_cardinality = text_cardinality
        self.max_tokens = max_tokens
        self.history_seconds = history_seconds
        self.target_tail_seconds = target_tail_seconds
        self.text_stream_index = text_stream_index
        self.zero_token_id = zero_token_id
        self.device = device

    @lru_cache(maxsize=96)
    def _cpu_example(self, example_id: str) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = self.rows[example_id]
        encoding = row["model_encoding"]
        codes = load_tensor(self.artifact_root / encoding["codes_path"], "codes")
        target_mask = load_tensor(
            self.artifact_root / encoding["target_mask_path"], "target_mask"
        ).bool()
        prefix_at = int(encoding["prefix_at"])
        frame_rate = float(encoding.get("codec", {}).get("frame_rate_hz", 0))
        if frame_rate <= 0:
            raise ValueError(f"{example_id}: native encoding lacks a positive frame rate")
        target_positions = torch.where(target_mask.any(dim=0))[0]
        if target_positions.numel() < 1:
            raise ValueError(f"{example_id}: target mask contains no supervised frame")
        history_frames = max(1, round(self.history_seconds * frame_rate))
        tail_frames = max(1, round(self.target_tail_seconds * frame_rate))
        start = max(0, prefix_at - history_frames)
        end = min(codes.shape[1], int(target_positions.max().item()) + 1 + tail_frames)
        if end <= prefix_at:
            raise ValueError(f"{example_id}: causal crop does not retain target suffix")
        cropped_codes = codes[:, start:end].contiguous()
        cropped_mask = target_mask[:, start:end].contiguous()
        cropped_prefix = prefix_at - start
        if cropped_mask[:, :cropped_prefix].any() or not cropped_mask[:, cropped_prefix:].any():
            raise ValueError(f"{example_id}: causal crop violates target-boundary isolation")
        return cropped_codes, cropped_mask, cropped_prefix

    def example(self, example_id: str) -> dict[str, torch.Tensor]:
        codes, target_mask, prefix_at = self._cpu_example(example_id)
        return {
            "codes": codes.unsqueeze(0).to(self.device, non_blocking=True),
            "agent_target_mask": target_mask.unsqueeze(0).to(self.device, non_blocking=True),
            "prefix_at": torch.tensor([prefix_at], device=self.device, dtype=torch.long),
        }

    @lru_cache(maxsize=96)
    def _cpu_pair_examples(
        self, example_a_id: str, example_b_id: str
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, int, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, int, torch.Tensor],
    ]:
        codes_a, target_a, prefix_a = self._cpu_example(example_a_id)
        codes_b, target_b, prefix_b = self._cpu_example(example_b_id)
        contrast = exact_text_contrast_masks(
            codes_a,
            target_a,
            codes_b,
            target_b,
            text_stream_index=self.text_stream_index,
            zero_token_id=self.zero_token_id,
        )
        return (
            (codes_a, target_a, prefix_a, contrast.mask_a),
            (codes_b, target_b, prefix_b, contrast.mask_b),
        )

    def pair_examples(
        self, example_a_id: str, example_b_id: str
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        rows = self._cpu_pair_examples(example_a_id, example_b_id)
        examples = []
        for codes, target_mask, prefix_at, contrast_mask in rows:
            examples.append(
                {
                    "codes": codes.unsqueeze(0).to(self.device, non_blocking=True),
                    "agent_target_mask": target_mask.unsqueeze(0).to(
                        self.device, non_blocking=True
                    ),
                    "contrast_target_mask": contrast_mask.unsqueeze(0).to(
                        self.device, non_blocking=True
                    ),
                    "prefix_at": torch.tensor(
                        [prefix_at], device=self.device, dtype=torch.long
                    ),
                }
            )
        return examples[0], examples[1]

    def control(self, example_id: str, *, expected_revision: int | None = None):
        row = self.rows[example_id]
        frame = validate_control_frame_mapping(row["control"]["frame"])
        evidence_value = row.get("evidence")
        raw_evidence = evidence_value.get("frame") if isinstance(evidence_value, dict) else None
        evidence = (
            validate_evidence_frame_mapping(raw_evidence)
            if isinstance(raw_evidence, dict)
            else None
        )
        return self.serializer.encode(
            frame,
            self.tokenizer,
            self.text_cardinality,
            evidence=evidence,
            expected_revision=expected_revision,
            max_tokens=self.max_tokens,
        )

    def revision(self, example_id: str) -> int:
        return validate_control_frame_mapping(self.rows[example_id]["control"]["frame"]).state_revision


def pair_components(pair: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    left = pair.get("member_a")
    right = pair.get("member_b")
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("causal pair lacks member_a/member_b")
    return left, right


def evaluate_pairs(
    trainer: CausalControlTrainer,
    pairs: list[dict[str, Any]],
    data: PairData,
    *,
    max_pairs: int,
) -> dict[str, Any]:
    selected = pairs[:max_pairs]
    if not selected:
        raise ValueError("checkpoint evaluation requires causal pairs")
    adapter_was_training = trainer.adapter.training
    trainer.adapter.eval()
    pair_passes = 0
    direction_passes = 0
    positive_direction_passes = 0
    null_passes = 0
    stale_passes = 0
    stale_total = 0
    deltas: list[float] = []
    focused_deltas: list[float] = []
    null_deltas: list[float] = []
    stale_deltas: list[float] = []
    details: list[dict[str, Any]] = []
    try:
        for pair in selected:
            left, right = pair_components(pair)
            left_id = str(left["example_id"])
            right_id = str(right["example_id"])
            left_revision = data.revision(left_id)
            right_revision = data.revision(right_id)
            control_left = data.control(left_id, expected_revision=left_revision)
            control_right = data.control(right_id, expected_revision=right_revision)
            stale_left_id = left.get("stale_example_id")
            stale_right_id = right.get("stale_example_id")
            example_left, example_right = data.pair_examples(left_id, right_id)
            result = trainer.evaluate_pair(
                example_left,
                example_right,
                control_left,
                control_right,
                stale_a=data.control(str(stale_left_id), expected_revision=left_revision)
                if stale_left_id in data.rows
                else None,
                stale_b=data.control(str(stale_right_id), expected_revision=right_revision)
                if stale_right_id in data.rows
                else None,
            )
            a_delta = float((result.a_cross_text - result.a_own_text).detach().cpu())
            b_delta = float((result.b_cross_text - result.b_own_text).detach().cpu())
            a_focused_delta = float(
                (result.a_cross_focused_text - result.a_own_focused_text).detach().cpu()
            )
            b_focused_delta = float(
                (result.b_cross_focused_text - result.b_own_focused_text).detach().cpu()
            )
            positive_direction_passes += int(a_delta > 0) + int(b_delta > 0)
            focused_deltas.extend((a_focused_delta, b_focused_delta))
            a_pass = (
                a_delta >= trainer.counterfactual_margin_value
                and a_focused_delta >= trainer.focused_counterfactual_margin_value
            )
            b_pass = (
                b_delta >= trainer.counterfactual_margin_value
                and b_focused_delta >= trainer.focused_counterfactual_margin_value
            )
            direction_passes += int(a_pass) + int(b_pass)
            pair_passes += int(a_pass and b_pass)
            deltas.extend((a_delta, b_delta))
            pair_null_deltas = (
                float((result.a_null_text - result.a_own_text).detach().cpu()),
                float((result.b_null_text - result.b_own_text).detach().cpu()),
            )
            null_deltas.extend(pair_null_deltas)
            null_passes += sum(
                int(delta >= trainer.null_margin_value) for delta in pair_null_deltas
            )
            pair_stale_deltas: list[float] = []
            if result.a_stale_text is not None:
                pair_stale_deltas.append(
                    float((result.a_stale_text - result.a_own_text).detach().cpu())
                )
            if result.b_stale_text is not None:
                pair_stale_deltas.append(
                    float((result.b_stale_text - result.b_own_text).detach().cpu())
                )
            stale_deltas.extend(pair_stale_deltas)
            stale_total += len(pair_stale_deltas)
            stale_passes += sum(
                int(delta >= trainer.stale_margin_value) for delta in pair_stale_deltas
            )
            details.append(
                {
                    "pair_id": pair["pair_id"],
                    "a_delta": a_delta,
                    "b_delta": b_delta,
                    "a_focused_delta": a_focused_delta,
                    "b_focused_delta": b_focused_delta,
                    "required_margin": trainer.counterfactual_margin_value,
                    "required_focused_margin": trainer.focused_counterfactual_margin_value,
                    "null_deltas": list(pair_null_deltas),
                    "stale_deltas": pair_stale_deltas,
                    "pair_pass": a_pass and b_pass,
                }
            )
    finally:
        trainer.adapter.train(adapter_was_training)
    return {
        "kind": "personaplex-v4-causal-checkpoint-evaluation",
        "pairs": len(selected),
        "pair_passes": pair_passes,
        "pair_sensitivity": pair_passes / len(selected),
        "pair_sensitivity_wilson_lower": wilson_lower(pair_passes, len(selected)),
        "direction_passes": direction_passes,
        "direction_total": len(selected) * 2,
        "positive_direction_passes": positive_direction_passes,
        "required_counterfactual_margin": trainer.counterfactual_margin_value,
        "required_focused_counterfactual_margin": trainer.focused_counterfactual_margin_value,
        "mean_cross_minus_own_text_nll": sum(deltas) / len(deltas),
        "mean_cross_minus_own_focused_text_nll": sum(focused_deltas)
        / len(focused_deltas),
        "mean_null_minus_own_text_nll": sum(null_deltas) / len(null_deltas),
        "null_margin_passes": null_passes,
        "null_margin_total": len(null_deltas),
        "mean_stale_minus_own_text_nll": (
            sum(stale_deltas) / len(stale_deltas) if stale_deltas else None
        ),
        "stale_margin_passes": stale_passes,
        "stale_margin_total": stale_total,
        "details": details,
        "promotion_scope": "teacher-forced causal diagnostic only; never the generated-audio release gate",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--pair-certificate", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--matched-weight", type=float, default=1.0)
    parser.add_argument("--causal-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-margin", type=float, default=0.08)
    parser.add_argument("--focused-counterfactual-margin", type=float, default=0.30)
    parser.add_argument("--null-weight", type=float, default=0.25)
    parser.add_argument("--stale-weight", type=float, default=0.25)
    parser.add_argument("--train-pair-limit", type=int, default=0)
    parser.add_argument("--eval-train-pairs", type=int, default=0)
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-pairs", type=int, default=32)
    parser.add_argument("--control-dim", type=int, default=1024)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--stream-frames", type=int, default=48)
    parser.add_argument("--max-control-tokens", type=int, default=512)
    parser.add_argument("--max-context-gate-adjustment", type=float, default=0.05)
    parser.add_argument("--max-stream-to-lexical-rms-ratio", type=float, default=0.25)
    parser.add_argument("--host-memory-limit", type=float, default=0.80)
    parser.add_argument("--history-seconds", type=float, default=20.0)
    parser.add_argument("--target-tail-seconds", type=float, default=0.64)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.max_steps < 1 or args.checkpoint_every < 1 or args.eval_pairs < 1:
        raise SystemExit("max-steps, checkpoint-every, and eval-pairs must be positive")
    if min(args.matched_weight, args.causal_weight, args.null_weight, args.stale_weight) < 0:
        raise SystemExit("objective weights must be non-negative")
    if args.counterfactual_margin <= 0 or args.focused_counterfactual_margin <= 0:
        raise SystemExit("counterfactual margins must be positive")
    if args.train_pair_limit < 0 or args.eval_train_pairs < 0:
        raise SystemExit("pair limits must be non-negative")
    if not 0.5 <= args.host_memory_limit <= 0.95:
        raise SystemExit("host-memory-limit must be between 0.5 and 0.95")
    if args.history_seconds <= 0 or args.target_tail_seconds <= 0:
        raise SystemExit("causal history and target-tail durations must be positive")
    rank, local_rank, world_size = rank_info()
    if not torch.cuda.is_available():
        raise SystemExit("v4 semantic-control training is CUDA-only; CPU fallback is prohibited")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42424242 + rank)
    manifest = args.manifest.resolve()
    pair_index_path = args.pair_index.resolve()
    contract = json.loads(args.model_contract.read_text(encoding="utf-8"))
    require_source_certificate(args.certificate.resolve(), manifest, contract)
    pair_certificate = require_pair_certificate(
        args.pair_certificate.resolve(), pair_index_path, manifest
    )
    require_moshi_source_contract(args.moshi_source_root.resolve(), contract)
    manifest_hash = hash_file(manifest)
    args.run_dir.parent.mkdir(parents=True, exist_ok=True)
    write_rank_stage(args.run_dir.parent, rank=rank, stage="integrity_wait")
    weights_hash, tokenizer_hash = shared_integrity(
        run_root=args.run_dir.parent,
        rank=rank,
        moshi_path=args.moshi_path.resolve(),
        tokenizer_path=args.tokenizer_path.resolve(),
    )
    if weights_hash != contract.get("moshi_weights_sha256"):
        raise SystemExit("PersonaPlex weights do not match the model contract")
    if tokenizer_hash != contract.get("tokenizer_sha256"):
        raise SystemExit("tokenizer does not match the model contract")
    sys.path.insert(0, str(args.moshi_source_root.resolve()))
    import sentencepiece
    from moshi.models.loaders import get_moshi_lm

    resource_events = args.run_dir.parent / "resource_events.jsonl"
    wait_for_host_memory(limit=args.host_memory_limit, rank=rank, event_path=resource_events)
    # Load independently on every admitted GPU. The live host-memory guard is
    # capacity-relative, so a large host can exploit parallel deserialization
    # while a smaller host waits before entering this phase.
    wait_for_host_memory(
        limit=args.host_memory_limit, rank=rank, event_path=resource_events
    )
    write_rank_stage(args.run_dir.parent, rank=rank, stage="model_load_started")
    lm = get_moshi_lm(args.moshi_path.resolve(), device=device, dtype=torch.bfloat16)
    lm.eval()
    write_rank_stage(args.run_dir.parent, rank=rank, stage="model_load_completed")
    if world_size > 1:
        write_rank_stage(args.run_dir.parent, rank=rank, stage="nccl_initialization_started")
        dist.init_process_group("nccl")
        dist.barrier()
        write_rank_stage(args.run_dir.parent, rank=rank, stage="nccl_initialization_completed")
    layout = StreamLayout.from_mapping(contract["stream_layout"])
    layout.validate_for_model(lm)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer_path.resolve()))
    config = ControlStreamConfig(
        control_dim=args.control_dim,
        encoder_layers=args.encoder_layers,
        attention_heads=args.attention_heads,
        stream_frames=args.stream_frames,
        max_tokens=args.max_control_tokens,
        max_context_gate_adjustment=args.max_context_gate_adjustment,
        max_stream_to_lexical_rms_ratio=args.max_stream_to_lexical_rms_ratio,
    )
    adapter: torch.nn.Module = SemanticControlStreamAdapter(
        lm_hidden_size=int(lm.dim), config=config
    ).to(device=device, dtype=torch.float32)
    resume_payload = None
    start_step = 0
    if args.resume_checkpoint is not None:
        try:
            resume_payload = torch.load(
                args.resume_checkpoint.resolve(), map_location="cpu", weights_only=True
            )
        except TypeError:
            resume_payload = torch.load(args.resume_checkpoint.resolve(), map_location="cpu")
        if resume_payload.get("schema_version") != 4:
            raise ValueError("resume checkpoint is not a v4 control-stream checkpoint")
        if resume_payload.get("model_revision") != contract["model_revision"]:
            raise ValueError("resume checkpoint model revision mismatch")
        if resume_payload.get("manifest_sha256") != manifest_hash:
            raise ValueError("resume checkpoint manifest mismatch")
        if resume_payload.get("adapter_config") != config.as_dict():
            raise ValueError("resume checkpoint adapter configuration mismatch")
        adapter.load_state_dict(resume_payload["adapter_state_dict"], strict=True)
        start_step = int(resume_payload.get("step", 0))
        if start_step >= args.max_steps:
            raise ValueError("resume checkpoint step must be below max-steps")
    if world_size > 1:
        adapter = DistributedDataParallel(
            adapter, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False
        )
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    if (
        resume_payload is not None
        and not args.reset_optimizer_on_resume
        and isinstance(resume_payload.get("optimizer_state_dict"), dict)
    ):
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device=device, non_blocking=True)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
    trainer = CausalControlTrainer(
        lm,
        adapter,
        optimizer,
        layout,
        matched_weight=args.matched_weight,
        causal_weight=args.causal_weight,
        null_weight=args.null_weight,
        stale_weight=args.stale_weight,
        counterfactual_margin=args.counterfactual_margin,
        focused_counterfactual_margin=args.focused_counterfactual_margin,
    )
    records = load_jsonl(manifest)
    pairs = load_jsonl(pair_index_path)
    train_pairs = [pair for pair in pairs if pair.get("split") == "train"]
    heldout_pairs = [pair for pair in pairs if pair.get("split") in {"validation", "test"}]
    random.Random(42424242).shuffle(train_pairs)
    if args.train_pair_limit:
        train_pairs = train_pairs[: args.train_pair_limit]
    if len(train_pairs) < world_size or not heldout_pairs:
        raise SystemExit("causal pair index lacks train or held-out coverage")
    local_pairs = train_pairs[rank::world_size]
    serializer = FieldAwareControlSerializer()
    data = PairData(
        records,
        artifact_root=args.artifact_root.resolve(),
        serializer=serializer,
        tokenizer=tokenizer,
        text_cardinality=int(lm.text_card),
        max_tokens=config.max_tokens,
        history_seconds=args.history_seconds,
        target_tail_seconds=args.target_tail_seconds,
        text_stream_index=layout.text_stream_indices[0],
        zero_token_id=int(lm.zero_token_id),
        device=device,
    )
    write_rank_stage(args.run_dir.parent, rank=rank, stage="pair_data_ready")
    module = adapter.module if isinstance(adapter, DistributedDataParallel) else adapter
    eval_trainer = CausalControlTrainer(
        lm,
        module,
        optimizer,
        layout,
        activation_checkpointing=False,
        matched_weight=args.matched_weight,
        causal_weight=args.causal_weight,
        null_weight=args.null_weight,
        stale_weight=args.stale_weight,
        counterfactual_margin=args.counterfactual_margin,
        focused_counterfactual_margin=args.focused_counterfactual_margin,
    )
    metrics_path = args.run_dir / "metrics.jsonl"
    if rank == 0:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        (args.run_dir / "run_contract.json").write_text(
            json.dumps(
                {
                    "schema_version": 4,
                    "kind": "personaplex-semantic-control-stream-stage1",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "manifest": str(manifest),
                    "manifest_sha256": manifest_hash,
                    "pair_index": str(pair_index_path),
                    "pair_index_sha256": hash_file(pair_index_path),
                    "pair_certificate": str(args.pair_certificate.resolve()),
                    "pair_counts": pair_certificate.get("pairs_by_split"),
                    "model_contract": str(args.model_contract.resolve()),
                    "model_revision": contract["model_revision"],
                    "world_size": world_size,
                    "adapter_config": config.as_dict(),
                    "base_model_frozen": True,
                    "lexical_embeddings": "frozen_personaplex_text_emb",
                    "conditioning": "gated_temporal_streaming_sum_on_real_frames",
                    "loss": "matched_agent_only_text_plus_0.02_audio_and_exact_branch_token_causal_margins",
                    "contrast_alignment": "exact_dynamic_programming_longest_common_subsequence_loss_mask_only",
                    "objective_weights": {
                        "matched": args.matched_weight,
                        "causal": args.causal_weight,
                        "null": args.null_weight,
                        "stale": args.stale_weight,
                    },
                    "counterfactual_margins": {
                        "whole_response": args.counterfactual_margin,
                        "branch_distinct_tokens": args.focused_counterfactual_margin,
                    },
                    "learning_rate": args.learning_rate,
                    "optimizer_reset_on_resume": args.reset_optimizer_on_resume,
                    "train_pair_limit": args.train_pair_limit or None,
                    "train_pair_order": "deterministic_shuffle_seed_42424242",
                    "caller_stream_supervision": "forbidden",
                    "generated_audio_required_for_promotion": True,
                    "resumed_from": str(args.resume_checkpoint.resolve())
                    if args.resume_checkpoint is not None
                    else None,
                    "start_step": start_step,
                    "causal_audio_window": {
                        "history_seconds": args.history_seconds,
                        "target_tail_seconds": args.target_tail_seconds,
                        "frame_count_derived_from_each_codec_contract": True,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if world_size > 1:
        write_rank_stage(args.run_dir.parent, rank=rank, stage="pre_baseline_barrier_started")
        dist.barrier()
        write_rank_stage(args.run_dir.parent, rank=rank, stage="pre_baseline_barrier_completed")

    def write_checkpoint(step: int) -> None:
        checkpoint = args.run_dir / f"control_stream_step_{step:06d}.pt"
        torch.save(
            {
                "schema_version": 4,
                "kind": "personaplex-semantic-control-stream-adapter",
                "adapter_state_dict": module.state_dict(),
                "adapter_config": config.as_dict(),
                "model_revision": contract["model_revision"],
                "stream_layout": layout.as_dict(),
                "manifest_sha256": manifest_hash,
                "pair_index_sha256": hash_file(pair_index_path),
                "serializer_version": serializer.version,
                "step": step,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint,
        )
        evaluations = [
            (
                "heldout",
                evaluate_pairs(eval_trainer, heldout_pairs, data, max_pairs=args.eval_pairs),
            )
        ]
        if args.eval_train_pairs:
            evaluations.append(
                (
                    "train",
                    evaluate_pairs(
                        eval_trainer,
                        train_pairs,
                        data,
                        max_pairs=args.eval_train_pairs,
                    ),
                )
            )
        with metrics_path.open("a", encoding="utf-8") as handle:
            for evaluation_split, evaluation in evaluations:
                handle.write(
                    json.dumps(
                        {
                            "event": "checkpoint_evaluation",
                            "evaluation_split": evaluation_split,
                            "step": step,
                            "checkpoint": checkpoint.name,
                            **evaluation,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    if rank == 0:
        write_rank_stage(args.run_dir.parent, rank=rank, stage="baseline_checkpoint_started")
        write_checkpoint(start_step)
        write_rank_stage(args.run_dir.parent, rank=rank, stage="baseline_checkpoint_completed")
    if world_size > 1:
        write_rank_stage(args.run_dir.parent, rank=rank, stage="post_baseline_barrier_started")
        dist.barrier()
        write_rank_stage(args.run_dir.parent, rank=rank, stage="post_baseline_barrier_completed")
    adapter.train()
    for step in range(start_step, args.max_steps):
        wait_for_host_memory(
            limit=args.host_memory_limit, rank=rank, event_path=resource_events
        )
        pair = local_pairs[step % len(local_pairs)]
        left, right = pair_components(pair)
        left_id = str(left["example_id"])
        right_id = str(right["example_id"])
        left_revision = data.revision(left_id)
        right_revision = data.revision(right_id)
        stale_left = left.get("stale_example_id")
        stale_right = right.get("stale_example_id")
        example_left, example_right = data.pair_examples(left_id, right_id)
        result = trainer.step_pair(
            example_left,
            example_right,
            data.control(left_id, expected_revision=left_revision),
            data.control(right_id, expected_revision=right_revision),
            stale_a=data.control(str(stale_left), expected_revision=left_revision)
            if stale_left in data.rows
            else None,
            stale_b=data.control(str(stale_right), expected_revision=right_revision)
            if stale_right in data.rows
            else None,
        )
        values = torch.stack(
            [
                result.total.detach().double(),
                result.matched_sft.detach().double(),
                result.counterfactual_margin.detach().double(),
                result.null_margin.detach().double(),
                result.stale_margin.detach().double(),
                (result.a_cross_text - result.a_own_text).detach().double(),
                (result.b_cross_text - result.b_own_text).detach().double(),
                (
                    result.a_cross_focused_text - result.a_own_focused_text
                ).detach().double(),
                (
                    result.b_cross_focused_text - result.b_own_focused_text
                ).detach().double(),
                module.mean_gate().detach().double(),
                result.stream_regularization.detach().double().sqrt(),
                module.last_effective_gate().detach().double(),
                module.last_stream_to_lexical_rms_ratio().detach().double(),
            ]
        )
        if world_size > 1:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
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
                            "null_margin_loss": float(values[3]),
                            "stale_margin_loss": float(values[4]),
                            "a_cross_minus_own_text_nll": float(values[5]),
                            "b_cross_minus_own_text_nll": float(values[6]),
                            "a_cross_minus_own_focused_text_nll": float(values[7]),
                            "b_cross_minus_own_focused_text_nll": float(values[8]),
                            "mean_gate": float(values[9]),
                            "control_stream_rms": float(values[10]),
                            "effective_gate_abs_mean": float(values[11]),
                            "stream_to_lexical_rms_ratio": float(values[12]),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        if completed % args.checkpoint_every == 0 or completed == args.max_steps:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                write_checkpoint(completed)
            if world_size > 1:
                dist.barrier()
            adapter.train()
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        torch.save(
            {
                "schema_version": 4,
                "kind": "personaplex-semantic-control-stream-adapter",
                "adapter_state_dict": module.state_dict(),
                "adapter_config": config.as_dict(),
                "model_revision": contract["model_revision"],
                "manifest_sha256": manifest_hash,
                "pair_index_sha256": hash_file(pair_index_path),
                "step": args.max_steps,
                "optimizer_state_dict": optimizer.state_dict(),
            },
            args.run_dir / "control_stream_last.pt",
        )
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
