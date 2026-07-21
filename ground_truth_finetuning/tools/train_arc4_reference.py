#!/usr/bin/env python3
"""Train a gated ARC-4 reference adapter against native agent-only targets."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.arc4_conditioning import (  # noqa: E402
    Arc4InjectionConfig,
    Arc4ReferenceTrainer,
    GatedArc4InjectionAdapter,
)
from ground_truth_finetuning.training.contracts import (  # noqa: E402
    StreamLayout,
    validate_control_frame_mapping,
)
from ground_truth_finetuning.training.native_source import require_moshi_source_contract  # noqa: E402
from ground_truth_finetuning.training.plan_serializer import PlanSerializer  # noqa: E402
from ground_truth_finetuning.training.semantic_prefix import SemanticPrefixAdapter  # noqa: E402


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("ARC-4 native manifest contains no rows")
    return rows


def load_tensor(path: Path, name: str) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    value = payload.get(name) if isinstance(payload, Mapping) else payload
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} does not contain tensor {name!r}")
    return value


def crop_native_turn(
    codes: torch.Tensor,
    target_mask: torch.Tensor,
    prefix_at: int,
    *,
    context_frames: int,
    post_target_tail_frames: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Retain recent duplex context and the complete supervised response."""
    if codes.ndim != 2 or target_mask.shape != codes.shape or target_mask.dtype != torch.bool:
        raise ValueError("native codes/target mask must match [codebooks, frames]")
    if not 0 <= prefix_at < codes.shape[1]:
        raise ValueError("prefix_at is outside native code frames")
    positions = target_mask.any(dim=0).nonzero().flatten()
    if positions.numel() < 1:
        raise ValueError("native example has no agent target frames")
    first_target = int(positions[0])
    last_target = int(positions[-1])
    if first_target < prefix_at:
        raise ValueError("agent target begins before the control boundary")
    start = max(0, prefix_at - context_frames)
    end = min(codes.shape[1], last_target + 1 + post_target_tail_frames)
    if end <= prefix_at:
        raise ValueError("native crop contains no post-control frames")
    cropped_codes = codes[:, start:end].contiguous()
    cropped_mask = target_mask[:, start:end].contiguous()
    if int(cropped_mask.sum()) != int(target_mask.sum()):
        raise ValueError("native crop would remove supervised agent targets")
    return cropped_codes, cropped_mask, prefix_at - start


def rank_info() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def rank_log(rank: int, phase: str, **values: Any) -> None:
    print(
        json.dumps({"rank": rank, "phase": phase, **values}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def host_memory_used_fraction() -> float:
    values: dict[str, int] = {}
    with Path("/proc/meminfo").open(encoding="ascii") as handle:
        for line in handle:
            name, raw, *_rest = line.split()
            values[name.rstrip(":")] = int(raw)
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    if total <= 0:
        raise RuntimeError("cannot discover host memory from /proc/meminfo")
    return 1.0 - available / total


def wait_for_host_memory(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    limit: float,
    poll_seconds: float,
) -> None:
    while True:
        over = host_memory_used_fraction() >= limit if rank == 0 else False
        flag = torch.tensor([int(over)], device=device, dtype=torch.uint8)
        if world_size > 1:
            dist.broadcast(flag, src=0)
        if not bool(flag.item()):
            return
        if rank == 0:
            rank_log(rank, "host_memory_throttle", usedFraction=host_memory_used_fraction(), limit=limit)
        time.sleep(poll_seconds)


def load_control_adapter(
    path: Path,
    lm: Any,
    device: torch.device,
    prefix_frames: int,
) -> tuple[SemanticPrefixAdapter, str]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    state = payload.get("adapter_state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(state, Mapping):
        raise ValueError("control checkpoint has no adapter_state_dict")
    adapter = SemanticPrefixAdapter(
        text_cardinality=int(lm.text_card),
        hidden_size=int(lm.dim),
        prefix_frames=prefix_frames,
    ).to(device=device, dtype=next(lm.parameters()).dtype)
    adapter.load_state_dict(state, strict=True)
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter, hash_file(path)


class Arc4TensorReader:
    def __init__(self, root: Path, device: torch.device, hidden_size: int) -> None:
        self.root = root
        self.device = device
        self.hidden_size = hidden_size
        self._verified_shards: dict[Path, str] = {}

    def load(self, row: Mapping[str, Any]) -> torch.Tensor:
        binding = row.get("arc4_reference")
        if not isinstance(binding, Mapping):
            raise ValueError("joined example lacks arc4_reference")
        path = (self.root / str(binding["shard_path"])).resolve()
        if self.root not in path.parents:
            raise ValueError("ARC-4 shard escapes certified root")
        expected = str(binding["shard_sha256"])
        if path not in self._verified_shards:
            self._verified_shards[path] = hash_file(path)
        actual = self._verified_shards[path]
        if expected != actual:
            raise ValueError(f"ARC-4 shard hash mismatch: {path}")
        with safe_open(str(path), framework="pt", device=str(self.device)) as handle:
            tensor = handle.get_tensor(str(binding["tensor_key"]))
        if tensor.ndim != 2 or tensor.shape[0] < 1 or tensor.shape[1] != self.hidden_size:
            raise ValueError(f"invalid ARC-4 training tensor {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError("ARC-4 training tensor contains non-finite values")
        return tensor.unsqueeze(0)


def batch_for_row(
    row: Mapping[str, Any],
    *,
    artifact_root: Path,
    serializer: PlanSerializer,
    tokenizer: Any,
    text_cardinality: int,
    max_control_tokens: int,
    context_frames: int,
    post_target_tail_frames: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    encoding = row.get("model_encoding")
    control = row.get("control")
    if not isinstance(encoding, Mapping) or not isinstance(control, Mapping):
        raise ValueError("joined example lacks model_encoding/control")
    frame = validate_control_frame_mapping(control["frame"])
    token_ids = serializer.encode_frame(frame, tokenizer, text_cardinality)[:max_control_tokens]
    if not token_ids:
        raise ValueError("control frame encoded to no tokens")
    ids = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(0)
    codes = load_tensor(artifact_root / str(encoding["codes_path"]), "codes")
    target_mask = load_tensor(
        artifact_root / str(encoding["target_mask_path"]),
        "target_mask",
    )
    codes, target_mask, cropped_prefix_at = crop_native_turn(
        codes,
        target_mask,
        int(encoding["prefix_at"]),
        context_frames=context_frames,
        post_target_tail_frames=post_target_tail_frames,
    )
    return {
        "control_token_ids": ids,
        "control_attention_mask": torch.ones_like(ids, dtype=torch.bool),
        "codes": codes.unsqueeze(0).to(device),
        "agent_target_mask": target_mask.unsqueeze(0).to(device),
        "prefix_at": torch.tensor([cropped_prefix_at], device=device),
    }


def wrong_reference_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        cf = row.get("counterfactual")
        if isinstance(cf, Mapping) and cf.get("groupId"):
            groups[str(cf["groupId"])].append(row)
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        example_id = str(row["example_id"])
        cf = row.get("counterfactual")
        candidates: list[dict[str, Any]] = []
        if isinstance(cf, Mapping) and cf.get("groupId"):
            candidates = [
                candidate
                for candidate in groups[str(cf["groupId"])]
                if candidate["example_id"] != example_id
                and candidate.get("counterfactual", {}).get("branchId") != cf.get("branchId")
            ]
        if not candidates:
            candidates = [
                rows[offset % len(rows)]
                for offset in range(index + 1, index + len(rows))
                if rows[offset % len(rows)]["arc4_reference"]["reference_hash"]
                != row["arc4_reference"]["reference_hash"]
            ][:1]
        if not candidates:
            raise ValueError(f"no wrong-reference candidate for {example_id}")
        result[example_id] = candidates[0]
    return result


@torch.no_grad()
def evaluate(
    trainer: Arc4ReferenceTrainer,
    rows: list[dict[str, Any]],
    wrong_rows: Mapping[str, dict[str, Any]],
    *,
    reader: Arc4TensorReader,
    artifact_root: Path,
    serializer: PlanSerializer,
    tokenizer: Any,
    text_cardinality: int,
    max_control_tokens: int,
    context_frames: int,
    post_target_tail_frames: int,
    device: torch.device,
    max_examples: int,
) -> dict[str, Any]:
    matched: list[float] = []
    wrong: list[float] = []
    directions = 0
    count = min(max_examples, len(rows))
    selected = [rows[(index * len(rows)) // count] for index in range(count)]
    for row in selected:
        batch = batch_for_row(
            row,
            artifact_root=artifact_root,
            serializer=serializer,
            tokenizer=tokenizer,
            text_cardinality=text_cardinality,
            max_control_tokens=max_control_tokens,
            context_frames=context_frames,
            post_target_tail_frames=post_target_tail_frames,
            device=device,
        )
        right_value = float(trainer.evaluate(batch, reader.load(row)).total)
        wrong_row = wrong_rows[str(row["example_id"])]
        wrong_value = float(trainer.evaluate(batch, reader.load(wrong_row)).total)
        matched.append(right_value)
        wrong.append(wrong_value)
        directions += int(right_value < wrong_value)
    mean = lambda values: sum(values) / len(values)
    return {
        "examples": len(matched),
        "matchedLoss": mean(matched),
        "wrongReferenceLoss": mean(wrong),
        "sensitivityDelta": mean(wrong) - mean(matched),
        "correctCausalDirections": directions,
        "causalDirectionRate": directions / len(matched),
        "scope": "teacher-forced-agent-only; generated duplex evaluation still required",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--arc4-root", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--control-adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-examples", type=int, default=32)
    parser.add_argument("--max-control-tokens", type=int, default=512)
    parser.add_argument("--context-frames", type=int, default=256)
    parser.add_argument("--post-target-tail-frames", type=int, default=16)
    parser.add_argument("--prefix-frames", type=int, default=16)
    parser.add_argument("--adapter-rank", type=int, default=128)
    parser.add_argument("--initial-gate", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--ranking-every", type=int, default=2)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--control-dropout-every", type=int, default=10)
    parser.add_argument("--max-host-memory-fraction", type=float, default=0.80)
    parser.add_argument("--memory-poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    positive = [args.max_steps, args.checkpoint_every, args.eval_examples, args.max_control_tokens, args.context_frames, args.prefix_frames, args.adapter_rank, args.ranking_every, args.control_dropout_every]
    if min(positive) < 1:
        raise SystemExit("step, size, rank, ranking, and dropout intervals must be positive")
    if not 0.0 < args.max_host_memory_fraction < 1.0 or args.memory_poll_seconds <= 0:
        raise SystemExit("host-memory fraction/poll settings are invalid")
    if args.post_target_tail_frames < 0:
        raise SystemExit("post-target tail frames must be non-negative")

    rank, local_rank, world_size = rank_info()
    rank_log(rank, "process_started", localRank=local_rank, worldSize=world_size)
    if not torch.cuda.is_available():
        raise SystemExit("ARC-4 training requires CUDA; CPU fallback is prohibited")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        rank_log(rank, "nccl_init_started")
        dist.init_process_group("nccl")
        rank_log(rank, "nccl_init_completed")
    device = torch.device("cuda", local_rank)

    manifest = args.manifest.resolve()
    certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
    if certificate.get("status") != "certified_for_arc4_adapter_training":
        raise SystemExit("ARC-4 native certificate does not authorize training")
    if certificate.get("joinedManifestSha256") != hash_file(manifest):
        raise SystemExit("ARC-4 native certificate does not match manifest")
    contract = json.loads(args.model_contract.read_text(encoding="utf-8"))
    require_moshi_source_contract(args.moshi_source_root.resolve(), contract)
    if contract.get("moshi_weights_sha256") != hash_file(args.moshi_path.resolve()):
        raise SystemExit("Moshi weights do not match native model contract")
    # The Torch 2.4 Inductor-wrapped RoPE backward is incompatible with
    # non-reentrant activation checkpoint RNG restoration. Training remains
    # entirely on CUDA, but uses the deterministic eager kernels.
    os.environ.setdefault("NO_TORCH_COMPILE", "1")
    os.environ.setdefault("NO_CUDA_GRAPH", "1")
    sys.path.insert(0, str(args.moshi_source_root.resolve()))
    import sentencepiece
    from moshi.models.loaders import get_moshi_lm

    rank_log(rank, "lm_load_started")
    lm = get_moshi_lm(args.moshi_path.resolve(), device=device, dtype=torch.bfloat16)
    rank_log(rank, "lm_load_completed", allocatedBytes=torch.cuda.memory_allocated(device))
    layout = StreamLayout.from_mapping(contract["stream_layout"])
    layout.validate_for_model(lm)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer_path.resolve()))
    rank_log(rank, "control_adapter_load_started")
    control_adapter, control_adapter_hash = load_control_adapter(
        args.control_adapter_checkpoint.resolve(),
        lm,
        device,
        args.prefix_frames,
    )
    rank_log(rank, "control_adapter_load_completed", allocatedBytes=torch.cuda.memory_allocated(device))
    config = Arc4InjectionConfig(
        hidden_size=int(lm.dim),
        rank=args.adapter_rank,
        initial_gate=args.initial_gate,
    )
    adapter: torch.nn.Module = GatedArc4InjectionAdapter(config).to(device=device)
    if world_size > 1:
        rank_log(rank, "ddp_wrap_started")
        adapter = DistributedDataParallel(adapter, device_ids=[local_rank], output_device=local_rank)
        rank_log(rank, "ddp_wrap_completed")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=0.01)
    trainer = Arc4ReferenceTrainer(lm, control_adapter, adapter, optimizer, layout)
    reader = Arc4TensorReader(args.arc4_root.resolve(), device, int(lm.dim))
    serializer = PlanSerializer()
    rows = load_jsonl(manifest)
    train_rows = [row for row in rows if row.get("split") == "train"]
    heldout_rows = [row for row in rows if row.get("split") in {"validation", "test"}]
    if len(train_rows) < world_size or not heldout_rows:
        raise SystemExit("ARC-4 native corpus lacks distributed train/held-out rows")
    local_rows = train_rows[rank::world_size]
    all_wrong = wrong_reference_rows(rows)
    rank_log(rank, "corpus_ready", localTrainRows=len(local_rows), heldoutRows=len(heldout_rows))

    if rank == 0:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        (args.run_dir / "run_contract.json").write_text(
            json.dumps(
                {
                    "schema": "personaplex.arc4-training-run.v1",
                    "createdAt": datetime.now(timezone.utc).isoformat(),
                    "manifest": str(manifest),
                    "manifestSha256": hash_file(manifest),
                    "certificate": str(args.certificate.resolve()),
                    "modelRevision": contract["model_revision"],
                    "conditionerRevision": certificate["conditionerRevision"],
                    "controlAdapterCheckpoint": str(args.control_adapter_checkpoint.resolve()),
                    "controlAdapterSha256": control_adapter_hash,
                    "adapterConfig": config.as_dict(),
                    "worldSize": world_size,
                    "nativeCrop": {
                        "contextFrames": args.context_frames,
                        "postTargetTailFrames": args.post_target_tail_frames,
                        "completeAgentTargetRequired": True,
                    },
                    "hostMemoryThrottleFraction": args.max_host_memory_fraction,
                    "callerStreamSupervision": "forbidden",
                    "targetTextPassedToConditioner": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if world_size > 1:
        rank_log(rank, "initial_barrier_started")
        dist.barrier()
        rank_log(rank, "initial_barrier_completed")
    metrics_path = args.run_dir / "metrics.jsonl"
    module = adapter.module if isinstance(adapter, DistributedDataParallel) else adapter
    evaluator = Arc4ReferenceTrainer(lm, control_adapter, module, optimizer, layout, activation_checkpointing=False)

    def checkpoint(step: int) -> None:
        path = args.run_dir / f"arc4_adapter_step_{step:06d}.pt"
        torch.save(
            {
                "arc4_adapter_state_dict": module.state_dict(),
                "arc4_adapter_config": config.as_dict(),
                "model_revision": contract["model_revision"],
                "conditioner_revision": certificate["conditionerRevision"],
                "control_adapter_checkpoint_sha256": control_adapter_hash,
                "manifest_sha256": hash_file(manifest),
                "step": step,
            },
            path,
        )
        report = evaluate(
            evaluator,
            heldout_rows,
            all_wrong,
            reader=reader,
            artifact_root=args.artifact_root.resolve(),
            serializer=serializer,
            tokenizer=tokenizer,
            text_cardinality=int(lm.text_card),
            max_control_tokens=args.max_control_tokens,
            context_frames=args.context_frames,
            post_target_tail_frames=args.post_target_tail_frames,
            device=device,
            max_examples=args.eval_examples,
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "checkpoint", "step": step, "checkpoint": path.name, **report}, sort_keys=True) + "\n")

    if rank == 0:
        checkpoint(0)
    if world_size > 1:
        dist.barrier()
    for step_index in range(args.max_steps):
        wait_for_host_memory(
            rank=rank,
            world_size=world_size,
            device=device,
            limit=args.max_host_memory_fraction,
            poll_seconds=args.memory_poll_seconds,
        )
        row = local_rows[step_index % len(local_rows)]
        batch = batch_for_row(
            row,
            artifact_root=args.artifact_root.resolve(),
            serializer=serializer,
            tokenizer=tokenizer,
            text_cardinality=int(lm.text_card),
            max_control_tokens=args.max_control_tokens,
            context_frames=args.context_frames,
            post_target_tail_frames=args.post_target_tail_frames,
            device=device,
        )
        use_ranking = (step_index + 1) % args.ranking_every == 0
        wrong_reference = reader.load(all_wrong[str(row["example_id"])]) if use_ranking else None
        drop_condition = (step_index + 1) % args.control_dropout_every == 0
        metrics = trainer.step(
            batch,
            reader.load(row),
            wrong_reference=wrong_reference,
            ranking_margin=args.ranking_margin,
            ranking_weight=args.ranking_weight,
            drop_condition=drop_condition,
        )
        values = torch.tensor(
            [metrics.total, metrics.matched, metrics.text, metrics.audio, metrics.ranking, metrics.gate],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= world_size
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "step": step_index + 1,
                            "loss": float(values[0]),
                            "matchedLoss": float(values[1]),
                            "textLoss": float(values[2]),
                            "audioLoss": float(values[3]),
                            "rankingLoss": float(values[4]),
                            "gate": float(values[5]),
                            "rankingScheduled": use_ranking,
                            "controlDropped": drop_condition,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        if (step_index + 1) % args.checkpoint_every == 0 or step_index + 1 == args.max_steps:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                checkpoint(step_index + 1)
            if world_size > 1:
                dist.barrier()
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
