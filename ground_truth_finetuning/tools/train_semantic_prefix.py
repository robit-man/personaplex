"""Distributed training entry point for the frozen-LM semantic-prefix adapter."""

from __future__ import annotations

import sys
from pathlib import Path
GTFT_TOOL_ROOT = Path(__file__).resolve().parents[2]
if str(GTFT_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(GTFT_TOOL_ROOT))

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ground_truth_finetuning.training.contracts import StreamLayout, validate_control_frame_mapping
from ground_truth_finetuning.training.native_source import require_moshi_source_contract
from ground_truth_finetuning.training.plan_serializer import PlanSerializer
from ground_truth_finetuning.training.semantic_prefix import SemanticPrefixAdapter
from ground_truth_finetuning.training.trainer import SemanticPrefixTrainer


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, local_rank, world_size


def write_startup_stage(run_root: Path, *, rank: int, stage: str) -> None:
    """Persist rank-zero startup progress before the training directory exists."""
    if rank != 0:
        return
    record = {
        "at": datetime.now(timezone.utc).isoformat(),
        "kind": "personaplex-semantic-prefix-startup",
        "stage": stage,
    }
    with (run_root / "startup.jsonl").open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def require_certificate(certificate_path: Path, manifest: Path, contract: dict[str, Any]) -> None:
    certificate = json.loads(certificate_path.read_text())
    if certificate.get("kind") != "personaplex-corpus-certificate":
        raise ValueError("unexpected tensor certificate kind")
    if certificate.get("status") != "certified_for_adapter_training":
        raise ValueError("tensor corpus is not certified for adapter training")
    if certificate.get("manifest_sha256") != hash_file(manifest):
        raise ValueError("tensor certificate does not match manifest bytes")
    revisions = set(certificate.get("model_revisions", []))
    if revisions != {contract["model_revision"]}:
        raise ValueError("tensor certificate model revision does not match native model contract")
    expected_codec = {
        "mimi_weights_sha256": contract.get("mimi_weights_sha256"),
        "tokenizer_sha256": contract.get("tokenizer_sha256"),
    }
    if not all(isinstance(value, str) for value in expected_codec.values()):
        raise ValueError("native model contract lacks codec/tokenizer provenance")
    if certificate.get("codec_artifacts") != [expected_codec]:
        raise ValueError("tensor certificate codec/tokenizer provenance does not match native model contract")


def batch_for_record(
    row: dict[str, Any],
    *,
    artifact_root: Path,
    serializer: PlanSerializer,
    tokenizer: Any,
    text_cardinality: int,
    max_plan_tokens: int,
    device: torch.device,
    frame_override: dict[str, Any] | None = None,
    include_text_context: bool = True,
) -> tuple[dict[str, torch.Tensor], Any]:
    encoding = row.get("model_encoding", {})
    codes = load_tensor(artifact_root / encoding["codes_path"], "codes").unsqueeze(0).to(device)
    target_mask = load_tensor(artifact_root / encoding["target_mask_path"], "target_mask").unsqueeze(0).to(device)
    frame_value = frame_override if frame_override is not None else row.get("control", {}).get("frame")
    if not isinstance(frame_value, dict):
        raise RuntimeError(f"{row.get('example_id')}: missing control.frame")
    frame = validate_control_frame_mapping(frame_value)
    token_ids = serializer.encode_frame(
        frame,
        tokenizer,
        text_cardinality,
        include_text_context=include_text_context,
    )[:max_plan_tokens]
    if not token_ids:
        raise RuntimeError(f"{row.get('example_id')}: control frame encoded to no SentencePiece tokens")
    plan_ids = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(0)
    return {
        "control_token_ids": plan_ids,
        "control_attention_mask": torch.ones_like(plan_ids, dtype=torch.bool),
        "codes": codes,
        "agent_target_mask": target_mask,
        "prefix_at": torch.tensor([int(encoding["prefix_at"])], device=device),
    }, frame


def evaluate_checkpoint(
    trainer: SemanticPrefixTrainer,
    records: list[dict[str, Any]],
    *,
    artifact_root: Path,
    serializer: PlanSerializer,
    tokenizer: Any,
    text_cardinality: int,
    max_plan_tokens: int,
    device: torch.device,
    audio_weight: float,
    max_examples: int,
) -> dict[str, Any]:
    selected = records[:max_examples]
    if not selected:
        raise RuntimeError("held-out checkpoint evaluation requires at least one record")
    adapter_was_training = trainer.adapter.training
    trainer.adapter.eval()
    correct_losses: list[float] = []
    shuffled_losses: list[float] = []
    context_ablated_losses: list[float] = []
    terminal_losses: list[float] = []
    context_examples = 0
    try:
        for index, row in enumerate(selected):
            batch, frame = batch_for_record(
                row,
                artifact_root=artifact_root,
                serializer=serializer,
                tokenizer=tokenizer,
                text_cardinality=text_cardinality,
                max_plan_tokens=max_plan_tokens,
                device=device,
            )
            correct = trainer.evaluate(batch, audio_weight=audio_weight)
            correct_losses.append(float(correct.total.detach().cpu()))
            shuffled_frame = selected[(index + 1) % len(selected)].get("control", {}).get("frame")
            shuffled_batch, _ = batch_for_record(
                row,
                artifact_root=artifact_root,
                serializer=serializer,
                tokenizer=tokenizer,
                text_cardinality=text_cardinality,
                max_plan_tokens=max_plan_tokens,
                device=device,
                frame_override=shuffled_frame,
            )
            shuffled = trainer.evaluate(shuffled_batch, audio_weight=audio_weight)
            shuffled_losses.append(float(shuffled.total.detach().cpu()))
            text_context = frame.state.get("textContext") if isinstance(frame.state, dict) else None
            if isinstance(text_context, dict) and text_context.get("turns"):
                ablated_batch, _ = batch_for_record(
                    row,
                    artifact_root=artifact_root,
                    serializer=serializer,
                    tokenizer=tokenizer,
                    text_cardinality=text_cardinality,
                    max_plan_tokens=max_plan_tokens,
                    device=device,
                    include_text_context=False,
                )
                ablated = trainer.evaluate(ablated_batch, audio_weight=audio_weight)
                context_ablated_losses.append(float(ablated.total.detach().cpu()))
                context_examples += 1
            if frame.state.get("endCallAuthorized") is True:
                terminal_losses.append(float(correct.total.detach().cpu()))
    finally:
        trainer.adapter.train(adapter_was_training)

    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    control_loss = mean(correct_losses)
    shuffled_loss = mean(shuffled_losses)
    ablated_loss = mean(context_ablated_losses)
    return {
        "schema_version": 1,
        "kind": "personaplex-semantic-prefix-heldout-evaluation",
        "examples": len(correct_losses),
        "control_loss": control_loss,
        "shuffled_control_loss": shuffled_loss,
        "plan_sensitivity_delta": (shuffled_loss - control_loss) if control_loss is not None and shuffled_loss is not None else None,
        "text_context_examples": context_examples,
        "text_context_ablated_loss": ablated_loss,
        "text_context_delta": (ablated_loss - control_loss) if control_loss is not None and ablated_loss is not None else None,
        "terminal_examples": len(terminal_losses),
        "terminal_control_loss": mean(terminal_losses),
        "metric_scope": "teacher_forced_native_agent_loss; expressive wording remains separately evaluated by the runtime control harness",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--prefix-frames", type=int, default=16)
    parser.add_argument("--max-plan-tokens", type=int, default=512)
    parser.add_argument("--audio-weight", type=float, default=0.02)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-examples", type=int, default=32)
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_plan_tokens < 1 or args.checkpoint_every < 1 or args.eval_examples < 1:
        raise SystemExit("max-steps, max-plan-tokens, checkpoint-every, and eval-examples must be positive")
    rank, local_rank, world_size = rank_info()
    if not torch.cuda.is_available():
        raise SystemExit("semantic-prefix training requires CUDA; CPU fallback is deliberately disabled")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    manifest = args.manifest.resolve()
    artifact_root = args.artifact_root.resolve()
    contract = json.loads(args.model_contract.read_text())
    require_certificate(args.certificate.resolve(), manifest, contract)
    require_moshi_source_contract(args.moshi_source_root.resolve(), contract)
    layout = StreamLayout.from_mapping(contract["stream_layout"])
    expected_weights_hash = contract.get("moshi_weights_sha256")
    write_startup_stage(args.run_dir.parent, rank=rank, stage="hashing_frozen_moshi_weights")
    actual_weights_hash = hash_file(args.moshi_path.resolve()) if rank == 0 else None
    if world_size > 1:
        shared_hash = [actual_weights_hash]
        dist.broadcast_object_list(shared_hash, src=0)
        actual_weights_hash = shared_hash[0]
    if expected_weights_hash != actual_weights_hash:
        raise SystemExit("LM weights do not match the inspected native model contract")
    if contract.get("tokenizer_sha256") != hash_file(args.tokenizer_path.resolve()):
        raise SystemExit("SentencePiece tokenizer does not match the inspected native model contract")
    sys.path.insert(0, str(args.moshi_source_root.resolve()))
    import sentencepiece
    from moshi.models.loaders import get_moshi_lm

    write_startup_stage(args.run_dir.parent, rank=rank, stage="loading_frozen_moshi")
    lm = get_moshi_lm(args.moshi_path.resolve(), device=device, dtype=torch.bfloat16)
    layout.validate_for_model(lm)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer_path.resolve()))
    write_startup_stage(args.run_dir.parent, rank=rank, stage="initializing_semantic_prefix")
    adapter = SemanticPrefixAdapter(
        text_cardinality=int(lm.text_card),
        hidden_size=int(lm.dim),
        prefix_frames=args.prefix_frames,
    ).to(device)
    if world_size > 1:
        adapter = DistributedDataParallel(adapter, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=0.01)
    trainer = SemanticPrefixTrainer(lm, adapter, optimizer, layout, activation_checkpointing=True)
    records = [row for row in load_jsonl(manifest) if row.get("split") == "train"]
    heldout_records = [row for row in load_jsonl(manifest) if row.get("split") in {"validation", "test"}]
    if len(records) < world_size:
        raise SystemExit("certified train split has fewer examples than requested distributed world size")
    if not heldout_records:
        raise SystemExit("certified corpus requires validation or test records for checkpoint evaluation")
    local_records = records[rank::world_size]
    if not local_records:
        raise SystemExit(f"rank {rank} received no training examples")
    if rank == 0:
        write_startup_stage(args.run_dir.parent, rank=rank, stage="writing_run_contract")
        args.run_dir.mkdir(parents=True, exist_ok=False)
        (args.run_dir / "run_contract.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "personaplex-semantic-prefix-run",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "manifest": str(manifest),
                    "manifest_sha256": hash_file(manifest),
                    "certificate": str(args.certificate.resolve()),
                    "model_contract": str(args.model_contract.resolve()),
                    "model_revision": contract["model_revision"],
                    "world_size": world_size,
                    "prefix_frames": args.prefix_frames,
                    "checkpoint_every": args.checkpoint_every,
                    "eval_examples": args.eval_examples,
                    "caller_stream_supervision": "forbidden",
                    "model_integrity_verification": "rank_zero_sha256_broadcast",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    if world_size > 1:
        dist.barrier()
    serializer = PlanSerializer()
    metrics_path = args.run_dir / "metrics.jsonl"
    module = adapter.module if isinstance(adapter, DistributedDataParallel) else adapter
    evaluation_trainer = SemanticPrefixTrainer(lm, module, optimizer, layout, activation_checkpointing=False)

    def write_checkpoint(step: int) -> None:
        checkpoint_path = args.run_dir / f"adapter_step_{step:06d}.pt"
        torch.save(
            {
                "adapter_state_dict": module.state_dict(),
                "model_revision": contract["model_revision"],
                "stream_layout": layout.as_dict(),
                "manifest_sha256": hash_file(manifest),
                "prefix_frames": args.prefix_frames,
                "serializer_version": serializer.version,
                "step": step,
            },
            checkpoint_path,
        )
        evaluation = evaluate_checkpoint(
            evaluation_trainer,
            heldout_records,
            artifact_root=artifact_root,
            serializer=serializer,
            tokenizer=tokenizer,
            text_cardinality=int(lm.text_card),
            max_plan_tokens=args.max_plan_tokens,
            device=device,
            audio_weight=args.audio_weight,
            max_examples=args.eval_examples,
        )
        with metrics_path.open("a") as output:
            output.write(json.dumps({"event": "checkpoint_evaluation", "step": step, "checkpoint": checkpoint_path.name, **evaluation}, sort_keys=True) + "\n")

    if rank == 0:
        write_checkpoint(0)
    if world_size > 1:
        dist.barrier()
    for step in range(args.max_steps):
        row = local_records[step % len(local_records)]
        batch, _ = batch_for_record(
            row,
            artifact_root=artifact_root,
            serializer=serializer,
            tokenizer=tokenizer,
            text_cardinality=int(lm.text_card),
            max_plan_tokens=args.max_plan_tokens,
            device=device,
        )
        loss = trainer.step(
            batch,
            audio_weight=args.audio_weight,
        )
        values = torch.stack(
            [
                loss.total.detach().to(torch.float64),
                loss.text.detach().to(torch.float64),
                loss.audio.detach().to(torch.float64),
                torch.tensor(loss.text_tokens, device=device, dtype=torch.float64),
                torch.tensor(loss.audio_tokens, device=device, dtype=torch.float64),
            ]
        )
        if world_size > 1:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values[:3] /= world_size
        if rank == 0:
            with metrics_path.open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "step": step + 1,
                            "loss": float(values[0]),
                            "text_loss": float(values[1]),
                            "audio_loss": float(values[2]),
                            "text_tokens_per_rank": int(values[3] / world_size),
                            "audio_tokens_per_rank": int(values[4] / world_size),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        completed_step = step + 1
        if completed_step % args.checkpoint_every == 0 or completed_step == args.max_steps:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                write_checkpoint(completed_step)
            if world_size > 1:
                dist.barrier()
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        torch.save(
            {
                "adapter_state_dict": module.state_dict(),
                "model_revision": contract["model_revision"],
                "stream_layout": layout.as_dict(),
                "manifest_sha256": hash_file(manifest),
            },
            args.run_dir / "adapter_last.pt",
        )
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
