"""Distributed training entry point for the frozen-LM semantic-prefix adapter."""

from __future__ import annotations

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

from ground_truth_finetuning.training.contracts import StreamLayout, canonical_json
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
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_plan_tokens < 1:
        raise SystemExit("max-steps and max-plan-tokens must be positive")
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
    layout = StreamLayout.from_mapping(contract["stream_layout"])
    expected_weights_hash = contract.get("moshi_weights_sha256")
    if expected_weights_hash != hash_file(args.moshi_path.resolve()):
        raise SystemExit("LM weights do not match the inspected native model contract")
    sys.path.insert(0, str(args.moshi_source_root.resolve()))
    import sentencepiece
    from moshi.models.loaders import get_moshi_lm

    lm = get_moshi_lm(args.moshi_path.resolve(), device=device, dtype=torch.bfloat16)
    layout.validate_for_model(lm)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer_path.resolve()))
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
    if len(records) < world_size:
        raise SystemExit("certified train split has fewer examples than requested distributed world size")
    local_records = records[rank::world_size]
    if not local_records:
        raise SystemExit(f"rank {rank} received no training examples")
    if rank == 0:
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
                    "caller_stream_supervision": "forbidden",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    if world_size > 1:
        dist.barrier()
    metrics_path = args.run_dir / "metrics.jsonl"
    for step in range(args.max_steps):
        row = local_records[step % len(local_records)]
        encoding = row.get("model_encoding", {})
        codes = load_tensor(artifact_root / encoding["codes_path"], "codes").unsqueeze(0).to(device)
        target_mask = load_tensor(artifact_root / encoding["target_mask_path"], "target_mask").unsqueeze(0).to(device)
        plan = row.get("semantics", {}).get("plan")
        token_ids = tokenizer.encode(canonical_json(plan))[: args.max_plan_tokens]
        if not token_ids:
            raise RuntimeError(f"{row.get('example_id')}: typed plan encoded to no SentencePiece tokens")
        plan_ids = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(0)
        attention = torch.ones_like(plan_ids, dtype=torch.bool)
        loss = trainer.step(
            {
                "plan_token_ids": plan_ids,
                "plan_attention_mask": attention,
                "codes": codes,
                "agent_target_mask": target_mask,
                "prefix_at": torch.tensor([int(encoding["prefix_at"])], device=device),
            },
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
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        module = adapter.module if isinstance(adapter, DistributedDataParallel) else adapter
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
