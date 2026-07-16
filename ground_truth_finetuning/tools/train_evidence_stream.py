"""Train the frozen-prefix, learned delayed-evidence adapter on native code shards.

The input manifest must contain certified V4/V7 examples with ``model_encoding``
plus typed ``controlFrame`` and ``evidenceFrame`` mappings.  The control prefix
is loaded from an accepted first-stage checkpoint and remains frozen.  Only the
evidence stream adapter receives gradients, and the native target loss is masked
to the agent text/audio streams.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.contracts import (  # noqa: E402
    EvidenceTrainingFrame,
    StreamLayout,
    assert_evidence_control_alignment,
    validate_control_frame_mapping,
    validate_evidence_frame_mapping,
)
from ground_truth_finetuning.training.evidence_conditioning import EvidenceStreamAdapter  # noqa: E402
from ground_truth_finetuning.training.native_source import require_moshi_source_contract  # noqa: E402
from ground_truth_finetuning.training.plan_serializer import PlanSerializer  # noqa: E402
from ground_truth_finetuning.training.semantic_prefix import SemanticPrefixAdapter  # noqa: E402
from ground_truth_finetuning.training.trainer import EvidenceStreamTrainer  # noqa: E402


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("encoded evidence manifest contains no rows")
    return rows


def load_tensor(path: Path, name: str) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    value = payload.get(name) if isinstance(payload, dict) else payload
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} does not contain tensor {name!r}")
    return value


def rank_info() -> tuple[int, int, int]:
    return (
        int(os.environ.get("RANK", "0")),
        int(os.environ.get("LOCAL_RANK", "0")),
        int(os.environ.get("WORLD_SIZE", "1")),
    )


def require_certificate(certificate_path: Path, manifest: Path, contract: Mapping[str, Any]) -> None:
    certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
    if certificate.get("kind") != "personaplex-corpus-certificate":
        raise ValueError("unexpected tensor certificate kind")
    if certificate.get("status") != "certified_for_adapter_training":
        raise ValueError("tensor corpus is not certified for adapter training")
    if certificate.get("manifest_sha256") != hash_file(manifest):
        raise ValueError("tensor certificate does not match encoded-manifest bytes")
    if set(certificate.get("model_revisions", [])) != {contract["model_revision"]}:
        raise ValueError("tensor certificate model revision does not match native model contract")
    expected_codec = {
        "mimi_weights_sha256": contract.get("mimi_weights_sha256"),
        "tokenizer_sha256": contract.get("tokenizer_sha256"),
    }
    if not all(isinstance(value, str) for value in expected_codec.values()):
        raise ValueError("native model contract lacks codec/tokenizer provenance")
    if certificate.get("codec_artifacts") != [expected_codec]:
        raise ValueError("tensor certificate codec/tokenizer provenance does not match native model contract")


def frame_mappings(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    control = row.get("controlFrame")
    evidence = row.get("evidenceFrame")
    if not isinstance(control, Mapping):
        control = (row.get("control") or {}).get("frame") if isinstance(row.get("control"), Mapping) else None
    if not isinstance(evidence, Mapping):
        evidence = (row.get("control") or {}).get("evidence") if isinstance(row.get("control"), Mapping) else None
    if not isinstance(control, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError(f"{row.get('exampleId', row.get('example_id', 'unknown'))}: control/evidence frames are required")
    return control, evidence


def typed_frames(row: Mapping[str, Any]):
    control_mapping, evidence_mapping = frame_mappings(row)
    control = validate_control_frame_mapping(control_mapping)
    evidence = validate_evidence_frame_mapping(evidence_mapping)
    assert_evidence_control_alignment(control, evidence)
    return control, evidence


def encode_evidence_tokens(
    row: Mapping[str, Any],
    *,
    serializer: PlanSerializer,
    tokenizer: Any,
    text_cardinality: int,
    max_evidence_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, EvidenceTrainingFrame]:
    control, evidence = typed_frames(row)
    token_ids = serializer.encode_evidence(evidence, control, tokenizer, text_cardinality)[:max_evidence_tokens]
    if not token_ids:
        raise ValueError("evidence serialization produced no tokens")
    ids = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(0)
    return ids, torch.ones_like(ids, dtype=torch.bool), evidence


def batch_for_record(
    row: Mapping[str, Any],
    *,
    artifact_root: Path,
    serializer: PlanSerializer,
    tokenizer: Any,
    text_cardinality: int,
    max_control_tokens: int,
    max_evidence_tokens: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], Any, Any]:
    encoding = row.get("model_encoding")
    if not isinstance(encoding, Mapping):
        raise ValueError("encoded evidence example has no model_encoding")
    codes = load_tensor(artifact_root / str(encoding["codes_path"]), "codes").unsqueeze(0).to(device)
    target_mask = load_tensor(artifact_root / str(encoding["target_mask_path"]), "target_mask").unsqueeze(0).to(device)
    control, evidence = typed_frames(row)
    control_ids = serializer.encode_frame(control, tokenizer, text_cardinality)[:max_control_tokens]
    if not control_ids:
        raise ValueError("control serialization produced no tokens")
    evidence_ids, evidence_mask, _ = encode_evidence_tokens(
        row,
        serializer=serializer,
        tokenizer=tokenizer,
        text_cardinality=text_cardinality,
        max_evidence_tokens=max_evidence_tokens,
        device=device,
    )
    control_ids_tensor = torch.tensor(control_ids, device=device, dtype=torch.long).unsqueeze(0)
    return {
        "control_token_ids": control_ids_tensor,
        "control_attention_mask": torch.ones_like(control_ids_tensor, dtype=torch.bool),
        "evidence_token_ids": evidence_ids,
        "evidence_attention_mask": evidence_mask,
        "codes": codes,
        "agent_target_mask": target_mask,
        "prefix_at": torch.tensor([int(encoding["prefix_at"])], device=device),
    }, control, evidence


@torch.no_grad()
def evaluate_checkpoint(
    trainer: EvidenceStreamTrainer,
    rows: list[dict[str, Any]],
    *,
    artifact_root: Path,
    serializer: PlanSerializer,
    tokenizer: Any,
    text_cardinality: int,
    max_control_tokens: int,
    max_evidence_tokens: int,
    device: torch.device,
    max_examples: int,
) -> dict[str, Any]:
    selected = rows[:max_examples]
    if not selected:
        raise ValueError("held-out evidence evaluation requires at least one example")
    right_losses: list[float] = []
    wrong_losses: list[float] = []
    evidence_pairs = 0
    for index, row in enumerate(selected):
        batch, _control, evidence = batch_for_record(
            row,
            artifact_root=artifact_root,
            serializer=serializer,
            tokenizer=tokenizer,
            text_cardinality=text_cardinality,
            max_control_tokens=max_control_tokens,
            max_evidence_tokens=max_evidence_tokens,
            device=device,
        )
        right = trainer.evaluate(batch)
        right_losses.append(float(right.total.detach().cpu()))
        alternate = next(
            (
                candidate
                for candidate in selected[index + 1 :] + selected[:index]
                if typed_frames(candidate)[1].evidence_hash != evidence.evidence_hash
            ),
            None,
        )
        if alternate is None:
            continue
        alternate_ids, alternate_mask, _ = encode_evidence_tokens(
            alternate,
            serializer=serializer,
            tokenizer=tokenizer,
            text_cardinality=text_cardinality,
            max_evidence_tokens=max_evidence_tokens,
            device=device,
        )
        wrong_batch = dict(batch)
        wrong_batch["evidence_token_ids"] = alternate_ids
        wrong_batch["evidence_attention_mask"] = alternate_mask
        wrong = trainer.evaluate(wrong_batch)
        wrong_losses.append(float(wrong.total.detach().cpu()))
        evidence_pairs += 1
    mean = lambda values: (sum(values) / len(values)) if values else None
    correct = mean(right_losses)
    wrong = mean(wrong_losses)
    return {
        "schema_version": 1,
        "kind": "personaplex-evidence-stream-heldout-evaluation",
        "examples": len(right_losses),
        "counterfactual_wrong_evidence_examples": evidence_pairs,
        "correct_evidence_loss": correct,
        "wrong_evidence_loss": wrong,
        "evidence_sensitivity_delta": (wrong - correct) if correct is not None and wrong is not None else None,
        "metric_scope": "teacher_forced_native_agent_loss; runtime causal speech and cancellation remain mandatory promotion gates",
    }


def load_control_adapter(path: Path, lm: Any, device: torch.device, prefix_frames: int) -> SemanticPrefixAdapter:
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
    ).to(device)
    adapter.load_state_dict(state, strict=True)
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--control-adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--prefix-frames", type=int, default=16)
    parser.add_argument("--evidence-stream-frames", type=int, default=16)
    parser.add_argument("--max-control-tokens", type=int, default=512)
    parser.add_argument("--max-evidence-tokens", type=int, default=256)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--eval-examples", type=int, default=32)
    args = parser.parse_args()
    if min(args.max_steps, args.prefix_frames, args.evidence_stream_frames, args.max_control_tokens, args.max_evidence_tokens, args.checkpoint_every, args.eval_examples) < 1:
        raise SystemExit("all size and step arguments must be positive")
    rank, local_rank, world_size = rank_info()
    if not torch.cuda.is_available():
        raise SystemExit("evidence-stream training requires CUDA; CPU fallback is prohibited")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    manifest = args.manifest.resolve()
    artifact_root = args.artifact_root.resolve()
    contract = json.loads(args.model_contract.read_text(encoding="utf-8"))
    require_certificate(args.certificate.resolve(), manifest, contract)
    require_moshi_source_contract(args.moshi_source_root.resolve(), contract)
    if contract.get("moshi_weights_sha256") != hash_file(args.moshi_path.resolve()):
        raise SystemExit("LM weights do not match the inspected native model contract")
    if contract.get("tokenizer_sha256") != hash_file(args.tokenizer_path.resolve()):
        raise SystemExit("SentencePiece tokenizer does not match the inspected native model contract")
    sys.path.insert(0, str(args.moshi_source_root.resolve()))
    import sentencepiece
    from moshi.models.loaders import get_moshi_lm

    lm = get_moshi_lm(args.moshi_path.resolve(), device=device, dtype=torch.bfloat16)
    if "streaming_sum" not in inspect.signature(lm.forward_embeddings).parameters:
        raise SystemExit("patched native PersonaPlex source lacks forward_embeddings(streaming_sum=...)")
    layout = StreamLayout.from_mapping(contract["stream_layout"])
    layout.validate_for_model(lm)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer_path.resolve()))
    control_adapter = load_control_adapter(args.control_adapter_checkpoint.resolve(), lm, device, args.prefix_frames)
    evidence_adapter: torch.nn.Module = EvidenceStreamAdapter(
        text_cardinality=int(lm.text_card),
        hidden_size=int(lm.dim),
        stream_frames=args.evidence_stream_frames,
    ).to(device)
    if world_size > 1:
        evidence_adapter = DistributedDataParallel(evidence_adapter, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.AdamW(evidence_adapter.parameters(), lr=args.learning_rate, weight_decay=0.01)
    trainer = EvidenceStreamTrainer(lm, control_adapter, evidence_adapter, optimizer, layout, activation_checkpointing=True)
    rows = load_jsonl(manifest)
    train_rows = [row for row in rows if row.get("split") == "train"]
    heldout_rows = [row for row in rows if row.get("split") in {"validation", "test"}]
    if len(train_rows) < world_size or not heldout_rows:
        raise SystemExit("certified evidence corpus needs distributed train rows and held-out rows")
    local_rows = train_rows[rank::world_size]
    if not local_rows:
        raise SystemExit(f"rank {rank} has no assigned training rows")
    if rank == 0:
        args.run_dir.mkdir(parents=True, exist_ok=False)
        (args.run_dir / "run_contract.json").write_text(json.dumps({
            "schema_version": 1,
            "kind": "personaplex-evidence-stream-run",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest),
            "manifest_sha256": hash_file(manifest),
            "certificate": str(args.certificate.resolve()),
            "model_contract": str(args.model_contract.resolve()),
            "model_revision": contract["model_revision"],
            "control_adapter_checkpoint": str(args.control_adapter_checkpoint.resolve()),
            "control_adapter_checkpoint_sha256": hash_file(args.control_adapter_checkpoint.resolve()),
            "world_size": world_size,
            "prefix_frames": args.prefix_frames,
            "evidence_stream_frames": args.evidence_stream_frames,
            "caller_stream_supervision": "forbidden",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if world_size > 1:
        dist.barrier()
    serializer = PlanSerializer()
    module = evidence_adapter.module if isinstance(evidence_adapter, DistributedDataParallel) else evidence_adapter
    evaluator = EvidenceStreamTrainer(lm, control_adapter, module, optimizer, layout, activation_checkpointing=False)
    metrics_path = args.run_dir / "metrics.jsonl"

    def write_checkpoint(step: int) -> None:
        checkpoint = args.run_dir / f"evidence_adapter_step_{step:06d}.pt"
        torch.save({
            "evidence_adapter_state_dict": module.state_dict(),
            "model_revision": contract["model_revision"],
            "stream_layout": layout.as_dict(),
            "manifest_sha256": hash_file(manifest),
            "control_adapter_checkpoint_sha256": hash_file(args.control_adapter_checkpoint.resolve()),
            "prefix_frames": args.prefix_frames,
            "evidence_stream_frames": args.evidence_stream_frames,
            "serializer_version": serializer.version,
            "step": step,
        }, checkpoint)
        evaluation = evaluate_checkpoint(
            evaluator,
            heldout_rows,
            artifact_root=artifact_root,
            serializer=serializer,
            tokenizer=tokenizer,
            text_cardinality=int(lm.text_card),
            max_control_tokens=args.max_control_tokens,
            max_evidence_tokens=args.max_evidence_tokens,
            device=device,
            max_examples=args.eval_examples,
        )
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "checkpoint_evaluation", "step": step, "checkpoint": checkpoint.name, **evaluation}, sort_keys=True) + "\n")

    if rank == 0:
        write_checkpoint(0)
    if world_size > 1:
        dist.barrier()
    for step in range(args.max_steps):
        batch, _control, _evidence = batch_for_record(
            local_rows[step % len(local_rows)],
            artifact_root=artifact_root,
            serializer=serializer,
            tokenizer=tokenizer,
            text_cardinality=int(lm.text_card),
            max_control_tokens=args.max_control_tokens,
            max_evidence_tokens=args.max_evidence_tokens,
            device=device,
        )
        loss = trainer.step(batch)
        values = torch.stack([loss.total.detach().to(torch.float64), loss.text.detach().to(torch.float64), loss.audio.detach().to(torch.float64)])
        if world_size > 1:
            dist.all_reduce(values, op=dist.ReduceOp.SUM)
            values /= world_size
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step + 1, "loss": float(values[0]), "text_loss": float(values[1]), "audio_loss": float(values[2])}, sort_keys=True) + "\n")
        if (step + 1) % args.checkpoint_every == 0 or step + 1 == args.max_steps:
            if world_size > 1:
                dist.barrier()
            if rank == 0:
                write_checkpoint(step + 1)
            if world_size > 1:
                dist.barrier()
    if rank == 0:
        torch.save({
            "evidence_adapter_state_dict": module.state_dict(),
            "model_revision": contract["model_revision"],
            "stream_layout": layout.as_dict(),
            "manifest_sha256": hash_file(manifest),
            "control_adapter_checkpoint_sha256": hash_file(args.control_adapter_checkpoint.resolve()),
            "prefix_frames": args.prefix_frames,
            "evidence_stream_frames": args.evidence_stream_frames,
        }, args.run_dir / "evidence_adapter_last.pt")
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
