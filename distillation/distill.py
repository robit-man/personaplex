#!/usr/bin/env python3
"""
distill.py — Knowledge distillation for PersonaPlex quantized models.

Uses the bf16 PersonaPlex as teacher to generate soft targets (logits),
then trains a student model (from NF4 or freshly quantized weights) to
match the teacher's output distributions.

Architecture:
  - Teacher (bf16, frozen) runs on GPU 0 → produces text/audio logits
  - Student (trainable bf16 copy for fine-tuning) runs on GPU 1
  - After training, student weights are re-quantized to NF4/2-bit

Loss:
  - KL divergence on text logits (what the model says)
  - KL divergence on audio logits (how it sounds)
  - Hard label cross-entropy (ground truth from teacher's argmax)

Usage:
  python distill.py --epochs 5 --steps-per-epoch 200 --lr 1e-5
  python distill.py --generate-dataset --num-samples 500
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "personaplex-setup" / "moshi"))
os.environ["NO_CUDA_GRAPH"] = "1"


def load_prompts(prompts_file: str = None) -> list[str]:
    """Load system prompts for diverse training data."""
    if prompts_file and os.path.exists(prompts_file):
        with open(prompts_file) as f:
            prompts = json.load(f)
        if isinstance(prompts, list):
            return prompts

    # Fallback: built-in diverse prompts
    return [
        "You are a helpful engineering assistant. You explain technical concepts clearly and concisely.",
        "You are a senior developer reviewing code changes. Give direct, actionable feedback.",
        "You are monitoring a production deployment. Report status clearly and flag any concerns.",
        "You enjoy casual conversation about technology trends and programming languages.",
        "You are helping debug a failing API. Ask clarifying questions and suggest fixes.",
        "You are onboarding a new team member. Explain the project structure patiently.",
        "You are a systems architect discussing trade-offs between different approaches.",
        "You are coordinating a release. Confirm each step before proceeding.",
        "You are analyzing performance metrics. Highlight anomalies and suggest optimizations.",
        "You are a friendly colleague having a coffee break conversation about tech.",
    ]


def generate_dataset(teacher_info: dict, prompts: list[str], num_steps: int = 100,
                     output_dir: str = "dataset") -> str:
    """Generate (input, teacher_logits) pairs for distillation training."""
    import torch
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    mimi = teacher_info["mimi"]
    lm_gen = teacher_info["lm_gen"]
    text_tokenizer = teacher_info["text_tokenizer"]
    device = teacher_info["device"]
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    samples = []
    total_samples = 0

    for pi, prompt in enumerate(prompts):
        log.info(f"Generating data for prompt {pi+1}/{len(prompts)}: {prompt[:50]}...")

        # Set prompt
        lm_gen.text_prompt_tokens = text_tokenizer.encode(f"<system> {prompt} <system>")

        if pi == 0:
            mimi.streaming_forever(1)
            lm_gen.streaming_forever(1)

        with torch.no_grad(), lm_gen.streaming(1):
            mimi.reset_streaming()
            lm_gen.reset_streaming()
            # Feed system prompts
            lm_gen.step_system_prompts(mimi)
            mimi.reset_streaming()

            for step in range(num_steps):
                silence = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
                codes = mimi.encode(silence)

                for c in range(codes.shape[-1]):
                    # Get the full model output including logits
                    tokens = lm_gen.step(codes[:, :, c:c+1])
                    if tokens is None:
                        continue

                    # Record the input codes and output tokens
                    sample = {
                        "input_codes": codes[:, :, c:c+1].cpu().numpy().tolist(),
                        "output_tokens": tokens.cpu().numpy().tolist(),
                        "prompt_idx": pi,
                        "step": step,
                    }
                    samples.append(sample)
                    total_samples += 1

        log.info(f"  Prompt {pi+1}: {total_samples} total samples")

    # Save dataset
    dataset_path = os.path.join(output_dir, "distill_dataset.json")
    with open(dataset_path, "w") as f:
        json.dump({
            "prompts": prompts,
            "num_samples": len(samples),
            "samples": samples,
        }, f)

    log.info(f"Dataset saved: {dataset_path} ({len(samples)} samples)")
    return dataset_path


def train_distillation(
    teacher_path: str,
    student_path: str,
    dataset_path: str,
    output_dir: str = "checkpoints",
    epochs: int = 3,
    lr: float = 1e-5,
    teacher_device: str = "cuda:0",
    student_device: str = "cuda:1",
):
    """Run knowledge distillation training.

    The student starts from NF4 dequanted weights (close to teacher)
    and is fine-tuned to better match the teacher's output distribution.
    After training, weights are re-quantized.
    """
    import torch
    import torch.nn.functional as F
    from moshi.models import loaders

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    log.info(f"Loading dataset from {dataset_path}")
    with open(dataset_path) as f:
        dataset = json.load(f)
    samples = dataset["samples"]
    prompts = dataset["prompts"]
    log.info(f"Dataset: {len(samples)} samples, {len(prompts)} prompts")

    # Load teacher (frozen)
    log.info(f"Loading teacher (bf16) on {teacher_device}...")
    teacher_lm = loaders.get_moshi_lm(teacher_path, device=teacher_device, dtype=torch.bfloat16)
    teacher_lm.eval()
    for p in teacher_lm.parameters():
        p.requires_grad = False
    log.info("Teacher loaded (frozen)")

    # Load student (trainable) — start from NF4 dequanted weights
    log.info(f"Loading student (from {student_path}) on {student_device}...")
    student_lm = loaders.get_moshi_lm(student_path, device=student_device, dtype=torch.bfloat16)
    student_lm.train()
    log.info("Student loaded (trainable)")

    # Count trainable params
    trainable = sum(p.numel() for p in student_lm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in student_lm.parameters())
    log.info(f"Student params: {trainable/1e6:.1f}M trainable / {total/1e6:.1f}M total")

    # Optimizer — only train the student
    optimizer = torch.optim.AdamW(
        [p for p in student_lm.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    # Training loop
    temperature = 2.0  # Soft targets temperature
    alpha = 0.7  # Weight for KL loss vs hard label loss

    best_loss = float("inf")
    losses_log = []

    for epoch in range(epochs):
        epoch_losses = []
        epoch_start = time.time()

        for i, sample in enumerate(samples):
            input_codes = torch.tensor(sample["input_codes"], device=student_device)
            target_tokens = torch.tensor(sample["output_tokens"], device=student_device)

            # Teacher forward (frozen, no grad)
            with torch.no_grad():
                input_codes_t = input_codes.to(teacher_device)
                # Get teacher embeddings and logits
                teacher_emb = teacher_lm.embed_codes(input_codes_t)
                teacher_out, teacher_text_logits = teacher_lm.forward_embeddings(teacher_emb)

            # Student forward
            input_codes_s = input_codes.to(student_device)
            student_emb = student_lm.embed_codes(input_codes_s)
            student_out, student_text_logits = student_lm.forward_embeddings(student_emb)

            # KL divergence loss on text logits
            teacher_text_soft = F.log_softmax(teacher_text_logits.to(student_device) / temperature, dim=-1)
            student_text_soft = F.log_softmax(student_text_logits / temperature, dim=-1)
            kl_loss = F.kl_div(student_text_soft, teacher_text_soft.exp(), reduction="batchmean") * (temperature ** 2)

            # Hard label loss (cross-entropy with teacher's argmax)
            teacher_hard = teacher_text_logits.to(student_device).argmax(dim=-1)
            hard_loss = F.cross_entropy(
                student_text_logits.view(-1, student_text_logits.size(-1)),
                teacher_hard.view(-1),
            )

            # Combined loss
            loss = alpha * kl_loss + (1 - alpha) * hard_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_lm.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

            if (i + 1) % 50 == 0:
                avg = sum(epoch_losses[-50:]) / 50
                log.info(f"  Epoch {epoch+1}/{epochs} Step {i+1}/{len(samples)} Loss: {avg:.4f} "
                         f"(KL: {kl_loss.item():.4f}, Hard: {hard_loss.item():.4f})")

        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses_log.append({"epoch": epoch + 1, "avg_loss": avg_loss, "time_s": epoch_time})

        log.info(f"Epoch {epoch+1}/{epochs} — Avg loss: {avg_loss:.4f}, Time: {epoch_time:.0f}s")

        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(output_dir, f"student_best.pt")
            torch.save(student_lm.state_dict(), ckpt_path)
            log.info(f"  Best checkpoint saved: {ckpt_path}")

    # Save final checkpoint
    final_path = os.path.join(output_dir, f"student_final.pt")
    torch.save(student_lm.state_dict(), final_path)

    # Save training log
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump({"losses": losses_log, "best_loss": best_loss, "config": {
            "epochs": epochs, "lr": lr, "temperature": temperature, "alpha": alpha,
            "teacher": teacher_path, "student": student_path, "samples": len(samples),
        }}, f, indent=2)

    log.info(f"Training complete. Best loss: {best_loss:.4f}")
    log.info(f"Checkpoints: {output_dir}/")

    # Cleanup
    del teacher_lm, student_lm
    torch.cuda.empty_cache()

    return final_path


def main():
    parser = argparse.ArgumentParser(description="PersonaPlex knowledge distillation")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate training dataset from teacher")
    parser.add_argument("--train", action="store_true", help="Run distillation training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-prompt", type=int, default=100, help="Inference steps per prompt for dataset")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--prompts-file", type=str, default="distillation/prompts.json")
    parser.add_argument("--dataset", type=str, default="distillation/dataset/distill_dataset.json")
    parser.add_argument("--output-dir", type=str, default="distillation/checkpoints")
    args = parser.parse_args()

    import torch

    bf16_path = "/home/roko/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309/model.safetensors"
    nf4_path = "/home/roko/.cache/huggingface/hub/models--cudabenchmarktest--personaplex-7b-nf4/snapshots/dd96c876a28608f089b02d1bbdb1c1131532414e/model-nf4.safetensors"

    prompts = load_prompts(args.prompts_file)
    log.info(f"Loaded {len(prompts)} prompts")

    if args.generate_dataset:
        from eval_quant import load_model
        teacher = load_model(bf16_path, "cuda:0", "teacher_bf16")
        generate_dataset(teacher, prompts, num_steps=args.steps_per_prompt,
                         output_dir="distillation/dataset")
        del teacher
        torch.cuda.empty_cache()

    if args.train:
        train_distillation(
            teacher_path=bf16_path,
            student_path=nf4_path,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            epochs=args.epochs,
            lr=args.lr,
        )


if __name__ == "__main__":
    main()
