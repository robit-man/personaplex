#!/usr/bin/env python3
"""
distill_v2.py — Knowledge distillation using LMGen's built-in return_logits.

Uses the existing return_logits=True flag on LMGen to get teacher logits
at each streaming step. Student trains to match via KL divergence.

Phase 1: Generate (codes, text_logits, audio_logits) from teacher
Phase 2: Train student to match teacher logits while processing same codes
Phase 3: Re-quantize trained student to NF4

Usage:
  # Phase 1: generate dataset
  python distill_v2.py --phase generate --steps-per-prompt 200

  # Phase 2: train
  python distill_v2.py --phase train --epochs 3 --lr 5e-6

  # Phase 3: eval
  python distill_v2.py --phase eval --steps 100
"""

import argparse
import json
import os
import sys
import time
import logging
import pickle
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "personaplex-setup" / "moshi"))
os.environ["NO_CUDA_GRAPH"] = "1"

PROMPTS = [
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
    "You are explaining how a complex distributed system works to a non-technical stakeholder.",
    "You are a security engineer reviewing an access control change. Be thorough but clear.",
    "You are helping plan a database migration. Walk through risks and rollback strategies.",
    "You have strong opinions about software architecture and enjoy debating them constructively.",
    "You are troubleshooting a CI/CD pipeline failure. Think step by step.",
]


def generate_dataset(device: str = "cuda:0", steps_per_prompt: int = 200, output_dir: str = "distillation/dataset_v2"):
    """Phase 1: Run teacher with return_logits=True, save everything."""
    import torch
    from moshi.models import loaders
    from moshi.models.lm import LMGen
    import sentencepiece
    from huggingface_hub import hf_hub_download

    os.makedirs(output_dir, exist_ok=True)

    bf16_path = "/home/roko/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309/model.safetensors"

    log.info(f"Loading teacher (bf16) on {device}...")
    mimi_path = hf_hub_download("nvidia/personaplex-7b-v1", loaders.MIMI_NAME)
    tok_path = hf_hub_download("nvidia/personaplex-7b-v1", loaders.TEXT_TOKENIZER_NAME)

    mimi = loaders.get_mimi(mimi_path, device)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tok_path)
    lm = loaders.get_moshi_lm(bf16_path, device=device, dtype=torch.bfloat16)
    lm.eval()

    # Create LMGen WITH return_logits=True — this is the key
    lm_gen = LMGen(lm, device=device, check=False, return_logits=True,
                   audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
                   sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate)

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    total_samples = 0

    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    for pi, prompt in enumerate(PROMPTS):
        log.info(f"[{pi+1}/{len(PROMPTS)}] {prompt[:60]}...")

        lm_gen.text_prompt_tokens = text_tokenizer.encode(f"<system> {prompt} <system>")

        prompt_samples = []

        with torch.no_grad(), lm_gen.streaming(1):
            mimi.reset_streaming()
            lm_gen.reset_streaming()

            # System prompts
            lm_gen.step_system_prompts(mimi)
            mimi.reset_streaming()

            text_pieces = []

            for step in range(steps_per_prompt):
                silence = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
                codes = mimi.encode(silence)

                for c in range(codes.shape[-1]):
                    result = lm_gen.step(codes[:, :, c:c+1])

                    if result is None or result[0] is None:
                        continue

                    tokens, (text_logits, audio_logits) = result

                    # Save the training tuple
                    sample = {
                        "codes": codes[:, :, c:c+1].cpu(),
                        "tokens": tokens.cpu(),
                        "text_logits": text_logits.cpu().float(),  # [B, 1, 1, vocab]
                        "audio_logits": audio_logits.cpu().float() if audio_logits is not None else None,
                    }
                    prompt_samples.append(sample)
                    total_samples += 1

                    # Track text for logging
                    text_token = tokens[0, 0, 0].item()
                    if text_token not in (0, 3):
                        piece = text_tokenizer.id_to_piece(text_token)
                        text_pieces.append(piece.replace("▁", " "))

            text_output = "".join(text_pieces)
            log.info(f"  [{pi+1}] {len(prompt_samples)} samples, text: \"{text_output[:100]}\"")

            # Save per-prompt
            prompt_file = os.path.join(output_dir, f"prompt_{pi:02d}.pt")
            torch.save({
                "prompt": prompt,
                "prompt_idx": pi,
                "samples": prompt_samples,
                "text_output": text_output,
            }, prompt_file)

    log.info(f"Dataset complete: {total_samples} samples across {len(PROMPTS)} prompts")
    log.info(f"Saved to {output_dir}/")

    del lm, lm_gen, mimi
    torch.cuda.empty_cache()


def train(teacher_device: str = "cuda:0", student_device: str = "cuda:1",
          dataset_dir: str = "distillation/dataset_v2", output_dir: str = "distillation/checkpoints_v2",
          epochs: int = 3, lr: float = 5e-6):
    """Phase 2: Train student to match teacher logits."""
    import torch
    import torch.nn.functional as F
    from moshi.models import loaders
    from moshi.models.lm import LMGen
    from huggingface_hub import hf_hub_download
    import glob

    os.makedirs(output_dir, exist_ok=True)

    bf16_path = "/home/roko/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309/model.safetensors"
    nf4_path = "/home/roko/.cache/huggingface/hub/models--cudabenchmarktest--personaplex-7b-nf4/snapshots/dd96c876a28608f089b02d1bbdb1c1131532414e/model-nf4.safetensors"

    # Load dataset
    prompt_files = sorted(glob.glob(os.path.join(dataset_dir, "prompt_*.pt")))
    if not prompt_files:
        log.error(f"No dataset files in {dataset_dir}. Run --phase generate first.")
        return
    log.info(f"Loading {len(prompt_files)} prompt datasets...")

    all_samples = []
    for pf in prompt_files:
        data = torch.load(pf, map_location="cpu", weights_only=False)
        all_samples.extend(data["samples"])
    log.info(f"Total training samples: {len(all_samples)}")

    # Load student model (from NF4 dequanted — starts close to teacher)
    log.info(f"Loading student (NF4 → bf16) on {student_device}...")
    student_lm = loaders.get_moshi_lm(nf4_path, device=student_device, dtype=torch.bfloat16)

    # Enable training mode on the LM (not LMGen — we train the raw model)
    student_lm.train()

    trainable = sum(p.numel() for p in student_lm.parameters() if p.requires_grad)
    log.info(f"Student: {trainable/1e9:.2f}B trainable parameters")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in student_lm.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01, betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(all_samples))

    temperature = 2.0
    alpha_kl = 0.7  # KL divergence weight
    alpha_hard = 0.3  # Hard label weight

    best_loss = float("inf")
    training_log = []

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_kl_losses = []
        epoch_hard_losses = []
        epoch_total_losses = []

        # Shuffle samples each epoch
        import random
        random.shuffle(all_samples)

        for i, sample in enumerate(all_samples):
            # Teacher targets (pre-computed)
            teacher_text_logits = sample["text_logits"].to(student_device)  # [B, 1, 1, vocab]

            # Use teacher's OUTPUT tokens as INPUT (autoregressive teacher forcing)
            # tokens shape: [1, 17, 1] — full codebook sequence
            input_sequence = sample["tokens"].to(student_device)  # [B, K=17, T=1]

            # Student forward on the full 17-codebook input
            try:
                student_out, student_text_logits = student_lm.forward_codes(input_sequence)
            except Exception as e:
                if i < 3:
                    log.warning(f"forward_codes failed (sample {i}): {e}")
                continue

            # KL divergence loss (soft targets)
            teacher_soft = F.softmax(teacher_text_logits.view(-1, teacher_text_logits.size(-1)) / temperature, dim=-1)
            student_log_soft = F.log_softmax(student_text_logits.view(-1, student_text_logits.size(-1)) / temperature, dim=-1)
            kl_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)

            # Hard label loss (teacher's argmax as target)
            teacher_hard = teacher_text_logits.view(-1, teacher_text_logits.size(-1)).argmax(dim=-1)
            hard_loss = F.cross_entropy(
                student_text_logits.view(-1, student_text_logits.size(-1)),
                teacher_hard,
            )

            # Combined loss
            loss = alpha_kl * kl_loss + alpha_hard * hard_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_lm.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_kl_losses.append(kl_loss.item())
            epoch_hard_losses.append(hard_loss.item())
            epoch_total_losses.append(loss.item())

            if (i + 1) % 100 == 0:
                avg_kl = sum(epoch_kl_losses[-100:]) / 100
                avg_hard = sum(epoch_hard_losses[-100:]) / 100
                avg_total = sum(epoch_total_losses[-100:]) / 100
                current_lr = scheduler.get_last_lr()[0]
                log.info(f"  E{epoch+1} [{i+1}/{len(all_samples)}] "
                         f"Loss: {avg_total:.4f} (KL: {avg_kl:.4f}, Hard: {avg_hard:.4f}) "
                         f"LR: {current_lr:.2e}")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_total_losses) / max(len(epoch_total_losses), 1)
        avg_kl = sum(epoch_kl_losses) / max(len(epoch_kl_losses), 1)
        avg_hard = sum(epoch_hard_losses) / max(len(epoch_hard_losses), 1)

        log.info(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} "
                 f"(KL: {avg_kl:.4f}, Hard: {avg_hard:.4f}) — {epoch_time:.0f}s")

        training_log.append({
            "epoch": epoch + 1, "avg_loss": avg_loss,
            "avg_kl": avg_kl, "avg_hard": avg_hard,
            "time_s": epoch_time, "samples": len(epoch_total_losses),
        })

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(output_dir, "student_best.pt")
            torch.save(student_lm.state_dict(), ckpt_path)
            log.info(f"  Best checkpoint: {ckpt_path} (loss={best_loss:.4f})")

    # Save final
    final_path = os.path.join(output_dir, "student_final.pt")
    torch.save(student_lm.state_dict(), final_path)

    # Save log
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump({"log": training_log, "best_loss": best_loss, "config": {
            "epochs": epochs, "lr": lr, "temperature": temperature,
            "alpha_kl": alpha_kl, "alpha_hard": alpha_hard,
            "total_samples": len(all_samples),
        }}, f, indent=2)

    log.info(f"Training complete. Best loss: {best_loss:.4f}")
    log.info(f"Run --phase eval to test the distilled model")

    del student_lm
    torch.cuda.empty_cache()


def evaluate(checkpoint_path: str, device: str = "cuda:0", steps: int = 100):
    """Phase 3: Compare distilled student vs bf16 teacher vs raw NF4."""
    import torch
    from moshi.models import loaders
    from moshi.models.lm import LMGen
    import sentencepiece
    from huggingface_hub import hf_hub_download

    bf16_path = "/home/roko/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309/model.safetensors"
    nf4_path = "/home/roko/.cache/huggingface/hub/models--cudabenchmarktest--personaplex-7b-nf4/snapshots/dd96c876a28608f089b02d1bbdb1c1131532414e/model-nf4.safetensors"

    mimi_path = hf_hub_download("nvidia/personaplex-7b-v1", loaders.MIMI_NAME)
    tok_path = hf_hub_download("nvidia/personaplex-7b-v1", loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tok_path)

    test_prompt = "You enjoy casual conversation about technology and science."

    results = {}

    for name, weight_path, dev in [
        ("bf16", bf16_path, "cuda:0"),
        ("nf4_raw", nf4_path, "cuda:1"),
    ]:
        log.info(f"Evaluating {name} on {dev}...")
        mimi = loaders.get_mimi(mimi_path, dev)
        lm = loaders.get_moshi_lm(weight_path, device=dev, dtype=torch.bfloat16)
        lm.eval()
        lm_gen = LMGen(lm, device=dev, check=False,
                       audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
                       sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate)
        frame_size = int(mimi.sample_rate / mimi.frame_rate)

        lm_gen.text_prompt_tokens = text_tokenizer.encode(f"<system> {test_prompt} <system>")
        mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)

        text_tokens = []
        text_pieces = []

        with torch.no_grad(), lm_gen.streaming(1):
            mimi.reset_streaming()
            lm_gen.reset_streaming()
            lm_gen.step_system_prompts(mimi)
            mimi.reset_streaming()

            for step in range(steps):
                silence = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=dev)
                codes = mimi.encode(silence)
                for c in range(codes.shape[-1]):
                    tokens = lm_gen.step(codes[:, :, c:c+1])
                    if tokens is None:
                        continue
                    text_token = tokens[0, 0, 0].item()
                    text_tokens.append(text_token)
                    if text_token not in (0, 3):
                        text_pieces.append(text_tokenizer.id_to_piece(text_token).replace("▁", " "))

        results[name] = {
            "text": "".join(text_pieces),
            "tokens": text_tokens,
            "num_tokens": len(text_tokens),
        }
        log.info(f"  [{name}] \"{results[name]['text'][:100]}\"")

        del lm, lm_gen, mimi
        torch.cuda.empty_cache()

    # Load distilled student if checkpoint exists
    if os.path.exists(checkpoint_path):
        log.info(f"Evaluating distilled student from {checkpoint_path}...")
        mimi = loaders.get_mimi(mimi_path, "cuda:2")
        lm_kwargs = dict(loaders._lm_kwargs)
        lm_kwargs["dep_q"] = 16
        student_lm = loaders.LMModel(device="cpu", dtype=torch.bfloat16, **lm_kwargs)
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Cast all to bf16 (training may have promoted to fp32)
        state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in state.items()}
        student_lm.load_state_dict(state, strict=False)
        student_lm = student_lm.to(device="cuda:2", dtype=torch.bfloat16)
        student_lm.eval()

        lm_gen = LMGen(student_lm, device="cuda:2", check=False,
                       audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
                       sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate)
        frame_size = int(mimi.sample_rate / mimi.frame_rate)
        lm_gen.text_prompt_tokens = text_tokenizer.encode(f"<system> {test_prompt} <system>")
        mimi.streaming_forever(1)
        lm_gen.streaming_forever(1)

        text_tokens = []
        text_pieces = []

        with torch.no_grad(), lm_gen.streaming(1):
            mimi.reset_streaming()
            lm_gen.reset_streaming()
            lm_gen.step_system_prompts(mimi)
            mimi.reset_streaming()
            for step in range(steps):
                silence = torch.zeros(1, 1, frame_size, dtype=torch.float32, device="cuda:2")
                codes = mimi.encode(silence)
                for c in range(codes.shape[-1]):
                    tokens = lm_gen.step(codes[:, :, c:c+1])
                    if tokens is None:
                        continue
                    text_token = tokens[0, 0, 0].item()
                    text_tokens.append(text_token)
                    if text_token not in (0, 3):
                        text_pieces.append(text_tokenizer.id_to_piece(text_token).replace("▁", " "))

        results["distilled"] = {
            "text": "".join(text_pieces),
            "tokens": text_tokens,
            "num_tokens": len(text_tokens),
        }
        log.info(f"  [distilled] \"{results['distilled']['text'][:100]}\"")

        del student_lm, lm_gen, mimi
        torch.cuda.empty_cache()

    # Compare
    print("\n" + "=" * 70)
    print("DISTILLATION EVAL RESULTS")
    print("=" * 70)
    bf16_tokens = results.get("bf16", {}).get("tokens", [])
    for name in ["nf4_raw", "distilled"]:
        if name not in results:
            continue
        qtoks = results[name]["tokens"]
        min_len = min(len(bf16_tokens), len(qtoks))
        match = sum(1 for a, b in zip(bf16_tokens[:min_len], qtoks[:min_len]) if a == b) / max(min_len, 1)
        print(f"\n{name}:")
        print(f"  Token match vs bf16: {match:.1%}")
        print(f"  Output: \"{results[name]['text'][:150]}\"")
    if "bf16" in results:
        print(f"\nbf16 (reference):")
        print(f"  Output: \"{results['bf16']['text'][:150]}\"")


def main():
    parser = argparse.ArgumentParser(description="PersonaPlex knowledge distillation v2")
    parser.add_argument("--phase", choices=["generate", "train", "eval", "all"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--steps-per-prompt", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="distillation/checkpoints_v2/student_best.pt")
    args = parser.parse_args()

    if args.phase in ("generate", "all"):
        generate_dataset(steps_per_prompt=args.steps_per_prompt)

    if args.phase in ("train", "all"):
        train(epochs=args.epochs, lr=args.lr)

    if args.phase in ("eval", "all"):
        evaluate(args.checkpoint, steps=args.eval_steps)


if __name__ == "__main__":
    main()
