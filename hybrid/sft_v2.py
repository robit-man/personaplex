#!/usr/bin/env python3
"""
sft_v2.py — SFT using LMModel.forward_train() for proper gradient flow.

Uses the teacher's output token sequences from the distillation dataset,
feeds them through forward_train, and trains to match the text targets.

The key: forward_train() takes [B, K=17, T] codebook sequences and returns
(text_logits, audio_logits) through the full transformer. This gives us
proper gradients through the entire model, not just the embedding layer.

Usage:
  python sft_v2.py --epochs 20 --lr 1e-5
"""

import argparse
import glob
import json
import os
import sys
import time
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "personaplex-setup" / "moshi"))
os.environ["NO_CUDA_GRAPH"] = "1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dataset-dir", type=str, default="distillation/dataset_v2")
    parser.add_argument("--model", type=str, default="/tmp/distilled_bf16.safetensors")
    parser.add_argument("--output-dir", type=str, default="hybrid/checkpoints")
    parser.add_argument("--batch-t", type=int, default=20, help="Temporal window size for forward_train")
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F
    from moshi.models import loaders

    os.makedirs(args.output_dir, exist_ok=True)

    # Load the distillation dataset (has teacher tokens as [1, 17, 1] per step)
    prompt_files = sorted(glob.glob(os.path.join(args.dataset_dir, "prompt_*.pt")))
    if not prompt_files:
        log.error(f"No dataset files in {args.dataset_dir}")
        return

    log.info(f"Loading {len(prompt_files)} prompt datasets...")

    # Build training sequences: concatenate tokens over time for each prompt
    # forward_train expects [B, K=17, T] where T is a temporal window
    training_sequences = []
    for pf in prompt_files:
        data = torch.load(pf, map_location="cpu", weights_only=False)
        samples = data["samples"]
        # Concatenate all tokens for this prompt into [1, 17, T]
        tokens_list = [s["tokens"] for s in samples if s["tokens"] is not None]
        if tokens_list:
            # Each is [1, 17, 1] — cat along dim=2
            full_seq = torch.cat(tokens_list, dim=2)  # [1, 17, T]
            training_sequences.append({
                "tokens": full_seq,
                "prompt": data["prompt"],
                "text_output": data.get("text_output", ""),
            })
            log.info(f"  {Path(pf).name}: T={full_seq.shape[2]} tokens, text: \"{data.get('text_output','')[:50]}\"")

    log.info(f"Total sequences: {len(training_sequences)}")

    # Load model
    log.info(f"Loading model on cuda:0...")
    model = loaders.get_moshi_lm(args.model, device="cuda:0", dtype=torch.bfloat16)
    model.train()

    # Freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze: last 8 transformer layers + text output head + depformer
    # These directly control what text the model generates
    unfrozen = set()
    for name, p in model.named_parameters():
        should_unfreeze = False
        # Last 8 transformer layers (24-31)
        for layer_idx in range(24, 32):
            if f"transformer.layers.{layer_idx}." in name:
                should_unfreeze = True
                break
        # Text output heads
        if "text_linear" in name or "text_emb" in name:
            should_unfreeze = True
        # Depformer (generates audio conditioned on text decisions)
        if "depformer" in name:
            should_unfreeze = True

        if should_unfreeze:
            p.requires_grad = True
            unfrozen.add(name.split(".")[0])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f"Trainable: {trainable/1e9:.2f}B / {total/1e9:.2f}B total ({trainable/total*100:.0f}%)")
    log.info(f"Unfrozen groups: {sorted(unfrozen)}")

    # Optimizer — only trainable params, saves ~75% memory
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = args.epochs * sum(
        max(1, (seq["tokens"].shape[2] - 1) // args.batch_t) for seq in training_sequences
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    best_loss = float("inf")
    training_log = []
    step_count = 0

    for epoch in range(args.epochs):
        epoch_start = time.time()
        epoch_text_losses = []
        epoch_audio_losses = []

        random.shuffle(training_sequences)

        for si, seq_data in enumerate(training_sequences):
            full_tokens = seq_data["tokens"].to("cuda:0")  # [1, 17, T]
            T = full_tokens.shape[2]

            # Slide windows of size batch_t through the sequence
            for t_start in range(0, T - 1, args.batch_t):
                t_end = min(t_start + args.batch_t + 1, T)  # +1 for target
                if t_end - t_start < 3:
                    continue

                window = full_tokens[:, :, t_start:t_end]  # [1, 17, window_size]

                try:
                    output = model.forward_train(window)
                except Exception as e:
                    if step_count < 3:
                        log.warning(f"forward_train failed: {e}")
                    continue

                # Text loss: cross-entropy on text logits vs actual text tokens
                # output.text_logits: [B, 1, T, vocab_size]
                # Target: window[:, 0, 1:] (text tokens, shifted by 1)
                text_logits = output.text_logits  # [B, 1, T', vocab]
                text_mask = output.text_mask  # [B, 1, T']

                if text_logits is not None and text_mask is not None:
                    # Get valid positions
                    valid = text_mask.squeeze(1)  # [B, T']
                    if valid.any():
                        target_text = window[:, 0, 1:t_end - t_start]  # shifted targets
                        min_t = min(text_logits.shape[2], target_text.shape[1])

                        logits_flat = text_logits[:, 0, :min_t][valid[:, :min_t]].reshape(-1, text_logits.shape[-1])
                        targets_flat = target_text[:, :min_t][valid[:, :min_t]].reshape(-1)

                        if logits_flat.shape[0] > 0:
                            text_loss = F.cross_entropy(logits_flat.float(), targets_flat)
                        else:
                            text_loss = torch.tensor(0.0, device="cuda:0")
                    else:
                        text_loss = torch.tensor(0.0, device="cuda:0")
                else:
                    text_loss = torch.tensor(0.0, device="cuda:0")

                # Audio loss: cross-entropy on audio logits
                audio_loss = torch.tensor(0.0, device="cuda:0")
                if output.logits is not None and output.mask is not None:
                    audio_mask = output.mask  # [B, K_audio, T']
                    if audio_mask.any():
                        # Target audio: window[:, audio_offset:, 1:]
                        audio_offset = model.audio_offset
                        target_audio = window[:, audio_offset:audio_offset + model.dep_q, 1:t_end - t_start]
                        min_t_a = min(output.logits.shape[2], target_audio.shape[2])
                        min_k = min(output.logits.shape[1], target_audio.shape[1])

                        for k in range(min_k):
                            k_valid = audio_mask[:, k, :min_t_a]
                            if k_valid.any():
                                k_logits = output.logits[:, k, :min_t_a][k_valid].reshape(-1, output.logits.shape[-1])
                                k_targets = target_audio[:, k, :min_t_a][k_valid].reshape(-1)
                                if k_logits.shape[0] > 0:
                                    audio_loss = audio_loss + F.cross_entropy(k_logits.float(), k_targets)

                        audio_loss = audio_loss / max(min_k, 1)

                # Combined loss
                loss = text_loss + 0.5 * audio_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_text_losses.append(text_loss.item())
                epoch_audio_losses.append(audio_loss.item())
                step_count += 1

                if step_count % 50 == 0:
                    avg_t = sum(epoch_text_losses[-50:]) / min(50, len(epoch_text_losses[-50:]))
                    avg_a = sum(epoch_audio_losses[-50:]) / min(50, len(epoch_audio_losses[-50:]))
                    lr = scheduler.get_last_lr()[0]
                    log.info(f"  E{epoch+1} Step {step_count} "
                             f"TextLoss: {avg_t:.4f} AudioLoss: {avg_a:.4f} LR: {lr:.2e}")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_text = sum(epoch_text_losses) / max(len(epoch_text_losses), 1)
        avg_audio = sum(epoch_audio_losses) / max(len(epoch_audio_losses), 1)
        total = avg_text + 0.5 * avg_audio

        log.info(f"Epoch {epoch+1}/{args.epochs} — "
                 f"Text: {avg_text:.4f} Audio: {avg_audio:.4f} Total: {total:.4f} — {epoch_time:.0f}s")

        training_log.append({
            "epoch": epoch + 1, "text_loss": avg_text, "audio_loss": avg_audio,
            "total_loss": total, "time_s": epoch_time, "steps": step_count,
        })

        # Save checkpoint
        if total < best_loss:
            best_loss = total
            ckpt = os.path.join(args.output_dir, "sft_best.pt")
            torch.save(model.state_dict(), ckpt)
            log.info(f"  Best: {ckpt} (loss={best_loss:.4f})")

        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt = os.path.join(args.output_dir, f"sft_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt)

    # Final save
    final = os.path.join(args.output_dir, "sft_final.pt")
    torch.save(model.state_dict(), final)

    with open(os.path.join(args.output_dir, "sft_v2_log.json"), "w") as f:
        json.dump({"log": training_log, "best_loss": best_loss,
                    "epochs": args.epochs, "lr": args.lr, "batch_t": args.batch_t}, f, indent=2)

    log.info(f"SFT v2 complete. Best loss: {best_loss:.4f}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
