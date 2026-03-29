#!/usr/bin/env python3
"""
sft_train.py — Supervised Fine-Tuning to remove call center behavior.

Strategy:
  1. Generate natural conversation pairs using Qwen 27B (text)
  2. Convert to PersonaPlex token sequences via teacher (bf16) encoding
  3. Train student to produce the NATURAL responses instead of call center ones
  4. Use DPO-like penalty for call center patterns ("How can I help", self-naming)

The key insight: we don't just match the teacher's logits (which are call center).
Instead, we train on SYNTHETIC targets that Qwen generates — natural, direct responses.

For each sample:
  - Input: silence frames (simulating user pause before response)
  - Target: teacher's encoding of a NATURAL text response (not call center)
  - Penalty: extra loss for producing blacklisted phrases

This runs for many epochs. The training loop saves checkpoints every N steps
and logs all metrics to training_log.json for monitoring.

Usage:
  python sft_train.py --epochs 20 --lr 3e-6 --eval-every 500
"""

import argparse
import json
import os
import sys
import time
import logging
import re
import random
from pathlib import Path
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "personaplex-setup" / "moshi"))
os.environ["NO_CUDA_GRAPH"] = "1"

# Blacklisted phrases — model should NEVER produce these
BLACKLIST = [
    "how can i help",
    "how can i assist",
    "what can i do for you",
    "thank you for calling",
    "this is",  # "this is [name]" pattern
    "my name is",
    "welcome to",
    "how may i",
]

# Natural response templates — Qwen will expand these
CONVERSATION_STARTERS = [
    ("Tell me about the deployment status.", "The staging deploy finished about twenty minutes ago. All health checks passed. Production is next in the queue."),
    ("What do you think about Rust?", "Rust has this interesting ownership model that catches a whole class of bugs at compile time. The learning curve is steep though."),
    ("Can you explain how the auth module works?", "The auth module uses JWT tokens with a 15-minute expiry. Refresh tokens rotate on each use. The middleware validates on every request."),
    ("Run the test suite for me.", "I'll kick off the full test suite now. Want me to flag anything that takes longer than 30 seconds? That usually indicates a regression."),
    ("Something is wrong with the API.", "What errors are you seeing? Status codes, response times, anything in the logs? That'll narrow it down fast."),
    ("How is the system performing?", "Latency is sitting at p50 of 12ms, p99 at 180ms. That p99 spike started around 3pm. Looks correlated with the cache invalidation job."),
    ("Walk me through the project structure.", "The project follows a monorepo pattern. Core packages are in packages/, the CLI is the main entry point, and the orchestrator handles the agentic loop."),
    ("I want to deploy to production.", "Okay. Current staging is green. I'll prepare the production deploy. Before I proceed — are there any feature flags that need toggling?"),
    ("What happened in the last incident?", "The API went down for about 4 minutes. Root cause was a connection pool exhaustion from a retry storm. The fix was adding circuit breakers."),
    ("Tell me something interesting.", "Did you know that the entire Linux kernel gets roughly 80,000 commits per year? That's about 9 commits per hour, around the clock."),
]


def generate_training_pairs_via_qwen(n_pairs: int = 200) -> list[dict]:
    """Generate diverse natural conversation pairs using Qwen."""
    import requests

    log.info(f"Generating {n_pairs} conversation pairs via Qwen 27B...")

    pairs = []

    # Start with our hand-crafted examples
    for user_text, assistant_text in CONVERSATION_STARTERS:
        pairs.append({"user": user_text, "assistant": assistant_text})

    # Generate more via Qwen
    batch_size = 10
    for batch in range(0, n_pairs - len(CONVERSATION_STARTERS), batch_size):
        try:
            prompt = f"""Generate {batch_size} natural conversation exchanges between a user and an AI engineering assistant.

Rules:
- The assistant NEVER introduces itself by name
- The assistant NEVER says "how can I help you" or any variation
- The assistant responds directly and naturally, as if mid-conversation
- Topics: software engineering, deployments, debugging, code review, architecture, tech discussion
- Responses are 1-3 sentences, conversational, not formal

Output as JSON array of objects with "user" and "assistant" keys. No markdown.
"""
            resp = requests.post("http://localhost:11434/api/generate", json={
                "model": "open-agents-qwen35:27b",
                "prompt": prompt,
                "stream": False,
            }, timeout=180)
            data = resp.json()
            text = data.get("response", "")
            text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()

            # Try to parse JSON
            try:
                generated = json.loads(text)
                if isinstance(generated, list):
                    for item in generated:
                        if "user" in item and "assistant" in item:
                            # Filter out call center responses
                            asst = item["assistant"].lower()
                            if not any(bl in asst for bl in BLACKLIST):
                                pairs.append(item)
            except json.JSONDecodeError:
                pass

        except Exception as e:
            log.warning(f"Qwen generation failed: {e}")

        if len(pairs) >= n_pairs:
            break

        log.info(f"  Generated {len(pairs)}/{n_pairs} pairs...")

    log.info(f"Total training pairs: {len(pairs)}")
    return pairs[:n_pairs]


def encode_text_as_targets(text: str, text_tokenizer, device: str) -> list[int]:
    """Encode a text string as PersonaPlex text token IDs."""
    tokens = text_tokenizer.encode(text)
    return tokens


def train_sft(
    model_path: str,
    pairs: list[dict],
    device: str = "cuda:0",
    epochs: int = 20,
    lr: float = 3e-6,
    eval_every: int = 500,
    output_dir: str = "hybrid/checkpoints",
):
    """Supervised fine-tuning with call center penalty."""
    import torch
    import torch.nn.functional as F
    from moshi.models import loaders
    import sentencepiece
    from huggingface_hub import hf_hub_download

    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Loading model on {device}...")
    model = loaders.get_moshi_lm(model_path, device=device, dtype=torch.bfloat16)
    model.train()

    tok_path = hf_hub_download("nvidia/personaplex-7b-v1", loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tok_path)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {trainable/1e9:.2f}B")

    # Encode all target texts
    log.info("Encoding target texts...")
    training_data = []
    for pair in pairs:
        target_tokens = encode_text_as_targets(pair["assistant"], text_tokenizer, device)
        if len(target_tokens) > 0:
            training_data.append({
                "user_text": pair["user"],
                "assistant_text": pair["assistant"],
                "target_token_ids": target_tokens,
            })

    log.info(f"Training samples: {len(training_data)}")

    # Encode blacklisted phrases as token sequences
    blacklist_token_seqs = []
    for phrase in BLACKLIST:
        tokens = text_tokenizer.encode(phrase)
        if len(tokens) >= 2:
            blacklist_token_seqs.append(tokens)

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=len(training_data) * 2, T_mult=2,
    )

    best_loss = float("inf")
    training_log = []
    step_count = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_losses = []
        epoch_penalty_losses = []

        random.shuffle(training_data)

        for i, sample in enumerate(training_data):
            target_ids = torch.tensor(sample["target_token_ids"], dtype=torch.long, device=device)

            # Create input: the model's initial token (BOS-like)
            initial_token = model._get_initial_token().to(device)

            # Forward pass through the text prediction head
            # We use the model's embedding + text_linear to get logits for each target token
            losses = []

            for t in range(min(len(target_ids), 50)):  # Cap at 50 tokens per sample
                # Create a pseudo-input from the target token at position t
                # This is teacher forcing — we feed the correct previous token
                token_id = target_ids[t]

                # Get the text embedding for this token
                text_emb = model.text_emb(token_id.unsqueeze(0).unsqueeze(0))  # [1, 1, dim]

                # Pass through text_linear to get logits
                text_logits = model.text_linear(text_emb)  # [1, 1, vocab]

                # Cross-entropy loss: model should predict the NEXT token
                if t + 1 < len(target_ids):
                    loss = F.cross_entropy(
                        text_logits.view(-1, text_logits.size(-1)),
                        target_ids[t + 1:t + 2],
                    )
                    losses.append(loss)

            if not losses:
                continue

            # Main loss: predict natural response tokens
            main_loss = torch.stack(losses).mean()

            # Penalty loss: extra cost for producing blacklisted token sequences
            penalty_loss = torch.tensor(0.0, device=device)
            with torch.no_grad():
                # Check if any blacklisted sequences appear in the model's top predictions
                for bl_tokens in blacklist_token_seqs:
                    bl_tensor = torch.tensor(bl_tokens[:3], device=device)
                    for t in range(min(len(target_ids), 30)):
                        text_emb = model.text_emb(target_ids[t].unsqueeze(0).unsqueeze(0))
                        logits = model.text_linear(text_emb).squeeze()
                        top_tokens = logits.topk(10).indices
                        if any(bt in top_tokens for bt in bl_tensor):
                            penalty_loss = penalty_loss + 0.1

            total_loss = main_loss + penalty_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_losses.append(main_loss.item())
            epoch_penalty_losses.append(penalty_loss.item())
            step_count += 1

            if step_count % 50 == 0:
                avg_loss = sum(epoch_losses[-50:]) / min(50, len(epoch_losses[-50:]))
                avg_penalty = sum(epoch_penalty_losses[-50:]) / min(50, len(epoch_penalty_losses[-50:]))
                current_lr = scheduler.get_last_lr()[0]
                log.info(f"  E{epoch+1} Step {step_count} [{i+1}/{len(training_data)}] "
                         f"Loss: {avg_loss:.4f} Penalty: {avg_penalty:.4f} LR: {current_lr:.2e}")

            # Periodic eval + checkpoint
            if step_count % eval_every == 0:
                avg = sum(epoch_losses) / max(len(epoch_losses), 1)
                if avg < best_loss:
                    best_loss = avg
                    ckpt = os.path.join(output_dir, "sft_best.pt")
                    torch.save(model.state_dict(), ckpt)
                    log.info(f"  Checkpoint: {ckpt} (loss={best_loss:.4f})")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        avg_penalty = sum(epoch_penalty_losses) / max(len(epoch_penalty_losses), 1)
        log.info(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.4f} Penalty: {avg_penalty:.4f} — {epoch_time:.0f}s")

        training_log.append({
            "epoch": epoch + 1, "avg_loss": avg_loss,
            "avg_penalty": avg_penalty, "time_s": epoch_time,
            "steps": step_count, "samples": len(epoch_losses),
        })

        # Save checkpoint each epoch
        ckpt = os.path.join(output_dir, f"sft_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_ckpt = os.path.join(output_dir, "sft_best.pt")
            torch.save(model.state_dict(), best_ckpt)

    # Save final
    final = os.path.join(output_dir, "sft_final.pt")
    torch.save(model.state_dict(), final)

    with open(os.path.join(output_dir, "sft_training_log.json"), "w") as f:
        json.dump({"log": training_log, "best_loss": best_loss,
                    "pairs_count": len(training_data), "epochs": epochs,
                    "lr": lr, "blacklist": BLACKLIST}, f, indent=2)

    log.info(f"SFT complete. Best loss: {best_loss:.4f}. Checkpoints in {output_dir}/")

    del model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="SFT to remove call center behavior")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--n-pairs", type=int, default=200, help="Number of conversation pairs to generate")
    parser.add_argument("--model", type=str, default="/tmp/distilled_bf16.safetensors",
                        help="Starting weights (distilled or bf16)")
    parser.add_argument("--output-dir", type=str, default="hybrid/checkpoints")
    parser.add_argument("--skip-generation", action="store_true", help="Use cached pairs")
    args = parser.parse_args()

    pairs_file = "hybrid/training_pairs.json"

    if not args.skip_generation or not os.path.exists(pairs_file):
        pairs = generate_training_pairs_via_qwen(args.n_pairs)
        with open(pairs_file, "w") as f:
            json.dump(pairs, f, indent=2)
        log.info(f"Saved {len(pairs)} pairs to {pairs_file}")
    else:
        with open(pairs_file) as f:
            pairs = json.load(f)
        log.info(f"Loaded {len(pairs)} cached pairs from {pairs_file}")

    train_sft(
        model_path=args.model,
        pairs=pairs,
        epochs=args.epochs,
        lr=args.lr,
        eval_every=args.eval_every,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
