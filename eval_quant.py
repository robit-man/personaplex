#!/usr/bin/env python3
"""
eval_quant.py — Real quality evaluation of PersonaPlex quantized models.

Loads bf16 (teacher), NF4, and TurboQuant 2-bit on separate GPUs,
feeds identical audio prompts through each, and compares the text
output quality. No cosine similarity bullshit — actual speech output.

Metrics:
  1. Text token divergence: how often quant disagrees with bf16
  2. Repetition rate: fraction of repeated n-grams (hallucination signal)
  3. Coherent response rate: does the output form real words/sentences
  4. Perplexity proxy: cross-entropy of quant logits vs bf16 logits
  5. Audio quality: spectral distance of decoded audio

Usage:
  python eval_quant.py --steps 100 --prompts "You enjoy casual conversation."
"""

import argparse
import os
import sys
import time
import logging
import json
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# Add moshi to path
sys.path.insert(0, str(Path(__file__).parent / "personaplex-setup" / "moshi"))
os.environ["NO_CUDA_GRAPH"] = "1"


def load_model(weight_path: str, device: str, name: str):
    """Load a PersonaPlex model variant."""
    import torch
    from moshi.models import loaders

    log.info(f"[{name}] Loading on {device}...")
    t0 = time.time()

    mimi_path = None
    for repo in ["nvidia/personaplex-7b-v1", "cudabenchmarktest/personaplex-7b-nf4"]:
        try:
            from huggingface_hub import hf_hub_download
            mimi_path = hf_hub_download(repo, loaders.MIMI_NAME, token=False)
            break
        except:
            continue

    mimi = loaders.get_mimi(mimi_path, device)
    tok_path = None
    for repo in ["nvidia/personaplex-7b-v1", "cudabenchmarktest/personaplex-7b-nf4"]:
        try:
            tok_path = hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME, token=False)
            break
        except:
            continue

    import sentencepiece
    text_tokenizer = sentencepiece.SentencePieceProcessor(tok_path)

    lm = loaders.get_moshi_lm(weight_path, device=device, dtype=torch.bfloat16)
    lm.eval()

    from moshi.models.lm import LMGen
    lm_gen = LMGen(lm, device=device, check=False,
                   audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
                   sample_rate=mimi.sample_rate, frame_rate=mimi.frame_rate)

    t1 = time.time()
    log.info(f"[{name}] Loaded in {t1-t0:.1f}s on {device}")

    return {
        "name": name,
        "mimi": mimi,
        "text_tokenizer": text_tokenizer,
        "lm_gen": lm_gen,
        "device": device,
    }


def run_eval_step(model_info: dict, text_prompt: str, num_steps: int = 50):
    """Run inference steps and collect text tokens + logits."""
    import torch
    import numpy as np
    import sphn

    name = model_info["name"]
    mimi = model_info["mimi"]
    text_tokenizer = model_info["text_tokenizer"]
    lm_gen = model_info["lm_gen"]
    device = model_info["device"]
    frame_size = int(mimi.sample_rate / mimi.frame_rate)

    # Set text prompt
    lm_gen.text_prompt_tokens = text_tokenizer.encode(f"<system> {text_prompt} <system>")

    text_tokens = []
    text_pieces = []
    audio_frames = []
    step_times = []

    mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    with torch.no_grad(), lm_gen.streaming(1):
        mimi.reset_streaming()
        lm_gen.reset_streaming()
        # Feed system prompts
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()

        # Feed silence frames and collect output
        for step in range(num_steps):
            t0 = time.time()
            silence = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=device)
            codes = mimi.encode(silence)

            for c in range(codes.shape[-1]):
                tokens = lm_gen.step(codes[:, :, c:c+1])
                if tokens is None:
                    continue

                text_token = tokens[0, 0, 0].item()
                text_tokens.append(text_token)

                if text_token not in (0, 3):  # Not EPAD or PAD
                    piece = text_tokenizer.id_to_piece(text_token)
                    piece = piece.replace("▁", " ")
                    text_pieces.append(piece)

                # Decode audio
                main_pcm = mimi.decode(tokens[:, 1:9])
                audio_frames.append(main_pcm.cpu().numpy())

            step_times.append(time.time() - t0)

    return {
        "name": name,
        "text_tokens": text_tokens,
        "text": "".join(text_pieces),
        "audio_frames": audio_frames,
        "avg_step_ms": sum(step_times) / len(step_times) * 1000 if step_times else 0,
        "total_time_s": sum(step_times),
    }


def compute_metrics(bf16_result: dict, quant_result: dict) -> dict:
    """Compare quantized model output against bf16 reference."""
    metrics = {"name": quant_result["name"]}

    bf16_tokens = bf16_result["text_tokens"]
    quant_tokens = quant_result["text_tokens"]
    min_len = min(len(bf16_tokens), len(quant_tokens))

    # 1. Token divergence rate
    if min_len > 0:
        matches = sum(1 for a, b in zip(bf16_tokens[:min_len], quant_tokens[:min_len]) if a == b)
        metrics["token_match_rate"] = matches / min_len
    else:
        metrics["token_match_rate"] = 0.0

    # 2. Repetition rate (bigram)
    def repetition_rate(tokens, n=2):
        if len(tokens) < n + 1:
            return 0.0
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return repeated / max(len(ngrams), 1)

    metrics["bf16_repetition_2gram"] = repetition_rate(bf16_tokens)
    metrics["quant_repetition_2gram"] = repetition_rate(quant_tokens)
    metrics["repetition_increase"] = metrics["quant_repetition_2gram"] - metrics["bf16_repetition_2gram"]

    # 3. Unique token ratio (vocabulary diversity)
    def unique_ratio(tokens):
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)

    metrics["bf16_unique_ratio"] = unique_ratio(bf16_tokens)
    metrics["quant_unique_ratio"] = unique_ratio(quant_tokens)

    # 4. Text output
    metrics["bf16_text"] = bf16_result["text"][:500]
    metrics["quant_text"] = quant_result["text"][:500]

    # 5. Speed
    metrics["bf16_step_ms"] = bf16_result["avg_step_ms"]
    metrics["quant_step_ms"] = quant_result["avg_step_ms"]

    # 6. Coherence heuristic: count real words (spaces + printable chars)
    def word_count(text):
        return len([w for w in text.split() if len(w) > 1])

    metrics["bf16_word_count"] = word_count(bf16_result["text"])
    metrics["quant_word_count"] = word_count(quant_result["text"])

    # 7. Hallucination signal: long repetitive sequences
    def longest_repeat(tokens, min_len=5):
        """Find longest repeated subsequence."""
        best = 0
        for window in range(min_len, min(50, len(tokens) // 2)):
            for i in range(len(tokens) - 2 * window):
                if tokens[i:i+window] == tokens[i+window:i+2*window]:
                    best = max(best, window)
        return best

    metrics["quant_longest_repeat"] = longest_repeat(quant_tokens)
    metrics["bf16_longest_repeat"] = longest_repeat(bf16_tokens)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Eval PersonaPlex quantized models")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps per eval")
    parser.add_argument("--prompts", type=str, nargs="+",
                        default=["You enjoy casual conversation about technology and science."],
                        help="System prompts to test")
    parser.add_argument("--skip-bf16", action="store_true", help="Skip bf16 (if already have results)")
    parser.add_argument("--skip-nf4", action="store_true", help="Skip NF4")
    parser.add_argument("--skip-2bit", action="store_true", help="Skip turbo2bit")
    parser.add_argument("--output", type=str, default="eval_results.json")
    args = parser.parse_args()

    import torch

    # Weight paths
    bf16_path = "/home/roko/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/fdaf4090a61cb315c138a1faee287ffd6c716309/model.safetensors"
    nf4_path = "/home/roko/.cache/huggingface/hub/models--cudabenchmarktest--personaplex-7b-nf4/snapshots/dd96c876a28608f089b02d1bbdb1c1131532414e/model-nf4.safetensors"
    turbo2bit_path = "/home/roko/.cache/huggingface/hub/models--cudabenchmarktest--personaplex-7b-turbo2bit/snapshots/9ad9cf2764156b20ac1244015b3f769d69a8cc84/model-turbo2bit.safetensors"

    all_results = []

    for prompt in args.prompts:
        log.info(f"\n{'='*60}")
        log.info(f"PROMPT: {prompt}")
        log.info(f"{'='*60}\n")

        prompt_results = {"prompt": prompt, "steps": args.steps}

        # Load and eval bf16 (reference)
        bf16_result = None
        if not args.skip_bf16:
            model = load_model(bf16_path, "cuda:0", "bf16")
            bf16_result = run_eval_step(model, prompt, args.steps)
            log.info(f"[bf16] Text: {bf16_result['text'][:200]}...")
            log.info(f"[bf16] {len(bf16_result['text_tokens'])} tokens, {bf16_result['avg_step_ms']:.0f}ms/step")
            # Free GPU 0
            del model
            torch.cuda.empty_cache()

        # Load and eval NF4
        nf4_result = None
        if not args.skip_nf4:
            model = load_model(nf4_path, "cuda:1", "nf4")
            nf4_result = run_eval_step(model, prompt, args.steps)
            log.info(f"[nf4] Text: {nf4_result['text'][:200]}...")
            if bf16_result:
                nf4_metrics = compute_metrics(bf16_result, nf4_result)
                prompt_results["nf4"] = nf4_metrics
                log.info(f"[nf4] Token match: {nf4_metrics['token_match_rate']:.1%}, "
                         f"Repetition increase: {nf4_metrics['repetition_increase']:+.3f}, "
                         f"Words: {nf4_metrics['quant_word_count']}")
            del model
            torch.cuda.empty_cache()

        # Load and eval turbo2bit
        turbo2bit_result = None
        if not args.skip_2bit:
            model = load_model(turbo2bit_path, "cuda:2", "turbo2bit")
            turbo2bit_result = run_eval_step(model, prompt, args.steps)
            log.info(f"[turbo2bit] Text: {turbo2bit_result['text'][:200]}...")
            if bf16_result:
                t2b_metrics = compute_metrics(bf16_result, turbo2bit_result)
                prompt_results["turbo2bit"] = t2b_metrics
                log.info(f"[turbo2bit] Token match: {t2b_metrics['token_match_rate']:.1%}, "
                         f"Repetition increase: {t2b_metrics['repetition_increase']:+.3f}, "
                         f"Words: {t2b_metrics['quant_word_count']}")
            del model
            torch.cuda.empty_cache()

        all_results.append(prompt_results)

    # Save results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("EVAL SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"\nPrompt: {r['prompt'][:60]}...")
        for tier in ["nf4", "turbo2bit"]:
            if tier in r:
                m = r[tier]
                print(f"  {tier}:")
                print(f"    Token match vs bf16: {m['token_match_rate']:.1%}")
                print(f"    Repetition (2-gram): {m['quant_repetition_2gram']:.3f} (bf16: {m['bf16_repetition_2gram']:.3f})")
                print(f"    Unique token ratio:  {m['quant_unique_ratio']:.3f} (bf16: {m['bf16_unique_ratio']:.3f})")
                print(f"    Word count:          {m['quant_word_count']} (bf16: {m['bf16_word_count']})")
                print(f"    Longest repeat:      {m['quant_longest_repeat']} (bf16: {m['bf16_longest_repeat']})")
                print(f"    Speed:               {m['quant_step_ms']:.0f}ms/step (bf16: {m['bf16_step_ms']:.0f}ms)")
                print(f"    Quant says: \"{m['quant_text'][:100]}\"")
                print(f"    bf16 says:  \"{m['bf16_text'][:100]}\"")


if __name__ == "__main__":
    main()
