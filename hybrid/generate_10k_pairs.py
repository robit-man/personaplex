#!/usr/bin/env python3
"""Generate 10,000+ diverse conversation pairs for SFT training."""

import json
import os
import re
import time
import random
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:4b"  # Fast model for bulk generation

BLACKLIST = [
    "how can i help", "how can i assist", "what can i do for you",
    "thank you for calling", "this is ", "my name is", "welcome to",
    "how may i", "good morning", "good afternoon", "good evening",
]

TOPIC_CATEGORIES = [
    # Software engineering
    "debugging a production issue with a REST API",
    "reviewing a pull request for a authentication module",
    "optimizing a slow database query",
    "designing a caching strategy",
    "migrating from monolith to microservices",
    "setting up CI/CD pipeline",
    "implementing rate limiting",
    "handling database migrations safely",
    "writing integration tests",
    "refactoring legacy code",
    # Deployment & ops
    "deploying to Kubernetes",
    "monitoring application health",
    "configuring auto-scaling",
    "investigating memory leaks",
    "rolling back a failed deployment",
    "setting up logging and alerting",
    "managing secrets and environment variables",
    "load testing a web service",
    # Architecture
    "choosing between SQL and NoSQL",
    "designing event-driven architecture",
    "implementing message queues",
    "building a real-time notification system",
    "designing API versioning strategy",
    "implementing circuit breakers",
    "choosing a state management approach",
    # Casual tech
    "comparing programming languages",
    "discussing AI and machine learning trends",
    "talking about open source projects",
    "opinions on cloud providers",
    "thoughts on developer experience",
    "discussing technical interviews",
    "talking about side projects",
    "opinions on TypeScript vs JavaScript",
    "discussing container orchestration",
    "thoughts on serverless architecture",
    # Task management
    "requesting a code review",
    "asking to run tests",
    "requesting deployment status",
    "asking for performance metrics",
    "requesting a security audit",
    "asking to investigate an alert",
    "requesting help with a merge conflict",
    "asking to set up a new service",
]


def generate_batch(batch_id: int, topic: str, batch_size: int = 10) -> list[dict]:
    """Generate a batch of conversation pairs."""
    prompt = f"""Generate {batch_size} natural conversation exchanges about: {topic}

Rules:
- The assistant NEVER introduces itself by name
- The assistant NEVER says "how can I help" or any variation
- The assistant NEVER says "thank you for calling" or "welcome to"
- The assistant responds directly and naturally, like a colleague
- Responses are 1-4 sentences, conversational, specific and technical
- Include varied response styles: direct answers, follow-up questions, opinions, confirmations
- Each exchange should be different — vary the user's question and assistant's tone

Output ONLY a JSON array: [{{"user":"...","assistant":"..."}}]
No markdown, no explanation, no think tags."""

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
        }, timeout=120)
        data = resp.json()
        text = data.get("response", "")
        text = re.sub(r'<think>[\s\S]*?</think>', '', text).strip()

        # Try to extract JSON
        # Sometimes model wraps in ```json ... ```
        json_match = re.search(r'\[[\s\S]*\]', text)
        if json_match:
            text = json_match.group(0)

        pairs = json.loads(text)
        if not isinstance(pairs, list):
            return []

        # Filter out blacklisted responses
        clean = []
        for item in pairs:
            if not isinstance(item, dict) or "user" not in item or "assistant" not in item:
                continue
            asst = item["assistant"].lower()
            if any(bl in asst for bl in BLACKLIST):
                continue
            if len(item["assistant"]) < 10:
                continue
            clean.append(item)

        return clean

    except Exception as e:
        return []


def main():
    target = 10000
    output_file = "hybrid/training_pairs_10k.json"
    all_pairs = []

    # Load existing pairs as seed
    if os.path.exists("hybrid/training_pairs.json"):
        with open("hybrid/training_pairs.json") as f:
            existing = json.load(f)
        all_pairs.extend(existing)
        log.info(f"Loaded {len(existing)} existing pairs as seed")

    batch_id = 0
    start_time = time.time()

    # Generate in rounds, cycling through topics
    while len(all_pairs) < target:
        round_start = time.time()
        round_pairs = 0

        # Generate batches sequentially (Ollama handles one at a time anyway)
        topics_this_round = random.sample(TOPIC_CATEGORIES, min(20, len(TOPIC_CATEGORIES)))

        for topic in topics_this_round:
            if len(all_pairs) >= target:
                break

            batch_id += 1
            pairs = generate_batch(batch_id, topic, batch_size=15)
            all_pairs.extend(pairs)
            round_pairs += len(pairs)

            if batch_id % 10 == 0:
                elapsed = time.time() - start_time
                rate = len(all_pairs) / elapsed * 60
                eta = (target - len(all_pairs)) / max(rate, 1)
                log.info(f"  Batch {batch_id}: {len(all_pairs)}/{target} pairs "
                         f"({rate:.0f}/min, ETA: {eta:.0f} min)")

        round_time = time.time() - round_start
        log.info(f"Round: +{round_pairs} pairs in {round_time:.0f}s. "
                 f"Total: {len(all_pairs)}/{target}")

        # Save incrementally every round
        with open(output_file, "w") as f:
            json.dump(all_pairs[:target], f)

    # Final save
    all_pairs = all_pairs[:target]
    with open(output_file, "w") as f:
        json.dump(all_pairs, f, indent=2)

    elapsed = time.time() - start_time
    log.info(f"\nDone: {len(all_pairs)} pairs in {elapsed/60:.0f} minutes")
    log.info(f"Saved to {output_file}")

    # Stats
    avg_user_len = sum(len(p["user"]) for p in all_pairs) / len(all_pairs)
    avg_asst_len = sum(len(p["assistant"]) for p in all_pairs) / len(all_pairs)
    log.info(f"Avg user length: {avg_user_len:.0f} chars")
    log.info(f"Avg assistant length: {avg_asst_len:.0f} chars")


if __name__ == "__main__":
    main()
