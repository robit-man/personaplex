"""Inspect a pinned PersonaPlex LM and emit its immutable training contract."""

from __future__ import annotations

import sys
from pathlib import Path
GTFT_TOOL_ROOT = Path(__file__).resolve().parents[2]
if str(GTFT_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(GTFT_TOOL_ROOT))

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

import torch

from ground_truth_finetuning.training.contracts import StreamLayout, canonical_json
from ground_truth_finetuning.training.native_source import moshi_source_fingerprint


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--mimi-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    source_root = args.moshi_source_root.resolve()
    weights = args.moshi_path.resolve()
    mimi_weights = args.mimi_path.resolve()
    tokenizer = args.tokenizer_path.resolve()
    if not weights.is_file():
        raise SystemExit(f"LM weights do not exist: {weights}")
    if not mimi_weights.is_file() or not tokenizer.is_file():
        raise SystemExit("matching Mimi weights and SentencePiece tokenizer are required")
    sys.path.insert(0, str(source_root))
    from moshi.models.loaders import get_moshi_lm

    source_fingerprint = moshi_source_fingerprint(source_root)
    lm = get_moshi_lm(weights, device=args.device, dtype=torch.bfloat16)
    layout = StreamLayout.from_mapping(
        {
            "text_stream_indices": [0],
            "agent_audio_stream_indices": list(range(1, 9)),
            "caller_audio_stream_indices": list(range(9, 17)),
        }
    )
    layout.validate_for_model(lm)
    delays = [int(delay) for delay in lm.delays]
    contract = {
        "schema_version": 3,
        "kind": "personaplex-native-model-contract",
        "model_revision": args.model_revision,
        "moshi_weights_sha256": sha256_file(weights),
        "mimi_weights_sha256": sha256_file(mimi_weights),
        "tokenizer_sha256": sha256_file(tokenizer),
        "moshi_source_fingerprint_schema": source_fingerprint["schema"],
        "moshi_source_sha256": source_fingerprint["sha256"],
        "moshi_source_file_count": source_fingerprint["file_count"],
        "num_codebooks": int(lm.num_codebooks),
        "audio_offset": int(lm.audio_offset),
        "dep_q": int(lm.dep_q),
        "delays": delays,
        "delay_config_sha256": "sha256:" + sha256(canonical_json(delays).encode()).hexdigest(),
        "text_padding_token_id": int(lm.text_padding_token_id),
        "end_of_text_padding_id": int(lm.end_of_text_padding_id),
        "zero_token_id": int(lm.zero_token_id),
        "stream_layout": layout.as_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"model_revision": args.model_revision, "delay_config_sha256": contract["delay_config_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
