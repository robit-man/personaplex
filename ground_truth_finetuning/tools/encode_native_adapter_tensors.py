"""Encode certified stereo examples for typed semantic-prefix adapter training."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from collections import deque
import math
import subprocess
import sys
import wave
from typing import Any

import numpy as np
import torch

from ground_truth_finetuning.training.contracts import ContractError, StreamLayout, canonical_json, validate_plan_mapping
from ground_truth_finetuning.tools.export_moshi_finetune_dataset import UPSTREAM_REVISION


TOOL_VERSION = "gtft-native-adapter-encoder-v1"


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: expected object")
        rows.append(row)
    if not rows:
        raise ValueError("no examples were provided")
    return rows


def read_stereo_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as input_file:
        if input_file.getframerate() != 24_000 or input_file.getnchannels() != 2:
            raise ValueError(f"{path.name} must be a 24 kHz stereo WAV")
        if input_file.getsampwidth() != 2:
            raise ValueError(f"{path.name} must be signed 16-bit PCM")
        samples = np.frombuffer(input_file.readframes(input_file.getnframes()), dtype="<i2")
    if samples.size == 0 or samples.size % 2:
        raise ValueError(f"{path.name} contains invalid stereo samples")
    return (samples.reshape(-1, 2).T.astype(np.float32) / 32768.0).copy()


def load_contract(path: Path) -> tuple[dict[str, Any], StreamLayout]:
    value = json.loads(path.read_text())
    required = {
        "model_revision",
        "num_codebooks",
        "audio_offset",
        "dep_q",
        "delays",
        "delay_config_sha256",
        "text_padding_token_id",
        "end_of_text_padding_id",
        "zero_token_id",
        "stream_layout",
    }
    if required.difference(value):
        raise ValueError(f"native model contract is missing {sorted(required.difference(value))}")
    layout = StreamLayout.from_mapping(value["stream_layout"])
    class ModelShape:
        num_codebooks = int(value["num_codebooks"])
        audio_offset = int(value["audio_offset"])
        dep_q = int(value["dep_q"])
    layout.validate_for_model(ModelShape())
    expected_delay_hash = "sha256:" + sha256(canonical_json(value["delays"]).encode()).hexdigest()
    if expected_delay_hash != value["delay_config_sha256"]:
        raise ValueError("native model delay hash does not match delays")
    return value, layout


def require_upstream_revision(root: Path) -> None:
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Moshi-Finetune checkout is {revision}; expected pinned {UPSTREAM_REVISION}")


def build_main_text_tokens(
    *,
    tokenizer: Any,
    alignments: list[Any],
    duration_seconds: float,
    frame_rate: float,
    text_padding: int,
    end_of_text_padding: int,
    zero_padding: int,
    device: str,
) -> torch.Tensor:
    """Pinned port of Moshi-Finetune Interleaver for SPEAKER_MAIN words only.

    The PersonaPlex source tree intentionally omits Moshi's optional conditioning
    module, which prevents importing the upstream package itself. This function
    preserves the relevant upstream v2acc879 algorithm verbatim in behavior:
    each word is SentencePiece-tokenized, starts on the first eligible 12.5 Hz
    frame, uses in-word padding, and closes a padding run with the EOT padding id.
    """
    tokenized = []
    for entry in sorted(alignments, key=lambda value: value[1][0]):
        if not isinstance(entry, list) or len(entry) != 3 or entry[2] != "SPEAKER_MAIN":
            continue
        word, timestamp, _speaker = entry
        if not isinstance(word, str) or not isinstance(timestamp, list) or len(timestamp) != 2:
            raise ValueError("invalid SPEAKER_MAIN alignment entry")
        start, end = timestamp
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not 0 <= start < end:
            raise ValueError("invalid SPEAKER_MAIN alignment timestamp")
        tokens = tokenizer.encode(word.strip())
        if not tokens:
            raise ValueError("target alignment word encoded to no SentencePiece tokens")
        tokenized.append((list(tokens), (float(start), float(end))))
    if not tokenized:
        raise ValueError("no valid SPEAKER_MAIN alignments")
    frames = math.ceil(duration_seconds * frame_rate)
    text_tokens = [text_padding] * frames
    index = 0
    pending: deque[int] = deque()
    last_word_end = -1
    for frame in range(frames):
        while index < len(tokenized) and tokenized[index][1][0] * frame_rate < frame + 1:
            pending = deque(tokenized[index][0])
            last_word_end = int(tokenized[index][1][1] * frame_rate)
            index += 1
        if pending:
            if frame > 0 and text_tokens[frame - 1] == text_padding:
                text_tokens[frame - 1] = end_of_text_padding
            text_tokens[frame] = pending.popleft()
        elif frame <= last_word_end:
            text_tokens[frame] = text_padding
    if not any(token not in {text_padding, end_of_text_padding, zero_padding} for token in text_tokens):
        raise ValueError("alignment did not place any target text token")
    return torch.tensor(text_tokens, device=device, dtype=torch.long).view(1, 1, -1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Original pre-codec Voryn manifest")
    parser.add_argument("--stereo-root", type=Path, required=True, help="Output of export_moshi_finetune_dataset")
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--mimi-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    stereo_root = args.stereo_root.resolve()
    artifact_root = args.artifact_root.resolve()
    contract, layout = load_contract(args.model_contract.resolve())
    if not args.mimi_path.is_file() or not args.tokenizer_path.is_file():
        raise SystemExit("Mimi weights and text tokenizer are required")
    sys.path.insert(0, str(args.moshi_source_root.resolve()))
    import sentencepiece
    from moshi.models.loaders import get_mimi

    require_upstream_revision(args.upstream_root.resolve())

    mimi = get_mimi(args.mimi_path.resolve(), device=args.device)
    mimi.eval()
    spm = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer_path.resolve()))
    labels = {
        row["example_id"]: row
        for row in read_jsonl(stereo_root / "control_labels.jsonl")
        if isinstance(row.get("example_id"), str)
    }
    source_rows = {row.get("example_id"): row for row in read_jsonl(manifest)}
    if set(labels) != set(source_rows):
        raise SystemExit("stereo control labels and source manifest must have identical example ids")
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "tensors").mkdir(exist_ok=True)
    (artifact_root / "alignments").mkdir(exist_ok=True)
    encoded_rows = []
    for example_id, label in sorted(labels.items()):
        source_row = source_rows[example_id]
        plan = validate_plan_mapping(label.get("plan", {}))
        if plan.mode != "expressive":
            raise SystemExit(f"{example_id}: strict plan cannot enter semantic prefix training")
        stem = example_id.removeprefix("sha256:")
        wav_path = stereo_root / "audio" / f"{stem}.wav"
        info_path = stereo_root / "audio" / f"{stem}.json"
        info = json.loads(info_path.read_text())
        alignments = info.get("alignments")
        if not isinstance(alignments, list) or not alignments:
            raise SystemExit(f"{example_id}: missing upstream-compatible target alignment")
        audio = read_stereo_wav(wav_path)
        with torch.no_grad():
            audio_tokens = mimi.encode(torch.from_numpy(audio).to(args.device)[:, None])
        if audio_tokens.ndim != 3 or audio_tokens.shape[0] != 2 or audio_tokens.shape[1] != 8:
            raise SystemExit(f"{example_id}: Mimi did not produce two eight-codebook streams")
        frames = audio_tokens.shape[-1]
        duration = frames / float(mimi.frame_rate)
        text_tokens = build_main_text_tokens(
            tokenizer=spm,
            alignments=alignments,
            duration_seconds=duration,
            frame_rate=float(mimi.frame_rate),
            text_padding=int(contract["text_padding_token_id"]),
            end_of_text_padding=int(contract["end_of_text_padding_id"]),
            zero_padding=int(contract["zero_token_id"]),
            device=args.device,
        ).to(audio_tokens.device)
        if text_tokens.shape != (1, 1, frames):
            raise SystemExit(f"{example_id}: text stream frame count does not match Mimi output")
        codes = torch.cat([text_tokens, audio_tokens.reshape(1, 16, frames)], dim=1).squeeze(0).to(torch.long).cpu()
        if codes.shape[0] != int(contract["num_codebooks"]):
            raise SystemExit(f"{example_id}: codebook count differs from native model contract")
        target_mask = torch.zeros_like(codes, dtype=torch.bool)
        target_mask[0] = codes[0] != int(contract["zero_token_id"])
        target_mask[list(layout.agent_audio_stream_indices)] = (
            codes[list(layout.agent_audio_stream_indices)] != int(contract["zero_token_id"])
        )
        if target_mask[list(layout.caller_audio_stream_indices)].any():
            raise SystemExit(f"{example_id}: caller stream target bits are forbidden")
        agent_start_seconds = info.get("ground_truth", {}).get("agent_start_seconds")
        if not isinstance(agent_start_seconds, (int, float)):
            raise SystemExit(f"{example_id}: measured agent start timestamp is missing")
        prefix_at = max(1, min(frames - 1, int(round(float(agent_start_seconds) * float(mimi.frame_rate)))))
        tensor_rel = Path("tensors") / f"{stem}.pt"
        mask_rel = Path("tensors") / f"{stem}.mask.pt"
        alignment_rel = Path("alignments") / f"{stem}.json"
        torch.save({"codes": codes}, artifact_root / tensor_rel)
        torch.save({"target_mask": target_mask}, artifact_root / mask_rel)
        code_hash = hash_file(artifact_root / tensor_rel)
        alignment = {
            "schema_version": 1,
            "verified": True,
            "verified_by": TOOL_VERSION,
            "model_revision": contract["model_revision"],
            "codes_sha256": code_hash,
            "target_alignment_sha256": hash_file(info_path),
            "canonical_response_sha256": label.get("canonical_response_sha256"),
            "prefix_at": prefix_at,
            "frame_rate": float(mimi.frame_rate),
            "stream_layout": layout.as_dict(),
        }
        (artifact_root / alignment_rel).write_text(json.dumps(alignment, indent=2, sort_keys=True) + "\n")
        updated = dict(source_row)
        updated["model_encoding"] = {
            "model_revision": contract["model_revision"],
            "codebook_layout": {
                "text": list(layout.text_stream_indices),
                "agent_audio": list(layout.agent_audio_stream_indices),
                "caller_audio": list(layout.caller_audio_stream_indices),
            },
            "delay_config_sha256": contract["delay_config_sha256"],
            "codes_path": str(tensor_rel),
            "codes_sha256": code_hash,
            "target_mask_path": str(mask_rel),
            "target_mask_sha256": hash_file(artifact_root / mask_rel),
            "text_alignment_path": str(alignment_rel),
            "text_alignment_sha256": hash_file(artifact_root / alignment_rel),
            "prefix_at": prefix_at,
        }
        encoded_rows.append(updated)
    encoded_path = artifact_root / "encoded_examples.jsonl"
    encoded_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in encoded_rows))
    report = {
        "schema_version": 1,
        "kind": "personaplex-native-adapter-encoding",
        "tool_version": TOOL_VERSION,
        "status": "encoded_pending_tensor_certification",
        "items": len(encoded_rows),
        "model_revision": contract["model_revision"],
        "manifest": str(encoded_path),
        "manifest_sha256": hash_file(encoded_path),
        "stream_layout": layout.as_dict(),
    }
    (artifact_root / "encoding_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "items": report["items"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
