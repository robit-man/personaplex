#!/usr/bin/env python3
"""Encode controlled duplex pre-codec examples into native PersonaPlex tensors."""

from __future__ import annotations

import sys
from pathlib import Path
GTFT_TOOL_ROOT = Path(__file__).resolve().parents[2]
if str(GTFT_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(GTFT_TOOL_ROOT))

import argparse
from collections import deque
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

from ground_truth_finetuning.tools.encode_native_adapter_tensors import (
    hash_file,
    load_contract,
    read_stereo_wav,
    require_codec_artifacts,
)
from ground_truth_finetuning.training.contracts import (
    ContractError,
    assert_evidence_control_alignment,
    validate_control_frame_mapping,
    validate_evidence_frame_mapping,
)
from ground_truth_finetuning.training.native_source import require_moshi_source_contract


TOOL_VERSION = "gtft-controlled-native-adapter-encoder-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("manifest contains no rows")
    return rows


def build_main_text_tokens_and_mask(
    *, tokenizer: Any, alignments: list[Any], frames: int, frame_rate: float,
    text_padding: int, end_of_text_padding: int, zero_padding: int, device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenized = []
    for entry in sorted(alignments, key=lambda value: value[1][0]):
        if not isinstance(entry, list) or len(entry) != 3 or entry[2] != "SPEAKER_MAIN":
            continue
        word, timestamp, _speaker = entry
        if not isinstance(word, str) or not isinstance(timestamp, list) or len(timestamp) != 2:
            raise ValueError("invalid target alignment entry")
        start, end = timestamp
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or not 0 <= start < end:
            raise ValueError("invalid target alignment timing")
        tokens = list(tokenizer.encode(word.strip()))
        if not tokens:
            raise ValueError("target alignment word encoded to no SentencePiece tokens")
        tokenized.append((tokens, (float(start), float(end))))
    if not tokenized:
        raise ValueError("no target word alignments")
    if frames < 1:
        raise ValueError("native codec emitted no frames")
    text_tokens = [text_padding] * frames
    target_mask = [False] * frames
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
            target_mask[frame] = True
        elif frame <= last_word_end:
            text_tokens[frame] = text_padding
    if not any(target_mask):
        raise ValueError("target alignment produced no supervised text tokens")
    return (
        torch.tensor(text_tokens, device=device, dtype=torch.long).view(1, 1, -1),
        torch.tensor(target_mask, device=device, dtype=torch.bool).view(1, 1, -1),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--precodec-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--mimi-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    precodec_root = args.precodec_root.resolve()
    artifact_root = args.artifact_root.resolve()
    contract, layout = load_contract(args.model_contract.resolve())
    require_moshi_source_contract(args.moshi_source_root.resolve(), contract)
    if not args.mimi_path.is_file() or not args.tokenizer_path.is_file():
        raise SystemExit("matching Mimi weights and text tokenizer are required")
    codec_artifacts = require_codec_artifacts(
        contract, args.mimi_path.resolve(), args.tokenizer_path.resolve()
    )
    sys.path.insert(0, str(args.moshi_source_root.resolve()))
    import sentencepiece
    from moshi.models.loaders import get_mimi

    mimi = get_mimi(args.mimi_path.resolve(), device=args.device)
    mimi.eval()
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer_path.resolve()))
    labels = {row["example_id"]: row for row in read_jsonl(precodec_root / "control_labels.jsonl")}
    source_rows = {row["example_id"]: row for row in read_jsonl(manifest)}
    if set(labels) != set(source_rows):
        raise SystemExit("pre-codec manifest and control labels have different example ids")
    (artifact_root / "tensors").mkdir(parents=True, exist_ok=True)
    (artifact_root / "alignments").mkdir(exist_ok=True)
    encoded_rows = []
    for example_id, source_row in sorted(source_rows.items()):
        label = labels[example_id]
        frame = validate_control_frame_mapping(label.get("control_frame", {}))
        evidence_mapping = label.get("evidence_frame")
        evidence = None
        if evidence_mapping is not None:
            evidence = validate_evidence_frame_mapping(evidence_mapping)
            assert_evidence_control_alignment(frame, evidence)
        if frame.plan.mode != "expressive":
            raise SystemExit(f"{example_id}: strict controls cannot enter prefix training")
        if source_row.get("control", {}).get("frame") != label.get("control_frame"):
            raise SystemExit(f"{example_id}: control frame differs between manifest and labels")
        if (source_row.get("evidence") or {}).get("frame") != evidence_mapping:
            raise SystemExit(f"{example_id}: evidence frame differs between manifest and labels")
        audio_rel = source_row.get("duplex", {}).get("path")
        info_rel = source_row.get("labels", {}).get("alignment_path")
        if not isinstance(audio_rel, str) or not isinstance(info_rel, str):
            raise SystemExit(f"{example_id}: pre-codec audio/alignment paths are missing")
        wav_path = precodec_root / audio_rel
        info_path = precodec_root / info_rel
        info = json.loads(info_path.read_text(encoding="utf-8"))
        alignments = info.get("alignments")
        ground_truth = info.get("ground_truth", {})
        start_seconds = ground_truth.get("agent_start_seconds")
        end_seconds = ground_truth.get("agent_end_seconds")
        if not isinstance(start_seconds, (int, float)) or not isinstance(end_seconds, (int, float)) or not 0 <= start_seconds < end_seconds:
            raise SystemExit(f"{example_id}: audible target bounds missing from alignment sidecar")
        audio = read_stereo_wav(wav_path)
        with torch.no_grad():
            audio_tokens = mimi.encode(torch.from_numpy(audio).to(args.device)[:, None])
        if audio_tokens.ndim != 3 or audio_tokens.shape[:2] != (2, 8):
            raise SystemExit(f"{example_id}: Mimi did not emit two eight-codebook streams")
        frames = int(audio_tokens.shape[-1])
        duration = frames / float(mimi.frame_rate)
        text_tokens, text_mask = build_main_text_tokens_and_mask(
            tokenizer=tokenizer, alignments=alignments, frames=frames,
            frame_rate=float(mimi.frame_rate), text_padding=int(contract["text_padding_token_id"]),
            end_of_text_padding=int(contract["end_of_text_padding_id"]), zero_padding=int(contract["zero_token_id"]),
            device=args.device,
        )
        if text_tokens.shape != (1, 1, frames) or text_mask.shape != (1, 1, frames):
            raise SystemExit(f"{example_id}: text stream frame count mismatch")
        codes = torch.cat([text_tokens, audio_tokens.reshape(1, 16, frames)], dim=1).squeeze(0).to(torch.long).cpu()
        if codes.shape[0] != int(contract["num_codebooks"]):
            raise SystemExit(f"{example_id}: codebook count differs from native model contract")
        start_frame = max(1, min(frames - 1, int(math.floor(float(start_seconds) * float(mimi.frame_rate)))))
        end_frame = max(start_frame + 1, min(frames, int(math.ceil(float(end_seconds) * float(mimi.frame_rate)))) )
        target_mask = torch.zeros_like(codes, dtype=torch.bool)
        target_mask[0] = text_mask.squeeze(0).squeeze(0).cpu()
        agent_indices = list(layout.agent_audio_stream_indices)
        target_mask[agent_indices, start_frame:end_frame] = (
            codes[agent_indices, start_frame:end_frame] != int(contract["zero_token_id"])
        )
        if target_mask[list(layout.caller_audio_stream_indices)].any():
            raise SystemExit(f"{example_id}: caller stream target bits are forbidden")
        if not target_mask[0].any() or not target_mask[agent_indices].any():
            raise SystemExit(f"{example_id}: current agent target has no supervised text/audio")
        stem = example_id.removeprefix("sha256:")
        tensor_rel = Path("tensors") / f"{stem}.pt"
        mask_rel = Path("tensors") / f"{stem}.mask.pt"
        alignment_rel = Path("alignments") / f"{stem}.json"
        torch.save({"codes": codes}, artifact_root / tensor_rel)
        torch.save({"target_mask": target_mask}, artifact_root / mask_rel)
        alignment = {
            "schema_version": 2,
            "verified": True,
            "verified_by": TOOL_VERSION,
            "model_revision": contract["model_revision"],
            "codes_sha256": hash_file(artifact_root / tensor_rel),
            "target_alignment_sha256": hash_file(info_path),
            "target_label_sha256": label.get("target_label_sha256"),
            "prefix_at": start_frame,
            "target_frames": [start_frame, end_frame],
            "frame_rate": float(mimi.frame_rate),
            "stream_layout": layout.as_dict(),
        }
        (artifact_root / alignment_rel).write_text(json.dumps(alignment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        updated = dict(source_row)
        updated["model_encoding"] = {
            "model_revision": contract["model_revision"],
            "codebook_layout": {"text": list(layout.text_stream_indices), "agent_audio": agent_indices, "caller_audio": list(layout.caller_audio_stream_indices)},
            "delay_config_sha256": contract["delay_config_sha256"],
            "codec": {**codec_artifacts, "frame_rate_hz": float(mimi.frame_rate)},
            "codes_path": str(tensor_rel), "codes_sha256": hash_file(artifact_root / tensor_rel),
            "target_mask_path": str(mask_rel), "target_mask_sha256": hash_file(artifact_root / mask_rel),
            "text_alignment_path": str(alignment_rel), "text_alignment_sha256": hash_file(artifact_root / alignment_rel),
            "prefix_at": start_frame,
        }
        encoded_rows.append(updated)
    encoded_path = artifact_root / "encoded_examples.jsonl"
    encoded_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in encoded_rows), encoding="utf-8")
    report = {
        "schema_version": 2, "kind": "personaplex-controlled-native-adapter-encoding",
        "tool_version": TOOL_VERSION, "status": "encoded_pending_tensor_certification",
        "items": len(encoded_rows), "model_revision": contract["model_revision"],
        "manifest": str(encoded_path), "manifest_sha256": hash_file(encoded_path),
        "stream_layout": layout.as_dict(), "caller_stream_supervision": "forbidden",
        "codec": codec_artifacts,
    }
    (artifact_root / "encoding_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "items": report["items"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, ValueError, OSError) as error:
        print(f"controlled native encoding refused: {error}", file=sys.stderr)
        raise
