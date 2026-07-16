#!/usr/bin/env python3
"""Fail-closed tensor certification for controlled native PersonaPlex examples."""

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
import wave
from typing import Any

from ground_truth_finetuning.training.contracts import ContractError, validate_control_frame_mapping
from ground_truth_finetuning.tools.certify_corpus import explicit_indices, hash_file, load_tensor, resolve_under


TOOL_VERSION = "gtft-controlled-native-corpus-certifier-v1"
REQUIRED_SPLITS = ("train", "validation", "test")


def rows(path: Path) -> list[dict[str, Any]]:
    result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not result:
        raise ValueError("manifest contains no examples")
    return result


def split_coverage(manifest_rows: list[dict[str, Any]]) -> tuple[dict[str, int], list[str]]:
    """Require a true held-out validation and test partition before promotion."""
    counts = {split: 0 for split in REQUIRED_SPLITS}
    invalid = set()
    for item in manifest_rows:
        split = item.get("split")
        if split in counts:
            counts[split] += 1
        else:
            invalid.add(repr(split))
    failures = []
    if invalid:
        failures.append(f"unknown split assignments: {', '.join(sorted(invalid))}")
    missing = [split for split, count in counts.items() if count == 0]
    if missing:
        failures.append(f"missing required split coverage: {', '.join(missing)}")
    return counts, failures


def certify_item(item: dict[str, Any], artifact_root: Path, precodec_root: Path) -> list[str]:
    errors: list[str] = []
    example_id = item.get("example_id")
    try:
        if item.get("schema") != "personaplex.controlled-native-precodec.v1":
            raise ValueError("unsupported controlled-native pre-codec schema")
        frame = validate_control_frame_mapping(item.get("control", {}).get("frame", {}))
        if frame.plan.mode != "expressive":
            raise ValueError("strict controls are not adapter-training data")
        duplex = item.get("duplex", {})
        audio_path = resolve_under(precodec_root, duplex.get("path"), "duplex.path")
        if hash_file(audio_path) != duplex.get("sha256"):
            raise ValueError("duplex SHA-256 mismatch")
        with wave.open(str(audio_path), "rb") as audio:
            if audio.getframerate() != 24_000 or audio.getnchannels() != 2 or audio.getsampwidth() != 2:
                raise ValueError("duplex audio must be 24 kHz signed-16-bit stereo")
        target = item.get("target", {})
        if not all(isinstance(target.get(key), int) for key in ("start_ms", "end_ms", "rendered_end_ms")) or not 0 <= target["start_ms"] < target["end_ms"] <= target["rendered_end_ms"]:
            raise ValueError("target audible bounds are invalid")
        quality = item.get("quality", {})
        if quality.get("accepted") is not True or not isinstance(quality.get("wer"), (int, float)) or quality["wer"] > quality.get("maxAsrWer", 0):
            raise ValueError("source ASR quality is not accepted")
    except (ContractError, ValueError, OSError, wave.Error) as error:
        errors.append(f"source: {error}")
    encoding = item.get("model_encoding", {})
    required = {"model_revision", "codebook_layout", "delay_config_sha256", "codec", "codes_path", "codes_sha256", "target_mask_path", "target_mask_sha256", "text_alignment_path", "text_alignment_sha256", "prefix_at"}
    if not isinstance(encoding, dict) or required.difference(encoding):
        return errors + [f"model_encoding missing {sorted(required.difference(encoding if isinstance(encoding, dict) else {}))}"]
    try:
        codec = encoding["codec"]
        if not isinstance(codec, dict) or not all(
            isinstance(codec.get(key), str) and codec[key].startswith("sha256:")
            for key in ("mimi_weights_sha256", "tokenizer_sha256")
        ):
            raise ValueError("native codec/tokenizer provenance is missing")
        if not isinstance(codec.get("frame_rate_hz"), (int, float)) or codec["frame_rate_hz"] <= 0:
            raise ValueError("native codec frame rate is invalid")
        codes_path = resolve_under(artifact_root, encoding["codes_path"], "codes_path")
        mask_path = resolve_under(artifact_root, encoding["target_mask_path"], "target_mask_path")
        alignment_path = resolve_under(artifact_root, encoding["text_alignment_path"], "text_alignment_path")
        for path, expected in ((codes_path, encoding["codes_sha256"]), (mask_path, encoding["target_mask_sha256"]), (alignment_path, encoding["text_alignment_sha256"])):
            if hash_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {path.name}")
        codes = load_tensor(codes_path, "codes")
        mask = load_tensor(mask_path, "target_mask")
        if codes.ndim != 2 or mask.shape != codes.shape or str(mask.dtype) != "torch.bool":
            raise ValueError("codes and bool target mask must have the same [K,T] shape")
        text_indices = explicit_indices(encoding["codebook_layout"], "text", codes.shape[0])
        agent_indices = explicit_indices(encoding["codebook_layout"], "agent_audio", codes.shape[0])
        caller_indices = explicit_indices(encoding["codebook_layout"], "caller_audio", codes.shape[0])
        if text_indices & agent_indices or text_indices & caller_indices or agent_indices & caller_indices or text_indices | agent_indices | caller_indices != set(range(codes.shape[0])):
            raise ValueError("codebook layout must be a disjoint complete partition")
        if mask[list(caller_indices)].any().item() or not mask[list(text_indices)].any().item() or not mask[list(agent_indices)].any().item():
            raise ValueError("caller supervision is forbidden and current agent text/audio supervision is required")
        alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
        if alignment.get("verified") is not True or alignment.get("codes_sha256") != encoding["codes_sha256"]:
            raise ValueError("alignment verification does not match code tensor")
        target_frames = alignment.get("target_frames")
        if not isinstance(target_frames, list) or len(target_frames) != 2 or not 0 < target_frames[0] < target_frames[1] <= codes.shape[1]:
            raise ValueError("target frame bounds are invalid")
        if mask[list(agent_indices), : target_frames[0]].any().item() or mask[list(agent_indices), target_frames[1] :].any().item():
            raise ValueError("agent mask supervises context outside the current target turn")
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as error:
        errors.append(f"encoding: {error}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--precodec-root", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    failures = []
    revisions = set()
    codec_artifacts = set()
    manifest_rows = rows(manifest)
    split_counts, coverage_failures = split_coverage(manifest_rows)
    for item in manifest_rows:
        revisions.add(item.get("model_encoding", {}).get("model_revision"))
        codec = item.get("model_encoding", {}).get("codec")
        if isinstance(codec, dict):
            mimi = codec.get("mimi_weights_sha256")
            tokenizer = codec.get("tokenizer_sha256")
            if isinstance(mimi, str) and isinstance(tokenizer, str):
                codec_artifacts.add((mimi, tokenizer))
        errors = certify_item(item, args.artifact_root.resolve(), args.precodec_root.resolve())
        if errors:
            failures.append({"example_id": item.get("example_id"), "errors": errors})
    if len(codec_artifacts) != 1:
        failures.append({"example_id": None, "errors": ["encoding: corpus mixes codec/tokenizer artifacts"]})
    report = {
        "schema_version": 2, "kind": "personaplex-corpus-certificate", "tool_version": TOOL_VERSION,
        "manifest": str(manifest), "manifest_sha256": hash_file(manifest),
        "artifact_root": str(args.artifact_root.resolve()), "model_revisions": sorted(str(value) for value in revisions if value),
        "items": len(manifest_rows), "failed_items": len(failures),
        "split_counts": split_counts,
        "coverage_failures": coverage_failures,
        "status": (
            "certified_for_adapter_training"
            if not failures and not coverage_failures
            else "insufficient_split_coverage" if not failures else "failed"
        ),
        "failures": failures, "caller_stream_supervision": "forbidden",
        "codec_artifacts": [
            {"mimi_weights_sha256": mimi, "tokenizer_sha256": tokenizer}
            for mimi, tokenizer in sorted(codec_artifacts)
        ],
    }
    args.certificate.parent.mkdir(parents=True, exist_ok=True)
    args.certificate.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "items": report["items"], "failed_items": report["failed_items"], "coverage_failures": len(coverage_failures)}))
    return 0 if report["status"] == "certified_for_adapter_training" else 1


if __name__ == "__main__":
    raise SystemExit(main())
