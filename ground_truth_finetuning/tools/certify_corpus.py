"""Fail-closed tensor-level certification for PersonaPlex adapter-training data."""

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
from typing import Any

from ground_truth_finetuning.training.contracts import ContractError, validate_plan_mapping


TOOL_VERSION = "gtft-corpus-certifier-v1"


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: item must be an object")
        rows.append(value)
    if not rows:
        raise ValueError("manifest contains no items")
    return rows


def resolve_under(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a relative artifact path")
    candidate = (root / value).resolve()
    if root.resolve() not in candidate.parents and candidate != root.resolve():
        raise ValueError(f"{label} escapes its approved root")
    if not candidate.is_file():
        raise ValueError(f"{label} does not exist")
    return candidate


def explicit_indices(layout: Any, name: str, codebooks: int) -> set[int]:
    values = layout.get(name) if isinstance(layout, dict) else None
    if not isinstance(values, list) or not values or not all(isinstance(item, int) for item in values):
        raise ValueError(f"codebook_layout.{name} must be a nonempty explicit integer list")
    result = set(values)
    if len(result) != len(values) or any(item < 0 or item >= codebooks for item in result):
        raise ValueError(f"codebook_layout.{name} contains duplicate or out-of-range index")
    return result


def load_tensor(path: Path, expected_name: str):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("certification requires PyTorch to inspect tensors") from exc
    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch
        loaded = torch.load(path, map_location="cpu")
    tensor = loaded.get(expected_name) if isinstance(loaded, dict) else loaded
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"{path.name} must contain tensor {expected_name!r} or be that tensor")
    return tensor


def certify_item(item: dict[str, Any], artifact_root: Path, source_audio_root: Path) -> list[str]:
    errors: list[str] = []
    try:
        plan = validate_plan_mapping(item.get("semantics", {}).get("plan", {}))
        canonical = item.get("semantics", {}).get("canonical_response")
        if not isinstance(canonical, str) or not canonical.strip():
            raise ValueError("canonical response label is missing")
        if plan.mode != "expressive":
            raise ValueError("strict plans are not adapter-training examples")
    except (ContractError, ValueError) as exc:
        errors.append(f"plan: {exc}")
    audio = item.get("audio", {})
    for role in ("caller", "agent"):
        try:
            file_path = resolve_under(source_audio_root, audio.get(f"{role}_path"), f"audio.{role}_path")
            expected = audio.get(f"{role}_sha256")
            if hash_file(file_path) != expected:
                raise ValueError(f"audio.{role}_sha256 mismatch")
        except ValueError as exc:
            errors.append(f"audio: {exc}")
    quality = item.get("asr_quality", {})
    threshold = quality.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0 < threshold <= 0.8:
        errors.append("asr_quality: threshold must be in (0, 0.8]")
    else:
        for role in ("caller", "agent"):
            wer = quality.get(role, {}).get("wer") if isinstance(quality.get(role), dict) else None
            segments = quality.get(role, {}).get("segments") if isinstance(quality.get(role), dict) else None
            if not isinstance(wer, (int, float)) or wer > threshold:
                errors.append(f"asr_quality: {role} WER is missing or above threshold")
            if not isinstance(segments, list) or not segments:
                errors.append(f"asr_quality: {role} Whisper alignment segments are missing")
    encoding = item.get("model_encoding", {})
    required = {"model_revision", "codebook_layout", "delay_config_sha256", "codes_path", "codes_sha256", "target_mask_path", "target_mask_sha256", "text_alignment_path", "text_alignment_sha256"}
    if not isinstance(encoding, dict) or required.difference(encoding):
        return errors + [f"model_encoding missing {sorted(required.difference(encoding if isinstance(encoding, dict) else {}))}"]
    try:
        codes_path = resolve_under(artifact_root, encoding["codes_path"], "model_encoding.codes_path")
        mask_path = resolve_under(artifact_root, encoding["target_mask_path"], "model_encoding.target_mask_path")
        alignment_path = resolve_under(artifact_root, encoding["text_alignment_path"], "model_encoding.text_alignment_path")
        for path, expected, label in ((codes_path, encoding["codes_sha256"], "codes"), (mask_path, encoding["target_mask_sha256"], "target mask"), (alignment_path, encoding["text_alignment_sha256"], "text alignment")):
            if hash_file(path) != expected:
                raise ValueError(f"{label} SHA-256 mismatch")
        codes = load_tensor(codes_path, "codes")
        mask = load_tensor(mask_path, "target_mask")
        if codes.ndim != 2 or mask.shape != codes.shape:
            raise ValueError("codes and target mask must have identical [K, T] shape")
        if str(mask.dtype) != "torch.bool":
            raise ValueError("target mask must have bool dtype")
        text_indices = explicit_indices(encoding["codebook_layout"], "text", codes.shape[0])
        agent_indices = explicit_indices(encoding["codebook_layout"], "agent_audio", codes.shape[0])
        caller_indices = explicit_indices(encoding["codebook_layout"], "caller_audio", codes.shape[0])
        if text_indices & agent_indices or text_indices & caller_indices or agent_indices & caller_indices:
            raise ValueError("codebook layout streams overlap")
        if text_indices | agent_indices | caller_indices != set(range(codes.shape[0])):
            raise ValueError("codebook layout must explicitly partition every codebook")
        if mask[list(caller_indices)].any().item():
            raise ValueError("caller audio contains supervised target bits")
        if not mask[list(text_indices)].any().item() or not mask[list(agent_indices)].any().item():
            raise ValueError("agent text and agent audio both require supervised target bits")
        alignment = json.loads(alignment_path.read_text())
        if alignment.get("verified") is not True:
            raise ValueError("text alignment has not been verified")
        if alignment.get("model_revision") != encoding["model_revision"]:
            raise ValueError("text alignment model revision mismatch")
        if alignment.get("codes_sha256") != encoding["codes_sha256"]:
            raise ValueError("text alignment codes hash mismatch")
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        errors.append(f"encoding: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-audio-root", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.manifest)
    failures = []
    revisions = set()
    for item in rows:
        revisions.add(item.get("model_encoding", {}).get("model_revision"))
        errors = certify_item(item, args.artifact_root, args.source_audio_root)
        if errors:
            failures.append({"example_id": item.get("example_id"), "errors": errors})
    report = {
        "schema_version": 1,
        "kind": "personaplex-corpus-certificate",
        "tool_version": TOOL_VERSION,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hash_file(args.manifest),
        "artifact_root": str(args.artifact_root.resolve()),
        "model_revisions": sorted(str(value) for value in revisions if value),
        "items": len(rows),
        "failed_items": len(failures),
        "status": "certified_for_adapter_training" if not failures else "failed",
        "failures": failures,
        "limits": "This certificate validates source and encoded training artifacts only. It does not certify a checkpoint or runtime deployment."
    }
    args.certificate.parent.mkdir(parents=True, exist_ok=True)
    args.certificate.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "items": report["items"], "failed_items": report["failed_items"]}))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
