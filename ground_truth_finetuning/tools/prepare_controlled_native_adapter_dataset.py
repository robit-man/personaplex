#!/usr/bin/env python3
"""Prepare certified controlled-duplex exports for native PersonaPlex encoding.

The manifest and control inputs contain no target wording. Target text remains
only in label-side artifacts: cropped word timing and ``control_labels.jsonl``.
The audible crop prevents an interrupted generated suffix from entering text or
audio supervision, while immutable post-render plan identity follows the example.
"""

from __future__ import annotations

import sys
from pathlib import Path
GTFT_TOOL_ROOT = Path(__file__).resolve().parents[2]
if str(GTFT_TOOL_ROOT) not in sys.path:
    sys.path.insert(0, str(GTFT_TOOL_ROOT))

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import sys
import wave
from typing import Any

from ground_truth_finetuning.training.contracts import (
    ContractError,
    assert_evidence_control_alignment,
    canonical_json,
    validate_control_frame_mapping,
    validate_evidence_frame_mapping,
)


SCHEMA = "personaplex.controlled-native-precodec.v1"
SAMPLE_RATE = 24_000


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_text(value: str) -> str:
    return "sha256:" + sha256(value.encode("utf-8")).hexdigest()


def optional_plan_record_id(example: dict[str, Any]) -> str | None:
    provenance = example.get("provenance") or {}
    value = provenance.get("planRecordId")
    if value is None:
        if provenance.get("branchArtifactSchema") == "personaplex.voryn-branch-artifact.v5":
            raise ValueError(f"{example.get('exampleId')}: v5 branch artifact lacks planRecordId")
        return None
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[a-f0-9]{64}", value):
        raise ValueError(f"{example.get('exampleId')}: malformed planRecordId")
    return value


def hardlink_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def normalise(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(value)
    if not rows:
        raise ValueError("controlled export contains no examples")
    return rows


def require_stereo_24k(path: Path) -> None:
    with wave.open(str(path), "rb") as audio:
        if audio.getframerate() != SAMPLE_RATE or audio.getnchannels() != 2 or audio.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 24 kHz signed-16-bit stereo WAV")
        if audio.getnframes() < 1:
            raise ValueError(f"{path.name}: contains no audio")


def word_alignments(example: dict[str, Any]) -> list[list[Any]]:
    target = example.get("target") or {}
    labels = example.get("labels") or {}
    asr = labels.get("asr") or {}
    start_ms = target.get("audibleStartMs")
    end_ms = target.get("audibleEndMs")
    if not isinstance(start_ms, int) or not isinstance(end_ms, int) or not 0 <= start_ms < end_ms:
        raise ValueError(f"{example.get('exampleId')}: invalid audible target bounds")
    audible_seconds = (end_ms - start_ms) / 1000.0
    alignments: list[list[Any]] = []
    for segment in asr.get("segments", []):
        if not isinstance(segment, dict):
            continue
        for word in segment.get("words", []):
            if not isinstance(word, dict):
                continue
            text = word.get("word")
            word_start = word.get("start")
            word_end = word.get("end")
            if not isinstance(text, str) or not text.strip() or not isinstance(word_start, (int, float)) or not isinstance(word_end, (int, float)):
                raise ValueError(f"{example.get('exampleId')}: malformed Whisper word alignment")
            if not 0 <= float(word_start) < float(word_end):
                raise ValueError(f"{example.get('exampleId')}: invalid Whisper word timing")
            if float(word_end) <= audible_seconds + 0.001:
                alignments.append(
                    [
                        text.strip(),
                        [round(start_ms / 1000.0 + float(word_start), 6), round(start_ms / 1000.0 + float(word_end), 6)],
                        "SPEAKER_MAIN",
                    ]
                )
    if not alignments:
        raise ValueError(f"{example.get('exampleId')}: no fully audible target words remain after crop")
    return alignments


def split_for_pair(pair: dict[str, Any], counterfactual: dict[str, Any] | None = None) -> str:
    group_id = counterfactual.get("groupId") if isinstance(counterfactual, dict) else None
    if isinstance(group_id, str) and group_id:
        bucket = int(sha256(group_id.encode("utf-8")).hexdigest()[:8], 16) % 100
        return "train" if bucket < 80 else "validation" if bucket < 90 else "test"
    caller = pair.get("caller") if isinstance(pair, dict) else None
    agent = pair.get("agent") if isinstance(pair, dict) else None
    caller_id = caller.get("id") if isinstance(caller, dict) else None
    agent_id = agent.get("id") if isinstance(agent, dict) else None
    if not isinstance(caller_id, str) or not isinstance(agent_id, str) or not caller_id or not agent_id:
        raise ValueError("directed approved caller/agent reference pair is required for split assignment")
    bucket = int(sha256(f"{caller_id}|{agent_id}".encode("utf-8")).hexdigest()[:8], 16) % 10
    return "validation" if bucket == 0 else "test" if bucket == 1 else "train"


def prepare(example: dict[str, Any], export_root: Path, output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if example.get("schema") != "personaplex.controlled-duplex.example.v1":
        raise ValueError("unsupported controlled export schema")
    frame_value = example.get("controlFrame")
    if not isinstance(frame_value, dict):
        raise ValueError(f"{example.get('exampleId')}: missing control frame")
    frame = validate_control_frame_mapping(frame_value)
    if frame.plan.mode != "expressive":
        raise ValueError(f"{example.get('exampleId')}: strict controls require the deterministic renderer route")
    evidence_value = example.get("evidenceFrame")
    evidence = None
    if evidence_value is not None:
        if not isinstance(evidence_value, dict):
            raise ValueError(f"{example.get('exampleId')}: evidence frame must be an object")
        evidence = validate_evidence_frame_mapping(evidence_value)
        assert_evidence_control_alignment(frame, evidence)
    label = str((example.get("labels") or {}).get("agentText") or "")
    serialised_frame = canonical_json(frame.as_wire_dict())
    if len(normalise(label)) >= 16 and normalise(label) in normalise(serialised_frame):
        raise ValueError(f"{example.get('exampleId')}: target label leaked into control frame")
    duplex = example.get("duplexAudio") or {}
    relative_audio = duplex.get("path")
    if not isinstance(relative_audio, str) or not relative_audio:
        raise ValueError(f"{example.get('exampleId')}: duplex path missing")
    source_audio = (export_root / relative_audio).resolve()
    if export_root.resolve() not in source_audio.parents or not source_audio.is_file():
        raise ValueError(f"{example.get('exampleId')}: duplex audio escapes export root")
    if sha256_file(source_audio) != duplex.get("sha256"):
        raise ValueError(f"{example.get('exampleId')}: duplex SHA-256 mismatch")
    require_stereo_24k(source_audio)
    quality = example.get("quality") or {}
    if quality.get("accepted") is not True:
        raise ValueError(f"{example.get('exampleId')}: source quality was not accepted")
    alignments = word_alignments(example)
    pair = (example.get("provenance") or {}).get("voicePair")
    split = split_for_pair(pair, example.get("counterfactual"))
    plan_record_id = optional_plan_record_id(example)
    stable_material = f"{example.get('exampleId')}|{example.get('controlFrameHash')}"
    if plan_record_id is not None:
        stable_material += f"|{plan_record_id}"
    stable_id = "sha256:" + sha256(
        stable_material.encode("utf-8")
    ).hexdigest()
    stem = stable_id.removeprefix("sha256:")
    audio_rel = Path("audio") / f"{stem}.wav"
    destination = output_root / audio_rel
    hardlink_or_copy(source_audio, destination)
    target = example["target"]
    info = {
        "schema_version": 2,
        "alignments": alignments,
        "ground_truth": {
            "source_example_id": example["exampleId"],
            "agent_channel": "left",
            "caller_channel": "right",
            "agent_start_seconds": round(target["audibleStartMs"] / 1000.0, 6),
            "agent_end_seconds": round(target["audibleEndMs"] / 1000.0, 6),
            "agent_rendered_end_seconds": round(target["renderedEndMs"] / 1000.0, 6),
            "target_label_sha256": sha256_text(label),
            "control_frame_hash": example.get("controlFrameHash"),
            "plan_record_id": plan_record_id,
        },
    }
    info_rel = Path("audio") / f"{stem}.json"
    (output_root / info_rel).write_text(json.dumps(info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_row = {
        "schema": SCHEMA,
        "example_id": stable_id,
        "split": split,
        "duplex": {
            "path": str(audio_rel),
            "sha256": sha256_file(destination),
            "sample_rate": SAMPLE_RATE,
            "channels": {"agent": 0, "caller": 1},
        },
        "target": {
            "start_ms": target["audibleStartMs"],
            "end_ms": target["audibleEndMs"],
            "rendered_end_ms": target["renderedEndMs"],
        },
        "control": {"frame": frame.as_wire_dict(), "frame_hash": example.get("controlFrameHash")},
        "evidence": {
            "frame": evidence.as_wire_dict(),
            "frame_hash": example.get("evidenceFrameHash"),
        } if evidence is not None else None,
        "counterfactual": example.get("counterfactual"),
        "quality": quality,
        "labels": {"agent_text_sha256": sha256_text(label), "alignment_path": str(info_rel)},
        "provenance": {
            "source_export_example_id": example["exampleId"],
            "source_export_root": str(export_root),
            "voice_pair": pair,
            "scenario_key": (example.get("provenance") or {}).get("scenarioKey"),
            **({"plan_record_id": plan_record_id} if plan_record_id is not None else {}),
        },
    }
    control_label = {
        "example_id": stable_id,
        "control_frame": frame.as_wire_dict(),
        "control_frame_hash": example.get("controlFrameHash"),
        "evidence_frame": evidence.as_wire_dict() if evidence is not None else None,
        "evidence_frame_hash": example.get("evidenceFrameHash") if evidence is not None else None,
        "plan_record_id": plan_record_id,
        "target_transcript": label,
        "target_label_sha256": sha256_text(label),
    }
    return manifest_row, control_label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    export_root = args.export_root.resolve()
    output_root = args.output_root.resolve()
    source = export_root / "examples.jsonl"
    if not source.is_file():
        raise SystemExit("export root must contain strict examples.jsonl")
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"refusing non-empty output root: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "audio").mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(source)
    try:
        prepared = [prepare(row, export_root, output_root) for row in rows]
    except (ContractError, OSError, ValueError, wave.Error) as error:
        raise SystemExit(f"controlled pre-codec preparation refused: {error}") from error
    manifest_rows = [item[0] for item in prepared]
    labels = [item[1] for item in prepared]
    manifest_path = output_root / "precodec_manifest.jsonl"
    labels_path = output_root / "control_labels.jsonl"
    manifest_path.write_text("".join(canonical_json(row) + "\n" for row in manifest_rows), encoding="utf-8")
    labels_path.write_text("".join(canonical_json(row) + "\n" for row in labels), encoding="utf-8")
    report = {
        "schema": "personaplex.controlled-native-precodec-report.v1",
        "items": len(manifest_rows),
        "splits": {name: sum(row["split"] == name for row in manifest_rows) for name in ("train", "validation", "test")},
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "caller_stream_supervision": "forbidden",
        "target_word_alignment": "cropped_to_audible_agent_window",
    }
    (output_root / "precodec_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
