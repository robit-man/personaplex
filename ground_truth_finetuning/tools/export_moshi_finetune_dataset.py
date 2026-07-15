"""Export certified Voryn examples as a native Moshi-Finetune stereo corpus.

The upstream trainer expects 24 kHz stereo WAV: agent output on the left channel,
caller input on the right, plus a sibling JSON file containing timestamped words
for the main (agent) speaker. This exporter preserves measured turn gaps and
refuses heuristic text alignment, missing transcripts, or caller-supervised data.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import wave
from typing import Any

import numpy as np

from ground_truth_finetuning.training.contracts import ContractError, StreamLayout, validate_plan_mapping


TOOL_VERSION = "gtft-moshi-finetune-exporter-v1"
UPSTREAM_REPOSITORY = "https://github.com/kyutai-labs/moshi-finetune"
UPSTREAM_REVISION = "2acc879fe7c48f885a18f6cc9548bccb2674d87b"
SAMPLE_RATE = 24_000


def sha256_file(path: Path) -> str:
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
            raise ValueError(f"line {line_number}: expected an object")
        rows.append(row)
    if not rows:
        raise ValueError("source manifest contains no examples")
    return rows


def resolve_audio(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty audio path")
    candidate = Path(value)
    candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    approved_root = root.resolve()
    if candidate != approved_root and approved_root not in candidate.parents:
        raise ValueError(f"{label} escapes source-audio-root")
    if not candidate.is_file():
        raise ValueError(f"{label} does not exist")
    return candidate


def decode_mono_24k(path: Path) -> np.ndarray:
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    completed = subprocess.run(command, check=True, capture_output=True)
    samples = np.frombuffer(completed.stdout, dtype="<f4").copy()
    if samples.size < 1:
        raise ValueError(f"{path.name} decoded to no audio")
    if not np.isfinite(samples).all():
        raise ValueError(f"{path.name} decoded to non-finite audio")
    return samples


def write_stereo_wav(path: Path, agent: np.ndarray, caller: np.ndarray) -> None:
    if agent.shape != caller.shape:
        raise ValueError("agent and caller timelines must have equal sample lengths")
    stereo = np.column_stack((agent, caller))
    pcm = np.clip(stereo, -1.0, 1.0)
    pcm = (pcm * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def normalise_text(value: str) -> str:
    return re.sub(r"[^\\w']+", "", value.casefold(), flags=re.UNICODE)


def extract_word_alignment(asr: Any, role: str) -> tuple[str, list[tuple[str, float, float]]]:
    if not isinstance(asr, dict):
        raise ValueError(f"{role} ASR metadata is missing")
    transcript = asr.get("transcript")
    if not isinstance(transcript, str) or not transcript.strip():
        raise ValueError(f"{role} Whisper transcript is missing")
    segments = asr.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"{role} Whisper word alignment segments are missing")
    words = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        segment_words = segment.get("words")
        if not isinstance(segment_words, list):
            continue
        for item in segment_words:
            if not isinstance(item, dict):
                continue
            word = item.get("word", item.get("text"))
            start = item.get("start")
            end = item.get("end")
            if not isinstance(word, str) or not word.strip() or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                raise ValueError(f"{role} has an invalid Whisper word entry")
            if not 0 <= float(start) < float(end):
                raise ValueError(f"{role} Whisper word timestamp is invalid")
            words.append((word.strip(), float(start), float(end)))
    if not words:
        raise ValueError(f"{role} requires word-level Whisper timestamps; segment-only timestamps are not sufficient")
    if any(words[index][1] < words[index - 1][1] for index in range(1, len(words))):
        raise ValueError(f"{role} Whisper words are not time ordered")
    if normalise_text(" ".join(word[0] for word in words)) != normalise_text(transcript):
        raise ValueError(f"{role} Whisper word sequence does not reproduce its transcript")
    return transcript, words


def measured_gap_ms(row: dict[str, Any]) -> int:
    timing = row.get("timing")
    value = timing.get("inter_turn_gap_ms") if isinstance(timing, dict) else None
    if not isinstance(value, (int, float)) or value < 0 or value > 30_000:
        raise ValueError("measured caller-to-agent inter_turn_gap_ms is required")
    return int(round(float(value)))


def build_example(row: dict[str, Any], source_audio_root: Path, output_audio_dir: Path) -> dict[str, Any]:
    example_id = row.get("example_id")
    if not isinstance(example_id, str) or not example_id.startswith("sha256:"):
        raise ValueError("example_id must be a stable sha256 identifier")
    plan = validate_plan_mapping(row.get("semantics", {}).get("plan", {}))
    if plan.mode != "expressive":
        raise ValueError("strict plans cannot enter the expressive PersonaPlex trainer")
    canonical = row.get("semantics", {}).get("canonical_response")
    if not isinstance(canonical, str) or not canonical.strip():
        raise ValueError("canonical_response is required")
    audio = row.get("audio", {})
    if not isinstance(audio, dict):
        raise ValueError("audio metadata is missing")
    caller_path = resolve_audio(source_audio_root, audio.get("caller_path"), "audio.caller_path")
    agent_path = resolve_audio(source_audio_root, audio.get("agent_path"), "audio.agent_path")
    if sha256_file(caller_path) != audio.get("caller_sha256"):
        raise ValueError("caller audio SHA-256 mismatch")
    if sha256_file(agent_path) != audio.get("agent_sha256"):
        raise ValueError("agent audio SHA-256 mismatch")
    asr = row.get("asr_quality", {})
    caller_transcript, caller_words = extract_word_alignment(asr.get("caller"), "caller")
    agent_transcript, agent_words = extract_word_alignment(asr.get("agent"), "agent")
    if normalise_text(agent_transcript) != normalise_text(canonical):
        raise ValueError("agent Whisper transcript does not exactly match the canonical response")
    gap_samples = round(measured_gap_ms(row) * SAMPLE_RATE / 1000)
    caller_audio = decode_mono_24k(caller_path)
    agent_audio = decode_mono_24k(agent_path)
    total_samples = caller_audio.size + gap_samples + agent_audio.size
    agent_timeline = np.zeros(total_samples, dtype=np.float32)
    caller_timeline = np.zeros(total_samples, dtype=np.float32)
    caller_timeline[: caller_audio.size] = caller_audio
    agent_start = caller_audio.size + gap_samples
    agent_timeline[agent_start : agent_start + agent_audio.size] = agent_audio
    stem = example_id.removeprefix("sha256:")
    wav_path = output_audio_dir / f"{stem}.wav"
    info_path = output_audio_dir / f"{stem}.json"
    write_stereo_wav(wav_path, agent_timeline, caller_timeline)
    agent_start_seconds = agent_start / SAMPLE_RATE
    alignments = [
        [word, [round(agent_start_seconds + start, 6), round(agent_start_seconds + end, 6)], "SPEAKER_MAIN"]
        for word, start, end in agent_words
    ]
    info = {
        "schema_version": 1,
        "alignments": alignments,
        "ground_truth": {
            "example_id": example_id,
            "plan_sha256": row.get("semantics", {}).get("plan_sha256"),
            "canonical_response_sha256": row.get("semantics", {}).get("canonical_response_sha256"),
            "agent_channel": "left",
            "caller_channel": "right",
            "agent_start_seconds": round(agent_start_seconds, 6),
            "caller_words": len(caller_words),
            "agent_words": len(agent_words),
            "caller_transcript_sha256": "sha256:" + sha256(caller_transcript.encode()).hexdigest(),
            "agent_transcript_sha256": "sha256:" + sha256(agent_transcript.encode()).hexdigest(),
        },
    }
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return {
        "example_id": example_id,
        "split": row.get("split"),
        "path": str(wav_path.resolve()),
        "duration": round(total_samples / SAMPLE_RATE, 6),
        "audio_sha256": sha256_file(wav_path),
        "alignment_sha256": sha256_file(info_path),
        "plan": plan.as_wire_dict(),
        "canonical_response_sha256": row.get("semantics", {}).get("canonical_response_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-audio-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    source_audio_root = args.source_audio_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to write into non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    audio_dir = output_root / "audio"
    audio_dir.mkdir(exist_ok=True)
    rows = read_jsonl(manifest)
    try:
        layout = StreamLayout.from_mapping(
            {
                "text_stream_indices": [0],
                "agent_audio_stream_indices": list(range(1, 9)),
                "caller_audio_stream_indices": list(range(9, 17)),
            }
        )
        exported = [build_example(row, source_audio_root, audio_dir) for row in rows]
    except (ContractError, OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise SystemExit(f"export refused: {exc}") from exc
    splits = {"train": [], "validation": [], "test": []}
    for record in exported:
        split = record["split"]
        if split not in splits:
            raise SystemExit(f"export refused: unsupported split {split!r}")
        splits[split].append({"path": record["path"], "duration": record["duration"]})
    for split, entries in splits.items():
        (output_root / f"{split}.jsonl").write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    (output_root / "stream_layout.json").write_text(json.dumps(layout.as_dict(), indent=2, sort_keys=True) + "\n")
    (output_root / "control_labels.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "example_id": record["example_id"],
                    "plan": record["plan"],
                    "canonical_response_sha256": record["canonical_response_sha256"],
                },
                sort_keys=True,
            )
            + "\n"
            for record in exported
        )
    )
    certificate = {
        "schema_version": 1,
        "kind": "personaplex-upstream-lora-dataset-certificate",
        "tool_version": TOOL_VERSION,
        "status": "certified_for_upstream_agent_only_lora",
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "upstream": {"repository": UPSTREAM_REPOSITORY, "revision": UPSTREAM_REVISION},
        "stream_layout": layout.as_dict(),
        "channel_order": {"left": "agent", "right": "caller"},
        "caller_stream_supervision": "forbidden_by_overlay",
        "items": len(exported),
        "splits": {split: len(entries) for split, entries in splits.items()},
        "records": [{key: value for key, value in record.items() if key != "plan"} for record in exported],
    }
    certificate_path = output_root / "certificate.json"
    certificate_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": certificate["status"], "items": certificate["items"], "certificate": str(certificate_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
