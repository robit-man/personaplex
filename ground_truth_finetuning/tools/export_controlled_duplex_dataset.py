#!/usr/bin/env python3
"""Export certified Voryn V3/V4 conversations as native duplex PersonaPlex examples.

This exporter deliberately keeps the silent control frame separate from the agent
text/audio label.  A conversation is training-admitted only when it has valid
provenance/audio/ASR data and, when it claims barge-in coverage, contains a real
caller overlap followed by a recovery agent turn.  ``--allow-incomplete`` is a
diagnostic-only mode: it materializes audio for inspection but never writes those
examples to ``examples.jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "personaplex.controlled-duplex.example.v1"


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def run(command: list[str]) -> None:
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr[-2000:]}"
        )


def audio_probe(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=channels,sample_rate,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"no audio stream: {path}")
    return streams[0]


def jsonl_inputs(paths: Iterable[Path]) -> Iterable[Path]:
    for path in paths:
        if path.is_file() and path.suffix == ".jsonl":
            yield path
        elif path.is_dir():
            yield from sorted(path.rglob("*.jsonl"))


def read_conversations(inputs: Iterable[Path]) -> dict[str, list[dict[str, Any]]]:
    conversations: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in jsonl_inputs(inputs):
        with source.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("schema") not in {"voxrn.synthetic-conversation.v3", "voxrn.synthetic-conversation.v4"}:
                    continue
                if not record.get("conversationId"):
                    raise ValueError(f"{source}:{number}: missing conversationId")
                record["_source_path"] = source
                conversations[record["conversationId"]].append(record)
    return conversations


def find_timeline(records: list[dict[str, Any]]) -> tuple[Path | None, dict[str, Any] | None]:
    source = records[0]["_source_path"]
    timeline_ref = records[0].get("duplexTimelinePath")
    if not timeline_ref:
        return None, None
    path = source.parent / timeline_ref
    if not path.is_file():
        return path, None
    return path, json.loads(path.read_text(encoding="utf-8"))


def target_control_issues(record: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    control = record.get("control") or {}
    frame = control.get("frame") or {}
    plan = control.get("plan") or {}
    quality = record.get("quality") or {}
    if record.get("speaker") != "target":
        return issues
    replay = record.get("replay") or {}
    if replay.get("role") == "shared_prefix_context_only":
        training = record.get("training") or {}
        if training.get("eligible") or "shared_prefix_replay_context_only" not in training.get("exclusionReasons", []):
            issues.append("shared_prefix_replay_not_quarantined")
        # The shared prefix is duplex conditioning context. Generic audio,
        # timeline, and provenance checks still run in conversation_issues(),
        # but it cannot be required to be a branch-local target label.
        return issues
    if not record.get("training", {}).get("eligible"):
        issues.append("target_not_marked_training_eligible")
    if record.get("semanticAdherence", {}).get("verificationStatus") != "batch_certified":
        issues.append("target_semantic_certificate_missing")
    if not quality.get("accepted"):
        issues.append("target_quality_not_accepted")
    if not frame or not plan:
        issues.append("missing_control_frame_or_plan")
        return issues
    if frame.get("stateHash") != plan.get("contextHash"):
        issues.append("control_plan_context_hash_mismatch")
    if not frame.get("frameId") or not frame.get("stateRevision"):
        issues.append("control_frame_identity_missing")
    evidence = control.get("evidence")
    if record.get("schema") == "voxrn.synthetic-conversation.v4" and evidence is not None:
        if not isinstance(evidence, dict) or (
            evidence.get("conversationId") != record.get("conversationId")
            or evidence.get("counterfactual", {}).get("groupId") != record.get("counterfactualGroupId")
            or evidence.get("counterfactual", {}).get("branchId") != record.get("counterfactualBranchId")
            or evidence.get("plan", {}).get("revision") != plan.get("revision")
        ):
            issues.append("v4_target_evidence_lineage_invalid")
    serialised = canonical({"frame": frame, "evidence": evidence})
    label = normalise(record.get("text", ""))
    if label and len(label) >= 16 and label in normalise(serialised):
        issues.append("target_text_leaked_into_control_frame")
    if "canonicalResponse" in serialised or "canonical_response" in serialised:
        issues.append("canonical_response_leaked_into_control_frame")
    return issues


def interruption_issues(records: list[dict[str, Any]]) -> list[str]:
    """Require real audio overlap and a recovery response when coverage says barge-in."""
    issues: list[str] = []
    requires_barge = max(
        (int(record.get("characteristics", {}).get("interruption", 0)) for record in records),
        default=0,
    ) >= 70
    if not requires_barge:
        return issues
    ordered = sorted(records, key=lambda item: item["turnIndex"])
    observed = False
    recovered = False
    for index, record in enumerate(ordered):
        if record.get("speaker") != "target":
            continue
        timing = record.get("timing") or {}
        audible_end = timing.get("audibleEndedAtMs")
        rendered_end = timing.get("endedAtMs")
        if not isinstance(audible_end, (int, float)) or not isinstance(rendered_end, (int, float)):
            continue
        if audible_end >= rendered_end:
            continue
        following = ordered[index + 1 :]
        caller = next((item for item in following if item.get("speaker") == "caller"), None)
        if not caller:
            continue
        caller_start = (caller.get("timing") or {}).get("startedAtMs")
        if not isinstance(caller_start, (int, float)) or caller_start >= rendered_end:
            continue
        observed = True
        recovery = next((item for item in following if item.get("speaker") == "target"), None)
        if recovery and (recovery.get("control") or {}).get("frame", {}).get("turnTaking", {}).get("recoveryExpected"):
            recovered = True
    if not observed:
        issues.append("claimed_barge_in_without_real_caller_overlap")
    elif not recovered:
        issues.append("barge_in_without_recovery_agent_turn")
    return issues


def conversation_issues(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str], dict[str, Any] | None]:
    issues: list[str] = []
    ordered = sorted(records, key=lambda item: item["turnIndex"])
    seen_turns: set[int] = set()
    for record in ordered:
        turn = record.get("turnIndex")
        if not isinstance(turn, int) or turn in seen_turns:
            issues.append("duplicate_or_invalid_turn_index")
        seen_turns.add(turn)
        quality = record.get("quality") or {}
        if not quality.get("accepted"):
            issues.append(f"turn_{turn}_quality_not_accepted")
        audio_ref = record.get("audioPath")
        audio_path = record["_source_path"].parent / str(audio_ref or "")
        if not audio_ref or not audio_path.is_file():
            issues.append(f"turn_{turn}_audio_missing")
        if record.get("speaker") == "caller" and record.get("authenticity", {}).get("status") != "batch_certified":
            issues.append(f"turn_{turn}_caller_authenticity_certificate_missing")
        if record.get("speaker") == "target":
            issues.extend(f"turn_{turn}_{item}" for item in target_control_issues(record))
    timeline_path, timeline = find_timeline(ordered)
    if not timeline:
        issues.append("duplex_timeline_missing")
    else:
        events = {event.get("turnIndex"): event for event in timeline.get("events", [])}
        if timeline.get("conversationId") != ordered[0].get("conversationId"):
            issues.append("duplex_timeline_conversation_mismatch")
        for record in ordered:
            event = events.get(record["turnIndex"])
            if not event:
                issues.append(f"turn_{record['turnIndex']}_missing_from_timeline")
            elif event.get("audioPath") != record.get("audioPath"):
                issues.append(f"turn_{record['turnIndex']}_timeline_audio_mismatch")
    issues.extend(interruption_issues(ordered))
    return ordered, sorted(set(issues)), timeline


def materialize_duplex(records: list[dict[str, Any]], output: Path, sample_rate: int) -> dict[str, Any]:
    total_ms = max(int((record.get("timing") or {}).get("audibleEndedAtMs", 0)) for record in records)
    if total_ms <= 0:
        raise ValueError("non-positive duplex duration")
    total_seconds = total_ms / 1000
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="personaplex-duplex-") as temp_name:
        temp = Path(temp_name)
        channel_files: dict[str, list[Path]] = {"target": [], "caller": []}
        for record in records:
            timing = record["timing"]
            audible_ms = int(timing.get("audibleSpeechMs", timing.get("speechMs", 0)))
            start_ms = int(timing.get("startedAtMs", 0))
            if audible_ms <= 0:
                raise ValueError(f"turn {record['turnIndex']} has no audible speech")
            source = record["_source_path"].parent / record["audioPath"]
            clip = temp / f"turn-{record['turnIndex']}.wav"
            run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-af",
                    (
                        f"atrim=duration={audible_ms / 1000:.3f},aresample={sample_rate},"
                        f"asetpts=PTS-STARTPTS,adelay={start_ms}:all=1,"
                        f"apad=whole_dur={total_seconds:.3f},atrim=duration={total_seconds:.3f}"
                    ),
                    "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(clip),
                ]
            )
            channel_files[record["speaker"]].append(clip)

        mixes: dict[str, Path] = {}
        for speaker, clips in channel_files.items():
            mix = temp / f"{speaker}.wav"
            if clips:
                run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                        *sum((["-i", str(clip)] for clip in clips), []),
                        "-filter_complex", f"amix=inputs={len(clips)}:normalize=0",
                        "-t", f"{total_seconds:.3f}", "-ac", "1", "-ar", str(sample_rate),
                        "-c:a", "pcm_s16le", str(mix),
                    ]
                )
            else:
                run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", f"anullsrc=r={sample_rate}:cl=mono", "-t", f"{total_seconds:.3f}",
                        "-c:a", "pcm_s16le", str(mix),
                    ]
                )
            mixes[speaker] = mix
        run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", str(mixes["target"]), "-i", str(mixes["caller"]),
                "-filter_complex", "[0:a][1:a]join=inputs=2:channel_layout=stereo[duplex]",
                "-map", "[duplex]", "-t", f"{total_seconds:.3f}", "-ar", str(sample_rate),
                "-ac", "2", "-c:a", "pcm_s16le", str(output),
            ]
        )
    probe = audio_probe(output)
    return {
        "path": str(output),
        "sha256": "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest(),
        "durationMs": total_ms,
        "sampleRate": int(probe["sample_rate"]),
        "channels": int(probe["channels"]),
    }


def target_example(record: dict[str, Any], duplex: dict[str, Any], root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    control = record["control"]
    frame = control["frame"]
    source_audio = record["_source_path"].parent / record["audioPath"]
    timing = record["timing"]
    caller_reference = next((item.get("voiceReference") for item in records if item.get("speaker") == "caller"), None)
    agent_reference = next((item.get("voiceReference") for item in records if item.get("speaker") == "target"), None)
    return {
        "schema": SCHEMA,
        "exampleId": f"{record['conversationId']}:target:{record['turnIndex']}",
        "conversationId": record["conversationId"],
        "targetTurnIndex": record["turnIndex"],
        "duplexAudio": {
            "path": str(Path(duplex["path"]).relative_to(root)),
            "sha256": duplex["sha256"],
            "sampleRate": duplex["sampleRate"],
            "channels": {"agent": 0, "caller": 1},
            "durationMs": duplex["durationMs"],
        },
        "target": {
            "audibleStartMs": int(timing["startedAtMs"]),
            "audibleEndMs": int(timing["audibleEndedAtMs"]),
            "renderedEndMs": int(timing["endedAtMs"]),
            "sourceAudioSha256": record.get("audioSha256"),
            "sourceAudioPath": str(source_audio),
        },
        "controlFrame": frame,
        "controlFrameHash": control.get("frameHash"),
        "evidenceFrame": control.get("evidence"),
        "evidenceFrameHash": control.get("evidenceHash"),
        "counterfactual": {
            "groupId": record.get("counterfactualGroupId"),
            "branchId": record.get("counterfactualBranchId"),
            "pivotTargetOrdinal": record.get("counterfactualPivotTargetOrdinal"),
        } if record.get("schema") == "voxrn.synthetic-conversation.v4" else None,
        "labels": {
            "agentText": record["text"],
            "agentHeardText": record.get("heardText"),
            "asr": record.get("asr"),
        },
        "quality": record.get("quality"),
        "provenance": {
            "datasetId": record["datasetId"],
            "scenarioKey": record.get("scenarioKey"),
            "voiceReference": record.get("voiceReference"),
            "voicePair": {
                "caller": caller_reference,
                "agent": agent_reference,
            },
            "sourceManifest": str(record["_source_path"]),
        },
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Certified Voryn V3/V4 JSONL file(s) or directory roots")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--allow-incomplete", action="store_true", help="materialize rejected conversations for diagnostic playback only")
    args = parser.parse_args()
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg and ffprobe are required")
    if args.sample_rate != 24000:
        raise ValueError("native PersonaPlex training export is fixed at 24000 Hz")

    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    conversations = read_conversations(args.inputs)
    examples: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    admitted_conversations = 0
    for conversation_id, raw_records in sorted(conversations.items()):
        records, issues, _timeline = conversation_issues(raw_records)
        admitted = not issues
        if not admitted:
            rejected.append({"conversationId": conversation_id, "issues": issues})
        if not admitted and not args.allow_incomplete:
            continue
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", conversation_id)
        duplex = materialize_duplex(records, root / "audio" / f"{safe_name}.wav", args.sample_rate)
        target_rows = [
            target_example(record, duplex, root, records)
            for record in records
            if record.get("speaker") == "target"
            and (record.get("schema") != "voxrn.synthetic-conversation.v4" or isinstance((record.get("control") or {}).get("evidence"), dict))
        ]
        if admitted:
            admitted_conversations += 1
            examples.extend(target_rows)
        else:
            diagnostics.extend([{**row, "trainingAdmitted": False, "rejectionIssues": issues} for row in target_rows])

    write_jsonl(root / "examples.jsonl", examples)
    write_jsonl(root / "diagnostic_examples.jsonl", diagnostics)
    write_jsonl(root / "rejections.jsonl", rejected)
    manifest = {
        "schema": "personaplex.controlled-duplex-export.v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "sampleRate": args.sample_rate,
        "sourceConversationCount": len(conversations),
        "admittedConversationCount": admitted_conversations,
        "admittedExampleCount": len(examples),
        "diagnosticExampleCount": len(diagnostics),
        "rejectedConversationCount": len(rejected),
        "strict": not args.allow_incomplete,
        "files": {
            "examples": "examples.jsonl",
            "diagnosticExamples": "diagnostic_examples.jsonl",
            "rejections": "rejections.jsonl",
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"export failed: {error}", file=sys.stderr)
        raise
