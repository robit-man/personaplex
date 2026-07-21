#!/usr/bin/env python3
"""Build leakage-safe paired live trials from certified native causal examples."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.contracts import validate_control_frame_mapping
from ground_truth_finetuning.training.diverse_cascade import assert_no_target_leak


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def member_rows(pair: dict[str, Any], rows: dict[str, dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    values = []
    for key in ("member_a", "member_b"):
        member = pair[key]
        example_id = str(member["example_id"])
        if example_id not in rows:
            raise ValueError(f"pair member is absent from manifest: {example_id}")
        values.append((member, rows[example_id]))
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair-index", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--voice-manifest", type=Path, required=True)
    parser.add_argument("--voice-prompt-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-specs", type=Path, required=True)
    parser.add_argument("--websocket-url", required=True)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--history-seconds", type=float, default=20.0)
    parser.add_argument("--split", action="append", default=["validation", "test"])
    args = parser.parse_args()
    if args.max_pairs < 1 or args.history_seconds <= 0:
        raise SystemExit("max-pairs and history-seconds must be positive")
    try:
        import numpy as np
        import soundfile
    except ImportError as exc:
        raise SystemExit(f"trial-spec audio dependencies are required: {exc}") from exc

    records = load_jsonl(args.manifest.resolve())
    rows = {str(row["example_id"]): row for row in records}
    voice_manifest = json.loads(args.voice_manifest.read_text(encoding="utf-8"))
    voice_records = {str(item["id"]): item for item in voice_manifest["references"]}
    voice_prompt_dir = args.voice_prompt_dir.resolve()
    selected = [
        pair for pair in load_jsonl(args.pair_index.resolve()) if pair.get("split") in set(args.split)
    ][: args.max_pairs]
    if not selected:
        raise SystemExit("no causal pairs matched the requested splits")
    audio_root = args.output_root.resolve() / "caller_audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, Any]] = []
    for pair in selected:
        members = member_rows(pair, rows)
        donor_row = members[0][1]
        duplex = donor_row["duplex"]
        source_path = args.audio_root.resolve() / str(duplex["path"])
        if file_hash(source_path) != duplex["sha256"]:
            raise ValueError(f"duplex hash mismatch for {source_path}")
        source, sample_rate = soundfile.read(str(source_path), dtype="float32", always_2d=True)
        if int(sample_rate) != int(duplex["sample_rate"]):
            raise ValueError(f"sample-rate mismatch for {source_path}")
        caller_channel = int(duplex["channels"]["caller"])
        boundary_ms = int(donor_row["target"]["start_ms"])
        boundary_sample = min(source.shape[0], round(boundary_ms * sample_rate / 1000))
        history_samples = round(args.history_seconds * sample_rate)
        start_sample = max(0, boundary_sample - history_samples)
        caller = np.asarray(source[start_sample:boundary_sample, caller_channel], dtype=np.float32)
        if caller.size < sample_rate // 10 or not np.isfinite(caller).all():
            raise ValueError(f"caller prefix is empty or invalid for pair {pair['pair_id']}")
        pair_token = str(pair["pair_id"]).removeprefix("sha256:")[:20]
        caller_path = audio_root / f"{pair_token}.wav"
        soundfile.write(str(caller_path), caller, sample_rate)
        caller_hash = file_hash(caller_path)
        for member, row in members:
            frame_value = row["control"]["frame"]
            frame = validate_control_frame_mapping(frame_value)
            state = frame_value["state"]
            binding = state.get("semanticBindings") or {}
            ttl_ms = max(int(frame_value["plan"]["expiryMs"]), int(frame_value["update"]["expiresAtMs"]))
            trial_id = f"{pair_token}-{member['branch_id']}"
            voice = row["provenance"]["voice_pair"]["agent"]
            voice_id = str(voice["id"])
            if voice_id not in voice_records:
                raise ValueError(f"agent voice is absent from approved manifest: {voice_id}")
            voice_path = voice_prompt_dir / Path(str(voice["filePath"])).name
            expected_voice_hash = str(voice_records[voice_id]["sha256"])
            if file_hash(voice_path) != expected_voice_hash or expected_voice_hash != voice["sha256"]:
                raise ValueError(f"agent voice provenance mismatch: {voice_id}")
            parsed_url = urlsplit(args.websocket_url)
            query = dict(parse_qsl(parsed_url.query, keep_blank_values=True))
            query.update(
                {
                    "call_id": frame.conversation_id,
                    "voice_prompt": voice_path.name,
                    "seed": str(int(pair_token[:8], 16)),
                }
            )
            websocket_url = urlunsplit(
                (parsed_url.scheme, parsed_url.netloc, parsed_url.path, urlencode(query), parsed_url.fragment)
            )
            update = {
                "type": "control.update",
                "protocolVersion": 2,
                "callId": frame.conversation_id,
                "revision": frame.state_revision,
                "contextHash": frame.state_hash,
                "expiresAtUnixMs": 1,
                "frame": frame.as_wire_dict(),
            }
            spec = {
                "trial_id": trial_id,
                "pair_id": pair["pair_id"],
                "branch_id": member["branch_id"],
                "websocket_url": websocket_url,
                "caller_wav": str(caller_path),
                "caller_audio_sha256": caller_hash,
                "sample_rate": int(sample_rate),
                "control_update": update,
                "control_ttl_ms": ttl_ms,
                "boundary": {
                    "type": "control.boundary",
                    "protocolVersion": 2,
                    "callId": frame.conversation_id,
                    "turnId": frame.target_turn_id,
                    "contextHash": frame.state_hash,
                },
                "response_window_ms": int(frame_value["plan"]["delivery"]["max_duration_ms"]),
                "impairment": {
                    "jitter_ms": 0.0,
                    "packet_loss_probability": 0.0,
                    "twilio_mulaw_roundtrip": True,
                    "barge_in_after_boundary_ms": None,
                },
                "slices": {
                    "counterfactual_axis": str(binding.get("counterfactualAxis", "unspecified")),
                    "control_kind": str(binding.get("controlKind", "unspecified")),
                    "turn_taking": (
                        "barge_in_expected"
                        if frame_value["turnTaking"].get("expectedBargeIn") is True
                        else "completed_turn"
                    ),
                    "policy_sensitive": str(bool(state.get("policyConstraints"))).lower(),
                },
                "source": {
                    "manifest_example_id": row["example_id"],
                    "shared_prefix_sha256": pair["shared_prefix_sha256"],
                    "caller_source_example_id": donor_row["example_id"],
                    "caller_window_start_ms": round(start_sample / sample_rate * 1000),
                    "caller_boundary_ms": boundary_ms,
                    "voice_prompt_id": voice_id,
                    "voice_prompt_sha256": expected_voice_hash,
                },
            }
            assert_no_target_leak(spec)
            specs.append(spec)
    args.output_specs.parent.mkdir(parents=True, exist_ok=True)
    args.output_specs.write_text(
        "".join(json.dumps(spec, sort_keys=True) + "\n" for spec in specs),
        encoding="utf-8",
    )
    print(json.dumps({"pairs": len(selected), "trials": len(specs), "specs": str(args.output_specs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
