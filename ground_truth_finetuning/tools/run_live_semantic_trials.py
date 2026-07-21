#!/usr/bin/env python3
"""Run paced live PersonaPlex trials, then transcribe every waveform on CUDA."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.evaluation.live_audio_harness import run_live_trial


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--whisper-model", required=True)
    parser.add_argument("--asr-device-index", type=int, default=0)
    args = parser.parse_args()
    try:
        import torch
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(f"CUDA Whisper dependencies are required: {exc}") from exc
    if not torch.cuda.is_available():
        raise SystemExit("live semantic trial ASR is CUDA-only; CPU fallback is prohibited")
    specs = [
        json.loads(line)
        for line in args.specs.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    results = []
    for spec in specs:
        output_wav = args.output_root / "audio" / f"{spec['trial_id']}.wav"
        try:
            result = asyncio.run(run_live_trial(spec, output_wav))
        except Exception as exc:
            result = {
                "trial_id": str(spec.get("trial_id", "unknown")),
                "control_frame": spec.get("control_update", {}).get("frame", {}),
                "pair_id": spec.get("pair_id"),
                "branch_id": spec.get("branch_id"),
                "slices": spec.get("slices", {}),
                "asr_transcript": "",
                "audio_checks": {"admitted": False},
                "transport_checks": {"passed": False, "error": str(exc)},
                "stale_emissions": 0,
                "unsupported_policy_claims": 0,
            }
        results.append(result)
    asr = WhisperModel(
        args.whisper_model,
        device="cuda",
        device_index=args.asr_device_index,
        compute_type="float16",
    )
    for result in results:
        wav = result.get("generated_wav")
        if not wav or result.get("audio_checks", {}).get("admitted") is not True:
            result["asr_transcript"] = ""
            continue
        segments, info = asr.transcribe(wav, beam_size=5, word_timestamps=True)
        segment_values = list(segments)
        result["asr_transcript"] = " ".join(segment.text.strip() for segment in segment_values).strip()
        result["asr"] = {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": [
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": [
                        {"start": word.start, "end": word.end, "word": word.word}
                        for word in (segment.words or [])
                    ],
                }
                for segment in segment_values
            ],
        }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "".join(json.dumps(result, sort_keys=True) + "\n" for result in results),
        encoding="utf-8",
    )
    admitted = sum(result.get("audio_checks", {}).get("admitted") is True for result in results)
    print(json.dumps({"trials": len(results), "audio_admitted": admitted}, sort_keys=True))
    return 0 if admitted == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
