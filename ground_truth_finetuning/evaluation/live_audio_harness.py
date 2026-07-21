"""Real-time Opus/Twilio-impairment trial runner for controlled PersonaPlex."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class MediaImpairment:
    jitter_ms: float = 0.0
    packet_loss_probability: float = 0.0
    twilio_mulaw_roundtrip: bool = True
    barge_in_after_boundary_ms: int | None = None


def _mulaw_roundtrip(pcm: np.ndarray, mu: float = 255.0) -> np.ndarray:
    clipped = np.clip(pcm.astype(np.float32), -1.0, 1.0)
    encoded = np.sign(clipped) * np.log1p(mu * np.abs(clipped)) / math.log1p(mu)
    quantized = np.round((encoded + 1.0) * 127.5).astype(np.uint8)
    restored = quantized.astype(np.float32) / 127.5 - 1.0
    return np.sign(restored) * np.expm1(np.abs(restored) * math.log1p(mu)) / mu


async def run_live_trial(spec: Mapping[str, Any], output_wav: Path) -> dict[str, Any]:
    try:
        import soundfile
        import sphn
        import torch
        import torchaudio.functional
        import websockets
    except ImportError as exc:
        raise RuntimeError("live trials require soundfile, sphn, torch, torchaudio, and websockets") from exc
    source, source_rate = soundfile.read(str(Path(spec["caller_wav"]).resolve()), dtype="float32")
    if source.ndim > 1:
        source = source.mean(axis=1)
    model_rate = int(spec.get("sample_rate", 24000))
    source_tensor = torch.from_numpy(source)
    if source_rate != model_rate:
        source_tensor = torchaudio.functional.resample(source_tensor, source_rate, model_rate)
    source = source_tensor.numpy()
    impairment_value = spec.get("impairment", {})
    impairment = MediaImpairment(**impairment_value)
    if impairment.twilio_mulaw_roundtrip:
        eight_k = torchaudio.functional.resample(torch.from_numpy(source), model_rate, 8000).numpy()
        source = torchaudio.functional.resample(
            torch.from_numpy(_mulaw_roundtrip(eight_k)), 8000, model_rate
        ).numpy()
    writer = sphn.OpusStreamWriter(model_rate)
    reader = sphn.OpusStreamReader(model_rate)
    emitted_text: list[str] = []
    acknowledgements: list[dict[str, Any]] = []
    output_chunks: list[np.ndarray] = []
    media_after_cancel = 0
    cancelled_at: float | None = None
    random_source = random.Random(int(spec.get("seed", 0)))
    response_window_ms = int(spec.get("response_window_ms", 8000))
    chunk_samples = max(1, model_rate // 50)
    async with websockets.connect(str(spec["websocket_url"]), max_size=8_000_000) as socket:
        while True:
            message = await asyncio.wait_for(socket.recv(), timeout=20)
            if isinstance(message, bytes) and message == bytes([0]):
                break

        async def receive() -> None:
            nonlocal media_after_cancel
            while True:
                try:
                    raw = await socket.recv()
                except Exception:
                    return
                if not isinstance(raw, bytes) or not raw:
                    continue
                kind, payload = raw[0], raw[1:]
                if kind == 0x01:
                    if cancelled_at is not None:
                        media_after_cancel += len(payload)
                    reader.append_bytes(payload)
                    pcm = reader.read_pcm()
                    if pcm.shape[-1]:
                        output_chunks.append(np.asarray(pcm, dtype=np.float32).reshape(-1))
                elif kind == 0x02:
                    emitted_text.append(payload.decode("utf-8", errors="strict"))
                elif kind == 0x05:
                    acknowledgements.append(json.loads(payload.decode("utf-8")))

        receiver = asyncio.create_task(receive())
        for offset in range(0, source.shape[0], chunk_samples):
            chunk = source[offset : offset + chunk_samples]
            if random_source.random() >= impairment.packet_loss_probability:
                writer.append_pcm(chunk)
                encoded = writer.read_bytes()
                if encoded:
                    await socket.send(bytes([0x01]) + encoded)
            delay = 0.02 + random_source.uniform(
                -impairment.jitter_ms / 1000.0, impairment.jitter_ms / 1000.0
            )
            await asyncio.sleep(max(0.0, delay))
        control_update = dict(spec["control_update"])
        control_ttl_ms = int(spec["control_ttl_ms"])
        if control_ttl_ms < 1:
            raise ValueError("control_ttl_ms must be positive")
        control_update["expiresAtUnixMs"] = int(time.time() * 1000) + control_ttl_ms
        await socket.send(
            bytes([0x04])
            + json.dumps(control_update, separators=(",", ":")).encode("utf-8")
        )
        await socket.send(
            bytes([0x04])
            + json.dumps(spec["boundary"], separators=(",", ":")).encode("utf-8")
        )
        if impairment.barge_in_after_boundary_ms is not None:
            await asyncio.sleep(impairment.barge_in_after_boundary_ms / 1000.0)
            cancelled_at = time.monotonic()
            await socket.send(
                bytes([0x04])
                + json.dumps(
                    {"type": "control.barge_in", "protocolVersion": 2},
                    separators=(",", ":"),
                ).encode("utf-8")
            )
        elapsed_ms = (
            impairment.barge_in_after_boundary_ms
            if impairment.barge_in_after_boundary_ms is not None
            else 0
        )
        await asyncio.sleep(max(0, response_window_ms - elapsed_ms) / 1000.0)
        receiver.cancel()
        await asyncio.gather(receiver, return_exceptions=True)
    output = np.concatenate(output_chunks) if output_chunks else np.zeros(0, dtype=np.float32)
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(output_wav), output, model_rate)
    applied = [
        ack for ack in acknowledgements
        if ack.get("type") == "control.ack" and ack.get("status") == "applied"
    ]
    return {
        "trial_id": str(spec["trial_id"]),
        "control_frame": control_update["frame"],
        "pair_id": spec.get("pair_id"),
        "branch_id": spec.get("branch_id"),
        "slices": spec.get("slices", {}),
        "generated_wav": str(output_wav),
        "emitted_model_text": "".join(emitted_text).strip(),
        "runtime_events": {"acknowledgements": acknowledgements},
        "transport_checks": {
            "passed": bool(applied) and media_after_cancel == 0,
            "applied_acknowledgement": bool(applied),
            "media_bytes_after_cancel": media_after_cancel,
            "impairment": impairment_value,
        },
        "stale_emissions": int(media_after_cancel > 0),
        "unsupported_policy_claims": 0,
        "audio_checks": {
            "admitted": output.size > model_rate // 10 and bool(np.isfinite(output).all()),
            "samples": int(output.size),
            "duration_ms": int(output.size / model_rate * 1000),
            "peak": float(np.max(np.abs(output))) if output.size else 0.0,
        },
    }
