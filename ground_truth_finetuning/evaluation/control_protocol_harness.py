"""Direct acknowledgement test for the semantic-control websocket protocol."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any


class ProtocolFailure(RuntimeError):
    pass


def _decode_message(message: str | bytes, bridge_wire_v1: bool) -> dict[str, Any] | None:
    if isinstance(message, str):
        return json.loads(message)
    if bridge_wire_v1:
        if not message or message[0] != 0x05:
            return None
        message = message[1:]
    return json.loads(message.decode("utf-8"))


def _encode_message(payload: dict[str, Any], bridge_wire_v1: bool) -> str | bytes:
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return bytes([0x04]) + encoded if bridge_wire_v1 else encoded.decode("utf-8")


async def exercise_control_protocol(
    url: str,
    update: dict[str, Any],
    boundary: dict[str, Any],
    *,
    timeout_s: float,
    bridge_wire_v1: bool,
) -> dict[str, Any]:
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Install websockets to run the control protocol harness") from exc
    expected = (update["callId"], update["revision"], update["contextHash"])
    deadline = time.monotonic() + timeout_s
    async with websockets.connect(url, max_size=1_000_000) as socket:
        await socket.send(_encode_message(update, bridge_wire_v1))
        await socket.send(_encode_message(boundary, bridge_wire_v1))
        terminal: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            try:
                raw = await asyncio.wait_for(socket.recv(), timeout=remaining)
            except TimeoutError:
                break
            message = _decode_message(raw, bridge_wire_v1)
            if not message or message.get("type") != "control.ack":
                continue
            received = (message.get("callId"), message.get("revision"), message.get("contextHash"))
            if received != expected:
                continue
            terminal = message
            if message.get("status") == "applied":
                return message
            if message.get("status") in {"rejected", "expired", "context_mismatch", "prefix_build_failed", "safe_fallback"}:
                raise ProtocolFailure(f"control terminal failure: {json.dumps(message, sort_keys=True)}")
        raise ProtocolFailure(f"missing applied acknowledgement for call/revision/context={expected}; last={terminal}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--update", type=Path, required=True, help="control.update JSON file")
    parser.add_argument("--boundary", type=Path, required=True, help="control.boundary JSON file")
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--bridge-wire-v1", action="store_true", help="prefix outgoing 0x04 and decode incoming 0x05")
    args = parser.parse_args()
    try:
        result = asyncio.run(
            exercise_control_protocol(
                args.url,
                json.loads(args.update.read_text()),
                json.loads(args.boundary.read_text()),
                timeout_s=args.timeout_s,
                bridge_wire_v1=args.bridge_wire_v1,
            )
        )
    except (ProtocolFailure, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1
    print(json.dumps({"ok": True, "ack": result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
