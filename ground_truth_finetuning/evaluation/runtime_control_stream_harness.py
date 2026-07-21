"""CUDA proof that a v4 checkpoint queues, applies, and cancels native control rows."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ground_truth_finetuning.training.contracts import validate_control_frame_mapping
from ground_truth_finetuning.training.control_stream import (
    ControlStreamConfig,
    SemanticControlStreamAdapter,
)
from personaplex_control.runtime import (
    RuntimeControlSession,
    RuntimeControlUpdate,
    SemanticControlStreamProvider,
)


def file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--moshi-source-root", type=Path, required=True)
    parser.add_argument("--moshi-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--control-stream-checkpoint", type=Path, required=True)
    parser.add_argument("--frame-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deadline-ms", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("v4 runtime harness is CUDA-only")
    sys.path.insert(0, str(args.moshi_source_root.resolve()))
    import sentencepiece
    from moshi.models.lm import LMGen
    from moshi.models.loaders import get_moshi_lm

    device = torch.device(args.device)
    frame = validate_control_frame_mapping(json.loads(args.frame_json.read_text(encoding="utf-8")))
    lm = get_moshi_lm(args.moshi_path.resolve(), device=device, dtype=torch.bfloat16)
    lm.eval()
    try:
        checkpoint = torch.load(args.control_stream_checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.control_stream_checkpoint, map_location="cpu")
    if checkpoint.get("schema_version") != 4:
        raise SystemExit("checkpoint is not a v4 semantic control stream")
    config = ControlStreamConfig.from_mapping(checkpoint["adapter_config"])
    adapter = SemanticControlStreamAdapter(lm_hidden_size=int(lm.dim), config=config).to(
        device=device, dtype=torch.float32
    )
    adapter.load_state_dict(checkpoint["adapter_state_dict"], strict=True)
    adapter.eval()
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer_path.resolve()))
    generator = LMGen(lm, device=device)
    generator.streaming_forever(1)
    provider = SemanticControlStreamProvider(
        lm_gen=generator,
        adapter=adapter,
        tokenizer=tokenizer,
        adapter_version=file_hash(args.control_stream_checkpoint.resolve()),
        max_control_tokens=config.max_tokens,
    )
    session = RuntimeControlSession(
        call_id=frame.conversation_id,
        prefix_provider=provider,
        prefill_deadline_ms=args.deadline_ms,
    )
    now = int(time.time() * 1000)
    update = RuntimeControlUpdate.from_mapping(
        {
            "type": "control.update",
            "protocolVersion": 2,
            "callId": frame.conversation_id,
            "revision": frame.state_revision,
            "contextHash": frame.state_hash,
            "expiresAtUnixMs": now + frame.plan.expiry_ms,
            "frame": frame.as_wire_dict(),
        }
    )
    queued = session.submit(update, now_unix_ms=now)
    applied = session.apply_boundary(
        call_id=frame.conversation_id,
        turn_id=frame.target_turn_id,
        context_hash=frame.state_hash,
        now_unix_ms=now + 1,
    )
    pending_before_cancel = generator._streaming_state.pending_streaming_sums[0]
    cancelled = session.caller_barge_in(reason="runtime_harness_barge_in")
    pending_after_cancel = generator._streaming_state.pending_streaming_sums[0]
    passed = (
        queued[-1].status == "queued"
        and applied.status == "applied"
        and applied.conditioning_mode == "temporal_stream_v4"
        and pending_before_cancel is not None
        and pending_after_cancel is None
        and not session.may_emit(applied.generation_id)
        and any(ack.status == "superseded" for ack in cancelled)
    )
    result = {
        "schema_version": 4,
        "kind": "personaplex-runtime-control-stream-harness",
        "status": "passed" if passed else "failed",
        "queued": [ack.as_wire_dict() for ack in queued],
        "applied": applied.as_wire_dict(),
        "cancelled": [ack.as_wire_dict() for ack in cancelled],
        "stream_rows_queued": int(pending_before_cancel.shape[0])
        if pending_before_cancel is not None
        else 0,
        "queue_empty_after_cancel": pending_after_cancel is None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "stream_rows": result["stream_rows_queued"]}))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
