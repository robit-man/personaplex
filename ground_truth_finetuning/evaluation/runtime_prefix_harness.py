"""CUDA-native proof harness for the PersonaPlex semantic-prefix application path.

It loads the pinned Moshi checkpoint plus a trained adapter, admits one real
``ControlTrainingFrame``, applies it at a matching boundary, and fails unless
the acknowledgement is ``applied`` after direct transformer prefill.
"""

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

from ground_truth_finetuning.training.contracts import (
    assert_evidence_control_alignment,
    validate_control_frame_mapping,
    validate_evidence_frame_mapping,
)
from ground_truth_finetuning.training.evidence_conditioning import EvidenceStreamAdapter
from ground_truth_finetuning.training.semantic_prefix import SemanticPrefixAdapter
from personaplex_control.runtime import (
    EvidenceStreamProvider,
    RuntimeControlSession,
    RuntimeControlUpdate,
    RuntimeEvidenceUpdate,
    SemanticPrefixProvider,
)


def sha256_file(path: Path) -> str:
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
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--frame-json", type=Path, required=True)
    parser.add_argument("--supporting-frame-json", type=Path, help="prior applied control frame required for evidence mode")
    parser.add_argument("--evidence-frame-json", type=Path, help="typed delayed-evidence frame aligned to --frame-json")
    parser.add_argument("--evidence-adapter-checkpoint", type=Path, help="trained evidence-stream adapter checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix-frames", type=int, default=16)
    parser.add_argument("--evidence-stream-frames", type=int, default=16)
    parser.add_argument("--deadline-ms", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not str(args.device).startswith("cuda") or not torch.cuda.is_available():
        raise SystemExit("runtime prefix harness requires CUDA")
    sys.path.insert(0, str(args.moshi_source_root.resolve()))
    import sentencepiece
    from moshi.models.lm import LMGen
    from moshi.models.loaders import get_moshi_lm

    frame = validate_control_frame_mapping(json.loads(args.frame_json.read_text(encoding="utf-8")))
    evidence_inputs = (args.supporting_frame_json, args.evidence_frame_json, args.evidence_adapter_checkpoint)
    evidence_mode = any(value is not None for value in evidence_inputs)
    if evidence_mode and not all(value is not None for value in evidence_inputs):
        raise SystemExit("evidence mode requires --supporting-frame-json, --evidence-frame-json, and --evidence-adapter-checkpoint together")
    supporting_frame = None
    evidence_frame = None
    if evidence_mode:
        supporting_frame = validate_control_frame_mapping(json.loads(args.supporting_frame_json.read_text(encoding="utf-8")))
        evidence_frame = validate_evidence_frame_mapping(json.loads(args.evidence_frame_json.read_text(encoding="utf-8")))
        assert_evidence_control_alignment(frame, evidence_frame)
        if supporting_frame.conversation_id != frame.conversation_id:
            raise SystemExit("supporting control frame and successor frame must use the same conversationId")
        if evidence_frame.supports_control_revision != supporting_frame.state_revision:
            raise SystemExit("evidence frame must support the supplied prior control revision")
    device = torch.device(args.device)
    lm = get_moshi_lm(args.moshi_path.resolve(), device=device, dtype=torch.bfloat16)
    lm.eval()
    try:
        checkpoint = torch.load(args.adapter_checkpoint.resolve(), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.adapter_checkpoint.resolve(), map_location="cpu")
    state = checkpoint.get("adapter_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise SystemExit("adapter checkpoint has no adapter_state_dict")
    adapter = SemanticPrefixAdapter(text_cardinality=int(lm.text_card), hidden_size=int(lm.dim), prefix_frames=args.prefix_frames).to(device, dtype=torch.bfloat16)
    adapter.load_state_dict(state, strict=True)
    adapter.eval()
    generator = LMGen(lm, device=device)
    generator.streaming_forever(1)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer_path.resolve()))
    provider = SemanticPrefixProvider(
        lm_gen=generator,
        adapter=adapter,
        tokenizer=tokenizer,
        adapter_version=sha256_file(args.adapter_checkpoint.resolve()),
    )
    evidence_provider = None
    evidence_adapter_sha256 = None
    if evidence_mode:
        try:
            evidence_checkpoint = torch.load(args.evidence_adapter_checkpoint.resolve(), map_location="cpu", weights_only=True)
        except TypeError:
            evidence_checkpoint = torch.load(args.evidence_adapter_checkpoint.resolve(), map_location="cpu")
        evidence_state = evidence_checkpoint.get("evidence_adapter_state_dict") if isinstance(evidence_checkpoint, dict) else None
        if not isinstance(evidence_state, dict):
            raise SystemExit("evidence adapter checkpoint has no evidence_adapter_state_dict")
        expected_control_sha256 = evidence_checkpoint.get("control_adapter_checkpoint_sha256") if isinstance(evidence_checkpoint, dict) else None
        actual_control_sha256 = sha256_file(args.adapter_checkpoint.resolve())
        if expected_control_sha256 != actual_control_sha256:
            raise SystemExit("evidence adapter checkpoint was not trained against this control adapter checkpoint")
        evidence_adapter = EvidenceStreamAdapter(
            text_cardinality=int(lm.text_card),
            hidden_size=int(lm.dim),
            stream_frames=args.evidence_stream_frames,
        ).to(device, dtype=torch.bfloat16)
        evidence_adapter.load_state_dict(evidence_state, strict=True)
        evidence_adapter.eval()
        evidence_adapter_sha256 = sha256_file(args.evidence_adapter_checkpoint.resolve())
        evidence_provider = EvidenceStreamProvider(
            lm_gen=generator,
            adapter=evidence_adapter,
            tokenizer=tokenizer,
            adapter_version=evidence_adapter_sha256,
        )
    session = RuntimeControlSession(
        call_id=frame.conversation_id,
        prefix_provider=provider,
        evidence_provider=evidence_provider,
        prefill_deadline_ms=args.deadline_ms,
    )
    now = int(time.time() * 1000)
    initial_frame = supporting_frame if evidence_mode else frame
    initial_update = RuntimeControlUpdate.from_mapping(
        {
            "type": "control.update",
            "protocolVersion": 2,
            "callId": initial_frame.conversation_id,
            "revision": initial_frame.state_revision,
            "contextHash": initial_frame.state_hash,
            "expiresAtUnixMs": now + 7000,
            "frame": initial_frame.as_wire_dict(),
        }
    )
    queued = session.submit(initial_update, now_unix_ms=now)
    applied = session.apply_boundary(
        call_id=initial_frame.conversation_id,
        turn_id=initial_frame.target_turn_id,
        context_hash=initial_frame.state_hash,
        now_unix_ms=now + 1,
    )
    evidence_result = None
    if evidence_mode:
        staged_update = RuntimeEvidenceUpdate.from_mapping(
            {
                "type": "evidence.update",
                "protocolVersion": 2,
                "callId": evidence_frame.conversation_id,
                "revision": evidence_frame.evidence_revision,
                "supportsControlRevision": evidence_frame.supports_control_revision,
                "contextHash": evidence_frame.context_hash,
                "expiresAtUnixMs": now + 7000,
                "evidenceId": evidence_frame.evidence_id,
                "provenance": dict(evidence_frame.provenance),
                "allowedClaims": list(evidence_frame.allowed_claims),
                "availability": evidence_frame.availability,
            }
        )
        staged = session.submit_evidence(staged_update, now_unix_ms=now + 2)
        successor = RuntimeControlUpdate.from_mapping(
            {
                "type": "control.update",
                "protocolVersion": 2,
                "callId": frame.conversation_id,
                "revision": frame.state_revision,
                "contextHash": frame.state_hash,
                "expiresAtUnixMs": now + 7000,
                "frame": frame.as_wire_dict(),
                "evidenceFrame": evidence_frame.as_wire_dict(),
            }
        )
        successor_queued = session.submit(successor, now_unix_ms=now + 3)
        successor_applied = session.apply_boundary(
            call_id=frame.conversation_id,
            turn_id=frame.target_turn_id,
            context_hash=frame.state_hash,
            now_unix_ms=now + 4,
        )
        barge_in = session.caller_barge_in(reason="harness_barge_in")
        evidence_ok = (
            applied.status == "applied"
            and staged[-1].status == "evidence_staged"
            and successor_applied.status == "applied"
            and not session.may_emit(successor_applied.generation_id)
            and any(ack.status == "superseded" and ack.revision == frame.state_revision for ack in barge_in)
        )
        evidence_result = {
            "staged": [ack.as_wire_dict() for ack in staged],
            "successor_queued": [ack.as_wire_dict() for ack in successor_queued],
            "successor_applied": successor_applied.as_wire_dict(),
            "barge_in": [ack.as_wire_dict() for ack in barge_in],
            "emit_invalidated": not session.may_emit(successor_applied.generation_id),
            "passed": evidence_ok,
        }
        applied = successor_applied
    result = {
        "schema_version": 1,
        "kind": "personaplex-runtime-prefix-harness",
        "status": "passed" if applied.status == "applied" and (evidence_result is None or evidence_result["passed"]) else "failed",
        "queued_statuses": [ack.status for ack in queued],
        "applied": applied.as_wire_dict(),
        "base_model_sha256": sha256_file(args.moshi_path.resolve()),
        "adapter_sha256": sha256_file(args.adapter_checkpoint.resolve()),
        "evidence_adapter_sha256": evidence_adapter_sha256,
        "evidence": evidence_result,
        "device": str(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "prefixPrefillMs": applied.prefix_prefill_ms}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
