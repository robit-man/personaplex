"""Structural control-plane regression tests without a synthetic target label."""

from __future__ import annotations

import unittest

import torch

from ground_truth_finetuning.training.contracts import validate_control_frame_mapping
from personaplex_control.runtime import PrefixCacheEntry, RuntimeControlSession, RuntimeControlUpdate


def frame_mapping(revision: int, context_hash: str) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "frameId": f"frame-{revision}",
        "conversationId": "runtime-control-test",
        "targetTurnId": revision,
        "stateRevision": revision,
        "baseStateHash": "sha256:" + "a" * 64,
        "stateHash": context_hash,
        "semanticSources": ["state_reducer", "policy_agent"],
        "state": {"intent": "provide a bounded status"},
        "update": {"applyAt": "next_agent_turn_boundary", "expiresAtMs": 1000},
        "turnTaking": {"callerMayInterrupt": True},
        "plan": {
            "schemaVersion": 1,
            "callId": "runtime-control-test",
            "turnId": revision,
            "revision": revision,
            "contextHash": context_hash,
            "mode": "expressive",
            "intent": "provide a bounded status",
            "dialogueAct": "inform",
            "entities": {},
            "constraints": {
                "required_facts": [],
                "forbidden_claims": [],
                "must_ask": [],
                "must_not_request": [],
            },
            "delivery": {
                "language": "en",
                "register": "neutral",
                "assertiveness": 0.5,
                "interruptibility": "yield_on_caller_speech",
                "max_duration_ms": 1000,
                "emphasis_targets": [],
            },
            "expiryMs": 1000,
        },
    }


def update(revision: int, context_hash: str) -> RuntimeControlUpdate:
    frame = validate_control_frame_mapping(frame_mapping(revision, context_hash))
    return RuntimeControlUpdate(
        call_id=frame.conversation_id,
        revision=frame.state_revision,
        context_hash=frame.state_hash,
        expires_at_unix_ms=9_999_999_999_999,
        frame=frame,
    )


class FakePrefixProvider:
    adapter_version = "test-adapter-sha256"

    def __init__(self) -> None:
        self.prefilled = []

    def build(self, frame):  # noqa: ANN001 - structural fake mirrors the provider protocol
        return PrefixCacheEntry(frame.frame_hash, torch.zeros((1, 1, 4)), 0.1)

    def prefill(self, entry):  # noqa: ANN001 - structural fake mirrors the provider protocol
        self.prefilled.append(entry.frame_hash)
        return 0.2


class FakeEvidenceProvider:
    def __init__(self) -> None:
        self.cancel_count = 0

    def cancel(self) -> None:
        self.cancel_count += 1


class RuntimeControlSessionTests(unittest.TestCase):
    def test_applied_revision_cancels_stale_audio_and_never_reapplies(self) -> None:
        prefix = FakePrefixProvider()
        evidence = FakeEvidenceProvider()
        session = RuntimeControlSession(
            call_id="runtime-control-test",
            prefix_provider=prefix,
            evidence_provider=evidence,
        )
        first = update(1, "sha256:" + "b" * 64)
        self.assertEqual(session.submit(first, now_unix_ms=1)[-1].status, "queued")
        applied = session.apply_boundary(
            call_id=first.call_id, turn_id=1, context_hash=first.context_hash, now_unix_ms=2
        )
        self.assertEqual(applied.status, "applied")
        self.assertTrue(session.may_emit(applied.generation_id))
        cancelled = session.caller_barge_in()
        self.assertEqual([ack.status for ack in cancelled], ["superseded"])
        self.assertEqual(evidence.cancel_count, 1)
        self.assertFalse(session.may_emit(applied.generation_id))
        self.assertEqual(session.submit(first, now_unix_ms=3)[0].status, "superseded")
        fallback = session.apply_boundary(
            call_id=first.call_id, turn_id=2, context_hash=first.context_hash, now_unix_ms=3
        )
        self.assertEqual(fallback.status, "safe_fallback")
        second = update(2, "sha256:" + "c" * 64)
        self.assertEqual(session.submit(second, now_unix_ms=4)[-1].status, "queued")
        self.assertEqual(
            session.apply_boundary(
                call_id=second.call_id, turn_id=3, context_hash=second.context_hash, now_unix_ms=5
            ).status,
            "applied",
        )
        self.assertEqual(len(prefix.prefilled), 2)


if __name__ == "__main__":
    unittest.main()
