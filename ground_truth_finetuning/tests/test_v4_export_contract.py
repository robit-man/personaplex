"""Focused regressions for V4 replay and control/evidence lineage admission."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

from ground_truth_finetuning.tools.export_controlled_duplex_dataset import target_control_issues
from ground_truth_finetuning.tools.export_v7_evidence_frames import (
    assert_record_frame_identity,
    require_turn_admission,
)


class V4ExportContractTests(unittest.TestCase):
    def test_replayed_target_is_context_not_a_label(self) -> None:
        record = {
            "speaker": "target",
            "turnIndex": 1,
            "quality": {"accepted": True},
            "replay": {"role": "shared_prefix_context_only"},
            "training": {
                "eligible": False,
                "exclusionReasons": ["shared_prefix_replay_context_only"],
            },
        }
        require_turn_admission(record)
        self.assertEqual(target_control_issues(record), [])

    def test_replayed_target_must_be_quarantined(self) -> None:
        record = {
            "speaker": "target",
            "turnIndex": 1,
            "quality": {"accepted": True},
            "replay": {"role": "shared_prefix_context_only"},
            "training": {"eligible": True, "exclusionReasons": []},
        }
        with self.assertRaisesRegex(ValueError, "not quarantined"):
            require_turn_admission(record)
        self.assertEqual(target_control_issues(record), ["shared_prefix_replay_not_quarantined"])

    def test_control_and_evidence_must_match_enclosing_turn(self) -> None:
        record = {"conversationId": "branch-b"}
        with self.assertRaisesRegex(ValueError, "different conversation"):
            assert_record_frame_identity(
                record,
                SimpleNamespace(conversation_id="branch-a"),
                SimpleNamespace(conversation_id="branch-a"),
            )
        assert_record_frame_identity(
            record,
            SimpleNamespace(conversation_id="branch-b"),
            SimpleNamespace(conversation_id="branch-b"),
        )


if __name__ == "__main__":
    unittest.main()
