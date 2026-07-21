"""Unit contracts for v4 encoding, temporal alignment, and release statistics."""

from __future__ import annotations

import unittest

import torch

from ground_truth_finetuning.evaluation.reliability import evaluate_release_gate
from ground_truth_finetuning.training.contracts import validate_control_frame_mapping
from ground_truth_finetuning.training.control_encoding import (
    ControlSegment,
    FieldAwareControlSerializer,
    _clean_text,
    pad_encoded_controls,
)
from ground_truth_finetuning.training.control_stream import (
    ControlStreamConfig,
    SemanticControlStreamAdapter,
)
from ground_truth_finetuning.training.native_training import (
    align_control_stream,
    exact_text_contrast_masks,
)


class Tokenizer:
    def __init__(self) -> None:
        self.ids: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        values = []
        for word in text.split():
            if word not in self.ids:
                self.ids[word] = len(self.ids) + 1
            values.append(self.ids[word])
        return values


def frame() -> object:
    state_hash = "sha256:" + "b" * 64
    return validate_control_frame_mapping(
        {
            "schemaVersion": 1,
            "frameId": "v4-test-frame",
            "conversationId": "v4-test-call",
            "targetTurnId": 2,
            "stateRevision": 4,
            "baseStateHash": "sha256:" + "a" * 64,
            "stateHash": state_hash,
            "semanticSources": ["state_reducer", "task_agent", "asr_finalizer"],
            "state": {
                "intent": "resolve a delayed delivery",
                "nextGoal": "explain the verified replacement status and offer escalation",
                "facts": ["the replacement shipped on July fourteenth"],
                "uncertainty": ["the carrier has not scanned the parcel"],
                "callerPosture": "skeptical",
                "endCallAuthorized": False,
                "textContext": {
                    "turns": [
                        {
                            "speaker": "caller",
                            "source": "asr",
                            "text": "I still do not see a tracking update.",
                        }
                    ]
                },
            },
            "update": {
                "applyAt": "next_agent_turn_boundary",
                "expiresAtMs": 5000,
                "reason": "caller_turn_finalized",
            },
            "turnTaking": {
                "expectedBargeIn": False,
                "interruptionPolicy": "yield_on_caller_speech",
            },
            "plan": {
                "schemaVersion": 1,
                "callId": "v4-test-call",
                "turnId": 2,
                "revision": 4,
                "contextHash": state_hash,
                "mode": "expressive",
                "intent": "resolve the delivery issue",
                "dialogueAct": "inform_and_offer",
                "entities": {"shipment status": "awaiting carrier scan"},
                "constraints": {
                    "required_facts": ["replacement shipped July fourteenth"],
                    "forbidden_claims": ["guaranteed delivery date"],
                    "must_ask": ["whether escalation would help"],
                    "must_not_request": ["payment card"],
                },
                "delivery": {
                    "language": "en-US",
                    "register": "warm",
                    "assertiveness": 0.35,
                    "interruptibility": "yield_on_caller_speech",
                    "max_duration_ms": 5000,
                    "speaking_rate_bucket": "normal",
                    "pause_density_bucket": "low",
                    "emphasis_targets": [],
                },
                "expiryMs": 5000,
            },
        }
    )


class SemanticControlV4Tests(unittest.TestCase):
    def test_control_text_sanitization_never_applies_a_character_ceiling(self) -> None:
        semantic_text = "A complete semantic statement. " * 100
        self.assertEqual(_clean_text(semantic_text), semantic_text.strip())

    def test_control_budget_omits_overflow_as_a_whole_semantic_segment(self) -> None:
        class WholeSegmentSerializer(FieldAwareControlSerializer):
            def segments(self, _frame, _evidence=None):
                return [
                    ControlSegment("header", "metadata", 0, " ".join(f"kept{n}" for n in range(28)), 0),
                    ControlSegment("other_state", "natural_text", 0, " ".join(f"omitted{n}" for n in range(10)), 8),
                ]

        tokenizer = Tokenizer()
        encoded = WholeSegmentSerializer().encode(frame(), tokenizer, 10000, max_tokens=32)
        self.assertEqual(len(encoded.token_ids), 28)
        self.assertFalse(any(tokenizer.ids[word] in encoded.token_ids for word in tokenizer.ids if word.startswith("omitted")))

    def test_natural_lexical_encoding_and_typed_channels_align(self) -> None:
        tokenizer = Tokenizer()
        encoded = FieldAwareControlSerializer().encode(
            frame(), tokenizer, 10000, max_tokens=128
        )
        self.assertGreater(len(encoded.token_ids), 20)
        self.assertEqual(len(encoded.token_ids), len(encoded.field_ids))
        self.assertGreater(len(set(encoded.field_ids)), 4)

    def test_adapter_is_initially_noop_and_null_remains_exact_zero(self) -> None:
        tokenizer = Tokenizer()
        encoded = FieldAwareControlSerializer().encode(
            frame(), tokenizer, 10000, max_tokens=64
        )
        typed = pad_encoded_controls(
            [encoded, encoded], device=torch.device("cpu"), present=[True, False]
        )
        adapter = SemanticControlStreamAdapter(
            lm_hidden_size=32,
            config=ControlStreamConfig(
                control_dim=16,
                encoder_layers=1,
                attention_heads=4,
                stream_frames=4,
                max_tokens=64,
                dropout=0.0,
            ),
        ).eval()
        lexical = torch.randn(2, typed.token_ids.shape[1], 32)
        stream = adapter(lexical, typed)
        self.assertEqual(tuple(stream.shape), (2, 4, 32))
        self.assertTrue(torch.equal(stream[0], torch.zeros_like(stream[0])))
        self.assertTrue(torch.equal(stream[1], torch.zeros_like(stream[1])))
        stream[0].sum().backward()
        gradients = [
            parameter.grad
            for parameter in adapter.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        adapter.zero_grad(set_to_none=True)
        with torch.no_grad():
            adapter.gate_bias.fill_(0.1)
            adapter.lexical_gate_bias.fill_(0.1)
        stream = adapter(lexical, typed)
        self.assertGreater(float(stream[0].abs().sum()), 0.0)
        self.assertTrue(torch.equal(stream[1], torch.zeros_like(stream[1])))
        lexical_rms = lexical[0].square().mean().sqrt()
        stream_rms = stream[0].square().mean().sqrt()
        self.assertLessEqual(
            float(stream_rms),
            float(lexical_rms * adapter.config.max_stream_to_lexical_rms_ratio) + 1e-5,
        )

    def test_temporal_stream_starts_on_real_boundary_and_stops(self) -> None:
        stream = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
        aligned = align_control_stream(stream, torch.tensor([2]), 7)
        self.assertTrue(torch.equal(aligned[:, :2], torch.zeros(1, 2, 4)))
        self.assertTrue(torch.equal(aligned[:, 2:5], stream))
        self.assertTrue(torch.equal(aligned[:, 5:], torch.zeros(1, 2, 4)))

    def test_exact_contrast_marks_only_branch_distinct_target_text(self) -> None:
        codes_a = torch.tensor(
            [[0, 11, 12, 13, 0, 14, 0], [91, 92, 93, 94, 95, 96, 97]]
        )
        codes_b = torch.tensor(
            [[0, 11, 22, 13, 0, 24, 0], [81, 82, 83, 84, 85, 86, 87]]
        )
        target_a = torch.ones_like(codes_a, dtype=torch.bool)
        target_b = torch.ones_like(codes_b, dtype=torch.bool)
        contrast = exact_text_contrast_masks(
            codes_a,
            target_a,
            codes_b,
            target_b,
            text_stream_index=0,
            zero_token_id=0,
        )
        self.assertEqual(contrast.shared_tokens, 2)
        self.assertEqual(contrast.changed_tokens_a, 2)
        self.assertEqual(contrast.changed_tokens_b, 2)
        self.assertEqual(torch.where(contrast.mask_a[0])[0].tolist(), [2, 5])
        self.assertEqual(torch.where(contrast.mask_b[0])[0].tolist(), [2, 5])
        self.assertFalse(contrast.mask_a[1].any())
        self.assertFalse(contrast.mask_b[1].any())

    def test_release_gate_counts_judge_failure_in_first_attempt_denominator(self) -> None:
        good_judgment = {
            "status": "ok",
            "semantic_adherence": True,
            "required_facts_supported": True,
            "forbidden_claims_avoided": True,
            "required_question_or_action": True,
            "next_goal_advanced": True,
            "caller_posture_respected": True,
            "style_adherence": True,
            "natural_conversational_response": True,
            "unsupported_policy_sensitive_claim": False,
            "stale_control_used": False,
        }
        trials = [
            {
                "trial_id": "a",
                "judgment": good_judgment,
                "audio_checks": {"admitted": True},
                "transport_checks": {"passed": True},
                "slices": {"topic": "service"},
                "pair_id": "pair",
                "branch_id": "available",
                "pair_discrimination_pass": True,
            },
            {
                "trial_id": "b",
                "judgment": {"status": "failed"},
                "audio_checks": {"admitted": True},
                "transport_checks": {"passed": True},
                "slices": {"topic": "service"},
                "pair_id": "pair",
                "branch_id": "constrained",
                "pair_discrimination_pass": True,
            },
        ]
        report = evaluate_release_gate(
            trials, minimum_trials=2, minimum_pairs=1, minimum_slice_trials=2
        )
        self.assertEqual(report["first_attempt_denominator"], 2)
        self.assertEqual(report["judge_failures"], 1)
        self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
