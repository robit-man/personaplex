"""Training primitives for semantically controlled PersonaPlex adapters."""

from .contracts import (
    ControlPlan,
    ControlTrainingFrame,
    EvidenceTrainingFrame,
    assert_evidence_control_alignment,
    validate_control_frame_mapping,
    validate_evidence_frame_mapping,
    validate_plan_mapping,
)
from .evidence_conditioning import EvidenceStreamAdapter, MoshiStreamingSumBridge
from .plan_serializer import PlanSerializer

__all__ = [
    "ControlPlan",
    "ControlTrainingFrame",
    "EvidenceStreamAdapter",
    "EvidenceTrainingFrame",
    "MoshiStreamingSumBridge",
    "PlanSerializer",
    "assert_evidence_control_alignment",
    "validate_control_frame_mapping",
    "validate_evidence_frame_mapping",
    "validate_plan_mapping",
]
