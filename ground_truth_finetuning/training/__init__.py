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
from .control_encoding import FieldAwareControlSerializer
from .control_stream import ControlStreamConfig, SemanticControlStreamAdapter
from .plan_serializer import PlanSerializer

__all__ = [
    "ControlPlan",
    "ControlTrainingFrame",
    "EvidenceStreamAdapter",
    "EvidenceTrainingFrame",
    "FieldAwareControlSerializer",
    "MoshiStreamingSumBridge",
    "PlanSerializer",
    "ControlStreamConfig",
    "SemanticControlStreamAdapter",
    "assert_evidence_control_alignment",
    "validate_control_frame_mapping",
    "validate_evidence_frame_mapping",
    "validate_plan_mapping",
]
