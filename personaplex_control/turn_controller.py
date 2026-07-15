"""Turn-boundary state machine for semantic guidance.

It prevents stale or mid-utterance prompt replacement. The adapter owns audio
reset/prefill details; this class only decides whether a revision is safe to
apply and emits an auditable acknowledgement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contracts import ControlAck, ControlUpdate


class ControlDisposition(str, Enum):
    QUEUED = "queued"
    REJECTED = "rejected"
    APPLIED = "applied"


@dataclass
class TurnController:
    call_id: str
    context_hash: str
    last_completed_turn_id: int = 0
    last_seen_revision: int = 0
    active_revision: int = 0
    pending: ControlUpdate | None = None

    def submit(self, update: ControlUpdate) -> tuple[ControlDisposition, ControlAck]:
        if update.call_id != self.call_id:
            return self._reject(update, "call_id_mismatch")
        if update.revision <= self.last_seen_revision:
            return self._reject(update, "stale_revision")
        if update.base_context_hash != self.context_hash:
            return self._reject(update, "context_hash_mismatch")
        if update.apply_after_turn_id < self.last_completed_turn_id:
            return self._reject(update, "expired_turn_boundary")

        self.last_seen_revision = update.revision
        self.pending = update
        return ControlDisposition.QUEUED, ControlAck(
            call_id=self.call_id,
            revision=update.revision,
            applied=False,
            reason="queued_for_caller_turn_boundary",
        )

    def complete_caller_turn(self, turn_id: int, context_hash: str) -> ControlAck | None:
        if turn_id <= self.last_completed_turn_id:
            raise ValueError("turn_id must advance")
        self.last_completed_turn_id = turn_id
        self.context_hash = context_hash
        if self.pending is None or self.pending.apply_after_turn_id > turn_id:
            return None
        if self.pending.base_context_hash != context_hash:
            update = self.pending
            self.pending = None
            return ControlAck(self.call_id, update.revision, False, "context_advanced_before_apply", turn_id)

        update = self.pending
        self.pending = None
        self.active_revision = update.revision
        return ControlAck(self.call_id, update.revision, True, "applied_at_caller_turn_boundary", turn_id)

    def cancel_pending(self, reason: str = "cancelled") -> ControlAck | None:
        if self.pending is None:
            return None
        update = self.pending
        self.pending = None
        return ControlAck(self.call_id, update.revision, False, reason)

    def _reject(self, update: ControlUpdate, reason: str) -> tuple[ControlDisposition, ControlAck]:
        return ControlDisposition.REJECTED, ControlAck(
            call_id=self.call_id,
            revision=update.revision,
            applied=False,
            reason=reason,
        )
