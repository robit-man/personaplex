from dataclasses import dataclass
from types import SimpleNamespace

from personaplex_control.moshirag_reference import (
    render_arc4_reference,
    render_arc4_reference_fields,
)


@dataclass(frozen=True)
class _Plan:
    plan_hash: str = "sha256:plan"


def test_arc4_reference_uses_canonical_field_envelope(monkeypatch) -> None:
    fields = {
        "decision": "CONTROL_WITHOUT_TARGET",
        "state": "BOUND_STATE_WITHOUT_TARGET",
        "delivery": "DELIVERY_WITHOUT_TARGET",
        "context": "CONTEXT_WITHOUT_TARGET",
    }
    monkeypatch.setattr(
        "personaplex_control.moshirag_reference.render_arc4_reference_fields",
        lambda control, evidence: fields,
    )
    value = render_arc4_reference(object(), object())
    assert '"v":"personaplex-semantic-reference-v6-budget-first-no-lineage"' in value
    for field_value in fields.values():
        assert field_value in value


def test_arc4_reference_excludes_corpus_lineage(monkeypatch) -> None:
    monkeypatch.setattr(
        "personaplex_control.moshirag_reference.assert_evidence_control_alignment",
        lambda control, evidence: None,
    )
    constraints = SimpleNamespace(
        required_facts=("replacement_shipped",),
        forbidden_claims=("invent_delivery_date",),
        must_ask=(),
        must_not_request=(),
    )
    delivery = SimpleNamespace(
        register="conversational",
        assertiveness=0.4,
        speaking_rate_bucket="moderate",
        pause_density_bucket="low",
        interruptibility="yield_on_caller_speech",
        max_duration_ms=5000,
    )
    plan = SimpleNamespace(
        intent="resolve_delivery_issue",
        dialogue_act="offer_options",
        constraints=constraints,
        delivery=delivery,
    )
    control = SimpleNamespace(
        state={
            "intent": "resolve_delivery_issue",
            "facts": ("control_event:unique_training_branch_available",),
            "semanticBindings": {
                "concreteUpdate": "The replacement shipped and awaits a carrier scan.",
                "controlKind": "tool_result_update",
                "counterfactualAxis": "shipment.status",
            },
        },
        plan=plan,
        state_revision=42,
        turn_taking={},
    )
    evidence = SimpleNamespace(
        availability="ready",
        allowed_claims=("The replacement has shipped.",),
        provenance={
            "record": "topic_99_target_8_available",
            "source": "shipment_tool",
            "confidence": "high",
        },
    )
    fields = render_arc4_reference_fields(control, evidence)
    serialized = "".join(fields.values())
    assert "replacement has shipped" in serialized
    assert "shipment_tool" in serialized
    assert "unique_training_branch" not in serialized
    assert "topic_99_target_8_available" not in serialized
