from ground_truth_finetuning.tools.certify_arc4_causal_budget import (
    changed_families,
    field_budget_result,
    first_difference,
)


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        prefix = [1] if add_special_tokens else []
        return prefix + [ord(character) for character in text]


def test_changed_families_are_structural_and_ignore_lineage() -> None:
    assert changed_families(
        [
            "state.facts",
            "state.semanticBindings.branchId",
            "plan.delivery.register",
        ]
    ) == ("delivery", "semantic")


def test_first_difference_accounts_for_length() -> None:
    assert first_difference([1, 2], [1, 3]) == 1
    assert first_difference([1], [1, 2]) == 1
    assert first_difference([1], [1]) is None


def test_field_budget_rejects_late_difference() -> None:
    result = field_budget_result(
        "common-a",
        "common-b",
        Tokenizer(),
        arc_frames=16,
        head_frames=4,
    )
    assert result["firstDifferentToken"] == 8
    assert result["reasons"] == ["first_difference_outside_causal_head"]
