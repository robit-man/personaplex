from ground_truth_finetuning.training.causal_coverage import (
    CausalCoverageThresholds,
    build_causal_coverage_report,
    classify_changed_path,
    intervention_kind,
)


def example(example_id: str, premise: str, *, barge_in: bool = False) -> dict:
    return {
        "example_id": example_id,
        "control": {
            "frame": {
                "state": {
                    "callerPosture": "cooperative",
                    "compliancePosture": "accepts_verified_next_steps",
                    "resistancePosture": "low_resistance",
                    "semanticBindings": {
                        "counterfactualAxis": "tool_result.status",
                        "scenarioPremise": premise,
                    },
                },
                "turnTaking": {
                    "expectedBargeIn": barge_in,
                    "recoveryExpected": barge_in,
                },
            }
        },
    }


def pair(pair_id: str, group_id: str, split: str, left: str, right: str, paths=None) -> dict:
    return {
        "pair_id": pair_id,
        "group_id": group_id,
        "split": split,
        "member_a": {"example_id": left},
        "member_b": {"example_id": right},
        "changed_paths": paths
        or [
            "state.facts",
            "state.semanticBindings.branchId",
            "state.semanticBindings.concreteUpdate",
            "state.toolResults",
        ],
    }


def thresholds(**overrides) -> CausalCoverageThresholds:
    values = {
        "expected_axes": ("tool_result.status",),
        "required_splits": ("train", "validation"),
        "min_pairs_per_axis": 2,
        "min_distinct_premises_per_axis": 2,
        "min_signature_support": 2,
        "min_supported_pair_fraction": 1.0,
        "max_composite_fraction": 0.0,
        "min_barge_in_pairs": 1,
        "min_recovery_pairs": 1,
    }
    values.update(overrides)
    return CausalCoverageThresholds(**values)


def certified_fixture():
    examples = {
        row["example_id"]: row
        for row in (
            example("a1", "premise-one", barge_in=True),
            example("b1", "premise-one", barge_in=True),
            example("a2", "premise-two"),
            example("b2", "premise-two"),
        )
    }
    pairs = [
        pair("pair-1", "group-1", "train", "a1", "b1"),
        pair("pair-2", "group-2", "validation", "a2", "b2"),
    ]
    return pairs, examples


def test_schema_path_classification_ignores_lineage() -> None:
    assert classify_changed_path("state.facts") == "semantic"
    assert classify_changed_path("plan.delivery.register") == "delivery"
    assert classify_changed_path("turnTaking.expectedBargeIn") == "turn_taking"
    assert intervention_kind(
        ["state.facts", "state.semanticBindings.branchId"]
    ) == "semantic_only"


def test_certifies_repeated_group_disjoint_semantic_interventions() -> None:
    pairs, examples = certified_fixture()
    report = build_causal_coverage_report(pairs, examples, thresholds())
    assert report["status"] == "certified"
    assert report["supportedPairFraction"] == 1.0
    assert report["groupSplitLeakageCount"] == 0


def test_rejects_group_split_leakage() -> None:
    pairs, examples = certified_fixture()
    pairs[1]["group_id"] = "group-1"
    report = build_causal_coverage_report(pairs, examples, thresholds())
    assert report["status"] == "rejected"
    assert report["groupSplitLeakageCount"] == 1


def test_rejects_composite_interventions() -> None:
    pairs, examples = certified_fixture()
    pairs[0]["changed_paths"].append("plan.delivery.register")
    report = build_causal_coverage_report(
        pairs,
        examples,
        thresholds(min_supported_pair_fraction=0.0),
    )
    assert report["status"] == "rejected"
    assert report["compositeFraction"] == 0.5
