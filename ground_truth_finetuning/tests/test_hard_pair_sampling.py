from collections import Counter

import pytest

from ground_truth_finetuning.training.hard_pair_sampling import (
    HARD_PAIR_CURRICULUM_SCHEMA,
    HardPairCurriculum,
)


def test_hard_curriculum_preserves_uniform_coverage_and_replays() -> None:
    curriculum = HardPairCurriculum(
        pair_ids=("a", "b", "c"),
        weights=(20.0, 1.0, 1.0),
        replay_ratio=1.0,
        seed=7,
    )
    first = curriculum.cycle(0)
    assert len(first) == 6
    counts = Counter(first)
    assert all(counts[pair_id] >= 1 for pair_id in curriculum.pair_ids)
    assert first == curriculum.cycle(0)
    assert [curriculum.pair_id_for_sample(i) for i in range(6)] == list(first)
    assert curriculum.cycle(1) != first


def test_hard_curriculum_requires_exact_pair_coverage() -> None:
    document = {
        "schema": HARD_PAIR_CURRICULUM_SCHEMA,
        "pairs": [{"pairId": "a", "weight": 1.0}],
    }
    with pytest.raises(ValueError, match="exactly cover"):
        HardPairCurriculum.from_document(
            document,
            expected_pair_ids=("a", "b"),
            replay_ratio=1.0,
            seed=0,
        )
