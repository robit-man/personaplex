"""Deterministic coverage-preserving replay for hard ARC-4 causal pairs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import random
from typing import Any, Mapping, Sequence


HARD_PAIR_CURRICULUM_SCHEMA = "personaplex.arc4-hard-pair-curriculum.v1"


@dataclass(frozen=True)
class HardPairCurriculum:
    pair_ids: tuple[str, ...]
    weights: tuple[float, ...]
    replay_ratio: float
    seed: int

    def __post_init__(self) -> None:
        if not self.pair_ids or len(self.pair_ids) != len(self.weights):
            raise ValueError("hard curriculum pairs and weights must be aligned")
        if len(set(self.pair_ids)) != len(self.pair_ids):
            raise ValueError("hard curriculum pair IDs must be unique")
        if self.replay_ratio <= 0.0:
            raise ValueError("hard replay ratio must be positive")
        if any(not math.isfinite(weight) or weight <= 0.0 for weight in self.weights):
            raise ValueError("hard curriculum weights must be finite and positive")

    @classmethod
    def from_document(
        cls,
        document: Mapping[str, Any],
        *,
        expected_pair_ids: Sequence[str],
        replay_ratio: float,
        seed: int,
    ) -> "HardPairCurriculum":
        if document.get("schema") != HARD_PAIR_CURRICULUM_SCHEMA:
            raise ValueError("unsupported hard-pair curriculum schema")
        rows = document.get("pairs")
        if not isinstance(rows, list):
            raise ValueError("hard-pair curriculum lacks pair rows")
        weights: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("hard-pair curriculum row is not an object")
            pair_id = str(row.get("pairId") or "")
            if not pair_id or pair_id in weights:
                raise ValueError("hard-pair curriculum has invalid or duplicate pair IDs")
            weights[pair_id] = float(row.get("weight"))
        expected = tuple(expected_pair_ids)
        if set(weights) != set(expected):
            raise ValueError("hard-pair curriculum does not exactly cover training pairs")
        return cls(
            pair_ids=expected,
            weights=tuple(weights[pair_id] for pair_id in expected),
            replay_ratio=replay_ratio,
            seed=seed,
        )

    @property
    def replay_count(self) -> int:
        return max(1, round(len(self.pair_ids) * self.replay_ratio))

    @property
    def cycle_size(self) -> int:
        return len(self.pair_ids) + self.replay_count

    @lru_cache(maxsize=16)
    def cycle(self, cycle_index: int) -> tuple[str, ...]:
        if cycle_index < 0:
            raise ValueError("hard curriculum cycle index cannot be negative")
        uniform = list(self.pair_ids)
        random.Random(self.seed + cycle_index * 2_000_003).shuffle(uniform)
        replay = random.Random(self.seed + cycle_index * 2_000_003 + 1).choices(
            self.pair_ids,
            weights=self.weights,
            k=self.replay_count,
        )
        output: list[str] = []
        uniform_index = 0
        replay_index = 0
        total = self.cycle_size
        for position in range(total):
            due = ((position + 1) * self.replay_count) // total
            if due > replay_index:
                output.append(replay[replay_index])
                replay_index += 1
            else:
                output.append(uniform[uniform_index])
                uniform_index += 1
        if uniform_index != len(uniform) or replay_index != len(replay):
            raise RuntimeError("hard curriculum interleave was incomplete")
        return tuple(output)

    def pair_id_for_sample(self, global_sample: int) -> str:
        if global_sample < 0:
            raise ValueError("global sample index cannot be negative")
        cycle_index, offset = divmod(global_sample, self.cycle_size)
        return self.cycle(cycle_index)[offset]
