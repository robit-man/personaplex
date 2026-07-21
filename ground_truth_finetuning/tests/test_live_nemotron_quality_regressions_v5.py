from __future__ import annotations

import json

from ground_truth_finetuning.training.scenario_blueprint_v5 import (
    MODEL_ATTEMPT_TRACE_SCHEMA,
    _write_model_attempt_trace,
)
from ground_truth_finetuning.training.scenario_taxonomy_admission_v5 import (
    ATOMIC_QUALITY_TAXONOMY_FINDING_CODES,
    TAXONOMY_FINDING_CODES,
    TAXONOMY_FINDING_DEFINITIONS,
    TAXONOMY_ATOMIC_QUALITY_SYSTEM,
    TAXONOMY_JUDGE_SYSTEM,
    TAXONOMY_REPAIR_SYSTEM,
    build_taxonomy_judge_response_schema,
    validate_taxonomy_judgment,
)


QUALITY_CODES = {
    "language_or_encoding_corruption",
    "incomplete_or_malformed_field",
    "unnatural_or_placeholder_content",
}


def test_live_quality_defects_are_typed_and_schema_admissible() -> None:
    ids = tuple(f"scenario_topic_test_{ordinal:02d}" for ordinal in range(1, 21))
    schema = build_taxonomy_judge_response_schema(ids)

    assert QUALITY_CODES.issubset(TAXONOMY_FINDING_CODES)
    assert QUALITY_CODES.issubset(TAXONOMY_FINDING_DEFINITIONS)
    assert QUALITY_CODES.issubset(
        set(schema["properties"]["findingClusters"]["items"]["properties"]["code"]["enum"])
    )
    normalized = validate_taxonomy_judgment(
        {
            "findingClusters": [
                {"code": code, "scenarioIds": [ids[index]]}
                for index, code in enumerate(sorted(QUALITY_CODES))
            ]
        },
        ids,
    )
    assert {item["code"] for item in normalized["findingClusters"]} == QUALITY_CODES


def test_quality_contract_is_model_judged_not_host_lexical_filtering() -> None:
    assert set(ATOMIC_QUALITY_TAXONOMY_FINDING_CODES) == QUALITY_CODES
    assert "exactly one supplied" in TAXONOMY_ATOMIC_QUALITY_SYSTEM
    assert "Do not audit for other defects" in TAXONOMY_ATOMIC_QUALITY_SYSTEM
    assert "fluent natural English" in TAXONOMY_REPAIR_SYSTEM
    assert "No field has a character" in TAXONOMY_REPAIR_SYSTEM
    assert "natural semantic boundary" in TAXONOMY_REPAIR_SYSTEM
    assert "apply lexical matching" in TAXONOMY_ATOMIC_QUALITY_SYSTEM
    assert "local language" in TAXONOMY_JUDGE_SYSTEM


def test_model_attempt_trace_persists_output_without_prompt_or_credentials(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PERSONAPLEX_MODEL_TRACE_ROOT", str(tmp_path))
    projection = {
        "canonicalSchemaHash": "sha256:canonical",
        "transportSchemaHash": "sha256:transport",
        "profile": {"profileHash": "sha256:profile"},
    }
    _write_model_attempt_trace(
        name="personaplex_test",
        context={"topicCard": {"topicId": "topic_test"}},
        route_name="fallback",
        endpoint="http://127.0.0.1:11434/v1/chat/completions",
        logical_model="logical-model",
        actual_model="nemotron-3-super:120b",
        attempt=1,
        finish_reason="stop",
        projection_binding=projection,
        messages=[{"role": "system", "content": "secret prompt"}],
        usage={"completion_tokens": 12},
        status="canonical_schema_accepted",
        response={"value": "synthetic output"},
    )

    paths = list(tmp_path.rglob("*.json"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text(encoding="utf-8"))
    assert record["schema"] == MODEL_ATTEMPT_TRACE_SCHEMA
    assert record["response"] == {"value": "synthetic output"}
    assert "secret prompt" not in paths[0].read_text(encoding="utf-8")
    assert "authorization" not in paths[0].read_text(encoding="utf-8").lower()
