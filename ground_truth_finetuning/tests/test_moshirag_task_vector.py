from ground_truth_finetuning.training.moshirag_task_vector import (
    candidate_targets,
    task_vector_scope,
)


def test_task_vector_scope_excludes_audio_paths() -> None:
    assert task_vector_scope("transformer.layers.2.self_attn.in_proj_weight") == "temporal"
    assert task_vector_scope("text_linear.weight") == "text"
    assert task_vector_scope("emb.0.weight") is None
    assert task_vector_scope("depformer.layers.0.self_attn.in_proj_weight") is None


def test_task_vector_targets_keep_text_unchanged_for_temporal_transfer() -> None:
    assert candidate_targets("temporal", 0.75) == {"temporal": 0.75, "text": 0.0}
