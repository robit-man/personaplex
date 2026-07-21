"""CUDA task-vector utilities for transferring Moshika-RAG behavior into PersonaPlex."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def task_vector_scope(key: str) -> str | None:
    if key.startswith("transformer.layers."):
        return "temporal"
    if key.startswith(("text_emb.", "text_linear.", "out_norm.")):
        return "text"
    return None


def candidate_targets(scope: str, alpha: float) -> dict[str, float]:
    if scope == "none":
        return {"temporal": 0.0, "text": 0.0}
    if scope == "temporal":
        return {"temporal": alpha, "text": 0.0}
    if scope == "temporal_text":
        return {"temporal": alpha, "text": alpha}
    raise ValueError(f"unsupported task-vector scope: {scope}")


@torch.no_grad()
def apply_task_vector_target(
    model: nn.Module,
    base_file: Any,
    rag_file: Any,
    current: dict[str, float],
    target: dict[str, float],
) -> dict[str, Any]:
    state = model.state_dict()
    changed_tensors = 0
    changed_parameters = 0
    for key, parameter in state.items():
        component = task_vector_scope(key)
        if component is None:
            continue
        increment = target[component] - current[component]
        if increment == 0.0:
            continue
        if key not in base_file.keys() or key not in rag_file.keys():
            raise ValueError(f"task-vector source lacks required tensor: {key}")
        base_shape = tuple(base_file.get_slice(key).get_shape())
        rag_shape = tuple(rag_file.get_slice(key).get_shape())
        if tuple(parameter.shape) != base_shape or base_shape != rag_shape:
            raise ValueError(f"task-vector tensor shape mismatch: {key}")
        base = base_file.get_tensor(key).to(device=parameter.device, dtype=torch.float32)
        rag = rag_file.get_tensor(key).to(device=parameter.device, dtype=torch.float32)
        updated = parameter.float().add(rag - base, alpha=increment)
        parameter.copy_(updated.to(dtype=parameter.dtype))
        changed_tensors += 1
        changed_parameters += parameter.numel()
        del base, rag, updated
    current.update(target)
    return {
        "changedTensors": changed_tensors,
        "changedParameters": changed_parameters,
        "target": dict(target),
    }
