"""Direct packed-NF4 execution for PersonaPlex on CUDA.

The Hugging Face checkpoint stores two signed 4-bit values per byte and one
scale for each 64-value group.  This module keeps those buffers packed on the
GPU and invokes a CUDA kernel for linear and embedding lookups.  It never
materializes a BF16 checkpoint or offers a CPU execution fallback.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from safetensors import safe_open
from torch import nn
from torch.utils.cpp_extension import load


_KERNEL: Any | None = None
_GROUP_SIZE = 64


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_directory() -> Path:
    path = Path(os.environ.get("PERSONAPLEX_NF4_BUILD_DIR", _repo_root() / ".cache/personaplex/nf4-kernel"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_kernel() -> Any:
    """Build/load the Jetson CUDA kernel from the persistent local cache."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    if not torch.cuda.is_available():
        raise RuntimeError("PersonaPlex direct NF4 requires CUDA; CPU fallback is disabled")

    os.environ.setdefault("MAX_JOBS", "1")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")
    source = _repo_root() / "personaplex_nf4/csrc/nf4_linear.cu"
    if not source.is_file():
        raise RuntimeError(f"direct NF4 CUDA source is missing: {source}")
    _KERNEL = load(
        name="personaplex_nf4_cuda",
        sources=[str(source)],
        build_directory=str(_build_directory()),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=os.environ.get("PERSONAPLEX_NF4_BUILD_VERBOSE", "0") == "1",
    )
    return _KERNEL


def verify_nf4_checkpoint(filename: str | Path) -> None:
    """Fail early unless this is the expected packed-NF4 safetensors layout."""
    path = Path(filename)
    if not path.is_file():
        raise RuntimeError(f"NF4 checkpoint does not exist: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
    scales = [key for key in keys if key.endswith(".__scales__")]
    if not scales:
        raise RuntimeError(f"{path} is not a supported packed-NF4 PersonaPlex checkpoint")
    for scale_key in scales:
        base = scale_key[: -len(".__scales__")]
        expected = (base, f"{base}.__shape__", f"{base}.__numel__")
        if not all(key in keys for key in expected):
            raise RuntimeError(f"incomplete NF4 tensor metadata for {base}")


def _is_nf4_checkpoint(filename: str | Path | None) -> bool:
    if filename is None:
        return False
    try:
        with safe_open(str(filename), framework="pt", device="cpu") as checkpoint:
            return any(key.endswith(".__scales__") for key in checkpoint.keys())
    except Exception:
        return False


def _requested_dtype() -> torch.dtype:
    value = os.environ.get("PERSONAPLEX_NF4_DTYPE", "fp16").strip().lower()
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise RuntimeError("PERSONAPLEX_NF4_DTYPE must be fp16 or bf16")


class NF4Matrix(nn.Module):
    """A packed two-dimensional NF4 tensor with CUDA-only matmul methods."""

    def __init__(self, rows: int, columns: int) -> None:
        super().__init__()
        self.rows = rows
        self.columns = columns
        self.register_buffer("packed_weight", None, persistent=False)
        self.register_buffer("scales", None, persistent=False)

    def load_(self, packed_weight: torch.Tensor, scales: torch.Tensor) -> None:
        if packed_weight.dtype != torch.uint8:
            raise RuntimeError("NF4 packed weights must be uint8")
        if packed_weight.numel() * 2 < self.rows * self.columns:
            raise RuntimeError("NF4 packed tensor is shorter than its declared matrix shape")
        if scales.numel() < (self.rows * self.columns + _GROUP_SIZE - 1) // _GROUP_SIZE:
            raise RuntimeError("NF4 scale tensor is shorter than its declared matrix shape")
        self.packed_weight = packed_weight.contiguous()
        self.scales = scales.contiguous()

    def forward_rows(self, values: torch.Tensor, row_offset: int = 0, row_count: int | None = None) -> torch.Tensor:
        if self.packed_weight is None or self.scales is None:
            raise RuntimeError("NF4 matrix was used before packed weights were loaded")
        if not values.is_cuda:
            raise RuntimeError("PersonaPlex direct NF4 only accepts CUDA activations")
        if values.shape[-1] != self.columns:
            raise RuntimeError(f"NF4 matrix expected {self.columns} input features, got {values.shape[-1]}")
        count = self.rows - row_offset if row_count is None else row_count
        output = build_kernel().nf4_linear(
            values.contiguous().reshape(-1, self.columns),
            self.packed_weight,
            self.scales,
            self.rows,
            self.columns,
            row_offset,
            count,
        )
        return output.reshape(*values.shape[:-1], count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward_rows(values)


class NF4Linear(NF4Matrix):
    def __init__(self, rows: int, columns: int, bias: bool, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__(rows, columns)
        if bias:
            self.bias = nn.Parameter(torch.empty(rows, device=device, dtype=dtype), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward_rows(self, values: torch.Tensor, row_offset: int = 0, row_count: int | None = None) -> torch.Tensor:
        output = super().forward_rows(values, row_offset, row_count)
        if self.bias is not None:
            count = self.rows - row_offset if row_count is None else row_count
            output = output + self.bias[row_offset : row_offset + count]
        return output


class NF4Embedding(NF4Matrix):
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self.packed_weight is None or self.scales is None:
            raise RuntimeError("NF4 embedding was used before packed weights were loaded")
        if not indices.is_cuda:
            raise RuntimeError("PersonaPlex direct NF4 only accepts CUDA token indices")
        values = indices.contiguous().reshape(-1).to(dtype=torch.long)
        if values.numel() and (int(values.min()) < 0 or int(values.max()) >= self.rows):
            raise RuntimeError("NF4 embedding index is out of range")
        output = build_kernel().nf4_embedding(values, self.packed_weight, self.scales, self.rows, self.columns)
        return output.reshape(*indices.shape, self.columns)


def _resolve_module(root: nn.Module, dotted_path: str) -> tuple[nn.Module, str, nn.Module]:
    parts = dotted_path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    return parent, leaf, parent[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)


def _replace_quantized_weight(model: nn.Module, name: str, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> NF4Matrix:
    module_path, attribute = name.rsplit(".", 1)
    parent, leaf, module = _resolve_module(model, module_path)
    if attribute == "weight" and isinstance(module, nn.Linear):
        if len(shape) != 2:
            raise RuntimeError(f"NF4 linear {name} has invalid shape {shape}")
        replacement = NF4Linear(shape[0], shape[1], module.bias is not None, device=device, dtype=dtype)
        setattr(parent, leaf, replacement)
        return replacement
    if attribute == "weight" and isinstance(module, nn.Embedding):
        if len(shape) != 2:
            raise RuntimeError(f"NF4 embedding {name} has invalid shape {shape}")
        replacement = NF4Embedding(shape[0], shape[1])
        setattr(parent, leaf, replacement)
        return replacement
    if attribute == "in_proj_weight" and module.__class__.__name__ == "StreamingMultiheadAttention":
        if len(shape) != 2:
            raise RuntimeError(f"NF4 attention projection {name} has invalid shape {shape}")
        replacement = NF4Matrix(shape[0], shape[1])
        module.in_proj_weight = nn.Parameter(torch.empty(0, device=device, dtype=dtype), requires_grad=False)
        module.add_module("nf4_in_proj", replacement)
        module.forward = MethodType(_nf4_streaming_attention_forward, module)
        return replacement
    raise RuntimeError(f"direct NF4 runtime cannot replace non-linear tensor {name} ({type(module).__name__}.{attribute})")


def _nf4_multi_linear(matrix: NF4Matrix, values: torch.Tensor, weights_per_step: int, offset: int) -> torch.Tensor:
    rows_per_step = matrix.rows // weights_per_step
    if matrix.rows % weights_per_step:
        raise RuntimeError("NF4 multi-linear weight rows are not divisible by weights_per_step")
    if offset + values.shape[1] > weights_per_step:
        raise RuntimeError("NF4 multi-linear step offset exceeds its configured context")
    return torch.cat(
        [matrix.forward_rows(values[:, step : step + 1], (offset + step) * rows_per_step, rows_per_step) for step in range(values.shape[1])],
        dim=1,
    )


def _nf4_streaming_attention_forward(self: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    from einops import rearrange
    import torch.nn.functional as functional

    state = self._streaming_state
    steps = query.shape[1]
    if state is None:
        offset = torch.zeros(1, device=query.device, dtype=torch.long)
        offset_cpu = 0
    else:
        assert self.causal, "Streaming only available for causal"
        offset = state.offset
        offset_cpu = state.offset_cpu

    if self.weights_per_step:
        projected = _nf4_multi_linear(self.nf4_in_proj, query, self.weights_per_step, offset_cpu)
    else:
        projected = self.nf4_in_proj(query)
    q, k, v = rearrange(projected, "b t (p h d) -> p b h t d", p=3, h=self.num_heads)
    if self.rope:
        q, k = self.rope(q, k, offset, time_before_heads=False)
    k, v, pos_k = self._complete_kv(k, v)
    if self.causal:
        pos_k = pos_k.view(1, -1)
        pos_q = offset + torch.arange(steps, device=q.device, dtype=torch.long).view(-1, 1)
        delta = pos_q - pos_k
        attention_bias = (pos_k >= 0) & (delta >= 0)
        if self.context is not None:
            attention_bias = attention_bias & (delta < self.context)
    else:
        attention_bias = None
    output = functional.scaled_dot_product_attention(q, k, v, attention_bias, dropout_p=0.0)
    output = rearrange(output, "b h t d -> b t (h d)")
    if self.weights_per_step:
        output = _nf4_multi_linear(self.out_proj, output, self.weights_per_step, offset_cpu)
    else:
        output = self.out_proj(output)
    if state is not None:
        state.offset.add_(steps)
        state.offset_cpu += steps
    return output


def _load_nf4_moshi_lm(
    filename: str | Path,
    copy_missing_weights: bool,
    device: torch.device | str,
    dtype: torch.dtype,
    delays: Any = None,
    cpu_offload: bool = False,
) -> nn.Module:
    if cpu_offload:
        raise RuntimeError("PERSONAPLEX direct NF4 does not permit CPU offload")
    if not torch.cuda.is_available():
        raise RuntimeError("PersonaPlex direct NF4 requires CUDA; CPU fallback is disabled")

    from moshi.models import loaders
    from moshi.models.lm import LMModel

    target = torch.device(device)
    if target.type != "cuda":
        raise RuntimeError(f"PersonaPlex direct NF4 requires a CUDA device, got {target}")
    model_kwargs = dict(loaders._lm_kwargs)
    model_kwargs["dep_q"] = 16
    if delays is not None:
        model_kwargs["delays"] = delays
    model = LMModel(device="meta", dtype=dtype, **model_kwargs)

    device_index = target.index if target.index is not None else 0
    with safe_open(str(filename), framework="pt", device=device_index) as checkpoint:
        keys = set(checkpoint.keys())
        quantized = sorted(key[: -len(".__scales__")] for key in keys if key.endswith(".__scales__"))
        holders: dict[str, NF4Matrix] = {}
        for name in quantized:
            metadata = (f"{name}.__shape__", f"{name}.__numel__")
            if name not in keys or not all(item in keys for item in metadata):
                raise RuntimeError(f"incomplete NF4 metadata for {name}")
            shape = tuple(int(value) for value in checkpoint.get_tensor(metadata[0]).detach().cpu().tolist() if value > 0)
            holders[name] = _replace_quantized_weight(model, name, shape, target, dtype)

        native_state: dict[str, torch.Tensor] = {}
        for name in keys:
            if name in holders or ".__" in name:
                continue
            tensor = checkpoint.get_tensor(name)
            if tensor.is_floating_point():
                tensor = tensor.to(device=target, dtype=dtype)
            else:
                tensor = tensor.to(device=target)
            native_state[name] = tensor
        model.load_state_dict(native_state, strict=False, assign=True)

        for name, holder in holders.items():
            holder.load_(
                checkpoint.get_tensor(name).to(device=target, dtype=torch.uint8),
                checkpoint.get_tensor(f"{name}.__scales__").to(device=target, dtype=dtype),
            )

    unresolved = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if unresolved:
        raise RuntimeError(f"NF4 checkpoint left meta parameters unresolved: {', '.join(unresolved[:8])}")
    model.eval()
    return model


def install_direct_nf4_loader() -> None:
    """Patch the vendored loader before ``moshi.server`` imports and uses it."""
    from moshi.models import loaders

    if getattr(loaders, "_personaplex_direct_nf4", False):
        return
    native_loader = loaders.get_moshi_lm

    def direct_loader(
        filename: str | Path | None,
        copy_missing_weights: bool = True,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        delays: Any = None,
        cpu_offload: bool = False,
    ) -> nn.Module:
        if _is_nf4_checkpoint(filename):
            verify_nf4_checkpoint(filename)
            return _load_nf4_moshi_lm(
                filename,
                copy_missing_weights,
                device,
                _requested_dtype(),
                delays,
                cpu_offload,
            )
        return native_loader(filename, copy_missing_weights, device, dtype, delays, cpu_offload)

    loaders.get_moshi_lm = direct_loader
    loaders._personaplex_direct_nf4 = True

