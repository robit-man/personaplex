#!/usr/bin/env python3
"""CUDA-only ARC-4 reference conditioner for PersonaPlex semantic control.

This service ports the conditioning path released by kyutai-labs/moshi-rag
without loading the 15 GiB MoshiRAG language model into host memory.  ARC-4
weights are loaded normally; only the two reference-conditioner tensors are
read from the MoshiRAG checkpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from personaplex_control.arc4_packing import (
    ARC4_FIELD_SLOTS_PACKING_REVISION,
    ARC4_FIELD_ORDER,
    ARC4_GLOBAL_FIRST_PACKING_REVISION,
    ARC4_PACKING_REVISION,
    ARC4_SUPPORTED_PACKING_REVISIONS,
    field_frame_allocation,
    pack_arc4_stream,
)
from personaplex_control.moshirag_reference import render_arc4_reference_envelope

LOG = logging.getLogger("personaplex.moshirag_conditioner")


def _insert_upstream_source(source_root: Path) -> None:
    package_root = source_root / "moshi"
    if not (package_root / "moshi" / "conditioners" / "arc_encoder.py").is_file():
        raise FileNotFoundError(
            f"pinned MoshiRAG source is incomplete: {package_root}"
        )
    sys.path.insert(0, str(package_root))


def _find_named_mapping(value: Any, name: str) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        candidate = value.get(name)
        if isinstance(candidate, Mapping):
            return candidate
        for child in value.values():
            found = _find_named_mapping(child, name)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_named_mapping(child, name)
            if found is not None:
                return found
    return None


def _strip_best_prefix(
    state: Mapping[str, Any], module_keys: set[str], prefixes: Iterable[str]
) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for prefix in prefixes:
        mapped = {
            key[len(prefix) :] if key.startswith(prefix) else key: value
            for key, value in state.items()
        }
        overlap = {key: value for key, value in mapped.items() if key in module_keys}
        if len(overlap) > len(best):
            best = overlap
    return best


def _select_reference_state(checkpoint: Path, device: str) -> dict[str, Any]:
    from safetensors import safe_open

    prefix = "condition_provider.conditioners.reference_with_time."
    selected: dict[str, Any] = {}
    with safe_open(str(checkpoint), framework="pt", device=device) as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                selected[key[len(prefix) :]] = handle.get_tensor(key)
    required = {"learnt_padding", "output_proj.weight"}
    missing = required.difference(selected)
    if missing:
        raise RuntimeError(
            "MoshiRAG checkpoint lacks reference conditioner tensors: "
            + ", ".join(sorted(missing))
        )
    return selected


def _build_conditioner(args: argparse.Namespace):
    import torch
    from safetensors.torch import load_file

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU inference fallback is forbidden")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError(f"CUDA device required, received {args.device!r}")
    torch.cuda.set_device(device)

    _insert_upstream_source(args.source_root)
    from moshi.conditioners.arc_encoder import MultiArcEncoderConditioner

    config = json.loads(args.rag_config.read_text(encoding="utf-8"))
    raw = _find_named_mapping(config, "multi_arc_encoder")
    if raw is None:
        raise RuntimeError("multi_arc_encoder configuration was not found")
    conditioner_args = dict(raw)
    conditioner_args["tokenizer_name"] = str(args.tokenizer)
    conditioner_args["hf_repo"] = None

    output_dim = int(config.get("dim", args.output_dim))
    conditioner = MultiArcEncoderConditioner(
        output_dim=output_dim,
        device=str(device),
        **conditioner_args,
    )
    module_keys = set(conditioner.state_dict())

    LOG.info("loading ARC-4 weights directly onto %s", device)
    arc_raw = load_file(str(args.arc_model), device=str(device))
    arc_state = _strip_best_prefix(
        arc_raw,
        module_keys,
        (
            "",
            "module.",
            "conditioner.",
            "reference_with_time.",
            "condition_provider.conditioners.reference_with_time.",
        ),
    )
    del arc_raw
    arc_state.update(_select_reference_state(args.rag_model, str(device)))

    covered = module_keys.intersection(arc_state)
    coverage = len(covered) / max(1, len(module_keys))
    if coverage < args.minimum_key_coverage:
        missing = sorted(module_keys.difference(covered))
        raise RuntimeError(
            f"conditioner checkpoint coverage {coverage:.3f} is below "
            f"{args.minimum_key_coverage:.3f}; first missing keys: {missing[:12]}"
        )
    incompatible = conditioner.load_state_dict(arc_state, strict=False, assign=True)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"unexpected conditioner keys: {incompatible.unexpected_keys[:12]}"
        )
    conditioner.eval()
    del arc_state
    torch.cuda.empty_cache()
    return conditioner, output_dim, coverage


class ConditionerService:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch

        self.args = args
        self.torch = torch
        self.conditioner, self.output_dim, self.key_coverage = _build_conditioner(args)
        self.lock = asyncio.Lock()
        identity = "\n".join(
            (
                args.release_revision,
                str(args.arc_model.resolve()),
                str(args.rag_model.resolve()),
                str(args.tokenizer.resolve()),
                args.packing_revision,
            )
        )
        self.revision = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]

    async def health(self, _request):
        from aiohttp import web

        return web.json_response(
            {
                "status": "ready",
                "backend": "kyutai-arc4",
                "release_revision": self.args.release_revision,
                "conditioner_revision": self.revision,
                "packing_revision": self.args.packing_revision,
                "field_order": list(ARC4_FIELD_ORDER),
                "device": self.args.device,
                "physical_cuda_device": self.args.physical_cuda_device,
                "output_dim": self.output_dim,
                "frame_rate_hz": self.args.frame_rate,
                "key_coverage": self.key_coverage,
                "cpu_fallback": False,
            }
        )

    async def spec(self, _request):
        from aiohttp import web

        return web.json_response(
            {
                "protocol": "personaplex.moshirag-conditioner.v1",
                "request": {"content_type": "application/json", "field": "fields", "field_order": list(ARC4_FIELD_ORDER)},
                "response": {
                    "content_type": "application/x-safetensors",
                    "tensor": "tensor",
                    "shape": [1, "reference_frames", self.output_dim],
                    "cadence_hz": self.args.frame_rate,
                },
                "semantics": "ARC-4 compressed reference stream; target text forbidden",
                "packing_revision": self.args.packing_revision,
                "field_allocation_at_96_frames": (
                    dict(zip(ARC4_FIELD_ORDER, field_frame_allocation(96)))
                    if self.args.packing_revision == ARC4_FIELD_SLOTS_PACKING_REVISION
                    else None
                ),
            }
        )

    async def embed(self, request):
        from aiohttp import web
        from safetensors.torch import save

        try:
            payload = await request.json()
        except Exception as exc:
            raise web.HTTPBadRequest(text=f"invalid JSON: {exc}") from exc
        fields = payload.get("fields") if isinstance(payload, dict) else None
        if not isinstance(fields, list) or len(fields) != len(ARC4_FIELD_ORDER):
            raise web.HTTPBadRequest(text="fields must contain the complete versioned field set")
        names = tuple(item.get("name") for item in fields if isinstance(item, dict))
        texts = tuple(str(item.get("text", "")).strip() for item in fields if isinstance(item, dict))
        if names != ARC4_FIELD_ORDER or len(texts) != len(ARC4_FIELD_ORDER) or any(not text for text in texts):
            raise web.HTTPBadRequest(text="fields must be non-empty and in the versioned order")
        text_size = sum(len(text) for text in texts)
        if text_size > self.args.max_text_chars:
            raise web.HTTPRequestEntityTooLarge(
                max_size=self.args.max_text_chars,
                actual_size=text_size,
            )
        requested_frames = payload.get("max_frames")
        packing = payload.get("packing")
        if packing != self.args.packing_revision:
            raise web.HTTPBadRequest(text=f"packing must be {self.args.packing_revision}")
        reference = payload.get("reference")
        expected_reference = render_arc4_reference_envelope(dict(zip(names, texts)))
        if reference != expected_reference:
            raise web.HTTPBadRequest(text="reference envelope does not match declared fields")
        if requested_frames is not None:
            if not isinstance(requested_frames, int) or isinstance(requested_frames, bool):
                raise web.HTTPBadRequest(text="max_frames must be an integer")
            if not 1 <= requested_frames <= self.args.max_response_frames:
                raise web.HTTPBadRequest(
                    text=f"max_frames must be between 1 and {self.args.max_response_frames}"
                )
            if self.args.packing_revision == ARC4_FIELD_SLOTS_PACKING_REVISION:
                try:
                    field_frame_allocation(requested_frames)
                except ValueError as exc:
                    raise web.HTTPBadRequest(text=str(exc)) from exc

        async with self.lock:
            with self.torch.inference_mode():
                if requested_frames is None:
                    raise web.HTTPBadRequest(text="bounded encoding requires max_frames")
                if self.args.packing_revision == ARC4_FIELD_SLOTS_PACKING_REVISION:
                    allocations = field_frame_allocation(requested_frames)
                    field_streams = []
                    for text, field_frames in zip(texts, allocations):
                        prepared = self.conditioner.prepare([text])
                        field_encoded, field_mask = self.conditioner(prepared)
                        field_encoded = field_encoded * field_mask.unsqueeze(-1).to(field_encoded.dtype)
                        field_streams.append(
                            pack_arc4_stream(field_encoded, field_mask, field_frames)
                        )
                    encoded = self.torch.cat(field_streams, dim=1).contiguous()
                elif self.args.packing_revision == ARC4_GLOBAL_FIRST_PACKING_REVISION:
                    prepared = self.conditioner.prepare([reference])
                    global_encoded, global_mask = self.conditioner(prepared)
                    global_encoded = global_encoded * global_mask.unsqueeze(-1).to(global_encoded.dtype)
                    encoded = pack_arc4_stream(global_encoded, global_mask, requested_frames)
                else:
                    raise web.HTTPInternalServerError(text="unsupported configured packing")
                encoded = encoded.detach().to(device="cpu").contiguous()
        if encoded.ndim != 3 or encoded.shape[0] != 1:
            raise web.HTTPInternalServerError(
                text=f"unexpected conditioner shape {tuple(encoded.shape)}"
            )
        if encoded.shape[-1] != self.output_dim or not self.torch.isfinite(encoded).all():
            raise web.HTTPInternalServerError(text="conditioner returned invalid values")
        body = save({"tensor": encoded})
        return web.Response(
            body=body,
            content_type="application/x-safetensors",
            headers={
                "X-PersonaPlex-Conditioner-Revision": self.revision,
                "X-PersonaPlex-Packing-Revision": self.args.packing_revision,
                "X-PersonaPlex-Reference-Frames": str(encoded.shape[1]),
            },
        )


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"path does not exist: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=_path, required=True)
    parser.add_argument("--rag-config", type=_path, required=True)
    parser.add_argument("--rag-model", type=_path, required=True)
    parser.add_argument("--arc-model", type=_path, required=True)
    parser.add_argument("--tokenizer", type=_path, required=True)
    parser.add_argument("--device", required=True, help="Visible CUDA device, e.g. cuda:0")
    parser.add_argument("--physical-cuda-device", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dim", type=int, default=4096)
    parser.add_argument("--frame-rate", type=float, default=12.5)
    parser.add_argument("--max-text-chars", type=int, default=32768)
    parser.add_argument("--max-request-bytes", type=int, default=131072)
    parser.add_argument("--max-response-frames", type=int, default=256)
    parser.add_argument(
        "--packing-revision",
        choices=ARC4_SUPPORTED_PACKING_REVISIONS,
        default=ARC4_PACKING_REVISION,
    )
    parser.add_argument("--minimum-key-coverage", type=float, default=0.95)
    parser.add_argument("--release-revision", required=True)
    return parser.parse_args()


def main() -> None:
    from aiohttp import web

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args()
    service = ConditionerService(args)
    app = web.Application(client_max_size=args.max_request_bytes)
    app.router.add_get("/health", service.health)
    app.router.add_get("/spec", service.spec)
    app.router.add_post("/embed", service.embed)
    web.run_app(app, host=args.host, port=args.port, access_log=LOG)


if __name__ == "__main__":
    main()
