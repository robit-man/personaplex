"""Controlled PersonaPlex server for upstream revision 3428dfd95309.

The server is intentionally a narrow audio-plane endpoint.  It preserves
PersonaPlex's Opus binary transport and adds two binary message kinds:

* ``0x04``: UTF-8 typed control JSON (``control.update``, ``evidence.update``,
  ``control.boundary``, or ``control.barge_in``).
* ``0x05``: UTF-8 control acknowledgement/event JSON.

It is not a prompt gateway.  Expressive control is admitted only through a
validated ``ControlTrainingFrame`` and an installed semantic-prefix adapter.
"""

from __future__ import annotations

import argparse
import asyncio
from hashlib import sha256
import json
from pathlib import Path
import random
import time
from typing import Any
import uuid

from aiohttp import WSMsgType, web
import numpy as np
import sentencepiece
import sphn
import torch

from moshi.models import LMGen, loaders

from ground_truth_finetuning.training.semantic_prefix import SemanticPrefixAdapter
from ground_truth_finetuning.training.evidence_conditioning import EvidenceStreamAdapter
from personaplex_control.moshirag_reference import Arc4ReferenceProvider
from ground_truth_finetuning.training.control_stream import (
    ControlStreamConfig,
    SemanticControlStreamAdapter,
)

from .runtime import (
    CONTROL_MESSAGE_IN,
    CONTROL_MESSAGE_OUT,
    ControlAck,
    ControlProtocolError,
    RuntimeControlSession,
    RuntimeControlUpdate,
    RuntimeEvidenceUpdate,
    EvidenceStreamProvider,
    SemanticControlStreamProvider,
    SemanticPrefixProvider,
)


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _system_prompt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if text.startswith("<system>") and text.endswith("<system>"):
        return text
    return f"<system> {text} <system>"


def _wire_error(reason: str) -> bytes:
    return json.dumps(
        {"type": "control.error", "protocolVersion": 2, "reason": reason},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class ControlledServerState:
    def __init__(
        self,
        *,
        mimi: Any,
        other_mimi: Any,
        tokenizer: Any,
        lm: Any,
        adapter: torch.nn.Module | None,
        adapter_version: str,
        conditioning_mode: str,
        evidence_adapter: EvidenceStreamAdapter | None,
        evidence_adapter_version: str | None,
        moshirag_conditioner_url: str | None,
        moshirag_arc4_adapter_checkpoint: Path | None,
        moshirag_primary_control: bool,
        model_revision: str,
        moshirag_conditioner_timeout_seconds: float,
        moshirag_max_reference_frames: int,
        device: torch.device,
        voice_prompt_dir: Path,
        prefill_deadline_ms: float,
        allow_uncontrolled_audio: bool,
    ) -> None:
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.tokenizer = tokenizer
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.prefill_deadline_ms = prefill_deadline_ms
        self.allow_uncontrolled_audio = allow_uncontrolled_audio
        self.frame_size = int(mimi.sample_rate / mimi.frame_rate)
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
            sample_rate=mimi.sample_rate,
            device=device,
            frame_rate=mimi.frame_rate,
        )
        if conditioning_mode == "arc4_primary_temporal_stream_v4_field_slots":
            if moshirag_conditioner_url is None or moshirag_arc4_adapter_checkpoint is None:
                raise ValueError("ARC-4 primary control requires conditioner and checkpoint")
            self.provider = Arc4ReferenceProvider(
                lm_gen=self.lm_gen,
                conditioner_url=moshirag_conditioner_url,
                adapter_checkpoint=moshirag_arc4_adapter_checkpoint,
                expected_model_revision=model_revision,
                expected_control_adapter_sha256=None,
                primary_control=True,
                timeout_seconds=moshirag_conditioner_timeout_seconds,
                max_reference_frames=moshirag_max_reference_frames,
            )
        elif conditioning_mode == "temporal_stream_v4":
            if adapter is None:
                raise ValueError("temporal_stream_v4 requires an adapter")
            self.provider = SemanticControlStreamProvider(
                lm_gen=self.lm_gen,
                adapter=adapter.eval(),
                tokenizer=tokenizer,
                adapter_version=adapter_version,
                max_control_tokens=int(adapter.config.max_tokens),
            )
        elif conditioning_mode == "virtual_prefix_v3":
            if adapter is None:
                raise ValueError("virtual_prefix_v3 requires an adapter")
            self.provider = SemanticPrefixProvider(
                lm_gen=self.lm_gen,
                adapter=adapter.eval(),
                tokenizer=tokenizer,
                adapter_version=adapter_version,
            )
        else:
            raise ValueError(f"unsupported conditioning mode: {conditioning_mode}")
        if moshirag_conditioner_url is not None and not moshirag_primary_control:
            if moshirag_arc4_adapter_checkpoint is None:
                raise ValueError("trained ARC-4 adapter checkpoint is required")
            self.evidence_provider = Arc4ReferenceProvider(
                lm_gen=self.lm_gen,
                conditioner_url=moshirag_conditioner_url,
                adapter_checkpoint=moshirag_arc4_adapter_checkpoint,
                expected_model_revision=model_revision,
                expected_control_adapter_sha256=adapter_version,
                timeout_seconds=moshirag_conditioner_timeout_seconds,
                max_reference_frames=moshirag_max_reference_frames,
            )
        else:
            self.evidence_provider = (
                EvidenceStreamProvider(
                lm_gen=self.lm_gen,
                adapter=evidence_adapter.eval(),
                tokenizer=tokenizer,
                adapter_version=evidence_adapter_version or "unknown",
            )
            if evidence_adapter is not None
            else None
            )
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def warmup(self) -> None:
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            self.other_mimi.encode(chunk)
            for index in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, index : index + 1])
                if tokens is not None:
                    self.mimi.decode(tokens[:, 1:9])
                    self.other_mimi.decode(tokens[:, 1:9])
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _voice_path(self, requested: str) -> Path:
        candidate = Path(requested)
        if candidate.name != requested:
            raise ValueError("voice_prompt must be a plain filename")
        path = self.voice_prompt_dir / candidate
        if not path.is_file():
            raise FileNotFoundError(f"voice prompt does not exist: {requested}")
        return path

    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        call_id = request.query.get("call_id") or f"personaplex:{uuid.uuid4()}"
        voice_name = request.query.get("voice_prompt", "")
        if not voice_name:
            await ws.send_bytes(bytes([CONTROL_MESSAGE_OUT]) + _wire_error("voice_prompt_required"))
            await ws.close()
            return ws
        try:
            voice_path = self._voice_path(voice_name)
        except (ValueError, FileNotFoundError) as exc:
            await ws.send_bytes(bytes([CONTROL_MESSAGE_OUT]) + _wire_error(str(exc)))
            await ws.close()
            return ws
        text_prompt = request.query.get("text_prompt", "")
        seed = int(request.query.get("seed", "-1"))
        async with self.lock:
            if seed >= 0:
                _seed(seed)
            if self.lm_gen.voice_prompt != str(voice_path):
                if voice_path.suffix == ".pt":
                    self.lm_gen.load_voice_prompt_embeddings(str(voice_path))
                else:
                    self.lm_gen.load_voice_prompt(str(voice_path))
            self.lm_gen.text_prompt_tokens = self.tokenizer.encode(_system_prompt(text_prompt)) if text_prompt else None
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            session = RuntimeControlSession(
                call_id=call_id,
                prefix_provider=self.provider,
                evidence_provider=self.evidence_provider,
                allow_uncontrolled_audio=self.allow_uncontrolled_audio,
                prefill_deadline_ms=self.prefill_deadline_ms,
            )
            reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            outbound: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(maxsize=128)
            closing = False

            async def send_ack(ack: ControlAck) -> None:
                await ws.send_bytes(bytes([CONTROL_MESSAGE_OUT]) + ack.to_wire())

            def queue_media(generation_id: int, payload: bytes) -> None:
                try:
                    outbound.put_nowait((generation_id, payload))
                except asyncio.QueueFull:
                    # Bounded egress is intentional.  Late media is worse than a
                    # dropped packet because it can speak stale state after a barge-in.
                    try:
                        outbound.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    try:
                        outbound.put_nowait((generation_id, payload))
                    except asyncio.QueueFull:
                        pass

            async def handle_control(payload: bytes) -> None:
                try:
                    message = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    await ws.send_bytes(bytes([CONTROL_MESSAGE_OUT]) + _wire_error("invalid_control_json"))
                    return
                if not isinstance(message, dict):
                    await ws.send_bytes(bytes([CONTROL_MESSAGE_OUT]) + _wire_error("control_message_must_be_object"))
                    return
                kind = message.get("type")
                try:
                    if kind == "control.update":
                        update = RuntimeControlUpdate.from_mapping(message)
                        for ack in session.submit(update):
                            await send_ack(ack)
                    elif kind == "evidence.update":
                        update = RuntimeEvidenceUpdate.from_mapping(message)
                        for ack in session.submit_evidence(update):
                            await send_ack(ack)
                    elif kind == "control.boundary":
                        boundary_call_id = message.get("callId")
                        boundary_turn_id = message.get("turnId")
                        boundary_hash = message.get("contextHash")
                        if not isinstance(boundary_call_id, str) or not isinstance(boundary_turn_id, int) or not isinstance(boundary_hash, str):
                            raise ControlProtocolError("control.boundary requires callId, non-negative turnId, and contextHash")
                        await send_ack(session.apply_boundary(call_id=boundary_call_id, turn_id=boundary_turn_id, context_hash=boundary_hash))
                    elif kind == "control.barge_in":
                        for ack in session.caller_barge_in(reason="caller_barge_in"):
                            await send_ack(ack)
                    else:
                        raise ControlProtocolError("unsupported_control_message_type")
                except ControlProtocolError as exc:
                    await ws.send_bytes(bytes([CONTROL_MESSAGE_OUT]) + _wire_error(str(exc)))

            async def recv_loop() -> None:
                nonlocal closing
                try:
                    async for message in ws:
                        if message.type in {WSMsgType.CLOSED, WSMsgType.CLOSE, WSMsgType.ERROR}:
                            return
                        if message.type != WSMsgType.BINARY or not isinstance(message.data, bytes) or not message.data:
                            continue
                        kind, payload = message.data[0], message.data[1:]
                        if kind == 0x01:
                            reader.append_bytes(payload)
                        elif kind == CONTROL_MESSAGE_IN:
                            await handle_control(payload)
                finally:
                    closing = True

            async def opus_loop() -> None:
                all_pcm: np.ndarray[Any, Any] | None = None
                while not closing:
                    await asyncio.sleep(0.001)
                    pcm = reader.read_pcm()
                    if pcm.shape[-1] == 0:
                        continue
                    all_pcm = pcm if all_pcm is None else np.concatenate((all_pcm, pcm))
                    while all_pcm.shape[-1] >= self.frame_size:
                        chunk, all_pcm = all_pcm[: self.frame_size], all_pcm[self.frame_size :]
                        codes = self.mimi.encode(torch.from_numpy(chunk).to(device=self.device)[None, None])
                        self.other_mimi.encode(torch.from_numpy(chunk).to(device=self.device)[None, None])
                        for index in range(codes.shape[-1]):
                            tokens = self.lm_gen.step(codes[:, :, index : index + 1])
                            if tokens is None:
                                continue
                            generation_id = session.generation_id
                            if not session.may_emit(generation_id):
                                continue
                            main_pcm = self.mimi.decode(tokens[:, 1:9]).cpu()
                            writer.append_pcm(main_pcm[0, 0].numpy())
                            encoded = writer.read_bytes()
                            if encoded:
                                queue_media(generation_id, bytes([0x01]) + encoded)
                            text_token = tokens[0, 0, 0].item()
                            if text_token not in (0, 3):
                                text = self.tokenizer.id_to_piece(text_token).replace("▁", " ")
                                queue_media(generation_id, bytes([0x02]) + text.encode("utf-8"))

            async def send_loop() -> None:
                while not closing:
                    try:
                        generation_id, payload = await asyncio.wait_for(outbound.get(), timeout=0.05)
                    except asyncio.TimeoutError:
                        continue
                    if session.may_emit(generation_id):
                        await ws.send_bytes(payload)

            async def is_alive() -> bool:
                return not closing and not ws.closed

            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            await ws.send_bytes(b"\x00")
            tasks = [asyncio.create_task(recv_loop()), asyncio.create_task(opus_loop()), asyncio.create_task(send_loop())]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        await ws.close()
        return ws


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint must be an object: {path}")
    return payload


def _load_adapter(path: Path, lm: Any, device: torch.device, prefix_frames: int) -> tuple[SemanticPrefixAdapter, str, dict[str, Any]]:
    payload = _load_checkpoint(path)
    state = payload.get("adapter_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("adapter checkpoint must contain adapter_state_dict")
    adapter = SemanticPrefixAdapter(
        text_cardinality=int(lm.text_card),
        hidden_size=int(lm.dim),
        prefix_frames=prefix_frames,
    ).to(device=device, dtype=next(lm.parameters()).dtype)
    adapter.load_state_dict(state, strict=True)
    return adapter.eval(), _sha256(path), payload


def _load_control_stream_adapter(
    path: Path, lm: Any, device: torch.device
) -> tuple[SemanticControlStreamAdapter, str, dict[str, Any]]:
    payload = _load_checkpoint(path)
    state = payload.get("adapter_state_dict")
    raw_config = payload.get("adapter_config")
    if payload.get("schema_version") != 4 or not isinstance(state, dict) or not isinstance(raw_config, dict):
        raise ValueError("v4 checkpoint must contain schema_version, adapter_state_dict, and adapter_config")
    config = ControlStreamConfig.from_mapping(raw_config)
    adapter = SemanticControlStreamAdapter(
        lm_hidden_size=int(lm.dim), config=config
    ).to(device=device, dtype=torch.float32)
    adapter.load_state_dict(state, strict=True)
    return adapter.eval(), _sha256(path), payload


def _load_evidence_adapter(path: Path, lm: Any, device: torch.device, stream_frames: int) -> tuple[EvidenceStreamAdapter, str, dict[str, Any]]:
    payload = _load_checkpoint(path)
    state = payload.get("evidence_adapter_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("evidence adapter checkpoint must contain evidence_adapter_state_dict")
    adapter = EvidenceStreamAdapter(
        text_cardinality=int(lm.text_card),
        hidden_size=int(lm.dim),
        stream_frames=stream_frames,
    ).to(device=device, dtype=next(lm.parameters()).dtype)
    adapter.load_state_dict(state, strict=True)
    return adapter.eval(), _sha256(path), payload


def _load_model_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read native model contract: {exc}") from exc
    if not isinstance(contract, dict) or contract.get("kind") != "personaplex-native-model-contract":
        raise ValueError("model contract must be a personaplex-native-model-contract object")
    for field in ("model_revision", "moshi_weights_sha256", "moshi_source_sha256"):
        if not isinstance(contract.get(field), str) or not contract[field]:
            raise ValueError(f"model contract is missing {field}")
    return contract


def _validate_evidence_compatibility(
    *,
    evidence_payload: dict[str, Any],
    control_payload: dict[str, Any],
    control_adapter_sha256: str,
    model_contract: dict[str, Any],
    moshi_weight: Path,
    evidence_stream_frames: int,
) -> None:
    if evidence_payload.get("control_adapter_checkpoint_sha256") != control_adapter_sha256:
        raise ValueError("evidence checkpoint was not trained against the installed semantic adapter")
    expected_revision = model_contract["model_revision"]
    if control_payload.get("model_revision") != expected_revision:
        raise ValueError("semantic adapter model revision does not match the installed native model contract")
    if evidence_payload.get("model_revision") != expected_revision:
        raise ValueError("evidence adapter model revision does not match the installed native model contract")
    if evidence_payload.get("evidence_stream_frames") != evidence_stream_frames:
        raise ValueError("evidence checkpoint stream-frame count does not match the server configuration")
    if _sha256(moshi_weight) != model_contract["moshi_weights_sha256"]:
        raise ValueError("installed Moshi weights do not match the native model contract")


def _validate_control_stream_compatibility(
    *,
    payload: dict[str, Any],
    model_contract: dict[str, Any],
    moshi_weight: Path,
) -> None:
    if payload.get("model_revision") != model_contract["model_revision"]:
        raise ValueError("v4 control checkpoint model revision does not match the native model contract")
    if _sha256(moshi_weight) != model_contract["moshi_weights_sha256"]:
        raise ValueError("installed Moshi weights do not match the native model contract")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--moshi-weight", required=True, type=Path)
    parser.add_argument("--mimi-weight", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    adapter_group = parser.add_mutually_exclusive_group(required=False)
    adapter_group.add_argument("--control-stream-checkpoint", type=Path)
    adapter_group.add_argument("--adapter-checkpoint", type=Path)
    parser.add_argument("--evidence-adapter-checkpoint", type=Path)
    parser.add_argument("--moshirag-conditioner-url")
    parser.add_argument("--moshirag-arc4-adapter-checkpoint", type=Path)
    parser.add_argument("--moshirag-primary-control", action="store_true")
    parser.add_argument("--moshirag-conditioner-timeout-seconds", type=float, default=2.0)
    parser.add_argument("--moshirag-max-reference-frames", type=int, default=96)
    parser.add_argument("--model-contract", type=Path, help="required when installing a trained evidence adapter")
    parser.add_argument("--voice-prompt-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prefix-frames", type=int, default=16)
    parser.add_argument("--evidence-stream-frames", type=int, default=16)
    parser.add_argument("--prefill-deadline-ms", type=float, default=120.0)
    parser.add_argument("--allow-uncontrolled-audio", action="store_true")
    args = parser.parse_args()
    if not str(args.device).startswith("cuda"):
        raise SystemExit("controlled PersonaPlex server requires CUDA")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    selected_adapter_path = args.control_stream_checkpoint or args.adapter_checkpoint
    required_paths = [args.moshi_weight, args.mimi_weight, args.tokenizer, args.voice_prompt_dir]
    if selected_adapter_path is not None:
        required_paths.append(selected_adapter_path)
    if args.moshirag_primary_control and selected_adapter_path is not None:
        raise SystemExit("ARC-4 primary control must not load a legacy control adapter")
    if not args.moshirag_primary_control and selected_adapter_path is None:
        raise SystemExit("a legacy adapter is required unless --moshirag-primary-control is selected")
    if args.evidence_adapter_checkpoint is not None and args.moshirag_conditioner_url is not None:
        raise SystemExit("choose either the provisional evidence adapter or upstream ARC-4, not both")
    if args.control_stream_checkpoint is not None and (
        args.evidence_adapter_checkpoint is not None or args.moshirag_conditioner_url is not None
    ):
        raise SystemExit("v4 control-stream checkpoints already unify evidence; a separate evidence path is invalid")
    if (args.moshirag_conditioner_url is None) != (args.moshirag_arc4_adapter_checkpoint is None):
        raise SystemExit("--moshirag-conditioner-url and --moshirag-arc4-adapter-checkpoint are required together")
    if args.control_stream_checkpoint is not None and args.model_contract is None:
        raise SystemExit("--model-contract is required with --control-stream-checkpoint")
    if args.control_stream_checkpoint is not None:
        required_paths.append(args.model_contract)
    if args.evidence_adapter_checkpoint is not None:
        required_paths.append(args.evidence_adapter_checkpoint)
        if args.model_contract is None:
            raise SystemExit("--model-contract is required with --evidence-adapter-checkpoint")
        required_paths.append(args.model_contract)
    if args.moshirag_conditioner_url is not None and args.model_contract is None:
        raise SystemExit("--model-contract is required with --moshirag-conditioner-url")
    if args.moshirag_arc4_adapter_checkpoint is not None:
        required_paths.append(args.moshirag_arc4_adapter_checkpoint)
    for path in required_paths:
        if not path.exists():
            raise SystemExit(f"required path does not exist: {path}")
    device = torch.device(args.device)
    _seed(42424242)
    mimi = loaders.get_mimi(args.mimi_weight, device)
    other_mimi = loaders.get_mimi(args.mimi_weight, device)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=device)
    lm.eval()
    tokenizer = sentencepiece.SentencePieceProcessor(str(args.tokenizer))
    adapter = None
    adapter_version = ""
    control_payload = None
    if args.moshirag_primary_control:
        model_contract = _load_model_contract(args.model_contract)
        if _sha256(args.moshi_weight) != model_contract["moshi_weights_sha256"]:
            raise SystemExit("ARC-4 primary PersonaPlex weights do not match model contract")
        conditioning_mode = "arc4_primary_temporal_stream_v4_field_slots"
    elif args.control_stream_checkpoint is not None:
        adapter, adapter_version, control_payload = _load_control_stream_adapter(
            args.control_stream_checkpoint, lm, device
        )
        _validate_control_stream_compatibility(
            payload=control_payload,
            model_contract=_load_model_contract(args.model_contract),
            moshi_weight=args.moshi_weight,
        )
        conditioning_mode = "temporal_stream_v4"
    else:
        adapter, adapter_version, control_payload = _load_adapter(
            args.adapter_checkpoint, lm, device, args.prefix_frames
        )
        conditioning_mode = "virtual_prefix_v3"
    evidence_adapter: EvidenceStreamAdapter | None = None
    evidence_adapter_version: str | None = None
    if args.evidence_adapter_checkpoint is not None:
        evidence_adapter, evidence_adapter_version, evidence_payload = _load_evidence_adapter(
            args.evidence_adapter_checkpoint,
            lm,
            device,
            args.evidence_stream_frames,
        )
        _validate_evidence_compatibility(
            evidence_payload=evidence_payload,
            control_payload=control_payload,
            control_adapter_sha256=adapter_version,
            model_contract=_load_model_contract(args.model_contract),
            moshi_weight=args.moshi_weight,
            evidence_stream_frames=args.evidence_stream_frames,
        )
    model_contract = _load_model_contract(args.model_contract) if args.model_contract is not None else None
    state = ControlledServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        tokenizer=tokenizer,
        lm=lm,
        adapter=adapter,
        adapter_version=adapter_version,
        conditioning_mode=conditioning_mode,
        evidence_adapter=evidence_adapter,
        evidence_adapter_version=evidence_adapter_version,
        moshirag_conditioner_url=args.moshirag_conditioner_url,
        moshirag_arc4_adapter_checkpoint=args.moshirag_arc4_adapter_checkpoint,
        moshirag_primary_control=args.moshirag_primary_control,
        model_revision=(model_contract or {}).get("model_revision", ""),
        moshirag_conditioner_timeout_seconds=args.moshirag_conditioner_timeout_seconds,
        moshirag_max_reference_frames=args.moshirag_max_reference_frames,
        device=device,
        voice_prompt_dir=args.voice_prompt_dir,
        prefill_deadline_ms=args.prefill_deadline_ms,
        allow_uncontrolled_audio=args.allow_uncontrolled_audio,
    )
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
    if args.moshirag_primary_control and args.moshirag_conditioner_url is None:
        raise SystemExit("--moshirag-primary-control requires the ARC-4 conditioner and checkpoint")
    if args.moshirag_primary_control and (args.evidence_adapter_checkpoint or args.control_stream_checkpoint):
        raise SystemExit("ARC-4 primary control already unifies evidence and cannot be combined")
