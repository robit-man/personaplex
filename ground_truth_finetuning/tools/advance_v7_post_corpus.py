#!/usr/bin/env python3
"""Gate the V7 corpus transition from synthesis to native training.

The coordinator is deliberately fail-closed: it snapshots only certified V7
records, verifies required provenance/audio/control fields, runs the existing
native tensor pipeline in prepare-only mode, then starts independent training
and publication services. Runtime settings come from environment variables so
the same unit can run on differently sized hosts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterator


GTFT_ROOT = Path(__file__).resolve().parents[1]
TOOLS = GTFT_ROOT / "tools"


def env_text(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def env_int(name: str, default: int) -> int:
    try:
        value = int(env_text(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def env_float(name: str, default: float) -> float:
    try:
        value = float(env_text(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def relative_audio_path(value: object) -> Path:
    path = Path(str(value or ""))
    if not str(path) or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe audio path: {value!r}")
    return path


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise ValueError(f"snapshot collision with different bytes: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def certified_sources(dataset_root: Path, namespace: str) -> list[Path]:
    default_pattern = f"personaplex-v7-paired-v8cf-{namespace}-*.certified.jsonl"
    patterns = [
        pattern.strip()
        for pattern in env_text(
            "PERSONAPLEX_CERTIFIED_ARTIFACT_PATTERNS", default_pattern
        ).split(",")
        if pattern.strip()
    ]
    if not patterns:
        raise ValueError("PERSONAPLEX_CERTIFIED_ARTIFACT_PATTERNS cannot be empty")
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_absolute() or "/" in pattern or ".." in candidate.parts:
            raise ValueError(f"unsafe certified artifact pattern: {pattern!r}")
    # Certifier workers create their destination before writing it.  A zero-byte
    # destination is therefore an in-progress artifact, not a completed corpus
    # member.  Ignore only that explicit transient state; nonempty artifacts are
    # still parsed and fail closed on malformed or uncertified content.
    return sorted({
        path
        for pattern in patterns
        for path in dataset_root.glob(f"gpu*/datasets/synthesize/{pattern}")
        if path.stat().st_size > 0
    })


def read_certified_source(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    records: list[dict[str, Any]] = []
    conversation_ids: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("schema") != "voxrn.synthetic-conversation.v4":
            raise ValueError(f"{path}:{number}: non-V4 record in certified corpus")
        conversation_id = str(record.get("conversationId") or "").strip()
        if not conversation_id:
            raise ValueError(f"{path}:{number}: missing conversationId")
        if not record.get("quality", {}).get("accepted"):
            raise ValueError(f"{path}:{number}: quality gate not accepted")
        audio = path.parent / relative_audio_path(record.get("audioPath"))
        if not audio.is_file():
            raise ValueError(f"{path}:{number}: missing audio {audio}")
        if record.get("speaker") == "target":
            replay_role = record.get("replay", {}).get("role")
            if replay_role != "shared_prefix_context_only":
                if not record.get("training", {}).get("eligible"):
                    raise ValueError(f"{path}:{number}: target label is not training eligible")
                if record.get("semanticAdherence", {}).get("verificationStatus") != "batch_certified":
                    raise ValueError(f"{path}:{number}: target semantic certificate missing")
                if not isinstance(record.get("control", {}).get("frame"), dict):
                    raise ValueError(f"{path}:{number}: target control frame missing")
        records.append(record)
        conversation_ids.add(conversation_id)
    if not records:
        raise ValueError(f"empty certified artifact: {path}")
    return records, conversation_ids


def snapshot_corpus(dataset_root: Path, sources: list[Path], work_root: Path, namespace: str) -> tuple[Path, dict[str, Any]]:
    snapshot_root = work_root / "snapshots" / f"v7-{namespace}-{utc_now()}"
    inputs_root = snapshot_root / "inputs"
    conversation_ids: set[str] = set()
    source_manifest: list[dict[str, Any]] = []
    record_count = 0
    for source in sources:
        records, ids = read_certified_source(source)
        relative_source = source.relative_to(dataset_root)
        destination = inputs_root / relative_source
        hardlink_or_copy(source, destination)
        for record in records:
            relative_audio = relative_audio_path(record.get("audioPath"))
            hardlink_or_copy(source.parent / relative_audio, destination.parent / relative_audio)
            timeline_reference = record.get("duplexTimelinePath")
            if timeline_reference:
                relative_timeline = relative_audio_path(timeline_reference)
                hardlink_or_copy(
                    source.parent / relative_timeline,
                    destination.parent / relative_timeline,
                )
        conversation_ids.update(ids)
        record_count += len(records)
        source_manifest.append({
            "path": str(relative_source), "sha256": sha256_file(source),
            "records": len(records), "conversationIds": len(ids),
        })
    audit = {
        "schema": "personaplex.v7-post-corpus-audit.v1",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "namespace": namespace,
        "sourceArtifacts": source_manifest,
        "certifiedConversationCount": len(conversation_ids),
        "certifiedRecordCount": record_count,
        "inputRoot": str(inputs_root),
    }
    write_json(snapshot_root / "corpus_audit.json", audit)
    return snapshot_root, audit


def configured_path(name: str) -> Path:
    value = env_text(name)
    if not value:
        raise ValueError(f"required runtime setting is missing: {name}")
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"configured artifact does not exist: {name}={path}")
    return path


def run(command: list[str]) -> None:
    completed = subprocess.run(command)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def allowed_gpus() -> list[int]:
    values = [value.strip() for value in env_text("PERSONAPLEX_ALLOWED_GPUS", "0,1,2").split(",") if value.strip()]
    gpus = [int(value) for value in values]
    if not gpus or any(gpu < 0 for gpu in gpus):
        raise ValueError("PERSONAPLEX_ALLOWED_GPUS must contain non-negative GPU indices")
    return gpus


def stop_synthesis() -> None:
    if env_text("PERSONAPLEX_STOP_SYNTHESIS_ON_READY", "1") != "1":
        return
    units: list[str] = []
    for lane in range(3):
        units.extend([
            f"personaplex-v7-lane@{lane}.service",
            f"personaplex-v7-certifier@{lane}.service",
            f"personaplex-v7-voicebox-worker@{lane}.service",
            f"personaplex-ornith-chatml-proxy-worker@{lane}.service",
            f"personaplex-ornith-worker@{lane}.service",
        ])
    run(["systemctl", "--user", "stop", *units])


def reusable_prepared_attempt(work_root: Path, namespace: str) -> tuple[Path, Path, dict[str, Any], str] | None:
    """Find a complete pre-codec stage that has not yet started tensor encoding.

    A failed source-contract or GPU-admission check must not force a new duplex
    export. Reuse is intentionally narrow: the original immutable snapshot and
    nonempty pre-codec manifest must both exist, while the tensor output must
    not exist at all. Any partially written tensor directory is fail-closed and
    requires a fresh pipeline attempt.
    """
    prepared_root = work_root / "prepared"
    if not prepared_root.is_dir():
        return None
    for output_root in sorted(prepared_root.glob(f"v7-{namespace}-*"), key=lambda path: path.stat().st_mtime, reverse=True):
        snapshot_root = work_root / "snapshots" / output_root.name
        audit_path = snapshot_root / "corpus_audit.json"
        precodec_manifest = output_root / "02_precodec" / "precodec_manifest.jsonl"
        tensor_root = output_root / "03_native_tensors"
        certificate = output_root / "04_certificate" / "controlled_native_certificate.json"
        encoded_manifest = tensor_root / "encoded_examples.jsonl"
        if not audit_path.is_file() or not precodec_manifest.is_file() or precodec_manifest.stat().st_size == 0:
            continue
        audit = read_json(audit_path)
        if audit.get("namespace") != namespace or not isinstance(audit.get("certifiedConversationCount"), int):
            continue
        if certificate.is_file():
            status = read_json(certificate).get("status")
            if status == "certified_for_adapter_training":
                return snapshot_root, output_root, audit, "certified"
            if encoded_manifest.is_file() and encoded_manifest.stat().st_size > 0:
                return snapshot_root, output_root, audit, "certify"
            continue
        if encoded_manifest.is_file() and encoded_manifest.stat().st_size > 0:
            return snapshot_root, output_root, audit, "certify"
        return snapshot_root, output_root, audit, "encode"
    return None


def certify_native_tensors(output_root: Path) -> Path:
    precodec_root = output_root / "02_precodec"
    artifact_root = output_root / "03_native_tensors"
    certificate = output_root / "04_certificate" / "controlled_native_certificate.json"
    run([
        sys.executable, str(TOOLS / "certify_controlled_native_corpus.py"),
        "--manifest", str(artifact_root / "encoded_examples.jsonl"),
        "--artifact-root", str(artifact_root), "--precodec-root", str(precodec_root),
        "--certificate", str(certificate),
    ])
    return certificate


def encode_and_certify_precodec(output_root: Path, source_root: Path, mimi_path: Path, tokenizer_path: Path, contract: Path, device: str) -> Path:
    precodec_root = output_root / "02_precodec"
    artifact_root = output_root / "03_native_tensors"
    certificate = output_root / "04_certificate" / "controlled_native_certificate.json"
    if artifact_root.exists():
        quarantined = output_root / f"03_native_tensors.incomplete-{utc_now()}"
        if quarantined.exists():
            raise RuntimeError(f"partial tensor quarantine collision: {quarantined}")
        artifact_root.replace(quarantined)
    run([
        sys.executable, str(TOOLS / "encode_controlled_native_adapter_tensors.py"),
        "--manifest", str(precodec_root / "precodec_manifest.jsonl"),
        "--precodec-root", str(precodec_root), "--artifact-root", str(artifact_root),
        "--moshi-source-root", str(source_root), "--mimi-path", str(mimi_path),
        "--tokenizer-path", str(tokenizer_path), "--model-contract", str(contract),
        "--device", device,
    ])
    return certify_native_tensors(output_root)


@contextmanager
def coordinator_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def finalise(args: argparse.Namespace) -> int:
    resource_root = Path(env_text("VOXRN_RESOURCE_ROOT", "/srv/voxrn_cache")).resolve()
    dataset_root = Path(env_text("PERSONAPLEX_SYNTHESIS_ROOT", str(resource_root / "personaplex-lanes"))).resolve()
    namespace = env_text("PERSONAPLEX_CORPUS_NAMESPACE", "p1000v5")
    target = env_int("PERSONAPLEX_TARGET_CONVERSATIONS", 1000)
    work_root = Path(env_text("PERSONAPLEX_TRANSITION_ROOT", str(resource_root / "personaplex-transition"))).resolve()
    state_path = Path(env_text("PERSONAPLEX_TRANSITION_STATE", str(work_root / f"v7-{namespace}.state.json"))).resolve()
    existing = read_json(state_path)
    if existing.get("status") in {"prepared", "training_started", "training_completed", "published"}:
        return 0
    sources = certified_sources(dataset_root, namespace)
    conversations: set[str] = set()
    for source in sources:
        _, ids = read_certified_source(source)
        conversations.update(ids)
    if len(conversations) < target:
        write_json(state_path, {
            "schema": "personaplex.v7-transition-state.v1", "status": "waiting_for_corpus",
            "checkedAt": datetime.now(timezone.utc).isoformat(), "targetConversations": target,
            "certifiedConversationCount": len(conversations), "namespace": namespace,
        })
        return 0
    reusable = reusable_prepared_attempt(work_root, namespace)
    if reusable:
        snapshot_root, output_root, audit, resume_stage = reusable
    else:
        snapshot_root, audit = snapshot_corpus(dataset_root, sources, work_root, namespace)
        output_root = work_root / "prepared" / snapshot_root.name
        resume_stage = "pipeline"
    stop_synthesis()
    contract = configured_path("PERSONAPLEX_MODEL_CONTRACT")
    source_root = configured_path("PERSONAPLEX_MOSHI_SOURCE_ROOT")
    mimi_path = configured_path("PERSONAPLEX_MIMI_PATH")
    tokenizer_path = configured_path("PERSONAPLEX_TOKENIZER_PATH")
    encode_device = env_text("PERSONAPLEX_ENCODE_DEVICE", f"cuda:{allowed_gpus()[0]}")
    if resume_stage == "certified":
        certificate = output_root / "04_certificate" / "controlled_native_certificate.json"
    elif resume_stage == "certify":
        certificate = certify_native_tensors(output_root)
    elif reusable:
        certificate = encode_and_certify_precodec(
            output_root, source_root, mimi_path, tokenizer_path, contract, encode_device
        )
    else:
        command = [
            sys.executable, str(TOOLS / "run_controlled_native_pipeline.py"),
            "--voryn-input", str(snapshot_root / "inputs"), "--output-root", str(output_root),
            "--moshi-source-root", str(source_root), "--moshi-path", str(configured_path("PERSONAPLEX_MOSHI_PATH")),
            "--mimi-path", str(mimi_path), "--tokenizer-path", str(tokenizer_path),
            "--model-contract", str(contract), "--world-size", str(len(allowed_gpus())),
            "--encode-device", encode_device, "--prepare-only",
        ]
        for gpu in allowed_gpus():
            command.extend(["--allow-gpu", str(gpu)])
        run(command)
        certificate = output_root / "04_certificate" / "controlled_native_certificate.json"
    certificate_data = read_json(certificate)
    if certificate_data.get("status") != "certified_for_adapter_training":
        raise RuntimeError("native tensor certificate did not authorize adapter training")
    state = {
        "schema": "personaplex.v7-transition-state.v1", "status": "prepared",
        "preparedAt": datetime.now(timezone.utc).isoformat(), "namespace": namespace,
        "targetConversations": target, "audit": audit, "snapshotRoot": str(snapshot_root),
        "outputRoot": str(output_root), "exportRoot": str(output_root / "01_export"),
        "encodedManifest": str(output_root / "03_native_tensors" / "encoded_examples.jsonl"),
        "artifactRoot": str(output_root / "03_native_tensors"), "certificate": str(certificate),
        "modelContract": str(contract), "moshiSourceRoot": str(source_root),
        "moshiPath": str(configured_path("PERSONAPLEX_MOSHI_PATH")),
        "tokenizerPath": str(configured_path("PERSONAPLEX_TOKENIZER_PATH")),
        "allowedGpus": allowed_gpus(),
    }
    write_json(state_path, state)
    run(["systemctl", "--user", "start", "--no-block", "personaplex-v7-train.service", "personaplex-v7-publish.service"])
    return 0


def train(args: argparse.Namespace) -> int:
    state = read_json(Path(env_text("PERSONAPLEX_TRANSITION_STATE", args.state)).resolve())
    if state.get("status") not in {"prepared", "training_started", "published"}:
        raise RuntimeError("training cannot start before a prepared tensor certificate exists")
    # The native launcher refuses to reuse a run root.  Give every service retry
    # an immutable attempt directory so a transient GPU-admission refusal cannot
    # make a later retry look successful.  systemd serializes a running service,
    # so an executing train run is never duplicated by this code path.
    run_root = Path(state["outputRoot"]) / "06_training_execution" / f"attempt-{utc_now()}"
    moshi_path = Path(state["moshiPath"])
    model_gib = moshi_path.stat().st_size / (1024 ** 3)
    # This is a model-runtime multiplier, not a host-specific memory constant.
    # The conservative default reflects measured native Moshi adapter residency;
    # installations can calibrate it through the environment contract.
    model_headroom = env_float("PERSONAPLEX_TRAIN_MODEL_HEADROOM_RATIO", 1.50)
    dynamic_min_free_gib = model_gib * model_headroom
    requested_world_size = min(
        len(state["allowedGpus"]),
        env_int("PERSONAPLEX_TRAIN_MAX_WORLD_SIZE", 1),
    )
    if requested_world_size < 1:
        raise RuntimeError("PERSONAPLEX_TRAIN_MAX_WORLD_SIZE must admit at least one configured GPU")
    command = [
        sys.executable, str(TOOLS / "launch_semantic_prefix.py"),
        "--manifest", state["encodedManifest"], "--artifact-root", state["artifactRoot"],
        "--certificate", state["certificate"], "--model-contract", state["modelContract"],
        "--moshi-source-root", state["moshiSourceRoot"], "--moshi-path", str(moshi_path),
        "--tokenizer-path", state["tokenizerPath"], "--run-root", str(run_root),
        "--world-size", str(requested_world_size),
        "--min-world-size", str(min(requested_world_size, env_int("PERSONAPLEX_TRAIN_MIN_WORLD_SIZE", 1))),
        "--max-steps", str(env_int("PERSONAPLEX_TRAIN_MAX_STEPS", 12000)),
        "--checkpoint-every", str(env_int("PERSONAPLEX_TRAIN_CHECKPOINT_EVERY", 500)),
        "--eval-examples", str(env_int("PERSONAPLEX_TRAIN_EVAL_EXAMPLES", 256)),
        "--min-free-gib", str(env_float("PERSONAPLEX_TRAIN_MIN_FREE_GIB", dynamic_min_free_gib)),
        "--reserve-ratio", str(env_float("PERSONAPLEX_TRAIN_GPU_RESERVE_RATIO", 0.10)),
        "--max-utilization-pct", str(env_int("PERSONAPLEX_TRAIN_MAX_GPU_UTILIZATION_PCT", 85)),
        "--execute",
    ]
    for gpu in state["allowedGpus"]:
        command.extend(["--allow-gpu", str(gpu)])
    run(command)
    state["status"] = "training_completed"
    state["trainingRunRoot"] = str(run_root)
    state["trainingCompletedAt"] = datetime.now(timezone.utc).isoformat()
    write_json(Path(env_text("PERSONAPLEX_TRANSITION_STATE", args.state)).resolve(), state)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("finalise", "train"), required=True)
    parser.add_argument("--state", default=env_text("PERSONAPLEX_TRANSITION_STATE", "/srv/voxrn_cache/personaplex-transition/v7-p1000.state.json"))
    args = parser.parse_args()
    lock_path = Path(args.state).resolve().with_suffix(".lock")
    try:
        with coordinator_lock(lock_path):
            return finalise(args) if args.phase == "finalise" else train(args)
    except BlockingIOError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
