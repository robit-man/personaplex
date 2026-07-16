"""Deterministic identity checks for the local Moshi source used by native training."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


SOURCE_FINGERPRINT_SCHEMA = "personaplex-moshi-source-fingerprint.v1"


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def moshi_source_fingerprint(source_root: Path) -> dict[str, Any]:
    """Return a stable source identity without depending on a Git checkout."""
    root = source_root.resolve()
    package = root / "moshi"
    if not (package / "__init__.py").is_file():
        raise ValueError(f"Moshi package is missing below {root}")
    files = []
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files.append({"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path)})
    if not files:
        raise ValueError(f"Moshi source contains no Python files below {root}")
    payload = json.dumps(
        {"schema": SOURCE_FINGERPRINT_SCHEMA, "files": files},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema": SOURCE_FINGERPRINT_SCHEMA,
        "sha256": f"sha256:{sha256(payload).hexdigest()}",
        "file_count": len(files),
    }


def require_moshi_source_contract(source_root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Fail closed when local source differs from the inspected model contract."""
    expected_schema = contract.get("moshi_source_fingerprint_schema")
    expected_hash = contract.get("moshi_source_sha256")
    expected_count = contract.get("moshi_source_file_count")
    if not isinstance(expected_schema, str) or not isinstance(expected_hash, str) or not isinstance(expected_count, int):
        raise ValueError("native model contract lacks a Moshi source fingerprint")
    actual = moshi_source_fingerprint(source_root)
    if (
        actual["schema"] != expected_schema
        or actual["sha256"] != expected_hash
        or actual["file_count"] != expected_count
    ):
        raise ValueError(
            "local Moshi source does not match the inspected native model contract "
            f"(expected {expected_hash}, got {actual['sha256']})"
        )
    return actual
