#!/usr/bin/env python3
"""Download and verify the pinned MoshiRAG release without embedding credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "semantic_control_v4"
    / "moshirag_release.v1.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def configured_resource_root(value: str | None) -> Path:
    configured = (
        value
        or os.environ.get("PERSONAPLEX_SHARED_CACHE_ROOT")
        or os.environ.get("VOXRN_SHARED_CACHE_ROOT")
        or os.environ.get("VOXRN_RESOURCE_ROOT")
        or "/srv/voxrn_cache"
    )
    return Path(configured).expanduser().resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def artifact_dir(resource_root: Path, artifact: dict[str, Any]) -> Path:
    return (
        resource_root
        / str(artifact["local_subdir"])
        / str(artifact["revision"])
    )


def download_artifact(
    artifact: dict[str, Any],
    destination: Path,
    max_workers: int,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for --download"
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(artifact["repo_id"]),
        revision=str(artifact["revision"]),
        local_dir=str(destination),
        allow_patterns=list(artifact["allow_patterns"]),
        max_workers=max_workers,
    )


def verify_artifact(
    name: str,
    artifact: dict[str, Any],
    destination: Path,
    quick: bool,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for expected in artifact["files"]:
        path = destination / str(expected["path"])
        check: dict[str, Any] = {
            "path": str(path),
            "expected_size": int(expected["size"]),
            "expected_sha256": str(expected["sha256"]),
        }
        if not path.is_file():
            check["status"] = "missing"
            checks.append(check)
            continue

        actual_size = path.stat().st_size
        check["actual_size"] = actual_size
        if actual_size != int(expected["size"]):
            check["status"] = "size_mismatch"
            checks.append(check)
            continue

        if quick:
            check["status"] = "size_verified"
        else:
            actual_sha256 = sha256_file(path)
            check["actual_sha256"] = actual_sha256
            check["status"] = (
                "verified"
                if actual_sha256 == str(expected["sha256"])
                else "sha256_mismatch"
            )
        checks.append(check)

    accepted_statuses = {"verified"} if not quick else {"size_verified"}
    complete = all(check["status"] in accepted_statuses for check in checks)
    return {
        "name": name,
        "repo_id": artifact["repo_id"],
        "revision": artifact["revision"],
        "role": artifact["role"],
        "required": bool(artifact["required"]),
        "destination": str(destination),
        "complete": complete,
        "verification_mode": "size" if quick else "sha256",
        "files": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import and verify exact MoshiRAG release artifacts."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--resource-root")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Import only a named manifest artifact; repeat as needed.",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional artifacts such as the Candle runtime checkpoint.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download pinned files before verifying them.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Verify file sizes only. Full SHA-256 verification is the default.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts: dict[str, dict[str, Any]] = manifest["artifacts"]

    unknown = sorted(set(args.artifact) - set(artifacts))
    if unknown:
        raise ValueError(f"Unknown artifact names: {', '.join(unknown)}")

    if args.artifact:
        selected_names = list(dict.fromkeys(args.artifact))
    else:
        selected_names = [
            name
            for name, artifact in artifacts.items()
            if bool(artifact["required"]) or args.include_optional
        ]

    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")

    resource_root = configured_resource_root(args.resource_root)
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else resource_root
        / "personaplex"
        / "imports"
        / "moshirag-release-v1.report.json"
    )

    results: list[dict[str, Any]] = []
    for name in selected_names:
        artifact = artifacts[name]
        destination = artifact_dir(resource_root, artifact)
        if args.download:
            download_artifact(artifact, destination, args.max_workers)
        result = verify_artifact(name, artifact, destination, args.quick)
        results.append(result)
        print(
            json.dumps(
                {
                    "artifact": name,
                    "complete": result["complete"],
                    "destination": result["destination"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    required_failures = [
        result["name"]
        for result in results
        if result["required"] and not result["complete"]
    ]
    selected_failures = [
        result["name"] for result in results if not result["complete"]
    ]
    payload = {
        "schema": "personaplex.moshirag-import-report.v1",
        "created_at": utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "resource_root": str(resource_root),
        "download_requested": bool(args.download),
        "verification_mode": "size" if args.quick else "sha256",
        "source_commit": manifest["source_commit"],
        "artifacts": results,
        "required_failures": required_failures,
        "selected_failures": selected_failures,
        "complete": not selected_failures,
    }
    atomic_json(report_path, payload)
    print(json.dumps({"report": str(report_path), "complete": payload["complete"]}))
    return 0 if not selected_failures else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
