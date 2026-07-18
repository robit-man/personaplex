#!/usr/bin/env python3
"""Publish a prepared PersonaPlex controlled-duplex export to a private HF dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import hashlib

from huggingface_hub import HfApi


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("transition state is not an object")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hardlink_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                try:
                    os.link(path, target)
                except OSError:
                    shutil.copy2(path, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", default=os.environ.get("PERSONAPLEX_TRANSITION_STATE", "/srv/voxrn_cache/personaplex-transition/v7-p1000.state.json"))
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN", "").strip()
    repo_id = os.environ.get("PERSONAPLEX_HF_DATASET_REPO", "").strip()
    if not token or not repo_id:
        raise SystemExit("HF_TOKEN and PERSONAPLEX_HF_DATASET_REPO are required for publication")
    state_path = Path(args.state).resolve()
    state = read_json(state_path)
    if state.get("status") not in {"prepared", "training_started", "training_completed", "published"}:
        raise SystemExit("publication requires a prepared native tensor certificate")
    export_root = Path(state["exportRoot"]).resolve()
    export_manifest = read_json(export_root / "manifest.json")
    if int(export_manifest.get("admittedConversationCount", 0)) < int(state["targetConversations"]):
        raise SystemExit("export did not retain the required number of admitted conversations")
    publish_root = Path(state["outputRoot"]).resolve() / "07_huggingface_dataset"
    publish_root.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "examples.jsonl"):
        source = export_root / name
        destination = publish_root / name
        if destination.exists() and sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"existing publication staging file differs from export: {destination}")
        if not destination.exists():
            shutil.copy2(source, destination)
    hardlink_tree(export_root / "audio", publish_root / "audio")
    readme = publish_root / "README.md"
    if not readme.exists():
        readme.write_text(
        "---\n"
        "task_categories:\n- text-to-speech\n"
        "language:\n- en\n"
        "license: other\n"
        "---\n\n"
        "# PersonaPlex V7 semantic-control synthetic corpus\n\n"
        "Private, provenance-tracked synthetic duplex conversations for training a "
        "semantic-prefix adapter. Each example contains aligned duplex audio and a "
        "typed control frame available before the agent response. The control frame "
        "does not contain target wording. Audio, timing, ASR, provenance, control "
        "adherence, counterfactual branch, and native export checks were required "
        "before publication. This dataset is restricted to approved research and "
        "model-development use.\n",
            encoding="utf-8",
        )
    api = HfApi(token=token)
    private = os.environ.get("PERSONAPLEX_HF_PRIVATE", "1").strip() != "0"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(publish_root))
    publication = {
        "schema": "personaplex.huggingface-publication.v1", "publishedAt": datetime.now(timezone.utc).isoformat(),
        "repoId": repo_id, "private": private, "sourceExport": str(export_root),
    }
    write_json(Path(state["outputRoot"]) / "huggingface_publication.json", publication)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
