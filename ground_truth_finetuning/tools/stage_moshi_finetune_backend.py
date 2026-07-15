"""Stage a pinned Moshi-Finetune checkout with the caller-loss safety overlay."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

from ground_truth_finetuning.training.contracts import StreamLayout
from ground_truth_finetuning.tools.export_moshi_finetune_dataset import UPSTREAM_REVISION


def revision(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def hash_file(path: Path) -> str:
    digest = sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def write_agent_only_module(path: Path) -> None:
    path.write_text(
        """\"\"\"Mask caller depformer streams from PersonaPlex LoRA loss.\"\"\"

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def agent_only_audio_mask(model: object, audio_mask: torch.Tensor) -> torch.Tensor:
    layout_path = os.environ.get(\"PERSONAPLEX_STREAM_LAYOUT_PATH\")
    if not layout_path:
        raise RuntimeError(\"PERSONAPLEX_STREAM_LAYOUT_PATH is required\")
    layout = json.loads(Path(layout_path).read_text())
    agent = layout.get(\"agent_audio_stream_indices\")
    caller = layout.get(\"caller_audio_stream_indices\")
    if not isinstance(agent, list) or not isinstance(caller, list):
        raise RuntimeError(\"invalid explicit PersonaPlex stream layout\")
    offset = int(model.audio_offset)
    dep_q = int(model.dep_q)
    agent_dep = [index - offset for index in agent]
    caller_dep = [index - offset for index in caller]
    if sorted(agent_dep + caller_dep) != list(range(dep_q)):
        raise RuntimeError(\"stream layout does not exactly partition native depformer streams\")
    if audio_mask.ndim != 3 or audio_mask.shape[1] != dep_q:
        raise RuntimeError(\"unexpected native depformer mask shape\")
    masked = torch.zeros_like(audio_mask, dtype=torch.bool)
    masked[:, agent_dep, :] = audio_mask[:, agent_dep, :].bool()
    if masked[:, caller_dep, :].any():
        raise RuntimeError(\"caller stream became supervised\")
    return masked
"""
    )


def patch_file(path: Path) -> None:
    original = path.read_text()
    import_line = "from finetune.personaplex_agent_only import agent_only_audio_mask\n"
    if import_line not in original:
        marker = next(
            (
                candidate
                for candidate in (
                    "from finetune.loss import compute_loss_with_mask\n",
                    "from .loss import compute_loss_with_mask\n",
                )
                if candidate in original
            ),
            None,
        )
        if marker is None:
            raise RuntimeError(f"cannot find upstream loss import in {path}")
        original = original.replace(marker, marker + import_line, 1)
    target = "                output.mask,\n                mode=\"audio\","
    replacement = "                agent_only_audio_mask(model, output.mask),\n                mode=\"audio\","
    if target not in original and replacement not in original:
        raise RuntimeError(f"cannot find upstream audio loss mask in {path}")
    path.write_text(original.replace(target, replacement))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--stream-layout", type=Path, required=True)
    parser.add_argument("--expected-revision", default=UPSTREAM_REVISION)
    args = parser.parse_args()
    upstream = args.upstream_root.resolve()
    destination = args.destination.resolve()
    layout_path = args.stream_layout.resolve()
    layout = StreamLayout.from_mapping(json.loads(layout_path.read_text()))
    layout.validate_static()
    actual_revision = revision(upstream)
    if actual_revision != args.expected_revision:
        raise SystemExit(f"refusing unpinned upstream revision {actual_revision}; expected {args.expected_revision}")
    if destination.exists():
        raise SystemExit(f"destination already exists: {destination}")
    shutil.copytree(upstream, destination, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache"))
    write_agent_only_module(destination / "finetune" / "personaplex_agent_only.py")
    patch_file(destination / "train.py")
    patch_file(destination / "finetune" / "eval.py")
    report = {
        "schema_version": 1,
        "kind": "personaplex-moshi-finetune-overlay",
        "upstream_revision": actual_revision,
        "stream_layout_sha256": hash_file(layout_path),
        "patched_files": ["train.py", "finetune/eval.py", "finetune/personaplex_agent_only.py"],
        "caller_stream_supervision": "forbidden",
    }
    (destination / "PERSONAPLEX_OVERLAY.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
