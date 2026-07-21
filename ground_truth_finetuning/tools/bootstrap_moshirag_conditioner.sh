#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESOURCE_ROOT="${PERSONAPLEX_SHARED_CACHE_ROOT:-${VOXRN_SHARED_CACHE_ROOT:-${VOXRN_RESOURCE_ROOT:-/srv/voxrn_cache}}}"
NATIVE_PYTHON="${PERSONAPLEX_NATIVE_PYTHON:-/srv/personaplex_workspace/.venvs/personaplex-native/bin/python}"
VENV_ROOT="${PERSONAPLEX_MOSHIRAG_CONDITIONER_VENV:-${RESOURCE_ROOT}/personaplex/venvs/moshirag-conditioner}"
TOKENIZER_REPO="${PERSONAPLEX_MOSHIRAG_TOKENIZER_REPO:-unsloth/Llama-3.2-3B-Instruct}"
TOKENIZER_REVISION="${PERSONAPLEX_MOSHIRAG_TOKENIZER_REVISION:-main}"

test -x "${NATIVE_PYTHON}" || { printf 'native Python missing: %s\n' "${NATIVE_PYTHON}" >&2; exit 1; }
mkdir -p "${VENV_ROOT}" "${RESOURCE_ROOT}/huggingface"

if [[ ! -x "${VENV_ROOT}/bin/python" ]]; then
  "${NATIVE_PYTHON}" -m venv "${VENV_ROOT}"
  native_site="$(${NATIVE_PYTHON} -c 'import site; print(site.getsitepackages()[0])')"
  venv_site="$(${VENV_ROOT}/bin/python -c 'import site; print(site.getsitepackages()[0])')"
  printf '%s\n' "${native_site}" > "${venv_site}/personaplex-native.pth"
fi

"${VENV_ROOT}/bin/python" -m pip install --disable-pip-version-check --upgrade \
  'huggingface-hub>=0.34,<1' 'transformers==4.55.3' 'protobuf>=4.25' \
  'tokenizers>=0.21,<0.22'
"${VENV_ROOT}/bin/python" -m pip install --disable-pip-version-check --no-deps \
  'xformers==0.0.28.post1'

TOKENIZER_REPO="${TOKENIZER_REPO}" \
TOKENIZER_REVISION="${TOKENIZER_REVISION}" \
RESOURCE_ROOT="${RESOURCE_ROOT}" \
"${VENV_ROOT}/bin/python" - <<'PY'
import hashlib
import json
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerFast

repo = os.environ["TOKENIZER_REPO"]
requested = os.environ["TOKENIZER_REVISION"]
info = HfApi().model_info(repo, revision=requested)
revision = info.sha
target = Path(os.environ["RESOURCE_ROOT"]) / "huggingface" / repo / revision
target.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo,
    revision=revision,
    local_dir=str(target),
    local_dir_use_symlinks=False,
    allow_patterns=[
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
        "config.json",
    ],
)
wrapper = Path(os.environ["RESOURCE_ROOT"]) / "personaplex" / "tokenizers" / "llama-3.2-arc4" / revision
wrapper.mkdir(parents=True, exist_ok=True)
for source_file in target.iterdir():
    if source_file.is_file():
        shutil.copy2(source_file, wrapper / source_file.name)
tokenizer_config_path = wrapper / "tokenizer_config.json"
tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
# Several public mirrors incorrectly declare the abstract PreTrainedTokenizer.
# ARC-4 only consumes token ids, so bind the exact tokenizer.json to its concrete
# fast implementation and certify id parity before recording the derivative.
tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n", encoding="utf-8")
direct = PreTrainedTokenizerFast.from_pretrained(target)
wrapped = AutoTokenizer.from_pretrained(wrapper, use_fast=False)
probes = [
    "typed control frame",
    "replacement shipped July 14",
    "caller interrupted - revise",
    "warm but concise; do not invent a delivery date",
]
for probe in probes:
    expected = direct.encode(probe, add_special_tokens=False)
    actual = wrapped.encode(probe, add_special_tokens=False)
    if actual != expected:
        raise RuntimeError(f"ARC tokenizer wrapper changed token ids for {probe!r}")
tokenizer_sha256 = hashlib.sha256((target / "tokenizer.json").read_bytes()).hexdigest()
record = {
    "schema_version": 1,
    "repo_id": repo,
    "requested_revision": requested,
    "resolved_revision": revision,
    "source_path": str(target),
    "path": str(wrapper),
    "tokenizer_json_sha256": tokenizer_sha256,
    "implementation": type(wrapped).__name__,
    "certification_probe_count": len(probes),
    "token_id_parity": True,
}
record_path = Path(os.environ["RESOURCE_ROOT"]) / "personaplex" / "imports" / "moshirag-tokenizer.json"
record_path.parent.mkdir(parents=True, exist_ok=True)
record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(str(record_path))
PY

"${VENV_ROOT}/bin/python" - <<'PY'
import torch
import transformers
import xformers
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; refusing a CPU conditioner installation")
print({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "xformers": xformers.__version__,
    "cuda_devices": torch.cuda.device_count(),
})
PY

printf 'MoshiRAG conditioner environment ready at %s\n' "${VENV_ROOT}"
printf 'Repository root: %s\n' "${REPO_ROOT}"
