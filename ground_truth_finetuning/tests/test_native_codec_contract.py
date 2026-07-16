"""Regression coverage for pinned native codec/tokenizer provenance."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ground_truth_finetuning.tools.encode_native_adapter_tensors import (
    hash_file,
    load_contract,
    require_codec_artifacts,
)
from ground_truth_finetuning.tools.certify_controlled_native_corpus import split_coverage
from ground_truth_finetuning.training.contracts import canonical_json


def contract() -> dict[str, object]:
    delays = [0, 0, *([1] * 7), 0, *([1] * 7)]
    return {
        "schema_version": 3,
        "model_revision": "test-native-revision",
        "moshi_weights_sha256": "sha256:lm",
        "mimi_weights_sha256": "sha256:mimi",
        "tokenizer_sha256": "sha256:tokenizer",
        "num_codebooks": 17,
        "audio_offset": 1,
        "dep_q": 16,
        "delays": delays,
        "delay_config_sha256": "sha256:" + __import__("hashlib").sha256(
            canonical_json(delays).encode()
        ).hexdigest(),
        "text_padding_token_id": 3,
        "end_of_text_padding_id": 0,
        "zero_token_id": -1,
        "stream_layout": {
            "text_stream_indices": [0],
            "agent_audio_stream_indices": list(range(1, 9)),
            "caller_audio_stream_indices": list(range(9, 17)),
        },
    }


class NativeCodecContractTests(unittest.TestCase):
    def test_only_train_rows_are_not_adapter_training_coverage(self) -> None:
        counts, failures = split_coverage([{"split": "train"}, {"split": "train"}])
        self.assertEqual(counts, {"train": 2, "validation": 0, "test": 0})
        self.assertEqual(failures, ["missing required split coverage: validation, test"])

    def test_full_partition_passes_split_coverage(self) -> None:
        counts, failures = split_coverage(
            [{"split": "train"}, {"split": "validation"}, {"split": "test"}]
        )
        self.assertEqual(counts, {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(failures, [])

    def test_legacy_contract_without_codec_hashes_is_refused(self) -> None:
        value = contract()
        value.pop("mimi_weights_sha256")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mimi_weights_sha256"):
                load_contract(path)

    def test_codec_and_tokenizer_bytes_must_match_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mimi = root / "mimi.safetensors"
            tokenizer = root / "tokenizer.model"
            mimi.write_bytes(b"mimi-bytes")
            tokenizer.write_bytes(b"tokenizer-bytes")
            value = contract()
            value["mimi_weights_sha256"] = hash_file(mimi)
            value["tokenizer_sha256"] = hash_file(tokenizer)
            self.assertEqual(
                require_codec_artifacts(value, mimi, tokenizer),
                {
                    "mimi_weights_sha256": hash_file(mimi),
                    "tokenizer_sha256": hash_file(tokenizer),
                },
            )
            tokenizer.write_bytes(b"different-tokenizer")
            with self.assertRaisesRegex(ValueError, "SentencePiece tokenizer"):
                require_codec_artifacts(value, mimi, tokenizer)


if __name__ == "__main__":
    unittest.main()
