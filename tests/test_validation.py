"""Tests for the prepared-checkpoint decoder and validator.

The validator exists because a comment-only edit to model.toml re-derived the
package digest, orphaned an already-prepared record, and only surfaced as a
typed exit inside a guarded GPU window. The case that matters most is therefore
`test_catches_the_orphaned_record`.
"""

import hashlib
import unittest
from dataclasses import replace

from tatara.artifact_manifest import parse_manifest
from tatara.prepared_checkpoint import (
    MAGIC,
    PreparedCheckpoint,
    PreparedCheckpointError,
    PreparedCheckpointIdentity,
    PreparedShard,
    PreparedTensor,
    TensorDataType,
    decode_prepared_checkpoint,
    encode_prepared_checkpoint,
)
from tatara.validation import (
    CHECK_ARTIFACT_ID,
    CHECK_FILE_COUNT,
    CHECK_MANIFEST_DIGEST,
    CHECK_PACKAGE_DIGEST,
    CHECK_SHARD_DIGEST,
    CHECK_TENSOR_BOUNDS,
    validate_prepared_checkpoint,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
PACKAGE_SHA = "c" * 64
MANIFEST_SHA = "d" * 64

MANIFEST_TEXT = f"""
schema_version = 1

[artifact]
id = "qwen-test"
task = "text-generation"
model_type = "qwen3_5_moe"
format = "safetensors"

[source]
repository = "local"
revision = "main"

[tensor]
count = 1
bytes = 1024

[[files]]
path = "model-00001.safetensors"
sha256 = "{DIGEST_A}"
size_bytes = 4096
role = "weight"
"""


def make_checkpoint(**overrides) -> PreparedCheckpoint:
    identity = PreparedCheckpointIdentity(
        package_id="qwen36-35b-a3b",
        package_sha256=PACKAGE_SHA,
        artifact_id="qwen-test",
        artifact_manifest_sha256=MANIFEST_SHA,
        model_type="qwen3_5_moe",
        format="safetensors",
        source_repository="local",
        source_revision="main",
        artifact_file_count=1,
    )
    checkpoint = PreparedCheckpoint(
        identity=replace(identity, **overrides.pop("identity", {})),
        shards=overrides.pop(
            "shards",
            (
                PreparedShard(
                    path="model-00001.safetensors",
                    sha256=DIGEST_A,
                    file_size_bytes=4096,
                    data_offset_bytes=128,
                    data_size_bytes=3968,
                ),
            ),
        ),
        tensors=overrides.pop(
            "tensors",
            (
                PreparedTensor(
                    name="model.embed_tokens.weight",
                    data_type=TensorDataType.BF16,
                    shape=(32, 16),
                    shard=0,
                    shard_offset_bytes=0,
                    size_bytes=1024,
                ),
            ),
        ),
    )
    return checkpoint


def manifest_and_digests():
    manifest = parse_manifest(MANIFEST_TEXT)
    return manifest, PACKAGE_SHA, MANIFEST_SHA


class PreparedCheckpointDecoding(unittest.TestCase):
    def test_decode_inverts_encode(self):
        checkpoint = make_checkpoint()

        self.assertEqual(
            decode_prepared_checkpoint(encode_prepared_checkpoint(checkpoint)), checkpoint
        )

    def test_rejects_foreign_magic(self):
        record = bytearray(encode_prepared_checkpoint(make_checkpoint()))
        record[0:8] = b"NOTCKPT\0"

        with self.assertRaisesRegex(PreparedCheckpointError, "magic"):
            decode_prepared_checkpoint(bytes(record))

    def test_rejects_truncation(self):
        record = encode_prepared_checkpoint(make_checkpoint())

        with self.assertRaises(PreparedCheckpointError):
            decode_prepared_checkpoint(record[:-8])

    def test_rejects_trailing_bytes(self):
        record = encode_prepared_checkpoint(make_checkpoint())

        with self.assertRaises(PreparedCheckpointError):
            decode_prepared_checkpoint(record + b"\0" * 8)

    def test_rejects_a_record_shorter_than_its_prefix(self):
        with self.assertRaisesRegex(PreparedCheckpointError, "prefix"):
            decode_prepared_checkpoint(MAGIC)


class PreparedCheckpointValidation(unittest.TestCase):
    def test_a_matching_record_passes(self):
        manifest, package, manifest_sha = manifest_and_digests()

        report = validate_prepared_checkpoint(
            make_checkpoint(), manifest, package, manifest_sha
        )

        self.assertTrue(report.ok, report.findings)
        self.assertEqual(report.findings, ())

    def test_catches_the_orphaned_record(self):
        """The real failure: model.toml is edited, its digest moves, and an
        already-prepared record silently no longer matches."""
        manifest, _package, manifest_sha = manifest_and_digests()

        report = validate_prepared_checkpoint(
            make_checkpoint(), manifest, "f" * 64, manifest_sha
        )

        self.assertFalse(report.ok)
        self.assertEqual([f.check for f in report.errors], [CHECK_PACKAGE_DIGEST])
        self.assertIn("model.toml", report.errors[0].detail)

    def test_catches_manifest_digest_drift(self):
        manifest, package, _manifest_sha = manifest_and_digests()

        report = validate_prepared_checkpoint(make_checkpoint(), manifest, package, "e" * 64)

        self.assertEqual([f.check for f in report.errors], [CHECK_MANIFEST_DIGEST])

    def test_catches_a_foreign_artifact(self):
        manifest, package, manifest_sha = manifest_and_digests()
        checkpoint = make_checkpoint(identity={"artifact_id": "someone-elses"})

        report = validate_prepared_checkpoint(checkpoint, manifest, package, manifest_sha)

        self.assertIn(CHECK_ARTIFACT_ID, [f.check for f in report.errors])

    def test_catches_file_count_drift(self):
        manifest, package, manifest_sha = manifest_and_digests()
        checkpoint = make_checkpoint(identity={"artifact_file_count": 7})

        report = validate_prepared_checkpoint(checkpoint, manifest, package, manifest_sha)

        self.assertIn(CHECK_FILE_COUNT, [f.check for f in report.errors])

    def test_catches_a_shard_whose_digest_moved(self):
        manifest, package, manifest_sha = manifest_and_digests()
        checkpoint = make_checkpoint(
            shards=(
                PreparedShard(
                    path="model-00001.safetensors",
                    sha256=DIGEST_B,
                    file_size_bytes=4096,
                    data_offset_bytes=128,
                    data_size_bytes=3968,
                ),
            )
        )

        report = validate_prepared_checkpoint(checkpoint, manifest, package, manifest_sha)

        self.assertIn(CHECK_SHARD_DIGEST, [f.check for f in report.errors])

    def test_catches_a_tensor_past_the_end_of_its_shard(self):
        manifest, package, manifest_sha = manifest_and_digests()
        checkpoint = make_checkpoint(
            tensors=(
                PreparedTensor(
                    name="model.embed_tokens.weight",
                    data_type=TensorDataType.BF16,
                    shape=(32, 16),
                    shard=0,
                    shard_offset_bytes=3500,
                    size_bytes=1024,
                ),
            )
        )

        report = validate_prepared_checkpoint(checkpoint, manifest, package, manifest_sha)

        self.assertIn(CHECK_TENSOR_BOUNDS, [f.check for f in report.errors])


if __name__ == "__main__":
    unittest.main()
