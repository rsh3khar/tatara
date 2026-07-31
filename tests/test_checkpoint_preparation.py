import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tatara.cli import main
from tatara.checkpoint_preparation import (
    CheckpointPreparationError,
    prepare_checkpoint,
    write_new_checkpoint,
)
from tatara.model_source import load_local
from tatara.model_types import HashMode
from tatara.prepared_checkpoint import (
    PreparedCheckpointError,
    encode_prepared_checkpoint,
)
from test_artifact_manifest import manifest_text
from test_model_inspection import make_local_model


def package_text():
    return "\n".join(
        (
            "schema_version = 1",
            'id = "fixture-qwen"',
            'family = "fixture-family"',
            'model_type = "qwen3_5_moe"',
            'artifact_manifest = "artifact.toml"',
        )
    )


def prepare_fixture(root):
    model = root / "model"
    package = root / "package"
    model.mkdir()
    package.mkdir()
    make_local_model(model)
    snapshot = load_local(model, HashMode.ALL)
    (package / "model.toml").write_text(package_text())
    (package / "artifact.toml").write_text(manifest_text(snapshot))
    return model, package / "model.toml"


class CheckpointPreparationTests(unittest.TestCase):
    def test_preparation_is_deterministic_and_header_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, package = prepare_fixture(Path(temporary))
            first, first_bytes = prepare_checkpoint(package, model)
            second, second_bytes = prepare_checkpoint(package, model)

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first.identity.package_id, "fixture-qwen")
        self.assertEqual(
            first.identity.package_sha256,
            hashlib.sha256(package_text().encode()).hexdigest(),
        )
        self.assertEqual(len(first.shards), 1)
        self.assertEqual(len(first.tensors), 2)
        self.assertEqual(first.tensor_payload_bytes, 12)
        self.assertEqual(first.tensors[0].name, "language_model.embed.weight")
        self.assertGreater(first.shards[0].data_offset_bytes, 8)

    def test_index_mapping_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, package = prepare_fixture(Path(temporary))
            index_path = model / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"]["language_model.embed.weight"] = "wrong.safetensors"
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(CheckpointPreparationError, "wrong shard"):
                prepare_checkpoint(package, model)

    def test_encoder_rejects_noncanonical_tensor_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            model, package = prepare_fixture(Path(temporary))
            checkpoint, _ = prepare_checkpoint(package, model)
        reversed_checkpoint = type(checkpoint)(
            checkpoint.identity,
            checkpoint.shards,
            tuple(reversed(checkpoint.tensors)),
        )
        with self.assertRaisesRegex(PreparedCheckpointError, "unordered tensor"):
            encode_prepared_checkpoint(reversed_checkpoint)

    def test_output_is_create_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, package = prepare_fixture(root)
            _, encoded = prepare_checkpoint(package, model)
            output = root / "output" / "checkpoint.tatara"
            write_new_checkpoint(output, encoded)
            with self.assertRaises(FileExistsError):
                write_new_checkpoint(output, b"replacement")
            self.assertEqual(output.read_bytes(), encoded)

    def test_cli_reports_scope_and_writes_the_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, package = prepare_fixture(root)
            output = root / "prepared" / "checkpoint.tatara"
            stdout = io.StringIO()
            with redirect_stdout(stdout), self.assertRaises(SystemExit) as exit_state:
                main(
                    [
                        "prepare",
                        str(model),
                        "--package",
                        str(package),
                        "--output",
                        str(output),
                        "--json",
                    ]
                )
            report = json.loads(stdout.getvalue())
            self.assertEqual(exit_state.exception.code, 0)
            self.assertTrue(output.exists())
            self.assertEqual(report["command"], "prepare")
            self.assertFalse(report["weight_payload_read"])
            self.assertFalse(report["weight_payload_hashed"])


if __name__ == "__main__":
    unittest.main()
