import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tatara.artifact_manifest import (
    ArtifactManifestError,
    parse_manifest,
    verify_manifest,
)
from tatara.cli import main
from tatara.model_source import load_local
from tatara.model_types import HashMode
from test_model_inspection import make_local_model


def manifest_text(snapshot):
    lines = [
        "schema_version = 1",
        "",
        "[artifact]",
        'id = "fixture-qwen"',
        'task = "text-generation"',
        'model_type = "qwen3_5_moe"',
        'format = "safetensors"',
        "",
        "[source]",
        'repository = "fixture/qwen"',
        'revision = "0123456789abcdef"',
        "",
        "[tensor]",
        f"count = {snapshot.tensor_summary.tensor_count}",
        f"bytes = {snapshot.tensor_summary.tensor_bytes}",
    ]
    for item in snapshot.files:
        lines.extend(
            [
                "",
                "[[files]]",
                f'path = "{item.path}"',
                f'role = "{item.role.value}"',
                f"size_bytes = {item.size_bytes}",
                f'sha256 = "{item.sha256}"',
            ]
        )
    return "\n".join(lines)


class ArtifactManifestTests(unittest.TestCase):
    def test_manifest_is_path_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "first"
            second = Path(temporary) / "second"
            first.mkdir()
            make_local_model(first)
            snapshot = load_local(first, HashMode.ALL)
            manifest = parse_manifest(manifest_text(snapshot))
            shutil.copytree(first, second)
            relocated = load_local(second, HashMode.ALL)

        self.assertTrue(verify_manifest(manifest, snapshot).matched)
        self.assertTrue(verify_manifest(manifest, relocated).matched)

    def test_tampered_file_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_local_model(root)
            snapshot = load_local(root, HashMode.ALL)
            manifest = parse_manifest(manifest_text(snapshot))
            (root / "tokenizer.json").write_text('{"changed":true}')
            changed = load_local(root, HashMode.ALL)
            result = verify_manifest(manifest, changed)

        self.assertFalse(result.matched)
        self.assertTrue(
            any("tokenizer.json" in mismatch for mismatch in result.mismatches)
        )

    def test_unsafe_manifest_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_local_model(root)
            snapshot = load_local(root, HashMode.ALL)
            text = manifest_text(snapshot).replace(
                'path = "config.json"', 'path = "../config.json"'
            )
        with self.assertRaisesRegex(ArtifactManifestError, "Unsafe"):
            parse_manifest(text)

    def test_first_target_manifest_is_well_formed(self):
        path = Path(
            "catalog/model_packages/qwen36-35b-a3b/artifact.toml"
        )
        manifest = parse_manifest(path.read_text())
        self.assertEqual(manifest.tensor_count, 2090)
        self.assertEqual(manifest.tensor_bytes, 20401929952)
        self.assertEqual(len(manifest.files), 17)
        self.assertNotIn("/Users/", path.read_text())

    def test_cli_verifies_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "model"
            root.mkdir()
            make_local_model(root)
            snapshot = load_local(root, HashMode.ALL)
            manifest_path = Path(temporary) / "artifact.toml"
            manifest_path.write_text(manifest_text(snapshot))
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as exit_state:
                main(
                    [
                        "inspect",
                        str(root),
                        "--verify-manifest",
                        str(manifest_path),
                        "--json",
                    ]
                )
        self.assertEqual(exit_state.exception.code, 0)
        report = json.loads(output.getvalue())
        self.assertTrue(report["artifact_manifest"]["matched"])


if __name__ == "__main__":
    unittest.main()
