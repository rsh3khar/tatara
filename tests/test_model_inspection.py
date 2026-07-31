import io
import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tatara.cli import main
from tatara.model_inspection import inspect_snapshot, render_human
from tatara.model_source import (
    ModelSourceError,
    load_huggingface,
    load_local,
    parse_huggingface_reference,
)
from tatara.model_types import EvidenceLevel, FileRole, HashMode
from tatara.safetensors import SafetensorsError, read_header


def qwen_config():
    return {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "quantization": {"mode": "affine", "bits": 4, "group_size": 64},
        "text_config": {
            "dtype": "bfloat16",
            "hidden_size": 16,
            "vocab_size": 32,
            "num_hidden_layers": 2,
            "max_position_embeddings": 1024,
            "layer_types": ["linear_attention", "full_attention"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "linear_num_key_heads": 2,
            "linear_key_head_dim": 3,
            "linear_num_value_heads": 2,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
        },
        "vision_config": {"hidden_size": 8},
    }


def write_safetensors(path, tensors):
    offset = 0
    header = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def make_local_model(root):
    (root / "config.json").write_text(json.dumps(qwen_config()))
    (root / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend", "model_max_length": 1024})
    )
    (root / "tokenizer.json").write_text("{}")
    shard = "model-00001-of-00001.safetensors"
    tensors = [
        ("language_model.embed.weight", "U8", [8], b"12345678"),
        ("vision_tower.proj.weight", "F32", [1], b"1234"),
    ]
    write_safetensors(root / shard, tensors)
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {name: shard for name, _, _, _ in tensors},
            }
        )
    )


class LocalInspectionTests(unittest.TestCase):
    def test_duplicate_header_keys_are_rejected(self):
        raw = (
            b'{"duplicate":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},'
            b'"duplicate":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.safetensors"
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"x")
            with self.assertRaisesRegex(SafetensorsError, "Duplicate"):
                read_header(path)

    def test_local_qwen_headers_and_capacity_are_inspected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_local_model(root)
            snapshot = load_local(root)
            report = inspect_snapshot(snapshot, context_tokens=128, slots=2)

        self.assertEqual(snapshot.tensor_summary.tensor_count, 2)
        self.assertEqual(snapshot.tensor_summary.tensor_bytes, 12)
        self.assertTrue(snapshot.tensor_summary.headers_verified)
        self.assertEqual(report["compatibility"]["architecture_status"], "recognized")
        self.assertNotIn(
            "qwen3.5-vision-execution",
            report["compatibility"]["missing_capabilities"],
        )
        self.assertEqual(report["compatibility"]["unrequested_modalities"], ["vision"])
        self.assertEqual(report["capacity"]["known_model_bytes_lower_bound"], 9068)
        self.assertFalse(report["scope"]["model_loaded"])

    def test_metadata_hashing_does_not_read_weight_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_local_model(root)
            snapshot = load_local(root, hash_mode=HashMode.METADATA)

        weights = [value for value in snapshot.files if value.role is FileRole.WEIGHT]
        metadata = [value for value in snapshot.files if value.role is FileRole.METADATA]
        self.assertTrue(all(value.sha256 is None for value in weights))
        self.assertTrue(all(value.sha256 for value in metadata))
        self.assertTrue(snapshot.scope.weight_headers_read)
        self.assertFalse(snapshot.scope.weight_payload_read)

    def test_index_header_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_local_model(root)
            index_path = root / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"]["missing.tensor"] = "model-00001-of-00001.safetensors"
            index_path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ModelSourceError, "index/header mismatch"):
                load_local(root)

    def test_human_report_states_safety_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_local_model(root)
            report = inspect_snapshot(load_local(root), 128, 1)
        rendered = render_human(report)
        self.assertIn("MISSING CAPABILITY", rendered)
        self.assertIn("no model was loaded", rendered)
        self.assertIn("no remote code executed", rendered)

    def test_cli_json_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_local_model(root)
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit) as exit_state:
                main(["inspect", str(root), "--context", "128", "--json"])
        self.assertEqual(exit_state.exception.code, 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(value["command"], "inspect")


class HuggingFaceInspectionTests(unittest.TestCase):
    def test_reference_parser_pins_main_by_default(self):
        self.assertEqual(parse_huggingface_reference("hf://owner/model"), ("owner/model", "main"))
        self.assertEqual(
            parse_huggingface_reference("hf://owner/model@abc123"),
            ("owner/model", "abc123"),
        )

    def test_network_requires_explicit_permission(self):
        with self.assertRaisesRegex(ModelSourceError, "--allow-network"):
            load_huggingface("hf://owner/model", allow_network=False)

    def test_remote_inspection_fetches_metadata_not_weights(self):
        calls = []

        def fetch(url, headers):
            calls.append(url)
            if "/api/models/" in url:
                return {
                    "sha": "a" * 40,
                    "pipeline_tag": "text-generation",
                    "evalResults": [{"task": "publisher-claim"}],
                    "siblings": [
                        {"rfilename": "config.json", "size": 100},
                        {"rfilename": "tokenizer.json", "size": 100},
                        {"rfilename": "model.safetensors.index.json", "size": 100},
                        {
                            "rfilename": "model-00001-of-00001.safetensors",
                            "size": 1024,
                            "lfs": {"size": 1024, "sha256": "b" * 64},
                        },
                    ],
                }
            if url.endswith("/config.json"):
                return qwen_config()
            if url.endswith("/model.safetensors.index.json"):
                return {
                    "metadata": {"total_size": 900},
                    "weight_map": {
                        "language_model.embed.weight": "model-00001-of-00001.safetensors"
                    },
                }
            self.fail(f"unexpected metadata request: {url}")

        snapshot = load_huggingface(
            "hf://owner/model@main", allow_network=True, fetch_json=fetch
        )
        report = inspect_snapshot(snapshot, 128, 1)
        self.assertEqual(snapshot.resolved_revision, "a" * 40)
        self.assertIs(snapshot.evidence, EvidenceLevel.REMOTE_METADATA)
        self.assertFalse(snapshot.tensor_summary.headers_verified)
        self.assertEqual(len(calls), 3)
        self.assertTrue(all("model-00001" not in url for url in calls))
        self.assertTrue(report["compatibility"]["remote_metadata_only"])
        self.assertEqual(len(report["evaluations"]["publisher_reported"]), 1)


if __name__ == "__main__":
    unittest.main()
