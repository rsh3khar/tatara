import hashlib
import unittest
from dataclasses import replace
from pathlib import Path

from tatara.artifact_manifest import parse_manifest
from tatara.qwen36_plan_generation import (
    TENSOR_NAME_FIELDS,
    ModelPlanGenerationError,
    parse_model_plan,
    render_model_plan_header,
)


PACKAGE = Path("catalog/model_packages/qwen36-35b-a3b/model.toml")
ARTIFACT = Path("catalog/model_packages/qwen36-35b-a3b/artifact.toml")


def parse_plan(package_text=None):
    text = package_text if package_text is not None else PACKAGE.read_text()
    artifact_bytes = ARTIFACT.read_bytes()
    artifact = parse_manifest(artifact_bytes.decode("utf-8"))
    return artifact, parse_model_plan(
        text,
        artifact,
        ARTIFACT.name,
        hashlib.sha256(text.encode()).hexdigest(),
        hashlib.sha256(artifact_bytes).hexdigest(),
    )


class Qwen36PlanGenerationTests(unittest.TestCase):
    def test_first_package_generates_the_exact_hybrid_schedule(self):
        artifact, plan = parse_plan()

        self.assertEqual(len(plan.layer_kinds), 40)
        self.assertEqual(plan.layer_kinds.count("FullAttention"), 10)
        self.assertEqual(plan.layer_kinds.count("GatedDelta"), 30)
        self.assertEqual(plan.layer_kinds[3::4], ("FullAttention",) * 10)
        self.assertEqual(plan.artifact_id, artifact.artifact_id)
        self.assertEqual(plan.weight_file_count, 4)
        self.assertEqual(len(plan.package_sha256), 64)
        self.assertEqual(len(plan.artifact_manifest_sha256), 64)
        self.assertEqual(plan.tokenizer_kind, "byte-level-bpe")
        self.assertEqual(plan.tokenizer_vocabulary, 248320)
        self.assertEqual(plan.tokenizer_populated_vocabulary, 248077)
        self.assertEqual(plan.maximum_context, 262144)
        self.assertEqual(plan.stop_token_ids, (248046, 248044))
        self.assertEqual(plan.tokenizer_data_size_bytes, 19989343)

    def test_render_is_deterministic_and_path_free(self):
        _, plan = parse_plan()
        first = render_model_plan_header(plan)
        second = render_model_plan_header(plan)

        self.assertEqual(first, second)
        self.assertIn("static_assert(valid_model_plan(kModelPlan));", first)
        self.assertNotIn("/Users/", first)

    def test_external_identity_strings_are_escaped(self):
        _, plan = parse_plan()
        changed = replace(plan, source_repository='owner/"model')

        rendered = render_model_plan_header(changed)

        self.assertIn('source_repository = "owner/\\\"model"', rendered)

    def test_package_cannot_select_a_different_artifact(self):
        changed = PACKAGE.read_text().replace(
            'artifact_manifest = "artifact.toml"',
            'artifact_manifest = "different.toml"',
        )

        with self.assertRaisesRegex(ModelPlanGenerationError, "different artifact"):
            parse_plan(changed)

    def test_invalid_expert_topology_is_rejected(self):
        changed = PACKAGE.read_text().replace("active_experts = 8", "active_experts = 512")

        with self.assertRaisesRegex(ModelPlanGenerationError, "Active expert"):
            parse_plan(changed)

    def test_first_package_declares_its_expert_dimension(self):
        _, plan = parse_plan()

        self.assertEqual(plan.expert_dimension, 512)
        self.assertIn(".expert_dimension = 512,", render_model_plan_header(plan))

    def test_excluded_tensor_prefixes_reach_the_generated_header(self):
        _, plan = parse_plan()

        self.assertEqual(plan.excluded_tensor_prefixes, ("vision_tower.",))
        self.assertIn(
            'kExcludedTensorPrefixes{\n    "vision_tower.",\n}',
            render_model_plan_header(plan),
        )

    def test_repeated_excluded_prefix_is_rejected(self):
        changed = PACKAGE.read_text().replace(
            'excluded_tensor_prefixes = ["vision_tower."]',
            'excluded_tensor_prefixes = ["vision_tower.", "vision_tower."]',
        )

        with self.assertRaisesRegex(ModelPlanGenerationError, "repeats an excluded"):
            parse_plan(changed)

    def test_tensor_naming_renders_every_field_in_declaration_order(self):
        _, plan = parse_plan()

        rendered = render_model_plan_header(plan)

        self.assertEqual(tuple(name for name, _ in plan.tensor_names), TENSOR_NAME_FIELDS)
        self.assertIn('.model_prefix = "language_model.model.",', rendered)
        self.assertIn('.layer_index_suffix = ".",', rendered)
        self.assertIn('.head_stem = "language_model.lm_head",', rendered)
        self.assertIn('.quantized_biases = ".biases",', rendered)
        positions = [rendered.index(f".{name} = ") for name in TENSOR_NAME_FIELDS]
        self.assertEqual(positions, sorted(positions))

    def test_incomplete_tensor_naming_table_is_rejected(self):
        changed = PACKAGE.read_text().replace('layer_index_suffix = "."\n', "")

        with self.assertRaisesRegex(ModelPlanGenerationError, "tensor naming table"):
            parse_plan(changed)

    def test_tokenizer_digest_must_match_the_artifact_manifest(self):
        changed = PACKAGE.read_text().replace(
            "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
            "0" * 64,
        )

        with self.assertRaisesRegex(
            ModelPlanGenerationError, "digest differs from manifest"
        ):
            parse_plan(changed)

    def test_tokenizer_vocabulary_must_match_the_model(self):
        changed = PACKAGE.read_text().replace(
            'decoder = "byte-level"\n'
            "vocabulary = 248320",
            'decoder = "byte-level"\n'
            "vocabulary = 248319",
        )

        with self.assertRaisesRegex(
            ModelPlanGenerationError, "vocabulary sizes differ"
        ):
            parse_plan(changed)

    def test_populated_tokenizer_vocabulary_is_bounded(self):
        changed = PACKAGE.read_text().replace(
            "populated_vocabulary = 248077",
            "populated_vocabulary = 248321",
        )

        with self.assertRaisesRegex(
            ModelPlanGenerationError, "Populated tokenizer vocabulary"
        ):
            parse_plan(changed)

    def test_stop_tokens_are_bounded_unique_and_complete(self):
        repeated = PACKAGE.read_text().replace(
            "stop_token_ids = [248046, 248044]",
            "stop_token_ids = [248046, 248046]",
        )
        missing = PACKAGE.read_text().replace(
            "stop_token_ids = [248046, 248044]",
            "stop_token_ids = [248046]",
        )

        with self.assertRaisesRegex(ModelPlanGenerationError, "repeats an ID"):
            parse_plan(repeated)
        with self.assertRaisesRegex(ModelPlanGenerationError, "must include"):
            parse_plan(missing)

    def test_generated_header_contains_tokenizer_identity(self):
        _, plan = parse_plan()
        rendered = render_model_plan_header(plan)

        self.assertIn(".kind = TokenizerKind::ByteLevelBpe", rendered)
        self.assertIn('.data_path = "tokenizer.json"', rendered)
        self.assertIn(".data_size_bytes = 19989343ULL", rendered)
        self.assertIn(".populated_vocabulary = 248077", rendered)
        self.assertIn(".maximum_context = 262144", rendered)
        self.assertIn(
            "std::array<std::uint32_t, kMaximumStopTokens>{248046, 248044}",
            rendered,
        )


if __name__ == "__main__":
    unittest.main()
