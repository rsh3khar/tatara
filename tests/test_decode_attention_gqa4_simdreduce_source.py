"""Static safety and exact-tree contract for A13 score reduction."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "src/backend/metal/kernels/attention.metal"
FIXTURE_PATH = ROOT / "tools/native/attention_fixture_probe.cpp"
HARNESS_HEADER_PATH = ROOT / "tools/native/decode_harness.h"
HARNESS_PATH = ROOT / "tools/native/decode_harness.cpp"
PERF_PATH = ROOT / "tools/native/decode_perf_probe.cpp"


def compact(source: str) -> str:
    return re.sub(r"\s+", "", source)


def braced_function(source: str, marker: str) -> str:
    begin = source.index(marker)
    body = source.index("{", begin)
    depth = 0
    for index in range(body, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[begin : index + 1]
    raise AssertionError(f"unterminated function after {marker}")


class DecodeAttentionGqa4SimdReduceSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = KERNEL_PATH.read_text()
        cls.kernel = braced_function(
            source, "kernel void attention_decode_scores_gqa4_simdreduce("
        )
        cls.maximum = braced_function(
            source, "inline float reduce_max_256_simd_tail_exact("
        )
        cls.total = braced_function(
            source, "inline float reduce_sum_256_simd_tail_exact("
        )
        cls.fixture = FIXTURE_PATH.read_text()
        cls.harness_header = HARNESS_HEADER_PATH.read_text()
        cls.harness = HARNESS_PATH.read_text()
        cls.perf = PERF_PATH.read_text()

    def test_dispatch_and_storage_match_permanent_gqa4(self) -> None:
        arrays = re.findall(
            r"threadgroup float (scores|red)\[(\d+)\]\[(\d+)\];",
            self.kernel,
        )
        self.assertEqual(
            arrays, [("scores", "4", "256"), ("red", "4", "256")]
        )
        self.assertEqual(2 * 4 * 256 * 4, 8192)
        fixture = compact(self.fixture)
        self.assertIn(
            compact(".width = 256, .height = 4, .depth = 1"), fixture
        )
        self.assertEqual(256 * 4, 1024)

    def test_cross_group_and_simd_tail_levels_are_exact(self) -> None:
        for helper in (self.maximum, self.total):
            body = compact(helper)
            self.assertIn(
                compact(
                    "for (uint off = 128u; off >= 32u; off >>= 1u)"
                ),
                body,
            )
            self.assertIn(
                compact("for (uint off = 16u; off; off >>= 1u)"), body
            )
            self.assertIn("simd_shuffle_down(value,off)", body)
            self.assertIn("if(lane<off)", body)
            self.assertEqual(helper.count("threadgroup_barrier("), 2)
        self.assertIn(
            "values[tid] = max(values[tid], values[tid + off]);",
            self.maximum,
        )
        self.assertIn(
            "value = max(value, other);", self.maximum
        )
        self.assertIn(
            "values[tid] += values[tid + off];", self.total
        )
        self.assertIn("value += other;", self.total)

    def test_score_generation_and_record_layout_are_unchanged(self) -> None:
        kernel = compact(self.kernel)
        self.assertIn(
            compact("for (uint t = cohort * 8u + sg; t < count; t += 32u)"),
            kernel,
        )
        for head in range(4):
            self.assertIn(f"floatdot{head}=0.0f;", kernel)
            self.assertIn(f"dot{head}=simd_sum(dot{head});", kernel)
            self.assertIn(
                f"scores[{head}][t]=dot{head}*kAttnScale;", kernel
            )
        self.assertIn(
            compact(
                "device float* dst = weights + "
                "(head * nparts + tg.y) * 258u;"
            ),
            kernel,
        )
        self.assertIn("dst[256]=part_max;", kernel)
        self.assertIn("dst[257]=part_sum;", kernel)

    def test_component_fixture_requires_complete_byte_identity(self) -> None:
        fixture = compact(self.fixture)
        self.assertIn(
            '"attention_decode_scores_gqa4_simdreduce"', self.fixture
        )
        self.assertIn("&c_long", self.fixture)
        self.assertIn("&c_split", self.fixture)
        self.assertIn(
            "weights2_simdreduce==weights2_device", fixture
        )
        self.assertIn(
            "weights1_simdreduce==weights1_gqa4", fixture
        )
        self.assertIn(
            compact(
                "if (!simdreduce_two_part_exact || "
                "!simdreduce_one_part_exact) { return 68; }"
            ),
            fixture,
        )

    def test_candidate_is_probe_only_beside_adaptive_default(self) -> None:
        header = compact(self.harness_header)
        harness = compact(self.harness)
        self.assertIn("Gqa4,Gqa4SimdReduce,Gqa8", header)
        self.assertIn(
            compact(
                "score_kernel == "
                "DecodeAttentionScoreKernel::Gqa4SimdReduce"
            ),
            harness,
        )
        self.assertIn(
            'kernel_name="attention_decode_scores_gqa4_simdreduce";',
            harness,
        )
        self.assertGreaterEqual(
            self.harness.count("DecodeAttentionScoreKernel::Adaptive"), 2
        )
        self.assertIn('"attention_decode_scores_gqa4"', self.harness)
        self.assertIn('"score-gqa4-simdreduce"', self.perf)
        self.assertIn(
            "required_capacity > plan.tokenizer.maximum_context", self.perf
        )


if __name__ == "__main__":
    unittest.main()
