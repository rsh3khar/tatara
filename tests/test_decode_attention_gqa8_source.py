"""Static safety and measurement contract for the score-only GQA8 candidate."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "src/backend/metal/kernels/attention.metal"
FIXTURE_PATH = ROOT / "tools/native/attention_fixture_probe.cpp"
HARNESS_PATH = ROOT / "tools/native/decode_harness.cpp"
PERF_PATH = ROOT / "tools/native/decode_perf_probe.cpp"


def braced_function(source: str, name: str) -> str:
    marker = f"kernel void {name}("
    begin = source.index(marker)
    body_begin = source.index("{", begin)
    depth = 0
    for index in range(body_begin, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[begin : index + 1]
    raise AssertionError(f"unterminated kernel {name}")


def compact(source: str) -> str:
    return re.sub(r"\s+", "", source)


class DecodeAttentionGqa8SourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kernel_source = KERNEL_PATH.read_text()
        cls.kernel = braced_function(
            cls.kernel_source, "attention_decode_scores_gqa8"
        )
        cls.fixture = FIXTURE_PATH.read_text()
        cls.harness = HARNESS_PATH.read_text()
        cls.perf = PERF_PATH.read_text()

    def test_dispatch_and_threadgroup_memory_are_statically_bounded(self) -> None:
        self.assertIn("if (tg.x >= kAttnKvHeads)", self.kernel)
        self.assertLess(
            self.kernel.index("if (tg.x >= kAttnKvHeads)"),
            self.kernel.index("threadgroup_barrier("),
        )
        arrays = re.findall(
            r"threadgroup float (scores|red)\[(\d+)\]\[(\d+)\];",
            self.kernel,
        )
        self.assertEqual(arrays, [("scores", "8", "256"), ("red", "8", "256")])
        threadgroup_bytes = sum(
            int(rows) * int(columns) * 4
            for _, rows, columns in arrays
        )
        self.assertEqual(threadgroup_bytes, 16_384)
        self.assertLessEqual(threadgroup_bytes, 32_768)
        self.assertIn(
            compact(".width = 256, .height = 4, .depth = 1"),
            compact(self.fixture),
        )

    def test_key_load_is_shared_across_all_eight_query_heads(self) -> None:
        kernel = compact(self.kernel)
        self.assertIn("constuintkv=tg.x;", kernel)
        self.assertIn("constuinthead_base=kv*8u;", kernel)
        self.assertIn(
            "for(uintt=cohort*8u+sg;t<count;t+=32u)", kernel
        )
        self.assertEqual(
            self.kernel.count(
                "float(keys[(kv * capacity + start + t) * "
                "kAttnHeadDimension + d])"
            ),
            1,
        )
        for head in range(8):
            self.assertIn(f"float dot{head} = 0.0f;", self.kernel)
            self.assertIn(
                f"dot{head} += float(q[(head_base + {head}u) * "
                "kAttnHeadDimension + d]) * key_value;",
                self.kernel,
            )
            self.assertIn(f"dot{head} = simd_sum(dot{head});", self.kernel)
            self.assertIn(
                f"scores[{head}][t] = dot{head} * kAttnScale;",
                self.kernel,
            )

    def test_reduction_and_record_layout_match_the_permanent_control(self) -> None:
        kernel = compact(self.kernel)
        self.assertIn(
            compact("for (uint phase = 0u; phase < 2u; ++phase)"), kernel
        )
        self.assertIn(
            compact("const uint local_head = cohort + phase * 4u;"), kernel
        )
        self.assertEqual(
            self.kernel.count(
                "for (uint off = 128u; off; off >>= 1u)"
            ),
            2,
        )
        self.assertIn(
            compact(
                "device float* dst = weights + "
                "(head * nparts + tg.y) * 258u;"
            ),
            kernel,
        )
        self.assertIn("dst[256] = part_max;", self.kernel)
        self.assertIn("dst[257] = red[local_head][0];", self.kernel)

    def test_component_gate_covers_partition_boundary_and_tail(self) -> None:
        fixture = compact(self.fixture)
        self.assertIn('"attention_decode_scores_gqa4"', self.fixture)
        self.assertIn('"attention_decode_scores_gqa8"', self.fixture)
        self.assertIn("&c_long", self.fixture)
        self.assertIn("&c_split", self.fixture)
        self.assertIn(
            compact("weights2_gqa8 == weights2_device"), fixture
        )
        self.assertIn(
            compact("weights1_gqa8 == weights1_gqa4"), fixture
        )
        self.assertIn(
            compact(
                "if (!gqa8_two_part_exact || !gqa8_one_part_exact) "
                "{ return 69; }"
            ),
            fixture,
        )

    def test_candidate_is_probe_selectable_beside_adaptive_default(self) -> None:
        harness = compact(self.harness)
        self.assertIn(
            compact(
                "} else if ("
                "score_kernel == DecodeAttentionScoreKernel::Gqa8) {"
                ' kernel_name = "attention_decode_scores_gqa8"; }'
            ),
            harness,
        )
        self.assertGreaterEqual(
            self.harness.count("DecodeAttentionScoreKernel::Adaptive"), 2
        )
        self.assertIn('"attention_decode_scores_gqa4"', self.harness)
        self.assertIn('"score-gqa4"', self.perf)
        self.assertIn('"score-gqa8"', self.perf)
        self.assertIn(
            "required_capacity > plan.tokenizer.maximum_context", self.perf
        )


if __name__ == "__main__":
    unittest.main()
