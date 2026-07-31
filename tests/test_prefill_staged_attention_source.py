import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "include/tatara/runtime/prefill_step.h"
GEOMETRY = ROOT / "include/tatara/runtime/prefill_geometry.h"
RUNTIME = ROOT / "src/runtime/prefill_step.cpp"
PROFILE = ROOT / "src/runtime/prefill_profile_plan.cpp"
METAL = ROOT / "src/backend/metal/kernels/prefill_attention.metal"
PROBE = ROOT / "tools/native/block_prefill_probe.cpp"


def function_body(source: str, marker: str) -> str:
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace : index + 1]
    raise AssertionError(f"unterminated function: {marker}")


class PrefillStagedAttentionSourceTest(unittest.TestCase):
    def test_policy_is_opt_in_and_exact_fallback_is_permanent(self):
        header = HEADER.read_text()
        runtime = RUNTIME.read_text()
        probe = PROBE.read_text()
        self.assertIn(
            "PrefillAttentionKernel::PartialCombine};", header
        )
        self.assertIn(
            "PrefillAttentionKernel::StagedGemmAdaptive", runtime
        )
        self.assertIn(
            "encode.pipeline(step.pipelines.attention_partial);",
            runtime,
        )
        self.assertIn(
            "encode.pipeline(step.pipelines.attention_combine);",
            runtime,
        )
        self.assertIn('"layer2048fast-n1-r2-a1"', probe)

    def test_causal_extent_is_additive_and_profile_uses_same_rule(self):
        runtime = RUNTIME.read_text()
        profile = PROFILE.read_text()
        self.assertIn(
            "const std::uint32_t tile_context = "
            "context_base + query_base;",
            runtime,
        )
        self.assertIn(
            "const std::uint32_t visible = "
            "tile_context + tile_rows;",
            runtime,
        )
        self.assertNotIn("context_base - query_base", runtime)
        self.assertIn(
            "static_cast<std::uint64_t>(initial_context) +",
            profile,
        )
        self.assertIn(
            "chunk.offset + query_base + tile_rows", profile
        )

    def test_staged_score_alias_is_checked_by_geometry(self):
        geometry = GEOMETRY.read_text()
        runtime = RUNTIME.read_text()
        self.assertIn(
            "attention_staged_score_bytes", geometry
        )
        self.assertIn(
            "geometry.attention_staged_score_bytes >\n"
            "        geometry.attention_partial_bytes",
            geometry,
        )
        self.assertIn(
            "geometry.attention_staged_score_bytes <=\n"
            "             geometry.attention_partial_bytes",
            runtime,
        )
        self.assertIn(
            "encode.buffer(step.attention_partials, 0, 2);",
            runtime,
        )

    def test_kernel_model_facts_are_generated_not_qwen_literals(self):
        metal = METAL.read_text()
        scores = function_body(
            metal, "kernel void attention_staged_scores_blk("
        )
        values = function_body(
            metal, "kernel void attention_staged_values_blk("
        )
        for body in (scores, values):
            self.assertIn("kAttnQueryHeads", body)
            self.assertIn("kAttnHeadsPerKv", body)
            self.assertIn("kAttnHeadDimension", body)
            self.assertNotIn("% 16u", body)
            self.assertNotIn("* 256u", body)
        self.assertIn("value * kAttnScale", scores)

    def test_host_abi_orders_three_staged_dispatches(self):
        runtime = RUNTIME.read_text()
        begin = runtime.index(
            "encode.pipeline(step.pipelines.attention_staged_scores);"
        )
        softmax = runtime.index(
            "encode.pipeline(step.pipelines.attention_staged_softmax);",
            begin,
        )
        values = runtime.index(
            "encode.pipeline(step.pipelines.attention_staged_values);",
            softmax,
        )
        self.assertLess(begin, softmax)
        self.assertLess(softmax, values)
        self.assertIn(
            "encode.constant(tile_context, 2);",
            runtime[softmax:values],
        )
        self.assertIn(
            "encode.constant(visible, 3);",
            runtime[values:],
        )

    def test_softmax_barrier_precedes_reduction_slot_reuse(self):
        softmax = function_body(
            METAL.read_text(),
            "kernel void attention_staged_softmax_blk(",
        )
        consumed = softmax.index(
            "denominator = simd_sum(denominator);"
        )
        barrier = softmax.index(
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            consumed,
        )
        reused = softmax.index(
            "reduction[simdgroup] = denominator;", consumed
        )
        self.assertLess(consumed, barrier)
        self.assertLess(barrier, reused)


if __name__ == "__main__":
    unittest.main()
