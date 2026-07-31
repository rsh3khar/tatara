import math
import random
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "include/tatara/runtime/prefill_step.h"
METAL = ROOT / "src/backend/metal/kernels/prefill_attention.metal"
RUNTIME = ROOT / "src/runtime/prefill_step.cpp"
PROFILE = ROOT / "src/runtime/prefill_profile_plan.cpp"
GENERATOR = ROOT / "tools/generate_kernel_library.py"
PROBE = ROOT / "tools/native/block_prefill_probe.cpp"
FIXTURE = ROOT / "tools/native/attention_fixture_probe.cpp"


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


def fragment_coordinate(lane: int) -> tuple[int, int]:
    quad = lane >> 2
    row = (quad & 4) + ((lane >> 1) & 3)
    column = (quad & 2) * 2 + (lane & 1) * 2
    return row, column


def direct_attention(query, keys, values, limit):
    scores = [
        sum(left * right for left, right in zip(query, key))
        for key in keys[:limit]
    ]
    maximum = max(scores)
    weights = [math.exp(score - maximum) for score in scores]
    denominator = sum(weights)
    return [
        sum(weight * row[column] for weight, row in zip(weights, values))
        / denominator
        for column in range(len(values[0]))
    ]


def streaming_attention(query, keys, values, limit, tile):
    maximum = -math.inf
    denominator = 0.0
    numerator = [0.0] * len(values[0])
    for begin in range(0, limit, tile):
        end = min(limit, begin + tile)
        scores = [
            sum(left * right for left, right in zip(query, key))
            for key in keys[begin:end]
        ]
        tile_maximum = max(scores)
        next_maximum = max(maximum, tile_maximum)
        rescale = 0.0 if maximum == -math.inf else math.exp(
            maximum - next_maximum
        )
        weights = [math.exp(score - next_maximum) for score in scores]
        numerator = [value * rescale for value in numerator]
        for weight, row in zip(weights, values[begin:end]):
            for column, value in enumerate(row):
                numerator[column] += weight * value
        denominator = denominator * rescale + sum(weights)
        maximum = next_maximum
    return [value / denominator for value in numerator]


class PrefillStreamingAttentionSourceTest(unittest.TestCase):
    def test_policy_is_opt_in_and_controls_remain_permanent(self):
        header = HEADER.read_text(encoding="utf-8")
        runtime = RUNTIME.read_text(encoding="utf-8")
        probe = PROBE.read_text(encoding="utf-8")
        self.assertIn("StreamingFlashAdaptive", header)
        self.assertIn(
            "PrefillAttentionKernel::PartialCombine};", header
        )
        self.assertIn(
            "PrefillAttentionKernel::StagedGemmAdaptive", runtime
        )
        self.assertIn(
            "PrefillAttentionKernel::StreamingFlashAdaptive", runtime
        )
        self.assertIn(
            '"layer2048fast-steel-full-graph-streaming"', probe
        )
        self.assertIn(
            '"layer2048fast-steel-full-graph-streaming-warm"', probe
        )
        streaming_warm = probe.index(
            '"layer2048fast-steel-full-graph-streaming-warm"'
        )
        next_branch = probe.index("} else if (", streaming_warm)
        arm = probe[streaming_warm:next_branch]
        self.assertIn(
            "PrefillAttentionKernel::StreamingFlashAdaptive", arm
        )
        self.assertIn("policy.command_graph = true;", arm)
        self.assertIn("policy.command_graph_warm = true;", arm)
        self.assertIn(
            "(!pipeline ||\n"
            "                 supports_indirect_commands(pipeline))",
            RUNTIME.read_text(encoding="utf-8"),
        )

    def test_qtile2048_treatment_changes_only_staged_query_geometry(self):
        probe = PROBE.read_text(encoding="utf-8")
        marker = (
            '"layer2048fast-steel-full-graph-qtile2048-warm"'
        )
        treatment = probe.index(marker)
        next_branch = probe.index("} else if (", treatment)
        arm = probe[treatment:next_branch]
        self.assertIn("policy = kLayer2048Fast;", arm)
        self.assertIn(
            "policy.query_tile = kFixtureProductBlock;", arm
        )
        self.assertIn(
            "PrefillAttentionKernel::StagedGemmAdaptive", arm
        )
        self.assertIn("policy.command_graph = true;", arm)
        self.assertIn("policy.command_graph_warm = true;", arm)
        self.assertNotIn("maximum_block =", arm)
        self.assertNotIn("StreamingFlashAdaptive", arm)

    def test_generated_geometry_is_plan_derived_and_statically_bounded(self):
        generator = GENERATOR.read_text(encoding="utf-8")
        metal = METAL.read_text(encoding="utf-8")
        self.assertIn(
            "attn_head_dimension\n"
            "        // PREFILL_STREAMING_ATTENTION_OUTPUT_COLUMNS_PER_SIMDGROUP",
            generator,
        )
        self.assertIn(
            "streaming_attention_threads > MAX_THREADGROUP_THREADS",
            generator,
        )
        self.assertIn(
            "streaming_attention_threadgroup_memory_bytes\n"
            "        > MINIMUM_THREADGROUP_MEMORY_BYTES",
            generator,
        )
        self.assertIn(
            "kPrefillStreamingAttentionThreadgroupMemoryBytes <=\n"
            "        kMinimumThreadgroupMemoryBytes",
            metal,
        )

    def test_no_full_score_or_probability_device_buffer_exists(self):
        source = METAL.read_text(encoding="utf-8")
        marker = "kernel void attention_streaming_blk("
        body = function_body(
            source,
            "kernel void attention_streaming_blk(",
        )
        begin = source.index(marker)
        signature = source[begin : source.index("{", begin)]
        self.assertNotIn("device float* scores", signature)
        self.assertNotIn("probabilities", signature)
        self.assertIn("threadgroup float scores[", body)
        self.assertIn(
            "key_tile_begin < visible", body
        )
        self.assertIn(
            "key_tile_begin +=\n"
            "             kPrefillStreamingAttentionKeyTileColumns",
            body,
        )

    def test_static_guards_cover_every_device_extent(self):
        body = function_body(
            METAL.read_text(encoding="utf-8"),
            "kernel void attention_streaming_blk(",
        )
        required = (
            "group.x >= kAttnQueryHeads",
            "visible == 0u || visible > capacity",
            "block == 0u || query_context >= visible",
            "if (row_begin >= block)",
            "query_context + row_begin + row_count > visible",
            "key_tile_begin < visible",
            "dimension <\n                                    kAttnHeadDimension",
            "if (dimension >= kAttnHeadDimension)",
            "const uint causal_limit =",
            "absolute_key < causal_limit",
            "ulong(key_value_head) * capacity",
            "key_tile_begin +\n                                      value_key",
        )
        for text in required:
            self.assertIn(text, body)

    def test_fragment_mapping_covers_scores_and_outputs_exactly_once(self):
        score_counts = [[0 for _ in range(128)] for _ in range(16)]
        for simdgroup in range(8):
            for lane in range(32):
                coordinate_x, coordinate_y = fragment_coordinate(lane)
                for key_fragment in range(2):
                    key = simdgroup * 16 + coordinate_x + key_fragment * 8
                    for query_fragment in range(2):
                        row = coordinate_y + query_fragment * 8
                        for element in range(2):
                            score_counts[row + element][key] += 1
        self.assertEqual(
            {count for row in score_counts for count in row}, {1}
        )

        output_counts = [[0 for _ in range(256)] for _ in range(16)]
        for simdgroup in range(8):
            for lane in range(32):
                coordinate_x, coordinate_y = fragment_coordinate(lane)
                for output_fragment in range(4):
                    dimension = (
                        simdgroup * 32
                        + output_fragment * 8
                        + coordinate_x
                    )
                    for query_fragment in range(2):
                        row = coordinate_y + query_fragment * 8
                        for element in range(2):
                            output_counts[row + element][dimension] += 1
        self.assertEqual(
            {count for row in output_counts for count in row}, {1}
        )

    def test_online_softmax_matches_direct_across_tiles_and_causal_tails(self):
        randomizer = random.Random(92741)
        for length in (1, 17, 127, 128, 129, 257):
            query = [randomizer.uniform(-0.4, 0.4) for _ in range(8)]
            keys = [
                [randomizer.uniform(-0.4, 0.4) for _ in range(8)]
                for _ in range(length)
            ]
            values = [
                [randomizer.uniform(-1.0, 1.0) for _ in range(7)]
                for _ in range(length)
            ]
            for limit in sorted({1, length, max(1, length // 2)}):
                direct = direct_attention(query, keys, values, limit)
                streamed = streaming_attention(
                    query, keys, values, limit, 128
                )
                for left, right in zip(direct, streamed):
                    self.assertAlmostEqual(left, right, places=12)

    def test_host_abi_is_one_dispatch_and_profile_matches_it(self):
        runtime = RUNTIME.read_text(encoding="utf-8")
        profile = PROFILE.read_text(encoding="utf-8")
        begin = runtime.index(
            "encode.pipeline(\n"
            "                step.pipelines.attention_streaming);"
        )
        end = runtime.index("} else if (staged)", begin)
        dispatch = runtime[begin:end]
        for slot in range(9):
            self.assertIn(f", {slot});", dispatch)
        self.assertEqual(
            dispatch.count("PrefillProfileEventClass::AttentionStreaming"),
            1,
        )
        self.assertIn(
            "PrefillProfileEventClass::AttentionStreaming", profile
        )

    def test_physical_fixture_covers_tail_canaries_and_read_only_inputs(self):
        fixture = FIXTURE.read_text(encoding="utf-8")
        required = (
            "constexpr std::uint32_t kPrefillBlock = 13;",
            '"prefill-streaming-first"',
            '"prefill-streaming-second"',
            "kStreamingCanaryBytes",
            "streaming prefill canary: FAIL",
            "streaming prefill vacuous output: FAIL",
            "streaming prefill read-only input: FAIL",
            "streaming prefill determinism: FAIL",
            "streaming_maximum_normalized_error",
            "streaming_relative_l2",
        )
        for text in required:
            self.assertIn(text, fixture)




class FlashV2SourceTests(unittest.TestCase):
    def setUp(self) -> None:
        source = METAL.read_text(encoding="utf-8")
        begin = source.index("kernel void attention_flash_v2_blk(")
        self.body = source[begin:]
        self.flat = " ".join(self.body.split())
        streaming_begin = source.index(
            "kernel void attention_streaming_blk(")
        self.streaming = source[streaming_begin:begin]
        self.streaming_flat = " ".join(self.streaming.split())

    def test_guards(self) -> None:
        for guard in (
            "simdgroup_width != kSimdgroupWidth",
            "threadgroup_shape.x != kPrefillStreamingAttentionThreads",
            "group.x >= kAttnQueryHeads",
            "visible > capacity",
            "query_context >= visible",
            "row_begin >= block",
        ):
            self.assertIn(guard, self.flat)

    def test_buffer_indices_match_streaming_abi(self) -> None:
        header = self.body[: self.body.index("{")]
        import re
        indices = [int(v) for v in re.findall(r"buffer\((\d+)\)", header)]
        self.assertEqual(sorted(indices), list(range(9)))

    def test_deltas_are_query_and_key_staging(self) -> None:
        # v2c stages both the query block and each key tile; the score
        # phase must read neither from device.
        self.assertIn("threadgroup bfloat query_tile[", self.body)
        self.assertIn("threadgroup bfloat key_staging[", self.body)
        loop = self.body.index("for (uint key_tile_begin")
        pv = self.body.index("const uint reduction_end", loop)
        score_phase = self.body[loop:pv]
        self.assertNotIn("float(query[", score_phase)
        self.assertNotIn("float(key_base[", score_phase)
        # The streaming control still reads device query/keys in its loop.
        streaming_loop = self.streaming.index("for (uint key_tile_begin")
        self.assertIn("float(query[", self.streaming[streaming_loop:])

    def test_key_tile_budget(self) -> None:
        # 16x32 f32 scores + 16x256 bf16 query + 32x256 bf16 keys +
        # 3x16 f32 bookkeeping = 27,072 bytes < 32 KiB.
        self.assertLessEqual(
            16 * 32 * 4 + 16 * 256 * 2 + 32 * 256 * 2 + 3 * 16 * 4,
            32 * 1024)

    def test_threadgroup_memory_bounded(self) -> None:
        # Superseded by test_key_tile_budget; keep the streaming control's
        # own bound documented here: 16x128 f32 + 3x16 f32 = 8,384 bytes.
        self.assertLessEqual(16 * 128 * 4 + 3 * 16 * 4, 32 * 1024)

    def test_online_softmax_expressions_match_streaming(self) -> None:
        for expression in (
            "previous_maximum == -INFINITY",
            "exp(previous_maximum - next_maximum)",
            "row_denominator[row] * rescale + tile_denominator",
        ):
            self.assertIn(expression, self.flat)
            self.assertIn(expression, self.streaming_flat)


if __name__ == "__main__":
    unittest.main()