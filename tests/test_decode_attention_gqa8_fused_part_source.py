"""Static safety, ABI, exactness, and adaptive policy contract for A21-A23."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "src/backend/metal/kernels/attention.metal"
STEP_HEADER_PATH = ROOT / "include/tatara/runtime/decode_step.h"
STEP_PATH = ROOT / "src/runtime/decode_step.cpp"
FIXTURE_PATH = ROOT / "tools/native/attention_fixture_probe.cpp"
HARNESS_HEADER_PATH = ROOT / "tools/native/decode_harness.h"
HARNESS_PATH = ROOT / "tools/native/decode_harness.cpp"
PERF_PATH = ROOT / "tools/native/decode_perf_probe.cpp"
IDENTITY_PATH = ROOT / "src/runtime/execution_identity.cpp"


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


class DecodeAttentionGqa8FusedPartSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = KERNEL_PATH.read_text()
        cls.kernel = braced_function(
            source,
            "kernel void attention_decode_scores_values_gqa8(",
        )
        cls.step_header = STEP_HEADER_PATH.read_text()
        cls.step = braced_function(
            STEP_PATH.read_text(), "void encode_attention_decode("
        )
        cls.fixture = FIXTURE_PATH.read_text()
        cls.harness_header = HARNESS_HEADER_PATH.read_text()
        cls.harness = HARNESS_PATH.read_text()
        cls.perf = PERF_PATH.read_text()
        cls.identity = IDENTITY_PATH.read_text()

    def test_buffer_abi_is_exact_and_read_only_inputs_stay_const(self) -> None:
        kernel = compact(self.kernel)
        expected = (
            "deviceconstbfloat*q[[buffer(0)]],"
            "deviceconstbfloat*keys[[buffer(1)]],"
            "deviceconstbfloat*values[[buffer(2)]],"
            "constantuint&context[[buffer(3)]],"
            "constantuint&capacity[[buffer(4)]],"
            "constantuint&part[[buffer(5)]],"
            "devicefloat*partials[[buffer(6)]],"
            "constantuint&nparts[[buffer(7)]]"
        )
        self.assertIn(expected, kernel)

    def test_threads_and_threadgroup_storage_fit_static_device_floor(self) -> None:
        self.assertIn("if (tg.x >= kAttnKvHeads)", self.kernel)
        self.assertLess(
            self.kernel.index("if (tg.x >= kAttnKvHeads)"),
            self.kernel.index("threadgroup_barrier("),
        )
        self.assertIn("threadgroup float scores[8][256];", self.kernel)
        self.assertNotIn("threadgroup float red[4][256];", self.kernel)
        self.assertNotIn(
            "threadgroup bfloat value_tile[32u * 256u];", self.kernel
        )
        self.assertIn(
            "threadgroup uint shared_workspace[4096];", self.kernel
        )
        self.assertIn(
            "reinterpret_cast<threadgroup float*>(shared_workspace)",
            self.kernel,
        )
        self.assertIn(
            "reinterpret_cast<threadgroup bfloat*>(shared_workspace)",
            self.kernel,
        )
        threadgroup_bytes = 8 * 256 * 4 + 4096 * 4
        self.assertEqual(threadgroup_bytes, 24_576)
        self.assertLess(threadgroup_bytes, 32_768)
        fixture = compact(self.fixture)
        self.assertIn(
            compact(".width = 256, .height = 4, .depth = 1"), fixture
        )
        self.assertEqual(256 * 4, 1024)
        self.assertIn(
            compact(
                ".width = kKvHeads, .height = kNParts, .depth = 1"
            ),
            fixture,
        )

    def test_all_eight_scores_keep_current_dot_and_tree_order(self) -> None:
        kernel = compact(self.kernel)
        self.assertIn(
            "for(uintt=cohort*8u+sg;t<count;t+=32u)", kernel
        )
        for head in range(8):
            self.assertIn(f"floatdot{head}=0.0f;", kernel)
            self.assertIn(f"dot{head}=simd_sum(dot{head});", kernel)
            self.assertIn(
                f"scores[{head}][t]=dot{head}*kAttnScale;", kernel
            )
        self.assertIn(
            "for(uintphase=0u;phase<2u;++phase)", kernel
        )
        self.assertEqual(
            self.kernel.count("for (uint off = 128u; off; off >>= 1u)"),
            2,
        )
        self.assertIn("dst[256] = part_max;", self.kernel)
        self.assertIn("dst[257] = red[cohort * 256u];", self.kernel)

    def test_value_mapping_covers_every_head_dimension_in_key_order(self) -> None:
        kernel = compact(self.kernel)
        value_phase = kernel.index(
            "constuintflat=tid+256u*cohort;"
        )
        self.assertLess(kernel.rfind("red[", 0, value_phase), value_phase)
        self.assertNotIn("red[", kernel[value_phase:])
        self.assertIn("constuintflat=tid+256u*cohort;", kernel)
        self.assertIn("constuintlocal_head=flat>>7u;", kernel)
        self.assertIn("constuintoutput_dim=flat&127u;", kernel)
        self.assertIn(
            "for(uintvb=0u;vb<count;vb+=32u)", kernel
        )
        self.assertIn("for(uintp=0u;p<vn;++p)", kernel)
        self.assertIn(
            "constfloatweight=scores[local_head][vb+p];", kernel
        )
        self.assertIn("dst[output_dim]=acc0;", kernel)
        self.assertIn("dst[output_dim+128u]=acc1;", kernel)

    def test_runtime_fused_arm_removes_probability_buffer_and_value_pass(self) -> None:
        step = compact(self.step)
        fused_begin = step.index(
            "DecodeAttentionSplitPolicy::FusedGqa8ScoreValue"
        )
        separate_begin = step.index("}else{", fused_begin)
        fused = step[fused_begin:separate_begin]
        self.assertIn(
            "step.pipelines.attention_scores_values_fused", fused
        )
        self.assertIn("encode.buffer(values,0,2);", fused)
        self.assertIn("encode.buffer(step.attn_partials,0,6);", fused)
        self.assertIn(
            "encode.dispatch(split.value_groups,split.score_threads,"
            "partitions,split.score_cohort_threads);",
            fused,
        )
        self.assertNotIn("attn_weights", fused)
        self.assertNotIn("attention_values", fused)
        separate = step[separate_begin:]
        self.assertIn("step.attn_weights", separate)
        self.assertIn("step.pipelines.attention_values", separate)
        self.assertIn(
            "SeparateScoreValue", compact(self.step_header)
        )

    def test_adaptive_policy_is_a_crossover_not_a_capacity_bound(self) -> None:
        header = compact(self.step_header)
        step = compact(self.step)
        harness_header = compact(self.harness_header)
        harness = compact(self.harness)
        identity = compact(self.identity)
        self.assertIn("AdaptiveGqa8ScoreValue", header)
        self.assertIn(
            "context>=step.pipelines."
            "fused_score_value_minimum_context",
            step,
        )
        self.assertIn(
            "kQualifiedFusedScoreValueMinimumContext=15000",
            harness_header,
        )
        self.assertIn(
            "DecodeAttentionScoreKernel::Adaptive", harness
        )
        self.assertIn(
            "pipelines.fused_score_value_minimum_context="
            "kQualifiedFusedScoreValueMinimumContext",
            harness,
        )
        self.assertIn(
            "&pipelines.attention_scores_values_fused", identity
        )
        self.assertIn(
            "pipelines.fused_score_value_minimum_context", identity
        )
        self.assertNotIn(
            "fused_score_value_minimum_context", compact(
                (ROOT / "include/tatara/runtime/serving_capacity.h").read_text()
            )
        )

    def test_fixture_and_full_token_instruments_are_fail_closed(self) -> None:
        fixture = compact(self.fixture)
        self.assertIn(
            '"attention_decode_scores_values_gqa8"', self.fixture
        )
        self.assertIn("partials2_fused==partials2_device", fixture)
        self.assertIn("partials1_fused==partials1_control", fixture)
        self.assertIn("fused_inputs_unchanged", fixture)
        self.assertIn(
            "if(!fused_two_part_exact||!fused_one_part_exact||"
            "!fused_inputs_unchanged){return145;}",
            fixture,
        )
        self.assertIn("Gqa8FusedPart", self.harness_header)
        self.assertIn(
            '"attention_decode_scores_values_gqa8"', self.harness
        )
        self.assertIn(
            "FusedGqa8ScoreValue", self.harness
        )
        self.assertIn('"score-value-gqa8-fused"', self.perf)
        self.assertIn(
            "required_capacity > plan.tokenizer.maximum_context", self.perf
        )


if __name__ == "__main__":
    unittest.main()
