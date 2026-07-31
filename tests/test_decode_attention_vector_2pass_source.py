"""Static crash-safety and lifecycle contract for A25 vector decode."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
KERNEL_PATH = ROOT / "src/backend/metal/kernels/attention.metal"
STEP_HEADER_PATH = ROOT / "include/tatara/runtime/decode_step.h"
STEP_PATH = ROOT / "src/runtime/decode_step.cpp"
HARNESS_HEADER_PATH = ROOT / "tools/native/decode_harness.h"
HARNESS_PATH = ROOT / "tools/native/decode_harness.cpp"
PERF_PATH = ROOT / "tools/native/decode_perf_probe.cpp"
FIXTURE_PATH = ROOT / "tools/native/attention_fixture_probe.cpp"
IDENTITY_PATH = ROOT / "src/runtime/execution_identity.cpp"
PREFILL_LIBRARY_PATH = ROOT / "tools/native/prefill_library_probe.cpp"
SERVE_PATH = ROOT / "tools/native/tatara_serve.cpp"


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


class DecodeAttentionVector2PassSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = KERNEL_PATH.read_text()
        cls.part = braced_function(
            source, "kernel void attention_decode_vector_2pass_part("
        )
        cls.combine = braced_function(
            source, "kernel void attention_decode_vector_2pass_combine("
        )
        cls.step_header = STEP_HEADER_PATH.read_text()
        step_source = STEP_PATH.read_text()
        cls.blocks = braced_function(
            step_source, "std::uint32_t attention_vector_blocks("
        )
        cls.encode = braced_function(
            step_source, "void encode_attention_decode("
        )
        cls.harness_header = HARNESS_HEADER_PATH.read_text()
        cls.harness = HARNESS_PATH.read_text()
        cls.perf = PERF_PATH.read_text()
        cls.fixture = FIXTURE_PATH.read_text()
        cls.identity = IDENTITY_PATH.read_text()
        cls.prefill_library = PREFILL_LIBRARY_PATH.read_text()
        cls.serve = SERVE_PATH.read_text()

    def test_first_pass_abi_and_inputs_are_exact(self) -> None:
        kernel = compact(self.part)
        expected = (
            "deviceconstbfloat*q[[buffer(0)]],"
            "deviceconstbfloat*keys[[buffer(1)]],"
            "deviceconstbfloat*values[[buffer(2)]],"
            "devicebfloat*partials[[buffer(3)]],"
            "devicefloat*sums[[buffer(4)]],"
            "devicefloat*maxs[[buffer(5)]],"
            "constantuint&context[[buffer(6)]],"
            "constantuint&capacity[[buffer(7)]],"
            "constantuint&blocks[[buffer(8)]]"
        )
        self.assertIn(expected, kernel)

    def test_first_pass_has_complete_independent_ownership_and_no_barrier(
        self,
    ) -> None:
        kernel = compact(self.part)
        self.assertNotIn("threadgroup_barrier", kernel)
        self.assertNotIn("threadgroup ", self.part)
        self.assertIn("constuintlocal_head=tpos.y;", kernel)
        self.assertIn("constuinthead=kv*8u+local_head;", kernel)
        self.assertIn("constuintdimension_base=lane*8u;", kernel)
        self.assertIn(
            "for(uintposition=block;position<n;position+=blocks)", kernel
        )
        self.assertIn("score=simd_sum(score);", kernel)
        self.assertIn(
            "(ulong(head)*ulong(blocks)+ulong(block))*"
            "ulong(kAttnHeadDimension)",
            kernel,
        )
        self.assertIn("sums[scalar]=part_sum;", kernel)
        self.assertIn("maxs[scalar]=part_max;", kernel)

    def test_second_pass_is_bounded_32_simdgroup_transpose(self) -> None:
        kernel = compact(self.combine)
        self.assertIn("threadgroupfloattranspose[32u*32u];", kernel)
        self.assertEqual(self.combine.count("threadgroup_barrier("), 2)
        self.assertIn(
            "for(uintblock_base=0u;block_base<blocks;"
            "block_base+=32u)",
            kernel,
        )
        self.assertIn(
            "for(uintblock=simdgroup;block<blocks;block+=32u)",
            kernel,
        )
        self.assertIn(
            "constuintdimension=simdgroup*8u+i;", kernel
        )
        self.assertIn(
            "out[head*kAttnHeadDimension+dimension]", kernel
        )
        self.assertEqual(32 * 32 * 4, 4096)
        self.assertEqual(32 * 32, 1024)

    def test_block_policy_is_scratch_derived_not_an_admission_bound(self) -> None:
        blocks = compact(self.blocks)
        encode = compact(self.encode)
        self.assertIn(
            "step.geometry.attn_record_scratch_bytes/"
            "step.geometry.attn_query_bytes",
            blocks,
        )
        self.assertIn("kReductionCohort=32", blocks)
        self.assertIn("kMaximumScheduledBlocks=1024", blocks)
        self.assertIn("returnstd::min(desired,fitting_cohorts*", blocks)
        self.assertIn("IndependentHeadVector2Pass", encode)
        self.assertIn("vector_blocks!=0", encode)
        self.assertNotIn(
            "IndependentHeadVector2Pass",
            (ROOT / "src/runtime/serving_capacity.cpp").read_text(),
        )
        self.assertNotIn(
            "IndependentHeadVector2Pass",
            (ROOT / "include/tatara/runtime/serving_capacity.h").read_text(),
        )

    def test_runtime_binds_disjoint_planes_and_exact_dispatches(self) -> None:
        encode = compact(self.encode)
        vector_begin = encode.index(
            "DecodeAttentionSplitPolicy::IndependentHeadVector2Pass"
        )
        vector_end = encode.index("constboolfused_score_value", vector_begin)
        vector = encode[vector_begin:vector_end]
        self.assertIn("step.pipelines.attention_vector_part", vector)
        self.assertIn("encode.buffer(step.attn_partials,0,3);", vector)
        self.assertIn("encode.buffer(step.attn_weights,0,4);", vector)
        self.assertIn(
            "encode.buffer(step.attn_weights,scalar_plane_bytes,5);",
            vector,
        )
        self.assertIn(
            "encode.dispatch(split.value_groups,kSimdgroupThreads,"
            "vector_blocks,split.value_head_threads);",
            vector,
        )
        self.assertIn("step.pipelines.attention_vector_combine", vector)
        self.assertIn(
            "encode.dispatch(shape.attention_head.groups,1024);", vector
        )
        self.assertTrue(vector.rstrip().endswith("return;}"))

    def test_probe_selector_identity_and_zero_cb_inventory_are_closed(
        self,
    ) -> None:
        self.assertIn("IndependentHeadVector2Pass", self.step_header)
        self.assertIn("IndependentHeadVector2Pass", self.harness_header)
        self.assertIn(
            '"attention_decode_vector_2pass_part"', self.harness
        )
        self.assertIn(
            '"attention_decode_vector_2pass_combine"', self.harness
        )
        self.assertIn('"score-vector-2pass"', self.perf)
        self.assertIn(
            "&pipelines.attention_vector_part", self.identity
        )
        self.assertIn(
            "&pipelines.attention_vector_combine", self.identity
        )
        self.assertIn(
            "std::array<const backend::metal::MetalComputePipeline*, 24>",
            self.identity,
        )
        self.assertIn(
            '"attention_decode_vector_2pass_part"',
            self.prefill_library,
        )
        self.assertIn(
            '"attention_decode_vector_2pass_combine"',
            self.prefill_library,
        )

    def test_a28_product_policy_is_a_crossover_not_a_limit(self) -> None:
        header = compact(self.step_header)
        encode = compact(self.encode)
        harness_header = compact(self.harness_header)
        harness = compact(self.harness)
        identity = compact(self.identity)
        serve = compact(self.serve)
        self.assertIn("AdaptiveVector2Pass", header)
        self.assertIn(
            "context>=step.pipelines.vector_minimum_context", encode
        )
        self.assertLess(
            encode.index("vector_minimum_context"),
            encode.index("fused_score_value_minimum_context"),
        )
        self.assertIn(
            "kQualifiedVectorMinimumContext=8000", harness_header
        )
        self.assertIn("DecodeAttentionScoreKernel::AdaptiveA23", harness)
        self.assertIn(
            "DecodeAttentionSplitPolicy::AdaptiveGqa8ScoreValue", harness
        )
        self.assertIn(
            "DecodeAttentionSplitPolicy::AdaptiveVector2Pass", harness
        )
        self.assertIn(
            "pipelines.vector_minimum_context="
            "kQualifiedVectorMinimumContext",
            harness,
        )
        self.assertIn("pipelines.vector_minimum_context", identity)
        self.assertIn('"score-adaptive-a23"', self.perf)
        self.assertIn("vector-at-or-after=%u", self.serve)
        self.assertIn(
            "no capacity or output limit", self.serve
        )
        for path in (
            ROOT / "src/runtime/serving_capacity.cpp",
            ROOT / "include/tatara/runtime/serving_capacity.h",
        ):
            source = path.read_text()
            self.assertNotIn("vector_minimum_context", source)
            self.assertNotIn("kQualifiedVectorMinimumContext", source)

    def test_component_gate_reports_full_numerical_family(self) -> None:
        fixture = compact(self.fixture)
        self.assertIn("kVectorBlocks=32", fixture)
        self.assertIn(
            ".width=32,.height=8,.depth=1", fixture
        )
        self.assertIn(
            ".width=1024,.height=1,.depth=1", fixture
        )
        self.assertIn(
            "kVectorMaximumAbsoluteError=0.015625f", fixture
        )
        self.assertIn("vector_mismatches", self.fixture)
        self.assertIn("vector_nonfinite", self.fixture)
        self.assertIn("vector_maximum_absolute_error", self.fixture)
        self.assertIn("vector_inputs_unchanged", self.fixture)
        self.assertIn(
            "if(vector_nonfinite!=0||"
            "vector_maximum_absolute_error>"
            "kVectorMaximumAbsoluteError||"
            "!vector_inputs_unchanged){return147;}",
            fixture,
        )

    def test_first_target_byte_proof_fits_existing_scratch(self) -> None:
        partial_bytes = 128 * 512 * 256 * 2
        scalar_bytes = 128 * 512 * 4
        existing_bytes = 128 * 257 * 258 * 4
        self.assertEqual(partial_bytes, 33_554_432)
        self.assertEqual(scalar_bytes, 262_144)
        self.assertEqual(existing_bytes, 33_948_672)
        self.assertLessEqual(partial_bytes, existing_bytes)
        self.assertLessEqual(2 * scalar_bytes, existing_bytes)


if __name__ == "__main__":
    unittest.main()
