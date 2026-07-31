import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
GENERATOR = (ROOT / "tools/generate_kernel_library.py").read_text(
    encoding="utf-8"
)
PROBE = (
    ROOT / "tools/native/native_dense_qgemm_perf_probe.cpp"
).read_text(encoding="utf-8")
PREFILL_STEP = (ROOT / "src/runtime/prefill_step.cpp").read_text(
    encoding="utf-8"
)
PREFILL_PLAN = (
    ROOT / "src/runtime/prefill_profile_plan.cpp"
).read_text(encoding="utf-8")
PREFILL_HEADER = (
    ROOT / "include/tatara/runtime/prefill_step.h"
).read_text(encoding="utf-8")
BLOCK_PROBE = (
    ROOT / "tools/native/block_prefill_probe.cpp"
).read_text(encoding="utf-8")


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


class NativeDenseGdnBm64Wm2Wn2SourceTest(unittest.TestCase):
    def test_generated_kernel_freezes_simd_grid(self) -> None:
        source = compact(GENERATOR)
        for marker in (
            "tatara_mlx_steel_gdn_bm64_wm2_wn2_affine_qmm",
            "constexprintBM=64",
            "constexprintBN=32",
            "constexprintBK=32",
            "constexprintWM=2",
            "constexprintWN=2",
            "WM*WN*SIMD_SIZE==128",
            "M%BM!=0",
            "K%BK!=0",
            "kExpectedThreadgroupBytes==7680",
            "threadgroup_shape.x!=uint(SIMD_SIZE)",
            "threadgroup_shape.y!=uint(WM)",
            "threadgroup_shape.z!=uint(WN)",
            "output_stride!=int(kGdnProjectionRows)",
        ):
            self.assertIn(marker, source)

    def test_probe_dispatches_full_tiles_for_all_regions(self) -> None:
        source = compact(PROBE)
        for marker in (
            "encode_steel_gdn_bm64_wm2_wn2",
            "constexprstd::uint32_tkTileRows=64U",
            "pipelines.steel_gdn_bm64_wm2_wn2",
            "region.columns/kN1TileColumns",
            "spec.rows/kTileRows",
            ".width=kSimdgroupThreads",
            ".height=2",
            ".depth=2",
            'mode=="--gpu-steel-gdn-bm64-wm2-wn2-component"',
            'mode=="--benchmark-steel-gdn-bm64-wm2-wn2"',
            '"--benchmark-steel-gdn-bm64-wm2-wn2-reverse"',
            "(scheduled=='A')!=reverse_order",
        ):
            self.assertIn(marker, source)

    def test_exactness_and_measurement_contract_are_reused(self) -> None:
        source = compact(PROBE)
        for marker in (
            "Arm::SteelGdnBm64Wm2Wn2",
            "kExitSteelGdnBm64Wm2Wn2Mismatch",
            "input_identity(resources)!=original_inputs",
            "std::mismatch(treatment_body.begin(),treatment_body.end(),"
            "resources.exact_snapshot.begin())",
            "kMeasuredSamplesPerArm",
            "control_ranges_overlap(control_samples)",
            '"tatara_mlx_steel_gdn_bm64_wm2_wn2_component"',
            '"treatment_dispatches=4"',
        ):
            self.assertIn(marker, source)

    def test_candidate_is_default_off_and_gdn_only(self) -> None:
        runtime = compact(PREFILL_STEP)
        profile_plan = compact(PREFILL_PLAN)
        for marker in (
            "native_dense_steel_gdn_bm64_wm2_wn2",
            "EventClass=="
            "PrefillProfileEventClass::GdnProjection",
            "body_rows=rows/kBm64TileRows*kBm64TileRows",
            "body_rows!=0U",
            "output_columns/"
            "kKernelLibraryNativeDenseQgemmN1TileColumns",
            ".height=body_rows/kBm64TileRows",
        ):
            self.assertIn(marker, runtime)
        for marker in (
            "gdn_projection_dispatches="
            "policy.native_dense_steel_gdn_bm64_wm2_wn2&&"
            "chunk.rows>=kBm64TileRows&&"
            "chunk.rows%kBm64TileRows!=0U?8U:4U",
            "PrefillProfileEventClass::GdnProjection,"
            "gdn_projection_dispatches",
        ):
            self.assertIn(marker, profile_plan)
        source = compact(PROBE)
        component = source.index(
            'if(mode=="--gpu-steel-gdn-bm64-wm2-wn2-component")'
        )
        benchmark = source.index(
            'mode=="--benchmark-steel-gdn-bm64-wm2-wn2"'
        )
        self.assertLess(component, benchmark)
        self.assertIn("timingsamples:0", source[component:benchmark])

    def test_graph_policy_is_explicit_and_falls_back_on_tails(self) -> None:
        header = compact(PREFILL_HEADER)
        runtime = compact(PREFILL_STEP)
        probe = compact(BLOCK_PROBE)
        self.assertIn(
            "boolnative_dense_steel_gdn_bm64_wm2_wn2{false};",
            header,
        )
        for marker in (
            "body_rows=rows/kBm64TileRows*kBm64TileRows",
            "fallback_row_begin=body_rows",
            "if(fallback_row_begin==rows){return;}",
            "steel_rows=rows-fallback_row_begin",
            "activation_offset+std::uint64_t{fallback_row_begin}*"
            "activation_row_stride_elements*kBf16Bytes",
            "output_offset+std::uint64_t{fallback_row_begin}*"
            "output_row_stride_elements*kBf16Bytes",
        ):
            self.assertIn(marker, runtime)
        self.assertLess(
            runtime.index(
                "native_dense_steel_gdn_bm64_wm2_wn2&&"
            ),
            runtime.index(
                "if(step.policy.native_dense_steel)"
            ),
        )
        for marker in (
            "kKernelLibraryMlxSteelGdnBm64Wm2Wn2KernelName",
            '"layer2048fast-steel-full-graph-'
            'bm64-wm2-wn2-compile"',
            '"layer2048fast-steel-full-graph-'
            'bm64-wm2-wn2-warm"',
        ):
            self.assertIn(marker, probe)


if __name__ == "__main__":
    unittest.main()
