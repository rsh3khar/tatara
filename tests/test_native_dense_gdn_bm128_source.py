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


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


class NativeDenseGdnBm128SourceTest(unittest.TestCase):
    def test_generated_kernel_freezes_row_reuse_geometry(self) -> None:
        source = compact(GENERATOR)
        for marker in (
            "tatara_mlx_steel_gdn_bm128_affine_qmm",
            "constexprintBM=128",
            "constexprintBN=32",
            "constexprintBK=32",
            "constexprintWM=8",
            "constexprintWN=1",
            "WM*WN*SIMD_SIZE==256",
            "M%BM!=0",
            "K%BK!=0",
            "kExpectedThreadgroupBytes==12800",
            "threadgroup_shape.x!=uint(SIMD_SIZE)",
            "threadgroup_shape.y!=uint(WM)",
            "threadgroup_shape.z!=uint(WN)",
            "output_stride!=int(kGdnProjectionRows)",
        ):
            self.assertIn(marker, source)

    def test_probe_dispatches_exact_full_tiles_for_all_regions(self) -> None:
        source = compact(PROBE)
        for marker in (
            "constexprstd::uint32_tkTileRows=128U",
            "spec.rows%kTileRows!=0U",
            "pipelines.steel_gdn_bm128",
            "region.columns/kN1TileColumns",
            "spec.rows/kTileRows",
            ".width=kSimdgroupThreads",
            ".height=8",
            ".depth=1",
            'mode=="--gpu-steel-gdn-bm128-component"',
            'mode=="--benchmark-steel-gdn-bm128"',
        ):
            self.assertIn(marker, source)

    def test_exactness_and_measurement_contract_are_reused(self) -> None:
        source = compact(PROBE)
        for marker in (
            "Arm::SteelGdnBm128",
            "kExitSteelGdnBm128Mismatch",
            "input_identity(resources)!=original_inputs",
            "std::mismatch(treatment_body.begin(),treatment_body.end(),"
            "resources.exact_snapshot.begin())",
            "kMeasuredSamplesPerArm",
            "control_ranges_overlap(control_samples)",
            '"tatara_mlx_steel_gdn_bm128_component"',
        ):
            self.assertIn(marker, source)

    def test_candidate_is_probe_only(self) -> None:
        self.assertNotIn("bm128", PREFILL_STEP.lower())
        source = compact(PROBE)
        component = source.index(
            'if(mode=="--gpu-steel-gdn-bm128-component")'
        )
        benchmark = source.index(
            'if(mode=="--benchmark-steel-gdn-bm128")'
        )
        self.assertLess(component, benchmark)
        self.assertIn("timingsamples:0", source[component:benchmark])


if __name__ == "__main__":
    unittest.main()
