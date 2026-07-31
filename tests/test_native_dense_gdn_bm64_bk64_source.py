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


class NativeDenseGdnBm64Bk64SourceTest(unittest.TestCase):
    def test_generated_kernel_composes_both_positive_axes(self) -> None:
        source = compact(GENERATOR)
        for marker in (
            "tatara_mlx_steel_gdn_bm64_bk64_affine_qmm",
            "constexprintBM=64",
            "constexprintBN=32",
            "constexprintBK=64",
            "constexprintWM=4",
            "constexprintWN=1",
            "WM*WN*SIMD_SIZE==128",
            "BK==int(kQ4GroupSize)",
            "M%BM!=0",
            "K%BK!=0",
            "kExpectedThreadgroupBytes==13824",
            "threadgroup_shape.x!=uint(SIMD_SIZE)",
            "threadgroup_shape.y!=uint(WM)",
            "threadgroup_shape.z!=uint(WN)",
        ):
            self.assertIn(marker, source)

    def test_probe_keeps_exact_full_tile_dispatch(self) -> None:
        source = compact(PROBE)
        for marker in (
            "pipelines.steel_gdn_bm64_bk64",
            "spec.operation.reduction%kGroupSize!=0U",
            "spec.rows%kTileRows!=0U",
            "region.columns/kN1TileColumns",
            "spec.rows/kTileRows",
            ".width=kSimdgroupThreads",
            ".height=4",
            ".depth=1",
            'mode=="--gpu-steel-gdn-bm64-bk64-component"',
            'mode=="--benchmark-steel-gdn-bm64-bk64"',
        ):
            self.assertIn(marker, source)

    def test_reuses_fail_closed_exactness_and_timing_gate(self) -> None:
        source = compact(PROBE)
        for marker in (
            "Arm::SteelGdnBm64Bk64",
            "kExitSteelGdnBm64Bk64Mismatch",
            "input_identity(resources)!=original_inputs",
            "std::equal(treatment_body.begin(),treatment_body.end(),"
            "resources.exact_snapshot.begin())",
            "kMeasuredSamplesPerArm",
            "control_ranges_overlap(control_samples)",
            '"tatara_mlx_steel_gdn_bm64_bk64_component"',
        ):
            self.assertIn(marker, source)

    def test_composition_is_probe_only_and_correctness_first(self) -> None:
        self.assertNotIn("bm64_bk64", PREFILL_STEP.lower())
        source = compact(PROBE)
        component = source.index(
            'if(mode=="--gpu-steel-gdn-bm64-bk64-component")'
        )
        benchmark = source.index(
            'if(mode=="--benchmark-steel-gdn-bm64-bk64")'
        )
        self.assertLess(component, benchmark)
        self.assertIn("timingsamples:0", source[component:benchmark])


if __name__ == "__main__":
    unittest.main()
