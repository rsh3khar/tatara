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


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


class NativeDenseGdnBk64SourceTest(unittest.TestCase):
    def test_generated_kernel_matches_one_quant_group(self) -> None:
        source = compact(GENERATOR)
        for marker in (
            "tatara_mlx_steel_gdn_bk64_affine_qmm",
            "constexprintBM=32",
            "constexprintBN=32",
            "constexprintBK=64",
            "constexprintWM=2",
            "constexprintWN=2",
            "WM*WN*SIMD_SIZE==128",
            "BK==int(kQ4GroupSize)",
            "K%BK!=0",
            "kExpectedThreadgroupBytes==9216",
            "threadgroup_shape.x!=uint(SIMD_SIZE)",
            "threadgroup_shape.y!=uint(WM)",
            "threadgroup_shape.z!=uint(WN)",
        ):
            self.assertIn(marker, source)

    def test_probe_preserves_grid_and_uses_bk64_for_all_regions(self) -> None:
        source = compact(PROBE)
        for marker in (
            "pipelines.steel_gdn_bk64",
            "region.columns/kN1TileColumns",
            "spec.rows/kN1TileRows",
            ".width=kSimdgroupThreads",
            ".height=2",
            ".depth=2",
            'mode=="--gpu-steel-gdn-bk64-component"',
            'mode=="--benchmark-steel-gdn-bk64"',
        ):
            self.assertIn(marker, source)

    def test_only_bk64_trims_redundant_edge_barriers(self) -> None:
        source = compact(GENERATOR)
        self.assertIn(
            "constbooltrim_edge_barriers=false", source
        )
        self.assertEqual(
            source.count("if(!trim_edge_barriers||k!=0)"), 1
        )
        self.assertIn("if(!trim_edge_barriers){", source)
        self.assertIn(
            "threadgroup_barrier(mem_flags::mem_threadgroup);",
            source,
        )
        self.assertIn(
            "bfloat16_t,64,4,true,BM,BK,BN,WM,WN,true",
            source,
        )
        self.assertIn(
            "qmm_t_impl<bfloat16_t,64,4,true,BM,BK,BN>(",
            source,
        )

    def test_exactness_and_measurement_contract_are_reused(self) -> None:
        source = compact(PROBE)
        for marker in (
            "Arm::SteelGdnBk64",
            "kExitSteelGdnBk64Mismatch",
            "input_identity(resources)!=original_inputs",
            "std::equal(treatment_body.begin(),treatment_body.end(),"
            "resources.exact_snapshot.begin())",
            "kMeasuredSamplesPerArm",
            "control_ranges_overlap(control_samples)",
            '"tatara_mlx_steel_gdn_bk64_component"',
            '"treatment_dispatches=4"',
        ):
            self.assertIn(marker, source)

    def test_component_precedes_benchmark(self) -> None:
        source = compact(PROBE)
        component = source.index(
            'if(mode=="--gpu-steel-gdn-bk64-component")'
        )
        benchmark = source.index(
            'if(mode=="--benchmark-steel-gdn-bk64")'
        )
        self.assertLess(component, benchmark)
        self.assertIn("timingsamples:0", source[component:benchmark])


if __name__ == "__main__":
    unittest.main()
