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


class NativeDenseGdnBn64SourceTest(unittest.TestCase):
    def test_generated_kernel_freezes_wide_tile_and_resources(self) -> None:
        source = compact(GENERATOR)
        for marker in (
            "tatara_mlx_steel_gdn_bn64_affine_qmm",
            "constexprintBM=32",
            "constexprintBN=64",
            "constexprintBK=32",
            "constexprintWM=2",
            "constexprintWN=4",
            "WM*WN*SIMD_SIZE==256",
            "kExpectedThreadgroupBytes==7680",
            "threadgroup_shape.x!=uint(SIMD_SIZE)",
            "threadgroup_shape.y!=uint(WM)",
            "threadgroup_shape.z!=uint(WN)",
            "output_stride!=int(kGdnProjectionRows)",
        ):
            self.assertIn(marker, source)

    def test_dense_control_instantiation_remains_explicit(self) -> None:
        source = compact(GENERATOR)
        self.assertIn(
            "qmm_t_impl<bfloat16_t,64,4,true,BM,BK,BN>(",
            source,
        )
        self.assertIn('"qmm_t_impl<"', source)
        self.assertIn(
            '"bfloat16_t,64,4,true,BM,BK,BN,WM,WN>("',
            source,
        )

    def test_probe_uses_bn64_only_for_qkv_and_z(self) -> None:
        source = compact(PROBE)
        for marker in (
            "constexprstd::uint32_tkWideTileColumns=64U",
            "constboolwide=index<2U",
            "wide?pipelines.steel_gdn_bn64:pipelines.steel",
            "wide?4U:2U",
            "region.columns%tile_columns!=0U",
            'mode=="--gpu-steel-gdn-bn64-component"',
            'mode=="--benchmark-steel-gdn-bn64"',
        ):
            self.assertIn(marker, source)

    def test_exactness_and_measurement_contract_are_reused(self) -> None:
        source = compact(PROBE)
        for marker in (
            "Arm::SteelGdnBn64",
            "kExitSteelGdnBn64Mismatch",
            "input_identity(resources)!=original_inputs",
            "std::equal(treatment_body.begin(),treatment_body.end(),"
            "resources.exact_snapshot.begin())",
            "kMeasuredSamplesPerArm",
            "control_ranges_overlap(control_samples)",
            "treatment_dispatches=4",
            "arm==treatment&&!treatment_determinism_checked",
        ):
            self.assertIn(marker, source)

    def test_component_precedes_benchmark(self) -> None:
        source = compact(PROBE)
        component = source.index(
            'if(mode=="--gpu-steel-gdn-bn64-component")'
        )
        benchmark = source.index(
            'if(mode=="--benchmark-steel-gdn-bn64")'
        )
        self.assertLess(component, benchmark)
        self.assertIn("timingsamples:0", source[component:benchmark])


if __name__ == "__main__":
    unittest.main()
