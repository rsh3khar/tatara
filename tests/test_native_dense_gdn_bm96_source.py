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


class NativeDenseGdnBm96SourceTest(unittest.TestCase):
    def test_generated_body_freezes_even_six_simdgroup_geometry(self) -> None:
        source = compact(GENERATOR)
        for marker in (
            "tatara_mlx_steel_gdn_bm96_affine_qmm",
            "constexprintBM=96",
            "constexprintBN=32",
            "constexprintBK=32",
            "constexprintWM=6",
            "constexprintWN=1",
            "WM*WN*SIMD_SIZE==192",
            "M%BM!=0",
            "K%BK!=0",
            "kExpectedThreadgroupBytes==10240",
            "threadgroup_shape.y!=uint(WM)",
            "threadgroup_shape.z!=uint(WN)",
        ):
            self.assertIn(marker, source)

    def test_probe_covers_2016_body_and_exact_tail(self) -> None:
        source = compact(PROBE)
        for marker in (
            "constexprstd::uint32_tkBodyTileRows=96U",
            "constexprstd::uint32_tkTailTileRows=32U",
            "body_rows=(spec.rows/kBodyTileRows)*kBodyTileRows",
            "tail_rows=spec.rows-body_rows",
            "tail_rows!=kTailTileRows",
            "pipelines.steel_gdn_bm96",
            "pipelines.steel",
            ".height=body_rows/kBodyTileRows",
            ".height=tail_rows/kTailTileRows",
            ".height=6",
            ".depth=1",
            ".height=2",
            ".depth=2",
            "std::uint64_t{body_rows}*spec.operation.columns",
            "std::uint64_t{body_rows}*spec.operation.reduction",
        ):
            self.assertIn(marker, source)

    def test_reuses_fail_closed_component_and_timing_contract(self) -> None:
        source = compact(PROBE)
        for marker in (
            "Arm::SteelGdnBm96",
            "kExitSteelGdnBm96Mismatch",
            "input_identity(resources)!=original_inputs",
            "first_mismatch=std::mismatch(",
            "kMeasuredSamplesPerArm",
            "control_ranges_overlap(control_samples)",
            '"tatara_mlx_steel_gdn_bm96_component"',
            '"treatment_dispatches=8"',
        ):
            self.assertIn(marker, source)

    def test_candidate_is_probe_only_and_correctness_first(self) -> None:
        self.assertNotIn("bm96", PREFILL_STEP.lower())
        source = compact(PROBE)
        component = source.index(
            'if(mode=="--gpu-steel-gdn-bm96-component")'
        )
        benchmark = source.index(
            'if(mode=="--benchmark-steel-gdn-bm96")'
        )
        self.assertLess(component, benchmark)
        self.assertIn("timingsamples:0", source[component:benchmark])


if __name__ == "__main__":
    unittest.main()
