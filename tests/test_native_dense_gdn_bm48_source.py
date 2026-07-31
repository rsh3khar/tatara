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


class NativeDenseGdnBm48SourceTest(unittest.TestCase):
    def test_generated_body_freezes_96_thread_geometry(self) -> None:
        source = compact(GENERATOR)
        for marker in (
            "tatara_mlx_steel_gdn_bm48_affine_qmm",
            "constexprintBM=48",
            "constexprintBN=32",
            "constexprintBK=32",
            "constexprintWM=3",
            "constexprintWN=1",
            "WM*WN*SIMD_SIZE==96",
            "M%BM!=0",
            "K%BK!=0",
            "kExpectedThreadgroupBytes==6400",
            "threadgroup_shape.y!=uint(WM)",
            "threadgroup_shape.z!=uint(WN)",
        ):
            self.assertIn(marker, source)

    def test_probe_covers_exact_body_and_tail(self) -> None:
        source = compact(PROBE)
        for marker in (
            "constexprstd::uint32_tkBodyTileRows=48U",
            "constexprstd::uint32_tkTailTileRows=32U",
            "body_rows=(spec.rows/kBodyTileRows)*kBodyTileRows",
            "tail_rows=spec.rows-body_rows",
            "tail_rows!=kTailTileRows",
            "pipelines.steel_gdn_bm48",
            "pipelines.steel",
            ".height=body_rows/kBodyTileRows",
            ".height=tail_rows/kTailTileRows",
            ".height=3",
            ".depth=1",
            ".height=2",
            ".depth=2",
            "std::uint64_t{body_rows}*spec.operation.columns",
            "std::uint64_t{body_rows}*spec.operation.reduction",
        ):
            self.assertIn(marker, source)

    def test_reuses_exactness_and_interleaved_timing_gate(self) -> None:
        source = compact(PROBE)
        for marker in (
            "Arm::SteelGdnBm48",
            "kExitSteelGdnBm48Mismatch",
            "input_identity(resources)!=original_inputs",
            "std::equal(treatment_body.begin(),treatment_body.end(),"
            "resources.exact_snapshot.begin())",
            "kMeasuredSamplesPerArm",
            "control_ranges_overlap(control_samples)",
            '"tatara_mlx_steel_gdn_bm48_component"',
            '"treatment_dispatches=8"',
        ):
            self.assertIn(marker, source)

    def test_candidate_is_probe_only_and_correctness_first(self) -> None:
        self.assertNotIn("bm48", PREFILL_STEP.lower())
        source = compact(PROBE)
        component = source.index(
            'if(mode=="--gpu-steel-gdn-bm48-component")'
        )
        benchmark = source.index(
            'if(mode=="--benchmark-steel-gdn-bm48")'
        )
        self.assertLess(component, benchmark)
        self.assertIn("timingsamples:0", source[component:benchmark])


if __name__ == "__main__":
    unittest.main()
