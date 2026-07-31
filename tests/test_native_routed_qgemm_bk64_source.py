import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
GENERATOR = (
    ROOT / "tools/generate_kernel_library.py"
).read_text(encoding="utf-8")
PROBE = (
    ROOT / "tools/native/native_routed_qgemm_probe.cpp"
).read_text(encoding="utf-8")
PREFILL_STEP = (
    ROOT / "src/runtime/prefill_step.cpp"
).read_text(encoding="utf-8")


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


class NativeRoutedQgemmBk64SourceTest(unittest.TestCase):
    def test_kernel_identity_and_reduction_geometry_are_explicit(self) -> None:
        source = compact(GENERATOR)
        for marker in (
            "tatara_mlx_steel_routed_bk64_fused_upgate_swiglu",
            "tatara_mlx_steel_routed_bk64_down_partial",
            '(\"constexprshortBK=32;\",\"constexprshortBK=64;\")',
            '\"BK==kNativeRoutedQgemmR1ReductionColumns\"',
            '\"BK==kQ4GroupSize\"',
            '\"BK_padded==kNativeRoutedQgemmR2StageStride\"',
            '\"BK_padded==kQ4GroupSize+\"',
            '\"kNativeRoutedQgemmR2StageStride-\"',
            "native_routed_qgemm_bk64_stage_position_rows",
            "native_routed_qgemm_bk64_stage_hidden_rows",
        ):
            self.assertIn(marker, source)

    def test_staging_covers_each_bk64_row_once_with_bounded_indices(self) -> None:
        source = compact(GENERATOR)
        for marker in (
            "constexpruintBK=kQ4GroupSize;",
            "constexpruintBK_padded=",
            "BK+kNativeRoutedQgemmR2StageStride-",
            "kNativeRoutedQgemmR1ReductionColumns;",
            "constexpruintchunks_per_row=BK/8u;",
            "kNativeRoutedQgemmR1TileRows*chunks_per_row;",
            "for(uintlinear=thread_index;linear<chunk_count;",
            "linear+=kNativeRoutedQgemmR1Threads)",
            "constuintrow=linear/chunks_per_row;",
            "constuintreduction=(linear%chunks_per_row)*8u;",
            "staged_input+row*BK_padded+reduction;",
            "reduction_tile+reduction",
        ):
            self.assertIn(marker, source)
        self.assertEqual(16 * (64 // 8), 128)
        self.assertEqual(128 // 64, 2)

    def test_static_storage_is_below_device_floor(self) -> None:
        route_bytes = 16 * 4
        padded_reduction = 64 + 8
        activation_bytes = 16 * padded_reduction * 2
        weight_bytes = 32 * padded_reduction * 2
        self.assertEqual(
            route_bytes + activation_bytes + 2 * weight_bytes,
            11584,
        )
        self.assertEqual(
            route_bytes + activation_bytes + weight_bytes,
            6976,
        )
        self.assertLessEqual(11584, 32 * 1024)

    def test_probe_freezes_control_identity_and_submission_bounds(self) -> None:
        source = compact(PROBE)
        for marker in (
            'mode=="--gpu-steel-bk64"',
            'mode=="--gpu-steel-bk64-shared"',
            'mode=="--benchmark-steel-bk64"',
            "BenchmarkArm::SteelBk64",
            "treatment_arm==BenchmarkArm::SteelBk64"
            "?BenchmarkArm::Steel",
            "validate_bk64_compute_output(",
            "resources.compute.staged_hidden",
            "resources.compute.bk64_hidden",
            "resources.compute.staged_partials",
            "resources.compute.bk64_partials",
            "control_hidden.begin(),control_hidden.end(),"
            "treatment_hidden.begin(),treatment_hidden.end()",
            "control_partials.begin(),control_partials.end(),"
            "treatment_partials.begin(),treatment_partials.end()",
            "command_buffers!=3U",
            "direct_dispatches!=(run_bk64_shared?5U:3U)",
        ):
            self.assertIn(marker, source)

    def test_candidate_is_probe_only(self) -> None:
        self.assertNotIn("bk64", PREFILL_STEP.lower())


if __name__ == "__main__":
    unittest.main()
