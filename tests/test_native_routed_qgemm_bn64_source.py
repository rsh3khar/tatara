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


class NativeRoutedQgemmBn64SourceTest(unittest.TestCase):
    def test_geometry_is_distinct_and_keeps_the_control_occupancy(self) -> None:
        source = compact(GENERATOR)
        for marker in (
            "tatara_mlx_steel_routed_bn64_fused_upgate_swiglu",
            "tatara_mlx_steel_routed_bn64_down_partial",
            '(\"constexprshortBN=32;\",\"constexprshortBN=64;\")',
            '\"BN==kNativeRoutedQgemmR1TileColumns\"',
            '\"BN==2*kNativeRoutedQgemmR1TileColumns\"',
            "constexprshortBK=32;",
            "constexprshortWM=1;",
            "constexprshortWN=2;",
            "WM*WN*SIMD_SIZE==kNativeRoutedQgemmR1Threads",
        ):
            self.assertIn(marker, source)

    def test_threadgroup_storage_is_bounded(self) -> None:
        route_bytes = 16 * 4
        stage_stride = 32 + 8
        activation_bytes = 16 * stage_stride * 2
        weight_bytes = 64 * stage_stride * 2
        self.assertEqual(
            route_bytes + activation_bytes + 2 * weight_bytes,
            11584,
        )
        self.assertEqual(
            route_bytes + activation_bytes + weight_bytes,
            6464,
        )
        self.assertLessEqual(11584, 32 * 1024)

    def test_probe_binds_exact_indirect_grids_and_byte_identity(self) -> None:
        source = compact(PROBE)
        for marker in (
            "kBn64TileColumns=2U*kTileColumns",
            "kBn64UpColumnGroups="
            "(kExpertDimension+kBn64TileColumns-1U)/"
            "kBn64TileColumns",
            "kBn64DownColumnGroups="
            "(kHidden+kBn64TileColumns-1U)/"
            "kBn64TileColumns",
            "initialize_resolved_producer_buffers("
            "device,numeric,kBn64UpColumnGroups,"
            "resources.numeric_bn64_up)",
            "initialize_resolved_producer_buffers("
            "device,numeric,kBn64DownColumnGroups,"
            "resources.numeric_bn64_down)",
            "numeric_task_bodies_identical("
            "resources.numeric_up,resources.numeric_bn64_up)",
            'mode==\"--gpu-steel-bn64\"',
            'mode==\"--gpu-steel-bn64-shared\"',
            'mode==\"--benchmark-steel-bn64\"',
            "BenchmarkArm::SteelBn64",
            "treatment_arm==BenchmarkArm::SteelBn64"
            "?BenchmarkArm::Steel",
            "validate_bn64_compute_output(",
            "resources.compute.staged_hidden",
            "resources.compute.bn64_hidden",
            "resources.compute.staged_partials",
            "resources.compute.bn64_partials",
            "command_buffers!=3U",
        ):
            self.assertIn(marker, source)

    def test_candidate_is_probe_only(self) -> None:
        self.assertNotIn("bn64", PREFILL_STEP.lower())


if __name__ == "__main__":
    unittest.main()
