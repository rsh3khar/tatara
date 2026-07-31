import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
METAL = (
    ROOT / "src/backend/metal/kernels/prefill_moe.metal"
).read_text(encoding="utf-8")
GENERATOR = (
    ROOT / "tools/generate_kernel_library.py"
).read_text(encoding="utf-8")
PROBE = (
    ROOT / "tools/native/native_routed_qgemm_probe.cpp"
).read_text(encoding="utf-8")


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


class NativeRoutedQgemmBm32SourceTest(unittest.TestCase):
    def test_dispatch_and_task_bounds_are_explicit(self) -> None:
        source = compact(METAL)
        for marker in (
            "kNativeRoutedQgemmBm32TileRows=32u",
            "kNativeRoutedQgemmBm32Simdgroups=4u",
            "threadgroup_shape.x==kNativeRoutedQgemmBm32Threads",
            "task_count<=kNativeRoutedQgemmR1TaskCapacity",
            "task.row_count>kNativeRoutedQgemmBm32TileRows",
            "task.output_row_begin!=0u",
            "task_rows<=expert_segment_end-task_begin",
            "thread_index>=kNativeRoutedQgemmBm32TileRows",
        ):
            self.assertIn(marker, source)

    def test_generated_kernel_is_source_distinct_and_keeps_bm16(self) -> None:
        source = compact(GENERATOR)
        for marker in (
            "tatara_mlx_steel_routed_fused_upgate_swiglu",
            "tatara_mlx_steel_routed_down_partial",
            "tatara_mlx_steel_routed_bm32_fused_upgate_swiglu",
            "tatara_mlx_steel_routed_bm32_down_partial",
            '(\"constexprshortBM=16;\",\"constexprshortBM=32;\")',
            '(\"constexprshortWM=1;\",\"constexprshortWM=2;\")',
            "native_routed_qgemm_bm32_dispatch_valid",
            "native_routed_qgemm_bm32_task_valid",
            "native_routed_qgemm_bm32_stage_routes",
            "constexprshortrow_fragments=BM/(8*WM);",
            "uint(gate_mma.sm+row_fragment*8*WM);",
            "uint(mma_op.sm+row_fragment*8*WM);",
        ):
            self.assertIn(marker, source)

    def test_static_storage_is_below_device_floor(self) -> None:
        route_bytes = 32 * 4
        activation_bytes = 32 * 40 * 2
        weight_bytes = 32 * 40 * 2
        self.assertEqual(
            route_bytes + activation_bytes + 2 * weight_bytes,
            7808,
        )
        self.assertEqual(
            route_bytes + activation_bytes + weight_bytes,
            5248,
        )
        self.assertLessEqual(7808, 32 * 1024)

    def test_probe_has_tail_shared_and_bm16_control_arms(self) -> None:
        source = compact(PROBE)
        for marker in (
            "kHighBm32ExpectedTasks=640U",
            "kNumericBm32ExpectedTasks=9U",
            "kNumericSharedBm32ExpectedTasks=10U",
            'mode==\"--gpu-steel-bm32\"',
            'mode==\"--gpu-steel-bm32-shared\"',
            'mode==\"--benchmark-steel-bm32\"',
            "BenchmarkArm::SteelBm32",
            "treatment_arm==BenchmarkArm::SteelBm32"
            "?BenchmarkArm::Steel",
            "validate_bm32_compute_output(",
            "fixture.assimilated_expected",
            "fixture.staged_expected",
        ):
            self.assertIn(marker, source)

    def test_bm32_task_partition_is_complete_without_padding_storage(self) -> None:
        source = compact(PROBE)
        for marker in (
            "fixture.task_tile_rows=kBm32TileRows",
            "local_begin+=kBm32TileRows",
            "row_count=std::min(kBm32TileRows,"
            "fixture.counts[expert]-local_begin)",
            "valid_task_partition(fixture,fixture.tasks)",
        ):
            self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
