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


class NativeDenseGdnFused2SourceTest(unittest.TestCase):
    def test_generated_kernel_has_plan_derived_pair_bounds(self) -> None:
        source = compact(GENERATOR)
        for marker in (
            "tatara_mlx_steel_gdn_fused2_affine_qmm",
            "kGdnQkvRows%pair_columns==0u",
            "kGdnZRows%pair_columns==0u",
            "kGdnValueHeads==uint(BN)",
            "kGdnBRowOffset==kGdnQkvRows+kGdnZRows",
            "kGdnARowOffset==kGdnBRowOffset+kGdnValueHeads",
            "kGdnProjectionRows%pair_columns==0u",
            "K!=kHiddenDimension",
            "output_stride!=kGdnProjectionRows",
            "pair_begin>=kGdnProjectionRows",
            "threadgroup_shape.x!=uint(SIMD_SIZE)",
            "threadgroup_shape.y!=uint(WM)",
            "threadgroup_shape.z!=uint(WN)",
        ):
            self.assertIn(marker, source)

    def test_activation_is_loaded_once_for_two_independent_mmas(self) -> None:
        source = compact(GENERATOR)
        kernel = source[
            source.index(
                '"[[kernel]]void"+MLX_STEEL_GDN_FUSED2_KERNEL_NAME'
            ) :
            source.index(
                '_source_lines(quantized,2399,2593)',
                source.index(
                    '"[[kernel]]void"+MLX_STEEL_GDN_FUSED2_KERNEL_NAME'
                ),
            )
        ]
        self.assertEqual(kernel.count("activation_loader.load_unsafe()"), 1)
        self.assertEqual(kernel.count("activation_loader.load_safe("), 1)
        self.assertEqual(kernel.count("first_mma.mma(Xs,Ws)"), 1)
        self.assertEqual(kernel.count("second_mma.mma(Xs,Ws)"), 1)
        self.assertIn("first_loader.load_unsafe()", kernel)
        self.assertIn("second_loader.load_unsafe()", kernel)
        self.assertIn("first_mma.store_result", kernel)
        self.assertIn("second_mma.store_result", kernel)

    def test_threadgroup_storage_stays_under_device_floor(self) -> None:
        activation_bytes = 32 * 40 * 2
        weight_bytes = 32 * 40 * 2
        self.assertEqual(activation_bytes + weight_bytes, 5120)
        self.assertLessEqual(activation_bytes + weight_bytes, 32 * 1024)

    def test_probe_freezes_binding_dispatch_and_exactness_gate(self) -> None:
        source = compact(PROBE)
        for marker in (
            'mode=="--gpu-steel-gdn-fused2-component"',
            'mode=="--benchmark-steel-gdn-fused2"',
            "kKernelLibraryMlxSteelGdnFused2KernelName",
            "Arm::SteelGdnFused2",
            "spec.operation.columns/(2U*kN1TileColumns)",
            "resources.activations,0,12",
            "resources.output,output_offset,13",
            "&rows,sizeof(rows),14",
            "&reduction,sizeof(reduction),15",
            "&output_stride,sizeof(output_stride),16",
            "input_identity(resources)!=original_inputs",
            "std::equal(treatment_body.begin(),treatment_body.end(),"
            "resources.exact_snapshot.begin())",
            "control_dispatches=4",
            "treatment_dispatches=1",
            "kMeasuredSamplesPerArm",
            "control_ranges_overlap(control_samples)",
        ):
            self.assertIn(marker, source)

    def test_component_precedes_benchmark_in_main(self) -> None:
        source = compact(PROBE)
        component = source.index(
            'if(mode=="--gpu-steel-gdn-fused2-component")'
        )
        benchmark = source.index(
            'if(mode=="--benchmark-steel-gdn-fused2")'
        )
        self.assertLess(component, benchmark)
        self.assertIn("timingsamples:0", source[component:benchmark])


if __name__ == "__main__":
    unittest.main()
