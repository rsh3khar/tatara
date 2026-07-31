import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = (
    REPOSITORY_ROOT / "tools/native/native_dense_qgemm_probe.cpp"
)
KERNEL_PATH = (
    REPOSITORY_ROOT
    / "src/backend/metal/kernels/native_dense_qgemm.metal"
)


def extract_braced_span(source, marker):
    marker_begin = source.index(marker)
    body_begin = source.index("{", marker_begin)
    depth = 0
    for index in range(body_begin, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return body_begin, index + 1
    raise AssertionError(f"unterminated braced block after {marker!r}")


def extract_braced_block(source, marker):
    begin, end = extract_braced_span(source, marker)
    return source[begin:end]


def extract_calls(source, name):
    calls = []
    positions = []
    cursor = 0
    marker = f"{name}("
    while True:
        begin = source.find(marker, cursor)
        if begin < 0:
            return calls, positions
        open_parenthesis = begin + len(name)
        depth = 0
        for index in range(open_parenthesis, len(source)):
            if source[index] == "(":
                depth += 1
            elif source[index] == ")":
                depth -= 1
                if depth == 0:
                    calls.append(source[begin : index + 1])
                    positions.append(begin)
                    cursor = index + 1
                    break
        else:
            raise AssertionError(f"unterminated {name} call")


def compact(value):
    return re.sub(r"\s+", "", value)


def parse_fixture_specs(source):
    pattern = re.compile(
        r"\{\s*"
        r'\.name\s*=\s*"([^"]+)",\s*'
        r"\.rows\s*=\s*(\d+),\s*"
        r"\.columns\s*=\s*(\d+),\s*"
        r"\.reduction\s*=\s*(\d+),\s*"
        r"\.activation_stride\s*=\s*(\d+),\s*"
        r"\.packed_stride_words\s*=\s*(\d+),\s*"
        r"\.parameter_stride\s*=\s*(\d+),\s*"
        r"\.output_stride\s*=\s*(\d+),\s*"
        r"\}",
        re.DOTALL,
    )
    return [
        (name, *(int(value) for value in values))
        for name, *values in pattern.findall(source)
    ]


class NativeDenseQgemmProbeSourceTest(unittest.TestCase):
    def setUp(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        self.source = source
        self.main = extract_braced_block(source, "int main(")
        self.run_gpu_case = extract_braced_block(source, "int run_gpu_case(")
        self.run_gpu_steel_case = extract_braced_block(
            source, "int run_gpu_steel_case("
        )

    def test_cpu_only_returns_before_device_creation(self):
        self.assertIn("--cpu-only|--gpu", self.source)
        self.assertIn(
            'mode != "--cpu-only" && mode != "--gpu"', self.source
        )
        device_calls, _ = extract_calls(
            self.source, "create_system_device"
        )
        self.assertEqual(len(device_calls), 1)

        cpu_begin, cpu_end = extract_braced_span(
            self.main, 'if (mode == "--cpu-only")'
        )
        cpu_branch = self.main[cpu_begin:cpu_end]
        self.assertEqual(cpu_branch.count("return 0;"), 1)
        for forbidden in (
            "create_system_device(",
            "create_command_queue(",
            "create_library_with_source(",
            "create_compute_pipeline(",
        ):
            self.assertNotIn(forbidden, cpu_branch)
        self.assertGreater(
            self.main.index("create_system_device()"), cpu_end
        )
        self.assertIn("command buffers submitted: 0", cpu_branch)
        self.assertNotIn("Qwen", self.source)

    def test_fixture_cases_and_submission_path(self):
        specs = parse_fixture_specs(self.source)
        self.assertEqual(
            specs,
            [
                ("minimal", 1, 1, 64, 67, 11, 3, 5),
                ("mn-tail", 33, 35, 128, 131, 19, 5, 41),
                ("long-reduction", 32, 32, 2048, 2051, 259, 35, 37),
                ("steel-aligned", 16, 32, 2048, 2048, 256, 32, 32),
            ],
        )
        self.assertIn(
            "constexpr std::array<FixtureSpec, 4>", self.source
        )

        ordered_names = (
            "create_command_buffer",
            "begin_compute_pass",
            "dispatch_threadgroups",
            "end_compute_pass",
            "commit",
            "wait_until_completed",
        )
        ordered_positions = []
        for name in ordered_names:
            calls, positions = extract_calls(self.run_gpu_case, name)
            self.assertEqual(len(calls), 1, name)
            ordered_positions.append(positions[0])
        self.assertEqual(ordered_positions, sorted(ordered_positions))
        self.assertEqual(self.run_gpu_case.count("++submissions;"), 1)

        for name in ("create_command_buffer", "commit"):
            calls, _ = extract_calls(self.source, name)
            self.assertEqual(len(calls), 2, name)
            steel_calls, _ = extract_calls(
                self.run_gpu_steel_case, name
            )
            self.assertEqual(len(steel_calls), 1, name)

        loop_begin, loop_end = extract_braced_span(
            self.main, "for (Fixture& fixture : fixtures)"
        )
        gpu_loop = self.main[loop_begin:loop_end]
        calls, _ = extract_calls(gpu_loop, "run_gpu_case")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("break;", gpu_loop)
        self.assertNotIn("continue;", gpu_loop)
        self.assertRegex(
            compact(gpu_loop),
            r"if\(result!=0\)\{returnresult;\}",
        )
        self.assertEqual(
            self.main.count("submissions != fixtures.size()"), 1
        )

    def test_cases_and_numerical_gate_are_frozen(self):
        self.assertIn("kMaximumBfloatUlp = 2", self.source)
        self.assertIn("kMaximumNormalizedError = 0.02F", self.source)
        self.assertIn(
            "stage_weights ? from_bfloat16(bfloat16(dequantized))",
            self.source,
        )
        self.assertIn("bfloat16(accumulator)", self.source)
        self.assertIn("std::isfinite(actual_value)", self.source)

    def test_binding_sequence_matches_kernel_abi_exactly(self):
        buffer_calls, buffer_positions = extract_calls(
            self.run_gpu_case, "set_buffer"
        )
        byte_calls, byte_positions = extract_calls(
            self.run_gpu_case, "set_bytes"
        )
        self.assertEqual(
            [compact(call) for call in buffer_calls],
            [
                "set_buffer(*pass.compute_pass,*activation.buffer,0,0)",
                "set_buffer(*pass.compute_pass,*packed.buffer,0,1)",
                "set_buffer(*pass.compute_pass,*scales.buffer,0,2)",
                "set_buffer(*pass.compute_pass,*biases.buffer,0,3)",
                (
                    "set_buffer(*pass.compute_pass,*output.buffer,"
                    "kGuardElements*sizeof(std::uint16_t),4)"
                ),
            ],
        )
        self.assertEqual(
            [compact(call) for call in byte_calls],
            [
                (
                    "set_bytes(*pass.compute_pass,&spec.rows,"
                    "sizeof(spec.rows),5)"
                ),
                (
                    "set_bytes(*pass.compute_pass,&spec.columns,"
                    "sizeof(spec.columns),6)"
                ),
                (
                    "set_bytes(*pass.compute_pass,&spec.reduction,"
                    "sizeof(spec.reduction),7)"
                ),
                (
                    "set_bytes(*pass.compute_pass,&spec.activation_stride,"
                    "sizeof(spec.activation_stride),8)"
                ),
                (
                    "set_bytes(*pass.compute_pass,&spec.packed_stride_words,"
                    "sizeof(spec.packed_stride_words),9)"
                ),
                (
                    "set_bytes(*pass.compute_pass,&spec.parameter_stride,"
                    "sizeof(spec.parameter_stride),10)"
                ),
                (
                    "set_bytes(*pass.compute_pass,&spec.output_stride,"
                    "sizeof(spec.output_stride),11)"
                ),
            ],
        )
        binding_positions = buffer_positions + byte_positions
        self.assertEqual(binding_positions, sorted(binding_positions))
        dispatch_calls, dispatch_positions = extract_calls(
            self.run_gpu_case, "dispatch_threadgroups"
        )
        self.assertEqual(len(dispatch_calls), 1)
        self.assertLess(binding_positions[-1], dispatch_positions[0])

        fixture_spec = compact(
            extract_braced_block(self.source, "struct FixtureSpec")
        )
        for member in ("rows", "columns", "reduction"):
            self.assertIn(f"std::uint32_t{member};", fixture_spec)
        for member in (
            "activation_stride",
            "packed_stride_words",
            "parameter_stride",
            "output_stride",
        ):
            self.assertIn(f"std::uint64_t{member};", fixture_spec)

        kernel = KERNEL_PATH.read_text(encoding="utf-8")
        signature_end = kernel.index(
            "{", kernel.index(
                "kernel void native_dense_qgemm_q4_bf16_n1("
            )
        )
        signature = kernel[
            kernel.index("kernel void native_dense_qgemm_q4_bf16_n1(")
            : signature_end
        ]
        self.assertEqual(
            re.findall(r"\[\[buffer\((\d+)\)\]\]", signature),
            [str(index) for index in range(12)],
        )
        compact_signature = compact(signature)
        for index, name in (
            (5, "input_rows"),
            (6, "output_columns"),
            (7, "reduction_columns"),
        ):
            self.assertIn(
                f"constantuint&{name}[[buffer({index})]]",
                compact_signature,
            )
        for index, name in (
            (8, "activation_row_stride_elements"),
            (9, "packed_weight_row_stride_words"),
            (10, "parameter_row_stride_elements"),
            (11, "output_row_stride_elements"),
        ):
            self.assertIn(
                f"constantulong&{name}[[buffer({index})]]",
                compact_signature,
            )

    def test_output_canary_layout_is_a_complete_partition(self):
        compact_source = compact(self.source)
        self.assertIn(
            "constexprstd::size_tkGuardElements=16;",
            compact_source,
        )
        self.assertIn(
            "2U*kGuardElements+output_body_count",
            compact_source,
        )
        self.assertIn(
            "body_begin+static_cast<std::size_t>(row)*"
            "spec.output_stride",
            compact_source,
        )
        canary_body = compact(
            extract_braced_block(self.source, "bool canaries_intact(")
        )
        self.assertIn("actual[index]!=kOutputCanary", canary_body)
        self.assertIn(
            "actual[actual.size()-kGuardElements+index]"
            "!=kOutputCanary",
            canary_body,
        )
        self.assertIn(
            "for(std::uint64_tcolumn=spec.columns;"
            "column<spec.output_stride;++column)",
            canary_body,
        )
        canary_calls, canary_positions = extract_calls(
            self.run_gpu_case, "canaries_intact"
        )
        compare_calls, compare_positions = extract_calls(
            self.run_gpu_case, "compare_output"
        )
        wait_calls, wait_positions = extract_calls(
            self.run_gpu_case, "wait_until_completed"
        )
        self.assertEqual(
            (len(wait_calls), len(canary_calls), len(compare_calls)),
            (1, 1, 1),
        )
        self.assertLess(wait_positions[0], canary_positions[0])
        self.assertLess(canary_positions[0], compare_positions[0])

        for (
            _,
            rows,
            columns,
            reduction,
            activation_stride,
            packed_stride,
            parameter_stride,
            output_stride,
        ) in parse_fixture_specs(self.source):
            self.assertGreaterEqual(activation_stride - reduction, 0)
            self.assertGreaterEqual(
                packed_stride - reduction // 8, 0
            )
            self.assertGreaterEqual(
                parameter_stride - reduction // 64, 0
            )
            self.assertGreaterEqual(output_stride - columns, 0)

            body_size = rows * output_stride
            storage_size = 2 * 16 + body_size
            prefix = set(range(0, 16))
            valid = {
                16 + row * output_stride + column
                for row in range(rows)
                for column in range(columns)
            }
            padding = {
                16 + row * output_stride + column
                for row in range(rows)
                for column in range(columns, output_stride)
            }
            suffix = set(range(16 + body_size, storage_size))
            partitions = (prefix, valid, padding, suffix)
            for left_index, left in enumerate(partitions):
                for right in partitions[left_index + 1 :]:
                    self.assertTrue(left.isdisjoint(right))
            self.assertEqual(
                set().union(*partitions), set(range(storage_size))
            )

    def test_end_compute_pass_has_its_dedicated_typed_exit(self):
        compact_run = compact(self.run_gpu_case)
        self.assertIn(
            "autoended=end_compute_pass(std::move(*pass.compute_pass));"
            "if(!ended){returnkExitComputePass;}",
            compact_run,
        )


if __name__ == "__main__":
    unittest.main()
