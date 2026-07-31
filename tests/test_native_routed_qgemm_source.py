import hashlib
import math
import re
import struct
import unittest
from dataclasses import dataclass, replace
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = (
    REPOSITORY_ROOT
    / "src/backend/metal/kernels/prefill_moe.metal"
)
R1_MARKER = "\n#include <metal_simdgroup_matrix>\n"
EXACT_SOURCE_SHA256 = (
    "014aea3331edd28a303eec5a00d99309ad91b68b5b939f015fe39d356ac1aa98"
)
TILE_ROWS = 16
TILE_COLUMNS = 32
SIMDGROUPS = 2
SIMDGROUP_WIDTH = 32
TASK_CAPACITY = 4096
KERNEL_NAMES = (
    "native_routed_qgemm_r1_fused_upgate_swiglu",
    "native_routed_qgemm_r1_gate",
    "native_routed_qgemm_r1_up_swiglu",
    "native_routed_qgemm_r1_down_partial",
)
COMMON_THREAD_ARGUMENTS = (
    "uint3 group [[threadgroup_position_in_grid]]",
    "uint lane [[thread_index_in_simdgroup]]",
    "uint simdgroup [[simdgroup_index_in_threadgroup]]",
    "uint simdgroup_width [[threads_per_simdgroup]]",
    "uint3 threadgroup_shape [[threads_per_threadgroup]]",
)
FUSED_ARGUMENTS = (
    "device const bfloat* input [[buffer(0)]]",
    "device const uint* route_list [[buffer(1)]]",
    "device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]]",
    "device const uint* gate_words [[buffer(3)]]",
    "device const bfloat* gate_scales [[buffer(4)]]",
    "device const bfloat* gate_biases [[buffer(5)]]",
    "device const uint* up_words [[buffer(6)]]",
    "device const bfloat* up_scales [[buffer(7)]]",
    "device const bfloat* up_biases [[buffer(8)]]",
    "device bfloat* hidden [[buffer(9)]]",
    "constant uint& task_count [[buffer(10)]]",
    "constant uint& route_list_expert_stride [[buffer(11)]]",
    "constant uint& route_list_capacity_per_expert [[buffer(12)]]",
    "constant ulong& route_list_total_extent [[buffer(13)]]",
    "constant uint& input_rows [[buffer(14)]]",
) + COMMON_THREAD_ARGUMENTS
GATE_ARGUMENTS = (
    "device const bfloat* input [[buffer(0)]]",
    "device const uint* route_list [[buffer(1)]]",
    "device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]]",
    "device const uint* gate_words [[buffer(3)]]",
    "device const bfloat* gate_scales [[buffer(4)]]",
    "device const bfloat* gate_biases [[buffer(5)]]",
    "device bfloat* hidden [[buffer(6)]]",
    "constant uint& task_count [[buffer(7)]]",
    "constant uint& route_list_expert_stride [[buffer(8)]]",
    "constant uint& route_list_capacity_per_expert [[buffer(9)]]",
    "constant ulong& route_list_total_extent [[buffer(10)]]",
    "constant uint& input_rows [[buffer(11)]]",
) + COMMON_THREAD_ARGUMENTS
UP_ARGUMENTS = (
    "device const bfloat* input [[buffer(0)]]",
    "device const uint* route_list [[buffer(1)]]",
    "device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]]",
    "device const uint* up_words [[buffer(3)]]",
    "device const bfloat* up_scales [[buffer(4)]]",
    "device const bfloat* up_biases [[buffer(5)]]",
    "device bfloat* hidden [[buffer(6)]]",
    "constant uint& task_count [[buffer(7)]]",
    "constant uint& route_list_expert_stride [[buffer(8)]]",
    "constant uint& route_list_capacity_per_expert [[buffer(9)]]",
    "constant ulong& route_list_total_extent [[buffer(10)]]",
    "constant uint& input_rows [[buffer(11)]]",
) + COMMON_THREAD_ARGUMENTS
DOWN_ARGUMENTS = (
    "device const bfloat* hidden [[buffer(0)]]",
    "device const uint* route_list [[buffer(1)]]",
    "device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]]",
    "device const uint* down_words [[buffer(3)]]",
    "device const bfloat* down_scales [[buffer(4)]]",
    "device const bfloat* down_biases [[buffer(5)]]",
    "device float* partials [[buffer(6)]]",
    "constant uint& task_count [[buffer(7)]]",
    "constant uint& route_list_expert_stride [[buffer(8)]]",
    "constant uint& route_list_capacity_per_expert [[buffer(9)]]",
    "constant ulong& route_list_total_extent [[buffer(10)]]",
    "constant uint& input_rows [[buffer(11)]]",
) + COMMON_THREAD_ARGUMENTS
R2_COMMON_THREAD_ARGUMENTS = (
    "uint3 group [[threadgroup_position_in_grid]]",
    "uint lane [[thread_index_in_simdgroup]]",
    "uint simdgroup [[simdgroup_index_in_threadgroup]]",
    "uint simdgroup_width [[threads_per_simdgroup]]",
    "uint thread_index [[thread_index_in_threadgroup]]",
    "uint3 threadgroup_shape [[threads_per_threadgroup]]",
)
R2_FUSED_ARGUMENTS = (
    "device const bfloat* input [[buffer(0)]]",
    "device const uint* route_list [[buffer(1)]]",
    "device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]]",
    "device const uint* gate_words [[buffer(3)]]",
    "device const bfloat* gate_scales [[buffer(4)]]",
    "device const bfloat* gate_biases [[buffer(5)]]",
    "device const uint* up_words [[buffer(6)]]",
    "device const bfloat* up_scales [[buffer(7)]]",
    "device const bfloat* up_biases [[buffer(8)]]",
    "device bfloat* hidden [[buffer(9)]]",
    "constant uint& task_count [[buffer(10)]]",
    "constant uint& route_list_expert_stride [[buffer(11)]]",
    "constant uint& route_list_capacity_per_expert [[buffer(12)]]",
    "constant ulong& route_list_total_extent [[buffer(13)]]",
    "constant uint& input_rows [[buffer(14)]]",
    "device const uint* shared_gate_words [[buffer(15)]]",
    "device const bfloat* shared_gate_scales [[buffer(16)]]",
    "device const bfloat* shared_gate_biases [[buffer(17)]]",
    "device const uint* shared_up_words [[buffer(18)]]",
    "device const bfloat* shared_up_scales [[buffer(19)]]",
    "device const bfloat* shared_up_biases [[buffer(20)]]",
    "constant uint& include_shared_expert [[buffer(21)]]",
) + R2_COMMON_THREAD_ARGUMENTS
R2_DOWN_ARGUMENTS = (
    "device const bfloat* hidden [[buffer(0)]]",
    "device const uint* route_list [[buffer(1)]]",
    "device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]]",
    "device const uint* down_words [[buffer(3)]]",
    "device const bfloat* down_scales [[buffer(4)]]",
    "device const bfloat* down_biases [[buffer(5)]]",
    "device float* partials [[buffer(6)]]",
    "constant uint& task_count [[buffer(7)]]",
    "constant uint& route_list_expert_stride [[buffer(8)]]",
    "constant uint& route_list_capacity_per_expert [[buffer(9)]]",
    "constant ulong& route_list_total_extent [[buffer(10)]]",
    "constant uint& input_rows [[buffer(11)]]",
    "device const uint* shared_down_words [[buffer(12)]]",
    "device const bfloat* shared_down_scales [[buffer(13)]]",
    "device const bfloat* shared_down_biases [[buffer(14)]]",
    "constant uint& include_shared_expert [[buffer(15)]]",
) + R2_COMMON_THREAD_ARGUMENTS


@dataclass(frozen=True)
class RoutedTask:
    expert_index: int
    absolute_route_list_begin: int
    row_count: int
    output_row_begin: int = 0


def fragment_coordinate(lane: int) -> tuple[int, int]:
    quarter = lane >> 2
    row = (quarter & 4) + ((lane >> 1) & 3)
    column = (quarter & 2) * 2 + (lane & 1) * 2
    return row, column


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bfloat16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def swiglu(gate: float, up: float) -> float:
    gate32 = float32(gate)
    up32 = float32(up)
    return float32(
        float32(gate32 / float32(1.0 + math.exp(-gate32))) * up32
    )


def build_routed_tasks(
    counts: tuple[int, ...],
    expert_stride: int,
    capacity_per_expert: int,
) -> tuple[RoutedTask, ...]:
    if expert_stride < 0 or capacity_per_expert < 0:
        raise ValueError("route-list geometry must be non-negative")
    if capacity_per_expert > expert_stride:
        raise ValueError("expert capacity must fit its stride")
    tasks = []
    for expert, count in enumerate(counts):
        if count < 0 or count > capacity_per_expert:
            raise ValueError("expert count exceeds its route-list segment")
        for local_begin in range(0, count, TILE_ROWS):
            tasks.append(
                RoutedTask(
                    expert_index=expert,
                    absolute_route_list_begin=(
                        expert * expert_stride + local_begin
                    ),
                    row_count=min(TILE_ROWS, count - local_begin),
                )
            )
    if len(tasks) > TASK_CAPACITY:
        raise ValueError("routed task count exceeds R1 capacity")
    return tuple(tasks)


def validate_routed_partition(
    tasks: tuple[RoutedTask, ...],
    counts: tuple[int, ...],
    expert_stride: int,
    capacity_per_expert: int,
) -> None:
    if expert_stride < 0 or capacity_per_expert < 0:
        raise ValueError("route-list geometry must be non-negative")
    if capacity_per_expert > expert_stride:
        raise ValueError("expert capacity must fit its stride")
    if len(tasks) > TASK_CAPACITY:
        raise ValueError("routed task count exceeds R1 capacity")
    actual_by_expert: list[list[tuple[int, int]]] = [
        [] for _ in counts
    ]
    for task in tasks:
        if task.expert_index < 0 or task.expert_index >= len(counts):
            raise ValueError("task names a shared or invalid expert")
        if task.row_count <= 0 or task.row_count > TILE_ROWS:
            raise ValueError("task row count is outside the R1 tile")
        if task.output_row_begin != 0:
            raise ValueError("task widens output ownership")
        segment_begin = task.expert_index * expert_stride
        segment_end = segment_begin + capacity_per_expert
        task_end = task.absolute_route_list_begin + task.row_count
        if (
            task.absolute_route_list_begin < segment_begin
            or task_end > segment_end
        ):
            raise ValueError("task escapes its expert segment")
        actual_by_expert[task.expert_index].append(
            (task.absolute_route_list_begin, task.row_count)
        )
    for expert, count in enumerate(counts):
        if count < 0 or count > capacity_per_expert:
            raise ValueError("expert count exceeds its route-list segment")
        expected = [
            (
                expert * expert_stride + local_begin,
                min(TILE_ROWS, count - local_begin),
            )
            for local_begin in range(0, count, TILE_ROWS)
        ]
        if sorted(actual_by_expert[expert]) != expected:
            raise ValueError(
                "tasks contain an overlap, gap, or wrong absolute begin"
            )


def split_source(source: str) -> tuple[str, str]:
    exact_source, marker, r1_source = source.partition(R1_MARKER)
    if not marker:
        raise AssertionError("R1 must remain an additive source suffix")
    return exact_source, r1_source


def kernel_span(source: str, name: str) -> str:
    marker = f"kernel void {name}("
    start = source.index(marker)
    next_kernel = source.find("\nkernel void ", start + len(marker))
    return source[start:] if next_kernel < 0 else source[start:next_kernel]


def function_span(source: str, name: str, next_name: str) -> str:
    return source[source.index(name) : source.index(next_name)]


def replace_in_function(
    source: str,
    function_name: str,
    original: str,
    replacement: str,
) -> str:
    start = source.index(function_name)
    suffix = source[start:]
    changed_suffix = suffix.replace(original, replacement, 1)
    if changed_suffix == suffix:
        raise AssertionError("mutation target was not found")
    return source[:start] + changed_suffix


def require_exact_signature(
    span: str,
    arguments: tuple[str, ...],
) -> None:
    signature_begin = span.index("(") + 1
    signature_end = span.index(") {")
    actual = tuple(
        " ".join(argument.split())
        for argument in span[signature_begin:signature_end].split(",")
    )
    expected = tuple(" ".join(argument.split()) for argument in arguments)
    if actual != expected:
        raise AssertionError("R1 kernel signature changed")


def require_accumulator_map(
    span: str,
    accumulator: str,
    weight: str,
    activation: str,
) -> None:
    expected = (
        (0, 0, 0),
        (1, 0, 1),
        (2, 1, 0),
        (3, 1, 1),
    )
    for accumulator_index, weight_index, activation_index in expected:
        pattern = (
            rf"simdgroup_multiply_accumulate\(\s*"
            rf"{accumulator}\[{accumulator_index}\],\s*"
            rf"{weight}{weight_index},\s*"
            rf"{activation}{activation_index},\s*"
            rf"{accumulator}\[{accumulator_index}\]\);"
        )
        if re.search(pattern, span) is None:
            raise AssertionError("R1 accumulator ownership map changed")


def validate_r1_source(source: str) -> None:
    exact_source, r1_source = split_source(source)
    r1_source, r2_marker, _ = r1_source.partition(
        "constant uint kNativeRoutedQgemmR2StageStride"
    )
    assert r2_marker
    assert (
        hashlib.sha256(exact_source.encode("utf-8")).hexdigest()
        == EXACT_SOURCE_SHA256
    )
    assert re.search(
        r"struct NativeRoutedQgemmR1Task \{\s*"
        r"uint expert_index;\s*"
        r"uint absolute_route_list_begin;\s*"
        r"uint row_count;\s*"
        r"uint output_row_begin;\s*"
        r"\};",
        r1_source,
    )
    for profile_assertion in (
        "kNativeRoutedQgemmR1TaskBytes == 16u",
        "kNativeRoutedQgemmR1TaskCapacity == 4096u",
        "kNativeRoutedQgemmR1FusedAccumulatorElements ==",
        "kNativeRoutedQgemmR1SingleAccumulatorElements ==",
        "kNativeRoutedQgemmR1AccumulatorElements ==",
        "kNativeRoutedQgemmR1ThreadgroupMemoryBytes == 0u",
    ):
        assert profile_assertion in r1_source

    coordinate = function_span(
        r1_source,
        "inline uint2 native_routed_qgemm_r1_fragment_coordinate",
        "inline bool native_routed_qgemm_r1_dispatch_valid",
    )
    assert "(quarter & 4u) + ((lane >> 1u) & 3u)" in coordinate
    assert (
        "(quarter & 2u) * 2u + (lane & 1u) * 2u"
        in coordinate
    )
    assert "return uint2(row, column);" in coordinate

    dispatch = function_span(
        r1_source,
        "inline bool native_routed_qgemm_r1_dispatch_valid",
        "inline bool native_routed_qgemm_r1_task_valid",
    )
    for guard in (
        "task_count <= kNativeRoutedQgemmR1TaskCapacity",
        "group.x < task_count",
        "group.z == 0u",
        "simdgroup < kNativeRoutedQgemmR1Simdgroups",
        "simdgroup_width == kSimdgroupWidth",
        "threadgroup_shape.x == kNativeRoutedQgemmR1Threads",
        "threadgroup_shape.y == 1u",
        "threadgroup_shape.z == 1u",
    ):
        assert guard in dispatch

    task_valid = function_span(
        r1_source,
        "inline bool native_routed_qgemm_r1_task_valid",
        "inline bool native_routed_qgemm_r1_route",
    )
    assert re.search(
        r"task\.row_count > "
        r"kNativeRoutedQgemmR1TileRows\s*\|\|",
        task_valid,
    )
    for guard in (
        "task.expert_index >= kMoeExperts",
        "task.row_count == 0u",
        "task.output_row_begin != 0u",
        "route_list_capacity_per_expert >",
        "route_list_expert_stride",
        "const ulong expert_segment_begin",
        "ulong(task.expert_index) *",
        "ulong(route_list_expert_stride)",
        "const ulong expert_segment_end",
        "ulong(route_list_capacity_per_expert)",
        "const ulong task_begin",
        "ulong(task.absolute_route_list_begin)",
        "const ulong task_rows = ulong(task.row_count)",
        "task_begin >= expert_segment_begin",
        "task_begin <= expert_segment_end",
        "task_rows <= expert_segment_end - task_begin",
        "expert_segment_end <= route_list_total_extent",
        "task_rows <= route_list_total_extent - task_begin",
    ):
        assert guard in task_valid

    route = function_span(
        r1_source,
        "inline bool native_routed_qgemm_r1_route",
        "inline void native_routed_qgemm_r1_zero",
    )
    assert "local_row >= task.row_count" in route
    assert (
        "ulong(task.absolute_route_list_begin) +\n"
        "            ulong(local_row)"
        in route
    )
    assert "position = packed >> kPrefillPackedSlotBits;" in route
    assert "position < input_rows" in route
    assert "slot < kMoeActiveExperts" in route
    assert "kMoeSlotCount" not in route

    weight = function_span(
        r1_source,
        "inline void native_routed_qgemm_r1_load_weight",
        "inline void native_routed_qgemm_r1_load_position_rows",
    )
    assert "output_column_begin + coordinate.x" in weight
    assert "reduction_column_begin + coordinate.y" in weight
    assert (
        "reduction_columns / kQ4ValuesPerWord"
        in weight
    )
    assert "reduction_columns / kQ4GroupSize" in weight
    assert (
        "ulong(reduction_column / kQ4ValuesPerWord)"
        in weight
    )
    assert (
        "ulong(reduction_column / kQ4GroupSize)"
        in weight
    )
    assert "reduction_column % kQ4ValuesPerWord" in weight
    assert (
        "float((word >> (4u * nibble)) & 15u) * scale + bias"
        in weight
    )
    assert (
        "float((word >> (4u * (nibble + 1u))) & 15u)"
        in weight
    )

    position_rows = function_span(
        r1_source,
        "inline void native_routed_qgemm_r1_load_position_rows",
        "inline void native_routed_qgemm_r1_load_padded_slot_rows",
    )
    assert "reduction_column_begin + coordinate.x" in position_rows
    assert "local_row_begin + coordinate.y" in position_rows
    assert (
        "ulong(position) * ulong(kHiddenDimension) +"
        in position_rows
    )
    assert "ulong(slot)" not in position_rows

    padded_slot_rows = function_span(
        r1_source,
        "inline void native_routed_qgemm_r1_load_padded_slot_rows",
        "inline float native_routed_qgemm_r1_value",
    )
    assert "reduction_column_begin + coordinate.x" in padded_slot_rows
    assert "local_row_begin + coordinate.y" in padded_slot_rows
    assert (
        "(ulong(position) * ulong(kMoeSlotCount) +"
        in padded_slot_rows
    )
    assert "ulong(slot)) *" in padded_slot_rows
    assert "ulong(kMoeExpertDimension) +" in padded_slot_rows

    hidden_index = function_span(
        r1_source,
        "inline ulong native_routed_qgemm_r1_hidden_index",
        "inline ulong native_routed_qgemm_r1_partial_index",
    )
    partial_index = function_span(
        r1_source,
        "inline ulong native_routed_qgemm_r1_partial_index",
        "kernel void native_routed_qgemm_r1_fused_upgate_swiglu",
    )
    for index_source, row_width in (
        (hidden_index, "kMoeExpertDimension"),
        (partial_index, "kHiddenDimension"),
    ):
        assert "ulong(position)" in index_source
        assert "ulong(kMoeSlotCount)" in index_source
        assert "ulong(slot)" in index_source
        assert f"ulong({row_width})" in index_source

    assert "shared_" not in r1_source
    assert "const bool shared" not in r1_source
    assert "threadgroup " not in r1_source
    assert "threadgroup_barrier" not in r1_source
    assert "gather" not in r1_source.lower()
    assert "flat_h" not in r1_source.lower()
    assert len(
        re.findall(r"\nkernel void native_routed_qgemm_r1_", r1_source)
    ) == 4

    spans = {
        name: kernel_span(r1_source, name)
        for name in KERNEL_NAMES
    }
    signatures = {
        KERNEL_NAMES[0]: FUSED_ARGUMENTS,
        KERNEL_NAMES[1]: GATE_ARGUMENTS,
        KERNEL_NAMES[2]: UP_ARGUMENTS,
        KERNEL_NAMES[3]: DOWN_ARGUMENTS,
    }
    for name, span in spans.items():
        require_exact_signature(span, signatures[name])
        assert "native_routed_qgemm_r1_dispatch_valid(" in span
        assert (
            "const NativeRoutedQgemmR1Task task = tasks[group.x];"
            in span
        )
        assert "native_routed_qgemm_r1_task_valid(" in span
        assert "route_list_expert_stride" in span
        assert "route_list_capacity_per_expert" in span
        assert "route_list_total_extent" in span
        assert "native_routed_qgemm_r1_route(" in span
        expected_output_dimension = (
            "kHiddenDimension"
            if name == KERNEL_NAMES[3]
            else "kMoeExpertDimension"
        )
        assert (
            f"(group.y >=\n        ({expected_output_dimension} +"
            in span
        )
        assert (
            "simdgroup *\n"
            "            (kNativeRoutedQgemmR1TileColumns /\n"
            "             kNativeRoutedQgemmR1Simdgroups)"
            in span
        )

    fused = spans[KERNEL_NAMES[0]]
    assert (
        "simdgroup_matrix<float, 8, 8> gate_accumulators[4]"
        in fused
    )
    assert (
        "simdgroup_matrix<float, 8, 8> up_accumulators[4]"
        in fused
    )
    require_accumulator_map(
        fused,
        "gate_accumulators",
        "gate_a",
        "input_b",
    )
    require_accumulator_map(
        fused,
        "up_accumulators",
        "up_a",
        "input_b",
    )
    assert "native_routed_qgemm_r1_load_position_rows(" in fused
    assert "native_routed_qgemm_r1_load_padded_slot_rows(" not in fused
    assert "(gate / (1.0f + exp(-gate))) * up" in fused

    gate = spans[KERNEL_NAMES[1]]
    require_accumulator_map(
        gate,
        "accumulators",
        "weight_a",
        "input_b",
    )
    assert "simdgroup_matrix<float, 8, 8> accumulators[4]" in gate
    assert "native_routed_qgemm_r1_load_position_rows(" in gate
    assert "native_routed_qgemm_r1_load_padded_slot_rows(" not in gate
    assert "static_cast<bfloat>(" in gate
    assert "exp(" not in gate

    up = spans[KERNEL_NAMES[2]]
    require_accumulator_map(
        up,
        "accumulators",
        "weight_a",
        "input_b",
    )
    assert "simdgroup_matrix<float, 8, 8> accumulators[4]" in up
    assert "native_routed_qgemm_r1_load_position_rows(" in up
    assert "native_routed_qgemm_r1_load_padded_slot_rows(" not in up
    assert "const float gate = float(hidden[hidden_index]);" in up
    assert "(gate / (1.0f + exp(-gate))) * up" in up

    down = spans[KERNEL_NAMES[3]]
    require_accumulator_map(
        down,
        "accumulators",
        "weight_a",
        "hidden_b",
    )
    assert "native_routed_qgemm_r1_load_position_rows(" not in down
    assert "native_routed_qgemm_r1_load_padded_slot_rows(" in down
    assert "native_routed_qgemm_r1_partial_index(" in down


class NativeRoutedQgemmSourceTest(unittest.TestCase):
    def test_exact_moe_source_is_byte_identical_and_r1_is_structurally_safe(
        self,
    ):
        validate_r1_source(SOURCE_PATH.read_text(encoding="utf-8"))

    def test_r2_shared_assimilation_abi_and_selection_are_fail_closed(
        self,
    ):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        up = kernel_span(
            source, "native_routed_qgemm_r2_fused_upgate_swiglu"
        )
        down = kernel_span(
            source, "native_routed_qgemm_r2_down_partial"
        )
        require_exact_signature(up, R2_FUSED_ARGUMENTS)
        require_exact_signature(down, R2_DOWN_ARGUMENTS)

        task_valid = function_span(
            source,
            "inline bool native_routed_qgemm_r2_task_valid",
            "inline bool native_routed_qgemm_r2_route",
        )
        for required in (
            "include_shared_expert > 1u",
            "task.expert_index < kMoeExperts",
            "task.expert_index != kMoeExperts",
            "ulong(kMoeExperts)",
            "task.row_count == 0u",
            "task.output_row_begin != 0u",
        ):
            self.assertIn(required, task_valid)
        route = function_span(
            source,
            "inline bool native_routed_qgemm_r2_route",
            "inline void native_routed_qgemm_r2_copy_bfloat8",
        )
        for required in (
            "task.expert_index == kMoeExperts",
            "slot == kMoeActiveExperts",
            "slot < kMoeActiveExperts",
            "position < input_rows",
        ):
            self.assertIn(required, route)

        for kernel, shared_triplets, row_stride in (
            (
                up,
                (
                    "shared ? shared_gate_words : gate_words",
                    "shared ? shared_gate_scales : gate_scales",
                    "shared ? shared_gate_biases : gate_biases",
                    "shared ? shared_up_words : up_words",
                    "shared ? shared_up_scales : up_scales",
                    "shared ? shared_up_biases : up_biases",
                ),
                "ulong(kMoeExpertDimension)",
            ),
            (
                down,
                (
                    "shared ? shared_down_words : down_words",
                    "shared ? shared_down_scales : down_scales",
                    "shared ? shared_down_biases : down_biases",
                ),
                "ulong(kHiddenDimension)",
            ),
        ):
            self.assertIn(
                "task.expert_index == kMoeExperts", kernel
            )
            self.assertIn("shared ? 0ul", kernel)
            self.assertIn(row_stride, kernel)
            for selection in shared_triplets:
                self.assertIn(selection, kernel)
            self.assertIn(
                "native_routed_qgemm_r2_task_valid(", kernel
            )
            self.assertIn(
                "native_routed_qgemm_r2_stage_routes(", kernel
            )

    def test_task_record_is_four_packed_uint32_fields(self):
        record = struct.pack("<IIII", 127, 4095, 16, 0)

        self.assertEqual(len(record), 16)
        self.assertEqual(
            struct.unpack("<IIII", record),
            (127, 4095, 16, 0),
        )

    def test_fragment_ownership_covers_one_ragged_tile_once(self):
        cells = []
        for simdgroup in range(SIMDGROUPS):
            for output_fragment in range(2):
                for route_fragment in range(2):
                    for lane in range(SIMDGROUP_WIDTH):
                        output_row, route_row = fragment_coordinate(lane)
                        for element in range(2):
                            cells.append(
                                (
                                    route_fragment * 8
                                    + route_row
                                    + element,
                                    simdgroup * 16
                                    + output_fragment * 8
                                    + output_row,
                                )
                            )

        expected = {
            (row, column)
            for row in range(TILE_ROWS)
            for column in range(TILE_COLUMNS)
        }
        self.assertEqual(len(cells), TILE_ROWS * TILE_COLUMNS)
        self.assertEqual(set(cells), expected)

    def test_routed_task_oracle_covers_every_expert_segment_exactly(self):
        counts = (0, 1, 16, 17, 33)
        tasks = build_routed_tasks(
            counts,
            expert_stride=64,
            capacity_per_expert=48,
        )

        validate_routed_partition(
            tasks,
            counts,
            expert_stride=64,
            capacity_per_expert=48,
        )
        self.assertEqual(
            tasks,
            (
                RoutedTask(1, 64, 1),
                RoutedTask(2, 128, 16),
                RoutedTask(3, 192, 16),
                RoutedTask(3, 208, 1),
                RoutedTask(4, 256, 16),
                RoutedTask(4, 272, 16),
                RoutedTask(4, 288, 1),
            ),
        )
        self.assertTrue(
            all(task.expert_index < len(counts) for task in tasks)
        )

    def test_routed_task_oracle_rejects_partition_adversaries(self):
        counts = (0, 1, 16, 17, 33)
        tasks = build_routed_tasks(counts, 64, 48)
        expert_four = [
            index
            for index, task in enumerate(tasks)
            if task.expert_index == 4
        ]
        adversaries = {
            "overlap": (
                tasks[: expert_four[1]]
                + (
                    replace(
                        tasks[expert_four[1]],
                        absolute_route_list_begin=271,
                    ),
                )
                + tasks[expert_four[1] + 1 :]
            ),
            "gap": (
                tasks[: expert_four[1]]
                + (
                    replace(
                        tasks[expert_four[1]],
                        absolute_route_list_begin=273,
                    ),
                )
                + tasks[expert_four[1] + 1 :]
            ),
            "wrong expert": (
                (replace(tasks[0], expert_index=2),)
                + tasks[1:]
            ),
            "out of segment": (
                tasks[:-1]
                + (
                    replace(
                        tasks[-1],
                        absolute_route_list_begin=304,
                    ),
                )
            ),
            "shared expert": (
                (replace(tasks[0], expert_index=len(counts)),)
                + tasks[1:]
            ),
        }
        for name, adversary in adversaries.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    validate_routed_partition(
                        adversary,
                        counts,
                        expert_stride=64,
                        capacity_per_expert=48,
                    )

    def test_routed_task_oracle_rejects_invalid_segment_geometry(self):
        with self.assertRaisesRegex(ValueError, "fit its stride"):
            build_routed_tasks((1,), 15, 16)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_routed_tasks((17,), 32, 16)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_routed_tasks((1,), -1, 0)

    def test_fused_and_split_gate_round_trip_are_distinct_families(self):
        gate = float32(0.731234)
        up = float32(-1.222345)
        fused = swiglu(gate, up)
        split = swiglu(bfloat16(gate), up)

        self.assertNotEqual(gate, bfloat16(gate))
        self.assertNotEqual(fused, split)

    def test_dangerous_source_mutations_are_rejected(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        mutations = {
            "exact path changed": source.replace(
                "kernel void block_down_combine(",
                "kernel void block_down_combine_changed(",
                1,
            ),
            "fragment orientation changed": source.replace(
                "return uint2(row, column);",
                "return uint2(column, row);",
                1,
            ),
            "A orientation changed": source.replace(
                "output_column_begin + coordinate.x",
                "output_column_begin + coordinate.y",
                1,
            ),
            "B orientation changed": source.replace(
                "reduction_column_begin + coordinate.x",
                "reduction_column_begin + coordinate.y",
                1,
            ),
            "accumulator map changed": source.replace(
                "gate_accumulators[1], gate_a0, input_b1",
                "gate_accumulators[1], gate_a1, input_b1",
                1,
            ),
            "Q4 word addressing changed": source.replace(
                "ulong(reduction_column / kQ4ValuesPerWord)",
                "ulong(reduction_column / kQ4GroupSize)",
                1,
            ),
            "Q4 group addressing changed": source.replace(
                "ulong(reduction_column / kQ4GroupSize)",
                "ulong(reduction_column / kQ4ValuesPerWord)",
                1,
            ),
            "Q4 nibble addressing changed": source.replace(
                "reduction_column % kQ4ValuesPerWord",
                "reduction_column % kQ4GroupSize",
                1,
            ),
            "expert guard widened": source.replace(
                "task.expert_index >= kMoeExperts",
                "task.expert_index > kMoeExperts",
                1,
            ),
            "task row guard widened": source.replace(
                "task.row_count > kNativeRoutedQgemmR1TileRows",
                "task.row_count > "
                "kNativeRoutedQgemmR1TileRows + 1u",
                1,
            ),
            "output ownership widened": source.replace(
                "task.output_row_begin != 0u",
                "task.output_row_begin > 1u",
                1,
            ),
            "expert capacity guard reversed": source.replace(
                "route_list_capacity_per_expert >\n"
                "            route_list_expert_stride",
                "route_list_capacity_per_expert <\n"
                "            route_list_expert_stride",
                1,
            ),
            "expert segment index narrowed": source.replace(
                "ulong(task.expert_index) *",
                "task.expert_index *",
                1,
            ),
            "expert segment capacity replaced": source.replace(
                "ulong(route_list_capacity_per_expert)",
                "ulong(route_list_expert_stride)",
                1,
            ),
            "expert segment row proof removed": source.replace(
                "task_rows <= expert_segment_end - task_begin",
                "task_rows <= route_list_total_extent - task_begin",
                1,
            ),
            "expert extent proof removed": source.replace(
                "expert_segment_end <= route_list_total_extent",
                "expert_segment_begin <= route_list_total_extent",
                1,
            ),
            "task-count guard widened": source.replace(
                "task_count <= kNativeRoutedQgemmR1TaskCapacity",
                "task_count < kNativeRoutedQgemmR1TaskCapacity",
                1,
            ),
            "group-x guard widened": source.replace(
                "group.x < task_count",
                "group.x <= task_count",
                1,
            ),
            "group-z ownership widened": source.replace(
                "group.z == 0u",
                "group.z <= 1u",
                1,
            ),
            "thread guard widened": source.replace(
                "threadgroup_shape.x == kNativeRoutedQgemmR1Threads",
                "threadgroup_shape.x <= kNativeRoutedQgemmR1Threads",
                1,
            ),
            "position guard widened": source.replace(
                "position < input_rows",
                "position <= input_rows",
                1,
            ),
            "slot guard widened": source.replace(
                "slot < kMoeActiveExperts",
                "slot <= kMoeActiveExperts",
                1,
            ),
            "group-y guard widened": source.replace(
                "if (group.y >=",
                "if (group.y >",
                1,
            ),
            "simdgroup guard widened": source.replace(
                "simdgroup < kNativeRoutedQgemmR1Simdgroups",
                "simdgroup <= kNativeRoutedQgemmR1Simdgroups",
                1,
            ),
            "list index narrowed": source.replace(
                "ulong(task.absolute_route_list_begin) +\n"
                "            ulong(local_row)",
                "task.absolute_route_list_begin + local_row",
                1,
            ),
            "hidden index narrowed": replace_in_function(
                source,
                "inline ulong native_routed_qgemm_r1_hidden_index",
                "ulong(kMoeExpertDimension) +",
                "kMoeExpertDimension +",
            ),
            "partial index narrowed": replace_in_function(
                source,
                "inline ulong native_routed_qgemm_r1_partial_index",
                "ulong(kHiddenDimension) +",
                "kHiddenDimension +",
            ),
            "signature binding changed": source.replace(
                "constant ulong& route_list_total_extent [[buffer(13)]]",
                "constant uint& route_list_total_extent [[buffer(13)]]",
                1,
            ),
            "shared branch added": source.replace(
                "const ulong expert_row_begin =",
                "const bool shared = task.expert_index == kMoeExperts;\n"
                "    const ulong expert_row_begin =",
                1,
            ),
            "threadgroup staging added": source.replace(
                "struct NativeRoutedQgemmR1Task {",
                "threadgroup bfloat staged_value;\n"
                "struct NativeRoutedQgemmR1Task {",
                1,
            ),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                self.assertNotEqual(mutation, source)
                with self.assertRaises(AssertionError):
                    validate_r1_source(mutation)


if __name__ == "__main__":
    unittest.main()
