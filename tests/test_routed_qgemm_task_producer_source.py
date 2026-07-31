import re
import unittest
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path

from tools.generate_kernel_library import (
    NATIVE_ROUTED_QGEMM_R1_TASK_CAPACITY,
    NATIVE_ROUTED_QGEMM_R1_TASK_STATUS_VALUES,
)


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = (
    REPOSITORY_ROOT
    / "src/backend/metal/kernels/routed_qgemm_tasks.metal"
)
PUBLIC_ABI_PATH = (
    REPOSITORY_ROOT / "include/tatara/runtime/quantized_gemm.h"
)
CMAKE_PATH = REPOSITORY_ROOT / "CMakeLists.txt"
PIPELINE_PROBE_PATH = (
    REPOSITORY_ROOT / "tools/native/prefill_library_probe.cpp"
)
TILE_ROWS = 16
PACKED_SLOT_BITS = 4
KERNEL_NAME = "native_routed_qgemm_r1_build_tasks"
KERNEL_ARGUMENTS = (
    "device const uint* counts [[buffer(0)]]",
    "device const uint* route_list [[buffer(1)]]",
    (
        "device NativeRoutedQgemmR1ProducedTask* tasks "
        "[[buffer(2)]]"
    ),
    "device uint* indirect_arguments [[buffer(3)]]",
    "device uint* status [[buffer(4)]]",
    "constant uint& position_count [[buffer(5)]]",
    "constant uint& expert_count [[buffer(6)]]",
    "constant uint& active_expert_count [[buffer(7)]]",
    "constant uint& route_list_expert_stride [[buffer(8)]]",
    "constant uint& position_capacity [[buffer(9)]]",
    "constant ulong& route_list_total_extent [[buffer(10)]]",
    "constant uint& planned_task_capacity [[buffer(11)]]",
    "constant uint& column_groups [[buffer(12)]]",
    "constant uint& packed_slot_bits [[buffer(13)]]",
    "constant uint& include_shared_expert [[buffer(14)]]",
    "uint3 thread_position [[thread_position_in_grid]]",
)
HELPER_ARGUMENTS = (
    "device const uint* counts",
    "device const uint* route_list",
    "device NativeRoutedQgemmR1ProducedTask* tasks",
    "uint position_count",
    "uint expert_count",
    "uint active_expert_count",
    "uint route_list_expert_stride",
    "uint position_capacity",
    "ulong route_list_total_extent",
    "uint planned_task_capacity",
    "uint column_groups",
    "uint packed_slot_bits",
    "uint include_shared_expert",
    "thread uint& produced_task_count",
)
HELPER_CALL_ARGUMENTS = (
    "counts",
    "route_list",
    "tasks",
    "position_count",
    "expert_count",
    "active_expert_count",
    "route_list_expert_stride",
    "position_capacity",
    "route_list_total_extent",
    "planned_task_capacity",
    "column_groups",
    "packed_slot_bits",
    "include_shared_expert",
    "task_count",
)
TASK_INITIALIZER = (
    "expert",
    "uint(segment_begin + ulong(local_begin))",
    "row_count",
    "0u",
)
PIPELINE_FUNCTION_NAMES = (
    "embed_rows_q4",
    "rms_blk",
    "residual_blk",
    "gdn_project_blk",
    "gdn_conv_blk",
    "gdn_gates_blk",
    "gdn_recurrence",
    "gdn_recurrence_blk",
    "gdn_recurrence_gates_blk",
    "gdn_gate_norm_blk",
    "attn_project_blk",
    "attn_qk_rope_blk",
    "attention_partial_blk",
    "attention_combine_blk",
    "attention_staged_scores_blk",
    "attention_staged_softmax_blk",
    "attention_staged_values_blk",
    "attention_flash_v2_blk",
    "attention_staged_softmax_bf16_blk",
    "tatara_mlx_steel_attn_scores_nt",
    "tatara_mlx_steel_attn_values_nn",
    "tatara_mlx_steel_attn_scores_nt_unaligned",
    "tatara_mlx_steel_attn_values_nn_unaligned",
    "attention_gate_apply_blk",
    "tatara_mlx_steel_attn_scores_nt_unaligned_l",
    "tatara_mlx_steel_attn_values_nn_unaligned_l",
    "gdn_conv_tape_blk",
    "gdn_recurrence_gates_tape_blk",
    "gdn_tape_replay_blk",
    "attention_streaming_blk",
    "attention_decode_scores_gqa4_simdreduce",
    "attention_decode_scores_gqa8",
    "attention_decode_scores_values_gqa8",
    "attention_decode_vector_2pass_part",
    "attention_decode_vector_2pass_combine",
    "attention_decode_values_gqa8_t512",
    "outproj_blk",
    "router_q8_block",
    "router_select_block",
    "router_select_block_parallel",
    "expert_union",
    "expert_union_fused_tasks",
    "block_upgate",
    "block_down_partial",
    "block_down_combine",
    "native_dense_qgemm_q4_bf16_n1",
    "native_routed_qgemm_r1_fused_upgate_swiglu",
    "native_routed_qgemm_r1_gate",
    "native_routed_qgemm_r1_up_swiglu",
    "native_routed_qgemm_r1_down_partial",
    "native_routed_qgemm_r1_build_tasks",
    "native_routed_qgemm_r2_fused_upgate_swiglu",
    "native_routed_qgemm_r2_down_partial",
    "draft_dense_bf16",
    "draft_gemm16_bf16",
    "draft_rms_rows",
    "draft_rope128",
    "draft_block_attention",
    "draft_swiglu",
    "draft_residual_add",
    "verify_dense_q4_m16",
)


class ProducerStatus(IntEnum):
    NOT_PRODUCED = 0
    READY = 1
    COUNT_OUT_OF_RANGE = 2
    ROUTE_CONSERVATION_FAILURE = 3
    TASK_CAPACITY_EXCEEDED = 4
    PACKED_SLOT_OUT_OF_RANGE = 5


@dataclass(frozen=True)
class ProducedTask:
    expert_index: int
    absolute_route_list_begin: int
    row_count: int
    output_row_begin: int = 0


@dataclass(frozen=True)
class ProducerResult:
    status: ProducerStatus
    tasks: tuple[ProducedTask, ...] = ()
    grid: tuple[int, int, int] = (0, 0, 0)


def compact(source: str) -> str:
    return re.sub(r"\s+", "", source)


def extract_braced_block(source: str, marker: str) -> str:
    marker_begin = source.index(marker)
    body_begin = source.index("{", marker_begin)
    depth = 0
    for index in range(body_begin, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[body_begin : index + 1]
    raise AssertionError(f"unterminated braced block after {marker!r}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def require_order(
    source: str, markers: tuple[str, ...], domain: str
) -> None:
    positions = []
    cursor = 0
    for marker in markers:
        position = source.find(marker, cursor)
        require(position >= 0, f"{domain}: missing {marker!r}")
        positions.append(position)
        cursor = position + len(marker)
    require(
        positions == sorted(positions),
        f"{domain}: operation order changed",
    )


def require_exact_signature(
    source: str, declaration: str, expected: tuple[str, ...]
) -> None:
    marker = f"{declaration}("
    begin = source.index(marker) + len(marker)
    end = source.index(") {", begin)
    actual = tuple(
        " ".join(argument.split())
        for argument in source[begin:end].split(",")
    )
    normalized_expected = tuple(
        " ".join(argument.split()) for argument in expected
    )
    require(
        actual == normalized_expected,
        "producer kernel signature changed",
    )


def require_exact_call(
    source: str, function_name: str, expected: tuple[str, ...]
) -> None:
    marker = f"{function_name}("
    begin = source.index(marker) + len(marker)
    end = source.index(");", begin)
    actual = tuple(
        " ".join(argument.split())
        for argument in source[begin:end].split(",")
    )
    require(
        actual == expected,
        f"{function_name} call argument order changed",
    )


def require_exact_task_initializer(
    source: str, expected: tuple[str, ...]
) -> None:
    marker = "tasks[task_index++] = {"
    begin = source.index(marker) + len(marker)
    end = source.index("};", begin)
    actual = tuple(
        " ".join(field.split()).rstrip(",")
        for field in source[begin:end].splitlines()
        if field.strip()
    )
    require(actual == expected, "producer task initializer changed")


def validate_pipeline_probe(source: str) -> None:
    marker = "constexpr std::array<std::string_view, 61> kFunctions"
    functions_block = extract_braced_block(source, marker)
    actual_names = tuple(re.findall(r'"([^"]+)"', functions_block))
    require(
        actual_names == PIPELINE_FUNCTION_NAMES,
        "pipeline probe lookup names or order changed",
    )
    require(
        source.count("create_indirect_compute_pipeline(") == 2,
        "pipeline probe must compile both native and Steel inventories for ICB",
    )
    require(
        "create_compute_pipeline(" not in source,
        "pipeline probe admitted a non-ICB pipeline",
    )
    forbidden_includes = re.findall(
        r'^#include\s+([<"].*[>"])$', source, flags=re.MULTILINE
    )
    for include in forbidden_includes:
        require(
            re.search(
                r"command|dispatch|encoder|queue", include, re.IGNORECASE
            )
            is None,
            f"pipeline-only probe admitted execution include {include}",
        )
    for forbidden_api in (
        "create_command_queue(",
        "create_command_buffer(",
        "begin_compute_pass(",
        "end_compute_pass(",
        "dispatch_threadgroups(",
        "dispatch_threads(",
        "commit(",
        "wait_until_completed(",
        "newCommandQueue(",
        "commandBuffer(",
        "computeCommandEncoder(",
        "dispatchThreadgroups(",
        "dispatchThreads(",
        "waitUntilCompleted(",
    ):
        require(
            forbidden_api not in source,
            f"pipeline-only probe admitted execution API {forbidden_api}",
        )
    source_without_strings = re.sub(
        r'"(?:\\.|[^"\\])*"', '""', source
    )
    for forbidden_pattern in (
        r"\b(?:create_)?command_?queue\s*\(",
        r"\b(?:create_)?command_?buffer\s*\(",
        r"\b(?:begin_|end_)?compute_(?:pass|encoder)\s*\(",
        r"\bdispatch[A-Za-z0-9_]*\s*\(",
        r"\bcommit\s*\(",
        r"\bwait[A-Za-z0-9_]*\s*\(",
    ):
        require(
            re.search(
                forbidden_pattern, source_without_strings, re.IGNORECASE
            )
            is None,
            (
                "pipeline-only probe admitted execution pattern "
                f"{forbidden_pattern}"
            ),
        )
    for forbidden_type in (
        "MetalCommandQueue",
        "MetalCommandBuffer",
        "MetalComputePass",
        "MTL::CommandQueue",
        "MTL::CommandBuffer",
        "MTL::ComputeCommandEncoder",
    ):
        require(
            forbidden_type not in source_without_strings,
            f"pipeline-only probe admitted execution type {forbidden_type}",
        )


def parse_public_status_values(source: str) -> dict[str, int]:
    marker = "enum class QuantizedGemmDeviceTaskStatus"
    block = extract_braced_block(source, marker)
    return {
        name: int(value)
        for name, value in re.findall(
            r"\b([A-Za-z][A-Za-z0-9]*)\s*=\s*(\d+)", block
        )
    }


def pack_route(position: int, slot: int) -> int:
    return (position << PACKED_SLOT_BITS) | slot


def make_valid_routes(
    expert_count: int,
    active_experts: int,
    position_count: int,
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    lists: list[list[int]] = [[] for _ in range(expert_count)]
    for position in range(position_count):
        for slot in range(active_experts):
            expert = (position * active_experts + slot) % expert_count
            lists[expert].append(pack_route(position, slot))
    return (
        tuple(len(entries) for entries in lists),
        tuple(tuple(entries) for entries in lists),
    )


def produce_reference(
    counts: tuple[int, ...],
    route_lists: tuple[tuple[int, ...], ...],
    *,
    active_experts: int,
    position_count: int,
    expert_stride: int,
    position_capacity: int,
    total_extent: int,
    planned_task_capacity: int,
    column_groups: int,
    maximum_column_groups: int,
    include_shared_expert: bool = False,
) -> ProducerResult:
    if len(counts) != len(route_lists):
        raise ValueError("counts and routed list prefixes must agree")
    if (
        not counts
        or column_groups <= 0
        or column_groups > maximum_column_groups
        or position_count <= 0
        or position_count > position_capacity
        or position_capacity > expert_stride
    ):
        return ProducerResult(ProducerStatus.COUNT_OUT_OF_RANGE)
    required_entries = (
        (len(counts) - 1) * expert_stride + position_capacity
    )
    if required_entries > total_extent:
        return ProducerResult(ProducerStatus.PACKED_SLOT_OUT_OF_RANGE)

    routed_count = 0
    required_tasks = 0
    for expert, count in enumerate(counts):
        if (
            count < 0
            or count > position_count
            or count > position_capacity
        ):
            return ProducerResult(ProducerStatus.COUNT_OUT_OF_RANGE)
        segment_begin = expert * expert_stride
        segment_end = segment_begin + position_capacity
        listed_end = segment_begin + count
        if (
            segment_end > total_extent
            or listed_end > segment_end
            or (count != 0 and listed_end > 1 << 32)
        ):
            return ProducerResult(
                ProducerStatus.PACKED_SLOT_OUT_OF_RANGE
            )
        routed_count += count
        required_tasks += (count + TILE_ROWS - 1) // TILE_ROWS

    if routed_count != position_count * active_experts:
        return ProducerResult(
            ProducerStatus.ROUTE_CONSERVATION_FAILURE
        )
    if (
        planned_task_capacity > NATIVE_ROUTED_QGEMM_R1_TASK_CAPACITY
        or required_tasks > planned_task_capacity
    ):
        return ProducerResult(ProducerStatus.TASK_CAPACITY_EXCEEDED)

    routed_active_experts = (
        active_experts - 1
        if include_shared_expert
        else active_experts
    )
    for expert, (count, entries) in enumerate(
        zip(counts, route_lists, strict=True)
    ):
        if len(entries) < count:
            return ProducerResult(
                ProducerStatus.PACKED_SLOT_OUT_OF_RANGE
            )
        for packed in entries[:count]:
            position = packed >> PACKED_SLOT_BITS
            slot = packed & ((1 << PACKED_SLOT_BITS) - 1)
            shared = (
                include_shared_expert and expert == len(counts) - 1
            )
            valid_slot = (
                slot == routed_active_experts
                if shared
                else slot < routed_active_experts
            )
            if position >= position_count or not valid_slot:
                return ProducerResult(
                    ProducerStatus.PACKED_SLOT_OUT_OF_RANGE
                )

    tasks = []
    for expert, count in enumerate(counts):
        segment_begin = expert * expert_stride
        for local_begin in range(0, count, TILE_ROWS):
            tasks.append(
                ProducedTask(
                    expert_index=expert,
                    absolute_route_list_begin=(
                        segment_begin + local_begin
                    ),
                    row_count=min(TILE_ROWS, count - local_begin),
                )
            )
    return ProducerResult(
        status=ProducerStatus.READY,
        tasks=tuple(tasks),
        grid=(len(tasks), column_groups, 1),
    )


def validate_source(source: str) -> None:
    helper = extract_braced_block(
        source, "inline uint native_routed_qgemm_r1_produce_tasks("
    )
    kernel = extract_braced_block(
        source, f"kernel void {KERNEL_NAME}("
    )
    compact_source = compact(source)
    compact_helper = compact(helper)
    compact_kernel = compact(kernel)

    require_exact_signature(
        source, f"kernel void {KERNEL_NAME}", KERNEL_ARGUMENTS
    )
    require_exact_signature(
        source,
        "inline uint native_routed_qgemm_r1_produce_tasks",
        HELPER_ARGUMENTS,
    )
    require_exact_call(
        kernel,
        "native_routed_qgemm_r1_produce_tasks",
        HELPER_CALL_ARGUMENTS,
    )
    require_exact_task_initializer(helper, TASK_INITIALIZER)
    require(
        re.search(
            r"struct NativeRoutedQgemmR1ProducedTask \{\s*"
            r"uint expert_index;\s*"
            r"uint absolute_route_list_begin;\s*"
            r"uint row_count;\s*"
            r"uint output_row_begin;\s*"
            r"\};",
            source,
        )
        is not None,
        "producer task field order changed",
    )
    require(
        "sizeof(NativeRoutedQgemmR1ProducedTask)=="
        "kNativeRoutedQgemmR1TaskBytes"
        in compact_source,
        "producer no longer binds the physical task ABI",
    )
    require(
        "kMoeActiveExperts<(1u<<kPrefillPackedSlotBits)"
        in compact_source,
        "packed routed-slot bound is missing",
    )
    require(
        "for(uintexpert=0u;expert<expert_count;++expert)"
        in compact_helper,
        "producer no longer iterates strict task experts",
    )
    require(
        compact_helper.count(
            "for(uintexpert=0u;expert<expert_count;++expert)"
        )
        == 3,
        "all three producer phases must use strict task bounds",
    )
    for forbidden in (
        "expert<=expert_count",
        "kMoeRouterRows",
        "kMoeSlotCount",
    ):
        require(
            forbidden not in compact_source,
            f"producer widened into shared ownership: {forbidden!r}",
        )

    for required in (
        "include_shared_expert>1u",
        "expert_count!=kMoeExperts+include_shared_expert",
        "active_expert_count!="
        "kMoeActiveExperts+include_shared_expert",
        "packed_slot_bits!=kPrefillPackedSlotBits",
        "constuintmaximum_column_groups=max("
        "kMoeExpertDimension/kNativeRoutedQgemmR1TileColumns,"
        "kHiddenDimension/kNativeRoutedQgemmR1TileColumns)",
        "column_groups==0u",
        "column_groups>maximum_column_groups",
        "position_count>maximum_position_count",
        "position_capacity>maximum_position_count",
        "position_count>position_capacity",
        "position_capacity>route_list_expert_stride",
        "constulongfinal_segment_begin="
        "ulong(expert_count-1u)*ulong(route_list_expert_stride)",
        "constulongrequired_list_entries="
        "final_segment_begin+ulong(position_capacity)",
        "required_list_entries>route_list_total_extent",
        "count>position_count",
        "count>position_capacity",
        "constulongsegment_begin="
        "ulong(expert)*ulong(route_list_expert_stride)",
        "segment_end>route_list_total_extent",
        "listed_end>segment_end",
        "count!=0u&&listed_end>0x100000000ul",
        "route_list[segment_begin+ulong(local_row)]",
        "packed>>packed_slot_bits",
        "constboolshared=expert==kMoeExperts",
        "shared?slot==kMoeActiveExperts:"
        "slot<kMoeActiveExperts",
        "routed_count+=ulong(count)",
        "ulong(position_count)*ulong(active_expert_count)",
        "routed_count!=expected_routed_count",
        "required_task_count>ulong(planned_task_capacity)",
        "planned_task_capacity>"
        "kNativeRoutedQgemmR1TaskCapacity",
        "uint(segment_begin+ulong(local_begin))",
        "min(kNativeRoutedQgemmR1TileRows,count-local_begin)",
        "0u,",
    ):
        require(
            required in compact_helper,
            f"producer validation/emission omits {required!r}",
        )
    require_order(
        helper,
        (
            "include_shared_expert > 1u",
            "expert_count !=",
            "active_expert_count !=",
            "packed_slot_bits != kPrefillPackedSlotBits",
            "column_groups == 0u",
            "1u << (32u - packed_slot_bits)",
            (
                "ulong(expert_count - 1u) * "
                "ulong(route_list_expert_stride)"
            ),
        ),
        "runtime plan binding before derived arithmetic",
    )
    require(
        compact_helper.count("route_list[") == 1,
        "producer must have exactly one packed-route read expression",
    )
    require(
        compact_helper.count(
            "route_list[segment_begin+ulong(local_row)]"
        )
        == 1,
        "producer must read packed routes only in phase three",
    )
    require_order(
        helper,
        (
            "routed_count += ulong(count);",
            "const ulong expected_routed_count =",
            "routed_count != expected_routed_count",
            "planned_task_capacity >",
            "required_task_count > ulong(planned_task_capacity)",
            "for (uint local_row = 0u; local_row < count;",
            "route_list[segment_begin + ulong(local_row)]",
            "const uint position =",
            "packed >> packed_slot_bits",
            "const uint slot =",
            "packed & ((1u << packed_slot_bits) - 1u)",
            "const bool shared =",
            "const bool valid_slot =",
            "position >= position_count",
            "!valid_slot",
            "uint task_index = 0u;",
        ),
        "count, capacity, packed-list, and emission phases",
    )

    require_order(
        kernel,
        (
            "indirect_arguments[0] = 0u;",
            "indirect_arguments[1] = 0u;",
            "indirect_arguments[2] = 0u;",
            (
                "status[0] = "
                "kNativeRoutedQgemmR1TaskStatusNotProduced;"
            ),
            "native_routed_qgemm_r1_produce_tasks(",
            "indirect_arguments[0] = task_count;",
            "indirect_arguments[1] = column_groups;",
            "indirect_arguments[2] = 1u;",
            "status[0] = result;",
            "threadgroup_barrier(mem_flags::mem_device);",
            (
                "status[0] = "
                "kNativeRoutedQgemmR1TaskStatusReady;"
            ),
        ),
        "fail-closed publication",
    )
    for required in (
        "constboolproducer=all(thread_position==uint3(0u))",
        "if(producer)",
        "if(result==kNativeRoutedQgemmR1TaskStatusReady)",
        "if(producer&&"
        "result==kNativeRoutedQgemmR1TaskStatusReady)",
    ):
        require(
            required in compact_kernel,
            f"one-thread publication omits {required!r}",
        )
    require(
        compact_kernel.rfind(
            "status[0]=kNativeRoutedQgemmR1TaskStatusReady;"
        )
        > compact_kernel.rfind(
            "threadgroup_barrier(mem_flags::mem_device);"
        ),
        "Ready must be published only after the device barrier",
    )


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise AssertionError(f"mutation marker not unique: {before!r}")
    return source.replace(before, after, 1)


class RoutedQgemmTaskProducerSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")

    def test_public_status_values_match_generated_metal_values(self) -> None:
        public_values = parse_public_status_values(
            PUBLIC_ABI_PATH.read_text(encoding="utf-8")
        )
        generated_values = dict(
            NATIVE_ROUTED_QGEMM_R1_TASK_STATUS_VALUES
        )

        self.assertEqual(public_values, generated_values)
        self.assertEqual(
            generated_values,
            {
                "NotProduced": 0,
                "Ready": 1,
                "CountOutOfRange": 2,
                "RouteConservationFailure": 3,
                "TaskCapacityExceeded": 4,
                "PackedSlotOutOfRange": 5,
            },
        )

    def test_source_satisfies_fail_closed_producer_contract(self) -> None:
        validate_source(self.source)

    def test_reference_oracle_partitions_every_routed_list_once(self) -> None:
        counts, route_lists = make_valid_routes(
            expert_count=7,
            active_experts=3,
            position_count=37,
        )
        result = produce_reference(
            counts,
            route_lists,
            active_experts=3,
            position_count=37,
            expert_stride=40,
            position_capacity=37,
            total_extent=(7 - 1) * 40 + 37,
            planned_task_capacity=64,
            column_groups=17,
            maximum_column_groups=17,
        )

        self.assertEqual(result.status, ProducerStatus.READY)
        self.assertEqual(result.grid, (len(result.tasks), 17, 1))
        self.assertTrue(
            all(task.expert_index < len(counts) for task in result.tasks)
        )
        self.assertTrue(
            all(task.output_row_begin == 0 for task in result.tasks)
        )
        cursor = 0
        for expert, count in enumerate(counts):
            expert_tasks = [
                task
                for task in result.tasks
                if task.expert_index == expert
            ]
            covered = []
            for task in expert_tasks:
                covered.extend(
                    range(
                        task.absolute_route_list_begin,
                        task.absolute_route_list_begin + task.row_count,
                    )
                )
            self.assertEqual(
                covered,
                list(
                    range(
                        expert * 40,
                        expert * 40 + count,
                    )
                ),
            )
            cursor += count
        self.assertEqual(cursor, 37 * 3)

    def test_reference_oracle_accepts_only_the_public_valid_tail(self) -> None:
        counts, route_lists = make_valid_routes(8, 4, 32)
        minimal_extent = (8 - 1) * 40 + 32
        ready = produce_reference(
            counts,
            route_lists,
            active_experts=4,
            position_count=32,
            expert_stride=40,
            position_capacity=32,
            total_extent=minimal_extent,
            planned_task_capacity=64,
            column_groups=16,
            maximum_column_groups=16,
        )
        short = produce_reference(
            counts,
            route_lists,
            active_experts=4,
            position_count=32,
            expert_stride=40,
            position_capacity=32,
            total_extent=minimal_extent - 1,
            planned_task_capacity=64,
            column_groups=16,
            maximum_column_groups=16,
        )

        self.assertEqual(ready.status, ProducerStatus.READY)
        self.assertEqual(
            short.status, ProducerStatus.PACKED_SLOT_OUT_OF_RANGE
        )
        self.assertEqual(short.grid, (0, 0, 0))
        self.assertEqual(short.tasks, ())

    def test_reference_oracle_assimilates_one_typed_shared_segment(
        self,
    ) -> None:
        for position_count in (1, 15, 16, 17):
            with self.subTest(position_count=position_count):
                routed_counts, routed_lists = make_valid_routes(
                    3, 2, position_count
                )
                shared_list = tuple(
                    pack_route(position, 2)
                    for position in range(position_count)
                )
                counts = routed_counts + (position_count,)
                route_lists = routed_lists + (shared_list,)
                result = produce_reference(
                    counts,
                    route_lists,
                    active_experts=3,
                    position_count=position_count,
                    expert_stride=17,
                    position_capacity=17,
                    total_extent=4 * 17,
                    planned_task_capacity=16,
                    column_groups=5,
                    maximum_column_groups=5,
                    include_shared_expert=True,
                )
                self.assertEqual(result.status, ProducerStatus.READY)
                self.assertEqual(
                    sum(counts), position_count * 3
                )
                shared_tasks = tuple(
                    task
                    for task in result.tasks
                    if task.expert_index == 3
                )
                expected_rows = (
                    (position_count,)
                    if position_count <= TILE_ROWS
                    else (TILE_ROWS, position_count - TILE_ROWS)
                )
                self.assertEqual(
                    tuple(task.row_count for task in shared_tasks),
                    expected_rows,
                )

        routed_counts, routed_lists = make_valid_routes(3, 2, 17)
        shared_list = tuple(
            pack_route(position, 2) for position in range(17)
        )
        counts = routed_counts + (17,)
        route_lists = routed_lists + (shared_list,)
        wrong_shared = route_lists[:-1] + (
            (pack_route(0, 1),) + shared_list[1:],
        )
        rejected = produce_reference(
            counts,
            wrong_shared,
            active_experts=3,
            position_count=17,
            expert_stride=17,
            position_capacity=17,
            total_extent=4 * 17,
            planned_task_capacity=16,
            column_groups=5,
            maximum_column_groups=5,
            include_shared_expert=True,
        )
        self.assertEqual(
            rejected.status,
            ProducerStatus.PACKED_SLOT_OUT_OF_RANGE,
        )

    def test_reference_oracle_returns_each_typed_failure_with_zero_grid(
        self,
    ) -> None:
        counts, route_lists = make_valid_routes(
            expert_count=8,
            active_experts=4,
            position_count=32,
        )
        ready = produce_reference(
            counts,
            route_lists,
            active_experts=4,
            position_count=32,
            expert_stride=32,
            position_capacity=32,
            total_extent=8 * 32,
            planned_task_capacity=64,
            column_groups=16,
            maximum_column_groups=16,
        )
        bad_position_lists = list(route_lists)
        bad_position_lists[0] = (
            pack_route(32, 0),
        ) + bad_position_lists[0][1:]
        bad_slot_lists = list(route_lists)
        bad_slot_lists[0] = (
            pack_route(0, 4),
        ) + bad_slot_lists[0][1:]
        unconserved_counts = list(counts)
        unconserved_lists = list(route_lists)
        unconserved_counts[0] -= 1
        unconserved_lists[0] = unconserved_lists[0][
            : unconserved_counts[0]
        ]
        cases = {
            "count": produce_reference(
                (33,) + counts[1:],
                route_lists,
                active_experts=4,
                position_count=32,
                expert_stride=32,
                position_capacity=32,
                total_extent=8 * 32,
                planned_task_capacity=64,
                column_groups=16,
                maximum_column_groups=16,
            ),
            "list extent": produce_reference(
                counts,
                route_lists,
                active_experts=4,
                position_count=32,
                expert_stride=32,
                position_capacity=32,
                total_extent=8 * 32 - 1,
                planned_task_capacity=64,
                column_groups=16,
                maximum_column_groups=16,
            ),
            "packed position": produce_reference(
                counts,
                tuple(bad_position_lists),
                active_experts=4,
                position_count=32,
                expert_stride=32,
                position_capacity=32,
                total_extent=8 * 32,
                planned_task_capacity=64,
                column_groups=16,
                maximum_column_groups=16,
            ),
            "packed slot": produce_reference(
                counts,
                tuple(bad_slot_lists),
                active_experts=4,
                position_count=32,
                expert_stride=32,
                position_capacity=32,
                total_extent=8 * 32,
                planned_task_capacity=64,
                column_groups=16,
                maximum_column_groups=16,
            ),
            "conservation": produce_reference(
                tuple(unconserved_counts),
                tuple(unconserved_lists),
                active_experts=4,
                position_count=32,
                expert_stride=32,
                position_capacity=32,
                total_extent=8 * 32,
                planned_task_capacity=64,
                column_groups=16,
                maximum_column_groups=16,
            ),
            "capacity": produce_reference(
                counts,
                route_lists,
                active_experts=4,
                position_count=32,
                expert_stride=32,
                position_capacity=32,
                total_extent=8 * 32,
                planned_task_capacity=len(ready.tasks) - 1,
                column_groups=16,
                maximum_column_groups=16,
            ),
            "zero column groups": produce_reference(
                counts,
                route_lists,
                active_experts=4,
                position_count=32,
                expert_stride=32,
                position_capacity=32,
                total_extent=8 * 32,
                planned_task_capacity=64,
                column_groups=0,
                maximum_column_groups=16,
            ),
            "column groups exceed plan": produce_reference(
                counts,
                route_lists,
                active_experts=4,
                position_count=32,
                expert_stride=32,
                position_capacity=32,
                total_extent=8 * 32,
                planned_task_capacity=64,
                column_groups=17,
                maximum_column_groups=16,
            ),
        }
        expected = {
            "count": ProducerStatus.COUNT_OUT_OF_RANGE,
            "list extent": ProducerStatus.PACKED_SLOT_OUT_OF_RANGE,
            "packed position": ProducerStatus.PACKED_SLOT_OUT_OF_RANGE,
            "packed slot": ProducerStatus.PACKED_SLOT_OUT_OF_RANGE,
            "conservation": ProducerStatus.ROUTE_CONSERVATION_FAILURE,
            "capacity": ProducerStatus.TASK_CAPACITY_EXCEEDED,
            "zero column groups": ProducerStatus.COUNT_OUT_OF_RANGE,
            "column groups exceed plan": ProducerStatus.COUNT_OUT_OF_RANGE,
        }
        for name, result in cases.items():
            with self.subTest(name=name):
                self.assertEqual(result.status, expected[name])
                self.assertEqual(result.grid, (0, 0, 0))
                self.assertEqual(result.tasks, ())

    def test_task_capacity_rejection_precedes_packed_route_reads(
        self,
    ) -> None:
        counts = (32, 32, 32, 32, 0, 0, 0, 0)
        route_lists = tuple(
            tuple(
                pack_route(position, expert % 4)
                for position in range(count)
            )
            for expert, count in enumerate(counts)
        )
        poisoned = list(route_lists)
        poisoned[0] = (pack_route(32, 0),) + poisoned[0][1:]

        low_capacity = produce_reference(
            counts,
            tuple(poisoned),
            active_experts=4,
            position_count=32,
            expert_stride=40,
            position_capacity=32,
            total_extent=(8 - 1) * 40 + 32,
            planned_task_capacity=7,
            column_groups=16,
            maximum_column_groups=16,
        )
        above_hard_capacity = produce_reference(
            counts,
            tuple(poisoned),
            active_experts=4,
            position_count=32,
            expert_stride=40,
            position_capacity=32,
            total_extent=(8 - 1) * 40 + 32,
            planned_task_capacity=(
                NATIVE_ROUTED_QGEMM_R1_TASK_CAPACITY + 1
            ),
            column_groups=16,
            maximum_column_groups=16,
        )

        self.assertEqual(
            low_capacity.status, ProducerStatus.TASK_CAPACITY_EXCEEDED
        )
        self.assertEqual(
            above_hard_capacity.status,
            ProducerStatus.TASK_CAPACITY_EXCEEDED,
        )

    def test_reference_oracle_rejects_a_shared_count_row(self) -> None:
        counts, route_lists = make_valid_routes(8, 4, 8)

        with self.assertRaisesRegex(ValueError, "must agree"):
            produce_reference(
                counts + (8,),
                route_lists,
                active_experts=4,
                position_count=8,
                expert_stride=8,
                position_capacity=8,
                total_extent=8 * 8,
                planned_task_capacity=32,
                column_groups=1,
                maximum_column_groups=1,
            )

    def test_pipeline_probe_is_exactly_lookup_only(self) -> None:
        probe = PIPELINE_PROBE_PATH.read_text(encoding="utf-8")
        cmake = CMAKE_PATH.read_text(encoding="utf-8")

        validate_pipeline_probe(probe)
        self.assertEqual(len(PIPELINE_FUNCTION_NAMES), 61)
        self.assertIn(
            '"  command buffers submitted: 0\\n"', probe
        )
        self.assertEqual(
            cmake.count("routed_qgemm_tasks.metal"), 1
        )

    def test_pipeline_probe_execution_mutations_are_rejected(self) -> None:
        probe = PIPELINE_PROBE_PATH.read_text(encoding="utf-8")
        insertion_point = "    return 0;"
        mutations = {
            "lookup order": replace_once(
                probe,
                (
                    '        "native_routed_qgemm_r1_gate",\n'
                    '        "native_routed_qgemm_r1_up_swiglu",'
                ),
                (
                    '        "native_routed_qgemm_r1_up_swiglu",\n'
                    '        "native_routed_qgemm_r1_gate",'
                ),
            ),
            "command include": replace_once(
                probe,
                '#include "tatara/backend/metal/pipeline.h"',
                (
                    '#include "tatara/backend/metal/pipeline.h"\n'
                    '#include "tatara/backend/metal/commands.h"'
                ),
            ),
            "queue replacement": replace_once(
                probe,
                "create_system_device(",
                "create_command_queue(",
            ),
            "indirect capability": replace_once(
                probe,
                (
                    "        auto pipeline =\n"
                    "            create_indirect_compute_pipeline("
                    "*device.device, *function.function);"
                ),
                (
                    "        auto pipeline =\n"
                    "            create_compute_pipeline("
                    "*device.device, *function.function);"
                ),
            ),
            "command buffer": replace_once(
                probe,
                insertion_point,
                "    create_command_buffer({});\n" + insertion_point,
            ),
            "encoder": replace_once(
                probe,
                insertion_point,
                "    begin_compute_pass({});\n" + insertion_point,
            ),
            "dispatch": replace_once(
                probe,
                insertion_point,
                "    dispatch_threadgroups({});\n" + insertion_point,
            ),
            "commit": replace_once(
                probe,
                insertion_point,
                "    commit({});\n" + insertion_point,
            ),
            "wait": replace_once(
                probe,
                insertion_point,
                "    wait_until_completed({});\n" + insertion_point,
            ),
        }
        for name, mutated in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    validate_pipeline_probe(mutated)

    def test_dangerous_source_mutations_are_rejected(self) -> None:
        mutations = {
            "expert loop widened": replace_once(
                self.source,
                (
                    "for (uint expert = 0u; "
                    "expert < expert_count; ++expert) {\n"
                    "        const uint count = counts[expert];\n"
                    "        if (count > position_count"
                ),
                (
                    "for (uint expert = 0u; "
                    "expert <= expert_count; ++expert) {\n"
                    "        const uint count = counts[expert];\n"
                    "        if (count > position_count"
                ),
            ),
            "expert plan binding removed": replace_once(
                self.source,
                (
                    "expert_count !=\n"
                    "            kMoeExperts + include_shared_expert"
                ),
                (
                    "expert_count ==\n"
                    "            kMoeExperts + include_shared_expert"
                ),
            ),
            "active count binding removed": replace_once(
                self.source,
                (
                    "active_expert_count !=\n"
                    "            kMoeActiveExperts + include_shared_expert"
                ),
                (
                    "active_expert_count ==\n"
                    "            kMoeActiveExperts + include_shared_expert"
                ),
            ),
            "include flag widened": replace_once(
                self.source,
                "include_shared_expert > 1u",
                "include_shared_expert > 2u",
            ),
            "packed width binding removed": replace_once(
                self.source,
                "packed_slot_bits != kPrefillPackedSlotBits",
                "packed_slot_bits == kPrefillPackedSlotBits",
            ),
            "count guard reversed": replace_once(
                self.source,
                "count > position_count",
                "count < position_count",
            ),
            "zero column groups accepted": replace_once(
                self.source,
                "column_groups == 0u",
                "column_groups < 0u",
            ),
            "column plan bound reversed": replace_once(
                self.source,
                "column_groups > maximum_column_groups",
                "column_groups < maximum_column_groups",
            ),
            "capacity exceeds stride": replace_once(
                self.source,
                "position_capacity > route_list_expert_stride",
                "position_capacity < route_list_expert_stride",
            ),
            "minimal tail widened to full stride": replace_once(
                self.source,
                "expert_count - 1u",
                "expert_count",
            ),
            "minimal tail rejected": replace_once(
                self.source,
                (
                    "required_list_entries > "
                    "route_list_total_extent"
                ),
                (
                    "required_list_entries >= "
                    "route_list_total_extent"
                ),
            ),
            "list extent weakened": replace_once(
                self.source,
                (
                    "required_list_entries > "
                    "route_list_total_extent"
                ),
                (
                    "required_list_entries < "
                    "route_list_total_extent"
                ),
            ),
            "packed read moved before task caps": replace_once(
                self.source,
                "planned_task_capacity >",
                "route_list[0];\n    planned_task_capacity >",
            ),
            "routed slot admits shared row": replace_once(
                self.source,
                "slot < kMoeActiveExperts",
                "slot <= kMoeActiveExperts",
            ),
            "shared slot admits routed row": replace_once(
                self.source,
                "slot == kMoeActiveExperts",
                "slot <= kMoeActiveExperts",
            ),
            "conservation reversed": replace_once(
                self.source,
                "routed_count != expected_routed_count",
                "routed_count == expected_routed_count",
            ),
            "task cap reversed": replace_once(
                self.source,
                "required_task_count > ulong(planned_task_capacity)",
                "required_task_count < ulong(planned_task_capacity)",
            ),
            "producer call capacities swapped": replace_once(
                self.source,
                (
                    "            planned_task_capacity,\n"
                    "            column_groups,"
                ),
                (
                    "            column_groups,\n"
                    "            planned_task_capacity,"
                ),
            ),
            "helper capacities swapped": replace_once(
                self.source,
                (
                    "    uint planned_task_capacity,\n"
                    "    uint column_groups,"
                ),
                (
                    "    uint column_groups,\n"
                    "    uint planned_task_capacity,"
                ),
            ),
            "task expert and route begin swapped": replace_once(
                self.source,
                (
                    "                expert,\n"
                    "                uint(segment_begin + "
                    "ulong(local_begin)),"
                ),
                (
                    "                uint(segment_begin + "
                    "ulong(local_begin)),\n"
                    "                expert,"
                ),
            ),
            "position decode changed to slot mask": replace_once(
                self.source,
                "packed >> packed_slot_bits",
                "packed & ((1u << packed_slot_bits) - 1u)",
            ),
            "output ownership widened": replace_once(
                self.source,
                "row_count,\n                0u,",
                "row_count,\n                1u,",
            ),
            "grid not zeroed": replace_once(
                self.source,
                "indirect_arguments[2] = 0u;",
                "indirect_arguments[2] = 1u;",
            ),
            "initial status ready": replace_once(
                self.source,
                (
                    "status[0] = "
                    "kNativeRoutedQgemmR1TaskStatusNotProduced;"
                ),
                (
                    "status[0] = "
                    "kNativeRoutedQgemmR1TaskStatusReady;"
                ),
            ),
            "grid depth widened": replace_once(
                self.source,
                "indirect_arguments[2] = 1u;",
                "indirect_arguments[2] = 2u;",
            ),
            "barrier removed": replace_once(
                self.source,
                "threadgroup_barrier(mem_flags::mem_device);",
                "",
            ),
            "producer ownership widened": replace_once(
                self.source,
                "all(thread_position == uint3(0u))",
                "any(thread_position == uint3(0u))",
            ),
        }
        for name, mutated in mutations.items():
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    validate_source(mutated)


if __name__ == "__main__":
    unittest.main()
