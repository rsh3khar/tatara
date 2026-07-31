import pathlib
import re
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PLAN_HEADER = ROOT / "include/tatara/runtime/prefill_command_plan.h"
PLAN_SOURCE = ROOT / "src/runtime/prefill_command_plan.cpp"
COMMAND_HEADER = ROOT / "include/tatara/backend/metal/commands.h"
COMMAND_SOURCE = ROOT / "src/backend/metal/commands.mm"
PIPELINE_HEADER = ROOT / "include/tatara/backend/metal/pipeline.h"
PIPELINE_SOURCE = ROOT / "src/backend/metal/pipeline.mm"
MOE_SOURCE = ROOT / "src/backend/metal/kernels/prefill_moe.metal"
PREFILL_HEADER = ROOT / "include/tatara/runtime/prefill_step.h"
PREFILL_SOURCE = ROOT / "src/runtime/prefill_step.cpp"
BLOCK_PROBE = ROOT / "tools/native/block_prefill_probe.cpp"


def function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function {signature}")


def validate_wrapper_sources(command_source: str, pipeline_source: str) -> None:
    create_icb = function_body(
        command_source,
        "MetalIndirectCommandBufferResult create_compute_indirect_command_buffer(",
    )
    required_icb_creation = (
        "MTLIndirectCommandTypeConcurrentDispatch",
        "descriptor.inheritPipelineState = NO;",
        "descriptor.inheritBuffers = NO;",
        "descriptor.maxKernelBufferBindCount",
        "newIndirectCommandBufferWithDescriptor:descriptor",
        "IndirectCommandBufferCreationFailed",
    )
    for token in required_icb_creation:
        if token not in create_icb:
            raise AssertionError(f"missing ICB creation contract: {token}")

    set_pipeline = function_body(
        command_source, "MetalCommandError set_indirect_compute_pipeline("
    )
    for token in (
        "supports_indirect_commands(pipeline)",
        "setComputePipelineState:native_pipeline",
        "InvalidIndirectCommandIndex",
    ):
        if token not in set_pipeline:
            raise AssertionError(f"missing indirect pipeline gate: {token}")

    set_buffer = function_body(
        command_source, "MetalCommandError set_indirect_buffer("
    )
    for token in (
        "maximum_kernel_buffer_bind_count_",
        "offset_bytes >= buffer.size_bytes()",
        "setKernelBuffer:native_buffer",
    ):
        if token not in set_buffer:
            raise AssertionError(f"missing indirect buffer bound: {token}")

    dispatch = function_body(
        command_source, "MetalCommandError dispatch_indirect_threadgroups("
    )
    for token in (
        "zero_extent",
        "exceeds_native_extent(threadgroups)",
        "concurrentDispatchThreadgroups:native_size(threadgroups)",
    ):
        if token not in dispatch:
            raise AssertionError(f"missing bounded direct dispatch: {token}")
    if "dispatchThreadgroupsWithIndirectBuffer" in dispatch:
        raise AssertionError("classic ICB command cannot use an indirect grid")

    use_resource = function_body(
        command_source, "MetalCommandError use_buffer_resource("
    )
    for token in (
        "MTLResourceUsageRead",
        "MTLResourceUsageWrite",
        "InvalidResourceUsage",
        "useResource:native_buffer usage:native_usage",
    ):
        if token not in use_resource:
            raise AssertionError(f"missing resource-residency truth: {token}")

    execute = function_body(
        command_source, "MetalCommandError execute_indirect_commands("
    )
    for token in (
        "valid_indirect_range(",
        "executeCommandsInBuffer:native_buffer",
        "withRange:",
    ):
        if token not in execute:
            raise AssertionError(f"missing bounded ICB execution: {token}")

    indirect_pipeline = function_body(
        pipeline_source,
        "MetalComputePipelineResult create_indirect_compute_pipeline(",
    )
    for token in (
        "MTLComputePipelineDescriptor",
        "descriptor.computeFunction = native_function;",
        "descriptor.supportIndirectCommandBuffers = YES;",
        "newComputePipelineStateWithDescriptor:descriptor",
        "!pipeline.supportIndirectCommandBuffers",
    ):
        if token not in indirect_pipeline:
            raise AssertionError(f"missing ICB-capable pipeline contract: {token}")


def validate_fused_task_producer(source: str) -> None:
    body = function_body(source, "kernel void expert_union_fused_tasks(")

    buffer_contract = (
        "device const uint* ids [[buffer(0)]]",
        "constant uint& block [[buffer(1)]]",
        "device uint* counts [[buffer(2)]]",
        "device uint* lists [[buffer(3)]]",
        "device NativeRoutedQgemmR1Task* tasks [[buffer(4)]]",
        "device uint* task_count [[buffer(5)]]",
        "device uint* status [[buffer(6)]]",
        "constant uint& list_capacity [[buffer(7)]]",
        "constant uint& include_shared_expert [[buffer(8)]]",
        "constant ulong& route_list_total_extent [[buffer(9)]]",
        "constant uint& planned_task_capacity [[buffer(10)]]",
        "constant uint& packed_slot_bits [[buffer(11)]]",
        "uint3 thread_position [[thread_position_in_threadgroup]]",
        "uint3 group [[threadgroup_position_in_grid]]",
        "uint3 threadgroup_shape [[threads_per_threadgroup]]",
        "const uint expert = thread_position.x;",
    )
    for token in buffer_contract:
        if token not in body:
            raise AssertionError(f"missing fused producer binding: {token}")

    required_safety = (
        "!all(group == uint3(0u))",
        "threadgroup_shape.x != kMoeExperts",
        "threadgroup_shape.y != 1u",
        "threadgroup_shape.z != 1u",
        "status[0] =",
        "task_count[0] = 0u;",
        "counts[kMoeExperts] = 0u;",
        "block == 0u",
        "list_capacity == 0u",
        "block > list_capacity",
        "list_capacity > maximum_position_count",
        "include_shared_expert > 1u",
        "planned_task_capacity == 0u",
        "planned_task_capacity >",
        "packed_slot_bits != kPrefillPackedSlotBits",
        "route_list_total_extent != required_list_extent",
        "route_list_total_extent > 0x100000000ul",
        "selected >= kMoeExperts",
        "matched || count >= list_capacity",
        "routed_count != expected_routed_count",
        "required_task_count >",
        "ulong(planned_task_capacity)",
        "next_task >=",
        "planned_task_capacity",
        "task_count[0] = next_task;",
        "status[0] = final_status;",
    )
    for token in required_safety:
        if token not in body:
            raise AssertionError(f"missing fused producer safety gate: {token}")

    shape_gate = body.index("if (!all(group == uint3(0u))")
    early_return = body.index("return;", shape_gate)
    first_threadgroup_state = body.index("threadgroup uint errors")
    if not shape_gate < early_return < first_threadgroup_state:
        raise AssertionError("dispatch-shape rejection must precede threadgroup state")

    first_barrier = body.index("threadgroup_barrier(")
    for token in (
        "status[0] =",
        "task_count[0] = 0u;",
        "counts[kMoeExperts] = 0u;",
        "errors[expert] = local_error;",
    ):
        if body.index(token) >= first_barrier:
            raise AssertionError(f"fail-closed initialization follows barrier: {token}")

    list_guard = body.index("if (matched || count >= list_capacity)")
    list_write = body.index("lists[expert * list_capacity + count]")
    if list_guard >= list_write:
        raise AssertionError("route-list write is not dominated by its capacity gate")

    required_task_gate = body.index("required_task_count >")
    first_task_write = body.index("tasks[next_task++]")
    direct_task_gate = body.index("if (next_task >=")
    if not required_task_gate < direct_task_gate < first_task_write:
        raise AssertionError("task writes are not dominated by both capacity gates")

    for deterministic_loop in (
        "for (uint task_expert = 0u;",
        "++task_expert)",
        "for (uint local_begin = 0u;",
        "local_begin +=\n                             kNativeRoutedQgemmR1TileRows)",
    ):
        if deterministic_loop not in body:
            raise AssertionError(
                f"missing deterministic task order: {deterministic_loop}"
            )

    if body.count("threadgroup uint") != 2:
        raise AssertionError("fused producer threadgroup storage changed")
    if "threadgroup uint errors[kMoeExperts];" not in body:
        raise AssertionError("per-expert error storage is not exactly bounded")
    if "threadgroup uint final_status;" not in body:
        raise AssertionError("final status is not in bounded threadgroup storage")
    if body.count("threadgroup_barrier(") != 3:
        raise AssertionError("fused producer barrier topology changed")

    ready = body.index("kNativeRoutedQgemmR1TaskStatusReady")
    count_publish = body.index("task_count[0] = next_task;")
    final_barrier = body.rindex("threadgroup_barrier(")
    status_publish = body.rindex("status[0] = final_status;")
    if not count_publish < ready < final_barrier < status_publish:
        raise AssertionError("Ready publication order is not fail-closed")
    if body.count("kNativeRoutedQgemmR1TaskStatusReady") != 1:
        raise AssertionError("Ready must have exactly one publication path")


def validate_host_command_graph(header: str, source: str) -> None:
    for token in (
        "bool command_graph{false};",
        "std::uint32_t command_graph_chunk_count{1};",
        "PrefillCommandGraphState",
        "MetalIndirectCommandBuffer commands;",
        "MetalBuffer argument_arena;",
        "std::vector<std::uint32_t> level_command_begins;",
        "PrefillCommandIdentity model_package_identity{};",
        "PrefillCommandIdentity execution_policy_identity{};",
        "std::uint64_t icb_capability_identity{0};",
        "prepare_prefill_command_graph(",
        "encode_prefill_command_graph(",
        "commit_prefill_command_graph(",
    ):
        if token not in header:
            raise AssertionError(f"missing opt-in graph contract: {token}")

    policy = function_body(source, "bool valid_policy(")
    for token in (
        "policy.command_graph",
        "PrefillSchedule::LayerMajor",
        "QuantizedGemmPolicy::NativeDenseMma",
        "policy.native_dense_steel",
        "QuantizedGemmPolicy::NativeRaggedMma",
        "policy.native_routed_shared_expert",
        "policy.native_routed_steel",
        "policy.command_graph_chunk_count != 0",
        "kPrefillMaximumUnitsPerSubmission",
        "policy.command_graph_chunk_count == 1",
    ):
        if token not in policy:
            raise AssertionError(f"missing graph admission gate: {token}")

    recorder = function_body(source, "struct RecordingEncoder")
    for token in (
        "static constexpr bool profiled = false;",
        "scratch_lane_stride(step, buffer_value)",
        "multiply(stride, lane, lane_base)",
        "current_bindings[index]",
        "commands.push_back(",
        "maximum_routed_task_count(step, chunk.rows)",
        "kKernelLibraryNativeRoutedQgemmR1TileColumns",
        ".width = task_count",
        "if (!failed() && level_has_command)",
        "++local_level;",
        "offsets[tensor] >> kPrefillImageWindowShift",
        "&image_windows[window]",
        "kIndirectKernelBufferOffsetLimitBytes",
        "plan_prefill_buffer_window(",
        "candidate.source == &buffer_value",
        ".buffer = &window->window",
    ):
        if token not in recorder:
            raise AssertionError(f"missing recording invariant: {token}")
    if "dispatch_threadgroups_indirect(" in recorder:
        raise AssertionError("recording path retained a GPU-indirect grid")

    moe = function_body(source, "void encode_moe(")
    for token in (
        "if (step.policy.command_graph)",
        "step.pipelines.expert_union_fused_tasks",
        "step.native_routed_up_tasks, 0, 4",
        "step.native_routed_up_arguments, 0, 5",
        "status_slot} * kNativeRoutedStatusBytes",
        "maximum_routed_task_count(step, block_rows)",
        "if (!step.policy.command_graph)",
        "encode_native_routed_task_builder(",
    ):
        if token not in moe:
            raise AssertionError(f"missing fused host treatment: {token}")

    prepare = function_body(
        source, "PrefillCommandGraphResult prepare_prefill_command_graph("
    )
    for token in (
        "graph_pipelines_are_indirect_capable(\n"
        "            prefill.pipelines, prefill.policy)",
        "build_prefill_wavefront_plan(",
        "RecordingEncoder encode(",
        "node.scratch_lane",
        "node.layer_major_index",
        "for (const PrefillCommandDiagonal& diagonal",
        "command.local_level != level",
        "level_command_begins.back() ==",
        "graph.level_command_begins =",
        "profile_plan.required_event_count -",
        "PrefillProfilePlanError::\n                    EventCapacityInsufficient",
        "removed_builders !=",
        "validate_prefill_command_plan_key(",
        "same_prefill_command_plan_key(",
        "metal_device_identity(device)",
        "create_shared_buffer(device, arena_cursor)",
        "create_compute_indirect_command_buffer(",
        "reset_indirect_commands(",
        "set_indirect_compute_pipeline(",
        "set_indirect_buffer(",
        "clear_indirect_barrier(",
        "dispatch_indirect_threadgroups(",
        "PrefillCommandGraphState::Ready",
        "create_buffer_window(",
        "graph.image_windows",
        "visit_prefill_scratch_lane_buffers(",
        "graph.scratch_windows",
    ):
        if token not in prepare:
            raise AssertionError(f"missing graph build invariant: {token}")
    if "set_indirect_barrier(" in prepare:
        raise AssertionError(
            "graph build reintroduced per-command barrier ordering; Apple "
            "documents setBarrier as ordering only its own command"
        )
    if len(
        re.findall(
            r"(?<!node_)level_command_begins\.push_back\(",
            prepare,
        )
    ) != 2:
        raise AssertionError(
            "graph build must push one level begin per emitted level plus "
            "one terminating end offset"
        )
    if prepare.count("create_buffer_window(") != 2:
        raise AssertionError(
            "graph build must create both model-image and large scratch "
            "placement-heap windows"
        )
    for prohibited in (
        "create_command_buffer(",
        "begin_compute_pass(",
        "execute_indirect_commands(",
        "commit(",
        "wait_until_completed(",
    ):
        if prohibited in prepare:
            raise AssertionError(f"graph build admitted submission: {prohibited}")

    execute = function_body(
        source, "PrefillCommandGraphResult encode_prefill_command_graph("
    )
    for token in (
        "pipeline_identities[pipeline_index]",
        "resource_identities[resource_index]",
        "execution_policy_identity(prefill.policy)",
        "validate_prefill_command_plan_key(",
        "use_buffer_resource(pass, buffer, usage)",
        "level_command_begins.size() < 2u",
        "level_command_begins.front() != 0u",
        "level_command_begins.back() !=",
        "level_end <= level_begin",
        "memory_barrier(pass)",
        "execute_indirect_commands(",
        "level_end - level_begin",
        "PrefillCommandGraphState::Pending",
        "PrefillProgressState::GraphPending",
    ):
        if token not in execute:
            raise AssertionError(f"missing graph execution gate: {token}")

    resources = function_body(source, "void visit_prefill_graph_buffers(")
    for token in (
        "decode.schedule[layer_index]",
        "model::qwen36::LayerKind::GatedDelta",
        "visit(layer.first_out",
        "visit(layer.second_out",
    ):
        if token not in resources:
            raise AssertionError(
                f"missing topology-bound state residency: {token}"
            )

    commit = function_body(
        source, "PrefillProgressResult commit_prefill_command_graph("
    )
    required_order = (
        "PrefillCommandGraphState::Pending",
        "PrefillProgressState::GraphPending",
        "device_task_status(",
        "QuantizedGemmDeviceTaskStatus::Ready",
        "advance_prefill_state(",
        "PrefillProgressState::Complete",
        "PrefillCommandGraphState::Ready",
    )
    cursor = 0
    for token in required_order:
        position = commit.find(token, cursor)
        if position < 0:
            raise AssertionError(f"missing graph publication order: {token}")
        cursor = position + len(token)

    for model_literal in ("2633", "2873", "3925"):
        if model_literal in source:
            raise AssertionError(f"host graph hardcodes profile fact: {model_literal}")


def validate_zero_submission_probe(source: str) -> None:
    parser = function_body(
        source, "bool parse_fixture_policy("
    )
    policy_start = parser.index(
        '"layer2048fast-steel-full-graph-compile"'
    )
    policy_end = parser.index("} else if (", policy_start)
    policy = parser[policy_start:policy_end]
    for token in (
        "policy = kLayer2048Fast;",
        "policy.name = text;",
        "policy.native_dense_steel = true;",
        "policy.native_routed_shared_expert = true;",
        "policy.native_routed_steel = true;",
        "policy.command_graph = true;",
        "policy.command_graph_compile_only = true;",
    ):
        if token not in policy:
            raise AssertionError(f"missing compile-only graph policy: {token}")

    branch_start = source.index(
        "if (fixture_policy.command_graph_compile_only)"
    )
    branch_end = source.index("SubmissionResult block;", branch_start)
    branch = source[branch_start:branch_end]
    for token in (
        "begin_prefill_progress(",
        "prepare_prefill_command_graph(",
        "!prepared || prepared.cache_hit",
        "!cached || !cached.cache_hit",
        '"  command buffers submitted: 0\\n"',
        "return 0;",
    ):
        if token not in branch:
            raise AssertionError(f"missing zero-submission graph gate: {token}")
    for prohibited in (
        "create_command_buffer(",
        "begin_compute_pass(",
        "encode_prefill_command_graph(",
        "commit(",
        "wait_until_completed(",
    ):
        if prohibited in branch:
            raise AssertionError(
                f"compile-only graph branch admitted submission: {prohibited}"
            )

    submit = function_body(source, "SubmissionResult submit_prefill_impl(")
    required_lifecycle = (
        "if (prefill.policy.command_graph)",
        "prepare_prefill_command_graph(",
        "create_command_buffer(*harness.queue)",
        "begin_compute_pass(",
        "encode_prefill_command_graph(",
        "end_compute_pass(",
        "commit(std::move(*ended.command_buffer))",
        "wait_until_completed_timed(",
        "commit_prefill_command_graph(",
        "return result;",
    )
    cursor = 0
    for token in required_lifecycle:
        position = submit.find(token, cursor)
        if position < 0:
            raise AssertionError(
                f"missing command-graph lifecycle order: {token}"
            )
        cursor = position + len(token)


class PrefillCommandPlanSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan_header = PLAN_HEADER.read_text()
        cls.plan_source = PLAN_SOURCE.read_text()
        cls.command_header = COMMAND_HEADER.read_text()
        cls.command_source = COMMAND_SOURCE.read_text()
        cls.pipeline_header = PIPELINE_HEADER.read_text()
        cls.pipeline_source = PIPELINE_SOURCE.read_text()
        cls.moe_source = MOE_SOURCE.read_text()
        cls.prefill_header = PREFILL_HEADER.read_text()
        cls.prefill_source = PREFILL_SOURCE.read_text()
        cls.block_probe = BLOCK_PROBE.read_text()

    def test_cpu_plan_is_allocation_free_and_general(self):
        combined = self.plan_header + self.plan_source
        for prohibited in ("std::vector", "malloc(", "calloc(", "realloc(", "new "):
            self.assertNotIn(prohibited, combined)
        for required in (
            "std::span<const std::uint32_t> chunk_rows",
            "std::span<const std::uint64_t> persistent_resource_identities",
            "request.layer_count",
            "request.chunk_rows.size()",
            "request.scratch_lane_count != request.chunk_rows.size()",
            "hidden_predecessor",
            "state_predecessor",
            "layer + chunk",
        ):
            self.assertIn(required, combined)
        for literal in ("40", "256", "2048", "1621", "3925"):
            self.assertNotIn(literal, self.plan_source)

    def test_key_compares_every_identity_and_schedule(self):
        body = function_body(
            self.plan_source, "bool same_prefill_command_plan_key("
        )
        for field in (
            "model_package_identity",
            "prepared_image_identity",
            "pipeline_identity",
            "execution_policy_identity",
            "icb_capability_identity",
            "state_slot_identity",
            "row_count",
            "context_base",
            "graph_schema_version",
            "chunk_rows",
            "persistent_resource_identities",
        ):
            self.assertIn(field, body)

    def test_arena_and_resource_tables_fail_closed(self):
        arena = function_body(
            self.plan_source,
            "PrefillArgumentArenaPlanResult build_prefill_argument_arena_plan(",
        )
        for token in (
            "InvalidCommandIndex",
            "InvalidBufferIndex",
            "InvalidArgumentSize",
            "InvalidArgumentAlignment",
            "DuplicateArgumentBinding",
            "ArgumentStorageTooSmall",
            "ArgumentArenaTooSmall",
        ):
            self.assertIn(token, arena)
        resources = function_body(
            self.plan_source,
            "PrefillResourceTableResult build_prefill_resource_table(",
        )
        for token in (
            "InvalidResourceIdentity",
            "InvalidResourceUsage",
            "merge_usage",
            "ResourceStorageTooSmall",
        ):
            self.assertIn(token, resources)

    def test_icb_wrappers_match_local_sdk_capabilities(self):
        validate_wrapper_sources(self.command_source, self.pipeline_source)
        for declaration in (
            "class MetalIndirectCommandBuffer",
            "create_compute_indirect_command_buffer(",
            "set_indirect_compute_pipeline(",
            "set_indirect_buffer(",
            "set_indirect_barrier(",
            "dispatch_indirect_threadgroups(",
            "use_buffer_resource(",
            "execute_indirect_commands(",
        ):
            self.assertIn(declaration, self.command_header)
        self.assertIn(
            "create_indirect_compute_pipeline(", self.pipeline_header
        )

    def test_named_wrapper_mutations_are_rejected(self):
        mutations = (
            "descriptor.inheritPipelineState = NO;",
            "descriptor.inheritBuffers = NO;",
            "supports_indirect_commands(pipeline)",
            "offset_bytes >= buffer.size_bytes()",
            "concurrentDispatchThreadgroups:native_size(threadgroups)",
            "useResource:native_buffer usage:native_usage",
            "executeCommandsInBuffer:native_buffer",
        )
        for token in mutations:
            with self.subTest(token=token):
                mutated = self.command_source.replace(token, "")
                with self.assertRaises(AssertionError):
                    validate_wrapper_sources(mutated, self.pipeline_source)
        pipeline_mutation = self.pipeline_source.replace(
            "descriptor.supportIndirectCommandBuffers = YES;", "", 1
        )
        with self.assertRaises(AssertionError):
            validate_wrapper_sources(self.command_source, pipeline_mutation)

    def test_fused_task_producer_is_separate_and_fail_closed(self):
        validate_fused_task_producer(self.moe_source)
        self.assertIn("kernel void expert_union(", self.moe_source)
        self.assertEqual(
            self.moe_source.count("kernel void expert_union_fused_tasks("), 1
        )

    def test_named_fused_task_producer_mutations_are_rejected(self):
        mutations = (
            "!all(group == uint3(0u))",
            "threadgroup_shape.x != kMoeExperts",
            "task_count[0] = 0u;",
            "counts[kMoeExperts] = 0u;",
            "block > list_capacity",
            "list_capacity > maximum_position_count",
            "include_shared_expert > 1u",
            "planned_task_capacity == 0u",
            "packed_slot_bits != kPrefillPackedSlotBits",
            "route_list_total_extent != required_list_extent",
            "route_list_total_extent > 0x100000000ul",
            "selected >= kMoeExperts",
            "matched || count >= list_capacity",
            "routed_count != expected_routed_count",
            "required_task_count >",
            "if (next_task >=",
            "task_count[0] = next_task;",
            "status[0] = final_status;",
        )
        original_body = function_body(
            self.moe_source, "kernel void expert_union_fused_tasks("
        )
        for token in mutations:
            with self.subTest(token=token):
                mutated_body = original_body.replace(token, "", 1)
                self.assertNotEqual(mutated_body, original_body)
                mutated = self.moe_source.replace(
                    original_body, mutated_body, 1
                )
                with self.assertRaises(AssertionError):
                    validate_fused_task_producer(mutated)

    def test_host_command_graph_is_opt_in_and_plan_derived(self):
        validate_host_command_graph(
            self.prefill_header, self.prefill_source
        )

    def test_named_host_command_graph_mutations_are_rejected(self):
        mutations = (
            "bool command_graph{false};",
            "std::uint32_t command_graph_chunk_count{1};",
        )
        for token in mutations:
            with self.subTest(token=token):
                mutated = self.prefill_header.replace(token, "", 1)
                with self.assertRaises(AssertionError):
                    validate_host_command_graph(
                        mutated, self.prefill_source
                    )

        prepare_body = function_body(
            self.prefill_source,
            "PrefillCommandGraphResult prepare_prefill_command_graph(",
        )
        for token in (
            "graph_pipelines_are_indirect_capable(\n"
            "            prefill.pipelines, prefill.policy)",
            "build_prefill_wavefront_plan(",
            "node.scratch_lane",
            "node.layer_major_index",
            "                level_command_begins.push_back(",
            "graph.level_command_begins =",
            "create_buffer_window(",
            "profile_plan.required_event_count -",
            "PrefillProfilePlanError::\n                    EventCapacityInsufficient",
            "same_prefill_command_plan_key(",
            "create_compute_indirect_command_buffer(",
            "clear_indirect_barrier(",
        ):
            with self.subTest(token=token):
                mutated_body = prepare_body.replace(token, "", 1)
                self.assertNotEqual(mutated_body, prepare_body)
                mutated = self.prefill_source.replace(
                    prepare_body, mutated_body, 1
                )
                with self.assertRaises(AssertionError):
                    validate_host_command_graph(
                        self.prefill_header, mutated
                    )

        execute_body = function_body(
            self.prefill_source,
            "PrefillCommandGraphResult encode_prefill_command_graph(",
        )
        for token in (
            "level_command_begins.size() < 2u",
            "level_command_begins.front() != 0u",
            "level_end <= level_begin",
            "memory_barrier(pass)",
            "level_end - level_begin",
        ):
            with self.subTest(token=token):
                mutated_body = execute_body.replace(token, "", 1)
                self.assertNotEqual(mutated_body, execute_body)
                mutated = self.prefill_source.replace(
                    execute_body, mutated_body, 1
                )
                with self.assertRaises(AssertionError):
                    validate_host_command_graph(
                        self.prefill_header, mutated
                    )

        rebarriered_body = prepare_body.replace(
            "clear_indirect_barrier(", "set_indirect_barrier(", 1
        )
        self.assertNotEqual(rebarriered_body, prepare_body)
        rebarriered = self.prefill_source.replace(
            prepare_body, rebarriered_body, 1
        )
        with self.assertRaises(AssertionError):
            validate_host_command_graph(
                self.prefill_header, rebarriered
            )

    def test_real_model_probe_builds_and_caches_without_submission(self):
        validate_zero_submission_probe(self.block_probe)

    def test_named_zero_submission_probe_mutations_are_rejected(self):
        for token in (
            "policy.command_graph = true;",
            "policy.command_graph_compile_only = true;",
            "!prepared || prepared.cache_hit",
            "!cached || !cached.cache_hit",
            '"  command buffers submitted: 0\\n"',
        ):
            with self.subTest(token=token):
                mutated = self.block_probe.replace(token, "", 1)
                with self.assertRaises(AssertionError):
                    validate_zero_submission_probe(mutated)


if __name__ == "__main__":
    unittest.main()
