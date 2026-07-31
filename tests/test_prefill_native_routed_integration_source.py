import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "include/tatara/runtime/prefill_step.h"
RUNTIME = ROOT / "src/runtime/prefill_step.cpp"
PROBE = ROOT / "tools/native/block_prefill_probe.cpp"
METAL = ROOT / "src/backend/metal/kernels/prefill_moe.metal"


def function_body(source: str, marker: str) -> str:
    start = source.index(marker)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace : index + 1]
    raise AssertionError(f"unterminated function: {marker}")


def assert_ordered(
    test: unittest.TestCase, source: str, snippets: tuple[str, ...]
) -> None:
    cursor = 0
    for snippet in snippets:
        found = source.find(snippet, cursor)
        test.assertNotEqual(found, -1, snippet)
        cursor = found + len(snippet)


class PrefillNativeRoutedIntegrationSourceTest(unittest.TestCase):
    def test_policy_is_opt_in_and_layer_major_only(self):
        header = HEADER.read_text()
        runtime = RUNTIME.read_text()
        probe = PROBE.read_text()
        self.assertIn(
            "QuantizedGemmPolicy routed_qgemm"
            "{QuantizedGemmPolicy::ExactRow};",
            header,
        )
        self.assertIn(
            "bool native_routed_shared_expert{false};",
            header,
        )
        self.assertIn("bool native_routed_steel{false};", header)
        self.assertIn("bool native_dense_steel{false};", header)
        self.assertIn(
            "std::uint32_t maximum_units_per_submission{1};",
            header,
        )
        self.assertIn(
            "std::uint32_t maximum_inflight_units{1};",
            header,
        )
        self.assertIn(
            "policy.routed_qgemm == "
            "QuantizedGemmPolicy::NativeRaggedMma",
            runtime,
        )
        self.assertIn(
            "policy.geometry.schedule == PrefillSchedule::LayerMajor",
            runtime,
        )
        self.assertIn('"layer2048fast-n1-r2"', probe)
        self.assertIn('"layer2048fast-profile-stage-r2"', probe)
        self.assertIn('"layer2048fast-n1-r2s-a1"', probe)
        self.assertIn(
            '"layer2048fast-n1-r2s-a1-steel"', probe
        )
        self.assertIn('"layer2048fast-steel-full"', probe)
        self.assertIn(
            '"layer2048fast-steel-full-submit2"', probe
        )
        self.assertIn(
            '"layer2048fast-profile-stage-steel-full"', probe
        )
        self.assertIn(
            '"layer2048fast-profile-stage-n1-r2s-a1"', probe
        )
        self.assertIn(
            "fixture_policy.routed_qgemm ==\n"
            "                QuantizedGemmPolicy::NativeRaggedMma",
            probe,
        )
        self.assertIn(
            "policy.native_routed_steel", runtime
        )

    def test_task_builder_binding_order_matches_kernel_abi(self):
        source = RUNTIME.read_text()
        body = function_body(
            source, "void encode_native_routed_task_builder("
        )
        assert_ordered(
            self,
            body,
            (
                "encode.pipeline("
                "step.pipelines.native_routed_task_builder);",
                "encode.buffer(step.expert_counts, 0, 0);",
                "encode.buffer(step.expert_lists, 0, 1);",
                "encode.buffer(tasks, 0, 2);",
                "encode.buffer(arguments, 0, 3);",
                "encode.buffer(status, status_offset, 4);",
                "encode.constant(block_rows, 5);",
                "encode.constant(task_expert_count, 6);",
                "encode.constant(routes_per_position, 7);",
                "encode.constant(list_stride, 8);",
                "encode.constant(list_stride, 9);",
                "encode.constant(list_extent, 10);",
                "encode.constant(task_capacity, 11);",
                "encode.constant(column_groups, 12);",
                "encode.constant(packed_slot_bits, 13);",
                "encode.constant(include_shared_expert, 14);",
                "MoeRoutedTaskBuild",
            ),
        )

    def test_r2_compute_and_shared_exact_paths_are_both_present(self):
        source = RUNTIME.read_text()
        up = function_body(source, "void encode_native_routed_upgate(")
        down = function_body(source, "void encode_native_routed_down(")
        self.assertIn(
            "step.pipelines.native_routed_steel_upgate", up
        )
        self.assertIn(
            "step.pipelines.native_routed_steel_down", down
        )
        assert_ordered(
            self,
            up,
            (
                "encode.buffer(step.normalized, 0, 0);",
                "encode.buffer(step.expert_lists, 0, 1);",
                "encode.buffer(step.native_routed_up_tasks, 0, 2);",
                "encode.quantized(bindings.expert_gate, 3);",
                "encode.quantized(bindings.expert_up, 6);",
                "encode.buffer(step.expert_hidden, 0, 9);",
                "encode.buffer(step.native_routed_up_arguments, 0, 10);",
                "encode.quantized(bindings.shared_gate, 15);",
                "encode.quantized(bindings.shared_up, 18);",
                "encode.constant(include_shared_expert, 21);",
                "PrefillProfileEventClass::MoeNativeRoutedUpGate",
            ),
        )
        assert_ordered(
            self,
            down,
            (
                "encode.buffer(step.expert_hidden, 0, 0);",
                "encode.buffer(step.expert_lists, 0, 1);",
                (
                    "step.policy.command_graph ? "
                    "step.native_routed_up_tasks"
                ),
                ": step.native_routed_down_tasks,",
                "encode.quantized(bindings.expert_down, 3);",
                "encode.buffer(step.expert_partials, 0, 6);",
                "? step.native_routed_up_arguments",
                ": step.native_routed_down_arguments,",
                "encode.quantized(bindings.shared_down, 12);",
                "encode.constant(include_shared_expert, 15);",
                "PrefillProfileEventClass::MoeNativeRoutedDown",
            ),
        )
        moe = function_body(source, "void encode_moe(")
        self.assertIn(
            "encode.buffer(step.shared_expert, 0, 10);", moe
        )
        self.assertIn(
            "encode.buffer(step.shared_expert, 0, 7);", moe
        )
        self.assertIn(
            "PrefillProfileEventClass::MoeSharedExpertUpGate",
            moe,
        )
        self.assertIn(
            "PrefillProfileEventClass::MoeSharedExpertDown",
            moe,
        )
        self.assertIn(
            "if (!step.policy.native_routed_shared_expert)", moe
        )
        self.assertIn(
            "PrefillProfileEventClass::"
            "MoeNativeRoutedSharedUpGate",
            up,
        )
        self.assertIn(
            "PrefillProfileEventClass::"
            "MoeNativeRoutedSharedDown",
            down,
        )
        self.assertIn("} else {\n        encode.pipeline(", moe)

    def test_profile_labels_keep_routed_and_shared_work_distinct(self):
        probe = PROBE.read_text()
        assert_ordered(
            self,
            probe,
            (
                '"moe_native_routed_upgate"',
                '"moe_shared_expert_upgate"',
                '"moe_native_routed_down"',
                '"moe_shared_expert_down"',
            ),
        )
        self.assertIn(
            '"tatara_prefill_profile_moe_pair"', probe
        )
        self.assertIn('" routed_exclusive_ticks="', probe)
        self.assertIn('" shared_exclusive_ticks="', probe)
        self.assertIn('" union_ticks="', probe)

    def test_r2_stages_bfloat_activations_and_float_weights(self):
        source = METAL.read_text()
        stage_weights = function_body(
            source, "inline void native_routed_qgemm_r2_stage_weights("
        )
        load_weight = function_body(
            source, "inline void native_routed_qgemm_r2_load_weight("
        )
        self.assertIn("threadgroup float* staged_weights", source)
        self.assertNotIn("static_cast<bfloat>", stage_weights)
        self.assertIn("threadgroup const float* staged_weights", source)
        self.assertIn("staged_weights[index]", load_weight)
        self.assertIn("threadgroup bfloat staged_input[", source)
        self.assertIn("threadgroup float staged_gate[", source)
        self.assertIn("threadgroup float staged_up[", source)

    def test_device_status_fails_before_progress_mutation(self):
        source = RUNTIME.read_text()
        direct = function_body(source, "PrefillEncodeResult encode_prefill_impl(")
        self.assertIn(
            "PrefillEncodeError::DeviceTaskValidationUnavailable",
            direct,
        )
        commit = function_body(
            source,
            "PrefillProgressResult commit_prefill_pending(",
        )
        assert_ordered(
            self,
            commit,
            (
                "unit < prefill.progress.pending_unit_count",
                "device_task_status(",
                "prefill.native_routed_up_status, unit",
                "prefill.native_routed_down_status, unit",
                "PrefillProgressError::DeviceTaskNotReady",
                "const std::uint32_t committed_layer",
            ),
        )
        self.assertIn("result.failed_unit_offset = unit;", commit)

    def test_multi_unit_submission_is_bounded_and_opt_in(self):
        header = HEADER.read_text()
        runtime = RUNTIME.read_text()
        probe = PROBE.read_text()
        self.assertIn(
            "kPrefillMaximumUnitsPerSubmission = 64", header
        )
        self.assertIn(
            "policy.maximum_units_per_submission <=\n"
            "            kPrefillMaximumUnitsPerSubmission",
            runtime,
        )
        self.assertIn(
            "policy.maximum_units_per_submission == 1 ||\n"
            "         policy.geometry.schedule == "
            "PrefillSchedule::LayerMajor",
            runtime,
        )
        encode = function_body(
            runtime, "PrefillProgressResult encode_prefill_units_impl("
        )
        assert_ordered(
            self,
            encode,
            (
                "total_units - first_unit",
                "unit < encoded_units",
                "output_offset, unit",
                "prefill.progress.state = pending_state",
                "prefill.progress.pending_unit_count = encoded_units",
            ),
        )
        self.assertIn(
            "prefill.policy.maximum_units_per_submission > 1",
            probe,
        )
        self.assertIn("encode_prefill_units(", probe)
        self.assertIn("commit_prefill_units(", probe)

    def test_inflight_window_defers_all_state_publication(self):
        runtime = RUNTIME.read_text()
        probe = PROBE.read_text()
        encode = function_body(
            runtime,
            "PrefillProgressResult encode_prefill_inflight_unit(",
        )
        assert_ordered(
            self,
            encode,
            (
                "first_unit + progress.pending_unit_count",
                "status_slot =\n"
                "        progress.pending_unit_count",
                "output_offset,\n        status_slot",
                "++prefill.progress.pending_unit_count",
                "PrefillProgressState::InflightPending",
            ),
        )
        submit = function_body(
            probe, "SubmissionResult submit_prefill_impl("
        )
        assert_ordered(
            self,
            submit,
            (
                "encode_prefill_inflight_unit(",
                "commit(",
                "pending_executions[pending_count]",
                "wait_until_completed_timed(",
                "commit_prefill_inflight(",
            ),
        )
        self.assertIn(
            "pending_count !=\n"
            "                    prefill.progress.pending_unit_count",
            submit,
        )
        self.assertIn(
            "maximum_inflight_units > 1", submit
        )


if __name__ == "__main__":
    unittest.main()
