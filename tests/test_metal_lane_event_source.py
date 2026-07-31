import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RESOURCES_HEADER = (
    ROOT / "include/tatara/backend/metal/resources.h"
).read_text(encoding="utf-8")
RESOURCES_SOURCE = (
    ROOT / "src/backend/metal/resources.mm"
).read_text(encoding="utf-8")
COMMANDS_HEADER = (
    ROOT / "include/tatara/backend/metal/commands.h"
).read_text(encoding="utf-8")
COMMANDS_SOURCE = (
    ROOT / "src/backend/metal/commands.mm"
).read_text(encoding="utf-8")
PLAN_HEADER = (
    ROOT / "include/tatara/runtime/prefill_command_plan.h"
).read_text(encoding="utf-8")
PLAN_SOURCE = (
    ROOT / "src/runtime/prefill_command_plan.cpp"
).read_text(encoding="utf-8")
STEP_HEADER = (
    ROOT / "include/tatara/runtime/prefill_step.h"
).read_text(encoding="utf-8")
STEP_SOURCE = (
    ROOT / "src/runtime/prefill_step.cpp"
).read_text(encoding="utf-8")
PROBE_SOURCE = (
    ROOT / "tools/native/block_prefill_probe.cpp"
).read_text(encoding="utf-8")


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


class MetalLaneEventSourceTest(unittest.TestCase):
    def test_event_is_nonshareable_and_move_only(self) -> None:
        header = compact(RESOURCES_HEADER)
        source = compact(RESOURCES_SOURCE)
        for marker in (
            "classMetalEvent{",
            "MetalEvent(constMetalEvent&)=delete;",
            "MetalEvent&operator=(constMetalEvent&)=delete;",
            "MetalEvent(MetalEvent&&)noexcept;",
            "MetalEventResultcreate_event(constMetalDevice&device);",
            "id<MTLEvent>event=[native_devicenewEvent];",
        ):
            self.assertIn(marker, header + source)
        self.assertNotIn("newSharedEvent", source)
        self.assertNotIn("MTLSharedEvent", source)

    def test_wait_and_signal_refuse_invalid_objects_and_zero(self) -> None:
        header = compact(COMMANDS_HEADER)
        source = compact(COMMANDS_SOURCE)
        for marker in (
            "encode_wait_for_event(MetalCommandBuffer&",
            "encode_signal_event(MetalCommandBuffer&",
            "if(!command_buffer){",
            "returnMetalCommandError::InvalidCommandBuffer;",
            "if(!event){",
            "returnMetalCommandError::InvalidEvent;",
            "if(value==0){",
            "returnMetalCommandError::InvalidEventValue;",
            "encodeWaitForEvent:native_eventvalue:value",
            "encodeSignalEvent:native_eventvalue:value",
        ):
            self.assertIn(marker, header + source)

    def test_cpu_plan_covers_both_predecessor_domains(self) -> None:
        header = compact(PLAN_HEADER)
        source = compact(PLAN_SOURCE)
        for marker in (
            "structPrefillLaneEventNode",
            "build_prefill_lane_event_plan(",
            "node.hidden_predecessor!=expected_hidden",
            "node.state_predecessor!=expected_state",
            "wait_event=lane==0?kNoPrefillLaneEvent:lane-1u",
            "signal_event=lane+1u==request.scratch_lane_count"
            "?kNoPrefillLaneEvent:lane",
            "InvalidEventValue",
            "LaneEventStorageTooSmall",
        ):
            self.assertIn(marker, header + source)

    def test_node_major_graph_retains_local_barriers(self) -> None:
        header = compact(STEP_HEADER)
        source = compact(STEP_SOURCE)
        for marker in (
            "boolcommand_graph_lane_events{false};",
            "node_level_boundary_offsets",
            "node_level_command_begins",
            "std::vector<PrefillLaneEventNode>lane_event_nodes;",
            "build_prefill_lane_event_plan(",
            "recorded_nodes[event_node.node_index]",
            "encode_prefill_command_graph_lane_node(",
            "if(boundary!=boundary_begin){",
            "command_error=memory_barrier(pass);",
            "mark_prefill_command_graph_lane_pending(",
            "PrefillProgressState::GraphPending",
        ):
            self.assertIn(marker, header + source)

    def test_probe_uses_three_queues_and_events_between_passes(self) -> None:
        source = compact(PROBE_SOURCE)
        for marker in (
            "layer2048fast-steel-full-graph-lane-events-compile",
            "layer2048fast-steel-full-graph-lane-events-warm",
            "run_lane_event_graph(",
            "create_command_buffer("
            "prefill.command_graph_lane_queues[lane])",
            "encode_wait_for_event(",
            "begin_compute_pass(std::move(command_buffer))",
            "end_compute_pass(std::move(*pass.compute_pass))",
            "encode_signal_event(",
            "mark_prefill_command_graph_lane_pending(",
            "wait_until_completed_timed(std::move(execution))",
        ):
            self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
