#include "tatara/runtime/prefill_command_plan.h"

#include <array>
#include <cstdint>
#include <limits>

namespace {

using namespace tatara::runtime;

PrefillCommandIdentity identity(std::uint8_t seed) {
    PrefillCommandIdentity value{};
    for (std::size_t index = 0; index < value.size(); ++index) {
        value[index] = static_cast<std::uint8_t>(seed + index);
    }
    return value;
}

bool key_contract() {
    const std::array<std::uint32_t, 3> chunks{256, 2048, 1621};
    const std::array<std::uint64_t, 4> resources{11, 22, 33, 44};
    const PrefillCommandPlanKey key{
        .model_package_identity = identity(1),
        .prepared_image_identity = identity(2),
        .pipeline_identity = identity(3),
        .execution_policy_identity = identity(4),
        .icb_capability_identity = 5,
        .state_slot_identity = 6,
        .row_count = 3925,
        .context_base = 17,
        .graph_schema_version = kPrefillCommandGraphSchemaVersion,
        .chunk_rows = chunks,
        .persistent_resource_identities = resources,
    };
    if (validate_prefill_command_plan_key(key) !=
            PrefillCommandPlanError::None ||
        !same_prefill_command_plan_key(key, key)) {
        return false;
    }

    PrefillCommandPlanKey changed = key;
    changed.model_package_identity[0] ^= 1u;
    if (same_prefill_command_plan_key(key, changed)) {
        return false;
    }
    changed = key;
    changed.prepared_image_identity[0] ^= 1u;
    if (same_prefill_command_plan_key(key, changed)) {
        return false;
    }
    changed = key;
    changed.pipeline_identity[0] ^= 1u;
    if (same_prefill_command_plan_key(key, changed)) {
        return false;
    }
    changed = key;
    changed.execution_policy_identity[0] ^= 1u;
    if (same_prefill_command_plan_key(key, changed)) {
        return false;
    }
    changed = key;
    ++changed.icb_capability_identity;
    if (same_prefill_command_plan_key(key, changed)) {
        return false;
    }
    changed = key;
    ++changed.state_slot_identity;
    if (same_prefill_command_plan_key(key, changed)) {
        return false;
    }
    changed = key;
    ++changed.row_count;
    if (same_prefill_command_plan_key(key, changed)) {
        return false;
    }
    changed = key;
    ++changed.context_base;
    if (same_prefill_command_plan_key(key, changed)) {
        return false;
    }
    changed = key;
    ++changed.graph_schema_version;
    if (same_prefill_command_plan_key(key, changed)) {
        return false;
    }
    std::array<std::uint32_t, 3> changed_chunks = chunks;
    ++changed_chunks[2];
    changed = key;
    changed.chunk_rows = changed_chunks;
    if (same_prefill_command_plan_key(key, changed)) {
        return false;
    }
    std::array<std::uint64_t, 4> changed_resources = resources;
    ++changed_resources[3];
    changed = key;
    changed.persistent_resource_identities = changed_resources;
    if (same_prefill_command_plan_key(key, changed)) {
        return false;
    }

    PrefillCommandPlanKey invalid = key;
    invalid.model_package_identity = {};
    if (validate_prefill_command_plan_key(invalid) !=
        PrefillCommandPlanError::InvalidIdentity) {
        return false;
    }
    invalid = key;
    invalid.persistent_resource_identities = {};
    if (validate_prefill_command_plan_key(invalid) !=
        PrefillCommandPlanError::InvalidIdentity) {
        return false;
    }
    invalid = key;
    invalid.graph_schema_version = 0;
    if (validate_prefill_command_plan_key(invalid) !=
        PrefillCommandPlanError::InvalidSchemaVersion) {
        return false;
    }
    invalid = key;
    invalid.chunk_rows = {};
    if (validate_prefill_command_plan_key(invalid) !=
        PrefillCommandPlanError::EmptyChunkSchedule) {
        return false;
    }
    std::array<std::uint32_t, 3> zero_chunk = chunks;
    zero_chunk[1] = 0;
    invalid = key;
    invalid.chunk_rows = zero_chunk;
    if (validate_prefill_command_plan_key(invalid) !=
        PrefillCommandPlanError::ZeroChunkRows) {
        return false;
    }
    invalid = key;
    --invalid.row_count;
    if (validate_prefill_command_plan_key(invalid) !=
        PrefillCommandPlanError::RowCountMismatch) {
        return false;
    }
    invalid = key;
    invalid.context_base =
        std::numeric_limits<std::uint32_t>::max() - invalid.row_count + 1u;
    return validate_prefill_command_plan_key(invalid) ==
           PrefillCommandPlanError::ContextOutOfRange;
}

bool wavefront_contract() {
    constexpr std::uint32_t kLayers = 40;
    constexpr std::uint32_t kChunks = 3;
    const std::array<std::uint32_t, kChunks> chunks{256, 2048, 1621};
    std::array<PrefillCommandNode, kLayers * kChunks> nodes{};
    std::array<PrefillCommandDiagonal, kLayers + kChunks - 1u> diagonals{};
    const PrefillWavefrontRequest request{
        .layer_count = kLayers,
        .row_count = 3925,
        .context_base = 0,
        .context_capacity = 262144,
        .scratch_lane_count = kChunks,
        .chunk_rows = chunks,
    };
    const PrefillWavefrontPlanResult result =
        build_prefill_wavefront_plan(request, nodes, diagonals);
    if (!result || result.plan.node_count != 120 ||
        result.plan.diagonal_count != 42 ||
        result.plan.scratch_lane_count != 3) {
        return false;
    }

    std::array<std::uint64_t, kLayers * kChunks> wavefront_index{};
    std::array<bool, kLayers * kChunks> seen{};
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const PrefillCommandNode& node = nodes[index];
        if (node.layer_index >= kLayers || node.chunk_ordinal >= kChunks ||
            node.scratch_lane != node.chunk_ordinal ||
            node.diagonal_index != node.layer_index + node.chunk_ordinal ||
            node.layer_major_index >= seen.size() ||
            seen[node.layer_major_index]) {
            return false;
        }
        seen[node.layer_major_index] = true;
        wavefront_index[node.layer_major_index] = index;
    }
    for (const bool present : seen) {
        if (!present) {
            return false;
        }
    }
    for (std::size_t index = 0; index < nodes.size(); ++index) {
        const PrefillCommandNode& node = nodes[index];
        const std::uint64_t expected_hidden =
            node.layer_index == 0
                ? kNoPrefillCommandPredecessor
                : wavefront_index[(node.layer_index - 1u) * kChunks +
                                  node.chunk_ordinal];
        const std::uint64_t expected_state =
            node.chunk_ordinal == 0
                ? kNoPrefillCommandPredecessor
                : wavefront_index[node.layer_index * kChunks +
                                  node.chunk_ordinal - 1u];
        if (node.hidden_predecessor != expected_hidden ||
            node.state_predecessor != expected_state ||
            (expected_hidden != kNoPrefillCommandPredecessor &&
             (expected_hidden >= index ||
              nodes[expected_hidden].diagonal_index + 1u !=
                  node.diagonal_index)) ||
            (expected_state != kNoPrefillCommandPredecessor &&
             (expected_state >= index ||
              nodes[expected_state].diagonal_index + 1u !=
                  node.diagonal_index))) {
            return false;
        }
    }
    std::uint64_t diagonal_nodes = 0;
    for (std::uint32_t diagonal = 0; diagonal < diagonals.size();
         ++diagonal) {
        const PrefillCommandDiagonal& range = diagonals[diagonal];
        if (range.node_begin != diagonal_nodes || range.node_count == 0 ||
            range.node_count > kChunks ||
            range.node_begin + range.node_count > nodes.size()) {
            return false;
        }
        for (std::uint64_t index = range.node_begin;
             index < range.node_begin + range.node_count; ++index) {
            if (nodes[index].diagonal_index != diagonal) {
                return false;
            }
        }
        diagonal_nodes += range.node_count;
    }
    if (diagonal_nodes != nodes.size()) {
        return false;
    }

    PrefillWavefrontRequest invalid = request;
    invalid.layer_count = 0;
    if (build_prefill_wavefront_plan(invalid, nodes, diagonals).error !=
        PrefillCommandPlanError::InvalidLayerCount) {
        return false;
    }
    invalid = request;
    invalid.chunk_rows = {};
    if (build_prefill_wavefront_plan(invalid, nodes, diagonals).error !=
        PrefillCommandPlanError::EmptyChunkSchedule) {
        return false;
    }
    invalid = request;
    invalid.scratch_lane_count = 2;
    if (build_prefill_wavefront_plan(invalid, nodes, diagonals).error !=
        PrefillCommandPlanError::ScratchLaneMismatch) {
        return false;
    }
    invalid = request;
    --invalid.row_count;
    if (build_prefill_wavefront_plan(invalid, nodes, diagonals).error !=
        PrefillCommandPlanError::RowCountMismatch) {
        return false;
    }
    invalid = request;
    invalid.context_capacity = 3924;
    if (build_prefill_wavefront_plan(invalid, nodes, diagonals).error !=
        PrefillCommandPlanError::ContextOutOfRange) {
        return false;
    }
    if (build_prefill_wavefront_plan(
            request, std::span(nodes).first(nodes.size() - 1u),
            diagonals)
            .error != PrefillCommandPlanError::NodeStorageTooSmall) {
        return false;
    }
    return build_prefill_wavefront_plan(
               request, nodes,
               std::span(diagonals).first(diagonals.size() - 1u))
               .error ==
           PrefillCommandPlanError::DiagonalStorageTooSmall;
}

bool lane_event_contract() {
    constexpr std::uint32_t kLayers = 40;
    constexpr std::uint32_t kLanes = 3;
    const std::array<std::uint32_t, kLanes> chunks{
        256, 2048, 1621};
    std::array<PrefillCommandNode, kLayers * kLanes> nodes{};
    std::array<
        PrefillCommandDiagonal, kLayers + kLanes - 1u>
        diagonals{};
    const PrefillWavefrontPlanResult wavefront =
        build_prefill_wavefront_plan(
            {
                .layer_count = kLayers,
                .row_count = 3925,
                .context_base = 0,
                .context_capacity = 262144,
                .scratch_lane_count = kLanes,
                .chunk_rows = chunks,
            },
            nodes, diagonals);
    if (!wavefront) {
        return false;
    }
    std::array<PrefillLaneEventNode, kLayers * kLanes>
        events{};
    const PrefillLaneEventRequest request{
        .layer_count = kLayers,
        .scratch_lane_count = kLanes,
        .event_value_base = 80,
        .nodes = nodes,
    };
    const PrefillLaneEventPlanResult result =
        build_prefill_lane_event_plan(request, events);
    if (!result || result.plan.node_count != 120 ||
        result.plan.event_count != 2 ||
        result.plan.terminal_event_value != 120) {
        return false;
    }
    for (std::uint32_t layer = 0; layer < kLayers;
         ++layer) {
        const std::uint64_t value = 81u + layer;
        for (std::uint32_t lane = 0; lane < kLanes;
             ++lane) {
            const PrefillLaneEventNode& event =
                events[layer * kLanes + lane];
            if (event.layer_index != layer ||
                event.scratch_lane != lane ||
                event.node_index >= nodes.size() ||
                nodes[event.node_index].layer_index != layer ||
                nodes[event.node_index].chunk_ordinal != lane ||
                event.wait_event !=
                    (lane == 0 ? kNoPrefillLaneEvent
                               : lane - 1u) ||
                event.wait_value !=
                    (lane == 0 ? 0 : value) ||
                event.signal_event !=
                    (lane + 1u == kLanes
                         ? kNoPrefillLaneEvent
                         : lane) ||
                event.signal_value !=
                    (lane + 1u == kLanes ? 0 : value)) {
                return false;
            }
        }
    }

    auto invalid_nodes = nodes;
    invalid_nodes[1].state_predecessor =
        kNoPrefillCommandPredecessor;
    PrefillLaneEventRequest invalid = request;
    invalid.nodes = invalid_nodes;
    if (build_prefill_lane_event_plan(invalid, events).error !=
        PrefillCommandPlanError::InvalidWavefrontNode) {
        return false;
    }
    invalid = request;
    invalid.event_value_base =
        std::numeric_limits<std::uint64_t>::max() -
        kLayers + 1u;
    if (build_prefill_lane_event_plan(invalid, events).error !=
        PrefillCommandPlanError::InvalidEventValue) {
        return false;
    }
    invalid = request;
    invalid.scratch_lane_count = 0;
    if (build_prefill_lane_event_plan(invalid, events).error !=
        PrefillCommandPlanError::ScratchLaneMismatch) {
        return false;
    }
    return build_prefill_lane_event_plan(
               request,
               std::span(events).first(events.size() - 1u))
               .error ==
           PrefillCommandPlanError::LaneEventStorageTooSmall;
}

bool arena_contract() {
    const std::array<PrefillArgumentBindingRequest, 3> requests{{
        {.command_index = 0, .buffer_index = 5, .size_bytes = 4,
         .alignment_bytes = 4},
        {.command_index = 0, .buffer_index = 6, .size_bytes = 8,
         .alignment_bytes = 8},
        {.command_index = 1, .buffer_index = 5, .size_bytes = 4,
         .alignment_bytes = 4},
    }};
    std::array<PrefillArgumentBinding, 3> bindings{};
    const PrefillArgumentArenaPlanResult result =
        build_prefill_argument_arena_plan(2, 30, 20, requests, bindings);
    if (!result || result.plan.required_bytes != 20 ||
        result.plan.binding_count != 3 ||
        bindings[0].offset_bytes != 0 || bindings[1].offset_bytes != 8 ||
        bindings[2].offset_bytes != 16) {
        return false;
    }
    if (build_prefill_argument_arena_plan(2, 30, 19, requests, bindings)
            .error != PrefillCommandPlanError::ArgumentArenaTooSmall ||
        build_prefill_argument_arena_plan(
            2, 30, 20, requests,
            std::span(bindings).first(bindings.size() - 1u))
                .error !=
            PrefillCommandPlanError::ArgumentStorageTooSmall ||
        build_prefill_argument_arena_plan(0, 30, 20, requests, bindings)
                .error != PrefillCommandPlanError::InvalidCommandCount) {
        return false;
    }

    std::array<PrefillArgumentBindingRequest, 3> invalid = requests;
    invalid[0].command_index = 2;
    if (build_prefill_argument_arena_plan(2, 30, 20, invalid, bindings)
            .error != PrefillCommandPlanError::InvalidCommandIndex) {
        return false;
    }
    invalid = requests;
    invalid[0].buffer_index = 31;
    if (build_prefill_argument_arena_plan(2, 30, 20, invalid, bindings)
            .error != PrefillCommandPlanError::InvalidBufferIndex) {
        return false;
    }
    invalid = requests;
    invalid[0].size_bytes = 0;
    if (build_prefill_argument_arena_plan(2, 30, 20, invalid, bindings)
            .error != PrefillCommandPlanError::InvalidArgumentSize) {
        return false;
    }
    invalid = requests;
    invalid[0].alignment_bytes = 3;
    if (build_prefill_argument_arena_plan(2, 30, 20, invalid, bindings)
            .error != PrefillCommandPlanError::InvalidArgumentAlignment) {
        return false;
    }
    invalid = requests;
    invalid[1].buffer_index = invalid[0].buffer_index;
    if (build_prefill_argument_arena_plan(2, 30, 20, invalid, bindings)
            .error != PrefillCommandPlanError::DuplicateArgumentBinding) {
        return false;
    }
    invalid = requests;
    invalid[0].size_bytes = std::numeric_limits<std::uint64_t>::max();
    invalid[1].alignment_bytes = 1;
    return build_prefill_argument_arena_plan(
               2, 30, std::numeric_limits<std::uint64_t>::max(), invalid,
               bindings)
               .error == PrefillCommandPlanError::ArithmeticOverflow;
}

bool resource_contract() {
    const std::array<PrefillResourceRequest, 3> requests{{
        {.identity = 11, .usage = PrefillResourceUsage::Read},
        {.identity = 22, .usage = PrefillResourceUsage::Write},
        {.identity = 11, .usage = PrefillResourceUsage::Write},
    }};
    std::array<PrefillResourceEntry, 3> resources{};
    const PrefillResourceTableResult result =
        build_prefill_resource_table(requests, resources);
    if (!result || result.resource_count != 2 ||
        resources[0].identity != 11 ||
        resources[0].usage != PrefillResourceUsage::ReadWrite ||
        resources[1].identity != 22 ||
        resources[1].usage != PrefillResourceUsage::Write) {
        return false;
    }
    std::array<PrefillResourceRequest, 3> invalid = requests;
    invalid[0].identity = 0;
    if (build_prefill_resource_table(invalid, resources).error !=
        PrefillCommandPlanError::InvalidResourceIdentity) {
        return false;
    }
    invalid = requests;
    invalid[0].usage = static_cast<PrefillResourceUsage>(4);
    if (build_prefill_resource_table(invalid, resources).error !=
        PrefillCommandPlanError::InvalidResourceUsage) {
        return false;
    }
    return build_prefill_resource_table(
               requests, std::span(resources).first(1))
               .error ==
           PrefillCommandPlanError::ResourceStorageTooSmall;
}

bool routed_grid_contract() {
    const std::array<std::uint32_t, 3> chunks{256, 2048, 1621};
    std::array<std::uint32_t, 3> maximum_tasks{};
    const PrefillRoutedGridRequest request{
        .chunk_rows = chunks,
        .routes_per_position = 9,
        .task_capacity = 4096,
        .tile_columns = 32,
        .upgate_columns = 512,
        .down_columns = 2048,
    };
    const PrefillRoutedGridPlanResult result =
        build_prefill_routed_grid_plan(request, maximum_tasks);
    if (!result || result.grid.maximum_task_count != 4096 ||
        result.grid.upgate_column_groups != 16 ||
        result.grid.down_column_groups != 64 ||
        maximum_tasks != std::array<std::uint32_t, 3>{2304, 4096, 4096}) {
        return false;
    }
    if (build_prefill_routed_grid_plan(
            request, std::span(maximum_tasks).first(2))
            .error != PrefillCommandPlanError::RoutedGridStorageTooSmall) {
        return false;
    }
    PrefillRoutedGridRequest invalid = request;
    invalid.routes_per_position = 0;
    return build_prefill_routed_grid_plan(invalid, maximum_tasks).error ==
           PrefillCommandPlanError::InvalidRoutedGrid;
}

} // namespace

int main() {
    return key_contract() && wavefront_contract() &&
                   lane_event_contract() && arena_contract() &&
                   resource_contract() && routed_grid_contract()
               ? 0
               : 1;
}
