#include "tatara/runtime/prefill_command_plan.h"

#include <algorithm>

namespace tatara::runtime {
namespace {

bool identity_is_zero(const PrefillCommandIdentity& identity) noexcept {
    return std::all_of(identity.begin(), identity.end(),
                       [](std::uint8_t value) { return value == 0; });
}

bool add(std::uint64_t left, std::uint64_t right,
         std::uint64_t& result) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool multiply(std::uint64_t left, std::uint64_t right,
              std::uint64_t& result) noexcept {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool power_of_two(std::uint64_t value) noexcept {
    return value != 0 && (value & (value - 1u)) == 0;
}

bool align_up(std::uint64_t value, std::uint64_t alignment,
              std::uint64_t& result) noexcept {
    if (!power_of_two(alignment)) {
        return false;
    }
    const std::uint64_t mask = alignment - 1u;
    std::uint64_t adjusted = 0;
    if (!add(value, mask, adjusted)) {
        return false;
    }
    result = adjusted & ~mask;
    return true;
}

std::uint32_t diagonal_node_count(std::uint32_t diagonal,
                                  std::uint32_t layers,
                                  std::uint32_t chunks) noexcept {
    const std::uint32_t first_layer =
        diagonal >= chunks ? diagonal - (chunks - 1u) : 0u;
    const std::uint32_t last_layer =
        std::min(diagonal, layers - 1u);
    return last_layer - first_layer + 1u;
}

bool diagonal_begin(std::uint32_t diagonal, std::uint32_t layers,
                    std::uint32_t chunks,
                    std::uint64_t& begin) noexcept {
    begin = 0;
    for (std::uint32_t index = 0; index < diagonal; ++index) {
        if (!add(begin, diagonal_node_count(index, layers, chunks), begin)) {
            return false;
        }
    }
    return true;
}

bool wavefront_index(std::uint32_t layer, std::uint32_t chunk,
                     std::uint32_t layers, std::uint32_t chunks,
                     std::uint64_t& index) noexcept {
    const std::uint32_t diagonal = layer + chunk;
    if (!diagonal_begin(diagonal, layers, chunks, index)) {
        return false;
    }
    const std::uint32_t first_layer =
        diagonal >= chunks ? diagonal - (chunks - 1u) : 0u;
    return add(index, layer - first_layer, index);
}

bool valid_usage(PrefillResourceUsage usage) noexcept {
    const auto value = static_cast<std::uint8_t>(usage);
    return value >= static_cast<std::uint8_t>(PrefillResourceUsage::Read) &&
           value <= static_cast<std::uint8_t>(
                        PrefillResourceUsage::ReadWrite);
}

PrefillResourceUsage merge_usage(PrefillResourceUsage left,
                                 PrefillResourceUsage right) noexcept {
    return static_cast<PrefillResourceUsage>(
        static_cast<std::uint8_t>(left) |
        static_cast<std::uint8_t>(right));
}

bool ceil_div(std::uint32_t value, std::uint32_t divisor,
              std::uint32_t& result) noexcept {
    if (value == 0 || divisor == 0) {
        return false;
    }
    result = 1u + (value - 1u) / divisor;
    return true;
}

} // namespace

PrefillCommandPlanError
validate_prefill_command_plan_key(const PrefillCommandPlanKey& key) noexcept {
    if (identity_is_zero(key.model_package_identity) ||
        identity_is_zero(key.prepared_image_identity) ||
        identity_is_zero(key.pipeline_identity) ||
        identity_is_zero(key.execution_policy_identity) ||
        key.icb_capability_identity == 0 || key.state_slot_identity == 0 ||
        key.persistent_resource_identities.empty() ||
        std::any_of(key.persistent_resource_identities.begin(),
                    key.persistent_resource_identities.end(),
                    [](std::uint64_t identity) { return identity == 0; })) {
        return PrefillCommandPlanError::InvalidIdentity;
    }
    if (key.graph_schema_version != kPrefillCommandGraphSchemaVersion) {
        return PrefillCommandPlanError::InvalidSchemaVersion;
    }
    if (key.chunk_rows.empty()) {
        return PrefillCommandPlanError::EmptyChunkSchedule;
    }
    std::uint64_t rows = 0;
    for (const std::uint32_t chunk_rows : key.chunk_rows) {
        if (chunk_rows == 0) {
            return PrefillCommandPlanError::ZeroChunkRows;
        }
        if (!add(rows, chunk_rows, rows)) {
            return PrefillCommandPlanError::ArithmeticOverflow;
        }
    }
    if (rows != key.row_count) {
        return PrefillCommandPlanError::RowCountMismatch;
    }
    std::uint64_t next_context = 0;
    if (!add(key.context_base, key.row_count, next_context) ||
        next_context > std::numeric_limits<std::uint32_t>::max()) {
        return PrefillCommandPlanError::ContextOutOfRange;
    }
    return PrefillCommandPlanError::None;
}

bool same_prefill_command_plan_key(const PrefillCommandPlanKey& left,
                                   const PrefillCommandPlanKey& right) noexcept {
    return left.model_package_identity == right.model_package_identity &&
           left.prepared_image_identity == right.prepared_image_identity &&
           left.pipeline_identity == right.pipeline_identity &&
           left.execution_policy_identity ==
               right.execution_policy_identity &&
           left.icb_capability_identity == right.icb_capability_identity &&
           left.state_slot_identity == right.state_slot_identity &&
           left.row_count == right.row_count &&
           left.context_base == right.context_base &&
           left.graph_schema_version == right.graph_schema_version &&
           std::ranges::equal(left.chunk_rows, right.chunk_rows) &&
           std::ranges::equal(left.persistent_resource_identities,
                              right.persistent_resource_identities);
}

PrefillWavefrontPlanResult build_prefill_wavefront_plan(
    const PrefillWavefrontRequest& request,
    std::span<PrefillCommandNode> nodes,
    std::span<PrefillCommandDiagonal> diagonals) noexcept {
    if (request.layer_count == 0) {
        return {.error = PrefillCommandPlanError::InvalidLayerCount};
    }
    if (request.chunk_rows.empty()) {
        return {.error = PrefillCommandPlanError::EmptyChunkSchedule};
    }
    if (request.scratch_lane_count != request.chunk_rows.size()) {
        return {.error = PrefillCommandPlanError::ScratchLaneMismatch};
    }

    std::uint64_t rows = 0;
    for (const std::uint32_t chunk_rows : request.chunk_rows) {
        if (chunk_rows == 0) {
            return {.error = PrefillCommandPlanError::ZeroChunkRows};
        }
        if (!add(rows, chunk_rows, rows)) {
            return {.error = PrefillCommandPlanError::ArithmeticOverflow};
        }
    }
    if (rows != request.row_count) {
        return {.error = PrefillCommandPlanError::RowCountMismatch};
    }
    std::uint64_t next_context = 0;
    if (!add(request.context_base, request.row_count, next_context) ||
        next_context > request.context_capacity) {
        return {.error = PrefillCommandPlanError::ContextOutOfRange};
    }

    const std::uint64_t chunks = request.chunk_rows.size();
    std::uint64_t node_count = 0;
    if (!multiply(request.layer_count, chunks, node_count)) {
        return {.error = PrefillCommandPlanError::ArithmeticOverflow};
    }
    std::uint64_t diagonal_count64 = 0;
    if (!add(request.layer_count, chunks - 1u, diagonal_count64) ||
        diagonal_count64 > std::numeric_limits<std::uint32_t>::max()) {
        return {.error = PrefillCommandPlanError::ArithmeticOverflow};
    }
    const auto diagonal_count =
        static_cast<std::uint32_t>(diagonal_count64);
    const PrefillWavefrontPlan plan{
        .node_count = node_count,
        .diagonal_count = diagonal_count,
        .scratch_lane_count = request.scratch_lane_count,
    };
    if (nodes.size() < node_count) {
        return {
            .error = PrefillCommandPlanError::NodeStorageTooSmall,
            .plan = plan,
        };
    }
    if (diagonals.size() < diagonal_count) {
        return {
            .error = PrefillCommandPlanError::DiagonalStorageTooSmall,
            .plan = plan,
        };
    }

    const auto chunk_count =
        static_cast<std::uint32_t>(request.chunk_rows.size());
    std::uint64_t output_index = 0;
    for (std::uint32_t diagonal = 0; diagonal < diagonal_count;
         ++diagonal) {
        const std::uint32_t first_layer =
            diagonal >= chunk_count
                ? diagonal - (chunk_count - 1u)
                : 0u;
        const std::uint32_t last_layer =
            std::min(diagonal, request.layer_count - 1u);
        const std::uint32_t count =
            last_layer - first_layer + 1u;
        diagonals[diagonal] = {
            .node_begin = output_index,
            .node_count = count,
        };
        for (std::uint32_t layer = first_layer; layer <= last_layer;
             ++layer) {
            const std::uint32_t chunk = diagonal - layer;
            std::uint64_t layer_major_index = 0;
            if (!multiply(layer, chunk_count, layer_major_index) ||
                !add(layer_major_index, chunk, layer_major_index)) {
                return {
                    .error = PrefillCommandPlanError::ArithmeticOverflow,
                    .plan = plan,
                };
            }
            std::uint64_t hidden_predecessor =
                kNoPrefillCommandPredecessor;
            std::uint64_t state_predecessor =
                kNoPrefillCommandPredecessor;
            if (layer != 0 &&
                !wavefront_index(layer - 1u, chunk, request.layer_count,
                                 chunk_count, hidden_predecessor)) {
                return {
                    .error = PrefillCommandPlanError::ArithmeticOverflow,
                    .plan = plan,
                };
            }
            if (chunk != 0 &&
                !wavefront_index(layer, chunk - 1u, request.layer_count,
                                 chunk_count, state_predecessor)) {
                return {
                    .error = PrefillCommandPlanError::ArithmeticOverflow,
                    .plan = plan,
                };
            }
            nodes[output_index++] = {
                .layer_index = layer,
                .chunk_ordinal = chunk,
                .diagonal_index = diagonal,
                .scratch_lane = chunk,
                .layer_major_index = layer_major_index,
                .hidden_predecessor = hidden_predecessor,
                .state_predecessor = state_predecessor,
            };
        }
    }
    if (output_index != node_count) {
        return {
            .error = PrefillCommandPlanError::ArithmeticOverflow,
            .plan = plan,
        };
    }
    return {.error = PrefillCommandPlanError::None, .plan = plan};
}

PrefillLaneEventPlanResult build_prefill_lane_event_plan(
    const PrefillLaneEventRequest& request,
    std::span<PrefillLaneEventNode> event_nodes) noexcept {
    if (request.layer_count == 0) {
        return {.error = PrefillCommandPlanError::InvalidLayerCount};
    }
    if (request.scratch_lane_count == 0) {
        return {.error = PrefillCommandPlanError::ScratchLaneMismatch};
    }
    std::uint64_t node_count = 0;
    if (!multiply(
            request.layer_count, request.scratch_lane_count,
            node_count)) {
        return {.error = PrefillCommandPlanError::ArithmeticOverflow};
    }
    const PrefillLaneEventPlan plan{
        .node_count = node_count,
        .event_count = request.scratch_lane_count - 1u,
        .terminal_event_value = 0,
    };
    if (request.nodes.size() != node_count) {
        return {
            .error = PrefillCommandPlanError::InvalidWavefrontNode,
            .plan = plan,
        };
    }
    if (event_nodes.size() < node_count) {
        return {
            .error = PrefillCommandPlanError::LaneEventStorageTooSmall,
            .plan = plan,
        };
    }
    std::uint64_t terminal_event_value = 0;
    if (!add(
            request.event_value_base, request.layer_count,
            terminal_event_value) ||
        terminal_event_value == 0) {
        return {
            .error = PrefillCommandPlanError::InvalidEventValue,
            .plan = plan,
        };
    }

    const auto find_node =
        [&request](std::uint32_t layer,
                   std::uint32_t lane) noexcept {
            std::uint64_t found = kNoPrefillCommandPredecessor;
            for (std::uint64_t index = 0;
                 index < request.nodes.size(); ++index) {
                const PrefillCommandNode& node =
                    request.nodes[index];
                if (node.layer_index == layer &&
                    node.chunk_ordinal == lane) {
                    if (found !=
                        kNoPrefillCommandPredecessor) {
                        return kNoPrefillCommandPredecessor;
                    }
                    found = index;
                }
            }
            return found;
        };

    for (std::uint32_t layer = 0;
         layer < request.layer_count; ++layer) {
        std::uint64_t event_value = 0;
        if (!add(
                request.event_value_base,
                std::uint64_t{layer} + 1u, event_value) ||
            event_value == 0) {
            return {
                .error =
                    PrefillCommandPlanError::InvalidEventValue,
                .plan = plan,
            };
        }
        for (std::uint32_t lane = 0;
             lane < request.scratch_lane_count; ++lane) {
            const std::uint64_t node_index =
                find_node(layer, lane);
            if (node_index ==
                kNoPrefillCommandPredecessor) {
                return {
                    .error = PrefillCommandPlanError::
                        InvalidWavefrontNode,
                    .plan = plan,
                };
            }
            const PrefillCommandNode& node =
                request.nodes[node_index];
            const std::uint64_t expected_layer_major =
                std::uint64_t{layer} *
                    request.scratch_lane_count +
                lane;
            const std::uint64_t expected_hidden =
                layer == 0
                    ? kNoPrefillCommandPredecessor
                    : find_node(layer - 1u, lane);
            const std::uint64_t expected_state =
                lane == 0
                    ? kNoPrefillCommandPredecessor
                    : find_node(layer, lane - 1u);
            if (node.scratch_lane != lane ||
                node.diagonal_index != layer + lane ||
                node.layer_major_index !=
                    expected_layer_major ||
                node.hidden_predecessor != expected_hidden ||
                node.state_predecessor != expected_state) {
                return {
                    .error = PrefillCommandPlanError::
                        InvalidWavefrontNode,
                    .plan = plan,
                };
            }
            event_nodes[expected_layer_major] = {
                .node_index = node_index,
                .layer_index = layer,
                .scratch_lane = lane,
                .wait_event =
                    lane == 0 ? kNoPrefillLaneEvent
                              : lane - 1u,
                .wait_value =
                    lane == 0 ? 0 : event_value,
                .signal_event =
                    lane + 1u ==
                            request.scratch_lane_count
                        ? kNoPrefillLaneEvent
                        : lane,
                .signal_value =
                    lane + 1u ==
                            request.scratch_lane_count
                        ? 0
                        : event_value,
            };
        }
    }
    PrefillLaneEventPlan completed = plan;
    completed.terminal_event_value = terminal_event_value;
    return {
        .error = PrefillCommandPlanError::None,
        .plan = completed,
    };
}

PrefillArgumentArenaPlanResult build_prefill_argument_arena_plan(
    std::uint32_t command_count, std::uint32_t maximum_buffer_index,
    std::uint64_t arena_capacity_bytes,
    std::span<const PrefillArgumentBindingRequest> requests,
    std::span<PrefillArgumentBinding> bindings) noexcept {
    if (command_count == 0) {
        return {
            .error = PrefillCommandPlanError::InvalidCommandCount,
        };
    }

    std::uint64_t cursor = 0;
    for (std::size_t index = 0; index < requests.size(); ++index) {
        const PrefillArgumentBindingRequest& request = requests[index];
        if (request.command_index >= command_count) {
            return {
                .error = PrefillCommandPlanError::InvalidCommandIndex,
            };
        }
        if (request.buffer_index > maximum_buffer_index) {
            return {
                .error = PrefillCommandPlanError::InvalidBufferIndex,
            };
        }
        if (request.size_bytes == 0) {
            return {
                .error = PrefillCommandPlanError::InvalidArgumentSize,
            };
        }
        if (!power_of_two(request.alignment_bytes)) {
            return {
                .error = PrefillCommandPlanError::InvalidArgumentAlignment,
            };
        }
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (requests[prior].command_index == request.command_index &&
                requests[prior].buffer_index == request.buffer_index) {
                return {
                    .error =
                        PrefillCommandPlanError::DuplicateArgumentBinding,
                };
            }
        }
        std::uint64_t offset = 0;
        if (!align_up(cursor, request.alignment_bytes, offset) ||
            !add(offset, request.size_bytes, cursor)) {
            return {
                .error = PrefillCommandPlanError::ArithmeticOverflow,
            };
        }
    }

    if (requests.size() >
        std::numeric_limits<std::uint32_t>::max()) {
        return {
            .error = PrefillCommandPlanError::ArithmeticOverflow,
        };
    }
    const PrefillArgumentArenaPlan plan{
        .required_bytes = cursor,
        .binding_count =
            static_cast<std::uint32_t>(requests.size()),
    };
    if (bindings.size() < requests.size()) {
        return {
            .error = PrefillCommandPlanError::ArgumentStorageTooSmall,
            .plan = plan,
        };
    }
    if (cursor > arena_capacity_bytes) {
        return {
            .error = PrefillCommandPlanError::ArgumentArenaTooSmall,
            .plan = plan,
        };
    }

    cursor = 0;
    for (std::size_t index = 0; index < requests.size(); ++index) {
        const PrefillArgumentBindingRequest& request = requests[index];
        std::uint64_t offset = 0;
        if (!align_up(cursor, request.alignment_bytes, offset) ||
            !add(offset, request.size_bytes, cursor)) {
            return {
                .error = PrefillCommandPlanError::ArithmeticOverflow,
                .plan = plan,
            };
        }
        bindings[index] = {
            .command_index = request.command_index,
            .buffer_index = request.buffer_index,
            .offset_bytes = offset,
            .size_bytes = request.size_bytes,
        };
    }
    return {.error = PrefillCommandPlanError::None, .plan = plan};
}

PrefillResourceTableResult build_prefill_resource_table(
    std::span<const PrefillResourceRequest> requests,
    std::span<PrefillResourceEntry> resources) noexcept {
    std::uint32_t resource_count = 0;
    for (const PrefillResourceRequest& request : requests) {
        if (request.identity == 0) {
            return {
                .error = PrefillCommandPlanError::InvalidResourceIdentity,
                .resource_count = resource_count,
            };
        }
        if (!valid_usage(request.usage)) {
            return {
                .error = PrefillCommandPlanError::InvalidResourceUsage,
                .resource_count = resource_count,
            };
        }
        std::uint32_t existing = resource_count;
        for (std::uint32_t index = 0; index < resource_count; ++index) {
            if (resources[index].identity == request.identity) {
                existing = index;
                break;
            }
        }
        if (existing != resource_count) {
            resources[existing].usage =
                merge_usage(resources[existing].usage, request.usage);
            continue;
        }
        if (resource_count == resources.size()) {
            return {
                .error = PrefillCommandPlanError::ResourceStorageTooSmall,
                .resource_count = resource_count,
            };
        }
        resources[resource_count++] = {
            .identity = request.identity,
            .usage = request.usage,
        };
    }
    return {
        .error = PrefillCommandPlanError::None,
        .resource_count = resource_count,
    };
}

PrefillRoutedGridPlanResult build_prefill_routed_grid_plan(
    const PrefillRoutedGridRequest& request,
    std::span<std::uint32_t> maximum_task_counts) noexcept {
    if (request.chunk_rows.empty() ||
        request.routes_per_position == 0 || request.task_capacity == 0 ||
        request.tile_columns == 0 || request.upgate_columns == 0 ||
        request.down_columns == 0) {
        return {.error = PrefillCommandPlanError::InvalidRoutedGrid};
    }
    if (maximum_task_counts.size() < request.chunk_rows.size()) {
        return {
            .error = PrefillCommandPlanError::RoutedGridStorageTooSmall,
        };
    }
    std::uint32_t upgate_column_groups = 0;
    std::uint32_t down_column_groups = 0;
    if (!ceil_div(request.upgate_columns, request.tile_columns,
                  upgate_column_groups) ||
        !ceil_div(request.down_columns, request.tile_columns,
                  down_column_groups)) {
        return {.error = PrefillCommandPlanError::InvalidRoutedGrid};
    }
    std::uint64_t maximum_tasks = 0;
    for (std::size_t index = 0; index < request.chunk_rows.size();
         ++index) {
        const std::uint32_t rows = request.chunk_rows[index];
        if (rows == 0) {
            return {.error = PrefillCommandPlanError::ZeroChunkRows};
        }
        std::uint64_t routed_rows = 0;
        if (!multiply(rows, request.routes_per_position, routed_rows)) {
            return {.error = PrefillCommandPlanError::ArithmeticOverflow};
        }
        const std::uint64_t bounded_tasks =
            std::min<std::uint64_t>(routed_rows,
                                    request.task_capacity);
        maximum_task_counts[index] =
            static_cast<std::uint32_t>(bounded_tasks);
        maximum_tasks = std::max(maximum_tasks, bounded_tasks);
    }
    return {
        .error = PrefillCommandPlanError::None,
        .grid = {
            .maximum_task_count =
                static_cast<std::uint32_t>(maximum_tasks),
            .upgate_column_groups = upgate_column_groups,
            .down_column_groups = down_column_groups,
        },
    };
}

} // namespace tatara::runtime
