#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tatara::runtime {

enum class QuantizedGemmShapeError : std::uint8_t {
    None,
    InvalidPlan,
    InvalidTile,
    Overflow,
    EncoderRepresentability,
    InputColumnLimit,
    BundleOutputLimit,
    DescriptorLimit,
    TaskLimit,
    InvalidLimits,
};

enum class QuantizedGemmModelWeightFormat : std::uint8_t {
    Unknown = 0,
    AffineQ4 = 4,
    AffineQ8 = 8,
};

enum class QuantizedGemmBundleKind : std::uint8_t {
    GatedDeltaInput,
    GatedDeltaOutput,
    AttentionInput,
    AttentionOutput,
    RouterQ8,
    RoutedGate,
    RoutedUp,
    RoutedDown,
    SharedGate,
    SharedUp,
    SharedDown,
    Count,
};

enum class QuantizedGemmRegionKind : std::uint8_t {
    GatedDeltaQkv,
    GatedDeltaZ,
    GatedDeltaB,
    GatedDeltaA,
    GatedDeltaOutput,
    AttentionQuery,
    AttentionKey,
    AttentionValue,
    AttentionOutput,
    RoutedRouter,
    SharedRouter,
    RoutedGate,
    RoutedUp,
    RoutedDown,
    SharedGate,
    SharedUp,
    SharedDown,
    Count,
};

inline constexpr std::size_t kQuantizedGemmBundleCount =
    static_cast<std::size_t>(QuantizedGemmBundleKind::Count);
inline constexpr std::size_t kQuantizedGemmRegionCount =
    static_cast<std::size_t>(QuantizedGemmRegionKind::Count);
inline constexpr std::uint32_t kQuantizedTensorDescriptorsPerRegion = 3;
inline constexpr std::uint32_t kQuantizedGemmMaximumTileRows = 1024;

struct QuantizedGemmShapeLimits {
    std::uint32_t output_tile_rows{0};
    std::uint32_t maximum_input_columns{0};
    std::uint32_t maximum_bundle_output_rows{0};
    std::uint64_t maximum_descriptor_count{0};
    std::uint64_t maximum_task_capacity{0};
};

struct QuantizedGemmModelShape {
    std::uint32_t hidden{0};
    std::uint32_t query_heads{0};
    std::uint32_t key_value_heads{0};
    std::uint32_t head_dimension{0};
    std::uint32_t recurrent_heads{0};
    std::uint32_t state_dimension{0};
    std::uint32_t experts{0};
    std::uint32_t per_token_active_experts{0};
    std::uint32_t expert_dimension{0};
    QuantizedGemmModelWeightFormat weight_format{
        QuantizedGemmModelWeightFormat::Unknown};
    std::uint32_t group_size{0};
    std::uint32_t model_layers{0};
    std::uint32_t gated_delta_layers{0};
    std::uint32_t attention_layers{0};
};

struct QuantizedGemmRegion {
    QuantizedGemmRegionKind kind{QuantizedGemmRegionKind::GatedDeltaQkv};
    QuantizedGemmBundleKind bundle{QuantizedGemmBundleKind::GatedDeltaInput};
    std::uint32_t bundle_row_begin{0};
    std::uint32_t bundle_row_end{0};
    std::uint32_t input_columns{0};
    std::uint32_t instance_count{0};
    std::uint32_t output_rows_per_instance{0};
    std::uint32_t total_output_rows{0};
    std::uint64_t task_begin{0};
    std::uint64_t task_end{0};
};

struct QuantizedGemmBundle {
    QuantizedGemmBundleKind kind{QuantizedGemmBundleKind::GatedDeltaInput};
    std::uint32_t region_begin{0};
    std::uint32_t region_end{0};
    std::uint32_t input_columns{0};
    std::uint32_t output_rows{0};
    std::uint64_t task_begin{0};
    std::uint64_t task_end{0};
};

struct QuantizedGemmShapeTable {
    std::uint32_t hidden{0};
    std::uint32_t query_heads{0};
    std::uint32_t key_value_heads{0};
    std::uint32_t head_dimension{0};
    std::uint32_t recurrent_heads{0};
    std::uint32_t state_dimension{0};
    std::uint32_t experts{0};
    std::uint32_t per_token_active_experts{0};
    std::uint32_t expert_dimension{0};
    std::uint32_t group_size{0};
    std::uint32_t gated_delta_layers{0};
    std::uint32_t attention_layers{0};
    std::uint32_t output_tile_rows{0};

    std::array<QuantizedGemmBundle, kQuantizedGemmBundleCount> bundles{};
    std::array<QuantizedGemmRegion, kQuantizedGemmRegionCount> regions{};

    // These capacities cover the static output-tile axis. The dynamic row
    // axis is bounded separately from the routed row counts, never from top-k.
    std::uint64_t table_task_capacity{0};
    std::uint64_t gated_delta_layer_task_capacity{0};
    std::uint64_t attention_layer_task_capacity{0};
    std::uint64_t model_task_capacity{0};
    std::uint64_t mixture_descriptor_count_per_layer{0};
    std::uint64_t gated_delta_layer_descriptor_count{0};
    std::uint64_t attention_layer_descriptor_count{0};
    std::uint64_t model_descriptor_count{0};
};

struct QuantizedGemmShapeTableResult {
    QuantizedGemmShapeError error{QuantizedGemmShapeError::InvalidPlan};
    QuantizedGemmShapeTable table{};

    constexpr explicit operator bool() const noexcept {
        return error == QuantizedGemmShapeError::None;
    }
};

namespace quantized_gemm_shape_detail {

constexpr bool add(std::uint64_t left, std::uint64_t right,
                   std::uint64_t& result) noexcept {
    if (left > std::numeric_limits<std::uint64_t>::max() - right) {
        return false;
    }
    result = left + right;
    return true;
}

constexpr bool multiply(std::uint64_t left, std::uint64_t right,
                        std::uint64_t& result) noexcept {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

constexpr bool is_power_of_two(std::uint32_t value) noexcept {
    return value != 0 && (value & (value - 1u)) == 0;
}

constexpr std::uint64_t ceil_div(std::uint64_t value,
                                 std::uint64_t divisor) noexcept {
    return value / divisor + (value % divisor != 0 ? 1u : 0u);
}

} // namespace quantized_gemm_shape_detail

constexpr QuantizedGemmShapeTableResult make_quantized_gemm_shape_table(
    const QuantizedGemmModelShape& shape,
    const QuantizedGemmShapeLimits& limits) noexcept {
    using namespace quantized_gemm_shape_detail;

    if (limits.maximum_input_columns == 0 ||
        limits.maximum_input_columns ==
            std::numeric_limits<std::uint32_t>::max() ||
        limits.maximum_bundle_output_rows == 0 ||
        limits.maximum_bundle_output_rows ==
            std::numeric_limits<std::uint32_t>::max() ||
        limits.maximum_descriptor_count == 0 ||
        limits.maximum_descriptor_count ==
            std::numeric_limits<std::uint64_t>::max() ||
        limits.maximum_task_capacity == 0 ||
        limits.maximum_task_capacity ==
            std::numeric_limits<std::uint64_t>::max()) {
        return {.error = QuantizedGemmShapeError::InvalidLimits};
    }
    if (shape.model_layers == 0 || shape.hidden == 0 ||
        shape.query_heads == 0 || shape.key_value_heads == 0 ||
        shape.head_dimension == 0 ||
        shape.query_heads % shape.key_value_heads != 0 ||
        shape.recurrent_heads == 0 ||
        shape.recurrent_heads % 2u != 0 ||
        shape.state_dimension == 0 || shape.experts == 0 ||
        shape.per_token_active_experts == 0 ||
        shape.per_token_active_experts > shape.experts ||
        shape.expert_dimension == 0 ||
        (shape.weight_format != QuantizedGemmModelWeightFormat::AffineQ4 &&
         shape.weight_format != QuantizedGemmModelWeightFormat::AffineQ8) ||
        shape.group_size == 0 || shape.gated_delta_layers == 0 ||
        shape.attention_layers == 0 ||
        std::uint64_t{shape.gated_delta_layers} + shape.attention_layers !=
            shape.model_layers) {
        return {.error = QuantizedGemmShapeError::InvalidPlan};
    }
    if (!is_power_of_two(limits.output_tile_rows) ||
        limits.output_tile_rows > kQuantizedGemmMaximumTileRows) {
        return {.error = QuantizedGemmShapeError::InvalidTile};
    }

    QuantizedGemmShapeTable table{
        .hidden = shape.hidden,
        .query_heads = shape.query_heads,
        .key_value_heads = shape.key_value_heads,
        .head_dimension = shape.head_dimension,
        .recurrent_heads = shape.recurrent_heads,
        .state_dimension = shape.state_dimension,
        .experts = shape.experts,
        .per_token_active_experts =
            shape.per_token_active_experts,
        .expert_dimension = shape.expert_dimension,
        .group_size = shape.group_size,
        .gated_delta_layers = shape.gated_delta_layers,
        .attention_layers = shape.attention_layers,
        .output_tile_rows = limits.output_tile_rows,
    };

    const std::uint64_t hidden = table.hidden;
    const std::uint64_t query_heads = table.query_heads;
    const std::uint64_t key_value_heads = table.key_value_heads;
    const std::uint64_t head_dimension = table.head_dimension;
    const std::uint64_t recurrent_heads = table.recurrent_heads;
    const std::uint64_t state_dimension = table.state_dimension;
    const std::uint64_t experts = table.experts;
    const std::uint64_t expert_dimension = table.expert_dimension;

    std::uint64_t recurrent_values = 0;
    std::uint64_t gated_delta_qkv = 0;
    std::uint64_t attention_query = 0;
    std::uint64_t attention_vector = 0;
    std::uint64_t attention_key = 0;
    std::uint64_t attention_value = 0;
    if (!multiply(recurrent_heads, state_dimension, recurrent_values) ||
        !multiply(2u, recurrent_values, gated_delta_qkv) ||
        !multiply(query_heads, head_dimension, attention_vector) ||
        !multiply(2u, attention_vector, attention_query) ||
        !multiply(key_value_heads, head_dimension, attention_key)) {
        return {.error = QuantizedGemmShapeError::Overflow};
    }
    attention_value = attention_key;

    const std::uint64_t input_columns[] = {
        hidden,
        recurrent_values,
        hidden,
        attention_vector,
        hidden,
        hidden,
        hidden,
        expert_dimension,
        hidden,
        hidden,
        expert_dimension,
    };
    for (const std::uint64_t columns : input_columns) {
        if (columns > std::numeric_limits<std::uint32_t>::max()) {
            return {.error =
                        QuantizedGemmShapeError::EncoderRepresentability};
        }
        if (columns > limits.maximum_input_columns) {
            return {.error = QuantizedGemmShapeError::InputColumnLimit};
        }
        if (columns % table.group_size != 0) {
            return {.error = QuantizedGemmShapeError::InvalidPlan};
        }
    }

    std::size_t bundle_count = 0;
    std::size_t region_count = 0;
    std::uint64_t next_task = 0;
    QuantizedGemmShapeError construction_error =
        QuantizedGemmShapeError::None;

    const auto begin_bundle =
        [&](QuantizedGemmBundleKind kind, std::uint64_t columns) constexpr {
            QuantizedGemmBundle& bundle = table.bundles[bundle_count];
            bundle.kind = kind;
            bundle.region_begin = static_cast<std::uint32_t>(region_count);
            bundle.region_end = bundle.region_begin;
            bundle.input_columns = static_cast<std::uint32_t>(columns);
            bundle.output_rows = 0;
            bundle.task_begin = next_task;
            bundle.task_end = next_task;
        };

    const auto add_region =
        [&](QuantizedGemmRegionKind kind, std::uint64_t instances,
            std::uint64_t rows_per_instance) constexpr {
            if (construction_error != QuantizedGemmShapeError::None) {
                return;
            }
            std::uint64_t total_rows = 0;
            std::uint64_t region_tasks = 0;
            std::uint64_t next_rows = 0;
            std::uint64_t next_task_value = 0;
            if (!multiply(instances, rows_per_instance, total_rows) ||
                !multiply(instances,
                          ceil_div(rows_per_instance,
                                   limits.output_tile_rows),
                          region_tasks) ||
                !add(table.bundles[bundle_count].output_rows, total_rows,
                     next_rows) ||
                !add(next_task, region_tasks, next_task_value)) {
                construction_error = QuantizedGemmShapeError::Overflow;
                return;
            }
            if (instances > std::numeric_limits<std::uint32_t>::max() ||
                rows_per_instance >
                    std::numeric_limits<std::uint32_t>::max() ||
                total_rows > std::numeric_limits<std::uint32_t>::max() ||
                next_rows > std::numeric_limits<std::uint32_t>::max()) {
                construction_error =
                    QuantizedGemmShapeError::EncoderRepresentability;
                return;
            }
            if (next_rows > limits.maximum_bundle_output_rows) {
                construction_error =
                    QuantizedGemmShapeError::BundleOutputLimit;
                return;
            }

            QuantizedGemmBundle& bundle = table.bundles[bundle_count];
            QuantizedGemmRegion& region = table.regions[region_count];
            region.kind = kind;
            region.bundle = bundle.kind;
            region.bundle_row_begin = bundle.output_rows;
            region.bundle_row_end = static_cast<std::uint32_t>(next_rows);
            region.input_columns = bundle.input_columns;
            region.instance_count = static_cast<std::uint32_t>(instances);
            region.output_rows_per_instance =
                static_cast<std::uint32_t>(rows_per_instance);
            region.total_output_rows =
                static_cast<std::uint32_t>(total_rows);
            region.task_begin = next_task;
            region.task_end = next_task_value;
            bundle.output_rows = static_cast<std::uint32_t>(next_rows);
            bundle.region_end = static_cast<std::uint32_t>(region_count + 1);
            next_task = next_task_value;
            ++region_count;
        };

    const auto finish_bundle = [&]() constexpr {
        table.bundles[bundle_count].task_end = next_task;
        ++bundle_count;
    };

    begin_bundle(QuantizedGemmBundleKind::GatedDeltaInput, hidden);
    add_region(QuantizedGemmRegionKind::GatedDeltaQkv, 1,
               gated_delta_qkv);
    add_region(QuantizedGemmRegionKind::GatedDeltaZ, 1,
               recurrent_values);
    add_region(QuantizedGemmRegionKind::GatedDeltaB, 1,
               recurrent_heads);
    add_region(QuantizedGemmRegionKind::GatedDeltaA, 1,
               recurrent_heads);
    finish_bundle();

    begin_bundle(QuantizedGemmBundleKind::GatedDeltaOutput,
                 recurrent_values);
    add_region(QuantizedGemmRegionKind::GatedDeltaOutput, 1, hidden);
    finish_bundle();

    begin_bundle(QuantizedGemmBundleKind::AttentionInput, hidden);
    add_region(QuantizedGemmRegionKind::AttentionQuery, 1,
               attention_query);
    add_region(QuantizedGemmRegionKind::AttentionKey, 1, attention_key);
    add_region(QuantizedGemmRegionKind::AttentionValue, 1,
               attention_value);
    finish_bundle();

    begin_bundle(QuantizedGemmBundleKind::AttentionOutput,
                 attention_vector);
    add_region(QuantizedGemmRegionKind::AttentionOutput, 1, hidden);
    finish_bundle();

    begin_bundle(QuantizedGemmBundleKind::RouterQ8, hidden);
    add_region(QuantizedGemmRegionKind::RoutedRouter, 1, experts);
    add_region(QuantizedGemmRegionKind::SharedRouter, 1, 1);
    finish_bundle();

    begin_bundle(QuantizedGemmBundleKind::RoutedGate, hidden);
    add_region(QuantizedGemmRegionKind::RoutedGate, experts,
               expert_dimension);
    finish_bundle();

    begin_bundle(QuantizedGemmBundleKind::RoutedUp, hidden);
    add_region(QuantizedGemmRegionKind::RoutedUp, experts,
               expert_dimension);
    finish_bundle();

    begin_bundle(QuantizedGemmBundleKind::RoutedDown, expert_dimension);
    add_region(QuantizedGemmRegionKind::RoutedDown, experts, hidden);
    finish_bundle();

    begin_bundle(QuantizedGemmBundleKind::SharedGate, hidden);
    add_region(QuantizedGemmRegionKind::SharedGate, 1,
               expert_dimension);
    finish_bundle();

    begin_bundle(QuantizedGemmBundleKind::SharedUp, hidden);
    add_region(QuantizedGemmRegionKind::SharedUp, 1, expert_dimension);
    finish_bundle();

    begin_bundle(QuantizedGemmBundleKind::SharedDown, expert_dimension);
    add_region(QuantizedGemmRegionKind::SharedDown, 1, hidden);
    finish_bundle();

    if (construction_error != QuantizedGemmShapeError::None) {
        return {.error = construction_error};
    }
    if (bundle_count != kQuantizedGemmBundleCount ||
        region_count != kQuantizedGemmRegionCount) {
        return {.error = QuantizedGemmShapeError::EncoderRepresentability};
    }
    table.table_task_capacity = next_task;

    const auto bundle_tasks =
        [&](QuantizedGemmBundleKind kind) constexpr -> std::uint64_t {
        const QuantizedGemmBundle& bundle =
            table.bundles[static_cast<std::size_t>(kind)];
        return bundle.task_end - bundle.task_begin;
    };
    std::uint64_t routed_gate_up_tasks = 0;
    std::uint64_t routed_all_tasks = 0;
    std::uint64_t shared_gate_up_tasks = 0;
    std::uint64_t shared_projection_tasks = 0;
    std::uint64_t shared_and_router_tasks = 0;
    std::uint64_t moe_all_tasks = 0;
    std::uint64_t gated_delta_dense_tasks = 0;
    std::uint64_t attention_dense_tasks = 0;
    if (!add(bundle_tasks(QuantizedGemmBundleKind::RoutedGate),
             bundle_tasks(QuantizedGemmBundleKind::RoutedUp),
             routed_gate_up_tasks) ||
        !add(routed_gate_up_tasks,
             bundle_tasks(QuantizedGemmBundleKind::RoutedDown),
             routed_all_tasks) ||
        !add(bundle_tasks(QuantizedGemmBundleKind::SharedGate),
             bundle_tasks(QuantizedGemmBundleKind::SharedUp),
             shared_gate_up_tasks) ||
        !add(shared_gate_up_tasks,
             bundle_tasks(QuantizedGemmBundleKind::SharedDown),
             shared_projection_tasks) ||
        !add(bundle_tasks(QuantizedGemmBundleKind::RouterQ8),
             shared_projection_tasks, shared_and_router_tasks) ||
        !add(routed_all_tasks, shared_and_router_tasks, moe_all_tasks) ||
        !add(bundle_tasks(QuantizedGemmBundleKind::GatedDeltaInput),
             bundle_tasks(QuantizedGemmBundleKind::GatedDeltaOutput),
             gated_delta_dense_tasks) ||
        !add(bundle_tasks(QuantizedGemmBundleKind::AttentionInput),
             bundle_tasks(QuantizedGemmBundleKind::AttentionOutput),
             attention_dense_tasks) ||
        !add(gated_delta_dense_tasks, moe_all_tasks,
             table.gated_delta_layer_task_capacity) ||
        !add(attention_dense_tasks, moe_all_tasks,
             table.attention_layer_task_capacity)) {
        return {.error = QuantizedGemmShapeError::Overflow};
    }

    std::uint64_t gated_delta_model_tasks = 0;
    std::uint64_t attention_model_tasks = 0;
    if (!multiply(table.gated_delta_layers,
                  table.gated_delta_layer_task_capacity,
                  gated_delta_model_tasks) ||
        !multiply(table.attention_layers,
                  table.attention_layer_task_capacity,
                  attention_model_tasks) ||
        !add(gated_delta_model_tasks, attention_model_tasks,
             table.model_task_capacity)) {
        return {.error = QuantizedGemmShapeError::Overflow};
    }

    constexpr std::uint64_t gated_delta_dense_regions =
        static_cast<std::uint64_t>(
            QuantizedGemmRegionKind::AttentionQuery);
    constexpr std::uint64_t attention_dense_regions =
        static_cast<std::uint64_t>(
            QuantizedGemmRegionKind::RoutedRouter) -
        gated_delta_dense_regions;
    constexpr std::uint64_t mixture_regions =
        static_cast<std::uint64_t>(QuantizedGemmRegionKind::Count) -
        static_cast<std::uint64_t>(
            QuantizedGemmRegionKind::RoutedRouter);
    // A routed expert axis is inside one bound tensor; each binding still owns
    // exactly the weight/scales/biases descriptor triple.
    if (!multiply(mixture_regions,
                  kQuantizedTensorDescriptorsPerRegion,
                  table.mixture_descriptor_count_per_layer) ||
        !multiply(mixture_regions + gated_delta_dense_regions,
                  kQuantizedTensorDescriptorsPerRegion,
                  table.gated_delta_layer_descriptor_count) ||
        !multiply(mixture_regions + attention_dense_regions,
                  kQuantizedTensorDescriptorsPerRegion,
                  table.attention_layer_descriptor_count)) {
        return {.error = QuantizedGemmShapeError::Overflow};
    }
    std::uint64_t gated_delta_model_descriptors = 0;
    std::uint64_t attention_model_descriptors = 0;
    if (!multiply(table.gated_delta_layers,
                  table.gated_delta_layer_descriptor_count,
                  gated_delta_model_descriptors) ||
        !multiply(table.attention_layers,
                  table.attention_layer_descriptor_count,
                  attention_model_descriptors) ||
        !add(gated_delta_model_descriptors, attention_model_descriptors,
             table.model_descriptor_count)) {
        return {.error = QuantizedGemmShapeError::Overflow};
    }
    if (table.model_descriptor_count > limits.maximum_descriptor_count) {
        return {.error = QuantizedGemmShapeError::DescriptorLimit};
    }
    if (table.model_task_capacity > limits.maximum_task_capacity) {
        return {.error = QuantizedGemmShapeError::TaskLimit};
    }
    return {.error = QuantizedGemmShapeError::None, .table = table};
}

} // namespace tatara::runtime
