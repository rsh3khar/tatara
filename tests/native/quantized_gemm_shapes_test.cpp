#include "tatara/generated/model_plan.h"
#include "tatara/runtime/quantized_gemm_shapes.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <type_traits>

namespace {

using namespace tatara::model::qwen36;
using namespace tatara::runtime;
using tatara::model::qwen36::generated::kModelPlan;

constexpr QuantizedGemmModelWeightFormat
model_weight_format(WeightFormat format) {
    return format == WeightFormat::AffineQ4
               ? QuantizedGemmModelWeightFormat::AffineQ4
               : QuantizedGemmModelWeightFormat::Unknown;
}

template <std::size_t LayerCount>
constexpr QuantizedGemmModelShape
model_shape(const StaticModelPlan<LayerCount>& plan) {
    if constexpr (LayerCount >
                  std::numeric_limits<std::uint32_t>::max()) {
        return {};
    }
    QuantizedGemmModelShape shape{
        .hidden = plan.dimensions.hidden,
        .query_heads = plan.attention.query_heads,
        .key_value_heads = plan.attention.key_value_heads,
        .head_dimension = plan.attention.head_dimension,
        .recurrent_heads = plan.gated_delta.recurrent_heads,
        .state_dimension = plan.gated_delta.state_dimension,
        .experts = plan.mixture_of_experts.experts,
        .per_token_active_experts =
            plan.mixture_of_experts.active_experts,
        .expert_dimension = plan.mixture_of_experts.expert_dimension,
        .weight_format = model_weight_format(plan.weights.format),
        .group_size = plan.weights.group_size,
        .model_layers = static_cast<std::uint32_t>(LayerCount),
    };
    for (const LayerKind kind : plan.layers) {
        if (kind == LayerKind::GatedDelta) {
            ++shape.gated_delta_layers;
        } else if (kind == LayerKind::FullAttention) {
            ++shape.attention_layers;
        }
    }
    return shape;
}

constexpr StaticModelPlan<6> make_synthetic_plan() {
    StaticModelPlan<6> plan{};
    plan.dimensions = {.hidden = 1536, .vocabulary = 32000};
    plan.attention = {
        .query_heads = 12, .key_value_heads = 3, .head_dimension = 128};
    plan.gated_delta = {.recurrent_heads = 24, .state_dimension = 64};
    plan.mixture_of_experts = {
        .experts = 64, .active_experts = 4, .expert_dimension = 768};
    plan.weights = {.format = WeightFormat::AffineQ4, .group_size = 64};
    plan.layers = {
        LayerKind::GatedDelta,
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
        LayerKind::GatedDelta,
    };
    return plan;
}

constexpr auto kSyntheticPlan = make_synthetic_plan();
constexpr auto kCurrentShape = model_shape(kModelPlan);
constexpr auto kSyntheticShape = model_shape(kSyntheticPlan);
constexpr QuantizedGemmShapeLimits kShapeLimits{
    .output_tile_rows = 32,
    .maximum_input_columns = 4096,
    .maximum_bundle_output_rows = 524288,
    .maximum_descriptor_count = 1530,
    .maximum_task_capacity = 1004260,
};
constexpr auto kCurrentResult =
    make_quantized_gemm_shape_table(kCurrentShape, kShapeLimits);
constexpr auto kSyntheticResult =
    make_quantized_gemm_shape_table(kSyntheticShape, kShapeLimits);
constexpr auto kSyntheticQ8Result = [] {
    auto shape = kSyntheticShape;
    shape.weight_format = QuantizedGemmModelWeightFormat::AffineQ8;
    return make_quantized_gemm_shape_table(shape, kShapeLimits);
}();

constexpr std::uint64_t ceil_div(std::uint64_t value,
                                 std::uint64_t divisor) {
    return value / divisor + (value % divisor != 0 ? 1u : 0u);
}

constexpr bool regions_conserve(const QuantizedGemmShapeTable& table) {
    std::size_t next_region = 0;
    std::uint64_t next_task = 0;
    for (std::size_t bundle_index = 0;
         bundle_index < table.bundles.size(); ++bundle_index) {
        const QuantizedGemmBundle& bundle = table.bundles[bundle_index];
        if (static_cast<std::size_t>(bundle.kind) != bundle_index ||
            bundle.region_begin != next_region ||
            bundle.task_begin != next_task ||
            bundle.region_end <= bundle.region_begin) {
            return false;
        }
        std::uint32_t next_row = 0;
        for (std::size_t region_index = bundle.region_begin;
             region_index < bundle.region_end; ++region_index) {
            const QuantizedGemmRegion& region = table.regions[region_index];
            const std::uint64_t expected_tasks =
                std::uint64_t{region.instance_count} *
                ceil_div(region.output_rows_per_instance,
                         table.output_tile_rows);
            if (static_cast<std::size_t>(region.kind) != region_index ||
                region.bundle != bundle.kind ||
                region.bundle_row_begin != next_row ||
                region.bundle_row_end - region.bundle_row_begin !=
                    region.total_output_rows ||
                std::uint64_t{region.instance_count} *
                        region.output_rows_per_instance !=
                    region.total_output_rows ||
                region.input_columns != bundle.input_columns ||
                region.task_begin != next_task ||
                region.task_end - region.task_begin != expected_tasks) {
                return false;
            }
            next_row = region.bundle_row_end;
            next_task = region.task_end;
            ++next_region;
        }
        if (next_row != bundle.output_rows ||
            next_task != bundle.task_end) {
            return false;
        }
    }
    return next_region == table.regions.size() &&
           next_task == table.table_task_capacity;
}

static_assert(kCurrentShape.hidden == 2048);
static_assert(kCurrentShape.gated_delta_layers == 30);
static_assert(kCurrentShape.attention_layers == 10);
static_assert(kCurrentResult.error == QuantizedGemmShapeError::None);
static_assert(kSyntheticResult.error == QuantizedGemmShapeError::None);
static_assert(kSyntheticQ8Result.error == QuantizedGemmShapeError::None);
static_assert(regions_conserve(kCurrentResult.table));
static_assert(regions_conserve(kSyntheticResult.table));
static_assert(regions_conserve(kSyntheticQ8Result.table));
static_assert(std::is_standard_layout_v<QuantizedGemmModelShape>);
static_assert(std::is_trivially_copyable_v<QuantizedGemmModelShape>);
static_assert(std::is_standard_layout_v<QuantizedGemmShapeTable>);
static_assert(std::is_trivially_copyable_v<QuantizedGemmShapeTable>);

static_assert(kCurrentResult.table.hidden == kModelPlan.dimensions.hidden);
static_assert(kCurrentResult.table.gated_delta_layers == 30);
static_assert(kCurrentResult.table.attention_layers == 10);
static_assert(
    kCurrentResult
        .table
        .bundles[static_cast<std::size_t>(
            QuantizedGemmBundleKind::GatedDeltaInput)]
        .output_rows == 12352);
static_assert(
    kCurrentResult
        .table
        .bundles[static_cast<std::size_t>(
            QuantizedGemmBundleKind::AttentionInput)]
        .output_rows == 9216);
static_assert(
    kCurrentResult
        .table
        .bundles[static_cast<std::size_t>(
            QuantizedGemmBundleKind::RouterQ8)]
        .output_rows == 257);
static_assert(kCurrentResult.table.table_task_capacity == 25483);
static_assert(kCurrentResult.table.model_task_capacity == 1004260);
static_assert(kCurrentResult.table.mixture_descriptor_count_per_layer == 24);
static_assert(kCurrentResult.table.gated_delta_layer_descriptor_count == 39);
static_assert(kCurrentResult.table.attention_layer_descriptor_count == 36);
static_assert(kCurrentResult.table.model_descriptor_count == 1530);

static_assert(kSyntheticResult.table.hidden == 1536);
static_assert(kSyntheticResult.table.query_heads == 12);
static_assert(kSyntheticResult.table.key_value_heads == 3);
static_assert(kSyntheticResult.table.recurrent_heads == 24);
static_assert(kSyntheticResult.table.state_dimension == 64);
static_assert(kSyntheticResult.table.experts == 64);
static_assert(kSyntheticResult.table.per_token_active_experts == 4);
static_assert(kSyntheticResult.table.expert_dimension == 768);
static_assert(kSyntheticResult.table.gated_delta_layers == 4);
static_assert(kSyntheticResult.table.attention_layers == 2);
static_assert(kSyntheticResult.table.table_task_capacity == 6605);
static_assert(kSyntheticResult.table.model_task_capacity == 38570);
static_assert(kSyntheticResult.table.model_descriptor_count == 228);
static_assert(kSyntheticQ8Result.table.model_task_capacity ==
              kSyntheticResult.table.model_task_capacity);
static_assert(model_weight_format(WeightFormat::AffineQ4) ==
              QuantizedGemmModelWeightFormat::AffineQ4);
static_assert(model_weight_format(static_cast<WeightFormat>(99)) ==
              QuantizedGemmModelWeightFormat::Unknown);

constexpr bool invalid_and_bounded_plans_are_typed() {
    auto shape = kSyntheticShape;
    shape.hidden = 0;
    if (make_quantized_gemm_shape_table(shape, kShapeLimits).error !=
        QuantizedGemmShapeError::InvalidPlan) {
        return false;
    }

    shape = kSyntheticShape;
    shape.hidden = 1537;
    if (make_quantized_gemm_shape_table(shape, kShapeLimits).error !=
        QuantizedGemmShapeError::InvalidPlan) {
        return false;
    }

    auto plan = kSyntheticPlan;
    plan.layers.fill(LayerKind::GatedDelta);
    shape = model_shape(plan);
    if (make_quantized_gemm_shape_table(shape, kShapeLimits).error !=
        QuantizedGemmShapeError::InvalidPlan) {
        return false;
    }

    shape = kSyntheticShape;
    shape.query_heads =
        std::numeric_limits<std::uint32_t>::max();
    shape.key_value_heads = 1;
    shape.head_dimension =
        std::numeric_limits<std::uint32_t>::max();
    if (make_quantized_gemm_shape_table(shape, kShapeLimits).error !=
        QuantizedGemmShapeError::Overflow) {
        return false;
    }

    shape = kSyntheticShape;
    shape.experts = 1U << 20U;
    shape.expert_dimension = 4096;
    auto large_limits = kShapeLimits;
    large_limits.maximum_bundle_output_rows = 2000000;
    if (make_quantized_gemm_shape_table(shape, large_limits).error !=
        QuantizedGemmShapeError::EncoderRepresentability) {
        return false;
    }

    shape = kSyntheticShape;
    shape.weight_format = QuantizedGemmModelWeightFormat::Unknown;
    if (make_quantized_gemm_shape_table(shape, kShapeLimits).error !=
        QuantizedGemmShapeError::InvalidPlan) {
        return false;
    }

    auto limits = kShapeLimits;
    limits.output_tile_rows = 0;
    if (make_quantized_gemm_shape_table(kSyntheticShape, limits).error !=
        QuantizedGemmShapeError::InvalidTile) {
        return false;
    }
    limits = kShapeLimits;
    limits.maximum_input_columns = 1024;
    if (make_quantized_gemm_shape_table(kSyntheticShape, limits).error !=
        QuantizedGemmShapeError::InputColumnLimit) {
        return false;
    }
    limits = kShapeLimits;
    limits.maximum_bundle_output_rows = 4000;
    if (make_quantized_gemm_shape_table(kSyntheticShape, limits).error !=
        QuantizedGemmShapeError::BundleOutputLimit) {
        return false;
    }
    limits = kShapeLimits;
    limits.maximum_descriptor_count = 227;
    if (make_quantized_gemm_shape_table(kSyntheticShape, limits).error !=
        QuantizedGemmShapeError::DescriptorLimit) {
        return false;
    }
    limits = kShapeLimits;
    limits.maximum_task_capacity = 38569;
    if (make_quantized_gemm_shape_table(kSyntheticShape, limits).error !=
        QuantizedGemmShapeError::TaskLimit) {
        return false;
    }
    if (make_quantized_gemm_shape_table(kSyntheticShape, {}).error !=
        QuantizedGemmShapeError::InvalidLimits) {
        return false;
    }
    limits = kShapeLimits;
    limits.maximum_input_columns =
        std::numeric_limits<std::uint32_t>::max();
    if (make_quantized_gemm_shape_table(kSyntheticShape, limits).error !=
        QuantizedGemmShapeError::InvalidLimits) {
        return false;
    }
    limits = kShapeLimits;
    limits.maximum_bundle_output_rows =
        std::numeric_limits<std::uint32_t>::max();
    if (make_quantized_gemm_shape_table(kSyntheticShape, limits).error !=
        QuantizedGemmShapeError::InvalidLimits) {
        return false;
    }
    limits = kShapeLimits;
    limits.maximum_descriptor_count =
        std::numeric_limits<std::uint64_t>::max();
    if (make_quantized_gemm_shape_table(kSyntheticShape, limits).error !=
        QuantizedGemmShapeError::InvalidLimits) {
        return false;
    }
    limits = kShapeLimits;
    limits.maximum_task_capacity =
        std::numeric_limits<std::uint64_t>::max();
    if (make_quantized_gemm_shape_table(kSyntheticShape, limits).error !=
        QuantizedGemmShapeError::InvalidLimits) {
        return false;
    }
    return true;
}

static_assert(invalid_and_bounded_plans_are_typed());

constexpr bool second_plan_changes_every_requested_axis() {
    return kCurrentResult.table.hidden != kSyntheticResult.table.hidden &&
           kCurrentResult.table.query_heads !=
               kSyntheticResult.table.query_heads &&
           kCurrentResult.table.key_value_heads !=
               kSyntheticResult.table.key_value_heads &&
           kCurrentResult.table.state_dimension !=
               kSyntheticResult.table.state_dimension &&
           kCurrentResult.table.experts != kSyntheticResult.table.experts &&
           kCurrentResult.table.per_token_active_experts !=
               kSyntheticResult.table.per_token_active_experts &&
           kCurrentResult.table.expert_dimension !=
               kSyntheticResult.table.expert_dimension &&
           kCurrentResult.table.gated_delta_layers +
                   kCurrentResult.table.attention_layers !=
               kSyntheticResult.table.gated_delta_layers +
                   kSyntheticResult.table.attention_layers;
}

static_assert(second_plan_changes_every_requested_axis());

} // namespace

int main() {
    if (!regions_conserve(kCurrentResult.table) ||
        !regions_conserve(kSyntheticResult.table) ||
        !regions_conserve(kSyntheticQ8Result.table) ||
        !invalid_and_bounded_plans_are_typed() ||
        !second_plan_changes_every_requested_axis()) {
        return 1;
    }
    std::printf("quantized_gemm_shapes: PASS\n");
    return 0;
}
