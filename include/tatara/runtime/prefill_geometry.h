#pragma once

#include "tatara/model/qwen36_plan.h"
#include "tatara/runtime/decode_geometry.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>

namespace tatara::runtime {

inline constexpr std::uint32_t kPrefillMaximumContext = 262144;
inline constexpr std::uint32_t kPrefillMaximumBlockRows = 2048;
inline constexpr std::uint32_t kPrefillQueryTileMultiple = 16;
inline constexpr std::uint32_t kPrefillExactPositionBatch = 4;
inline constexpr std::uint32_t kPrefillExpertSlotBits = 4;
inline constexpr std::uint32_t kPrefillMaximumExactRowsPerThreadgroup = 32;

enum class PrefillSchedule : std::uint8_t {
    ChunkMajor,
    LayerMajor,
};

struct PrefillPolicy {
    PrefillSchedule schedule{PrefillSchedule::ChunkMajor};
    std::uint32_t context_capacity{0};
    std::uint32_t maximum_block_rows{0};
    std::uint32_t first_chunk_rows{0};
    std::uint32_t query_tile_rows{0};
    std::uint32_t attention_partition{kAttentionPartition};
    std::uint32_t exact_rows_per_threadgroup{16};
    bool gdn_gate_hoist{false};
};

enum class PrefillGeometryError : std::uint8_t {
    None,
    InvalidPlan,
    InvalidPolicy,
    Overflow,
};

enum class PrefillRequestError : std::uint8_t {
    None,
    EmptyPrefix,
    BlockOutOfRange,
    ContextOutOfRange,
    ContextOverflow,
    TokenOutOfRange,
};

struct PrefillRequestValidation {
    PrefillRequestError error{PrefillRequestError::EmptyPrefix};
    std::uint32_t next_context{0};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillRequestError::None;
    }
};

// Allocation extents for one exact block-prefill step. Equal-size logical
// buffers remain separate fields: collapsing them into a subtotal would make
// an omitted live allocation invisible to the capacity model.
struct PrefillGeometry {
    PrefillSchedule schedule{PrefillSchedule::ChunkMajor};
    std::uint32_t context_capacity{0};
    std::uint32_t maximum_block_rows{0};
    std::uint32_t first_chunk_rows{0};
    std::uint32_t query_tile_rows{0};
    std::uint32_t attention_partition{0};
    std::uint32_t exact_rows_per_threadgroup{0};
    bool gdn_gate_hoist{false};

    std::uint32_t hidden{0};
    std::uint32_t vocabulary{0};
    std::uint32_t query_heads{0};
    std::uint32_t key_value_heads{0};
    std::uint32_t attention_head_dimension{0};
    std::uint32_t recurrent_heads{0};
    std::uint32_t state_dimension{0};
    std::uint32_t experts{0};
    std::uint32_t active_experts{0};
    std::uint32_t expert_dimension{0};
    std::uint32_t gdn_projection_rows{0};
    std::uint32_t gdn_qk_values{0};
    std::uint32_t gdn_value_values{0};
    std::uint32_t attention_projection_rows{0};
    std::uint32_t attention_vector_values{0};

    std::uint32_t gated_delta_layers{0};
    std::uint32_t attention_layers{0};
    std::uint32_t attention_partitions{0};

    // Persistent state is not scratch. One slot owns two GDN ping-pong
    // planes and full-capacity K/V, while a cache snapshot canonicalizes one
    // live GDN plane and packs only populated K/V positions.
    std::uint64_t gdn_live_state_bytes_per_layer{0};
    std::uint64_t gdn_slot_state_bytes_per_layer{0};
    std::uint64_t attention_state_bytes_per_position_per_layer{0};
    std::uint64_t attention_slot_state_bytes_per_layer{0};
    std::uint64_t slot_state_bytes{0};
    std::uint64_t hidden_slab_bytes{0};
    std::uint64_t token_bytes{0};

    std::uint64_t block_hidden_bytes{0};
    std::uint64_t gdn_projection_bytes{0};
    std::uint64_t gdn_qk_bytes{0};
    std::uint64_t gdn_value_bytes{0};
    std::uint64_t gdn_parameter_bytes{0};
    std::uint64_t attention_projection_bytes{0};
    std::uint64_t attention_vector_bytes{0};

    std::uint64_t moe_logits_bytes{0};
    std::uint64_t moe_id_bytes{0};
    std::uint64_t moe_coefficient_bytes{0};
    std::uint64_t moe_shared_coefficient_bytes{0};
    std::uint64_t moe_count_bytes{0};
    std::uint64_t moe_list_bytes{0};
    std::uint64_t moe_active_bytes{0};
    std::uint64_t moe_indirect_argument_bytes{0};
    std::uint64_t moe_hidden_bytes{0};
    std::uint64_t moe_partial_bytes{0};

    std::uint64_t attention_partial_bytes{0};
    // The opt-in staged attention path reuses attention_partials. Keeping its
    // full-capacity extent explicit makes that alias a checked geometry fact.
    std::uint64_t attention_staged_score_bytes{0};
    std::uint64_t reusable_scratch_bytes{0};
    std::uint64_t steady_prefill_bytes{0};
};

constexpr bool native_routed_shared_task_capacity_supported(
    const PrefillGeometry& geometry,
    std::uint32_t task_tile_rows,
    std::uint32_t task_capacity) noexcept {
    if (geometry.maximum_block_rows == 0 ||
        geometry.maximum_block_rows > kPrefillMaximumBlockRows ||
        geometry.experts == 0 || geometry.active_experts == 0 ||
        task_tile_rows == 0 || task_capacity == 0) {
        return false;
    }
    const std::uint64_t maximum_rows =
        geometry.maximum_block_rows;
    const std::uint64_t routed_routes =
        maximum_rows * geometry.active_experts;
    const std::uint64_t routed_tasks =
        (routed_routes + task_tile_rows - 1U) /
            task_tile_rows +
        geometry.experts;
    const std::uint64_t shared_tasks =
        (maximum_rows + task_tile_rows - 1U) /
        task_tile_rows;
    return routed_tasks + shared_tasks <= task_capacity;
}

struct PrefillGeometryResult {
    PrefillGeometryError error{PrefillGeometryError::InvalidPlan};
    PrefillGeometry geometry;

    explicit constexpr operator bool() const noexcept {
        return error == PrefillGeometryError::None;
    }
};

enum class PrefixStateSnapshotError : std::uint8_t {
    None,
    InvalidGeometry,
    PositionOutOfRange,
    Overflow,
};

struct PrefixStateSnapshotGeometry {
    PrefixStateSnapshotError error{PrefixStateSnapshotError::InvalidGeometry};
    std::uint32_t positions{0};
    std::uint64_t gated_delta_bytes{0};
    std::uint64_t attention_bytes{0};
    std::uint64_t total_bytes{0};

    explicit constexpr operator bool() const noexcept {
        return error == PrefixStateSnapshotError::None;
    }
};

namespace prefill_geometry_detail {

constexpr bool add(std::uint64_t left, std::uint64_t right, std::uint64_t& out) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    out = left + right;
    return true;
}

constexpr bool multiply(std::uint64_t left, std::uint64_t right, std::uint64_t& out) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    out = left * right;
    return true;
}

constexpr bool accumulate(std::uint64_t value, std::uint64_t& total) {
    return add(total, value, total);
}

constexpr bool scaled(std::uint64_t count, std::uint64_t width, std::uint64_t element_bytes,
                      std::uint64_t& out) {
    std::uint64_t elements = 0;
    return multiply(count, width, elements) && multiply(elements, element_bytes, out);
}

} // namespace prefill_geometry_detail

// Canonical cache payload: one live conv+recurrent plane for every GDN layer,
// followed by packed K/V prefix positions for every attention layer. Restore
// installs GDN bytes into the canonical input half and resets its phase;
// keeping both ping-pong halves would waste cache capacity without preserving
// any additional model state.
constexpr PrefixStateSnapshotGeometry
make_prefix_state_snapshot_geometry(const PrefillGeometry& geometry,
                                    std::uint32_t positions) noexcept {
    using namespace prefill_geometry_detail;
    if (geometry.context_capacity == 0 || geometry.gated_delta_layers == 0 ||
        geometry.attention_layers == 0 || geometry.gdn_live_state_bytes_per_layer == 0 ||
        geometry.attention_state_bytes_per_position_per_layer == 0) {
        return {.error = PrefixStateSnapshotError::InvalidGeometry};
    }
    if (positions == 0 || positions > geometry.context_capacity) {
        return {.error = PrefixStateSnapshotError::PositionOutOfRange};
    }
    PrefixStateSnapshotGeometry snapshot{
        .error = PrefixStateSnapshotError::None,
        .positions = positions,
    };
    std::uint64_t attention_per_layer = 0;
    if (!multiply(geometry.gated_delta_layers, geometry.gdn_live_state_bytes_per_layer,
                  snapshot.gated_delta_bytes) ||
        !multiply(positions, geometry.attention_state_bytes_per_position_per_layer,
                  attention_per_layer) ||
        !multiply(geometry.attention_layers, attention_per_layer, snapshot.attention_bytes) ||
        !add(snapshot.gated_delta_bytes, snapshot.attention_bytes, snapshot.total_bytes)) {
        return {.error = PrefixStateSnapshotError::Overflow};
    }
    return snapshot;
}

constexpr PrefillRequestValidation validate_prefill_request(const PrefillPolicy& policy,
                                                            std::uint32_t live_context,
                                                            std::uint32_t context_base,
                                                            std::span<const std::uint32_t> tokens,
                                                            std::uint32_t vocabulary) noexcept {
    if (tokens.empty()) {
        return {.error = PrefillRequestError::EmptyPrefix};
    }
    if (tokens.size() > policy.maximum_block_rows) {
        return {.error = PrefillRequestError::BlockOutOfRange};
    }
    if (context_base != live_context || context_base > policy.context_capacity) {
        return {.error = PrefillRequestError::ContextOutOfRange};
    }
    if (tokens.size() > policy.context_capacity - context_base) {
        return {.error = PrefillRequestError::ContextOverflow};
    }
    for (const std::uint32_t token : tokens) {
        if (token >= vocabulary) {
            return {.error = PrefillRequestError::TokenOutOfRange};
        }
    }
    return {
        .error = PrefillRequestError::None,
        .next_context = context_base + static_cast<std::uint32_t>(tokens.size()),
    };
}

constexpr PrefillRequestValidation validate_prefill_prefix(const PrefillPolicy& policy,
                                                           std::uint32_t live_context,
                                                           std::uint32_t context_base,
                                                           std::span<const std::uint32_t> tokens,
                                                           std::uint32_t vocabulary) noexcept {
    if (tokens.empty()) {
        return {.error = PrefillRequestError::EmptyPrefix};
    }
    if (context_base != live_context || context_base > policy.context_capacity) {
        return {.error = PrefillRequestError::ContextOutOfRange};
    }
    if (tokens.size() > policy.context_capacity - context_base) {
        return {.error = PrefillRequestError::ContextOverflow};
    }
    for (const std::uint32_t token : tokens) {
        if (token >= vocabulary) {
            return {.error = PrefillRequestError::TokenOutOfRange};
        }
    }
    return {
        .error = PrefillRequestError::None,
        .next_context = context_base + static_cast<std::uint32_t>(tokens.size()),
    };
}

template <std::size_t LayerCount>
constexpr PrefillGeometryResult
make_prefill_geometry(const model::qwen36::StaticModelPlan<LayerCount>& plan,
                      const PrefillPolicy& policy) {
    using namespace prefill_geometry_detail;

    const std::uint64_t hidden = plan.dimensions.hidden;
    const std::uint64_t query_heads = plan.attention.query_heads;
    const std::uint64_t key_value_heads = plan.attention.key_value_heads;
    const std::uint64_t head_dimension = plan.attention.head_dimension;
    const std::uint64_t recurrent_heads = plan.gated_delta.recurrent_heads;
    const std::uint64_t state_dimension = plan.gated_delta.state_dimension;
    const std::uint64_t experts = plan.mixture_of_experts.experts;
    const std::uint64_t active_experts = plan.mixture_of_experts.active_experts;
    const std::uint64_t expert_dimension = plan.mixture_of_experts.expert_dimension;

    // These are the exact C1 kernel bounds, not general mathematical
    // requirements. A plan outside them is rejected before source generation
    // or command encoding rather than discovered by Metal.
    if (LayerCount == 0 || hidden == 0 || plan.dimensions.vocabulary == 0 ||
        hidden % kRmsValuesPerThread != 0 || hidden / kRmsValuesPerThread > 1024 ||
        query_heads == 0 || key_value_heads == 0 || query_heads % key_value_heads != 0 ||
        query_heads / key_value_heads != 8 || head_dimension != kAttentionPartition ||
        recurrent_heads == 0 || recurrent_heads % 2 != 0 || state_dimension != 128 || experts < 2 ||
        experts > 1024 || (experts & (experts - 1)) != 0 || active_experts == 0 ||
        active_experts >= experts || active_experts >= (1u << kPrefillExpertSlotBits) ||
        expert_dimension == 0 || plan.weights.format != model::qwen36::WeightFormat::AffineQ4 ||
        plan.weights.group_size != 64 || hidden % 512u != 0 || expert_dimension % 512u != 0 ||
        hidden % plan.weights.group_size != 0 || expert_dimension % plan.weights.group_size != 0) {
        return {.error = PrefillGeometryError::InvalidPlan};
    }

    if ((policy.schedule != PrefillSchedule::ChunkMajor &&
         policy.schedule != PrefillSchedule::LayerMajor) ||
        policy.context_capacity == 0 || policy.context_capacity > kPrefillMaximumContext ||
        policy.maximum_block_rows == 0 || policy.maximum_block_rows > kPrefillMaximumBlockRows ||
        policy.first_chunk_rows == 0 || policy.first_chunk_rows > policy.maximum_block_rows ||
        policy.query_tile_rows == 0 || policy.query_tile_rows > policy.maximum_block_rows ||
        policy.query_tile_rows % kPrefillQueryTileMultiple != 0 ||
        policy.attention_partition != kAttentionPartition ||
        policy.exact_rows_per_threadgroup == 0 ||
        policy.exact_rows_per_threadgroup > kPrefillMaximumExactRowsPerThreadgroup) {
        return {.error = PrefillGeometryError::InvalidPolicy};
    }

    PrefillGeometry geometry;
    geometry.schedule = policy.schedule;
    geometry.context_capacity = policy.context_capacity;
    geometry.maximum_block_rows = policy.maximum_block_rows;
    geometry.first_chunk_rows = policy.first_chunk_rows;
    geometry.query_tile_rows = policy.query_tile_rows;
    geometry.attention_partition = policy.attention_partition;
    geometry.exact_rows_per_threadgroup = policy.exact_rows_per_threadgroup;
    geometry.gdn_gate_hoist = policy.gdn_gate_hoist;
    geometry.hidden = static_cast<std::uint32_t>(hidden);
    geometry.vocabulary = plan.dimensions.vocabulary;
    geometry.query_heads = static_cast<std::uint32_t>(query_heads);
    geometry.key_value_heads = static_cast<std::uint32_t>(key_value_heads);
    geometry.attention_head_dimension = static_cast<std::uint32_t>(head_dimension);
    geometry.recurrent_heads = static_cast<std::uint32_t>(recurrent_heads);
    geometry.state_dimension = static_cast<std::uint32_t>(state_dimension);
    geometry.experts = static_cast<std::uint32_t>(experts);
    geometry.active_experts = static_cast<std::uint32_t>(active_experts);
    geometry.expert_dimension = static_cast<std::uint32_t>(expert_dimension);
    for (const model::qwen36::LayerKind kind : plan.layers) {
        if (kind == model::qwen36::LayerKind::GatedDelta) {
            ++geometry.gated_delta_layers;
        } else if (kind == model::qwen36::LayerKind::FullAttention) {
            ++geometry.attention_layers;
        } else {
            return {.error = PrefillGeometryError::InvalidPlan};
        }
    }
    if (geometry.gated_delta_layers == 0 || geometry.attention_layers == 0) {
        return {.error = PrefillGeometryError::InvalidPlan};
    }

    const std::uint64_t block = policy.maximum_block_rows;
    const std::uint64_t capacity = policy.context_capacity;
    const std::uint64_t query_tile = policy.query_tile_rows;
    geometry.attention_partitions =
        (policy.context_capacity + policy.attention_partition - 1u) / policy.attention_partition;

    const std::uint64_t gdn_key_heads = recurrent_heads / 2u;
    const std::uint64_t gdn_qk_rows = 2u * gdn_key_heads * state_dimension;
    const std::uint64_t gdn_value_rows = recurrent_heads * state_dimension;
    const std::uint64_t gdn_conv_rows = gdn_qk_rows + gdn_value_rows;
    const std::uint64_t gdn_projection_rows = gdn_conv_rows + gdn_value_rows + 2u * recurrent_heads;
    const std::uint64_t attention_projection_rows =
        query_heads * 2u * head_dimension + 2u * key_value_heads * head_dimension;
    const std::uint64_t attention_vector_rows = query_heads * head_dimension;
    const std::uint64_t expert_slots = active_experts + 1u;
    const std::uint64_t router_rows = experts + 1u;
    geometry.gdn_projection_rows = static_cast<std::uint32_t>(gdn_projection_rows);
    geometry.gdn_qk_values = static_cast<std::uint32_t>(gdn_qk_rows);
    geometry.gdn_value_values = static_cast<std::uint32_t>(gdn_value_rows);
    geometry.attention_projection_rows = static_cast<std::uint32_t>(attention_projection_rows);
    geometry.attention_vector_values = static_cast<std::uint32_t>(attention_vector_rows);

    std::uint64_t attention_staged_score_row_bytes = 0;
    if (!scaled(query_heads, query_tile, kF32Bytes,
                attention_staged_score_row_bytes) ||
        !multiply(attention_staged_score_row_bytes, capacity,
                  geometry.attention_staged_score_bytes) ||
        (policy.schedule == PrefillSchedule::LayerMajor &&
         !scaled(capacity, hidden, kBf16Bytes, geometry.hidden_slab_bytes)) ||
        !scaled(capacity, sizeof(std::uint32_t), 1, geometry.token_bytes) ||
        !scaled(block, hidden, kBf16Bytes, geometry.block_hidden_bytes) ||
        !scaled(block, gdn_projection_rows, kBf16Bytes, geometry.gdn_projection_bytes) ||
        !scaled(block, gdn_qk_rows, kBf16Bytes, geometry.gdn_qk_bytes) ||
        !scaled(block, gdn_value_rows, kBf16Bytes, geometry.gdn_value_bytes) ||
        (policy.gdn_gate_hoist &&
         !scaled(block, recurrent_heads, kF32Bytes, geometry.gdn_parameter_bytes)) ||
        !scaled(block, attention_projection_rows, kBf16Bytes,
                geometry.attention_projection_bytes) ||
        !scaled(block, attention_vector_rows, kBf16Bytes, geometry.attention_vector_bytes) ||
        !scaled(block, router_rows, kF32Bytes, geometry.moe_logits_bytes) ||
        !scaled(block, active_experts, sizeof(std::uint32_t), geometry.moe_id_bytes) ||
        !scaled(block, active_experts, kF32Bytes, geometry.moe_coefficient_bytes) ||
        !scaled(block, 1, kF32Bytes, geometry.moe_shared_coefficient_bytes) ||
        !scaled(router_rows, sizeof(std::uint32_t), 1, geometry.moe_count_bytes) ||
        !scaled(router_rows * block, sizeof(std::uint32_t), 1, geometry.moe_list_bytes) ||
        !scaled(router_rows, sizeof(std::uint32_t), 1, geometry.moe_active_bytes) ||
        !scaled(2u * 3u, sizeof(std::uint32_t), 1, geometry.moe_indirect_argument_bytes) ||
        !scaled(block * expert_slots, expert_dimension, kBf16Bytes, geometry.moe_hidden_bytes) ||
        !scaled(block * expert_slots, hidden, kF32Bytes, geometry.moe_partial_bytes) ||
        !scaled(query_heads * query_tile * geometry.attention_partitions, head_dimension + 2u,
                kF32Bytes, geometry.attention_partial_bytes)) {
        return {.error = PrefillGeometryError::Overflow};
    }
    if (geometry.attention_staged_score_bytes >
        geometry.attention_partial_bytes) {
        return {.error = PrefillGeometryError::Overflow};
    }

    std::uint64_t conv_state_per_layer = 0;
    std::uint64_t recurrent_state_per_layer = 0;
    std::uint64_t attention_cache_per_plane = 0;
    if (!scaled(kConvShiftSteps, gdn_conv_rows, kBf16Bytes, conv_state_per_layer) ||
        !scaled(recurrent_heads * state_dimension, state_dimension, kF32Bytes,
                recurrent_state_per_layer) ||
        !scaled(key_value_heads * capacity, head_dimension, kBf16Bytes,
                attention_cache_per_plane)) {
        return {.error = PrefillGeometryError::Overflow};
    }
    std::uint64_t gdn_live_state_per_layer = 0;
    std::uint64_t gdn_slot_state_per_layer = 0;
    std::uint64_t attention_slot_state_per_layer = 0;
    std::uint64_t all_gdn_state = 0;
    std::uint64_t all_attention_state = 0;
    if (!add(conv_state_per_layer, recurrent_state_per_layer, gdn_live_state_per_layer) ||
        !multiply(2u, gdn_live_state_per_layer, gdn_slot_state_per_layer) ||
        !scaled(2u * key_value_heads, head_dimension, kBf16Bytes,
                geometry.attention_state_bytes_per_position_per_layer) ||
        !multiply(2u, attention_cache_per_plane, attention_slot_state_per_layer) ||
        !multiply(geometry.gated_delta_layers, gdn_slot_state_per_layer, all_gdn_state) ||
        !multiply(geometry.attention_layers, attention_slot_state_per_layer, all_attention_state) ||
        !add(all_gdn_state, all_attention_state, geometry.slot_state_bytes)) {
        return {.error = PrefillGeometryError::Overflow};
    }
    geometry.gdn_live_state_bytes_per_layer = gdn_live_state_per_layer;
    geometry.gdn_slot_state_bytes_per_layer = gdn_slot_state_per_layer;
    geometry.attention_slot_state_bytes_per_layer = attention_slot_state_per_layer;

    // Four hidden-width block buffers; one Q/K plus four value-width GDN
    // buffers; two f32 GDN parameter planes; three attention vectors.
    std::uint64_t four_hidden = 0;
    std::uint64_t four_gdn_values = 0;
    std::uint64_t two_gdn_parameters = 0;
    std::uint64_t three_attention_vectors = 0;
    if (!multiply(4u, geometry.block_hidden_bytes, four_hidden) ||
        !multiply(4u, geometry.gdn_value_bytes, four_gdn_values) ||
        !multiply(2u, geometry.gdn_parameter_bytes, two_gdn_parameters) ||
        !multiply(3u, geometry.attention_vector_bytes, three_attention_vectors)) {
        return {.error = PrefillGeometryError::Overflow};
    }

    for (const std::uint64_t bytes :
         {geometry.token_bytes, four_hidden, geometry.gdn_projection_bytes, geometry.gdn_qk_bytes,
          four_gdn_values, two_gdn_parameters, geometry.attention_projection_bytes,
          three_attention_vectors, geometry.moe_logits_bytes, geometry.moe_id_bytes,
          geometry.moe_coefficient_bytes, geometry.moe_shared_coefficient_bytes,
          geometry.moe_count_bytes, geometry.moe_list_bytes, geometry.moe_active_bytes,
          geometry.moe_indirect_argument_bytes, geometry.moe_hidden_bytes,
          geometry.moe_partial_bytes, geometry.attention_partial_bytes}) {
        if (!accumulate(bytes, geometry.reusable_scratch_bytes)) {
            return {.error = PrefillGeometryError::Overflow};
        }
    }
    if (!add(geometry.hidden_slab_bytes, geometry.reusable_scratch_bytes,
             geometry.steady_prefill_bytes)) {
        return {.error = PrefillGeometryError::Overflow};
    }

    return {.error = PrefillGeometryError::None, .geometry = geometry};
}

} // namespace tatara::runtime
