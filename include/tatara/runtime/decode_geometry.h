#pragma once

#include "tatara/model/qwen36_plan.h"

#include <cstdint>

namespace tatara::runtime {

// Threadgroup count and threadgroup width of a one-dimensional dispatch.
struct DispatchShape {
    std::uint32_t groups;
    std::uint32_t threads;
};

// The gated-delta recurrence walks (lane, value dimension, recurrent head):
// one simdgroup covers a head's state row, and the threadgroup packs several
// value dimensions so the group stays one head wide.
struct RecurrenceDispatch {
    std::uint32_t dimension_groups;
    std::uint32_t head_groups;
    std::uint32_t lane_threads;
    std::uint32_t dimension_threads;
};

// The long-context attention split. Both stages carry a partition axis that
// varies with the live context, so only the fixed axes live here: the score
// stage runs one head cohort per threadgroup over a partition of keys, and
// the value stage runs one KV group per threadgroup with a thread per
// (lane, dimension pair, head).
struct AttentionSplitDispatch {
    std::uint32_t score_groups;
    std::uint32_t score_threads;
    std::uint32_t score_cohort_threads;
    std::uint32_t value_groups;
    std::uint32_t value_lane_threads;
    std::uint32_t value_dimension_threads;
    std::uint32_t value_head_threads;
};

// Every dispatch argument the sealed token walk issues that is fixed by the
// model package rather than by the live context. Naming follows the pipeline
// it drives; the walk reads these instead of restating the shapes inline.
struct DecodeDispatch {
    DispatchShape embed;
    // rms_only, at both the layer input and the final norm, and residual_rms.
    DispatchShape rms;
    DispatchShape gdn_project;
    DispatchShape gdn_prepare;
    RecurrenceDispatch gdn_recurrence;
    DispatchShape gdn_gate_norm;
    // Shared by the gated-delta and attention output projections.
    DispatchShape out_projection;
    DispatchShape attn_project;
    DispatchShape attn_qk_rope;
    // One threadgroup per query head, one thread per head dimension: the
    // short-context attention_decode and the long-context combine.
    DispatchShape attention_head;
    AttentionSplitDispatch attention_split;
    DispatchShape router;
    DispatchShape router_select;
    DispatchShape grouped_upgate;
    DispatchShape grouped_down_res;
    DispatchShape lmhead;
    DispatchShape argmax_stage1;
    // The lmhead row count, which is also the argmax element count.
    std::uint32_t vocabulary_rows;
    std::uint32_t top_experts;
};

// Byte geometry of everything one sealed decode step owns, derived purely
// from the model plan and the serving capacity. The executor allocates
// exactly these buffers once; the per-token walk allocates nothing.
//
// Layer streams hold one hidden vector per layer so the fused kernels can
// write back at layer offsets; the sealed schedule reads layer L's input
// from stream slot L-1. Scratch buffers are shared across layers because
// the step is strictly sequential within one encoder.
struct DecodeGeometry {
    std::uint32_t gated_delta_layers;
    std::uint32_t attention_layers;

    std::uint64_t hidden_bytes;
    std::uint64_t layer_stream_bytes;

    std::uint64_t gdn_projection_bytes;
    std::uint64_t gdn_qk_bytes;
    std::uint64_t gdn_value_bytes;
    std::uint64_t gdn_gate_bytes;
    std::uint64_t gdn_conv_state_bytes;
    std::uint64_t gdn_recurrent_state_bytes;

    std::uint64_t attn_projection_bytes;
    std::uint64_t attn_query_bytes;
    std::uint64_t attn_record_scratch_bytes;
    std::uint64_t attn_cache_bytes;

    std::uint64_t router_logits_bytes;
    std::uint64_t expert_id_bytes;
    std::uint64_t expert_coefficient_bytes;
    std::uint64_t expert_hidden_bytes;

    std::uint64_t logits_bytes;
    std::uint64_t argmax_value_bytes;
    std::uint64_t argmax_index_bytes;
    std::uint64_t token_id_bytes;

    DecodeDispatch dispatch;
};

inline constexpr std::uint32_t kArgmaxGroupCount = 256;
inline constexpr std::uint32_t kAttentionRecordFloats = 258;
inline constexpr std::uint32_t kAttentionPartition = 256;
// This constant is declared twice: here, where it sizes the score dispatch
// (`.score_threads = kAttentionPartition`), and again as bare literals inside
// attention.metal, which writes `threadgroup float scores[256]` and `red[256]`
// and starts its tile-max reduction at `off = 128u`. Nothing links the two --
// the kernel generator does not know the partition exists -- so raising this
// constant alone would dispatch more threads than those threadgroup arrays
// hold and index past the end of them, on the decode hot path, silently.
//
// The assert is the cheap half of the fix: it cannot see the Metal literal, so
// it pins the C++ side to the value the kernel was written for and fails the
// build the moment someone edits one without the other. Deriving the kernel's
// sizes from a generated constant, as router_select now does with kMoeExperts,
// is the real repair and belongs with the Phase 4 work that tunes partitioning.
static_assert(kAttentionPartition == 256,
              "attention.metal hardcodes scores[256], red[256] and off = 128u; "
              "change the kernel literals in the same edit as this constant");

// A context is a cache position, so equality with capacity is already one past
// the allocation. Kept with the pure geometry so the safety boundary is
// testable without constructing a Metal pass.
constexpr bool context_in_capacity(std::uint32_t context, std::uint32_t capacity) {
    return capacity != 0 && context < capacity;
}

inline constexpr std::uint32_t kConvShiftSteps = 3;
inline constexpr std::uint32_t kBf16Bytes = 2;
inline constexpr std::uint32_t kF32Bytes = 4;

// Threadgroup widths the kernel sources fix, not the package: the Apple
// simdgroup width, the RMS elements-per-thread the generated prelude emits,
// and the width each row kernel was sealed with.
inline constexpr std::uint32_t kSimdgroupThreads = 32;
inline constexpr std::uint32_t kRmsValuesPerThread = 4;
inline constexpr std::uint32_t kEmbedThreads = 256;
inline constexpr std::uint32_t kProjectionRowThreads = 128;
inline constexpr std::uint32_t kOutProjectionRowThreads = 64;
inline constexpr std::uint32_t kArgmaxGroupThreads = 256;
// The recurrence packs four value dimensions per threadgroup; the attention
// split scores four query heads per threadgroup and carries two output
// dimensions per value-stage thread.
inline constexpr std::uint32_t kRecurrenceDimensionThreads = 4;
inline constexpr std::uint32_t kAttentionScoreCohortThreads = 4;
inline constexpr std::uint32_t kAttentionValueDimensionsPerThread = 2;

// One simdgroup per row, packed `threads / kSimdgroupThreads` rows to a
// threadgroup. Every row kernel guards its own row index, so a partial
// trailing threadgroup is safe.
constexpr std::uint32_t row_dispatch_groups(std::uint32_t rows, std::uint32_t threads) {
    const std::uint32_t rows_per_group = threads / kSimdgroupThreads;
    return (rows + rows_per_group - 1u) / rows_per_group;
}

template <std::size_t LayerCount>
constexpr DecodeGeometry
make_decode_geometry(const model::qwen36::StaticModelPlan<LayerCount>& plan,
                     std::uint32_t capacity) {
    const std::uint32_t hidden = plan.dimensions.hidden;
    const std::uint32_t vocabulary = plan.dimensions.vocabulary;
    const std::uint32_t query_heads = plan.attention.query_heads;
    const std::uint32_t key_value_heads = plan.attention.key_value_heads;
    const std::uint32_t head_dimension = plan.attention.head_dimension;
    const std::uint32_t recurrent_heads = plan.gated_delta.recurrent_heads;
    const std::uint32_t state_dimension = plan.gated_delta.state_dimension;
    const std::uint32_t experts = plan.mixture_of_experts.experts;
    const std::uint32_t active_experts = plan.mixture_of_experts.active_experts;
    std::uint32_t gated_delta_layers = 0;
    std::uint32_t attention_layers = 0;
    for (const model::qwen36::LayerKind kind : plan.layers) {
        if (kind == model::qwen36::LayerKind::GatedDelta) {
            ++gated_delta_layers;
        } else if (kind == model::qwen36::LayerKind::FullAttention) {
            ++attention_layers;
        }
    }

    // Row counts the sealed kernels walk, shared by the byte sizes and the
    // dispatch shapes below. The gated-delta projection holds q and k for the
    // key heads, then the value block, then the gate block, then one b and one
    // a row per recurrent head; gdn_prepare walks everything below the b rows,
    // one head per threadgroup.
    const std::uint32_t gdn_qk = 2u * (recurrent_heads / 2u) * state_dimension;
    const std::uint32_t gdn_values = recurrent_heads * state_dimension;
    const std::uint32_t gdn_conv_channels = gdn_qk + gdn_values;
    const std::uint32_t gdn_prepare_channels = gdn_conv_channels + gdn_values;
    const std::uint32_t gdn_projection_rows = gdn_prepare_channels + 2u * recurrent_heads;
    const std::uint32_t attn_projection_rows =
        query_heads * 2u * head_dimension + 2u * key_value_heads * head_dimension;
    const std::uint32_t attn_query = query_heads * head_dimension;
    const std::uint32_t expert_hidden_rows =
        (active_experts + 1u) * plan.mixture_of_experts.expert_dimension;
    const std::uint32_t partitions = (capacity + kAttentionPartition - 1u) / kAttentionPartition;

    const DecodeDispatch dispatch{
        .embed = {.groups = hidden / kEmbedThreads, .threads = kEmbedThreads},
        .rms = {.groups = 1, .threads = hidden / kRmsValuesPerThread},
        .gdn_project = {.groups = row_dispatch_groups(gdn_projection_rows, kProjectionRowThreads),
                        .threads = kProjectionRowThreads},
        .gdn_prepare = {.groups = gdn_prepare_channels / state_dimension,
                        .threads = state_dimension},
        .gdn_recurrence = {.dimension_groups = state_dimension / kRecurrenceDimensionThreads,
                           .head_groups = recurrent_heads,
                           .lane_threads = kSimdgroupThreads,
                           .dimension_threads = kRecurrenceDimensionThreads},
        .gdn_gate_norm = {.groups = recurrent_heads, .threads = state_dimension},
        .out_projection = {.groups = row_dispatch_groups(hidden, kOutProjectionRowThreads),
                           .threads = kOutProjectionRowThreads},
        .attn_project = {.groups = row_dispatch_groups(attn_projection_rows, kProjectionRowThreads),
                         .threads = kProjectionRowThreads},
        .attn_qk_rope = {.groups = query_heads + key_value_heads, .threads = head_dimension},
        .attention_head = {.groups = query_heads, .threads = head_dimension},
        .attention_split = {.score_groups = query_heads / kAttentionScoreCohortThreads,
                            .score_threads = kAttentionPartition,
                            .score_cohort_threads = kAttentionScoreCohortThreads,
                            .value_groups = key_value_heads,
                            .value_lane_threads = kSimdgroupThreads,
                            .value_dimension_threads =
                                head_dimension /
                                (kSimdgroupThreads * kAttentionValueDimensionsPerThread),
                            .value_head_threads = query_heads / key_value_heads},
        .router = {.groups = row_dispatch_groups(experts + 1u, kSimdgroupThreads),
                   .threads = kSimdgroupThreads},
        .router_select = {.groups = 1, .threads = experts},
        .grouped_upgate = {.groups = row_dispatch_groups(expert_hidden_rows, kSimdgroupThreads),
                           .threads = kSimdgroupThreads},
        .grouped_down_res = {.groups = row_dispatch_groups(hidden, kSimdgroupThreads),
                             .threads = kSimdgroupThreads},
        .lmhead = {.groups = row_dispatch_groups(vocabulary, kSimdgroupThreads),
                   .threads = kSimdgroupThreads},
        .argmax_stage1 = {.groups = kArgmaxGroupCount, .threads = kArgmaxGroupThreads},
        .vocabulary_rows = vocabulary,
        .top_experts = active_experts,
    };

    return DecodeGeometry{
        .gated_delta_layers = gated_delta_layers,
        .attention_layers = attention_layers,
        .hidden_bytes = std::uint64_t{hidden} * kBf16Bytes,
        .layer_stream_bytes = LayerCount * std::uint64_t{hidden} * kBf16Bytes,
        .gdn_projection_bytes = std::uint64_t{gdn_projection_rows} * kBf16Bytes,
        .gdn_qk_bytes = std::uint64_t{gdn_qk} * kBf16Bytes,
        .gdn_value_bytes = std::uint64_t{gdn_values} * kBf16Bytes,
        .gdn_gate_bytes = std::uint64_t{gdn_values} * kBf16Bytes,
        .gdn_conv_state_bytes = std::uint64_t{kConvShiftSteps} * gdn_conv_channels * kBf16Bytes,
        .gdn_recurrent_state_bytes =
            std::uint64_t{recurrent_heads} * state_dimension * state_dimension * kF32Bytes,
        .attn_projection_bytes = std::uint64_t{attn_projection_rows} * kBf16Bytes,
        .attn_query_bytes = std::uint64_t{attn_query} * kBf16Bytes,
        .attn_record_scratch_bytes =
            std::uint64_t{query_heads} * partitions * kAttentionRecordFloats * kF32Bytes,
        .attn_cache_bytes = std::uint64_t{key_value_heads} * capacity * head_dimension * kBf16Bytes,
        .router_logits_bytes = (std::uint64_t{experts} + 1) * kF32Bytes,
        .expert_id_bytes = std::uint64_t{active_experts} * 4,
        .expert_coefficient_bytes = std::uint64_t{active_experts} * kF32Bytes,
        .expert_hidden_bytes = std::uint64_t{expert_hidden_rows} * kBf16Bytes,
        .logits_bytes = std::uint64_t{vocabulary} * kBf16Bytes,
        .argmax_value_bytes = std::uint64_t{kArgmaxGroupCount} * kF32Bytes,
        .argmax_index_bytes = std::uint64_t{kArgmaxGroupCount} * 4,
        .token_id_bytes = 4,
        .dispatch = dispatch,
    };
}

} // namespace tatara::runtime
