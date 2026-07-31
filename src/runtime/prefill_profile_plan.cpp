#include "tatara/runtime/prefill_profile_plan.h"

#include "tatara/generated/kernel_library.h"
#include "tatara/runtime/prefill_step.h"

#include <cstddef>
#include <limits>

namespace tatara::runtime {
namespace {

struct Chunk {
    std::uint32_t offset{0};
    std::uint32_t rows{0};
    std::uint32_t ordinal{0};
};

bool valid_geometry(const PrefillGeometry& geometry) noexcept {
    return geometry.hidden != 0 && geometry.vocabulary != 0 &&
           geometry.query_heads != 0 && geometry.key_value_heads != 0 &&
           geometry.attention_head_dimension != 0 &&
           geometry.recurrent_heads != 0 && geometry.state_dimension != 0 &&
           geometry.experts != 0 && geometry.active_experts != 0 &&
           geometry.expert_dimension != 0 &&
           geometry.gdn_projection_rows != 0 &&
           geometry.gdn_qk_values != 0 && geometry.gdn_value_values != 0 &&
           geometry.attention_projection_rows != 0 &&
           geometry.attention_vector_values != 0 &&
           geometry.gated_delta_layers != 0 &&
           geometry.attention_layers != 0 && geometry.token_bytes != 0 &&
           geometry.block_hidden_bytes != 0 &&
           geometry.reusable_scratch_bytes != 0 &&
           geometry.steady_prefill_bytes != 0;
}

bool valid_policy(const PrefillGeometry& geometry,
                  const PrefillExecutionPolicy& policy) noexcept {
    const PrefillPolicy& value = policy.geometry;
    const bool known_schedule =
        value.schedule == PrefillSchedule::ChunkMajor ||
        value.schedule == PrefillSchedule::LayerMajor;
    const bool known_selector =
        policy.router_selector == PrefillRouterSelector::Serial ||
        policy.router_selector == PrefillRouterSelector::Parallel;
    const bool known_recurrence =
        policy.gdn_recurrence == PrefillGdnRecurrence::SerialSteps ||
        (policy.gdn_recurrence == PrefillGdnRecurrence::RegisterLoop || policy.gdn_recurrence == PrefillGdnRecurrence::RegisterLoopTape);
    const bool known_attention =
        policy.attention_kernel ==
            PrefillAttentionKernel::PartialCombine ||
        policy.attention_kernel ==
            PrefillAttentionKernel::StagedGemmAdaptive ||
        policy.attention_kernel ==
            PrefillAttentionKernel::StreamingFlashAdaptive ||
        policy.attention_kernel ==
            PrefillAttentionKernel::FlashMmaV2 ||
        policy.attention_kernel ==
            PrefillAttentionKernel::SteelGemm;
    const bool known_dense_qgemm =
        policy.dense_qgemm == QuantizedGemmPolicy::ExactRow ||
        policy.dense_qgemm == QuantizedGemmPolicy::NativeDenseMma;
    const bool known_routed_qgemm =
        policy.routed_qgemm == QuantizedGemmPolicy::ExactRow ||
        policy.routed_qgemm == QuantizedGemmPolicy::NativeRaggedMma;
    const bool routed_schedule_matches =
        policy.routed_qgemm == QuantizedGemmPolicy::ExactRow ||
        value.schedule == PrefillSchedule::LayerMajor;
    const bool shared_native_matches =
        !policy.native_routed_shared_expert ||
        policy.routed_qgemm ==
            QuantizedGemmPolicy::NativeRaggedMma;
    const bool shared_task_capacity_matches =
        !policy.native_routed_shared_expert ||
        native_routed_shared_task_capacity_supported(
            geometry,
            backend::metal::generated::
                kKernelLibraryNativeRoutedQgemmR1TileRows,
            backend::metal::generated::
                kKernelLibraryNativeRoutedQgemmR1TaskCapacity);
    const bool geometry_matches =
        value.schedule == geometry.schedule &&
        value.context_capacity == geometry.context_capacity &&
        value.maximum_block_rows == geometry.maximum_block_rows &&
        value.first_chunk_rows == geometry.first_chunk_rows &&
        value.query_tile_rows == geometry.query_tile_rows &&
        value.attention_partition == geometry.attention_partition &&
        value.exact_rows_per_threadgroup ==
            geometry.exact_rows_per_threadgroup &&
        value.gdn_gate_hoist == geometry.gdn_gate_hoist;
    const bool slab_matches =
        value.schedule == PrefillSchedule::LayerMajor
            ? geometry.hidden_slab_bytes != 0
            : geometry.hidden_slab_bytes == 0;
    const bool hoist_matches =
        value.gdn_gate_hoist ? geometry.gdn_parameter_bytes != 0
                             : geometry.gdn_parameter_bytes == 0;
    const bool recurrence_matches =
        (policy.gdn_recurrence == PrefillGdnRecurrence::RegisterLoop || policy.gdn_recurrence == PrefillGdnRecurrence::RegisterLoopTape) ||
        !value.gdn_gate_hoist;
    const bool attention_family_matches =
        policy.attention_kernel ==
            PrefillAttentionKernel::PartialCombine ||
        (policy.attention_kernel ==
             PrefillAttentionKernel::StagedGemmAdaptive
             ? policy.staged_attention_minimum_context <
                       value.context_capacity &&
                   geometry.attention_staged_score_bytes != 0 &&
                   geometry.attention_staged_score_bytes <=
                       geometry.attention_partial_bytes
             : policy.streaming_attention_minimum_context <
                   value.context_capacity);

    return known_schedule && known_selector && known_recurrence &&
           known_attention &&
           known_dense_qgemm && known_routed_qgemm &&
           routed_schedule_matches && shared_native_matches &&
           shared_task_capacity_matches &&
           attention_family_matches &&
           geometry_matches && slab_matches && hoist_matches &&
           recurrence_matches && value.context_capacity != 0 &&
           value.context_capacity <= kPrefillMaximumContext &&
           value.maximum_block_rows != 0 &&
           value.maximum_block_rows <= kPrefillMaximumBlockRows &&
           value.first_chunk_rows != 0 &&
           value.first_chunk_rows <= value.maximum_block_rows &&
           value.query_tile_rows != 0 &&
           value.query_tile_rows <= value.maximum_block_rows &&
           value.query_tile_rows % kPrefillQueryTileMultiple == 0 &&
           value.attention_partition == kAttentionPartition &&
           value.exact_rows_per_threadgroup != 0 &&
           value.exact_rows_per_threadgroup <=
               kPrefillMaximumExactRowsPerThreadgroup;
}

bool valid_schedule(
    const PrefillGeometry& geometry,
    std::span<const model::qwen36::LayerKind> schedule) noexcept {
    const std::uint64_t expected =
        static_cast<std::uint64_t>(geometry.gated_delta_layers) +
        geometry.attention_layers;
    if (expected > std::numeric_limits<std::size_t>::max() ||
        schedule.size() != static_cast<std::size_t>(expected)) {
        return false;
    }

    std::uint64_t gated_delta_layers = 0;
    std::uint64_t attention_layers = 0;
    for (const model::qwen36::LayerKind kind : schedule) {
        if (kind == model::qwen36::LayerKind::GatedDelta) {
            ++gated_delta_layers;
        } else if (kind == model::qwen36::LayerKind::FullAttention) {
            ++attention_layers;
        } else {
            return false;
        }
    }
    return gated_delta_layers == geometry.gated_delta_layers &&
           attention_layers == geometry.attention_layers;
}

std::uint32_t first_chunk_rows(
    const PrefillExecutionPolicy& policy,
    std::uint32_t initial_context) noexcept {
    return policy.geometry.schedule == PrefillSchedule::LayerMajor &&
                   initial_context == 0
               ? policy.geometry.first_chunk_rows
               : policy.geometry.maximum_block_rows;
}

bool make_chunk_count(const PrefillExecutionPolicy& policy,
                      std::uint32_t initial_context,
                      std::uint32_t rows,
                      std::uint32_t& count) noexcept {
    const std::uint32_t first =
        first_chunk_rows(policy, initial_context);
    if (rows <= first) {
        count = 1;
        return true;
    }
    const std::uint64_t remainder =
        static_cast<std::uint64_t>(rows) - first;
    const std::uint64_t tail_chunks =
        remainder / policy.geometry.maximum_block_rows +
        (remainder % policy.geometry.maximum_block_rows != 0 ? 1U : 0U);
    if (tail_chunks >= std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    count = static_cast<std::uint32_t>(tail_chunks + 1U);
    return true;
}

bool make_chunk(const PrefillExecutionPolicy& policy,
                std::uint32_t initial_context,
                std::uint32_t total_rows,
                std::uint32_t ordinal,
                Chunk& chunk) noexcept {
    const std::uint32_t first =
        first_chunk_rows(policy, initial_context);
    if (ordinal == 0) {
        chunk = {
            .offset = 0,
            .rows = total_rows < first ? total_rows : first,
            .ordinal = 0,
        };
        return true;
    }

    const std::uint64_t offset =
        static_cast<std::uint64_t>(first) +
        static_cast<std::uint64_t>(ordinal - 1U) *
            policy.geometry.maximum_block_rows;
    if (offset >= total_rows ||
        offset > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    const std::uint32_t remaining =
        total_rows - static_cast<std::uint32_t>(offset);
    chunk = {
        .offset = static_cast<std::uint32_t>(offset),
        .rows = remaining < policy.geometry.maximum_block_rows
                    ? remaining
                    : policy.geometry.maximum_block_rows,
        .ordinal = ordinal,
    };
    return true;
}

class EventSink {
  public:
    EventSink(std::span<PrefillProfileEvent> output, bool write) noexcept
        : output_(output), write_(write) {}

    bool append(PrefillProfileEvent event) noexcept {
        if (count_ == std::numeric_limits<std::uint64_t>::max()) {
            return false;
        }
        if (write_) {
            if (count_ >= output_.size()) {
                return false;
            }
            output_[static_cast<std::size_t>(count_)] = event;
        }
        ++count_;
        return true;
    }

    [[nodiscard]] std::uint64_t count() const noexcept {
        return count_;
    }

  private:
    std::span<PrefillProfileEvent> output_;
    std::uint64_t count_{0};
    bool write_{false};
};

PrefillProfileEvent make_event(PrefillProfileEventClass event_class,
                               std::uint64_t layer_index,
                               const Chunk& chunk,
                               std::uint32_t operation_row_begin,
                               std::uint32_t operation_row_count) noexcept {
    return {
        .event_class = event_class,
        .layer_index = layer_index,
        .chunk_ordinal = chunk.ordinal,
        .chunk_offset = chunk.offset,
        .chunk_rows = chunk.rows,
        .operation_row_begin = operation_row_begin,
        .operation_row_count = operation_row_count,
    };
}

bool append_chunk_event(EventSink& sink,
                        PrefillProfileEventClass event_class,
                        std::uint64_t layer_index,
                        const Chunk& chunk) noexcept {
    return sink.append(
        make_event(event_class, layer_index, chunk, 0, chunk.rows));
}

bool append_dense_events(EventSink& sink,
                         const PrefillExecutionPolicy& policy,
                         PrefillProfileEventClass event_class,
                         std::uint32_t native_dispatch_count,
                         std::uint64_t layer_index,
                         const Chunk& chunk) noexcept {
    const std::uint32_t count =
        policy.dense_qgemm == QuantizedGemmPolicy::NativeDenseMma
            ? native_dispatch_count
            : 1u;
    for (std::uint32_t index = 0; index < count; ++index) {
        if (!append_chunk_event(
                sink, event_class, layer_index, chunk)) {
            return false;
        }
    }
    return true;
}

bool append_moe(EventSink& sink,
                const PrefillExecutionPolicy& policy,
                std::uint64_t layer_index,
                const Chunk& chunk) noexcept {
    const PrefillProfileEventClass selector =
        policy.router_selector == PrefillRouterSelector::Parallel
            ? PrefillProfileEventClass::MoeRouterSelectParallel
            : PrefillProfileEventClass::MoeRouterSelectSerial;
    if (!append_chunk_event(
            sink, PrefillProfileEventClass::MoeResidualInput,
            layer_index, chunk) ||
        !append_chunk_event(
            sink, PrefillProfileEventClass::MoePostNormalization,
            layer_index, chunk) ||
        !append_chunk_event(sink, PrefillProfileEventClass::MoeRouter,
                            layer_index, chunk) ||
        !append_chunk_event(sink, selector, layer_index, chunk) ||
        !append_chunk_event(sink, PrefillProfileEventClass::MoeExpertUnion,
                            layer_index, chunk)) {
        return false;
    }
    if (policy.routed_qgemm == QuantizedGemmPolicy::NativeRaggedMma) {
        const PrefillProfileEventClass native_upgate =
            policy.native_routed_shared_expert
                ? PrefillProfileEventClass::
                      MoeNativeRoutedSharedUpGate
                : PrefillProfileEventClass::
                      MoeNativeRoutedUpGate;
        const PrefillProfileEventClass native_down =
            policy.native_routed_shared_expert
                ? PrefillProfileEventClass::
                      MoeNativeRoutedSharedDown
                : PrefillProfileEventClass::
                      MoeNativeRoutedDown;
        if (!append_chunk_event(
                sink, PrefillProfileEventClass::MoeRoutedTaskBuild,
                layer_index, chunk) ||
            !append_chunk_event(
                sink, native_upgate,
                layer_index, chunk)) {
            return false;
        }
        if (!policy.native_routed_shared_expert &&
            !append_chunk_event(
                sink,
                PrefillProfileEventClass::MoeSharedExpertUpGate,
                layer_index, chunk)) {
            return false;
        }
        if (!append_chunk_event(
                sink, PrefillProfileEventClass::MoeRoutedTaskBuild,
                layer_index, chunk) ||
            !append_chunk_event(
                sink, native_down,
                layer_index, chunk)) {
            return false;
        }
        if (!policy.native_routed_shared_expert &&
            !append_chunk_event(
                sink,
                PrefillProfileEventClass::MoeSharedExpertDown,
                layer_index, chunk)) {
            return false;
        }
    } else if (!append_chunk_event(
                   sink, PrefillProfileEventClass::MoeExpertUpGate,
                   layer_index, chunk) ||
               !append_chunk_event(
                   sink, PrefillProfileEventClass::MoeExpertDown,
                   layer_index, chunk)) {
        return false;
    }
    if (!append_chunk_event(
            sink, PrefillProfileEventClass::MoeExpertCombine,
            layer_index, chunk) ||
        !append_chunk_event(
            sink, PrefillProfileEventClass::MoeResidualOutput,
            layer_index, chunk)) {
        return false;
    }
    // A41a: the conditioning-capture copy is one extra dispatch after a
    // captured layer's residual.
    if (policy.conditioning_capture) {
        for (std::uint32_t slot = 0; slot < 8u; ++slot) {
            if (policy.capture_layers[slot] ==
                static_cast<std::uint32_t>(layer_index)) {
                if (!append_chunk_event(
                        sink,
                        PrefillProfileEventClass::MoeResidualOutput,
                        layer_index, chunk)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool append_gated_delta(EventSink& sink,
                        const PrefillExecutionPolicy& policy,
                        std::uint64_t layer_index,
                        const Chunk& chunk) noexcept {
    constexpr std::uint32_t kBm64TileRows = 64U;
    const std::uint32_t gdn_projection_dispatches =
        policy.native_dense_steel_gdn_bm64_wm2_wn2 &&
                chunk.rows >= kBm64TileRows &&
                chunk.rows % kBm64TileRows != 0U
            ? 8U
            : 4U;
    if (!append_dense_events(
            sink, policy,
            PrefillProfileEventClass::GdnProjection,
            gdn_projection_dispatches,
            layer_index, chunk) ||
        !append_chunk_event(sink, PrefillProfileEventClass::GdnConvolution,
                            layer_index, chunk)) {
        return false;
    }

    if (policy.gdn_recurrence == PrefillGdnRecurrence::SerialSteps) {
        for (std::uint32_t row = 0; row < chunk.rows; ++row) {
            if (!sink.append(make_event(
                    PrefillProfileEventClass::GdnRecurrenceSerialStep,
                    layer_index, chunk, row, 1))) {
                return false;
            }
        }
    } else {
        if (policy.geometry.gdn_gate_hoist &&
            !append_chunk_event(sink,
                                PrefillProfileEventClass::GdnGateHoist,
                                layer_index, chunk)) {
            return false;
        }
        if (!append_chunk_event(
                sink, PrefillProfileEventClass::GdnRecurrenceRegisterLoop,
                layer_index, chunk)) {
            return false;
        }
    }

    return append_chunk_event(
               sink, PrefillProfileEventClass::GdnGateNormalization,
               layer_index, chunk) &&
           append_chunk_event(
               sink, PrefillProfileEventClass::GdnOutputProjection,
               layer_index, chunk);
}

bool append_attention(EventSink& sink,
                      const PrefillExecutionPolicy& policy,
                      std::uint32_t initial_context,
                      std::uint64_t layer_index,
                      const Chunk& chunk) noexcept {
    if (!append_dense_events(
            sink, policy, PrefillProfileEventClass::AttentionProjection, 3,
            layer_index, chunk) ||
        !append_chunk_event(sink, PrefillProfileEventClass::AttentionQkRope,
                            layer_index, chunk)) {
        return false;
    }

    for (std::uint32_t query_base = 0; query_base < chunk.rows;) {
        const std::uint32_t remaining = chunk.rows - query_base;
        const std::uint32_t tile_rows =
            remaining < policy.geometry.query_tile_rows
                ? remaining
                : policy.geometry.query_tile_rows;
        const std::uint64_t visible =
            static_cast<std::uint64_t>(initial_context) +
            chunk.offset + query_base + tile_rows;
        const bool staged =
            policy.attention_kernel ==
                PrefillAttentionKernel::StagedGemmAdaptive &&
            visible > policy.staged_attention_minimum_context;
        const bool steel_gemm =
            policy.attention_kernel ==
                PrefillAttentionKernel::SteelGemm &&
            visible > policy.streaming_attention_minimum_context;
        const bool streaming =
            (policy.attention_kernel ==
                 PrefillAttentionKernel::StreamingFlashAdaptive ||
             policy.attention_kernel ==
                 PrefillAttentionKernel::FlashMmaV2) &&
            visible >
                policy.streaming_attention_minimum_context;
        if (steel_gemm) {
            // Two score GEMMs + softmax + two value GEMMs + gate apply.
            if (!sink.append(make_event(
                    PrefillProfileEventClass::AttentionStagedScores,
                    layer_index, chunk, query_base, tile_rows)) ||
                !sink.append(make_event(
                    PrefillProfileEventClass::AttentionStagedScores,
                    layer_index, chunk, query_base, tile_rows)) ||
                !sink.append(make_event(
                    PrefillProfileEventClass::AttentionStagedSoftmax,
                    layer_index, chunk, query_base, tile_rows)) ||
                !sink.append(make_event(
                    PrefillProfileEventClass::AttentionStagedValues,
                    layer_index, chunk, query_base, tile_rows)) ||
                !sink.append(make_event(
                    PrefillProfileEventClass::AttentionStagedValues,
                    layer_index, chunk, query_base, tile_rows)) ||
                !sink.append(make_event(
                    PrefillProfileEventClass::AttentionCombine,
                    layer_index, chunk, query_base, tile_rows))) {
                return false;
            }
        } else if (streaming) {
            if (!sink.append(make_event(
                    PrefillProfileEventClass::AttentionStreaming,
                    layer_index, chunk, query_base, tile_rows))) {
                return false;
            }
        } else if (staged) {
            if (!sink.append(make_event(
                    PrefillProfileEventClass::AttentionStagedScores,
                    layer_index, chunk, query_base, tile_rows)) ||
                !sink.append(make_event(
                    PrefillProfileEventClass::AttentionStagedSoftmax,
                    layer_index, chunk, query_base, tile_rows)) ||
                !sink.append(make_event(
                    PrefillProfileEventClass::AttentionStagedValues,
                    layer_index, chunk, query_base, tile_rows))) {
                return false;
            }
        } else if (
            !sink.append(make_event(
                PrefillProfileEventClass::AttentionPartial,
                layer_index, chunk, query_base, tile_rows)) ||
            !sink.append(make_event(
                PrefillProfileEventClass::AttentionCombine,
                layer_index, chunk, query_base, tile_rows))) {
            return false;
        }
        query_base += tile_rows;
    }

    return append_chunk_event(
        sink, PrefillProfileEventClass::AttentionOutputProjection, layer_index,
        chunk);
}

bool append_layer(EventSink& sink,
                  const PrefillExecutionPolicy& policy,
                  std::uint32_t initial_context,
                  model::qwen36::LayerKind kind,
                  std::uint64_t layer_index,
                  const Chunk& chunk) noexcept {
    if (!append_chunk_event(
            sink, PrefillProfileEventClass::LayerInputNormalization,
            layer_index, chunk)) {
        return false;
    }
    if (kind == model::qwen36::LayerKind::GatedDelta) {
        if (!append_gated_delta(sink, policy, layer_index, chunk)) {
            return false;
        }
    } else if (kind == model::qwen36::LayerKind::FullAttention) {
        if (!append_attention(
                sink, policy, initial_context, layer_index, chunk)) {
            return false;
        }
    } else {
        return false;
    }
    return append_moe(sink, policy, layer_index, chunk);
}

bool append_plan(EventSink& sink,
                 const PrefillExecutionPolicy& policy,
                 std::uint32_t initial_context,
                 std::uint32_t request_rows,
                 std::span<const model::qwen36::LayerKind> schedule,
                 std::uint32_t chunk_count) noexcept {
    if (policy.geometry.schedule == PrefillSchedule::ChunkMajor) {
        for (std::uint32_t ordinal = 0; ordinal < chunk_count; ++ordinal) {
            Chunk chunk;
            if (!make_chunk(
                    policy, initial_context, request_rows, ordinal,
                    chunk) ||
                !append_chunk_event(sink,
                                    PrefillProfileEventClass::Embedding,
                                    kNoPrefillProfileLayerIndex, chunk)) {
                return false;
            }
            for (std::size_t layer = 0; layer < schedule.size(); ++layer) {
                if (!append_layer(sink, policy, initial_context,
                                  schedule[layer], layer, chunk)) {
                    return false;
                }
            }
        }
        return true;
    }

    for (std::size_t layer = 0; layer < schedule.size(); ++layer) {
        for (std::uint32_t ordinal = 0; ordinal < chunk_count; ++ordinal) {
            Chunk chunk;
            if (!make_chunk(
                    policy, initial_context, request_rows, ordinal,
                    chunk)) {
                return false;
            }
            if (layer == 0 &&
                !append_chunk_event(sink,
                                    PrefillProfileEventClass::Embedding,
                                    kNoPrefillProfileLayerIndex, chunk)) {
                return false;
            }
            if (!append_layer(sink, policy, initial_context,
                              schedule[layer], layer, chunk)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

PrefillProfilePlanResult make_prefill_profile_plan(
    const PrefillGeometry& geometry,
    const PrefillExecutionPolicy& policy,
    std::uint32_t request_rows,
    std::span<const model::qwen36::LayerKind> schedule,
    std::span<PrefillProfileEvent> output,
    std::uint32_t initial_context) noexcept {
    if (!valid_geometry(geometry)) {
        return {.error = PrefillProfilePlanError::InvalidGeometry};
    }
    if (!valid_policy(geometry, policy)) {
        return {.error = PrefillProfilePlanError::InvalidPolicy};
    }
    if (request_rows == 0) {
        return {.error = PrefillProfilePlanError::EmptyRequest};
    }
    if (request_rows > geometry.context_capacity ||
        initial_context > geometry.context_capacity - request_rows) {
        return {.error = PrefillProfilePlanError::RequestRowsOutOfRange};
    }
    if (!valid_schedule(geometry, schedule)) {
        return {.error = PrefillProfilePlanError::InvalidSchedule};
    }

    std::uint32_t chunks = 0;
    if (!make_chunk_count(
            policy, initial_context, request_rows, chunks)) {
        return {.error = PrefillProfilePlanError::ArithmeticOverflow};
    }

    EventSink counter({}, false);
    if (!append_plan(
            counter, policy, initial_context, request_rows, schedule,
            chunks)) {
        return {.error = PrefillProfilePlanError::ArithmeticOverflow,
                .required_event_count = counter.count(),
                .chunk_count = chunks};
    }
    const std::uint64_t required = counter.count();
    if (required > std::numeric_limits<std::size_t>::max() ||
        output.size() < static_cast<std::size_t>(required)) {
        return {
            .error = PrefillProfilePlanError::EventCapacityInsufficient,
            .required_event_count = required,
            .chunk_count = chunks,
        };
    }

    EventSink writer(output.first(static_cast<std::size_t>(required)), true);
    if (!append_plan(
            writer, policy, initial_context, request_rows, schedule,
            chunks) ||
        writer.count() != required) {
        return {
            .error = PrefillProfilePlanError::ArithmeticOverflow,
            .required_event_count = required,
            .written_event_count = writer.count(),
            .chunk_count = chunks,
        };
    }
    return {
        .error = PrefillProfilePlanError::None,
        .required_event_count = required,
        .written_event_count = required,
        .chunk_count = chunks,
    };
}

} // namespace tatara::runtime
