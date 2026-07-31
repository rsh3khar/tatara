#include "tatara/runtime/decode_step.h"

#include "tatara/model/image_population.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace tatara::runtime {

namespace {

using namespace backend::metal;

std::uint32_t attention_partitions(std::uint32_t context) {
    const std::uint32_t count = context + 1;
    return count <= kAttentionPartition ? 1
                                        : (count + kAttentionPartition - 1) / kAttentionPartition;
}

std::uint32_t attention_vector_blocks(const DecodeStep& step,
                                      std::uint32_t partitions) {
    constexpr std::uint32_t kReductionCohort = 32;
    constexpr std::uint32_t kMaximumScheduledBlocks = 1024;
    if (step.geometry.attn_query_bytes == 0) {
        return 0;
    }
    const std::uint64_t available =
        step.geometry.attn_record_scratch_bytes /
        step.geometry.attn_query_bytes;
    const std::uint32_t maximum_fitting =
        available > std::numeric_limits<std::uint32_t>::max()
            ? std::numeric_limits<std::uint32_t>::max()
            : static_cast<std::uint32_t>(available);
    const std::uint32_t fitting_cohorts =
        maximum_fitting / kReductionCohort;
    if (fitting_cohorts == 0) {
        return 0;
    }
    const std::uint64_t doubled = std::uint64_t{partitions} * 2u;
    const std::uint64_t rounded =
        ((doubled + kReductionCohort - 1u) / kReductionCohort) *
        kReductionCohort;
    const std::uint32_t desired = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(rounded, kMaximumScheduledBlocks));
    return std::min(desired, fitting_cohorts * kReductionCohort);
}

bool allocate_zeroed(const MetalDevice& device, std::uint64_t size_bytes, MetalBuffer& buffer) {
    auto result = create_shared_buffer(device, size_bytes);
    if (!result) {
        return false;
    }
    buffer = std::move(*result.buffer);
    std::memset(buffer.contents(), 0, size_bytes);
    return true;
}

DecodeStateSlotResult allocate_state_slot(const MetalDevice& device, const DecodeGeometry& geometry,
                                          std::uint32_t capacity,
                                          std::span<const model::qwen36::LayerKind> schedule,
                                          const model::qwen36::LayerKind* schedule_identity) {
    if (!device) {
        return {.error = DecodeStepError::InvalidDevice, .slot = std::nullopt};
    }
    if (capacity == 0 || schedule.empty() || schedule_identity == nullptr) {
        return {.error = DecodeStepError::OffsetCountMismatch, .slot = std::nullopt};
    }
    DecodeStateSlot slot{
        .capacity = capacity,
        .schedule_identity = schedule_identity,
    };
    slot.layers.reserve(schedule.size());
    for (const model::qwen36::LayerKind kind : schedule) {
        DecodeLayerState state{
            .first = {}, .first_out = {}, .second = {}, .second_out = {}, .swapped = false};
        const bool gated = kind == model::qwen36::LayerKind::GatedDelta;
        const std::uint64_t first_bytes =
            gated ? geometry.gdn_conv_state_bytes : geometry.attn_cache_bytes;
        const std::uint64_t second_bytes =
            gated ? geometry.gdn_recurrent_state_bytes : geometry.attn_cache_bytes;
        if (!allocate_zeroed(device, first_bytes, state.first) ||
            !allocate_zeroed(device, second_bytes, state.second)) {
            return {.error = DecodeStepError::BufferAllocationFailed, .slot = std::nullopt};
        }
        if (gated && (!allocate_zeroed(device, first_bytes, state.first_out) ||
                      !allocate_zeroed(device, second_bytes, state.second_out))) {
            return {.error = DecodeStepError::BufferAllocationFailed, .slot = std::nullopt};
        }
        slot.layers.push_back(std::move(state));
    }
    return {.error = DecodeStepError::None, .slot = std::move(slot)};
}

// Issues the walk's calls in order and latches the first error, so the
// schedule reads as the sealed dispatch sequence rather than error plumbing.
//
// Latching also GATES: once an error is held, every later call is skipped. A
// latch that only records would let a rejected bind be followed by a recorded
// dispatch, and a dispatch whose argument slot was never bound reads whatever
// the previous pipeline left there -- a scratch-sized allocation under a
// weight-shaped kernel, which is an out-of-bounds GPU read rather than a typed
// failure. Callers do check the returned error before committing, so nothing
// reaches the device today; this makes the guarantee structural instead of
// resting on every future caller remembering to look.
struct Encoder {
    MetalComputePass& pass;
    const MetalBuffer& image;
    std::span<const std::uint64_t> offsets;
    MetalCommandError error = MetalCommandError::None;

    bool failed() const {
        return error != MetalCommandError::None;
    }
    void check(MetalCommandError result) {
        if (error == MetalCommandError::None && result != MetalCommandError::None) {
            error = result;
        }
    }
    void pipeline(const MetalComputePipeline& state) {
        if (failed()) {
            return;
        }
        check(set_compute_pipeline(pass, state));
    }
    void buffer(const MetalBuffer& target, std::uint64_t offset, std::uint32_t index) {
        if (failed()) {
            return;
        }
        check(set_buffer(pass, target, offset, index));
    }
    void weight(std::uint32_t tensor, std::uint32_t index) {
        if (failed()) {
            return;
        }
        // An excluded tensor carries a sentinel offset. It is unreachable --
        // the bound set is provably disjoint from the excluded set -- but the
        // sentinel is in-band, so it is rejected here by name rather than
        // relying on set_buffer's range check to catch it incidentally.
        if (offsets[tensor] == model::kExcludedTensorOffset) {
            check(MetalCommandError::InvalidBufferOffset);
            return;
        }
        check(set_buffer(pass, image, offsets[tensor], index));
    }
    void quantized(const QuantizedBinding& binding, std::uint32_t first_index) {
        weight(binding.weight, first_index);
        weight(binding.scales, first_index + 1);
        weight(binding.biases, first_index + 2);
    }
    void constant(std::uint32_t value, std::uint32_t index) {
        if (failed()) {
            return;
        }
        check(set_bytes(pass, &value, sizeof(value), index));
    }
    void dispatch(std::uint64_t groups_x, std::uint64_t threads_x, std::uint64_t groups_y = 1,
                  std::uint64_t threads_y = 1, std::uint64_t threads_z = 1) {
        if (failed()) {
            return;
        }
        check(dispatch_threadgroups(
            pass, MetalSize{.width = groups_x, .height = groups_y, .depth = 1},
            MetalSize{.width = threads_x, .height = threads_y, .depth = threads_z}));
    }
    void dispatch(DispatchShape shape) {
        dispatch(shape.groups, shape.threads);
    }
    // The recurrence is the one dispatch that is three-dimensional on both
    // extents. It goes through the Encoder like every other call so the gate
    // above covers it; calling dispatch_threadgroups directly would slip past.
    void dispatch(MetalSize threadgroups, MetalSize threads_per_group) {
        if (failed()) {
            return;
        }
        check(dispatch_threadgroups(pass, threadgroups, threads_per_group));
    }
    void barrier() {
        if (failed()) {
            return;
        }
        check(memory_barrier(pass));
    }
};

void encode_attention_decode(Encoder& encode, DecodeStep& step, const MetalBuffer& keys,
                             const MetalBuffer& values, std::uint32_t context) {
    const DecodeDispatch& shape = step.geometry.dispatch;
    const AttentionSplitDispatch& split = shape.attention_split;
    const std::uint32_t partitions = attention_partitions(context);
    if (partitions == 1) {
        encode.pipeline(step.pipelines.attention_decode);
        encode.buffer(step.attn_query, 0, 0);
        encode.buffer(step.attn_gate, 0, 1);
        encode.buffer(keys, 0, 2);
        encode.buffer(values, 0, 3);
        encode.constant(context, 4);
        encode.buffer(step.attn_attended, 0, 5);
        encode.constant(step.capacity, 6);
        encode.dispatch(shape.attention_head);
        return;
    }
    const std::uint32_t vector_blocks =
        attention_vector_blocks(step, partitions);
    const bool vector_requested =
        step.pipelines.attention_split_policy ==
            DecodeAttentionSplitPolicy::IndependentHeadVector2Pass ||
        (step.pipelines.attention_split_policy ==
             DecodeAttentionSplitPolicy::AdaptiveVector2Pass &&
         context >= step.pipelines.vector_minimum_context);
    if (vector_requested &&
        vector_blocks != 0) {
        const std::uint64_t scalar_plane_bytes =
            std::uint64_t{shape.attention_head.groups} *
            vector_blocks * sizeof(float);
        encode.pipeline(step.pipelines.attention_vector_part);
        encode.buffer(step.attn_query, 0, 0);
        encode.buffer(keys, 0, 1);
        encode.buffer(values, 0, 2);
        encode.buffer(step.attn_partials, 0, 3);
        encode.buffer(step.attn_weights, 0, 4);
        encode.buffer(step.attn_weights, scalar_plane_bytes, 5);
        encode.constant(context, 6);
        encode.constant(step.capacity, 7);
        encode.constant(vector_blocks, 8);
        encode.dispatch(split.value_groups, kSimdgroupThreads,
                        vector_blocks, split.value_head_threads);
        encode.barrier();
        encode.pipeline(step.pipelines.attention_vector_combine);
        encode.buffer(step.attn_partials, 0, 0);
        encode.buffer(step.attn_weights, 0, 1);
        encode.buffer(step.attn_weights, scalar_plane_bytes, 2);
        encode.buffer(step.attn_gate, 0, 3);
        encode.constant(vector_blocks, 4);
        encode.buffer(step.attn_attended, 0, 5);
        encode.dispatch(shape.attention_head.groups, 1024);
        return;
    }
    const bool fused_score_value =
        step.pipelines.attention_split_policy ==
            DecodeAttentionSplitPolicy::FusedGqa8ScoreValue ||
        ((step.pipelines.attention_split_policy ==
              DecodeAttentionSplitPolicy::AdaptiveGqa8ScoreValue ||
          step.pipelines.attention_split_policy ==
              DecodeAttentionSplitPolicy::AdaptiveVector2Pass) &&
         context >=
             step.pipelines.fused_score_value_minimum_context);
    if (fused_score_value) {
        encode.pipeline(step.pipelines.attention_scores_values_fused);
        encode.buffer(step.attn_query, 0, 0);
        encode.buffer(keys, 0, 1);
        encode.buffer(values, 0, 2);
        encode.constant(context, 3);
        encode.constant(step.capacity, 4);
        encode.constant(kAttentionPartition, 5);
        encode.buffer(step.attn_partials, 0, 6);
        encode.constant(partitions, 7);
        encode.dispatch(split.value_groups, split.score_threads, partitions,
                        split.score_cohort_threads);
        encode.barrier();
    } else {
        encode.pipeline(step.pipelines.attention_scores);
        encode.buffer(step.attn_query, 0, 0);
        encode.buffer(keys, 0, 1);
        encode.constant(context, 2);
        encode.constant(step.capacity, 3);
        encode.constant(kAttentionPartition, 4);
        encode.buffer(step.attn_weights, 0, 5);
        encode.constant(partitions, 6);
        encode.dispatch(split.score_groups, split.score_threads, partitions,
                        split.score_cohort_threads);
        encode.barrier();
        encode.pipeline(step.pipelines.attention_values);
        encode.buffer(step.attn_weights, 0, 0);
        encode.buffer(values, 0, 1);
        encode.constant(context, 2);
        encode.constant(step.capacity, 3);
        encode.constant(kAttentionPartition, 4);
        encode.buffer(step.attn_partials, 0, 5);
        encode.constant(partitions, 6);
        encode.dispatch(split.value_groups, split.value_lane_threads, partitions,
                        split.value_dimension_threads, split.value_head_threads);
        encode.barrier();
    }
    encode.pipeline(step.pipelines.attention_combine);
    encode.buffer(step.attn_partials, 0, 0);
    encode.buffer(step.attn_gate, 0, 1);
    encode.constant(partitions, 2);
    encode.buffer(step.attn_attended, 0, 3);
    encode.dispatch(shape.attention_head);
}

} // namespace

DecodeStepResult create_decode_step(const MetalDevice& device, const DecodeGeometry& geometry,
                                    std::uint32_t capacity,
                                    std::span<const model::qwen36::LayerKind> schedule,
                                    DecodeBindings bindings, const MetalBuffer& image,
                                    std::span<const std::uint64_t> tensor_offsets,
                                    DecodePipelines pipelines) {
    if (!device) {
        return {.error = DecodeStepError::InvalidDevice, .step = std::nullopt};
    }
    if (!image) {
        return {.error = DecodeStepError::InvalidImage, .step = std::nullopt};
    }
    if (tensor_offsets.empty() || bindings.layers.size() != schedule.size()) {
        return {.error = DecodeStepError::OffsetCountMismatch, .step = std::nullopt};
    }
    DecodeStep step{
        .geometry = geometry,
        .capacity = capacity,
        .bindings = std::move(bindings),
        .schedule = {schedule.begin(), schedule.end()},
        .image = &image,
        .tensor_offsets = {tensor_offsets.begin(), tensor_offsets.end()},
        .pipelines = std::move(pipelines),
    };
    const struct {
        std::uint64_t size;
        MetalBuffer* buffer;
    } allocations[] = {
        {geometry.hidden_bytes, &step.input},
        {geometry.hidden_bytes, &step.normed},
        {geometry.layer_stream_bytes, &step.branch_stream},
        {geometry.layer_stream_bytes, &step.residual_stream},
        {geometry.layer_stream_bytes, &step.moe_stream},
        {geometry.layer_stream_bytes, &step.layer_stream},
        {geometry.gdn_projection_bytes, &step.gdn_projection},
        {geometry.gdn_qk_bytes, &step.gdn_qk},
        {geometry.gdn_value_bytes, &step.gdn_value},
        {geometry.gdn_gate_bytes, &step.gdn_z},
        {geometry.gdn_value_bytes, &step.gdn_y},
        {geometry.gdn_gate_bytes, &step.gdn_gated},
        {geometry.attn_projection_bytes, &step.attn_projection},
        {geometry.attn_query_bytes, &step.attn_query},
        {geometry.attn_query_bytes, &step.attn_gate},
        {geometry.attn_query_bytes, &step.attn_attended},
        {geometry.attn_record_scratch_bytes, &step.attn_partials},
        {geometry.attn_record_scratch_bytes, &step.attn_weights},
        {geometry.router_logits_bytes, &step.router_logits},
        {geometry.expert_id_bytes, &step.expert_ids},
        {geometry.expert_coefficient_bytes, &step.expert_coefficients},
        {4, &step.shared_coefficient},
        {geometry.expert_hidden_bytes, &step.expert_hidden},
        {geometry.hidden_bytes, &step.final_hidden},
        {geometry.logits_bytes, &step.logits},
        {geometry.argmax_value_bytes, &step.argmax_values},
        {geometry.argmax_index_bytes, &step.argmax_indices},
        {geometry.token_id_bytes, &step.token_id},
    };
    for (const auto& allocation : allocations) {
        if (!allocate_zeroed(device, allocation.size, *allocation.buffer)) {
            return {.error = DecodeStepError::BufferAllocationFailed, .step = std::nullopt};
        }
    }
    auto state =
        allocate_state_slot(device, geometry, capacity, step.schedule, step.schedule.data());
    if (!state) {
        return {.error = state.error, .step = std::nullopt};
    }
    step.state = std::move(*state.slot);
    return {.error = DecodeStepError::None, .step = std::move(step)};
}

DecodeStateSlotResult create_decode_state_slot(const MetalDevice& device, const DecodeStep& step) {
    return allocate_state_slot(device, step.geometry, step.capacity, step.schedule,
                               step.schedule.data());
}

bool decode_state_slot_compatible(const DecodeStep& step, const DecodeStateSlot& state) noexcept {
    return step.capacity != 0 && state.capacity == step.capacity &&
           state.schedule_identity == step.schedule.data() &&
           state.layers.size() == step.schedule.size();
}

bool decode_state_slot_complete(const DecodeStep& step, const DecodeStateSlot& state) noexcept {
    if (!decode_state_slot_compatible(step, state)) {
        return false;
    }
    const auto enough = [](const MetalBuffer& buffer, std::uint64_t bytes) {
        return buffer && buffer.size_bytes() >= bytes;
    };
    for (std::size_t layer = 0; layer < step.schedule.size(); ++layer) {
        const DecodeLayerState& layer_state = state.layers[layer];
        if (step.schedule[layer] == model::qwen36::LayerKind::GatedDelta) {
            if (!enough(layer_state.first, step.geometry.gdn_conv_state_bytes) ||
                !enough(layer_state.first_out, step.geometry.gdn_conv_state_bytes) ||
                !enough(layer_state.second, step.geometry.gdn_recurrent_state_bytes) ||
                !enough(layer_state.second_out, step.geometry.gdn_recurrent_state_bytes)) {
                return false;
            }
        } else if (!enough(layer_state.first, step.geometry.attn_cache_bytes) ||
                   !enough(layer_state.second, step.geometry.attn_cache_bytes)) {
            return false;
        }
    }
    return true;
}

bool decode_state_slot_available(const DecodeStep& step, const DecodeStateSlot& state) noexcept {
    return decode_state_slot_compatible(step, state) &&
           state.status == DecodeStateSlotStatus::Ready && state.active_transfer_generation == 0 &&
           state.active_transfer_arena == nullptr && state.active_transfer_positions == 0 &&
           state.active_transfer_offset_bytes == 0 && state.active_transfer_state_bytes == 0 &&
           state.active_reset_generation == 0 && state.active_reset_segments == 0 &&
           state.active_reset_bytes == 0;
}

bool decode_state_slot_ready(const DecodeStep& step, const DecodeStateSlot& state) noexcept {
    return decode_state_slot_available(step, state) && decode_state_slot_complete(step, state);
}

MetalCommandError encode_token(DecodeStep& step, DecodeStateSlot& state, MetalComputePass& pass,
                               std::uint32_t context) {
    if (!decode_state_slot_compatible(step, state)) {
        return MetalCommandError::StateSlotMismatch;
    }
    if (!context_in_capacity(context, step.capacity)) {
        return MetalCommandError::ContextOutOfRange;
    }
    if (!decode_state_slot_ready(step, state)) {
        return MetalCommandError::StateSlotMismatch;
    }
    Encoder encode{.pass = pass, .image = *step.image, .offsets = step.tensor_offsets};
    const DecodeDispatch& shape = step.geometry.dispatch;
    const std::uint64_t hidden_bytes = step.geometry.hidden_bytes;

    encode.pipeline(step.pipelines.embed);
    encode.quantized(step.bindings.embedding, 0);
    encode.buffer(step.token_id, 0, 3);
    encode.buffer(step.input, 0, 4);
    encode.dispatch(shape.embed);

    for (std::size_t layer = 0; layer < step.schedule.size(); ++layer) {
        const LayerBindings& bound = step.bindings.layers[layer];
        DecodeLayerState& layer_state = state.layers[layer];
        const MetalBuffer& layer_input = layer == 0 ? step.input : step.layer_stream;
        const std::uint64_t input_offset = layer == 0 ? 0 : (layer - 1) * hidden_bytes;
        const std::uint64_t layer_offset = layer * hidden_bytes;

        encode.barrier();
        encode.pipeline(step.pipelines.rms);
        encode.buffer(layer_input, input_offset, 0);
        encode.weight(bound.input_norm, 1);
        encode.buffer(step.normed, 0, 2);
        encode.dispatch(shape.rms);
        encode.barrier();

        if (bound.kind == model::qwen36::LayerKind::GatedDelta) {
            const MetalBuffer& conv_in =
                layer_state.swapped ? layer_state.first_out : layer_state.first;
            const MetalBuffer& conv_out =
                layer_state.swapped ? layer_state.first : layer_state.first_out;
            const MetalBuffer& recurrent_in =
                layer_state.swapped ? layer_state.second_out : layer_state.second;
            const MetalBuffer& recurrent_out =
                layer_state.swapped ? layer_state.second : layer_state.second_out;
            encode.pipeline(step.pipelines.gdn_project);
            encode.buffer(step.normed, 0, 0);
            encode.quantized(bound.gated_delta.qkv, 1);
            encode.quantized(bound.gated_delta.z, 4);
            encode.quantized(bound.gated_delta.b, 7);
            encode.quantized(bound.gated_delta.a, 10);
            encode.buffer(step.gdn_projection, 0, 13);
            encode.dispatch(shape.gdn_project);
            encode.barrier();
            encode.pipeline(step.pipelines.gdn_prepare);
            encode.buffer(step.gdn_projection, 0, 0);
            encode.buffer(conv_in, 0, 1);
            encode.weight(bound.gated_delta.conv_weight, 2);
            encode.buffer(step.gdn_qk, 0, 3);
            encode.buffer(step.gdn_value, 0, 4);
            encode.buffer(step.gdn_z, 0, 5);
            encode.buffer(conv_out, 0, 6);
            encode.dispatch(shape.gdn_prepare);
            encode.barrier();
            encode.pipeline(step.pipelines.gdn_recurrence);
            encode.buffer(step.gdn_qk, 0, 0);
            encode.buffer(step.gdn_value, 0, 1);
            encode.buffer(step.gdn_projection, 0, 2);
            encode.weight(bound.gated_delta.a_log, 3);
            encode.weight(bound.gated_delta.dt_bias, 4);
            encode.buffer(recurrent_in, 0, 5);
            encode.buffer(step.gdn_y, 0, 6);
            encode.buffer(recurrent_out, 0, 7);
            encode.dispatch(MetalSize{.width = 1,
                                      .height = shape.gdn_recurrence.dimension_groups,
                                      .depth = shape.gdn_recurrence.head_groups},
                            MetalSize{.width = shape.gdn_recurrence.lane_threads,
                                      .height = shape.gdn_recurrence.dimension_threads,
                                      .depth = 1});
            encode.barrier();
            encode.pipeline(step.pipelines.gdn_gate_norm);
            encode.buffer(step.gdn_y, 0, 0);
            encode.buffer(step.gdn_z, 0, 1);
            encode.weight(bound.gated_delta.norm_weight, 2);
            encode.buffer(step.gdn_gated, 0, 3);
            encode.dispatch(shape.gdn_gate_norm);
            encode.barrier();
            encode.pipeline(step.pipelines.out_projection);
            encode.buffer(step.gdn_gated, 0, 0);
            encode.quantized(bound.gated_delta.out, 1);
            encode.buffer(step.branch_stream, layer_offset, 4);
            encode.dispatch(shape.out_projection);
        } else {
            encode.pipeline(step.pipelines.attn_project);
            encode.buffer(step.normed, 0, 0);
            encode.quantized(bound.attention.query, 1);
            encode.quantized(bound.attention.key, 4);
            encode.quantized(bound.attention.value, 7);
            encode.buffer(step.attn_projection, 0, 10);
            encode.dispatch(shape.attn_project);
            encode.barrier();
            encode.pipeline(step.pipelines.attn_qk_rope);
            encode.buffer(step.attn_projection, 0, 0);
            encode.weight(bound.attention.query_norm, 1);
            encode.weight(bound.attention.key_norm, 2);
            encode.buffer(step.attn_query, 0, 3);
            encode.buffer(step.attn_gate, 0, 4);
            encode.buffer(layer_state.first, 0, 5);
            encode.buffer(layer_state.second, 0, 6);
            encode.constant(context, 7);
            encode.constant(step.capacity, 8);
            encode.dispatch(shape.attn_qk_rope);
            encode.barrier();
            encode_attention_decode(encode, step, layer_state.first, layer_state.second, context);
            encode.barrier();
            encode.pipeline(step.pipelines.out_projection);
            encode.buffer(step.attn_attended, 0, 0);
            encode.quantized(bound.attention.out, 1);
            encode.buffer(step.branch_stream, layer_offset, 4);
            encode.dispatch(shape.out_projection);
        }

        encode.barrier();
        encode.pipeline(step.pipelines.residual_rms);
        encode.buffer(layer_input, input_offset, 0);
        encode.buffer(step.branch_stream, layer_offset, 1);
        encode.buffer(step.residual_stream, layer_offset, 2);
        encode.weight(bound.post_norm, 3);
        encode.buffer(step.normed, 0, 4);
        encode.dispatch(shape.rms);
        encode.barrier();
        encode.pipeline(step.pipelines.router);
        encode.buffer(step.normed, 0, 0);
        encode.quantized(bound.router, 1);
        encode.quantized(bound.shared_router, 4);
        encode.buffer(step.router_logits, 0, 7);
        encode.dispatch(shape.router);
        encode.barrier();
        encode.pipeline(step.pipelines.router_select);
        encode.buffer(step.router_logits, 0, 0);
        encode.buffer(step.expert_ids, 0, 1);
        encode.buffer(step.expert_coefficients, 0, 2);
        encode.buffer(step.shared_coefficient, 0, 3);
        encode.constant(shape.top_experts, 4);
        encode.dispatch(shape.router_select);
        encode.barrier();
        encode.pipeline(step.pipelines.grouped_upgate);
        encode.buffer(step.normed, 0, 0);
        encode.buffer(step.expert_ids, 0, 1);
        encode.quantized(bound.expert_gate, 2);
        encode.quantized(bound.expert_up, 5);
        encode.quantized(bound.shared_gate, 8);
        encode.quantized(bound.shared_up, 11);
        encode.buffer(step.expert_hidden, 0, 14);
        encode.dispatch(shape.grouped_upgate);
        encode.barrier();
        encode.pipeline(step.pipelines.grouped_down_res);
        encode.buffer(step.expert_hidden, 0, 0);
        encode.buffer(step.expert_ids, 0, 1);
        encode.buffer(step.expert_coefficients, 0, 2);
        encode.buffer(step.shared_coefficient, 0, 3);
        encode.quantized(bound.expert_down, 4);
        encode.quantized(bound.shared_down, 7);
        encode.buffer(step.moe_stream, layer_offset, 10);
        encode.buffer(step.residual_stream, layer_offset, 11);
        encode.buffer(step.layer_stream, layer_offset, 12);
        encode.dispatch(shape.grouped_down_res);
    }

    encode.barrier();
    encode.pipeline(step.pipelines.rms);
    encode.buffer(step.layer_stream, (step.schedule.size() - 1) * hidden_bytes, 0);
    encode.weight(step.bindings.final_norm, 1);
    encode.buffer(step.final_hidden, 0, 2);
    encode.dispatch(shape.rms);
    encode.barrier();
    encode.pipeline(step.pipelines.lmhead);
    encode.buffer(step.final_hidden, 0, 0);
    encode.quantized(step.bindings.head, 1);
    encode.buffer(step.logits, 0, 4);
    encode.dispatch(shape.lmhead);
    encode.barrier();
    encode.pipeline(step.pipelines.argmax_stage1);
    encode.buffer(step.logits, 0, 0);
    encode.constant(shape.vocabulary_rows, 1);
    encode.buffer(step.argmax_values, 0, 2);
    encode.buffer(step.argmax_indices, 0, 3);
    encode.dispatch(shape.argmax_stage1);
    encode.barrier();
    encode.pipeline(step.pipelines.argmax_stage2);
    encode.buffer(step.argmax_values, 0, 0);
    encode.buffer(step.argmax_indices, 0, 1);
    encode.buffer(step.token_id, 0, 2);
    encode.dispatch(1, 1);
    return encode.error;
}

MetalCommandError encode_token(DecodeStep& step, MetalComputePass& pass, std::uint32_t context) {
    return encode_token(step, step.state, pass, context);
}

void advance_decode_state(const DecodeStep& step, DecodeStateSlot& state) {
    if (!decode_state_slot_available(step, state)) {
        return;
    }
    for (std::size_t layer = 0; layer < step.schedule.size(); ++layer) {
        if (step.schedule[layer] == model::qwen36::LayerKind::GatedDelta) {
            state.layers[layer].swapped = !state.layers[layer].swapped;
        }
    }
}

void advance_decode_state(DecodeStep& step) {
    advance_decode_state(step, step.state);
}

} // namespace tatara::runtime
