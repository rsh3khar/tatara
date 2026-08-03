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

// Reference view over one stream's token scratch; built from the step's
// own buffers or from a DecodeStreamScratch.
struct ScratchView {
    const MetalBuffer& input;
    const MetalBuffer& normed;
    const MetalBuffer& branch_stream;
    const MetalBuffer& residual_stream;
    const MetalBuffer& moe_stream;
    const MetalBuffer& layer_stream;
    const MetalBuffer& gdn_projection;
    const MetalBuffer& gdn_qk;
    const MetalBuffer& gdn_value;
    const MetalBuffer& gdn_z;
    const MetalBuffer& gdn_y;
    const MetalBuffer& gdn_gated;
    const MetalBuffer& attn_projection;
    const MetalBuffer& attn_query;
    const MetalBuffer& attn_gate;
    const MetalBuffer& attn_attended;
    const MetalBuffer& attn_partials;
    const MetalBuffer& attn_weights;
    const MetalBuffer& router_logits;
    const MetalBuffer& expert_ids;
    const MetalBuffer& expert_coefficients;
    const MetalBuffer& shared_coefficient;
    const MetalBuffer& expert_hidden;
    const MetalBuffer& final_hidden;
    const MetalBuffer& logits;
    const MetalBuffer& argmax_values;
    const MetalBuffer& argmax_indices;
    const MetalBuffer& token_id;
};

template <typename Owner>
ScratchView scratch_view(const Owner& o) {
    return ScratchView{o.input, o.normed, o.branch_stream, o.residual_stream,
                       o.moe_stream, o.layer_stream, o.gdn_projection,
                       o.gdn_qk, o.gdn_value, o.gdn_z, o.gdn_y, o.gdn_gated,
                       o.attn_projection, o.attn_query, o.attn_gate,
                       o.attn_attended, o.attn_partials, o.attn_weights,
                       o.router_logits, o.expert_ids, o.expert_coefficients,
                       o.shared_coefficient, o.expert_hidden, o.final_hidden,
                       o.logits, o.argmax_values, o.argmax_indices,
                       o.token_id};
}

void encode_attention_decode(Encoder& encode, DecodeStep& step, const ScratchView& scratch,
                             const MetalBuffer& keys, const MetalBuffer& values,
                             std::uint32_t context) {
    const DecodeDispatch& shape = step.geometry.dispatch;
    const AttentionSplitDispatch& split = shape.attention_split;
    const std::uint32_t partitions = attention_partitions(context);
    if (partitions == 1) {
        encode.pipeline(step.pipelines.attention_decode);
        encode.buffer(scratch.attn_query, 0, 0);
        encode.buffer(scratch.attn_gate, 0, 1);
        encode.buffer(keys, 0, 2);
        encode.buffer(values, 0, 3);
        encode.constant(context, 4);
        encode.buffer(scratch.attn_attended, 0, 5);
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
        encode.buffer(scratch.attn_query, 0, 0);
        encode.buffer(keys, 0, 1);
        encode.buffer(values, 0, 2);
        encode.buffer(scratch.attn_partials, 0, 3);
        encode.buffer(scratch.attn_weights, 0, 4);
        encode.buffer(scratch.attn_weights, scalar_plane_bytes, 5);
        encode.constant(context, 6);
        encode.constant(step.capacity, 7);
        encode.constant(vector_blocks, 8);
        encode.dispatch(split.value_groups, kSimdgroupThreads,
                        vector_blocks, split.value_head_threads);
        encode.barrier();
        encode.pipeline(step.pipelines.attention_vector_combine);
        encode.buffer(scratch.attn_partials, 0, 0);
        encode.buffer(scratch.attn_weights, 0, 1);
        encode.buffer(scratch.attn_weights, scalar_plane_bytes, 2);
        encode.buffer(scratch.attn_gate, 0, 3);
        encode.constant(vector_blocks, 4);
        encode.buffer(scratch.attn_attended, 0, 5);
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
        encode.buffer(scratch.attn_query, 0, 0);
        encode.buffer(keys, 0, 1);
        encode.buffer(values, 0, 2);
        encode.constant(context, 3);
        encode.constant(step.capacity, 4);
        encode.constant(kAttentionPartition, 5);
        encode.buffer(scratch.attn_partials, 0, 6);
        encode.constant(partitions, 7);
        encode.dispatch(split.value_groups, split.score_threads, partitions,
                        split.score_cohort_threads);
        encode.barrier();
    } else {
        encode.pipeline(step.pipelines.attention_scores);
        encode.buffer(scratch.attn_query, 0, 0);
        encode.buffer(keys, 0, 1);
        encode.constant(context, 2);
        encode.constant(step.capacity, 3);
        encode.constant(kAttentionPartition, 4);
        encode.buffer(scratch.attn_weights, 0, 5);
        encode.constant(partitions, 6);
        encode.dispatch(split.score_groups, split.score_threads, partitions,
                        split.score_cohort_threads);
        encode.barrier();
        encode.pipeline(step.pipelines.attention_values);
        encode.buffer(scratch.attn_weights, 0, 0);
        encode.buffer(values, 0, 1);
        encode.constant(context, 2);
        encode.constant(step.capacity, 3);
        encode.constant(kAttentionPartition, 4);
        encode.buffer(scratch.attn_partials, 0, 5);
        encode.constant(partitions, 6);
        encode.dispatch(split.value_groups, split.value_lane_threads, partitions,
                        split.value_dimension_threads, split.value_head_threads);
        encode.barrier();
    }
    encode.pipeline(step.pipelines.attention_combine);
    encode.buffer(scratch.attn_partials, 0, 0);
    encode.buffer(scratch.attn_gate, 0, 1);
    encode.constant(partitions, 2);
    encode.buffer(scratch.attn_attended, 0, 3);
    encode.dispatch(shape.attention_head);
}

} // namespace

namespace {

bool allocate_stream_scratch(const MetalDevice& device, const DecodeGeometry& geometry,
                             DecodeStreamScratch& scratch,
                             std::uint64_t rows = 1) {
    const struct {
        std::uint64_t size;
        MetalBuffer* buffer;
    } allocations[] = {
        {geometry.hidden_bytes, &scratch.input},
        {geometry.hidden_bytes, &scratch.normed},
        {geometry.layer_stream_bytes, &scratch.branch_stream},
        {geometry.layer_stream_bytes, &scratch.residual_stream},
        {geometry.layer_stream_bytes, &scratch.moe_stream},
        {geometry.layer_stream_bytes, &scratch.layer_stream},
        {geometry.gdn_projection_bytes, &scratch.gdn_projection},
        {geometry.gdn_qk_bytes, &scratch.gdn_qk},
        {geometry.gdn_value_bytes, &scratch.gdn_value},
        {geometry.gdn_gate_bytes, &scratch.gdn_z},
        {geometry.gdn_value_bytes, &scratch.gdn_y},
        {geometry.gdn_gate_bytes, &scratch.gdn_gated},
        {geometry.attn_projection_bytes, &scratch.attn_projection},
        {geometry.attn_query_bytes, &scratch.attn_query},
        {geometry.attn_query_bytes, &scratch.attn_gate},
        {geometry.attn_query_bytes, &scratch.attn_attended},
        {geometry.attn_record_scratch_bytes, &scratch.attn_partials},
        {geometry.attn_record_scratch_bytes, &scratch.attn_weights},
        {geometry.router_logits_bytes, &scratch.router_logits},
        {geometry.expert_id_bytes, &scratch.expert_ids},
        {geometry.expert_coefficient_bytes, &scratch.expert_coefficients},
        {4, &scratch.shared_coefficient},
        {geometry.expert_hidden_bytes, &scratch.expert_hidden},
        {geometry.hidden_bytes, &scratch.final_hidden},
        {geometry.logits_bytes, &scratch.logits},
        {geometry.argmax_value_bytes, &scratch.argmax_values},
        {geometry.argmax_index_bytes, &scratch.argmax_indices},
        {geometry.token_id_bytes, &scratch.token_id},
    };
    for (const auto& allocation : allocations) {
        if (!allocate_zeroed(device, allocation.size * rows, *allocation.buffer)) {
            return false;
        }
    }
    return true;
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
    DecodeStreamScratch scratch;
    if (!allocate_stream_scratch(device, geometry, scratch)) {
        return {.error = DecodeStepError::BufferAllocationFailed, .step = std::nullopt};
    }
    step.input = std::move(scratch.input);
    step.normed = std::move(scratch.normed);
    step.branch_stream = std::move(scratch.branch_stream);
    step.residual_stream = std::move(scratch.residual_stream);
    step.moe_stream = std::move(scratch.moe_stream);
    step.layer_stream = std::move(scratch.layer_stream);
    step.gdn_projection = std::move(scratch.gdn_projection);
    step.gdn_qk = std::move(scratch.gdn_qk);
    step.gdn_value = std::move(scratch.gdn_value);
    step.gdn_z = std::move(scratch.gdn_z);
    step.gdn_y = std::move(scratch.gdn_y);
    step.gdn_gated = std::move(scratch.gdn_gated);
    step.attn_projection = std::move(scratch.attn_projection);
    step.attn_query = std::move(scratch.attn_query);
    step.attn_gate = std::move(scratch.attn_gate);
    step.attn_attended = std::move(scratch.attn_attended);
    step.attn_partials = std::move(scratch.attn_partials);
    step.attn_weights = std::move(scratch.attn_weights);
    step.router_logits = std::move(scratch.router_logits);
    step.expert_ids = std::move(scratch.expert_ids);
    step.expert_coefficients = std::move(scratch.expert_coefficients);
    step.shared_coefficient = std::move(scratch.shared_coefficient);
    step.expert_hidden = std::move(scratch.expert_hidden);
    step.final_hidden = std::move(scratch.final_hidden);
    step.logits = std::move(scratch.logits);
    step.argmax_values = std::move(scratch.argmax_values);
    step.argmax_indices = std::move(scratch.argmax_indices);
    step.token_id = std::move(scratch.token_id);
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

DecodeStateSlotPoolResult create_decode_state_slot_pool(const MetalDevice& device,
                                                        const DecodeStep& step,
                                                        std::uint32_t count) {
    DecodeStateSlotPoolResult result{
        .error = DecodeStepError::None, .pool = std::nullopt, .slots = {}};
    if (!device) {
        result.error = DecodeStepError::InvalidDevice;
        return result;
    }
    if (count == 0 || step.capacity == 0 || step.schedule.empty()) {
        result.error = DecodeStepError::OffsetCountMismatch;
        return result;
    }
    const auto make_pool = [&](std::uint64_t stripe_bytes, std::uint32_t stripes,
                               MetalBuffer& pool, std::uint64_t& stride) {
        auto pooled = backend::metal::create_striped_pool_buffer(device, stripe_bytes, stripes);
        if (!pooled || pooled.buffer->size_bytes() > 0xffffffffull * 2) {
            return false;
        }
        std::memset(pooled.buffer->contents(), 0,
                    static_cast<std::size_t>(pooled.buffer->size_bytes()));
        stride = pooled.stripe_stride_bytes;
        pool = std::move(*pooled.buffer);
        return true;
    };
    DecodeStatePool pool;
    pool.stripes = count;
    pool.layers.reserve(step.schedule.size());
    result.slots.resize(count);
    for (std::uint32_t slot = 0; slot < count; ++slot) {
        result.slots[slot].capacity = step.capacity;
        result.slots[slot].schedule_identity = step.schedule.data();
        result.slots[slot].layers.reserve(step.schedule.size());
    }
    for (const model::qwen36::LayerKind kind : step.schedule) {
        const bool gated = kind == model::qwen36::LayerKind::GatedDelta;
        const std::uint64_t first_bytes =
            gated ? step.geometry.gdn_conv_state_bytes : step.geometry.attn_cache_bytes;
        const std::uint64_t second_bytes =
            gated ? step.geometry.gdn_recurrent_state_bytes : step.geometry.attn_cache_bytes;
        const std::uint32_t stripes = gated ? 2u * count : count;
        DecodeStatePoolLayer layer;
        if (!make_pool(first_bytes, stripes, layer.first, layer.first_stride) ||
            !make_pool(second_bytes, stripes, layer.second, layer.second_stride)) {
            result.error = DecodeStepError::BufferAllocationFailed;
            result.slots.clear();
            return result;
        }
        for (std::uint32_t slot = 0; slot < count; ++slot) {
            DecodeLayerState state{
                .first = {}, .first_out = {}, .second = {}, .second_out = {}, .swapped = false};
            const auto window = [&](const MetalBuffer& source, std::uint64_t stripe,
                                    std::uint64_t stride, std::uint64_t bytes,
                                    MetalBuffer& view) {
                auto carved =
                    backend::metal::create_buffer_window(source, stripe * stride, bytes);
                if (!carved) {
                    return false;
                }
                view = std::move(*carved.buffer);
                return true;
            };
            const std::uint64_t base = gated ? std::uint64_t{slot} * 2 : std::uint64_t{slot};
            if (!window(layer.first, base, layer.first_stride, first_bytes, state.first) ||
                !window(layer.second, base, layer.second_stride, second_bytes, state.second) ||
                (gated &&
                 (!window(layer.first, base + 1, layer.first_stride, first_bytes,
                          state.first_out) ||
                  !window(layer.second, base + 1, layer.second_stride, second_bytes,
                          state.second_out)))) {
                result.error = DecodeStepError::BufferAllocationFailed;
                result.slots.clear();
                return result;
            }
            result.slots[slot].layers.push_back(std::move(state));
        }
        pool.layers.push_back(std::move(layer));
    }
    result.pool = std::move(pool);
    return result;
}

DecodeStreamScratchResult create_decode_stream_scratch(const MetalDevice& device,
                                                       const DecodeStep& step) {
    DecodeStreamScratch scratch;
    if (!allocate_stream_scratch(device, step.geometry, scratch)) {
        return {.error = DecodeStepError::BufferAllocationFailed, .scratch = std::nullopt};
    }
    return {.error = DecodeStepError::None, .scratch = std::move(scratch)};
}

DecodeBatchScratchResult create_decode_batch_scratch(const MetalDevice& device,
                                                     const DecodeStep& step,
                                                     std::uint32_t rows) {
    if (rows == 0 || rows > 16) {
        return {.error = DecodeStepError::OffsetCountMismatch, .scratch = std::nullopt};
    }
    DecodeBatchScratch batch;
    batch.rows = rows;
    if (!allocate_stream_scratch(device, step.geometry, batch.slabs, rows)) {
        return {.error = DecodeStepError::BufferAllocationFailed, .scratch = std::nullopt};
    }
    const std::uint64_t experts = step.geometry.router_logits_bytes / 4 - 1;
    const std::uint64_t active = step.geometry.dispatch.top_experts;
    const std::uint64_t union_table_bytes =
        experts * 4 + experts * 8 + 4 + (experts + 1) * 4;
    const std::uint64_t parts_bytes = std::uint64_t{rows} * (active + 1) *
                                      (step.geometry.hidden_bytes / 2) * 4;
    const std::uint64_t offsets_bytes = step.schedule.size() * 16 * 4 * 4 + 64;
    if (!allocate_zeroed(device, union_table_bytes, batch.union_table) ||
        !allocate_zeroed(device, parts_bytes, batch.moe_parts) ||
        !allocate_zeroed(device, offsets_bytes, batch.state_offsets)) {
        return {.error = DecodeStepError::BufferAllocationFailed, .scratch = std::nullopt};
    }
    return {.error = DecodeStepError::None, .scratch = std::move(batch)};
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

namespace {

void encode_stream_embed(Encoder& encode, DecodeStep& step, const ScratchView& scratch) {
    encode.pipeline(step.pipelines.embed);
    encode.quantized(step.bindings.embedding, 0);
    encode.buffer(scratch.token_id, 0, 3);
    encode.buffer(scratch.input, 0, 4);
    encode.dispatch(step.geometry.dispatch.embed);
}

void encode_stream_layer(Encoder& encode, DecodeStep& step, const ScratchView& scratch,
                         DecodeStateSlot& state, std::size_t layer, std::uint32_t context) {
    const DecodeDispatch& shape = step.geometry.dispatch;
    const std::uint64_t hidden_bytes = step.geometry.hidden_bytes;
    const LayerBindings& bound = step.bindings.layers[layer];
    DecodeLayerState& layer_state = state.layers[layer];
    const MetalBuffer& layer_input = layer == 0 ? scratch.input : scratch.layer_stream;
    const std::uint64_t input_offset = layer == 0 ? 0 : (layer - 1) * hidden_bytes;
    const std::uint64_t layer_offset = layer * hidden_bytes;

    encode.barrier();
    encode.pipeline(step.pipelines.rms);
    encode.buffer(layer_input, input_offset, 0);
    encode.weight(bound.input_norm, 1);
    encode.buffer(scratch.normed, 0, 2);
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
        encode.buffer(scratch.normed, 0, 0);
        encode.quantized(bound.gated_delta.qkv, 1);
        encode.quantized(bound.gated_delta.z, 4);
        encode.quantized(bound.gated_delta.b, 7);
        encode.quantized(bound.gated_delta.a, 10);
        encode.buffer(scratch.gdn_projection, 0, 13);
        encode.dispatch(shape.gdn_project);
        encode.barrier();
        encode.pipeline(step.pipelines.gdn_prepare);
        encode.buffer(scratch.gdn_projection, 0, 0);
        encode.buffer(conv_in, 0, 1);
        encode.weight(bound.gated_delta.conv_weight, 2);
        encode.buffer(scratch.gdn_qk, 0, 3);
        encode.buffer(scratch.gdn_value, 0, 4);
        encode.buffer(scratch.gdn_z, 0, 5);
        encode.buffer(conv_out, 0, 6);
        encode.dispatch(shape.gdn_prepare);
        encode.barrier();
        encode.pipeline(step.pipelines.gdn_recurrence);
        encode.buffer(scratch.gdn_qk, 0, 0);
        encode.buffer(scratch.gdn_value, 0, 1);
        encode.buffer(scratch.gdn_projection, 0, 2);
        encode.weight(bound.gated_delta.a_log, 3);
        encode.weight(bound.gated_delta.dt_bias, 4);
        encode.buffer(recurrent_in, 0, 5);
        encode.buffer(scratch.gdn_y, 0, 6);
        encode.buffer(recurrent_out, 0, 7);
        encode.dispatch(MetalSize{.width = 1,
                                  .height = shape.gdn_recurrence.dimension_groups,
                                  .depth = shape.gdn_recurrence.head_groups},
                        MetalSize{.width = shape.gdn_recurrence.lane_threads,
                                  .height = shape.gdn_recurrence.dimension_threads,
                                  .depth = 1});
        encode.barrier();
        encode.pipeline(step.pipelines.gdn_gate_norm);
        encode.buffer(scratch.gdn_y, 0, 0);
        encode.buffer(scratch.gdn_z, 0, 1);
        encode.weight(bound.gated_delta.norm_weight, 2);
        encode.buffer(scratch.gdn_gated, 0, 3);
        encode.dispatch(shape.gdn_gate_norm);
        encode.barrier();
        encode.pipeline(step.pipelines.out_projection);
        encode.buffer(scratch.gdn_gated, 0, 0);
        encode.quantized(bound.gated_delta.out, 1);
        encode.buffer(scratch.branch_stream, layer_offset, 4);
        encode.dispatch(shape.out_projection);
    } else {
        encode.pipeline(step.pipelines.attn_project);
        encode.buffer(scratch.normed, 0, 0);
        encode.quantized(bound.attention.query, 1);
        encode.quantized(bound.attention.key, 4);
        encode.quantized(bound.attention.value, 7);
        encode.buffer(scratch.attn_projection, 0, 10);
        encode.dispatch(shape.attn_project);
        encode.barrier();
        encode.pipeline(step.pipelines.attn_qk_rope);
        encode.buffer(scratch.attn_projection, 0, 0);
        encode.weight(bound.attention.query_norm, 1);
        encode.weight(bound.attention.key_norm, 2);
        encode.buffer(scratch.attn_query, 0, 3);
        encode.buffer(scratch.attn_gate, 0, 4);
        encode.buffer(layer_state.first, 0, 5);
        encode.buffer(layer_state.second, 0, 6);
        encode.constant(context, 7);
        encode.constant(step.capacity, 8);
        encode.dispatch(shape.attn_qk_rope);
        encode.barrier();
        encode_attention_decode(encode, step, scratch, layer_state.first, layer_state.second,
                                context);
        encode.barrier();
        encode.pipeline(step.pipelines.out_projection);
        encode.buffer(scratch.attn_attended, 0, 0);
        encode.quantized(bound.attention.out, 1);
        encode.buffer(scratch.branch_stream, layer_offset, 4);
        encode.dispatch(shape.out_projection);
    }

    encode.barrier();
    encode.pipeline(step.pipelines.residual_rms);
    encode.buffer(layer_input, input_offset, 0);
    encode.buffer(scratch.branch_stream, layer_offset, 1);
    encode.buffer(scratch.residual_stream, layer_offset, 2);
    encode.weight(bound.post_norm, 3);
    encode.buffer(scratch.normed, 0, 4);
    encode.dispatch(shape.rms);
    encode.barrier();
    encode.pipeline(step.pipelines.router);
    encode.buffer(scratch.normed, 0, 0);
    encode.quantized(bound.router, 1);
    encode.quantized(bound.shared_router, 4);
    encode.buffer(scratch.router_logits, 0, 7);
    encode.dispatch(shape.router);
    encode.barrier();
    encode.pipeline(step.pipelines.router_select);
    encode.buffer(scratch.router_logits, 0, 0);
    encode.buffer(scratch.expert_ids, 0, 1);
    encode.buffer(scratch.expert_coefficients, 0, 2);
    encode.buffer(scratch.shared_coefficient, 0, 3);
    encode.constant(shape.top_experts, 4);
    encode.dispatch(shape.router_select);
    encode.barrier();
    encode.pipeline(step.pipelines.grouped_upgate);
    encode.buffer(scratch.normed, 0, 0);
    encode.buffer(scratch.expert_ids, 0, 1);
    encode.quantized(bound.expert_gate, 2);
    encode.quantized(bound.expert_up, 5);
    encode.quantized(bound.shared_gate, 8);
    encode.quantized(bound.shared_up, 11);
    encode.buffer(scratch.expert_hidden, 0, 14);
    encode.dispatch(shape.grouped_upgate);
    encode.barrier();
    encode.pipeline(step.pipelines.grouped_down_res);
    encode.buffer(scratch.expert_hidden, 0, 0);
    encode.buffer(scratch.expert_ids, 0, 1);
    encode.buffer(scratch.expert_coefficients, 0, 2);
    encode.buffer(scratch.shared_coefficient, 0, 3);
    encode.quantized(bound.expert_down, 4);
    encode.quantized(bound.shared_down, 7);
    encode.buffer(scratch.moe_stream, layer_offset, 10);
    encode.buffer(scratch.residual_stream, layer_offset, 11);
    encode.buffer(scratch.layer_stream, layer_offset, 12);
    encode.dispatch(shape.grouped_down_res);
}

void encode_stream_head(Encoder& encode, DecodeStep& step, const ScratchView& scratch) {
    const DecodeDispatch& shape = step.geometry.dispatch;
    const std::uint64_t hidden_bytes = step.geometry.hidden_bytes;
    encode.barrier();
    encode.pipeline(step.pipelines.rms);
    encode.buffer(scratch.layer_stream, (step.schedule.size() - 1) * hidden_bytes, 0);
    encode.weight(step.bindings.final_norm, 1);
    encode.buffer(scratch.final_hidden, 0, 2);
    encode.dispatch(shape.rms);
    encode.barrier();
    encode.pipeline(step.pipelines.lmhead);
    encode.buffer(scratch.final_hidden, 0, 0);
    encode.quantized(step.bindings.head, 1);
    encode.buffer(scratch.logits, 0, 4);
    encode.dispatch(shape.lmhead);
    encode.barrier();
    encode.pipeline(step.pipelines.argmax_stage1);
    encode.buffer(scratch.logits, 0, 0);
    encode.constant(shape.vocabulary_rows, 1);
    encode.buffer(scratch.argmax_values, 0, 2);
    encode.buffer(scratch.argmax_indices, 0, 3);
    encode.dispatch(shape.argmax_stage1);
    encode.barrier();
    encode.pipeline(step.pipelines.argmax_stage2);
    encode.buffer(scratch.argmax_values, 0, 0);
    encode.buffer(scratch.argmax_indices, 0, 1);
    encode.buffer(scratch.token_id, 0, 2);
    encode.dispatch(1, 1);
}

MetalCommandError stream_admissible(DecodeStep& step, DecodeStateSlot& state,
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
    return MetalCommandError::None;
}

} // namespace

MetalCommandError encode_token(DecodeStep& step, DecodeStateSlot& state, MetalComputePass& pass,
                               std::uint32_t context) {
    const MetalCommandError admissible = stream_admissible(step, state, context);
    if (admissible != MetalCommandError::None) {
        return admissible;
    }
    Encoder encode{.pass = pass, .image = *step.image, .offsets = step.tensor_offsets};
    const ScratchView scratch = scratch_view(step);
    encode_stream_embed(encode, step, scratch);
    for (std::size_t layer = 0; layer < step.schedule.size(); ++layer) {
        encode_stream_layer(encode, step, scratch, state, layer, context);
    }
    encode_stream_head(encode, step, scratch);
    return encode.error;
}

MetalCommandError encode_token_group(DecodeStep& step, std::span<const DecodeStream> streams,
                                     MetalComputePass& pass) {
    if (streams.empty()) {
        return MetalCommandError::InvalidStreamGroup;
    }
    for (std::size_t index = 0; index < streams.size(); ++index) {
        const DecodeStream& stream = streams[index];
        if (stream.state == nullptr) {
            return MetalCommandError::InvalidStreamGroup;
        }
        const MetalCommandError admissible =
            stream_admissible(step, *stream.state, stream.context);
        if (admissible != MetalCommandError::None) {
            return admissible;
        }
        for (std::size_t other = 0; other < index; ++other) {
            if (streams[other].state == stream.state ||
                (stream.scratch != nullptr &&
                 streams[other].scratch == stream.scratch) ||
                (stream.scratch == nullptr &&
                 streams[other].scratch == nullptr)) {
                return MetalCommandError::InvalidStreamGroup;
            }
        }
    }
    Encoder encode{.pass = pass, .image = *step.image, .offsets = step.tensor_offsets};
    const auto view = [&step](const DecodeStream& stream) {
        return stream.scratch == nullptr ? scratch_view(step)
                                         : scratch_view(*stream.scratch);
    };
    for (const DecodeStream& stream : streams) {
        encode_stream_embed(encode, step, view(stream));
    }
    for (std::size_t layer = 0; layer < step.schedule.size(); ++layer) {
        for (const DecodeStream& stream : streams) {
            encode_stream_layer(encode, step, view(stream), *stream.state, layer,
                                stream.context);
        }
    }
    for (const DecodeStream& stream : streams) {
        encode_stream_head(encode, step, view(stream));
    }
    return encode.error;
}

namespace {

// Attention core over slab rows: identical dispatch logic to
// encode_attention_decode with per-row base offsets into the batch slabs.
void encode_attention_decode_at(Encoder& encode, DecodeStep& step,
                                const DecodeStreamScratch& slabs,
                                std::uint64_t query_offset,
                                std::uint64_t record_offset,
                                const MetalBuffer& keys, const MetalBuffer& values,
                                std::uint32_t context) {
    const DecodeDispatch& shape = step.geometry.dispatch;
    const AttentionSplitDispatch& split = shape.attention_split;
    const std::uint32_t partitions = attention_partitions(context);
    if (partitions == 1) {
        encode.pipeline(step.pipelines.attention_decode);
        encode.buffer(slabs.attn_query, query_offset, 0);
        encode.buffer(slabs.attn_gate, query_offset, 1);
        encode.buffer(keys, 0, 2);
        encode.buffer(values, 0, 3);
        encode.constant(context, 4);
        encode.buffer(slabs.attn_attended, query_offset, 5);
        encode.constant(step.capacity, 6);
        encode.dispatch(shape.attention_head);
        return;
    }
    const std::uint32_t vector_blocks = attention_vector_blocks(step, partitions);
    const bool vector_requested =
        step.pipelines.attention_split_policy ==
            DecodeAttentionSplitPolicy::IndependentHeadVector2Pass ||
        (step.pipelines.attention_split_policy ==
             DecodeAttentionSplitPolicy::AdaptiveVector2Pass &&
         context >= step.pipelines.vector_minimum_context);
    if (vector_requested && vector_blocks != 0) {
        const std::uint64_t scalar_plane_bytes =
            std::uint64_t{shape.attention_head.groups} * vector_blocks * sizeof(float);
        encode.pipeline(step.pipelines.attention_vector_part);
        encode.buffer(slabs.attn_query, query_offset, 0);
        encode.buffer(keys, 0, 1);
        encode.buffer(values, 0, 2);
        encode.buffer(slabs.attn_partials, record_offset, 3);
        encode.buffer(slabs.attn_weights, record_offset, 4);
        encode.buffer(slabs.attn_weights, record_offset + scalar_plane_bytes, 5);
        encode.constant(context, 6);
        encode.constant(step.capacity, 7);
        encode.constant(vector_blocks, 8);
        encode.dispatch(split.value_groups, kSimdgroupThreads, vector_blocks,
                        split.value_head_threads);
        encode.barrier();
        encode.pipeline(step.pipelines.attention_vector_combine);
        encode.buffer(slabs.attn_partials, record_offset, 0);
        encode.buffer(slabs.attn_weights, record_offset, 1);
        encode.buffer(slabs.attn_weights, record_offset + scalar_plane_bytes, 2);
        encode.buffer(slabs.attn_gate, query_offset, 3);
        encode.constant(vector_blocks, 4);
        encode.buffer(slabs.attn_attended, query_offset, 5);
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
         context >= step.pipelines.fused_score_value_minimum_context);
    if (fused_score_value) {
        encode.pipeline(step.pipelines.attention_scores_values_fused);
        encode.buffer(slabs.attn_query, query_offset, 0);
        encode.buffer(keys, 0, 1);
        encode.buffer(values, 0, 2);
        encode.constant(context, 3);
        encode.constant(step.capacity, 4);
        encode.constant(kAttentionPartition, 5);
        encode.buffer(slabs.attn_partials, record_offset, 6);
        encode.constant(partitions, 7);
        encode.dispatch(split.value_groups, split.score_threads, partitions,
                        split.score_cohort_threads);
        encode.barrier();
    } else {
        encode.pipeline(step.pipelines.attention_scores);
        encode.buffer(slabs.attn_query, query_offset, 0);
        encode.buffer(keys, 0, 1);
        encode.constant(context, 2);
        encode.constant(step.capacity, 3);
        encode.constant(kAttentionPartition, 4);
        encode.buffer(slabs.attn_weights, record_offset, 5);
        encode.constant(partitions, 6);
        encode.dispatch(split.score_groups, split.score_threads, partitions,
                        split.score_cohort_threads);
        encode.barrier();
        encode.pipeline(step.pipelines.attention_values);
        encode.buffer(slabs.attn_weights, record_offset, 0);
        encode.buffer(values, 0, 1);
        encode.constant(context, 2);
        encode.constant(step.capacity, 3);
        encode.constant(kAttentionPartition, 4);
        encode.buffer(slabs.attn_partials, record_offset, 5);
        encode.constant(partitions, 6);
        encode.dispatch(split.value_groups, split.value_lane_threads, partitions,
                        split.value_dimension_threads, split.value_head_threads);
        encode.barrier();
    }
    encode.pipeline(step.pipelines.attention_combine);
    encode.buffer(slabs.attn_partials, record_offset, 0);
    encode.buffer(slabs.attn_gate, query_offset, 1);
    encode.constant(partitions, 2);
    encode.buffer(slabs.attn_attended, query_offset, 3);
    encode.dispatch(shape.attention_head);
}

} // namespace

MetalCommandError encode_token_batch(DecodeStep& step, std::span<const DecodeStream> streams,
                                     const DecodeStatePool& pool, DecodeBatchScratch& batch,
                                     const DecodeBatchPipelines& kernels,
                                     MetalComputePass& pass,
                                     const MetalBuffer* chain_input) {
    const std::uint32_t rows = static_cast<std::uint32_t>(streams.size());
    if (rows == 0 || rows > batch.rows || rows > 16 ||
        pool.layers.size() != step.schedule.size() || pool.stripes == 0 ||
        !batch.state_offsets ||
        !kernels.gdn_project_ms || !kernels.gdn_outproj_ms ||
        !kernels.attn_project_ms || !kernels.lmhead_ms ||
        !kernels.embed_ms || !kernels.rms_ms || !kernels.residual_rms_ms ||
        !kernels.gate_norm_ms || !kernels.router_ms ||
        !kernels.upgate_rows_ms || !kernels.down_rows_ms ||
        !kernels.argmax_stage1_ms || !kernels.argmax_stage2_ms ||
        !kernels.gdn_prepare_ms || !kernels.gdn_recurrence_ms ||
        !kernels.attn_qk_rope_ms || !kernels.router_select_ms ||
        !kernels.attention_decode_ms) {
        return MetalCommandError::InvalidStreamGroup;
    }
    for (std::size_t index = 0; index < streams.size(); ++index) {
        if (streams[index].state == nullptr || streams[index].stripe >= pool.stripes) {
            return MetalCommandError::InvalidStreamGroup;
        }
        const MetalCommandError admissible =
            stream_admissible(step, *streams[index].state, streams[index].context);
        if (admissible != MetalCommandError::None) {
            return admissible;
        }
        for (std::size_t other = 0; other < index; ++other) {
            if (streams[other].state == streams[index].state) {
                return MetalCommandError::InvalidStreamGroup;
            }
        }
    }
    auto* offsets_table = static_cast<std::uint32_t*>(batch.state_offsets.contents());
    for (std::size_t layer = 0; layer < step.schedule.size(); ++layer) {
        const DecodeStatePoolLayer& pool_layer = pool.layers[layer];
        const bool gated = step.schedule[layer] == model::qwen36::LayerKind::GatedDelta;
        for (std::uint32_t r = 0; r < rows; ++r) {
            const DecodeLayerState& layer_state = streams[r].state->layers[layer];
            const std::uint32_t stripe = streams[r].stripe;
            std::uint32_t* entry = offsets_table + layer * 64 + std::size_t{r} * 4;
            if (gated) {
                const std::uint32_t conv_elems =
                    static_cast<std::uint32_t>(pool_layer.first_stride / 2);
                const std::uint32_t rec_elems =
                    static_cast<std::uint32_t>(pool_layer.second_stride / 4);
                const std::uint32_t in_parity = layer_state.swapped ? 1u : 0u;
                entry[0] = (2u * stripe + in_parity) * conv_elems;
                entry[1] = (2u * stripe + 1u - in_parity) * conv_elems;
                entry[2] = (2u * stripe + in_parity) * rec_elems;
                entry[3] = (2u * stripe + 1u - in_parity) * rec_elems;
            } else {
                entry[0] = stripe * static_cast<std::uint32_t>(pool_layer.first_stride / 2);
                entry[1] = streams[r].context;
                entry[2] = 0;
                entry[3] = 0;
            }
        }
    }
    std::uint32_t* bucket_rows = offsets_table + step.schedule.size() * 64;
    std::uint32_t decode_bucket_rows = 0;
    for (std::uint32_t r = 0; r < rows; ++r) {
        if (attention_partitions(streams[r].context) == 1) {
            bucket_rows[decode_bucket_rows] = r;
            ++decode_bucket_rows;
        }
    }
    const std::uint64_t bucket_offset = step.schedule.size() * 256;
    const auto clamp_tile = [rows](std::uint32_t tile) {
        return tile == 0 ? 1u : (tile > rows ? rows : tile);
    };
    const std::uint32_t project_tile = clamp_tile(batch.project_row_tile);
    const std::uint32_t project_tiles = (rows + project_tile - 1) / project_tile;
    const std::uint32_t outproj_tile = clamp_tile(batch.outproj_row_tile);
    const std::uint32_t outproj_tiles = (rows + outproj_tile - 1) / outproj_tile;
    const std::uint32_t attnproj_tile = clamp_tile(batch.attnproj_row_tile);
    const std::uint32_t attnproj_tiles = (rows + attnproj_tile - 1) / attnproj_tile;
    Encoder encode{.pass = pass, .image = *step.image, .offsets = step.tensor_offsets};
    const DecodeDispatch& shape = step.geometry.dispatch;
    const DecodeGeometry& g = step.geometry;
    const DecodeStreamScratch& s = batch.slabs;
    const std::uint64_t hidden = g.hidden_bytes;
    const std::uint64_t lanebytes = g.layer_stream_bytes;

    encode.pipeline(kernels.embed_ms);
    encode.quantized(step.bindings.embedding, 0);
    encode.buffer(chain_input != nullptr ? *chain_input : s.token_id, 0, 3);
    encode.buffer(s.input, 0, 4);
    encode.constant(static_cast<std::uint32_t>(hidden / 2), 5);
    encode.dispatch(shape.embed.groups, shape.embed.threads, rows);

    for (std::size_t layer = 0; layer < step.schedule.size(); ++layer) {
        const LayerBindings& bound = step.bindings.layers[layer];
        const std::uint64_t layer_offset = layer * hidden;
        const std::uint64_t input_inner = layer == 0 ? 0 : (layer - 1) * hidden;
        const MetalBuffer& layer_input = layer == 0 ? s.input : s.layer_stream;
        const std::uint64_t input_stride = layer == 0 ? hidden : lanebytes;

        encode.barrier();
        encode.pipeline(kernels.rms_ms);
        encode.buffer(layer_input, input_inner, 0);
        encode.weight(bound.input_norm, 1);
        encode.buffer(s.normed, 0, 2);
        encode.constant(static_cast<std::uint32_t>(input_stride / 2), 3);
        encode.constant(static_cast<std::uint32_t>(hidden / 2), 4);
        encode.dispatch(shape.rms.groups, shape.rms.threads, rows);
        if (batch.profile_double_stage == 7u) {
            encode.barrier();
            encode.dispatch(shape.rms.groups, shape.rms.threads, rows);
        }
        encode.barrier();

        if (bound.kind == model::qwen36::LayerKind::GatedDelta) {
            encode.pipeline(batch.dense_variant == 1 && kernels.gdn_project_ms_v2
                                ? kernels.gdn_project_ms_v2
                                : kernels.gdn_project_ms);
            encode.buffer(s.normed, 0, 0);
            encode.quantized(bound.gated_delta.qkv, 1);
            encode.quantized(bound.gated_delta.z, 4);
            encode.quantized(bound.gated_delta.b, 7);
            encode.quantized(bound.gated_delta.a, 10);
            encode.buffer(s.gdn_projection, 0, 13);
            encode.constant(rows, 14);
            encode.constant(project_tile, 15);
            encode.dispatch(shape.gdn_project.groups, shape.gdn_project.threads,
                            project_tiles);
            if (batch.profile_double_stage == 1u) {
                encode.barrier();
                encode.dispatch(shape.gdn_project.groups, shape.gdn_project.threads,
                                project_tiles);
            }
            encode.barrier();
            encode.pipeline(kernels.gdn_prepare_ms);
            encode.buffer(s.gdn_projection, 0, 0);
            encode.buffer(pool.layers[layer].first, 0, 1);
            encode.weight(bound.gated_delta.conv_weight, 2);
            encode.buffer(s.gdn_qk, 0, 3);
            encode.buffer(s.gdn_value, 0, 4);
            encode.buffer(s.gdn_z, 0, 5);
            encode.buffer(batch.state_offsets, layer * 256, 6);
            encode.constant(static_cast<std::uint32_t>(g.gdn_projection_bytes / 2), 7);
            encode.constant(static_cast<std::uint32_t>(g.gdn_qk_bytes / 2), 8);
            encode.constant(static_cast<std::uint32_t>(g.gdn_value_bytes / 2), 9);
            encode.constant(static_cast<std::uint32_t>(g.gdn_gate_bytes / 2), 10);
            encode.dispatch(shape.gdn_prepare.groups, shape.gdn_prepare.threads, rows);
            if (batch.profile_double_stage == 2u) {
                encode.barrier();
                encode.dispatch(shape.gdn_prepare.groups, shape.gdn_prepare.threads, rows);
            }
            encode.barrier();
            encode.pipeline(kernels.gdn_recurrence_ms);
            encode.buffer(s.gdn_qk, 0, 0);
            encode.buffer(s.gdn_value, 0, 1);
            encode.buffer(s.gdn_projection, 0, 2);
            encode.weight(bound.gated_delta.a_log, 3);
            encode.weight(bound.gated_delta.dt_bias, 4);
            encode.buffer(pool.layers[layer].second, 0, 5);
            encode.buffer(s.gdn_y, 0, 6);
            encode.buffer(batch.state_offsets, layer * 256, 7);
            encode.constant(static_cast<std::uint32_t>(g.gdn_qk_bytes / 2), 8);
            encode.constant(static_cast<std::uint32_t>(g.gdn_value_bytes / 2), 9);
            encode.constant(static_cast<std::uint32_t>(g.gdn_projection_bytes / 2), 10);
            encode.constant(static_cast<std::uint32_t>(g.gdn_value_bytes / 2), 11);
            encode.constant(shape.gdn_recurrence.head_groups, 12);
            encode.dispatch(MetalSize{.width = 1,
                                      .height = shape.gdn_recurrence.dimension_groups,
                                      .depth = shape.gdn_recurrence.head_groups * rows},
                            MetalSize{.width = shape.gdn_recurrence.lane_threads,
                                      .height = shape.gdn_recurrence.dimension_threads,
                                      .depth = 1});
            if (batch.profile_double_stage == 3u) {
                encode.barrier();
                encode.dispatch(MetalSize{.width = 1,
                                          .height = shape.gdn_recurrence.dimension_groups,
                                          .depth = shape.gdn_recurrence.head_groups * rows},
                                MetalSize{.width = shape.gdn_recurrence.lane_threads,
                                          .height = shape.gdn_recurrence.dimension_threads,
                                          .depth = 1});
            }
            encode.barrier();
            encode.pipeline(kernels.gate_norm_ms);
            encode.buffer(s.gdn_y, 0, 0);
            encode.buffer(s.gdn_z, 0, 1);
            encode.weight(bound.gated_delta.norm_weight, 2);
            encode.buffer(s.gdn_gated, 0, 3);
            encode.constant(static_cast<std::uint32_t>(g.gdn_value_bytes / 2), 4);
            encode.dispatch(shape.gdn_gate_norm.groups, shape.gdn_gate_norm.threads, rows);
            if (batch.profile_double_stage == 7u) {
                encode.barrier();
                encode.dispatch(shape.gdn_gate_norm.groups, shape.gdn_gate_norm.threads, rows);
            }
            encode.barrier();
            encode.pipeline(batch.dense_variant == 1 && kernels.gdn_outproj_ms_v2
                                ? kernels.gdn_outproj_ms_v2
                                : kernels.gdn_outproj_ms);
            encode.buffer(s.gdn_gated, 0, 0);
            encode.quantized(bound.gated_delta.out, 1);
            encode.buffer(s.branch_stream, layer_offset, 4);
            encode.constant(rows, 5);
            encode.constant(static_cast<std::uint32_t>(lanebytes / 2), 6);
            encode.constant(outproj_tile, 7);
            encode.dispatch(shape.out_projection.groups, shape.out_projection.threads,
                            outproj_tiles);
            if (batch.profile_double_stage == 6u) {
                encode.barrier();
                encode.dispatch(shape.out_projection.groups, shape.out_projection.threads,
                                outproj_tiles);
            }
        } else {
            encode.pipeline(batch.dense_variant == 1 && kernels.attn_project_ms_v2
                                ? kernels.attn_project_ms_v2
                                : kernels.attn_project_ms);
            encode.buffer(s.normed, 0, 0);
            encode.quantized(bound.attention.query, 1);
            encode.quantized(bound.attention.key, 4);
            encode.quantized(bound.attention.value, 7);
            encode.buffer(s.attn_projection, 0, 10);
            encode.constant(rows, 11);
            encode.constant(attnproj_tile, 12);
            encode.dispatch(shape.attn_project.groups, shape.attn_project.threads,
                            attnproj_tiles);
            if (batch.profile_double_stage == 4u) {
                encode.barrier();
                encode.dispatch(shape.attn_project.groups, shape.attn_project.threads,
                                attnproj_tiles);
            }
            encode.barrier();
            encode.pipeline(kernels.attn_qk_rope_ms);
            encode.buffer(s.attn_projection, 0, 0);
            encode.weight(bound.attention.query_norm, 1);
            encode.weight(bound.attention.key_norm, 2);
            encode.buffer(s.attn_query, 0, 3);
            encode.buffer(s.attn_gate, 0, 4);
            encode.buffer(pool.layers[layer].first, 0, 5);
            encode.buffer(pool.layers[layer].second, 0, 6);
            encode.constant(step.capacity, 7);
            encode.buffer(batch.state_offsets, layer * 256, 8);
            encode.constant(static_cast<std::uint32_t>(g.attn_projection_bytes / 2), 9);
            encode.constant(static_cast<std::uint32_t>(g.attn_query_bytes / 2), 10);
            encode.dispatch(shape.attn_qk_rope.groups, shape.attn_qk_rope.threads, rows);
            if (batch.profile_double_stage == 5u) {
                encode.barrier();
                encode.dispatch(shape.attn_qk_rope.groups, shape.attn_qk_rope.threads, rows);
            }
            encode.barrier();
            for (std::uint32_t r = 0; r < rows; ++r) {
                if (attention_partitions(streams[r].context) == 1) {
                    continue;
                }
                DecodeLayerState& layer_state = streams[r].state->layers[layer];
                encode_attention_decode_at(
                    encode, step, s, std::uint64_t{r} * g.attn_query_bytes,
                    std::uint64_t{r} * g.attn_record_scratch_bytes,
                    layer_state.first, layer_state.second, streams[r].context);
                encode.barrier();
            }
            if (decode_bucket_rows != 0) {
                encode.pipeline(kernels.attention_decode_ms);
                encode.buffer(s.attn_query, 0, 0);
                encode.buffer(s.attn_gate, 0, 1);
                encode.buffer(pool.layers[layer].first, 0, 2);
                encode.buffer(pool.layers[layer].second, 0, 3);
                encode.buffer(batch.state_offsets, layer * 256, 4);
                encode.buffer(s.attn_attended, 0, 5);
                encode.constant(step.capacity, 6);
                encode.buffer(batch.state_offsets, bucket_offset, 7);
                encode.constant(static_cast<std::uint32_t>(g.attn_query_bytes / 2), 8);
                encode.dispatch(shape.attention_head.groups, shape.attention_head.threads,
                                decode_bucket_rows);
                if (batch.profile_double_stage == 5u) {
                    encode.barrier();
                    encode.dispatch(shape.attention_head.groups, shape.attention_head.threads,
                                    decode_bucket_rows);
                }
                encode.barrier();
            }
            encode.pipeline(batch.dense_variant == 1 && kernels.gdn_outproj_ms_v2
                                ? kernels.gdn_outproj_ms_v2
                                : kernels.gdn_outproj_ms);
            encode.buffer(s.attn_attended, 0, 0);
            encode.quantized(bound.attention.out, 1);
            encode.buffer(s.branch_stream, layer_offset, 4);
            encode.constant(rows, 5);
            encode.constant(static_cast<std::uint32_t>(lanebytes / 2), 6);
            encode.constant(outproj_tile, 7);
            encode.dispatch(shape.out_projection.groups, shape.out_projection.threads,
                            outproj_tiles);
            if (batch.profile_double_stage == 6u) {
                encode.barrier();
                encode.dispatch(shape.out_projection.groups, shape.out_projection.threads,
                                outproj_tiles);
            }
        }

        encode.barrier();
        encode.pipeline(kernels.residual_rms_ms);
        encode.buffer(layer_input, input_inner, 0);
        encode.buffer(s.branch_stream, layer_offset, 1);
        encode.buffer(s.residual_stream, layer_offset, 2);
        encode.weight(bound.post_norm, 3);
        encode.buffer(s.normed, 0, 4);
        encode.constant(static_cast<std::uint32_t>(input_stride / 2), 5);
        encode.constant(static_cast<std::uint32_t>(lanebytes / 2), 6);
        encode.constant(static_cast<std::uint32_t>(hidden / 2), 7);
        encode.dispatch(shape.rms.groups, shape.rms.threads, rows);
        if (batch.profile_double_stage == 7u) {
            encode.barrier();
            encode.dispatch(shape.rms.groups, shape.rms.threads, rows);
        }
        encode.barrier();
        encode.pipeline(kernels.router_ms);
        encode.buffer(s.normed, 0, 0);
        encode.quantized(bound.router, 1);
        encode.quantized(bound.shared_router, 4);
        encode.buffer(s.router_logits, 0, 7);
        encode.constant(rows, 8);
        encode.constant(static_cast<std::uint32_t>(hidden / 2), 9);
        encode.constant(static_cast<std::uint32_t>(g.router_logits_bytes / 4), 10);
        encode.dispatch(shape.router);
        if (batch.profile_double_stage == 8u) {
            encode.barrier();
            encode.dispatch(shape.router);
        }
        encode.barrier();
        encode.pipeline(kernels.router_select_ms);
        encode.buffer(s.router_logits, 0, 0);
        encode.buffer(s.expert_ids, 0, 1);
        encode.buffer(s.expert_coefficients, 0, 2);
        encode.buffer(s.shared_coefficient, 0, 3);
        encode.constant(shape.top_experts, 4);
        encode.constant(static_cast<std::uint32_t>(g.router_logits_bytes / 4), 5);
        encode.constant(static_cast<std::uint32_t>(g.expert_id_bytes / 4), 6);
        encode.constant(static_cast<std::uint32_t>(g.expert_coefficient_bytes / 4), 7);
        encode.constant(1u, 8);
        encode.dispatch(shape.router_select.groups, shape.router_select.threads, rows);
            if (batch.profile_double_stage == 8u) {
                encode.barrier();
                encode.dispatch(shape.router_select.groups, shape.router_select.threads, rows);
            }
        encode.barrier();
        encode.pipeline(kernels.upgate_rows_ms);
        encode.buffer(s.normed, 0, 0);
        encode.buffer(s.expert_ids, 0, 1);
        encode.quantized(bound.expert_gate, 2);
        encode.quantized(bound.expert_up, 5);
        encode.quantized(bound.shared_gate, 8);
        encode.quantized(bound.shared_up, 11);
        encode.buffer(s.expert_hidden, 0, 14);
        encode.constant(static_cast<std::uint32_t>(hidden / 2), 15);
        encode.constant(static_cast<std::uint32_t>(g.expert_id_bytes / 4), 16);
        encode.constant(static_cast<std::uint32_t>(g.expert_hidden_bytes / 2), 17);
        encode.dispatch(shape.grouped_upgate.groups, shape.grouped_upgate.threads, rows);
            if (batch.profile_double_stage == 9u) {
                encode.barrier();
                encode.dispatch(shape.grouped_upgate.groups, shape.grouped_upgate.threads, rows);
            }
        encode.barrier();
        encode.pipeline(kernels.down_rows_ms);
        encode.buffer(s.expert_hidden, 0, 0);
        encode.buffer(s.expert_ids, 0, 1);
        encode.buffer(s.expert_coefficients, 0, 2);
        encode.buffer(s.shared_coefficient, 0, 3);
        encode.quantized(bound.expert_down, 4);
        encode.quantized(bound.shared_down, 7);
        encode.buffer(s.moe_stream, layer_offset, 10);
        encode.buffer(s.residual_stream, layer_offset, 11);
        encode.buffer(s.layer_stream, layer_offset, 12);
        encode.constant(static_cast<std::uint32_t>(g.expert_hidden_bytes / 2), 13);
        encode.constant(static_cast<std::uint32_t>(g.expert_id_bytes / 4), 14);
        encode.constant(static_cast<std::uint32_t>(g.expert_coefficient_bytes / 4), 15);
        encode.constant(static_cast<std::uint32_t>(lanebytes / 2), 16);
        encode.dispatch(shape.grouped_down_res.groups, shape.grouped_down_res.threads, rows);
            if (batch.profile_double_stage == 10u) {
                encode.barrier();
                encode.dispatch(shape.grouped_down_res.groups, shape.grouped_down_res.threads, rows);
            }
    }
    encode.barrier();
    encode.pipeline(kernels.rms_ms);
    encode.buffer(s.layer_stream, (step.schedule.size() - 1) * hidden, 0);
    encode.weight(step.bindings.final_norm, 1);
    encode.buffer(s.final_hidden, 0, 2);
    encode.constant(static_cast<std::uint32_t>(lanebytes / 2), 3);
    encode.constant(static_cast<std::uint32_t>(hidden / 2), 4);
    encode.dispatch(shape.rms.groups, shape.rms.threads, rows);
    encode.barrier();
    const std::uint32_t head_tile =
        batch.head_row_tile == 0
            ? 1u
            : (batch.head_row_tile > rows ? rows : batch.head_row_tile);
    const std::uint32_t head_tiles = (rows + head_tile - 1) / head_tile;
    encode.pipeline(batch.dense_variant == 1 && kernels.lmhead_ms_v2
                        ? kernels.lmhead_ms_v2
                        : kernels.lmhead_ms);
    encode.buffer(s.final_hidden, 0, 0);
    encode.quantized(step.bindings.head, 1);
    encode.buffer(s.logits, 0, 4);
    encode.constant(rows, 5);
    encode.constant(head_tile, 6);
    encode.dispatch(shape.lmhead.groups, shape.lmhead.threads, head_tiles);
    if (batch.profile_double_stage == 11u) {
        encode.barrier();
        encode.dispatch(shape.lmhead.groups, shape.lmhead.threads, head_tiles);
    }
    encode.barrier();
    encode.pipeline(kernels.argmax_stage1_ms);
    encode.buffer(s.logits, 0, 0);
    encode.constant(shape.vocabulary_rows, 1);
    encode.buffer(s.argmax_values, 0, 2);
    encode.buffer(s.argmax_indices, 0, 3);
    encode.constant(static_cast<std::uint32_t>(g.logits_bytes / 2), 4);
    encode.constant(static_cast<std::uint32_t>(g.argmax_value_bytes / 4), 5);
    encode.dispatch(shape.argmax_stage1.groups, shape.argmax_stage1.threads, rows);
    encode.barrier();
    encode.pipeline(kernels.argmax_stage2_ms);
    encode.buffer(s.argmax_values, 0, 0);
    encode.buffer(s.argmax_indices, 0, 1);
    encode.buffer(s.token_id, 0, 2);
    encode.constant(static_cast<std::uint32_t>(g.argmax_value_bytes / 4), 3);
    encode.dispatch(1, rows);
    encode.barrier();
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
