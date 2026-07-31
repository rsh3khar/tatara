#include "tatara/runtime/state_slot_reset.h"

#include <cstddef>
#include <limits>

namespace tatara::runtime {
namespace {

using namespace backend::metal;

bool add(std::uint64_t left, std::uint64_t right, std::uint64_t& result) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    result = left + right;
    return true;
}

bool multiply(std::uint64_t left, std::uint64_t right, std::uint64_t& result) noexcept {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

StateSlotResetResult
reset_failure(StateSlotResetError error,
              StateSlotResetLayoutError layout_error = StateSlotResetLayoutError::None,
              MetalCommandError command_error = MetalCommandError::None) noexcept {
    return {
        .error = error,
        .layout_error = layout_error,
        .command_error = command_error,
    };
}

void clear_active_reset(DecodeStateSlot& state) noexcept {
    state.active_reset_generation = 0;
    state.active_reset_segments = 0;
    state.active_reset_bytes = 0;
}

MetalBuffer* reset_buffer(DecodeLayerState& state, StateSlotResetPlane plane) noexcept {
    switch (plane) {
    case StateSlotResetPlane::Convolution:
        return &state.first;
    case StateSlotResetPlane::Recurrent:
        return &state.second;
    }
    return nullptr;
}

bool ticket_matches(const StateSlotResetTicket& ticket, const DecodeStep& decode,
                    const DecodeStateSlot& state) noexcept {
    return ticket.owner == &decode && ticket.state_owner == &state && ticket.generation != 0 &&
           ticket.segment_count != 0 && ticket.state_bytes != 0 &&
           decode_state_slot_compatible(decode, state) &&
           state.status == DecodeStateSlotStatus::ResetPending &&
           state.active_reset_generation == ticket.generation &&
           state.active_reset_segments == ticket.segment_count &&
           state.active_reset_bytes == ticket.state_bytes;
}

} // namespace

StateSlotResetLayoutResult
make_state_slot_reset_layout(const StateSlotResetLayoutSpec& specification) noexcept {
    if (specification.schedule.empty() ||
        specification.schedule.size() > std::numeric_limits<std::uint32_t>::max()) {
        return {.error = StateSlotResetLayoutError::InvalidSchedule};
    }

    std::uint32_t gated_delta_layers = 0;
    std::uint32_t attention_layers = 0;
    for (const model::qwen36::LayerKind kind : specification.schedule) {
        if (kind == model::qwen36::LayerKind::GatedDelta) {
            ++gated_delta_layers;
        } else if (kind == model::qwen36::LayerKind::FullAttention) {
            ++attention_layers;
        } else {
            return {.error = StateSlotResetLayoutError::InvalidSchedule};
        }
    }
    if (gated_delta_layers != specification.expected_gated_delta_layers ||
        attention_layers != specification.expected_attention_layers ||
        std::uint64_t{gated_delta_layers} + attention_layers != specification.schedule.size()) {
        return {.error = StateSlotResetLayoutError::FamilyCountMismatch};
    }
    if (gated_delta_layers != 0 &&
        (specification.convolution_bytes == 0 || specification.recurrent_bytes == 0)) {
        return {.error = StateSlotResetLayoutError::InvalidGeometry};
    }

    std::uint64_t per_layer_bytes = 0;
    std::uint64_t total_bytes = 0;
    std::uint64_t segment_count = 0;
    if (!add(specification.convolution_bytes, specification.recurrent_bytes, per_layer_bytes) ||
        !multiply(gated_delta_layers, per_layer_bytes, total_bytes) ||
        !multiply(gated_delta_layers, 2, segment_count) ||
        segment_count > std::numeric_limits<std::uint32_t>::max()) {
        return {.error = StateSlotResetLayoutError::Overflow};
    }
    if (total_bytes != specification.expected_total_bytes) {
        return {.error = StateSlotResetLayoutError::TotalMismatch};
    }
    return {
        .error = StateSlotResetLayoutError::None,
        .layout =
            StateSlotResetLayout{
                .specification = specification,
                .gated_delta_layers = gated_delta_layers,
                .attention_layers = attention_layers,
                .segment_count = static_cast<std::uint32_t>(segment_count),
                .total_bytes = total_bytes,
            },
    };
}

StateSlotResetLayoutResult make_state_slot_reset_layout(const DecodeStep& decode) noexcept {
    std::uint64_t per_layer_bytes = 0;
    std::uint64_t expected_total_bytes = 0;
    if (!add(decode.geometry.gdn_conv_state_bytes, decode.geometry.gdn_recurrent_state_bytes,
             per_layer_bytes) ||
        !multiply(decode.geometry.gated_delta_layers, per_layer_bytes, expected_total_bytes)) {
        return {.error = StateSlotResetLayoutError::Overflow};
    }
    return make_state_slot_reset_layout({
        .schedule = decode.schedule,
        .expected_gated_delta_layers = decode.geometry.gated_delta_layers,
        .expected_attention_layers = decode.geometry.attention_layers,
        .convolution_bytes = decode.geometry.gdn_conv_state_bytes,
        .recurrent_bytes = decode.geometry.gdn_recurrent_state_bytes,
        .expected_total_bytes = expected_total_bytes,
    });
}

StateSlotResetSegmentResult state_slot_reset_segment(const StateSlotResetLayout& layout,
                                                     std::uint32_t segment_index) noexcept {
    if (segment_index >= layout.segment_count) {
        return {.error = StateSlotResetLayoutError::SegmentOutOfRange};
    }
    const std::uint32_t gated_delta_ordinal = segment_index / 2u;
    const StateSlotResetPlane plane = (segment_index & 1u) == 0u ? StateSlotResetPlane::Convolution
                                                                 : StateSlotResetPlane::Recurrent;
    std::uint32_t found = 0;
    for (std::uint32_t layer = 0; layer < layout.specification.schedule.size(); ++layer) {
        if (layout.specification.schedule[layer] != model::qwen36::LayerKind::GatedDelta) {
            continue;
        }
        if (found == gated_delta_ordinal) {
            const std::uint64_t per_layer =
                layout.specification.convolution_bytes + layout.specification.recurrent_bytes;
            const std::uint64_t logical_offset =
                std::uint64_t{gated_delta_ordinal} * per_layer +
                (plane == StateSlotResetPlane::Recurrent ? layout.specification.convolution_bytes
                                                         : 0);
            return {
                .error = StateSlotResetLayoutError::None,
                .segment =
                    StateSlotResetSegment{
                        .plane = plane,
                        .layer_index = layer,
                        .logical_offset_bytes = logical_offset,
                        .length_bytes = plane == StateSlotResetPlane::Convolution
                                            ? layout.specification.convolution_bytes
                                            : layout.specification.recurrent_bytes,
                    },
            };
        }
        ++found;
    }
    return {.error = StateSlotResetLayoutError::FamilyCountMismatch};
}

StateSlotResetResult encode_state_slot_reset(const DecodeStep& decode, DecodeStateSlot& state,
                                             MetalBlitPass& blit_pass) noexcept {
    if (state.status == DecodeStateSlotStatus::Poisoned) {
        return reset_failure(StateSlotResetError::PoisonedStateSlot);
    }
    if (!decode_state_slot_compatible(decode, state) ||
        !decode_state_slot_complete(decode, state)) {
        return reset_failure(StateSlotResetError::InvalidStateSlot);
    }
    if (!decode_state_slot_ready(decode, state)) {
        return reset_failure(StateSlotResetError::StateSlotNotReady);
    }
    const StateSlotResetLayoutResult planned = make_state_slot_reset_layout(decode);
    if (!planned) {
        return reset_failure(StateSlotResetError::InvalidLayout, planned.error);
    }
    if (planned.layout.segment_count == 0) {
        return {
            .error = StateSlotResetError::None,
            .layout_error = StateSlotResetLayoutError::None,
            .command_error = MetalCommandError::None,
            .segment_count = 0,
            .state_bytes = 0,
            .completed = true,
            .ticket = std::nullopt,
        };
    }
    if (state.next_reset_generation == 0) {
        return reset_failure(StateSlotResetError::GenerationExhausted);
    }
    if (!blit_pass) {
        return reset_failure(StateSlotResetError::CommandEncodingFailed,
                             StateSlotResetLayoutError::None, MetalCommandError::InvalidBlitPass);
    }

    for (std::uint32_t index = 0; index < planned.layout.segment_count; ++index) {
        const StateSlotResetSegmentResult selected =
            state_slot_reset_segment(planned.layout, index);
        if (!selected || selected.segment.layer_index >= state.layers.size()) {
            return reset_failure(StateSlotResetError::InvalidLayout, selected.error);
        }
        DecodeLayerState& layer = state.layers[selected.segment.layer_index];
        MetalBuffer* destination = reset_buffer(layer, selected.segment.plane);
        if (destination == nullptr || !*destination ||
            destination->size_bytes() < selected.segment.length_bytes) {
            return reset_failure(StateSlotResetError::InvalidStateSlot);
        }
    }

    for (std::uint32_t index = 0; index < planned.layout.segment_count; ++index) {
        const StateSlotResetSegment segment =
            state_slot_reset_segment(planned.layout, index).segment;
        MetalBuffer* destination = reset_buffer(state.layers[segment.layer_index], segment.plane);
        const MetalCommandError command_error =
            fill_buffer(blit_pass, *destination, 0, segment.length_bytes, std::byte{0});
        if (command_error != MetalCommandError::None) {
            state.status = DecodeStateSlotStatus::Poisoned;
            clear_active_reset(state);
            return reset_failure(StateSlotResetError::CommandEncodingFailed,
                                 StateSlotResetLayoutError::None, command_error);
        }
    }

    const std::uint64_t generation = state.next_reset_generation;
    state.next_reset_generation =
        generation == std::numeric_limits<std::uint64_t>::max() ? 0 : generation + 1u;
    state.active_reset_generation = generation;
    state.active_reset_segments = planned.layout.segment_count;
    state.active_reset_bytes = planned.layout.total_bytes;
    state.status = DecodeStateSlotStatus::ResetPending;
    return {
        .error = StateSlotResetError::None,
        .layout_error = StateSlotResetLayoutError::None,
        .command_error = MetalCommandError::None,
        .segment_count = planned.layout.segment_count,
        .state_bytes = planned.layout.total_bytes,
        .completed = false,
        .ticket =
            StateSlotResetTicket{
                .owner = &decode,
                .state_owner = &state,
                .segment_count = planned.layout.segment_count,
                .state_bytes = planned.layout.total_bytes,
                .generation = generation,
                .pending = true,
            },
    };
}

StateSlotResetError complete_state_slot_reset(StateSlotResetTicket& ticket,
                                              const DecodeStep& decode,
                                              DecodeStateSlot& state) noexcept {
    if (!ticket.pending) {
        return StateSlotResetError::AlreadyCompleted;
    }
    if (state.status == DecodeStateSlotStatus::Poisoned) {
        return StateSlotResetError::PoisonedStateSlot;
    }
    if (!ticket_matches(ticket, decode, state)) {
        return StateSlotResetError::StaleTicket;
    }
    for (std::size_t layer = 0; layer < decode.schedule.size(); ++layer) {
        if (decode.schedule[layer] == model::qwen36::LayerKind::GatedDelta) {
            state.layers[layer].swapped = false;
        }
    }
    clear_active_reset(state);
    state.status = DecodeStateSlotStatus::Ready;
    ticket.pending = false;
    return StateSlotResetError::None;
}

StateSlotResetError abort_state_slot_reset(StateSlotResetTicket& ticket, const DecodeStep& decode,
                                           DecodeStateSlot& state) noexcept {
    if (!ticket.pending) {
        return StateSlotResetError::AlreadyCompleted;
    }
    if (state.status == DecodeStateSlotStatus::Poisoned) {
        clear_active_reset(state);
        ticket.pending = false;
        return StateSlotResetError::PoisonedStateSlot;
    }
    if (!ticket_matches(ticket, decode, state)) {
        return StateSlotResetError::StaleTicket;
    }
    clear_active_reset(state);
    state.status = DecodeStateSlotStatus::Poisoned;
    ticket.pending = false;
    return StateSlotResetError::None;
}

} // namespace tatara::runtime
