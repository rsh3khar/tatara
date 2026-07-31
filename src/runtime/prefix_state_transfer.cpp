#include "tatara/runtime/prefix_state_transfer.h"

#include <algorithm>
#include <limits>

namespace tatara::runtime {
namespace {

using backend::metal::MetalBlitPass;
using backend::metal::MetalBuffer;
using backend::metal::MetalCommandError;

bool add(std::uint64_t left, std::uint64_t right, std::uint64_t& out) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    out = left + right;
    return true;
}

bool multiply(std::uint64_t left, std::uint64_t right, std::uint64_t& out) noexcept {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    out = left * right;
    return true;
}

bool range_fits(std::uint64_t offset, std::uint64_t length, std::uint64_t capacity) noexcept {
    return offset <= capacity && length <= capacity - offset;
}

bool is_gated_delta(model::qwen36::LayerKind kind) noexcept {
    return kind == model::qwen36::LayerKind::GatedDelta;
}

bool is_known_layer(model::qwen36::LayerKind kind) noexcept {
    return kind == model::qwen36::LayerKind::GatedDelta ||
           kind == model::qwen36::LayerKind::FullAttention;
}

void clear_active_transfer(DecodeStateSlot& state) noexcept {
    state.active_transfer_generation = 0;
    state.active_transfer_arena = nullptr;
    state.active_transfer_positions = 0;
    state.active_transfer_offset_bytes = 0;
    state.active_transfer_state_bytes = 0;
}

std::uint32_t family_layer(const PrefixStateLayout& layout, bool gated_delta,
                           std::uint32_t ordinal) noexcept {
    std::uint32_t seen = 0;
    for (std::uint32_t layer = 0; layer < layout.specification.schedule.size(); ++layer) {
        if (is_gated_delta(layout.specification.schedule[layer]) == gated_delta) {
            if (seen == ordinal) {
                return layer;
            }
            ++seen;
        }
    }
    return std::numeric_limits<std::uint32_t>::max();
}

const MetalBuffer* state_buffer(const DecodeLayerState& layer,
                                PrefixStateBufferPlane plane) noexcept {
    switch (plane) {
    case PrefixStateBufferPlane::First:
        return &layer.first;
    case PrefixStateBufferPlane::FirstOutput:
        return &layer.first_out;
    case PrefixStateBufferPlane::Second:
        return &layer.second;
    case PrefixStateBufferPlane::SecondOutput:
        return &layer.second_out;
    case PrefixStateBufferPlane::Arena:
        return nullptr;
    }
    return nullptr;
}

MetalBuffer* state_buffer(DecodeLayerState& layer, PrefixStateBufferPlane plane) noexcept {
    return const_cast<MetalBuffer*>(
        state_buffer(static_cast<const DecodeLayerState&>(layer), plane));
}

PrefixStateTransferResult
transfer_failure(PrefixStateTransferError error,
                 PrefixStateLayoutError layout_error = PrefixStateLayoutError::None,
                 MetalCommandError command_error = MetalCommandError::None) noexcept {
    return {
        .error = error,
        .layout_error = layout_error,
        .command_error = command_error,
    };
}

bool plan_matches_runtime(const DecodeStep& decode, const PrefillGeometry& geometry,
                          std::uint32_t positions, PrefixStateLayoutSpec& specification,
                          PrefixStateLayoutError& error) noexcept {
    if (decode.capacity == 0 || geometry.context_capacity != decode.capacity ||
        geometry.key_value_heads == 0 || geometry.attention_head_dimension == 0 ||
        geometry.gated_delta_layers == 0 || geometry.attention_layers == 0) {
        error = PrefixStateLayoutError::InvalidGeometry;
        return false;
    }
    std::uint64_t gated_delta_live = 0;
    std::uint64_t attention_position = 0;
    std::uint64_t attention_slot = 0;
    if (!add(decode.geometry.gdn_conv_state_bytes, decode.geometry.gdn_recurrent_state_bytes,
             gated_delta_live) ||
        !multiply(2u * std::uint64_t{geometry.key_value_heads}, geometry.attention_head_dimension,
                  attention_position) ||
        !multiply(attention_position, kBf16Bytes, attention_position) ||
        !multiply(2u, decode.geometry.attn_cache_bytes, attention_slot)) {
        error = PrefixStateLayoutError::Overflow;
        return false;
    }
    if (gated_delta_live != geometry.gdn_live_state_bytes_per_layer ||
        attention_position != geometry.attention_state_bytes_per_position_per_layer ||
        attention_slot != geometry.attention_slot_state_bytes_per_layer) {
        error = PrefixStateLayoutError::InvalidGeometry;
        return false;
    }
    const PrefixStateSnapshotGeometry snapshot =
        make_prefix_state_snapshot_geometry(geometry, positions);
    if (!snapshot) {
        error = snapshot.error == PrefixStateSnapshotError::PositionOutOfRange
                    ? PrefixStateLayoutError::PositionOutOfRange
                : snapshot.error == PrefixStateSnapshotError::Overflow
                    ? PrefixStateLayoutError::Overflow
                    : PrefixStateLayoutError::InvalidGeometry;
        return false;
    }
    specification = {
        .schedule = decode.schedule,
        .capacity = decode.capacity,
        .positions = positions,
        .key_value_heads = geometry.key_value_heads,
        .head_dimension = geometry.attention_head_dimension,
        .expected_gated_delta_layers = geometry.gated_delta_layers,
        .expected_attention_layers = geometry.attention_layers,
        .gated_delta_convolution_bytes = decode.geometry.gdn_conv_state_bytes,
        .gated_delta_recurrent_bytes = decode.geometry.gdn_recurrent_state_bytes,
        .expected_total_bytes = snapshot.total_bytes,
    };
    error = PrefixStateLayoutError::None;
    return true;
}

PrefixStateTransferResult encode_transfer(PrefixStateTransferDirection direction,
                                          const DecodeStep& decode, DecodeStateSlot& state,
                                          const PrefillGeometry& geometry, MetalBlitPass& blit_pass,
                                          MetalBuffer& state_arena,
                                          std::uint64_t arena_offset_bytes,
                                          std::uint32_t positions,
                                          std::span<std::byte> source_phase_evidence) noexcept {
    if (state.status == DecodeStateSlotStatus::Poisoned) {
        return transfer_failure(PrefixStateTransferError::PoisonedStateSlot);
    }
    if (!decode_state_slot_compatible(decode, state) ||
        !decode_state_slot_complete(decode, state)) {
        return transfer_failure(PrefixStateTransferError::InvalidStateSlot);
    }
    if (!decode_state_slot_ready(decode, state)) {
        return transfer_failure(PrefixStateTransferError::StateSlotNotReady);
    }
    if (!state_arena) {
        return transfer_failure(PrefixStateTransferError::InvalidArena);
    }
    const std::size_t phase_bytes =
        direction == PrefixStateTransferDirection::Snapshot
            ? prefix_state_phase_evidence_bytes(decode.schedule)
            : 0;
    if (direction == PrefixStateTransferDirection::Snapshot &&
        source_phase_evidence.size() < phase_bytes) {
        return transfer_failure(PrefixStateTransferError::PhaseEvidenceTooSmall);
    }

    const PrefixStateLayoutResult planned = make_prefix_state_layout(decode, geometry, positions);
    if (!planned) {
        return transfer_failure(PrefixStateTransferError::InvalidLayout, planned.error);
    }
    const PrefixStateLayout& layout = planned.layout;
    if (validate_prefix_state_arena_extent(layout, arena_offset_bytes, state_arena.size_bytes()) !=
        PrefixStateTransferError::None) {
        return transfer_failure(PrefixStateTransferError::ArenaOutOfRange);
    }
    if (state.next_transfer_generation == 0) {
        return transfer_failure(PrefixStateTransferError::GenerationExhausted);
    }
    if (!blit_pass) {
        return transfer_failure(PrefixStateTransferError::CommandEncodingFailed,
                                PrefixStateLayoutError::None, MetalCommandError::InvalidBlitPass);
    }

    // Validate the complete plan before issuing its first copy. The second
    // walk can then fail only at the Metal command boundary.
    for (std::uint32_t index = 0; index < layout.segment_count; ++index) {
        const PrefixStateSegmentResult selected = prefix_state_segment(layout, index);
        if (!selected || selected.segment.layer_index >= state.layers.size()) {
            return transfer_failure(PrefixStateTransferError::InvalidLayout, selected.error);
        }
        const DecodeLayerState& layer = state.layers[selected.segment.layer_index];
        const PrefixStateCopyRoute route =
            prefix_state_copy_route(selected.segment, direction, layer.swapped);
        const PrefixStateBufferPlane slot_plane =
            direction == PrefixStateTransferDirection::Snapshot ? route.source : route.destination;
        const MetalBuffer* buffer = state_buffer(layer, slot_plane);
        std::uint64_t arena_segment_offset = 0;
        if (buffer == nullptr || buffer == &state_arena ||
            !range_fits(selected.segment.slot_offset_bytes, selected.segment.length_bytes,
                        buffer->size_bytes()) ||
            !add(arena_offset_bytes, selected.segment.snapshot_offset_bytes,
                 arena_segment_offset) ||
            !range_fits(arena_segment_offset, selected.segment.length_bytes,
                        state_arena.size_bytes())) {
            return transfer_failure(PrefixStateTransferError::InvalidStateSlot);
        }
    }

    for (std::uint32_t index = 0; index < layout.segment_count; ++index) {
        const PrefixStateSegment segment = prefix_state_segment(layout, index).segment;
        DecodeLayerState& layer = state.layers[segment.layer_index];
        const PrefixStateCopyRoute route =
            prefix_state_copy_route(segment, direction, layer.swapped);
        std::uint64_t arena_segment_offset = 0;
        (void)add(arena_offset_bytes, segment.snapshot_offset_bytes, arena_segment_offset);
        MetalCommandError command_error = MetalCommandError::None;
        if (direction == PrefixStateTransferDirection::Snapshot) {
            const MetalBuffer* source = state_buffer(layer, route.source);
            command_error = backend::metal::copy_buffer(blit_pass, *source,
                                                        segment.slot_offset_bytes, state_arena,
                                                        arena_segment_offset, segment.length_bytes);
        } else {
            MetalBuffer* destination = state_buffer(layer, route.destination);
            command_error = backend::metal::copy_buffer(
                blit_pass, state_arena, arena_segment_offset, *destination,
                segment.slot_offset_bytes, segment.length_bytes);
        }
        if (command_error != MetalCommandError::None) {
            state.status = DecodeStateSlotStatus::Poisoned;
            clear_active_transfer(state);
            return transfer_failure(PrefixStateTransferError::CommandEncodingFailed,
                                    PrefixStateLayoutError::None, command_error);
        }
    }
    if (direction == PrefixStateTransferDirection::Snapshot) {
        const PrefixStateTransferError phase_error =
            capture_prefix_state_phase_evidence(
                decode, state, source_phase_evidence.first(phase_bytes));
        if (phase_error != PrefixStateTransferError::None) {
            return transfer_failure(phase_error);
        }
    }

    const std::uint64_t generation = state.next_transfer_generation;
    state.next_transfer_generation =
        generation == std::numeric_limits<std::uint64_t>::max() ? 0 : generation + 1u;
    state.active_transfer_generation = generation;
    state.active_transfer_arena = &state_arena;
    state.active_transfer_positions = positions;
    state.active_transfer_offset_bytes = arena_offset_bytes;
    state.active_transfer_state_bytes = layout.total_bytes;
    state.status = direction == PrefixStateTransferDirection::Snapshot
                       ? DecodeStateSlotStatus::SnapshotPending
                       : DecodeStateSlotStatus::RestorePending;
    return {
        .error = PrefixStateTransferError::None,
        .layout_error = PrefixStateLayoutError::None,
        .command_error = MetalCommandError::None,
        .segment_count = layout.segment_count,
        .state_bytes = layout.total_bytes,
        .source_phase_evidence_bytes = phase_bytes,
        .ticket =
            PrefixStateTransferTicket{
                .direction = direction,
                .owner = &decode,
                .state_owner = &state,
                .arena_owner = &state_arena,
                .positions = positions,
                .arena_offset_bytes = arena_offset_bytes,
                .state_bytes = layout.total_bytes,
                .generation = generation,
                .pending = true,
            },
    };
}

bool ticket_matches(const PrefixStateTransferTicket& ticket, const DecodeStep& decode,
                    const DecodeStateSlot& state, const MetalBuffer& state_arena) noexcept {
    const DecodeStateSlotStatus expected =
        ticket.direction == PrefixStateTransferDirection::Snapshot
            ? DecodeStateSlotStatus::SnapshotPending
            : DecodeStateSlotStatus::RestorePending;
    return ticket.owner == &decode && ticket.state_owner == &state &&
           ticket.arena_owner == &state_arena && ticket.generation != 0 &&
           decode_state_slot_compatible(decode, state) &&
           state.active_transfer_generation == ticket.generation &&
           state.active_transfer_arena == &state_arena &&
           state.active_transfer_positions == ticket.positions &&
           state.active_transfer_offset_bytes == ticket.arena_offset_bytes &&
           state.active_transfer_state_bytes == ticket.state_bytes && state.status == expected;
}

} // namespace

PrefixStateLayoutResult
make_prefix_state_layout(const PrefixStateLayoutSpec& specification) noexcept {
    if (specification.schedule.empty() ||
        specification.schedule.size() > std::numeric_limits<std::uint32_t>::max()) {
        return {.error = PrefixStateLayoutError::InvalidSchedule};
    }
    if (specification.capacity == 0 || specification.key_value_heads == 0 ||
        specification.head_dimension == 0 || specification.expected_gated_delta_layers == 0 ||
        specification.expected_attention_layers == 0 ||
        specification.gated_delta_convolution_bytes == 0 ||
        specification.gated_delta_recurrent_bytes == 0 || specification.expected_total_bytes == 0) {
        return {.error = PrefixStateLayoutError::InvalidGeometry};
    }
    if (specification.positions == 0 || specification.positions > specification.capacity) {
        return {.error = PrefixStateLayoutError::PositionOutOfRange};
    }

    PrefixStateLayout layout{.specification = specification};
    for (const model::qwen36::LayerKind kind : specification.schedule) {
        if (!is_known_layer(kind)) {
            return {.error = PrefixStateLayoutError::InvalidSchedule};
        }
        if (is_gated_delta(kind)) {
            ++layout.gated_delta_layers;
        } else {
            ++layout.attention_layers;
        }
    }
    if (layout.gated_delta_layers != specification.expected_gated_delta_layers ||
        layout.attention_layers != specification.expected_attention_layers) {
        return {.error = PrefixStateLayoutError::FamilyCountMismatch};
    }

    std::uint64_t gated_delta_per_layer = 0;
    std::uint64_t attention_segments = 0;
    std::uint64_t segment_count = 0;
    if (!add(specification.gated_delta_convolution_bytes, specification.gated_delta_recurrent_bytes,
             gated_delta_per_layer) ||
        !multiply(specification.positions, specification.head_dimension,
                  layout.attention_segment_bytes) ||
        !multiply(layout.attention_segment_bytes, kBf16Bytes, layout.attention_segment_bytes) ||
        !multiply(specification.capacity, specification.head_dimension,
                  layout.attention_slot_head_stride_bytes) ||
        !multiply(layout.attention_slot_head_stride_bytes, kBf16Bytes,
                  layout.attention_slot_head_stride_bytes) ||
        !multiply(layout.gated_delta_layers, gated_delta_per_layer, layout.gated_delta_bytes) ||
        !multiply(layout.attention_layers, 2u * std::uint64_t{specification.key_value_heads},
                  attention_segments) ||
        !multiply(attention_segments, layout.attention_segment_bytes, layout.attention_bytes) ||
        !add(layout.gated_delta_bytes, layout.attention_bytes, layout.total_bytes) ||
        !add(2u * std::uint64_t{layout.gated_delta_layers}, attention_segments, segment_count) ||
        segment_count > std::numeric_limits<std::uint32_t>::max()) {
        return {.error = PrefixStateLayoutError::Overflow};
    }
    if (layout.total_bytes != specification.expected_total_bytes) {
        return {.error = PrefixStateLayoutError::TotalMismatch};
    }
    layout.segment_count = static_cast<std::uint32_t>(segment_count);
    return {.error = PrefixStateLayoutError::None, .layout = layout};
}

PrefixStateLayoutResult make_prefix_state_layout(const DecodeStep& decode,
                                                 const PrefillGeometry& geometry,
                                                 std::uint32_t positions) noexcept {
    PrefixStateLayoutSpec specification;
    PrefixStateLayoutError error = PrefixStateLayoutError::None;
    if (!plan_matches_runtime(decode, geometry, positions, specification, error)) {
        return {.error = error};
    }
    return make_prefix_state_layout(specification);
}

PrefixStateSegmentResult prefix_state_segment(const PrefixStateLayout& layout,
                                              std::uint32_t segment_index) noexcept {
    if (segment_index >= layout.segment_count) {
        return {.error = PrefixStateLayoutError::SegmentOutOfRange};
    }
    const std::uint32_t gated_delta_segments = 2u * layout.gated_delta_layers;
    if (segment_index < gated_delta_segments) {
        const std::uint32_t ordinal = segment_index / 2u;
        const bool recurrent = (segment_index & 1u) != 0;
        const std::uint32_t layer = family_layer(layout, true, ordinal);
        if (layer == std::numeric_limits<std::uint32_t>::max()) {
            return {.error = PrefixStateLayoutError::FamilyCountMismatch};
        }
        const std::uint64_t per_layer = layout.specification.gated_delta_convolution_bytes +
                                        layout.specification.gated_delta_recurrent_bytes;
        return {
            .error = PrefixStateLayoutError::None,
            .segment =
                {
                    .kind = recurrent ? PrefixStateSegmentKind::GatedDeltaRecurrent
                                      : PrefixStateSegmentKind::GatedDeltaConvolution,
                    .layer_index = layer,
                    .head_index = 0,
                    .slot_offset_bytes = 0,
                    .snapshot_offset_bytes =
                        std::uint64_t{ordinal} * per_layer +
                        (recurrent ? layout.specification.gated_delta_convolution_bytes : 0),
                    .length_bytes = recurrent ? layout.specification.gated_delta_recurrent_bytes
                                              : layout.specification.gated_delta_convolution_bytes,
                },
        };
    }

    const std::uint32_t relative = segment_index - gated_delta_segments;
    const std::uint32_t segments_per_layer = 2u * layout.specification.key_value_heads;
    const std::uint32_t attention_ordinal = relative / segments_per_layer;
    const std::uint32_t within_layer = relative % segments_per_layer;
    const bool value = within_layer >= layout.specification.key_value_heads;
    const std::uint32_t head =
        value ? within_layer - layout.specification.key_value_heads : within_layer;
    const std::uint32_t layer = family_layer(layout, false, attention_ordinal);
    if (layer == std::numeric_limits<std::uint32_t>::max()) {
        return {.error = PrefixStateLayoutError::FamilyCountMismatch};
    }
    return {
        .error = PrefixStateLayoutError::None,
        .segment =
            {
                .kind = value ? PrefixStateSegmentKind::AttentionValue
                              : PrefixStateSegmentKind::AttentionKey,
                .layer_index = layer,
                .head_index = head,
                .slot_offset_bytes = std::uint64_t{head} * layout.attention_slot_head_stride_bytes,
                .snapshot_offset_bytes = layout.gated_delta_bytes +
                                         std::uint64_t{relative} * layout.attention_segment_bytes,
                .length_bytes = layout.attention_segment_bytes,
            },
    };
}

PrefixStateCopyRoute prefix_state_copy_route(const PrefixStateSegment& segment,
                                             PrefixStateTransferDirection direction,
                                             bool gated_delta_swapped) noexcept {
    PrefixStateBufferPlane live = PrefixStateBufferPlane::First;
    switch (segment.kind) {
    case PrefixStateSegmentKind::GatedDeltaConvolution:
        live = gated_delta_swapped ? PrefixStateBufferPlane::FirstOutput
                                   : PrefixStateBufferPlane::First;
        break;
    case PrefixStateSegmentKind::GatedDeltaRecurrent:
        live = gated_delta_swapped ? PrefixStateBufferPlane::SecondOutput
                                   : PrefixStateBufferPlane::Second;
        break;
    case PrefixStateSegmentKind::AttentionKey:
        live = PrefixStateBufferPlane::First;
        break;
    case PrefixStateSegmentKind::AttentionValue:
        live = PrefixStateBufferPlane::Second;
        break;
    }
    if (direction == PrefixStateTransferDirection::Snapshot) {
        return {.source = live, .destination = PrefixStateBufferPlane::Arena};
    }
    if (segment.kind == PrefixStateSegmentKind::GatedDeltaConvolution) {
        live = PrefixStateBufferPlane::First;
    } else if (segment.kind == PrefixStateSegmentKind::GatedDeltaRecurrent) {
        live = PrefixStateBufferPlane::Second;
    }
    return {.source = PrefixStateBufferPlane::Arena, .destination = live};
}

PrefixStateTransferError
validate_prefix_state_arena_extent(const PrefixStateLayout& layout,
                                   std::uint64_t arena_offset_bytes,
                                   std::uint64_t arena_capacity_bytes) noexcept {
    if (layout.total_bytes == 0 || layout.segment_count == 0) {
        return PrefixStateTransferError::InvalidLayout;
    }
    return range_fits(arena_offset_bytes, layout.total_bytes, arena_capacity_bytes)
               ? PrefixStateTransferError::None
               : PrefixStateTransferError::ArenaOutOfRange;
}

std::size_t prefix_state_phase_evidence_bytes(
    std::span<const model::qwen36::LayerKind> schedule) noexcept {
    return schedule.size() / 8u + (schedule.size() % 8u != 0u ? 1u : 0u);
}

PrefixStateTransferError capture_prefix_state_phase_evidence(
    const DecodeStep& decode, const DecodeStateSlot& state,
    std::span<std::byte> phase_evidence) noexcept {
    const std::size_t required = prefix_state_phase_evidence_bytes(decode.schedule);
    if (phase_evidence.size() < required) {
        return PrefixStateTransferError::PhaseEvidenceTooSmall;
    }
    if (!decode_state_slot_compatible(decode, state)) {
        return PrefixStateTransferError::InvalidStateSlot;
    }
    std::fill(phase_evidence.begin(), phase_evidence.end(), std::byte{0});
    for (std::size_t layer = 0; layer < decode.schedule.size(); ++layer) {
        if (decode.schedule[layer] == model::qwen36::LayerKind::GatedDelta &&
            state.layers[layer].swapped) {
            phase_evidence[layer / 8u] |=
                static_cast<std::byte>(1u << (layer % 8u));
        }
    }
    return PrefixStateTransferError::None;
}

PrefixStateTransferResult
encode_prefix_state_snapshot(const DecodeStep& decode, DecodeStateSlot& state,
                             const PrefillGeometry& geometry, MetalBlitPass& blit_pass,
                             MetalBuffer& state_arena, std::uint64_t arena_offset_bytes,
                             std::uint32_t positions,
                             std::span<std::byte> source_phase_evidence) noexcept {
    return encode_transfer(PrefixStateTransferDirection::Snapshot, decode, state, geometry,
                           blit_pass, state_arena, arena_offset_bytes, positions,
                           source_phase_evidence);
}

PrefixStateTransferResult
encode_prefix_state_restore(const DecodeStep& decode, DecodeStateSlot& state,
                            const PrefillGeometry& geometry, MetalBlitPass& blit_pass,
                            MetalBuffer& state_arena, std::uint64_t arena_offset_bytes,
                            std::uint32_t positions) noexcept {
    return encode_transfer(PrefixStateTransferDirection::Restore, decode, state, geometry,
                           blit_pass, state_arena, arena_offset_bytes, positions, {});
}

PrefixStateTransferError complete_prefix_state_transfer(PrefixStateTransferTicket& ticket,
                                                        const DecodeStep& decode,
                                                        DecodeStateSlot& state,
                                                        const MetalBuffer& state_arena) noexcept {
    if (!ticket.pending) {
        return PrefixStateTransferError::AlreadyCompleted;
    }
    if (state.status == DecodeStateSlotStatus::Poisoned) {
        return PrefixStateTransferError::PoisonedStateSlot;
    }
    if (!ticket_matches(ticket, decode, state, state_arena)) {
        return PrefixStateTransferError::StaleTicket;
    }
    if (ticket.direction == PrefixStateTransferDirection::Restore) {
        for (std::size_t layer = 0; layer < decode.schedule.size(); ++layer) {
            if (decode.schedule[layer] == model::qwen36::LayerKind::GatedDelta) {
                state.layers[layer].swapped = false;
            }
        }
    }
    state.status = DecodeStateSlotStatus::Ready;
    clear_active_transfer(state);
    ticket.pending = false;
    return PrefixStateTransferError::None;
}

PrefixStateTransferError abort_prefix_state_transfer(PrefixStateTransferTicket& ticket,
                                                     const DecodeStep& decode,
                                                     DecodeStateSlot& state,
                                                     const MetalBuffer& state_arena) noexcept {
    if (!ticket.pending) {
        return PrefixStateTransferError::AlreadyCompleted;
    }
    if (state.status == DecodeStateSlotStatus::Poisoned) {
        clear_active_transfer(state);
        ticket.pending = false;
        return PrefixStateTransferError::PoisonedStateSlot;
    }
    if (!ticket_matches(ticket, decode, state, state_arena)) {
        return PrefixStateTransferError::StaleTicket;
    }
    state.status = DecodeStateSlotStatus::Poisoned;
    clear_active_transfer(state);
    ticket.pending = false;
    return PrefixStateTransferError::None;
}

} // namespace tatara::runtime
