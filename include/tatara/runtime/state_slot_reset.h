#pragma once

#include "tatara/backend/metal/commands.h"
#include "tatara/runtime/decode_step.h"

#include <cstdint>
#include <optional>
#include <span>

namespace tatara::runtime {

inline constexpr std::uint32_t kStateSlotResetSchemaVersion = 1;

enum class StateSlotResetLayoutError : std::uint8_t {
    None,
    InvalidSchedule,
    InvalidGeometry,
    FamilyCountMismatch,
    TotalMismatch,
    Overflow,
    SegmentOutOfRange,
};

enum class StateSlotResetPlane : std::uint8_t {
    Convolution,
    Recurrent,
};

struct StateSlotResetLayoutSpec {
    std::span<const model::qwen36::LayerKind> schedule;
    std::uint32_t expected_gated_delta_layers{0};
    std::uint32_t expected_attention_layers{0};
    std::uint64_t convolution_bytes{0};
    std::uint64_t recurrent_bytes{0};
    std::uint64_t expected_total_bytes{0};
};

struct StateSlotResetLayout {
    StateSlotResetLayoutSpec specification;
    std::uint32_t gated_delta_layers{0};
    std::uint32_t attention_layers{0};
    std::uint32_t segment_count{0};
    std::uint64_t total_bytes{0};
};

struct StateSlotResetLayoutResult {
    StateSlotResetLayoutError error{StateSlotResetLayoutError::InvalidGeometry};
    StateSlotResetLayout layout;

    explicit constexpr operator bool() const noexcept {
        return error == StateSlotResetLayoutError::None;
    }
};

struct StateSlotResetSegment {
    StateSlotResetPlane plane{StateSlotResetPlane::Convolution};
    std::uint32_t layer_index{0};
    std::uint64_t logical_offset_bytes{0};
    std::uint64_t length_bytes{0};
};

struct StateSlotResetSegmentResult {
    StateSlotResetLayoutError error{StateSlotResetLayoutError::SegmentOutOfRange};
    StateSlotResetSegment segment;

    explicit constexpr operator bool() const noexcept {
        return error == StateSlotResetLayoutError::None;
    }
};

StateSlotResetLayoutResult
make_state_slot_reset_layout(const StateSlotResetLayoutSpec& specification) noexcept;
StateSlotResetLayoutResult make_state_slot_reset_layout(const DecodeStep& decode) noexcept;
StateSlotResetSegmentResult state_slot_reset_segment(const StateSlotResetLayout& layout,
                                                     std::uint32_t segment_index) noexcept;

enum class StateSlotResetError : std::uint8_t {
    None,
    InvalidLayout,
    InvalidStateSlot,
    StateSlotNotReady,
    GenerationExhausted,
    CommandEncodingFailed,
    StaleTicket,
    AlreadyCompleted,
    PoisonedStateSlot,
};

struct StateSlotResetTicket {
    const DecodeStep* owner{nullptr};
    DecodeStateSlot* state_owner{nullptr};
    std::uint32_t segment_count{0};
    std::uint64_t state_bytes{0};
    std::uint64_t generation{0};
    bool pending{false};
};

struct StateSlotResetResult {
    StateSlotResetError error{StateSlotResetError::InvalidLayout};
    StateSlotResetLayoutError layout_error{StateSlotResetLayoutError::None};
    backend::metal::MetalCommandError command_error{backend::metal::MetalCommandError::None};
    std::uint32_t segment_count{0};
    std::uint64_t state_bytes{0};
    bool completed{false};
    std::optional<StateSlotResetTicket> ticket;

    explicit operator bool() const noexcept {
        return error == StateSlotResetError::None && (completed || ticket.has_value());
    }
};

StateSlotResetResult encode_state_slot_reset(const DecodeStep& decode, DecodeStateSlot& state,
                                             backend::metal::MetalBlitPass& blit_pass) noexcept;
StateSlotResetError complete_state_slot_reset(StateSlotResetTicket& ticket,
                                              const DecodeStep& decode,
                                              DecodeStateSlot& state) noexcept;
StateSlotResetError abort_state_slot_reset(StateSlotResetTicket& ticket, const DecodeStep& decode,
                                           DecodeStateSlot& state) noexcept;

} // namespace tatara::runtime
