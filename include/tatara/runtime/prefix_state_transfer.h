#pragma once

#include "tatara/backend/metal/commands.h"
#include "tatara/runtime/decode_step.h"
#include "tatara/runtime/prefill_geometry.h"

#include <cstdint>
#include <optional>
#include <span>

namespace tatara::runtime {

inline constexpr std::uint32_t kPrefixStateLayoutSchemaVersion = 1;
inline constexpr std::uint32_t kPrefixStateTransferSchemaVersion = 1;

enum class PrefixStateLayoutError : std::uint8_t {
    None,
    InvalidSchedule,
    InvalidGeometry,
    PositionOutOfRange,
    FamilyCountMismatch,
    TotalMismatch,
    Overflow,
    SegmentOutOfRange,
};

enum class PrefixStateSegmentKind : std::uint8_t {
    GatedDeltaConvolution,
    GatedDeltaRecurrent,
    AttentionKey,
    AttentionValue,
};

enum class PrefixStateBufferPlane : std::uint8_t {
    First,
    FirstOutput,
    Second,
    SecondOutput,
    Arena,
};

enum class PrefixStateTransferDirection : std::uint8_t {
    Snapshot,
    Restore,
};

struct PrefixStateLayoutSpec {
    std::span<const model::qwen36::LayerKind> schedule;
    std::uint32_t capacity{0};
    std::uint32_t positions{0};
    std::uint32_t key_value_heads{0};
    std::uint32_t head_dimension{0};
    std::uint32_t expected_gated_delta_layers{0};
    std::uint32_t expected_attention_layers{0};
    std::uint64_t gated_delta_convolution_bytes{0};
    std::uint64_t gated_delta_recurrent_bytes{0};
    std::uint64_t expected_total_bytes{0};
};

struct PrefixStateLayout {
    PrefixStateLayoutSpec specification;
    std::uint32_t gated_delta_layers{0};
    std::uint32_t attention_layers{0};
    std::uint32_t segment_count{0};
    std::uint64_t attention_segment_bytes{0};
    std::uint64_t attention_slot_head_stride_bytes{0};
    std::uint64_t gated_delta_bytes{0};
    std::uint64_t attention_bytes{0};
    std::uint64_t total_bytes{0};
};

struct PrefixStateLayoutResult {
    PrefixStateLayoutError error{PrefixStateLayoutError::InvalidGeometry};
    PrefixStateLayout layout;

    explicit constexpr operator bool() const noexcept {
        return error == PrefixStateLayoutError::None;
    }
};

struct PrefixStateSegment {
    PrefixStateSegmentKind kind{PrefixStateSegmentKind::GatedDeltaConvolution};
    std::uint32_t layer_index{0};
    std::uint32_t head_index{0};
    std::uint64_t slot_offset_bytes{0};
    std::uint64_t snapshot_offset_bytes{0};
    std::uint64_t length_bytes{0};
};

struct PrefixStateSegmentResult {
    PrefixStateLayoutError error{PrefixStateLayoutError::SegmentOutOfRange};
    PrefixStateSegment segment;

    explicit constexpr operator bool() const noexcept {
        return error == PrefixStateLayoutError::None;
    }
};

struct PrefixStateCopyRoute {
    PrefixStateBufferPlane source{PrefixStateBufferPlane::First};
    PrefixStateBufferPlane destination{PrefixStateBufferPlane::Arena};
};

PrefixStateLayoutResult
make_prefix_state_layout(const PrefixStateLayoutSpec& specification) noexcept;
PrefixStateLayoutResult make_prefix_state_layout(const DecodeStep& decode,
                                                 const PrefillGeometry& geometry,
                                                 std::uint32_t positions) noexcept;
PrefixStateSegmentResult prefix_state_segment(const PrefixStateLayout& layout,
                                              std::uint32_t segment_index) noexcept;
PrefixStateCopyRoute prefix_state_copy_route(const PrefixStateSegment& segment,
                                             PrefixStateTransferDirection direction,
                                             bool gated_delta_swapped) noexcept;

enum class PrefixStateTransferError : std::uint8_t {
    None,
    InvalidLayout,
    InvalidStateSlot,
    StateSlotNotReady,
    InvalidArena,
    ArenaOutOfRange,
    GenerationExhausted,
    PhaseEvidenceTooSmall,
    CommandEncodingFailed,
    StaleTicket,
    AlreadyCompleted,
    PoisonedStateSlot,
};

PrefixStateTransferError
validate_prefix_state_arena_extent(const PrefixStateLayout& layout,
                                   std::uint64_t arena_offset_bytes,
                                   std::uint64_t arena_capacity_bytes) noexcept;
std::size_t prefix_state_phase_evidence_bytes(
    std::span<const model::qwen36::LayerKind> schedule) noexcept;
PrefixStateTransferError capture_prefix_state_phase_evidence(
    const DecodeStep& decode, const DecodeStateSlot& state,
    std::span<std::byte> phase_evidence) noexcept;

struct PrefixStateTransferTicket {
    PrefixStateTransferDirection direction{PrefixStateTransferDirection::Snapshot};
    const DecodeStep* owner{nullptr};
    DecodeStateSlot* state_owner{nullptr};
    const backend::metal::MetalBuffer* arena_owner{nullptr};
    std::uint32_t positions{0};
    std::uint64_t arena_offset_bytes{0};
    std::uint64_t state_bytes{0};
    std::uint64_t generation{0};
    bool pending{false};
};

struct PrefixStateTransferResult {
    PrefixStateTransferError error{PrefixStateTransferError::InvalidLayout};
    PrefixStateLayoutError layout_error{PrefixStateLayoutError::None};
    backend::metal::MetalCommandError command_error{backend::metal::MetalCommandError::None};
    std::uint32_t segment_count{0};
    std::uint64_t state_bytes{0};
    std::size_t source_phase_evidence_bytes{0};
    std::optional<PrefixStateTransferTicket> ticket;

    explicit operator bool() const noexcept {
        return error == PrefixStateTransferError::None && ticket.has_value();
    }
};

PrefixStateTransferResult encode_prefix_state_snapshot(
    const DecodeStep& decode, DecodeStateSlot& state, const PrefillGeometry& geometry,
    backend::metal::MetalBlitPass& blit_pass, backend::metal::MetalBuffer& state_arena,
    std::uint64_t arena_offset_bytes, std::uint32_t positions,
    std::span<std::byte> source_phase_evidence) noexcept;

PrefixStateTransferResult encode_prefix_state_restore(
    const DecodeStep& decode, DecodeStateSlot& state, const PrefillGeometry& geometry,
    backend::metal::MetalBlitPass& blit_pass, backend::metal::MetalBuffer& state_arena,
    std::uint64_t arena_offset_bytes, std::uint32_t positions) noexcept;

PrefixStateTransferError
complete_prefix_state_transfer(PrefixStateTransferTicket& ticket, const DecodeStep& decode,
                               DecodeStateSlot& state,
                               const backend::metal::MetalBuffer& state_arena) noexcept;

PrefixStateTransferError
abort_prefix_state_transfer(PrefixStateTransferTicket& ticket, const DecodeStep& decode,
                            DecodeStateSlot& state,
                            const backend::metal::MetalBuffer& state_arena) noexcept;

} // namespace tatara::runtime
