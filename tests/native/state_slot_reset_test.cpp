#include "tatara/generated/model_plan.h"
#include "tatara/runtime/prefill_step.h"
#include "tatara/runtime/state_slot_reset.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>

namespace {

std::size_t allocation_count = 0;
bool track_allocations = false;

} // namespace

void* operator new(std::size_t size) {
    if (track_allocations) {
        ++allocation_count;
    }
    if (void* memory = std::malloc(size)) {
        return memory;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void operator delete[](void* memory) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

void operator delete[](void* memory, std::size_t) noexcept {
    std::free(memory);
}

namespace {

using namespace tatara::runtime;
using tatara::model::qwen36::LayerKind;
using tatara::model::qwen36::generated::kModelPlan;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

StateSlotResetTicket ticket(DecodeStep& decode, DecodeStateSlot& state, std::uint64_t generation) {
    state.status = DecodeStateSlotStatus::ResetPending;
    state.active_reset_generation = generation;
    state.active_reset_segments = 4;
    state.active_reset_bytes = 128;
    return {
        .owner = &decode,
        .state_owner = &state,
        .segment_count = 4,
        .state_bytes = 128,
        .generation = generation,
        .pending = true,
    };
}

void first_package_and_synthetic_layouts_are_plan_driven() {
    DecodeStep decode;
    decode.capacity = kModelPlan.initial_serving_capacity;
    decode.geometry = make_decode_geometry(kModelPlan, decode.capacity);
    decode.schedule.assign(kModelPlan.layers.begin(), kModelPlan.layers.end());

    track_allocations = true;
    const std::size_t before = allocation_count;
    const StateSlotResetLayoutResult first = make_state_slot_reset_layout(decode);
    std::uint64_t next_offset = 0;
    for (std::uint32_t index = 0; index < first.layout.segment_count; ++index) {
        const StateSlotResetSegmentResult selected = state_slot_reset_segment(first.layout, index);
        check(static_cast<bool>(selected), "every first-package reset segment resolves");
        check(selected.segment.logical_offset_bytes == next_offset,
              "first-package reset segments are contiguous");
        next_offset += selected.segment.length_bytes;
    }
    track_allocations = false;

    check(allocation_count == before, "layout and segment walks allocate nothing");
    check(first && first.layout.gated_delta_layers == 30 && first.layout.attention_layers == 10 &&
              first.layout.segment_count == 60,
          "first-package family and segment counts");
    check(first.layout.total_bytes == 64'389'120 && next_offset == first.layout.total_bytes,
          "first-package minimal reset bytes are exact");
    const StateSlotResetSegment first_convolution =
        state_slot_reset_segment(first.layout, 0).segment;
    const StateSlotResetSegment first_recurrent = state_slot_reset_segment(first.layout, 1).segment;
    check(first_convolution.layer_index == 0 &&
              first_convolution.plane == StateSlotResetPlane::Convolution &&
              first_convolution.length_bytes == 49'152,
          "first GDN convolution plane derives from geometry");
    check(first_recurrent.layer_index == 0 &&
              first_recurrent.plane == StateSlotResetPlane::Recurrent &&
              first_recurrent.length_bytes == 2'097'152,
          "first GDN recurrent plane derives from geometry");

    constexpr std::array<LayerKind, 4> kSyntheticSchedule{
        LayerKind::FullAttention,
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
        LayerKind::GatedDelta,
    };
    const StateSlotResetLayoutSpec synthetic_spec{
        .schedule = kSyntheticSchedule,
        .expected_gated_delta_layers = 2,
        .expected_attention_layers = 2,
        .convolution_bytes = 24,
        .recurrent_bytes = 40,
        .expected_total_bytes = 128,
    };
    const StateSlotResetLayoutResult synthetic = make_state_slot_reset_layout(synthetic_spec);
    check(synthetic && synthetic.layout.segment_count == 4 && synthetic.layout.total_bytes == 128,
          "synthetic mixed schedule regenerates without source edits");
    check(state_slot_reset_segment(synthetic.layout, 0).segment.layer_index == 1 &&
              state_slot_reset_segment(synthetic.layout, 2).segment.layer_index == 3,
          "synthetic reset skips attention in schedule order");

    constexpr std::array<LayerKind, 2> kAttentionOnlySchedule{
        LayerKind::FullAttention,
        LayerKind::FullAttention,
    };
    const StateSlotResetLayoutResult attention_only = make_state_slot_reset_layout({
        .schedule = kAttentionOnlySchedule,
        .expected_gated_delta_layers = 0,
        .expected_attention_layers = 2,
        .convolution_bytes = 0,
        .recurrent_bytes = 0,
        .expected_total_bytes = 0,
    });
    check(attention_only && attention_only.layout.segment_count == 0 &&
              attention_only.layout.total_bytes == 0,
          "attention-only schedule needs no recurrent reset");
}

void invalid_layouts_are_typed_before_encoding() {
    constexpr std::array<LayerKind, 2> kSchedule{
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
    };
    StateSlotResetLayoutSpec specification{
        .schedule = kSchedule,
        .expected_gated_delta_layers = 1,
        .expected_attention_layers = 1,
        .convolution_bytes = 24,
        .recurrent_bytes = 40,
        .expected_total_bytes = 64,
    };
    StateSlotResetLayoutSpec invalid = specification;
    invalid.expected_gated_delta_layers = 2;
    check(make_state_slot_reset_layout(invalid).error ==
              StateSlotResetLayoutError::FamilyCountMismatch,
          "family-count mismatch is typed");
    invalid = specification;
    invalid.convolution_bytes = 0;
    check(make_state_slot_reset_layout(invalid).error == StateSlotResetLayoutError::InvalidGeometry,
          "zero state geometry is typed");
    invalid = specification;
    ++invalid.expected_total_bytes;
    check(make_state_slot_reset_layout(invalid).error == StateSlotResetLayoutError::TotalMismatch,
          "total mismatch is typed");
    invalid = specification;
    invalid.convolution_bytes = std::numeric_limits<std::uint64_t>::max();
    check(make_state_slot_reset_layout(invalid).error == StateSlotResetLayoutError::Overflow,
          "state-byte addition overflow is typed");
    constexpr std::array<LayerKind, 2> kInvalidSchedule{
        LayerKind::GatedDelta,
        static_cast<LayerKind>(255),
    };
    invalid = specification;
    invalid.schedule = kInvalidSchedule;
    check(make_state_slot_reset_layout(invalid).error == StateSlotResetLayoutError::InvalidSchedule,
          "unknown layer kind is typed");

    const StateSlotResetLayoutResult layout = make_state_slot_reset_layout(specification);
    check(state_slot_reset_segment(layout.layout, layout.layout.segment_count).error ==
              StateSlotResetLayoutError::SegmentOutOfRange,
          "segment bound is typed");

    tatara::backend::metal::MetalBlitPass invalid_pass;
    tatara::backend::metal::MetalBuffer invalid_buffer;
    check(tatara::backend::metal::fill_buffer(invalid_pass, invalid_buffer, 0, 1, std::byte{0}) ==
              tatara::backend::metal::MetalCommandError::InvalidBlitPass,
          "fill rejects invalid pass before buffer");
}

void completion_is_single_use_and_canonicalizes_phase() {
    DecodeStep decode;
    decode.capacity = 8;
    decode.schedule = {LayerKind::GatedDelta, LayerKind::FullAttention, LayerKind::GatedDelta};
    DecodeStateSlot state;
    state.capacity = decode.capacity;
    state.schedule_identity = decode.schedule.data();
    state.layers.resize(decode.schedule.size());
    state.layers[0].swapped = true;
    state.layers[2].swapped = true;

    StateSlotResetTicket reset = ticket(decode, state, 7);
    StateSlotResetTicket copied = reset;
    StateSlotResetTicket modified = reset;
    ++modified.state_bytes;
    check(!decode_state_slot_available(decode, state), "reset-pending state is unavailable");
    check(complete_state_slot_reset(modified, decode, state) == StateSlotResetError::StaleTicket,
          "modified ticket cannot release the slot");

    track_allocations = true;
    const std::size_t before = allocation_count;
    check(complete_state_slot_reset(reset, decode, state) == StateSlotResetError::None,
          "reset completes once");
    track_allocations = false;
    check(allocation_count == before, "reset completion allocates nothing");
    check(state.status == DecodeStateSlotStatus::Ready && state.active_reset_generation == 0 &&
              state.active_reset_segments == 0 && state.active_reset_bytes == 0 &&
              !state.layers[0].swapped && !state.layers[2].swapped,
          "completion canonicalizes every GDN phase and clears ownership");
    check(complete_state_slot_reset(reset, decode, state) == StateSlotResetError::AlreadyCompleted,
          "completed ticket cannot complete twice");
    check(complete_state_slot_reset(copied, decode, state) == StateSlotResetError::StaleTicket,
          "copied ticket is stale after completion");

    StateSlotResetTicket aborted = ticket(decode, state, 8);
    check(abort_state_slot_reset(aborted, decode, state) == StateSlotResetError::None,
          "pending reset aborts once");
    check(state.status == DecodeStateSlotStatus::Poisoned && state.active_reset_generation == 0,
          "abort poisons and retains the slot");
    check(abort_state_slot_reset(aborted, decode, state) == StateSlotResetError::AlreadyCompleted,
          "aborted ticket cannot abort twice");
}

void partial_prefill_cursor_release_is_exact_and_allocation_free() {
    DecodeStep decode;
    decode.capacity = 8;
    decode.schedule = {LayerKind::GatedDelta};
    DecodeStateSlot state;
    state.capacity = decode.capacity;
    state.schedule_identity = decode.schedule.data();
    state.layers.resize(1);

    PrefillStep prefill;
    prefill.policy.geometry.schedule = PrefillSchedule::ChunkMajor;
    prefill.policy.geometry.context_capacity = decode.capacity;
    prefill.policy.geometry.maximum_block_rows = 4;
    prefill.progress = {
        .state = PrefillProgressState::Ready,
        .owner = &decode,
        .state_owner = &state,
        .live_context = 0,
        .context_base = 0,
        .next_context = 3,
        .row_count = 3,
        .chunk_count = 1,
        .current_layer = 0,
        .current_chunk = 0,
    };

    DecodeStep wrong_owner;
    wrong_owner.capacity = decode.capacity;
    wrong_owner.schedule = decode.schedule;
    check(release_prefill_progress(prefill, wrong_owner, state) == PrefillProgressError::Invalid &&
              prefill.progress.state == PrefillProgressState::Ready,
          "wrong owner cannot release a live cursor");
    prefill.progress.state = PrefillProgressState::UnitPending;
    check(release_prefill_progress(prefill, decode, state) == PrefillProgressError::UnitPending,
          "pending unit cannot be abandoned");
    prefill.progress.state = PrefillProgressState::Ready;

    track_allocations = true;
    const std::size_t before = allocation_count;
    check(release_prefill_progress(prefill, decode, state) == PrefillProgressError::None,
          "ready partial cursor releases");
    track_allocations = false;
    check(allocation_count == before && prefill.progress.state == PrefillProgressState::Idle &&
              decode_state_slot_available(decode, state),
          "cursor release allocates nothing and does not claim state reset");
}

} // namespace

int main() {
    first_package_and_synthetic_layouts_are_plan_driven();
    invalid_layouts_are_typed_before_encoding();
    completion_is_single_use_and_canonicalizes_phase();
    partial_prefill_cursor_release_is_exact_and_allocation_free();
    if (failures == 0) {
        std::printf("state_slot_reset: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
