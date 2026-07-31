#include "tatara/generated/model_plan.h"
#include "tatara/runtime/prefix_state_transfer.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

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

PrefixStateTransferTicket ticket(PrefixStateTransferDirection direction, DecodeStep& decode,
                                 DecodeStateSlot& state,
                                 const tatara::backend::metal::MetalBuffer& arena,
                                 std::uint64_t generation) {
    state.active_transfer_arena = &arena;
    state.active_transfer_positions = 4;
    state.active_transfer_offset_bytes = 16;
    state.active_transfer_state_bytes = 32;
    return {
        .direction = direction,
        .owner = &decode,
        .state_owner = &state,
        .arena_owner = &arena,
        .positions = 4,
        .arena_offset_bytes = 16,
        .state_bytes = 32,
        .generation = generation,
        .pending = true,
    };
}

struct SyntheticLayerPlanes {
    std::array<std::vector<std::byte>, 4> planes;
    bool swapped{false};
};

std::vector<std::byte>& plane(
    SyntheticLayerPlanes& layer, PrefixStateBufferPlane selected) {
    switch (selected) {
    case PrefixStateBufferPlane::First:
        return layer.planes[0];
    case PrefixStateBufferPlane::FirstOutput:
        return layer.planes[1];
    case PrefixStateBufferPlane::Second:
        return layer.planes[2];
    case PrefixStateBufferPlane::SecondOutput:
        return layer.planes[3];
    case PrefixStateBufferPlane::Arena:
        std::abort();
    }
    std::abort();
}

void logical_round_trip(const PrefixStateLayout& layout, bool gated_swapped) {
    constexpr std::size_t kPlaneBytes = 1'216;
    std::vector<SyntheticLayerPlanes> source(layout.specification.schedule.size());
    std::vector<SyntheticLayerPlanes> restored(layout.specification.schedule.size());
    for (std::size_t layer_index = 0; layer_index < source.size(); ++layer_index) {
        for (std::size_t plane_index = 0; plane_index < 4; ++plane_index) {
            source[layer_index].planes[plane_index].resize(kPlaneBytes);
            restored[layer_index].planes[plane_index].assign(
                kPlaneBytes, std::byte{0xcd});
            for (std::size_t byte_index = 0; byte_index < kPlaneBytes;
                 ++byte_index) {
                source[layer_index].planes[plane_index][byte_index] =
                    static_cast<std::byte>(
                        (17u * layer_index + 37u * plane_index +
                         byte_index) %
                        251u);
            }
        }
        const bool gated =
            layout.specification.schedule[layer_index] ==
            LayerKind::GatedDelta;
        source[layer_index].swapped = gated && gated_swapped;
        restored[layer_index].swapped = gated;
    }

    std::vector<std::byte> first_snapshot(layout.total_bytes);
    std::vector<std::byte> second_snapshot(layout.total_bytes);
    for (std::uint32_t index = 0; index < layout.segment_count; ++index) {
        const PrefixStateSegment segment =
            prefix_state_segment(layout, index).segment;
        const PrefixStateCopyRoute route = prefix_state_copy_route(
            segment, PrefixStateTransferDirection::Snapshot,
            source[segment.layer_index].swapped);
        const std::vector<std::byte>& source_plane =
            plane(source[segment.layer_index], route.source);
        std::memcpy(
            first_snapshot.data() + segment.snapshot_offset_bytes,
            source_plane.data() + segment.slot_offset_bytes,
            segment.length_bytes);
    }

    for (std::uint32_t index = 0; index < layout.segment_count; ++index) {
        const PrefixStateSegment segment =
            prefix_state_segment(layout, index).segment;
        const PrefixStateCopyRoute route = prefix_state_copy_route(
            segment, PrefixStateTransferDirection::Restore,
            restored[segment.layer_index].swapped);
        std::vector<std::byte>& destination =
            plane(restored[segment.layer_index], route.destination);
        std::memcpy(
            destination.data() + segment.slot_offset_bytes,
            first_snapshot.data() + segment.snapshot_offset_bytes,
            segment.length_bytes);
    }
    for (std::size_t layer_index = 0; layer_index < restored.size();
         ++layer_index) {
        if (layout.specification.schedule[layer_index] ==
            LayerKind::GatedDelta) {
            restored[layer_index].swapped = false;
        }
    }

    for (std::uint32_t index = 0; index < layout.segment_count; ++index) {
        const PrefixStateSegment segment =
            prefix_state_segment(layout, index).segment;
        const PrefixStateCopyRoute route = prefix_state_copy_route(
            segment, PrefixStateTransferDirection::Snapshot,
            restored[segment.layer_index].swapped);
        const std::vector<std::byte>& source_plane =
            plane(restored[segment.layer_index], route.source);
        std::memcpy(
            second_snapshot.data() + segment.snapshot_offset_bytes,
            source_plane.data() + segment.slot_offset_bytes,
            segment.length_bytes);
    }

    check(first_snapshot == second_snapshot,
          "snapshot-reset-restore-snapshot preserves canonical payload bytes");
    for (std::size_t layer_index = 0; layer_index < restored.size();
         ++layer_index) {
        if (layout.specification.schedule[layer_index] ==
            LayerKind::GatedDelta) {
            check(!restored[layer_index].swapped,
                  "logical restore canonicalizes every GDN layer phase");
        }
    }
}

} // namespace

int main() {
    constexpr std::uint32_t kCapacity = 16'384;
    constexpr std::uint32_t kPositions = 4'096;
    DecodeStep decode;
    decode.capacity = kCapacity;
    decode.geometry = make_decode_geometry(kModelPlan, kCapacity);
    decode.schedule.assign(kModelPlan.layers.begin(), kModelPlan.layers.end());

    const PrefillPolicy policy{
        .schedule = PrefillSchedule::LayerMajor,
        .context_capacity = kCapacity,
        .maximum_block_rows = 2'048,
        .first_chunk_rows = 256,
        .query_tile_rows = 256,
        .attention_partition = kAttentionPartition,
        .exact_rows_per_threadgroup = 16,
        .gdn_gate_hoist = true,
    };
    const PrefillGeometryResult prefill = make_prefill_geometry(kModelPlan, policy);
    check(static_cast<bool>(prefill), "first-package prefill geometry");

    const PrefixStateLayoutResult planned =
        make_prefix_state_layout(decode, prefill.geometry, kPositions);
    check(static_cast<bool>(planned), "first-package transfer layout");
    check(planned.layout.gated_delta_layers == 30 && planned.layout.attention_layers == 10,
          "first-package family counts");
    check(planned.layout.segment_count == 100, "first-package segment count");
    check(kPrefixStateLayoutSchemaVersion == 1 &&
              prefix_state_phase_evidence_bytes(decode.schedule) == 5,
          "layout schema and first-package phase evidence are explicit");
    check(planned.layout.total_bytes == 148'275'200, "first-package 4k canonical state bytes");
    check(planned.layout.attention_segment_bytes == 2'097'152, "one packed 4k KV-head segment");
    check(planned.layout.attention_slot_head_stride_bytes == 8'388'608,
          "one full-capacity KV-head stride");

    DecodeStep qualification_decode;
    qualification_decode.capacity = 65'536;
    qualification_decode.geometry =
        make_decode_geometry(kModelPlan, qualification_decode.capacity);
    qualification_decode.schedule.assign(
        kModelPlan.layers.begin(), kModelPlan.layers.end());
    PrefillPolicy qualification_policy = policy;
    qualification_policy.context_capacity =
        qualification_decode.capacity;
    const PrefillGeometryResult qualification_prefill =
        make_prefill_geometry(kModelPlan, qualification_policy);
    check(static_cast<bool>(qualification_prefill),
          "qualification-capacity prefill geometry");
    constexpr std::array<std::pair<std::uint32_t, std::uint64_t>, 4>
        kLayoutCases{{
            {4'096, 148'275'200},
            {49'408, 1'076'264'960},
            {50'000, 1'088'389'120},
            {65'536, 1'406'566'400},
        }};
    for (const auto& [positions, expected_bytes] : kLayoutCases) {
        const PrefixStateLayoutResult layout =
            make_prefix_state_layout(
                qualification_decode,
                qualification_prefill.geometry, positions);
        check(
            layout && layout.layout.segment_count == 100 &&
                layout.layout.total_bytes == expected_bytes,
            "first-package canonical layout formula matches a frozen gate position");
    }

    std::uint64_t next_offset = 0;
    std::uint32_t gated_delta_segments = 0;
    std::uint32_t attention_segments = 0;
    track_allocations = true;
    const std::size_t before_plan_walk = allocation_count;
    for (std::uint32_t index = 0; index < planned.layout.segment_count; ++index) {
        const PrefixStateSegmentResult selected = prefix_state_segment(planned.layout, index);
        check(static_cast<bool>(selected), "every first-package segment resolves");
        check(selected.segment.snapshot_offset_bytes == next_offset,
              "canonical payload has no implicit gap");
        next_offset += selected.segment.length_bytes;
        const bool gated = selected.segment.kind == PrefixStateSegmentKind::GatedDeltaConvolution ||
                           selected.segment.kind == PrefixStateSegmentKind::GatedDeltaRecurrent;
        check((decode.schedule[selected.segment.layer_index] == LayerKind::GatedDelta) == gated,
              "segment names the matching schedule family");
        if (gated) {
            ++gated_delta_segments;
            const bool convolution =
                selected.segment.kind ==
                PrefixStateSegmentKind::GatedDeltaConvolution;
            const PrefixStateCopyRoute phase_false =
                prefix_state_copy_route(
                    selected.segment,
                    PrefixStateTransferDirection::Snapshot, false);
            const PrefixStateCopyRoute phase_true =
                prefix_state_copy_route(
                    selected.segment,
                    PrefixStateTransferDirection::Snapshot, true);
            const PrefixStateCopyRoute restored =
                prefix_state_copy_route(
                    selected.segment,
                    PrefixStateTransferDirection::Restore, true);
            check(
                phase_false.source ==
                        (convolution
                             ? PrefixStateBufferPlane::First
                             : PrefixStateBufferPlane::Second) &&
                    phase_true.source ==
                        (convolution
                             ? PrefixStateBufferPlane::FirstOutput
                             : PrefixStateBufferPlane::SecondOutput) &&
                    restored.destination ==
                        (convolution
                             ? PrefixStateBufferPlane::First
                             : PrefixStateBufferPlane::Second),
                "both live phases and canonical restore route for every GDN layer");
        } else {
            ++attention_segments;
        }
        (void)prefix_state_copy_route(selected.segment, PrefixStateTransferDirection::Snapshot,
                                      false);
        (void)prefix_state_copy_route(selected.segment, PrefixStateTransferDirection::Restore,
                                      true);
    }
    track_allocations = false;
    check(allocation_count == before_plan_walk, "plan and route walk allocates nothing");
    check(next_offset == planned.layout.total_bytes, "segments exactly fill canonical payload");
    check(gated_delta_segments == 60 && attention_segments == 40,
          "family segment totals are exact");
    check(validate_prefix_state_arena_extent(planned.layout, 64, planned.layout.total_bytes + 64) ==
              PrefixStateTransferError::None,
          "exact arena extent is admitted");
    check(validate_prefix_state_arena_extent(planned.layout, 64, planned.layout.total_bytes + 63) ==
              PrefixStateTransferError::ArenaOutOfRange,
          "one-byte-short arena is rejected");
    check(validate_prefix_state_arena_extent(planned.layout,
                                             std::numeric_limits<std::uint64_t>::max(),
                                             std::numeric_limits<std::uint64_t>::max()) ==
              PrefixStateTransferError::ArenaOutOfRange,
          "arena offset arithmetic cannot wrap");

    const PrefixStateSegment first_attention = prefix_state_segment(planned.layout, 60).segment;
    const PrefixStateSegment second_attention = prefix_state_segment(planned.layout, 61).segment;
    const PrefixStateSegment first_value = prefix_state_segment(planned.layout, 62).segment;
    check(first_attention.layer_index == 3 &&
              first_attention.kind == PrefixStateSegmentKind::AttentionKey &&
              first_attention.head_index == 0 && first_attention.slot_offset_bytes == 0,
          "attention family starts with first key head");
    check(second_attention.kind == PrefixStateSegmentKind::AttentionKey &&
              second_attention.head_index == 1 &&
              second_attention.slot_offset_bytes == planned.layout.attention_slot_head_stride_bytes,
          "key heads are ascending and capacity-strided");
    check(first_value.kind == PrefixStateSegmentKind::AttentionValue && first_value.head_index == 0,
          "value heads follow every key head");

    const PrefixStateSegment convolution{
        .kind = PrefixStateSegmentKind::GatedDeltaConvolution,
    };
    const PrefixStateSegment recurrent{
        .kind = PrefixStateSegmentKind::GatedDeltaRecurrent,
    };
    const PrefixStateCopyRoute convolution_a =
        prefix_state_copy_route(convolution, PrefixStateTransferDirection::Snapshot, false);
    const PrefixStateCopyRoute convolution_b =
        prefix_state_copy_route(convolution, PrefixStateTransferDirection::Snapshot, true);
    const PrefixStateCopyRoute recurrent_b =
        prefix_state_copy_route(recurrent, PrefixStateTransferDirection::Snapshot, true);
    const PrefixStateCopyRoute restored_convolution =
        prefix_state_copy_route(convolution, PrefixStateTransferDirection::Restore, true);
    const PrefixStateCopyRoute restored_recurrent =
        prefix_state_copy_route(recurrent, PrefixStateTransferDirection::Restore, true);
    check(convolution_a.source == PrefixStateBufferPlane::First &&
              convolution_b.source == PrefixStateBufferPlane::FirstOutput &&
              recurrent_b.source == PrefixStateBufferPlane::SecondOutput,
          "snapshot selects the live GDN phase");
    check(restored_convolution.source == PrefixStateBufferPlane::Arena &&
              restored_convolution.destination == PrefixStateBufferPlane::First &&
              restored_recurrent.destination == PrefixStateBufferPlane::Second,
          "restore targets the canonical GDN input pair");

    constexpr std::array<LayerKind, 4> kSyntheticSchedule{
        LayerKind::FullAttention,
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
        LayerKind::GatedDelta,
    };
    const PrefixStateLayoutSpec synthetic_spec{
        .schedule = kSyntheticSchedule,
        .capacity = 19,
        .positions = 7,
        .key_value_heads = 4,
        .head_dimension = 8,
        .expected_gated_delta_layers = 2,
        .expected_attention_layers = 2,
        .gated_delta_convolution_bytes = 24,
        .gated_delta_recurrent_bytes = 40,
        .expected_total_bytes = 1'920,
    };
    const PrefixStateLayoutResult synthetic = make_prefix_state_layout(synthetic_spec);
    check(static_cast<bool>(synthetic) && synthetic.layout.segment_count == 20 &&
              synthetic.layout.total_bytes == 1'920,
          "synthetic mixed schedule regenerates without source edits");
    check(prefix_state_segment(synthetic.layout, 0).segment.layer_index == 1 &&
              prefix_state_segment(synthetic.layout, 4).segment.layer_index == 0,
          "canonical layout is family-major, each family schedule-ordered");
    logical_round_trip(synthetic.layout, false);
    logical_round_trip(synthetic.layout, true);

    PrefixStateLayoutSpec invalid = synthetic_spec;
    invalid.positions = 0;
    check(make_prefix_state_layout(invalid).error == PrefixStateLayoutError::PositionOutOfRange,
          "zero position is typed");
    invalid = synthetic_spec;
    invalid.positions = 20;
    check(make_prefix_state_layout(invalid).error == PrefixStateLayoutError::PositionOutOfRange,
          "position beyond capacity is typed");
    invalid = synthetic_spec;
    invalid.expected_attention_layers = 3;
    check(make_prefix_state_layout(invalid).error == PrefixStateLayoutError::FamilyCountMismatch,
          "family-count mismatch is typed");
    invalid = synthetic_spec;
    ++invalid.expected_total_bytes;
    check(make_prefix_state_layout(invalid).error == PrefixStateLayoutError::TotalMismatch,
          "canonical total mismatch is typed");
    invalid = synthetic_spec;
    invalid.gated_delta_convolution_bytes = std::numeric_limits<std::uint64_t>::max();
    check(make_prefix_state_layout(invalid).error == PrefixStateLayoutError::Overflow,
          "layout overflow is typed");
    constexpr std::array<LayerKind, 2> kInvalidSchedule{
        LayerKind::GatedDelta,
        static_cast<LayerKind>(255),
    };
    invalid = synthetic_spec;
    invalid.schedule = kInvalidSchedule;
    invalid.expected_gated_delta_layers = 1;
    invalid.expected_attention_layers = 1;
    check(make_prefix_state_layout(invalid).error == PrefixStateLayoutError::InvalidSchedule,
          "unknown layer kind is typed");
    check(prefix_state_segment(synthetic.layout, synthetic.layout.segment_count).error ==
              PrefixStateLayoutError::SegmentOutOfRange,
          "segment bound is typed");

    tatara::backend::metal::MetalCommandBuffer invalid_command_buffer;
    const auto invalid_blit =
        tatara::backend::metal::begin_blit_pass(std::move(invalid_command_buffer));
    check(invalid_blit.error == tatara::backend::metal::MetalCommandError::InvalidCommandBuffer,
          "blit begin rejects an invalid command buffer");
    tatara::backend::metal::MetalBlitPass invalid_blit_pass;
    tatara::backend::metal::MetalBuffer invalid_source;
    tatara::backend::metal::MetalBuffer invalid_destination;
    check(tatara::backend::metal::copy_buffer(invalid_blit_pass, invalid_source, 0,
                                              invalid_destination, 0, 1) ==
              tatara::backend::metal::MetalCommandError::InvalidBlitPass,
          "copy rejects an invalid blit pass before buffers");
    check(tatara::backend::metal::end_blit_pass(std::move(invalid_blit_pass)).error ==
              tatara::backend::metal::MetalCommandError::InvalidBlitPass,
          "blit end rejects invalid ownership");

    DecodeStep lifecycle_decode;
    lifecycle_decode.capacity = 8;
    lifecycle_decode.schedule = {LayerKind::GatedDelta, LayerKind::FullAttention};
    DecodeStateSlot lifecycle_state;
    lifecycle_state.capacity = lifecycle_decode.capacity;
    lifecycle_state.schedule_identity = lifecycle_decode.schedule.data();
    lifecycle_state.layers.resize(lifecycle_decode.schedule.size());
    tatara::backend::metal::MetalBuffer arena;

    lifecycle_state.layers[0].swapped = true;
    std::array<std::byte, 1> phase_evidence{};
    check(capture_prefix_state_phase_evidence(lifecycle_decode, lifecycle_state,
                                              phase_evidence) ==
                  PrefixStateTransferError::None &&
              phase_evidence[0] == std::byte{1},
          "snapshot evidence captures the live GDN phase by schedule index");
    lifecycle_state.layers[0].swapped = false;
    check(capture_prefix_state_phase_evidence(lifecycle_decode, lifecycle_state,
                                              phase_evidence) ==
                  PrefixStateTransferError::None &&
              phase_evidence[0] == std::byte{0},
          "false GDN phase is captured without stale bits");
    check(capture_prefix_state_phase_evidence(lifecycle_decode, lifecycle_state, {}) ==
              PrefixStateTransferError::PhaseEvidenceTooSmall,
          "undersized coordinator phase storage is rejected");
    lifecycle_state.layers[0].swapped = true;
    lifecycle_state.status = DecodeStateSlotStatus::RestorePending;
    lifecycle_state.active_transfer_generation = 7;
    PrefixStateTransferTicket restored =
        ticket(PrefixStateTransferDirection::Restore, lifecycle_decode, lifecycle_state, arena, 7);
    track_allocations = true;
    const std::size_t before_complete = allocation_count;
    check(complete_prefix_state_transfer(restored, lifecycle_decode, lifecycle_state, arena) ==
              PrefixStateTransferError::None,
          "restore completion succeeds once");
    track_allocations = false;
    check(allocation_count == before_complete, "completion allocates nothing");
    check(lifecycle_state.status == DecodeStateSlotStatus::Ready &&
              lifecycle_state.active_transfer_generation == 0 &&
              lifecycle_state.active_transfer_arena == nullptr &&
              lifecycle_state.active_transfer_positions == 0 &&
              lifecycle_state.active_transfer_offset_bytes == 0 &&
              lifecycle_state.active_transfer_state_bytes == 0 &&
              !lifecycle_state.layers[0].swapped,
          "restore completion canonicalizes phase and releases slot");
    check(complete_prefix_state_transfer(restored, lifecycle_decode, lifecycle_state, arena) ==
              PrefixStateTransferError::AlreadyCompleted,
          "completed ticket cannot complete twice");

    lifecycle_state.status = DecodeStateSlotStatus::SnapshotPending;
    lifecycle_state.active_transfer_generation = 8;
    PrefixStateTransferTicket snapshot =
        ticket(PrefixStateTransferDirection::Snapshot, lifecycle_decode, lifecycle_state, arena, 8);
    PrefixStateTransferTicket copied_snapshot = snapshot;
    PrefixStateTransferTicket modified_snapshot = snapshot;
    ++modified_snapshot.arena_offset_bytes;
    check(complete_prefix_state_transfer(modified_snapshot, lifecycle_decode, lifecycle_state,
                                         arena) == PrefixStateTransferError::StaleTicket,
          "modified ticket cannot release its slot");
    check(complete_prefix_state_transfer(snapshot, lifecycle_decode, lifecycle_state, arena) ==
              PrefixStateTransferError::None,
          "snapshot completion releases without phase mutation");
    check(complete_prefix_state_transfer(copied_snapshot, lifecycle_decode, lifecycle_state,
                                         arena) == PrefixStateTransferError::StaleTicket,
          "copied ticket is stale after first completion");

    lifecycle_state.status = DecodeStateSlotStatus::SnapshotPending;
    lifecycle_state.active_transfer_generation = 9;
    PrefixStateTransferTicket aborted =
        ticket(PrefixStateTransferDirection::Snapshot, lifecycle_decode, lifecycle_state, arena, 9);
    check(abort_prefix_state_transfer(aborted, lifecycle_decode, lifecycle_state, arena) ==
              PrefixStateTransferError::None,
          "pending transfer abort succeeds once");
    check(lifecycle_state.status == DecodeStateSlotStatus::Poisoned &&
              lifecycle_state.active_transfer_generation == 0,
          "abort poisons and retains slot");
    check(abort_prefix_state_transfer(aborted, lifecycle_decode, lifecycle_state, arena) ==
              PrefixStateTransferError::AlreadyCompleted,
          "aborted ticket cannot abort twice");

    return failures == 0 ? 0 : 1;
}
