#include "tatara/runtime/prefill_step.h"

#include "tatara/generated/model_plan.h"

namespace {

using namespace tatara::runtime;

void bind_state(DecodeStep& decode, std::uint32_t capacity = 100) {
    decode.capacity = capacity;
    decode.state.capacity = capacity;
    decode.state.schedule_identity = decode.schedule.data();
    decode.state.layers.resize(decode.schedule.size());
}

} // namespace

int main() {
    const tatara::backend::metal::MetalDevice invalid_device;
    const PrefillGeometry invalid_geometry;
    PrefillExecutionPolicy policy;
    PrefillPipelines pipelines;
    static_assert(PrefillExecutionPolicy{}.gdn_recurrence == PrefillGdnRecurrence::SerialSteps);
    static_assert(
        PrefillExecutionPolicy{}.attention_kernel ==
        PrefillAttentionKernel::PartialCombine);
    static_assert(PrefillExecutionPolicy{}.dense_qgemm ==
                  QuantizedGemmPolicy::ExactRow);
    static_assert(PrefillExecutionPolicy{}.routed_qgemm ==
                  QuantizedGemmPolicy::ExactRow);
    static_assert(
        !PrefillExecutionPolicy{}.native_routed_shared_expert);
    static_assert(!PrefillExecutionPolicy{}.native_dense_steel);
    static_assert(!PrefillExecutionPolicy{}.native_routed_steel);
    static_assert(!PrefillExecutionPolicy{}.command_graph);
    static_assert(
        !PrefillExecutionPolicy{}.command_graph_lane_events);
    static_assert(
        PrefillExecutionPolicy{}.command_graph_chunk_count == 1);
    static_assert(
        PrefillExecutionPolicy{}.maximum_units_per_submission == 1);
    static_assert(
        PrefillExecutionPolicy{}.maximum_inflight_units == 1);
    if (create_prefill_step(invalid_device, invalid_geometry, policy, std::move(pipelines)).error !=
        PrefillStepError::InvalidDevice) {
        return 1;
    }

    DecodeStep decode;
    decode.schedule = {
        tatara::model::qwen36::LayerKind::GatedDelta,
        tatara::model::qwen36::LayerKind::FullAttention,
        tatara::model::qwen36::LayerKind::GatedDelta,
    };
    bind_state(decode);
    advance_prefill_state(decode, 2);
    if (decode.state.layers[0].swapped || decode.state.layers[1].swapped ||
        decode.state.layers[2].swapped) {
        return 2;
    }
    advance_prefill_state(decode, 3);
    if (!decode.state.layers[0].swapped || decode.state.layers[1].swapped ||
        !decode.state.layers[2].swapped) {
        return 3;
    }

    DecodeStep layer_decode;
    layer_decode.schedule = {
        tatara::model::qwen36::LayerKind::GatedDelta,
        tatara::model::qwen36::LayerKind::FullAttention,
        tatara::model::qwen36::LayerKind::GatedDelta,
    };
    bind_state(layer_decode);
    PrefillStep layer_step;
    layer_step.policy.geometry = {
        .schedule = PrefillSchedule::LayerMajor,
        .context_capacity = 100,
        .maximum_block_rows = 2,
        .first_chunk_rows = 2,
        .query_tile_rows = 2,
    };
    layer_step.progress = {
        .state = PrefillProgressState::Ready,
        .owner = &layer_decode,
        .state_owner = &layer_decode.state,
        .live_context = 10,
        .context_base = 10,
        .next_context = 15,
        .row_count = 5,
        .chunk_count = 3,
        .current_layer = 0,
        .current_chunk = 0,
    };
    if (commit_prefill_unit(layer_step, layer_decode).error !=
        PrefillProgressError::UnitNotPending) {
        return 4;
    }
    for (std::uint32_t unit = 0; unit < 9; ++unit) {
        layer_step.progress.state = PrefillProgressState::UnitPending;
        layer_step.progress.pending_unit_count = 1;
        const PrefillProgressResult committed = commit_prefill_unit(layer_step, layer_decode);
        if (!committed || committed.layer_index != unit / 3u ||
            committed.chunk_ordinal != unit % 3u) {
            return 5;
        }
        if (unit < 2 && layer_decode.state.layers[0].swapped) {
            return 6;
        }
        if (unit == 2 && !layer_decode.state.layers[0].swapped) {
            return 7;
        }
        if (unit < 8 && committed.state != PrefillProgressState::Ready) {
            return 8;
        }
    }
    if (layer_step.progress.state != PrefillProgressState::Complete ||
        !layer_decode.state.layers[0].swapped || layer_decode.state.layers[1].swapped ||
        !layer_decode.state.layers[2].swapped) {
        return 9;
    }
    const PrefillProgressResult layer_complete = commit_prefill_unit(layer_step, layer_decode);
    if (layer_complete.error != PrefillProgressError::Complete ||
        layer_complete.next_context != 10) {
        return 10;
    }

    DecodeStep even_decode;
    even_decode.schedule = {tatara::model::qwen36::LayerKind::GatedDelta};
    bind_state(even_decode);
    PrefillStep even_step;
    even_step.policy.geometry = {
        .schedule = PrefillSchedule::LayerMajor,
        .context_capacity = 100,
        .maximum_block_rows = 2,
        .first_chunk_rows = 2,
        .query_tile_rows = 2,
    };
    even_step.progress = {
        .state = PrefillProgressState::UnitPending,
        .owner = &even_decode,
        .state_owner = &even_decode.state,
        .live_context = 0,
        .context_base = 0,
        .next_context = 4,
        .row_count = 4,
        .chunk_count = 2,
        .current_layer = 0,
        .current_chunk = 0,
        .pending_unit_count = 1,
    };
    if (!commit_prefill_unit(even_step, even_decode)) {
        return 11;
    }
    even_step.progress.state = PrefillProgressState::UnitPending;
    even_step.progress.pending_unit_count = 1;
    const PrefillProgressResult even_complete = commit_prefill_unit(even_step, even_decode);
    if (!even_complete || even_complete.state != PrefillProgressState::Complete ||
        even_complete.next_context != 4 || even_decode.state.layers[0].swapped) {
        return 12;
    }

    DecodeStep chunk_decode;
    chunk_decode.schedule = {
        tatara::model::qwen36::LayerKind::GatedDelta,
        tatara::model::qwen36::LayerKind::FullAttention,
        tatara::model::qwen36::LayerKind::GatedDelta,
    };
    bind_state(chunk_decode);
    PrefillStep chunk_step;
    chunk_step.policy.geometry = {
        .schedule = PrefillSchedule::ChunkMajor,
        .context_capacity = 100,
        .maximum_block_rows = 2,
        .first_chunk_rows = 2,
        .query_tile_rows = 2,
    };
    chunk_step.progress = {
        .state = PrefillProgressState::UnitPending,
        .owner = &chunk_decode,
        .state_owner = &chunk_decode.state,
        .live_context = 20,
        .context_base = 20,
        .next_context = 25,
        .row_count = 5,
        .chunk_count = 3,
        .current_layer = 0,
        .current_chunk = 0,
        .pending_unit_count = 1,
    };
    for (std::uint32_t chunk = 0; chunk < 3; ++chunk) {
        if (chunk != 0) {
            chunk_step.progress.state = PrefillProgressState::UnitPending;
            chunk_step.progress.pending_unit_count = 1;
        }
        const PrefillProgressResult committed = commit_prefill_unit(chunk_step, chunk_decode);
        if (!committed || committed.chunk_ordinal != chunk ||
            chunk_decode.state.layers[0].swapped != ((chunk & 1u) == 0u) ||
            chunk_decode.state.layers[1].swapped ||
            chunk_decode.state.layers[2].swapped != ((chunk & 1u) == 0u)) {
            return 13;
        }
    }
    if (chunk_step.progress.state != PrefillProgressState::Complete) {
        return 14;
    }

    PrefillStep poisoned_step;
    poisoned_step.progress.state = PrefillProgressState::Ready;
    if (poison_prefill(poisoned_step) != PrefillProgressError::None ||
        poisoned_step.progress.state != PrefillProgressState::Poisoned ||
        poison_prefill(poisoned_step) != PrefillProgressError::Poisoned) {
        return 15;
    }
    if (commit_prefill_unit(poisoned_step, chunk_decode).error != PrefillProgressError::Poisoned) {
        return 16;
    }

    PrefillStep invalid_step;
    invalid_step.policy.geometry = {
        .schedule = PrefillSchedule::LayerMajor,
        .context_capacity = 100,
        .maximum_block_rows = 2,
        .first_chunk_rows = 2,
        .query_tile_rows = 2,
    };
    invalid_step.progress = {
        .state = PrefillProgressState::UnitPending,
        .owner = &layer_decode,
        .state_owner = &layer_decode.state,
        .live_context = 0,
        .context_base = 0,
        .next_context = 5,
        .row_count = 5,
        .chunk_count = 3,
        .current_layer = 0,
        .current_chunk = 3,
        .pending_unit_count = 1,
    };
    if (commit_prefill_unit(invalid_step, layer_decode).error != PrefillProgressError::Invalid ||
        invalid_step.progress.state != PrefillProgressState::Poisoned) {
        return 17;
    }

    DecodeStep owner_decode;
    owner_decode.schedule = {tatara::model::qwen36::LayerKind::GatedDelta};
    bind_state(owner_decode);
    DecodeStep other_decode;
    other_decode.schedule = owner_decode.schedule;
    bind_state(other_decode);
    PrefillStep wrong_owner_step;
    wrong_owner_step.policy.geometry = {
        .schedule = PrefillSchedule::LayerMajor,
        .context_capacity = 100,
        .maximum_block_rows = 2,
        .first_chunk_rows = 2,
        .query_tile_rows = 2,
    };
    wrong_owner_step.progress = {
        .state = PrefillProgressState::UnitPending,
        .owner = &owner_decode,
        .state_owner = &owner_decode.state,
        .live_context = 0,
        .context_base = 0,
        .next_context = 2,
        .row_count = 2,
        .chunk_count = 1,
        .current_layer = 0,
        .current_chunk = 0,
        .pending_unit_count = 1,
    };
    if (commit_prefill_unit(wrong_owner_step, other_decode).error !=
            PrefillProgressError::Invalid ||
        wrong_owner_step.progress.state != PrefillProgressState::Poisoned ||
        owner_decode.state.layers[0].swapped || other_decode.state.layers[0].swapped) {
        return 18;
    }

    PrefillStep wrong_state_step;
    wrong_state_step.policy = wrong_owner_step.policy;
    wrong_state_step.progress = {
        .state = PrefillProgressState::UnitPending,
        .owner = &owner_decode,
        .state_owner = &owner_decode.state,
        .live_context = 0,
        .context_base = 0,
        .next_context = 2,
        .row_count = 2,
        .chunk_count = 1,
        .current_layer = 0,
        .current_chunk = 0,
        .pending_unit_count = 1,
    };
    if (commit_prefill_unit(wrong_state_step, owner_decode, other_decode.state).error !=
            PrefillProgressError::Invalid ||
        wrong_state_step.progress.state != PrefillProgressState::Poisoned ||
        owner_decode.state.layers[0].swapped || other_decode.state.layers[0].swapped) {
        return 19;
    }

    DecodeStep batch_decode;
    batch_decode.schedule = {
        tatara::model::qwen36::LayerKind::GatedDelta,
        tatara::model::qwen36::LayerKind::FullAttention,
        tatara::model::qwen36::LayerKind::GatedDelta,
    };
    bind_state(batch_decode);
    PrefillStep batch_step;
    batch_step.policy.geometry = {
        .schedule = PrefillSchedule::LayerMajor,
        .context_capacity = 100,
        .maximum_block_rows = 2,
        .first_chunk_rows = 2,
        .query_tile_rows = 2,
    };
    batch_step.policy.maximum_units_per_submission = 4;
    batch_step.progress = {
        .state = PrefillProgressState::BatchPending,
        .owner = &batch_decode,
        .state_owner = &batch_decode.state,
        .live_context = 0,
        .context_base = 0,
        .next_context = 5,
        .row_count = 5,
        .chunk_count = 3,
        .current_layer = 0,
        .current_chunk = 1,
        .pending_unit_count = 4,
    };
    if (commit_prefill_unit(batch_step, batch_decode).error !=
            PrefillProgressError::BatchPending ||
        batch_step.progress.current_layer != 0 ||
        batch_step.progress.current_chunk != 1 ||
        batch_decode.state.layers[0].swapped) {
        return 20;
    }
    const PrefillProgressResult first_batch =
        commit_prefill_units(batch_step, batch_decode);
    if (!first_batch || first_batch.unit_count != 4 ||
        first_batch.layer_index != 0 ||
        first_batch.chunk_ordinal != 1 ||
        batch_step.progress.state != PrefillProgressState::Ready ||
        batch_step.progress.current_layer != 1 ||
        batch_step.progress.current_chunk != 2 ||
        batch_step.progress.pending_unit_count != 0 ||
        !batch_decode.state.layers[0].swapped ||
        batch_decode.state.layers[1].swapped ||
        batch_decode.state.layers[2].swapped) {
        return 21;
    }
    batch_step.progress.state = PrefillProgressState::BatchPending;
    batch_step.progress.pending_unit_count = 4;
    const PrefillProgressResult final_batch =
        commit_prefill_units(batch_step, batch_decode);
    if (!final_batch || final_batch.unit_count != 4 ||
        final_batch.state != PrefillProgressState::Complete ||
        final_batch.next_context != 5 ||
        batch_step.progress.pending_unit_count != 0 ||
        !batch_decode.state.layers[0].swapped ||
        batch_decode.state.layers[1].swapped ||
        !batch_decode.state.layers[2].swapped) {
        return 22;
    }

    PrefillStep oversized_batch_step;
    oversized_batch_step.policy = batch_step.policy;
    oversized_batch_step.progress = {
        .state = PrefillProgressState::BatchPending,
        .owner = &batch_decode,
        .state_owner = &batch_decode.state,
        .live_context = 0,
        .context_base = 0,
        .next_context = 5,
        .row_count = 5,
        .chunk_count = 3,
        .current_layer = 0,
        .current_chunk = 0,
        .pending_unit_count = 5,
    };
    if (commit_prefill_units(oversized_batch_step, batch_decode).error !=
            PrefillProgressError::Invalid ||
        oversized_batch_step.progress.state !=
            PrefillProgressState::Poisoned ||
        batch_decode.state.layers[0].swapped != true ||
        batch_decode.state.layers[1].swapped ||
        batch_decode.state.layers[2].swapped != true) {
        return 23;
    }

    DecodeStep inflight_decode;
    inflight_decode.schedule = {
        tatara::model::qwen36::LayerKind::GatedDelta,
        tatara::model::qwen36::LayerKind::FullAttention,
    };
    bind_state(inflight_decode);
    PrefillStep inflight_step;
    inflight_step.policy.geometry = {
        .schedule = PrefillSchedule::LayerMajor,
        .context_capacity = 100,
        .maximum_block_rows = 2,
        .first_chunk_rows = 2,
        .query_tile_rows = 2,
    };
    inflight_step.policy.maximum_inflight_units = 2;
    inflight_step.progress = {
        .state = PrefillProgressState::InflightEncoding,
        .owner = &inflight_decode,
        .state_owner = &inflight_decode.state,
        .live_context = 0,
        .context_base = 0,
        .next_context = 5,
        .row_count = 5,
        .chunk_count = 3,
        .current_layer = 0,
        .current_chunk = 1,
        .pending_unit_count = 1,
    };
    if (commit_prefill_inflight(inflight_step, inflight_decode).error !=
            PrefillProgressError::InflightEncoding ||
        inflight_step.progress.current_layer != 0 ||
        inflight_step.progress.current_chunk != 1 ||
        inflight_decode.state.layers[0].swapped) {
        return 24;
    }
    inflight_step.progress.state =
        PrefillProgressState::InflightPending;
    inflight_step.progress.pending_unit_count = 2;
    const PrefillProgressResult inflight_committed =
        commit_prefill_inflight(inflight_step, inflight_decode);
    if (!inflight_committed ||
        inflight_committed.unit_count != 2 ||
        inflight_step.progress.state !=
            PrefillProgressState::Ready ||
        inflight_step.progress.current_layer != 1 ||
        inflight_step.progress.current_chunk != 0 ||
        inflight_step.progress.pending_unit_count != 0 ||
        !inflight_decode.state.layers[0].swapped ||
        inflight_decode.state.layers[1].swapped) {
        return 25;
    }

    const PrefillPolicy band_geometry{
        .schedule = PrefillSchedule::LayerMajor,
        .context_capacity = 20000,
        .maximum_block_rows = 2048,
        .first_chunk_rows = 256,
        .query_tile_rows = 256,
        .attention_partition = 256,
        .exact_rows_per_threadgroup = 16,
        .gdn_gate_hoist = true,
    };
    const PrefillBandPlan rows3925 =
        plan_next_prefill_band(band_geometry, 0, 3925, 3);
    if (!rows3925 || rows3925.context_base != 0 ||
        rows3925.row_count != 3925 || rows3925.chunk_count != 3 ||
        rows3925.next_context != 3925) {
        return 26;
    }
    const PrefillBandPlan first_long =
        plan_next_prefill_band(band_geometry, 0, 10000, 3);
    const PrefillBandPlan second_long =
        plan_next_prefill_band(
            band_geometry, first_long.next_context,
            10000 - first_long.row_count, 3);
    if (!first_long || first_long.row_count != 4352 ||
        first_long.chunk_count != 3 || first_long.next_context != 4352 ||
        !second_long || second_long.context_base != 4352 ||
        second_long.row_count != 5648 || second_long.chunk_count != 3 ||
        second_long.next_context != 10000) {
        return 27;
    }
    if (plan_next_prefill_band(band_geometry, 1, 10, 3).error !=
            PrefillBandPlanError::ContextOutOfRange ||
        plan_next_prefill_band(band_geometry, 0, 0, 3).error !=
            PrefillBandPlanError::Empty ||
        plan_next_prefill_band(band_geometry, 0, 20001, 3).error !=
            PrefillBandPlanError::ContextOverflow) {
        return 28;
    }

    const auto& generated_plan =
        tatara::model::qwen36::generated::kModelPlan;
    const PrefillPolicy memory_geometry_policy{
        .schedule = PrefillSchedule::LayerMajor,
        .context_capacity = 65536,
        .maximum_block_rows = 2048,
        .first_chunk_rows = 256,
        .query_tile_rows = 256,
        .attention_partition = 256,
        .exact_rows_per_threadgroup = 16,
        .gdn_gate_hoist = true,
    };
    const auto memory_geometry =
        make_prefill_geometry(generated_plan, memory_geometry_policy);
    if (!memory_geometry) {
        return 29;
    }
    const PrefillExecutionPolicy memory_policy{
        .geometry = memory_geometry_policy,
        .router_selector = PrefillRouterSelector::Parallel,
        .gdn_recurrence = PrefillGdnRecurrence::RegisterLoop,
        .attention_kernel = PrefillAttentionKernel::StagedGemmAdaptive,
        .dense_qgemm = QuantizedGemmPolicy::NativeDenseMma,
        .routed_qgemm = QuantizedGemmPolicy::NativeRaggedMma,
        .native_dense_steel = true,
        .native_routed_shared_expert = true,
        .native_routed_steel = true,
        .command_graph = true,
        .command_graph_chunk_count = 3,
        .maximum_units_per_submission = 1,
        .maximum_inflight_units = 1,
    };
    const PrefillMemoryPlan memory =
        plan_prefill_step_memory(memory_geometry.geometry, memory_policy);
    if (!memory || memory.maximum_scratch_lanes != 3 ||
        memory.maximum_task_status_count != 120 ||
        memory.native_routed_workspace_bytes != 394276 ||
        memory.total_bytes != 4808064132ULL) {
        return 30;
    }
    constexpr std::uint64_t gib = 1ULL << 30U;
    const PrefillBufferWindowPlan below_limit =
        plan_prefill_buffer_window(8 * gib, 4 * gib - 1);
    const PrefillBufferWindowPlan at_limit =
        plan_prefill_buffer_window(8 * gib, 4 * gib);
    const PrefillBufferWindowPlan second_window =
        plan_prefill_buffer_window(10 * gib, 7 * gib);
    if (!below_limit || below_limit.use_window ||
        below_limit.binding_offset != 4 * gib - 1 ||
        !at_limit || !at_limit.use_window ||
        at_limit.source_begin != 4 * gib ||
        at_limit.window_length != 4 * gib ||
        at_limit.binding_offset != 0 ||
        !second_window || !second_window.use_window ||
        second_window.source_begin != 6 * gib ||
        second_window.window_length != 4 * gib ||
        second_window.binding_offset != gib ||
        plan_prefill_buffer_window(8 * gib, 8 * gib)) {
        return 31;
    }
    return 0;
}
