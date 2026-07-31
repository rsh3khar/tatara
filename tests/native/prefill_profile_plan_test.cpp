#include "tatara/runtime/prefill_profile_plan.h"
#include "tatara/runtime/prefill_step.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

namespace {

using tatara::model::qwen36::LayerKind;
using tatara::model::qwen36::StaticModelPlan;
using tatara::runtime::PrefillExecutionPolicy;
using tatara::runtime::PrefillGdnRecurrence;
using tatara::runtime::PrefillGeometry;
using tatara::runtime::PrefillPolicy;
using tatara::runtime::PrefillProfileEvent;
using tatara::runtime::PrefillProfileEventClass;
using tatara::runtime::PrefillProfilePlanError;
using tatara::runtime::PrefillRouterSelector;
using tatara::runtime::PrefillSchedule;
using tatara::runtime::QuantizedGemmPolicy;

std::uint64_t g_allocation_count = 0;

static_assert(std::is_trivially_copyable_v<PrefillProfileEvent>);
static_assert(noexcept(tatara::runtime::make_prefill_profile_plan(
    std::declval<const PrefillGeometry&>(),
    std::declval<const PrefillExecutionPolicy&>(), std::uint32_t{},
    std::declval<std::span<const LayerKind>>(),
    std::declval<std::span<PrefillProfileEvent>>())));

constexpr PrefillProfileEvent kCanary{
    .event_class = PrefillProfileEventClass::MoeResidualOutput,
    .layer_index = 0x123456789abcdef0ULL,
    .chunk_ordinal = 0x13579bdfU,
    .chunk_offset = 0x2468ace0U,
    .chunk_rows = 0x11223344U,
    .operation_row_begin = 0x55667788U,
    .operation_row_count = 0x99aabbccU,
};

PrefillExecutionPolicy make_policy(
    PrefillSchedule schedule,
    PrefillGdnRecurrence recurrence = PrefillGdnRecurrence::SerialSteps,
    bool gate_hoist = false,
    PrefillRouterSelector selector = PrefillRouterSelector::Serial) {
    return {
        .geometry =
            {
                .schedule = schedule,
                .context_capacity = 64,
                .maximum_block_rows = 32,
                .first_chunk_rows = 16,
                .query_tile_rows = 16,
                .attention_partition = 256,
                .exact_rows_per_threadgroup = 8,
                .gdn_gate_hoist = gate_hoist,
            },
        .router_selector = selector,
        .gdn_recurrence = recurrence,
    };
}

PrefillGeometry make_geometry(const PrefillExecutionPolicy& policy,
                              std::uint32_t gated_delta_layers = 1,
                              std::uint32_t attention_layers = 1) {
    return {
        .schedule = policy.geometry.schedule,
        .context_capacity = policy.geometry.context_capacity,
        .maximum_block_rows = policy.geometry.maximum_block_rows,
        .first_chunk_rows = policy.geometry.first_chunk_rows,
        .query_tile_rows = policy.geometry.query_tile_rows,
        .attention_partition = policy.geometry.attention_partition,
        .exact_rows_per_threadgroup =
            policy.geometry.exact_rows_per_threadgroup,
        .gdn_gate_hoist = policy.geometry.gdn_gate_hoist,
        .hidden = 512,
        .vocabulary = 1024,
        .query_heads = 8,
        .key_value_heads = 1,
        .attention_head_dimension = 256,
        .recurrent_heads = 16,
        .state_dimension = 128,
        .experts = 16,
        .active_experts = 2,
        .expert_dimension = 512,
        .gdn_projection_rows = 1024,
        .gdn_qk_values = 512,
        .gdn_value_values = 512,
        .attention_projection_rows = 2048,
        .attention_vector_values = 2048,
        .gated_delta_layers = gated_delta_layers,
        .attention_layers = attention_layers,
        .hidden_slab_bytes =
            policy.geometry.schedule == PrefillSchedule::LayerMajor ? 1U : 0U,
        .token_bytes = 1,
        .block_hidden_bytes = 1,
        .gdn_parameter_bytes = policy.geometry.gdn_gate_hoist ? 1U : 0U,
        .reusable_scratch_bytes = 1,
        .steady_prefill_bytes = 1,
    };
}

std::uint64_t count_class(std::span<const PrefillProfileEvent> events,
                          PrefillProfileEventClass event_class) {
    std::uint64_t count = 0;
    for (const PrefillProfileEvent& event : events) {
        count += event.event_class == event_class ? 1U : 0U;
    }
    return count;
}

bool is_event(const PrefillProfileEvent& event,
              PrefillProfileEventClass event_class,
              std::uint64_t layer,
              std::uint32_t chunk,
              std::uint32_t offset,
              std::uint32_t rows,
              std::uint32_t operation_begin,
              std::uint32_t operation_rows) {
    return event.event_class == event_class && event.layer_index == layer &&
           event.chunk_ordinal == chunk && event.chunk_offset == offset &&
           event.chunk_rows == rows &&
           event.operation_row_begin == operation_begin &&
           event.operation_row_count == operation_rows;
}

bool is_moe_sequence(std::span<const PrefillProfileEvent> events,
                     std::size_t begin,
                     PrefillProfileEventClass selector) {
    constexpr std::array<PrefillProfileEventClass, 9> kSerialClasses{
        PrefillProfileEventClass::MoeResidualInput,
        PrefillProfileEventClass::MoePostNormalization,
        PrefillProfileEventClass::MoeRouter,
        PrefillProfileEventClass::MoeRouterSelectSerial,
        PrefillProfileEventClass::MoeExpertUnion,
        PrefillProfileEventClass::MoeExpertUpGate,
        PrefillProfileEventClass::MoeExpertDown,
        PrefillProfileEventClass::MoeExpertCombine,
        PrefillProfileEventClass::MoeResidualOutput,
    };
    if (begin > events.size() ||
        events.size() - begin < kSerialClasses.size()) {
        return false;
    }
    for (std::size_t index = 0; index < kSerialClasses.size(); ++index) {
        const PrefillProfileEventClass expected =
            index == 3 ? selector : kSerialClasses[index];
        if (events[begin + index].event_class != expected) {
            return false;
        }
    }
    return true;
}

int test_layer_major_serial_order() {
    constexpr std::array<LayerKind, 2> kSchedule{
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
    };
    const PrefillExecutionPolicy policy =
        make_policy(PrefillSchedule::LayerMajor);
    const PrefillGeometry geometry = make_geometry(policy);
    std::array<PrefillProfileEvent, 96> events;
    events.fill(kCanary);

    const std::uint64_t allocations_before = g_allocation_count;
    const auto result = tatara::runtime::make_prefill_profile_plan(
        geometry, policy, 17, kSchedule, events);
    if (!result || result.required_event_count != 77 ||
        result.written_event_count != 77 || result.chunk_count != 2 ||
        g_allocation_count != allocations_before || events[77] != kCanary) {
        return 1;
    }

    const auto written =
        std::span<const PrefillProfileEvent>{events}.first(77);
    if (!is_event(events[0], PrefillProfileEventClass::Embedding,
                  tatara::runtime::kNoPrefillProfileLayerIndex, 0, 0, 16, 0,
                  16) ||
        !is_event(events[1],
                  PrefillProfileEventClass::LayerInputNormalization, 0, 0, 0,
                  16, 0, 16) ||
        events[2].event_class != PrefillProfileEventClass::GdnProjection ||
        events[3].event_class != PrefillProfileEventClass::GdnConvolution ||
        !is_event(events[4],
                  PrefillProfileEventClass::GdnRecurrenceSerialStep, 0, 0, 0,
                  16, 0, 1) ||
        !is_event(events[19],
                  PrefillProfileEventClass::GdnRecurrenceSerialStep, 0, 0, 0,
                  16, 15, 1) ||
        events[20].event_class !=
            PrefillProfileEventClass::GdnGateNormalization ||
        events[21].event_class !=
            PrefillProfileEventClass::GdnOutputProjection ||
        !is_moe_sequence(written, 22,
                         PrefillProfileEventClass::MoeRouterSelectSerial)) {
        return 2;
    }
    if (!is_event(events[31], PrefillProfileEventClass::Embedding,
                  tatara::runtime::kNoPrefillProfileLayerIndex, 1, 16, 1, 0,
                  1) ||
        !is_event(events[35],
                  PrefillProfileEventClass::GdnRecurrenceSerialStep, 0, 1, 16,
                  1, 0, 1) ||
        !is_event(events[47],
                  PrefillProfileEventClass::LayerInputNormalization, 1, 0, 0,
                  16, 0, 16) ||
        events[48].event_class !=
            PrefillProfileEventClass::AttentionProjection ||
        events[49].event_class != PrefillProfileEventClass::AttentionQkRope ||
        !is_event(events[50], PrefillProfileEventClass::AttentionPartial, 1,
                  0, 0, 16, 0, 16) ||
        !is_event(events[51], PrefillProfileEventClass::AttentionCombine, 1,
                  0, 0, 16, 0, 16) ||
        events[52].event_class !=
            PrefillProfileEventClass::AttentionOutputProjection ||
        !is_moe_sequence(written, 38,
                         PrefillProfileEventClass::MoeRouterSelectSerial) ||
        !is_moe_sequence(written, 53,
                         PrefillProfileEventClass::MoeRouterSelectSerial) ||
        !is_moe_sequence(written, 68,
                         PrefillProfileEventClass::MoeRouterSelectSerial)) {
        return 3;
    }
    if (count_class(written, PrefillProfileEventClass::Embedding) != 2 ||
        count_class(written,
                    PrefillProfileEventClass::GdnRecurrenceSerialStep) != 17 ||
        count_class(written, PrefillProfileEventClass::AttentionPartial) != 2 ||
        count_class(written, PrefillProfileEventClass::AttentionCombine) != 2) {
        return 4;
    }
    return 0;
}

int test_chunk_major_and_attention_tiles() {
    constexpr std::array<LayerKind, 2> kSchedule{
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
    };
    const PrefillExecutionPolicy policy =
        make_policy(PrefillSchedule::ChunkMajor);
    const PrefillGeometry geometry = make_geometry(policy);
    std::array<PrefillProfileEvent, 64> events;
    events.fill(kCanary);

    const auto result = tatara::runtime::make_prefill_profile_plan(
        geometry, policy, 17, kSchedule, events);
    if (!result || result.required_event_count != 49 ||
        result.written_event_count != 49 || result.chunk_count != 1 ||
        events[49] != kCanary) {
        return 1;
    }
    if (!is_event(events[0], PrefillProfileEventClass::Embedding,
                  tatara::runtime::kNoPrefillProfileLayerIndex, 0, 0, 17, 0,
                  17) ||
        events[31].event_class != PrefillProfileEventClass::MoeResidualOutput ||
        !is_event(events[32],
                  PrefillProfileEventClass::LayerInputNormalization, 1, 0, 0,
                  17, 0, 17) ||
        !is_event(events[35], PrefillProfileEventClass::AttentionPartial, 1,
                  0, 0, 17, 0, 16) ||
        !is_event(events[37], PrefillProfileEventClass::AttentionPartial, 1,
                  0, 0, 17, 16, 1) ||
        events[48].event_class != PrefillProfileEventClass::MoeResidualOutput) {
        return 2;
    }
    return 0;
}

int test_layer_major_resumed_context_uses_full_first_chunk() {
    constexpr std::array<LayerKind, 2> kSchedule{
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
    };
    const PrefillExecutionPolicy policy =
        make_policy(PrefillSchedule::LayerMajor);
    const PrefillGeometry geometry = make_geometry(policy);
    std::array<PrefillProfileEvent, 128> events;
    events.fill(kCanary);

    const auto result = tatara::runtime::make_prefill_profile_plan(
        geometry, policy, 17, kSchedule, events, 16);
    if (!result || result.chunk_count != 1 ||
        result.written_event_count == 0 ||
        result.written_event_count >= events.size() ||
        events[result.written_event_count] != kCanary) {
        return 1;
    }
    const auto written =
        std::span<const PrefillProfileEvent>{events}.first(
            static_cast<std::size_t>(
                result.written_event_count));
    if (count_class(
            written, PrefillProfileEventClass::Embedding) != 1) {
        return 2;
    }
    for (const PrefillProfileEvent& event : written) {
        if (event.chunk_ordinal != 0 ||
            event.chunk_offset != 0 ||
            event.chunk_rows != 17) {
            return 3;
        }
    }
    return 0;
}

int test_register_loop_policies() {
    constexpr std::array<LayerKind, 2> kSchedule{
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
    };
    auto policy = make_policy(PrefillSchedule::LayerMajor,
                              PrefillGdnRecurrence::RegisterLoop, false,
                              PrefillRouterSelector::Parallel);
    auto geometry = make_geometry(policy);
    std::array<PrefillProfileEvent, 96> events;
    events.fill(kCanary);

    auto result = tatara::runtime::make_prefill_profile_plan(
        geometry, policy, 17, kSchedule, events);
    auto written = std::span<const PrefillProfileEvent>{events}.first(
        static_cast<std::size_t>(result.written_event_count));
    if (!result || result.required_event_count != 62 ||
        count_class(written,
                    PrefillProfileEventClass::GdnRecurrenceRegisterLoop) != 2 ||
        count_class(written,
                    PrefillProfileEventClass::GdnRecurrenceSerialStep) != 0 ||
        count_class(written, PrefillProfileEventClass::GdnGateHoist) != 0 ||
        count_class(written,
                    PrefillProfileEventClass::MoeRouterSelectParallel) != 4 ||
        count_class(written,
                    PrefillProfileEventClass::MoeRouterSelectSerial) != 0) {
        return 1;
    }

    policy = make_policy(PrefillSchedule::LayerMajor,
                         PrefillGdnRecurrence::RegisterLoop, true,
                         PrefillRouterSelector::Serial);
    geometry = make_geometry(policy);
    events.fill(kCanary);
    result = tatara::runtime::make_prefill_profile_plan(
        geometry, policy, 17, kSchedule, events);
    written = std::span<const PrefillProfileEvent>{events}.first(
        static_cast<std::size_t>(result.written_event_count));
    if (!result || result.required_event_count != 64 ||
        count_class(written, PrefillProfileEventClass::GdnGateHoist) != 2 ||
        count_class(written,
                    PrefillProfileEventClass::GdnRecurrenceRegisterLoop) != 2) {
        return 2;
    }
    return 0;
}

constexpr StaticModelPlan<4> kSecondPlan{
    .id = "profile-second",
    .family = "qwen3_5_moe",
    .package_sha256 =
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    .artifact =
        {
            .id = "profile-second-artifact",
            .model_type = "qwen3_5_moe",
            .format = "safetensors",
            .source_repository = "local",
            .source_revision = "main",
            .manifest_sha256 =
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            .tensor_count = 1,
            .tensor_bytes = 1,
            .file_count = 1,
            .weight_file_count = 1,
        },
    .dimensions = {.hidden = 2048, .vocabulary = 32768},
    .attention =
        {.query_heads = 16, .key_value_heads = 2, .head_dimension = 256},
    .gated_delta = {.recurrent_heads = 32, .state_dimension = 128},
    .mixture_of_experts =
        {.experts = 64, .active_experts = 4, .expert_dimension = 512},
    .weights =
        {.format = tatara::model::qwen36::WeightFormat::AffineQ4,
         .group_size = 64},
    .initial_serving_capacity = 64,
    .layers =
        {
            LayerKind::GatedDelta,
            LayerKind::GatedDelta,
            LayerKind::GatedDelta,
            LayerKind::FullAttention,
        },
};

int test_second_plan_and_policies() {
    auto policy = make_policy(PrefillSchedule::LayerMajor,
                              PrefillGdnRecurrence::RegisterLoop, true,
                              PrefillRouterSelector::Parallel);
    const auto layer_geometry =
        tatara::runtime::make_prefill_geometry(kSecondPlan, policy.geometry);
    if (!layer_geometry) {
        return 1;
    }
    std::array<PrefillProfileEvent, 256> events;
    events.fill(kCanary);
    auto result = tatara::runtime::make_prefill_profile_plan(
        layer_geometry.geometry, policy, 35, kSecondPlan.layers, events);
    if (!result || result.required_event_count != 130 ||
        result.chunk_count != 2 || events[130] != kCanary) {
        return 2;
    }

    policy = make_policy(PrefillSchedule::ChunkMajor,
                         PrefillGdnRecurrence::SerialSteps, false,
                         PrefillRouterSelector::Serial);
    const auto chunk_geometry =
        tatara::runtime::make_prefill_geometry(kSecondPlan, policy.geometry);
    if (!chunk_geometry) {
        return 3;
    }
    events.fill(kCanary);
    result = tatara::runtime::make_prefill_profile_plan(
        chunk_geometry.geometry, policy, 35, kSecondPlan.layers, events);
    const auto written = std::span<const PrefillProfileEvent>{events}.first(
        static_cast<std::size_t>(result.written_event_count));
    if (!result || result.required_event_count != 223 ||
        result.chunk_count != 2 || events[223] != kCanary ||
        count_class(written, PrefillProfileEventClass::Embedding) != 2 ||
        count_class(written,
                    PrefillProfileEventClass::GdnRecurrenceSerialStep) != 105) {
        return 4;
    }
    return 0;
}

int test_invalid_capacity_and_canaries() {
    constexpr std::array<LayerKind, 2> kSchedule{
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
    };
    const PrefillExecutionPolicy policy =
        make_policy(PrefillSchedule::LayerMajor);
    const PrefillGeometry geometry = make_geometry(policy);
    std::array<PrefillProfileEvent, 96> events;
    events.fill(kCanary);

    const auto insufficient = tatara::runtime::make_prefill_profile_plan(
        geometry, policy, 17, kSchedule,
        std::span<PrefillProfileEvent>{events}.first(76));
    if (insufficient.error !=
            PrefillProfilePlanError::EventCapacityInsufficient ||
        insufficient.required_event_count != 77 ||
        insufficient.written_event_count != 0) {
        return 1;
    }
    for (const PrefillProfileEvent& event : events) {
        if (event != kCanary) {
            return 2;
        }
    }

    if (tatara::runtime::make_prefill_profile_plan(
            PrefillGeometry{}, policy, 17, kSchedule, events)
            .error != PrefillProfilePlanError::InvalidGeometry) {
        return 3;
    }
    auto invalid_policy = policy;
    invalid_policy.router_selector =
        static_cast<PrefillRouterSelector>(255);
    if (tatara::runtime::make_prefill_profile_plan(
            geometry, invalid_policy, 17, kSchedule, events)
            .error != PrefillProfilePlanError::InvalidPolicy) {
        return 4;
    }
    invalid_policy = policy;
    invalid_policy.geometry.gdn_gate_hoist = true;
    if (tatara::runtime::make_prefill_profile_plan(
            geometry, invalid_policy, 17, kSchedule, events)
            .error != PrefillProfilePlanError::InvalidPolicy) {
        return 5;
    }
    if (tatara::runtime::make_prefill_profile_plan(
            geometry, policy, 0, kSchedule, events)
            .error != PrefillProfilePlanError::EmptyRequest ||
        tatara::runtime::make_prefill_profile_plan(
            geometry, policy, 65, kSchedule, events)
                .error != PrefillProfilePlanError::RequestRowsOutOfRange) {
        return 6;
    }

    constexpr std::array<LayerKind, 1> kShortSchedule{
        LayerKind::GatedDelta};
    constexpr std::array<LayerKind, 2> kWrongCounts{
        LayerKind::GatedDelta,
        LayerKind::GatedDelta,
    };
    constexpr std::array<LayerKind, 2> kUnknownKind{
        LayerKind::GatedDelta,
        static_cast<LayerKind>(255),
    };
    if (tatara::runtime::make_prefill_profile_plan(
            geometry, policy, 17, kShortSchedule, events)
            .error != PrefillProfilePlanError::InvalidSchedule ||
        tatara::runtime::make_prefill_profile_plan(
            geometry, policy, 17, kWrongCounts, events)
                .error != PrefillProfilePlanError::InvalidSchedule ||
        tatara::runtime::make_prefill_profile_plan(
            geometry, policy, 17, kUnknownKind, events)
                .error != PrefillProfilePlanError::InvalidSchedule) {
        return 7;
    }
    for (const PrefillProfileEvent& event : events) {
        if (event != kCanary) {
            return 8;
        }
    }
    return 0;
}

int test_native_dense_dispatch_events() {
    constexpr std::array<LayerKind, 2> kSchedule{
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
    };
    PrefillExecutionPolicy policy =
        make_policy(PrefillSchedule::LayerMajor);
    policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
    const PrefillGeometry geometry = make_geometry(policy);
    std::array<PrefillProfileEvent, 64> events;
    events.fill(kCanary);

    const auto result = tatara::runtime::make_prefill_profile_plan(
        geometry, policy, 1, kSchedule, events);
    if (!result || result.required_event_count != 36 ||
        result.written_event_count != 36 ||
        events[36] != kCanary) {
        return 1;
    }
    const auto written =
        std::span<const PrefillProfileEvent>{events}.first(36);
    if (count_class(
            written, PrefillProfileEventClass::GdnProjection) != 4 ||
        count_class(
            written,
            PrefillProfileEventClass::AttentionProjection) != 3 ||
        count_class(
            written,
            PrefillProfileEventClass::GdnOutputProjection) != 1 ||
        count_class(
            written,
            PrefillProfileEventClass::AttentionOutputProjection) != 1) {
        return 2;
    }
    return 0;
}

int test_native_routed_dispatch_events() {
    constexpr std::array<LayerKind, 2> kSchedule{
        LayerKind::GatedDelta,
        LayerKind::FullAttention,
    };
    PrefillExecutionPolicy policy =
        make_policy(PrefillSchedule::LayerMajor);
    policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
    policy.routed_qgemm = QuantizedGemmPolicy::NativeRaggedMma;
    const PrefillGeometry geometry = make_geometry(policy);
    std::array<PrefillProfileEvent, 64> events;
    events.fill(kCanary);

    const auto result = tatara::runtime::make_prefill_profile_plan(
        geometry, policy, 1, kSchedule, events);
    if (!result || result.required_event_count != 44 ||
        result.written_event_count != 44 ||
        events[44] != kCanary) {
        return 1;
    }
    const auto written =
        std::span<const PrefillProfileEvent>{events}.first(44);
    if (count_class(
            written,
            PrefillProfileEventClass::MoeRoutedTaskBuild) != 4 ||
        count_class(
            written,
            PrefillProfileEventClass::MoeNativeRoutedUpGate) != 2 ||
        count_class(
            written,
            PrefillProfileEventClass::MoeSharedExpertUpGate) != 2 ||
        count_class(
            written,
            PrefillProfileEventClass::MoeNativeRoutedDown) != 2 ||
        count_class(
            written,
            PrefillProfileEventClass::MoeSharedExpertDown) != 2 ||
        count_class(
            written,
            PrefillProfileEventClass::MoeExpertUpGate) != 0 ||
        count_class(
            written,
            PrefillProfileEventClass::MoeExpertDown) != 0) {
        return 2;
    }

    policy.native_routed_shared_expert = true;
    events.fill(kCanary);
    const auto assimilated =
        tatara::runtime::make_prefill_profile_plan(
            geometry, policy, 1, kSchedule, events);
    if (!assimilated ||
        assimilated.required_event_count != 40 ||
        assimilated.written_event_count != 40 ||
        events[40] != kCanary) {
        return 3;
    }
    const auto assimilated_events =
        std::span<const PrefillProfileEvent>{events}.first(40);
    if (count_class(
            assimilated_events,
            PrefillProfileEventClass::MoeRoutedTaskBuild) != 4 ||
        count_class(
            assimilated_events,
            PrefillProfileEventClass::
                MoeNativeRoutedSharedUpGate) != 2 ||
        count_class(
            assimilated_events,
            PrefillProfileEventClass::
                MoeNativeRoutedSharedDown) != 2 ||
        count_class(
            assimilated_events,
            PrefillProfileEventClass::MoeNativeRoutedUpGate) != 0 ||
        count_class(
            assimilated_events,
            PrefillProfileEventClass::MoeNativeRoutedDown) != 0 ||
        count_class(
            assimilated_events,
            PrefillProfileEventClass::MoeSharedExpertUpGate) != 0 ||
        count_class(
            assimilated_events,
            PrefillProfileEventClass::MoeSharedExpertDown) != 0) {
        return 4;
    }

    PrefillExecutionPolicy chunk_policy =
        make_policy(PrefillSchedule::ChunkMajor);
    chunk_policy.routed_qgemm =
        QuantizedGemmPolicy::NativeRaggedMma;
    const PrefillGeometry chunk_geometry =
        make_geometry(chunk_policy);
    if (tatara::runtime::make_prefill_profile_plan(
            chunk_geometry, chunk_policy, 1, kSchedule, events)
            .error != PrefillProfilePlanError::InvalidPolicy) {
        return 5;
    }
    PrefillExecutionPolicy invalid_shared_policy =
        make_policy(PrefillSchedule::LayerMajor);
    invalid_shared_policy.native_routed_shared_expert = true;
    if (tatara::runtime::make_prefill_profile_plan(
            make_geometry(invalid_shared_policy),
            invalid_shared_policy, 1, kSchedule, events)
            .error != PrefillProfilePlanError::InvalidPolicy) {
        return 6;
    }
    return 0;
}

} // namespace

void* operator new(std::size_t size) {
    ++g_allocation_count;
    if (void* memory = std::malloc(size)) {
        return memory;
    }
    std::abort();
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void* operator new[](std::size_t size) {
    ++g_allocation_count;
    if (void* memory = std::malloc(size)) {
        return memory;
    }
    std::abort();
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

int main() {
    if (const int result = test_layer_major_serial_order()) {
        return 10 + result;
    }
    if (const int result = test_chunk_major_and_attention_tiles()) {
        return 20 + result;
    }
    if (const int result =
            test_layer_major_resumed_context_uses_full_first_chunk()) {
        return 30 + result;
    }
    if (const int result = test_register_loop_policies()) {
        return 40 + result;
    }
    if (const int result = test_second_plan_and_policies()) {
        return 50 + result;
    }
    if (const int result = test_invalid_capacity_and_canaries()) {
        return 60 + result;
    }
    if (const int result = test_native_dense_dispatch_events()) {
        return 70 + result;
    }
    if (const int result = test_native_routed_dispatch_events()) {
        return 80 + result;
    }
    return 0;
}
