#pragma once

#include "tatara/runtime/prefill_geometry.h"

#include <cstdint>
#include <span>

namespace tatara::runtime {

struct PrefillExecutionPolicy;

inline constexpr std::uint64_t kNoPrefillProfileLayerIndex = UINT64_MAX;

enum class PrefillProfileEventClass : std::uint8_t {
    Embedding,
    LayerInputNormalization,
    GdnProjection,
    GdnConvolution,
    GdnGateHoist,
    GdnRecurrenceSerialStep,
    GdnRecurrenceRegisterLoop,
    GdnGateNormalization,
    GdnOutputProjection,
    AttentionProjection,
    AttentionQkRope,
    AttentionPartial,
    AttentionCombine,
    AttentionStagedScores,
    AttentionStagedSoftmax,
    AttentionStagedValues,
    AttentionOutputProjection,
    MoeResidualInput,
    MoePostNormalization,
    MoeRouter,
    MoeRouterSelectSerial,
    MoeRouterSelectParallel,
    MoeExpertUnion,
    MoeRoutedTaskBuild,
    MoeExpertUpGate,
    MoeExpertDown,
    MoeExpertCombine,
    MoeResidualOutput,
    MoeNativeRoutedUpGate,
    MoeSharedExpertUpGate,
    MoeNativeRoutedDown,
    MoeSharedExpertDown,
    MoeNativeRoutedSharedUpGate,
    MoeNativeRoutedSharedDown,
    AttentionStreaming,
};

struct PrefillProfileEvent {
    PrefillProfileEventClass event_class{PrefillProfileEventClass::Embedding};
    std::uint64_t layer_index{kNoPrefillProfileLayerIndex};
    std::uint32_t chunk_ordinal{0};
    std::uint32_t chunk_offset{0};
    std::uint32_t chunk_rows{0};
    std::uint32_t operation_row_begin{0};
    std::uint32_t operation_row_count{0};

    constexpr bool operator==(const PrefillProfileEvent&) const noexcept =
        default;
};

enum class PrefillProfilePlanError : std::uint8_t {
    None,
    InvalidGeometry,
    InvalidPolicy,
    InvalidSchedule,
    EmptyRequest,
    RequestRowsOutOfRange,
    ArithmeticOverflow,
    EventCapacityInsufficient,
};

struct PrefillProfilePlanResult {
    PrefillProfilePlanError error{PrefillProfilePlanError::InvalidGeometry};
    std::uint64_t required_event_count{0};
    std::uint64_t written_event_count{0};
    std::uint32_t chunk_count{0};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillProfilePlanError::None;
    }
};

[[nodiscard]] PrefillProfilePlanResult make_prefill_profile_plan(
    const PrefillGeometry& geometry,
    const PrefillExecutionPolicy& policy,
    std::uint32_t request_rows,
    std::span<const model::qwen36::LayerKind> schedule,
    std::span<PrefillProfileEvent> output,
    std::uint32_t initial_context = 0) noexcept;

} // namespace tatara::runtime
