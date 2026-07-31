#pragma once

#include "tatara/backend/metal/counter_sampling.h"
#include "tatara/runtime/prefill_profile_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tatara::runtime {

inline constexpr std::size_t kInvalidPrefillProfileEventIndex = static_cast<std::size_t>(-1);

inline constexpr std::array<PrefillProfileEventClass, 35> kPrefillProfileDispatchTaxonomy{
    PrefillProfileEventClass::Embedding,
    PrefillProfileEventClass::LayerInputNormalization,
    PrefillProfileEventClass::GdnProjection,
    PrefillProfileEventClass::GdnConvolution,
    PrefillProfileEventClass::GdnGateHoist,
    PrefillProfileEventClass::GdnRecurrenceSerialStep,
    PrefillProfileEventClass::GdnRecurrenceRegisterLoop,
    PrefillProfileEventClass::GdnGateNormalization,
    PrefillProfileEventClass::GdnOutputProjection,
    PrefillProfileEventClass::AttentionProjection,
    PrefillProfileEventClass::AttentionQkRope,
    PrefillProfileEventClass::AttentionPartial,
    PrefillProfileEventClass::AttentionCombine,
    PrefillProfileEventClass::AttentionStagedScores,
    PrefillProfileEventClass::AttentionStagedSoftmax,
    PrefillProfileEventClass::AttentionStagedValues,
    PrefillProfileEventClass::AttentionOutputProjection,
    PrefillProfileEventClass::MoeResidualInput,
    PrefillProfileEventClass::MoePostNormalization,
    PrefillProfileEventClass::MoeRouter,
    PrefillProfileEventClass::MoeRouterSelectSerial,
    PrefillProfileEventClass::MoeRouterSelectParallel,
    PrefillProfileEventClass::MoeExpertUnion,
    PrefillProfileEventClass::MoeRoutedTaskBuild,
    PrefillProfileEventClass::MoeExpertUpGate,
    PrefillProfileEventClass::MoeExpertDown,
    PrefillProfileEventClass::MoeExpertCombine,
    PrefillProfileEventClass::MoeResidualOutput,
    PrefillProfileEventClass::MoeNativeRoutedUpGate,
    PrefillProfileEventClass::MoeSharedExpertUpGate,
    PrefillProfileEventClass::MoeNativeRoutedDown,
    PrefillProfileEventClass::MoeSharedExpertDown,
    PrefillProfileEventClass::MoeNativeRoutedSharedUpGate,
    PrefillProfileEventClass::MoeNativeRoutedSharedDown,
    PrefillProfileEventClass::AttentionStreaming,
};

constexpr bool is_prefill_profile_dispatch_class(PrefillProfileEventClass event_class) noexcept {
    for (const PrefillProfileEventClass known : kPrefillProfileDispatchTaxonomy) {
        if (event_class == known) {
            return true;
        }
    }
    return false;
}

enum class PrefillProfilerState : std::uint8_t {
    Disabled,
    Ready,
    DispatchPending,
    Finalized,
    Failed,
};

enum class PrefillProfilerError : std::uint8_t {
    None,
    Disabled,
    EmptyEventPlan,
    SampleCountOverflow,
    SampleCapacityInsufficient,
    SampleCapacityMismatch,
    EventSequenceExhausted,
    EventMismatch,
    DispatchAlreadyPending,
    DispatchNotPending,
    DispatchTicketMismatch,
    CounterSamplingFailed,
    Incomplete,
    AlreadyFinalized,
    SamplingModeMismatch,
    InvalidSamplingMode,
    SampleWindowRequired,
    SampleWindowAlreadyActive,
    SampleWindowNotActive,
    SampleWindowEmpty,
    SampleWindowCapacityExceeded,
    StageEncoderSplitFailed,
};

struct PrefillProfileDispatchTicket {
    std::size_t event_index{kInvalidPrefillProfileEventIndex};
    backend::metal::CounterSamplePair samples{};
};

struct PrefillProfileSampleWindow {
    std::size_t event_begin{kInvalidPrefillProfileEventIndex};
    std::size_t event_count{0};
    std::size_t sample_count{0};
};

struct PrefillProfilerStatus {
    PrefillProfilerState state{PrefillProfilerState::Disabled};
    PrefillProfilerError error{PrefillProfilerError::None};
    backend::metal::CounterSamplingMode sampling_mode{
        backend::metal::CounterSamplingMode::DispatchBoundary};
    backend::metal::CounterSampleError counter_error{backend::metal::CounterSampleError::None};
    backend::metal::CounterStageSampleError stage_error{
        backend::metal::CounterStageSampleError::None};
    std::size_t event_cursor{0};
    std::size_t event_count{0};
    std::size_t required_sample_count{0};
    std::size_t sample_capacity{0};
    std::size_t mismatch_index{kInvalidPrefillProfileEventIndex};
    PrefillProfileEvent expected_event{};
    PrefillProfileEvent actual_event{};
    bool sample_window_active{false};
    std::size_t sample_window_event_begin{kInvalidPrefillProfileEventIndex};
    std::size_t sample_window_event_count{0};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillProfilerError::None;
    }
};

class PrefillProfiler {
  public:
    PrefillProfiler() noexcept = default;
    PrefillProfiler(std::span<const PrefillProfileEvent> events,
                    std::size_t sample_capacity) noexcept;
    PrefillProfiler(std::span<const PrefillProfileEvent> events,
                    std::size_t sample_capacity,
                    backend::metal::CounterSamplingMode sampling_mode) noexcept;

    PrefillProfiler(const PrefillProfiler&) = delete;
    PrefillProfiler& operator=(const PrefillProfiler&) = delete;
    PrefillProfiler(PrefillProfiler&&) = delete;
    PrefillProfiler& operator=(PrefillProfiler&&) = delete;

    [[nodiscard]] constexpr PrefillProfilerError error() const noexcept {
        return error_;
    }
    [[nodiscard]] PrefillProfilerStatus status() const noexcept;
    [[nodiscard]] PrefillProfilerStatus finalize() noexcept;

    PrefillProfilerError validate_sample_capacity(std::size_t sample_capacity) noexcept;
    PrefillProfilerError
    validate_sampling_mode(backend::metal::CounterSamplingMode sampling_mode) noexcept;
    PrefillProfilerError begin_sample_window() noexcept;
    PrefillProfilerError finish_sample_window(PrefillProfileSampleWindow& window) noexcept;
    PrefillProfilerError
    next_stage_sample_pair(backend::metal::CounterSamplePair& pair) noexcept;
    PrefillProfilerError begin_dispatch(const PrefillProfileEvent& event,
                                        PrefillProfileDispatchTicket& ticket) noexcept;
    PrefillProfilerError complete_dispatch(PrefillProfileDispatchTicket ticket) noexcept;
    PrefillProfilerError fail_counter(backend::metal::CounterSampleError error) noexcept;
    PrefillProfilerError
    fail_stage(backend::metal::CounterStageSampleError error) noexcept;

  private:
    PrefillProfilerError fail(PrefillProfilerError error) noexcept;

    std::span<const PrefillProfileEvent> events_{};
    std::size_t sample_capacity_{0};
    std::size_t required_sample_count_{0};
    std::size_t cursor_{0};
    std::size_t mismatch_index_{kInvalidPrefillProfileEventIndex};
    PrefillProfileEvent expected_event_{};
    PrefillProfileEvent actual_event_{};
    backend::metal::CounterSampleError counter_error_{backend::metal::CounterSampleError::None};
    backend::metal::CounterStageSampleError stage_error_{
        backend::metal::CounterStageSampleError::None};
    backend::metal::CounterSamplingMode sampling_mode_{
        backend::metal::CounterSamplingMode::DispatchBoundary};
    bool sample_window_active_{false};
    std::size_t sample_window_event_begin_{kInvalidPrefillProfileEventIndex};
    std::size_t sample_window_event_count_{0};
    PrefillProfilerState state_{PrefillProfilerState::Disabled};
    PrefillProfilerError error_{PrefillProfilerError::None};
};

static_assert(static_cast<std::size_t>(
                  PrefillProfileEventClass::AttentionStreaming) +
                  1U ==
              kPrefillProfileDispatchTaxonomy.size());

} // namespace tatara::runtime
