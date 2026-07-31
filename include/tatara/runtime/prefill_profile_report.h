#pragma once

#include "tatara/backend/metal/counter_sampling.h"
#include "tatara/runtime/prefill_profile_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tatara::runtime {

inline constexpr std::size_t kPrefillProfileEventClassCount =
    static_cast<std::size_t>(
        PrefillProfileEventClass::AttentionStreaming) +
    1U;

struct PrefillProfileStageReport {
    std::uint64_t event_count{0};
    std::uint64_t kernel_ticks{0};
    std::uint64_t gap_ticks{0};

    constexpr bool
    operator==(const PrefillProfileStageReport&) const noexcept = default;
};

struct PrefillProfileReport {
    std::array<PrefillProfileStageReport, kPrefillProfileEventClassCount>
        stages{};
    std::uint64_t event_count{0};
    std::uint64_t kernel_ticks{0};
    std::uint64_t gap_ticks{0};

    constexpr bool operator==(const PrefillProfileReport&) const noexcept =
        default;
};

struct PrefillProfilePairReport {
    std::uint64_t pair_count{0};
    std::uint64_t first_kernel_ticks{0};
    std::uint64_t second_kernel_ticks{0};
    std::uint64_t overlap_ticks{0};
    std::uint64_t first_exclusive_ticks{0};
    std::uint64_t second_exclusive_ticks{0};
    std::uint64_t union_ticks{0};

    constexpr bool operator==(const PrefillProfilePairReport&) const noexcept =
        default;
};

enum class PrefillProfileReportError : std::uint8_t {
    None,
    EmptyEvents,
    EventCountMismatch,
    EventIndexMismatch,
    InvalidEventClass,
    InvalidTimingClass,
    EventClassMismatch,
    InvalidTimingRange,
    InvalidPairClass,
    PairSequenceMismatch,
    EmptyPairs,
    ArithmeticOverflow,
};

[[nodiscard]] PrefillProfileReportError aggregate_prefill_profile_report(
    std::span<const PrefillProfileEvent> events,
    std::span<const backend::metal::CounterEventTiming> timings,
    PrefillProfileReport& output) noexcept;

[[nodiscard]] PrefillProfileReportError aggregate_prefill_profile_pair_report(
    std::span<const PrefillProfileEvent> events,
    std::span<const backend::metal::CounterEventTiming> timings,
    PrefillProfileEventClass first_class,
    PrefillProfileEventClass second_class,
    PrefillProfilePairReport& output) noexcept;

} // namespace tatara::runtime
