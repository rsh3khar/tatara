#include "tatara/runtime/prefill_profile_report.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

namespace {

using tatara::backend::metal::CounterEventTiming;
using tatara::runtime::aggregate_prefill_profile_report;
using tatara::runtime::aggregate_prefill_profile_pair_report;
using tatara::runtime::kPrefillProfileEventClassCount;
using tatara::runtime::PrefillProfileEvent;
using tatara::runtime::PrefillProfileEventClass;
using tatara::runtime::PrefillProfilePairReport;
using tatara::runtime::PrefillProfileReport;
using tatara::runtime::PrefillProfileReportError;
using tatara::runtime::PrefillProfileStageReport;

std::uint64_t g_allocation_count = 0;

static_assert(kPrefillProfileEventClassCount == 35);
static_assert(std::is_trivially_copyable_v<PrefillProfileReport>);
static_assert(noexcept(aggregate_prefill_profile_report(
    std::declval<std::span<const PrefillProfileEvent>>(),
    std::declval<std::span<const CounterEventTiming>>(),
    std::declval<PrefillProfileReport&>())));
static_assert(noexcept(aggregate_prefill_profile_pair_report(
    std::declval<std::span<const PrefillProfileEvent>>(),
    std::declval<std::span<const CounterEventTiming>>(),
    std::declval<PrefillProfileEventClass>(),
    std::declval<PrefillProfileEventClass>(),
    std::declval<PrefillProfilePairReport&>())));

constexpr std::uint32_t class_id(
    PrefillProfileEventClass event_class) noexcept {
    return static_cast<std::uint32_t>(event_class);
}

constexpr PrefillProfileEvent
make_event(PrefillProfileEventClass event_class) noexcept {
    return {.event_class = event_class};
}

constexpr CounterEventTiming
make_timing(PrefillProfileEventClass event_class,
            std::size_t event_index,
            std::uint64_t kernel_ticks,
            std::uint64_t gap_ticks) noexcept {
    return {
        .class_id = class_id(event_class),
        .event_index = event_index,
        .start_timestamp = 0,
        .end_timestamp = kernel_ticks,
        .kernel_ticks = kernel_ticks,
        .gap_to_next_ticks = gap_ticks,
    };
}

constexpr CounterEventTiming make_interval_timing(
    PrefillProfileEventClass event_class,
    std::size_t event_index,
    std::uint64_t start,
    std::uint64_t end) noexcept {
    return {
        .class_id = class_id(event_class),
        .event_index = event_index,
        .start_timestamp = start,
        .end_timestamp = end,
        .kernel_ticks = end - start,
        .gap_to_next_ticks = 0,
    };
}

PrefillProfileReport make_canary_report() {
    PrefillProfileReport report{
        .event_count = 0x1111111111111111ULL,
        .kernel_ticks = 0x2222222222222222ULL,
        .gap_ticks = 0x3333333333333333ULL,
    };
    for (std::size_t index = 0; index < report.stages.size(); ++index) {
        report.stages[index] = {
            .event_count = 0x4444444444444444ULL + index,
            .kernel_ticks = 0x5555555555555555ULL + index,
            .gap_ticks = 0x6666666666666666ULL + index,
        };
    }
    return report;
}

const PrefillProfileStageReport&
stage(const PrefillProfileReport& report,
      PrefillProfileEventClass event_class) {
    return report.stages[static_cast<std::size_t>(event_class)];
}

constexpr std::array<PrefillProfileEvent, 8> kEvents{
    make_event(PrefillProfileEventClass::GdnProjection),
    make_event(PrefillProfileEventClass::GdnRecurrenceSerialStep),
    make_event(PrefillProfileEventClass::GdnRecurrenceSerialStep),
    make_event(PrefillProfileEventClass::MoeRouter),
    make_event(PrefillProfileEventClass::MoeExpertDown),
    make_event(PrefillProfileEventClass::MoeExpertDown),
    make_event(PrefillProfileEventClass::MoeResidualOutput),
    make_event(PrefillProfileEventClass::GdnProjection),
};

constexpr std::array<CounterEventTiming, 8> kTimings{
    make_timing(PrefillProfileEventClass::GdnProjection, 0, 10, 1),
    make_timing(PrefillProfileEventClass::GdnRecurrenceSerialStep, 1, 20, 2),
    make_timing(PrefillProfileEventClass::GdnRecurrenceSerialStep, 2, 30, 3),
    make_timing(PrefillProfileEventClass::MoeRouter, 3, 40, 4),
    make_timing(PrefillProfileEventClass::MoeExpertDown, 4, 50, 5),
    make_timing(PrefillProfileEventClass::MoeExpertDown, 5, 60, 6),
    make_timing(PrefillProfileEventClass::MoeResidualOutput, 6, 70, 7),
    make_timing(PrefillProfileEventClass::GdnProjection, 7, 80, 0),
};

int test_repeated_stage_aggregation() {
    PrefillProfileReport report = make_canary_report();
    const std::uint64_t allocations_before = g_allocation_count;
    const PrefillProfileReportError error =
        aggregate_prefill_profile_report(kEvents, kTimings, report);
    if (error != PrefillProfileReportError::None ||
        g_allocation_count != allocations_before || report.event_count != 8 ||
        report.kernel_ticks != 360 || report.gap_ticks != 28) {
        return 1;
    }

    const PrefillProfileStageReport& projection =
        stage(report, PrefillProfileEventClass::GdnProjection);
    const PrefillProfileStageReport& recurrence =
        stage(report, PrefillProfileEventClass::GdnRecurrenceSerialStep);
    const PrefillProfileStageReport& router =
        stage(report, PrefillProfileEventClass::MoeRouter);
    const PrefillProfileStageReport& down =
        stage(report, PrefillProfileEventClass::MoeExpertDown);
    const PrefillProfileStageReport& residual =
        stage(report, PrefillProfileEventClass::MoeResidualOutput);
    if (projection != PrefillProfileStageReport{2, 90, 1} ||
        recurrence != PrefillProfileStageReport{2, 50, 5} ||
        router != PrefillProfileStageReport{1, 40, 4} ||
        down != PrefillProfileStageReport{2, 110, 11} ||
        residual != PrefillProfileStageReport{1, 70, 7}) {
        return 2;
    }

    std::uint64_t populated_stages = 0;
    std::uint64_t stage_events = 0;
    std::uint64_t stage_kernel = 0;
    std::uint64_t stage_gap = 0;
    for (const PrefillProfileStageReport& value : report.stages) {
        populated_stages += value.event_count != 0 ? 1U : 0U;
        stage_events += value.event_count;
        stage_kernel += value.kernel_ticks;
        stage_gap += value.gap_ticks;
    }
    if (populated_stages != 5 || stage_events != report.event_count ||
        stage_kernel != report.kernel_ticks || stage_gap != report.gap_ticks) {
        return 3;
    }
    return 0;
}

int test_identity_failures_preserve_output() {
    const PrefillProfileReport canary = make_canary_report();
    PrefillProfileReport output = canary;
    if (aggregate_prefill_profile_report(
            kEvents,
            std::span<const CounterEventTiming>{kTimings}.first(7), output) !=
            PrefillProfileReportError::EventCountMismatch ||
        output != canary) {
        return 1;
    }

    auto reordered = kTimings;
    std::swap(reordered[0], reordered[3]);
    if (aggregate_prefill_profile_report(kEvents, reordered, output) !=
            PrefillProfileReportError::EventIndexMismatch ||
        output != canary) {
        return 2;
    }

    auto mismatched = kTimings;
    mismatched[0].class_id =
        class_id(PrefillProfileEventClass::GdnConvolution);
    if (aggregate_prefill_profile_report(kEvents, mismatched, output) !=
            PrefillProfileReportError::EventClassMismatch ||
        output != canary) {
        return 3;
    }

    auto invalid_event = kEvents;
    invalid_event[0].event_class =
        static_cast<PrefillProfileEventClass>(255);
    if (aggregate_prefill_profile_report(invalid_event, kTimings, output) !=
            PrefillProfileReportError::InvalidEventClass ||
        output != canary) {
        return 4;
    }

    auto invalid_timing = kTimings;
    invalid_timing[0].class_id =
        static_cast<std::uint32_t>(kPrefillProfileEventClassCount);
    if (aggregate_prefill_profile_report(kEvents, invalid_timing, output) !=
            PrefillProfileReportError::InvalidTimingClass ||
        output != canary) {
        return 5;
    }

    if (aggregate_prefill_profile_report({}, {}, output) !=
            PrefillProfileReportError::EmptyEvents ||
        output != canary) {
        return 6;
    }
    return 0;
}

int test_overflow_preserves_output() {
    constexpr std::array<PrefillProfileEvent, 2> kRepeated{
        make_event(PrefillProfileEventClass::MoeExpertDown),
        make_event(PrefillProfileEventClass::MoeExpertDown),
    };
    constexpr std::array<CounterEventTiming, 2> kStageOverflow{
        make_timing(PrefillProfileEventClass::MoeExpertDown, 0,
                    std::numeric_limits<std::uint64_t>::max(), 0),
        make_timing(PrefillProfileEventClass::MoeExpertDown, 1, 1, 0),
    };
    const PrefillProfileReport canary = make_canary_report();
    PrefillProfileReport output = canary;
    if (aggregate_prefill_profile_report(kRepeated, kStageOverflow, output) !=
            PrefillProfileReportError::ArithmeticOverflow ||
        output != canary) {
        return 1;
    }

    constexpr std::array<PrefillProfileEvent, 2> kDifferent{
        make_event(PrefillProfileEventClass::GdnProjection),
        make_event(PrefillProfileEventClass::MoeExpertDown),
    };
    constexpr std::array<CounterEventTiming, 2> kGrandOverflow{
        make_timing(PrefillProfileEventClass::GdnProjection, 0, 0,
                    std::numeric_limits<std::uint64_t>::max()),
        make_timing(PrefillProfileEventClass::MoeExpertDown, 1, 0, 1),
    };
    if (aggregate_prefill_profile_report(kDifferent, kGrandOverflow, output) !=
            PrefillProfileReportError::ArithmeticOverflow ||
        output != canary) {
        return 2;
    }
    return 0;
}

int test_split_moe_stages_and_pair_overlap() {
    constexpr std::array<PrefillProfileEvent, 4> kSplitEvents{
        make_event(PrefillProfileEventClass::MoeNativeRoutedUpGate),
        make_event(PrefillProfileEventClass::MoeSharedExpertUpGate),
        make_event(PrefillProfileEventClass::MoeNativeRoutedDown),
        make_event(PrefillProfileEventClass::MoeSharedExpertDown),
    };
    constexpr std::array<CounterEventTiming, 4> kSplitTimings{
        make_interval_timing(
            PrefillProfileEventClass::MoeNativeRoutedUpGate, 0, 100, 160),
        make_interval_timing(
            PrefillProfileEventClass::MoeSharedExpertUpGate, 1, 140, 180),
        make_interval_timing(
            PrefillProfileEventClass::MoeNativeRoutedDown, 2, 200, 230),
        make_interval_timing(
            PrefillProfileEventClass::MoeSharedExpertDown, 3, 240, 260),
    };

    PrefillProfileReport report;
    if (aggregate_prefill_profile_report(
            kSplitEvents, kSplitTimings, report) !=
            PrefillProfileReportError::None ||
        report.event_count != 4 || report.kernel_ticks != 150 ||
        stage(report, PrefillProfileEventClass::MoeNativeRoutedUpGate) !=
            PrefillProfileStageReport{1, 60, 0} ||
        stage(report, PrefillProfileEventClass::MoeSharedExpertUpGate) !=
            PrefillProfileStageReport{1, 40, 0} ||
        stage(report, PrefillProfileEventClass::MoeNativeRoutedDown) !=
            PrefillProfileStageReport{1, 30, 0} ||
        stage(report, PrefillProfileEventClass::MoeSharedExpertDown) !=
            PrefillProfileStageReport{1, 20, 0}) {
        return 1;
    }

    PrefillProfilePairReport up;
    if (aggregate_prefill_profile_pair_report(
            kSplitEvents, kSplitTimings,
            PrefillProfileEventClass::MoeNativeRoutedUpGate,
            PrefillProfileEventClass::MoeSharedExpertUpGate, up) !=
            PrefillProfileReportError::None ||
        up != PrefillProfilePairReport{1, 60, 40, 20, 40, 20, 80}) {
        return 2;
    }
    PrefillProfilePairReport down;
    if (aggregate_prefill_profile_pair_report(
            kSplitEvents, kSplitTimings,
            PrefillProfileEventClass::MoeNativeRoutedDown,
            PrefillProfileEventClass::MoeSharedExpertDown, down) !=
            PrefillProfileReportError::None ||
        down != PrefillProfilePairReport{1, 30, 20, 0, 30, 20, 50}) {
        return 3;
    }

    auto malformed_events = kSplitEvents;
    malformed_events[1].chunk_ordinal = 1;
    const PrefillProfilePairReport canary{9, 8, 7, 6, 5, 4, 3};
    up = canary;
    if (aggregate_prefill_profile_pair_report(
            malformed_events, kSplitTimings,
            PrefillProfileEventClass::MoeNativeRoutedUpGate,
            PrefillProfileEventClass::MoeSharedExpertUpGate, up) !=
            PrefillProfileReportError::PairSequenceMismatch ||
        up != canary) {
        return 4;
    }
    if (aggregate_prefill_profile_pair_report(
            kSplitEvents, kSplitTimings,
            PrefillProfileEventClass::MoeRouter,
            PrefillProfileEventClass::MoeExpertUnion, up) !=
            PrefillProfileReportError::EmptyPairs ||
        up != canary) {
        return 5;
    }
    auto invalid_timing = kSplitTimings;
    invalid_timing[0].kernel_ticks = 59;
    if (aggregate_prefill_profile_pair_report(
            kSplitEvents, invalid_timing,
            PrefillProfileEventClass::MoeNativeRoutedUpGate,
            PrefillProfileEventClass::MoeSharedExpertUpGate, up) !=
            PrefillProfileReportError::InvalidTimingRange ||
        up != canary) {
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
    if (const int result = test_repeated_stage_aggregation()) {
        return 10 + result;
    }
    if (const int result = test_identity_failures_preserve_output()) {
        return 20 + result;
    }
    if (const int result = test_overflow_preserves_output()) {
        return 30 + result;
    }
    if (const int result = test_split_moe_stages_and_pair_overlap()) {
        return 40 + result;
    }
    return 0;
}
