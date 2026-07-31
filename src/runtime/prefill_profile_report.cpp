#include "tatara/runtime/prefill_profile_report.h"

#include <limits>

namespace tatara::runtime {
namespace {

bool checked_add(std::uint64_t left,
                 std::uint64_t right,
                 std::uint64_t& output) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    output = left + right;
    return true;
}

bool same_event_extent(const PrefillProfileEvent& first,
                       const PrefillProfileEvent& second) noexcept {
    return first.layer_index == second.layer_index &&
           first.chunk_ordinal == second.chunk_ordinal &&
           first.chunk_offset == second.chunk_offset &&
           first.chunk_rows == second.chunk_rows &&
           first.operation_row_begin == second.operation_row_begin &&
           first.operation_row_count == second.operation_row_count;
}

PrefillProfileReportError validate_events_and_timings(
    std::span<const PrefillProfileEvent> events,
    std::span<const backend::metal::CounterEventTiming> timings) noexcept {
    if (events.size() != timings.size()) {
        return PrefillProfileReportError::EventCountMismatch;
    }
    if (events.empty()) {
        return PrefillProfileReportError::EmptyEvents;
    }
    for (std::size_t index = 0; index < events.size(); ++index) {
        const std::uint32_t event_class =
            static_cast<std::uint32_t>(events[index].event_class);
        if (event_class >= kPrefillProfileEventClassCount) {
            return PrefillProfileReportError::InvalidEventClass;
        }
        const backend::metal::CounterEventTiming& timing = timings[index];
        if (timing.event_index != index) {
            return PrefillProfileReportError::EventIndexMismatch;
        }
        if (timing.class_id >= kPrefillProfileEventClassCount) {
            return PrefillProfileReportError::InvalidTimingClass;
        }
        if (timing.class_id != event_class) {
            return PrefillProfileReportError::EventClassMismatch;
        }
        if (timing.end_timestamp < timing.start_timestamp ||
            timing.kernel_ticks !=
                timing.end_timestamp - timing.start_timestamp) {
            return PrefillProfileReportError::InvalidTimingRange;
        }
    }
    return PrefillProfileReportError::None;
}

} // namespace

PrefillProfileReportError aggregate_prefill_profile_report(
    std::span<const PrefillProfileEvent> events,
    std::span<const backend::metal::CounterEventTiming> timings,
    PrefillProfileReport& output) noexcept {
    const PrefillProfileReportError validation =
        validate_events_and_timings(events, timings);
    if (validation != PrefillProfileReportError::None) {
        return validation;
    }

    PrefillProfileReport candidate;
    for (std::size_t index = 0; index < events.size(); ++index) {
        const std::uint32_t event_class =
            static_cast<std::uint32_t>(events[index].event_class);
        const backend::metal::CounterEventTiming& timing = timings[index];

        PrefillProfileStageReport& stage = candidate.stages[event_class];
        std::uint64_t next_stage_count = 0;
        std::uint64_t next_stage_kernel = 0;
        std::uint64_t next_stage_gap = 0;
        std::uint64_t next_total_count = 0;
        std::uint64_t next_total_kernel = 0;
        std::uint64_t next_total_gap = 0;
        if (!checked_add(stage.event_count, 1, next_stage_count) ||
            !checked_add(stage.kernel_ticks, timing.kernel_ticks,
                         next_stage_kernel) ||
            !checked_add(stage.gap_ticks, timing.gap_to_next_ticks,
                         next_stage_gap) ||
            !checked_add(candidate.event_count, 1, next_total_count) ||
            !checked_add(candidate.kernel_ticks, timing.kernel_ticks,
                         next_total_kernel) ||
            !checked_add(candidate.gap_ticks, timing.gap_to_next_ticks,
                         next_total_gap)) {
            return PrefillProfileReportError::ArithmeticOverflow;
        }
        stage.event_count = next_stage_count;
        stage.kernel_ticks = next_stage_kernel;
        stage.gap_ticks = next_stage_gap;
        candidate.event_count = next_total_count;
        candidate.kernel_ticks = next_total_kernel;
        candidate.gap_ticks = next_total_gap;
    }

    output = candidate;
    return PrefillProfileReportError::None;
}

PrefillProfileReportError aggregate_prefill_profile_pair_report(
    std::span<const PrefillProfileEvent> events,
    std::span<const backend::metal::CounterEventTiming> timings,
    PrefillProfileEventClass first_class,
    PrefillProfileEventClass second_class,
    PrefillProfilePairReport& output) noexcept {
    const PrefillProfileReportError validation =
        validate_events_and_timings(events, timings);
    if (validation != PrefillProfileReportError::None) {
        return validation;
    }
    const std::uint32_t first_id =
        static_cast<std::uint32_t>(first_class);
    const std::uint32_t second_id =
        static_cast<std::uint32_t>(second_class);
    if (first_id >= kPrefillProfileEventClassCount ||
        second_id >= kPrefillProfileEventClassCount ||
        first_class == second_class) {
        return PrefillProfileReportError::InvalidPairClass;
    }

    std::uint64_t second_count = 0;
    for (std::size_t index = 0; index < events.size(); ++index) {
        if (events[index].event_class == second_class) {
            ++second_count;
            if (index == 0 ||
                events[index - 1].event_class != first_class) {
                return PrefillProfileReportError::PairSequenceMismatch;
            }
        }
    }

    PrefillProfilePairReport candidate;
    for (std::size_t index = 0; index < events.size(); ++index) {
        if (events[index].event_class != first_class) {
            continue;
        }
        if (index + 1 >= events.size() ||
            events[index + 1].event_class != second_class ||
            !same_event_extent(events[index], events[index + 1])) {
            return PrefillProfileReportError::PairSequenceMismatch;
        }

        const backend::metal::CounterEventTiming& first = timings[index];
        const backend::metal::CounterEventTiming& second = timings[index + 1];
        const std::uint64_t overlap_begin =
            first.start_timestamp > second.start_timestamp
                ? first.start_timestamp
                : second.start_timestamp;
        const std::uint64_t overlap_end =
            first.end_timestamp < second.end_timestamp
                ? first.end_timestamp
                : second.end_timestamp;
        const std::uint64_t overlap =
            overlap_end > overlap_begin ? overlap_end - overlap_begin : 0;
        const std::uint64_t first_exclusive =
            first.kernel_ticks - overlap;
        const std::uint64_t second_exclusive =
            second.kernel_ticks - overlap;
        std::uint64_t pair_union = 0;
        if (!checked_add(first.kernel_ticks, second.kernel_ticks,
                         pair_union)) {
            return PrefillProfileReportError::ArithmeticOverflow;
        }
        pair_union -= overlap;

        std::uint64_t next_count = 0;
        std::uint64_t next_first = 0;
        std::uint64_t next_second = 0;
        std::uint64_t next_overlap = 0;
        std::uint64_t next_first_exclusive = 0;
        std::uint64_t next_second_exclusive = 0;
        std::uint64_t next_union = 0;
        if (!checked_add(candidate.pair_count, 1, next_count) ||
            !checked_add(candidate.first_kernel_ticks, first.kernel_ticks,
                         next_first) ||
            !checked_add(candidate.second_kernel_ticks, second.kernel_ticks,
                         next_second) ||
            !checked_add(candidate.overlap_ticks, overlap, next_overlap) ||
            !checked_add(candidate.first_exclusive_ticks, first_exclusive,
                         next_first_exclusive) ||
            !checked_add(candidate.second_exclusive_ticks, second_exclusive,
                         next_second_exclusive) ||
            !checked_add(candidate.union_ticks, pair_union, next_union)) {
            return PrefillProfileReportError::ArithmeticOverflow;
        }
        candidate = {
            .pair_count = next_count,
            .first_kernel_ticks = next_first,
            .second_kernel_ticks = next_second,
            .overlap_ticks = next_overlap,
            .first_exclusive_ticks = next_first_exclusive,
            .second_exclusive_ticks = next_second_exclusive,
            .union_ticks = next_union,
        };
    }
    if (candidate.pair_count == 0) {
        return PrefillProfileReportError::EmptyPairs;
    }
    if (candidate.pair_count != second_count) {
        return PrefillProfileReportError::PairSequenceMismatch;
    }
    output = candidate;
    return PrefillProfileReportError::None;
}

} // namespace tatara::runtime
