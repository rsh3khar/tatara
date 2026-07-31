#include "tatara/runtime/prefill_profiler.h"

#include <limits>

namespace tatara::runtime {

PrefillProfiler::PrefillProfiler(std::span<const PrefillProfileEvent> events,
                                 std::size_t sample_capacity) noexcept
    : PrefillProfiler(events, sample_capacity,
                      backend::metal::CounterSamplingMode::DispatchBoundary) {}

PrefillProfiler::PrefillProfiler(
    std::span<const PrefillProfileEvent> events, std::size_t sample_capacity,
    backend::metal::CounterSamplingMode sampling_mode) noexcept
    : events_(events), sample_capacity_(sample_capacity), sampling_mode_(sampling_mode) {
    if (sampling_mode != backend::metal::CounterSamplingMode::DispatchBoundary &&
        sampling_mode != backend::metal::CounterSamplingMode::StageBoundaryEncoderSplit) {
        fail(PrefillProfilerError::InvalidSamplingMode);
        return;
    }
    if (events.empty()) {
        fail(PrefillProfilerError::EmptyEventPlan);
        return;
    }
    if (events.size() > std::numeric_limits<std::size_t>::max() / 2U) {
        fail(PrefillProfilerError::SampleCountOverflow);
        return;
    }
    required_sample_count_ = events.size() * 2U;
    const std::size_t minimum_capacity =
        sampling_mode == backend::metal::CounterSamplingMode::DispatchBoundary
            ? required_sample_count_
            : 2U;
    if (sample_capacity < minimum_capacity) {
        fail(PrefillProfilerError::SampleCapacityInsufficient);
        return;
    }
    state_ = PrefillProfilerState::Ready;
}

PrefillProfilerStatus PrefillProfiler::status() const noexcept {
    return {
        .state = state_,
        .error = error_,
        .sampling_mode = sampling_mode_,
        .counter_error = counter_error_,
        .stage_error = stage_error_,
        .event_cursor = cursor_,
        .event_count = events_.size(),
        .required_sample_count = required_sample_count_,
        .sample_capacity = sample_capacity_,
        .mismatch_index = mismatch_index_,
        .expected_event = expected_event_,
        .actual_event = actual_event_,
        .sample_window_active = sample_window_active_,
        .sample_window_event_begin = sample_window_event_begin_,
        .sample_window_event_count = sample_window_event_count_,
    };
}

PrefillProfilerStatus PrefillProfiler::finalize() noexcept {
    if (state_ == PrefillProfilerState::Disabled) {
        PrefillProfilerStatus result = status();
        result.error = PrefillProfilerError::Disabled;
        return result;
    }
    if (state_ == PrefillProfilerState::Failed) {
        return status();
    }
    if (state_ == PrefillProfilerState::Finalized) {
        PrefillProfilerStatus result = status();
        result.error = PrefillProfilerError::AlreadyFinalized;
        return result;
    }
    if (state_ == PrefillProfilerState::DispatchPending || sample_window_active_ ||
        cursor_ != events_.size()) {
        PrefillProfilerStatus result = status();
        result.error = PrefillProfilerError::Incomplete;
        return result;
    }
    state_ = PrefillProfilerState::Finalized;
    return status();
}

PrefillProfilerError
PrefillProfiler::validate_sample_capacity(std::size_t sample_capacity) noexcept {
    if (state_ == PrefillProfilerState::Disabled) {
        return fail(PrefillProfilerError::Disabled);
    }
    if (state_ == PrefillProfilerState::Failed) {
        return error_;
    }
    if (state_ == PrefillProfilerState::Finalized) {
        return fail(PrefillProfilerError::AlreadyFinalized);
    }
    if (sample_capacity != sample_capacity_) {
        return fail(PrefillProfilerError::SampleCapacityMismatch);
    }
    return PrefillProfilerError::None;
}

PrefillProfilerError PrefillProfiler::validate_sampling_mode(
    backend::metal::CounterSamplingMode sampling_mode) noexcept {
    if (state_ == PrefillProfilerState::Disabled) {
        return fail(PrefillProfilerError::Disabled);
    }
    if (state_ == PrefillProfilerState::Failed) {
        return error_;
    }
    if (state_ == PrefillProfilerState::Finalized) {
        return fail(PrefillProfilerError::AlreadyFinalized);
    }
    if (sampling_mode != sampling_mode_) {
        return fail(PrefillProfilerError::SamplingModeMismatch);
    }
    return PrefillProfilerError::None;
}

PrefillProfilerError PrefillProfiler::begin_sample_window() noexcept {
    if (state_ == PrefillProfilerState::Disabled) {
        return fail(PrefillProfilerError::Disabled);
    }
    if (state_ == PrefillProfilerState::Failed) {
        return error_;
    }
    if (state_ == PrefillProfilerState::Finalized) {
        return fail(PrefillProfilerError::AlreadyFinalized);
    }
    if (sampling_mode_ !=
        backend::metal::CounterSamplingMode::StageBoundaryEncoderSplit) {
        return fail(PrefillProfilerError::InvalidSamplingMode);
    }
    if (sample_window_active_) {
        return fail(PrefillProfilerError::SampleWindowAlreadyActive);
    }
    if (cursor_ >= events_.size()) {
        return fail(PrefillProfilerError::EventSequenceExhausted);
    }
    sample_window_active_ = true;
    sample_window_event_begin_ = cursor_;
    sample_window_event_count_ = 0;
    return PrefillProfilerError::None;
}

PrefillProfilerError
PrefillProfiler::finish_sample_window(PrefillProfileSampleWindow& window) noexcept {
    window = {};
    if (state_ == PrefillProfilerState::Disabled) {
        return fail(PrefillProfilerError::Disabled);
    }
    if (state_ == PrefillProfilerState::Failed) {
        return error_;
    }
    if (state_ == PrefillProfilerState::Finalized) {
        return fail(PrefillProfilerError::AlreadyFinalized);
    }
    if (sampling_mode_ !=
        backend::metal::CounterSamplingMode::StageBoundaryEncoderSplit) {
        return fail(PrefillProfilerError::InvalidSamplingMode);
    }
    if (!sample_window_active_) {
        return fail(PrefillProfilerError::SampleWindowNotActive);
    }
    if (state_ == PrefillProfilerState::DispatchPending) {
        return fail(PrefillProfilerError::Incomplete);
    }
    if (sample_window_event_count_ == 0) {
        return fail(PrefillProfilerError::SampleWindowEmpty);
    }
    window = {
        .event_begin = sample_window_event_begin_,
        .event_count = sample_window_event_count_,
        .sample_count = sample_window_event_count_ * 2U,
    };
    sample_window_active_ = false;
    sample_window_event_begin_ = kInvalidPrefillProfileEventIndex;
    sample_window_event_count_ = 0;
    return PrefillProfilerError::None;
}

PrefillProfilerError
PrefillProfiler::next_stage_sample_pair(
    backend::metal::CounterSamplePair& pair) noexcept {
    pair = {};
    if (state_ == PrefillProfilerState::Disabled) {
        return fail(PrefillProfilerError::Disabled);
    }
    if (state_ == PrefillProfilerState::Failed) {
        return error_;
    }
    if (state_ == PrefillProfilerState::Finalized) {
        return fail(PrefillProfilerError::AlreadyFinalized);
    }
    if (sampling_mode_ !=
        backend::metal::CounterSamplingMode::StageBoundaryEncoderSplit) {
        return fail(PrefillProfilerError::InvalidSamplingMode);
    }
    if (!sample_window_active_) {
        return fail(PrefillProfilerError::SampleWindowRequired);
    }
    if (sample_window_event_count_ >
        std::numeric_limits<std::size_t>::max() / 2U) {
        return fail(PrefillProfilerError::SampleCountOverflow);
    }
    pair.start = sample_window_event_count_ * 2U;
    pair.end = pair.start + 1U;
    if (pair.end >= sample_capacity_) {
        pair = {};
        return fail(PrefillProfilerError::SampleWindowCapacityExceeded);
    }
    return PrefillProfilerError::None;
}

PrefillProfilerError
PrefillProfiler::begin_dispatch(const PrefillProfileEvent& event,
                                PrefillProfileDispatchTicket& ticket) noexcept {
    ticket = {};
    if (state_ == PrefillProfilerState::Disabled) {
        return fail(PrefillProfilerError::Disabled);
    }
    if (state_ == PrefillProfilerState::Failed) {
        return error_;
    }
    if (state_ == PrefillProfilerState::Finalized) {
        return fail(PrefillProfilerError::AlreadyFinalized);
    }
    if (state_ == PrefillProfilerState::DispatchPending) {
        return fail(PrefillProfilerError::DispatchAlreadyPending);
    }
    if (cursor_ >= events_.size()) {
        return fail(PrefillProfilerError::EventSequenceExhausted);
    }
    if (sampling_mode_ ==
            backend::metal::CounterSamplingMode::StageBoundaryEncoderSplit &&
        !sample_window_active_) {
        return fail(PrefillProfilerError::SampleWindowRequired);
    }
    if (events_[cursor_] != event) {
        mismatch_index_ = cursor_;
        expected_event_ = events_[cursor_];
        actual_event_ = event;
        return fail(PrefillProfilerError::EventMismatch);
    }
    ticket.event_index = cursor_;
    if (sampling_mode_ ==
        backend::metal::CounterSamplingMode::StageBoundaryEncoderSplit) {
        if (next_stage_sample_pair(ticket.samples) != PrefillProfilerError::None) {
            return error_;
        }
    } else {
        ticket.samples = {
            .start = cursor_ * 2U,
            .end = cursor_ * 2U + 1U,
        };
    }
    state_ = PrefillProfilerState::DispatchPending;
    return PrefillProfilerError::None;
}

PrefillProfilerError
PrefillProfiler::complete_dispatch(PrefillProfileDispatchTicket ticket) noexcept {
    if (state_ == PrefillProfilerState::Failed) {
        return error_;
    }
    if (state_ != PrefillProfilerState::DispatchPending) {
        return fail(PrefillProfilerError::DispatchNotPending);
    }
    const PrefillProfileDispatchTicket expected{
        .event_index = cursor_,
        .samples =
            {
                .start =
                    sampling_mode_ ==
                            backend::metal::CounterSamplingMode::
                                StageBoundaryEncoderSplit
                        ? sample_window_event_count_ * 2U
                        : cursor_ * 2U,
                .end =
                    sampling_mode_ ==
                            backend::metal::CounterSamplingMode::
                                StageBoundaryEncoderSplit
                        ? sample_window_event_count_ * 2U + 1U
                        : cursor_ * 2U + 1U,
            },
    };
    if (ticket.event_index != expected.event_index ||
        ticket.samples.start != expected.samples.start ||
        ticket.samples.end != expected.samples.end) {
        return fail(PrefillProfilerError::DispatchTicketMismatch);
    }
    ++cursor_;
    if (sampling_mode_ ==
        backend::metal::CounterSamplingMode::StageBoundaryEncoderSplit) {
        ++sample_window_event_count_;
    }
    state_ = PrefillProfilerState::Ready;
    return PrefillProfilerError::None;
}

PrefillProfilerError
PrefillProfiler::fail_stage(backend::metal::CounterStageSampleError error) noexcept {
    if (state_ == PrefillProfilerState::Failed) {
        return error_;
    }
    stage_error_ = error;
    return fail(PrefillProfilerError::StageEncoderSplitFailed);
}

PrefillProfilerError
PrefillProfiler::fail_counter(backend::metal::CounterSampleError error) noexcept {
    if (state_ == PrefillProfilerState::Failed) {
        return error_;
    }
    counter_error_ = error;
    return fail(PrefillProfilerError::CounterSamplingFailed);
}

PrefillProfilerError PrefillProfiler::fail(PrefillProfilerError error) noexcept {
    error_ = error;
    state_ = PrefillProfilerState::Failed;
    return error;
}

} // namespace tatara::runtime
