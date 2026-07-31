#include "tatara/backend/metal/counter_sampling.h"

#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/resources.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <span>
#include <type_traits>
#include <utility>

namespace {

using tatara::backend::metal::counter_event_sample_pair;
using tatara::backend::metal::CounterEvent;
using tatara::backend::metal::CounterEventPlan;
using tatara::backend::metal::CounterEventTiming;
using tatara::backend::metal::CounterPlanError;
using tatara::backend::metal::CounterResolveError;
using tatara::backend::metal::CounterSampleBufferCreateResult;
using tatara::backend::metal::CounterSampleBufferCreateError;
using tatara::backend::metal::CounterSampleError;
using tatara::backend::metal::CounterSamplePair;
using tatara::backend::metal::CounterSamplingMode;
using tatara::backend::metal::CounterStageSampleError;
using tatara::backend::metal::begin_stage_sampled_compute_pass;
using tatara::backend::metal::create_timestamp_counter_sample_buffer;
using tatara::backend::metal::create_stage_timestamp_counter_sample_buffer;
using tatara::backend::metal::kInvalidCounterTimestamp;
using tatara::backend::metal::MetalCommandBuffer;
using tatara::backend::metal::MetalComputePass;
using tatara::backend::metal::MetalCounterSampleBuffer;
using tatara::backend::metal::MetalDevice;
using tatara::backend::metal::plan_counter_events;
using tatara::backend::metal::resolve_counter_event_timings;
using tatara::backend::metal::resolve_counter_samples;
using tatara::backend::metal::sample_counter;
using tatara::backend::metal::validate_counter_sample_index;
using tatara::backend::metal::validate_counter_sample_pair;
using tatara::backend::metal::validate_dispatch_counter_sampling_mode;

std::atomic<std::size_t> allocation_count{0};
int failures = 0;

void check(bool condition, const char* message) noexcept {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

bool timing_is_sentinel(const CounterEventTiming& timing) noexcept {
    return timing.class_id == 99 && timing.event_index == 99 && timing.start_timestamp == 99 &&
           timing.end_timestamp == 99 && timing.kernel_ticks == 99 &&
           timing.gap_to_next_ticks == 99;
}

} // namespace

void* operator new(std::size_t size) {
    allocation_count.fetch_add(1, std::memory_order_relaxed);
    if (void* pointer = std::malloc(size)) {
        return pointer;
    }
    std::abort();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* pointer) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

int main() {
    const CounterSampleBufferCreateResult default_create;
    check(default_create.error == CounterSampleBufferCreateError::Uninitialized &&
          !default_create,
          "default counter-sample result must not report success");

    static_assert(!std::is_copy_constructible_v<MetalCounterSampleBuffer>);
    static_assert(!std::is_copy_assignable_v<MetalCounterSampleBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<MetalCounterSampleBuffer>);
    static_assert(std::is_nothrow_move_assignable_v<MetalCounterSampleBuffer>);

    MetalCounterSampleBuffer empty;
    check(!empty && empty.sample_capacity() == 0, "sample buffer begins empty");
    MetalCounterSampleBuffer moved = std::move(empty);
    check(!empty && !moved && moved.sample_capacity() == 0,
          "empty sample-buffer ownership moves without fabricating a handle");
    check(moved.sampling_mode() == CounterSamplingMode::DispatchBoundary,
          "empty sample buffers preserve the dispatch-boundary control mode");

    MetalDevice device;
    const auto zero_create = create_timestamp_counter_sample_buffer(device, 0);
    check(zero_create.error == CounterSampleBufferCreateError::InvalidSampleCount &&
              !zero_create.buffer,
          "zero sample capacity is rejected before device access");
    const auto invalid_device = create_timestamp_counter_sample_buffer(device, 2);
    check(invalid_device.error == CounterSampleBufferCreateError::InvalidDevice &&
              !invalid_device.buffer,
          "empty typed devices return a typed creation failure");
    MetalCommandBuffer stage_command_buffer;
    const auto oversized_stage = create_stage_timestamp_counter_sample_buffer(
        stage_command_buffer, tatara::backend::metal::kMaxStageBoundarySampleCount + 1U);
    check(oversized_stage.error == CounterSampleBufferCreateError::SampleCountOutOfRange &&
              !oversized_stage,
          "stage sample capacity is capped at the experiments-proven fixed window");
    const auto invalid_stage_command =
        create_stage_timestamp_counter_sample_buffer(
            stage_command_buffer, tatara::backend::metal::kMaxStageBoundarySampleCount);
    check(invalid_stage_command.error ==
                  CounterSampleBufferCreateError::InvalidCommandBuffer &&
              !invalid_stage_command,
          "stage capability creation requires a typed command-buffer device");

    MetalComputePass compute_pass;
    check(sample_counter(compute_pass, moved, 0) == CounterSampleError::InvalidComputePass,
          "sampling rejects an empty typed compute pass before native access");
    std::array<std::uint64_t, 2> unresolved = {91, 91};
    check(resolve_counter_samples(moved, 0, 2, unresolved) == CounterResolveError::InvalidBuffer,
          "native resolution rejects an empty RAII buffer");
    const auto invalid_stage_pass =
        begin_stage_sampled_compute_pass(std::move(stage_command_buffer), moved, {0, 1});
    check(invalid_stage_pass.error == CounterStageSampleError::InvalidCommandBuffer &&
              !invalid_stage_pass,
          "stage sampling reports an empty command buffer with a typed exit");

    const std::array<CounterEvent, 4> events = {
        CounterEvent{10},
        CounterEvent{20},
        CounterEvent{20},
        CounterEvent{30},
    };
    CounterEventPlan disabled{
        .enabled = true,
        .event_count = 9,
        .sample_count = 18,
        .identity = 9,
    };
    check(plan_counter_events(false, events, disabled) == CounterPlanError::None &&
              !disabled.enabled && disabled.event_count == 0 && disabled.sample_count == 0 &&
              disabled.identity == 0,
          "disabled measurement produces a zero-cost empty plan");

    CounterSamplePair pair{99, 99};
    check(counter_event_sample_pair(disabled, 0, pair) == CounterPlanError::MeasurementDisabled &&
              pair.start == 0 && pair.end == 0,
          "disabled plans cannot produce sample pairs");

    CounterEventPlan plan{};
    check(plan_counter_events(true, events, plan) == CounterPlanError::None && plan.enabled &&
              plan.mode == CounterSamplingMode::DispatchBoundary &&
              plan.event_count == 4 && plan.sample_count == 8 && plan.identity != 0,
          "event planning allows repeated class IDs and assigns two samples per occurrence");
    check(counter_event_sample_pair(plan, 0, pair) == CounterPlanError::None && pair.start == 0 &&
              pair.end == 1,
          "first event owns samples zero and one");
    check(counter_event_sample_pair(plan, 3, pair) == CounterPlanError::None && pair.start == 6 &&
              pair.end == 7,
          "tail event owns the final bounded pair");
    check(counter_event_sample_pair(plan, 4, pair) == CounterPlanError::EventIndexOutOfBounds &&
              pair.start == 0 && pair.end == 0,
          "event indices cannot address beyond the planned sample count");

    CounterEventPlan stage_plan{};
    check(plan_counter_events(true, CounterSamplingMode::StageBoundaryEncoderSplit,
                              events, stage_plan) == CounterPlanError::None &&
              stage_plan.mode == CounterSamplingMode::StageBoundaryEncoderSplit &&
              stage_plan.event_count == plan.event_count &&
              stage_plan.sample_count == plan.sample_count &&
              stage_plan.identity != plan.identity,
          "sampling mode is bound into counter-plan identity");
    CounterEventPlan forged_mode = stage_plan;
    forged_mode.mode = CounterSamplingMode::DispatchBoundary;
    std::array<CounterEventTiming, 4> mode_rejected_timings{};
    mode_rejected_timings.fill(CounterEventTiming{99, 99, 99, 99, 99, 99});
    const std::array<std::uint64_t, 8> mode_timestamps = {
        100, 140, 150, 210, 230, 260, 270, 300,
    };
    check(resolve_counter_event_timings(forged_mode, events, mode_timestamps,
                                        mode_rejected_timings) ==
              CounterResolveError::EventIdentityMismatch,
          "a stage plan cannot authenticate dispatch-boundary timestamps after mode forgery");

    check(validate_counter_sample_index(8, 7) == CounterSampleError::None,
          "in-encoder sampling accepts the exact final index");
    check(validate_counter_sample_index(8, 8) == CounterSampleError::SampleIndexOutOfBounds,
          "in-encoder sampling rejects an index at capacity");
    check(validate_counter_sample_pair(8, CounterSamplePair{6, 7}) == CounterSampleError::None,
          "event sample-pair validation accepts the exact final pair");
    check(validate_counter_sample_pair(8, CounterSamplePair{7, 8}) ==
              CounterSampleError::SampleIndexOutOfBounds,
          "event sample-pair validation rejects a final index at capacity");
    check(validate_counter_sample_pair(8, CounterSamplePair{2, 4}) ==
              CounterSampleError::InvalidSamplePair,
          "event sample pairs must contain adjacent indices");
    check(validate_counter_sample_pair(
              8, CounterSamplePair{std::numeric_limits<std::size_t>::max(), 0}) ==
              CounterSampleError::InvalidSamplePair,
          "event sample-pair validation rejects wrapped adjacency");
    check(validate_counter_sample_index(0, 0) == CounterSampleError::InvalidBuffer,
          "zero-capacity sample storage fails closed");
    check(validate_dispatch_counter_sampling_mode(
              CounterSamplingMode::DispatchBoundary) == CounterSampleError::None,
          "dispatch sampling accepts only its matching buffer mode");
    check(validate_dispatch_counter_sampling_mode(
              CounterSamplingMode::StageBoundaryEncoderSplit) ==
              CounterSampleError::SamplingModeMismatch,
          "dispatch sampling rejects a stage-boundary buffer mode");
    check(validate_dispatch_counter_sampling_mode(
              static_cast<CounterSamplingMode>(0xffU)) ==
              CounterSampleError::SamplingModeMismatch,
          "dispatch sampling rejects a forged buffer mode");

    CounterEventPlan rejected_plan{
        .enabled = true,
        .event_count = 1,
        .sample_count = 2,
        .identity = 1,
    };
    check(plan_counter_events(true, {}, rejected_plan) == CounterPlanError::EmptyEvents &&
              !rejected_plan.enabled,
          "an enabled empty event list is rejected and clears stale plan state");

    CounterEventPlan malformed = plan;
    malformed.sample_count = 7;
    check(counter_event_sample_pair(malformed, 0, pair) == CounterPlanError::InvalidPlan,
          "a forged event/sample cardinality fails closed");

    const std::array<std::uint64_t, 8> timestamps = {
        100, 140, 150, 210, 230, 260, 270, 300,
    };
    std::array<CounterEventTiming, 5> timings{};
    timings.fill(CounterEventTiming{99, 99, 99, 99, 99, 99});
    check(resolve_counter_event_timings(plan, events, timestamps, timings) ==
              CounterResolveError::None,
          "pure timestamp resolution accepts a coherent repeated-class event sequence");
    check(timings[0].class_id == 10 && timings[0].event_index == 0 &&
              timings[0].start_timestamp == 100 && timings[0].end_timestamp == 140 &&
              timings[0].kernel_ticks == 40 && timings[0].gap_to_next_ticks == 10,
          "first event kernel and gap ticks are exact");
    check(timings[1].class_id == 20 && timings[1].event_index == 1 &&
              timings[1].kernel_ticks == 60 && timings[1].gap_to_next_ticks == 20,
          "first occurrence of a repeated class remains independently identified");
    check(timings[2].class_id == 20 && timings[2].event_index == 2 &&
              timings[2].kernel_ticks == 30 && timings[2].gap_to_next_ticks == 10,
          "second occurrence of a repeated class retains its ordered event identity");
    check(timings[3].class_id == 30 && timings[3].event_index == 3 &&
              timings[3].kernel_ticks == 30 && timings[3].gap_to_next_ticks == 0,
          "last event has no fabricated trailing gap");
    check(timing_is_sentinel(timings[4]), "resolution respects caller output extent");

    const std::array<std::uint64_t, 8> independent_window_timestamps = {
        100, 120, 130, 150, 5, 10, 12, 18,
    };
    const std::array<std::uint8_t, 4> window_end_markers = {0, 1, 0, 1};
    std::array<CounterEventTiming, 4> window_timings{};
    window_timings.fill(CounterEventTiming{99, 99, 99, 99, 99, 99});
    check(resolve_counter_event_timings(
              stage_plan, events, independent_window_timestamps,
              window_end_markers, window_timings) == CounterResolveError::None,
          "stage windows accept independent timestamp epochs");
    check(window_timings[0].kernel_ticks == 20 &&
              window_timings[0].gap_to_next_ticks == 10 &&
              window_timings[1].kernel_ticks == 20 &&
              window_timings[1].gap_to_next_ticks == 0 &&
              window_timings[2].kernel_ticks == 5 &&
              window_timings[2].gap_to_next_ticks == 2 &&
              window_timings[3].kernel_ticks == 6 &&
              window_timings[3].gap_to_next_ticks == 0,
          "stage resolution retains intra-window gaps and zeros every window tail");
    check(resolve_counter_event_timings(
              stage_plan, events, independent_window_timestamps,
              window_timings) == CounterResolveError::InvalidWindowPlan,
          "stage resolution requires explicit window boundaries");
    check(resolve_counter_event_timings(
              plan, events, timestamps, window_end_markers,
              window_timings) == CounterResolveError::InvalidWindowPlan,
          "dispatch resolution rejects stage-window metadata");
    std::array<CounterEventTiming, 4> invalid_window_timings{};
    invalid_window_timings.fill(CounterEventTiming{99, 99, 99, 99, 99, 99});
    std::array<std::uint64_t, 8> nonmonotonic_window_timestamps =
        independent_window_timestamps;
    nonmonotonic_window_timestamps[2] = 119;
    check(resolve_counter_event_timings(
              stage_plan, events, nonmonotonic_window_timestamps,
              window_end_markers, invalid_window_timings) ==
              CounterResolveError::None,
          "stage windows accept independent dispatch overlap or reordering");
    check(invalid_window_timings[0].kernel_ticks == 20 &&
              invalid_window_timings[0].gap_to_next_ticks == 0 &&
              invalid_window_timings[1].kernel_ticks == 31,
          "stage overlap keeps exact durations and clamps the absent gap");
    invalid_window_timings.fill(
        CounterEventTiming{99, 99, 99, 99, 99, 99});
    const std::array<std::uint8_t, 4> missing_final_marker = {0, 1, 0, 0};
    check(resolve_counter_event_timings(
              stage_plan, events, independent_window_timestamps,
              missing_final_marker, invalid_window_timings) ==
              CounterResolveError::InvalidWindowPlan,
          "the final stage event must close a sample window");
    const std::array<std::uint8_t, 4> forged_window_marker = {0, 2, 0, 1};
    check(resolve_counter_event_timings(
              stage_plan, events, independent_window_timestamps,
              forged_window_marker, invalid_window_timings) ==
              CounterResolveError::InvalidWindowPlan,
          "window markers are a one-byte boolean domain");
    check(resolve_counter_event_timings(
              stage_plan, events, independent_window_timestamps,
              std::span<const std::uint8_t>{window_end_markers}.first(3),
              invalid_window_timings) ==
              CounterResolveError::InvalidWindowPlan,
          "stage resolution requires exactly one marker per event");
    for (const CounterEventTiming& timing : invalid_window_timings) {
        check(timing_is_sentinel(timing),
              "invalid window metadata leaves caller timings untouched");
    }

    std::array<CounterEventTiming, 4> rejected_timings{};
    rejected_timings.fill(CounterEventTiming{99, 99, 99, 99, 99, 99});
    const std::array<CounterEvent, 4> reordered_events = {
        CounterEvent{10},
        CounterEvent{20},
        CounterEvent{30},
        CounterEvent{20},
    };
    check(resolve_counter_event_timings(plan, reordered_events, timestamps, rejected_timings) ==
              CounterResolveError::EventIdentityMismatch,
          "resolution binds the exact ordered event-class sequence");
    check(resolve_counter_event_timings(
              plan, events, std::span<const std::uint64_t>{timestamps}.first(7),
              rejected_timings) == CounterResolveError::InsufficientTimestampData,
          "resolution rejects a missing tail sample");
    check(resolve_counter_event_timings(plan, events, timestamps,
                                        std::span<CounterEventTiming>{rejected_timings}.first(3)) ==
              CounterResolveError::OutputTooSmall,
          "resolution rejects a short timing destination");

    std::array<std::uint64_t, 8> invalid_timestamps = timestamps;
    invalid_timestamps[5] = kInvalidCounterTimestamp;
    check(resolve_counter_event_timings(plan, events, invalid_timestamps, rejected_timings) ==
              CounterResolveError::InvalidCounterValue,
          "counter error values fail the complete resolution");
    invalid_timestamps = timestamps;
    invalid_timestamps[1] = 99;
    check(resolve_counter_event_timings(plan, events, invalid_timestamps, rejected_timings) ==
              CounterResolveError::NonMonotonicTimestamp,
          "event end before start fails closed");
    invalid_timestamps = timestamps;
    invalid_timestamps[2] = 139;
    check(resolve_counter_event_timings(plan, events, invalid_timestamps, rejected_timings) ==
              CounterResolveError::NonMonotonicTimestamp,
          "next-event start before the prior end cannot become a wrapped gap");
    for (const CounterEventTiming& timing : rejected_timings) {
        check(timing_is_sentinel(timing), "failed resolution leaves caller timings untouched");
    }

    const std::size_t before = allocation_count.load(std::memory_order_relaxed);
    CounterEventPlan allocation_plan{};
    CounterSamplePair allocation_pair{};
    std::array<CounterEventTiming, 4> allocation_timings{};
    const CounterPlanError allocation_plan_status =
        plan_counter_events(true, events, allocation_plan);
    const CounterPlanError allocation_pair_status =
        counter_event_sample_pair(allocation_plan, 1, allocation_pair);
    const CounterResolveError allocation_resolve_status =
        resolve_counter_event_timings(allocation_plan, events, timestamps, allocation_timings);
    CounterEventPlan allocation_stage_plan{};
    const CounterPlanError allocation_stage_plan_status =
        plan_counter_events(true, CounterSamplingMode::StageBoundaryEncoderSplit,
                            events, allocation_stage_plan);
    const CounterSampleError allocation_mode_status =
        validate_dispatch_counter_sampling_mode(
            CounterSamplingMode::DispatchBoundary);
    const CounterResolveError allocation_window_resolve_status =
        resolve_counter_event_timings(
            allocation_stage_plan, events, independent_window_timestamps,
            window_end_markers, allocation_timings);
    const std::size_t after = allocation_count.load(std::memory_order_relaxed);
    check(allocation_plan_status == CounterPlanError::None &&
              allocation_pair_status == CounterPlanError::None &&
              allocation_resolve_status == CounterResolveError::None &&
              allocation_stage_plan_status == CounterPlanError::None &&
              allocation_mode_status == CounterSampleError::None &&
              allocation_window_resolve_status == CounterResolveError::None &&
              before == after,
          "planning, mode validation, and both pure resolvers allocate no heap memory");

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("counter_sampling_contract_test: PASS");
    return 0;
}
