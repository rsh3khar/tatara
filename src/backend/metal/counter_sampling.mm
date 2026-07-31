#include "tatara/backend/metal/counter_sampling.h"

#include "tatara/backend/metal/commands.h"
#include "backend/metal/resource_storage.h"

#include <limits>
#include <utility>

namespace tatara::backend::metal {
namespace {

std::uint64_t event_identity(std::span<const CounterEvent> events,
                             CounterSamplingMode mode) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const CounterEvent event : events) {
        for (unsigned shift = 0; shift != 32; shift += 8) {
            hash ^= static_cast<std::uint8_t>(event.class_id >> shift);
            hash *= 1099511628211ULL;
        }
    }
    hash ^= static_cast<std::uint64_t>(events.size());
    hash *= 1099511628211ULL;
    // Retain the original dispatch-boundary identity while ensuring that a
    // stage diagnostic can never authenticate dispatch-boundary timestamps.
    if (mode == CounterSamplingMode::StageBoundaryEncoderSplit) {
        hash ^= 0x53544147455f5350ULL;
        hash *= 1099511628211ULL;
    }
    return hash == 0 ? 1 : hash;
}

bool valid_plan(const CounterEventPlan& plan) noexcept {
    const bool known_mode = plan.mode == CounterSamplingMode::DispatchBoundary ||
                            plan.mode == CounterSamplingMode::StageBoundaryEncoderSplit;
    return plan.enabled && known_mode && plan.event_count != 0 &&
           plan.event_count <= std::numeric_limits<std::size_t>::max() / 2 &&
           plan.sample_count == plan.event_count * 2 && plan.identity != 0;
}

bool checked_range(std::size_t first, std::size_t count, std::size_t capacity,
                   std::size_t& end) noexcept {
    if (count > std::numeric_limits<std::size_t>::max() - first) {
        return false;
    }
    end = first + count;
    return end <= capacity;
}

} // namespace

static_assert(static_cast<std::uint64_t>(MTLCounterErrorValue) == kInvalidCounterTimestamp);

MetalCounterSampleBuffer::MetalCounterSampleBuffer() noexcept = default;

MetalCounterSampleBuffer::~MetalCounterSampleBuffer() {
    reset();
}

MetalCounterSampleBuffer::MetalCounterSampleBuffer(MetalCounterSampleBuffer&& other) noexcept
    : object_(std::exchange(other.object_, nullptr)),
      stage_descriptor_(std::exchange(other.stage_descriptor_, nullptr)),
      sample_capacity_(std::exchange(other.sample_capacity_, 0)),
      mode_(std::exchange(other.mode_, CounterSamplingMode::DispatchBoundary)) {}

MetalCounterSampleBuffer&
MetalCounterSampleBuffer::operator=(MetalCounterSampleBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        object_ = std::exchange(other.object_, nullptr);
        stage_descriptor_ = std::exchange(other.stage_descriptor_, nullptr);
        sample_capacity_ = std::exchange(other.sample_capacity_, 0);
        mode_ = std::exchange(other.mode_, CounterSamplingMode::DispatchBoundary);
    }
    return *this;
}

MetalCounterSampleBuffer::MetalCounterSampleBuffer(void* object, void* stage_descriptor,
                                                   std::size_t sample_capacity,
                                                   CounterSamplingMode mode) noexcept
    : object_(object), stage_descriptor_(stage_descriptor),
      sample_capacity_(sample_capacity), mode_(mode) {}

std::size_t MetalCounterSampleBuffer::sample_capacity() const noexcept {
    return sample_capacity_;
}

CounterSamplingMode MetalCounterSampleBuffer::sampling_mode() const noexcept {
    return mode_;
}

MetalCounterSampleBuffer::operator bool() const noexcept {
    const bool descriptor_matches =
        mode_ == CounterSamplingMode::StageBoundaryEncoderSplit
            ? stage_descriptor_ != nullptr
            : stage_descriptor_ == nullptr;
    return object_ != nullptr && sample_capacity_ != 0 && descriptor_matches;
}

void MetalCounterSampleBuffer::reset() noexcept {
    sample_capacity_ = 0;
    mode_ = CounterSamplingMode::DispatchBoundary;
    if (void* descriptor = std::exchange(stage_descriptor_, nullptr);
        descriptor != nullptr) {
        release_object(descriptor);
    }
    if (void* object = std::exchange(object_, nullptr); object != nullptr) {
        release_object(object);
    }
}

CounterPlanError plan_counter_events(bool enabled, std::span<const CounterEvent> events,
                                     CounterEventPlan& plan) noexcept {
    return plan_counter_events(enabled, CounterSamplingMode::DispatchBoundary, events, plan);
}

CounterPlanError plan_counter_events(bool enabled, CounterSamplingMode mode,
                                     std::span<const CounterEvent> events,
                                     CounterEventPlan& plan) noexcept {
    plan = {};
    if (!enabled) {
        return CounterPlanError::None;
    }
    if (mode != CounterSamplingMode::DispatchBoundary &&
        mode != CounterSamplingMode::StageBoundaryEncoderSplit) {
        return CounterPlanError::InvalidPlan;
    }
    if (events.empty()) {
        return CounterPlanError::EmptyEvents;
    }
    if (events.size() > std::numeric_limits<std::size_t>::max() / 2) {
        return CounterPlanError::SampleCountOverflow;
    }
    plan = {
        .enabled = true,
        .mode = mode,
        .event_count = events.size(),
        .sample_count = events.size() * 2,
        .identity = event_identity(events, mode),
    };
    return CounterPlanError::None;
}

CounterPlanError counter_event_sample_pair(const CounterEventPlan& plan, std::size_t event_index,
                                           CounterSamplePair& pair) noexcept {
    pair = {};
    if (!plan.enabled) {
        return CounterPlanError::MeasurementDisabled;
    }
    if (!valid_plan(plan)) {
        return CounterPlanError::InvalidPlan;
    }
    if (event_index >= plan.event_count) {
        return CounterPlanError::EventIndexOutOfBounds;
    }
    pair.start = event_index * 2;
    pair.end = pair.start + 1;
    return CounterPlanError::None;
}

CounterSampleError validate_counter_sample_index(std::size_t sample_capacity,
                                                 std::size_t sample_index) noexcept {
    if (sample_capacity == 0) {
        return CounterSampleError::InvalidBuffer;
    }
    return sample_index < sample_capacity ? CounterSampleError::None
                                          : CounterSampleError::SampleIndexOutOfBounds;
}

CounterSampleError validate_counter_sample_pair(std::size_t sample_capacity,
                                                CounterSamplePair pair) noexcept {
    if (sample_capacity == 0) {
        return CounterSampleError::InvalidBuffer;
    }
    if (pair.start == std::numeric_limits<std::size_t>::max() || pair.end != pair.start + 1) {
        return CounterSampleError::InvalidSamplePair;
    }
    if (pair.start >= sample_capacity || pair.end >= sample_capacity) {
        return CounterSampleError::SampleIndexOutOfBounds;
    }
    return CounterSampleError::None;
}

CounterSampleError
validate_dispatch_counter_sampling_mode(CounterSamplingMode mode) noexcept {
    return mode == CounterSamplingMode::DispatchBoundary
               ? CounterSampleError::None
               : CounterSampleError::SamplingModeMismatch;
}

namespace {

CounterResolveError resolve_counter_event_timings_impl(
    const CounterEventPlan& plan, std::span<const CounterEvent> events,
    std::span<const std::uint64_t> timestamps,
    std::span<const std::uint8_t> window_end_markers,
    bool windowed, std::span<CounterEventTiming> timings) noexcept {
    if (!plan.enabled) {
        return CounterResolveError::MeasurementDisabled;
    }
    if (!valid_plan(plan)) {
        return CounterResolveError::InvalidPlan;
    }
    if (events.size() != plan.event_count ||
        event_identity(events, plan.mode) != plan.identity) {
        return CounterResolveError::EventIdentityMismatch;
    }
    const bool stage_plan =
        plan.mode == CounterSamplingMode::StageBoundaryEncoderSplit;
    if (windowed != stage_plan) {
        return CounterResolveError::InvalidWindowPlan;
    }
    if (windowed) {
        if (window_end_markers.size() != plan.event_count ||
            window_end_markers.back() != 1U) {
            return CounterResolveError::InvalidWindowPlan;
        }
        for (const std::uint8_t marker : window_end_markers) {
            if (marker > 1U) {
                return CounterResolveError::InvalidWindowPlan;
            }
        }
    }
    if (timestamps.size() < plan.sample_count) {
        return CounterResolveError::InsufficientTimestampData;
    }
    if (timings.size() < plan.event_count) {
        return CounterResolveError::OutputTooSmall;
    }
    for (std::size_t event = 0; event < plan.event_count; ++event) {
        const std::uint64_t start = timestamps[event * 2];
        const std::uint64_t end = timestamps[event * 2 + 1];
        if (start == kInvalidCounterTimestamp || end == kInvalidCounterTimestamp) {
            return CounterResolveError::InvalidCounterValue;
        }
        if (end < start) {
            return CounterResolveError::NonMonotonicTimestamp;
        }
        if (!windowed && event + 1 < plan.event_count &&
            timestamps[event * 2 + 2] < end) {
            return CounterResolveError::NonMonotonicTimestamp;
        }
    }
    for (std::size_t event = 0; event < plan.event_count; ++event) {
        const std::uint64_t start = timestamps[event * 2];
        const std::uint64_t end = timestamps[event * 2 + 1];
        const bool window_end = windowed && window_end_markers[event] != 0U;
        const std::uint64_t next =
            !window_end && event + 1 < plan.event_count
                ? timestamps[event * 2 + 2]
                : end;
        const std::uint64_t gap = next >= end ? next - end : 0;
        timings[event] = {
            .class_id = events[event].class_id,
            .event_index = event,
            .start_timestamp = start,
            .end_timestamp = end,
            .kernel_ticks = end - start,
            .gap_to_next_ticks = gap,
        };
    }
    return CounterResolveError::None;
}

} // namespace

CounterResolveError resolve_counter_event_timings(
    const CounterEventPlan& plan, std::span<const CounterEvent> events,
    std::span<const std::uint64_t> timestamps,
    std::span<CounterEventTiming> timings) noexcept {
    return resolve_counter_event_timings_impl(
        plan, events, timestamps, {}, false, timings);
}

CounterResolveError resolve_counter_event_timings(
    const CounterEventPlan& plan, std::span<const CounterEvent> events,
    std::span<const std::uint64_t> timestamps,
    std::span<const std::uint8_t> window_end_markers,
    std::span<CounterEventTiming> timings) noexcept {
    return resolve_counter_event_timings_impl(
        plan, events, timestamps, window_end_markers, true, timings);
}

CounterSampleBufferCreateResult
create_timestamp_counter_sample_buffer(const MetalDevice& device,
                                       std::size_t sample_count) noexcept {
    if (sample_count == 0) {
        return {
            .error = CounterSampleBufferCreateError::InvalidSampleCount,
            .buffer = std::nullopt,
        };
    }
    if constexpr (sizeof(NSUInteger) < sizeof(std::size_t)) {
        if (sample_count > std::numeric_limits<NSUInteger>::max()) {
            return {
                .error = CounterSampleBufferCreateError::SampleCountOutOfRange,
                .buffer = std::nullopt,
            };
        }
    }
    if (!device) {
        return {
            .error = CounterSampleBufferCreateError::InvalidDevice,
            .buffer = std::nullopt,
        };
    }
    void* native_device = device.storage_->object.get();
    return MetalCounterSampleBuffer::create_for_native_device(
        native_device, sample_count, CounterSamplingMode::DispatchBoundary);
}

CounterSampleBufferCreateResult
create_stage_timestamp_counter_sample_buffer(
    const MetalCommandBuffer& command_buffer, std::size_t sample_count) noexcept {
    if (sample_count == 0) {
        return {
            .error = CounterSampleBufferCreateError::InvalidSampleCount,
            .buffer = std::nullopt,
        };
    }
    if (sample_count > kMaxStageBoundarySampleCount || (sample_count & 1U) != 0U) {
        return {
            .error = CounterSampleBufferCreateError::SampleCountOutOfRange,
            .buffer = std::nullopt,
        };
    }
    if (!command_buffer) {
        return {
            .error = CounterSampleBufferCreateError::InvalidCommandBuffer,
            .buffer = std::nullopt,
        };
    }
    @autoreleasepool {
        id<MTLCommandBuffer> native_command_buffer =
            (__bridge id<MTLCommandBuffer>)command_buffer.object_;
        id<MTLDevice> native_device = native_command_buffer.device;
        return MetalCounterSampleBuffer::create_for_native_device(
            (__bridge void*)native_device, sample_count,
            CounterSamplingMode::StageBoundaryEncoderSplit);
    }
}

CounterSampleBufferCreateResult
MetalCounterSampleBuffer::create_for_native_device(
    void* native_device_object, std::size_t sample_count,
    CounterSamplingMode mode) noexcept {
    if (native_device_object == nullptr) {
        return {
            .error = CounterSampleBufferCreateError::InvalidDevice,
            .buffer = std::nullopt,
        };
    }

    @autoreleasepool {
        id<MTLDevice> native_device = (__bridge id<MTLDevice>)native_device_object;
        const MTLCounterSamplingPoint point =
            mode == CounterSamplingMode::DispatchBoundary
                ? MTLCounterSamplingPointAtDispatchBoundary
                : MTLCounterSamplingPointAtStageBoundary;
        if (![native_device supportsCounterSampling:point]) {
            return {
                .error =
                    mode == CounterSamplingMode::DispatchBoundary
                        ? CounterSampleBufferCreateError::DispatchBoundarySamplingUnsupported
                        : CounterSampleBufferCreateError::StageBoundarySamplingUnsupported,
                .buffer = std::nullopt,
            };
        }
        id<MTLCounterSet> timestamp_set = nil;
        for (id<MTLCounterSet> candidate in native_device.counterSets) {
            if ([candidate.name isEqualToString:MTLCommonCounterSetTimestamp]) {
                timestamp_set = candidate;
                break;
            }
        }
        if (timestamp_set == nil) {
            return {
                .error = CounterSampleBufferCreateError::TimestampCounterSetUnavailable,
                .buffer = std::nullopt,
            };
        }
        MTLCounterSampleBufferDescriptor* sample_descriptor =
            [MTLCounterSampleBufferDescriptor new];
        sample_descriptor.counterSet = timestamp_set;
        sample_descriptor.storageMode = MTLStorageModeShared;
        sample_descriptor.sampleCount = static_cast<NSUInteger>(sample_count);
        NSError* error = nil;
        id<MTLCounterSampleBuffer> native_buffer =
            [native_device newCounterSampleBufferWithDescriptor:sample_descriptor
                                                           error:&error];
        (void)error;
        if (native_buffer == nil) {
            return {
                .error = CounterSampleBufferCreateError::SampleBufferCreationFailed,
                .buffer = std::nullopt,
            };
        }
        void* retained = (__bridge_retained void*)native_buffer;
        if (retained == nullptr) {
            return {
                .error = CounterSampleBufferCreateError::OwnershipFailure,
                .buffer = std::nullopt,
            };
        }
        void* retained_descriptor = nullptr;
        if (mode == CounterSamplingMode::StageBoundaryEncoderSplit) {
            NSMutableArray<MTLComputePassDescriptor*>* descriptors =
                [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(sample_count / 2U)];
            if (descriptors == nil) {
                release_object(retained);
                return {
                    .error = CounterSampleBufferCreateError::StageDescriptorCreationFailed,
                    .buffer = std::nullopt,
                };
            }
            for (std::size_t pair_index = 0; pair_index < sample_count / 2U; ++pair_index) {
                MTLComputePassDescriptor* pass_descriptor =
                    [MTLComputePassDescriptor new];
                if (pass_descriptor == nil) {
                    release_object(retained);
                    return {
                        .error =
                            CounterSampleBufferCreateError::StageDescriptorCreationFailed,
                        .buffer = std::nullopt,
                    };
                }
                pass_descriptor.sampleBufferAttachments[0].sampleBuffer = native_buffer;
                pass_descriptor.sampleBufferAttachments[0].startOfEncoderSampleIndex =
                    static_cast<NSUInteger>(pair_index * 2U);
                pass_descriptor.sampleBufferAttachments[0].endOfEncoderSampleIndex =
                    static_cast<NSUInteger>(pair_index * 2U + 1U);
                [descriptors addObject:pass_descriptor];
            }
            NSArray<MTLComputePassDescriptor*>* frozen_descriptors =
                [descriptors copy];
            retained_descriptor = (__bridge_retained void*)frozen_descriptors;
            if (retained_descriptor == nullptr) {
                release_object(retained);
                return {
                    .error = CounterSampleBufferCreateError::OwnershipFailure,
                    .buffer = std::nullopt,
                };
            }
        }
        return {
            .error = CounterSampleBufferCreateError::None,
            .buffer =
                MetalCounterSampleBuffer(retained, retained_descriptor, sample_count, mode),
        };
    }
}

CounterSampleError sample_counter(MetalComputePass& compute_pass,
                                  const MetalCounterSampleBuffer& buffer,
                                  std::size_t sample_index) noexcept {
    if (!compute_pass) {
        return CounterSampleError::InvalidComputePass;
    }
    if (!buffer) {
        return CounterSampleError::InvalidBuffer;
    }
    const CounterSampleError mode_validation =
        validate_dispatch_counter_sampling_mode(buffer.mode_);
    if (mode_validation != CounterSampleError::None) {
        return mode_validation;
    }
    const CounterSampleError validation =
        validate_counter_sample_index(buffer.sample_capacity_, sample_index);
    if (validation != CounterSampleError::None) {
        return validation;
    }
    id<MTLComputeCommandEncoder> native_encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_pass.encoder_;
    id<MTLCounterSampleBuffer> native_buffer = (__bridge id<MTLCounterSampleBuffer>)buffer.object_;
    [native_encoder sampleCountersInBuffer:native_buffer
                             atSampleIndex:static_cast<NSUInteger>(sample_index)
                               withBarrier:YES];
    return CounterSampleError::None;
}

CounterResolveError resolve_counter_samples(const MetalCounterSampleBuffer& buffer,
                                            std::size_t first_sample, std::size_t sample_count,
                                            std::span<std::uint64_t> timestamps) noexcept {
    if (!buffer) {
        return CounterResolveError::InvalidBuffer;
    }
    if (sample_count == 0) {
        return CounterResolveError::EmptyRange;
    }
    std::size_t end = 0;
    if (!checked_range(first_sample, sample_count, buffer.sample_capacity_, end)) {
        return CounterResolveError::RangeOutOfBounds;
    }
    (void)end;
    if (timestamps.size() < sample_count) {
        return CounterResolveError::OutputTooSmall;
    }
    @autoreleasepool {
        id<MTLCounterSampleBuffer> native_buffer =
            (__bridge id<MTLCounterSampleBuffer>)buffer.object_;
        NSData* data =
            [native_buffer resolveCounterRange:NSMakeRange(static_cast<NSUInteger>(first_sample),
                                                           static_cast<NSUInteger>(sample_count))];
        if (data == nil) {
            return CounterResolveError::NativeResolveFailed;
        }
        if (sample_count >
            std::numeric_limits<std::size_t>::max() / sizeof(MTLCounterResultTimestamp)) {
            return CounterResolveError::RangeOutOfBounds;
        }
        const std::size_t required_bytes = sample_count * sizeof(MTLCounterResultTimestamp);
        if (data.length < required_bytes) {
            return CounterResolveError::NativeDataTruncated;
        }
        if (data.bytes == nullptr) {
            return CounterResolveError::NativeResolveFailed;
        }
        const auto* resolved = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
        for (std::size_t index = 0; index < sample_count; ++index) {
            if (resolved[index].timestamp == MTLCounterErrorValue) {
                return CounterResolveError::InvalidCounterValue;
            }
        }
        for (std::size_t index = 0; index < sample_count; ++index) {
            timestamps[index] = resolved[index].timestamp;
        }
    }
    return CounterResolveError::None;
}

} // namespace tatara::backend::metal
