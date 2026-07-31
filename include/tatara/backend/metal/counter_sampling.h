#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace tatara::backend::metal {

inline constexpr std::uint64_t kInvalidCounterTimestamp = static_cast<std::uint64_t>(-1);
inline constexpr std::size_t kMaxStageBoundarySampleCount = 1024;

class MetalComputePass;
class MetalCommandBuffer;
class MetalDevice;
struct MetalStageComputePassResult;

enum class CounterSamplingMode : std::uint8_t {
    DispatchBoundary,
    StageBoundaryEncoderSplit,
};

struct CounterEvent {
    std::uint32_t class_id{0};
};

struct CounterEventPlan {
    bool enabled{false};
    CounterSamplingMode mode{CounterSamplingMode::DispatchBoundary};
    std::size_t event_count{0};
    std::size_t sample_count{0};
    std::uint64_t identity{0};
};

struct CounterSamplePair {
    std::size_t start{0};
    std::size_t end{0};
};

struct CounterEventTiming {
    std::uint32_t class_id{0};
    std::size_t event_index{0};
    std::uint64_t start_timestamp{0};
    std::uint64_t end_timestamp{0};
    std::uint64_t kernel_ticks{0};
    std::uint64_t gap_to_next_ticks{0};
};

enum class CounterPlanError : std::uint8_t {
    None,
    MeasurementDisabled,
    EmptyEvents,
    SampleCountOverflow,
    InvalidPlan,
    EventIndexOutOfBounds,
};

enum class CounterSampleBufferCreateError : std::uint8_t {
    None,
    Uninitialized,
    InvalidSampleCount,
    SampleCountOutOfRange,
    InvalidDevice,
    DispatchBoundarySamplingUnsupported,
    TimestampCounterSetUnavailable,
    SampleBufferCreationFailed,
    OwnershipFailure,
    InvalidCommandBuffer,
    InvalidSamplingMode,
    StageBoundarySamplingUnsupported,
    StageDescriptorCreationFailed,
};

enum class CounterSampleError : std::uint8_t {
    None,
    InvalidComputePass,
    InvalidBuffer,
    SampleIndexOutOfBounds,
    InvalidSamplePair,
    SamplingModeMismatch,
};

enum class CounterStageSampleError : std::uint8_t {
    None,
    InvalidCommandBuffer,
    InvalidComputePass,
    InvalidBuffer,
    SamplingModeMismatch,
    InvalidSamplePair,
    SampleIndexOutOfBounds,
    EncoderCreationFailed,
};

enum class CounterResolveError : std::uint8_t {
    None,
    MeasurementDisabled,
    InvalidPlan,
    EventIdentityMismatch,
    InvalidBuffer,
    EmptyRange,
    RangeOutOfBounds,
    InsufficientTimestampData,
    OutputTooSmall,
    NativeResolveFailed,
    NativeDataTruncated,
    InvalidCounterValue,
    NonMonotonicTimestamp,
    InvalidWindowPlan,
};

struct CounterSampleBufferCreateResult;

class MetalCounterSampleBuffer {
  public:
    MetalCounterSampleBuffer() noexcept;
    ~MetalCounterSampleBuffer();

    MetalCounterSampleBuffer(const MetalCounterSampleBuffer&) = delete;
    MetalCounterSampleBuffer& operator=(const MetalCounterSampleBuffer&) = delete;
    MetalCounterSampleBuffer(MetalCounterSampleBuffer&&) noexcept;
    MetalCounterSampleBuffer& operator=(MetalCounterSampleBuffer&&) noexcept;

    std::size_t sample_capacity() const noexcept;
    CounterSamplingMode sampling_mode() const noexcept;
    explicit operator bool() const noexcept;

  private:
    friend struct CounterSampleBufferCreateResult;
    friend CounterSampleError sample_counter(MetalComputePass&, const MetalCounterSampleBuffer&,
                                             std::size_t) noexcept;
    friend struct MetalStageComputePassResult;
    friend MetalStageComputePassResult
    begin_stage_sampled_compute_pass(MetalCommandBuffer&&,
                                     const MetalCounterSampleBuffer&,
                                     CounterSamplePair) noexcept;
    friend CounterStageSampleError
    split_stage_sampled_compute_pass(MetalComputePass&,
                                     const MetalCounterSampleBuffer&,
                                     CounterSamplePair) noexcept;
    friend CounterResolveError resolve_counter_samples(const MetalCounterSampleBuffer&, std::size_t,
                                                       std::size_t,
                                                       std::span<std::uint64_t>) noexcept;
    friend CounterSampleBufferCreateResult
    create_timestamp_counter_sample_buffer(const MetalDevice&,
                                           std::size_t) noexcept;
    friend CounterSampleBufferCreateResult
    create_stage_timestamp_counter_sample_buffer(const MetalCommandBuffer&,
                                                 std::size_t) noexcept;

    MetalCounterSampleBuffer(void* object, void* stage_descriptor,
                             std::size_t sample_capacity,
                             CounterSamplingMode mode) noexcept;
    static CounterSampleBufferCreateResult
    create_for_native_device(void* native_device, std::size_t sample_count,
                             CounterSamplingMode mode) noexcept;
    void reset() noexcept;

    void* object_{nullptr};
    void* stage_descriptor_{nullptr};
    std::size_t sample_capacity_{0};
    CounterSamplingMode mode_{CounterSamplingMode::DispatchBoundary};
};

struct CounterSampleBufferCreateResult {
    CounterSampleBufferCreateError error{CounterSampleBufferCreateError::Uninitialized};
    std::optional<MetalCounterSampleBuffer> buffer{};

    explicit operator bool() const noexcept {
        return error == CounterSampleBufferCreateError::None && buffer.has_value();
    }
};

CounterPlanError plan_counter_events(bool enabled, std::span<const CounterEvent> events,
                                     CounterEventPlan& plan) noexcept;
CounterPlanError plan_counter_events(bool enabled, CounterSamplingMode mode,
                                     std::span<const CounterEvent> events,
                                     CounterEventPlan& plan) noexcept;
CounterPlanError counter_event_sample_pair(const CounterEventPlan& plan, std::size_t event_index,
                                           CounterSamplePair& pair) noexcept;
CounterSampleError validate_counter_sample_index(std::size_t sample_capacity,
                                                 std::size_t sample_index) noexcept;
CounterSampleError validate_counter_sample_pair(std::size_t sample_capacity,
                                                CounterSamplePair pair) noexcept;
CounterSampleError
validate_dispatch_counter_sampling_mode(CounterSamplingMode mode) noexcept;
CounterResolveError resolve_counter_event_timings(const CounterEventPlan& plan,
                                                  std::span<const CounterEvent> events,
                                                  std::span<const std::uint64_t> timestamps,
                                                  std::span<CounterEventTiming> timings) noexcept;
CounterResolveError resolve_counter_event_timings(
    const CounterEventPlan& plan, std::span<const CounterEvent> events,
    std::span<const std::uint64_t> timestamps,
    std::span<const std::uint8_t> window_end_markers,
    std::span<CounterEventTiming> timings) noexcept;

CounterSampleBufferCreateResult
create_timestamp_counter_sample_buffer(const MetalDevice& device,
                                       std::size_t sample_count) noexcept;
CounterSampleBufferCreateResult
create_stage_timestamp_counter_sample_buffer(const MetalCommandBuffer& command_buffer,
                                             std::size_t sample_count) noexcept;
CounterSampleError sample_counter(MetalComputePass& compute_pass,
                                  const MetalCounterSampleBuffer& buffer,
                                  std::size_t sample_index) noexcept;
CounterResolveError resolve_counter_samples(const MetalCounterSampleBuffer& buffer,
                                            std::size_t first_sample, std::size_t sample_count,
                                            std::span<std::uint64_t> timestamps) noexcept;

} // namespace tatara::backend::metal
