#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"

namespace tatara::backend::metal {

struct MetalComputePassResult;
struct MetalBlitPassResult;
struct MetalIndirectCommandBufferResult;
struct MetalPendingExecutionResult;
struct MetalExecutionResult;
struct MetalTimedExecutionResult;
struct MetalStageComputePassResult;
struct CounterSampleBufferCreateResult;
class MetalCounterSampleBuffer;
class MetalIndirectCommandBuffer;
enum class CounterSampleError : std::uint8_t;
enum class CounterStageSampleError : std::uint8_t;
enum class MetalResourceUsage : std::uint8_t;
struct CounterSamplePair;

inline constexpr std::uint32_t kMaxBufferArgumentIndex = 30;
// Metal's ceiling on an inline constant bind; larger payloads must travel in a
// shared buffer instead.
inline constexpr std::uint32_t kMaxInlineConstantBytes = 4096;

struct MetalSize {
    std::uint64_t width;
    std::uint64_t height;
    std::uint64_t depth;
};

class MetalCommandBuffer {
  public:
    MetalCommandBuffer() noexcept;
    ~MetalCommandBuffer();

    MetalCommandBuffer(const MetalCommandBuffer&) = delete;
    MetalCommandBuffer& operator=(const MetalCommandBuffer&) = delete;
    MetalCommandBuffer(MetalCommandBuffer&&) noexcept;
    MetalCommandBuffer& operator=(MetalCommandBuffer&&) noexcept;

    explicit operator bool() const noexcept;

  private:
    friend MetalCommandBufferResult create_command_buffer(const MetalCommandQueue&);
    friend MetalComputePassResult begin_compute_pass(MetalCommandBuffer&&);
    friend MetalStageComputePassResult
    begin_stage_sampled_compute_pass(MetalCommandBuffer&&,
                                     const MetalCounterSampleBuffer&,
                                     CounterSamplePair) noexcept;
    friend MetalBlitPassResult begin_blit_pass(MetalCommandBuffer&&);
    friend MetalCommandError
    encode_wait_for_event(MetalCommandBuffer&, const MetalEvent&,
                          std::uint64_t);
    friend MetalCommandError
    encode_signal_event(MetalCommandBuffer&, const MetalEvent&,
                        std::uint64_t);
    friend MetalCommandBufferResult end_compute_pass(MetalComputePass&&);
    friend MetalCommandBufferResult end_blit_pass(MetalBlitPass&&);
    friend MetalPendingExecutionResult commit(MetalCommandBuffer&&);
    friend CounterSampleBufferCreateResult
    create_stage_timestamp_counter_sample_buffer(const MetalCommandBuffer&,
                                                 std::size_t) noexcept;

    explicit MetalCommandBuffer(void* object) noexcept;
    void reset() noexcept;

    void* object_ = nullptr;
};

class MetalIndirectCommandBuffer {
  public:
    MetalIndirectCommandBuffer() noexcept;
    ~MetalIndirectCommandBuffer();

    MetalIndirectCommandBuffer(const MetalIndirectCommandBuffer&) = delete;
    MetalIndirectCommandBuffer&
    operator=(const MetalIndirectCommandBuffer&) = delete;
    MetalIndirectCommandBuffer(MetalIndirectCommandBuffer&&) noexcept;
    MetalIndirectCommandBuffer&
    operator=(MetalIndirectCommandBuffer&&) noexcept;

    std::uint32_t maximum_command_count() const noexcept;
    std::uint32_t maximum_kernel_buffer_bind_count() const noexcept;
    explicit operator bool() const noexcept;

  private:
    friend MetalIndirectCommandBufferResult
    create_compute_indirect_command_buffer(const MetalDevice&, std::uint32_t,
                                           std::uint32_t);
    friend MetalCommandError
    reset_indirect_commands(MetalIndirectCommandBuffer&, std::uint32_t,
                            std::uint32_t);
    friend MetalCommandError
    set_indirect_compute_pipeline(MetalIndirectCommandBuffer&, std::uint32_t,
                                  const MetalComputePipeline&);
    friend MetalCommandError
    set_indirect_buffer(MetalIndirectCommandBuffer&, std::uint32_t,
                        const MetalBuffer&, std::uint64_t, std::uint32_t);
    friend MetalCommandError
    set_indirect_barrier(MetalIndirectCommandBuffer&, std::uint32_t);
    friend MetalCommandError
    clear_indirect_barrier(MetalIndirectCommandBuffer&, std::uint32_t);
    friend MetalCommandError
    dispatch_indirect_threadgroups(MetalIndirectCommandBuffer&, std::uint32_t,
                                   MetalSize, MetalSize);
    friend MetalCommandError
    execute_indirect_commands(MetalComputePass&,
                              const MetalIndirectCommandBuffer&,
                              std::uint32_t, std::uint32_t);
    friend std::uint64_t
    indirect_command_buffer_identity(
        const MetalIndirectCommandBuffer&) noexcept;

    MetalIndirectCommandBuffer(
        void* object, std::uint32_t maximum_command_count,
        std::uint32_t maximum_kernel_buffer_bind_count) noexcept;
    void reset() noexcept;

    void* object_ = nullptr;
    std::uint32_t maximum_command_count_ = 0;
    std::uint32_t maximum_kernel_buffer_bind_count_ = 0;
};

class MetalComputePass {
  public:
    MetalComputePass() noexcept;
    ~MetalComputePass();

    MetalComputePass(const MetalComputePass&) = delete;
    MetalComputePass& operator=(const MetalComputePass&) = delete;
    MetalComputePass(MetalComputePass&&) noexcept;
    MetalComputePass& operator=(MetalComputePass&&) noexcept;

    explicit operator bool() const noexcept;

  private:
    friend MetalComputePassResult begin_compute_pass(MetalCommandBuffer&&);
    friend MetalCommandError set_compute_pipeline(MetalComputePass&, const MetalComputePipeline&);
    friend MetalCommandError set_buffer(MetalComputePass&, const MetalBuffer&, std::uint64_t,
                                        std::uint32_t);
    friend MetalCommandError dispatch_threadgroups(MetalComputePass&, MetalSize, MetalSize);
    friend MetalCommandError dispatch_threadgroups_indirect(MetalComputePass&, const MetalBuffer&,
                                                            std::uint64_t, MetalSize);
    friend MetalCommandError set_bytes(MetalComputePass&, const void*, std::uint32_t,
                                       std::uint32_t);
    friend MetalCommandError memory_barrier(MetalComputePass&);
    friend MetalCommandError
    use_buffer_resource(MetalComputePass&, const MetalBuffer&,
                        MetalResourceUsage);
    friend MetalCommandError
    execute_indirect_commands(MetalComputePass&,
                              const MetalIndirectCommandBuffer&,
                              std::uint32_t, std::uint32_t);
    friend CounterSampleError sample_counter(MetalComputePass&, const MetalCounterSampleBuffer&,
                                             std::size_t) noexcept;
    friend MetalStageComputePassResult
    begin_stage_sampled_compute_pass(MetalCommandBuffer&&,
                                     const MetalCounterSampleBuffer&,
                                     CounterSamplePair) noexcept;
    friend CounterStageSampleError
    split_stage_sampled_compute_pass(MetalComputePass&,
                                     const MetalCounterSampleBuffer&,
                                     CounterSamplePair) noexcept;
    friend MetalCommandBufferResult end_compute_pass(MetalComputePass&&);

    explicit MetalComputePass(void* command_buffer, void* encoder) noexcept;
    void reset() noexcept;

    void* command_buffer_ = nullptr;
    void* encoder_ = nullptr;
};

class MetalBlitPass {
  public:
    MetalBlitPass() noexcept;
    ~MetalBlitPass();

    MetalBlitPass(const MetalBlitPass&) = delete;
    MetalBlitPass& operator=(const MetalBlitPass&) = delete;
    MetalBlitPass(MetalBlitPass&&) noexcept;
    MetalBlitPass& operator=(MetalBlitPass&&) noexcept;

    explicit operator bool() const noexcept;

  private:
    friend MetalBlitPassResult begin_blit_pass(MetalCommandBuffer&&);
    friend MetalCommandError copy_buffer(MetalBlitPass&, const MetalBuffer&, std::uint64_t,
                                         MetalBuffer&, std::uint64_t, std::uint64_t);
    friend MetalCommandError fill_buffer(MetalBlitPass&, MetalBuffer&, std::uint64_t, std::uint64_t,
                                         std::byte);
    friend MetalCommandBufferResult end_blit_pass(MetalBlitPass&&);

    explicit MetalBlitPass(void* command_buffer, void* encoder) noexcept;
    void reset() noexcept;

    void* command_buffer_ = nullptr;
    void* encoder_ = nullptr;
};

class MetalPendingExecution {
  public:
    MetalPendingExecution() noexcept;
    ~MetalPendingExecution();

    MetalPendingExecution(const MetalPendingExecution&) = delete;
    MetalPendingExecution& operator=(const MetalPendingExecution&) = delete;
    MetalPendingExecution(MetalPendingExecution&&) noexcept;
    MetalPendingExecution& operator=(MetalPendingExecution&&) noexcept;

    explicit operator bool() const noexcept;

  private:
    friend MetalPendingExecutionResult commit(MetalCommandBuffer&&);
    friend MetalExecutionResult wait_until_completed(MetalPendingExecution&&);
    friend MetalTimedExecutionResult wait_until_completed_timed(MetalPendingExecution&&);

    explicit MetalPendingExecution(void* object) noexcept;
    void reset() noexcept;

    void* object_ = nullptr;
};

enum class MetalCommandError : std::uint8_t {
    None,
    InvalidCommandQueue,
    InvalidCommandBuffer,
    InvalidEvent,
    InvalidEventValue,
    InvalidComputePass,
    InvalidBlitPass,
    InvalidPipeline,
    InvalidBuffer,
    InvalidBufferIndex,
    InvalidBufferOffset,
    InvalidSourceRange,
    InvalidDestinationRange,
    InvalidCopyExtent,
    InvalidFillExtent,
    InvalidDispatchExtent,
    ContextOutOfRange,
    StateSlotMismatch,
    InvalidPendingExecution,
    CommandBufferCreationFailed,
    EncoderCreationFailed,
    InvalidIndirectCommandBuffer,
    InvalidIndirectCommandCount,
    InvalidIndirectCommandIndex,
    InvalidIndirectCommandRange,
    InvalidResourceUsage,
    IndirectCommandBufferCreationFailed,
    InvalidDevice,
};

enum class MetalResourceUsage : std::uint8_t {
    Read,
    Write,
    ReadWrite,
};

enum class MetalExecutionState : std::uint8_t {
    NotObserved,
    NotEnqueued,
    Enqueued,
    Committed,
    Scheduled,
    Completed,
    Error,
    Unknown,
};

enum class MetalExecutionFailure : std::uint8_t {
    None,
    Internal,
    Timeout,
    PageFault,
    AccessRevoked,
    NotPermitted,
    OutOfMemory,
    InvalidResource,
    Memoryless,
    DeviceRemoved,
    StackOverflow,
    UnknownError,
    UnexpectedStatus,
};

// Fixed storage includes the terminal NUL. Completion reporting therefore
// retains at most 511 diagnostic bytes and never allocates a dynamic string.
inline constexpr std::size_t kExecutionDiagnosticStorageBytes = 512;

struct MetalExecutionDiagnostic {
    std::array<char, kExecutionDiagnosticStorageBytes> bytes{};
    std::size_t length_bytes = 0;
    bool truncated = false;

    const char* c_str() const noexcept {
        return bytes.data();
    }

    std::string_view view() const noexcept {
        const std::size_t bounded_length =
            length_bytes < bytes.size() ? length_bytes : bytes.size() - 1;
        return std::string_view(bytes.data(), bounded_length);
    }

    bool empty() const noexcept {
        return length_bytes == 0;
    }
};

struct MetalCommandBufferResult {
    MetalCommandError error;
    std::optional<MetalCommandBuffer> command_buffer;

    explicit operator bool() const noexcept {
        return error == MetalCommandError::None && command_buffer.has_value();
    }
};

struct MetalComputePassResult {
    MetalCommandError error;
    std::optional<MetalComputePass> compute_pass;

    explicit operator bool() const noexcept {
        return error == MetalCommandError::None && compute_pass.has_value();
    }
};

struct MetalIndirectCommandBufferResult {
    MetalCommandError error;
    std::optional<MetalIndirectCommandBuffer> indirect_command_buffer;

    explicit operator bool() const noexcept {
        return error == MetalCommandError::None &&
               indirect_command_buffer.has_value();
    }
};

struct MetalStageComputePassResult {
    CounterStageSampleError error;
    std::optional<MetalComputePass> compute_pass;

    explicit operator bool() const noexcept;
};

struct MetalBlitPassResult {
    MetalCommandError error;
    std::optional<MetalBlitPass> blit_pass;

    explicit operator bool() const noexcept {
        return error == MetalCommandError::None && blit_pass.has_value();
    }
};

struct MetalPendingExecutionResult {
    MetalCommandError error;
    std::optional<MetalPendingExecution> pending_execution;

    explicit operator bool() const noexcept {
        return error == MetalCommandError::None && pending_execution.has_value();
    }
};

struct MetalExecutionResult {
    MetalCommandError error;
    bool completed;
    MetalExecutionState state;
    MetalExecutionFailure failure;
    std::int64_t native_error_code;
    bool has_native_error_code;
    MetalExecutionDiagnostic failure_description;

    explicit operator bool() const noexcept {
        return error == MetalCommandError::None && completed &&
               state == MetalExecutionState::Completed && failure == MetalExecutionFailure::None;
    }
};

// Host-clock timestamps a command buffer publishes once it completes. The
// schedule pair brackets the driver's own work for the submission, which is
// where per-command-buffer resource residency validation lands; the GPU pair
// brackets execution itself. Metal leaves a timestamp at zero when it did not
// record one, so a reader treats a non-increasing pair as unavailable rather
// than as a zero-length interval.
struct MetalExecutionTiming {
    double schedule_start_seconds;
    double schedule_end_seconds;
    double gpu_start_seconds;
    double gpu_end_seconds;
};

struct MetalTimedExecutionResult {
    MetalCommandError error;
    bool completed;
    MetalExecutionState state;
    MetalExecutionFailure failure;
    std::int64_t native_error_code;
    bool has_native_error_code;
    MetalExecutionDiagnostic failure_description;
    MetalExecutionTiming timing;

    explicit operator bool() const noexcept {
        return error == MetalCommandError::None && completed &&
               state == MetalExecutionState::Completed && failure == MetalExecutionFailure::None;
    }
};

[[nodiscard]] MetalCommandBufferResult
create_command_buffer(const MetalCommandQueue& command_queue);
[[nodiscard]] MetalComputePassResult begin_compute_pass(MetalCommandBuffer&& command_buffer);
MetalCommandError encode_wait_for_event(
    MetalCommandBuffer& command_buffer, const MetalEvent& event,
    std::uint64_t value);
MetalCommandError encode_signal_event(
    MetalCommandBuffer& command_buffer, const MetalEvent& event,
    std::uint64_t value);
[[nodiscard]] MetalIndirectCommandBufferResult
create_compute_indirect_command_buffer(
    const MetalDevice& device, std::uint32_t maximum_command_count,
    std::uint32_t maximum_kernel_buffer_bind_count);
[[nodiscard]] MetalStageComputePassResult
begin_stage_sampled_compute_pass(MetalCommandBuffer&& command_buffer,
                                 const MetalCounterSampleBuffer& samples,
                                 CounterSamplePair pair) noexcept;
CounterStageSampleError
split_stage_sampled_compute_pass(MetalComputePass& compute_pass,
                                 const MetalCounterSampleBuffer& samples,
                                 CounterSamplePair pair) noexcept;
[[nodiscard]] MetalBlitPassResult begin_blit_pass(MetalCommandBuffer&& command_buffer);
MetalCommandError set_compute_pipeline(MetalComputePass& compute_pass,
                                       const MetalComputePipeline& pipeline);
MetalCommandError set_buffer(MetalComputePass& compute_pass, const MetalBuffer& buffer,
                             std::uint64_t offset_bytes, std::uint32_t index);
MetalCommandError dispatch_threadgroups(MetalComputePass& compute_pass, MetalSize threadgroups,
                                        MetalSize threads_per_threadgroup);
// Dispatches a grid written by an earlier GPU kernel as three consecutive
// uint32 extents. Bounds and the fixed threadgroup shape are validated before
// the Objective-C call; the grid itself is intentionally never read by the CPU.
MetalCommandError dispatch_threadgroups_indirect(MetalComputePass& compute_pass,
                                                 const MetalBuffer& arguments,
                                                 std::uint64_t argument_offset_bytes,
                                                 MetalSize threads_per_threadgroup);
// Embeds a small constant argument in the command stream (the sealed
// per-token context/capacity constants; a shared buffer would race the
// pipelined token loop). Bounded to 4 KiB, the Metal setBytes limit.
MetalCommandError set_bytes(MetalComputePass& compute_pass, const void* bytes,
                            std::uint32_t length_bytes, std::uint32_t index);
// Buffer-scope barrier ordering earlier dispatches before later ones inside
// one pass; the sealed decode step is one pass with barriers between
// dispatches (contract extension for the composition boundary).
MetalCommandError memory_barrier(MetalComputePass& compute_pass);
MetalCommandError reset_indirect_commands(
    MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t first_command, std::uint32_t command_count);
MetalCommandError set_indirect_compute_pipeline(
    MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t command_index, const MetalComputePipeline& pipeline);
MetalCommandError set_indirect_buffer(
    MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t command_index, const MetalBuffer& buffer,
    std::uint64_t offset_bytes, std::uint32_t buffer_index);
MetalCommandError set_indirect_barrier(
    MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t command_index);
MetalCommandError clear_indirect_barrier(
    MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t command_index);
MetalCommandError dispatch_indirect_threadgroups(
    MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t command_index, MetalSize threadgroups,
    MetalSize threads_per_threadgroup);
MetalCommandError use_buffer_resource(MetalComputePass& compute_pass,
                                      const MetalBuffer& buffer,
                                      MetalResourceUsage usage);
MetalCommandError execute_indirect_commands(
    MetalComputePass& compute_pass,
    const MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t first_command, std::uint32_t command_count);
std::uint64_t metal_buffer_identity(const MetalBuffer& buffer) noexcept;
std::uint64_t indirect_command_buffer_identity(
    const MetalIndirectCommandBuffer& indirect_command_buffer) noexcept;
[[nodiscard]] MetalCommandBufferResult end_compute_pass(MetalComputePass&& compute_pass);
MetalCommandError copy_buffer(MetalBlitPass& blit_pass, const MetalBuffer& source,
                              std::uint64_t source_offset_bytes, MetalBuffer& destination,
                              std::uint64_t destination_offset_bytes, std::uint64_t length_bytes);
MetalCommandError fill_buffer(MetalBlitPass& blit_pass, MetalBuffer& destination,
                              std::uint64_t destination_offset_bytes, std::uint64_t length_bytes,
                              std::byte value);
[[nodiscard]] MetalCommandBufferResult end_blit_pass(MetalBlitPass&& blit_pass);
[[nodiscard]] MetalPendingExecutionResult commit(MetalCommandBuffer&& command_buffer);
[[nodiscard]] MetalExecutionResult wait_until_completed(MetalPendingExecution&& pending_execution);
// The same wait, additionally reporting the timestamps above. Measurement
// path only: a performance probe needs GPU execution time separated from the
// wall time the caller observes, and those timestamps are readable only after
// the buffer completes.
[[nodiscard]] MetalTimedExecutionResult
wait_until_completed_timed(MetalPendingExecution&& pending_execution);

} // namespace tatara::backend::metal
