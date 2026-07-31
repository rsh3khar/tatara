#include "tatara/backend/metal/commands.h"

#include "tatara/backend/metal/counter_sampling.h"
#include "backend/metal/command_report.h"
#include "backend/metal/resource_storage.h"

#include <limits>
#include <utility>

namespace tatara::backend::metal {
namespace {

// The pure command-report unit deliberately has no Metal dependency. These
// assertions bind its stable numeric input table to the exact SDK used for
// this Objective-C++ build so an Apple enum change fails compilation.
static_assert(static_cast<std::uint64_t>(MTLCommandBufferStatusNotEnqueued) ==
              detail::kNativeStatusNotEnqueued);
static_assert(static_cast<std::uint64_t>(MTLCommandBufferStatusEnqueued) ==
              detail::kNativeStatusEnqueued);
static_assert(static_cast<std::uint64_t>(MTLCommandBufferStatusCommitted) ==
              detail::kNativeStatusCommitted);
static_assert(static_cast<std::uint64_t>(MTLCommandBufferStatusScheduled) ==
              detail::kNativeStatusScheduled);
static_assert(static_cast<std::uint64_t>(MTLCommandBufferStatusCompleted) ==
              detail::kNativeStatusCompleted);
static_assert(static_cast<std::uint64_t>(MTLCommandBufferStatusError) ==
              detail::kNativeStatusError);
static_assert(static_cast<std::int64_t>(MTLCommandBufferErrorNone) == detail::kNativeErrorNone);
static_assert(static_cast<std::int64_t>(MTLCommandBufferErrorInternal) ==
              detail::kNativeErrorInternal);
static_assert(static_cast<std::int64_t>(MTLCommandBufferErrorTimeout) ==
              detail::kNativeErrorTimeout);
static_assert(static_cast<std::int64_t>(MTLCommandBufferErrorPageFault) ==
              detail::kNativeErrorPageFault);
static_assert(static_cast<std::int64_t>(MTLCommandBufferErrorAccessRevoked) ==
              detail::kNativeErrorAccessRevoked);
static_assert(static_cast<std::int64_t>(MTLCommandBufferErrorNotPermitted) ==
              detail::kNativeErrorNotPermitted);
static_assert(static_cast<std::int64_t>(MTLCommandBufferErrorOutOfMemory) ==
              detail::kNativeErrorOutOfMemory);
static_assert(static_cast<std::int64_t>(MTLCommandBufferErrorInvalidResource) ==
              detail::kNativeErrorInvalidResource);
static_assert(static_cast<std::int64_t>(MTLCommandBufferErrorMemoryless) ==
              detail::kNativeErrorMemoryless);
static_assert(static_cast<std::int64_t>(MTLCommandBufferErrorDeviceRemoved) ==
              detail::kNativeErrorDeviceRemoved);
static_assert(static_cast<std::int64_t>(MTLCommandBufferErrorStackOverflow) ==
              detail::kNativeErrorStackOverflow);

bool exceeds_native_extent(MetalSize size) noexcept {
    if constexpr (sizeof(NSUInteger) < sizeof(std::uint64_t)) {
        const auto maximum = std::numeric_limits<NSUInteger>::max();
        return size.width > maximum || size.height > maximum || size.depth > maximum;
    }
    return false;
}

MTLSize native_size(MetalSize size) noexcept {
    return MTLSizeMake(static_cast<NSUInteger>(size.width), static_cast<NSUInteger>(size.height),
                       static_cast<NSUInteger>(size.depth));
}

bool valid_indirect_range(std::uint32_t maximum_command_count,
                          std::uint32_t first_command,
                          std::uint32_t command_count) noexcept {
    return command_count != 0 && first_command < maximum_command_count &&
           command_count <= maximum_command_count - first_command;
}

id<MTLIndirectComputeCommand>
indirect_compute_command(void* indirect_command_buffer,
                         std::uint32_t command_index) noexcept {
    id<MTLIndirectCommandBuffer> native_buffer =
        (__bridge id<MTLIndirectCommandBuffer>)indirect_command_buffer;
    return [native_buffer
        indirectComputeCommandAtIndex:
            static_cast<NSUInteger>(command_index)];
}

// Shared body of the two waits: the completed buffer is the only place the
// status, the failure text, and the timestamps can all be read from.
MetalTimedExecutionResult complete_and_report(id<MTLCommandBuffer> native_buffer) {
    [native_buffer waitUntilCompleted];
    const MTLCommandBufferStatus native_status = native_buffer.status;
    const MetalExecutionState state =
        detail::execution_state_from_native(static_cast<std::uint64_t>(native_status));
    const MetalExecutionTiming timing{
        .schedule_start_seconds = native_buffer.kernelStartTime,
        .schedule_end_seconds = native_buffer.kernelEndTime,
        .gpu_start_seconds = native_buffer.GPUStartTime,
        .gpu_end_seconds = native_buffer.GPUEndTime,
    };
    if (native_status == MTLCommandBufferStatusCompleted) {
        return {
            .error = MetalCommandError::None,
            .completed = true,
            .state = state,
            .failure = MetalExecutionFailure::None,
            .native_error_code = 0,
            .has_native_error_code = false,
            .failure_description = {},
            .timing = timing,
        };
    }
    NSError* error = native_buffer.error;
    const char* raw = error == nil ? nullptr : error.localizedDescription.UTF8String;
    const bool is_metal_error_domain =
        error != nil && [error.domain isEqualToString:MTLCommandBufferErrorDomain];
    const std::int64_t native_error_code = error == nil ? 0 : static_cast<std::int64_t>(error.code);
    return {
        .error = MetalCommandError::None,
        .completed = false,
        .state = state,
        .failure =
            detail::execution_failure_from_native(state, is_metal_error_domain, native_error_code),
        .native_error_code = native_error_code,
        .has_native_error_code = error != nil,
        .failure_description = detail::bounded_execution_diagnostic(raw),
        .timing = timing,
    };
}

} // namespace

MetalStageComputePassResult::operator bool() const noexcept {
    return error == CounterStageSampleError::None && compute_pass.has_value();
}

MetalCommandBuffer::MetalCommandBuffer() noexcept = default;
MetalCommandBuffer::~MetalCommandBuffer() {
    reset();
}

MetalCommandBuffer::MetalCommandBuffer(MetalCommandBuffer&& other) noexcept
    : object_(std::exchange(other.object_, nullptr)) {}

MetalCommandBuffer& MetalCommandBuffer::operator=(MetalCommandBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        object_ = std::exchange(other.object_, nullptr);
    }
    return *this;
}

MetalCommandBuffer::MetalCommandBuffer(void* object) noexcept : object_(object) {}

MetalCommandBuffer::operator bool() const noexcept {
    return object_ != nullptr;
}

void MetalCommandBuffer::reset() noexcept {
    if (void* object = std::exchange(object_, nullptr); object != nullptr) {
        release_object(object);
    }
}

MetalIndirectCommandBuffer::MetalIndirectCommandBuffer() noexcept = default;
MetalIndirectCommandBuffer::~MetalIndirectCommandBuffer() {
    reset();
}

MetalIndirectCommandBuffer::MetalIndirectCommandBuffer(
    MetalIndirectCommandBuffer&& other) noexcept
    : object_(std::exchange(other.object_, nullptr)),
      maximum_command_count_(
          std::exchange(other.maximum_command_count_, 0)),
      maximum_kernel_buffer_bind_count_(
          std::exchange(other.maximum_kernel_buffer_bind_count_, 0)) {}

MetalIndirectCommandBuffer&
MetalIndirectCommandBuffer::operator=(
    MetalIndirectCommandBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        object_ = std::exchange(other.object_, nullptr);
        maximum_command_count_ =
            std::exchange(other.maximum_command_count_, 0);
        maximum_kernel_buffer_bind_count_ =
            std::exchange(other.maximum_kernel_buffer_bind_count_, 0);
    }
    return *this;
}

MetalIndirectCommandBuffer::MetalIndirectCommandBuffer(
    void* object, std::uint32_t maximum_command_count,
    std::uint32_t maximum_kernel_buffer_bind_count) noexcept
    : object_(object), maximum_command_count_(maximum_command_count),
      maximum_kernel_buffer_bind_count_(
          maximum_kernel_buffer_bind_count) {}

std::uint32_t
MetalIndirectCommandBuffer::maximum_command_count() const noexcept {
    return maximum_command_count_;
}

std::uint32_t MetalIndirectCommandBuffer::
    maximum_kernel_buffer_bind_count() const noexcept {
    return maximum_kernel_buffer_bind_count_;
}

MetalIndirectCommandBuffer::operator bool() const noexcept {
    return object_ != nullptr && maximum_command_count_ != 0 &&
           maximum_kernel_buffer_bind_count_ != 0;
}

void MetalIndirectCommandBuffer::reset() noexcept {
    maximum_command_count_ = 0;
    maximum_kernel_buffer_bind_count_ = 0;
    if (void* object = std::exchange(object_, nullptr);
        object != nullptr) {
        release_object(object);
    }
}

MetalComputePass::MetalComputePass() noexcept = default;
MetalComputePass::~MetalComputePass() {
    reset();
}

MetalComputePass::MetalComputePass(MetalComputePass&& other) noexcept
    : command_buffer_(std::exchange(other.command_buffer_, nullptr)),
      encoder_(std::exchange(other.encoder_, nullptr)) {}

MetalComputePass& MetalComputePass::operator=(MetalComputePass&& other) noexcept {
    if (this != &other) {
        reset();
        command_buffer_ = std::exchange(other.command_buffer_, nullptr);
        encoder_ = std::exchange(other.encoder_, nullptr);
    }
    return *this;
}

MetalComputePass::MetalComputePass(void* command_buffer, void* encoder) noexcept
    : command_buffer_(command_buffer), encoder_(encoder) {}

MetalComputePass::operator bool() const noexcept {
    return command_buffer_ != nullptr && encoder_ != nullptr;
}

void MetalComputePass::reset() noexcept {
    void* encoder = std::exchange(encoder_, nullptr);
    void* command_buffer = std::exchange(command_buffer_, nullptr);
    if (encoder != nullptr) {
        id<MTLComputeCommandEncoder> native_encoder =
            (__bridge id<MTLComputeCommandEncoder>)encoder;
        [native_encoder endEncoding];
        release_object(encoder);
    }
    if (command_buffer != nullptr) {
        release_object(command_buffer);
    }
}

MetalBlitPass::MetalBlitPass() noexcept = default;
MetalBlitPass::~MetalBlitPass() {
    reset();
}

MetalBlitPass::MetalBlitPass(MetalBlitPass&& other) noexcept
    : command_buffer_(std::exchange(other.command_buffer_, nullptr)),
      encoder_(std::exchange(other.encoder_, nullptr)) {}

MetalBlitPass& MetalBlitPass::operator=(MetalBlitPass&& other) noexcept {
    if (this != &other) {
        reset();
        command_buffer_ = std::exchange(other.command_buffer_, nullptr);
        encoder_ = std::exchange(other.encoder_, nullptr);
    }
    return *this;
}

MetalBlitPass::MetalBlitPass(void* command_buffer, void* encoder) noexcept
    : command_buffer_(command_buffer), encoder_(encoder) {}

MetalBlitPass::operator bool() const noexcept {
    return command_buffer_ != nullptr && encoder_ != nullptr;
}

void MetalBlitPass::reset() noexcept {
    void* encoder = std::exchange(encoder_, nullptr);
    void* command_buffer = std::exchange(command_buffer_, nullptr);
    if (encoder != nullptr) {
        id<MTLBlitCommandEncoder> native_encoder = (__bridge id<MTLBlitCommandEncoder>)encoder;
        [native_encoder endEncoding];
        release_object(encoder);
    }
    if (command_buffer != nullptr) {
        release_object(command_buffer);
    }
}

MetalPendingExecution::MetalPendingExecution() noexcept = default;
MetalPendingExecution::~MetalPendingExecution() {
    reset();
}

MetalPendingExecution::MetalPendingExecution(MetalPendingExecution&& other) noexcept
    : object_(std::exchange(other.object_, nullptr)) {}

MetalPendingExecution& MetalPendingExecution::operator=(MetalPendingExecution&& other) noexcept {
    if (this != &other) {
        reset();
        object_ = std::exchange(other.object_, nullptr);
    }
    return *this;
}

MetalPendingExecution::MetalPendingExecution(void* object) noexcept : object_(object) {}

MetalPendingExecution::operator bool() const noexcept {
    return object_ != nullptr;
}

void MetalPendingExecution::reset() noexcept {
    if (void* object = std::exchange(object_, nullptr); object != nullptr) {
        release_object(object);
    }
}

MetalCommandBufferResult create_command_buffer(const MetalCommandQueue& command_queue) {
    if (!command_queue) {
        return {.error = MetalCommandError::InvalidCommandQueue, .command_buffer = std::nullopt};
    }
    @autoreleasepool {
        id<MTLCommandQueue> native_queue =
            (__bridge id<MTLCommandQueue>)command_queue.storage_->object.get();
        id<MTLCommandBuffer> command_buffer = [native_queue commandBuffer];
        if (command_buffer == nil) {
            return {
                .error = MetalCommandError::CommandBufferCreationFailed,
                .command_buffer = std::nullopt,
            };
        }
        void* retained = (__bridge_retained void*)command_buffer;
        return {
            .error = MetalCommandError::None,
            .command_buffer = MetalCommandBuffer(retained),
        };
    }
}

MetalCommandError encode_wait_for_event(
    MetalCommandBuffer& command_buffer, const MetalEvent& event,
    std::uint64_t value) {
    if (!command_buffer) {
        return MetalCommandError::InvalidCommandBuffer;
    }
    if (!event) {
        return MetalCommandError::InvalidEvent;
    }
    if (value == 0) {
        return MetalCommandError::InvalidEventValue;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> native_buffer =
            (__bridge id<MTLCommandBuffer>)command_buffer.object_;
        id<MTLEvent> native_event =
            (__bridge id<MTLEvent>)event.storage_->object.get();
        [native_buffer encodeWaitForEvent:native_event value:value];
    }
    return MetalCommandError::None;
}

MetalCommandError encode_signal_event(
    MetalCommandBuffer& command_buffer, const MetalEvent& event,
    std::uint64_t value) {
    if (!command_buffer) {
        return MetalCommandError::InvalidCommandBuffer;
    }
    if (!event) {
        return MetalCommandError::InvalidEvent;
    }
    if (value == 0) {
        return MetalCommandError::InvalidEventValue;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> native_buffer =
            (__bridge id<MTLCommandBuffer>)command_buffer.object_;
        id<MTLEvent> native_event =
            (__bridge id<MTLEvent>)event.storage_->object.get();
        [native_buffer encodeSignalEvent:native_event value:value];
    }
    return MetalCommandError::None;
}

MetalComputePassResult begin_compute_pass(MetalCommandBuffer&& command_buffer) {
    if (!command_buffer) {
        return {.error = MetalCommandError::InvalidCommandBuffer, .compute_pass = std::nullopt};
    }
    void* command_buffer_object = std::exchange(command_buffer.object_, nullptr);
    @autoreleasepool {
        id<MTLCommandBuffer> native_buffer = (__bridge id<MTLCommandBuffer>)command_buffer_object;
        id<MTLComputeCommandEncoder> encoder = [native_buffer computeCommandEncoder];
        if (encoder == nil) {
            release_object(command_buffer_object);
            return {.error = MetalCommandError::EncoderCreationFailed,
                    .compute_pass = std::nullopt};
        }
        void* retained_encoder = (__bridge_retained void*)encoder;
        return {
            .error = MetalCommandError::None,
            .compute_pass = MetalComputePass(command_buffer_object, retained_encoder),
        };
    }
}

MetalIndirectCommandBufferResult create_compute_indirect_command_buffer(
    const MetalDevice& device, std::uint32_t maximum_command_count,
    std::uint32_t maximum_kernel_buffer_bind_count) {
    if (!device) {
        return {
            .error = MetalCommandError::InvalidDevice,
            .indirect_command_buffer = std::nullopt,
        };
    }
    if (maximum_command_count == 0) {
        return {
            .error = MetalCommandError::InvalidIndirectCommandCount,
            .indirect_command_buffer = std::nullopt,
        };
    }
    if (maximum_kernel_buffer_bind_count == 0 ||
        maximum_kernel_buffer_bind_count >
            kMaxBufferArgumentIndex + 1u) {
        return {
            .error = MetalCommandError::InvalidBufferIndex,
            .indirect_command_buffer = std::nullopt,
        };
    }
    @autoreleasepool {
        id<MTLDevice> native_device =
            (__bridge id<MTLDevice>)device.storage_->object.get();
        MTLIndirectCommandBufferDescriptor* descriptor =
            [MTLIndirectCommandBufferDescriptor new];
        descriptor.commandTypes =
            MTLIndirectCommandTypeConcurrentDispatch;
        descriptor.inheritPipelineState = NO;
        descriptor.inheritBuffers = NO;
        descriptor.maxKernelBufferBindCount =
            static_cast<NSUInteger>(
                maximum_kernel_buffer_bind_count);
        id<MTLIndirectCommandBuffer> indirect_command_buffer =
            [native_device
                newIndirectCommandBufferWithDescriptor:descriptor
                                      maxCommandCount:
                                          static_cast<NSUInteger>(
                                              maximum_command_count)
                                               options:
                                                   MTLResourceStorageModeShared];
        if (indirect_command_buffer == nil) {
            return {
                .error =
                    MetalCommandError::
                        IndirectCommandBufferCreationFailed,
                .indirect_command_buffer = std::nullopt,
            };
        }
        void* retained =
            (__bridge_retained void*)indirect_command_buffer;
        return {
            .error = MetalCommandError::None,
            .indirect_command_buffer = MetalIndirectCommandBuffer(
                retained, maximum_command_count,
                maximum_kernel_buffer_bind_count),
        };
    }
}

MetalStageComputePassResult
begin_stage_sampled_compute_pass(MetalCommandBuffer&& command_buffer,
                                 const MetalCounterSampleBuffer& samples,
                                 CounterSamplePair pair) noexcept {
    if (!command_buffer) {
        return {
            .error = CounterStageSampleError::InvalidCommandBuffer,
            .compute_pass = std::nullopt,
        };
    }
    if (!samples) {
        return {
            .error = CounterStageSampleError::InvalidBuffer,
            .compute_pass = std::nullopt,
        };
    }
    if (samples.mode_ != CounterSamplingMode::StageBoundaryEncoderSplit) {
        return {
            .error = CounterStageSampleError::SamplingModeMismatch,
            .compute_pass = std::nullopt,
        };
    }
    const CounterSampleError pair_error =
        validate_counter_sample_pair(samples.sample_capacity_, pair);
    if (pair_error != CounterSampleError::None || (pair.start & 1U) != 0U) {
        return {
            .error = pair_error == CounterSampleError::SampleIndexOutOfBounds
                         ? CounterStageSampleError::SampleIndexOutOfBounds
                         : CounterStageSampleError::InvalidSamplePair,
            .compute_pass = std::nullopt,
        };
    }

    void* command_buffer_object = std::exchange(command_buffer.object_, nullptr);
    @autoreleasepool {
        id<MTLCommandBuffer> native_buffer =
            (__bridge id<MTLCommandBuffer>)command_buffer_object;
        NSArray<MTLComputePassDescriptor*>* descriptors =
            (__bridge NSArray<MTLComputePassDescriptor*>*)samples.stage_descriptor_;
        MTLComputePassDescriptor* descriptor =
            descriptors[static_cast<NSUInteger>(pair.start / 2U)];
        id<MTLComputeCommandEncoder> encoder =
            [native_buffer computeCommandEncoderWithDescriptor:descriptor];
        if (encoder == nil) {
            release_object(command_buffer_object);
            return {
                .error = CounterStageSampleError::EncoderCreationFailed,
                .compute_pass = std::nullopt,
            };
        }
        void* retained_encoder = (__bridge_retained void*)encoder;
        return {
            .error = CounterStageSampleError::None,
            .compute_pass = MetalComputePass(command_buffer_object, retained_encoder),
        };
    }
}

CounterStageSampleError
split_stage_sampled_compute_pass(MetalComputePass& compute_pass,
                                 const MetalCounterSampleBuffer& samples,
                                 CounterSamplePair pair) noexcept {
    if (!compute_pass) {
        return CounterStageSampleError::InvalidComputePass;
    }
    if (!samples) {
        return CounterStageSampleError::InvalidBuffer;
    }
    if (samples.mode_ != CounterSamplingMode::StageBoundaryEncoderSplit) {
        return CounterStageSampleError::SamplingModeMismatch;
    }
    const CounterSampleError pair_error =
        validate_counter_sample_pair(samples.sample_capacity_, pair);
    if (pair_error != CounterSampleError::None || (pair.start & 1U) != 0U) {
        return pair_error == CounterSampleError::SampleIndexOutOfBounds
                   ? CounterStageSampleError::SampleIndexOutOfBounds
                   : CounterStageSampleError::InvalidSamplePair;
    }

    void* old_encoder = std::exchange(compute_pass.encoder_, nullptr);
    id<MTLComputeCommandEncoder> native_old =
        (__bridge id<MTLComputeCommandEncoder>)old_encoder;
    [native_old endEncoding];
    release_object(old_encoder);

    @autoreleasepool {
        id<MTLCommandBuffer> native_buffer =
            (__bridge id<MTLCommandBuffer>)compute_pass.command_buffer_;
        NSArray<MTLComputePassDescriptor*>* descriptors =
            (__bridge NSArray<MTLComputePassDescriptor*>*)samples.stage_descriptor_;
        MTLComputePassDescriptor* descriptor =
            descriptors[static_cast<NSUInteger>(pair.start / 2U)];
        id<MTLComputeCommandEncoder> encoder =
            [native_buffer computeCommandEncoderWithDescriptor:descriptor];
        if (encoder == nil) {
            return CounterStageSampleError::EncoderCreationFailed;
        }
        compute_pass.encoder_ = (__bridge_retained void*)encoder;
    }
    return CounterStageSampleError::None;
}

MetalBlitPassResult begin_blit_pass(MetalCommandBuffer&& command_buffer) {
    if (!command_buffer) {
        return {.error = MetalCommandError::InvalidCommandBuffer, .blit_pass = std::nullopt};
    }
    void* command_buffer_object = std::exchange(command_buffer.object_, nullptr);
    @autoreleasepool {
        id<MTLCommandBuffer> native_buffer = (__bridge id<MTLCommandBuffer>)command_buffer_object;
        id<MTLBlitCommandEncoder> encoder = [native_buffer blitCommandEncoder];
        if (encoder == nil) {
            release_object(command_buffer_object);
            return {.error = MetalCommandError::EncoderCreationFailed, .blit_pass = std::nullopt};
        }
        void* retained_encoder = (__bridge_retained void*)encoder;
        return {
            .error = MetalCommandError::None,
            .blit_pass = MetalBlitPass(command_buffer_object, retained_encoder),
        };
    }
}

MetalCommandError set_compute_pipeline(MetalComputePass& compute_pass,
                                       const MetalComputePipeline& pipeline) {
    if (!compute_pass) {
        return MetalCommandError::InvalidComputePass;
    }
    if (!pipeline) {
        return MetalCommandError::InvalidPipeline;
    }
    id<MTLComputeCommandEncoder> native_encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_pass.encoder_;
    id<MTLComputePipelineState> native_pipeline =
        (__bridge id<MTLComputePipelineState>)pipeline.storage_->object.get();
    [native_encoder setComputePipelineState:native_pipeline];
    return MetalCommandError::None;
}

MetalCommandError set_buffer(MetalComputePass& compute_pass, const MetalBuffer& buffer,
                             std::uint64_t offset_bytes, std::uint32_t index) {
    if (!compute_pass) {
        return MetalCommandError::InvalidComputePass;
    }
    if (!buffer) {
        return MetalCommandError::InvalidBuffer;
    }
    if (index > kMaxBufferArgumentIndex) {
        return MetalCommandError::InvalidBufferIndex;
    }
    if (offset_bytes >= buffer.size_bytes()) {
        return MetalCommandError::InvalidBufferOffset;
    }
    id<MTLComputeCommandEncoder> native_encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_pass.encoder_;
    id<MTLBuffer> native_buffer = (__bridge id<MTLBuffer>)buffer.storage_->object.get();
    [native_encoder setBuffer:native_buffer
                       offset:static_cast<NSUInteger>(offset_bytes)
                      atIndex:index];
    return MetalCommandError::None;
}

MetalCommandError dispatch_threadgroups(MetalComputePass& compute_pass, MetalSize threadgroups,
                                        MetalSize threads_per_threadgroup) {
    if (!compute_pass) {
        return MetalCommandError::InvalidComputePass;
    }
    const bool zero_extent = threadgroups.width == 0 || threadgroups.height == 0 ||
                             threadgroups.depth == 0 || threads_per_threadgroup.width == 0 ||
                             threads_per_threadgroup.height == 0 ||
                             threads_per_threadgroup.depth == 0;
    if (zero_extent || exceeds_native_extent(threadgroups) ||
        exceeds_native_extent(threads_per_threadgroup)) {
        return MetalCommandError::InvalidDispatchExtent;
    }
    id<MTLComputeCommandEncoder> native_encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_pass.encoder_;
    [native_encoder dispatchThreadgroups:native_size(threadgroups)
                   threadsPerThreadgroup:native_size(threads_per_threadgroup)];
    return MetalCommandError::None;
}

MetalCommandError dispatch_threadgroups_indirect(MetalComputePass& compute_pass,
                                                 const MetalBuffer& arguments,
                                                 std::uint64_t argument_offset_bytes,
                                                 MetalSize threads_per_threadgroup) {
    if (!compute_pass) {
        return MetalCommandError::InvalidComputePass;
    }
    if (!arguments) {
        return MetalCommandError::InvalidBuffer;
    }
    constexpr std::uint64_t kArgumentBytes = 3u * sizeof(std::uint32_t);
    if (argument_offset_bytes > arguments.size_bytes() ||
        arguments.size_bytes() - argument_offset_bytes < kArgumentBytes) {
        return MetalCommandError::InvalidBufferOffset;
    }
    const bool zero_extent = threads_per_threadgroup.width == 0 ||
                             threads_per_threadgroup.height == 0 ||
                             threads_per_threadgroup.depth == 0;
    if (zero_extent || exceeds_native_extent(threads_per_threadgroup)) {
        return MetalCommandError::InvalidDispatchExtent;
    }
    id<MTLComputeCommandEncoder> native_encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_pass.encoder_;
    id<MTLBuffer> native_arguments = (__bridge id<MTLBuffer>)arguments.storage_->object.get();
    [native_encoder
        dispatchThreadgroupsWithIndirectBuffer:native_arguments
                          indirectBufferOffset:static_cast<NSUInteger>(argument_offset_bytes)
                         threadsPerThreadgroup:native_size(threads_per_threadgroup)];
    return MetalCommandError::None;
}

MetalCommandError set_bytes(MetalComputePass& compute_pass, const void* bytes,
                            std::uint32_t length_bytes, std::uint32_t index) {
    if (!compute_pass) {
        return MetalCommandError::InvalidComputePass;
    }
    if (bytes == nullptr || length_bytes == 0 || length_bytes > kMaxInlineConstantBytes) {
        return MetalCommandError::InvalidBuffer;
    }
    if (index > kMaxBufferArgumentIndex) {
        return MetalCommandError::InvalidBufferIndex;
    }
    id<MTLComputeCommandEncoder> native_encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_pass.encoder_;
    [native_encoder setBytes:bytes length:length_bytes atIndex:index];
    return MetalCommandError::None;
}

MetalCommandError memory_barrier(MetalComputePass& compute_pass) {
    if (!compute_pass) {
        return MetalCommandError::InvalidComputePass;
    }
    id<MTLComputeCommandEncoder> native_encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_pass.encoder_;
    [native_encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
    return MetalCommandError::None;
}

MetalCommandError reset_indirect_commands(
    MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t first_command, std::uint32_t command_count) {
    if (!indirect_command_buffer) {
        return MetalCommandError::InvalidIndirectCommandBuffer;
    }
    if (!valid_indirect_range(
            indirect_command_buffer.maximum_command_count_, first_command,
            command_count)) {
        return MetalCommandError::InvalidIndirectCommandRange;
    }
    id<MTLIndirectCommandBuffer> native_buffer =
        (__bridge id<MTLIndirectCommandBuffer>)
            indirect_command_buffer.object_;
    [native_buffer
        resetWithRange:NSMakeRange(
                           static_cast<NSUInteger>(first_command),
                           static_cast<NSUInteger>(command_count))];
    return MetalCommandError::None;
}

MetalCommandError set_indirect_compute_pipeline(
    MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t command_index, const MetalComputePipeline& pipeline) {
    if (!indirect_command_buffer) {
        return MetalCommandError::InvalidIndirectCommandBuffer;
    }
    if (command_index >=
        indirect_command_buffer.maximum_command_count_) {
        return MetalCommandError::InvalidIndirectCommandIndex;
    }
    if (!pipeline || !supports_indirect_commands(pipeline)) {
        return MetalCommandError::InvalidPipeline;
    }
    id<MTLComputePipelineState> native_pipeline =
        (__bridge id<MTLComputePipelineState>)
            pipeline.storage_->object.get();
    id<MTLIndirectComputeCommand> command =
        indirect_compute_command(indirect_command_buffer.object_,
                                 command_index);
    if (command == nil) {
        return MetalCommandError::InvalidIndirectCommandBuffer;
    }
    [command setComputePipelineState:native_pipeline];
    return MetalCommandError::None;
}

MetalCommandError set_indirect_buffer(
    MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t command_index, const MetalBuffer& buffer,
    std::uint64_t offset_bytes, std::uint32_t buffer_index) {
    if (!indirect_command_buffer) {
        return MetalCommandError::InvalidIndirectCommandBuffer;
    }
    if (command_index >=
        indirect_command_buffer.maximum_command_count_) {
        return MetalCommandError::InvalidIndirectCommandIndex;
    }
    if (!buffer) {
        return MetalCommandError::InvalidBuffer;
    }
    if (buffer_index >=
        indirect_command_buffer.maximum_kernel_buffer_bind_count_) {
        return MetalCommandError::InvalidBufferIndex;
    }
    if (offset_bytes >= buffer.size_bytes()) {
        return MetalCommandError::InvalidBufferOffset;
    }
    if (offset_bytes >= kIndirectKernelBufferOffsetLimitBytes) {
        return MetalCommandError::InvalidBufferOffset;
    }
    id<MTLBuffer> native_buffer =
        (__bridge id<MTLBuffer>)buffer.storage_->object.get();
    id<MTLIndirectComputeCommand> command =
        indirect_compute_command(indirect_command_buffer.object_,
                                 command_index);
    if (command == nil) {
        return MetalCommandError::InvalidIndirectCommandBuffer;
    }
    [command setKernelBuffer:native_buffer
                      offset:static_cast<NSUInteger>(offset_bytes)
                     atIndex:buffer_index];
    return MetalCommandError::None;
}

MetalCommandError set_indirect_barrier(
    MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t command_index) {
    if (!indirect_command_buffer) {
        return MetalCommandError::InvalidIndirectCommandBuffer;
    }
    if (command_index >=
        indirect_command_buffer.maximum_command_count_) {
        return MetalCommandError::InvalidIndirectCommandIndex;
    }
    id<MTLIndirectComputeCommand> command =
        indirect_compute_command(indirect_command_buffer.object_,
                                 command_index);
    if (command == nil) {
        return MetalCommandError::InvalidIndirectCommandBuffer;
    }
    [command setBarrier];
    return MetalCommandError::None;
}

MetalCommandError clear_indirect_barrier(
    MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t command_index) {
    if (!indirect_command_buffer) {
        return MetalCommandError::InvalidIndirectCommandBuffer;
    }
    if (command_index >=
        indirect_command_buffer.maximum_command_count_) {
        return MetalCommandError::InvalidIndirectCommandIndex;
    }
    id<MTLIndirectComputeCommand> command =
        indirect_compute_command(indirect_command_buffer.object_,
                                 command_index);
    if (command == nil) {
        return MetalCommandError::InvalidIndirectCommandBuffer;
    }
    [command clearBarrier];
    return MetalCommandError::None;
}

MetalCommandError dispatch_indirect_threadgroups(
    MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t command_index, MetalSize threadgroups,
    MetalSize threads_per_threadgroup) {
    if (!indirect_command_buffer) {
        return MetalCommandError::InvalidIndirectCommandBuffer;
    }
    if (command_index >=
        indirect_command_buffer.maximum_command_count_) {
        return MetalCommandError::InvalidIndirectCommandIndex;
    }
    const bool zero_extent =
        threadgroups.width == 0 || threadgroups.height == 0 ||
        threadgroups.depth == 0 ||
        threads_per_threadgroup.width == 0 ||
        threads_per_threadgroup.height == 0 ||
        threads_per_threadgroup.depth == 0;
    if (zero_extent || exceeds_native_extent(threadgroups) ||
        exceeds_native_extent(threads_per_threadgroup)) {
        return MetalCommandError::InvalidDispatchExtent;
    }
    id<MTLIndirectComputeCommand> command =
        indirect_compute_command(indirect_command_buffer.object_,
                                 command_index);
    if (command == nil) {
        return MetalCommandError::InvalidIndirectCommandBuffer;
    }
    [command concurrentDispatchThreadgroups:native_size(threadgroups)
                      threadsPerThreadgroup:
                          native_size(threads_per_threadgroup)];
    return MetalCommandError::None;
}

MetalCommandError use_buffer_resource(MetalComputePass& compute_pass,
                                      const MetalBuffer& buffer,
                                      MetalResourceUsage usage) {
    if (!compute_pass) {
        return MetalCommandError::InvalidComputePass;
    }
    if (!buffer) {
        return MetalCommandError::InvalidBuffer;
    }
    MTLResourceUsage native_usage;
    switch (usage) {
    case MetalResourceUsage::Read:
        native_usage = MTLResourceUsageRead;
        break;
    case MetalResourceUsage::Write:
        native_usage = MTLResourceUsageWrite;
        break;
    case MetalResourceUsage::ReadWrite:
        native_usage =
            MTLResourceUsageRead | MTLResourceUsageWrite;
        break;
    default:
        return MetalCommandError::InvalidResourceUsage;
    }
    id<MTLComputeCommandEncoder> native_encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_pass.encoder_;
    id<MTLBuffer> native_buffer =
        (__bridge id<MTLBuffer>)buffer.storage_->object.get();
    [native_encoder useResource:native_buffer usage:native_usage];
    return MetalCommandError::None;
}

MetalCommandError execute_indirect_commands(
    MetalComputePass& compute_pass,
    const MetalIndirectCommandBuffer& indirect_command_buffer,
    std::uint32_t first_command, std::uint32_t command_count) {
    if (!compute_pass) {
        return MetalCommandError::InvalidComputePass;
    }
    if (!indirect_command_buffer) {
        return MetalCommandError::InvalidIndirectCommandBuffer;
    }
    if (!valid_indirect_range(
            indirect_command_buffer.maximum_command_count_, first_command,
            command_count)) {
        return MetalCommandError::InvalidIndirectCommandRange;
    }
    id<MTLComputeCommandEncoder> native_encoder =
        (__bridge id<MTLComputeCommandEncoder>)compute_pass.encoder_;
    id<MTLIndirectCommandBuffer> native_buffer =
        (__bridge id<MTLIndirectCommandBuffer>)
            indirect_command_buffer.object_;
    [native_encoder
        executeCommandsInBuffer:native_buffer
                      withRange:
                          NSMakeRange(
                              static_cast<NSUInteger>(first_command),
                              static_cast<NSUInteger>(command_count))];
    return MetalCommandError::None;
}

std::uint64_t metal_buffer_identity(const MetalBuffer& buffer) noexcept {
    if (!buffer) {
        return 0;
    }
    id<MTLBuffer> native_buffer =
        (__bridge id<MTLBuffer>)buffer.storage_->object.get();
    return static_cast<std::uint64_t>(
        reinterpret_cast<std::uintptr_t>(
            (__bridge void*)native_buffer));
}

std::uint64_t indirect_command_buffer_identity(
    const MetalIndirectCommandBuffer& indirect_command_buffer) noexcept {
    return indirect_command_buffer
               ? static_cast<std::uint64_t>(
                     reinterpret_cast<std::uintptr_t>(
                         indirect_command_buffer.object_))
               : 0;
}

MetalCommandBufferResult end_compute_pass(MetalComputePass&& compute_pass) {
    if (!compute_pass) {
        return {.error = MetalCommandError::InvalidComputePass, .command_buffer = std::nullopt};
    }
    void* encoder = std::exchange(compute_pass.encoder_, nullptr);
    void* command_buffer = std::exchange(compute_pass.command_buffer_, nullptr);
    id<MTLComputeCommandEncoder> native_encoder = (__bridge id<MTLComputeCommandEncoder>)encoder;
    [native_encoder endEncoding];
    release_object(encoder);
    return {
        .error = MetalCommandError::None,
        .command_buffer = MetalCommandBuffer(command_buffer),
    };
}

MetalCommandError copy_buffer(MetalBlitPass& blit_pass, const MetalBuffer& source,
                              std::uint64_t source_offset_bytes, MetalBuffer& destination,
                              std::uint64_t destination_offset_bytes, std::uint64_t length_bytes) {
    if (!blit_pass) {
        return MetalCommandError::InvalidBlitPass;
    }
    if (!source || !destination) {
        return MetalCommandError::InvalidBuffer;
    }
    if (length_bytes == 0) {
        return MetalCommandError::InvalidCopyExtent;
    }
    if (source_offset_bytes > source.size_bytes() ||
        length_bytes > source.size_bytes() - source_offset_bytes) {
        return MetalCommandError::InvalidSourceRange;
    }
    if (destination_offset_bytes > destination.size_bytes() ||
        length_bytes > destination.size_bytes() - destination_offset_bytes) {
        return MetalCommandError::InvalidDestinationRange;
    }
    if constexpr (sizeof(NSUInteger) < sizeof(std::uint64_t)) {
        const auto maximum = std::numeric_limits<NSUInteger>::max();
        if (source_offset_bytes > maximum || destination_offset_bytes > maximum ||
            length_bytes > maximum) {
            return MetalCommandError::InvalidCopyExtent;
        }
    }
    id<MTLBlitCommandEncoder> native_encoder =
        (__bridge id<MTLBlitCommandEncoder>)blit_pass.encoder_;
    id<MTLBuffer> native_source = (__bridge id<MTLBuffer>)source.storage_->object.get();
    id<MTLBuffer> native_destination = (__bridge id<MTLBuffer>)destination.storage_->object.get();
    [native_encoder copyFromBuffer:native_source
                      sourceOffset:static_cast<NSUInteger>(source_offset_bytes)
                          toBuffer:native_destination
                 destinationOffset:static_cast<NSUInteger>(destination_offset_bytes)
                              size:static_cast<NSUInteger>(length_bytes)];
    return MetalCommandError::None;
}

MetalCommandError fill_buffer(MetalBlitPass& blit_pass, MetalBuffer& destination,
                              std::uint64_t destination_offset_bytes, std::uint64_t length_bytes,
                              std::byte value) {
    if (!blit_pass) {
        return MetalCommandError::InvalidBlitPass;
    }
    if (!destination) {
        return MetalCommandError::InvalidBuffer;
    }
    if (length_bytes == 0) {
        return MetalCommandError::InvalidFillExtent;
    }
    if (destination_offset_bytes > destination.size_bytes() ||
        length_bytes > destination.size_bytes() - destination_offset_bytes) {
        return MetalCommandError::InvalidDestinationRange;
    }
    if constexpr (sizeof(NSUInteger) < sizeof(std::uint64_t)) {
        const auto maximum = std::numeric_limits<NSUInteger>::max();
        if (destination_offset_bytes > maximum || length_bytes > maximum) {
            return MetalCommandError::InvalidFillExtent;
        }
    }
    id<MTLBlitCommandEncoder> native_encoder =
        (__bridge id<MTLBlitCommandEncoder>)blit_pass.encoder_;
    id<MTLBuffer> native_destination = (__bridge id<MTLBuffer>)destination.storage_->object.get();
    const NSRange range = NSMakeRange(static_cast<NSUInteger>(destination_offset_bytes),
                                      static_cast<NSUInteger>(length_bytes));
    [native_encoder fillBuffer:native_destination
                         range:range
                         value:static_cast<std::uint8_t>(value)];
    return MetalCommandError::None;
}

MetalCommandBufferResult end_blit_pass(MetalBlitPass&& blit_pass) {
    if (!blit_pass) {
        return {.error = MetalCommandError::InvalidBlitPass, .command_buffer = std::nullopt};
    }
    void* encoder = std::exchange(blit_pass.encoder_, nullptr);
    void* command_buffer = std::exchange(blit_pass.command_buffer_, nullptr);
    id<MTLBlitCommandEncoder> native_encoder = (__bridge id<MTLBlitCommandEncoder>)encoder;
    [native_encoder endEncoding];
    release_object(encoder);
    return {
        .error = MetalCommandError::None,
        .command_buffer = MetalCommandBuffer(command_buffer),
    };
}

MetalPendingExecutionResult commit(MetalCommandBuffer&& command_buffer) {
    if (!command_buffer) {
        return {
            .error = MetalCommandError::InvalidCommandBuffer,
            .pending_execution = std::nullopt,
        };
    }
    MetalPendingExecution pending(std::exchange(command_buffer.object_, nullptr));
    id<MTLCommandBuffer> native_buffer = (__bridge id<MTLCommandBuffer>)pending.object_;
    [native_buffer commit];
    return {
        .error = MetalCommandError::None,
        .pending_execution = std::move(pending),
    };
}

MetalExecutionResult wait_until_completed(MetalPendingExecution&& pending_execution) {
    if (!pending_execution) {
        return {
            .error = MetalCommandError::InvalidPendingExecution,
            .completed = false,
            .state = MetalExecutionState::NotObserved,
            .failure = MetalExecutionFailure::None,
            .native_error_code = 0,
            .has_native_error_code = false,
            .failure_description = {},
        };
    }
    @autoreleasepool {
        MetalPendingExecution observed(std::exchange(pending_execution.object_, nullptr));
        MetalTimedExecutionResult result =
            complete_and_report((__bridge id<MTLCommandBuffer>)observed.object_);
        return {
            .error = result.error,
            .completed = result.completed,
            .state = result.state,
            .failure = result.failure,
            .native_error_code = result.native_error_code,
            .has_native_error_code = result.has_native_error_code,
            .failure_description = std::move(result.failure_description),
        };
    }
}

MetalTimedExecutionResult wait_until_completed_timed(MetalPendingExecution&& pending_execution) {
    if (!pending_execution) {
        return {
            .error = MetalCommandError::InvalidPendingExecution,
            .completed = false,
            .state = MetalExecutionState::NotObserved,
            .failure = MetalExecutionFailure::None,
            .native_error_code = 0,
            .has_native_error_code = false,
            .failure_description = {},
            .timing = {},
        };
    }
    @autoreleasepool {
        MetalPendingExecution observed(std::exchange(pending_execution.object_, nullptr));
        return complete_and_report((__bridge id<MTLCommandBuffer>)observed.object_);
    }
}

} // namespace tatara::backend::metal
