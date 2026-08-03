#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace tatara::backend::metal {

struct MetalDeviceResult;
struct MetalCommandQueueResult;
struct MetalEventResult;
struct MetalBufferResult;
struct MetalPooledBufferResult;
struct MetalLibraryResult;
struct MetalComputePipelineResult;
struct MetalCommandBufferResult;
struct MetalIndirectCommandBufferResult;
struct CounterSampleBufferCreateResult;
struct MetalSize;
class MetalCommandBuffer;
class MetalFunction;
class MetalComputePass;
class MetalIndirectCommandBuffer;
class MetalBlitPass;
enum class MetalCommandError : std::uint8_t;
enum class MetalResourceUsage : std::uint8_t;

class MetalDevice {
  public:
    MetalDevice() noexcept;
    ~MetalDevice();

    MetalDevice(const MetalDevice&) = delete;
    MetalDevice& operator=(const MetalDevice&) = delete;
    MetalDevice(MetalDevice&&) noexcept;
    MetalDevice& operator=(MetalDevice&&) noexcept;

    const std::string& name() const noexcept;
    explicit operator bool() const noexcept;

  private:
    struct Storage;

    friend MetalDeviceResult create_system_device();
    friend MetalCommandQueueResult create_command_queue(const MetalDevice&);
    friend MetalEventResult create_event(const MetalDevice&);
    friend MetalBufferResult create_shared_buffer(const MetalDevice&, std::uint64_t);
    friend MetalPooledBufferResult create_striped_pool_buffer(const MetalDevice&, std::uint64_t,
                                                              std::uint32_t);
    friend MetalLibraryResult create_library_with_source(const MetalDevice&, std::string_view);
    friend MetalComputePipelineResult create_compute_pipeline(const MetalDevice&,
                                                              const MetalFunction&);
    friend MetalComputePipelineResult
    create_indirect_compute_pipeline(const MetalDevice&,
                                     const MetalFunction&);
    friend MetalIndirectCommandBufferResult
    create_compute_indirect_command_buffer(const MetalDevice&, std::uint32_t,
                                           std::uint32_t);
    friend std::uint64_t metal_device_identity(const MetalDevice&) noexcept;
    friend CounterSampleBufferCreateResult
    create_timestamp_counter_sample_buffer(const MetalDevice&, std::size_t) noexcept;

    MetalDevice(std::unique_ptr<Storage> storage, std::string name);

    std::unique_ptr<Storage> storage_;
    std::string name_;
};

class MetalCommandQueue {
  public:
    MetalCommandQueue() noexcept;
    ~MetalCommandQueue();

    MetalCommandQueue(const MetalCommandQueue&) = delete;
    MetalCommandQueue& operator=(const MetalCommandQueue&) = delete;
    MetalCommandQueue(MetalCommandQueue&&) noexcept;
    MetalCommandQueue& operator=(MetalCommandQueue&&) noexcept;

    explicit operator bool() const noexcept;

  private:
    struct Storage;

    friend MetalCommandQueueResult create_command_queue(const MetalDevice&);
    friend MetalCommandBufferResult create_command_buffer(const MetalCommandQueue&);

    explicit MetalCommandQueue(std::unique_ptr<Storage> storage);

    std::unique_ptr<Storage> storage_;
};

class MetalEvent {
  public:
    MetalEvent() noexcept;
    ~MetalEvent();

    MetalEvent(const MetalEvent&) = delete;
    MetalEvent& operator=(const MetalEvent&) = delete;
    MetalEvent(MetalEvent&&) noexcept;
    MetalEvent& operator=(MetalEvent&&) noexcept;

    explicit operator bool() const noexcept;

  private:
    struct Storage;

    friend MetalEventResult create_event(const MetalDevice&);
    friend MetalCommandError
    encode_wait_for_event(MetalCommandBuffer&, const MetalEvent&,
                          std::uint64_t);
    friend MetalCommandError
    encode_signal_event(MetalCommandBuffer&, const MetalEvent&,
                        std::uint64_t);

    explicit MetalEvent(std::unique_ptr<Storage> storage);

    std::unique_ptr<Storage> storage_;
};

class MetalBuffer {
  public:
    MetalBuffer() noexcept;
    ~MetalBuffer();

    MetalBuffer(const MetalBuffer&) = delete;
    MetalBuffer& operator=(const MetalBuffer&) = delete;
    MetalBuffer(MetalBuffer&&) noexcept;
    MetalBuffer& operator=(MetalBuffer&&) noexcept;

    std::uint64_t size_bytes() const noexcept;
    void* contents() const noexcept;
    explicit operator bool() const noexcept;

  private:
    struct Storage;

    friend MetalBufferResult create_shared_buffer(const MetalDevice&, std::uint64_t);
    friend MetalPooledBufferResult create_striped_pool_buffer(const MetalDevice&, std::uint64_t,
                                                              std::uint32_t);
    friend MetalCommandError set_buffer(MetalComputePass&, const MetalBuffer&, std::uint64_t,
                                        std::uint32_t);
    friend MetalCommandError copy_buffer(MetalBlitPass&, const MetalBuffer&, std::uint64_t,
                                         MetalBuffer&, std::uint64_t, std::uint64_t);
    friend MetalCommandError fill_buffer(MetalBlitPass&, MetalBuffer&, std::uint64_t, std::uint64_t,
                                         std::byte);
    friend MetalCommandError dispatch_threadgroups_indirect(MetalComputePass&, const MetalBuffer&,
                                                            std::uint64_t, MetalSize);
    friend MetalCommandError
    set_indirect_buffer(MetalIndirectCommandBuffer&, std::uint32_t,
                        const MetalBuffer&, std::uint64_t, std::uint32_t);
    friend MetalCommandError
    use_buffer_resource(MetalComputePass&, const MetalBuffer&,
                        MetalResourceUsage);
    friend std::uint64_t metal_buffer_identity(const MetalBuffer&) noexcept;
    friend MetalBufferResult create_buffer_window(
        const MetalBuffer&, std::uint64_t, std::uint64_t);

    MetalBuffer(std::unique_ptr<Storage> storage, std::uint64_t size_bytes);

    std::unique_ptr<Storage> storage_;
    std::uint64_t size_bytes_ = 0;
};

// Indirect-command-buffer kernel bindings silently truncate buffer offsets
// to 32 bits on this hardware generation, while ordinary encoder bindings
// accept 64-bit offsets; Apple documents no limit. Offsets at or above this
// bound must be rebased through a window buffer before entering an ICB.
inline constexpr std::uint64_t kIndirectKernelBufferOffsetLimitBytes =
    1ull << 32;

enum class MetalResourceError : std::uint8_t {
    None,
    InvalidDevice,
    SystemDeviceUnavailable,
    CommandQueueCreationFailed,
    EventCreationFailed,
    InvalidBufferSize,
    BufferCreationFailed,
    OwnershipFailure,
    BufferWindowUnsupported,
};

struct MetalDeviceResult {
    MetalResourceError error;
    std::optional<MetalDevice> device;

    explicit operator bool() const noexcept {
        return error == MetalResourceError::None && device.has_value();
    }
};

struct MetalCommandQueueResult {
    MetalResourceError error;
    std::optional<MetalCommandQueue> command_queue;

    explicit operator bool() const noexcept {
        return error == MetalResourceError::None && command_queue.has_value();
    }
};

struct MetalEventResult {
    MetalResourceError error;
    std::optional<MetalEvent> event;

    explicit operator bool() const noexcept {
        return error == MetalResourceError::None &&
               event.has_value();
    }
};

struct MetalBufferResult {
    MetalResourceError error;
    std::optional<MetalBuffer> buffer;

    explicit operator bool() const noexcept {
        return error == MetalResourceError::None && buffer.has_value();
    }
};

// A tracked placement-heap buffer holding stripe_count equal stripes at
// stripe_stride_bytes apart; windows over each stripe bind as ordinary
// buffers while row-batched kernels bind the pool once and address rows
// through the stride.
struct MetalPooledBufferResult {
    MetalResourceError error;
    std::optional<MetalBuffer> buffer;
    std::uint64_t stripe_stride_bytes;

    explicit operator bool() const noexcept {
        return error == MetalResourceError::None && buffer.has_value();
    }
};

MetalDeviceResult create_system_device();
std::uint64_t metal_device_identity(const MetalDevice& device) noexcept;
MetalCommandQueueResult create_command_queue(const MetalDevice& device);
MetalEventResult create_event(const MetalDevice& device);
MetalBufferResult create_shared_buffer(const MetalDevice& device, std::uint64_t size_bytes);

// Creates an aliasing view of a placement-heap-backed buffer beginning at
// `offset` with `length` bytes. Fails typed when the buffer is not
// heap-backed or the placement is invalid; the view's contents pointer is
// verified against the parent before it is returned.
MetalPooledBufferResult create_striped_pool_buffer(const MetalDevice& device,
                                                   std::uint64_t stripe_bytes,
                                                   std::uint32_t stripe_count);
MetalBufferResult create_buffer_window(const MetalBuffer& buffer,
                                       std::uint64_t offset,
                                       std::uint64_t length);

} // namespace tatara::backend::metal
