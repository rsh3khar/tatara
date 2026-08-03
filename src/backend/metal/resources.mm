#include "tatara/backend/metal/resources.h"

#include "backend/metal/resource_storage.h"

#include <limits>
#include <utility>

namespace tatara::backend::metal {

MetalDevice::MetalDevice() noexcept = default;
MetalDevice::~MetalDevice() = default;
MetalDevice::MetalDevice(MetalDevice&&) noexcept = default;
MetalDevice& MetalDevice::operator=(MetalDevice&&) noexcept = default;

MetalDevice::MetalDevice(std::unique_ptr<Storage> storage, std::string name)
    : storage_(std::move(storage)), name_(std::move(name)) {}

const std::string& MetalDevice::name() const noexcept {
    return name_;
}

MetalDevice::operator bool() const noexcept {
    return storage_ != nullptr && storage_->object;
}

std::uint64_t metal_device_identity(const MetalDevice& device) noexcept {
    return device
               ? static_cast<std::uint64_t>(
                     reinterpret_cast<std::uintptr_t>(
                         device.storage_->object.get()))
               : 0;
}

MetalCommandQueue::MetalCommandQueue() noexcept = default;
MetalCommandQueue::~MetalCommandQueue() = default;
MetalCommandQueue::MetalCommandQueue(MetalCommandQueue&&) noexcept = default;
MetalCommandQueue& MetalCommandQueue::operator=(MetalCommandQueue&&) noexcept = default;

MetalCommandQueue::MetalCommandQueue(std::unique_ptr<Storage> storage)
    : storage_(std::move(storage)) {}

MetalCommandQueue::operator bool() const noexcept {
    return storage_ != nullptr && storage_->object;
}

MetalEvent::MetalEvent() noexcept = default;
MetalEvent::~MetalEvent() = default;
MetalEvent::MetalEvent(MetalEvent&&) noexcept = default;
MetalEvent& MetalEvent::operator=(MetalEvent&&) noexcept = default;

MetalEvent::MetalEvent(std::unique_ptr<Storage> storage)
    : storage_(std::move(storage)) {}

MetalEvent::operator bool() const noexcept {
    return storage_ != nullptr && storage_->object;
}

MetalBuffer::MetalBuffer() noexcept = default;
MetalBuffer::~MetalBuffer() = default;
MetalBuffer::MetalBuffer(MetalBuffer&&) noexcept = default;
MetalBuffer& MetalBuffer::operator=(MetalBuffer&&) noexcept = default;

MetalBuffer::MetalBuffer(std::unique_ptr<Storage> storage, std::uint64_t size_bytes)
    : storage_(std::move(storage)), size_bytes_(size_bytes) {}

std::uint64_t MetalBuffer::size_bytes() const noexcept {
    return size_bytes_;
}

void* MetalBuffer::contents() const noexcept {
    if (storage_ == nullptr || !storage_->object) {
        return nullptr;
    }
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)storage_->object.get();
    return buffer.contents;
}

MetalBuffer::operator bool() const noexcept {
    return storage_ != nullptr && storage_->object && size_bytes_ != 0;
}

MetalDeviceResult create_system_device() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return {.error = MetalResourceError::SystemDeviceUnavailable, .device = std::nullopt};
        }
        auto owner = retain_object(device);
        if (!owner) {
            return {.error = MetalResourceError::OwnershipFailure, .device = std::nullopt};
        }
        const char* raw_name = device.name.UTF8String;
        std::string name = raw_name == nullptr ? "unknown" : raw_name;
        auto storage = std::make_unique<MetalDevice::Storage>();
        storage->object = std::move(*owner);
        return {
            .error = MetalResourceError::None,
            .device = MetalDevice(std::move(storage), std::move(name)),
        };
    }
}

MetalCommandQueueResult create_command_queue(const MetalDevice& device) {
    if (!device) {
        return {.error = MetalResourceError::InvalidDevice, .command_queue = std::nullopt};
    }
    @autoreleasepool {
        id<MTLDevice> native_device = (__bridge id<MTLDevice>)device.storage_->object.get();
        id<MTLCommandQueue> command_queue = [native_device newCommandQueue];
        if (command_queue == nil) {
            return {
                .error = MetalResourceError::CommandQueueCreationFailed,
                .command_queue = std::nullopt,
            };
        }
        auto owner = retain_object(command_queue);
        if (!owner) {
            return {.error = MetalResourceError::OwnershipFailure, .command_queue = std::nullopt};
        }
        auto storage = std::make_unique<MetalCommandQueue::Storage>();
        storage->object = std::move(*owner);
        return {
            .error = MetalResourceError::None,
            .command_queue = MetalCommandQueue(std::move(storage)),
        };
    }
}

MetalEventResult create_event(const MetalDevice& device) {
    if (!device) {
        return {
            .error = MetalResourceError::InvalidDevice,
            .event = std::nullopt,
        };
    }
    @autoreleasepool {
        id<MTLDevice> native_device =
            (__bridge id<MTLDevice>)device.storage_->object.get();
        id<MTLEvent> event = [native_device newEvent];
        if (event == nil) {
            return {
                .error = MetalResourceError::EventCreationFailed,
                .event = std::nullopt,
            };
        }
        auto owner = retain_object(event);
        if (!owner) {
            return {
                .error = MetalResourceError::OwnershipFailure,
                .event = std::nullopt,
            };
        }
        auto storage = std::make_unique<MetalEvent::Storage>();
        storage->object = std::move(*owner);
        return {
            .error = MetalResourceError::None,
            .event = MetalEvent(std::move(storage)),
        };
    }
}

MetalBufferResult create_shared_buffer(const MetalDevice& device, std::uint64_t size_bytes) {
    if (!device) {
        return {.error = MetalResourceError::InvalidDevice, .buffer = std::nullopt};
    }
    if (size_bytes == 0) {
        return {.error = MetalResourceError::InvalidBufferSize, .buffer = std::nullopt};
    }
    if constexpr (sizeof(NSUInteger) < sizeof(std::uint64_t)) {
        if (size_bytes > std::numeric_limits<NSUInteger>::max()) {
            return {.error = MetalResourceError::InvalidBufferSize, .buffer = std::nullopt};
        }
    }
    @autoreleasepool {
        id<MTLDevice> native_device = (__bridge id<MTLDevice>)device.storage_->object.get();
        id<MTLBuffer> buffer = nil;
        if (size_bytes > kIndirectKernelBufferOffsetLimitBytes) {
            // A buffer this large has bindable offsets past the indirect
            // kernel-binding limit, so it is placed on a tracked placement
            // heap; window views created over the same heap keep every
            // indirect binding offset below the limit.
            const MTLSizeAndAlign heap_extent = [native_device
                heapBufferSizeAndAlignWithLength:static_cast<NSUInteger>(size_bytes)
                                         options:MTLResourceStorageModeShared];
            MTLHeapDescriptor* heap_descriptor = [MTLHeapDescriptor new];
            heap_descriptor.type = MTLHeapTypePlacement;
            heap_descriptor.storageMode = MTLStorageModeShared;
            heap_descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
            heap_descriptor.size = heap_extent.size;
            id<MTLHeap> heap = [native_device newHeapWithDescriptor:heap_descriptor];
            if (heap == nil) {
                return {.error = MetalResourceError::BufferCreationFailed, .buffer = std::nullopt};
            }
            buffer = [heap newBufferWithLength:static_cast<NSUInteger>(size_bytes)
                                       options:MTLResourceStorageModeShared
                                        offset:0];
        } else {
            buffer = [native_device newBufferWithLength:static_cast<NSUInteger>(size_bytes)
                                                options:MTLResourceStorageModeShared];
        }
        if (buffer == nil) {
            return {.error = MetalResourceError::BufferCreationFailed, .buffer = std::nullopt};
        }
        if (buffer.length != static_cast<NSUInteger>(size_bytes) ||
            buffer.storageMode != MTLStorageModeShared) {
            return {.error = MetalResourceError::BufferCreationFailed, .buffer = std::nullopt};
        }
        auto owner = retain_object(buffer);
        if (!owner) {
            return {.error = MetalResourceError::OwnershipFailure, .buffer = std::nullopt};
        }
        auto storage = std::make_unique<MetalBuffer::Storage>();
        storage->object = std::move(*owner);
        return {
            .error = MetalResourceError::None,
            .buffer = MetalBuffer(std::move(storage), size_bytes),
        };
    }
}

MetalBufferResult create_buffer_window(const MetalBuffer& buffer,
                                       std::uint64_t offset,
                                       std::uint64_t length) {
    if (!buffer) {
        return {.error = MetalResourceError::InvalidDevice, .buffer = std::nullopt};
    }
    if (length == 0 || offset >= buffer.size_bytes() ||
        length > buffer.size_bytes() - offset) {
        return {.error = MetalResourceError::InvalidBufferSize, .buffer = std::nullopt};
    }
    @autoreleasepool {
        id<MTLBuffer> native_buffer =
            (__bridge id<MTLBuffer>)buffer.storage_->object.get();
        id<MTLHeap> heap = native_buffer.heap;
        if (heap == nil || heap.type != MTLHeapTypePlacement) {
            return {.error = MetalResourceError::BufferWindowUnsupported,
                    .buffer = std::nullopt};
        }
        id<MTLBuffer> window =
            [heap newBufferWithLength:static_cast<NSUInteger>(length)
                              options:MTLResourceStorageModeShared
                               offset:native_buffer.heapOffset +
                                      static_cast<NSUInteger>(offset)];
        if (window == nil) {
            return {.error = MetalResourceError::BufferCreationFailed, .buffer = std::nullopt};
        }
        if (window.length != static_cast<NSUInteger>(length) ||
            window.contents !=
                static_cast<std::byte*>(buffer.contents()) + offset) {
            return {.error = MetalResourceError::BufferCreationFailed, .buffer = std::nullopt};
        }
        auto owner = retain_object(window);
        if (!owner) {
            return {.error = MetalResourceError::OwnershipFailure, .buffer = std::nullopt};
        }
        auto storage = std::make_unique<MetalBuffer::Storage>();
        storage->object = std::move(*owner);
        return {
            .error = MetalResourceError::None,
            .buffer = MetalBuffer(std::move(storage), length),
        };
    }
}

MetalPooledBufferResult create_striped_pool_buffer(const MetalDevice& device,
                                                   std::uint64_t stripe_bytes,
                                                   std::uint32_t stripe_count) {
    if (!device) {
        return {.error = MetalResourceError::InvalidDevice,
                .buffer = std::nullopt,
                .stripe_stride_bytes = 0};
    }
    if (stripe_bytes == 0 || stripe_count == 0) {
        return {.error = MetalResourceError::InvalidBufferSize,
                .buffer = std::nullopt,
                .stripe_stride_bytes = 0};
    }
    @autoreleasepool {
        id<MTLDevice> native_device = (__bridge id<MTLDevice>)device.storage_->object.get();
        const MTLSizeAndAlign stripe_extent = [native_device
            heapBufferSizeAndAlignWithLength:static_cast<NSUInteger>(stripe_bytes)
                                     options:MTLResourceStorageModeShared];
        std::uint64_t stride = stripe_extent.size;
        if (stripe_extent.align != 0 && stride % stripe_extent.align != 0) {
            stride += stripe_extent.align - stride % stripe_extent.align;
        }
        if (stride == 0 ||
            stride > std::numeric_limits<NSUInteger>::max() / stripe_count) {
            return {.error = MetalResourceError::InvalidBufferSize,
                    .buffer = std::nullopt,
                    .stripe_stride_bytes = 0};
        }
        const std::uint64_t total_bytes = stride * stripe_count;
        MTLHeapDescriptor* heap_descriptor = [MTLHeapDescriptor new];
        heap_descriptor.type = MTLHeapTypePlacement;
        heap_descriptor.storageMode = MTLStorageModeShared;
        heap_descriptor.hazardTrackingMode = MTLHazardTrackingModeTracked;
        heap_descriptor.size = static_cast<NSUInteger>(total_bytes);
        id<MTLHeap> heap = [native_device newHeapWithDescriptor:heap_descriptor];
        if (heap == nil) {
            return {.error = MetalResourceError::BufferCreationFailed,
                    .buffer = std::nullopt,
                    .stripe_stride_bytes = 0};
        }
        id<MTLBuffer> buffer = [heap newBufferWithLength:static_cast<NSUInteger>(total_bytes)
                                                 options:MTLResourceStorageModeShared
                                                  offset:0];
        if (buffer == nil || buffer.length != static_cast<NSUInteger>(total_bytes) ||
            buffer.storageMode != MTLStorageModeShared) {
            return {.error = MetalResourceError::BufferCreationFailed,
                    .buffer = std::nullopt,
                    .stripe_stride_bytes = 0};
        }
        auto owner = retain_object(buffer);
        if (!owner) {
            return {.error = MetalResourceError::OwnershipFailure,
                    .buffer = std::nullopt,
                    .stripe_stride_bytes = 0};
        }
        auto storage = std::make_unique<MetalBuffer::Storage>();
        storage->object = std::move(*owner);
        return {
            .error = MetalResourceError::None,
            .buffer = MetalBuffer(std::move(storage), total_bytes),
            .stripe_stride_bytes = stride,
        };
    }
}

} // namespace tatara::backend::metal
