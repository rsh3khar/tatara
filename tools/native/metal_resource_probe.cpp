#include "tatara/backend/metal/resources.h"

#include <iostream>

int main() {
    using namespace tatara::backend::metal;

    auto device = create_system_device();
    if (!device || !device.device) {
        std::cerr << "system Metal device creation failed\n";
        return 1;
    }
    auto command_queue = create_command_queue(*device.device);
    if (!command_queue || !command_queue.command_queue) {
        std::cerr << "Metal command queue creation failed\n";
        return 2;
    }
    if (create_shared_buffer(*device.device, 0).error != MetalResourceError::InvalidBufferSize) {
        std::cerr << "zero-sized Metal buffer was not rejected\n";
        return 3;
    }
    constexpr std::uint64_t kProbeBufferBytes = 4096;
    auto buffer = create_shared_buffer(*device.device, kProbeBufferBytes);
    if (!buffer || !buffer.buffer || buffer.buffer->size_bytes() != kProbeBufferBytes) {
        std::cerr << "shared Metal buffer creation failed\n";
        return 4;
    }
    std::cout << "metal resource ownership: PASS\n"
              << "  device: " << device.device->name() << '\n'
              << "  command queue: created\n"
              << "  shared buffer bytes: " << buffer.buffer->size_bytes() << '\n'
              << "  command buffers submitted: 0\n";
    return 0;
}
