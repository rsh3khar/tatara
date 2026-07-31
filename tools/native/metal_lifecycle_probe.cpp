#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

using namespace tatara::backend::metal;

constexpr std::uint64_t kFixtureBufferBytes = 4096;
constexpr std::uint32_t kFixtureSentinel = 0x54415241u;
constexpr std::uint64_t kFixtureWords = kFixtureBufferBytes / sizeof(std::uint32_t);
constexpr std::uint64_t kFixtureThreadsPerGroup = 256;

constexpr const char* kFixtureSource = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void tatara_lifecycle_fill(device uint* words [[buffer(0)]],
                                  uint index [[thread_position_in_grid]]) {
    words[index] = 0x54415241u;
}
)metal";

} // namespace

int main() {
    auto device = create_system_device();
    if (!device) {
        std::cerr << "system Metal device creation failed\n";
        return 1;
    }
    auto command_queue = create_command_queue(*device.device);
    if (!command_queue) {
        std::cerr << "Metal command queue creation failed\n";
        return 2;
    }

    auto library = create_library_with_source(*device.device, kFixtureSource);
    if (!library) {
        std::cerr << "fixture library compilation failed: " << library.failure_description << '\n';
        return 3;
    }
    if (create_function(*library.library, "missing_function").error !=
        MetalPipelineError::FunctionLookupFailed) {
        std::cerr << "missing fixture function was not rejected\n";
        return 4;
    }
    auto function = create_function(*library.library, "tatara_lifecycle_fill");
    if (!function) {
        std::cerr << "fixture function lookup failed\n";
        return 5;
    }
    auto pipeline = create_compute_pipeline(*device.device, *function.function);
    if (!pipeline) {
        std::cerr << "fixture pipeline creation failed: " << pipeline.failure_description << '\n';
        return 6;
    }

    auto buffer = create_shared_buffer(*device.device, kFixtureBufferBytes);
    if (!buffer || buffer.buffer->size_bytes() != kFixtureBufferBytes) {
        std::cerr << "fixture shared buffer creation failed\n";
        return 7;
    }
    void* contents = buffer.buffer->contents();
    if (contents == nullptr) {
        std::cerr << "fixture shared buffer has no CPU contents\n";
        return 8;
    }
    std::memset(contents, 0, kFixtureBufferBytes);

    auto command_buffer = create_command_buffer(*command_queue.command_queue);
    if (!command_buffer) {
        std::cerr << "command buffer creation failed\n";
        return 9;
    }
    auto compute_pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
    if (!compute_pass) {
        std::cerr << "compute pass creation failed\n";
        return 10;
    }
    if (set_buffer(*compute_pass.compute_pass, *buffer.buffer, 0, kMaxBufferArgumentIndex + 1) !=
        MetalCommandError::InvalidBufferIndex) {
        std::cerr << "out-of-range buffer argument index was not rejected\n";
        return 11;
    }
    constexpr MetalSize kZeroExtent{.width = 0, .height = 1, .depth = 1};
    constexpr MetalSize kUnitExtent{.width = 1, .height = 1, .depth = 1};
    if (dispatch_threadgroups(*compute_pass.compute_pass, kZeroExtent, kUnitExtent) !=
        MetalCommandError::InvalidDispatchExtent) {
        std::cerr << "zero dispatch extent was not rejected\n";
        return 12;
    }

    if (set_compute_pipeline(*compute_pass.compute_pass, *pipeline.pipeline) !=
        MetalCommandError::None) {
        std::cerr << "pipeline bind failed\n";
        return 13;
    }
    if (set_buffer(*compute_pass.compute_pass, *buffer.buffer, 0, 0) != MetalCommandError::None) {
        std::cerr << "fixture buffer bind failed\n";
        return 14;
    }
    constexpr MetalSize kThreadgroups{
        .width = kFixtureWords / kFixtureThreadsPerGroup,
        .height = 1,
        .depth = 1,
    };
    constexpr MetalSize kThreadsPerGroup{
        .width = kFixtureThreadsPerGroup,
        .height = 1,
        .depth = 1,
    };
    if (dispatch_threadgroups(*compute_pass.compute_pass, kThreadgroups, kThreadsPerGroup) !=
        MetalCommandError::None) {
        std::cerr << "fixture dispatch failed\n";
        return 15;
    }
    auto ended = end_compute_pass(std::move(*compute_pass.compute_pass));
    if (!ended) {
        std::cerr << "compute pass end failed\n";
        return 16;
    }
    auto pending = commit(std::move(*ended.command_buffer));
    if (!pending) {
        std::cerr << "command buffer commit failed\n";
        return 17;
    }
    auto execution = wait_until_completed(std::move(*pending.pending_execution));
    if (!execution) {
        std::cerr << "fixture execution failed: " << execution.failure_description.view() << '\n';
        return 18;
    }

    const auto* words = static_cast<const std::uint32_t*>(contents);
    for (std::uint64_t index = 0; index < kFixtureWords; ++index) {
        if (words[index] != kFixtureSentinel) {
            std::cerr << "fixture word " << index << " holds " << words[index]
                      << " instead of the sentinel\n";
            return 19;
        }
    }

    std::cout << "metal lifecycle ownership: PASS\n"
              << "  device: " << device.device->name() << '\n'
              << "  fixture words verified: " << kFixtureWords << '\n'
              << "  command buffers submitted: 1\n";
    return 0;
}
