#include "tatara/backend/metal/resources.h"

#include <type_traits>
#include <utility>

namespace {

using tatara::backend::metal::create_command_queue;
using tatara::backend::metal::create_shared_buffer;
using tatara::backend::metal::MetalBuffer;
using tatara::backend::metal::MetalCommandQueue;
using tatara::backend::metal::MetalDevice;
using tatara::backend::metal::MetalResourceError;

template <typename Resource> consteval bool move_only() {
    return !std::is_copy_constructible_v<Resource> && !std::is_copy_assignable_v<Resource> &&
           std::is_nothrow_move_constructible_v<Resource> &&
           std::is_nothrow_move_assignable_v<Resource>;
}

} // namespace

int main() {
    static_assert(move_only<MetalDevice>());
    static_assert(move_only<MetalCommandQueue>());
    static_assert(move_only<MetalBuffer>());

    MetalDevice device;
    if (device || !device.name().empty()) {
        return 1;
    }
    if (create_command_queue(device).error != MetalResourceError::InvalidDevice) {
        return 2;
    }
    if (create_shared_buffer(device, 4096).error != MetalResourceError::InvalidDevice) {
        return 3;
    }
    MetalDevice moved = std::move(device);
    return moved ? 4 : 0;
}
