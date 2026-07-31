#pragma once

#include "platform/apple/object_owner.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <optional>

#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"

namespace tatara::backend::metal {

inline void release_object(void* pointer) noexcept {
    id object = (__bridge_transfer id)pointer;
    (void)object;
}

inline std::optional<platform::apple::ObjectOwner> retain_object(id object) noexcept {
    if (object == nil) {
        return std::nullopt;
    }
    void* retained = (__bridge_retained void*)object;
    auto owner = platform::apple::ObjectOwner::adopt(retained, release_object);
    if (!owner) {
        release_object(retained);
    }
    return owner;
}

struct MetalDevice::Storage {
    platform::apple::ObjectOwner object;
};

struct MetalCommandQueue::Storage {
    platform::apple::ObjectOwner object;
};

struct MetalEvent::Storage {
    platform::apple::ObjectOwner object;
};

struct MetalBuffer::Storage {
    platform::apple::ObjectOwner object;
};

struct MetalLibrary::Storage {
    platform::apple::ObjectOwner object;
};

struct MetalFunction::Storage {
    platform::apple::ObjectOwner object;
};

struct MetalComputePipeline::Storage {
    platform::apple::ObjectOwner object;
};

} // namespace tatara::backend::metal
