#include "tatara/backend/metal/model_backing.h"

#include <cstdint>
#include <utility>

namespace tatara::backend::metal {
namespace {

struct BackingContext {
    MetalBuffer buffer;
};

void release_backing(void* context) noexcept {
    delete static_cast<BackingContext*>(context);
}

ModelBackingResult backing_failure(ModelBackingError error, model::ModelImageError image_error) {
    return {.error = error, .image_error = image_error, .image = std::nullopt};
}

} // namespace

ModelBackingResult adopt_buffer_as_image(MetalBuffer&& buffer, std::size_t alignment_bytes) {
    if (!buffer) {
        return backing_failure(ModelBackingError::InvalidBuffer, model::ModelImageError::None);
    }
    if (alignment_bytes == 0 || (alignment_bytes & (alignment_bytes - 1)) != 0) {
        return backing_failure(ModelBackingError::InvalidAlignment, model::ModelImageError::None);
    }
    void* contents = buffer.contents();
    if (contents == nullptr) {
        return backing_failure(ModelBackingError::MissingContents, model::ModelImageError::None);
    }
    if (reinterpret_cast<std::uintptr_t>(contents) % alignment_bytes != 0) {
        return backing_failure(ModelBackingError::MisalignedContents, model::ModelImageError::None);
    }
    const std::size_t size_bytes = static_cast<std::size_t>(buffer.size_bytes());
    auto* context = new BackingContext{.buffer = std::move(buffer)};
    auto adoption = model::ModelImage::adopt(static_cast<std::byte*>(contents), size_bytes,
                                             alignment_bytes, context, release_backing);
    if (!adoption) {
        delete context;
        return backing_failure(ModelBackingError::AdoptionFailed, adoption.error);
    }
    return {
        .error = ModelBackingError::None,
        .image_error = model::ModelImageError::None,
        .image = std::move(adoption.image),
    };
}

} // namespace tatara::backend::metal
