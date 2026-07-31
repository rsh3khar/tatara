#pragma once

#include "tatara/backend/metal/resources.h"
#include "tatara/model/model_image.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace tatara::backend::metal {

enum class ModelBackingError : std::uint8_t {
    None,
    InvalidBuffer,
    InvalidAlignment,
    MissingContents,
    MisalignedContents,
    AdoptionFailed,
};

struct ModelBackingResult {
    ModelBackingError error;
    model::ModelImageError image_error;
    std::optional<model::ModelImage> image;

    explicit operator bool() const noexcept {
        return error == ModelBackingError::None && image.has_value();
    }
};

// Consumes the shared buffer; the returned image's exactly-once release
// destroys the retained Metal object. Adopt only after population succeeded.
ModelBackingResult adopt_buffer_as_image(MetalBuffer&& buffer, std::size_t alignment_bytes);

} // namespace tatara::backend::metal
