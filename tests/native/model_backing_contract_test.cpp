#include "tatara/backend/metal/model_backing.h"

#include <type_traits>
#include <utility>

namespace {

using tatara::backend::metal::adopt_buffer_as_image;
using tatara::backend::metal::MetalBuffer;
using tatara::backend::metal::ModelBackingError;
using tatara::backend::metal::ModelBackingResult;

} // namespace

int main() {
    static_assert(!std::is_copy_constructible_v<ModelBackingResult>);
    static_assert(std::is_move_constructible_v<ModelBackingResult>);

    MetalBuffer empty;
    auto invalid = adopt_buffer_as_image(std::move(empty), 256);
    if (invalid.error != ModelBackingError::InvalidBuffer || invalid.image.has_value() ||
        static_cast<bool>(invalid)) {
        return 1;
    }
    MetalBuffer another;
    if (adopt_buffer_as_image(std::move(another), 0).error != ModelBackingError::InvalidBuffer) {
        return 2;
    }
    return 0;
}
