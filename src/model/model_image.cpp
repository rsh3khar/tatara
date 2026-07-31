#include "tatara/model/model_image.h"

#include <cstdint>
#include <utility>

namespace tatara::model {

ModelImage::ModelImage(std::byte* data, std::size_t size_bytes, std::size_t alignment_bytes,
                       void* context, Release release) noexcept
    : data_(data), size_bytes_(size_bytes), alignment_bytes_(alignment_bytes), context_(context),
      release_(release) {}

ModelImage::ModelImage(ModelImage&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_bytes_(std::exchange(other.size_bytes_, 0)),
      alignment_bytes_(std::exchange(other.alignment_bytes_, 0)),
      context_(std::exchange(other.context_, nullptr)),
      release_(std::exchange(other.release_, nullptr)) {}

ModelImage& ModelImage::operator=(ModelImage&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    reset();
    data_ = std::exchange(other.data_, nullptr);
    size_bytes_ = std::exchange(other.size_bytes_, 0);
    alignment_bytes_ = std::exchange(other.alignment_bytes_, 0);
    context_ = std::exchange(other.context_, nullptr);
    release_ = std::exchange(other.release_, nullptr);
    return *this;
}

ModelImage::~ModelImage() {
    reset();
}

ModelImageAdoption ModelImage::adopt(std::byte* data, std::size_t size_bytes,
                                     std::size_t alignment_bytes, void* context,
                                     Release release) noexcept {
    if (data == nullptr) {
        return {.error = ModelImageError::NullData, .image = std::nullopt};
    }
    if (size_bytes == 0) {
        return {.error = ModelImageError::ZeroSize, .image = std::nullopt};
    }
    if (alignment_bytes == 0 || (alignment_bytes & (alignment_bytes - 1)) != 0) {
        return {.error = ModelImageError::InvalidAlignment, .image = std::nullopt};
    }
    if (reinterpret_cast<std::uintptr_t>(data) % alignment_bytes != 0) {
        return {.error = ModelImageError::MisalignedData, .image = std::nullopt};
    }
    if (context == nullptr) {
        return {.error = ModelImageError::NullContext, .image = std::nullopt};
    }
    if (release == nullptr) {
        return {.error = ModelImageError::MissingRelease, .image = std::nullopt};
    }
    return {
        .error = ModelImageError::None,
        .image = ModelImage(data, size_bytes, alignment_bytes, context, release),
    };
}

void ModelImage::reset() noexcept {
    if (release_ != nullptr) {
        release_(context_);
    }
    data_ = nullptr;
    size_bytes_ = 0;
    alignment_bytes_ = 0;
    context_ = nullptr;
    release_ = nullptr;
}

} // namespace tatara::model
