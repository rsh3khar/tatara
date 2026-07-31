#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

namespace tatara::model {

enum class ModelImageError : std::uint8_t {
    None,
    NullData,
    ZeroSize,
    InvalidAlignment,
    MisalignedData,
    NullContext,
    MissingRelease,
};

struct ModelImageAdoption;

class ModelImage {
  public:
    using Release = void (*)(void*) noexcept;

    ModelImage(const ModelImage&) = delete;
    ModelImage& operator=(const ModelImage&) = delete;
    ModelImage(ModelImage&& other) noexcept;
    ModelImage& operator=(ModelImage&& other) noexcept;
    ~ModelImage();

    // Ownership transfers only on success; release(context) is then called exactly once.
    static ModelImageAdoption adopt(std::byte* data, std::size_t size_bytes,
                                    std::size_t alignment_bytes, void* context,
                                    Release release) noexcept;

    const std::byte* data() const noexcept {
        return data_;
    }
    std::size_t size_bytes() const noexcept {
        return size_bytes_;
    }
    std::size_t alignment_bytes() const noexcept {
        return alignment_bytes_;
    }
    explicit operator bool() const noexcept {
        return data_ != nullptr;
    }

  private:
    ModelImage(std::byte* data, std::size_t size_bytes, std::size_t alignment_bytes, void* context,
               Release release) noexcept;
    void reset() noexcept;

    std::byte* data_ = nullptr;
    std::size_t size_bytes_ = 0;
    std::size_t alignment_bytes_ = 0;
    void* context_ = nullptr;
    Release release_ = nullptr;
};

struct ModelImageAdoption {
    ModelImageError error;
    std::optional<ModelImage> image;

    explicit operator bool() const noexcept {
        return error == ModelImageError::None && image.has_value();
    }
};

} // namespace tatara::model
