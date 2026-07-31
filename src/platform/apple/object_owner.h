#pragma once

#include <optional>

namespace tatara::platform::apple {

class ObjectOwner {
  public:
    using Release = void (*)(void*) noexcept;

    ObjectOwner() noexcept = default;
    ~ObjectOwner();

    ObjectOwner(const ObjectOwner&) = delete;
    ObjectOwner& operator=(const ObjectOwner&) = delete;
    ObjectOwner(ObjectOwner&& other) noexcept;
    ObjectOwner& operator=(ObjectOwner&& other) noexcept;

    static std::optional<ObjectOwner> adopt(void* object, Release release) noexcept;

    void* get() const noexcept;
    explicit operator bool() const noexcept;

  private:
    ObjectOwner(void* object, Release release) noexcept;
    void reset() noexcept;

    void* object_ = nullptr;
    Release release_ = nullptr;
};

} // namespace tatara::platform::apple
