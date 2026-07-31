#include "platform/apple/object_owner.h"

#include <utility>

namespace tatara::platform::apple {

ObjectOwner::ObjectOwner(void* object, Release release) noexcept
    : object_(object), release_(release) {}

ObjectOwner::~ObjectOwner() {
    reset();
}

ObjectOwner::ObjectOwner(ObjectOwner&& other) noexcept
    : object_(std::exchange(other.object_, nullptr)),
      release_(std::exchange(other.release_, nullptr)) {}

ObjectOwner& ObjectOwner::operator=(ObjectOwner&& other) noexcept {
    if (this != &other) {
        reset();
        object_ = std::exchange(other.object_, nullptr);
        release_ = std::exchange(other.release_, nullptr);
    }
    return *this;
}

std::optional<ObjectOwner> ObjectOwner::adopt(void* object, Release release) noexcept {
    if (object == nullptr || release == nullptr) {
        return std::nullopt;
    }
    return ObjectOwner(object, release);
}

void* ObjectOwner::get() const noexcept {
    return object_;
}

ObjectOwner::operator bool() const noexcept {
    return object_ != nullptr;
}

void ObjectOwner::reset() noexcept {
    if (object_ != nullptr) {
        release_(object_);
    }
    object_ = nullptr;
    release_ = nullptr;
}

} // namespace tatara::platform::apple
