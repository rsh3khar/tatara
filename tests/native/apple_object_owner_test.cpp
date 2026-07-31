#include "platform/apple/object_owner.h"

#include <type_traits>
#include <utility>

namespace {

using tatara::platform::apple::ObjectOwner;

struct FakeObject {
    int* releases;
};

void release_fake(void* pointer) noexcept {
    auto* object = static_cast<FakeObject*>(pointer);
    ++*object->releases;
    delete object;
}

int test_adoption() {
    int unowned = 0;
    if (ObjectOwner::adopt(nullptr, release_fake) || ObjectOwner::adopt(&unowned, nullptr)) {
        return 1;
    }

    int releases = 0;
    auto adopted = ObjectOwner::adopt(new FakeObject{.releases = &releases}, release_fake);
    if (!adopted || !*adopted || adopted->get() == nullptr || releases != 0) {
        return 2;
    }
    {
        ObjectOwner first = std::move(*adopted);
        ObjectOwner second = std::move(first);
        if (first || !second || releases != 0) {
            return 3;
        }

        int replacement_releases = 0;
        auto replacement =
            ObjectOwner::adopt(new FakeObject{.releases = &replacement_releases}, release_fake);
        if (!replacement) {
            return 4;
        }
        *replacement = std::move(second);
        if (second || !*replacement || replacement_releases != 1 || releases != 0) {
            return 5;
        }
    }
    return releases == 1 ? 0 : 6;
}

} // namespace

int main() {
    static_assert(!std::is_copy_constructible_v<ObjectOwner>);
    static_assert(!std::is_copy_assignable_v<ObjectOwner>);
    static_assert(std::is_nothrow_move_constructible_v<ObjectOwner>);
    static_assert(std::is_nothrow_move_assignable_v<ObjectOwner>);
    return test_adoption();
}
