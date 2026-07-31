#include "tatara/model/sha256.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using tatara::model::Sha256;
using tatara::model::sha256_hex;

std::span<const std::byte> bytes_of(std::string_view text) {
    return {reinterpret_cast<const std::byte*>(text.data()), text.size()};
}

std::string one_shot_hex(std::string_view text) {
    Sha256 hasher;
    hasher.update(bytes_of(text));
    return sha256_hex(hasher.finish());
}

} // namespace

int main() {
    if (one_shot_hex("") != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") {
        return 1;
    }
    if (one_shot_hex("abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
        return 2;
    }
    const std::string_view two_block = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    if (one_shot_hex(two_block) !=
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1") {
        return 3;
    }

    const std::string million(1000000, 'a');
    if (one_shot_hex(million) !=
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0") {
        return 4;
    }

    Sha256 chunked;
    std::size_t position = 0;
    std::size_t step = 1;
    while (position < two_block.size()) {
        const std::size_t take = std::min(step, two_block.size() - position);
        chunked.update(bytes_of(two_block.substr(position, take)));
        position += take;
        step = step % 7 + 1;
    }
    if (sha256_hex(chunked.finish()) != one_shot_hex(two_block)) {
        return 5;
    }

    Sha256 reused;
    reused.update(bytes_of("abc"));
    if (sha256_hex(reused.finish()) != one_shot_hex("abc")) {
        return 6;
    }
    if (sha256_hex(reused.finish()) != one_shot_hex("")) {
        return 7;
    }
    return 0;
}
