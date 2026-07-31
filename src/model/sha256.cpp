#include "tatara/model/sha256.h"

#include <CommonCrypto/CommonDigest.h>

#include <limits>

namespace tatara::model {
namespace {

static_assert(sizeof(CC_SHA256_CTX) <= sizeof(Sha256), "opaque digest storage too small");
static_assert(alignof(CC_SHA256_CTX) <= 8, "opaque digest storage under-aligned");

CC_SHA256_CTX* context_of(std::array<std::byte, 128>& storage) noexcept {
    return reinterpret_cast<CC_SHA256_CTX*>(storage.data());
}

} // namespace

Sha256::Sha256() noexcept {
    CC_SHA256_Init(context_of(context_));
}

void Sha256::update(std::span<const std::byte> bytes) noexcept {
    const std::byte* cursor = bytes.data();
    std::size_t remaining = bytes.size();
    constexpr std::size_t kMaxStep = std::numeric_limits<CC_LONG>::max();
    while (remaining > 0) {
        const std::size_t step = remaining < kMaxStep ? remaining : kMaxStep;
        CC_SHA256_Update(context_of(context_), cursor, static_cast<CC_LONG>(step));
        cursor += step;
        remaining -= step;
    }
}

Sha256Digest Sha256::finish() noexcept {
    Sha256Digest digest;
    CC_SHA256_Final(digest.data(), context_of(context_));
    CC_SHA256_Init(context_of(context_));
    return digest;
}

std::string sha256_hex(const Sha256Digest& digest) {
    static constexpr char kHexDigits[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(digest.size() * 2);
    for (const std::uint8_t value : digest) {
        hex.push_back(kHexDigits[value >> 4]);
        hex.push_back(kHexDigits[value & 0x0F]);
    }
    return hex;
}

} // namespace tatara::model
