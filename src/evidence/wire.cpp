#include "tatara/evidence/wire.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>

namespace tatara::evidence {
namespace {

constexpr std::string_view kSidecarMagic = "tatara-sha256-sidecar-v1";
constexpr std::string_view kSha256Prefix = "sha256=";
constexpr std::string_view kSizePrefix = "size_bytes=";
constexpr std::array<std::string_view, 5> kSchemaV1AggregateDomains = {
    "tatara.command.v1",
    "tatara.pinned-file-set.v1",
    "tatara.registry-snapshot.v1",
    "tatara.tree-manifest.v1",
    "tatara.aggregate-test.v1",
};

bool valid_utf8(std::string_view value) noexcept {
    std::size_t index = 0;
    while (index < value.size()) {
        const auto first = static_cast<unsigned char>(value[index]);
        if (first <= 0x7f) {
            ++index;
            continue;
        }

        std::size_t continuation_count = 0;
        std::uint32_t code_point = 0;
        if (first >= 0xc2 && first <= 0xdf) {
            continuation_count = 1;
            code_point = first & 0x1fU;
        } else if (first >= 0xe0 && first <= 0xef) {
            continuation_count = 2;
            code_point = first & 0x0fU;
        } else if (first >= 0xf0 && first <= 0xf4) {
            continuation_count = 3;
            code_point = first & 0x07U;
        } else {
            return false;
        }
        if (continuation_count > value.size() - index - 1) {
            return false;
        }
        for (std::size_t offset = 1; offset <= continuation_count; ++offset) {
            const auto next = static_cast<unsigned char>(value[index + offset]);
            if ((next & 0xc0U) != 0x80U) {
                return false;
            }
            code_point = (code_point << 6U) | (next & 0x3fU);
        }
        if ((continuation_count == 2 && code_point < 0x800U) ||
            (continuation_count == 3 && code_point < 0x10000U) ||
            (code_point >= 0xd800U && code_point <= 0xdfffU) ||
            code_point > 0x10ffffU) {
            return false;
        }
        index += continuation_count + 1;
    }
    return true;
}

std::span<const std::byte> bytes_of(std::string_view value) noexcept {
    return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
}

std::array<std::byte, 8> encode_u64_be(std::uint64_t value) noexcept {
    std::array<std::byte, 8> encoded{};
    for (std::size_t index = 0; index < encoded.size(); ++index) {
        const std::size_t shift = (encoded.size() - index - 1) * 8;
        encoded[index] = static_cast<std::byte>(value >> shift);
    }
    return encoded;
}

SidecarError parse_u64(std::string_view text, std::uint64_t& value) noexcept {
    if (text.empty() || (text.size() > 1 && text.front() == '0')) {
        return SidecarError::InvalidInteger;
    }

    std::uint64_t parsed = 0;
    for (const char character : text) {
        if (character < '0' || character > '9') {
            return SidecarError::InvalidInteger;
        }
        const auto digit = static_cast<std::uint64_t>(character - '0');
        if (parsed > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
            return SidecarError::NumericRange;
        }
        parsed = parsed * 10U + digit;
    }
    value = parsed;
    return SidecarError::None;
}

DigestSidecarDecodeResult sidecar_failure(SidecarError error) noexcept {
    return {.error = error, .sidecar = {}};
}

} // namespace

IdentifierError validate_identifier(std::string_view value) noexcept {
    if (value.empty()) {
        return IdentifierError::Empty;
    }
    if (value.size() > kMaximumIdentifierBytes) {
        return IdentifierError::TooLong;
    }
    const auto valid_character = [](char character) {
        return (character >= 'A' && character <= 'Z') ||
               (character >= 'a' && character <= 'z') ||
               (character >= '0' && character <= '9') || character == '.' ||
               character == '_' || character == '-';
    };
    if (!std::all_of(value.begin(), value.end(), valid_character) ||
        !((value.front() >= 'A' && value.front() <= 'Z') ||
          (value.front() >= 'a' && value.front() <= 'z') ||
          (value.front() >= '0' && value.front() <= '9'))) {
        return IdentifierError::InvalidCharacter;
    }
    return IdentifierError::None;
}

Sha256SpellingError validate_sha256_spelling(std::string_view value) noexcept {
    if (value.size() != kSha256TextBytes) {
        return Sha256SpellingError::WrongLength;
    }
    const bool valid = std::all_of(value.begin(), value.end(), [](char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
    });
    return valid ? Sha256SpellingError::None : Sha256SpellingError::InvalidCharacter;
}

RelativePathError validate_relative_path(std::string_view value) noexcept {
    if (value.empty()) {
        return RelativePathError::Empty;
    }
    if (value.size() > kMaximumRelativePathBytes) {
        return RelativePathError::TooLong;
    }
    if (!valid_utf8(value)) {
        return RelativePathError::InvalidUtf8;
    }
    if (value.front() == '/') {
        return RelativePathError::Absolute;
    }
    if (value.find_first_of(std::string_view{"\0\r\n", 3}) != std::string_view::npos) {
        return RelativePathError::ForbiddenCharacter;
    }
    std::size_t component_begin = 0;
    while (component_begin <= value.size()) {
        const std::size_t component_end = value.find('/', component_begin);
        const std::string_view component =
            component_end == std::string_view::npos
                ? value.substr(component_begin)
                : value.substr(component_begin, component_end - component_begin);
        if (component.empty()) {
            return RelativePathError::EmptyComponent;
        }
        if (component == ".") {
            return RelativePathError::DotComponent;
        }
        if (component == "..") {
            return RelativePathError::ParentComponent;
        }
        if (component_end == std::string_view::npos) {
            break;
        }
        component_begin = component_end + 1;
    }
    return RelativePathError::None;
}

AggregateDigestResult
schema_v1_aggregate_digest(std::string_view domain,
                           std::span<const std::span<const std::byte>> elements) noexcept {
    if (domain.empty()) {
        return {.error = AggregateError::EmptyDomain, .digest = {}};
    }
    if (std::find(kSchemaV1AggregateDomains.begin(), kSchemaV1AggregateDomains.end(), domain) ==
        kSchemaV1AggregateDomains.end()) {
        return {.error = AggregateError::InvalidDomain, .digest = {}};
    }

    model::Sha256 hasher;
    hasher.update(bytes_of(domain));
    constexpr std::array<std::byte, 1> kDomainTerminator = {std::byte{0}};
    hasher.update(kDomainTerminator);
    for (const std::span<const std::byte> element : elements) {
        if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
            if (element.size() > std::numeric_limits<std::uint64_t>::max()) {
                return {.error = AggregateError::ElementSizeOverflow, .digest = {}};
            }
        }
        const auto length = encode_u64_be(static_cast<std::uint64_t>(element.size()));
        hasher.update(length);
        hasher.update(element);
    }
    return {.error = AggregateError::None, .digest = hasher.finish()};
}

DigestSidecarDecodeResult decode_digest_sidecar(std::string_view bytes) noexcept {
    if (bytes.size() > kMaximumDigestSidecarBytes) {
        return sidecar_failure(SidecarError::ArtifactTooLarge);
    }
    if (bytes.empty()) {
        return sidecar_failure(SidecarError::InvalidFrame);
    }
    if (bytes.find('\r') != std::string_view::npos) {
        return sidecar_failure(SidecarError::TrailingData);
    }
    if (bytes.back() != '\n') {
        return sidecar_failure(SidecarError::NonCanonicalBytes);
    }

    const std::size_t first_end = bytes.find('\n');
    if (first_end == std::string_view::npos || bytes.substr(0, first_end) != kSidecarMagic) {
        return sidecar_failure(SidecarError::InvalidFrame);
    }
    const std::size_t second_begin = first_end + 1;
    const std::size_t second_end = bytes.find('\n', second_begin);
    if (second_end == std::string_view::npos) {
        return sidecar_failure(SidecarError::InvalidFrame);
    }
    const std::string_view digest_line = bytes.substr(second_begin, second_end - second_begin);
    if (!digest_line.starts_with(kSha256Prefix)) {
        return sidecar_failure(SidecarError::InvalidFrame);
    }
    const std::string_view digest = digest_line.substr(kSha256Prefix.size());
    if (validate_sha256_spelling(digest) != Sha256SpellingError::None) {
        return sidecar_failure(SidecarError::InvalidDigest);
    }

    const std::size_t third_begin = second_end + 1;
    const std::size_t third_end = bytes.find('\n', third_begin);
    if (third_end == std::string_view::npos) {
        return sidecar_failure(SidecarError::NonCanonicalBytes);
    }
    if (third_end != bytes.size() - 1) {
        return sidecar_failure(SidecarError::TrailingData);
    }
    const std::string_view size_line = bytes.substr(third_begin, third_end - third_begin);
    if (!size_line.starts_with(kSizePrefix)) {
        return sidecar_failure(SidecarError::InvalidFrame);
    }

    std::uint64_t size_bytes = 0;
    const SidecarError integer_error = parse_u64(size_line.substr(kSizePrefix.size()), size_bytes);
    if (integer_error != SidecarError::None) {
        return sidecar_failure(integer_error);
    }

    DigestSidecar sidecar;
    std::copy(digest.begin(), digest.end(), sidecar.sha256.begin());
    sidecar.size_bytes = size_bytes;
    return {.error = SidecarError::None, .sidecar = sidecar};
}

DigestSidecarEncodeResult encode_digest_sidecar(std::string_view sha256,
                                                std::uint64_t size_bytes) {
    if (validate_sha256_spelling(sha256) != Sha256SpellingError::None) {
        return {.error = SidecarError::InvalidDigest, .bytes = {}};
    }

    std::array<char, std::numeric_limits<std::uint64_t>::digits10 + 1> decimal{};
    const auto converted = std::to_chars(decimal.data(), decimal.data() + decimal.size(), size_bytes);

    DigestSidecarEncodeResult result;
    if (converted.ec != std::errc{}) {
        result.error = SidecarError::NumericRange;
        return result;
    }
    result.bytes.reserve(kMaximumDigestSidecarBytes);
    result.bytes.append(kSidecarMagic);
    result.bytes.push_back('\n');
    result.bytes.append(kSha256Prefix);
    result.bytes.append(sha256);
    result.bytes.push_back('\n');
    result.bytes.append(kSizePrefix);
    result.bytes.append(decimal.data(), converted.ptr);
    result.bytes.push_back('\n');
    return result;
}

} // namespace tatara::evidence
