#pragma once

#include "tatara/model/sha256.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace tatara::evidence {

inline constexpr std::size_t kMaximumIdentifierBytes = 128;
inline constexpr std::size_t kMaximumRelativePathBytes = 1024;
inline constexpr std::size_t kSha256TextBytes = 64;
inline constexpr std::size_t kMaximumDigestSidecarBytes = 160;

enum class IdentifierError : std::uint8_t {
    None,
    Empty,
    TooLong,
    InvalidCharacter,
};

enum class Sha256SpellingError : std::uint8_t {
    None,
    WrongLength,
    InvalidCharacter,
};

enum class RelativePathError : std::uint8_t {
    None,
    Empty,
    TooLong,
    InvalidUtf8,
    Absolute,
    EmptyComponent,
    DotComponent,
    ParentComponent,
    ForbiddenCharacter,
};

IdentifierError validate_identifier(std::string_view value) noexcept;
Sha256SpellingError validate_sha256_spelling(std::string_view value) noexcept;
RelativePathError validate_relative_path(std::string_view value) noexcept;

enum class AggregateError : std::uint8_t {
    None,
    EmptyDomain,
    InvalidDomain,
    ElementSizeOverflow,
};

struct AggregateDigestResult {
    AggregateError error{AggregateError::None};
    model::Sha256Digest digest{};

    explicit operator bool() const noexcept {
        return error == AggregateError::None;
    }
};

// Hashes the schema-v1 aggregate framing directly into the incremental
// hasher. Element storage remains caller-owned for the duration of this call.
AggregateDigestResult
schema_v1_aggregate_digest(std::string_view domain,
                           std::span<const std::span<const std::byte>> elements) noexcept;

struct DigestSidecar {
    std::array<char, kSha256TextBytes> sha256{};
    std::uint64_t size_bytes{0};

    std::string_view sha256_text() const noexcept {
        return {sha256.data(), sha256.size()};
    }
};

enum class SidecarError : std::uint8_t {
    None,
    ArtifactTooLarge,
    InvalidFrame,
    InvalidDigest,
    InvalidInteger,
    NumericRange,
    NonCanonicalBytes,
    TrailingData,
};

struct DigestSidecarDecodeResult {
    SidecarError error{SidecarError::None};
    DigestSidecar sidecar{};

    explicit operator bool() const noexcept {
        return error == SidecarError::None;
    }
};

struct DigestSidecarEncodeResult {
    SidecarError error{SidecarError::None};
    std::string bytes;

    explicit operator bool() const noexcept {
        return error == SidecarError::None;
    }
};

DigestSidecarDecodeResult decode_digest_sidecar(std::string_view bytes) noexcept;
DigestSidecarEncodeResult encode_digest_sidecar(std::string_view sha256,
                                                std::uint64_t size_bytes);

} // namespace tatara::evidence
