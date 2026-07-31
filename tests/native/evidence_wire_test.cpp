#include "tatara/evidence/wire.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using tatara::evidence::AggregateError;
using tatara::evidence::IdentifierError;
using tatara::evidence::RelativePathError;
using tatara::evidence::Sha256SpellingError;
using tatara::evidence::SidecarError;
using tatara::evidence::decode_digest_sidecar;
using tatara::evidence::encode_digest_sidecar;
using tatara::evidence::schema_v1_aggregate_digest;
using tatara::evidence::validate_identifier;
using tatara::evidence::validate_relative_path;
using tatara::evidence::validate_sha256_spelling;
using tatara::model::Sha256;
using tatara::model::sha256_hex;

constexpr std::string_view kEmptySha256 =
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
constexpr std::string_view kSidecar =
    "tatara-sha256-sidecar-v1\n"
    "sha256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855\n"
    "size_bytes=0\n";

void check(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

std::span<const std::byte> bytes_of(std::string_view value) {
    return {reinterpret_cast<const std::byte*>(value.data()), value.size()};
}

std::string sha256_of(std::string_view value) {
    Sha256 hasher;
    hasher.update(bytes_of(value));
    return sha256_hex(hasher.finish());
}

std::string digest_of(std::span<const std::span<const std::byte>> elements) {
    const auto result =
        schema_v1_aggregate_digest("tatara.aggregate-test.v1", elements);
    check(result.error == AggregateError::None, "aggregate unexpectedly failed");
    return sha256_hex(result.digest);
}

void check_text_validators() {
    check(validate_identifier("") == IdentifierError::Empty, "identifier empty");
    check(validate_identifier("a") == IdentifierError::None, "identifier one byte");
    check(validate_identifier("Tatara.v0_1-rc2") == IdentifierError::None,
          "identifier grammar");
    check(validate_identifier("_bad") == IdentifierError::InvalidCharacter,
          "identifier leading punctuation");
    check(validate_identifier("a bad") == IdentifierError::InvalidCharacter,
          "identifier space");
    check(validate_identifier("\xc3\xa9") == IdentifierError::InvalidCharacter,
          "identifier non-ASCII");

    const std::string identifier_max(tatara::evidence::kMaximumIdentifierBytes, 'a');
    check(validate_identifier(identifier_max) == IdentifierError::None,
          "identifier exact maximum");
    check(validate_identifier(identifier_max + "a") == IdentifierError::TooLong,
          "identifier maximum plus one");

    check(validate_sha256_spelling(kEmptySha256) == Sha256SpellingError::None,
          "digest canonical spelling");
    check(validate_sha256_spelling("") == Sha256SpellingError::WrongLength,
          "digest zero bytes");
    check(validate_sha256_spelling(kEmptySha256.substr(0, 63)) ==
              Sha256SpellingError::WrongLength,
          "digest short");
    check(validate_sha256_spelling(std::string(kEmptySha256) + "0") ==
              Sha256SpellingError::WrongLength,
          "digest long");
    std::string uppercase_digest(kEmptySha256);
    uppercase_digest.front() = 'E';
    check(validate_sha256_spelling(uppercase_digest) ==
              Sha256SpellingError::InvalidCharacter,
          "digest uppercase");

    check(validate_relative_path("") == RelativePathError::Empty, "path empty");
    check(validate_relative_path("a") == RelativePathError::None, "path one component");
    check(validate_relative_path("models/\xc3\xa9") == RelativePathError::None,
          "path valid UTF-8");
    check(validate_relative_path("/a") == RelativePathError::Absolute, "path absolute");
    check(validate_relative_path("a/") == RelativePathError::EmptyComponent,
          "path trailing separator");
    check(validate_relative_path("a//b") == RelativePathError::EmptyComponent,
          "path empty component");
    check(validate_relative_path(".") == RelativePathError::DotComponent,
          "path dot component");
    check(validate_relative_path("a/..") == RelativePathError::ParentComponent,
          "path parent component");
    check(validate_relative_path("a\\b") == RelativePathError::None,
          "path reverse solidus is an ordinary component byte");
    const std::string path_with_nul{"a\0b", 3};
    check(validate_relative_path(path_with_nul) == RelativePathError::ForbiddenCharacter,
          "path NUL");
    check(validate_relative_path("a\rb") == RelativePathError::ForbiddenCharacter,
          "path carriage return");
    check(validate_relative_path("a\nb") == RelativePathError::ForbiddenCharacter,
          "path newline");
    const std::string invalid_utf8{"\xc0\xaf", 2};
    check(validate_relative_path(invalid_utf8) == RelativePathError::InvalidUtf8,
          "path invalid UTF-8");

    const std::string path_max(tatara::evidence::kMaximumRelativePathBytes, 'a');
    check(validate_relative_path(path_max) == RelativePathError::None,
          "path exact maximum");
    check(validate_relative_path(path_max + "a") == RelativePathError::TooLong,
          "path maximum plus one");
}

void check_aggregate_vectors() {
    const std::array<std::span<const std::byte>, 0> no_elements{};
    check(digest_of(no_elements) ==
              "c683fc94bd40fcf7674cbea9315bda4c49fcca28afb1041e64515ffdf7959f75",
          "zero-element aggregate");

    const std::string_view empty;
    constexpr std::string_view abc = "abc";
    const std::array ag001 = {bytes_of(empty), bytes_of(abc)};
    check(digest_of(ag001) ==
              "c9984f744200f8c965f6a9a4db9b42685db991cacd7ad7b2ebdd2fb5f0974373",
          "AG-001");

    const std::array ag002 = {bytes_of(abc), bytes_of(empty)};
    check(digest_of(ag002) != digest_of(ag001), "AG-002");

    constexpr std::string_view a = "a";
    constexpr std::string_view bc = "bc";
    constexpr std::string_view ab = "ab";
    constexpr std::string_view c = "c";
    const std::array ag003_left = {bytes_of(a), bytes_of(bc)};
    const std::array ag003_right = {bytes_of(ab), bytes_of(c)};
    check(digest_of(ag003_left) != digest_of(ag003_right), "AG-003");

    const auto empty_domain = schema_v1_aggregate_digest("", no_elements);
    check(empty_domain.error == AggregateError::EmptyDomain, "aggregate empty domain");
    const std::string domain_with_nul{"tatara\0v1", 9};
    const auto invalid_domain = schema_v1_aggregate_digest(domain_with_nul, no_elements);
    check(invalid_domain.error == AggregateError::InvalidDomain,
          "aggregate embedded-NUL domain");
    const auto unknown_domain = schema_v1_aggregate_digest("x", no_elements);
    check(unknown_domain.error == AggregateError::InvalidDomain,
          "aggregate unknown schema domain");
}

void check_sidecar_vectors() {
    static_assert(kSidecar.size() == 110);

    const auto sc001 = decode_digest_sidecar(kSidecar);
    check(sc001.error == SidecarError::None, "SC-001 decode");
    check(sc001.sidecar.sha256_text() == kEmptySha256, "SC-001 digest");
    check(sc001.sidecar.size_bytes == 0, "SC-001 size");
    check(sha256_of(kSidecar) ==
              "abc5b81a869cf362a7521c348978b9546744d007dd5d25614cc356747313cda9",
          "SC-001 sidecar digest");

    const auto encoded = encode_digest_sidecar(kEmptySha256, 0);
    check(encoded.error == SidecarError::None && encoded.bytes == kSidecar,
          "SC-001 encode");

    std::string sc002(kSidecar);
    sc002[kSidecar.find(kEmptySha256)] = 'E';
    check(decode_digest_sidecar(sc002).error == SidecarError::InvalidDigest, "SC-002");

    std::string sc003(kSidecar);
    sc003.replace(sc003.find("size_bytes=0"), std::string_view{"size_bytes=0"}.size(),
                  "size_bytes=00");
    check(decode_digest_sidecar(sc003).error == SidecarError::InvalidInteger, "SC-003");

    std::string sc004(kSidecar);
    sc004.replace(sc004.find('\n'), 1, "\r\n");
    check(decode_digest_sidecar(sc004).error == SidecarError::TrailingData, "SC-004");

    std::string sc005(kSidecar);
    sc005.append("empty.bin\n");
    check(decode_digest_sidecar(sc005).error == SidecarError::TrailingData, "SC-005");

    check(decode_digest_sidecar("").error == SidecarError::InvalidFrame,
          "sidecar zero-byte input");

    std::string exact_bound(kSidecar);
    exact_bound.append(tatara::evidence::kMaximumDigestSidecarBytes - exact_bound.size(), 'x');
    check(exact_bound.size() == tatara::evidence::kMaximumDigestSidecarBytes,
          "sidecar exact-bound fixture");
    check(decode_digest_sidecar(exact_bound).error == SidecarError::NonCanonicalBytes,
          "sidecar exact bound is admitted before syntax rejection");
    exact_bound.push_back('x');
    check(decode_digest_sidecar(exact_bound).error == SidecarError::ArtifactTooLarge,
          "sidecar maximum plus one");

    std::string missing_final_lf(kSidecar);
    missing_final_lf.pop_back();
    check(decode_digest_sidecar(missing_final_lf).error == SidecarError::NonCanonicalBytes,
          "sidecar missing final LF");

    std::string overflow(kSidecar);
    overflow.replace(overflow.find("size_bytes=0"), std::string_view{"size_bytes=0"}.size(),
                     "size_bytes=18446744073709551616");
    check(decode_digest_sidecar(overflow).error == SidecarError::NumericRange,
          "sidecar U64 overflow");

    const auto maximum =
        encode_digest_sidecar(kEmptySha256, std::numeric_limits<std::uint64_t>::max());
    check(maximum.error == SidecarError::None, "sidecar U64 maximum encode");
    check(maximum.bytes.size() <= tatara::evidence::kMaximumDigestSidecarBytes,
          "sidecar encoded bound");
    const auto maximum_round_trip = decode_digest_sidecar(maximum.bytes);
    check(maximum_round_trip.error == SidecarError::None &&
              maximum_round_trip.sidecar.size_bytes ==
                  std::numeric_limits<std::uint64_t>::max(),
          "sidecar U64 maximum round trip");

    check(encode_digest_sidecar("bad", 0).error == SidecarError::InvalidDigest,
          "sidecar invalid digest encode");
}

} // namespace

int main() {
    check_text_validators();
    check_aggregate_vectors();
    check_sidecar_vectors();
    return 0;
}
