#include "tatara/model/sha256.h"
#include "tatara/runtime/execution_identity.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

using namespace tatara;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

void canonical_encoding_has_a_frozen_digest() {
    runtime::ExecutionIdentityEncoder encoder("domain-A");
    encoder.append_u8(0xab);
    encoder.append_u32(0x78563412U);
    encoder.append_u64(0x0807060504030201ULL);
    encoder.append_bool(true);
    encoder.append_text("tatara");
    const std::array<std::byte, 3> bytes{
        std::byte{0x00}, std::byte{0xff}, std::byte{0x41}};
    encoder.append_bytes(bytes);
    runtime::PrefillCommandIdentity identity{};
    for (std::size_t index = 0; index < identity.size(); ++index) {
        identity[index] = static_cast<std::uint8_t>(index);
    }
    encoder.append_identity(identity);

    check(
        model::sha256_hex(encoder.finish()) ==
            "37c259445f829dd2463d151a332864751b0c46473fbb8d37bbbf11c5656988e9",
        "identity encoding is schema-versioned, little-endian and length-prefixed");
}

void variable_width_fields_cannot_alias() {
    runtime::ExecutionIdentityEncoder left("ambiguity");
    left.append_text("a");
    left.append_text("bc");
    runtime::ExecutionIdentityEncoder right("ambiguity");
    right.append_text("ab");
    right.append_text("c");
    check(left.finish() != right.finish(),
          "length-prefixed text fields separate concatenation ambiguities");

    runtime::ExecutionIdentityEncoder bytes_left("ambiguity");
    const std::array<std::byte, 1> a{std::byte{'a'}};
    const std::array<std::byte, 2> bc{std::byte{'b'}, std::byte{'c'}};
    bytes_left.append_bytes(a);
    bytes_left.append_bytes(bc);
    runtime::ExecutionIdentityEncoder bytes_right("ambiguity");
    const std::array<std::byte, 2> ab{std::byte{'a'}, std::byte{'b'}};
    const std::array<std::byte, 1> c{std::byte{'c'}};
    bytes_right.append_bytes(ab);
    bytes_right.append_bytes(c);
    check(bytes_left.finish() != bytes_right.finish(),
          "length-prefixed byte fields separate concatenation ambiguities");
}

void domains_and_boolean_values_are_separate() {
    runtime::ExecutionIdentityEncoder first_domain("first");
    first_domain.append_u32(7);
    runtime::ExecutionIdentityEncoder second_domain("second");
    second_domain.append_u32(7);
    check(first_domain.finish() != second_domain.finish(),
          "domain separation is part of every identity");

    runtime::ExecutionIdentityEncoder false_value("boolean");
    false_value.append_bool(false);
    runtime::ExecutionIdentityEncoder true_value("boolean");
    true_value.append_bool(true);
    check(false_value.finish() != true_value.finish(),
          "booleans have one canonical byte and remain distinct");
}

} // namespace

int main() {
    canonical_encoding_has_a_frozen_digest();
    variable_width_fields_cannot_alias();
    domains_and_boolean_values_are_separate();
    if (failures == 0) {
        std::printf("execution_identity: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
