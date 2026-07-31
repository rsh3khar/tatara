#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace tatara::runtime {

struct CheckedU64 {
    std::uint64_t value{0};
    bool ok{false};

    explicit constexpr operator bool() const noexcept {
        return ok;
    }
};

constexpr CheckedU64 checked_u64_add(std::uint64_t left, std::uint64_t right) noexcept {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        return {};
    }
    return {.value = left + right, .ok = true};
}

constexpr CheckedU64 checked_u64_multiply(std::uint64_t left, std::uint64_t right) noexcept {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
        return {};
    }
    return {.value = left * right, .ok = true};
}

constexpr CheckedU64 checked_u64_ceil_divide(std::uint64_t dividend,
                                             std::uint64_t divisor) noexcept {
    if (divisor == 0) {
        return {};
    }
    const std::uint64_t quotient = dividend / divisor;
    const std::uint64_t remainder = dividend % divisor;
    return {.value = quotient + (remainder == 0 ? 0u : 1u), .ok = true};
}

constexpr CheckedU64 checked_u64_align_up(std::uint64_t value,
                                          std::uint64_t alignment) noexcept {
    if (alignment == 0) {
        return {};
    }
    const std::uint64_t remainder = value % alignment;
    if (remainder == 0) {
        return {.value = value, .ok = true};
    }
    return checked_u64_add(value, alignment - remainder);
}

template <typename Target> struct CheckedNarrow {
    Target value{0};
    bool ok{false};

    explicit constexpr operator bool() const noexcept {
        return ok;
    }
};

template <typename Target>
constexpr CheckedNarrow<Target> checked_u64_narrow(std::uint64_t value) noexcept {
    static_assert(std::is_integral_v<Target>);
    static_assert(std::is_unsigned_v<Target>);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<Target>::max())) {
        return {};
    }
    return {.value = static_cast<Target>(value), .ok = true};
}

} // namespace tatara::runtime
