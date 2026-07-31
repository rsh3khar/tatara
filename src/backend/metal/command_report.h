#pragma once

#include "tatara/backend/metal/commands.h"

#include <cstdint>

namespace tatara::backend::metal::detail {

// Stable native values consumed by the pure classifier. commands.mm pins each
// one to the currently building Metal SDK with compile-time assertions.
inline constexpr std::uint64_t kNativeStatusNotEnqueued = 0;
inline constexpr std::uint64_t kNativeStatusEnqueued = 1;
inline constexpr std::uint64_t kNativeStatusCommitted = 2;
inline constexpr std::uint64_t kNativeStatusScheduled = 3;
inline constexpr std::uint64_t kNativeStatusCompleted = 4;
inline constexpr std::uint64_t kNativeStatusError = 5;

inline constexpr std::int64_t kNativeErrorNone = 0;
inline constexpr std::int64_t kNativeErrorInternal = 1;
inline constexpr std::int64_t kNativeErrorTimeout = 2;
inline constexpr std::int64_t kNativeErrorPageFault = 3;
inline constexpr std::int64_t kNativeErrorAccessRevoked = 4;
inline constexpr std::int64_t kNativeErrorNotPermitted = 7;
inline constexpr std::int64_t kNativeErrorOutOfMemory = 8;
inline constexpr std::int64_t kNativeErrorInvalidResource = 9;
inline constexpr std::int64_t kNativeErrorMemoryless = 10;
inline constexpr std::int64_t kNativeErrorDeviceRemoved = 11;
inline constexpr std::int64_t kNativeErrorStackOverflow = 12;

MetalExecutionState execution_state_from_native(std::uint64_t status) noexcept;

MetalExecutionFailure execution_failure_from_native(MetalExecutionState state,
                                                    bool is_metal_error_domain,
                                                    std::int64_t error_code) noexcept;

MetalExecutionDiagnostic bounded_execution_diagnostic(const char* raw) noexcept;

} // namespace tatara::backend::metal::detail
