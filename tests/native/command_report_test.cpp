#include "backend/metal/command_report.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace {

using tatara::backend::metal::kExecutionDiagnosticStorageBytes;
using tatara::backend::metal::MetalExecutionFailure;
using tatara::backend::metal::MetalExecutionState;
using tatara::backend::metal::detail::bounded_execution_diagnostic;
using tatara::backend::metal::detail::execution_failure_from_native;
using tatara::backend::metal::detail::execution_state_from_native;

struct StateCase {
    std::uint64_t native;
    MetalExecutionState expected;
};

struct FailureCase {
    std::int64_t native;
    MetalExecutionFailure expected;
};

} // namespace

int main() {
    constexpr std::array<StateCase, 6> kStates{{
        {0, MetalExecutionState::NotEnqueued},
        {1, MetalExecutionState::Enqueued},
        {2, MetalExecutionState::Committed},
        {3, MetalExecutionState::Scheduled},
        {4, MetalExecutionState::Completed},
        {5, MetalExecutionState::Error},
    }};
    for (const StateCase& item : kStates) {
        if (execution_state_from_native(item.native) != item.expected) {
            return 1;
        }
    }
    if (execution_state_from_native(6) != MetalExecutionState::Unknown ||
        execution_state_from_native(std::numeric_limits<std::uint64_t>::max()) !=
            MetalExecutionState::Unknown) {
        return 2;
    }

    constexpr std::array<FailureCase, 10> kFailures{{
        {1, MetalExecutionFailure::Internal},
        {2, MetalExecutionFailure::Timeout},
        {3, MetalExecutionFailure::PageFault},
        {4, MetalExecutionFailure::AccessRevoked},
        {7, MetalExecutionFailure::NotPermitted},
        {8, MetalExecutionFailure::OutOfMemory},
        {9, MetalExecutionFailure::InvalidResource},
        {10, MetalExecutionFailure::Memoryless},
        {11, MetalExecutionFailure::DeviceRemoved},
        {12, MetalExecutionFailure::StackOverflow},
    }};
    for (const FailureCase& item : kFailures) {
        if (execution_failure_from_native(MetalExecutionState::Error, true, item.native) !=
            item.expected) {
            return 3;
        }
    }
    for (const std::int64_t unknown : {0, 5, 6, 13, -1}) {
        if (execution_failure_from_native(MetalExecutionState::Error, true, unknown) !=
            MetalExecutionFailure::UnknownError) {
            return 4;
        }
    }
    if (execution_failure_from_native(MetalExecutionState::Error, false, 8) !=
            MetalExecutionFailure::UnknownError ||
        execution_failure_from_native(MetalExecutionState::Completed, true, 0) !=
            MetalExecutionFailure::UnexpectedStatus ||
        execution_failure_from_native(MetalExecutionState::Unknown, false, 0) !=
            MetalExecutionFailure::UnexpectedStatus) {
        return 5;
    }

    const auto absent = bounded_execution_diagnostic(nullptr);
    if (!absent.empty() || absent.truncated || absent.c_str()[0] != '\0') {
        return 6;
    }
    const auto short_text = bounded_execution_diagnostic("tatara");
    if (short_text.view() != std::string_view{"tatara"} || short_text.truncated ||
        short_text.c_str()[short_text.length_bytes] != '\0') {
        return 7;
    }

    std::array<char, kExecutionDiagnosticStorageBytes> exact{};
    exact.fill('x');
    exact.back() = '\0';
    const auto exact_result = bounded_execution_diagnostic(exact.data());
    if (exact_result.length_bytes != kExecutionDiagnosticStorageBytes - 1 ||
        exact_result.truncated ||
        exact_result.c_str()[kExecutionDiagnosticStorageBytes - 1] != '\0') {
        return 8;
    }

    std::array<char, kExecutionDiagnosticStorageBytes + 1> oversized{};
    oversized.fill('y');
    oversized.back() = '\0';
    const auto oversized_result = bounded_execution_diagnostic(oversized.data());
    if (oversized_result.length_bytes != kExecutionDiagnosticStorageBytes - 1 ||
        !oversized_result.truncated ||
        oversized_result.c_str()[kExecutionDiagnosticStorageBytes - 1] != '\0') {
        return 9;
    }
    for (std::size_t index = 0; index < kExecutionDiagnosticStorageBytes - 1; ++index) {
        if (oversized_result.bytes[index] != 'y') {
            return 10;
        }
    }
    auto malformed_public_length = short_text;
    malformed_public_length.length_bytes = std::numeric_limits<std::size_t>::max();
    if (malformed_public_length.view().size() != kExecutionDiagnosticStorageBytes - 1) {
        return 11;
    }
    return 0;
}
