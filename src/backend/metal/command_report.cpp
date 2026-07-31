#include "backend/metal/command_report.h"

#include <cstddef>

namespace tatara::backend::metal::detail {

MetalExecutionState execution_state_from_native(std::uint64_t status) noexcept {
    switch (status) {
    case kNativeStatusNotEnqueued:
        return MetalExecutionState::NotEnqueued;
    case kNativeStatusEnqueued:
        return MetalExecutionState::Enqueued;
    case kNativeStatusCommitted:
        return MetalExecutionState::Committed;
    case kNativeStatusScheduled:
        return MetalExecutionState::Scheduled;
    case kNativeStatusCompleted:
        return MetalExecutionState::Completed;
    case kNativeStatusError:
        return MetalExecutionState::Error;
    default:
        return MetalExecutionState::Unknown;
    }
}

MetalExecutionFailure execution_failure_from_native(MetalExecutionState state,
                                                    bool is_metal_error_domain,
                                                    std::int64_t error_code) noexcept {
    if (state != MetalExecutionState::Error) {
        return MetalExecutionFailure::UnexpectedStatus;
    }
    if (!is_metal_error_domain) {
        return MetalExecutionFailure::UnknownError;
    }
    switch (error_code) {
    case kNativeErrorInternal:
        return MetalExecutionFailure::Internal;
    case kNativeErrorTimeout:
        return MetalExecutionFailure::Timeout;
    case kNativeErrorPageFault:
        return MetalExecutionFailure::PageFault;
    case kNativeErrorAccessRevoked:
        return MetalExecutionFailure::AccessRevoked;
    case kNativeErrorNotPermitted:
        return MetalExecutionFailure::NotPermitted;
    case kNativeErrorOutOfMemory:
        return MetalExecutionFailure::OutOfMemory;
    case kNativeErrorInvalidResource:
        return MetalExecutionFailure::InvalidResource;
    case kNativeErrorMemoryless:
        return MetalExecutionFailure::Memoryless;
    case kNativeErrorDeviceRemoved:
        return MetalExecutionFailure::DeviceRemoved;
    case kNativeErrorStackOverflow:
        return MetalExecutionFailure::StackOverflow;
    default:
        return MetalExecutionFailure::UnknownError;
    }
}

MetalExecutionDiagnostic bounded_execution_diagnostic(const char* raw) noexcept {
    MetalExecutionDiagnostic diagnostic;
    if (raw == nullptr) {
        return diagnostic;
    }
    const std::size_t maximum = diagnostic.bytes.size() - 1;
    while (diagnostic.length_bytes < maximum && raw[diagnostic.length_bytes] != '\0') {
        diagnostic.bytes[diagnostic.length_bytes] = raw[diagnostic.length_bytes];
        ++diagnostic.length_bytes;
    }
    diagnostic.bytes[diagnostic.length_bytes] = '\0';
    diagnostic.truncated = diagnostic.length_bytes == maximum && raw[maximum] != '\0';
    return diagnostic;
}

} // namespace tatara::backend::metal::detail
