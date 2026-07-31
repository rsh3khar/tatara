#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"

#include <string_view>
#include <type_traits>
#include <utility>

namespace {

using tatara::backend::metal::begin_blit_pass;
using tatara::backend::metal::begin_compute_pass;
using tatara::backend::metal::commit;
using tatara::backend::metal::create_command_buffer;
using tatara::backend::metal::create_compute_pipeline;
using tatara::backend::metal::create_function;
using tatara::backend::metal::create_library_with_source;
using tatara::backend::metal::dispatch_threadgroups;
using tatara::backend::metal::dispatch_threadgroups_indirect;
using tatara::backend::metal::end_compute_pass;
using tatara::backend::metal::kExecutionDiagnosticStorageBytes;
using tatara::backend::metal::MetalBlitPass;
using tatara::backend::metal::MetalBuffer;
using tatara::backend::metal::MetalCommandBuffer;
using tatara::backend::metal::MetalCommandError;
using tatara::backend::metal::MetalCommandQueue;
using tatara::backend::metal::MetalComputePass;
using tatara::backend::metal::MetalComputePipeline;
using tatara::backend::metal::MetalDevice;
using tatara::backend::metal::MetalExecutionFailure;
using tatara::backend::metal::MetalExecutionState;
using tatara::backend::metal::MetalFunction;
using tatara::backend::metal::MetalLibrary;
using tatara::backend::metal::MetalPendingExecution;
using tatara::backend::metal::MetalPipelineError;
using tatara::backend::metal::MetalSize;
using tatara::backend::metal::set_buffer;
using tatara::backend::metal::set_compute_pipeline;
using tatara::backend::metal::wait_until_completed;
using tatara::backend::metal::wait_until_completed_timed;

template <typename Resource> consteval bool move_only() {
    return !std::is_copy_constructible_v<Resource> && !std::is_copy_assignable_v<Resource> &&
           std::is_nothrow_move_constructible_v<Resource> &&
           std::is_nothrow_move_assignable_v<Resource>;
}

} // namespace

int main() {
    static_assert(move_only<MetalLibrary>());
    static_assert(move_only<MetalFunction>());
    static_assert(move_only<MetalComputePipeline>());
    static_assert(move_only<MetalCommandBuffer>());
    static_assert(move_only<MetalComputePass>());
    static_assert(move_only<MetalBlitPass>());
    static_assert(move_only<MetalPendingExecution>());
    static_assert(sizeof(MetalCommandBuffer) == sizeof(void*));
    static_assert(sizeof(MetalComputePass) == 2 * sizeof(void*));
    static_assert(sizeof(MetalBlitPass) == 2 * sizeof(void*));
    static_assert(sizeof(MetalPendingExecution) == sizeof(void*));
    static_assert(kExecutionDiagnosticStorageBytes == 512);

    MetalLibrary library;
    MetalFunction function;
    MetalComputePipeline pipeline;
    MetalCommandBuffer command_buffer;
    MetalComputePass compute_pass;
    MetalBlitPass blit_pass;
    MetalPendingExecution pending_execution;
    if (library || function || pipeline || command_buffer || compute_pass || blit_pass ||
        pending_execution) {
        return 1;
    }

    const MetalDevice device;
    if (create_library_with_source(device, "kernel void k() {}").error !=
        MetalPipelineError::InvalidDevice) {
        return 2;
    }
    if (create_library_with_source(device, "").error != MetalPipelineError::InvalidDevice) {
        return 3;
    }
    if (create_function(library, "k").error != MetalPipelineError::InvalidLibrary) {
        return 4;
    }
    if (create_function(library, "").error != MetalPipelineError::InvalidLibrary) {
        return 5;
    }
    if (create_compute_pipeline(device, function).error != MetalPipelineError::InvalidDevice) {
        return 6;
    }

    const MetalCommandQueue command_queue;
    if (create_command_buffer(command_queue).error != MetalCommandError::InvalidCommandQueue) {
        return 7;
    }
    if (begin_compute_pass(std::move(command_buffer)).error !=
        MetalCommandError::InvalidCommandBuffer) {
        return 8;
    }
    MetalCommandBuffer blit_command_buffer;
    if (begin_blit_pass(std::move(blit_command_buffer)).error !=
        MetalCommandError::InvalidCommandBuffer) {
        return 20;
    }
    if (set_compute_pipeline(compute_pass, pipeline) != MetalCommandError::InvalidComputePass) {
        return 9;
    }
    const MetalBuffer buffer;
    if (set_buffer(compute_pass, buffer, 0, 0) != MetalCommandError::InvalidComputePass) {
        return 10;
    }
    constexpr MetalSize kUnitExtent{.width = 1, .height = 1, .depth = 1};
    if (dispatch_threadgroups(compute_pass, kUnitExtent, kUnitExtent) !=
        MetalCommandError::InvalidComputePass) {
        return 11;
    }
    if (dispatch_threadgroups_indirect(compute_pass, buffer, 0, kUnitExtent) !=
        MetalCommandError::InvalidComputePass) {
        return 19;
    }
    if (memory_barrier(compute_pass) != MetalCommandError::InvalidComputePass) {
        return 17;
    }
    const std::uint32_t constant_value = 0;
    if (set_bytes(compute_pass, &constant_value, 4, 0) != MetalCommandError::InvalidComputePass) {
        return 18;
    }
    if (end_compute_pass(std::move(compute_pass)).error != MetalCommandError::InvalidComputePass) {
        return 12;
    }

    MetalCommandBuffer second_command_buffer;
    if (commit(std::move(second_command_buffer)).error != MetalCommandError::InvalidCommandBuffer) {
        return 13;
    }
    auto execution = wait_until_completed(std::move(pending_execution));
    if (execution.error != MetalCommandError::InvalidPendingExecution || execution.completed ||
        execution.state != MetalExecutionState::NotObserved ||
        execution.failure != MetalExecutionFailure::None || execution.has_native_error_code ||
        execution.native_error_code != 0 || !execution.failure_description.empty() ||
        execution.failure_description.truncated ||
        execution.failure_description.view() != std::string_view{} ||
        execution.failure_description.c_str()[0] != '\0' || static_cast<bool>(execution)) {
        return 14;
    }
    MetalPendingExecution timed_pending_execution;
    auto timed_execution = wait_until_completed_timed(std::move(timed_pending_execution));
    if (timed_execution.error != MetalCommandError::InvalidPendingExecution ||
        timed_execution.completed || timed_execution.state != MetalExecutionState::NotObserved ||
        timed_execution.failure != MetalExecutionFailure::None ||
        timed_execution.has_native_error_code || timed_execution.native_error_code != 0 ||
        !timed_execution.failure_description.empty() ||
        timed_execution.failure_description.truncated || static_cast<bool>(timed_execution)) {
        return 21;
    }

    MetalLibrary moved_library = std::move(library);
    MetalComputePass moved_pass = std::move(compute_pass);
    MetalBlitPass moved_blit_pass = std::move(blit_pass);
    MetalCommandBuffer moved_command_buffer = std::move(command_buffer);
    MetalPendingExecution moved_pending_execution = std::move(pending_execution);
    return moved_library || moved_pass || moved_blit_pass || moved_command_buffer ||
                   moved_pending_execution
               ? 15
               : 0;
}
