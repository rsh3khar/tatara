#include "tatara/backend/metal/pipeline.h"

#include "backend/metal/resource_storage.h"

#include <utility>

namespace tatara::backend::metal {
namespace {

NSString* utf8_string(std::string_view text) {
    return [[NSString alloc] initWithBytes:text.data()
                                    length:text.size()
                                  encoding:NSUTF8StringEncoding];
}

std::string error_description(NSError* error) {
    if (error == nil) {
        return {};
    }
    const char* raw = error.localizedDescription.UTF8String;
    return raw == nullptr ? std::string{} : std::string{raw};
}

bool invalid_text(std::string_view text) {
    return text.empty() || text.find('\0') != std::string_view::npos;
}

} // namespace

MetalLibrary::MetalLibrary() noexcept = default;
MetalLibrary::~MetalLibrary() = default;
MetalLibrary::MetalLibrary(MetalLibrary&&) noexcept = default;
MetalLibrary& MetalLibrary::operator=(MetalLibrary&&) noexcept = default;

MetalLibrary::MetalLibrary(std::unique_ptr<Storage> storage) : storage_(std::move(storage)) {}

MetalLibrary::operator bool() const noexcept {
    return storage_ != nullptr && storage_->object;
}

MetalFunction::MetalFunction() noexcept = default;
MetalFunction::~MetalFunction() = default;
MetalFunction::MetalFunction(MetalFunction&&) noexcept = default;
MetalFunction& MetalFunction::operator=(MetalFunction&&) noexcept = default;

MetalFunction::MetalFunction(std::unique_ptr<Storage> storage) : storage_(std::move(storage)) {}

MetalFunction::operator bool() const noexcept {
    return storage_ != nullptr && storage_->object;
}

MetalComputePipeline::MetalComputePipeline() noexcept = default;
MetalComputePipeline::~MetalComputePipeline() = default;
MetalComputePipeline::MetalComputePipeline(MetalComputePipeline&&) noexcept = default;
MetalComputePipeline& MetalComputePipeline::operator=(MetalComputePipeline&&) noexcept = default;

MetalComputePipeline::MetalComputePipeline(std::unique_ptr<Storage> storage)
    : storage_(std::move(storage)) {}

MetalComputePipeline::operator bool() const noexcept {
    return storage_ != nullptr && storage_->object;
}

MetalLibraryResult create_library_with_source(const MetalDevice& device, std::string_view source) {
    if (!device) {
        return {.error = MetalPipelineError::InvalidDevice, .library = std::nullopt};
    }
    if (invalid_text(source)) {
        return {.error = MetalPipelineError::InvalidSource, .library = std::nullopt};
    }
    @autoreleasepool {
        NSString* native_source = utf8_string(source);
        if (native_source == nil) {
            return {.error = MetalPipelineError::InvalidSource, .library = std::nullopt};
        }
        id<MTLDevice> native_device = (__bridge id<MTLDevice>)device.storage_->object.get();
        NSError* error = nil;
        // The sealed arithmetic contract compiles under safe math
        // (TATARA-EXEC6-MATH1): the champion library sets MTLMathModeSafe,
        // and the fast-math default is a global low-bit divergence that
        // self-consistent fixtures cannot see.
        MTLCompileOptions* options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        id<MTLLibrary> library = [native_device newLibraryWithSource:native_source
                                                             options:options
                                                               error:&error];
        if (library == nil) {
            return {
                .error = MetalPipelineError::LibraryCompilationFailed,
                .library = std::nullopt,
                .failure_description = error_description(error),
            };
        }
        auto owner = retain_object(library);
        if (!owner) {
            return {.error = MetalPipelineError::OwnershipFailure, .library = std::nullopt};
        }
        auto storage = std::make_unique<MetalLibrary::Storage>();
        storage->object = std::move(*owner);
        return {
            .error = MetalPipelineError::None,
            .library = MetalLibrary(std::move(storage)),
        };
    }
}

MetalFunctionResult create_function(const MetalLibrary& library, std::string_view name) {
    if (!library) {
        return {.error = MetalPipelineError::InvalidLibrary, .function = std::nullopt};
    }
    if (invalid_text(name)) {
        return {.error = MetalPipelineError::InvalidFunctionName, .function = std::nullopt};
    }
    @autoreleasepool {
        NSString* native_name = utf8_string(name);
        if (native_name == nil) {
            return {.error = MetalPipelineError::InvalidFunctionName, .function = std::nullopt};
        }
        id<MTLLibrary> native_library = (__bridge id<MTLLibrary>)library.storage_->object.get();
        id<MTLFunction> function = [native_library newFunctionWithName:native_name];
        if (function == nil) {
            return {.error = MetalPipelineError::FunctionLookupFailed, .function = std::nullopt};
        }
        auto owner = retain_object(function);
        if (!owner) {
            return {.error = MetalPipelineError::OwnershipFailure, .function = std::nullopt};
        }
        auto storage = std::make_unique<MetalFunction::Storage>();
        storage->object = std::move(*owner);
        return {
            .error = MetalPipelineError::None,
            .function = MetalFunction(std::move(storage)),
        };
    }
}

MetalComputePipelineResult create_compute_pipeline(const MetalDevice& device,
                                                   const MetalFunction& function) {
    if (!device) {
        return {.error = MetalPipelineError::InvalidDevice, .pipeline = std::nullopt};
    }
    if (!function) {
        return {.error = MetalPipelineError::InvalidFunction, .pipeline = std::nullopt};
    }
    @autoreleasepool {
        id<MTLDevice> native_device = (__bridge id<MTLDevice>)device.storage_->object.get();
        id<MTLFunction> native_function = (__bridge id<MTLFunction>)function.storage_->object.get();
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [native_device newComputePipelineStateWithFunction:native_function error:&error];
        if (pipeline == nil) {
            return {
                .error = MetalPipelineError::PipelineCreationFailed,
                .pipeline = std::nullopt,
                .failure_description = error_description(error),
            };
        }
        auto owner = retain_object(pipeline);
        if (!owner) {
            return {.error = MetalPipelineError::OwnershipFailure, .pipeline = std::nullopt};
        }
        auto storage = std::make_unique<MetalComputePipeline::Storage>();
        storage->object = std::move(*owner);
        return {
            .error = MetalPipelineError::None,
            .pipeline = MetalComputePipeline(std::move(storage)),
        };
    }
}

MetalComputePipelineResult create_indirect_compute_pipeline(
    const MetalDevice& device, const MetalFunction& function) {
    if (!device) {
        return {
            .error = MetalPipelineError::InvalidDevice,
            .pipeline = std::nullopt,
        };
    }
    if (!function) {
        return {
            .error = MetalPipelineError::InvalidFunction,
            .pipeline = std::nullopt,
        };
    }
    @autoreleasepool {
        id<MTLDevice> native_device =
            (__bridge id<MTLDevice>)device.storage_->object.get();
        id<MTLFunction> native_function =
            (__bridge id<MTLFunction>)function.storage_->object.get();
        MTLComputePipelineDescriptor* descriptor =
            [MTLComputePipelineDescriptor new];
        descriptor.computeFunction = native_function;
        descriptor.supportIndirectCommandBuffers = YES;
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [native_device
                newComputePipelineStateWithDescriptor:descriptor
                                              options:MTLPipelineOptionNone
                                           reflection:nil
                                                error:&error];
        if (pipeline == nil || !pipeline.supportIndirectCommandBuffers) {
            return {
                .error = MetalPipelineError::PipelineCreationFailed,
                .pipeline = std::nullopt,
                .failure_description = error_description(error),
            };
        }
        auto owner = retain_object(pipeline);
        if (!owner) {
            return {
                .error = MetalPipelineError::OwnershipFailure,
                .pipeline = std::nullopt,
            };
        }
        auto storage = std::make_unique<MetalComputePipeline::Storage>();
        storage->object = std::move(*owner);
        return {
            .error = MetalPipelineError::None,
            .pipeline = MetalComputePipeline(std::move(storage)),
        };
    }
}

bool supports_indirect_commands(
    const MetalComputePipeline& pipeline) noexcept {
    if (!pipeline) {
        return false;
    }
    id<MTLComputePipelineState> native_pipeline =
        (__bridge id<MTLComputePipelineState>)
            pipeline.storage_->object.get();
    return native_pipeline.supportIndirectCommandBuffers;
}

std::uint64_t compute_pipeline_identity(
    const MetalComputePipeline& pipeline) noexcept {
    if (!pipeline) {
        return 0;
    }
    id<MTLComputePipelineState> native_pipeline =
        (__bridge id<MTLComputePipelineState>)
            pipeline.storage_->object.get();
    return static_cast<std::uint64_t>(
        reinterpret_cast<std::uintptr_t>(
            (__bridge void*)native_pipeline));
}

} // namespace tatara::backend::metal
