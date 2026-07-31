#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "tatara/backend/metal/resources.h"

namespace tatara::backend::metal {

struct MetalFunctionResult;

class MetalLibrary {
  public:
    MetalLibrary() noexcept;
    ~MetalLibrary();

    MetalLibrary(const MetalLibrary&) = delete;
    MetalLibrary& operator=(const MetalLibrary&) = delete;
    MetalLibrary(MetalLibrary&&) noexcept;
    MetalLibrary& operator=(MetalLibrary&&) noexcept;

    explicit operator bool() const noexcept;

  private:
    struct Storage;

    friend MetalLibraryResult create_library_with_source(const MetalDevice&, std::string_view);
    friend MetalFunctionResult create_function(const MetalLibrary&, std::string_view);

    explicit MetalLibrary(std::unique_ptr<Storage> storage);

    std::unique_ptr<Storage> storage_;
};

class MetalFunction {
  public:
    MetalFunction() noexcept;
    ~MetalFunction();

    MetalFunction(const MetalFunction&) = delete;
    MetalFunction& operator=(const MetalFunction&) = delete;
    MetalFunction(MetalFunction&&) noexcept;
    MetalFunction& operator=(MetalFunction&&) noexcept;

    explicit operator bool() const noexcept;

  private:
    struct Storage;

    friend MetalFunctionResult create_function(const MetalLibrary&, std::string_view);
    friend MetalComputePipelineResult create_compute_pipeline(const MetalDevice&,
                                                              const MetalFunction&);
    friend MetalComputePipelineResult
    create_indirect_compute_pipeline(const MetalDevice&,
                                     const MetalFunction&);

    explicit MetalFunction(std::unique_ptr<Storage> storage);

    std::unique_ptr<Storage> storage_;
};

class MetalComputePipeline {
  public:
    MetalComputePipeline() noexcept;
    ~MetalComputePipeline();

    MetalComputePipeline(const MetalComputePipeline&) = delete;
    MetalComputePipeline& operator=(const MetalComputePipeline&) = delete;
    MetalComputePipeline(MetalComputePipeline&&) noexcept;
    MetalComputePipeline& operator=(MetalComputePipeline&&) noexcept;

    explicit operator bool() const noexcept;

  private:
    struct Storage;

    friend MetalComputePipelineResult create_compute_pipeline(const MetalDevice&,
                                                              const MetalFunction&);
    friend MetalComputePipelineResult
    create_indirect_compute_pipeline(const MetalDevice&,
                                     const MetalFunction&);
    friend MetalCommandError set_compute_pipeline(MetalComputePass&, const MetalComputePipeline&);
    friend MetalCommandError
    set_indirect_compute_pipeline(MetalIndirectCommandBuffer&, std::uint32_t,
                                  const MetalComputePipeline&);
    friend bool
    supports_indirect_commands(const MetalComputePipeline&) noexcept;
    friend std::uint64_t
    compute_pipeline_identity(const MetalComputePipeline&) noexcept;

    explicit MetalComputePipeline(std::unique_ptr<Storage> storage);

    std::unique_ptr<Storage> storage_;
};

enum class MetalPipelineError : std::uint8_t {
    None,
    InvalidDevice,
    InvalidLibrary,
    InvalidFunction,
    InvalidSource,
    InvalidFunctionName,
    LibraryCompilationFailed,
    FunctionLookupFailed,
    PipelineCreationFailed,
    OwnershipFailure,
};

struct MetalLibraryResult {
    MetalPipelineError error;
    std::optional<MetalLibrary> library;
    std::string failure_description;

    explicit operator bool() const noexcept {
        return error == MetalPipelineError::None && library.has_value();
    }
};

struct MetalFunctionResult {
    MetalPipelineError error;
    std::optional<MetalFunction> function;

    explicit operator bool() const noexcept {
        return error == MetalPipelineError::None && function.has_value();
    }
};

struct MetalComputePipelineResult {
    MetalPipelineError error;
    std::optional<MetalComputePipeline> pipeline;
    std::string failure_description;

    explicit operator bool() const noexcept {
        return error == MetalPipelineError::None && pipeline.has_value();
    }
};

MetalLibraryResult create_library_with_source(const MetalDevice& device, std::string_view source);
MetalFunctionResult create_function(const MetalLibrary& library, std::string_view name);
MetalComputePipelineResult create_compute_pipeline(const MetalDevice& device,
                                                   const MetalFunction& function);
MetalComputePipelineResult create_indirect_compute_pipeline(
    const MetalDevice& device, const MetalFunction& function);
bool supports_indirect_commands(
    const MetalComputePipeline& pipeline) noexcept;
std::uint64_t compute_pipeline_identity(
    const MetalComputePipeline& pipeline) noexcept;

} // namespace tatara::backend::metal
