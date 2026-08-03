
#pragma once

#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/runtime/decode_step.h"
#include "tatara/runtime/prefill_step.h"

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace tatara::engine {

enum class SpeculativeError : std::uint8_t {
    None,
    DraftCheckpointRefused,
    PipelineUnavailable,
    BufferAllocationFailed,
    VerifyStepUnavailable,
    CommandFailed,
    CaptureOverflow,
    NotConditioned,
};

struct SpeculativeStepResult {
    SpeculativeError error{SpeculativeError::CommandFailed};
    std::vector<std::uint32_t> committed;
    std::uint32_t next_staged{0};
    bool used_copy{false};

    explicit operator bool() const {
        return error == SpeculativeError::None;
    }
};

namespace detail {
struct SpeculativeState;
} // namespace detail

class SpeculativeEngine;

struct SpeculativeEngineResult {
    SpeculativeError error{SpeculativeError::PipelineUnavailable};
    std::unique_ptr<SpeculativeEngine> engine;

    explicit operator bool() const {
        return error == SpeculativeError::None && engine != nullptr;
    }
};

// slot selects the persistent state the engine drafts against and rolls
// back; null binds the step's primary slot (the single-stream default).
SpeculativeEngineResult create_speculative_engine(
    const backend::metal::MetalDevice& device,
    const backend::metal::MetalLibrary& library,
    const backend::metal::MetalCommandQueue& queue,
    runtime::DecodeStep& decode, runtime::DecodeStateSlot* slot,
    std::uint32_t context_capacity,
    std::string_view draft_checkpoint_root);

class SpeculativeEngine {
  public:
    ~SpeculativeEngine();
    SpeculativeEngine(const SpeculativeEngine&) = delete;
    SpeculativeEngine& operator=(const SpeculativeEngine&) = delete;

    SpeculativeError observe_prompt_band(
        const backend::metal::MetalBuffer& captures, std::uint32_t rows,
        std::uint32_t base);

    SpeculativeError observe_handoff_row(
        const backend::metal::MetalBuffer& features, std::uint32_t base);

    SpeculativeStepResult step(std::uint32_t staged, std::uint32_t context);

    void extend_history(const std::uint32_t* ids, std::size_t count);

    void reset_request();

    std::uint32_t conditioned_rows() const;

  private:
    friend SpeculativeEngineResult create_speculative_engine(
        const backend::metal::MetalDevice&,
        const backend::metal::MetalLibrary&,
        const backend::metal::MetalCommandQueue&, runtime::DecodeStep&,
        runtime::DecodeStateSlot*, std::uint32_t, std::string_view);
    SpeculativeEngine();
    std::unique_ptr<detail::SpeculativeState> state_;
};

} // namespace tatara::engine
