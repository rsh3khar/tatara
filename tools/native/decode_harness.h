#pragma once

#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/model/qwen36_plan.h"
#include "tatara/runtime/decode_step.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

// Shared boot and state-record I/O for the real-weight decode probes.
//
// Extracted when a third probe would have cloned the same ~270-line sequence:
// parse the prepared record, validate identity, plan the image layout, open
// shards, build the device, queue, kernel library and twenty-one pipelines,
// allocate and populate the image, resolve bindings, construct the step. The
// duplication had already put the Q36BRS01 record format and the sealed
// continuation under two owners that could drift silently, and had begun to.
namespace tatara::tools {

std::vector<std::byte> read_file(const char* path);

// The booted engine. The image is heap-held because DecodeStep borrows it by
// pointer: a moved-from inline buffer would leave that pointer dangling, so
// the address has to stay stable across the move out of boot_decode.
struct DecodeHarness {
    int exit_code = 0;
    std::optional<backend::metal::MetalDevice> device;
    std::optional<backend::metal::MetalCommandQueue> queue;
    std::optional<backend::metal::MetalLibrary> library;
    std::unique_ptr<backend::metal::MetalBuffer> image;
    std::optional<runtime::DecodeStep> step;
    std::uint32_t capacity = 0;
    double load_seconds = 0.0;

    explicit operator bool() const {
        return exit_code == 0;
    }
};

struct DecodeImagePlan {
    int exit_code = 0;
    std::uint64_t prepared_record_bytes = 0;
    std::uint64_t image_bytes = 0;

    explicit operator bool() const {
        return exit_code == 0;
    }
};

// Probe-selectable score implementation. Gqa4 is the permanent exact control;
// Gqa8 is a source-distinct candidate that must pass the component and
// same-trajectory performance gates before any production integration.
enum class DecodeAttentionScoreKernel {
    Adaptive,
    AdaptiveA23,
    Gqa4,
    Gqa4SimdReduce,
    Gqa8,
    Gqa8FusedPart,
    IndependentHeadVector2Pass,
};

// Smallest current-artifact context at which the fused GQA8 score/value path
// has passed an exact forward/reverse performance bracket. This is a dispatch
// crossover, not a context or output-token bound.
inline constexpr std::uint32_t
    kQualifiedFusedScoreValueMinimumContext = 15000;
inline constexpr std::uint32_t
    kQualifiedVectorMinimumContext = 8000;

enum class DecodeAttentionValueKernel {
    Gqa8T1024,
    Gqa8T512,
};

// Validates the prepared record identity and derives the exact image extent
// without opening shards, creating Metal objects, or allocating the model.
DecodeImagePlan inspect_decode_image(const char* record_path);

// Exit codes are the sealed ones the probes already returned, so a failure
// keeps the meaning it had before extraction: 3 unreadable record, 4 parse,
// 5 identity, 6 layout, 7 shards, 9 device, 10 queue, 11 library, 12 function,
// 13 pipeline, 14 allocation, 15 population, 16 bindings, 17 step.
DecodeHarness boot_decode(const char* record_path, const char* artifact_root);
DecodeHarness boot_decode(const char* record_path, const char* artifact_root,
                          std::uint32_t capacity);
DecodeHarness boot_decode(const char* record_path, const char* artifact_root,
                          std::uint32_t capacity,
                          DecodeAttentionScoreKernel score_kernel);
DecodeHarness boot_decode(const char* record_path, const char* artifact_root,
                          std::uint32_t capacity,
                          DecodeAttentionScoreKernel score_kernel,
                          DecodeAttentionValueKernel value_kernel);

// Which half of a gated-delta ping-pong pair holds the state the last step
// wrote. The pair is selected by a FLAG, not swapped physically, and
// `advance_decode_state` toggles that flag AFTER the write -- so at rest the
// live plane is the opposite of the one the next step would target. Live is
// therefore `first_out` when `swapped`, which for a walk of N steps starting
// from `swapped == false` means the live half depends on **N's parity**.
//
// This is the invariant a 16-token caller satisfies by luck and a 3925-token
// caller violates. It is a named, tested predicate rather than a comment
// because it was previously only a comment, and the comment was lost when this
// code was extracted -- which would have dumped 30 of 40 layers one token
// stale while every size check still passed.
constexpr bool gated_delta_live_is_out(bool swapped) {
    return swapped;
}

// Q36BRS01: magic, u32 count, then per layer in schedule order -- gated-delta
// conv then recurrent, or attention K then V, each contiguous per head. The
// caller's positions count is written as-is, so a record never overstates what
// it holds. Attention layers append in place and never ping-pong.
// Installs a sealed Q36BRS01 state record into a booted step, so decode can
// be measured at production context instead of from an empty cache.
bool load_state_record(std::span<const std::byte> record, runtime::DecodeStep& step,
                       std::span<const model::qwen36::LayerKind> schedule, std::uint32_t capacity);

bool write_state_record(const char* path, const runtime::DecodeStep& step,
                        std::span<const model::qwen36::LayerKind> schedule, std::uint32_t capacity,
                        std::uint32_t positions);

// Whether the walk actually advanced on its final step: for every gated-delta
// layer the live plane must differ from its stale partner, which holds the
// state as of N-1. A walk that stalled leaves them identical. This is the
// evolution gate the eighth boundary contract requires.
//
// It does NOT check plane selection, and cannot: memcmp equality is symmetric,
// so this returns the same answer under either polarity of
// `gated_delta_live_is_out`. An earlier version of this comment claimed the
// gate would have caught the stale-plane defect. It would not have. Correctness
// of the selection rests entirely on the predicate above and on the derivation
// written out there -- which is exactly why that derivation is spelled out
// rather than left as an assertion.
bool gated_delta_advanced(const runtime::DecodeStep& step,
                          std::span<const model::qwen36::LayerKind> schedule,
                          std::size_t& stalled_layer);

} // namespace tatara::tools
