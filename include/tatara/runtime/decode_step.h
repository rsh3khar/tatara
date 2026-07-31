#pragma once

#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/runtime/decode_bindings.h"
#include "tatara/runtime/decode_geometry.h"

#include <optional>
#include <span>
#include <vector>

namespace tatara::runtime {

enum class DecodeAttentionSplitPolicy : std::uint8_t {
    SeparateScoreValue,
    FusedGqa8ScoreValue,
    AdaptiveGqa8ScoreValue,
    IndependentHeadVector2Pass,
    AdaptiveVector2Pass,
};

// The admitted pipelines the sealed decode step dispatches, resolved by the
// caller from the generated kernel library. out_projection serves both the
// gated-delta and attention output projections (identical geometry).
struct DecodePipelines {
    backend::metal::MetalComputePipeline embed;
    backend::metal::MetalComputePipeline rms;
    backend::metal::MetalComputePipeline gdn_project;
    backend::metal::MetalComputePipeline gdn_prepare;
    backend::metal::MetalComputePipeline gdn_recurrence;
    backend::metal::MetalComputePipeline gdn_gate_norm;
    backend::metal::MetalComputePipeline out_projection;
    backend::metal::MetalComputePipeline attn_project;
    backend::metal::MetalComputePipeline attn_qk_rope;
    backend::metal::MetalComputePipeline attention_decode;
    backend::metal::MetalComputePipeline attention_scores;
    backend::metal::MetalComputePipeline attention_scores_values_fused;
    backend::metal::MetalComputePipeline attention_vector_part;
    backend::metal::MetalComputePipeline attention_vector_combine;
    backend::metal::MetalComputePipeline attention_values;
    backend::metal::MetalComputePipeline attention_combine;
    backend::metal::MetalComputePipeline residual_rms;
    backend::metal::MetalComputePipeline router;
    backend::metal::MetalComputePipeline router_select;
    backend::metal::MetalComputePipeline grouped_upgate;
    backend::metal::MetalComputePipeline grouped_down_res;
    backend::metal::MetalComputePipeline lmhead;
    backend::metal::MetalComputePipeline argmax_stage1;
    backend::metal::MetalComputePipeline argmax_stage2;
    DecodeAttentionSplitPolicy attention_split_policy{
        DecodeAttentionSplitPolicy::SeparateScoreValue};
    // Performance crossover only. This never constrains admission, context,
    // or output tokens; AdaptiveGqa8ScoreValue selects the fused path iff the
    // live decode context reaches this evidence-qualified boundary.
    std::uint32_t fused_score_value_minimum_context{0};
    // A second evidence-only crossover. AdaptiveVector2Pass selects the
    // independent-head vector path at/above this context, then falls through
    // to the fused/split crossover above. It is never an admission bound.
    std::uint32_t vector_minimum_context{0};
};

// One layer's persistent state. Gated-delta layers ping-pong their conv and
// recurrent pairs each token; attention layers append K/V in place at the
// step position and never swap.
struct DecodeLayerState {
    backend::metal::MetalBuffer first;
    backend::metal::MetalBuffer first_out;
    backend::metal::MetalBuffer second;
    backend::metal::MetalBuffer second_out;
    bool swapped;
};

enum class DecodeStateSlotStatus : std::uint8_t {
    Ready,
    SnapshotPending,
    RestorePending,
    ResetPending,
    Poisoned,
};

// Persistent per-request state. schedule_identity points into the immutable
// schedule storage of the DecodeStep that created this slot; vector moves
// preserve that storage address. A slot cannot be used with another executor.
struct DecodeStateSlot {
    std::uint32_t capacity{0};
    const model::qwen36::LayerKind* schedule_identity{nullptr};
    std::vector<DecodeLayerState> layers;
    DecodeStateSlotStatus status{DecodeStateSlotStatus::Ready};
    std::uint64_t next_transfer_generation{1};
    std::uint64_t active_transfer_generation{0};
    const backend::metal::MetalBuffer* active_transfer_arena{nullptr};
    std::uint32_t active_transfer_positions{0};
    std::uint64_t active_transfer_offset_bytes{0};
    std::uint64_t active_transfer_state_bytes{0};
    std::uint64_t next_reset_generation{1};
    std::uint64_t active_reset_generation{0};
    std::uint32_t active_reset_segments{0};
    std::uint64_t active_reset_bytes{0};
};

enum class DecodeStepError : std::uint8_t {
    None,
    InvalidDevice,
    InvalidImage,
    OffsetCountMismatch,
    BufferAllocationFailed,
};

// Owns one reusable scratch set and a primary persistent state slot, with
// weight bindings resolved to (image, offset) pairs at construction.
// Additional slots share this scratch through the explicit encode overload;
// the single engine owner serializes command encoding across them. The
// per-token walk allocates nothing and looks up nothing.
struct DecodeStep {
    DecodeGeometry geometry;
    std::uint32_t capacity;
    DecodeBindings bindings;
    std::vector<model::qwen36::LayerKind> schedule;
    const backend::metal::MetalBuffer* image;
    std::vector<std::uint64_t> tensor_offsets;
    DecodePipelines pipelines;

    backend::metal::MetalBuffer input;
    backend::metal::MetalBuffer normed;
    backend::metal::MetalBuffer branch_stream;
    backend::metal::MetalBuffer residual_stream;
    backend::metal::MetalBuffer moe_stream;
    backend::metal::MetalBuffer layer_stream;
    backend::metal::MetalBuffer gdn_projection;
    backend::metal::MetalBuffer gdn_qk;
    backend::metal::MetalBuffer gdn_value;
    backend::metal::MetalBuffer gdn_z;
    backend::metal::MetalBuffer gdn_y;
    backend::metal::MetalBuffer gdn_gated;
    backend::metal::MetalBuffer attn_projection;
    backend::metal::MetalBuffer attn_query;
    backend::metal::MetalBuffer attn_gate;
    backend::metal::MetalBuffer attn_attended;
    backend::metal::MetalBuffer attn_partials;
    backend::metal::MetalBuffer attn_weights;
    backend::metal::MetalBuffer router_logits;
    backend::metal::MetalBuffer expert_ids;
    backend::metal::MetalBuffer expert_coefficients;
    backend::metal::MetalBuffer shared_coefficient;
    backend::metal::MetalBuffer expert_hidden;
    backend::metal::MetalBuffer final_hidden;
    backend::metal::MetalBuffer logits;
    backend::metal::MetalBuffer argmax_values;
    backend::metal::MetalBuffer argmax_indices;
    backend::metal::MetalBuffer token_id;
    DecodeStateSlot state;
};

struct DecodeStepResult {
    DecodeStepError error;
    std::optional<DecodeStep> step;

    explicit operator bool() const noexcept {
        return error == DecodeStepError::None && step.has_value();
    }
};

struct DecodeStateSlotResult {
    DecodeStepError error;
    std::optional<DecodeStateSlot> slot;

    explicit operator bool() const noexcept {
        return error == DecodeStepError::None && slot.has_value();
    }
};

// Allocates zeroed working buffers and states (the sealed zeroed-allocation
// rule: allocator recycling must never perturb results). The image buffer
// must outlive the step; tensor_offsets is the image layout parallel to the
// bound record indices.
DecodeStepResult
create_decode_step(const backend::metal::MetalDevice& device, const DecodeGeometry& geometry,
                   std::uint32_t capacity, std::span<const model::qwen36::LayerKind> schedule,
                   DecodeBindings bindings, const backend::metal::MetalBuffer& image,
                   std::span<const std::uint64_t> tensor_offsets, DecodePipelines pipelines);

// Allocates one additional zeroed persistent slot compatible only with step.
// The step and its immutable schedule storage must outlive the slot.
DecodeStateSlotResult create_decode_state_slot(const backend::metal::MetalDevice& device,
                                               const DecodeStep& step);

bool decode_state_slot_compatible(const DecodeStep& step, const DecodeStateSlot& state) noexcept;
bool decode_state_slot_complete(const DecodeStep& step, const DecodeStateSlot& state) noexcept;
bool decode_state_slot_available(const DecodeStep& step, const DecodeStateSlot& state) noexcept;
bool decode_state_slot_ready(const DecodeStep& step, const DecodeStateSlot& state) noexcept;

// Encodes one sealed token step into the pass: embed from the on-device
// token id, the full layer walk with barriers between dispatches, then the
// head and the argmax write-back the next step's embedding reads.
// ContextOutOfRange is returned before any dispatch is encoded.
backend::metal::MetalCommandError
encode_token(DecodeStep& step, backend::metal::MetalComputePass& pass, std::uint32_t context);

backend::metal::MetalCommandError encode_token(DecodeStep& step, DecodeStateSlot& state,
                                               backend::metal::MetalComputePass& pass,
                                               std::uint32_t context);

// Swaps the gated-delta conv and recurrent pairs after a token completes.
void advance_decode_state(DecodeStep& step);
void advance_decode_state(const DecodeStep& step, DecodeStateSlot& state);

} // namespace tatara::runtime
