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

// One pool buffer per layer per state kind, holding every slot's state as
// equal stripes. Gated layers interleave ping-pong parities: slot s holds
// stripes 2s and 2s+1, so a kernel reaches both through one binding.
// Slots built over a pool behave as ordinary slots everywhere else.
struct DecodeStatePoolLayer {
    backend::metal::MetalBuffer first;
    backend::metal::MetalBuffer second;
    std::uint64_t first_stride{0};
    std::uint64_t second_stride{0};
};

struct DecodeStatePool {
    std::uint32_t stripes{0};
    std::vector<DecodeStatePoolLayer> layers;
};

struct DecodeStateSlotPoolResult {
    DecodeStepError error;
    std::optional<DecodeStatePool> pool;
    std::vector<DecodeStateSlot> slots;

    explicit operator bool() const noexcept {
        return error == DecodeStepError::None && pool.has_value();
    }
};

// Per-stream token scratch: every buffer one token walk writes. A group
// submission interleaves layers across streams, so each stream owns its
// scratch and the step's own buffers serve at most one stream.
struct DecodeStreamScratch {
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
};

struct DecodeStreamScratchResult {
    DecodeStepError error;
    std::optional<DecodeStreamScratch> scratch;

    explicit operator bool() const noexcept {
        return error == DecodeStepError::None && scratch.has_value();
    }
};

// One stream of a group submission. A null scratch selects the step's
// own scratch; at most one stream per group may do so, and states and
// scratches must be pairwise distinct.
struct DecodeStream {
    DecodeStateSlot* state{nullptr};
    DecodeStreamScratch* scratch{nullptr};
    std::uint32_t context{0};
    std::uint32_t stripe{0};
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

// Allocates count zeroed persistent slots striped over shared per-layer
// pool buffers. The slots behave exactly like create_decode_state_slot
// slots on every existing path; the pool exists for row-batched kernels.
DecodeStateSlotPoolResult create_decode_state_slot_pool(
    const backend::metal::MetalDevice& device, const DecodeStep& step, std::uint32_t count);

// Allocates one zeroed per-stream scratch set sized by the step's
// geometry, for group submissions.
DecodeStreamScratchResult create_decode_stream_scratch(
    const backend::metal::MetalDevice& device, const DecodeStep& step);

// Batched-token scratch: every per-token buffer as an [rows x stride]
// slab. Unbatched kernels run per row through buffer offsets; the _ms
// kernels consume whole slabs. Strides are the single-stream byte sizes.
struct DecodeBatchScratch {
    std::uint32_t rows{0};
    // Rows per threadgroup in each dense projection stage: tile-of-rows
    // grids trade weight-load amortization against row parallelism. Any
    // value in [1, rows] is exact per stage.
    std::uint32_t project_row_tile{8};
    std::uint32_t outproj_row_tile{8};
    std::uint32_t attnproj_row_tile{8};
    // Timing instrument: when nonzero, the numbered stage encodes its
    // dispatch twice. Stage inputs and outputs are distinct buffers, so
    // doubling is idempotent and walks stay byte-exact.
    std::uint32_t profile_double_stage{0};
    // Rows per threadgroup in the vocabulary head. The full loop maximizes
    // weight amortization of the largest read; smaller tiles trade re-reads
    // for row parallelism. Any value in [1, rows] is exact.
    std::uint32_t head_row_tile{8};
    // Dense-stage kernel variant: 0 scalar loads, 1 vector loads
    // (bit-identical per row; load width only). Variant pipelines must
    // be resolved when set.
    std::uint32_t dense_variant{0};
    DecodeStreamScratch slabs;
    backend::metal::MetalBuffer union_table;
    backend::metal::MetalBuffer moe_parts;
    backend::metal::MetalBuffer state_offsets;
};

struct DecodeBatchScratchResult {
    DecodeStepError error;
    std::optional<DecodeBatchScratch> scratch;

    explicit operator bool() const noexcept {
        return error == DecodeStepError::None && scratch.has_value();
    }
};

DecodeBatchScratchResult create_decode_batch_scratch(
    const backend::metal::MetalDevice& device, const DecodeStep& step,
    std::uint32_t rows);

// Row-batched dense kernels (weight traffic shared across rows; per-row
// numerics identical to the serial kernels).
struct DecodeBatchPipelines {
    backend::metal::MetalComputePipeline gdn_project_ms;
    backend::metal::MetalComputePipeline gdn_outproj_ms;
    backend::metal::MetalComputePipeline attn_project_ms;
    backend::metal::MetalComputePipeline lmhead_ms;
    backend::metal::MetalComputePipeline embed_ms;
    backend::metal::MetalComputePipeline rms_ms;
    backend::metal::MetalComputePipeline residual_rms_ms;
    backend::metal::MetalComputePipeline gate_norm_ms;
    backend::metal::MetalComputePipeline router_ms;
    backend::metal::MetalComputePipeline upgate_rows_ms;
    backend::metal::MetalComputePipeline down_rows_ms;
    backend::metal::MetalComputePipeline argmax_stage1_ms;
    backend::metal::MetalComputePipeline argmax_stage2_ms;
    backend::metal::MetalComputePipeline gdn_prepare_ms;
    backend::metal::MetalComputePipeline gdn_recurrence_ms;
    backend::metal::MetalComputePipeline attn_qk_rope_ms;
    backend::metal::MetalComputePipeline router_select_ms;
    backend::metal::MetalComputePipeline attention_decode_ms;
    backend::metal::MetalComputePipeline gdn_project_ms_v2;
    backend::metal::MetalComputePipeline lmhead_ms_v2;
    backend::metal::MetalComputePipeline attn_project_ms_v2;
    backend::metal::MetalComputePipeline gdn_outproj_ms_v2;
};

// One batched token step for up to eight streams over pool-striped slots.
// The host writes the per-step state-offset table before encoding, so the
// previous step must have completed. Per-stream trajectories are
// bit-identical to the serial walk. chain_input, when set, replaces this
// scratch's token slab as the embed input, chaining steps on the GPU so a
// next step can be committed before the current one completes.
backend::metal::MetalCommandError encode_token_batch(
    DecodeStep& step, std::span<const DecodeStream> streams,
    const DecodeStatePool& pool, DecodeBatchScratch& batch,
    const DecodeBatchPipelines& kernels,
    backend::metal::MetalComputePass& pass,
    const backend::metal::MetalBuffer* chain_input = nullptr);

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

// Encodes one token step for every stream into the same pass,
// layer-interleaved: each layer's weights are visited once while all
// streams consume them. Per-stream numerics are identical to the serial
// walk; streams advance independently.
backend::metal::MetalCommandError encode_token_group(
    DecodeStep& step, std::span<const DecodeStream> streams,
    backend::metal::MetalComputePass& pass);

// Swaps the gated-delta conv and recurrent pairs after a token completes.
void advance_decode_state(DecodeStep& step);
void advance_decode_state(const DecodeStep& step, DecodeStateSlot& state);

} // namespace tatara::runtime
