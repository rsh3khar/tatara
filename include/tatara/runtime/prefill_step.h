#pragma once

#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/runtime/decode_step.h"
#include "tatara/runtime/prefill_command_plan.h"
#include "tatara/runtime/prefill_geometry.h"

#include <array>
#include "tatara/runtime/prefill_profiler.h"
#include "tatara/runtime/quantized_gemm.h"

#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace tatara::runtime {

inline constexpr std::uint32_t kPrefillMaximumUnitsPerSubmission = 64;

enum class PrefillRouterSelector : std::uint8_t {
    Serial,
    Parallel,
};

enum class PrefillGdnRecurrence : std::uint8_t {
    SerialSteps,
    RegisterLoop,
    RegisterLoopTape,
};

enum class PrefillAttentionKernel : std::uint8_t {
    PartialCombine,
    StagedGemmAdaptive,
    StreamingFlashAdaptive,
    FlashMmaV2,
    SteelGemm,
};

struct PrefillExecutionPolicy {
    PrefillPolicy geometry;
    PrefillRouterSelector router_selector{PrefillRouterSelector::Serial};
    PrefillGdnRecurrence gdn_recurrence{PrefillGdnRecurrence::SerialSteps};
    PrefillAttentionKernel attention_kernel{
        PrefillAttentionKernel::PartialCombine};
    std::uint32_t staged_attention_minimum_context{256};
    std::uint32_t streaming_attention_minimum_context{256};
    QuantizedGemmPolicy dense_qgemm{QuantizedGemmPolicy::ExactRow};
    QuantizedGemmPolicy routed_qgemm{QuantizedGemmPolicy::ExactRow};
    bool native_dense_steel{false};
    // Default-off GDN-only treatment. Full BM64 tiles use the generated
    // Steel BM64/WM2/WN2 pipeline; all nonconforming rows remain on the
    // permanent dense Steel path.
    bool native_dense_steel_gdn_bm64_wm2_wn2{false};
    bool native_routed_shared_expert{false};
    bool native_routed_steel{false};
    bool command_graph{false};
    // Default-off scheduling treatment. This changes how an admitted graph
    // band is submitted; it does not change admission or impose a context
    // or output-token bound.
    bool command_graph_lane_events{false};
    // Maximum scratch lanes owned by the step. One active graph band may use
    // any count in [1, command_graph_chunk_count]; it is not the request's
    // total chunk count and therefore never imposes a context ceiling.
    std::uint32_t command_graph_chunk_count{1};
    std::uint32_t maximum_units_per_submission{1};
    std::uint32_t maximum_inflight_units{1};
    bool conditioning_capture{false};
    std::uint32_t conditioning_capture_rows{16};
    std::array<std::uint32_t, 8> capture_layers{};
};

struct PrefillPipelines {
    backend::metal::MetalComputePipeline embed;
    backend::metal::MetalComputePipeline rms;
    backend::metal::MetalComputePipeline residual;
    backend::metal::MetalComputePipeline gdn_project;
    backend::metal::MetalComputePipeline gdn_conv;
    backend::metal::MetalComputePipeline gdn_gates;
    backend::metal::MetalComputePipeline gdn_recurrence_step;
    backend::metal::MetalComputePipeline gdn_recurrence_block;
    backend::metal::MetalComputePipeline gdn_recurrence_gates;
    backend::metal::MetalComputePipeline gdn_gate_norm;
    backend::metal::MetalComputePipeline attn_project;
    backend::metal::MetalComputePipeline attn_qk_rope;
    backend::metal::MetalComputePipeline attention_partial;
    backend::metal::MetalComputePipeline attention_combine;
    backend::metal::MetalComputePipeline attention_staged_scores;
    backend::metal::MetalComputePipeline attention_staged_softmax;
    backend::metal::MetalComputePipeline attention_staged_values;
    backend::metal::MetalComputePipeline attention_streaming;
    backend::metal::MetalComputePipeline attention_flash_v2;
    backend::metal::MetalComputePipeline attention_steel_scores;
    backend::metal::MetalComputePipeline attention_steel_values;
    backend::metal::MetalComputePipeline attention_steel_scores_large;
    backend::metal::MetalComputePipeline attention_steel_values_large;
    backend::metal::MetalComputePipeline gdn_conv_tape;
    backend::metal::MetalComputePipeline gdn_recurrence_tape;
    backend::metal::MetalComputePipeline capture_rows;
    backend::metal::MetalComputePipeline attention_softmax_bf16;
    backend::metal::MetalComputePipeline attention_gate_apply;
    backend::metal::MetalComputePipeline out_projection;
    backend::metal::MetalComputePipeline router;
    backend::metal::MetalComputePipeline router_select_serial;
    backend::metal::MetalComputePipeline router_select_parallel;
    backend::metal::MetalComputePipeline expert_union;
    backend::metal::MetalComputePipeline expert_union_fused_tasks;
    backend::metal::MetalComputePipeline expert_upgate;
    backend::metal::MetalComputePipeline expert_down;
    backend::metal::MetalComputePipeline expert_combine;
    backend::metal::MetalComputePipeline native_dense_qgemm;
    backend::metal::MetalComputePipeline native_dense_steel;
    backend::metal::MetalComputePipeline
        native_dense_steel_gdn_bm64_wm2_wn2;
    backend::metal::MetalComputePipeline native_routed_task_builder;
    backend::metal::MetalComputePipeline native_routed_upgate;
    backend::metal::MetalComputePipeline native_routed_down;
    backend::metal::MetalComputePipeline native_routed_steel_upgate;
    backend::metal::MetalComputePipeline native_routed_steel_down;
};

enum class PrefillStepError : std::uint8_t {
    None,
    InvalidDevice,
    InvalidGeometry,
    InvalidPolicy,
    PipelineUnavailable,
    BufferAllocationFailed,
    CommandGraphTopologyUnavailable,
};

enum class PrefillEncodeError : std::uint8_t {
    None,
    EmptyPrefix,
    BlockOutOfRange,
    ContextOutOfRange,
    ContextOverflow,
    TokenOutOfRange,
    BindingMismatch,
    ScratchTooSmall,
    CommandEncodingFailed,
    DeviceTaskValidationUnavailable,
    CommandGraphRequired,
};

enum class PrefillProgressState : std::uint8_t {
    Idle,
    Ready,
    UnitPending,
    BatchPending,
    InflightEncoding,
    InflightPending,
    GraphPending,
    Complete,
    Poisoned,
};

enum class PrefillProgressError : std::uint8_t {
    None,
    Active,
    Unavailable,
    UnitPending,
    UnitNotPending,
    BatchPending,
    BatchNotPending,
    BatchUnavailable,
    InflightEncoding,
    InflightPending,
    InflightNotPending,
    InflightUnavailable,
    GraphPending,
    GraphNotPending,
    GraphRequired,
    Complete,
    Poisoned,
    Invalid,
    DeviceTaskNotReady,
};

struct PrefillProgress {
    PrefillProgressState state{PrefillProgressState::Idle};
    const DecodeStep* owner{nullptr};
    const DecodeStateSlot* state_owner{nullptr};
    std::uint32_t live_context{0};
    std::uint32_t context_base{0};
    std::uint32_t next_context{0};
    std::uint32_t row_count{0};
    std::uint32_t chunk_count{0};
    std::uint32_t current_layer{0};
    std::uint32_t current_chunk{0};
    std::uint32_t pending_unit_count{0};
};

enum class PrefillCommandGraphState : std::uint8_t {
    Empty,
    Ready,
    Pending,
    Poisoned,
};

struct PrefillCommandGraph {
    PrefillCommandGraphState state{PrefillCommandGraphState::Empty};
    const DecodeStep* decode_owner{nullptr};
    const DecodeStateSlot* state_owner{nullptr};
    std::uint32_t row_count{0};
    std::uint32_t context_base{0};
    std::uint32_t command_count{0};
    std::uint64_t argument_arena_bytes{0};
    backend::metal::MetalIndirectCommandBuffer commands;
    backend::metal::MetalBuffer argument_arena;
    std::vector<std::uint32_t> chunk_rows;
    std::vector<PrefillCommandNode> nodes;
    std::vector<PrefillCommandDiagonal> diagonals;
    std::vector<std::uint32_t> level_command_begins;
    // Lane-event graphs store commands node-major. For node N,
    // node_level_boundary_offsets[N..N+1] selects its local-level command
    // boundaries (including the terminal boundary) in the flat array.
    std::vector<std::uint32_t> node_level_boundary_offsets;
    std::vector<std::uint32_t> node_level_command_begins;
    std::vector<PrefillLaneEventNode> lane_event_nodes;
    std::vector<std::uint8_t> command_classes;
    std::vector<backend::metal::MetalBuffer> image_windows;
    struct BufferWindow {
        const backend::metal::MetalBuffer* source{nullptr};
        std::uint64_t source_begin{0};
        backend::metal::MetalBuffer window;
    };
    std::vector<BufferWindow> scratch_windows;
    PrefillCommandIdentity model_package_identity{};
    PrefillCommandIdentity prepared_image_identity{};
    PrefillCommandIdentity pipeline_identity{};
    PrefillCommandIdentity execution_policy_identity{};
    std::uint64_t icb_capability_identity{0};
    std::uint64_t state_slot_identity{0};
    std::uint32_t graph_schema_version{
        kPrefillCommandGraphSchemaVersion};
    std::vector<std::uint64_t> pipeline_identities;
    std::vector<std::uint64_t> resource_identities;
    std::vector<std::uint8_t> state_phases;
};

struct PrefillBufferWindowPlan {
    bool valid{false};
    bool use_window{false};
    std::uint64_t source_begin{0};
    std::uint64_t window_length{0};
    std::uint64_t binding_offset{0};

    explicit constexpr operator bool() const noexcept {
        return valid;
    }
};

// ICB buffer offsets are 32-bit on the supported M4 generation. Bindings
// below 4 GiB use the original buffer. Higher bindings are rebased through a
// placement-heap window beginning on a 2 GiB boundary and extending to the
// end of the source, so the kernel retains its complete accessible suffix.
PrefillBufferWindowPlan plan_prefill_buffer_window(
    std::uint64_t source_bytes, std::uint64_t absolute_offset) noexcept;

struct PrefillStep {
    PrefillGeometry geometry;
    PrefillExecutionPolicy policy;
    PrefillPipelines pipelines;
    PrefillProgress progress;
    PrefillCommandGraph command_graph;
    std::vector<backend::metal::MetalCommandQueue>
        command_graph_lane_queues;
    std::vector<backend::metal::MetalEvent>
        command_graph_lane_events;
    std::uint64_t command_graph_event_value_base{0};

    backend::metal::MetalBuffer tokens;
    backend::metal::MetalBuffer attention_steel_params;
    backend::metal::MetalBuffer gdn_tape;
    backend::metal::MetalBuffer gdn_conv_tape_buffer;
    backend::metal::MetalBuffer capture_buffer;
    backend::metal::MetalBuffer hidden_slab;
    backend::metal::MetalBuffer block_hidden;
    backend::metal::MetalBuffer normalized;
    backend::metal::MetalBuffer branch;
    backend::metal::MetalBuffer moe_output;

    backend::metal::MetalBuffer gdn_projection;
    backend::metal::MetalBuffer gdn_qk;
    backend::metal::MetalBuffer gdn_value;
    backend::metal::MetalBuffer gdn_gate;
    backend::metal::MetalBuffer gdn_recurrence;
    backend::metal::MetalBuffer gdn_gated;
    backend::metal::MetalBuffer gdn_decay;
    backend::metal::MetalBuffer gdn_beta;

    backend::metal::MetalBuffer attention_projection;
    backend::metal::MetalBuffer attention_query;
    backend::metal::MetalBuffer attention_gate;
    backend::metal::MetalBuffer attention_attended;
    backend::metal::MetalBuffer attention_partials;

    backend::metal::MetalBuffer router_logits;
    backend::metal::MetalBuffer expert_ids;
    backend::metal::MetalBuffer expert_coefficients;
    backend::metal::MetalBuffer shared_coefficients;
    backend::metal::MetalBuffer expert_counts;
    backend::metal::MetalBuffer expert_lists;
    backend::metal::MetalBuffer active_experts;
    backend::metal::MetalBuffer expert_arguments;
    backend::metal::MetalBuffer expert_hidden;
    backend::metal::MetalBuffer expert_partials;
    backend::metal::MetalBuffer native_routed_up_tasks;
    backend::metal::MetalBuffer native_routed_up_arguments;
    backend::metal::MetalBuffer native_routed_up_status;
    backend::metal::MetalBuffer native_routed_down_tasks;
    backend::metal::MetalBuffer native_routed_down_arguments;
    backend::metal::MetalBuffer native_routed_down_status;
    backend::metal::MetalBuffer shared_expert;
    backend::metal::MetalBuffer shared_expert_arguments;
    std::uint64_t native_routed_workspace_bytes{0};
    std::uint32_t command_graph_task_status_count{0};
};

struct PrefillStepResult {
    PrefillStepError error{PrefillStepError::InvalidGeometry};
    std::uint64_t requested_bytes{0};
    std::optional<PrefillStep> step;

    explicit operator bool() const noexcept {
        return error == PrefillStepError::None && step.has_value();
    }
};

enum class PrefillMemoryPlanError : std::uint8_t {
    None,
    InvalidPolicy,
    ArithmeticOverflow,
};

struct PrefillMemoryPlan {
    PrefillMemoryPlanError error{PrefillMemoryPlanError::InvalidPolicy};
    std::uint32_t maximum_scratch_lanes{0};
    std::uint32_t maximum_task_status_count{0};
    std::uint64_t token_bytes{0};
    std::uint64_t hidden_slab_bytes{0};
    std::uint64_t per_lane_bytes{0};
    std::uint64_t native_routed_workspace_bytes{0};
    std::uint64_t total_bytes{0};
    std::uint64_t maximum_single_buffer_bytes{0};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillMemoryPlanError::None;
    }
};

// Exact pure mirror of create_prefill_step's allocation list. Admission calls
// this before any Metal allocation; construction uses the same result.
PrefillMemoryPlan plan_prefill_step_memory(
    const PrefillGeometry& geometry,
    const PrefillExecutionPolicy& policy) noexcept;

enum class PrefillBandPlanError : std::uint8_t {
    None,
    Empty,
    InvalidPolicy,
    ContextOutOfRange,
    ContextOverflow,
};

struct PrefillBandPlan {
    PrefillBandPlanError error{PrefillBandPlanError::InvalidPolicy};
    std::uint32_t context_base{0};
    std::uint32_t row_count{0};
    std::uint32_t chunk_count{0};
    std::uint32_t next_context{0};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillBandPlanError::None;
    }
};

// Returns the next band of complete global chunks. The first global chunk is
// geometry.first_chunk_rows only at context zero; every later chunk is at
// most geometry.maximum_block_rows. A final partial chunk is legal only when
// it consumes the request remainder.
PrefillBandPlan plan_next_prefill_band(
    const PrefillPolicy& geometry, std::uint32_t context_base,
    std::uint32_t remaining_rows,
    std::uint32_t maximum_scratch_lanes) noexcept;

struct PrefillEncodeResult {
    PrefillEncodeError error{PrefillEncodeError::BindingMismatch};
    backend::metal::MetalCommandError command_error{backend::metal::MetalCommandError::None};
    std::uint32_t next_context{0};
    std::uint32_t chunk_count{0};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillEncodeError::None &&
               command_error == backend::metal::MetalCommandError::None;
    }
};

struct PrefillProgressResult {
    PrefillProgressError error{PrefillProgressError::Invalid};
    PrefillEncodeError encode_error{PrefillEncodeError::None};
    backend::metal::MetalCommandError command_error{backend::metal::MetalCommandError::None};
    PrefillProgressState state{PrefillProgressState::Idle};
    std::uint32_t next_context{0};
    std::uint32_t chunk_count{0};
    std::uint32_t layer_index{0};
    std::uint32_t chunk_ordinal{0};
    std::uint32_t unit_count{0};
    std::uint32_t failed_unit_offset{
        std::numeric_limits<std::uint32_t>::max()};
    QuantizedGemmDeviceTaskStatus routed_up_status{
        QuantizedGemmDeviceTaskStatus::NotProduced};
    QuantizedGemmDeviceTaskStatus routed_down_status{
        QuantizedGemmDeviceTaskStatus::NotProduced};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillProgressError::None && encode_error == PrefillEncodeError::None &&
               command_error == backend::metal::MetalCommandError::None;
    }
};

enum class PrefillCommandGraphError : std::uint8_t {
    None,
    Disabled,
    ProgressUnavailable,
    PlanMismatch,
    PipelineUnavailable,
    RecordingFailed,
    ArgumentArenaFailed,
    IndirectCommandBufferFailed,
    ResourceMismatch,
    Pending,
    NotPending,
    DeviceTaskNotReady,
    Poisoned,
};

enum class PrefillCommandGraphStage : std::uint8_t {
    None,
    Admission,
    Wavefront,
    Recording,
    Ordering,
    Census,
    ArgumentArena,
    IndirectCommands,
    CacheKey,
    Execution,
    Publication,
};

struct PrefillCommandGraphResult {
    PrefillCommandGraphError error{PrefillCommandGraphError::Disabled};
    PrefillCommandGraphStage stage{PrefillCommandGraphStage::None};
    PrefillCommandPlanError plan_error{PrefillCommandPlanError::None};
    backend::metal::MetalCommandError command_error{
        backend::metal::MetalCommandError::None};
    std::uint32_t command_count{0};
    std::uint32_t node_count{0};
    std::uint32_t diagonal_count{0};
    std::uint64_t argument_arena_bytes{0};
    bool cache_hit{false};

    explicit constexpr operator bool() const noexcept {
        return error == PrefillCommandGraphError::None &&
               plan_error == PrefillCommandPlanError::None &&
               command_error ==
                   backend::metal::MetalCommandError::None;
    }
};

struct ProfiledPrefillEncodeResult {
    PrefillEncodeResult encode;
    PrefillProfilerStatus profile;

    explicit constexpr operator bool() const noexcept {
        return static_cast<bool>(encode) && static_cast<bool>(profile);
    }
};

struct ProfiledPrefillProgressResult {
    PrefillProgressResult progress;
    PrefillProfilerStatus profile;

    explicit constexpr operator bool() const noexcept {
        return static_cast<bool>(progress) && static_cast<bool>(profile);
    }
};

PrefillStepResult create_prefill_step(const backend::metal::MetalDevice& device,
                                      const PrefillGeometry& geometry,
                                      PrefillExecutionPolicy policy, PrefillPipelines pipelines);

// Diagnostic reference: encodes a complete non-empty prefix into one existing
// compute pass. Product execution uses the bounded progress interface below.
PrefillEncodeResult encode_prefill(PrefillStep& prefill, DecodeStep& decode,
                                   backend::metal::MetalComputePass& pass,
                                   std::uint32_t live_context, std::uint32_t context_base,
                                   std::span<const std::uint32_t> tokens);
PrefillEncodeResult encode_prefill(PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
                                   backend::metal::MetalComputePass& pass,
                                   std::uint32_t live_context, std::uint32_t context_base,
                                   std::span<const std::uint32_t> tokens);
ProfiledPrefillEncodeResult encode_prefill(
    PrefillStep& prefill, DecodeStep& decode,
    backend::metal::MetalComputePass& pass, std::uint32_t live_context,
    std::uint32_t context_base, std::span<const std::uint32_t> tokens,
    PrefillProfiler& profiler,
    const backend::metal::MetalCounterSampleBuffer& samples);
ProfiledPrefillEncodeResult encode_prefill(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    backend::metal::MetalComputePass& pass, std::uint32_t live_context,
    std::uint32_t context_base, std::span<const std::uint32_t> tokens,
    PrefillProfiler& profiler,
    const backend::metal::MetalCounterSampleBuffer& samples);

// Called only after the diagnostic single-pass command buffer completes
// successfully.
void advance_prefill_state(DecodeStep& decode, std::uint32_t chunk_count);
void advance_prefill_state(const DecodeStep& decode, DecodeStateSlot& state,
                           std::uint32_t chunk_count);

// Validates one complete request and copies its token ids once into
// preallocated shared storage. No command is encoded and no persistent state
// changes. A completed step may begin another transaction; a poisoned step
// must be destroyed with its state slot.
PrefillProgressResult begin_prefill_progress(PrefillStep& prefill, DecodeStep& decode,
                                             std::uint32_t live_context, std::uint32_t context_base,
                                             std::span<const std::uint32_t> tokens);
PrefillProgressResult begin_prefill_progress(PrefillStep& prefill, const DecodeStep& decode,
                                             DecodeStateSlot& state, std::uint32_t live_context,
                                             std::uint32_t context_base,
                                             std::span<const std::uint32_t> tokens);

// Encodes one bounded unit into an existing compute pass and marks it pending.
// Layer-major units are one (layer, chunk); chunk-major units are one chunk's
// fixed layer walk. Product overloads allocate and resolve nothing. The
// diagnostic stage-boundary overload creates replacement native encoders from
// preallocated descriptors; its counter storage resolves only after completion.
PrefillProgressResult encode_prefill_unit(PrefillStep& prefill, DecodeStep& decode,
                                          backend::metal::MetalComputePass& pass);
PrefillProgressResult encode_prefill_unit(PrefillStep& prefill, DecodeStep& decode,
                                          DecodeStateSlot& state,
                                          backend::metal::MetalComputePass& pass);
ProfiledPrefillProgressResult encode_prefill_unit(
    PrefillStep& prefill, DecodeStep& decode,
    backend::metal::MetalComputePass& pass, PrefillProfiler& profiler,
    const backend::metal::MetalCounterSampleBuffer& samples);
ProfiledPrefillProgressResult encode_prefill_unit(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    backend::metal::MetalComputePass& pass, PrefillProfiler& profiler,
    const backend::metal::MetalCounterSampleBuffer& samples);

// Encodes up to policy.maximum_units_per_submission consecutive LayerMajor
// units into one existing compute pass. The canonical cursor and GDN host
// phase remain unchanged until commit_prefill_units validates every routed
// producer status after successful command-buffer completion.
PrefillProgressResult encode_prefill_units(
    PrefillStep& prefill, DecodeStep& decode,
    backend::metal::MetalComputePass& pass);

// Explicit-count batch encode: encodes up to `maximum_units` consecutive
// LayerMajor units (1 <= maximum_units <= policy.maximum_units_per_submission)
// without touching the policy. Probes that must break unit batches at exact
// layer boundaries (e.g. conditioning-capture blits) use this; the
// policy-driven overloads above are unchanged.
PrefillProgressResult encode_prefill_units(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    backend::metal::MetalComputePass& pass, std::uint32_t maximum_units);
PrefillProgressResult encode_prefill_units(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    backend::metal::MetalComputePass& pass);

// Encodes one LayerMajor unit into a distinct command buffer within a bounded
// in-flight window. Repeated calls use a shadow cursor and unique status slot;
// the canonical cursor and GDN host phase remain unchanged until every
// submitted command buffer completes and commit_prefill_inflight validates
// the full window.
PrefillProgressResult encode_prefill_inflight_unit(
    PrefillStep& prefill, DecodeStep& decode,
    backend::metal::MetalComputePass& pass);
PrefillProgressResult encode_prefill_inflight_unit(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    backend::metal::MetalComputePass& pass);

// Commits a pending unit only after its command buffer completed successfully.
// The returned next_context is publishable only when state is Complete.
PrefillProgressResult commit_prefill_unit(PrefillStep& prefill, DecodeStep& decode);
PrefillProgressResult commit_prefill_unit(PrefillStep& prefill, DecodeStep& decode,
                                          DecodeStateSlot& state);

// Commits a pending multi-unit submission atomically after successful
// command-buffer completion. Any failed routed producer poisons the request
// without advancing the canonical cursor or GDN host phase.
PrefillProgressResult commit_prefill_units(PrefillStep& prefill,
                                           DecodeStep& decode);
PrefillProgressResult commit_prefill_units(PrefillStep& prefill,
                                           DecodeStep& decode,
                                           DecodeStateSlot& state);

PrefillProgressResult commit_prefill_inflight(PrefillStep& prefill,
                                              DecodeStep& decode);
PrefillProgressResult commit_prefill_inflight(PrefillStep& prefill,
                                              DecodeStep& decode,
                                              DecodeStateSlot& state);

// Builds or reuses the immutable command graph for the active progress
// transaction. This may allocate and encode the ICB, but it creates no parent
// command buffer and submits no work.
PrefillCommandGraphResult prepare_prefill_command_graph(
    const backend::metal::MetalDevice& device, PrefillStep& prefill,
    DecodeStep& decode, DecodeStateSlot& state);
PrefillCommandGraphResult prepare_prefill_command_graph(
    const backend::metal::MetalDevice& device, PrefillStep& prefill,
    DecodeStep& decode);

// Declares the graph's exact buffer residency and appends one bounded ICB
// execution range to an existing parent compute pass. State is publishable
// only after successful command-buffer completion and commit below.
PrefillCommandGraphResult encode_prefill_command_graph(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    backend::metal::MetalComputePass& pass);
PrefillCommandGraphResult encode_prefill_command_graph(
    PrefillStep& prefill, DecodeStep& decode,
    backend::metal::MetalComputePass& pass);

// Encodes one node of a node-major lane-event graph. Cross-lane events are
// command-buffer operations and are encoded by the caller between these
// passes. Every intra-node local-level barrier remains intact.
PrefillCommandGraphResult encode_prefill_command_graph_lane_node(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    backend::metal::MetalComputePass& pass,
    std::uint32_t layer_index, std::uint32_t scratch_lane);
PrefillCommandGraphResult encode_prefill_command_graph_lane_node(
    PrefillStep& prefill, DecodeStep& decode,
    backend::metal::MetalComputePass& pass,
    std::uint32_t layer_index, std::uint32_t scratch_lane);

// Publishes the pending lifecycle only after every lane command buffer has
// encoded successfully.
PrefillCommandGraphResult mark_prefill_command_graph_lane_pending(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state);
PrefillCommandGraphResult mark_prefill_command_graph_lane_pending(
    PrefillStep& prefill, DecodeStep& decode);

PrefillProgressResult commit_prefill_command_graph(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state);
PrefillProgressResult commit_prefill_command_graph(
    PrefillStep& prefill, DecodeStep& decode);

// Releases a non-pending partial cursor after cancellation or deadline. This
// does not make its state slot canonical; the engine must reset that slot
// before scheduler retirement.
PrefillProgressError release_prefill_progress(PrefillStep& prefill, const DecodeStep& decode,
                                              DecodeStateSlot& state) noexcept;

// A submitted unit that fails or is abandoned may have partially changed
// persistent buffers. Poisoning is irreversible: callers must discard the
// associated state slot and fail readiness.
PrefillProgressError poison_prefill(PrefillStep& prefill) noexcept;

} // namespace tatara::runtime
