#include "tatara/runtime/prefill_step.h"

#include "tatara/generated/kernel_library.h"
#include "tatara/generated/model_plan.h"
#include "tatara/model/image_population.h"
#include "tatara/model/sha256.h"
#include "tatara/runtime/execution_identity.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

namespace tatara::runtime {

namespace {

// Host mirror of the sealed steel GEMMParams (layout validated by the A39
// bracket probe against the closure's own struct).
struct SteelAttnGemmParams {
    int M;
    int N;
    int K;
    int lda;
    int ldb;
    int ldd;
    int tiles_n;
    int tiles_m;
    std::int64_t batch_stride_a;
    std::int64_t batch_stride_b;
    std::int64_t batch_stride_d;
    int swizzle_log;
    int gemm_k_iterations_aligned;
    int batch_ndim;
};
static_assert(sizeof(SteelAttnGemmParams) == 72,
              "sealed GEMMParams layout drifted");

using namespace backend::metal;

constexpr std::uint32_t kDenseProjectionThreads = 128;
constexpr std::uint32_t kOutputProjectionThreads = 64;
constexpr std::uint32_t kElementwiseThreads = 256;
constexpr std::uint32_t kGateHoistRowsPerThreadgroup = 4;

using backend::metal::generated::
    kKernelLibraryNativeDenseQgemmN1Threads;
using backend::metal::generated::
    kKernelLibraryNativeDenseQgemmN1TileColumns;
using backend::metal::generated::
    kKernelLibraryNativeDenseQgemmN1TileRows;
using backend::metal::generated::
    kKernelLibraryNativeRoutedQgemmR1TaskCapacity;
using backend::metal::generated::
    kKernelLibraryNativeRoutedQgemmR1Threads;
using backend::metal::generated::
    kKernelLibraryNativeRoutedQgemmR1TileColumns;
using backend::metal::generated::
    kKernelLibraryNativeRoutedQgemmR1TileRows;
using backend::metal::generated::
    kKernelLibraryAttnHeadDimension;
using backend::metal::generated::
    kKernelLibraryAttnQueryHeads;
using backend::metal::generated::
    kKernelLibraryPrefillPackedSlotBits;
using backend::metal::generated::
    kKernelLibraryPrefillStreamingAttentionQueryTileRows;
using backend::metal::generated::
    kKernelLibraryPrefillStreamingAttentionThreads;
using backend::metal::generated::kKernelLibraryPrefillFlashV2QueryTileRows;
using backend::metal::generated::kKernelLibraryPrefillFlashV2Threads;
using backend::metal::generated::
    kKernelLibraryPrefillStagedAttentionKeyTileColumns;
using backend::metal::generated::
    kKernelLibraryPrefillStagedAttentionOutputTileColumns;
using backend::metal::generated::
    kKernelLibraryPrefillStagedAttentionQueryTileRows;
using backend::metal::generated::
    kKernelLibraryPrefillStagedAttentionSoftmaxThreads;
using backend::metal::generated::
    kKernelLibraryPrefillStagedAttentionThreads;

static_assert(sizeof(QuantizedGemmTaskDescriptor) ==
              backend::metal::generated::
                  kKernelLibraryNativeRoutedQgemmR1TaskBytes);

constexpr std::uint64_t kNativeRoutedTaskBytes =
    std::uint64_t{kKernelLibraryNativeRoutedQgemmR1TaskCapacity} *
    sizeof(QuantizedGemmTaskDescriptor);
constexpr std::uint64_t kNativeRoutedArgumentBytes =
    3u * sizeof(std::uint32_t);
constexpr std::uint64_t kNativeRoutedStatusBytes =
    sizeof(QuantizedGemmDeviceTaskStatus);
// Image windows keep every indirect kernel-binding offset below half the
// indirect limit: a window begins at each multiple of 2^31 bytes and spans
// up to two windows, so any tensor whose offset falls inside a window also
// ends inside it.
constexpr std::uint64_t kPrefillImageWindowShift = 31;
constexpr std::uint64_t kPrefillImageWindowBytes =
    1ull << kPrefillImageWindowShift;
constexpr std::uint64_t kNativeRoutedSharedExpertBytes =
    sizeof(std::uint32_t);
constexpr std::uint64_t kNativeRoutedSharedArgumentBytes =
    2u * kNativeRoutedArgumentBytes;
constexpr std::uint64_t kNativeRoutedFixedWorkspaceBytes =
    2u * (kNativeRoutedTaskBytes + kNativeRoutedArgumentBytes) +
    kNativeRoutedSharedExpertBytes +
    kNativeRoutedSharedArgumentBytes;

std::uint32_t ceil_div(std::uint32_t value, std::uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

bool multiply(std::uint64_t left, std::uint64_t right,
              std::uint64_t& product) {
    if (left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left) {
        return false;
    }
    product = left * right;
    return true;
}

bool native_dense_binding_matches(const QuantizedBinding& binding,
                                  std::uint32_t output_columns,
                                  std::uint32_t reduction_columns) {
    if (output_columns == 0 || reduction_columns == 0 ||
        reduction_columns % 64u != 0) {
        return false;
    }
    std::uint64_t weight_bytes = 0;
    std::uint64_t parameter_elements = 0;
    std::uint64_t parameter_bytes = 0;
    return multiply(output_columns, reduction_columns / 2u, weight_bytes) &&
           multiply(output_columns, reduction_columns / 64u,
                    parameter_elements) &&
           multiply(parameter_elements, kBf16Bytes, parameter_bytes) &&
           binding.weight_size_bytes == weight_bytes &&
           binding.scale_size_bytes == parameter_bytes &&
           binding.bias_size_bytes == parameter_bytes;
}

bool valid_native_dense_bindings(const PrefillGeometry& geometry,
                                 const DecodeBindings& bindings) {
    const std::uint64_t gdn_qkv =
        std::uint64_t{geometry.gdn_qk_values} +
        geometry.gdn_value_values;
    const std::uint64_t attention_qgate =
        std::uint64_t{2u} * geometry.query_heads *
        geometry.attention_head_dimension;
    const std::uint64_t attention_kv =
        std::uint64_t{geometry.key_value_heads} *
        geometry.attention_head_dimension;
    if (gdn_qkv > std::numeric_limits<std::uint32_t>::max() ||
        attention_qgate > std::numeric_limits<std::uint32_t>::max() ||
        attention_kv > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    for (const LayerBindings& layer : bindings.layers) {
        if (layer.kind == model::qwen36::LayerKind::GatedDelta) {
            if (!native_dense_binding_matches(
                    layer.gated_delta.qkv,
                    static_cast<std::uint32_t>(gdn_qkv),
                    geometry.hidden) ||
                !native_dense_binding_matches(
                    layer.gated_delta.z, geometry.gdn_value_values,
                    geometry.hidden) ||
                !native_dense_binding_matches(
                    layer.gated_delta.b, geometry.recurrent_heads,
                    geometry.hidden) ||
                !native_dense_binding_matches(
                    layer.gated_delta.a, geometry.recurrent_heads,
                    geometry.hidden) ||
                !native_dense_binding_matches(
                    layer.gated_delta.out, geometry.hidden,
                    geometry.gdn_value_values)) {
                return false;
            }
        } else if (layer.kind == model::qwen36::LayerKind::FullAttention) {
            if (!native_dense_binding_matches(
                    layer.attention.query,
                    static_cast<std::uint32_t>(attention_qgate),
                    geometry.hidden) ||
                !native_dense_binding_matches(
                    layer.attention.key,
                    static_cast<std::uint32_t>(attention_kv),
                    geometry.hidden) ||
                !native_dense_binding_matches(
                    layer.attention.value,
                    static_cast<std::uint32_t>(attention_kv),
                    geometry.hidden) ||
                !native_dense_binding_matches(
                    layer.attention.out, geometry.hidden,
                    geometry.attention_vector_values)) {
                return false;
            }
        } else {
            return false;
        }
    }
    return !bindings.layers.empty();
}

bool native_routed_binding_matches(
    const QuantizedBinding& binding, std::uint32_t experts,
    std::uint32_t output_columns, std::uint32_t reduction_columns) {
    std::uint64_t rows = 0;
    if (!multiply(experts, output_columns, rows) ||
        rows > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    return native_dense_binding_matches(
        binding, static_cast<std::uint32_t>(rows), reduction_columns);
}

bool valid_native_routed_bindings(const PrefillGeometry& geometry,
                                  const DecodeBindings& bindings) {
    for (const LayerBindings& layer : bindings.layers) {
        if ((layer.kind != model::qwen36::LayerKind::GatedDelta &&
             layer.kind != model::qwen36::LayerKind::FullAttention) ||
            !native_routed_binding_matches(
                layer.expert_gate, geometry.experts,
                geometry.expert_dimension, geometry.hidden) ||
            !native_routed_binding_matches(
                layer.expert_up, geometry.experts,
                geometry.expert_dimension, geometry.hidden) ||
            !native_routed_binding_matches(
                layer.expert_down, geometry.experts, geometry.hidden,
                geometry.expert_dimension) ||
            !native_dense_binding_matches(
                layer.shared_gate, geometry.expert_dimension,
                geometry.hidden) ||
            !native_dense_binding_matches(
                layer.shared_up, geometry.expert_dimension,
                geometry.hidden) ||
            !native_dense_binding_matches(
                layer.shared_down, geometry.hidden,
                geometry.expert_dimension)) {
            return false;
        }
    }
    return !bindings.layers.empty();
}

bool allocate_zeroed(const MetalDevice& device, std::uint64_t size_bytes, MetalBuffer& buffer) {
    if (size_bytes == 0) {
        return true;
    }
    auto result = create_shared_buffer(device, size_bytes);
    if (!result) {
        return false;
    }
    buffer = std::move(*result.buffer);
    std::memset(buffer.contents(), 0, static_cast<std::size_t>(size_bytes));
    return true;
}

bool enough(const MetalBuffer& buffer, std::uint64_t bytes) {
    return bytes == 0 ? !buffer : buffer && buffer.size_bytes() >= bytes;
}

bool valid_policy(const PrefillExecutionPolicy& policy, const PrefillGeometry& geometry) {
    const bool known_schedule = policy.geometry.schedule == PrefillSchedule::ChunkMajor ||
                                policy.geometry.schedule == PrefillSchedule::LayerMajor;
    const bool known_selector = policy.router_selector == PrefillRouterSelector::Serial ||
                                policy.router_selector == PrefillRouterSelector::Parallel;
    const bool known_recurrence = policy.gdn_recurrence == PrefillGdnRecurrence::SerialSteps ||
                                  policy.gdn_recurrence == PrefillGdnRecurrence::RegisterLoop ||
                                  policy.gdn_recurrence == PrefillGdnRecurrence::RegisterLoopTape;
    const bool known_attention =
        policy.attention_kernel ==
            PrefillAttentionKernel::PartialCombine ||
        policy.attention_kernel ==
            PrefillAttentionKernel::StagedGemmAdaptive ||
        policy.attention_kernel ==
            PrefillAttentionKernel::StreamingFlashAdaptive ||
        policy.attention_kernel ==
            PrefillAttentionKernel::FlashMmaV2 ||
        policy.attention_kernel ==
            PrefillAttentionKernel::SteelGemm;
    const bool known_dense_qgemm =
        policy.dense_qgemm == QuantizedGemmPolicy::ExactRow ||
        policy.dense_qgemm == QuantizedGemmPolicy::NativeDenseMma;
    const bool known_routed_qgemm =
        policy.routed_qgemm == QuantizedGemmPolicy::ExactRow ||
        policy.routed_qgemm == QuantizedGemmPolicy::NativeRaggedMma;
    const bool routed_schedule_matches =
        policy.routed_qgemm == QuantizedGemmPolicy::ExactRow ||
        policy.geometry.schedule == PrefillSchedule::LayerMajor;
    const bool routed_plan_matches =
        policy.routed_qgemm == QuantizedGemmPolicy::ExactRow ||
        (geometry.hidden ==
             backend::metal::generated::kKernelLibraryHidden &&
         geometry.experts ==
             backend::metal::generated::kKernelLibraryMoeExperts &&
         geometry.active_experts ==
             backend::metal::generated::
                 kKernelLibraryMoeActiveExperts &&
         geometry.expert_dimension ==
             backend::metal::generated::
                 kKernelLibraryMoeExpertDimension);
    constexpr std::uint32_t kSteelSignedExtentMaximum =
        static_cast<std::uint32_t>(
            std::numeric_limits<std::int32_t>::max());
    const bool dense_steel_matches =
        !policy.native_dense_steel ||
        (policy.dense_qgemm ==
             QuantizedGemmPolicy::NativeDenseMma &&
         geometry.maximum_block_rows <= kSteelSignedExtentMaximum &&
         geometry.hidden <= kSteelSignedExtentMaximum &&
         geometry.gdn_projection_rows <=
             kSteelSignedExtentMaximum &&
         geometry.gdn_value_values <= kSteelSignedExtentMaximum &&
         geometry.attention_projection_rows <=
             kSteelSignedExtentMaximum &&
         geometry.attention_vector_values <=
             kSteelSignedExtentMaximum);
    const bool dense_steel_gdn_bm64_matches =
        !policy.native_dense_steel_gdn_bm64_wm2_wn2 ||
        (policy.native_dense_steel &&
         policy.dense_qgemm ==
             QuantizedGemmPolicy::NativeDenseMma &&
         geometry.hidden ==
             backend::metal::generated::
                 kKernelLibraryHidden);
    const bool shared_native_matches =
        !policy.native_routed_shared_expert ||
        policy.routed_qgemm ==
            QuantizedGemmPolicy::NativeRaggedMma;
    const bool steel_native_matches =
        !policy.native_routed_steel ||
        policy.routed_qgemm ==
            QuantizedGemmPolicy::NativeRaggedMma;
    const bool shared_task_capacity_matches =
        !policy.native_routed_shared_expert ||
        native_routed_shared_task_capacity_supported(
            geometry,
            kKernelLibraryNativeRoutedQgemmR1TileRows,
            kKernelLibraryNativeRoutedQgemmR1TaskCapacity);
    const bool slab_matches = policy.geometry.schedule == PrefillSchedule::LayerMajor
                                  ? geometry.hidden_slab_bytes != 0
                                  : geometry.hidden_slab_bytes == 0;
    const bool hoist_matches = policy.geometry.gdn_gate_hoist ? geometry.gdn_parameter_bytes != 0
                                                              : geometry.gdn_parameter_bytes == 0;
    const bool geometry_matches =
        policy.geometry.schedule == geometry.schedule &&
        policy.geometry.context_capacity == geometry.context_capacity &&
        policy.geometry.maximum_block_rows == geometry.maximum_block_rows &&
        policy.geometry.first_chunk_rows == geometry.first_chunk_rows &&
        policy.geometry.query_tile_rows == geometry.query_tile_rows &&
        policy.geometry.attention_partition == geometry.attention_partition &&
        policy.geometry.exact_rows_per_threadgroup == geometry.exact_rows_per_threadgroup &&
        policy.geometry.gdn_gate_hoist == geometry.gdn_gate_hoist;
    const bool recurrence_matches = policy.gdn_recurrence == PrefillGdnRecurrence::RegisterLoop ||
        policy.gdn_recurrence == PrefillGdnRecurrence::RegisterLoopTape ||
                                    !policy.geometry.gdn_gate_hoist;
    const bool attention_family_matches =
        policy.attention_kernel ==
            PrefillAttentionKernel::PartialCombine ||
        (policy.attention_kernel ==
             PrefillAttentionKernel::StagedGemmAdaptive
             ? policy.staged_attention_minimum_context <
                       policy.geometry.context_capacity &&
                   geometry.attention_staged_score_bytes != 0 &&
                   geometry.attention_staged_score_bytes <=
             geometry.attention_partial_bytes
             : policy.streaming_attention_minimum_context <
                       policy.geometry.context_capacity &&
                   geometry.query_heads ==
                       kKernelLibraryAttnQueryHeads &&
                   geometry.attention_head_dimension ==
                       kKernelLibraryAttnHeadDimension);
    const bool submission_matches =
        policy.maximum_units_per_submission != 0 &&
        policy.maximum_units_per_submission <=
            kPrefillMaximumUnitsPerSubmission &&
        (policy.maximum_units_per_submission == 1 ||
         policy.geometry.schedule == PrefillSchedule::LayerMajor);
    const bool inflight_matches =
        policy.maximum_inflight_units != 0 &&
        policy.maximum_inflight_units <=
            kPrefillMaximumUnitsPerSubmission &&
        (policy.maximum_inflight_units == 1 ||
         policy.geometry.schedule == PrefillSchedule::LayerMajor);
    const bool command_graph_matches =
        policy.command_graph
            ? policy.geometry.schedule == PrefillSchedule::LayerMajor &&
                  policy.dense_qgemm ==
                      QuantizedGemmPolicy::NativeDenseMma &&
                  policy.native_dense_steel &&
                  policy.routed_qgemm ==
                      QuantizedGemmPolicy::NativeRaggedMma &&
                  policy.native_routed_shared_expert &&
                  policy.native_routed_steel &&
                  policy.command_graph_chunk_count != 0 &&
                  policy.command_graph_chunk_count <=
                      kPrefillMaximumUnitsPerSubmission &&
                  (!policy.command_graph_lane_events ||
                   policy.command_graph_chunk_count == 3u)
            : policy.command_graph_chunk_count == 1 &&
                  !policy.command_graph_lane_events;
    return known_schedule && known_selector && known_recurrence &&
           known_attention &&
           known_dense_qgemm && known_routed_qgemm &&
           routed_schedule_matches && routed_plan_matches &&
           dense_steel_matches && dense_steel_gdn_bm64_matches &&
           shared_native_matches && steel_native_matches &&
           shared_task_capacity_matches &&
           recurrence_matches && attention_family_matches &&
           submission_matches && inflight_matches &&
           command_graph_matches &&
           slab_matches && hoist_matches && geometry_matches &&
           policy.geometry.context_capacity != 0 && policy.geometry.maximum_block_rows != 0 &&
           policy.geometry.first_chunk_rows != 0 && policy.geometry.query_tile_rows != 0 &&
           policy.geometry.attention_partition == kAttentionPartition &&
           policy.geometry.exact_rows_per_threadgroup != 0;
}

bool valid_geometry(const PrefillGeometry& geometry) {
    return geometry.hidden != 0 && geometry.vocabulary != 0 && geometry.query_heads != 0 &&
           geometry.key_value_heads != 0 && geometry.attention_head_dimension != 0 &&
           geometry.recurrent_heads != 0 && geometry.state_dimension != 0 &&
           geometry.experts != 0 && geometry.active_experts != 0 &&
           geometry.expert_dimension != 0 && geometry.gdn_projection_rows != 0 &&
           geometry.gdn_qk_values != 0 && geometry.gdn_value_values != 0 &&
           geometry.attention_projection_rows != 0 && geometry.attention_vector_values != 0 &&
           geometry.token_bytes != 0 && geometry.block_hidden_bytes != 0 &&
           geometry.attention_staged_score_bytes != 0 &&
           geometry.attention_staged_score_bytes <=
               geometry.attention_partial_bytes &&
           geometry.reusable_scratch_bytes != 0 && geometry.steady_prefill_bytes != 0;
}

bool valid_pipelines(const PrefillPipelines& pipelines,
                     const PrefillExecutionPolicy& policy) {
    const bool dense_available =
        policy.dense_qgemm == QuantizedGemmPolicy::ExactRow ||
        (pipelines.native_dense_qgemm &&
         (!policy.native_dense_steel ||
          pipelines.native_dense_steel) &&
         (!policy.native_dense_steel_gdn_bm64_wm2_wn2 ||
          pipelines.native_dense_steel_gdn_bm64_wm2_wn2));
    const bool routed_available =
        policy.routed_qgemm == QuantizedGemmPolicy::ExactRow ||
        (pipelines.native_routed_task_builder &&
         pipelines.native_routed_upgate &&
         pipelines.native_routed_down &&
         (!policy.native_routed_steel ||
          (pipelines.native_routed_steel_upgate &&
           pipelines.native_routed_steel_down)));
    const bool command_graph_available =
        !policy.command_graph ||
        pipelines.expert_union_fused_tasks;
    const bool attention_available =
        policy.attention_kernel ==
            PrefillAttentionKernel::PartialCombine ||
        (policy.attention_kernel ==
             PrefillAttentionKernel::StagedGemmAdaptive
             ? pipelines.attention_staged_scores &&
                   pipelines.attention_staged_softmax &&
                   pipelines.attention_staged_values
             : policy.attention_kernel ==
                       PrefillAttentionKernel::FlashMmaV2
                   ? static_cast<bool>(pipelines.attention_flash_v2)
                   : policy.attention_kernel ==
                             PrefillAttentionKernel::SteelGemm
                         ? pipelines.attention_steel_scores &&
                               pipelines.attention_steel_values &&
                               pipelines.attention_softmax_bf16 &&
                               pipelines.attention_gate_apply
                         : static_cast<bool>(
                               pipelines.attention_streaming));
    return dense_available && routed_available &&
           command_graph_available &&
           attention_available &&
           pipelines.embed && pipelines.rms &&
           pipelines.residual && pipelines.gdn_project &&
           pipelines.gdn_conv && pipelines.gdn_gates && pipelines.gdn_recurrence_step &&
           (policy.gdn_recurrence != PrefillGdnRecurrence::RegisterLoopTape ||
            (pipelines.gdn_conv_tape && pipelines.gdn_recurrence_tape)) &&
           (!policy.conditioning_capture ||
            static_cast<bool>(pipelines.capture_rows)) &&
           pipelines.gdn_recurrence_block && pipelines.gdn_recurrence_gates &&
           pipelines.gdn_gate_norm && pipelines.attn_project && pipelines.attn_qk_rope &&
           pipelines.attention_partial && pipelines.attention_combine && pipelines.out_projection &&
           pipelines.router && pipelines.router_select_serial && pipelines.router_select_parallel &&
           pipelines.expert_union && pipelines.expert_upgate && pipelines.expert_down &&
           pipelines.expert_combine;
}

PrefillEncodeError request_error(PrefillRequestError error) {
    switch (error) {
    case PrefillRequestError::None:
        return PrefillEncodeError::None;
    case PrefillRequestError::EmptyPrefix:
        return PrefillEncodeError::EmptyPrefix;
    case PrefillRequestError::BlockOutOfRange:
        return PrefillEncodeError::BlockOutOfRange;
    case PrefillRequestError::ContextOutOfRange:
        return PrefillEncodeError::ContextOutOfRange;
    case PrefillRequestError::ContextOverflow:
        return PrefillEncodeError::ContextOverflow;
    case PrefillRequestError::TokenOutOfRange:
        return PrefillEncodeError::TokenOutOfRange;
    }
    return PrefillEncodeError::BindingMismatch;
}

std::uint32_t first_chunk_rows(const PrefillExecutionPolicy& policy,
                               std::uint32_t context_base) {
    return policy.geometry.schedule == PrefillSchedule::LayerMajor &&
                   context_base == 0
               ? policy.geometry.first_chunk_rows
               : policy.geometry.maximum_block_rows;
}

std::uint32_t chunk_count(const PrefillExecutionPolicy& policy,
                          std::uint32_t context_base,
                          std::uint32_t rows) {
    const std::uint32_t first = first_chunk_rows(policy, context_base);
    if (rows <= first) {
        return 1;
    }
    return 1u + ceil_div(rows - first, policy.geometry.maximum_block_rows);
}

struct Chunk {
    std::uint32_t offset;
    std::uint32_t rows;
    std::uint32_t ordinal;
};

Chunk chunk_at(const PrefillExecutionPolicy& policy,
               std::uint32_t context_base, std::uint32_t total_rows,
               std::uint32_t ordinal) {
    const std::uint32_t first = first_chunk_rows(policy, context_base);
    if (ordinal == 0) {
        return {.offset = 0, .rows = total_rows < first ? total_rows : first, .ordinal = 0};
    }
    const std::uint32_t offset = first + (ordinal - 1u) * policy.geometry.maximum_block_rows;
    const std::uint32_t remaining = total_rows - offset;
    return {
        .offset = offset,
        .rows = remaining < policy.geometry.maximum_block_rows ? remaining
                                                               : policy.geometry.maximum_block_rows,
        .ordinal = ordinal,
    };
}

std::uint32_t maximum_routed_task_count(
    const PrefillStep& step, std::uint32_t block_rows) {
    const std::uint64_t routes_per_position =
        std::uint64_t{step.geometry.active_experts} +
        (step.policy.native_routed_shared_expert ? 1u : 0u);
    const std::uint64_t route_count =
        std::uint64_t{block_rows} * routes_per_position;
    return static_cast<std::uint32_t>(
        std::min<std::uint64_t>(
            route_count,
            kKernelLibraryNativeRoutedQgemmR1TaskCapacity));
}

// Latches and gates every command operation. Once one bind or dispatch fails,
// later calls are skipped so no partially rebound pipeline can be dispatched.
struct UnprofiledDispatches {
    static constexpr bool profiled = false;
};

struct ProfiledDispatches {
    static constexpr bool profiled = true;
    PrefillProfiler& profiler;
    const MetalCounterSampleBuffer& samples;
    bool stage_event_completed{false};
};

template <typename DispatchProfile>
struct Encoder : private DispatchProfile {
    static constexpr bool profiled = DispatchProfile::profiled;

    Encoder(MetalComputePass& compute_pass, const MetalBuffer& model_image,
            std::span<const std::uint64_t> tensor_offsets,
            DispatchProfile dispatch_profile)
        : DispatchProfile(dispatch_profile), pass(compute_pass),
          image(model_image), offsets(tensor_offsets) {}

    MetalComputePass& pass;
    const MetalBuffer& image;
    std::span<const std::uint64_t> offsets;
    MetalCommandError error{MetalCommandError::None};

    bool stage_mode() const {
        if constexpr (DispatchProfile::profiled) {
            return this->profiler.status().sampling_mode ==
                   CounterSamplingMode::StageBoundaryEncoderSplit;
        }
        return false;
    }

    void prepare() {
        if constexpr (DispatchProfile::profiled) {
            if (!failed() && stage_mode() && this->stage_event_completed) {
                CounterSamplePair pair;
                if (this->profiler.next_stage_sample_pair(pair) !=
                    PrefillProfilerError::None) {
                    return;
                }
                const CounterStageSampleError stage_error =
                    split_stage_sampled_compute_pass(pass, this->samples, pair);
                if (stage_error != CounterStageSampleError::None) {
                    this->profiler.fail_stage(stage_error);
                    return;
                }
                this->stage_event_completed = false;
            }
        }
    }

    bool failed() const {
        if (error != MetalCommandError::None) {
            return true;
        }
        if constexpr (DispatchProfile::profiled) {
            return this->profiler.error() != PrefillProfilerError::None;
        }
        return false;
    }
    void check(MetalCommandError result) {
        if (!failed() && result != MetalCommandError::None) {
            error = result;
        }
    }
    void pipeline(const MetalComputePipeline& pipeline) {
        prepare();
        if (!failed()) {
            check(set_compute_pipeline(pass, pipeline));
        }
    }
    void buffer(const MetalBuffer& buffer, std::uint64_t offset, std::uint32_t index) {
        prepare();
        if (!failed()) {
            check(set_buffer(pass, buffer, offset, index));
        }
    }
    void weight(std::uint32_t tensor, std::uint32_t index) {
        prepare();
        if (failed()) {
            return;
        }
        if (tensor >= offsets.size() || offsets[tensor] == model::kExcludedTensorOffset) {
            check(MetalCommandError::InvalidBufferOffset);
            return;
        }
        check(set_buffer(pass, image, offsets[tensor], index));
    }
    void quantized(const QuantizedBinding& binding, std::uint32_t first_index) {
        weight(binding.weight, first_index);
        weight(binding.scales, first_index + 1u);
        weight(binding.biases, first_index + 2u);
    }
    void constant(std::uint32_t value, std::uint32_t index) {
        prepare();
        if (!failed()) {
            check(set_bytes(pass, &value, sizeof(value), index));
        }
    }
    void constant_bytes(const void* value, std::size_t size,
                        std::uint32_t index) {
        prepare();
        if (!failed()) {
            check(set_bytes(pass, value, size, index));
        }
    }
    void constant(std::uint64_t value, std::uint32_t index) {
        prepare();
        if (!failed()) {
            check(set_bytes(pass, &value, sizeof(value), index));
        }
    }
    template <PrefillProfileEventClass EventClass>
    void dispatch(MetalSize groups, MetalSize threads, std::uint64_t layer,
                  const Chunk& chunk, std::uint32_t operation_row_begin = 0,
                  std::uint32_t operation_row_count = 0) {
        static_assert(is_prefill_profile_dispatch_class(EventClass));
        prepare();
        if (failed()) {
            return;
        }
        if constexpr (DispatchProfile::profiled) {
            const PrefillProfileEvent event{
                .event_class = EventClass,
                .layer_index = layer,
                .chunk_ordinal = chunk.ordinal,
                .chunk_offset = chunk.offset,
                .chunk_rows = chunk.rows,
                .operation_row_begin = operation_row_begin,
                .operation_row_count =
                    operation_row_count == 0 ? chunk.rows
                                             : operation_row_count,
            };
            PrefillProfileDispatchTicket ticket;
            if (this->profiler.begin_dispatch(event, ticket) !=
                PrefillProfilerError::None) {
                return;
            }
            MetalCommandError dispatch_error = MetalCommandError::None;
            if (stage_mode()) {
                dispatch_error = dispatch_threadgroups(pass, groups, threads);
                this->profiler.complete_dispatch(ticket);
                this->stage_event_completed = true;
            } else {
                CounterSampleError sample_error =
                    sample_counter(pass, this->samples, ticket.samples.start);
                if (sample_error != CounterSampleError::None) {
                    this->profiler.fail_counter(sample_error);
                    return;
                }
                dispatch_error = dispatch_threadgroups(pass, groups, threads);
                sample_error =
                    sample_counter(pass, this->samples, ticket.samples.end);
                if (sample_error != CounterSampleError::None) {
                    this->profiler.fail_counter(sample_error);
                } else {
                    this->profiler.complete_dispatch(ticket);
                }
            }
            if (dispatch_error != MetalCommandError::None &&
                error == MetalCommandError::None) {
                error = dispatch_error;
            }
        } else {
            check(dispatch_threadgroups(pass, groups, threads));
        }
    }
    template <PrefillProfileEventClass EventClass>
    void indirect(const MetalBuffer& arguments, std::uint64_t offset,
                  MetalSize threads, std::uint64_t layer,
                  const Chunk& chunk) {
        static_assert(is_prefill_profile_dispatch_class(EventClass));
        prepare();
        if (failed()) {
            return;
        }
        if constexpr (DispatchProfile::profiled) {
            const PrefillProfileEvent event{
                .event_class = EventClass,
                .layer_index = layer,
                .chunk_ordinal = chunk.ordinal,
                .chunk_offset = chunk.offset,
                .chunk_rows = chunk.rows,
                .operation_row_begin = 0,
                .operation_row_count = chunk.rows,
            };
            PrefillProfileDispatchTicket ticket;
            if (this->profiler.begin_dispatch(event, ticket) !=
                PrefillProfilerError::None) {
                return;
            }
            MetalCommandError dispatch_error = MetalCommandError::None;
            if (stage_mode()) {
                dispatch_error =
                    dispatch_threadgroups_indirect(pass, arguments, offset, threads);
                this->profiler.complete_dispatch(ticket);
                this->stage_event_completed = true;
            } else {
                CounterSampleError sample_error =
                    sample_counter(pass, this->samples, ticket.samples.start);
                if (sample_error != CounterSampleError::None) {
                    this->profiler.fail_counter(sample_error);
                    return;
                }
                dispatch_error =
                    dispatch_threadgroups_indirect(pass, arguments, offset, threads);
                sample_error =
                    sample_counter(pass, this->samples, ticket.samples.end);
                if (sample_error != CounterSampleError::None) {
                    this->profiler.fail_counter(sample_error);
                } else {
                    this->profiler.complete_dispatch(ticket);
                }
            }
            if (dispatch_error != MetalCommandError::None &&
                error == MetalCommandError::None) {
                error = dispatch_error;
            }
        } else {
            check(dispatch_threadgroups_indirect(pass, arguments, offset,
                                                 threads));
        }
    }
    void unprofiled_dispatch(MetalSize groups, MetalSize threads) {
        static_assert(!DispatchProfile::profiled);
        if (!failed()) {
            check(dispatch_threadgroups(pass, groups, threads));
        }
    }
    void barrier() {
        prepare();
        if (!failed()) {
            check(memory_barrier(pass));
        }
    }
};

enum class RecordedBindingKind : std::uint8_t {
    None,
    Buffer,
    Constant32,
    Constant64,
};

struct RecordedBinding {
    RecordedBindingKind kind{RecordedBindingKind::None};
    const MetalBuffer* buffer{nullptr};
    std::uint64_t offset{0};
    std::uint64_t value{0};
};

struct RecordedCommand {
    const MetalComputePipeline* pipeline{nullptr};
    std::array<RecordedBinding, kMaxBufferArgumentIndex + 1u> bindings{};
    MetalSize groups{0, 0, 0};
    MetalSize threads{0, 0, 0};
    std::uint32_t local_level{0};
    PrefillProfileEventClass event_class{
        PrefillProfileEventClass::MoeRouterSelectParallel};
};

struct RecordedNode {
    std::vector<RecordedCommand> commands;
};

struct OrderedRecordedCommand {
    const RecordedCommand* command{nullptr};
};

std::uint64_t scratch_lane_stride(
    const PrefillStep& step, const MetalBuffer& buffer) {
    const PrefillGeometry& geometry = step.geometry;
    if (&buffer == &step.block_hidden ||
        &buffer == &step.normalized ||
        &buffer == &step.branch ||
        &buffer == &step.moe_output) {
        return geometry.block_hidden_bytes;
    }
    if (&buffer == &step.gdn_projection) {
        return geometry.gdn_projection_bytes;
    }
    if (&buffer == &step.gdn_qk) {
        return geometry.gdn_qk_bytes;
    }
    if (&buffer == &step.gdn_value ||
        &buffer == &step.gdn_gate ||
        &buffer == &step.gdn_recurrence ||
        &buffer == &step.gdn_gated) {
        return geometry.gdn_value_bytes;
    }
    if (&buffer == &step.gdn_decay ||
        &buffer == &step.gdn_beta) {
        return geometry.gdn_parameter_bytes;
    }
    if (&buffer == &step.attention_projection) {
        return geometry.attention_projection_bytes;
    }
    if (&buffer == &step.attention_query ||
        &buffer == &step.attention_gate ||
        &buffer == &step.attention_attended) {
        return geometry.attention_vector_bytes;
    }
    if (&buffer == &step.attention_partials) {
        return geometry.attention_partial_bytes;
    }
    if (&buffer == &step.router_logits) {
        return geometry.moe_logits_bytes;
    }
    if (&buffer == &step.expert_ids) {
        return geometry.moe_id_bytes;
    }
    if (&buffer == &step.expert_coefficients) {
        return geometry.moe_coefficient_bytes;
    }
    if (&buffer == &step.shared_coefficients) {
        return geometry.moe_shared_coefficient_bytes;
    }
    if (&buffer == &step.expert_counts) {
        return geometry.moe_count_bytes;
    }
    if (&buffer == &step.expert_lists) {
        return geometry.moe_list_bytes;
    }
    if (&buffer == &step.active_experts) {
        return geometry.moe_active_bytes;
    }
    if (&buffer == &step.expert_arguments) {
        return geometry.moe_indirect_argument_bytes;
    }
    if (&buffer == &step.expert_hidden) {
        return geometry.moe_hidden_bytes;
    }
    if (&buffer == &step.expert_partials) {
        return geometry.moe_partial_bytes;
    }
    if (&buffer == &step.native_routed_up_tasks ||
        &buffer == &step.native_routed_down_tasks) {
        return kNativeRoutedTaskBytes;
    }
    if (&buffer == &step.native_routed_up_arguments ||
        &buffer == &step.native_routed_down_arguments) {
        return kNativeRoutedArgumentBytes;
    }
    return 0;
}

template <typename Visitor>
void visit_prefill_scratch_lane_buffers(
    PrefillStep& step, Visitor&& visit) {
    visit(step.block_hidden);
    visit(step.normalized);
    visit(step.branch);
    visit(step.moe_output);
    visit(step.gdn_projection);
    visit(step.gdn_qk);
    visit(step.gdn_value);
    visit(step.gdn_gate);
    visit(step.gdn_recurrence);
    visit(step.gdn_gated);
    visit(step.gdn_decay);
    visit(step.gdn_beta);
    visit(step.attention_projection);
    visit(step.attention_query);
    visit(step.attention_gate);
    visit(step.attention_attended);
    visit(step.attention_partials);
    visit(step.router_logits);
    visit(step.expert_ids);
    visit(step.expert_coefficients);
    visit(step.shared_coefficients);
    visit(step.expert_counts);
    visit(step.expert_lists);
    visit(step.active_experts);
    visit(step.expert_arguments);
    visit(step.expert_hidden);
    visit(step.expert_partials);
    visit(step.native_routed_up_tasks);
    visit(step.native_routed_up_arguments);
    visit(step.native_routed_down_tasks);
    visit(step.native_routed_down_arguments);
}

struct RecordingEncoder {
    static constexpr bool profiled = false;

    RecordingEncoder(
        PrefillStep& prefill, const MetalBuffer& model_image,
        std::span<const std::uint64_t> tensor_offsets,
        std::span<const MetalBuffer> model_image_windows,
        std::span<const PrefillCommandGraph::BufferWindow>
            scratch_windows_value,
        std::uint32_t scratch_lane,
        std::vector<RecordedCommand>& output)
        : step(prefill), image(model_image), offsets(tensor_offsets),
          image_windows(model_image_windows),
          scratch_windows(scratch_windows_value),
          lane(scratch_lane), commands(output) {}

    PrefillStep& step;
    const MetalBuffer& image;
    std::span<const std::uint64_t> offsets;
    std::span<const MetalBuffer> image_windows;
    std::span<const PrefillCommandGraph::BufferWindow>
        scratch_windows;
    std::uint32_t lane{0};
    std::vector<RecordedCommand>& commands;
    const MetalComputePipeline* current_pipeline{nullptr};
    std::array<RecordedBinding, kMaxBufferArgumentIndex + 1u>
        current_bindings{};
    MetalCommandError error{MetalCommandError::None};
    std::uint32_t local_level{0};
    bool level_has_command{false};

    bool failed() const {
        return error != MetalCommandError::None;
    }

    void check(MetalCommandError result) {
        if (!failed() && result != MetalCommandError::None) {
            error = result;
        }
    }

    void pipeline(const MetalComputePipeline& pipeline_state) {
        if (!failed()) {
            if (!pipeline_state) {
                error = MetalCommandError::InvalidPipeline;
            } else {
                current_pipeline = &pipeline_state;
            }
        }
    }

    void buffer(
        const MetalBuffer& buffer_value, std::uint64_t offset,
        std::uint32_t index) {
        if (failed()) {
            return;
        }
        if (!buffer_value) {
            error = MetalCommandError::InvalidBuffer;
            return;
        }
        if (index > kMaxBufferArgumentIndex) {
            error = MetalCommandError::InvalidBufferIndex;
            return;
        }
        const std::uint64_t stride =
            scratch_lane_stride(step, buffer_value);
        std::uint64_t lane_base = 0;
        std::uint64_t absolute_offset = 0;
        if ((stride != 0 &&
             !multiply(stride, lane, lane_base)) ||
            lane_base > buffer_value.size_bytes() ||
            offset >= buffer_value.size_bytes() - lane_base) {
            error = MetalCommandError::InvalidBufferOffset;
            return;
        }
        absolute_offset = lane_base + offset;
        const PrefillBufferWindowPlan window_plan =
            plan_prefill_buffer_window(
                buffer_value.size_bytes(), absolute_offset);
        if (!window_plan) {
            error = MetalCommandError::InvalidBufferOffset;
            return;
        }
        if (window_plan.use_window) {
            const auto window = std::find_if(
                scratch_windows.begin(), scratch_windows.end(),
                [&](const PrefillCommandGraph::BufferWindow&
                        candidate) {
                    return candidate.source == &buffer_value &&
                           candidate.source_begin ==
                               window_plan.source_begin;
                });
            if (window == scratch_windows.end() ||
                window_plan.binding_offset >=
                    window->window.size_bytes()) {
                error = MetalCommandError::InvalidBufferOffset;
                return;
            }
            current_bindings[index] = {
                .kind = RecordedBindingKind::Buffer,
                .buffer = &window->window,
                .offset = window_plan.binding_offset,
            };
            return;
        }
        current_bindings[index] = {
            .kind = RecordedBindingKind::Buffer,
            .buffer = &buffer_value,
            .offset = absolute_offset,
        };
    }

    void weight(std::uint32_t tensor, std::uint32_t index) {
        if (failed()) {
            return;
        }
        if (tensor >= offsets.size() ||
            offsets[tensor] == model::kExcludedTensorOffset) {
            error = MetalCommandError::InvalidBufferOffset;
            return;
        }
        if (index > kMaxBufferArgumentIndex ||
            offsets[tensor] >= image.size_bytes()) {
            error = index > kMaxBufferArgumentIndex
                        ? MetalCommandError::InvalidBufferIndex
                        : MetalCommandError::InvalidBufferOffset;
            return;
        }
        if (!image_windows.empty()) {
            const std::uint64_t window =
                offsets[tensor] >> kPrefillImageWindowShift;
            const std::uint64_t window_base =
                window << kPrefillImageWindowShift;
            if (window >= image_windows.size()) {
                error = MetalCommandError::InvalidBufferOffset;
                return;
            }
            current_bindings[index] = {
                .kind = RecordedBindingKind::Buffer,
                .buffer = &image_windows[window],
                .offset = offsets[tensor] - window_base,
            };
            return;
        }
        if (offsets[tensor] >=
            kIndirectKernelBufferOffsetLimitBytes) {
            error = MetalCommandError::InvalidBufferOffset;
            return;
        }
        current_bindings[index] = {
            .kind = RecordedBindingKind::Buffer,
            .buffer = &image,
            .offset = offsets[tensor],
        };
    }

    void quantized(
        const QuantizedBinding& binding,
        std::uint32_t first_index) {
        weight(binding.weight, first_index);
        weight(binding.scales, first_index + 1u);
        weight(binding.biases, first_index + 2u);
    }

    void constant(std::uint32_t value, std::uint32_t index) {
        if (failed()) {
            return;
        }
        if (index > kMaxBufferArgumentIndex) {
            error = MetalCommandError::InvalidBufferIndex;
            return;
        }
        current_bindings[index] = {
            .kind = RecordedBindingKind::Constant32,
            .value = value,
        };
    }

    void constant(std::uint64_t value, std::uint32_t index) {
        if (failed()) {
            return;
        }
        if (index > kMaxBufferArgumentIndex) {
            error = MetalCommandError::InvalidBufferIndex;
            return;
        }
        current_bindings[index] = {
            .kind = RecordedBindingKind::Constant64,
            .value = value,
        };
    }

    void record_dispatch(
        MetalSize groups, MetalSize threads,
        PrefillProfileEventClass event_class =
            PrefillProfileEventClass::MoeRouterSelectParallel) {
        if (failed()) {
            return;
        }
        if (current_pipeline == nullptr) {
            error = MetalCommandError::InvalidPipeline;
            return;
        }
        commands.push_back({
            .pipeline = current_pipeline,
            .bindings = current_bindings,
            .groups = groups,
            .threads = threads,
            .local_level = local_level,
            .event_class = event_class,
        });
        level_has_command = true;
    }

    template <PrefillProfileEventClass EventClass>
    void dispatch(
        MetalSize groups, MetalSize threads, std::uint64_t,
        const Chunk&, std::uint32_t = 0, std::uint32_t = 0) {
        static_assert(is_prefill_profile_dispatch_class(EventClass));
        record_dispatch(groups, threads, EventClass);
    }

    template <PrefillProfileEventClass EventClass>
    void indirect(
        const MetalBuffer&, std::uint64_t, MetalSize threads,
        std::uint64_t, const Chunk& chunk) {
        static_assert(is_prefill_profile_dispatch_class(EventClass));
        const std::uint32_t task_count =
            maximum_routed_task_count(step, chunk.rows);
        if constexpr (
            EventClass ==
                PrefillProfileEventClass::
                    MoeNativeRoutedSharedUpGate ||
            EventClass ==
                PrefillProfileEventClass::MoeNativeRoutedUpGate) {
            record_dispatch(
                {
                    .width = task_count,
                    .height = ceil_div(
                        step.geometry.expert_dimension,
                        kKernelLibraryNativeRoutedQgemmR1TileColumns),
                    .depth = 1,
                },
                threads, EventClass);
        } else if constexpr (
            EventClass ==
                PrefillProfileEventClass::
                    MoeNativeRoutedSharedDown ||
            EventClass ==
                PrefillProfileEventClass::MoeNativeRoutedDown) {
            record_dispatch(
                {
                    .width = task_count,
                    .height = ceil_div(
                        step.geometry.hidden,
                        kKernelLibraryNativeRoutedQgemmR1TileColumns),
                    .depth = 1,
                },
                threads, EventClass);
        } else {
            error = MetalCommandError::InvalidDispatchExtent;
        }
    }

    void unprofiled_dispatch(MetalSize groups, MetalSize threads) {
        record_dispatch(groups, threads);
    }

    void barrier() {
        if (!failed() && level_has_command) {
            ++local_level;
            level_has_command = false;
        }
    }
};

template <PrefillProfileEventClass EventClass, typename CommandEncoder>
void encode_native_dense_qgemm(
    CommandEncoder& encode,
    const PrefillStep& step,
    const MetalBuffer& activations,
    std::uint64_t activation_offset, const QuantizedBinding& weights,
    const MetalBuffer& output, std::uint64_t output_offset,
    std::uint32_t rows, std::uint32_t output_columns,
    std::uint32_t reduction_columns,
    std::uint64_t activation_row_stride_elements,
    std::uint64_t output_row_stride_elements, std::size_t layer,
    const Chunk& chunk) {
    std::uint32_t fallback_row_begin = 0U;
    if constexpr (
        EventClass ==
        PrefillProfileEventClass::GdnProjection) {
        constexpr std::uint32_t kBm64TileRows = 64U;
        const std::uint32_t body_rows =
            rows / kBm64TileRows * kBm64TileRows;
        if (step.policy
                .native_dense_steel_gdn_bm64_wm2_wn2 &&
            body_rows != 0U &&
            output_columns %
                    kKernelLibraryNativeDenseQgemmN1TileColumns ==
                0U &&
            reduction_columns == step.geometry.hidden &&
            output_row_stride_elements ==
                step.geometry.gdn_projection_rows) {
            const std::uint32_t steel_rows = body_rows;
            const std::uint32_t steel_output_columns =
                output_columns;
            const std::uint32_t steel_reduction_columns =
                reduction_columns;
            const std::uint32_t steel_output_row_stride =
                static_cast<std::uint32_t>(
                    output_row_stride_elements);
            encode.pipeline(
                step.pipelines
                    .native_dense_steel_gdn_bm64_wm2_wn2);
            encode.quantized(weights, 0);
            encode.buffer(
                activations, activation_offset, 3);
            encode.buffer(output, output_offset, 4);
            encode.constant(steel_reduction_columns, 5);
            encode.constant(steel_output_columns, 6);
            encode.constant(steel_rows, 7);
            encode.constant(steel_output_row_stride, 8);
            encode.template dispatch<EventClass>(
                {
                    .width =
                        output_columns /
                        kKernelLibraryNativeDenseQgemmN1TileColumns,
                    .height = body_rows / kBm64TileRows,
                    .depth = 1,
                },
                {
                    .width = kSimdgroupThreads,
                    .height = 2,
                    .depth = 2,
                },
                layer, chunk);
            fallback_row_begin = body_rows;
            if (fallback_row_begin == rows) {
                return;
            }
        }
    }
    if (step.policy.native_dense_steel) {
        const std::uint32_t steel_rows =
            rows - fallback_row_begin;
        const std::uint32_t steel_output_columns = output_columns;
        const std::uint32_t steel_reduction_columns =
            reduction_columns;
        const std::uint32_t steel_output_row_stride =
            static_cast<std::uint32_t>(output_row_stride_elements);
        encode.pipeline(step.pipelines.native_dense_steel);
        encode.quantized(weights, 0);
        encode.buffer(
            activations,
            activation_offset +
                std::uint64_t{fallback_row_begin} *
                    activation_row_stride_elements *
                    kBf16Bytes,
            3);
        encode.buffer(
            output,
            output_offset +
                std::uint64_t{fallback_row_begin} *
                    output_row_stride_elements *
                    kBf16Bytes,
            4);
        encode.constant(steel_reduction_columns, 5);
        encode.constant(steel_output_columns, 6);
        encode.constant(steel_rows, 7);
        encode.constant(steel_output_row_stride, 8);
        encode.template dispatch<EventClass>(
            {
                .width = ceil_div(
                    output_columns,
                    kKernelLibraryNativeDenseQgemmN1TileColumns),
                .height = ceil_div(
                    steel_rows,
                    kKernelLibraryNativeDenseQgemmN1TileRows),
                .depth = 1,
            },
            {
                .width = kSimdgroupThreads,
                .height = 2,
                .depth = 2,
            },
            layer, chunk);
        return;
    }
    encode.pipeline(step.pipelines.native_dense_qgemm);
    encode.buffer(activations, activation_offset, 0);
    encode.quantized(weights, 1);
    encode.buffer(output, output_offset, 4);
    encode.constant(rows, 5);
    encode.constant(output_columns, 6);
    encode.constant(reduction_columns, 7);
    encode.constant(activation_row_stride_elements, 8);
    encode.constant(
        std::uint64_t{reduction_columns / 8u}, 9);
    encode.constant(
        std::uint64_t{reduction_columns / 64u}, 10);
    encode.constant(output_row_stride_elements, 11);
    encode.template dispatch<EventClass>(
        {
            .width = ceil_div(
                output_columns,
                kKernelLibraryNativeDenseQgemmN1TileColumns),
            .height = ceil_div(
                rows, kKernelLibraryNativeDenseQgemmN1TileRows),
            .depth = 1,
        },
        {
            .width = kKernelLibraryNativeDenseQgemmN1Threads,
            .height = 1,
            .depth = 1,
        },
        layer, chunk);
}

template <typename CommandEncoder>
void encode_embedding(CommandEncoder& encode, PrefillStep& step,
                      const DecodeBindings& bindings, const Chunk& chunk) {
    const PrefillGeometry& geometry = step.geometry;
    encode.pipeline(step.pipelines.embed);
    encode.quantized(bindings.embedding, 0);
    encode.buffer(step.tokens, std::uint64_t{chunk.offset} * sizeof(std::uint32_t), 3);
    encode.buffer(step.block_hidden, 0, 4);
    encode.template dispatch<PrefillProfileEventClass::Embedding>(
        {.width = geometry.hidden / kElementwiseThreads, .height = chunk.rows, .depth = 1},
        {.width = kElementwiseThreads, .height = 1, .depth = 1},
        kNoPrefillProfileLayerIndex, chunk);
}

template <typename CommandEncoder>
void encode_attention(CommandEncoder& encode, PrefillStep& step,
                      const LayerBindings& bindings,
                      const DecodeLayerState& state, std::size_t layer,
                      const Chunk& chunk, std::uint32_t context_base) {
    const PrefillGeometry& geometry = step.geometry;
    const PrefillPolicy& policy = step.policy.geometry;
    const std::uint32_t block_rows = chunk.rows;
    if (step.policy.dense_qgemm ==
        QuantizedGemmPolicy::NativeDenseMma) {
        const std::uint32_t query_gate_columns =
            2u * geometry.query_heads *
            geometry.attention_head_dimension;
        const std::uint32_t key_value_columns =
            geometry.key_value_heads *
            geometry.attention_head_dimension;
        const std::uint64_t key_offset =
            std::uint64_t{query_gate_columns} * kBf16Bytes;
        const std::uint64_t value_offset =
            std::uint64_t{query_gate_columns + key_value_columns} *
            kBf16Bytes;
        encode_native_dense_qgemm<
            PrefillProfileEventClass::AttentionProjection>(
            encode, step,
            step.normalized, 0, bindings.attention.query,
            step.attention_projection, 0, block_rows,
            query_gate_columns, geometry.hidden, geometry.hidden,
            geometry.attention_projection_rows, layer, chunk);
        encode_native_dense_qgemm<
            PrefillProfileEventClass::AttentionProjection>(
            encode, step,
            step.normalized, 0, bindings.attention.key,
            step.attention_projection, key_offset, block_rows,
            key_value_columns, geometry.hidden, geometry.hidden,
            geometry.attention_projection_rows, layer, chunk);
        encode_native_dense_qgemm<
            PrefillProfileEventClass::AttentionProjection>(
            encode, step,
            step.normalized, 0, bindings.attention.value,
            step.attention_projection, value_offset, block_rows,
            key_value_columns, geometry.hidden, geometry.hidden,
            geometry.attention_projection_rows, layer, chunk);
    } else {
        encode.pipeline(step.pipelines.attn_project);
        encode.buffer(step.normalized, 0, 0);
        encode.quantized(bindings.attention.query, 1);
        encode.quantized(bindings.attention.key, 4);
        encode.quantized(bindings.attention.value, 7);
        encode.buffer(step.attention_projection, 0, 10);
        encode.constant(block_rows, 11);
        encode
            .template dispatch<
                PrefillProfileEventClass::AttentionProjection>(
                {.width = ceil_div(
                     geometry.attention_projection_rows,
                     kDenseProjectionThreads / kSimdgroupThreads),
                 .height = 1,
                 .depth = 1},
                {.width = kDenseProjectionThreads,
                 .height = 1,
                 .depth = 1},
                layer, chunk);
    }
    encode.barrier();

    encode.pipeline(step.pipelines.attn_qk_rope);
    encode.buffer(step.attention_projection, 0, 0);
    encode.weight(bindings.attention.query_norm, 1);
    encode.weight(bindings.attention.key_norm, 2);
    encode.buffer(step.attention_query, 0, 3);
    encode.buffer(step.attention_gate, 0, 4);
    encode.buffer(state.first, 0, 5);
    encode.buffer(state.second, 0, 6);
    encode.constant(context_base, 7);
    encode.constant(policy.context_capacity, 8);
    encode.template dispatch<PrefillProfileEventClass::AttentionQkRope>(
        {.width = geometry.query_heads + geometry.key_value_heads,
         .height = block_rows,
         .depth = 1},
        {.width = geometry.attention_head_dimension, .height = 1, .depth = 1},
        layer, chunk);

    const std::uint64_t vector_row_bytes =
        std::uint64_t{geometry.attention_vector_values} * kBf16Bytes;
    for (std::uint32_t query_base = 0; query_base < block_rows;
         query_base += policy.query_tile_rows) {
        const std::uint32_t tile_rows = block_rows - query_base < policy.query_tile_rows
                                            ? block_rows - query_base
                                            : policy.query_tile_rows;
        const std::uint32_t tile_context = context_base + query_base;
        const std::uint32_t visible = tile_context + tile_rows;
        encode.barrier();
        const std::uint64_t query_offset =
            std::uint64_t{query_base} * vector_row_bytes;
        const bool staged =
            step.policy.attention_kernel ==
                PrefillAttentionKernel::StagedGemmAdaptive &&
            visible >
                step.policy.staged_attention_minimum_context;
        const bool streaming =
            step.policy.attention_kernel ==
                PrefillAttentionKernel::StreamingFlashAdaptive &&
            visible >
                step.policy.streaming_attention_minimum_context;
        const bool flash_v2 =
            step.policy.attention_kernel ==
                PrefillAttentionKernel::FlashMmaV2 &&
            visible >
                step.policy.streaming_attention_minimum_context;
        const bool steel_gemm =
            step.policy.attention_kernel ==
                PrefillAttentionKernel::SteelGemm &&
            visible >
                step.policy.streaming_attention_minimum_context;
        if (steel_gemm) {
            // A39: sealed MLX steel GEMM pair around our softmax and gate.
            // Two dispatches per GEMM (one per KV plane; batch_stride_b is
            // uniform inside a plane), bf16 score plane reusing the
            // partials buffer at half the staged footprint.
            const bool large_tiles =
                visible >= 8192u && tile_rows == 256u &&
                step.pipelines.attention_steel_scores_large &&
                step.pipelines.attention_steel_values_large;
            const std::uint32_t block_m = large_tiles ? 64u : 32u;
            const std::uint32_t block_n = large_tiles ? 64u : 32u;
            const std::uint32_t kv_heads = geometry.key_value_heads;
            const std::uint32_t heads_per_kv =
                geometry.query_heads / kv_heads;
            const std::uint32_t head_dimension = geometry.attention_head_dimension;
            const std::uint64_t kv_plane_bytes =
                std::uint64_t{policy.context_capacity} * head_dimension * 2;
            // Slot key must be unique per (layer, chunk, query tile): the
            // attention section encodes one dispatch group per query tile
            // and the graph path executes them all from this arena.
            const std::uint32_t tile_ordinal =
                policy.query_tile_rows != 0
                    ? (query_base / policy.query_tile_rows) & 15u
                    : 0u;
            const std::uint64_t params_slot =
                ((std::uint64_t{layer} * 8u + (chunk.ordinal & 7u)) * 16u +
                 tile_ordinal) *
                2u * sizeof(SteelAttnGemmParams);
            SteelAttnGemmParams scores_params{
                .M = int(tile_rows),
                .N = int(visible),
                .K = int(head_dimension),
                .lda = int(geometry.query_heads * head_dimension),
                .ldb = int(head_dimension),
                .ldd = int(geometry.query_heads * visible),
                .tiles_n = int((visible + block_n - 1u) / block_n),
                .tiles_m = int((tile_rows + block_m - 1u) / block_m),
                .batch_stride_a = head_dimension,
                .batch_stride_b = 0,
                .batch_stride_d = visible,
                .swizzle_log = 0,
                .gemm_k_iterations_aligned = int(head_dimension / 16u),
                .batch_ndim = 1,
            };
            for (std::uint32_t plane = 0; plane < kv_heads; ++plane) {
                encode.pipeline(
                    large_tiles ? step.pipelines.attention_steel_scores_large
                                : step.pipelines.attention_steel_scores);
                encode.buffer(step.attention_query,
                              query_offset +
                                  std::uint64_t{plane} * heads_per_kv *
                                      head_dimension * 2,
                              0);
                encode.buffer(state.first, plane * kv_plane_bytes, 1);
                encode.buffer(step.tokens, 0, 2);
                encode.buffer(step.attention_partials,
                              std::uint64_t{plane} * heads_per_kv *
                                  visible * 2,
                              3);
                std::memcpy(static_cast<std::uint8_t*>(
                                step.attention_steel_params.contents()) +
                                params_slot,
                            &scores_params, sizeof(scores_params));
                encode.buffer(step.attention_steel_params, params_slot, 4);
                encode.buffer(step.tokens, 0, 5);
                encode.buffer(step.tokens, 0, 6);
                encode.buffer(step.tokens, 0, 7);
                encode.template dispatch<
                    PrefillProfileEventClass::AttentionStagedScores>(
                    {
                        .width = (visible + block_n - 1u) / block_n,
                        .height = (tile_rows + block_m - 1u) / block_m,
                        .depth = heads_per_kv,
                    },
                    {.width = 32, .height = 2, .depth = 2},
                    layer, chunk, query_base, tile_rows);
            }
            encode.barrier();
            encode.pipeline(step.pipelines.attention_softmax_bf16);
            encode.buffer(step.attention_partials, 0, 0);
            encode.constant(visible, 1);
            encode.constant(tile_context, 2);
            encode.template dispatch<
                PrefillProfileEventClass::AttentionStagedSoftmax>(
                {
                    .width = tile_rows,
                    .height = geometry.query_heads,
                    .depth = 1,
                },
                {.width = 256, .height = 1, .depth = 1},
                layer, chunk, query_base, tile_rows);
            encode.barrier();
            SteelAttnGemmParams values_params{
                .M = int(tile_rows),
                .N = int(head_dimension),
                .K = int(visible),
                .lda = int(geometry.query_heads * visible),
                .ldb = int(head_dimension),
                .ldd = int(geometry.query_heads * head_dimension),
                .tiles_n = int((head_dimension + block_n - 1u) / block_n),
                .tiles_m = int((tile_rows + block_m - 1u) / block_m),
                .batch_stride_a = visible,
                .batch_stride_b = 0,
                .batch_stride_d = head_dimension,
                .swizzle_log = 0,
                .gemm_k_iterations_aligned = int(visible / 16u),
                .batch_ndim = 1,
            };
            for (std::uint32_t plane = 0; plane < kv_heads; ++plane) {
                encode.pipeline(
                    large_tiles ? step.pipelines.attention_steel_values_large
                                : step.pipelines.attention_steel_values);
                encode.buffer(step.attention_partials,
                              std::uint64_t{plane} * heads_per_kv *
                                  visible * 2,
                              0);
                encode.buffer(state.second, plane * kv_plane_bytes, 1);
                encode.buffer(step.tokens, 0, 2);
                encode.buffer(step.attention_attended,
                              query_offset +
                                  std::uint64_t{plane} * heads_per_kv *
                                      head_dimension * 2,
                              3);
                std::memcpy(static_cast<std::uint8_t*>(
                                step.attention_steel_params.contents()) +
                                params_slot + sizeof(SteelAttnGemmParams),
                            &values_params, sizeof(values_params));
                encode.buffer(step.attention_steel_params,
                              params_slot + sizeof(SteelAttnGemmParams), 4);
                encode.buffer(step.tokens, 0, 5);
                encode.buffer(step.tokens, 0, 6);
                encode.buffer(step.tokens, 0, 7);
                encode.template dispatch<
                    PrefillProfileEventClass::AttentionStagedValues>(
                    {
                        .width = (head_dimension + block_n - 1u) / block_n,
                        .height = (tile_rows + block_m - 1u) / block_m,
                        .depth = heads_per_kv,
                    },
                    {.width = 32, .height = 2, .depth = 2},
                    layer, chunk, query_base, tile_rows);
            }
            encode.barrier();
            const std::uint32_t gate_elements =
                tile_rows * geometry.query_heads * head_dimension;
            encode.pipeline(step.pipelines.attention_gate_apply);
            encode.buffer(step.attention_attended, query_offset, 0);
            encode.buffer(step.attention_gate, query_offset, 1);
            encode.constant(gate_elements, 2);
            encode.template dispatch<
                PrefillProfileEventClass::AttentionCombine>(
                {
                    .width = (gate_elements + 255u) / 256u,
                    .height = 1,
                    .depth = 1,
                },
                {.width = 256, .height = 1, .depth = 1},
                layer, chunk, query_base, tile_rows);
        } else if (flash_v2) {
            const std::uint32_t query_groups =
                ceil_div(tile_rows,
                         kKernelLibraryPrefillFlashV2QueryTileRows);
            encode.pipeline(step.pipelines.attention_flash_v2);
            encode.buffer(step.attention_query, query_offset, 0);
            encode.buffer(state.first, 0, 1);
            encode.buffer(state.second, 0, 2);
            encode.buffer(step.attention_gate, query_offset, 3);
            encode.buffer(step.attention_attended, query_offset, 4);
            encode.constant(visible, 5);
            encode.constant(policy.context_capacity, 6);
            encode.constant(tile_rows, 7);
            encode.constant(tile_context, 8);
            encode.template dispatch<
                PrefillProfileEventClass::AttentionStreaming>(
                {
                    .width = geometry.query_heads,
                    .height = query_groups,
                    .depth = 1,
                },
                {
                    .width = kKernelLibraryPrefillFlashV2Threads,
                    .height = 1,
                    .depth = 1,
                },
                layer, chunk, query_base, tile_rows);
        } else if (streaming) {
            const std::uint32_t query_groups =
                ceil_div(
                    tile_rows,
                    kKernelLibraryPrefillStreamingAttentionQueryTileRows);
            encode.pipeline(
                step.pipelines.attention_streaming);
            encode.buffer(
                step.attention_query, query_offset, 0);
            encode.buffer(state.first, 0, 1);
            encode.buffer(state.second, 0, 2);
            encode.buffer(
                step.attention_gate, query_offset, 3);
            encode.buffer(
                step.attention_attended, query_offset, 4);
            encode.constant(visible, 5);
            encode.constant(
                policy.context_capacity, 6);
            encode.constant(tile_rows, 7);
            encode.constant(tile_context, 8);
            encode.template dispatch<
                PrefillProfileEventClass::AttentionStreaming>(
                {
                    .width = geometry.query_heads,
                    .height = query_groups,
                    .depth = 1,
                },
                {
                    .width =
                        kKernelLibraryPrefillStreamingAttentionThreads,
                    .height = 1,
                    .depth = 1,
                },
                layer, chunk, query_base, tile_rows);
        } else if (staged) {
            const std::uint32_t query_groups =
                ceil_div(
                    tile_rows,
                    kKernelLibraryPrefillStagedAttentionQueryTileRows);
            encode.pipeline(step.pipelines.attention_staged_scores);
            encode.buffer(step.attention_query, query_offset, 0);
            encode.buffer(state.first, 0, 1);
            encode.buffer(step.attention_partials, 0, 2);
            encode.constant(visible, 3);
            encode.constant(policy.context_capacity, 4);
            encode.constant(tile_rows, 5);
            encode.template dispatch<
                PrefillProfileEventClass::AttentionStagedScores>(
                {.width = ceil_div(
                     visible,
                     kKernelLibraryPrefillStagedAttentionKeyTileColumns),
                 .height = 1,
                 .depth = geometry.query_heads * query_groups},
                {.width =
                     kKernelLibraryPrefillStagedAttentionThreads,
                 .height = 1,
                 .depth = 1},
                layer, chunk, query_base, tile_rows);
            encode.barrier();
            encode.pipeline(step.pipelines.attention_staged_softmax);
            encode.buffer(step.attention_partials, 0, 0);
            encode.constant(visible, 1);
            encode.constant(tile_context, 2);
            encode.template dispatch<
                PrefillProfileEventClass::AttentionStagedSoftmax>(
                {.width = tile_rows,
                 .height = geometry.query_heads,
                 .depth = 1},
                {.width =
                     kKernelLibraryPrefillStagedAttentionSoftmaxThreads,
                 .height = 1,
                 .depth = 1},
                layer, chunk, query_base, tile_rows);
            encode.barrier();
            encode.pipeline(step.pipelines.attention_staged_values);
            encode.buffer(step.attention_partials, 0, 0);
            encode.buffer(state.second, 0, 1);
            encode.buffer(
                step.attention_attended, query_offset, 2);
            encode.constant(visible, 3);
            encode.constant(policy.context_capacity, 4);
            encode.constant(tile_rows, 5);
            encode.buffer(step.attention_gate, query_offset, 6);
            encode.template dispatch<
                PrefillProfileEventClass::AttentionStagedValues>(
                {.width = ceil_div(
                     geometry.attention_head_dimension,
                     kKernelLibraryPrefillStagedAttentionOutputTileColumns),
                 .height = 1,
                 .depth = geometry.query_heads * query_groups},
                {.width =
                     kKernelLibraryPrefillStagedAttentionThreads,
                 .height = 1,
                 .depth = 1},
                layer, chunk, query_base, tile_rows);
        } else {
            const std::uint32_t partitions =
                ceil_div(visible, policy.attention_partition);
            encode.pipeline(step.pipelines.attention_partial);
            encode.buffer(step.attention_query, query_offset, 0);
            encode.buffer(state.first, 0, 1);
            encode.buffer(state.second, 0, 2);
            encode.constant(tile_context, 3);
            encode.constant(policy.context_capacity, 4);
            encode.constant(policy.attention_partition, 5);
            encode.buffer(step.attention_partials, 0, 6);
            encode.constant(partitions, 7);
            encode.constant(tile_rows, 8);
            encode.template dispatch<
                PrefillProfileEventClass::AttentionPartial>(
                {.width = geometry.query_heads,
                 .height = partitions,
                 .depth = tile_rows},
                {.width = geometry.attention_head_dimension,
                 .height = 1,
                 .depth = 1},
                layer, chunk, query_base, tile_rows);
            encode.barrier();
            encode.pipeline(step.pipelines.attention_combine);
            encode.buffer(step.attention_partials, 0, 0);
            encode.buffer(step.attention_gate, query_offset, 1);
            encode.constant(partitions, 2);
            encode.buffer(
                step.attention_attended, query_offset, 3);
            encode.constant(tile_rows, 4);
            encode.template dispatch<
                PrefillProfileEventClass::AttentionCombine>(
                {.width = geometry.query_heads,
                 .height = tile_rows,
                 .depth = 1},
                {.width = geometry.attention_head_dimension,
                 .height = 1,
                 .depth = 1},
                layer, chunk, query_base, tile_rows);
        }
    }

    encode.barrier();
    if (step.policy.dense_qgemm ==
        QuantizedGemmPolicy::NativeDenseMma) {
        encode_native_dense_qgemm<
            PrefillProfileEventClass::AttentionOutputProjection>(
            encode, step,
            step.attention_attended, 0,
            bindings.attention.out, step.branch, 0, block_rows,
            geometry.hidden, geometry.attention_vector_values,
            geometry.attention_vector_values, geometry.hidden, layer,
            chunk);
    } else {
        encode.pipeline(step.pipelines.out_projection);
        encode.buffer(step.attention_attended, 0, 0);
        encode.quantized(bindings.attention.out, 1);
        encode.buffer(step.branch, 0, 4);
        encode.constant(block_rows, 5);
        encode.constant(geometry.attention_vector_values, 6);
        encode.template dispatch<
            PrefillProfileEventClass::AttentionOutputProjection>(
            {.width = ceil_div(
                 geometry.hidden,
                 kOutputProjectionThreads / kSimdgroupThreads),
             .height = 1,
             .depth = 1},
            {.width = kOutputProjectionThreads,
             .height = 1,
             .depth = 1},
            layer, chunk);
    }
}

template <typename CommandEncoder>
void encode_gated_delta(CommandEncoder& encode, PrefillStep& step,
                        const LayerBindings& bindings,
                        const DecodeLayerState& state, std::size_t layer,
                        const Chunk& chunk,
                        std::uint32_t recurrence_ordinal) {
    const PrefillGeometry& geometry = step.geometry;
    const std::uint32_t block_rows = chunk.rows;
    const bool phase = state.swapped ^ ((recurrence_ordinal & 1u) != 0u);
    const MetalBuffer& conv_in = phase ? state.first_out : state.first;
    const MetalBuffer& conv_out = phase ? state.first : state.first_out;
    const MetalBuffer& recurrent_in = phase ? state.second_out : state.second;
    const MetalBuffer& recurrent_out = phase ? state.second : state.second_out;

    if (step.policy.dense_qgemm ==
        QuantizedGemmPolicy::NativeDenseMma) {
        const std::uint32_t qkv_columns =
            geometry.gdn_qk_values + geometry.gdn_value_values;
        const std::uint32_t z_column_begin = qkv_columns;
        const std::uint32_t b_column_begin =
            z_column_begin + geometry.gdn_value_values;
        const std::uint32_t a_column_begin =
            b_column_begin + geometry.recurrent_heads;
        encode_native_dense_qgemm<
            PrefillProfileEventClass::GdnProjection>(
            encode, step,
            step.normalized, 0, bindings.gated_delta.qkv,
            step.gdn_projection, 0, block_rows, qkv_columns,
            geometry.hidden, geometry.hidden,
            geometry.gdn_projection_rows, layer, chunk);
        encode_native_dense_qgemm<
            PrefillProfileEventClass::GdnProjection>(
            encode, step,
            step.normalized, 0, bindings.gated_delta.z,
            step.gdn_projection,
            std::uint64_t{z_column_begin} * kBf16Bytes, block_rows,
            geometry.gdn_value_values, geometry.hidden,
            geometry.hidden, geometry.gdn_projection_rows, layer,
            chunk);
        encode_native_dense_qgemm<
            PrefillProfileEventClass::GdnProjection>(
            encode, step,
            step.normalized, 0, bindings.gated_delta.b,
            step.gdn_projection,
            std::uint64_t{b_column_begin} * kBf16Bytes, block_rows,
            geometry.recurrent_heads, geometry.hidden, geometry.hidden,
            geometry.gdn_projection_rows, layer, chunk);
        encode_native_dense_qgemm<
            PrefillProfileEventClass::GdnProjection>(
            encode, step,
            step.normalized, 0, bindings.gated_delta.a,
            step.gdn_projection,
            std::uint64_t{a_column_begin} * kBf16Bytes, block_rows,
            geometry.recurrent_heads, geometry.hidden, geometry.hidden,
            geometry.gdn_projection_rows, layer, chunk);
    } else {
        encode.pipeline(step.pipelines.gdn_project);
        encode.buffer(step.normalized, 0, 0);
        encode.quantized(bindings.gated_delta.qkv, 1);
        encode.quantized(bindings.gated_delta.z, 4);
        encode.quantized(bindings.gated_delta.b, 7);
        encode.quantized(bindings.gated_delta.a, 10);
        encode.buffer(step.gdn_projection, 0, 13);
        encode.constant(block_rows, 14);
        encode.template dispatch<
            PrefillProfileEventClass::GdnProjection>(
            {.width = ceil_div(
                 geometry.gdn_projection_rows,
                 kDenseProjectionThreads / kSimdgroupThreads),
             .height = 1,
             .depth = 1},
            {.width = kDenseProjectionThreads,
             .height = 1,
             .depth = 1},
            layer, chunk);
    }
    encode.barrier();

    const bool gdn_tape_mode =
        step.policy.gdn_recurrence ==
        PrefillGdnRecurrence::RegisterLoopTape;
    encode.pipeline(gdn_tape_mode ? step.pipelines.gdn_conv_tape
                                  : step.pipelines.gdn_conv);
    encode.buffer(step.gdn_projection, 0, 0);
    encode.buffer(conv_in, 0, 1);
    encode.weight(bindings.gated_delta.conv_weight, 2);
    encode.buffer(step.gdn_qk, 0, 3);
    encode.buffer(step.gdn_value, 0, 4);
    encode.buffer(step.gdn_gate, 0, 5);
    encode.buffer(conv_out, 0, 6);
    encode.constant(block_rows, 7);
    if (gdn_tape_mode) {
        encode.buffer(step.gdn_conv_tape_buffer, 0, 8);
        encode.constant(static_cast<std::uint32_t>(layer), 9);
    }
    encode.template dispatch<PrefillProfileEventClass::GdnConvolution>(
        {.width =
             (geometry.gdn_qk_values + 2u * geometry.gdn_value_values) /
             geometry.state_dimension,
         .height = block_rows,
         .depth = 1},
        {.width = geometry.state_dimension, .height = 1, .depth = 1}, layer,
        chunk);
    encode.barrier();

    if (step.policy.gdn_recurrence == PrefillGdnRecurrence::SerialSteps) {
        const std::uint64_t projection_row_bytes =
            std::uint64_t{geometry.gdn_projection_rows} * kBf16Bytes;
        const std::uint64_t qk_row_bytes = std::uint64_t{geometry.gdn_qk_values} * kBf16Bytes;
        const std::uint64_t value_row_bytes = std::uint64_t{geometry.gdn_value_values} * kBf16Bytes;
        for (std::uint32_t row = 0; row < block_rows; ++row) {
            encode.pipeline(step.pipelines.gdn_recurrence_step);
            encode.buffer(step.gdn_qk, std::uint64_t{row} * qk_row_bytes, 0);
            encode.buffer(step.gdn_value, std::uint64_t{row} * value_row_bytes, 1);
            encode.buffer(step.gdn_projection, std::uint64_t{row} * projection_row_bytes, 2);
            encode.weight(bindings.gated_delta.a_log, 3);
            encode.weight(bindings.gated_delta.dt_bias, 4);
            // The first row crosses from the live plane to its opposite,
            // exactly like decode. Later rows read and write that opposite
            // plane in place. Each recurrence thread loads only its own four
            // state elements before storing those same elements, and the
            // barrier completes every store before the next dispatch.
            encode.buffer(row == 0 ? recurrent_in : recurrent_out, 0, 5);
            encode.buffer(step.gdn_recurrence, std::uint64_t{row} * value_row_bytes, 6);
            encode.buffer(recurrent_out, 0, 7);
            encode
                .template dispatch<
                    PrefillProfileEventClass::GdnRecurrenceSerialStep>(
                    {.width = 1,
                     .height = geometry.state_dimension / 4u,
                     .depth = geometry.recurrent_heads},
                    {.width = kSimdgroupThreads, .height = 4, .depth = 1},
                    layer, chunk, row, 1);
            encode.barrier();
        }
    } else if (step.policy.geometry.gdn_gate_hoist) {
        encode.pipeline(step.pipelines.gdn_gates);
        encode.buffer(step.gdn_projection, 0, 0);
        encode.weight(bindings.gated_delta.a_log, 1);
        encode.weight(bindings.gated_delta.dt_bias, 2);
        encode.buffer(step.gdn_decay, 0, 3);
        encode.buffer(step.gdn_beta, 0, 4);
        encode.constant(block_rows, 5);
        encode.template dispatch<PrefillProfileEventClass::GdnGateHoist>(
            {.width = 1, .height = ceil_div(block_rows, kGateHoistRowsPerThreadgroup), .depth = 1},
            {.width = geometry.recurrent_heads,
             .height = kGateHoistRowsPerThreadgroup,
             .depth = 1},
            layer, chunk);
        encode.barrier();
        encode.pipeline(gdn_tape_mode
                            ? step.pipelines.gdn_recurrence_tape
                            : step.pipelines.gdn_recurrence_gates);
        encode.buffer(step.gdn_qk, 0, 0);
        encode.buffer(step.gdn_value, 0, 1);
        encode.buffer(step.gdn_decay, 0, 2);
        encode.weight(bindings.gated_delta.a_log, 3);
        encode.buffer(step.gdn_beta, 0, 4);
    } else {
        encode.pipeline(step.pipelines.gdn_recurrence_block);
        encode.buffer(step.gdn_qk, 0, 0);
        encode.buffer(step.gdn_value, 0, 1);
        encode.buffer(step.gdn_projection, 0, 2);
        encode.weight(bindings.gated_delta.a_log, 3);
        encode.weight(bindings.gated_delta.dt_bias, 4);
    }
    if (step.policy.gdn_recurrence == PrefillGdnRecurrence::RegisterLoop ||
        gdn_tape_mode) {
        encode.buffer(recurrent_in, 0, 5);
        encode.buffer(step.gdn_recurrence, 0, 6);
        encode.buffer(recurrent_out, 0, 7);
        encode.constant(block_rows, 8);
        if (gdn_tape_mode) {
            encode.buffer(step.gdn_tape, 0, 9);
            encode.constant(static_cast<std::uint32_t>(layer), 10);
        }
        encode
            .template dispatch<
                PrefillProfileEventClass::GdnRecurrenceRegisterLoop>(
                {.width = 1,
                 .height = geometry.state_dimension / 4u,
                 .depth = geometry.recurrent_heads},
                {.width = kSimdgroupThreads, .height = 4, .depth = 1}, layer,
                chunk);
        encode.barrier();
    }

    encode.pipeline(step.pipelines.gdn_gate_norm);
    encode.buffer(step.gdn_recurrence, 0, 0);
    encode.buffer(step.gdn_gate, 0, 1);
    encode.weight(bindings.gated_delta.norm_weight, 2);
    encode.buffer(step.gdn_gated, 0, 3);
    encode.template dispatch<PrefillProfileEventClass::GdnGateNormalization>(
        {.width = geometry.recurrent_heads,
         .height = block_rows,
         .depth = 1},
        {.width = geometry.state_dimension, .height = 1, .depth = 1}, layer,
        chunk);
    encode.barrier();

    if (step.policy.dense_qgemm ==
        QuantizedGemmPolicy::NativeDenseMma) {
        encode_native_dense_qgemm<
            PrefillProfileEventClass::GdnOutputProjection>(
            encode, step,
            step.gdn_gated, 0, bindings.gated_delta.out,
            step.branch, 0, block_rows, geometry.hidden,
            geometry.gdn_value_values, geometry.gdn_value_values,
            geometry.hidden, layer, chunk);
    } else {
        encode.pipeline(step.pipelines.out_projection);
        encode.buffer(step.gdn_gated, 0, 0);
        encode.quantized(bindings.gated_delta.out, 1);
        encode.buffer(step.branch, 0, 4);
        encode.constant(block_rows, 5);
        encode.constant(geometry.gdn_value_values, 6);
        encode.template dispatch<
            PrefillProfileEventClass::GdnOutputProjection>(
            {.width = ceil_div(
                 geometry.hidden,
                 kOutputProjectionThreads / kSimdgroupThreads),
             .height = 1,
             .depth = 1},
            {.width = kOutputProjectionThreads,
             .height = 1,
             .depth = 1},
            layer, chunk);
    }
}

template <typename CommandEncoder>
void encode_native_routed_task_builder(
    CommandEncoder& encode, PrefillStep& step,
    const MetalBuffer& tasks, const MetalBuffer& arguments,
    const MetalBuffer& status, std::uint64_t status_offset,
    std::uint32_t block_rows,
    std::uint32_t column_groups, std::size_t layer,
    const Chunk& chunk) {
    const PrefillGeometry& geometry = step.geometry;
    const std::uint32_t include_shared_expert =
        step.policy.native_routed_shared_expert ? 1U : 0U;
    const std::uint32_t task_expert_count =
        geometry.experts + include_shared_expert;
    const std::uint32_t routes_per_position =
        geometry.active_experts + include_shared_expert;
    const std::uint32_t list_stride =
        step.policy.geometry.maximum_block_rows;
    const std::uint64_t list_extent =
        std::uint64_t{task_expert_count} * list_stride;
    const std::uint32_t task_capacity =
        kKernelLibraryNativeRoutedQgemmR1TaskCapacity;
    const std::uint32_t packed_slot_bits =
        kKernelLibraryPrefillPackedSlotBits;
    encode.pipeline(step.pipelines.native_routed_task_builder);
    encode.buffer(step.expert_counts, 0, 0);
    encode.buffer(step.expert_lists, 0, 1);
    encode.buffer(tasks, 0, 2);
    encode.buffer(arguments, 0, 3);
    encode.buffer(status, status_offset, 4);
    encode.constant(block_rows, 5);
    encode.constant(task_expert_count, 6);
    encode.constant(routes_per_position, 7);
    encode.constant(list_stride, 8);
    encode.constant(list_stride, 9);
    encode.constant(list_extent, 10);
    encode.constant(task_capacity, 11);
    encode.constant(column_groups, 12);
    encode.constant(packed_slot_bits, 13);
    encode.constant(include_shared_expert, 14);
    encode.template dispatch<
        PrefillProfileEventClass::MoeRoutedTaskBuild>(
        {.width = 1, .height = 1, .depth = 1},
        {.width = 1, .height = 1, .depth = 1}, layer, chunk);
}

template <typename CommandEncoder>
void encode_native_routed_upgate(
    CommandEncoder& encode, PrefillStep& step,
    const LayerBindings& bindings, std::uint32_t block_rows,
    std::size_t layer, const Chunk& chunk) {
    const PrefillGeometry& geometry = step.geometry;
    const std::uint32_t include_shared_expert =
        step.policy.native_routed_shared_expert ? 1U : 0U;
    const std::uint32_t task_expert_count =
        geometry.experts + include_shared_expert;
    const std::uint32_t list_stride =
        step.policy.geometry.maximum_block_rows;
    const std::uint64_t list_extent =
        std::uint64_t{task_expert_count} * list_stride;
    encode.pipeline(
        step.policy.native_routed_steel
            ? step.pipelines.native_routed_steel_upgate
            : step.pipelines.native_routed_upgate);
    encode.buffer(step.normalized, 0, 0);
    encode.buffer(step.expert_lists, 0, 1);
    encode.buffer(step.native_routed_up_tasks, 0, 2);
    encode.quantized(bindings.expert_gate, 3);
    encode.quantized(bindings.expert_up, 6);
    encode.buffer(step.expert_hidden, 0, 9);
    encode.buffer(step.native_routed_up_arguments, 0, 10);
    encode.constant(list_stride, 11);
    encode.constant(list_stride, 12);
    encode.constant(list_extent, 13);
    encode.constant(block_rows, 14);
    encode.quantized(bindings.shared_gate, 15);
    encode.quantized(bindings.shared_up, 18);
    encode.constant(include_shared_expert, 21);
    if (step.policy.native_routed_shared_expert) {
        encode.template indirect<
            PrefillProfileEventClass::MoeNativeRoutedSharedUpGate>(
            step.native_routed_up_arguments, 0,
            {.width = kKernelLibraryNativeRoutedQgemmR1Threads,
             .height = 1,
             .depth = 1},
            layer, chunk);
    } else {
        encode.template indirect<
            PrefillProfileEventClass::MoeNativeRoutedUpGate>(
            step.native_routed_up_arguments, 0,
            {.width = kKernelLibraryNativeRoutedQgemmR1Threads,
             .height = 1,
             .depth = 1},
            layer, chunk);
    }
}

template <typename CommandEncoder>
void encode_native_routed_down(
    CommandEncoder& encode, PrefillStep& step,
    const LayerBindings& bindings, std::uint32_t block_rows,
    std::size_t layer, const Chunk& chunk) {
    const PrefillGeometry& geometry = step.geometry;
    const std::uint32_t include_shared_expert =
        step.policy.native_routed_shared_expert ? 1U : 0U;
    const std::uint32_t task_expert_count =
        geometry.experts + include_shared_expert;
    const std::uint32_t list_stride =
        step.policy.geometry.maximum_block_rows;
    const std::uint64_t list_extent =
        std::uint64_t{task_expert_count} * list_stride;
    encode.pipeline(
        step.policy.native_routed_steel
            ? step.pipelines.native_routed_steel_down
            : step.pipelines.native_routed_down);
    encode.buffer(step.expert_hidden, 0, 0);
    encode.buffer(step.expert_lists, 0, 1);
    encode.buffer(
        step.policy.command_graph ? step.native_routed_up_tasks
                                  : step.native_routed_down_tasks,
        0, 2);
    encode.quantized(bindings.expert_down, 3);
    encode.buffer(step.expert_partials, 0, 6);
    encode.buffer(
        step.policy.command_graph
            ? step.native_routed_up_arguments
            : step.native_routed_down_arguments,
        0, 7);
    encode.constant(list_stride, 8);
    encode.constant(list_stride, 9);
    encode.constant(list_extent, 10);
    encode.constant(block_rows, 11);
    encode.quantized(bindings.shared_down, 12);
    encode.constant(include_shared_expert, 15);
    if (step.policy.native_routed_shared_expert) {
        encode.template indirect<
            PrefillProfileEventClass::MoeNativeRoutedSharedDown>(
            step.native_routed_down_arguments, 0,
            {.width = kKernelLibraryNativeRoutedQgemmR1Threads,
             .height = 1,
             .depth = 1},
            layer, chunk);
    } else {
        encode.template indirect<
            PrefillProfileEventClass::MoeNativeRoutedDown>(
            step.native_routed_down_arguments, 0,
            {.width = kKernelLibraryNativeRoutedQgemmR1Threads,
             .height = 1,
             .depth = 1},
            layer, chunk);
    }
}

template <typename CommandEncoder>
void encode_moe(CommandEncoder& encode, PrefillStep& step,
                const LayerBindings& bindings, std::size_t layer,
                const Chunk& chunk, const MetalBuffer& input,
                std::uint64_t input_offset, const MetalBuffer& output,
                std::uint64_t output_offset,
                std::uint32_t status_slot) {
    const PrefillGeometry& geometry = step.geometry;
    const PrefillPolicy& policy = step.policy.geometry;
    const std::uint32_t block_rows = chunk.rows;
    encode.barrier();
    encode.pipeline(step.pipelines.residual);
    encode.buffer(input, input_offset, 0);
    encode.buffer(step.branch, 0, 1);
    encode.buffer(step.block_hidden, 0, 2);
    encode.template dispatch<PrefillProfileEventClass::MoeResidualInput>(
        {.width = geometry.hidden / kElementwiseThreads, .height = block_rows, .depth = 1},
        {.width = kElementwiseThreads, .height = 1, .depth = 1}, layer,
        chunk);
    encode.barrier();

    encode.pipeline(step.pipelines.rms);
    encode.buffer(step.block_hidden, 0, 0);
    encode.weight(bindings.post_norm, 1);
    encode.buffer(step.normalized, 0, 2);
    encode
        .template dispatch<
            PrefillProfileEventClass::MoePostNormalization>(
            {.width = 1, .height = block_rows, .depth = 1},
            {.width = geometry.hidden / kRmsValuesPerThread,
             .height = 1,
             .depth = 1},
            layer, chunk);
    encode.barrier();

    encode.pipeline(step.pipelines.router);
    encode.buffer(step.normalized, 0, 0);
    encode.quantized(bindings.router, 1);
    encode.quantized(bindings.shared_router, 4);
    encode.buffer(step.router_logits, 0, 7);
    encode.template dispatch<PrefillProfileEventClass::MoeRouter>(
        {.width = ceil_div((geometry.experts + 1u) * kSimdgroupThreads, kDenseProjectionThreads),
         .height = block_rows,
         .depth = 1},
        {.width = kDenseProjectionThreads, .height = 1, .depth = 1}, layer,
        chunk);
    encode.barrier();

    encode.pipeline(step.policy.router_selector == PrefillRouterSelector::Parallel
                        ? step.pipelines.router_select_parallel
                        : step.pipelines.router_select_serial);
    encode.buffer(step.router_logits, 0, 0);
    encode.buffer(step.expert_ids, 0, 1);
    encode.buffer(step.expert_coefficients, 0, 2);
    encode.buffer(step.shared_coefficients, 0, 3);
    encode.constant(geometry.active_experts, 4);
    if constexpr (CommandEncoder::profiled) {
        if (step.policy.router_selector == PrefillRouterSelector::Parallel) {
            encode
                .template dispatch<
                    PrefillProfileEventClass::MoeRouterSelectParallel>(
                {.width = block_rows, .height = 1, .depth = 1},
                {.width = geometry.experts, .height = 1, .depth = 1}, layer,
                chunk);
        } else {
            encode
                .template dispatch<
                    PrefillProfileEventClass::MoeRouterSelectSerial>(
                    {.width = block_rows, .height = 1, .depth = 1},
                    {.width = geometry.experts, .height = 1, .depth = 1}, layer,
                    chunk);
        }
    } else {
        encode.unprofiled_dispatch(
            {.width = block_rows, .height = 1, .depth = 1},
            {.width = geometry.experts, .height = 1, .depth = 1});
    }
    encode.barrier();

    if (step.policy.command_graph) {
        const std::uint32_t include_shared_expert = 1u;
        const std::uint64_t list_extent =
            std::uint64_t{geometry.experts + include_shared_expert} *
            policy.maximum_block_rows;
        const std::uint32_t planned_task_capacity =
            maximum_routed_task_count(step, block_rows);
        const std::uint32_t packed_slot_bits =
            kKernelLibraryPrefillPackedSlotBits;
        encode.pipeline(step.pipelines.expert_union_fused_tasks);
        encode.buffer(step.expert_ids, 0, 0);
        encode.constant(block_rows, 1);
        encode.buffer(step.expert_counts, 0, 2);
        encode.buffer(step.expert_lists, 0, 3);
        encode.buffer(step.native_routed_up_tasks, 0, 4);
        encode.buffer(step.native_routed_up_arguments, 0, 5);
        encode.buffer(
            step.native_routed_up_status,
            std::uint64_t{status_slot} * kNativeRoutedStatusBytes,
            6);
        encode.constant(policy.maximum_block_rows, 7);
        encode.constant(include_shared_expert, 8);
        encode.constant(list_extent, 9);
        encode.constant(planned_task_capacity, 10);
        encode.constant(packed_slot_bits, 11);
    } else {
        encode.pipeline(step.pipelines.expert_union);
        encode.buffer(step.expert_ids, 0, 0);
        encode.constant(block_rows, 1);
        encode.buffer(step.expert_counts, 0, 2);
        encode.buffer(step.expert_lists, 0, 3);
        encode.buffer(step.active_experts, 0, 4);
        encode.buffer(step.expert_arguments, 0, 5);
        encode.buffer(
            step.expert_arguments,
            3u * sizeof(std::uint32_t), 6);
        encode.constant(policy.maximum_block_rows, 7);
        encode.constant(policy.exact_rows_per_threadgroup, 8);
    }
    encode.template dispatch<PrefillProfileEventClass::MoeExpertUnion>(
        {.width = 1, .height = 1, .depth = 1},
        {.width = geometry.experts, .height = 1, .depth = 1}, layer, chunk);
    encode.barrier();

    if (step.policy.routed_qgemm ==
        QuantizedGemmPolicy::NativeRaggedMma) {
        const std::uint32_t up_column_groups =
            ceil_div(
                geometry.expert_dimension,
                kKernelLibraryNativeRoutedQgemmR1TileColumns);
        if (!step.policy.command_graph) {
            encode_native_routed_task_builder(
                encode, step, step.native_routed_up_tasks,
                step.native_routed_up_arguments,
                step.native_routed_up_status,
                std::uint64_t{status_slot} *
                    kNativeRoutedStatusBytes,
                block_rows, up_column_groups, layer, chunk);
            encode.barrier();
        }
        encode_native_routed_upgate(
            encode, step, bindings, block_rows, layer, chunk);

        if (!step.policy.native_routed_shared_expert) {
            encode.pipeline(step.pipelines.expert_upgate);
            encode.buffer(step.normalized, 0, 0);
            encode.buffer(step.expert_counts, 0, 1);
            encode.buffer(step.expert_lists, 0, 2);
            encode.quantized(bindings.expert_gate, 3);
            encode.quantized(bindings.expert_up, 6);
            encode.buffer(step.expert_hidden, 0, 9);
            encode.buffer(step.shared_expert, 0, 10);
            encode.quantized(bindings.shared_gate, 11);
            encode.quantized(bindings.shared_up, 14);
            encode.constant(policy.maximum_block_rows, 17);
            encode.template indirect<
                PrefillProfileEventClass::MoeSharedExpertUpGate>(
                step.shared_expert_arguments, 0,
                {.width =
                     std::uint64_t{
                         policy.exact_rows_per_threadgroup} *
                     kSimdgroupThreads,
                 .height = 1,
                 .depth = 1},
                layer, chunk);
        }
        encode.barrier();

        const std::uint32_t down_column_groups =
            ceil_div(
                geometry.hidden,
                kKernelLibraryNativeRoutedQgemmR1TileColumns);
        if (!step.policy.command_graph) {
            encode_native_routed_task_builder(
                encode, step, step.native_routed_down_tasks,
                step.native_routed_down_arguments,
                step.native_routed_down_status,
                std::uint64_t{status_slot} *
                    kNativeRoutedStatusBytes,
                block_rows, down_column_groups, layer, chunk);
            encode.barrier();
        }
        encode_native_routed_down(
            encode, step, bindings, block_rows, layer, chunk);

        if (!step.policy.native_routed_shared_expert) {
            encode.pipeline(step.pipelines.expert_down);
            encode.buffer(step.expert_hidden, 0, 0);
            encode.buffer(step.expert_counts, 0, 1);
            encode.buffer(step.expert_lists, 0, 2);
            encode.quantized(bindings.expert_down, 3);
            encode.buffer(step.expert_partials, 0, 6);
            encode.buffer(step.shared_expert, 0, 7);
            encode.quantized(bindings.shared_down, 8);
            encode.constant(policy.maximum_block_rows, 11);
            encode.template indirect<
                PrefillProfileEventClass::MoeSharedExpertDown>(
                step.shared_expert_arguments,
                3u * sizeof(std::uint32_t),
                {.width =
                     std::uint64_t{
                         policy.exact_rows_per_threadgroup} *
                     kSimdgroupThreads,
                 .height = 1,
                 .depth = 1},
                layer, chunk);
        }
        encode.barrier();
    } else {
        encode.pipeline(step.pipelines.expert_upgate);
        encode.buffer(step.normalized, 0, 0);
        encode.buffer(step.expert_counts, 0, 1);
        encode.buffer(step.expert_lists, 0, 2);
        encode.quantized(bindings.expert_gate, 3);
        encode.quantized(bindings.expert_up, 6);
        encode.buffer(step.expert_hidden, 0, 9);
        encode.buffer(step.active_experts, 0, 10);
        encode.quantized(bindings.shared_gate, 11);
        encode.quantized(bindings.shared_up, 14);
        encode.constant(policy.maximum_block_rows, 17);
        encode.template indirect<
            PrefillProfileEventClass::MoeExpertUpGate>(
            step.expert_arguments, 0,
            {.width =
                 std::uint64_t{
                     policy.exact_rows_per_threadgroup} *
                 kSimdgroupThreads,
             .height = 1,
             .depth = 1},
            layer, chunk);
        encode.barrier();

        encode.pipeline(step.pipelines.expert_down);
        encode.buffer(step.expert_hidden, 0, 0);
        encode.buffer(step.expert_counts, 0, 1);
        encode.buffer(step.expert_lists, 0, 2);
        encode.quantized(bindings.expert_down, 3);
        encode.buffer(step.expert_partials, 0, 6);
        encode.buffer(step.active_experts, 0, 7);
        encode.quantized(bindings.shared_down, 8);
        encode.constant(policy.maximum_block_rows, 11);
        encode.template indirect<
            PrefillProfileEventClass::MoeExpertDown>(
            step.expert_arguments,
            3u * sizeof(std::uint32_t),
            {.width =
                 std::uint64_t{
                     policy.exact_rows_per_threadgroup} *
                 kSimdgroupThreads,
             .height = 1,
             .depth = 1},
            layer, chunk);
        encode.barrier();
    }

    encode.pipeline(step.pipelines.expert_combine);
    encode.buffer(step.expert_partials, 0, 0);
    encode.buffer(step.expert_coefficients, 0, 1);
    encode.buffer(step.shared_coefficients, 0, 2);
    encode.buffer(step.moe_output, 0, 3);
    encode.template dispatch<PrefillProfileEventClass::MoeExpertCombine>(
        {.width = geometry.hidden / kElementwiseThreads, .height = block_rows, .depth = 1},
        {.width = kElementwiseThreads, .height = 1, .depth = 1}, layer,
        chunk);
    encode.barrier();

    encode.pipeline(step.pipelines.residual);
    encode.buffer(step.block_hidden, 0, 0);
    encode.buffer(step.moe_output, 0, 1);
    encode.buffer(output, output_offset, 2);
    encode.template dispatch<PrefillProfileEventClass::MoeResidualOutput>(
        {.width = geometry.hidden / kElementwiseThreads, .height = block_rows, .depth = 1},
        {.width = kElementwiseThreads, .height = 1, .depth = 1}, layer,
        chunk);
    // A41a: conditioning capture rides the graph — one copy dispatch after
    // each captured layer's residual.
    if (step.policy.conditioning_capture) {
        for (std::uint32_t slot = 0; slot < 8u; ++slot) {
            if (step.policy.capture_layers[slot] !=
                static_cast<std::uint32_t>(layer)) {
                continue;
            }
            encode.barrier();
            encode.pipeline(step.pipelines.capture_rows);
            encode.buffer(output, output_offset, 0);
            encode.buffer(step.capture_buffer, 0, 1);
            encode.constant(block_rows, 2);
            encode.constant(slot, 3);
            encode.constant(step.policy.conditioning_capture_rows, 4);
            encode.template dispatch<
                PrefillProfileEventClass::MoeResidualOutput>(
                {.width = geometry.hidden / kElementwiseThreads,
                 .height = block_rows,
                 .depth = 1},
                {.width = kElementwiseThreads, .height = 1, .depth = 1},
                layer, chunk);
        }
    }
}

template <typename CommandEncoder>
void encode_layer(CommandEncoder& encode, PrefillStep& step,
                  DecodeStep& decode, DecodeStateSlot& state,
                  std::size_t layer, const Chunk& chunk,
                  std::uint32_t recurrence_ordinal,
                  std::uint32_t context_base, const MetalBuffer& input,
                  std::uint64_t input_offset, const MetalBuffer& output,
                  std::uint64_t output_offset,
                  std::uint32_t status_slot) {
    const PrefillGeometry& geometry = step.geometry;
    const LayerBindings& bindings = decode.bindings.layers[layer];
    encode.barrier();
    encode.pipeline(step.pipelines.rms);
    encode.buffer(input, input_offset, 0);
    encode.weight(bindings.input_norm, 1);
    encode.buffer(step.normalized, 0, 2);
    encode
        .template dispatch<
            PrefillProfileEventClass::LayerInputNormalization>(
            {.width = 1, .height = chunk.rows, .depth = 1},
            {.width = geometry.hidden / kRmsValuesPerThread,
             .height = 1,
             .depth = 1},
            layer, chunk);
    encode.barrier();

    if (bindings.kind == model::qwen36::LayerKind::GatedDelta) {
        encode_gated_delta(encode, step, bindings, state.layers[layer], layer,
                           chunk, recurrence_ordinal);
    } else {
        encode_attention(encode, step, bindings, state.layers[layer], layer,
                         chunk, context_base + chunk.offset);
    }
    encode_moe(encode, step, bindings, layer, chunk, input, input_offset,
               output, output_offset, status_slot);
}

bool scratch_is_complete(const PrefillStep& step) {
    const PrefillGeometry& geometry = step.geometry;
    const std::uint64_t scratch_lanes =
        step.policy.command_graph
            ? step.policy.command_graph_chunk_count
            : 1u;
    const auto enough_scratch =
        [scratch_lanes](const MetalBuffer& buffer,
                        std::uint64_t bytes) {
            std::uint64_t required = 0;
            return multiply(bytes, scratch_lanes, required) &&
                   enough(buffer, required);
        };
    const bool native_routed =
        step.policy.routed_qgemm ==
        QuantizedGemmPolicy::NativeRaggedMma;
    const std::uint64_t task_bytes =
        native_routed ? kNativeRoutedTaskBytes : 0;
    const std::uint64_t argument_bytes =
        native_routed ? kNativeRoutedArgumentBytes : 0;
    const std::uint64_t status_bytes =
        native_routed
            ? std::uint64_t{
                  std::max(
                      std::max(
                          step.policy.maximum_units_per_submission,
                          step.policy.maximum_inflight_units),
                      step.command_graph_task_status_count)} *
                  kNativeRoutedStatusBytes
            : 0;
    return step.native_routed_workspace_bytes ==
               (native_routed
                    ? scratch_lanes *
                              (kNativeRoutedFixedWorkspaceBytes -
                               kNativeRoutedSharedExpertBytes -
                               kNativeRoutedSharedArgumentBytes) +
                          kNativeRoutedSharedExpertBytes +
                          kNativeRoutedSharedArgumentBytes +
                          2u * status_bytes
                    : 0) &&
           enough(step.tokens, geometry.token_bytes) &&
           enough(step.hidden_slab, geometry.hidden_slab_bytes) &&
           enough_scratch(step.block_hidden, geometry.block_hidden_bytes) &&
           enough_scratch(step.normalized, geometry.block_hidden_bytes) &&
           enough_scratch(step.branch, geometry.block_hidden_bytes) &&
           enough_scratch(step.moe_output, geometry.block_hidden_bytes) &&
           enough_scratch(step.gdn_projection, geometry.gdn_projection_bytes) &&
           enough_scratch(step.gdn_qk, geometry.gdn_qk_bytes) &&
           enough_scratch(step.gdn_value, geometry.gdn_value_bytes) &&
           enough_scratch(step.gdn_gate, geometry.gdn_value_bytes) &&
           enough_scratch(step.gdn_recurrence, geometry.gdn_value_bytes) &&
           enough_scratch(step.gdn_gated, geometry.gdn_value_bytes) &&
           enough_scratch(step.gdn_decay, geometry.gdn_parameter_bytes) &&
           enough_scratch(step.gdn_beta, geometry.gdn_parameter_bytes) &&
           enough_scratch(step.attention_projection, geometry.attention_projection_bytes) &&
           enough_scratch(step.attention_query, geometry.attention_vector_bytes) &&
           enough_scratch(step.attention_gate, geometry.attention_vector_bytes) &&
           enough_scratch(step.attention_attended, geometry.attention_vector_bytes) &&
           enough_scratch(step.attention_partials, geometry.attention_partial_bytes) &&
           enough_scratch(step.router_logits, geometry.moe_logits_bytes) &&
           enough_scratch(step.expert_ids, geometry.moe_id_bytes) &&
           enough_scratch(step.expert_coefficients, geometry.moe_coefficient_bytes) &&
           enough_scratch(step.shared_coefficients, geometry.moe_shared_coefficient_bytes) &&
           enough_scratch(step.expert_counts, geometry.moe_count_bytes) &&
           enough_scratch(step.expert_lists, geometry.moe_list_bytes) &&
           enough_scratch(step.active_experts, geometry.moe_active_bytes) &&
           enough_scratch(step.expert_arguments, geometry.moe_indirect_argument_bytes) &&
           enough_scratch(step.expert_hidden, geometry.moe_hidden_bytes) &&
           enough_scratch(step.expert_partials, geometry.moe_partial_bytes) &&
           enough_scratch(step.native_routed_up_tasks, task_bytes) &&
           enough_scratch(step.native_routed_up_arguments, argument_bytes) &&
           enough(step.native_routed_up_status, status_bytes) &&
           enough_scratch(step.native_routed_down_tasks, task_bytes) &&
           enough_scratch(step.native_routed_down_arguments, argument_bytes) &&
           enough(step.native_routed_down_status, status_bytes) &&
           enough(step.shared_expert,
                  native_routed ? kNativeRoutedSharedExpertBytes : 0) &&
           enough(step.shared_expert_arguments,
                  native_routed ? kNativeRoutedSharedArgumentBytes : 0);
}

PrefillEncodeError owner_error(const PrefillStep& prefill, const DecodeStep& decode,
                               const DecodeStateSlot& state) {
    if (!decode.image || !*decode.image || decode.tensor_offsets.empty() ||
        decode.bindings.layers.size() != decode.schedule.size() ||
        !decode_state_slot_ready(decode, state) ||
        decode.schedule.size() !=
            std::size_t{prefill.geometry.gated_delta_layers + prefill.geometry.attention_layers} ||
        decode.capacity != prefill.policy.geometry.context_capacity) {
        return PrefillEncodeError::BindingMismatch;
    }
    if (!scratch_is_complete(prefill)) {
        return PrefillEncodeError::ScratchTooSmall;
    }
    return PrefillEncodeError::None;
}

bool valid_active_progress(const PrefillStep& prefill, const DecodeStep& decode,
                           const DecodeStateSlot& state) {
    const PrefillProgress& progress = prefill.progress;
    const bool ready =
        progress.state == PrefillProgressState::Ready &&
        progress.pending_unit_count == 0;
    const bool unit_pending =
        progress.state == PrefillProgressState::UnitPending &&
        progress.pending_unit_count == 1;
    const bool batch_pending =
        progress.state == PrefillProgressState::BatchPending &&
        progress.pending_unit_count != 0 &&
        progress.pending_unit_count <=
        prefill.policy.maximum_units_per_submission &&
        prefill.policy.geometry.schedule == PrefillSchedule::LayerMajor;
    const bool inflight_encoding =
        progress.state == PrefillProgressState::InflightEncoding &&
        progress.pending_unit_count != 0 &&
        progress.pending_unit_count <
            prefill.policy.maximum_inflight_units &&
        prefill.policy.geometry.schedule == PrefillSchedule::LayerMajor;
    const bool inflight_pending =
        progress.state == PrefillProgressState::InflightPending &&
        progress.pending_unit_count != 0 &&
        progress.pending_unit_count <=
            prefill.policy.maximum_inflight_units &&
        prefill.policy.geometry.schedule == PrefillSchedule::LayerMajor;
    if ((!ready && !unit_pending && !batch_pending &&
         !inflight_encoding && !inflight_pending) ||
        progress.owner != &decode || progress.state_owner != &state ||
        !decode_state_slot_available(decode, state) || progress.row_count == 0 ||
        progress.chunk_count == 0 || progress.live_context != progress.context_base ||
        progress.context_base > progress.next_context ||
        progress.next_context - progress.context_base != progress.row_count ||
        progress.next_context > prefill.policy.geometry.context_capacity ||
        progress.chunk_count !=
            chunk_count(prefill.policy, progress.context_base,
                        progress.row_count)) {
        return false;
    }
    if (prefill.policy.geometry.schedule == PrefillSchedule::ChunkMajor) {
        return !batch_pending && !inflight_encoding &&
               !inflight_pending && progress.current_layer == 0 &&
               progress.current_chunk < progress.chunk_count;
    }
    if (progress.current_layer >= decode.schedule.size() ||
        progress.current_chunk >= progress.chunk_count) {
        return false;
    }
    const std::uint64_t current =
        std::uint64_t{progress.current_layer} * progress.chunk_count +
        progress.current_chunk;
    const std::uint64_t total =
        std::uint64_t{decode.schedule.size()} * progress.chunk_count;
    return progress.pending_unit_count <= total - current;
}

PrefillProgressResult progress_failure(const PrefillStep& prefill, PrefillProgressError error) {
    return {
        .error = error,
        .state = prefill.progress.state,
        .next_context = prefill.progress.context_base,
        .chunk_count = prefill.progress.chunk_count,
        .layer_index = prefill.progress.current_layer,
        .chunk_ordinal = prefill.progress.current_chunk,
        .unit_count = prefill.progress.pending_unit_count,
    };
}

QuantizedGemmDeviceTaskStatus device_task_status(
    const MetalBuffer& buffer, std::uint32_t slot) {
    std::uint32_t value = 0;
    const auto* contents =
        static_cast<const std::byte*>(buffer.contents());
    std::memcpy(
        &value,
        contents + std::uint64_t{slot} * kNativeRoutedStatusBytes,
        sizeof(value));
    return static_cast<QuantizedGemmDeviceTaskStatus>(value);
}

template <typename Visitor>
void visit_prefill_pipelines(
    const PrefillPipelines& pipelines, Visitor&& visit) {
    visit(pipelines.embed);
    visit(pipelines.rms);
    visit(pipelines.residual);
    visit(pipelines.gdn_project);
    visit(pipelines.gdn_conv);
    visit(pipelines.gdn_conv_tape);
    visit(pipelines.gdn_recurrence_tape);
    visit(pipelines.capture_rows);
    visit(pipelines.gdn_gates);
    visit(pipelines.gdn_recurrence_step);
    visit(pipelines.gdn_recurrence_block);
    visit(pipelines.gdn_recurrence_gates);
    visit(pipelines.gdn_gate_norm);
    visit(pipelines.attn_project);
    visit(pipelines.attn_qk_rope);
    visit(pipelines.attention_partial);
    visit(pipelines.attention_combine);
    visit(pipelines.attention_staged_scores);
    visit(pipelines.attention_staged_softmax);
    visit(pipelines.attention_staged_values);
    visit(pipelines.attention_streaming);
    visit(pipelines.attention_flash_v2);
    visit(pipelines.attention_steel_scores);
    visit(pipelines.attention_steel_values);
    visit(pipelines.attention_steel_scores_large);
    visit(pipelines.attention_steel_values_large);
    visit(pipelines.attention_softmax_bf16);
    visit(pipelines.attention_gate_apply);
    visit(pipelines.out_projection);
    visit(pipelines.router);
    visit(pipelines.router_select_serial);
    visit(pipelines.router_select_parallel);
    visit(pipelines.expert_union);
    visit(pipelines.expert_union_fused_tasks);
    visit(pipelines.expert_upgate);
    visit(pipelines.expert_down);
    visit(pipelines.expert_combine);
    visit(pipelines.native_dense_qgemm);
    visit(pipelines.native_dense_steel);
    visit(
        pipelines
            .native_dense_steel_gdn_bm64_wm2_wn2);
    visit(pipelines.native_routed_task_builder);
    visit(pipelines.native_routed_upgate);
    visit(pipelines.native_routed_down);
    visit(pipelines.native_routed_steel_upgate);
    visit(pipelines.native_routed_steel_down);
}

template <typename Visitor>
void visit_prefill_graph_buffers(
    const PrefillStep& prefill, const DecodeStep& decode,
    const DecodeStateSlot& state, Visitor&& visit) {
    visit(*decode.image, MetalResourceUsage::Read);
    visit(prefill.tokens, MetalResourceUsage::Read);
    visit(prefill.attention_steel_params, MetalResourceUsage::Read);
    visit(prefill.gdn_tape, MetalResourceUsage::ReadWrite);
    visit(prefill.gdn_conv_tape_buffer, MetalResourceUsage::ReadWrite);
    visit(prefill.capture_buffer, MetalResourceUsage::ReadWrite);
    visit(prefill.hidden_slab, MetalResourceUsage::ReadWrite);
    visit(prefill.block_hidden, MetalResourceUsage::ReadWrite);
    visit(prefill.normalized, MetalResourceUsage::ReadWrite);
    visit(prefill.branch, MetalResourceUsage::ReadWrite);
    visit(prefill.moe_output, MetalResourceUsage::ReadWrite);
    visit(prefill.gdn_projection, MetalResourceUsage::ReadWrite);
    visit(prefill.gdn_qk, MetalResourceUsage::ReadWrite);
    visit(prefill.gdn_value, MetalResourceUsage::ReadWrite);
    visit(prefill.gdn_gate, MetalResourceUsage::ReadWrite);
    visit(prefill.gdn_recurrence, MetalResourceUsage::ReadWrite);
    visit(prefill.gdn_gated, MetalResourceUsage::ReadWrite);
    if (prefill.gdn_decay) {
        visit(prefill.gdn_decay, MetalResourceUsage::ReadWrite);
        visit(prefill.gdn_beta, MetalResourceUsage::ReadWrite);
    }
    visit(prefill.attention_projection, MetalResourceUsage::ReadWrite);
    visit(prefill.attention_query, MetalResourceUsage::ReadWrite);
    visit(prefill.attention_gate, MetalResourceUsage::ReadWrite);
    visit(prefill.attention_attended, MetalResourceUsage::ReadWrite);
    visit(prefill.attention_partials, MetalResourceUsage::ReadWrite);
    visit(prefill.router_logits, MetalResourceUsage::ReadWrite);
    visit(prefill.expert_ids, MetalResourceUsage::ReadWrite);
    visit(prefill.expert_coefficients, MetalResourceUsage::ReadWrite);
    visit(prefill.shared_coefficients, MetalResourceUsage::ReadWrite);
    visit(prefill.expert_counts, MetalResourceUsage::ReadWrite);
    visit(prefill.expert_lists, MetalResourceUsage::ReadWrite);
    visit(prefill.active_experts, MetalResourceUsage::ReadWrite);
    visit(prefill.expert_arguments, MetalResourceUsage::ReadWrite);
    visit(prefill.expert_hidden, MetalResourceUsage::ReadWrite);
    visit(prefill.expert_partials, MetalResourceUsage::ReadWrite);
    visit(prefill.native_routed_up_tasks, MetalResourceUsage::ReadWrite);
    visit(
        prefill.native_routed_up_arguments,
        MetalResourceUsage::ReadWrite);
    visit(prefill.native_routed_up_status, MetalResourceUsage::ReadWrite);
    visit(
        prefill.native_routed_down_tasks,
        MetalResourceUsage::ReadWrite);
    visit(
        prefill.native_routed_down_arguments,
        MetalResourceUsage::ReadWrite);
    visit(
        prefill.native_routed_down_status,
        MetalResourceUsage::ReadWrite);
    visit(prefill.shared_expert, MetalResourceUsage::Read);
    visit(prefill.shared_expert_arguments, MetalResourceUsage::Read);
    for (std::size_t layer_index = 0;
         layer_index < state.layers.size(); ++layer_index) {
        const DecodeLayerState& layer = state.layers[layer_index];
        visit(layer.first, MetalResourceUsage::ReadWrite);
        visit(layer.second, MetalResourceUsage::ReadWrite);
        if (decode.schedule[layer_index] ==
            model::qwen36::LayerKind::GatedDelta) {
            visit(layer.first_out, MetalResourceUsage::ReadWrite);
            visit(layer.second_out, MetalResourceUsage::ReadWrite);
        }
    }
    for (const MetalBuffer& window :
         prefill.command_graph.image_windows) {
        visit(window, MetalResourceUsage::Read);
    }
    for (const PrefillCommandGraph::BufferWindow& window :
         prefill.command_graph.scratch_windows) {
        visit(window.window, MetalResourceUsage::ReadWrite);
    }
    if (prefill.command_graph.argument_arena) {
        visit(
            prefill.command_graph.argument_arena,
            MetalResourceUsage::Read);
    }
}

std::vector<std::uint64_t> prefill_pipeline_identities(
    const PrefillPipelines& pipelines) {
    std::vector<std::uint64_t> identities;
    identities.reserve(34);
    visit_prefill_pipelines(
        pipelines, [&identities](const MetalComputePipeline& pipeline) {
            identities.push_back(compute_pipeline_identity(pipeline));
        });
    return identities;
}

std::vector<std::uint64_t> prefill_resource_identities(
    const PrefillStep& prefill, const DecodeStep& decode,
    const DecodeStateSlot& state) {
    std::vector<std::uint64_t> identities;
    identities.reserve(48u + state.layers.size() * 4u);
    visit_prefill_graph_buffers(
        prefill, decode, state,
        [&identities](const MetalBuffer& buffer, MetalResourceUsage) {
            identities.push_back(metal_buffer_identity(buffer));
        });
    return identities;
}

PrefillCommandPlanKey command_graph_key(
    const PrefillCommandGraph& graph) noexcept {
    return {
        .model_package_identity = graph.model_package_identity,
        .prepared_image_identity = graph.prepared_image_identity,
        .pipeline_identity = graph.pipeline_identity,
        .execution_policy_identity =
            graph.execution_policy_identity,
        .icb_capability_identity =
            graph.icb_capability_identity,
        .state_slot_identity = graph.state_slot_identity,
        .row_count = graph.row_count,
        .context_base = graph.context_base,
        .graph_schema_version = graph.graph_schema_version,
        .chunk_rows = graph.chunk_rows,
        .persistent_resource_identities =
            graph.resource_identities,
    };
}

bool graph_pipelines_are_indirect_capable(
    const PrefillPipelines& pipelines,
    const PrefillExecutionPolicy& policy) {
    if (!valid_pipelines(pipelines, policy)) {
        return false;
    }
    bool valid = true;
    visit_prefill_pipelines(
        pipelines, [&valid](const MetalComputePipeline& pipeline) {
            valid =
                valid &&
                (!pipeline ||
                 supports_indirect_commands(pipeline));
        });
    return valid;
}

} // namespace

PrefillBufferWindowPlan plan_prefill_buffer_window(
    std::uint64_t source_bytes,
    std::uint64_t absolute_offset) noexcept {
    if (source_bytes == 0 || absolute_offset >= source_bytes) {
        return {};
    }
    if (absolute_offset <
        kIndirectKernelBufferOffsetLimitBytes) {
        return {
            .valid = true,
            .use_window = false,
            .source_begin = 0,
            .window_length = source_bytes,
            .binding_offset = absolute_offset,
        };
    }
    const std::uint64_t source_begin =
        (absolute_offset >> kPrefillImageWindowShift)
        << kPrefillImageWindowShift;
    return {
        .valid = true,
        .use_window = true,
        .source_begin = source_begin,
        .window_length = source_bytes - source_begin,
        .binding_offset = absolute_offset - source_begin,
    };
}

PrefillMemoryPlan plan_prefill_step_memory(
    const PrefillGeometry& geometry,
    const PrefillExecutionPolicy& policy) noexcept {
    if (!valid_geometry(geometry) || !valid_policy(policy, geometry) ||
        geometry.reusable_scratch_bytes < geometry.token_bytes) {
        return {};
    }

    const std::uint32_t scratch_lanes =
        policy.command_graph ? policy.command_graph_chunk_count : 1u;
    const std::uint64_t graph_status_count64 =
        policy.command_graph
            ? (std::uint64_t{geometry.gated_delta_layers} +
               geometry.attention_layers) *
                  scratch_lanes
            : 0u;
    if (graph_status_count64 >
        std::numeric_limits<std::uint32_t>::max()) {
        return {};
    }
    const std::uint32_t graph_status_count =
        static_cast<std::uint32_t>(graph_status_count64);
    const std::uint32_t task_status_count =
        std::max(
            std::max(policy.maximum_units_per_submission,
                     policy.maximum_inflight_units),
            graph_status_count);
    const bool native_routed =
        policy.routed_qgemm == QuantizedGemmPolicy::NativeRaggedMma;

    std::uint64_t native_routed_status_bytes = 0;
    if (native_routed &&
        !multiply(task_status_count, kNativeRoutedStatusBytes,
                  native_routed_status_bytes)) {
        return {
            .error = PrefillMemoryPlanError::ArithmeticOverflow,
        };
    }
    std::uint64_t native_lane_bytes = 0;
    if (native_routed &&
        !multiply(
            scratch_lanes,
            kNativeRoutedFixedWorkspaceBytes -
                kNativeRoutedSharedExpertBytes -
                kNativeRoutedSharedArgumentBytes,
            native_lane_bytes)) {
        return {
            .error = PrefillMemoryPlanError::ArithmeticOverflow,
        };
    }
    std::uint64_t native_workspace = 0;
    const auto add_to = [](std::uint64_t value,
                           std::uint64_t& total) noexcept {
        if (value > std::numeric_limits<std::uint64_t>::max() - total) {
            return false;
        }
        total += value;
        return true;
    };
    if (native_routed &&
        (!add_to(native_lane_bytes, native_workspace) ||
         !add_to(kNativeRoutedSharedExpertBytes, native_workspace) ||
         !add_to(kNativeRoutedSharedArgumentBytes, native_workspace) ||
         !add_to(native_routed_status_bytes, native_workspace) ||
         !add_to(native_routed_status_bytes, native_workspace))) {
        return {
            .error = PrefillMemoryPlanError::ArithmeticOverflow,
        };
    }

    const std::uint64_t per_lane_bytes =
        geometry.reusable_scratch_bytes - geometry.token_bytes;
    std::uint64_t all_lane_bytes = 0;
    if (!multiply(per_lane_bytes, scratch_lanes, all_lane_bytes)) {
        return {
            .error = PrefillMemoryPlanError::ArithmeticOverflow,
        };
    }
    std::uint64_t total_bytes = 0;
    if (!add_to(geometry.token_bytes, total_bytes) ||
        !add_to(geometry.hidden_slab_bytes, total_bytes) ||
        !add_to(all_lane_bytes, total_bytes) ||
        !add_to(native_workspace, total_bytes)) {
        return {
            .error = PrefillMemoryPlanError::ArithmeticOverflow,
        };
    }

    std::uint64_t maximum_single_buffer =
        std::max(geometry.token_bytes, geometry.hidden_slab_bytes);
    const std::array<std::uint64_t, 19> per_lane_allocations{
        geometry.block_hidden_bytes,
        geometry.gdn_projection_bytes,
        geometry.gdn_qk_bytes,
        geometry.gdn_value_bytes,
        geometry.gdn_parameter_bytes,
        geometry.attention_projection_bytes,
        geometry.attention_vector_bytes,
        geometry.attention_partial_bytes,
        geometry.moe_logits_bytes,
        geometry.moe_id_bytes,
        geometry.moe_coefficient_bytes,
        geometry.moe_shared_coefficient_bytes,
        geometry.moe_count_bytes,
        geometry.moe_list_bytes,
        geometry.moe_active_bytes,
        geometry.moe_indirect_argument_bytes,
        geometry.moe_hidden_bytes,
        geometry.moe_partial_bytes,
        native_routed ? kNativeRoutedTaskBytes : 0u,
    };
    for (const std::uint64_t bytes : per_lane_allocations) {
        std::uint64_t allocation_bytes = 0;
        if (!multiply(bytes, scratch_lanes, allocation_bytes)) {
            return {
                .error = PrefillMemoryPlanError::ArithmeticOverflow,
            };
        }
        maximum_single_buffer =
            std::max(maximum_single_buffer, allocation_bytes);
    }
    maximum_single_buffer =
        std::max(maximum_single_buffer, native_routed_status_bytes);

    return {
        .error = PrefillMemoryPlanError::None,
        .maximum_scratch_lanes = scratch_lanes,
        .maximum_task_status_count = graph_status_count,
        .token_bytes = geometry.token_bytes,
        .hidden_slab_bytes = geometry.hidden_slab_bytes,
        .per_lane_bytes = per_lane_bytes,
        .native_routed_workspace_bytes = native_workspace,
        .total_bytes = total_bytes,
        .maximum_single_buffer_bytes = maximum_single_buffer,
    };
}

PrefillBandPlan plan_next_prefill_band(
    const PrefillPolicy& geometry, std::uint32_t context_base,
    std::uint32_t remaining_rows,
    std::uint32_t maximum_scratch_lanes) noexcept {
    if (remaining_rows == 0) {
        return {.error = PrefillBandPlanError::Empty};
    }
    if (geometry.context_capacity == 0 ||
        geometry.first_chunk_rows == 0 ||
        geometry.maximum_block_rows == 0 ||
        geometry.first_chunk_rows > geometry.maximum_block_rows ||
        maximum_scratch_lanes == 0 ||
        geometry.schedule != PrefillSchedule::LayerMajor) {
        return {.error = PrefillBandPlanError::InvalidPolicy};
    }
    if (context_base >= geometry.context_capacity ||
        (context_base != 0 &&
         (context_base < geometry.first_chunk_rows ||
          (context_base - geometry.first_chunk_rows) %
                  geometry.maximum_block_rows !=
              0))) {
        return {.error = PrefillBandPlanError::ContextOutOfRange};
    }
    const std::uint64_t next_context64 =
        std::uint64_t{context_base} + remaining_rows;
    if (next_context64 > geometry.context_capacity ||
        next_context64 > std::numeric_limits<std::uint32_t>::max()) {
        return {.error = PrefillBandPlanError::ContextOverflow};
    }

    std::uint32_t rows = 0;
    std::uint32_t chunks = 0;
    std::uint32_t context = context_base;
    while (rows < remaining_rows && chunks < maximum_scratch_lanes) {
        const std::uint32_t chunk_limit =
            context == 0 ? geometry.first_chunk_rows
                         : geometry.maximum_block_rows;
        const std::uint32_t remainder = remaining_rows - rows;
        const std::uint32_t chunk_rows =
            std::min(chunk_limit, remainder);
        rows += chunk_rows;
        context += chunk_rows;
        ++chunks;
    }
    return {
        .error = PrefillBandPlanError::None,
        .context_base = context_base,
        .row_count = rows,
        .chunk_count = chunks,
        .next_context = context,
    };
}

PrefillStepResult create_prefill_step(const MetalDevice& device, const PrefillGeometry& geometry,
                                      PrefillExecutionPolicy policy, PrefillPipelines pipelines) {
    if (!device) {
        return {
            .error = PrefillStepError::InvalidDevice, .requested_bytes = 0, .step = std::nullopt};
    }
    if (!valid_geometry(geometry)) {
        return {
            .error = PrefillStepError::InvalidGeometry, .requested_bytes = 0, .step = std::nullopt};
    }
    if (!valid_policy(policy, geometry)) {
        return {
            .error = PrefillStepError::InvalidPolicy, .requested_bytes = 0, .step = std::nullopt};
    }
    if (!valid_pipelines(pipelines, policy)) {
        return {
            .error = PrefillStepError::PipelineUnavailable,
            .requested_bytes = 0,
            .step = std::nullopt,
        };
    }
    const PrefillMemoryPlan memory =
        plan_prefill_step_memory(geometry, policy);
    if (!memory) {
        return {
            .error = PrefillStepError::InvalidPolicy,
            .requested_bytes = 0,
            .step = std::nullopt,
        };
    }
    PrefillStep step{
        .geometry = geometry,
        .policy = policy,
        .pipelines = std::move(pipelines),
    };
    const bool native_routed =
        policy.routed_qgemm ==
        QuantizedGemmPolicy::NativeRaggedMma;
    const std::uint32_t scratch_lanes =
        memory.maximum_scratch_lanes;
    step.command_graph_task_status_count =
        memory.maximum_task_status_count;
    const std::uint32_t task_status_count =
        std::max(
            std::max(policy.maximum_units_per_submission,
                     policy.maximum_inflight_units),
            step.command_graph_task_status_count);
    const std::uint64_t native_routed_status_bytes =
        native_routed
            ? std::uint64_t{task_status_count} *
                  kNativeRoutedStatusBytes
            : 0;
    step.native_routed_workspace_bytes =
        memory.native_routed_workspace_bytes;
    const struct {
        std::uint64_t bytes;
        MetalBuffer* buffer;
        bool per_scratch_lane;
    } allocations[] = {
        {geometry.token_bytes, &step.tokens, false},
        // A39 steel-attention GEMM params arena: 8 chunk slots x 2 GEMMs
        // per layer, 72 bytes each (host-written at encode, read by the
        // sealed kernels at execution; recorder-compatible buffer binding).
        {std::uint64_t{40} * 8u * 16u * 2u * 72u,
         &step.attention_steel_params, false},
        {policy.gdn_recurrence == PrefillGdnRecurrence::RegisterLoopTape
             ? std::uint64_t{40} * 16u * (32u * 128u + 16u * 128u + 64u) * 4u
             : 64u,
         &step.gdn_tape, false},
        {policy.gdn_recurrence == PrefillGdnRecurrence::RegisterLoopTape
             ? std::uint64_t{40} * 16u * 3u * 8192u * 2u
             : 64u,
         &step.gdn_conv_tape_buffer, false},
        {policy.conditioning_capture
             ? std::uint64_t{policy.conditioning_capture_rows} * 8u *
                   2048u * 2u
             : 64u,
         &step.capture_buffer, false},
        {geometry.hidden_slab_bytes, &step.hidden_slab, false},
        {geometry.block_hidden_bytes, &step.block_hidden, true},
        {geometry.block_hidden_bytes, &step.normalized, true},
        {geometry.block_hidden_bytes, &step.branch, true},
        {geometry.block_hidden_bytes, &step.moe_output, true},
        {geometry.gdn_projection_bytes, &step.gdn_projection, true},
        {geometry.gdn_qk_bytes, &step.gdn_qk, true},
        {geometry.gdn_value_bytes, &step.gdn_value, true},
        {geometry.gdn_value_bytes, &step.gdn_gate, true},
        {geometry.gdn_value_bytes, &step.gdn_recurrence, true},
        {geometry.gdn_value_bytes, &step.gdn_gated, true},
        {geometry.gdn_parameter_bytes, &step.gdn_decay, true},
        {geometry.gdn_parameter_bytes, &step.gdn_beta, true},
        {geometry.attention_projection_bytes, &step.attention_projection, true},
        {geometry.attention_vector_bytes, &step.attention_query, true},
        {geometry.attention_vector_bytes, &step.attention_gate, true},
        {geometry.attention_vector_bytes, &step.attention_attended, true},
        {geometry.attention_partial_bytes, &step.attention_partials, true},
        {geometry.moe_logits_bytes, &step.router_logits, true},
        {geometry.moe_id_bytes, &step.expert_ids, true},
        {geometry.moe_coefficient_bytes, &step.expert_coefficients, true},
        {geometry.moe_shared_coefficient_bytes, &step.shared_coefficients, true},
        {geometry.moe_count_bytes, &step.expert_counts, true},
        {geometry.moe_list_bytes, &step.expert_lists, true},
        {geometry.moe_active_bytes, &step.active_experts, true},
        {geometry.moe_indirect_argument_bytes, &step.expert_arguments, true},
        {geometry.moe_hidden_bytes, &step.expert_hidden, true},
        {geometry.moe_partial_bytes, &step.expert_partials, true},
        {native_routed ? kNativeRoutedTaskBytes : 0,
         &step.native_routed_up_tasks, true},
        {native_routed ? kNativeRoutedArgumentBytes : 0,
         &step.native_routed_up_arguments, true},
        {native_routed_status_bytes,
         &step.native_routed_up_status, false},
        {native_routed ? kNativeRoutedTaskBytes : 0,
         &step.native_routed_down_tasks, true},
        {native_routed ? kNativeRoutedArgumentBytes : 0,
         &step.native_routed_down_arguments, true},
        {native_routed_status_bytes,
         &step.native_routed_down_status, false},
        {native_routed ? kNativeRoutedSharedExpertBytes : 0,
         &step.shared_expert, false},
        {native_routed ? kNativeRoutedSharedArgumentBytes : 0,
         &step.shared_expert_arguments, false},
    };
    for (const auto& allocation : allocations) {
        std::uint64_t allocation_bytes = allocation.bytes;
        if (allocation.per_scratch_lane &&
            !multiply(
                allocation.bytes, scratch_lanes,
                allocation_bytes)) {
            return {
                .error = PrefillStepError::BufferAllocationFailed,
                .requested_bytes = allocation.bytes,
                .step = std::nullopt,
            };
        }
        if (!allocate_zeroed(
                device, allocation_bytes, *allocation.buffer)) {
            return {
                .error = PrefillStepError::BufferAllocationFailed,
                .requested_bytes = allocation_bytes,
                .step = std::nullopt,
            };
        }
    }
    if (native_routed) {
        const std::uint32_t shared_expert = geometry.experts;
        const std::uint32_t shared_arguments[6]{
            1u,
            ceil_div(
                geometry.expert_dimension,
                policy.geometry.exact_rows_per_threadgroup),
            1u,
            1u,
            ceil_div(
                geometry.hidden,
                policy.geometry.exact_rows_per_threadgroup),
            1u,
        };
        std::memcpy(
            step.shared_expert.contents(), &shared_expert,
            sizeof(shared_expert));
        std::memcpy(
            step.shared_expert_arguments.contents(), shared_arguments,
            sizeof(shared_arguments));
    }
    if (policy.command_graph_lane_events) {
        step.command_graph_lane_queues.reserve(
            policy.command_graph_chunk_count);
        step.command_graph_lane_events.reserve(
            policy.command_graph_chunk_count - 1u);
        for (std::uint32_t lane = 0;
             lane < policy.command_graph_chunk_count; ++lane) {
            auto queue = create_command_queue(device);
            if (!queue) {
                return {
                    .error = PrefillStepError::
                        CommandGraphTopologyUnavailable,
                    .requested_bytes = 0,
                    .step = std::nullopt,
                };
            }
            step.command_graph_lane_queues.push_back(
                std::move(*queue.command_queue));
        }
        for (std::uint32_t boundary = 1;
             boundary < policy.command_graph_chunk_count;
             ++boundary) {
            auto event = create_event(device);
            if (!event) {
                return {
                    .error = PrefillStepError::
                        CommandGraphTopologyUnavailable,
                    .requested_bytes = 0,
                    .step = std::nullopt,
                };
            }
            step.command_graph_lane_events.push_back(
                std::move(*event.event));
        }
    }
    return {.error = PrefillStepError::None, .requested_bytes = 0, .step = std::move(step)};
}

template <typename DispatchProfile>
PrefillEncodeResult encode_prefill_impl(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    MetalComputePass& pass, std::uint32_t live_context,
    std::uint32_t context_base, std::span<const std::uint32_t> tokens,
    DispatchProfile dispatch_profile) {
    if (prefill.policy.command_graph) {
        return {
            .error = PrefillEncodeError::CommandGraphRequired,
            .command_error = MetalCommandError::None,
        };
    }
    const auto validation = validate_prefill_prefix(
        prefill.policy.geometry, live_context, context_base, tokens, prefill.geometry.vocabulary);
    if (!validation) {
        return {
            .error = request_error(validation.error),
            .command_error = MetalCommandError::None,
        };
    }
    const PrefillEncodeError ownership = owner_error(prefill, decode, state);
    if (ownership != PrefillEncodeError::None) {
        return {
            .error = ownership,
            .command_error = MetalCommandError::None,
        };
    }
    if (prefill.policy.dense_qgemm ==
            QuantizedGemmPolicy::NativeDenseMma &&
        !valid_native_dense_bindings(
            prefill.geometry, decode.bindings)) {
        return {
            .error = PrefillEncodeError::BindingMismatch,
            .command_error = MetalCommandError::None,
        };
    }
    if (prefill.policy.routed_qgemm ==
        QuantizedGemmPolicy::NativeRaggedMma) {
        return {
            .error =
                PrefillEncodeError::DeviceTaskValidationUnavailable,
            .command_error = MetalCommandError::None,
        };
    }
    if constexpr (DispatchProfile::profiled) {
        if (dispatch_profile.profiler.validate_sample_capacity(
                dispatch_profile.samples.sample_capacity()) !=
                PrefillProfilerError::None ||
            dispatch_profile.profiler.validate_sampling_mode(
                dispatch_profile.samples.sampling_mode()) !=
            PrefillProfilerError::None) {
            return {
                .error = PrefillEncodeError::None,
                .command_error = MetalCommandError::None,
                .next_context = context_base,
            };
        }
    }
    std::memcpy(prefill.tokens.contents(), tokens.data(), tokens.size_bytes());

    Encoder<DispatchProfile> encode(pass, *decode.image, decode.tensor_offsets,
                                    dispatch_profile);
    const std::uint32_t rows = static_cast<std::uint32_t>(tokens.size());
    const std::uint32_t chunks =
        chunk_count(prefill.policy, context_base, rows);
    const std::uint64_t hidden_row_bytes = std::uint64_t{prefill.geometry.hidden} * kBf16Bytes;

    if (prefill.policy.geometry.schedule == PrefillSchedule::ChunkMajor) {
        for (std::uint32_t ordinal = 0; ordinal < chunks; ++ordinal) {
            const Chunk chunk =
                chunk_at(prefill.policy, context_base, rows, ordinal);
            encode.barrier();
            encode_embedding(encode, prefill, decode.bindings, chunk);
            for (std::size_t layer = 0; layer < decode.schedule.size(); ++layer) {
                encode_layer(encode, prefill, decode, state, layer, chunk,
                             chunk.ordinal, context_base, prefill.block_hidden,
                             0, prefill.block_hidden, 0, 0);
            }
        }
    } else {
        for (std::size_t layer = 0; layer < decode.schedule.size(); ++layer) {
            for (std::uint32_t ordinal = 0; ordinal < chunks; ++ordinal) {
                const Chunk chunk =
                    chunk_at(prefill.policy, context_base, rows, ordinal);
                if (layer == 0) {
                    encode.barrier();
                    encode_embedding(encode, prefill, decode.bindings, chunk);
                }
                const MetalBuffer& input = layer == 0 ? prefill.block_hidden : prefill.hidden_slab;
                const std::uint64_t input_offset =
                    layer == 0 ? 0 : std::uint64_t{chunk.offset} * hidden_row_bytes;
                const std::uint64_t output_offset = std::uint64_t{chunk.offset} * hidden_row_bytes;
                encode_layer(encode, prefill, decode, state, layer, chunk,
                             chunk.ordinal, context_base, input, input_offset,
                             prefill.hidden_slab, output_offset, 0);
            }
        }
    }
    if (encode.failed()) {
        if (encode.error == MetalCommandError::None) {
            return {
                .error = PrefillEncodeError::None,
                .command_error = MetalCommandError::None,
                .next_context = context_base,
                .chunk_count = chunks,
            };
        }
        return {
            .error = PrefillEncodeError::CommandEncodingFailed,
            .command_error = encode.error,
            .next_context = context_base,
            .chunk_count = chunks,
        };
    }
    return {
        .error = PrefillEncodeError::None,
        .command_error = MetalCommandError::None,
        .next_context = validation.next_context,
        .chunk_count = chunks,
    };
}

PrefillEncodeResult encode_prefill(PrefillStep& prefill, DecodeStep& decode,
                                   DecodeStateSlot& state,
                                   MetalComputePass& pass,
                                   std::uint32_t live_context,
                                   std::uint32_t context_base,
                                   std::span<const std::uint32_t> tokens) {
    return encode_prefill_impl(prefill, decode, state, pass, live_context,
                               context_base, tokens, UnprofiledDispatches{});
}

PrefillEncodeResult encode_prefill(PrefillStep& prefill, DecodeStep& decode, MetalComputePass& pass,
                                   std::uint32_t live_context, std::uint32_t context_base,
                                   std::span<const std::uint32_t> tokens) {
    return encode_prefill(prefill, decode, decode.state, pass, live_context, context_base, tokens);
}

ProfiledPrefillEncodeResult encode_prefill(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    MetalComputePass& pass, std::uint32_t live_context,
    std::uint32_t context_base, std::span<const std::uint32_t> tokens,
    PrefillProfiler& profiler, const MetalCounterSampleBuffer& samples) {
    const PrefillEncodeResult encode =
        encode_prefill_impl(prefill, decode, state, pass, live_context,
                            context_base, tokens,
                            ProfiledDispatches{profiler, samples});
    return {.encode = encode, .profile = profiler.status()};
}

ProfiledPrefillEncodeResult encode_prefill(
    PrefillStep& prefill, DecodeStep& decode, MetalComputePass& pass,
    std::uint32_t live_context, std::uint32_t context_base,
    std::span<const std::uint32_t> tokens, PrefillProfiler& profiler,
    const MetalCounterSampleBuffer& samples) {
    return encode_prefill(prefill, decode, decode.state, pass, live_context,
                          context_base, tokens, profiler, samples);
}

void advance_prefill_state(const DecodeStep& decode, DecodeStateSlot& state, std::uint32_t chunks) {
    if (!decode_state_slot_available(decode, state)) {
        return;
    }
    if ((chunks & 1u) == 0u) {
        return;
    }
    for (std::size_t layer = 0; layer < decode.schedule.size(); ++layer) {
        if (decode.schedule[layer] == model::qwen36::LayerKind::GatedDelta) {
            state.layers[layer].swapped = !state.layers[layer].swapped;
        }
    }
}

void advance_prefill_state(DecodeStep& decode, std::uint32_t chunks) {
    advance_prefill_state(decode, decode.state, chunks);
}

PrefillProgressResult begin_prefill_progress(PrefillStep& prefill, const DecodeStep& decode,
                                             DecodeStateSlot& state, std::uint32_t live_context,
                                             std::uint32_t context_base,
                                             std::span<const std::uint32_t> tokens) {
    if (prefill.progress.state == PrefillProgressState::Ready ||
        prefill.progress.state == PrefillProgressState::UnitPending ||
        prefill.progress.state == PrefillProgressState::BatchPending ||
        prefill.progress.state ==
            PrefillProgressState::InflightEncoding ||
        prefill.progress.state ==
            PrefillProgressState::InflightPending ||
        prefill.progress.state ==
            PrefillProgressState::GraphPending) {
        return progress_failure(prefill, PrefillProgressError::Active);
    }
    if (prefill.progress.state == PrefillProgressState::Poisoned ||
        prefill.command_graph.state ==
            PrefillCommandGraphState::Poisoned) {
        return progress_failure(prefill, PrefillProgressError::Poisoned);
    }

    const auto validation = validate_prefill_prefix(
        prefill.policy.geometry, live_context, context_base, tokens, prefill.geometry.vocabulary);
    if (!validation) {
        PrefillProgressResult result;
        result.error = PrefillProgressError::None;
        result.encode_error = request_error(validation.error);
        result.state = prefill.progress.state;
        return result;
    }
    const PrefillEncodeError ownership = owner_error(prefill, decode, state);
    if (ownership != PrefillEncodeError::None) {
        PrefillProgressResult result;
        result.error = PrefillProgressError::None;
        result.encode_error = ownership;
        result.state = prefill.progress.state;
        return result;
    }
    if (prefill.policy.dense_qgemm ==
            QuantizedGemmPolicy::NativeDenseMma &&
        !valid_native_dense_bindings(
            prefill.geometry, decode.bindings)) {
        PrefillProgressResult result;
        result.error = PrefillProgressError::None;
        result.encode_error = PrefillEncodeError::BindingMismatch;
        result.state = prefill.progress.state;
        return result;
    }
    if (prefill.policy.routed_qgemm ==
            QuantizedGemmPolicy::NativeRaggedMma &&
        !valid_native_routed_bindings(
            prefill.geometry, decode.bindings)) {
        PrefillProgressResult result;
        result.error = PrefillProgressError::None;
        result.encode_error = PrefillEncodeError::BindingMismatch;
        result.state = prefill.progress.state;
        return result;
    }

    std::memcpy(prefill.tokens.contents(), tokens.data(), tokens.size_bytes());
    const std::uint32_t rows = static_cast<std::uint32_t>(tokens.size());
    prefill.progress = {
        .state = PrefillProgressState::Ready,
        .owner = &decode,
        .state_owner = &state,
        .live_context = live_context,
        .context_base = context_base,
        .next_context = validation.next_context,
        .row_count = rows,
        .chunk_count = chunk_count(prefill.policy, context_base, rows),
        .current_layer = 0,
        .current_chunk = 0,
    };
    return {
        .error = PrefillProgressError::None,
        .state = PrefillProgressState::Ready,
        .next_context = context_base,
        .chunk_count = prefill.progress.chunk_count,
    };
}

PrefillProgressResult begin_prefill_progress(PrefillStep& prefill, DecodeStep& decode,
                                             std::uint32_t live_context, std::uint32_t context_base,
                                             std::span<const std::uint32_t> tokens) {
    return begin_prefill_progress(prefill, decode, decode.state, live_context, context_base,
                                  tokens);
}

template <typename DispatchProfile>
PrefillProgressResult encode_prefill_units_impl(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    MetalComputePass& pass, std::uint32_t maximum_units,
    PrefillProgressState pending_state, DispatchProfile dispatch_profile) {
    if (prefill.policy.command_graph) {
        return progress_failure(
            prefill, PrefillProgressError::GraphRequired);
    }
    switch (prefill.progress.state) {
    case PrefillProgressState::Idle:
        return progress_failure(prefill, PrefillProgressError::Unavailable);
    case PrefillProgressState::UnitPending:
        return progress_failure(prefill, PrefillProgressError::UnitPending);
    case PrefillProgressState::BatchPending:
        return progress_failure(prefill, PrefillProgressError::BatchPending);
    case PrefillProgressState::InflightEncoding:
        return progress_failure(
            prefill, PrefillProgressError::InflightEncoding);
    case PrefillProgressState::InflightPending:
        return progress_failure(
            prefill, PrefillProgressError::InflightPending);
    case PrefillProgressState::GraphPending:
        return progress_failure(
            prefill, PrefillProgressError::GraphPending);
    case PrefillProgressState::Complete:
        return progress_failure(prefill, PrefillProgressError::Complete);
    case PrefillProgressState::Poisoned:
        return progress_failure(prefill, PrefillProgressError::Poisoned);
    case PrefillProgressState::Ready:
        break;
    }
    if (maximum_units == 0 ||
        maximum_units > prefill.policy.maximum_units_per_submission ||
        (pending_state != PrefillProgressState::UnitPending &&
         pending_state != PrefillProgressState::BatchPending)) {
        return progress_failure(prefill, PrefillProgressError::Invalid);
    }
    if (pending_state == PrefillProgressState::BatchPending &&
        (prefill.policy.geometry.schedule != PrefillSchedule::LayerMajor ||
         prefill.policy.maximum_units_per_submission == 1)) {
        return progress_failure(
            prefill, PrefillProgressError::BatchUnavailable);
    }
    if (!valid_active_progress(prefill, decode, state)) {
        return progress_failure(prefill, PrefillProgressError::Invalid);
    }
    const PrefillEncodeError ownership = owner_error(prefill, decode, state);
    if (ownership != PrefillEncodeError::None) {
        PrefillProgressResult result = progress_failure(prefill, PrefillProgressError::None);
        result.encode_error = ownership;
        return result;
    }
    if constexpr (DispatchProfile::profiled) {
        if (dispatch_profile.profiler.validate_sample_capacity(
                dispatch_profile.samples.sample_capacity()) !=
                PrefillProfilerError::None ||
            dispatch_profile.profiler.validate_sampling_mode(
                dispatch_profile.samples.sampling_mode()) !=
            PrefillProfilerError::None) {
            return progress_failure(prefill, PrefillProgressError::None);
        }
    }

    Encoder<DispatchProfile> encode(pass, *decode.image, decode.tensor_offsets,
                                    dispatch_profile);
    const PrefillProgress& progress = prefill.progress;
    const std::uint64_t hidden_row_bytes =
        std::uint64_t{prefill.geometry.hidden} * kBf16Bytes;
    const std::uint32_t encoded_layer = progress.current_layer;
    const std::uint32_t encoded_chunk = progress.current_chunk;
    std::uint32_t encoded_units = 1;

    if (prefill.policy.geometry.schedule == PrefillSchedule::ChunkMajor) {
        const Chunk chunk =
            chunk_at(
                prefill.policy, progress.context_base, progress.row_count,
                progress.current_chunk);
        encode.barrier();
        encode_embedding(encode, prefill, decode.bindings, chunk);
        for (std::size_t layer = 0; layer < decode.schedule.size(); ++layer) {
            // Each bounded chunk commits the GDN planes before the next
            // chunk, so its recurrence phase starts at zero while its
            // profiling identity retains the request chunk ordinal.
            encode_layer(encode, prefill, decode, state, layer, chunk, 0,
                         progress.context_base, prefill.block_hidden, 0,
                         prefill.block_hidden, 0, 0);
        }
    } else {
        const std::uint64_t first_unit =
            std::uint64_t{progress.current_layer} *
                progress.chunk_count +
            progress.current_chunk;
        const std::uint64_t total_units =
            std::uint64_t{decode.schedule.size()} *
            progress.chunk_count;
        encoded_units = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(
                maximum_units, total_units - first_unit));
        for (std::uint32_t unit = 0; unit < encoded_units; ++unit) {
            const std::uint64_t logical_unit = first_unit + unit;
            const std::size_t layer =
                static_cast<std::size_t>(
                    logical_unit / progress.chunk_count);
            const std::uint32_t chunk_ordinal =
                static_cast<std::uint32_t>(
                    logical_unit % progress.chunk_count);
            const Chunk chunk =
                chunk_at(
                    prefill.policy, progress.context_base, progress.row_count,
                    chunk_ordinal);
            if (layer == 0) {
                encode.barrier();
                encode_embedding(
                    encode, prefill, decode.bindings, chunk);
            }
            const MetalBuffer& input =
                layer == 0 ? prefill.block_hidden
                           : prefill.hidden_slab;
            const std::uint64_t input_offset =
                layer == 0
                    ? 0
                    : std::uint64_t{chunk.offset} *
                          hidden_row_bytes;
            const std::uint64_t output_offset =
                std::uint64_t{chunk.offset} *
                hidden_row_bytes;
            encode_layer(
                encode, prefill, decode, state, layer, chunk,
                chunk.ordinal, progress.context_base, input,
                input_offset, prefill.hidden_slab,
                output_offset, unit);
        }
    }

    if (encode.failed()) {
        if (encode.error == MetalCommandError::None) {
            return progress_failure(prefill, PrefillProgressError::None);
        }
        PrefillProgressResult result = progress_failure(prefill, PrefillProgressError::None);
        result.encode_error = PrefillEncodeError::CommandEncodingFailed;
        result.command_error = encode.error;
        return result;
    }
    prefill.progress.state = pending_state;
    prefill.progress.pending_unit_count = encoded_units;
    return {
        .error = PrefillProgressError::None,
        .state = pending_state,
        .next_context = progress.context_base,
        .chunk_count = progress.chunk_count,
        .layer_index = encoded_layer,
        .chunk_ordinal = encoded_chunk,
        .unit_count = encoded_units,
    };
}

PrefillProgressResult encode_prefill_unit(PrefillStep& prefill,
                                          DecodeStep& decode,
                                          DecodeStateSlot& state,
                                          MetalComputePass& pass) {
    return encode_prefill_units_impl(
        prefill, decode, state, pass, 1,
        PrefillProgressState::UnitPending,
        UnprofiledDispatches{});
}

PrefillProgressResult encode_prefill_unit(PrefillStep& prefill, DecodeStep& decode,
                                          MetalComputePass& pass) {
    return encode_prefill_unit(prefill, decode, decode.state, pass);
}

ProfiledPrefillProgressResult encode_prefill_unit(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    MetalComputePass& pass, PrefillProfiler& profiler,
    const MetalCounterSampleBuffer& samples) {
    const PrefillProgressResult progress = encode_prefill_units_impl(
        prefill, decode, state, pass, 1,
        PrefillProgressState::UnitPending,
        ProfiledDispatches{profiler, samples});
    return {.progress = progress, .profile = profiler.status()};
}

ProfiledPrefillProgressResult encode_prefill_unit(
    PrefillStep& prefill, DecodeStep& decode, MetalComputePass& pass,
    PrefillProfiler& profiler, const MetalCounterSampleBuffer& samples) {
    return encode_prefill_unit(prefill, decode, decode.state, pass, profiler,
                               samples);
}

PrefillProgressResult encode_prefill_units(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    MetalComputePass& pass) {
    return encode_prefill_units_impl(
        prefill, decode, state, pass,
        prefill.policy.maximum_units_per_submission,
        PrefillProgressState::BatchPending,
        UnprofiledDispatches{});
}

PrefillProgressResult encode_prefill_units(
    PrefillStep& prefill, DecodeStep& decode,
    MetalComputePass& pass) {
    return encode_prefill_units(
        prefill, decode, decode.state, pass);
}

PrefillProgressResult encode_prefill_units(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    MetalComputePass& pass, std::uint32_t maximum_units) {
    return encode_prefill_units_impl(
        prefill, decode, state, pass, maximum_units,
        PrefillProgressState::BatchPending,
        UnprofiledDispatches{});
}

PrefillProgressResult encode_prefill_inflight_unit(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    MetalComputePass& pass) {
    if (prefill.policy.command_graph) {
        return progress_failure(
            prefill, PrefillProgressError::GraphRequired);
    }
    switch (prefill.progress.state) {
    case PrefillProgressState::Idle:
        return progress_failure(
            prefill, PrefillProgressError::Unavailable);
    case PrefillProgressState::Ready:
    case PrefillProgressState::InflightEncoding:
        break;
    case PrefillProgressState::UnitPending:
        return progress_failure(
            prefill, PrefillProgressError::UnitPending);
    case PrefillProgressState::BatchPending:
        return progress_failure(
            prefill, PrefillProgressError::BatchPending);
    case PrefillProgressState::InflightPending:
        return progress_failure(
            prefill, PrefillProgressError::InflightPending);
    case PrefillProgressState::GraphPending:
        return progress_failure(
            prefill, PrefillProgressError::GraphPending);
    case PrefillProgressState::Complete:
        return progress_failure(
            prefill, PrefillProgressError::Complete);
    case PrefillProgressState::Poisoned:
        return progress_failure(
            prefill, PrefillProgressError::Poisoned);
    }
    if (prefill.policy.geometry.schedule != PrefillSchedule::LayerMajor ||
        prefill.policy.maximum_inflight_units == 1) {
        return progress_failure(
            prefill, PrefillProgressError::InflightUnavailable);
    }
    if (!valid_active_progress(prefill, decode, state)) {
        return progress_failure(
            prefill, PrefillProgressError::Invalid);
    }
    const PrefillEncodeError ownership =
        owner_error(prefill, decode, state);
    if (ownership != PrefillEncodeError::None) {
        PrefillProgressResult result =
            progress_failure(
                prefill, PrefillProgressError::None);
        result.encode_error = ownership;
        return result;
    }

    const PrefillProgress& progress = prefill.progress;
    const std::uint64_t first_unit =
        std::uint64_t{progress.current_layer} *
            progress.chunk_count +
        progress.current_chunk;
    const std::uint64_t logical_unit =
        first_unit + progress.pending_unit_count;
    const std::uint64_t total_units =
        std::uint64_t{decode.schedule.size()} *
        progress.chunk_count;
    if (logical_unit >= total_units) {
        return progress_failure(
            prefill, PrefillProgressError::Invalid);
    }
    const std::size_t layer =
        static_cast<std::size_t>(
            logical_unit / progress.chunk_count);
    const std::uint32_t chunk_ordinal =
        static_cast<std::uint32_t>(
            logical_unit % progress.chunk_count);
    const Chunk chunk =
        chunk_at(
            prefill.policy, progress.context_base, progress.row_count,
            chunk_ordinal);
    const std::uint32_t status_slot =
        progress.pending_unit_count;
    const std::uint64_t hidden_row_bytes =
        std::uint64_t{prefill.geometry.hidden} * kBf16Bytes;

    Encoder<UnprofiledDispatches> encode(
        pass, *decode.image, decode.tensor_offsets,
        UnprofiledDispatches{});
    if (layer == 0) {
        encode.barrier();
        encode_embedding(
            encode, prefill, decode.bindings, chunk);
    }
    const MetalBuffer& input =
        layer == 0 ? prefill.block_hidden
                   : prefill.hidden_slab;
    const std::uint64_t input_offset =
        layer == 0
            ? 0
            : std::uint64_t{chunk.offset} *
                  hidden_row_bytes;
    const std::uint64_t output_offset =
        std::uint64_t{chunk.offset} * hidden_row_bytes;
    encode_layer(
        encode, prefill, decode, state, layer, chunk,
        chunk.ordinal, progress.context_base, input,
        input_offset, prefill.hidden_slab, output_offset,
        status_slot);
    if (encode.failed()) {
        if (encode.error == MetalCommandError::None) {
            return progress_failure(
                prefill, PrefillProgressError::None);
        }
        PrefillProgressResult result =
            progress_failure(
                prefill, PrefillProgressError::None);
        result.encode_error =
            PrefillEncodeError::CommandEncodingFailed;
        result.command_error = encode.error;
        return result;
    }

    ++prefill.progress.pending_unit_count;
    const bool window_full =
        prefill.progress.pending_unit_count ==
        prefill.policy.maximum_inflight_units;
    const bool request_exhausted =
        logical_unit + 1u == total_units;
    prefill.progress.state =
        window_full || request_exhausted
            ? PrefillProgressState::InflightPending
            : PrefillProgressState::InflightEncoding;
    return {
        .error = PrefillProgressError::None,
        .state = prefill.progress.state,
        .next_context = progress.context_base,
        .chunk_count = progress.chunk_count,
        .layer_index = static_cast<std::uint32_t>(layer),
        .chunk_ordinal = chunk_ordinal,
        .unit_count = 1,
    };
}

PrefillProgressResult encode_prefill_inflight_unit(
    PrefillStep& prefill, DecodeStep& decode,
    MetalComputePass& pass) {
    return encode_prefill_inflight_unit(
        prefill, decode, decode.state, pass);
}

PrefillProgressResult commit_prefill_pending(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    PrefillProgressState pending_state) {
    switch (prefill.progress.state) {
    case PrefillProgressState::Idle:
        return progress_failure(prefill, PrefillProgressError::Unavailable);
    case PrefillProgressState::Ready:
        return progress_failure(
            prefill,
            pending_state == PrefillProgressState::UnitPending
                ? PrefillProgressError::UnitNotPending
                : pending_state ==
                          PrefillProgressState::BatchPending
                      ? PrefillProgressError::BatchNotPending
                      : PrefillProgressError::InflightNotPending);
    case PrefillProgressState::UnitPending:
        if (pending_state != PrefillProgressState::UnitPending) {
            return progress_failure(
                prefill, PrefillProgressError::UnitPending);
        }
        break;
    case PrefillProgressState::BatchPending:
        if (pending_state != PrefillProgressState::BatchPending) {
            return progress_failure(
                prefill, PrefillProgressError::BatchPending);
        }
        break;
    case PrefillProgressState::InflightEncoding:
        return progress_failure(
            prefill, PrefillProgressError::InflightEncoding);
    case PrefillProgressState::InflightPending:
        if (pending_state !=
            PrefillProgressState::InflightPending) {
            return progress_failure(
                prefill,
                PrefillProgressError::InflightPending);
        }
        break;
    case PrefillProgressState::GraphPending:
        return progress_failure(
            prefill, PrefillProgressError::GraphPending);
    case PrefillProgressState::Complete:
        return progress_failure(prefill, PrefillProgressError::Complete);
    case PrefillProgressState::Poisoned:
        return progress_failure(prefill, PrefillProgressError::Poisoned);
    }
    if (!valid_active_progress(prefill, decode, state)) {
        prefill.progress.state = PrefillProgressState::Poisoned;
        return progress_failure(prefill, PrefillProgressError::Invalid);
    }
    if (prefill.policy.routed_qgemm ==
        QuantizedGemmPolicy::NativeRaggedMma) {
        for (std::uint32_t unit = 0;
             unit < prefill.progress.pending_unit_count; ++unit) {
            const QuantizedGemmDeviceTaskStatus up_status =
                device_task_status(
                    prefill.native_routed_up_status, unit);
            const QuantizedGemmDeviceTaskStatus down_status =
                device_task_status(
                    prefill.native_routed_down_status, unit);
            if (up_status != QuantizedGemmDeviceTaskStatus::Ready ||
                down_status != QuantizedGemmDeviceTaskStatus::Ready) {
                prefill.progress.state =
                    PrefillProgressState::Poisoned;
                PrefillProgressResult result = progress_failure(
                    prefill,
                    PrefillProgressError::DeviceTaskNotReady);
                result.failed_unit_offset = unit;
                result.routed_up_status = up_status;
                result.routed_down_status = down_status;
                return result;
            }
        }
    }

    const std::uint32_t committed_layer = prefill.progress.current_layer;
    const std::uint32_t committed_chunk = prefill.progress.current_chunk;
    const std::uint32_t committed_units =
        prefill.progress.pending_unit_count;
    if (prefill.policy.geometry.schedule == PrefillSchedule::ChunkMajor) {
        if (committed_units != 1) {
            prefill.progress.state = PrefillProgressState::Poisoned;
            return progress_failure(
                prefill, PrefillProgressError::Invalid);
        }
        advance_prefill_state(decode, state, 1);
        ++prefill.progress.current_chunk;
        prefill.progress.state =
            prefill.progress.current_chunk ==
                    prefill.progress.chunk_count
                ? PrefillProgressState::Complete
                : PrefillProgressState::Ready;
    } else {
        for (std::uint32_t unit = 0; unit < committed_units; ++unit) {
            const bool layer_complete =
                prefill.progress.current_chunk + 1u ==
                prefill.progress.chunk_count;
            if (!layer_complete) {
                ++prefill.progress.current_chunk;
            } else {
                if ((prefill.progress.chunk_count & 1u) != 0u &&
                    decode.schedule[prefill.progress.current_layer] ==
                        model::qwen36::LayerKind::GatedDelta) {
                    DecodeLayerState& layer_state =
                        state.layers[prefill.progress.current_layer];
                    layer_state.swapped = !layer_state.swapped;
                }
                ++prefill.progress.current_layer;
                prefill.progress.current_chunk = 0;
            }
        }
        prefill.progress.state =
            prefill.progress.current_layer == decode.schedule.size()
                ? PrefillProgressState::Complete
                : PrefillProgressState::Ready;
    }
    prefill.progress.pending_unit_count = 0;

    return {
        .error = PrefillProgressError::None,
        .state = prefill.progress.state,
        .next_context = prefill.progress.state == PrefillProgressState::Complete
                            ? prefill.progress.next_context
                            : prefill.progress.context_base,
        .chunk_count = prefill.progress.chunk_count,
        .layer_index = committed_layer,
        .chunk_ordinal = committed_chunk,
        .unit_count = committed_units,
    };
}

PrefillProgressResult commit_prefill_unit(
    PrefillStep& prefill, DecodeStep& decode,
    DecodeStateSlot& state) {
    return commit_prefill_pending(
        prefill, decode, state,
        PrefillProgressState::UnitPending);
}

PrefillProgressResult commit_prefill_unit(PrefillStep& prefill, DecodeStep& decode) {
    return commit_prefill_unit(prefill, decode, decode.state);
}

PrefillProgressResult commit_prefill_units(
    PrefillStep& prefill, DecodeStep& decode,
    DecodeStateSlot& state) {
    return commit_prefill_pending(
        prefill, decode, state,
        PrefillProgressState::BatchPending);
}

PrefillProgressResult commit_prefill_units(
    PrefillStep& prefill, DecodeStep& decode) {
    return commit_prefill_units(
        prefill, decode, decode.state);
}

PrefillProgressResult commit_prefill_inflight(
    PrefillStep& prefill, DecodeStep& decode,
    DecodeStateSlot& state) {
    return commit_prefill_pending(
        prefill, decode, state,
        PrefillProgressState::InflightPending);
}

PrefillProgressResult commit_prefill_inflight(
    PrefillStep& prefill, DecodeStep& decode) {
    return commit_prefill_inflight(
        prefill, decode, decode.state);
}

PrefillCommandGraphResult prepare_prefill_command_graph(
    const MetalDevice& device, PrefillStep& prefill,
    DecodeStep& decode, DecodeStateSlot& state) {
    if (!prefill.policy.command_graph) {
        return {.error = PrefillCommandGraphError::Disabled};
    }
    if (prefill.command_graph.state ==
            PrefillCommandGraphState::Pending ||
        prefill.progress.state ==
            PrefillProgressState::GraphPending) {
        return {.error = PrefillCommandGraphError::Pending};
    }
    if (prefill.command_graph.state ==
            PrefillCommandGraphState::Poisoned ||
        prefill.progress.state ==
            PrefillProgressState::Poisoned) {
        return {.error = PrefillCommandGraphError::Poisoned};
    }
    if (!device || prefill.progress.state !=
                       PrefillProgressState::Ready ||
        prefill.progress.owner != &decode ||
        prefill.progress.state_owner != &state ||
        !valid_active_progress(prefill, decode, state)) {
        return {
            .error =
                PrefillCommandGraphError::ProgressUnavailable,
        };
    }
    if (prefill.progress.chunk_count == 0 ||
        prefill.progress.chunk_count >
            prefill.policy.command_graph_chunk_count ||
        prefill.progress.pending_unit_count != 0 ||
        prefill.progress.current_layer != 0 ||
        prefill.progress.current_chunk != 0) {
        return {
            .error = PrefillCommandGraphError::PlanMismatch,
            .stage = PrefillCommandGraphStage::Admission,
        };
    }
    if (!graph_pipelines_are_indirect_capable(
            prefill.pipelines, prefill.policy)) {
        return {
            .error =
                PrefillCommandGraphError::PipelineUnavailable,
        };
    }

    try {
        const std::vector<std::uint64_t> pipeline_identities =
            prefill_pipeline_identities(prefill.pipelines);
        const PrefillCommandIdentity package_identity =
            execution_model_package_identity();
        const PrefillCommandIdentity image_identity =
            execution_prepared_image_identity(decode);
        const PrefillCommandIdentity pipelines_identity =
            execution_pipeline_identity(pipeline_identities);
        const PrefillCommandIdentity policy_identity =
            prefill_execution_policy_identity(prefill.policy);
        const std::uint64_t capability_identity =
            metal_device_identity(device);
        const std::uint64_t state_slot_identity =
            static_cast<std::uint64_t>(
                reinterpret_cast<std::uintptr_t>(&state));
        std::vector<std::uint8_t> state_phases;
        state_phases.reserve(state.layers.size());
        for (const DecodeLayerState& layer : state.layers) {
            state_phases.push_back(layer.swapped ? 1u : 0u);
        }

        bool chunk_schedule_matches =
            prefill.command_graph.chunk_rows.size() ==
            prefill.progress.chunk_count;
        for (std::uint32_t ordinal = 0;
             chunk_schedule_matches &&
             ordinal < prefill.progress.chunk_count;
             ++ordinal) {
            chunk_schedule_matches =
                prefill.command_graph.chunk_rows[ordinal] ==
                chunk_at(
                    prefill.policy, prefill.progress.context_base,
                    prefill.progress.row_count,
                    ordinal)
                    .rows;
        }
        const bool basic_cache_match =
            prefill.command_graph.state ==
                PrefillCommandGraphState::Ready &&
            prefill.command_graph.decode_owner == &decode &&
            prefill.command_graph.state_owner == &state &&
            prefill.command_graph.row_count ==
                prefill.progress.row_count &&
            prefill.command_graph.context_base ==
                prefill.progress.context_base &&
            chunk_schedule_matches &&
            prefill.command_graph.model_package_identity ==
                package_identity &&
            prefill.command_graph.prepared_image_identity ==
                image_identity &&
            prefill.command_graph.pipeline_identity ==
                pipelines_identity &&
            prefill.command_graph.execution_policy_identity ==
                policy_identity &&
            prefill.command_graph.icb_capability_identity ==
                capability_identity &&
            prefill.command_graph.state_slot_identity ==
                state_slot_identity &&
            prefill.command_graph.graph_schema_version ==
                kPrefillCommandGraphSchemaVersion &&
            prefill.command_graph.pipeline_identities ==
                pipeline_identities &&
            prefill.command_graph.state_phases == state_phases;
        if (basic_cache_match) {
            const std::vector<std::uint64_t> resource_identities =
                prefill_resource_identities(
                    prefill, decode, state);
            PrefillCommandPlanKey candidate =
                command_graph_key(prefill.command_graph);
            candidate.persistent_resource_identities =
                resource_identities;
            const PrefillCommandPlanKey cached =
                command_graph_key(prefill.command_graph);
            if (validate_prefill_command_plan_key(candidate) ==
                    PrefillCommandPlanError::None &&
                validate_prefill_command_plan_key(cached) ==
                    PrefillCommandPlanError::None &&
                same_prefill_command_plan_key(candidate, cached)) {
                return {
                    .error = PrefillCommandGraphError::None,
                    .command_count =
                        prefill.command_graph.command_count,
                    .node_count = static_cast<std::uint32_t>(
                        prefill.command_graph.nodes.size()),
                    .diagonal_count =
                        static_cast<std::uint32_t>(
                            prefill.command_graph.diagonals.size()),
                    .argument_arena_bytes =
                        prefill.command_graph.argument_arena_bytes,
                    .cache_hit = true,
                };
            }
        }

        PrefillCommandGraph graph;
        graph.decode_owner = &decode;
        graph.state_owner = &state;
        graph.row_count = prefill.progress.row_count;
        graph.context_base = prefill.progress.context_base;
        graph.model_package_identity = package_identity;
        graph.prepared_image_identity = image_identity;
        graph.pipeline_identity = pipelines_identity;
        graph.execution_policy_identity = policy_identity;
        graph.icb_capability_identity = capability_identity;
        graph.state_slot_identity = state_slot_identity;
        graph.graph_schema_version =
            kPrefillCommandGraphSchemaVersion;
        graph.pipeline_identities = pipeline_identities;
        graph.state_phases = state_phases;
        graph.chunk_rows.resize(prefill.progress.chunk_count);
        for (std::uint32_t ordinal = 0;
             ordinal < prefill.progress.chunk_count; ++ordinal) {
            graph.chunk_rows[ordinal] =
                chunk_at(
                    prefill.policy, prefill.progress.context_base,
                    prefill.progress.row_count,
                    ordinal)
                    .rows;
        }

        const std::uint64_t layer_count = decode.schedule.size();
        const std::uint64_t node_count64 =
            layer_count * prefill.progress.chunk_count;
        const std::uint64_t diagonal_count64 =
            layer_count + prefill.progress.chunk_count - 1u;
        if (layer_count == 0 ||
            node_count64 >
                std::numeric_limits<std::uint32_t>::max() ||
            diagonal_count64 >
                std::numeric_limits<std::uint32_t>::max()) {
            return {
                .error = PrefillCommandGraphError::PlanMismatch,
                .stage = PrefillCommandGraphStage::Wavefront,
            };
        }
        graph.nodes.resize(
            static_cast<std::size_t>(node_count64));
        graph.diagonals.resize(
            static_cast<std::size_t>(diagonal_count64));
        const PrefillWavefrontPlanResult wavefront =
            build_prefill_wavefront_plan(
                {
                    .layer_count =
                        static_cast<std::uint32_t>(layer_count),
                    .row_count = prefill.progress.row_count,
                    .context_base =
                        prefill.progress.context_base,
                    .context_capacity =
                        prefill.policy.geometry.context_capacity,
                    .scratch_lane_count =
                        prefill.progress.chunk_count,
                    .chunk_rows = graph.chunk_rows,
                },
                graph.nodes, graph.diagonals);
        if (!wavefront) {
            return {
                .error = PrefillCommandGraphError::PlanMismatch,
                .stage = PrefillCommandGraphStage::Wavefront,
                .plan_error = wavefront.error,
            };
        }

        if (decode.image->size_bytes() >
            kIndirectKernelBufferOffsetLimitBytes) {
            const std::uint64_t window_count =
                (decode.image->size_bytes() +
                 kPrefillImageWindowBytes - 1u) /
                kPrefillImageWindowBytes;
            graph.image_windows.reserve(
                static_cast<std::size_t>(window_count));
            for (std::uint64_t window = 0;
                 window < window_count; ++window) {
                const std::uint64_t window_begin =
                    window * kPrefillImageWindowBytes;
                const std::uint64_t window_length = std::min(
                    2u * kPrefillImageWindowBytes,
                    decode.image->size_bytes() - window_begin);
                auto view = create_buffer_window(
                    *decode.image, window_begin, window_length);
                if (!view) {
                    return {
                        .error = PrefillCommandGraphError::
                            ResourceMismatch,
                        .stage =
                            PrefillCommandGraphStage::Admission,
                    };
                }
                graph.image_windows.push_back(
                    std::move(*view.buffer));
            }
        }

        bool scratch_windows_valid = true;
        visit_prefill_scratch_lane_buffers(
            prefill,
            [&](MetalBuffer& buffer) {
                if (!scratch_windows_valid ||
                    buffer.size_bytes() <=
                        kIndirectKernelBufferOffsetLimitBytes) {
                    return;
                }
                for (std::uint64_t source_begin =
                         kIndirectKernelBufferOffsetLimitBytes;
                     source_begin < buffer.size_bytes();
                     source_begin +=
                         kPrefillImageWindowBytes) {
                    auto view = create_buffer_window(
                        buffer, source_begin,
                        buffer.size_bytes() - source_begin);
                    if (!view) {
                        scratch_windows_valid = false;
                        return;
                    }
                    graph.scratch_windows.push_back({
                        .source = &buffer,
                        .source_begin = source_begin,
                        .window =
                            std::move(*view.buffer),
                    });
                }
            });
        if (!scratch_windows_valid) {
            return {
                .error =
                    PrefillCommandGraphError::ResourceMismatch,
                .stage =
                    PrefillCommandGraphStage::Admission,
            };
        }

        std::vector<RecordedNode> recorded_nodes(
            graph.nodes.size());
        const std::uint64_t hidden_row_bytes =
            std::uint64_t{prefill.geometry.hidden} * kBf16Bytes;
        for (std::size_t node_index = 0;
             node_index < graph.nodes.size(); ++node_index) {
            const PrefillCommandNode& node =
                graph.nodes[node_index];
            const Chunk chunk = chunk_at(
                prefill.policy, prefill.progress.context_base,
                prefill.progress.row_count,
                node.chunk_ordinal);
            RecordingEncoder encode(
                prefill, *decode.image, decode.tensor_offsets,
                graph.image_windows,
                graph.scratch_windows,
                node.scratch_lane,
                recorded_nodes[node_index].commands);
            if (node.layer_index == 0) {
                encode_embedding(
                    encode, prefill, decode.bindings, chunk);
            }
            const MetalBuffer& input =
                node.layer_index == 0 ? prefill.block_hidden
                                      : prefill.hidden_slab;
            const std::uint64_t input_offset =
                node.layer_index == 0
                    ? 0
                    : std::uint64_t{chunk.offset} *
                          hidden_row_bytes;
            const std::uint64_t output_offset =
                std::uint64_t{chunk.offset} * hidden_row_bytes;
            encode_layer(
                encode, prefill, decode, state,
                node.layer_index, chunk, chunk.ordinal,
                prefill.progress.context_base, input,
                input_offset, prefill.hidden_slab,
                output_offset,
                static_cast<std::uint32_t>(
                    node.layer_major_index));
            if (encode.failed() ||
                recorded_nodes[node_index].commands.empty()) {
                return {
                    .error =
                        PrefillCommandGraphError::RecordingFailed,
                    .stage = PrefillCommandGraphStage::Recording,
                    .command_error = encode.error,
                };
            }
        }

        std::uint64_t recorded_command_count = 0;
        for (const RecordedNode& node : recorded_nodes) {
            recorded_command_count += node.commands.size();
        }
        if (recorded_command_count == 0 ||
            recorded_command_count >
                std::numeric_limits<std::uint32_t>::max()) {
            return {
                .error = PrefillCommandGraphError::PlanMismatch,
                .stage = PrefillCommandGraphStage::Recording,
            };
        }
        std::vector<OrderedRecordedCommand> ordered;
        ordered.reserve(
            static_cast<std::size_t>(recorded_command_count));
        std::vector<std::uint32_t> level_command_begins;
        std::vector<std::uint32_t>
            node_level_boundary_offsets;
        std::vector<std::uint32_t>
            node_level_command_begins;
        if (prefill.policy.command_graph_lane_events) {
            graph.lane_event_nodes.resize(graph.nodes.size());
            const PrefillLaneEventPlanResult lane_plan =
                build_prefill_lane_event_plan(
                    {
                        .layer_count = static_cast<std::uint32_t>(
                            layer_count),
                        .scratch_lane_count =
                            prefill.progress.chunk_count,
                        .event_value_base = 0,
                        .nodes = graph.nodes,
                    },
                    graph.lane_event_nodes);
            if (!lane_plan ||
                lane_plan.plan.node_count != graph.nodes.size() ||
                lane_plan.plan.event_count + 1u !=
                    prefill.progress.chunk_count) {
                return {
                    .error =
                        PrefillCommandGraphError::PlanMismatch,
                    .stage =
                        PrefillCommandGraphStage::Ordering,
                    .plan_error = lane_plan.error,
                };
            }
            node_level_boundary_offsets.reserve(
                graph.nodes.size() + 1u);
            for (const PrefillLaneEventNode& event_node :
                 graph.lane_event_nodes) {
                if (event_node.node_index >=
                    recorded_nodes.size()) {
                    return {
                        .error =
                            PrefillCommandGraphError::PlanMismatch,
                        .stage =
                            PrefillCommandGraphStage::Ordering,
                    };
                }
                node_level_boundary_offsets.push_back(
                    static_cast<std::uint32_t>(
                        node_level_command_begins.size()));
                const RecordedNode& node =
                    recorded_nodes[event_node.node_index];
                const std::uint32_t maximum_level =
                    node.commands.back().local_level;
                for (std::uint32_t level = 0;
                     level <= maximum_level; ++level) {
                    node_level_command_begins.push_back(
                        static_cast<std::uint32_t>(
                            ordered.size()));
                    for (const RecordedCommand& command :
                         node.commands) {
                        if (command.local_level == level) {
                            ordered.push_back(
                                {.command = &command});
                        }
                    }
                    if (node_level_command_begins.back() ==
                        static_cast<std::uint32_t>(
                            ordered.size())) {
                        return {
                            .error =
                                PrefillCommandGraphError::
                                    PlanMismatch,
                            .stage =
                                PrefillCommandGraphStage::
                                    Ordering,
                        };
                    }
                }
                node_level_command_begins.push_back(
                    static_cast<std::uint32_t>(
                        ordered.size()));
            }
            node_level_boundary_offsets.push_back(
                static_cast<std::uint32_t>(
                    node_level_command_begins.size()));
        } else {
            // Apple documents MTLIndirectComputeCommand setBarrier as
            // ordering only the command that carries it, never commands
            // after it. Control levels therefore remain contiguous ICB
            // ranges with an encoder memory barrier between ranges.
            for (const PrefillCommandDiagonal& diagonal :
                 graph.diagonals) {
                std::uint32_t maximum_level = 0;
                for (std::uint64_t offset = 0;
                     offset < diagonal.node_count; ++offset) {
                    const RecordedNode& node =
                        recorded_nodes[
                            diagonal.node_begin + offset];
                    maximum_level = std::max(
                        maximum_level,
                        node.commands.back().local_level);
                }
                for (std::uint32_t level = 0;
                     level <= maximum_level; ++level) {
                    level_command_begins.push_back(
                        static_cast<std::uint32_t>(
                            ordered.size()));
                    for (std::uint64_t offset = 0;
                         offset < diagonal.node_count; ++offset) {
                        const RecordedNode& node =
                            recorded_nodes[
                                diagonal.node_begin + offset];
                        for (const RecordedCommand& command :
                             node.commands) {
                            if (command.local_level != level) {
                                continue;
                            }
                            ordered.push_back(
                                {.command = &command});
                        }
                    }
                    if (level_command_begins.back() ==
                        static_cast<std::uint32_t>(
                            ordered.size())) {
                        return {
                            .error =
                                PrefillCommandGraphError::
                                    PlanMismatch,
                            .stage =
                                PrefillCommandGraphStage::
                                    Ordering,
                        };
                    }
                }
            }
            level_command_begins.push_back(
                static_cast<std::uint32_t>(ordered.size()));
        }
        graph.command_classes.reserve(ordered.size());
        for (const OrderedRecordedCommand& ordered_command :
             ordered) {
            graph.command_classes.push_back(
                static_cast<std::uint8_t>(
                    ordered_command.command->event_class));
        }
        if (ordered.size() != recorded_command_count) {
            return {
                .error = PrefillCommandGraphError::PlanMismatch,
                .stage = PrefillCommandGraphStage::Ordering,
            };
        }

        const PrefillProfilePlanResult profile_plan =
            make_prefill_profile_plan(
                prefill.geometry, prefill.policy,
                prefill.progress.row_count, decode.schedule, {},
                prefill.progress.context_base);
        const std::uint64_t removed_builders =
            2u * graph.nodes.size();
        if (profile_plan.error !=
                PrefillProfilePlanError::
                    EventCapacityInsufficient ||
            profile_plan.required_event_count <
                removed_builders ||
            profile_plan.required_event_count -
                    removed_builders !=
                ordered.size()) {
            return {
                .error = PrefillCommandGraphError::PlanMismatch,
                .stage = PrefillCommandGraphStage::Census,
                .command_count = static_cast<std::uint32_t>(
                    ordered.size()),
                .node_count = static_cast<std::uint32_t>(
                    graph.nodes.size()),
            };
        }

        constexpr std::uint64_t kNoConstantOffset =
            std::numeric_limits<std::uint64_t>::max();
        std::vector<std::array<std::uint64_t,
                               kMaxBufferArgumentIndex + 1u>>
            constant_offsets(ordered.size());
        std::uint64_t arena_cursor = 0;
        for (std::size_t command_index = 0;
             command_index < ordered.size(); ++command_index) {
            constant_offsets[command_index].fill(
                kNoConstantOffset);
            const RecordedCommand& command =
                *ordered[command_index].command;
            for (std::uint32_t index = 0;
                 index <= kMaxBufferArgumentIndex; ++index) {
                const RecordedBindingKind kind =
                    command.bindings[index].kind;
                if (kind != RecordedBindingKind::Constant32 &&
                    kind != RecordedBindingKind::Constant64) {
                    continue;
                }
                const std::uint64_t alignment =
                    kind == RecordedBindingKind::Constant32 ? 4u
                                                            : 8u;
                const std::uint64_t mask = alignment - 1u;
                if (arena_cursor >
                    std::numeric_limits<std::uint64_t>::max() -
                        mask) {
                    return {
                        .error =
                            PrefillCommandGraphError::
                                ArgumentArenaFailed,
                        .stage =
                            PrefillCommandGraphStage::ArgumentArena,
                    };
                }
                arena_cursor =
                    (arena_cursor + mask) & ~mask;
                constant_offsets[command_index][index] =
                    arena_cursor;
                if (arena_cursor >
                    std::numeric_limits<std::uint64_t>::max() -
                        alignment) {
                    return {
                        .error =
                            PrefillCommandGraphError::
                                ArgumentArenaFailed,
                        .stage =
                            PrefillCommandGraphStage::ArgumentArena,
                    };
                }
                arena_cursor += alignment;
            }
        }
        if (arena_cursor == 0) {
            return {
                .error =
                    PrefillCommandGraphError::ArgumentArenaFailed,
                .stage = PrefillCommandGraphStage::ArgumentArena,
            };
        }
        auto arena = create_shared_buffer(device, arena_cursor);
        if (!arena) {
            return {
                .error =
                    PrefillCommandGraphError::ArgumentArenaFailed,
                .stage = PrefillCommandGraphStage::ArgumentArena,
            };
        }
        graph.argument_arena = std::move(*arena.buffer);
        graph.argument_arena_bytes = arena_cursor;
        std::memset(
            graph.argument_arena.contents(), 0,
            static_cast<std::size_t>(arena_cursor));

        auto indirect =
            create_compute_indirect_command_buffer(
                device,
                static_cast<std::uint32_t>(ordered.size()),
                kMaxBufferArgumentIndex + 1u);
        if (!indirect) {
            return {
                .error = PrefillCommandGraphError::
                    IndirectCommandBufferFailed,
                .stage =
                    PrefillCommandGraphStage::IndirectCommands,
                .command_error = indirect.error,
            };
        }
        graph.commands =
            std::move(*indirect.indirect_command_buffer);
        MetalCommandError command_error =
            reset_indirect_commands(
                graph.commands, 0,
                static_cast<std::uint32_t>(ordered.size()));
        auto check_command =
            [&command_error](MetalCommandError error) {
                if (command_error == MetalCommandError::None &&
                    error != MetalCommandError::None) {
                    command_error = error;
                }
            };
        auto* arena_bytes = static_cast<std::byte*>(
            graph.argument_arena.contents());
        for (std::uint32_t command_index = 0;
             command_error == MetalCommandError::None &&
             command_index < ordered.size(); ++command_index) {
            const OrderedRecordedCommand& ordered_command =
                ordered[command_index];
            const RecordedCommand& command =
                *ordered_command.command;
            check_command(set_indirect_compute_pipeline(
                graph.commands, command_index,
                *command.pipeline));
            for (std::uint32_t index = 0;
                 command_error == MetalCommandError::None &&
                 index <= kMaxBufferArgumentIndex; ++index) {
                const RecordedBinding& binding =
                    command.bindings[index];
                if (binding.kind ==
                    RecordedBindingKind::None) {
                    continue;
                }
                if (binding.kind ==
                    RecordedBindingKind::Buffer) {
                    check_command(set_indirect_buffer(
                        graph.commands, command_index,
                        *binding.buffer, binding.offset, index));
                    continue;
                }
                const std::uint64_t offset =
                    constant_offsets[command_index][index];
                const std::size_t bytes =
                    binding.kind ==
                            RecordedBindingKind::Constant32
                        ? sizeof(std::uint32_t)
                        : sizeof(std::uint64_t);
                std::memcpy(
                    arena_bytes + offset, &binding.value, bytes);
                check_command(set_indirect_buffer(
                    graph.commands, command_index,
                    graph.argument_arena, offset, index));
            }
            check_command(clear_indirect_barrier(
                graph.commands, command_index));
            check_command(dispatch_indirect_threadgroups(
                graph.commands, command_index, command.groups,
                command.threads));
        }
        if (command_error != MetalCommandError::None) {
            return {
                .error = PrefillCommandGraphError::
                    IndirectCommandBufferFailed,
                .stage =
                    PrefillCommandGraphStage::IndirectCommands,
                .command_error = command_error,
            };
        }
        graph.level_command_begins =
            std::move(level_command_begins);
        graph.node_level_boundary_offsets =
            std::move(node_level_boundary_offsets);
        graph.node_level_command_begins =
            std::move(node_level_command_begins);
        graph.command_count =
            static_cast<std::uint32_t>(ordered.size());
        graph.state = PrefillCommandGraphState::Ready;
        prefill.command_graph = std::move(graph);
        prefill.command_graph.resource_identities =
            prefill_resource_identities(prefill, decode, state);
        const PrefillCommandPlanError key_error =
            validate_prefill_command_plan_key(
                command_graph_key(prefill.command_graph));
        if (key_error != PrefillCommandPlanError::None) {
            prefill.command_graph = {};
            return {
                .error = PrefillCommandGraphError::PlanMismatch,
                .stage = PrefillCommandGraphStage::CacheKey,
                .plan_error = key_error,
            };
        }
        return {
            .error = PrefillCommandGraphError::None,
            .command_count =
                prefill.command_graph.command_count,
            .node_count = static_cast<std::uint32_t>(
                prefill.command_graph.nodes.size()),
            .diagonal_count = static_cast<std::uint32_t>(
                prefill.command_graph.diagonals.size()),
            .argument_arena_bytes =
                prefill.command_graph.argument_arena_bytes,
            .cache_hit = false,
        };
    } catch (const std::bad_alloc&) {
        return {
            .error =
                PrefillCommandGraphError::RecordingFailed,
        };
    }
}

PrefillCommandGraphResult prepare_prefill_command_graph(
    const MetalDevice& device, PrefillStep& prefill,
    DecodeStep& decode) {
    return prepare_prefill_command_graph(
        device, prefill, decode, decode.state);
}

PrefillCommandGraphResult encode_prefill_command_graph(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    MetalComputePass& pass) {
    if (!prefill.policy.command_graph) {
        return {.error = PrefillCommandGraphError::Disabled};
    }
    if (prefill.policy.command_graph_lane_events) {
        return {
            .error = PrefillCommandGraphError::PlanMismatch,
            .stage = PrefillCommandGraphStage::Ordering,
        };
    }
    if (prefill.command_graph.state ==
            PrefillCommandGraphState::Poisoned ||
        prefill.progress.state ==
            PrefillProgressState::Poisoned) {
        return {.error = PrefillCommandGraphError::Poisoned};
    }
    if (prefill.command_graph.state !=
            PrefillCommandGraphState::Ready ||
        prefill.progress.state != PrefillProgressState::Ready ||
        prefill.command_graph.decode_owner != &decode ||
        prefill.command_graph.state_owner != &state ||
        prefill.progress.owner != &decode ||
        prefill.progress.state_owner != &state ||
        prefill.command_graph.row_count !=
            prefill.progress.row_count ||
        prefill.command_graph.context_base !=
            prefill.progress.context_base) {
        return {
            .error =
                PrefillCommandGraphError::ProgressUnavailable,
        };
    }
    if (prefill.command_graph.chunk_rows.size() !=
            prefill.progress.chunk_count ||
        prefill.command_graph.state_phases.size() !=
            state.layers.size()) {
        return {.error = PrefillCommandGraphError::PlanMismatch};
    }
    for (std::uint32_t ordinal = 0;
         ordinal < prefill.progress.chunk_count; ++ordinal) {
        if (prefill.command_graph.chunk_rows[ordinal] !=
            chunk_at(
                prefill.policy, prefill.progress.context_base,
                prefill.progress.row_count,
                ordinal)
                .rows) {
            return {
                .error = PrefillCommandGraphError::PlanMismatch,
            };
        }
    }
    for (std::size_t layer = 0; layer < state.layers.size();
         ++layer) {
        if (prefill.command_graph.state_phases[layer] !=
            (state.layers[layer].swapped ? 1u : 0u)) {
            return {
                .error = PrefillCommandGraphError::PlanMismatch,
            };
        }
    }
    if (prefill.command_graph.model_package_identity !=
            execution_model_package_identity() ||
        prefill.command_graph.prepared_image_identity !=
            execution_prepared_image_identity(decode) ||
        prefill.command_graph.execution_policy_identity !=
            prefill_execution_policy_identity(prefill.policy) ||
        prefill.command_graph.state_slot_identity !=
            static_cast<std::uint64_t>(
                reinterpret_cast<std::uintptr_t>(&state)) ||
        prefill.command_graph.graph_schema_version !=
            kPrefillCommandGraphSchemaVersion) {
        return {.error = PrefillCommandGraphError::PlanMismatch};
    }

    std::size_t pipeline_index = 0;
    bool pipeline_match = true;
    visit_prefill_pipelines(
        prefill.pipelines,
        [&](const MetalComputePipeline& pipeline) {
            if (pipeline_index >=
                    prefill.command_graph.pipeline_identities.size() ||
                prefill.command_graph
                        .pipeline_identities[pipeline_index] !=
                    compute_pipeline_identity(pipeline)) {
                pipeline_match = false;
            }
            ++pipeline_index;
        });
    if (!pipeline_match ||
        pipeline_index !=
            prefill.command_graph.pipeline_identities.size()) {
        return {
            .error =
                PrefillCommandGraphError::PipelineUnavailable,
        };
    }
    if (prefill.command_graph.pipeline_identity !=
            execution_pipeline_identity(
                prefill.command_graph.pipeline_identities) ||
        validate_prefill_command_plan_key(
            command_graph_key(prefill.command_graph)) !=
            PrefillCommandPlanError::None) {
        return {.error = PrefillCommandGraphError::PlanMismatch};
    }

    std::size_t resource_index = 0;
    MetalCommandError command_error = MetalCommandError::None;
    visit_prefill_graph_buffers(
        prefill, decode, state,
        [&](const MetalBuffer& buffer, MetalResourceUsage usage) {
            if (command_error != MetalCommandError::None) {
                return;
            }
            if (resource_index >=
                    prefill.command_graph
                        .resource_identities.size() ||
                prefill.command_graph
                        .resource_identities[resource_index] !=
                    metal_buffer_identity(buffer)) {
                command_error =
                    MetalCommandError::InvalidBuffer;
                return;
            }
            command_error =
                use_buffer_resource(pass, buffer, usage);
            ++resource_index;
        });
    if (command_error != MetalCommandError::None ||
        resource_index !=
            prefill.command_graph.resource_identities.size()) {
        return {
            .error = PrefillCommandGraphError::ResourceMismatch,
            .command_error = command_error,
        };
    }
    const std::vector<std::uint32_t>& level_command_begins =
        prefill.command_graph.level_command_begins;
    if (level_command_begins.size() < 2u ||
        level_command_begins.front() != 0u ||
        level_command_begins.back() !=
            prefill.command_graph.command_count) {
        return {.error = PrefillCommandGraphError::PlanMismatch};
    }
    for (std::size_t level = 0;
         level + 1u < level_command_begins.size(); ++level) {
        const std::uint32_t level_begin =
            level_command_begins[level];
        const std::uint32_t level_end =
            level_command_begins[level + 1u];
        if (level_end <= level_begin) {
            return {
                .error = PrefillCommandGraphError::PlanMismatch,
            };
        }
        if (level != 0u) {
            command_error = memory_barrier(pass);
            if (command_error != MetalCommandError::None) {
                return {
                    .error = PrefillCommandGraphError::
                        IndirectCommandBufferFailed,
                    .command_error = command_error,
                };
            }
        }
        command_error = execute_indirect_commands(
            pass, prefill.command_graph.commands, level_begin,
            level_end - level_begin);
        if (command_error != MetalCommandError::None) {
            return {
                .error = PrefillCommandGraphError::
                    IndirectCommandBufferFailed,
                .command_error = command_error,
            };
        }
    }
    prefill.command_graph.state =
        PrefillCommandGraphState::Pending;
    prefill.progress.state = PrefillProgressState::GraphPending;
    prefill.progress.pending_unit_count =
        static_cast<std::uint32_t>(
            prefill.command_graph.nodes.size());
    return {
        .error = PrefillCommandGraphError::None,
        .command_count = prefill.command_graph.command_count,
        .node_count = static_cast<std::uint32_t>(
            prefill.command_graph.nodes.size()),
        .diagonal_count = static_cast<std::uint32_t>(
            prefill.command_graph.diagonals.size()),
        .argument_arena_bytes =
            prefill.command_graph.argument_arena_bytes,
        .cache_hit = true,
    };
}

PrefillCommandGraphResult encode_prefill_command_graph(
    PrefillStep& prefill, DecodeStep& decode,
    MetalComputePass& pass) {
    return encode_prefill_command_graph(
        prefill, decode, decode.state, pass);
}

PrefillCommandGraphResult encode_prefill_command_graph_lane_node(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state,
    MetalComputePass& pass, std::uint32_t layer_index,
    std::uint32_t scratch_lane) {
    if (!prefill.policy.command_graph ||
        !prefill.policy.command_graph_lane_events) {
        return {.error = PrefillCommandGraphError::Disabled};
    }
    if (prefill.command_graph.state ==
            PrefillCommandGraphState::Poisoned ||
        prefill.progress.state ==
            PrefillProgressState::Poisoned) {
        return {.error = PrefillCommandGraphError::Poisoned};
    }
    if (prefill.command_graph.state !=
            PrefillCommandGraphState::Ready ||
        prefill.progress.state != PrefillProgressState::Ready ||
        prefill.command_graph.decode_owner != &decode ||
        prefill.command_graph.state_owner != &state ||
        prefill.progress.owner != &decode ||
        prefill.progress.state_owner != &state ||
        prefill.progress.chunk_count == 0 ||
        prefill.command_graph.chunk_rows.size() !=
            prefill.progress.chunk_count ||
        prefill.command_graph.lane_event_nodes.size() !=
            prefill.command_graph.nodes.size() ||
        prefill.command_graph.node_level_boundary_offsets.size() !=
            prefill.command_graph.nodes.size() + 1u) {
        return {
            .error =
                PrefillCommandGraphError::ProgressUnavailable,
        };
    }
    const std::uint64_t layer_count =
        prefill.command_graph.nodes.size() /
        prefill.progress.chunk_count;
    if (layer_count != state.layers.size() ||
        layer_index >= layer_count ||
        scratch_lane >= prefill.progress.chunk_count) {
        return {
            .error = PrefillCommandGraphError::PlanMismatch,
            .stage = PrefillCommandGraphStage::Ordering,
        };
    }
    const std::uint64_t layer_major_index =
        std::uint64_t{layer_index} *
            prefill.progress.chunk_count +
        scratch_lane;
    const PrefillLaneEventNode& event_node =
        prefill.command_graph
            .lane_event_nodes[layer_major_index];
    if (event_node.layer_index != layer_index ||
        event_node.scratch_lane != scratch_lane ||
        event_node.node_index >=
            prefill.command_graph.nodes.size()) {
        return {
            .error = PrefillCommandGraphError::PlanMismatch,
            .stage = PrefillCommandGraphStage::Ordering,
        };
    }
    const PrefillCommandNode& node =
        prefill.command_graph.nodes[event_node.node_index];
    if (node.layer_major_index != layer_major_index ||
        node.layer_index != layer_index ||
        node.scratch_lane != scratch_lane) {
        return {
            .error = PrefillCommandGraphError::PlanMismatch,
            .stage = PrefillCommandGraphStage::Ordering,
        };
    }
    const std::uint32_t boundary_begin =
        prefill.command_graph
            .node_level_boundary_offsets[layer_major_index];
    const std::uint32_t boundary_end =
        prefill.command_graph
            .node_level_boundary_offsets[
                layer_major_index + 1u];
    if (boundary_end <= boundary_begin + 1u ||
        boundary_end >
            prefill.command_graph
                .node_level_command_begins.size()) {
        return {
            .error = PrefillCommandGraphError::PlanMismatch,
            .stage = PrefillCommandGraphStage::Ordering,
        };
    }

    std::size_t resource_index = 0;
    MetalCommandError command_error = MetalCommandError::None;
    visit_prefill_graph_buffers(
        prefill, decode, state,
        [&](const MetalBuffer& buffer, MetalResourceUsage usage) {
            if (command_error != MetalCommandError::None) {
                return;
            }
            if (resource_index >=
                    prefill.command_graph
                        .resource_identities.size() ||
                prefill.command_graph
                        .resource_identities[resource_index] !=
                    metal_buffer_identity(buffer)) {
                command_error = MetalCommandError::InvalidBuffer;
                return;
            }
            command_error =
                use_buffer_resource(pass, buffer, usage);
            ++resource_index;
        });
    if (command_error != MetalCommandError::None ||
        resource_index !=
            prefill.command_graph.resource_identities.size()) {
        return {
            .error = PrefillCommandGraphError::ResourceMismatch,
            .command_error = command_error,
        };
    }

    for (std::uint32_t boundary = boundary_begin;
         boundary + 1u < boundary_end; ++boundary) {
        const std::uint32_t command_begin =
            prefill.command_graph
                .node_level_command_begins[boundary];
        const std::uint32_t command_end =
            prefill.command_graph
                .node_level_command_begins[boundary + 1u];
        if (command_end <= command_begin ||
            command_end > prefill.command_graph.command_count) {
            return {
                .error = PrefillCommandGraphError::PlanMismatch,
                .stage = PrefillCommandGraphStage::Ordering,
            };
        }
        if (boundary != boundary_begin) {
            command_error = memory_barrier(pass);
            if (command_error != MetalCommandError::None) {
                return {
                    .error = PrefillCommandGraphError::
                        IndirectCommandBufferFailed,
                    .command_error = command_error,
                };
            }
        }
        command_error = execute_indirect_commands(
            pass, prefill.command_graph.commands,
            command_begin, command_end - command_begin);
        if (command_error != MetalCommandError::None) {
            return {
                .error = PrefillCommandGraphError::
                    IndirectCommandBufferFailed,
                .command_error = command_error,
            };
        }
    }
    return {
        .error = PrefillCommandGraphError::None,
        .command_count = prefill.command_graph.command_count,
        .node_count = static_cast<std::uint32_t>(
            prefill.command_graph.nodes.size()),
        .diagonal_count = static_cast<std::uint32_t>(
            prefill.command_graph.diagonals.size()),
        .argument_arena_bytes =
            prefill.command_graph.argument_arena_bytes,
        .cache_hit = true,
    };
}

PrefillCommandGraphResult encode_prefill_command_graph_lane_node(
    PrefillStep& prefill, DecodeStep& decode, MetalComputePass& pass,
    std::uint32_t layer_index, std::uint32_t scratch_lane) {
    return encode_prefill_command_graph_lane_node(
        prefill, decode, decode.state, pass, layer_index,
        scratch_lane);
}

PrefillCommandGraphResult mark_prefill_command_graph_lane_pending(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state) {
    if (!prefill.policy.command_graph ||
        !prefill.policy.command_graph_lane_events) {
        return {.error = PrefillCommandGraphError::Disabled};
    }
    if (prefill.command_graph.state !=
            PrefillCommandGraphState::Ready ||
        prefill.progress.state != PrefillProgressState::Ready ||
        prefill.command_graph.decode_owner != &decode ||
        prefill.command_graph.state_owner != &state ||
        prefill.progress.owner != &decode ||
        prefill.progress.state_owner != &state ||
        prefill.progress.chunk_count == 0 ||
        prefill.command_graph_lane_queues.size() <
            prefill.progress.chunk_count ||
        prefill.command_graph_lane_events.size() + 1u <
            prefill.progress.chunk_count) {
        return {
            .error =
                PrefillCommandGraphError::ProgressUnavailable,
        };
    }
    std::vector<PrefillLaneEventNode> dynamic_nodes(
        prefill.command_graph.nodes.size());
    const PrefillLaneEventPlanResult plan =
        build_prefill_lane_event_plan(
            {
                .layer_count = static_cast<std::uint32_t>(
                    state.layers.size()),
                .scratch_lane_count =
                    prefill.progress.chunk_count,
                .event_value_base =
                    prefill.command_graph_event_value_base,
                .nodes = prefill.command_graph.nodes,
            },
            dynamic_nodes);
    if (!plan ||
        plan.plan.node_count !=
            prefill.command_graph.nodes.size()) {
        return {
            .error = PrefillCommandGraphError::PlanMismatch,
            .stage = PrefillCommandGraphStage::Ordering,
            .plan_error = plan.error,
        };
    }
    prefill.command_graph.state =
        PrefillCommandGraphState::Pending;
    prefill.progress.state = PrefillProgressState::GraphPending;
    prefill.progress.pending_unit_count =
        static_cast<std::uint32_t>(
            prefill.command_graph.nodes.size());
    return {
        .error = PrefillCommandGraphError::None,
        .command_count = prefill.command_graph.command_count,
        .node_count = static_cast<std::uint32_t>(
            prefill.command_graph.nodes.size()),
        .diagonal_count = static_cast<std::uint32_t>(
            prefill.command_graph.diagonals.size()),
        .argument_arena_bytes =
            prefill.command_graph.argument_arena_bytes,
        .cache_hit = true,
    };
}

PrefillCommandGraphResult mark_prefill_command_graph_lane_pending(
    PrefillStep& prefill, DecodeStep& decode) {
    return mark_prefill_command_graph_lane_pending(
        prefill, decode, decode.state);
}

PrefillProgressResult commit_prefill_command_graph(
    PrefillStep& prefill, DecodeStep& decode, DecodeStateSlot& state) {
    if (!prefill.policy.command_graph) {
        return progress_failure(
            prefill, PrefillProgressError::GraphNotPending);
    }
    if (prefill.command_graph.state ==
            PrefillCommandGraphState::Poisoned ||
        prefill.progress.state ==
            PrefillProgressState::Poisoned) {
        return progress_failure(
            prefill, PrefillProgressError::Poisoned);
    }
    if (prefill.command_graph.state !=
            PrefillCommandGraphState::Pending ||
        prefill.progress.state !=
            PrefillProgressState::GraphPending) {
        return progress_failure(
            prefill, PrefillProgressError::GraphNotPending);
    }
    if (prefill.command_graph.decode_owner != &decode ||
        prefill.command_graph.state_owner != &state ||
        prefill.progress.owner != &decode ||
        prefill.progress.state_owner != &state ||
        prefill.progress.pending_unit_count !=
            prefill.command_graph.nodes.size()) {
        prefill.command_graph.state =
            PrefillCommandGraphState::Poisoned;
        prefill.progress.state = PrefillProgressState::Poisoned;
        return progress_failure(
            prefill, PrefillProgressError::Invalid);
    }
    for (std::uint32_t node = 0;
         node < prefill.command_graph.nodes.size(); ++node) {
        const QuantizedGemmDeviceTaskStatus status =
            device_task_status(
                prefill.native_routed_up_status, node);
        if (status != QuantizedGemmDeviceTaskStatus::Ready) {
            prefill.command_graph.state =
                PrefillCommandGraphState::Poisoned;
            prefill.progress.state =
                PrefillProgressState::Poisoned;
            PrefillProgressResult result = progress_failure(
                prefill,
                PrefillProgressError::DeviceTaskNotReady);
            result.failed_unit_offset = node;
            result.routed_up_status = status;
            return result;
        }
    }

    advance_prefill_state(
        decode, state, prefill.progress.chunk_count);
    const std::uint32_t committed_units =
        prefill.progress.pending_unit_count;
    prefill.progress.current_layer =
        static_cast<std::uint32_t>(decode.schedule.size());
    prefill.progress.current_chunk = 0;
    prefill.progress.pending_unit_count = 0;
    prefill.progress.state = PrefillProgressState::Complete;
    prefill.command_graph.state =
        PrefillCommandGraphState::Ready;
    return {
        .error = PrefillProgressError::None,
        .state = PrefillProgressState::Complete,
        .next_context = prefill.progress.next_context,
        .chunk_count = prefill.progress.chunk_count,
        .layer_index = 0,
        .chunk_ordinal = 0,
        .unit_count = committed_units,
    };
}

PrefillProgressResult commit_prefill_command_graph(
    PrefillStep& prefill, DecodeStep& decode) {
    return commit_prefill_command_graph(
        prefill, decode, decode.state);
}

PrefillProgressError release_prefill_progress(PrefillStep& prefill, const DecodeStep& decode,
                                              DecodeStateSlot& state) noexcept {
    switch (prefill.progress.state) {
    case PrefillProgressState::Idle:
        return PrefillProgressError::Unavailable;
    case PrefillProgressState::UnitPending:
        return PrefillProgressError::UnitPending;
    case PrefillProgressState::BatchPending:
        return PrefillProgressError::BatchPending;
    case PrefillProgressState::InflightEncoding:
        return PrefillProgressError::InflightEncoding;
    case PrefillProgressState::InflightPending:
        return PrefillProgressError::InflightPending;
    case PrefillProgressState::GraphPending:
        return PrefillProgressError::GraphPending;
    case PrefillProgressState::Complete:
        return PrefillProgressError::Complete;
    case PrefillProgressState::Poisoned:
        return PrefillProgressError::Poisoned;
    case PrefillProgressState::Ready:
        break;
    }
    if (!valid_active_progress(prefill, decode, state)) {
        return PrefillProgressError::Invalid;
    }
    prefill.progress = {};
    return PrefillProgressError::None;
}

PrefillProgressError poison_prefill(PrefillStep& prefill) noexcept {
    switch (prefill.progress.state) {
    case PrefillProgressState::Idle:
        return PrefillProgressError::Unavailable;
    case PrefillProgressState::Complete:
        return PrefillProgressError::Complete;
    case PrefillProgressState::Poisoned:
        return PrefillProgressError::Poisoned;
    case PrefillProgressState::Ready:
    case PrefillProgressState::UnitPending:
    case PrefillProgressState::BatchPending:
    case PrefillProgressState::InflightEncoding:
    case PrefillProgressState::InflightPending:
    case PrefillProgressState::GraphPending:
        prefill.progress.state = PrefillProgressState::Poisoned;
        prefill.command_graph.state =
            PrefillCommandGraphState::Poisoned;
        return PrefillProgressError::None;
    }
    return PrefillProgressError::Invalid;
}

} // namespace tatara::runtime
