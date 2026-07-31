#include "decode_harness.h"

#include "tatara/backend/metal/pipeline.h"
#include "tatara/generated/kernel_library.h"
#include "tatara/generated/model_plan.h"
#include "tatara/model/image_population.h"
#include "tatara/model/prepared_checkpoint.h"
#include "tatara/model/source_shards.h"
#include "tatara/runtime/decode_bindings.h"
#include "tatara/runtime/decode_geometry.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>

namespace tatara::tools {
namespace {

using namespace backend::metal;
using model::qwen36::LayerKind;
using runtime::DecodeLayerState;
using runtime::DecodeStep;

constexpr char kStateMagic[8] = {'Q', '3', '6', 'B', 'R', 'S', '0', '1'};
constexpr std::uint64_t kStateHeads = 2;
constexpr std::uint64_t kStateHeadDimensionBytes = 256 * 2;

// The sealed decode schedule's twenty-one kernels, in the order the executor
// binds them. The slot list below must stay parallel; the static_assert holds
// that.
constexpr const char* kKernelNames[] = {
    "embed_row_q4",
    "rms_only",
    "gdn_project",
    "gdn_prepare",
    "gdn_recurrence",
    "gdn_gate_norm",
    "gdn_outproj",
    "attn_project",
    "attn_qk_rope",
    "attention_decode",
    "attention_decode_scores_gqa4",
    "attention_decode_scores_values_gqa8",
    "attention_decode_vector_2pass_part",
    "attention_decode_vector_2pass_combine",
    "attention_decode_values_gqa8",
    "attention_decode_combine",
    "residual_rms",
    "router_q8",
    "router_select",
    "grouped_upgate",
    "grouped_down_res",
    "lmhead_q4",
    "logits_argmax_stage1",
    "logits_argmax_stage2",
};
constexpr std::size_t kAttentionScoresPipelineIndex = 10;
constexpr std::size_t kAttentionFusedPipelineIndex = 11;
constexpr std::size_t kAttentionVectorPartPipelineIndex = 12;
constexpr std::size_t kAttentionVectorCombinePipelineIndex = 13;
constexpr std::size_t kAttentionValuesPipelineIndex = 14;
static_assert(
    std::string_view{kKernelNames[kAttentionScoresPipelineIndex]} ==
    "attention_decode_scores_gqa4");
static_assert(
    std::string_view{kKernelNames[kAttentionFusedPipelineIndex]} ==
    "attention_decode_scores_values_gqa8");
static_assert(
    std::string_view{kKernelNames[kAttentionVectorPartPipelineIndex]} ==
    "attention_decode_vector_2pass_part");
static_assert(
    std::string_view{kKernelNames[kAttentionVectorCombinePipelineIndex]} ==
    "attention_decode_vector_2pass_combine");
static_assert(
    std::string_view{kKernelNames[kAttentionValuesPipelineIndex]} ==
    "attention_decode_values_gqa8");

} // namespace

std::vector<std::byte> read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        return {};
    }
    const std::streamsize size = stream.tellg();
    stream.seekg(0);
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));
    stream.read(reinterpret_cast<char*>(bytes.data()), size);
    return stream ? bytes : std::vector<std::byte>{};
}

DecodeImagePlan inspect_decode_image(const char* record_path) {
    DecodeImagePlan result;
    const auto record_bytes = read_file(record_path);
    if (record_bytes.empty()) {
        result.exit_code = 3;
        return result;
    }
    const auto parsed = model::parse_prepared_checkpoint(record_bytes);
    if (!parsed || !parsed.checkpoint) {
        result.exit_code = 4;
        return result;
    }
    const model::PreparedCheckpointExpectation expectation = {
        .package_id = model::generated::kModelPackageId,
        .package_sha256 = model::generated::kModelPackageSha256,
        .artifact = model::generated::kArtifactIdentity,
    };
    if (model::validate_prepared_checkpoint_identity(
            *parsed.checkpoint, expectation) !=
        model::PreparedCheckpointIdentityError::None) {
        result.exit_code = 5;
        return result;
    }
    const auto layout = model::plan_image_layout(
        parsed.checkpoint->tensors(), model::kTensorAlignmentBytes);
    if (!layout || !layout.layout) {
        result.exit_code = 6;
        return result;
    }
    result.prepared_record_bytes = record_bytes.size();
    result.image_bytes = layout.layout->total_bytes;
    return result;
}

DecodeHarness boot_decode(const char* record_path,
                          const char* artifact_root) {
    return boot_decode(
        record_path, artifact_root,
        model::qwen36::generated::kModelPlan.initial_serving_capacity,
        DecodeAttentionScoreKernel::Adaptive);
}

DecodeHarness boot_decode(const char* record_path, const char* artifact_root,
                          std::uint32_t capacity) {
    return boot_decode(record_path, artifact_root, capacity,
                       DecodeAttentionScoreKernel::Adaptive);
}

DecodeHarness boot_decode(const char* record_path, const char* artifact_root,
                          std::uint32_t capacity,
                          DecodeAttentionScoreKernel score_kernel) {
    return boot_decode(record_path, artifact_root, capacity, score_kernel,
                       DecodeAttentionValueKernel::Gqa8T1024);
}

DecodeHarness boot_decode(const char* record_path, const char* artifact_root,
                          std::uint32_t capacity,
                          DecodeAttentionScoreKernel score_kernel,
                          DecodeAttentionValueKernel value_kernel) {
    DecodeHarness harness;
    const auto record_bytes = read_file(record_path);
    if (record_bytes.empty()) {
        std::cerr << "prepared checkpoint is empty or unreadable\n";
        harness.exit_code = 3;
        return harness;
    }
    const auto parsed = model::parse_prepared_checkpoint(record_bytes);
    if (!parsed || !parsed.checkpoint) {
        std::cerr << "prepared checkpoint parse failed\n";
        harness.exit_code = 4;
        return harness;
    }
    const model::PreparedCheckpointExpectation expectation = {
        .package_id = model::generated::kModelPackageId,
        .package_sha256 = model::generated::kModelPackageSha256,
        .artifact = model::generated::kArtifactIdentity,
    };
    if (model::validate_prepared_checkpoint_identity(*parsed.checkpoint, expectation) !=
        model::PreparedCheckpointIdentityError::None) {
        std::cerr << "prepared checkpoint identity failed\n";
        harness.exit_code = 5;
        return harness;
    }
    const auto layout =
        model::plan_image_layout(parsed.checkpoint->tensors(), model::kTensorAlignmentBytes);
    if (!layout || !layout.layout) {
        std::cerr << "image layout planning failed\n";
        harness.exit_code = 6;
        return harness;
    }
    auto shards = model::open_source_shards(artifact_root, parsed.checkpoint->shards());
    if (!shards || !shards.shard_set) {
        std::cerr << "source shard open failed: path=" << shards.path << '\n';
        harness.exit_code = 7;
        return harness;
    }

    auto device = create_system_device();
    if (!device) {
        std::cerr << "system Metal device creation failed\n";
        harness.exit_code = 9;
        return harness;
    }
    auto queue = create_command_queue(*device.device);
    if (!queue) {
        std::cerr << "command queue creation failed\n";
        harness.exit_code = 10;
        return harness;
    }
    auto library = create_library_with_source(*device.device, generated::kernel_library_source());
    if (!library) {
        std::cerr << "kernel library compilation failed:\n" << library.failure_description << '\n';
        harness.exit_code = 11;
        return harness;
    }

    runtime::DecodePipelines pipelines;
    MetalComputePipeline* slots[] = {
        &pipelines.embed,
        &pipelines.rms,
        &pipelines.gdn_project,
        &pipelines.gdn_prepare,
        &pipelines.gdn_recurrence,
        &pipelines.gdn_gate_norm,
        &pipelines.out_projection,
        &pipelines.attn_project,
        &pipelines.attn_qk_rope,
        &pipelines.attention_decode,
        &pipelines.attention_scores,
        &pipelines.attention_scores_values_fused,
        &pipelines.attention_vector_part,
        &pipelines.attention_vector_combine,
        &pipelines.attention_values,
        &pipelines.attention_combine,
        &pipelines.residual_rms,
        &pipelines.router,
        &pipelines.router_select,
        &pipelines.grouped_upgate,
        &pipelines.grouped_down_res,
        &pipelines.lmhead,
        &pipelines.argmax_stage1,
        &pipelines.argmax_stage2,
    };
    static_assert(std::size(kKernelNames) == std::size(slots));
    for (std::size_t slot = 0; slot < std::size(kKernelNames); ++slot) {
        const char* kernel_name = kKernelNames[slot];
        if (slot == kAttentionScoresPipelineIndex) {
            if (score_kernel == DecodeAttentionScoreKernel::Gqa4SimdReduce) {
                kernel_name = "attention_decode_scores_gqa4_simdreduce";
            } else if (score_kernel == DecodeAttentionScoreKernel::Gqa8) {
                kernel_name = "attention_decode_scores_gqa8";
            }
        } else if (
            slot == kAttentionValuesPipelineIndex &&
            value_kernel == DecodeAttentionValueKernel::Gqa8T512) {
            kernel_name = "attention_decode_values_gqa8_t512";
        }
        auto function = create_function(*library.library, kernel_name);
        if (!function) {
            std::cerr << "function lookup failed: " << kernel_name << '\n';
            harness.exit_code = 12;
            return harness;
        }
        auto pipeline = create_compute_pipeline(*device.device, *function.function);
        if (!pipeline) {
            std::cerr << "pipeline creation failed: " << kernel_name << '\n';
            harness.exit_code = 13;
            return harness;
        }
        *slots[slot] = std::move(*pipeline.pipeline);
    }
    if (score_kernel == DecodeAttentionScoreKernel::Gqa8FusedPart) {
        pipelines.attention_split_policy =
            runtime::DecodeAttentionSplitPolicy::FusedGqa8ScoreValue;
    } else if (
        score_kernel ==
        DecodeAttentionScoreKernel::IndependentHeadVector2Pass) {
        pipelines.attention_split_policy =
            runtime::DecodeAttentionSplitPolicy::
                IndependentHeadVector2Pass;
    } else if (score_kernel == DecodeAttentionScoreKernel::AdaptiveA23) {
        pipelines.attention_split_policy =
            runtime::DecodeAttentionSplitPolicy::AdaptiveGqa8ScoreValue;
        pipelines.fused_score_value_minimum_context =
            kQualifiedFusedScoreValueMinimumContext;
    } else if (score_kernel == DecodeAttentionScoreKernel::Adaptive) {
        pipelines.attention_split_policy =
            runtime::DecodeAttentionSplitPolicy::AdaptiveVector2Pass;
        pipelines.fused_score_value_minimum_context =
            kQualifiedFusedScoreValueMinimumContext;
        pipelines.vector_minimum_context =
            kQualifiedVectorMinimumContext;
    }

    const auto allocation_start = std::chrono::steady_clock::now();
    auto image = create_shared_buffer(*device.device, layout.layout->total_bytes);
    if (!image) {
        std::cerr << "model image allocation failed\n";
        harness.exit_code = 14;
        return harness;
    }
    auto owned_image = std::make_unique<MetalBuffer>(std::move(*image.buffer));
    const std::span<std::byte> destination(static_cast<std::byte*>(owned_image->contents()),
                                           static_cast<std::size_t>(layout.layout->total_bytes));
    const auto populated = model::populate_model_image(*parsed.checkpoint, *layout.layout,
                                                       shards.shard_set->shards(), destination);
    if (!populated) {
        std::cerr << "image population failed: shard=" << populated.shard_index << '\n';
        harness.exit_code = 15;
        return harness;
    }
    harness.load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - allocation_start).count();

    const auto& plan = model::qwen36::generated::kModelPlan;
    if (capacity < 2 || capacity > plan.tokenizer.maximum_context) {
        std::cerr << "decode capacity is outside the generated model bound\n";
        harness.exit_code = 17;
        return harness;
    }
    const std::span<const LayerKind> schedule(plan.layers.data(), plan.layers.size());
    auto bindings = runtime::resolve_decode_bindings(schedule, parsed.checkpoint->tensors());
    if (!bindings) {
        std::cerr << "binding resolution failed: " << bindings.missing_name << '\n';
        harness.exit_code = 16;
        return harness;
    }
    harness.capacity = capacity;
    const auto geometry = runtime::make_decode_geometry(plan, harness.capacity);
    auto step_result = runtime::create_decode_step(
        *device.device, geometry, harness.capacity, schedule, std::move(*bindings.bindings),
        *owned_image, layout.layout->tensor_offsets, std::move(pipelines));
    if (!step_result) {
        std::cerr << "decode step construction failed\n";
        harness.exit_code = 17;
        return harness;
    }

    harness.device = std::move(*device.device);
    harness.queue = std::move(*queue.command_queue);
    harness.library = std::move(*library.library);
    harness.image = std::move(owned_image);
    harness.step = std::move(*step_result.step);
    return harness;
}

bool write_state_record(const char* path, const DecodeStep& step,
                        std::span<const LayerKind> schedule, std::uint32_t capacity,
                        std::uint32_t positions) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        std::cerr << "state record: cannot open " << path << '\n';
        return false;
    }
    stream.write(kStateMagic, 8);
    stream.write(reinterpret_cast<const char*>(&positions), 4);

    const std::uint64_t head_bytes = std::uint64_t{positions} * kStateHeadDimensionBytes;
    const std::uint64_t cache_head_stride = std::uint64_t{capacity} * kStateHeadDimensionBytes;
    for (std::size_t layer = 0; layer < schedule.size(); ++layer) {
        const DecodeLayerState& state = step.state.layers[layer];
        if (schedule[layer] == LayerKind::GatedDelta) {
            const bool live_out = gated_delta_live_is_out(state.swapped);
            const MetalBuffer& conv = live_out ? state.first_out : state.first;
            const MetalBuffer& recurrent = live_out ? state.second_out : state.second;
            stream.write(static_cast<const char*>(conv.contents()),
                         static_cast<std::streamsize>(step.geometry.gdn_conv_state_bytes));
            stream.write(static_cast<const char*>(recurrent.contents()),
                         static_cast<std::streamsize>(step.geometry.gdn_recurrent_state_bytes));
        } else {
            for (const MetalBuffer* plane : {&state.first, &state.second}) {
                for (std::uint64_t head = 0; head < kStateHeads; ++head) {
                    stream.write(static_cast<const char*>(plane->contents()) +
                                     head * cache_head_stride,
                                 static_cast<std::streamsize>(head_bytes));
                }
            }
        }
    }
    stream.flush();
    if (!stream) {
        std::cerr << "state record: write failed\n";
        return false;
    }
    return true;
}

// Whether the final step advanced the state: the live plane must differ from
// its stale partner, which holds N-1. Compares the conv window, which every
// gated-delta step writes.
bool gated_delta_advanced(const DecodeStep& step, std::span<const LayerKind> schedule,
                          std::size_t& stalled_layer) {
    for (std::size_t layer = 0; layer < schedule.size(); ++layer) {
        if (schedule[layer] != LayerKind::GatedDelta) {
            continue;
        }
        const DecodeLayerState& state = step.state.layers[layer];
        const bool live_out = gated_delta_live_is_out(state.swapped);
        const void* live = live_out ? state.first_out.contents() : state.first.contents();
        const void* stale = live_out ? state.first.contents() : state.first_out.contents();
        if (std::memcmp(live, stale, step.geometry.gdn_conv_state_bytes) == 0) {
            stalled_layer = layer;
            return false;
        }
    }
    return true;
}

constexpr char kStateRecordMagic[8] = {'Q', '3', '6', 'B', 'R', 'S', '0', '1'};

bool load_state_record(std::span<const std::byte> record, DecodeStep& step,
                       std::span<const LayerKind> schedule, std::uint32_t capacity) {
    if (record.size() < 12 || std::memcmp(record.data(), kStateRecordMagic, 8) != 0) {
        std::cerr << "state record magic mismatch\n";
        return false;
    }
    std::uint32_t count = 0;
    std::memcpy(&count, record.data() + 8, 4);
    if (count == 0 || count > capacity) {
        std::cerr << "state record declares " << count << " positions, capacity " << capacity
                  << '\n';
        return false;
    }
    std::size_t cursor = 12;
    const auto take = [&](std::size_t bytes) -> const std::byte* {
        if (cursor + bytes > record.size()) {
            return nullptr;
        }
        const std::byte* view = record.data() + cursor;
        cursor += bytes;
        return view;
    };
    const std::uint64_t head_bytes = std::uint64_t{count} * kStateHeadDimensionBytes;
    const std::uint64_t cache_head_stride = std::uint64_t{capacity} * kStateHeadDimensionBytes;
    for (std::size_t layer = 0; layer < schedule.size(); ++layer) {
        DecodeLayerState& state = step.state.layers[layer];
        if (schedule[layer] == LayerKind::GatedDelta) {
            const std::byte* conv = take(step.geometry.gdn_conv_state_bytes);
            const std::byte* recurrent = take(step.geometry.gdn_recurrent_state_bytes);
            if (conv == nullptr || recurrent == nullptr) {
                std::cerr << "state record truncated at layer " << layer << '\n';
                return false;
            }
            std::memcpy(state.first.contents(), conv, step.geometry.gdn_conv_state_bytes);
            std::memcpy(state.second.contents(), recurrent,
                        step.geometry.gdn_recurrent_state_bytes);
        } else {
            const std::byte* keys = take(kStateHeads * head_bytes);
            const std::byte* values = take(kStateHeads * head_bytes);
            if (keys == nullptr || values == nullptr) {
                std::cerr << "state record truncated at layer " << layer << '\n';
                return false;
            }
            for (std::uint64_t head = 0; head < kStateHeads; ++head) {
                std::memcpy(static_cast<std::byte*>(state.first.contents()) +
                                head * cache_head_stride,
                            keys + head * head_bytes, head_bytes);
                std::memcpy(static_cast<std::byte*>(state.second.contents()) +
                                head * cache_head_stride,
                            values + head * head_bytes, head_bytes);
            }
        }
    }
    if (cursor != record.size()) {
        std::cerr << "state record has " << record.size() - cursor << " unconsumed bytes\n";
        return false;
    }
    return true;
}

} // namespace tatara::tools
