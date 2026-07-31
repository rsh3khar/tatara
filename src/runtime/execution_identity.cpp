#include "tatara/runtime/execution_identity.h"

#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/generated/kernel_library.h"
#include "tatara/generated/model_plan.h"
#include "tatara/runtime/decode_step.h"
#include "tatara/runtime/prefill_command_plan.h"
#include "tatara/runtime/prefill_step.h"
#include "tatara/version.h"

#include <array>

namespace tatara::runtime {
namespace {

using backend::metal::compute_pipeline_identity;
using backend::metal::metal_buffer_identity;

void append_artifact(ExecutionIdentityEncoder& encoder,
                     const model::ArtifactIdentity& artifact) noexcept {
    encoder.append_text(artifact.id);
    encoder.append_text(artifact.model_type);
    encoder.append_text(artifact.format);
    encoder.append_text(artifact.source_repository);
    encoder.append_text(artifact.source_revision);
    encoder.append_text(artifact.manifest_sha256);
    encoder.append_u32(artifact.tensor_count);
    encoder.append_u64(artifact.tensor_bytes);
    encoder.append_u32(artifact.file_count);
    encoder.append_u32(artifact.weight_file_count);
}

void append_tokenizer(ExecutionIdentityEncoder& encoder,
                      const model::qwen36::TokenizerSpec& tokenizer) noexcept {
    encoder.append_u8(static_cast<std::uint8_t>(tokenizer.kind));
    encoder.append_u8(static_cast<std::uint8_t>(tokenizer.normalization));
    encoder.append_u8(static_cast<std::uint8_t>(tokenizer.pretokenizer));
    encoder.append_u8(static_cast<std::uint8_t>(tokenizer.decoder));
    encoder.append_u8(static_cast<std::uint8_t>(tokenizer.template_kind));
    encoder.append_text(tokenizer.data_path);
    encoder.append_text(tokenizer.data_sha256);
    encoder.append_u64(tokenizer.data_size_bytes);
    encoder.append_text(tokenizer.config_path);
    encoder.append_text(tokenizer.config_sha256);
    encoder.append_u64(tokenizer.config_size_bytes);
    encoder.append_text(tokenizer.template_path);
    encoder.append_text(tokenizer.template_sha256);
    encoder.append_u64(tokenizer.template_size_bytes);
    encoder.append_text(tokenizer.split_pattern);
    encoder.append_u32(tokenizer.vocabulary);
    encoder.append_u32(tokenizer.populated_vocabulary);
    encoder.append_u32(tokenizer.maximum_context);
    encoder.append_u32(tokenizer.end_of_text_id);
    encoder.append_u32(tokenizer.message_start_id);
    encoder.append_u32(tokenizer.message_end_id);
    encoder.append_u32(tokenizer.thinking_start_id);
    encoder.append_u32(tokenizer.thinking_end_id);
    encoder.append_u32(tokenizer.padding_id);
    encoder.append_u32(tokenizer.stop_token_count);
    for (std::uint32_t stop : tokenizer.stop_token_ids) {
        encoder.append_u32(stop);
    }
    encoder.append_bool(tokenizer.default_thinking);
}

template <std::size_t Size>
PrefillCommandIdentity pipeline_slots(
    std::string_view domain,
    const std::array<const backend::metal::MetalComputePipeline*, Size>& pipelines) noexcept {
    ExecutionIdentityEncoder encoder(domain);
    encoder.append_u32(static_cast<std::uint32_t>(Size));
    for (const backend::metal::MetalComputePipeline* pipeline : pipelines) {
        encoder.append_u64(pipeline == nullptr ? 0 : compute_pipeline_identity(*pipeline));
    }
    return encoder.finish();
}

} // namespace

ExecutionIdentityEncoder::ExecutionIdentityEncoder(std::string_view domain) noexcept {
    append_u32(kExecutionIdentitySchemaVersion);
    append_text(domain);
}

void ExecutionIdentityEncoder::append_u8(std::uint8_t value) noexcept {
    const std::array<std::byte, 1> bytes{static_cast<std::byte>(value)};
    hash_.update(bytes);
}

void ExecutionIdentityEncoder::append_u32(std::uint32_t value) noexcept {
    std::array<std::byte, 4> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::byte>((value >> (index * 8U)) & 0xFFU);
    }
    hash_.update(bytes);
}

void ExecutionIdentityEncoder::append_u64(std::uint64_t value) noexcept {
    std::array<std::byte, 8> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = static_cast<std::byte>((value >> (index * 8U)) & 0xFFU);
    }
    hash_.update(bytes);
}

void ExecutionIdentityEncoder::append_bool(bool value) noexcept {
    append_u8(value ? 1U : 0U);
}

void ExecutionIdentityEncoder::append_text(std::string_view value) noexcept {
    append_u64(value.size());
    hash_.update(std::as_bytes(std::span(value)));
}

void ExecutionIdentityEncoder::append_bytes(std::span<const std::byte> value) noexcept {
    append_u64(value.size());
    hash_.update(value);
}

void ExecutionIdentityEncoder::append_identity(const PrefillCommandIdentity& value) noexcept {
    append_bytes(std::as_bytes(std::span(value)));
}

PrefillCommandIdentity ExecutionIdentityEncoder::finish() noexcept {
    return hash_.finish();
}

PrefillCommandIdentity execution_model_package_identity() noexcept {
    const auto& plan = model::qwen36::generated::kModelPlan;
    ExecutionIdentityEncoder encoder("tatara.execution.model-package");
    encoder.append_text(plan.id);
    encoder.append_text(plan.family);
    encoder.append_text(plan.package_sha256);
    append_artifact(encoder, plan.artifact);
    return encoder.finish();
}

PrefillCommandIdentity execution_model_plan_identity() noexcept {
    const auto& plan = model::qwen36::generated::kModelPlan;
    ExecutionIdentityEncoder encoder("tatara.execution.generated-plan");
    encoder.append_identity(execution_model_package_identity());
    encoder.append_u32(plan.dimensions.hidden);
    encoder.append_u32(plan.dimensions.vocabulary);
    encoder.append_u32(plan.attention.query_heads);
    encoder.append_u32(plan.attention.key_value_heads);
    encoder.append_u32(plan.attention.head_dimension);
    encoder.append_u32(plan.gated_delta.recurrent_heads);
    encoder.append_u32(plan.gated_delta.state_dimension);
    encoder.append_u32(plan.mixture_of_experts.experts);
    encoder.append_u32(plan.mixture_of_experts.active_experts);
    encoder.append_u32(plan.mixture_of_experts.expert_dimension);
    encoder.append_u8(static_cast<std::uint8_t>(plan.weights.format));
    encoder.append_u32(plan.weights.group_size);
    encoder.append_identity(execution_tokenizer_identity());
    encoder.append_u32(plan.initial_serving_capacity);
    encoder.append_u32(static_cast<std::uint32_t>(plan.layers.size()));
    for (model::qwen36::LayerKind layer : plan.layers) {
        encoder.append_u8(static_cast<std::uint8_t>(layer));
    }
    return encoder.finish();
}

PrefillCommandIdentity execution_tokenizer_identity() noexcept {
    ExecutionIdentityEncoder encoder("tatara.execution.tokenizer-template");
    append_tokenizer(encoder, model::qwen36::generated::kModelPlan.tokenizer);
    return encoder.finish();
}

PrefillCommandIdentity execution_host_identity() noexcept {
    ExecutionIdentityEncoder encoder("tatara.execution.host");
    encoder.append_text(kVersion);
    encoder.append_u32(kDecodeExecutionSchemaVersion);
    encoder.append_u32(kDecodeBindingSchemaVersion);
    encoder.append_u32(kPrefillCommandGraphSchemaVersion);
    return encoder.finish();
}

PrefillCommandIdentity
execution_prepared_record_identity(std::span<const std::byte> record) noexcept {
    ExecutionIdentityEncoder encoder("tatara.execution.prepared-record");
    encoder.append_bytes(record);
    return encoder.finish();
}

PrefillCommandIdentity execution_prepared_image_identity(const DecodeStep& decode) noexcept {
    ExecutionIdentityEncoder encoder("tatara.execution.prepared-image");
    encoder.append_text(model::generated::kArtifactIdentity.manifest_sha256);
    encoder.append_u64(decode.image == nullptr ? 0 : metal_buffer_identity(*decode.image));
    encoder.append_u64(decode.image == nullptr ? 0 : decode.image->size_bytes());
    encoder.append_u64(decode.tensor_offsets.size());
    for (std::uint64_t offset : decode.tensor_offsets) {
        encoder.append_u64(offset);
    }
    return encoder.finish();
}

PrefillCommandIdentity
execution_decode_pipeline_identity(const DecodePipelines& pipelines) noexcept {
    const std::array<const backend::metal::MetalComputePipeline*, 24> slots{
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
    ExecutionIdentityEncoder encoder("tatara.execution.decode-pipelines");
    encoder.append_u32(static_cast<std::uint32_t>(slots.size()));
    for (const backend::metal::MetalComputePipeline* pipeline : slots) {
        encoder.append_u64(
            pipeline == nullptr ? 0 : compute_pipeline_identity(*pipeline));
    }
    encoder.append_u8(
        static_cast<std::uint8_t>(pipelines.attention_split_policy));
    encoder.append_u32(pipelines.fused_score_value_minimum_context);
    encoder.append_u32(pipelines.vector_minimum_context);
    return encoder.finish();
}

PrefillCommandIdentity
execution_prefill_pipeline_identity(const PrefillPipelines& pipelines) noexcept {
    const std::array<const backend::metal::MetalComputePipeline*, 35> slots{
        &pipelines.embed,
        &pipelines.rms,
        &pipelines.residual,
        &pipelines.gdn_project,
        &pipelines.gdn_conv,
        &pipelines.gdn_gates,
        &pipelines.gdn_recurrence_step,
        &pipelines.gdn_recurrence_block,
        &pipelines.gdn_recurrence_gates,
        &pipelines.gdn_gate_norm,
        &pipelines.attn_project,
        &pipelines.attn_qk_rope,
        &pipelines.attention_partial,
        &pipelines.attention_combine,
        &pipelines.attention_staged_scores,
        &pipelines.attention_staged_softmax,
        &pipelines.attention_staged_values,
        &pipelines.attention_streaming,
        &pipelines.out_projection,
        &pipelines.router,
        &pipelines.router_select_serial,
        &pipelines.router_select_parallel,
        &pipelines.expert_union,
        &pipelines.expert_union_fused_tasks,
        &pipelines.expert_upgate,
        &pipelines.expert_down,
        &pipelines.expert_combine,
        &pipelines.native_dense_qgemm,
        &pipelines.native_dense_steel,
        &pipelines.native_dense_steel_gdn_bm64_wm2_wn2,
        &pipelines.native_routed_task_builder,
        &pipelines.native_routed_upgate,
        &pipelines.native_routed_down,
        &pipelines.native_routed_steel_upgate,
        &pipelines.native_routed_steel_down,
    };
    return pipeline_slots("tatara.execution.prefill-pipelines", slots);
}

PrefillCommandIdentity
prefill_execution_policy_identity(const PrefillExecutionPolicy& policy) noexcept {
    ExecutionIdentityEncoder encoder("tatara.execution.prefill-policy");
    encoder.append_u8(static_cast<std::uint8_t>(policy.geometry.schedule));
    encoder.append_u32(policy.geometry.context_capacity);
    encoder.append_u32(policy.geometry.maximum_block_rows);
    encoder.append_u32(policy.geometry.first_chunk_rows);
    encoder.append_u32(policy.geometry.query_tile_rows);
    encoder.append_u32(policy.geometry.attention_partition);
    encoder.append_u32(policy.geometry.exact_rows_per_threadgroup);
    encoder.append_bool(policy.geometry.gdn_gate_hoist);
    encoder.append_u8(static_cast<std::uint8_t>(policy.router_selector));
    encoder.append_u8(static_cast<std::uint8_t>(policy.gdn_recurrence));
    encoder.append_u8(static_cast<std::uint8_t>(policy.attention_kernel));
    encoder.append_u32(policy.staged_attention_minimum_context);
    encoder.append_u32(policy.streaming_attention_minimum_context);
    encoder.append_u8(static_cast<std::uint8_t>(policy.dense_qgemm));
    encoder.append_u8(static_cast<std::uint8_t>(policy.routed_qgemm));
    encoder.append_bool(policy.native_dense_steel);
    encoder.append_bool(
        policy.native_dense_steel_gdn_bm64_wm2_wn2);
    encoder.append_bool(policy.native_routed_shared_expert);
    encoder.append_bool(policy.native_routed_steel);
    encoder.append_bool(policy.command_graph);
    encoder.append_bool(policy.command_graph_lane_events);
    encoder.append_u32(policy.command_graph_chunk_count);
    encoder.append_u32(policy.maximum_units_per_submission);
    encoder.append_u32(policy.maximum_inflight_units);
    return encoder.finish();
}

PrefillCommandIdentity execution_kernel_library_identity() noexcept {
    ExecutionIdentityEncoder encoder("tatara.execution.kernel-library");
    encoder.append_text(backend::metal::generated::kernel_library_source());
    return encoder.finish();
}

PrefillCommandIdentity
execution_pipeline_identity(std::span<const std::uint64_t> identities) noexcept {
    ExecutionIdentityEncoder encoder("tatara.execution.pipeline-list");
    encoder.append_u64(identities.size());
    for (std::uint64_t identity : identities) {
        encoder.append_u64(identity);
    }
    return encoder.finish();
}

} // namespace tatara::runtime
