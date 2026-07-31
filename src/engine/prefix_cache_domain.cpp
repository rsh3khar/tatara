#include "tatara/engine/prefix_cache_domain.h"

#include "tatara/runtime/execution_identity.h"

#include <algorithm>

namespace tatara::engine {
namespace {

bool nonzero(const runtime::PrefillCommandIdentity& identity) noexcept {
    return std::any_of(identity.begin(), identity.end(),
                       [](std::uint8_t value) { return value != 0; });
}

bool valid_identities(const PrefixCacheDomainInputs& inputs) noexcept {
    return nonzero(inputs.model_package_identity) &&
           nonzero(inputs.generated_plan_identity) &&
           nonzero(inputs.host_execution_identity) &&
           nonzero(inputs.prepared_record_identity) &&
           nonzero(inputs.prepared_image_identity) &&
           nonzero(inputs.kernel_library_identity) &&
           nonzero(inputs.decode_pipeline_identity) &&
           nonzero(inputs.prefill_pipeline_identity) &&
           nonzero(inputs.prefill_policy_identity) &&
           nonzero(inputs.tokenizer_template_identity);
}

} // namespace

PrefixCacheDomainResult
make_prefix_cache_domain(const PrefixCacheDomainInputs& inputs) noexcept {
    if (!valid_identities(inputs) || inputs.device_identity == 0) {
        return {.error = PrefixCacheDomainError::InvalidIdentity};
    }
    if (inputs.domain_generation == 0) {
        return {.error = PrefixCacheDomainError::InvalidGeneration};
    }
    if (inputs.state_capacity == 0 || inputs.layer_count == 0 ||
        inputs.gated_delta_layers == 0 || inputs.attention_layers == 0 ||
        inputs.gated_delta_layers + inputs.attention_layers != inputs.layer_count ||
        inputs.key_value_heads == 0 || inputs.attention_head_dimension == 0 ||
        inputs.gated_delta_convolution_bytes == 0 ||
        inputs.gated_delta_recurrent_bytes == 0 ||
        inputs.state_layout_schema_version == 0 ||
        inputs.transfer_schema_version == 0 || inputs.reset_schema_version == 0) {
        return {.error = PrefixCacheDomainError::InvalidStateLayout};
    }
    if (inputs.total_cache_budget_bytes == 0 || inputs.state_block_bytes == 0 ||
        inputs.maximum_entries == 0 || inputs.maximum_tokens_per_entry == 0 ||
        inputs.minimum_prefix_tokens == 0 ||
        inputs.minimum_prefix_tokens > inputs.maximum_tokens_per_entry ||
        inputs.boundary_origin_tokens > inputs.maximum_tokens_per_entry ||
        inputs.boundary_stride_tokens == 0 || inputs.graph_scratch_lanes == 0 ||
        inputs.eviction_schema_version == 0 ||
        inputs.token_digest_schema_version == 0) {
        return {.error = PrefixCacheDomainError::InvalidPolicy};
    }

    runtime::ExecutionIdentityEncoder encoder("tatara.prefix-cache.boot-domain");
    encoder.append_u32(kPrefixCacheDomainSchemaVersion);
    encoder.append_identity(inputs.model_package_identity);
    encoder.append_identity(inputs.generated_plan_identity);
    encoder.append_identity(inputs.host_execution_identity);
    encoder.append_identity(inputs.prepared_record_identity);
    encoder.append_identity(inputs.prepared_image_identity);
    encoder.append_identity(inputs.kernel_library_identity);
    encoder.append_identity(inputs.decode_pipeline_identity);
    encoder.append_identity(inputs.prefill_pipeline_identity);
    encoder.append_identity(inputs.prefill_policy_identity);
    encoder.append_identity(inputs.tokenizer_template_identity);
    encoder.append_u64(inputs.device_identity);
    encoder.append_u64(inputs.domain_generation);
    encoder.append_u32(inputs.state_capacity);
    encoder.append_u32(inputs.layer_count);
    encoder.append_u32(inputs.gated_delta_layers);
    encoder.append_u32(inputs.attention_layers);
    encoder.append_u32(inputs.key_value_heads);
    encoder.append_u32(inputs.attention_head_dimension);
    encoder.append_u64(inputs.gated_delta_convolution_bytes);
    encoder.append_u64(inputs.gated_delta_recurrent_bytes);
    encoder.append_u32(inputs.state_layout_schema_version);
    encoder.append_u32(inputs.transfer_schema_version);
    encoder.append_u32(inputs.reset_schema_version);
    encoder.append_u8(static_cast<std::uint8_t>(inputs.prefill_mode));
    encoder.append_u64(inputs.total_cache_budget_bytes);
    encoder.append_u64(inputs.state_block_bytes);
    encoder.append_u32(inputs.maximum_entries);
    encoder.append_u32(inputs.maximum_tokens_per_entry);
    encoder.append_u32(inputs.minimum_prefix_tokens);
    encoder.append_u32(inputs.tail_guard_tokens);
    encoder.append_u32(inputs.boundary_origin_tokens);
    encoder.append_u32(inputs.boundary_stride_tokens);
    encoder.append_u32(inputs.graph_scratch_lanes);
    encoder.append_u32(inputs.eviction_schema_version);
    encoder.append_u32(inputs.token_digest_schema_version);
    return {
        .error = PrefixCacheDomainError::None,
        .domain =
            service::PrefixCacheDomain{
                .diagnostic_digest = encoder.finish(),
                .generation = inputs.domain_generation,
            },
    };
}

} // namespace tatara::engine
