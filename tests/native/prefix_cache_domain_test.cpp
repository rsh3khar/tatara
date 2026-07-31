#include "tatara/engine/prefix_cache_domain.h"

#include <array>
#include <cstdio>

namespace {

using namespace tatara;

int failures = 0;

void check(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

runtime::PrefillCommandIdentity identity(std::uint8_t value) {
    runtime::PrefillCommandIdentity result{};
    result.fill(value);
    return result;
}

engine::PrefixCacheDomainInputs inputs() {
    return {
        .model_package_identity = identity(1),
        .generated_plan_identity = identity(2),
        .host_execution_identity = identity(3),
        .prepared_record_identity = identity(4),
        .prepared_image_identity = identity(5),
        .kernel_library_identity = identity(6),
        .decode_pipeline_identity = identity(7),
        .prefill_pipeline_identity = identity(8),
        .prefill_policy_identity = identity(9),
        .tokenizer_template_identity = identity(10),
        .device_identity = 11,
        .domain_generation = 12,
        .state_capacity = 65'536,
        .layer_count = 40,
        .gated_delta_layers = 30,
        .attention_layers = 10,
        .key_value_heads = 2,
        .attention_head_dimension = 256,
        .gated_delta_convolution_bytes = 16'384,
        .gated_delta_recurrent_bytes = 524'288,
        .state_layout_schema_version = 1,
        .transfer_schema_version = 1,
        .reset_schema_version = 1,
        .prefill_mode = engine::PrefixCachePrefillMode::Graph,
        .total_cache_budget_bytes = 8ULL << 30U,
        .state_block_bytes = 2ULL << 20U,
        .maximum_entries = 64,
        .maximum_tokens_per_entry = 65'536,
        .minimum_prefix_tokens = 1'024,
        .tail_guard_tokens = 256,
        .boundary_origin_tokens = 256,
        .boundary_stride_tokens = 2'048,
        .graph_scratch_lanes = 3,
    };
}

void every_identity_field_separates_the_domain() {
    const engine::PrefixCacheDomainInputs base = inputs();
    const auto expected = engine::make_prefix_cache_domain(base);
    check(static_cast<bool>(expected), "valid domain constructs");

    std::array<engine::PrefixCacheDomainInputs, 10> mutations{};
    mutations.fill(base);
    mutations[0].model_package_identity[0] ^= 1;
    mutations[1].generated_plan_identity[0] ^= 1;
    mutations[2].host_execution_identity[0] ^= 1;
    mutations[3].prepared_record_identity[0] ^= 1;
    mutations[4].prepared_image_identity[0] ^= 1;
    mutations[5].kernel_library_identity[0] ^= 1;
    mutations[6].decode_pipeline_identity[0] ^= 1;
    mutations[7].prefill_pipeline_identity[0] ^= 1;
    mutations[8].prefill_policy_identity[0] ^= 1;
    mutations[9].tokenizer_template_identity[0] ^= 1;
    for (const auto& mutation : mutations) {
        const auto result = engine::make_prefix_cache_domain(mutation);
        check(result && result.domain->diagnostic_digest !=
                            expected.domain->diagnostic_digest,
              "each execution identity digest separates the cache domain");
    }
}

void every_policy_and_layout_field_separates_the_domain() {
    const engine::PrefixCacheDomainInputs base = inputs();
    const auto expected = engine::make_prefix_cache_domain(base);
    std::array<engine::PrefixCacheDomainInputs, 23> mutations{};
    mutations.fill(base);
    ++mutations[0].device_identity;
    ++mutations[1].domain_generation;
    ++mutations[2].state_capacity;
    ++mutations[3].layer_count;
    ++mutations[3].attention_layers;
    ++mutations[4].gated_delta_layers;
    --mutations[4].attention_layers;
    ++mutations[5].key_value_heads;
    ++mutations[6].attention_head_dimension;
    ++mutations[7].gated_delta_convolution_bytes;
    ++mutations[8].gated_delta_recurrent_bytes;
    ++mutations[9].state_layout_schema_version;
    ++mutations[10].transfer_schema_version;
    ++mutations[11].reset_schema_version;
    mutations[12].prefill_mode = engine::PrefixCachePrefillMode::SingleToken;
    ++mutations[13].total_cache_budget_bytes;
    ++mutations[14].state_block_bytes;
    ++mutations[15].maximum_entries;
    ++mutations[16].maximum_tokens_per_entry;
    ++mutations[17].minimum_prefix_tokens;
    ++mutations[18].tail_guard_tokens;
    ++mutations[19].boundary_origin_tokens;
    ++mutations[20].boundary_stride_tokens;
    ++mutations[21].graph_scratch_lanes;
    ++mutations[22].eviction_schema_version;
    for (const auto& mutation : mutations) {
        const auto result = engine::make_prefix_cache_domain(mutation);
        check(result && result.domain->diagnostic_digest !=
                            expected.domain->diagnostic_digest,
              "each layout/cache-policy field separates the cache domain");
    }
    engine::PrefixCacheDomainInputs digest_schema = base;
    ++digest_schema.token_digest_schema_version;
    const auto result = engine::make_prefix_cache_domain(digest_schema);
    check(result && result.domain->diagnostic_digest != expected.domain->diagnostic_digest,
          "token digest schema separates the cache domain");
}

void invalid_domains_fail_typed() {
    engine::PrefixCacheDomainInputs invalid = inputs();
    invalid.prepared_record_identity = {};
    check(engine::make_prefix_cache_domain(invalid).error ==
              engine::PrefixCacheDomainError::InvalidIdentity,
          "zero identity is rejected");
    invalid = inputs();
    invalid.domain_generation = 0;
    check(engine::make_prefix_cache_domain(invalid).error ==
              engine::PrefixCacheDomainError::InvalidGeneration,
          "zero boot generation is rejected");
    invalid = inputs();
    invalid.attention_layers = 9;
    check(engine::make_prefix_cache_domain(invalid).error ==
              engine::PrefixCacheDomainError::InvalidStateLayout,
          "family census mismatch is rejected");
    invalid = inputs();
    invalid.boundary_stride_tokens = 0;
    check(engine::make_prefix_cache_domain(invalid).error ==
              engine::PrefixCacheDomainError::InvalidPolicy,
          "zero cache boundary stride is rejected");
}

} // namespace

int main() {
    every_identity_field_separates_the_domain();
    every_policy_and_layout_field_separates_the_domain();
    invalid_domains_fail_typed();
    if (failures == 0) {
        std::printf("prefix_cache_domain: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
