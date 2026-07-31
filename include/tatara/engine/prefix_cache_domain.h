#pragma once

#include "tatara/runtime/prefill_command_plan.h"
#include "tatara/service/prefix_cache.h"

#include <cstdint>
#include <optional>

namespace tatara::engine {

inline constexpr std::uint32_t kPrefixCacheDomainSchemaVersion = 1;
inline constexpr std::uint32_t kPrefixCacheEvictionSchemaVersion = 1;
inline constexpr std::uint32_t kPrefixCacheTokenDigestSchemaVersion = 1;

enum class PrefixCachePrefillMode : std::uint8_t {
    SingleToken,
    Graph,
};

struct PrefixCacheDomainInputs {
    runtime::PrefillCommandIdentity model_package_identity{};
    runtime::PrefillCommandIdentity generated_plan_identity{};
    runtime::PrefillCommandIdentity host_execution_identity{};
    runtime::PrefillCommandIdentity prepared_record_identity{};
    runtime::PrefillCommandIdentity prepared_image_identity{};
    runtime::PrefillCommandIdentity kernel_library_identity{};
    runtime::PrefillCommandIdentity decode_pipeline_identity{};
    runtime::PrefillCommandIdentity prefill_pipeline_identity{};
    runtime::PrefillCommandIdentity prefill_policy_identity{};
    runtime::PrefillCommandIdentity tokenizer_template_identity{};
    std::uint64_t device_identity{0};
    std::uint64_t domain_generation{0};
    std::uint32_t state_capacity{0};
    std::uint32_t layer_count{0};
    std::uint32_t gated_delta_layers{0};
    std::uint32_t attention_layers{0};
    std::uint32_t key_value_heads{0};
    std::uint32_t attention_head_dimension{0};
    std::uint64_t gated_delta_convolution_bytes{0};
    std::uint64_t gated_delta_recurrent_bytes{0};
    std::uint32_t state_layout_schema_version{0};
    std::uint32_t transfer_schema_version{0};
    std::uint32_t reset_schema_version{0};
    PrefixCachePrefillMode prefill_mode{PrefixCachePrefillMode::Graph};
    std::uint64_t total_cache_budget_bytes{0};
    std::uint64_t state_block_bytes{0};
    std::uint32_t maximum_entries{0};
    std::uint32_t maximum_tokens_per_entry{0};
    std::uint32_t minimum_prefix_tokens{0};
    std::uint32_t tail_guard_tokens{0};
    std::uint32_t boundary_origin_tokens{0};
    std::uint32_t boundary_stride_tokens{0};
    std::uint32_t graph_scratch_lanes{0};
    std::uint32_t eviction_schema_version{kPrefixCacheEvictionSchemaVersion};
    std::uint32_t token_digest_schema_version{kPrefixCacheTokenDigestSchemaVersion};
};

enum class PrefixCacheDomainError : std::uint8_t {
    None,
    InvalidIdentity,
    InvalidGeneration,
    InvalidStateLayout,
    InvalidPolicy,
};

struct PrefixCacheDomainResult {
    PrefixCacheDomainError error{PrefixCacheDomainError::InvalidIdentity};
    std::optional<service::PrefixCacheDomain> domain;

    explicit operator bool() const noexcept {
        return error == PrefixCacheDomainError::None && domain.has_value();
    }
};

PrefixCacheDomainResult
make_prefix_cache_domain(const PrefixCacheDomainInputs& inputs) noexcept;

} // namespace tatara::engine
