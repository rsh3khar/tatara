#pragma once

#include "tatara/model/qwen36_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tatara::runtime {

inline constexpr std::size_t kAdmissionIdentifierBytes = 96;
inline constexpr std::size_t kAdmissionPathBytes = 1024;
inline constexpr std::size_t kBootstrapInputCapacity = 4;
inline constexpr std::size_t kBootstrapProfileRegistryCount = 2;
inline constexpr std::size_t kAdmissionSecondaryDiagnosticCapacity = 8;
inline constexpr std::uint16_t kNoAdmissionDiagnosticIndex = 0xffffu;
inline constexpr std::uint32_t kBootstrapContractVersion = 1;
inline constexpr std::uint32_t kBootstrapProfileRegistryVersion = 1;
inline constexpr std::uint32_t kImplementationProfileVersion = 1;
inline constexpr std::uint32_t kAdmissionContractVersion = 1;

template <std::size_t Capacity> struct BoundedAdmissionText {
    std::array<char, Capacity> bytes{};
    std::uint16_t size{0};
    bool complete{true};

    constexpr BoundedAdmissionText() noexcept = default;

    constexpr explicit BoundedAdmissionText(std::string_view source) noexcept {
        assign(source);
    }

    constexpr bool assign(std::string_view source) noexcept {
        bytes = {};
        size = 0;
        complete = source.size() <= Capacity &&
                   source.size() <= static_cast<std::size_t>(UINT16_MAX);
        if (!complete) {
            return false;
        }
        for (std::size_t index = 0; index < source.size(); ++index) {
            bytes[index] = source[index];
        }
        size = static_cast<std::uint16_t>(source.size());
        return true;
    }

    [[nodiscard]] constexpr bool well_formed() const noexcept {
        return complete && static_cast<std::size_t>(size) <= Capacity;
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return !well_formed() || size == 0;
    }

    [[nodiscard]] constexpr std::string_view view() const noexcept {
        return well_formed() ? std::string_view{bytes.data(), size} : std::string_view{};
    }
};

template <std::size_t Capacity>
[[nodiscard]] constexpr bool
operator==(const BoundedAdmissionText<Capacity>& left,
           const BoundedAdmissionText<Capacity>& right) noexcept {
    return left.well_formed() && right.well_formed() && left.view() == right.view();
}

enum class AdmissionBoundary : std::uint8_t {
    B_MINUS_1 = 0,
    B0 = 1,
    B1 = 2,
    B2 = 3,
    B3 = 4,
    B4 = 5,
    B5 = 6,
    B6 = 7,
    B7 = 8,
    B8 = 9,
    B9 = 10,
    B10_E = 11,
    B10_S = 12,
};

enum class AdmissionErrorKind : std::uint8_t {
    BOOTSTRAP_PROFILE_UNAVAILABLE = 0,
    BOOTSTRAP_PROFILE_MISMATCH = 1,
    BOOTSTRAP_INPUT_LIMIT = 2,
    BOOTSTRAP_INPUT_CHANGED = 3,
    BOOTSTRAP_EVIDENCE_INCOMPLETE = 4,
    BOOTSTRAP_LIMIT = 5,
    DEVICE_ACTION_PROHIBITED = 6,
    UNSUPPORTED_GENERATED_PLAN = 7,
    UNSUPPORTED_CONFIGURATION = 8,
    SERVICE_START_AUTHORITY_INVALID = 9,
    HARDWARE_FACT_UNAVAILABLE = 10,
    EXTERNAL_OCCUPANCY_UNAVAILABLE = 11,
    HARDWARE_STATE_CHANGED = 12,
    EVIDENCE_IDENTITY_MISMATCH = 13,
    EVIDENCE_EXPIRED = 14,
    LOWER_BOUND_NONFIT = 15,
    EVIDENCE_MISSING = 16,
    ADMISSION_ARITHMETIC_OVERFLOW = 17,
    SINGLE_BUFFER_LIMIT = 18,
    PROCESS_DESCRIPTOR_LIMIT = 19,
    LISTEN_BACKLOG_LIMIT = 20,
    THREAD_RESOURCE_LIMIT = 21,
    METAL_WORKING_SET_LIMIT = 22,
    UNIFIED_MEMORY_POLICY_LIMIT = 23,
    ALLOCATION_FAILURE = 24,
    RESOURCE_ACQUISITION_FAILURE = 25,
    OPAQUE_CONSTRUCTION_FAILURE = 26,
    LIFETIME_CONSERVATION_FAILURE = 27,
    RESOURCE_CONSERVATION_FAILURE = 28,
    UNWIND_FAILURE = 29,
    ADMISSION_IDENTITY_CHANGED = 30,
    NONE = 255,
};

enum class ServiceStartAuthority : std::uint8_t {
    SourceOnly = 0,
    EphemeralLab = 1,
    Production = 2,
};

enum class BootstrapInputKind : std::uint8_t {
    Configuration = 0,
    Package = 1,
    PreparedRecord = 2,
    Evidence = 3,
};

enum class QgemmExecutionPolicy : std::uint8_t {
    ExactRow = 0,
    DenseMatrix = 1,
    RaggedMatrix = 2,
    SplitKMatrix = 3,
};

enum class AdmissionResourceKind : std::uint8_t {
    None = 0,
    Bytes = 1,
    FileDescriptors = 2,
    ListenBacklogEntries = 3,
    Threads = 4,
    RingUsableCells = 5,
    RingPhysicalCells = 6,
    FixedRecords = 7,
    Tokens = 8,
};

enum class AdmissionUnderlyingCause : std::uint8_t {
    InvalidProfile = 0,
    ArithmeticOverflow = 1,
    DescriptorBudgetInsufficient = 2,
    MemoryBudgetInsufficient = 3,
    ExternalOccupancyUnavailable = 4,
    ListenerCreateFailed = 5,
    ListenerConfigureFailed = 6,
    BindFailed = 7,
    ListenFailed = 8,
    ThreadCreateFailed = 9,
    WakeCreateFailed = 10,
    IdentityInvalid = 11,
    BootGateFailed = 12,
    StartupUnwindFailed = 13,
    None = 255,
};

enum class AdmissionUnwindStatus : std::uint8_t {
    NotRequired = 0,
    NotStarted = 1,
    Complete = 2,
    Failed = 3,
    Retained = 4,
};

struct BootstrapInputRecord {
    BootstrapInputKind kind{BootstrapInputKind::Configuration};
    BoundedAdmissionText<kAdmissionPathBytes> path;
    std::uint64_t declared_bytes{0};
    std::uint64_t acquired_bytes{0};
    bool supported_object_type{true};
    bool read_complete{true};
    bool unchanged{true};
};

struct BootstrapRootRecord {
    std::uint32_t bootstrap_contract_version{0};
    BoundedAdmissionText<kAdmissionIdentifierBytes> requested_profile_id;
    BoundedAdmissionText<kAdmissionIdentifierBytes> root_invocation_hash;
    std::array<BootstrapInputRecord, kBootstrapInputCapacity> inputs{};
    std::uint16_t input_count{0};
    std::uint64_t total_owned_bytes{0};
    std::uint64_t diagnostic_bytes{0};
};

struct BootstrapProfileRecord {
    std::uint32_t bootstrap_contract_version{0};
    BoundedAdmissionText<kAdmissionIdentifierBytes> profile_id;
    BoundedAdmissionText<kAdmissionIdentifierBytes> envelope_sha256;
    bool applicable{false};
    std::uint16_t maximum_input_count{0};
    std::array<std::uint64_t, kBootstrapInputCapacity> maximum_input_bytes{};
    std::uint64_t fixed_owned_bytes{0};
    std::uint64_t maximum_total_owned_bytes{0};
    std::uint64_t maximum_diagnostic_bytes{0};
};

struct BootstrapProfileRegistryRecord {
    std::uint32_t registry_version{0};
    BoundedAdmissionText<kAdmissionIdentifierBytes> registry_sha256;
    std::array<BootstrapProfileRecord, kBootstrapProfileRegistryCount> profiles{};
};

struct BootstrapEvidenceRecord {
    std::uint32_t bootstrap_contract_version{0};
    std::uint32_t profile_registry_version{0};
    BoundedAdmissionText<kAdmissionIdentifierBytes> evidence_id;
    BoundedAdmissionText<kAdmissionIdentifierBytes> selected_profile_id;
    BoundedAdmissionText<kAdmissionIdentifierBytes> profile_registry_sha256;
    BoundedAdmissionText<kAdmissionIdentifierBytes> envelope_sha256;
    bool complete{false};
    bool current{false};
    bool identity_matches{false};
    bool envelope_independently_admitted{false};
};

// Constexpr mirror of the frozen model::qwen36::valid_model_plan contract.
// The model header's checker is consteval and therefore cannot be applied to
// a plan received through a runtime reference; this mirror performs the same
// structural validation and must stay in exact lockstep with it. It contains
// no model dimensions: every bound it checks is structural (nonzero, sha-256
// length, divisibility, layer mix), so an alternate generated plan validates
// without source edits.
template <std::size_t LayerCount>
[[nodiscard]] constexpr bool valid_generated_model_plan(
    const model::qwen36::StaticModelPlan<LayerCount>& plan) noexcept {
    if (LayerCount == 0 || plan.id.empty() || plan.family.empty() ||
        plan.package_sha256.size() != 64) {
        return false;
    }
    if (plan.artifact.id.empty() || plan.artifact.model_type.empty() ||
        plan.artifact.manifest_sha256.size() != 64 || plan.artifact.tensor_count == 0 ||
        plan.artifact.tensor_bytes == 0 || plan.artifact.file_count == 0 ||
        plan.artifact.weight_file_count == 0) {
        return false;
    }
    if (plan.dimensions.hidden == 0 || plan.dimensions.vocabulary == 0 ||
        plan.attention.query_heads == 0 || plan.attention.key_value_heads == 0 ||
        plan.attention.head_dimension == 0 ||
        plan.attention.query_heads % plan.attention.key_value_heads != 0) {
        return false;
    }
    if (plan.gated_delta.recurrent_heads == 0 || plan.gated_delta.state_dimension == 0 ||
        plan.mixture_of_experts.experts == 0 ||
        plan.mixture_of_experts.active_experts == 0 ||
        plan.mixture_of_experts.active_experts > plan.mixture_of_experts.experts ||
        plan.mixture_of_experts.expert_dimension == 0 || plan.weights.group_size == 0 ||
        plan.initial_serving_capacity == 0) {
        return false;
    }
    const model::qwen36::TokenizerSpec& tokenizer = plan.tokenizer;
    if (tokenizer.data_path.empty() || tokenizer.data_sha256.size() != 64 ||
        tokenizer.data_size_bytes == 0 ||
        tokenizer.data_size_bytes > 64ULL * 1024ULL * 1024ULL ||
        tokenizer.config_path.empty() || tokenizer.config_sha256.size() != 64 ||
        tokenizer.config_size_bytes == 0 || tokenizer.template_path.empty() ||
        tokenizer.template_sha256.size() != 64 || tokenizer.template_size_bytes == 0 ||
        tokenizer.split_pattern.empty() ||
        tokenizer.vocabulary != plan.dimensions.vocabulary ||
        tokenizer.populated_vocabulary == 0 ||
        tokenizer.populated_vocabulary > tokenizer.vocabulary ||
        tokenizer.maximum_context < plan.initial_serving_capacity ||
        tokenizer.stop_token_count == 0 ||
        tokenizer.stop_token_count > tokenizer.stop_token_ids.size() ||
        tokenizer.end_of_text_id >= tokenizer.populated_vocabulary ||
        tokenizer.message_start_id >= tokenizer.populated_vocabulary ||
        tokenizer.message_end_id >= tokenizer.populated_vocabulary ||
        tokenizer.thinking_start_id >= tokenizer.populated_vocabulary ||
        tokenizer.thinking_end_id >= tokenizer.populated_vocabulary ||
        tokenizer.padding_id >= tokenizer.populated_vocabulary) {
        return false;
    }
    for (std::size_t index = 0; index < tokenizer.stop_token_count; ++index) {
        if (tokenizer.stop_token_ids[index] >= tokenizer.populated_vocabulary) {
            return false;
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (tokenizer.stop_token_ids[index] == tokenizer.stop_token_ids[previous]) {
                return false;
            }
        }
    }

    std::size_t full_attention_layers = 0;
    for (const model::qwen36::LayerKind kind : plan.layers) {
        full_attention_layers += kind == model::qwen36::LayerKind::FullAttention ? 1 : 0;
    }
    return full_attention_layers > 0 && full_attention_layers < LayerCount;
}

class GeneratedPlanAdmissionFacts {
  public:
    constexpr GeneratedPlanAdmissionFacts() noexcept = default;

    [[nodiscard]] constexpr const auto& plan_id() const noexcept {
        return plan_id_;
    }
    [[nodiscard]] constexpr const auto& family() const noexcept {
        return family_;
    }
    [[nodiscard]] constexpr const auto& package_sha256() const noexcept {
        return package_sha256_;
    }
    [[nodiscard]] constexpr const auto& artifact_manifest_sha256() const noexcept {
        return artifact_manifest_sha256_;
    }
    [[nodiscard]] constexpr bool copied_completely() const noexcept {
        return copied_completely_;
    }
    [[nodiscard]] constexpr bool valid() const noexcept {
        return valid_;
    }
    [[nodiscard]] constexpr model::qwen36::WeightFormat weight_format() const noexcept {
        return weight_format_;
    }
    [[nodiscard]] constexpr std::uint32_t weight_group_size() const noexcept {
        return weight_group_size_;
    }
    [[nodiscard]] constexpr std::uint32_t maximum_context() const noexcept {
        return maximum_context_;
    }
    [[nodiscard]] constexpr std::uint32_t initial_serving_capacity() const noexcept {
        return initial_serving_capacity_;
    }
    [[nodiscard]] constexpr std::uint32_t hidden() const noexcept {
        return hidden_;
    }
    [[nodiscard]] constexpr std::uint32_t vocabulary() const noexcept {
        return vocabulary_;
    }
    [[nodiscard]] constexpr std::uint32_t query_heads() const noexcept {
        return query_heads_;
    }
    [[nodiscard]] constexpr std::uint32_t key_value_heads() const noexcept {
        return key_value_heads_;
    }
    [[nodiscard]] constexpr std::uint32_t head_dimension() const noexcept {
        return head_dimension_;
    }
    [[nodiscard]] constexpr std::uint32_t recurrent_heads() const noexcept {
        return recurrent_heads_;
    }
    [[nodiscard]] constexpr std::uint32_t state_dimension() const noexcept {
        return state_dimension_;
    }
    [[nodiscard]] constexpr std::uint32_t experts() const noexcept {
        return experts_;
    }
    [[nodiscard]] constexpr std::uint32_t active_experts() const noexcept {
        return active_experts_;
    }
    [[nodiscard]] constexpr std::uint32_t expert_dimension() const noexcept {
        return expert_dimension_;
    }
    [[nodiscard]] constexpr std::uint32_t layer_count() const noexcept {
        return layer_count_;
    }
    [[nodiscard]] constexpr std::uint32_t gated_delta_layers() const noexcept {
        return gated_delta_layers_;
    }
    [[nodiscard]] constexpr std::uint32_t attention_layers() const noexcept {
        return attention_layers_;
    }

  private:
    template <std::size_t LayerCount>
    friend constexpr GeneratedPlanAdmissionFacts make_generated_plan_admission_facts(
        const model::qwen36::StaticModelPlan<LayerCount>& plan) noexcept;

    BoundedAdmissionText<kAdmissionIdentifierBytes> plan_id_;
    BoundedAdmissionText<kAdmissionIdentifierBytes> family_;
    BoundedAdmissionText<kAdmissionIdentifierBytes> package_sha256_;
    BoundedAdmissionText<kAdmissionIdentifierBytes> artifact_manifest_sha256_;
    bool copied_completely_{false};
    bool valid_{false};
    model::qwen36::WeightFormat weight_format_{model::qwen36::WeightFormat::AffineQ4};
    std::uint32_t weight_group_size_{0};
    std::uint32_t maximum_context_{0};
    std::uint32_t initial_serving_capacity_{0};
    std::uint32_t hidden_{0};
    std::uint32_t vocabulary_{0};
    std::uint32_t query_heads_{0};
    std::uint32_t key_value_heads_{0};
    std::uint32_t head_dimension_{0};
    std::uint32_t recurrent_heads_{0};
    std::uint32_t state_dimension_{0};
    std::uint32_t experts_{0};
    std::uint32_t active_experts_{0};
    std::uint32_t expert_dimension_{0};
    std::uint32_t layer_count_{0};
    std::uint32_t gated_delta_layers_{0};
    std::uint32_t attention_layers_{0};
};

// Plan-parameterized admission-facts extraction. Every model fact and the
// validity verdict are computed from the generated plan itself; the private
// members are reachable only through this factory, so a caller cannot forge
// copied/valid bits without presenting a plan that actually validates.
template <std::size_t LayerCount>
[[nodiscard]] constexpr GeneratedPlanAdmissionFacts make_generated_plan_admission_facts(
    const model::qwen36::StaticModelPlan<LayerCount>& plan) noexcept {
    GeneratedPlanAdmissionFacts facts;
    facts.plan_id_.assign(plan.id);
    facts.family_.assign(plan.family);
    facts.package_sha256_.assign(plan.package_sha256);
    facts.artifact_manifest_sha256_.assign(plan.artifact.manifest_sha256);
    facts.copied_completely_ =
        facts.plan_id_.well_formed() && !facts.plan_id_.empty() &&
        facts.family_.well_formed() && !facts.family_.empty() &&
        facts.package_sha256_.well_formed() && facts.package_sha256_.size == 64 &&
        facts.artifact_manifest_sha256_.well_formed() &&
        facts.artifact_manifest_sha256_.size == 64;
    facts.weight_format_ = plan.weights.format;
    facts.weight_group_size_ = plan.weights.group_size;
    facts.maximum_context_ = plan.tokenizer.maximum_context;
    facts.initial_serving_capacity_ = plan.initial_serving_capacity;
    facts.hidden_ = plan.dimensions.hidden;
    facts.vocabulary_ = plan.dimensions.vocabulary;
    facts.query_heads_ = plan.attention.query_heads;
    facts.key_value_heads_ = plan.attention.key_value_heads;
    facts.head_dimension_ = plan.attention.head_dimension;
    facts.recurrent_heads_ = plan.gated_delta.recurrent_heads;
    facts.state_dimension_ = plan.gated_delta.state_dimension;
    facts.experts_ = plan.mixture_of_experts.experts;
    facts.active_experts_ = plan.mixture_of_experts.active_experts;
    facts.expert_dimension_ = plan.mixture_of_experts.expert_dimension;
    facts.layer_count_ = static_cast<std::uint32_t>(plan.layers.size());
    for (const model::qwen36::LayerKind kind : plan.layers) {
        if (kind == model::qwen36::LayerKind::GatedDelta) {
            ++facts.gated_delta_layers_;
        } else if (kind == model::qwen36::LayerKind::FullAttention) {
            ++facts.attention_layers_;
        }
    }
    facts.valid_ = valid_generated_model_plan(plan) &&
                   facts.copied_completely_ &&
                   facts.gated_delta_layers_ != 0 && facts.attention_layers_ != 0 &&
                   facts.attention_layers_ < plan.layers.size();
    return facts;
}

// Compiled-plan form: zero function arguments, the plan bound at compile
// time. It delegates to the plan-parameterized form above and additionally
// requires the frozen consteval model checker to agree with the constexpr
// mirror, so the two validation paths cannot drift silently.
template <const auto& Plan>
[[nodiscard]] consteval GeneratedPlanAdmissionFacts
make_generated_plan_admission_facts() noexcept {
    static_assert(model::qwen36::valid_model_plan(Plan) ==
                  valid_generated_model_plan(Plan));
    return make_generated_plan_admission_facts(Plan);
}

struct ImplementationProfileRecord {
    std::uint32_t profile_version{0};
    BoundedAdmissionText<kAdmissionIdentifierBytes> profile_id;
    BoundedAdmissionText<kAdmissionIdentifierBytes> profile_sha256;
    BoundedAdmissionText<kAdmissionIdentifierBytes> generated_plan_id;
    BoundedAdmissionText<kAdmissionIdentifierBytes> generated_plan_package_sha256;
    BoundedAdmissionText<kAdmissionIdentifierBytes> artifact_manifest_sha256;
    model::qwen36::WeightFormat weight_format{model::qwen36::WeightFormat::AffineQ4};
    std::uint32_t weight_group_size{0};
    std::uint32_t context_representability_bound{0};
    std::uint32_t minimum_hidden{0};
    std::uint32_t maximum_hidden{0};
    std::uint32_t hidden_multiple{0};
    std::uint32_t maximum_vocabulary{0};
    std::uint32_t minimum_query_heads{0};
    std::uint32_t maximum_query_heads{0};
    std::uint32_t query_heads_multiple{0};
    std::uint32_t minimum_key_value_heads{0};
    std::uint32_t maximum_key_value_heads{0};
    std::uint32_t key_value_heads_multiple{0};
    std::uint32_t query_to_key_value_ratio{0};
    std::uint32_t minimum_head_dimension{0};
    std::uint32_t maximum_head_dimension{0};
    std::uint32_t head_dimension_multiple{0};
    std::uint32_t minimum_recurrent_heads{0};
    std::uint32_t maximum_recurrent_heads{0};
    std::uint32_t recurrent_heads_multiple{0};
    std::uint32_t minimum_state_dimension{0};
    std::uint32_t maximum_state_dimension{0};
    std::uint32_t state_dimension_multiple{0};
    std::uint32_t minimum_experts{0};
    std::uint32_t maximum_experts{0};
    bool experts_must_be_power_of_two{false};
    std::uint32_t minimum_active_experts{0};
    std::uint32_t maximum_active_experts_exclusive{0};
    std::uint32_t minimum_expert_dimension{0};
    std::uint32_t maximum_expert_dimension{0};
    std::uint32_t expert_dimension_multiple{0};
    std::uint32_t minimum_layer_count{0};
    std::uint32_t maximum_layer_count{0};
    std::uint32_t minimum_gated_delta_layers{0};
    std::uint32_t minimum_attention_layers{0};
};

struct AdmissionIdentity {
    BoundedAdmissionText<kAdmissionIdentifierBytes> generated_plan_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> artifact_manifest_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> model_image_layout_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> tokenizer_template_hashes;
    BoundedAdmissionText<kAdmissionIdentifierBytes> source_commit;
    BoundedAdmissionText<kAdmissionIdentifierBytes> binary_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> metallib_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> compiler_and_sdk_identity;
    BoundedAdmissionText<kAdmissionIdentifierBytes> configuration_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> owner_registry_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> phase_graph_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> evidence_bundle_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> bootstrap_envelope_and_evidence_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> bootstrap_root_invocation_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> bootstrap_profile_registry_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> hardware_profile_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> os_build;
    BoundedAdmissionText<kAdmissionIdentifierBytes> device_registry_id;
    std::uint32_t admission_contract_version{0};
    BoundedAdmissionText<kAdmissionIdentifierBytes> frozen_c4_owner_contract_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> frozen_c4_owner_census_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> http_ingress_contract_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> service_contract_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> resource_registry_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> canonical_order_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> hardware_fact_api_ledger_hash;
    BoundedAdmissionText<kAdmissionIdentifierBytes> implementation_profile_hash;
    ServiceStartAuthority service_start_authority_scope{ServiceStartAuthority::SourceOnly};
};

struct PreDeviceAdmissionRequest {
    std::uint32_t configured_context_capacity{0};
    std::uint32_t physical_slot_count{1};
    std::uint32_t maximum_concurrent_requests{1};
    std::uint32_t queue_depth{0};
    std::uint32_t request_deadline_milliseconds{0};
    std::uint32_t drain_timeout_milliseconds{0};
    bool prompt_cache{false};
    bool composed_prefill{false};
    QgemmExecutionPolicy qgemm_policy{QgemmExecutionPolicy::ExactRow};
    ServiceStartAuthority service_start_authority{ServiceStartAuthority::SourceOnly};
    bool gpu_halt_active{true};
    bool requests_live_device_facts{false};
    bool requests_device_construction{false};
    bool requests_model_execution{false};
    bool requests_command_authorization{false};
    bool requests_listener{false};
    bool requests_service_start{false};
};

struct PreDeviceAdmissionInput {
    BootstrapRootRecord bootstrap;
    BootstrapProfileRegistryRecord bootstrap_profile_registry;
    BootstrapEvidenceRecord bootstrap_evidence;
    GeneratedPlanAdmissionFacts generated_plan;
    ImplementationProfileRecord implementation_profile;
    AdmissionIdentity admission_identity;
    PreDeviceAdmissionRequest request;
};

struct AdmissionOptionalUnsigned {
    bool present{false};
    std::uint64_t value{0};
};

struct AdmissionOptionalSignedMagnitude {
    bool present{false};
    bool deficit{false};
    std::uint64_t magnitude{0};
};

struct AdmissionOptionalErrno {
    bool present{false};
    std::int32_t value{0};
};

struct AdmissionDiagnostic {
    AdmissionErrorKind error_kind{AdmissionErrorKind::NONE};
    AdmissionBoundary boundary{AdmissionBoundary::B_MINUS_1};
    BoundedAdmissionText<kAdmissionIdentifierBytes> owner_id_if_any;
    BoundedAdmissionText<kAdmissionIdentifierBytes> phase_if_any;
    BoundedAdmissionText<kAdmissionIdentifierBytes> unknown_or_evidence_id_if_any;
    BoundedAdmissionText<kAdmissionIdentifierBytes> resource_id_if_any;
    std::uint16_t array_index{kNoAdmissionDiagnosticIndex};
    AdmissionOptionalUnsigned requested_bytes;
    AdmissionOptionalUnsigned byte_limit;
    AdmissionOptionalUnsigned requested_resource_count;
    AdmissionOptionalUnsigned resource_limit;
    AdmissionResourceKind resource_kind{AdmissionResourceKind::None};
    AdmissionOptionalSignedMagnitude headroom_or_deficit;
    AdmissionOptionalUnsigned hardware_fact_generation;
    AdmissionUnderlyingCause underlying_cause{AdmissionUnderlyingCause::None};
    AdmissionOptionalErrno underlying_errno;
    bool retry_safe{false};
    AdmissionUnwindStatus unwind_status{AdmissionUnwindStatus::NotRequired};
};

struct AdmissionError {
    AdmissionDiagnostic primary;
    AdmissionIdentity admission_identity;
    std::array<AdmissionDiagnostic, kAdmissionSecondaryDiagnosticCapacity>
        secondary_diagnostics{};
    std::uint16_t secondary_diagnostic_count{0};
    std::uint16_t total_observed_count{0};
    bool secondary_diagnostics_truncated{false};
};

struct PreDeviceAdmissionPlan {
    PreDeviceAdmissionInput input;
    AdmissionBoundary completed_through{AdmissionBoundary::B1};
    bool allocation_authorized{false};
    bool device_action_authorized{false};
    bool command_authorized{false};
    bool listener_authorized{false};
    bool service_start_authorized{false};
};

struct PreDeviceAdmissionResult {
    bool passed{false};
    PreDeviceAdmissionPlan plan;
    AdmissionError error;

    explicit constexpr operator bool() const noexcept {
        return passed;
    }
};

[[nodiscard]] std::string_view admission_error_name(AdmissionErrorKind kind) noexcept;

[[nodiscard]] PreDeviceAdmissionResult
plan_pre_device_admission(const PreDeviceAdmissionInput& input) noexcept;

} // namespace tatara::runtime
