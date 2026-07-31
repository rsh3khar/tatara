#include "tatara/generated/model_plan.h"
#include "tatara/runtime/checked_arithmetic.h"
#include "tatara/runtime/startup_admission.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {

std::uint64_t g_allocation_count = 0;

using tatara::model::qwen36::ChatTemplateKind;
using tatara::model::qwen36::LayerKind;
using tatara::model::qwen36::StaticModelPlan;
using tatara::model::qwen36::TokenizerDecoder;
using tatara::model::qwen36::TokenizerKind;
using tatara::model::qwen36::TokenizerNormalization;
using tatara::model::qwen36::TokenizerPretokenizer;
using tatara::model::qwen36::WeightFormat;
using tatara::runtime::AdmissionBoundary;
using tatara::runtime::AdmissionErrorKind;
using tatara::runtime::AdmissionResourceKind;
using tatara::runtime::AdmissionUnderlyingCause;
using tatara::runtime::AdmissionUnwindStatus;
using tatara::runtime::GeneratedPlanAdmissionFacts;
using tatara::runtime::ImplementationProfileRecord;
using tatara::runtime::PreDeviceAdmissionInput;
using tatara::runtime::QgemmExecutionPolicy;
using tatara::runtime::ServiceStartAuthority;

constexpr std::string_view kHash1 =
    "1111111111111111111111111111111111111111111111111111111111111111";
constexpr std::string_view kHash2 =
    "2222222222222222222222222222222222222222222222222222222222222222";
constexpr std::string_view kHash3 =
    "3333333333333333333333333333333333333333333333333333333333333333";
constexpr std::string_view kHash4 =
    "4444444444444444444444444444444444444444444444444444444444444444";
constexpr std::string_view kHash5 =
    "5555555555555555555555555555555555555555555555555555555555555555";
constexpr std::string_view kHash6 =
    "6666666666666666666666666666666666666666666666666666666666666666";
constexpr std::string_view kHash7 =
    "7777777777777777777777777777777777777777777777777777777777777777";
constexpr std::string_view kHash8 =
    "8888888888888888888888888888888888888888888888888888888888888888";
constexpr std::string_view kHash9 =
    "9999999999999999999999999999999999999999999999999999999999999999";
constexpr std::string_view kHashA =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr std::string_view kHashB =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
constexpr std::string_view kHashC =
    "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
constexpr std::string_view kHashD =
    "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
constexpr std::string_view kHashE =
    "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
constexpr std::string_view kHashF =
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

constexpr StaticModelPlan<6> kSecondPlan{
    .id = "second-generated-plan",
    .family = "second-hybrid-moe",
    .package_sha256 = kHashA,
    .artifact =
        {
            .id = "second-artifact",
            .model_type = "second_hybrid_moe",
            .format = "safetensors",
            .source_repository = "local",
            .source_revision = "frozen",
            .manifest_sha256 = kHashB,
            .tensor_count = 1,
            .tensor_bytes = 1,
            .file_count = 1,
            .weight_file_count = 1,
        },
    .dimensions = {.hidden = 1024, .vocabulary = 32768},
    .attention = {.query_heads = 8, .key_value_heads = 2, .head_dimension = 128},
    .gated_delta = {.recurrent_heads = 16, .state_dimension = 64},
    .mixture_of_experts = {.experts = 32, .active_experts = 2, .expert_dimension = 256},
    .weights = {.format = WeightFormat::AffineQ4, .group_size = 32},
    .tokenizer =
        {
            .kind = TokenizerKind::ByteLevelBpe,
            .normalization = TokenizerNormalization::Nfc,
            .pretokenizer = TokenizerPretokenizer::QwenRegexByteLevelV1,
            .decoder = TokenizerDecoder::ByteLevel,
            .template_kind = ChatTemplateKind::Qwen36TextV1,
            .data_path = "second-tokenizer.json",
            .data_sha256 = kHashC,
            .data_size_bytes = 1,
            .config_path = "second-tokenizer-config.json",
            .config_sha256 = kHashD,
            .config_size_bytes = 1,
            .template_path = "second-template.jinja",
            .template_sha256 = kHashE,
            .template_size_bytes = 1,
            .split_pattern = "bounded",
            .vocabulary = 32768,
            .populated_vocabulary = 32768,
            .maximum_context = 65536,
            .end_of_text_id = 1,
            .message_start_id = 2,
            .message_end_id = 3,
            .thinking_start_id = 4,
            .thinking_end_id = 5,
            .padding_id = 0,
            .stop_token_ids = std::array<std::uint32_t,
                                         tatara::model::qwen36::kMaximumStopTokens>{3, 1},
            .stop_token_count = 2,
            .default_thinking = false,
        },
    .initial_serving_capacity = 4096,
    .layers =
        {
            LayerKind::GatedDelta,
            LayerKind::GatedDelta,
            LayerKind::GatedDelta,
            LayerKind::GatedDelta,
            LayerKind::FullAttention,
            LayerKind::FullAttention,
        },
};

static_assert(tatara::model::qwen36::valid_model_plan(kSecondPlan));
static_assert(tatara::runtime::valid_generated_model_plan(kSecondPlan) ==
              tatara::model::qwen36::valid_model_plan(kSecondPlan));
static_assert(
    tatara::runtime::make_generated_plan_admission_facts(kSecondPlan).valid());
static_assert(tatara::runtime::make_generated_plan_admission_facts<
                  tatara::model::qwen36::generated::kModelPlan>()
                  .valid());
static_assert(!std::is_aggregate_v<GeneratedPlanAdmissionFacts>);
static_assert(static_cast<std::uint8_t>(AdmissionBoundary::B_MINUS_1) == 0);
static_assert(static_cast<std::uint8_t>(AdmissionBoundary::B10_S) == 12);
static_assert(static_cast<std::uint8_t>(
                  AdmissionErrorKind::BOOTSTRAP_PROFILE_UNAVAILABLE) == 0);
static_assert(static_cast<std::uint8_t>(
                  AdmissionErrorKind::DEVICE_ACTION_PROHIBITED) == 6);
static_assert(static_cast<std::uint8_t>(
                  AdmissionErrorKind::UNSUPPORTED_GENERATED_PLAN) == 7);
static_assert(static_cast<std::uint8_t>(
                  AdmissionErrorKind::UNSUPPORTED_CONFIGURATION) == 8);
static_assert(static_cast<std::uint8_t>(
                  AdmissionErrorKind::SERVICE_START_AUTHORITY_INVALID) == 9);
static_assert(static_cast<std::uint8_t>(
                  AdmissionErrorKind::ADMISSION_IDENTITY_CHANGED) == 30);
static_assert(noexcept(tatara::runtime::plan_pre_device_admission(
    std::declval<const PreDeviceAdmissionInput&>())));

ImplementationProfileRecord
make_profile(const GeneratedPlanAdmissionFacts& facts, std::uint32_t context_bound,
             std::string_view profile_id, std::string_view profile_hash) {
    ImplementationProfileRecord profile;
    profile.profile_version = tatara::runtime::kImplementationProfileVersion;
    profile.profile_id.assign(profile_id);
    profile.profile_sha256.assign(profile_hash);
    profile.generated_plan_id = facts.plan_id();
    profile.generated_plan_package_sha256 = facts.package_sha256();
    profile.artifact_manifest_sha256 = facts.artifact_manifest_sha256();
    profile.weight_format = facts.weight_format();
    profile.weight_group_size = facts.weight_group_size();
    profile.context_representability_bound = context_bound;
    profile.minimum_hidden = facts.hidden();
    profile.maximum_hidden = facts.hidden();
    profile.hidden_multiple = facts.hidden();
    profile.maximum_vocabulary = facts.vocabulary();
    profile.minimum_query_heads = facts.query_heads();
    profile.maximum_query_heads = facts.query_heads();
    profile.query_heads_multiple = facts.query_heads();
    profile.minimum_key_value_heads = facts.key_value_heads();
    profile.maximum_key_value_heads = facts.key_value_heads();
    profile.key_value_heads_multiple = facts.key_value_heads();
    profile.query_to_key_value_ratio =
        facts.query_heads() / facts.key_value_heads();
    profile.minimum_head_dimension = facts.head_dimension();
    profile.maximum_head_dimension = facts.head_dimension();
    profile.head_dimension_multiple = facts.head_dimension();
    profile.minimum_recurrent_heads = facts.recurrent_heads();
    profile.maximum_recurrent_heads = facts.recurrent_heads();
    profile.recurrent_heads_multiple = facts.recurrent_heads();
    profile.minimum_state_dimension = facts.state_dimension();
    profile.maximum_state_dimension = facts.state_dimension();
    profile.state_dimension_multiple = facts.state_dimension();
    profile.minimum_experts = facts.experts();
    profile.maximum_experts = facts.experts();
    profile.experts_must_be_power_of_two = true;
    profile.minimum_active_experts = facts.active_experts();
    profile.maximum_active_experts_exclusive = facts.active_experts() + 1;
    profile.minimum_expert_dimension = facts.expert_dimension();
    profile.maximum_expert_dimension = facts.expert_dimension();
    profile.expert_dimension_multiple = facts.expert_dimension();
    profile.minimum_layer_count = facts.layer_count();
    profile.maximum_layer_count = facts.layer_count();
    profile.minimum_gated_delta_layers = facts.gated_delta_layers();
    profile.minimum_attention_layers = facts.attention_layers();
    return profile;
}

void fill_identity(PreDeviceAdmissionInput& input) {
    auto& identity = input.admission_identity;
    identity.generated_plan_hash = input.generated_plan.package_sha256();
    identity.artifact_manifest_hash =
        input.generated_plan.artifact_manifest_sha256();
    identity.model_image_layout_hash.assign(kHash1);
    identity.tokenizer_template_hashes.assign(kHash2);
    identity.source_commit.assign(kHash3);
    identity.binary_hash.assign(kHash4);
    identity.metallib_hash.assign(kHash5);
    identity.compiler_and_sdk_identity.assign("clang-sdk-source-only");
    identity.configuration_hash.assign(kHash6);
    identity.owner_registry_hash.assign(kHash7);
    identity.phase_graph_hash.assign(kHash8);
    identity.evidence_bundle_hash.assign(kHash9);
    identity.bootstrap_envelope_and_evidence_hash =
        input.bootstrap_evidence.envelope_sha256;
    identity.bootstrap_root_invocation_hash =
        input.bootstrap.root_invocation_hash;
    identity.bootstrap_profile_registry_hash =
        input.bootstrap_profile_registry.registry_sha256;
    identity.hardware_profile_hash.assign(kHashA);
    identity.os_build.assign("source-only-os");
    identity.device_registry_id.assign("source-only-device");
    identity.admission_contract_version =
        tatara::runtime::kAdmissionContractVersion;
    identity.frozen_c4_owner_contract_hash.assign(kHashB);
    identity.frozen_c4_owner_census_hash.assign(kHashC);
    identity.http_ingress_contract_hash.assign(kHashD);
    identity.service_contract_hash.assign(kHashE);
    identity.resource_registry_hash.assign(kHashF);
    identity.canonical_order_hash.assign(kHash1);
    identity.hardware_fact_api_ledger_hash.assign(kHash2);
    identity.implementation_profile_hash =
        input.implementation_profile.profile_sha256;
    identity.service_start_authority_scope =
        ServiceStartAuthority::SourceOnly;
}

template <std::size_t LayerCount>
PreDeviceAdmissionInput make_valid_input(
    const StaticModelPlan<LayerCount>& plan, std::uint32_t context,
    std::uint32_t implementation_bound, std::string_view profile_id,
    std::string_view profile_hash) {
    PreDeviceAdmissionInput input;
    input.bootstrap.bootstrap_contract_version =
        tatara::runtime::kBootstrapContractVersion;
    input.bootstrap.requested_profile_id.assign("bootstrap-m4pro-v1");
    input.bootstrap.root_invocation_hash.assign(kHash1);
    input.bootstrap.input_count =
        static_cast<std::uint16_t>(input.bootstrap.inputs.size());
    input.bootstrap.total_owned_bytes = 364;
    input.bootstrap.diagnostic_bytes = 64;
    constexpr std::array<std::string_view, 4> kPaths{
        "configuration.toml",
        "package.plan",
        "prepared.record",
        "evidence.record",
    };
    for (std::size_t index = 0; index < input.bootstrap.inputs.size(); ++index) {
        auto& record = input.bootstrap.inputs[index];
        record.kind =
            static_cast<tatara::runtime::BootstrapInputKind>(index);
        record.path.assign(kPaths[index]);
        record.declared_bytes = 50;
        record.acquired_bytes = 50;
    }

    auto& registry = input.bootstrap_profile_registry;
    registry.registry_version =
        tatara::runtime::kBootstrapProfileRegistryVersion;
    registry.registry_sha256.assign(kHash2);
    auto& selected = registry.profiles[0];
    selected.bootstrap_contract_version =
        tatara::runtime::kBootstrapContractVersion;
    selected.profile_id.assign("bootstrap-m4pro-v1");
    selected.envelope_sha256.assign(kHash3);
    selected.applicable = true;
    selected.maximum_input_count =
        static_cast<std::uint16_t>(input.bootstrap.inputs.size());
    selected.maximum_input_bytes = {100, 100, 100, 100};
    selected.fixed_owned_bytes = 100;
    selected.maximum_total_owned_bytes = 1000;
    selected.maximum_diagnostic_bytes = 128;

    auto& alternate = registry.profiles[1];
    alternate = selected;
    alternate.profile_id.assign("bootstrap-alternate-v1");
    alternate.envelope_sha256.assign(kHash4);

    auto& evidence = input.bootstrap_evidence;
    evidence.bootstrap_contract_version =
        tatara::runtime::kBootstrapContractVersion;
    evidence.profile_registry_version =
        tatara::runtime::kBootstrapProfileRegistryVersion;
    evidence.evidence_id.assign("bootstrap-evidence-v1");
    evidence.selected_profile_id = selected.profile_id;
    evidence.profile_registry_sha256 = registry.registry_sha256;
    evidence.envelope_sha256 = selected.envelope_sha256;
    evidence.complete = true;
    evidence.current = true;
    evidence.identity_matches = true;
    evidence.envelope_independently_admitted = true;

    input.generated_plan =
        tatara::runtime::make_generated_plan_admission_facts(plan);
    input.implementation_profile =
        make_profile(input.generated_plan, implementation_bound, profile_id,
                     profile_hash);
    input.request.configured_context_capacity = context;
    fill_identity(input);
    return input;
}

template <std::size_t LayerCount>
PreDeviceAdmissionInput make_valid_input(
    const StaticModelPlan<LayerCount>& plan, std::uint32_t context,
    std::uint32_t implementation_bound) {
    return make_valid_input(plan, context, implementation_bound,
                            "decode-only-exact-row-v1", kHash5);
}

bool is_error(const tatara::runtime::PreDeviceAdmissionResult& result,
              AdmissionErrorKind kind, AdmissionBoundary boundary) {
    return !result && result.error.primary.error_kind == kind &&
           result.error.primary.boundary == boundary;
}

int test_arithmetic() {
    using namespace tatara::runtime;
    constexpr std::uint64_t maximum =
        std::numeric_limits<std::uint64_t>::max();

    if (checked_u64_add(maximum - 1, 1).value != maximum ||
        checked_u64_add(maximum, 1) ||
        checked_u64_multiply(maximum, 1).value != maximum ||
        checked_u64_multiply(maximum, 2) ||
        checked_u64_multiply(0, maximum).value != 0 ||
        checked_u64_ceil_divide(0, 3).value != 0 ||
        checked_u64_ceil_divide(maximum, maximum).value != 1 ||
        checked_u64_ceil_divide(maximum, maximum - 1).value != 2 ||
        checked_u64_ceil_divide(1, 0) ||
        checked_u64_align_up(17, 8).value != 24 ||
        checked_u64_align_up(16, 8).value != 16 ||
        checked_u64_align_up(maximum, 2) ||
        checked_u64_align_up(1, 0) ||
        checked_u64_narrow<std::uint32_t>(
            std::numeric_limits<std::uint32_t>::max())
                .value != std::numeric_limits<std::uint32_t>::max() ||
        checked_u64_narrow<std::uint32_t>(
            std::uint64_t{std::numeric_limits<std::uint32_t>::max()} + 1)) {
        return 1;
    }
    return 0;
}

int test_fixed_names() {
    constexpr std::array<std::string_view, 31> kNames{
        "BOOTSTRAP_PROFILE_UNAVAILABLE",
        "BOOTSTRAP_PROFILE_MISMATCH",
        "BOOTSTRAP_INPUT_LIMIT",
        "BOOTSTRAP_INPUT_CHANGED",
        "BOOTSTRAP_EVIDENCE_INCOMPLETE",
        "BOOTSTRAP_LIMIT",
        "DEVICE_ACTION_PROHIBITED",
        "UNSUPPORTED_GENERATED_PLAN",
        "UNSUPPORTED_CONFIGURATION",
        "SERVICE_START_AUTHORITY_INVALID",
        "HARDWARE_FACT_UNAVAILABLE",
        "EXTERNAL_OCCUPANCY_UNAVAILABLE",
        "HARDWARE_STATE_CHANGED",
        "EVIDENCE_IDENTITY_MISMATCH",
        "EVIDENCE_EXPIRED",
        "LOWER_BOUND_NONFIT",
        "EVIDENCE_MISSING",
        "ADMISSION_ARITHMETIC_OVERFLOW",
        "SINGLE_BUFFER_LIMIT",
        "PROCESS_DESCRIPTOR_LIMIT",
        "LISTEN_BACKLOG_LIMIT",
        "THREAD_RESOURCE_LIMIT",
        "METAL_WORKING_SET_LIMIT",
        "UNIFIED_MEMORY_POLICY_LIMIT",
        "ALLOCATION_FAILURE",
        "RESOURCE_ACQUISITION_FAILURE",
        "OPAQUE_CONSTRUCTION_FAILURE",
        "LIFETIME_CONSERVATION_FAILURE",
        "RESOURCE_CONSERVATION_FAILURE",
        "UNWIND_FAILURE",
        "ADMISSION_IDENTITY_CHANGED",
    };
    for (std::size_t index = 0; index < kNames.size(); ++index) {
        if (tatara::runtime::admission_error_name(
                static_cast<AdmissionErrorKind>(index)) != kNames[index]) {
            return 1;
        }
    }
    return 0;
}

int test_bounded_representation_and_bootstrap_registry() {
    using tatara::model::qwen36::generated::kModelPlan;
    auto input = make_valid_input(kModelPlan, 16384, 262144);
    if (!tatara::runtime::plan_pre_device_admission(input)) {
        return 1;
    }

    auto missing = input;
    missing.bootstrap.requested_profile_id.assign("not-present");
    if (!is_error(tatara::runtime::plan_pre_device_admission(missing),
                  AdmissionErrorKind::BOOTSTRAP_PROFILE_UNAVAILABLE,
                  AdmissionBoundary::B_MINUS_1)) {
        return 2;
    }

    auto duplicate = input;
    duplicate.bootstrap_profile_registry.profiles[1].profile_id =
        duplicate.bootstrap_profile_registry.profiles[0].profile_id;
    if (!is_error(tatara::runtime::plan_pre_device_admission(duplicate),
                  AdmissionErrorKind::BOOTSTRAP_PROFILE_MISMATCH,
                  AdmissionBoundary::B_MINUS_1)) {
        return 3;
    }

    auto wrong_version = input;
    wrong_version.bootstrap_profile_registry.registry_version = 2;
    if (!is_error(tatara::runtime::plan_pre_device_admission(wrong_version),
                  AdmissionErrorKind::BOOTSTRAP_PROFILE_MISMATCH,
                  AdmissionBoundary::B_MINUS_1)) {
        return 4;
    }

    auto wrong_envelope = input;
    wrong_envelope.bootstrap_evidence.envelope_sha256.assign(kHashF);
    wrong_envelope.admission_identity.bootstrap_envelope_and_evidence_hash.assign(
        kHashF);
    if (!is_error(tatara::runtime::plan_pre_device_admission(wrong_envelope),
                  AdmissionErrorKind::BOOTSTRAP_EVIDENCE_INCOMPLETE,
                  AdmissionBoundary::B_MINUS_1)) {
        return 5;
    }

    auto forged_selection_metadata = input;
    forged_selection_metadata.bootstrap.requested_profile_id.size =
        std::numeric_limits<std::uint16_t>::max();
    forged_selection_metadata.bootstrap.requested_profile_id.complete = true;
    if (!forged_selection_metadata.bootstrap.requested_profile_id.view().empty() ||
        !is_error(
            tatara::runtime::plan_pre_device_admission(
                forged_selection_metadata),
            AdmissionErrorKind::BOOTSTRAP_PROFILE_UNAVAILABLE,
            AdmissionBoundary::B_MINUS_1)) {
        return 6;
    }

    auto forged_path_metadata = input;
    forged_path_metadata.bootstrap.inputs[0].path.size =
        std::numeric_limits<std::uint16_t>::max();
    forged_path_metadata.bootstrap.inputs[0].path.complete = true;
    if (!forged_path_metadata.bootstrap.inputs[0].path.view().empty() ||
        !is_error(tatara::runtime::plan_pre_device_admission(
                      forged_path_metadata),
                  AdmissionErrorKind::BOOTSTRAP_INPUT_LIMIT,
                  AdmissionBoundary::B_MINUS_1)) {
        return 7;
    }

    auto forged_evidence_metadata = input;
    forged_evidence_metadata.bootstrap_evidence.evidence_id.size =
        std::numeric_limits<std::uint16_t>::max();
    forged_evidence_metadata.bootstrap_evidence.evidence_id.complete = true;
    if (!is_error(tatara::runtime::plan_pre_device_admission(
                      forged_evidence_metadata),
                  AdmissionErrorKind::BOOTSTRAP_EVIDENCE_INCOMPLETE,
                  AdmissionBoundary::B_MINUS_1)) {
        return 8;
    }

    auto input_limit = input;
    input_limit.bootstrap.inputs[1].declared_bytes = 101;
    input_limit.bootstrap.inputs[1].acquired_bytes = 101;
    const auto input_limit_result =
        tatara::runtime::plan_pre_device_admission(input_limit);
    const auto& diagnostic = input_limit_result.error.primary;
    if (!is_error(input_limit_result, AdmissionErrorKind::BOOTSTRAP_INPUT_LIMIT,
                  AdmissionBoundary::B_MINUS_1) ||
        diagnostic.owner_id_if_any.view() != "bootstrap" ||
        diagnostic.phase_if_any.view() != "bootstrap" ||
        diagnostic.resource_id_if_any.view() != "bootstrap.input" ||
        diagnostic.array_index != 1 || !diagnostic.requested_bytes.present ||
        diagnostic.requested_bytes.value != 101 ||
        !diagnostic.byte_limit.present || diagnostic.byte_limit.value != 100 ||
        diagnostic.resource_kind != AdmissionResourceKind::Bytes ||
        !diagnostic.headroom_or_deficit.present ||
        !diagnostic.headroom_or_deficit.deficit ||
        diagnostic.headroom_or_deficit.magnitude != 1 ||
        diagnostic.hardware_fact_generation.present ||
        diagnostic.underlying_cause !=
            AdmissionUnderlyingCause::BootGateFailed ||
        diagnostic.underlying_errno.present || !diagnostic.retry_safe ||
        diagnostic.unwind_status != AdmissionUnwindStatus::NotRequired ||
        !(input_limit_result.error.admission_identity.configuration_hash ==
          input.admission_identity.configuration_hash)) {
        return 9;
    }

    auto changed = input;
    changed.bootstrap.inputs[0].unchanged = false;
    if (!is_error(tatara::runtime::plan_pre_device_admission(changed),
                  AdmissionErrorKind::BOOTSTRAP_INPUT_CHANGED,
                  AdmissionBoundary::B_MINUS_1)) {
        return 10;
    }

    auto overflow = input;
    overflow.bootstrap_profile_registry.profiles[0]
        .maximum_input_bytes[0] =
        std::numeric_limits<std::uint64_t>::max();
    overflow.bootstrap.inputs[0].declared_bytes =
        std::numeric_limits<std::uint64_t>::max();
    overflow.bootstrap.inputs[0].acquired_bytes =
        std::numeric_limits<std::uint64_t>::max();
    if (!is_error(tatara::runtime::plan_pre_device_admission(overflow),
                  AdmissionErrorKind::BOOTSTRAP_LIMIT,
                  AdmissionBoundary::B_MINUS_1)) {
        return 11;
    }

    auto prohibited = input;
    prohibited.request.requests_device_construction = true;
    if (!is_error(tatara::runtime::plan_pre_device_admission(prohibited),
                  AdmissionErrorKind::DEVICE_ACTION_PROHIBITED,
                  AdmissionBoundary::B_MINUS_1)) {
        return 12;
    }
    return 0;
}

int test_b0_provenance_and_profile_generality() {
    using tatara::model::qwen36::generated::kModelPlan;

    auto forged = make_valid_input(kModelPlan, 16384, 262144);
    forged.generated_plan = GeneratedPlanAdmissionFacts{};
    forged.implementation_profile.generated_plan_id = {};
    forged.implementation_profile.generated_plan_package_sha256 = {};
    forged.implementation_profile.artifact_manifest_sha256 = {};
    if (!is_error(tatara::runtime::plan_pre_device_admission(forged),
                  AdmissionErrorKind::UNSUPPORTED_GENERATED_PLAN,
                  AdmissionBoundary::B0)) {
        return 1;
    }

    auto identity = make_valid_input(kModelPlan, 16384, 262144);
    identity.implementation_profile.generated_plan_id.assign("wrong-plan");
    if (!is_error(tatara::runtime::plan_pre_device_admission(identity),
                  AdmissionErrorKind::UNSUPPORTED_GENERATED_PLAN,
                  AdmissionBoundary::B0)) {
        return 2;
    }

    auto profile_constraint = make_valid_input(kModelPlan, 16384, 262144);
    ++profile_constraint.implementation_profile.minimum_hidden;
    if (!is_error(
            tatara::runtime::plan_pre_device_admission(profile_constraint),
            AdmissionErrorKind::UNSUPPORTED_GENERATED_PLAN,
            AdmissionBoundary::B0)) {
        return 3;
    }

    const GeneratedPlanAdmissionFacts second_facts =
        tatara::runtime::make_generated_plan_admission_facts(kSecondPlan);
    if (!second_facts.valid() || second_facts.hidden() != 1024 ||
        second_facts.query_heads() != 8 ||
        second_facts.head_dimension() != 128 ||
        second_facts.state_dimension() != 64 ||
        second_facts.experts() != 32 ||
        second_facts.expert_dimension() != 256 ||
        second_facts.weight_group_size() != 32 ||
        second_facts.maximum_context() != 65536 ||
        second_facts.layer_count() != 6 ||
        second_facts.gated_delta_layers() != 4 ||
        second_facts.attention_layers() != 2) {
        return 4;
    }
    const auto second =
        make_valid_input(kSecondPlan, 65536, 65536,
                         "second-exact-row-profile", kHash6);
    if (!tatara::runtime::plan_pre_device_admission(second)) {
        return 5;
    }
    return 0;
}

int test_b1_error_surface_and_no_authority() {
    using tatara::model::qwen36::generated::kModelPlan;
    auto input = make_valid_input(kModelPlan, 100000, 262144);
    const std::uint64_t allocations_before = g_allocation_count;
    const auto source_bound_only =
        tatara::runtime::plan_pre_device_admission(input);
    if (!source_bound_only || g_allocation_count != allocations_before ||
        source_bound_only.plan.allocation_authorized ||
        source_bound_only.plan.device_action_authorized ||
        source_bound_only.plan.command_authorized ||
        source_bound_only.plan.listener_authorized ||
        source_bound_only.plan.service_start_authorized ||
        source_bound_only.plan.completed_through != AdmissionBoundary::B1) {
        return 1;
    }

    auto exact_maximum = make_valid_input(kModelPlan, 262144, 262144);
    if (!tatara::runtime::plan_pre_device_admission(exact_maximum)) {
        return 2;
    }

    auto implementation_over = input;
    implementation_over.implementation_profile
        .context_representability_bound = 99999;
    const auto implementation_over_result =
        tatara::runtime::plan_pre_device_admission(implementation_over);
    const auto& context_error = implementation_over_result.error.primary;
    if (!is_error(implementation_over_result,
                  AdmissionErrorKind::UNSUPPORTED_CONFIGURATION,
                  AdmissionBoundary::B1) ||
        context_error.owner_id_if_any.view() !=
            "configuration.context.implementation" ||
        context_error.phase_if_any.view() != "budgeted" ||
        !context_error.requested_resource_count.present ||
        context_error.requested_resource_count.value != 100000 ||
        !context_error.resource_limit.present ||
        context_error.resource_limit.value != 99999 ||
        context_error.resource_kind != AdmissionResourceKind::Tokens ||
        !context_error.headroom_or_deficit.present ||
        !context_error.headroom_or_deficit.deficit ||
        context_error.headroom_or_deficit.magnitude != 1) {
        return 3;
    }

    auto widened = input;
    widened.request.physical_slot_count = 2;
    widened.request.maximum_concurrent_requests = 2;
    widened.request.queue_depth = 1;
    widened.request.request_deadline_milliseconds = 1;
    widened.request.drain_timeout_milliseconds = 1;
    widened.request.prompt_cache = true;
    widened.request.composed_prefill = true;
    widened.request.qgemm_policy = QgemmExecutionPolicy::DenseMatrix;
    widened.request.service_start_authority =
        ServiceStartAuthority::Production;
    widened.request.gpu_halt_active = false;
    widened.request.requests_live_device_facts = true;
    widened.request.requests_listener = true;
    const auto widened_result =
        tatara::runtime::plan_pre_device_admission(widened);
    if (!is_error(widened_result,
                  AdmissionErrorKind::UNSUPPORTED_CONFIGURATION,
                  AdmissionBoundary::B1) ||
        widened_result.error.total_observed_count != 10 ||
        widened_result.error.secondary_diagnostic_count !=
            tatara::runtime::kAdmissionSecondaryDiagnosticCapacity ||
        !widened_result.error.secondary_diagnostics_truncated) {
        return 4;
    }
    for (std::size_t index = 1;
         index < widened_result.error.secondary_diagnostic_count; ++index) {
        const auto& previous =
            widened_result.error.secondary_diagnostics[index - 1];
        const auto& current =
            widened_result.error.secondary_diagnostics[index];
        if (previous.error_kind == current.error_kind &&
            previous.owner_id_if_any.view() >
                current.owner_id_if_any.view()) {
            return 5;
        }
    }

    auto authority = input;
    authority.request.service_start_authority =
        ServiceStartAuthority::EphemeralLab;
    if (!is_error(tatara::runtime::plan_pre_device_admission(authority),
                  AdmissionErrorKind::SERVICE_START_AUTHORITY_INVALID,
                  AdmissionBoundary::B1)) {
        return 6;
    }

    auto source_listener = input;
    source_listener.request.gpu_halt_active = false;
    source_listener.request.requests_listener = true;
    if (!is_error(tatara::runtime::plan_pre_device_admission(source_listener),
                  AdmissionErrorKind::SERVICE_START_AUTHORITY_INVALID,
                  AdmissionBoundary::B1)) {
        return 7;
    }
    return 0;
}

} // namespace

void* operator new(std::size_t size) {
    ++g_allocation_count;
    if (void* memory = std::malloc(size)) {
        return memory;
    }
    std::abort();
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void* operator new[](std::size_t size) {
    ++g_allocation_count;
    if (void* memory = std::malloc(size)) {
        return memory;
    }
    std::abort();
}

void operator delete[](void* memory) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

void operator delete[](void* memory, std::size_t) noexcept {
    std::free(memory);
}

int main() {
    if (const int result = test_arithmetic()) {
        return 10 + result;
    }
    if (const int result = test_fixed_names()) {
        return 20 + result;
    }
    if (const int result =
            test_bounded_representation_and_bootstrap_registry()) {
        return 30 + result;
    }
    if (const int result = test_b0_provenance_and_profile_generality()) {
        return 50 + result;
    }
    if (const int result = test_b1_error_surface_and_no_authority()) {
        return 70 + result;
    }
    return 0;
}
