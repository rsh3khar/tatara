#include "tatara/runtime/startup_admission.h"

#include "tatara/runtime/checked_arithmetic.h"

#include <array>

namespace tatara::runtime {
namespace {

template <std::size_t Capacity>
constexpr bool nonempty_text(const BoundedAdmissionText<Capacity>& value) noexcept {
    return value.well_formed() && !value.empty();
}

template <std::size_t Capacity>
constexpr bool sha256_text(const BoundedAdmissionText<Capacity>& value) noexcept {
    if (!value.well_formed() || value.size != 64) {
        return false;
    }
    for (const char character : value.view()) {
        const bool decimal = character >= '0' && character <= '9';
        const bool lowercase_hex = character >= 'a' && character <= 'f';
        if (!decimal && !lowercase_hex) {
            return false;
        }
    }
    return true;
}

template <std::size_t Capacity>
constexpr int compare_text(const BoundedAdmissionText<Capacity>& left,
                           const BoundedAdmissionText<Capacity>& right) noexcept {
    const std::string_view left_view = left.view();
    const std::string_view right_view = right.view();
    const std::size_t common =
        left_view.size() < right_view.size() ? left_view.size() : right_view.size();
    for (std::size_t index = 0; index < common; ++index) {
        if (left_view[index] < right_view[index]) {
            return -1;
        }
        if (left_view[index] > right_view[index]) {
            return 1;
        }
    }
    if (left_view.size() < right_view.size()) {
        return -1;
    }
    if (left_view.size() > right_view.size()) {
        return 1;
    }
    return 0;
}

constexpr bool diagnostic_less(const AdmissionDiagnostic& left,
                               const AdmissionDiagnostic& right) noexcept {
    if (left.boundary != right.boundary) {
        return static_cast<std::uint8_t>(left.boundary) <
               static_cast<std::uint8_t>(right.boundary);
    }
    if (left.error_kind != right.error_kind) {
        return static_cast<std::uint8_t>(left.error_kind) <
               static_cast<std::uint8_t>(right.error_kind);
    }
    const int owner_order = compare_text(left.owner_id_if_any, right.owner_id_if_any);
    if (owner_order != 0) {
        return owner_order < 0;
    }
    const int phase_order = compare_text(left.phase_if_any, right.phase_if_any);
    if (phase_order != 0) {
        return phase_order < 0;
    }
    const int evidence_order =
        compare_text(left.unknown_or_evidence_id_if_any,
                     right.unknown_or_evidence_id_if_any);
    if (evidence_order != 0) {
        return evidence_order < 0;
    }
    const int resource_order =
        compare_text(left.resource_id_if_any, right.resource_id_if_any);
    if (resource_order != 0) {
        return resource_order < 0;
    }
    return left.array_index < right.array_index;
}

constexpr AdmissionOptionalUnsigned optional_unsigned(std::uint64_t value) noexcept {
    return {.present = true, .value = value};
}

constexpr AdmissionOptionalSignedMagnitude
headroom_or_deficit(std::uint64_t requested, std::uint64_t limit) noexcept {
    if (requested > limit) {
        return {.present = true, .deficit = true, .magnitude = requested - limit};
    }
    return {.present = true, .deficit = false, .magnitude = limit - requested};
}

AdmissionDiagnostic make_diagnostic(
    AdmissionErrorKind kind, AdmissionBoundary boundary, std::string_view owner_id,
    std::string_view phase, std::string_view evidence_id, std::string_view resource_id,
    std::uint16_t array_index, AdmissionUnderlyingCause cause) noexcept {
    AdmissionDiagnostic diagnostic;
    diagnostic.error_kind = kind;
    diagnostic.boundary = boundary;
    diagnostic.owner_id_if_any.assign(owner_id);
    diagnostic.phase_if_any.assign(phase);
    diagnostic.unknown_or_evidence_id_if_any.assign(evidence_id);
    diagnostic.resource_id_if_any.assign(resource_id);
    diagnostic.array_index = array_index;
    diagnostic.underlying_cause = cause;
    diagnostic.retry_safe = true;
    diagnostic.unwind_status = AdmissionUnwindStatus::NotRequired;
    return diagnostic;
}

struct DiagnosticCollector {
    static constexpr std::size_t kCapacity = kAdmissionSecondaryDiagnosticCapacity + 1;
    std::array<AdmissionDiagnostic, kCapacity> diagnostics{};
    std::uint16_t retained{0};
    std::uint16_t observed{0};

    void add(const AdmissionDiagnostic& diagnostic) noexcept {
        if (observed != UINT16_MAX) {
            ++observed;
        }
        if (retained < diagnostics.size()) {
            diagnostics[retained++] = diagnostic;
            return;
        }
        auto worst = diagnostics.begin();
        for (auto it = diagnostics.begin() + 1; it != diagnostics.end(); ++it) {
            if (diagnostic_less(*worst, *it)) {
                worst = it;
            }
        }
        if (diagnostic_less(diagnostic, *worst)) {
            *worst = diagnostic;
        }
    }

    [[nodiscard]] AdmissionError finish(const AdmissionIdentity& identity) noexcept {
        for (std::size_t index = 1; index < retained; ++index) {
            const AdmissionDiagnostic value = diagnostics[index];
            std::size_t position = index;
            while (position != 0 && diagnostic_less(value, diagnostics[position - 1])) {
                diagnostics[position] = diagnostics[position - 1];
                --position;
            }
            diagnostics[position] = value;
        }
        AdmissionError error;
        error.admission_identity = identity;
        error.total_observed_count = observed;
        error.secondary_diagnostics_truncated = observed > diagnostics.size();
        if (retained == 0) {
            return error;
        }
        error.primary = diagnostics[0];
        error.secondary_diagnostic_count =
            static_cast<std::uint16_t>(retained > 0 ? retained - 1 : 0);
        for (std::size_t index = 1; index < retained; ++index) {
            error.secondary_diagnostics[index - 1] = diagnostics[index];
        }
        return error;
    }
};

bool any_prohibited_action(const PreDeviceAdmissionRequest& request) noexcept {
    return request.requests_live_device_facts || request.requests_device_construction ||
           request.requests_model_execution || request.requests_command_authorization ||
           request.requests_listener || request.requests_service_start;
}

bool valid_relative_path(std::string_view path) noexcept {
    if (path.empty() || path.front() == '/') {
        return false;
    }
    std::size_t component_begin = 0;
    for (std::size_t index = 0; index <= path.size(); ++index) {
        if (index != path.size() && path[index] != '/') {
            if (path[index] == '\0' || path[index] == '\r' || path[index] == '\n') {
                return false;
            }
            continue;
        }
        const std::size_t component_size = index - component_begin;
        if (component_size == 0 ||
            (component_size == 1 && path[component_begin] == '.') ||
            (component_size == 2 && path[component_begin] == '.' &&
             path[component_begin + 1] == '.')) {
            return false;
        }
        component_begin = index + 1;
    }
    return true;
}

bool valid_bootstrap_profile(const BootstrapProfileRecord& profile) noexcept {
    if (profile.bootstrap_contract_version != kBootstrapContractVersion ||
        !nonempty_text(profile.profile_id) || !sha256_text(profile.envelope_sha256) ||
        !profile.applicable ||
        profile.maximum_input_count != kBootstrapInputCapacity ||
        profile.maximum_total_owned_bytes == 0 ||
        profile.maximum_diagnostic_bytes == 0) {
        return false;
    }
    for (const std::uint64_t maximum : profile.maximum_input_bytes) {
        if (maximum == 0) {
            return false;
        }
    }
    return true;
}

bool valid_admission_identity(const AdmissionIdentity& identity) noexcept {
    return sha256_text(identity.generated_plan_hash) &&
           sha256_text(identity.artifact_manifest_hash) &&
           sha256_text(identity.model_image_layout_hash) &&
           nonempty_text(identity.tokenizer_template_hashes) &&
           nonempty_text(identity.source_commit) && sha256_text(identity.binary_hash) &&
           sha256_text(identity.metallib_hash) &&
           nonempty_text(identity.compiler_and_sdk_identity) &&
           sha256_text(identity.configuration_hash) &&
           sha256_text(identity.owner_registry_hash) &&
           sha256_text(identity.phase_graph_hash) &&
           sha256_text(identity.evidence_bundle_hash) &&
           sha256_text(identity.bootstrap_envelope_and_evidence_hash) &&
           sha256_text(identity.bootstrap_root_invocation_hash) &&
           sha256_text(identity.bootstrap_profile_registry_hash) &&
           sha256_text(identity.hardware_profile_hash) &&
           nonempty_text(identity.os_build) && nonempty_text(identity.device_registry_id) &&
           identity.admission_contract_version == kAdmissionContractVersion &&
           sha256_text(identity.frozen_c4_owner_contract_hash) &&
           sha256_text(identity.frozen_c4_owner_census_hash) &&
           sha256_text(identity.http_ingress_contract_hash) &&
           sha256_text(identity.service_contract_hash) &&
           sha256_text(identity.resource_registry_hash) &&
           sha256_text(identity.canonical_order_hash) &&
           sha256_text(identity.hardware_fact_api_ledger_hash) &&
           sha256_text(identity.implementation_profile_hash);
}

constexpr bool valid_range(std::uint32_t minimum, std::uint32_t maximum,
                           std::uint32_t multiple) noexcept {
    return minimum != 0 && maximum >= minimum && multiple != 0;
}

constexpr bool in_range_and_multiple(std::uint32_t value, std::uint32_t minimum,
                                     std::uint32_t maximum,
                                     std::uint32_t multiple) noexcept {
    return value >= minimum && value <= maximum && multiple != 0 &&
           value % multiple == 0;
}

bool valid_implementation_profile(const ImplementationProfileRecord& profile) noexcept {
    return profile.profile_version == kImplementationProfileVersion &&
           nonempty_text(profile.profile_id) && sha256_text(profile.profile_sha256) &&
           nonempty_text(profile.generated_plan_id) &&
           sha256_text(profile.generated_plan_package_sha256) &&
           sha256_text(profile.artifact_manifest_sha256) &&
           profile.weight_group_size != 0 &&
           profile.context_representability_bound != 0 &&
           valid_range(profile.minimum_hidden, profile.maximum_hidden,
                       profile.hidden_multiple) &&
           profile.maximum_vocabulary != 0 &&
           valid_range(profile.minimum_query_heads, profile.maximum_query_heads,
                       profile.query_heads_multiple) &&
           valid_range(profile.minimum_key_value_heads,
                       profile.maximum_key_value_heads,
                       profile.key_value_heads_multiple) &&
           profile.query_to_key_value_ratio != 0 &&
           valid_range(profile.minimum_head_dimension,
                       profile.maximum_head_dimension,
                       profile.head_dimension_multiple) &&
           valid_range(profile.minimum_recurrent_heads,
                       profile.maximum_recurrent_heads,
                       profile.recurrent_heads_multiple) &&
           valid_range(profile.minimum_state_dimension,
                       profile.maximum_state_dimension,
                       profile.state_dimension_multiple) &&
           profile.minimum_experts != 0 &&
           profile.maximum_experts >= profile.minimum_experts &&
           profile.minimum_active_experts != 0 &&
           profile.maximum_active_experts_exclusive >
               profile.minimum_active_experts &&
           valid_range(profile.minimum_expert_dimension,
                       profile.maximum_expert_dimension,
                       profile.expert_dimension_multiple) &&
           profile.minimum_layer_count != 0 &&
           profile.maximum_layer_count >= profile.minimum_layer_count &&
           profile.minimum_gated_delta_layers != 0 &&
           profile.minimum_attention_layers != 0;
}

bool implementation_accepts_plan(const ImplementationProfileRecord& profile,
                                 const GeneratedPlanAdmissionFacts& plan) noexcept {
    if (!valid_implementation_profile(profile) || !plan.copied_completely() ||
        !plan.valid() || !(plan.plan_id() == profile.generated_plan_id) ||
        !(plan.package_sha256() == profile.generated_plan_package_sha256) ||
        !(plan.artifact_manifest_sha256() == profile.artifact_manifest_sha256) ||
        plan.weight_format() != profile.weight_format ||
        plan.weight_group_size() != profile.weight_group_size ||
        !in_range_and_multiple(plan.hidden(), profile.minimum_hidden,
                               profile.maximum_hidden, profile.hidden_multiple) ||
        plan.vocabulary() > profile.maximum_vocabulary ||
        !in_range_and_multiple(plan.query_heads(), profile.minimum_query_heads,
                               profile.maximum_query_heads,
                               profile.query_heads_multiple) ||
        !in_range_and_multiple(plan.key_value_heads(),
                               profile.minimum_key_value_heads,
                               profile.maximum_key_value_heads,
                               profile.key_value_heads_multiple) ||
        plan.key_value_heads() == 0 ||
        plan.query_heads() % plan.key_value_heads() != 0 ||
        plan.query_heads() / plan.key_value_heads() !=
            profile.query_to_key_value_ratio ||
        !in_range_and_multiple(plan.head_dimension(),
                               profile.minimum_head_dimension,
                               profile.maximum_head_dimension,
                               profile.head_dimension_multiple) ||
        !in_range_and_multiple(plan.recurrent_heads(),
                               profile.minimum_recurrent_heads,
                               profile.maximum_recurrent_heads,
                               profile.recurrent_heads_multiple) ||
        !in_range_and_multiple(plan.state_dimension(),
                               profile.minimum_state_dimension,
                               profile.maximum_state_dimension,
                               profile.state_dimension_multiple) ||
        plan.experts() < profile.minimum_experts ||
        plan.experts() > profile.maximum_experts ||
        plan.active_experts() < profile.minimum_active_experts ||
        plan.active_experts() >= profile.maximum_active_experts_exclusive ||
        plan.active_experts() > plan.experts() ||
        !in_range_and_multiple(plan.expert_dimension(),
                               profile.minimum_expert_dimension,
                               profile.maximum_expert_dimension,
                               profile.expert_dimension_multiple) ||
        plan.layer_count() < profile.minimum_layer_count ||
        plan.layer_count() > profile.maximum_layer_count ||
        plan.gated_delta_layers() < profile.minimum_gated_delta_layers ||
        plan.attention_layers() < profile.minimum_attention_layers) {
        return false;
    }
    return !profile.experts_must_be_power_of_two ||
           (plan.experts() != 0 &&
            (plan.experts() & (plan.experts() - 1U)) == 0);
}

bool identity_matches_inputs(const AdmissionIdentity& identity,
                             const PreDeviceAdmissionInput& input) noexcept {
    return identity.generated_plan_hash == input.generated_plan.package_sha256() &&
           identity.artifact_manifest_hash ==
               input.generated_plan.artifact_manifest_sha256() &&
           identity.bootstrap_envelope_and_evidence_hash ==
               input.bootstrap_evidence.envelope_sha256 &&
           identity.bootstrap_root_invocation_hash ==
               input.bootstrap.root_invocation_hash &&
           identity.bootstrap_profile_registry_hash ==
               input.bootstrap_profile_registry.registry_sha256 &&
           identity.implementation_profile_hash ==
               input.implementation_profile.profile_sha256;
}

PreDeviceAdmissionResult refuse(const PreDeviceAdmissionInput& input,
                                DiagnosticCollector& diagnostics) noexcept {
    PreDeviceAdmissionResult result;
    result.plan.input = input;
    result.error = diagnostics.finish(input.admission_identity);
    return result;
}

} // namespace

std::string_view admission_error_name(AdmissionErrorKind kind) noexcept {
    switch (kind) {
    case AdmissionErrorKind::BOOTSTRAP_PROFILE_UNAVAILABLE:
        return "BOOTSTRAP_PROFILE_UNAVAILABLE";
    case AdmissionErrorKind::BOOTSTRAP_PROFILE_MISMATCH:
        return "BOOTSTRAP_PROFILE_MISMATCH";
    case AdmissionErrorKind::BOOTSTRAP_INPUT_LIMIT:
        return "BOOTSTRAP_INPUT_LIMIT";
    case AdmissionErrorKind::BOOTSTRAP_INPUT_CHANGED:
        return "BOOTSTRAP_INPUT_CHANGED";
    case AdmissionErrorKind::BOOTSTRAP_EVIDENCE_INCOMPLETE:
        return "BOOTSTRAP_EVIDENCE_INCOMPLETE";
    case AdmissionErrorKind::BOOTSTRAP_LIMIT:
        return "BOOTSTRAP_LIMIT";
    case AdmissionErrorKind::DEVICE_ACTION_PROHIBITED:
        return "DEVICE_ACTION_PROHIBITED";
    case AdmissionErrorKind::UNSUPPORTED_GENERATED_PLAN:
        return "UNSUPPORTED_GENERATED_PLAN";
    case AdmissionErrorKind::UNSUPPORTED_CONFIGURATION:
        return "UNSUPPORTED_CONFIGURATION";
    case AdmissionErrorKind::SERVICE_START_AUTHORITY_INVALID:
        return "SERVICE_START_AUTHORITY_INVALID";
    case AdmissionErrorKind::HARDWARE_FACT_UNAVAILABLE:
        return "HARDWARE_FACT_UNAVAILABLE";
    case AdmissionErrorKind::EXTERNAL_OCCUPANCY_UNAVAILABLE:
        return "EXTERNAL_OCCUPANCY_UNAVAILABLE";
    case AdmissionErrorKind::HARDWARE_STATE_CHANGED:
        return "HARDWARE_STATE_CHANGED";
    case AdmissionErrorKind::EVIDENCE_IDENTITY_MISMATCH:
        return "EVIDENCE_IDENTITY_MISMATCH";
    case AdmissionErrorKind::EVIDENCE_EXPIRED:
        return "EVIDENCE_EXPIRED";
    case AdmissionErrorKind::LOWER_BOUND_NONFIT:
        return "LOWER_BOUND_NONFIT";
    case AdmissionErrorKind::EVIDENCE_MISSING:
        return "EVIDENCE_MISSING";
    case AdmissionErrorKind::ADMISSION_ARITHMETIC_OVERFLOW:
        return "ADMISSION_ARITHMETIC_OVERFLOW";
    case AdmissionErrorKind::SINGLE_BUFFER_LIMIT:
        return "SINGLE_BUFFER_LIMIT";
    case AdmissionErrorKind::PROCESS_DESCRIPTOR_LIMIT:
        return "PROCESS_DESCRIPTOR_LIMIT";
    case AdmissionErrorKind::LISTEN_BACKLOG_LIMIT:
        return "LISTEN_BACKLOG_LIMIT";
    case AdmissionErrorKind::THREAD_RESOURCE_LIMIT:
        return "THREAD_RESOURCE_LIMIT";
    case AdmissionErrorKind::METAL_WORKING_SET_LIMIT:
        return "METAL_WORKING_SET_LIMIT";
    case AdmissionErrorKind::UNIFIED_MEMORY_POLICY_LIMIT:
        return "UNIFIED_MEMORY_POLICY_LIMIT";
    case AdmissionErrorKind::ALLOCATION_FAILURE:
        return "ALLOCATION_FAILURE";
    case AdmissionErrorKind::RESOURCE_ACQUISITION_FAILURE:
        return "RESOURCE_ACQUISITION_FAILURE";
    case AdmissionErrorKind::OPAQUE_CONSTRUCTION_FAILURE:
        return "OPAQUE_CONSTRUCTION_FAILURE";
    case AdmissionErrorKind::LIFETIME_CONSERVATION_FAILURE:
        return "LIFETIME_CONSERVATION_FAILURE";
    case AdmissionErrorKind::RESOURCE_CONSERVATION_FAILURE:
        return "RESOURCE_CONSERVATION_FAILURE";
    case AdmissionErrorKind::UNWIND_FAILURE:
        return "UNWIND_FAILURE";
    case AdmissionErrorKind::ADMISSION_IDENTITY_CHANGED:
        return "ADMISSION_IDENTITY_CHANGED";
    case AdmissionErrorKind::NONE:
        return "NONE";
    }
    return "UNKNOWN_ADMISSION_ERROR";
}

PreDeviceAdmissionResult
plan_pre_device_admission(const PreDeviceAdmissionInput& input) noexcept {
    DiagnosticCollector diagnostics;
    const BootstrapRootRecord& root = input.bootstrap;
    const BootstrapProfileRegistryRecord& registry =
        input.bootstrap_profile_registry;
    const BootstrapEvidenceRecord& evidence = input.bootstrap_evidence;

    const bool requested_profile_valid =
        nonempty_text(root.requested_profile_id);
    std::size_t matching_profile_count = 0;
    const BootstrapProfileRecord* selected_profile = nullptr;
    if (requested_profile_valid) {
        for (const BootstrapProfileRecord& candidate : registry.profiles) {
            if (candidate.profile_id == root.requested_profile_id) {
                ++matching_profile_count;
                selected_profile = &candidate;
            }
        }
    }
    if (!requested_profile_valid || matching_profile_count == 0) {
        diagnostics.add(make_diagnostic(
            AdmissionErrorKind::BOOTSTRAP_PROFILE_UNAVAILABLE,
            AdmissionBoundary::B_MINUS_1, "bootstrap", "bootstrap",
            "bootstrap.profile_registry", "bootstrap.profile_selection",
            kNoAdmissionDiagnosticIndex,
            AdmissionUnderlyingCause::InvalidProfile));
    }

    bool registry_valid =
        registry.registry_version == kBootstrapProfileRegistryVersion &&
        sha256_text(registry.registry_sha256);
    for (std::size_t index = 0; index < registry.profiles.size(); ++index) {
        registry_valid =
            valid_bootstrap_profile(registry.profiles[index]) && registry_valid;
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (registry.profiles[index].profile_id ==
                registry.profiles[previous].profile_id) {
                registry_valid = false;
            }
        }
    }
    if (!registry_valid || matching_profile_count > 1 ||
        (matching_profile_count == 1 &&
         !valid_bootstrap_profile(*selected_profile)) ||
        root.bootstrap_contract_version != kBootstrapContractVersion) {
        diagnostics.add(make_diagnostic(
            AdmissionErrorKind::BOOTSTRAP_PROFILE_MISMATCH,
            AdmissionBoundary::B_MINUS_1, "bootstrap", "bootstrap",
            "bootstrap.profile_registry", "bootstrap.profile_selection",
            kNoAdmissionDiagnosticIndex,
            AdmissionUnderlyingCause::InvalidProfile));
    }
    if (matching_profile_count != 1) {
        selected_profile = nullptr;
    }

    const std::size_t bounded_input_count =
        root.input_count < root.inputs.size() ? root.input_count
                                              : root.inputs.size();
    std::uint64_t acquired_input_bytes = 0;
    bool total_overflow = false;
    for (std::size_t index = 0; index < bounded_input_count; ++index) {
        const BootstrapInputRecord& record = root.inputs[index];
        const std::uint64_t input_limit =
            selected_profile == nullptr
                ? 0
                : selected_profile->maximum_input_bytes[index];
        const bool record_invalid =
            record.kind != static_cast<BootstrapInputKind>(index) ||
            !record.path.well_formed() ||
            !valid_relative_path(record.path.view()) ||
            !record.supported_object_type || !record.read_complete ||
            record.declared_bytes == 0 ||
            (selected_profile != nullptr &&
             record.declared_bytes > input_limit) ||
            record.acquired_bytes != record.declared_bytes;
        if (record_invalid) {
            AdmissionDiagnostic diagnostic = make_diagnostic(
                AdmissionErrorKind::BOOTSTRAP_INPUT_LIMIT,
                AdmissionBoundary::B_MINUS_1, "bootstrap", "bootstrap",
                evidence.evidence_id.view(), "bootstrap.input",
                static_cast<std::uint16_t>(index),
                AdmissionUnderlyingCause::BootGateFailed);
            diagnostic.requested_bytes =
                optional_unsigned(record.declared_bytes);
            if (selected_profile != nullptr) {
                diagnostic.byte_limit = optional_unsigned(input_limit);
                diagnostic.headroom_or_deficit =
                    headroom_or_deficit(record.declared_bytes, input_limit);
            }
            diagnostic.resource_kind = AdmissionResourceKind::Bytes;
            diagnostics.add(diagnostic);
        }
        if (!record.unchanged) {
            AdmissionDiagnostic diagnostic = make_diagnostic(
                AdmissionErrorKind::BOOTSTRAP_INPUT_CHANGED,
                AdmissionBoundary::B_MINUS_1, "bootstrap", "bootstrap",
                evidence.evidence_id.view(), "bootstrap.input",
                static_cast<std::uint16_t>(index),
                AdmissionUnderlyingCause::IdentityInvalid);
            diagnostic.requested_bytes =
                optional_unsigned(record.acquired_bytes);
            diagnostic.byte_limit = optional_unsigned(record.declared_bytes);
            diagnostic.headroom_or_deficit =
                headroom_or_deficit(record.acquired_bytes,
                                    record.declared_bytes);
            diagnostic.resource_kind = AdmissionResourceKind::Bytes;
            diagnostics.add(diagnostic);
        }
        const CheckedU64 next_total =
            checked_u64_add(acquired_input_bytes, record.acquired_bytes);
        if (!next_total) {
            total_overflow = true;
        } else {
            acquired_input_bytes = next_total.value;
        }
    }

    const bool evidence_valid =
        evidence.bootstrap_contract_version == kBootstrapContractVersion &&
        evidence.profile_registry_version ==
            kBootstrapProfileRegistryVersion &&
        nonempty_text(evidence.evidence_id) &&
        nonempty_text(evidence.selected_profile_id) &&
        sha256_text(evidence.profile_registry_sha256) &&
        sha256_text(evidence.envelope_sha256) && evidence.complete &&
        evidence.current && evidence.identity_matches &&
        evidence.envelope_independently_admitted &&
        evidence.profile_registry_sha256 == registry.registry_sha256 &&
        selected_profile != nullptr &&
        evidence.selected_profile_id == selected_profile->profile_id &&
        evidence.envelope_sha256 == selected_profile->envelope_sha256 &&
        valid_admission_identity(input.admission_identity) &&
        input.admission_identity.bootstrap_envelope_and_evidence_hash ==
            evidence.envelope_sha256 &&
        input.admission_identity.bootstrap_root_invocation_hash ==
            root.root_invocation_hash &&
        input.admission_identity.bootstrap_profile_registry_hash ==
            registry.registry_sha256;
    if (!evidence_valid) {
        diagnostics.add(make_diagnostic(
            AdmissionErrorKind::BOOTSTRAP_EVIDENCE_INCOMPLETE,
            AdmissionBoundary::B_MINUS_1, "bootstrap", "bootstrap",
            evidence.evidence_id.view(), "bootstrap.evidence",
            kNoAdmissionDiagnosticIndex,
            AdmissionUnderlyingCause::IdentityInvalid));
    }

    CheckedU64 recomputed_total;
    if (!total_overflow && selected_profile != nullptr) {
        const CheckedU64 input_and_fixed =
            checked_u64_add(acquired_input_bytes,
                            selected_profile->fixed_owned_bytes);
        if (input_and_fixed) {
            recomputed_total =
                checked_u64_add(input_and_fixed.value, root.diagnostic_bytes);
        }
    }
    const std::uint64_t total_limit =
        selected_profile == nullptr
            ? 0
            : selected_profile->maximum_total_owned_bytes;
    const bool bootstrap_limit_invalid =
        root.input_count != root.inputs.size() ||
        (selected_profile != nullptr &&
         root.input_count > selected_profile->maximum_input_count) ||
        (selected_profile != nullptr && !recomputed_total) ||
        (selected_profile != nullptr &&
         root.total_owned_bytes != recomputed_total.value) ||
        (selected_profile != nullptr &&
         root.total_owned_bytes > total_limit) ||
        (selected_profile != nullptr &&
         root.diagnostic_bytes >
             selected_profile->maximum_diagnostic_bytes) ||
        !sha256_text(root.root_invocation_hash);
    if (bootstrap_limit_invalid) {
        AdmissionDiagnostic diagnostic = make_diagnostic(
            AdmissionErrorKind::BOOTSTRAP_LIMIT,
            AdmissionBoundary::B_MINUS_1, "bootstrap", "bootstrap",
            evidence.evidence_id.view(), "bootstrap.owned_bytes",
            kNoAdmissionDiagnosticIndex,
            total_overflow ? AdmissionUnderlyingCause::ArithmeticOverflow
                           : AdmissionUnderlyingCause::MemoryBudgetInsufficient);
        diagnostic.requested_bytes =
            optional_unsigned(root.total_owned_bytes);
        if (selected_profile != nullptr) {
            diagnostic.byte_limit = optional_unsigned(total_limit);
            diagnostic.headroom_or_deficit =
                headroom_or_deficit(root.total_owned_bytes, total_limit);
        }
        diagnostic.resource_kind = AdmissionResourceKind::Bytes;
        diagnostics.add(diagnostic);
    }
    if (input.request.gpu_halt_active &&
        any_prohibited_action(input.request)) {
        diagnostics.add(make_diagnostic(
            AdmissionErrorKind::DEVICE_ACTION_PROHIBITED,
            AdmissionBoundary::B_MINUS_1, "startup", "bootstrap",
            "gpu_halt", "device_action",
            kNoAdmissionDiagnosticIndex,
            AdmissionUnderlyingCause::BootGateFailed));
    }
    if (diagnostics.observed != 0) {
        return refuse(input, diagnostics);
    }

    const GeneratedPlanAdmissionFacts& plan = input.generated_plan;
    const ImplementationProfileRecord& implementation =
        input.implementation_profile;
    if (!implementation_accepts_plan(implementation, plan) ||
        !identity_matches_inputs(input.admission_identity, input)) {
        diagnostics.add(make_diagnostic(
            AdmissionErrorKind::UNSUPPORTED_GENERATED_PLAN,
            AdmissionBoundary::B0, "generated_plan", "budgeted",
            implementation.profile_id.view(), "generated_plan.identity",
            kNoAdmissionDiagnosticIndex,
            AdmissionUnderlyingCause::IdentityInvalid));
        return refuse(input, diagnostics);
    }

    const PreDeviceAdmissionRequest& request = input.request;
    if (request.configured_context_capacity == 0) {
        AdmissionDiagnostic diagnostic = make_diagnostic(
            AdmissionErrorKind::UNSUPPORTED_CONFIGURATION,
            AdmissionBoundary::B1, "configuration.context.nonzero",
            "budgeted", "configuration", "context_capacity",
            kNoAdmissionDiagnosticIndex,
            AdmissionUnderlyingCause::InvalidProfile);
        diagnostic.requested_resource_count =
            optional_unsigned(request.configured_context_capacity);
        diagnostic.resource_kind = AdmissionResourceKind::Tokens;
        diagnostics.add(diagnostic);
    }
    if (request.configured_context_capacity > plan.maximum_context()) {
        AdmissionDiagnostic diagnostic = make_diagnostic(
            AdmissionErrorKind::UNSUPPORTED_CONFIGURATION,
            AdmissionBoundary::B1, "configuration.context.plan",
            "budgeted", plan.plan_id().view(), "context_capacity",
            kNoAdmissionDiagnosticIndex,
            AdmissionUnderlyingCause::InvalidProfile);
        diagnostic.requested_resource_count =
            optional_unsigned(request.configured_context_capacity);
        diagnostic.resource_limit =
            optional_unsigned(plan.maximum_context());
        diagnostic.headroom_or_deficit =
            headroom_or_deficit(request.configured_context_capacity,
                                plan.maximum_context());
        diagnostic.resource_kind = AdmissionResourceKind::Tokens;
        diagnostics.add(diagnostic);
    }
    if (implementation.context_representability_bound == 0 ||
        request.configured_context_capacity >
            implementation.context_representability_bound) {
        AdmissionDiagnostic diagnostic = make_diagnostic(
            AdmissionErrorKind::UNSUPPORTED_CONFIGURATION,
            AdmissionBoundary::B1, "configuration.context.implementation",
            "budgeted", implementation.profile_id.view(),
            "context_capacity", kNoAdmissionDiagnosticIndex,
            AdmissionUnderlyingCause::InvalidProfile);
        diagnostic.requested_resource_count =
            optional_unsigned(request.configured_context_capacity);
        if (implementation.context_representability_bound != 0) {
            diagnostic.resource_limit = optional_unsigned(
                implementation.context_representability_bound);
            diagnostic.headroom_or_deficit =
                headroom_or_deficit(
                    request.configured_context_capacity,
                    implementation.context_representability_bound);
        }
        diagnostic.resource_kind = AdmissionResourceKind::Tokens;
        diagnostics.add(diagnostic);
    }

    const auto add_count_configuration_error =
        [&diagnostics](std::string_view owner, std::string_view resource,
                       std::uint64_t requested, std::uint64_t limit) noexcept {
            AdmissionDiagnostic diagnostic = make_diagnostic(
                AdmissionErrorKind::UNSUPPORTED_CONFIGURATION,
                AdmissionBoundary::B1, owner, "budgeted", "configuration",
                resource, kNoAdmissionDiagnosticIndex,
                AdmissionUnderlyingCause::InvalidProfile);
            diagnostic.requested_resource_count =
                optional_unsigned(requested);
            diagnostic.resource_limit = optional_unsigned(limit);
            diagnostic.headroom_or_deficit =
                headroom_or_deficit(requested, limit);
            diagnostic.resource_kind = AdmissionResourceKind::FixedRecords;
            diagnostics.add(diagnostic);
        };
    if (request.physical_slot_count != 1) {
        add_count_configuration_error("configuration.physical_slots",
                                      "physical_slots",
                                      request.physical_slot_count, 1);
    }
    if (request.maximum_concurrent_requests != 1) {
        add_count_configuration_error(
            "configuration.concurrent_requests",
            "maximum_concurrent_requests",
            request.maximum_concurrent_requests, 1);
    }
    if (request.queue_depth != 0) {
        add_count_configuration_error("configuration.queue_depth",
                                      "queue_depth", request.queue_depth, 0);
    }
    if (request.request_deadline_milliseconds != 0) {
        add_count_configuration_error(
            "configuration.request_deadline",
            "request_deadline_milliseconds",
            request.request_deadline_milliseconds, 0);
    }
    if (request.drain_timeout_milliseconds != 0) {
        add_count_configuration_error(
            "configuration.drain_timeout",
            "drain_timeout_milliseconds",
            request.drain_timeout_milliseconds, 0);
    }
    if (request.prompt_cache) {
        add_count_configuration_error("configuration.prompt_cache",
                                      "prompt_cache", 1, 0);
    }
    if (request.composed_prefill) {
        add_count_configuration_error("configuration.composed_prefill",
                                      "composed_prefill", 1, 0);
    }
    if (request.qgemm_policy != QgemmExecutionPolicy::ExactRow) {
        add_count_configuration_error("configuration.qgemm_policy",
                                      "qgemm_policy",
                                      static_cast<std::uint8_t>(
                                          request.qgemm_policy),
                                      static_cast<std::uint8_t>(
                                          QgemmExecutionPolicy::ExactRow));
    }
    if (request.requests_live_device_facts ||
        request.requests_device_construction ||
        request.requests_model_execution ||
        request.requests_command_authorization) {
        diagnostics.add(make_diagnostic(
            AdmissionErrorKind::UNSUPPORTED_CONFIGURATION,
            AdmissionBoundary::B1, "configuration.device_action",
            "budgeted", "gpu_halt", "device_action",
            kNoAdmissionDiagnosticIndex,
            AdmissionUnderlyingCause::BootGateFailed));
    }
    if (request.service_start_authority !=
            ServiceStartAuthority::SourceOnly ||
        request.requests_listener || request.requests_service_start ||
        input.admission_identity.service_start_authority_scope !=
            request.service_start_authority) {
        diagnostics.add(make_diagnostic(
            AdmissionErrorKind::SERVICE_START_AUTHORITY_INVALID,
            AdmissionBoundary::B1, "service_start_authority", "budgeted",
            "source_only", "service_start",
            kNoAdmissionDiagnosticIndex,
            AdmissionUnderlyingCause::BootGateFailed));
    }
    if (diagnostics.observed != 0) {
        return refuse(input, diagnostics);
    }

    PreDeviceAdmissionResult result;
    result.passed = true;
    result.plan.input = input;
    result.plan.completed_through = AdmissionBoundary::B1;
    return result;
}

} // namespace tatara::runtime
