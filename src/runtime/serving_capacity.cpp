#include "tatara/runtime/serving_capacity.h"

#include "tatara/runtime/checked_arithmetic.h"

#include <algorithm>
#include <array>
#include <limits>

namespace tatara::runtime {
namespace {

bool add(std::uint64_t value, std::uint64_t& total) noexcept {
    const CheckedU64 next = checked_u64_add(total, value);
    if (!next) {
        return false;
    }
    total = next.value;
    return true;
}

bool scaled(std::uint64_t value, std::uint64_t count,
            std::uint64_t& result) noexcept {
    const CheckedU64 product = checked_u64_multiply(value, count);
    if (!product) {
        return false;
    }
    result = product.value;
    return true;
}

std::uint64_t deficit(std::uint64_t required,
                      std::uint64_t available) noexcept {
    return required > available ? required - available : 0;
}

} // namespace

std::string_view serving_capacity_error_name(
    ServingCapacityError error) noexcept {
    switch (error) {
    case ServingCapacityError::None:
        return "none";
    case ServingCapacityError::InvalidProfile:
        return "invalid_profile";
    case ServingCapacityError::ModelOrImplementationBoundInvalid:
        return "model_or_implementation_bound_invalid";
    case ServingCapacityError::GeometryInvalid:
        return "geometry_invalid";
    case ServingCapacityError::ArithmeticOverflow:
        return "arithmetic_overflow";
    case ServingCapacityError::NoCapacityFits:
        return "no_capacity_fits";
    case ServingCapacityError::RequestedCapacityNotAdmissible:
        return "requested_capacity_not_admissible";
    }
    return "unknown";
}

ServingCapacityCandidate evaluate_serving_capacity_candidate(
    const DecodeGeometry& decode,
    const PrefillGeometry& prefill,
    const PrefillExecutionPolicy& execution_policy,
    const ServingMemoryProfile& profile,
    std::uint32_t context_capacity) noexcept {
    ServingCapacityCandidate result;
    ServingCapacityBreakdown& breakdown = result.breakdown;
    breakdown.context_capacity = context_capacity;
    breakdown.state_slots =
        profile.concurrent_state_slots == 0
            ? 1u
            : profile.concurrent_state_slots;
    breakdown.prepared_image_bytes = profile.prepared_image_bytes;
    if (!scaled(prefill.slot_state_bytes, breakdown.state_slots,
                breakdown.decode_slot_bytes)) {
        result.error = ServingCapacityError::ArithmeticOverflow;
        return result;
    }
    breakdown.cache_budget_bytes = profile.cache_budget_bytes;
    breakdown.graph_object_budget_bytes =
        profile.graph_object_budget_bytes;
    breakdown.bootstrap_and_tokenizer_bytes =
        profile.bootstrap_and_tokenizer_bytes;
    breakdown.maximum_single_buffer_bytes =
        profile.maximum_single_buffer_bytes;

    if (context_capacity == 0 ||
        decode.attn_cache_bytes == 0 ||
        prefill.context_capacity != context_capacity ||
        profile.physical_memory_bytes == 0 ||
        profile.metal_working_set_bytes == 0 ||
        profile.maximum_single_buffer_bytes == 0 ||
        profile.prepared_image_bytes == 0 ||
        profile.os_runtime_reserve_bytes == 0 ||
        profile.unified_external_occupancy_bytes >
            profile.physical_memory_bytes ||
        profile.metal_external_occupancy_bytes >
            profile.metal_working_set_bytes) {
        result.error = ServingCapacityError::InvalidProfile;
        return result;
    }

    std::uint64_t scratch = 0;
    std::uint64_t term = 0;
    const auto add_scaled =
        [&](std::uint64_t value, std::uint64_t count) noexcept {
            return scaled(value, count, term) && add(term, scratch);
        };
    if (!add_scaled(decode.hidden_bytes, 2) ||
        !add_scaled(decode.layer_stream_bytes, 4) ||
        !add(decode.gdn_projection_bytes, scratch) ||
        !add(decode.gdn_qk_bytes, scratch) ||
        !add_scaled(decode.gdn_value_bytes, 2) ||
        !add_scaled(decode.gdn_gate_bytes, 2) ||
        !add(decode.attn_projection_bytes, scratch) ||
        !add_scaled(decode.attn_query_bytes, 3) ||
        !add_scaled(decode.attn_record_scratch_bytes, 2) ||
        !add(decode.router_logits_bytes, scratch) ||
        !add(decode.expert_id_bytes, scratch) ||
        !add(decode.expert_coefficient_bytes, scratch) ||
        !add(4, scratch) ||
        !add(decode.expert_hidden_bytes, scratch) ||
        !add(decode.hidden_bytes, scratch) ||
        !add(decode.logits_bytes, scratch) ||
        !add(decode.argmax_value_bytes, scratch) ||
        !add(decode.argmax_index_bytes, scratch) ||
        !add(decode.token_id_bytes, scratch)) {
        result.error = ServingCapacityError::ArithmeticOverflow;
        return result;
    }
    breakdown.decode_scratch_bytes = scratch;

    PrefillMemoryPlan prefill_memory{
        .error = PrefillMemoryPlanError::None,
    };
    if (profile.composed_prefill) {
        prefill_memory =
            plan_prefill_step_memory(prefill, execution_policy);
        if (!prefill_memory) {
            result.error =
                prefill_memory.error ==
                        PrefillMemoryPlanError::ArithmeticOverflow
                    ? ServingCapacityError::ArithmeticOverflow
                    : ServingCapacityError::GeometryInvalid;
            return result;
        }
        breakdown.prefill_step_bytes = prefill_memory.total_bytes;
    }

    std::uint64_t metal_required = 0;
    if (!add(profile.prepared_image_bytes, metal_required) ||
        !add(prefill.slot_state_bytes, metal_required) ||
        !add(scratch, metal_required) ||
        !add(breakdown.prefill_step_bytes, metal_required) ||
        !add(profile.cache_budget_bytes, metal_required) ||
        !add(profile.graph_object_budget_bytes, metal_required)) {
        result.error = ServingCapacityError::ArithmeticOverflow;
        return result;
    }
    breakdown.metal_required_bytes = metal_required;
    std::uint64_t unified_required = metal_required;
    if (!add(profile.bootstrap_and_tokenizer_bytes,
             unified_required)) {
        result.error = ServingCapacityError::ArithmeticOverflow;
        return result;
    }
    breakdown.unified_required_bytes = unified_required;

    const std::uint64_t metal_available =
        profile.metal_working_set_bytes -
        profile.metal_external_occupancy_bytes;
    const std::uint64_t unified_after_external =
        profile.physical_memory_bytes -
        profile.unified_external_occupancy_bytes;
    if (profile.os_runtime_reserve_bytes >
        unified_after_external) {
        result.error = ServingCapacityError::InvalidProfile;
        return result;
    }
    const std::uint64_t unified_available =
        unified_after_external -
        profile.os_runtime_reserve_bytes;
    breakdown.metal_available_bytes = metal_available;
    breakdown.unified_available_bytes = unified_available;

    const std::array<std::uint64_t, 13> decode_buffers{
        decode.hidden_bytes,
        decode.layer_stream_bytes,
        decode.gdn_projection_bytes,
        decode.gdn_qk_bytes,
        decode.gdn_value_bytes,
        decode.gdn_gate_bytes,
        decode.gdn_conv_state_bytes,
        decode.gdn_recurrent_state_bytes,
        decode.attn_projection_bytes,
        decode.attn_query_bytes,
        decode.attn_record_scratch_bytes,
        decode.attn_cache_bytes,
        decode.logits_bytes,
    };
    // Cache and graph-object budgets are aggregate reservations, not claims
    // that either owner allocates one buffer of the full budget size. Only
    // concrete buffer extents participate in the Metal maxBufferLength gate.
    std::uint64_t maximum_buffer = profile.prepared_image_bytes;
    for (const std::uint64_t bytes : decode_buffers) {
        maximum_buffer = std::max(maximum_buffer, bytes);
    }
    maximum_buffer =
        std::max(maximum_buffer,
                 prefill_memory.maximum_single_buffer_bytes);
    breakdown.maximum_planned_buffer_bytes = maximum_buffer;

    breakdown.metal_deficit_bytes =
        deficit(metal_required, metal_available);
    breakdown.unified_deficit_bytes =
        deficit(unified_required, unified_available);
    breakdown.single_buffer_deficit_bytes =
        deficit(maximum_buffer,
                profile.maximum_single_buffer_bytes);
    breakdown.fits = breakdown.metal_deficit_bytes == 0 &&
                     breakdown.unified_deficit_bytes == 0 &&
                     breakdown.single_buffer_deficit_bytes == 0;
    result.error = ServingCapacityError::None;
    return result;
}

} // namespace tatara::runtime
