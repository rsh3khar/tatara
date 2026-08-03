#pragma once

#include "tatara/model/qwen36_plan.h"
#include "tatara/runtime/decode_geometry.h"
#include "tatara/runtime/prefill_step.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace tatara::runtime {

enum class ServingCapacityError : std::uint8_t {
    None,
    InvalidProfile,
    ModelOrImplementationBoundInvalid,
    GeometryInvalid,
    ArithmeticOverflow,
    NoCapacityFits,
    RequestedCapacityNotAdmissible,
};

struct ServingMemoryProfile {
    // Zero requests automatic maximum admission. A nonzero value is an
    // operator request, not a second model/product ceiling.
    std::uint32_t requested_context_capacity{0};
    std::uint32_t graph_scratch_lanes{1};
    // Persistent decode state slots the plan must hold resident. Every
    // slot carries the full per-request state (KV at the admitted
    // context plus the gated-delta planes); one slot is the serial
    // serving shape.
    std::uint32_t concurrent_state_slots{1};
    bool composed_prefill{false};

    std::uint64_t physical_memory_bytes{0};
    std::uint64_t metal_working_set_bytes{0};
    std::uint64_t maximum_single_buffer_bytes{0};
    std::uint64_t unified_external_occupancy_bytes{0};
    std::uint64_t metal_external_occupancy_bytes{0};
    std::uint64_t os_runtime_reserve_bytes{0};

    std::uint64_t prepared_image_bytes{0};
    std::uint64_t bootstrap_and_tokenizer_bytes{0};
    std::uint64_t cache_budget_bytes{0};
    std::uint64_t graph_object_budget_bytes{0};
};

struct ServingCapacityBreakdown {
    std::uint32_t context_capacity{0};
    std::uint32_t state_slots{1};
    std::uint64_t prepared_image_bytes{0};
    std::uint64_t decode_slot_bytes{0};
    std::uint64_t decode_scratch_bytes{0};
    std::uint64_t prefill_step_bytes{0};
    std::uint64_t cache_budget_bytes{0};
    std::uint64_t graph_object_budget_bytes{0};
    std::uint64_t bootstrap_and_tokenizer_bytes{0};
    std::uint64_t metal_required_bytes{0};
    std::uint64_t unified_required_bytes{0};
    std::uint64_t metal_available_bytes{0};
    std::uint64_t unified_available_bytes{0};
    std::uint64_t maximum_planned_buffer_bytes{0};
    std::uint64_t maximum_single_buffer_bytes{0};
    std::uint64_t metal_deficit_bytes{0};
    std::uint64_t unified_deficit_bytes{0};
    std::uint64_t single_buffer_deficit_bytes{0};
    bool fits{false};
};

struct ServingCapacityCandidate {
    ServingCapacityError error{ServingCapacityError::InvalidProfile};
    ServingCapacityBreakdown breakdown;

    explicit constexpr operator bool() const noexcept {
        return error == ServingCapacityError::None;
    }
};

struct ServingCapacityResult {
    ServingCapacityError error{ServingCapacityError::InvalidProfile};
    std::uint32_t model_maximum_context{0};
    std::uint32_t representability_maximum_context{0};
    std::uint32_t requested_context_capacity{0};
    std::uint32_t maximum_admissible_context{0};
    std::uint32_t admitted_context_capacity{0};
    std::uint32_t maximum_without_cache{0};
    ServingCapacityBreakdown admitted;
    ServingCapacityBreakdown limiting;

    explicit constexpr operator bool() const noexcept {
        return error == ServingCapacityError::None;
    }
};

[[nodiscard]] ServingCapacityCandidate evaluate_serving_capacity_candidate(
    const DecodeGeometry& decode_geometry,
    const PrefillGeometry& prefill_geometry,
    const PrefillExecutionPolicy& execution_policy,
    const ServingMemoryProfile& profile,
    std::uint32_t context_capacity) noexcept;

[[nodiscard]] std::string_view serving_capacity_error_name(
    ServingCapacityError error) noexcept;

template <std::size_t LayerCount>
[[nodiscard]] ServingCapacityResult plan_serving_capacity(
    const model::qwen36::StaticModelPlan<LayerCount>& plan,
    const ServingMemoryProfile& profile,
    PrefillPolicy geometry_policy,
    PrefillExecutionPolicy execution_policy) noexcept {
    ServingCapacityResult result;
    result.model_maximum_context = plan.tokenizer.maximum_context;
    result.requested_context_capacity =
        profile.requested_context_capacity;
    // An engine implementation bound may validate the model contract, but it
    // may not silently shrink it into a product capability. If the generated
    // model maximum is not representable, boot is defective and fails typed.
    const std::uint32_t representability =
        plan.tokenizer.maximum_context;
    result.representability_maximum_context = representability;

    if (plan.tokenizer.maximum_context == 0 || representability < 2 ||
        kPrefillMaximumContext < plan.tokenizer.maximum_context) {
        result.error =
            ServingCapacityError::ModelOrImplementationBoundInvalid;
        return result;
    }
    if (
        (profile.requested_context_capacity != 0 &&
         profile.requested_context_capacity < 2) ||
        profile.graph_scratch_lanes == 0 ||
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
    if (profile.requested_context_capacity > representability) {
        result.error =
            ServingCapacityError::ModelOrImplementationBoundInvalid;
        return result;
    }

    const auto evaluate =
        [&](std::uint32_t capacity,
            const ServingMemoryProfile& candidate_profile) noexcept {
            PrefillPolicy candidate_geometry_policy = geometry_policy;
            candidate_geometry_policy.context_capacity = capacity;
            const auto prefill_geometry =
                make_prefill_geometry(plan, candidate_geometry_policy);
            if (!prefill_geometry) {
                ServingCapacityCandidate candidate;
                candidate.error = ServingCapacityError::GeometryInvalid;
                candidate.breakdown.context_capacity = capacity;
                return candidate;
            }
            PrefillExecutionPolicy candidate_execution_policy =
                execution_policy;
            candidate_execution_policy.geometry =
                candidate_geometry_policy;
            candidate_execution_policy.command_graph_chunk_count =
                candidate_profile.composed_prefill
                    ? candidate_profile.graph_scratch_lanes
                    : 1u;
            if (candidate_execution_policy.staged_attention_minimum_context >=
                capacity) {
                candidate_execution_policy.staged_attention_minimum_context =
                    capacity - 1u;
            }
            if (candidate_execution_policy
                    .streaming_attention_minimum_context >= capacity) {
                candidate_execution_policy
                    .streaming_attention_minimum_context =
                    capacity - 1u;
            }
            const DecodeGeometry decode_geometry =
                make_decode_geometry(plan, capacity);
            return evaluate_serving_capacity_candidate(
                decode_geometry, prefill_geometry.geometry,
                candidate_execution_policy, candidate_profile, capacity);
        };

    const auto maximum_for =
        [&](const ServingMemoryProfile& candidate_profile,
            std::uint32_t& maximum,
            ServingCapacityBreakdown& limiting) noexcept {
            std::uint32_t low = 2;
            std::uint32_t high = representability;
            maximum = 0;
            while (low <= high) {
                const std::uint32_t midpoint =
                    low + (high - low) / 2u;
                const ServingCapacityCandidate candidate =
                    evaluate(midpoint, candidate_profile);
                if (!candidate) {
                    result.error = candidate.error;
                    limiting = candidate.breakdown;
                    return false;
                }
                if (candidate.breakdown.fits) {
                    maximum = midpoint;
                    if (midpoint ==
                        std::numeric_limits<std::uint32_t>::max()) {
                        break;
                    }
                    low = midpoint + 1u;
                } else {
                    limiting = candidate.breakdown;
                    if (midpoint == 0) {
                        break;
                    }
                    high = midpoint - 1u;
                }
            }
            return true;
        };

    if (!maximum_for(profile, result.maximum_admissible_context,
                     result.limiting)) {
        return result;
    }
    if (result.maximum_admissible_context == 0) {
        result.error = ServingCapacityError::NoCapacityFits;
        return result;
    }

    ServingMemoryProfile no_cache = profile;
    no_cache.requested_context_capacity = 0;
    no_cache.cache_budget_bytes = 0;
    ServingCapacityBreakdown no_cache_limiting;
    if (!maximum_for(no_cache, result.maximum_without_cache,
                     no_cache_limiting)) {
        return result;
    }

    const std::uint32_t admitted =
        profile.requested_context_capacity == 0
            ? result.maximum_admissible_context
            : profile.requested_context_capacity;
    const ServingCapacityCandidate admitted_candidate =
        evaluate(admitted, profile);
    if (!admitted_candidate) {
        result.error = admitted_candidate.error;
        result.limiting = admitted_candidate.breakdown;
        return result;
    }
    if (!admitted_candidate.breakdown.fits) {
        result.error =
            ServingCapacityError::RequestedCapacityNotAdmissible;
        result.limiting = admitted_candidate.breakdown;
        return result;
    }
    result.error = ServingCapacityError::None;
    result.admitted_context_capacity = admitted;
    result.admitted = admitted_candidate.breakdown;
    return result;
}

} // namespace tatara::runtime
