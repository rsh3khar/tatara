#include "tatara/generated/model_plan.h"
#include "tatara/runtime/serving_capacity.h"

#include <cstdint>
#include <cstdio>

namespace {

using namespace tatara::runtime;

constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;

int failures = 0;

void check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

PrefillPolicy geometry_policy(std::uint32_t capacity) {
    return {
        .schedule = PrefillSchedule::LayerMajor,
        .context_capacity = capacity,
        .maximum_block_rows = 2048,
        .first_chunk_rows = 256,
        .query_tile_rows = 256,
        .attention_partition = 256,
        .exact_rows_per_threadgroup = 16,
        .gdn_gate_hoist = true,
    };
}

PrefillExecutionPolicy execution_policy(
    PrefillPolicy geometry, std::uint32_t lanes) {
    return {
        .geometry = geometry,
        .router_selector = PrefillRouterSelector::Parallel,
        .gdn_recurrence = PrefillGdnRecurrence::RegisterLoop,
        .attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive,
        .staged_attention_minimum_context = 256,
        .dense_qgemm = QuantizedGemmPolicy::NativeDenseMma,
        .routed_qgemm = QuantizedGemmPolicy::NativeRaggedMma,
        .native_dense_steel = true,
        .native_routed_shared_expert = true,
        .native_routed_steel = true,
        .command_graph = true,
        .command_graph_chunk_count = lanes,
        .maximum_units_per_submission = 1,
        .maximum_inflight_units = 1,
    };
}

ServingMemoryProfile profile(std::uint64_t cache_budget) {
    return {
        .requested_context_capacity = 0,
        .graph_scratch_lanes = 3,
        .composed_prefill = true,
        .physical_memory_bytes = 48 * kGiB,
        .metal_working_set_bytes = 40 * kGiB,
        .maximum_single_buffer_bytes = 32 * kGiB,
        .unified_external_occupancy_bytes = 0,
        .metal_external_occupancy_bytes = 0,
        .os_runtime_reserve_bytes = 4 * kGiB,
        .prepared_image_bytes = 19'508'814'336ull,
        .bootstrap_and_tokenizer_bytes = 64 * 1024 * 1024,
        .cache_budget_bytes = cache_budget,
        .graph_object_budget_bytes = 256 * 1024 * 1024,
    };
}

void exact_65536_budget() {
    const auto& model =
        tatara::model::qwen36::generated::kModelPlan;
    constexpr std::uint32_t capacity = 65'536;
    const PrefillPolicy policy = geometry_policy(capacity);
    const auto prefill = make_prefill_geometry(model, policy);
    check(static_cast<bool>(prefill), "65,536 prefill geometry");
    if (!prefill) {
        return;
    }
    const ServingCapacityCandidate candidate =
        evaluate_serving_capacity_candidate(
            make_decode_geometry(model, capacity),
            prefill.geometry, execution_policy(policy, 3),
            profile(8 * kGiB), capacity);
    check(static_cast<bool>(candidate), "65,536 budget evaluates");
    check(candidate.breakdown.decode_slot_bytes ==
              1'470'955'520ull,
          "65,536 exact slot bytes");
    check(candidate.breakdown.decode_scratch_bytes == 9'739'468ull,
          "65,536 exact decode scratch bytes");
    check(candidate.breakdown.prefill_step_bytes ==
              4'808'064'132ull,
          "65,536 exact three-lane prefill bytes");
}

void model_maximum_has_no_engine_ceiling() {
    const auto& model =
        tatara::model::qwen36::generated::kModelPlan;
    const PrefillPolicy policy =
        geometry_policy(model.tokenizer.maximum_context);
    const ServingCapacityResult result = plan_serving_capacity(
        model, profile(0), policy, execution_policy(policy, 3));
    check(static_cast<bool>(result), "automatic admission succeeds");
    check(result.representability_maximum_context ==
              model.tokenizer.maximum_context,
          "engine representability does not shrink model maximum");
    check(result.maximum_admissible_context ==
              model.tokenizer.maximum_context,
          "real fitting profile admits full model context");
    check(result.admitted_context_capacity ==
              model.tokenizer.maximum_context,
          "automatic capacity selects full model context");
}

void cache_tradeoff_and_typed_refusal() {
    const auto& model =
        tatara::model::qwen36::generated::kModelPlan;
    const PrefillPolicy policy =
        geometry_policy(model.tokenizer.maximum_context);
    const ServingCapacityResult no_cache = plan_serving_capacity(
        model, profile(0), policy, execution_policy(policy, 3));
    ServingMemoryProfile with_cache = profile(16 * kGiB);
    const ServingCapacityResult cached = plan_serving_capacity(
        model, with_cache, policy, execution_policy(policy, 3));
    check(static_cast<bool>(no_cache) && static_cast<bool>(cached),
          "cache tradeoff profiles admit some capacity");
    check(cached.maximum_admissible_context <=
              no_cache.maximum_admissible_context,
          "increasing cache never increases context admission");

    with_cache.requested_context_capacity =
        model.tokenizer.maximum_context;
    const ServingCapacityResult refused = plan_serving_capacity(
        model, with_cache, policy, execution_policy(policy, 3));
    check(refused.error ==
              ServingCapacityError::RequestedCapacityNotAdmissible,
          "non-fitting requested model maximum is typed");
    check(refused.limiting.metal_deficit_bytes != 0 ||
              refused.limiting.unified_deficit_bytes != 0 ||
              refused.limiting.single_buffer_deficit_bytes != 0,
          "typed refusal reports an exact deficit");
}

} // namespace

int main() {
    exact_65536_budget();
    model_maximum_has_no_engine_ceiling();
    cache_tradeoff_and_typed_refusal();
    if (failures == 0) {
        std::printf("serving capacity: PASS\n");
    }
    return failures == 0 ? 0 : 1;
}
