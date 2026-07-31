#include "tatara/generated/model_plan.h"
#include "tatara/runtime/prefill_geometry.h"

#include <array>
#include <limits>

namespace {

using namespace tatara::runtime;

constexpr PrefillPolicy kInitialPolicy{
    .schedule = PrefillSchedule::LayerMajor,
    .context_capacity = 16384,
    .maximum_block_rows = 2048,
    .first_chunk_rows = 256,
    .query_tile_rows = 256,
    .attention_partition = 256,
    .exact_rows_per_threadgroup = 16,
    .gdn_gate_hoist = true,
};

constexpr auto kInitial =
    make_prefill_geometry(tatara::model::qwen36::generated::kModelPlan, kInitialPolicy);
static_assert(kInitial);
static_assert(kInitial.geometry.gated_delta_layers == 30);
static_assert(kInitial.geometry.attention_layers == 10);
static_assert(kInitial.geometry.attention_partitions == 64);
static_assert(kInitial.geometry.schedule == kInitialPolicy.schedule);
static_assert(kInitial.geometry.context_capacity == kInitialPolicy.context_capacity);
static_assert(kInitial.geometry.maximum_block_rows == kInitialPolicy.maximum_block_rows);
static_assert(kInitial.geometry.first_chunk_rows == kInitialPolicy.first_chunk_rows);
static_assert(kInitial.geometry.query_tile_rows == kInitialPolicy.query_tile_rows);
static_assert(kInitial.geometry.attention_partition == kInitialPolicy.attention_partition);
static_assert(kInitial.geometry.exact_rows_per_threadgroup ==
              kInitialPolicy.exact_rows_per_threadgroup);
static_assert(kInitial.geometry.gdn_gate_hoist == kInitialPolicy.gdn_gate_hoist);
static_assert(kInitial.geometry.hidden ==
              tatara::model::qwen36::generated::kModelPlan.dimensions.hidden);
static_assert(kInitial.geometry.vocabulary ==
              tatara::model::qwen36::generated::kModelPlan.dimensions.vocabulary);
static_assert(kInitial.geometry.query_heads ==
              tatara::model::qwen36::generated::kModelPlan.attention.query_heads);
static_assert(kInitial.geometry.key_value_heads ==
              tatara::model::qwen36::generated::kModelPlan.attention.key_value_heads);
static_assert(kInitial.geometry.attention_head_dimension ==
              tatara::model::qwen36::generated::kModelPlan.attention.head_dimension);
static_assert(kInitial.geometry.recurrent_heads ==
              tatara::model::qwen36::generated::kModelPlan.gated_delta.recurrent_heads);
static_assert(kInitial.geometry.state_dimension ==
              tatara::model::qwen36::generated::kModelPlan.gated_delta.state_dimension);
static_assert(kInitial.geometry.experts ==
              tatara::model::qwen36::generated::kModelPlan.mixture_of_experts.experts);
static_assert(kInitial.geometry.active_experts ==
              tatara::model::qwen36::generated::kModelPlan.mixture_of_experts.active_experts);
static_assert(kInitial.geometry.expert_dimension ==
              tatara::model::qwen36::generated::kModelPlan.mixture_of_experts.expert_dimension);
static_assert(kInitial.geometry.gdn_live_state_bytes_per_layer == 2146304);
static_assert(kInitial.geometry.gdn_slot_state_bytes_per_layer == 4292608);
static_assert(kInitial.geometry.attention_state_bytes_per_position_per_layer == 2048);
static_assert(kInitial.geometry.attention_slot_state_bytes_per_layer == 33554432);
static_assert(kInitial.geometry.slot_state_bytes == 464322560);
static_assert(kInitial.geometry.hidden_slab_bytes == 67108864);
static_assert(kInitial.geometry.block_hidden_bytes == 8388608);
static_assert(kInitial.geometry.gdn_projection_bytes == 50593792);
static_assert(kInitial.geometry.attention_projection_bytes == 37748736);
static_assert(kInitial.geometry.attention_partial_bytes == 270532608);
static_assert(
    kInitial.geometry.attention_staged_score_bytes ==
    268435456);
static_assert(
    kInitial.geometry.attention_staged_score_bytes <
    kInitial.geometry.attention_partial_bytes);
static_assert(kInitial.geometry.token_bytes == 65536);
static_assert(kInitial.geometry.reusable_scratch_bytes == 701458464);
static_assert(kInitial.geometry.steady_prefill_bytes == 768567328);
static_assert(native_routed_shared_task_capacity_supported(
    kInitial.geometry, 16, 1408));
static_assert(!native_routed_shared_task_capacity_supported(
    kInitial.geometry, 16, 1407));

constexpr auto kSnapshot4k = make_prefix_state_snapshot_geometry(kInitial.geometry, 4096);
static_assert(kSnapshot4k);
static_assert(kSnapshot4k.gated_delta_bytes == 64389120);
static_assert(kSnapshot4k.attention_bytes == 83886080);
static_assert(kSnapshot4k.total_bytes == 148275200);

constexpr auto kSnapshot16k = make_prefix_state_snapshot_geometry(kInitial.geometry, 16384);
static_assert(kSnapshot16k);
static_assert(kSnapshot16k.gated_delta_bytes == 64389120);
static_assert(kSnapshot16k.attention_bytes == 335544320);
static_assert(kSnapshot16k.total_bytes == 399933440);

constexpr auto kSnapshot100k = [] {
    auto policy = kInitialPolicy;
    policy.context_capacity = 100000;
    const auto geometry =
        make_prefill_geometry(tatara::model::qwen36::generated::kModelPlan, policy);
    return make_prefix_state_snapshot_geometry(geometry.geometry, 100000);
}();
static_assert(kSnapshot100k);
static_assert(kSnapshot100k.gated_delta_bytes == 64389120);
static_assert(kSnapshot100k.attention_bytes == 2048000000);
static_assert(kSnapshot100k.total_bytes == 2112389120);

static_assert(make_prefix_state_snapshot_geometry(PrefillGeometry{}, 1).error ==
              PrefixStateSnapshotError::InvalidGeometry);
static_assert(make_prefix_state_snapshot_geometry(kInitial.geometry, 0).error ==
              PrefixStateSnapshotError::PositionOutOfRange);
static_assert(make_prefix_state_snapshot_geometry(kInitial.geometry, 16385).error ==
              PrefixStateSnapshotError::PositionOutOfRange);

constexpr auto kOverflowSnapshot = [] {
    PrefillGeometry geometry;
    geometry.context_capacity = 1;
    geometry.gated_delta_layers = 2;
    geometry.attention_layers = 1;
    geometry.gdn_live_state_bytes_per_layer = std::numeric_limits<std::uint64_t>::max();
    geometry.attention_state_bytes_per_position_per_layer = 1;
    return make_prefix_state_snapshot_geometry(geometry, 1);
}();
static_assert(kOverflowSnapshot.error == PrefixStateSnapshotError::Overflow);

constexpr auto kChunkMajor = [] {
    auto policy = kInitialPolicy;
    policy.schedule = PrefillSchedule::ChunkMajor;
    return make_prefill_geometry(tatara::model::qwen36::generated::kModelPlan, policy);
}();
static_assert(kChunkMajor);
static_assert(kChunkMajor.geometry.hidden_slab_bytes == 0);
static_assert(kChunkMajor.geometry.reusable_scratch_bytes ==
              kInitial.geometry.reusable_scratch_bytes);
static_assert(kChunkMajor.geometry.steady_prefill_bytes ==
              kInitial.geometry.steady_prefill_bytes - kInitial.geometry.hidden_slab_bytes);

constexpr auto kNoGateHoist = [] {
    auto policy = kInitialPolicy;
    policy.gdn_gate_hoist = false;
    return make_prefill_geometry(tatara::model::qwen36::generated::kModelPlan, policy);
}();
static_assert(kNoGateHoist);
static_assert(kNoGateHoist.geometry.gdn_parameter_bytes == 0);
static_assert(kNoGateHoist.geometry.reusable_scratch_bytes ==
              kInitial.geometry.reusable_scratch_bytes -
                  2u * kInitial.geometry.gdn_parameter_bytes);

constexpr auto invalid_tile = [] {
    auto policy = kInitialPolicy;
    policy.query_tile_rows = 17;
    return make_prefill_geometry(tatara::model::qwen36::generated::kModelPlan, policy);
}();
static_assert(!invalid_tile && invalid_tile.error == PrefillGeometryError::InvalidPolicy);

constexpr auto invalid_context = [] {
    auto policy = kInitialPolicy;
    policy.context_capacity = kPrefillMaximumContext + 1;
    return make_prefill_geometry(tatara::model::qwen36::generated::kModelPlan, policy);
}();
static_assert(!invalid_context && invalid_context.error == PrefillGeometryError::InvalidPolicy);

constexpr auto invalid_schedule = [] {
    auto policy = kInitialPolicy;
    policy.schedule = static_cast<PrefillSchedule>(255);
    return make_prefill_geometry(tatara::model::qwen36::generated::kModelPlan, policy);
}();
static_assert(!invalid_schedule && invalid_schedule.error == PrefillGeometryError::InvalidPolicy);

// Generality compliance: the same geometry code accepts a regenerated model
// plan with a different expert count and top-k. No Qwen source literal moves.
constexpr tatara::model::qwen36::StaticModelPlan<4> kSecondMoePlan{
    .id = "second-moe",
    .family = "qwen3_5_moe",
    .package_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    .artifact =
        {
            .id = "second-artifact",
            .model_type = "qwen3_5_moe",
            .format = "safetensors",
            .source_repository = "local",
            .source_revision = "main",
            .manifest_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            .tensor_count = 1,
            .tensor_bytes = 1,
            .file_count = 1,
            .weight_file_count = 1,
        },
    .dimensions = {.hidden = 2048, .vocabulary = 32768},
    .attention = {.query_heads = 16, .key_value_heads = 2, .head_dimension = 256},
    .gated_delta = {.recurrent_heads = 32, .state_dimension = 128},
    .mixture_of_experts = {.experts = 64, .active_experts = 4, .expert_dimension = 512},
    .weights = {.format = tatara::model::qwen36::WeightFormat::AffineQ4, .group_size = 64},
    .initial_serving_capacity = 16384,
    .layers =
        {
            tatara::model::qwen36::LayerKind::GatedDelta,
            tatara::model::qwen36::LayerKind::GatedDelta,
            tatara::model::qwen36::LayerKind::GatedDelta,
            tatara::model::qwen36::LayerKind::FullAttention,
        },
};

constexpr auto kSecondMoe = make_prefill_geometry(kSecondMoePlan, kInitialPolicy);
static_assert(kSecondMoe);
static_assert(kSecondMoe.geometry.gated_delta_layers == 3);
static_assert(kSecondMoe.geometry.attention_layers == 1);
static_assert(kSecondMoe.geometry.moe_logits_bytes == 2048ull * 65 * 4);
static_assert(kSecondMoe.geometry.moe_id_bytes == 2048ull * 4 * 4);
static_assert(kSecondMoe.geometry.moe_hidden_bytes == 2048ull * 5 * 512 * 2);
static_assert(kSecondMoe.geometry.moe_partial_bytes == 2048ull * 5 * 2048 * 4);
static_assert(native_routed_shared_task_capacity_supported(
    kSecondMoe.geometry, 16, 704));

constexpr auto invalid_top_k = [] {
    auto plan = kSecondMoePlan;
    plan.mixture_of_experts.active_experts = 16;
    return make_prefill_geometry(plan, kInitialPolicy);
}();
static_assert(!invalid_top_k && invalid_top_k.error == PrefillGeometryError::InvalidPlan);

constexpr auto invalid_layer = [] {
    auto plan = kSecondMoePlan;
    plan.layers[0] = static_cast<tatara::model::qwen36::LayerKind>(255);
    return make_prefill_geometry(plan, kInitialPolicy);
}();
static_assert(!invalid_layer && invalid_layer.error == PrefillGeometryError::InvalidPlan);

constexpr std::array<std::uint32_t, 3> kTokens{7, 11, 13};
constexpr auto kValidRequest = validate_prefill_request(kInitialPolicy, 41, 41, kTokens, 151936);
static_assert(kValidRequest && kValidRequest.next_context == 44);

constexpr std::array<std::uint32_t, 0> kEmptyTokens{};
constexpr auto kEmptyRequest = validate_prefill_request(kInitialPolicy, 0, 0, kEmptyTokens, 151936);
static_assert(!kEmptyRequest && kEmptyRequest.error == PrefillRequestError::EmptyPrefix);

constexpr auto kStaleContext = validate_prefill_request(kInitialPolicy, 40, 41, kTokens, 151936);
static_assert(!kStaleContext && kStaleContext.error == PrefillRequestError::ContextOutOfRange);

constexpr auto kContextOverflow =
    validate_prefill_request(kInitialPolicy, 16382, 16382, kTokens, 151936);
static_assert(!kContextOverflow && kContextOverflow.error == PrefillRequestError::ContextOverflow);

constexpr std::array<std::uint32_t, 1> kInvalidToken{151936};
constexpr auto kTokenOutOfRange =
    validate_prefill_request(kInitialPolicy, 0, 0, kInvalidToken, 151936);
static_assert(!kTokenOutOfRange && kTokenOutOfRange.error == PrefillRequestError::TokenOutOfRange);

constexpr std::array<std::uint32_t, kPrefillMaximumBlockRows + 1> kOversizedBlock{};
constexpr auto kBlockOutOfRange =
    validate_prefill_request(kInitialPolicy, 0, 0, kOversizedBlock, 151936);
static_assert(!kBlockOutOfRange && kBlockOutOfRange.error == PrefillRequestError::BlockOutOfRange);

constexpr std::array<std::uint32_t, kPrefillMaximumBlockRows + 1> kLongPrefix{};
constexpr auto kValidLongPrefix =
    validate_prefill_prefix(kInitialPolicy, 0, 0, kLongPrefix, 151936);
static_assert(kValidLongPrefix && kValidLongPrefix.next_context == kLongPrefix.size());

} // namespace

int main() {
    return 0;
}
