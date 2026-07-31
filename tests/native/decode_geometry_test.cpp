#include "tatara/generated/kernel_library.h"
#include "tatara/generated/model_plan.h"
#include "tatara/runtime/decode_geometry.h"

namespace {

using namespace tatara::runtime;

namespace kernel_library = tatara::backend::metal::generated;

// The package record now owns the per-expert intermediate dimension, and
// make_decode_geometry reads it from the plan; this pins that the compiled
// kernels agree with it, so geometry and kernel cannot silently disagree.
static_assert(tatara::model::qwen36::generated::kModelPlan.mixture_of_experts.expert_dimension ==
              kernel_library::kKernelLibraryMoeExpertDimension);

// kArgmaxGroupCount stays a hand-written mirror deliberately. Substituting the
// generated constant would make decode_geometry.h -- a pure-core header that
// today includes only the plan and <cstdint> -- depend on a backend generated
// header, trading a pinned duplicate for a broken layering. The value is not
// package-derived in any case: the 256x256 argmax grid is a literal in
// head.metal, so the generator's constant is itself a mirror of kernel source.
static_assert(kArgmaxGroupCount == kernel_library::kKernelLibraryArgmaxGroups);

// The first-package geometry, hand-derived from the sealed dimensions:
// hidden 2048, 40 layers, GDN 32x128 heads over 8192 conv channels,
// attention 16/2 heads at 256, MoE 256/8 experts at 512, vocabulary
// 248320, capacity 16384.
constexpr auto kGeometry =
    make_decode_geometry(tatara::model::qwen36::generated::kModelPlan, 16384);

static_assert(kGeometry.gated_delta_layers == 30);
static_assert(kGeometry.attention_layers == 10);
static_assert(kGeometry.hidden_bytes == 4096);
static_assert(kGeometry.layer_stream_bytes == 40ull * 4096);
static_assert(kGeometry.gdn_projection_bytes == 24704);
static_assert(kGeometry.gdn_qk_bytes == 8192);
static_assert(kGeometry.gdn_value_bytes == 8192);
static_assert(kGeometry.gdn_conv_state_bytes == 49152);
static_assert(kGeometry.gdn_recurrent_state_bytes == 2097152);
static_assert(kGeometry.attn_projection_bytes == 18432);
static_assert(kGeometry.attn_query_bytes == 8192);
static_assert(kGeometry.attn_record_scratch_bytes == 16ull * 64 * 258 * 4);
static_assert(kGeometry.attn_cache_bytes == 2ull * 16384 * 256 * 2);
static_assert(kGeometry.router_logits_bytes == 257 * 4);
static_assert(kGeometry.expert_id_bytes == 32);
static_assert(kGeometry.expert_hidden_bytes == 9216);
static_assert(kGeometry.logits_bytes == 496640);
static_assert(kGeometry.argmax_value_bytes == 1024);
static_assert(kGeometry.token_id_bytes == 4);

// Every dispatch argument the sealed token walk issues, locked to the shape
// the sealed encoder was measured with. These are the literals the walk
// carried inline before the geometry derived them; a derivation that moves
// any of them is a numerical change, not a refactor.
constexpr DecodeDispatch kDispatch = kGeometry.dispatch;

static_assert(kDispatch.embed.groups == 8 && kDispatch.embed.threads == 256);
static_assert(kDispatch.rms.groups == 1 && kDispatch.rms.threads == 512);
static_assert(kDispatch.gdn_project.groups == 3088 && kDispatch.gdn_project.threads == 128);
static_assert(kDispatch.gdn_prepare.groups == 96 && kDispatch.gdn_prepare.threads == 128);
static_assert(kDispatch.gdn_recurrence.dimension_groups == 32 &&
              kDispatch.gdn_recurrence.head_groups == 32 &&
              kDispatch.gdn_recurrence.lane_threads == 32 &&
              kDispatch.gdn_recurrence.dimension_threads == 4);
static_assert(kDispatch.gdn_gate_norm.groups == 32 && kDispatch.gdn_gate_norm.threads == 128);
static_assert(kDispatch.out_projection.groups == 1024 && kDispatch.out_projection.threads == 64);
static_assert(kDispatch.attn_project.groups == 2304 && kDispatch.attn_project.threads == 128);
static_assert(kDispatch.attn_qk_rope.groups == 18 && kDispatch.attn_qk_rope.threads == 256);
static_assert(kDispatch.attention_head.groups == 16 && kDispatch.attention_head.threads == 256);
static_assert(kDispatch.attention_split.score_groups == 4 &&
              kDispatch.attention_split.score_threads == 256 &&
              kDispatch.attention_split.score_cohort_threads == 4);
static_assert(kDispatch.attention_split.value_groups == 2 &&
              kDispatch.attention_split.value_lane_threads == 32 &&
              kDispatch.attention_split.value_dimension_threads == 4 &&
              kDispatch.attention_split.value_head_threads == 8);
static_assert(kDispatch.router.groups == 257 && kDispatch.router.threads == 32);
static_assert(kDispatch.router_select.groups == 1 && kDispatch.router_select.threads == 256);
static_assert(kDispatch.grouped_upgate.groups == 4608 && kDispatch.grouped_upgate.threads == 32);
static_assert(kDispatch.grouped_down_res.groups == 2048 &&
              kDispatch.grouped_down_res.threads == 32);
static_assert(kDispatch.lmhead.groups == 248320 && kDispatch.lmhead.threads == 32);
static_assert(kDispatch.argmax_stage1.groups == 256 && kDispatch.argmax_stage1.threads == 256);
static_assert(kDispatch.vocabulary_rows == 248320);
static_assert(kDispatch.top_experts == 8);

// The kernels the walk dispatches without a row guard must be covered
// exactly: the embedding writes one element per grid thread, and the RMS
// family reads four contiguous elements per thread across one threadgroup.
static_assert(kDispatch.embed.groups * kDispatch.embed.threads == 2048);
static_assert(kDispatch.rms.threads * kRmsValuesPerThread == 2048);

// Row-kernel coverage, the invariant the row dispatches actually owe. Each
// pins that the grid carries one simdgroup per row for every row, and that
// the trailing threadgroup is the minimal one -- dropping a threadgroup would
// leave rows uncomputed. The threadgroup literals above conflate that
// numerical contract with the scheduling choice of how simdgroups are bundled;
// only the former is output-affecting, and these hold under any repacking. A
// width change that silently dropped or double-covered a row would fail here
// even though the literals above were updated to match it.
constexpr bool covers_rows(DispatchShape shape, std::uint32_t rows) {
    const std::uint32_t per_group = shape.threads / kSimdgroupThreads;
    return shape.threads % kSimdgroupThreads == 0 && shape.groups * per_group >= rows &&
           (shape.groups - 1u) * per_group < rows;
}

static_assert(covers_rows(kDispatch.gdn_project, 12352));
static_assert(covers_rows(kDispatch.out_projection, 2048));
static_assert(covers_rows(kDispatch.attn_project, 9216));
static_assert(covers_rows(kDispatch.router, 257));
static_assert(covers_rows(kDispatch.grouped_upgate, 4608));
static_assert(covers_rows(kDispatch.grouped_down_res, 2048));
static_assert(covers_rows(kDispatch.lmhead, 248320));

static_assert(context_in_capacity(0, 1));
static_assert(context_in_capacity(16383, 16384));
static_assert(!context_in_capacity(16384, 16384));
static_assert(!context_in_capacity(0, 0));

} // namespace

int main() {
    return 0;
}
