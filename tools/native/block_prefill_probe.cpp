#include "decode_harness.h"

#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/generated/kernel_library.h"
#include "tatara/generated/model_plan.h"
#include "tatara/runtime/decode_step.h"
#include "tatara/runtime/prefill_geometry.h"
#include "tatara/runtime/prefill_profile_plan.h"
#include "tatara/runtime/prefill_profile_report.h"
#include "tatara/runtime/prefill_step.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Guarded real-weight equality fixture for the exact block-prefill seam.
//
// For an N-token prompt, the block path consumes ids[0:N-1), commits its
// persistent state only after that command buffer completes, then the sealed
// single-token executor consumes ids[N-1] at context N-1. The resulting state
// record and fixed continuation are compared externally with the explicit
// same-binary serial policy over all N ids.

namespace {

using namespace tatara::backend::metal;
using namespace tatara::runtime;
using tatara::model::qwen36::LayerKind;

constexpr std::uint32_t kFixtureControlBlock = 256;
constexpr std::uint32_t kFixtureControlQueryTile = 16;
constexpr std::uint32_t kFixtureProductBlock = 2048;
constexpr std::uint32_t kFixtureProductQueryTile = 256;
constexpr std::uint32_t kFixtureExactRows = 16;
constexpr std::uint32_t kFixtureContinuationTokens = 16;

constexpr int kBootExitBase = 40;
constexpr int kExitUsage = 90;
constexpr int kExitIdsUnreadable = 91;
constexpr int kExitIdsInvalid = 92;
constexpr int kExitPositions = 93;
constexpr int kExitLibrary = 94;
constexpr int kExitFunction = 95;
constexpr int kExitPipeline = 96;
constexpr int kExitGeometry = 97;
constexpr int kExitPrefillStep = 98;
constexpr int kExitCommandBuffer = 100;
constexpr int kExitComputePass = 101;
constexpr int kExitEncode = 102;
constexpr int kExitEndPass = 103;
constexpr int kExitCommit = 104;
constexpr int kExitExecution = 105;
constexpr int kExitDecodeEncode = 106;
constexpr int kExitTokenRange = 107;
constexpr int kExitStalled = 108;
constexpr int kExitDump = 109;
constexpr int kExitPolicy = 110;
constexpr int kExitCapacity = 111;
constexpr int kExitProfilePlan = 112;
constexpr int kExitCounterPlan = 113;
constexpr int kExitCounterBuffer = 114;
constexpr int kExitProfile = 115;
constexpr int kExitCounterResolve = 116;
constexpr int kExitProfileReport = 117;
constexpr int kExitExpertCountCaptureContract = 118;
constexpr int kExitExpertCountCaptureWrite = 119;
constexpr int kExitCommandGraph = 120;
constexpr int kExitGraphSerialDiagnostic = 121;
constexpr int kExitGraphReplayDiagnostic = 122;
constexpr int kExitGraphLevelProfile = 123;

// Set once in main: replays the prepared ICB one command per execution
// range so recorded-command content is discriminated from concurrency.
bool graph_serial_diagnostic_requested = false;

// Set once in main: after the correctness flow completes, re-executes the
// cached ICB twice and reports each wall, measuring steady-state replay
// cost. The re-executions deliberately run after the state record and
// continuation, so the scratch they clobber is never read again.
bool graph_repeat_requested = false;

// Set once in main: runs the complete product prefill lifecycle this many
// times in one process, resetting the state slot between requests, so
// requests after the first must hit the cached graph. The final request's
// state feeds the ordinary record and continuation instrument.
bool graph_warm_requested = false;
constexpr std::uint32_t kGraphWarmRequests = 4;

// Set once in main: executes the normal graph path through bounded global
// bands with at most three scratch lanes. This is the physical equivalence
// arm for the serving scheduler, not a performance policy.
bool graph_banded_requested = false;
constexpr std::uint32_t kGraphBandedScratchLanes = 3;

// Set once in main: after commit, re-executes the cached ICB one level per
// command buffer and aggregates GPU time by recorded command class. Timing
// diagnostic only; exits typed without publishing state.
bool graph_level_profile_requested = false;

struct FixturePolicy {
    std::string_view name;
    bool serial;
    bool profile;
    CounterSamplingMode sampling_mode;
    PrefillSchedule schedule;
    std::uint32_t maximum_block;
    std::uint32_t first_chunk;
    std::uint32_t query_tile;
    PrefillRouterSelector router_selector;
    PrefillGdnRecurrence gdn_recurrence;
    bool gdn_gate_hoist;
    PrefillAttentionKernel attention_kernel{
        PrefillAttentionKernel::PartialCombine};
    QuantizedGemmPolicy dense_qgemm{QuantizedGemmPolicy::ExactRow};
    QuantizedGemmPolicy routed_qgemm{QuantizedGemmPolicy::ExactRow};
    bool native_dense_steel{false};
    bool native_dense_steel_gdn_bm64_wm2_wn2{false};
    bool native_routed_shared_expert{false};
    bool native_routed_steel{false};
    bool command_graph{false};
    bool command_graph_lane_events{false};
    bool command_graph_compile_only{false};
    bool command_graph_serial_diagnostic{false};
    bool command_graph_repeat{false};
    bool command_graph_warm{false};
    bool command_graph_banded{false};
    bool command_graph_level_profile{false};
    std::uint32_t maximum_units_per_submission{1};
    std::uint32_t maximum_inflight_units{1};
};

constexpr bool same_execution_policy(const FixturePolicy& left,
                                     const FixturePolicy& right) noexcept {
    return left.serial == right.serial && left.schedule == right.schedule &&
           left.maximum_block == right.maximum_block &&
           left.first_chunk == right.first_chunk && left.query_tile == right.query_tile &&
           left.router_selector == right.router_selector &&
           left.gdn_recurrence == right.gdn_recurrence &&
           left.gdn_gate_hoist == right.gdn_gate_hoist &&
           left.attention_kernel == right.attention_kernel &&
           left.dense_qgemm == right.dense_qgemm &&
           left.routed_qgemm == right.routed_qgemm &&
           left.native_dense_steel ==
               right.native_dense_steel &&
           left.native_dense_steel_gdn_bm64_wm2_wn2 ==
               right.native_dense_steel_gdn_bm64_wm2_wn2 &&
           left.native_routed_shared_expert ==
               right.native_routed_shared_expert &&
           left.native_routed_steel ==
               right.native_routed_steel &&
           left.command_graph == right.command_graph &&
           left.command_graph_lane_events ==
               right.command_graph_lane_events &&
           left.maximum_units_per_submission ==
               right.maximum_units_per_submission &&
           left.maximum_inflight_units ==
               right.maximum_inflight_units;
}

constexpr FixturePolicy kChunk256{
    .name = "chunk256",
    .serial = false,
    .profile = false,
    .sampling_mode = CounterSamplingMode::DispatchBoundary,
    .schedule = PrefillSchedule::ChunkMajor,
    .maximum_block = kFixtureControlBlock,
    .first_chunk = kFixtureControlBlock,
    .query_tile = kFixtureControlQueryTile,
    .router_selector = PrefillRouterSelector::Serial,
    .gdn_recurrence = PrefillGdnRecurrence::SerialSteps,
    .gdn_gate_hoist = false,
};

constexpr FixturePolicy kLayer2048Fast{
    .name = "layer2048fast",
    .serial = false,
    .profile = false,
    .sampling_mode = CounterSamplingMode::DispatchBoundary,
    .schedule = PrefillSchedule::LayerMajor,
    .maximum_block = kFixtureProductBlock,
    .first_chunk = kFixtureControlBlock,
    .query_tile = kFixtureProductQueryTile,
    .router_selector = PrefillRouterSelector::Parallel,
    .gdn_recurrence = PrefillGdnRecurrence::RegisterLoop,
    .gdn_gate_hoist = true,
};

constexpr FixturePolicy kLayer2048FastProfileSerial{
    .name = "layer2048fast-profile-serial",
    .serial = false,
    .profile = true,
    .sampling_mode = CounterSamplingMode::DispatchBoundary,
    .schedule = PrefillSchedule::LayerMajor,
    .maximum_block = kFixtureProductBlock,
    .first_chunk = kFixtureControlBlock,
    .query_tile = kFixtureProductQueryTile,
    .router_selector = PrefillRouterSelector::Serial,
    .gdn_recurrence = PrefillGdnRecurrence::RegisterLoop,
    .gdn_gate_hoist = true,
};

constexpr FixturePolicy kLayer2048FastProfile{
    .name = "layer2048fast-profile",
    .serial = false,
    .profile = true,
    .sampling_mode = CounterSamplingMode::DispatchBoundary,
    .schedule = PrefillSchedule::LayerMajor,
    .maximum_block = kFixtureProductBlock,
    .first_chunk = kFixtureControlBlock,
    .query_tile = kFixtureProductQueryTile,
    .router_selector = PrefillRouterSelector::Parallel,
    .gdn_recurrence = PrefillGdnRecurrence::RegisterLoop,
    .gdn_gate_hoist = true,
};

static_assert(!kLayer2048Fast.profile && kLayer2048FastProfile.profile);
static_assert(same_execution_policy(kLayer2048Fast, kLayer2048FastProfile));

constexpr FixturePolicy kLayer2048FastProfileStageSerial{
    .name = "layer2048fast-profile-stage-serial",
    .serial = false,
    .profile = true,
    .sampling_mode = CounterSamplingMode::StageBoundaryEncoderSplit,
    .schedule = PrefillSchedule::LayerMajor,
    .maximum_block = kFixtureProductBlock,
    .first_chunk = kFixtureControlBlock,
    .query_tile = kFixtureProductQueryTile,
    .router_selector = PrefillRouterSelector::Serial,
    .gdn_recurrence = PrefillGdnRecurrence::RegisterLoop,
    .gdn_gate_hoist = true,
};

constexpr FixturePolicy kLayer2048FastProfileStage{
    .name = "layer2048fast-profile-stage",
    .serial = false,
    .profile = true,
    .sampling_mode = CounterSamplingMode::StageBoundaryEncoderSplit,
    .schedule = PrefillSchedule::LayerMajor,
    .maximum_block = kFixtureProductBlock,
    .first_chunk = kFixtureControlBlock,
    .query_tile = kFixtureProductQueryTile,
    .router_selector = PrefillRouterSelector::Parallel,
    .gdn_recurrence = PrefillGdnRecurrence::RegisterLoop,
    .gdn_gate_hoist = true,
};

static_assert(same_execution_policy(kLayer2048Fast, kLayer2048FastProfileStage));

bool parse_fixture_policy(std::string_view text, FixturePolicy& policy) {
    if (text == "serial") {
        policy = {
            .name = "serial",
            .serial = true,
            .profile = false,
            .sampling_mode = CounterSamplingMode::DispatchBoundary,
            .schedule = PrefillSchedule::ChunkMajor,
            .maximum_block = 0,
            .first_chunk = 0,
            .query_tile = 0,
            .router_selector = PrefillRouterSelector::Serial,
            .gdn_recurrence = PrefillGdnRecurrence::SerialSteps,
            .gdn_gate_hoist = false,
        };
    } else if (text == "chunk256") {
        policy = kChunk256;
    } else if (text == "chunk2048") {
        policy = {
            .name = "chunk2048",
            .serial = false,
            .profile = false,
            .sampling_mode = CounterSamplingMode::DispatchBoundary,
            .schedule = PrefillSchedule::ChunkMajor,
            .maximum_block = kFixtureProductBlock,
            .first_chunk = kFixtureProductBlock,
            .query_tile = kFixtureProductQueryTile,
            .router_selector = PrefillRouterSelector::Serial,
            .gdn_recurrence = PrefillGdnRecurrence::SerialSteps,
            .gdn_gate_hoist = false,
        };
    } else if (text == "layer2048") {
        policy = {
            .name = "layer2048",
            .serial = false,
            .profile = false,
            .sampling_mode = CounterSamplingMode::DispatchBoundary,
            .schedule = PrefillSchedule::LayerMajor,
            .maximum_block = kFixtureProductBlock,
            .first_chunk = kFixtureControlBlock,
            .query_tile = kFixtureProductQueryTile,
            .router_selector = PrefillRouterSelector::Serial,
            .gdn_recurrence = PrefillGdnRecurrence::SerialSteps,
            .gdn_gate_hoist = false,
        };
    } else if (text == "layer2048fast") {
        policy = kLayer2048Fast;
    } else if (text == "layer2048fast-n1") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
    } else if (text == "layer2048fast-r2") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
    } else if (text == "layer2048fast-n1-r2") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
    } else if (text == "layer2048fast-n1-r2-a1") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
    } else if (text == "layer2048fast-n1-r2s-a1") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_routed_shared_expert = true;
    } else if (text == "layer2048fast-n1-r2s-a1-steel") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
    } else if (text == "layer2048fast-steel-full") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
    } else if (
        text ==
        "layer2048fast-steel-full-graph-compile") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
        policy.command_graph_compile_only = true;
    } else if (
        text ==
        "layer2048fast-steel-full-graph-lane-events-compile") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
        policy.command_graph_lane_events = true;
        policy.command_graph_compile_only = true;
    } else if (
        text == "layer2048fast-steel-full-graph") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
    } else if (
        text ==
        "layer2048fast-steel-full-graph-bm64-wm2-wn2-compile") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_dense_steel_gdn_bm64_wm2_wn2 =
            true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
        policy.command_graph_compile_only = true;
    } else if (
        text ==
        "layer2048fast-steel-full-graph-bm64-wm2-wn2-warm") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_dense_steel_gdn_bm64_wm2_wn2 =
            true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
        policy.command_graph_warm = true;
    } else if (
        text ==
        "layer2048fast-steel-full-graph-streaming") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StreamingFlashAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
    } else if (
        text ==
        "layer2048fast-steel-full-graph-streaming-warm") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StreamingFlashAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
        policy.command_graph_warm = true;
    } else if (
        text ==
        "layer2048fast-steel-full-graph-banded3") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
        policy.command_graph_banded = true;
    } else if (
        text ==
        "layer2048fast-steel-full-graph-levelprof") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
        policy.command_graph_level_profile = true;
    } else if (
        text == "layer2048fast-steel-full-graph-warm") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
        policy.command_graph_warm = true;
    } else if (
        text ==
        "layer2048fast-steel-full-graph-lane-events-warm") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
        policy.command_graph_lane_events = true;
        policy.command_graph_warm = true;
    } else if (
        text ==
        "layer2048fast-steel-full-graph-qtile2048-warm") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.query_tile = kFixtureProductBlock;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
        policy.command_graph_warm = true;
    } else if (
        text == "layer2048fast-steel-full-graph-repeat") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
        policy.command_graph_repeat = true;
    } else if (
        text ==
        "layer2048fast-steel-full-graph-serialdiag") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.command_graph = true;
        policy.command_graph_serial_diagnostic = true;
    } else if (
        text == "layer2048fast-steel-full-submit2" ||
        text == "layer2048fast-steel-full-submit4" ||
        text == "layer2048fast-steel-full-submit8" ||
        text == "layer2048fast-steel-full-submit16" ||
        text == "layer2048fast-steel-full-submit32" ||
        text == "layer2048fast-steel-full-submit64") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.maximum_units_per_submission =
            text.ends_with("submit2")
                ? 2
                : text.ends_with("submit4")
                      ? 4
                      : text.ends_with("submit8")
                            ? 8
                            : text.ends_with("submit16")
                                  ? 16
                                  : text.ends_with("submit32") ? 32 : 64;
    } else if (
        text == "layer2048fast-steel-full-inflight2" ||
        text == "layer2048fast-steel-full-inflight4" ||
        text == "layer2048fast-steel-full-inflight8" ||
        text == "layer2048fast-steel-full-inflight16" ||
        text == "layer2048fast-steel-full-inflight32" ||
        text == "layer2048fast-steel-full-inflight64") {
        policy = kLayer2048Fast;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm =
            QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
        policy.maximum_inflight_units =
            text.ends_with("inflight2")
                ? 2
                : text.ends_with("inflight4")
                      ? 4
                      : text.ends_with("inflight8")
                            ? 8
                            : text.ends_with("inflight16")
                                  ? 16
                                  : text.ends_with("inflight32") ? 32 : 64;
    } else if (text == "layer2048fast-profile") {
        policy = kLayer2048FastProfile;
    } else if (text == "layer2048fast-profile-n1") {
        policy = kLayer2048FastProfile;
        policy.name = text;
        policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
    } else if (text == "layer2048fast-profile-serial") {
        policy = kLayer2048FastProfileSerial;
    } else if (text == "layer2048fast-profile-stage") {
        policy = kLayer2048FastProfileStage;
    } else if (text == "layer2048fast-profile-stage-n1") {
        policy = kLayer2048FastProfileStage;
        policy.name = text;
        policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
    } else if (text == "layer2048fast-profile-stage-r2") {
        policy = kLayer2048FastProfileStage;
        policy.name = text;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
    } else if (
        text ==
        "layer2048fast-profile-stage-n1-r2s-a1") {
        policy = kLayer2048FastProfileStage;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_routed_shared_expert = true;
    } else if (
        text ==
        "layer2048fast-profile-stage-steel-full") {
        policy = kLayer2048FastProfileStage;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
        policy.native_dense_steel = true;
        policy.native_routed_shared_expert = true;
        policy.native_routed_steel = true;
    } else if (text == "layer2048fast-profile-stage-n1-r2") {
        policy = kLayer2048FastProfileStage;
        policy.name = text;
        policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
    } else if (
        text ==
        "layer2048fast-profile-stage-n1-r2-a1") {
        policy = kLayer2048FastProfileStage;
        policy.name = text;
        policy.attention_kernel =
            PrefillAttentionKernel::StagedGemmAdaptive;
        policy.dense_qgemm = QuantizedGemmPolicy::NativeDenseMma;
        policy.routed_qgemm =
            QuantizedGemmPolicy::NativeRaggedMma;
    } else if (text == "layer2048fast-profile-stage-serial") {
        policy = kLayer2048FastProfileStageSerial;
    } else {
        return false;
    }
    return true;
}

struct PromptIds {
    bool valid{false};
    std::vector<std::uint32_t> ids;
};

PromptIds parse_prompt_ids(std::span<const std::byte> bytes, std::uint32_t vocabulary,
                           std::uint32_t maximum_positions) {
    PromptIds parsed;
    if (bytes.size() < sizeof(std::uint32_t)) {
        std::cerr << "ids file is shorter than its header\n";
        return parsed;
    }
    std::uint32_t count = 0;
    std::memcpy(&count, bytes.data(), sizeof(count));
    if (count == 0 || count > maximum_positions) {
        std::cerr << "ids count " << count << " outside 1.." << maximum_positions << '\n';
        return parsed;
    }
    const std::size_t expected = sizeof(std::uint32_t) + std::size_t{count} * sizeof(std::uint32_t);
    if (bytes.size() != expected) {
        std::cerr << "ids file is " << bytes.size() << " bytes, expected " << expected << '\n';
        return parsed;
    }
    parsed.ids.resize(count);
    std::memcpy(parsed.ids.data(), bytes.data() + sizeof(std::uint32_t),
                std::size_t{count} * sizeof(std::uint32_t));
    for (std::size_t index = 0; index < parsed.ids.size(); ++index) {
        if (parsed.ids[index] >= vocabulary) {
            std::cerr << "prompt id " << index << " is " << parsed.ids[index]
                      << ", outside the vocabulary\n";
            return parsed;
        }
    }
    parsed.valid = true;
    return parsed;
}

bool parse_u32(std::string_view text, std::uint32_t& value) {
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return parsed.ec == std::errc{} && parsed.ptr == text.data() + text.size();
}

struct PipelineResult {
    int exit_code{kExitLibrary};
    PrefillPipelines pipelines;

    explicit operator bool() const {
        return exit_code == 0;
    }
};

PipelineResult resolve_prefill_pipelines(
    const MetalDevice& device, const MetalLibrary& library,
    bool native_dense_qgemm, bool native_dense_steel,
    bool native_dense_steel_gdn_bm64_wm2_wn2,
    bool native_routed_qgemm,
    bool native_routed_steel, bool staged_attention,
    bool command_graph) {
    PipelineResult result;
    const auto create_pipeline =
        [&](const MetalFunction& function) {
            return command_graph
                       ? create_indirect_compute_pipeline(
                             device, function)
                       : create_compute_pipeline(device, function);
        };
    constexpr std::array<std::string_view, 23> names{
        "embed_rows_q4",
        "rms_blk",
        "residual_blk",
        "gdn_project_blk",
        "gdn_conv_blk",
        "gdn_gates_blk",
        "gdn_recurrence",
        "gdn_recurrence_blk",
        "gdn_recurrence_gates_blk",
        "gdn_gate_norm_blk",
        "attn_project_blk",
        "attn_qk_rope_blk",
        "attention_partial_blk",
        "attention_combine_blk",
        "attention_streaming_blk",
        "outproj_blk",
        "router_q8_block",
        "router_select_block",
        "router_select_block_parallel",
        "expert_union",
        "block_upgate",
        "block_down_partial",
        "block_down_combine",
    };
    MetalComputePipeline* slots[] = {
        &result.pipelines.embed,
        &result.pipelines.rms,
        &result.pipelines.residual,
        &result.pipelines.gdn_project,
        &result.pipelines.gdn_conv,
        &result.pipelines.gdn_gates,
        &result.pipelines.gdn_recurrence_step,
        &result.pipelines.gdn_recurrence_block,
        &result.pipelines.gdn_recurrence_gates,
        &result.pipelines.gdn_gate_norm,
        &result.pipelines.attn_project,
        &result.pipelines.attn_qk_rope,
        &result.pipelines.attention_partial,
        &result.pipelines.attention_combine,
        &result.pipelines.attention_streaming,
        &result.pipelines.out_projection,
        &result.pipelines.router,
        &result.pipelines.router_select_serial,
        &result.pipelines.router_select_parallel,
        &result.pipelines.expert_union,
        &result.pipelines.expert_upgate,
        &result.pipelines.expert_down,
        &result.pipelines.expert_combine,
    };
    static_assert(names.size() == std::size(slots));
    for (std::size_t index = 0; index < names.size(); ++index) {
        auto function = create_function(library, names[index]);
        if (!function) {
            std::cerr << "prefill function lookup failed: " << names[index] << '\n';
            result.exit_code = kExitFunction;
            return result;
        }
        auto pipeline = create_pipeline(*function.function);
        if (!pipeline) {
            std::cerr << "prefill pipeline creation failed: " << names[index] << '\n';
            result.exit_code = kExitPipeline;
            return result;
        }
        *slots[index] = std::move(*pipeline.pipeline);
    }
    if (staged_attention) {
        constexpr std::array<std::string_view, 9> staged_names{
            "attention_staged_scores_blk",
            "attention_staged_softmax_blk",
            "attention_staged_values_blk",
            "tatara_mlx_steel_attn_scores_nt_unaligned",
            "tatara_mlx_steel_attn_values_nn_unaligned",
            "tatara_mlx_steel_attn_scores_nt_unaligned_l",
            "tatara_mlx_steel_attn_values_nn_unaligned_l",
            "attention_staged_softmax_bf16_blk",
            "attention_gate_apply_blk",
        };
        MetalComputePipeline* staged_slots[]{
            &result.pipelines.attention_staged_scores,
            &result.pipelines.attention_staged_softmax,
            &result.pipelines.attention_staged_values,
            &result.pipelines.attention_steel_scores,
            &result.pipelines.attention_steel_values,
            &result.pipelines.attention_steel_scores_large,
            &result.pipelines.attention_steel_values_large,
            &result.pipelines.attention_softmax_bf16,
            &result.pipelines.attention_gate_apply,
        };
        static_assert(
            staged_names.size() == std::size(staged_slots));
        for (std::size_t index = 0;
             index < staged_names.size(); ++index) {
            auto function =
                create_function(library, staged_names[index]);
            if (!function) {
                std::cerr << "prefill function lookup failed: "
                          << staged_names[index] << '\n';
                result.exit_code = kExitFunction;
                return result;
            }
            auto pipeline =
                create_pipeline(*function.function);
            if (!pipeline) {
                std::cerr << "prefill pipeline creation failed: "
                          << staged_names[index] << '\n';
                result.exit_code = kExitPipeline;
                return result;
            }
            *staged_slots[index] =
                std::move(*pipeline.pipeline);
        }
    }
    if (native_dense_qgemm) {
        auto function = create_function(
            library, "native_dense_qgemm_q4_bf16_n1");
        if (!function) {
            std::cerr
                << "prefill function lookup failed: "
                << "native_dense_qgemm_q4_bf16_n1\n";
            result.exit_code = kExitFunction;
            return result;
        }
        auto pipeline =
            create_pipeline(*function.function);
        if (!pipeline) {
            std::cerr
                << "prefill pipeline creation failed: "
                << "native_dense_qgemm_q4_bf16_n1\n";
            result.exit_code = kExitPipeline;
            return result;
        }
        result.pipelines.native_dense_qgemm =
            std::move(*pipeline.pipeline);
    }
    if (native_dense_steel) {
        if constexpr (
            !tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            result.exit_code = kExitPipeline;
            return result;
        }
        auto function = create_function(
            library,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelDenseKernelName);
        if (!function) {
            std::cerr
                << "prefill function lookup failed: "
                << tatara::backend::metal::generated::
                       kKernelLibraryMlxSteelDenseKernelName
                << '\n';
            result.exit_code = kExitFunction;
            return result;
        }
        auto pipeline =
            create_pipeline(*function.function);
        if (!pipeline) {
            std::cerr
                << "prefill pipeline creation failed: "
                << tatara::backend::metal::generated::
                       kKernelLibraryMlxSteelDenseKernelName
                << '\n';
            result.exit_code = kExitPipeline;
            return result;
        }
        result.pipelines.native_dense_steel =
            std::move(*pipeline.pipeline);
    }
    if (native_dense_steel_gdn_bm64_wm2_wn2) {
        if constexpr (
            !tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            result.exit_code = kExitPipeline;
            return result;
        }
        auto function = create_function(
            library,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelGdnBm64Wm2Wn2KernelName);
        if (!function) {
            std::cerr
                << "prefill function lookup failed: "
                << tatara::backend::metal::generated::
                       kKernelLibraryMlxSteelGdnBm64Wm2Wn2KernelName
                << '\n';
            result.exit_code = kExitFunction;
            return result;
        }
        auto pipeline =
            create_pipeline(*function.function);
        if (!pipeline) {
            std::cerr
                << "prefill pipeline creation failed: "
                << tatara::backend::metal::generated::
                       kKernelLibraryMlxSteelGdnBm64Wm2Wn2KernelName
                << '\n';
            result.exit_code = kExitPipeline;
            return result;
        }
        result.pipelines
            .native_dense_steel_gdn_bm64_wm2_wn2 =
            std::move(*pipeline.pipeline);
    }
    if (native_routed_qgemm) {
        constexpr std::array<std::string_view, 3> native_names{
            "native_routed_qgemm_r1_build_tasks",
            "native_routed_qgemm_r2_fused_upgate_swiglu",
            "native_routed_qgemm_r2_down_partial",
        };
        MetalComputePipeline* native_slots[]{
            &result.pipelines.native_routed_task_builder,
            &result.pipelines.native_routed_upgate,
            &result.pipelines.native_routed_down,
        };
        static_assert(
            native_names.size() == std::size(native_slots));
        for (std::size_t index = 0;
             index < native_names.size(); ++index) {
            auto function =
                create_function(library, native_names[index]);
            if (!function) {
                std::cerr << "prefill function lookup failed: "
                          << native_names[index] << '\n';
                result.exit_code = kExitFunction;
                return result;
            }
            auto pipeline =
                create_pipeline(*function.function);
            if (!pipeline) {
                std::cerr << "prefill pipeline creation failed: "
                          << native_names[index] << '\n';
                result.exit_code = kExitPipeline;
                return result;
            }
            *native_slots[index] =
                std::move(*pipeline.pipeline);
        }
    }
    if (native_routed_steel) {
        if constexpr (
            !tatara::backend::metal::generated::
                kKernelLibraryMlxSteelEnabled) {
            result.exit_code = kExitPipeline;
            return result;
        }
        constexpr std::array<std::string_view, 2> steel_names{
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelRoutedUpgateKernelName,
            tatara::backend::metal::generated::
                kKernelLibraryMlxSteelRoutedDownKernelName,
        };
        MetalComputePipeline* steel_slots[]{
            &result.pipelines.native_routed_steel_upgate,
            &result.pipelines.native_routed_steel_down,
        };
        static_assert(
            steel_names.size() == std::size(steel_slots));
        for (std::size_t index = 0;
             index < steel_names.size(); ++index) {
            auto function =
                create_function(library, steel_names[index]);
            if (!function) {
                std::cerr << "prefill function lookup failed: "
                          << steel_names[index] << '\n';
                result.exit_code = kExitFunction;
                return result;
            }
            auto pipeline =
                create_pipeline(*function.function);
            if (!pipeline) {
                std::cerr << "prefill pipeline creation failed: "
                          << steel_names[index] << '\n';
                result.exit_code = kExitPipeline;
                return result;
            }
            *steel_slots[index] =
                std::move(*pipeline.pipeline);
        }
    }
    if (command_graph) {
        auto function = create_function(
            library, "expert_union_fused_tasks");
        if (!function) {
            std::cerr
                << "prefill function lookup failed: "
                << "expert_union_fused_tasks\n";
            result.exit_code = kExitFunction;
            return result;
        }
        auto pipeline =
            create_indirect_compute_pipeline(
                device, *function.function);
        if (!pipeline) {
            std::cerr
                << "prefill indirect pipeline creation failed: "
                << "expert_union_fused_tasks\n";
            result.exit_code = kExitPipeline;
            return result;
        }
        result.pipelines.expert_union_fused_tasks =
            std::move(*pipeline.pipeline);
    }
    result.exit_code = 0;
    return result;
}

struct SubmissionResult {
    int exit_code{0};
    double seconds{0.0};
    double gpu_seconds{0.0};
    double schedule_seconds{0.0};
    std::uint32_t command_buffers{0};
    std::uint32_t timed_command_buffers{0};
    std::uint32_t chunks{0};
};

struct ExpertCountCaptureRecord {
    std::uint32_t layer_index{0};
    std::uint32_t chunk_ordinal{0};
    std::uint32_t chunk_rows{0};
    std::size_t count_offset{0};
};

struct ExpertCountCapture {
    std::uint32_t prefill_rows{0};
    std::uint32_t layer_count{0};
    std::uint32_t experts{0};
    std::uint32_t active_experts{0};
    std::uint32_t chunk_count{0};
    std::uint32_t first_chunk_rows{0};
    std::uint32_t maximum_block_rows{0};
    std::size_t router_rows{0};
    std::vector<ExpertCountCaptureRecord> records;
    std::vector<std::uint32_t> counts;
    std::vector<std::uint8_t> written;
};

bool checked_size_multiply(std::size_t left, std::size_t right,
                           std::size_t& output) noexcept {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    output = left * right;
    return true;
}

bool checked_size_add(std::size_t left, std::size_t right,
                      std::size_t& output) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        return false;
    }
    output = left + right;
    return true;
}

bool prepare_expert_count_capture(const PrefillStep& prefill,
                                  std::uint32_t prefill_rows,
                                  std::size_t layer_count,
                                  std::uint32_t experts,
                                  std::uint32_t active_experts,
                                  const PrefillPolicy& policy,
                                  ExpertCountCapture& capture) {
    if (prefill_rows == 0 || layer_count == 0 ||
        layer_count > std::numeric_limits<std::uint32_t>::max() ||
        experts == 0 || active_experts == 0 || active_experts >= experts ||
        policy.schedule != PrefillSchedule::LayerMajor ||
        policy.first_chunk_rows == 0 ||
        policy.first_chunk_rows > policy.maximum_block_rows) {
        return false;
    }
    std::size_t chunk_count_size = 1u;
    if (prefill_rows > policy.first_chunk_rows) {
        const std::size_t remaining_rows =
            prefill_rows - policy.first_chunk_rows;
        const std::size_t tail_chunks =
            remaining_rows / policy.maximum_block_rows +
            (remaining_rows % policy.maximum_block_rows != 0u ? 1u : 0u);
        if (!checked_size_add(1u, tail_chunks, chunk_count_size) ||
            chunk_count_size >
                std::numeric_limits<std::uint32_t>::max()) {
            return false;
        }
    }
    const std::uint32_t chunk_count =
        static_cast<std::uint32_t>(chunk_count_size);
    std::size_t router_rows = 0;
    std::size_t record_count = 0;
    std::size_t word_count = 0;
    std::size_t count_bytes = 0;
    if (!checked_size_add(experts, 1u, router_rows) ||
        !checked_size_multiply(layer_count, chunk_count, record_count) ||
        !checked_size_multiply(record_count, router_rows, word_count) ||
        !checked_size_multiply(router_rows, sizeof(std::uint32_t), count_bytes) ||
        prefill.expert_counts.size_bytes() != count_bytes ||
        prefill.expert_counts.contents() == nullptr) {
        return false;
    }

    capture.prefill_rows = prefill_rows;
    capture.layer_count = static_cast<std::uint32_t>(layer_count);
    capture.experts = experts;
    capture.active_experts = active_experts;
    capture.chunk_count = chunk_count;
    capture.first_chunk_rows = policy.first_chunk_rows;
    capture.maximum_block_rows = policy.maximum_block_rows;
    capture.router_rows = router_rows;
    try {
        capture.records.resize(record_count);
        capture.counts.resize(word_count);
        capture.written.resize(record_count, 0u);
    } catch (...) {
        return false;
    }

    for (std::size_t layer = 0; layer < layer_count; ++layer) {
        for (std::uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
            const std::uint64_t chunk_offset =
                chunk == 0
                    ? 0
                    : std::uint64_t{policy.first_chunk_rows} +
                          std::uint64_t{chunk - 1u} *
                              policy.maximum_block_rows;
            if (chunk_offset >= prefill_rows) {
                return false;
            }
            const std::uint64_t remaining = prefill_rows - chunk_offset;
            const std::uint32_t chunk_limit =
                chunk == 0 ? policy.first_chunk_rows
                           : policy.maximum_block_rows;
            const std::uint32_t chunk_rows = static_cast<std::uint32_t>(
                remaining < chunk_limit ? remaining : chunk_limit);
            const std::size_t record_index =
                layer * std::size_t{chunk_count} + chunk;
            capture.records[record_index] = {
                .layer_index = static_cast<std::uint32_t>(layer),
                .chunk_ordinal = chunk,
                .chunk_rows = chunk_rows,
                .count_offset = record_index * router_rows,
            };
        }
    }
    return true;
}

bool capture_expert_counts(const PrefillStep& prefill,
                           const PrefillProgressResult& encoded,
                           ExpertCountCapture& capture) noexcept {
    if (encoded.layer_index >= capture.layer_count ||
        encoded.chunk_ordinal >= capture.chunk_count) {
        return false;
    }
    const std::size_t record_index =
        std::size_t{encoded.layer_index} * capture.chunk_count +
        encoded.chunk_ordinal;
    if (record_index >= capture.records.size() ||
        capture.written[record_index] != 0u) {
        return false;
    }
    const ExpertCountCaptureRecord& record = capture.records[record_index];
    if (record.layer_index != encoded.layer_index ||
        record.chunk_ordinal != encoded.chunk_ordinal ||
        record.count_offset > capture.counts.size() ||
        capture.router_rows > capture.counts.size() - record.count_offset) {
        return false;
    }
    const auto* source =
        static_cast<const std::uint32_t*>(prefill.expert_counts.contents());
    if (source == nullptr) {
        return false;
    }
    std::uint32_t* destination =
        capture.counts.data() + record.count_offset;
    std::memcpy(destination, source,
                capture.router_rows * sizeof(std::uint32_t));

    if (destination[capture.experts] != record.chunk_rows) {
        return false;
    }
    std::uint64_t routed_sum = 0;
    for (std::uint32_t expert = 0; expert < capture.experts; ++expert) {
        if (destination[expert] > record.chunk_rows) {
            return false;
        }
        routed_sum += destination[expert];
    }
    if (routed_sum !=
        std::uint64_t{record.chunk_rows} * capture.active_experts) {
        return false;
    }
    capture.written[record_index] = 1u;
    return true;
}

bool expert_count_capture_complete(const ExpertCountCapture& capture) noexcept {
    if (capture.records.empty() ||
        capture.records.size() != capture.written.size() ||
        capture.router_rows == 0 ||
        capture.counts.size() % capture.router_rows != 0 ||
        capture.counts.size() / capture.router_rows !=
            capture.records.size()) {
        return false;
    }
    for (const std::uint8_t written : capture.written) {
        if (written != 1u) {
            return false;
        }
    }
    return true;
}

std::uint64_t bf16_nan_count(const MetalBuffer& buffer) {
    const auto* words =
        static_cast<const std::uint16_t*>(buffer.contents());
    const std::uint64_t count =
        buffer.size_bytes() / sizeof(std::uint16_t);
    std::uint64_t nans = 0;
    for (std::uint64_t index = 0; index < count; ++index) {
        const std::uint16_t word = words[index];
        if ((word & 0x7f80u) == 0x7f80u &&
            (word & 0x007fu) != 0u) {
            ++nans;
        }
    }
    return nans;
}

std::uint64_t f32_nan_count(const MetalBuffer& buffer) {
    const auto* words =
        static_cast<const std::uint32_t*>(buffer.contents());
    const std::uint64_t count =
        buffer.size_bytes() / sizeof(std::uint32_t);
    std::uint64_t nans = 0;
    for (std::uint64_t index = 0; index < count; ++index) {
        const std::uint32_t word = words[index];
        if ((word & 0x7f800000u) == 0x7f800000u &&
            (word & 0x007fffffu) != 0u) {
            ++nans;
        }
    }
    return nans;
}

void print_graph_unit_statuses(const PrefillStep& prefill) {
    const std::uint32_t status_count =
        prefill.command_graph_task_status_count;
    const auto* status_bytes = static_cast<const std::byte*>(
        prefill.native_routed_up_status.contents());
    std::cerr << "unit statuses:";
    for (std::uint32_t slot = 0; slot < status_count; ++slot) {
        std::uint32_t value = 0;
        std::memcpy(
            &value,
            status_bytes +
                std::uint64_t{slot} *
                    sizeof(QuantizedGemmDeviceTaskStatus),
            sizeof(value));
        std::cerr << ' ' << value;
    }
    std::cerr << '\n';
}

void print_graph_failed_unit_ids(
    const PrefillStep& prefill,
    std::uint32_t failed_unit_offset) {
    const std::uint32_t status_count =
        prefill.command_graph_task_status_count;
    const auto chunk_count = static_cast<std::uint32_t>(
        prefill.command_graph.chunk_rows.size());
    if (failed_unit_offset >= status_count || chunk_count == 0) {
        return;
    }
    const std::uint32_t failed_chunk =
        failed_unit_offset % chunk_count;
    const std::uint32_t block =
        prefill.command_graph.chunk_rows[failed_chunk];
    const std::uint32_t experts = prefill.geometry.experts;
    const std::uint32_t active =
        prefill.geometry.active_experts;
    const auto* ids_bytes =
        static_cast<const std::byte*>(
            prefill.expert_ids.contents()) +
        std::uint64_t{failed_chunk} *
            prefill.geometry.moe_id_bytes;
    std::uint64_t out_of_range = 0;
    std::uint64_t duplicate_positions = 0;
    std::uint64_t routed_sum = 0;
    std::uint32_t reported_positions = 0;
    for (std::uint32_t position = 0; position < block;
         ++position) {
        bool position_duplicate = false;
        std::array<std::uint32_t, 32> seen{};
        const std::uint32_t checked = std::min(
            active, static_cast<std::uint32_t>(seen.size()));
        for (std::uint32_t slot = 0; slot < checked; ++slot) {
            std::uint32_t selected = 0;
            std::memcpy(
                &selected,
                ids_bytes +
                    (std::uint64_t{position} * active + slot) *
                        sizeof(std::uint32_t),
                sizeof(selected));
            seen[slot] = selected;
            if (selected >= experts) {
                ++out_of_range;
                continue;
            }
            ++routed_sum;
            for (std::uint32_t prior = 0; prior < slot;
                 ++prior) {
                if (seen[prior] == selected) {
                    position_duplicate = true;
                }
            }
        }
        if (position_duplicate) {
            ++duplicate_positions;
            if (reported_positions < 4) {
                std::cerr << "  position " << position
                          << " ids:";
                for (std::uint32_t slot = 0; slot < checked;
                     ++slot) {
                    std::cerr << ' ' << seen[slot];
                }
                std::cerr << '\n';
                ++reported_positions;
            }
        }
    }
    std::cerr << "failed unit ids: chunk=" << failed_chunk
              << " block=" << block
              << " out_of_range=" << out_of_range
              << " duplicate_positions=" << duplicate_positions
              << " routed_sum=" << routed_sum << " expected="
              << std::uint64_t{block} * active << '\n';
}

void print_graph_nan_census(const PrefillStep& prefill,
                            const DecodeStateSlot& state) {
    std::cerr << "state plane nan census"
                 " (f32 interpretation):\n";
    for (std::size_t layer = 0; layer < state.layers.size();
         ++layer) {
        const DecodeLayerState& planes = state.layers[layer];
        const std::uint64_t first_nans =
            f32_nan_count(planes.first);
        const std::uint64_t second_nans =
            f32_nan_count(planes.second);
        const std::uint64_t first_out_nans =
            planes.first_out ? f32_nan_count(planes.first_out)
                             : 0;
        const std::uint64_t second_out_nans =
            planes.second_out
                ? f32_nan_count(planes.second_out)
                : 0;
        if (first_nans != 0 || second_nans != 0 ||
            first_out_nans != 0 || second_out_nans != 0) {
            std::cerr << "  layer " << layer << " f32-nan:"
                      << " first=" << first_nans
                      << " first_out=" << first_out_nans
                      << " second=" << second_nans
                      << " second_out=" << second_out_nans
                      << '\n';
        }
    }
    std::cerr << "scratch nan census:"
              << " hidden_slab_bf16="
              << bf16_nan_count(prefill.hidden_slab)
              << " normalized_bf16="
              << bf16_nan_count(prefill.normalized)
              << " block_hidden_bf16="
              << bf16_nan_count(prefill.block_hidden)
              << " router_logits_f32="
              << f32_nan_count(prefill.router_logits)
              << " moe_output_bf16="
              << bf16_nan_count(prefill.moe_output)
              << " gdn_recurrence_f32="
              << f32_nan_count(prefill.gdn_recurrence)
              << " gdn_projection_bf16="
              << bf16_nan_count(prefill.gdn_projection)
              << '\n';
}

struct SampledBuffer {
    const char* name;
    const MetalBuffer* buffer;
    bool bf16;
};

constexpr std::uint64_t kNanSampleCount = 2048;

bool sampled_nan(const SampledBuffer& sampled) {
    const MetalBuffer& buffer = *sampled.buffer;
    if (!buffer) {
        return false;
    }
    if (sampled.bf16) {
        const auto* words = static_cast<const std::uint16_t*>(
            buffer.contents());
        const std::uint64_t count =
            buffer.size_bytes() / sizeof(std::uint16_t);
        const std::uint64_t stride =
            std::max<std::uint64_t>(1, count / kNanSampleCount);
        for (std::uint64_t index = 0; index < count;
             index += stride) {
            const std::uint16_t word = words[index];
            if ((word & 0x7f80u) == 0x7f80u &&
                (word & 0x007fu) != 0u) {
                return true;
            }
        }
        return false;
    }
    const auto* words =
        static_cast<const std::uint32_t*>(buffer.contents());
    const std::uint64_t count =
        buffer.size_bytes() / sizeof(std::uint32_t);
    const std::uint64_t stride =
        std::max<std::uint64_t>(1, count / kNanSampleCount);
    for (std::uint64_t index = 0; index < count;
         index += stride) {
        const std::uint32_t word = words[index];
        if ((word & 0x7f800000u) == 0x7f800000u &&
            (word & 0x007fffffu) != 0u) {
            return true;
        }
    }
    return false;
}

// Restores the fresh-slot condition the product reset produces: zeroed
// state planes and cleared phase flags. CPU-only on shared storage with no
// command in flight.
void reset_state_slot_for_warm_request(DecodeStateSlot& state) {
    for (DecodeLayerState& layer : state.layers) {
        for (MetalBuffer* plane :
             {&layer.first, &layer.first_out, &layer.second,
              &layer.second_out}) {
            if (*plane) {
                std::memset(
                    plane->contents(), 0,
                    static_cast<std::size_t>(
                        plane->size_bytes()));
            }
        }
        layer.swapped = false;
    }
}

MetalCommandError declare_graph_residency(
    MetalComputePass& pass,
    tatara::tools::DecodeHarness& harness,
    PrefillStep& prefill) {
    MetalCommandError residency = MetalCommandError::None;
    const auto declare = [&](const MetalBuffer& buffer) {
        if (residency == MetalCommandError::None && buffer) {
            residency = use_buffer_resource(
                pass, buffer, MetalResourceUsage::ReadWrite);
        }
    };
    declare(*harness.step->image);
    declare(prefill.tokens);
    declare(prefill.hidden_slab);
    declare(prefill.block_hidden);
    declare(prefill.normalized);
    declare(prefill.branch);
    declare(prefill.moe_output);
    declare(prefill.gdn_projection);
    declare(prefill.gdn_qk);
    declare(prefill.gdn_value);
    declare(prefill.gdn_gate);
    declare(prefill.gdn_recurrence);
    declare(prefill.gdn_gated);
    declare(prefill.gdn_decay);
    declare(prefill.gdn_beta);
    declare(prefill.attention_projection);
    declare(prefill.attention_query);
    declare(prefill.attention_gate);
    declare(prefill.attention_attended);
    declare(prefill.attention_partials);
    declare(prefill.router_logits);
    declare(prefill.expert_ids);
    declare(prefill.expert_coefficients);
    declare(prefill.shared_coefficients);
    declare(prefill.expert_counts);
    declare(prefill.expert_lists);
    declare(prefill.active_experts);
    declare(prefill.expert_arguments);
    declare(prefill.expert_hidden);
    declare(prefill.expert_partials);
    declare(prefill.native_routed_up_tasks);
    declare(prefill.native_routed_up_arguments);
    declare(prefill.native_routed_up_status);
    declare(prefill.native_routed_down_tasks);
    declare(prefill.native_routed_down_arguments);
    declare(prefill.native_routed_down_status);
    declare(prefill.shared_expert);
    declare(prefill.shared_expert_arguments);
    for (const DecodeLayerState& layer :
         harness.step->state.layers) {
        declare(layer.first);
        declare(layer.first_out);
        declare(layer.second);
        declare(layer.second_out);
    }
    for (const MetalBuffer& window :
         prefill.command_graph.image_windows) {
        declare(window);
    }
    declare(prefill.command_graph.argument_arena);
    return residency;
}

// Re-executes the cached ICB with the product per-level range topology and
// reports the wall of each replay. Called only after the state record and
// continuation are complete: the replays read stale scratch and publish
// nothing, so they are timing evidence only.
int run_graph_replay_measurements(
    tatara::tools::DecodeHarness& harness,
    PrefillStep& prefill, std::uint32_t replays) {
    const std::vector<std::uint32_t>& levels =
        prefill.command_graph.level_command_begins;
    if (levels.size() < 2u) {
        return kExitCommandGraph;
    }
    for (std::uint32_t replay = 0; replay < replays; ++replay) {
        auto command_buffer =
            create_command_buffer(*harness.queue);
        if (!command_buffer) {
            return kExitCommandBuffer;
        }
        auto pass = begin_compute_pass(
            std::move(*command_buffer.command_buffer));
        if (!pass) {
            return kExitComputePass;
        }
        if (declare_graph_residency(
                *pass.compute_pass, harness, prefill) !=
            MetalCommandError::None) {
            return kExitEncode;
        }
        MetalCommandError encoded = MetalCommandError::None;
        for (std::size_t level = 0;
             level + 1u < levels.size() &&
             encoded == MetalCommandError::None;
             ++level) {
            if (level != 0u) {
                encoded = memory_barrier(*pass.compute_pass);
            }
            if (encoded == MetalCommandError::None) {
                encoded = execute_indirect_commands(
                    *pass.compute_pass,
                    prefill.command_graph.commands,
                    levels[level],
                    levels[level + 1u] - levels[level]);
            }
        }
        if (encoded != MetalCommandError::None) {
            return kExitEncode;
        }
        auto ended =
            end_compute_pass(std::move(*pass.compute_pass));
        if (!ended) {
            return kExitEndPass;
        }
        const auto start = std::chrono::steady_clock::now();
        auto pending = commit(std::move(*ended.command_buffer));
        if (!pending) {
            return kExitCommit;
        }
        const auto execution = wait_until_completed_timed(
            std::move(*pending.pending_execution));
        const double wall =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start)
                .count();
        if (!execution) {
            std::cerr << "graph replay " << replay
                      << " failed: "
                      << execution.failure_description.view()
                      << '\n';
            return kExitExecution;
        }
        const MetalExecutionTiming timing = execution.timing;
        double gpu = 0.0;
        double schedule = 0.0;
        if (std::isfinite(timing.gpu_start_seconds) &&
            std::isfinite(timing.gpu_end_seconds) &&
            timing.gpu_end_seconds > timing.gpu_start_seconds) {
            gpu = timing.gpu_end_seconds -
                  timing.gpu_start_seconds;
        }
        if (std::isfinite(timing.schedule_start_seconds) &&
            std::isfinite(timing.schedule_end_seconds) &&
            timing.schedule_end_seconds >
                timing.schedule_start_seconds) {
            schedule = timing.schedule_end_seconds -
                       timing.schedule_start_seconds;
        }
        std::cerr << "  graph replay " << replay
                  << ": wall=" << wall << " s gpu=" << gpu
                  << " s schedule=" << schedule << " s\n";
    }
    return 0;
}

// Replays the prepared ICB one command per command buffer with a strided
// CPU NaN sample of every scratch and state buffer between commands, and
// reports the first command whose execution mints a NaN. Recorded-command
// content is thereby discriminated from concurrent execution with one
// bounded run. No state is published.
SubmissionResult run_graph_serial_diagnostic(
    tatara::tools::DecodeHarness& harness,
    PrefillStep& prefill) {
    const std::vector<std::uint32_t>& levels =
        prefill.command_graph.level_command_begins;
    std::cerr << "graph level boundaries (" << levels.size()
              << "):";
    for (const std::uint32_t begin : levels) {
        std::cerr << ' ' << begin;
    }
    std::cerr << '\n';
    std::cerr << "image bytes: "
              << harness.step->image->size_bytes() << '\n';
    for (std::size_t layer = 0;
         layer < std::min<std::size_t>(
                     4, harness.step->bindings.layers.size());
         ++layer) {
        const LayerBindings& bindings =
            harness.step->bindings.layers[layer];
        if (bindings.kind != LayerKind::GatedDelta) {
            continue;
        }
        const auto offset = [&](std::uint32_t tensor) {
            return harness.step->tensor_offsets[tensor];
        };
        std::cerr << "layer " << layer << " qkv offsets:"
                  << " words="
                  << offset(bindings.gated_delta.qkv.weight)
                  << " scales="
                  << offset(bindings.gated_delta.qkv.scales)
                  << " biases="
                  << offset(bindings.gated_delta.qkv.biases)
                  << '\n';
    }

    const std::array<SampledBuffer, 16> sampled_buffers{{
        {"hidden_slab", &prefill.hidden_slab, true},
        {"block_hidden", &prefill.block_hidden, true},
        {"normalized", &prefill.normalized, true},
        {"branch", &prefill.branch, true},
        {"moe_output", &prefill.moe_output, true},
        {"gdn_projection", &prefill.gdn_projection, true},
        {"gdn_qk", &prefill.gdn_qk, true},
        {"gdn_value", &prefill.gdn_value, true},
        {"gdn_gate", &prefill.gdn_gate, true},
        {"gdn_recurrence", &prefill.gdn_recurrence, true},
        {"gdn_gated", &prefill.gdn_gated, true},
        {"gdn_decay", &prefill.gdn_decay, true},
        {"gdn_beta", &prefill.gdn_beta, true},
        {"router_logits", &prefill.router_logits, false},
        {"expert_hidden", &prefill.expert_hidden, true},
        {"expert_partials", &prefill.expert_partials, false},
    }};

    for (std::uint32_t command = 0;
         command < prefill.command_graph.command_count;
         ++command) {
        auto command_buffer =
            create_command_buffer(*harness.queue);
        if (!command_buffer) {
            poison_prefill(prefill);
            return {.exit_code = kExitCommandBuffer};
        }
        auto pass = begin_compute_pass(
            std::move(*command_buffer.command_buffer));
        if (!pass) {
            poison_prefill(prefill);
            return {.exit_code = kExitComputePass};
        }
        if (declare_graph_residency(
                *pass.compute_pass, harness, prefill) !=
            MetalCommandError::None) {
            poison_prefill(prefill);
            return {.exit_code = kExitEncode};
        }
        const MetalCommandError executed =
            execute_indirect_commands(
                *pass.compute_pass,
                prefill.command_graph.commands, command, 1);
        if (executed != MetalCommandError::None) {
            poison_prefill(prefill);
            return {.exit_code = kExitEncode};
        }
        auto ended =
            end_compute_pass(std::move(*pass.compute_pass));
        if (!ended) {
            poison_prefill(prefill);
            return {.exit_code = kExitEndPass};
        }
        auto pending = commit(std::move(*ended.command_buffer));
        if (!pending) {
            poison_prefill(prefill);
            return {.exit_code = kExitCommit};
        }
        const auto execution = wait_until_completed_timed(
            std::move(*pending.pending_execution));
        if (!execution) {
            std::cerr
                << "graph serial diagnostic failed at command "
                << command << ": "
                << execution.failure_description.view() << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitExecution};
        }
        for (const SampledBuffer& sampled : sampled_buffers) {
            if (sampled_nan(sampled)) {
                std::cerr << "first sampled nan after command "
                          << command << " in " << sampled.name
                          << '\n';
                print_graph_unit_statuses(prefill);
                print_graph_nan_census(
                    prefill, harness.step->state);
                poison_prefill(prefill);
                return {
                    .exit_code = kExitGraphSerialDiagnostic};
            }
        }
    }
    std::cerr << "graph serial diagnostic: "
              << prefill.command_graph.command_count
              << " per-command submissions completed without a"
                 " sampled nan\n";
    print_graph_unit_statuses(prefill);
    print_graph_nan_census(prefill, harness.step->state);
    poison_prefill(prefill);
    return {.exit_code = kExitGraphSerialDiagnostic};
}

int run_graph_level_profile(
    tatara::tools::DecodeHarness& harness, PrefillStep& prefill);

SubmissionResult submit_banded_graph(
    tatara::tools::DecodeHarness& harness, PrefillStep& prefill,
    std::span<const std::uint32_t> ids) {
    if (!prefill.policy.command_graph || ids.empty() ||
        ids.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        return {.exit_code = kExitCommandGraph};
    }
    SubmissionResult result{
        .exit_code = 0,
        .seconds = 0.0,
        .gpu_seconds = 0.0,
        .schedule_seconds = 0.0,
        .command_buffers = 0,
        .timed_command_buffers = 0,
        .chunks = 0,
    };
    std::uint32_t context = 0;
    std::uint32_t offset = 0;
    std::uint32_t remaining =
        static_cast<std::uint32_t>(ids.size());
    while (remaining != 0) {
        const PrefillBandPlan band = plan_next_prefill_band(
            prefill.policy.geometry, context, remaining,
            prefill.policy.command_graph_chunk_count);
        if (!band) {
            std::cerr << "band planning failed: error="
                      << static_cast<unsigned>(band.error)
                      << " context=" << context
                      << " remaining=" << remaining << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitCommandGraph,
                    .chunks = result.chunks};
        }
        const std::span<const std::uint32_t> band_ids(
            ids.data() + offset, band.row_count);
        const PrefillProgressResult begun =
            begin_prefill_progress(
                prefill, *harness.step, band.context_base,
                band.context_base, band_ids);
        if (!begun) {
            std::cerr << "band begin failed: progress_error="
                      << static_cast<unsigned>(begun.error)
                      << " encode_error="
                      << static_cast<unsigned>(begun.encode_error)
                      << " context=" << band.context_base
                      << " rows=" << band.row_count << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitEncode,
                    .chunks = result.chunks};
        }
        result.chunks += begun.chunk_count;
        const PrefillCommandGraphResult prepared =
            prepare_prefill_command_graph(
                *harness.device, prefill, *harness.step);
        if (!prepared) {
            std::cerr << "band graph preparation failed:"
                      << " graph_error="
                      << static_cast<unsigned>(prepared.error)
                      << " stage="
                      << static_cast<unsigned>(prepared.stage)
                      << " context=" << band.context_base
                      << " rows=" << band.row_count << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitCommandGraph,
                    .chunks = result.chunks};
        }
        auto command_buffer =
            create_command_buffer(*harness.queue);
        if (!command_buffer) {
            poison_prefill(prefill);
            return {.exit_code = kExitCommandBuffer,
                    .chunks = result.chunks};
        }
        auto pass = begin_compute_pass(
            std::move(*command_buffer.command_buffer));
        if (!pass) {
            poison_prefill(prefill);
            return {.exit_code = kExitComputePass,
                    .chunks = result.chunks};
        }
        const PrefillCommandGraphResult encoded =
            encode_prefill_command_graph(
                prefill, *harness.step, *pass.compute_pass);
        if (!encoded) {
            std::cerr << "band graph encoding failed:"
                      << " graph_error="
                      << static_cast<unsigned>(encoded.error)
                      << " stage="
                      << static_cast<unsigned>(encoded.stage)
                      << " context=" << band.context_base
                      << " rows=" << band.row_count << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitEncode,
                    .chunks = result.chunks};
        }
        auto ended =
            end_compute_pass(std::move(*pass.compute_pass));
        if (!ended) {
            poison_prefill(prefill);
            return {.exit_code = kExitEndPass,
                    .chunks = result.chunks};
        }
        const auto start = std::chrono::steady_clock::now();
        auto pending =
            commit(std::move(*ended.command_buffer));
        if (!pending) {
            poison_prefill(prefill);
            return {.exit_code = kExitCommit,
                    .chunks = result.chunks};
        }
        const auto execution = wait_until_completed_timed(
            std::move(*pending.pending_execution));
        result.seconds +=
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start)
                .count();
        ++result.command_buffers;
        if (!execution) {
            std::cerr << "band graph execution failed: "
                      << execution.failure_description.view()
                      << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitExecution,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        const MetalExecutionTiming timing = execution.timing;
        const bool gpu_timing_present =
            std::isfinite(timing.gpu_start_seconds) &&
            std::isfinite(timing.gpu_end_seconds) &&
            timing.gpu_end_seconds >
                timing.gpu_start_seconds;
        const bool schedule_timing_present =
            std::isfinite(timing.schedule_start_seconds) &&
            std::isfinite(timing.schedule_end_seconds) &&
            timing.schedule_end_seconds >
                timing.schedule_start_seconds;
        if (gpu_timing_present && schedule_timing_present) {
            result.gpu_seconds +=
                timing.gpu_end_seconds -
                timing.gpu_start_seconds;
            result.schedule_seconds +=
                timing.schedule_end_seconds -
                timing.schedule_start_seconds;
            ++result.timed_command_buffers;
        }
        const PrefillProgressResult committed =
            commit_prefill_command_graph(
                prefill, *harness.step);
        if (!committed ||
            committed.next_context != band.next_context) {
            std::cerr << "band graph commit failed:"
                      << " progress_error="
                      << static_cast<unsigned>(committed.error)
                      << " next_context="
                      << committed.next_context
                      << " expected=" << band.next_context << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitExecution,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        context = band.next_context;
        offset += band.row_count;
        remaining -= band.row_count;
    }
    if (context != static_cast<std::uint32_t>(ids.size())) {
        poison_prefill(prefill);
        return {.exit_code = kExitExecution,
                .seconds = result.seconds,
                .command_buffers = result.command_buffers,
                .chunks = result.chunks};
    }
    return result;
}

SubmissionResult run_lane_event_graph(
    tatara::tools::DecodeHarness& harness, PrefillStep& prefill) {
    SubmissionResult result{
        .exit_code = 0,
        .seconds = 0.0,
        .gpu_seconds = 0.0,
        .schedule_seconds = 0.0,
        .command_buffers = 0,
        .timed_command_buffers = 0,
        .chunks = prefill.progress.chunk_count,
    };
    const std::uint32_t lane_count =
        prefill.progress.chunk_count;
    const std::uint32_t layer_count = static_cast<std::uint32_t>(
        harness.step->state.layers.size());
    if (!prefill.policy.command_graph_lane_events ||
        lane_count == 0 ||
        lane_count >
            prefill.command_graph_lane_queues.size() ||
        lane_count >
            prefill.policy.command_graph_chunk_count ||
        prefill.command_graph_lane_events.size() + 1u <
            lane_count) {
        return {.exit_code = kExitCommandGraph,
                .chunks = result.chunks};
    }

    std::vector<PrefillLaneEventNode> event_nodes(
        prefill.command_graph.nodes.size());
    const PrefillLaneEventPlanResult event_plan =
        build_prefill_lane_event_plan(
            {
                .layer_count = layer_count,
                .scratch_lane_count = lane_count,
                .event_value_base =
                    prefill.command_graph_event_value_base,
                .nodes = prefill.command_graph.nodes,
            },
            event_nodes);
    if (!event_plan ||
        event_plan.plan.node_count != event_nodes.size() ||
        event_plan.plan.event_count + 1u != lane_count ||
        event_nodes.size() !=
            prefill.command_graph.lane_event_nodes.size()) {
        std::cerr << "lane event plan failed: plan_error="
                  << static_cast<unsigned>(event_plan.error)
                  << '\n';
        poison_prefill(prefill);
        return {.exit_code = kExitCommandGraph,
                .chunks = result.chunks};
    }
    for (std::size_t index = 0; index < event_nodes.size();
         ++index) {
        const PrefillLaneEventNode& dynamic = event_nodes[index];
        const PrefillLaneEventNode& recorded =
            prefill.command_graph.lane_event_nodes[index];
        if (dynamic.node_index != recorded.node_index ||
            dynamic.layer_index != recorded.layer_index ||
            dynamic.scratch_lane != recorded.scratch_lane ||
            dynamic.wait_event != recorded.wait_event ||
            dynamic.signal_event != recorded.signal_event) {
            std::cerr << "lane event topology mismatch at node "
                      << index << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitCommandGraph,
                    .chunks = result.chunks};
        }
    }

    std::vector<MetalCommandBuffer> command_buffers;
    command_buffers.reserve(lane_count);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        auto created = create_command_buffer(
            prefill.command_graph_lane_queues[lane]);
        if (!created) {
            poison_prefill(prefill);
            return {.exit_code = kExitCommandBuffer,
                    .chunks = result.chunks};
        }
        MetalCommandBuffer command_buffer =
            std::move(*created.command_buffer);
        for (std::uint32_t layer = 0;
             layer < layer_count; ++layer) {
            const PrefillLaneEventNode& node =
                event_nodes[
                    std::uint64_t{layer} * lane_count + lane];
            if (node.wait_event != kNoPrefillLaneEvent &&
                encode_wait_for_event(
                    command_buffer,
                    prefill.command_graph_lane_events[
                        node.wait_event],
                    node.wait_value) != MetalCommandError::None) {
                poison_prefill(prefill);
                return {.exit_code = kExitEncode,
                        .chunks = result.chunks};
            }
            auto pass =
                begin_compute_pass(std::move(command_buffer));
            if (!pass) {
                poison_prefill(prefill);
                return {.exit_code = kExitComputePass,
                        .chunks = result.chunks};
            }
            const PrefillCommandGraphResult encoded =
                encode_prefill_command_graph_lane_node(
                    prefill, *harness.step, *pass.compute_pass,
                    layer, lane);
            if (!encoded) {
                std::cerr
                    << "lane graph node encode failed: layer="
                    << layer << " lane=" << lane
                    << " graph_error="
                    << static_cast<unsigned>(encoded.error)
                    << " stage="
                    << static_cast<unsigned>(encoded.stage)
                    << " command_error="
                    << static_cast<unsigned>(
                           encoded.command_error)
                    << '\n';
                poison_prefill(prefill);
                return {.exit_code = kExitEncode,
                        .chunks = result.chunks};
            }
            auto ended =
                end_compute_pass(std::move(*pass.compute_pass));
            if (!ended) {
                poison_prefill(prefill);
                return {.exit_code = kExitEndPass,
                        .chunks = result.chunks};
            }
            command_buffer =
                std::move(*ended.command_buffer);
            if (node.signal_event != kNoPrefillLaneEvent &&
                encode_signal_event(
                    command_buffer,
                    prefill.command_graph_lane_events[
                        node.signal_event],
                    node.signal_value) != MetalCommandError::None) {
                poison_prefill(prefill);
                return {.exit_code = kExitEncode,
                        .chunks = result.chunks};
            }
        }
        command_buffers.push_back(std::move(command_buffer));
    }
    const PrefillCommandGraphResult pending_state =
        mark_prefill_command_graph_lane_pending(
            prefill, *harness.step);
    if (!pending_state) {
        poison_prefill(prefill);
        return {.exit_code = kExitCommandGraph,
                .chunks = result.chunks};
    }

    std::vector<MetalPendingExecution> pending;
    pending.reserve(command_buffers.size());
    const auto start = std::chrono::steady_clock::now();
    bool commit_failed = false;
    for (MetalCommandBuffer& command_buffer : command_buffers) {
        auto committed = commit(std::move(command_buffer));
        if (!committed) {
            commit_failed = true;
            break;
        }
        pending.push_back(
            std::move(*committed.pending_execution));
    }
    double gpu_start =
        std::numeric_limits<double>::infinity();
    double gpu_end = 0.0;
    double schedule_start =
        std::numeric_limits<double>::infinity();
    double schedule_end = 0.0;
    bool execution_failed = commit_failed;
    for (MetalPendingExecution& execution : pending) {
        const MetalTimedExecutionResult completed =
            wait_until_completed_timed(std::move(execution));
        if (!completed) {
            execution_failed = true;
            std::cerr << "lane graph execution failed: "
                      << completed.failure_description.view()
                      << '\n';
            continue;
        }
        const MetalExecutionTiming timing = completed.timing;
        if (std::isfinite(timing.gpu_start_seconds) &&
            std::isfinite(timing.gpu_end_seconds) &&
            timing.gpu_end_seconds >
                timing.gpu_start_seconds) {
            gpu_start =
                std::min(gpu_start, timing.gpu_start_seconds);
            gpu_end = std::max(gpu_end, timing.gpu_end_seconds);
        }
        if (std::isfinite(timing.schedule_start_seconds) &&
            std::isfinite(timing.schedule_end_seconds) &&
            timing.schedule_end_seconds >
                timing.schedule_start_seconds) {
            schedule_start = std::min(
                schedule_start,
                timing.schedule_start_seconds);
            schedule_end = std::max(
                schedule_end, timing.schedule_end_seconds);
        }
    }
    result.seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start)
            .count();
    result.command_buffers =
        static_cast<std::uint32_t>(pending.size());
    if (execution_failed ||
        pending.size() != lane_count) {
        poison_prefill(prefill);
        return {.exit_code = commit_failed ? kExitCommit
                                           : kExitExecution,
                .seconds = result.seconds,
                .command_buffers = result.command_buffers,
                .chunks = result.chunks};
    }
    if (std::isfinite(gpu_start) && gpu_end > gpu_start &&
        std::isfinite(schedule_start) &&
        schedule_end > schedule_start) {
        result.gpu_seconds = gpu_end - gpu_start;
        result.schedule_seconds =
            schedule_end - schedule_start;
        result.timed_command_buffers = lane_count;
    }
    const PrefillProgressResult graph_commit =
        commit_prefill_command_graph(prefill, *harness.step);
    if (!graph_commit) {
        std::cerr << "lane graph state commit failed:"
                  << " progress_error="
                  << static_cast<unsigned>(graph_commit.error)
                  << " failed_unit_offset="
                  << graph_commit.failed_unit_offset << '\n';
        print_graph_unit_statuses(prefill);
        poison_prefill(prefill);
        return {.exit_code = kExitExecution,
                .seconds = result.seconds,
                .command_buffers = result.command_buffers,
                .chunks = result.chunks};
    }
    prefill.command_graph_event_value_base =
        event_plan.plan.terminal_event_value;
    return result;
}

template <bool CaptureExpertCounts>
SubmissionResult submit_prefill_impl(
    tatara::tools::DecodeHarness& harness, PrefillStep& prefill,
    std::span<const std::uint32_t> ids,
    ExpertCountCapture* expert_count_capture) {
    if constexpr (CaptureExpertCounts) {
        if (expert_count_capture == nullptr ||
            prefill.policy.maximum_units_per_submission != 1 ||
            prefill.policy.maximum_inflight_units != 1) {
            return {.exit_code = kExitExpertCountCaptureContract};
        }
    }
    if (graph_banded_requested) {
        if (CaptureExpertCounts) {
            return {
                .exit_code = kExitExpertCountCaptureContract};
        }
        return submit_banded_graph(harness, prefill, ids);
    }
    const PrefillProgressResult begun = begin_prefill_progress(prefill, *harness.step, 0, 0, ids);
    if (!begun) {
        std::cerr << "begin_prefill_progress failed: progress_error="
                  << static_cast<unsigned>(begun.error)
                  << " encode_error=" << static_cast<unsigned>(begun.encode_error) << '\n';
        return {.exit_code = kExitEncode};
    }

    SubmissionResult result{
        .exit_code = 0,
        .seconds = 0.0,
        .gpu_seconds = 0.0,
        .schedule_seconds = 0.0,
        .command_buffers = 0,
        .timed_command_buffers = 0,
        .chunks = begun.chunk_count,
    };
    if (prefill.policy.command_graph) {
        const std::uint32_t graph_requests =
            graph_warm_requested ? kGraphWarmRequests : 1u;
        for (std::uint32_t request = 0;
             request < graph_requests; ++request) {
            if (request != 0) {
                reset_state_slot_for_warm_request(
                    harness.step->state);
                const PrefillProgressResult again =
                    begin_prefill_progress(
                        prefill, *harness.step, 0, 0, ids);
                if (!again) {
                    std::cerr
                        << "graph warm re-begin failed:"
                        << " progress_error="
                        << static_cast<unsigned>(again.error)
                        << '\n';
                    poison_prefill(prefill);
                    return {.exit_code = kExitEncode,
                            .chunks = result.chunks};
                }
            }
            const PrefillCommandGraphResult prepared =
                prepare_prefill_command_graph(
                    *harness.device, prefill, *harness.step);
            if (!prepared) {
                std::cerr
                    << "prepare_prefill_command_graph failed:"
                    << " graph_error="
                    << static_cast<unsigned>(prepared.error)
                    << " stage="
                    << static_cast<unsigned>(prepared.stage)
                    << " plan_error="
                    << static_cast<unsigned>(prepared.plan_error)
                    << " command_error="
                    << static_cast<unsigned>(
                           prepared.command_error)
                    << '\n';
                poison_prefill(prefill);
                return {.exit_code = kExitCommandGraph,
                        .chunks = result.chunks};
            }
            if (graph_warm_requested &&
                prepared.cache_hit != (request != 0)) {
                std::cerr << "graph warm cache expectation"
                             " failed: request "
                          << request << " cache_hit="
                          << prepared.cache_hit << '\n';
                poison_prefill(prefill);
                return {.exit_code = kExitCommandGraph,
                        .chunks = result.chunks};
            }
            if (graph_serial_diagnostic_requested) {
                SubmissionResult diagnostic =
                    run_graph_serial_diagnostic(
                        harness, prefill);
                diagnostic.chunks = result.chunks;
                return diagnostic;
            }
            if (prefill.policy.command_graph_lane_events) {
                result = run_lane_event_graph(harness, prefill);
                if (result.exit_code != 0) {
                    return result;
                }
                if (graph_warm_requested) {
                    std::cerr << "  graph request " << request
                              << ": wall=" << result.seconds
                              << " s gpu=" << result.gpu_seconds
                              << " s schedule="
                              << result.schedule_seconds
                              << " s cache_hit="
                              << prepared.cache_hit << '\n';
                }
                continue;
            }
            auto command_buffer =
                create_command_buffer(*harness.queue);
            if (!command_buffer) {
                poison_prefill(prefill);
                return {.exit_code = kExitCommandBuffer,
                        .chunks = result.chunks};
            }
            auto pass = begin_compute_pass(
                std::move(*command_buffer.command_buffer));
            if (!pass) {
                poison_prefill(prefill);
                return {.exit_code = kExitComputePass,
                        .chunks = result.chunks};
            }
            const PrefillCommandGraphResult encoded =
                encode_prefill_command_graph(
                    prefill, *harness.step, *pass.compute_pass);
            if (!encoded) {
                std::cerr
                    << "encode_prefill_command_graph failed:"
                    << " graph_error="
                    << static_cast<unsigned>(encoded.error)
                    << " stage="
                    << static_cast<unsigned>(encoded.stage)
                    << " command_error="
                    << static_cast<unsigned>(
                           encoded.command_error)
                    << '\n';
                poison_prefill(prefill);
                return {.exit_code = kExitEncode,
                        .chunks = result.chunks};
            }
            auto ended =
                end_compute_pass(std::move(*pass.compute_pass));
            if (!ended) {
                poison_prefill(prefill);
                return {.exit_code = kExitEndPass,
                        .chunks = result.chunks};
            }
            const auto start = std::chrono::steady_clock::now();
            auto pending =
                commit(std::move(*ended.command_buffer));
            if (!pending) {
                poison_prefill(prefill);
                return {.exit_code = kExitCommit,
                        .chunks = result.chunks};
            }
            const auto execution = wait_until_completed_timed(
                std::move(*pending.pending_execution));
            result.seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - start)
                    .count();
            result.command_buffers = 1;
            if (!execution) {
                std::cerr
                    << "command graph execution failed: "
                    << execution.failure_description.view()
                    << '\n';
                poison_prefill(prefill);
                return {.exit_code = kExitExecution,
                        .seconds = result.seconds,
                        .command_buffers =
                            result.command_buffers,
                        .chunks = result.chunks};
            }
            const MetalExecutionTiming timing =
                execution.timing;
            const bool gpu_timing_present =
                std::isfinite(timing.gpu_start_seconds) &&
                std::isfinite(timing.gpu_end_seconds) &&
                timing.gpu_end_seconds >
                    timing.gpu_start_seconds;
            const bool schedule_timing_present =
                std::isfinite(timing.schedule_start_seconds) &&
                std::isfinite(timing.schedule_end_seconds) &&
                timing.schedule_end_seconds >
                    timing.schedule_start_seconds;
            if (gpu_timing_present && schedule_timing_present) {
                result.gpu_seconds =
                    timing.gpu_end_seconds -
                    timing.gpu_start_seconds;
                result.schedule_seconds =
                    timing.schedule_end_seconds -
                    timing.schedule_start_seconds;
                result.timed_command_buffers = 1;
            }
            const PrefillProgressResult committed =
                commit_prefill_command_graph(
                    prefill, *harness.step);
            if (!committed) {
                std::cerr
                    << "commit_prefill_command_graph failed:"
                    << " progress_error="
                    << static_cast<unsigned>(committed.error)
                    << " failed_unit_offset="
                    << committed.failed_unit_offset
                    << " routed_up_status="
                    << static_cast<std::uint32_t>(
                           committed.routed_up_status)
                    << '\n';
                print_graph_unit_statuses(prefill);
                print_graph_failed_unit_ids(
                    prefill, committed.failed_unit_offset);
                print_graph_nan_census(
                    prefill, harness.step->state);
                poison_prefill(prefill);
                return {.exit_code = kExitExecution,
                        .seconds = result.seconds,
                        .command_buffers =
                            result.command_buffers,
                        .chunks = result.chunks};
            }
            if (graph_warm_requested) {
                std::cerr << "  graph request " << request
                          << ": wall=" << result.seconds
                          << " s gpu=" << result.gpu_seconds
                          << " s schedule="
                          << result.schedule_seconds
                          << " s cache_hit="
                          << prepared.cache_hit << '\n';
            }
        }
        if (graph_level_profile_requested) {
            // Timing diagnostic only: per-level re-execution clobbers
            // scratch and published state, so this mode exits typed
            // without the state record or continuation.
            const int profile_exit =
                run_graph_level_profile(harness, prefill);
            poison_prefill(prefill);
            return {.exit_code = profile_exit == 0
                                     ? kExitGraphLevelProfile
                                     : profile_exit,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        if (graph_repeat_requested) {
            // Timing diagnostic only: the replays clobber scratch and
            // published state, so this mode never continues into the
            // state record or continuation and exits typed instead.
            std::cerr << "graph first execution: wall="
                      << result.seconds << " s gpu="
                      << result.gpu_seconds << " s schedule="
                      << result.schedule_seconds << " s\n";
            std::cerr << "graph replay measurements:\n";
            const int replay_exit =
                run_graph_replay_measurements(
                    harness, prefill, 4u);
            poison_prefill(prefill);
            return {.exit_code = replay_exit == 0
                                     ? kExitGraphReplayDiagnostic
                                     : replay_exit,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        return result;
    }
    if (prefill.policy.maximum_inflight_units > 1) {
        std::array<MetalPendingExecution,
                   kPrefillMaximumUnitsPerSubmission>
            pending_executions;
        const auto submission_start =
            std::chrono::steady_clock::now();
        while (prefill.progress.state !=
               PrefillProgressState::Complete) {
            std::uint32_t pending_count = 0;
            while (
                prefill.progress.state ==
                    PrefillProgressState::Ready ||
                prefill.progress.state ==
                    PrefillProgressState::InflightEncoding) {
                auto command_buffer =
                    create_command_buffer(*harness.queue);
                if (!command_buffer) {
                    poison_prefill(prefill);
                    for (std::uint32_t index = 0;
                         index < pending_count; ++index) {
                        static_cast<void>(
                            wait_until_completed(std::move(
                                pending_executions[index])));
                    }
                    return {
                        .exit_code = kExitCommandBuffer,
                        .seconds =
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                submission_start)
                                .count(),
                        .command_buffers =
                            result.command_buffers,
                        .chunks = result.chunks,
                    };
                }
                auto pass = begin_compute_pass(
                    std::move(
                        *command_buffer.command_buffer));
                if (!pass) {
                    poison_prefill(prefill);
                    for (std::uint32_t index = 0;
                         index < pending_count; ++index) {
                        static_cast<void>(
                            wait_until_completed(std::move(
                                pending_executions[index])));
                    }
                    return {
                        .exit_code = kExitComputePass,
                        .seconds =
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                submission_start)
                                .count(),
                        .command_buffers =
                            result.command_buffers,
                        .chunks = result.chunks,
                    };
                }
                const PrefillProgressResult encoded =
                    encode_prefill_inflight_unit(
                        prefill, *harness.step,
                        *pass.compute_pass);
                if (!encoded) {
                    std::cerr
                        << "encode_prefill_inflight_unit failed:"
                        << " progress_error="
                        << static_cast<unsigned>(encoded.error)
                        << " encode_error="
                        << static_cast<unsigned>(
                               encoded.encode_error)
                        << " command_error="
                        << static_cast<unsigned>(
                               encoded.command_error)
                        << '\n';
                    poison_prefill(prefill);
                    for (std::uint32_t index = 0;
                         index < pending_count; ++index) {
                        static_cast<void>(
                            wait_until_completed(std::move(
                                pending_executions[index])));
                    }
                    return {
                        .exit_code = kExitEncode,
                        .seconds =
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                submission_start)
                                .count(),
                        .command_buffers =
                            result.command_buffers,
                        .chunks = result.chunks,
                    };
                }
                auto ended = end_compute_pass(
                    std::move(*pass.compute_pass));
                if (!ended) {
                    poison_prefill(prefill);
                    for (std::uint32_t index = 0;
                         index < pending_count; ++index) {
                        static_cast<void>(
                            wait_until_completed(std::move(
                                pending_executions[index])));
                    }
                    return {
                        .exit_code = kExitEndPass,
                        .seconds =
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                submission_start)
                                .count(),
                        .command_buffers =
                            result.command_buffers,
                        .chunks = result.chunks,
                    };
                }
                auto pending = commit(
                    std::move(*ended.command_buffer));
                if (!pending) {
                    poison_prefill(prefill);
                    for (std::uint32_t index = 0;
                         index < pending_count; ++index) {
                        static_cast<void>(
                            wait_until_completed(std::move(
                                pending_executions[index])));
                    }
                    return {
                        .exit_code = kExitCommit,
                        .seconds =
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                submission_start)
                                .count(),
                        .command_buffers =
                            result.command_buffers,
                        .chunks = result.chunks,
                    };
                }
                pending_executions[pending_count] =
                    std::move(*pending.pending_execution);
                ++pending_count;
                ++result.command_buffers;
            }
            if (prefill.progress.state !=
                    PrefillProgressState::InflightPending ||
                pending_count == 0 ||
                pending_count !=
                    prefill.progress.pending_unit_count) {
                poison_prefill(prefill);
                for (std::uint32_t index = 0;
                     index < pending_count; ++index) {
                    static_cast<void>(
                        wait_until_completed(std::move(
                            pending_executions[index])));
                }
                return {
                    .exit_code = kExitEncode,
                    .seconds =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            submission_start)
                            .count(),
                    .command_buffers =
                        result.command_buffers,
                    .chunks = result.chunks,
                };
            }

            bool executions_complete = true;
            for (std::uint32_t index = 0;
                 index < pending_count; ++index) {
                const auto execution =
                    wait_until_completed_timed(std::move(
                        pending_executions[index]));
                if (!execution) {
                    executions_complete = false;
                    std::cerr
                        << "in-flight prefill execution failed: "
                        << execution.failure_description.view()
                        << '\n';
                    continue;
                }
                const MetalExecutionTiming timing =
                    execution.timing;
                const bool gpu_timing_present =
                    std::isfinite(
                        timing.gpu_start_seconds) &&
                    std::isfinite(
                        timing.gpu_end_seconds) &&
                    timing.gpu_end_seconds >
                        timing.gpu_start_seconds;
                const bool schedule_timing_present =
                    std::isfinite(
                        timing.schedule_start_seconds) &&
                    std::isfinite(
                        timing.schedule_end_seconds) &&
                    timing.schedule_end_seconds >
                        timing.schedule_start_seconds;
                if (gpu_timing_present &&
                    schedule_timing_present) {
                    result.gpu_seconds +=
                        timing.gpu_end_seconds -
                        timing.gpu_start_seconds;
                    result.schedule_seconds +=
                        timing.schedule_end_seconds -
                        timing.schedule_start_seconds;
                    ++result.timed_command_buffers;
                }
            }
            if (!executions_complete) {
                poison_prefill(prefill);
                return {
                    .exit_code = kExitExecution,
                    .seconds =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            submission_start)
                            .count(),
                    .command_buffers =
                        result.command_buffers,
                    .chunks = result.chunks,
                };
            }
            const PrefillProgressResult committed =
                commit_prefill_inflight(
                    prefill, *harness.step);
            if (!committed) {
                std::cerr
                    << "commit_prefill_inflight failed:"
                    << " progress_error="
                    << static_cast<unsigned>(
                           committed.error)
                    << " failed_unit_offset="
                    << committed.failed_unit_offset
                    << " routed_up_status="
                    << static_cast<std::uint32_t>(
                           committed.routed_up_status)
                    << " routed_down_status="
                    << static_cast<std::uint32_t>(
                           committed.routed_down_status)
                    << '\n';
                poison_prefill(prefill);
                return {
                    .exit_code = kExitExecution,
                    .seconds =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            submission_start)
                            .count(),
                    .command_buffers =
                        result.command_buffers,
                    .chunks = result.chunks,
                };
            }
        }
        result.seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() -
                submission_start)
                .count();
        return result;
    }
    while (prefill.progress.state != PrefillProgressState::Complete) {
        auto command_buffer = create_command_buffer(*harness.queue);
        if (!command_buffer) {
            poison_prefill(prefill);
            return {.exit_code = kExitCommandBuffer,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
        if (!pass) {
            poison_prefill(prefill);
            return {.exit_code = kExitComputePass,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        const bool batched =
            prefill.policy.maximum_units_per_submission > 1;
        const PrefillProgressResult encoded =
            batched
                ? encode_prefill_units(
                      prefill, *harness.step,
                      *pass.compute_pass)
                : encode_prefill_unit(
                      prefill, *harness.step,
                      *pass.compute_pass);
        if (!encoded) {
            std::cerr << "encode_prefill submission failed: progress_error="
                      << static_cast<unsigned>(encoded.error)
                      << " encode_error=" << static_cast<unsigned>(encoded.encode_error)
                      << " command_error=" << static_cast<unsigned>(encoded.command_error) << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitEncode,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        auto ended = end_compute_pass(std::move(*pass.compute_pass));
        if (!ended) {
            poison_prefill(prefill);
            return {.exit_code = kExitEndPass,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        const auto start = std::chrono::steady_clock::now();
        auto pending = commit(std::move(*ended.command_buffer));
        if (!pending) {
            poison_prefill(prefill);
            return {.exit_code = kExitCommit,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        const auto execution =
            wait_until_completed_timed(
                std::move(*pending.pending_execution));
        result.seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        ++result.command_buffers;
        if (!execution) {
            std::cerr << "block prefill unit execution failed: "
                      << execution.failure_description.view() << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitExecution,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        const MetalExecutionTiming timing = execution.timing;
        const bool gpu_timing_present =
            std::isfinite(timing.gpu_start_seconds) &&
            std::isfinite(timing.gpu_end_seconds) &&
            timing.gpu_end_seconds > timing.gpu_start_seconds;
        const bool schedule_timing_present =
            std::isfinite(timing.schedule_start_seconds) &&
            std::isfinite(timing.schedule_end_seconds) &&
            timing.schedule_end_seconds >
                timing.schedule_start_seconds;
        if (gpu_timing_present && schedule_timing_present) {
            result.gpu_seconds +=
                timing.gpu_end_seconds - timing.gpu_start_seconds;
            result.schedule_seconds +=
                timing.schedule_end_seconds -
                timing.schedule_start_seconds;
            ++result.timed_command_buffers;
        }
        if constexpr (CaptureExpertCounts) {
            if (!capture_expert_counts(prefill, encoded,
                                       *expert_count_capture)) {
                poison_prefill(prefill);
                return {.exit_code = kExitExpertCountCaptureContract,
                        .seconds = result.seconds,
                        .command_buffers = result.command_buffers,
                        .chunks = result.chunks};
            }
        }
        const PrefillProgressResult committed =
            batched
                ? commit_prefill_units(prefill, *harness.step)
                : commit_prefill_unit(prefill, *harness.step);
        if (!committed) {
            std::cerr << "commit_prefill submission failed: progress_error="
                      << static_cast<unsigned>(committed.error)
                      << " failed_unit_offset="
                      << committed.failed_unit_offset
                      << " routed_up_status="
                      << static_cast<std::uint32_t>(
                             committed.routed_up_status)
                      << " routed_down_status="
                      << static_cast<std::uint32_t>(
                             committed.routed_down_status)
                      << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitExecution,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
    }
    if constexpr (CaptureExpertCounts) {
        if (!expert_count_capture_complete(*expert_count_capture)) {
            return {.exit_code = kExitExpertCountCaptureContract,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
    }
    return result;
}

SubmissionResult submit_prefill(tatara::tools::DecodeHarness& harness,
                                PrefillStep& prefill,
                                std::span<const std::uint32_t> ids) {
    return submit_prefill_impl<false>(harness, prefill, ids, nullptr);
}

SubmissionResult submit_prefill(tatara::tools::DecodeHarness& harness,
                                PrefillStep& prefill,
                                std::span<const std::uint32_t> ids,
                                ExpertCountCapture& expert_count_capture) {
    return submit_prefill_impl<true>(harness, prefill, ids,
                                     &expert_count_capture);
}

SubmissionResult submit_profiled_prefill(
    tatara::tools::DecodeHarness& harness, PrefillStep& prefill,
    std::span<const std::uint32_t> ids, PrefillProfiler& profiler,
    const MetalCounterSampleBuffer& samples, CounterSamplingMode sampling_mode,
    std::span<std::uint64_t> timestamps,
    std::span<std::uint8_t> window_end_markers) {
    const PrefillProfilerStatus initial_profile = profiler.status();
    const bool valid_window_storage =
        sampling_mode == CounterSamplingMode::StageBoundaryEncoderSplit
            ? window_end_markers.size() == initial_profile.event_count
            : window_end_markers.empty();
    if (timestamps.size() != initial_profile.required_sample_count ||
        !valid_window_storage) {
        std::cerr << "prefill profile persistent recorder mismatch\n";
        return {.exit_code = kExitProfile};
    }
    const PrefillProgressResult begun = begin_prefill_progress(prefill, *harness.step, 0, 0, ids);
    if (!begun) {
        std::cerr << "begin_prefill_progress failed: progress_error="
                  << static_cast<unsigned>(begun.error)
                  << " encode_error=" << static_cast<unsigned>(begun.encode_error) << '\n';
        return {.exit_code = kExitEncode};
    }

    SubmissionResult result{
        .exit_code = 0,
        .seconds = 0.0,
        .command_buffers = 0,
        .chunks = begun.chunk_count,
    };
    while (prefill.progress.state != PrefillProgressState::Complete) {
        if (sampling_mode == CounterSamplingMode::StageBoundaryEncoderSplit &&
            profiler.begin_sample_window() != PrefillProfilerError::None) {
            std::cerr << "prefill profile window begin failed: profile_error="
                      << static_cast<unsigned>(profiler.status().error) << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitProfile,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        auto command_buffer = create_command_buffer(*harness.queue);
        if (!command_buffer) {
            poison_prefill(prefill);
            return {.exit_code = kExitCommandBuffer,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        std::optional<MetalComputePass> compute_pass;
        if (sampling_mode == CounterSamplingMode::StageBoundaryEncoderSplit) {
            auto pass = begin_stage_sampled_compute_pass(
                std::move(*command_buffer.command_buffer), samples, {0, 1});
            if (!pass) {
                std::cerr << "stage sampled compute pass failed: stage_error="
                          << static_cast<unsigned>(pass.error) << '\n';
                poison_prefill(prefill);
                return {.exit_code = kExitComputePass,
                        .seconds = result.seconds,
                        .command_buffers = result.command_buffers,
                        .chunks = result.chunks};
            }
            compute_pass = std::move(*pass.compute_pass);
        } else {
            auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
            if (!pass) {
                poison_prefill(prefill);
                return {.exit_code = kExitComputePass,
                        .seconds = result.seconds,
                        .command_buffers = result.command_buffers,
                        .chunks = result.chunks};
            }
            compute_pass = std::move(*pass.compute_pass);
        }
        const ProfiledPrefillProgressResult encoded = encode_prefill_unit(
            prefill, *harness.step, *compute_pass, profiler, samples);
        if (!encoded.profile) {
            std::cerr << "prefill profile encoding failed: profile_error="
                      << static_cast<unsigned>(encoded.profile.error)
                      << " counter_error="
                      << static_cast<unsigned>(encoded.profile.counter_error)
                      << " stage_error="
                      << static_cast<unsigned>(encoded.profile.stage_error)
                      << " event_cursor=" << encoded.profile.event_cursor
                      << " mismatch_index=" << encoded.profile.mismatch_index << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitProfile,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        if (!encoded.progress) {
            std::cerr << "encode_prefill_unit failed: progress_error="
                      << static_cast<unsigned>(encoded.progress.error)
                      << " encode_error=" << static_cast<unsigned>(encoded.progress.encode_error)
                      << " command_error="
                      << static_cast<unsigned>(encoded.progress.command_error) << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitEncode,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        auto ended = end_compute_pass(std::move(*compute_pass));
        if (!ended) {
            poison_prefill(prefill);
            return {.exit_code = kExitEndPass,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        const auto start = std::chrono::steady_clock::now();
        auto pending = commit(std::move(*ended.command_buffer));
        if (!pending) {
            poison_prefill(prefill);
            return {.exit_code = kExitCommit,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        const auto execution = wait_until_completed(std::move(*pending.pending_execution));
        result.seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        ++result.command_buffers;
        if (!execution) {
            std::cerr << "profiled prefill unit execution failed: "
                      << execution.failure_description.view() << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitExecution,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
        if (sampling_mode == CounterSamplingMode::StageBoundaryEncoderSplit) {
            PrefillProfileSampleWindow window;
            if (profiler.finish_sample_window(window) !=
                PrefillProfilerError::None) {
                std::cerr << "prefill profile window finish failed: profile_error="
                          << static_cast<unsigned>(profiler.status().error) << '\n';
                poison_prefill(prefill);
                return {.exit_code = kExitProfile,
                        .seconds = result.seconds,
                        .command_buffers = result.command_buffers,
                        .chunks = result.chunks};
            }
            const std::size_t destination_offset = window.event_begin * 2U;
            if (window.event_begin >= window_end_markers.size() ||
                window.event_count >
                    window_end_markers.size() - window.event_begin ||
                destination_offset > timestamps.size() ||
                window.sample_count > timestamps.size() - destination_offset) {
                std::cerr << "prefill profile sample window exceeds persistent recorder\n";
                poison_prefill(prefill);
                return {.exit_code = kExitCounterResolve,
                        .seconds = result.seconds,
                        .command_buffers = result.command_buffers,
                        .chunks = result.chunks};
            }
            window_end_markers[window.event_begin + window.event_count - 1U] = 1U;
            const CounterResolveError sample_resolve = resolve_counter_samples(
                samples, 0, window.sample_count,
                timestamps.subspan(destination_offset, window.sample_count));
            if (sample_resolve != CounterResolveError::None) {
                std::cerr << "counter sample window resolution failed: counter_resolve_error="
                          << static_cast<unsigned>(sample_resolve) << '\n';
                poison_prefill(prefill);
                return {.exit_code = kExitCounterResolve,
                        .seconds = result.seconds,
                        .command_buffers = result.command_buffers,
                        .chunks = result.chunks};
            }
        }
        const PrefillProgressResult committed = commit_prefill_unit(prefill, *harness.step);
        if (!committed) {
            std::cerr << "commit_prefill_unit failed: progress_error="
                      << static_cast<unsigned>(committed.error)
                      << " routed_up_status="
                      << static_cast<std::uint32_t>(
                             committed.routed_up_status)
                      << " routed_down_status="
                      << static_cast<std::uint32_t>(
                             committed.routed_down_status)
                      << '\n';
            poison_prefill(prefill);
            return {.exit_code = kExitExecution,
                    .seconds = result.seconds,
                    .command_buffers = result.command_buffers,
                    .chunks = result.chunks};
        }
    }
    return result;
}

constexpr std::array<std::string_view, kPrefillProfileEventClassCount>
    kPrefillProfileClassNames{
        "embedding",
        "layer_input_normalization",
        "gdn_projection",
        "gdn_convolution",
        "gdn_gate_hoist",
        "gdn_recurrence_serial_step",
        "gdn_recurrence_register_loop",
        "gdn_gate_normalization",
        "gdn_output_projection",
        "attention_projection",
        "attention_qk_rope",
        "attention_partial",
        "attention_combine",
        "attention_staged_scores",
        "attention_staged_softmax",
        "attention_staged_values",
        "attention_output_projection",
        "moe_residual_input",
        "moe_post_normalization",
        "moe_router",
        "moe_router_select_serial",
        "moe_router_select_parallel",
        "moe_expert_union",
        "moe_routed_task_build",
        "moe_expert_upgate",
        "moe_expert_down",
        "moe_expert_combine",
        "moe_residual_output",
        "moe_native_routed_upgate",
        "moe_shared_expert_upgate",
        "moe_native_routed_down",
        "moe_shared_expert_down",
        "moe_native_routed_shared_upgate",
        "moe_native_routed_shared_down",
        "attention_streaming",
    };

// Re-executes the cached ICB one level per command buffer after a full
// warm re-execution, and aggregates each level's command-buffer GPU time
// by the recorded command classes. A level containing mixed classes is
// split proportionally by command count and counted separately so
// approximation cannot masquerade as exact attribution.
int run_graph_level_profile(
    tatara::tools::DecodeHarness& harness,
    PrefillStep& prefill) {
    const std::vector<std::uint32_t>& levels =
        prefill.command_graph.level_command_begins;
    const std::vector<std::uint8_t>& classes =
        prefill.command_graph.command_classes;
    if (levels.size() < 2u ||
        classes.size() != prefill.command_graph.command_count) {
        return kExitCommandGraph;
    }
    const int warm_exit =
        run_graph_replay_measurements(harness, prefill, 1u);
    if (warm_exit != 0) {
        return warm_exit;
    }
    std::array<double, kPrefillProfileEventClassCount>
        class_seconds{};
    double total_seconds = 0.0;
    double mixed_seconds = 0.0;
    std::uint32_t mixed_levels = 0;
    for (std::size_t level = 0; level + 1u < levels.size();
         ++level) {
        auto command_buffer =
            create_command_buffer(*harness.queue);
        if (!command_buffer) {
            return kExitCommandBuffer;
        }
        auto pass = begin_compute_pass(
            std::move(*command_buffer.command_buffer));
        if (!pass) {
            return kExitComputePass;
        }
        if (declare_graph_residency(
                *pass.compute_pass, harness, prefill) !=
            MetalCommandError::None) {
            return kExitEncode;
        }
        const std::uint32_t level_begin = levels[level];
        const std::uint32_t level_end = levels[level + 1u];
        if (execute_indirect_commands(
                *pass.compute_pass,
                prefill.command_graph.commands, level_begin,
                level_end - level_begin) !=
            MetalCommandError::None) {
            return kExitEncode;
        }
        auto ended =
            end_compute_pass(std::move(*pass.compute_pass));
        if (!ended) {
            return kExitEndPass;
        }
        auto pending = commit(std::move(*ended.command_buffer));
        if (!pending) {
            return kExitCommit;
        }
        const auto execution = wait_until_completed_timed(
            std::move(*pending.pending_execution));
        if (!execution) {
            std::cerr << "graph level profile failed at level "
                      << level << ": "
                      << execution.failure_description.view()
                      << '\n';
            return kExitExecution;
        }
        const MetalExecutionTiming timing = execution.timing;
        if (!std::isfinite(timing.gpu_start_seconds) ||
            !std::isfinite(timing.gpu_end_seconds) ||
            timing.gpu_end_seconds <= timing.gpu_start_seconds) {
            continue;
        }
        const double level_seconds =
            timing.gpu_end_seconds - timing.gpu_start_seconds;
        total_seconds += level_seconds;
        const std::uint32_t level_commands =
            level_end - level_begin;
        bool homogeneous = true;
        for (std::uint32_t command = level_begin + 1u;
             command < level_end; ++command) {
            if (classes[command] != classes[level_begin]) {
                homogeneous = false;
                break;
            }
        }
        if (!homogeneous) {
            ++mixed_levels;
            mixed_seconds += level_seconds;
        }
        for (std::uint32_t command = level_begin;
             command < level_end; ++command) {
            class_seconds[classes[command]] +=
                level_seconds / level_commands;
        }
    }
    std::cerr << "graph level profile: levels "
              << levels.size() - 1u << ", total gpu "
              << total_seconds << " s, mixed levels "
              << mixed_levels << " carrying " << mixed_seconds
              << " s\n";
    std::array<std::size_t, kPrefillProfileEventClassCount>
        order{};
    for (std::size_t index = 0; index < order.size(); ++index) {
        order[index] = index;
    }
    std::sort(order.begin(), order.end(),
              [&class_seconds](std::size_t left,
                               std::size_t right) {
                  return class_seconds[left] >
                         class_seconds[right];
              });
    for (const std::size_t index : order) {
        if (class_seconds[index] <= 0.0) {
            continue;
        }
        std::cerr << "  " << kPrefillProfileClassNames[index]
                  << ": " << class_seconds[index] << " s ("
                  << 100.0 * class_seconds[index] /
                         total_seconds
                  << "%)\n";
    }
    return 0;
}

std::size_t maximum_layer_major_window_samples(
    std::span<const PrefillProfileEvent> events) noexcept {
    std::size_t maximum_events = 0;
    std::size_t window_events = 0;
    std::uint64_t prior_layer = 0;
    std::uint32_t prior_chunk = 0;
    bool have_window = false;
    for (const PrefillProfileEvent& event : events) {
        const std::uint64_t layer =
            event.layer_index == kNoPrefillProfileLayerIndex ? 0 : event.layer_index;
        if (!have_window || layer != prior_layer ||
            event.chunk_ordinal != prior_chunk) {
            if (window_events > maximum_events) {
                maximum_events = window_events;
            }
            window_events = 0;
            prior_layer = layer;
            prior_chunk = event.chunk_ordinal;
            have_window = true;
        }
        ++window_events;
    }
    if (window_events > maximum_events) {
        maximum_events = window_events;
    }
    return maximum_events * 2U;
}

void print_profile_report(const PrefillProfileReport& report,
                          CounterSamplingMode sampling_mode,
                          std::size_t maximum_window_samples) {
    std::cout << "tatara_prefill_profile_metadata sampling_mode="
              << (sampling_mode == CounterSamplingMode::StageBoundaryEncoderSplit
                      ? "stage-boundary"
                      : "dispatch-boundary")
              << " artifact_claim="
              << (sampling_mode == CounterSamplingMode::StageBoundaryEncoderSplit
                      ? "encoder-split-diagnostic-not-shipping-artifact"
                      : "dispatch-boundary-control")
              << " maximum_planned_window_samples=" << maximum_window_samples
              << '\n';
    for (std::size_t index = 0; index < report.stages.size(); ++index) {
        const PrefillProfileStageReport& stage = report.stages[index];
        std::cout << "tatara_prefill_profile_stage"
                  << " class=" << kPrefillProfileClassNames[index]
                  << " class_id=" << index
                  << " events=" << stage.event_count
                  << " kernel_ticks=" << stage.kernel_ticks
                  << " gap_ticks=" << stage.gap_ticks << '\n';
    }
    std::cout << "tatara_prefill_profile_total"
              << " events=" << report.event_count
              << " kernel_ticks=" << report.kernel_ticks
              << " gap_ticks=" << report.gap_ticks << '\n';
}

void print_moe_pair_report(std::string_view stage,
                           const PrefillProfilePairReport& report) {
    std::cout << "tatara_prefill_profile_moe_pair"
              << " stage=" << stage
              << " pairs=" << report.pair_count
              << " routed_ticks=" << report.first_kernel_ticks
              << " shared_ticks=" << report.second_kernel_ticks
              << " overlap_ticks=" << report.overlap_ticks
              << " routed_exclusive_ticks="
              << report.first_exclusive_ticks
              << " shared_exclusive_ticks="
              << report.second_exclusive_ticks
              << " union_ticks=" << report.union_ticks << '\n';
}

bool write_expert_count_capture(std::string_view path,
                                const ExpertCountCapture& capture) {
    if (path.empty() || !expert_count_capture_complete(capture)) {
        return false;
    }
    try {
        std::ofstream output(std::string{path},
                             std::ios::binary | std::ios::trunc);
        if (!output) {
            return false;
        }
        output << "{\n"
               << "  \"schema_version\": 1,\n"
               << "  \"prefill_rows\": " << capture.prefill_rows << ",\n"
               << "  \"plan\": {\n"
               << "    \"layers\": " << capture.layer_count << ",\n"
               << "    \"experts\": " << capture.experts << ",\n"
               << "    \"active_experts\": " << capture.active_experts << "\n"
               << "  },\n"
               << "  \"policy\": {\n"
               << "    \"schedule\": \"layer_major\",\n"
               << "    \"first_chunk_rows\": " << capture.first_chunk_rows
               << ",\n"
               << "    \"maximum_block_rows\": "
               << capture.maximum_block_rows << "\n"
               << "  },\n"
               << "  \"records\": [\n";
        for (std::size_t index = 0; index < capture.records.size();
             ++index) {
            const ExpertCountCaptureRecord& record =
                capture.records[index];
            output << "    {\"layer_index\": " << record.layer_index
                   << ", \"chunk_ordinal\": " << record.chunk_ordinal
                   << ", \"chunk_rows\": " << record.chunk_rows
                   << ", \"routed_row_counts\": [";
            const std::uint32_t* counts =
                capture.counts.data() + record.count_offset;
            for (std::uint32_t expert = 0; expert < capture.experts;
                 ++expert) {
                if (expert != 0) {
                    output << ", ";
                }
                output << counts[expert];
            }
            output << "], \"shared_rows\": "
                   << counts[capture.experts] << '}';
            if (index + 1u != capture.records.size()) {
                output << ',';
            }
            output << '\n';
        }
        output << "  ]\n"
               << "}\n";
        output.flush();
        return static_cast<bool>(output);
    } catch (...) {
        return false;
    }
}

SubmissionResult submit_decode_token(tatara::tools::DecodeHarness& harness, std::uint32_t token,
                                     std::uint32_t context) {
    DecodeStep& decode = *harness.step;
    std::memcpy(decode.token_id.contents(), &token, sizeof(token));
    auto command_buffer = create_command_buffer(*harness.queue);
    if (!command_buffer) {
        return {.exit_code = kExitCommandBuffer};
    }
    auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
    if (!pass) {
        return {.exit_code = kExitComputePass};
    }
    if (encode_token(decode, *pass.compute_pass, context) != MetalCommandError::None) {
        std::cerr << "decode encoding failed at context " << context << '\n';
        return {.exit_code = kExitDecodeEncode};
    }
    auto ended = end_compute_pass(std::move(*pass.compute_pass));
    if (!ended) {
        return {.exit_code = kExitEndPass};
    }
    const auto start = std::chrono::steady_clock::now();
    auto pending = commit(std::move(*ended.command_buffer));
    if (!pending) {
        return {.exit_code = kExitCommit};
    }
    const auto execution = wait_until_completed(std::move(*pending.pending_execution));
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (!execution) {
        std::cerr << "decode execution failed: " << execution.failure_description.view() << '\n';
        return {.exit_code = kExitExecution, .seconds = seconds};
    }
    return {.exit_code = 0, .seconds = seconds, .command_buffers = 1};
}

} // namespace

int main(int argument_count, char** arguments) {
    if (argument_count < 5 || argument_count > 8) {
        std::cerr << "usage: tatara_block_prefill_probe RECORD ARTIFACT_ROOT"
                     " IDS DUMP_OUT [POSITIONS]"
                     " [serial|chunk256|chunk2048|layer2048|layer2048fast|"
                     "layer2048fast-n1|layer2048fast-r2|"
                     "layer2048fast-n1-r2|layer2048fast-n1-r2-a1|"
                     "layer2048fast-n1-r2s-a1|"
                     "layer2048fast-n1-r2s-a1-steel|"
                     "layer2048fast-steel-full|"
                     "layer2048fast-steel-full-graph-compile|"
                     "layer2048fast-steel-full-graph-lane-events-compile|"
                     "layer2048fast-steel-full-graph|"
                     "layer2048fast-steel-full-graph-streaming|"
                     "layer2048fast-steel-full-graph-streaming-warm|"
                     "layer2048fast-steel-full-graph-lane-events-warm|"
                     "layer2048fast-steel-full-graph-qtile2048-warm|"
                     "layer2048fast-steel-full-graph-banded3|"
                     "layer2048fast-steel-full-submit2|"
                     "layer2048fast-steel-full-submit4|"
                     "layer2048fast-steel-full-submit8|"
                     "layer2048fast-steel-full-submit16|"
                     "layer2048fast-steel-full-submit32|"
                     "layer2048fast-steel-full-submit64|"
                     "layer2048fast-steel-full-inflight2|"
                     "layer2048fast-steel-full-inflight4|"
                     "layer2048fast-steel-full-inflight8|"
                     "layer2048fast-steel-full-inflight16|"
                     "layer2048fast-steel-full-inflight32|"
                     "layer2048fast-steel-full-inflight64|"
                     "layer2048fast-profile|"
                     "layer2048fast-profile-n1|"
                     "layer2048fast-profile-serial|"
                     "layer2048fast-profile-stage|"
                     "layer2048fast-profile-stage-n1|"
                     "layer2048fast-profile-stage-r2|"
                     "layer2048fast-profile-stage-n1-r2|"
                     "layer2048fast-profile-stage-n1-r2-a1|"
                     "layer2048fast-profile-stage-steel-full|"
                     "layer2048fast-profile-stage-serial]"
                     " [EXPERT_COUNTS_JSON]\n";
        return kExitUsage;
    }

    const auto& plan = tatara::model::qwen36::generated::kModelPlan;
    const auto ids_bytes = tatara::tools::read_file(arguments[3]);
    if (ids_bytes.empty()) {
        std::cerr << "ids file is empty or unreadable\n";
        return kExitIdsUnreadable;
    }
    const PromptIds prompt =
        parse_prompt_ids(ids_bytes, plan.dimensions.vocabulary,
                         plan.tokenizer.maximum_context);
    if (!prompt.valid) {
        return kExitIdsInvalid;
    }
    std::uint32_t positions = static_cast<std::uint32_t>(prompt.ids.size());
    if (argument_count >= 6 && !parse_u32(arguments[5], positions)) {
        std::cerr << "positions is not an unsigned integer\n";
        return kExitPositions;
    }
    if (positions < 2 || positions > prompt.ids.size() ||
        positions > plan.tokenizer.maximum_context) {
        std::cerr << "positions " << positions << " must be in 2..min(ids_count, "
                  << plan.tokenizer.maximum_context << ")\n";
        return kExitPositions;
    }
    FixturePolicy fixture_policy = kChunk256;
    if (argument_count >= 7 && !parse_fixture_policy(arguments[6], fixture_policy)) {
        std::cerr << "fixture policy must be serial, chunk256, chunk2048, layer2048,"
                     " layer2048fast, layer2048fast-n1,"
                     " layer2048fast-r2, layer2048fast-n1-r2,"
                     " layer2048fast-n1-r2-a1,"
                     " layer2048fast-n1-r2s-a1,"
                     " layer2048fast-n1-r2s-a1-steel,"
                     " layer2048fast-steel-full,"
                     " layer2048fast-steel-full-graph-compile,"
                     " layer2048fast-steel-full-graph-lane-events-compile,"
                     " layer2048fast-steel-full-graph,"
                     " layer2048fast-steel-full-graph-bm64-wm2-wn2-compile,"
                     " layer2048fast-steel-full-graph-bm64-wm2-wn2-warm,"
                     " layer2048fast-steel-full-graph-streaming,"
                     " layer2048fast-steel-full-graph-streaming-warm,"
                     " layer2048fast-steel-full-graph-lane-events-warm,"
                     " layer2048fast-steel-full-graph-qtile2048-warm,"
                     " layer2048fast-steel-full-graph-banded3,"
                     " layer2048fast-steel-full-submit{2,4,8,16,32,64},"
                     " layer2048fast-steel-full-inflight{2,4,8,16,32,64},"
                     " layer2048fast-profile, layer2048fast-profile-n1,"
                     " layer2048fast-profile-serial,"
                     " layer2048fast-profile-stage,"
                     " layer2048fast-profile-stage-n1,"
                     " layer2048fast-profile-stage-r2,"
                     " layer2048fast-profile-stage-n1-r2,"
                     " layer2048fast-profile-stage-n1-r2-a1, or"
                     " layer2048fast-profile-stage-steel-full, or"
                     " layer2048fast-profile-stage-serial\n";
        return kExitPolicy;
    }
    graph_serial_diagnostic_requested =
        fixture_policy.command_graph_serial_diagnostic;
    graph_repeat_requested = fixture_policy.command_graph_repeat;
    graph_warm_requested = fixture_policy.command_graph_warm;
    graph_banded_requested =
        fixture_policy.command_graph_banded;
    graph_level_profile_requested =
        fixture_policy.command_graph_level_profile;
    const bool expert_count_capture_requested = argument_count == 8;
    const std::string_view expert_count_output_path =
        expert_count_capture_requested ? std::string_view{arguments[7]}
                                       : std::string_view{};
    if (expert_count_capture_requested &&
        (expert_count_output_path.empty() ||
         fixture_policy.serial || fixture_policy.profile ||
         fixture_policy.schedule != PrefillSchedule::LayerMajor)) {
        std::cerr << "expert-count capture requires an unprofiled"
                     " layer-major policy and a non-empty output path\n";
        return kExitExpertCountCaptureContract;
    }

    const std::uint64_t required_capacity =
        std::uint64_t{positions} +
        kFixtureContinuationTokens - 1u;
    if (required_capacity > plan.tokenizer.maximum_context ||
        required_capacity >
            std::numeric_limits<std::uint32_t>::max()) {
        std::cerr << "prompt plus " << kFixtureContinuationTokens
                  << "-token continuation exceeds model context "
                  << plan.tokenizer.maximum_context << '\n';
        return kExitCapacity;
    }
    auto harness = tatara::tools::boot_decode(
        arguments[1], arguments[2],
        static_cast<std::uint32_t>(required_capacity));
    if (!harness) {
        return kBootExitBase + harness.exit_code;
    }
    if (required_capacity > harness.capacity) {
        std::cerr << "prompt plus " << kFixtureContinuationTokens
                  << "-token continuation requires capacity " << required_capacity << ", allocated "
                  << harness.capacity << '\n';
        return kExitCapacity;
    }

    const std::uint32_t prefill_rows = positions - 1u;
    std::uint32_t chunks = 0;
    std::uint32_t prompt_command_buffers = 0;
    std::uint64_t scratch_bytes = 0;
    double prompt_seconds = 0.0;
    double prompt_gpu_seconds = 0.0;
    double prompt_schedule_seconds = 0.0;
    std::uint32_t prompt_timed_command_buffers = 0;
    double handoff_seconds = 0.0;
    PrefillProfileReport profile_report;
    PrefillProfilePairReport moe_up_pair_report;
    PrefillProfilePairReport moe_down_pair_report;
    bool profile_report_ready = false;
    bool moe_pair_reports_ready = false;
    std::size_t profile_maximum_window_samples = 0;
    std::optional<ExpertCountCapture> expert_count_capture;

    if (fixture_policy.serial) {
        for (std::uint32_t index = 0; index < positions; ++index) {
            const SubmissionResult serial = submit_decode_token(harness, prompt.ids[index], index);
            if (serial.exit_code != 0) {
                return serial.exit_code;
            }
            prompt_seconds += serial.seconds;
            ++prompt_command_buffers;
            advance_decode_state(*harness.step);
        }
    } else {
        if (!harness.library) {
            return kExitLibrary;
        }
        PipelineResult pipelines = resolve_prefill_pipelines(
            *harness.device, *harness.library,
            fixture_policy.dense_qgemm ==
                QuantizedGemmPolicy::NativeDenseMma,
            fixture_policy.native_dense_steel,
            fixture_policy
                .native_dense_steel_gdn_bm64_wm2_wn2,
            fixture_policy.routed_qgemm ==
                QuantizedGemmPolicy::NativeRaggedMma,
            fixture_policy.native_routed_steel,
            fixture_policy.attention_kernel ==
                PrefillAttentionKernel::StagedGemmAdaptive,
            fixture_policy.command_graph);
        if (!pipelines) {
            return pipelines.exit_code;
        }
        const PrefillPolicy geometry_policy{
            .schedule = fixture_policy.schedule,
            .context_capacity = harness.capacity,
            .maximum_block_rows = fixture_policy.maximum_block,
            .first_chunk_rows = fixture_policy.first_chunk,
            .query_tile_rows = fixture_policy.query_tile,
            .attention_partition = kAttentionPartition,
            .exact_rows_per_threadgroup = kFixtureExactRows,
            .gdn_gate_hoist = fixture_policy.gdn_gate_hoist,
        };
        const auto geometry = make_prefill_geometry(plan, geometry_policy);
        if (!geometry) {
            std::cerr << "prefill geometry failed: " << static_cast<unsigned>(geometry.error)
                      << '\n';
            return kExitGeometry;
        }
        const std::uint32_t full_graph_chunk_count =
            fixture_policy.command_graph
                ? (prefill_rows <= fixture_policy.first_chunk
                       ? 1u
                       : 1u +
                             (prefill_rows -
                                  fixture_policy.first_chunk +
                              fixture_policy.maximum_block - 1u) /
                                 fixture_policy.maximum_block)
                : 1u;
        const std::uint32_t command_graph_chunk_count =
            fixture_policy.command_graph_banded
                ? std::min(kGraphBandedScratchLanes,
                           full_graph_chunk_count)
                : full_graph_chunk_count;
        PrefillExecutionPolicy execution_policy{
            .geometry = geometry_policy,
            .router_selector = fixture_policy.router_selector,
            .gdn_recurrence = fixture_policy.gdn_recurrence,
            .attention_kernel = fixture_policy.attention_kernel,
            .dense_qgemm = fixture_policy.dense_qgemm,
            .routed_qgemm = fixture_policy.routed_qgemm,
            .native_dense_steel =
                fixture_policy.native_dense_steel,
            .native_dense_steel_gdn_bm64_wm2_wn2 =
                fixture_policy
                    .native_dense_steel_gdn_bm64_wm2_wn2,
            .native_routed_shared_expert =
                fixture_policy.native_routed_shared_expert,
            .native_routed_steel =
                fixture_policy.native_routed_steel,
            .command_graph = fixture_policy.command_graph,
            .command_graph_lane_events =
                fixture_policy.command_graph_lane_events,
            .command_graph_chunk_count =
                command_graph_chunk_count,
            .maximum_units_per_submission =
                fixture_policy.maximum_units_per_submission,
            .maximum_inflight_units =
                fixture_policy.maximum_inflight_units,
        };
        auto prefill = create_prefill_step(*harness.device, geometry.geometry, execution_policy,
                                           std::move(pipelines.pipelines));
        if (!prefill) {
            std::cerr << "prefill step construction failed: "
                      << static_cast<unsigned>(prefill.error)
                      << ", requested_bytes=" << prefill.requested_bytes << '\n';
            return kExitPrefillStep;
        }
        scratch_bytes =
            geometry.geometry.steady_prefill_bytes +
            std::uint64_t{command_graph_chunk_count - 1u} *
                geometry.geometry.reusable_scratch_bytes +
            prefill.step->native_routed_workspace_bytes;
        const std::span<const std::uint32_t> prefill_ids(prompt.ids.data(), prefill_rows);
        if (fixture_policy.command_graph_compile_only) {
            const PrefillProgressResult begun =
                begin_prefill_progress(
                    *prefill.step, *harness.step, 0, 0,
                    prefill_ids);
            if (!begun) {
                std::cerr
                    << "command graph progress failed:"
                    << " progress_error="
                    << static_cast<unsigned>(begun.error)
                    << " encode_error="
                    << static_cast<unsigned>(begun.encode_error)
                    << '\n';
                return kExitCommandGraph;
            }
            const PrefillCommandGraphResult prepared =
                prepare_prefill_command_graph(
                    *harness.device, *prefill.step,
                    *harness.step);
            if (!prepared || prepared.cache_hit) {
                std::cerr
                    << "command graph preparation failed:"
                    << " graph_error="
                    << static_cast<unsigned>(prepared.error)
                    << " stage="
                    << static_cast<unsigned>(prepared.stage)
                    << " plan_error="
                    << static_cast<unsigned>(
                           prepared.plan_error)
                    << " command_error="
                    << static_cast<unsigned>(
                           prepared.command_error)
                    << " commands="
                    << prepared.command_count
                    << " nodes="
                    << prepared.node_count
                    << '\n';
                return kExitCommandGraph;
            }
            const PrefillCommandGraphResult cached =
                prepare_prefill_command_graph(
                    *harness.device, *prefill.step,
                    *harness.step);
            if (!cached || !cached.cache_hit ||
                cached.command_count != prepared.command_count ||
                cached.node_count != prepared.node_count ||
                cached.diagonal_count !=
                    prepared.diagonal_count ||
                cached.argument_arena_bytes !=
                    prepared.argument_arena_bytes) {
                std::cerr
                    << "command graph cache validation failed\n";
                return kExitCommandGraph;
            }
            std::cout
                << "block prefill command graph: PASS\n"
                << "  rows: " << prefill_rows << '\n'
                << "  chunks: " << command_graph_chunk_count
                << '\n'
                << "  nodes: " << prepared.node_count << '\n'
                << "  diagonals: " << prepared.diagonal_count
                << '\n'
                << "  commands: " << prepared.command_count
                << '\n'
                << "  argument arena bytes: "
                << prepared.argument_arena_bytes << '\n'
                << "  scratch bytes: " << scratch_bytes << '\n'
                << "  cache hit: 1\n"
                << "  command buffers submitted: 0\n";
            return 0;
        }
        SubmissionResult block;
        if (fixture_policy.profile) {
            const std::span<const LayerKind> profile_schedule(plan.layers.data(),
                                                              plan.layers.size());
            const PrefillProfilePlanResult count = make_prefill_profile_plan(
                geometry.geometry, execution_policy, prefill_rows, profile_schedule, {});
            if (count.error != PrefillProfilePlanError::EventCapacityInsufficient ||
                count.required_event_count == 0 ||
                count.required_event_count > std::numeric_limits<std::size_t>::max()) {
                std::cerr << "prefill profile sizing failed: profile_plan_error="
                          << static_cast<unsigned>(count.error)
                          << " required_events=" << count.required_event_count << '\n';
                return kExitProfilePlan;
            }
            std::vector<PrefillProfileEvent> profile_events(
                static_cast<std::size_t>(count.required_event_count));
            const PrefillProfilePlanResult profile_plan = make_prefill_profile_plan(
                geometry.geometry, execution_policy, prefill_rows, profile_schedule,
                profile_events);
            if (!profile_plan ||
                profile_plan.written_event_count != profile_events.size()) {
                std::cerr << "prefill profile plan failed: profile_plan_error="
                          << static_cast<unsigned>(profile_plan.error)
                          << " required_events=" << profile_plan.required_event_count
                          << " written_events=" << profile_plan.written_event_count << '\n';
                return kExitProfilePlan;
            }

            std::vector<CounterEvent> counter_events(profile_events.size());
            for (std::size_t index = 0; index < profile_events.size(); ++index) {
                counter_events[index].class_id =
                    static_cast<std::uint32_t>(profile_events[index].event_class);
            }
            CounterEventPlan counter_plan;
            const CounterPlanError counter_plan_error =
                plan_counter_events(true, fixture_policy.sampling_mode,
                                    counter_events, counter_plan);
            if (counter_plan_error != CounterPlanError::None) {
                std::cerr << "counter event planning failed: counter_plan_error="
                          << static_cast<unsigned>(counter_plan_error) << '\n';
                return kExitCounterPlan;
            }

            if (fixture_policy.sampling_mode ==
                CounterSamplingMode::StageBoundaryEncoderSplit) {
                profile_maximum_window_samples =
                    maximum_layer_major_window_samples(profile_events);
                if (fixture_policy.schedule != PrefillSchedule::LayerMajor ||
                    profile_maximum_window_samples == 0 ||
                    profile_maximum_window_samples >
                        kMaxStageBoundarySampleCount) {
                    std::cerr << "stage sample window rejected before encoding:"
                              << " schedule="
                              << static_cast<unsigned>(fixture_policy.schedule)
                              << " maximum_window_samples="
                              << profile_maximum_window_samples
                              << " capacity=" << kMaxStageBoundarySampleCount << '\n';
                    return kExitCounterPlan;
                }
            }
            const std::size_t native_sample_capacity =
                fixture_policy.sampling_mode ==
                        CounterSamplingMode::StageBoundaryEncoderSplit
                    ? kMaxStageBoundarySampleCount
                    : counter_plan.sample_count;
            CounterSampleBufferCreateResult sample_create;
            if (fixture_policy.sampling_mode ==
                CounterSamplingMode::StageBoundaryEncoderSplit) {
                auto capability_command_buffer =
                    create_command_buffer(*harness.queue);
                if (!capability_command_buffer) {
                    return kExitCommandBuffer;
                }
                sample_create = create_stage_timestamp_counter_sample_buffer(
                    *capability_command_buffer.command_buffer,
                    native_sample_capacity);
            } else {
                sample_create = create_timestamp_counter_sample_buffer(
                    *harness.device, native_sample_capacity);
            }
            if (!sample_create) {
                std::cerr << "counter sample buffer creation failed: counter_buffer_error="
                          << static_cast<unsigned>(sample_create.error)
                          << " sample_count=" << native_sample_capacity
                          << " full_request_samples=" << counter_plan.sample_count << '\n';
                return kExitCounterBuffer;
            }
            MetalCounterSampleBuffer samples = std::move(*sample_create.buffer);
            PrefillProfiler profiler(profile_events, samples.sample_capacity(),
                                     fixture_policy.sampling_mode);
            const PrefillProfilerStatus created = profiler.status();
            if (!created) {
                std::cerr << "prefill profiler construction failed: profile_error="
                          << static_cast<unsigned>(created.error)
                          << " required_samples=" << created.required_sample_count
                          << " sample_capacity=" << created.sample_capacity << '\n';
                return kExitProfile;
            }

            std::vector<std::uint64_t> timestamps(counter_plan.sample_count);
            std::vector<std::uint8_t> window_end_markers;
            if (fixture_policy.sampling_mode ==
                CounterSamplingMode::StageBoundaryEncoderSplit) {
                window_end_markers.resize(counter_plan.event_count, 0U);
            }
            block = submit_profiled_prefill(
                harness, *prefill.step, prefill_ids, profiler, samples,
                fixture_policy.sampling_mode, timestamps, window_end_markers);
            if (block.exit_code != 0) {
                return block.exit_code;
            }
            const PrefillProfilerStatus finalized = profiler.finalize();
            if (!finalized) {
                std::cerr << "prefill profiler finalization failed: profile_error="
                          << static_cast<unsigned>(finalized.error)
                          << " event_cursor=" << finalized.event_cursor
                          << " event_count=" << finalized.event_count << '\n';
                return kExitProfile;
            }

            if (fixture_policy.sampling_mode ==
                CounterSamplingMode::DispatchBoundary) {
                const CounterResolveError sample_resolve =
                    resolve_counter_samples(samples, 0, counter_plan.sample_count,
                                            timestamps);
                if (sample_resolve != CounterResolveError::None) {
                    std::cerr << "counter sample resolution failed: counter_resolve_error="
                              << static_cast<unsigned>(sample_resolve) << '\n';
                    return kExitCounterResolve;
                }
            }
            std::vector<CounterEventTiming> timings(counter_plan.event_count);
            const CounterResolveError timing_resolve =
                fixture_policy.sampling_mode ==
                        CounterSamplingMode::StageBoundaryEncoderSplit
                    ? resolve_counter_event_timings(
                          counter_plan, counter_events, timestamps,
                          window_end_markers, timings)
                    : resolve_counter_event_timings(
                          counter_plan, counter_events, timestamps, timings);
            if (timing_resolve != CounterResolveError::None) {
                std::cerr << "counter timing resolution failed: counter_resolve_error="
                          << static_cast<unsigned>(timing_resolve) << '\n';
                if (timing_resolve == CounterResolveError::NonMonotonicTimestamp) {
                    for (std::size_t event = 0; event < counter_events.size(); ++event) {
                        const std::uint64_t start = timestamps[event * 2U];
                        const std::uint64_t end = timestamps[event * 2U + 1U];
                        const bool window_end =
                            !window_end_markers.empty() &&
                            window_end_markers[event] != 0U;
                        const std::uint64_t next =
                            event + 1U < counter_events.size()
                                ? timestamps[event * 2U + 2U]
                                : end;
                        if (end < start ||
                            (!window_end && event + 1U < counter_events.size() &&
                             next < end)) {
                            std::cerr << "  first nonmonotonic event=" << event
                                      << " class_id=" << counter_events[event].class_id
                                      << " start=" << start << " end=" << end
                                      << " next=" << next
                                      << " window_end=" << (window_end ? 1 : 0) << '\n';
                            break;
                        }
                    }
                }
                return kExitCounterResolve;
            }
            const PrefillProfileReportError report_error =
                aggregate_prefill_profile_report(profile_events, timings, profile_report);
            if (report_error != PrefillProfileReportError::None) {
                std::cerr << "prefill profile aggregation failed: profile_report_error="
                          << static_cast<unsigned>(report_error) << '\n';
                return kExitProfileReport;
            }
            if (fixture_policy.routed_qgemm ==
                    QuantizedGemmPolicy::NativeRaggedMma &&
                !fixture_policy.native_routed_shared_expert) {
                const PrefillProfileReportError up_pair_error =
                    aggregate_prefill_profile_pair_report(
                        profile_events, timings,
                        PrefillProfileEventClass::MoeNativeRoutedUpGate,
                        PrefillProfileEventClass::MoeSharedExpertUpGate,
                        moe_up_pair_report);
                const PrefillProfileReportError down_pair_error =
                    aggregate_prefill_profile_pair_report(
                        profile_events, timings,
                        PrefillProfileEventClass::MoeNativeRoutedDown,
                        PrefillProfileEventClass::MoeSharedExpertDown,
                        moe_down_pair_report);
                if (up_pair_error != PrefillProfileReportError::None ||
                    down_pair_error != PrefillProfileReportError::None) {
                    std::cerr
                        << "prefill MoE pair aggregation failed:"
                        << " up_profile_report_error="
                        << static_cast<unsigned>(up_pair_error)
                        << " down_profile_report_error="
                        << static_cast<unsigned>(down_pair_error) << '\n';
                    return kExitProfileReport;
                }
                moe_pair_reports_ready = true;
            }
            profile_report_ready = true;
        } else {
            if (expert_count_capture_requested) {
                expert_count_capture.emplace();
                if (!prepare_expert_count_capture(
                        *prefill.step, prefill_rows, plan.layers.size(),
                        plan.mixture_of_experts.experts,
                        plan.mixture_of_experts.active_experts,
                        geometry_policy, *expert_count_capture)) {
                    std::cerr << "expert-count capture sizing or buffer"
                                 " contract failed before execution\n";
                    return kExitExpertCountCaptureContract;
                }
                block = submit_prefill(harness, *prefill.step, prefill_ids,
                                       *expert_count_capture);
            } else {
                block = submit_prefill(harness, *prefill.step, prefill_ids);
            }
        }
        if (block.exit_code != 0) {
            return block.exit_code;
        }
        if (expert_count_capture &&
            !write_expert_count_capture(expert_count_output_path,
                                        *expert_count_capture)) {
            std::cerr << "expert-count capture output write failed\n";
            return kExitExpertCountCaptureWrite;
        }
        prompt_seconds = block.seconds;
        prompt_gpu_seconds = block.gpu_seconds;
        prompt_schedule_seconds = block.schedule_seconds;
        prompt_timed_command_buffers =
            block.timed_command_buffers;
        prompt_command_buffers += block.command_buffers;
        chunks = block.chunks;

        const SubmissionResult decode =
            submit_decode_token(harness, prompt.ids[prefill_rows], prefill_rows);
        if (decode.exit_code != 0) {
            return decode.exit_code;
        }
        handoff_seconds = decode.seconds;
        ++prompt_command_buffers;
        advance_decode_state(*harness.step);
    }

    const std::uint32_t prediction =
        *static_cast<const std::uint32_t*>(harness.step->token_id.contents());
    if (prediction >= plan.dimensions.vocabulary) {
        std::cerr << "prompt produced out-of-range token " << prediction << '\n';
        return kExitTokenRange;
    }
    const std::span<const LayerKind> schedule(plan.layers.data(), plan.layers.size());
    std::size_t stalled_layer = 0;
    if (!tatara::tools::gated_delta_advanced(*harness.step, schedule, stalled_layer)) {
        std::cerr << "gated-delta state stalled at layer " << stalled_layer << '\n';
        return kExitStalled;
    }
    if (!tatara::tools::write_state_record(arguments[4], *harness.step, schedule, harness.capacity,
                                           positions)) {
        return kExitDump;
    }

    std::array<std::uint32_t, kFixtureContinuationTokens> continuation{};
    continuation[0] = prediction;
    double continuation_seconds = 0.0;
    for (std::uint32_t index = 1; index < kFixtureContinuationTokens; ++index) {
        const std::uint32_t context = positions + index - 1u;
        const SubmissionResult decode =
            submit_decode_token(harness, continuation[index - 1u], context);
        if (decode.exit_code != 0) {
            return decode.exit_code;
        }
        continuation_seconds += decode.seconds;
        advance_decode_state(*harness.step);
        continuation[index] = *static_cast<const std::uint32_t*>(harness.step->token_id.contents());
        if (continuation[index] >= plan.dimensions.vocabulary) {
            std::cerr << "continuation produced out-of-range token " << continuation[index]
                      << " at index " << index << '\n';
            return kExitTokenRange;
        }
    }
    if (!tatara::tools::gated_delta_advanced(*harness.step, schedule, stalled_layer)) {
        std::cerr << "gated-delta continuation stalled at layer " << stalled_layer
                  << "\n  continuation tokens before stall:";
        for (const std::uint32_t token : continuation) {
            std::cerr << ' ' << token;
        }
        std::cerr << '\n';
        return kExitStalled;
    }

    if (profile_report_ready) {
        print_profile_report(profile_report, fixture_policy.sampling_mode,
                             profile_maximum_window_samples);
    }
    if (moe_pair_reports_ready) {
        print_moe_pair_report("upgate", moe_up_pair_report);
        print_moe_pair_report("down", moe_down_pair_report);
    }
    std::cout << "prefill equality fixture: PASS_EXECUTION\n"
              << "  device: " << harness.device->name() << '\n'
              << "  positions: " << positions << ", prefill rows: " << prefill_rows
              << ", chunks: " << chunks << '\n'
              << "  fixture policy: " << fixture_policy.name;
    if (fixture_policy.serial) {
        std::cout << ", one-token reference\n";
    } else {
        std::cout << ", first/max chunk: " << fixture_policy.first_chunk << '/'
                  << fixture_policy.maximum_block << ", query tile: " << fixture_policy.query_tile
                  << ", router: "
                  << (fixture_policy.router_selector == PrefillRouterSelector::Serial
                          ? "serial"
                          : "parallel")
                  << ", gdn recurrence: "
                  << (fixture_policy.gdn_recurrence == PrefillGdnRecurrence::SerialSteps
                          ? "serial-steps"
                          : "register-loop")
                  << ", gate hoist: " << (fixture_policy.gdn_gate_hoist ? "on" : "off")
                  << ", attention: "
                  << (fixture_policy.attention_kernel ==
                              PrefillAttentionKernel::PartialCombine
                          ? "partial-combine"
                          : fixture_policy.attention_kernel ==
                                    PrefillAttentionKernel::
                                        StagedGemmAdaptive
                                ? "staged-gemm-adaptive"
                                : "streaming-flash-adaptive")
                  << ", profile: " << (fixture_policy.profile ? "on" : "off");
        if (fixture_policy.profile) {
            std::cout << ", sampling mode: "
                      << (fixture_policy.sampling_mode ==
                                  CounterSamplingMode::StageBoundaryEncoderSplit
                              ? "stage-boundary encoder-split diagnostic"
                              : "dispatch-boundary control")
                      << ", shipping artifact claim: none";
        }
        std::cout << '\n';
    }
    std::cout << "  scratch bytes: " << scratch_bytes << '\n'
              << "  image load: " << harness.load_seconds
              << " s, prompt GPU waits: " << prompt_seconds
              << " s, final prompt handoff: " << handoff_seconds
              << " s, continuation GPU waits: " << continuation_seconds << " s\n";
    if (prompt_timed_command_buffers != 0) {
        std::cout
            << "  prompt command-buffer timing: gpu="
            << prompt_gpu_seconds << " s, schedule="
            << prompt_schedule_seconds << " s, wall-minus-gpu="
            << prompt_seconds - prompt_gpu_seconds
            << " s, timed=" << prompt_timed_command_buffers << '/'
            << prompt_command_buffers << '\n';
    }
    std::cout
              << "  continuation tokens:";
    for (const std::uint32_t token : continuation) {
        std::cout << ' ' << token;
    }
    std::cout << '\n'
              << "  state record at prompt handoff: " << arguments[4] << '\n'
              << "  prompt command buffers submitted: " << prompt_command_buffers << '\n'
              << "  continuation command buffers submitted: " << kFixtureContinuationTokens - 1u
              << '\n'
              << "  correctness verdict: OPEN pending byte comparison with"
                 " serial oracle\n";
    return 0;
}
