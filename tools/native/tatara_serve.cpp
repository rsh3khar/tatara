#include "tatara_serve.h"

#include "decode_harness.h"

#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/engine/prefix_cache_domain.h"
#include "tatara/engine/prefix_cache_transaction.h"
#include "tatara/engine/speculative_decode.h"
#include "tatara/generated/kernel_library.h"
#include "tatara/generated/model_plan.h"
#include "tatara/host/capability.h"
#include "tatara/runtime/checked_arithmetic.h"
#include "tatara/runtime/decode_step.h"
#include "tatara/runtime/execution_identity.h"
#include "tatara/runtime/prefill_geometry.h"
#include "tatara/runtime/prefix_state_transfer.h"
#include "tatara/runtime/prefill_step.h"
#include "tatara/runtime/serving_capacity.h"
#include "tatara/runtime/state_slot_reset.h"
#include "tatara/service/completion_request.h"
#include "tatara/draft/dflash_checkpoint.h"
#include "tatara/service/configuration.h"
#include "tatara/service/observability.h"
#include "tatara/service/prefix_cache.h"
#include "tatara/service/server.h"
#include "tatara/text/chat_template.h"
#include "tatara/text/tokenizer.h"
#include "tatara/version.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace tatara::tools {
namespace {

using namespace tatara::backend::metal;
using namespace tatara::runtime;
using namespace tatara::service;

constexpr int kExitOk = 0;
constexpr int kExitUsage = 2;
constexpr int kExitConfigurationInvalid = 3;
constexpr int kExitRuntime = 7;
constexpr int kExitInterrupted = 8;

Server* g_server = nullptr;
std::atomic<bool> g_draining{false};

void request_drain(int) {
    g_draining.store(true, std::memory_order_relaxed);
    if (g_server != nullptr) {
        g_server->stop();
    }
}

std::string read_text(const std::string& path, bool& ok) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        ok = false;
        return {};
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    ok = true;
    return buffer.str();
}

std::string json_escape(std::string_view value) {
    std::string output;
    output.reserve(value.size());
    constexpr char hexadecimal[] = "0123456789abcdef";
    for (const char value_character : value) {
        const auto character = static_cast<unsigned char>(value_character);
        switch (character) {
        case '"':
            output += "\\\"";
            break;
        case '\\':
            output += "\\\\";
            break;
        case '\b':
            output += "\\b";
            break;
        case '\f':
            output += "\\f";
            break;
        case '\n':
            output += "\\n";
            break;
        case '\r':
            output += "\\r";
            break;
        case '\t':
            output += "\\t";
            break;
        default:
            if (character < 0x20U) {
                output += "\\u00";
                output.push_back(hexadecimal[character >> 4U]);
                output.push_back(hexadecimal[character & 0x0FU]);
            } else {
                output.push_back(static_cast<char>(character));
            }
            break;
        }
    }
    return output;
}

constexpr std::uint32_t kFixtureControlBlock = 256;
constexpr std::uint32_t kFixtureProductBlock = 2048;
constexpr std::uint32_t kFixtureProductQueryTile = 256;
constexpr std::uint32_t kFixtureExactRows = 16;
constexpr std::uint64_t kPrefixCacheStateBlockBytes = 2ULL << 20U;

constexpr int kExitLibrary = 94;
constexpr int kExitFunction = 95;
constexpr int kExitPipeline = 96;

enum class PrefillMode : std::uint8_t {
    SingleToken,
    Graph,
};

constexpr std::string_view kPrefillModeSingleToken = "single-token";
constexpr std::string_view kPrefillModeGraph = "graph";
constexpr char kServeUsage[] =
    "usage: tatara serve <config.toml> [graph|single-token]\n";

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
    constexpr std::array<std::string_view, 24> names{
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
        "capture_rows_blk",
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
        &result.pipelines.capture_rows,
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

PrefillPolicy serve_prefill_geometry_policy(
    std::uint32_t context_capacity) {
    return PrefillPolicy{
        .schedule = PrefillSchedule::LayerMajor,
        .context_capacity = context_capacity,
        .maximum_block_rows = kFixtureProductBlock,
        .first_chunk_rows = kFixtureControlBlock,
        .query_tile_rows = kFixtureProductQueryTile,
        .attention_partition = kAttentionPartition,
        .exact_rows_per_threadgroup = kFixtureExactRows,
        .gdn_gate_hoist = true,
    };
}

PrefillExecutionPolicy serve_prefill_execution_policy(
    PrefillPolicy geometry, std::uint32_t graph_scratch_lanes,
    bool conditioning_capture) {
    PrefillExecutionPolicy policy{
        .geometry = geometry,
        .router_selector = PrefillRouterSelector::Parallel,
        .gdn_recurrence = PrefillGdnRecurrence::RegisterLoop,
        .attention_kernel = PrefillAttentionKernel::SteelGemm,
        .staged_attention_minimum_context =
            std::min<std::uint32_t>(256, geometry.context_capacity - 1u),
        .dense_qgemm = QuantizedGemmPolicy::NativeDenseMma,
        .routed_qgemm = QuantizedGemmPolicy::NativeRaggedMma,
        .native_dense_steel = true,
        .native_routed_shared_expert = true,
        .native_routed_steel = true,
        .command_graph = true,
        .command_graph_chunk_count = graph_scratch_lanes,
        .maximum_units_per_submission = 1,
        .maximum_inflight_units = 1,
    };
    if (conditioning_capture) {
        policy.conditioning_capture = true;
        policy.conditioning_capture_rows = kFixtureProductBlock;
        for (std::uint32_t slot = 0;
             slot < tatara::draft::kDraftCaptureLayers; ++slot) {
            policy.capture_layers[slot] =
                tatara::draft::kDraftCaptureAfterTargetLayer[slot];
        }
    }
    return policy;
}

PrefillStepResult boot_graph_prefill(
    const DecodeHarness& harness,
    std::uint32_t graph_scratch_lanes,
    bool conditioning_capture) {
    if (!harness.library) {
        std::fprintf(stderr, "serve: prefill kernel library unavailable\n");
        return {};
    }
    PipelineResult pipelines = resolve_prefill_pipelines(
        *harness.device, *harness.library,
        /*native_dense_qgemm=*/true,
        /*native_dense_steel=*/true,
        /*native_dense_steel_gdn_bm64_wm2_wn2=*/false,
        /*native_routed_qgemm=*/true,
        /*native_routed_steel=*/true,
        /*staged_attention=*/true,
        /*command_graph=*/true);
    if (!pipelines) {
        std::fprintf(stderr, "serve: prefill pipeline resolution failed (%d)\n",
                     pipelines.exit_code);
        return {};
    }
    const auto& plan = tatara::model::qwen36::generated::kModelPlan;
    const PrefillPolicy geometry_policy =
        serve_prefill_geometry_policy(harness.capacity);
    const auto geometry = make_prefill_geometry(plan, geometry_policy);
    if (!geometry) {
        std::fprintf(stderr, "serve: prefill geometry failed (%u)\n",
                     static_cast<unsigned>(geometry.error));
        return {};
    }
    const PrefillExecutionPolicy execution_policy =
        serve_prefill_execution_policy(
            geometry_policy, graph_scratch_lanes, conditioning_capture);
    PrefillStepResult prefill = create_prefill_step(
        *harness.device, geometry.geometry, execution_policy,
        std::move(pipelines.pipelines));
    if (!prefill) {
        std::fprintf(stderr,
                     "serve: prefill step construction failed (%u),"
                     " requested_bytes=%llu\n",
                     static_cast<unsigned>(prefill.error),
                     static_cast<unsigned long long>(prefill.requested_bytes));
    }
    return prefill;
}

struct ServePrefixCache {
    std::optional<PrefixCache> cache;
    std::optional<MetalBuffer> state_arena;
    std::vector<std::byte> phase_evidence;
    std::uint32_t boundary_origin_tokens{0};
    std::uint32_t boundary_stride_tokens{0};
    std::uint32_t minimum_prefix_tokens{0};
    std::uint32_t tail_guard_tokens{0};
    std::uint64_t next_request_generation{1};
    std::uint64_t next_command_generation{1};

    bool enabled() const noexcept {
        return cache.has_value() && state_arena.has_value();
    }
};

struct SnapshotPublicationGuard {
    PrefixCache* cache_owner{nullptr};
    std::optional<PrefixCacheReservation> reservation;
    std::optional<engine::PrefixSnapshotTransaction> transaction;
    bool published{false};

    ~SnapshotPublicationGuard() {
        if (transaction &&
            transaction->state ==
                engine::PrefixCacheTransactionState::PendingPublication) {
            (void)engine::resolve_prefix_snapshot_terminal(
                *transaction,
                engine::PrefixCacheTerminalDisposition::EngineFailure);
        } else if (!transaction && cache_owner != nullptr &&
                   reservation) {
            (void)cache_owner->abort_snapshot(*reservation);
        }
    }

    bool resolve(engine::PrefixCacheTerminalDisposition disposition) noexcept {
        if (!transaction) {
            if (cache_owner == nullptr || !reservation) {
                return true;
            }
            const PrefixCacheError aborted =
                cache_owner->abort_snapshot(*reservation);
            reservation.reset();
            const bool successful =
                disposition ==
                    engine::PrefixCacheTerminalDisposition::
                        SuccessfulStopToken ||
                disposition ==
                    engine::PrefixCacheTerminalDisposition::
                        SuccessfulMaximumOutput;
            return aborted == PrefixCacheError::None &&
                   !successful;
        }
        if (transaction->state !=
            engine::PrefixCacheTransactionState::PendingPublication) {
            return transaction->state ==
                   engine::PrefixCacheTransactionState::Finished;
        }
        const bool resolved = static_cast<bool>(
            engine::resolve_prefix_snapshot_terminal(
                *transaction, disposition));
        if (resolved) {
            published =
                disposition ==
                    engine::PrefixCacheTerminalDisposition::
                        SuccessfulStopToken ||
                disposition ==
                    engine::PrefixCacheTerminalDisposition::
                        SuccessfulMaximumOutput;
            reservation.reset();
        }
        return resolved;
    }
};

engine::PrefixCacheCommandObservation
cache_command_observation(const MetalExecutionResult& execution) noexcept {
    if (execution) {
        return engine::PrefixCacheCommandObservation::Success;
    }
    return execution.state == MetalExecutionState::Completed ||
                   execution.state == MetalExecutionState::Error
               ? engine::PrefixCacheCommandObservation::ObservedFailure
               : engine::PrefixCacheCommandObservation::UnobservedFailure;
}

std::optional<runtime::CommandTicket>
next_cache_command(ServePrefixCache& cache, runtime::CommandKind kind,
                   runtime::RequestHandle request,
                   runtime::SlotHandle slot) noexcept {
    if (cache.next_command_generation == 0) {
        return std::nullopt;
    }
    const std::uint64_t generation = cache.next_command_generation;
    cache.next_command_generation =
        generation == std::numeric_limits<std::uint64_t>::max()
            ? 0
            : generation + 1u;
    return runtime::CommandTicket{
        .command_generation = generation,
        .kind = kind,
        .request = request,
        .slot = slot,
    };
}

std::optional<runtime::RequestHandle>
next_cache_request(ServePrefixCache& cache) noexcept {
    if (cache.next_request_generation == 0) {
        return std::nullopt;
    }
    const std::uint64_t generation = cache.next_request_generation;
    cache.next_request_generation =
        generation == std::numeric_limits<std::uint64_t>::max()
            ? 0
            : generation + 1u;
    return runtime::RequestHandle{
        .owner_index = 0,
        .owner_generation = generation,
    };
}

std::uint32_t largest_cache_boundary(
    std::uint32_t end_position, std::uint32_t after_position,
    const ServePrefixCache& cache) noexcept {
    if (!cache.enabled() || end_position < cache.tail_guard_tokens) {
        return 0;
    }
    const std::uint32_t limit = end_position - cache.tail_guard_tokens;
    if (limit < cache.boundary_origin_tokens ||
        limit < cache.minimum_prefix_tokens) {
        return 0;
    }
    const std::uint32_t steps =
        (limit - cache.boundary_origin_tokens) /
        cache.boundary_stride_tokens;
    const std::uint64_t selected =
        std::uint64_t{cache.boundary_origin_tokens} +
        std::uint64_t{steps} * cache.boundary_stride_tokens;
    if (selected > std::numeric_limits<std::uint32_t>::max() ||
        selected <= after_position ||
        selected < cache.minimum_prefix_tokens) {
        return 0;
    }
    return static_cast<std::uint32_t>(selected);
}

bool nonfatal_cache_reservation_error(PrefixCacheError error) noexcept {
    switch (error) {
    case PrefixCacheError::None:
    case PrefixCacheError::AlreadyPresent:
    case PrefixCacheError::PublicationInFlight:
    case PrefixCacheError::NoEvictableEntry:
    case PrefixCacheError::StateArenaExhausted:
    case PrefixCacheError::GenerationExhausted:
    case PrefixCacheError::StateTooLarge:
        return true;
    case PrefixCacheError::InvalidConfiguration:
    case PrefixCacheError::AllocationFailed:
    case PrefixCacheError::MetadataBudgetInsufficient:
    case PrefixCacheError::StateArenaBudgetInsufficient:
    case PrefixCacheError::ForeignDomain:
    case PrefixCacheError::InvalidOwner:
    case PrefixCacheError::EmptyPrefix:
    case PrefixCacheError::TokenLimitExceeded:
    case PrefixCacheError::BoundaryViolation:
    case PrefixCacheError::InvalidMaximumPosition:
    case PrefixCacheError::StaleReservation:
    case PrefixCacheError::StaleRestoreLease:
    case PrefixCacheError::InvalidTransition:
        return false;
    }
    return false;
}

bool execute_state_slot_reset(DecodeHarness& harness) {
    auto command_buffer = create_command_buffer(*harness.queue);
    if (!command_buffer) {
        return false;
    }
    auto pass = begin_blit_pass(std::move(*command_buffer.command_buffer));
    if (!pass) {
        return false;
    }
    StateSlotResetResult encoded =
        encode_state_slot_reset(*harness.step, harness.step->state,
                                *pass.blit_pass);
    if (!encoded) {
        return false;
    }
    auto ended = end_blit_pass(std::move(*pass.blit_pass));
    if (!ended) {
        if (encoded.ticket) {
            (void)abort_state_slot_reset(*encoded.ticket, *harness.step,
                                         harness.step->state);
        }
        return false;
    }
    if (encoded.completed) {
        return true;
    }
    auto pending = commit(std::move(*ended.command_buffer));
    if (!pending) {
        (void)abort_state_slot_reset(*encoded.ticket, *harness.step,
                                     harness.step->state);
        return false;
    }
    const MetalExecutionResult execution =
        wait_until_completed(std::move(*pending.pending_execution));
    if (!execution) {
        if (execution.state == MetalExecutionState::Completed ||
            execution.state == MetalExecutionState::Error) {
            (void)abort_state_slot_reset(*encoded.ticket, *harness.step,
                                         harness.step->state);
        }
        return false;
    }
    return complete_state_slot_reset(*encoded.ticket, *harness.step,
                                     harness.step->state) ==
           StateSlotResetError::None;
}

bool execute_prefix_restore(
    ServePrefixCache& cache, DecodeHarness& harness,
    runtime::RequestHandle request, runtime::SlotHandle slot,
    PrefixCacheRestoreLease lease) {
    const std::optional<runtime::CommandTicket> command =
        next_cache_command(cache, runtime::CommandKind::Restore, request, slot);
    if (!command) {
        (void)cache.cache->release_restore(
            lease, PrefixCacheRestoreDisposition::ObservedFailure);
        return false;
    }
    auto command_buffer = create_command_buffer(*harness.queue);
    if (!command_buffer) {
        (void)cache.cache->release_restore(
            lease, PrefixCacheRestoreDisposition::ObservedFailure);
        return false;
    }
    auto pass = begin_blit_pass(std::move(*command_buffer.command_buffer));
    if (!pass) {
        (void)cache.cache->release_restore(
            lease, PrefixCacheRestoreDisposition::ObservedFailure);
        return false;
    }
    const PrefillGeometryResult geometry = make_prefill_geometry(
        model::qwen36::generated::kModelPlan,
        serve_prefill_geometry_policy(harness.capacity));
    if (!geometry) {
        (void)cache.cache->release_restore(
            lease, PrefixCacheRestoreDisposition::ObservedFailure);
        return false;
    }
    PrefixStateTransferResult encoded = encode_prefix_state_restore(
        *harness.step, harness.step->state, geometry.geometry,
        *pass.blit_pass, *cache.state_arena, lease.state_offset_bytes,
        lease.position_tokens);
    if (!encoded) {
        (void)cache.cache->release_restore(
            lease, PrefixCacheRestoreDisposition::ObservedFailure);
        return false;
    }
    auto transaction = engine::make_prefix_restore_transaction(
        *cache.cache, *harness.step, harness.step->state,
        *cache.state_arena, request, slot, *command, lease,
        *encoded.ticket);
    if (!transaction) {
        (void)abort_prefix_state_transfer(*encoded.ticket, *harness.step,
                                          harness.step->state,
                                          *cache.state_arena);
        (void)cache.cache->release_restore(
            lease, PrefixCacheRestoreDisposition::ObservedFailure);
        return false;
    }
    auto ended = end_blit_pass(std::move(*pass.blit_pass));
    if (!ended) {
        (void)engine::observe_prefix_restore(
            *transaction.transaction,
            engine::PrefixCacheCommandObservation::ObservedFailure);
        return false;
    }
    auto pending = commit(std::move(*ended.command_buffer));
    if (!pending) {
        (void)engine::observe_prefix_restore(
            *transaction.transaction,
            engine::PrefixCacheCommandObservation::ObservedFailure);
        return false;
    }
    const MetalExecutionResult execution =
        wait_until_completed(std::move(*pending.pending_execution));
    return static_cast<bool>(engine::observe_prefix_restore(
        *transaction.transaction, cache_command_observation(execution)));
}

bool execute_prefix_snapshot(
    ServePrefixCache& cache, DecodeHarness& harness,
    runtime::RequestHandle request, runtime::SlotHandle slot,
    PrefixCacheReservation reservation,
    std::optional<engine::PrefixSnapshotTransaction>& transaction_out) {
    const std::optional<runtime::CommandTicket> command =
        next_cache_command(cache, runtime::CommandKind::Snapshot, request, slot);
    if (!command) {
        (void)cache.cache->abort_snapshot(reservation);
        return false;
    }
    auto command_buffer = create_command_buffer(*harness.queue);
    if (!command_buffer) {
        (void)cache.cache->abort_snapshot(reservation);
        return false;
    }
    auto pass = begin_blit_pass(std::move(*command_buffer.command_buffer));
    if (!pass) {
        (void)cache.cache->abort_snapshot(reservation);
        return false;
    }
    const PrefillGeometryResult geometry = make_prefill_geometry(
        model::qwen36::generated::kModelPlan,
        serve_prefill_geometry_policy(harness.capacity));
    if (!geometry) {
        (void)cache.cache->abort_snapshot(reservation);
        return false;
    }
    PrefixStateTransferResult encoded = encode_prefix_state_snapshot(
        *harness.step, harness.step->state, geometry.geometry,
        *pass.blit_pass, *cache.state_arena,
        reservation.state_offset_bytes, reservation.position_tokens,
        cache.phase_evidence);
    if (!encoded) {
        (void)cache.cache->abort_snapshot(reservation);
        return false;
    }
    auto made = engine::make_prefix_snapshot_transaction(
        *cache.cache, *harness.step, harness.step->state,
        *cache.state_arena, request, slot, *command, reservation,
        *encoded.ticket);
    if (!made) {
        (void)abort_prefix_state_transfer(*encoded.ticket, *harness.step,
                                          harness.step->state,
                                          *cache.state_arena);
        (void)cache.cache->abort_snapshot(reservation);
        return false;
    }
    transaction_out = std::move(*made.transaction);
    auto ended = end_blit_pass(std::move(*pass.blit_pass));
    if (!ended) {
        (void)engine::observe_prefix_snapshot(
            *transaction_out,
            engine::PrefixCacheCommandObservation::ObservedFailure);
        return false;
    }
    auto pending = commit(std::move(*ended.command_buffer));
    if (!pending) {
        (void)engine::observe_prefix_snapshot(
            *transaction_out,
            engine::PrefixCacheCommandObservation::ObservedFailure);
        return false;
    }
    const MetalExecutionResult execution =
        wait_until_completed(std::move(*pending.pending_execution));
    return static_cast<bool>(engine::observe_prefix_snapshot(
        *transaction_out, cache_command_observation(execution)));
}

bool boot_prefix_cache(
    const Configuration& configuration, PrefillMode prefill_mode,
    DecodeHarness& harness, const PrefillStepResult& prefill,
    ServePrefixCache& output) {
    output.boundary_origin_tokens = kFixtureControlBlock;
    output.boundary_stride_tokens = kFixtureProductBlock;
    // Cache every legal global-chunk boundary. These are reuse-policy values,
    // not serving limits; misses always execute the full admitted context.
    output.minimum_prefix_tokens = output.boundary_origin_tokens;
    output.tail_guard_tokens = 0;
    if (!configuration.cache.prompt_reuse ||
        configuration.cache.budget_bytes == 0) {
        std::printf(
            "serve: prefix cache disabled prompt_reuse=%u budget=%llu\n",
            configuration.cache.prompt_reuse ? 1U : 0U,
            static_cast<unsigned long long>(
                configuration.cache.budget_bytes));
        return true;
    }

    const PrefillGeometryResult geometry = make_prefill_geometry(
        model::qwen36::generated::kModelPlan,
        serve_prefill_geometry_policy(harness.capacity));
    if (!geometry) {
        return false;
    }
    const PrefixStateLayoutResult minimum_layout =
        make_prefix_state_layout(
            *harness.step, geometry.geometry,
            output.minimum_prefix_tokens);
    if (!minimum_layout) {
        return false;
    }
    PrefixCacheDomain provisional_domain{
        .diagnostic_digest = {},
        .generation = 1,
    };
    provisional_domain.diagnostic_digest.fill(1);
    PrefixCacheBudgetCreateResult provisional =
        create_prefix_cache_for_budget({
            .total_budget_bytes = configuration.cache.budget_bytes,
            .state_block_bytes = kPrefixCacheStateBlockBytes,
            .minimum_entry_state_bytes =
                minimum_layout.layout.total_bytes,
            .max_tokens_per_entry = harness.capacity,
            .boundary_origin_tokens = output.boundary_origin_tokens,
            .boundary_stride_tokens = output.boundary_stride_tokens,
            .domain = provisional_domain,
        });
    if (!provisional) {
        std::fprintf(
            stderr,
            "serve: prefix-cache budget planning failed error=%u"
            " budget=%llu minimum_state=%llu\n",
            static_cast<unsigned>(provisional.error),
            static_cast<unsigned long long>(
                configuration.cache.budget_bytes),
            static_cast<unsigned long long>(
                minimum_layout.layout.total_bytes));
        return false;
    }
    const PrefixCacheEvidence provisional_evidence =
        provisional.cache->evidence();
    if (provisional_evidence.free_entries == 0 ||
        provisional_evidence.free_entries >
            std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    const std::uint32_t maximum_entries =
        static_cast<std::uint32_t>(
            provisional_evidence.free_entries);
    provisional.cache.reset();

    const std::vector<std::byte> prepared_record =
        read_file(configuration.model.record.c_str());
    if (prepared_record.empty()) {
        return false;
    }
    runtime::PrefillCommandIdentity prefill_pipelines;
    runtime::PrefillCommandIdentity prefill_policy;
    if (prefill_mode == PrefillMode::Graph) {
        prefill_pipelines =
            execution_prefill_pipeline_identity(
                prefill.step->pipelines);
        prefill_policy =
            prefill_execution_policy_identity(
                prefill.step->policy);
    } else {
        ExecutionIdentityEncoder pipeline_encoder(
            "tatara.execution.single-token-prefill");
        pipeline_encoder.append_u8(1);
        prefill_pipelines = pipeline_encoder.finish();
        ExecutionIdentityEncoder policy_encoder(
            "tatara.execution.single-token-policy");
        policy_encoder.append_u8(1);
        prefill_policy = policy_encoder.finish();
    }

    const auto domain = engine::make_prefix_cache_domain({
        .model_package_identity =
            execution_model_package_identity(),
        .generated_plan_identity =
            execution_model_plan_identity(),
        .host_execution_identity =
            execution_host_identity(),
        .prepared_record_identity =
            execution_prepared_record_identity(prepared_record),
        .prepared_image_identity =
            execution_prepared_image_identity(*harness.step),
        .kernel_library_identity =
            execution_kernel_library_identity(),
        .decode_pipeline_identity =
            execution_decode_pipeline_identity(
                harness.step->pipelines),
        .prefill_pipeline_identity = prefill_pipelines,
        .prefill_policy_identity = prefill_policy,
        .tokenizer_template_identity =
            execution_tokenizer_identity(),
        .device_identity = metal_device_identity(*harness.device),
        .domain_generation = 1,
        .state_capacity = harness.capacity,
        .layer_count =
            static_cast<std::uint32_t>(
                harness.step->schedule.size()),
        .gated_delta_layers =
            harness.step->geometry.gated_delta_layers,
        .attention_layers =
            harness.step->geometry.attention_layers,
        .key_value_heads =
            geometry.geometry.key_value_heads,
        .attention_head_dimension =
            geometry.geometry.attention_head_dimension,
        .gated_delta_convolution_bytes =
            harness.step->geometry.gdn_conv_state_bytes,
        .gated_delta_recurrent_bytes =
            harness.step->geometry.gdn_recurrent_state_bytes,
        .state_layout_schema_version =
            kPrefixStateLayoutSchemaVersion,
        .transfer_schema_version =
            kPrefixStateTransferSchemaVersion,
        .reset_schema_version =
            kStateSlotResetSchemaVersion,
        .prefill_mode =
            prefill_mode == PrefillMode::Graph
                ? engine::PrefixCachePrefillMode::Graph
                : engine::PrefixCachePrefillMode::SingleToken,
        .total_cache_budget_bytes =
            configuration.cache.budget_bytes,
        .state_block_bytes = kPrefixCacheStateBlockBytes,
        .maximum_entries = maximum_entries,
        .maximum_tokens_per_entry = harness.capacity,
        .minimum_prefix_tokens =
            output.minimum_prefix_tokens,
        .tail_guard_tokens = output.tail_guard_tokens,
        .boundary_origin_tokens =
            output.boundary_origin_tokens,
        .boundary_stride_tokens =
            output.boundary_stride_tokens,
        .graph_scratch_lanes =
            configuration.memory.graph_scratch_lanes,
    });
    if (!domain) {
        std::fprintf(
            stderr,
            "serve: prefix-cache domain failed error=%u\n",
            static_cast<unsigned>(domain.error));
        return false;
    }
    PrefixCacheBudgetCreateResult created =
        create_prefix_cache_for_budget({
            .total_budget_bytes = configuration.cache.budget_bytes,
            .state_block_bytes = kPrefixCacheStateBlockBytes,
            .minimum_entry_state_bytes =
                minimum_layout.layout.total_bytes,
            .max_tokens_per_entry = harness.capacity,
            .boundary_origin_tokens = output.boundary_origin_tokens,
            .boundary_stride_tokens = output.boundary_stride_tokens,
            .domain = *domain.domain,
        });
    if (!created) {
        return false;
    }
    auto arena = create_shared_buffer(
        *harness.device,
        created.cache->budget().state_arena_bytes);
    if (!arena) {
        std::fprintf(
            stderr,
            "serve: prefix-cache state arena allocation failed"
            " bytes=%llu\n",
            static_cast<unsigned long long>(
                created.cache->budget().state_arena_bytes));
        return false;
    }
    output.phase_evidence.resize(
        prefix_state_phase_evidence_bytes(
            harness.step->schedule));
    output.state_arena =
        std::move(*arena.buffer);
    output.cache = std::move(*created.cache);
    const PrefixCacheBudget& budget = output.cache->budget();
    std::printf(
        "serve: prefix cache ready domain=%s entries=%zu"
        " state_arena=%llu total=%llu/%llu slack=%llu"
        " boundary=%u+%u*k minimum=%u tail_guard=%u\n",
        model::sha256_hex(
            output.cache->diagnostic_domain_digest()).c_str(),
        output.cache->evidence().free_entries,
        static_cast<unsigned long long>(
            budget.state_arena_bytes),
        static_cast<unsigned long long>(
            budget.total_payload_bytes),
        static_cast<unsigned long long>(
            budget.configured_total_budget_bytes),
        static_cast<unsigned long long>(
            budget.unallocated_budget_bytes),
        output.boundary_origin_tokens,
        output.boundary_stride_tokens,
        output.minimum_prefix_tokens,
        output.tail_guard_tokens);
    return true;
}

} // namespace

EngineCapabilities serve_capabilities() {
    const auto& plan = tatara::model::qwen36::generated::kModelPlan;
    return EngineCapabilities{
        .context_capacity = plan.tokenizer.maximum_context,
        .concurrent_requests = 1,
        .queued_admission = false,
        .request_deadlines = false,
        .bounded_drain = false,
        .prompt_reuse = true,
    };
}

int run_serve(std::vector<std::string_view> arguments) {
    if (arguments.empty() || arguments.size() > 2) {
        std::fputs(kServeUsage, stderr);
        return kExitUsage;
    }
    PrefillMode prefill_mode = PrefillMode::SingleToken;
    if (arguments.size() == 2) {
        if (arguments[1] == kPrefillModeGraph) {
            prefill_mode = PrefillMode::Graph;
        } else if (arguments[1] == kPrefillModeSingleToken) {
            prefill_mode = PrefillMode::SingleToken;
        } else {
            std::fputs(kServeUsage, stderr);
            return kExitUsage;
        }
    }
    bool ok = false;
    const std::string text = read_text(std::string(arguments[0]), ok);
    if (!ok) {
        std::fprintf(stderr, "serve: cannot read %.*s\n", static_cast<int>(arguments[0].size()),
                     arguments[0].data());
        return kExitConfigurationInvalid;
    }
    const auto parsed = parse_configuration(text);
    if (!parsed.ok) {
        std::fprintf(stderr, "serve: configuration invalid\n");
        for (const auto& diagnostic : parsed.diagnostics) {
            std::fprintf(stderr, "  %s\n", diagnostic.message.c_str());
        }
        return kExitConfigurationInvalid;
    }
    Configuration configuration = parsed.configuration;
    const auto execution_diagnostics =
        validate_configuration_for_engine(configuration, serve_capabilities());
    if (!execution_diagnostics.empty()) {
        std::fprintf(stderr, "serve: configuration exceeds the composed engine\n");
        for (const auto& diagnostic : execution_diagnostics) {
            std::fprintf(stderr, "  %s\n", diagnostic.message.c_str());
        }
        return kExitConfigurationInvalid;
    }

    const auto& plan = tatara::model::qwen36::generated::kModelPlan;
    const DecodeImagePlan image_plan =
        inspect_decode_image(configuration.model.record.c_str());
    if (!image_plan) {
        std::fprintf(stderr,
                     "serve: model image planning failed (%d)\n",
                     image_plan.exit_code);
        return kExitRuntime;
    }
    CheckedU64 bootstrap = checked_u64_add(
        image_plan.prepared_record_bytes,
        plan.tokenizer.data_size_bytes);
    if (bootstrap) {
        bootstrap = checked_u64_add(
            bootstrap.value, plan.tokenizer.config_size_bytes);
    }
    if (bootstrap) {
        bootstrap = checked_u64_add(
            bootstrap.value, plan.tokenizer.template_size_bytes);
    }
    if (!bootstrap) {
        std::fputs(
            "serve: bootstrap memory arithmetic overflow\n", stderr);
        return kExitConfigurationInvalid;
    }

    const host::HostFacts host_facts = host::read_host_facts();
    const bool graph_mode = prefill_mode == PrefillMode::Graph;
    const ServingMemoryProfile memory_profile{
        .requested_context_capacity =
            configuration.service.max_context_tokens,
        .graph_scratch_lanes =
            configuration.memory.graph_scratch_lanes,
        .composed_prefill = graph_mode,
        .physical_memory_bytes = host_facts.memory_bytes,
        .metal_working_set_bytes =
            host_facts.metal_recommended_working_set_bytes,
        .maximum_single_buffer_bytes =
            host_facts.metal_maximum_buffer_bytes,
        .unified_external_occupancy_bytes =
            configuration.memory.unified_external_occupancy_bytes,
        .metal_external_occupancy_bytes =
            configuration.memory.metal_external_occupancy_bytes,
        .os_runtime_reserve_bytes =
            configuration.memory.os_runtime_reserve_bytes,
        .prepared_image_bytes = image_plan.image_bytes,
        .bootstrap_and_tokenizer_bytes = bootstrap.value,
        .cache_budget_bytes = configuration.cache.budget_bytes,
        .graph_object_budget_bytes =
            graph_mode
                ? configuration.memory.graph_object_budget_bytes
                : 0,
    };
    const PrefillPolicy admission_geometry =
        serve_prefill_geometry_policy(plan.tokenizer.maximum_context);
    const PrefillExecutionPolicy admission_execution =
        serve_prefill_execution_policy(
            admission_geometry,
            configuration.memory.graph_scratch_lanes,
            configuration.speculative.enabled);
    const ServingCapacityResult capacity = plan_serving_capacity(
        plan, memory_profile, admission_geometry,
        admission_execution);
    if (!capacity) {
        const auto& budget = capacity.limiting;
        const std::string_view error_name =
            serving_capacity_error_name(capacity.error);
        std::fprintf(
            stderr,
            "serve: capacity admission refused: error=%.*s"
            " requested=%u maximum_admissible=%u"
            " candidate=%u metal_required=%llu metal_available=%llu"
            " unified_required=%llu unified_available=%llu"
            " maximum_buffer=%llu buffer_limit=%llu"
            " metal_deficit=%llu unified_deficit=%llu"
            " buffer_deficit=%llu cache_budget=%llu"
            " os_runtime_reserve=%llu\n",
            static_cast<int>(error_name.size()), error_name.data(),
            capacity.requested_context_capacity,
            capacity.maximum_admissible_context,
            budget.context_capacity,
            static_cast<unsigned long long>(
                budget.metal_required_bytes),
            static_cast<unsigned long long>(
                budget.metal_available_bytes),
            static_cast<unsigned long long>(
                budget.unified_required_bytes),
            static_cast<unsigned long long>(
                budget.unified_available_bytes),
            static_cast<unsigned long long>(
                budget.maximum_planned_buffer_bytes),
            static_cast<unsigned long long>(
                budget.maximum_single_buffer_bytes),
            static_cast<unsigned long long>(
                budget.metal_deficit_bytes),
            static_cast<unsigned long long>(
                budget.unified_deficit_bytes),
            static_cast<unsigned long long>(
                budget.single_buffer_deficit_bytes),
            static_cast<unsigned long long>(
                configuration.cache.budget_bytes),
            static_cast<unsigned long long>(
                configuration.memory.os_runtime_reserve_bytes));
        return kExitConfigurationInvalid;
    }
    configuration.service.max_context_tokens =
        capacity.admitted_context_capacity;
    const auto& admitted = capacity.admitted;
    std::printf(
        "serve: capacity ADMITTED=%u MODEL_MAXIMUM=%u"
        " QUALIFIED=%u maximum_without_cache=%u"
        " cache_budget=%llu graph_lanes=%u"
        " metal_required=%llu/%llu"
        " unified_required=%llu/%llu"
        " os_runtime_reserve=%llu\n",
        capacity.admitted_context_capacity,
        capacity.model_maximum_context,
        tatara::kQualifiedContextTokens,
        capacity.maximum_without_cache,
        static_cast<unsigned long long>(
            configuration.cache.budget_bytes),
        configuration.memory.graph_scratch_lanes,
        static_cast<unsigned long long>(
            admitted.metal_required_bytes),
        static_cast<unsigned long long>(
            admitted.metal_available_bytes),
        static_cast<unsigned long long>(
            admitted.unified_required_bytes),
        static_cast<unsigned long long>(
            admitted.unified_available_bytes),
        static_cast<unsigned long long>(
            configuration.memory.os_runtime_reserve_bytes));

    auto tokenizer =
        text::Tokenizer::load(plan.tokenizer, configuration.model.artifact_root);
    if (!tokenizer) {
        std::fprintf(stderr, "serve: tokenizer boot failed (%u): %s\n",
                     static_cast<unsigned>(tokenizer.error),
                     tokenizer.detail.c_str());
        return kExitRuntime;
    }
    auto chat_template = text::Qwen36ChatTemplate::load(
        plan.tokenizer, configuration.model.artifact_root);
    if (!chat_template) {
        std::fprintf(stderr, "serve: chat-template boot failed (%u): %s\n",
                     static_cast<unsigned>(chat_template.error),
                     chat_template.detail.c_str());
        return kExitRuntime;
    }

    std::atomic<Readiness> readiness{Readiness::ModelLoading};
    ServiceCounters counters;
    const auto started_at = std::chrono::steady_clock::now();

    const std::string_view prefill_mode_name = prefill_mode == PrefillMode::Graph
                                                   ? kPrefillModeGraph
                                                   : kPrefillModeSingleToken;
    std::printf("serve: prefill mode %.*s\n",
                static_cast<int>(prefill_mode_name.size()), prefill_mode_name.data());
    if (!configuration.speculative.draft_checkpoint.empty()) {
        const auto draft = tatara::draft::load_draft_checkpoint(
            configuration.speculative.draft_checkpoint);
        if (!draft) {
            std::fprintf(stderr,
                         "serve: speculative.draft_checkpoint refused by "
                         "the inventory gate: %s\n",
                         configuration.speculative.draft_checkpoint.c_str());
            return kExitConfigurationInvalid;
        }
        std::printf("serve: draft checkpoint verified (69-tensor "
                    "inventory): %s\n",
                    configuration.speculative.draft_checkpoint.c_str());
    }
    std::printf("serve: loading %s\n", configuration.model.record.c_str());
    auto harness = boot_decode(
        configuration.model.record.c_str(),
        configuration.model.artifact_root.c_str(),
        configuration.service.max_context_tokens);
    if (!harness) {
        std::fprintf(stderr, "serve: model boot failed (%d)\n", harness.exit_code);
        return kExitRuntime;
    }
    DecodeStep& step = *harness.step;
    std::printf(
        "serve: decode attention policy adaptive"
        " split-before=%u"
        " vector-at-or-after=%u"
        " rollback-a23-fused-at=%u"
        " (performance crossovers; no capacity or output limit)\n",
        step.pipelines.vector_minimum_context,
        step.pipelines.vector_minimum_context,
        step.pipelines.fused_score_value_minimum_context);
    PrefillStepResult prefill =
        prefill_mode == PrefillMode::Graph
            ? boot_graph_prefill(
                  harness, configuration.memory.graph_scratch_lanes,
                  configuration.speculative.enabled)
            : PrefillStepResult{};
    if (prefill_mode == PrefillMode::Graph && !prefill) {
        return kExitRuntime;
    }
    std::unique_ptr<engine::SpeculativeEngine> speculative;
    MetalBuffer speculative_handoff;
    if (configuration.speculative.enabled) {
        if (prefill_mode != PrefillMode::Graph) {
            std::fputs(
                "serve: speculative.enabled requires the graph prefill"
                " mode (prompt conditioning captures ride the prefill"
                " bands)\n",
                stderr);
            return kExitConfigurationInvalid;
        }
        auto speculative_result = engine::create_speculative_engine(
            *harness.device, *harness.library, *harness.queue, step,
            harness.capacity, configuration.speculative.draft_checkpoint);
        if (!speculative_result) {
            std::fprintf(stderr,
                         "serve: speculative engine construction refused"
                         " (%u)\n",
                         static_cast<unsigned>(speculative_result.error));
            return kExitConfigurationInvalid;
        }
        speculative = std::move(speculative_result.engine);
        auto handoff = create_shared_buffer(
            *harness.device,
            std::uint64_t{tatara::draft::kDraftCaptureLayers} *
                step.geometry.hidden_bytes);
        if (!handoff) {
            std::fputs("serve: speculative handoff staging allocation"
                       " failed\n",
                       stderr);
            return kExitRuntime;
        }
        speculative_handoff = std::move(*handoff.buffer);
        const std::uint64_t draft_cache_bytes =
            2ull * tatara::draft::kDraftLayers *
            (((std::uint64_t{harness.capacity} +
               tatara::draft::kDraftSinkPositions + 15) / 16) * 16) *
            tatara::draft::kDraftKeyValueHeads *
            tatara::draft::kDraftHeadDimension * 2;
        std::printf(
            "serve: speculative decoding enabled;"
            " draft cache %llu MiB at capacity %u; serial walk remains"
            " the fallback on every typed refusal\n",
            static_cast<unsigned long long>(draft_cache_bytes >> 20),
            harness.capacity);
    }
    ServePrefixCache prefix_cache;
    if (!boot_prefix_cache(
            configuration, prefill_mode, harness, prefill,
            prefix_cache)) {
        return kExitRuntime;
    }
    if (speculative && prefix_cache.enabled()) {
        std::puts(
            "serve: prefix cache bypassed under speculative decoding");
    }

    ServerHooks hooks;
    hooks.snapshot = [&]() {
        ServiceSnapshot snapshot;
        snapshot.readiness = readiness.load();
        snapshot.counters = counters;
        snapshot.gauges.queue_depth = configuration.service.queue_depth;
        snapshot.gauges.uptime_seconds =
            static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
                                           std::chrono::steady_clock::now() - started_at)
                                           .count());
        snapshot.engine_version = std::string(kVersion);
        snapshot.model_package_id = std::string(plan.id);
        return snapshot;
    };

    const CompletionRequestLimits request_limits{
        .maximum_body_bytes = kMaximumBodyBytes,
        .maximum_input_bytes = kMaximumBodyBytes,
        .maximum_prompt_tokens =
            configuration.service.max_context_tokens - 1u,
        .maximum_context_tokens = configuration.service.max_context_tokens,
        .default_maximum_tokens =
            configuration.service.default_max_output_tokens,
    };
    hooks.completions =
        [&](Route route, const std::string& body, Generation& generation) {
        const auto fail_engine =
            [&](std::string_view detail) {
                readiness.store(Readiness::Failed);
                generation.response_status = 500;
                generation.body =
                    render_error(detail, "server_error",
                                 "tatara.engine_failed");
                if (g_server != nullptr) {
                    g_server->stop();
                }
                return false;
            };
        auto parsed_request =
            parse_completion_request(route, body, plan.id, plan.tokenizer,
                                     request_limits);
        if (!parsed_request) {
            generation.response_status =
                completion_request_http_status(parsed_request.error);
            generation.body =
                render_error(parsed_request.detail, "invalid_request_error",
                             completion_request_code(parsed_request.error));
            if (generation.response_status == 413) {
                counters.requests_rejected_context += 1;
            }
            return false;
        }
        auto prepared = prepare_completion_request(
            std::move(parsed_request.request), *tokenizer.tokenizer,
            *chat_template.chat_template, request_limits);
        if (!prepared) {
            generation.response_status =
                completion_request_http_status(prepared.error);
            generation.body =
                render_error(prepared.detail, "invalid_request_error",
                             completion_request_code(prepared.error));
            if (generation.response_status == 413) {
                counters.requests_rejected_context += 1;
            }
            return false;
        }
        const PreparedCompletionRequest& request = prepared.request;
        generation.stream = request.stream;
        const std::vector<std::uint32_t>& prompt = request.prompt_tokens;
        const std::uint32_t requested = request.maximum_tokens;
        counters.requests_admitted += 1;
        counters.prompt_tokens += prompt.size();

        std::vector<std::uint32_t> produced;
        std::string completion_text;
        std::string reasoning_text;
        const bool text_response =
            request.prompt_kind != PromptKind::TokenIds;
        auto stream_decoder = text_response
                                  ? text::StreamingDecoder::create(
                                        *tokenizer.tokenizer,
                                        text::StreamingDecoderConfig{
                                            .skip_special = true,
                                            .suppress_thinking = false,
                                            .maximum_output_bytes =
                                                std::numeric_limits<
                                                    std::size_t>::max(),
                                            .stop_token_ids =
                                                request.stop_token_ids,
                                            .thinking_start_id =
                                                plan.tokenizer.thinking_start_id,
                                            .thinking_end_id =
                                                plan.tokenizer.thinking_end_id,
                                        })
                                  : text::StreamingDecoderCreateResult{};
        if (text_response && !stream_decoder) {
            return false;
        }
        bool in_reasoning =
            route == Route::ChatCompletions && request.enable_thinking;
        bool stopped = false;
        bool spec_request_active = static_cast<bool>(speculative);
        const bool cache_active =
            prefix_cache.enabled() && !speculative;
        const std::uint32_t prefill_end =
            static_cast<std::uint32_t>(prompt.size()) - 1u;
        const runtime::SlotHandle cache_slot{
            .slot_index = 0,
            .slot_generation = 1,
        };
        std::optional<runtime::RequestHandle> cache_request;
        if (cache_active) {
            cache_request = next_cache_request(prefix_cache);
            if (!cache_request) {
                std::fputs(
                    "serve: prefix-cache request generation exhausted\n",
                    stderr);
                return fail_engine(
                    "prefix-cache request generation exhausted");
            }
        }
        std::uint32_t cache_hit_position = 0;
        if (cache_active) {
            const std::span<const std::uint32_t> prepared_prefix(
                prompt.data(), prefill_end);
            PrefixCacheLookupResult lookup =
                prefix_cache.cache->lookup_longest(
                    prefix_cache.cache->domain(),
                    prepared_prefix, prefill_end,
                    {.owner_index = cache_request->owner_index,
                     .owner_generation =
                         cache_request->owner_generation},
                    {.slot_index = cache_slot.slot_index,
                     .slot_generation =
                         cache_slot.slot_generation});
            if (lookup.error != PrefixCacheError::None) {
                counters.prefix_cache_lookup_failures += 1;
                std::fprintf(
                    stderr,
                    "serve: prefix-cache lookup failed error=%u\n",
                    static_cast<unsigned>(lookup.error));
                return fail_engine("prefix-cache lookup failed");
            }
            if (lookup) {
                cache_hit_position =
                    lookup.lease->position_tokens;
                if (!execute_prefix_restore(
                        prefix_cache, harness, *cache_request,
                        cache_slot, *lookup.lease)) {
                    counters.prefix_cache_restore_failures += 1;
                    std::fputs(
                        "serve: prefix-cache restore failed\n",
                        stderr);
                    return fail_engine("prefix-cache restore failed");
                }
                counters.prefix_cache_hits += 1;
            } else {
                counters.prefix_cache_misses += 1;
            }
        }
        if (cache_hit_position == 0 &&
            !execute_state_slot_reset(harness)) {
            std::fputs(
                "serve: typed state-slot reset failed\n", stderr);
            return fail_engine("typed state-slot reset failed");
        }
        if (spec_request_active) {
            speculative->reset_request();
        }

        std::optional<PrefixCacheReservation> cache_reservation;
        std::uint32_t cache_publish_position = 0;
        if (cache_active) {
            cache_publish_position = largest_cache_boundary(
                prefill_end, cache_hit_position, prefix_cache);
            if (cache_publish_position != 0) {
                const PrefillGeometryResult cache_geometry =
                    make_prefill_geometry(
                        plan,
                        serve_prefill_geometry_policy(
                            harness.capacity));
                const PrefixStateLayoutResult cache_layout =
                    cache_geometry
                        ? make_prefix_state_layout(
                              step, cache_geometry.geometry,
                              cache_publish_position)
                        : PrefixStateLayoutResult{};
                if (!cache_geometry || !cache_layout) {
                    std::fputs(
                        "serve: prefix-cache snapshot layout failed\n",
                        stderr);
                    counters.prefix_cache_snapshot_failures += 1;
                    return fail_engine(
                        "prefix-cache snapshot layout failed");
                }
                const std::span<const std::uint32_t>
                    publish_tokens(
                        prompt.data(), cache_publish_position);
                PrefixCacheReserveResult reserved =
                    prefix_cache.cache->reserve_snapshot(
                        prefix_cache.cache->domain(),
                        {
                            .token_digest =
                                digest_prefix_tokens(
                                    publish_tokens),
                            .tokens = publish_tokens,
                        },
                        cache_layout.layout.total_bytes);
                if (reserved) {
                    cache_reservation =
                        *reserved.reservation;
                } else if (!nonfatal_cache_reservation_error(
                               reserved.error)) {
                    std::fprintf(
                        stderr,
                        "serve: prefix-cache reservation failed"
                        " error=%u\n",
                        static_cast<unsigned>(reserved.error));
                    counters.prefix_cache_snapshot_failures += 1;
                    return fail_engine(
                        "prefix-cache reservation invariant failed");
                } else {
                    counters.prefix_cache_reservation_skips += 1;
                    const std::string_view skip =
                        prefix_cache_error_name(reserved.error);
                    std::fprintf(
                        stderr,
                        "serve: prefix-cache snapshot skipped"
                        " outcome=%.*s position=%u state_bytes=%llu\n",
                        static_cast<int>(skip.size()), skip.data(),
                        cache_publish_position,
                        static_cast<unsigned long long>(
                            cache_layout.layout.total_bytes));
                    cache_publish_position = 0;
                }
            }
        }
        SnapshotPublicationGuard snapshot_publication;
        if (cache_reservation) {
            snapshot_publication.cache_owner =
                &*prefix_cache.cache;
            snapshot_publication.reservation =
                cache_reservation;
        }
        std::uint32_t context = cache_hit_position;
        // The final prompt token already yields the first generated token, so
        // the walk is one shorter than prompt + requested or it returns an
        // extra token the caller did not ask for.
        const std::uint32_t steps = static_cast<std::uint32_t>(prompt.size()) + requested - 1u;
        std::uint32_t start_index = cache_hit_position;
        if (prefill_mode == PrefillMode::Graph &&
            prefill_end > cache_hit_position) {
            PrefillStep& graph_prefill = *prefill.step;
            const std::span<const std::uint32_t> prefill_ids(
                prompt.data(), prompt.size() - 1u);
            std::uint32_t band_context =
                cache_hit_position;
            std::uint32_t band_offset =
                cache_hit_position;
            std::uint32_t remaining_rows =
                prefill_end - cache_hit_position;
            while (remaining_rows != 0) {
                std::uint32_t plannable_rows =
                    remaining_rows;
                if (cache_reservation &&
                    !snapshot_publication.transaction &&
                    band_context < cache_publish_position) {
                    plannable_rows =
                        std::min(
                            plannable_rows,
                            cache_publish_position -
                                band_context);
                }
                // Speculative serving plans one lane per band so a band
                // never exceeds the conditioning-capture row capacity.
                const PrefillBandPlan band = plan_next_prefill_band(
                    graph_prefill.policy.geometry, band_context,
                    plannable_rows,
                    speculative
                        ? 1u
                        : graph_prefill.policy.command_graph_chunk_count);
                if (!band) {
                    std::fprintf(
                        stderr,
                        "serve: prefill band planning failed:"
                        " error=%u context=%u remaining=%u\n",
                        static_cast<unsigned>(band.error),
                        band_context, remaining_rows);
                    poison_prefill(graph_prefill);
                    return fail_engine(
                        "prefill band planning failed");
                }
                const std::span<const std::uint32_t> band_ids(
                    prefill_ids.data() + band_offset,
                    band.row_count);
                const PrefillProgressResult begun =
                    begin_prefill_progress(
                        graph_prefill, *harness.step,
                        band.context_base, band.context_base,
                        band_ids);
                if (!begun) {
                    std::fprintf(
                        stderr,
                        "serve: begin_prefill_progress failed:"
                        " progress_error=%u encode_error=%u"
                        " context=%u rows=%u\n",
                        static_cast<unsigned>(begun.error),
                        static_cast<unsigned>(begun.encode_error),
                        band.context_base, band.row_count);
                    poison_prefill(graph_prefill);
                    return fail_engine(
                        "prefill progress initialization failed");
                }
                const PrefillCommandGraphResult graph_prepared =
                    prepare_prefill_command_graph(
                        *harness.device, graph_prefill,
                        *harness.step);
                if (!graph_prepared) {
                    std::fprintf(
                        stderr,
                        "serve: prepare_prefill_command_graph failed:"
                        " graph_error=%u stage=%u plan_error=%u"
                        " command_error=%u context=%u rows=%u\n",
                        static_cast<unsigned>(graph_prepared.error),
                        static_cast<unsigned>(graph_prepared.stage),
                        static_cast<unsigned>(graph_prepared.plan_error),
                        static_cast<unsigned>(
                            graph_prepared.command_error),
                        band.context_base, band.row_count);
                    poison_prefill(graph_prefill);
                    return fail_engine(
                        "prefill command graph preparation failed");
                }
                auto graph_command_buffer =
                    create_command_buffer(*harness.queue);
                if (!graph_command_buffer) {
                    poison_prefill(graph_prefill);
                    return fail_engine(
                        "prefill command buffer creation failed");
                }
                auto graph_pass = begin_compute_pass(
                    std::move(
                        *graph_command_buffer.command_buffer));
                if (!graph_pass) {
                    poison_prefill(graph_prefill);
                    return fail_engine(
                        "prefill compute pass creation failed");
                }
                const PrefillCommandGraphResult encoded =
                    encode_prefill_command_graph(
                        graph_prefill, *harness.step,
                        *graph_pass.compute_pass);
                if (!encoded) {
                    std::fprintf(
                        stderr,
                        "serve: encode_prefill_command_graph failed:"
                        " graph_error=%u stage=%u command_error=%u"
                        " context=%u rows=%u\n",
                        static_cast<unsigned>(encoded.error),
                        static_cast<unsigned>(encoded.stage),
                        static_cast<unsigned>(
                            encoded.command_error),
                        band.context_base, band.row_count);
                    poison_prefill(graph_prefill);
                    return fail_engine(
                        "prefill command graph encoding failed");
                }
                auto graph_ended = end_compute_pass(
                    std::move(*graph_pass.compute_pass));
                if (!graph_ended) {
                    poison_prefill(graph_prefill);
                    return fail_engine(
                        "prefill compute pass completion failed");
                }
                auto graph_pending = commit(
                    std::move(*graph_ended.command_buffer));
                if (!graph_pending) {
                    poison_prefill(graph_prefill);
                    return fail_engine(
                        "prefill command graph commit failed");
                }
                if (auto graph_execution =
                        wait_until_completed(std::move(
                            *graph_pending.pending_execution));
                    !graph_execution) {
                    const auto description =
                        graph_execution.failure_description.view();
                    std::fprintf(
                        stderr,
                        "serve: prefill command graph execution failed:"
                        " context=%u rows=%u %.*s\n",
                        band.context_base, band.row_count,
                        static_cast<int>(description.size()),
                        description.data());
                    poison_prefill(graph_prefill);
                    return fail_engine(
                        "prefill command graph execution failed");
                }
                const PrefillProgressResult committed =
                    commit_prefill_command_graph(
                        graph_prefill, *harness.step);
                if (!committed ||
                    committed.next_context != band.next_context) {
                    std::fprintf(
                        stderr,
                        "serve: commit_prefill_command_graph failed:"
                        " progress_error=%u failed_unit_offset=%u"
                        " routed_up_status=%u next_context=%u"
                        " expected_context=%u\n",
                        static_cast<unsigned>(committed.error),
                        committed.failed_unit_offset,
                        static_cast<std::uint32_t>(
                            committed.routed_up_status),
                        committed.next_context,
                        band.next_context);
                    poison_prefill(graph_prefill);
                    return fail_engine(
                        "prefill command graph state commit failed");
                }
                band_context = band.next_context;
                band_offset += band.row_count;
                remaining_rows -= band.row_count;
                if (spec_request_active) {
                    const engine::SpeculativeError observed =
                        speculative->observe_prompt_band(
                            graph_prefill.capture_buffer, band.row_count,
                            band.context_base);
                    if (observed != engine::SpeculativeError::None) {
                        std::fprintf(
                            stderr,
                            "serve: speculative prompt conditioning"
                            " refused (%u) at band context=%u rows=%u;"
                            " serial walk for this request\n",
                            static_cast<unsigned>(observed),
                            band.context_base, band.row_count);
                        spec_request_active = false;
                    }
                }
                if (cache_reservation &&
                    !snapshot_publication.transaction &&
                    band_context == cache_publish_position) {
                    if (!execute_prefix_snapshot(
                            prefix_cache, harness,
                            *cache_request, cache_slot,
                            *cache_reservation,
                            snapshot_publication.transaction)) {
                        counters.prefix_cache_snapshot_failures += 1;
                        std::fputs(
                            "serve: prefix-cache snapshot failed\n",
                            stderr);
                        return fail_engine(
                            "prefix-cache snapshot failed");
                    }
                }
            }
            if (band_context != prefill_ids.size()) {
                std::fprintf(
                    stderr,
                    "serve: prefill band conservation failed:"
                    " committed=%u expected=%zu\n",
                    band_context, prefill_ids.size());
                poison_prefill(graph_prefill);
                return fail_engine(
                    "prefill band conservation failed");
            }
            start_index = prefill_end;
            context = start_index;
        } else if (prefill_mode == PrefillMode::Graph) {
            start_index = prefill_end;
            context = prefill_end;
        }
        // Shared per-token tail: stop handling, accounting, streaming.
        // The serial walk and the speculative commit path both emit
        // through here so the two can never drift.
        enum class EmitOutcome : std::uint8_t {
            Continue,
            Stopped,
            CancelledDone,
            StreamFail,
        };
        const auto emit_token = [&](std::uint32_t token) -> EmitOutcome {
            if (std::find(request.stop_token_ids.begin(),
                          request.stop_token_ids.end(),
                          token) != request.stop_token_ids.end()) {
                if (text_response) {
                    const auto stop = stream_decoder.decoder->push(token);
                    if (!stop || !stop.stopped) {
                        return EmitOutcome::StreamFail;
                    }
                }
                stopped = true;
                return EmitOutcome::Stopped;
            }
            produced.push_back(token);
            counters.generated_tokens += 1;
            if (text_response) {
                if (route == Route::ChatCompletions &&
                    token == plan.tokenizer.thinking_start_id) {
                    in_reasoning = true;
                    return EmitOutcome::Continue;
                }
                if (route == Route::ChatCompletions &&
                    token == plan.tokenizer.thinking_end_id) {
                    in_reasoning = false;
                    return EmitOutcome::Continue;
                }
                auto chunk = stream_decoder.decoder->push(token);
                if (!chunk) {
                    return EmitOutcome::StreamFail;
                }
                if (!chunk.bytes.empty()) {
                    std::string& destination =
                        in_reasoning ? reasoning_text : completion_text;
                    destination += chunk.bytes;
                    std::ostringstream event;
                    if (route == Route::ChatCompletions) {
                        event << "{\"choices\":[{\"index\":0,\"delta\":{\""
                              << (in_reasoning ? "reasoning_content"
                                               : "content")
                              << "\":\"" << json_escape(chunk.bytes)
                              << "\"}}]}";
                    } else {
                        event << "{\"choices\":[{\"index\":0,\"text\":\""
                              << json_escape(chunk.bytes) << "\"}]}";
                    }
                    if (!generation.emit(event.str())) {
                        counters.requests_cancelled += 1;
                        if (!snapshot_publication.resolve(
                                engine::PrefixCacheTerminalDisposition::Cancelled)) {
                            counters.prefix_cache_snapshot_failures += 1;
                            fail_engine(
                                "prefix-cache cancellation resolution failed");
                            return EmitOutcome::StreamFail;
                        }
                        return EmitOutcome::CancelledDone;
                    }
                }
            } else {
                std::ostringstream event;
                event << "{\"choices\":[{\"delta\":{\"token\":" << token
                      << "}}]}";
                if (!generation.emit(event.str())) {
                    counters.requests_cancelled += 1;
                    if (!snapshot_publication.resolve(
                            engine::PrefixCacheTerminalDisposition::Cancelled)) {
                        counters.prefix_cache_snapshot_failures += 1;
                        fail_engine(
                            "prefix-cache cancellation resolution failed");
                        return EmitOutcome::StreamFail;
                    }
                    return EmitOutcome::CancelledDone;
                }
            }
            return EmitOutcome::Continue;
        };
        const bool spec_generation = spec_request_active;
        bool spec_serial_handoff = false;
        std::uint32_t spec_staged = 0;
        if (spec_generation) {
            std::memcpy(step.token_id.contents(), &prompt[prefill_end], 4);
            auto command_buffer = create_command_buffer(*harness.queue);
            if (!command_buffer) {
                return fail_engine("decode command buffer creation failed");
            }
            auto pass = begin_compute_pass(
                std::move(*command_buffer.command_buffer));
            if (!pass) {
                return fail_engine("decode compute pass creation failed");
            }
            if (encode_token(step, *pass.compute_pass, context) !=
                MetalCommandError::None) {
                return fail_engine("decode command encoding failed");
            }
            auto ended = end_compute_pass(std::move(*pass.compute_pass));
            if (!ended) {
                return fail_engine("decode compute pass completion failed");
            }
            auto pending = commit(std::move(*ended.command_buffer));
            if (!pending) {
                return fail_engine("decode command commit failed");
            }
            if (auto execution = wait_until_completed(
                    std::move(*pending.pending_execution));
                !execution) {
                return fail_engine("decode command execution failed");
            }
            const std::uint32_t handoff_token =
                *static_cast<std::uint32_t*>(step.token_id.contents());
            if (handoff_token >= plan.dimensions.vocabulary) {
                return fail_engine(
                    "decode produced an out-of-vocabulary token");
            }
            {
                const std::uint64_t hidden_bytes =
                    step.geometry.hidden_bytes;
                const std::uint8_t* stream =
                    static_cast<const std::uint8_t*>(
                        step.layer_stream.contents());
                std::uint8_t* destination = static_cast<std::uint8_t*>(
                    speculative_handoff.contents());
                for (std::uint32_t slot = 0;
                     slot < tatara::draft::kDraftCaptureLayers; ++slot) {
                    std::memcpy(
                        destination + std::uint64_t{slot} * hidden_bytes,
                        stream +
                            std::uint64_t{
                                tatara::draft::kDraftCaptureAfterTargetLayer
                                    [slot]} *
                                hidden_bytes,
                        hidden_bytes);
                }
            }
            advance_decode_state(step);
            ++context;
            spec_staged = handoff_token;
            const engine::SpeculativeError handoff_observed =
                speculative->observe_handoff_row(speculative_handoff,
                                                 prefill_end);
            if (handoff_observed != engine::SpeculativeError::None) {
                std::fprintf(stderr,
                             "serve: speculative handoff conditioning"
                             " refused (%u); serial continuation\n",
                             static_cast<unsigned>(handoff_observed));
                spec_serial_handoff = true;
            } else {
                speculative->extend_history(prompt.data(), prompt.size());
                std::uint32_t spec_cycles = 0;
                std::uint32_t spec_copy_cycles = 0;
                std::uint64_t spec_committed_total = 0;
                while (true) {
                    if (generation.cancelled()) {
                        counters.requests_cancelled += 1;
                        if (!snapshot_publication.resolve(
                                engine::PrefixCacheTerminalDisposition::Cancelled)) {
                            counters.prefix_cache_snapshot_failures += 1;
                            return fail_engine(
                                "prefix-cache cancellation resolution failed");
                        }
                        return true;
                    }
                    engine::SpeculativeStepResult cycle =
                        speculative->step(spec_staged, context);
                    if (!cycle) {
                        std::fprintf(
                            stderr,
                            "serve: speculative step refused (%u) at"
                            " context=%u; serial continuation\n",
                            static_cast<unsigned>(cycle.error), context);
                        spec_serial_handoff = true;
                        break;
                    }
                    spec_cycles += 1;
                    spec_copy_cycles += cycle.used_copy ? 1u : 0u;
                    spec_committed_total += cycle.committed.size();
                    bool request_complete = false;
                    for (const std::uint32_t token : cycle.committed) {
                        const EmitOutcome outcome = emit_token(token);
                        if (outcome == EmitOutcome::Stopped) {
                            request_complete = true;
                            break;
                        }
                        if (outcome == EmitOutcome::CancelledDone) {
                            return true;
                        }
                        if (outcome == EmitOutcome::StreamFail) {
                            return false;
                        }
                        if (produced.size() >= requested) {
                            request_complete = true;
                            break;
                        }
                    }
                    context += static_cast<std::uint32_t>(
                        cycle.committed.size());
                    spec_staged = cycle.next_staged;
                    if (request_complete) {
                        break;
                    }
                }
                if (spec_cycles != 0) {
                    std::fprintf(
                        stderr,
                        "serve: speculative request cycles=%u copied=%u"
                        " committed=%llu tau=%.3f emitted=%zu\n",
                        spec_cycles, spec_copy_cycles,
                        static_cast<unsigned long long>(
                            spec_committed_total),
                        double(spec_committed_total) / double(spec_cycles),
                        produced.size());
                }
            }
            // Serial continuation after a mid-request refusal: spec_staged
            // is produced but not yet emitted.
            if (spec_serial_handoff && !stopped &&
                produced.size() < requested) {
                std::uint32_t current = spec_staged;
                while (true) {
                    if (generation.cancelled()) {
                        counters.requests_cancelled += 1;
                        if (!snapshot_publication.resolve(
                                engine::PrefixCacheTerminalDisposition::Cancelled)) {
                            counters.prefix_cache_snapshot_failures += 1;
                            return fail_engine(
                                "prefix-cache cancellation resolution failed");
                        }
                        return true;
                    }
                    const EmitOutcome outcome = emit_token(current);
                    if (outcome == EmitOutcome::Stopped) {
                        break;
                    }
                    if (outcome == EmitOutcome::CancelledDone) {
                        return true;
                    }
                    if (outcome == EmitOutcome::StreamFail) {
                        return false;
                    }
                    if (produced.size() >= requested) {
                        break;
                    }
                    std::memcpy(step.token_id.contents(), &current, 4);
                    auto serial_command_buffer =
                        create_command_buffer(*harness.queue);
                    if (!serial_command_buffer) {
                        return fail_engine(
                            "decode command buffer creation failed");
                    }
                    auto serial_pass = begin_compute_pass(std::move(
                        *serial_command_buffer.command_buffer));
                    if (!serial_pass) {
                        return fail_engine(
                            "decode compute pass creation failed");
                    }
                    if (encode_token(step, *serial_pass.compute_pass,
                                     context) != MetalCommandError::None) {
                        return fail_engine(
                            "decode command encoding failed");
                    }
                    auto serial_ended = end_compute_pass(
                        std::move(*serial_pass.compute_pass));
                    if (!serial_ended) {
                        return fail_engine(
                            "decode compute pass completion failed");
                    }
                    auto serial_pending =
                        commit(std::move(*serial_ended.command_buffer));
                    if (!serial_pending) {
                        return fail_engine("decode command commit failed");
                    }
                    if (auto serial_execution = wait_until_completed(
                            std::move(*serial_pending.pending_execution));
                        !serial_execution) {
                        return fail_engine(
                            "decode command execution failed");
                    }
                    const std::uint32_t serial_token =
                        *static_cast<std::uint32_t*>(
                            step.token_id.contents());
                    if (serial_token >= plan.dimensions.vocabulary) {
                        return fail_engine(
                            "decode produced an out-of-vocabulary token");
                    }
                    advance_decode_state(step);
                    ++context;
                    current = serial_token;
                }
            }
        }
        if (!spec_generation)
        for (std::uint32_t index = start_index; index < steps; ++index) {
            if (generation.cancelled()) {
                counters.requests_cancelled += 1;
                if (!snapshot_publication.resolve(
                        engine::PrefixCacheTerminalDisposition::Cancelled)) {
                    counters.prefix_cache_snapshot_failures += 1;
                    return fail_engine(
                        "prefix-cache cancellation resolution failed");
                }
                return true;
            }
            if (index < prompt.size()) {
                std::memcpy(step.token_id.contents(), &prompt[index], 4);
            }
            auto command_buffer = create_command_buffer(*harness.queue);
            if (!command_buffer) {
                return fail_engine(
                    "decode command buffer creation failed");
            }
            auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
            if (!pass) {
                return fail_engine(
                    "decode compute pass creation failed");
            }
            if (encode_token(step, *pass.compute_pass, context) != MetalCommandError::None) {
                return fail_engine("decode command encoding failed");
            }
            auto ended = end_compute_pass(std::move(*pass.compute_pass));
            if (!ended) {
                return fail_engine(
                    "decode compute pass completion failed");
            }
            auto pending = commit(std::move(*ended.command_buffer));
            if (!pending) {
                return fail_engine("decode command commit failed");
            }
            if (auto execution = wait_until_completed(std::move(*pending.pending_execution));
                !execution) {
                return fail_engine("decode command execution failed");
            }
            const std::uint32_t token = *static_cast<std::uint32_t*>(step.token_id.contents());
            if (token >= plan.dimensions.vocabulary) {
                return fail_engine(
                    "decode produced an out-of-vocabulary token");
            }
            advance_decode_state(step);
            ++context;
            if (prefill_mode == PrefillMode::SingleToken &&
                cache_reservation &&
                !snapshot_publication.transaction &&
                context == cache_publish_position) {
                if (!execute_prefix_snapshot(
                        prefix_cache, harness, *cache_request,
                        cache_slot, *cache_reservation,
                        snapshot_publication.transaction)) {
                    counters.prefix_cache_snapshot_failures += 1;
                    std::fputs(
                        "serve: prefix-cache snapshot failed\n",
                        stderr);
                    return fail_engine(
                        "prefix-cache snapshot failed");
                }
            }
            if (index + 1 >= prompt.size()) {
                const EmitOutcome outcome = emit_token(token);
                if (outcome == EmitOutcome::Stopped) {
                    break;
                }
                if (outcome == EmitOutcome::CancelledDone) {
                    return true;
                }
                if (outcome == EmitOutcome::StreamFail) {
                    return false;
                }
            }
        }
        if (text_response && !stopped) {
            if (auto finished = stream_decoder.decoder->finish(); !finished) {
                return false;
            }
        }
        const engine::PrefixCacheTerminalDisposition
            cache_terminal =
                g_draining.load(std::memory_order_relaxed)
                    ? engine::PrefixCacheTerminalDisposition::
                          AdministrativeDrain
                    : stopped
                          ? engine::PrefixCacheTerminalDisposition::
                                SuccessfulStopToken
                          : engine::PrefixCacheTerminalDisposition::
                                SuccessfulMaximumOutput;
        if (!snapshot_publication.resolve(cache_terminal)) {
            counters.prefix_cache_snapshot_failures += 1;
            std::fputs(
                "serve: prefix-cache terminal publication failed\n",
                stderr);
            return fail_engine(
                "prefix-cache terminal publication failed");
        }
        if (snapshot_publication.published) {
            counters.prefix_cache_publications += 1;
        }

        std::ostringstream out;
        if (request.prompt_kind == PromptKind::TokenIds) {
            out << "{\"object\":\"text_completion\",\"choices\":[{\"tokens\":[";
            for (std::size_t index = 0; index < produced.size(); ++index) {
                out << produced[index]
                    << (index + 1 == produced.size() ? "" : ",");
            }
            out << "]}]";
        } else if (route == Route::ChatCompletions) {
            out << "{\"object\":\"chat.completion\",\"choices\":[{\"index\":0,"
                   "\"message\":{\"role\":\"assistant\",\"reasoning_content\":\""
                << json_escape(reasoning_text) << "\",\"content\":\""
                << json_escape(completion_text) << "\"}}]";
        } else {
            out << "{\"object\":\"text_completion\",\"choices\":[{\"index\":0,"
                   "\"text\":\""
                << json_escape(completion_text) << "\"}]";
        }
        out << ",\"usage\":{\"prompt_tokens\":" << prompt.size()
            << ",\"completion_tokens\":" << produced.size() << "}}";
        generation.body = out.str();
        counters.requests_completed += 1;
        return true;
    };

    Server server(configuration, std::move(hooks));
    if (!server.start()) {
        std::fprintf(stderr, "serve: cannot bind %s:%u\n", configuration.service.bind.c_str(),
                     configuration.service.port);
        return kExitRuntime;
    }
    g_server = &server;
    g_draining.store(false, std::memory_order_relaxed);
    std::signal(SIGINT, request_drain);
    std::signal(SIGTERM, request_drain);

    // Boot gates for this increment: the model booted, the decode step
    // constructed, and — in graph mode — the prefill step, pipelines and
    // geometry all constructed (boot_graph_prefill already exited typed on
    // any failure before this point). The full product boot battery joins at
    // serving composition; until then these are the gates that exist, and
    // readiness reports exactly their outcome.
    readiness.store(Readiness::Ready);
    std::printf("serve: ready on %s:%u after %.2f s\n",
                configuration.service.bind.c_str(), server.port(), harness.load_seconds);
    std::fflush(stdout);

    server.run();
    g_server = nullptr;
    std::printf("serve: drained\n");
    return readiness.load() == Readiness::Failed ? kExitInterrupted : kExitOk;
}

} // namespace tatara::tools
