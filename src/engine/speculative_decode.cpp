
#include "tatara/engine/speculative_decode.h"

#include "tatara/draft/dflash_checkpoint.h"
#include "tatara/generated/kernel_library.h"
#include "tatara/generated/model_plan.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace tatara::engine {
namespace {

using namespace tatara::backend::metal;
using namespace tatara::runtime;
using namespace tatara::draft;
using tatara::runtime::DecodeStep;

constexpr std::uint32_t kBlock = 16;
constexpr std::uint32_t kDenseThreads = 128;
constexpr std::uint32_t kDenseRowsPerGroup = kDenseThreads / 32;
constexpr std::uint32_t kGemmColumnsPerGroup = (kDenseThreads / 32) * 16;
constexpr std::uint32_t kChunkRows = 16;
constexpr std::uint32_t kUnitBatch = 8;

struct CaptureSegment {
    std::uint32_t units;
    int capture_slot;
};
constexpr CaptureSegment kCaptureSegments[9] = {
    {2, 0}, {5, 1}, {5, 2}, {5, 3}, {6, 4}, {5, 5}, {5, 6}, {5, 7}, {2, -1},
};

#include "speculative_decode_extracted.inc"

} // namespace

namespace detail {
struct SpeculativeState {
    const MetalDevice* device{nullptr};
    const MetalLibrary* library{nullptr};
    const MetalCommandQueue* queue{nullptr};
    DecodeStep* decode{nullptr};
    std::uint32_t context_capacity{0};

    DraftCheckpointLoad draft_load;
    DraftEngine draft;
    std::optional<PrefillStep> verify_step_owner;
    PrefillStep* verify{nullptr};
    MetalComputePipeline n1_head;
    MetalComputePipeline tape_replay;

    MetalBuffer drafted_ids;
    MetalBuffer block_id_slots;
    MetalBuffer verify_out_ids;
    MetalBuffer draft_logits;
    MetalBuffer cycle_features;
    MetalBuffer normed;

    std::vector<std::uint32_t> history;
    bool copyspec_enabled{true};
    std::uint32_t conditioned{0};
};
} // namespace detail
using detail::SpeculativeState;

namespace {

SpeculativeError run_verify_band(SpeculativeState& state,
                                 std::uint32_t context_base,
                                 std::span<const std::uint32_t> ids) {
    PrefillStep& prefill = *state.verify;
    DecodeStep& decode = *state.decode;
    const auto begun = begin_prefill_progress(prefill, decode, context_base,
                                              context_base, ids);
    if (!begun) {
        return SpeculativeError::CommandFailed;
    }
    const std::uint64_t hidden_bytes = decode.geometry.hidden_bytes;
    for (const CaptureSegment& segment : kCaptureSegments) {
        auto command_buffer = create_command_buffer(*state.queue);
        if (!command_buffer) {
            return SpeculativeError::CommandFailed;
        }
        auto pass =
            begin_compute_pass(std::move(*command_buffer.command_buffer));
        if (!pass) {
            return SpeculativeError::CommandFailed;
        }
        const auto encoded = encode_prefill_units(
            prefill, decode, decode.state, *pass.compute_pass,
            segment.units);
        if (!encoded) {
            return SpeculativeError::CommandFailed;
        }
        auto ended = end_compute_pass(std::move(*pass.compute_pass));
        if (!ended) {
            return SpeculativeError::CommandFailed;
        }
        MetalPendingExecutionResult pending;
        if (segment.capture_slot >= 0) {
            auto blit = begin_blit_pass(std::move(*ended.command_buffer));
            if (!blit) {
                return SpeculativeError::CommandFailed;
            }
            for (std::size_t row = 0; row < ids.size(); ++row) {
                if (copy_buffer(*blit.blit_pass, prefill.hidden_slab,
                                std::uint64_t{row} * hidden_bytes,
                                state.cycle_features,
                                (std::uint64_t{row} * 8 +
                                 std::uint64_t(segment.capture_slot)) *
                                    hidden_bytes,
                                hidden_bytes) != MetalCommandError::None) {
                    return SpeculativeError::CommandFailed;
                }
            }
            auto blit_ended = end_blit_pass(std::move(*blit.blit_pass));
            if (!blit_ended) {
                return SpeculativeError::CommandFailed;
            }
            pending = commit(std::move(*blit_ended.command_buffer));
        } else {
            pending = commit(std::move(*ended.command_buffer));
        }
        if (!pending ||
            !wait_until_completed(std::move(*pending.pending_execution))) {
            return SpeculativeError::CommandFailed;
        }
        const auto committed_units = commit_prefill_units(prefill, decode);
        if (!committed_units) {
            return SpeculativeError::CommandFailed;
        }
    }
    if (prefill.progress.state != PrefillProgressState::Complete) {
        return SpeculativeError::CommandFailed;
    }
    return SpeculativeError::None;
}

SpeculativeError band_head(SpeculativeState& state) {
    DecodeStep& step = *state.decode;
    const std::uint32_t rows = kBlock;
    auto command_buffer = create_command_buffer(*state.queue);
    if (!command_buffer) {
        return SpeculativeError::CommandFailed;
    }
    auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
    if (!pass) {
        return SpeculativeError::CommandFailed;
    }
    MetalComputePass& compute = *pass.compute_pass;
    bool ok = true;
    const auto check = [&](MetalCommandError error) {
        ok = ok && error == MetalCommandError::None;
    };
    const std::uint64_t hidden_bytes = step.geometry.hidden_bytes;
    for (std::uint32_t row = 0; row < rows; ++row) {
        check(set_compute_pipeline(compute, step.pipelines.rms));
        check(set_buffer(compute, state.verify->hidden_slab,
                         std::uint64_t{row} * hidden_bytes, 0));
        check(set_buffer(compute, *step.image,
                         step.tensor_offsets[step.bindings.final_norm], 1));
        check(set_buffer(compute, state.normed,
                         std::uint64_t{row} * hidden_bytes, 2));
        check(dispatch_threadgroups(
            compute,
            MetalSize{.width = step.geometry.dispatch.rms.groups,
                      .height = 1, .depth = 1},
            MetalSize{.width = step.geometry.dispatch.rms.threads,
                      .height = 1, .depth = 1}));
    }
    check(memory_barrier(compute));
    {
        const std::uint32_t vocabulary = step.geometry.dispatch.vocabulary_rows;
        const std::uint32_t reduction = kDraftHidden;
        const std::uint64_t activation_stride = kDraftHidden;
        const std::uint64_t packed_stride = kDraftHidden / 8;
        const std::uint64_t parameter_stride = kDraftHidden / 64;
        const std::uint64_t output_stride = vocabulary;
        check(set_compute_pipeline(compute, state.n1_head));
        check(set_buffer(compute, state.normed, 0, 0));
        check(set_buffer(compute, *step.image,
                         step.tensor_offsets[step.bindings.head.weight], 1));
        check(set_buffer(compute, *step.image,
                         step.tensor_offsets[step.bindings.head.scales], 2));
        check(set_buffer(compute, *step.image,
                         step.tensor_offsets[step.bindings.head.biases], 3));
        check(set_buffer(compute, state.draft_logits, 0, 4));
        check(set_bytes(compute, &rows, 4, 5));
        check(set_bytes(compute, &vocabulary, 4, 6));
        check(set_bytes(compute, &reduction, 4, 7));
        check(set_bytes(compute, &activation_stride, 8, 8));
        check(set_bytes(compute, &packed_stride, 8, 9));
        check(set_bytes(compute, &parameter_stride, 8, 10));
        check(set_bytes(compute, &output_stride, 8, 11));
        check(dispatch_threadgroups(
            compute,
            MetalSize{.width = (vocabulary + 31) / 32,
                      .height = (rows + 31) / 32, .depth = 1},
            MetalSize{.width = 128, .height = 1, .depth = 1}));
        check(memory_barrier(compute));
    }
    for (std::uint32_t row = 0; row < rows; ++row) {
        check(set_compute_pipeline(compute, step.pipelines.argmax_stage1));
        check(set_buffer(compute, state.draft_logits,
                         std::uint64_t{row} *
                             step.geometry.dispatch.vocabulary_rows * 2, 0));
        check(set_bytes(compute, &step.geometry.dispatch.vocabulary_rows, 4, 1));
        check(set_buffer(compute, step.argmax_values, 0, 2));
        check(set_buffer(compute, step.argmax_indices, 0, 3));
        check(dispatch_threadgroups(
            compute,
            MetalSize{.width = step.geometry.dispatch.argmax_stage1.groups,
                      .height = 1, .depth = 1},
            MetalSize{.width = step.geometry.dispatch.argmax_stage1.threads,
                      .height = 1, .depth = 1}));
        check(memory_barrier(compute));
        check(set_compute_pipeline(compute, step.pipelines.argmax_stage2));
        check(set_buffer(compute, step.argmax_values, 0, 0));
        check(set_buffer(compute, step.argmax_indices, 0, 1));
        check(set_buffer(compute, state.verify_out_ids,
                         std::uint64_t{row} * 4, 2));
        check(dispatch_threadgroups(
            compute, MetalSize{.width = 1, .height = 1, .depth = 1},
            MetalSize{.width = 1, .height = 1, .depth = 1}));
        check(memory_barrier(compute));
    }
    if (!ok) {
        return SpeculativeError::CommandFailed;
    }
    auto ended = end_compute_pass(std::move(*pass.compute_pass));
    auto pending = ended ? commit(std::move(*ended.command_buffer))
                         : MetalPendingExecutionResult{};
    if (!pending ||
        !wait_until_completed(std::move(*pending.pending_execution))) {
        return SpeculativeError::CommandFailed;
    }
    return SpeculativeError::None;
}

SpeculativeError run_draft_block(SpeculativeState& state,
                                 std::uint32_t staged) {
    DecodeStep& step = *state.decode;
    DraftEngine& engine = state.draft;
    const std::uint64_t kv_row_bytes =
        std::uint64_t{kDraftKeyValueHeads} * kDraftHeadDimension * 2;
    std::uint32_t block_ids[kBlock];
    block_ids[0] = staged;
    for (std::uint32_t index = 1; index < kBlock; ++index) {
        block_ids[index] = kDraftMaskTokenId;
    }
    {
        std::uint32_t* id_slots =
            static_cast<std::uint32_t*>(state.block_id_slots.contents());
        for (std::uint32_t row = 0; row < kBlock; ++row) {
            id_slots[row] = block_ids[row];
        }
        auto command_buffer = create_command_buffer(*state.queue);
        if (!command_buffer) {
            return SpeculativeError::CommandFailed;
        }
        auto pass =
            begin_compute_pass(std::move(*command_buffer.command_buffer));
        if (!pass) {
            return SpeculativeError::CommandFailed;
        }
        Encoder encode{.pass = &*pass.compute_pass, .engine = &engine};
        for (std::uint32_t row = 0; row < kBlock; ++row) {
            encode.check(set_compute_pipeline(*pass.compute_pass,
                                              step.pipelines.embed));
            encode.check(set_buffer(
                *pass.compute_pass, *step.image,
                step.tensor_offsets[step.bindings.embedding.weight], 0));
            encode.check(set_buffer(
                *pass.compute_pass, *step.image,
                step.tensor_offsets[step.bindings.embedding.scales], 1));
            encode.check(set_buffer(
                *pass.compute_pass, *step.image,
                step.tensor_offsets[step.bindings.embedding.biases], 2));
            encode.check(set_buffer(*pass.compute_pass, state.block_id_slots,
                                    std::uint64_t{row} * 4, 3));
            encode.check(set_buffer(*pass.compute_pass, engine.x,
                                    std::uint64_t{row} * kDraftHidden * 2,
                                    4));
            encode.check(dispatch_threadgroups(
                *pass.compute_pass,
                MetalSize{.width = step.geometry.dispatch.embed.groups,
                          .height = 1, .depth = 1},
                MetalSize{.width = step.geometry.dispatch.embed.threads,
                          .height = 1, .depth = 1}));
        }
        if (encode.failed) {
            return SpeculativeError::CommandFailed;
        }
        auto ended = end_compute_pass(std::move(*pass.compute_pass));
        auto pending = ended ? commit(std::move(*ended.command_buffer))
                             : MetalPendingExecutionResult{};
        if (!pending ||
            !wait_until_completed(std::move(*pending.pending_execution))) {
            return SpeculativeError::CommandFailed;
        }
    }
    {
        auto command_buffer = create_command_buffer(*state.queue);
        if (!command_buffer) {
            return SpeculativeError::CommandFailed;
        }
        auto pass =
            begin_compute_pass(std::move(*command_buffer.command_buffer));
        if (!pass) {
            return SpeculativeError::CommandFailed;
        }
        Encoder encode{.pass = &*pass.compute_pass, .engine = &engine};
        const std::uint32_t key_count = engine.cached_rows + kBlock;
        for (std::uint32_t layer = 0; layer < kDraftLayers; ++layer) {
            const std::string prefix =
                "layers." + std::to_string(layer) + ".";
            encode.rms(engine.x, 0, prefix + "input_layernorm.weight",
                       engine.normed, 0, kBlock, kDraftHidden);
            encode.dense(engine.normed, 0,
                         prefix + "self_attn.k_proj.weight",
                         engine.key_caches[layer],
                         std::uint64_t{engine.cached_rows} * kv_row_bytes,
                         kBlock, kDraftHidden,
                         kDraftKeyValueHeads * kDraftHeadDimension);
            encode.dense(engine.normed, 0,
                         prefix + "self_attn.v_proj.weight",
                         engine.value_caches[layer],
                         std::uint64_t{engine.cached_rows} * kv_row_bytes,
                         kBlock, kDraftHidden,
                         kDraftKeyValueHeads * kDraftHeadDimension);
            encode.rms(engine.key_caches[layer],
                       std::uint64_t{engine.cached_rows} * kv_row_bytes,
                       prefix + "self_attn.k_norm.weight",
                       engine.key_caches[layer],
                       std::uint64_t{engine.cached_rows} * kv_row_bytes,
                       kBlock * kDraftKeyValueHeads, kDraftHeadDimension);
            encode.rope(engine.key_caches[layer],
                        std::uint64_t{engine.cached_rows} * kv_row_bytes,
                        kBlock, kDraftKeyValueHeads, engine.offset);
            encode.dense(engine.normed, 0,
                         prefix + "self_attn.q_proj.weight", engine.q, 0,
                         kBlock, kDraftHidden,
                         kDraftQueryHeads * kDraftHeadDimension);
            encode.rms(engine.q, 0, prefix + "self_attn.q_norm.weight",
                       engine.q, 0, kBlock * kDraftQueryHeads,
                       kDraftHeadDimension);
            encode.rope(engine.q, 0, kBlock, kDraftQueryHeads,
                        engine.offset);
            {
                int* position_values =
                    static_cast<int*>(engine.positions.contents());
                for (std::uint32_t row = 0; row < kBlock; ++row) {
                    position_values[engine.cached_rows + row] =
                        static_cast<int>(engine.offset + row);
                }
            }
            encode.pipeline("draft_block_attention");
            encode.buffer(engine.q, 0, 0);
            encode.buffer(engine.key_caches[layer], 0, 1);
            encode.buffer(engine.value_caches[layer], 0, 2);
            encode.buffer(engine.positions, 0, 3);
            encode.buffer(engine.attn, 0, 4);
            encode.constant(key_count, 5);
            encode.constant(kBlock, 6);
            encode.constant(engine.offset, 7);
            encode.constant(layer == kDraftLayers - 1 ? 1u : 0u, 8);
            encode.constant(engine.capacity, 9);
            encode.dispatch(kDraftKeyValueHeads, 1, 128);
            encode.barrier();
            encode.dense(engine.attn, 0, prefix + "self_attn.o_proj.weight",
                         engine.branch, 0, kBlock,
                         kDraftQueryHeads * kDraftHeadDimension,
                         kDraftHidden);
            encode.pipeline("draft_residual_add");
            encode.buffer(engine.x, 0, 0);
            encode.buffer(engine.branch, 0, 1);
            encode.constant(kBlock, 2);
            encode.constant(kDraftHidden, 3);
            encode.check(dispatch_threadgroups(
                *encode.pass,
                MetalSize{.width = (kDraftHidden + 255) / 256,
                          .height = kBlock, .depth = 1},
                MetalSize{.width = 256, .height = 1, .depth = 1}));
            encode.barrier();
            encode.rms(engine.x, 0,
                       prefix + "post_attention_layernorm.weight",
                       engine.normed, 0, kBlock, kDraftHidden);
            encode.dense(engine.normed, 0, prefix + "mlp.gate_proj.weight",
                         engine.gate, 0, kBlock, kDraftHidden,
                         kDraftIntermediate);
            encode.dense(engine.normed, 0, prefix + "mlp.up_proj.weight",
                         engine.up, 0, kBlock, kDraftHidden,
                         kDraftIntermediate);
            encode.pipeline("draft_swiglu");
            encode.buffer(engine.gate, 0, 0);
            encode.buffer(engine.up, 0, 1);
            encode.buffer(engine.swiglu, 0, 2);
            encode.constant(kBlock, 3);
            encode.constant(kDraftIntermediate, 4);
            encode.check(dispatch_threadgroups(
                *encode.pass,
                MetalSize{.width = (kDraftIntermediate + 255) / 256,
                          .height = kBlock, .depth = 1},
                MetalSize{.width = 256, .height = 1, .depth = 1}));
            encode.barrier();
            encode.dense(engine.swiglu, 0, prefix + "mlp.down_proj.weight",
                         engine.branch, 0, kBlock, kDraftIntermediate,
                         kDraftHidden);
            encode.pipeline("draft_residual_add");
            encode.buffer(engine.x, 0, 0);
            encode.buffer(engine.branch, 0, 1);
            encode.constant(kBlock, 2);
            encode.constant(kDraftHidden, 3);
            encode.check(dispatch_threadgroups(
                *encode.pass,
                MetalSize{.width = (kDraftHidden + 255) / 256,
                          .height = kBlock, .depth = 1},
                MetalSize{.width = 256, .height = 1, .depth = 1}));
            encode.barrier();
        }
        encode.rms(engine.x, 0, "norm.weight", engine.final_hidden, 0,
                   kBlock, kDraftHidden);
        {
            const std::uint32_t head_rows = kBlock - 1;
            const std::uint32_t vocabulary =
                step.geometry.dispatch.vocabulary_rows;
            const std::uint32_t reduction = kDraftHidden;
            const std::uint64_t activation_stride = kDraftHidden;
            const std::uint64_t packed_stride = kDraftHidden / 8;
            const std::uint64_t parameter_stride = kDraftHidden / 64;
            const std::uint64_t output_stride = vocabulary;
            encode.check(set_compute_pipeline(*encode.pass, state.n1_head));
            encode.check(set_buffer(*encode.pass, engine.final_hidden,
                                    std::uint64_t{1} * kDraftHidden * 2, 0));
            encode.check(set_buffer(
                *encode.pass, *step.image,
                step.tensor_offsets[step.bindings.head.weight], 1));
            encode.check(set_buffer(
                *encode.pass, *step.image,
                step.tensor_offsets[step.bindings.head.scales], 2));
            encode.check(set_buffer(
                *encode.pass, *step.image,
                step.tensor_offsets[step.bindings.head.biases], 3));
            encode.check(set_buffer(*encode.pass, state.draft_logits, 0, 4));
            encode.check(set_bytes(*encode.pass, &head_rows, 4, 5));
            encode.check(set_bytes(*encode.pass, &vocabulary, 4, 6));
            encode.check(set_bytes(*encode.pass, &reduction, 4, 7));
            encode.check(set_bytes(*encode.pass, &activation_stride, 8, 8));
            encode.check(set_bytes(*encode.pass, &packed_stride, 8, 9));
            encode.check(set_bytes(*encode.pass, &parameter_stride, 8, 10));
            encode.check(set_bytes(*encode.pass, &output_stride, 8, 11));
            encode.check(dispatch_threadgroups(
                *encode.pass,
                MetalSize{.width = (vocabulary + 31) / 32,
                          .height = (head_rows + 31) / 32, .depth = 1},
                MetalSize{.width = 128, .height = 1, .depth = 1}));
            encode.barrier();
        }
        for (std::uint32_t row = 1; row < kBlock; ++row) {
            encode.check(set_compute_pipeline(*encode.pass,
                                              step.pipelines.argmax_stage1));
            encode.check(set_buffer(*encode.pass, state.draft_logits,
                                    std::uint64_t{row - 1} *
                                        step.geometry.dispatch
                                            .vocabulary_rows * 2, 0));
            encode.check(set_bytes(*encode.pass,
                                   &step.geometry.dispatch.vocabulary_rows,
                                   4, 1));
            encode.check(set_buffer(*encode.pass, step.argmax_values, 0, 2));
            encode.check(set_buffer(*encode.pass, step.argmax_indices, 0, 3));
            encode.check(dispatch_threadgroups(
                *encode.pass,
                MetalSize{.width = step.geometry.dispatch.argmax_stage1.groups,
                          .height = 1, .depth = 1},
                MetalSize{.width = step.geometry.dispatch.argmax_stage1.threads,
                          .height = 1, .depth = 1}));
            encode.barrier();
            encode.check(set_compute_pipeline(*encode.pass,
                                              step.pipelines.argmax_stage2));
            encode.check(set_buffer(*encode.pass, step.argmax_values, 0, 0));
            encode.check(set_buffer(*encode.pass, step.argmax_indices, 0, 1));
            encode.check(set_buffer(*encode.pass, state.drafted_ids,
                                    std::uint64_t{row} * 4, 2));
            encode.check(dispatch_threadgroups(
                *encode.pass,
                MetalSize{.width = 1, .height = 1, .depth = 1},
                MetalSize{.width = 1, .height = 1, .depth = 1}));
            encode.barrier();
        }
        if (encode.failed) {
            return SpeculativeError::CommandFailed;
        }
        auto ended = end_compute_pass(std::move(*pass.compute_pass));
        auto pending = ended ? commit(std::move(*ended.command_buffer))
                             : MetalPendingExecutionResult{};
        if (!pending ||
            !wait_until_completed(std::move(*pending.pending_execution))) {
            return SpeculativeError::CommandFailed;
        }
    }
    return SpeculativeError::None;
}

SpeculativeError tape_rollback(SpeculativeState& state,
                               const GdnSnapshot& pre_verify,
                               std::uint32_t committed) {
    DecodeStep& step = *state.decode;
    restore_gdn(step, pre_verify);
    auto command_buffer = create_command_buffer(*state.queue);
    if (!command_buffer) {
        return SpeculativeError::CommandFailed;
    }
    auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
    if (!pass) {
        return SpeculativeError::CommandFailed;
    }
    bool ok = true;
    const auto check = [&](MetalCommandError error) {
        ok = ok && error == MetalCommandError::None;
    };
    for (std::size_t l = 0; l < step.schedule.size(); ++l) {
        if (step.schedule[l] != model::qwen36::LayerKind::GatedDelta) {
            continue;
        }
        auto& layer_state = step.state.layers[l];
        MetalBuffer& recurrent_in = layer_state.swapped
                                        ? layer_state.second_out
                                        : layer_state.second;
        MetalBuffer& recurrent_out = layer_state.swapped
                                         ? layer_state.second
                                         : layer_state.second_out;
        const std::uint32_t layer_index = static_cast<std::uint32_t>(l);
        check(set_compute_pipeline(*pass.compute_pass, state.tape_replay));
        check(set_buffer(*pass.compute_pass, state.verify->gdn_tape, 0, 0));
        check(set_buffer(*pass.compute_pass, recurrent_in, 0, 1));
        check(set_buffer(*pass.compute_pass, recurrent_out, 0, 2));
        check(set_bytes(*pass.compute_pass, &committed, 4, 3));
        check(set_bytes(*pass.compute_pass, &layer_index, 4, 4));
        check(dispatch_threadgroups(
            *pass.compute_pass,
            MetalSize{.width = 1, .height = 32, .depth = 32},
            MetalSize{.width = 32, .height = 4, .depth = 1}));
    }
    if (!ok) {
        return SpeculativeError::CommandFailed;
    }
    auto ended = end_compute_pass(std::move(*pass.compute_pass));
    if (!ended) {
        return SpeculativeError::CommandFailed;
    }
    auto blit = begin_blit_pass(std::move(*ended.command_buffer));
    if (!blit) {
        return SpeculativeError::CommandFailed;
    }
    const std::uint64_t conv_row_bytes = std::uint64_t{3} * 8192u * 2u;
    for (std::size_t l = 0; l < step.schedule.size(); ++l) {
        if (step.schedule[l] != model::qwen36::LayerKind::GatedDelta) {
            continue;
        }
        auto& layer_state = step.state.layers[l];
        MetalBuffer& conv_out = layer_state.swapped ? layer_state.first
                                                    : layer_state.first_out;
        if (copy_buffer(*blit.blit_pass,
                        state.verify->gdn_conv_tape_buffer,
                        (std::uint64_t{l} * 16u + (committed - 1u)) *
                            conv_row_bytes,
                        conv_out, 0,
                        conv_row_bytes) != MetalCommandError::None) {
            return SpeculativeError::CommandFailed;
        }
    }
    auto blit_ended = end_blit_pass(std::move(*blit.blit_pass));
    auto pending = blit_ended
                       ? commit(std::move(*blit_ended.command_buffer))
                       : MetalPendingExecutionResult{};
    if (!pending ||
        !wait_until_completed(std::move(*pending.pending_execution))) {
        return SpeculativeError::CommandFailed;
    }
    for (std::size_t l = 0; l < step.schedule.size(); ++l) {
        if (step.schedule[l] == model::qwen36::LayerKind::GatedDelta) {
            step.state.layers[l].swapped = !step.state.layers[l].swapped;
        }
    }
    return SpeculativeError::None;
}

} // namespace

SpeculativeEngine::SpeculativeEngine() : state_(new detail::SpeculativeState) {}
SpeculativeEngine::~SpeculativeEngine() = default;

std::uint32_t SpeculativeEngine::conditioned_rows() const {
    return state_->conditioned;
}

void SpeculativeEngine::extend_history(const std::uint32_t* ids,
                                       std::size_t count) {
    state_->history.insert(state_->history.end(), ids, ids + count);
}

void SpeculativeEngine::reset_request() {
    SpeculativeState& state = *state_;
    state.conditioned = 0;
    state.draft.cached_rows = 0;
    state.draft.offset = 0;
    state.history.clear();
    state.copyspec_enabled = true;
}

SpeculativeEngineResult create_speculative_engine(
    const MetalDevice& device, const MetalLibrary& library,
    const MetalCommandQueue& queue, DecodeStep& decode,
    std::uint32_t context_capacity, std::string_view draft_checkpoint_root) {
    SpeculativeEngineResult result;
    auto engine =
        std::unique_ptr<SpeculativeEngine>(new SpeculativeEngine());
    SpeculativeState& state = *engine->state_;
    state.device = &device;
    state.library = &library;
    state.queue = &queue;
    state.decode = &decode;
    state.context_capacity = context_capacity;

    state.draft_load = load_draft_checkpoint(draft_checkpoint_root);
    if (!state.draft_load) {
        result.error = SpeculativeError::DraftCheckpointRefused;
        return result;
    }
    for (const std::string_view name :
         {"draft_dense_bf16", "draft_gemm16_bf16", "draft_rms_rows",
          "draft_rope128", "draft_block_attention", "draft_swiglu",
          "draft_residual_add"}) {
        auto function = create_function(library, name);
        auto pipeline = function
                            ? create_compute_pipeline(device,
                                                      *function.function)
                            : MetalComputePipelineResult{};
        if (!pipeline) {
            result.error = SpeculativeError::PipelineUnavailable;
            return result;
        }
        state.draft.pipelines.emplace(name, std::move(*pipeline.pipeline));
    }
    for (const auto& [name, slot] :
         std::initializer_list<std::pair<std::string_view,
                                         MetalComputePipeline*>>{
             {"native_dense_qgemm_q4_bf16_n1", &state.n1_head},
             {"gdn_tape_replay_blk", &state.tape_replay}}) {
        auto function = create_function(library, name);
        auto pipeline = function
                            ? create_compute_pipeline(device,
                                                      *function.function)
                            : MetalComputePipelineResult{};
        if (!pipeline) {
            result.error = SpeculativeError::PipelineUnavailable;
            return result;
        }
        *slot = std::move(*pipeline.pipeline);
    }

    PipelineResult verify_pipelines =
        resolve_prefill_pipelines(device, library, false);
    if (!verify_pipelines) {
        result.error = SpeculativeError::PipelineUnavailable;
        return result;
    }
    const auto& plan = model::qwen36::generated::kModelPlan;
    const PrefillPolicy geometry_policy{
        .schedule = PrefillSchedule::LayerMajor,
        .context_capacity = context_capacity,
        .maximum_block_rows = 2048,
        .first_chunk_rows = 256,
        .query_tile_rows = 256,
        .attention_partition = kAttentionPartition,
        .exact_rows_per_threadgroup = 16,
        .gdn_gate_hoist = true,
    };
    const auto geometry = make_prefill_geometry(plan, geometry_policy);
    if (!geometry) {
        result.error = SpeculativeError::VerifyStepUnavailable;
        return result;
    }
    const PrefillExecutionPolicy verify_policy{
        .geometry = geometry_policy,
        .router_selector = PrefillRouterSelector::Parallel,
        .gdn_recurrence = PrefillGdnRecurrence::RegisterLoopTape,
        .attention_kernel = PrefillAttentionKernel::StagedGemmAdaptive,
        .staged_attention_minimum_context =
            std::min<std::uint32_t>(256, context_capacity - 1u),
        .dense_qgemm = QuantizedGemmPolicy::NativeDenseMma,
        .routed_qgemm = QuantizedGemmPolicy::NativeRaggedMma,
        .native_dense_steel = true,
        .native_routed_shared_expert = true,
        .native_routed_steel = true,
        .command_graph = false,
        .command_graph_chunk_count = 1,
        .maximum_units_per_submission = kUnitBatch,
        .maximum_inflight_units = 1,
    };
    auto verify_result = create_prefill_step(
        device, geometry.geometry, verify_policy,
        std::move(verify_pipelines.pipelines));
    if (!verify_result) {
        result.error = SpeculativeError::VerifyStepUnavailable;
        return result;
    }
    state.verify_step_owner = std::move(verify_result.step);
    state.verify = &*state.verify_step_owner;

    const auto make_buffer = [&](std::uint64_t bytes) {
        auto created = create_shared_buffer(device, bytes);
        return created ? std::move(*created.buffer) : MetalBuffer{};
    };
    const std::uint64_t kv_row_bytes =
        std::uint64_t{kDraftKeyValueHeads} * kDraftHeadDimension * 2;
    state.draft.capacity =
        ((context_capacity + kDraftSinkPositions + 15) / 16) * 16;
    for (std::uint32_t layer = 0; layer < kDraftLayers; ++layer) {
        state.draft.key_caches.push_back(make_buffer(
            std::uint64_t{state.draft.capacity} * kv_row_bytes));
        state.draft.value_caches.push_back(make_buffer(
            std::uint64_t{state.draft.capacity} * kv_row_bytes));
    }
    state.draft.positions =
        make_buffer(std::uint64_t{state.draft.capacity} * 4);
    state.draft.ctx_rows =
        make_buffer(std::uint64_t{kChunkRows} * kDraftHidden * 2);
    state.draft.x = make_buffer(std::uint64_t{kBlock} * kDraftHidden * 2);
    state.draft.normed =
        make_buffer(std::uint64_t{kBlock} * kDraftHidden * 2);
    state.draft.q = make_buffer(
        std::uint64_t{kBlock} * kDraftQueryHeads * kDraftHeadDimension * 2);
    state.draft.attn = make_buffer(
        std::uint64_t{kBlock} * kDraftQueryHeads * kDraftHeadDimension * 2);
    state.draft.gate =
        make_buffer(std::uint64_t{kBlock} * kDraftIntermediate * 2);
    state.draft.up =
        make_buffer(std::uint64_t{kBlock} * kDraftIntermediate * 2);
    state.draft.swiglu =
        make_buffer(std::uint64_t{kBlock} * kDraftIntermediate * 2);
    state.draft.branch =
        make_buffer(std::uint64_t{kBlock} * kDraftHidden * 2);
    state.draft.final_hidden =
        make_buffer(std::uint64_t{kBlock} * kDraftHidden * 2);
    state.drafted_ids = make_buffer(std::uint64_t{kBlock} * 4);
    state.block_id_slots = make_buffer(std::uint64_t{kBlock} * 4);
    state.verify_out_ids = make_buffer(std::uint64_t{kBlock} * 4);
    state.draft_logits = make_buffer(
        std::uint64_t{kBlock} *
        decode.geometry.dispatch.vocabulary_rows * 2);
    state.cycle_features = make_buffer(
        std::uint64_t{kBlock} * 8u * kDraftHidden * 2);
    state.normed = make_buffer(std::uint64_t{kBlock} * kDraftHidden * 2);

    {
        std::vector<std::string> names{"fc.weight", "hidden_norm.weight",
                                       "norm.weight"};
        for (std::uint32_t layer = 0; layer < kDraftLayers; ++layer) {
            for (const char* stem :
                 {"input_layernorm.weight", "self_attn.q_proj.weight",
                  "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                  "self_attn.o_proj.weight", "self_attn.q_norm.weight",
                  "self_attn.k_norm.weight",
                  "post_attention_layernorm.weight", "mlp.gate_proj.weight",
                  "mlp.up_proj.weight", "mlp.down_proj.weight"}) {
                names.push_back("layers." + std::to_string(layer) + "." +
                                stem);
            }
        }
        for (const std::string& name : names) {
            const DraftTensorView view =
                state.draft_load.checkpoint.tensor(name);
            MetalBuffer buffer = make_buffer(view.elements * 2);
            if (view.data == nullptr || !buffer) {
                result.error = SpeculativeError::BufferAllocationFailed;
                return result;
            }
            std::memcpy(buffer.contents(), view.data, view.elements * 2);
            state.draft.weights.emplace(name, std::move(buffer));
        }
    }
    if (!state.draft.x || !state.cycle_features) {
        result.error = SpeculativeError::BufferAllocationFailed;
        return result;
    }
    result.engine = std::move(engine);
    result.error = SpeculativeError::None;
    return result;
}

SpeculativeError SpeculativeEngine::observe_prompt_band(
    const MetalBuffer& captures, std::uint32_t rows, std::uint32_t base) {
    SpeculativeState& state = *state_;
    if (base != state.conditioned) {
        return SpeculativeError::CaptureOverflow;
    }
    auto command_buffer = create_command_buffer(*state.queue);
    if (!command_buffer) {
        return SpeculativeError::CommandFailed;
    }
    auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
    if (!pass) {
        return SpeculativeError::CommandFailed;
    }
    Encoder encode{.pass = &*pass.compute_pass, .engine = &state.draft};
    for (std::uint32_t chunk = 0; chunk < rows; chunk += kChunkRows) {
        const std::uint32_t chunk_rows = std::min(kChunkRows, rows - chunk);
        encode.dense(captures,
                     std::uint64_t{chunk} * kDraftFeatureWidth * 2,
                     "fc.weight", state.draft.ctx_rows, 0, chunk_rows,
                     kDraftFeatureWidth, kDraftHidden);
        encode.rms(state.draft.ctx_rows, 0, "hidden_norm.weight",
                   state.draft.ctx_rows, 0, chunk_rows, kDraftHidden);
        append_context_rows(encode, state.draft, chunk_rows, base + chunk,
                            base + chunk);
    }
    if (encode.failed) {
        return SpeculativeError::CommandFailed;
    }
    auto ended = end_compute_pass(std::move(*pass.compute_pass));
    auto pending = ended ? commit(std::move(*ended.command_buffer))
                         : MetalPendingExecutionResult{};
    if (!pending ||
        !wait_until_completed(std::move(*pending.pending_execution))) {
        return SpeculativeError::CommandFailed;
    }
    state.conditioned = base + rows;
    state.draft.cached_rows = state.conditioned;
    state.draft.offset = state.conditioned;
    return SpeculativeError::None;
}

SpeculativeError SpeculativeEngine::observe_handoff_row(
    const MetalBuffer& features, std::uint32_t base) {
    return observe_prompt_band(features, 1, base);
}

SpeculativeStepResult SpeculativeEngine::step(std::uint32_t staged,
                                              std::uint32_t context) {
    SpeculativeStepResult result;
    SpeculativeState& state = *state_;
    if (state.conditioned == 0) {
        result.error = SpeculativeError::NotConditioned;
        return result;
    }
    DecodeStep& decode = *state.decode;

    std::uint32_t verify_ids[kBlock];
    verify_ids[0] = staged;
    std::uint32_t copied[kBlock];
    const bool copy_hit =
        state.copyspec_enabled &&
        find_copy(state.history, staged, kBlock - 1, copied + 1);
    result.used_copy = copy_hit;
    if (copy_hit) {
        std::uint32_t* device_ids =
            static_cast<std::uint32_t*>(state.drafted_ids.contents());
        for (std::uint32_t row = 1; row < kBlock; ++row) {
            device_ids[row] = copied[row];
        }
    } else {
        const SpeculativeError draft_error = run_draft_block(state, staged);
        if (draft_error != SpeculativeError::None) {
            result.error = draft_error;
            return result;
        }
    }
    {
        const std::uint32_t* drafted =
            static_cast<const std::uint32_t*>(state.drafted_ids.contents());
        for (std::uint32_t row = 1; row < kBlock; ++row) {
            verify_ids[row] = drafted[row];
        }
    }

    GdnSnapshot pre_verify = snapshot_gdn(decode);
    const auto fail_restored = [&](SpeculativeError error) {
        restore_gdn(decode, pre_verify);
        result.error = error;
        return result;
    };
    if (run_verify_band(state, context,
                        std::span<const std::uint32_t>(verify_ids, kBlock)) !=
        SpeculativeError::None) {
        return fail_restored(SpeculativeError::CommandFailed);
    }
    if (band_head(state) != SpeculativeError::None) {
        return fail_restored(SpeculativeError::CommandFailed);
    }
    std::uint32_t verify_posterior[kBlock];
    std::memcpy(verify_posterior, state.verify_out_ids.contents(),
                std::size_t{kBlock} * 4);

    std::uint32_t acceptance = 0;
    while (acceptance + 1 < kBlock &&
           verify_ids[acceptance + 1] == verify_posterior[acceptance]) {
        ++acceptance;
    }
    const std::uint32_t committed = acceptance + 1;
    result.next_staged = verify_posterior[acceptance];
    if (copy_hit && acceptance == 0) {
        state.copyspec_enabled = false;
    }

    if (committed < kBlock) {
        const SpeculativeError rollback =
            tape_rollback(state, pre_verify, committed);
        if (rollback != SpeculativeError::None) {
            return fail_restored(rollback);
        }
    }

    const SpeculativeError reproject = observe_prompt_band(
        state.cycle_features, committed, state.conditioned);
    if (reproject != SpeculativeError::None) {
        return fail_restored(reproject);
    }
    result.committed.assign(verify_ids, verify_ids + committed);
    state.history.insert(state.history.end(), verify_ids,
                         verify_ids + committed);
    result.error = SpeculativeError::None;
    return result;
}

} // namespace tatara::engine
