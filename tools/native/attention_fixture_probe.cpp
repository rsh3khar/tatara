#include "fixture_batteries.h"
#include "kernel_reference.h"
#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/generated/kernel_library.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace tatara::backend::metal;
using namespace tatara::testing;

constexpr std::uint32_t kQueryHeads = generated::kKernelLibraryAttnQueryHeads;
constexpr std::uint32_t kKvHeads = generated::kKernelLibraryAttnKvHeads;
constexpr std::uint32_t kDim = generated::kKernelLibraryAttnHeadDimension;
constexpr std::uint32_t kHeads = kQueryHeads + kKvHeads;
constexpr std::uint32_t kCapacity = 512;
constexpr std::uint32_t kPositions = 5;
constexpr std::uint32_t kShortContext = 4;
constexpr std::uint32_t kLongContext = 300;
constexpr std::uint32_t kPart = 256;
constexpr std::uint32_t kNParts = 2;
// The sealed record format requires part <= 256; the single-partition
// split-equals-single check therefore runs at a context that fits one
// partition: count = kSplitContext + 1 <= 256.
constexpr std::uint32_t kSplitContext = 200;
constexpr std::uint32_t kProjRows = kQueryHeads * 2 * kDim + 2 * kKvHeads * kDim;
constexpr float kEpsilon = 1e-6f;
constexpr AttnGeometry kGeometry{
    .query_heads = kQueryHeads, .kv_heads = kKvHeads, .head_dimension = kDim};

int submissions = 0;

template <typename Value> void upload(const MetalBuffer& buffer, std::span<const Value> values) {
    std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(Value));
}

template <typename Value>
std::vector<Value> download(const MetalBuffer& buffer, std::size_t count,
                            std::size_t offset_values = 0) {
    std::vector<Value> values(count);
    std::memcpy(values.data(), static_cast<const Value*>(buffer.contents()) + offset_values,
                count * sizeof(Value));
    return values;
}

struct Pass {
    const MetalComputePipeline* pipeline;
    std::vector<const MetalBuffer*> buffers;
    std::vector<std::uint64_t> offsets;
    MetalSize threadgroups;
    MetalSize threads;
};

int run_batch(const MetalCommandQueue& queue, std::span<const Pass> passes, const char* label) {
    auto command_buffer = create_command_buffer(queue);
    if (!command_buffer) {
        return 1;
    }
    MetalCommandBuffer chain = std::move(*command_buffer.command_buffer);
    for (const Pass& pass : passes) {
        auto begun = begin_compute_pass(std::move(chain));
        if (!begun) {
            return 2;
        }
        if (set_compute_pipeline(*begun.compute_pass, *pass.pipeline) != MetalCommandError::None) {
            return 3;
        }
        for (std::uint32_t index = 0; index < pass.buffers.size(); ++index) {
            if (set_buffer(*begun.compute_pass, *pass.buffers[index], pass.offsets[index], index) !=
                MetalCommandError::None) {
                return 4;
            }
        }
        if (dispatch_threadgroups(*begun.compute_pass, pass.threadgroups, pass.threads) !=
            MetalCommandError::None) {
            return 5;
        }
        auto ended = end_compute_pass(std::move(*begun.compute_pass));
        if (!ended) {
            return 6;
        }
        chain = std::move(*ended.command_buffer);
    }
    auto pending = commit(std::move(chain));
    if (!pending) {
        return 7;
    }
    ++submissions;
    auto execution = wait_until_completed(std::move(*pending.pending_execution));
    if (!execution) {
        std::cerr << label << " execution failed: " << execution.failure_description.view() << '\n';
        return 8;
    }
    return 0;
}

std::uint32_t mismatches_u16(std::span<const std::uint16_t> expected,
                             std::span<const std::uint16_t> actual, const char* label) {
    std::uint32_t count = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (expected[i] != actual[i]) {
            if (count < 4) {
                std::cout << "  " << label << " mismatch at " << i << ": expected 0x" << std::hex
                          << expected[i] << " actual 0x" << actual[i] << std::dec << '\n';
            }
            ++count;
        }
    }
    return count;
}

std::vector<float> query_exps(std::span<const float> device_exps, std::size_t& cursor,
                              std::size_t count) {
    std::vector<float> slice(device_exps.begin() + static_cast<std::ptrdiff_t>(cursor),
                             device_exps.begin() + static_cast<std::ptrdiff_t>(cursor + count));
    cursor += count;
    return slice;
}

} // namespace

int run_attention_battery() {
    auto device = create_system_device();
    if (!device) {
        std::cerr << "system Metal device creation failed\n";
        return 1;
    }
    auto queue = create_command_queue(*device.device);
    if (!queue) {
        std::cerr << "command queue creation failed\n";
        return 2;
    }
    auto library = create_library_with_source(*device.device, generated::kernel_library_source());
    if (!library) {
        std::cerr << "kernel library compilation failed:\n" << library.failure_description << '\n';
        return 3;
    }
    struct {
        const char* name;
        MetalComputePipeline pipeline;
    } kernels[] = {
        {"attn_project", {}},
        {"attn_qk_rope", {}},
        {"attention_decode", {}},
        {"attention_decode_scores_gqa4", {}},
        {"attention_decode_scores_gqa8", {}},
        {"attention_decode_values_gqa8", {}},
        {"attention_decode_values_gqa8_t512", {}},
        {"attention_decode_combine", {}},
        {"adjudicate_rsqrt", {}},
        {"adjudicate_rope_trig", {}},
        {"adjudicate_f32_exp", {}},
        {"adjudicate_q4_row", {}},
        {"attention_partial_blk", {}},
        {"attention_combine_blk", {}},
        {"attention_staged_scores_blk", {}},
        {"attention_staged_softmax_blk", {}},
        {"attention_staged_values_blk", {}},
        {"attention_streaming_blk", {}},
        {"attention_decode_scores_gqa4_simdreduce", {}},
        {"attention_decode_scores_values_gqa8", {}},
        {"attention_decode_vector_2pass_part", {}},
        {"attention_decode_vector_2pass_combine", {}},
    };
    for (auto& entry : kernels) {
        auto function = create_function(*library.library, entry.name);
        if (!function) {
            std::cerr << "function lookup failed: " << entry.name << '\n';
            return 4;
        }
        auto pipeline = create_compute_pipeline(*device.device, *function.function);
        if (!pipeline) {
            std::cerr << "pipeline creation failed: " << entry.name << '\n'
                      << pipeline.failure_description << '\n';
            return 5;
        }
        entry.pipeline = std::move(*pipeline.pipeline);
    }
    const auto& project = kernels[0].pipeline;
    const auto& qk_rope = kernels[1].pipeline;
    const auto& decode = kernels[2].pipeline;
    const auto& scores_gqa4 = kernels[3].pipeline;
    const auto& scores_gqa8 = kernels[4].pipeline;
    const auto& values_gqa8 = kernels[5].pipeline;
    const auto& values_gqa8_t512 = kernels[6].pipeline;
    const auto& combine = kernels[7].pipeline;
    const auto& adj_rsqrt = kernels[8].pipeline;
    const auto& adj_trig = kernels[9].pipeline;
    const auto& adj_exp = kernels[10].pipeline;
    const auto& adj_q4 = kernels[11].pipeline;
    const auto& prefill_partial = kernels[12].pipeline;
    const auto& prefill_combine = kernels[13].pipeline;
    const auto& staged_scores = kernels[14].pipeline;
    const auto& staged_softmax = kernels[15].pipeline;
    const auto& staged_values = kernels[16].pipeline;
    const auto& streaming = kernels[17].pipeline;
    const auto& scores_gqa4_simdreduce = kernels[18].pipeline;
    const auto& scores_values_gqa8 = kernels[19].pipeline;
    const auto& vector_2pass_part = kernels[20].pipeline;
    const auto& vector_2pass_combine = kernels[21].pipeline;

    Xorshift64Star generator(0xA77E1755ull);
    std::vector<std::uint16_t> projections(std::size_t{kPositions} * kProjRows);
    for (auto& value : projections) {
        value = generator.next_banded_bf16();
    }
    std::vector<std::uint16_t> q_weight(kDim), k_weight(kDim);
    for (std::uint32_t d = 0; d < kDim; ++d) {
        q_weight[d] = generator.next_banded_bf16();
        k_weight[d] = generator.next_banded_bf16();
    }

    auto dev_proj = create_shared_buffer(*device.device, projections.size() * 2);
    auto dev_qw = create_shared_buffer(*device.device, kDim * 2);
    auto dev_kw = create_shared_buffer(*device.device, kDim * 2);
    auto dev_q = create_shared_buffer(*device.device, kQueryHeads * kDim * 2);
    auto dev_gate = create_shared_buffer(*device.device, kQueryHeads * kDim * 2);
    auto dev_keys =
        create_shared_buffer(*device.device, std::size_t{kKvHeads} * kCapacity * kDim * 2);
    auto dev_vals =
        create_shared_buffer(*device.device, std::size_t{kKvHeads} * kCapacity * kDim * 2);
    std::vector<MetalBufferResult> dev_consts;
    constexpr std::uint32_t kVectorBlocks = 32;
    const std::uint32_t const_values[] = {
        0, 1, 2, 3, 4, kCapacity, kShortContext, kLongContext, kPart,
        kNParts, kSplitContext, 1, kVectorBlocks};
    for (const std::uint32_t value : const_values) {
        auto buffer = create_shared_buffer(*device.device, 4);
        if (!buffer) {
            return 6;
        }
        std::memcpy(buffer.buffer->contents(), &value, 4);
        dev_consts.push_back(std::move(buffer));
    }
    const MetalBuffer& c_pos0 = *dev_consts[0].buffer;
    const MetalBuffer& c_capacity = *dev_consts[5].buffer;
    const MetalBuffer& c_short = *dev_consts[6].buffer;
    const MetalBuffer& c_long = *dev_consts[7].buffer;
    const MetalBuffer& c_part = *dev_consts[8].buffer;
    const MetalBuffer& c_nparts = *dev_consts[9].buffer;
    const MetalBuffer& c_split = *dev_consts[10].buffer;
    const MetalBuffer& c_one = *dev_consts[11].buffer;
    const MetalBuffer& c_vector_blocks = *dev_consts[12].buffer;
    (void)c_pos0;
    for (auto* result : {&dev_proj, &dev_qw, &dev_kw, &dev_q, &dev_gate, &dev_keys, &dev_vals}) {
        if (!*result) {
            return 6;
        }
    }
    upload<std::uint16_t>(*dev_proj.buffer, projections);
    upload<std::uint16_t>(*dev_qw.buffer, q_weight);
    upload<std::uint16_t>(*dev_kw.buffer, k_weight);
    std::memset(dev_keys.buffer->contents(), 0, std::size_t{kKvHeads} * kCapacity * kDim * 2);
    std::memset(dev_vals.buffer->contents(), 0, std::size_t{kKvHeads} * kCapacity * kDim * 2);

    // Chain qk_rope across five positions to populate the K/V cache.
    {
        std::vector<Pass> passes;
        for (std::uint32_t position = 0; position < kPositions; ++position) {
            passes.push_back({&qk_rope,
                              {&*dev_proj.buffer, &*dev_qw.buffer, &*dev_kw.buffer, &*dev_q.buffer,
                               &*dev_gate.buffer, &*dev_keys.buffer, &*dev_vals.buffer,
                               &*dev_consts[position].buffer, &c_capacity},
                              {std::uint64_t{position} * kProjRows * 2, 0, 0, 0, 0, 0, 0, 0, 0},
                              {.width = kHeads, .height = 1, .depth = 1},
                              {.width = kDim, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "qk-rope"); rc != 0) {
            return 20 + rc;
        }
    }
    const auto device_q = download<std::uint16_t>(*dev_q.buffer, kQueryHeads * kDim);
    const auto device_gate = download<std::uint16_t>(*dev_gate.buffer, kQueryHeads * kDim);
    const auto device_keys =
        download<std::uint16_t>(*dev_keys.buffer, std::size_t{kKvHeads} * kCapacity * kDim);
    const auto device_vals =
        download<std::uint16_t>(*dev_vals.buffer, std::size_t{kKvHeads} * kCapacity * kDim);

    // Query batch: rsqrt arguments, RoPE trig, and every exp argument.
    std::vector<float> rsqrt_args;
    for (std::uint32_t position = 0; position < kPositions; ++position) {
        const std::span<const std::uint16_t> proj(
            projections.data() + std::size_t{position} * kProjRows, kProjRows);
        for (std::uint32_t head = 0; head < kHeads; ++head) {
            const bool is_query = head < kQueryHeads;
            const std::uint32_t base =
                is_query ? head * 2 * kDim : kQueryHeads * 2 * kDim + (head - kQueryHeads) * kDim;
            rsqrt_args.push_back(attn_rms_argument(proj, base, kDim, kEpsilon));
        }
    }
    const std::size_t rsqrt_count = rsqrt_args.size();
    rsqrt_args.resize(((rsqrt_count + 255) / 256) * 256, 1.0f);
    std::vector<std::uint32_t> trig_positions, trig_pairs;
    for (std::uint32_t position = 0; position < kPositions; ++position) {
        for (std::uint32_t pair = 0; pair < 32; ++pair) {
            trig_positions.push_back(position);
            trig_pairs.push_back(pair);
        }
    }
    const std::size_t trig_count = trig_positions.size();
    trig_positions.resize(((trig_count + 255) / 256) * 256, 0);
    trig_pairs.resize(trig_positions.size(), 0);

    std::vector<float> exp_args;
    std::vector<std::size_t> decode_short_offsets, decode_long_offsets;
    for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
        decode_short_offsets.push_back(exp_args.size());
        attention_decode_arguments(device_q, device_keys, head, kShortContext, kCapacity, kGeometry,
                                   exp_args);
    }
    for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
        decode_long_offsets.push_back(exp_args.size());
        attention_decode_arguments(device_q, device_keys, head, kLongContext, kCapacity, kGeometry,
                                   exp_args);
    }
    const std::size_t gate_offset = exp_args.size();
    for (std::uint32_t i = 0; i < kQueryHeads * kDim; ++i) {
        exp_args.push_back(-f32_from_bf16(device_gate[i]));
    }
    std::vector<std::size_t> gqa4_offsets;
    for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
        for (std::uint32_t partition = 0; partition < kNParts; ++partition) {
            gqa4_offsets.push_back(exp_args.size());
            attention_scores_gqa4_arguments(device_q, device_keys, head, kLongContext, kCapacity,
                                            kPart, partition, kGeometry, exp_args);
        }
    }
    const std::size_t exp_count = exp_args.size();
    exp_args.resize(((exp_count + 255) / 256) * 256, 0.0f);

    auto dev_rsq_i = create_shared_buffer(*device.device, rsqrt_args.size() * 4);
    auto dev_rsq_o = create_shared_buffer(*device.device, rsqrt_args.size() * 4);
    auto dev_trig_p = create_shared_buffer(*device.device, trig_positions.size() * 4);
    auto dev_trig_i = create_shared_buffer(*device.device, trig_pairs.size() * 4);
    auto dev_trig_c = create_shared_buffer(*device.device, trig_positions.size() * 4);
    auto dev_trig_s = create_shared_buffer(*device.device, trig_positions.size() * 4);
    auto dev_exp_i = create_shared_buffer(*device.device, exp_args.size() * 4);
    auto dev_exp_o = create_shared_buffer(*device.device, exp_args.size() * 4);
    for (auto* result : {&dev_rsq_i, &dev_rsq_o, &dev_trig_p, &dev_trig_i, &dev_trig_c, &dev_trig_s,
                         &dev_exp_i, &dev_exp_o}) {
        if (!*result) {
            return 7;
        }
    }
    upload<float>(*dev_rsq_i.buffer, rsqrt_args);
    upload<std::uint32_t>(*dev_trig_p.buffer, trig_positions);
    upload<std::uint32_t>(*dev_trig_i.buffer, trig_pairs);
    upload<float>(*dev_exp_i.buffer, exp_args);
    {
        const Pass passes[] = {
            {&adj_rsqrt,
             {&*dev_rsq_i.buffer, &*dev_rsq_o.buffer},
             {0, 0},
             {.width = rsqrt_args.size() / 256, .height = 1, .depth = 1},
             {.width = 256, .height = 1, .depth = 1}},
            {&adj_trig,
             {&*dev_trig_p.buffer, &*dev_trig_i.buffer, &*dev_trig_c.buffer, &*dev_trig_s.buffer},
             {0, 0, 0, 0},
             {.width = trig_positions.size() / 256, .height = 1, .depth = 1},
             {.width = 256, .height = 1, .depth = 1}},
            {&adj_exp,
             {&*dev_exp_i.buffer, &*dev_exp_o.buffer},
             {0, 0},
             {.width = exp_args.size() / 256, .height = 1, .depth = 1},
             {.width = 256, .height = 1, .depth = 1}},
        };
        if (const int rc = run_batch(*queue.command_queue, passes, "queries"); rc != 0) {
            return 30 + rc;
        }
    }
    const auto rsqrt_device = download<float>(*dev_rsq_o.buffer, rsqrt_args.size());
    const auto cos_device = download<float>(*dev_trig_c.buffer, trig_positions.size());
    const auto sin_device = download<float>(*dev_trig_s.buffer, trig_positions.size());
    const auto exp_device = download<float>(*dev_exp_o.buffer, exp_args.size());

    // qk_rope comparison across rotation chain candidates.
    std::uint32_t rope_matches[2] = {0, 0};
    for (int order = 0; order < 2; ++order) {
        std::vector<std::uint16_t> ref_q(kQueryHeads * kDim), ref_gate(kQueryHeads * kDim);
        std::vector<std::uint16_t> ref_keys(std::size_t{kKvHeads} * kCapacity * kDim, 0);
        std::vector<std::uint16_t> ref_vals(std::size_t{kKvHeads} * kCapacity * kDim, 0);
        for (std::uint32_t position = 0; position < kPositions; ++position) {
            const std::span<const std::uint16_t> proj(
                projections.data() + std::size_t{position} * kProjRows, kProjRows);
            const std::span<const float> inverses(rsqrt_device.data() + position * kHeads, kHeads);
            const std::span<const float> cosines(cos_device.data() + position * 32, 32);
            const std::span<const float> sines(sin_device.data() + position * 32, 32);
            attn_qk_rope_reference(proj, q_weight, k_weight, position, kCapacity, kGeometry,
                                   inverses, cosines, sines,
                                   order == 0 ? ChainOrder::Fused : ChainOrder::Separate, ref_q,
                                   ref_gate, ref_keys, ref_vals);
        }
        if (ref_q == device_q && ref_keys == device_keys && ref_gate == device_gate &&
            ref_vals == device_vals) {
            ++rope_matches[order];
        }
    }
    std::cout << "attn_qk_rope: fused " << rope_matches[0] << ", separate " << rope_matches[1]
              << '\n';
    if (rope_matches[0] + rope_matches[1] == 0) {
        return 40;
    }

    // attention_decode fixtures at short and long context.
    auto dev_out_short = create_shared_buffer(*device.device, kQueryHeads * kDim * 2);
    auto dev_out_long = create_shared_buffer(*device.device, kQueryHeads * kDim * 2);
    auto dev_out_split = create_shared_buffer(*device.device, kQueryHeads * kDim * 2);
    if (!dev_out_short || !dev_out_long || !dev_out_split) {
        return 8;
    }
    {
        const Pass passes[] = {
            {&decode,
             {&*dev_q.buffer, &*dev_gate.buffer, &*dev_keys.buffer, &*dev_vals.buffer, &c_short,
              &*dev_out_short.buffer, &c_capacity},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = kQueryHeads, .height = 1, .depth = 1},
             {.width = kDim, .height = 1, .depth = 1}},
            {&decode,
             {&*dev_q.buffer, &*dev_gate.buffer, &*dev_keys.buffer, &*dev_vals.buffer, &c_long,
              &*dev_out_long.buffer, &c_capacity},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = kQueryHeads, .height = 1, .depth = 1},
             {.width = kDim, .height = 1, .depth = 1}},
            {&decode,
             {&*dev_q.buffer, &*dev_gate.buffer, &*dev_keys.buffer, &*dev_vals.buffer, &c_split,
              &*dev_out_split.buffer, &c_capacity},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = kQueryHeads, .height = 1, .depth = 1},
             {.width = kDim, .height = 1, .depth = 1}},
        };
        if (const int rc = run_batch(*queue.command_queue, passes, "decode"); rc != 0) {
            return 50 + rc;
        }
    }
    std::uint32_t decode_matches[2] = {0, 0};
    for (int order = 0; order < 2; ++order) {
        bool all = true;
        for (const bool long_context : {false, true}) {
            const auto actual = download<std::uint16_t>(
                long_context ? *dev_out_long.buffer : *dev_out_short.buffer, kQueryHeads * kDim);
            std::vector<std::uint16_t> expected(kQueryHeads * kDim);
            for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
                std::size_t cursor =
                    long_context ? decode_long_offsets[head] : decode_short_offsets[head];
                std::vector<float> exps;
                {
                    std::vector<float> args;
                    attention_decode_arguments(device_q, device_keys, head,
                                               long_context ? kLongContext : kShortContext,
                                               kCapacity, kGeometry, args);
                    exps = query_exps(exp_device, cursor, args.size());
                }
                std::vector<std::uint16_t> head_out(kQueryHeads * kDim);
                std::vector<float> gate_exp(kDim);
                for (std::uint32_t d = 0; d < kDim; ++d) {
                    gate_exp[d] = exp_device[gate_offset + head * kDim + d];
                }
                attention_decode_reference(
                    device_q, device_gate, device_keys, device_vals, head,
                    long_context ? kLongContext : kShortContext, kCapacity, kGeometry, exps,
                    order == 0 ? ChainOrder::Fused : ChainOrder::Separate, head_out);
                for (std::uint32_t d = 0; d < kDim; ++d) {
                    expected[head * kDim + d] = head_out[head * kDim + d];
                }
            }
            if (!std::equal(expected.begin(), expected.end(), actual.begin())) {
                all = false;
            }
        }
        if (all) {
            ++decode_matches[order];
        }
    }
    std::cout << "attention_decode: fused " << decode_matches[0] << ", separate "
              << decode_matches[1] << '\n';
    if (decode_matches[0] + decode_matches[1] == 0) {
        const auto actual = download<std::uint16_t>(*dev_out_short.buffer, kQueryHeads * kDim);
        std::vector<std::uint16_t> expected(kQueryHeads * kDim, 0);
        mismatches_u16(expected, actual, "decode-short-raw");
        return 41;
    }

    // Long-context split path: scores, values, combine — plus the
    // single-partition split whose merged result must equal attention_decode.
    const std::size_t record_floats = std::size_t{kQueryHeads} * kNParts * 258;
    auto dev_w2 = create_shared_buffer(*device.device, record_floats * 4);
    auto dev_w2_simdreduce =
        create_shared_buffer(*device.device, record_floats * 4);
    auto dev_w2_gqa8 =
        create_shared_buffer(*device.device, record_floats * 4);
    auto dev_p2 = create_shared_buffer(*device.device, record_floats * 4);
    auto dev_p2_t512 =
        create_shared_buffer(*device.device, record_floats * 4);
    auto dev_p2_fused =
        create_shared_buffer(*device.device, record_floats * 4);
    auto dev_w1 = create_shared_buffer(*device.device, std::size_t{kQueryHeads} * 258 * 4);
    auto dev_w1_simdreduce = create_shared_buffer(
        *device.device, std::size_t{kQueryHeads} * 258 * 4);
    auto dev_w1_gqa8 = create_shared_buffer(
        *device.device, std::size_t{kQueryHeads} * 258 * 4);
    auto dev_p1 = create_shared_buffer(*device.device, std::size_t{kQueryHeads} * 258 * 4);
    auto dev_p1_t512 = create_shared_buffer(
        *device.device, std::size_t{kQueryHeads} * 258 * 4);
    auto dev_p1_fused = create_shared_buffer(
        *device.device, std::size_t{kQueryHeads} * 258 * 4);
    auto dev_c2 = create_shared_buffer(*device.device, kQueryHeads * kDim * 2);
    auto dev_c1 = create_shared_buffer(*device.device, kQueryHeads * kDim * 2);
    for (auto* result : {
             &dev_w2,
             &dev_w2_simdreduce,
             &dev_w2_gqa8,
             &dev_p2,
             &dev_p2_t512,
             &dev_p2_fused,
             &dev_w1,
             &dev_w1_simdreduce,
             &dev_w1_gqa8,
             &dev_p1,
             &dev_p1_t512,
             &dev_p1_fused,
             &dev_c2,
             &dev_c1,
         }) {
        if (!*result) {
            return 9;
        }
    }
    {
        const Pass passes[] = {
            {&scores_gqa4,
             {&*dev_q.buffer, &*dev_keys.buffer, &c_long, &c_capacity, &c_part, &*dev_w2.buffer,
              &c_nparts},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = 4, .height = kNParts, .depth = 1},
             {.width = 256, .height = 4, .depth = 1}},
            {&scores_gqa4,
             {&*dev_q.buffer, &*dev_keys.buffer, &c_split, &c_capacity, &c_part, &*dev_w1.buffer,
              &c_one},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = 4, .height = 1, .depth = 1},
             {.width = 256, .height = 4, .depth = 1}},
        };
        if (const int rc = run_batch(*queue.command_queue, passes, "scores-gqa4"); rc != 0) {
            return 60 + rc;
        }
    }
    const auto weights2_device = download<float>(*dev_w2.buffer, record_floats);
    std::uint32_t gqa4_bad = 0;
    for (std::uint32_t head = 0; head < kQueryHeads; ++head) {
        for (std::uint32_t partition = 0; partition < kNParts; ++partition) {
            std::size_t cursor = gqa4_offsets[head * kNParts + partition];
            std::vector<float> args;
            attention_scores_gqa4_arguments(device_q, device_keys, head, kLongContext, kCapacity,
                                            kPart, partition, kGeometry, args);
            const auto exps = query_exps(exp_device, cursor, args.size());
            std::vector<float> expected(258);
            attention_scores_gqa4_reference(device_q, device_keys, head, kLongContext, kCapacity,
                                            kPart, partition, kGeometry, exps, expected);
            const std::span<const float> actual(
                weights2_device.data() + (std::size_t{head} * kNParts + partition) * 258, 258);
            for (std::uint32_t i = 0; i < 258; ++i) {
                if (std::bit_cast<std::uint32_t>(expected[i]) !=
                    std::bit_cast<std::uint32_t>(actual[i])) {
                    if (gqa4_bad < 4) {
                        std::cout << "  gqa4 head " << head << " part " << partition << " slot "
                                  << i << ": expected 0x" << std::hex
                                  << std::bit_cast<std::uint32_t>(expected[i]) << " actual 0x"
                                  << std::bit_cast<std::uint32_t>(actual[i]) << std::dec << '\n';
                    }
                    ++gqa4_bad;
                }
            }
        }
    }
    std::cout << "scores_gqa4 fixtures: mismatches " << gqa4_bad << '\n';
    if (gqa4_bad != 0) {
        return 61;
    }

    {
        const Pass passes[] = {
            {&scores_gqa4_simdreduce,
             {&*dev_q.buffer, &*dev_keys.buffer, &c_long,
              &c_capacity, &c_part, &*dev_w2_simdreduce.buffer,
              &c_nparts},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = 4, .height = kNParts, .depth = 1},
             {.width = 256, .height = 4, .depth = 1}},
            {&scores_gqa4_simdreduce,
             {&*dev_q.buffer, &*dev_keys.buffer, &c_split,
              &c_capacity, &c_part, &*dev_w1_simdreduce.buffer,
              &c_one},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = 4, .height = 1, .depth = 1},
             {.width = 256, .height = 4, .depth = 1}},
        };
        if (const int rc = run_batch(
                *queue.command_queue, passes, "scores-gqa4-simdreduce");
            rc != 0) {
            return 63 + rc;
        }
    }

    {
        const Pass passes[] = {
            {&scores_gqa8,
             {&*dev_q.buffer, &*dev_keys.buffer, &c_long,
              &c_capacity, &c_part, &*dev_w2_gqa8.buffer,
              &c_nparts},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = 4, .height = kNParts, .depth = 1},
             {.width = 256, .height = 4, .depth = 1}},
            {&scores_gqa8,
             {&*dev_q.buffer, &*dev_keys.buffer, &c_split,
              &c_capacity, &c_part, &*dev_w1_gqa8.buffer,
              &c_one},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = 4, .height = 1, .depth = 1},
             {.width = 256, .height = 4, .depth = 1}},
        };
        if (const int rc = run_batch(
                *queue.command_queue, passes, "scores-gqa8");
            rc != 0) {
            return 62 + rc;
        }
    }
    const auto weights2_gqa8 = download<float>(
        *dev_w2_gqa8.buffer, record_floats);
    const auto weights1_gqa4 = download<float>(
        *dev_w1.buffer, std::size_t{kQueryHeads} * 258);
    const auto weights2_simdreduce = download<float>(
        *dev_w2_simdreduce.buffer, record_floats);
    const auto weights1_simdreduce = download<float>(
        *dev_w1_simdreduce.buffer, std::size_t{kQueryHeads} * 258);
    const auto weights1_gqa8 = download<float>(
        *dev_w1_gqa8.buffer, std::size_t{kQueryHeads} * 258);
    const bool simdreduce_two_part_exact =
        weights2_simdreduce == weights2_device;
    const bool simdreduce_one_part_exact =
        weights1_simdreduce == weights1_gqa4;
    std::cout << "scores_gqa4_simdreduce bit-exact: two-part "
              << (simdreduce_two_part_exact ? "yes" : "no")
              << ", one-part "
              << (simdreduce_one_part_exact ? "yes" : "no") << '\n';
    if (!simdreduce_two_part_exact || !simdreduce_one_part_exact) {
        return 68;
    }
    const bool gqa8_two_part_exact =
        weights2_gqa8 == weights2_device;
    const bool gqa8_one_part_exact =
        weights1_gqa8 == weights1_gqa4;
    std::cout << "scores_gqa8 bit-exact: two-part "
              << (gqa8_two_part_exact ? "yes" : "no")
              << ", one-part "
              << (gqa8_one_part_exact ? "yes" : "no") << '\n';
    if (!gqa8_two_part_exact || !gqa8_one_part_exact) {
        return 69;
    }

    {
        const Pass passes[] = {
            {&values_gqa8,
             {&*dev_w2.buffer, &*dev_vals.buffer, &c_long, &c_capacity, &c_part, &*dev_p2.buffer,
              &c_nparts},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = kKvHeads, .height = kNParts, .depth = 1},
             {.width = 32, .height = 4, .depth = 8}},
            {&values_gqa8,
             {&*dev_w1.buffer, &*dev_vals.buffer, &c_split, &c_capacity, &c_part, &*dev_p1.buffer,
              &c_one},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = kKvHeads, .height = 1, .depth = 1},
             {.width = 32, .height = 4, .depth = 8}},
        };
        if (const int rc = run_batch(*queue.command_queue, passes, "values-gqa8"); rc != 0) {
            return 70 + rc;
        }
    }
    const auto partials2_device = download<float>(*dev_p2.buffer, record_floats);
    {
        const Pass passes[] = {
            {&scores_values_gqa8,
             {&*dev_q.buffer, &*dev_keys.buffer, &*dev_vals.buffer,
              &c_long, &c_capacity, &c_part, &*dev_p2_fused.buffer,
              &c_nparts},
             {0, 0, 0, 0, 0, 0, 0, 0},
             {.width = kKvHeads, .height = kNParts, .depth = 1},
             {.width = 256, .height = 4, .depth = 1}},
            {&scores_values_gqa8,
             {&*dev_q.buffer, &*dev_keys.buffer, &*dev_vals.buffer,
              &c_split, &c_capacity, &c_part, &*dev_p1_fused.buffer,
              &c_one},
             {0, 0, 0, 0, 0, 0, 0, 0},
             {.width = kKvHeads, .height = 1, .depth = 1},
             {.width = 256, .height = 4, .depth = 1}},
        };
        if (const int rc = run_batch(
                *queue.command_queue, passes, "scores-values-gqa8");
            rc != 0) {
            return 140 + rc;
        }
    }
    const auto partials2_fused = download<float>(
        *dev_p2_fused.buffer, record_floats);
    const auto partials1_fused = download<float>(
        *dev_p1_fused.buffer, std::size_t{kQueryHeads} * 258);
    const auto partials1_control = download<float>(
        *dev_p1.buffer, std::size_t{kQueryHeads} * 258);
    const bool fused_two_part_exact =
        partials2_fused == partials2_device;
    const bool fused_one_part_exact =
        partials1_fused == partials1_control;
    const bool fused_inputs_unchanged =
        download<std::uint16_t>(
            *dev_q.buffer, kQueryHeads * kDim) == device_q &&
        download<std::uint16_t>(
            *dev_keys.buffer,
            std::size_t{kKvHeads} * kCapacity * kDim) ==
            device_keys &&
        download<std::uint16_t>(
            *dev_vals.buffer,
            std::size_t{kKvHeads} * kCapacity * kDim) ==
            device_vals;
    std::cout << "scores_values_gqa8 bit-exact: two-part "
              << (fused_two_part_exact ? "yes" : "no")
              << ", one-part "
              << (fused_one_part_exact ? "yes" : "no")
              << ", inputs unchanged "
              << (fused_inputs_unchanged ? "yes" : "no") << '\n';
    if (!fused_two_part_exact || !fused_one_part_exact ||
        !fused_inputs_unchanged) {
        return 145;
    }
    {
        const Pass passes[] = {
            {&values_gqa8_t512,
             {&*dev_w2.buffer, &*dev_vals.buffer, &c_long,
              &c_capacity, &c_part, &*dev_p2_t512.buffer, &c_nparts},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = kKvHeads, .height = kNParts, .depth = 1},
             {.width = 32, .height = 2, .depth = 8}},
            {&values_gqa8_t512,
             {&*dev_w1.buffer, &*dev_vals.buffer, &c_split,
              &c_capacity, &c_part, &*dev_p1_t512.buffer, &c_one},
             {0, 0, 0, 0, 0, 0, 0},
             {.width = kKvHeads, .height = 1, .depth = 1},
             {.width = 32, .height = 2, .depth = 8}},
        };
        if (const int rc = run_batch(
                *queue.command_queue, passes, "values-gqa8-t512");
            rc != 0) {
            return 72 + rc;
        }
    }
    const auto partials2_t512 = download<float>(
        *dev_p2_t512.buffer, record_floats);
    const auto partials1_t512 = download<float>(
        *dev_p1_t512.buffer, std::size_t{kQueryHeads} * 258);
    const bool values_t512_two_part_exact =
        partials2_t512 == partials2_device;
    const bool values_t512_one_part_exact =
        partials1_t512 == partials1_control;
    std::cout << "values_gqa8_t512 bit-exact: two-part "
              << (values_t512_two_part_exact ? "yes" : "no")
              << ", one-part "
              << (values_t512_one_part_exact ? "yes" : "no") << '\n';
    if (!values_t512_two_part_exact || !values_t512_one_part_exact) {
        return 79;
    }
    std::uint32_t values_matches[2] = {0, 0};
    for (int order = 0; order < 2; ++order) {
        bool all = true;
        for (std::uint32_t head = 0; head < kQueryHeads && all; ++head) {
            for (std::uint32_t partition = 0; partition < kNParts && all; ++partition) {
                const std::span<const float> record(
                    weights2_device.data() + (std::size_t{head} * kNParts + partition) * 258, 258);
                std::vector<float> expected(258);
                attention_values_gqa8_reference(
                    record, device_vals, head >> 3u, kLongContext, kCapacity, kPart, partition,
                    kGeometry, order == 0 ? ChainOrder::Fused : ChainOrder::Separate, expected);
                const std::span<const float> actual(
                    partials2_device.data() + (std::size_t{head} * kNParts + partition) * 258, 258);
                for (std::uint32_t i = 0; i < 258; ++i) {
                    if (std::bit_cast<std::uint32_t>(expected[i]) !=
                        std::bit_cast<std::uint32_t>(actual[i])) {
                        all = false;
                        break;
                    }
                }
            }
        }
        if (all) {
            ++values_matches[order];
        }
    }
    std::cout << "values_gqa8: fused " << values_matches[0] << ", separate " << values_matches[1]
              << '\n';
    if (values_matches[0] + values_matches[1] == 0) {
        return 71;
    }

    {
        const Pass passes[] = {
            {&combine,
             {&*dev_p2.buffer, &*dev_gate.buffer, &c_nparts, &*dev_c2.buffer},
             {0, 0, 0, 0},
             {.width = kQueryHeads, .height = 1, .depth = 1},
             {.width = kDim, .height = 1, .depth = 1}},
            {&combine,
             {&*dev_p1.buffer, &*dev_gate.buffer, &c_one, &*dev_c1.buffer},
             {0, 0, 0, 0},
             {.width = kQueryHeads, .height = 1, .depth = 1},
             {.width = kDim, .height = 1, .depth = 1}},
        };
        if (const int rc = run_batch(*queue.command_queue, passes, "combine"); rc != 0) {
            return 80 + rc;
        }
    }
    const auto combine2_device = download<std::uint16_t>(*dev_c2.buffer, kQueryHeads * kDim);
    std::uint32_t combine_matches[2] = {0, 0};
    for (int order = 0; order < 2; ++order) {
        bool all = true;
        for (std::uint32_t head = 0; head < kQueryHeads && all; ++head) {
            const std::span<const float> records(
                partials2_device.data() + std::size_t{head} * kNParts * 258, kNParts * 258);
            std::vector<float> args;
            attention_combine_arguments(records, kNParts, args);
            std::vector<float> exps(args.size());
            for (std::size_t i = 0; i < args.size(); ++i) {
                exps[i] = std::exp(args[i]);
            }
            std::vector<float> gate_exp(kDim);
            for (std::uint32_t d = 0; d < kDim; ++d) {
                gate_exp[d] = exp_device[gate_offset + head * kDim + d];
            }
            std::vector<std::uint16_t> expected(kDim);
            attention_combine_reference(
                records, device_gate, head, kNParts, kGeometry, exps, gate_exp,
                order == 0 ? ChainOrder::Fused : ChainOrder::Separate, expected);
            for (std::uint32_t d = 0; d < kDim; ++d) {
                if (expected[d] != combine2_device[head * kDim + d]) {
                    all = false;
                    break;
                }
            }
        }
        if (all) {
            ++combine_matches[order];
        }
    }
    std::cout << "combine: fused " << combine_matches[0] << ", separate " << combine_matches[1]
              << '\n';
    if (combine_matches[0] + combine_matches[1] == 0) {
        return 81;
    }

    // Split-equals-single: the one-partition split path must reproduce the
    // single-kernel decode byte-for-byte, device against device.
    const auto combine1_device = download<std::uint16_t>(*dev_c1.buffer, kQueryHeads * kDim);
    const auto decode_split_device =
        download<std::uint16_t>(*dev_out_split.buffer, kQueryHeads * kDim);
    const std::uint32_t split_bad =
        mismatches_u16(decode_split_device, combine1_device, "split-vs-single");
    std::cout << "split-equals-single: mismatches " << split_bad << '\n';
    if (split_bad != 0) {
        return 82;
    }

    // A25 independent-head vector family. Its BF16 partition numerators are
    // intentionally not bit-identical to the permanent float-record path, so
    // compare the final gated BF16 vectors under the frozen numerical band.
    constexpr float kVectorMaximumAbsoluteError = 0.015625f;
    const std::size_t vector_partial_values =
        std::size_t{kQueryHeads} * kVectorBlocks * kDim;
    const std::size_t vector_scalar_values =
        std::size_t{kQueryHeads} * kVectorBlocks;
    auto vector_partials = create_shared_buffer(
        *device.device, vector_partial_values * sizeof(std::uint16_t));
    auto vector_sums = create_shared_buffer(
        *device.device, vector_scalar_values * sizeof(float));
    auto vector_maxs = create_shared_buffer(
        *device.device, vector_scalar_values * sizeof(float));
    auto vector_output = create_shared_buffer(
        *device.device, std::size_t{kQueryHeads} * kDim *
                            sizeof(std::uint16_t));
    for (auto* result :
         {&vector_partials, &vector_sums, &vector_maxs, &vector_output}) {
        if (!*result) {
            return 146;
        }
    }
    const Pass vector_passes[] = {
        {&vector_2pass_part,
         {&*dev_q.buffer, &*dev_keys.buffer, &*dev_vals.buffer,
          &*vector_partials.buffer, &*vector_sums.buffer,
          &*vector_maxs.buffer, &c_long, &c_capacity,
          &c_vector_blocks},
         {0, 0, 0, 0, 0, 0, 0, 0, 0},
         {.width = kKvHeads, .height = kVectorBlocks, .depth = 1},
         {.width = 32, .height = 8, .depth = 1}},
        {&vector_2pass_combine,
         {&*vector_partials.buffer, &*vector_sums.buffer,
          &*vector_maxs.buffer, &*dev_gate.buffer,
          &c_vector_blocks, &*vector_output.buffer},
         {0, 0, 0, 0, 0, 0},
         {.width = kQueryHeads, .height = 1, .depth = 1},
         {.width = 1024, .height = 1, .depth = 1}},
    };
    if (const int rc = run_batch(
            *queue.command_queue, vector_passes,
            "vector-2pass");
        rc != 0) {
        return 146 + rc;
    }
    const auto vector_actual = download<std::uint16_t>(
        *vector_output.buffer, std::size_t{kQueryHeads} * kDim);
    std::uint32_t vector_mismatches = 0;
    std::uint32_t vector_nonfinite = 0;
    float vector_maximum_absolute_error = 0.0f;
    for (std::size_t index = 0; index < vector_actual.size(); ++index) {
        const float expected = f32_from_bf16(combine2_device[index]);
        const float actual = f32_from_bf16(vector_actual[index]);
        if (vector_actual[index] != combine2_device[index]) {
            ++vector_mismatches;
        }
        if (!std::isfinite(actual)) {
            ++vector_nonfinite;
        } else {
            vector_maximum_absolute_error = std::max(
                vector_maximum_absolute_error,
                std::abs(actual - expected));
        }
    }
    const bool vector_inputs_unchanged =
        download<std::uint16_t>(
            *dev_q.buffer, kQueryHeads * kDim) == device_q &&
        download<std::uint16_t>(
            *dev_keys.buffer,
            std::size_t{kKvHeads} * kCapacity * kDim) ==
            device_keys &&
        download<std::uint16_t>(
            *dev_vals.buffer,
            std::size_t{kKvHeads} * kCapacity * kDim) ==
            device_vals;
    std::cout << "vector_2pass numerical: mismatches "
              << vector_mismatches << ", nonfinite "
              << vector_nonfinite << ", max_abs "
              << vector_maximum_absolute_error
              << ", inputs unchanged "
              << (vector_inputs_unchanged ? "yes" : "no") << '\n';
    if (vector_nonfinite != 0 ||
        vector_maximum_absolute_error >
            kVectorMaximumAbsoluteError ||
        !vector_inputs_unchanged) {
        return 147;
    }

    // Staged prefill is a numerical family: compare two independent staged
    // executions byte-for-byte, then compare that deterministic result with
    // the permanent partial/combine path under the declared normalized gate.
    // Thirteen rows exercises the streaming kernel's partial query tile.
    // Visible=313 also crosses two complete 128-key tiles and a 57-key tail,
    // while every query row has a distinct causal boundary in that tail.
    constexpr std::uint32_t kPrefillBlock = 13;
    constexpr std::uint32_t kPrefillContext = 300;
    constexpr std::uint32_t kPrefillVisible =
        kPrefillContext + kPrefillBlock;
    constexpr std::uint32_t kPrefillPartitions =
        (kPrefillVisible + kPart - 1u) / kPart;
    constexpr float kMaximumNormalizedError = 0.02f;
    std::vector<std::uint16_t> prefill_q(
        std::size_t{kPrefillBlock} * kQueryHeads * kDim);
    std::vector<std::uint16_t> prefill_gate(prefill_q.size());
    std::vector<std::uint16_t> prefill_keys(
        std::size_t{kKvHeads} * kCapacity * kDim);
    std::vector<std::uint16_t> prefill_values(prefill_keys.size());
    const auto scaled_bfloat =
        [&](float magnitude) {
            const std::int32_t centered =
                static_cast<std::int32_t>(
                    (generator.next() >> 32u) % 2049u) -
                1024;
            return bf16_from_f32(
                float(centered) * (magnitude / 1024.0f));
        };
    for (auto& value : prefill_q) {
        value = scaled_bfloat(1.0f);
    }
    for (auto& value : prefill_gate) {
        value = scaled_bfloat(4.0f);
    }
    for (auto& value : prefill_keys) {
        value = scaled_bfloat(1.0f);
    }
    for (auto& value : prefill_values) {
        value = scaled_bfloat(2.0f);
    }
    auto prefill_q_buffer =
        create_shared_buffer(*device.device, prefill_q.size() * 2u);
    auto prefill_gate_buffer =
        create_shared_buffer(*device.device, prefill_gate.size() * 2u);
    auto prefill_key_buffer =
        create_shared_buffer(*device.device, prefill_keys.size() * 2u);
    auto prefill_value_buffer =
        create_shared_buffer(*device.device, prefill_values.size() * 2u);
    const std::size_t prefill_output_bytes =
        prefill_q.size() * 2u;
    const std::size_t partial_bytes =
        std::size_t{kQueryHeads} * kPrefillBlock *
        kPrefillPartitions * (kDim + 2u) * sizeof(float);
    const std::size_t score_bytes =
        std::size_t{kPrefillBlock} * kQueryHeads *
        kPrefillVisible * sizeof(float);
    auto prefill_partials =
        create_shared_buffer(*device.device, partial_bytes);
    auto prefill_exact_output =
        create_shared_buffer(*device.device, prefill_output_bytes);
    auto prefill_scores_first =
        create_shared_buffer(*device.device, score_bytes);
    auto prefill_scores_second =
        create_shared_buffer(*device.device, score_bytes);
    auto prefill_staged_first =
        create_shared_buffer(*device.device, prefill_output_bytes);
    auto prefill_staged_second =
        create_shared_buffer(*device.device, prefill_output_bytes);
    constexpr std::size_t kStreamingCanaryBytes = 256;
    constexpr unsigned char kStreamingCanary = 0xa5;
    auto prefill_streaming_first = create_shared_buffer(
        *device.device,
        prefill_output_bytes + 2u * kStreamingCanaryBytes);
    auto prefill_streaming_second = create_shared_buffer(
        *device.device,
        prefill_output_bytes + 2u * kStreamingCanaryBytes);
    const std::uint32_t prefill_constants[] = {
        kPrefillContext,
        kPrefillVisible,
        kCapacity,
        kPart,
        kPrefillPartitions,
        kPrefillBlock,
    };
    std::vector<MetalBufferResult> prefill_constant_buffers;
    for (const std::uint32_t value : prefill_constants) {
        auto buffer = create_shared_buffer(
            *device.device, sizeof(value));
        if (!buffer) {
            return 83;
        }
        std::memcpy(buffer.buffer->contents(), &value, sizeof(value));
        prefill_constant_buffers.push_back(std::move(buffer));
    }
    for (auto* result : {
             &prefill_q_buffer,
             &prefill_gate_buffer,
             &prefill_key_buffer,
             &prefill_value_buffer,
             &prefill_partials,
             &prefill_exact_output,
             &prefill_scores_first,
             &prefill_scores_second,
             &prefill_staged_first,
             &prefill_staged_second,
             &prefill_streaming_first,
             &prefill_streaming_second,
         }) {
        if (!*result) {
            return 83;
        }
    }
    upload<std::uint16_t>(*prefill_q_buffer.buffer, prefill_q);
    upload<std::uint16_t>(*prefill_gate_buffer.buffer, prefill_gate);
    upload<std::uint16_t>(*prefill_key_buffer.buffer, prefill_keys);
    upload<std::uint16_t>(*prefill_value_buffer.buffer, prefill_values);
    std::memset(
        prefill_streaming_first.buffer->contents(),
        kStreamingCanary,
        prefill_output_bytes + 2u * kStreamingCanaryBytes);
    std::memset(
        prefill_streaming_second.buffer->contents(),
        kStreamingCanary,
        prefill_output_bytes + 2u * kStreamingCanaryBytes);
    const MetalBuffer& prefill_context =
        *prefill_constant_buffers[0].buffer;
    const MetalBuffer& prefill_visible =
        *prefill_constant_buffers[1].buffer;
    const MetalBuffer& prefill_capacity =
        *prefill_constant_buffers[2].buffer;
    const MetalBuffer& prefill_partition =
        *prefill_constant_buffers[3].buffer;
    const MetalBuffer& prefill_partition_count =
        *prefill_constant_buffers[4].buffer;
    const MetalBuffer& prefill_block =
        *prefill_constant_buffers[5].buffer;
    const Pass exact_prefill_passes[] = {
        {&prefill_partial,
         {&*prefill_q_buffer.buffer,
          &*prefill_key_buffer.buffer,
          &*prefill_value_buffer.buffer,
          &prefill_context,
          &prefill_capacity,
          &prefill_partition,
          &*prefill_partials.buffer,
          &prefill_partition_count,
          &prefill_block},
         {0, 0, 0, 0, 0, 0, 0, 0, 0},
         {.width = kQueryHeads,
          .height = kPrefillPartitions,
          .depth = kPrefillBlock},
         {.width = kDim, .height = 1, .depth = 1}},
        {&prefill_combine,
         {&*prefill_partials.buffer,
          &*prefill_gate_buffer.buffer,
          &prefill_partition_count,
          &*prefill_exact_output.buffer,
          &prefill_block},
         {0, 0, 0, 0, 0},
         {.width = kQueryHeads,
          .height = kPrefillBlock,
          .depth = 1},
         {.width = kDim, .height = 1, .depth = 1}},
    };
    if (const int rc = run_batch(
            *queue.command_queue, exact_prefill_passes,
            "prefill-partial-combine");
        rc != 0) {
        return 83 + rc;
    }
    const auto run_staged =
        [&](MetalBuffer& score_buffer, MetalBuffer& output_buffer,
            const char* label) {
            const Pass passes[] = {
                {&staged_scores,
                 {&*prefill_q_buffer.buffer,
                  &*prefill_key_buffer.buffer,
                  &score_buffer,
                  &prefill_visible,
                  &prefill_capacity,
                  &prefill_block},
                 {0, 0, 0, 0, 0, 0},
                 {.width =
                      (kPrefillVisible +
                       generated::
                           kKernelLibraryPrefillStagedAttentionKeyTileColumns -
                       1u) /
                      generated::
                          kKernelLibraryPrefillStagedAttentionKeyTileColumns,
                  .height = 1,
                  .depth =
                      kQueryHeads *
                      ((kPrefillBlock +
                        generated::
                            kKernelLibraryPrefillStagedAttentionQueryTileRows -
                        1u) /
                       generated::
                           kKernelLibraryPrefillStagedAttentionQueryTileRows)},
                 {.width =
                      generated::
                          kKernelLibraryPrefillStagedAttentionThreads,
                  .height = 1,
                  .depth = 1}},
                {&staged_softmax,
                 {&score_buffer,
                  &prefill_visible,
                  &prefill_context},
                 {0, 0, 0},
                 {.width = kPrefillBlock,
                  .height = kQueryHeads,
                  .depth = 1},
                 {.width =
                      generated::
                          kKernelLibraryPrefillStagedAttentionSoftmaxThreads,
                  .height = 1,
                  .depth = 1}},
                {&staged_values,
                 {&score_buffer,
                  &*prefill_value_buffer.buffer,
                  &output_buffer,
                  &prefill_visible,
                  &prefill_capacity,
                  &prefill_block,
                  &*prefill_gate_buffer.buffer},
                 {0, 0, 0, 0, 0, 0, 0},
                 {.width =
                      kDim /
                      generated::
                          kKernelLibraryPrefillStagedAttentionOutputTileColumns,
                  .height = 1,
                  .depth =
                      kQueryHeads *
                      ((kPrefillBlock +
                        generated::
                            kKernelLibraryPrefillStagedAttentionQueryTileRows -
                        1u) /
                       generated::
                           kKernelLibraryPrefillStagedAttentionQueryTileRows)},
                 {.width =
                      generated::
                          kKernelLibraryPrefillStagedAttentionThreads,
                  .height = 1,
                  .depth = 1}},
            };
            return run_batch(
                *queue.command_queue, passes, label);
        };
    if (const int rc = run_staged(
            *prefill_scores_first.buffer,
            *prefill_staged_first.buffer,
            "prefill-staged-first");
        rc != 0) {
        return 91 + rc;
    }
    if (const int rc = run_staged(
            *prefill_scores_second.buffer,
            *prefill_staged_second.buffer,
            "prefill-staged-second");
        rc != 0) {
        return 99 + rc;
    }
    const auto exact_prefill = download<std::uint16_t>(
        *prefill_exact_output.buffer, prefill_q.size());
    const auto staged_prefill_first = download<std::uint16_t>(
        *prefill_staged_first.buffer, prefill_q.size());
    const auto staged_prefill_second = download<std::uint16_t>(
        *prefill_staged_second.buffer, prefill_q.size());
    if (staged_prefill_first != staged_prefill_second) {
        std::cout << "staged prefill determinism: FAIL\n";
        return 107;
    }
    float maximum_normalized_error = 0.0f;
    double squared_error = 0.0;
    double squared_reference = 0.0;
    for (std::size_t index = 0;
         index < exact_prefill.size(); ++index) {
        const float expected = f32_from_bf16(exact_prefill[index]);
        const float actual =
            f32_from_bf16(staged_prefill_first[index]);
        const float difference = std::abs(actual - expected);
        maximum_normalized_error = std::max(
            maximum_normalized_error,
            difference / (1.0f + std::abs(expected)));
        squared_error +=
            static_cast<double>(difference) * difference;
        squared_reference +=
            static_cast<double>(expected) * expected;
    }
    const double relative_l2 =
        squared_reference == 0.0
            ? (squared_error == 0.0 ? 0.0
                                    : std::numeric_limits<double>::infinity())
            : std::sqrt(squared_error / squared_reference);
    std::cout << "staged prefill: max_norm "
              << maximum_normalized_error << ", relative_l2 "
              << relative_l2 << ", deterministic yes\n";
    if (!std::isfinite(relative_l2) ||
        maximum_normalized_error > kMaximumNormalizedError ||
        relative_l2 > kMaximumNormalizedError) {
        return 108;
    }
    const auto run_streaming =
        [&](MetalBuffer& output_buffer, const char* label) {
            const Pass pass{
                &streaming,
                {
                    &*prefill_q_buffer.buffer,
                    &*prefill_key_buffer.buffer,
                    &*prefill_value_buffer.buffer,
                    &*prefill_gate_buffer.buffer,
                    &output_buffer,
                    &prefill_visible,
                    &prefill_capacity,
                    &prefill_block,
                    &prefill_context,
                },
                {
                    0,
                    0,
                    0,
                    0,
                    kStreamingCanaryBytes,
                    0,
                    0,
                    0,
                    0,
                },
                {
                    .width = kQueryHeads,
                    .height =
                        (kPrefillBlock +
                         generated::
                             kKernelLibraryPrefillStreamingAttentionQueryTileRows -
                         1u) /
                        generated::
                            kKernelLibraryPrefillStreamingAttentionQueryTileRows,
                    .depth = 1,
                },
                {
                    .width =
                        generated::
                            kKernelLibraryPrefillStreamingAttentionThreads,
                    .height = 1,
                    .depth = 1,
                },
            };
            return run_batch(
                *queue.command_queue,
                std::span<const Pass>(&pass, 1), label);
        };
    if (const int rc = run_streaming(
            *prefill_streaming_first.buffer,
            "prefill-streaming-first");
        rc != 0) {
        return 109 + rc;
    }
    if (const int rc = run_streaming(
            *prefill_streaming_second.buffer,
            "prefill-streaming-second");
        rc != 0) {
        return 117 + rc;
    }
    const auto canaries_intact =
        [&](const MetalBuffer& buffer) {
            const auto* bytes =
                static_cast<const unsigned char*>(
                    buffer.contents());
            return std::all_of(
                       bytes,
                       bytes + kStreamingCanaryBytes,
                       [](unsigned char value) {
                           return value == kStreamingCanary;
                       }) &&
                   std::all_of(
                       bytes + kStreamingCanaryBytes +
                           prefill_output_bytes,
                       bytes + 2u * kStreamingCanaryBytes +
                           prefill_output_bytes,
                       [](unsigned char value) {
                           return value == kStreamingCanary;
                       });
        };
    if (!canaries_intact(*prefill_streaming_first.buffer) ||
        !canaries_intact(*prefill_streaming_second.buffer)) {
        std::cout << "streaming prefill canary: FAIL\n";
        return 125;
    }
    const auto streaming_prefill_first =
        download<std::uint16_t>(
            *prefill_streaming_first.buffer, prefill_q.size(),
            kStreamingCanaryBytes / sizeof(std::uint16_t));
    const auto streaming_prefill_second =
        download<std::uint16_t>(
            *prefill_streaming_second.buffer, prefill_q.size(),
            kStreamingCanaryBytes / sizeof(std::uint16_t));
    if (streaming_prefill_first != streaming_prefill_second) {
        std::cout << "streaming prefill determinism: FAIL\n";
        return 126;
    }
    const auto* first_streaming_bytes =
        static_cast<const unsigned char*>(
            prefill_streaming_first.buffer->contents()) +
        kStreamingCanaryBytes;
    if (std::all_of(
            first_streaming_bytes,
            first_streaming_bytes + prefill_output_bytes,
            [](unsigned char value) {
                return value == kStreamingCanary;
            })) {
        std::cout << "streaming prefill vacuous output: FAIL\n";
        return 127;
    }
    if (download<std::uint16_t>(
            *prefill_q_buffer.buffer, prefill_q.size()) !=
            prefill_q ||
        download<std::uint16_t>(
            *prefill_gate_buffer.buffer, prefill_gate.size()) !=
            prefill_gate ||
        download<std::uint16_t>(
            *prefill_key_buffer.buffer, prefill_keys.size()) !=
            prefill_keys ||
        download<std::uint16_t>(
            *prefill_value_buffer.buffer,
            prefill_values.size()) != prefill_values) {
        std::cout << "streaming prefill read-only input: FAIL\n";
        return 128;
    }
    float streaming_maximum_normalized_error = 0.0f;
    double streaming_squared_error = 0.0;
    for (std::size_t index = 0;
         index < exact_prefill.size(); ++index) {
        const float expected =
            f32_from_bf16(exact_prefill[index]);
        const float actual =
            f32_from_bf16(streaming_prefill_first[index]);
        const float difference = std::abs(actual - expected);
        streaming_maximum_normalized_error = std::max(
            streaming_maximum_normalized_error,
            difference / (1.0f + std::abs(expected)));
        streaming_squared_error +=
            static_cast<double>(difference) * difference;
    }
    const double streaming_relative_l2 =
        squared_reference == 0.0
            ? (streaming_squared_error == 0.0
                   ? 0.0
                   : std::numeric_limits<double>::infinity())
            : std::sqrt(
                  streaming_squared_error /
                  squared_reference);
    std::cout << "streaming prefill: max_norm "
              << streaming_maximum_normalized_error
              << ", relative_l2 " << streaming_relative_l2
              << ", deterministic yes, canaries intact, inputs unchanged\n";
    if (!std::isfinite(streaming_relative_l2) ||
        streaming_maximum_normalized_error >
            kMaximumNormalizedError ||
        streaming_relative_l2 > kMaximumNormalizedError) {
        return 129;
    }

    // attn_project: device-defined q4 wiring proof over concatenated tables.
    const std::uint32_t proj_rows = kProjRows;
    std::vector<std::uint32_t> pw(std::size_t{proj_rows} * (generated::kKernelLibraryHidden / 8));
    std::vector<std::uint16_t> ps(std::size_t{proj_rows} * (generated::kKernelLibraryHidden / 64));
    std::vector<std::uint16_t> pb(ps.size());
    std::vector<std::uint16_t> px(generated::kKernelLibraryHidden);
    for (auto& w : pw) {
        w = static_cast<std::uint32_t>(generator.next());
    }
    for (std::size_t i = 0; i < ps.size(); ++i) {
        ps[i] = generator.next_banded_bf16();
        pb[i] = generator.next_banded_bf16();
    }
    for (auto& x : px) {
        x = generator.next_banded_bf16();
    }
    const std::uint32_t qg_rows = kQueryHeads * 2 * kDim;
    const std::uint32_t k_rows = kKvHeads * kDim;
    auto dev_px = create_shared_buffer(*device.device, px.size() * 2);
    auto dev_pw = create_shared_buffer(*device.device, pw.size() * 4);
    auto dev_ps = create_shared_buffer(*device.device, ps.size() * 2);
    auto dev_pb = create_shared_buffer(*device.device, pb.size() * 2);
    auto dev_po = create_shared_buffer(*device.device, proj_rows * 2);
    auto dev_pq = create_shared_buffer(*device.device, proj_rows * 4);
    for (auto* result : {&dev_px, &dev_pw, &dev_ps, &dev_pb, &dev_po, &dev_pq}) {
        if (!*result) {
            return 15;
        }
    }
    upload<std::uint16_t>(*dev_px.buffer, px);
    upload<std::uint32_t>(*dev_pw.buffer, pw);
    upload<std::uint16_t>(*dev_ps.buffer, ps);
    upload<std::uint16_t>(*dev_pb.buffer, pb);
    {
        const std::uint64_t wpr = generated::kKernelLibraryHidden / 8;
        const std::uint64_t gpr = generated::kKernelLibraryHidden / 64;
        const Pass passes[] = {
            {&adj_q4,
             {&*dev_px.buffer, &*dev_pw.buffer, &*dev_ps.buffer, &*dev_pb.buffer, &*dev_pq.buffer},
             {0, 0, 0, 0, 0},
             {.width = proj_rows, .height = 1, .depth = 1},
             {.width = 32, .height = 1, .depth = 1}},
            {&project,
             {&*dev_px.buffer, &*dev_pw.buffer, &*dev_ps.buffer, &*dev_pb.buffer, &*dev_pw.buffer,
              &*dev_ps.buffer, &*dev_pb.buffer, &*dev_pw.buffer, &*dev_ps.buffer, &*dev_pb.buffer,
              &*dev_po.buffer},
             {0, 0, 0, 0, std::uint64_t{qg_rows} * wpr * 4, std::uint64_t{qg_rows} * gpr * 2,
              std::uint64_t{qg_rows} * gpr * 2, std::uint64_t{qg_rows + k_rows} * wpr * 4,
              std::uint64_t{qg_rows + k_rows} * gpr * 2, std::uint64_t{qg_rows + k_rows} * gpr * 2,
              0},
             {.width = (proj_rows * 32u + 255u) / 256u, .height = 1, .depth = 1},
             {.width = 256, .height = 1, .depth = 1}},
        };
        if (const int rc = run_batch(*queue.command_queue, passes, "attn-project"); rc != 0) {
            return 90 + rc;
        }
    }
    const auto project_q4 = download<float>(*dev_pq.buffer, proj_rows);
    const auto project_out = download<std::uint16_t>(*dev_po.buffer, proj_rows);
    std::uint32_t project_bad = 0;
    for (std::uint32_t row = 0; row < proj_rows; ++row) {
        const std::uint16_t expected =
            flush_subnormal_bf16(bf16_from_f32(flush_subnormal_f32(project_q4[row])));
        if (expected != project_out[row]) {
            if (project_bad < 4) {
                std::cout << "  attn-project mismatch at " << row << ": expected 0x" << std::hex
                          << expected << " actual 0x" << project_out[row] << std::dec << '\n';
            }
            ++project_bad;
        }
    }
    std::cout << "attn_project fixtures: mismatches " << project_bad << '\n';
    if (project_bad != 0) {
        return 95;
    }

    std::cout << "attention fixtures: PASS\n"
              << "  device: " << device.device->name() << '\n'
              << "  rope chain: " << (rope_matches[0] ? "fused" : "separate")
              << (rope_matches[0] && rope_matches[1] ? " (both)" : "") << '\n'
              << "  decode chain: " << (decode_matches[0] ? "fused" : "separate")
              << (decode_matches[0] && decode_matches[1] ? " (both)" : "") << '\n'
              << "  command buffers submitted: " << submissions << '\n';
    return 0;
}
