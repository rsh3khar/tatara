#include "fixture_batteries.h"
#include "kernel_reference.h"
#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/generated/kernel_library.h"

#include <bit>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <span>
#include <vector>

namespace {

using namespace tatara::backend::metal;
using namespace tatara::testing;

constexpr std::uint32_t kHidden = generated::kKernelLibraryHidden;
constexpr std::uint32_t kExperts = generated::kKernelLibraryMoeExperts;
constexpr std::uint32_t kActive = generated::kKernelLibraryMoeActiveExperts;
constexpr std::uint32_t kExpertDim = generated::kKernelLibraryMoeExpertDimension;
constexpr std::uint32_t kSlots = kActive + 1;
constexpr std::uint32_t kGroupSize = generated::kKernelLibraryGroupSize;
constexpr std::uint32_t kQ4WordsHidden = kHidden / 8;
constexpr std::uint32_t kQ4GroupsHidden = kHidden / kGroupSize;
constexpr std::uint32_t kQ8WordsHidden = kHidden / 4;
constexpr std::uint32_t kQ4WordsExpert = kExpertDim / 8;
constexpr std::uint32_t kQ4GroupsExpert = kExpertDim / kGroupSize;
constexpr std::uint32_t kRouterRows = kExperts + 1;
constexpr std::uint32_t kUpgateRows = kSlots * kExpertDim;
constexpr std::uint32_t kSentinelTop = 6;
constexpr float kEpsilon = 1e-6f;
constexpr MoeGeometry kGeometry{
    .experts = kExperts, .active_experts = kActive, .expert_dimension = kExpertDim};

int submissions = 0;

template <typename Value> void upload(const MetalBuffer& buffer, std::span<const Value> values) {
    std::memcpy(buffer.contents(), values.data(), values.size() * sizeof(Value));
}

template <typename Value>
std::vector<Value> download(const MetalBuffer& buffer, std::size_t count) {
    std::vector<Value> values(count);
    std::memcpy(values.data(), buffer.contents(), count * sizeof(Value));
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

std::uint32_t mismatches_f32(std::span<const float> expected, std::span<const float> actual,
                             const char* label) {
    std::uint32_t count = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::bit_cast<std::uint32_t>(expected[i]) != std::bit_cast<std::uint32_t>(actual[i])) {
            if (count < 4) {
                std::cout << "  " << label << " mismatch at " << i << ": expected " << expected[i]
                          << " actual " << actual[i] << '\n';
            }
            ++count;
        }
    }
    return count;
}

} // namespace

int run_moe_battery() {
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
        {"residual_rms", {}},          {"router_q8", {}},
        {"router_select", {}},         {"grouped_upgate", {}},
        {"grouped_down_res", {}},      {"adjudicate_rsqrt", {}},
        {"adjudicate_rms_sum", {}},    {"adjudicate_f32_exp", {}},
        {"adjudicate_q8_row", {}},     {"adjudicate_q4p_row", {}},
        {"adjudicate_q4p_row_e", {}},  {"adjudicate_down_total", {}},
        {"adjudicate_f32_divide", {}},
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
    const auto& residual_rms = kernels[0].pipeline;
    const auto& router_q8 = kernels[1].pipeline;
    const auto& router_select = kernels[2].pipeline;
    const auto& grouped_upgate = kernels[3].pipeline;
    const auto& grouped_down_res = kernels[4].pipeline;
    const auto& adj_rsqrt = kernels[5].pipeline;
    const auto& adj_rms_sum = kernels[6].pipeline;
    const auto& adj_exp = kernels[7].pipeline;
    const auto& adj_q8 = kernels[8].pipeline;
    const auto& adj_q4p = kernels[9].pipeline;
    const auto& adj_q4pe = kernels[10].pipeline;
    const auto& adj_down_total = kernels[11].pipeline;
    const auto& adj_div = kernels[12].pipeline;
    std::uint64_t divide_ieee = 0;
    std::uint64_t divide_reciprocal = 0;
    std::uint64_t divide_pairs = 0;
    const auto tally_divisions = [&](std::span<const float> numerators,
                                     std::span<const float> denominators,
                                     std::span<const float> quotients) {
        for (std::size_t i = 0; i < quotients.size(); ++i) {
            const std::uint32_t observed = std::bit_cast<std::uint32_t>(quotients[i]);
            divide_ieee +=
                std::bit_cast<std::uint32_t>(f32_divide_candidate(
                    numerators[i], denominators[i], DivisionModel::IeeeDivision)) == observed
                    ? 1
                    : 0;
            divide_reciprocal +=
                std::bit_cast<std::uint32_t>(f32_divide_candidate(
                    numerators[i], denominators[i], DivisionModel::ReciprocalMultiply)) == observed
                    ? 1
                    : 0;
            ++divide_pairs;
        }
    };

    // Fixture inputs. Banded values keep every chained square and dot finite;
    // extreme-magnitude adds are already pinned by the EXEC2 sweep. Edges
    // cover signed zero, bfloat16 subnormals, exact cancellation, and the
    // quantization nibble extremes on always-active rows.
    Xorshift64Star generator(0x40E7C4E4ull);
    std::vector<std::uint16_t> input_a(kHidden), input_b(kHidden), rms_weight(kHidden);
    for (std::uint32_t i = 0; i < kHidden; ++i) {
        input_a[i] = generator.next_banded_bf16();
        input_b[i] = generator.next_banded_bf16();
        rms_weight[i] = generator.next_banded_bf16();
    }
    input_a[0] = 0x0000;
    input_b[0] = 0x8000;
    input_a[1] = 0x0001;
    input_b[1] = 0x0080;
    input_a[2] = 0x4300;
    input_b[2] = 0xC300;
    input_a[3] = 0x4B00;
    input_b[3] = 0x8000;
    rms_weight[0] = 0x8000;
    rms_weight[1] = 0x0000;

    std::vector<std::uint32_t> router_words(std::size_t{kExperts} * kQ8WordsHidden);
    std::vector<std::uint16_t> router_scales(std::size_t{kExperts} * kQ4GroupsHidden);
    std::vector<std::uint16_t> router_biases(router_scales.size());
    for (auto& word : router_words) {
        word = static_cast<std::uint32_t>(generator.next());
    }
    for (std::size_t i = 0; i < router_scales.size(); ++i) {
        router_scales[i] = generator.next_banded_bf16();
        router_biases[i] = generator.next_banded_bf16();
    }
    std::vector<std::uint32_t> shared_router_words(kQ8WordsHidden);
    std::vector<std::uint16_t> shared_router_scales(kQ4GroupsHidden);
    std::vector<std::uint16_t> shared_router_biases(kQ4GroupsHidden);
    for (auto& word : shared_router_words) {
        word = static_cast<std::uint32_t>(generator.next());
    }
    for (std::size_t i = 0; i < shared_router_scales.size(); ++i) {
        shared_router_scales[i] = generator.next_banded_bf16();
        shared_router_biases[i] = generator.next_banded_bf16();
    }
    // Quantization extremes on rows the router always dots.
    router_words[0] = 0x00000000u;
    router_words[1] = 0xFFFFFFFFu;
    router_scales[0] = 0x0000;
    router_biases[1] = 0x8000;
    shared_router_words[0] = 0xFFFFFFFFu;

    const std::size_t expert_word_count = std::size_t{kExperts} * kExpertDim * kQ4WordsHidden;
    const std::size_t expert_group_count = std::size_t{kExperts} * kExpertDim * kQ4GroupsHidden;
    std::vector<std::uint32_t> gate_words(expert_word_count), up_words(expert_word_count);
    std::vector<std::uint16_t> gate_scales(expert_group_count), gate_biases(expert_group_count);
    std::vector<std::uint16_t> up_scales(expert_group_count), up_biases(expert_group_count);
    for (std::size_t i = 0; i < expert_word_count; ++i) {
        gate_words[i] = static_cast<std::uint32_t>(generator.next());
        up_words[i] = static_cast<std::uint32_t>(generator.next());
    }
    for (std::size_t i = 0; i < expert_group_count; ++i) {
        gate_scales[i] = generator.next_banded_bf16();
        gate_biases[i] = generator.next_banded_bf16();
        up_scales[i] = generator.next_banded_bf16();
        up_biases[i] = generator.next_banded_bf16();
    }
    std::vector<std::uint32_t> shared_gate_words(std::size_t{kExpertDim} * kQ4WordsHidden);
    std::vector<std::uint32_t> shared_up_words(shared_gate_words.size());
    std::vector<std::uint16_t> shared_gate_scales(std::size_t{kExpertDim} * kQ4GroupsHidden);
    std::vector<std::uint16_t> shared_gate_biases(shared_gate_scales.size());
    std::vector<std::uint16_t> shared_up_scales(shared_gate_scales.size());
    std::vector<std::uint16_t> shared_up_biases(shared_gate_scales.size());
    for (std::size_t i = 0; i < shared_gate_words.size(); ++i) {
        shared_gate_words[i] = static_cast<std::uint32_t>(generator.next());
        shared_up_words[i] = static_cast<std::uint32_t>(generator.next());
    }
    for (std::size_t i = 0; i < shared_gate_scales.size(); ++i) {
        shared_gate_scales[i] = generator.next_banded_bf16();
        shared_gate_biases[i] = generator.next_banded_bf16();
        shared_up_scales[i] = generator.next_banded_bf16();
        shared_up_biases[i] = generator.next_banded_bf16();
    }
    shared_gate_words[0] = 0x00000000u;
    shared_up_words[0] = 0xFFFFFFFFu;
    shared_gate_scales[0] = 0x8000;

    const std::size_t down_word_count = std::size_t{kExperts} * kHidden * kQ4WordsExpert;
    const std::size_t down_group_count = std::size_t{kExperts} * kHidden * kQ4GroupsExpert;
    std::vector<std::uint32_t> down_words(down_word_count);
    std::vector<std::uint16_t> down_scales(down_group_count), down_biases(down_group_count);
    for (auto& word : down_words) {
        word = static_cast<std::uint32_t>(generator.next());
    }
    for (std::size_t i = 0; i < down_group_count; ++i) {
        down_scales[i] = generator.next_banded_bf16();
        down_biases[i] = generator.next_banded_bf16();
    }
    std::vector<std::uint32_t> shared_down_words(std::size_t{kHidden} * kQ4WordsExpert);
    std::vector<std::uint16_t> shared_down_scales(std::size_t{kHidden} * kQ4GroupsExpert);
    std::vector<std::uint16_t> shared_down_biases(shared_down_scales.size());
    for (auto& word : shared_down_words) {
        word = static_cast<std::uint32_t>(generator.next());
    }
    for (std::size_t i = 0; i < shared_down_scales.size(); ++i) {
        shared_down_scales[i] = generator.next_banded_bf16();
        shared_down_biases[i] = generator.next_banded_bf16();
    }
    shared_down_words[0] = 0xFFFFFFFFu;

    // The exact-tie selection case: identical float logits at tree-distant
    // indices, a secondary tie pair for a later pass, and a shared logit.
    std::vector<float> tie_logits(kRouterRows, 0.0f);
    for (std::uint32_t e = 0; e < kExperts; ++e) {
        tie_logits[e] = -1.0f - 0.25f * static_cast<float>(e % 7u);
    }
    tie_logits[3] = 1.5f;
    tie_logits[200] = 1.5f;
    tie_logits[10] = 1.25f;
    tie_logits[130] = 1.25f;
    tie_logits[kExperts] = 0.75f;

    // Device buffers.
    auto dev_a = create_shared_buffer(*device.device, kHidden * 2);
    auto dev_b = create_shared_buffer(*device.device, kHidden * 2);
    auto dev_res = create_shared_buffer(*device.device, kHidden * 2);
    auto dev_rms_w = create_shared_buffer(*device.device, kHidden * 2);
    auto dev_normed = create_shared_buffer(*device.device, kHidden * 2);
    auto dev_rw = create_shared_buffer(*device.device, router_words.size() * 4);
    auto dev_rs = create_shared_buffer(*device.device, router_scales.size() * 2);
    auto dev_rb = create_shared_buffer(*device.device, router_biases.size() * 2);
    auto dev_srw = create_shared_buffer(*device.device, shared_router_words.size() * 4);
    auto dev_srs = create_shared_buffer(*device.device, shared_router_scales.size() * 2);
    auto dev_srb = create_shared_buffer(*device.device, shared_router_biases.size() * 2);
    auto dev_q8_obs_w =
        create_shared_buffer(*device.device, std::size_t{kRouterRows} * kQ8WordsHidden * 4);
    auto dev_q8_obs_s =
        create_shared_buffer(*device.device, std::size_t{kRouterRows} * kQ4GroupsHidden * 2);
    auto dev_q8_obs_b =
        create_shared_buffer(*device.device, std::size_t{kRouterRows} * kQ4GroupsHidden * 2);
    auto dev_q8_obs_out = create_shared_buffer(*device.device, kRouterRows * 4);
    auto dev_logits = create_shared_buffer(*device.device, kRouterRows * 4);
    auto dev_tie_logits = create_shared_buffer(*device.device, kRouterRows * 4);
    auto dev_gw = create_shared_buffer(*device.device, gate_words.size() * 4);
    auto dev_gs = create_shared_buffer(*device.device, gate_scales.size() * 2);
    auto dev_gb = create_shared_buffer(*device.device, gate_biases.size() * 2);
    auto dev_uw = create_shared_buffer(*device.device, up_words.size() * 4);
    auto dev_us = create_shared_buffer(*device.device, up_scales.size() * 2);
    auto dev_ub = create_shared_buffer(*device.device, up_biases.size() * 2);
    auto dev_sgw = create_shared_buffer(*device.device, shared_gate_words.size() * 4);
    auto dev_sgs = create_shared_buffer(*device.device, shared_gate_scales.size() * 2);
    auto dev_sgb = create_shared_buffer(*device.device, shared_gate_biases.size() * 2);
    auto dev_suw = create_shared_buffer(*device.device, shared_up_words.size() * 4);
    auto dev_sus = create_shared_buffer(*device.device, shared_up_scales.size() * 2);
    auto dev_sub = create_shared_buffer(*device.device, shared_up_biases.size() * 2);
    auto dev_dw = create_shared_buffer(*device.device, down_words.size() * 4);
    auto dev_ds = create_shared_buffer(*device.device, down_scales.size() * 2);
    auto dev_db = create_shared_buffer(*device.device, down_biases.size() * 2);
    auto dev_sdw = create_shared_buffer(*device.device, shared_down_words.size() * 4);
    auto dev_sds = create_shared_buffer(*device.device, shared_down_scales.size() * 2);
    auto dev_sdb = create_shared_buffer(*device.device, shared_down_biases.size() * 2);
    for (auto* result :
         {&dev_a,      &dev_b,          &dev_res,      &dev_rms_w,    &dev_normed,
          &dev_rw,     &dev_rs,         &dev_rb,       &dev_srw,      &dev_srs,
          &dev_srb,    &dev_q8_obs_w,   &dev_q8_obs_s, &dev_q8_obs_b, &dev_q8_obs_out,
          &dev_logits, &dev_tie_logits, &dev_gw,       &dev_gs,       &dev_gb,
          &dev_uw,     &dev_us,         &dev_ub,       &dev_sgw,      &dev_sgs,
          &dev_sgb,    &dev_suw,        &dev_sus,      &dev_sub,      &dev_dw,
          &dev_ds,     &dev_db,         &dev_sdw,      &dev_sds,      &dev_sdb}) {
        if (!*result) {
            return 6;
        }
    }
    struct SelectCase {
        const char* name;
        MetalBufferResult ids;
        MetalBufferResult coefficients;
        MetalBufferResult shared_coefficient;
        MetalBufferResult top_n;
    };
    SelectCase select_cases[3] = {
        {"chained", {}, {}, {}, {}}, {"exact-tie", {}, {}, {}, {}}, {"sentinel", {}, {}, {}, {}}};
    const std::uint32_t top_values[3] = {kActive, kActive, kSentinelTop};
    for (std::size_t c = 0; c < 3; ++c) {
        select_cases[c].ids = create_shared_buffer(*device.device, kActive * 4);
        select_cases[c].coefficients = create_shared_buffer(*device.device, kActive * 4);
        select_cases[c].shared_coefficient = create_shared_buffer(*device.device, 4);
        select_cases[c].top_n = create_shared_buffer(*device.device, 4);
        if (!select_cases[c].ids || !select_cases[c].coefficients ||
            !select_cases[c].shared_coefficient || !select_cases[c].top_n) {
            return 6;
        }
        std::memcpy(select_cases[c].top_n.buffer->contents(), &top_values[c], 4);
    }
    auto dev_gate_obs = create_shared_buffer(*device.device, std::size_t{kUpgateRows} * 4);
    auto dev_up_obs = create_shared_buffer(*device.device, std::size_t{kUpgateRows} * 4);
    auto dev_h_full = create_shared_buffer(*device.device, std::size_t{kUpgateRows} * 2);
    auto dev_h_sentinel = create_shared_buffer(*device.device, std::size_t{kUpgateRows} * 2);
    auto dev_parts_full = create_shared_buffer(*device.device, std::size_t{kSlots} * kHidden * 4);
    auto dev_parts_sentinel =
        create_shared_buffer(*device.device, std::size_t{kSlots} * kHidden * 4);
    auto dev_total_full = create_shared_buffer(*device.device, kHidden * 4);
    auto dev_total_sentinel = create_shared_buffer(*device.device, kHidden * 4);
    auto dev_moe_full = create_shared_buffer(*device.device, kHidden * 2);
    auto dev_moe_sentinel = create_shared_buffer(*device.device, kHidden * 2);
    auto dev_layer_full = create_shared_buffer(*device.device, kHidden * 2);
    auto dev_layer_sentinel = create_shared_buffer(*device.device, kHidden * 2);
    for (auto* result : {&dev_gate_obs, &dev_up_obs, &dev_h_full, &dev_h_sentinel, &dev_parts_full,
                         &dev_parts_sentinel, &dev_total_full, &dev_total_sentinel, &dev_moe_full,
                         &dev_moe_sentinel, &dev_layer_full, &dev_layer_sentinel}) {
        if (!*result) {
            return 6;
        }
    }

    upload<std::uint16_t>(*dev_a.buffer, input_a);
    upload<std::uint16_t>(*dev_b.buffer, input_b);
    upload<std::uint16_t>(*dev_rms_w.buffer, rms_weight);
    upload<std::uint32_t>(*dev_rw.buffer, router_words);
    upload<std::uint16_t>(*dev_rs.buffer, router_scales);
    upload<std::uint16_t>(*dev_rb.buffer, router_biases);
    upload<std::uint32_t>(*dev_srw.buffer, shared_router_words);
    upload<std::uint16_t>(*dev_srs.buffer, shared_router_scales);
    upload<std::uint16_t>(*dev_srb.buffer, shared_router_biases);
    upload<float>(*dev_tie_logits.buffer, tie_logits);
    upload<std::uint32_t>(*dev_gw.buffer, gate_words);
    upload<std::uint16_t>(*dev_gs.buffer, gate_scales);
    upload<std::uint16_t>(*dev_gb.buffer, gate_biases);
    upload<std::uint32_t>(*dev_uw.buffer, up_words);
    upload<std::uint16_t>(*dev_us.buffer, up_scales);
    upload<std::uint16_t>(*dev_ub.buffer, up_biases);
    upload<std::uint32_t>(*dev_sgw.buffer, shared_gate_words);
    upload<std::uint16_t>(*dev_sgs.buffer, shared_gate_scales);
    upload<std::uint16_t>(*dev_sgb.buffer, shared_gate_biases);
    upload<std::uint32_t>(*dev_suw.buffer, shared_up_words);
    upload<std::uint16_t>(*dev_sus.buffer, shared_up_scales);
    upload<std::uint16_t>(*dev_sub.buffer, shared_up_biases);
    upload<std::uint32_t>(*dev_dw.buffer, down_words);
    upload<std::uint16_t>(*dev_ds.buffer, down_scales);
    upload<std::uint16_t>(*dev_db.buffer, down_biases);
    upload<std::uint32_t>(*dev_sdw.buffer, shared_down_words);
    upload<std::uint16_t>(*dev_sds.buffer, shared_down_scales);
    upload<std::uint16_t>(*dev_sdb.buffer, shared_down_biases);
    // Concatenated 257-row observation copy of the router table, shared last.
    {
        std::vector<std::uint32_t> concat_words(std::size_t{kRouterRows} * kQ8WordsHidden);
        std::vector<std::uint16_t> concat_scales(std::size_t{kRouterRows} * kQ4GroupsHidden);
        std::vector<std::uint16_t> concat_biases(concat_scales.size());
        std::memcpy(concat_words.data(), router_words.data(), router_words.size() * 4);
        std::memcpy(concat_words.data() + router_words.size(), shared_router_words.data(),
                    shared_router_words.size() * 4);
        std::memcpy(concat_scales.data(), router_scales.data(), router_scales.size() * 2);
        std::memcpy(concat_scales.data() + router_scales.size(), shared_router_scales.data(),
                    shared_router_scales.size() * 2);
        std::memcpy(concat_biases.data(), router_biases.data(), router_biases.size() * 2);
        std::memcpy(concat_biases.data() + router_biases.size(), shared_router_biases.data(),
                    shared_router_biases.size() * 2);
        upload<std::uint32_t>(*dev_q8_obs_w.buffer, concat_words);
        upload<std::uint16_t>(*dev_q8_obs_s.buffer, concat_scales);
        upload<std::uint16_t>(*dev_q8_obs_b.buffer, concat_biases);
    }

    // Stage A: residual_rms. The rounded sums and the rsqrt argument are
    // CPU-derived from the pinned add and tree; the inverse is queried.
    std::vector<std::uint16_t> rounded(kHidden);
    residual_rms_rounded_sums(input_a, input_b, rounded);
    const float rms_argument =
        residual_rms_argument(rounded, kHidden, kEpsilon, SimdTreeShape::AdjacentPairs);
    std::vector<float> rsqrt_args(256, 1.0f);
    rsqrt_args[0] = rms_argument;
    auto dev_rsq_i = create_shared_buffer(*device.device, rsqrt_args.size() * 4);
    auto dev_rsq_o = create_shared_buffer(*device.device, rsqrt_args.size() * 4);
    auto dev_rounded = create_shared_buffer(*device.device, kHidden * 2);
    auto dev_rms_total = create_shared_buffer(*device.device, 4);
    if (!dev_rsq_i || !dev_rsq_o || !dev_rounded || !dev_rms_total) {
        return 6;
    }
    upload<float>(*dev_rsq_i.buffer, rsqrt_args);
    upload<std::uint16_t>(*dev_rounded.buffer, rounded);
    {
        const Pass passes[] = {
            {&adj_rsqrt,
             {&*dev_rsq_i.buffer, &*dev_rsq_o.buffer},
             {0, 0},
             {.width = 1, .height = 1, .depth = 1},
             {.width = 256, .height = 1, .depth = 1}},
            {&adj_rms_sum,
             {&*dev_rounded.buffer, &*dev_rms_total.buffer},
             {0, 0},
             {.width = 1, .height = 1, .depth = 1},
             {.width = kHidden / 4, .height = 1, .depth = 1}},
        };
        if (const int rc = run_batch(*queue.command_queue, passes, "stage-a queries"); rc != 0) {
            return 10 + rc;
        }
    }
    const float inverse = download<float>(*dev_rsq_o.buffer, 1)[0];
    const float device_rms_total = download<float>(*dev_rms_total.buffer, 1)[0];
    const float oracle_rms_total =
        rms_total_of_squares(rounded, kHidden, SimdTreeShape::AdjacentPairs);
    if (std::bit_cast<std::uint32_t>(device_rms_total) !=
        std::bit_cast<std::uint32_t>(oracle_rms_total)) {
        std::cout << "residual reduction total mismatch: oracle " << oracle_rms_total << " device "
                  << device_rms_total << '\n';
        return 19;
    }
    {
        const Pass passes[] = {{&residual_rms,
                                {&*dev_a.buffer, &*dev_b.buffer, &*dev_res.buffer,
                                 &*dev_rms_w.buffer, &*dev_normed.buffer},
                                {0, 0, 0, 0, 0},
                                {.width = 1, .height = 1, .depth = 1},
                                {.width = kHidden / 4, .height = 1, .depth = 1}}};
        if (const int rc = run_batch(*queue.command_queue, passes, "residual_rms"); rc != 0) {
            return 10 + rc;
        }
    }
    const auto device_res = download<std::uint16_t>(*dev_res.buffer, kHidden);
    const auto device_normed = download<std::uint16_t>(*dev_normed.buffer, kHidden);
    std::vector<std::uint16_t> expected_normed(kHidden);
    rms_only_reference_with_inverse(rounded, rms_weight, kHidden, inverse, expected_normed);
    const std::uint32_t residual_bad = mismatches_u16(rounded, device_res, "residual");
    const std::uint32_t normed_bad = mismatches_u16(expected_normed, device_normed, "normed");
    std::cout << "residual_rms fixtures: residual mismatches " << residual_bad
              << ", normed mismatches " << normed_bad << '\n';
    if (residual_bad != 0 || normed_bad != 0) {
        return 18;
    }

    // Stage B: router_q8 against the same-source 257-row observation.
    {
        const Pass passes[] = {{&adj_q8,
                                {&*dev_normed.buffer, &*dev_q8_obs_w.buffer, &*dev_q8_obs_s.buffer,
                                 &*dev_q8_obs_b.buffer, &*dev_q8_obs_out.buffer},
                                {0, 0, 0, 0, 0},
                                {.width = kRouterRows, .height = 1, .depth = 1},
                                {.width = 32, .height = 1, .depth = 1}}};
        if (const int rc = run_batch(*queue.command_queue, passes, "q8 observation"); rc != 0) {
            return 20 + rc;
        }
    }
    {
        const Pass passes[] = {
            {&router_q8,
             {&*dev_normed.buffer, &*dev_rw.buffer, &*dev_rs.buffer, &*dev_rb.buffer,
              &*dev_srw.buffer, &*dev_srs.buffer, &*dev_srb.buffer, &*dev_logits.buffer},
             {0, 0, 0, 0, 0, 0, 0, 0},
             {.width = kRouterRows, .height = 1, .depth = 1},
             {.width = 32, .height = 1, .depth = 1}}};
        if (const int rc = run_batch(*queue.command_queue, passes, "router_q8"); rc != 0) {
            return 20 + rc;
        }
    }
    const auto q8_observed = download<float>(*dev_q8_obs_out.buffer, kRouterRows);
    const auto device_logits = download<float>(*dev_logits.buffer, kRouterRows);
    std::vector<float> expected_logits(kRouterRows);
    for (std::uint32_t row = 0; row < kRouterRows; ++row) {
        expected_logits[row] = f32_from_bf16(flush_subnormal_bf16(bf16_from_f32(q8_observed[row])));
    }
    const std::uint32_t logit_bad = mismatches_f32(expected_logits, device_logits, "logit");
    std::cout << "router_q8 fixtures: mismatches " << logit_bad << '\n';
    if (logit_bad != 0) {
        return 29;
    }

    // Stage C: selection exponentials for the chained and tie logits, then
    // the three selection cases (the sentinel case reuses the chained
    // exponentials with top_n below the slot count).
    std::vector<float> exp_args;
    router_select_arguments(device_logits, kExperts, exp_args);
    std::vector<float> tie_args;
    router_select_arguments(tie_logits, kExperts, tie_args);
    std::vector<float> select_exp_args(exp_args);
    select_exp_args.insert(select_exp_args.end(), tie_args.begin(), tie_args.end());
    const std::size_t select_exp_count = select_exp_args.size();
    select_exp_args.resize(((select_exp_count + 255) / 256) * 256, 0.0f);
    auto dev_sel_exp_i = create_shared_buffer(*device.device, select_exp_args.size() * 4);
    auto dev_sel_exp_o = create_shared_buffer(*device.device, select_exp_args.size() * 4);
    if (!dev_sel_exp_i || !dev_sel_exp_o) {
        return 6;
    }
    upload<float>(*dev_sel_exp_i.buffer, select_exp_args);
    {
        const Pass passes[] = {{&adj_exp,
                                {&*dev_sel_exp_i.buffer, &*dev_sel_exp_o.buffer},
                                {0, 0},
                                {.width = static_cast<std::uint32_t>(select_exp_args.size()) / 256,
                                 .height = 1,
                                 .depth = 1},
                                {.width = 256, .height = 1, .depth = 1}}};
        if (const int rc = run_batch(*queue.command_queue, passes, "selection exp"); rc != 0) {
            return 30 + rc;
        }
    }
    const auto select_exps = download<float>(*dev_sel_exp_o.buffer, select_exp_count);
    const std::span<const float> chained_exps(select_exps.data(), kRouterRows);
    const std::span<const float> tie_exps(select_exps.data() + kRouterRows, kRouterRows);
    // Round-1 division queries: each exponential against
    // the tree sum, for the chained and tie logits.
    std::vector<float> divide1_numerators, divide1_denominators;
    {
        std::vector<float> pair_numerators, pair_denominators;
        router_select_normalize_arguments(chained_exps, kExperts, pair_numerators,
                                          pair_denominators);
        divide1_numerators = pair_numerators;
        divide1_denominators = pair_denominators;
        router_select_normalize_arguments(tie_exps, kExperts, pair_numerators, pair_denominators);
        divide1_numerators.insert(divide1_numerators.end(), pair_numerators.begin(),
                                  pair_numerators.end());
        divide1_denominators.insert(divide1_denominators.end(), pair_denominators.begin(),
                                    pair_denominators.end());
    }
    const std::size_t divide1_count = divide1_numerators.size();
    divide1_numerators.resize(((divide1_count + 255) / 256) * 256, 1.0f);
    divide1_denominators.resize(divide1_numerators.size(), 1.0f);
    auto dev_div1_n = create_shared_buffer(*device.device, divide1_numerators.size() * 4);
    auto dev_div1_d = create_shared_buffer(*device.device, divide1_numerators.size() * 4);
    auto dev_div1_q = create_shared_buffer(*device.device, divide1_numerators.size() * 4);
    if (!dev_div1_n || !dev_div1_d || !dev_div1_q) {
        return 6;
    }
    upload<float>(*dev_div1_n.buffer, divide1_numerators);
    upload<float>(*dev_div1_d.buffer, divide1_denominators);
    {
        const Pass passes[] = {
            {&adj_div,
             {&*dev_div1_n.buffer, &*dev_div1_d.buffer, &*dev_div1_q.buffer},
             {0, 0, 0},
             {.width = static_cast<std::uint32_t>(divide1_numerators.size()) / 256,
              .height = 1,
              .depth = 1},
             {.width = 256, .height = 1, .depth = 1}}};
        if (const int rc = run_batch(*queue.command_queue, passes, "normalize divide"); rc != 0) {
            return 30 + rc;
        }
    }
    const auto divide1_quotients = download<float>(*dev_div1_q.buffer, divide1_count);
    tally_divisions(std::span<const float>(divide1_numerators.data(), divide1_count),
                    std::span<const float>(divide1_denominators.data(), divide1_count),
                    divide1_quotients);
    const std::span<const float> chained_normalized(divide1_quotients.data(), kExperts);
    const std::span<const float> tie_normalized(divide1_quotients.data() + kExperts, kExperts);
    // Picks on the queried quotients, then round-2 division queries for the
    // renormalizations and shared sigmoids.
    std::vector<std::uint32_t> expected_ids[3];
    std::vector<float> expected_coefficients[3];
    std::vector<float> raw_coefficients[3];
    float expected_shared[3] = {0.0f, 0.0f, 0.0f};
    const std::span<const float> case_normalized[3] = {chained_normalized, tie_normalized,
                                                       chained_normalized};
    const std::span<const float> case_exps[3] = {chained_exps, tie_exps, chained_exps};
    std::vector<float> divide2_numerators, divide2_denominators;
    std::size_t divide2_offsets[3] = {0, 0, 0};
    for (std::size_t c = 0; c < 3; ++c) {
        expected_ids[c].resize(kActive);
        expected_coefficients[c].resize(kActive);
        raw_coefficients[c].resize(kActive);
        router_select_pick(kGeometry, top_values[c], case_normalized[c], expected_ids[c],
                           raw_coefficients[c]);
        std::vector<float> pair_numerators, pair_denominators;
        router_select_renormalize_arguments(raw_coefficients[c], top_values[c],
                                            case_exps[c][kExperts], pair_numerators,
                                            pair_denominators);
        divide2_offsets[c] = divide2_numerators.size();
        divide2_numerators.insert(divide2_numerators.end(), pair_numerators.begin(),
                                  pair_numerators.end());
        divide2_denominators.insert(divide2_denominators.end(), pair_denominators.begin(),
                                    pair_denominators.end());
    }
    const std::size_t divide2_count = divide2_numerators.size();
    divide2_numerators.resize(256, 1.0f);
    divide2_denominators.resize(256, 1.0f);
    auto dev_div2_n = create_shared_buffer(*device.device, divide2_numerators.size() * 4);
    auto dev_div2_d = create_shared_buffer(*device.device, divide2_numerators.size() * 4);
    auto dev_div2_q = create_shared_buffer(*device.device, divide2_numerators.size() * 4);
    if (!dev_div2_n || !dev_div2_d || !dev_div2_q) {
        return 6;
    }
    upload<float>(*dev_div2_n.buffer, divide2_numerators);
    upload<float>(*dev_div2_d.buffer, divide2_denominators);
    {
        const Pass passes[] = {{&adj_div,
                                {&*dev_div2_n.buffer, &*dev_div2_d.buffer, &*dev_div2_q.buffer},
                                {0, 0, 0},
                                {.width = 1, .height = 1, .depth = 1},
                                {.width = 256, .height = 1, .depth = 1}}};
        if (const int rc = run_batch(*queue.command_queue, passes, "renormalize divide"); rc != 0) {
            return 30 + rc;
        }
    }
    const auto divide2_quotients = download<float>(*dev_div2_q.buffer, divide2_count);
    tally_divisions(std::span<const float>(divide2_numerators.data(), divide2_count),
                    std::span<const float>(divide2_denominators.data(), divide2_count),
                    divide2_quotients);
    for (std::size_t c = 0; c < 3; ++c) {
        const std::span<const float> quotients(divide2_quotients.data() + divide2_offsets[c],
                                               std::size_t{top_values[c]} + 1);
        router_select_finalize(kGeometry, top_values[c], quotients, expected_ids[c],
                               expected_coefficients[c], expected_shared[c]);
    }
    {
        std::vector<Pass> passes;
        const MetalBuffer* logits_buffers[3] = {&*dev_logits.buffer, &*dev_tie_logits.buffer,
                                                &*dev_logits.buffer};
        for (std::size_t c = 0; c < 3; ++c) {
            passes.push_back(
                {&router_select,
                 {logits_buffers[c], &*select_cases[c].ids.buffer,
                  &*select_cases[c].coefficients.buffer,
                  &*select_cases[c].shared_coefficient.buffer, &*select_cases[c].top_n.buffer},
                 {0, 0, 0, 0, 0},
                 {.width = 1, .height = 1, .depth = 1},
                 {.width = kExperts, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "router_select"); rc != 0) {
            return 30 + rc;
        }
    }
    std::vector<std::uint32_t> case_ids[3];
    std::vector<float> case_coefficients[3];
    float case_shared[3] = {0.0f, 0.0f, 0.0f};
    for (std::size_t c = 0; c < 3; ++c) {
        case_ids[c] = download<std::uint32_t>(*select_cases[c].ids.buffer, kActive);
        case_coefficients[c] = download<float>(*select_cases[c].coefficients.buffer, kActive);
        case_shared[c] = download<float>(*select_cases[c].shared_coefficient.buffer, 1)[0];
        std::uint32_t id_bad = 0;
        for (std::uint32_t k = 0; k < kActive; ++k) {
            if (case_ids[c][k] != expected_ids[c][k]) {
                if (id_bad < 4) {
                    std::cout << "  " << select_cases[c].name << " id mismatch at " << k
                              << ": expected " << expected_ids[c][k] << " actual " << case_ids[c][k]
                              << '\n';
                }
                ++id_bad;
            }
        }
        const std::uint32_t coefficient_bad =
            mismatches_f32(expected_coefficients[c], case_coefficients[c], "coefficient");
        const bool shared_bad = std::bit_cast<std::uint32_t>(case_shared[c]) !=
                                std::bit_cast<std::uint32_t>(expected_shared[c]);
        std::cout << "router_select " << select_cases[c].name << ": id mismatches " << id_bad
                  << ", coefficient mismatches " << coefficient_bad
                  << (shared_bad ? ", shared MISMATCH" : ", shared exact") << '\n';
        if (id_bad != 0 || coefficient_bad != 0 || shared_bad) {
            return 39;
        }
    }
    // The tie case must have exercised the rule: both tied maxima selected.
    if (case_ids[1][0] != 200u || case_ids[1][1] != 3u) {
        std::cout << "tie case did not select the tied pair in tree order\n";
        return 38;
    }

    // Stage D: grouped_upgate. Same-source per-slot observations of the gate
    // and up dots at the selected experts, silu exponent queries, then the
    // full and sentinel dispatches.
    const std::uint64_t gate_word_stride = std::uint64_t{kExpertDim} * kQ4WordsHidden * 4;
    const std::uint64_t gate_group_stride = std::uint64_t{kExpertDim} * kQ4GroupsHidden * 2;
    {
        std::vector<Pass> passes;
        for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
            const bool shared = slot == kActive;
            const std::uint64_t expert = shared ? 0 : std::uint64_t{expected_ids[0][slot]};
            const MetalBuffer* gw = shared ? &*dev_sgw.buffer : &*dev_gw.buffer;
            const MetalBuffer* gs = shared ? &*dev_sgs.buffer : &*dev_gs.buffer;
            const MetalBuffer* gb = shared ? &*dev_sgb.buffer : &*dev_gb.buffer;
            const MetalBuffer* uw = shared ? &*dev_suw.buffer : &*dev_uw.buffer;
            const MetalBuffer* us = shared ? &*dev_sus.buffer : &*dev_us.buffer;
            const MetalBuffer* ub = shared ? &*dev_sub.buffer : &*dev_ub.buffer;
            const std::uint64_t word_offset = expert * gate_word_stride;
            const std::uint64_t group_offset = expert * gate_group_stride;
            const std::uint64_t out_offset = std::uint64_t{slot} * kExpertDim * 4;
            passes.push_back({&adj_q4p,
                              {&*dev_normed.buffer, gw, gs, gb, &*dev_gate_obs.buffer},
                              {0, word_offset, group_offset, group_offset, out_offset},
                              {.width = kExpertDim, .height = 1, .depth = 1},
                              {.width = 32, .height = 1, .depth = 1}});
            passes.push_back({&adj_q4p,
                              {&*dev_normed.buffer, uw, us, ub, &*dev_up_obs.buffer},
                              {0, word_offset, group_offset, group_offset, out_offset},
                              {.width = kExpertDim, .height = 1, .depth = 1},
                              {.width = 32, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "upgate observation"); rc != 0) {
            return 40 + rc;
        }
    }
    const auto gate_observed = download<float>(*dev_gate_obs.buffer, kUpgateRows);
    const auto up_observed = download<float>(*dev_up_obs.buffer, kUpgateRows);
    std::vector<float> upgate_exp_args(kUpgateRows);
    for (std::uint32_t sg = 0; sg < kUpgateRows; ++sg) {
        upgate_exp_args[sg] = -gate_observed[sg];
    }
    auto dev_up_exp_i = create_shared_buffer(*device.device, upgate_exp_args.size() * 4);
    auto dev_up_exp_o = create_shared_buffer(*device.device, upgate_exp_args.size() * 4);
    if (!dev_up_exp_i || !dev_up_exp_o) {
        return 6;
    }
    upload<float>(*dev_up_exp_i.buffer, upgate_exp_args);
    {
        const Pass passes[] = {{&adj_exp,
                                {&*dev_up_exp_i.buffer, &*dev_up_exp_o.buffer},
                                {0, 0},
                                {.width = kUpgateRows / 256, .height = 1, .depth = 1},
                                {.width = 256, .height = 1, .depth = 1}}};
        if (const int rc = run_batch(*queue.command_queue, passes, "upgate exp"); rc != 0) {
            return 40 + rc;
        }
    }
    const auto upgate_exps = download<float>(*dev_up_exp_o.buffer, kUpgateRows);
    // Silu quotient queries: gate over 1 + exp(-gate) for every slot row.
    std::vector<float> silu_numerators(kUpgateRows), silu_denominators(kUpgateRows);
    for (std::uint32_t sg = 0; sg < kUpgateRows; ++sg) {
        silu_numerators[sg] = gate_observed[sg];
        silu_denominators[sg] = 1.0f + upgate_exps[sg];
    }
    auto dev_silu_n = create_shared_buffer(*device.device, silu_numerators.size() * 4);
    auto dev_silu_d = create_shared_buffer(*device.device, silu_denominators.size() * 4);
    auto dev_silu_q = create_shared_buffer(*device.device, silu_numerators.size() * 4);
    if (!dev_silu_n || !dev_silu_d || !dev_silu_q) {
        return 6;
    }
    upload<float>(*dev_silu_n.buffer, silu_numerators);
    upload<float>(*dev_silu_d.buffer, silu_denominators);
    {
        const Pass passes[] = {{&adj_div,
                                {&*dev_silu_n.buffer, &*dev_silu_d.buffer, &*dev_silu_q.buffer},
                                {0, 0, 0},
                                {.width = kUpgateRows / 256, .height = 1, .depth = 1},
                                {.width = 256, .height = 1, .depth = 1}}};
        if (const int rc = run_batch(*queue.command_queue, passes, "silu divide"); rc != 0) {
            return 40 + rc;
        }
    }
    const auto silu_quotients = download<float>(*dev_silu_q.buffer, kUpgateRows);
    tally_divisions(silu_numerators, silu_denominators, silu_quotients);
    {
        std::vector<Pass> passes;
        const MetalBuffer* h_buffers[2] = {&*dev_h_full.buffer, &*dev_h_sentinel.buffer};
        const MetalBuffer* id_buffers[2] = {&*select_cases[0].ids.buffer,
                                            &*select_cases[2].ids.buffer};
        for (std::size_t c = 0; c < 2; ++c) {
            passes.push_back({&grouped_upgate,
                              {&*dev_normed.buffer, id_buffers[c], &*dev_gw.buffer, &*dev_gs.buffer,
                               &*dev_gb.buffer, &*dev_uw.buffer, &*dev_us.buffer, &*dev_ub.buffer,
                               &*dev_sgw.buffer, &*dev_sgs.buffer, &*dev_sgb.buffer,
                               &*dev_suw.buffer, &*dev_sus.buffer, &*dev_sub.buffer, h_buffers[c]},
                              {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                              {.width = kUpgateRows, .height = 1, .depth = 1},
                              {.width = 32, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "grouped_upgate"); rc != 0) {
            return 40 + rc;
        }
    }
    const auto device_h_full = download<std::uint16_t>(*dev_h_full.buffer, kUpgateRows);
    const auto device_h_sentinel = download<std::uint16_t>(*dev_h_sentinel.buffer, kUpgateRows);
    std::vector<std::uint16_t> expected_h_full(kUpgateRows), expected_h_sentinel(kUpgateRows);
    for (std::uint32_t sg = 0; sg < kUpgateRows; ++sg) {
        const std::uint32_t slot = sg / kExpertDim;
        const std::uint16_t value = grouped_upgate_epilogue(silu_quotients[sg], up_observed[sg]);
        expected_h_full[sg] = value;
        const bool live = slot >= kSentinelTop && slot < kActive ? false : true;
        expected_h_sentinel[sg] = live ? value : 0x0000;
    }
    const std::uint32_t h_full_bad = mismatches_u16(expected_h_full, device_h_full, "h-full");
    const std::uint32_t h_sentinel_bad =
        mismatches_u16(expected_h_sentinel, device_h_sentinel, "h-sentinel");
    std::cout << "grouped_upgate fixtures: full mismatches " << h_full_bad
              << ", sentinel mismatches " << h_sentinel_bad << '\n';
    if (h_full_bad != 0 || h_sentinel_bad != 0) {
        return 49;
    }

    // Stage E: grouped_down_res. Per-slot part observations on each case's
    // actual h buffer, the raw-total mirror for the chain axis, then the
    // fused kernels and their write-back contracts.
    const std::uint64_t down_word_stride = std::uint64_t{kHidden} * kQ4WordsExpert * 4;
    const std::uint64_t down_group_stride = std::uint64_t{kHidden} * kQ4GroupsExpert * 2;
    {
        std::vector<Pass> passes;
        const MetalBuffer* h_buffers[2] = {&*dev_h_full.buffer, &*dev_h_sentinel.buffer};
        const MetalBuffer* parts_buffers[2] = {&*dev_parts_full.buffer,
                                               &*dev_parts_sentinel.buffer};
        const MetalBuffer* total_buffers[2] = {&*dev_total_full.buffer,
                                               &*dev_total_sentinel.buffer};
        for (std::size_t c = 0; c < 2; ++c) {
            const auto& ids = c == 0 ? case_ids[0] : case_ids[2];
            for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
                const bool shared = slot == kActive;
                if (!shared && ids[slot] == kMoeSentinelId) {
                    continue;
                }
                const std::uint64_t expert = shared ? 0 : std::uint64_t{ids[slot]};
                const MetalBuffer* dw = shared ? &*dev_sdw.buffer : &*dev_dw.buffer;
                const MetalBuffer* ds = shared ? &*dev_sds.buffer : &*dev_ds.buffer;
                const MetalBuffer* db = shared ? &*dev_sdb.buffer : &*dev_db.buffer;
                passes.push_back({&adj_q4pe,
                                  {h_buffers[c], dw, ds, db, parts_buffers[c]},
                                  {std::uint64_t{slot} * kExpertDim * 2, expert * down_word_stride,
                                   expert * down_group_stride, expert * down_group_stride,
                                   std::uint64_t{slot} * kHidden * 4},
                                  {.width = kHidden, .height = 1, .depth = 1},
                                  {.width = 32, .height = 1, .depth = 1}});
            }
            passes.push_back({&adj_down_total,
                              {h_buffers[c],
                               c == 0 ? &*select_cases[0].ids.buffer : &*select_cases[2].ids.buffer,
                               c == 0 ? &*select_cases[0].coefficients.buffer
                                      : &*select_cases[2].coefficients.buffer,
                               c == 0 ? &*select_cases[0].shared_coefficient.buffer
                                      : &*select_cases[2].shared_coefficient.buffer,
                               &*dev_dw.buffer, &*dev_ds.buffer, &*dev_db.buffer, &*dev_sdw.buffer,
                               &*dev_sds.buffer, &*dev_sdb.buffer, total_buffers[c]},
                              {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                              {.width = kHidden, .height = 1, .depth = 1},
                              {.width = 32, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "down observation"); rc != 0) {
            return 50 + rc;
        }
    }
    const auto parts_full = download<float>(*dev_parts_full.buffer, std::size_t{kSlots} * kHidden);
    const auto parts_sentinel =
        download<float>(*dev_parts_sentinel.buffer, std::size_t{kSlots} * kHidden);
    const auto total_full = download<float>(*dev_total_full.buffer, kHidden);
    const auto total_sentinel = download<float>(*dev_total_sentinel.buffer, kHidden);
    // Chain adjudication at float resolution across both cases.
    std::uint32_t fused_matches = 0, separate_matches = 0;
    const std::uint32_t chain_rows = kHidden * 2;
    for (std::uint32_t c = 0; c < 2; ++c) {
        const auto& ids = c == 0 ? case_ids[0] : case_ids[2];
        const auto& coefficients = c == 0 ? case_coefficients[0] : case_coefficients[2];
        const float shared_coefficient = c == 0 ? case_shared[0] : case_shared[2];
        const auto& parts = c == 0 ? parts_full : parts_sentinel;
        const auto& totals = c == 0 ? total_full : total_sentinel;
        for (std::uint32_t row = 0; row < kHidden; ++row) {
            float slot_parts[16];
            for (std::uint32_t slot = 0; slot < kSlots; ++slot) {
                slot_parts[slot] = parts[std::size_t{slot} * kHidden + row];
            }
            const std::span<const float> row_parts(slot_parts, kSlots);
            const float fused = grouped_down_total(ids, coefficients, shared_coefficient, row_parts,
                                                   ChainOrder::Fused);
            const float separate = grouped_down_total(ids, coefficients, shared_coefficient,
                                                      row_parts, ChainOrder::Separate);
            const std::uint32_t observed = std::bit_cast<std::uint32_t>(totals[row]);
            fused_matches += std::bit_cast<std::uint32_t>(fused) == observed ? 1 : 0;
            separate_matches += std::bit_cast<std::uint32_t>(separate) == observed ? 1 : 0;
        }
    }
    const bool fused_all = fused_matches == chain_rows;
    const bool separate_all = separate_matches == chain_rows;
    std::cout << "down chain: fused " << fused_matches << "/" << chain_rows << ", separate "
              << separate_matches << "/" << chain_rows << '\n';
    if (fused_all == separate_all) {
        std::cout << "down chain: "
                  << (fused_all ? "degenerate fixtures, both candidates match"
                                : "NEITHER candidate matches; device-defined totals stand")
                  << " — expected totals pin to the observed mirror\n";
    }
    {
        std::vector<Pass> passes;
        const MetalBuffer* h_buffers[2] = {&*dev_h_full.buffer, &*dev_h_sentinel.buffer};
        const MetalBuffer* moe_buffers[2] = {&*dev_moe_full.buffer, &*dev_moe_sentinel.buffer};
        const MetalBuffer* layer_buffers[2] = {&*dev_layer_full.buffer,
                                               &*dev_layer_sentinel.buffer};
        for (std::size_t c = 0; c < 2; ++c) {
            const auto& select_case = c == 0 ? select_cases[0] : select_cases[2];
            passes.push_back(
                {&grouped_down_res,
                 {h_buffers[c], &*select_case.ids.buffer, &*select_case.coefficients.buffer,
                  &*select_case.shared_coefficient.buffer, &*dev_dw.buffer, &*dev_ds.buffer,
                  &*dev_db.buffer, &*dev_sdw.buffer, &*dev_sds.buffer, &*dev_sdb.buffer,
                  moe_buffers[c], &*dev_res.buffer, layer_buffers[c]},
                 {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                 {.width = kHidden, .height = 1, .depth = 1},
                 {.width = 32, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "grouped_down_res"); rc != 0) {
            return 50 + rc;
        }
    }
    std::uint32_t down_bad = 0;
    for (std::uint32_t c = 0; c < 2; ++c) {
        const auto& totals = c == 0 ? total_full : total_sentinel;
        const auto device_moe = download<std::uint16_t>(
            c == 0 ? *dev_moe_full.buffer : *dev_moe_sentinel.buffer, kHidden);
        const auto device_layer = download<std::uint16_t>(
            c == 0 ? *dev_layer_full.buffer : *dev_layer_sentinel.buffer, kHidden);
        std::vector<std::uint16_t> expected_moe(kHidden), expected_layer(kHidden);
        for (std::uint32_t row = 0; row < kHidden; ++row) {
            grouped_down_res_epilogue(totals[row], device_res[row], expected_moe[row],
                                      expected_layer[row]);
        }
        const char* label = c == 0 ? "down-full" : "down-sentinel";
        down_bad += mismatches_u16(expected_moe, device_moe, label);
        down_bad += mismatches_u16(expected_layer, device_layer, label);
    }
    std::cout << "grouped_down_res fixtures: mismatches " << down_bad << '\n';
    if (down_bad != 0) {
        return 59;
    }

    std::cout << "moe fixtures: PASS\n"
              << "  device: " << device.device->name() << '\n'
              << "  division evidence: ieee " << divide_ieee << "/" << divide_pairs
              << ", reciprocal-multiply " << divide_reciprocal << "/" << divide_pairs << '\n'
              << "  down chain: "
              << (fused_all && !separate_all
                      ? "fused"
                      : (separate_all && !fused_all ? "separate"
                                                    : (fused_all ? "both" : "device-defined")))
              << '\n'
              << "  command buffers submitted: " << submissions << '\n';
    return 0;
}
