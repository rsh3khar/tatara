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
constexpr std::uint32_t kVocabulary = generated::kKernelLibraryVocabulary;
constexpr std::uint32_t kArgmaxGroups = generated::kKernelLibraryArgmaxGroups;
constexpr std::uint32_t kQ4WordsHidden = kHidden / 8;
constexpr std::uint32_t kQ4GroupsHidden = kHidden / generated::kKernelLibraryGroupSize;
constexpr std::uint32_t kTailCount = 70000;

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

struct ArgmaxCase {
    const char* name;
    const MetalBuffer* logits;
    std::span<const std::uint16_t> host_logits;
    std::uint32_t count;
    const MetalBuffer* count_buffer;
    const MetalBuffer* values;
    const MetalBuffer* indices;
    const MetalBuffer* token;
};

} // namespace

int run_head_battery() {
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
        {"lmhead_q4", {}},
        {"logits_argmax_stage1", {}},
        {"logits_argmax_stage2", {}},
        {"adjudicate_q4_row", {}},
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
    const auto& lmhead = kernels[0].pipeline;
    const auto& stage1 = kernels[1].pipeline;
    const auto& stage2 = kernels[2].pipeline;
    const auto& adj_q4 = kernels[3].pipeline;

    // Fixture inputs: a banded hidden vector with signed-zero and subnormal
    // edges, and a deterministic synthetic vocabulary table with nibble-word
    // extremes on early rows.
    Xorshift64Star generator(0x4EAD5EEDull);
    std::vector<std::uint16_t> x(kHidden);
    for (auto& value : x) {
        value = generator.next_banded_bf16();
    }
    x[0] = 0x0000;
    x[1] = 0x8000;
    x[2] = 0x0001;
    x[3] = 0x0080;
    std::vector<std::uint32_t> words(std::size_t{kVocabulary} * kQ4WordsHidden);
    std::vector<std::uint16_t> scales(std::size_t{kVocabulary} * kQ4GroupsHidden);
    std::vector<std::uint16_t> biases(scales.size());
    for (auto& word : words) {
        word = static_cast<std::uint32_t>(generator.next());
    }
    for (std::size_t i = 0; i < scales.size(); ++i) {
        scales[i] = generator.next_banded_bf16();
        biases[i] = generator.next_banded_bf16();
    }
    words[0] = 0x00000000u;
    words[1] = 0xFFFFFFFFu;
    scales[0] = 0x0000;
    biases[0] = 0x8000;

    // The planted-tie logits: banded values with 0x7F00 pairs above every
    // draw, covering the five merge levels the oracle test locks.
    std::vector<std::uint16_t> tie_logits(kVocabulary);
    for (auto& value : tie_logits) {
        value = generator.next_banded_bf16();
    }
    tie_logits[123] = 0x7F00;
    tie_logits[60123] = 0x7F00;
    tie_logits[4 * 256 + 5] = 0x7F00;
    tie_logits[4 * 256 + 17] = 0x7F00;
    tie_logits[9 * 256 + 10] = 0x7F00;
    tie_logits[9 * 256 + 200] = 0x7F00;
    tie_logits[300] = 0x7F00;
    tie_logits[40000] = 0x7F00;
    tie_logits[1234] = 0x7F00;
    tie_logits[1234 + 65536] = 0x7F00;

    auto dev_x = create_shared_buffer(*device.device, kHidden * 2);
    auto dev_words = create_shared_buffer(*device.device, words.size() * 4);
    auto dev_scales = create_shared_buffer(*device.device, scales.size() * 2);
    auto dev_biases = create_shared_buffer(*device.device, biases.size() * 2);
    auto dev_obs = create_shared_buffer(*device.device, std::size_t{kVocabulary} * 4);
    auto dev_out = create_shared_buffer(*device.device, std::size_t{kVocabulary} * 2);
    auto dev_tie = create_shared_buffer(*device.device, std::size_t{kVocabulary} * 2);
    auto dev_count_full = create_shared_buffer(*device.device, 4);
    auto dev_count_tail = create_shared_buffer(*device.device, 4);
    for (auto* result : {&dev_x, &dev_words, &dev_scales, &dev_biases, &dev_obs, &dev_out, &dev_tie,
                         &dev_count_full, &dev_count_tail}) {
        if (!*result) {
            return 6;
        }
    }
    struct CaseBuffers {
        MetalBufferResult values;
        MetalBufferResult indices;
        MetalBufferResult token;
    };
    CaseBuffers case_buffers[3];
    for (auto& buffers : case_buffers) {
        buffers.values = create_shared_buffer(*device.device, kArgmaxGroups * 4);
        buffers.indices = create_shared_buffer(*device.device, kArgmaxGroups * 4);
        buffers.token = create_shared_buffer(*device.device, 4);
        if (!buffers.values || !buffers.indices || !buffers.token) {
            return 6;
        }
    }
    upload<std::uint16_t>(*dev_x.buffer, x);
    upload<std::uint32_t>(*dev_words.buffer, words);
    upload<std::uint16_t>(*dev_scales.buffer, scales);
    upload<std::uint16_t>(*dev_biases.buffer, biases);
    upload<std::uint16_t>(*dev_tie.buffer, tie_logits);
    std::memcpy(dev_count_full.buffer->contents(), &kVocabulary, 4);
    std::memcpy(dev_count_tail.buffer->contents(), &kTailCount, 4);

    // Stage 1: the head projection against its same-source observation.
    {
        const Pass passes[] = {{&adj_q4,
                                {&*dev_x.buffer, &*dev_words.buffer, &*dev_scales.buffer,
                                 &*dev_biases.buffer, &*dev_obs.buffer},
                                {0, 0, 0, 0, 0},
                                {.width = kVocabulary, .height = 1, .depth = 1},
                                {.width = 32, .height = 1, .depth = 1}}};
        if (const int rc = run_batch(*queue.command_queue, passes, "q4 observation"); rc != 0) {
            return 10 + rc;
        }
    }
    {
        const Pass passes[] = {{&lmhead,
                                {&*dev_x.buffer, &*dev_words.buffer, &*dev_scales.buffer,
                                 &*dev_biases.buffer, &*dev_out.buffer},
                                {0, 0, 0, 0, 0},
                                {.width = kVocabulary, .height = 1, .depth = 1},
                                {.width = 32, .height = 1, .depth = 1}}};
        if (const int rc = run_batch(*queue.command_queue, passes, "lmhead_q4"); rc != 0) {
            return 10 + rc;
        }
    }
    const auto observed = download<float>(*dev_obs.buffer, kVocabulary);
    const auto device_out = download<std::uint16_t>(*dev_out.buffer, kVocabulary);
    std::uint32_t head_bad = 0;
    for (std::uint32_t row = 0; row < kVocabulary; ++row) {
        const std::uint16_t expected = flush_subnormal_bf16(bf16_from_f32(observed[row]));
        if (expected != device_out[row]) {
            if (head_bad < 4) {
                std::cout << "  lmhead mismatch at " << row << ": expected 0x" << std::hex
                          << expected << " actual 0x" << device_out[row] << std::dec << '\n';
            }
            ++head_bad;
        }
    }
    std::cout << "lmhead_q4 fixtures: mismatches " << head_bad << '\n';
    if (head_bad != 0) {
        return 19;
    }

    // Stage 2: the two-stage argmax — chained on the head output, planted
    // ties, and the grid-stride tail count.
    const ArgmaxCase cases[3] = {
        {"chained", &*dev_out.buffer, device_out, kVocabulary, &*dev_count_full.buffer,
         &*case_buffers[0].values.buffer, &*case_buffers[0].indices.buffer,
         &*case_buffers[0].token.buffer},
        {"ties", &*dev_tie.buffer, tie_logits, kVocabulary, &*dev_count_full.buffer,
         &*case_buffers[1].values.buffer, &*case_buffers[1].indices.buffer,
         &*case_buffers[1].token.buffer},
        {"tail", &*dev_tie.buffer, tie_logits, kTailCount, &*dev_count_tail.buffer,
         &*case_buffers[2].values.buffer, &*case_buffers[2].indices.buffer,
         &*case_buffers[2].token.buffer},
    };
    {
        std::vector<Pass> passes;
        for (const ArgmaxCase& current : cases) {
            passes.push_back(
                {&stage1,
                 {current.logits, current.count_buffer, current.values, current.indices},
                 {0, 0, 0, 0},
                 {.width = kArgmaxGroups, .height = 1, .depth = 1},
                 {.width = 256, .height = 1, .depth = 1}});
            passes.push_back({&stage2,
                              {current.values, current.indices, current.token},
                              {0, 0, 0},
                              {.width = 1, .height = 1, .depth = 1},
                              {.width = 1, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "argmax"); rc != 0) {
            return 20 + rc;
        }
    }
    for (const ArgmaxCase& current : cases) {
        std::vector<float> expected_values(kArgmaxGroups, 0.0f);
        std::vector<std::uint32_t> expected_indices(kArgmaxGroups, 0u);
        for (std::uint32_t group = 0; group < kArgmaxGroups; ++group) {
            logits_argmax_stage1_reference(current.host_logits, current.count, group,
                                           expected_values[group], expected_indices[group]);
        }
        const std::uint32_t expected_token =
            logits_argmax_stage2_reference(expected_values, expected_indices);
        const auto device_values = download<float>(*current.values, kArgmaxGroups);
        const auto device_indices = download<std::uint32_t>(*current.indices, kArgmaxGroups);
        const std::uint32_t device_token = download<std::uint32_t>(*current.token, 1)[0];
        std::uint32_t record_bad = 0;
        for (std::uint32_t group = 0; group < kArgmaxGroups; ++group) {
            if (std::bit_cast<std::uint32_t>(expected_values[group]) !=
                    std::bit_cast<std::uint32_t>(device_values[group]) ||
                expected_indices[group] != device_indices[group]) {
                if (record_bad < 4) {
                    std::cout << "  " << current.name << " record mismatch at group " << group
                              << ": expected (" << expected_values[group] << ", "
                              << expected_indices[group] << ") actual (" << device_values[group]
                              << ", " << device_indices[group] << ")\n";
                }
                ++record_bad;
            }
        }
        const bool token_bad = device_token != expected_token;
        std::cout << "argmax " << current.name << ": record mismatches " << record_bad
                  << (token_bad ? ", token MISMATCH" : ", token exact") << '\n';
        if (record_bad != 0 || token_bad) {
            return 29;
        }
    }

    std::cout << "head fixtures: PASS\n"
              << "  device: " << device.device->name() << '\n'
              << "  command buffers submitted: " << submissions << '\n';
    return 0;
}
