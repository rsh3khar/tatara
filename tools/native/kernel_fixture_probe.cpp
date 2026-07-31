#include "fixture_batteries.h"
#include "kernel_reference.h"
#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/generated/kernel_library.h"
#include "tatara/generated/model_plan.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace tatara::backend::metal;
using namespace tatara::testing;

constexpr std::uint32_t kSweepCount = 4096;
constexpr std::uint32_t kEmbedRows = 64;
constexpr std::uint32_t kRmsVectors = 32;
constexpr std::uint32_t kThreadsPerGroup = 256;
constexpr float kRmsEpsilon = 1e-6f;
constexpr std::uint32_t kMismatchDumpLimit = 8;

constexpr std::uint32_t kHidden = generated::kKernelLibraryHidden;
constexpr std::uint32_t kGroupSize = generated::kKernelLibraryGroupSize;
constexpr std::uint32_t kWordsPerRow = kHidden / 8;
constexpr std::uint32_t kGroupsPerRow = kHidden / kGroupSize;

constexpr std::uint32_t kTreeShapeCount = 6;
constexpr std::uint32_t kSumObservationVectors = 64;
constexpr std::uint32_t kSimdLaneCount = 32;

// The rsqrt dispatch carries the evidence sweep plus one device query per RMS
// vector per candidate tree shape, padded up to whole threadgroups.
constexpr std::uint32_t kRsqrtQueryCount = kRmsVectors * kTreeShapeCount;
constexpr std::uint32_t kRsqrtTotal =
    ((kSweepCount + kRsqrtQueryCount + kThreadsPerGroup - 1) / kThreadsPerGroup) * kThreadsPerGroup;

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

struct PipelineSet {
    MetalComputePipeline embed;
    MetalComputePipeline rms;
    MetalComputePipeline rms_sum;
    MetalComputePipeline simd_sum;
    MetalComputePipeline rsqrt;
    MetalComputePipeline bfloat_multiply;
};

int run_single_pass(const MetalCommandQueue& queue, const MetalComputePipeline& pipeline,
                    std::span<const MetalBuffer*> buffers, MetalSize threadgroups,
                    MetalSize threads_per_group) {
    auto command_buffer = create_command_buffer(queue);
    if (!command_buffer) {
        return 1;
    }
    auto pass = begin_compute_pass(std::move(*command_buffer.command_buffer));
    if (!pass) {
        return 2;
    }
    if (set_compute_pipeline(*pass.compute_pass, pipeline) != MetalCommandError::None) {
        return 3;
    }
    for (std::uint32_t index = 0; index < buffers.size(); ++index) {
        if (set_buffer(*pass.compute_pass, *buffers[index], 0, index) != MetalCommandError::None) {
            return 4;
        }
    }
    if (dispatch_threadgroups(*pass.compute_pass, threadgroups, threads_per_group) !=
        MetalCommandError::None) {
        return 5;
    }
    auto ended = end_compute_pass(std::move(*pass.compute_pass));
    if (!ended) {
        return 6;
    }
    auto pending = commit(std::move(*ended.command_buffer));
    if (!pending) {
        return 7;
    }
    ++submissions;
    auto execution = wait_until_completed(std::move(*pending.pending_execution));
    if (!execution) {
        std::cerr << "execution failed: " << execution.failure_description.view() << '\n';
        return 8;
    }
    return 0;
}

void dump_mismatches(std::string_view label, std::span<const std::uint16_t> expected,
                     std::span<const std::uint16_t> actual) {
    std::uint32_t dumped = 0;
    for (std::size_t index = 0; index < expected.size() && dumped < kMismatchDumpLimit; ++index) {
        if (expected[index] != actual[index]) {
            std::cout << "  " << label << " mismatch at " << index << ": expected 0x" << std::hex
                      << expected[index] << " actual 0x" << actual[index] << std::dec << '\n';
            ++dumped;
        }
    }
}

} // namespace

int run_kernel_battery() {
    static_assert(kHidden == tatara::model::qwen36::generated::kModelPlan.dimensions.hidden);
    static_assert(kGroupSize == tatara::model::qwen36::generated::kModelPlan.weights.group_size);

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
    PipelineSet pipelines;
    const struct {
        const char* name;
        MetalComputePipeline* slot;
    } wanted[] = {
        {"embed_row_q4", &pipelines.embed},
        {"rms_only", &pipelines.rms},
        {"adjudicate_rms_sum", &pipelines.rms_sum},
        {"adjudicate_simd_sum", &pipelines.simd_sum},
        {"adjudicate_rsqrt", &pipelines.rsqrt},
        {"adjudicate_bfloat_multiply", &pipelines.bfloat_multiply},
    };
    for (const auto& entry : wanted) {
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
        *entry.slot = std::move(*pipeline.pipeline);
    }

    Xorshift64Star generator(0x7A7A5EEDull);

    // RMS fixture inputs come first: their candidate means feed the rsqrt
    // device queries appended to the evidence sweep.
    std::vector<std::uint16_t> rms_inputs(std::size_t{kRmsVectors} * kHidden);
    std::vector<std::uint16_t> rms_weights(std::size_t{kRmsVectors} * kHidden);
    const std::span<std::uint16_t> first_input(rms_inputs.data(), kHidden);
    const std::span<std::uint16_t> first_weight(rms_weights.data(), kHidden);
    if (!find_rms_sum_discriminator(0x50D4ull, 20000, kHidden, kRmsEpsilon, first_input,
                                    first_weight)) {
        std::cerr << "sum-order discriminator search failed\n";
        return 52;
    }
    for (std::size_t index = std::size_t{1} * kHidden; index < rms_inputs.size(); ++index) {
        rms_inputs[index] = generator.next_banded_bf16();
        rms_weights[index] = generator.next_banded_bf16();
    }

    std::vector<float> rsqrt_inputs;
    rsqrt_inputs.reserve(kRsqrtTotal);
    while (rsqrt_inputs.size() < kSweepCount) {
        const float value = f32_from_bf16(generator.next_finite_bf16());
        const float magnitude = value < 0.0f ? -value : value;
        if (magnitude > 0.0f) {
            rsqrt_inputs.push_back(magnitude);
        }
    }
    for (std::uint32_t vector = 0; vector < kRmsVectors; ++vector) {
        const std::span<const std::uint16_t> input(
            rms_inputs.data() + std::size_t{vector} * kHidden, kHidden);
        for (const SimdTreeShape shape : kAllTreeShapes) {
            rsqrt_inputs.push_back(rms_mean_of_squares(input, kHidden, shape) + kRmsEpsilon);
        }
    }
    rsqrt_inputs.resize(kRsqrtTotal, 1.0f);

    auto rsqrt_in = create_shared_buffer(*device.device, kRsqrtTotal * sizeof(float));
    auto rsqrt_out = create_shared_buffer(*device.device, kRsqrtTotal * sizeof(float));
    if (!rsqrt_in || !rsqrt_out) {
        return 6;
    }
    upload<float>(*rsqrt_in.buffer, rsqrt_inputs);
    const MetalBuffer* rsqrt_buffers[] = {&*rsqrt_in.buffer, &*rsqrt_out.buffer};
    if (const int result =
            run_single_pass(*queue.command_queue, pipelines.rsqrt, rsqrt_buffers,
                            {.width = kRsqrtTotal / kThreadsPerGroup, .height = 1, .depth = 1},
                            {.width = kThreadsPerGroup, .height = 1, .depth = 1});
        result != 0) {
        return 10 + result;
    }
    const auto rsqrt_device = download<float>(*rsqrt_out.buffer, kRsqrtTotal);
    std::uint32_t rsqrt_matches[2] = {0, 0};
    for (std::uint32_t i = 0; i < kSweepCount; ++i) {
        if (rsqrt_device[i] == rsqrt_candidate(rsqrt_inputs[i], RsqrtOrder::ReciprocalOfSqrt)) {
            ++rsqrt_matches[0];
        }
        if (rsqrt_device[i] == rsqrt_candidate(rsqrt_inputs[i], RsqrtOrder::CorrectlyRounded)) {
            ++rsqrt_matches[1];
        }
    }
    std::cout << "rsqrt evidence: reciprocal-of-sqrt " << rsqrt_matches[0] << "/" << kSweepCount
              << ", correctly-rounded " << rsqrt_matches[1] << "/" << kSweepCount
              << "; pinned device-defined\n";
    std::vector<float> device_inverse(kRsqrtQueryCount);
    for (std::uint32_t i = 0; i < kRsqrtQueryCount; ++i) {
        device_inverse[i] = rsqrt_device[kSweepCount + i];
    }

    // Adjudication sweep: native bfloat multiply.
    std::vector<std::uint16_t> multiply_left(kSweepCount);
    std::vector<std::uint16_t> multiply_right(kSweepCount);
    for (std::uint32_t i = 0; i < kSweepCount; ++i) {
        multiply_left[i] = generator.next_finite_bf16();
        multiply_right[i] = generator.next_finite_bf16();
    }
    auto multiply_a = create_shared_buffer(*device.device, kSweepCount * sizeof(std::uint16_t));
    auto multiply_b = create_shared_buffer(*device.device, kSweepCount * sizeof(std::uint16_t));
    auto multiply_out = create_shared_buffer(*device.device, kSweepCount * sizeof(std::uint16_t));
    if (!multiply_a || !multiply_b || !multiply_out) {
        return 7;
    }
    upload<std::uint16_t>(*multiply_a.buffer, multiply_left);
    upload<std::uint16_t>(*multiply_b.buffer, multiply_right);
    const MetalBuffer* multiply_buffers[] = {&*multiply_a.buffer, &*multiply_b.buffer,
                                             &*multiply_out.buffer};
    if (const int result =
            run_single_pass(*queue.command_queue, pipelines.bfloat_multiply, multiply_buffers,
                            {.width = kSweepCount / kThreadsPerGroup, .height = 1, .depth = 1},
                            {.width = kThreadsPerGroup, .height = 1, .depth = 1});
        result != 0) {
        return 30 + result;
    }
    const auto multiply_device = download<std::uint16_t>(*multiply_out.buffer, kSweepCount);
    std::uint32_t multiply_matches = 0;
    std::vector<std::uint16_t> multiply_expected(kSweepCount);
    for (std::uint32_t i = 0; i < kSweepCount; ++i) {
        multiply_expected[i] = bfloat_multiply_candidate(multiply_left[i], multiply_right[i]);
        if (multiply_device[i] == multiply_expected[i]) {
            ++multiply_matches;
        }
    }
    std::cout << "bfloat multiply adjudication: single-rounding " << multiply_matches << "/"
              << kSweepCount << '\n';
    if (multiply_matches != kSweepCount) {
        dump_mismatches("bfloat-multiply", multiply_expected, multiply_device);
        std::cerr << "bfloat multiply candidate rejected; pin policy required\n";
        return 40;
    }

    // Kernel fixtures: embed_row_q4 across both multiply-add candidates.
    const auto discriminator = find_multiply_add_discriminator(0xD15Cull, 200000);
    if (!discriminator.found) {
        std::cerr << "multiply-add discriminator search failed\n";
        return 50;
    }
    std::vector<std::uint32_t> words(kEmbedRows * kWordsPerRow);
    std::vector<std::uint16_t> scales(kEmbedRows * kGroupsPerRow);
    std::vector<std::uint16_t> biases(kEmbedRows * kGroupsPerRow);
    for (auto& word : words) {
        word = static_cast<std::uint32_t>(generator.next());
    }
    for (std::uint32_t i = 0; i < scales.size(); ++i) {
        scales[i] = generator.next_finite_bf16();
        biases[i] = generator.next_finite_bf16();
    }
    scales[0] = discriminator.scale;
    biases[0] = discriminator.bias;
    words[0] = (words[0] & ~0xFu) | static_cast<std::uint32_t>(discriminator.quant);

    auto words_buffer = create_shared_buffer(*device.device, words.size() * sizeof(std::uint32_t));
    auto scales_buffer =
        create_shared_buffer(*device.device, scales.size() * sizeof(std::uint16_t));
    auto biases_buffer =
        create_shared_buffer(*device.device, biases.size() * sizeof(std::uint16_t));
    auto row_buffer = create_shared_buffer(*device.device, std::uint64_t{kEmbedRows} * kHidden *
                                                               sizeof(std::uint16_t));
    if (!words_buffer || !scales_buffer || !biases_buffer || !row_buffer) {
        return 8;
    }
    upload<std::uint32_t>(*words_buffer.buffer, words);
    upload<std::uint16_t>(*scales_buffer.buffer, scales);
    upload<std::uint16_t>(*biases_buffer.buffer, biases);

    std::vector<MetalBufferResult> token_buffers;
    token_buffers.reserve(kEmbedRows);
    for (std::uint32_t token = 0; token < kEmbedRows; ++token) {
        auto buffer = create_shared_buffer(*device.device, sizeof(std::uint32_t));
        if (!buffer) {
            return 9;
        }
        std::memcpy(buffer.buffer->contents(), &token, sizeof(std::uint32_t));
        token_buffers.push_back(std::move(buffer));
    }

    {
        auto command_buffer = create_command_buffer(*queue.command_queue);
        if (!command_buffer) {
            return 60;
        }
        MetalCommandBuffer chain = std::move(*command_buffer.command_buffer);
        for (std::uint32_t token = 0; token < kEmbedRows; ++token) {
            auto pass = begin_compute_pass(std::move(chain));
            if (!pass) {
                return 61;
            }
            if (set_compute_pipeline(*pass.compute_pass, pipelines.embed) !=
                    MetalCommandError::None ||
                set_buffer(*pass.compute_pass, *words_buffer.buffer, 0, 0) !=
                    MetalCommandError::None ||
                set_buffer(*pass.compute_pass, *scales_buffer.buffer, 0, 1) !=
                    MetalCommandError::None ||
                set_buffer(*pass.compute_pass, *biases_buffer.buffer, 0, 2) !=
                    MetalCommandError::None ||
                set_buffer(*pass.compute_pass, *token_buffers[token].buffer, 0, 3) !=
                    MetalCommandError::None ||
                set_buffer(*pass.compute_pass, *row_buffer.buffer,
                           std::uint64_t{token} * kHidden * sizeof(std::uint16_t),
                           4) != MetalCommandError::None) {
                return 62;
            }
            if (dispatch_threadgroups(
                    *pass.compute_pass,
                    {.width = kHidden / kThreadsPerGroup, .height = 1, .depth = 1},
                    {.width = kThreadsPerGroup, .height = 1, .depth = 1}) !=
                MetalCommandError::None) {
                return 63;
            }
            auto ended = end_compute_pass(std::move(*pass.compute_pass));
            if (!ended) {
                return 64;
            }
            chain = std::move(*ended.command_buffer);
        }
        auto pending = commit(std::move(chain));
        if (!pending) {
            return 65;
        }
        ++submissions;
        auto execution = wait_until_completed(std::move(*pending.pending_execution));
        if (!execution) {
            std::cerr << "embed execution failed: " << execution.failure_description.view() << '\n';
            return 66;
        }
    }

    std::uint32_t embed_matches[2] = {0, 0};
    std::vector<std::uint16_t> expected_row(kHidden);
    std::vector<std::uint16_t> first_mismatch_expected;
    std::vector<std::uint16_t> first_mismatch_actual;
    for (std::uint32_t token = 0; token < kEmbedRows; ++token) {
        const auto actual =
            download<std::uint16_t>(*row_buffer.buffer, kHidden, std::size_t{token} * kHidden);
        for (int order = 0; order < 2; ++order) {
            embed_row_q4_reference(
                words, scales, biases, token, kHidden, kGroupSize,
                order == 0 ? MultiplyAddOrder::Fused : MultiplyAddOrder::Separate, expected_row);
            if (std::equal(expected_row.begin(), expected_row.end(), actual.begin())) {
                ++embed_matches[order];
            } else if (first_mismatch_expected.empty()) {
                first_mismatch_expected = expected_row;
                first_mismatch_actual = actual;
            }
        }
    }
    std::cout << "embed_row_q4 fixtures: fused " << embed_matches[0] << "/" << kEmbedRows
              << ", separate " << embed_matches[1] << "/" << kEmbedRows << '\n';
    const bool embed_fused = embed_matches[0] == kEmbedRows;
    const bool embed_separate = embed_matches[1] == kEmbedRows;
    if (embed_fused == embed_separate) {
        if (!first_mismatch_expected.empty()) {
            dump_mismatches("embed", first_mismatch_expected, first_mismatch_actual);
        }
        std::cerr << "embed adjudication is not exactly one candidate; pin policy required\n";
        return 70;
    }

    // Adjudication: the bare 32-lane simd_sum intrinsic, observed directly.
    // The first vectors are searched discriminators, one per candidate pair,
    // so any device-matching shape is isolated structurally.
    std::vector<float> observation_lanes(std::size_t{kSumObservationVectors} * kSimdLaneCount);
    std::size_t observation_cursor = 0;
    for (std::uint32_t a = 0; a < kTreeShapeCount; ++a) {
        for (std::uint32_t b = a + 1; b < kTreeShapeCount; ++b) {
            const std::span<float> lanes(
                observation_lanes.data() + observation_cursor * kSimdLaneCount, kSimdLaneCount);
            if (!find_tree_shape_discriminator(0xD15C0 + a * 8 + b, 200000, kAllTreeShapes[a],
                                               kAllTreeShapes[b], lanes)) {
                std::cerr << "tree-shape discriminator search failed for pair " << a << "," << b
                          << '\n';
                return 118;
            }
            ++observation_cursor;
        }
    }
    for (std::size_t index = observation_cursor * kSimdLaneCount; index < observation_lanes.size();
         ++index) {
        float value = 0.0f;
        while (value <= 0.0f) {
            value = f32_from_bf16(generator.next_banded_bf16());
            value = value < 0.0f ? -value : value;
        }
        observation_lanes[index] = value;
    }
    auto observation_in =
        create_shared_buffer(*device.device, observation_lanes.size() * sizeof(float));
    auto observation_out =
        create_shared_buffer(*device.device, kSumObservationVectors * sizeof(float));
    if (!observation_in || !observation_out) {
        return 110;
    }
    upload<float>(*observation_in.buffer, observation_lanes);
    {
        auto command_buffer = create_command_buffer(*queue.command_queue);
        if (!command_buffer) {
            return 111;
        }
        MetalCommandBuffer chain = std::move(*command_buffer.command_buffer);
        for (std::uint32_t vector = 0; vector < kSumObservationVectors; ++vector) {
            auto pass = begin_compute_pass(std::move(chain));
            if (!pass) {
                return 112;
            }
            if (set_compute_pipeline(*pass.compute_pass, pipelines.simd_sum) !=
                    MetalCommandError::None ||
                set_buffer(*pass.compute_pass, *observation_in.buffer,
                           std::uint64_t{vector} * kSimdLaneCount * sizeof(float),
                           0) != MetalCommandError::None ||
                set_buffer(*pass.compute_pass, *observation_out.buffer,
                           std::uint64_t{vector} * sizeof(float), 1) != MetalCommandError::None) {
                return 113;
            }
            if (dispatch_threadgroups(*pass.compute_pass, {.width = 1, .height = 1, .depth = 1},
                                      {.width = kSimdLaneCount, .height = 1, .depth = 1}) !=
                MetalCommandError::None) {
                return 114;
            }
            auto ended = end_compute_pass(std::move(*pass.compute_pass));
            if (!ended) {
                return 115;
            }
            chain = std::move(*ended.command_buffer);
        }
        auto pending = commit(std::move(chain));
        if (!pending) {
            return 116;
        }
        ++submissions;
        auto execution = wait_until_completed(std::move(*pending.pending_execution));
        if (!execution) {
            std::cerr << "simd-sum execution failed: " << execution.failure_description.view()
                      << '\n';
            return 117;
        }
    }
    const auto observed_totals = download<float>(*observation_out.buffer, kSumObservationVectors);
    std::uint32_t shape_matches[kTreeShapeCount] = {};
    for (std::uint32_t vector = 0; vector < kSumObservationVectors; ++vector) {
        const std::span<const float> lanes(
            observation_lanes.data() + std::size_t{vector} * kSimdLaneCount, kSimdLaneCount);
        for (std::uint32_t shape = 0; shape < kTreeShapeCount; ++shape) {
            if (observed_totals[vector] == simd_tree_sum(lanes, kAllTreeShapes[shape])) {
                ++shape_matches[shape];
            }
        }
    }
    static constexpr const char* kShapeNames[kTreeShapeCount] = {
        "halving",
        "adjacent-pairs",
        "linear",
        "quad-linear+halving",
        "quad-linear+adjacent",
        "quad-linear+linear",
    };
    int pinned_shape = -1;
    int full_shape_matches = 0;
    for (std::uint32_t shape = 0; shape < kTreeShapeCount; ++shape) {
        std::cout << "simd_sum intrinsic: " << kShapeNames[shape] << " " << shape_matches[shape]
                  << "/" << kSumObservationVectors << '\n';
        if (shape_matches[shape] == kSumObservationVectors) {
            ++full_shape_matches;
            pinned_shape = static_cast<int>(shape);
        }
    }
    if (full_shape_matches != 1) {
        for (std::uint32_t vector = 0; vector < 4; ++vector) {
            const std::span<const float> lanes(
                observation_lanes.data() + std::size_t{vector} * kSimdLaneCount, kSimdLaneCount);
            std::cout << "  vector " << vector << " device total bits 0x" << std::hex
                      << std::bit_cast<std::uint32_t>(observed_totals[vector]) << std::dec;
            for (std::uint32_t shape = 0; shape < kTreeShapeCount; ++shape) {
                std::cout << " " << kShapeNames[shape] << "=0x" << std::hex
                          << std::bit_cast<std::uint32_t>(
                                 simd_tree_sum(lanes, kAllTreeShapes[shape]))
                          << std::dec;
            }
            std::cout << '\n';
        }
        std::cerr << "simd_sum intrinsic adjudication is not exactly one shape\n";
        return 95;
    }

    // Composite check: the full RMS reduction total under the pinned shape.
    auto rms_input_buffer =
        create_shared_buffer(*device.device, rms_inputs.size() * sizeof(std::uint16_t));
    auto rms_weight_buffer =
        create_shared_buffer(*device.device, rms_weights.size() * sizeof(std::uint16_t));
    auto rms_output_buffer =
        create_shared_buffer(*device.device, rms_inputs.size() * sizeof(std::uint16_t));
    auto rms_total_buffer = create_shared_buffer(*device.device, kRmsVectors * sizeof(float));
    if (!rms_input_buffer || !rms_weight_buffer || !rms_output_buffer || !rms_total_buffer) {
        return 53;
    }
    upload<std::uint16_t>(*rms_input_buffer.buffer, rms_inputs);
    upload<std::uint16_t>(*rms_weight_buffer.buffer, rms_weights);

    {
        auto command_buffer = create_command_buffer(*queue.command_queue);
        if (!command_buffer) {
            return 100;
        }
        MetalCommandBuffer chain = std::move(*command_buffer.command_buffer);
        for (std::uint32_t vector = 0; vector < kRmsVectors; ++vector) {
            auto pass = begin_compute_pass(std::move(chain));
            if (!pass) {
                return 101;
            }
            if (set_compute_pipeline(*pass.compute_pass, pipelines.rms_sum) !=
                    MetalCommandError::None ||
                set_buffer(*pass.compute_pass, *rms_input_buffer.buffer,
                           std::uint64_t{vector} * kHidden * sizeof(std::uint16_t),
                           0) != MetalCommandError::None ||
                set_buffer(*pass.compute_pass, *rms_total_buffer.buffer,
                           std::uint64_t{vector} * sizeof(float), 1) != MetalCommandError::None) {
                return 102;
            }
            if (dispatch_threadgroups(*pass.compute_pass, {.width = 1, .height = 1, .depth = 1},
                                      {.width = kHidden / 4, .height = 1, .depth = 1}) !=
                MetalCommandError::None) {
                return 103;
            }
            auto ended = end_compute_pass(std::move(*pass.compute_pass));
            if (!ended) {
                return 104;
            }
            chain = std::move(*ended.command_buffer);
        }
        auto pending = commit(std::move(chain));
        if (!pending) {
            return 105;
        }
        ++submissions;
        auto execution = wait_until_completed(std::move(*pending.pending_execution));
        if (!execution) {
            std::cerr << "rms-sum execution failed: " << execution.failure_description.view()
                      << '\n';
            return 106;
        }
    }
    const auto device_totals = download<float>(*rms_total_buffer.buffer, kRmsVectors);
    std::uint32_t composite_matches = 0;
    for (std::uint32_t vector = 0; vector < kRmsVectors; ++vector) {
        const std::span<const std::uint16_t> input(
            rms_inputs.data() + std::size_t{vector} * kHidden, kHidden);
        if (device_totals[vector] ==
            rms_total_of_squares(input, kHidden,
                                 kAllTreeShapes[static_cast<std::uint32_t>(pinned_shape)])) {
            ++composite_matches;
        }
    }
    std::cout << "rms reduction totals under pinned shape: " << composite_matches << "/"
              << kRmsVectors << '\n';
    if (composite_matches != kRmsVectors) {
        std::cerr << "composite reduction disagrees with the pinned intrinsic shape\n";
        return 96;
    }

    // Kernel fixtures: rms_only against the pinned order's device inverses.
    {
        auto command_buffer = create_command_buffer(*queue.command_queue);
        if (!command_buffer) {
            return 80;
        }
        MetalCommandBuffer chain = std::move(*command_buffer.command_buffer);
        for (std::uint32_t vector = 0; vector < kRmsVectors; ++vector) {
            auto pass = begin_compute_pass(std::move(chain));
            if (!pass) {
                return 81;
            }
            const std::uint64_t offset = std::uint64_t{vector} * kHidden * sizeof(std::uint16_t);
            if (set_compute_pipeline(*pass.compute_pass, pipelines.rms) !=
                    MetalCommandError::None ||
                set_buffer(*pass.compute_pass, *rms_input_buffer.buffer, offset, 0) !=
                    MetalCommandError::None ||
                set_buffer(*pass.compute_pass, *rms_weight_buffer.buffer, offset, 1) !=
                    MetalCommandError::None ||
                set_buffer(*pass.compute_pass, *rms_output_buffer.buffer, offset, 2) !=
                    MetalCommandError::None) {
                return 82;
            }
            if (dispatch_threadgroups(*pass.compute_pass, {.width = 1, .height = 1, .depth = 1},
                                      {.width = kHidden / 4, .height = 1, .depth = 1}) !=
                MetalCommandError::None) {
                return 83;
            }
            auto ended = end_compute_pass(std::move(*pass.compute_pass));
            if (!ended) {
                return 84;
            }
            chain = std::move(*ended.command_buffer);
        }
        auto pending = commit(std::move(chain));
        if (!pending) {
            return 85;
        }
        ++submissions;
        auto execution = wait_until_completed(std::move(*pending.pending_execution));
        if (!execution) {
            std::cerr << "rms execution failed: " << execution.failure_description.view() << '\n';
            return 86;
        }
    }

    std::uint32_t rms_matched = 0;
    std::vector<std::uint16_t> expected_vector(kHidden);
    first_mismatch_expected.clear();
    first_mismatch_actual.clear();
    for (std::uint32_t vector = 0; vector < kRmsVectors; ++vector) {
        const auto actual = download<std::uint16_t>(*rms_output_buffer.buffer, kHidden,
                                                    std::size_t{vector} * kHidden);
        const std::span<const std::uint16_t> input(
            rms_inputs.data() + std::size_t{vector} * kHidden, kHidden);
        const std::span<const std::uint16_t> weight(
            rms_weights.data() + std::size_t{vector} * kHidden, kHidden);
        rms_only_reference_with_inverse(
            input, weight, kHidden,
            device_inverse[vector * kTreeShapeCount + static_cast<std::uint32_t>(pinned_shape)],
            expected_vector);
        if (std::equal(expected_vector.begin(), expected_vector.end(), actual.begin())) {
            ++rms_matched;
        } else if (first_mismatch_expected.empty()) {
            first_mismatch_expected = expected_vector;
            first_mismatch_actual = actual;
        }
    }
    std::cout << "rms_only fixtures: " << kShapeNames[pinned_shape] << " " << rms_matched << "/"
              << kRmsVectors << '\n';
    if (rms_matched != kRmsVectors) {
        if (!first_mismatch_expected.empty()) {
            dump_mismatches("rms", first_mismatch_expected, first_mismatch_actual);
        }
        std::cerr << "rms fixtures do not match the pinned semantics\n";
        return 90;
    }

    std::cout << "kernel fixtures: PASS\n"
              << "  device: " << device.device->name() << '\n'
              << "  rsqrt: device-defined\n"
              << "  pinned multiply-add: " << (embed_fused ? "fused" : "separate") << '\n'
              << "  pinned simd sum: " << kShapeNames[pinned_shape] << '\n'
              << "  command buffers submitted: " << submissions << '\n';
    return 0;
}
