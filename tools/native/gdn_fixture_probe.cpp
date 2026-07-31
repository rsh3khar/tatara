#include "fixture_batteries.h"
#include "kernel_reference.h"
#include "tatara/backend/metal/commands.h"
#include "tatara/backend/metal/pipeline.h"
#include "tatara/backend/metal/resources.h"
#include "tatara/generated/kernel_library.h"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstring>
#include <functional>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace tatara::backend::metal;
using namespace tatara::testing;

constexpr std::uint32_t kHidden = generated::kKernelLibraryHidden;
constexpr std::uint32_t kGroup = generated::kKernelLibraryGroupSize;
constexpr std::uint32_t kKeyHeads = generated::kKernelLibraryGdnKeyHeads;
constexpr std::uint32_t kValueHeads = generated::kKernelLibraryGdnValueHeads;
constexpr std::uint32_t kHeadDim = generated::kKernelLibraryGdnHeadDimension;
constexpr std::uint32_t kQkValues = 2u * kKeyHeads * kHeadDim;
constexpr std::uint32_t kValueValues = kValueHeads * kHeadDim;
constexpr std::uint32_t kQkvRows = kQkValues + kValueValues;
constexpr std::uint32_t kBRowOffset = kQkvRows + kValueValues;
constexpr std::uint32_t kARowOffset = kBRowOffset + kValueHeads;
constexpr std::uint32_t kProjectionRows = kARowOffset + kValueHeads;
constexpr std::uint32_t kConvChannels = kQkvRows;
constexpr std::size_t kStateValues = std::size_t{kValueHeads} * kHeadDim * kHeadDim;
constexpr std::uint32_t kWordsPerRow = kHidden / 8u;
constexpr std::uint32_t kGroupsPerRow = kHidden / kGroup;
constexpr std::uint32_t kOutWordsPerRow = kValueValues / 8u;
constexpr std::uint32_t kOutGroupsPerRow = kValueValues / kGroup;
constexpr float kEpsilon = 1e-6f;
constexpr std::uint32_t kSweep = 4096;
constexpr std::uint32_t kFixtures = 2;
constexpr std::uint32_t kQ4Rows = 65;
constexpr std::uint32_t kChainSteps = 3;
constexpr SimdTreeShape kTree = SimdTreeShape::AdjacentPairs;

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

MetalBufferResult buffer_of(const MetalDevice& device, std::size_t bytes) {
    return create_shared_buffer(device, bytes);
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

} // namespace

int run_gdn_battery() {
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
        {"gdn_project", {}},
        {"gdn_prepare", {}},
        {"gdn_recurrence", {}},
        {"gdn_gate_norm", {}},
        {"gdn_outproj", {}},
        {"adjudicate_bfloat_add", {}},
        {"adjudicate_conv4", {}},
        {"adjudicate_q4_row", {}},
        {"adjudicate_q4_row_v", {}},
        {"adjudicate_bf16_sigmoid", {}},
        {"adjudicate_f32_sigmoid", {}},
        {"adjudicate_gdn_decay", {}},
        {"adjudicate_rsqrt", {}},
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
    const auto& prepare = kernels[1].pipeline;
    const auto& recurrence = kernels[2].pipeline;
    const auto& gate_norm = kernels[3].pipeline;
    const auto& outproj = kernels[4].pipeline;
    const auto& adj_add = kernels[5].pipeline;
    const auto& adj_conv = kernels[6].pipeline;
    const auto& adj_q4 = kernels[7].pipeline;
    const auto& adj_q4_v = kernels[8].pipeline;
    const auto& adj_bsig = kernels[9].pipeline;
    const auto& adj_fsig = kernels[10].pipeline;
    const auto& adj_decay = kernels[11].pipeline;
    const auto& adj_rsqrt = kernels[12].pipeline;

    Xorshift64Star generator(0x6D46EED5ull);
    const auto banded = [&] { return generator.next_banded_bf16(); };

    // Primitive adjudication inputs.
    std::vector<std::uint16_t> add_left(kSweep), add_right(kSweep);
    for (std::uint32_t i = 0; i < kSweep; ++i) {
        add_left[i] = generator.next_finite_bf16();
        add_right[i] = generator.next_finite_bf16();
    }
    std::vector<std::uint16_t> conv_w_sweep(kSweep * 4);
    std::vector<float> conv_t_sweep(kSweep * 4);
    for (std::size_t i = 0; i < conv_w_sweep.size(); ++i) {
        conv_w_sweep[i] = banded();
        conv_t_sweep[i] = f32_from_bf16(banded());
    }
    std::vector<std::uint16_t> q4_x(std::size_t{kQ4Rows} * 0 + kHidden);
    for (auto& value : q4_x) {
        value = banded();
    }
    std::vector<std::uint32_t> q4_words(std::size_t{kQ4Rows} * kWordsPerRow);
    std::vector<std::uint16_t> q4_scales(std::size_t{kQ4Rows} * kGroupsPerRow);
    std::vector<std::uint16_t> q4_biases(std::size_t{kQ4Rows} * kGroupsPerRow);
    for (auto& word : q4_words) {
        word = static_cast<std::uint32_t>(generator.next());
    }
    for (std::size_t i = 0; i < q4_scales.size(); ++i) {
        q4_scales[i] = banded();
        q4_biases[i] = banded();
    }
    // Inject a chain discriminator into row zero.
    {
        bool found = false;
        for (int trial = 0; trial < 500 && !found; ++trial) {
            for (std::uint32_t w = 0; w < kWordsPerRow; ++w) {
                q4_words[w] = static_cast<std::uint32_t>(generator.next());
            }
            for (std::uint32_t g = 0; g < kGroupsPerRow; ++g) {
                q4_scales[g] = banded();
                q4_biases[g] = banded();
            }
            const std::span<const std::uint32_t> words(q4_words.data(), kWordsPerRow);
            const std::span<const std::uint16_t> scales(q4_scales.data(), kGroupsPerRow);
            const std::span<const std::uint16_t> biases(q4_biases.data(), kGroupsPerRow);
            found =
                q4_dot_reference(q4_x, words, scales, biases, kHidden, kTree, ChainOrder::Fused) !=
                q4_dot_reference(q4_x, words, scales, biases, kHidden, kTree, ChainOrder::Separate);
        }
        if (!found) {
            std::cerr << "q4 chain discriminator search failed\n";
            return 50;
        }
    }

    // Kernel fixture inputs.
    std::vector<std::uint16_t> project_x(std::size_t{kFixtures} * kHidden);
    for (auto& value : project_x) {
        value = banded();
    }
    std::vector<std::uint32_t> qkv_words(std::size_t{kQkvRows} * kWordsPerRow);
    std::vector<std::uint32_t> z_words(std::size_t{kValueValues} * kWordsPerRow);
    std::vector<std::uint32_t> b_words(std::size_t{kValueHeads} * kWordsPerRow);
    std::vector<std::uint32_t> a_words(std::size_t{kValueHeads} * kWordsPerRow);
    for (auto* words : {&qkv_words, &z_words, &b_words, &a_words}) {
        for (auto& word : *words) {
            word = static_cast<std::uint32_t>(generator.next());
        }
    }
    std::vector<std::uint16_t> qkv_scales(std::size_t{kQkvRows} * kGroupsPerRow);
    std::vector<std::uint16_t> qkv_biases(qkv_scales.size());
    std::vector<std::uint16_t> z_scales(std::size_t{kValueValues} * kGroupsPerRow);
    std::vector<std::uint16_t> z_biases(z_scales.size());
    std::vector<std::uint16_t> b_scales(std::size_t{kValueHeads} * kGroupsPerRow);
    std::vector<std::uint16_t> b_biases(b_scales.size());
    std::vector<std::uint16_t> a_scales(b_scales.size());
    std::vector<std::uint16_t> a_biases(b_scales.size());
    for (auto* values : {&qkv_scales, &qkv_biases, &z_scales, &z_biases, &b_scales, &b_biases,
                         &a_scales, &a_biases}) {
        for (auto& value : *values) {
            value = banded();
        }
    }

    std::vector<std::uint16_t> prep_projection(std::size_t{kFixtures} * kProjectionRows);
    std::vector<std::uint16_t> prep_conv_state(std::size_t{kFixtures} * 3u * kConvChannels);
    std::vector<std::uint16_t> prep_conv_weights(std::size_t{kConvChannels} * 4u);
    for (auto* values : {&prep_projection, &prep_conv_state, &prep_conv_weights}) {
        for (auto& value : *values) {
            value = banded();
        }
    }

    std::vector<std::uint16_t> rec_qk(kQkValues);
    std::vector<std::uint16_t> rec_value(kValueValues);
    std::vector<std::uint16_t> rec_projection(kProjectionRows);
    std::vector<std::uint16_t> rec_a_log(kValueHeads);
    std::vector<std::uint16_t> rec_dt_bias(kValueHeads);
    std::vector<float> rec_state(kStateValues);
    for (auto* values : {&rec_qk, &rec_value, &rec_projection, &rec_a_log, &rec_dt_bias}) {
        for (auto& value : *values) {
            value = banded();
        }
    }
    for (auto& value : rec_state) {
        value = f32_from_bf16(banded());
    }

    std::vector<std::uint16_t> gate_y(std::size_t{kFixtures} * kValueValues);
    std::vector<std::uint16_t> gate_z(gate_y.size());
    std::vector<std::uint16_t> gate_weight(kHeadDim);
    for (auto* values : {&gate_y, &gate_z, &gate_weight}) {
        for (auto& value : *values) {
            value = banded();
        }
    }

    std::vector<std::uint16_t> out_x(std::size_t{kFixtures} * kValueValues);
    std::vector<std::uint32_t> out_words(std::size_t{kHidden} * kOutWordsPerRow);
    std::vector<std::uint16_t> out_scales(std::size_t{kHidden} * kOutGroupsPerRow);
    std::vector<std::uint16_t> out_biases(out_scales.size());
    for (auto& value : out_x) {
        value = banded();
    }
    for (auto& word : out_words) {
        word = static_cast<std::uint32_t>(generator.next());
    }
    for (std::size_t i = 0; i < out_scales.size(); ++i) {
        out_scales[i] = banded();
        out_biases[i] = banded();
    }

    // CPU precomputation feeding the query batch.
    std::vector<std::uint16_t> conv_b(std::size_t{kFixtures} * kConvChannels);
    std::vector<std::uint16_t> expected_new_conv(std::size_t{kFixtures} * 3u * kConvChannels);
    for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
        gdn_conv_forward(
            std::span<const std::uint16_t>(
                prep_projection.data() + std::size_t{fixture} * kProjectionRows, kProjectionRows),
            std::span<const std::uint16_t>(prep_conv_state.data() +
                                               std::size_t{fixture} * 3u * kConvChannels,
                                           3u * kConvChannels),
            prep_conv_weights, kConvChannels,
            std::span<std::uint16_t>(conv_b.data() + std::size_t{fixture} * kConvChannels,
                                     kConvChannels),
            std::span<std::uint16_t>(expected_new_conv.data() +
                                         std::size_t{fixture} * 3u * kConvChannels,
                                     3u * kConvChannels));
    }
    const std::uint32_t sigmoid_count = kFixtures * kConvChannels + kValueHeads;
    std::vector<std::uint16_t> sigmoid_points;
    sigmoid_points.reserve(((sigmoid_count + 255u) / 256u) * 256u);
    sigmoid_points.insert(sigmoid_points.end(), conv_b.begin(), conv_b.end());
    for (std::uint32_t head = 0; head < kValueHeads; ++head) {
        sigmoid_points.push_back(rec_projection[kBRowOffset + head]);
    }
    sigmoid_points.resize(((sigmoid_count + 255u) / 256u) * 256u, 0x3F80);

    std::vector<std::uint16_t> decay_shifted(kValueHeads);
    for (std::uint32_t head = 0; head < kValueHeads; ++head) {
        decay_shifted[head] =
            bfloat_add_candidate(rec_projection[kARowOffset + head], rec_dt_bias[head]);
    }
    std::vector<std::uint16_t> decay_shift_points(256, 0);
    std::vector<std::uint16_t> decay_alog_points(256, 0);
    std::copy(decay_shifted.begin(), decay_shifted.end(), decay_shift_points.begin());
    std::copy(rec_a_log.begin(), rec_a_log.end(), decay_alog_points.begin());

    std::vector<float> fsig_points(((gate_z.size() + 255u) / 256u) * 256u, 0.0f);
    for (std::size_t i = 0; i < gate_z.size(); ++i) {
        fsig_points[i] = f32_from_bf16(gate_z[i]);
    }

    const std::uint32_t qk_heads = 2u * kKeyHeads;
    std::vector<float> rsqrt_points(256, 1.0f);
    std::uint32_t rsqrt_cursor = 0;
    // Prepare-head arguments come after silu, which needs device sigmoid
    // values; they are computed and queried in a second query pass below.
    for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
        for (std::uint32_t head = 0; head < kValueHeads; ++head) {
            rsqrt_points[rsqrt_cursor++] = gdn_head_rsqrt_argument(
                std::span<const std::uint16_t>(gate_y.data() + std::size_t{fixture} * kValueValues,
                                               kValueValues),
                head, kHeadDim, kEpsilon, kTree);
        }
    }

    // Device buffers.
    auto dev_add_l = buffer_of(*device.device, add_left.size() * 2);
    auto dev_add_r = buffer_of(*device.device, add_right.size() * 2);
    auto dev_add_o = buffer_of(*device.device, add_left.size() * 2);
    auto dev_conv_w = buffer_of(*device.device, conv_w_sweep.size() * 2);
    auto dev_conv_t = buffer_of(*device.device, conv_t_sweep.size() * 4);
    auto dev_conv_o = buffer_of(*device.device, kSweep * 4);
    auto dev_q4_x = buffer_of(*device.device, q4_x.size() * 2);
    auto dev_q4_w = buffer_of(*device.device, q4_words.size() * 4);
    auto dev_q4_s = buffer_of(*device.device, q4_scales.size() * 2);
    auto dev_q4_b = buffer_of(*device.device, q4_biases.size() * 2);
    auto dev_q4_o = buffer_of(*device.device, kQ4Rows * 4);
    auto dev_sig_i = buffer_of(*device.device, sigmoid_points.size() * 2);
    auto dev_sig_o = buffer_of(*device.device, sigmoid_points.size() * 2);
    auto dev_fsig_i = buffer_of(*device.device, fsig_points.size() * 4);
    auto dev_fsig_o = buffer_of(*device.device, fsig_points.size() * 4);
    auto dev_dec_s = buffer_of(*device.device, decay_shift_points.size() * 2);
    auto dev_dec_a = buffer_of(*device.device, decay_alog_points.size() * 2);
    auto dev_dec_o = buffer_of(*device.device, decay_shift_points.size() * 4);
    auto dev_rsq_i = buffer_of(*device.device, rsqrt_points.size() * 4);
    auto dev_rsq_o = buffer_of(*device.device, rsqrt_points.size() * 4);
    for (auto* result :
         {&dev_add_l,  &dev_add_r, &dev_add_o, &dev_conv_w, &dev_conv_t, &dev_conv_o, &dev_q4_x,
          &dev_q4_w,   &dev_q4_s,  &dev_q4_b,  &dev_q4_o,   &dev_sig_i,  &dev_sig_o,  &dev_fsig_i,
          &dev_fsig_o, &dev_dec_s, &dev_dec_a, &dev_dec_o,  &dev_rsq_i,  &dev_rsq_o}) {
        if (!*result) {
            return 6;
        }
    }
    upload<std::uint16_t>(*dev_add_l.buffer, add_left);
    upload<std::uint16_t>(*dev_add_r.buffer, add_right);
    upload<std::uint16_t>(*dev_conv_w.buffer, conv_w_sweep);
    upload<float>(*dev_conv_t.buffer, conv_t_sweep);
    upload<std::uint16_t>(*dev_q4_x.buffer, q4_x);
    upload<std::uint32_t>(*dev_q4_w.buffer, q4_words);
    upload<std::uint16_t>(*dev_q4_s.buffer, q4_scales);
    upload<std::uint16_t>(*dev_q4_b.buffer, q4_biases);
    upload<std::uint16_t>(*dev_sig_i.buffer, sigmoid_points);
    upload<float>(*dev_fsig_i.buffer, fsig_points);
    upload<std::uint16_t>(*dev_dec_s.buffer, decay_shift_points);
    upload<std::uint16_t>(*dev_dec_a.buffer, decay_alog_points);
    upload<float>(*dev_rsq_i.buffer, rsqrt_points);

    const MetalSize kUnit{.width = 1, .height = 1, .depth = 1};
    {
        const Pass passes[] = {
            {&adj_add,
             {&*dev_add_l.buffer, &*dev_add_r.buffer, &*dev_add_o.buffer},
             {0, 0, 0},
             {.width = kSweep / 256u, .height = 1, .depth = 1},
             {.width = 256, .height = 1, .depth = 1}},
            {&adj_conv,
             {&*dev_conv_w.buffer, &*dev_conv_t.buffer, &*dev_conv_o.buffer},
             {0, 0, 0},
             {.width = kSweep / 256u, .height = 1, .depth = 1},
             {.width = 256, .height = 1, .depth = 1}},
            {&adj_q4,
             {&*dev_q4_x.buffer, &*dev_q4_w.buffer, &*dev_q4_s.buffer, &*dev_q4_b.buffer,
              &*dev_q4_o.buffer},
             {0, 0, 0, 0, 0},
             {.width = kQ4Rows, .height = 1, .depth = 1},
             {.width = 32, .height = 1, .depth = 1}},
        };
        if (const int rc = run_batch(*queue.command_queue, passes, "primitive"); rc != 0) {
            return 10 + rc;
        }
    }
    const auto add_device = download<std::uint16_t>(*dev_add_o.buffer, kSweep);
    std::uint32_t add_matches = 0;
    for (std::uint32_t i = 0; i < kSweep; ++i) {
        if (add_device[i] == bfloat_add_candidate(add_left[i], add_right[i])) {
            ++add_matches;
        }
    }
    std::cout << "bfloat add adjudication: single-rounding " << add_matches << "/" << kSweep
              << '\n';
    if (add_matches != kSweep) {
        std::cerr << "bfloat add candidate rejected\n";
        return 20;
    }
    const auto conv_device = download<float>(*dev_conv_o.buffer, kSweep);
    std::uint32_t conv_matches = 0;
    for (std::uint32_t i = 0; i < kSweep; ++i) {
        const std::span<const std::uint16_t> weights(conv_w_sweep.data() + std::size_t{i} * 4, 4);
        const std::span<const float> taps(conv_t_sweep.data() + std::size_t{i} * 4, 4);
        if (conv_device[i] == conv4_reference(weights, taps)) {
            ++conv_matches;
        }
    }
    std::cout << "conv4 adjudication: invariant-chain " << conv_matches << "/" << kSweep << '\n';
    if (conv_matches != kSweep) {
        std::cerr << "conv4 reference rejected\n";
        return 21;
    }
    const auto q4_device = download<float>(*dev_q4_o.buffer, kQ4Rows);
    std::uint32_t q4_matches[2] = {0, 0};
    for (std::uint32_t row = 0; row < kQ4Rows; ++row) {
        const std::span<const std::uint32_t> words(
            q4_words.data() + std::size_t{row} * kWordsPerRow, kWordsPerRow);
        const std::span<const std::uint16_t> scales(
            q4_scales.data() + std::size_t{row} * kGroupsPerRow, kGroupsPerRow);
        const std::span<const std::uint16_t> biases(
            q4_biases.data() + std::size_t{row} * kGroupsPerRow, kGroupsPerRow);
        if (q4_device[row] ==
            q4_dot_reference(q4_x, words, scales, biases, kHidden, kTree, ChainOrder::Fused)) {
            ++q4_matches[0];
        }
        if (q4_device[row] ==
            q4_dot_reference(q4_x, words, scales, biases, kHidden, kTree, ChainOrder::Separate)) {
            ++q4_matches[1];
        }
    }
    std::cout << "q4 chain evidence: fused " << q4_matches[0] << "/" << kQ4Rows << ", separate "
              << q4_matches[1] << "/" << kQ4Rows << "; pinned device-defined\n";

    // Device-defined q4: query the same-source observation kernels at exactly
    // the project and outproj fixture arguments.
    std::vector<std::uint32_t> concat_words;
    concat_words.reserve(std::size_t{kProjectionRows} * kWordsPerRow);
    for (const auto* words : {&qkv_words, &z_words, &b_words, &a_words}) {
        concat_words.insert(concat_words.end(), words->begin(), words->end());
    }
    std::vector<std::uint16_t> concat_scales;
    std::vector<std::uint16_t> concat_biases;
    concat_scales.reserve(std::size_t{kProjectionRows} * kGroupsPerRow);
    concat_biases.reserve(concat_scales.capacity());
    for (const auto* values : {&qkv_scales, &z_scales, &b_scales, &a_scales}) {
        concat_scales.insert(concat_scales.end(), values->begin(), values->end());
    }
    for (const auto* values : {&qkv_biases, &z_biases, &b_biases, &a_biases}) {
        concat_biases.insert(concat_biases.end(), values->begin(), values->end());
    }
    auto dev_cat_w = buffer_of(*device.device, concat_words.size() * 4);
    auto dev_cat_s = buffer_of(*device.device, concat_scales.size() * 2);
    auto dev_cat_b = buffer_of(*device.device, concat_biases.size() * 2);
    auto dev_cat_x = buffer_of(*device.device, project_x.size() * 2);
    auto dev_cat_o = buffer_of(*device.device, std::size_t{kFixtures} * kProjectionRows * 4);
    for (auto* result : {&dev_cat_w, &dev_cat_s, &dev_cat_b, &dev_cat_x, &dev_cat_o}) {
        if (!*result) {
            return 13;
        }
    }
    upload<std::uint32_t>(*dev_cat_w.buffer, concat_words);
    upload<std::uint16_t>(*dev_cat_s.buffer, concat_scales);
    upload<std::uint16_t>(*dev_cat_b.buffer, concat_biases);
    upload<std::uint16_t>(*dev_cat_x.buffer, project_x);
    {
        std::vector<Pass> passes;
        for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
            passes.push_back({&adj_q4,
                              {&*dev_cat_x.buffer, &*dev_cat_w.buffer, &*dev_cat_s.buffer,
                               &*dev_cat_b.buffer, &*dev_cat_o.buffer},
                              {std::uint64_t{fixture} * kHidden * 2, 0, 0, 0,
                               std::uint64_t{fixture} * kProjectionRows * 4},
                              {.width = kProjectionRows, .height = 1, .depth = 1},
                              {.width = 32, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "project-q4"); rc != 0) {
            return 50 + rc;
        }
    }
    const auto project_q4_device =
        download<float>(*dev_cat_o.buffer, std::size_t{kFixtures} * kProjectionRows);

    {
        const Pass passes[] = {
            {&adj_bsig,
             {&*dev_sig_i.buffer, &*dev_sig_o.buffer},
             {0, 0},
             {.width = static_cast<std::uint64_t>(sigmoid_points.size() / 256u),
              .height = 1,
              .depth = 1},
             {.width = 256, .height = 1, .depth = 1}},
            {&adj_fsig,
             {&*dev_fsig_i.buffer, &*dev_fsig_o.buffer},
             {0, 0},
             {.width = static_cast<std::uint64_t>(fsig_points.size() / 256u),
              .height = 1,
              .depth = 1},
             {.width = 256, .height = 1, .depth = 1}},
            {&adj_decay,
             {&*dev_dec_s.buffer, &*dev_dec_a.buffer, &*dev_dec_o.buffer},
             {0, 0, 0},
             kUnit,
             {.width = 256, .height = 1, .depth = 1}},
            {&adj_rsqrt,
             {&*dev_rsq_i.buffer, &*dev_rsq_o.buffer},
             {0, 0},
             kUnit,
             {.width = 256, .height = 1, .depth = 1}},
        };
        if (const int rc = run_batch(*queue.command_queue, passes, "query"); rc != 0) {
            return 30 + rc;
        }
    }
    const auto sigmoid_device = download<std::uint16_t>(*dev_sig_o.buffer, sigmoid_points.size());
    const auto fsig_device = download<float>(*dev_fsig_o.buffer, fsig_points.size());
    const auto decay_device = download<float>(*dev_dec_o.buffer, decay_shift_points.size());
    const auto gate_rsqrt_device = download<float>(*dev_rsq_o.buffer, rsqrt_points.size());
    std::uint32_t sigmoid_model_matches[2] = {0, 0};
    for (std::uint32_t i = 0; i < sigmoid_count; ++i) {
        for (int model = 0; model < 2; ++model) {
            if (sigmoid_device[i] ==
                bf16_sigmoid_candidate(sigmoid_points[i], model == 0
                                                              ? TranscendentalModel::FloatLibm
                                                              : TranscendentalModel::DoubleLibm)) {
                ++sigmoid_model_matches[model];
            }
        }
    }
    std::cout << "bf16 sigmoid evidence: float-libm " << sigmoid_model_matches[0] << "/"
              << sigmoid_count << ", double-libm " << sigmoid_model_matches[1] << "/"
              << sigmoid_count << "; using device values\n";

    // Prepare expected outputs from device-queried scalars.
    std::vector<std::uint16_t> activated(std::size_t{kFixtures} * kConvChannels);
    for (std::size_t i = 0; i < activated.size(); ++i) {
        activated[i] = bfloat_multiply_candidate(conv_b[i], sigmoid_device[i]);
    }
    std::vector<float> prep_rsqrt_points(256, 1.0f);
    std::uint32_t prep_cursor = 0;
    for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
        for (std::uint32_t head = 0; head < qk_heads; ++head) {
            prep_rsqrt_points[prep_cursor++] = gdn_head_rsqrt_argument(
                std::span<const std::uint16_t>(
                    activated.data() + std::size_t{fixture} * kConvChannels, kConvChannels),
                head, kHeadDim, kEpsilon, kTree);
        }
    }
    upload<float>(*dev_rsq_i.buffer, prep_rsqrt_points);
    {
        const Pass passes[] = {
            {&adj_rsqrt,
             {&*dev_rsq_i.buffer, &*dev_rsq_o.buffer},
             {0, 0},
             kUnit,
             {.width = 256, .height = 1, .depth = 1}},
        };
        if (const int rc = run_batch(*queue.command_queue, passes, "prepare-rsqrt"); rc != 0) {
            return 40 + rc;
        }
    }
    const auto prep_rsqrt_device = download<float>(*dev_rsq_o.buffer, prep_rsqrt_points.size());

    // gdn_project fixtures.
    auto dev_px = buffer_of(*device.device, project_x.size() * 2);
    auto dev_qkv_w = buffer_of(*device.device, qkv_words.size() * 4);
    auto dev_qkv_s = buffer_of(*device.device, qkv_scales.size() * 2);
    auto dev_qkv_b = buffer_of(*device.device, qkv_biases.size() * 2);
    auto dev_z_w = buffer_of(*device.device, z_words.size() * 4);
    auto dev_z_s = buffer_of(*device.device, z_scales.size() * 2);
    auto dev_z_b = buffer_of(*device.device, z_biases.size() * 2);
    auto dev_b_w = buffer_of(*device.device, b_words.size() * 4);
    auto dev_b_s = buffer_of(*device.device, b_scales.size() * 2);
    auto dev_b_b = buffer_of(*device.device, b_biases.size() * 2);
    auto dev_a_w = buffer_of(*device.device, a_words.size() * 4);
    auto dev_a_s = buffer_of(*device.device, a_scales.size() * 2);
    auto dev_a_b = buffer_of(*device.device, a_biases.size() * 2);
    auto dev_proj_o = buffer_of(*device.device, std::size_t{kFixtures} * kProjectionRows * 2);
    for (auto* result : {&dev_px, &dev_qkv_w, &dev_qkv_s, &dev_qkv_b, &dev_z_w, &dev_z_s, &dev_z_b,
                         &dev_b_w, &dev_b_s, &dev_b_b, &dev_a_w, &dev_a_s, &dev_a_b, &dev_proj_o}) {
        if (!*result) {
            return 7;
        }
    }
    upload<std::uint16_t>(*dev_px.buffer, project_x);
    upload<std::uint32_t>(*dev_qkv_w.buffer, qkv_words);
    upload<std::uint16_t>(*dev_qkv_s.buffer, qkv_scales);
    upload<std::uint16_t>(*dev_qkv_b.buffer, qkv_biases);
    upload<std::uint32_t>(*dev_z_w.buffer, z_words);
    upload<std::uint16_t>(*dev_z_s.buffer, z_scales);
    upload<std::uint16_t>(*dev_z_b.buffer, z_biases);
    upload<std::uint32_t>(*dev_b_w.buffer, b_words);
    upload<std::uint16_t>(*dev_b_s.buffer, b_scales);
    upload<std::uint16_t>(*dev_b_b.buffer, b_biases);
    upload<std::uint32_t>(*dev_a_w.buffer, a_words);
    upload<std::uint16_t>(*dev_a_s.buffer, a_scales);
    upload<std::uint16_t>(*dev_a_b.buffer, a_biases);
    {
        std::vector<Pass> passes;
        for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
            passes.push_back(
                {&project,
                 {&*dev_px.buffer, &*dev_qkv_w.buffer, &*dev_qkv_s.buffer, &*dev_qkv_b.buffer,
                  &*dev_z_w.buffer, &*dev_z_s.buffer, &*dev_z_b.buffer, &*dev_b_w.buffer,
                  &*dev_b_s.buffer, &*dev_b_b.buffer, &*dev_a_w.buffer, &*dev_a_s.buffer,
                  &*dev_a_b.buffer, &*dev_proj_o.buffer},
                 {std::uint64_t{fixture} * kHidden * 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                  std::uint64_t{fixture} * kProjectionRows * 2},
                 {.width = (kProjectionRows * 32u + 255u) / 256u, .height = 1, .depth = 1},
                 {.width = 256, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "project"); rc != 0) {
            return 60 + rc;
        }
    }
    std::uint32_t project_bad = 0;
    for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
        const auto actual = download<std::uint16_t>(*dev_proj_o.buffer, kProjectionRows,
                                                    std::size_t{fixture} * kProjectionRows);
        std::vector<std::uint16_t> expected(kProjectionRows);
        for (std::uint32_t row = 0; row < kProjectionRows; ++row) {
            expected[row] = flush_subnormal_bf16(bf16_from_f32(flush_subnormal_f32(
                project_q4_device[std::size_t{fixture} * kProjectionRows + row])));
        }
        project_bad += mismatches_u16(expected, actual, "project");
    }
    std::cout << "gdn_project fixtures: " << (kFixtures * kProjectionRows - project_bad) << "/"
              << kFixtures * kProjectionRows << '\n';
    if (project_bad != 0) {
        return 61;
    }

    // gdn_prepare fixtures.
    auto dev_prep_p = buffer_of(*device.device, prep_projection.size() * 2);
    auto dev_prep_cs = buffer_of(*device.device, prep_conv_state.size() * 2);
    auto dev_prep_cw = buffer_of(*device.device, prep_conv_weights.size() * 2);
    auto dev_prep_qk = buffer_of(*device.device, std::size_t{kFixtures} * kQkValues * 2);
    auto dev_prep_v = buffer_of(*device.device, std::size_t{kFixtures} * kValueValues * 2);
    auto dev_prep_z = buffer_of(*device.device, std::size_t{kFixtures} * kValueValues * 2);
    auto dev_prep_nc = buffer_of(*device.device, prep_conv_state.size() * 2);
    for (auto* result : {&dev_prep_p, &dev_prep_cs, &dev_prep_cw, &dev_prep_qk, &dev_prep_v,
                         &dev_prep_z, &dev_prep_nc}) {
        if (!*result) {
            return 8;
        }
    }
    upload<std::uint16_t>(*dev_prep_p.buffer, prep_projection);
    upload<std::uint16_t>(*dev_prep_cs.buffer, prep_conv_state);
    upload<std::uint16_t>(*dev_prep_cw.buffer, prep_conv_weights);
    {
        std::vector<Pass> passes;
        for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
            passes.push_back(
                {&prepare,
                 {&*dev_prep_p.buffer, &*dev_prep_cs.buffer, &*dev_prep_cw.buffer,
                  &*dev_prep_qk.buffer, &*dev_prep_v.buffer, &*dev_prep_z.buffer,
                  &*dev_prep_nc.buffer},
                 {std::uint64_t{fixture} * kProjectionRows * 2,
                  std::uint64_t{fixture} * 3u * kConvChannels * 2, 0,
                  std::uint64_t{fixture} * kQkValues * 2, std::uint64_t{fixture} * kValueValues * 2,
                  std::uint64_t{fixture} * kValueValues * 2,
                  std::uint64_t{fixture} * 3u * kConvChannels * 2},
                 {.width = kConvChannels / 128u + kValueValues / 128u, .height = 1, .depth = 1},
                 {.width = 128, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "prepare"); rc != 0) {
            return 70 + rc;
        }
    }
    std::uint32_t prepare_bad = 0;
    for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
        const std::span<const std::uint16_t> fixture_activated(
            activated.data() + std::size_t{fixture} * kConvChannels, kConvChannels);
        std::vector<float> head_inverses(qk_heads);
        for (std::uint32_t head = 0; head < qk_heads; ++head) {
            head_inverses[head] = prep_rsqrt_device[fixture * qk_heads + head];
        }
        std::vector<std::uint16_t> expected_qk(kQkValues);
        gdn_qk_normalize(
            fixture_activated, head_inverses,
            {.key_heads = kKeyHeads, .value_heads = kValueHeads, .head_dimension = kHeadDim},
            bf16_from_f32(generated::kKernelLibraryGdnQueryScale),
            bf16_from_f32(generated::kKernelLibraryGdnKeyScale), expected_qk);
        const auto actual_qk = download<std::uint16_t>(*dev_prep_qk.buffer, kQkValues,
                                                       std::size_t{fixture} * kQkValues);
        prepare_bad += mismatches_u16(expected_qk, actual_qk, "prepare-qk");
        std::vector<std::uint16_t> expected_v(fixture_activated.begin() + kQkValues,
                                              fixture_activated.end());
        const auto actual_v = download<std::uint16_t>(*dev_prep_v.buffer, kValueValues,
                                                      std::size_t{fixture} * kValueValues);
        prepare_bad += mismatches_u16(expected_v, actual_v, "prepare-v");
        std::vector<std::uint16_t> expected_z(kValueValues);
        for (std::uint32_t i = 0; i < kValueValues; ++i) {
            expected_z[i] = prep_projection[std::size_t{fixture} * kProjectionRows + kQkvRows + i];
        }
        const auto actual_z = download<std::uint16_t>(*dev_prep_z.buffer, kValueValues,
                                                      std::size_t{fixture} * kValueValues);
        prepare_bad += mismatches_u16(expected_z, actual_z, "prepare-z");
        const std::span<const std::uint16_t> expected_nc(
            expected_new_conv.data() + std::size_t{fixture} * 3u * kConvChannels,
            3u * kConvChannels);
        const auto actual_nc = download<std::uint16_t>(*dev_prep_nc.buffer, 3u * kConvChannels,
                                                       std::size_t{fixture} * 3u * kConvChannels);
        prepare_bad += mismatches_u16(expected_nc, actual_nc, "prepare-conv-state");
    }
    std::cout << "gdn_prepare fixtures: mismatches " << prepare_bad << '\n';
    if (prepare_bad != 0) {
        return 71;
    }

    // gdn_recurrence: one single step, then a three-step chain.
    std::vector<float> rec_decay(kValueHeads);
    std::vector<float> rec_beta(kValueHeads);
    for (std::uint32_t head = 0; head < kValueHeads; ++head) {
        rec_decay[head] = decay_device[head];
        const std::uint16_t sigmoid_b = sigmoid_device[kFixtures * kConvChannels + head];
        rec_beta[head] = f32_from_bf16(sigmoid_b);
    }
    auto dev_rec_qk = buffer_of(*device.device, rec_qk.size() * 2);
    auto dev_rec_v = buffer_of(*device.device, rec_value.size() * 2);
    auto dev_rec_p = buffer_of(*device.device, rec_projection.size() * 2);
    auto dev_rec_al = buffer_of(*device.device, rec_a_log.size() * 2);
    auto dev_rec_dt = buffer_of(*device.device, rec_dt_bias.size() * 2);
    auto dev_rec_y = buffer_of(*device.device, std::size_t{kChainSteps + 1} * kValueValues * 2);
    std::vector<MetalBufferResult> dev_states;
    for (std::uint32_t i = 0; i < kChainSteps + 2; ++i) {
        dev_states.push_back(buffer_of(*device.device, kStateValues * 4));
        if (!dev_states.back()) {
            return 9;
        }
    }
    for (auto* result :
         {&dev_rec_qk, &dev_rec_v, &dev_rec_p, &dev_rec_al, &dev_rec_dt, &dev_rec_y}) {
        if (!*result) {
            return 9;
        }
    }
    upload<std::uint16_t>(*dev_rec_qk.buffer, rec_qk);
    upload<std::uint16_t>(*dev_rec_v.buffer, rec_value);
    upload<std::uint16_t>(*dev_rec_p.buffer, rec_projection);
    upload<std::uint16_t>(*dev_rec_al.buffer, rec_a_log);
    upload<std::uint16_t>(*dev_rec_dt.buffer, rec_dt_bias);
    upload<float>(*dev_states[0].buffer, rec_state);
    {
        std::vector<Pass> passes;
        for (std::uint32_t step = 0; step <= kChainSteps; ++step) {
            const std::uint32_t in_index = step == 0 ? 0 : step;
            passes.push_back(
                {&recurrence,
                 {&*dev_rec_qk.buffer, &*dev_rec_v.buffer, &*dev_rec_p.buffer, &*dev_rec_al.buffer,
                  &*dev_rec_dt.buffer, &*dev_states[in_index].buffer, &*dev_rec_y.buffer,
                  &*dev_states[step + 1].buffer},
                 {0, 0, 0, 0, 0, 0, std::uint64_t{step} * kValueValues * 2, 0},
                 {.width = 1, .height = kHeadDim, .depth = kValueHeads},
                 {.width = 32, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "recurrence"); rc != 0) {
            return 80 + rc;
        }
    }
    std::vector<float> state_a(rec_state);
    std::uint32_t rec_matches[2] = {0, 0};
    int rec_pinned = -1;
    for (int order = 0; order < 2; ++order) {
        std::vector<float> state(rec_state);
        std::vector<float> next(kStateValues);
        std::vector<std::uint16_t> expected_y(kValueValues);
        bool all = true;
        for (std::uint32_t step = 0; step <= kChainSteps; ++step) {
            gdn_recurrence_reference(
                rec_qk, rec_value, rec_decay, rec_beta, state,
                {.key_heads = kKeyHeads, .value_heads = kValueHeads, .head_dimension = kHeadDim},
                kTree, order == 0 ? ChainOrder::Fused : ChainOrder::Separate, expected_y, next);
            const auto actual_y = download<std::uint16_t>(*dev_rec_y.buffer, kValueValues,
                                                          std::size_t{step} * kValueValues);
            const auto actual_state = download<float>(*dev_states[step + 1].buffer, kStateValues);
            if (!std::equal(expected_y.begin(), expected_y.end(), actual_y.begin()) ||
                !std::equal(next.begin(), next.end(), actual_state.begin())) {
                all = false;
                break;
            }
            state = next;
        }
        if (all) {
            ++rec_matches[order];
            rec_pinned = order;
        }
    }
    std::cout << "gdn_recurrence chain: fused " << rec_matches[0] << ", separate " << rec_matches[1]
              << " (four steps, float-exact states)\n";
    if (rec_matches[0] + rec_matches[1] != 1) {
        std::cerr << "recurrence chain adjudication is not exactly one candidate\n";
        return 81;
    }

    // gdn_gate_norm fixtures.
    auto dev_gate_y = buffer_of(*device.device, gate_y.size() * 2);
    auto dev_gate_z = buffer_of(*device.device, gate_z.size() * 2);
    auto dev_gate_w = buffer_of(*device.device, gate_weight.size() * 2);
    auto dev_gate_o = buffer_of(*device.device, gate_y.size() * 2);
    for (auto* result : {&dev_gate_y, &dev_gate_z, &dev_gate_w, &dev_gate_o}) {
        if (!*result) {
            return 11;
        }
    }
    upload<std::uint16_t>(*dev_gate_y.buffer, gate_y);
    upload<std::uint16_t>(*dev_gate_z.buffer, gate_z);
    upload<std::uint16_t>(*dev_gate_w.buffer, gate_weight);
    {
        std::vector<Pass> passes;
        for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
            passes.push_back({&gate_norm,
                              {&*dev_gate_y.buffer, &*dev_gate_z.buffer, &*dev_gate_w.buffer,
                               &*dev_gate_o.buffer},
                              {std::uint64_t{fixture} * kValueValues * 2,
                               std::uint64_t{fixture} * kValueValues * 2, 0,
                               std::uint64_t{fixture} * kValueValues * 2},
                              {.width = kValueHeads, .height = 1, .depth = 1},
                              {.width = 128, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "gate-norm"); rc != 0) {
            return 90 + rc;
        }
    }
    std::uint32_t gate_bad = 0;
    for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
        std::vector<float> head_inverses(kValueHeads);
        for (std::uint32_t head = 0; head < kValueHeads; ++head) {
            head_inverses[head] = gate_rsqrt_device[fixture * kValueHeads + head];
        }
        std::vector<float> gate_sigmoid(kValueValues);
        for (std::uint32_t i = 0; i < kValueValues; ++i) {
            gate_sigmoid[i] = fsig_device[std::size_t{fixture} * kValueValues + i];
        }
        std::vector<std::uint16_t> expected(kValueValues);
        gdn_gate_norm_reference(
            std::span<const std::uint16_t>(gate_y.data() + std::size_t{fixture} * kValueValues,
                                           kValueValues),
            std::span<const std::uint16_t>(gate_z.data() + std::size_t{fixture} * kValueValues,
                                           kValueValues),
            gate_weight, head_inverses, gate_sigmoid, kValueHeads, kHeadDim, expected);
        const auto actual = download<std::uint16_t>(*dev_gate_o.buffer, kValueValues,
                                                    std::size_t{fixture} * kValueValues);
        gate_bad += mismatches_u16(expected, actual, "gate-norm");
    }
    std::cout << "gdn_gate_norm fixtures: mismatches " << gate_bad << '\n';
    if (gate_bad != 0) {
        return 91;
    }

    // gdn_outproj fixtures.
    auto dev_ox = buffer_of(*device.device, out_x.size() * 2);
    auto dev_ow = buffer_of(*device.device, out_words.size() * 4);
    auto dev_os = buffer_of(*device.device, out_scales.size() * 2);
    auto dev_ob = buffer_of(*device.device, out_biases.size() * 2);
    auto dev_oo = buffer_of(*device.device, std::size_t{kFixtures} * kHidden * 2);
    for (auto* result : {&dev_ox, &dev_ow, &dev_os, &dev_ob, &dev_oo}) {
        if (!*result) {
            return 12;
        }
    }
    upload<std::uint16_t>(*dev_ox.buffer, out_x);
    upload<std::uint32_t>(*dev_ow.buffer, out_words);
    upload<std::uint16_t>(*dev_os.buffer, out_scales);
    upload<std::uint16_t>(*dev_ob.buffer, out_biases);
    {
        std::vector<Pass> passes;
        for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
            passes.push_back({&outproj,
                              {&*dev_ox.buffer, &*dev_ow.buffer, &*dev_os.buffer, &*dev_ob.buffer,
                               &*dev_oo.buffer},
                              {std::uint64_t{fixture} * kValueValues * 2, 0, 0, 0,
                               std::uint64_t{fixture} * kHidden * 2},
                              {.width = kHidden * 32u / 256u, .height = 1, .depth = 1},
                              {.width = 256, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "outproj"); rc != 0) {
            return 95 + rc;
        }
    }
    auto dev_ov = buffer_of(*device.device, std::size_t{kFixtures} * kHidden * 4);
    if (!dev_ov) {
        return 14;
    }
    {
        std::vector<Pass> passes;
        for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
            passes.push_back({&adj_q4_v,
                              {&*dev_ox.buffer, &*dev_ow.buffer, &*dev_os.buffer, &*dev_ob.buffer,
                               &*dev_ov.buffer},
                              {std::uint64_t{fixture} * kValueValues * 2, 0, 0, 0,
                               std::uint64_t{fixture} * kHidden * 4},
                              {.width = kHidden, .height = 1, .depth = 1},
                              {.width = 32, .height = 1, .depth = 1}});
        }
        if (const int rc = run_batch(*queue.command_queue, passes, "outproj-q4"); rc != 0) {
            return 97 + rc;
        }
    }
    const auto outproj_q4_device =
        download<float>(*dev_ov.buffer, std::size_t{kFixtures} * kHidden);
    std::uint32_t outproj_bad = 0;
    for (std::uint32_t fixture = 0; fixture < kFixtures; ++fixture) {
        std::vector<std::uint16_t> expected(kHidden);
        for (std::uint32_t row = 0; row < kHidden; ++row) {
            expected[row] = flush_subnormal_bf16(bf16_from_f32(
                flush_subnormal_f32(outproj_q4_device[std::size_t{fixture} * kHidden + row])));
        }
        const auto actual =
            download<std::uint16_t>(*dev_oo.buffer, kHidden, std::size_t{fixture} * kHidden);
        outproj_bad += mismatches_u16(expected, actual, "outproj");
    }
    std::cout << "gdn_outproj fixtures: mismatches " << outproj_bad << '\n';
    if (outproj_bad != 0) {
        return 96;
    }

    std::cout << "gdn fixtures: PASS\n"
              << "  device: " << device.device->name() << '\n'
              << "  q4 composite: device-defined\n"
              << "  pinned recurrence chain: " << (rec_pinned == 0 ? "fused" : "separate") << '\n'
              << "  command buffers submitted: " << submissions << '\n';
    return 0;
}
