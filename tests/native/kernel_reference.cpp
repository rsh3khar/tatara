#include "kernel_reference.h"

#include <bit>
#include <cmath>
#include <limits>
#include <vector>

namespace tatara::testing {
namespace {

constexpr std::uint32_t kSimdLanes = 32;
constexpr std::uint32_t kStageSlots = 32;
constexpr std::uint32_t kValuesPerThread = 4;
constexpr std::uint32_t kNibblesPerWord = 8;

float tree_reduce(std::span<float> values, SimdTreeShape shape) noexcept {
    if (shape == SimdTreeShape::Linear) {
        float sum = 0.0f;
        for (const float value : values) {
            sum += value;
        }
        return sum;
    }
    if (shape == SimdTreeShape::AdjacentPairs) {
        std::vector<float> level(values.begin(), values.end());
        while (level.size() > 1) {
            std::vector<float> next((level.size() + 1) / 2);
            for (std::size_t pair = 0; pair < next.size(); ++pair) {
                const std::size_t left = pair * 2;
                next[pair] = left + 1 < level.size() ? level[left] + level[left + 1] : level[left];
            }
            level = std::move(next);
        }
        return level[0];
    }
    for (std::size_t offset = values.size() / 2; offset > 0; offset /= 2) {
        for (std::size_t lane = 0; lane < offset; ++lane) {
            values[lane] += values[lane + offset];
        }
    }
    return values[0];
}

} // namespace

float simd_tree_sum(std::span<const float> lanes, SimdTreeShape shape) {
    std::vector<float> values(lanes.begin(), lanes.end());
    switch (shape) {
    case SimdTreeShape::Halving:
    case SimdTreeShape::AdjacentPairs:
    case SimdTreeShape::Linear:
        return tree_reduce(values, shape);
    case SimdTreeShape::QuadLinearHalving:
    case SimdTreeShape::QuadLinearAdjacent:
    case SimdTreeShape::QuadLinearLinear: {
        std::vector<float> quads((values.size() + 3) / 4, 0.0f);
        for (std::size_t quad = 0; quad < quads.size(); ++quad) {
            float sum = 0.0f;
            for (std::size_t i = quad * 4; i < std::min(values.size(), quad * 4 + 4); ++i) {
                sum += values[i];
            }
            quads[quad] = sum;
        }
        const SimdTreeShape upper =
            shape == SimdTreeShape::QuadLinearHalving
                ? SimdTreeShape::Halving
                : (shape == SimdTreeShape::QuadLinearAdjacent ? SimdTreeShape::AdjacentPairs
                                                              : SimdTreeShape::Linear);
        return tree_reduce(quads, upper);
    }
    }
    return 0.0f;
}

namespace {} // namespace

std::uint16_t bf16_from_f32(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F800000u) == 0x7F800000u) {
        return static_cast<std::uint16_t>(bits >> 16);
    }
    const std::uint32_t rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    return static_cast<std::uint16_t>(rounded >> 16);
}

float f32_from_bf16(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

std::uint16_t flush_subnormal_bf16(std::uint16_t bits) noexcept {
    if ((bits & 0x7F80u) == 0) {
        return static_cast<std::uint16_t>(bits & 0x8000u);
    }
    return bits;
}

float flush_subnormal_f32(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F800000u) == 0) {
        return std::bit_cast<float>(bits & 0x80000000u);
    }
    return value;
}

Xorshift64Star::Xorshift64Star(std::uint64_t seed) noexcept
    : state_(seed == 0 ? 0x9E3779B97F4A7C15ull : seed) {}

std::uint64_t Xorshift64Star::next() noexcept {
    state_ ^= state_ >> 12;
    state_ ^= state_ << 25;
    state_ ^= state_ >> 27;
    return state_ * 0x2545F4914F6CDD1Dull;
}

std::uint16_t Xorshift64Star::next_finite_bf16() noexcept {
    auto bits = static_cast<std::uint16_t>(next() >> 48);
    if ((bits & 0x7F80u) == 0x7F80u) {
        bits = static_cast<std::uint16_t>((bits & 0x807Fu) | 0x3F80u);
    }
    return bits;
}

std::uint16_t Xorshift64Star::next_banded_bf16() noexcept {
    const std::uint64_t draw = next();
    const auto sign = static_cast<std::uint16_t>((draw >> 15) & 0x8000u);
    const auto exponent = static_cast<std::uint16_t>((0x70u + draw % 0x18u) << 7);
    const auto mantissa = static_cast<std::uint16_t>((draw >> 32) & 0x7Fu);
    return static_cast<std::uint16_t>(sign | exponent | mantissa);
}

float multiply_add_candidate(float quant, float scale, float bias,
                             MultiplyAddOrder order) noexcept {
#pragma clang fp contract(off)
    if (order == MultiplyAddOrder::Fused) {
        return flush_subnormal_f32(std::fma(quant, scale, bias));
    }
    const float product = flush_subnormal_f32(quant * scale);
    return flush_subnormal_f32(product + bias);
}

float rsqrt_candidate(float value, RsqrtOrder order) noexcept {
    if (order == RsqrtOrder::ReciprocalOfSqrt) {
        return 1.0f / std::sqrt(value);
    }
    return static_cast<float>(1.0 / std::sqrt(static_cast<double>(value)));
}

std::uint16_t bfloat_multiply_candidate(std::uint16_t left, std::uint16_t right) noexcept {
    const float product = flush_subnormal_f32(f32_from_bf16(flush_subnormal_bf16(left)) *
                                              f32_from_bf16(flush_subnormal_bf16(right)));
    return flush_subnormal_bf16(bf16_from_f32(product));
}

void embed_row_q4_reference(std::span<const std::uint32_t> quant_words,
                            std::span<const std::uint16_t> scales,
                            std::span<const std::uint16_t> biases, std::uint32_t token,
                            std::uint32_t hidden, std::uint32_t group_size, MultiplyAddOrder order,
                            std::span<std::uint16_t> row) {
    const std::uint32_t words_per_row = hidden / kNibblesPerWord;
    const std::uint32_t groups_per_row = hidden / group_size;
    for (std::uint32_t element = 0; element < hidden; ++element) {
        const std::uint32_t group = element / group_size;
        const float scale =
            f32_from_bf16(flush_subnormal_bf16(scales[token * groups_per_row + group]));
        const float bias =
            f32_from_bf16(flush_subnormal_bf16(biases[token * groups_per_row + group]));
        const std::uint32_t word = quant_words[token * words_per_row + element / kNibblesPerWord];
        const std::uint32_t quant = (word >> (4u * (element % kNibblesPerWord))) & 15u;
        row[element] = flush_subnormal_bf16(
            bf16_from_f32(multiply_add_candidate(static_cast<float>(quant), scale, bias, order)));
    }
}

float rms_total_of_squares(std::span<const std::uint16_t> input, std::uint32_t hidden,
                           SimdTreeShape shape) {
#pragma clang fp contract(off)
    const std::uint32_t threads = hidden / kValuesPerThread;
    const std::uint32_t simdgroups = threads / kSimdLanes;

    std::vector<float> partials(threads, 0.0f);
    for (std::uint32_t thread = 0; thread < threads; ++thread) {
        float sum = 0.0f;
        for (std::uint32_t i = 0; i < kValuesPerThread; ++i) {
            // A bfloat16 square is exact in float, so contraction cannot
            // change this accumulation observably; no adjudication axis.
            const float value = f32_from_bf16(input[thread * kValuesPerThread + i]);
            sum += value * value;
        }
        partials[thread] = sum;
    }

    std::vector<float> stage(kStageSlots, 0.0f);
    for (std::uint32_t group = 0; group < simdgroups; ++group) {
        const std::span<const float> lanes(partials.data() + group * kSimdLanes, kSimdLanes);
        stage[group] = simd_tree_sum(lanes, shape);
    }
    return simd_tree_sum(stage, shape);
}

float rms_mean_of_squares(std::span<const std::uint16_t> input, std::uint32_t hidden,
                          SimdTreeShape shape) {
    return rms_total_of_squares(input, hidden, shape) / static_cast<float>(hidden);
}

void rms_only_reference_with_inverse(std::span<const std::uint16_t> input,
                                     std::span<const std::uint16_t> weight, std::uint32_t hidden,
                                     float inverse, std::span<std::uint16_t> output) {
#pragma clang fp contract(off)
    for (std::uint32_t element = 0; element < hidden; ++element) {
        const std::uint16_t scaled = flush_subnormal_bf16(
            bf16_from_f32(flush_subnormal_f32(f32_from_bf16(input[element]) * inverse)));
        output[element] = bfloat_multiply_candidate(weight[element], scaled);
    }
}

void rms_only_reference(std::span<const std::uint16_t> input, std::span<const std::uint16_t> weight,
                        std::uint32_t hidden, float epsilon, SimdTreeShape shape,
                        RsqrtOrder rsqrt_order, std::span<std::uint16_t> output) {
    const float mean = rms_mean_of_squares(input, hidden, shape);
    const float inverse = rsqrt_candidate(mean + epsilon, rsqrt_order);
    rms_only_reference_with_inverse(input, weight, hidden, inverse, output);
}

MultiplyAddDiscriminator find_multiply_add_discriminator(std::uint64_t seed, std::uint32_t trials) {
    Xorshift64Star generator(seed);
    for (std::uint32_t trial = 0; trial < trials; ++trial) {
        const auto quant = static_cast<float>(generator.next() & 15u);
        const std::uint16_t scale = generator.next_finite_bf16();
        const std::uint16_t bias = generator.next_finite_bf16();
        const float scale_value = f32_from_bf16(flush_subnormal_bf16(scale));
        const float bias_value = f32_from_bf16(flush_subnormal_bf16(bias));
        const std::uint16_t fused = flush_subnormal_bf16(bf16_from_f32(
            multiply_add_candidate(quant, scale_value, bias_value, MultiplyAddOrder::Fused)));
        const std::uint16_t separate = flush_subnormal_bf16(bf16_from_f32(
            multiply_add_candidate(quant, scale_value, bias_value, MultiplyAddOrder::Separate)));
        if (fused != separate) {
            return {.found = true, .quant = quant, .scale = scale, .bias = bias};
        }
    }
    return {.found = false, .quant = 0.0f, .scale = 0, .bias = 0};
}

bool find_rsqrt_discriminator(std::uint64_t seed, std::uint32_t trials, float& value) {
    Xorshift64Star generator(seed);
    for (std::uint32_t trial = 0; trial < trials; ++trial) {
        const float candidate = f32_from_bf16(generator.next_finite_bf16());
        const float magnitude = candidate < 0.0f ? -candidate : candidate;
        if (magnitude == 0.0f) {
            continue;
        }
        const float first = rsqrt_candidate(magnitude, RsqrtOrder::ReciprocalOfSqrt);
        const float second = rsqrt_candidate(magnitude, RsqrtOrder::CorrectlyRounded);
        if (first != second) {
            value = magnitude;
            return true;
        }
    }
    return false;
}

bool find_tree_shape_discriminator(std::uint64_t seed, std::uint32_t trials, SimdTreeShape first,
                                   SimdTreeShape second, std::span<float> lanes) {
    Xorshift64Star generator(seed);
    for (std::uint32_t trial = 0; trial < trials; ++trial) {
        for (auto& lane : lanes) {
            float value = 0.0f;
            while (value <= 0.0f) {
                value = f32_from_bf16(generator.next_banded_bf16());
                value = value < 0.0f ? -value : value;
            }
            lane = value;
        }
        if (simd_tree_sum(lanes, first) != simd_tree_sum(lanes, second)) {
            return true;
        }
    }
    return false;
}

bool find_rms_sum_discriminator(std::uint64_t seed, std::uint32_t trials, std::uint32_t hidden,
                                float epsilon, std::span<std::uint16_t> input,
                                std::span<std::uint16_t> weight) {
    (void)epsilon;
    Xorshift64Star generator(seed);
    for (std::uint32_t trial = 0; trial < trials; ++trial) {
        for (std::uint32_t element = 0; element < hidden; ++element) {
            input[element] = generator.next_banded_bf16();
            weight[element] = generator.next_banded_bf16();
        }
        const float halving = rms_total_of_squares(input, hidden, SimdTreeShape::Halving);
        const float linear = rms_total_of_squares(input, hidden, SimdTreeShape::Linear);
        if (halving != linear) {
            return true;
        }
    }
    return false;
}

std::uint16_t bfloat_add_candidate(std::uint16_t left, std::uint16_t right) noexcept {
    const float sum = flush_subnormal_f32(f32_from_bf16(flush_subnormal_bf16(left)) +
                                          f32_from_bf16(flush_subnormal_bf16(right)));
    return flush_subnormal_bf16(bf16_from_f32(sum));
}

namespace {

std::uint16_t bfloat_subtract_reference(std::uint16_t left, std::uint16_t right) noexcept {
    const float difference = flush_subnormal_f32(f32_from_bf16(flush_subnormal_bf16(left)) -
                                                 f32_from_bf16(flush_subnormal_bf16(right)));
    return flush_subnormal_bf16(bf16_from_f32(difference));
}

std::uint16_t bfloat_divide_reference(std::uint16_t left, std::uint16_t right) noexcept {
    const float quotient = flush_subnormal_f32(f32_from_bf16(flush_subnormal_bf16(left)) /
                                               f32_from_bf16(flush_subnormal_bf16(right)));
    return flush_subnormal_bf16(bf16_from_f32(quotient));
}

float exp_model(float value, TranscendentalModel model) noexcept {
    if (model == TranscendentalModel::FloatLibm) {
        return std::exp(value);
    }
    return static_cast<float>(std::exp(static_cast<double>(value)));
}

float log_model(float value, TranscendentalModel model) noexcept {
    if (model == TranscendentalModel::FloatLibm) {
        return std::log(value);
    }
    return static_cast<float>(std::log(static_cast<double>(value)));
}

} // namespace

std::uint16_t bf16_sigmoid_candidate(std::uint16_t input, TranscendentalModel model) noexcept {
#pragma clang fp contract(off)
    const float x = f32_from_bf16(flush_subnormal_bf16(input));
    const std::uint16_t exp_b =
        flush_subnormal_bf16(bf16_from_f32(flush_subnormal_f32(exp_model(std::fabs(x), model))));
    const std::uint16_t one = 0x3F80;
    const std::uint16_t low = bfloat_divide_reference(one, bfloat_add_candidate(one, exp_b));
    if (x < 0.0f) {
        return low;
    }
    return bfloat_subtract_reference(one, low);
}

float f32_sigmoid_candidate(float input, TranscendentalModel model) noexcept {
#pragma clang fp contract(off)
    const float low = flush_subnormal_f32(
        1.0f / (1.0f + flush_subnormal_f32(exp_model(std::fabs(input), model))));
    return input < 0.0f ? low : 1.0f - low;
}

float gdn_decay_candidate(std::uint16_t shifted, std::uint16_t a_log,
                          TranscendentalModel model) noexcept {
#pragma clang fp contract(off)
    const std::uint16_t s = flush_subnormal_bf16(shifted);
    const bool positive = f32_from_bf16(s) > 0.0f;
    const std::uint16_t branch_max = positive ? s : 0;
    const std::uint16_t branch_min = positive ? 0 : s;
    const std::uint16_t difference = bfloat_subtract_reference(branch_min, branch_max);
    const std::uint16_t exp_b = flush_subnormal_bf16(
        bf16_from_f32(flush_subnormal_f32(exp_model(f32_from_bf16(difference), model))));
    const float one_plus = 1.0f + f32_from_bf16(exp_b);
    std::uint16_t log1p_b = exp_b;
    if (one_plus != 1.0f) {
        log1p_b = flush_subnormal_bf16(bf16_from_f32(flush_subnormal_f32(
            f32_from_bf16(exp_b) * (log_model(one_plus, model) / (one_plus - 1.0f)))));
    }
    const std::uint16_t softplus_b = bfloat_add_candidate(branch_max, log1p_b);
    return exp_model(-exp_model(f32_from_bf16(flush_subnormal_bf16(a_log)), model) *
                         f32_from_bf16(softplus_b),
                     model);
}

float conv4_reference(std::span<const std::uint16_t> weights,
                      std::span<const float> taps) noexcept {
#pragma clang fp contract(off)
    float acc = 0.0f;
    for (std::size_t i = 0; i < 4; ++i) {
        acc = flush_subnormal_f32(
            acc + flush_subnormal_f32(f32_from_bf16(flush_subnormal_bf16(weights[i])) * taps[i]));
    }
    return acc;
}

float q4_dot_reference(std::span<const std::uint16_t> x, std::span<const std::uint32_t> words,
                       std::span<const std::uint16_t> scales, std::span<const std::uint16_t> biases,
                       std::uint32_t k_size, SimdTreeShape tree, ChainOrder order) {
#pragma clang fp contract(off)
    std::vector<float> lane_results(kSimdLanes, 0.0f);
    const auto* packed = reinterpret_cast<const std::uint16_t*>(words.data());
    for (std::uint32_t lane = 0; lane < kSimdLanes; ++lane) {
        float result = 0.0f;
        for (std::uint32_t k = 0; k < k_size; k += 512u) {
            const std::uint32_t x0 = k + lane * 16u;
            float xv[16];
            float sum = 0.0f;
            for (std::uint32_t i = 0; i < 16u; i += 4u) {
                const float a0 = f32_from_bf16(x[x0 + i]);
                const float a1 = f32_from_bf16(x[x0 + i + 1u]);
                const float a2 = f32_from_bf16(x[x0 + i + 2u]);
                const float a3 = f32_from_bf16(x[x0 + i + 3u]);
                sum = flush_subnormal_f32(
                    sum + flush_subnormal_f32(flush_subnormal_f32(a0 + a1) + a2 + a3));
                xv[i] = a0;
                xv[i + 1u] = a1 / 16.0f;
                xv[i + 2u] = a2 / 256.0f;
                xv[i + 3u] = a3 / 4096.0f;
            }
            float quant = 0.0f;
            const std::uint32_t word0 = (k >> 2u) + lane * 4u;
            for (std::uint32_t i = 0; i < 4u; ++i) {
                const std::uint16_t word = packed[word0 + i];
                const float t0 = xv[i * 4u] * static_cast<float>(word & 0x000Fu);
                const float t1 = xv[i * 4u + 1u] * static_cast<float>(word & 0x00F0u);
                const float t2 = xv[i * 4u + 2u] * static_cast<float>(word & 0x0F00u);
                const float t3 = xv[i * 4u + 3u] * static_cast<float>(word & 0xF000u);
                if (order == ChainOrder::Fused) {
                    quant = std::fma(
                        xv[i * 4u + 3u], static_cast<float>(word & 0xF000u),
                        std::fma(xv[i * 4u + 2u], static_cast<float>(word & 0x0F00u),
                                 std::fma(xv[i * 4u + 1u], static_cast<float>(word & 0x00F0u),
                                          std::fma(xv[i * 4u], static_cast<float>(word & 0x000Fu),
                                                   quant))));
                } else {
                    quant = quant + (((t0 + t1) + t2) + t3);
                }
            }
            const std::uint32_t group = (k >> 6u) + (lane >> 2u);
            const float scale = f32_from_bf16(scales[group]);
            const float bias = f32_from_bf16(biases[group]);
            if (order == ChainOrder::Fused) {
                result = std::fma(sum, bias, std::fma(scale, quant, result));
            } else {
                result = result + (scale * quant + sum * bias);
            }
        }
        lane_results[lane] = result;
    }
    return simd_tree_sum(lane_results, tree);
}

void gdn_conv_forward(std::span<const std::uint16_t> projection,
                      std::span<const std::uint16_t> conv_state,
                      std::span<const std::uint16_t> conv_weights, std::uint32_t channels,
                      std::span<std::uint16_t> convolved, std::span<std::uint16_t> new_conv_state) {
    for (std::uint32_t channel = 0; channel < channels; ++channel) {
        const float taps[4] = {
            f32_from_bf16(conv_state[channel]),
            f32_from_bf16(conv_state[channels + channel]),
            f32_from_bf16(conv_state[2u * channels + channel]),
            f32_from_bf16(projection[channel]),
        };
        const std::uint16_t weights[4] = {
            conv_weights[channel * 4u], conv_weights[channel * 4u + 1u],
            conv_weights[channel * 4u + 2u], conv_weights[channel * 4u + 3u]};
        convolved[channel] = flush_subnormal_bf16(bf16_from_f32(conv4_reference(weights, taps)));
        new_conv_state[channel] = bf16_from_f32(taps[1]);
        new_conv_state[channels + channel] = bf16_from_f32(taps[2]);
        new_conv_state[2u * channels + channel] = bf16_from_f32(taps[3]);
    }
}

float gdn_head_rsqrt_argument(std::span<const std::uint16_t> activated, std::uint32_t head,
                              std::uint32_t head_dimension, float epsilon, SimdTreeShape tree) {
#pragma clang fp contract(off)
    const std::uint32_t threads = head_dimension / 4u;
    std::vector<float> partials(kSimdLanes, 0.0f);
    for (std::uint32_t lane = 0; lane < threads; ++lane) {
        float sum = 0.0f;
        for (std::uint32_t i = 0; i < 4u; ++i) {
            const float value = f32_from_bf16(activated[head * head_dimension + lane * 4u + i]);
            sum += value * value;
        }
        partials[lane] = sum;
    }
    return simd_tree_sum(partials, tree) / static_cast<float>(head_dimension) + epsilon;
}

void gdn_qk_normalize(std::span<const std::uint16_t> activated,
                      std::span<const float> head_inverses, const GdnGeometry& geometry,
                      std::uint16_t query_scale, std::uint16_t key_scale,
                      std::span<std::uint16_t> qk) {
    const std::uint32_t heads = 2u * geometry.key_heads;
    for (std::uint32_t head = 0; head < heads; ++head) {
        const std::uint16_t scale = head < geometry.key_heads ? query_scale : key_scale;
        for (std::uint32_t i = 0; i < geometry.head_dimension; ++i) {
            const std::uint32_t index = head * geometry.head_dimension + i;
            const std::uint16_t normed = flush_subnormal_bf16(bf16_from_f32(
                flush_subnormal_f32(f32_from_bf16(activated[index]) * head_inverses[head])));
            qk[index] = bfloat_multiply_candidate(normed, scale);
        }
    }
}

void gdn_recurrence_reference(std::span<const std::uint16_t> qk,
                              std::span<const std::uint16_t> value, std::span<const float> decay,
                              std::span<const float> beta, std::span<const float> state_in,
                              const GdnGeometry& geometry, SimdTreeShape tree, ChainOrder order,
                              std::span<std::uint16_t> output, std::span<float> state_out) {
#pragma clang fp contract(off)
    const std::uint32_t head_dimension = geometry.head_dimension;
    const std::uint32_t key_offset = geometry.key_heads * head_dimension;
    for (std::uint32_t value_head = 0; value_head < geometry.value_heads; ++value_head) {
        const std::uint32_t key_head = value_head >> 1u;
        const std::uint32_t query_base = key_head * head_dimension;
        const std::uint32_t key_base = key_offset + query_base;
        for (std::uint32_t value_dim = 0; value_dim < head_dimension; ++value_dim) {
            const std::size_t state_base =
                (std::size_t{value_head} * head_dimension + value_dim) * head_dimension;
            std::vector<float> state(head_dimension);
            std::vector<float> kv_partials(kSimdLanes, 0.0f);
            for (std::uint32_t lane = 0; lane < kSimdLanes; ++lane) {
                float kv = 0.0f;
                for (std::uint32_t i = 0; i < 4u; ++i) {
                    const std::uint32_t element = lane * 4u + i;
                    const float decayed =
                        flush_subnormal_f32(state_in[state_base + element] * decay[value_head]);
                    state[element] = decayed;
                    const float key = f32_from_bf16(qk[key_base + element]);
                    if (order == ChainOrder::Fused) {
                        kv = std::fma(decayed, key, kv);
                    } else {
                        kv = kv + flush_subnormal_f32(decayed * key);
                    }
                }
                kv_partials[lane] = kv;
            }
            const float key_value = simd_tree_sum(kv_partials, tree);
            const float delta = flush_subnormal_f32(
                flush_subnormal_f32(f32_from_bf16(value[value_head * head_dimension + value_dim]) -
                                    key_value) *
                beta[value_head]);
            std::vector<float> out_partials(kSimdLanes, 0.0f);
            for (std::uint32_t lane = 0; lane < kSimdLanes; ++lane) {
                float out = 0.0f;
                for (std::uint32_t i = 0; i < 4u; ++i) {
                    const std::uint32_t element = lane * 4u + i;
                    const float key = f32_from_bf16(qk[key_base + element]);
                    float updated;
                    if (order == ChainOrder::Fused) {
                        updated = std::fma(key, delta, state[element]);
                    } else {
                        updated = state[element] + flush_subnormal_f32(key * delta);
                    }
                    state[element] = updated;
                    state_out[state_base + element] = updated;
                    const float query = f32_from_bf16(qk[query_base + element]);
                    if (order == ChainOrder::Fused) {
                        out = std::fma(updated, query, out);
                    } else {
                        out = out + flush_subnormal_f32(updated * query);
                    }
                }
                out_partials[lane] = out;
            }
            output[value_head * head_dimension + value_dim] =
                flush_subnormal_bf16(bf16_from_f32(simd_tree_sum(out_partials, tree)));
        }
    }
}

void gdn_gate_norm_reference(std::span<const std::uint16_t> recurrence_out,
                             std::span<const std::uint16_t> gate,
                             std::span<const std::uint16_t> weight,
                             std::span<const float> head_inverses,
                             std::span<const float> gate_sigmoid, std::uint32_t heads,
                             std::uint32_t head_dimension, std::span<std::uint16_t> output) {
#pragma clang fp contract(off)
    for (std::uint32_t head = 0; head < heads; ++head) {
        for (std::uint32_t i = 0; i < head_dimension; ++i) {
            const std::uint32_t index = head * head_dimension + i;
            const std::uint16_t normed = flush_subnormal_bf16(bf16_from_f32(
                flush_subnormal_f32(f32_from_bf16(recurrence_out[index]) * head_inverses[head])));
            const std::uint16_t weighted = bfloat_multiply_candidate(weight[i], normed);
            const float gate_value = f32_from_bf16(gate[index]);
            output[index] = flush_subnormal_bf16(bf16_from_f32(flush_subnormal_f32(
                flush_subnormal_f32(gate_value * gate_sigmoid[index]) * f32_from_bf16(weighted))));
        }
    }
}

namespace {

float lane_strided_dot(std::span<const std::uint16_t> q, std::span<const std::uint16_t> keys,
                       std::uint32_t q_base, std::uint32_t k_base, std::uint32_t head_dimension) {
#pragma clang fp contract(off)
    std::vector<float> lanes(kSimdLanes, 0.0f);
    for (std::uint32_t lane = 0; lane < kSimdLanes; ++lane) {
        float dot = 0.0f;
        for (std::uint32_t d = lane; d < head_dimension; d += kSimdLanes) {
            dot += f32_from_bf16(q[q_base + d]) * f32_from_bf16(keys[k_base + d]);
        }
        lanes[lane] = dot;
    }
    return simd_tree_sum(lanes, SimdTreeShape::AdjacentPairs);
}

float halving_tree_sum_256(std::span<const float> slots) {
#pragma clang fp contract(off)
    std::vector<float> red(slots.begin(), slots.end());
    for (std::uint32_t off = 128; off; off >>= 1) {
        for (std::uint32_t i = 0; i < off; ++i) {
            red[i] += red[i + off];
        }
    }
    return red[0];
}

constexpr float kAttnScaleValue = 0.0625f;

} // namespace

float attn_rms_argument(std::span<const std::uint16_t> projection, std::uint32_t base,
                        std::uint32_t head_dimension, float epsilon) {
#pragma clang fp contract(off)
    std::vector<float> lanes(kSimdLanes, 0.0f);
    std::vector<float> stage(kStageSlots, 0.0f);
    for (std::uint32_t thread = 0; thread < 64; ++thread) {
        float acc = 0.0f;
        for (std::uint32_t j = 0; j < 4; ++j) {
            const float value = f32_from_bf16(projection[base + thread * 4 + j]);
            acc += value * value;
        }
        lanes[thread % kSimdLanes] = acc;
        if (thread % kSimdLanes == kSimdLanes - 1) {
            stage[thread / kSimdLanes] = simd_tree_sum(lanes, SimdTreeShape::AdjacentPairs);
        }
    }
    const float total = simd_tree_sum(stage, SimdTreeShape::AdjacentPairs);
    return total / static_cast<float>(head_dimension) + epsilon;
}

void attn_qk_rope_reference(std::span<const std::uint16_t> projection,
                            std::span<const std::uint16_t> q_weight,
                            std::span<const std::uint16_t> k_weight, std::uint32_t position,
                            std::uint32_t capacity, const AttnGeometry& geometry,
                            std::span<const float> head_inverses, std::span<const float> cosines,
                            std::span<const float> sines, ChainOrder order,
                            std::span<std::uint16_t> q_out, std::span<std::uint16_t> gate_out,
                            std::span<std::uint16_t> keys, std::span<std::uint16_t> values) {
#pragma clang fp contract(off)
    const std::uint32_t dim = geometry.head_dimension;
    const std::uint32_t q_gate_rows = geometry.query_heads * 2 * dim;
    const std::uint32_t v_offset = q_gate_rows + geometry.kv_heads * dim;
    const std::uint32_t heads = geometry.query_heads + geometry.kv_heads;
    std::vector<float> normed(dim);
    for (std::uint32_t head = 0; head < heads; ++head) {
        const bool is_query = head < geometry.query_heads;
        const std::uint32_t local_head = is_query ? head : head - geometry.query_heads;
        const std::uint32_t base = is_query ? head * 2 * dim : q_gate_rows + local_head * dim;
        const float inverse = head_inverses[head];
        for (std::uint32_t d = 0; d < dim; ++d) {
            const std::uint16_t scaled = flush_subnormal_bf16(
                bf16_from_f32(flush_subnormal_f32(f32_from_bf16(projection[base + d]) * inverse)));
            const std::uint16_t weight = is_query ? q_weight[d] : k_weight[d];
            normed[d] = f32_from_bf16(bfloat_multiply_candidate(weight, scaled));
        }
        for (std::uint32_t d = 0; d < dim; ++d) {
            float rotated = normed[d];
            if (d < 64) {
                const std::uint32_t pair = d & 31u;
                const float cosine = cosines[pair];
                const float sine = sines[pair];
                if (order == ChainOrder::Fused) {
                    rotated = d < 32 ? std::fma(-normed[d + 32], sine, normed[d] * cosine)
                                     : std::fma(normed[d - 32], sine, normed[d] * cosine);
                    rotated = flush_subnormal_f32(rotated);
                } else {
                    const float first = flush_subnormal_f32(normed[d] * cosine);
                    const float second =
                        flush_subnormal_f32((d < 32 ? normed[d + 32] : normed[d - 32]) * sine);
                    rotated = flush_subnormal_f32(d < 32 ? first - second : first + second);
                }
            }
            const std::uint16_t stored = flush_subnormal_bf16(bf16_from_f32(rotated));
            if (is_query) {
                q_out[head * dim + d] = stored;
                gate_out[head * dim + d] = projection[head * 2 * dim + dim + d];
            } else {
                keys[(local_head * capacity + position) * dim + d] = stored;
                values[(local_head * capacity + position) * dim + d] =
                    projection[v_offset + local_head * dim + d];
            }
        }
    }
}

void attention_decode_arguments(std::span<const std::uint16_t> q,
                                std::span<const std::uint16_t> keys, std::uint32_t head,
                                std::uint32_t context, std::uint32_t capacity,
                                const AttnGeometry& geometry, std::vector<float>& exp_arguments) {
#pragma clang fp contract(off)
    const std::uint32_t dim = geometry.head_dimension;
    const std::uint32_t kv = head >> 3u;
    const std::uint32_t n = context + 1;
    float running_max = -std::numeric_limits<float>::infinity();
    std::vector<float> scores(256);
    for (std::uint32_t tile = 0; tile < n; tile += 256) {
        const std::uint32_t count = std::min(n - tile, 256u);
        std::vector<float> slots(256, -std::numeric_limits<float>::infinity());
        for (std::uint32_t t = 0; t < count; ++t) {
            const float dot =
                lane_strided_dot(q, keys, head * dim, (kv * capacity + tile + t) * dim, dim);
            scores[t] = flush_subnormal_f32(dot * kAttnScaleValue);
            slots[t] = scores[t];
        }
        float tile_max = -std::numeric_limits<float>::infinity();
        for (std::uint32_t t = 0; t < count; ++t) {
            tile_max = std::max(tile_max, slots[t]);
        }
        const float new_max = std::max(running_max, tile_max);
        if (running_max != -std::numeric_limits<float>::infinity()) {
            exp_arguments.push_back(running_max - new_max);
        }
        for (std::uint32_t t = 0; t < count; ++t) {
            exp_arguments.push_back(scores[t] - new_max);
        }
        running_max = new_max;
    }
}

void attention_decode_reference(std::span<const std::uint16_t> q,
                                std::span<const std::uint16_t> gate,
                                std::span<const std::uint16_t> keys,
                                std::span<const std::uint16_t> values, std::uint32_t head,
                                std::uint32_t context, std::uint32_t capacity,
                                const AttnGeometry& geometry, std::span<const float> exp_values,
                                ChainOrder order, std::span<std::uint16_t> out) {
#pragma clang fp contract(off)
    const std::uint32_t dim = geometry.head_dimension;
    const std::uint32_t kv = head >> 3u;
    const std::uint32_t n = context + 1;
    float running_max = -std::numeric_limits<float>::infinity();
    float denom = 0.0f;
    std::vector<float> acc(dim, 0.0f);
    std::vector<float> scores(256);
    std::size_t cursor = 0;
    for (std::uint32_t tile = 0; tile < n; tile += 256) {
        const std::uint32_t count = std::min(n - tile, 256u);
        for (std::uint32_t t = 0; t < count; ++t) {
            const float dot =
                lane_strided_dot(q, keys, head * dim, (kv * capacity + tile + t) * dim, dim);
            scores[t] = flush_subnormal_f32(dot * kAttnScaleValue);
        }
        float tile_max = -std::numeric_limits<float>::infinity();
        for (std::uint32_t t = 0; t < count; ++t) {
            tile_max = std::max(tile_max, scores[t]);
        }
        const float new_max = std::max(running_max, tile_max);
        float rescale = 0.0f;
        if (running_max != -std::numeric_limits<float>::infinity()) {
            rescale = exp_values[cursor++];
        }
        std::vector<float> probs(256, 0.0f);
        for (std::uint32_t t = 0; t < count; ++t) {
            probs[t] = exp_values[cursor++];
        }
        const float tile_sum = halving_tree_sum_256(probs);
        for (std::uint32_t d = 0; d < dim; ++d) {
            acc[d] = flush_subnormal_f32(acc[d] * rescale);
            for (std::uint32_t t = 0; t < count; ++t) {
                const float value = f32_from_bf16(values[(kv * capacity + tile + t) * dim + d]);
                if (order == ChainOrder::Fused) {
                    acc[d] = std::fma(probs[t], value, acc[d]);
                } else {
                    acc[d] = acc[d] + flush_subnormal_f32(probs[t] * value);
                }
            }
        }
        denom = flush_subnormal_f32(flush_subnormal_f32(denom * rescale) + tile_sum);
        running_max = new_max;
    }
    for (std::uint32_t d = 0; d < dim; ++d) {
        const float value = flush_subnormal_f32(acc[d] / denom);
        const float g = f32_from_bf16(gate[head * dim + d]);
        const float sigmoid =
            flush_subnormal_f32(1.0f / (1.0f + flush_subnormal_f32(std::exp(-g))));
        out[head * dim + d] = flush_subnormal_bf16(
            bf16_from_f32(flush_subnormal_f32(flush_subnormal_f32(value * sigmoid))));
    }
}

void attention_scores_gqa4_arguments(std::span<const std::uint16_t> q,
                                     std::span<const std::uint16_t> keys, std::uint32_t head,
                                     std::uint32_t context, std::uint32_t capacity,
                                     std::uint32_t part, std::uint32_t partition,
                                     const AttnGeometry& geometry,
                                     std::vector<float>& exp_arguments) {
#pragma clang fp contract(off)
    const std::uint32_t dim = geometry.head_dimension;
    const std::uint32_t kv = head >> 3u;
    const std::uint32_t n = context + 1;
    const std::uint32_t start = partition * part;
    const std::uint32_t count = start < n ? std::min(n - start, part) : 0u;
    std::vector<float> scores(256, -std::numeric_limits<float>::infinity());
    for (std::uint32_t t = 0; t < count; ++t) {
        const float dot =
            lane_strided_dot(q, keys, head * dim, (kv * capacity + start + t) * dim, dim);
        scores[t] = flush_subnormal_f32(dot * kAttnScaleValue);
    }
    float part_max = -std::numeric_limits<float>::infinity();
    for (std::uint32_t t = 0; t < 256; ++t) {
        part_max = std::max(part_max, scores[t]);
    }
    for (std::uint32_t t = 0; t < count; ++t) {
        exp_arguments.push_back(scores[t] - part_max);
    }
}

void attention_scores_gqa4_reference(std::span<const std::uint16_t> q,
                                     std::span<const std::uint16_t> keys, std::uint32_t head,
                                     std::uint32_t context, std::uint32_t capacity,
                                     std::uint32_t part, std::uint32_t partition,
                                     const AttnGeometry& geometry,
                                     std::span<const float> exp_values, std::span<float> record) {
#pragma clang fp contract(off)
    const std::uint32_t dim = geometry.head_dimension;
    const std::uint32_t kv = head >> 3u;
    const std::uint32_t n = context + 1;
    const std::uint32_t start = partition * part;
    const std::uint32_t count = start < n ? std::min(n - start, part) : 0u;
    std::vector<float> scores(256, -std::numeric_limits<float>::infinity());
    for (std::uint32_t t = 0; t < count; ++t) {
        const float dot =
            lane_strided_dot(q, keys, head * dim, (kv * capacity + start + t) * dim, dim);
        scores[t] = flush_subnormal_f32(dot * kAttnScaleValue);
    }
    float part_max = -std::numeric_limits<float>::infinity();
    for (std::uint32_t t = 0; t < 256; ++t) {
        part_max = std::max(part_max, scores[t]);
    }
    std::vector<float> probs(256, 0.0f);
    for (std::uint32_t t = 0; t < count; ++t) {
        probs[t] = exp_values[t];
    }
    const float part_sum = halving_tree_sum_256(probs);
    for (std::uint32_t t = 0; t < 256; ++t) {
        record[t] = t < count ? probs[t] : 0.0f;
    }
    record[256] = part_max;
    record[257] = part_sum;
}

void attention_values_gqa8_reference(std::span<const float> weight_record,
                                     std::span<const std::uint16_t> values, std::uint32_t kv_head,
                                     std::uint32_t context, std::uint32_t capacity,
                                     std::uint32_t part, std::uint32_t partition,
                                     const AttnGeometry& geometry, ChainOrder order,
                                     std::span<float> partial_record) {
#pragma clang fp contract(off)
    const std::uint32_t dim = geometry.head_dimension;
    const std::uint32_t n = context + 1;
    const std::uint32_t start = partition * part;
    const std::uint32_t count = start < n ? std::min(n - start, part) : 0u;
    for (std::uint32_t d = 0; d < dim; ++d) {
        float acc = 0.0f;
        for (std::uint32_t p = 0; p < count; ++p) {
            const float weight = weight_record[p];
            const float value = f32_from_bf16(values[(kv_head * capacity + start + p) * dim + d]);
            if (order == ChainOrder::Fused) {
                acc = std::fma(weight, value, acc);
            } else {
                acc = acc + flush_subnormal_f32(weight * value);
            }
        }
        partial_record[d] = acc;
    }
    partial_record[256] = weight_record[256];
    partial_record[257] = weight_record[257];
}

void attention_combine_arguments(std::span<const float> partial_records, std::uint32_t nparts,
                                 std::vector<float>& exp_arguments) {
    float global_max = -std::numeric_limits<float>::infinity();
    for (std::uint32_t p = 0; p < nparts; ++p) {
        global_max = std::max(global_max, partial_records[p * 258 + 256]);
    }
    for (std::uint32_t p = 0; p < nparts; ++p) {
        exp_arguments.push_back(partial_records[p * 258 + 256] - global_max);
    }
}

void attention_combine_reference(std::span<const float> partial_records,
                                 std::span<const std::uint16_t> gate, std::uint32_t head,
                                 std::uint32_t nparts, const AttnGeometry& geometry,
                                 std::span<const float> exp_values,
                                 std::span<const float> gate_exp_values, ChainOrder order,
                                 std::span<std::uint16_t> out) {
#pragma clang fp contract(off)
    const std::uint32_t dim = geometry.head_dimension;
    std::vector<float> acc(dim, 0.0f);
    float denom = 0.0f;
    for (std::uint32_t p = 0; p < nparts; ++p) {
        const float weight = exp_values[p];
        for (std::uint32_t d = 0; d < dim; ++d) {
            if (order == ChainOrder::Fused) {
                acc[d] = std::fma(weight, partial_records[p * 258 + d], acc[d]);
            } else {
                acc[d] = acc[d] + flush_subnormal_f32(weight * partial_records[p * 258 + d]);
            }
        }
        if (order == ChainOrder::Fused) {
            denom = std::fma(weight, partial_records[p * 258 + 257], denom);
        } else {
            denom = denom + flush_subnormal_f32(weight * partial_records[p * 258 + 257]);
        }
    }
    for (std::uint32_t d = 0; d < dim; ++d) {
        const float value = flush_subnormal_f32(acc[d] / denom);
        const float sigmoid = flush_subnormal_f32(1.0f / (1.0f + gate_exp_values[d]));
        (void)gate;
        (void)head;
        out[d] = flush_subnormal_bf16(bf16_from_f32(flush_subnormal_f32(value * sigmoid)));
    }
}

float q4_dot_packed_reference(std::span<const std::uint16_t> x,
                              std::span<const std::uint32_t> words,
                              std::span<const std::uint16_t> scales,
                              std::span<const std::uint16_t> biases, std::uint32_t k_size,
                              SimdTreeShape tree, ChainOrder order) {
#pragma clang fp contract(off)
    std::vector<float> lane_results(kSimdLanes, 0.0f);
    for (std::uint32_t lane = 0; lane < kSimdLanes; ++lane) {
        float result = 0.0f;
        for (std::uint32_t k = 0; k < k_size; k += 512u) {
            const std::uint32_t word0 = (k >> 3u) + lane * 2u;
            const std::uint32_t group = (k >> 6u) + (lane >> 2u);
            const float scale = f32_from_bf16(scales[group]);
            const float bias = f32_from_bf16(biases[group]);
            const std::uint32_t x0 = k + lane * 16u;
            float sum = 0.0f;
            float quant = 0.0f;
            for (std::uint32_t wi = 0; wi < 2u; ++wi) {
                const std::uint32_t word = words[word0 + wi];
                const std::uint32_t i = wi * 8u;
                float even[4];
                float odd[4];
                float quant_even[4];
                float quant_odd[4];
                for (std::uint32_t j = 0; j < 4u; ++j) {
                    even[j] = f32_from_bf16(x[x0 + i + 2u * j]);
                    odd[j] = f32_from_bf16(x[x0 + i + 2u * j + 1u]);
                    quant_even[j] = static_cast<float>((word >> (8u * j)) & 15u);
                    quant_odd[j] = static_cast<float>((word >> (8u * j + 4u)) & 15u);
                }
                // dot() and the vector add are modeled sequentially; the
                // nibble products are exact in float, so only add order and
                // the block accumulation below carry rounding.
                sum = sum + ((((even[0] + odd[0]) + (even[1] + odd[1])) + (even[2] + odd[2])) +
                             (even[3] + odd[3]));
                const float dot_even = ((even[0] * quant_even[0] + even[1] * quant_even[1]) +
                                        even[2] * quant_even[2]) +
                                       even[3] * quant_even[3];
                const float dot_odd =
                    ((odd[0] * quant_odd[0] + odd[1] * quant_odd[1]) + odd[2] * quant_odd[2]) +
                    odd[3] * quant_odd[3];
                quant = quant + (dot_even + dot_odd);
            }
            if (order == ChainOrder::Fused) {
                result = std::fma(sum, bias, std::fma(scale, quant, result));
            } else {
                result = result + (scale * quant + sum * bias);
            }
        }
        lane_results[lane] = result;
    }
    return simd_tree_sum(lane_results, tree);
}

float q8_dot_reference(std::span<const std::uint16_t> x, std::span<const std::uint8_t> bytes,
                       std::span<const std::uint16_t> scales, std::span<const std::uint16_t> biases,
                       std::uint32_t k_size, SimdTreeShape tree, ChainOrder order) {
#pragma clang fp contract(off)
    std::vector<float> lane_results(kSimdLanes, 0.0f);
    for (std::uint32_t lane = 0; lane < kSimdLanes; ++lane) {
        float result = 0.0f;
        for (std::uint32_t k = 0; k < k_size; k += 256u) {
            const std::uint32_t x0 = k + lane * 8u;
            float sum = 0.0f;
            float quant = 0.0f;
            for (std::uint32_t i = 0; i < 8u; ++i) {
                const float value = f32_from_bf16(x[x0 + i]);
                sum = sum + value;
                // A bfloat16 value times a byte carries at most sixteen
                // significand bits, so the product is exact and only the
                // accumulation order rounds.
                quant = quant + value * static_cast<float>(bytes[x0 + i]);
            }
            const std::uint32_t group = (k >> 6u) + (lane >> 3u);
            const float scale = f32_from_bf16(scales[group]);
            const float bias = f32_from_bf16(biases[group]);
            if (order == ChainOrder::Fused) {
                result = std::fma(sum, bias, std::fma(scale, quant, result));
            } else {
                result = result + (scale * quant + sum * bias);
            }
        }
        lane_results[lane] = result;
    }
    return simd_tree_sum(lane_results, tree);
}

void router_select_arguments(std::span<const float> logits, std::uint32_t experts,
                             std::vector<float>& exp_arguments) {
#pragma clang fp contract(off)
    // The max tree is exactly associative over finite floats, so a linear
    // scan reproduces the sealed halving tree.
    float maximum = logits[0];
    for (std::uint32_t e = 1; e < experts; ++e) {
        maximum = std::max(maximum, logits[e]);
    }
    exp_arguments.clear();
    for (std::uint32_t e = 0; e < experts; ++e) {
        exp_arguments.push_back(logits[e] - maximum);
    }
    exp_arguments.push_back(-logits[experts]);
}

float f32_divide_candidate(float numerator, float denominator, DivisionModel model) noexcept {
#pragma clang fp contract(off)
    if (model == DivisionModel::IeeeDivision) {
        return flush_subnormal_f32(numerator / denominator);
    }
    const float reciprocal = flush_subnormal_f32(1.0f / denominator);
    return flush_subnormal_f32(numerator * reciprocal);
}

bool find_divide_discriminator(std::uint64_t seed, std::uint32_t trials, float& numerator,
                               float& denominator) {
    Xorshift64Star generator(seed);
    for (std::uint32_t trial = 0; trial < trials; ++trial) {
        const float n = f32_from_bf16(generator.next_banded_bf16());
        const float d = f32_from_bf16(generator.next_banded_bf16());
        if (d == 0.0f) {
            continue;
        }
        if (f32_divide_candidate(n, d, DivisionModel::IeeeDivision) !=
            f32_divide_candidate(n, d, DivisionModel::ReciprocalMultiply)) {
            numerator = n;
            denominator = d;
            return true;
        }
    }
    return false;
}

void router_select_normalize_arguments(std::span<const float> exp_values, std::uint32_t experts,
                                       std::vector<float>& numerators,
                                       std::vector<float>& denominators) {
#pragma clang fp contract(off)
    const std::vector<float> values(exp_values.begin(), exp_values.begin() + experts);
    const float total = halving_tree_sum_256(values);
    numerators.clear();
    denominators.clear();
    for (std::uint32_t e = 0; e < experts; ++e) {
        numerators.push_back(exp_values[e]);
        denominators.push_back(total);
    }
}

void router_select_pick(const MoeGeometry& geometry, std::uint32_t top_n,
                        std::span<const float> normalized, std::span<std::uint32_t> ids,
                        std::span<float> raw_coefficients) {
    const std::uint32_t experts = geometry.experts;
    std::vector<float> values(normalized.begin(), normalized.begin() + experts);
    std::vector<float> red(experts, 0.0f);
    std::vector<std::uint32_t> indices(experts, 0u);
    for (std::uint32_t k = 0; k < top_n; ++k) {
        for (std::uint32_t e = 0; e < experts; ++e) {
            red[e] = values[e];
            indices[e] = e;
        }
        for (std::uint32_t off = experts / 2u; off; off >>= 1u) {
            for (std::uint32_t e = 0; e < off; ++e) {
                if (red[e + off] > red[e]) {
                    red[e] = red[e + off];
                    indices[e] = indices[e + off];
                }
            }
        }
        ids[k] = indices[0];
        raw_coefficients[k] = red[0];
        values[indices[0]] = -1.0f;
    }
}

void router_select_renormalize_arguments(std::span<const float> raw_coefficients,
                                         std::uint32_t top_n, float shared_exp_value,
                                         std::vector<float>& numerators,
                                         std::vector<float>& denominators) {
#pragma clang fp contract(off)
    float selected_sum = 0.0f;
    for (std::uint32_t k = 0; k < top_n; ++k) {
        selected_sum += raw_coefficients[k];
    }
    numerators.clear();
    denominators.clear();
    for (std::uint32_t k = 0; k < top_n; ++k) {
        numerators.push_back(raw_coefficients[k]);
        denominators.push_back(selected_sum);
    }
    numerators.push_back(1.0f);
    denominators.push_back(1.0f + shared_exp_value);
}

void router_select_finalize(const MoeGeometry& geometry, std::uint32_t top_n,
                            std::span<const float> quotients, std::span<std::uint32_t> ids,
                            std::span<float> coefficients, float& shared_coefficient) {
    for (std::uint32_t k = 0; k < top_n; ++k) {
        coefficients[k] = quotients[k];
    }
    for (std::uint32_t k = top_n; k < geometry.active_experts; ++k) {
        ids[k] = kMoeSentinelId;
        coefficients[k] = 0.0f;
    }
    shared_coefficient = quotients[top_n];
}

std::uint16_t grouped_upgate_epilogue(float silu_quotient, float up) noexcept {
#pragma clang fp contract(off)
    const float product = flush_subnormal_f32(silu_quotient * up);
    return flush_subnormal_bf16(bf16_from_f32(product));
}

float grouped_down_total(std::span<const std::uint32_t> ids, std::span<const float> coefficients,
                         float shared_coefficient, std::span<const float> parts,
                         ChainOrder order) noexcept {
#pragma clang fp contract(off)
    float total = 0.0f;
    for (std::size_t slot = 0; slot < parts.size(); ++slot) {
        const bool shared = slot + 1 == parts.size();
        if (!shared && ids[slot] == kMoeSentinelId) {
            continue;
        }
        const float coefficient = shared ? shared_coefficient : coefficients[slot];
        if (order == ChainOrder::Fused) {
            total = flush_subnormal_f32(std::fma(coefficient, parts[slot], total));
        } else {
            const float product = flush_subnormal_f32(coefficient * parts[slot]);
            total = flush_subnormal_f32(total + product);
        }
    }
    return total;
}

void grouped_down_res_epilogue(float total, std::uint16_t residual, std::uint16_t& moe_out,
                               std::uint16_t& layer_out) noexcept {
#pragma clang fp contract(off)
    const std::uint16_t moe = flush_subnormal_bf16(bf16_from_f32(total));
    moe_out = moe;
    const float sum =
        flush_subnormal_f32(f32_from_bf16(flush_subnormal_bf16(residual)) + f32_from_bf16(moe));
    layer_out = flush_subnormal_bf16(bf16_from_f32(sum));
}

void residual_rms_rounded_sums(std::span<const std::uint16_t> a, std::span<const std::uint16_t> b,
                               std::span<std::uint16_t> rounded) {
    for (std::size_t i = 0; i < rounded.size(); ++i) {
        rounded[i] = bfloat_add_candidate(a[i], b[i]);
    }
}

float residual_rms_argument(std::span<const std::uint16_t> rounded, std::uint32_t hidden,
                            float epsilon, SimdTreeShape tree) {
#pragma clang fp contract(off)
    return rms_mean_of_squares(rounded, hidden, tree) + epsilon;
}

bool find_down_chain_discriminator(std::uint64_t seed, std::uint32_t trials,
                                   std::span<float> coefficients, std::span<float> parts) {
    Xorshift64Star generator(seed);
    const std::vector<std::uint32_t> ids(parts.size() - 1, 0u);
    for (std::uint32_t trial = 0; trial < trials; ++trial) {
        for (auto& coefficient : coefficients) {
            coefficient = static_cast<float>(generator.next() >> 40) * 0x1p-24f;
        }
        for (auto& part : parts) {
            part = f32_from_bf16(generator.next_banded_bf16());
        }
        const std::span<const float> slot_coefficients(coefficients.data(), parts.size() - 1);
        const float shared = coefficients[parts.size() - 1];
        const float fused =
            grouped_down_total(ids, slot_coefficients, shared, parts, ChainOrder::Fused);
        const float separate =
            grouped_down_total(ids, slot_coefficients, shared, parts, ChainOrder::Separate);
        if (fused != separate) {
            return true;
        }
    }
    return false;
}

namespace {

constexpr std::uint32_t kArgmaxGroupThreads = 256;
constexpr std::uint32_t kArgmaxStride = 65536;
constexpr std::uint32_t kArgmaxSimdgroups = 8;

struct ArgmaxBest {
    float value;
    std::uint32_t index;
};

// The kernel's explicit comparison: the candidate replaces the current best
// when strictly greater, or equal with a lower index.
ArgmaxBest argmax_merge(ArgmaxBest current, ArgmaxBest candidate) noexcept {
    if (candidate.value > current.value ||
        (candidate.value == current.value && candidate.index < current.index)) {
        return candidate;
    }
    return current;
}

} // namespace

void logits_argmax_stage1_reference(std::span<const std::uint16_t> logits, std::uint32_t count,
                                    std::uint32_t threadgroup_id, float& group_value,
                                    std::uint32_t& group_index) {
    std::vector<ArgmaxBest> threads(
        kArgmaxGroupThreads, ArgmaxBest{-std::numeric_limits<float>::infinity(), 0xffffffffu});
    for (std::uint32_t thread = 0; thread < kArgmaxGroupThreads; ++thread) {
        ArgmaxBest best = threads[thread];
        for (std::uint32_t i = threadgroup_id * kArgmaxGroupThreads + thread; i < count;
             i += kArgmaxStride) {
            const float value = f32_from_bf16(logits[i]);
            // The ascending scan keeps the lowest index through strict
            // greater alone.
            if (value > best.value) {
                best = ArgmaxBest{value, i};
            }
        }
        threads[thread] = best;
    }
    ArgmaxBest stage[kArgmaxSimdgroups];
    for (std::uint32_t simdgroup = 0; simdgroup < kArgmaxSimdgroups; ++simdgroup) {
        std::vector<ArgmaxBest> lanes(threads.begin() + simdgroup * 32,
                                      threads.begin() + (simdgroup + 1) * 32);
        for (std::uint32_t off = 16; off; off >>= 1) {
            std::vector<ArgmaxBest> next(lanes);
            for (std::uint32_t lane = 0; lane + off < 32; ++lane) {
                next[lane] = argmax_merge(lanes[lane], lanes[lane + off]);
            }
            lanes = std::move(next);
        }
        stage[simdgroup] = lanes[0];
    }
    ArgmaxBest best = stage[0];
    for (std::uint32_t simdgroup = 1; simdgroup < kArgmaxSimdgroups; ++simdgroup) {
        best = argmax_merge(best, stage[simdgroup]);
    }
    group_value = best.value;
    group_index = best.index;
}

std::uint32_t logits_argmax_stage2_reference(std::span<const float> group_values,
                                             std::span<const std::uint32_t> group_indices) {
    ArgmaxBest best{group_values[0], group_indices[0]};
    for (std::size_t group = 1; group < group_values.size(); ++group) {
        best = argmax_merge(best, ArgmaxBest{group_values[group], group_indices[group]});
    }
    return best.index;
}

} // namespace tatara::testing
