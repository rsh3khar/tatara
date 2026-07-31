#include "kernel_reference.h"

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

namespace {

using namespace tatara::testing;

constexpr std::uint32_t kTestHidden = 256;
constexpr float kRmsEpsilon = 1e-6f;

int test_bf16_conversion() {
    if (bf16_from_f32(0.0f) != 0x0000 || bf16_from_f32(1.0f) != 0x3F80 ||
        bf16_from_f32(-2.0f) != 0xC000) {
        return 1;
    }
    // 0x3F808000 sits exactly between 0x3F80 and 0x3F81; even wins.
    if (bf16_from_f32(std::bit_cast<float>(0x3F808000u)) != 0x3F80) {
        return 2;
    }
    // 0x3F818000 also ties; the odd upper half rounds up to even 0x3F82.
    if (bf16_from_f32(std::bit_cast<float>(0x3F818000u)) != 0x3F82) {
        return 3;
    }
    if (bf16_from_f32(std::bit_cast<float>(0x7F800000u)) != 0x7F80) {
        return 4;
    }
    if (f32_from_bf16(0x3F80) != 1.0f || f32_from_bf16(0xC000) != -2.0f) {
        return 5;
    }
    return 0;
}

int test_flush_to_zero() {
    // Subnormal input times large: the flushed input zeroes the product.
    if (bfloat_multiply_candidate(0x0001, 0x7F00) != 0x0000) {
        return 50;
    }
    if (bfloat_multiply_candidate(0x3F80, 0x8014) != 0x8000) {
        return 51;
    }
    // Normal times normal landing subnormal: the result flushes with sign.
    if (bfloat_multiply_candidate(0x0080, 0x3F00) != 0x0000) {
        return 52;
    }
    if (bfloat_multiply_candidate(0x8080, 0x3F00) != 0x8000) {
        return 53;
    }
    if (flush_subnormal_bf16(0x0080) != 0x0080 || flush_subnormal_bf16(0x8000) != 0x8000) {
        return 54;
    }
    // A float-subnormal product must flush instead of rounding up into the
    // minimum normal; search for a pair that separates the two behaviors.
    Xorshift64Star generator(23);
    bool separated = false;
    for (int trial = 0; trial < 200000 && !separated; ++trial) {
        const std::uint16_t left = generator.next_finite_bf16();
        const std::uint16_t right = generator.next_finite_bf16();
        const float raw =
            f32_from_bf16(flush_subnormal_bf16(left)) * f32_from_bf16(flush_subnormal_bf16(right));
        const std::uint16_t unflushed = flush_subnormal_bf16(bf16_from_f32(raw));
        const std::uint16_t flushed = bfloat_multiply_candidate(left, right);
        if ((unflushed & 0x7FFFu) == 0x0080u && (flushed & 0x7FFFu) == 0u) {
            separated = true;
        }
    }
    if (!separated) {
        return 55;
    }
    return 0;
}

int test_embed_hand_vector() {
    const std::array<std::uint32_t, 1> words = {0x76543210u};
    const std::array<std::uint16_t, 1> scales = {bf16_from_f32(2.0f)};
    const std::array<std::uint16_t, 1> biases = {bf16_from_f32(-1.0f)};
    std::array<std::uint16_t, 8> fused_row = {};
    std::array<std::uint16_t, 8> separate_row = {};
    embed_row_q4_reference(words, scales, biases, 0, 8, 8, MultiplyAddOrder::Fused, fused_row);
    embed_row_q4_reference(words, scales, biases, 0, 8, 8, MultiplyAddOrder::Separate,
                           separate_row);
    for (std::uint32_t element = 0; element < 8; ++element) {
        const float expected = 2.0f * static_cast<float>(element) - 1.0f;
        if (fused_row[element] != bf16_from_f32(expected) ||
            separate_row[element] != fused_row[element]) {
            return 10 + static_cast<int>(element);
        }
    }
    return 0;
}

int test_rms_closed_form() {
    std::vector<std::uint16_t> ones(kTestHidden, bf16_from_f32(1.0f));
    std::vector<std::uint16_t> output(kTestHidden, 0);
    for (const SimdTreeShape shape : kAllTreeShapes) {
        for (const auto rsqrt : {RsqrtOrder::ReciprocalOfSqrt, RsqrtOrder::CorrectlyRounded}) {
            rms_only_reference(ones, ones, kTestHidden, kRmsEpsilon, shape, rsqrt, output);
            for (const std::uint16_t value : output) {
                if (value != 0x3F80) {
                    return 20;
                }
            }
        }
    }
    return 0;
}

int test_discriminators_exist() {
    const auto multiply_add = find_multiply_add_discriminator(7, 100000);
    if (!multiply_add.found) {
        return 30;
    }
    float rsqrt_input = 0.0f;
    if (!find_rsqrt_discriminator(11, 100000, rsqrt_input) || rsqrt_input <= 0.0f) {
        return 31;
    }
    // The discriminator must exist at the production shape, where the
    // wide-gap condition has room to occur.
    constexpr std::uint32_t kProductionHidden = 2048;
    std::vector<std::uint16_t> input(kProductionHidden, 0);
    std::vector<std::uint16_t> weight(kProductionHidden, 0);
    if (!find_rms_sum_discriminator(17, 20000, kProductionHidden, kRmsEpsilon, input, weight)) {
        return 33;
    }
    return 0;
}

int test_tree_shapes_distinct() {
    std::vector<float> lanes(32);
    for (int a = 0; a < 6; ++a) {
        for (int b = a + 1; b < 6; ++b) {
            if (!find_tree_shape_discriminator(29, 200000, kAllTreeShapes[a], kAllTreeShapes[b],
                                               lanes)) {
                return 60 + a * 6 + b;
            }
        }
    }
    return 0;
}

int test_gdn_oracle() {
    // The convolution chain is contraction-invariant: bfloat16 times
    // bfloat16 is exact in float, so a fused chain cannot differ. Verify the
    // invariance directly against a fused evaluation.
    Xorshift64Star generator(31);
    std::vector<float> taps(4);
    std::vector<std::uint16_t> weights(4);
    for (int trial = 0; trial < 20000; ++trial) {
        float fused = 0.0f;
        for (std::size_t i = 0; i < 4; ++i) {
            taps[i] = f32_from_bf16(generator.next_banded_bf16());
            weights[i] = generator.next_banded_bf16();
        }
        for (std::size_t i = 0; i < 4; ++i) {
            fused = flush_subnormal_f32(std::fma(f32_from_bf16(weights[i]), taps[i], fused));
        }
        if (conv4_reference(weights, taps) != fused) {
            return 70;
        }
    }
    std::vector<std::uint16_t> x(512);
    std::vector<std::uint32_t> words(64);
    std::vector<std::uint16_t> scales(8);
    std::vector<std::uint16_t> biases(8);
    bool q4_differs = false;
    for (int trial = 0; trial < 2000 && !q4_differs; ++trial) {
        for (auto& value : x) {
            value = generator.next_banded_bf16();
        }
        for (auto& word : words) {
            word = static_cast<std::uint32_t>(generator.next());
        }
        for (std::size_t i = 0; i < scales.size(); ++i) {
            scales[i] = generator.next_banded_bf16();
            biases[i] = generator.next_banded_bf16();
        }
        q4_differs = q4_dot_reference(x, words, scales, biases, 512, SimdTreeShape::AdjacentPairs,
                                      ChainOrder::Fused) !=
                     q4_dot_reference(x, words, scales, biases, 512, SimdTreeShape::AdjacentPairs,
                                      ChainOrder::Separate);
    }
    if (!q4_differs) {
        return 71;
    }
    bool sigmoid_differs = false;
    for (int trial = 0; trial < 200000 && !sigmoid_differs; ++trial) {
        const float input = f32_from_bf16(generator.next_banded_bf16());
        sigmoid_differs = f32_sigmoid_candidate(input, TranscendentalModel::FloatLibm) !=
                          f32_sigmoid_candidate(input, TranscendentalModel::DoubleLibm);
    }
    if (!sigmoid_differs) {
        return 72;
    }

    // Closed form: zero q/k and zero beta leave the decayed state untouched
    // and produce zero output.
    const GdnGeometry geometry{.key_heads = 2, .value_heads = 4, .head_dimension = 128};
    const std::uint32_t qk_values = 2 * geometry.key_heads * geometry.head_dimension;
    const std::uint32_t value_values = geometry.value_heads * geometry.head_dimension;
    const std::size_t state_values =
        std::size_t{geometry.value_heads} * geometry.head_dimension * geometry.head_dimension;
    std::vector<std::uint16_t> qk(qk_values, 0);
    std::vector<std::uint16_t> value(value_values, 0);
    std::vector<float> decay(geometry.value_heads, 1.0f);
    std::vector<float> beta(geometry.value_heads, 0.0f);
    std::vector<float> state_in(state_values);
    for (auto& element : state_in) {
        element = f32_from_bf16(generator.next_banded_bf16());
    }
    std::vector<std::uint16_t> output(value_values, 0xFFFF);
    std::vector<float> state_out(state_values, -1.0f);
    for (auto& element : value) {
        element = generator.next_banded_bf16();
    }
    gdn_recurrence_reference(qk, value, decay, beta, state_in, geometry,
                             SimdTreeShape::AdjacentPairs, ChainOrder::Fused, output, state_out);
    for (std::size_t i = 0; i < state_values; ++i) {
        if (state_out[i] != state_in[i]) {
            return 73;
        }
    }
    for (const std::uint16_t element : output) {
        if (element != 0x0000) {
            return 74;
        }
    }
    return 0;
}

int test_attention_oracle() {
    const AttnGeometry geometry{.query_heads = 16, .kv_heads = 2, .head_dimension = 256};
    // Argument-count law: one probability argument per position plus one
    // rescale argument per tile after the first.
    Xorshift64Star generator(37);
    std::vector<std::uint16_t> q(16 * 256);
    std::vector<std::uint16_t> keys(2 * 512 * 256);
    for (auto& value : q) {
        value = generator.next_banded_bf16();
    }
    for (auto& value : keys) {
        value = generator.next_banded_bf16();
    }
    std::vector<float> arguments;
    attention_decode_arguments(q, keys, 0, 0, 512, geometry, arguments);
    if (arguments.size() != 1) {
        return 80;
    }
    arguments.clear();
    attention_decode_arguments(q, keys, 3, 300, 512, geometry, arguments);
    if (arguments.size() != 302) {
        return 81;
    }

    // Closed form at context 0: the single probability is exp(0) = 1, the
    // denominator is 1, so out = bf16(value * sigmoid(gate)).
    std::vector<std::uint16_t> values(2 * 512 * 256);
    for (auto& value : values) {
        value = generator.next_banded_bf16();
    }
    std::vector<std::uint16_t> gate(16 * 256, 0x0000);
    const std::vector<float> exps = {1.0f};
    std::vector<std::uint16_t> out(16 * 256, 0xFFFF);
    attention_decode_reference(q, gate, keys, values, 0, 0, 512, geometry, exps, ChainOrder::Fused,
                               out);
    for (std::uint32_t d = 0; d < 256; ++d) {
        const float value = f32_from_bf16(values[d]);
        const std::uint16_t expected =
            flush_subnormal_bf16(bf16_from_f32(flush_subnormal_f32(value * 0.5f)));
        if (out[d] != expected) {
            return 82;
        }
    }

    // The RoPE rotation chain axis must be discriminable.
    std::vector<std::uint16_t> projection(16 * 512 + 2 * 256 + 2 * 256);
    std::vector<std::uint16_t> weight(256, 0x3F80);
    std::vector<float> inverses(18, 1.0f);
    std::vector<float> cosines(32), sines(32);
    bool differs = false;
    for (int trial = 0; trial < 2000 && !differs; ++trial) {
        for (auto& value : projection) {
            value = generator.next_banded_bf16();
        }
        for (std::uint32_t p = 0; p < 32; ++p) {
            // Full-precision trig values in [-1, 1], as the device produces.
            cosines[p] = static_cast<float>(static_cast<std::int64_t>(generator.next() >> 32) -
                                            2147483648LL) /
                         2147483648.0f;
            sines[p] = static_cast<float>(static_cast<std::int64_t>(generator.next() >> 32) -
                                          2147483648LL) /
                       2147483648.0f;
        }
        std::vector<std::uint16_t> q_a(16 * 256), q_b(16 * 256), gate_a(16 * 256), gate_b(16 * 256);
        std::vector<std::uint16_t> keys_a(2 * 8 * 256), keys_b(2 * 8 * 256), vals_a(2 * 8 * 256),
            vals_b(2 * 8 * 256);
        attn_qk_rope_reference(projection, weight, weight, 2, 8, geometry, inverses, cosines, sines,
                               ChainOrder::Fused, q_a, gate_a, keys_a, vals_a);
        attn_qk_rope_reference(projection, weight, weight, 2, 8, geometry, inverses, cosines, sines,
                               ChainOrder::Separate, q_b, gate_b, keys_b, vals_b);
        differs = q_a != q_b || keys_a != keys_b;
    }
    if (!differs) {
        return 83;
    }
    return 0;
}

int test_generator_determinism() {
    Xorshift64Star first(7);
    Xorshift64Star second(7);
    for (int i = 0; i < 1000; ++i) {
        if (first.next() != second.next()) {
            return 40;
        }
    }
    return 0;
}

} // namespace

int test_moe_oracle() {
    // Packed-Q4 addressing: uniform 1.0 inputs except element 1 = 2.0; the
    // single word 0x000000A0 places nibble 10 in byte 0's high half, which
    // must map to the odd element 1. Scales 1, biases 0.
    std::vector<std::uint16_t> x(512, 0x3F80);
    x[1] = 0x4000;
    std::vector<std::uint32_t> words(64, 0u);
    words[0] = 0x000000A0u;
    std::vector<std::uint16_t> ones(8, 0x3F80);
    std::vector<std::uint16_t> zeros(8, 0x0000);
    const float packed = q4_dot_packed_reference(
        x, words, ones, zeros, 512, SimdTreeShape::AdjacentPairs, ChainOrder::Separate);
    if (packed != 20.0f) {
        return 60;
    }
    // Affine terms: scale 2, bias 0.5, sum of x = 513.
    std::vector<std::uint16_t> twos(8, 0x4000);
    std::vector<std::uint16_t> halves(8, 0x3F00);
    const float affine = q4_dot_packed_reference(
        x, words, twos, halves, 512, SimdTreeShape::AdjacentPairs, ChainOrder::Separate);
    if (affine != 2.0f * 20.0f + 513.0f * 0.5f) {
        return 61;
    }
    // Q8 addressing: byte 3 = 7 against element 3 = 2.0, sum 513.
    std::vector<std::uint16_t> x8(256, 0x3F80);
    x8[3] = 0x4000;
    std::vector<std::uint8_t> bytes(256, 0u);
    bytes[3] = 7u;
    std::vector<std::uint16_t> group_scales(4, 0x3F80);
    std::vector<std::uint16_t> group_biases(4, 0x0000);
    const float q8 = q8_dot_reference(x8, bytes, group_scales, group_biases, 256,
                                      SimdTreeShape::AdjacentPairs, ChainOrder::Separate);
    if (q8 != 14.0f) {
        return 62;
    }
    // Selection: the strict-greater tree resolves exact ties by tree
    // position — the lower slot survives each equal merge, so the winner is
    // the tied index with the lowest bit-reversed value, not the lowest
    // index. rev(200)=19 beats rev(3)=192; the all-tied remainder fills in
    // bit-reversal order 0, 128, 64, 192, 32, ...
    const MoeGeometry geometry{256, 8, 512};
    std::vector<float> exp_values(257, 1.0f);
    exp_values[3] = 4.0f;
    exp_values[200] = 4.0f;
    std::vector<float> numerators, denominators;
    router_select_normalize_arguments(exp_values, 256, numerators, denominators);
    if (numerators.size() != 256 || numerators[3] != 4.0f || denominators[0] != 262.0f) {
        return 78;
    }
    std::vector<float> normalized(256, 0.0f);
    for (std::size_t e = 0; e < 256; ++e) {
        normalized[e] =
            f32_divide_candidate(numerators[e], denominators[e], DivisionModel::IeeeDivision);
    }
    std::vector<std::uint32_t> ids(8, 0u);
    std::vector<float> raw_coefficients(8, 0.0f);
    router_select_pick(geometry, 8, normalized, ids, raw_coefficients);
    const std::uint32_t expected_ids[8] = {200, 3, 0, 128, 64, 192, 32, 160};
    for (std::size_t k = 0; k < 8; ++k) {
        if (ids[k] != expected_ids[k]) {
            return 63;
        }
    }
    router_select_renormalize_arguments(raw_coefficients, 8, 1.0f, numerators, denominators);
    if (numerators.size() != 9 || numerators[8] != 1.0f || denominators[8] != 2.0f) {
        return 79;
    }
    std::vector<float> quotients(9, 0.0f);
    for (std::size_t k = 0; k < 9; ++k) {
        quotients[k] =
            f32_divide_candidate(numerators[k], denominators[k], DivisionModel::IeeeDivision);
    }
    std::vector<float> coefficients(8, 0.0f);
    float shared_coefficient = 0.0f;
    router_select_finalize(geometry, 8, quotients, ids, coefficients, shared_coefficient);
    float renormalized = 0.0f;
    for (std::size_t k = 0; k < 8; ++k) {
        renormalized += coefficients[k];
    }
    if (renormalized < 0.999999f || renormalized > 1.000001f) {
        return 64;
    }
    if (coefficients[0] != coefficients[1] || coefficients[0] <= coefficients[2]) {
        return 65;
    }
    if (shared_coefficient != 0.5f) {
        return 66;
    }
    // top_n below the slot count fills sentinels with zero coefficients.
    router_select_pick(geometry, 6, normalized, ids, raw_coefficients);
    router_select_renormalize_arguments(raw_coefficients, 6, 1.0f, numerators, denominators);
    if (numerators.size() != 7) {
        return 80;
    }
    quotients.assign(7, 0.25f);
    router_select_finalize(geometry, 6, quotients, ids, coefficients, shared_coefficient);
    if (ids[6] != kMoeSentinelId || ids[7] != kMoeSentinelId || coefficients[6] != 0.0f ||
        coefficients[7] != 0.0f || shared_coefficient != 0.25f) {
        return 67;
    }
    // The division candidates must be discriminable.
    float divide_numerator = 0.0f;
    float divide_denominator = 0.0f;
    if (!find_divide_discriminator(0xD1EC0DE5ull, 200000, divide_numerator, divide_denominator)) {
        return 81;
    }
    // Argument builder: softmax arguments against the tree maximum, then the
    // negated shared logit.
    std::vector<float> logits(257, 0.5f);
    logits[5] = 2.0f;
    logits[256] = -1.5f;
    std::vector<float> exp_arguments;
    router_select_arguments(logits, 256, exp_arguments);
    if (exp_arguments.size() != 257 || exp_arguments[5] != 0.0f || exp_arguments[0] != -1.5f ||
        exp_arguments[256] != 1.5f) {
        return 68;
    }
    // Up-gate epilogue closed forms on device-queried silu quotients.
    if (grouped_upgate_epilogue(0.0f, 5.0f) != 0x0000) {
        return 69;
    }
    if (grouped_upgate_epilogue(1.0f, 3.0f) != 0x4040) {
        return 70;
    }
    if (grouped_upgate_epilogue(0.5f, 1.0f) != 0x3F00) {
        return 71;
    }
    // Down accumulation: the chain axis discriminates at float resolution,
    // and a sentinel slot's part cannot reach the total.
    std::vector<float> chain_coefficients(9, 0.0f);
    std::vector<float> chain_parts(9, 0.0f);
    if (!find_down_chain_discriminator(0x9E3779B97F4A7C15ull, 4096, chain_coefficients,
                                       chain_parts)) {
        return 72;
    }
    std::vector<std::uint32_t> slot_ids(8, 0u);
    slot_ids[2] = kMoeSentinelId;
    std::vector<float> parts_a(chain_parts.begin(), chain_parts.end());
    std::vector<float> parts_b(chain_parts.begin(), chain_parts.end());
    parts_a[2] = 100.0f;
    parts_b[2] = -3.0f;
    const std::span<const float> slot_coefficients(chain_coefficients.data(), 8);
    const float total_a =
        grouped_down_total(slot_ids, slot_coefficients, 0.25f, parts_a, ChainOrder::Separate);
    const float total_b =
        grouped_down_total(slot_ids, slot_coefficients, 0.25f, parts_b, ChainOrder::Separate);
    if (total_a != total_b) {
        return 73;
    }
    std::uint16_t moe_out = 0;
    std::uint16_t layer_out = 0;
    grouped_down_res_epilogue(1.5f, 0x3F80, moe_out, layer_out);
    if (moe_out != 0x3FC0 || layer_out != 0x4020) {
        return 74;
    }
    // Fused residual entry: 1.0 + 2^-9 rounds down to 1.0; 1.0 + 2^-8 is the
    // tie and even wins; the uniform closed form gives mean 4 + epsilon.
    const std::uint16_t small[1] = {0x3B00};
    const std::uint16_t tie[1] = {0x3B80};
    const std::uint16_t unit[1] = {0x3F80};
    std::uint16_t rounded[1] = {0};
    residual_rms_rounded_sums(unit, small, rounded);
    if (rounded[0] != 0x3F80) {
        return 75;
    }
    residual_rms_rounded_sums(unit, tie, rounded);
    if (rounded[0] != 0x3F80) {
        return 76;
    }
    std::vector<std::uint16_t> uniform(2048, 0x4000);
    if (residual_rms_argument(uniform, 2048, kRmsEpsilon, SimdTreeShape::AdjacentPairs) !=
        4.0f + 1e-6f) {
        return 77;
    }
    return 0;
}

int test_head_oracle() {
    // Property: the modeled two-stage argmax equals the brute-force
    // lowest-index argmax on random vectors with ties planted at every
    // merge level. count exercises the grid-stride tail (70000 spans one
    // full stride plus a partial second pass).
    constexpr std::uint32_t kCount = 70000;
    constexpr std::uint32_t kGroups = 256;
    Xorshift64Star generator(0xA96Aull);
    std::vector<std::uint16_t> logits(kCount);
    for (std::uint32_t trial = 0; trial < 24; ++trial) {
        for (auto& value : logits) {
            value = generator.next_banded_bf16();
        }
        // 0x7F00 exceeds every banded draw, so planted pairs are the maxima.
        switch (trial % 6) {
        case 0:
            logits[123] = 0x7F00;
            logits[60123] = 0x7F00;
            break;
        case 1: // same simdgroup: threadgroup 4, lanes 5 and 17
            logits[4 * 256 + 5] = 0x7F00;
            logits[4 * 256 + 17] = 0x7F00;
            break;
        case 2: // same threadgroup, different simdgroups
            logits[9 * 256 + 10] = 0x7F00;
            logits[9 * 256 + 200] = 0x7F00;
            break;
        case 3: // different threadgroups
            logits[300] = 0x7F00;
            logits[40000] = 0x7F00;
            break;
        case 4: // same thread, one stride apart: kept by the ascending scan
            logits[1234] = 0x7F00;
            logits[1234 + 65536] = 0x7F00;
            break;
        default: // no planted tie: pure random
            break;
        }
        float brute_best = f32_from_bf16(logits[0]);
        std::uint32_t brute_index = 0;
        for (std::uint32_t i = 1; i < kCount; ++i) {
            const float value = f32_from_bf16(logits[i]);
            if (value > brute_best) {
                brute_best = value;
                brute_index = i;
            }
        }
        std::vector<float> group_values(kGroups, 0.0f);
        std::vector<std::uint32_t> group_indices(kGroups, 0u);
        for (std::uint32_t group = 0; group < kGroups; ++group) {
            logits_argmax_stage1_reference(logits, kCount, group, group_values[group],
                                           group_indices[group]);
        }
        if (logits_argmax_stage2_reference(group_values, group_indices) != brute_index) {
            return 90 + static_cast<int>(trial % 6);
        }
    }
    return 0;
}

int main() {
    if (const int result = test_bf16_conversion(); result != 0) {
        return result;
    }
    if (const int result = test_flush_to_zero(); result != 0) {
        return result;
    }
    if (const int result = test_embed_hand_vector(); result != 0) {
        return result;
    }
    if (const int result = test_rms_closed_form(); result != 0) {
        return result;
    }
    if (const int result = test_discriminators_exist(); result != 0) {
        return result;
    }
    if (const int result = test_tree_shapes_distinct(); result != 0) {
        return result;
    }
    if (const int result = test_gdn_oracle(); result != 0) {
        return result;
    }
    if (const int result = test_attention_oracle(); result != 0) {
        return result;
    }
    if (const int result = test_moe_oracle(); result != 0) {
        return result;
    }
    if (const int result = test_head_oracle(); result != 0) {
        return result;
    }
    return test_generator_determinism();
}
