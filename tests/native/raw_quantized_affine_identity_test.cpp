#include "raw_quantized_affine_identity.h"

#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <span>

namespace {

using tatara::testing::RawAffineErrorBounds;
using tatara::testing::RawAffineIdentityOutput;
using tatara::testing::RawAffineIdentityProblem;
using tatara::testing::RawAffineIdentityResult;
using tatara::testing::RawAffineIdentityStatus;
using tatara::testing::RawAffineQuantizedFormat;
using tatara::testing::RawQuantizedInterpretation;

std::atomic<std::size_t> allocation_count{0};
int failures = 0;

std::uint16_t bfloat16(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F800000U) == 0x7F800000U) {
        return static_cast<std::uint16_t>(bits >> 16U);
    }
    const std::uint32_t rounded =
        bits + 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(rounded >> 16U);
}

void check(bool condition, const char* message) noexcept {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

bool same_bits(float left, float right) noexcept {
    return std::bit_cast<std::uint32_t>(left) ==
           std::bit_cast<std::uint32_t>(right);
}

template <std::size_t Rows, std::size_t Columns,
          std::size_t PackedStride>
void pack_q4(
    const std::array<std::array<std::uint8_t, Columns>, Rows>& values,
    std::array<std::uint8_t, Rows * PackedStride>& packed) noexcept {
    packed.fill(0xDEU);
    for (std::size_t row = 0; row < Rows; ++row) {
        for (std::size_t column = 0; column < Columns; ++column) {
            const std::size_t index = row * PackedStride + column / 2;
            const unsigned shift =
                static_cast<unsigned>((column % 2) * 4);
            const std::uint8_t mask =
                static_cast<std::uint8_t>(0x0FU << shift);
            const std::uint8_t nibble = static_cast<std::uint8_t>(
                (values[row][column] & 0x0FU) << shift);
            packed[index] = static_cast<std::uint8_t>(
                (packed[index] & static_cast<std::uint8_t>(~mask)) |
                nibble);
        }
        if constexpr ((Columns % 2) != 0) {
            const std::size_t tail =
                row * PackedStride + Columns / 2;
            packed[tail] =
                static_cast<std::uint8_t>(packed[tail] | 0xF0U);
        }
    }
}

template <std::size_t Extent>
bool output_canaries_unchanged(
    const std::array<float, Extent>& values, float canary,
    std::size_t rows, std::size_t columns,
    std::size_t stride) noexcept {
    if (!same_bits(values.front(), canary) ||
        !same_bits(values.back(), canary)) {
        return false;
    }
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = columns; column < stride; ++column) {
            if (!same_bits(values[1 + row * stride + column],
                           canary)) {
                return false;
            }
        }
    }
    return true;
}

int test_q4_partial_groups_and_odd_tail(
    RawAffineIdentityResult& measured) noexcept {
    constexpr std::size_t kRows = 3;
    constexpr std::size_t kOutputs = 4;
    constexpr std::size_t kReduction = 7;
    constexpr std::size_t kGroupSize = 3;
    constexpr std::size_t kActivationStride = 9;
    constexpr std::size_t kPackedStride = 5;
    constexpr std::size_t kParameterStride = 5;
    constexpr std::size_t kOutputStride = 6;
    constexpr float kCanary = -12345.25F;

    std::array<std::uint16_t, kRows * kActivationStride>
        activations{};
    const std::array<std::array<float, kReduction>, kRows>
        activation_values = {{
            {{1.0F, -2.0F, 0.5F, 4.0F, -8.0F, 16.0F, 0.25F}},
            {{-3.0F, 5.0F, 7.0F, -11.0F, 13.0F, -17.0F, 19.0F}},
            {{0.03125F, -0.0625F, 0.125F, -0.25F, 0.5F, -1.0F,
              2.0F}},
        }};
    activations.fill(0x7FC1U);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t column = 0; column < kReduction; ++column) {
            activations[row * kActivationStride + column] =
                bfloat16(activation_values[row][column]);
        }
    }

    const std::array<std::array<std::uint8_t, kReduction>, kOutputs>
        quantized = {{
            {{0U, 1U, 2U, 3U, 4U, 5U, 15U}},
            {{15U, 14U, 13U, 12U, 11U, 10U, 9U}},
            {{8U, 0U, 7U, 1U, 6U, 2U, 5U}},
            {{4U, 15U, 3U, 14U, 2U, 13U, 1U}},
        }};
    std::array<std::uint8_t, kOutputs * kPackedStride>
        packed{};
    pack_q4<kOutputs, kReduction, kPackedStride>(quantized, packed);

    std::array<std::uint16_t, kOutputs * kParameterStride>
        scales{};
    std::array<std::uint16_t, kOutputs * kParameterStride>
        biases{};
    scales.fill(0x7FC1U);
    biases.fill(0x7FC1U);
    const std::array<std::array<float, 3>, kOutputs> scale_values = {{
        {{0.5F, -1.25F, 3.0F}},
        {{-0.75F, 0.125F, 2.0F}},
        {{1.5F, -0.5F, -4.0F}},
        {{0.0625F, 2.5F, -1.0F}},
    }};
    const std::array<std::array<float, 3>, kOutputs> bias_values = {{
        {{-1.0F, 0.75F, 2.0F}},
        {{3.0F, -2.0F, -0.5F}},
        {{-4.0F, 1.0F, 8.0F}},
        {{0.5F, -3.0F, 1.25F}},
    }};
    for (std::size_t output = 0; output < kOutputs; ++output) {
        for (std::size_t group = 0; group < 3; ++group) {
            scales[output * kParameterStride + group] =
                bfloat16(scale_values[output][group]);
            biases[output * kParameterStride + group] =
                bfloat16(bias_values[output][group]);
        }
    }

    std::array<float, 2 + kRows * kOutputStride> direct{};
    std::array<float, 2 + kRows * kOutputStride> grouped{};
    direct.fill(kCanary);
    grouped.fill(kCanary);
    const RawAffineIdentityProblem problem{
        .format = RawAffineQuantizedFormat::Q4,
        .interpretation = RawQuantizedInterpretation::Unsigned,
        .activations = activations,
        .packed_weights = packed,
        .scales = scales,
        .biases = biases,
        .input_rows = kRows,
        .output_columns = kOutputs,
        .reduction_columns = kReduction,
        .group_size = kGroupSize,
        .activation_row_stride = kActivationStride,
        .packed_weight_row_stride = kPackedStride,
        .parameter_row_stride = kParameterStride,
    };
    const RawAffineIdentityOutput output{
        .direct_affine =
            std::span<float>{direct}.subspan(1, kRows * kOutputStride),
        .grouped_correction =
            std::span<float>{grouped}.subspan(1,
                                             kRows * kOutputStride),
        .row_stride = kOutputStride,
    };
    measured = tatara::testing::compare_raw_affine_associations(
        problem, output, RawAffineErrorBounds{0.0005, 0.00001, 32});
    if (!measured) {
        return 1;
    }
    if (measured.report.comparison_count != kRows * kOutputs ||
        !measured.report.all_within_bounds() ||
        !measured.report.all_bit_exact() ||
        measured.report.maximum_absolute_error != 0.0 ||
        measured.report.maximum_ulp_distance != 0) {
        return 2;
    }
    if (!output_canaries_unchanged(direct, kCanary, kRows, kOutputs,
                                   kOutputStride) ||
        !output_canaries_unchanged(grouped, kCanary, kRows, kOutputs,
                                   kOutputStride)) {
        return 3;
    }
    if (direct[1] != -54.25F || grouped[1] != -54.25F) {
        return 4;
    }

    const auto original = measured;
    const auto original_direct = direct;
    const auto original_grouped = grouped;
    for (std::size_t row = 0; row < kOutputs; ++row) {
        packed[row * kPackedStride + kReduction / 2] ^=
            0xF0U;
    }
    direct.fill(kCanary);
    grouped.fill(kCanary);
    measured = tatara::testing::compare_raw_affine_associations(
        problem, output, RawAffineErrorBounds{0.0005, 0.00001, 32});
    if (!measured ||
        measured.report.maximum_ulp_distance !=
            original.report.maximum_ulp_distance) {
        return 5;
    }
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t column = 0; column < kOutputs; ++column) {
            const std::size_t index =
                1 + row * kOutputStride + column;
            if (!same_bits(direct[index], original_direct[index]) ||
                !same_bits(grouped[index],
                           original_grouped[index])) {
                return 6;
            }
        }
    }
    return 0;
}

int test_q4_group64_regrouping_error(
    RawAffineIdentityResult& measured) noexcept {
    constexpr std::size_t kRows = 2;
    constexpr std::size_t kOutputs = 3;
    constexpr std::size_t kReduction = 129;
    constexpr std::size_t kGroupSize = 64;
    constexpr std::size_t kActivationStride = 131;
    constexpr std::size_t kPackedStride = 67;
    constexpr std::size_t kParameterStride = 4;
    constexpr std::size_t kOutputStride = 4;
    constexpr float kCanary = -6543.25F;

    std::array<std::uint16_t, kRows * kActivationStride>
        activations{};
    activations.fill(0x7FC1U);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t column = 0; column < kReduction; ++column) {
            const std::size_t magnitude_index =
                (column * 43U + row * 61U) % 193U + 1U;
            float magnitude =
                static_cast<float>(magnitude_index) / 10.0F;
            if (column % 17U == 0) {
                magnitude *= 31.0F;
            }
            const float sign =
                (column + row * 5U) % 3U == 0 ? -1.0F : 1.0F;
            activations[row * kActivationStride + column] =
                bfloat16(sign * magnitude);
        }
    }

    std::array<std::array<std::uint8_t, kReduction>, kOutputs>
        quantized{};
    for (std::size_t output = 0; output < kOutputs; ++output) {
        for (std::size_t column = 0; column < kReduction; ++column) {
            quantized[output][column] = static_cast<std::uint8_t>(
                (output * 11U + column * 7U + column / 5U) % 16U);
        }
    }
    std::array<std::uint8_t, kOutputs * kPackedStride>
        packed{};
    pack_q4<kOutputs, kReduction, kPackedStride>(quantized, packed);

    std::array<std::uint16_t, kOutputs * kParameterStride>
        scales{};
    std::array<std::uint16_t, kOutputs * kParameterStride>
        biases{};
    scales.fill(0x7FC1U);
    biases.fill(0x7FC1U);
    const std::array<float, 5> scale_values =
        {0.1F, -1.3F, 2.7F, -0.45F, 3.05F};
    const std::array<float, 5> bias_values =
        {-0.2F, 1.1F, -2.6F, 0.35F, 4.2F};
    for (std::size_t output = 0; output < kOutputs; ++output) {
        for (std::size_t group = 0; group < 3; ++group) {
            scales[output * kParameterStride + group] =
                bfloat16(scale_values[(output + group) % 5]);
            biases[output * kParameterStride + group] =
                bfloat16(bias_values[(output * 2U + group) % 5]);
        }
    }

    std::array<float, 2 + kRows * kOutputStride> direct{};
    std::array<float, 2 + kRows * kOutputStride> grouped{};
    direct.fill(kCanary);
    grouped.fill(kCanary);
    const RawAffineIdentityProblem problem{
        .format = RawAffineQuantizedFormat::Q4,
        .interpretation = RawQuantizedInterpretation::Unsigned,
        .activations = activations,
        .packed_weights = packed,
        .scales = scales,
        .biases = biases,
        .input_rows = kRows,
        .output_columns = kOutputs,
        .reduction_columns = kReduction,
        .group_size = kGroupSize,
        .activation_row_stride = kActivationStride,
        .packed_weight_row_stride = kPackedStride,
        .parameter_row_stride = kParameterStride,
    };
    const RawAffineIdentityOutput output{
        .direct_affine =
            std::span<float>{direct}.subspan(1, kRows * kOutputStride),
        .grouped_correction =
            std::span<float>{grouped}.subspan(1,
                                             kRows * kOutputStride),
        .row_stride = kOutputStride,
    };
    measured = tatara::testing::compare_raw_affine_associations(
        problem, output, RawAffineErrorBounds{0.004, 0.00001, 64});
    if (!measured || !measured.report.all_within_bounds()) {
        return 1;
    }
    if (measured.report.comparison_count != kRows * kOutputs ||
        measured.report.all_bit_exact() ||
        measured.report.maximum_ulp_distance == 0) {
        return 2;
    }
    if (measured.report.comparison_count != 6 ||
        measured.report.bit_exact_count != 0 ||
        measured.report.maximum_absolute_error != 0.001953125 ||
        measured.report.maximum_ulp_distance != 26 ||
        measured.report.maximum_absolute_error_at.input_row != 0 ||
        measured.report.maximum_absolute_error_at.output_column != 1 ||
        measured.report.maximum_ulp_distance_at.input_row != 0 ||
        measured.report.maximum_ulp_distance_at.output_column != 0) {
        return 3;
    }
    if (!output_canaries_unchanged(direct, kCanary, kRows, kOutputs,
                                   kOutputStride) ||
        !output_canaries_unchanged(grouped, kCanary, kRows, kOutputs,
                                   kOutputStride)) {
        return 4;
    }
    const RawAffineIdentityResult exact_bound_result =
        tatara::testing::compare_raw_affine_associations(
            problem, output, RawAffineErrorBounds{0.0, 0.0, 0});
    if (!exact_bound_result ||
        exact_bound_result.report.all_within_bounds() ||
        exact_bound_result.report.within_bounds_count != 0) {
        return 5;
    }
    return 0;
}

int test_q8_unsigned_and_regrouping_error(
    RawAffineIdentityResult& measured) noexcept {
    constexpr std::size_t kRows = 2;
    constexpr std::size_t kOutputs = 3;
    constexpr std::size_t kReduction = 65;
    constexpr std::size_t kGroupSize = 16;
    constexpr std::size_t kActivationStride = 67;
    constexpr std::size_t kPackedStride = 68;
    constexpr std::size_t kParameterStride = 6;
    constexpr std::size_t kOutputStride = 4;
    constexpr float kCanary = 9876.5F;

    std::array<std::uint16_t, kRows * kActivationStride>
        activations{};
    activations.fill(0x7FC1U);
    for (std::size_t row = 0; row < kRows; ++row) {
        for (std::size_t column = 0; column < kReduction; ++column) {
            const std::size_t magnitude_index =
                (column * 37U + row * 53U) % 251U + 1U;
            const float magnitude =
                static_cast<float>(magnitude_index) *
                (column % 7U == 0 ? 8.0F : 0.03125F);
            const float sign =
                (column + row * 3U) % 2U == 0 ? 1.0F : -1.0F;
            activations[row * kActivationStride + column] =
                bfloat16(sign * magnitude);
        }
    }

    std::array<std::uint8_t, kOutputs * kPackedStride> packed{};
    packed.fill(0xEEU);
    for (std::size_t output = 0; output < kOutputs; ++output) {
        for (std::size_t column = 0; column < kReduction; ++column) {
            const std::size_t quantized =
                (output * 83U + column * 29U) % 256U;
            packed[output * kPackedStride + column] =
                static_cast<std::uint8_t>(quantized);
        }
    }
    packed[0] = 0U;
    packed[1] = 127U;
    packed[2] = 128U;
    packed[3] = 255U;

    std::array<std::uint16_t, kOutputs * kParameterStride>
        scales{};
    std::array<std::uint16_t, kOutputs * kParameterStride>
        biases{};
    scales.fill(0x7FC1U);
    biases.fill(0x7FC1U);
    const std::array<float, 5> scale_values =
        {0.03125F, -1.5F, 3.25F, -0.0078125F, 5.5F};
    const std::array<float, 5> bias_values =
        {-17.0F, 0.125F, 33.0F, -2.75F, 0.015625F};
    for (std::size_t output = 0; output < kOutputs; ++output) {
        for (std::size_t group = 0; group < 5; ++group) {
            scales[output * kParameterStride + group] =
                bfloat16(scale_values[(output + group) % 5]);
            biases[output * kParameterStride + group] =
                bfloat16(bias_values[(output * 2 + group) % 5]);
        }
    }

    std::array<float, 2 + kRows * kOutputStride> direct{};
    std::array<float, 2 + kRows * kOutputStride> grouped{};
    direct.fill(kCanary);
    grouped.fill(kCanary);
    RawAffineIdentityProblem problem{
        .format = RawAffineQuantizedFormat::Q8,
        .interpretation = RawQuantizedInterpretation::Unsigned,
        .activations = activations,
        .packed_weights = packed,
        .scales = scales,
        .biases = biases,
        .input_rows = kRows,
        .output_columns = kOutputs,
        .reduction_columns = kReduction,
        .group_size = kGroupSize,
        .activation_row_stride = kActivationStride,
        .packed_weight_row_stride = kPackedStride,
        .parameter_row_stride = kParameterStride,
    };
    const RawAffineIdentityOutput output{
        .direct_affine =
            std::span<float>{direct}.subspan(1, kRows * kOutputStride),
        .grouped_correction =
            std::span<float>{grouped}.subspan(1,
                                             kRows * kOutputStride),
        .row_stride = kOutputStride,
    };
    measured = tatara::testing::compare_raw_affine_associations(
        problem, output, RawAffineErrorBounds{0.25, 0.00005, 512});
    if (!measured) {
        return 1;
    }
    if (!measured.report.all_within_bounds()) {
        std::fprintf(
            stderr,
            "Q8 bounds: max_abs=%.9g max_rel=%.9g max_ulp=%u "
            "within=%zu/%zu\n",
            measured.report.maximum_absolute_error,
            measured.report.maximum_relative_error,
            measured.report.maximum_ulp_distance,
            measured.report.within_bounds_count,
            measured.report.comparison_count);
        return 1;
    }
    if (measured.report.maximum_ulp_distance == 0 ||
        measured.report.all_bit_exact()) {
        return 2;
    }
    if (measured.report.comparison_count != 6 ||
        measured.report.bit_exact_count != 0 ||
        measured.report.maximum_absolute_error != 0.125 ||
        measured.report.maximum_ulp_distance != 304 ||
        measured.report.maximum_absolute_error_at.input_row != 1 ||
        measured.report.maximum_absolute_error_at.output_column != 1 ||
        measured.report.maximum_ulp_distance_at.input_row != 0 ||
        measured.report.maximum_ulp_distance_at.output_column != 0) {
        return 3;
    }
    if (!output_canaries_unchanged(direct, kCanary, kRows, kOutputs,
                                   kOutputStride) ||
        !output_canaries_unchanged(grouped, kCanary, kRows, kOutputs,
                                   kOutputStride)) {
        return 4;
    }
    const RawAffineIdentityResult exact_bound_result =
        tatara::testing::compare_raw_affine_associations(
            problem, output, RawAffineErrorBounds{0.0, 0.0, 0});
    if (!exact_bound_result ||
        exact_bound_result.report.all_within_bounds() ||
        exact_bound_result.report.within_bounds_count != 0) {
        return 5;
    }

    problem.interpretation = RawQuantizedInterpretation::Signed;
    direct.fill(kCanary);
    grouped.fill(kCanary);
    const RawAffineIdentityResult signed_result =
        tatara::testing::compare_raw_affine_associations(
            problem, output, RawAffineErrorBounds{0.25, 0.00005, 512});
    if (signed_result.status !=
            RawAffineIdentityStatus::UnsupportedInterpretation ||
        signed_result.report.comparison_count != 0) {
        return 6;
    }
    for (float value : direct) {
        if (!same_bits(value, kCanary)) {
            return 7;
        }
    }
    for (float value : grouped) {
        if (!same_bits(value, kCanary)) {
            return 8;
        }
    }
    return 0;
}

RawAffineIdentityProblem make_single_cell_problem(
    std::span<const std::uint16_t> activations,
    std::span<const std::uint8_t> weights,
    std::span<const std::uint16_t> scales,
    std::span<const std::uint16_t> biases) noexcept {
    return RawAffineIdentityProblem{
        .format = RawAffineQuantizedFormat::Q8,
        .interpretation = RawQuantizedInterpretation::Unsigned,
        .activations = activations,
        .packed_weights = weights,
        .scales = scales,
        .biases = biases,
        .input_rows = 1,
        .output_columns = 1,
        .reduction_columns = 2,
        .group_size = 2,
        .activation_row_stride = 2,
        .packed_weight_row_stride = 2,
        .parameter_row_stride = 1,
    };
}

int test_nonfinite_and_overflow_rejection() noexcept {
    constexpr float kCanary = -777.0F;
    alignas(float) std::array<std::uint8_t, 4> weights =
        {255U, 1U, 0xEEU, 0xEEU};
    alignas(float) std::array<std::uint16_t, 2> activations =
        {bfloat16(1.0F), bfloat16(2.0F)};
    alignas(float) std::array<std::uint16_t, 2> scales =
        {bfloat16(1.0F), 0x7FC1U};
    alignas(float) std::array<std::uint16_t, 2> biases =
        {bfloat16(0.0F), 0x7FC1U};
    std::array<float, 3> direct = {kCanary, kCanary, kCanary};
    std::array<float, 3> grouped = {kCanary, kCanary, kCanary};
    const RawAffineIdentityOutput output{
        .direct_affine = std::span<float>{direct}.subspan(1, 1),
        .grouped_correction =
            std::span<float>{grouped}.subspan(1, 1),
        .row_stride = 1,
    };
    RawAffineIdentityProblem problem = make_single_cell_problem(
        activations, weights, scales, biases);
    const RawAffineIdentityResult unsigned_result =
        tatara::testing::compare_raw_affine_associations(
            problem, output, RawAffineErrorBounds{0.0, 0.0, 0});
    if (!unsigned_result || direct[1] != 257.0F ||
        grouped[1] != 257.0F ||
        !unsigned_result.report.all_bit_exact()) {
        return 1;
    }

    auto check_rejection_with_output =
        [&](RawAffineIdentityStatus expected,
            RawAffineIdentityProblem problem,
            RawAffineIdentityOutput candidate_output,
            RawAffineErrorBounds bounds) noexcept {
            direct.fill(kCanary);
            grouped.fill(kCanary);
            const RawAffineIdentityResult result =
                tatara::testing::compare_raw_affine_associations(
                    problem, candidate_output, bounds);
            if (result.status != expected ||
                result.report.comparison_count != 0) {
                return false;
            }
            for (float value : direct) {
                if (!same_bits(value, kCanary)) {
                    return false;
                }
            }
            for (float value : grouped) {
                if (!same_bits(value, kCanary)) {
                    return false;
                }
            }
            return true;
        };
    auto check_rejection =
        [&](RawAffineIdentityStatus expected,
            RawAffineIdentityProblem problem,
            RawAffineErrorBounds bounds) noexcept {
            return check_rejection_with_output(
                expected, problem, output, bounds);
        };

    activations[0] = 0x7FC1U;
    if (!check_rejection(RawAffineIdentityStatus::NonFiniteActivation,
                         problem, RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 2;
    }
    activations[0] = bfloat16(1.0F);
    scales[0] = 0x7F80U;
    if (!check_rejection(RawAffineIdentityStatus::NonFiniteScale,
                         problem, RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 3;
    }
    scales[0] = bfloat16(1.0F);
    biases[0] = 0xFFC1U;
    if (!check_rejection(RawAffineIdentityStatus::NonFiniteBias,
                         problem, RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 4;
    }
    biases[0] = bfloat16(0.0F);
    activations[0] = 0x7F7FU;
    if (!check_rejection(RawAffineIdentityStatus::NumericalOverflow,
                         problem, RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 5;
    }
    activations[0] = bfloat16(1.0F);
    const double infinity = std::numeric_limits<double>::infinity();
    if (!check_rejection(RawAffineIdentityStatus::InvalidErrorBounds,
                         problem,
                         RawAffineErrorBounds{infinity, 1.0, 100})) {
        return 6;
    }
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (!check_rejection(RawAffineIdentityStatus::InvalidErrorBounds,
                         problem, RawAffineErrorBounds{1.0, nan, 100})) {
        return 7;
    }

    auto invalid_format = problem;
    invalid_format.format = static_cast<RawAffineQuantizedFormat>(99);
    if (!check_rejection(RawAffineIdentityStatus::UnsupportedFormat,
                         invalid_format,
                         RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 8;
    }
    auto invalid_stride = problem;
    invalid_stride.activation_row_stride = 1;
    if (!check_rejection(RawAffineIdentityStatus::InvalidActivationStride,
                         invalid_stride,
                         RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 9;
    }
    auto arithmetic_overflow = problem;
    arithmetic_overflow.input_rows =
        std::numeric_limits<std::size_t>::max();
    arithmetic_overflow.activation_row_stride =
        std::numeric_limits<std::size_t>::max();
    if (!check_rejection(RawAffineIdentityStatus::ArithmeticOverflow,
                         arithmetic_overflow,
                         RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 10;
    }

    auto zero_dimension = problem;
    zero_dimension.input_rows = 0;
    if (!check_rejection(RawAffineIdentityStatus::ZeroDimension,
                         zero_dimension,
                         RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 11;
    }
    auto invalid_group_size = problem;
    invalid_group_size.group_size = 0;
    if (!check_rejection(RawAffineIdentityStatus::InvalidGroupSize,
                         invalid_group_size,
                         RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 12;
    }
    auto insufficient_activations = problem;
    insufficient_activations.activations =
        std::span<const std::uint16_t>{activations}.first(1);
    if (!check_rejection(
            RawAffineIdentityStatus::InsufficientActivationData,
            insufficient_activations,
            RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 13;
    }
    auto invalid_packed_stride = problem;
    invalid_packed_stride.packed_weight_row_stride = 1;
    if (!check_rejection(
            RawAffineIdentityStatus::InvalidPackedWeightStride,
            invalid_packed_stride,
            RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 14;
    }
    auto insufficient_packed_weights = problem;
    insufficient_packed_weights.packed_weights =
        std::span<const std::uint8_t>{weights}.first(1);
    if (!check_rejection(
            RawAffineIdentityStatus::InsufficientPackedWeightData,
            insufficient_packed_weights,
            RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 15;
    }
    auto invalid_parameter_stride = problem;
    invalid_parameter_stride.parameter_row_stride = 0;
    if (!check_rejection(
            RawAffineIdentityStatus::InvalidParameterStride,
            invalid_parameter_stride,
            RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 16;
    }
    auto insufficient_scales = problem;
    insufficient_scales.scales =
        std::span<const std::uint16_t>{scales}.first(0);
    if (!check_rejection(RawAffineIdentityStatus::InsufficientScaleData,
                         insufficient_scales,
                         RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 17;
    }
    auto insufficient_biases = problem;
    insufficient_biases.biases =
        std::span<const std::uint16_t>{biases}.first(0);
    if (!check_rejection(RawAffineIdentityStatus::InsufficientBiasData,
                         insufficient_biases,
                         RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 18;
    }
    auto invalid_output_stride = output;
    invalid_output_stride.row_stride = 0;
    if (!check_rejection_with_output(
            RawAffineIdentityStatus::InvalidOutputStride, problem,
            invalid_output_stride,
            RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 19;
    }
    auto insufficient_direct_output = output;
    insufficient_direct_output.direct_affine =
        std::span<float>{direct}.subspan(1, 0);
    if (!check_rejection_with_output(
            RawAffineIdentityStatus::InsufficientDirectOutputData,
            problem, insufficient_direct_output,
            RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 20;
    }
    auto insufficient_corrected_output = output;
    insufficient_corrected_output.grouped_correction =
        std::span<float>{grouped}.subspan(1, 0);
    if (!check_rejection_with_output(
            RawAffineIdentityStatus::InsufficientCorrectedOutputData,
            problem, insufficient_corrected_output,
            RawAffineErrorBounds{1.0, 1.0, 100})) {
        return 21;
    }

    direct.fill(kCanary);
    grouped.fill(kCanary);
    const RawAffineIdentityOutput overlapping_outputs{
        .direct_affine = std::span<float>{direct}.subspan(1, 1),
        .grouped_correction =
            std::span<float>{direct}.subspan(1, 1),
        .row_stride = 1,
    };
    const RawAffineIdentityResult overlapping_result =
        tatara::testing::compare_raw_affine_associations(
            problem, overlapping_outputs,
            RawAffineErrorBounds{1.0, 1.0, 100});
    if (overlapping_result.status !=
            RawAffineIdentityStatus::OverlappingOutputs ||
        !same_bits(direct[1], kCanary)) {
        return 22;
    }

    const auto activations_before = activations;
    const auto weights_before = weights;
    const auto scales_before = scales;
    const auto biases_before = biases;
    auto check_input_alias =
        [&](void* pointer) noexcept {
            const RawAffineIdentityOutput aliased{
                .direct_affine = std::span<float>{
                    static_cast<float*>(pointer), 1},
                .grouped_correction =
                    std::span<float>{grouped}.subspan(1, 1),
                .row_stride = 1,
            };
            const RawAffineIdentityResult result =
                tatara::testing::compare_raw_affine_associations(
                    problem, aliased,
                    RawAffineErrorBounds{1.0, 1.0, 100});
            return result.status ==
                       RawAffineIdentityStatus::OutputAliasesInput &&
                   result.report.comparison_count == 0;
        };
    if (!check_input_alias(activations.data()) ||
        !check_input_alias(weights.data()) ||
        !check_input_alias(scales.data()) ||
        !check_input_alias(biases.data()) ||
        activations != activations_before || weights != weights_before ||
        scales != scales_before || biases != biases_before) {
        return 23;
    }
    return 0;
}

} // namespace

void* operator new(std::size_t size) {
    allocation_count.fetch_add(1, std::memory_order_relaxed);
    if (void* pointer = std::malloc(size)) {
        return pointer;
    }
    std::abort();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* pointer) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
    std::free(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
    std::free(pointer);
}

int main() {
    RawAffineIdentityResult q4_result{};
    RawAffineIdentityResult q4_group64_result{};
    RawAffineIdentityResult q8_result{};
    const std::size_t allocations_before =
        allocation_count.load(std::memory_order_relaxed);

    if (const int result =
            test_q4_partial_groups_and_odd_tail(q4_result)) {
        std::fprintf(stderr, "Q4 fixture failed at %d\n", result);
        return 10 + result;
    }
    if (const int result = test_q4_group64_regrouping_error(
            q4_group64_result)) {
        std::fprintf(stderr,
                     "Q4 group-64 fixture failed at %d\n", result);
        return 30 + result;
    }
    if (const int result =
            test_q8_unsigned_and_regrouping_error(q8_result)) {
        std::fprintf(stderr, "Q8 fixture failed at %d\n", result);
        return 50 + result;
    }
    if (const int result = test_nonfinite_and_overflow_rejection()) {
        std::fprintf(stderr, "rejection fixture failed at %d\n", result);
        return 70 + result;
    }

    const std::size_t allocations_after =
        allocation_count.load(std::memory_order_relaxed);
    check(allocations_before == allocations_after,
          "oracle and fixtures allocate no heap memory");
    check(q4_group64_result.report.maximum_ulp_distance > 0 ||
              q8_result.report.maximum_ulp_distance > 0,
          "at least one regrouped result is measured as non-bit-exact");

    std::printf(
        "raw_quantized_affine_identity_test: PASS "
        "q4={count=%zu exact=%zu max_abs=%.9g@(%zu,%zu) "
        "max_rel=%.9g max_ulp=%u@(%zu,%zu)} "
        "q4g64k129={count=%zu exact=%zu max_abs=%.9g@(%zu,%zu) "
        "max_rel=%.9g max_ulp=%u@(%zu,%zu)} "
        "q8={count=%zu exact=%zu max_abs=%.9g@(%zu,%zu) "
        "max_rel=%.9g max_ulp=%u@(%zu,%zu)} allocations=%zu\n",
        q4_result.report.comparison_count,
        q4_result.report.bit_exact_count,
        q4_result.report.maximum_absolute_error,
        q4_result.report.maximum_absolute_error_at.input_row,
        q4_result.report.maximum_absolute_error_at.output_column,
        q4_result.report.maximum_relative_error,
        q4_result.report.maximum_ulp_distance,
        q4_result.report.maximum_ulp_distance_at.input_row,
        q4_result.report.maximum_ulp_distance_at.output_column,
        q4_group64_result.report.comparison_count,
        q4_group64_result.report.bit_exact_count,
        q4_group64_result.report.maximum_absolute_error,
        q4_group64_result.report.maximum_absolute_error_at.input_row,
        q4_group64_result.report.maximum_absolute_error_at.output_column,
        q4_group64_result.report.maximum_relative_error,
        q4_group64_result.report.maximum_ulp_distance,
        q4_group64_result.report.maximum_ulp_distance_at.input_row,
        q4_group64_result.report.maximum_ulp_distance_at.output_column,
        q8_result.report.comparison_count,
        q8_result.report.bit_exact_count,
        q8_result.report.maximum_absolute_error,
        q8_result.report.maximum_absolute_error_at.input_row,
        q8_result.report.maximum_absolute_error_at.output_column,
        q8_result.report.maximum_relative_error,
        q8_result.report.maximum_ulp_distance,
        q8_result.report.maximum_ulp_distance_at.input_row,
        q8_result.report.maximum_ulp_distance_at.output_column,
        allocations_after - allocations_before);
    return failures == 0 ? 0 : 1;
}
