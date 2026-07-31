#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace tatara::testing {

enum class RawAffineQuantizedFormat : std::uint8_t {
    Q4,
    Q8,
};

enum class RawQuantizedInterpretation : std::uint8_t {
    Unsigned,
    Signed,
};

enum class RawAffineIdentityStatus : std::uint8_t {
    Ok,
    UnsupportedFormat,
    UnsupportedInterpretation,
    ZeroDimension,
    InvalidGroupSize,
    ArithmeticOverflow,
    InvalidActivationStride,
    InsufficientActivationData,
    InvalidPackedWeightStride,
    InsufficientPackedWeightData,
    InvalidParameterStride,
    InsufficientScaleData,
    InsufficientBiasData,
    InvalidOutputStride,
    InsufficientDirectOutputData,
    InsufficientCorrectedOutputData,
    OverlappingOutputs,
    OutputAliasesInput,
    InvalidErrorBounds,
    NonFiniteActivation,
    NonFiniteScale,
    NonFiniteBias,
    NumericalOverflow,
};

struct RawAffineIdentityProblem {
    RawAffineQuantizedFormat format{RawAffineQuantizedFormat::Q4};
    RawQuantizedInterpretation interpretation{
        RawQuantizedInterpretation::Unsigned};
    std::span<const std::uint16_t> activations{};
    std::span<const std::uint8_t> packed_weights{};
    std::span<const std::uint16_t> scales{};
    std::span<const std::uint16_t> biases{};
    std::size_t input_rows{0};
    std::size_t output_columns{0};
    std::size_t reduction_columns{0};
    std::size_t group_size{0};
    std::size_t activation_row_stride{0};
    std::size_t packed_weight_row_stride{0};
    std::size_t parameter_row_stride{0};
};

struct RawAffineIdentityOutput {
    std::span<float> direct_affine{};
    std::span<float> grouped_correction{};
    std::size_t row_stride{0};
};

struct RawAffineErrorBounds {
    double absolute{0.0};
    double relative{0.0};
    std::uint32_t ulp{0};
};

struct RawAffineErrorLocation {
    std::size_t input_row{0};
    std::size_t output_column{0};
};

struct RawAffineIdentityReport {
    std::size_t comparison_count{0};
    std::size_t bit_exact_count{0};
    std::size_t within_bounds_count{0};
    double maximum_absolute_error{0.0};
    double maximum_relative_error{0.0};
    std::uint32_t maximum_ulp_distance{0};
    RawAffineErrorLocation maximum_absolute_error_at{};
    RawAffineErrorLocation maximum_relative_error_at{};
    RawAffineErrorLocation maximum_ulp_distance_at{};

    [[nodiscard]] bool all_within_bounds() const noexcept {
        return comparison_count != 0 &&
               within_bounds_count == comparison_count;
    }

    [[nodiscard]] bool all_bit_exact() const noexcept {
        return comparison_count != 0 &&
               bit_exact_count == comparison_count;
    }
};

struct RawAffineIdentityResult {
    RawAffineIdentityStatus status{RawAffineIdentityStatus::ZeroDimension};
    RawAffineIdentityReport report{};

    [[nodiscard]] explicit operator bool() const noexcept {
        return status == RawAffineIdentityStatus::Ok;
    }
};

// The current Tatara affine contract interprets Q4 nibbles and Q8 bytes as
// unsigned. Signed requests fail before any output write.
//
// direct_affine uses increasing K:
//   dequantized = fma(q, scale[group], bias[group])
//   direct = fma(x, dequantized, direct)
//
// grouped_correction uses increasing groups and increasing K within each group:
//   raw_dot = fma(x, q, raw_dot)
//   row_sum = fma(x, 1, row_sum)
//   contribution = fma(raw_dot, scale, fma(row_sum, bias, 0))
//   grouped = fma(1, contribution, grouped)
//
// This is a numerical-family association instrument, not a bit-identity gate.
// Every bfloat operand and every arithmetic intermediate must be finite.
// Outputs must be mutually disjoint and disjoint from all input storage.
// Validation and a complete arithmetic preflight happen before output writes.
// A cell is within bounds when its absolute or symmetric-relative error is
// accepted and its ULP distance is accepted.
RawAffineIdentityResult compare_raw_affine_associations(
    const RawAffineIdentityProblem& problem,
    RawAffineIdentityOutput output,
    RawAffineErrorBounds bounds) noexcept;

float raw_affine_float_from_bfloat16(std::uint16_t bits) noexcept;

} // namespace tatara::testing
