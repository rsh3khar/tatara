#include "raw_quantized_affine_identity.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>

namespace tatara::testing {
namespace {

using Status = RawAffineIdentityStatus;

struct CellValues {
    float direct{0.0F};
    float grouped{0.0F};
};

struct ByteRange {
    std::uintptr_t begin{0};
    std::uintptr_t end{0};
};

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t& result) noexcept {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        return false;
    }
    result = left + right;
    return true;
}

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t& result) noexcept {
    if (left != 0 &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checked_extent(std::size_t rows, std::size_t stride,
                    std::size_t row_width, std::size_t& extent) noexcept {
    if (rows == 0) {
        extent = 0;
        return true;
    }
    std::size_t preceding_rows = 0;
    return checked_multiply(rows - 1, stride, preceding_rows) &&
           checked_add(preceding_rows, row_width, extent);
}

std::size_t ceiling_divide(std::size_t value,
                           std::size_t divisor) noexcept {
    return value / divisor +
           static_cast<std::size_t>(value % divisor != 0);
}

Status validate_extents(const RawAffineIdentityProblem& problem,
                        RawAffineIdentityOutput output) noexcept {
    if (problem.input_rows == 0 || problem.output_columns == 0 ||
        problem.reduction_columns == 0) {
        return Status::ZeroDimension;
    }
    if (problem.group_size == 0 ||
        problem.group_size > problem.reduction_columns) {
        return Status::InvalidGroupSize;
    }
    if (problem.interpretation != RawQuantizedInterpretation::Unsigned) {
        return Status::UnsupportedInterpretation;
    }

    std::size_t packed_width = 0;
    switch (problem.format) {
    case RawAffineQuantizedFormat::Q4:
        packed_width = ceiling_divide(problem.reduction_columns, 2);
        break;
    case RawAffineQuantizedFormat::Q8:
        packed_width = problem.reduction_columns;
        break;
    default:
        return Status::UnsupportedFormat;
    }

    const std::size_t parameter_width =
        ceiling_divide(problem.reduction_columns, problem.group_size);
    if (problem.activation_row_stride < problem.reduction_columns) {
        return Status::InvalidActivationStride;
    }
    if (problem.packed_weight_row_stride < packed_width) {
        return Status::InvalidPackedWeightStride;
    }
    if (problem.parameter_row_stride < parameter_width) {
        return Status::InvalidParameterStride;
    }
    if (output.row_stride < problem.output_columns) {
        return Status::InvalidOutputStride;
    }

    std::size_t required = 0;
    if (!checked_extent(problem.input_rows,
                        problem.activation_row_stride,
                        problem.reduction_columns, required)) {
        return Status::ArithmeticOverflow;
    }
    if (problem.activations.size() < required) {
        return Status::InsufficientActivationData;
    }
    if (!checked_extent(problem.output_columns,
                        problem.packed_weight_row_stride, packed_width,
                        required)) {
        return Status::ArithmeticOverflow;
    }
    if (problem.packed_weights.size() < required) {
        return Status::InsufficientPackedWeightData;
    }
    if (!checked_extent(problem.output_columns,
                        problem.parameter_row_stride, parameter_width,
                        required)) {
        return Status::ArithmeticOverflow;
    }
    if (problem.scales.size() < required) {
        return Status::InsufficientScaleData;
    }
    if (problem.biases.size() < required) {
        return Status::InsufficientBiasData;
    }
    if (!checked_extent(problem.input_rows, output.row_stride,
                        problem.output_columns, required)) {
        return Status::ArithmeticOverflow;
    }
    if (output.direct_affine.size() < required) {
        return Status::InsufficientDirectOutputData;
    }
    return output.grouped_correction.size() < required
               ? Status::InsufficientCorrectedOutputData
               : Status::Ok;
}

bool make_byte_range(const void* pointer, std::size_t elements,
                     std::size_t element_bytes,
                     ByteRange& range) noexcept {
    std::size_t bytes = 0;
    if (!checked_multiply(elements, element_bytes, bytes)) {
        return false;
    }
    const std::uintptr_t begin =
        reinterpret_cast<std::uintptr_t>(pointer);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return false;
    }
    range = ByteRange{begin, begin + bytes};
    return true;
}

bool ranges_overlap(ByteRange left, ByteRange right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

Status validate_aliasing(const RawAffineIdentityProblem& problem,
                         RawAffineIdentityOutput output) noexcept {
    const std::size_t packed_width =
        problem.format == RawAffineQuantizedFormat::Q4
            ? ceiling_divide(problem.reduction_columns, 2)
            : problem.reduction_columns;
    const std::size_t parameter_width =
        ceiling_divide(problem.reduction_columns, problem.group_size);
    std::size_t activation_extent = 0;
    std::size_t packed_extent = 0;
    std::size_t parameter_extent = 0;
    std::size_t output_extent = 0;
    if (!checked_extent(problem.input_rows,
                        problem.activation_row_stride,
                        problem.reduction_columns, activation_extent) ||
        !checked_extent(problem.output_columns,
                        problem.packed_weight_row_stride, packed_width,
                        packed_extent) ||
        !checked_extent(problem.output_columns,
                        problem.parameter_row_stride, parameter_width,
                        parameter_extent) ||
        !checked_extent(problem.input_rows, output.row_stride,
                        problem.output_columns, output_extent)) {
        return Status::ArithmeticOverflow;
    }

    ByteRange direct{};
    ByteRange grouped{};
    ByteRange activations{};
    ByteRange packed{};
    ByteRange scales{};
    ByteRange biases{};
    if (!make_byte_range(output.direct_affine.data(), output_extent,
                         sizeof(float), direct) ||
        !make_byte_range(output.grouped_correction.data(), output_extent,
                         sizeof(float), grouped) ||
        !make_byte_range(problem.activations.data(), activation_extent,
                         sizeof(std::uint16_t), activations) ||
        !make_byte_range(problem.packed_weights.data(), packed_extent,
                         sizeof(std::uint8_t), packed) ||
        !make_byte_range(problem.scales.data(), parameter_extent,
                         sizeof(std::uint16_t), scales) ||
        !make_byte_range(problem.biases.data(), parameter_extent,
                         sizeof(std::uint16_t), biases)) {
        return Status::ArithmeticOverflow;
    }
    if (ranges_overlap(direct, grouped)) {
        return Status::OverlappingOutputs;
    }
    const std::array<ByteRange, 4> inputs = {
        activations, packed, scales, biases};
    for (const ByteRange input : inputs) {
        if (ranges_overlap(direct, input) ||
            ranges_overlap(grouped, input)) {
            return Status::OutputAliasesInput;
        }
    }
    return Status::Ok;
}

Status validate_finite_operands(
    const RawAffineIdentityProblem& problem) noexcept {
    for (std::size_t row = 0; row < problem.input_rows; ++row) {
        const std::size_t row_begin =
            row * problem.activation_row_stride;
        for (std::size_t column = 0;
             column < problem.reduction_columns; ++column) {
            if (!std::isfinite(raw_affine_float_from_bfloat16(
                    problem.activations[row_begin + column]))) {
                return Status::NonFiniteActivation;
            }
        }
    }

    const std::size_t group_count =
        ceiling_divide(problem.reduction_columns, problem.group_size);
    for (std::size_t row = 0; row < problem.output_columns; ++row) {
        const std::size_t row_begin =
            row * problem.parameter_row_stride;
        for (std::size_t group = 0; group < group_count; ++group) {
            if (!std::isfinite(raw_affine_float_from_bfloat16(
                    problem.scales[row_begin + group]))) {
                return Status::NonFiniteScale;
            }
            if (!std::isfinite(raw_affine_float_from_bfloat16(
                    problem.biases[row_begin + group]))) {
                return Status::NonFiniteBias;
            }
        }
    }
    return Status::Ok;
}

std::uint8_t quantized_value(
    const RawAffineIdentityProblem& problem, std::size_t output_column,
    std::size_t reduction_column) noexcept {
    const std::size_t row_begin =
        output_column * problem.packed_weight_row_stride;
    if (problem.format == RawAffineQuantizedFormat::Q8) {
        return problem.packed_weights[row_begin + reduction_column];
    }
    const std::uint8_t packed =
        problem.packed_weights[row_begin + reduction_column / 2];
    const unsigned shift =
        static_cast<unsigned>((reduction_column % 2) * 4);
    return static_cast<std::uint8_t>((packed >> shift) & 0x0FU);
}

Status compute_cell(const RawAffineIdentityProblem& problem,
                    std::size_t input_row, std::size_t output_column,
                    CellValues& values) noexcept {
    const std::size_t activation_begin =
        input_row * problem.activation_row_stride;
    const std::size_t parameter_begin =
        output_column * problem.parameter_row_stride;

    float direct = 0.0F;
    for (std::size_t reduction_column = 0;
         reduction_column < problem.reduction_columns;
         ++reduction_column) {
        const std::size_t group =
            reduction_column / problem.group_size;
        const float activation = raw_affine_float_from_bfloat16(
            problem.activations[activation_begin + reduction_column]);
        const float scale = raw_affine_float_from_bfloat16(
            problem.scales[parameter_begin + group]);
        const float bias = raw_affine_float_from_bfloat16(
            problem.biases[parameter_begin + group]);
        const float quantized = static_cast<float>(quantized_value(
            problem, output_column, reduction_column));
        const float dequantized = std::fma(quantized, scale, bias);
        if (!std::isfinite(dequantized)) {
            return Status::NumericalOverflow;
        }
        direct = std::fma(activation, dequantized, direct);
        if (!std::isfinite(direct)) {
            return Status::NumericalOverflow;
        }
    }

    float grouped = 0.0F;
    std::size_t group_begin = 0;
    std::size_t group = 0;
    while (group_begin < problem.reduction_columns) {
        const std::size_t remaining =
            problem.reduction_columns - group_begin;
        const std::size_t group_width =
            std::min(problem.group_size, remaining);
        const std::size_t group_end = group_begin + group_width;
        float raw_dot = 0.0F;
        float row_sum = 0.0F;
        for (std::size_t reduction_column = group_begin;
             reduction_column < group_end; ++reduction_column) {
            const float activation = raw_affine_float_from_bfloat16(
                problem.activations[activation_begin + reduction_column]);
            const float quantized = static_cast<float>(quantized_value(
                problem, output_column, reduction_column));
            raw_dot = std::fma(activation, quantized, raw_dot);
            row_sum = std::fma(activation, 1.0F, row_sum);
            if (!std::isfinite(raw_dot) || !std::isfinite(row_sum)) {
                return Status::NumericalOverflow;
            }
        }
        const float scale = raw_affine_float_from_bfloat16(
            problem.scales[parameter_begin + group]);
        const float bias = raw_affine_float_from_bfloat16(
            problem.biases[parameter_begin + group]);
        const float scaled_bias = std::fma(row_sum, bias, 0.0F);
        const float contribution =
            std::fma(raw_dot, scale, scaled_bias);
        grouped = std::fma(1.0F, contribution, grouped);
        if (!std::isfinite(scaled_bias) ||
            !std::isfinite(contribution) || !std::isfinite(grouped)) {
            return Status::NumericalOverflow;
        }
        group_begin = group_end;
        ++group;
    }
    values = CellValues{direct, grouped};
    return Status::Ok;
}

std::uint32_t ordered_float_bits(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x80000000U) != 0) {
        return 0x80000000U - (bits & 0x7FFFFFFFU);
    }
    return 0x80000000U + bits;
}

std::uint32_t ulp_distance(float left, float right) noexcept {
    const std::uint32_t ordered_left = ordered_float_bits(left);
    const std::uint32_t ordered_right = ordered_float_bits(right);
    return ordered_left >= ordered_right
               ? ordered_left - ordered_right
               : ordered_right - ordered_left;
}

void update_report(float direct, float grouped, std::size_t input_row,
                   std::size_t output_column,
                   RawAffineErrorBounds bounds,
                   RawAffineIdentityReport& report) noexcept {
    const double direct_double = static_cast<double>(direct);
    const double grouped_double = static_cast<double>(grouped);
    const double absolute =
        std::abs(direct_double - grouped_double);
    const double denominator =
        std::max({std::abs(direct_double), std::abs(grouped_double),
                  static_cast<double>(
                      std::numeric_limits<float>::min())});
    const double relative = absolute / denominator;
    const std::uint32_t ulp = ulp_distance(direct, grouped);
    const bool bit_exact =
        std::bit_cast<std::uint32_t>(direct) ==
        std::bit_cast<std::uint32_t>(grouped);
    const bool magnitude_within =
        absolute <= bounds.absolute || relative <= bounds.relative;
    const bool within = magnitude_within && ulp <= bounds.ulp;

    ++report.comparison_count;
    report.bit_exact_count += static_cast<std::size_t>(bit_exact);
    report.within_bounds_count += static_cast<std::size_t>(within);
    if (absolute > report.maximum_absolute_error) {
        report.maximum_absolute_error = absolute;
        report.maximum_absolute_error_at =
            RawAffineErrorLocation{input_row, output_column};
    }
    if (relative > report.maximum_relative_error) {
        report.maximum_relative_error = relative;
        report.maximum_relative_error_at =
            RawAffineErrorLocation{input_row, output_column};
    }
    if (ulp > report.maximum_ulp_distance) {
        report.maximum_ulp_distance = ulp;
        report.maximum_ulp_distance_at =
            RawAffineErrorLocation{input_row, output_column};
    }
}

} // namespace

float raw_affine_float_from_bfloat16(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

RawAffineIdentityResult compare_raw_affine_associations(
    const RawAffineIdentityProblem& problem,
    RawAffineIdentityOutput output,
    RawAffineErrorBounds bounds) noexcept {
    RawAffineIdentityResult result{};
    result.status = validate_extents(problem, output);
    if (result.status != Status::Ok) {
        return result;
    }
    result.status = validate_aliasing(problem, output);
    if (result.status != Status::Ok) {
        return result;
    }
    if (!std::isfinite(bounds.absolute) ||
        !std::isfinite(bounds.relative) || bounds.absolute < 0.0 ||
        bounds.relative < 0.0) {
        result.status = Status::InvalidErrorBounds;
        return result;
    }
    result.status = validate_finite_operands(problem);
    if (result.status != Status::Ok) {
        return result;
    }

    CellValues values{};
    for (std::size_t input_row = 0; input_row < problem.input_rows;
         ++input_row) {
        for (std::size_t output_column = 0;
             output_column < problem.output_columns; ++output_column) {
            result.status = compute_cell(problem, input_row,
                                         output_column, values);
            if (result.status != Status::Ok) {
                result.report = {};
                return result;
            }
        }
    }

    result.status = Status::Ok;
    for (std::size_t input_row = 0; input_row < problem.input_rows;
         ++input_row) {
        const std::size_t output_begin = input_row * output.row_stride;
        for (std::size_t output_column = 0;
             output_column < problem.output_columns; ++output_column) {
            const Status status = compute_cell(
                problem, input_row, output_column, values);
            if (status != Status::Ok) {
                result.status = status;
                result.report = {};
                return result;
            }
            output.direct_affine[output_begin + output_column] =
                values.direct;
            output.grouped_correction[output_begin + output_column] =
                values.grouped;
            update_report(values.direct, values.grouped, input_row,
                          output_column, bounds, result.report);
        }
    }
    return result;
}

} // namespace tatara::testing
