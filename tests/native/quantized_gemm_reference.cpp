#include "quantized_gemm_reference.h"

#include <bit>
#include <cmath>
#include <limits>

namespace tatara::testing {
namespace {

using Status = QuantizedGemmReferenceStatus;

bool checked_add(std::size_t left, std::size_t right, std::size_t& result) noexcept {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        return false;
    }
    result = left + right;
    return true;
}

bool checked_multiply(std::size_t left, std::size_t right, std::size_t& result) noexcept {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        return false;
    }
    result = left * right;
    return true;
}

bool checked_extent(std::size_t rows, std::size_t stride, std::size_t row_width,
                    std::size_t& extent) noexcept {
    if (rows == 0) {
        extent = 0;
        return true;
    }
    std::size_t preceding_rows = 0;
    return checked_multiply(rows - 1, stride, preceding_rows) &&
           checked_add(preceding_rows, row_width, extent);
}

std::size_t ceiling_divide(std::size_t value, std::size_t divisor) noexcept {
    return value / divisor + static_cast<std::size_t>(value % divisor != 0);
}

Status validate_input(BfloatMatrixView input) noexcept {
    if (input.rows == 0 || input.columns == 0) {
        return Status::ZeroDimension;
    }
    if (input.row_stride < input.columns) {
        return Status::InvalidInputStride;
    }
    std::size_t required = 0;
    if (!checked_extent(input.rows, input.row_stride, input.columns, required)) {
        return Status::ArithmeticOverflow;
    }
    return input.values.size() < required ? Status::InsufficientInputData : Status::Ok;
}

Status validate_output(FloatMatrixView output) noexcept {
    if (output.rows == 0 || output.columns == 0) {
        return Status::ZeroDimension;
    }
    if (output.row_stride < output.columns) {
        return Status::InvalidOutputStride;
    }
    std::size_t required = 0;
    if (!checked_extent(output.rows, output.row_stride, output.columns, required)) {
        return Status::ArithmeticOverflow;
    }
    return output.values.size() < required ? Status::InsufficientOutputData : Status::Ok;
}

Status validate_matrix(const AffineQuantizedMatrixView& matrix) noexcept {
    if (matrix.rows == 0 || matrix.columns == 0) {
        return Status::ZeroDimension;
    }
    if (matrix.group_size == 0 || matrix.group_size > matrix.columns) {
        return Status::InvalidGroupSize;
    }

    std::size_t packed_width = 0;
    switch (matrix.format) {
    case AffineQuantizedFormat::Q4:
        packed_width = ceiling_divide(matrix.columns, 2);
        break;
    case AffineQuantizedFormat::Q8:
        packed_width = matrix.columns;
        break;
    default:
        return Status::UnsupportedFormat;
    }
    const std::size_t parameter_width = ceiling_divide(matrix.columns, matrix.group_size);
    if (matrix.packed_row_stride < packed_width) {
        return Status::InvalidPackedStride;
    }
    if (matrix.parameter_row_stride < parameter_width) {
        return Status::InvalidParameterStride;
    }

    std::size_t required = 0;
    if (!checked_extent(matrix.rows, matrix.packed_row_stride, packed_width, required)) {
        return Status::ArithmeticOverflow;
    }
    if (matrix.packed.size() < required) {
        return Status::InsufficientPackedData;
    }
    if (!checked_extent(matrix.rows, matrix.parameter_row_stride, parameter_width, required)) {
        return Status::ArithmeticOverflow;
    }
    if (matrix.scales.size() < required) {
        return Status::InsufficientScaleData;
    }
    return matrix.biases.size() < required ? Status::InsufficientBiasData : Status::Ok;
}

Status validate_region(const DenseQuantizedRegion& region, std::size_t input_columns,
                       std::size_t output_columns) noexcept {
    const Status matrix_status = validate_matrix(region.matrix);
    if (matrix_status != Status::Ok) {
        return matrix_status;
    }
    if (region.matrix.columns != input_columns) {
        return Status::IncompatibleColumns;
    }
    if (region.row_count == 0) {
        return Status::InvalidRegionRange;
    }

    std::size_t source_end = 0;
    std::size_t output_end = 0;
    if (!checked_add(region.source_row_begin, region.row_count, source_end) ||
        !checked_add(region.output_column_begin, region.row_count, output_end)) {
        return Status::ArithmeticOverflow;
    }
    if (source_end > region.matrix.rows || output_end > output_columns) {
        return Status::InvalidRegionRange;
    }
    return Status::Ok;
}

std::uint8_t quantized_value(const AffineQuantizedMatrixView& matrix, std::size_t row,
                             std::size_t column) noexcept {
    const std::size_t row_begin = row * matrix.packed_row_stride;
    if (matrix.format == AffineQuantizedFormat::Q8) {
        return matrix.packed[row_begin + column];
    }
    const std::uint8_t byte = matrix.packed[row_begin + column / 2];
    const unsigned shift = static_cast<unsigned>((column % 2) * 4);
    return static_cast<std::uint8_t>((byte >> shift) & 0x0FU);
}

float evaluate(const BfloatMatrixView& input, std::size_t input_row,
               const AffineQuantizedMatrixView& matrix, std::size_t weight_row) noexcept {
    float accumulator = 0.0F;
    const std::size_t parameter_begin = weight_row * matrix.parameter_row_stride;
    const std::size_t input_begin = input_row * input.row_stride;
    for (std::size_t column = 0; column < input.columns; ++column) {
        const std::size_t group = column / matrix.group_size;
        const float scale = float_from_bfloat16(matrix.scales[parameter_begin + group]);
        const float bias = float_from_bfloat16(matrix.biases[parameter_begin + group]);
        const float quantized = static_cast<float>(quantized_value(matrix, weight_row, column));
        const float dequantized = std::fma(quantized, scale, bias);
        accumulator = std::fma(float_from_bfloat16(input.values[input_begin + column]), dequantized,
                               accumulator);
    }
    return accumulator;
}

bool intervals_overlap(std::size_t first_begin, std::size_t first_count, std::size_t second_begin,
                       std::size_t second_count) noexcept {
    return first_begin < second_begin + second_count && second_begin < first_begin + first_count;
}

Status validate_dense(BfloatMatrixView input, std::span<const DenseQuantizedRegion> regions,
                      FloatMatrixView output) noexcept {
    const Status input_status = validate_input(input);
    if (input_status != Status::Ok) {
        return input_status;
    }
    const Status output_status = validate_output(output);
    if (output_status != Status::Ok) {
        return output_status;
    }
    if (input.rows != output.rows) {
        return Status::InvalidRegionRange;
    }
    if (regions.empty()) {
        return Status::EmptyRegions;
    }

    std::size_t next_output_column = 0;
    for (const DenseQuantizedRegion& region : regions) {
        const Status region_status = validate_region(region, input.columns, output.columns);
        if (region_status != Status::Ok) {
            return region_status;
        }
        if (region.output_column_begin != next_output_column) {
            return Status::InvalidRegionOrder;
        }
        if (!checked_add(next_output_column, region.row_count, next_output_column)) {
            return Status::ArithmeticOverflow;
        }
    }
    return next_output_column == output.columns ? Status::Ok : Status::InvalidRegionOrder;
}

Status validate_routed(BfloatMatrixView input, std::span<const RoutedQuantizedRegion> regions,
                       FloatMatrixView output) noexcept {
    const Status input_status = validate_input(input);
    if (input_status != Status::Ok) {
        return input_status;
    }
    const Status output_status = validate_output(output);
    if (output_status != Status::Ok) {
        return output_status;
    }
    if (regions.empty()) {
        return Status::EmptyRegions;
    }

    for (std::size_t region_index = 0; region_index < regions.size(); ++region_index) {
        const RoutedQuantizedRegion& region = regions[region_index];
        const Status region_status = validate_region(region.region, input.columns, output.columns);
        if (region_status != Status::Ok) {
            return region_status;
        }
        for (std::size_t route_index = 0; route_index < region.routes.size(); ++route_index) {
            const RoutedRowIndex route = region.routes[route_index];
            if (route.input_row >= input.rows || route.output_row >= output.rows) {
                return Status::InvalidRouteIndex;
            }
            for (std::size_t previous_region = 0; previous_region <= region_index;
                 ++previous_region) {
                const RoutedQuantizedRegion& previous = regions[previous_region];
                const std::size_t route_limit =
                    previous_region == region_index ? route_index : previous.routes.size();
                if (!intervals_overlap(region.region.output_column_begin, region.region.row_count,
                                       previous.region.output_column_begin,
                                       previous.region.row_count)) {
                    continue;
                }
                for (std::size_t previous_route = 0; previous_route < route_limit;
                     ++previous_route) {
                    if (previous.routes[previous_route].output_row == route.output_row) {
                        return Status::ConflictingRouteDestination;
                    }
                }
            }
        }
    }
    return Status::Ok;
}

} // namespace

float float_from_bfloat16(std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

QuantizedGemmReferenceStatus
dense_affine_quantized_gemm_reference(BfloatMatrixView input,
                                      std::span<const DenseQuantizedRegion> regions,
                                      FloatMatrixView output) noexcept {
    const Status status = validate_dense(input, regions, output);
    if (status != Status::Ok) {
        return status;
    }
    for (std::size_t input_row = 0; input_row < input.rows; ++input_row) {
        for (const DenseQuantizedRegion& region : regions) {
            for (std::size_t local_row = 0; local_row < region.row_count; ++local_row) {
                output.values[input_row * output.row_stride + region.output_column_begin +
                              local_row] =
                    evaluate(input, input_row, region.matrix, region.source_row_begin + local_row);
            }
        }
    }
    return Status::Ok;
}

QuantizedGemmReferenceStatus
routed_affine_quantized_gemm_reference(BfloatMatrixView input,
                                       std::span<const RoutedQuantizedRegion> regions,
                                       FloatMatrixView output) noexcept {
    const Status status = validate_routed(input, regions, output);
    if (status != Status::Ok) {
        return status;
    }
    for (const RoutedQuantizedRegion& routed : regions) {
        const DenseQuantizedRegion& region = routed.region;
        for (const RoutedRowIndex route : routed.routes) {
            for (std::size_t local_row = 0; local_row < region.row_count; ++local_row) {
                output.values[route.output_row * output.row_stride + region.output_column_begin +
                              local_row] = evaluate(input, route.input_row, region.matrix,
                                                    region.source_row_begin + local_row);
            }
        }
    }
    return Status::Ok;
}

} // namespace tatara::testing
