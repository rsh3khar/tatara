#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace tatara::testing {

enum class AffineQuantizedFormat : std::uint8_t {
    Q4,
    Q8,
};

enum class QuantizedGemmReferenceStatus : std::uint8_t {
    Ok,
    UnsupportedFormat,
    ZeroDimension,
    InvalidGroupSize,
    ArithmeticOverflow,
    InvalidInputStride,
    InsufficientInputData,
    InvalidPackedStride,
    InsufficientPackedData,
    InvalidParameterStride,
    InsufficientScaleData,
    InsufficientBiasData,
    InvalidOutputStride,
    InsufficientOutputData,
    EmptyRegions,
    InvalidRegionRange,
    InvalidRegionOrder,
    IncompatibleColumns,
    InvalidRouteIndex,
    ConflictingRouteDestination,
};

struct BfloatMatrixView {
    std::span<const std::uint16_t> values{};
    std::size_t rows{0};
    std::size_t columns{0};
    std::size_t row_stride{0};
};

struct FloatMatrixView {
    std::span<float> values{};
    std::size_t rows{0};
    std::size_t columns{0};
    std::size_t row_stride{0};
};

struct AffineQuantizedMatrixView {
    AffineQuantizedFormat format{AffineQuantizedFormat::Q4};
    std::span<const std::uint8_t> packed{};
    std::span<const std::uint16_t> scales{};
    std::span<const std::uint16_t> biases{};
    std::size_t rows{0};
    std::size_t columns{0};
    std::size_t group_size{0};
    std::size_t packed_row_stride{0};
    std::size_t parameter_row_stride{0};
};

struct DenseQuantizedRegion {
    AffineQuantizedMatrixView matrix{};
    std::size_t source_row_begin{0};
    std::size_t row_count{0};
    std::size_t output_column_begin{0};
};

struct RoutedRowIndex {
    std::size_t input_row{0};
    std::size_t output_row{0};
};

struct RoutedQuantizedRegion {
    DenseQuantizedRegion region{};
    std::span<const RoutedRowIndex> routes{};
};

float float_from_bfloat16(std::uint16_t bits) noexcept;

QuantizedGemmReferenceStatus
dense_affine_quantized_gemm_reference(BfloatMatrixView input,
                                      std::span<const DenseQuantizedRegion> regions,
                                      FloatMatrixView output) noexcept;

QuantizedGemmReferenceStatus
routed_affine_quantized_gemm_reference(BfloatMatrixView input,
                                       std::span<const RoutedQuantizedRegion> regions,
                                       FloatMatrixView output) noexcept;

} // namespace tatara::testing
