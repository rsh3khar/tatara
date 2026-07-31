#include "quantized_gemm_reference.h"

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <span>

namespace {

using tatara::testing::AffineQuantizedFormat;
using tatara::testing::AffineQuantizedMatrixView;
using tatara::testing::BfloatMatrixView;
using tatara::testing::DenseQuantizedRegion;
using tatara::testing::FloatMatrixView;
using tatara::testing::QuantizedGemmReferenceStatus;
using tatara::testing::RoutedQuantizedRegion;
using tatara::testing::RoutedRowIndex;

std::atomic<std::size_t> allocation_count{0};
int failures = 0;

std::uint16_t bfloat16(float value) noexcept {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7F800000U) == 0x7F800000U) {
        return static_cast<std::uint16_t>(bits >> 16U);
    }
    const std::uint32_t rounded = bits + 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(rounded >> 16U);
}

void check(bool condition, const char* message) noexcept {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        ++failures;
    }
}

void check_value(float actual, float expected, const char* message) noexcept {
    check(actual == expected, message);
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
    using tatara::testing::dense_affine_quantized_gemm_reference;
    using tatara::testing::float_from_bfloat16;
    using tatara::testing::routed_affine_quantized_gemm_reference;

    check(float_from_bfloat16(0x3F80U) == 1.0F && float_from_bfloat16(0xC020U) == -2.5F,
          "bfloat16 widening preserves exact sign and magnitude");

    const std::array<std::uint16_t, 7> q4_input = {
        bfloat16(1.0F), bfloat16(-2.0F), bfloat16(3.0F), bfloat16(-4.0F),
        bfloat16(0.5F), 0x7FC1U,         0x7FC1U,
    };
    const std::array<std::uint8_t, 8> q4_packed = {
        0x10U, 0x2FU, 0xA3U, 0xCCU, 0x0FU, 0xF8U, 0xF0U, 0xCCU,
    };
    const std::array<std::uint16_t, 6> q4_scales = {
        bfloat16(2.0F),  bfloat16(-1.0F), bfloat16(99.0F),
        bfloat16(-0.5F), bfloat16(2.0F),  bfloat16(99.0F),
    };
    const std::array<std::uint16_t, 6> q4_biases = {
        bfloat16(-1.0F), bfloat16(4.0F),  bfloat16(99.0F),
        bfloat16(1.0F),  bfloat16(-3.0F), bfloat16(99.0F),
    };
    const AffineQuantizedMatrixView q4_matrix{
        AffineQuantizedFormat::Q4, q4_packed, q4_scales, q4_biases, 2, 5, 3, 4, 3,
    };
    const std::array<DenseQuantizedRegion, 1> q4_regions = {
        DenseQuantizedRegion{q4_matrix, 0, 2, 0},
    };
    std::array<float, 3> q4_output = {-91.0F, -91.0F, -91.0F};
    const QuantizedGemmReferenceStatus q4_status = dense_affine_quantized_gemm_reference(
        BfloatMatrixView{q4_input, 1, 5, 7}, q4_regions, FloatMatrixView{q4_output, 1, 2, 3});
    check(q4_status == QuantizedGemmReferenceStatus::Ok,
          "Q4 generic-group reference accepts a partial final group");
    check_value(q4_output[0], 76.5F,
                "Q4 low/high nibbles, negative inputs, scale and bias match hand result");
    check_value(q4_output[1], -127.0F, "Q4 negative scale and bias match hand result");
    check_value(q4_output[2], -91.0F, "Q4 ignores row padding and the unused tail nibble");

    std::array<std::uint16_t, 132> q8_input{};
    for (std::size_t column = 0; column < 65; ++column) {
        q8_input[column] = bfloat16(1.0F);
        q8_input[67 + column] = bfloat16(-1.0F);
    }
    std::array<std::uint8_t, 65> q8_packed{};
    q8_packed.fill(2U);
    q8_packed[0] = 0U;
    q8_packed[1] = 255U;
    q8_packed[64] = 255U;
    const std::array<std::uint16_t, 2> q8_scales = {
        bfloat16(0.5F),
        bfloat16(-1.0F),
    };
    const std::array<std::uint16_t, 2> q8_biases = {
        bfloat16(-1.0F),
        bfloat16(2.0F),
    };
    const AffineQuantizedMatrixView q8_matrix{
        AffineQuantizedFormat::Q8, q8_packed, q8_scales, q8_biases, 1, 65, 64, 65, 2,
    };
    const std::array<DenseQuantizedRegion, 1> q8_regions = {
        DenseQuantizedRegion{q8_matrix, 0, 1, 0},
    };
    std::array<float, 3> q8_output = {-92.0F, -92.0F, -92.0F};
    check(dense_affine_quantized_gemm_reference(BfloatMatrixView{q8_input, 2, 65, 67}, q8_regions,
                                                FloatMatrixView{q8_output, 2, 1, 2}) ==
              QuantizedGemmReferenceStatus::Ok,
          "Q8 group-64 reference accepts a one-column K tail");
    check_value(q8_output[0], -127.5F, "Q8 treats 255 as unsigned and applies both affine groups");
    check_value(q8_output[2], 127.5F, "Q8 preserves negative activation signs");
    check_value(q8_output[1], -92.0F, "Q8 leaves output row padding untouched");

    const std::array<std::uint16_t, 11> bundle_input = {
        bfloat16(1.0F), bfloat16(0.0F), bfloat16(0.0F), 0x7FC1U,
        bfloat16(0.0F), bfloat16(1.0F), bfloat16(0.0F), 0x7FC1U,
        bfloat16(0.0F), bfloat16(0.0F), bfloat16(1.0F),
    };
    const std::array<std::uint8_t, 4> bundle_first_packed = {
        0x00U,
        0x00U,
        0x54U,
        0xE6U,
    };
    const std::array<std::uint16_t, 4> bundle_first_scales = {
        bfloat16(1.0F),
        bfloat16(1.0F),
        bfloat16(1.0F),
        bfloat16(2.0F),
    };
    const std::array<std::uint16_t, 4> bundle_first_biases = {
        bfloat16(0.0F),
        bfloat16(0.0F),
        bfloat16(0.0F),
        bfloat16(-1.0F),
    };
    const AffineQuantizedMatrixView bundle_first{
        AffineQuantizedFormat::Q4,
        bundle_first_packed,
        bundle_first_scales,
        bundle_first_biases,
        2,
        3,
        2,
        2,
        2,
    };
    const std::array<std::uint8_t, 4> bundle_second_packed = {
        0x21U,
        0xF3U,
        0x87U,
        0xF9U,
    };
    const std::array<std::uint16_t, 4> bundle_second_scales = {
        bfloat16(1.0F),
        bfloat16(1.0F),
        bfloat16(-1.0F),
        bfloat16(0.5F),
    };
    const std::array<std::uint16_t, 4> bundle_second_biases = {
        bfloat16(0.0F),
        bfloat16(0.0F),
        bfloat16(10.0F),
        bfloat16(1.0F),
    };
    const AffineQuantizedMatrixView bundle_second{
        AffineQuantizedFormat::Q4,
        bundle_second_packed,
        bundle_second_scales,
        bundle_second_biases,
        2,
        3,
        2,
        2,
        2,
    };
    const std::array<DenseQuantizedRegion, 2> bundle_regions = {
        DenseQuantizedRegion{bundle_first, 1, 1, 0},
        DenseQuantizedRegion{bundle_second, 0, 2, 1},
    };
    std::array<float, 12> bundle_output{};
    bundle_output.fill(-93.0F);
    check(dense_affine_quantized_gemm_reference(
              BfloatMatrixView{bundle_input, 3, 3, 4}, bundle_regions,
              FloatMatrixView{bundle_output, 3, 3, 4}) == QuantizedGemmReferenceStatus::Ok,
          "dense ordered bundle handles M, N and K tails");
    const std::array<float, 9> bundle_expected = {
        4.0F, 1.0F, 3.0F, 5.0F, 2.0F, 2.0F, 11.0F, 3.0F, 5.5F,
    };
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 3; ++column) {
            check_value(bundle_output[row * 4 + column], bundle_expected[row * 3 + column],
                        "dense bundle output matches hand-dequantized basis rows");
        }
        check_value(bundle_output[row * 4 + 3], -93.0F,
                    "dense bundle leaves output stride padding untouched");
    }

    std::array<float, 11> rejected_output{};
    rejected_output.fill(-94.0F);
    auto invalid_matrix = bundle_first;
    invalid_matrix.format = static_cast<AffineQuantizedFormat>(255);
    std::array<DenseQuantizedRegion, 1> invalid_regions = {
        DenseQuantizedRegion{invalid_matrix, 0, 1, 0},
    };
    check(dense_affine_quantized_gemm_reference(BfloatMatrixView{bundle_input, 3, 3, 4},
                                                invalid_regions,
                                                FloatMatrixView{rejected_output, 3, 1, 4}) ==
              QuantizedGemmReferenceStatus::UnsupportedFormat,
          "unknown quantized formats fail closed");

    invalid_matrix = bundle_first;
    invalid_matrix.group_size = 0;
    invalid_regions[0].matrix = invalid_matrix;
    check(dense_affine_quantized_gemm_reference(BfloatMatrixView{bundle_input, 3, 3, 4},
                                                invalid_regions,
                                                FloatMatrixView{rejected_output, 3, 1, 4}) ==
              QuantizedGemmReferenceStatus::InvalidGroupSize,
          "zero quantization group fails before output");
    invalid_matrix.group_size = 4;
    invalid_regions[0].matrix = invalid_matrix;
    check(dense_affine_quantized_gemm_reference(BfloatMatrixView{bundle_input, 3, 3, 4},
                                                invalid_regions,
                                                FloatMatrixView{rejected_output, 3, 1, 4}) ==
              QuantizedGemmReferenceStatus::InvalidGroupSize,
          "a quantization group wider than K fails closed");

    invalid_matrix = bundle_first;
    invalid_matrix.scales = std::span<const std::uint16_t>{bundle_first_scales}.first(3);
    invalid_regions[0] = DenseQuantizedRegion{invalid_matrix, 0, 1, 0};
    check(dense_affine_quantized_gemm_reference(BfloatMatrixView{bundle_input, 3, 3, 4},
                                                invalid_regions,
                                                FloatMatrixView{rejected_output, 3, 1, 4}) ==
              QuantizedGemmReferenceStatus::InsufficientScaleData,
          "a missing final-group scale fails before computation");

    invalid_matrix = bundle_first;
    invalid_matrix.packed_row_stride = 1;
    invalid_regions[0].matrix = invalid_matrix;
    check(dense_affine_quantized_gemm_reference(BfloatMatrixView{bundle_input, 3, 3, 4},
                                                invalid_regions,
                                                FloatMatrixView{rejected_output, 3, 1, 4}) ==
              QuantizedGemmReferenceStatus::InvalidPackedStride,
          "short packed row stride fails closed");

    std::array<DenseQuantizedRegion, 2> disordered_regions = bundle_regions;
    disordered_regions[0].output_column_begin = 1;
    check(dense_affine_quantized_gemm_reference(BfloatMatrixView{bundle_input, 3, 3, 4},
                                                disordered_regions,
                                                FloatMatrixView{rejected_output, 3, 3, 4}) ==
              QuantizedGemmReferenceStatus::InvalidRegionOrder,
          "a gap in dense bundle output columns fails closed");

    invalid_matrix = bundle_first;
    invalid_matrix.rows = std::numeric_limits<std::size_t>::max();
    invalid_matrix.packed_row_stride = std::numeric_limits<std::size_t>::max();
    invalid_regions[0].matrix = invalid_matrix;
    check(dense_affine_quantized_gemm_reference(BfloatMatrixView{bundle_input, 3, 3, 4},
                                                invalid_regions,
                                                FloatMatrixView{rejected_output, 3, 1, 4}) ==
              QuantizedGemmReferenceStatus::ArithmeticOverflow,
          "packed extent overflow is typed before span access");
    for (const float value : rejected_output) {
        check_value(value, -94.0F, "all dense validation failures leave output untouched");
    }

    const std::array<std::uint16_t, 15> routed_input = {
        bfloat16(1.0F), bfloat16(1.0F), bfloat16(1.0F),  bfloat16(1.0F), bfloat16(0.0F),
        bfloat16(1.0F), bfloat16(2.0F), bfloat16(1.0F),  bfloat16(0.0F), bfloat16(0.0F),
        bfloat16(2.0F), bfloat16(0.0F), bfloat16(-1.0F), bfloat16(1.0F), bfloat16(2.0F),
    };
    const std::array<std::uint8_t, 12> routed_packed = {
        1U, 2U, 3U, 0xEEU, 4U, 5U, 6U, 0xEEU, 7U, 8U, 9U, 0xEEU,
    };
    const std::array<std::uint16_t, 6> routed_scales = {
        bfloat16(1.0F), bfloat16(1.0F), bfloat16(1.0F),
        bfloat16(1.0F), bfloat16(1.0F), bfloat16(1.0F),
    };
    const std::array<std::uint16_t, 6> routed_biases = {
        bfloat16(0.0F), bfloat16(0.0F), bfloat16(0.0F),
        bfloat16(0.0F), bfloat16(0.0F), bfloat16(0.0F),
    };
    const AffineQuantizedMatrixView routed_matrix{
        AffineQuantizedFormat::Q8, routed_packed, routed_scales, routed_biases, 3, 3, 2, 4, 2,
    };
    const std::array<RoutedRowIndex, 5> routed_rows = {
        RoutedRowIndex{4, 6}, RoutedRowIndex{1, 2}, RoutedRowIndex{3, 5},
        RoutedRowIndex{2, 1}, RoutedRowIndex{0, 4},
    };
    const std::array<std::uint8_t, 3> other_expert_packed = {1U, 2U, 3U};
    const std::array<std::uint16_t, 2> other_expert_scales = {
        bfloat16(-1.0F),
        bfloat16(-1.0F),
    };
    const std::array<std::uint16_t, 2> other_expert_biases = {
        bfloat16(10.0F),
        bfloat16(10.0F),
    };
    const AffineQuantizedMatrixView other_expert{
        AffineQuantizedFormat::Q8,
        other_expert_packed,
        other_expert_scales,
        other_expert_biases,
        1,
        3,
        2,
        3,
        2,
    };
    const std::array<RoutedRowIndex, 1> other_route = {RoutedRowIndex{0, 0}};
    const std::array<RoutedQuantizedRegion, 3> routed_regions = {
        RoutedQuantizedRegion{DenseQuantizedRegion{routed_matrix, 0, 2, 0}, routed_rows},
        RoutedQuantizedRegion{DenseQuantizedRegion{routed_matrix, 2, 1, 2}, routed_rows},
        RoutedQuantizedRegion{DenseQuantizedRegion{other_expert, 0, 1, 0}, other_route},
    };
    std::array<float, 28> routed_output{};
    routed_output.fill(-95.0F);
    check(routed_affine_quantized_gemm_reference(
              BfloatMatrixView{routed_input, 5, 3, 3}, routed_regions,
              FloatMatrixView{routed_output, 7, 3, 4}) == QuantizedGemmReferenceStatus::Ok,
          "ragged routed oracle accepts independent expert lists and a five-row tail");
    check_value(routed_output[0], 24.0F, "second expert route writes its exact destination row");
    const std::array<std::size_t, 5> routed_destinations = {6, 2, 5, 1, 4};
    const std::array<std::array<float, 3>, 5> routed_expected = {
        std::array<float, 3>{7.0F, 13.0F, 19.0F}, std::array<float, 3>{4.0F, 10.0F, 16.0F},
        std::array<float, 3>{4.0F, 10.0F, 16.0F}, std::array<float, 3>{4.0F, 13.0F, 22.0F},
        std::array<float, 3>{6.0F, 15.0F, 24.0F},
    };
    for (std::size_t route = 0; route < routed_rows.size(); ++route) {
        const std::size_t output_begin = routed_destinations[route] * 4;
        for (std::size_t column = 0; column < 3; ++column) {
            check_value(routed_output[output_begin + column], routed_expected[route][column],
                        "routed input/output row indirection preserves list order and values");
        }
    }
    check_value(routed_output[3], -95.0F,
                "partially populated routed rows retain untouched columns");
    check_value(routed_output[12], -95.0F, "unrouted destination rows remain untouched");
    for (std::size_t row = 0; row < 7; ++row) {
        check_value(routed_output[row * 4 + 3], -95.0F,
                    "routed computation leaves output stride padding untouched");
    }

    std::array<float, 28> rejected_routed{};
    rejected_routed.fill(-96.0F);
    const std::array<RoutedRowIndex, 1> bad_input_route = {RoutedRowIndex{5, 0}};
    std::array<RoutedQuantizedRegion, 1> invalid_routed = {
        RoutedQuantizedRegion{DenseQuantizedRegion{routed_matrix, 0, 1, 0}, bad_input_route},
    };
    check(routed_affine_quantized_gemm_reference(BfloatMatrixView{routed_input, 5, 3, 3},
                                                 invalid_routed,
                                                 FloatMatrixView{rejected_routed, 7, 3, 4}) ==
              QuantizedGemmReferenceStatus::InvalidRouteIndex,
          "out-of-range routed input rows fail before writes");

    const std::array<RoutedRowIndex, 1> bad_output_route = {RoutedRowIndex{0, 7}};
    invalid_routed[0].routes = bad_output_route;
    check(routed_affine_quantized_gemm_reference(BfloatMatrixView{routed_input, 5, 3, 3},
                                                 invalid_routed,
                                                 FloatMatrixView{rejected_routed, 7, 3, 4}) ==
              QuantizedGemmReferenceStatus::InvalidRouteIndex,
          "out-of-range routed output rows fail before writes");

    const std::array<RoutedRowIndex, 2> duplicate_routes = {
        RoutedRowIndex{0, 2},
        RoutedRowIndex{1, 2},
    };
    invalid_routed[0].routes = duplicate_routes;
    check(routed_affine_quantized_gemm_reference(BfloatMatrixView{routed_input, 5, 3, 3},
                                                 invalid_routed,
                                                 FloatMatrixView{rejected_routed, 7, 3, 4}) ==
              QuantizedGemmReferenceStatus::ConflictingRouteDestination,
          "overlapping writes to one routed destination fail closed");
    for (const float value : rejected_routed) {
        check_value(value, -96.0F, "all routed validation failures leave output untouched");
    }

    const std::size_t before = allocation_count.load(std::memory_order_relaxed);
    std::array<float, 12> allocation_dense{};
    std::array<float, 28> allocation_routed{};
    const QuantizedGemmReferenceStatus allocation_dense_status =
        dense_affine_quantized_gemm_reference(BfloatMatrixView{bundle_input, 3, 3, 4},
                                              bundle_regions,
                                              FloatMatrixView{allocation_dense, 3, 3, 4});
    const QuantizedGemmReferenceStatus allocation_routed_status =
        routed_affine_quantized_gemm_reference(BfloatMatrixView{routed_input, 5, 3, 3},
                                               routed_regions,
                                               FloatMatrixView{allocation_routed, 7, 3, 4});
    const std::size_t after = allocation_count.load(std::memory_order_relaxed);
    check(allocation_dense_status == QuantizedGemmReferenceStatus::Ok &&
              allocation_routed_status == QuantizedGemmReferenceStatus::Ok && before == after,
          "dense and routed fixed-span computation allocates no heap memory");

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::puts("quantized_gemm_reference_test: PASS");
    return 0;
}
