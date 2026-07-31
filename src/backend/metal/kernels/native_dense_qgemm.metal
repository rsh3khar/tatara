#include <metal_simdgroup_matrix>

static_assert(
    kNativeDenseQgemmN1TileRows == 32u,
    "N1 requires a 32-row tile");
static_assert(
    kNativeDenseQgemmN1TileColumns == 32u,
    "N1 requires a 32-column tile");
static_assert(
    kNativeDenseQgemmN1ReductionColumns == 32u,
    "N1 requires a 32-column reduction tile");
static_assert(
    kNativeDenseQgemmN1Simdgroups == 4u,
    "N1 requires four simdgroups");
static_assert(
    kNativeDenseQgemmN1SimdgroupGridRows == 2u &&
        kNativeDenseQgemmN1SimdgroupGridColumns == 2u,
    "N1 requires a balanced 2x2 simdgroup grid");
static_assert(
    kNativeDenseQgemmN1SimdgroupGridRows *
            kNativeDenseQgemmN1SimdgroupGridColumns ==
        kNativeDenseQgemmN1Simdgroups,
    "N1 simdgroup grid must cover the threadgroup");
static_assert(
    kNativeDenseQgemmN1Threads == 128u,
    "N1 requires 128 threads");
static_assert(
    kNativeDenseQgemmN1StageRowPadding * sizeof(bfloat) == 16u,
    "N1 staging rows require 16-byte padding");
static_assert(
    kNativeDenseQgemmN1ActivationStageStride ==
            kNativeDenseQgemmN1ReductionColumns +
                kNativeDenseQgemmN1StageRowPadding &&
        kNativeDenseQgemmN1WeightStageStride ==
            kNativeDenseQgemmN1TileColumns +
                kNativeDenseQgemmN1StageRowPadding,
    "N1 stage strides must include the generated padding");
static_assert(
    kNativeDenseQgemmN1ReductionColumns <= kQ4GroupSize,
    "one N1 reduction tile may not cross an affine group");
static_assert(
    kQ4GroupSize % kNativeDenseQgemmN1ReductionColumns == 0u,
    "affine groups must contain complete N1 reduction tiles");
static_assert(
    kNativeDenseQgemmN1ThreadgroupMemoryBytes <=
        kMinimumThreadgroupMemoryBytes,
    "N1 staging must fit the guaranteed threadgroup-memory budget");
static_assert(
    kNativeDenseQgemmN1ThreadgroupMemoryBytes ==
        (kNativeDenseQgemmN1TileRows *
             kNativeDenseQgemmN1ActivationStageStride +
         kNativeDenseQgemmN1ReductionColumns *
             kNativeDenseQgemmN1WeightStageStride) *
            sizeof(bfloat),
    "N1 threadgroup-memory accounting must match both staged tiles");
static_assert(
    kNativeDenseQgemmN1AccumulatorElements ==
        kNativeDenseQgemmN1TileRows *
            kNativeDenseQgemmN1TileColumns,
    "N1 accumulator coverage must equal one output tile");
static_assert(
    kNativeDenseQgemmN1TileRows % 8u == 0u &&
        kNativeDenseQgemmN1TileColumns % 8u == 0u &&
        kNativeDenseQgemmN1ReductionColumns % 8u == 0u,
    "N1 tiles must contain complete 8x8 matrix fragments");
static_assert(
    kNativeDenseQgemmN1PackedWordsPerTileRow *
            kQ4ValuesPerWord ==
        kNativeDenseQgemmN1ReductionColumns,
    "N1 packed-word rows must cover the reduction tile exactly");
static_assert(
    kNativeDenseQgemmN1PackedWordsPerTile ==
            kNativeDenseQgemmN1TileColumns *
                kNativeDenseQgemmN1PackedWordsPerTileRow &&
        kNativeDenseQgemmN1PackedWordsPerTile ==
            kNativeDenseQgemmN1Threads,
    "N1 requires exactly one packed-word read per thread");
static_assert(
    kNativeDenseQgemmN1Threads * 8u ==
            kNativeDenseQgemmN1TileRows *
                kNativeDenseQgemmN1ReductionColumns &&
        kNativeDenseQgemmN1ActivationStageStride % 8u == 0u,
    "N1 aligned activation staging requires one bfloat8 load per thread");
static_assert(
    kNativeDenseQgemmN1MetadataPairsPerGroupTile ==
            kNativeDenseQgemmN1TileColumns &&
        kNativeDenseQgemmN1Threads /
                kNativeDenseQgemmN1PackedWordsPerTileRow ==
            kNativeDenseQgemmN1MetadataPairsPerGroupTile,
    "N1 requires one affine metadata owner per output column");

inline ushort2 native_dense_qgemm_n1_fragment_coordinate(uint lane) {
    const ushort row =
        ushort(((lane >> 2u) & 4u) | ((lane >> 1u) & 3u));
    const ushort column =
        ushort(((lane >> 1u) & 4u) | ((lane & 1u) << 1u));
    return ushort2(column, row);
}

inline void native_dense_qgemm_n1_load_fragment(
    thread float2& values,
    threadgroup const bfloat* source,
    uint row_stride,
    uint row_offset,
    uint column_offset,
    uint lane) {
    const ushort2 coordinate =
        native_dense_qgemm_n1_fragment_coordinate(lane);
    const uint base =
        (row_offset + uint(coordinate.y)) * row_stride +
        column_offset + uint(coordinate.x);
    values.x = float(source[base]);
    values.y = float(source[base + 1u]);
}

inline void native_dense_qgemm_n1_mma(
    thread float2& accumulator,
    float2 left,
    float2 right) {
    simdgroup_matrix<float, 8, 8> result;
    simdgroup_matrix<float, 8, 8> left_matrix;
    simdgroup_matrix<float, 8, 8> right_matrix;
    simdgroup_matrix<float, 8, 8> accumulator_matrix;
    left_matrix.thread_elements()[0] = left.x;
    left_matrix.thread_elements()[1] = left.y;
    right_matrix.thread_elements()[0] = right.x;
    right_matrix.thread_elements()[1] = right.y;
    accumulator_matrix.thread_elements()[0] = accumulator.x;
    accumulator_matrix.thread_elements()[1] = accumulator.y;
    simdgroup_multiply_accumulate(
        result,
        left_matrix,
        right_matrix,
        accumulator_matrix);
    accumulator.x = result.thread_elements()[0];
    accumulator.y = result.thread_elements()[1];
}

inline void native_dense_qgemm_n1_copy_bfloat8(
    threadgroup bfloat* destination,
    device const bfloat* source) {
    *((threadgroup uint4*)destination) =
        *((device const uint4*)source);
}

inline void native_dense_qgemm_n1_zero_bfloat8(
    threadgroup bfloat* destination) {
    *((threadgroup uint4*)destination) = uint4(0u);
}

kernel void native_dense_qgemm_q4_bf16_n1(
    device const bfloat* activations [[buffer(0)]],
    device const uint* packed_weights [[buffer(1)]],
    device const bfloat* scales [[buffer(2)]],
    device const bfloat* biases [[buffer(3)]],
    device bfloat* output [[buffer(4)]],
    constant uint& input_rows [[buffer(5)]],
    constant uint& output_columns [[buffer(6)]],
    constant uint& reduction_columns [[buffer(7)]],
    constant ulong& activation_row_stride_elements [[buffer(8)]],
    constant ulong& packed_weight_row_stride_words [[buffer(9)]],
    constant ulong& parameter_row_stride_elements [[buffer(10)]],
    constant ulong& output_row_stride_elements [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroup_width [[threads_per_simdgroup]],
    uint3 threadgroup_shape [[threads_per_threadgroup]]) {
    if (simdgroup_width != kSimdgroupWidth ||
        threadgroup_shape.x != kNativeDenseQgemmN1Threads ||
        threadgroup_shape.y != 1u ||
        threadgroup_shape.z != 1u ||
        reduction_columns == 0u ||
        reduction_columns % kQ4GroupSize != 0u ||
        activation_row_stride_elements < ulong(reduction_columns) ||
        packed_weight_row_stride_words <
            ulong(reduction_columns / kQ4ValuesPerWord) ||
        parameter_row_stride_elements <
            ulong(reduction_columns / kQ4GroupSize) ||
        output_row_stride_elements < ulong(output_columns)) {
        return;
    }

    threadgroup bfloat
        activation_tile[kNativeDenseQgemmN1TileRows *
                        kNativeDenseQgemmN1ActivationStageStride];
    threadgroup bfloat
        weight_tile[kNativeDenseQgemmN1ReductionColumns *
                    kNativeDenseQgemmN1WeightStageStride];

    float2 accumulators[
            (kNativeDenseQgemmN1TileRows / 8u /
             kNativeDenseQgemmN1SimdgroupGridRows) *
            (kNativeDenseQgemmN1TileColumns / 8u /
             kNativeDenseQgemmN1SimdgroupGridColumns)];
    const uint row_fragment_count =
        kNativeDenseQgemmN1TileRows / 8u /
        kNativeDenseQgemmN1SimdgroupGridRows;
    const uint column_fragment_count =
        kNativeDenseQgemmN1TileColumns / 8u /
        kNativeDenseQgemmN1SimdgroupGridColumns;
    for (uint accumulator_index = 0u;
         accumulator_index <
         row_fragment_count * column_fragment_count;
         ++accumulator_index) {
        accumulators[accumulator_index] = float2(0.0f);
    }

    const ulong tile_row =
        ulong(group.y) * ulong(kNativeDenseQgemmN1TileRows);
    const ulong tile_column =
        ulong(group.x) * ulong(kNativeDenseQgemmN1TileColumns);
    const uint simdgroup_row =
        simdgroup / kNativeDenseQgemmN1SimdgroupGridColumns;
    const uint simdgroup_column =
        simdgroup % kNativeDenseQgemmN1SimdgroupGridColumns;
    const uint local_weight_column =
        thread_index / kNativeDenseQgemmN1PackedWordsPerTileRow;
    const uint packed_word_in_row =
        thread_index % kNativeDenseQgemmN1PackedWordsPerTileRow;
    const ulong global_weight_column =
        tile_column + ulong(local_weight_column);
    const ushort metadata_source_lane =
        ushort(lane - packed_word_in_row);
    float affine_scale = 0.0f;
    float affine_bias = 0.0f;

    for (uint reduction_tile = 0u;
         reduction_tile < reduction_columns;
         reduction_tile += kNativeDenseQgemmN1ReductionColumns) {
        if (activation_row_stride_elements % 8u == 0u) {
            const uint local_row = thread_index / 4u;
            const uint local_reduction =
                (thread_index % 4u) * 8u;
            const ulong global_row = tile_row + ulong(local_row);
            threadgroup bfloat* destination =
                activation_tile +
                local_row *
                    kNativeDenseQgemmN1ActivationStageStride +
                local_reduction;
            if (global_row < ulong(input_rows)) {
                native_dense_qgemm_n1_copy_bfloat8(
                    destination,
                    activations +
                        global_row *
                            activation_row_stride_elements +
                        ulong(
                            reduction_tile +
                            local_reduction));
            } else {
                native_dense_qgemm_n1_zero_bfloat8(
                    destination);
            }
        } else {
            for (uint tile_index = thread_index;
                 tile_index <
                 kNativeDenseQgemmN1TileRows *
                     kNativeDenseQgemmN1ReductionColumns;
                 tile_index += kNativeDenseQgemmN1Threads) {
                const uint local_row =
                    tile_index /
                    kNativeDenseQgemmN1ReductionColumns;
                const uint local_reduction =
                    tile_index %
                    kNativeDenseQgemmN1ReductionColumns;
                const ulong global_row =
                    tile_row + ulong(local_row);
                const uint stage_index =
                    local_row *
                        kNativeDenseQgemmN1ActivationStageStride +
                    local_reduction;
                activation_tile[stage_index] =
                    global_row < ulong(input_rows)
                        ? activations[
                              global_row *
                                      activation_row_stride_elements +
                                  ulong(
                                      reduction_tile +
                                      local_reduction)]
                        : static_cast<bfloat>(0.0f);
            }
        }

        if (reduction_tile % kQ4GroupSize == 0u) {
            float next_scale = 0.0f;
            float next_bias = 0.0f;
            if (packed_word_in_row == 0u &&
                global_weight_column < ulong(output_columns)) {
                const ulong parameter_index =
                    global_weight_column *
                        parameter_row_stride_elements +
                    ulong(reduction_tile / kQ4GroupSize);
                next_scale = float(scales[parameter_index]);
                next_bias = float(biases[parameter_index]);
            }
            affine_scale =
                simd_broadcast(next_scale, metadata_source_lane);
            affine_bias =
                simd_broadcast(next_bias, metadata_source_lane);
        }

        const uint global_reduction_base =
            reduction_tile +
            packed_word_in_row * kQ4ValuesPerWord;
        uint word = 0u;
        if (global_weight_column < ulong(output_columns) &&
            global_reduction_base < reduction_columns) {
            const ulong word_index =
                global_weight_column *
                    packed_weight_row_stride_words +
                ulong(global_reduction_base /
                      kQ4ValuesPerWord);
            word = packed_weights[word_index];
        }
        for (uint packed_value = 0u;
             packed_value < kQ4ValuesPerWord;
             ++packed_value) {
            const uint local_reduction =
                packed_word_in_row * kQ4ValuesPerWord +
                packed_value;
            const uint quantized =
                (word >> (4u * packed_value)) & 15u;
            const uint stage_index =
                local_reduction *
                    kNativeDenseQgemmN1WeightStageStride +
                local_weight_column;
            weight_tile[stage_index] =
                static_cast<bfloat>(
                    float(quantized) * affine_scale +
                    affine_bias);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint reduction_fragment = 0u;
             reduction_fragment <
             kNativeDenseQgemmN1ReductionColumns / 8u;
            ++reduction_fragment) {
            float2 activation_fragments[
                    kNativeDenseQgemmN1TileRows / 8u /
                    kNativeDenseQgemmN1SimdgroupGridRows];
            float2 weight_fragments[
                    kNativeDenseQgemmN1TileColumns / 8u /
                    kNativeDenseQgemmN1SimdgroupGridColumns];
            for (uint row_fragment = 0u;
                 row_fragment < row_fragment_count;
                 ++row_fragment) {
                native_dense_qgemm_n1_load_fragment(
                    activation_fragments[row_fragment],
                    activation_tile,
                    kNativeDenseQgemmN1ActivationStageStride,
                    simdgroup_row *
                            (kNativeDenseQgemmN1TileRows /
                             kNativeDenseQgemmN1SimdgroupGridRows) +
                        row_fragment * 8u,
                    reduction_fragment * 8u,
                    lane);
            }
            for (uint column_fragment = 0u;
                 column_fragment < column_fragment_count;
                 ++column_fragment) {
                native_dense_qgemm_n1_load_fragment(
                    weight_fragments[column_fragment],
                    weight_tile,
                    kNativeDenseQgemmN1WeightStageStride,
                    reduction_fragment * 8u,
                    simdgroup_column *
                            (kNativeDenseQgemmN1TileColumns /
                             kNativeDenseQgemmN1SimdgroupGridColumns) +
                        column_fragment * 8u,
                    lane);
            }
            for (uint row_fragment = 0u;
                 row_fragment < row_fragment_count;
                 ++row_fragment) {
                for (uint column_fragment = 0u;
                     column_fragment < column_fragment_count;
                     ++column_fragment) {
                    const uint accumulator =
                        row_fragment * column_fragment_count +
                        column_fragment;
                    native_dense_qgemm_n1_mma(
                        accumulators[accumulator],
                        activation_fragments[row_fragment],
                        weight_fragments[column_fragment]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const ushort2 coordinate =
        native_dense_qgemm_n1_fragment_coordinate(lane);
    for (uint row_fragment = 0u;
         row_fragment < row_fragment_count;
         ++row_fragment) {
        const ulong global_row =
            tile_row +
            ulong(
                simdgroup_row *
                        (kNativeDenseQgemmN1TileRows /
                         kNativeDenseQgemmN1SimdgroupGridRows) +
                    row_fragment * 8u + uint(coordinate.y));
        if (global_row >= ulong(input_rows)) {
            continue;
        }
        for (uint column_fragment = 0u;
             column_fragment < column_fragment_count;
             ++column_fragment) {
            const uint accumulator =
                row_fragment * column_fragment_count +
                column_fragment;
            const ulong first_column =
                tile_column +
                ulong(
                    simdgroup_column *
                            (kNativeDenseQgemmN1TileColumns /
                             kNativeDenseQgemmN1SimdgroupGridColumns) +
                        column_fragment * 8u +
                        uint(coordinate.x));
            if (first_column < ulong(output_columns)) {
                output[
                    global_row * output_row_stride_elements +
                    first_column] =
                    static_cast<bfloat>(
                        accumulators[accumulator].x);
            }
            if (first_column + 1u < ulong(output_columns)) {
                output[
                    global_row * output_row_stride_elements +
                    first_column + 1u] =
                    static_cast<bfloat>(
                        accumulators[accumulator].y);
            }
        }
    }
}
