import re
import struct
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = (
    REPOSITORY_ROOT
    / "src/backend/metal/kernels/native_dense_qgemm.metal"
)
RUNTIME_SOURCE_PATH = REPOSITORY_ROOT / "src/runtime/prefill_step.cpp"

TILE_ROWS = 32
TILE_COLUMNS = 32
REDUCTION_COLUMNS = 32
SIMDGROUPS = 4
THREADS = 128
Q4_GROUP_SIZE = 64
SIMDGROUP_GRID_ROWS = 2
SIMDGROUP_GRID_COLUMNS = 2
STAGE_ROW_PADDING = 8
Q4_VALUES_PER_WORD = 8


def fragment_coordinate(lane: int) -> tuple[int, int]:
    row = ((lane >> 2) & 4) | ((lane >> 1) & 3)
    column = ((lane >> 1) & 4) | ((lane & 1) << 1)
    return column, row


def packed_q4_value(words: list[int], index: int) -> int:
    return (words[index // 8] >> (4 * (index % 8))) & 15


def pack_q4(values: list[int]) -> list[int]:
    words = [0] * ((len(values) + 7) // 8)
    for index, value in enumerate(values):
        words[index // 8] |= value << (4 * (index % 8))
    return words


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bfloat16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFF0000))[0]


def q4_stage_schedule(
    valid_columns: int, reduction_columns: int
) -> tuple[
    list[tuple[int, int, int]],
    list[tuple[int, int]],
    list[tuple[int, int, int]],
]:
    packed_reads = []
    metadata_reads = []
    stage_writes = []
    words_per_tile_row = REDUCTION_COLUMNS // Q4_VALUES_PER_WORD
    for reduction_tile in range(
        0, reduction_columns, REDUCTION_COLUMNS
    ):
        for thread in range(THREADS):
            column = thread // words_per_tile_row
            word_in_tile_row = thread % words_per_tile_row
            for packed_value in range(Q4_VALUES_PER_WORD):
                stage_writes.append(
                    (
                        reduction_tile,
                        word_in_tile_row * Q4_VALUES_PER_WORD
                        + packed_value,
                        column,
                    )
                )
            if column >= valid_columns:
                continue
            global_word = (
                reduction_tile // Q4_VALUES_PER_WORD
                + word_in_tile_row
            )
            packed_reads.append(
                (reduction_tile, column, global_word)
            )
            if (
                reduction_tile % Q4_GROUP_SIZE == 0
                and word_in_tile_row == 0
            ):
                metadata_reads.append(
                    (column, reduction_tile // Q4_GROUP_SIZE)
                )
    return packed_reads, metadata_reads, stage_writes


def tiled_dense_qgemm(
    activations: list[list[float]],
    packed_weights: list[list[int]],
    scales: list[list[float]],
    biases: list[list[float]],
    reduction_columns: int,
    *,
    stage_weights_to_bfloat16: bool = True,
) -> list[list[float]]:
    result = [
        [0.0 for _ in range(len(packed_weights))]
        for _ in range(len(activations))
    ]
    for tile_row in range(0, len(activations), TILE_ROWS):
        for tile_column in range(0, len(packed_weights), TILE_COLUMNS):
            accumulators = [
                [0.0 for _ in range(TILE_COLUMNS)]
                for _ in range(TILE_ROWS)
            ]
            for reduction_tile in range(
                0, reduction_columns, REDUCTION_COLUMNS
            ):
                for local_row in range(TILE_ROWS):
                    global_row = tile_row + local_row
                    if global_row >= len(activations):
                        continue
                    for local_column in range(TILE_COLUMNS):
                        global_column = tile_column + local_column
                        if global_column >= len(packed_weights):
                            continue
                        for local_reduction in range(REDUCTION_COLUMNS):
                            reduction = reduction_tile + local_reduction
                            if reduction >= reduction_columns:
                                continue
                            quantized = packed_q4_value(
                                packed_weights[global_column], reduction
                            )
                            group = reduction // Q4_GROUP_SIZE
                            dequantized = float32(
                                float32(
                                    float(quantized)
                                    * scales[global_column][group]
                                )
                                + biases[global_column][group]
                            )
                            weight = (
                                bfloat16(dequantized)
                                if stage_weights_to_bfloat16
                                else dequantized
                            )
                            product = float32(
                                activations[global_row][reduction]
                                * weight
                            )
                            accumulators[local_row][local_column] = (
                                float32(
                                    accumulators[local_row][local_column]
                                    + product
                                )
                            )
            for local_row in range(TILE_ROWS):
                global_row = tile_row + local_row
                if global_row >= len(activations):
                    continue
                for local_column in range(TILE_COLUMNS):
                    global_column = tile_column + local_column
                    if global_column < len(packed_weights):
                        result[global_row][global_column] = accumulators[
                            local_row
                        ][local_column]
    return result


class NativeDenseQgemmSourceTest(unittest.TestCase):
    def test_every_native_dispatch_rebinds_its_pipeline(self):
        source = RUNTIME_SOURCE_PATH.read_text()
        helper = re.search(
            r"void encode_native_dense_qgemm\((.*?)\n\}",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(helper)
        body = helper.group(1)
        self.assertIn(
            "const PrefillStep& step", body
        )
        compact_body = re.sub(r"\s+", "", body)
        self.assertLess(
            compact_body.index(
                "encode.pipeline(step.pipelines."
                "native_dense_steel_gdn_bm64_wm2_wn2);"
            ),
            compact_body.index("encode.quantized(weights,0);"),
        )
        fallback = compact_body[
            compact_body.index(
                "if(step.policy.native_dense_steel)"
            ):
        ]
        self.assertLess(
            fallback.index(
                "encode.pipeline(step.pipelines.native_dense_steel);"
            ),
            fallback.index("encode.quantized(weights,0);"),
        )
        self.assertLess(
            compact_body.index(
                "encode.pipeline(step.pipelines.native_dense_qgemm);"
            ),
            compact_body.index(
                "encode.buffer(activations,activation_offset,0);"
            ),
        )
        self.assertEqual(
            source.count(
                "encode.pipeline(step.pipelines.native_dense_qgemm);"
            ),
            1,
        )

    def test_fragment_mapping_covers_each_8_by_8_cell_once(self):
        cells = []
        for lane in range(32):
            column, row = fragment_coordinate(lane)
            cells.extend(((row, column), (row, column + 1)))

        self.assertEqual(len(cells), 64)
        self.assertEqual(
            set(cells),
            {(row, column) for row in range(8) for column in range(8)},
        )

    def test_threadgroup_partitions_the_output_and_staging_exactly(self):
        output_cells = []
        for simdgroup in range(SIMDGROUPS):
            simdgroup_row = simdgroup // SIMDGROUP_GRID_COLUMNS
            simdgroup_column = simdgroup % SIMDGROUP_GRID_COLUMNS
            for row_fragment in range(2):
                for column_fragment in range(2):
                    for lane in range(32):
                        column, row = fragment_coordinate(lane)
                        output_cells.extend(
                            (
                                (
                                    simdgroup_row * 16
                                    + row_fragment * 8
                                    + row,
                                    simdgroup_column * 16
                                    + column_fragment * 8
                                    + column,
                                ),
                                (
                                    simdgroup_row * 16
                                    + row_fragment * 8
                                    + row,
                                    simdgroup_column * 16
                                    + column_fragment * 8
                                    + column
                                    + 1,
                                ),
                            )
                        )
        expected_output = {
            (row, column)
            for row in range(TILE_ROWS)
            for column in range(TILE_COLUMNS)
        }
        self.assertEqual(len(output_cells), TILE_ROWS * TILE_COLUMNS)
        self.assertEqual(set(output_cells), expected_output)
        activation_fragments = [
            (simdgroup // SIMDGROUP_GRID_COLUMNS, row_fragment)
            for simdgroup in range(SIMDGROUPS)
            for row_fragment in range(2)
        ]
        weight_fragments = [
            (simdgroup % SIMDGROUP_GRID_COLUMNS, column_fragment)
            for simdgroup in range(SIMDGROUPS)
            for column_fragment in range(2)
        ]
        self.assertTrue(
            all(
                activation_fragments.count(fragment) == 2
                for fragment in set(activation_fragments)
            )
        )
        self.assertTrue(
            all(
                weight_fragments.count(fragment) == 2
                for fragment in set(weight_fragments)
            )
        )

        staged = [
            index
            for thread in range(THREADS)
            for index in range(
                thread, TILE_ROWS * REDUCTION_COLUMNS, THREADS
            )
        ]
        self.assertEqual(
            sorted(staged), list(range(TILE_ROWS * REDUCTION_COLUMNS))
        )
        activation_stage_stride = REDUCTION_COLUMNS + STAGE_ROW_PADDING
        activation_destinations = {
            (index // REDUCTION_COLUMNS) * activation_stage_stride
            + index % REDUCTION_COLUMNS
            for index in staged
        }
        self.assertEqual(len(activation_destinations), len(staged))
        self.assertTrue(
            all(
                index % activation_stage_stride < REDUCTION_COLUMNS
                for index in activation_destinations
            )
        )
        self.assertEqual(
            2
            * TILE_ROWS
            * activation_stage_stride
            * 2,
            5120,
        )

    def test_q4_loader_reads_words_and_metadata_once(self):
        packed_reads, metadata_reads, stage_writes = (
            q4_stage_schedule(TILE_COLUMNS, 128)
        )

        self.assertEqual(len(packed_reads), TILE_COLUMNS * 128 // 8)
        self.assertEqual(len(set(packed_reads)), len(packed_reads))
        self.assertEqual(len(metadata_reads), TILE_COLUMNS * 2)
        self.assertEqual(
            len(set(metadata_reads)), len(metadata_reads)
        )
        self.assertEqual(len(stage_writes), TILE_COLUMNS * 128)
        self.assertEqual(len(set(stage_writes)), len(stage_writes))
        for thread in range(THREADS):
            lane = thread % 32
            word_in_tile_row = thread % 4
            source_lane = lane - word_in_tile_row
            self.assertEqual(source_lane % 4, 0)
            self.assertEqual(source_lane // 4, lane // 4)
        for reduction_tile in range(0, 128, REDUCTION_COLUMNS):
            self.assertEqual(
                sum(
                    tile == reduction_tile
                    for tile, _, _ in packed_reads
                ),
                THREADS,
            )

        tail_reads, tail_metadata, tail_writes = q4_stage_schedule(
            19, 128
        )
        self.assertEqual(len(tail_reads), 19 * 128 // 8)
        self.assertEqual(len(set(tail_reads)), len(tail_reads))
        self.assertEqual(len(tail_metadata), 19 * 2)
        self.assertEqual(
            len(set(tail_metadata)), len(tail_metadata)
        )
        self.assertEqual(len(tail_writes), TILE_COLUMNS * 128)
        self.assertEqual(len(set(tail_writes)), len(tail_writes))

    def test_tiled_q4_affine_contract_covers_m_n_tails(self):
        input_rows = 33
        output_columns = 35
        reduction_columns = 128
        activations = [
            [
                float(((row * 3 + reduction * 5) % 9) - 4) / 4.0
                for reduction in range(reduction_columns)
            ]
            for row in range(input_rows)
        ]
        quantized_rows = [
            [
                (column * 7 + reduction * 3) % 16
                for reduction in range(reduction_columns)
            ]
            for column in range(output_columns)
        ]
        packed_weights = [pack_q4(row) for row in quantized_rows]
        scales = [
            [
                0.125 * (group + 1)
                for group in range(
                    reduction_columns // Q4_GROUP_SIZE
                )
            ]
            for _ in range(output_columns)
        ]
        biases = [
            [
                -0.25
                + 0.125 * (column % 3)
                + 0.0625 * group
                for group in range(
                    reduction_columns // Q4_GROUP_SIZE
                )
            ]
            for column in range(output_columns)
        ]

        tiled = tiled_dense_qgemm(
            activations,
            packed_weights,
            scales,
            biases,
            reduction_columns,
        )
        for row in range(input_rows):
            for column in range(output_columns):
                expected = sum(
                    activations[row][reduction]
                    * (
                        quantized_rows[column][reduction]
                        * scales[column][reduction // Q4_GROUP_SIZE]
                        + biases[column][reduction // Q4_GROUP_SIZE]
                    )
                    for reduction in range(reduction_columns)
                )
                self.assertAlmostEqual(
                    tiled[row][column], expected, places=6
                )

    def test_bfloat_weight_staging_is_a_distinct_numerical_family(self):
        input_rows = 3
        output_columns = 5
        reduction_columns = 128
        activations = [
            [
                bfloat16(
                    ((row * 13 + reduction * 7) % 29 - 14)
                    / 11.0
                    + 0.03
                )
                for reduction in range(reduction_columns)
            ]
            for row in range(input_rows)
        ]
        quantized_rows = [
            [
                (column * 11 + reduction * 5 + 3) % 16
                for reduction in range(reduction_columns)
            ]
            for column in range(output_columns)
        ]
        packed_weights = [pack_q4(row) for row in quantized_rows]
        scales = [
            [
                bfloat16(
                    0.071 + 0.013 * column + 0.019 * group
                )
                for group in range(2)
            ]
            for column in range(output_columns)
        ]
        biases = [
            [
                bfloat16(
                    -0.137 + 0.017 * column - 0.011 * group
                )
                for group in range(2)
            ]
            for column in range(output_columns)
        ]

        staged = tiled_dense_qgemm(
            activations,
            packed_weights,
            scales,
            biases,
            reduction_columns,
        )
        direct = tiled_dense_qgemm(
            activations,
            packed_weights,
            scales,
            biases,
            reduction_columns,
            stage_weights_to_bfloat16=False,
        )
        dequantized_weights = [
            float32(
                float32(
                    float(quantized_rows[column][reduction])
                    * scales[column][reduction // Q4_GROUP_SIZE]
                )
                + biases[column][reduction // Q4_GROUP_SIZE]
            )
            for column in range(output_columns)
            for reduction in range(reduction_columns)
        ]
        self.assertTrue(
            any(
                weight != bfloat16(weight)
                for weight in dequantized_weights
            )
        )

        deltas = [
            abs(staged[row][column] - direct[row][column])
            for row in range(input_rows)
            for column in range(output_columns)
        ]
        self.assertEqual(sum(delta != 0.0 for delta in deltas), 15)
        self.assertAlmostEqual(
            staged[0][0], 6.134232521057129, places=12
        )
        self.assertAlmostEqual(
            direct[0][0], 6.151630401611328, places=12
        )
        self.assertAlmostEqual(
            max(deltas), 0.01739788055419922, places=12
        )
        self.assertAlmostEqual(
            sum(deltas) / len(deltas),
            0.010683838526407878,
            places=12,
        )
        staged_output = [
            bfloat16(value) for row in staged for value in row
        ]
        direct_output = [
            bfloat16(value) for row in direct for value in row
        ]
        output_deltas = [
            abs(staged_value - direct_value)
            for staged_value, direct_value in zip(
                staged_output, direct_output, strict=True
            )
        ]
        self.assertEqual(
            sum(delta != 0.0 for delta in output_deltas), 7
        )
        self.assertEqual(staged_output[0], 6.125)
        self.assertEqual(direct_output[0], 6.15625)
        self.assertEqual(max(output_deltas), 0.0625)
        self.assertAlmostEqual(
            sum(output_deltas) / len(output_deltas),
            0.011197916666666667,
            places=12,
        )

    def test_source_freezes_dense_n1_safety_and_resource_contract(self):
        source = SOURCE_PATH.read_text(encoding="utf-8")
        cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(
            encoding="utf-8"
        )

        self.assertIn("#include <metal_simdgroup_matrix>", source)
        self.assertIn("native_dense_qgemm.metal", cmake)
        self.assertRegex(
            source,
            r"kernel\s+void\s+native_dense_qgemm_q4_bf16_n1\(",
        )
        self.assertIn("simdgroup_multiply_accumulate(", source)
        self.assertIn("[[threads_per_simdgroup]]", source)
        self.assertIn("[[threads_per_threadgroup]]", source)
        self.assertNotRegex(source, r"\bfragment\b")
        self.assertIn(
            "uint3 group [[threadgroup_position_in_grid]]", source
        )
        self.assertIn(
            "uint3 threadgroup_shape [[threads_per_threadgroup]]",
            source,
        )
        self.assertIn(
            "reduction_columns % kQ4GroupSize != 0u", source
        )
        self.assertIn(
            "kNativeDenseQgemmN1SimdgroupGridRows == 2u", source
        )
        self.assertIn(
            "kNativeDenseQgemmN1StageRowPadding * sizeof(bfloat) "
            "== 16u",
            source,
        )
        self.assertIn(
            "kNativeDenseQgemmN1PackedWordsPerTile ==\n"
            "            kNativeDenseQgemmN1Threads",
            source,
        )
        self.assertIn("simd_broadcast(", source)
        self.assertRegex(
            source,
            r"kNativeDenseQgemmN1ThreadgroupMemoryBytes\s*<=\s*"
            r"kMinimumThreadgroupMemoryBytes",
        )
        first_barrier = source.index(
            "threadgroup_barrier(mem_flags::mem_threadgroup)"
        )
        self.assertLess(
            source.index("if (simdgroup_width !="), first_barrier
        )
        self.assertLess(
            source.index("packed_weight_row_stride_words <"),
            first_barrier,
        )
        self.assertEqual(source.count("return;"), 1)
        self.assertEqual(
            source.count(
                "threadgroup bfloat\n"
                "        activation_tile"
            ),
            1,
        )
        self.assertEqual(
            source.count(
                "threadgroup bfloat\n"
                "        weight_tile"
            ),
            1,
        )
        self.assertIn(
            "kNativeDenseQgemmN1Threads * 8u ==",
            source,
        )
        self.assertIn(
            "*((threadgroup uint4*)destination) =",
            source,
        )
        self.assertIn("thread float2& accumulator", source)
        self.assertNotRegex(
            source.lower(),
            r"\b(?:new|malloc|free|atomic|mutex|lock|expert|route|fusion)\b",
        )


if __name__ == "__main__":
    unittest.main()
