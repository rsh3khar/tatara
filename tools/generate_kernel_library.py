#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from tatara.artifact_manifest import ArtifactManifestError, parse_manifest
from tatara.qwen36_plan_generation import (
    GeneratedModelPlan,
    ModelPlanGenerationError,
    parse_model_plan,
)

KERNEL_FILE_NAMES = (
    "q4_dot.metal",
    "native_dense_qgemm.metal",
    "prefill_dense.metal",
    "embed_row_q4.metal",
    "rms_only.metal",
    "gdn.metal",
    "prefill_gdn.metal",
    "attention.metal",
    "prefill_attention.metal",
    "q4_dot_packed.metal",
    "q8_dot.metal",
    "moe.metal",
    "prefill_moe.metal",
    "routed_qgemm_tasks.metal",
    "head.metal",
    "adjudication.metal",
    "draft.metal",
    "verify_m16.metal",
)
RAW_STRING_DELIMITER = "TATARA_MSL"
NIBBLES_PER_QUANT_WORD = 8
RMS_VALUES_PER_THREAD = 4
# C++ compilers need only support 65536-character string literals, and Tatara
# builds -Werror -Woverlength-strings, so an oversized literal is a build break.
# The source is emitted as separately NAMED literals joined at runtime. Adjacent
# literals do not work: the limit applies to the concatenated result, so the
# diagnostic still fires. Chunking rather than suppressing the warning keeps the
# ceiling from being a trap the next time a kernel grows -- it had six
# characters of headroom when this was found.
MAX_STRING_LITERAL_CHARS = 65536
SOURCE_CHUNK_CHARS = 60000
# Metal's ceiling on threads in one threadgroup. router_select dispatches one
# threadgroup of `experts` threads, so this bounds the expert count.
MAX_THREADGROUP_THREADS = 1024
MINIMUM_THREADGROUP_MEMORY_BYTES = 32 * 1024
NATIVE_DENSE_QGEMM_N1_TILE_ROWS = 32
NATIVE_DENSE_QGEMM_N1_TILE_COLUMNS = 32
NATIVE_DENSE_QGEMM_N1_REDUCTION_COLUMNS = 32
NATIVE_DENSE_QGEMM_N1_SIMDGROUPS = 4
NATIVE_DENSE_QGEMM_N1_THREADS = 128
NATIVE_DENSE_QGEMM_N1_SIMDGROUP_GRID_ROWS = 2
NATIVE_DENSE_QGEMM_N1_SIMDGROUP_GRID_COLUMNS = 2
NATIVE_DENSE_QGEMM_N1_STAGE_ROW_PADDING = 8
VERIFY_M16_THREADS = 128
VERIFY_M16_MAX_ROWS = 16
VERIFY_M16_COLUMNS_PER_GROUP = 8
DRAFT_DENSE_THREADS = 128
DRAFT_GEMM_THREADS = 128
DRAFT_GEMM_COLUMNS_PER_SIMDGROUP = 16
DRAFT_MAX_BLOCK_ROWS = 16
DRAFT_RMS_MAX_SIMDGROUPS = 16
DRAFT_HEAD_DIMENSION = 128
DRAFT_QUERY_HEADS = 32
DRAFT_KV_HEADS = 8
DRAFT_ATTENTION_THREADS = 128
DRAFT_ATTENTION_KEY_TILE = 32
DRAFT_WINDOW_POSITIONS = 4096
PREFILL_STAGED_ATTENTION_QUERY_TILE_ROWS = 16
PREFILL_STAGED_ATTENTION_KEY_TILE_COLUMNS = 32
PREFILL_STAGED_ATTENTION_OUTPUT_TILE_COLUMNS = 32
PREFILL_STAGED_ATTENTION_SIMDGROUPS = 2
PREFILL_STAGED_ATTENTION_THREADS = 64
PREFILL_STAGED_ATTENTION_SOFTMAX_THREADS = 256
PREFILL_STREAMING_ATTENTION_QUERY_TILE_ROWS = 16
PREFILL_STREAMING_ATTENTION_KEYS_PER_SIMDGROUP = 16
PREFILL_STREAMING_ATTENTION_OUTPUT_COLUMNS_PER_SIMDGROUP = 32
BFLOAT_BYTES = 2
FLOAT_BYTES = 4
NATIVE_DENSE_QGEMM_N1_ACTIVATION_STAGE_STRIDE = (
    NATIVE_DENSE_QGEMM_N1_REDUCTION_COLUMNS
    + NATIVE_DENSE_QGEMM_N1_STAGE_ROW_PADDING
)
NATIVE_DENSE_QGEMM_N1_WEIGHT_STAGE_STRIDE = (
    NATIVE_DENSE_QGEMM_N1_TILE_COLUMNS
    + NATIVE_DENSE_QGEMM_N1_STAGE_ROW_PADDING
)
NATIVE_DENSE_QGEMM_N1_PACKED_WORDS_PER_TILE_ROW = (
    NATIVE_DENSE_QGEMM_N1_REDUCTION_COLUMNS
    // NIBBLES_PER_QUANT_WORD
)
NATIVE_DENSE_QGEMM_N1_PACKED_WORDS_PER_TILE = (
    NATIVE_DENSE_QGEMM_N1_TILE_COLUMNS
    * NATIVE_DENSE_QGEMM_N1_PACKED_WORDS_PER_TILE_ROW
)
NATIVE_DENSE_QGEMM_N1_METADATA_PAIRS_PER_GROUP_TILE = (
    NATIVE_DENSE_QGEMM_N1_TILE_COLUMNS
)
NATIVE_DENSE_QGEMM_N1_THREADGROUP_MEMORY_BYTES = (
    NATIVE_DENSE_QGEMM_N1_TILE_ROWS
    * NATIVE_DENSE_QGEMM_N1_ACTIVATION_STAGE_STRIDE
    * BFLOAT_BYTES
    + NATIVE_DENSE_QGEMM_N1_REDUCTION_COLUMNS
    * NATIVE_DENSE_QGEMM_N1_WEIGHT_STAGE_STRIDE
    * BFLOAT_BYTES
)
NATIVE_DENSE_QGEMM_N1_ACCUMULATOR_ELEMENTS = (
    NATIVE_DENSE_QGEMM_N1_TILE_ROWS
    * NATIVE_DENSE_QGEMM_N1_TILE_COLUMNS
)
NATIVE_ROUTED_QGEMM_R1_TILE_ROWS = 16
NATIVE_ROUTED_QGEMM_R1_TILE_COLUMNS = 32
NATIVE_ROUTED_QGEMM_R1_REDUCTION_COLUMNS = 32
NATIVE_ROUTED_QGEMM_R1_SIMDGROUPS = 2
NATIVE_ROUTED_QGEMM_R1_THREADS = 64
NATIVE_ROUTED_QGEMM_R1_TASK_BYTES = 16
NATIVE_ROUTED_QGEMM_R1_TASK_CAPACITY = 4096
NATIVE_ROUTED_QGEMM_R1_SINGLE_ACCUMULATOR_ELEMENTS = (
    NATIVE_ROUTED_QGEMM_R1_TILE_ROWS
    * NATIVE_ROUTED_QGEMM_R1_TILE_COLUMNS
)
NATIVE_ROUTED_QGEMM_R1_FUSED_ACCUMULATOR_ELEMENTS = (
    2 * NATIVE_ROUTED_QGEMM_R1_SINGLE_ACCUMULATOR_ELEMENTS
)
NATIVE_ROUTED_QGEMM_R1_THREADGROUP_MEMORY_BYTES = 0
# Compatibility identity: the original R1 profile named only the maximum
# treatment footprint, which is the fused gate-and-up treatment.
NATIVE_ROUTED_QGEMM_R1_ACCUMULATOR_ELEMENTS = (
    NATIVE_ROUTED_QGEMM_R1_FUSED_ACCUMULATOR_ELEMENTS
)
NATIVE_ROUTED_QGEMM_R2_STAGE_ROW_PADDING = 8
NATIVE_ROUTED_QGEMM_R2_STAGE_STRIDE = (
    NATIVE_ROUTED_QGEMM_R1_REDUCTION_COLUMNS
    + NATIVE_ROUTED_QGEMM_R2_STAGE_ROW_PADDING
)
NATIVE_ROUTED_QGEMM_R2_FUSED_THREADGROUP_MEMORY_BYTES = (
    NATIVE_ROUTED_QGEMM_R1_TILE_ROWS * 4
    + NATIVE_ROUTED_QGEMM_R1_TILE_ROWS
    * NATIVE_ROUTED_QGEMM_R2_STAGE_STRIDE
    * BFLOAT_BYTES
    + 2
    * NATIVE_ROUTED_QGEMM_R1_TILE_COLUMNS
    * NATIVE_ROUTED_QGEMM_R2_STAGE_STRIDE
    * FLOAT_BYTES
)
NATIVE_ROUTED_QGEMM_R2_DOWN_THREADGROUP_MEMORY_BYTES = (
    NATIVE_ROUTED_QGEMM_R1_TILE_ROWS * 4
    + NATIVE_ROUTED_QGEMM_R1_TILE_ROWS
    * NATIVE_ROUTED_QGEMM_R2_STAGE_STRIDE
    * BFLOAT_BYTES
    + NATIVE_ROUTED_QGEMM_R1_TILE_COLUMNS
    * NATIVE_ROUTED_QGEMM_R2_STAGE_STRIDE
    * FLOAT_BYTES
)
NATIVE_ROUTED_QGEMM_R1_TASK_STATUS_VALUES = (
    ("NotProduced", 0),
    ("Ready", 1),
    ("CountOutOfRange", 2),
    ("RouteConservationFailure", 3),
    ("TaskCapacityExceeded", 4),
    ("PackedSlotOutOfRange", 5),
)

# First-package constants; they move into the model package record when a
# second package needs different values. The key scale is the bfloat16 cast
# of 1/sqrt(128); the query scale 1/128 is exact in bfloat16.
FIRST_PACKAGE_RMS_EPSILON = "1e-6f"
FIRST_PACKAGE_GDN_CONV_WIDTH = 4
FIRST_PACKAGE_GDN_HEAD_DIMENSION = 128
FIRST_PACKAGE_GDN_QUERY_SCALE = "0.0078125f"
FIRST_PACKAGE_GDN_KEY_SCALE = "0.08837890625f"
# RoPE base and the 1/16 attention scale (rsqrt of head dimension 256).
FIRST_PACKAGE_ATTN_HEAD_DIMENSION = 256
FIRST_PACKAGE_ATTN_HEADS_PER_KV = 8
FIRST_PACKAGE_ATTN_ROPE_BASE = "10000000.0f"
FIRST_PACKAGE_ATTN_SCALE = "0.0625f"
# Threadgroup count of the two-stage vocabulary argmax, fixed by the grid-stride
# literals in head.metal rather than by the package.
ARGMAX_GROUPS = 256


class KernelLibraryGenerationError(ValueError):
    pass


@dataclass(frozen=True)
class KernelLibrary:
    hidden: int
    group_size: int
    key_heads: int
    value_heads: int
    head_dimension: int
    attn_query_heads: int
    attn_kv_heads: int
    attn_head_dimension: int
    moe_experts: int
    moe_active_experts: int
    moe_expert_dimension: int
    vocabulary: int
    mlx_steel_enabled: bool
    source: str


MLX_STEEL_UPSTREAM_REVISION = "mlx-v0.32.0"
MLX_STEEL_FILES = {
    "LICENSE": "ccfab7ccb2ea306f71531c8ca77bb55507606cd90768b1e32b8b52ab5b48cf01",
    "mlx/backend/metal/kernels/quantized.h":
        "4da52bf4ee688165a65b84c52a5f4e82efcae7f69e8c74d9ee3e00bef463c99f",
    "mlx/backend/metal/kernels/quantized_utils.h":
        "a12841b57d505f6cca81631901b845dbf76bc90dfc099e7f40c763b2cc838c51",
    "mlx/backend/metal/kernels/steel/defines.h":
        "b03cea6a7d5cfe814e6838d8536e2aacdcafd2744cb408e954fda336ce21759a",
    "mlx/backend/metal/kernels/steel/gemm/loader.h":
        "703cd05158c22625dd636b85f7219bb374c65eeaed03f7b770bf8ac49f2fb16d",
    "mlx/backend/metal/kernels/steel/gemm/mma.h":
        "57fb5f0aeddf2a760d5b9317fde2adcc562a2aba3942fda86d9a71c3d228a735",
    "mlx/backend/metal/kernels/steel/gemm/transforms.h":
        "aead73fa873184cf7d930fb9d20308f765c52c35a775a9a0131162cc332fa814",
    "mlx/backend/metal/kernels/steel/utils/integral_constant.h":
        "5a4c559c092132b54d3959054fd1fb01bd01adfa65f95d3a60a01b161f11f0e6",
    "mlx/backend/metal/kernels/steel/utils/type_traits.h":
        "4766c5a3809d3cf7740e63acc5703ef76b6b4e85d8baf16602c9d517a2fd426c",
    "mlx/backend/metal/kernels/steel/gemm/params.h":
        "6711e72e32310eafb088895d5a6d20fcfca814158f5b38eb72b6e69d9d773e43",
    "mlx/backend/metal/kernels/steel/utils.h":
        "d4c36298145d1c5617a94f07dbfabf6ac932afe2fd7ea0c9291f1fb385c5404a",
    "mlx/backend/metal/kernels/steel/gemm/gemm.h":
        "c84af31e2c57154f2a8a24fa7f9fe2449765cce9a66f3f035846ed3c03b6a8b0",
    "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h":
        "ed49c8b9c73957f0cbc9dea30f484cf84fb821aece68233d6d1967856200d5c7",
}
MLX_STEEL_KERNEL_NAME = (
    "tatara_mlx_steel_affine_gather_qmm_rhs_nt_"
    "bfloat16_t_gs_64_b_4_bm_16_bn_32_bk_32_wm_1_wn_2_aligned"
)
MLX_STEEL_DENSE_KERNEL_NAME = (
    "tatara_mlx_steel_affine_qmm_t_"
    "bfloat16_t_gs_64_b_4_bm_32_bn_32_bk_32_wm_2_wn_2"
)
MLX_STEEL_GDN_FUSED2_KERNEL_NAME = (
    "tatara_mlx_steel_gdn_fused2_affine_qmm"
)
MLX_STEEL_GDN_BM64_KERNEL_NAME = (
    "tatara_mlx_steel_gdn_bm64_affine_qmm"
)
MLX_STEEL_GDN_BM64_WM2_WN2_KERNEL_NAME = (
    "tatara_mlx_steel_gdn_bm64_wm2_wn2_affine_qmm"
)
MLX_STEEL_GDN_BM64_BK64_KERNEL_NAME = (
    "tatara_mlx_steel_gdn_bm64_bk64_affine_qmm"
)
MLX_STEEL_GDN_BM48_KERNEL_NAME = (
    "tatara_mlx_steel_gdn_bm48_affine_qmm"
)
MLX_STEEL_GDN_BM96_KERNEL_NAME = (
    "tatara_mlx_steel_gdn_bm96_affine_qmm"
)
MLX_STEEL_GDN_BM128_KERNEL_NAME = (
    "tatara_mlx_steel_gdn_bm128_affine_qmm"
)
MLX_STEEL_GDN_BN64_KERNEL_NAME = (
    "tatara_mlx_steel_gdn_bn64_affine_qmm"
)
MLX_STEEL_GDN_BK64_KERNEL_NAME = (
    "tatara_mlx_steel_gdn_bk64_affine_qmm"
)
MLX_STEEL_ROUTED_DOWN_KERNEL_NAME = (
    "tatara_mlx_steel_routed_down_partial"
)
MLX_STEEL_ROUTED_UPGATE_KERNEL_NAME = (
    "tatara_mlx_steel_routed_fused_upgate_swiglu"
)
MLX_STEEL_ROUTED_BM32_DOWN_KERNEL_NAME = (
    "tatara_mlx_steel_routed_bm32_down_partial"
)
MLX_STEEL_ROUTED_BM32_UPGATE_KERNEL_NAME = (
    "tatara_mlx_steel_routed_bm32_fused_upgate_swiglu"
)
MLX_STEEL_ROUTED_BK64_DOWN_KERNEL_NAME = (
    "tatara_mlx_steel_routed_bk64_down_partial"
)
MLX_STEEL_ROUTED_BK64_UPGATE_KERNEL_NAME = (
    "tatara_mlx_steel_routed_bk64_fused_upgate_swiglu"
)
MLX_STEEL_ROUTED_BN64_DOWN_KERNEL_NAME = (
    "tatara_mlx_steel_routed_bn64_down_partial"
)
MLX_STEEL_ROUTED_BN64_UPGATE_KERNEL_NAME = (
    "tatara_mlx_steel_routed_bn64_fused_upgate_swiglu"
)


def _verified_upstream_text(source_root: Path, relative_path: str) -> str:
    path = source_root / relative_path
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    expected = MLX_STEEL_FILES[relative_path]
    if actual != expected:
        raise KernelLibraryGenerationError(
            f"MLX Steel source identity mismatch for {relative_path}: "
            f"expected {expected}, got {actual}"
        )
    return payload.decode("utf-8")


def _source_lines(text: str, first: int, last: int) -> str:
    lines = text.splitlines()
    if first < 1 or last < first or last > len(lines):
        raise KernelLibraryGenerationError(
            f"Invalid MLX Steel source span {first}:{last}"
        )
    return "\n".join(lines[first - 1:last])


def build_mlx_steel_source(source_root: Path) -> str:
    """Assemble the sealed, symbol-minimal affine RHS QGEMM closure."""
    texts = {
        path: _verified_upstream_text(source_root, path)
        for path in MLX_STEEL_FILES
    }
    quantized = texts["mlx/backend/metal/kernels/quantized.h"]
    mma = texts["mlx/backend/metal/kernels/steel/gemm/mma.h"]
    if (
        "template <int bits, int wsize = 8>" not in
            _source_lines(quantized, 17, 26)
        or "struct QuantizedBlockLoader" not in
            _source_lines(quantized, 564, 691)
        or "METAL_FUNC void qmm_t_impl(" not in
            _source_lines(quantized, 1185, 1311)
        or "[[kernel]] void affine_gather_qmm_rhs(" not in
            _source_lines(quantized, 2399, 2593)
        or "struct BlockMMA" not in _source_lines(mma, 434, 566)
        or "struct GEMMParams" not in _source_lines(
            texts["mlx/backend/metal/kernels/steel/gemm/params.h"], 12, 31)
        or "void gemm(" not in _source_lines(
            texts[
                "mlx/backend/metal/kernels/steel/gemm/kernels/"
                "steel_gemm_fused.h"
            ], 29, 29)
    ):
        raise KernelLibraryGenerationError("MLX Steel source span contract changed")

    dense_impl = _source_lines(quantized, 1185, 1311)
    dense_replacements = (
        (
            "    const int BM = 32,\n"
            "    const int BK = 32,\n"
            "    const int BN = 32>",
            "    const int BM = 32,\n"
            "    const int BK = 32,\n"
            "    const int BN = 32,\n"
            "    const int WM = 2,\n"
            "    const int WN = 2,\n"
            "    const bool trim_edge_barriers = false>",
        ),
        (
            "  constexpr int WM = 2;\n"
            "  constexpr int WN = 2;\n",
            "",
        ),
        (
            "    const constant int& K_eff,\n"
            "    uint3 tid [[threadgroup_position_in_grid]],",
            "    const constant int& K_eff,\n"
            "    const int output_stride,\n"
            "    uint3 tid [[threadgroup_position_in_grid]],",
        ),
        (
            "  y += y_row * static_cast<int64_t>(N) + y_col;",
            "  y += y_row * static_cast<int64_t>(output_stride) + y_col;",
        ),
        (
            "    mma_op.store_result_safe(y, N, short2(num_outs, num_els));",
            "    mma_op.store_result_safe("
            "y, output_stride, short2(num_outs, num_els));",
        ),
        (
            "    mma_op.store_result(y, N);",
            "    mma_op.store_result(y, output_stride);",
        ),
    )
    for old, new in dense_replacements:
        if dense_impl.count(old) != 1:
            raise KernelLibraryGenerationError(
                "MLX Steel dense output-stride adaptation contract changed"
            )
        dense_impl = dense_impl.replace(old, new)
    loop_barrier = (
        "      for (int k = 0; k < K_eff; k += BK) {\n"
        "        threadgroup_barrier(mem_flags::mem_threadgroup);"
    )
    trimmed_loop_barrier = (
        "      for (int k = 0; k < K_eff; k += BK) {\n"
        "        if (!trim_edge_barriers || k != 0) {\n"
        "          threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "        }"
    )
    if dense_impl.count(loop_barrier) != 4:
        raise KernelLibraryGenerationError(
            "MLX Steel dense reduction-loop barrier contract changed"
        )
    dense_impl = dense_impl.replace(
        loop_barrier, trimmed_loop_barrier
    )
    final_barrier = (
        "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "  if (num_els < BM || num_outs < BN) {"
    )
    trimmed_final_barrier = (
        "  if (!trim_edge_barriers) {\n"
        "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
        "  }\n"
        "  if (num_els < BM || num_outs < BN) {"
    )
    if dense_impl.count(final_barrier) != 1:
        raise KernelLibraryGenerationError(
            "MLX Steel dense final barrier contract changed"
        )
    dense_impl = dense_impl.replace(
        final_barrier, trimmed_final_barrier
    )

    license_comment = "\n".join(
        f"// {line}" if line else "//"
        for line in texts["LICENSE"].splitlines()
    )
    pieces = [
        "// Probe-only closure adapted from Apple MLX v0.32.0.",
        "// Upstream paths and exact identities are frozen by the generator.",
        license_comment,
        "#include <metal_simdgroup>",
        "#include <metal_simdgroup_matrix>",
        "#include <metal_stdlib>",
        "using namespace metal;",
        "typedef bfloat bfloat16_t;",
        _source_lines(
            texts["mlx/backend/metal/kernels/steel/defines.h"], 5, 7
        ),
        "#pragma METAL internals : enable",
        "namespace metal {",
        _source_lines(
            texts[
                "mlx/backend/metal/kernels/steel/utils/type_traits.h"
            ],
            30,
            51,
        ),
        "} // namespace metal",
        "namespace mlx { namespace steel {",
        _source_lines(
            texts[
                "mlx/backend/metal/kernels/steel/utils/integral_constant.h"
            ],
            17,
            44,
        ),
        "} } // namespace mlx::steel",
        "#pragma METAL internals : disable",
        "namespace mlx { namespace steel {",
        _source_lines(
            texts[
                "mlx/backend/metal/kernels/steel/gemm/transforms.h"
            ],
            14,
            23,
        ),
        "} } // namespace mlx::steel",
        _source_lines(
            texts["mlx/backend/metal/kernels/steel/gemm/loader.h"], 10, 137
        ),
        _source_lines(mma, 13, 422),
        _source_lines(mma, 434, 566),
        _source_lines(mma, 568, 584),
        _source_lines(mma, 585, 742),
        "} // namespace steel",
        "} // namespace mlx",
        _source_lines(
            texts["mlx/backend/metal/kernels/quantized_utils.h"], 3, 90
        ),
        "#define MLX_MTL_CONST static constant constexpr const",
        "MLX_MTL_CONST int SIMD_SIZE = 32;",
        "constant bool align_M = true;",
        "constant bool align_N = true;",
        "constant bool align_K = true;",
        _source_lines(quantized, 17, 26),
        _source_lines(quantized, 483, 691),
        dense_impl,
        "[[kernel]] void " + MLX_STEEL_DENSE_KERNEL_NAME + "(",
        "    const device uint32_t* w [[buffer(0)]],",
        "    const device bfloat16_t* scales [[buffer(1)]],",
        "    const device bfloat16_t* biases [[buffer(2)]],",
        "    const device bfloat16_t* x [[buffer(3)]],",
        "    device bfloat16_t* y [[buffer(4)]],",
        "    const constant int& K [[buffer(5)]],",
        "    const constant int& N [[buffer(6)]],",
        "    const constant int& M [[buffer(7)]],",
        "    const constant int& output_stride [[buffer(8)]],",
        "    uint3 tid [[threadgroup_position_in_grid]],",
        "    uint lid [[thread_index_in_threadgroup]],",
        "    uint simd_gid [[simdgroup_index_in_threadgroup]],",
        "    uint simd_lid [[thread_index_in_simdgroup]]) {",
        "  constexpr int BM = 32;",
        "  constexpr int BN = 32;",
        "  constexpr int BK = 32;",
        "  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);",
        "  threadgroup bfloat16_t Xs[BM * BK_padded];",
        "  threadgroup bfloat16_t Ws[BN * BK_padded];",
        "  qmm_t_impl<bfloat16_t, 64, 4, true, BM, BK, BN>(",
        "      w, scales, biases, x, y, Xs, Ws, K, N, M, K,",
        "      output_stride, tid, lid, simd_gid, simd_lid);",
        "}",
        "// 16-row dense tile for the M<=16 verify band (WM=1).",
        "[[kernel]] void tatara_mlx_steel_dense_bm16_affine_qmm(",
        "    const device uint32_t* w [[buffer(0)]],",
        "    const device bfloat16_t* scales [[buffer(1)]],",
        "    const device bfloat16_t* biases [[buffer(2)]],",
        "    const device bfloat16_t* x [[buffer(3)]],",
        "    device bfloat16_t* y [[buffer(4)]],",
        "    const constant int& K [[buffer(5)]],",
        "    const constant int& N [[buffer(6)]],",
        "    const constant int& M [[buffer(7)]],",
        "    const constant int& output_stride [[buffer(8)]],",
        "    uint3 tid [[threadgroup_position_in_grid]],",
        "    uint lid [[thread_index_in_threadgroup]],",
        "    uint simd_gid [[simdgroup_index_in_threadgroup]],",
        "    uint simd_lid [[thread_index_in_simdgroup]]) {",
        "  constexpr int BM = 16;",
        "  constexpr int BN = 32;",
        "  constexpr int BK = 32;",
        "  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);",
        "  threadgroup bfloat16_t Xs[BM * BK_padded];",
        "  threadgroup bfloat16_t Ws[BN * BK_padded];",
        "  qmm_t_impl<bfloat16_t, 64, 4, true, BM, BK, BN, 1, 2, false>(",
        "      w, scales, biases, x, y, Xs, Ws, K, N, M, K,",
        "      output_stride, tid, lid, simd_gid, simd_lid);",
        "}",
        "[[kernel]] void " + MLX_STEEL_GDN_BM64_KERNEL_NAME + "(",
        "    const device uint32_t* w [[buffer(0)]],",
        "    const device bfloat16_t* scales [[buffer(1)]],",
        "    const device bfloat16_t* biases [[buffer(2)]],",
        "    const device bfloat16_t* x [[buffer(3)]],",
        "    device bfloat16_t* y [[buffer(4)]],",
        "    const constant int& K [[buffer(5)]],",
        "    const constant int& N [[buffer(6)]],",
        "    const constant int& M [[buffer(7)]],",
        "    const constant int& output_stride [[buffer(8)]],",
        "    uint3 tid [[threadgroup_position_in_grid]],",
        "    uint lid [[thread_index_in_threadgroup]],",
        "    uint simd_gid [[simdgroup_index_in_threadgroup]],",
        "    uint simd_lid [[thread_index_in_simdgroup]],",
        "    uint simd_width [[threads_per_simdgroup]],",
        "    uint3 threadgroup_shape [[threads_per_threadgroup]]) {",
        "  constexpr int BM = 64;",
        "  constexpr int BN = 32;",
        "  constexpr int BK = 32;",
        "  constexpr int WM = 4;",
        "  constexpr int WN = 1;",
        "  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);",
        "  static_assert(",
        "      BM == 64 && BN == 32 && BK == 32 &&",
        "      WM == 4 && WN == 1 &&",
        "      WM * WN * SIMD_SIZE == 128,",
        "      \"BM64 dense Steel geometry is invalid\");",
        "  if (N <= 0 || N % BN != 0 || M <= 0 || M % BM != 0 ||",
        "      K != int(kHiddenDimension) || K % BK != 0 ||",
        "      output_stride != int(kGdnProjectionRows) ||",
        "      simd_width != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.x != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.y != uint(WM) ||",
        "      threadgroup_shape.z != uint(WN)) {",
        "    return;",
        "  }",
        "  constexpr int kExpectedThreadgroupBytes =",
        "      (BM + BN) * BK_padded * sizeof(bfloat16_t);",
        "  static_assert(kExpectedThreadgroupBytes == 7680);",
        "  threadgroup bfloat16_t Xs[BM * BK_padded];",
        "  threadgroup bfloat16_t Ws[BN * BK_padded];",
        "  qmm_t_impl<",
        "      bfloat16_t, 64, 4, true, BM, BK, BN, WM, WN>(",
        "      w, scales, biases, x, y, Xs, Ws, K, N, M, K,",
        "      output_stride, tid, lid, simd_gid, simd_lid);",
        "}",
        "[[kernel]] void " + MLX_STEEL_GDN_BM64_WM2_WN2_KERNEL_NAME + "(",
        "    const device uint32_t* w [[buffer(0)]],",
        "    const device bfloat16_t* scales [[buffer(1)]],",
        "    const device bfloat16_t* biases [[buffer(2)]],",
        "    const device bfloat16_t* x [[buffer(3)]],",
        "    device bfloat16_t* y [[buffer(4)]],",
        "    const constant int& K [[buffer(5)]],",
        "    const constant int& N [[buffer(6)]],",
        "    const constant int& M [[buffer(7)]],",
        "    const constant int& output_stride [[buffer(8)]],",
        "    uint3 tid [[threadgroup_position_in_grid]],",
        "    uint lid [[thread_index_in_threadgroup]],",
        "    uint simd_gid [[simdgroup_index_in_threadgroup]],",
        "    uint simd_lid [[thread_index_in_simdgroup]],",
        "    uint simd_width [[threads_per_simdgroup]],",
        "    uint3 threadgroup_shape [[threads_per_threadgroup]]) {",
        "  constexpr int BM = 64;",
        "  constexpr int BN = 32;",
        "  constexpr int BK = 32;",
        "  constexpr int WM = 2;",
        "  constexpr int WN = 2;",
        "  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);",
        "  static_assert(",
        "      BM == 64 && BN == 32 && BK == 32 &&",
        "      WM == 2 && WN == 2 &&",
        "      WM * WN * SIMD_SIZE == 128,",
        "      \"BM64/WM2/WN2 dense Steel geometry is invalid\");",
        "  if (N <= 0 || N % BN != 0 || M <= 0 || M % BM != 0 ||",
        "      K != int(kHiddenDimension) || K % BK != 0 ||",
        "      output_stride != int(kGdnProjectionRows) ||",
        "      simd_width != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.x != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.y != uint(WM) ||",
        "      threadgroup_shape.z != uint(WN)) {",
        "    return;",
        "  }",
        "  constexpr int kExpectedThreadgroupBytes =",
        "      (BM + BN) * BK_padded * sizeof(bfloat16_t);",
        "  static_assert(kExpectedThreadgroupBytes == 7680);",
        "  threadgroup bfloat16_t Xs[BM * BK_padded];",
        "  threadgroup bfloat16_t Ws[BN * BK_padded];",
        "  qmm_t_impl<",
        "      bfloat16_t, 64, 4, true, BM, BK, BN, WM, WN>(",
        "      w, scales, biases, x, y, Xs, Ws, K, N, M, K,",
        "      output_stride, tid, lid, simd_gid, simd_lid);",
        "}",
        "[[kernel]] void " + MLX_STEEL_GDN_BM64_BK64_KERNEL_NAME + "(",
        "    const device uint32_t* w [[buffer(0)]],",
        "    const device bfloat16_t* scales [[buffer(1)]],",
        "    const device bfloat16_t* biases [[buffer(2)]],",
        "    const device bfloat16_t* x [[buffer(3)]],",
        "    device bfloat16_t* y [[buffer(4)]],",
        "    const constant int& K [[buffer(5)]],",
        "    const constant int& N [[buffer(6)]],",
        "    const constant int& M [[buffer(7)]],",
        "    const constant int& output_stride [[buffer(8)]],",
        "    uint3 tid [[threadgroup_position_in_grid]],",
        "    uint lid [[thread_index_in_threadgroup]],",
        "    uint simd_gid [[simdgroup_index_in_threadgroup]],",
        "    uint simd_lid [[thread_index_in_simdgroup]],",
        "    uint simd_width [[threads_per_simdgroup]],",
        "    uint3 threadgroup_shape [[threads_per_threadgroup]]) {",
        "  constexpr int BM = 64;",
        "  constexpr int BN = 32;",
        "  constexpr int BK = 64;",
        "  constexpr int WM = 4;",
        "  constexpr int WN = 1;",
        "  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);",
        "  static_assert(",
        "      BM == 64 && BN == 32 && BK == 64 &&",
        "      WM == 4 && WN == 1 &&",
        "      WM * WN * SIMD_SIZE == 128 &&",
        "      BK == int(kQ4GroupSize),",
        "      \"BM64/BK64 dense Steel geometry is invalid\");",
        "  if (N <= 0 || N % BN != 0 || M <= 0 || M % BM != 0 ||",
        "      K != int(kHiddenDimension) || K % BK != 0 ||",
        "      output_stride != int(kGdnProjectionRows) ||",
        "      simd_width != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.x != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.y != uint(WM) ||",
        "      threadgroup_shape.z != uint(WN)) {",
        "    return;",
        "  }",
        "  constexpr int kExpectedThreadgroupBytes =",
        "      (BM + BN) * BK_padded * sizeof(bfloat16_t);",
        "  static_assert(kExpectedThreadgroupBytes == 13824);",
        "  threadgroup bfloat16_t Xs[BM * BK_padded];",
        "  threadgroup bfloat16_t Ws[BN * BK_padded];",
        "  qmm_t_impl<",
        "      bfloat16_t, 64, 4, true, BM, BK, BN, WM, WN>(",
        "      w, scales, biases, x, y, Xs, Ws, K, N, M, K,",
        "      output_stride, tid, lid, simd_gid, simd_lid);",
        "}",
        "[[kernel]] void " + MLX_STEEL_GDN_BM48_KERNEL_NAME + "(",
        "    const device uint32_t* w [[buffer(0)]],",
        "    const device bfloat16_t* scales [[buffer(1)]],",
        "    const device bfloat16_t* biases [[buffer(2)]],",
        "    const device bfloat16_t* x [[buffer(3)]],",
        "    device bfloat16_t* y [[buffer(4)]],",
        "    const constant int& K [[buffer(5)]],",
        "    const constant int& N [[buffer(6)]],",
        "    const constant int& M [[buffer(7)]],",
        "    const constant int& output_stride [[buffer(8)]],",
        "    uint3 tid [[threadgroup_position_in_grid]],",
        "    uint lid [[thread_index_in_threadgroup]],",
        "    uint simd_gid [[simdgroup_index_in_threadgroup]],",
        "    uint simd_lid [[thread_index_in_simdgroup]],",
        "    uint simd_width [[threads_per_simdgroup]],",
        "    uint3 threadgroup_shape [[threads_per_threadgroup]]) {",
        "  constexpr int BM = 48;",
        "  constexpr int BN = 32;",
        "  constexpr int BK = 32;",
        "  constexpr int WM = 3;",
        "  constexpr int WN = 1;",
        "  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);",
        "  static_assert(",
        "      BM == 48 && BN == 32 && BK == 32 &&",
        "      WM == 3 && WN == 1 &&",
        "      WM * WN * SIMD_SIZE == 96,",
        "      \"BM48 dense Steel geometry is invalid\");",
        "  if (N <= 0 || N % BN != 0 || M <= 0 || M % BM != 0 ||",
        "      K != int(kHiddenDimension) || K % BK != 0 ||",
        "      output_stride != int(kGdnProjectionRows) ||",
        "      simd_width != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.x != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.y != uint(WM) ||",
        "      threadgroup_shape.z != uint(WN)) {",
        "    return;",
        "  }",
        "  constexpr int kExpectedThreadgroupBytes =",
        "      (BM + BN) * BK_padded * sizeof(bfloat16_t);",
        "  static_assert(kExpectedThreadgroupBytes == 6400);",
        "  threadgroup bfloat16_t Xs[BM * BK_padded];",
        "  threadgroup bfloat16_t Ws[BN * BK_padded];",
        "  qmm_t_impl<",
        "      bfloat16_t, 64, 4, true, BM, BK, BN, WM, WN>(",
        "      w, scales, biases, x, y, Xs, Ws, K, N, M, K,",
        "      output_stride, tid, lid, simd_gid, simd_lid);",
        "}",
        "[[kernel]] void " + MLX_STEEL_GDN_BM96_KERNEL_NAME + "(",
        "    const device uint32_t* w [[buffer(0)]],",
        "    const device bfloat16_t* scales [[buffer(1)]],",
        "    const device bfloat16_t* biases [[buffer(2)]],",
        "    const device bfloat16_t* x [[buffer(3)]],",
        "    device bfloat16_t* y [[buffer(4)]],",
        "    const constant int& K [[buffer(5)]],",
        "    const constant int& N [[buffer(6)]],",
        "    const constant int& M [[buffer(7)]],",
        "    const constant int& output_stride [[buffer(8)]],",
        "    uint3 tid [[threadgroup_position_in_grid]],",
        "    uint lid [[thread_index_in_threadgroup]],",
        "    uint simd_gid [[simdgroup_index_in_threadgroup]],",
        "    uint simd_lid [[thread_index_in_simdgroup]],",
        "    uint simd_width [[threads_per_simdgroup]],",
        "    uint3 threadgroup_shape [[threads_per_threadgroup]]) {",
        "  constexpr int BM = 96;",
        "  constexpr int BN = 32;",
        "  constexpr int BK = 32;",
        "  constexpr int WM = 6;",
        "  constexpr int WN = 1;",
        "  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);",
        "  static_assert(",
        "      BM == 96 && BN == 32 && BK == 32 &&",
        "      WM == 6 && WN == 1 &&",
        "      WM * WN * SIMD_SIZE == 192,",
        "      \"BM96 dense Steel geometry is invalid\");",
        "  if (N <= 0 || N % BN != 0 || M <= 0 || M % BM != 0 ||",
        "      K != int(kHiddenDimension) || K % BK != 0 ||",
        "      output_stride != int(kGdnProjectionRows) ||",
        "      simd_width != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.x != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.y != uint(WM) ||",
        "      threadgroup_shape.z != uint(WN)) {",
        "    return;",
        "  }",
        "  constexpr int kExpectedThreadgroupBytes =",
        "      (BM + BN) * BK_padded * sizeof(bfloat16_t);",
        "  static_assert(kExpectedThreadgroupBytes == 10240);",
        "  threadgroup bfloat16_t Xs[BM * BK_padded];",
        "  threadgroup bfloat16_t Ws[BN * BK_padded];",
        "  qmm_t_impl<",
        "      bfloat16_t, 64, 4, true, BM, BK, BN, WM, WN>(",
        "      w, scales, biases, x, y, Xs, Ws, K, N, M, K,",
        "      output_stride, tid, lid, simd_gid, simd_lid);",
        "}",
        "[[kernel]] void " + MLX_STEEL_GDN_BM128_KERNEL_NAME + "(",
        "    const device uint32_t* w [[buffer(0)]],",
        "    const device bfloat16_t* scales [[buffer(1)]],",
        "    const device bfloat16_t* biases [[buffer(2)]],",
        "    const device bfloat16_t* x [[buffer(3)]],",
        "    device bfloat16_t* y [[buffer(4)]],",
        "    const constant int& K [[buffer(5)]],",
        "    const constant int& N [[buffer(6)]],",
        "    const constant int& M [[buffer(7)]],",
        "    const constant int& output_stride [[buffer(8)]],",
        "    uint3 tid [[threadgroup_position_in_grid]],",
        "    uint lid [[thread_index_in_threadgroup]],",
        "    uint simd_gid [[simdgroup_index_in_threadgroup]],",
        "    uint simd_lid [[thread_index_in_simdgroup]],",
        "    uint simd_width [[threads_per_simdgroup]],",
        "    uint3 threadgroup_shape [[threads_per_threadgroup]]) {",
        "  constexpr int BM = 128;",
        "  constexpr int BN = 32;",
        "  constexpr int BK = 32;",
        "  constexpr int WM = 8;",
        "  constexpr int WN = 1;",
        "  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);",
        "  static_assert(",
        "      BM == 128 && BN == 32 && BK == 32 &&",
        "      WM == 8 && WN == 1 &&",
        "      WM * WN * SIMD_SIZE == 256,",
        "      \"BM128 dense Steel geometry is invalid\");",
        "  if (N <= 0 || N % BN != 0 || M <= 0 || M % BM != 0 ||",
        "      K != int(kHiddenDimension) || K % BK != 0 ||",
        "      output_stride != int(kGdnProjectionRows) ||",
        "      simd_width != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.x != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.y != uint(WM) ||",
        "      threadgroup_shape.z != uint(WN)) {",
        "    return;",
        "  }",
        "  constexpr int kExpectedThreadgroupBytes =",
        "      (BM + BN) * BK_padded * sizeof(bfloat16_t);",
        "  static_assert(kExpectedThreadgroupBytes == 12800);",
        "  threadgroup bfloat16_t Xs[BM * BK_padded];",
        "  threadgroup bfloat16_t Ws[BN * BK_padded];",
        "  qmm_t_impl<",
        "      bfloat16_t, 64, 4, true, BM, BK, BN, WM, WN>(",
        "      w, scales, biases, x, y, Xs, Ws, K, N, M, K,",
        "      output_stride, tid, lid, simd_gid, simd_lid);",
        "}",
        "[[kernel]] void " + MLX_STEEL_GDN_BK64_KERNEL_NAME + "(",
        "    const device uint32_t* w [[buffer(0)]],",
        "    const device bfloat16_t* scales [[buffer(1)]],",
        "    const device bfloat16_t* biases [[buffer(2)]],",
        "    const device bfloat16_t* x [[buffer(3)]],",
        "    device bfloat16_t* y [[buffer(4)]],",
        "    const constant int& K [[buffer(5)]],",
        "    const constant int& N [[buffer(6)]],",
        "    const constant int& M [[buffer(7)]],",
        "    const constant int& output_stride [[buffer(8)]],",
        "    uint3 tid [[threadgroup_position_in_grid]],",
        "    uint lid [[thread_index_in_threadgroup]],",
        "    uint simd_gid [[simdgroup_index_in_threadgroup]],",
        "    uint simd_lid [[thread_index_in_simdgroup]],",
        "    uint simd_width [[threads_per_simdgroup]],",
        "    uint3 threadgroup_shape [[threads_per_threadgroup]]) {",
        "  constexpr int BM = 32;",
        "  constexpr int BN = 32;",
        "  constexpr int BK = 64;",
        "  constexpr int WM = 2;",
        "  constexpr int WN = 2;",
        "  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);",
        "  static_assert(",
        "      BM == 32 && BN == 32 && BK == 64 &&",
        "      WM == 2 && WN == 2 &&",
        "      WM * WN * SIMD_SIZE == 128 &&",
        "      BK == int(kQ4GroupSize),",
        "      \"BK64 dense Steel geometry is invalid\");",
        "  if (N <= 0 || N % BN != 0 || M <= 0 ||",
        "      K != int(kHiddenDimension) || K % BK != 0 ||",
        "      output_stride != int(kGdnProjectionRows) ||",
        "      simd_width != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.x != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.y != uint(WM) ||",
        "      threadgroup_shape.z != uint(WN)) {",
        "    return;",
        "  }",
        "  constexpr int kExpectedThreadgroupBytes =",
        "      (BM + BN) * BK_padded * sizeof(bfloat16_t);",
        "  static_assert(kExpectedThreadgroupBytes == 9216);",
        "  threadgroup bfloat16_t Xs[BM * BK_padded];",
        "  threadgroup bfloat16_t Ws[BN * BK_padded];",
        "  qmm_t_impl<",
        "      bfloat16_t, 64, 4, true, BM, BK, BN, WM, WN, true>(",
        "      w, scales, biases, x, y, Xs, Ws, K, N, M, K,",
        "      output_stride, tid, lid, simd_gid, simd_lid);",
        "}",
        "[[kernel]] void " + MLX_STEEL_GDN_BN64_KERNEL_NAME + "(",
        "    const device uint32_t* w [[buffer(0)]],",
        "    const device bfloat16_t* scales [[buffer(1)]],",
        "    const device bfloat16_t* biases [[buffer(2)]],",
        "    const device bfloat16_t* x [[buffer(3)]],",
        "    device bfloat16_t* y [[buffer(4)]],",
        "    const constant int& K [[buffer(5)]],",
        "    const constant int& N [[buffer(6)]],",
        "    const constant int& M [[buffer(7)]],",
        "    const constant int& output_stride [[buffer(8)]],",
        "    uint3 tid [[threadgroup_position_in_grid]],",
        "    uint lid [[thread_index_in_threadgroup]],",
        "    uint simd_gid [[simdgroup_index_in_threadgroup]],",
        "    uint simd_lid [[thread_index_in_simdgroup]],",
        "    uint simd_width [[threads_per_simdgroup]],",
        "    uint3 threadgroup_shape [[threads_per_threadgroup]]) {",
        "  constexpr int BM = 32;",
        "  constexpr int BN = 64;",
        "  constexpr int BK = 32;",
        "  constexpr int WM = 2;",
        "  constexpr int WN = 4;",
        "  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);",
        "  static_assert(",
        "      BM == 32 && BN == 64 && BK == 32 &&",
        "      WM == 2 && WN == 4 &&",
        "      WM * WN * SIMD_SIZE == 256,",
        "      \"BN64 dense Steel geometry is invalid\");",
        "  if (N <= 0 || N % BN != 0 || M <= 0 ||",
        "      K != int(kHiddenDimension) ||",
        "      output_stride != int(kGdnProjectionRows) ||",
        "      simd_width != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.x != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.y != uint(WM) ||",
        "      threadgroup_shape.z != uint(WN)) {",
        "    return;",
        "  }",
        "  constexpr int kExpectedThreadgroupBytes =",
        "      (BM + BN) * BK_padded * sizeof(bfloat16_t);",
        "  static_assert(kExpectedThreadgroupBytes == 7680);",
        "  threadgroup bfloat16_t Xs[BM * BK_padded];",
        "  threadgroup bfloat16_t Ws[BN * BK_padded];",
        "  qmm_t_impl<",
        "      bfloat16_t, 64, 4, true, BM, BK, BN, WM, WN>(",
        "      w, scales, biases, x, y, Xs, Ws, K, N, M, K,",
        "      output_stride, tid, lid, simd_gid, simd_lid);",
        "}",
        "[[kernel]] void " + MLX_STEEL_GDN_FUSED2_KERNEL_NAME + "(",
        "    const device uint32_t* qkv_w [[buffer(0)]],",
        "    const device bfloat16_t* qkv_scales [[buffer(1)]],",
        "    const device bfloat16_t* qkv_biases [[buffer(2)]],",
        "    const device uint32_t* z_w [[buffer(3)]],",
        "    const device bfloat16_t* z_scales [[buffer(4)]],",
        "    const device bfloat16_t* z_biases [[buffer(5)]],",
        "    const device uint32_t* b_w [[buffer(6)]],",
        "    const device bfloat16_t* b_scales [[buffer(7)]],",
        "    const device bfloat16_t* b_biases [[buffer(8)]],",
        "    const device uint32_t* a_w [[buffer(9)]],",
        "    const device bfloat16_t* a_scales [[buffer(10)]],",
        "    const device bfloat16_t* a_biases [[buffer(11)]],",
        "    const device bfloat16_t* x [[buffer(12)]],",
        "    device bfloat16_t* y [[buffer(13)]],",
        "    const constant uint& M [[buffer(14)]],",
        "    const constant uint& K [[buffer(15)]],",
        "    const constant uint& output_stride [[buffer(16)]],",
        "    uint3 tid [[threadgroup_position_in_grid]],",
        "    uint lid [[thread_index_in_threadgroup]],",
        "    uint simd_gid [[simdgroup_index_in_threadgroup]],",
        "    uint simd_lid [[thread_index_in_simdgroup]],",
        "    uint simd_width [[threads_per_simdgroup]],",
        "    uint3 threadgroup_shape [[threads_per_threadgroup]]) {",
        "  constexpr int BM = 32;",
        "  constexpr int BN = 32;",
        "  constexpr int BK = 32;",
        "  constexpr int WM = 2;",
        "  constexpr int WN = 2;",
        "  constexpr int BK_padded = BK + 16 / sizeof(bfloat16_t);",
        "  constexpr uint pair_columns = 2u * uint(BN);",
        "  static_assert(",
        "      BM == 32 && BN == 32 && BK == 32 &&",
        "      WM * WN * SIMD_SIZE == 128 &&",
        "      kHiddenDimension % BK == 0u &&",
        "      kGdnQkvRows % pair_columns == 0u &&",
        "      kGdnZRows % pair_columns == 0u &&",
        "      kGdnValueHeads == uint(BN) &&",
        "      kGdnBRowOffset == kGdnQkvRows + kGdnZRows &&",
        "      kGdnARowOffset == kGdnBRowOffset + kGdnValueHeads &&",
        "      kGdnProjectionRows ==",
        "          kGdnARowOffset + kGdnValueHeads &&",
        "      kGdnProjectionRows % pair_columns == 0u,",
        "      \"fused-pair GDN projection geometry is invalid\");",
        "  const uint row_begin = tid.y * uint(BM);",
        "  const uint pair_begin = tid.x * pair_columns;",
        "  if (M == 0u || K != kHiddenDimension ||",
        "      output_stride != kGdnProjectionRows ||",
        "      tid.z != 0u || row_begin >= M ||",
        "      pair_begin >= kGdnProjectionRows ||",
        "      simd_width != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.x != uint(SIMD_SIZE) ||",
        "      threadgroup_shape.y != uint(WM) ||",
        "      threadgroup_shape.z != uint(WN)) {",
        "    return;",
        "  }",
        "  (void)lid;",
        "",
        "  const device uint32_t* first_w = qkv_w;",
        "  const device bfloat16_t* first_scales = qkv_scales;",
        "  const device bfloat16_t* first_biases = qkv_biases;",
        "  const device uint32_t* second_w = qkv_w;",
        "  const device bfloat16_t* second_scales = qkv_scales;",
        "  const device bfloat16_t* second_biases = qkv_biases;",
        "  uint first_row = pair_begin;",
        "  uint second_row = pair_begin + uint(BN);",
        "  if (pair_begin >= kGdnBRowOffset) {",
        "    first_w = b_w;",
        "    first_scales = b_scales;",
        "    first_biases = b_biases;",
        "    second_w = a_w;",
        "    second_scales = a_scales;",
        "    second_biases = a_biases;",
        "    first_row = 0u;",
        "    second_row = 0u;",
        "  } else if (pair_begin >= kGdnQkvRows) {",
        "    first_w = z_w;",
        "    first_scales = z_scales;",
        "    first_biases = z_biases;",
        "    second_w = z_w;",
        "    second_scales = z_scales;",
        "    second_biases = z_biases;",
        "    first_row = pair_begin - kGdnQkvRows;",
        "    second_row = first_row + uint(BN);",
        "  }",
        "",
        "  const uint packed_row_bytes = kHiddenDimension / 2u;",
        "  const uint parameter_row_stride =",
        "      kHiddenDimension / kQ4GroupSize;",
        "  const device uint8_t* first_bytes =",
        "      reinterpret_cast<device const uint8_t*>(first_w) +",
        "      ulong(first_row) * ulong(packed_row_bytes);",
        "  const device uint8_t* second_bytes =",
        "      reinterpret_cast<device const uint8_t*>(second_w) +",
        "      ulong(second_row) * ulong(packed_row_bytes);",
        "  first_scales += ulong(first_row) *",
        "      ulong(parameter_row_stride);",
        "  first_biases += ulong(first_row) *",
        "      ulong(parameter_row_stride);",
        "  second_scales += ulong(second_row) *",
        "      ulong(parameter_row_stride);",
        "  second_biases += ulong(second_row) *",
        "      ulong(parameter_row_stride);",
        "  x += ulong(row_begin) * ulong(K);",
        "  y += ulong(row_begin) * ulong(output_stride) +",
        "      ulong(pair_begin);",
        "",
        "  using mma_t = mlx::steel::BlockMMA<",
        "      bfloat16_t, bfloat16_t, BM, BN, BK, WM, WN,",
        "      false, true, BK_padded, BK_padded>;",
        "  using activation_loader_t = mlx::steel::BlockLoader<",
        "      bfloat16_t, BM, BK, BK_padded, 1,",
        "      WM * WN * SIMD_SIZE>;",
        "  using weight_loader_t = QuantizedBlockLoader<",
        "      bfloat16_t, BN, BK, BK_padded, 1,",
        "      WM * WN * SIMD_SIZE, kQ4GroupSize, 4>;",
        "  threadgroup bfloat16_t Xs[BM * BK_padded];",
        "  threadgroup bfloat16_t Ws[BN * BK_padded];",
        "  thread activation_loader_t activation_loader(",
        "      x, int(K), Xs, simd_gid, simd_lid);",
        "  thread weight_loader_t first_loader(",
        "      first_bytes, first_scales, first_biases, int(K),",
        "      Ws, simd_gid, simd_lid);",
        "  thread weight_loader_t second_loader(",
        "      second_bytes, second_scales, second_biases, int(K),",
        "      Ws, simd_gid, simd_lid);",
        "  thread mma_t first_mma(simd_gid, simd_lid);",
        "  thread mma_t second_mma(simd_gid, simd_lid);",
        "  const short live_rows =",
        "      short(min(uint(BM), M - row_begin));",
        "",
        "  for (uint reduction_tile = 0u;",
        "       reduction_tile < K; reduction_tile += uint(BK)) {",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    if (live_rows == BM) {",
        "      activation_loader.load_unsafe();",
        "    } else {",
        "      activation_loader.load_safe(short2(BK, live_rows));",
        "    }",
        "    first_loader.load_unsafe();",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    first_mma.mma(Xs, Ws);",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    second_loader.load_unsafe();",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    second_mma.mma(Xs, Ws);",
        "    activation_loader.next();",
        "    first_loader.next();",
        "    second_loader.next();",
        "  }",
        "",
        "  threadgroup_barrier(mem_flags::mem_threadgroup);",
        "  if (live_rows == BM) {",
        "    first_mma.store_result(y, int(output_stride));",
        "    second_mma.store_result(",
        "        y + BN, int(output_stride));",
        "  } else {",
        "    first_mma.store_result_safe(",
        "        y, int(output_stride), short2(BN, live_rows));",
        "    second_mma.store_result_safe(",
        "        y + BN, int(output_stride),",
        "        short2(BN, live_rows));",
        "  }",
        "}",
        _source_lines(quantized, 2399, 2593),
        "template [[host_name(\"" + MLX_STEEL_KERNEL_NAME + "\")]] [[kernel]]",
        "decltype(affine_gather_qmm_rhs<"
        "bfloat16_t, 64, 4, 16, 32, 32, 1, 2, true>)",
        "affine_gather_qmm_rhs<"
        "bfloat16_t, 64, 4, 16, 32, 32, 1, 2, true>;",
        "inline void native_routed_qgemm_bk64_stage_position_rows(",
        "    device const bfloat* input,",
        "    threadgroup const uint* staged_routes,",
        "    uint reduction_tile,",
        "    threadgroup bfloat* staged_input,",
        "    uint thread_index) {",
        "  constexpr uint BK = kQ4GroupSize;",
        "  constexpr uint BK_padded =",
        "      BK + kNativeRoutedQgemmR2StageStride -",
        "      kNativeRoutedQgemmR1ReductionColumns;",
        "  constexpr uint chunks_per_row = BK / 8u;",
        "  constexpr uint chunk_count =",
        "      kNativeRoutedQgemmR1TileRows * chunks_per_row;",
        "  for (uint linear = thread_index; linear < chunk_count;",
        "       linear += kNativeRoutedQgemmR1Threads) {",
        "    const uint row = linear / chunks_per_row;",
        "    const uint reduction = (linear % chunks_per_row) * 8u;",
        "    threadgroup bfloat* destination =",
        "        staged_input + row * BK_padded + reduction;",
        "    const uint packed = staged_routes[row];",
        "    if (packed == kNativeRoutedQgemmR2InvalidRoute) {",
        "      native_routed_qgemm_r2_zero_bfloat8(destination);",
        "    } else {",
        "      native_routed_qgemm_r2_copy_bfloat8(",
        "          destination,",
        "          input +",
        "              ulong(packed >> kPrefillPackedSlotBits) *",
        "                  ulong(kHiddenDimension) +",
        "              ulong(reduction_tile + reduction));",
        "    }",
        "  }",
        "}",
        "inline void native_routed_qgemm_bk64_stage_hidden_rows(",
        "    device const bfloat* hidden,",
        "    threadgroup const uint* staged_routes,",
        "    uint reduction_tile,",
        "    threadgroup bfloat* staged_input,",
        "    uint thread_index) {",
        "  constexpr uint BK = kQ4GroupSize;",
        "  constexpr uint BK_padded =",
        "      BK + kNativeRoutedQgemmR2StageStride -",
        "      kNativeRoutedQgemmR1ReductionColumns;",
        "  constexpr uint chunks_per_row = BK / 8u;",
        "  constexpr uint chunk_count =",
        "      kNativeRoutedQgemmR1TileRows * chunks_per_row;",
        "  for (uint linear = thread_index; linear < chunk_count;",
        "       linear += kNativeRoutedQgemmR1Threads) {",
        "    const uint row = linear / chunks_per_row;",
        "    const uint reduction = (linear % chunks_per_row) * 8u;",
        "    threadgroup bfloat* destination =",
        "        staged_input + row * BK_padded + reduction;",
        "    const uint packed = staged_routes[row];",
        "    if (packed == kNativeRoutedQgemmR2InvalidRoute) {",
        "      native_routed_qgemm_r2_zero_bfloat8(destination);",
        "    } else {",
        "      const uint position =",
        "          packed >> kPrefillPackedSlotBits;",
        "      const uint slot =",
        "          packed & ((1u << kPrefillPackedSlotBits) - 1u);",
        "      native_routed_qgemm_r2_copy_bfloat8(",
        "          destination,",
        "          hidden +",
        "              (ulong(position) * ulong(kMoeSlotCount) +",
        "               ulong(slot)) * ulong(kMoeExpertDimension) +",
        "              ulong(reduction_tile + reduction));",
        "    }",
        "  }",
        "}",
        "[[kernel]] void " + MLX_STEEL_ROUTED_UPGATE_KERNEL_NAME + "(",
        "    device const bfloat* input [[buffer(0)]],",
        "    device const uint* route_list [[buffer(1)]],",
        "    device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]],",
        "    device const uint* gate_words [[buffer(3)]],",
        "    device const bfloat* gate_scales [[buffer(4)]],",
        "    device const bfloat* gate_biases [[buffer(5)]],",
        "    device const uint* up_words [[buffer(6)]],",
        "    device const bfloat* up_scales [[buffer(7)]],",
        "    device const bfloat* up_biases [[buffer(8)]],",
        "    device bfloat* hidden [[buffer(9)]],",
        "    constant uint& task_count [[buffer(10)]],",
        "    constant uint& route_list_expert_stride [[buffer(11)]],",
        "    constant uint& route_list_capacity_per_expert [[buffer(12)]],",
        "    constant ulong& route_list_total_extent [[buffer(13)]],",
        "    constant uint& input_rows [[buffer(14)]],",
        "    device const uint* shared_gate_words [[buffer(15)]],",
        "    device const bfloat* shared_gate_scales [[buffer(16)]],",
        "    device const bfloat* shared_gate_biases [[buffer(17)]],",
        "    device const uint* shared_up_words [[buffer(18)]],",
        "    device const bfloat* shared_up_scales [[buffer(19)]],",
        "    device const bfloat* shared_up_biases [[buffer(20)]],",
        "    constant uint& include_shared_expert [[buffer(21)]],",
        "    uint3 group [[threadgroup_position_in_grid]],",
        "    uint lane [[thread_index_in_simdgroup]],",
        "    uint simdgroup [[simdgroup_index_in_threadgroup]],",
        "    uint simdgroup_width [[threads_per_simdgroup]],",
        "    uint thread_index [[thread_index_in_threadgroup]],",
        "    uint3 threadgroup_shape [[threads_per_threadgroup]]) {",
        "  constexpr short BM = 16;",
        "  constexpr short BN = 32;",
        "  constexpr short BK = 32;",
        "  constexpr short WM = 1;",
        "  constexpr short WN = 2;",
        "  constexpr short BK_padded = BK + 16 / sizeof(bfloat);",
        "  static_assert(",
        "      BM == kNativeRoutedQgemmR1TileRows &&",
        "      BN == kNativeRoutedQgemmR1TileColumns &&",
        "      BK == kNativeRoutedQgemmR1ReductionColumns &&",
        "      BK_padded == kNativeRoutedQgemmR2StageStride &&",
        "      WM * WN * SIMD_SIZE == kNativeRoutedQgemmR1Threads &&",
        "      kHiddenDimension % BK == 0 &&",
        "      kMoeExpertDimension % BN == 0,",
        "      \"Steel routed-upgate tile contract is invalid\");",
        "  if (!native_routed_qgemm_r1_dispatch_valid(",
        "          task_count, group, simdgroup, simdgroup_width,",
        "          threadgroup_shape)) {",
        "    return;",
        "  }",
        "  const NativeRoutedQgemmR1Task task = tasks[group.x];",
        "  if (!native_routed_qgemm_r2_task_valid(",
        "          task, route_list_expert_stride,",
        "          route_list_capacity_per_expert,",
        "          route_list_total_extent, include_shared_expert) ||",
        "      group.y >= kMoeExpertDimension / BN) {",
        "    return;",
        "  }",
        "",
        "  threadgroup uint staged_routes[BM];",
        "  threadgroup bfloat Xs[BM * BK_padded];",
        "  threadgroup bfloat gate_Ws[BN * BK_padded];",
        "  threadgroup bfloat up_Ws[BN * BK_padded];",
        "  native_routed_qgemm_r2_stage_routes(",
        "      route_list, task, input_rows, include_shared_expert,",
        "      staged_routes, thread_index);",
        "  threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "  const bool shared = task.expert_index == kMoeExperts;",
        "  device const uint* selected_gate_words =",
        "      shared ? shared_gate_words : gate_words;",
        "  device const bfloat* selected_gate_scales =",
        "      shared ? shared_gate_scales : gate_scales;",
        "  device const bfloat* selected_gate_biases =",
        "      shared ? shared_gate_biases : gate_biases;",
        "  device const uint* selected_up_words =",
        "      shared ? shared_up_words : up_words;",
        "  device const bfloat* selected_up_scales =",
        "      shared ? shared_up_scales : up_scales;",
        "  device const bfloat* selected_up_biases =",
        "      shared ? shared_up_biases : up_biases;",
        "  const ulong expert_row_begin =",
        "      shared ? 0ul",
        "             : ulong(task.expert_index) *",
        "                   ulong(kMoeExpertDimension);",
        "  const uint output_tile = group.y * BN;",
        "  const ulong matrix_row = expert_row_begin + output_tile;",
        "  const ulong packed_row_bytes =",
        "      ulong(kHiddenDimension / 2u);",
        "  const ulong parameter_row_stride =",
        "      ulong(kHiddenDimension / kQ4GroupSize);",
        "  device const uint8_t* gate_bytes =",
        "      reinterpret_cast<device const uint8_t*>(",
        "          selected_gate_words) +",
        "      matrix_row * packed_row_bytes;",
        "  device const uint8_t* up_bytes =",
        "      reinterpret_cast<device const uint8_t*>(",
        "          selected_up_words) +",
        "      matrix_row * packed_row_bytes;",
        "",
        "  using mma_t = mlx::steel::BlockMMA<",
        "      bfloat, float, BM, BN, BK, WM, WN, false, true,",
        "      BK_padded, BK_padded>;",
        "  using weight_loader_t = QuantizedBlockLoader<",
        "      bfloat, BN, BK, BK_padded, 1,",
        "      kNativeRoutedQgemmR1Threads, kQ4GroupSize, 4>;",
        "  thread mma_t gate_mma(simdgroup, lane);",
        "  thread mma_t up_mma(simdgroup, lane);",
        "  thread weight_loader_t gate_loader(",
        "      gate_bytes,",
        "      selected_gate_scales +",
        "          matrix_row * parameter_row_stride,",
        "      selected_gate_biases +",
        "          matrix_row * parameter_row_stride,",
        "      kHiddenDimension, gate_Ws, simdgroup, lane);",
        "  thread weight_loader_t up_loader(",
        "      up_bytes,",
        "      selected_up_scales +",
        "          matrix_row * parameter_row_stride,",
        "      selected_up_biases +",
        "          matrix_row * parameter_row_stride,",
        "      kHiddenDimension, up_Ws, simdgroup, lane);",
        "",
        "  for (uint reduction_tile = 0u;",
        "       reduction_tile < kHiddenDimension;",
        "       reduction_tile += BK) {",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    native_routed_qgemm_r2_stage_position_rows(",
        "        input, staged_routes, reduction_tile, Xs,",
        "        thread_index);",
        "    gate_loader.load_unsafe();",
        "    up_loader.load_unsafe();",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    gate_mma.mma(Xs, gate_Ws);",
        "    up_mma.mma(Xs, up_Ws);",
        "    gate_loader.next();",
        "    up_loader.next();",
        "  }",
        "",
        "  constexpr short row_fragments = BM / 8;",
        "  constexpr short column_fragments = BN / (8 * WN);",
        "  for (short row_fragment = 0;",
        "       row_fragment < row_fragments; ++row_fragment) {",
        "    const uint local_row =",
        "        uint(gate_mma.sm + row_fragment * 8);",
        "    const uint packed = staged_routes[local_row];",
        "    if (packed == kNativeRoutedQgemmR2InvalidRoute) {",
        "      continue;",
        "    }",
        "    const uint position = packed >> kPrefillPackedSlotBits;",
        "    const uint slot =",
        "        packed & ((1u << kPrefillPackedSlotBits) - 1u);",
        "    for (short column_fragment = 0;",
        "         column_fragment < column_fragments;",
        "         ++column_fragment) {",
        "      for (short element = 0; element < 2; ++element) {",
        "        const uint output_column =",
        "            output_tile +",
        "            uint(gate_mma.sn +",
        "                 column_fragment * 8 * WN + element);",
        "        const float gate = gate_mma.Ctile.frag_at(",
        "            row_fragment, column_fragment)[element];",
        "        const float up = up_mma.Ctile.frag_at(",
        "            row_fragment, column_fragment)[element];",
        "        hidden[native_routed_qgemm_r1_hidden_index(",
        "            position, slot, output_column)] =",
        "            static_cast<bfloat>(",
        "                (gate / (1.0f + exp(-gate))) * up);",
        "      }",
        "    }",
        "  }",
        "}",
        "[[kernel]] void " + MLX_STEEL_ROUTED_DOWN_KERNEL_NAME + "(",
        "    device const bfloat* hidden [[buffer(0)]],",
        "    device const uint* route_list [[buffer(1)]],",
        "    device const NativeRoutedQgemmR1Task* tasks [[buffer(2)]],",
        "    device const uint* down_words [[buffer(3)]],",
        "    device const bfloat* down_scales [[buffer(4)]],",
        "    device const bfloat* down_biases [[buffer(5)]],",
        "    device float* partials [[buffer(6)]],",
        "    constant uint& task_count [[buffer(7)]],",
        "    constant uint& route_list_expert_stride [[buffer(8)]],",
        "    constant uint& route_list_capacity_per_expert [[buffer(9)]],",
        "    constant ulong& route_list_total_extent [[buffer(10)]],",
        "    constant uint& input_rows [[buffer(11)]],",
        "    device const uint* shared_down_words [[buffer(12)]],",
        "    device const bfloat* shared_down_scales [[buffer(13)]],",
        "    device const bfloat* shared_down_biases [[buffer(14)]],",
        "    constant uint& include_shared_expert [[buffer(15)]],",
        "    uint3 group [[threadgroup_position_in_grid]],",
        "    uint lane [[thread_index_in_simdgroup]],",
        "    uint simdgroup [[simdgroup_index_in_threadgroup]],",
        "    uint simdgroup_width [[threads_per_simdgroup]],",
        "    uint thread_index [[thread_index_in_threadgroup]],",
        "    uint3 threadgroup_shape [[threads_per_threadgroup]]) {",
        "  constexpr short BM = 16;",
        "  constexpr short BN = 32;",
        "  constexpr short BK = 32;",
        "  constexpr short WM = 1;",
        "  constexpr short WN = 2;",
        "  constexpr short BK_padded = BK + 16 / sizeof(bfloat);",
        "  static_assert(",
        "      BM == kNativeRoutedQgemmR1TileRows &&",
        "      BN == kNativeRoutedQgemmR1TileColumns &&",
        "      BK == kNativeRoutedQgemmR1ReductionColumns &&",
        "      BK_padded == kNativeRoutedQgemmR2StageStride &&",
        "      WM * WN * SIMD_SIZE == kNativeRoutedQgemmR1Threads &&",
        "      kMoeExpertDimension % BK == 0 &&",
        "      kHiddenDimension % BN == 0,",
        "      \"Steel routed-down tile contract is invalid\");",
        "  if (!native_routed_qgemm_r1_dispatch_valid(",
        "          task_count, group, simdgroup, simdgroup_width,",
        "          threadgroup_shape)) {",
        "    return;",
        "  }",
        "  const NativeRoutedQgemmR1Task task = tasks[group.x];",
        "  if (!native_routed_qgemm_r2_task_valid(",
        "          task, route_list_expert_stride,",
        "          route_list_capacity_per_expert,",
        "          route_list_total_extent, include_shared_expert) ||",
        "      group.y >= kHiddenDimension / BN) {",
        "    return;",
        "  }",
        "",
        "  threadgroup uint staged_routes[BM];",
        "  threadgroup bfloat Xs[BM * BK_padded];",
        "  threadgroup bfloat Ws[BN * BK_padded];",
        "  native_routed_qgemm_r2_stage_routes(",
        "      route_list, task, input_rows, include_shared_expert,",
        "      staged_routes, thread_index);",
        "  threadgroup_barrier(mem_flags::mem_threadgroup);",
        "",
        "  const bool shared = task.expert_index == kMoeExperts;",
        "  device const uint* selected_words =",
        "      shared ? shared_down_words : down_words;",
        "  device const bfloat* selected_scales =",
        "      shared ? shared_down_scales : down_scales;",
        "  device const bfloat* selected_biases =",
        "      shared ? shared_down_biases : down_biases;",
        "  const ulong expert_row_begin =",
        "      shared ? 0ul",
        "             : ulong(task.expert_index) *",
        "                   ulong(kHiddenDimension);",
        "  const uint output_tile = group.y * BN;",
        "  const ulong matrix_row = expert_row_begin + output_tile;",
        "  device const uint8_t* weight_bytes =",
        "      reinterpret_cast<device const uint8_t*>(selected_words) +",
        "      matrix_row * ulong(kMoeExpertDimension / 2u);",
        "  const ulong parameter_row_stride =",
        "      ulong(kMoeExpertDimension / kQ4GroupSize);",
        "",
        "  using mma_t = mlx::steel::BlockMMA<",
        "      bfloat, float, BM, BN, BK, WM, WN, false, true,",
        "      BK_padded, BK_padded>;",
        "  using weight_loader_t = QuantizedBlockLoader<",
        "      bfloat, BN, BK, BK_padded, 1,",
        "      kNativeRoutedQgemmR1Threads, kQ4GroupSize, 4>;",
        "  thread mma_t mma_op(simdgroup, lane);",
        "  thread weight_loader_t weight_loader(",
        "      weight_bytes,",
        "      selected_scales + matrix_row * parameter_row_stride,",
        "      selected_biases + matrix_row * parameter_row_stride,",
        "      kMoeExpertDimension, Ws, simdgroup, lane);",
        "",
        "  for (uint reduction_tile = 0u;",
        "       reduction_tile < kMoeExpertDimension;",
        "       reduction_tile += BK) {",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    native_routed_qgemm_r2_stage_hidden_rows(",
        "        hidden, staged_routes, reduction_tile, Xs,",
        "        thread_index);",
        "    weight_loader.load_unsafe();",
        "    threadgroup_barrier(mem_flags::mem_threadgroup);",
        "    mma_op.mma(Xs, Ws);",
        "    weight_loader.next();",
        "  }",
        "",
        "  constexpr short row_fragments = BM / 8;",
        "  constexpr short column_fragments = BN / (8 * WN);",
        "  for (short row_fragment = 0;",
        "       row_fragment < row_fragments; ++row_fragment) {",
        "    const uint local_row =",
        "        uint(mma_op.sm + row_fragment * 8);",
        "    const uint packed = staged_routes[local_row];",
        "    if (packed == kNativeRoutedQgemmR2InvalidRoute) {",
        "      continue;",
        "    }",
        "    const uint position = packed >> kPrefillPackedSlotBits;",
        "    const uint slot =",
        "        packed & ((1u << kPrefillPackedSlotBits) - 1u);",
        "    for (short column_fragment = 0;",
        "         column_fragment < column_fragments;",
        "         ++column_fragment) {",
        "      for (short element = 0; element < 2; ++element) {",
        "        const uint output_column =",
        "            output_tile +",
        "            uint(mma_op.sn + column_fragment * 8 * WN +",
        "                 element);",
        "        partials[native_routed_qgemm_r1_partial_index(",
        "            position, slot, output_column)] =",
        "            mma_op.Ctile.frag_at(",
        "                row_fragment, column_fragment)[element];",
        "      }",
        "    }",
        "  }",
        "}",
        "#undef MLX_MTL_CONST",
    ]
    source = "\n".join(pieces)
    routed_begin = source.index(
        "[[kernel]] void " + MLX_STEEL_ROUTED_UPGATE_KERNEL_NAME
    )
    routed_end = source.index("#undef MLX_MTL_CONST", routed_begin)
    routed_template = source[routed_begin:routed_end]
    routed_bm8 = routed_template
    bm8_replacements = (
        (
            MLX_STEEL_ROUTED_UPGATE_KERNEL_NAME,
            "tatara_mlx_steel_routed_bm8_fused_upgate_swiglu",
        ),
        (
            MLX_STEEL_ROUTED_DOWN_KERNEL_NAME,
            "tatara_mlx_steel_routed_bm8_down_partial",
        ),
        ("constexpr short BM = 16;", "constexpr short BM = 8;"),
        (
            "BM == kNativeRoutedQgemmR1TileRows",
            "BM * 2 == kNativeRoutedQgemmR1TileRows",
        ),
        (
            "native_routed_qgemm_r1_dispatch_valid",
            "native_routed_qgemm_bm8_dispatch_valid",
        ),
        (
            "native_routed_qgemm_r2_task_valid",
            "native_routed_qgemm_bm8_task_valid",
        ),
        (
            "native_routed_qgemm_r2_stage_routes",
            "native_routed_qgemm_bm8_stage_routes",
        ),
    )
    for old_marker, new_marker in bm8_replacements:
        if old_marker not in routed_bm8:
            raise KernelLibraryGenerationError(
                f"BM8 routed Steel template marker is missing: {old_marker}"
            )
        routed_bm8 = routed_bm8.replace(old_marker, new_marker)

    routed_bm32 = routed_template
    replacements = (
        (
            MLX_STEEL_ROUTED_UPGATE_KERNEL_NAME,
            MLX_STEEL_ROUTED_BM32_UPGATE_KERNEL_NAME,
        ),
        (
            MLX_STEEL_ROUTED_DOWN_KERNEL_NAME,
            MLX_STEEL_ROUTED_BM32_DOWN_KERNEL_NAME,
        ),
        ("constexpr short BM = 16;", "constexpr short BM = 32;"),
        ("constexpr short WM = 1;", "constexpr short WM = 2;"),
        (
            "BM == kNativeRoutedQgemmR1TileRows",
            "BM == kNativeRoutedQgemmBm32TileRows",
        ),
        (
            "WM * WN * SIMD_SIZE == kNativeRoutedQgemmR1Threads",
            "WM * WN * SIMD_SIZE == kNativeRoutedQgemmBm32Threads",
        ),
        (
            "native_routed_qgemm_r1_dispatch_valid",
            "native_routed_qgemm_bm32_dispatch_valid",
        ),
        (
            "native_routed_qgemm_r2_task_valid",
            "native_routed_qgemm_bm32_task_valid",
        ),
        (
            "native_routed_qgemm_r2_stage_routes",
            "native_routed_qgemm_bm32_stage_routes",
        ),
        (
            "kNativeRoutedQgemmR1Threads, kQ4GroupSize, 4>",
            "kNativeRoutedQgemmBm32Threads, kQ4GroupSize, 4>",
        ),
        (
            "constexpr short row_fragments = BM / 8;",
            "constexpr short row_fragments = BM / (8 * WM);",
        ),
        (
            "uint(gate_mma.sm + row_fragment * 8);",
            "uint(gate_mma.sm + row_fragment * 8 * WM);",
        ),
        (
            "uint(mma_op.sm + row_fragment * 8);",
            "uint(mma_op.sm + row_fragment * 8 * WM);",
        ),
    )
    for old, new in replacements:
        if old not in routed_bm32:
            raise KernelLibraryGenerationError(
                f"BM32 routed Steel template marker is missing: {old}"
            )
        routed_bm32 = routed_bm32.replace(old, new)
    routed_bk64 = routed_template
    bk64_replacements = (
        (
            MLX_STEEL_ROUTED_UPGATE_KERNEL_NAME,
            MLX_STEEL_ROUTED_BK64_UPGATE_KERNEL_NAME,
        ),
        (
            MLX_STEEL_ROUTED_DOWN_KERNEL_NAME,
            MLX_STEEL_ROUTED_BK64_DOWN_KERNEL_NAME,
        ),
        ("constexpr short BK = 32;", "constexpr short BK = 64;"),
        (
            "BK == kNativeRoutedQgemmR1ReductionColumns",
            "BK == kQ4GroupSize",
        ),
        (
            "BK_padded == kNativeRoutedQgemmR2StageStride",
            "BK_padded == kQ4GroupSize + "
            "kNativeRoutedQgemmR2StageStride - "
            "kNativeRoutedQgemmR1ReductionColumns",
        ),
        (
            "native_routed_qgemm_r2_stage_position_rows",
            "native_routed_qgemm_bk64_stage_position_rows",
        ),
        (
            "native_routed_qgemm_r2_stage_hidden_rows",
            "native_routed_qgemm_bk64_stage_hidden_rows",
        ),
    )
    for old, new in bk64_replacements:
        if old not in routed_bk64:
            raise KernelLibraryGenerationError(
                f"BK64 routed Steel template marker is missing: {old}"
            )
        routed_bk64 = routed_bk64.replace(old, new)
    down_begin = routed_bk64.index(
        "[[kernel]] void " + MLX_STEEL_ROUTED_BK64_DOWN_KERNEL_NAME
    )
    bk64_upgate = routed_bk64[:down_begin].replace(
        "  threadgroup uint staged_routes[BM];",
        "  static_assert("
        "BM * sizeof(uint) + (BM + 2 * BN) * BK_padded * "
        "sizeof(bfloat) == 11584);\n"
        "  threadgroup uint staged_routes[BM];",
        1,
    )
    bk64_down = routed_bk64[down_begin:].replace(
        "  threadgroup uint staged_routes[BM];",
        "  static_assert("
        "BM * sizeof(uint) + (BM + BN) * BK_padded * "
        "sizeof(bfloat) == 6976);\n"
        "  threadgroup uint staged_routes[BM];",
        1,
    )
    routed_bk64 = bk64_upgate + bk64_down
    routed_bn64 = routed_template
    bn64_replacements = (
        (
            MLX_STEEL_ROUTED_UPGATE_KERNEL_NAME,
            MLX_STEEL_ROUTED_BN64_UPGATE_KERNEL_NAME,
        ),
        (
            MLX_STEEL_ROUTED_DOWN_KERNEL_NAME,
            MLX_STEEL_ROUTED_BN64_DOWN_KERNEL_NAME,
        ),
        ("constexpr short BN = 32;", "constexpr short BN = 64;"),
        (
            "BN == kNativeRoutedQgemmR1TileColumns",
            "BN == 2 * kNativeRoutedQgemmR1TileColumns",
        ),
    )
    for old, new in bn64_replacements:
        if old not in routed_bn64:
            raise KernelLibraryGenerationError(
                f"BN64 routed Steel template marker is missing: {old}"
            )
        routed_bn64 = routed_bn64.replace(old, new)
    down_begin = routed_bn64.index(
        "[[kernel]] void " + MLX_STEEL_ROUTED_BN64_DOWN_KERNEL_NAME
    )
    bn64_upgate = routed_bn64[:down_begin].replace(
        "  threadgroup uint staged_routes[BM];",
        "  static_assert("
        "BM * sizeof(uint) + (BM + 2 * BN) * BK_padded * "
        "sizeof(bfloat) == 11584);\n"
        "  threadgroup uint staged_routes[BM];",
        1,
    )
    bn64_down = routed_bn64[down_begin:].replace(
        "  threadgroup uint staged_routes[BM];",
        "  static_assert("
        "BM * sizeof(uint) + (BM + BN) * BK_padded * "
        "sizeof(bfloat) == 6464);\n"
        "  threadgroup uint staged_routes[BM];",
        1,
    )
    routed_bn64 = bn64_upgate + bn64_down

    # Plain bf16 steel_gemm_fused
    # instantiations for the staged-attention score (NT) and value (NN)
    # GEMMs. Function constants are sealed to the simple path: no batch
    # arrays (uniform batch_stride_* covers the head batching; the two KV
    # planes get one dispatch each), no out-source, no axpby; align_M/N/K
    # inherit the closure's existing `true` constants, so every dispatch
    # MUST present 32/32/16-aligned shapes (the bracket shapes are; any
    # future unaligned integration needs its own sealed variant).
    fused = texts[
        "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h"
    ]
    fused_body = _source_lines(fused, 18, 346)
    for gated in (
        " [[buffer(2), function_constant(use_out_source)]]",
        " [[buffer(5), function_constant(use_out_source)]]",
        " [[buffer(6), function_constant(has_batch)]]",
        " [[buffer(7), function_constant(has_batch)]]",
    ):
        stripped = gated.replace(", function_constant(use_out_source)", "")
        stripped = stripped.replace(", function_constant(has_batch)", "")
        if gated not in fused_body:
            raise KernelLibraryGenerationError(
                f"steel_gemm_fused gated-argument marker missing: {gated}"
            )
        fused_body = fused_body.replace(gated, stripped)
    fused_unaligned = fused_body
    for old_name, new_name in (
        ("void gemm(", "void tatara_attn_gemm_unaligned("),
        ("align_M", "tatara_attn_align_M"),
        ("align_N", "tatara_attn_align_N"),
        ("align_K", "tatara_attn_align_K"),
    ):
        if old_name not in fused_unaligned:
            raise KernelLibraryGenerationError(
                f"unaligned rename marker missing: {old_name}"
            )
        fused_unaligned = fused_unaligned.replace(old_name, new_name)
    attn_gemm_section = "\n".join([
        "",
        "// ---- Steel attention GEMM closure (MIT, (c) Apple Inc.) ----",
        _source_lines(
            texts["mlx/backend/metal/kernels/steel/gemm/params.h"], 9, 65),
        _source_lines(
            texts["mlx/backend/metal/kernels/steel/utils.h"], 6, 42),
        "namespace mlx { namespace steel {",
        _source_lines(
            texts["mlx/backend/metal/kernels/steel/gemm/transforms.h"],
            24, 69),
        "} } // namespace mlx::steel",
        "namespace mlx { namespace steel {",
        _source_lines(
            texts["mlx/backend/metal/kernels/steel/gemm/gemm.h"], 19, 293),
        "} } // namespace mlx::steel",
        "// elem_to_loc shim (mlx/backend/metal/kernels/utils.h:97-110,",
        "// MIT (c) Apple Inc.): required by the sealed-off has_batch/",
        "// use_out_source branches, which must still compile.",
        "template <typename IdxT = int64_t>",
        "METAL_FUNC IdxT elem_to_loc(",
        "    IdxT elem,",
        "    constant const int* shape,",
        "    constant const int64_t* strides,",
        "    int ndim) {",
        "  IdxT loc = 0;",
        "  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {",
        "    loc += (elem % shape[i]) * IdxT(strides[i]);",
        "    elem /= shape[i];",
        "  }",
        "  return loc;",
        "}",
        "using namespace mlx::steel;",
        "constant bool has_batch = false;",
        "constant bool use_out_source = false;",
        "constant bool do_axpby = false;",
        fused_body,
        "template [[host_name(\"tatara_mlx_steel_attn_scores_nt\")]]",
        "[[kernel]] decltype(gemm<bfloat16_t, 32, 32, 16, 2, 2, false,",
        "                         true, float>)",
        "    gemm<bfloat16_t, 32, 32, 16, 2, 2, false, true, float>;",
        "template [[host_name(\"tatara_mlx_steel_attn_values_nn\")]]",
        "[[kernel]] decltype(gemm<bfloat16_t, 32, 32, 16, 2, 2, false,",
        "                         false, float>)",
        "    gemm<bfloat16_t, 32, 32, 16, 2, 2, false, false, float>;",
        "// Unaligned variants: the product path dispatches arbitrary",
        "// visible/row shapes, so these run the bounds-checked tails",
        "// everywhere (align constants false via rename).",
        "constant bool tatara_attn_align_M = false;",
        "constant bool tatara_attn_align_N = false;",
        "constant bool tatara_attn_align_K = false;",
        fused_unaligned,
        "template",
        "[[host_name(\"tatara_mlx_steel_attn_scores_nt_unaligned\")]]",
        "[[kernel]] decltype(tatara_attn_gemm_unaligned<bfloat16_t, 32,",
        "                        32, 16, 2, 2, false, true, float>)",
        "    tatara_attn_gemm_unaligned<bfloat16_t, 32, 32, 16, 2, 2,",
        "                               false, true, float>;",
        "template",
        "[[host_name(\"tatara_mlx_steel_attn_values_nn_unaligned\")]]",
        "[[kernel]] decltype(tatara_attn_gemm_unaligned<bfloat16_t, 32,",
        "                        32, 16, 2, 2, false, false, float>)",
        "    tatara_attn_gemm_unaligned<bfloat16_t, 32, 32, 16, 2, 2,",
        "                               false, false, float>;",
        "// Large-tile (64x64) pair for deep-context dispatches:",
        "// same sealed template, bigger blocks.",
        "template",
        "[[host_name(\"tatara_mlx_steel_attn_scores_nt_unaligned_l\")]]",
        "[[kernel]] decltype(tatara_attn_gemm_unaligned<bfloat16_t, 64,",
        "                        64, 16, 2, 2, false, true, float>)",
        "    tatara_attn_gemm_unaligned<bfloat16_t, 64, 64, 16, 2, 2,",
        "                               false, true, float>;",
        "template",
        "[[host_name(\"tatara_mlx_steel_attn_values_nn_unaligned_l\")]]",
        "[[kernel]] decltype(tatara_attn_gemm_unaligned<bfloat16_t, 64,",
        "                        64, 16, 2, 2, false, false, float>)",
        "    tatara_attn_gemm_unaligned<bfloat16_t, 64, 64, 16, 2, 2,",
        "                               false, false, float>;",
    ])
    return (
        source[:routed_end]
        + routed_bm32
        + routed_bk64
        + routed_bn64
        + routed_bm8
        + source[routed_end:]
        + attn_gemm_section
    )


def _prelude(plan: GeneratedModelPlan) -> str:
    hidden = plan.hidden
    group_size = plan.group_size
    # All admitted Q4/Q8 dot helpers use group-64 addressing (`k >> 6`) and
    # load scales/biases at that cadence. Emitting a different package value
    # would make the generated metadata disagree with execution.
    if group_size != 64:
        raise KernelLibraryGenerationError(
            f"Quantization group size {group_size} is unsupported; expected 64"
        )
    if hidden % NIBBLES_PER_QUANT_WORD != 0:
        raise KernelLibraryGenerationError("Hidden dimension is not nibble-packable")
    if hidden % group_size != 0:
        raise KernelLibraryGenerationError("Hidden dimension is not group-aligned")
    if hidden % RMS_VALUES_PER_THREAD != 0:
        raise KernelLibraryGenerationError("Hidden dimension is not RMS-thread-aligned")
    if hidden // RMS_VALUES_PER_THREAD > MAX_THREADGROUP_THREADS:
        raise KernelLibraryGenerationError("Hidden dimension exceeds the RMS threadgroup")
    value_heads = plan.recurrent_heads
    head_dimension = plan.state_dimension
    if value_heads % 2 != 0:
        raise KernelLibraryGenerationError("GDN value heads must divide into query/key pairs")
    key_heads = value_heads // 2
    if head_dimension != FIRST_PACKAGE_GDN_HEAD_DIMENSION:
        raise KernelLibraryGenerationError(
            f"GDN head dimension {head_dimension} is unsupported; "
            f"expected {FIRST_PACKAGE_GDN_HEAD_DIMENSION}"
        )
    qk_values = 2 * key_heads * head_dimension
    value_values = value_heads * head_dimension
    qkv_rows = qk_values + value_values
    b_offset = qkv_rows + value_values
    a_offset = b_offset + value_heads
    projection_rows = a_offset + value_heads
    k_offset = key_heads * head_dimension
    attn_query_heads = plan.query_heads
    attn_kv_heads = plan.key_value_heads
    attn_head_dimension = plan.head_dimension
    if attn_head_dimension != FIRST_PACKAGE_ATTN_HEAD_DIMENSION:
        raise KernelLibraryGenerationError(
            f"Attention head dimension {attn_head_dimension} is unsupported; "
            f"expected {FIRST_PACKAGE_ATTN_HEAD_DIMENSION}"
        )
    if attn_query_heads % attn_kv_heads != 0:
        raise KernelLibraryGenerationError("Attention geometry is not lane-aligned")
    if attn_query_heads // attn_kv_heads != FIRST_PACKAGE_ATTN_HEADS_PER_KV:
        raise KernelLibraryGenerationError(
            f"Attention GQA ratio is unsupported; expected {FIRST_PACKAGE_ATTN_HEADS_PER_KV}"
        )
    if (
        attn_head_dimension %
        PREFILL_STREAMING_ATTENTION_OUTPUT_COLUMNS_PER_SIMDGROUP
        != 0
    ):
        raise KernelLibraryGenerationError(
            "Attention head dimension is not streaming-output-tile aligned"
        )
    streaming_attention_simdgroups = (
        attn_head_dimension
        // PREFILL_STREAMING_ATTENTION_OUTPUT_COLUMNS_PER_SIMDGROUP
    )
    streaming_attention_threads = streaming_attention_simdgroups * 32
    if streaming_attention_threads > MAX_THREADGROUP_THREADS:
        raise KernelLibraryGenerationError(
            "Streaming attention exceeds the Metal threadgroup limit"
        )
    streaming_attention_key_tile_columns = (
        streaming_attention_simdgroups
        * PREFILL_STREAMING_ATTENTION_KEYS_PER_SIMDGROUP
    )
    streaming_attention_threadgroup_memory_bytes = (
        PREFILL_STREAMING_ATTENTION_QUERY_TILE_ROWS
        * streaming_attention_key_tile_columns
        + 3 * PREFILL_STREAMING_ATTENTION_QUERY_TILE_ROWS
    ) * FLOAT_BYTES
    if (
        streaming_attention_threadgroup_memory_bytes
        > MINIMUM_THREADGROUP_MEMORY_BYTES
    ):
        raise KernelLibraryGenerationError(
            "Streaming attention exceeds the minimum Metal threadgroup-memory guarantee"
        )
    attn_qgate_rows = attn_query_heads * 2 * attn_head_dimension
    attn_v_offset = attn_qgate_rows + attn_kv_heads * attn_head_dimension
    attn_projection_rows = attn_v_offset + attn_kv_heads * attn_head_dimension
    moe_experts = plan.experts
    moe_active = plan.active_experts
    moe_expert_dimension = plan.expert_dimension
    # The router selection trees and threadgroup arrays size themselves from
    # kMoeExperts, so the expert count is no longer pinned to one value -- but
    # it is not free either, and the two real constraints are checked here
    # rather than left to fail on silicon:
    #
    #   power of two   the reduction halves the width each pass, so with 6
    #                  experts the off=3 pass folds 3..5 into 0..2 and the
    #                  off=1 pass then never merges slot 2.
    #   <= 1024        router_select dispatches {groups = 1, threads = experts}
    #                  and a Metal threadgroup tops out at 1024 threads.
    #
    # Threadgroup memory is not a binding constraint at that ceiling: the three
    # arrays cost 12 bytes per expert, so 1024 experts is 12 KiB against the
    # 32 KiB limit.
    if moe_experts < 2 or moe_experts > MAX_THREADGROUP_THREADS:
        raise KernelLibraryGenerationError(
            f"Expert count {moe_experts} is outside the router dispatch range "
            f"2..{MAX_THREADGROUP_THREADS}"
        )
    if moe_experts & (moe_experts - 1) != 0:
        raise KernelLibraryGenerationError(
            f"Expert count {moe_experts} is not a power of two, which the router "
            "selection tree requires"
        )
    # The dot helpers walk fixed 512-value (packed Q4) and 256-value (Q8)
    # chunks, so every dotted width must be chunk-aligned.
    if not 0 < moe_active < moe_experts:
        raise KernelLibraryGenerationError("Active expert count is out of range")
    packed_slot_bits = moe_active.bit_length()
    if hidden % 512 != 0 or moe_expert_dimension % 512 != 0:
        raise KernelLibraryGenerationError("A packed-Q4 dot width is not chunk-aligned")
    if hidden % 256 != 0:
        raise KernelLibraryGenerationError("The Q8 dot width is not chunk-aligned")
    if moe_expert_dimension % group_size != 0:
        raise KernelLibraryGenerationError("Expert dimension is not group-aligned")
    vocabulary = plan.vocabulary
    # The argmax grid-stride literals cover 256 threadgroups of 256 threads.
    if vocabulary == 0 or vocabulary > ARGMAX_GROUPS * 65536:
        raise KernelLibraryGenerationError("Vocabulary is outside the argmax grid coverage")
    return "\n".join(
        (
            "#include <metal_stdlib>",
            "using namespace metal;",
            "",
            f"constant uint kHiddenDimension = {hidden}u;",
            f"constant uint kQ4GroupSize = {group_size}u;",
            f"constant uint kQuantWordsPerRow = {hidden // NIBBLES_PER_QUANT_WORD}u;",
            f"constant uint kGroupsPerRow = {hidden // group_size}u;",
            "constant uint kNativeDenseQgemmN1TileRows = "
            f"{NATIVE_DENSE_QGEMM_N1_TILE_ROWS}u;",
            "constant uint kNativeDenseQgemmN1TileColumns = "
            f"{NATIVE_DENSE_QGEMM_N1_TILE_COLUMNS}u;",
            "constant uint kNativeDenseQgemmN1ReductionColumns = "
            f"{NATIVE_DENSE_QGEMM_N1_REDUCTION_COLUMNS}u;",
            "constant uint kNativeDenseQgemmN1Simdgroups = "
            f"{NATIVE_DENSE_QGEMM_N1_SIMDGROUPS}u;",
            "constant uint kNativeDenseQgemmN1Threads = "
            f"{NATIVE_DENSE_QGEMM_N1_THREADS}u;",
            "constant uint kNativeDenseQgemmN1SimdgroupGridRows = "
            f"{NATIVE_DENSE_QGEMM_N1_SIMDGROUP_GRID_ROWS}u;",
            "constant uint kNativeDenseQgemmN1SimdgroupGridColumns = "
            f"{NATIVE_DENSE_QGEMM_N1_SIMDGROUP_GRID_COLUMNS}u;",
            "constant uint kNativeDenseQgemmN1StageRowPadding = "
            f"{NATIVE_DENSE_QGEMM_N1_STAGE_ROW_PADDING}u;",
            "constant uint kNativeDenseQgemmN1ActivationStageStride = "
            f"{NATIVE_DENSE_QGEMM_N1_ACTIVATION_STAGE_STRIDE}u;",
            "constant uint kNativeDenseQgemmN1WeightStageStride = "
            f"{NATIVE_DENSE_QGEMM_N1_WEIGHT_STAGE_STRIDE}u;",
            "constant uint kNativeDenseQgemmN1PackedWordsPerTileRow = "
            f"{NATIVE_DENSE_QGEMM_N1_PACKED_WORDS_PER_TILE_ROW}u;",
            "constant uint kNativeDenseQgemmN1PackedWordsPerTile = "
            f"{NATIVE_DENSE_QGEMM_N1_PACKED_WORDS_PER_TILE}u;",
            "constant uint kNativeDenseQgemmN1MetadataPairsPerGroupTile = "
            f"{NATIVE_DENSE_QGEMM_N1_METADATA_PAIRS_PER_GROUP_TILE}u;",
            "constant uint kNativeDenseQgemmN1ThreadgroupMemoryBytes = "
            f"{NATIVE_DENSE_QGEMM_N1_THREADGROUP_MEMORY_BYTES}u;",
            "constant uint kNativeDenseQgemmN1AccumulatorElements = "
            f"{NATIVE_DENSE_QGEMM_N1_ACCUMULATOR_ELEMENTS}u;",
            "constant uint kNativeRoutedQgemmR1TileRows = "
            f"{NATIVE_ROUTED_QGEMM_R1_TILE_ROWS}u;",
            "constant uint kNativeRoutedQgemmR1TileColumns = "
            f"{NATIVE_ROUTED_QGEMM_R1_TILE_COLUMNS}u;",
            "constant uint kNativeRoutedQgemmR1ReductionColumns = "
            f"{NATIVE_ROUTED_QGEMM_R1_REDUCTION_COLUMNS}u;",
            "constant uint kNativeRoutedQgemmR1Simdgroups = "
            f"{NATIVE_ROUTED_QGEMM_R1_SIMDGROUPS}u;",
            "constant uint kNativeRoutedQgemmR1Threads = "
            f"{NATIVE_ROUTED_QGEMM_R1_THREADS}u;",
            "constant uint kNativeRoutedQgemmR1TaskBytes = "
            f"{NATIVE_ROUTED_QGEMM_R1_TASK_BYTES}u;",
            "constant uint kNativeRoutedQgemmR1TaskCapacity = "
            f"{NATIVE_ROUTED_QGEMM_R1_TASK_CAPACITY}u;",
            "constant uint kNativeRoutedQgemmR1FusedAccumulatorElements = "
            f"{NATIVE_ROUTED_QGEMM_R1_FUSED_ACCUMULATOR_ELEMENTS}u;",
            "constant uint kNativeRoutedQgemmR1SingleAccumulatorElements = "
            f"{NATIVE_ROUTED_QGEMM_R1_SINGLE_ACCUMULATOR_ELEMENTS}u;",
            "constant uint kNativeRoutedQgemmR1ThreadgroupMemoryBytes = "
            f"{NATIVE_ROUTED_QGEMM_R1_THREADGROUP_MEMORY_BYTES}u;",
            "constant uint kNativeRoutedQgemmR1AccumulatorElements = "
            f"{NATIVE_ROUTED_QGEMM_R1_ACCUMULATOR_ELEMENTS}u;",
            *(
                "constant uint kNativeRoutedQgemmR1TaskStatus"
                f"{name} = {value}u;"
                for name, value in NATIVE_ROUTED_QGEMM_R1_TASK_STATUS_VALUES
            ),
            "constant uint kMinimumThreadgroupMemoryBytes = "
            f"{MINIMUM_THREADGROUP_MEMORY_BYTES}u;",
            f"constant uint kRmsValuesPerThread = {RMS_VALUES_PER_THREAD}u;",
            "constant uint kSimdgroupWidth = 32u;",
            "constant uint kQ4ValuesPerWord = 8u;",
            "constant uint kQ4DotChunk = 512u;",
            "constant uint kQ4LaneValues = 16u;",
            "constant uint kQ4PackedWordsPerLane = 2u;",
            f"constant float kRmsNormEpsilon = {FIRST_PACKAGE_RMS_EPSILON};",
            f"constant uint kGdnValueHeads = {value_heads}u;",
            f"constant uint kGdnKeyHeads = {key_heads}u;",
            f"constant uint kGdnHeadDimension = {head_dimension}u;",
            f"constant uint kGdnConvWidth = {FIRST_PACKAGE_GDN_CONV_WIDTH}u;",
            f"constant uint kGdnQkValues = {qk_values}u;",
            f"constant uint kGdnValueValues = {value_values}u;",
            f"constant uint kGdnQkvRows = {qkv_rows}u;",
            f"constant uint kGdnConvChannels = {qkv_rows}u;",
            f"constant uint kGdnZRows = {value_values}u;",
            f"constant uint kGdnBRowOffset = {b_offset}u;",
            f"constant uint kGdnARowOffset = {a_offset}u;",
            f"constant uint kGdnProjectionRows = {projection_rows}u;",
            f"constant uint kGdnKOffset = {k_offset}u;",
            f"constant float kGdnQueryScale = {FIRST_PACKAGE_GDN_QUERY_SCALE};",
            f"constant float kGdnKeyScale = {FIRST_PACKAGE_GDN_KEY_SCALE};",
            f"constant uint kAttnQueryHeads = {attn_query_heads}u;",
            f"constant uint kAttnKvHeads = {attn_kv_heads}u;",
            f"constant uint kAttnHeadDimension = {attn_head_dimension}u;",
            f"constant uint kAttnQGateRows = {attn_qgate_rows}u;",
            f"constant uint kAttnVRowOffset = {attn_v_offset}u;",
            f"constant uint kAttnProjectionRows = {attn_projection_rows}u;",
            f"constant uint kAttnRopePairs = {attn_head_dimension // 8}u;",
            f"constant uint kAttnHeadsPerKv = {attn_query_heads // attn_kv_heads}u;",
            f"constant float kAttnRopeBase = {FIRST_PACKAGE_ATTN_ROPE_BASE};",
            f"constant float kAttnScale = {FIRST_PACKAGE_ATTN_SCALE};",
            f"constant uint kMoeExperts = {moe_experts}u;",
            f"constant uint kMoeActiveExperts = {moe_active}u;",
            f"constant uint kMoeExpertDimension = {moe_expert_dimension}u;",
            f"constant uint kMoeRouterRows = {moe_experts + 1}u;",
            f"constant uint kMoeSlotCount = {moe_active + 1}u;",
            f"constant uint kPrefillPackedSlotBits = {packed_slot_bits}u;",
            "constant uint kPrefillPositionBatch = 4u;",
            "constant uint kPrefillAttentionPartition = 256u;",
            f"constant uint kPrefillAttentionRecordFloats = {attn_head_dimension + 2}u;",
            "constant uint kPrefillStagedAttentionQueryTileRows = "
            f"{PREFILL_STAGED_ATTENTION_QUERY_TILE_ROWS}u;",
            "constant uint kPrefillStagedAttentionKeyTileColumns = "
            f"{PREFILL_STAGED_ATTENTION_KEY_TILE_COLUMNS}u;",
            "constant uint kPrefillStagedAttentionOutputTileColumns = "
            f"{PREFILL_STAGED_ATTENTION_OUTPUT_TILE_COLUMNS}u;",
            "constant uint kPrefillStagedAttentionSimdgroups = "
            f"{PREFILL_STAGED_ATTENTION_SIMDGROUPS}u;",
            "constant uint kPrefillStagedAttentionThreads = "
            f"{PREFILL_STAGED_ATTENTION_THREADS}u;",
            "constant uint kPrefillStagedAttentionSoftmaxThreads = "
            f"{PREFILL_STAGED_ATTENTION_SOFTMAX_THREADS}u;",
            "constant uint kPrefillStreamingAttentionQueryTileRows = "
            f"{PREFILL_STREAMING_ATTENTION_QUERY_TILE_ROWS}u;",
            "constant uint kPrefillStreamingAttentionKeysPerSimdgroup = "
            f"{PREFILL_STREAMING_ATTENTION_KEYS_PER_SIMDGROUP}u;",
            "constant uint kPrefillStreamingAttentionOutputColumnsPerSimdgroup = "
            f"{PREFILL_STREAMING_ATTENTION_OUTPUT_COLUMNS_PER_SIMDGROUP}u;",
            "constant uint kPrefillStreamingAttentionSimdgroups = "
            f"{streaming_attention_simdgroups}u;",
            "constant uint kPrefillStreamingAttentionThreads = "
            f"{streaming_attention_threads}u;",
            "constant uint kPrefillStreamingAttentionKeyTileColumns = "
            f"{streaming_attention_key_tile_columns}u;",
            "constant uint kPrefillStreamingAttentionThreadgroupMemoryBytes = "
            f"{streaming_attention_threadgroup_memory_bytes}u;",
            "constant uint kPrefillFlashV2QueryTileRows = 16u;",
            "constant uint kPrefillFlashV2Threads = 256u;",
            "constant uint kPrefillFlashV2KeyTileColumns = 32u;",
            "constant uint kGdnTapeRows = 16u;",
            "constant uint kGdnTapeKeyOffset = "
            "40u * 16u * 32u * 128u;",
            "constant uint kGdnTapeGateOffset = "
            "40u * 16u * 32u * 128u + 40u * 16u * 16u * 128u;",
            f"constant uint kVocabularyRows = {vocabulary}u;",
            f"constant uint kVerifyM16Threads = {VERIFY_M16_THREADS}u;",
            f"constant uint kVerifyM16MaxRows = {VERIFY_M16_MAX_ROWS}u;",
            "constant uint kVerifyM16ColumnsPerGroup = "
            f"{VERIFY_M16_COLUMNS_PER_GROUP}u;",
            f"constant uint kDraftDenseThreads = {DRAFT_DENSE_THREADS}u;",
            f"constant uint kDraftGemmThreads = {DRAFT_GEMM_THREADS}u;",
            f"constant uint kDraftGemmColumnsPerSimdgroup = "
            f"{DRAFT_GEMM_COLUMNS_PER_SIMDGROUP}u;",
            f"constant uint kDraftMaxBlockRows = {DRAFT_MAX_BLOCK_ROWS}u;",
            f"constant uint kDraftRmsMaxSimdgroups = {DRAFT_RMS_MAX_SIMDGROUPS}u;",
            "constant float kDraftRmsEpsilon = 1e-6f;",
            f"constant uint kDraftHeadDim = {DRAFT_HEAD_DIMENSION}u;",
            "constant float kDraftRopeBase = 1e7f;",
            f"constant uint kDraftQHeads = {DRAFT_QUERY_HEADS}u;",
            f"constant uint kDraftKvHeads = {DRAFT_KV_HEADS}u;",
            f"constant uint kDraftAttnThreads = {DRAFT_ATTENTION_THREADS}u;",
            f"constant uint kDraftAttnKeyTile = {DRAFT_ATTENTION_KEY_TILE}u;",
            "constant uint kDraftAttnLocalPairs = "
            f"{(DRAFT_MAX_BLOCK_ROWS * (DRAFT_QUERY_HEADS // DRAFT_KV_HEADS)) // (DRAFT_ATTENTION_THREADS // 32)}u;",
            f"constant uint kDraftWindowPositions = {DRAFT_WINDOW_POSITIONS}u;",
            "",
        )
    )


def build_kernel_library(
    plan: GeneratedModelPlan,
    kernel_directory: Path,
    mlx_steel_source_root: Path | None = None,
) -> KernelLibrary:
    pieces = [_prelude(plan)]
    for name in KERNEL_FILE_NAMES:
        text = (kernel_directory / name).read_text(encoding="utf-8")
        if RAW_STRING_DELIMITER in text:
            raise KernelLibraryGenerationError(f"{name} contains the raw string delimiter")
        pieces.append(f"// kernel file: {name}\n{text}")
    if mlx_steel_source_root is not None:
        steel_source = build_mlx_steel_source(mlx_steel_source_root)
        if RAW_STRING_DELIMITER in steel_source:
            raise KernelLibraryGenerationError(
                "MLX Steel source contains the raw string delimiter"
            )
        pieces.append("// vendored Steel GEMM closure\n" + steel_source)
    source = "\n".join(pieces)
    if RAW_STRING_DELIMITER in _prelude(plan):
        raise KernelLibraryGenerationError("Prelude contains the raw string delimiter")
    return KernelLibrary(
        hidden=plan.hidden,
        group_size=plan.group_size,
        key_heads=plan.recurrent_heads // 2,
        value_heads=plan.recurrent_heads,
        head_dimension=plan.state_dimension,
        attn_query_heads=plan.query_heads,
        attn_kv_heads=plan.key_value_heads,
        attn_head_dimension=plan.head_dimension,
        moe_experts=plan.experts,
        moe_active_experts=plan.active_experts,
        moe_expert_dimension=plan.expert_dimension,
        vocabulary=plan.vocabulary,
        mlx_steel_enabled=mlx_steel_source_root is not None,
        source=source,
    )


def chunk_source(source: str, limit: int = SOURCE_CHUNK_CHARS) -> list[str]:
    """Split on line boundaries so each emitted literal stays under the C++ bound.

    Joining the chunks reproduces the input exactly, so the concatenated array
    is byte-identical to the unsplit source.
    """
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for line in source.splitlines(keepends=True):
        if len(line) > limit:
            raise KernelLibraryGenerationError(
                f"A kernel source line is {len(line)} characters, over the {limit} chunk bound"
            )
        if size + len(line) > limit and current:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line)
    if current:
        chunks.append("".join(current))
    return chunks or [""]


def _render_source_parts(source: str) -> str:
    """Emit the source as separately NAMED literals plus a joining accessor.

    Adjacent literals do not help: the implementation limit applies to the
    concatenated result, so `-Woverlength-strings` still fires. Distinct
    variables are never concatenated by the compiler, which is the same shape
    the reference runtime uses.
    """
    chunks = chunk_source(source)
    lines = []
    for index, chunk in enumerate(chunks):
        lines.append(f"inline constexpr char kQwen36KernelLibraryPart{index}[] =")
        lines.append(f'    R"{RAW_STRING_DELIMITER}({chunk}){RAW_STRING_DELIMITER}";')
        lines.append("")
    joined = " + ".join(
        (f"std::string(kQwen36KernelLibraryPart{i})" if i == 0 else f"kQwen36KernelLibraryPart{i}")
        for i in range(len(chunks))
    )
    lines.append("inline std::string kernel_library_source() {")
    lines.append(f"    return {joined};")
    lines.append("}")
    return "\n".join(lines)


def render_kernel_library_header(library: KernelLibrary) -> str:
    return "\n".join(
        (
            "// Generated by tools/generate_kernel_library.py; do not edit.",
            "#pragma once",
            "",
            "#include <string>",
            "",
            "namespace tatara::backend::metal::generated {",
            "",
            f"inline constexpr unsigned int kKernelLibraryHidden = {library.hidden}u;",
            f"inline constexpr unsigned int kKernelLibraryGroupSize = {library.group_size}u;",
            "inline constexpr unsigned int kKernelLibraryNativeDenseQgemmN1TileRows = "
            f"{NATIVE_DENSE_QGEMM_N1_TILE_ROWS}u;",
            "inline constexpr unsigned int kKernelLibraryNativeDenseQgemmN1TileColumns = "
            f"{NATIVE_DENSE_QGEMM_N1_TILE_COLUMNS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeDenseQgemmN1ReductionColumns = "
            f"{NATIVE_DENSE_QGEMM_N1_REDUCTION_COLUMNS}u;",
            "inline constexpr unsigned int kKernelLibraryNativeDenseQgemmN1Simdgroups = "
            f"{NATIVE_DENSE_QGEMM_N1_SIMDGROUPS}u;",
            "inline constexpr unsigned int kKernelLibraryNativeDenseQgemmN1Threads = "
            f"{NATIVE_DENSE_QGEMM_N1_THREADS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeDenseQgemmN1SimdgroupGridRows = "
            f"{NATIVE_DENSE_QGEMM_N1_SIMDGROUP_GRID_ROWS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeDenseQgemmN1SimdgroupGridColumns = "
            f"{NATIVE_DENSE_QGEMM_N1_SIMDGROUP_GRID_COLUMNS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeDenseQgemmN1StageRowPadding = "
            f"{NATIVE_DENSE_QGEMM_N1_STAGE_ROW_PADDING}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeDenseQgemmN1ActivationStageStride = "
            f"{NATIVE_DENSE_QGEMM_N1_ACTIVATION_STAGE_STRIDE}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeDenseQgemmN1WeightStageStride = "
            f"{NATIVE_DENSE_QGEMM_N1_WEIGHT_STAGE_STRIDE}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeDenseQgemmN1PackedWordsPerTileRow = "
            f"{NATIVE_DENSE_QGEMM_N1_PACKED_WORDS_PER_TILE_ROW}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeDenseQgemmN1PackedWordsPerTile = "
            f"{NATIVE_DENSE_QGEMM_N1_PACKED_WORDS_PER_TILE}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeDenseQgemmN1MetadataPairsPerGroupTile = "
            f"{NATIVE_DENSE_QGEMM_N1_METADATA_PAIRS_PER_GROUP_TILE}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeDenseQgemmN1ThreadgroupMemoryBytes = "
            f"{NATIVE_DENSE_QGEMM_N1_THREADGROUP_MEMORY_BYTES}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeDenseQgemmN1AccumulatorElements = "
            f"{NATIVE_DENSE_QGEMM_N1_ACCUMULATOR_ELEMENTS}u;",
            "inline constexpr bool kKernelLibraryNativeDenseQgemmN1DenseScope = true;",
            "inline constexpr bool kKernelLibraryNativeDenseQgemmN1RaggedScope = false;",
            "inline constexpr bool kKernelLibraryNativeDenseQgemmN1HostReachable = true;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR1TileRows = "
            f"{NATIVE_ROUTED_QGEMM_R1_TILE_ROWS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR1TileColumns = "
            f"{NATIVE_ROUTED_QGEMM_R1_TILE_COLUMNS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR1ReductionColumns = "
            f"{NATIVE_ROUTED_QGEMM_R1_REDUCTION_COLUMNS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR1Simdgroups = "
            f"{NATIVE_ROUTED_QGEMM_R1_SIMDGROUPS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR1Threads = "
            f"{NATIVE_ROUTED_QGEMM_R1_THREADS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR1TaskBytes = "
            f"{NATIVE_ROUTED_QGEMM_R1_TASK_BYTES}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR1TaskCapacity = "
            f"{NATIVE_ROUTED_QGEMM_R1_TASK_CAPACITY}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR1FusedAccumulatorElements = "
            f"{NATIVE_ROUTED_QGEMM_R1_FUSED_ACCUMULATOR_ELEMENTS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR1SingleAccumulatorElements = "
            f"{NATIVE_ROUTED_QGEMM_R1_SINGLE_ACCUMULATOR_ELEMENTS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR1ThreadgroupMemoryBytes = "
            f"{NATIVE_ROUTED_QGEMM_R1_THREADGROUP_MEMORY_BYTES}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR1AccumulatorElements = "
            f"{NATIVE_ROUTED_QGEMM_R1_ACCUMULATOR_ELEMENTS}u;",
            "inline constexpr bool kKernelLibraryNativeRoutedQgemmR1DenseScope = false;",
            "inline constexpr bool kKernelLibraryNativeRoutedQgemmR1RaggedScope = true;",
            "inline constexpr bool kKernelLibraryNativeRoutedQgemmR1HostReachable = true;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR2StageStride = "
            f"{NATIVE_ROUTED_QGEMM_R2_STAGE_STRIDE}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR2FusedThreadgroupMemoryBytes = "
            f"{NATIVE_ROUTED_QGEMM_R2_FUSED_THREADGROUP_MEMORY_BYTES}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryNativeRoutedQgemmR2DownThreadgroupMemoryBytes = "
            f"{NATIVE_ROUTED_QGEMM_R2_DOWN_THREADGROUP_MEMORY_BYTES}u;",
            "inline constexpr bool kKernelLibraryNativeRoutedQgemmR2HostReachable = true;",
            f"inline constexpr unsigned int kKernelLibraryGdnKeyHeads = {library.key_heads}u;",
            "inline constexpr unsigned int kKernelLibraryGdnValueHeads = "
            f"{library.value_heads}u;",
            "inline constexpr unsigned int kKernelLibraryGdnHeadDimension = "
            f"{library.head_dimension}u;",
            "inline constexpr float kKernelLibraryGdnQueryScale = "
            f"{FIRST_PACKAGE_GDN_QUERY_SCALE};",
            f"inline constexpr float kKernelLibraryGdnKeyScale = {FIRST_PACKAGE_GDN_KEY_SCALE};",
            "inline constexpr unsigned int kKernelLibraryAttnQueryHeads = "
            f"{library.attn_query_heads}u;",
            f"inline constexpr unsigned int kKernelLibraryAttnKvHeads = {library.attn_kv_heads}u;",
            "inline constexpr unsigned int kKernelLibraryAttnHeadDimension = "
            f"{library.attn_head_dimension}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillStagedAttentionQueryTileRows = "
            f"{PREFILL_STAGED_ATTENTION_QUERY_TILE_ROWS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillStagedAttentionKeyTileColumns = "
            f"{PREFILL_STAGED_ATTENTION_KEY_TILE_COLUMNS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillStagedAttentionOutputTileColumns = "
            f"{PREFILL_STAGED_ATTENTION_OUTPUT_TILE_COLUMNS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillStagedAttentionThreads = "
            f"{PREFILL_STAGED_ATTENTION_THREADS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillStagedAttentionSoftmaxThreads = "
            f"{PREFILL_STAGED_ATTENTION_SOFTMAX_THREADS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillStreamingAttentionQueryTileRows = "
            f"{PREFILL_STREAMING_ATTENTION_QUERY_TILE_ROWS}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillStreamingAttentionKeysPerSimdgroup = "
            f"{PREFILL_STREAMING_ATTENTION_KEYS_PER_SIMDGROUP}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillStreamingAttentionOutputColumnsPerSimdgroup = "
            f"{PREFILL_STREAMING_ATTENTION_OUTPUT_COLUMNS_PER_SIMDGROUP}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillStreamingAttentionSimdgroups = "
            f"{library.attn_head_dimension // PREFILL_STREAMING_ATTENTION_OUTPUT_COLUMNS_PER_SIMDGROUP}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillStreamingAttentionThreads = "
            f"{(library.attn_head_dimension // PREFILL_STREAMING_ATTENTION_OUTPUT_COLUMNS_PER_SIMDGROUP) * 32}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillStreamingAttentionKeyTileColumns = "
            f"{(library.attn_head_dimension // PREFILL_STREAMING_ATTENTION_OUTPUT_COLUMNS_PER_SIMDGROUP) * PREFILL_STREAMING_ATTENTION_KEYS_PER_SIMDGROUP}u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillFlashV2QueryTileRows = 16u;",
            "inline constexpr unsigned int "
            "kKernelLibraryPrefillFlashV2Threads = 256u;",
            f"inline constexpr unsigned int kKernelLibraryMoeExperts = {library.moe_experts}u;",
            "inline constexpr unsigned int kKernelLibraryMoeActiveExperts = "
            f"{library.moe_active_experts}u;",
            "inline constexpr unsigned int kKernelLibraryPrefillPackedSlotBits = "
            f"{library.moe_active_experts.bit_length()}u;",
            "inline constexpr unsigned int kKernelLibraryMoeExpertDimension = "
            f"{library.moe_expert_dimension}u;",
            f"inline constexpr unsigned int kKernelLibraryVocabulary = {library.vocabulary}u;",
            f"inline constexpr unsigned int kKernelLibraryArgmaxGroups = {ARGMAX_GROUPS}u;",
            "inline constexpr bool kKernelLibraryMlxSteelEnabled = "
            f"{str(library.mlx_steel_enabled).lower()};",
            "inline constexpr char kKernelLibraryMlxSteelKernelName[] = "
            f"\"{MLX_STEEL_KERNEL_NAME}\";",
            "inline constexpr char kKernelLibraryMlxSteelDenseKernelName[] = "
            f"\"{MLX_STEEL_DENSE_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelGdnFused2KernelName[] = "
            f"\"{MLX_STEEL_GDN_FUSED2_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelGdnBm64KernelName[] = "
            f"\"{MLX_STEEL_GDN_BM64_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelGdnBm64Wm2Wn2KernelName[] = "
            f"\"{MLX_STEEL_GDN_BM64_WM2_WN2_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelGdnBm64Bk64KernelName[] = "
            f"\"{MLX_STEEL_GDN_BM64_BK64_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelGdnBm48KernelName[] = "
            f"\"{MLX_STEEL_GDN_BM48_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelGdnBm96KernelName[] = "
            f"\"{MLX_STEEL_GDN_BM96_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelGdnBm128KernelName[] = "
            f"\"{MLX_STEEL_GDN_BM128_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelGdnBn64KernelName[] = "
            f"\"{MLX_STEEL_GDN_BN64_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelGdnBk64KernelName[] = "
            f"\"{MLX_STEEL_GDN_BK64_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelRoutedDownKernelName[] = "
            f"\"{MLX_STEEL_ROUTED_DOWN_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelRoutedUpgateKernelName[] = "
            f"\"{MLX_STEEL_ROUTED_UPGATE_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelRoutedBm32DownKernelName[] = "
            f"\"{MLX_STEEL_ROUTED_BM32_DOWN_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelRoutedBm32UpgateKernelName[] = "
            f"\"{MLX_STEEL_ROUTED_BM32_UPGATE_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelRoutedBk64DownKernelName[] = "
            f"\"{MLX_STEEL_ROUTED_BK64_DOWN_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelRoutedBk64UpgateKernelName[] = "
            f"\"{MLX_STEEL_ROUTED_BK64_UPGATE_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelRoutedBn64DownKernelName[] = "
            f"\"{MLX_STEEL_ROUTED_BN64_DOWN_KERNEL_NAME}\";",
            "inline constexpr char "
            "kKernelLibraryMlxSteelRoutedBn64UpgateKernelName[] = "
            f"\"{MLX_STEEL_ROUTED_BN64_UPGATE_KERNEL_NAME}\";",
            "",
            _render_source_parts(library.source),
            "",
            "} // namespace tatara::backend::metal::generated",
            "",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the Tatara kernel library header")
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--kernel-directory", required=True, type=Path)
    parser.add_argument("--mlx-steel-source-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    try:
        package_bytes = arguments.package.read_bytes()
        artifact_bytes = arguments.artifact.read_bytes()
        artifact = parse_manifest(artifact_bytes.decode("utf-8"))
        plan = parse_model_plan(
            package_bytes.decode("utf-8"),
            artifact,
            arguments.artifact.name,
            hashlib.sha256(package_bytes).hexdigest(),
            hashlib.sha256(artifact_bytes).hexdigest(),
        )
        library = build_kernel_library(
            plan,
            arguments.kernel_directory,
            arguments.mlx_steel_source_root,
        )
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(render_kernel_library_header(library))
    except (
        ArtifactManifestError,
        ModelPlanGenerationError,
        KernelLibraryGenerationError,
        OSError,
    ) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
