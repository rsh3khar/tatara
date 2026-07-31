import hashlib
import re
import unittest
from pathlib import Path

from dataclasses import replace

from tatara.artifact_manifest import parse_manifest
from tatara.qwen36_plan_generation import parse_model_plan
from tools.generate_kernel_library import (
    KERNEL_FILE_NAMES,
    MAX_STRING_LITERAL_CHARS,
    NATIVE_DENSE_QGEMM_N1_ACCUMULATOR_ELEMENTS,
    NATIVE_DENSE_QGEMM_N1_METADATA_PAIRS_PER_GROUP_TILE,
    NATIVE_DENSE_QGEMM_N1_PACKED_WORDS_PER_TILE,
    NATIVE_DENSE_QGEMM_N1_PACKED_WORDS_PER_TILE_ROW,
    NATIVE_DENSE_QGEMM_N1_REDUCTION_COLUMNS,
    NATIVE_DENSE_QGEMM_N1_SIMDGROUPS,
    NATIVE_DENSE_QGEMM_N1_SIMDGROUP_GRID_COLUMNS,
    NATIVE_DENSE_QGEMM_N1_SIMDGROUP_GRID_ROWS,
    NATIVE_DENSE_QGEMM_N1_STAGE_ROW_PADDING,
    NATIVE_DENSE_QGEMM_N1_THREADS,
    NATIVE_DENSE_QGEMM_N1_THREADGROUP_MEMORY_BYTES,
    NATIVE_DENSE_QGEMM_N1_TILE_COLUMNS,
    NATIVE_DENSE_QGEMM_N1_TILE_ROWS,
    NATIVE_ROUTED_QGEMM_R1_ACCUMULATOR_ELEMENTS,
    NATIVE_ROUTED_QGEMM_R1_FUSED_ACCUMULATOR_ELEMENTS,
    NATIVE_ROUTED_QGEMM_R1_REDUCTION_COLUMNS,
    NATIVE_ROUTED_QGEMM_R1_SIMDGROUPS,
    NATIVE_ROUTED_QGEMM_R1_SINGLE_ACCUMULATOR_ELEMENTS,
    NATIVE_ROUTED_QGEMM_R1_TASK_BYTES,
    NATIVE_ROUTED_QGEMM_R1_TASK_CAPACITY,
    NATIVE_ROUTED_QGEMM_R1_TASK_STATUS_VALUES,
    NATIVE_ROUTED_QGEMM_R1_THREADGROUP_MEMORY_BYTES,
    NATIVE_ROUTED_QGEMM_R1_THREADS,
    NATIVE_ROUTED_QGEMM_R1_TILE_COLUMNS,
    NATIVE_ROUTED_QGEMM_R1_TILE_ROWS,
    NATIVE_ROUTED_QGEMM_R2_DOWN_THREADGROUP_MEMORY_BYTES,
    NATIVE_ROUTED_QGEMM_R2_FUSED_THREADGROUP_MEMORY_BYTES,
    RAW_STRING_DELIMITER,
    KernelLibraryGenerationError,
    build_kernel_library,
    chunk_source,
    render_kernel_library_header,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_PATH = REPOSITORY_ROOT / "catalog/model_packages/qwen36-35b-a3b/model.toml"
ARTIFACT_PATH = REPOSITORY_ROOT / "catalog/model_packages/qwen36-35b-a3b/artifact.toml"
KERNEL_DIRECTORY = REPOSITORY_ROOT / "src/backend/metal/kernels"


def load_plan():
    package_bytes = PACKAGE_PATH.read_bytes()
    artifact_bytes = ARTIFACT_PATH.read_bytes()
    return parse_model_plan(
        package_bytes.decode("utf-8"),
        parse_manifest(artifact_bytes.decode("utf-8")),
        ARTIFACT_PATH.name,
        hashlib.sha256(package_bytes).hexdigest(),
        hashlib.sha256(artifact_bytes).hexdigest(),
    )


class KernelLibraryGenerationTest(unittest.TestCase):
    def test_native_routed_qgemm_treatment_resources_are_frozen(self):
        self.assertEqual(
            NATIVE_ROUTED_QGEMM_R1_FUSED_ACCUMULATOR_ELEMENTS,
            1024,
        )
        self.assertEqual(
            NATIVE_ROUTED_QGEMM_R1_SINGLE_ACCUMULATOR_ELEMENTS,
            512,
        )
        self.assertEqual(
            NATIVE_ROUTED_QGEMM_R1_THREADGROUP_MEMORY_BYTES,
            0,
        )
        self.assertEqual(
            NATIVE_ROUTED_QGEMM_R1_ACCUMULATOR_ELEMENTS,
            NATIVE_ROUTED_QGEMM_R1_FUSED_ACCUMULATOR_ELEMENTS,
        )
        self.assertEqual(
            NATIVE_ROUTED_QGEMM_R2_FUSED_THREADGROUP_MEMORY_BYTES,
            11584,
        )
        self.assertEqual(
            NATIVE_ROUTED_QGEMM_R2_DOWN_THREADGROUP_MEMORY_BYTES,
            6464,
        )
    def test_generation_is_deterministic(self):
        plan = load_plan()
        first = render_kernel_library_header(build_kernel_library(plan, KERNEL_DIRECTORY))
        second = render_kernel_library_header(build_kernel_library(plan, KERNEL_DIRECTORY))
        self.assertEqual(first, second)

    def test_prelude_matches_plan_dimensions(self):
        plan = load_plan()
        library = build_kernel_library(plan, KERNEL_DIRECTORY)
        self.assertEqual(library.hidden, plan.hidden)
        self.assertEqual(library.group_size, plan.group_size)
        self.assertEqual(library.moe_expert_dimension, plan.expert_dimension)
        self.assertIn(f"constant uint kHiddenDimension = {plan.hidden}u;", library.source)
        self.assertIn(f"constant uint kQ4GroupSize = {plan.group_size}u;", library.source)
        self.assertIn(
            f"constant uint kQuantWordsPerRow = {plan.hidden // 8}u;", library.source
        )
        self.assertIn(
            f"constant uint kGroupsPerRow = {plan.hidden // plan.group_size}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1TileRows = "
            f"{NATIVE_DENSE_QGEMM_N1_TILE_ROWS}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1TileColumns = "
            f"{NATIVE_DENSE_QGEMM_N1_TILE_COLUMNS}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1ReductionColumns = "
            f"{NATIVE_DENSE_QGEMM_N1_REDUCTION_COLUMNS}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1Simdgroups = "
            f"{NATIVE_DENSE_QGEMM_N1_SIMDGROUPS}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1Threads = "
            f"{NATIVE_DENSE_QGEMM_N1_THREADS}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1SimdgroupGridRows = "
            f"{NATIVE_DENSE_QGEMM_N1_SIMDGROUP_GRID_ROWS}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1SimdgroupGridColumns = "
            f"{NATIVE_DENSE_QGEMM_N1_SIMDGROUP_GRID_COLUMNS}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1StageRowPadding = "
            f"{NATIVE_DENSE_QGEMM_N1_STAGE_ROW_PADDING}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1ThreadgroupMemoryBytes = "
            f"{NATIVE_DENSE_QGEMM_N1_THREADGROUP_MEMORY_BYTES}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1AccumulatorElements = "
            f"{NATIVE_DENSE_QGEMM_N1_ACCUMULATOR_ELEMENTS}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1PackedWordsPerTileRow = "
            f"{NATIVE_DENSE_QGEMM_N1_PACKED_WORDS_PER_TILE_ROW}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1PackedWordsPerTile = "
            f"{NATIVE_DENSE_QGEMM_N1_PACKED_WORDS_PER_TILE}u;",
            library.source,
        )
        self.assertIn(
            "constant uint kNativeDenseQgemmN1MetadataPairsPerGroupTile = "
            f"{NATIVE_DENSE_QGEMM_N1_METADATA_PAIRS_PER_GROUP_TILE}u;",
            library.source,
        )
        routed_profile = {
            "TileRows": NATIVE_ROUTED_QGEMM_R1_TILE_ROWS,
            "TileColumns": NATIVE_ROUTED_QGEMM_R1_TILE_COLUMNS,
            "ReductionColumns": (
                NATIVE_ROUTED_QGEMM_R1_REDUCTION_COLUMNS
            ),
            "Simdgroups": NATIVE_ROUTED_QGEMM_R1_SIMDGROUPS,
            "Threads": NATIVE_ROUTED_QGEMM_R1_THREADS,
            "TaskBytes": NATIVE_ROUTED_QGEMM_R1_TASK_BYTES,
            "TaskCapacity": NATIVE_ROUTED_QGEMM_R1_TASK_CAPACITY,
            "FusedAccumulatorElements": (
                NATIVE_ROUTED_QGEMM_R1_FUSED_ACCUMULATOR_ELEMENTS
            ),
            "SingleAccumulatorElements": (
                NATIVE_ROUTED_QGEMM_R1_SINGLE_ACCUMULATOR_ELEMENTS
            ),
            "ThreadgroupMemoryBytes": (
                NATIVE_ROUTED_QGEMM_R1_THREADGROUP_MEMORY_BYTES
            ),
            "AccumulatorElements": (
                NATIVE_ROUTED_QGEMM_R1_ACCUMULATOR_ELEMENTS
            ),
        }
        for suffix, value in routed_profile.items():
            self.assertIn(
                f"constant uint kNativeRoutedQgemmR1{suffix} = "
                f"{value}u;",
                library.source,
            )

    def test_every_kernel_file_is_embedded(self):
        plan = load_plan()
        library = build_kernel_library(plan, KERNEL_DIRECTORY)
        for name in KERNEL_FILE_NAMES:
            self.assertIn(f"// kernel file: {name}", library.source)
        self.assertIn("kernel void embed_row_q4", library.source)
        self.assertIn(
            "kernel void native_dense_qgemm_q4_bf16_n1", library.source
        )
        self.assertIn("kernel void rms_only", library.source)
        self.assertIn("kernel void adjudicate_rsqrt", library.source)
        self.assertIn("kernel void adjudicate_bfloat_multiply", library.source)
        self.assertIn("kernel void adjudicate_rms_sum", library.source)
        self.assertRegex(library.source, r"kernel\s+void\s+adjudicate_q4_row_v\(")
        for name in ("attn_project", "attn_qk_rope", "attention_decode",
                     "attention_decode_scores_gqa4",
                     "attention_decode_scores_gqa4_simdreduce",
                     "attention_decode_scores_gqa8",
                     "attention_decode_values_gqa8",
                     "attention_decode_values_gqa8_t512",
                     "attention_decode_combine"):
            self.assertRegex(library.source, rf"kernel\s+void\s+{name}\(")
        self.assertIn("constant uint kAttnProjectionRows = 9216u;", library.source)
        self.assertIn("constant uint kAttnVRowOffset = 8704u;", library.source)
        self.assertIn("constant uint kAttnRopePairs = 32u;", library.source)
        self.assertIn("constant float kAttnScale = 0.0625f;", library.source)
        for name in ("adjudicate_rope_trig", "adjudicate_f32_exp", "adjudicate_tg_trees"):
            self.assertRegex(library.source, rf"kernel\s+void\s+{name}\(")
        for name in ("residual_rms", "router_q8", "router_select", "grouped_upgate",
                     "grouped_down_res", "adjudicate_q4p_row", "adjudicate_q4p_row_e",
                     "adjudicate_q8_row", "adjudicate_down_total", "adjudicate_f32_divide",
                     "lmhead_q4", "logits_argmax_stage1", "logits_argmax_stage2"):
            self.assertRegex(library.source, rf"kernel\s+void\s+{name}\(")
        for name in (
            "embed_rows_q4",
            "rms_blk",
            "residual_blk",
            "gdn_project_blk",
            "gdn_conv_blk",
            "gdn_gates_blk",
            "gdn_recurrence_blk",
            "gdn_recurrence_gates_blk",
            "gdn_gate_norm_blk",
            "attn_project_blk",
            "attn_qk_rope_blk",
            "attention_partial_blk",
            "attention_combine_blk",
            "attention_staged_scores_blk",
            "attention_staged_softmax_blk",
            "attention_staged_values_blk",
            "attention_streaming_blk",
            "outproj_blk",
            "router_q8_block",
            "router_select_block",
            "router_select_block_parallel",
            "expert_union",
            "block_upgate",
            "block_down_partial",
            "block_down_combine",
            "native_routed_qgemm_r1_fused_upgate_swiglu",
            "native_routed_qgemm_r1_gate",
            "native_routed_qgemm_r1_up_swiglu",
            "native_routed_qgemm_r1_down_partial",
            "native_routed_qgemm_r1_build_tasks",
        ):
            self.assertRegex(library.source, rf"kernel\s+void\s+{name}\(")
        serial_router = library.source[
            library.source.index("kernel void router_select_block("):
            library.source.index("kernel void router_select_block_parallel(")
        ]
        parallel_router = library.source[
            library.source.index("kernel void router_select_block_parallel("):
            library.source.index("kernel void expert_union(")
        ]
        self.assertIn("indices[index] = index;", serial_router)
        self.assertIn("reduction[index + offset] > reduction[index]", serial_router)
        self.assertNotIn("values[index] > best", serial_router)
        self.assertNotIn("reduction[expert + offset] ==", parallel_router)
        self.assertIn("inline float q4_dot_packed", library.source)
        self.assertIn("inline float q8_dot", library.source)
        self.assertIn("constant uint kMoeExperts = 256u;", library.source)
        self.assertIn("constant uint kMoeExpertDimension = 512u;", library.source)
        self.assertIn("constant uint kVocabularyRows = 248320u;", library.source)
        self.assertIn("inline float q4_dot", library.source)
        for name in ("gdn_project", "gdn_prepare", "gdn_recurrence", "gdn_gate_norm",
                     "gdn_outproj"):
            self.assertRegex(library.source, rf"kernel\s+void\s+{name}\(")
        self.assertIn("constant uint kGdnProjectionRows = 12352u;", library.source)
        self.assertIn("constant uint kGdnBRowOffset = 12288u;", library.source)
        self.assertIn("constant uint kGdnARowOffset = 12320u;", library.source)
        self.assertIn("constant float kGdnKeyScale = 0.08837890625f;", library.source)
        header = render_kernel_library_header(library)
        self.assertIn("kKernelLibraryGdnKeyHeads = 16u;", header)
        self.assertIn("kKernelLibraryGdnValueHeads = 32u;", header)
        self.assertIn("kKernelLibraryGdnHeadDimension = 128u;", header)
        self.assertIn("kKernelLibraryGdnKeyScale = 0.08837890625f;", header)
        self.assertIn("kKernelLibraryMoeExperts = 256u;", header)
        self.assertIn("kKernelLibraryMoeActiveExperts = 8u;", header)
        self.assertIn("kKernelLibraryMoeExpertDimension = 512u;", header)
        self.assertIn("kKernelLibraryVocabulary = 248320u;", header)
        self.assertIn("kKernelLibraryArgmaxGroups = 256u;", header)
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1TileRows = 32u;", header
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1TileColumns = 32u;", header
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1ReductionColumns = 32u;",
            header,
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1Simdgroups = 4u;", header
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1Threads = 128u;", header
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1ThreadgroupMemoryBytes = 5120u;",
            header,
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1SimdgroupGridRows = 2u;",
            header,
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1SimdgroupGridColumns = 2u;",
            header,
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1StageRowPadding = 8u;",
            header,
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1AccumulatorElements = 1024u;",
            header,
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1PackedWordsPerTileRow = 4u;",
            header,
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1PackedWordsPerTile = 128u;",
            header,
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1MetadataPairsPerGroupTile = 32u;",
            header,
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1DenseScope = true;", header
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1RaggedScope = false;", header
        )
        self.assertIn(
            "kKernelLibraryNativeDenseQgemmN1HostReachable = true;", header
        )
        routed_header_profile = {
            "TileRows": NATIVE_ROUTED_QGEMM_R1_TILE_ROWS,
            "TileColumns": NATIVE_ROUTED_QGEMM_R1_TILE_COLUMNS,
            "ReductionColumns": (
                NATIVE_ROUTED_QGEMM_R1_REDUCTION_COLUMNS
            ),
            "Simdgroups": NATIVE_ROUTED_QGEMM_R1_SIMDGROUPS,
            "Threads": NATIVE_ROUTED_QGEMM_R1_THREADS,
            "TaskBytes": NATIVE_ROUTED_QGEMM_R1_TASK_BYTES,
            "TaskCapacity": NATIVE_ROUTED_QGEMM_R1_TASK_CAPACITY,
            "FusedAccumulatorElements": (
                NATIVE_ROUTED_QGEMM_R1_FUSED_ACCUMULATOR_ELEMENTS
            ),
            "SingleAccumulatorElements": (
                NATIVE_ROUTED_QGEMM_R1_SINGLE_ACCUMULATOR_ELEMENTS
            ),
            "ThreadgroupMemoryBytes": (
                NATIVE_ROUTED_QGEMM_R1_THREADGROUP_MEMORY_BYTES
            ),
            "AccumulatorElements": (
                NATIVE_ROUTED_QGEMM_R1_ACCUMULATOR_ELEMENTS
            ),
        }
        for suffix, value in routed_header_profile.items():
            self.assertIn(
                "kKernelLibraryNativeRoutedQgemmR1"
                f"{suffix} = {value}u;",
                header,
            )
        self.assertIn(
            "kKernelLibraryNativeRoutedQgemmR1DenseScope = false;",
            header,
        )
        self.assertIn(
            "kKernelLibraryNativeRoutedQgemmR1RaggedScope = true;",
            header,
        )
        self.assertIn(
            "kKernelLibraryNativeRoutedQgemmR1HostReachable = true;",
            header,
        )

    def test_expert_dimension_comes_from_the_package(self):
        plan = load_plan()
        wider = replace(plan, expert_dimension=1024)

        library = build_kernel_library(wider, KERNEL_DIRECTORY)

        self.assertIn("constant uint kMoeExpertDimension = 1024u;", library.source)
        self.assertIn(
            "kKernelLibraryMoeExpertDimension = 1024u;",
            render_kernel_library_header(library),
        )

    def test_expert_dimension_must_stay_chunk_and_group_aligned(self):
        plan = load_plan()

        with self.assertRaisesRegex(KernelLibraryGenerationError, "chunk-aligned"):
            build_kernel_library(replace(plan, expert_dimension=768), KERNEL_DIRECTORY)

    def test_quantization_group_must_match_the_kernel_family(self):
        plan = load_plan()

        with self.assertRaisesRegex(KernelLibraryGenerationError, "expected 64"):
            build_kernel_library(replace(plan, group_size=32), KERNEL_DIRECTORY)

    def test_gdn_head_dimension_must_match_the_exact_family(self):
        plan = load_plan()

        with self.assertRaisesRegex(KernelLibraryGenerationError, "GDN head dimension"):
            build_kernel_library(replace(plan, state_dimension=256), KERNEL_DIRECTORY)

    def test_attention_geometry_must_match_the_exact_family(self):
        plan = load_plan()

        with self.assertRaisesRegex(KernelLibraryGenerationError, "Attention head dimension"):
            build_kernel_library(replace(plan, head_dimension=128), KERNEL_DIRECTORY)
        with self.assertRaisesRegex(KernelLibraryGenerationError, "GQA ratio"):
            build_kernel_library(replace(plan, query_heads=8), KERNEL_DIRECTORY)

    def test_prefill_slot_field_is_derived_from_active_experts(self):
        plan = load_plan()

        top_four = build_kernel_library(
            replace(plan, active_experts=4), KERNEL_DIRECTORY
        )
        top_eight = build_kernel_library(plan, KERNEL_DIRECTORY)
        self.assertIn(
            "constant uint kPrefillPackedSlotBits = 3u;",
            top_four.source,
        )
        self.assertIn(
            "constant uint kPrefillPackedSlotBits = 4u;",
            top_eight.source,
        )

    def test_expert_count_comes_from_the_package(self):
        """router_select sizes its threadgroup arrays and reduction trees from
        kMoeExperts, so a different power-of-two width must generate cleanly
        rather than be pinned to the first package's 256."""
        plan = load_plan()

        library = build_kernel_library(
            replace(plan, experts=128, active_experts=4), KERNEL_DIRECTORY
        )

        self.assertIn("constant uint kMoeExperts = 128u;", library.source)
        self.assertIn("constant uint kMoeActiveExperts = 4u;", library.source)
        self.assertIn("constant uint kMoeSlotCount = 5u;", library.source)
        self.assertIn("threadgroup float values[kMoeExperts];", library.source)
        self.assertIn("for (uint off = kMoeExperts / 2u; off; off >>= 1u) {", library.source)

    def test_native_dense_qgemm_source_is_plan_topology_independent(self):
        plan = load_plan()
        current = build_kernel_library(plan, KERNEL_DIRECTORY).source
        synthetic = build_kernel_library(
            replace(plan, experts=64, active_experts=4), KERNEL_DIRECTORY
        ).source

        marker = "// kernel file: native_dense_qgemm.metal\n"
        next_marker = "// kernel file: prefill_dense.metal\n"
        current_kernel = current[
            current.index(marker) : current.index(next_marker)
        ]
        synthetic_kernel = synthetic[
            synthetic.index(marker) : synthetic.index(next_marker)
        ]
        self.assertEqual(current_kernel, synthetic_kernel)
        self.assertNotRegex(
            current_kernel.lower(), r"qwen|expert|top.?k|route|fusion"
        )

    def test_native_routed_qgemm_source_is_plan_topology_independent(self):
        plan = load_plan()
        current = build_kernel_library(plan, KERNEL_DIRECTORY).source
        synthetic = build_kernel_library(
            replace(
                plan,
                experts=128,
                active_experts=4,
                expert_dimension=1024,
            ),
            KERNEL_DIRECTORY,
        ).source

        marker = "struct NativeRoutedQgemmR1Task"
        next_marker = "// kernel file: head.metal\n"
        current_kernel = current[
            current.index(marker) : current.index(next_marker)
        ]
        synthetic_kernel = synthetic[
            synthetic.index(marker) : synthetic.index(next_marker)
        ]
        self.assertEqual(current_kernel, synthetic_kernel)
        for qwen_topology_literal in (
            "256u",
            "512u",
            "2048u",
            "4608u",
        ):
            self.assertNotIn(qwen_topology_literal, current_kernel)
        for generated_dimension in (
            "kHiddenDimension",
            "kMoeExperts",
            "kMoeActiveExperts",
            "kMoeExpertDimension",
            "kMoeSlotCount",
            "kPrefillPackedSlotBits",
        ):
            self.assertIn(generated_dimension, current_kernel)
        self.assertIn("include_shared_expert", current_kernel)
        self.assertIn("const bool shared", current_kernel)
        self.assertIn(
            "task.expert_index == kMoeExperts",
            current_kernel,
        )
        self.assertIn(
            "slot == kMoeActiveExperts",
            current_kernel,
        )
        self.assertIn(
            "task.expert_index >= kMoeExperts",
            current_kernel,
        )

    def test_routed_task_producer_regenerates_for_alternate_moe_topology(self):
        plan = load_plan()
        alternate_plan = replace(
            plan,
            experts=128,
            active_experts=4,
            expert_dimension=1024,
        )
        current = build_kernel_library(plan, KERNEL_DIRECTORY).source
        alternate = build_kernel_library(
            alternate_plan, KERNEL_DIRECTORY
        ).source
        marker = "// kernel file: routed_qgemm_tasks.metal\n"
        next_marker = "// kernel file: head.metal\n"
        current_producer = current[
            current.index(marker) : current.index(next_marker)
        ]
        alternate_producer = alternate[
            alternate.index(marker) : alternate.index(next_marker)
        ]

        self.assertEqual(current_producer, alternate_producer)
        self.assertIn("constant uint kMoeExperts = 128u;", alternate)
        self.assertIn("constant uint kMoeActiveExperts = 4u;", alternate)
        self.assertIn(
            "kernel void native_routed_qgemm_r1_build_tasks(",
            alternate_producer,
        )
        for name, value in NATIVE_ROUTED_QGEMM_R1_TASK_STATUS_VALUES:
            self.assertIn(
                "constant uint kNativeRoutedQgemmR1TaskStatus"
                f"{name} = {value}u;",
                alternate,
            )

    def test_no_emitted_string_literal_exceeds_the_cxx_bound(self):
        """Tatara builds -Werror -Woverlength-strings, so an oversized literal
        is a build break, not a warning. Chunking removes the ceiling instead
        of leaving it as a trap for the next kernel that grows."""
        plan = load_plan()

        header = render_kernel_library_header(build_kernel_library(plan, KERNEL_DIRECTORY))

        literals = re.findall(
            rf'R"{RAW_STRING_DELIMITER}\((.*?)\){RAW_STRING_DELIMITER}"', header, re.S
        )
        self.assertTrue(literals)
        for literal in literals:
            self.assertLess(len(literal), MAX_STRING_LITERAL_CHARS)

    def test_chunking_reproduces_the_source_exactly(self):
        plan = load_plan()
        library = build_kernel_library(plan, KERNEL_DIRECTORY)

        self.assertEqual("".join(chunk_source(library.source)), library.source)

    def test_router_width_must_be_a_power_of_two(self):
        """The reduction halves the width each pass, so a non-power-of-two
        count silently drops a slot -- with 6 experts the off=3 pass folds
        3..5 into 0..2 and the off=1 pass never merges slot 2."""
        plan = load_plan()

        with self.assertRaisesRegex(KernelLibraryGenerationError, "power of two"):
            build_kernel_library(replace(plan, experts=192), KERNEL_DIRECTORY)

    def test_router_width_must_fit_one_threadgroup(self):
        """router_select dispatches {groups = 1, threads = experts} and a Metal
        threadgroup tops out at 1024 threads."""
        plan = load_plan()

        with self.assertRaisesRegex(KernelLibraryGenerationError, "router dispatch range"):
            build_kernel_library(replace(plan, experts=2048), KERNEL_DIRECTORY)

    def test_header_contains_no_checkout_path_and_safe_delimiter(self):
        plan = load_plan()
        header = render_kernel_library_header(build_kernel_library(plan, KERNEL_DIRECTORY))
        self.assertNotIn(str(REPOSITORY_ROOT), header)
        # The source is emitted as separately named literals joined at runtime,
        # so the delimiter appears once per part rather than exactly once. What
        # must hold is that every opener is closed and the delimiter never
        # occurs inside the source it delimits.
        opens = header.count(f'R"{RAW_STRING_DELIMITER}(')
        self.assertEqual(opens, header.count(f'){RAW_STRING_DELIMITER}"'))
        self.assertGreaterEqual(opens, 1)


if __name__ == "__main__":
    unittest.main()
