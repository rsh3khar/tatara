"""Static safety pins for the draft Metal family.

Freezes the crash-safety surface of src/backend/metal/kernels/draft.metal
before any dispatch: kernel inventory and order, per-kernel bounds guards,
buffer-index integrity, threadgroup-memory bounds, register-state sizing,
and the generator/CMake/probe wiring. Every pin is a named mutation kill:
weakening a guard or drifting the wiring fails here, not on the device.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "src/backend/metal/kernels/draft.metal"
GENERATOR = ROOT / "tools/generate_kernel_library.py"
CMAKE = ROOT / "CMakeLists.txt"
PROBE = ROOT / "tools/native/prefill_library_probe.cpp"

KERNEL_NAMES = (
    "draft_dense_bf16",
    "draft_gemm16_bf16",
    "draft_rms_rows",
    "draft_rope128",
    "draft_block_attention",
    "draft_swiglu",
    "draft_residual_add",
)

REQUIRED_GUARDS = {
    "draft_dense_bf16": (
        "simdgroup_width != 32u",
        "threadgroup_shape.x != kDraftDenseThreads",
        "rows > kDraftMaxBlockRows",
        "(reduction & 3u) != 0u",
        "column >= columns",
    ),
    "draft_gemm16_bf16": (
        "simdgroup_width != 32u",
        "threadgroup_shape.x != kDraftGemmThreads",
        "rows != kDraftMaxBlockRows",
        "(reduction & 7u) != 0u",
        "(columns % kDraftGemmColumnsPerSimdgroup) != 0u",
        "column_base >= columns",
    ),
    "draft_rms_rows": (
        "threads * 4u != dimension",
        "threads / 32u > kDraftRmsMaxSimdgroups",
        "group.x >= rows",
    ),
    "draft_rope128": (
        "threadgroup_shape.x != kDraftHeadDim / 2u",
        "group.x >= heads",
        "group.y >= rows",
    ),
    "draft_block_attention": (
        "group.x >= kDraftKvHeads",
        "block_rows > kDraftMaxBlockRows",
        "key_count > key_capacity",
        "threadgroup_shape.x != kDraftAttnThreads",
    ),
    "draft_swiglu": (
        "position.y >= rows",
        "position.x >= dimension",
    ),
    "draft_residual_add": (
        "position.y >= rows",
        "position.x >= dimension",
    ),
}


def kernel_bodies(source: str) -> dict[str, str]:
    bodies: dict[str, str] = {}
    for index, name in enumerate(KERNEL_NAMES):
        begin = source.index(f"kernel void {name}(")
        end = (
            source.index(f"kernel void {KERNEL_NAMES[index + 1]}(")
            if index + 1 < len(KERNEL_NAMES)
            else len(source)
        )
        bodies[name] = source[begin:end]
    return bodies


class DraftKernelSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = KERNELS.read_text(encoding="utf-8")
        self.bodies = kernel_bodies(self.source)

    def test_kernel_inventory_and_order(self) -> None:
        found = re.findall(r"kernel void (\w+)\(", self.source)
        self.assertEqual(tuple(found), KERNEL_NAMES)

    def test_every_bounds_guard_is_present(self) -> None:
        for name, guards in REQUIRED_GUARDS.items():
            body = self.bodies[name]
            for guard in guards:
                self.assertIn(guard, body, f"{name} lost guard: {guard}")

    def test_buffer_indices_are_unique_and_contiguous(self) -> None:
        for name, body in self.bodies.items():
            header = body[: body.index("{")]
            indices = [int(value) for value in re.findall(r"buffer\((\d+)\)", header)]
            self.assertEqual(
                sorted(indices), list(range(len(indices))),
                f"{name} buffer indices must be unique and contiguous",
            )

    def test_attention_state_is_simdgroup_local(self) -> None:
        body = self.bodies["draft_block_attention"]
        self.assertIn("float row_max[kDraftAttnLocalPairs]", body)
        self.assertIn("float row_sum[kDraftAttnLocalPairs]", body)
        self.assertIn("float acc[kDraftAttnLocalPairs][4]", body)
        self.assertIn("const uint local = (pair - simdgroup) / simdgroups;", body)
        self.assertNotIn("row_max[pair]", body)
        self.assertNotIn("acc[pair]", body)

    def test_attention_threadgroup_memory_is_bounded(self) -> None:
        body = self.bodies["draft_block_attention"]
        self.assertIn(
            "threadgroup bfloat key_tile[kDraftAttnKeyTile * kDraftHeadDim]", body)
        self.assertIn(
            "threadgroup bfloat value_tile[kDraftAttnKeyTile * kDraftHeadDim]", body)
        self.assertIn("threadgroup int position_tile[kDraftAttnKeyTile]", body)
        # 32 * 128 * 2 bytes * 2 tiles + 32 * 4 = 16,512 bytes, under the
        # 32 KiB device guarantee with margin.
        self.assertLessEqual(32 * 128 * 2 * 2 + 32 * 4, 32 * 1024)

    def test_sliding_mask_rule_is_exact(self) -> None:
        body = self.bodies["draft_block_attention"]
        self.assertIn("query_position >= key_position", body)
        self.assertIn("key_position + int(kDraftWindowPositions)", body)
        self.assertIn("if (full_attention == 0u)", body)

    def test_no_unbounded_loops_or_atomics(self) -> None:
        self.assertNotIn("while (true)", self.source)
        self.assertNotIn("atomic_", self.source)

    def test_generator_constants_and_wiring(self) -> None:
        generator = GENERATOR.read_text(encoding="utf-8")
        self.assertIn('"draft.metal",', generator)
        self.assertIn("DRAFT_MAX_BLOCK_ROWS = 16", generator)
        self.assertIn("DRAFT_GEMM_THREADS = 128", generator)
        self.assertIn("DRAFT_GEMM_COLUMNS_PER_SIMDGROUP = 16", generator)
        self.assertIn("DRAFT_ATTENTION_KEY_TILE = 32", generator)
        self.assertIn("DRAFT_WINDOW_POSITIONS = 4096", generator)
        # Local-pair register bound: (16 rows * 4 group width) / 4 simdgroups.
        self.assertIn(
            "{(DRAFT_MAX_BLOCK_ROWS * (DRAFT_QUERY_HEADS // DRAFT_KV_HEADS))"
            " // (DRAFT_ATTENTION_THREADS // 32)}u;",
            generator,
        )
        self.assertIn("draft.metal", CMAKE.read_text(encoding="utf-8"))
        probe = PROBE.read_text(encoding="utf-8")
        for name in KERNEL_NAMES:
            self.assertIn(f'"{name}"', probe)


if __name__ == "__main__":
    unittest.main()
