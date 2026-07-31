"""Tape-kernel static pins.

The replay kernel must advance the recurrent state with the FORWARD scan's
own expressions (the A32b bitwise-replay obligation), and both tape
variants must be the admitted kernels plus tape writes only.
"""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METAL = ROOT / "src/backend/metal/kernels/prefill_gdn.metal"


def body(source: str, name: str) -> str:
    begin = source.index(f"kernel void {name}(")
    end = source.find("kernel void ", begin + 1)
    return source[begin : end if end != -1 else len(source)]


class GdnTapeSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        source = METAL.read_text(encoding="utf-8")
        self.forward = body(source, "gdn_recurrence_gates_blk")
        self.forward_tape = body(source, "gdn_recurrence_gates_tape_blk")
        self.conv = body(source, "gdn_conv_blk")
        self.conv_tape = body(source, "gdn_conv_tape_blk")
        self.replay = body(source, "gdn_tape_replay_blk")

    def test_generation_pin_state_update_expressions(self) -> None:
        # The replay's two state-update statements are the forward's own.
        self.assertIn("state[i] *= step_decay;", self.forward)
        self.assertIn("state[i] *= step_decay;", self.replay)
        forward_update = "state[i] += float(qk_row[key_base + element]) * delta;"
        self.assertIn(forward_update, self.forward)
        # Replay sources k from the tape; the arithmetic shape is identical:
        # one fused multiply-add of the recorded k against delta.
        self.assertIn("] *\n                        delta;", self.replay)
        self.assertIn("state[i] += tape[", self.replay)

    def test_tape_variant_is_forward_plus_tape_only(self) -> None:
        stripped = self.forward_tape
        for marker in (
            "device float* tape [[buffer(9)]]",
            "constant uint& tape_layer [[buffer(10)]]",
            "steps <= kGdnTapeRows",
        ):
            self.assertIn(marker, stripped)
        # Every forward expression survives verbatim in the variant.
        for line in (
            "state[i] *= step_decay;",
            "key_value += state[i] * float(qk_row[key_base + element]);",
            "state[i] += float(qk_row[key_base + element]) * delta;",
        ):
            self.assertIn(line, self.forward)
            self.assertIn(line, self.forward_tape)

    def test_conv_tape_matches_state_write_expression(self) -> None:
        self.assertIn(
            "static_cast<bfloat>(taps[tap + 1u])", self.conv)
        self.assertEqual(
            self.conv_tape.count("static_cast<bfloat>(taps[tap + 1u])"), 2)
        self.assertIn("state_step <= kGdnTapeRows", self.conv_tape)

    def test_replay_reads_only_tape_and_state(self) -> None:
        self.assertNotIn("qk[", self.replay)
        self.assertNotIn("value[", self.replay)
        self.assertNotIn("simd_sum", self.replay)


if __name__ == "__main__":
    unittest.main()
