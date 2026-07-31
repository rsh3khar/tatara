"""Pins the guarded physical equivalence arm to bounded graph bands."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "tools/native/block_prefill_probe.cpp"


class BandedPrefillProbeSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SOURCE.read_text(encoding="utf-8")

    def test_banded_policy_is_explicit_and_three_lanes_is_scratch_only(self) -> None:
        self.assertIn(
            '"layer2048fast-steel-full-graph-banded3"', self.source
        )
        self.assertIn(
            "constexpr std::uint32_t kGraphBandedScratchLanes = 3;",
            self.source,
        )
        self.assertIn(
            "policy.command_graph_banded = true;", self.source
        )
        self.assertIn(
            "std::min(kGraphBandedScratchLanes,\n"
            "                           full_graph_chunk_count)",
            self.source,
        )

    def test_physical_arm_uses_the_pure_band_planner_and_conserves_rows(self) -> None:
        self.assertIn("SubmissionResult submit_banded_graph(", self.source)
        self.assertIn("while (remaining != 0)", self.source)
        self.assertIn("plan_next_prefill_band(", self.source)
        self.assertIn(
            "committed.next_context != band.next_context", self.source
        )
        self.assertIn("offset += band.row_count;", self.source)
        self.assertIn("remaining -= band.row_count;", self.source)

    def test_probe_capacity_derives_from_the_request_and_model(self) -> None:
        self.assertIn(
            "plan.tokenizer.maximum_context", self.source
        )
        self.assertIn(
            "static_cast<std::uint32_t>(required_capacity)", self.source
        )
        self.assertNotIn(
            "positions > plan.initial_serving_capacity", self.source
        )


if __name__ == "__main__":
    unittest.main()
