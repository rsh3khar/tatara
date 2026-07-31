import hashlib
import unittest

from tatara.prepared_checkpoint import encode_prepared_checkpoint
from tools.generate_population_fixture import (
    FIXTURE_SHARD_SHAPES,
    FIXTURE_TENSORS,
    build_fixture,
    shard_payload,
)


class PopulationFixtureTest(unittest.TestCase):
    def test_fixture_is_deterministic(self):
        first_checkpoint, first_payloads = build_fixture()
        second_checkpoint, second_payloads = build_fixture()
        self.assertEqual(first_payloads, second_payloads)
        self.assertEqual(
            encode_prepared_checkpoint(first_checkpoint),
            encode_prepared_checkpoint(second_checkpoint),
        )

    def test_recorded_digests_match_payloads(self):
        checkpoint, payloads = build_fixture()
        for shard in checkpoint.shards:
            self.assertEqual(
                shard.sha256, hashlib.sha256(payloads[shard.path]).hexdigest()
            )
            self.assertEqual(shard.file_size_bytes, len(payloads[shard.path]))

    def test_payload_pattern_is_position_dependent(self):
        payload = shard_payload(1, 96)
        self.assertEqual(len(payload), 96)
        self.assertEqual(payload[0], 13)
        self.assertEqual(payload[3], (3 * 7 + 13) & 0xFF)

    def test_record_order_differs_from_offset_order(self):
        shard_zero = [tensor for tensor in FIXTURE_TENSORS if tensor.shard == 0]
        record_names = [tensor.name for tensor in shard_zero]
        offset_names = [
            tensor.name
            for tensor in sorted(shard_zero, key=lambda tensor: tensor.shard_offset_bytes)
        ]
        self.assertEqual(record_names, sorted(record_names))
        self.assertNotEqual(record_names, offset_names)

    def test_data_regions_leave_unclaimed_bytes(self):
        for index, shape in enumerate(FIXTURE_SHARD_SHAPES):
            claimed = sum(
                tensor.size_bytes for tensor in FIXTURE_TENSORS if tensor.shard == index
            )
            self.assertLess(claimed, shape.data_size_bytes)


if __name__ == "__main__":
    unittest.main()
