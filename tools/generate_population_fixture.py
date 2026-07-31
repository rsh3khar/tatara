#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

from tatara.prepared_checkpoint import (
    PreparedCheckpoint,
    PreparedCheckpointIdentity,
    PreparedShard,
    PreparedTensor,
    TensorDataType,
    encode_prepared_checkpoint,
)

RECORD_FILE_NAME = "population.tatara"
PATTERN_POSITION_MULTIPLIER = 7
PATTERN_SHARD_STEP = 13

FIXTURE_IDENTITY = PreparedCheckpointIdentity(
    package_id="population-fixture-model",
    package_sha256="1" * 64,
    artifact_id="population-fixture-artifact",
    artifact_manifest_sha256="2" * 64,
    model_type="fixture_type",
    format="safetensors",
    source_repository="fixture/repository",
    source_revision="3" * 40,
    artifact_file_count=4,
)


@dataclass(frozen=True)
class FixtureShardShape:
    path: str
    file_size_bytes: int
    data_offset_bytes: int
    data_size_bytes: int


FIXTURE_SHARD_SHAPES = (
    FixtureShardShape("model-00001.safetensors", 160, 32, 128),
    FixtureShardShape("model-00002.safetensors", 96, 32, 64),
)

# tensor-0 sorts first by name but sits at the highest shard offset, so the
# canonical record order deliberately differs from the byte-offset order.
FIXTURE_TENSORS = (
    PreparedTensor("tensor-0", TensorDataType.U8, (8,), 0, 100, 8),
    PreparedTensor("tensor-a", TensorDataType.BF16, (2, 4), 0, 16, 16),
    PreparedTensor("tensor-b", TensorDataType.U32, (2,), 0, 64, 8),
    PreparedTensor("tensor-c", TensorDataType.I8, (4,), 1, 8, 4),
)


def shard_payload(shard_index: int, size_bytes: int) -> bytes:
    return bytes(
        (position * PATTERN_POSITION_MULTIPLIER + shard_index * PATTERN_SHARD_STEP) & 0xFF
        for position in range(size_bytes)
    )


def build_fixture() -> tuple[PreparedCheckpoint, dict[str, bytes]]:
    payloads = {
        shape.path: shard_payload(index, shape.file_size_bytes)
        for index, shape in enumerate(FIXTURE_SHARD_SHAPES)
    }
    shards = tuple(
        PreparedShard(
            path=shape.path,
            sha256=hashlib.sha256(payloads[shape.path]).hexdigest(),
            file_size_bytes=shape.file_size_bytes,
            data_offset_bytes=shape.data_offset_bytes,
            data_size_bytes=shape.data_size_bytes,
        )
        for shape in FIXTURE_SHARD_SHAPES
    )
    checkpoint = PreparedCheckpoint(
        identity=FIXTURE_IDENTITY,
        shards=shards,
        tensors=FIXTURE_TENSORS,
    )
    return checkpoint, payloads


def write_fixture(directory: Path) -> None:
    checkpoint, payloads = build_fixture()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RECORD_FILE_NAME).write_bytes(encode_prepared_checkpoint(checkpoint))
    for name, payload in payloads.items():
        (directory / name).write_bytes(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", required=True, type=Path)
    arguments = parser.parse_args()
    write_fixture(arguments.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
