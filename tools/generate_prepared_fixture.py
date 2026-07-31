#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from tatara.prepared_checkpoint import (
    PreparedCheckpoint,
    PreparedCheckpointIdentity,
    PreparedShard,
    PreparedTensor,
    TensorDataType,
    encode_prepared_checkpoint,
)


def fixture() -> PreparedCheckpoint:
    return PreparedCheckpoint(
        identity=PreparedCheckpointIdentity(
            package_id="fixture-model",
            package_sha256="1" * 64,
            artifact_id="fixture-artifact",
            artifact_manifest_sha256="2" * 64,
            model_type="fixture_type",
            format="safetensors",
            source_repository="fixture/repository",
            source_revision="3" * 40,
            artifact_file_count=4,
        ),
        shards=(
            PreparedShard("model-00001.safetensors", "4" * 64, 160, 32, 128),
            PreparedShard("model-00002.safetensors", "5" * 64, 96, 32, 64),
        ),
        tensors=(
            PreparedTensor("tensor-a", TensorDataType.BF16, (2, 4), 0, 16, 16),
            PreparedTensor("tensor-b", TensorDataType.U32, (2,), 0, 64, 8),
            PreparedTensor("tensor-c", TensorDataType.I8, (4,), 1, 8, 4),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_bytes(encode_prepared_checkpoint(fixture()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
