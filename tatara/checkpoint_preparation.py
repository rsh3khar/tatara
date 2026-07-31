"""Filesystem adapter for deterministic Safetensors checkpoint preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tatara.artifact_manifest import (
    ArtifactManifestError,
    ManifestFile,
    parse_manifest,
)
from tatara.model_package import ModelPackageError, parse_model_package_identity
from tatara.model_types import FileRole
from tatara.prepared_checkpoint import (
    PreparedCheckpoint,
    PreparedCheckpointError,
    PreparedCheckpointIdentity,
    PreparedShard,
    PreparedTensor,
    TensorDataType,
    encode_prepared_checkpoint,
)
from tatara.safetensors import SafetensorsError, read_layout


MAX_INDEX_BYTES = 64 * 1024 * 1024


class CheckpointPreparationError(ValueError):
    pass


def prepare_checkpoint(
    package_path: Path,
    artifact_root: Path,
    artifact_manifest_path: Path | None = None,
) -> tuple[PreparedCheckpoint, bytes]:
    try:
        package_bytes = package_path.read_bytes()
        package_text = package_bytes.decode("utf-8")
        package = parse_model_package_identity(package_text)
        manifest_path = artifact_manifest_path or package_path.parent / package.artifact_manifest
        if manifest_path.name != package.artifact_manifest:
            raise CheckpointPreparationError(
                "Model package selects a different artifact manifest"
            )
        manifest_bytes = manifest_path.read_bytes()
        manifest = parse_manifest(manifest_bytes.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise CheckpointPreparationError("Model package metadata is not UTF-8") from error
    except (ArtifactManifestError, ModelPackageError, OSError) as error:
        raise CheckpointPreparationError(str(error)) from error

    if manifest.format != "safetensors":
        raise CheckpointPreparationError(
            f"Checkpoint preparation does not support {manifest.format}"
        )
    if package.model_type != manifest.model_type:
        raise CheckpointPreparationError("Model package and artifact model types differ")

    weights = sorted(
        (item for item in manifest.files if item.role is FileRole.WEIGHT),
        key=lambda item: item.path.encode("utf-8"),
    )
    if not weights:
        raise CheckpointPreparationError("Artifact manifest has no weight shards")

    root = artifact_root.expanduser().resolve()
    shards = []
    tensors = []
    tensor_locations: dict[str, str] = {}
    try:
        for shard_index, weight in enumerate(weights):
            path = root / weight.path
            if not path.is_file():
                raise CheckpointPreparationError(
                    f"Safetensors shard is missing: {weight.path}"
                )
            layout = read_layout(path)
            if layout.file_size_bytes != weight.size_bytes:
                raise CheckpointPreparationError(
                    f"Safetensors shard size mismatch: {weight.path}"
                )
            shards.append(
                PreparedShard(
                    path=weight.path,
                    sha256=weight.sha256,
                    file_size_bytes=layout.file_size_bytes,
                    data_offset_bytes=layout.data_offset_bytes,
                    data_size_bytes=layout.data_size_bytes,
                )
            )
            for tensor in layout.tensors:
                if tensor.name in tensor_locations:
                    raise CheckpointPreparationError(
                        f"Duplicate tensor across shards: {tensor.name}"
                    )
                tensor_locations[tensor.name] = weight.path
                try:
                    data_type = TensorDataType[tensor.dtype]
                except KeyError as error:
                    raise CheckpointPreparationError(
                        f"Unsupported tensor dtype {tensor.dtype}: {tensor.name}"
                    ) from error
                tensors.append(
                    PreparedTensor(
                        name=tensor.name,
                        data_type=data_type,
                        shape=tensor.shape,
                        shard=shard_index,
                        shard_offset_bytes=tensor.offset_bytes,
                        size_bytes=tensor.size_bytes,
                    )
                )
    except (OSError, SafetensorsError) as error:
        raise CheckpointPreparationError(str(error)) from error

    _verify_index(root, manifest.files, tensor_locations, manifest.tensor_bytes)
    tensors.sort(key=lambda tensor: tensor.name.encode("utf-8"))
    if len(tensors) != manifest.tensor_count:
        raise CheckpointPreparationError(
            f"Tensor count mismatch: expected {manifest.tensor_count}, got {len(tensors)}"
        )
    tensor_bytes = sum(tensor.size_bytes for tensor in tensors)
    if tensor_bytes != manifest.tensor_bytes:
        raise CheckpointPreparationError(
            f"Tensor bytes mismatch: expected {manifest.tensor_bytes}, got {tensor_bytes}"
        )

    checkpoint = PreparedCheckpoint(
        identity=PreparedCheckpointIdentity(
            package_id=package.identifier,
            package_sha256=hashlib.sha256(package_bytes).hexdigest(),
            artifact_id=manifest.artifact_id,
            artifact_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            model_type=manifest.model_type,
            format=manifest.format,
            source_repository=manifest.source_repository,
            source_revision=manifest.source_revision,
            artifact_file_count=len(manifest.files),
        ),
        shards=tuple(shards),
        tensors=tuple(tensors),
    )
    try:
        encoded = encode_prepared_checkpoint(checkpoint)
    except PreparedCheckpointError as error:
        raise CheckpointPreparationError(str(error)) from error
    return checkpoint, encoded


def write_new_checkpoint(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        raise
    except OSError:
        if path.is_file():
            path.unlink()
        raise


def _verify_index(
    root: Path,
    files: tuple[ManifestFile, ...],
    tensor_locations: dict[str, str],
    expected_tensor_bytes: int,
) -> None:
    candidates = sorted(
        item.path
        for item in files
        if item.role is FileRole.METADATA
        and item.path.endswith(".safetensors.index.json")
    )
    if len(candidates) > 1:
        raise CheckpointPreparationError("Artifact contains multiple Safetensors indexes")
    if not candidates:
        return
    path = root / candidates[0]
    if not path.is_file() or path.stat().st_size > MAX_INDEX_BYTES:
        raise CheckpointPreparationError("Safetensors index is missing or too large")
    try:
        index = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
        raise CheckpointPreparationError("Safetensors index is invalid") from error
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise CheckpointPreparationError("Safetensors index has no weight_map")
    weight_map = index["weight_map"]
    if any(not isinstance(name, str) for name in weight_map):
        raise CheckpointPreparationError("Safetensors index contains an invalid tensor name")
    if set(weight_map) != set(tensor_locations):
        raise CheckpointPreparationError("Safetensors index/header tensor sets differ")
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise CheckpointPreparationError("Safetensors index contains an invalid mapping")
        if tensor_locations[name] != shard:
            raise CheckpointPreparationError(
                f"Safetensors index maps {name} to the wrong shard"
            )
    metadata = index.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise CheckpointPreparationError("Safetensors index metadata is invalid")
    declared = (metadata or {}).get("total_size")
    if declared is not None and declared != expected_tensor_bytes:
        raise CheckpointPreparationError("Safetensors index total_size differs from manifest")
