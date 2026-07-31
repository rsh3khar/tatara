"""Pure contract and encoder for Tatara prepared-checkpoint records."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import PurePosixPath


MAGIC = b"TATCKPT\0"
SCHEMA_VERSION = 1
MAX_RECORD_BYTES = 512 * 1024 * 1024
MAX_STRING_BYTES = 1024 * 1024
MAX_ARTIFACT_FILE_COUNT = 1_000_000
MAX_SHARD_COUNT = 65_536
MAX_TENSOR_COUNT = 1_000_000
MAX_TENSOR_RANK = 64
UINT64_MAXIMUM = 2**64 - 1
SHA256 = re.compile(r"[0-9a-f]{64}")
PREFIX = struct.Struct("<8sIIQIIIIQ")


class PreparedCheckpointError(ValueError):
    pass


class TensorDataType(IntEnum):
    BOOL = 1
    U8 = 2
    I8 = 3
    F16 = 4
    BF16 = 5
    U16 = 6
    I16 = 7
    F32 = 8
    U32 = 9
    I32 = 10
    F64 = 11
    U64 = 12
    I64 = 13
    F8_E4M3 = 14
    F8_E5M2 = 15

    @property
    def bytes(self) -> int:
        if self in {
            TensorDataType.BOOL,
            TensorDataType.U8,
            TensorDataType.I8,
            TensorDataType.F8_E4M3,
            TensorDataType.F8_E5M2,
        }:
            return 1
        if self in {
            TensorDataType.F16,
            TensorDataType.BF16,
            TensorDataType.U16,
            TensorDataType.I16,
        }:
            return 2
        if self in {
            TensorDataType.F32,
            TensorDataType.U32,
            TensorDataType.I32,
        }:
            return 4
        return 8


@dataclass(frozen=True)
class PreparedCheckpointIdentity:
    package_id: str
    package_sha256: str
    artifact_id: str
    artifact_manifest_sha256: str
    model_type: str
    format: str
    source_repository: str
    source_revision: str
    artifact_file_count: int


@dataclass(frozen=True)
class PreparedShard:
    path: str
    sha256: str
    file_size_bytes: int
    data_offset_bytes: int
    data_size_bytes: int


@dataclass(frozen=True)
class PreparedTensor:
    name: str
    data_type: TensorDataType
    shape: tuple[int, ...]
    shard: int
    shard_offset_bytes: int
    size_bytes: int


@dataclass(frozen=True)
class PreparedCheckpoint:
    identity: PreparedCheckpointIdentity
    shards: tuple[PreparedShard, ...]
    tensors: tuple[PreparedTensor, ...]

    @property
    def tensor_payload_bytes(self) -> int:
        return sum(tensor.size_bytes for tensor in self.tensors)


def encode_prepared_checkpoint(checkpoint: PreparedCheckpoint) -> bytes:
    _validate(checkpoint)
    chunks = [b""]
    identity = checkpoint.identity
    for value in (
        identity.package_id,
        identity.package_sha256,
        identity.artifact_id,
        identity.artifact_manifest_sha256,
        identity.model_type,
        identity.format,
        identity.source_repository,
        identity.source_revision,
    ):
        chunks.append(_encode_string(value))
    for shard in checkpoint.shards:
        chunks.extend(
            (
                _encode_string(shard.path),
                _encode_string(shard.sha256),
                struct.pack(
                    "<QQQ",
                    shard.file_size_bytes,
                    shard.data_offset_bytes,
                    shard.data_size_bytes,
                ),
            )
        )
    for tensor in checkpoint.tensors:
        chunks.extend(
            (
                _encode_string(tensor.name),
                struct.pack(
                    "<IIIIQQ",
                    tensor.data_type,
                    tensor.shard,
                    len(tensor.shape),
                    0,
                    tensor.shard_offset_bytes,
                    tensor.size_bytes,
                ),
                struct.pack(f"<{len(tensor.shape)}Q", *tensor.shape),
            )
        )
    record_bytes = PREFIX.size + sum(len(chunk) for chunk in chunks[1:])
    if record_bytes > MAX_RECORD_BYTES:
        raise PreparedCheckpointError("Prepared checkpoint record is too large")
    chunks[0] = PREFIX.pack(
        MAGIC,
        SCHEMA_VERSION,
        0,
        record_bytes,
        identity.artifact_file_count,
        len(checkpoint.shards),
        len(checkpoint.tensors),
        0,
        checkpoint.tensor_payload_bytes,
    )
    return b"".join(chunks)


def decode_prepared_checkpoint(record: bytes) -> PreparedCheckpoint:
    """Inverse of `encode_prepared_checkpoint`.

    Every length is read from the record and checked against what remains, so a
    truncated or hostile record raises rather than over-reading.
    """
    if len(record) < PREFIX.size:
        raise PreparedCheckpointError("Record is shorter than its prefix")
    (
        magic,
        schema_version,
        _reserved0,
        record_bytes,
        artifact_file_count,
        shard_count,
        tensor_count,
        _reserved1,
        payload_bytes,
    ) = PREFIX.unpack_from(record, 0)
    if magic != MAGIC:
        raise PreparedCheckpointError("Record magic is not TATCKPT")
    if schema_version != SCHEMA_VERSION:
        raise PreparedCheckpointError(
            f"Record schema version {schema_version}, expected {SCHEMA_VERSION}"
        )
    if record_bytes != len(record):
        raise PreparedCheckpointError(
            f"Record declares {record_bytes} bytes but is {len(record)}"
        )
    if shard_count > MAX_SHARD_COUNT or tensor_count > MAX_TENSOR_COUNT:
        raise PreparedCheckpointError("Record shard or tensor count is out of range")

    cursor = PREFIX.size
    strings = []
    for _ in range(8):
        value, cursor = _decode_string(record, cursor)
        strings.append(value)
    identity = PreparedCheckpointIdentity(
        package_id=strings[0],
        package_sha256=strings[1],
        artifact_id=strings[2],
        artifact_manifest_sha256=strings[3],
        model_type=strings[4],
        format=strings[5],
        source_repository=strings[6],
        source_revision=strings[7],
        artifact_file_count=artifact_file_count,
    )

    shards = []
    for _ in range(shard_count):
        path, cursor = _decode_string(record, cursor)
        digest, cursor = _decode_string(record, cursor)
        file_size, offset, size = _unpack(record, cursor, "<QQQ")
        cursor += struct.calcsize("<QQQ")
        shards.append(
            PreparedShard(
                path=path,
                sha256=digest,
                file_size_bytes=file_size,
                data_offset_bytes=offset,
                data_size_bytes=size,
            )
        )

    tensors = []
    for _ in range(tensor_count):
        name, cursor = _decode_string(record, cursor)
        data_type, shard, rank, _pad, shard_offset, size = _unpack(record, cursor, "<IIIIQQ")
        cursor += struct.calcsize("<IIIIQQ")
        if rank > MAX_TENSOR_RANK:
            raise PreparedCheckpointError(f"Tensor {name} declares rank {rank}")
        shape = _unpack(record, cursor, f"<{rank}Q")
        cursor += struct.calcsize(f"<{rank}Q")
        tensors.append(
            PreparedTensor(
                name=name,
                data_type=TensorDataType(data_type),
                shape=tuple(shape),
                shard=shard,
                shard_offset_bytes=shard_offset,
                size_bytes=size,
            )
        )

    if cursor != len(record):
        raise PreparedCheckpointError(
            f"Record has {len(record) - cursor} trailing bytes"
        )
    checkpoint = PreparedCheckpoint(
        identity=identity, shards=tuple(shards), tensors=tuple(tensors)
    )
    if checkpoint.tensor_payload_bytes != payload_bytes:
        raise PreparedCheckpointError(
            f"Record declares {payload_bytes} payload bytes but the tensors sum to "
            f"{checkpoint.tensor_payload_bytes}"
        )
    _validate(checkpoint)
    return checkpoint


def _unpack(record: bytes, cursor: int, layout: str) -> tuple[int, ...]:
    size = struct.calcsize(layout)
    if cursor + size > len(record):
        raise PreparedCheckpointError("Record ends inside a fixed-width field")
    return struct.unpack_from(layout, record, cursor)


def _decode_string(record: bytes, cursor: int) -> tuple[str, int]:
    (length,) = _unpack(record, cursor, "<I")
    cursor += 4
    if length == 0 or length > MAX_STRING_BYTES or cursor + length > len(record):
        raise PreparedCheckpointError("Record string length is out of range")
    try:
        value = record[cursor : cursor + length].decode("utf-8")
    except UnicodeDecodeError as error:
        raise PreparedCheckpointError("Record string is not valid UTF-8") from error
    return value, cursor + length


def _validate(checkpoint: PreparedCheckpoint) -> None:
    identity = checkpoint.identity
    for name, value in (
        ("package id", identity.package_id),
        ("artifact id", identity.artifact_id),
        ("model type", identity.model_type),
        ("format", identity.format),
        ("source repository", identity.source_repository),
        ("source revision", identity.source_revision),
    ):
        if not isinstance(value, str) or not value or "\x00" in value:
            raise PreparedCheckpointError(f"Invalid {name}")
    for name, value in (
        ("package SHA-256", identity.package_sha256),
        ("artifact manifest SHA-256", identity.artifact_manifest_sha256),
    ):
        if not isinstance(value, str) or not SHA256.fullmatch(value):
            raise PreparedCheckpointError(f"Invalid {name}")
    if not checkpoint.shards or len(checkpoint.shards) > MAX_SHARD_COUNT:
        raise PreparedCheckpointError("Invalid prepared shard count")
    if not checkpoint.tensors or len(checkpoint.tensors) > MAX_TENSOR_COUNT:
        raise PreparedCheckpointError("Invalid prepared tensor count")
    if (
        not isinstance(identity.artifact_file_count, int)
        or isinstance(identity.artifact_file_count, bool)
        or identity.artifact_file_count < len(checkpoint.shards)
        or identity.artifact_file_count > MAX_ARTIFACT_FILE_COUNT
    ):
        raise PreparedCheckpointError("Invalid artifact file count")

    previous_path = b""
    for shard in checkpoint.shards:
        if not isinstance(shard.path, str):
            raise PreparedCheckpointError("Invalid shard path")
        path = PurePosixPath(shard.path)
        encoded_path = _utf8(shard.path, "shard path")
        if (
            path.is_absolute()
            or "\\" in shard.path
            or any(part in {"", ".", ".."} for part in shard.path.split("/"))
            or encoded_path <= previous_path
        ):
            raise PreparedCheckpointError(f"Invalid or unordered shard path: {shard.path}")
        previous_path = encoded_path
        if not SHA256.fullmatch(shard.sha256):
            raise PreparedCheckpointError(f"Invalid shard SHA-256: {shard.path}")
        _uint64(shard.file_size_bytes, "shard file size")
        _uint64(shard.data_offset_bytes, "shard data offset")
        _uint64(shard.data_size_bytes, "shard data size")
        if (
            shard.data_offset_bytes < 10
            or shard.data_offset_bytes + shard.data_size_bytes
            != shard.file_size_bytes
        ):
            raise PreparedCheckpointError(f"Invalid shard data region: {shard.path}")

    previous_name = b""
    extents: list[tuple[int, int, int, str]] = []
    total_bytes = 0
    for tensor in checkpoint.tensors:
        if not isinstance(tensor.name, str):
            raise PreparedCheckpointError("Invalid tensor name")
        encoded_name = _utf8(tensor.name, "tensor name")
        if not tensor.name or "\x00" in tensor.name or encoded_name <= previous_name:
            raise PreparedCheckpointError(f"Invalid or unordered tensor name: {tensor.name}")
        previous_name = encoded_name
        if (
            not isinstance(tensor.shard, int)
            or isinstance(tensor.shard, bool)
            or tensor.shard < 0
            or tensor.shard >= len(checkpoint.shards)
        ):
            raise PreparedCheckpointError(f"Invalid tensor shard: {tensor.name}")
        if not isinstance(tensor.data_type, TensorDataType):
            raise PreparedCheckpointError(f"Invalid tensor dtype: {tensor.name}")
        if len(tensor.shape) > MAX_TENSOR_RANK:
            raise PreparedCheckpointError(f"Tensor rank is too large: {tensor.name}")
        elements = 1
        for dimension in tensor.shape:
            _uint64(dimension, "tensor dimension")
            elements *= dimension
            if elements > UINT64_MAXIMUM:
                raise PreparedCheckpointError(f"Tensor shape overflows: {tensor.name}")
        expected_size = elements * tensor.data_type.bytes
        if expected_size > UINT64_MAXIMUM or expected_size != tensor.size_bytes:
            raise PreparedCheckpointError(f"Tensor size mismatch: {tensor.name}")
        _uint64(tensor.shard_offset_bytes, "tensor shard offset")
        end = tensor.shard_offset_bytes + tensor.size_bytes
        if end > checkpoint.shards[tensor.shard].data_size_bytes:
            raise PreparedCheckpointError(f"Tensor exceeds its shard: {tensor.name}")
        total_bytes += tensor.size_bytes
        if total_bytes > UINT64_MAXIMUM:
            raise PreparedCheckpointError("Tensor payload size overflows")
        if tensor.size_bytes:
            extents.append((tensor.shard, tensor.shard_offset_bytes, end, tensor.name))
    extents.sort()
    for previous, current in zip(extents, extents[1:]):
        if previous[0] == current[0] and current[1] < previous[2]:
            raise PreparedCheckpointError(f"Tensor data overlaps: {current[3]}")


def _encode_string(value: str) -> bytes:
    encoded = _utf8(value, "prepared checkpoint string")
    if not encoded or len(encoded) > MAX_STRING_BYTES:
        raise PreparedCheckpointError("Prepared checkpoint string is invalid")
    return struct.pack("<I", len(encoded)) + encoded


def _utf8(value: str, name: str) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise PreparedCheckpointError(f"Invalid UTF-8 {name}") from error


def _uint64(value: int, name: str) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > UINT64_MAXIMUM
    ):
        raise PreparedCheckpointError(f"Invalid {name}")
