"""Structural Safetensors inspection without mapping tensor data."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MAX_HEADER_BYTES = 256 * 1024 * 1024

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "U16": 2,
    "I16": 2,
    "F32": 4,
    "U32": 4,
    "I32": 4,
    "F64": 8,
    "U64": 8,
    "I64": 8,
}


class SafetensorsError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value = {}
    for name, item in pairs:
        if name in value:
            raise SafetensorsError(f"Duplicate Safetensors header field: {name}")
        value[name] = item
    return value


@dataclass(frozen=True)
class SafetensorsTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset_bytes: int
    size_bytes: int


@dataclass(frozen=True)
class SafetensorsLayout:
    file_size_bytes: int
    data_offset_bytes: int
    data_size_bytes: int
    tensors: tuple[SafetensorsTensor, ...]


def _read_document(path: Path) -> tuple[dict[str, Any], int, int]:
    file_bytes = path.stat().st_size
    if file_bytes < 10:
        raise SafetensorsError(f"Safetensors file is too small: {path}")
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise SafetensorsError(f"Cannot read Safetensors header length: {path}")
        header_bytes = struct.unpack("<Q", raw_length)[0]
        if header_bytes < 2 or header_bytes > MAX_HEADER_BYTES:
            raise SafetensorsError(
                f"Invalid Safetensors header length {header_bytes}: {path}"
            )
        if 8 + header_bytes > file_bytes:
            raise SafetensorsError(f"Safetensors header exceeds file size: {path}")
        raw_header = stream.read(header_bytes)
    try:
        header = json.loads(raw_header, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SafetensorsError(f"Invalid Safetensors header JSON: {path}") from error
    if not isinstance(header, dict):
        raise SafetensorsError(f"Safetensors header is not an object: {path}")
    _validate_tensors(header, file_bytes - 8 - header_bytes, path)
    return header, file_bytes, 8 + header_bytes


def read_header(path: Path) -> dict[str, Any]:
    header, _, _ = _read_document(path)
    return header


def read_layout(path: Path) -> SafetensorsLayout:
    header, file_bytes, data_offset_bytes = _read_document(path)
    tensors = []
    for name, record in sorted(tensor_records(header).items()):
        start, end = record["data_offsets"]
        tensors.append(
            SafetensorsTensor(
                name=name,
                dtype=record["dtype"],
                shape=tuple(record["shape"]),
                offset_bytes=start,
                size_bytes=end - start,
            )
        )
    return SafetensorsLayout(
        file_size_bytes=file_bytes,
        data_offset_bytes=data_offset_bytes,
        data_size_bytes=file_bytes - data_offset_bytes,
        tensors=tuple(tensors),
    )


def tensor_records(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {name: value for name, value in header.items() if name != "__metadata__"}


def _validate_tensors(
    header: dict[str, Any], data_bytes: int, path: Path
) -> None:
    ranges: list[tuple[int, int, str]] = []
    for name, record in tensor_records(header).items():
        if (
            not isinstance(name, str)
            or not name
            or "\x00" in name
            or not isinstance(record, dict)
        ):
            raise SafetensorsError(f"Invalid tensor record in {path}")
        try:
            name.encode("utf-8")
        except UnicodeEncodeError as error:
            raise SafetensorsError(f"Tensor name is not valid UTF-8 in {path}") from error
        dtype = record.get("dtype")
        shape = record.get("shape")
        offsets = record.get("data_offsets")
        if dtype not in DTYPE_BYTES:
            raise SafetensorsError(f"Tensor {name} has an unsupported dtype in {path}")
        if not isinstance(shape, list) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in shape
        ):
            raise SafetensorsError(f"Tensor {name} has an invalid shape in {path}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in offsets)
        ):
            raise SafetensorsError(f"Tensor {name} has invalid offsets in {path}")
        start, end = offsets
        if start < 0 or end < start or end > data_bytes:
            raise SafetensorsError(f"Tensor {name} exceeds the data region in {path}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        if elements * DTYPE_BYTES[dtype] != end - start:
            raise SafetensorsError(f"Tensor {name} has an invalid byte size in {path}")
        ranges.append((start, end, name))

    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] < previous[1]:
            raise SafetensorsError(
                f"Tensor data overlaps between {previous[2]} and {current[2]} in {path}"
            )
