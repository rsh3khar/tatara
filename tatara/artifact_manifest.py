"""Pure parsing and verification for relocatable model artifact identities."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from tatara.model_types import FileRole, ModelSnapshot


SHA256 = re.compile(r"[0-9a-f]{64}")


class ArtifactManifestError(ValueError):
    pass


@dataclass(frozen=True)
class ManifestFile:
    path: str
    role: FileRole
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactManifest:
    schema_version: int
    artifact_id: str
    task: str
    model_type: str
    format: str
    tensor_count: int
    tensor_bytes: int
    source_repository: str
    source_revision: str
    files: tuple[ManifestFile, ...]


@dataclass(frozen=True)
class ManifestVerification:
    artifact_id: str
    matched: bool
    checked_files: int
    mismatches: tuple[str, ...]


def parse_manifest(text: str) -> ArtifactManifest:
    try:
        value = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ArtifactManifestError("Artifact manifest is not valid TOML") from error
    if value.get("schema_version") != 1:
        raise ArtifactManifestError("Unsupported artifact manifest schema")
    artifact = _table(value, "artifact")
    source = _table(value, "source")
    tensor = _table(value, "tensor")
    raw_files = value.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ArtifactManifestError("Artifact manifest has no files")
    files = tuple(_manifest_file(item) for item in raw_files)
    paths = [item.path for item in files]
    if len(paths) != len(set(paths)):
        raise ArtifactManifestError("Artifact manifest contains duplicate paths")
    return ArtifactManifest(
        schema_version=1,
        artifact_id=_required_string(artifact, "id"),
        task=_required_string(artifact, "task"),
        model_type=_required_string(artifact, "model_type"),
        format=_required_string(artifact, "format"),
        tensor_count=_nonnegative_integer(tensor, "count"),
        tensor_bytes=_nonnegative_integer(tensor, "bytes"),
        source_repository=_required_string(source, "repository"),
        source_revision=_required_string(source, "revision"),
        files=files,
    )


def verify_manifest(
    manifest: ArtifactManifest, snapshot: ModelSnapshot
) -> ManifestVerification:
    mismatches: list[str] = []
    if snapshot.tensor_summary.tensor_count != manifest.tensor_count:
        mismatches.append(
            "tensor count: "
            f"expected {manifest.tensor_count}, got {snapshot.tensor_summary.tensor_count}"
        )
    if snapshot.tensor_summary.tensor_bytes != manifest.tensor_bytes:
        mismatches.append(
            "tensor bytes: "
            f"expected {manifest.tensor_bytes}, got {snapshot.tensor_summary.tensor_bytes}"
        )
    model_type = snapshot.config.get("model_type")
    if model_type != manifest.model_type:
        mismatches.append(
            f"model type: expected {manifest.model_type}, got {model_type}"
        )

    actual = {item.path: item for item in snapshot.files}
    expected_paths = {item.path for item in manifest.files}
    for path in sorted(set(actual) - expected_paths):
        mismatches.append(f"unexpected top-level file: {path}")
    weight_paths = [
        item.path for item in manifest.files if item.role is FileRole.WEIGHT
    ]
    actual_format = _weight_format(weight_paths)
    if actual_format != manifest.format:
        mismatches.append(
            f"artifact format: expected {manifest.format}, got {actual_format}"
        )
    for expected in manifest.files:
        found = actual.get(expected.path)
        if found is None:
            mismatches.append(f"missing file: {expected.path}")
            continue
        if found.role is not expected.role:
            mismatches.append(
                f"file role {expected.path}: expected {expected.role.value}, "
                f"got {found.role.value}"
            )
        if found.size_bytes != expected.size_bytes:
            mismatches.append(
                f"file size {expected.path}: expected {expected.size_bytes}, "
                f"got {found.size_bytes}"
            )
        if found.sha256 != expected.sha256:
            got = found.sha256 or "not-computed"
            mismatches.append(
                f"file sha256 {expected.path}: expected {expected.sha256}, got {got}"
            )
    return ManifestVerification(
        artifact_id=manifest.artifact_id,
        matched=not mismatches,
        checked_files=len(manifest.files),
        mismatches=tuple(mismatches),
    )


def _weight_format(paths: list[str]) -> str:
    has_safetensors = any(path.endswith(".safetensors") for path in paths)
    has_gguf = any(path.endswith(".gguf") for path in paths)
    if has_safetensors and not has_gguf:
        return "safetensors"
    if has_gguf and not has_safetensors:
        return "gguf"
    if has_safetensors or has_gguf:
        return "mixed"
    return "unknown"


def _manifest_file(value: Any) -> ManifestFile:
    if not isinstance(value, dict):
        raise ArtifactManifestError("Artifact manifest file entry is not a table")
    path = _required_string(value, "path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or path in {"", "."}:
        raise ArtifactManifestError(f"Unsafe artifact manifest path: {path}")
    try:
        role = FileRole(_required_string(value, "role"))
    except ValueError as error:
        raise ArtifactManifestError(f"Invalid artifact file role: {path}") from error
    sha256 = _required_string(value, "sha256")
    if not SHA256.fullmatch(sha256):
        raise ArtifactManifestError(f"Invalid SHA-256 for artifact file: {path}")
    return ManifestFile(
        path=path,
        role=role,
        size_bytes=_nonnegative_integer(value, "size_bytes"),
        sha256=sha256,
    )


def _table(value: dict[str, Any], name: str) -> dict[str, Any]:
    table = value.get(name)
    if not isinstance(table, dict):
        raise ArtifactManifestError(f"Artifact manifest has no [{name}] table")
    return table


def _required_string(value: dict[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ArtifactManifestError(f"Artifact manifest field is invalid: {name}")
    return item


def _nonnegative_integer(value: dict[str, Any], name: str) -> int:
    item = value.get(name)
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise ArtifactManifestError(f"Artifact manifest field is invalid: {name}")
    return item
