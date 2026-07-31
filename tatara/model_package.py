"""Common identity fields shared by model-package tooling."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class ModelPackageError(ValueError):
    pass


@dataclass(frozen=True)
class ModelPackageIdentity:
    schema_version: int
    identifier: str
    family: str
    model_type: str
    artifact_manifest: str


def parse_model_package_identity(text: str) -> ModelPackageIdentity:
    try:
        value = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ModelPackageError("Model package is not valid TOML") from error
    if value.get("schema_version") != 1:
        raise ModelPackageError("Unsupported model package schema")
    artifact_manifest = _required_string(value, "artifact_manifest")
    path = PurePosixPath(artifact_manifest)
    if path.is_absolute() or len(path.parts) != 1 or artifact_manifest in {"", ".", ".."}:
        raise ModelPackageError("Model package artifact manifest must be adjacent")
    return ModelPackageIdentity(
        schema_version=1,
        identifier=_identifier(value, "id"),
        family=_identifier(value, "family"),
        model_type=_identifier(value, "model_type"),
        artifact_manifest=artifact_manifest,
    )


def _required_string(value: dict[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ModelPackageError(f"Model package field is invalid: {name}")
    return item


def _identifier(value: dict[str, Any], name: str) -> str:
    item = _required_string(value, name)
    if not IDENTIFIER.fullmatch(item):
        raise ModelPackageError(f"Model package identifier is invalid: {name}")
    return item
