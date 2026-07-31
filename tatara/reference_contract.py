"""Typed parsing and comparison for sealed Tatara references."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}")


class ReferenceContractError(ValueError):
    pass


@dataclass(frozen=True)
class ReferenceFile:
    identifier: str
    role: str
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SourceFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ByteEquality:
    left: str
    right: str


@dataclass(frozen=True)
class QualityGate:
    name: str
    passed: int
    total: int
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class PerformanceRecord:
    metric: str
    context_tokens: int | None
    lower: float
    upper: float
    unit: str
    conditions: tuple[str, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceContract:
    schema_version: int
    reference_id: str
    model_package: str
    model_manifest: str
    authority: str
    production_status: str
    source_repository: str
    source_commit: str
    source_relationship: str
    binary_link: str
    launch_executable: str
    launch_configuration: str
    launch_arguments: tuple[str, ...]
    files: tuple[ReferenceFile, ...]
    source_files: tuple[SourceFile, ...]
    equalities: tuple[ByteEquality, ...]
    token_prompt_tokens: int
    token_mode: str
    token_expected_ids: tuple[int, ...]
    api_passed: int
    api_total: int
    api_evidence: str
    quality: tuple[QualityGate, ...]
    performance: tuple[PerformanceRecord, ...]
    comparison_correctness_before_timing: bool
    comparison_contexts: tuple[int, ...]
    comparison_statistic: str
    comparison_order: str
    comparison_admission: str


@dataclass(frozen=True)
class ReferenceFileObservation:
    identifier: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class SourceFileObservation:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ReferenceVerification:
    matched: bool
    checked_files: int
    mismatches: tuple[str, ...]


def parse_reference_contract(text: str) -> ReferenceContract:
    try:
        value = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ReferenceContractError("Reference contract is not valid TOML") from error
    if value.get("schema_version") != 1:
        raise ReferenceContractError("Unsupported reference contract schema")

    reference = _table(value, "reference")
    source = _table(value, "source")
    launch = _table(value, "launch")
    token = _table(value, "token")
    api = _table(value, "api")
    comparison = _table(value, "comparison")

    files = tuple(_reference_file(item) for item in _table_list(value, "files"))
    file_ids = [item.identifier for item in files]
    if len(file_ids) != len(set(file_ids)):
        raise ReferenceContractError("Reference contract contains duplicate file ids")
    file_paths = [item.path for item in files]
    if len(file_paths) != len(set(file_paths)):
        raise ReferenceContractError("Reference contract contains duplicate file paths")
    by_id = {item.identifier: item for item in files}

    source_files = tuple(
        _source_file(item) for item in _table_list(value, "source_files")
    )
    source_paths = [item.path for item in source_files]
    if len(source_paths) != len(set(source_paths)):
        raise ReferenceContractError("Reference contract contains duplicate source paths")

    equalities = tuple(
        _byte_equality(item) for item in _table_list(value, "equalities")
    )
    for equality in equalities:
        left = by_id.get(equality.left)
        right = by_id.get(equality.right)
        if left is None or right is None:
            raise ReferenceContractError("Byte equality names an unknown file id")
        if left.size_bytes != right.size_bytes or left.sha256 != right.sha256:
            raise ReferenceContractError(
                f"Byte equality is not identical: {equality.left}, {equality.right}"
            )

    launch_executable = _required_string(launch, "executable")
    launch_configuration = _required_string(launch, "configuration")
    if launch_executable not in by_id or by_id[launch_executable].role != "executable":
        raise ReferenceContractError("Launch executable is not an executable file")
    if (
        launch_configuration not in by_id
        or by_id[launch_configuration].role != "configuration"
    ):
        raise ReferenceContractError("Launch configuration is not a configuration file")

    api_evidence = _required_string(api, "evidence")
    api_passed = _nonnegative_integer(api, "passed")
    api_total = _positive_integer(api, "total")
    if api_passed > api_total:
        raise ReferenceContractError("API passed count exceeds total")
    _require_evidence(by_id, (api_evidence,))
    quality = tuple(_quality_gate(item) for item in _table_list(value, "quality"))
    performance = tuple(
        _performance_record(item) for item in _table_list(value, "performance")
    )
    for gate in quality:
        _require_evidence(by_id, gate.evidence)
    for record in performance:
        _require_evidence(by_id, record.evidence)

    token_expected_ids = tuple(_integer_list(token, "expected_ids"))
    if not token_expected_ids:
        raise ReferenceContractError("Reference token stream is empty")
    if any(token_id < 0 for token_id in token_expected_ids):
        raise ReferenceContractError("Reference token id is negative")

    contexts = tuple(_integer_list(comparison, "contexts"))
    if len(contexts) < 3 or any(context <= 0 for context in contexts):
        raise ReferenceContractError("Comparison contexts are incomplete")
    correctness_before_timing = _required_bool(
        comparison, "correctness_before_timing"
    )
    if not correctness_before_timing:
        raise ReferenceContractError("Reference comparison must require correctness first")

    return ReferenceContract(
        schema_version=1,
        reference_id=_required_string(reference, "id"),
        model_package=_required_string(reference, "model_package"),
        model_manifest=_safe_path(_required_string(reference, "model_manifest")),
        authority=_required_string(reference, "authority"),
        production_status=_required_string(reference, "production_status"),
        source_repository=_required_string(source, "repository"),
        source_commit=_commit(source, "commit"),
        source_relationship=_required_string(source, "relationship"),
        binary_link=_required_string(source, "binary_link"),
        launch_executable=launch_executable,
        launch_configuration=launch_configuration,
        launch_arguments=tuple(_string_list(launch, "arguments")),
        files=files,
        source_files=source_files,
        equalities=equalities,
        token_prompt_tokens=_positive_integer(token, "prompt_tokens"),
        token_mode=_required_string(token, "mode"),
        token_expected_ids=token_expected_ids,
        api_passed=api_passed,
        api_total=api_total,
        api_evidence=api_evidence,
        quality=quality,
        performance=performance,
        comparison_correctness_before_timing=correctness_before_timing,
        comparison_contexts=contexts,
        comparison_statistic=_required_string(comparison, "statistic"),
        comparison_order=_required_string(comparison, "order"),
        comparison_admission=_required_string(comparison, "admission"),
    )


def verify_reference_observations(
    contract: ReferenceContract,
    observations: tuple[ReferenceFileObservation, ...],
    roles: frozenset[str],
) -> ReferenceVerification:
    expected = tuple(item for item in contract.files if item.role in roles)
    actual = {item.identifier: item for item in observations}
    mismatches: list[str] = []
    for item in expected:
        found = actual.get(item.identifier)
        if found is None:
            mismatches.append(f"missing file observation: {item.identifier}")
            continue
        if found.size_bytes != item.size_bytes:
            mismatches.append(
                f"file size {item.identifier}: expected {item.size_bytes}, "
                f"got {found.size_bytes}"
            )
        if found.sha256 != item.sha256:
            mismatches.append(
                f"file sha256 {item.identifier}: expected {item.sha256}, "
                f"got {found.sha256}"
            )
    unexpected = sorted(set(actual) - {item.identifier for item in expected})
    for identifier in unexpected:
        mismatches.append(f"unexpected file observation: {identifier}")
    return ReferenceVerification(
        matched=not mismatches,
        checked_files=len(expected),
        mismatches=tuple(mismatches),
    )


def verify_source_observations(
    contract: ReferenceContract,
    observations: tuple[SourceFileObservation, ...],
) -> ReferenceVerification:
    expected = {item.path: item for item in contract.source_files}
    actual = {item.path: item for item in observations}
    mismatches: list[str] = []
    for path, item in expected.items():
        found = actual.get(path)
        if found is None:
            mismatches.append(f"missing source observation: {path}")
            continue
        if found.size_bytes != item.size_bytes:
            mismatches.append(
                f"source size {path}: expected {item.size_bytes}, "
                f"got {found.size_bytes}"
            )
        if found.sha256 != item.sha256:
            mismatches.append(
                f"source sha256 {path}: expected {item.sha256}, "
                f"got {found.sha256}"
            )
    for path in sorted(set(actual) - set(expected)):
        mismatches.append(f"unexpected source observation: {path}")
    return ReferenceVerification(
        matched=not mismatches,
        checked_files=len(expected),
        mismatches=tuple(mismatches),
    )


def _reference_file(value: Any) -> ReferenceFile:
    if not isinstance(value, dict):
        raise ReferenceContractError("Reference file entry is not a table")
    sha256 = _required_string(value, "sha256")
    if not SHA256.fullmatch(sha256):
        raise ReferenceContractError("Reference file has an invalid SHA-256")
    return ReferenceFile(
        identifier=_required_string(value, "id"),
        role=_required_string(value, "role"),
        path=_safe_path(_required_string(value, "path")),
        size_bytes=_nonnegative_integer(value, "size_bytes"),
        sha256=sha256,
    )


def _source_file(value: Any) -> SourceFile:
    if not isinstance(value, dict):
        raise ReferenceContractError("Source file entry is not a table")
    sha256 = _required_string(value, "sha256")
    if not SHA256.fullmatch(sha256):
        raise ReferenceContractError("Source file has an invalid SHA-256")
    return SourceFile(
        path=_safe_path(_required_string(value, "path")),
        size_bytes=_nonnegative_integer(value, "size_bytes"),
        sha256=sha256,
    )


def _byte_equality(value: Any) -> ByteEquality:
    if not isinstance(value, dict):
        raise ReferenceContractError("Byte equality entry is not a table")
    return ByteEquality(
        left=_required_string(value, "left"),
        right=_required_string(value, "right"),
    )


def _quality_gate(value: Any) -> QualityGate:
    if not isinstance(value, dict):
        raise ReferenceContractError("Quality gate entry is not a table")
    passed = _nonnegative_integer(value, "passed")
    total = _positive_integer(value, "total")
    if passed > total:
        raise ReferenceContractError("Quality gate passed count exceeds total")
    return QualityGate(
        name=_required_string(value, "name"),
        passed=passed,
        total=total,
        evidence=tuple(_string_list(value, "evidence")),
    )


def _performance_record(value: Any) -> PerformanceRecord:
    if not isinstance(value, dict):
        raise ReferenceContractError("Performance record entry is not a table")
    lower = _number(value, "lower")
    upper = _number(value, "upper")
    if lower < 0 or upper < lower:
        raise ReferenceContractError("Performance record range is invalid")
    context = value.get("context_tokens")
    if context is not None and (
        not isinstance(context, int) or isinstance(context, bool) or context <= 0
    ):
        raise ReferenceContractError("Performance context is invalid")
    return PerformanceRecord(
        metric=_required_string(value, "metric"),
        context_tokens=context,
        lower=lower,
        upper=upper,
        unit=_required_string(value, "unit"),
        conditions=tuple(_string_list(value, "conditions")),
        evidence=tuple(_string_list(value, "evidence")),
    )


def _require_evidence(by_id: dict[str, ReferenceFile], ids: tuple[str, ...]) -> None:
    if not ids:
        raise ReferenceContractError("Gate has no evidence")
    for identifier in ids:
        item = by_id.get(identifier)
        if item is None or item.role != "evidence":
            raise ReferenceContractError(f"Unknown evidence file id: {identifier}")


def _safe_path(path: str) -> str:
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or path in {"", "."}:
        raise ReferenceContractError(f"Unsafe reference path: {path}")
    return path


def _table(value: dict[str, Any], name: str) -> dict[str, Any]:
    item = value.get(name)
    if not isinstance(item, dict):
        raise ReferenceContractError(f"Reference contract has no [{name}] table")
    return item


def _table_list(value: dict[str, Any], name: str) -> list[dict[str, Any]]:
    items = value.get(name)
    if not isinstance(items, list) or not items:
        raise ReferenceContractError(f"Reference contract has no [[{name}]] entries")
    if not all(isinstance(item, dict) for item in items):
        raise ReferenceContractError(f"Reference contract [[{name}]] is invalid")
    return items


def _required_string(value: dict[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ReferenceContractError(f"Reference contract field is invalid: {name}")
    return item


def _required_bool(value: dict[str, Any], name: str) -> bool:
    item = value.get(name)
    if not isinstance(item, bool):
        raise ReferenceContractError(f"Reference contract field is invalid: {name}")
    return item


def _string_list(value: dict[str, Any], name: str) -> list[str]:
    item = value.get(name)
    if not isinstance(item, list) or not item or not all(
        isinstance(entry, str) and entry for entry in item
    ):
        raise ReferenceContractError(f"Reference contract field is invalid: {name}")
    return item


def _integer_list(value: dict[str, Any], name: str) -> list[int]:
    item = value.get(name)
    if not isinstance(item, list) or not all(
        isinstance(entry, int) and not isinstance(entry, bool) for entry in item
    ):
        raise ReferenceContractError(f"Reference contract field is invalid: {name}")
    return item


def _nonnegative_integer(value: dict[str, Any], name: str) -> int:
    item = value.get(name)
    if not isinstance(item, int) or isinstance(item, bool) or item < 0:
        raise ReferenceContractError(f"Reference contract field is invalid: {name}")
    return item


def _positive_integer(value: dict[str, Any], name: str) -> int:
    item = _nonnegative_integer(value, name)
    if item == 0:
        raise ReferenceContractError(f"Reference contract field is invalid: {name}")
    return item


def _number(value: dict[str, Any], name: str) -> float:
    item = value.get(name)
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise ReferenceContractError(f"Reference contract field is invalid: {name}")
    return float(item)


def _commit(value: dict[str, Any], name: str) -> str:
    item = _required_string(value, name)
    if not GIT_COMMIT.fullmatch(item):
        raise ReferenceContractError("Reference source commit is invalid")
    return item
