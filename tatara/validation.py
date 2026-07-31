"""Validation of a prepared checkpoint against the artifact it was built from.

Pure core: every function here takes bytes or already-loaded values and returns
a result. Reading files belongs to the CLI shell.

This exists because of a real failure. A comment-only edit to `model.toml`
re-derived the package digest and silently orphaned an already-prepared record;
the mismatch only surfaced as a typed exit *inside a guarded GPU window*, after
the model had loaded. Validation is cheap and pure, so it belongs before the
window, not inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from tatara.artifact_manifest import ArtifactManifest
from tatara.prepared_checkpoint import PreparedCheckpoint


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    severity: Severity
    check: str
    detail: str


@dataclass(frozen=True)
class ValidationReport:
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.ERROR)

    @property
    def ok(self) -> bool:
        return not self.errors


# Check names are referenced by tests and by the human output, so they are
# defined once here rather than spelled inline at each raise site.
CHECK_PACKAGE_DIGEST = "package-digest"
CHECK_MANIFEST_DIGEST = "manifest-digest"
CHECK_ARTIFACT_ID = "artifact-id"
CHECK_FILE_COUNT = "artifact-file-count"
CHECK_SHARD_DIGEST = "shard-digest"
CHECK_SHARD_COVERAGE = "shard-coverage"
CHECK_TENSOR_BOUNDS = "tensor-bounds"


def validate_prepared_checkpoint(
    checkpoint: PreparedCheckpoint,
    manifest: ArtifactManifest,
    package_sha256: str,
    manifest_sha256: str,
) -> ValidationReport:
    """Check a decoded record against the package and manifest it pins."""
    findings: list[Finding] = []
    identity = checkpoint.identity

    if identity.package_sha256 != package_sha256:
        findings.append(
            Finding(
                Severity.ERROR,
                CHECK_PACKAGE_DIGEST,
                f"record pins {identity.package_sha256[:16]}… but the package hashes "
                f"{package_sha256[:16]}… — regenerate the record; any model.toml edit, "
                "including a comment, re-derives this digest",
            )
        )
    if identity.artifact_manifest_sha256 != manifest_sha256:
        findings.append(
            Finding(
                Severity.ERROR,
                CHECK_MANIFEST_DIGEST,
                f"record pins {identity.artifact_manifest_sha256[:16]}… but the manifest "
                f"hashes {manifest_sha256[:16]}…",
            )
        )
    if identity.artifact_id != manifest.artifact_id:
        findings.append(
            Finding(
                Severity.ERROR,
                CHECK_ARTIFACT_ID,
                f"record names artifact {identity.artifact_id!r}, manifest names "
                f"{manifest.artifact_id!r}",
            )
        )
    if identity.artifact_file_count != len(manifest.files):
        findings.append(
            Finding(
                Severity.ERROR,
                CHECK_FILE_COUNT,
                f"record counts {identity.artifact_file_count} artifact files, manifest "
                f"lists {len(manifest.files)}",
            )
        )

    by_path = {file.path: file for file in manifest.files}
    for shard in checkpoint.shards:
        manifest_file = by_path.get(shard.path)
        if manifest_file is None:
            findings.append(
                Finding(
                    Severity.ERROR,
                    CHECK_SHARD_COVERAGE,
                    f"shard {shard.path!r} is not in the manifest",
                )
            )
            continue
        if manifest_file.sha256 != shard.sha256:
            findings.append(
                Finding(
                    Severity.ERROR,
                    CHECK_SHARD_DIGEST,
                    f"shard {shard.path!r} pins {shard.sha256[:16]}… but the manifest "
                    f"records {manifest_file.sha256[:16]}…",
                )
            )
        if shard.data_offset_bytes + shard.data_size_bytes > shard.file_size_bytes:
            findings.append(
                Finding(
                    Severity.ERROR,
                    CHECK_SHARD_COVERAGE,
                    f"shard {shard.path!r} data window runs past the file",
                )
            )

    findings.extend(_tensor_bounds(checkpoint))
    return ValidationReport(findings=tuple(findings))


def _tensor_bounds(checkpoint: PreparedCheckpoint) -> list[Finding]:
    """Every tensor must land inside the shard it names.

    The encoder validates this on the way out, but a record can be edited or
    truncated after the fact, and it is the reader that pays for that.
    """
    findings: list[Finding] = []
    for tensor in checkpoint.tensors:
        if tensor.shard >= len(checkpoint.shards):
            findings.append(
                Finding(
                    Severity.ERROR,
                    CHECK_TENSOR_BOUNDS,
                    f"tensor {tensor.name!r} names shard {tensor.shard}, record has "
                    f"{len(checkpoint.shards)}",
                )
            )
            continue
        shard = checkpoint.shards[tensor.shard]
        if tensor.shard_offset_bytes + tensor.size_bytes > shard.data_size_bytes:
            findings.append(
                Finding(
                    Severity.ERROR,
                    CHECK_TENSOR_BOUNDS,
                    f"tensor {tensor.name!r} runs past the end of shard {shard.path!r}",
                )
            )
    return findings


def render_human(report: ValidationReport) -> str:
    if report.ok and not report.findings:
        return "validate: PASS — record matches its package, manifest and shards"
    lines = ["validate: FAIL" if not report.ok else "validate: PASS with warnings"]
    for finding in report.findings:
        lines.append(f"  [{finding.severity.value}] {finding.check}: {finding.detail}")
    return "\n".join(lines)
