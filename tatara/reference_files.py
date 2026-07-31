"""Filesystem adapter for sealed reference identities."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from tatara.reference_contract import (
    ReferenceContract,
    ReferenceFileObservation,
    SourceFileObservation,
)


def observe_reference_files(
    contract: ReferenceContract, root: Path, roles: frozenset[str]
) -> tuple[ReferenceFileObservation, ...]:
    observations = []
    for item in contract.files:
        if item.role not in roles:
            continue
        path = root / item.path
        stat = path.stat()
        observations.append(
            ReferenceFileObservation(
                identifier=item.identifier,
                size_bytes=stat.st_size,
                sha256=_sha256(path),
            )
        )
    return tuple(observations)


def observe_source_files(
    contract: ReferenceContract, root: Path
) -> tuple[SourceFileObservation, ...]:
    observations = []
    for item in contract.source_files:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "cat-file",
                "blob",
                f"{contract.source_commit}:{item.path}",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise OSError(
                f"Cannot read source at {contract.source_commit}: {item.path}"
            )
        observations.append(
            SourceFileObservation(
                path=item.path,
                size_bytes=len(result.stdout),
                sha256=hashlib.sha256(result.stdout).hexdigest(),
            )
        )
    return tuple(observations)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
