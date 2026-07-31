"""Model metadata sources for local artifacts and Hugging Face repositories."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from tatara.model_types import (
    ArtifactFile,
    DigestEvidence,
    EvidenceLevel,
    FileRole,
    HashMode,
    InspectionScope,
    ModelSnapshot,
    SourceKind,
    TensorSummary,
)
from tatara.safetensors import SafetensorsError, read_header, tensor_records


MAX_JSON_BYTES = 64 * 1024 * 1024
HF_ENDPOINT = "https://huggingface.co"


class ModelSourceError(ValueError):
    pass


def load_local(
    path: Path, hash_mode: HashMode = HashMode.METADATA
) -> ModelSnapshot:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise ModelSourceError(f"Model directory does not exist: {path}")

    config = _read_local_json(root / "config.json", required=True)
    tokenizer = _read_local_json(root / "tokenizer_config.json", required=False)
    index_path = root / "model.safetensors.index.json"
    index = _read_local_json(index_path, required=False)
    weight_names = _weight_names(root, index)
    tensor_summary = _verify_local_weights(root, index, weight_names)
    files = _local_files(root, weight_names, hash_mode)

    return ModelSnapshot(
        kind=SourceKind.LOCAL,
        identifier=str(root),
        revision=None,
        resolved_revision=None,
        evidence=EvidenceLevel.LOCAL_HEADERS,
        config=config,
        tokenizer_config=tokenizer,
        weight_index=index,
        files=tuple(files),
        tensor_summary=tensor_summary,
        repository_metadata={},
        scope=InspectionScope(
            weight_headers_read=True,
            weight_payload_read=hash_mode is HashMode.ALL,
        ),
    )


def load_huggingface(
    reference: str,
    allow_network: bool,
    fetch_json: Callable[[str, dict[str, str]], dict[str, Any]] | None = None,
) -> ModelSnapshot:
    if not allow_network:
        raise ModelSourceError(
            "Hugging Face inspection requires --allow-network; weight shards are not fetched"
        )
    repo_id, revision = parse_huggingface_reference(reference)
    request_json = fetch_json or _http_json
    headers = _huggingface_headers()
    api_url = (
        f"{HF_ENDPOINT}/api/models/{quote(repo_id, safe='/')}/revision/"
        f"{quote(revision, safe='')}?blobs=true"
    )
    info = request_json(api_url, headers)
    resolved = info.get("sha")
    if not isinstance(resolved, str) or not resolved:
        raise ModelSourceError("Hugging Face did not return a resolved commit")

    siblings = info.get("siblings")
    if not isinstance(siblings, list):
        raise ModelSourceError("Hugging Face response has no file manifest")
    manifest = _remote_manifest(siblings)
    names = {record.path for record in manifest}
    if "config.json" not in names:
        raise ModelSourceError("Hugging Face repository has no config.json")

    def remote_json(name: str, required: bool) -> dict[str, Any]:
        if name not in names:
            if required:
                raise ModelSourceError(f"Hugging Face repository has no {name}")
            return {}
        url = (
            f"{HF_ENDPOINT}/{quote(repo_id, safe='/')}/resolve/"
            f"{quote(resolved, safe='')}/{quote(name, safe='/')}"
        )
        return request_json(url, headers)

    config = remote_json("config.json", True)
    tokenizer = remote_json("tokenizer_config.json", False)
    index = remote_json("model.safetensors.index.json", False) or None
    tensor_summary = _remote_tensor_summary(index, manifest)
    repository_metadata = {
        "gated": bool(info.get("gated")),
        "private": bool(info.get("private")),
        "pipeline_tag": info.get("pipeline_tag"),
        "library_name": info.get("library_name"),
        "tags": info.get("tags") if isinstance(info.get("tags"), list) else [],
        "card_data": info.get("cardData")
        if isinstance(info.get("cardData"), dict)
        else {},
        "publisher_reported_evaluations": info.get("evalResults")
        if isinstance(info.get("evalResults"), list)
        else [],
    }
    return ModelSnapshot(
        kind=SourceKind.HUGGING_FACE,
        identifier=repo_id,
        revision=revision,
        resolved_revision=resolved,
        evidence=EvidenceLevel.REMOTE_METADATA,
        config=config,
        tokenizer_config=tokenizer,
        weight_index=index,
        files=tuple(manifest),
        tensor_summary=tensor_summary,
        repository_metadata=repository_metadata,
        scope=InspectionScope(network_access=True),
    )


def parse_huggingface_reference(reference: str) -> tuple[str, str]:
    if not reference.startswith("hf://"):
        raise ModelSourceError("Hugging Face references must start with hf://")
    value = reference[5:]
    repo_id, separator, revision = value.partition("@")
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        raise ModelSourceError("Expected hf://owner/model[@revision]")
    if any(part in {".", ".."} for part in repo_id.split("/")):
        raise ModelSourceError("Invalid Hugging Face repository identifier")
    if separator and not revision:
        raise ModelSourceError("Hugging Face revision cannot be empty")
    return repo_id, revision or "main"


def _read_local_json(path: Path, required: bool) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise ModelSourceError(f"Required model metadata is missing: {path.name}")
        return {}
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ModelSourceError(f"Model metadata is too large: {path.name}")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelSourceError(f"Invalid JSON metadata: {path.name}") from error
    if not isinstance(value, dict):
        raise ModelSourceError(f"Model metadata is not an object: {path.name}")
    return value


def _weight_names(root: Path, index: dict[str, Any]) -> tuple[str, ...]:
    if index:
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelSourceError("Safetensors index has no weight_map")
        raw_names = set(weight_map.values())
        if any(not isinstance(name, str) for name in raw_names):
            raise ModelSourceError("Safetensors index contains an invalid shard name")
        names = sorted(raw_names)
    else:
        names = [path.name for path in sorted(root.glob("*.safetensors"))]
    if not names:
        raise ModelSourceError("No Safetensors weights were found")
    for name in names:
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise ModelSourceError(f"Unsafe Safetensors shard path: {name}")
    return tuple(names)


def _verify_local_weights(
    root: Path, index: dict[str, Any], weight_names: tuple[str, ...]
) -> TensorSummary:
    headers: dict[str, dict[str, Any]] = {}
    dtype_counts: dict[str, int] = {}
    tensor_bytes = 0
    try:
        for name in weight_names:
            path = root / name
            if not path.is_file():
                raise ModelSourceError(f"Safetensors shard is missing: {name}")
            tensors = tensor_records(read_header(path))
            headers[name] = tensors
            for record in tensors.values():
                dtype = record["dtype"]
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                start, end = record["data_offsets"]
                tensor_bytes += end - start
    except (OSError, SafetensorsError) as error:
        raise ModelSourceError(str(error)) from error

    if index:
        weight_map = index["weight_map"]
        indexed = set(weight_map)
        actual = {tensor for tensors in headers.values() for tensor in tensors}
        if indexed != actual:
            missing = sorted(indexed - actual)[:3]
            extra = sorted(actual - indexed)[:3]
            raise ModelSourceError(
                f"Safetensors index/header mismatch; missing={missing}, extra={extra}"
            )
        for tensor, shard in weight_map.items():
            if tensor not in headers[shard]:
                raise ModelSourceError(
                    f"Safetensors index maps {tensor} to the wrong shard"
                )
        declared = index.get("metadata", {}).get("total_size")
        if declared is not None and declared != tensor_bytes:
            raise ModelSourceError(
                f"Safetensors total_size mismatch: declared {declared}, actual {tensor_bytes}"
            )

    return TensorSummary(
        tensor_count=sum(len(value) for value in headers.values()),
        tensor_bytes=tensor_bytes,
        dtype_tensor_counts=dict(sorted(dtype_counts.items())),
        shard_count=len(weight_names),
        headers_verified=True,
    )


def _local_files(
    root: Path, weight_names: tuple[str, ...], hash_mode: HashMode
) -> list[ArtifactFile]:
    weights = set(weight_names)
    paths = {path.relative_to(root).as_posix(): path for path in root.iterdir() if path.is_file()}
    for name in weight_names:
        paths[name] = root / name
    records = []
    for name, path in sorted(paths.items()):
        is_weight = name in weights
        should_hash = hash_mode is HashMode.ALL or (
            hash_mode is HashMode.METADATA and not is_weight
        )
        records.append(
            ArtifactFile(
                path=name,
                role=FileRole.WEIGHT if is_weight else FileRole.METADATA,
                size_bytes=path.stat().st_size,
                sha256=_sha256(path) if should_hash else None,
                digest_evidence=DigestEvidence.COMPUTED
                if should_hash
                else DigestEvidence.NOT_COMPUTED,
            )
        )
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _remote_manifest(siblings: list[Any]) -> list[ArtifactFile]:
    records = []
    for value in siblings:
        if not isinstance(value, dict):
            continue
        name = value.get("rfilename")
        if not isinstance(name, str) or not name:
            continue
        lfs = value.get("lfs") if isinstance(value.get("lfs"), dict) else {}
        size = value.get("size")
        if not isinstance(size, int):
            size = lfs.get("size") if isinstance(lfs.get("size"), int) else None
        digest = lfs.get("sha256") if isinstance(lfs.get("sha256"), str) else None
        records.append(
            ArtifactFile(
                path=name,
                role=FileRole.WEIGHT
                if name.endswith((".safetensors", ".gguf"))
                else FileRole.METADATA,
                size_bytes=size,
                sha256=digest,
                digest_evidence=DigestEvidence.HUB_DECLARED
                if digest
                else DigestEvidence.UNAVAILABLE,
            )
        )
    return sorted(records, key=lambda item: item.path)


def _remote_tensor_summary(
    index: dict[str, Any] | None, manifest: list[ArtifactFile]
) -> TensorSummary:
    if index:
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ModelSourceError("Remote Safetensors index has no weight_map")
        shards = set(weight_map.values())
        names = {record.path for record in manifest}
        missing = sorted(name for name in shards if name not in names)
        if missing:
            raise ModelSourceError(f"Remote Safetensors shards are missing: {missing[:3]}")
        declared = index.get("metadata", {}).get("total_size")
        if not isinstance(declared, int) or declared < 0:
            declared = None
        return TensorSummary(
            tensor_count=len(weight_map),
            tensor_bytes=declared,
            dtype_tensor_counts={},
            shard_count=len(shards),
            headers_verified=False,
        )
    weights = [record for record in manifest if record.path.endswith(".safetensors")]
    if not weights:
        return TensorSummary(
            tensor_count=None,
            tensor_bytes=None,
            dtype_tensor_counts={},
            shard_count=0,
            headers_verified=False,
        )
    return TensorSummary(
        tensor_count=None,
        tensor_bytes=None,
        dtype_tensor_counts={},
        shard_count=len(weights),
        headers_verified=False,
    )


def _huggingface_headers() -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "tatara/0.0.1"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=20) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > MAX_JSON_BYTES:
                raise ModelSourceError("Remote metadata exceeds the size limit")
            payload = response.read(MAX_JSON_BYTES + 1)
    except HTTPError as error:
        raise ModelSourceError(f"Hugging Face request failed with HTTP {error.code}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise ModelSourceError(f"Hugging Face request failed: {error}") from error
    if len(payload) > MAX_JSON_BYTES:
        raise ModelSourceError("Remote metadata exceeds the size limit")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelSourceError("Hugging Face returned invalid JSON metadata") from error
    if not isinstance(value, dict):
        raise ModelSourceError("Hugging Face metadata is not an object")
    return value
