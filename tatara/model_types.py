"""Typed contracts shared by model source adapters and inspection logic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class SourceKind(str, Enum):
    LOCAL = "local"
    HUGGING_FACE = "huggingface"


class EvidenceLevel(str, Enum):
    REMOTE_METADATA = "remote-metadata-only"
    LOCAL_HEADERS = "local-safetensors-headers-verified"


class FileRole(str, Enum):
    METADATA = "metadata"
    WEIGHT = "weight"


class DigestEvidence(str, Enum):
    COMPUTED = "computed"
    HUB_DECLARED = "hub-declared"
    NOT_COMPUTED = "not-computed"
    UNAVAILABLE = "unavailable"


class HashMode(str, Enum):
    NONE = "none"
    METADATA = "metadata"
    ALL = "all"


class InspectionTask(str, Enum):
    TEXT_GENERATION = "text-generation"
    MULTIMODAL_GENERATION = "multimodal-generation"


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    role: FileRole
    size_bytes: int | None
    sha256: str | None
    digest_evidence: DigestEvidence


@dataclass(frozen=True)
class InspectionScope:
    network_access: bool = False
    model_loaded: bool = False
    weight_headers_read: bool = False
    weight_payload_read: bool = False
    gpu_work: bool = False
    remote_code_executed: bool = False


@dataclass(frozen=True)
class TensorSummary:
    tensor_count: int | None
    tensor_bytes: int | None
    dtype_tensor_counts: Mapping[str, int]
    shard_count: int
    headers_verified: bool


@dataclass(frozen=True)
class QuantizationFacts:
    mode: str | None
    bits: int | None
    group_size: int | None


@dataclass(frozen=True)
class DimensionFacts:
    layers: int | None
    hidden: int | None
    vocabulary: int | None
    maximum_context: int | None


@dataclass(frozen=True)
class AttentionFacts:
    query_heads: int | None
    kv_heads: int | None
    head_dimension: int | None
    full_layers: int | None
    linear_layers: int | None


@dataclass(frozen=True)
class MoeFacts:
    experts: int | None
    active_experts: int | None


@dataclass(frozen=True)
class LinearAttentionFacts:
    key_heads: int | None
    key_dimension: int | None
    value_heads: int | None
    value_dimension: int | None
    convolution_width: int | None


@dataclass(frozen=True)
class TokenizerFacts:
    tokenizer_class: str | None
    model_maximum_context: int | None
    tokenizer_json_present: bool
    chat_template_present: bool
    eos_token_ids: tuple[int, ...]
    pad_token_id: int | None


@dataclass(frozen=True)
class ModelFacts:
    model_type: str | None
    architectures: tuple[str, ...]
    modalities: tuple[str, ...]
    dtype: str | None
    quantization: QuantizationFacts
    dimensions: DimensionFacts
    attention: AttentionFacts
    moe: MoeFacts
    linear_attention: LinearAttentionFacts
    tokenizer: TokenizerFacts


@dataclass(frozen=True)
class ModelSnapshot:
    kind: SourceKind
    identifier: str
    revision: str | None
    resolved_revision: str | None
    evidence: EvidenceLevel
    config: Mapping[str, Any]
    tokenizer_config: Mapping[str, Any]
    weight_index: Mapping[str, Any] | None
    files: tuple[ArtifactFile, ...]
    tensor_summary: TensorSummary
    repository_metadata: Mapping[str, Any]
    scope: InspectionScope
