"""Pure parsing and rendering for generated Qwen3.6 native plans."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from typing import Any

from tatara.artifact_manifest import ArtifactManifest
from tatara.model_types import FileRole


IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
UINT32_MAXIMUM = 2**32 - 1
UINT64_MAXIMUM = 2**64 - 1
MAXIMUM_MODEL_LAYERS = 4096
MAXIMUM_EXCLUDED_PREFIXES = 64
MAXIMUM_STOP_TOKENS = 8
MAXIMUM_TOKENIZER_BYTES = 64 * 1024 * 1024

# The tensor-naming table the package ships, in the declaration order of
# `TensorNaming` in include/tatara/model/qwen36_tensor_names.h. C++20 requires
# designated initializers in declaration order, so a reordering here fails the
# native build rather than composing silently wrong names.
TENSOR_NAME_FIELDS = (
    "model_prefix",
    "layer_prefix",
    "layer_index_suffix",
    "embedding_stem",
    "final_norm",
    "head_stem",
    "input_norm",
    "post_norm",
    "gated_delta_prefix",
    "gated_delta_qkv_stem",
    "gated_delta_z_stem",
    "gated_delta_b_stem",
    "gated_delta_a_stem",
    "gated_delta_out_stem",
    "gated_delta_conv",
    "gated_delta_a_log",
    "gated_delta_dt_bias",
    "gated_delta_norm",
    "attention_prefix",
    "attention_query_stem",
    "attention_key_stem",
    "attention_value_stem",
    "attention_out_stem",
    "attention_query_norm",
    "attention_key_norm",
    "mixture_prefix",
    "router_stem",
    "shared_router_stem",
    "expert_gate_stem",
    "expert_up_stem",
    "expert_down_stem",
    "shared_gate_stem",
    "shared_up_stem",
    "shared_down_stem",
    "quantized_weight",
    "quantized_scales",
    "quantized_biases",
)


class ModelPlanGenerationError(ValueError):
    pass


@dataclass(frozen=True)
class GeneratedModelPlan:
    identifier: str
    family: str
    package_sha256: str
    model_type: str
    layer_kinds: tuple[str, ...]
    attention_layer_period: int
    first_attention_layer: int
    hidden: int
    vocabulary: int
    query_heads: int
    key_value_heads: int
    head_dimension: int
    recurrent_heads: int
    state_dimension: int
    experts: int
    active_experts: int
    expert_dimension: int
    weight_format: str
    group_size: int
    initial_serving_capacity: int
    maximum_context: int
    tokenizer_kind: str
    tokenizer_data_path: str
    tokenizer_data_sha256: str
    tokenizer_data_size_bytes: int
    tokenizer_config_path: str
    tokenizer_config_sha256: str
    tokenizer_config_size_bytes: int
    tokenizer_normalization: str
    tokenizer_pretokenizer: str
    tokenizer_split_pattern: str
    tokenizer_decoder: str
    tokenizer_vocabulary: int
    tokenizer_populated_vocabulary: int
    chat_template_kind: str
    chat_template_path: str
    chat_template_sha256: str
    chat_template_size_bytes: int
    end_of_text_id: int
    message_start_id: int
    message_end_id: int
    thinking_start_id: int
    thinking_end_id: int
    padding_id: int
    stop_token_ids: tuple[int, ...]
    default_thinking: bool
    excluded_tensor_prefixes: tuple[str, ...]
    tensor_names: tuple[tuple[str, str], ...]
    artifact_id: str
    artifact_manifest_sha256: str
    artifact_format: str
    source_repository: str
    source_revision: str
    tensor_count: int
    tensor_bytes: int
    artifact_file_count: int
    weight_file_count: int


def parse_model_plan(
    text: str,
    artifact: ArtifactManifest,
    artifact_manifest_name: str,
    package_sha256: str,
    artifact_manifest_sha256: str,
) -> GeneratedModelPlan:
    try:
        value = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise ModelPlanGenerationError("Model package is not valid TOML") from error
    if value.get("schema_version") != 1:
        raise ModelPlanGenerationError("Unsupported model package schema")

    identifier = _identifier(value, "id")
    family = _identifier(value, "family")
    model_type = _identifier(value, "model_type")
    if _required_string(value, "artifact_manifest") != artifact_manifest_name:
        raise ModelPlanGenerationError("Model package selects a different artifact manifest")
    if model_type != artifact.model_type:
        raise ModelPlanGenerationError("Model package and artifact model types differ")
    if not SHA256.fullmatch(package_sha256) or not SHA256.fullmatch(
        artifact_manifest_sha256
    ):
        raise ModelPlanGenerationError("Model package identity digest is invalid")

    dimensions = _table(value, "dimensions")
    attention = _table(value, "attention")
    gated_delta = _table(value, "gated_delta")
    mixture_of_experts = _table(value, "moe")
    weights = _table(value, "weights")
    context = _table(value, "context")
    tokenizer = _table(value, "tokenizer")
    residency = _table(value, "residency")

    layer_count = _positive_uint32(dimensions, "layers")
    if layer_count > MAXIMUM_MODEL_LAYERS:
        raise ModelPlanGenerationError("Model package layer count is unreasonable")
    layer_period = _positive_uint32(attention, "layer_period")
    first_layer = _nonnegative_uint32(attention, "first_layer")
    if _nonnegative_uint32(attention, "index_base") != 0:
        raise ModelPlanGenerationError("Only zero-based layer indexes are supported")
    if first_layer >= layer_count:
        raise ModelPlanGenerationError("First attention layer exceeds the model")

    layer_kinds = tuple(
        "FullAttention"
        if index >= first_layer and (index - first_layer) % layer_period == 0
        else "GatedDelta"
        for index in range(layer_count)
    )
    full_attention_layers = layer_kinds.count("FullAttention")
    if full_attention_layers == 0 or full_attention_layers == layer_count:
        raise ModelPlanGenerationError("Hybrid model package must contain both layer kinds")

    query_heads = _positive_uint32(attention, "query_heads")
    key_value_heads = _positive_uint32(attention, "kv_heads")
    if query_heads % key_value_heads:
        raise ModelPlanGenerationError("Query heads must be divisible by key/value heads")

    experts = _positive_uint32(mixture_of_experts, "experts")
    active_experts = _positive_uint32(mixture_of_experts, "active_experts")
    expert_dimension = _positive_uint32(mixture_of_experts, "expert_dimension")
    if active_experts > experts:
        raise ModelPlanGenerationError("Active expert count exceeds expert count")

    weight_format = _required_string(weights, "format")
    if weight_format != "affine-q4":
        raise ModelPlanGenerationError(f"Unsupported weight format: {weight_format}")
    weight_file_count = sum(
        item.role is FileRole.WEIGHT for item in artifact.files
    )
    if weight_file_count == 0:
        raise ModelPlanGenerationError("Artifact manifest has no weight files")
    if artifact.tensor_count == 0 or artifact.tensor_count > UINT32_MAXIMUM:
        raise ModelPlanGenerationError("Artifact tensor count is invalid")
    if artifact.tensor_bytes == 0 or artifact.tensor_bytes > UINT64_MAXIMUM:
        raise ModelPlanGenerationError("Artifact tensor byte count is invalid")
    if len(artifact.files) > UINT32_MAXIMUM:
        raise ModelPlanGenerationError("Artifact file count is invalid")

    hidden = _positive_uint32(dimensions, "hidden")
    vocabulary = _positive_uint32(dimensions, "vocabulary")
    initial_serving_capacity = _positive_uint32(
        context, "initial_serving_capacity"
    )
    maximum_context = _positive_uint32(context, "maximum_capacity")
    if initial_serving_capacity > maximum_context:
        raise ModelPlanGenerationError(
            "Initial serving capacity exceeds maximum context"
        )

    tokenizer_kind = _required_string(tokenizer, "kind")
    if tokenizer_kind != "byte-level-bpe":
        raise ModelPlanGenerationError(
            f"Unsupported tokenizer kind: {tokenizer_kind}"
        )
    tokenizer_normalization = _required_string(tokenizer, "normalization")
    if tokenizer_normalization != "nfc":
        raise ModelPlanGenerationError(
            f"Unsupported tokenizer normalization: {tokenizer_normalization}"
        )
    tokenizer_pretokenizer = _required_string(tokenizer, "pretokenizer")
    if tokenizer_pretokenizer != "qwen-regex-byte-level-v1":
        raise ModelPlanGenerationError(
            f"Unsupported tokenizer pretokenizer: {tokenizer_pretokenizer}"
        )
    tokenizer_decoder = _required_string(tokenizer, "decoder")
    if tokenizer_decoder != "byte-level":
        raise ModelPlanGenerationError(
            f"Unsupported tokenizer decoder: {tokenizer_decoder}"
        )
    chat_template_kind = _required_string(tokenizer, "template_kind")
    if chat_template_kind != "qwen36-text-v1":
        raise ModelPlanGenerationError(
            f"Unsupported chat template: {chat_template_kind}"
        )
    tokenizer_vocabulary = _positive_uint32(tokenizer, "vocabulary")
    if tokenizer_vocabulary != vocabulary:
        raise ModelPlanGenerationError(
            "Tokenizer and model vocabulary sizes differ"
        )
    tokenizer_populated_vocabulary = _positive_uint32(
        tokenizer, "populated_vocabulary"
    )
    if tokenizer_populated_vocabulary > tokenizer_vocabulary:
        raise ModelPlanGenerationError(
            "Populated tokenizer vocabulary exceeds the model vocabulary"
        )
    tokenizer_data_path = _required_string(tokenizer, "data_path")
    tokenizer_data_sha256 = _required_sha256(tokenizer, "data_sha256")
    tokenizer_data_size_bytes = _artifact_metadata_size(
        artifact, tokenizer_data_path, tokenizer_data_sha256
    )
    if tokenizer_data_size_bytes > MAXIMUM_TOKENIZER_BYTES:
        raise ModelPlanGenerationError("Tokenizer data exceeds the admitted bound")
    tokenizer_config_path = _required_string(tokenizer, "config_path")
    tokenizer_config_sha256 = _required_sha256(tokenizer, "config_sha256")
    tokenizer_config_size_bytes = _artifact_metadata_size(
        artifact, tokenizer_config_path, tokenizer_config_sha256
    )
    chat_template_path = _required_string(tokenizer, "template_path")
    chat_template_sha256 = _required_sha256(tokenizer, "template_sha256")
    chat_template_size_bytes = _artifact_metadata_size(
        artifact, chat_template_path, chat_template_sha256
    )
    if len(
        {tokenizer_data_path, tokenizer_config_path, chat_template_path}
    ) != 3:
        raise ModelPlanGenerationError("Tokenizer artifact paths must be distinct")

    token_ids = {
        name: _nonnegative_uint32(tokenizer, name)
        for name in (
            "end_of_text_id",
            "message_start_id",
            "message_end_id",
            "thinking_start_id",
            "thinking_end_id",
            "padding_id",
        )
    }
    if any(
        token_id >= tokenizer_populated_vocabulary
        for token_id in token_ids.values()
    ):
        raise ModelPlanGenerationError(
            "Tokenizer control token is not in the populated vocabulary"
        )
    stop_token_ids = _token_id_list(
        tokenizer,
        "stop_token_ids",
        tokenizer_populated_vocabulary,
        MAXIMUM_STOP_TOKENS,
    )
    if (
        token_ids["end_of_text_id"] not in stop_token_ids
        or token_ids["message_end_id"] not in stop_token_ids
    ):
        raise ModelPlanGenerationError(
            "Qwen stop tokens must include end-of-text and message-end"
        )
    default_thinking = tokenizer.get("default_thinking")
    if not isinstance(default_thinking, bool):
        raise ModelPlanGenerationError(
            "Model package field is invalid: default_thinking"
        )

    return GeneratedModelPlan(
        identifier=identifier,
        family=family,
        package_sha256=package_sha256,
        model_type=model_type,
        layer_kinds=layer_kinds,
        attention_layer_period=layer_period,
        first_attention_layer=first_layer,
        hidden=hidden,
        vocabulary=vocabulary,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dimension=_positive_uint32(attention, "head_dimension"),
        recurrent_heads=_positive_uint32(gated_delta, "recurrent_heads"),
        state_dimension=_positive_uint32(gated_delta, "state_dimension"),
        experts=experts,
        active_experts=active_experts,
        expert_dimension=expert_dimension,
        weight_format=weight_format,
        group_size=_positive_uint32(weights, "group_size"),
        initial_serving_capacity=initial_serving_capacity,
        maximum_context=maximum_context,
        tokenizer_kind=tokenizer_kind,
        tokenizer_data_path=tokenizer_data_path,
        tokenizer_data_sha256=tokenizer_data_sha256,
        tokenizer_data_size_bytes=tokenizer_data_size_bytes,
        tokenizer_config_path=tokenizer_config_path,
        tokenizer_config_sha256=tokenizer_config_sha256,
        tokenizer_config_size_bytes=tokenizer_config_size_bytes,
        tokenizer_normalization=tokenizer_normalization,
        tokenizer_pretokenizer=tokenizer_pretokenizer,
        tokenizer_split_pattern=_required_string(tokenizer, "split_pattern"),
        tokenizer_decoder=tokenizer_decoder,
        tokenizer_vocabulary=tokenizer_vocabulary,
        tokenizer_populated_vocabulary=tokenizer_populated_vocabulary,
        chat_template_kind=chat_template_kind,
        chat_template_path=chat_template_path,
        chat_template_sha256=chat_template_sha256,
        chat_template_size_bytes=chat_template_size_bytes,
        end_of_text_id=token_ids["end_of_text_id"],
        message_start_id=token_ids["message_start_id"],
        message_end_id=token_ids["message_end_id"],
        thinking_start_id=token_ids["thinking_start_id"],
        thinking_end_id=token_ids["thinking_end_id"],
        padding_id=token_ids["padding_id"],
        stop_token_ids=stop_token_ids,
        default_thinking=default_thinking,
        excluded_tensor_prefixes=_excluded_tensor_prefixes(residency),
        tensor_names=_tensor_names(_table(value, "tensor_names")),
        artifact_id=artifact.artifact_id,
        artifact_manifest_sha256=artifact_manifest_sha256,
        artifact_format=artifact.format,
        source_repository=artifact.source_repository,
        source_revision=artifact.source_revision,
        tensor_count=artifact.tensor_count,
        tensor_bytes=artifact.tensor_bytes,
        artifact_file_count=len(artifact.files),
        weight_file_count=weight_file_count,
    )


def render_model_plan_header(plan: GeneratedModelPlan) -> str:
    identifier = _cpp_string_literal(plan.identifier)
    family = _cpp_string_literal(plan.family)
    package_sha256 = _cpp_string_literal(plan.package_sha256)
    artifact_id = _cpp_string_literal(plan.artifact_id)
    model_type = _cpp_string_literal(plan.model_type)
    artifact_format = _cpp_string_literal(plan.artifact_format)
    source_repository = _cpp_string_literal(plan.source_repository)
    source_revision = _cpp_string_literal(plan.source_revision)
    artifact_manifest_sha256 = _cpp_string_literal(plan.artifact_manifest_sha256)
    tokenizer_data_path = _cpp_string_literal(plan.tokenizer_data_path)
    tokenizer_data_sha256 = _cpp_string_literal(plan.tokenizer_data_sha256)
    tokenizer_config_path = _cpp_string_literal(plan.tokenizer_config_path)
    tokenizer_config_sha256 = _cpp_string_literal(plan.tokenizer_config_sha256)
    tokenizer_split_pattern = _cpp_string_literal(plan.tokenizer_split_pattern)
    chat_template_path = _cpp_string_literal(plan.chat_template_path)
    chat_template_sha256 = _cpp_string_literal(plan.chat_template_sha256)
    stop_token_ids = ", ".join(str(token_id) for token_id in plan.stop_token_ids)
    excluded_prefixes = "".join(
        f"    {_cpp_string_literal(prefix)},\n" for prefix in plan.excluded_tensor_prefixes
    )
    tensor_names = "".join(
        f"    .{name} = {_cpp_string_literal(text)},\n" for name, text in plan.tensor_names
    )
    return f'''#pragma once

#include "tatara/model/qwen36_plan.h"
#include "tatara/model/qwen36_tensor_names.h"

#include <array>

namespace tatara::model::generated {{

inline constexpr std::string_view kModelPackageId = {identifier};
inline constexpr std::string_view kModelPackageSha256 = {package_sha256};
inline constexpr ArtifactIdentity kArtifactIdentity{{
    .id = {artifact_id},
    .model_type = {model_type},
    .format = {artifact_format},
    .source_repository = {source_repository},
    .source_revision = {source_revision},
    .manifest_sha256 = {artifact_manifest_sha256},
    .tensor_count = {plan.tensor_count},
    .tensor_bytes = {plan.tensor_bytes}ULL,
    .file_count = {plan.artifact_file_count},
    .weight_file_count = {plan.weight_file_count},
}};

// Tensor-name prefixes this package does not serve. Layout planning gives them
// no place in the model image and population never copies them.
inline constexpr std::array<std::string_view, {len(plan.excluded_tensor_prefixes)}> kExcludedTensorPrefixes{{
{excluded_prefixes}}};

}} // namespace tatara::model::generated

namespace tatara::model::qwen36::generated {{

inline constexpr TensorNaming kTensorNaming{{
{tensor_names}}};

inline constexpr StaticModelPlan<{len(plan.layer_kinds)}> kModelPlan{{
    .id = ::tatara::model::generated::kModelPackageId,
    .family = {family},
    .package_sha256 = ::tatara::model::generated::kModelPackageSha256,
    .artifact = ::tatara::model::generated::kArtifactIdentity,
    .dimensions =
        ModelDimensions{{
            .hidden = {plan.hidden},
            .vocabulary = {plan.vocabulary},
        }},
    .attention =
        AttentionSpec{{
            .query_heads = {plan.query_heads},
            .key_value_heads = {plan.key_value_heads},
            .head_dimension = {plan.head_dimension},
        }},
    .gated_delta =
        GatedDeltaSpec{{
            .recurrent_heads = {plan.recurrent_heads},
            .state_dimension = {plan.state_dimension},
        }},
    .mixture_of_experts =
        MixtureOfExpertsSpec{{
            .experts = {plan.experts},
            .active_experts = {plan.active_experts},
            .expert_dimension = {plan.expert_dimension},
        }},
    .weights =
        WeightSpec{{
            .format = WeightFormat::AffineQ4,
            .group_size = {plan.group_size},
        }},
    .tokenizer =
        TokenizerSpec{{
            .kind = TokenizerKind::ByteLevelBpe,
            .normalization = TokenizerNormalization::Nfc,
            .pretokenizer = TokenizerPretokenizer::QwenRegexByteLevelV1,
            .decoder = TokenizerDecoder::ByteLevel,
            .template_kind = ChatTemplateKind::Qwen36TextV1,
            .data_path = {tokenizer_data_path},
            .data_sha256 = {tokenizer_data_sha256},
            .data_size_bytes = {plan.tokenizer_data_size_bytes}ULL,
            .config_path = {tokenizer_config_path},
            .config_sha256 = {tokenizer_config_sha256},
            .config_size_bytes = {plan.tokenizer_config_size_bytes}ULL,
            .template_path = {chat_template_path},
            .template_sha256 = {chat_template_sha256},
            .template_size_bytes = {plan.chat_template_size_bytes}ULL,
            .split_pattern = {tokenizer_split_pattern},
            .vocabulary = {plan.tokenizer_vocabulary},
            .populated_vocabulary = {plan.tokenizer_populated_vocabulary},
            .maximum_context = {plan.maximum_context},
            .end_of_text_id = {plan.end_of_text_id},
            .message_start_id = {plan.message_start_id},
            .message_end_id = {plan.message_end_id},
            .thinking_start_id = {plan.thinking_start_id},
            .thinking_end_id = {plan.thinking_end_id},
            .padding_id = {plan.padding_id},
            .stop_token_ids =
                std::array<std::uint32_t, kMaximumStopTokens>{{{stop_token_ids}}},
            .stop_token_count = {len(plan.stop_token_ids)},
            .default_thinking = {_cpp_bool(plan.default_thinking)},
        }},
    .initial_serving_capacity = {plan.initial_serving_capacity},
    .layers = make_hybrid_layer_schedule<{len(plan.layer_kinds)}, {plan.attention_layer_period}, {plan.first_attention_layer}>(),
}};

static_assert(valid_model_plan(kModelPlan));

}} // namespace tatara::model::qwen36::generated
'''


def _cpp_string_literal(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _cpp_bool(value: bool) -> str:
    return "true" if value else "false"


def _table(value: dict[str, Any], name: str) -> dict[str, Any]:
    item = value.get(name)
    if not isinstance(item, dict):
        raise ModelPlanGenerationError(f"Model package has no [{name}] table")
    return item


def _required_string(value: dict[str, Any], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ModelPlanGenerationError(f"Model package field is invalid: {name}")
    return item


def _required_sha256(value: dict[str, Any], name: str) -> str:
    item = _required_string(value, name)
    if not SHA256.fullmatch(item):
        raise ModelPlanGenerationError(
            f"Model package SHA-256 is invalid: {name}"
        )
    return item


def _artifact_metadata_size(
    artifact: ArtifactManifest, path: str, sha256: str
) -> int:
    matches = [item for item in artifact.files if item.path == path]
    if len(matches) != 1:
        raise ModelPlanGenerationError(
            f"Tokenizer artifact file is not declared exactly once: {path}"
        )
    item = matches[0]
    if item.role is not FileRole.METADATA:
        raise ModelPlanGenerationError(
            f"Tokenizer artifact file is not metadata: {path}"
        )
    if item.sha256 != sha256:
        raise ModelPlanGenerationError(
            f"Tokenizer artifact digest differs from manifest: {path}"
        )
    if item.size_bytes <= 0:
        raise ModelPlanGenerationError(
            f"Tokenizer artifact file is empty: {path}"
        )
    return item.size_bytes


def _token_id_list(
    value: dict[str, Any],
    name: str,
    vocabulary: int,
    maximum_count: int,
) -> tuple[int, ...]:
    item = value.get(name)
    if (
        not isinstance(item, list)
        or not item
        or len(item) > maximum_count
    ):
        raise ModelPlanGenerationError(
            f"Model package token list is invalid: {name}"
        )
    for token_id in item:
        if (
            not isinstance(token_id, int)
            or isinstance(token_id, bool)
            or token_id < 0
            or token_id >= vocabulary
        ):
            raise ModelPlanGenerationError(
                f"Model package token list is invalid: {name}"
            )
    if len(set(item)) != len(item):
        raise ModelPlanGenerationError(
            f"Model package token list repeats an ID: {name}"
        )
    return tuple(item)


def _identifier(value: dict[str, Any], name: str) -> str:
    item = _required_string(value, name)
    if not IDENTIFIER.fullmatch(item):
        raise ModelPlanGenerationError(f"Model package identifier is invalid: {name}")
    return item


def _excluded_tensor_prefixes(residency: dict[str, Any]) -> tuple[str, ...]:
    item = residency.get("excluded_tensor_prefixes")
    if not isinstance(item, list) or len(item) > MAXIMUM_EXCLUDED_PREFIXES:
        raise ModelPlanGenerationError("Model package excluded tensor prefixes are invalid")
    for prefix in item:
        if not isinstance(prefix, str) or not prefix:
            raise ModelPlanGenerationError("Model package excluded tensor prefix is invalid")
    if len(set(item)) != len(item):
        raise ModelPlanGenerationError("Model package repeats an excluded tensor prefix")
    return tuple(item)


def _tensor_names(table: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    if set(table) != set(TENSOR_NAME_FIELDS):
        raise ModelPlanGenerationError("Model package tensor naming table is incomplete")
    return tuple((name, _required_string(table, name)) for name in TENSOR_NAME_FIELDS)


def _nonnegative_uint32(value: dict[str, Any], name: str) -> int:
    item = value.get(name)
    if (
        not isinstance(item, int)
        or isinstance(item, bool)
        or item < 0
        or item > UINT32_MAXIMUM
    ):
        raise ModelPlanGenerationError(f"Model package field is invalid: {name}")
    return item


def _positive_uint32(value: dict[str, Any], name: str) -> int:
    item = _nonnegative_uint32(value, name)
    if item == 0:
        raise ModelPlanGenerationError(f"Model package field is invalid: {name}")
    return item
