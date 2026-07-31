"""Model-centric compatibility and capacity inspection."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from tatara.capabilities import architecture_for
from tatara.model_types import (
    AttentionFacts,
    DimensionFacts,
    FileRole,
    InspectionTask,
    LinearAttentionFacts,
    ModelFacts,
    ModelSnapshot,
    MoeFacts,
    QuantizationFacts,
    SourceKind,
    TokenizerFacts,
)


SCHEMA_VERSION = 1
GIB = 1024 ** 3


def inspect_snapshot(
    snapshot: ModelSnapshot,
    context_tokens: int,
    slots: int,
    task: InspectionTask = InspectionTask.TEXT_GENERATION,
) -> dict[str, Any]:
    if context_tokens < 1:
        raise ValueError("context must be at least one token")
    if slots < 1:
        raise ValueError("slots must be at least one")
    config = snapshot.config
    text = config.get("text_config")
    if not isinstance(text, dict):
        text = config
    model_type = config.get("model_type") or text.get("model_type")
    architectures = config.get("architectures")
    if not isinstance(architectures, list):
        architectures = []

    facts = _model_facts(
        config,
        text,
        snapshot.tokenizer_config,
        {value.path for value in snapshot.files},
        model_type,
        architectures,
    )
    artifact = _artifact_report(snapshot)
    compatibility = _compatibility(snapshot, facts, artifact["format"], task)
    capacity = _capacity(snapshot, facts, context_tokens, slots)
    evaluations = {
        "publisher_reported": snapshot.repository_metadata.get(
            "publisher_reported_evaluations", []
        ),
        "tatara_validated": [],
        "warning": (
            "Publisher-reported evaluations are unverified metadata, not Tatara validation."
            if snapshot.repository_metadata.get("publisher_reported_evaluations")
            else None
        ),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "command": "inspect",
        "task": task.value,
        "source": {
            "kind": snapshot.kind.value,
            "identifier": snapshot.identifier,
            "requested_revision": snapshot.revision,
            "resolved_revision": snapshot.resolved_revision,
            "evidence": snapshot.evidence.value,
        },
        "model": asdict(facts),
        "artifact": artifact,
        "compatibility": compatibility,
        "capacity": capacity,
        "evaluations": evaluations,
        "repository_metadata": snapshot.repository_metadata,
        "scope": asdict(snapshot.scope),
    }


def render_human(report: dict[str, Any]) -> str:
    source = report["source"]
    model = report["model"]
    artifact = report["artifact"]
    compatibility = report["compatibility"]
    capacity = report["capacity"]
    status = compatibility["outcome"].upper().replace("-", " ")
    lines = [f"Tatara model inspection: {status}"]
    lines.append(f"  Source: {source['kind']} {source['identifier']}")
    if source["resolved_revision"]:
        lines.append(f"  Revision: {source['resolved_revision']}")
    lines.append(f"  Evidence: {source['evidence']}")
    lines.append(f"  Requested task: {report['task']}")
    lines.append(
        f"  Architecture: {model['model_type'] or 'unknown'} "
        f"({compatibility['architecture_status']})"
    )
    lines.append(f"  Modalities: {', '.join(model['modalities'])}")
    lines.append(
        f"  Weights: {_format_bytes(artifact['tensor_bytes'])}; "
        f"{artifact['tensor_count'] if artifact['tensor_count'] is not None else 'unknown'} "
        f"tensors in {artifact['shard_count']} shard(s)"
    )
    if compatibility["missing_capabilities"]:
        lines.append(
            "  Missing: " + ", ".join(compatibility["missing_capabilities"])
        )
    if compatibility["unrequested_modalities"]:
        lines.append(
            "  Outside requested task: "
            + ", ".join(compatibility["unrequested_modalities"])
        )
    lines.append(
        f"  Capacity model: {capacity['context_tokens']} context x "
        f"{capacity['slots']} slot(s), lower bound "
        f"{_format_bytes(capacity['known_model_bytes_lower_bound'])}"
    )
    if capacity["unresolved_components"]:
        lines.append(
            "  Unresolved: " + ", ".join(capacity["unresolved_components"])
        )
    published = report["evaluations"]["publisher_reported"]
    if published:
        lines.append(
            f"  Evaluations: {len(published)} publisher-reported record(s), not validated"
        )
    manifest = report.get("artifact_manifest")
    if manifest:
        status = "MATCH" if manifest["matched"] else "MISMATCH"
        lines.append(
            f"  Artifact manifest: {status}; {manifest['checked_files']} file(s) checked"
        )
        for mismatch in manifest["mismatches"][:5]:
            lines.append(f"    {mismatch}")
    lines.append("  INFO  no model was loaded, no GPU work ran, and no remote code executed")
    return "\n".join(lines)


def _model_facts(
    config: Mapping[str, Any],
    text: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    file_names: set[str],
    model_type: Any,
    architectures: list[Any],
) -> ModelFacts:
    modalities = ["text"]
    if isinstance(config.get("vision_config"), dict):
        modalities.append("vision")
    if isinstance(config.get("audio_config"), dict):
        modalities.append("audio")
    quantization = config.get("quantization") or config.get("quantization_config")
    if not isinstance(quantization, dict):
        quantization = {}
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list):
        layer_types = []
    dtype = text.get("dtype") or config.get("torch_dtype")
    return ModelFacts(
        model_type=model_type if isinstance(model_type, str) else None,
        architectures=tuple(
            value for value in architectures if isinstance(value, str)
        ),
        modalities=tuple(modalities),
        dtype=dtype if isinstance(dtype, str) else None,
        quantization=QuantizationFacts(
            mode=_string(quantization.get("mode") or quantization.get("quant_method")),
            bits=_integer(quantization.get("bits")),
            group_size=_integer(quantization.get("group_size")),
        ),
        dimensions=DimensionFacts(
            layers=_integer(text.get("num_hidden_layers")),
            hidden=_integer(text.get("hidden_size")),
            vocabulary=_integer(text.get("vocab_size")),
            maximum_context=_integer(text.get("max_position_embeddings")),
        ),
        attention=AttentionFacts(
            query_heads=_integer(text.get("num_attention_heads")),
            kv_heads=_integer(text.get("num_key_value_heads")),
            head_dimension=_integer(text.get("head_dim")),
            full_layers=layer_types.count("full_attention") if layer_types else None,
            linear_layers=layer_types.count("linear_attention") if layer_types else None,
        ),
        moe=MoeFacts(
            experts=_integer(text.get("num_experts")),
            active_experts=_integer(text.get("num_experts_per_tok")),
        ),
        linear_attention=LinearAttentionFacts(
            key_heads=_integer(text.get("linear_num_key_heads")),
            key_dimension=_integer(text.get("linear_key_head_dim")),
            value_heads=_integer(text.get("linear_num_value_heads")),
            value_dimension=_integer(text.get("linear_value_head_dim")),
            convolution_width=_integer(text.get("linear_conv_kernel_dim")),
        ),
        tokenizer=TokenizerFacts(
            tokenizer_class=_string(tokenizer.get("tokenizer_class")),
            model_maximum_context=_integer(tokenizer.get("model_max_length")),
            tokenizer_json_present="tokenizer.json" in file_names,
            chat_template_present="chat_template.jinja" in file_names,
            eos_token_ids=_token_ids(config.get("eos_token_id")),
            pad_token_id=_integer(text.get("pad_token_id")),
        ),
    )


def _artifact_report(snapshot: ModelSnapshot) -> dict[str, Any]:
    weight_files = [value for value in snapshot.files if value.role is FileRole.WEIGHT]
    computed = sum(value.sha256 is not None for value in weight_files)
    has_safetensors = any(value.path.endswith(".safetensors") for value in weight_files)
    has_gguf = any(value.path.endswith(".gguf") for value in weight_files)
    if has_safetensors and not has_gguf:
        artifact_format = "safetensors"
    elif has_gguf and not has_safetensors:
        artifact_format = "gguf"
    elif has_safetensors or has_gguf:
        artifact_format = "mixed"
    else:
        artifact_format = "unknown"
    return {
        **asdict(snapshot.tensor_summary),
        "format": artifact_format,
        "files": [
            {
                "path": value.path,
                "role": value.role.value,
                "size_bytes": value.size_bytes,
                "sha256": value.sha256,
                "digest_evidence": value.digest_evidence.value,
            }
            for value in snapshot.files
        ],
        "weight_hashes": {
            "computed_or_declared": computed,
            "total": len(weight_files),
            "complete": computed == len(weight_files) and bool(weight_files),
        },
    }


def _compatibility(
    snapshot: ModelSnapshot,
    facts: ModelFacts,
    artifact_format: str,
    task: InspectionTask,
) -> dict[str, Any]:
    model_type = facts.model_type
    architecture = architecture_for(model_type)
    recognized = architecture is not None
    required = [artifact_format]
    missing = []
    if artifact_format != "safetensors":
        missing.append(f"artifact-format:{artifact_format}")
    if not facts.tokenizer.tokenizer_json_present:
        missing.append("tokenizer-json")
    else:
        required.append("tokenizer-json")
    if architecture:
        text_task = architecture["tasks"]["text-generation"]
        required.extend(text_task["required"])
        missing.extend(text_task["missing"])
    else:
        missing.append(f"architecture:{model_type or 'unknown'}")
    if task is InspectionTask.MULTIMODAL_GENERATION and "vision" in facts.modalities:
        if architecture and task.value in architecture["tasks"]:
            task_contract = architecture["tasks"][task.value]
            required.extend(task_contract["required"])
            missing.extend(task_contract["missing"])
        else:
            missing.append("architecture-multimodal-execution")
    elif task is InspectionTask.MULTIMODAL_GENERATION:
        missing.append("model-vision-modality")
    quantization = facts.quantization
    if quantization.mode == "affine":
        required.append(
            f"affine-q{quantization.bits}-g{quantization.group_size}"
        )
    return {
        "outcome": "missing-capability",
        "architecture_status": "recognized" if recognized else "unknown",
        "artifact_status": "structurally-verified"
        if snapshot.tensor_summary.headers_verified
        else "metadata-declared",
        "required_capabilities": required,
        "missing_capabilities": missing,
        "unrequested_modalities": [
            modality
            for modality in facts.modalities
            if modality != "text" and task is InspectionTask.TEXT_GENERATION
        ],
        "remote_metadata_only": snapshot.kind is SourceKind.HUGGING_FACE,
        "claims": {
            "importable": False,
            "validated": False,
            "performance_qualified": False,
            "production_supported": False,
        },
    }


def _capacity(
    snapshot: ModelSnapshot,
    facts: ModelFacts,
    context_tokens: int,
    slots: int,
) -> dict[str, Any]:
    weight_bytes = snapshot.tensor_summary.tensor_bytes
    components: dict[str, int | None] = {
        "weight_tensor_bytes": weight_bytes,
        "attention_kv_bytes": None,
        "linear_recurrent_state_bytes": None,
        "linear_convolution_state_bytes": None,
    }
    unresolved = [
        "execution scratch",
        "compiled kernels and command resources",
        "service queues and prompt cache",
        "operating-system headroom",
    ]
    maximum_context = facts.dimensions.maximum_context
    context_within_declared_limit = (
        maximum_context is None or context_tokens <= maximum_context
    )

    if architecture_for(facts.model_type):
        attention = facts.attention
        linear = facts.linear_attention
        if all(
            value is not None
            for value in (
                attention.full_layers,
                attention.kv_heads,
                attention.head_dimension,
                attention.linear_layers,
                linear.key_heads,
                linear.key_dimension,
                linear.value_heads,
                linear.value_dimension,
                linear.convolution_width,
            )
        ):
            components["attention_kv_bytes"] = (
                attention.full_layers
                * 2
                * attention.kv_heads
                * attention.head_dimension
                * 2
                * context_tokens
                * slots
            )
            components["linear_recurrent_state_bytes"] = (
                attention.linear_layers
                * linear.value_heads
                * linear.key_dimension
                * linear.value_dimension
                * 4
                * 2
                * slots
            )
            projected_width = (
                2 * linear.key_heads * linear.key_dimension
                + linear.value_heads * linear.value_dimension
            )
            components["linear_convolution_state_bytes"] = (
                attention.linear_layers
                * (linear.convolution_width - 1)
                * projected_width
                * 2
                * 2
                * slots
            )

    known = sum(value for value in components.values() if isinstance(value, int))
    return {
        "context_tokens": context_tokens,
        "slots": slots,
        "declared_maximum_context": maximum_context,
        "context_within_declared_limit": context_within_declared_limit,
        "components": components,
        "known_model_bytes_lower_bound": known,
        "complete_machine_fit_calculation": False,
        "unresolved_components": unresolved,
        "fit_status": "not-evaluated",
    }


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _token_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, int) and value >= 0:
        return (value,)
    if isinstance(value, list) and all(
        isinstance(item, int) and item >= 0 for item in value
    ):
        return tuple(value)
    return ()


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    return f"{value / GIB:.2f} GiB"
