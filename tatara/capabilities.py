"""Declarative model capability catalog used by inspection and preparation."""

from __future__ import annotations

from typing import Any


ARCHITECTURES: tuple[dict[str, Any], ...] = (
    {
        "id": "qwen3.5-hybrid-moe",
        "model_types": ("qwen3_5_moe", "qwen3_5_moe_text"),
        "tasks": {
            "text-generation": {
                "required": (
                    "qwen3.5-hybrid-moe-text",
                    "gated-delta-attention",
                    "full-attention",
                    "mixture-of-experts",
                ),
                "missing": ("native-qwen3.5-hybrid-moe-execution",),
            },
            "multimodal-generation": {
                "required": ("vision-encoder",),
                "missing": ("qwen3.5-vision-execution",),
            },
        },
    },
)


def architecture_for(model_type: str | None) -> dict[str, Any] | None:
    for architecture in ARCHITECTURES:
        if model_type in architecture["model_types"]:
            return architecture
    return None
