#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from tatara.artifact_manifest import ArtifactManifestError, parse_manifest
from tatara.qwen36_plan_generation import (
    ModelPlanGenerationError,
    parse_model_plan,
    render_model_plan_header,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a static Tatara model plan")
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()

    try:
        package_bytes = arguments.package.read_bytes()
        artifact_bytes = arguments.artifact.read_bytes()
        artifact = parse_manifest(artifact_bytes.decode("utf-8"))
        plan = parse_model_plan(
            package_bytes.decode("utf-8"),
            artifact,
            arguments.artifact.name,
            hashlib.sha256(package_bytes).hexdigest(),
            hashlib.sha256(artifact_bytes).hexdigest(),
        )
        output = render_model_plan_header(plan)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(output)
    except (ArtifactManifestError, ModelPlanGenerationError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
