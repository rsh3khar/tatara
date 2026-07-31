"""Tatara command-line interface."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tatara import __version__
from tatara.artifact_manifest import (
    ArtifactManifestError,
    parse_manifest,
    verify_manifest,
)
from tatara.checkpoint_preparation import (
    CheckpointPreparationError,
    prepare_checkpoint,
    write_new_checkpoint,
)
from tatara.doctor import collect_report, render_human
from tatara.model_inspection import inspect_snapshot
from tatara.model_inspection import render_human as render_model
from tatara.model_source import ModelSourceError, load_huggingface, load_local
from tatara.model_types import HashMode, InspectionTask
from tatara.prepared_checkpoint import (
    PreparedCheckpointError,
    decode_prepared_checkpoint,
)
from tatara.validation import render_human as render_validation
from tatara.validation import validate_prepared_checkpoint
from tatara.reference_contract import (
    ReferenceContractError,
    parse_reference_contract,
    verify_reference_observations,
    verify_source_observations,
)
from tatara.reference_files import observe_reference_files, observe_source_files

# Typed exits, so a supervisor can tell a bad record from an unreadable path.
EXIT_OK = 0
EXIT_VALIDATION_FAILED = 1
EXIT_INPUT_UNREADABLE = 2
EXIT_RECORD_INVALID = 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tatara",
        description="Performance-specialized Apple-silicon inference engine",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="inspect host compatibility without changing the machine"
    )
    doctor.add_argument("--json", action="store_true", dest="as_json")
    reference = commands.add_parser(
        "reference", help="inspect or verify a sealed behavioral reference"
    )
    reference_commands = reference.add_subparsers(
        dest="reference_command", required=True
    )
    reference_check = reference_commands.add_parser(
        "check", help="verify reference artifacts without executing them"
    )
    reference_check.add_argument("contract", type=Path)
    reference_check.add_argument(
        "--root", type=Path, required=True, help="external artifact root"
    )
    reference_check.add_argument(
        "--scope",
        choices=("identity", "evidence", "source", "all"),
        default="identity",
        help="select identity artifacts, evidence, the source snapshot, or all",
    )
    reference_check.add_argument("--json", action="store_true", dest="as_json")
    inspect = commands.add_parser(
        "inspect", help="inspect a local model or Hugging Face repository without loading it"
    )
    inspect.add_argument("model", help="local directory or hf://owner/model[@revision]")
    inspect.add_argument("--json", action="store_true", dest="as_json")
    inspect.add_argument("--context", type=int, default=4096, dest="context_tokens")
    inspect.add_argument("--slots", type=int, default=1)
    inspect.add_argument(
        "--task",
        choices=("text-generation", "multimodal-generation"),
        default="text-generation",
        help="capability surface to assess",
    )
    inspect.add_argument(
        "--hash",
        choices=("none", "metadata", "all"),
        default="metadata",
        dest="hash_mode",
        help="local hashing scope; 'all' reads every weight byte",
    )
    inspect.add_argument(
        "--allow-network",
        action="store_true",
        help="allow metadata-only Hugging Face requests; weight shards are never fetched",
    )
    inspect.add_argument(
        "--verify-manifest",
        type=Path,
        help="verify every listed local file by size and SHA-256",
    )
    prepare = commands.add_parser(
        "prepare", help="prepare checkpoint metadata without reading weight payloads"
    )
    prepare.add_argument("model", type=Path, help="external local artifact root")
    prepare.add_argument("--package", required=True, type=Path)
    prepare.add_argument("--artifact", type=Path, help="override the adjacent manifest path")
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--json", action="store_true", dest="as_json")
    validate = commands.add_parser(
        "validate",
        help="check a prepared checkpoint against its package, manifest and shards",
    )
    validate.add_argument("record", type=Path, help="prepared .tatara record")
    validate.add_argument("--package", required=True, type=Path)
    validate.add_argument("--artifact", type=Path, help="override the adjacent manifest path")
    validate.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _run_validate(args) -> int:
    """Reads files, delegates the judgement to tatara.validation, prints."""
    try:
        package_bytes = args.package.read_bytes()
        manifest_path = args.artifact or args.package.parent / "artifact.toml"
        manifest_bytes = manifest_path.read_bytes()
        record_bytes = args.record.read_bytes()
    except OSError as error:
        print(f"validate: cannot read input: {error}", file=sys.stderr)
        return EXIT_INPUT_UNREADABLE
    try:
        manifest = parse_manifest(manifest_bytes.decode("utf-8"))
        checkpoint = decode_prepared_checkpoint(record_bytes)
    except (ArtifactManifestError, PreparedCheckpointError, UnicodeDecodeError) as error:
        print(f"validate: {error}", file=sys.stderr)
        return EXIT_RECORD_INVALID
    report = validate_prepared_checkpoint(
        checkpoint,
        manifest,
        hashlib.sha256(package_bytes).hexdigest(),
        hashlib.sha256(manifest_bytes).hexdigest(),
    )
    if args.as_json:
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "findings": [
                        {
                            "severity": f.severity.value,
                            "check": f.check,
                            "detail": f.detail,
                        }
                        for f in report.findings
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(render_validation(report))
    return EXIT_OK if report.ok else EXIT_VALIDATION_FAILED


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        raise SystemExit(_run_validate(args))
    if args.command == "doctor":
        report = collect_report()
        if args.as_json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_human(report))
        raise SystemExit(0 if report["supported_host"] else 1)
    if args.command == "reference" and args.reference_command == "check":
        roles = {
            "identity": frozenset(
                {"executable", "configuration", "dependency", "fixture"}
            ),
            "evidence": frozenset({"evidence"}),
            "source": frozenset(),
            "all": frozenset(
                {
                    "executable",
                    "configuration",
                    "dependency",
                    "fixture",
                    "evidence",
                }
            ),
        }[args.scope]
        try:
            contract = parse_reference_contract(args.contract.read_text())
            checked_files = 0
            mismatches = []
            if roles:
                observations = observe_reference_files(contract, args.root, roles)
                verification = verify_reference_observations(
                    contract, observations, roles
                )
                checked_files += verification.checked_files
                mismatches.extend(verification.mismatches)
            if args.scope in {"source", "all"}:
                source_observations = observe_source_files(contract, args.root)
                source_verification = verify_source_observations(
                    contract, source_observations
                )
                checked_files += source_verification.checked_files
                mismatches.extend(source_verification.mismatches)
        except (OSError, ReferenceContractError) as error:
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "command": "reference.check",
                            "error": {
                                "code": "reference-check-failed",
                                "message": str(error),
                            },
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"Tatara reference check: ERROR\n  {error}", file=sys.stderr)
            raise SystemExit(2)
        report = {
            "schema_version": 1,
            "command": "reference.check",
            "reference_id": contract.reference_id,
            "scope": args.scope,
            "matched": not mismatches,
            "checked_files": checked_files,
            "mismatches": list(mismatches),
        }
        if args.as_json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            status = "PASS" if not mismatches else "FAIL"
            print(f"Tatara reference check: {status}")
            print(f"  reference: {contract.reference_id}")
            print(f"  scope: {args.scope}")
            print(f"  files: {checked_files}")
            for mismatch in mismatches:
                print(f"  mismatch: {mismatch}")
        raise SystemExit(0 if not mismatches else 3)
    if args.command == "prepare":
        try:
            checkpoint, encoded = prepare_checkpoint(
                args.package,
                args.model,
                args.artifact,
            )
            write_new_checkpoint(args.output, encoded)
        except (CheckpointPreparationError, OSError, ValueError) as error:
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "command": "prepare",
                            "error": {
                                "code": "checkpoint-preparation-failed",
                                "message": str(error),
                            },
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"Tatara checkpoint preparation: ERROR\n  {error}", file=sys.stderr)
            raise SystemExit(2)
        report = {
            "schema_version": 1,
            "command": "prepare",
            "package_id": checkpoint.identity.package_id,
            "artifact_id": checkpoint.identity.artifact_id,
            "output": str(args.output),
            "record_bytes": len(encoded),
            "record_sha256": hashlib.sha256(encoded).hexdigest(),
            "shard_count": len(checkpoint.shards),
            "tensor_count": len(checkpoint.tensors),
            "tensor_bytes": checkpoint.tensor_payload_bytes,
            "weight_headers_read": True,
            "weight_payload_read": False,
            "weight_payload_hashed": False,
        }
        if args.as_json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print("Tatara checkpoint preparation: PASS")
            print(f"  package: {checkpoint.identity.package_id}")
            print(f"  artifact: {checkpoint.identity.artifact_id}")
            print(f"  shards: {len(checkpoint.shards)}")
            print(f"  tensors: {len(checkpoint.tensors)}")
            print(f"  tensor bytes: {checkpoint.tensor_payload_bytes}")
            print(f"  record: {args.output}")
            print("  weight payloads: not read")
        raise SystemExit(0)
    if args.command == "inspect":
        try:
            manifest = None
            if args.verify_manifest:
                manifest = parse_manifest(args.verify_manifest.read_text())
            if args.model.startswith("hf://"):
                if manifest:
                    raise ModelSourceError(
                        "exact manifest verification requires local artifacts"
                    )
                if args.hash_mode == "all":
                    raise ModelSourceError(
                        "--hash all is local-only because remote inspection never fetches weights"
                    )
                snapshot = load_huggingface(args.model, args.allow_network)
            else:
                if args.allow_network:
                    raise ModelSourceError(
                        "--allow-network is valid only for an hf:// model reference"
                    )
                hash_mode = HashMode.ALL if manifest else HashMode(args.hash_mode)
                snapshot = load_local(Path(args.model), hash_mode)
            report = inspect_snapshot(
                snapshot,
                args.context_tokens,
                args.slots,
                task=InspectionTask(args.task),
            )
            if manifest:
                verification = verify_manifest(manifest, snapshot)
                report["artifact_manifest"] = {
                    "path": str(args.verify_manifest),
                    "artifact_id": verification.artifact_id,
                    "matched": verification.matched,
                    "checked_files": verification.checked_files,
                    "mismatches": list(verification.mismatches),
                }
        except (ArtifactManifestError, ModelSourceError, OSError, ValueError) as error:
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "command": "inspect",
                            "error": {
                                "code": "inspection-failed",
                                "message": str(error),
                            },
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
            else:
                print(f"Tatara model inspection: ERROR\n  {error}", file=sys.stderr)
            raise SystemExit(2)
        if args.as_json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(render_model(report))
        manifest_report = report.get("artifact_manifest")
        raise SystemExit(0 if not manifest_report or manifest_report["matched"] else 3)
    raise AssertionError(f"unhandled command: {args.command}")
