"""Read-only host inspection for Tatara's Apple-silicon backend."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
GIB = 1024 ** 3


def _run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _sysctl(name: str) -> str | None:
    return _run(["/usr/sbin/sysctl", "-n", name])


def _integer(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _display_profile() -> dict[str, Any]:
    raw = _run(["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"])
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    displays = parsed.get("SPDisplaysDataType", [])
    if not displays:
        return {}
    first = displays[0]
    return {
        "chipset": first.get("sppci_model") or first.get("_name"),
        "metal_support": first.get("spdisplays_metal"),
        "gpu_cores": _integer(str(first.get("sppci_cores", ""))),
    }


def assess(report: dict[str, Any]) -> list[dict[str, Any]]:
    host = report["host"]
    tools = report["tools"]
    memory = host.get("memory_bytes")
    return [
        {
            "name": "macOS",
            "required": True,
            "pass": host.get("system") == "Darwin",
            "detail": host.get("system") or "unknown",
        },
        {
            "name": "Apple silicon",
            "required": True,
            "pass": host.get("machine") == "arm64",
            "detail": host.get("machine") or "unknown",
        },
        {
            "name": "M-series chip",
            "required": True,
            "pass": "Apple M" in (host.get("chip") or ""),
            "detail": host.get("chip") or "unknown",
        },
        {
            "name": "Metal framework SDK",
            "required": True,
            "pass": bool(tools.get("metal_framework")),
            "detail": tools.get("metal_framework") or "not found",
        },
        {
            "name": "Clang compiler",
            "required": True,
            "pass": bool(tools.get("clang")),
            "detail": tools.get("clang") or "not found",
        },
        {
            "name": "Offline Metal compiler",
            "required": False,
            "pass": bool(tools.get("metal_compiler")),
            "detail": tools.get("metal_compiler") or
                      "not installed; runtime compilation remains available",
        },
        {
            "name": "Qwen3.6-35B target memory",
            "required": False,
            "pass": memory is not None and memory >= 32 * GIB,
            "detail": "unknown" if memory is None else f"{memory / GIB:.1f} GiB",
        },
    ]


def collect_report() -> dict[str, Any]:
    xcrun = shutil.which("xcrun")
    metal_compiler = _run([xcrun, "--find", "metal"]) if xcrun else None
    clang = _run([xcrun, "--find", "clang++"]) if xcrun else None
    sdk = _run([xcrun, "--sdk", "macosx", "--show-sdk-path"]) if xcrun else None
    metal_framework = None
    if sdk:
        candidate = Path(sdk) / "System/Library/Frameworks/Metal.framework/Headers/Metal.h"
        if candidate.is_file():
            metal_framework = str(candidate.parents[1])
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "host": {
            "system": platform.system(),
            "os_version": platform.mac_ver()[0] or platform.release(),
            "machine": platform.machine(),
            "chip": _sysctl("machdep.cpu.brand_string"),
            "memory_bytes": _integer(_sysctl("hw.memsize")),
            "logical_cpus": _integer(_sysctl("hw.logicalcpu")),
            "display": _display_profile(),
        },
        "tools": {
            "xcrun": xcrun,
            "macos_sdk": sdk,
            "metal_framework": metal_framework,
            "metal_compiler": metal_compiler,
            "clang": clang,
            "git": shutil.which("git"),
        },
        "scope": {
            "mutates_host": False,
            "runs_gpu_work": False,
            "measures_performance": False,
        },
    }
    report["checks"] = assess(report)
    report["supported_host"] = all(
        check["pass"] for check in report["checks"] if check["required"]
    )
    return report


def render_human(report: dict[str, Any]) -> str:
    status = "SUPPORTED" if report["supported_host"] else "NOT SUPPORTED"
    lines = [f"Tatara doctor: {status}"]
    for check in report["checks"]:
        marker = "PASS" if check["pass"] else ("FAIL" if check["required"] else "WARN")
        lines.append(f"  {marker:4}  {check['name']}: {check['detail']}")
    lines.append("  INFO  read-only inspection; no GPU work was run")
    return "\n".join(lines)
