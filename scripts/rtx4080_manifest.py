"""Capture the software and RTX 4080 Super state attached to a benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def source_git_state() -> tuple[str | None, bool | None]:
    """Return (HEAD sha, tree has uncommitted tracked changes).

    A bare HEAD sha misattributes results produced from an uncommitted tree, so
    the dirty flag travels with it. The WSL mirror has no .git; the setup
    script writes ``<sha>`` or ``<sha>-dirty`` to ``.source-git-sha`` instead.
    """

    git = ["git", "-c", f"safe.directory={ROOT}", "-C", str(ROOT)]
    sha = command_output([*git, "rev-parse", "HEAD"])
    if sha:
        status = command_output([*git, "status", "--porcelain", "--untracked-files=no"])
        return sha, None if status is None else bool(status)
    marker = ROOT / ".source-git-sha"
    if marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        if not value:
            return None, None
        if value.endswith("-dirty"):
            return value.removesuffix("-dirty"), True
        return value, False
    return None, None


def torch_manifest() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"installed": False}

    result: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "built_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        return result

    props = torch.cuda.get_device_properties(0)
    result["device"] = {
        "name": props.name,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "total_memory_gib": round(props.total_memory / 2**30, 3),
    }
    return result


def build_manifest() -> dict[str, Any]:
    git_sha, git_dirty = source_git_state()
    smi_query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap,memory.total,memory.free,"
            "pstate,power.limit,clocks.sm,clocks.mem",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: package_version(name)
            for name in ("vllm", "torch", "triton", "transformers", "httpx")
        },
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "wsl_kernel": command_output(["uname", "-a"]),
        "nvidia_smi_csv": smi_query,
        "torch": torch_manifest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/rtx4080/manifest.json"))
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
