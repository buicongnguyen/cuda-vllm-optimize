"""Run a reproducible vLLM experiment on an RTX 4080 Super under WSL2.

This orchestrator starts one server at a time, waits for health, executes the
existing deterministic streaming replay, preserves artifacts, and optionally
runs the complete R0/B/R0-prime comparison. It reproduces the method locally;
it does not claim H200 MIG score equivalence.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
from time import monotonic, sleep
from typing import Any, Iterator
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "configs/vllm/rtx4080-r0.args"
DEFAULT_CANDIDATE = ROOT / "configs/vllm/rtx4080-prefix-cache.args"
DEFAULT_MODEL = "LiquidAI/LFM2.5-1.2B-Instruct"


@dataclass(frozen=True)
class ServerConfig:
    path: Path
    model: str
    arguments: tuple[str, ...]


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True


def command_output(command: list[str], timeout: float = 20) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return 127, f"{type(error).__name__}: {error}"
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return completed.returncode, output


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def parse_args_file(path: Path) -> ServerConfig:
    if not path.is_file():
        raise ValueError(f"config file does not exist: {path}")
    lexer = shlex.shlex(path.read_text(encoding="utf-8"), posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    tokens = list(lexer)
    if not tokens:
        raise ValueError(f"config file is empty: {path}")
    if tokens[0].startswith("-"):
        raise ValueError(f"first token in {path} must be the model id")
    return ServerConfig(path.resolve(), tokens[0], tuple(tokens[1:]))


def option_map(arguments: tuple[str, ...]) -> dict[str, str | bool]:
    """Normalize CLI options so an A/B diff is human-auditable."""

    normalized: dict[str, str | bool] = {}
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected positional server argument: {token}")
        if "=" in token:
            key, value = token.split("=", 1)
            normalized[key] = value
        elif index + 1 < len(arguments) and not arguments[index + 1].startswith("--"):
            normalized[token] = arguments[index + 1]
            index += 1
        else:
            normalized[token] = True
        index += 1
    return normalized


def config_diff(baseline: ServerConfig, candidate: ServerConfig) -> dict[str, Any]:
    before = option_map(baseline.arguments)
    after = option_map(candidate.arguments)
    keys = sorted(before.keys() | after.keys())
    changes = {
        key: {"baseline": before.get(key), "candidate": after.get(key)}
        for key in keys
        if before.get(key) != after.get(key)
    }
    if baseline.model != candidate.model:
        changes["model"] = {"baseline": baseline.model, "candidate": candidate.model}
    return changes


def server_command(config: ServerConfig, port: int) -> list[str]:
    executable = shutil.which("vllm") or "vllm"
    return [executable, "serve", config.model, *config.arguments, "--port", str(port)]


def server_environment() -> dict[str, str]:
    environment = os.environ.copy()
    # vLLM 0.25.1 enables the FlashInfer sampler by default. Its wheel does not
    # contain every Ada/CUDA combination, so an RTX 4080 Super can fall back to
    # a JIT build that requires a full nvcc toolkit. The torch sampler is the
    # supported no-nvcc path and keeps the setup reproducible on stock WSL2.
    environment.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if not environment.get("CC") and not (shutil.which("gcc") or shutil.which("clang")):
        portable = Path(sys.prefix) / "bin/triton-cc"
        if portable.is_file():
            environment["CC"] = str(portable)
    return environment


def doctor_report(config_path: Path = DEFAULT_BASELINE) -> dict[str, Any]:
    checks: list[Check] = []
    system = platform.system()
    release = platform.release()
    checks.append(Check("linux", system == "Linux", f"{system} {release}"))
    is_wsl = system == "Linux" and "microsoft" in release.lower()
    checks.append(Check("wsl2", is_wsl, platform.platform()))
    checks.append(
        Check(
            "python",
            sys.version_info >= (3, 11),
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )
    )

    smi_code, smi_output = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_visible = smi_code == 0 and bool(smi_output)
    checks.append(Check("nvidia-smi", gpu_visible, smi_output or "not available"))
    checks.append(
        Check(
            "rtx4080-super",
            gpu_visible and "RTX 4080 SUPER" in smi_output.upper(),
            smi_output.splitlines()[0] if smi_output else "GPU not detected",
        )
    )

    for package in ("vllm", "torch", "httpx", "lfm-racebench"):
        version = package_version(package)
        checks.append(Check(f"package:{package}", version is not None, version or "not installed"))

    torch_ok = False
    torch_detail = "torch not importable"
    try:
        import torch

        torch_ok = torch.cuda.is_available()
        torch_detail = (
            f"cuda={torch_ok}; build={torch.version.cuda}; "
            f"device={torch.cuda.get_device_name(0) if torch_ok else 'none'}"
        )
    except Exception as error:  # environment evidence, not application correctness
        torch_detail = f"{type(error).__name__}: {error}"
    checks.append(Check("torch-cuda", torch_ok, torch_detail))

    compiler = os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        portable = Path(sys.prefix) / "bin/triton-cc"
        compiler = str(portable) if portable.is_file() else None
    checks.append(Check("triton-c-compiler", compiler is not None, compiler or "not installed"))

    try:
        config = parse_args_file(config_path)
        config_detail = f"{config.model}; {len(config.arguments)} arguments"
        config_ok = True
    except ValueError as error:
        config_detail = str(error)
        config_ok = False
    checks.append(Check("baseline-config", config_ok, config_detail))

    disk = shutil.disk_usage(Path.home())
    free_gib = disk.free / 2**30
    checks.append(Check("disk-free", free_gib >= 25, f"{free_gib:.1f} GiB", required=False))

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "method_scope": "RTX 4080 Super method reproduction; not H200 MIG score equivalence",
        "ok": all(check.ok for check in checks if check.required),
        "checks": [asdict(check) for check in checks],
    }


def print_doctor(report: dict[str, Any]) -> None:
    print("RTX 4080 Super lab doctor")
    print("Method reproduction only; scores are not portable to H200 MIG.\n")
    for check in report["checks"]:
        marker = "PASS" if check["ok"] else "WARN" if not check["required"] else "FAIL"
        print(f"[{marker:4}] {check['name']:<20} {check['detail']}")
    print(f"\nOverall: {'READY' if report['ok'] else 'NOT READY'}")


def wait_until_ready(process: subprocess.Popen[str], base_url: str, timeout: float, log_path: Path) -> None:
    deadline = monotonic() + timeout
    health_urls = [f"{base_url}/health", f"{base_url}/v1/models"]
    last_error = "server has not answered"
    while monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            raise RuntimeError(f"vLLM exited with code {exit_code}\n{tail}")
        for url in health_urls:
            try:
                with urlopen(url, timeout=3) as response:
                    if 200 <= response.status < 300:
                        return
            except (URLError, TimeoutError, OSError) as error:
                last_error = f"{type(error).__name__}: {error}"
        sleep(1)
    raise TimeoutError(f"vLLM did not become ready within {timeout}s: {last_error}; log={log_path}")


@contextmanager
def managed_server(command: list[str], base_url: str, timeout: float, log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=server_environment(),
        )
        try:
            wait_until_ready(process, base_url, timeout, log_path)
            yield
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)


def run_checked(command: list[str]) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def run_stage(
    *,
    label: str,
    config: ServerConfig,
    run_dir: Path,
    port: int,
    startup_timeout: float,
    conversations: int,
    turns: int,
    rate: float,
    max_tokens: int,
    seed: int,
) -> Path:
    base_url = f"http://127.0.0.1:{port}"
    output = run_dir / f"{label}.jsonl"
    shutil.copy2(config.path, run_dir / f"{label}.args")
    run_checked(
        [
            sys.executable,
            str(ROOT / "scripts/rtx4080_manifest.py"),
            "--output",
            str(run_dir / f"{label}-manifest.json"),
        ]
    )
    command = server_command(config, port)
    print(f"\n[{label}] starting server\n+ {shlex.join(command)}", flush=True)
    with managed_server(command, base_url, startup_timeout, run_dir / f"{label}-server.log"):
        run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/rtx4080_replay.py"),
                "--base-url",
                base_url,
                "--model",
                config.model,
                "--conversations",
                str(conversations),
                "--turns",
                str(turns),
                "--rate",
                str(rate),
                "--seed",
                str(seed),
                "--max-tokens",
                str(max_tokens),
                "--output",
                str(output),
            ]
        )
    return output


def experiment_directory(requested: Path | None) -> Path:
    if requested is not None:
        return requested.resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (ROOT / f"results/rtx4080/{stamp}").resolve()


def execute_run(args: argparse.Namespace) -> int:
    baseline = parse_args_file(args.baseline_config)
    candidate = parse_args_file(args.candidate_config)
    run_dir = experiment_directory(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "smoke":
        shape = {"conversations": 2, "turns": 2, "rate": 1.0, "max_tokens": 16}
        stages = [("R0-smoke", baseline)]
    elif args.mode == "baseline":
        shape = {
            "conversations": args.conversations,
            "turns": args.turns,
            "rate": args.rate,
            "max_tokens": args.max_tokens,
        }
        stages = [("R0-baseline", baseline)]
    else:
        shape = {
            "conversations": args.conversations,
            "turns": args.turns,
            "rate": args.rate,
            "max_tokens": args.max_tokens,
        }
        stages = [("R0-baseline", baseline), ("B-candidate", candidate), ("R0-prime", baseline)]

    plan = {
        "mode": args.mode,
        "scope": "method reproduction on RTX 4080 Super; not H200 MIG score equivalence",
        "output_dir": str(run_dir),
        "shape": shape,
        "seed": args.seed,
        "baseline_config": str(baseline.path),
        "candidate_config": str(candidate.path),
        "candidate_diff": config_diff(baseline, candidate),
        "stages": [
            {"label": label, "command": server_command(config, args.port)}
            for label, config in stages
        ],
    }
    (run_dir / "experiment-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        print("Dry run only; no server was started.")
        return 0

    report = doctor_report(args.baseline_config)
    print_doctor(report)
    (run_dir / "doctor.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["ok"]:
        raise RuntimeError("doctor failed; run scripts/rtx4080_setup_wsl.sh and retry")

    outputs: dict[str, Path] = {}
    for label, config in stages:
        outputs[label] = run_stage(
            label=label,
            config=config,
            run_dir=run_dir,
            port=args.port,
            startup_timeout=args.startup_timeout,
            conversations=shape["conversations"],
            turns=shape["turns"],
            rate=shape["rate"],
            max_tokens=shape["max_tokens"],
            seed=args.seed,
        )

    if args.mode == "aba":
        run_checked(
            [
                sys.executable,
                str(ROOT / "scripts/rtx4080_compare.py"),
                str(outputs["R0-baseline"]),
                str(outputs["B-candidate"]),
                "--baseline-return",
                str(outputs["R0-prime"]),
                "--seed",
                str(args.seed),
                "--output",
                str(run_dir / "comparison.json"),
            ]
        )
    print(f"\nArtifacts: {run_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="validate WSL2, GPU and Python environment")
    doctor.add_argument("--config", type=Path, default=DEFAULT_BASELINE)
    doctor.add_argument("--json", action="store_true", dest="as_json")

    command = subparsers.add_parser("command", help="print the exact vLLM serve command")
    command.add_argument("--config", type=Path, default=DEFAULT_BASELINE)
    command.add_argument("--port", type=int, default=8000)

    serve = subparsers.add_parser("serve", help="run vLLM in the foreground")
    serve.add_argument("--config", type=Path, default=DEFAULT_BASELINE)
    serve.add_argument("--port", type=int, default=8000)

    run = subparsers.add_parser("run", help="run smoke, baseline or full A/B/A")
    run.add_argument("--mode", choices=("smoke", "baseline", "aba"), default="smoke")
    run.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE)
    run.add_argument("--candidate-config", type=Path, default=DEFAULT_CANDIDATE)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--port", type=int, default=8000)
    run.add_argument("--startup-timeout", type=float, default=900)
    run.add_argument("--conversations", type=int, default=70)
    run.add_argument("--turns", type=int, default=6)
    run.add_argument("--rate", type=float, default=7.0)
    run.add_argument("--max-tokens", type=int, default=64)
    run.add_argument("--seed", type=int, default=2025)
    run.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "doctor":
        report = doctor_report(args.config)
        if args.as_json:
            print(json.dumps(report, indent=2))
        else:
            print_doctor(report)
        return 0 if report["ok"] else 2
    if args.command == "command":
        print(shlex.join(server_command(parse_args_file(args.config), args.port)))
        return 0
    if args.command == "serve":
        if platform.system() != "Linux":
            raise SystemExit("vLLM must run inside Ubuntu WSL2, not native Windows")
        return subprocess.run(
            server_command(parse_args_file(args.config), args.port),
            check=False,
            env=server_environment(),
        ).returncode
    if min(args.port, args.startup_timeout, args.conversations, args.turns, args.rate, args.max_tokens) <= 0:
        raise SystemExit("port, timeouts and workload parameters must be positive")
    return execute_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
