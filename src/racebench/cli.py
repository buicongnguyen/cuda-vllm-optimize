"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .claims import load_claims, validate_claims
from .experiments import load_ledger, validate_ledger
from .score import score_report
from .workload import poisson_arrivals, write_jsonl


def _score(args: argparse.Namespace) -> int:
    print(json.dumps(score_report(args.ttft, args.tpot, args.target), indent=2))
    return 0


def _validate_ledger(args: argparse.Namespace) -> int:
    errors = validate_ledger(load_ledger(Path(args.path)))
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"OK: {args.path}")
    return 0


def _validate_claims(args: argparse.Namespace) -> int:
    errors = validate_claims(load_claims(Path(args.path)))
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"OK: {args.path}")
    return 0


def _workload(args: argparse.Namespace) -> int:
    arrivals = poisson_arrivals(args.conversations, args.turns, args.rate, args.seed)
    write_jsonl(arrivals, Path(args.output))
    print(f"wrote {len(arrivals)} arrivals to {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="racebench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="calculate ERS and target requirements")
    score.add_argument("--ttft", type=float, required=True, help="TTFT in milliseconds")
    score.add_argument("--tpot", type=float, required=True, help="TPOT in milliseconds")
    score.add_argument("--target", type=float, default=72.0)
    score.set_defaults(func=_score)

    ledger = subparsers.add_parser("validate-ledger", help="validate experiments CSV")
    ledger.add_argument("path")
    ledger.set_defaults(func=_validate_ledger)

    claims = subparsers.add_parser("validate-claims", help="validate claim audit JSON")
    claims.add_argument("path")
    claims.set_defaults(func=_validate_claims)

    workload = subparsers.add_parser("workload", help="generate a deterministic arrival plan")
    workload.add_argument("--conversations", type=int, default=70)
    workload.add_argument("--turns", type=int, default=6)
    workload.add_argument("--rate", type=float, default=7.0, help="requests per second")
    workload.add_argument("--seed", type=int, default=2025)
    workload.add_argument("--output", default="results/arrivals.jsonl")
    workload.set_defaults(func=_workload)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

