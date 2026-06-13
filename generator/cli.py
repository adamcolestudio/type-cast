"""``type-cast`` CLI — the operator entry point.

Subcommands:

  * ``type-cast run <yaml> [--output DIR] [--device cuda:N]``
        Real run: loads model, generates the full matrix, encodes,
        writes manifest. Long-running.

  * ``type-cast run <yaml> --dry-run [--output DIR]``
        Walks the matrix with a stub adapter (no model load, no encode).
        Prints every operation that would happen. Use to validate YAML,
        sweep math, folder/file layout before burning a GPU minute.

  * ``type-cast validate <yaml>``
        Parse the YAML and print the resolved spec. Fast feedback when
        editing experiments.

Logging is INFO by default; ``-v`` flips DEBUG (chunk-level progress
from the AR driver). Output goes to stderr so stdout stays clean for
piping the resulting experiment directory path.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    if args.command == "run":
        return _cmd_run(args)
    if args.command == "validate":
        return _cmd_validate(args)
    parser.print_help(sys.stderr)
    return 2


# ── Subcommand handlers ─────────────────────────────────────────────────

def _cmd_run(args) -> int:
    from .experiment import load_experiment, ExperimentLoadError
    from .pipeline_factory import build_adapter
    from .runner import run_experiment

    try:
        spec = load_experiment(args.experiment)
    except ExperimentLoadError as e:
        print(f"[type-cast] error: {e}", file=sys.stderr)
        return 1

    # Resolve output root. CLI flag wins; otherwise the spec lives next
    # to its YAML and we write to ``<yaml-dir>/../output/`` so experiments
    # and outputs share a tidy sibling pair under the repo root.
    if args.output:
        output_root = Path(args.output)
    else:
        yaml_path = Path(args.experiment).resolve()
        output_root = yaml_path.parent.parent / "output"

    try:
        adapter = build_adapter(spec, device=args.device, stub=args.dry_run)
    except ImportError as e:
        print(
            f"[type-cast] error: scope-attention-bender not installed in this "
            f"env ({e}). Install it via\n"
            f"  pip install -e /path/to/scope-attention-bender\n"
            f"or pass --dry-run to walk the matrix with a stub.",
            file=sys.stderr,
        )
        return 1

    try:
        exp_dir = run_experiment(
            spec, adapter=adapter, output_root=output_root, dry_run=args.dry_run,
        )
    except Exception as e:                                     # noqa: BLE001
        logging.exception("[type-cast] FAILED — %s", e)
        return 1

    # Print the result path on STDOUT so shell scripts can pipe it
    # (logs went to stderr).
    print(exp_dir)
    return 0


def _cmd_validate(args) -> int:
    from dataclasses import asdict, is_dataclass
    from .experiment import load_experiment, ExperimentLoadError
    import json

    try:
        spec = load_experiment(args.experiment)
    except ExperimentLoadError as e:
        print(f"[type-cast] error: {e}", file=sys.stderr)
        return 1

    # Pretty-print the resolved spec so the operator can confirm what
    # the runner will use. asdict handles the dataclass tree; the
    # Sweep field is its own dataclass so it serializes cleanly.
    out = asdict(spec) if is_dataclass(spec) else dict(spec)
    if is_dataclass(spec.sweep):
        out["sweep"] = asdict(spec.sweep)
        out["sweep"]["kind"] = type(spec.sweep).__name__
    print(json.dumps(out, indent=2, default=str))

    # Also report the matrix size so the operator can sanity-check.
    bands = list(spec.sweep.iter_bands(layer_count=30))    # 30 = LongLive default
    print(f"\n# bands (assuming 30-layer model): {len(bands)}", file=sys.stderr)
    print(f"# total videos: {spec.total_video_count(layer_count=30)}", file=sys.stderr)
    return 0


# ── Plumbing ────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="type-cast", description="Type Cast generation driver")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG-level logging")

    subs = p.add_subparsers(dest="command", required=True)

    run = subs.add_parser("run", help="Execute an experiment YAML")
    run.add_argument("experiment", help="path to the experiment YAML")
    run.add_argument("--output", help="output root dir (default: <yaml-dir>/../output/)")
    run.add_argument("--device", help="torch device override (e.g. cuda:1)")
    run.add_argument(
        "--dry-run", action="store_true",
        help="walk the matrix with a stub adapter — no model, no encode, no I/O",
    )

    val = subs.add_parser("validate", help="Parse + echo an experiment YAML")
    val.add_argument("experiment", help="path to the experiment YAML")

    return p


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )


if __name__ == "__main__":
    raise SystemExit(main())
