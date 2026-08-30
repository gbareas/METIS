"""Environment-agnostic resolution of the DNS data root and project config.

No METIS module should know the author's home-directory layout (see
`METIS_detailed_next_steps.md` milestone R1). The data root is resolved
in this order, first hit wins:

    1. explicit value (a CLI `--data-root` flag)
    2. `data.root` in a YAML config file (`--config`, else `configs/default.yaml`)
    3. the `METIS_DATA_ROOT` environment variable
    4. otherwise raise, with a message explaining all three options

The group's standard layout under that root is::

    <data_root>/raw/case{NN}/*.h5
    <data_root>/processed/case{NN}/metadata.json
    <data_root>/processed_slices/case{NN}/{slice_id}/
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DATA_ROOT_ENV_VAR = "METIS_DATA_ROOT"


def load_config(path: str | Path | None = None) -> dict:
    """Parse a YAML project config. Falls back to `configs/default.yaml`;
    returns `{}` if neither an explicit path nor the default exists."""
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        if path == DEFAULT_CONFIG_PATH:
            return {}
        raise FileNotFoundError(f"config file not found: {path}")
    return yaml.safe_load(path.read_text()) or {}


def resolve_data_root(
    cli_value: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Resolve the DNS data root (see module docstring for the order).

    Raises `RuntimeError` if none of the three sources supplies one.
    """
    if cli_value:
        return Path(cli_value).expanduser()

    config = load_config(config_path)
    from_config = (config.get("data") or {}).get("root")
    if from_config:
        return Path(from_config).expanduser()

    from_env = os.environ.get(DATA_ROOT_ENV_VAR)
    if from_env:
        return Path(from_env).expanduser()

    raise RuntimeError(
        "No DNS data root configured. Provide one of:\n"
        "  - the --data-root CLI flag\n"
        f"  - data.root in a config file (default: {DEFAULT_CONFIG_PATH})\n"
        f"  - the {DATA_ROOT_ENV_VAR} environment variable"
    )


def add_data_root_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard `--data-root` / `--config` flags to a script's parser."""
    parser.add_argument(
        "--data-root",
        default=None,
        help=f"DNS data root (overrides config and ${DATA_ROOT_ENV_VAR})",
    )
    parser.add_argument(
        "--config",
        default=None,
        help=f"YAML project config (default: {DEFAULT_CONFIG_PATH})",
    )


def data_root_from_args(args: argparse.Namespace) -> Path:
    """Resolve the data root from a parser populated by `add_data_root_args`."""
    return resolve_data_root(cli_value=args.data_root, config_path=args.config)


def git_commit() -> str:
    """Best-effort HEAD commit of the metis repo, for run provenance.
    Returns 'unknown' if git isn't available or this isn't a checkout."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"
