"""Thin shim for `metis benchmark regime-v1` (milestone I7 — logic in metis.cli).

    python scripts/benchmark_regime_v1.py --data-root /path/to/dns_data
    # equivalent to:  metis benchmark regime-v1 --data-root ...
"""
import sys

from metis.cli import main

if __name__ == "__main__":
    sys.exit(main(["benchmark", "regime-v1", *sys.argv[1:]]))
