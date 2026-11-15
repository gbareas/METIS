"""Thin shim for `metis report` (milestone I7 — logic lives in metis.cli).

    python scripts/report.py regime-v1 --from results/regime_v1.json
    # equivalent to:  metis report regime-v1 --from ...
"""
import sys

from metis.cli import main

if __name__ == "__main__":
    sys.exit(main(["report", *sys.argv[1:]]))
