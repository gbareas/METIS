"""Thin shim for `metis analyze` (milestone I7 — logic lives in metis.cli).

    python scripts/analyze.py physics case01 --data-root /path/to/dns_data
    # equivalent to:  metis analyze physics case01 --data-root ...
"""
import sys

from metis.cli import main

if __name__ == "__main__":
    sys.exit(main(["analyze", *sys.argv[1:]]))
