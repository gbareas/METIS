"""Thin shim for `metis dataset build` (milestone I7 — logic in metis.cli).

    python scripts/build_feature_dataset.py --feature-set compact --out artifacts/datasets/compact_v1
    # equivalent to:  metis dataset build --feature-set compact --out ...
"""
import sys

from metis.cli import main

if __name__ == "__main__":
    sys.exit(main(["dataset", "build", *sys.argv[1:]]))
