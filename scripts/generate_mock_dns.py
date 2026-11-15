"""CLI wrapper: python scripts/generate_mock_dns.py --out data/mock/case_mock.h5"""
from __future__ import annotations

import argparse

from metis.testing.mock_dns import generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/mock/case_mock.h5")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    h5_path, metadata_path = generate(args.out, seed=args.seed)
    print(f"Wrote mock DNS case to {h5_path}")
    print(f"Wrote companion metadata to {metadata_path}")


if __name__ == "__main__":
    main()
