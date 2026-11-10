"""Run one experiment sweep from a config file. Owner: Sobaan (D).

Usage: python experiments/benchmark.py configs/<sweep>.json
Writes raw rows to results/<sweep>.csv. Every planner runs on the same seeded maps
and start/goal pairs. Unreachable pairs are recorded with found = False.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a JSON sweep config in configs/")
    parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
