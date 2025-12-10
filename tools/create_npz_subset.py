"""
Utility to carve out a tiny NPZ subset from a PulseDB .mat file.

This lets us avoid loading 10+ GB files on laptops: run this once on a
bigger machine, then point `--train_mat/--val_mat` to the resulting `.npz`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bpnet.data import load_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small NPZ subset from a PulseDB .mat file.")
    parser.add_argument("--input", required=True, help="Path to PulseDB .mat subset (e.g., VitalDB_Train_Subset.mat)")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--max_segments", type=int, default=1000, help="Maximum segments to keep")
    parser.add_argument("--fraction", type=float, default=1.0, help="Alternatively keep this fraction (0-1]")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset = load_subset(args.input)
    total = subset["signals"].shape[0]
    keep = total
    if args.fraction < 1.0:
        keep = int(total * max(0.0, min(1.0, args.fraction)))
    keep = max(1, min(keep, args.max_segments))

    rng = np.random.default_rng(args.seed)
    indices = np.sort(rng.choice(total, size=keep, replace=False))

    signals = subset["signals"][indices]
    sbp = subset["sbp"][indices]
    dbp = subset["dbp"][indices]
    subjects = np.asarray(subset["subjects"])[indices]

    np.savez_compressed(output_path, signals=signals, sbp=sbp, dbp=dbp, subjects=subjects)
    print(f"Wrote {keep} segments to {output_path} (~{output_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
