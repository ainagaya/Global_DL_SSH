#!/usr/bin/env python3
from __future__ import annotations

import argparse

from gdlssh.plots import plot_losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train and validation losses from CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epoch-col")
    parser.add_argument("--train-col")
    parser.add_argument("--val-col")
    args = parser.parse_args()
    plot_losses(args.csv, args.output, epoch_col=args.epoch_col, train_col=args.train_col, val_col=args.val_col)


if __name__ == "__main__":
    main()
