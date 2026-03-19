#!/usr/bin/env python3
from __future__ import annotations

import argparse

from gdlssh.plots import plot_epoch_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one predicted map from each epoch file.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pattern", default="simvp_ssh_sst_ns1000000_global_pred_viz_epoch*.npy")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--time-idx", type=int, default=0)
    parser.add_argument("--channel-idx", type=int, default=0)
    parser.add_argument("--cmap", default="viridis")
    args = parser.parse_args()
    plot_epoch_maps(args.input_dir, args.output_dir, args.pattern, args.sample_idx, args.time_idx, args.channel_idx, args.cmap)


if __name__ == "__main__":
    main()
