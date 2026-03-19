#!/usr/bin/env python3
import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def extract_epoch(path):
    name = os.path.basename(path)
    m = re.search(r"epoch(\d+)", name)
    if m:
        return int(m.group(1))
    return float("inf")


def prepare_array(arr, sample_idx=0, time_idx=0, channel_idx=0):
    arr = np.asarray(arr)

    if arr.ndim == 5:
        # expected: (sample, time, channel, H, W)
        if not (0 <= sample_idx < arr.shape[0]):
            raise ValueError(f"sample_idx={sample_idx} out of range for shape {arr.shape}")
        if not (0 <= time_idx < arr.shape[1]):
            raise ValueError(f"time_idx={time_idx} out of range for shape {arr.shape}")
        if not (0 <= channel_idx < arr.shape[2]):
            raise ValueError(f"channel_idx={channel_idx} out of range for shape {arr.shape}")
        return arr[sample_idx, time_idx, channel_idx]

    if arr.ndim == 4:
        # maybe already squeezed: (sample, time, H, W)
        if not (0 <= sample_idx < arr.shape[0]):
            raise ValueError(f"sample_idx={sample_idx} out of range for shape {arr.shape}")
        if not (0 <= time_idx < arr.shape[1]):
            raise ValueError(f"time_idx={time_idx} out of range for shape {arr.shape}")
        return arr[sample_idx, time_idx]

    if arr.ndim == 3:
        # fallback: (N, H, W)
        if not (0 <= sample_idx < arr.shape[0]):
            raise ValueError(f"sample_idx={sample_idx} out of range for shape {arr.shape}")
        return arr[sample_idx]

    if arr.ndim == 2:
        return arr

    raise ValueError(f"Unsupported array shape: {arr.shape}")


def main():
    parser = argparse.ArgumentParser(description="Plot one predicted map from each epoch .npy file")
    parser.add_argument("--input-dir", required=True, help="Directory containing pred_viz_epoch*.npy")
    parser.add_argument("--output-dir", required=True, help="Directory to save plots")
    parser.add_argument("--pattern", default="simvp_ssh_sst_ns1000000_global_pred_viz_epoch*.npy", help="File pattern")
    parser.add_argument("--sample-idx", type=int, default=0, help="Which sample to plot")
    parser.add_argument("--time-idx", type=int, default=0, help="Which predicted timestep to plot")
    parser.add_argument("--channel-idx", type=int, default=0, help="Which channel to plot")
    parser.add_argument("--cmap", default="viridis", help="Colormap")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(
        glob.glob(os.path.join(args.input_dir, args.pattern)),
        key=extract_epoch
    )

    if not files:
        raise FileNotFoundError(f"No files found matching {args.pattern} in {args.input_dir}")

    for path in files:
        arr = np.load(path)
        map_2d = prepare_array(
            arr,
            sample_idx=args.sample_idx,
            time_idx=args.time_idx,
            channel_idx=args.channel_idx,
        )

        epoch = extract_epoch(path)
        out_path = os.path.join(args.output_dir, f"epoch_{epoch:03d}.png")

        plt.figure(figsize=(6, 5))
        im = plt.imshow(map_2d, origin="lower", aspect="auto", cmap=args.cmap)
        plt.colorbar(im)
        plt.title(
            f"Epoch {epoch} | sample={args.sample_idx}, time={args.time_idx}, shape={arr.shape}"
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()