#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# This script is intended for fast visual debugging of prediction outputs saved by
# `simvp_predict_ssh.py`. Each `.npz` file stores:
#
# - `prediction`: model output
# - `target`: reference field
# - `bbox`: [lon_min, lon_max, lat_min, lat_max]
# - `time_range`: [start_date, end_date]
# - `target_date`: date associated with the selected target index
#
# The main challenge is that prediction arrays may have different ranks:
#
# - 2D: [H, W]
# - 3D: [C, H, W] or [T, H, W]
# - 4D: [T, C, H, W]
#
# To keep the script useful for debugging, we expose `--time-index` and
# `--channel-index`, then reduce everything to one 2D map for plotting.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prediction .npz files produced by simvp_predict_ssh.py.")
    parser.add_argument(
        "input_path",
        help="A single .npz prediction file or a directory containing .npz files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated PNGs. Defaults to <input>/plots for directories or the file parent for single files.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Time index to plot when arrays contain a time dimension.",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=0,
        help="Channel index to plot when arrays contain a channel dimension.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output DPI for saved figures.",
    )
    return parser.parse_args()


def resolve_npz_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".npz":
            raise ValueError(f"Expected a .npz file, got: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist or is not a directory: {input_path}")

    files = sorted(input_path.glob("*.npz"))
    if not files:
        raise ValueError(f"No .npz files found in directory: {input_path}")
    return files


def choose_output_dir(input_path: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg is not None:
        output_dir = Path(output_dir_arg)
    elif input_path.is_dir():
        output_dir = input_path / "plots"
    else:
        output_dir = input_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def clamp_index(index: int, size: int) -> int:
    # We clamp instead of erroring because this script is for rapid debugging and
    # should be forgiving when users do not know the exact output rank upfront.
    if size <= 0:
        raise ValueError("Cannot select an index from an empty dimension.")
    return min(max(index, 0), size - 1)


def select_2d_slice(array: np.ndarray, time_index: int, channel_index: int) -> np.ndarray:
    # Reduce arrays of shape [H, W], [N, H, W], or [T, C, H, W] down to one 2D
    # image for plotting.
    if array.ndim == 2:
        return array

    if array.ndim == 3:
        chosen_index = clamp_index(channel_index, array.shape[0])
        return array[chosen_index]

    if array.ndim == 4:
        chosen_time = clamp_index(time_index, array.shape[0])
        chosen_channel = clamp_index(channel_index, array.shape[1])
        return array[chosen_time, chosen_channel]

    raise ValueError(f"Unsupported array rank for plotting: shape={array.shape}")


def compute_common_limits(prediction_2d: np.ndarray, target_2d: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([prediction_2d.ravel(), target_2d.ravel()])
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return -1.0, 1.0
    return float(finite.min()), float(finite.max())


def compute_symmetric_limit(diff_2d: np.ndarray) -> float:
    finite = diff_2d[np.isfinite(diff_2d)]
    if finite.size == 0:
        return 1.0
    max_abs = float(np.max(np.abs(finite)))
    return max(max_abs, 1e-6)


def format_title(npz_path: Path, bbox: np.ndarray | None, time_range: np.ndarray | None, target_date: str | None) -> str:
    parts = [npz_path.name]
    if target_date:
        parts.append(f"target_date={target_date}")
    if time_range is not None and len(time_range) == 2:
        parts.append(f"time_range={time_range[0]}..{time_range[1]}")
    if bbox is not None and len(bbox) == 4:
        lon_min, lon_max, lat_min, lat_max = bbox
        parts.append(f"bbox=({lon_min:.2f},{lon_max:.2f},{lat_min:.2f},{lat_max:.2f})")
    return "\n".join(parts)


def plot_single_file(npz_path: Path, output_dir: Path, time_index: int, channel_index: int, dpi: int) -> Path:
    with np.load(npz_path, allow_pickle=True) as payload:
        prediction = np.asarray(payload["prediction"])
        target = np.asarray(payload["target"])
        bbox = np.asarray(payload["bbox"]) if "bbox" in payload else None
        time_range = np.asarray(payload["time_range"]) if "time_range" in payload else None
        target_date = str(payload["target_date"]) if "target_date" in payload else None

    prediction_2d = select_2d_slice(prediction, time_index=time_index, channel_index=channel_index)
    target_2d = select_2d_slice(target, time_index=time_index, channel_index=channel_index)
    diff_2d = prediction_2d - target_2d

    value_min, value_max = compute_common_limits(prediction_2d, target_2d)
    diff_limit = compute_symmetric_limit(diff_2d)

    # If bbox is available, use it as the image extent so the axes are displayed
    # in geographic coordinates rather than raw pixel coordinates.
    extent = None
    if bbox is not None and len(bbox) == 4:
        lon_min, lon_max, lat_min, lat_max = [float(value) for value in bbox]
        extent = [lon_min, lon_max, lat_min, lat_max]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    image_pred = axes[0].imshow(
        prediction_2d,
        origin="lower",
        cmap="viridis",
        vmin=value_min,
        vmax=value_max,
        extent=extent,
        aspect="auto",
    )
    axes[0].set_title("Prediction")
    fig.colorbar(image_pred, ax=axes[0], shrink=0.85)

    image_target = axes[1].imshow(
        target_2d,
        origin="lower",
        cmap="viridis",
        vmin=value_min,
        vmax=value_max,
        extent=extent,
        aspect="auto",
    )
    axes[1].set_title("Target")
    fig.colorbar(image_target, ax=axes[1], shrink=0.85)

    image_diff = axes[2].imshow(
        diff_2d,
        origin="lower",
        cmap="coolwarm",
        vmin=-diff_limit,
        vmax=diff_limit,
        extent=extent,
        aspect="auto",
    )
    axes[2].set_title("Prediction - Target")
    fig.colorbar(image_diff, ax=axes[2], shrink=0.85)

    for axis in axes:
        axis.set_xlabel("Longitude" if extent is not None else "X")
        axis.set_ylabel("Latitude" if extent is not None else "Y")

    fig.suptitle(format_title(npz_path, bbox=bbox, time_range=time_range, target_date=target_date))

    output_path = output_dir / f"{npz_path.stem}.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = choose_output_dir(input_path, args.output_dir)
    npz_files = resolve_npz_files(input_path)

    print(f"Found {len(npz_files)} prediction files")
    print(f"Saving plots to {output_dir}")

    generated_paths: list[Path] = []
    for npz_path in npz_files:
        output_path = plot_single_file(
            npz_path,
            output_dir=output_dir,
            time_index=args.time_index,
            channel_index=args.channel_index,
            dpi=args.dpi,
        )
        generated_paths.append(output_path)
        print(f"saved {output_path}")

    print(f"Generated {len(generated_paths)} plot files")


if __name__ == "__main__":
    main()
