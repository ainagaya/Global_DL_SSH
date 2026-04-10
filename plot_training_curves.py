#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training and validation losses from a CSV log.")
    parser.add_argument("loss_csv", help="Path to a loss CSV file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Defaults to the CSV path with a .png suffix.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output DPI for the saved figure.",
    )
    return parser.parse_args()


def load_loss_rows(loss_csv: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs: list[int] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    with loss_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_text = row["val_loss"].strip().lower()
            val_losses.append(float("nan") if val_text == "nan" else float(row["val_loss"]))

    if not epochs:
        raise ValueError(f"No training rows found in {loss_csv}")

    return np.asarray(epochs), np.asarray(train_losses), np.asarray(val_losses)


def plot_curves(loss_csv: Path, output_path: Path, dpi: int) -> Path:
    epochs, train_losses, val_losses = load_loss_rows(loss_csv)

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    axis.plot(epochs, train_losses, color="#1b6ca8", marker="o", linewidth=2.0, label="Train loss")

    valid_mask = np.isfinite(val_losses)
    if np.any(valid_mask):
        axis.plot(
            epochs[valid_mask],
            val_losses[valid_mask],
            color="#d95f02",
            marker="s",
            linewidth=2.0,
            label="Validation loss",
        )

    axis.set_title(loss_csv.stem)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    loss_csv = Path(args.loss_csv)
    if not loss_csv.exists():
        raise FileNotFoundError(f"Loss CSV not found: {loss_csv}")

    output_path = Path(args.output) if args.output else loss_csv.with_suffix(".png")
    saved_path = plot_curves(loss_csv, output_path, args.dpi)
    print(f"Saved {saved_path}")


if __name__ == "__main__":
    main()
