from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_epoch(path: str | Path) -> int:
    match = re.search(r"epoch(\d+)", Path(path).name)
    return int(match.group(1)) if match else 10**9


def prepare_array(array: np.ndarray, sample_idx: int = 0, time_idx: int = 0, channel_idx: int = 0) -> np.ndarray:
    if array.ndim == 5:
        return array[sample_idx, time_idx, channel_idx]
    if array.ndim == 4:
        return array[sample_idx, time_idx]
    if array.ndim == 3:
        return array[sample_idx]
    if array.ndim == 2:
        return array
    raise ValueError(f"Unsupported array shape: {array.shape}")


def plot_epoch_maps(input_dir: str | Path, output_dir: str | Path, pattern: str, sample_idx: int, time_idx: int, channel_idx: int, cmap: str) -> None:
    input_root = Path(input_dir).expanduser()
    output_root = Path(output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    files = sorted(input_root.glob(pattern), key=extract_epoch)
    if not files:
        raise FileNotFoundError(f"No files found in {input_root} matching {pattern}")
    for path in files:
        array = np.load(path)
        image = prepare_array(array, sample_idx=sample_idx, time_idx=time_idx, channel_idx=channel_idx)
        epoch = extract_epoch(path)
        fig, ax = plt.subplots(figsize=(6, 5))
        handle = ax.imshow(image, origin="lower", aspect="auto", cmap=cmap)
        plt.colorbar(handle, ax=ax)
        ax.set_title(f"Epoch {epoch}")
        fig.tight_layout()
        fig.savefig(output_root / f"epoch_{epoch:03d}.png", dpi=150)
        plt.close(fig)


def plot_losses(csv_path: str | Path, output_path: str | Path, epoch_col: str | None = None, train_col: str | None = None, val_col: str | None = None) -> None:
    frame = pd.read_csv(csv_path)
    columns = {col.lower(): col for col in frame.columns}
    epoch_col = epoch_col or columns.get("epoch")
    train_col = train_col or columns.get("train_loss") or columns.get("train loss")
    val_col = val_col or columns.get("val_loss") or columns.get("val loss")
    if not epoch_col or not train_col or not val_col:
        raise ValueError(f"Could not infer columns from {list(frame.columns)}")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(frame[epoch_col], frame[train_col], marker="o", label="Train")
    ax.plot(frame[epoch_col], frame[val_col], marker="o", label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training history")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
