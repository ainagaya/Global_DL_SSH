#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def find_column(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot train/val losses against epoch from CSV.")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--output", required=True, help="Output image path")
    parser.add_argument("--epoch-col", default=None, help="Epoch column name")
    parser.add_argument("--train-col", default=None, help="Train loss column name")
    parser.add_argument("--val-col", default=None, help="Validation loss column name")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    epoch_col = args.epoch_col or find_column(df, ["Epoch", "epochs"])
    train_col = args.train_col or find_column(df, ["Train Loss", "loss_train", "training_loss", "train"])
    val_col = args.val_col or find_column(df, ["Val Loss", "valid_loss", "validation_loss", "val"])

    if epoch_col is None:
        raise ValueError(
            f"Could not find epoch column in CSV. Available columns: {list(df.columns)}"
        )
    if train_col is None:
        raise ValueError(
            f"Could not find train loss column in CSV. Available columns: {list(df.columns)}"
        )
    if val_col is None:
        raise ValueError(
            f"Could not find validation loss column in CSV. Available columns: {list(df.columns)}"
        )

    plt.figure(figsize=(8, 5))
    plt.plot(df[epoch_col], df[train_col], marker="o", label="Train loss")
    plt.plot(df[epoch_col], df[val_col], marker="o", label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train/Validation Loss vs Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Saved loss plot to: {args.output}")


if __name__ == "__main__":
    main()