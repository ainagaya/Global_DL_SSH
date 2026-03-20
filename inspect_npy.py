"""
Inspect and plot a .npy file.
Usage: python inspect_npy.py [path/to/file.npy]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# ── file path ──────────────────────────────────────────────────────────────────
NPY_FILE = sys.argv[1] if len(sys.argv) > 1 else "file.npy"

# ── load & inspect ─────────────────────────────────────────────────────────────
data = np.load(NPY_FILE, allow_pickle=False)

print("=" * 60)
print(f"File  : {NPY_FILE}")
print(f"Shape : {data.shape}")
print(f"Dtype : {data.dtype}")
print(f"Ndim  : {data.ndim}")
print(f"Size  : {data.size} elements")
print(f"Memory: {data.nbytes / 1e6:.2f} MB")
print("=" * 60)

# Convert to float for stats (handles int/uint arrays too)
data_f = data.astype(float)
finite = data_f[np.isfinite(data_f)]

print(f"\nMin   : {finite.min():.6f}")
print(f"Max   : {finite.max():.6f}")
print(f"Mean  : {finite.mean():.6f}")
print(f"Std   : {finite.std():.6f}")
print(f"NaNs  : {np.isnan(data_f).sum()}")
print(f"Infs  : {np.isinf(data_f).sum()}")

# ── plot ───────────────────────────────────────────────────────────────────────

# 1-D
if data.ndim == 1:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(data_f, lw=1)
    axes[0].set_title("Values")
    axes[0].set_xlabel("Index")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(finite, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[1].set_title("Histogram")
    axes[1].set_xlabel("Value")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"{NPY_FILE}  |  shape={data.shape}", fontsize=10)
    plt.tight_layout()
    plt.savefig("npy_1d.png", dpi=150)
    plt.show()

# 2-D
elif data.ndim == 2:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(data_f, cmap="RdBu_r", aspect="auto",
                        vmin=np.nanpercentile(data_f, 2),
                        vmax=np.nanpercentile(data_f, 98))
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("2D field")

    axes[1].hist(finite, bins=50, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[1].set_title("Histogram")
    axes[1].set_xlabel("Value")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f"{NPY_FILE}  |  shape={data.shape}", fontsize=10)
    plt.tight_layout()
    plt.savefig("npy_2d.png", dpi=150)
    plt.show()

# 3-D  (e.g. time × lat × lon  or  frames × H × W)
elif data.ndim == 3:
    n_t = data.shape[0]
    vmin = np.nanpercentile(data_f, 2)
    vmax = np.nanpercentile(data_f, 98)

    # First / middle / last frames
    indices = sorted({0, n_t // 2, n_t - 1})
    fig, axes = plt.subplots(1, len(indices), figsize=(6 * len(indices), 5))
    if len(indices) == 1:
        axes = [axes]

    for ax, t in zip(axes, indices):
        im = ax.imshow(data_f[t], cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Frame {t}")

    plt.suptitle(f"{NPY_FILE}  |  shape={data.shape}", fontsize=10)
    plt.tight_layout()
    plt.savefig("npy_3d_frames.png", dpi=150)
    plt.show()

    # Spatial-mean time series
    ts = np.nanmean(data_f, axis=(1, 2))
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ts, lw=1.5)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Spatial mean")
    ax.set_title(f"{NPY_FILE}  —  spatial mean over time")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("npy_3d_timeseries.png", dpi=150)
    plt.show()

    # Histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(finite, bins=80, color="steelblue", edgecolor="white", linewidth=0.3)
    ax.set_title(f"{NPY_FILE}  —  histogram")
    ax.set_xlabel("Value")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("npy_3d_hist.png", dpi=150)
    plt.show()

# 4-D  (e.g. time × channels × H × W)
elif data.ndim == 4:
    n_t, n_c = data.shape[:2]
    print(f"\n4D array: {n_t} time steps, {n_c} channels")

    vmin = np.nanpercentile(data_f, 2)
    vmax = np.nanpercentile(data_f, 98)

    # Show all channels at t=0
    fig, axes = plt.subplots(1, n_c, figsize=(5 * n_c, 4))
    if n_c == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        im = ax.imshow(data_f[0, c], cmap="RdBu_r", aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"t=0, ch={c}")

    plt.suptitle(f"{NPY_FILE}  |  shape={data.shape}  (t=0, all channels)", fontsize=10)
    plt.tight_layout()
    plt.savefig("npy_4d_channels.png", dpi=150)
    plt.show()

else:
    print(f"Shape {data.shape} — no plot implemented for {data.ndim}-D arrays.")

print("\nDone.")