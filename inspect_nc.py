"""
Inspect and plot a NetCDF file.
Usage: python inspect_nc.py [path/to/file.nc]
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import netCDF4 as nc

# ── file path ──────────────────────────────────────────────────────────────────
NC_FILE = (
    sys.argv[1] if len(sys.argv) > 1
    else "SimVP_SSH-SST_1M_grads/SimVP_SSH_SST_1M_global_L250km_mappedSLA_20250101_20260319.nc"
)

# ── open & inspect ─────────────────────────────────────────────────────────────
ds = nc.Dataset(NC_FILE, "r")

print("=" * 60)
print(f"File : {NC_FILE}")
print(f"Format: {ds.file_format}")
print("=" * 60)

print("\n── Global attributes ──────────────────────────────────────")
for attr in ds.ncattrs():
    print(f"  {attr}: {getattr(ds, attr)}")

print("\n── Dimensions ─────────────────────────────────────────────")
for name, dim in ds.dimensions.items():
    print(f"  {name}: {len(dim)}{' (unlimited)' if dim.isunlimited() else ''}")

print("\n── Variables ───────────────────────────────────────────────")
for name, var in ds.variables.items():
    print(f"  {name} {var.dimensions} dtype={var.dtype}")
    for attr in var.ncattrs():
        print(f"      {attr}: {getattr(var, attr)}")

# ── helper: get coordinate arrays ─────────────────────────────────────────────
def get_coord(ds, *candidates):
    """Return the first matching variable name found in the dataset."""
    for name in candidates:
        if name in ds.variables:
            return ds.variables[name][:]
    return None

lats = get_coord(ds, "lat", "latitude", "y")
lons = get_coord(ds, "lon", "longitude", "x")
times = get_coord(ds, "time", "t")

# ── pick the main data variable (skip coordinate vars) ────────────────────────
COORD_NAMES = {"lat", "latitude", "lon", "longitude", "x", "y", "time", "t"}
data_vars = [v for v in ds.variables if v not in COORD_NAMES]
print(f"\nData variables to plot: {data_vars}")

# ── plot ───────────────────────────────────────────────────────────────────────
for var_name in data_vars:
    var = ds.variables[var_name]
    data = var[:]  # masked array

    # Squeeze out size-1 dimensions and convert to plain float array
    data = np.squeeze(data)
    data = np.ma.filled(data.astype(float), np.nan)  # read-write float64, masked → NaN

    units = getattr(var, "units", "")
    long_name = getattr(var, "long_name", var_name)

    print(f"\n{var_name}: shape={data.shape}  min={np.nanmin(data):.4f}  "
          f"max={np.nanmax(data):.4f}  mean={np.nanmean(data):.4f}")

    # ── 2-D field ──────────────────────────────────────────────────────────────
    if data.ndim == 2:
        fig, ax = plt.subplots(figsize=(12, 5))
        if lons is not None and lats is not None:
            im = ax.pcolormesh(lons, lats, data, cmap="RdBu_r", shading="auto")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        else:
            im = ax.imshow(data, origin="lower", cmap="RdBu_r", aspect="auto")
        plt.colorbar(im, ax=ax, label=f"{var_name} [{units}]")
        ax.set_title(f"{long_name}  |  shape={data.shape}")
        plt.tight_layout()
        plt.savefig(f"{var_name}_2d.png", dpi=150)
        plt.show()

    # ── 3-D field (time, lat, lon)  → first & last time step + time series ────
    elif data.ndim == 3:
        n_t = data.shape[0]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, t_idx, label in zip(axes, [0, n_t - 1], ["t=0 (first)", f"t={n_t-1} (last)"]):
            frame = data[t_idx]
            if lons is not None and lats is not None:
                im = ax.pcolormesh(lons, lats, frame, cmap="RdBu_r", shading="auto")
                ax.set_xlabel("Longitude")
                ax.set_ylabel("Latitude")
            else:
                im = ax.imshow(frame, origin="lower", cmap="RdBu_r", aspect="auto")
            plt.colorbar(im, ax=ax, label=f"{var_name} [{units}]")
            ax.set_title(f"{long_name} — {label}")
        plt.tight_layout()
        plt.savefig(f"{var_name}_first_last.png", dpi=150)
        plt.show()

        # spatial-mean time series
        ts = np.nanmean(data, axis=(1, 2))
        fig, ax = plt.subplots(figsize=(10, 3))
        x = times if times is not None else np.arange(n_t)
        ax.plot(x, ts, lw=1.5)
        ax.set_xlabel("Time index" if times is None else "Time")
        ax.set_ylabel(f"Spatial mean [{units}]")
        ax.set_title(f"{long_name} — spatial mean over time")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{var_name}_timeseries.png", dpi=150)
        plt.show()

    else:
        print(f"  Skipping {var_name} — unsupported shape {data.shape}")

ds.close()
print("\nDone.")