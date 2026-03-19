from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

LOGGER = logging.getLogger(__name__)


def get_coord(dataset: nc.Dataset, *candidates: str) -> np.ndarray | None:
    for name in candidates:
        if name in dataset.variables:
            return dataset.variables[name][:]
    return None


def inspect_netcdf(path: str | Path, output_dir: str | Path | None = None) -> None:
    netcdf_path = Path(path).expanduser()
    output_root = Path(output_dir).expanduser() if output_dir else netcdf_path.parent
    output_root.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(netcdf_path, "r") as dataset:
        LOGGER.info("File: %s", netcdf_path)
        LOGGER.info("Format: %s", dataset.file_format)
        for name, dim in dataset.dimensions.items():
            LOGGER.info("dim %s = %s", name, len(dim))
        for name, var in dataset.variables.items():
            LOGGER.info("var %s dims=%s dtype=%s", name, var.dimensions, var.dtype)
        lats = get_coord(dataset, "lat", "latitude", "y")
        lons = get_coord(dataset, "lon", "longitude", "x")
        times = get_coord(dataset, "time", "t")
        coord_names = {"lat", "latitude", "lon", "longitude", "x", "y", "time", "t"}
        data_vars = [name for name in dataset.variables if name not in coord_names]
        for var_name in data_vars:
            var = dataset.variables[var_name]
            data = np.ma.filled(np.squeeze(var[:]).astype(float), np.nan)
            units = getattr(var, "units", "")
            title = getattr(var, "long_name", var_name)
            if data.ndim == 2:
                fig, ax = plt.subplots(figsize=(12, 5))
                if lons is not None and lats is not None:
                    mesh = ax.pcolormesh(lons, lats, data, shading="auto")
                else:
                    mesh = ax.imshow(data, origin="lower", aspect="auto")
                plt.colorbar(mesh, ax=ax, label=f"{var_name} [{units}]")
                ax.set_title(title)
                fig.tight_layout()
                fig.savefig(output_root / f"{var_name}_2d.png", dpi=150)
                plt.close(fig)
            elif data.ndim == 3:
                for index, suffix in ((0, "first"), (data.shape[0] - 1, "last")):
                    frame = data[index]
                    fig, ax = plt.subplots(figsize=(12, 5))
                    if lons is not None and lats is not None:
                        mesh = ax.pcolormesh(lons, lats, frame, shading="auto")
                    else:
                        mesh = ax.imshow(frame, origin="lower", aspect="auto")
                    plt.colorbar(mesh, ax=ax, label=f"{var_name} [{units}]")
                    ax.set_title(f"{title} ({suffix})")
                    fig.tight_layout()
                    fig.savefig(output_root / f"{var_name}_{suffix}.png", dpi=150)
                    plt.close(fig)
                series = np.nanmean(data, axis=(1, 2))
                fig, ax = plt.subplots(figsize=(10, 3))
                x = times if times is not None else np.arange(data.shape[0])
                ax.plot(x, series)
                ax.set_title(f"{title} spatial mean")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(output_root / f"{var_name}_timeseries.png", dpi=150)
                plt.close(fig)
