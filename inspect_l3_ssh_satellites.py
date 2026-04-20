#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect L3 SSH NetCDF files and list available satellite/platform coverage using xarray."
    )
    parser.add_argument(
        "input_path",
        help="A single l3_ssh.nc file or a directory containing OceanTACO l3_ssh.nc files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional maximum number of files to inspect.",
    )
    return parser.parse_args()


def resolve_l3_ssh_files(input_path: Path, max_files: int | None) -> list[Path]:
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(input_path.rglob("l3_ssh.nc"))
    else:
        raise ValueError(f"Input path does not exist: {input_path}")

    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise ValueError(f"No l3_ssh.nc files found under: {input_path}")
    return files


def inspect_file(path: Path) -> Counter[str]:
    with xr.open_dataset(path) as ds:
        if "track_platforms" not in ds or "primary_track" not in ds:
            print(f"{path}: missing track_platforms or primary_track")
            return Counter()

        platforms = [str(value) for value in ds["track_platforms"].values]
        primary_track = ds["primary_track"]
        if "time" in primary_track.dims:
            primary_track = primary_track.isel(time=0)

        counts: Counter[str] = Counter()
        primary_values = primary_track.values
        for track_index, platform in enumerate(platforms):
            counts[platform] += int(np.nansum(primary_values == track_index))

        print(path)
        for platform, count in counts.most_common():
            print(f"  {platform}: {count} grid cells")
        return counts


def main() -> None:
    args = parse_args()
    files = resolve_l3_ssh_files(Path(args.input_path), args.max_files)
    total_counts: Counter[str] = Counter()

    print(f"Inspecting {len(files)} L3 SSH files")
    for path in files:
        total_counts.update(inspect_file(path))

    print("\nTotal coverage across inspected files:")
    for platform, count in total_counts.most_common():
        print(f"  {platform}: {count} grid cells")


if __name__ == "__main__":
    main()
