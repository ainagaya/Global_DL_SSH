#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# This script is meant for geographically debugging a prediction campaign.
# Unlike `plot_predictions.py`, which treats each `.npz` file independently,
# this script groups files by target date and can place many predicted regions
# into a single figure.
#
# Expected `.npz` keys from `simvp_predict_ssh.py`:
# - prediction
# - target
# - bbox
# - target_date
# - input_l3_ssh / input_l4_sst (or both)
#
# The requested output layout is:
# - one row per predicted region/sample
# - a wider "context" map around the sample bbox
# - the sample bbox drawn on that wider map
# - source, prediction, and target panels for the selected date
# - coastlines when Cartopy is available
#
# The script is intentionally forgiving:
# - it can read a single file or a whole directory
# - it clamps time/channel indices
# - it can fall back to plain matplotlib if Cartopy is unavailable


@dataclass
class PredictionRecord:
    path: Path
    bbox: tuple[float, float, float, float]
    target_date: str
    prediction: np.ndarray
    target: np.ndarray
    source_name: str
    source: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot prediction regions on geographically contextual maps."
    )
    parser.add_argument(
        "input_path",
        help="A single .npz file or a directory containing prediction .npz files.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Target date to plot, for example 2025-03-21.",
    )
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="Generate one regional plot per available target date in the input files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Defaults to <input>/regional_plots/<date>.png.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Time index to use when arrays contain a time dimension.",
    )
    parser.add_argument(
        "--channel-index",
        type=int,
        default=0,
        help="Channel index to use when arrays contain a channel dimension.",
    )
    parser.add_argument(
        "--context-pad-lon",
        type=float,
        default=30.0,
        help="Extra longitude degrees to show around each bbox on the context map.",
    )
    parser.add_argument(
        "--context-pad-lat",
        type=float,
        default=30.0,
        help="Extra latitude degrees to show around each bbox on the context map.",
    )
    parser.add_argument(
        "--source-key",
        default="auto",
        help="Which source field to show. Use 'auto', 'input_l3_ssh', or 'input_l4_sst'.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output DPI for the generated PNG.",
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


def default_output_path(input_path: Path, date_text: str) -> Path:
    if input_path.is_file():
        output_dir = input_path.parent / "regional_plots"
    else:
        output_dir = input_path / "regional_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_date = date_text.replace(":", "-")
    return output_dir / f"prediction_regions_{safe_date}.png"


def clamp_index(index: int, size: int) -> int:
    if size <= 0:
        raise ValueError("Cannot select from an empty dimension.")
    return min(max(index, 0), size - 1)


def select_2d_slice(array: np.ndarray, time_index: int, channel_index: int) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        chosen_index = clamp_index(channel_index, array.shape[0])
        return array[chosen_index]
    if array.ndim == 4:
        chosen_time = clamp_index(time_index, array.shape[0])
        chosen_channel = clamp_index(channel_index, array.shape[1])
        return array[chosen_time, chosen_channel]
    raise ValueError(f"Unsupported array rank: shape={array.shape}")


def choose_source_key(payload: np.lib.npyio.NpzFile, source_key: str) -> str:
    def is_usable_field(key: str) -> bool:
        array = np.asarray(payload[key])
        return array.ndim >= 2 and array.size > 0

    if source_key != "auto":
        if source_key not in payload:
            raise KeyError(f"Requested source key '{source_key}' not found in prediction file.")
        if not is_usable_field(source_key):
            raise ValueError(f"Requested source key '{source_key}' exists but does not contain plottable data.")
        return source_key

    preferred_keys = ["input_l3_ssh", "input_l4_sst"]
    for key in preferred_keys:
        if key in payload and is_usable_field(key):
            return key

    input_keys = sorted(key for key in payload.files if key.startswith("input_") and is_usable_field(key))
    if not input_keys:
        raise KeyError("No source field found. Expected one of the saved input_* arrays.")
    return input_keys[0]


def load_prediction_record(
    npz_path: Path,
    *,
    date_text: str,
    time_index: int,
    channel_index: int,
    source_key: str,
) -> PredictionRecord | None:
    with np.load(npz_path, allow_pickle=True) as payload:
        target_date = str(payload["target_date"]) if "target_date" in payload else ""
        if target_date != date_text:
            return None

        bbox = tuple(float(value) for value in np.asarray(payload["bbox"]).tolist())
        prediction = select_2d_slice(np.asarray(payload["prediction"]), time_index, channel_index)
        target = select_2d_slice(np.asarray(payload["target"]), time_index, channel_index)
        chosen_source_key = choose_source_key(payload, source_key)
        source = select_2d_slice(np.asarray(payload[chosen_source_key]), time_index, channel_index)

    return PredictionRecord(
        path=npz_path,
        bbox=bbox,
        target_date=target_date,
        prediction=prediction,
        target=target,
        source_name=chosen_source_key,
        source=source,
    )


def collect_available_dates(npz_files: list[Path]) -> list[str]:
    dates: set[str] = set()
    for npz_path in npz_files:
        with np.load(npz_path, allow_pickle=True) as payload:
            if "target_date" in payload:
                dates.add(str(payload["target_date"]))
    return sorted(dates)


def compute_value_limits(*arrays: np.ndarray) -> tuple[float, float]:
    finite_values = []
    for array in arrays:
        finite = array[np.isfinite(array)]
        if finite.size:
            finite_values.append(finite)
    if not finite_values:
        return -1.0, 1.0
    combined = np.concatenate(finite_values)
    return float(combined.min()), float(combined.max())


def valid_swot_target_mask(target: np.ndarray) -> np.ndarray:
    return np.isfinite(target) & (target != 0.0)


def masked_prediction_squared_error(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.where(valid_swot_target_mask(target), np.square(prediction - target), np.nan)


def source_uses_ssh_scale(source_name: str) -> bool:
    return "ssh" in source_name.lower()


def compute_panel_limits(records: list[PredictionRecord]) -> tuple[tuple[float, float], dict[str, tuple[float, float]]]:
    prediction_target_arrays = [array for record in records for array in (record.prediction, record.target)]
    value_limits = compute_value_limits(*prediction_target_arrays)

    source_arrays_by_name: dict[str, list[np.ndarray]] = {}
    for record in records:
        source_arrays_by_name.setdefault(record.source_name, []).append(record.source)

    source_limits: dict[str, tuple[float, float]] = {}
    for source_name, source_arrays in source_arrays_by_name.items():
        if source_uses_ssh_scale(source_name):
            source_limits[source_name] = value_limits
        else:
            source_limits[source_name] = compute_value_limits(*source_arrays)

    return value_limits, source_limits


def import_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy not installed")
        return None, None
    return ccrs, cfeature


def sort_records(records: list[PredictionRecord]) -> list[PredictionRecord]:
    return sorted(records, key=lambda record: (record.bbox[2], record.bbox[0], record.path.name))


def add_context_panel(
    axis,
    record: PredictionRecord,
    context_pad_lon: float,
    context_pad_lat: float,
    *,
    use_cartopy: bool,
    ccrs_module,
    cfeature_module,
) -> None:
    lon_min, lon_max, lat_min, lat_max = record.bbox
    context_extent = [
        lon_min - context_pad_lon,
        lon_max + context_pad_lon,
        lat_min - context_pad_lat,
        lat_max + context_pad_lat,
    ]

    if use_cartopy:
        axis.set_extent(context_extent, crs=ccrs_module.PlateCarree())
        axis.coastlines(resolution="110m", linewidth=0.8)
        axis.add_feature(cfeature_module.LAND, facecolor="#f1efe8", edgecolor="none", zorder=0)
        axis.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")
    else:
        axis.set_xlim(context_extent[0], context_extent[1])
        axis.set_ylim(context_extent[2], context_extent[3])
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    bbox_patch = mpatches.Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        linewidth=2.0,
        edgecolor="crimson",
        facecolor="none",
        zorder=3,
    )
    if use_cartopy:
        axis.add_patch(bbox_patch)
        bbox_patch.set_transform(ccrs_module.PlateCarree())
    else:
        axis.add_patch(bbox_patch)

    axis.set_title("Context + bbox")


def add_data_panel(axis, field: np.ndarray, bbox: tuple[float, float, float, float], title: str, cmap: str, vmin: float, vmax: float) -> None:
    lon_min, lon_max, lat_min, lat_max = bbox
    image = axis.imshow(
        field,
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    axis.set_title(title)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    return image


def plot_date_records(
    records: list[PredictionRecord],
    output_path: Path,
    *,
    date_text: str,
    context_pad_lon: float,
    context_pad_lat: float,
    dpi: int,
) -> Path:
    if not records:
        raise ValueError(f"No prediction records found for target_date={date_text}")

    ccrs, cfeature = import_cartopy()
    use_cartopy = ccrs is not None and cfeature is not None
    sorted_records = sort_records(records)
    (value_min, value_max), source_limits = compute_panel_limits(sorted_records)

    row_count = len(sorted_records)
    fig = plt.figure(figsize=(18, 4.6 * row_count), constrained_layout=True)
    subplot_spec = fig.add_gridspec(row_count, 4, width_ratios=[1.1, 1.0, 1.0, 1.0])

    for row_index, record in enumerate(sorted_records):
        source_min, source_max = source_limits[record.source_name]

        if use_cartopy:
            context_axis = fig.add_subplot(subplot_spec[row_index, 0], projection=ccrs.PlateCarree())
        else:
            context_axis = fig.add_subplot(subplot_spec[row_index, 0])

        source_axis = fig.add_subplot(subplot_spec[row_index, 1])
        prediction_axis = fig.add_subplot(subplot_spec[row_index, 2])
        target_axis = fig.add_subplot(subplot_spec[row_index, 3])

        add_context_panel(
            context_axis,
            record,
            context_pad_lon=context_pad_lon,
            context_pad_lat=context_pad_lat,
            use_cartopy=use_cartopy,
            ccrs_module=ccrs,
            cfeature_module=cfeature,
        )

        source_image = add_data_panel(
            source_axis,
            record.source,
            record.bbox,
            f"Source: {record.source_name}",
            "viridis",
            source_min,
            source_max,
        )
        prediction_image = add_data_panel(
            prediction_axis,
            record.prediction,
            record.bbox,
            "Prediction",
            "viridis",
            value_min,
            value_max,
        )
        target_image = add_data_panel(
            target_axis,
            record.target,
            record.bbox,
            "Target",
            "viridis",
            value_min,
            value_max,
        )

        fig.colorbar(source_image, ax=source_axis, shrink=0.8)
        fig.colorbar(prediction_image, ax=prediction_axis, shrink=0.8)
        fig.colorbar(target_image, ax=target_axis, shrink=0.8)

        lon_min, lon_max, lat_min, lat_max = record.bbox
        context_axis.text(
            0.02,
            0.02,
            f"{record.path.name}\nlon=({lon_min:.2f}, {lon_max:.2f}) lat=({lat_min:.2f}, {lat_max:.2f})",
            transform=context_axis.transAxes,
            fontsize=9,
            ha="left",
            va="bottom",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    coastline_note = "with coastlines" if use_cartopy else "without coastlines (Cartopy not installed)"
    fig.suptitle(
        f"Predictions for {date_text} across {len(sorted_records)} region(s), {coastline_note}",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    npz_files = resolve_npz_files(input_path)
    if args.all_dates:
        date_values = collect_available_dates(npz_files)
        if not date_values:
            raise ValueError("No target_date values found in the provided prediction files.")
    else:
        if args.date is None:
            raise ValueError("Provide --date <YYYY-MM-DD> or use --all-dates.")
        date_values = [args.date]

    for date_text in date_values:
        output_path = (
            Path(args.output)
            if args.output is not None and len(date_values) == 1
            else default_output_path(input_path, date_text)
        )
        records = []
        for npz_path in npz_files:
            record = load_prediction_record(
                npz_path,
                date_text=date_text,
                time_index=args.time_index,
                channel_index=args.channel_index,
                source_key=args.source_key,
            )
            if record is not None:
                records.append(record)

        saved_path = plot_date_records(
            records,
            output_path,
            date_text=date_text,
            context_pad_lon=args.context_pad_lon,
            context_pad_lat=args.context_pad_lat,
            dpi=args.dpi,
        )
        print(f"Saved {saved_path}")


if __name__ == "__main__":
    main()
