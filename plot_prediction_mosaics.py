#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from plot_prediction_regions import (
    PredictionRecord,
    choose_source_key,
    collect_available_dates,
    compute_log_color_limits,
    compute_sst_limits,
    compute_value_limits,
    field_uses_sst_scale,
    has_plottable_values,
    load_optional_sst,
    mask_sst_padding,
    masked_prediction_squared_error,
    prepare_log_scaled_field,
    resolve_npz_files,
    select_2d_slice,
    source_uses_ssh_scale,
)


# Build one geographically merged figure per target date. Each saved prediction
# patch is drawn at its bbox lon/lat extent, with alpha so overlapping patches
# remain visible instead of hiding each other completely.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay prediction patches into one lon/lat mosaic plot per target date."
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
        help="Generate one merged mosaic plot per available target date.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output PNG path. Only valid when plotting a single date.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated PNGs. Defaults to <input>/merged_plots.",
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
        "--source-key",
        default="auto",
        help="Which source field to show. Use 'auto', 'input_l3_ssh', or 'input_l4_sst'.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Patch transparency for overlapping regions, from 0.0 to 1.0.",
    )
    parser.add_argument(
        "--pad-lon",
        type=float,
        default=1.0,
        help="Longitude padding around the combined mosaic extent.",
    )
    parser.add_argument(
        "--pad-lat",
        type=float,
        default=1.0,
        help="Latitude padding around the combined mosaic extent.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output DPI for the generated PNG.",
    )
    return parser.parse_args()


def default_output_dir(input_path: Path, output_dir_arg: str | None) -> Path:
    if output_dir_arg is not None:
        output_dir = Path(output_dir_arg)
    elif input_path.is_file():
        output_dir = input_path.parent / "merged_plots"
    else:
        output_dir = input_path / "merged_plots"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def default_output_path(input_path: Path, date_text: str, output_dir_arg: str | None) -> Path:
    output_dir = default_output_dir(input_path, output_dir_arg)
    safe_date = date_text.replace(":", "-")
    return output_dir / f"prediction_mosaic_{safe_date}.png"


def load_prediction_records(
    npz_files: list[Path],
    *,
    date_text: str,
    time_index: int,
    channel_index: int,
    source_key: str,
) -> list[PredictionRecord]:
    records: list[PredictionRecord] = []
    for npz_path in npz_files:
        with np.load(npz_path, allow_pickle=True) as payload:
            target_date = str(payload["target_date"]) if "target_date" in payload else ""
            if target_date != date_text:
                continue

            bbox = tuple(float(value) for value in np.asarray(payload["bbox"]).tolist())
            prediction = select_2d_slice(np.asarray(payload["prediction"]), time_index, channel_index)
            target = select_2d_slice(np.asarray(payload["target"]), time_index, channel_index)
            chosen_source_key = choose_source_key(payload, source_key)
            source = select_2d_slice(np.asarray(payload[chosen_source_key]), time_index, channel_index)
            sst = load_optional_sst(payload, time_index=time_index, channel_index=channel_index)

        records.append(
            PredictionRecord(
                path=npz_path,
                bbox=bbox,
                target_date=target_date,
                prediction=prediction,
                target=target,
                source_name=chosen_source_key,
                source=source,
                sst=sst,
            )
        )
    return records


def import_cartopy():
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("Cartopy not installed")
        return None, None
    return ccrs, cfeature


def compute_mosaic_extent(
    records: list[PredictionRecord],
    *,
    pad_lon: float,
    pad_lat: float,
) -> tuple[float, float, float, float]:
    lon_mins = [record.bbox[0] for record in records]
    lon_maxs = [record.bbox[1] for record in records]
    lat_mins = [record.bbox[2] for record in records]
    lat_maxs = [record.bbox[3] for record in records]
    return (
        min(lon_mins) - pad_lon,
        max(lon_maxs) + pad_lon,
        min(lat_mins) - pad_lat,
        max(lat_maxs) + pad_lat,
    )


def compute_mosaic_limits(records: list[PredictionRecord]) -> dict[str, tuple[float, float]]:
    prediction_target_arrays = [array for record in records for array in (record.prediction, record.target)]
    prediction_limits = compute_value_limits(*prediction_target_arrays)

    source_arrays = [record.source for record in records]
    source_names = {record.source_name for record in records}
    if len(source_names) == 1 and source_uses_ssh_scale(next(iter(source_names))):
        source_limits = prediction_limits
    elif len(source_names) == 1 and field_uses_sst_scale(next(iter(source_names))):
        source_limits = compute_sst_limits(*source_arrays)
    else:
        source_limits = compute_value_limits(*source_arrays)

    squared_error_arrays = [masked_prediction_squared_error(record.prediction, record.target) for record in records]
    squared_error_limits = compute_log_color_limits(*squared_error_arrays)
    sst_arrays = [record.sst for record in records if has_plottable_values(record.sst, ignore_zeros=True)]
    sst_limits = compute_sst_limits(*sst_arrays) if sst_arrays else (-1.0, 1.0)

    return {
        "source": source_limits,
        "sst": sst_limits,
        "prediction": prediction_limits,
        "target": prediction_limits,
        "squared_error": squared_error_limits,
    }


def sorted_records(records: list[PredictionRecord]) -> list[PredictionRecord]:
    return sorted(records, key=lambda record: (record.bbox[2], record.bbox[0], record.path.name))


def masked_for_overlay(field: np.ndarray, *, mask_zero_padding: bool = False) -> np.ma.MaskedArray:
    return mask_sst_padding(field) if mask_zero_padding else np.ma.masked_invalid(field)


def add_base_map(axis, *, extent: tuple[float, float, float, float], use_cartopy: bool, ccrs_module, cfeature_module) -> None:
    lon_min, lon_max, lat_min, lat_max = extent
    if use_cartopy:
        axis.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs_module.PlateCarree())
        axis.coastlines(resolution="110m", linewidth=0.8)
        axis.add_feature(cfeature_module.LAND, facecolor="#f2efe8", edgecolor="none", zorder=0)
        axis.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.45, linestyle="--")
    else:
        axis.set_xlim(lon_min, lon_max)
        axis.set_ylim(lat_min, lat_max)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)


def overlay_field(
    axis,
    records: list[PredictionRecord],
    *,
    field_name: str,
    cmap_name: str,
    value_limits: tuple[float, float],
    alpha: float,
    use_cartopy: bool,
    ccrs_module,
    log_scale: bool = False,
):
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(alpha=0.0)
    vmin, vmax = value_limits
    last_image = None

    for record in records:
        if field_name == "source":
            field = record.source
            mask_zero_padding = field_uses_sst_scale(record.source_name)
        elif field_name == "prediction":
            field = record.prediction
            mask_zero_padding = False
        elif field_name == "target":
            field = record.target
            mask_zero_padding = False
        elif field_name == "squared_error":
            field = masked_prediction_squared_error(record.prediction, record.target)
            mask_zero_padding = False
        elif field_name == "sst":
            if record.sst is None:
                continue
            field = record.sst
            mask_zero_padding = True
        else:
            raise ValueError(f"Unknown field name: {field_name}")

        lon_min, lon_max, lat_min, lat_max = record.bbox
        display_field = (
            prepare_log_scaled_field(field, vmin)
            if log_scale
            else masked_for_overlay(field, mask_zero_padding=mask_zero_padding)
        )
        image_kwargs = {
            "origin": "lower",
            "extent": [lon_min, lon_max, lat_min, lat_max],
            "cmap": cmap,
            "alpha": alpha,
            "aspect": "auto",
            "zorder": 2,
        }
        if log_scale:
            image_kwargs["norm"] = LogNorm(vmin=vmin, vmax=vmax)
        else:
            image_kwargs["vmin"] = vmin
            image_kwargs["vmax"] = vmax
        if use_cartopy:
            image_kwargs["transform"] = ccrs_module.PlateCarree()
        last_image = axis.imshow(display_field, **image_kwargs)

    return last_image


def plot_date_mosaic(
    records: list[PredictionRecord],
    output_path: Path,
    *,
    date_text: str,
    alpha: float,
    pad_lon: float,
    pad_lat: float,
    dpi: int,
) -> Path:
    if not records:
        raise ValueError(f"No prediction records found for target_date={date_text}")

    alpha = min(max(alpha, 0.0), 1.0)
    ccrs, cfeature = import_cartopy()
    use_cartopy = ccrs is not None and cfeature is not None
    records = sorted_records(records)
    extent = compute_mosaic_extent(records, pad_lon=pad_lon, pad_lat=pad_lat)
    limits = compute_mosaic_limits(records)
    include_sst_panel = any(
        has_plottable_values(record.sst, ignore_zeros=True) for record in records
    ) and not all(field_uses_sst_scale(record.source_name) for record in records)

    subplot_kwargs = {"projection": ccrs.PlateCarree()} if use_cartopy else {}
    if include_sst_panel:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), subplot_kw=subplot_kwargs, constrained_layout=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw=subplot_kwargs, constrained_layout=True)
    panel_specs = [
        (
            "source",
            f"Source: {', '.join(sorted({record.source_name for record in records}))}",
            "inferno" if all(field_uses_sst_scale(record.source_name) for record in records) else "viridis",
            False,
        ),
        ("prediction", "Prediction", "viridis", False),
        ("target", "Target", "viridis", False),
        ("squared_error", "Squared Error", "magma", True),
    ]
    if include_sst_panel:
        panel_specs.insert(1, ("sst", "Input L4 SST", "inferno", False))

    for axis, (field_name, title, cmap_name, log_scale) in zip(axes.ravel(), panel_specs):
        add_base_map(axis, extent=extent, use_cartopy=use_cartopy, ccrs_module=ccrs, cfeature_module=cfeature)
        image = overlay_field(
            axis,
            records,
            field_name=field_name,
            cmap_name=cmap_name,
            value_limits=limits[field_name],
            alpha=alpha,
            use_cartopy=use_cartopy,
            ccrs_module=ccrs,
            log_scale=log_scale,
        )
        axis.set_title(title)
        if image is not None:
            fig.colorbar(image, ax=axis, shrink=0.82)
    for axis in axes.ravel()[len(panel_specs):]:
        axis.set_visible(False)

    coastline_note = "with coastlines" if use_cartopy else "without coastlines (Cartopy not installed)"
    fig.suptitle(
        f"Merged prediction mosaic for {date_text}: {len(records)} region(s), alpha={alpha:g}, {coastline_note}",
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
    if args.output is not None and args.all_dates:
        raise ValueError("--output can only be used when plotting a single date.")

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
            if args.output is not None
            else default_output_path(input_path, date_text, args.output_dir)
        )
        records = load_prediction_records(
            npz_files,
            date_text=date_text,
            time_index=args.time_index,
            channel_index=args.channel_index,
            source_key=args.source_key,
        )
        saved_path = plot_date_mosaic(
            records,
            output_path,
            date_text=date_text,
            alpha=args.alpha,
            pad_lon=args.pad_lon,
            pad_lat=args.pad_lat,
            dpi=args.dpi,
        )
        print(f"Saved {saved_path}")


if __name__ == "__main__":
    main()
