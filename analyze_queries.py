#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# This script is intentionally comment-heavy because it is meant to support
# debugging sessions where we are trying to understand *why* OceanTACO-based
# datasets look empty. There are two different debugging levels:
#
# 1. Query metadata inspection
#    This answers questions like:
#    - Were queries written at all?
#    - Which bbox/date ranges do they cover?
#    - Are they duplicated or malformed?
#
# 2. Data inspection
#    This goes one step further and actually loads samples for stored queries
#    using the OceanTACO dataset API, then checks each configured variable for:
#    - `None`
#    - all-NaN / no finite values
#    - all-zero values
#
# That second mode is especially useful here because "empty dataset" problems
# often turn out to mean:
# - files exist, but a variable resolves to `None`
# - values are technically present, but every value is NaN
# - upstream loaders convert NaNs to zeros, so everything looks blank


def emit_warning(message: str) -> None:
    # Keep warnings visually obvious in plain terminal output.
    print(f"WARNING: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect saved query files and optionally validate their loaded data.")
    parser.add_argument(
        "query_dir",
        nargs="?",
        default="queries",
        help="Directory containing saved query files. Defaults to ./queries",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="How many sample records to print per file.",
    )
    parser.add_argument(
        "--config",
        default="configs/oceantaco.yaml",
        help="YAML config used to determine OceanTACO path, variables, and grid size when --check-data is enabled.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Load stored queries back through OceanTACO and inspect whether variables are None / all-NaN / all-zero.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=25,
        help="Maximum number of loaded query samples to inspect per query file when --check-data is enabled.",
    )
    return parser.parse_args()


def iter_query_files(root: Path) -> list[Path]:
    # We scan recursively because OceanTACO may save one file per split or may
    # create a directory tree. This also lets us point the script at a top-level
    # `queries/` directory without caring about the exact layout.
    file_paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".parquet"}:
            file_paths.append(path)
    return file_paths


def read_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # This is the pure metadata reader used even when OceanTACO is not
    # installed. It makes the script useful in lightweight environments too.
    suffix = path.suffix.lower()
    metadata: dict[str, Any] = {}

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if isinstance(payload.get("metadata"), dict):
                metadata = payload["metadata"]
            if isinstance(payload.get("queries"), list):
                return [as_record(item) for item in payload["queries"]], metadata
            if isinstance(payload.get("items"), list):
                return [as_record(item) for item in payload["items"]], metadata
            return [as_record(payload)], metadata
        if isinstance(payload, list):
            return [as_record(item) for item in payload], metadata
        raise ValueError(f"Unsupported JSON payload in {path}")

    if suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(as_record(json.loads(line)))
        return records, metadata

    if suffix == ".csv":
        frame = pd.read_csv(path)
        return frame.to_dict(orient="records"), metadata

    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        return frame.to_dict(orient="records"), metadata

    raise ValueError(f"Unsupported file type for {path}")


def as_record(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    return {"value": item}


def flatten_keys(record: dict[str, Any], prefix: str = "") -> Iterable[str]:
    for key, value in record.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        yield full_key
        if isinstance(value, dict):
            yield from flatten_keys(value, full_key)


def extract_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    candidates = [record.get("bbox"), record.get("bbox_constraint")]

    for candidate in candidates:
        if isinstance(candidate, (list, tuple)) and len(candidate) == 4:
            try:
                return tuple(float(value) for value in candidate)
            except (TypeError, ValueError):
                pass

    direct_keys = ("lon_min", "lon_max", "lat_min", "lat_max")
    if all(key in record for key in direct_keys):
        try:
            return (
                float(record["lon_min"]),
                float(record["lon_max"]),
                float(record["lat_min"]),
                float(record["lat_max"]),
            )
        except (TypeError, ValueError):
            return None

    for value in record.values():
        if isinstance(value, dict):
            nested_bbox = extract_bbox(value)
            if nested_bbox is not None:
                return nested_bbox

    return None


def extract_date_range(record: dict[str, Any]) -> tuple[str, str] | None:
    candidates = [record.get("time_range"), record.get("date_range")]
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)) and len(candidate) == 2:
            return str(candidate[0]), str(candidate[1])

    if "start_date" in record and "end_date" in record:
        return str(record["start_date"]), str(record["end_date"])

    for value in record.values():
        if isinstance(value, dict):
            nested_range = extract_date_range(value)
            if nested_range is not None:
                return nested_range

    return None


def short_json(value: Any, max_length: int = 160) -> str:
    text = json.dumps(value, default=str, sort_keys=True)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": len(records),
        "key_counts": Counter(),
        "bbox_count": 0,
        "date_range_count": 0,
        "duplicate_signatures": 0,
        "invalid_bbox_count": 0,
    }

    bbox_values: list[tuple[float, float, float, float]] = []
    date_ranges: list[tuple[str, str]] = []
    signatures = Counter()

    for record in records:
        summary["key_counts"].update(flatten_keys(record))

        bbox = extract_bbox(record)
        if bbox is not None:
            summary["bbox_count"] += 1
            bbox_values.append(bbox)
            lon_min, lon_max, lat_min, lat_max = bbox
            if lon_min > lon_max or lat_min > lat_max:
                summary["invalid_bbox_count"] += 1

        date_range = extract_date_range(record)
        if date_range is not None:
            summary["date_range_count"] += 1
            date_ranges.append(date_range)

        signatures[short_json(record, max_length=2000)] += 1

    summary["duplicate_signatures"] = sum(count - 1 for count in signatures.values() if count > 1)

    if bbox_values:
        lon_mins = [bbox[0] for bbox in bbox_values]
        lon_maxs = [bbox[1] for bbox in bbox_values]
        lat_mins = [bbox[2] for bbox in bbox_values]
        lat_maxs = [bbox[3] for bbox in bbox_values]
        summary["bbox_extent"] = {
            "lon_min_min": min(lon_mins),
            "lon_max_max": max(lon_maxs),
            "lat_min_min": min(lat_mins),
            "lat_max_max": max(lat_maxs),
            "unique_bboxes": len(set(bbox_values)),
        }

    if date_ranges:
        summary["date_extent"] = {
            "min_start": min(start for start, _ in date_ranges),
            "max_end": max(end for _, end in date_ranges),
            "unique_ranges": len(set(date_ranges)),
        }

    return summary


def print_file_report(path: Path, records: list[dict[str, Any]], metadata: dict[str, Any], limit: int) -> None:
    summary = summarize_records(records)

    print(f"\n=== {path} ===")
    print(f"records: {summary['count']}")

    if metadata:
        print(f"metadata: {short_json(metadata)}")

    print(
        "coverage:"
        f" bbox_records={summary['bbox_count']}"
        f" date_range_records={summary['date_range_count']}"
        f" invalid_bboxes={summary['invalid_bbox_count']}"
        f" duplicate_records={summary['duplicate_signatures']}"
    )

    if "bbox_extent" in summary:
        bbox_extent = summary["bbox_extent"]
        print(
            "bbox_extent:"
            f" lon=[{bbox_extent['lon_min_min']}, {bbox_extent['lon_max_max']}]"
            f" lat=[{bbox_extent['lat_min_min']}, {bbox_extent['lat_max_max']}]"
            f" unique_bboxes={bbox_extent['unique_bboxes']}"
        )

    if "date_extent" in summary:
        date_extent = summary["date_extent"]
        print(
            "date_extent:"
            f" start={date_extent['min_start']}"
            f" end={date_extent['max_end']}"
            f" unique_ranges={date_extent['unique_ranges']}"
        )

    key_counts: Counter[str] = summary["key_counts"]
    if key_counts:
        print("top_keys:")
        for key, count in key_counts.most_common(12):
            print(f"  - {key}: {count}")

    print("sample_records:")
    for record in records[:limit]:
        print(f"  - {short_json(record, max_length=240)}")


def import_runtime_helpers():
    # The data-inspection mode depends on the local project config helpers and
    # the OceanTACO-backed dataset builder. We import lazily so plain metadata
    # inspection still works even when the ML dependencies are not available.
    try:
        from src.config_utils import load_config
        from src.oceantaco import _build_patched_dataset_class, _import_oceantaco, resolve_oceantaco_path
    except ImportError as exc:
        emit_warning(
            "Data inspection requires the project runtime dependencies, including OceanTACO and its loader stack. "
            f"Import failed with: {exc}"
        )
        raise SystemExit(2) from exc

    return load_config, _build_patched_dataset_class, _import_oceantaco, resolve_oceantaco_path


def describe_variable_payload(value: Any) -> dict[str, Any]:
    # This is the key debugging routine for "empty" data. It turns a loaded
    # variable payload into flags we can count and report.
    if value is None:
        return {
            "is_none": True,
            "all_nan": False,
            "all_non_finite": False,
            "all_zero": False,
            "shape": None,
        }

    # OceanTACO variables usually arrive as dictionaries with a `data` tensor,
    # but we still handle raw arrays / tensors defensively.
    if isinstance(value, dict):
        payload = value.get("data")
    else:
        payload = value

    if payload is None:
        return {
            "is_none": True,
            "all_nan": False,
            "all_non_finite": False,
            "all_zero": False,
            "shape": None,
        }

    if hasattr(payload, "detach"):
        array = payload.detach().cpu().numpy()
    else:
        array = np.asarray(payload)

    # Empty arrays are suspicious and effectively equivalent to "no data".
    if array.size == 0:
        return {
            "is_none": False,
            "all_nan": False,
            "all_non_finite": True,
            "all_zero": True,
            "shape": tuple(array.shape),
        }

    finite_mask = np.isfinite(array)
    all_non_finite = not bool(np.any(finite_mask))
    all_nan = bool(np.isnan(array).all()) if np.issubdtype(array.dtype, np.floating) else False

    # Important note: in our patched OceanTACO pipeline, NaNs are converted to
    # zero before tensors are returned. That means "all_zero" is a useful
    # secondary flag for suspiciously empty fields even when `all_nan` is false.
    all_zero = bool(np.all(array == 0))

    return {
        "is_none": False,
        "all_nan": all_nan,
        "all_non_finite": all_non_finite,
        "all_zero": all_zero,
        "shape": tuple(int(dim) for dim in array.shape),
    }


def inspect_query_file_data(path: Path, config_path: str | Path, max_samples: int) -> None:
    load_config, build_patched_dataset_class, import_oceantaco, resolve_oceantaco_path = import_runtime_helpers()
    config = load_config(config_path)
    OceanTACODataset, _, _, QueryGenerator, HF_DEFAULT_URL, ocean_dataset_module = import_oceantaco()

    # We load the actual stored queries from disk using OceanTACO's own query
    # loader. This is much safer than trying to reconstruct Query objects from
    # raw JSON by hand.
    queries, metadata = QueryGenerator.load_queries(path)
    if not queries:
        print("data_check: no queries could be loaded for this file")
        return

    data_cfg = config["data"]
    grid_cfg = data_cfg["grid"]
    taco_path = resolve_oceantaco_path(config, HF_DEFAULT_URL)
    dataset_cls = build_patched_dataset_class(OceanTACODataset, ocean_dataset_module)
    dataset = dataset_cls(
        taco_path=taco_path,
        queries=queries,
        input_variables=[source_cfg["key"] for source_cfg in data_cfg["inputs"]],
        target_variables=[source_cfg["key"] for source_cfg in data_cfg["targets"]],
        target_resolution=data_cfg.get("target_resolution"),
        temporal_agg=data_cfg.get("temporal_agg", "stack"),
        default_patch_size=(int(grid_cfg["height"]), int(grid_cfg["width"])),
    )

    # Before we even inspect loaded variables, it is very useful to know whether
    # OceanTACO matched any candidate files for each query. If these counts are
    # all zero, the problem is upstream of variable loading.
    file_index_counts = [len(file_df) for file_df in getattr(dataset, "_file_index", [])]
    if file_index_counts:
        zero_match_queries = sum(count == 0 for count in file_index_counts)
        print(
            "file_index_summary:"
            f" min_matches={min(file_index_counts)}"
            f" max_matches={max(file_index_counts)}"
            f" zero_match_queries={zero_match_queries}"
            f" total_queries={len(file_index_counts)}"
        )

        observed_sources = Counter()
        for file_df in dataset._file_index[: min(len(dataset._file_index), max_samples)]:
            if "data_source" in file_df:
                observed_sources.update(file_df["data_source"].astype(str).tolist())
        if observed_sources:
            print("file_index_data_sources:")
            for source_name, count in observed_sources.most_common():
                print(f"  - {source_name}: {count}")
        else:
            print("file_index_data_sources: none")

    print(
        "data_check:"
        f" loaded_queries={len(queries)}"
        f" metadata={short_json(metadata) if metadata else '{}'}"
        f" inspected_samples={min(len(dataset), max_samples)}"
    )

    variable_stats: dict[str, Counter[str]] = defaultdict(Counter)
    sample_flags: list[str] = []

    for sample_index in range(min(len(dataset), max_samples)):
        try:
            sample = dataset[sample_index]
        except Exception as exc:
            sample_flags.append(f"sample[{sample_index}] failed_to_load={exc}")
            continue

        # We inspect both inputs and targets because either side may be empty.
        for group_name in ("inputs", "targets"):
            group = sample.get(group_name, {})
            for variable_name, value in group.items():
                description = describe_variable_payload(value)
                stats = variable_stats[f"{group_name}.{variable_name}"]
                stats["samples_seen"] += 1

                if description["is_none"]:
                    stats["none_count"] += 1
                if description["all_nan"]:
                    stats["all_nan_count"] += 1
                if description["all_non_finite"]:
                    stats["all_non_finite_count"] += 1
                if description["all_zero"]:
                    stats["all_zero_count"] += 1

                if (
                    description["is_none"]
                    or description["all_nan"]
                    or description["all_non_finite"]
                    or description["all_zero"]
                ):
                    sample_flags.append(
                        f"sample[{sample_index}] {group_name}.{variable_name}"
                        f" none={description['is_none']}"
                        f" all_nan={description['all_nan']}"
                        f" all_non_finite={description['all_non_finite']}"
                        f" all_zero={description['all_zero']}"
                        f" shape={description['shape']}"
                    )

    print("variable_data_flags:")
    for variable_name in sorted(variable_stats):
        stats = variable_stats[variable_name]
        print(
            f"  - {variable_name}:"
            f" samples_seen={stats['samples_seen']}"
            f" none={stats['none_count']}"
            f" all_nan={stats['all_nan_count']}"
            f" all_non_finite={stats['all_non_finite_count']}"
            f" all_zero={stats['all_zero_count']}"
        )

    if sample_flags:
        print("flagged_samples:")
        for line in sample_flags[:50]:
            print(f"  - {line}")
        if len(sample_flags) > 50:
            print(f"  - ... and {len(sample_flags) - 50} more flagged sample entries")
    else:
        print("flagged_samples: none")


def main() -> None:
    args = parse_args()
    root = Path(args.query_dir)

    if not root.exists():
        raise SystemExit(f"Query directory does not exist: {root}")
    if not root.is_dir():
        raise SystemExit(f"Query path is not a directory: {root}")

    query_files = iter_query_files(root)
    if not query_files:
        print(f"No supported query files found under {root}")
        print("Supported formats: .json, .jsonl, .csv, .parquet")
        return

    print(f"Found {len(query_files)} query files under {root}")

    total_records = 0
    for path in query_files:
        try:
            records, metadata = read_records(path)
        except Exception as exc:
            print(f"\n=== {path} ===")
            print(f"failed_to_read: {exc}")
            continue

        total_records += len(records)
        print_file_report(path, records, metadata, args.limit)

        if args.check_data:
            try:
                inspect_query_file_data(path, config_path=args.config, max_samples=args.max_samples)
            except Exception as exc:
                print(f"data_check_failed: {exc}")

    print(f"\nTotal query records across all readable files: {total_records}")


if __name__ == "__main__":
    main()
