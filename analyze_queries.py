#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# This script is intentionally verbose and heavily commented because its main
# purpose is debugging. When query-driven dataset creation returns empty data,
# we first want to answer:
#
# 1. Were query files actually written?
# 2. How many queries do they contain?
# 3. What bbox and time ranges do they cover?
# 4. Are the bboxes malformed or overly large / tiny?
# 5. Are there duplicates or suspiciously identical queries?
# 6. Do different files / splits cover what we expect?
#
# The script is format-tolerant on purpose. OceanTACO query exports may be JSON,
# JSONL, CSV, Parquet, or directory-based structures depending on version and
# local tooling. Rather than assuming one exact schema, we inspect common
# patterns and summarize what we can.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect saved query files and summarize their coverage.")
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
    return parser.parse_args()


def iter_query_files(root: Path) -> list[Path]:
    # We scan recursively because queries may be saved as one file per split,
    # one file per batch, or a directory tree created by a library helper.
    file_paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv", ".parquet"}:
            file_paths.append(path)
    return file_paths


def read_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # We normalize everything into a list[dict]. Metadata is returned separately
    # so we can report split names or export-time information if present.
    suffix = path.suffix.lower()
    metadata: dict[str, Any] = {}

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            # Common cases:
            # - {"queries": [...], "metadata": {...}}
            # - {"metadata": {...}, "items": [...]}
            # - a single query dict
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
    # Query objects should normally serialize as dictionaries. If something else
    # slips through, we still wrap it so the caller gets a stable shape.
    if isinstance(item, dict):
        return item
    return {"value": item}


def flatten_keys(record: dict[str, Any], prefix: str = "") -> Iterable[str]:
    # Key-frequency summaries are useful when we do not know the exact export
    # schema in advance.
    for key, value in record.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        yield full_key
        if isinstance(value, dict):
            yield from flatten_keys(value, full_key)


def extract_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    # We support several common patterns:
    # - {"bbox": [lon_min, lon_max, lat_min, lat_max]}
    # - {"bbox_constraint": [...]}
    # - {"lon_min": ..., "lon_max": ..., "lat_min": ..., "lat_max": ...}
    # - nested dicts with those names
    candidates = [
        record.get("bbox"),
        record.get("bbox_constraint"),
    ]

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
    # Again, several schema variants are accepted because debugging data
    # pipelines is easier when the tool is forgiving.
    candidates = [
        record.get("time_range"),
        record.get("date_range"),
    ]
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

    # Show the most common keys to help understand the query schema quickly.
    key_counts: Counter[str] = summary["key_counts"]
    if key_counts:
        print("top_keys:")
        for key, count in key_counts.most_common(12):
            print(f"  - {key}: {count}")

    print("sample_records:")
    for record in records[:limit]:
        print(f"  - {short_json(record, max_length=240)}")


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

    print(f"\nTotal query records across all readable files: {total_records}")


if __name__ == "__main__":
    main()
