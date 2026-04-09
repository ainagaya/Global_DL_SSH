#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

import requests

sys.path.append("src")

from src.config_utils import ensure_dir, load_config
from src.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)

DEFAULT_METADATA_FILES = (
    ".gitattributes",
    "README.md",
    "COLLECTION.json",
    "level0.parquet",
    "level1.parquet",
    "level2.parquet",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the OceanTACO files required by the configured queries into a local dataset mirror."
    )
    parser.add_argument(
        "--config",
        default="configs/oceantaco.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=None,
        help="Optional split names to download. Defaults to all configured splits.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the files that would be downloaded without downloading them.",
    )
    return parser.parse_args()


def _runtime_helpers():
    from src.oceantaco import (
        _import_oceantaco,
        _looks_like_remote_taco_path,
        build_queries,
        get_configured_local_oceantaco_path,
        get_oceantaco_repo_id,
        get_oceantaco_revision,
    )

    return (
        _import_oceantaco,
        _looks_like_remote_taco_path,
        build_queries,
        get_configured_local_oceantaco_path,
        get_oceantaco_repo_id,
        get_oceantaco_revision,
    )


def determine_download_splits(config: dict, requested_splits: list[str] | None) -> list[str]:
    configured_splits = list(config.get("splits", {}).keys())
    if requested_splits:
        unknown = [split for split in requested_splits if split not in config.get("splits", {})]
        if unknown:
            raise KeyError(f"Unknown splits requested: {unknown}. Available splits: {configured_splits}")
        return requested_splits
    return configured_splits


def select_remote_taco_path(config: dict, hf_default_url: str) -> str:
    _, looks_like_remote_taco_path, _, _, _, _ = _runtime_helpers()
    configured_taco_path = config.get("oceantaco", {}).get("taco_path")
    if configured_taco_path and looks_like_remote_taco_path(configured_taco_path):
        return str(configured_taco_path)
    return hf_default_url


def extract_repo_relative_path(vsi_path: str) -> str | None:
    path_value = vsi_path
    if path_value.startswith("/vsisubfile/"):
        _, _, remainder = path_value.partition(",")
        path_value = remainder
    if path_value.startswith("/vsicurl/"):
        path_value = path_value.removeprefix("/vsicurl/")

    parsed = urlparse(path_value)
    if parsed.scheme in {"http", "https"}:
        parts = [part for part in parsed.path.split("/") if part]
        if "resolve" in parts:
            resolve_index = parts.index("resolve")
            if resolve_index + 2 < len(parts):
                return unquote("/".join(parts[resolve_index + 2 :]))
        return None

    parts = [part for part in Path(path_value).parts if part not in ("/", "")]
    if "DATA" in parts:
        data_index = parts.index("DATA")
        return "/".join(parts[data_index:])
    return None


def collect_required_data_files(file_indices: Iterable, vsi_column_name: str) -> list[str]:
    repo_paths = set()
    for file_df in file_indices:
        for vsi_path in file_df[vsi_column_name].tolist():
            relative_path = extract_repo_relative_path(str(vsi_path))
            if relative_path is not None:
                repo_paths.add(relative_path)
    return sorted(repo_paths)


def list_repo_files(repo_id: str, revision: str) -> list[str]:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        LOGGER.warning(
            "huggingface_hub is not installed. Falling back to a conservative metadata file list."
        )
        return list(DEFAULT_METADATA_FILES)

    api = HfApi()
    return sorted(api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision))


def determine_metadata_files(repo_id: str, revision: str, data_files: list[str]) -> list[str]:
    try:
        repo_files = list_repo_files(repo_id, revision)
    except Exception as exc:
        LOGGER.warning(
            "Could not list OceanTACO repo files for %s@%s (%s). Falling back to default metadata file list.",
            repo_id,
            revision,
            exc,
        )
        return list(DEFAULT_METADATA_FILES)

    metadata_files = [path for path in repo_files if not path.startswith("DATA/")]
    # If listing failed silently and returned only data files, keep at least the
    # known metadata essentials.
    if not metadata_files:
        metadata_files = list(DEFAULT_METADATA_FILES)
    return sorted(set(metadata_files))


def download_repo_file(repo_id: str, revision: str, repo_path: str, destination_root: Path, force: bool) -> Path:
    target_path = destination_root / repo_path
    if target_path.exists() and not force:
        return target_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{repo_path}"
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        with target_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        return target_path

    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=repo_path,
        local_dir=str(destination_root),
        local_dir_use_symlinks=False,
        force_download=force,
    )
    return Path(downloaded_path)


def write_download_manifest(
    destination_root: Path,
    *,
    config: dict,
    splits: list[str],
    repo_id: str,
    revision: str,
    downloaded_files: list[str],
) -> Path:
    manifest_path = destination_root / "download_manifest.json"
    payload = {
        "config_path": config["__config_path__"],
        "repo_id": repo_id,
        "revision": revision,
        "splits": splits,
        "downloaded_files": downloaded_files,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config)

    (
        import_oceantaco,
        _looks_like_remote_taco_path,
        build_queries,
        get_configured_local_oceantaco_path,
        get_oceantaco_repo_id,
        get_oceantaco_revision,
    ) = _runtime_helpers()
    OceanTACODataset, _, _, _, HF_DEFAULT_URL, ocean_dataset_module = import_oceantaco()

    destination_root = get_configured_local_oceantaco_path(config)
    if destination_root is None:
        raise RuntimeError(
            "Config must define a local OceanTACO destination under oceantaco.download_path or a local oceantaco.taco_path."
        )
    destination_root = ensure_dir(destination_root, config)

    split_names = determine_download_splits(config, args.splits)
    repo_id = get_oceantaco_repo_id(config)
    revision = get_oceantaco_revision(config)
    remote_taco_path = select_remote_taco_path(config, HF_DEFAULT_URL)

    all_queries = []
    for split_name in split_names:
        split_queries = build_queries(config, split_name)
        all_queries.extend(split_queries)
    if not all_queries:
        raise RuntimeError("No queries were generated from the selected config splits.")

    data_cfg = config["data"]
    grid_cfg = data_cfg["grid"]
    dataset = OceanTACODataset(
        taco_path=remote_taco_path,
        queries=all_queries,
        input_variables=[source_cfg["key"] for source_cfg in data_cfg["inputs"]],
        target_variables=[source_cfg["key"] for source_cfg in data_cfg["targets"]],
        target_resolution=data_cfg.get("target_resolution"),
        temporal_agg=data_cfg.get("temporal_agg", "stack"),
        default_patch_size=(int(grid_cfg["height"]), int(grid_cfg["width"])),
    )

    required_data_files = collect_required_data_files(dataset._file_index, ocean_dataset_module.COL_VSI)
    metadata_files = determine_metadata_files(repo_id, revision, required_data_files)
    repo_files = sorted(set(metadata_files) | set(required_data_files))

    LOGGER.info(
        "OceanTACO local mirror target=%s repo=%s revision=%s splits=%s queries=%s files=%s",
        destination_root,
        repo_id,
        revision,
        split_names,
        len(all_queries),
        len(repo_files),
    )

    if args.dry_run:
        for repo_path in repo_files:
            print(repo_path)
        return

    downloaded_files = []
    for index, repo_path in enumerate(repo_files, start=1):
        download_repo_file(repo_id, revision, repo_path, destination_root, force=args.force)
        downloaded_files.append(repo_path)
        if index == 1 or index == len(repo_files) or index % 25 == 0:
            LOGGER.info("Downloaded %s/%s OceanTACO files", index, len(repo_files))

    manifest_path = write_download_manifest(
        destination_root,
        config=config,
        splits=split_names,
        repo_id=repo_id,
        revision=revision,
        downloaded_files=downloaded_files,
    )
    LOGGER.info("Wrote OceanTACO download manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
