from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import requests
import xarray as xr

LOGGER = logging.getLogger(__name__)


def list_files(root: str | Path, suffix: str) -> list[Path]:
    root_path = Path(root).expanduser()
    return sorted(path for path in root_path.rglob(f"*{suffix}") if path.is_file())


def fetch_nc(record: dict[str, str], timeout_s: int = 120) -> tuple[str, xr.Dataset]:
    filename = record["id"].split("/")[-1]
    response = requests.get(record["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout_s)
    response.raise_for_status()
    return filename, xr.open_dataset(io.BytesIO(response.content), engine="h5netcdf")


def download_all(records: Iterable[dict[str, str]], max_workers: int = 4) -> dict[str, xr.Dataset]:
    datasets: dict[str, xr.Dataset] = {}
    rows = list(records)
    if not rows:
        return datasets
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_nc, row): row for row in rows}
        for future in as_completed(futures):
            filename, dataset = future.result()
            datasets[filename] = dataset
            LOGGER.debug("Downloaded %s", filename)
    return datasets
