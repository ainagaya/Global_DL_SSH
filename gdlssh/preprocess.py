from __future__ import annotations

import datetime as dt
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
import xarray as xr
from scipy import stats

from .fs import ensure_dir
from .io_utils import download_all
from .tfrecords import serialize_example

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RegionGeometry:
    grid_size: int
    domain_x_m: float
    domain_y_m: float
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    region_count: int

    def center_for_region(self, region: int) -> tuple[float, float]:
        lat = self.lat_min + (self.lat_max - self.lat_min) * region / self.region_count
        lon = self.lon_min + (self.lon_max - self.lon_min) * region / self.region_count
        return lat, lon

    def bounds_for_region(self, region: int) -> tuple[float, float, float, float]:
        lat_center, lon_center = self.center_for_region(region)
        lat_half_extent = self.domain_y_m / 2 / 111e3
        lon_half_extent = self.domain_x_m / 2 / 111e3
        return (
            lat_center - lat_half_extent,
            lat_center + lat_half_extent,
            lon_center - lon_half_extent,
            lon_center + lon_half_extent,
        )


def bin_dataarray_onto_grid(data: xr.DataArray, grid_size: int) -> np.ndarray:
    lats = data.coords["lat"].to_numpy()
    lons = data.coords["lon"].to_numpy()
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    values = data.to_numpy().flatten()
    if np.isnan(values).all():
        return np.zeros((grid_size, grid_size), dtype=np.float32)
    gridded, _, _, _ = stats.binned_statistic_2d(
        lon_grid.flatten(),
        lat_grid.flatten(),
        values,
        statistic="mean",
        bins=grid_size,
    )
    gridded = np.rot90(gridded)
    gridded[np.isnan(gridded)] = 0.0
    return gridded.astype(np.float32)


def output_tracks_from_dataarray(data: xr.DataArray) -> np.ndarray:
    lats = data.coords["lat"].to_numpy()
    lons = data.coords["lon"].to_numpy()
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    tracks = np.stack((lon_grid.flatten(), lat_grid.flatten(), data.to_numpy().flatten()), axis=-1)
    tracks[np.isnan(tracks)] = 0.0
    return tracks.astype(np.float32)


def pad_tracks(tracks: np.ndarray, max_observations: int) -> np.ndarray:
    padded = np.zeros((max_observations, 3), dtype=np.float32)
    length = min(max_observations, tracks.shape[0])
    padded[:length, :] = tracks[:length, :]
    return padded


def split_dates(start_date: dt.date, end_date: dt.date, train_fraction: float, val_fraction: float, buffer_days: int) -> dict[str, list[dt.date]]:
    total_days = (end_date - start_date).days + 1
    all_dates = [start_date + dt.timedelta(days=i) for i in range(total_days)]
    train_end = int(total_days * train_fraction)
    val_end = int(total_days * (train_fraction + val_fraction))
    train_dates = all_dates[:train_end]
    val_dates = all_dates[train_end:val_end]
    test_dates = all_dates[val_end:]
    filtered_train = []
    for candidate in train_dates:
        near_val = any(abs((candidate - date).days) < buffer_days for date in val_dates)
        near_test = any(abs((candidate - date).days) < buffer_days for date in test_dates)
        if not (near_val or near_test):
            filtered_train.append(candidate)
    return {"train": filtered_train, "val": val_dates, "test": test_dates}


def build_oceantaco_query(date_value: dt.date, region_name: str) -> str:
    return f'''
        SELECT
            "l2:id" AS id,
            REPLACE("l2:internal:gdal_vsi", '/vsicurl/', '') AS url
        FROM l2
        WHERE "l0:stac:time_start" LIKE '{date_value}%'
        AND "l1:id" LIKE '%{region_name}%'
        AND (
            "l2:id" LIKE '%l3_swot.nc'
            OR "l2:id" LIKE '%l3_sst.nc'
            OR "l2:id" LIKE '%l3_ssh.nc'
        )
    '''


def build_training_sample(dataset: object, region: int, mid_date: dt.date, geometry: RegionGeometry, n_t: int, n_obs_max: int, download_workers: int, region_name: str) -> tuple[np.ndarray, np.ndarray]:
    input_array = np.zeros((n_t + 1, geometry.grid_size, geometry.grid_size, 2), dtype=np.float32)
    output_array = np.zeros((n_t, n_obs_max, 3), dtype=np.float32)
    lat_min, lat_max, lon_min, lon_max = geometry.bounds_for_region(region)
    for t_index in range(n_t):
        date_loop = mid_date - dt.timedelta(days=int(n_t / 2) - t_index)
        query = build_oceantaco_query(date_loop, region_name)
        table = dataset.sql(query)
        records = [dict(row) for _, row in table.iterrows()]
        remote = download_all(records, max_workers=download_workers)
        sst = remote.get("l3_sst.nc")
        swot = remote.get("l3_swot.nc")
        ssh = remote.get("l3_ssh.nc")
        if sst is not None and "adjusted_sea_surface_temperature" in sst:
            sst_da = sst["adjusted_sea_surface_temperature"].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            input_array[t_index, :, :, 0] = bin_dataarray_onto_grid(sst_da, geometry.grid_size)
        source_da = None
        if swot is not None and "ssha_filtered" in swot:
            source_da = swot["ssha_filtered"].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        elif ssh is not None and "sla_filtered" in ssh:
            source_da = ssh["sla_filtered"].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        if source_da is not None:
            input_array[t_index, :, :, 1] = bin_dataarray_onto_grid(source_da, geometry.grid_size)
            output_array[t_index, :, :] = pad_tracks(output_tracks_from_dataarray(source_da), n_obs_max)
    return input_array, output_array


def write_training_tfrecords(dataset: object, output_dir: str | Path, batch_size: int, n_batches: int, dates: list[dt.date], geometry: RegionGeometry, n_t: int, n_obs_max: int, region_name: str, download_workers: int = 4, random_seed: int = 42) -> None:
    rng = random.Random(random_seed)
    ensure_dir(output_dir)
    available_regions = list(range(geometry.region_count))
    for batch_idx in range(n_batches):
        input_batch = np.zeros((batch_size, n_t + 1, geometry.grid_size, geometry.grid_size, 2), dtype=np.float32)
        output_batch = np.zeros((batch_size, n_t, n_obs_max, 3), dtype=np.float32)
        for sample_idx in range(batch_size):
            region = rng.choice(available_regions)
            mid_date = rng.choice(dates)
            sample_input, sample_output = build_training_sample(dataset, region, mid_date, geometry, n_t, n_obs_max, download_workers, region_name)
            input_batch[sample_idx] = sample_input
            output_batch[sample_idx] = sample_output
            LOGGER.info("Prepared batch=%s sample=%s region=%s date=%s", batch_idx, sample_idx, region, mid_date)
        record_path = Path(output_dir) / f"batch_{batch_idx}.tfrecord"
        with tf.io.TFRecordWriter(str(record_path)) as writer:
            writer.write(serialize_example(input_batch, output_batch))
        LOGGER.info("Saved %s", record_path)


def write_testing_arrays(dataset: object, output_dir: str | Path, geometry: RegionGeometry, n_t: int, start_date: dt.date, num_days: int, region_name: str, download_workers: int = 4) -> None:
    ensure_dir(output_dir)
    total_steps = num_days + n_t
    for region in range(geometry.region_count):
        region_array = np.zeros((total_steps, geometry.grid_size, geometry.grid_size, 2), dtype=np.float32)
        lat_min, lat_max, lon_min, lon_max = geometry.bounds_for_region(region)
        for offset in range(total_steps):
            date_loop = start_date + dt.timedelta(days=offset)
            query = build_oceantaco_query(date_loop, region_name)
            table = dataset.sql(query)
            records = [dict(row) for _, row in table.iterrows()]
            remote = download_all(records, max_workers=download_workers)
            sst = remote.get("l3_sst.nc")
            swot = remote.get("l3_swot.nc")
            ssh = remote.get("l3_ssh.nc")
            if sst is not None and "adjusted_sea_surface_temperature" in sst:
                sst_da = sst["adjusted_sea_surface_temperature"].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                region_array[offset, :, :, 0] = bin_dataarray_onto_grid(sst_da, geometry.grid_size)
            if swot is not None and "ssha_filtered" in swot:
                ssh_da = swot["ssha_filtered"].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                region_array[offset, :, :, 1] = bin_dataarray_onto_grid(ssh_da, geometry.grid_size)
            elif ssh is not None and "sla_filtered" in ssh:
                ssh_da = ssh["sla_filtered"].sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
                region_array[offset, :, :, 1] = bin_dataarray_onto_grid(ssh_da, geometry.grid_size)
            LOGGER.info("Prepared testing region=%s date=%s", region, date_loop)
        save_path = Path(output_dir) / f"input_data_region{region}.npy"
        np.save(save_path, region_array)
        LOGGER.info("Saved %s", save_path)
