#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt

import tacoreader

from gdlssh.config import load_config
from gdlssh.logging_utils import configure_logging
from gdlssh.preprocess import RegionGeometry, split_dates, write_training_tfrecords


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training TFRecords for Global_DL_SSH.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--region-name", default="NORTH_PACIFIC_EAST")
    parser.add_argument("--mode", choices=["train", "val"], default="train")
    parser.add_argument("--n-batches", type=int, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--buffer-days", type=int, default=30)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    dataset = tacoreader.load(args.dataset_url)
    geometry = RegionGeometry(
        grid_size=config["geometry"]["grid_size"],
        domain_x_m=config["geometry"]["domain_x_m"],
        domain_y_m=config["geometry"]["domain_y_m"],
        lat_min=config["geometry"]["lat_min"],
        lat_max=config["geometry"]["lat_max"],
        lon_min=config["geometry"]["lon_min"],
        lon_max=config["geometry"]["lon_max"],
        region_count=config["geometry"]["region_count"],
    )
    date_splits = split_dates(dt.date.fromisoformat(args.start_date), dt.date.fromisoformat(args.end_date), args.train_fraction, args.val_fraction, args.buffer_days)
    write_training_tfrecords(
        dataset=dataset,
        output_dir=args.output_dir,
        batch_size=config["sequence"]["batch_size"],
        n_batches=args.n_batches,
        dates=date_splits[args.mode],
        geometry=geometry,
        n_t=config["sequence"]["n_t"],
        n_obs_max=config["sequence"]["n_obs_max"],
        region_name=args.region_name,
        download_workers=config["runtime"]["download_workers"],
        random_seed=config["runtime"]["random_seed"],
    )


if __name__ == "__main__":
    main()
