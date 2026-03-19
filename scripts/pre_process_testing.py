#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt

import tacoreader

from gdlssh.config import load_config
from gdlssh.logging_utils import configure_logging
from gdlssh.preprocess import RegionGeometry, write_testing_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate testing arrays for regional SSH prediction.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-url", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--num-days", type=int, required=True)
    parser.add_argument("--region-name", default="NORTH_PACIFIC_EAST")
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
    write_testing_arrays(
        dataset=dataset,
        output_dir=args.output_dir,
        geometry=geometry,
        n_t=config["sequence"]["n_t"],
        start_date=dt.date.fromisoformat(args.start_date),
        num_days=args.num_days,
        region_name=args.region_name,
        download_workers=config["runtime"]["download_workers"],
    )


if __name__ == "__main__":
    main()
