#!/usr/bin/env python3
from __future__ import annotations

import argparse

from gdlssh.config import load_config
from gdlssh.importing import import_from_string
from gdlssh.logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge regional predictions into NetCDF maps.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    merge_fn = import_from_string(config["legacy"]["merge_function"])
    for pred_date in config["merge"]["prediction_dates"]:
        merge_fn(
            pred_dir=config["paths"]["refactored_preds_dir"],
            pred_file_pattern=config["merge"]["pred_file_pattern"],
            pred_date=pred_date,
            output_nc_dir=config["paths"]["merged_output_dir"],
            mask_filename=config["merge"]["mask_filename"],
            dist_filename=config["merge"]["dist_filename"],
            mdt_filename=config["merge"]["mdt_filename"],
            network_name=config["merge"]["output_network_name"],
            available_regions=config["merge"]["available_regions"],
            L=config["merge"]["L_m"],
            crop_pixels=config["merge"]["crop_pixels"],
            dx=config["geometry"]["dx_m"],
            with_grads=config["merge"]["with_grads"],
            mask_coast_dist=config["merge"]["mask_coast_dist_km"],
            lon_min=config["geometry"]["lon_min"],
            lon_max=config["geometry"]["lon_max"],
            lat_min=config["geometry"]["lat_min"],
            lat_max=config["geometry"]["lat_max"],
            res=config["merge"]["res_deg"],
            progress=False,
        )


if __name__ == "__main__":
    main()
