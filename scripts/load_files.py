#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from gdlssh.config import load_config
from gdlssh.logging_utils import configure_logging
from gdlssh.tfrecords import SequenceSchema
from gdlssh.validators import validate_tfrecords


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate TFRecord files using the shared schema.")
    parser.add_argument("--config", required=True)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    schema = SequenceSchema(
        batch_size=config["sequence"]["batch_size"],
        n_t=config["sequence"]["n_t"],
        grid_size=config["geometry"]["grid_size"],
        n_obs_max=config["sequence"]["n_obs_max"],
        domain_x_m=config["geometry"]["domain_x_m"],
        domain_y_m=config["geometry"]["domain_y_m"],
        mean_ssh=config["normalization"]["mean_ssh"],
        std_ssh=config["normalization"]["std_ssh"],
        mean_sst=config["normalization"]["mean_sst"],
        std_sst=config["normalization"]["std_sst"],
    )
    for path, ok, message in validate_tfrecords(args.paths, schema):
        tag = "OK" if ok else "FAIL"
        print(f"[{tag}] {Path(path)} -> {message}")


if __name__ == "__main__":
    main()
