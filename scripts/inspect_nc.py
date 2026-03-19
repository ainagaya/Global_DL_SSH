#!/usr/bin/env python3
from __future__ import annotations

import argparse

from gdlssh.inspect import inspect_netcdf
from gdlssh.logging_utils import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a NetCDF file and save diagnostic plots.")
    parser.add_argument("path")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    configure_logging(args.log_level)
    inspect_netcdf(args.path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
