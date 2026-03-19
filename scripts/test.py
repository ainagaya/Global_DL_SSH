#!/usr/bin/env python3
from __future__ import annotations

import argparse

from gdlssh.validators import dump_first_tfrecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump the first TFRecord example for quick inspection.")
    parser.add_argument("path")
    args = parser.parse_args()
    print(dump_first_tfrecord(args.path))


if __name__ == "__main__":
    main()
