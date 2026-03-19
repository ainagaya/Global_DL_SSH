from __future__ import annotations

from pathlib import Path

import tensorflow as tf

from .tfrecords import SequenceSchema, TfrecordTorchDataset


def validate_tfrecords(paths: list[str | Path], schema: SequenceSchema) -> list[tuple[str, bool, str]]:
    dataset = TfrecordTorchDataset(paths, schema)
    results: list[tuple[str, bool, str]] = []
    for idx, path in enumerate(dataset.tfrecord_paths):
        try:
            x, y = dataset[idx]
            results.append((path, True, f"input={tuple(x.shape)} output={tuple(y.shape)}"))
        except StopIteration:
            results.append((path, False, "file opened but contains no records"))
        except Exception as exc:
            results.append((path, False, f"{type(exc).__name__}: {exc}"))
    return results


def dump_first_tfrecord(path: str | Path) -> tf.train.Example:
    raw_ds = tf.data.TFRecordDataset(str(Path(path).expanduser()))
    raw_record = next(iter(raw_ds))
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    return example
