from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset

from .normalization import normalize_nonzero_tf, rescale_x_tf, rescale_y_tf


@dataclass(slots=True)
class SequenceSchema:
    batch_size: int
    n_t: int
    grid_size: int
    n_obs_max: int
    domain_x_m: float
    domain_y_m: float
    mean_ssh: float
    std_ssh: float
    mean_sst: float
    std_sst: float
    include_extra_timestep: bool = True


def serialize_example(input_array: np.ndarray, output_array: np.ndarray) -> bytes:
    feature = {
        "input": tf.train.Feature(float_list=tf.train.FloatList(value=input_array.flatten())),
        "output": tf.train.Feature(float_list=tf.train.FloatList(value=output_array.flatten())),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def parse_example_sst(serialized_example: tf.Tensor, schema: SequenceSchema) -> tuple[tf.Tensor, tf.Tensor]:
    input_steps = schema.n_t + 1 if schema.include_extra_timestep else schema.n_t
    feature_description = {
        "input": tf.io.FixedLenFeature(int(schema.batch_size * input_steps * schema.grid_size * schema.grid_size * 2), tf.float32),
        "output": tf.io.FixedLenFeature(int(schema.batch_size * schema.n_t * schema.n_obs_max * 3), tf.float32),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    input_data = tf.reshape(example["input"], [schema.batch_size, input_steps, schema.grid_size, schema.grid_size, 2])
    input_data = input_data[:, : schema.n_t, :, :, :]

    ssh = normalize_nonzero_tf(input_data[:, :, :, :, 0], schema.mean_ssh, schema.std_ssh)
    sst = normalize_nonzero_tf(input_data[:, :, :, :, 1], schema.mean_sst, schema.std_sst)
    invar = tf.transpose(tf.stack((ssh, sst), axis=-1), perm=[0, 1, 4, 2, 3])

    output_data = tf.reshape(example["output"], [schema.batch_size, schema.n_t, schema.n_obs_max, 3])
    x = rescale_x_tf(output_data[:, :, :, 0], schema.domain_x_m, schema.grid_size)
    y = rescale_y_tf(output_data[:, :, :, 1], schema.domain_y_m, schema.grid_size)
    sla = normalize_nonzero_tf(output_data[:, :, :, 2], schema.mean_ssh, schema.std_ssh)
    outvar = tf.stack((x, y, sla), axis=-1)
    return invar, outvar


class TfrecordTorchDataset(Dataset):
    def __init__(self, tfrecord_paths: Iterable[str | Path], schema: SequenceSchema) -> None:
        self.tfrecord_paths = [str(Path(path).expanduser()) for path in tfrecord_paths]
        self.schema = schema

    def __len__(self) -> int:
        return len(self.tfrecord_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        dataset = tf.data.TFRecordDataset(self.tfrecord_paths[index])
        parsed = dataset.map(lambda record: parse_example_sst(record, self.schema))
        invar, outvar = next(iter(parsed))
        return torch.from_numpy(invar.numpy()), torch.from_numpy(outvar.numpy())
