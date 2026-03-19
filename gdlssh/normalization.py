from __future__ import annotations

import numpy as np
import tensorflow as tf

EPSILON = 1e-12


def normalize_nonzero_np(array: np.ndarray, mean: float, std: float, invalid_below: float | None = None) -> np.ndarray:
    data = np.array(array, copy=True, dtype=np.float32)
    if invalid_below is not None:
        data[data < invalid_below] = 0.0
    mask = data != 0
    data[mask] = (data[mask] - mean) / max(std, EPSILON)
    return data


def normalize_nonzero_tf(tensor: tf.Tensor, mean: float, std: float) -> tf.Tensor:
    mask = tf.not_equal(tensor, 0)
    values = tf.boolean_mask(tensor, mask)
    values = (values - mean) / tf.maximum(std, EPSILON)
    return tf.tensor_scatter_nd_update(tensor, tf.where(mask), values)


def rescale_x_tf(tensor: tf.Tensor, domain_x_m: float, grid_size: int) -> tf.Tensor:
    mask = tf.not_equal(tensor, 0)
    values = tf.boolean_mask(tensor, mask)
    values = (values + 0.5 * domain_x_m) / (domain_x_m / (grid_size - 1))
    return tf.tensor_scatter_nd_update(tensor, tf.where(mask), values)


def rescale_y_tf(tensor: tf.Tensor, domain_y_m: float, grid_size: int) -> tf.Tensor:
    mask = tf.not_equal(tensor, 0)
    values = tf.boolean_mask(tensor, mask)
    values = (-values + 0.5 * domain_y_m) / (domain_y_m / (grid_size - 1))
    return tf.tensor_scatter_nd_update(tensor, tf.where(mask), values)
