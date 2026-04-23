from __future__ import annotations

from pathlib import Path

import numpy as np

from plot_prediction_regions import PredictionRecord, compute_panel_limits, masked_prediction_squared_error


def make_record(*, name: str, prediction: np.ndarray, target: np.ndarray, source: np.ndarray) -> PredictionRecord:
    return PredictionRecord(
        path=Path(f"{name}.npz"),
        bbox=(0.0, 1.0, 2.0, 3.0),
        target_date="2025-05-02",
        prediction=prediction,
        target=target,
        source_name=name,
        source=source,
    )


def test_compute_panel_limits_uses_shared_ssh_scale_across_records():
    records = [
        make_record(
            name="input_l3_ssh",
            prediction=np.array([[1.0, 2.0]], dtype=np.float32),
            target=np.array([[0.5, 3.0]], dtype=np.float32),
            source=np.array([[10.0, 12.0]], dtype=np.float32),
        ),
        make_record(
            name="input_l3_ssh",
            prediction=np.array([[-2.0, 4.0]], dtype=np.float32),
            target=np.array([[5.0, 6.0]], dtype=np.float32),
            source=np.array([[20.0, 22.0]], dtype=np.float32),
        ),
    ]

    value_limits, source_limits = compute_panel_limits(records)

    assert value_limits == (-2.0, 6.0)
    assert source_limits["input_l3_ssh"] == value_limits


def test_compute_panel_limits_keeps_non_ssh_source_scale_shared_by_source_name():
    records = [
        make_record(
            name="input_l4_sst",
            prediction=np.array([[1.0, 2.0]], dtype=np.float32),
            target=np.array([[0.5, 3.0]], dtype=np.float32),
            source=np.array([[15.0, 17.0]], dtype=np.float32),
        ),
        make_record(
            name="input_l4_sst",
            prediction=np.array([[-2.0, 4.0]], dtype=np.float32),
            target=np.array([[5.0, 6.0]], dtype=np.float32),
            source=np.array([[10.0, 25.0]], dtype=np.float32),
        ),
    ]

    value_limits, source_limits = compute_panel_limits(records)

    assert value_limits == (-2.0, 6.0)
    assert source_limits["input_l4_sst"] == (10.0, 25.0)


def test_masked_prediction_squared_error_discards_missing_swot_targets():
    prediction = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    target = np.array([[1.0, 0.0, np.nan, -2.0]], dtype=np.float32)

    squared_error = masked_prediction_squared_error(prediction, target)

    assert np.allclose(squared_error[0, [0, 3]], np.array([81.0, 1764.0], dtype=np.float32))
    assert np.isnan(squared_error[0, 1])
    assert np.isnan(squared_error[0, 2])
