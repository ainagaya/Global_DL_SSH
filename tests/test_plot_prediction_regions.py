from __future__ import annotations

from pathlib import Path

import numpy as np

from plot_prediction_regions import (
    PredictionRecord,
    compute_log_color_limits,
    compute_panel_limits,
    compute_sst_limits,
    default_output_path,
    mask_sst_padding,
    masked_prediction_squared_error,
    prepare_log_scaled_field,
    plot_date_records,
)


def make_record(
    *,
    name: str,
    prediction: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    sst: np.ndarray | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        path=Path(f"{name}.npz"),
        bbox=(0.0, 1.0, 2.0, 3.0),
        target_date="2025-05-02",
        prediction=prediction,
        target=target,
        source_name=name,
        source=source,
        sst=sst,
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


def test_compute_sst_limits_ignores_zero_padding():
    field = np.array([[0.0, 21.0, 22.5], [np.nan, 0.0, 24.0]], dtype=np.float32)

    assert compute_sst_limits(field) == (21.0, 24.0)

    masked = mask_sst_padding(field)
    assert np.ma.is_masked(masked[0, 0])
    assert np.ma.is_masked(masked[1, 0])
    assert float(masked[0, 1]) == 21.0


def test_compute_panel_limits_ignores_sst_source_zero_padding():
    records = [
        make_record(
            name="input_l4_sst",
            prediction=np.array([[1.0, 2.0]], dtype=np.float32),
            target=np.array([[0.5, 3.0]], dtype=np.float32),
            source=np.array([[0.0, 19.0]], dtype=np.float32),
        ),
        make_record(
            name="input_l4_sst",
            prediction=np.array([[-2.0, 4.0]], dtype=np.float32),
            target=np.array([[5.0, 6.0]], dtype=np.float32),
            source=np.array([[23.0, 0.0]], dtype=np.float32),
        ),
    ]

    _, source_limits = compute_panel_limits(records)

    assert source_limits["input_l4_sst"] == (19.0, 23.0)


def test_masked_prediction_squared_error_discards_missing_swot_targets():
    prediction = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    target = np.array([[1.0, 0.0, np.nan, -2.0]], dtype=np.float32)

    squared_error = masked_prediction_squared_error(prediction, target)

    assert np.allclose(squared_error[0, [0, 3]], np.array([81.0, 1764.0], dtype=np.float32))
    assert np.isnan(squared_error[0, 1])
    assert np.isnan(squared_error[0, 2])


def test_log_color_helpers_floor_exact_zero_values():
    field = np.array([[0.0, 1.0e-4, 2.0], [np.nan, 0.0, 4.0]], dtype=np.float32)

    vmin, vmax = compute_log_color_limits(field)
    display = prepare_log_scaled_field(field, vmin)

    assert vmin == float(np.float32(1.0e-4))
    assert vmax == 4.0
    assert float(display[0, 0]) == vmin
    assert float(display[1, 1]) == vmin
    assert np.ma.is_masked(display[1, 0])


def test_default_output_path_honors_output_dir(tmp_path):
    output_path = default_output_path(tmp_path / "predictions", "2025-05-02", str(tmp_path / "analysis"))

    assert output_path == tmp_path / "analysis" / "prediction_regions_2025-05-02.png"
    assert output_path.parent.exists()


def test_plot_date_records_supports_separate_sst_panel(tmp_path):
    record = make_record(
        name="input_l3_ssh",
        prediction=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        target=np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        source=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        sst=np.array([[0.0, 21.0], [22.0, 0.0]], dtype=np.float32),
    )

    output_path = plot_date_records(
        [record],
        tmp_path / "regions.png",
        date_text="2025-05-02",
        context_pad_lon=0.0,
        context_pad_lat=0.0,
        dpi=80,
    )

    assert output_path.exists()
