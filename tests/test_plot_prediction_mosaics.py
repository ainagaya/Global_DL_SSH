from __future__ import annotations

import numpy as np

from plot_prediction_mosaics import (
    compute_mosaic_extent,
    compute_mosaic_limits,
    load_prediction_records,
    plot_date_mosaic,
)


def test_compute_mosaic_extent_wraps_all_record_bboxes():
    records = [
        _make_record(bbox=(0.0, 2.0, 10.0, 12.0)),
        _make_record(bbox=(1.5, 3.5, 11.0, 13.0)),
    ]

    extent = compute_mosaic_extent(records, pad_lon=0.5, pad_lat=1.0)

    assert extent == (-0.5, 4.0, 9.0, 14.0)


def test_compute_mosaic_limits_uses_valid_swot_squared_error_scale():
    records = [
        _make_record(
            prediction=np.array([[3.0, 5.0, 100.0, 100.0]], dtype=np.float32),
            target=np.array([[1.0, 10.0, 0.0, np.nan]], dtype=np.float32),
        )
    ]

    limits = compute_mosaic_limits(records)

    assert limits["prediction"] == (0.0, 100.0)
    assert limits["target"] == (0.0, 100.0)
    assert limits["squared_error"] == (4.0, 25.0)


def test_plot_date_mosaic_overlays_overlapping_prediction_files(tmp_path):
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    _write_npz(prediction_dir / "a.npz", bbox=(0.0, 2.0, 10.0, 12.0), value=1.0)
    _write_npz(prediction_dir / "b.npz", bbox=(1.0, 3.0, 10.5, 12.5), value=2.0)

    records = load_prediction_records(
        sorted(prediction_dir.glob("*.npz")),
        date_text="2025-05-02",
        time_index=0,
        channel_index=0,
        source_key="auto",
    )
    output_path = plot_date_mosaic(
        records,
        tmp_path / "mosaic.png",
        date_text="2025-05-02",
        alpha=0.5,
        pad_lon=0.0,
        pad_lat=0.0,
        dpi=80,
    )

    assert len(records) == 2
    assert output_path.exists()


def _make_record(
    *,
    bbox: tuple[float, float, float, float] = (0.0, 1.0, 2.0, 3.0),
    prediction: np.ndarray | None = None,
    target: np.ndarray | None = None,
):
    from pathlib import Path

    from plot_prediction_regions import PredictionRecord

    if prediction is None:
        prediction = np.array([[1.0, 2.0]], dtype=np.float32)
    if target is None:
        target = np.array([[0.0, 3.0]], dtype=np.float32)
    return PredictionRecord(
        path=Path("sample.npz"),
        bbox=bbox,
        target_date="2025-05-02",
        prediction=prediction,
        target=target,
        source_name="input_l3_ssh",
        source=np.array([[0.5, 1.5]], dtype=np.float32),
    )


def _write_npz(path, *, bbox: tuple[float, float, float, float], value: float) -> None:
    field = np.full((1, 1, 6, 6), value, dtype=np.float32)
    source = np.full((1, 6, 6), value + 0.25, dtype=np.float32)
    np.savez_compressed(
        path,
        prediction=field,
        target=field - 0.5,
        input_l3_ssh=source,
        bbox=np.array(bbox, dtype=np.float32),
        time_range=np.array(["2025-05-01", "2025-05-03"]),
        target_date="2025-05-02",
    )
