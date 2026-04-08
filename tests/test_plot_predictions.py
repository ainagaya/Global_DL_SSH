from __future__ import annotations

from pathlib import Path

import numpy as np

from plot_predictions import plot_single_file


def test_plot_single_file_supports_input_panels(tmp_path):
    npz_path = tmp_path / "sample.npz"
    output_dir = tmp_path / "plots"

    prediction = np.ones((3, 1, 8, 8), dtype=np.float32)
    target = np.zeros((3, 1, 8, 8), dtype=np.float32)
    input_l3_ssh = np.full((3, 8, 8), 0.5, dtype=np.float32)
    input_l4_sst = np.full((3, 8, 8), 22.0, dtype=np.float32)

    np.savez_compressed(
        npz_path,
        prediction=prediction,
        target=target,
        input_l3_ssh=input_l3_ssh,
        input_l4_sst=input_l4_sst,
        bbox=np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
        time_range=np.array(["2025-05-01", "2025-05-03"]),
        target_date="2025-05-02",
    )

    output_path = plot_single_file(npz_path, output_dir=output_dir, time_index=0, channel_index=0, dpi=80)

    assert output_path == output_dir / "sample.png"
    assert output_path.exists()
