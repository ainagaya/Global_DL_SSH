from __future__ import annotations

from pathlib import Path

import torch

import simvp_predict_ssh as prediction_script


def test_nonempty_target_mask_marks_all_zero_targets_as_empty():
    targets = torch.tensor(
        [
            [[[[0.0, 0.0], [0.0, 0.0]]]],
            [[[[1.0, 0.0], [0.0, 0.0]]]],
        ]
    )

    mask = prediction_script.nonempty_target_mask(targets)

    assert mask.tolist() == [False, True]


def test_merge_prediction_runtime_config_prefers_checkpoint_model_settings(base_config):
    runtime_config = base_config
    runtime_config["model"]["hidden_temporal"] = 128
    runtime_config["data"]["grid"]["height"] = 32

    checkpoint_config = {
        "__config_path__": str(Path("/tmp/training-config.yaml")),
        "model": {"hidden_temporal": 64, "type": "gsta", "hidden_spatial": 16, "spatial_depth": 2, "temporal_depth": 2, "drop": 0.0, "drop_path": 0.0},
        "data": {
            "grid": {"height": 128, "width": 128},
            "sequence_length": 5,
            "inputs": runtime_config["data"]["inputs"],
            "targets": runtime_config["data"]["targets"],
        },
    }

    merged = prediction_script._merge_prediction_runtime_config(runtime_config, checkpoint_config)

    assert merged["model"]["hidden_temporal"] == 64
    assert merged["data"]["grid"]["height"] == 128
    assert merged["prediction"] == runtime_config["prediction"]
