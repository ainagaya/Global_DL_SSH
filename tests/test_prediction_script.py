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


def test_build_prediction_collate_fn_preserves_raw_samples():
    collate_fn = prediction_script.build_prediction_collate_fn()

    sample = {
        "inputs": {"l4_sst": {"data": torch.full((5, 94, 94), 25.0)}},
        "targets": {"l3_swot": {"data": torch.ones(5, 94, 94)}},
        "coords": {},
        "metadata": {"bbox": (0.0, 1.0, 2.0, 3.0), "time_range": ("2025-04-01", "2025-04-05")},
    }

    collated = collate_fn([sample])

    assert "raw_samples" in collated
    assert len(collated["raw_samples"]) == 1
    assert torch.all(collated["raw_samples"][0]["inputs"]["l4_sst"]["data"] == 25.0)


def test_sample_plot_input_fields_uses_raw_sample_tensor_without_collate_padding(base_config):
    sample = {
        "inputs": {
            "l3_ssh": {"data": torch.ones(5, 94, 94)},
            "l4_sst": {"data": torch.full((5, 94, 94), 25.0)},
        },
        "targets": {"l3_swot": {"data": torch.ones(5, 94, 94)}},
        "metadata": {"bbox": (0.0, 1.0, 2.0, 3.0), "time_range": ("2025-04-01", "2025-04-05")},
    }

    fields = prediction_script.sample_plot_input_fields(sample, base_config)

    assert fields is not None
    assert fields["l4_sst"].shape == (5, 94, 94)
    assert torch.all(fields["l4_sst"] == 25.0)
