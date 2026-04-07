from __future__ import annotations

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
