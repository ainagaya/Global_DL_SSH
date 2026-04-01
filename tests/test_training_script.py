from __future__ import annotations

import torch

import simvp_ddp_training as training_script


def test_build_model_produces_expected_output_shape(base_config):
    model = training_script.build_model(base_config)
    model.eval()

    inputs = torch.randn(2, 5, 2, 128, 128)
    outputs = model(inputs)

    assert outputs.shape == (2, 5, 1, 128, 128)


def test_evaluate_runs_with_cpu_autocast_disabled(base_config):
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4], dtype=x.dtype, device=x.device)

    batch = {
        "inputs": {
            "l3_ssh": torch.ones(1, 128, 128),
            "l4_sst": torch.ones(1, 128, 128),
        },
        "targets": {
            "l3_swot": torch.ones(1, 128, 128),
        },
        "metadata": {"bboxes": [], "time_ranges": []},
    }

    loss = training_script.evaluate(DummyModel(), [batch], torch.device("cpu"), False, base_config)

    assert isinstance(loss, float)
    assert loss >= 0.0


def test_masked_mse_zero_target_batch_keeps_grad():
    prediction = torch.randn(1, 5, 1, 16, 16, requires_grad=True)
    target = torch.zeros(1, 5, 1, 16, 16)

    from src.pytorch_losses import torch_masked_mse

    loss = torch_masked_mse(prediction, target)
    loss.backward()

    assert loss.item() == 0.0
    assert prediction.grad is not None
