from __future__ import annotations

import sys
import types

import torch

import simvp_ddp_training as training_script
from src.mlflow_utils import MLflowTracker


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

    loss, skipped = training_script.evaluate(DummyModel(), [batch], torch.device("cpu"), False, base_config)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert skipped == 0


def test_evaluate_skips_empty_input_batches(base_config):
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return x

    empty_batch = {
        "inputs": {
            "l3_ssh": None,
            "l4_sst": None,
        },
        "targets": {
            "l3_swot": torch.ones(1, 128, 128),
        },
        "metadata": {"bboxes": [], "time_ranges": []},
    }

    loss, skipped = training_script.evaluate(DummyModel(), [empty_batch], torch.device("cpu"), False, base_config)

    assert loss == 0.0
    assert skipped == 1


def test_masked_mse_zero_target_batch_keeps_grad():
    prediction = torch.randn(1, 5, 1, 16, 16, requires_grad=True)
    target = torch.zeros(1, 5, 1, 16, 16)

    from src.pytorch_losses import torch_masked_mse

    loss = torch_masked_mse(prediction, target)
    loss.backward()

    assert loss.item() == 0.0
    assert prediction.grad is not None


def test_mlflow_tracker_can_be_disabled(base_config):
    base_config["tracking"]["mlflow"]["enabled"] = False

    tracker = MLflowTracker(base_config, stage="training")
    tracker.start_run(run_name="test")
    tracker.log_config()
    tracker.log_metrics({"train_loss": 1.0}, step=0)
    tracker.end_run()


def test_mlflow_tracker_logs_config_artifact(monkeypatch, base_config):
    class FakeMLflow:
        def __init__(self):
            self.logged_artifacts = []

        def set_tracking_uri(self, uri):
            self.tracking_uri = uri

        def set_experiment(self, name):
            self.experiment_name = name

        def start_run(self, run_name=None, tags=None):
            self.run_name = run_name
            self.tags = tags
            return object()

        def log_params(self, params):
            self.params = params

        def log_artifact(self, local_path, artifact_path=None):
            self.logged_artifacts.append((local_path, artifact_path))

        def end_run(self):
            self.ended = True

    fake_module = types.SimpleNamespace(mlflow=FakeMLflow())
    monkeypatch.setitem(sys.modules, "mlflow", fake_module.mlflow)

    base_config["tracking"]["mlflow"]["enabled"] = True
    tracker = MLflowTracker(base_config, stage="training")
    tracker.start_run(run_name="test")
    tracker.log_config()

    assert tracker._mlflow.logged_artifacts
    logged_path, logged_subdir = tracker._mlflow.logged_artifacts[0]
    assert logged_path == str(base_config["__config_path__"])
    assert logged_subdir == "config"
