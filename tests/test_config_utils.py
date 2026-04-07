from __future__ import annotations

from datetime import date
from pathlib import Path

from src.config_utils import ensure_dir, get_split_config, load_config, resolve_path
from src.mlflow_utils import flatten_config


def test_load_config_adds_internal_paths():
    config = load_config("configs/oceantaco.yaml")

    assert "__config_path__" in config
    assert "__repo_root__" in config
    assert config["data"]["sequence_length"] == 5


def test_resolve_path_uses_repo_root(base_config):
    resolved = resolve_path("model_weights", base_config)

    assert resolved.is_absolute()
    assert resolved.name == "model_weights"


def test_ensure_dir_creates_directory(tmp_path, base_config):
    base_config["__repo_root__"] = str(tmp_path)

    created = ensure_dir("artifacts/checkpoints", base_config)

    assert created == tmp_path / "artifacts" / "checkpoints"
    assert created.exists()
    assert created.is_dir()


def test_get_split_config_returns_named_split(base_config):
    split = get_split_config(base_config, "train")

    assert split["strategy"] == "training"
    assert split["n_queries"] == 256


def test_flatten_config_omits_internal_keys(base_config):
    flat = flatten_config(base_config)

    assert "__config_path__" not in flat
    assert "__repo_root__" not in flat
    assert flat["training.batch_size"] == "4"
    assert '"key": "l3_ssh"' in flat["data.inputs"]
    assert '"key": "l4_sst"' in flat["data.inputs"]


def test_flatten_config_stringifies_yaml_dates():
    config = {
        "splits": {
            "train": {
                "start_date": date(2025, 1, 15),
                "end_date": date(2025, 3, 15),
            }
        }
    }

    flat = flatten_config(config)

    assert flat["splits.train.start_date"] == "2025-01-15"
    assert flat["splits.train.end_date"] == "2025-03-15"
